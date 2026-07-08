//! Config-driven loading of a standard-transformer fp model from HuggingFace
//! safetensors (plan 0035 step 4). Reads `config.json` → [`ArchSpec`], resolves the
//! (possibly sharded) safetensors, and builds [`ModelWeights`] with **exact-fp** dense
//! projections ([`DenseLinear::new_exact`]) on the standard llama/qwen tensor-name schema.
//!
//! Scope: standard SwiGLU/GQA/RoPE models (Llama, SmolLM2, …). Arches needing QK-norm or
//! QKV-bias are rejected here (plan 0037); SSM/MoE are later still. Weights are read eagerly
//! (fine for the small conformance model); streaming/mmap for 50GB+ masters is plan 0040.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use tritium_format::SafeTensors;

use crate::config::{ArchSpec, MlpKind, ModelConfig};
use crate::error::NnError;
use crate::layers::{DenseLinear, Mlp, Projection, Relu2Mlp, SwiGluMlp, TransformerBlock};
use crate::model::ModelWeights;

impl ModelWeights {
    /// Load a standard-transformer fp model from a directory holding `config.json` and
    /// (possibly sharded) `*.safetensors`. Returns the parsed [`ModelConfig`] + [`ArchSpec`]
    /// alongside the weights.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] (bad/absent `config.json`, or an arch needing QK-norm/
    /// QKV-bias — not yet supported), [`NnError::MissingTensor`] (a schema tensor absent or
    /// unreadable), or [`NnError::Shape`] (a projection shape mismatch).
    pub fn load_hf(dir: &Path) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError> {
        let cfg_path = dir.join("config.json");
        let cfg_json = std::fs::read_to_string(&cfg_path)
            .map_err(|e| NnError::MissingConfig(format!("read {}: {e}", cfg_path.display())))?;
        let (config, spec) = ModelConfig::from_hf_config(&cfg_json)?;

        if spec.qk_norm || spec.qkv_bias {
            return Err(NnError::MissingConfig(
                "arch needs QK-norm/QKV-bias — not yet supported (plan 0037)".to_owned(),
            ));
        }

        // Read every shard eagerly, parse each, and index tensor name → shard.
        let shards = resolve_shards(dir)?;
        let shard_bytes: Vec<Vec<u8>> = shards
            .iter()
            .map(|p| {
                std::fs::read(p)
                    .map_err(|e| NnError::MissingTensor(format!("read {}: {e}", p.display())))
            })
            .collect::<Result<_, _>>()?;
        let views: Vec<SafeTensors> = shard_bytes
            .iter()
            .map(|b| {
                SafeTensors::parse(b)
                    .map_err(|e| NnError::MissingTensor(format!("parse safetensors: {e}")))
            })
            .collect::<Result<_, _>>()?;
        let mut loc: HashMap<&str, usize> = HashMap::new();
        for (i, v) in views.iter().enumerate() {
            for n in v.names() {
                loc.insert(n, i);
            }
        }
        let get = |name: &str| -> Result<Vec<f32>, NnError> {
            let i = *loc
                .get(name)
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
            views[i]
                .tensor_f32(name)
                .map_err(|e| NnError::MissingTensor(format!("{name}: {e}")))
        };
        let dense = |name: &str, n_out: usize, k_in: usize| -> Result<Projection, NnError> {
            Ok(Projection::Dense(DenseLinear::new_exact(
                get(name)?,
                n_out,
                k_in,
            )?))
        };

        let n_embd = config.n_embd as usize;
        let head_dim = config.head_dim() as usize;
        let q_width = config.n_head as usize * head_dim;
        let kv_width = config.n_head_kv as usize * head_dim;
        let n_ff = config.n_ff as usize;

        let token_embd = get("model.embed_tokens.weight")?;
        let vocab = token_embd.len() / n_embd;
        let output_norm = get("model.norm.weight")?;

        let mut layers = Vec::with_capacity(config.n_layers as usize);
        for i in 0..config.n_layers as usize {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            let (gate, up, down) = (
                dense(&p("mlp.gate_proj.weight"), n_ff, n_embd)?,
                dense(&p("mlp.up_proj.weight"), n_ff, n_embd)?,
                dense(&p("mlp.down_proj.weight"), n_embd, n_ff)?,
            );
            let mlp = match spec.mlp {
                MlpKind::SwiGlu => Mlp::SwiGlu(SwiGluMlp { gate, up, down }),
                MlpKind::Relu2 => Mlp::Relu2(Relu2Mlp {
                    gate,
                    up,
                    down,
                    ffn_sub_norm: Vec::new(),
                    rms_eps: config.rms_eps,
                }),
            };
            layers.push(TransformerBlock {
                attn_norm: get(&p("input_layernorm.weight"))?,
                q_proj: dense(&p("self_attn.q_proj.weight"), q_width, n_embd)?,
                k_proj: dense(&p("self_attn.k_proj.weight"), kv_width, n_embd)?,
                v_proj: dense(&p("self_attn.v_proj.weight"), kv_width, n_embd)?,
                o_proj: dense(&p("self_attn.o_proj.weight"), n_embd, q_width)?,
                // Standard transformer: no BitNet sub-norm.
                attn_sub_norm: Vec::new(),
                ffn_norm: get(&p("post_attention_layernorm.weight"))?,
                mlp,
            });
        }

        // Untied lm_head when present / the config says so; else tie to the embedding.
        let lm_head = if spec.tied_embeddings {
            None
        } else {
            Some(dense("lm_head.weight", vocab, n_embd)?)
        };

        let weights = ModelWeights {
            token_embd,
            vocab,
            n_embd,
            layers,
            output_norm,
            lm_head,
        };
        Ok((config, spec, weights))
    }
}

/// Resolve a model directory to its safetensors shard files, in deterministic order:
/// `model.safetensors.index.json` (`weight_map`) → a lone `model.safetensors` → else every
/// `*.safetensors` in the directory, sorted.
fn resolve_shards(dir: &Path) -> Result<Vec<PathBuf>, NnError> {
    let idx = dir.join("model.safetensors.index.json");
    if idx.exists() {
        let txt = std::fs::read_to_string(&idx)
            .map_err(|e| NnError::MissingTensor(format!("read index: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| NnError::MissingTensor(format!("parse index: {e}")))?;
        let wm = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| NnError::MissingTensor("index.json has no weight_map".to_owned()))?;
        let mut set = BTreeSet::new();
        for v in wm.values() {
            if let Some(s) = v.as_str() {
                set.insert(dir.join(s));
            }
        }
        if set.is_empty() {
            return Err(NnError::MissingTensor(
                "index.json lists no shards".to_owned(),
            ));
        }
        return Ok(set.into_iter().collect());
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| NnError::MissingTensor(format!("read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    v.sort();
    if v.is_empty() {
        return Err(NnError::MissingTensor(format!(
            "no `.safetensors` in {}",
            dir.display()
        )));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use crate::MlpKind;
    use crate::model::ModelWeights;

    /// Build a minimal F32 safetensors blob: `[u64 header_len][JSON header][f32 data]`.
    fn safetensors(tensors: &[(&str, &[usize], Vec<f32>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut hdr = String::from("{");
        for (i, (name, shape, vals)) in tensors.iter().enumerate() {
            let start = data.len();
            for &v in vals.iter() {
                data.extend_from_slice(&v.to_le_bytes());
            }
            if i > 0 {
                hdr.push(',');
            }
            let shape_s = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            hdr.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_s}],\"data_offsets\":[{start},{}]}}",
                data.len()
            ));
        }
        hdr.push('}');
        let hb = hdr.into_bytes();
        let mut buf = (hb.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(&hb);
        buf.extend_from_slice(&data);
        buf
    }

    #[test]
    fn load_hf_builds_a_standard_swiglu_model() {
        // Tiny SmolLM2-shaped model: n_embd=8, n_head=2, n_head_kv=1, n_ff=16, 1 layer,
        // vocab=10, tied. head_dim=4 ⇒ q_width=8, kv_width=4.
        let (n_embd, n_head, n_kv, n_ff, vocab) = (8usize, 2usize, 1usize, 16usize, 10usize);
        let hd = n_embd / n_head;
        let (qw, kw) = (n_head * hd, n_kv * hd);
        let fill = |n: usize| (0..n).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>();
        let t = |name: &'static str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            (name, shape.to_vec(), fill(n))
        };
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            t("model.embed_tokens.weight", &[vocab, n_embd]),
            t("model.norm.weight", &[n_embd]),
            t("model.layers.0.input_layernorm.weight", &[n_embd]),
            t("model.layers.0.post_attention_layernorm.weight", &[n_embd]),
            t("model.layers.0.self_attn.q_proj.weight", &[qw, n_embd]),
            t("model.layers.0.self_attn.k_proj.weight", &[kw, n_embd]),
            t("model.layers.0.self_attn.v_proj.weight", &[kw, n_embd]),
            t("model.layers.0.self_attn.o_proj.weight", &[n_embd, qw]),
            t("model.layers.0.mlp.gate_proj.weight", &[n_ff, n_embd]),
            t("model.layers.0.mlp.up_proj.weight", &[n_ff, n_embd]),
            t("model.layers.0.mlp.down_proj.weight", &[n_embd, n_ff]),
        ];
        let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();

        let dir = std::env::temp_dir().join(format!("tritium-hf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), safetensors(&refs)).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":16,
                "rms_norm_eps":1e-5,"hidden_act":"silu","tie_word_embeddings":true,
                "rope_theta":10000.0}"#,
        )
        .unwrap();

        let (config, spec, weights) = ModelWeights::load_hf(&dir).expect("load_hf");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(config.n_layers, 1);
        assert_eq!(config.n_embd, 8);
        assert_eq!(config.gqa_group(), 2);
        assert_eq!(config.head_dim(), 4);
        assert_eq!(spec.mlp, MlpKind::SwiGlu);
        assert!(spec.tied_embeddings);

        assert_eq!(weights.vocab, vocab);
        assert_eq!(weights.token_embd.len(), vocab * n_embd);
        assert_eq!(weights.layers.len(), 1);
        assert!(weights.lm_head.is_none(), "tied ⇒ no lm_head");
        let l0 = &weights.layers[0];
        assert!(l0.mlp.as_relu2().is_none(), "should be SwiGLU, not Relu2");
        assert_eq!(l0.q_proj.n_out(), qw);
        assert_eq!(l0.q_proj.k_in(), n_embd);
        assert_eq!(l0.k_proj.n_out(), kw);
        assert!(l0.attn_sub_norm.is_empty(), "standard ⇒ no attn_sub_norm");
    }
}
