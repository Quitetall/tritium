//! Config-driven loading of a standard-transformer model from HuggingFace artifacts.
//!
//! - [`ModelWeights::load_hf`] (plan 0035): fp model from `config.json` + safetensors —
//!   exact-fp dense projections ([`DenseLinear::new_exact`]).
//! - [`ModelWeights::load_salt`] (plan 0036): a **SALT-quantized** model — ternary 2D weights
//!   from a SALT bundle via dequant-to-dense + the A8 int8-activation `DenseLinear` (a CPU/eval
//!   path — numerically ≈ the native multi-plane ternary GEMM within ~1e-4, not bit-identical),
//!   with 1D norms + config from the original model dir.
//!
//! All loading paths — these two AND the BitNet GGUF path
//! ([`ModelWeights::load`], P2e) — share [`build_standard_model`], the one
//! config-driven skeleton, each supplying its [`NameSchema`] dialect and
//! projection provider.
//! Scope: standard SwiGLU/GQA/RoPE models (Llama, SmolLM2, …). Arches needing QK-norm or
//! QKV-bias are rejected (plan 0037); SSM/MoE are later still. Weights are read eagerly (fine
//! for the small conformance model); streaming/mmap for 50GB+ masters is plan 0040.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use tritium_format::{SafeTensors, read_salt_bundle, read_salt_gguf, salt_rows_to_dense};

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
        let (config, mut spec) = ModelConfig::from_hf_config(&cfg_json)?;

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
        // Detect Qwen-family features from the actual weights (config flags don't always
        // advertise them) and enable them: Qwen2/2.5 QKV bias, Qwen3 per-head QK-norm.
        spec.qkv_bias = loc.keys().any(|k| k.ends_with(".q_proj.bias"));
        spec.qk_norm = loc.keys().any(|k| k.ends_with(".q_norm.weight"));
        let get = |name: &str| -> Result<Vec<f32>, NnError> {
            let i = *loc
                .get(name)
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
            views[i]
                .tensor_f32(name)
                .map_err(|e| NnError::MissingTensor(format!("{name}: {e}")))
        };
        // Every tensor is read from the safetensors as exact fp32.
        let weights =
            build_standard_model(&config, &spec, NameSchema::Hf, &get, |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new_exact(
                    get(name)?,
                    n_out,
                    k_in,
                )?))
            })?;
        Ok((config, spec, weights))
    }

    /// Load a **SALT-quantized** standard-transformer model: the ternary 2D weights come from a
    /// SALT bundle (`.tslb` or SALT-GGUF) via dequant-to-dense (numerically ≈ the native
    /// multi-plane ternary GEMM within ~1e-4, a CPU/eval path — not bit-identical), while the 1D
    /// norms + `config.json` come from `model_dir` (bundles carry neither). A 2D weight missing
    /// from the bundle is a **hard error** (never a silent fp fallback). Runs a degraded-but-working
    /// ternary model.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] (bad config / unsupported arch), [`NnError::MissingTensor`]
    /// (bundle unreadable, or a norm absent from `model_dir`), [`NnError::Shape`].
    pub fn load_salt(
        model_dir: &Path,
        bundle: &Path,
    ) -> Result<(ModelConfig, ModelWeights), NnError> {
        let cfg_path = model_dir.join("config.json");
        let cfg_json = std::fs::read_to_string(&cfg_path)
            .map_err(|e| NnError::MissingConfig(format!("read {}: {e}", cfg_path.display())))?;
        let (config, mut spec) = ModelConfig::from_hf_config(&cfg_json)?;

        // Dequantize every bundle tensor (the 2D ternary weights + the embedding) to dense fp32.
        let bundle_bytes = std::fs::read(bundle)
            .map_err(|e| NnError::MissingTensor(format!("read {}: {e}", bundle.display())))?;
        // Sniff by magic, not filename: GGUF starts with `b"GGUF"`, a SALT bundle with `b"TSLB"`.
        let tensors = if bundle_bytes.starts_with(b"GGUF") {
            read_salt_gguf(&bundle_bytes)
        } else {
            read_salt_bundle(&bundle_bytes)
        }
        .map_err(|e| NnError::MissingTensor(format!("parse SALT bundle: {e}")))?;
        let mut dequant: HashMap<String, Vec<f32>> = HashMap::new();
        for t in &tensors {
            let d = salt_rows_to_dense(&t.salt_rows)
                .map_err(|e| NnError::MissingTensor(format!("dequant {}: {e}", t.name)))?;
            dequant.insert(t.name.clone(), d);
        }

        // The 1D norms (and any tensor the bundle lacks) come from the original safetensors.
        let shard_bytes: Vec<Vec<u8>> = resolve_shards(model_dir)?
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

        // Detect Qwen-family features from the MASTER weights (bias/QK-norm are 1D → they live in
        // the safetensors, not the bundle). SALT inference of those isn't supported yet — reject
        // loudly rather than silently building a bias-less / QK-norm-less model. (load_hf runs this
        // same detection; load_salt must too, or its guard is dead.)
        spec.qkv_bias = loc.keys().any(|k| k.ends_with(".q_proj.bias"));
        spec.qk_norm = loc.keys().any(|k| k.ends_with(".q_norm.weight"));
        if spec.qkv_bias || spec.qk_norm {
            return Err(NnError::MissingConfig(
                "SALT inference of QKV-bias/QK-norm (Qwen) models is not yet supported".to_owned(),
            ));
        }

        // 2D weights (in the bundle) → dequanted ternary; 1D norms → the original safetensors.
        let provider = |name: &str| -> Result<Vec<f32>, NnError> {
            if let Some(d) = dequant.get(name) {
                return Ok(d.clone());
            }
            // Only the 1D norms come from the fp master. A 2D weight (any non-`*norm.weight`
            // name — projections, embedding, lm_head) absent from the bundle must NOT silently
            // fall back to fp: that would load a secretly-unquantized weight. Fail loudly.
            if !name.ends_with("norm.weight") {
                return Err(NnError::MissingTensor(format!(
                    "`{name}` absent from the SALT bundle (a 2D weight must be quantized, not read fp)"
                )));
            }
            let i = *loc
                .get(name)
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
            views[i]
                .tensor_f32(name)
                .map_err(|e| NnError::MissingTensor(format!("{name}: {e}")))
        };
        // `exact_fp = false` ⇒ the A8 int8-activation `DenseLinear`. dequant-to-dense is
        // *numerically* equivalent to the native multi-plane ternary GEMM (within the ~1e-4
        // kernel tolerance) but not bit-identical — a CPU/eval path; the GPU native path (plan
        // 0040) is the deployed one.
        let weights = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            &provider,
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new(
                    provider(name)?,
                    n_out,
                    k_in,
                )?))
            },
        )?;
        Ok((config, weights))
    }
}

/// Tensor-name schema: which source names fill the canonical model slots.
/// The one config-driven builder (P2e loader unification) is schema-agnostic;
/// each loading path picks its dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameSchema {
    /// HuggingFace safetensors names (`model.layers.N.self_attn.q_proj.weight`).
    Hf,
    /// GGUF/ggml names (`blk.N.attn_q.weight`).
    Gguf,
}

impl NameSchema {
    /// Top-level (non-layer) slot name.
    fn top(self, slot: &str) -> &'static str {
        match (self, slot) {
            (NameSchema::Hf, "token_embd") => "model.embed_tokens.weight",
            (NameSchema::Hf, "output_norm") => "model.norm.weight",
            (NameSchema::Hf, "lm_head") => "lm_head.weight",
            (NameSchema::Gguf, "token_embd") => "token_embd.weight",
            (NameSchema::Gguf, "output_norm") => "output_norm.weight",
            (NameSchema::Gguf, "lm_head") => "output.weight",
            _ => unreachable!("unknown top-level slot {slot}"),
        }
    }

    /// Per-layer slot name.
    fn layer(self, i: usize, slot: &str) -> String {
        let s = match (self, slot) {
            (NameSchema::Hf, _) => {
                return format!(
                    "model.layers.{i}.{}",
                    match slot {
                        "attn_norm" => "input_layernorm.weight",
                        "q" => "self_attn.q_proj.weight",
                        "k" => "self_attn.k_proj.weight",
                        "v" => "self_attn.v_proj.weight",
                        "o" => "self_attn.o_proj.weight",
                        "q_bias" => "self_attn.q_proj.bias",
                        "k_bias" => "self_attn.k_proj.bias",
                        "v_bias" => "self_attn.v_proj.bias",
                        "q_norm" => "self_attn.q_norm.weight",
                        "k_norm" => "self_attn.k_norm.weight",
                        "attn_sub_norm" => "self_attn.attn_sub_norm.weight",
                        "ffn_norm" => "post_attention_layernorm.weight",
                        "ffn_sub_norm" => "mlp.ffn_sub_norm.weight",
                        "gate" => "mlp.gate_proj.weight",
                        "up" => "mlp.up_proj.weight",
                        "down" => "mlp.down_proj.weight",
                        _ => unreachable!("unknown layer slot {slot}"),
                    }
                );
            }
            (NameSchema::Gguf, "attn_norm") => "attn_norm.weight",
            (NameSchema::Gguf, "q") => "attn_q.weight",
            (NameSchema::Gguf, "k") => "attn_k.weight",
            (NameSchema::Gguf, "v") => "attn_v.weight",
            (NameSchema::Gguf, "o") => "attn_output.weight",
            (NameSchema::Gguf, "q_bias") => "attn_q.bias",
            (NameSchema::Gguf, "k_bias") => "attn_k.bias",
            (NameSchema::Gguf, "v_bias") => "attn_v.bias",
            (NameSchema::Gguf, "q_norm") => "attn_q_norm.weight",
            (NameSchema::Gguf, "k_norm") => "attn_k_norm.weight",
            (NameSchema::Gguf, "attn_sub_norm") => "attn_sub_norm.weight",
            (NameSchema::Gguf, "ffn_norm") => "ffn_norm.weight",
            (NameSchema::Gguf, "ffn_sub_norm") => "ffn_sub_norm.weight",
            (NameSchema::Gguf, "gate") => "ffn_gate.weight",
            (NameSchema::Gguf, "up") => "ffn_up.weight",
            (NameSchema::Gguf, "down") => "ffn_down.weight",
            _ => unreachable!("unknown layer slot {slot}"),
        };
        format!("blk.{i}.{s}")
    }
}

/// Assemble [`ModelWeights`] for a standard transformer — THE one config-driven
/// loading skeleton (P2e). `dense` provides 1D norms + the embedding (name →
/// fp32); `proj` builds each 2D projection (name, n_out, k_in) — exact-fp
/// [`DenseLinear`], A8 `DenseLinear`, or a backend-uploaded ternary linear.
/// `spec` drives every architecture axis: MLP family, QKV bias, QK-norm,
/// BitNet sub-norms, tied embeddings. Used by [`ModelWeights::load_hf`],
/// `load_salt` AND the BitNet GGUF path ([`ModelWeights::load`]).
pub(crate) fn build_standard_model(
    config: &ModelConfig,
    spec: &ArchSpec,
    schema: NameSchema,
    dense: impl Fn(&str) -> Result<Vec<f32>, NnError>,
    mut proj: impl FnMut(&str, usize, usize) -> Result<Projection, NnError>,
) -> Result<ModelWeights, NnError> {
    let n_embd = config.n_embd as usize;
    let head_dim = config.head_dim() as usize;
    let q_width = config.n_head as usize * head_dim;
    let kv_width = config.n_head_kv as usize * head_dim;
    let n_ff = config.n_ff as usize;

    let token_embd = dense(schema.top("token_embd"))?;
    let vocab = token_embd.len() / n_embd;
    let output_norm = dense(schema.top("output_norm"))?;

    let mut layers = Vec::with_capacity(config.n_layers as usize);
    for i in 0..config.n_layers as usize {
        let p = |s: &str| schema.layer(i, s);
        let (gate, up, down) = (
            proj(&p("gate"), n_ff, n_embd)?,
            proj(&p("up"), n_ff, n_embd)?,
            proj(&p("down"), n_embd, n_ff)?,
        );
        let mlp = match spec.mlp {
            MlpKind::SwiGlu => Mlp::SwiGlu(SwiGluMlp { gate, up, down }),
            MlpKind::Relu2 => Mlp::Relu2(Relu2Mlp {
                gate,
                up,
                down,
                ffn_sub_norm: if spec.ffn_sub_norm {
                    dense(&p("ffn_sub_norm"))?
                } else {
                    Vec::new()
                },
                rms_eps: config.rms_eps,
            }),
        };
        // Optional Qwen2/2.5 QKV bias and Qwen3 QK-norm (empty = absent).
        let (q_bias, k_bias, v_bias) = if spec.qkv_bias {
            (
                dense(&p("q_bias"))?,
                dense(&p("k_bias"))?,
                dense(&p("v_bias"))?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let (q_norm, k_norm) = if spec.qk_norm {
            (dense(&p("q_norm"))?, dense(&p("k_norm"))?)
        } else {
            (Vec::new(), Vec::new())
        };
        // Validate optional-weight lengths at load — a wrong-length bias would mis-stride
        // `add_bias`, a wrong-length QK-norm would mis-normalize. Fail loudly here, not later.
        for (b, w) in [(&q_bias, q_width), (&k_bias, kv_width), (&v_bias, kv_width)] {
            if !b.is_empty() && b.len() != w {
                return Err(NnError::Shape {
                    expected: w,
                    got: b.len(),
                });
            }
        }
        for n in [&q_norm, &k_norm] {
            if !n.is_empty() && n.len() != head_dim {
                return Err(NnError::Shape {
                    expected: head_dim,
                    got: n.len(),
                });
            }
        }
        layers.push(TransformerBlock {
            attn_norm: dense(&p("attn_norm"))?,
            q_proj: proj(&p("q"), q_width, n_embd)?,
            k_proj: proj(&p("k"), kv_width, n_embd)?,
            v_proj: proj(&p("v"), kv_width, n_embd)?,
            o_proj: proj(&p("o"), n_embd, q_width)?,
            // BitNet applies attn_sub_norm before o_proj; absent elsewhere.
            attn_sub_norm: if spec.attn_sub_norm {
                dense(&p("attn_sub_norm"))?
            } else {
                Vec::new()
            },
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            ffn_norm: dense(&p("ffn_norm"))?,
            mlp,
        });
    }

    // Untied lm_head when the config says so; else tie to the embedding.
    let lm_head = if spec.tied_embeddings {
        None
    } else {
        Some(proj(schema.top("lm_head"), vocab, n_embd)?)
    };

    Ok(ModelWeights {
        token_embd,
        vocab,
        n_embd,
        layers,
        output_norm,
        lm_head,
    })
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

    #[test]
    fn load_salt_dequants_bundle_and_runs() {
        use crate::{ModelRunner, Projection};
        use tritium_format::{SaltRow, salt_rows_to_dense, write_salt_bundle};
        use tritium_quantize::{QuantConfig, quantize_tensor};

        let (n_embd, n_head, n_kv, n_ff, vocab) = (8usize, 2usize, 1usize, 16usize, 10usize);
        let hd = n_embd / n_head;
        let (qw, kw) = (n_head * hd, n_kv * hd);
        let fill = |n: usize| {
            (0..n)
                .map(|i| (i as f32 * 0.017).sin() * 0.5)
                .collect::<Vec<f32>>()
        };
        // The 2D weights SALT quantizes: (name, n_out, k_in).
        let w2d: Vec<(&str, usize, usize)> = vec![
            ("model.embed_tokens.weight", vocab, n_embd),
            ("model.layers.0.self_attn.q_proj.weight", qw, n_embd),
            ("model.layers.0.self_attn.k_proj.weight", kw, n_embd),
            ("model.layers.0.self_attn.v_proj.weight", kw, n_embd),
            ("model.layers.0.self_attn.o_proj.weight", n_embd, qw),
            ("model.layers.0.mlp.gate_proj.weight", n_ff, n_embd),
            ("model.layers.0.mlp.up_proj.weight", n_ff, n_embd),
            ("model.layers.0.mlp.down_proj.weight", n_embd, n_ff),
        ];
        let norms = [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];

        // Original fp safetensors (2D weights + 1D norms).
        let mut st: Vec<(&str, Vec<usize>, Vec<f32>)> = Vec::new();
        for &(n, no, ki) in &w2d {
            st.push((n, vec![no, ki], fill(no * ki)));
        }
        for &n in &norms {
            st.push((n, vec![n_embd], fill(n_embd)));
        }
        let refs: Vec<(&str, &[usize], Vec<f32>)> = st
            .iter()
            .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
            .collect();

        let dir = std::env::temp_dir().join(format!("tritium-salt-test-{}", std::process::id()));
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

        // SALT-quantize each 2D weight → a bundle.
        let cfg = QuantConfig {
            budget_bpw: 2.0,
            ..Default::default()
        };
        let salt: Vec<(String, Vec<SaltRow>)> = w2d
            .iter()
            .map(|&(n, no, ki)| {
                let w = &st.iter().find(|(t, _, _)| *t == n).unwrap().2;
                (
                    n.to_owned(),
                    quantize_tensor(w, no, ki, &cfg).unwrap().salt_rows,
                )
            })
            .collect();
        let bundle_refs: Vec<(&str, &[SaltRow])> = salt
            .iter()
            .map(|(n, r)| (n.as_str(), r.as_slice()))
            .collect();
        let bundle_path = dir.join("model.tslb");
        std::fs::write(&bundle_path, write_salt_bundle(&bundle_refs).unwrap()).unwrap();

        // Load via from_salt and run.
        let backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut runner = ModelRunner::from_salt(&dir, &bundle_path, backend).expect("from_salt");

        // Wiring: the loaded q_proj weights equal the bundle's dequant (bundle → read → dequant).
        let q = "model.layers.0.self_attn.q_proj.weight";
        let q_expect = salt_rows_to_dense(&salt.iter().find(|(n, _)| n == q).unwrap().1).unwrap();
        match &runner.weights.layers[0].q_proj {
            Projection::Dense(d) => assert_eq!(d.weights, q_expect, "q_proj = bundle dequant"),
            Projection::Ternary(_) => panic!("expected a Dense (dequant-to-dense) projection"),
        }
        assert!(runner.weights.lm_head.is_none(), "tied ⇒ no lm_head");

        // Runs: a forward yields finite, vocab-length logits.
        let logits = runner.forward(&[0u32], &[0]).expect("forward");
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );

        // Negative: a bundle MISSING a 2D weight must error — never silently load the fp master.
        let partial: Vec<(&str, &[SaltRow])> = salt
            .iter()
            .filter(|(n, _)| !n.ends_with("q_proj.weight"))
            .map(|(n, r)| (n.as_str(), r.as_slice()))
            .collect();
        let partial_path = dir.join("partial.tslb");
        std::fs::write(&partial_path, write_salt_bundle(&partial).unwrap()).unwrap();
        let err = ModelWeights::load_salt(&dir, &partial_path)
            .err()
            .expect("load_salt must error on a missing 2D weight");
        assert!(
            matches!(err, crate::NnError::MissingTensor(_)),
            "missing 2D weight must error, got {err:?}"
        );

        // Negative: a master carrying a QKV bias (Qwen) → SALT inference rejected, not silently
        // built bias-less. (load_salt detects bias/QK-norm from the master weights, like load_hf.)
        let qdir = std::env::temp_dir().join(format!("tritium-salt-qwen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&qdir);
        std::fs::create_dir_all(&qdir).unwrap();
        let qshape = [qw];
        let bias: Vec<(&str, &[usize], Vec<f32>)> =
            vec![("model.layers.0.self_attn.q_proj.bias", &qshape, fill(qw))];
        std::fs::write(qdir.join("model.safetensors"), safetensors(&bias)).unwrap();
        std::fs::copy(dir.join("config.json"), qdir.join("config.json")).unwrap();
        std::fs::write(
            qdir.join("model.tslb"),
            write_salt_bundle(&bundle_refs).unwrap(),
        )
        .unwrap();
        let err = ModelWeights::load_salt(&qdir, &qdir.join("model.tslb"))
            .err()
            .expect("load_salt must reject a QKV-bias (Qwen) master");
        assert!(
            matches!(err, crate::NnError::MissingConfig(_)),
            "Qwen SALT must be rejected, got {err:?}"
        );
        let _ = std::fs::remove_dir_all(&qdir);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
