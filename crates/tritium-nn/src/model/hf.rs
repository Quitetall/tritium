//! Config-driven loading of a standard-transformer model from HuggingFace artifacts.
//!
//! - [`ModelWeights::load_hf`] (plan 0035): fp model from `config.json` + safetensors —
//!   exact-fp dense projections ([`DenseLinear::new_exact`]).
//! - [`ModelWeights::load_salt`] (plan 0036): a **SALT-quantized** model — ternary 2D weights
//!   retained as packed additive planes and contracted through the A8 activation path, with 1D
//!   norms + config from the original model dir. The embedding/tied head remains dense for now.
//!
//! All loading paths — these two AND the BitNet GGUF path
//! ([`ModelWeights::load`], P2e) — share [`build_standard_model`], the one
//! config-driven skeleton, each supplying its [`NameSchema`] dialect and
//! projection provider.
//! Scope: standard SwiGLU/GQA/RoPE models (Llama, SmolLM2, …). Arches needing QK-norm or
//! QKV-bias are rejected (plan 0037); SSM/MoE are later still. Weights are read eagerly (fine
//! for the small conformance model); streaming/mmap for 50GB+ masters is plan 0040.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use tritium_format::{
    SafeTensors, SaltBundleIndex, SaltTensor, read_salt_gguf, salt_rows_to_dense,
};

use crate::config::{ArchSpec, MlpKind, ModelConfig};
use crate::error::NnError;
use crate::layers::{
    DenseLinear, Mlp, Projection, Relu2Mlp, SaltLinear, SwiGluMlp, TransformerBlock,
};
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
        let weights = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            |n| get(n),
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new_exact(
                    get(name)?,
                    n_out,
                    k_in,
                )?))
            },
        )?;
        Ok((config, spec, weights))
    }

    /// Load a **SALT-quantized** standard-transformer model: the ternary 2D weights come from a
    /// SALT bundle (`.tslb` or SALT-GGUF) as packed additive projections. Projection weights never
    /// materialize as retained fp32 matrices; only the token embedding remains dense because gather
    /// and tied-head execution still use the legacy table. 1D norms + `config.json` come from
    /// `model_dir` (bundles carry neither). A 2D weight missing from the bundle is a **hard error**.
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
        let config_value: serde_json::Value = serde_json::from_str(&cfg_json)
            .map_err(|e| NnError::MissingConfig(format!("invalid config.json: {e}")))?;
        let declared_vocab = match config_value.get("vocab_size") {
            None => None,
            Some(value) => {
                let raw = value
                    .as_u64()
                    .ok_or_else(|| NnError::MissingConfig("vocab_size".to_owned()))?;
                Some(
                    usize::try_from(raw)
                        .map_err(|_| NnError::MissingConfig("vocab_size overflows usize".into()))?,
                )
            }
        };

        // TSLB uses a validated borrowed index so each requested tensor decodes directly into its
        // final packed projection. SALT-GGUF already owns a tensor index, but its current public
        // reader returns owned tensors; retain those packed rows without fp32 projection clones.
        let bundle_bytes = std::fs::read(bundle)
            .map_err(|e| NnError::MissingTensor(format!("read {}: {e}", bundle.display())))?;
        let source = if bundle_bytes.starts_with(b"GGUF") {
            let tensors = read_salt_gguf(&bundle_bytes)
                .map_err(|e| NnError::MissingTensor(format!("parse SALT bundle: {e}")))?;
            let mut by_name = HashMap::with_capacity(tensors.len());
            for tensor in tensors {
                let name = tensor.name.clone();
                if by_name.insert(name.clone(), tensor).is_some() {
                    return Err(NnError::MissingTensor(format!(
                        "duplicate SALT tensor `{name}`"
                    )));
                }
            }
            SaltTensorSource::Gguf(RefCell::new(by_name))
        } else {
            SaltTensorSource::Bundle(
                SaltBundleIndex::new(&bundle_bytes)
                    .map_err(|e| NnError::MissingTensor(format!("parse SALT bundle: {e}")))?,
            )
        };

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

        // Embedding (in bundle) remains dense for gather/tied-head; 1D norms come from master.
        let provider = |name: &str| -> Result<Vec<f32>, NnError> {
            if !name.ends_with("norm.weight") {
                let tensor = source.tensor(name)?;
                if name == NameSchema::Hf.top("token_embd") {
                    let n_embd = config.n_embd as usize;
                    if tensor.k != n_embd {
                        return Err(NnError::Shape {
                            expected: n_embd,
                            got: tensor.k,
                        });
                    }
                    if let Some(vocab) = declared_vocab
                        && tensor.rows != vocab
                    {
                        return Err(NnError::Shape {
                            expected: vocab,
                            got: tensor.rows,
                        });
                    }
                    if tensor.salt_rows.len() != tensor.rows {
                        return Err(NnError::Shape {
                            expected: tensor.rows,
                            got: tensor.salt_rows.len(),
                        });
                    }
                }
                return salt_rows_to_dense(&tensor.salt_rows)
                    .map_err(|e| NnError::MissingTensor(format!("dequant {name}: {e}")));
            }
            let i = *loc
                .get(name)
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
            views[i]
                .tensor_f32(name)
                .map_err(|e| NnError::MissingTensor(format!("{name}: {e}")))
        };
        let weights = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            |n| provider(n),
            |name, n_out, k_in| {
                let tensor = source.tensor(name)?;
                Ok(Projection::Salt(SaltLinear::new(
                    tensor.salt_rows,
                    n_out,
                    k_in,
                )?))
            },
        )?;
        Ok((config, weights))
    }
}

enum SaltTensorSource<'a> {
    Bundle(SaltBundleIndex<'a>),
    Gguf(RefCell<HashMap<String, SaltTensor>>),
}

impl SaltTensorSource<'_> {
    fn tensor(&self, name: &str) -> Result<SaltTensor, NnError> {
        match self {
            Self::Bundle(index) => index
                .tensor(name)
                .ok_or_else(|| missing_salt_tensor(name))?
                .decode()
                .map_err(|error| {
                    NnError::MissingTensor(format!("decode SALT tensor `{name}`: {error}"))
                }),
            Self::Gguf(tensors) => tensors
                .borrow_mut()
                .remove(name)
                .ok_or_else(|| missing_salt_tensor(name)),
        }
    }
}

fn missing_salt_tensor(name: &str) -> NnError {
    NnError::MissingTensor(format!(
        "`{name}` absent from the SALT bundle (a 2D weight must be quantized, not read fp)"
    ))
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
    if n_embd == 0 || token_embd.is_empty() || token_embd.len() % n_embd != 0 {
        return Err(NnError::Shape {
            expected: n_embd,
            got: token_embd.len(),
        });
    }
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
    use std::collections::HashMap;

    use super::{NameSchema, build_standard_model};
    use crate::model::ModelWeights;
    use crate::{DenseLinear, MlpKind, ModelConfig, NnError};

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
                "rope_theta":10000.0,"vocab_size":10}"#,
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
    fn load_salt_retains_packed_projections_and_runs() {
        use crate::{Mlp, ModelRunner, Projection};
        use tritium_format::{
            DEFAULT_SPARSE_RESIDUAL_DENSITY, SaltRow, salt_rows_to_dense,
            write_progressive_salt_bundle, write_salt_bundle, write_salt_gguf,
        };
        use tritium_quantize::{QuantConfig, quantize_tensor};

        let (n_embd, n_head, n_kv, n_ff, vocab) = (8usize, 2usize, 1usize, 16usize, 10usize);
        let hd = n_embd / n_head;
        let (qw, kw) = (n_head * hd, n_kv * hd);
        let fill_seeded = |n: usize, seed: usize| {
            (0..n)
                .map(|i| ((i + seed * 37) as f32 * 0.017).sin() * 0.5)
                .collect::<Vec<f32>>()
        };
        let fill = |n: usize| fill_seeded(n, 0);
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
        for (seed, &(n, no, ki)) in w2d.iter().enumerate() {
            st.push((n, vec![no, ki], fill_seeded(no * ki, seed + 1)));
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
                "rope_theta":10000.0,"vocab_size":10}"#,
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
        std::fs::write(
            &bundle_path,
            write_progressive_salt_bundle(&bundle_refs, DEFAULT_SPARSE_RESIDUAL_DENSITY).unwrap(),
        )
        .unwrap();

        // Load via from_salt and run.
        let backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut runner = ModelRunner::from_salt(&dir, &bundle_path, backend).expect("from_salt");

        // Wiring: every attention/MLP projection retains packed SALT planes.
        match &runner.weights.layers[0].q_proj {
            Projection::Salt(linear) => {
                assert_eq!((linear.n_out(), linear.k_in()), (qw, n_embd));
                assert!(linear.packed_bytes() > 0);
            }
            Projection::Dense(_) | Projection::Ternary(_) => {
                panic!("SALT loader must retain packed additive planes")
            }
        }
        for layer in &runner.weights.layers {
            for projection in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
                assert!(matches!(projection, Projection::Salt(_)));
            }
            let Mlp::SwiGlu(mlp) = &layer.mlp else {
                panic!("Llama fixture must build a SwiGLU MLP");
            };
            for projection in [&mlp.gate, &mlp.up, &mlp.down] {
                assert!(matches!(projection, Projection::Salt(_)));
            }
        }
        assert!(runner.weights.lm_head.is_none(), "tied ⇒ no lm_head");

        // Runs: a forward yields finite, vocab-length logits.
        let tokens = [0u32, 1u32];
        let positions = [0usize, 1usize];
        let logits = runner.forward(&tokens, &positions).expect("forward");
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );

        // Exact end-to-end oracle: dequantize the same named rows into the former A8 dense
        // representation. This catches equal-shaped q/k/v tensors being wired under the wrong
        // names, which representation/shape checks and v1/v2 equality alone cannot detect.
        let dequant: HashMap<&str, Vec<f32>> = salt
            .iter()
            .map(|(name, rows)| (name.as_str(), salt_rows_to_dense(rows).unwrap()))
            .collect();
        let dense_provider = |name: &str| -> Result<Vec<f32>, NnError> {
            if let Some(weight) = dequant.get(name) {
                return Ok(weight.clone());
            }
            st.iter()
                .find(|(candidate, _, _)| *candidate == name)
                .map(|(_, _, values)| values.clone())
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))
        };
        let cfg_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let (oracle_config, oracle_spec) = ModelConfig::from_hf_config(&cfg_json).unwrap();
        let oracle_weights = build_standard_model(
            &oracle_config,
            &oracle_spec,
            NameSchema::Hf,
            |name| dense_provider(name),
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new(
                    dense_provider(name)?,
                    n_out,
                    k_in,
                )?))
            },
        )
        .unwrap();
        let oracle_backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut oracle = ModelRunner::from_weights(oracle_config, oracle_weights, oracle_backend);
        assert_eq!(
            logits,
            oracle
                .forward(&tokens, &positions)
                .expect("dense SALT oracle"),
            "packed loader must preserve exact named SALT wiring"
        );

        // Keep the legacy v1 bundle on the positive runtime path too: both encodings must
        // reconstruct the same SALT rows and therefore produce exactly the same logits.
        let legacy_path = dir.join("legacy.tslb");
        std::fs::write(&legacy_path, write_salt_bundle(&bundle_refs).unwrap()).unwrap();
        let legacy_backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut legacy_runner =
            ModelRunner::from_salt(&dir, &legacy_path, legacy_backend).expect("from_salt v1");
        let legacy_logits = legacy_runner
            .forward(&tokens, &positions)
            .expect("forward v1");
        assert_eq!(legacy_logits, logits, "v1 and v2 runtime logits must match");

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

        // Negative: embedding rows may not be silently flattened across the model width.
        let bad_embedding = quantize_tensor(&fill(vocab * (n_embd + 1)), vocab, n_embd + 1, &cfg)
            .unwrap()
            .salt_rows;
        let malformed_embedding: Vec<(&str, &[SaltRow])> = salt
            .iter()
            .map(|(name, rows)| {
                if name == "model.embed_tokens.weight" {
                    (name.as_str(), bad_embedding.as_slice())
                } else {
                    (name.as_str(), rows.as_slice())
                }
            })
            .collect();
        let malformed_embedding_path = dir.join("bad-embedding.tslb");
        std::fs::write(
            &malformed_embedding_path,
            write_salt_bundle(&malformed_embedding).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            ModelWeights::load_salt(&dir, &malformed_embedding_path),
            Err(NnError::Shape { .. })
        ));

        // Negative: correct embedding width with the wrong declared vocabulary is also invalid.
        let short_embedding = quantize_tensor(&fill((vocab - 1) * n_embd), vocab - 1, n_embd, &cfg)
            .unwrap()
            .salt_rows;
        let wrong_vocab: Vec<(&str, &[SaltRow])> = salt
            .iter()
            .map(|(name, rows)| {
                if name == "model.embed_tokens.weight" {
                    (name.as_str(), short_embedding.as_slice())
                } else {
                    (name.as_str(), rows.as_slice())
                }
            })
            .collect();
        let wrong_vocab_path = dir.join("wrong-vocab.tslb");
        std::fs::write(&wrong_vocab_path, write_salt_bundle(&wrong_vocab).unwrap()).unwrap();
        assert!(matches!(
            ModelWeights::load_salt(&dir, &wrong_vocab_path),
            Err(NnError::Shape { .. })
        ));

        // Negative: a crafted zero-row embedding may not construct a vocab=0 model.
        let mut zero_embedding = write_salt_bundle(&bundle_refs).unwrap();
        let embedding_name = "model.embed_tokens.weight";
        let embedding_rows_offset = 10 + 2 + embedding_name.len();
        let embedding_data_len_offset = embedding_rows_offset + 4 + 4;
        let embedding_data_len = u64::from_le_bytes(
            zero_embedding[embedding_data_len_offset..embedding_data_len_offset + 8]
                .try_into()
                .unwrap(),
        ) as usize;
        zero_embedding[embedding_rows_offset..embedding_rows_offset + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        zero_embedding[embedding_data_len_offset..embedding_data_len_offset + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        let data_start = 10
            + bundle_refs
                .iter()
                .map(|(name, _)| 2 + name.len() + 4 + 4 + 8)
                .sum::<usize>();
        zero_embedding.drain(data_start..data_start + embedding_data_len);
        let zero_embedding_path = dir.join("zero-embedding.tslb");
        std::fs::write(&zero_embedding_path, zero_embedding).unwrap();
        assert!(matches!(
            ModelWeights::load_salt(&dir, &zero_embedding_path),
            Err(NnError::Shape { .. })
        ));

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

        // The GGUF source branch and untied-head path must also retain packed rows.
        let untied_dir =
            std::env::temp_dir().join(format!("tritium-salt-untied-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&untied_dir);
        std::fs::create_dir_all(&untied_dir).unwrap();
        std::fs::copy(
            dir.join("model.safetensors"),
            untied_dir.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            untied_dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":16,
                "rms_norm_eps":1e-5,"hidden_act":"silu","tie_word_embeddings":false,
                "rope_theta":10000.0,"vocab_size":10}"#,
        )
        .unwrap();
        let head_rows = quantize_tensor(&fill(vocab * n_embd), vocab, n_embd, &cfg)
            .unwrap()
            .salt_rows;
        let mut untied_refs = bundle_refs.clone();
        untied_refs.push(("lm_head.weight", &head_rows));
        let gguf_path = untied_dir.join("model.salt.gguf");
        std::fs::write(&gguf_path, write_salt_gguf(&untied_refs).unwrap()).unwrap();
        let mut untied_runner = ModelRunner::from_salt(
            &untied_dir,
            &gguf_path,
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .expect("untied SALT-GGUF");
        assert!(matches!(
            &untied_runner.weights.lm_head,
            Some(Projection::Salt(_))
        ));
        for layer in &untied_runner.weights.layers {
            for projection in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
                assert!(matches!(projection, Projection::Salt(_)));
            }
            let Mlp::SwiGlu(mlp) = &layer.mlp else {
                panic!("Llama fixture must build a SwiGLU MLP");
            };
            for projection in [&mlp.gate, &mlp.up, &mlp.down] {
                assert!(matches!(projection, Projection::Salt(_)));
            }
        }
        let gguf_logits = untied_runner
            .forward(&tokens, &positions)
            .expect("untied GGUF forward");

        let tslb_path = untied_dir.join("model.tslb");
        std::fs::write(&tslb_path, write_salt_bundle(&untied_refs).unwrap()).unwrap();
        let mut tslb_runner = ModelRunner::from_salt(
            &untied_dir,
            &tslb_path,
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .expect("untied TSLB");
        let tslb_logits = tslb_runner
            .forward(&tokens, &positions)
            .expect("untied TSLB forward");

        let head_dense = salt_rows_to_dense(&head_rows).unwrap();
        let untied_dense_provider = |name: &str| -> Result<Vec<f32>, NnError> {
            if name == "lm_head.weight" {
                Ok(head_dense.clone())
            } else {
                dense_provider(name)
            }
        };
        let untied_config_json = std::fs::read_to_string(untied_dir.join("config.json")).unwrap();
        let (untied_config, untied_spec) =
            ModelConfig::from_hf_config(&untied_config_json).unwrap();
        let untied_oracle_weights = build_standard_model(
            &untied_config,
            &untied_spec,
            NameSchema::Hf,
            |name| untied_dense_provider(name),
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new(
                    untied_dense_provider(name)?,
                    n_out,
                    k_in,
                )?))
            },
        )
        .unwrap();
        let mut untied_oracle = ModelRunner::from_weights(
            untied_config,
            untied_oracle_weights,
            Box::new(tritium_cpu::CpuBackend::new()),
        );
        let oracle_logits = untied_oracle
            .forward(&tokens, &positions)
            .expect("untied dense SALT oracle");
        assert_eq!(
            gguf_logits, oracle_logits,
            "untied GGUF must preserve exact named SALT wiring"
        );
        assert_eq!(
            tslb_logits, oracle_logits,
            "untied TSLB must preserve exact named SALT wiring"
        );

        let missing_head_path = untied_dir.join("missing-head.salt.gguf");
        std::fs::write(&missing_head_path, write_salt_gguf(&bundle_refs).unwrap()).unwrap();
        assert!(matches!(
            ModelWeights::load_salt(&untied_dir, &missing_head_path),
            Err(NnError::MissingTensor(_))
        ));
        let _ = std::fs::remove_dir_all(&untied_dir);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
