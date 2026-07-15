//! Config-driven loading of a standard-transformer model from HuggingFace artifacts.
//!
//! - [`ModelWeights::load_hf`] (plan 0035): fp model from `config.json` + safetensors —
//!   exact-fp dense projections ([`DenseLinear::new_exact`]).
//! - [`ModelWeights::load_salt`] (plan 0036): a **SALT-quantized** model — ternary 2D weights
//!   retained as packed additive planes, including one shared packed embedding/tied-head table,
//!   with 1D norms + config from the original model dir.
//!
//! All loading paths — these two AND the BitNet GGUF path
//! ([`ModelWeights::load`], P2e) — share [`build_standard_model`], the one
//! config-driven skeleton, each supplying its [`NameSchema`] dialect and
//! projection provider.
//! Scope: standard SwiGLU/GQA/RoPE models (Llama, SmolLM2, Qwen2.5, Qwen3). `load_hf` supports
//! QKV bias and QK-norm; `load_salt` detects and rejects them until plan 0037. SSM/MoE are later.
//! Safetensors shards are indexed by header and seek-read per requested tensor. The fp loader still
//! retains the resulting fp32 model; the SALT loader retains only widened 1D norms from the master.

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use tritium_format::{SALT_BUNDLE_MAGIC, SaltBundleReader, SaltGgufReader};

use crate::config::{ArchSpec, MlpKind, ModelConfig};
use crate::error::NnError;
use crate::layers::{
    DenseLinear, Mlp, PackedSaltMatrix, PackedSaltMatrixBuilder, Projection, Relu2Mlp, SaltLinear,
    SwiGluMlp, TokenEmbedding, TransformerBlock,
};
use crate::model::ModelWeights;
use crate::model::hf_shards::HfShardSet;

impl ModelWeights {
    /// Load a standard-transformer fp model from a directory holding `config.json` and
    /// (possibly sharded) `*.safetensors`. Returns the parsed [`ModelConfig`] + [`ArchSpec`]
    /// alongside the weights.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] (bad/absent `config.json`), [`NnError::MissingTensor`] (a
    /// schema tensor absent, malformed, or unreadable), or [`NnError::Shape`] (a projection shape
    /// mismatch).
    pub fn load_hf(dir: &Path) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError> {
        let cfg_path = dir.join("config.json");
        let cfg_json = read_config_json(&cfg_path)?;
        let (config, mut spec) = ModelConfig::from_hf_config(&cfg_json)?;
        let config_value: serde_json::Value = serde_json::from_str(&cfg_json)
            .map_err(|error| NnError::MissingConfig(format!("invalid config.json: {error}")))?;
        let declared_vocab = declared_vocab_size(&config_value)?;

        let shards = HfShardSet::open(dir)?;
        // Detect Qwen-family features from the actual weights (config flags don't always
        // advertise them) and enable them: Qwen2/2.5 QKV bias, Qwen3 per-head QK-norm.
        (spec.qkv_bias, spec.qk_norm) =
            resolve_optional_attention_weights(&config, &config_value, &shards)?;
        let get = |name: &str, request: DenseTensorRequest| -> Result<Vec<f32>, NnError> {
            match request {
                DenseTensorRequest::Vector { len } => shards.tensor_f32_exact(name, &[len]),
                DenseTensorRequest::TokenEmbedding { columns } => {
                    shards.tensor_f32_matrix(name, declared_vocab, columns)
                }
            }
        };
        // Every requested tensor is seek-read from its shard and widened exactly to fp32.
        let weights = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            |name, request| get(name, request),
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new_exact(
                    shards.tensor_f32_exact(name, &[n_out, k_in])?,
                    n_out,
                    k_in,
                )?))
            },
        )?;
        Ok((config, spec, weights))
    }

    /// Load a **SALT-quantized** standard-transformer model: the ternary 2D weights come from a
    /// SALT bundle (`.tslb` or SALT-GGUF) as packed additive projections and a shared packed token
    /// table. No 2D weight materializes as a retained fp32 matrix. 1D norms + `config.json` come
    /// from `model_dir` (bundles carry neither). A 2D weight missing from the bundle is a hard error.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] (bad config / unsupported arch), [`NnError::MissingTensor`]
    /// (bundle unreadable, or a norm absent from `model_dir`), [`NnError::Shape`], or
    /// [`NnError::Backend`] (for example, a packed plane with a non-finite scale).
    pub fn load_salt(
        model_dir: &Path,
        bundle: &Path,
    ) -> Result<(ModelConfig, ModelWeights), NnError> {
        let cfg_path = model_dir.join("config.json");
        let cfg_json = read_config_json(&cfg_path)?;
        let (config, mut spec) = ModelConfig::from_hf_config(&cfg_json)?;
        let config_value: serde_json::Value = serde_json::from_str(&cfg_json)
            .map_err(|e| NnError::MissingConfig(format!("invalid config.json: {e}")))?;
        let declared_vocab = declared_vocab_size(&config_value)?;

        // Index only the master headers. Feature detection and all norm geometry checks happen
        // before any payload is read; unsupported Qwen masters also fail before loading the SALT
        // artifact itself.
        let shards = HfShardSet::open(model_dir)?;
        (spec.qkv_bias, spec.qk_norm) =
            resolve_optional_attention_weights(&config, &config_value, &shards)?;
        if spec.qkv_bias || spec.qk_norm {
            return Err(NnError::MissingConfig(
                "SALT inference of QKV-bias/QK-norm (Qwen) models is not yet supported".to_owned(),
            ));
        }

        // Open once, pinning the source handle before format sniffing. Both canonical TSLB and
        // legacy SALT-GGUF are strict-seek scanned, then stream selected rows directly into final
        // arenas without retaining the artifact or intermediate owned rows.
        let mut artifact = File::open(bundle)
            .map_err(|e| NnError::MissingTensor(format!("open {}: {e}", bundle.display())))?;
        if !artifact
            .metadata()
            .map_err(|e| NnError::MissingTensor(format!("stat {}: {e}", bundle.display())))?
            .is_file()
        {
            return Err(NnError::MissingTensor(format!(
                "{} is not a regular file",
                bundle.display()
            )));
        }
        let mut magic = [0u8; 4];
        artifact
            .read_exact(&mut magic)
            .map_err(|e| NnError::MissingTensor(format!("read {} magic: {e}", bundle.display())))?;
        artifact
            .seek(SeekFrom::Start(0))
            .map_err(|e| NnError::MissingTensor(format!("seek {}: {e}", bundle.display())))?;
        let source = if magic == *b"GGUF" {
            SaltTensorSource::Gguf(RefCell::new(
                SaltGgufReader::new_strict(BufReader::with_capacity(64 * 1024, artifact)).map_err(
                    |e| NnError::MissingTensor(format!("index {}: {e}", bundle.display())),
                )?,
            ))
        } else if magic == SALT_BUNDLE_MAGIC {
            SaltTensorSource::Bundle(RefCell::new(
                SaltBundleReader::new_strict(BufReader::with_capacity(64 * 1024, artifact))
                    .map_err(|e| {
                        NnError::MissingTensor(format!("index {}: {e}", bundle.display()))
                    })?,
            ))
        } else {
            return Err(NnError::MissingTensor(format!(
                "{} is neither TSLB nor SALT-GGUF",
                bundle.display()
            )));
        };

        // The token table is one packed allocation shared by gather and the tied head. Validate
        // its declared config geometry before any model assembly.
        let n_embd = config.n_embd as usize;
        let embedding_matrix =
            source.matrix(NameSchema::Hf.top("token_embd"), declared_vocab, n_embd)?;
        let embedding_rows = embedding_matrix.n_out();
        let token_embd =
            TokenEmbedding::from_packed_matrix(embedding_matrix, embedding_rows, n_embd)?;

        // Only 1D norms come from the fp master.
        let provider = |name: &str, request: DenseTensorRequest| -> Result<Vec<f32>, NnError> {
            let DenseTensorRequest::Vector { len } = request else {
                return Err(NnError::Backend(format!(
                    "unexpected token-table request for `{name}`"
                )));
            };
            shards.tensor_f32_exact(name, &[len])
        };
        let weights = build_standard_model_with_embedding(
            &config,
            &spec,
            NameSchema::Hf,
            token_embd,
            |name, request| provider(name, request),
            |name, n_out, k_in| {
                Ok(Projection::Salt(SaltLinear::from_packed_matrix(
                    source.matrix(name, Some(n_out), k_in)?,
                )))
            },
        )?;
        Ok((config, weights))
    }
}

enum SaltTensorSource {
    Bundle(RefCell<SaltBundleReader<BufReader<File>>>),
    Gguf(RefCell<SaltGgufReader<BufReader<File>>>),
}

impl SaltTensorSource {
    fn matrix(
        &self,
        name: &str,
        expected_rows: Option<usize>,
        expected_k: usize,
    ) -> Result<PackedSaltMatrix, NnError> {
        match self {
            Self::Bundle(reader) => {
                let mut reader = reader.try_borrow_mut().map_err(|_| {
                    NnError::Backend(format!("reentrant SALT tensor read for `{name}`"))
                })?;
                let info = reader
                    .tensor_info(name)
                    .cloned()
                    .ok_or_else(|| missing_salt_tensor(name))?;
                validate_salt_shape(name, info.shape(), expected_rows, expected_k)?;
                let mut builder = PackedSaltMatrixBuilder::from_streamed(
                    info.shape().0,
                    info.shape().1,
                    info.storage_requirements(),
                )?;
                let mut builder_error = None;
                reader
                    .visit_packed_tensor(name, |row| {
                        if builder_error.is_none()
                            && let Err(error) = builder.push_ref(row)
                        {
                            builder_error = Some(error);
                        }
                    })
                    .map_err(|error| {
                        NnError::MissingTensor(format!("read SALT tensor `{name}`: {error}"))
                    })?;
                if let Some(error) = builder_error {
                    return Err(error);
                }
                builder.finish()
            }
            Self::Gguf(reader) => {
                let mut reader = reader.try_borrow_mut().map_err(|_| {
                    NnError::Backend(format!("reentrant SALT tensor read for `{name}`"))
                })?;
                let info = reader
                    .tensor_info(name)
                    .cloned()
                    .ok_or_else(|| missing_salt_tensor(name))?;
                validate_salt_shape(name, info.shape(), expected_rows, expected_k)?;
                let mut builder = PackedSaltMatrixBuilder::from_streamed(
                    info.shape().0,
                    info.shape().1,
                    info.storage_requirements(),
                )?;
                let mut builder_error = None;
                reader
                    .visit_packed_tensor(name, |row| {
                        if builder_error.is_none()
                            && let Err(error) = builder.push_ref(row)
                        {
                            builder_error = Some(error);
                        }
                    })
                    .map_err(|error| {
                        NnError::MissingTensor(format!("read SALT tensor `{name}`: {error}"))
                    })?;
                if let Some(error) = builder_error {
                    return Err(error);
                }
                builder.finish()
            }
        }
    }
}

fn validate_salt_shape(
    name: &str,
    actual: (usize, usize),
    expected_rows: Option<usize>,
    expected_k: usize,
) -> Result<(), NnError> {
    if actual.1 != expected_k {
        return Err(NnError::Shape {
            expected: expected_k,
            got: actual.1,
        });
    }
    if let Some(rows) = expected_rows
        && actual.0 != rows
    {
        return Err(NnError::Shape {
            expected: rows,
            got: actual.0,
        });
    }
    if actual.0 == 0 {
        return Err(NnError::MissingTensor(format!(
            "SALT tensor `{name}` has zero rows"
        )));
    }
    Ok(())
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
    pub(super) fn top(self, slot: &str) -> &'static str {
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
    pub(super) fn layer(self, i: usize, slot: &str) -> String {
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

/// Geometry requested from a model builder's dense-tensor provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseTensorRequest {
    /// Dense token table with a configuration-derived hidden width.
    TokenEmbedding { columns: usize },
    /// One-dimensional tensor with an exact configuration-derived length.
    Vector { len: usize },
}

/// Assemble [`ModelWeights`] for a standard transformer — THE one config-driven
/// loading skeleton (P2e). `dense` provides 1D norms + a dense embedding (name →
/// fp32); `proj` builds each other 2D projection (name, n_out, k_in) — exact-fp
/// [`DenseLinear`], A8 `DenseLinear`, or a backend-uploaded ternary linear.
/// `spec` drives every architecture axis: MLP family, QKV bias, QK-norm,
/// BitNet sub-norms, tied embeddings. Used by [`ModelWeights::load_hf`],
/// `load_salt` AND the BitNet GGUF path ([`ModelWeights::load`]).
pub(crate) fn build_standard_model(
    config: &ModelConfig,
    spec: &ArchSpec,
    schema: NameSchema,
    dense: impl Fn(&str, DenseTensorRequest) -> Result<Vec<f32>, NnError>,
    proj: impl FnMut(&str, usize, usize) -> Result<Projection, NnError>,
) -> Result<ModelWeights, NnError> {
    let n_embd = config.n_embd as usize;
    let token_values = dense(
        schema.top("token_embd"),
        DenseTensorRequest::TokenEmbedding { columns: n_embd },
    )?;
    if n_embd == 0 || token_values.is_empty() || token_values.len() % n_embd != 0 {
        return Err(NnError::Shape {
            expected: n_embd,
            got: token_values.len(),
        });
    }
    let vocab = token_values.len() / n_embd;
    let token_embd = TokenEmbedding::from_dense(token_values, vocab, n_embd)?;
    build_standard_model_with_embedding(config, spec, schema, token_embd, dense, proj)
}

/// Assemble a standard model around a pre-built dense or packed token table.
///
/// This is the SALT memory-floor seam: 1D tensors still come from `dense`, while
/// the embedding/tied-head table can remain packed and is never materialized as fp32.
pub(crate) fn build_standard_model_with_embedding(
    config: &ModelConfig,
    spec: &ArchSpec,
    schema: NameSchema,
    token_embd: TokenEmbedding,
    dense: impl Fn(&str, DenseTensorRequest) -> Result<Vec<f32>, NnError>,
    mut proj: impl FnMut(&str, usize, usize) -> Result<Projection, NnError>,
) -> Result<ModelWeights, NnError> {
    let n_embd = config.n_embd as usize;
    if n_embd == 0 || token_embd.cols() != n_embd || token_embd.rows() == 0 {
        return Err(NnError::Shape {
            expected: n_embd.max(1),
            got: token_embd.cols(),
        });
    }
    let vocab = token_embd.rows();
    let head_dim = config.head_dim() as usize;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(NnError::Backend(
            "attention head dimension must be nonzero and even".to_owned(),
        ));
    }
    let q_width = (config.n_head as usize)
        .checked_mul(head_dim)
        .ok_or_else(|| NnError::Backend("query projection width overflows usize".to_owned()))?;
    let kv_width = (config.n_head_kv as usize)
        .checked_mul(head_dim)
        .ok_or_else(|| NnError::Backend("KV projection width overflows usize".to_owned()))?;
    let n_ff = config.n_ff as usize;
    let dense_exact = |name: &str, expected: usize| -> Result<Vec<f32>, NnError> {
        let values = dense(name, DenseTensorRequest::Vector { len: expected })?;
        if values.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: values.len(),
            });
        }
        Ok(values)
    };
    let output_norm = dense_exact(schema.top("output_norm"), n_embd)?;

    let layer_count = config.n_layers as usize;
    let mut layers = Vec::new();
    layers.try_reserve_exact(layer_count).map_err(|_| {
        NnError::Backend(format!(
            "allocate metadata for {layer_count} transformer layers"
        ))
    })?;
    for i in 0..layer_count {
        let p = |s: &str| schema.layer(i, s);
        let (gate, up, down) = (
            proj(&p("gate"), n_ff, n_embd)?,
            proj(&p("up"), n_ff, n_embd)?,
            proj(&p("down"), n_embd, n_ff)?,
        );
        let mlp = match spec.mlp {
            MlpKind::SwiGlu => Mlp::SwiGlu(SwiGluMlp::new(gate, up, down)?),
            MlpKind::Relu2 => Mlp::Relu2(Relu2Mlp {
                gate,
                up,
                down,
                ffn_sub_norm: if spec.ffn_sub_norm {
                    dense_exact(&p("ffn_sub_norm"), n_ff)?
                } else {
                    Vec::new()
                },
                rms_eps: config.rms_eps,
            }),
        };
        // Optional Qwen2/2.5 QKV bias and Qwen3 QK-norm (empty = absent).
        let (q_bias, k_bias, v_bias) = if spec.qkv_bias {
            (
                dense_exact(&p("q_bias"), q_width)?,
                dense_exact(&p("k_bias"), kv_width)?,
                dense_exact(&p("v_bias"), kv_width)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let (q_norm, k_norm) = if spec.qk_norm {
            (
                dense_exact(&p("q_norm"), head_dim)?,
                dense_exact(&p("k_norm"), head_dim)?,
            )
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
            attn_norm: dense_exact(&p("attn_norm"), n_embd)?,
            q_proj: proj(&p("q"), q_width, n_embd)?,
            k_proj: proj(&p("k"), kv_width, n_embd)?,
            v_proj: proj(&p("v"), kv_width, n_embd)?,
            o_proj: proj(&p("o"), n_embd, q_width)?,
            // BitNet applies attn_sub_norm before o_proj; absent elsewhere.
            attn_sub_norm: if spec.attn_sub_norm {
                dense_exact(&p("attn_sub_norm"), q_width)?
            } else {
                Vec::new()
            },
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            ffn_norm: dense_exact(&p("ffn_norm"), n_embd)?,
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

pub(super) fn declared_vocab_size(config: &serde_json::Value) -> Result<Option<usize>, NnError> {
    let Some(value) = config.get("vocab_size") else {
        return Ok(None);
    };
    let raw = value
        .as_u64()
        .ok_or_else(|| NnError::MissingConfig("vocab_size".to_owned()))?;
    usize::try_from(raw)
        .map(Some)
        .map_err(|_| NnError::MissingConfig("vocab_size overflows usize".to_owned()))
}

const MAX_CONFIG_JSON_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn read_config_json(path: &Path) -> Result<String, NnError> {
    let file = File::open(path)
        .map_err(|error| NnError::MissingConfig(format!("open {}: {error}", path.display())))?;
    let metadata = file
        .metadata()
        .map_err(|error| NnError::MissingConfig(format!("stat {}: {error}", path.display())))?;
    if !metadata.is_file() || metadata.len() > MAX_CONFIG_JSON_BYTES {
        return Err(NnError::MissingConfig(format!(
            "{} must be a regular config.json no larger than {MAX_CONFIG_JSON_BYTES} bytes",
            path.display()
        )));
    }
    let declared = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
    let mut text = String::new();
    text.try_reserve_exact(declared)
        .map_err(|_| NnError::MissingConfig(format!("allocate {} bytes", metadata.len())))?;
    file.take(MAX_CONFIG_JSON_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| NnError::MissingConfig(format!("read {}: {error}", path.display())))?;
    if text.len() as u64 > MAX_CONFIG_JSON_BYTES {
        return Err(NnError::MissingConfig(format!(
            "{} grew beyond {MAX_CONFIG_JSON_BYTES} bytes while reading",
            path.display()
        )));
    }
    Ok(text)
}

pub(super) fn resolve_optional_attention_weights(
    config: &ModelConfig,
    config_value: &serde_json::Value,
    shards: &HfShardSet,
) -> Result<(bool, bool), NnError> {
    let detected_qkv_bias = shards.names().any(|name| {
        [".q_proj.bias", ".k_proj.bias", ".v_proj.bias"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    });
    let detected_qk_norm = shards.names().any(|name| {
        [".q_norm.weight", ".k_norm.weight"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    });
    let arch = config.arch.to_ascii_lowercase();
    let architecture_requires_qkv_bias = arch.starts_with("qwen2");
    let architecture_requires_qk_norm = arch.starts_with("qwen3");
    let config_requires_qkv_bias = optional_config_bool(config_value, "attention_bias")?;
    let mut config_requires_qk_norm = false;
    for key in ["use_qk_norm", "qk_norm"] {
        config_requires_qk_norm |= optional_config_bool(config_value, key)?;
    }
    Ok((
        detected_qkv_bias || architecture_requires_qkv_bias || config_requires_qkv_bias,
        detected_qk_norm || architecture_requires_qk_norm || config_requires_qk_norm,
    ))
}

fn optional_config_bool(config: &serde_json::Value, key: &str) -> Result<bool, NnError> {
    match config.get(key) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| NnError::MissingConfig(key.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        DenseTensorRequest, MAX_CONFIG_JSON_BYTES, NameSchema, build_standard_model,
        optional_config_bool, read_config_json,
    };
    use crate::model::ModelWeights;
    use crate::{ArchSpec, DenseLinear, MlpKind, ModelConfig, NnError, Projection};

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
    fn config_reader_rejects_nonregular_and_oversized_files() {
        let dir = std::env::temp_dir().join(format!("tritium-config-limit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            read_config_json(&dir),
            Err(NnError::MissingConfig(_))
        ));

        let oversized = dir.join("config.json");
        std::fs::File::create(&oversized)
            .unwrap()
            .set_len(MAX_CONFIG_JSON_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_config_json(&oversized),
            Err(NnError::MissingConfig(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optional_attention_flags_reject_present_wrong_types() {
        for key in ["attention_bias", "use_qk_norm", "qk_norm"] {
            let config: serde_json::Value =
                serde_json::from_str(&format!(r#"{{"{key}":"false"}}"#)).unwrap();
            assert!(matches!(
                optional_config_bool(&config, key),
                Err(NnError::MissingConfig(_))
            ));
        }
    }

    #[test]
    fn standard_builder_rejects_nonfinite_swiglu_projection() {
        let config = ModelConfig {
            arch: "llama".to_owned(),
            n_layers: 1,
            n_embd: 2,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 3,
            n_ctx: 8,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let spec = ArchSpec {
            mlp: MlpKind::SwiGlu,
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: true,
        };
        let error = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            |_, request| {
                Ok(match request {
                    DenseTensorRequest::TokenEmbedding { columns } => vec![0.0; 2 * columns],
                    DenseTensorRequest::Vector { len } => vec![0.0; len],
                })
            },
            |name, n_out, k_in| {
                let mut weights = vec![0.0; n_out * k_in];
                if name.ends_with("mlp.gate_proj.weight") {
                    weights[0] = f32::NAN;
                }
                Ok(Projection::Dense(DenseLinear::new_exact(
                    weights, n_out, k_in,
                )?))
            },
        )
        .err()
        .expect("non-finite SwiGLU weights must fail during model binding");

        assert!(matches!(error, NnError::Backend(message) if message.contains("non-finite")));
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
        let dir = std::env::temp_dir().join(format!("tritium-hf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let shard_for = |name: &str| {
            if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                "model-00002-of-00002.safetensors"
            } else {
                "model-00001-of-00002.safetensors"
            }
        };
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
                .iter()
                .filter(|(name, _, _)| shard_for(name) == shard)
                .map(|(name, shape, values)| (*name, shape.as_slice(), values.clone()))
                .collect();
            std::fs::write(dir.join(shard), safetensors(&refs)).unwrap();
        }
        let weight_map = tensors
            .iter()
            .map(|(name, _, _)| format!(r#""{name}":"{}""#, shard_for(name)))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
        )
        .unwrap();
        let config_json = r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":16,
                "rms_norm_eps":1e-5,"hidden_act":"silu","tie_word_embeddings":true,
                "rope_theta":10000.0,"vocab_size":10}"#;
        std::fs::write(dir.join("config.json"), config_json).unwrap();

        let (config, spec, weights) = ModelWeights::load_hf(&dir).expect("load_hf");
        // Architecture-required Qwen families cannot disappear completely. Header detection
        // catches partial families; the config/architecture fallback catches total omission.
        for (arch, expected_missing) in [("qwen2", "q_proj.bias"), ("qwen3", "q_norm.weight")] {
            std::fs::write(
                dir.join("config.json"),
                config_json.replace(
                    r#""model_type":"llama""#,
                    &format!(r#""model_type":"{arch}""#),
                ),
            )
            .unwrap();
            let error = ModelWeights::load_hf(&dir)
                .err()
                .expect("Qwen optional family omission must fail");
            assert!(
                error.to_string().contains(expected_missing),
                "{arch} omitted family error: {error}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(config.n_layers, 1);
        assert_eq!(config.n_embd, 8);
        assert_eq!(config.gqa_group(), 2);
        assert_eq!(config.head_dim(), 4);
        assert_eq!(spec.mlp, MlpKind::SwiGlu);
        assert!(spec.tied_embeddings);

        assert_eq!(weights.vocab, vocab);
        assert_eq!(
            weights.token_embd.as_dense().map(<[f32]>::len),
            Some(vocab * n_embd)
        );
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
    fn load_salt_retains_packed_weights_and_runs() {
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
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => {
                panic!("legacy SALT loader must retain host-packed additive planes")
            }
        }
        assert!(runner.weights.token_embd.is_packed_salt());
        assert!(runner.weights.token_embd.as_dense().is_none());
        assert!(runner.weights.token_embd.resident_bytes() > 0);
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
            |name, _expected_len| dense_provider(name),
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
        assert!(legacy_runner.weights.token_embd.is_packed_salt());
        let legacy_logits = legacy_runner
            .forward(&tokens, &positions)
            .expect("forward v1");
        assert_eq!(legacy_logits, logits, "v1 and v2 runtime logits must match");

        // Master norm metadata is checked before seek-reading its payload. Rank matters, not
        // merely numel: `[1, hidden]` must not masquerade as `[hidden]`.
        for (label, bad_shape, bad_values) in [
            ("rank", vec![1, n_embd], fill(n_embd)),
            ("length", vec![n_embd + 1], fill(n_embd + 1)),
        ] {
            let malformed = [
                (
                    "model.norm.weight",
                    bad_shape.as_slice(),
                    bad_values.clone(),
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    &[n_embd][..],
                    fill(n_embd),
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    &[n_embd][..],
                    fill(n_embd),
                ),
            ];
            std::fs::write(dir.join("model.safetensors"), safetensors(&malformed)).unwrap();
            let error = ModelWeights::load_salt(&dir, &bundle_path)
                .err()
                .expect("malformed norm shape must fail");
            assert!(
                error.to_string().contains("has shape"),
                "{label} error: {error}"
            );
        }
        std::fs::write(dir.join("model.safetensors"), safetensors(&refs)).unwrap();

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

        // Negative: any member of an optional Qwen family must enable that family. A partial
        // checkpoint is rejected, never silently treated as bias/QK-norm-free. This check runs
        // before the SALT path is opened.
        let qdir = std::env::temp_dir().join(format!("tritium-salt-qwen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&qdir);
        std::fs::create_dir_all(&qdir).unwrap();
        let base_config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        for (label, arch, name, len) in [
            ("omitted-bias", "qwen2", "model.norm.weight", n_embd),
            ("omitted-qk-norm", "qwen3", "model.norm.weight", n_embd),
            (
                "partial-bias",
                "llama",
                "model.layers.0.self_attn.k_proj.bias",
                kw,
            ),
            (
                "partial-qk-norm",
                "llama",
                "model.layers.0.self_attn.k_norm.weight",
                hd,
            ),
        ] {
            std::fs::write(
                qdir.join("config.json"),
                base_config.replace(
                    r#""model_type":"llama""#,
                    &format!(r#""model_type":"{arch}""#),
                ),
            )
            .unwrap();
            let shape = [len];
            std::fs::write(
                qdir.join("model.safetensors"),
                safetensors(&[(name, &shape, fill(len))]),
            )
            .unwrap();
            let error = ModelWeights::load_salt(&qdir, &qdir.join("missing.tslb"))
                .err()
                .expect("load_salt must reject a partial Qwen master");
            assert!(
                matches!(error, crate::NnError::MissingConfig(_)),
                "Qwen SALT case {label} must be rejected, got {error:?}"
            );
        }
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
        assert!(untied_runner.weights.token_embd.is_packed_salt());
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
            |name, _expected_len| untied_dense_provider(name),
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
