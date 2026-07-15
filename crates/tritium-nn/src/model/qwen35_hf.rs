//! Dense-reference loading for Qwen3.5-family Hugging Face checkpoints.
//!
//! This adapter binds the exact hybrid language schema to the already-open
//! safetensors shards and widens supported floating-point source tensors into
//! the existing exact-fp32 runner.  It is deliberately a family/reference
//! loader: its receipt is not a pinned campaign identity, it does not load MTP,
//! and it explicitly defers vision tensors.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{
    DenseLinear, Projection, Qwen35DeltaNetWeights, Qwen35FullAttentionWeights, SwiGluMlp,
    TokenEmbedding,
};
use crate::model::hf::read_config_json;
use crate::model::hf_shards::HfShardSet;
use crate::model::{
    Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextRunner, Qwen35TextWeights,
};
use crate::qwen35_config::{Qwen35CheckpointConfig, Qwen35LayerType, Qwen35TextConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorRole {
    Matrix,
    Preserved,
}

#[derive(Debug)]
struct TensorSpec {
    shape: Vec<usize>,
    role: TensorRole,
}

/// Non-campaign evidence emitted by the generic Qwen3.5-family language loader.
///
/// Counts prove only which schema entries this load consumed or explicitly
/// deferred. They do not bind a repository revision, tensor payload identity,
/// MTP parity, or the complete Qwen3.6 campaign coverage manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35HfLanguageReceipt {
    language_tensors: usize,
    language_matrices: usize,
    language_preserved_tensors: usize,
    deferred_mtp_tensors: usize,
    deferred_vision_tensors: usize,
}

impl Qwen35HfLanguageReceipt {
    /// Exact language tensors consumed by this load.
    #[must_use]
    pub const fn language_tensors(&self) -> usize {
        self.language_tensors
    }

    /// Consumed rank-two projection and token-table tensors.
    #[must_use]
    pub const fn language_matrices(&self) -> usize {
        self.language_matrices
    }

    /// Consumed language vectors and rank-three convolution tensors.
    #[must_use]
    pub const fn language_preserved_tensors(&self) -> usize {
        self.language_preserved_tensors
    }

    /// Present `mtp.*` tensors intentionally left unloaded.
    #[must_use]
    pub const fn deferred_mtp_tensors(&self) -> usize {
        self.deferred_mtp_tensors
    }

    /// Present `model.visual.*` tensors intentionally left unloaded.
    #[must_use]
    pub const fn deferred_vision_tensors(&self) -> usize {
        self.deferred_vision_tensors
    }
}

/// Exact-fp32 Qwen3.5-family language runner loaded from HF safetensors.
///
/// This is a dense correctness reference. It is not the memory-efficient SALT
/// path and cannot be promoted into a pinned language-plus-MTP campaign result.
#[allow(missing_debug_implementations)]
pub struct Qwen35HfLanguageModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    receipt: Qwen35HfLanguageReceipt,
}

impl Qwen35HfLanguageModel {
    /// Load a validated Qwen3.5-family language core from `config.json` and all
    /// indexed safetensors in `dir`.
    ///
    /// Every expected language tensor must occur exactly once with its exact
    /// shape. Unknown language or top-level tensors fail closed; `mtp.*` and
    /// `model.visual.*` are the only deferred namespaces. F32, F16, and BF16
    /// payloads are accepted for small family fixtures and widened to fp32.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for an invalid Qwen configuration,
    /// [`NnError::MissingTensor`] for incomplete or contradictory source
    /// coverage, or a model-construction/backend error.
    pub fn load_family(dir: &Path, backend: Box<dyn TernaryBackend>) -> Result<Self, NnError> {
        let config_json = read_config_json(&dir.join("config.json"))?;
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json)?;
        let shards = HfShardSet::open(dir)?;
        let schema = language_schema(&config.text)?;
        let receipt = preflight_source(&shards, &schema)?;
        let weights = load_language_weights(&shards, &config.text)?;
        let runner = Qwen35TextRunner::new(&config.text, weights, backend)?;
        Ok(Self {
            config,
            runner,
            receipt,
        })
    }

    /// Validated family configuration used to assemble the runner.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    /// Exact-fp32 hybrid language runner.
    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    /// Non-campaign schema-consumption receipt for this load.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen35HfLanguageReceipt {
        &self.receipt
    }
}

fn preflight_source(
    shards: &HfShardSet,
    schema: &BTreeMap<String, TensorSpec>,
) -> Result<Qwen35HfLanguageReceipt, NnError> {
    let mut consumed = BTreeSet::new();
    let mut deferred_mtp_tensors = 0usize;
    let mut deferred_vision_tensors = 0usize;
    for tensor in shards.metadata() {
        if let Some(expected) = schema.get(tensor.name) {
            if !matches!(tensor.dtype, "F32" | "F16" | "BF16") {
                return Err(NnError::MissingTensor(format!(
                    "tensor `{}` has unsupported dtype {}, expected F32, F16, or BF16",
                    tensor.name, tensor.dtype
                )));
            }
            if tensor.shape != expected.shape {
                return Err(NnError::MissingTensor(format!(
                    "tensor `{}` in {} has shape {:?}, expected {:?}",
                    tensor.name,
                    tensor.shard_path.display(),
                    tensor.shape,
                    expected.shape
                )));
            }
            consumed.insert(tensor.name);
        } else if tensor.name.starts_with("mtp.") {
            deferred_mtp_tensors = deferred_mtp_tensors.checked_add(1).ok_or_else(|| {
                NnError::MissingTensor("deferred MTP tensor count overflow".to_owned())
            })?;
        } else if tensor.name.starts_with("model.visual.") {
            deferred_vision_tensors = deferred_vision_tensors.checked_add(1).ok_or_else(|| {
                NnError::MissingTensor("deferred vision tensor count overflow".to_owned())
            })?;
        } else {
            return Err(NnError::MissingTensor(format!(
                "unexpected Qwen3.5 source tensor `{}`",
                tensor.name
            )));
        }
    }

    if let Some(missing) = schema.keys().find(|name| !consumed.contains(name.as_str())) {
        return Err(NnError::MissingTensor(missing.clone()));
    }
    let language_matrices = schema
        .values()
        .filter(|tensor| tensor.role == TensorRole::Matrix)
        .count();
    Ok(Qwen35HfLanguageReceipt {
        language_tensors: schema.len(),
        language_matrices,
        language_preserved_tensors: schema.len() - language_matrices,
        deferred_mtp_tensors,
        deferred_vision_tensors,
    })
}

fn language_schema(config: &Qwen35TextConfig) -> Result<BTreeMap<String, TensorSpec>, NnError> {
    let hidden = axis(config.hidden_size);
    let intermediate = axis(config.intermediate_size);
    let vocab = axis(config.vocab_size);
    let mut schema = BTreeMap::new();
    insert_spec(
        &mut schema,
        "model.language_model.embed_tokens.weight".to_owned(),
        &[vocab, hidden],
        TensorRole::Matrix,
    )?;
    insert_spec(
        &mut schema,
        "model.language_model.norm.weight".to_owned(),
        &[hidden],
        TensorRole::Preserved,
    )?;
    insert_spec(
        &mut schema,
        "lm_head.weight".to_owned(),
        &[vocab, hidden],
        TensorRole::Matrix,
    )?;

    let key_width = checked_product(
        axis(config.delta_net.num_key_heads),
        axis(config.delta_net.key_head_dim),
        "DeltaNet key width",
    )?;
    let value_width = checked_product(
        axis(config.delta_net.num_value_heads),
        axis(config.delta_net.value_head_dim),
        "DeltaNet value width",
    )?;
    let qkv_width = key_width
        .checked_mul(2)
        .and_then(|width| width.checked_add(value_width))
        .ok_or_else(|| invalid_geometry("DeltaNet QKV width overflow"))?;
    let query_width = checked_product(
        axis(config.full_attention.num_heads),
        axis(config.full_attention.head_dim),
        "full-attention query width",
    )?;
    let kv_width = checked_product(
        axis(config.full_attention.num_key_value_heads),
        axis(config.full_attention.head_dim),
        "full-attention KV width",
    )?;
    let gated_query_width = query_width
        .checked_mul(2)
        .ok_or_else(|| invalid_geometry("gated full-attention query width overflow"))?;

    for (index, layer_type) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        for (suffix, shape, role) in [
            (
                "input_layernorm.weight",
                vec![hidden],
                TensorRole::Preserved,
            ),
            (
                "post_attention_layernorm.weight",
                vec![hidden],
                TensorRole::Preserved,
            ),
            (
                "mlp.gate_proj.weight",
                vec![intermediate, hidden],
                TensorRole::Matrix,
            ),
            (
                "mlp.up_proj.weight",
                vec![intermediate, hidden],
                TensorRole::Matrix,
            ),
            (
                "mlp.down_proj.weight",
                vec![hidden, intermediate],
                TensorRole::Matrix,
            ),
        ] {
            insert_owned_spec(&mut schema, &prefix, suffix, shape, role)?;
        }
        match layer_type {
            Qwen35LayerType::DeltaNet => {
                let value_heads = axis(config.delta_net.num_value_heads);
                let value_dim = axis(config.delta_net.value_head_dim);
                let kernel = axis(config.delta_net.conv_kernel_dim);
                for (suffix, shape, role) in [
                    (
                        "linear_attn.in_proj_qkv.weight",
                        vec![qkv_width, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "linear_attn.in_proj_z.weight",
                        vec![value_width, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "linear_attn.in_proj_b.weight",
                        vec![value_heads, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "linear_attn.in_proj_a.weight",
                        vec![value_heads, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "linear_attn.out_proj.weight",
                        vec![hidden, value_width],
                        TensorRole::Matrix,
                    ),
                    (
                        "linear_attn.conv1d.weight",
                        vec![qkv_width, 1, kernel],
                        TensorRole::Preserved,
                    ),
                    (
                        "linear_attn.norm.weight",
                        vec![value_dim],
                        TensorRole::Preserved,
                    ),
                    (
                        "linear_attn.dt_bias",
                        vec![value_heads],
                        TensorRole::Preserved,
                    ),
                    (
                        "linear_attn.A_log",
                        vec![value_heads],
                        TensorRole::Preserved,
                    ),
                ] {
                    insert_owned_spec(&mut schema, &prefix, suffix, shape, role)?;
                }
            }
            Qwen35LayerType::FullAttention => {
                let head_dim = axis(config.full_attention.head_dim);
                for (suffix, shape, role) in [
                    (
                        "self_attn.q_proj.weight",
                        vec![gated_query_width, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "self_attn.k_proj.weight",
                        vec![kv_width, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "self_attn.v_proj.weight",
                        vec![kv_width, hidden],
                        TensorRole::Matrix,
                    ),
                    (
                        "self_attn.o_proj.weight",
                        vec![hidden, query_width],
                        TensorRole::Matrix,
                    ),
                    (
                        "self_attn.q_norm.weight",
                        vec![head_dim],
                        TensorRole::Preserved,
                    ),
                    (
                        "self_attn.k_norm.weight",
                        vec![head_dim],
                        TensorRole::Preserved,
                    ),
                ] {
                    insert_owned_spec(&mut schema, &prefix, suffix, shape, role)?;
                }
            }
        }
    }
    Ok(schema)
}

fn load_language_weights(
    shards: &HfShardSet,
    config: &Qwen35TextConfig,
) -> Result<Qwen35TextWeights, NnError> {
    let hidden = axis(config.hidden_size);
    let intermediate = axis(config.intermediate_size);
    let vocab = axis(config.vocab_size);
    let embedding = TokenEmbedding::from_dense(
        shards.tensor_f32_exact("model.language_model.embed_tokens.weight", &[vocab, hidden])?,
        vocab,
        hidden,
    )?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layer_types.len())
        .map_err(|error| {
            NnError::Backend(format!(
                "allocate {} Qwen3.5 source layers: {error}",
                config.layer_types.len()
            ))
        })?;
    for (index, layer_type) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let input_norm = vector(shards, &prefix, "input_layernorm.weight", hidden)?;
        let mixer = match layer_type {
            Qwen35LayerType::DeltaNet => {
                Qwen35TextMixerWeights::DeltaNet(load_delta_net(shards, config, &prefix, hidden)?)
            }
            Qwen35LayerType::FullAttention => Qwen35TextMixerWeights::FullAttention(
                load_full_attention(shards, config, &prefix, hidden)?,
            ),
        };
        let post_attention_norm =
            vector(shards, &prefix, "post_attention_layernorm.weight", hidden)?;
        let mlp = SwiGluMlp::new(
            dense(
                shards,
                &prefix,
                "mlp.gate_proj.weight",
                intermediate,
                hidden,
            )?,
            dense(shards, &prefix, "mlp.up_proj.weight", intermediate, hidden)?,
            dense(
                shards,
                &prefix,
                "mlp.down_proj.weight",
                hidden,
                intermediate,
            )?,
        )?;
        layers.push(Qwen35TextLayerWeights::new(
            input_norm,
            mixer,
            post_attention_norm,
            mlp,
        ));
    }
    let final_norm = shards.tensor_f32_exact("model.language_model.norm.weight", &[hidden])?;
    let lm_head = Projection::Dense(DenseLinear::new_exact(
        shards.tensor_f32_exact("lm_head.weight", &[vocab, hidden])?,
        vocab,
        hidden,
    )?);
    Ok(Qwen35TextWeights::new(
        embedding, layers, final_norm, lm_head,
    ))
}

fn load_delta_net(
    shards: &HfShardSet,
    config: &Qwen35TextConfig,
    prefix: &str,
    hidden: usize,
) -> Result<Qwen35DeltaNetWeights, NnError> {
    let key_width = checked_product(
        axis(config.delta_net.num_key_heads),
        axis(config.delta_net.key_head_dim),
        "DeltaNet key width",
    )?;
    let value_heads = axis(config.delta_net.num_value_heads);
    let value_dim = axis(config.delta_net.value_head_dim);
    let value_width = checked_product(value_heads, value_dim, "DeltaNet value width")?;
    let qkv_width = key_width
        .checked_mul(2)
        .and_then(|width| width.checked_add(value_width))
        .ok_or_else(|| invalid_geometry("DeltaNet QKV width overflow"))?;
    let kernel = axis(config.delta_net.conv_kernel_dim);
    Ok(Qwen35DeltaNetWeights::new(
        dense(
            shards,
            prefix,
            "linear_attn.in_proj_qkv.weight",
            qkv_width,
            hidden,
        )?,
        dense(
            shards,
            prefix,
            "linear_attn.in_proj_z.weight",
            value_width,
            hidden,
        )?,
        dense(
            shards,
            prefix,
            "linear_attn.in_proj_b.weight",
            value_heads,
            hidden,
        )?,
        dense(
            shards,
            prefix,
            "linear_attn.in_proj_a.weight",
            value_heads,
            hidden,
        )?,
        dense(
            shards,
            prefix,
            "linear_attn.out_proj.weight",
            hidden,
            value_width,
        )?,
        shards.tensor_f32_exact(
            &tensor_name(prefix, "linear_attn.conv1d.weight"),
            &[qkv_width, 1, kernel],
        )?,
        vector(shards, prefix, "linear_attn.norm.weight", value_dim)?,
        vector(shards, prefix, "linear_attn.dt_bias", value_heads)?,
        vector(shards, prefix, "linear_attn.A_log", value_heads)?,
    ))
}

fn load_full_attention(
    shards: &HfShardSet,
    config: &Qwen35TextConfig,
    prefix: &str,
    hidden: usize,
) -> Result<Qwen35FullAttentionWeights, NnError> {
    let head_dim = axis(config.full_attention.head_dim);
    let query_width = checked_product(
        axis(config.full_attention.num_heads),
        head_dim,
        "full-attention query width",
    )?;
    let gated_query_width = query_width
        .checked_mul(2)
        .ok_or_else(|| invalid_geometry("gated full-attention query width overflow"))?;
    let kv_width = checked_product(
        axis(config.full_attention.num_key_value_heads),
        head_dim,
        "full-attention KV width",
    )?;
    Ok(Qwen35FullAttentionWeights::new(
        dense(
            shards,
            prefix,
            "self_attn.q_proj.weight",
            gated_query_width,
            hidden,
        )?,
        dense(shards, prefix, "self_attn.k_proj.weight", kv_width, hidden)?,
        dense(shards, prefix, "self_attn.v_proj.weight", kv_width, hidden)?,
        dense(
            shards,
            prefix,
            "self_attn.o_proj.weight",
            hidden,
            query_width,
        )?,
        vector(shards, prefix, "self_attn.q_norm.weight", head_dim)?,
        vector(shards, prefix, "self_attn.k_norm.weight", head_dim)?,
    ))
}

fn dense(
    shards: &HfShardSet,
    prefix: &str,
    suffix: &str,
    rows: usize,
    columns: usize,
) -> Result<Projection, NnError> {
    let name = tensor_name(prefix, suffix);
    Ok(Projection::Dense(DenseLinear::new_exact(
        shards.tensor_f32_exact(&name, &[rows, columns])?,
        rows,
        columns,
    )?))
}

fn vector(
    shards: &HfShardSet,
    prefix: &str,
    suffix: &str,
    len: usize,
) -> Result<Vec<f32>, NnError> {
    shards.tensor_f32_exact(&tensor_name(prefix, suffix), &[len])
}

fn tensor_name(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
}

fn insert_owned_spec(
    schema: &mut BTreeMap<String, TensorSpec>,
    prefix: &str,
    suffix: &str,
    shape: Vec<usize>,
    role: TensorRole,
) -> Result<(), NnError> {
    insert_spec(schema, tensor_name(prefix, suffix), &shape, role)
}

fn insert_spec(
    schema: &mut BTreeMap<String, TensorSpec>,
    name: String,
    shape: &[usize],
    role: TensorRole,
) -> Result<(), NnError> {
    if schema
        .insert(
            name.clone(),
            TensorSpec {
                shape: shape.to_vec(),
                role,
            },
        )
        .is_some()
    {
        return Err(NnError::MissingConfig(format!(
            "Qwen3.5 language schema contains duplicate tensor `{name}`"
        )));
    }
    Ok(())
}

const fn axis(value: u32) -> usize {
    value as usize
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_geometry(format!("{label} overflow")))
}

fn invalid_geometry(reason: impl Into<String>) -> NnError {
    NnError::MissingConfig(reason.into())
}
