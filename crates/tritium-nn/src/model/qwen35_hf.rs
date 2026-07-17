//! Dense-reference loading for Qwen3.5-family Hugging Face checkpoints.
//!
//! This adapter binds the exact hybrid language schema to the already-open
//! safetensors shards and widens supported floating-point source tensors into
//! the existing exact-fp32 runner. It is deliberately a family/reference
//! loader: its receipt is not a pinned campaign identity, its language-only
//! entry point defers MTP, and its combined entry point exact-loads MTP into an
//! unverified graph that still requires numerical promotion. Vision remains
//! explicitly deferred.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{
    DenseLinear, Projection, Qwen35DeltaNetWeights, Qwen35FullAttentionWeights, SwiGluMlp,
    TokenEmbedding,
};
use crate::model::hf_shards::HfShardSet;
use crate::model::qwen35_hf_source::{Qwen35HfSource, Qwen35HfSourceIdentity};
use crate::model::qwen35_mtp_oracle::load_authorized_qwen35_mtp_oracle;
use crate::model::{
    Qwen35MtpLayerWeights, Qwen35MtpParityReceipt, Qwen35MtpRunner, Qwen35MtpWeights,
    Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextRunner, Qwen35TextWeights,
    UnverifiedQwen35Mtp,
};
use crate::qwen35_config::{Qwen35CheckpointConfig, Qwen35LayerType, Qwen35TextConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorRole {
    Matrix,
    Preserved,
}

#[derive(Debug)]
pub(super) struct TensorSpec {
    shape: Vec<usize>,
    role: TensorRole,
}

pub(super) trait Qwen35HfTensorSource {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError>;
}

impl Qwen35HfTensorSource for HfShardSet {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        HfShardSet::tensor_f32_exact(self, name, expected)
    }
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

/// Exact schema-consumption receipt for a language-plus-MTP dense reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35HfLanguageMtpReceipt {
    language: Qwen35HfLanguageReceipt,
    mtp_tensors: usize,
    mtp_matrices: usize,
    mtp_preserved_tensors: usize,
}

impl Qwen35HfLanguageMtpReceipt {
    /// Language-core schema receipt.
    #[must_use]
    pub const fn language(&self) -> &Qwen35HfLanguageReceipt {
        &self.language
    }

    /// Exact MTP tensors consumed by the combined loader.
    #[must_use]
    pub const fn mtp_tensors(&self) -> usize {
        self.mtp_tensors
    }

    /// Rank-two MTP matrices consumed by the combined loader.
    #[must_use]
    pub const fn mtp_matrices(&self) -> usize {
        self.mtp_matrices
    }

    /// MTP normalization vectors consumed by the combined loader.
    #[must_use]
    pub const fn mtp_preserved_tensors(&self) -> usize {
        self.mtp_preserved_tensors
    }
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
    source_identity: Qwen35HfSourceIdentity,
}

impl Qwen35HfLanguageModel {
    /// Load a validated Qwen3.5-family language core from `config.json` and all
    /// indexed safetensors in `dir`.
    ///
    /// Every expected language tensor must occur exactly once with its exact
    /// shape. Unknown language or top-level tensors fail closed; `mtp.*` and
    /// `model.visual.*` are the only deferred namespaces. F32, F16, and BF16
    /// payloads are accepted for small family fixtures and widened to fp32.
    /// Before assembly, all source tensors are streamed into a semantic
    /// identity; every language tensor is then re-hashed from the exact chunks
    /// widened by the dense loader.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for an invalid Qwen configuration,
    /// [`NnError::MissingTensor`] for incomplete or contradictory source
    /// coverage, or a model-construction/backend error.
    pub fn load_family(dir: &Path, backend: Box<dyn TernaryBackend>) -> Result<Self, NnError> {
        Qwen35HfSource::open(dir)?
            .verify_semantic_identity()?
            .load_language(backend)
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

    /// Content-derived identity of every tensor in the once-opened source.
    #[must_use]
    pub const fn source_identity(&self) -> &Qwen35HfSourceIdentity {
        &self.source_identity
    }

    /// Assemble a language model after verified exact-byte source consumption.
    pub(super) fn from_verified_source(
        config: Qwen35CheckpointConfig,
        runner: Qwen35TextRunner,
        receipt: Qwen35HfLanguageReceipt,
        source_identity: Qwen35HfSourceIdentity,
    ) -> Self {
        Self {
            config,
            runner,
            receipt,
            source_identity,
        }
    }
}

/// Content-bound dense language core plus an exact but unverified MTP graph.
#[allow(missing_debug_implementations)]
pub struct Qwen35HfLanguageMtpModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    mtp: UnverifiedQwen35Mtp,
    receipt: Qwen35HfLanguageMtpReceipt,
    source_identity: Qwen35HfSourceIdentity,
}

/// Failed MTP promotion together with the still-loaded source-bound model.
///
/// Authentication and parity use fresh caches and do not mutate model weights.
/// Callers may inspect the error, correct the artifact/backend issue, and retry
/// without reloading a checkpoint-scale source.
pub struct Qwen35MtpPromotionError {
    model: Box<Qwen35HfLanguageMtpModel>,
    error: NnError,
}

impl Qwen35MtpPromotionError {
    /// Promotion failure that prevented executable MTP publication.
    #[must_use]
    pub const fn error(&self) -> &NnError {
        &self.error
    }

    /// Still-loaded unverified model available for inspection or retry.
    #[must_use]
    pub fn model(&self) -> &Qwen35HfLanguageMtpModel {
        &self.model
    }

    /// Recover ownership of the model and failure.
    #[must_use]
    pub fn into_parts(self) -> (Qwen35HfLanguageMtpModel, NnError) {
        (*self.model, self.error)
    }
}

impl fmt::Debug for Qwen35MtpPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Qwen35MtpPromotionError")
            .field("source_model_id", &self.model.source_identity.model_id())
            .field("error", &self.error)
            .finish()
    }
}

impl fmt::Display for Qwen35MtpPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for Qwen35MtpPromotionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl Qwen35HfLanguageMtpModel {
    /// Validated family configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    /// Exact-fp32 target language runner.
    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    /// Structurally complete MTP graph awaiting official-oracle parity.
    #[must_use]
    pub const fn mtp(&self) -> &UnverifiedQwen35Mtp {
        &self.mtp
    }

    /// Exact language-plus-MTP schema receipt.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen35HfLanguageMtpReceipt {
        &self.receipt
    }

    /// Content-derived identity of every source tensor.
    #[must_use]
    pub const fn source_identity(&self) -> &Qwen35HfSourceIdentity {
        &self.source_identity
    }

    /// Compare this exact source against a compiled-authorized vLLM artifact.
    ///
    /// The artifact body, oracle manifest, numeric policy, coverage policy, and
    /// source-derived [`tritium_format::ModelId`] must match one exact private
    /// authorization row. Callers cannot select a tolerance, inject expected
    /// values, or promote weights under an unrelated model identity. The only
    /// compiled authorization currently carries synthetic-fixture evidence; a
    /// successful return authorizes correctness testing, not production
    /// campaign admission.
    ///
    /// # Errors
    /// Returns an execution, provenance, shape, or numeric mismatch together
    /// with the still-loaded unverified model, so checkpoint-scale callers can
    /// correct the failure and retry without reloading source weights.
    pub fn verify_mtp(
        self,
        authorized_oracle_bytes: &[u8],
    ) -> Result<Qwen35VerifiedHfLanguageMtpModel, Qwen35MtpPromotionError> {
        let trace = match load_authorized_qwen35_mtp_oracle(
            authorized_oracle_bytes,
            &self.config,
            &self.source_identity,
        ) {
            Ok(trace) => trace,
            Err(error) => {
                return Err(Qwen35MtpPromotionError {
                    model: Box::new(self),
                    error,
                });
            }
        };
        let (promoted_mtp, mtp_receipt) = match self.mtp.verify_trace(&self.runner, trace) {
            Ok(result) => result,
            Err(error) => {
                return Err(Qwen35MtpPromotionError {
                    model: Box::new(self),
                    error,
                });
            }
        };
        let Self {
            config,
            runner,
            mtp: _,
            receipt,
            source_identity,
        } = self;
        Ok(Qwen35VerifiedHfLanguageMtpModel {
            config,
            runner,
            mtp: promoted_mtp,
            receipt,
            source_identity,
            mtp_receipt,
        })
    }

    pub(super) fn from_verified_source(
        config: Qwen35CheckpointConfig,
        runner: Qwen35TextRunner,
        mtp: UnverifiedQwen35Mtp,
        receipt: Qwen35HfLanguageMtpReceipt,
        source_identity: Qwen35HfSourceIdentity,
    ) -> Self {
        Self {
            config,
            runner,
            mtp,
            receipt,
            source_identity,
        }
    }
}

/// Content-bound language-plus-MTP model promoted by pinned serving-oracle parity.
///
/// The current compiled authorization is synthetic fixture evidence. This type
/// permits MTP execution for correctness work, but its receipt deliberately
/// fails production admission until a separately reviewed production policy
/// and exact checkpoint artifact are compiled.
#[allow(missing_debug_implementations)]
pub struct Qwen35VerifiedHfLanguageMtpModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    mtp: Qwen35MtpRunner,
    receipt: Qwen35HfLanguageMtpReceipt,
    source_identity: Qwen35HfSourceIdentity,
    mtp_receipt: Qwen35MtpParityReceipt,
}

impl Qwen35VerifiedHfLanguageMtpModel {
    /// Validated family configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    /// Exact target language runner sharing embedding and head with MTP.
    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    /// Receipt-gated executable MTP runner.
    #[must_use]
    pub const fn mtp(&self) -> &Qwen35MtpRunner {
        &self.mtp
    }

    /// Exact language-plus-MTP schema receipt.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen35HfLanguageMtpReceipt {
        &self.receipt
    }

    /// Content-derived identity of every source tensor.
    #[must_use]
    pub const fn source_identity(&self) -> &Qwen35HfSourceIdentity {
        &self.source_identity
    }

    /// Pinned vLLM parity evidence authorizing MTP execution.
    #[must_use]
    pub const fn mtp_receipt(&self) -> &Qwen35MtpParityReceipt {
        &self.mtp_receipt
    }
}

pub(super) fn preflight_source(
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

pub(super) fn language_schema(
    config: &Qwen35TextConfig,
) -> Result<BTreeMap<String, TensorSpec>, NnError> {
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

pub(super) fn mtp_schema(
    config: &Qwen35TextConfig,
) -> Result<BTreeMap<String, TensorSpec>, NnError> {
    let hidden = axis(config.hidden_size);
    let intermediate = axis(config.intermediate_size);
    let head_dim = axis(config.full_attention.head_dim);
    let query_width = checked_product(
        axis(config.full_attention.num_heads),
        head_dim,
        "MTP full-attention query width",
    )?;
    let gated_query_width = query_width
        .checked_mul(2)
        .ok_or_else(|| invalid_geometry("MTP gated query width overflow"))?;
    let kv_width = checked_product(
        axis(config.full_attention.num_key_value_heads),
        head_dim,
        "MTP full-attention KV width",
    )?;
    let fused_width = hidden
        .checked_mul(2)
        .ok_or_else(|| invalid_geometry("MTP fusion width overflow"))?;
    let mut schema = BTreeMap::new();
    for (name, shape, role) in [
        (
            "mtp.pre_fc_norm_embedding.weight",
            vec![hidden],
            TensorRole::Preserved,
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            vec![hidden],
            TensorRole::Preserved,
        ),
        (
            "mtp.fc.weight",
            vec![hidden, fused_width],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.input_layernorm.weight",
            vec![hidden],
            TensorRole::Preserved,
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            vec![gated_query_width, hidden],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.self_attn.k_proj.weight",
            vec![kv_width, hidden],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight",
            vec![kv_width, hidden],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            vec![hidden, query_width],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            vec![head_dim],
            TensorRole::Preserved,
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            vec![head_dim],
            TensorRole::Preserved,
        ),
        (
            "mtp.layers.0.post_attention_layernorm.weight",
            vec![hidden],
            TensorRole::Preserved,
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            vec![intermediate, hidden],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.mlp.up_proj.weight",
            vec![intermediate, hidden],
            TensorRole::Matrix,
        ),
        (
            "mtp.layers.0.mlp.down_proj.weight",
            vec![hidden, intermediate],
            TensorRole::Matrix,
        ),
        ("mtp.norm.weight", vec![hidden], TensorRole::Preserved),
    ] {
        insert_spec(&mut schema, name.to_owned(), &shape, role)?;
    }
    Ok(schema)
}

pub(super) fn preflight_mtp_source(
    shards: &HfShardSet,
    schema: &BTreeMap<String, TensorSpec>,
    mut language: Qwen35HfLanguageReceipt,
) -> Result<Qwen35HfLanguageMtpReceipt, NnError> {
    let mut consumed = BTreeSet::new();
    for tensor in shards
        .metadata()
        .filter(|tensor| tensor.name.starts_with("mtp."))
    {
        let expected = schema.get(tensor.name).ok_or_else(|| {
            NnError::MissingTensor(format!(
                "unexpected Qwen3.5 MTP source tensor `{}`",
                tensor.name
            ))
        })?;
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
    }
    if let Some(missing) = schema.keys().find(|name| !consumed.contains(name.as_str())) {
        return Err(NnError::MissingTensor(missing.clone()));
    }
    let mtp_matrices = schema
        .values()
        .filter(|tensor| tensor.role == TensorRole::Matrix)
        .count();
    language.deferred_mtp_tensors = 0;
    Ok(Qwen35HfLanguageMtpReceipt {
        language,
        mtp_tensors: schema.len(),
        mtp_matrices,
        mtp_preserved_tensors: schema.len() - mtp_matrices,
    })
}

pub(super) fn load_mtp_weights<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
    config: &Qwen35TextConfig,
) -> Result<Qwen35MtpWeights, NnError> {
    let hidden = axis(config.hidden_size);
    let intermediate = axis(config.intermediate_size);
    let fused_width = hidden
        .checked_mul(2)
        .ok_or_else(|| invalid_geometry("MTP fusion width overflow"))?;
    let layer = "mtp.layers.0";
    Ok(Qwen35MtpWeights::new(
        vector(shards, "mtp", "pre_fc_norm_embedding.weight", hidden)?,
        vector(shards, "mtp", "pre_fc_norm_hidden.weight", hidden)?,
        dense(shards, "mtp", "fc.weight", hidden, fused_width)?,
        Qwen35MtpLayerWeights::new(
            vector(shards, layer, "input_layernorm.weight", hidden)?,
            load_full_attention(shards, config, layer, hidden)?,
            vector(shards, layer, "post_attention_layernorm.weight", hidden)?,
            SwiGluMlp::new(
                dense(shards, layer, "mlp.gate_proj.weight", intermediate, hidden)?,
                dense(shards, layer, "mlp.up_proj.weight", intermediate, hidden)?,
                dense(shards, layer, "mlp.down_proj.weight", hidden, intermediate)?,
            )?,
        ),
        vector(shards, "mtp", "norm.weight", hidden)?,
    ))
}

pub(super) fn load_language_weights<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
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

fn load_delta_net<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
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

fn load_full_attention<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
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

fn dense<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
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

fn vector<S: Qwen35HfTensorSource + ?Sized>(
    shards: &S,
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
