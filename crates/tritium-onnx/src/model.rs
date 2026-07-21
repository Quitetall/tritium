//! Deterministic ONNX protobuf serialization for Tritium inference graphs.

use std::collections::BTreeMap;

use half::f16;
use prost::Message;
use tritium_core::{TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};

use crate::{
    ATTR_FORMAT, ATTR_HEAD_DIM, ATTR_K, ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_PAST_TOKENS, ONNX_DOMAIN,
    ONNX_EMBEDDING_OP_NAME, ONNX_KV_ATTENTION_OP_NAME, ONNX_OP_NAME, QwenDeltaNetGeometry,
};

const ONNX_IR_VERSION: i64 = 10;
const ONNX_OPSET: i64 = 21;
const TRITIUM_OPSET: i64 = 1;
const TENSOR_FLOAT: i32 = 1;
const TENSOR_UINT8: i32 = 2;
const TENSOR_INT64: i32 = 7;
const ATTRIBUTE_INT: i32 = 2;
const ATTRIBUTE_INTS: i32 = 7;
const EXTERNAL_DATA: i32 = 1;
const EXTERNAL_WEIGHTS_FILE: &str = "weights.bin";
const EXTERNAL_ALIGNMENT: usize = 64;
const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;
const QWEN_SHARED_EXTERNAL_INITIALIZERS: [&str; 4] = [
    "tok_embeddings.packed",
    "tok_embeddings.scales",
    "lm_head.packed",
    "lm_head.scales",
];

/// A deterministic tied packed embedding/head graph.
///
/// The emitted model accepts one fixed-length `tokens` tensor, gathers its
/// packed embedding rows, and multiplies the hidden states by the same packed
/// table to produce `logits`. `packed` and `scales` become ONNX initializers and
/// are referenced by both nodes, preserving physical tying without a dense
/// shadow or duplicate initializer.
#[derive(Debug, Clone, Copy)]
pub struct TiedEmbeddingHeadModel<'a> {
    /// Number of input tokens in the fixed test/export graph.
    pub tokens: usize,
    /// Vocabulary/table row count.
    pub vocab: usize,
    /// Hidden/table column count.
    pub hidden: usize,
    /// Packed TQ2_0 or TQ1_0 table bytes, output-major.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per vocabulary row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
    /// Non-empty immutable source-model identity.
    pub source_model_id: &'a str,
    /// Non-empty conversion recipe identity.
    pub recipe_id: &'a str,
    /// Non-empty packed artifact/package identity.
    pub package_id: &'a str,
}

/// Complete schema-v2 identity binding for a Tritium ONNX artifact.
#[derive(Debug, Clone, Copy)]
pub struct OnnxArtifactIdentityV2<'a> {
    /// Immutable source-model identity, including resolved revision.
    pub source_model_id: &'a str,
    /// Tokenizer identity, including resolved revision.
    pub tokenizer_id: &'a str,
    /// Conversion recipe identity.
    pub recipe_id: &'a str,
    /// Tritium build/source identity that produced the graph.
    pub tritium_build_id: &'a str,
    /// Packed artifact/package identity.
    pub package_id: &'a str,
    /// Identity of the exact in-scope converted coverage ledger.
    pub converted_coverage_id: &'a str,
    /// Identity of the explicit deferred/preserved coverage ledger.
    pub deferred_coverage_id: &'a str,
}

/// A schema-v2 tied embedding/head graph with complete artifact identity.
///
/// This additive type leaves [`TiedEmbeddingHeadModel`] and its schema-v1 wire
/// format source-compatible and readable while allowing release artifacts to
/// bind the full conversion provenance contract.
#[derive(Debug, Clone, Copy)]
pub struct TiedEmbeddingHeadModelV2<'a> {
    /// Number of input tokens in the fixed graph.
    pub tokens: usize,
    /// Vocabulary/table row count.
    pub vocab: usize,
    /// Hidden/table column count.
    pub hidden: usize,
    /// Packed TQ2_0 or TQ1_0 table bytes, output-major.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per vocabulary row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
    /// Complete schema-v2 artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// One packed output-major ternary matrix consumed directly by Tritium ONNX
/// custom operators.
#[derive(Debug, Clone, Copy)]
pub struct PackedTernaryMatrix<'a> {
    /// Output row count.
    pub rows: usize,
    /// Input/contraction column count.
    pub columns: usize,
    /// Canonical TQ2_0 or TQ1_0 packed rows.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per output row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
}

/// Qwen-style full-head rotary position embedding parameters.
#[derive(Debug, Clone, Copy)]
pub struct RotaryEmbedding {
    /// Positive finite frequency base (`rope_theta`).
    pub theta: f32,
    /// Even prefix width rotated within each attention head.
    pub dimensions: usize,
}

/// Query and optional output-gate projection layout for one attention layer.
#[derive(Debug, Clone, Copy)]
pub enum CausalQueryProjection<'a> {
    /// Independent query and optional sigmoid output-gate projections.
    Separate {
        /// Query projection `[query_width, hidden]`.
        query: PackedTernaryMatrix<'a>,
        /// Optional gate projection `[query_width, hidden]`.
        gate: Option<PackedTernaryMatrix<'a>>,
    },
    /// Qwen head-interleaved `[query lanes..., gate lanes...]` projection.
    HeadInterleavedQueryGate {
        /// Fused projection `[2 * query_width, hidden]`.
        fused: PackedTernaryMatrix<'a>,
    },
}

/// Packed weights and preserved RMSNorm vectors for one causal decoder layer.
#[derive(Debug, Clone, Copy)]
pub struct CausalLmDecoderLayer<'a> {
    /// Pre-attention RMSNorm weight.
    pub attention_norm: &'a [f32],
    /// Optional per-query-head RMSNorm weight applied before RoPE.
    pub query_norm: Option<&'a [f32]>,
    /// Optional per-key-head RMSNorm weight applied before RoPE/cache write.
    pub key_norm: Option<&'a [f32]>,
    /// Query and optional sigmoid output-gate projection layout.
    pub query: CausalQueryProjection<'a>,
    /// Key projection.
    pub key: PackedTernaryMatrix<'a>,
    /// Value projection.
    pub value: PackedTernaryMatrix<'a>,
    /// Attention output projection.
    pub attention_output: PackedTernaryMatrix<'a>,
    /// Optional RMSNorm over attention context before the output projection.
    pub attention_sub_norm: Option<&'a [f32]>,
    /// Pre-FFN RMSNorm weight.
    pub ffn_norm: &'a [f32],
    /// Gate projection.
    pub gate: PackedTernaryMatrix<'a>,
    /// Up projection multiplied by the activated gate.
    pub up: PackedTernaryMatrix<'a>,
    /// Optional RMSNorm over the gated intermediate before the down projection.
    pub ffn_sub_norm: Option<&'a [f32]>,
    /// Gate activation semantics.
    pub activation: CausalActivation,
    /// Down projection.
    pub down: PackedTernaryMatrix<'a>,
}

#[derive(Debug, Clone, Copy)]
struct CausalFfnLayer<'a> {
    ffn_norm: &'a [f32],
    gate: PackedTernaryMatrix<'a>,
    up: PackedTernaryMatrix<'a>,
    ffn_sub_norm: Option<&'a [f32]>,
    activation: CausalActivation,
    down: PackedTernaryMatrix<'a>,
}

impl<'a> CausalLmDecoderLayer<'a> {
    fn ffn(self) -> CausalFfnLayer<'a> {
        CausalFfnLayer {
            ffn_norm: self.ffn_norm,
            gate: self.gate,
            up: self.up,
            ffn_sub_norm: self.ffn_sub_norm,
            activation: self.activation,
            down: self.down,
        }
    }
}

/// Packed projections and preserved parameters for one Qwen DeltaNet decoder layer.
#[derive(Debug, Clone, Copy)]
pub struct QwenDeltaNetDecoderLayer<'a> {
    /// Zero-centered pre-mixer RMSNorm parameter.
    pub attention_norm: &'a [f32],
    /// Globally split packed Q/K/V projection `[conv_width, hidden]`.
    pub qkv: PackedTernaryMatrix<'a>,
    /// Packed output-gate projection `[value_width, hidden]`.
    pub z: PackedTernaryMatrix<'a>,
    /// Packed beta-logit projection `[num_value_heads, hidden]`.
    pub beta: PackedTernaryMatrix<'a>,
    /// Packed decay-logit projection `[num_value_heads, hidden]`.
    pub decay: PackedTernaryMatrix<'a>,
    /// Depthwise convolution weights `[conv_width, conv_kernel_dim]`.
    pub conv_weight: &'a [f32],
    /// Per-value-head RMSNorm weights `[value_head_dim]`.
    pub norm_weight: &'a [f32],
    /// Per-value-head delta-time bias.
    pub dt_bias: &'a [f32],
    /// Per-value-head logarithmic decay coefficient.
    pub a_log: &'a [f32],
    /// Packed mixer output projection `[hidden, value_width]`.
    pub output: PackedTernaryMatrix<'a>,
    /// Zero-centered pre-FFN RMSNorm parameter.
    pub ffn_norm: &'a [f32],
    /// Packed SwiGLU gate projection.
    pub gate: PackedTernaryMatrix<'a>,
    /// Packed SwiGLU up projection.
    pub up: PackedTernaryMatrix<'a>,
    /// Packed SwiGLU down projection.
    pub down: PackedTernaryMatrix<'a>,
}

/// Architecture-exact Qwen full-attention decoder layer.
///
/// Q/K normalization, fused head-interleaved query/gate projection, and
/// SwiGLU are structural rather than optional. Qwen does not apply an
/// additional attention-context or FFN-intermediate subnorm.
#[derive(Debug, Clone, Copy)]
pub struct QwenFullAttentionDecoderLayer<'a> {
    /// Zero-centered pre-attention RMSNorm parameter.
    pub attention_norm: &'a [f32],
    /// Mandatory per-query-head RMSNorm parameter.
    pub query_norm: &'a [f32],
    /// Mandatory per-key-head RMSNorm parameter.
    pub key_norm: &'a [f32],
    /// Fused head-interleaved query/output-gate projection.
    pub fused_query_gate: PackedTernaryMatrix<'a>,
    /// Key projection.
    pub key: PackedTernaryMatrix<'a>,
    /// Value projection.
    pub value: PackedTernaryMatrix<'a>,
    /// Attention output projection.
    pub attention_output: PackedTernaryMatrix<'a>,
    /// Zero-centered pre-FFN RMSNorm parameter.
    pub ffn_norm: &'a [f32],
    /// SwiGLU gate projection.
    pub gate: PackedTernaryMatrix<'a>,
    /// SwiGLU up projection.
    pub up: PackedTernaryMatrix<'a>,
    /// SwiGLU down projection.
    pub down: PackedTernaryMatrix<'a>,
}

impl<'a> QwenFullAttentionDecoderLayer<'a> {
    fn causal(self) -> CausalLmDecoderLayer<'a> {
        CausalLmDecoderLayer {
            attention_norm: self.attention_norm,
            query_norm: Some(self.query_norm),
            key_norm: Some(self.key_norm),
            query: CausalQueryProjection::HeadInterleavedQueryGate {
                fused: self.fused_query_gate,
            },
            key: self.key,
            value: self.value,
            attention_output: self.attention_output,
            attention_sub_norm: None,
            ffn_norm: self.ffn_norm,
            gate: self.gate,
            up: self.up,
            ffn_sub_norm: None,
            activation: CausalActivation::SwiGlu,
            down: self.down,
        }
    }
}

impl<'a> QwenDeltaNetDecoderLayer<'a> {
    fn ffn(self) -> CausalFfnLayer<'a> {
        CausalFfnLayer {
            ffn_norm: self.ffn_norm,
            gate: self.gate,
            up: self.up,
            ffn_sub_norm: None,
            activation: CausalActivation::SwiGlu,
            down: self.down,
        }
    }
}

/// Fixed-shape whole Qwen DeltaNet decoder-layer graph.
///
/// Inputs are `hidden`, `conv_state`, and `recurrent_state`. Outputs are
/// `next_hidden`, `next_conv`, and `next_recurrent`.
#[derive(Debug, Clone, Copy)]
pub struct QwenDeltaNetLayerModel<'a> {
    /// Token rows processed by this transition.
    pub tokens: usize,
    /// Residual-stream width.
    pub hidden: usize,
    /// Positive finite RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// DeltaNet state geometry.
    pub geometry: QwenDeltaNetGeometry,
    /// Layer weights and preserved parameters.
    pub layer: QwenDeltaNetDecoderLayer<'a>,
    /// Complete artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// One layer in an ordered heterogeneous Qwen language-model schedule.
#[derive(Debug, Clone, Copy)]
pub enum QwenCausalLmDecoderLayer<'a> {
    /// Gated DeltaNet recurrence with explicit convolution and recurrent state.
    DeltaNet(QwenDeltaNetDecoderLayer<'a>),
    /// Gated grouped-query causal attention with KV cache state.
    FullAttention(QwenFullAttentionDecoderLayer<'a>),
}

/// Fixed-shape packed heterogeneous Qwen causal language model.
///
/// DeltaNet layers consume `conv_state.{layer}` and
/// `recurrent_state.{layer}`. Full-attention layers consume `past_k.{layer}`
/// and `past_v.{layer}` when `past_tokens > 0`. Every layer publishes its
/// corresponding complete next state under the same sparse layer index.
#[derive(Debug, Clone, Copy)]
pub struct QwenCausalLmModel<'a> {
    /// Query token count fixed into this graph.
    pub tokens: usize,
    /// Prefix-cache token count fixed into full-attention layers.
    pub past_tokens: usize,
    /// Full-attention query head count.
    pub n_head: usize,
    /// Full-attention key/value head count.
    pub n_kv_head: usize,
    /// Full-attention elements per head.
    pub head_dim: usize,
    /// Qwen rotary embedding applied only to full-attention layers.
    pub rotary: RotaryEmbedding,
    /// Positive finite zero-centered RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Shared DeltaNet geometry for every recurrent layer.
    pub delta_geometry: QwenDeltaNetGeometry,
    /// Packed token embedding table.
    pub embedding: PackedTernaryMatrix<'a>,
    /// Optional untied packed language head.
    pub lm_head: Option<PackedTernaryMatrix<'a>>,
    /// Exact ordered heterogeneous schedule.
    pub layers: &'a [QwenCausalLmDecoderLayer<'a>],
    /// Zero-centered final RMSNorm parameter.
    pub final_norm: &'a [f32],
    /// Complete artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// One checkpoint layer in Qwen3.5/Qwen3.6 mixed language schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerType {
    /// Gated DeltaNet (`linear_attention`).
    DeltaNet,
    /// Gated grouped-query causal attention (`full_attention`).
    FullAttention,
}

/// Exact packed Qwen3.5/Qwen3.6 language-plus-MTP architecture contract.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35Config<'a> {
    /// Residual-stream width.
    pub hidden: usize,
    /// SwiGLU intermediate width.
    pub intermediate: usize,
    /// Padded vocabulary width.
    pub vocab: usize,
    /// Full-attention query heads.
    pub n_head: usize,
    /// Full-attention key/value heads.
    pub n_kv_head: usize,
    /// Full-attention width per head.
    pub head_dim: usize,
    /// Prefix of each full-attention head rotated by RoPE.
    pub rotary_dim: usize,
    /// Unscaled RoPE base.
    pub rope_theta: f32,
    /// Zero-centered decoder RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Shared Gated DeltaNet geometry.
    pub delta_geometry: QwenDeltaNetGeometry,
    /// Exact layer-by-layer schedule.
    pub layer_types: &'a [Qwen35LayerType],
    /// Canonical interval whose final layer is full attention.
    pub full_attention_interval: usize,
    /// Whether language embedding and head alias.
    pub tied_embeddings: bool,
    /// Bundled MTP decoder-layer count.
    pub mtp_layers: usize,
    /// Whether MTP owns a separate token table.
    pub mtp_dedicated_embeddings: bool,
}

/// Exact 15-tensor one-layer Qwen MTP drafter bundle.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MtpDecoder<'a> {
    /// Zero-centered norm over shifted shared token embeddings.
    pub pre_fc_norm_embedding: &'a [f32],
    /// Zero-centered norm over target final hidden rows.
    pub pre_fc_norm_hidden: &'a [f32],
    /// Fusion projection `[hidden, 2 * hidden]`.
    pub fusion: PackedTernaryMatrix<'a>,
    /// Forced full-attention decoder layer.
    pub layer: QwenFullAttentionDecoderLayer<'a>,
    /// Zero-centered final MTP RMSNorm parameter.
    pub final_norm: &'a [f32],
}

/// Fixed-shape executable Qwen MTP prompt or cached-decode graph.
///
/// Callers supply already aligned shifted token IDs and exact final-normalized
/// target hidden rows. Cached graphs consume `past_k.0`/`past_v.0` and publish
/// complete `present_k.0`/`present_v.0` state.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MtpModel<'a> {
    /// Shifted token rows processed by this graph.
    pub tokens: usize,
    /// Existing MTP KV-cache rows.
    pub past_tokens: usize,
    /// MTP query heads.
    pub n_head: usize,
    /// MTP key/value heads.
    pub n_kv_head: usize,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Partial Qwen RoPE contract.
    pub rotary: RotaryEmbedding,
    /// Positive finite zero-centered RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Shared language token embedding.
    pub embedding: PackedTernaryMatrix<'a>,
    /// Shared untied language head.
    pub lm_head: PackedTernaryMatrix<'a>,
    /// Exact one-layer MTP weights.
    pub mtp: Qwen35MtpDecoder<'a>,
    /// Complete artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// Borrowed canonical tensor source for packed Qwen language-plus-MTP export.
pub trait Qwen35TensorProvider<'a> {
    /// Enumerate complete in-scope canonical tensor names.
    fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError>;
    /// Resolve one canonical rank-two tensor as packed ternary data.
    fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError>;
    /// Resolve one preserved tensor flattened in canonical row-major order.
    fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError>;
}

/// Exact Qwen3.5/Qwen3.6 mapping after fail-closed tensor admission.
#[derive(Debug)]
pub struct MappedQwen35<'a> {
    config: Qwen35Config<'a>,
    embedding: PackedTernaryMatrix<'a>,
    lm_head: PackedTernaryMatrix<'a>,
    layers: Vec<QwenCausalLmDecoderLayer<'a>>,
    final_norm: &'a [f32],
    mtp: Qwen35MtpDecoder<'a>,
    identity: OnnxArtifactIdentityV2<'a>,
}

impl<'a> MappedQwen35<'a> {
    /// Canonically ordered heterogeneous language schedule.
    #[must_use]
    pub fn layers(&self) -> &[QwenCausalLmDecoderLayer<'a>] {
        &self.layers
    }

    /// Exact bundled one-layer MTP weights.
    #[must_use]
    pub const fn mtp(&self) -> &Qwen35MtpDecoder<'a> {
        &self.mtp
    }

    /// Borrow language weights as fixed prompt/decode graph.
    #[must_use]
    pub fn model(&self, tokens: usize, past_tokens: usize) -> QwenCausalLmModel<'_> {
        QwenCausalLmModel {
            tokens,
            past_tokens,
            n_head: self.config.n_head,
            n_kv_head: self.config.n_kv_head,
            head_dim: self.config.head_dim,
            rotary: RotaryEmbedding {
                theta: self.config.rope_theta,
                dimensions: self.config.rotary_dim,
            },
            rms_epsilon: self.config.rms_epsilon,
            delta_geometry: self.config.delta_geometry,
            embedding: self.embedding,
            lm_head: Some(self.lm_head),
            layers: &self.layers,
            final_norm: self.final_norm,
            identity: self.identity,
        }
    }

    /// Borrow bundled MTP weights as a fixed prompt/decode graph.
    #[must_use]
    pub fn mtp_model(&self, tokens: usize, past_tokens: usize) -> Qwen35MtpModel<'_> {
        Qwen35MtpModel {
            tokens,
            past_tokens,
            n_head: self.config.n_head,
            n_kv_head: self.config.n_kv_head,
            head_dim: self.config.head_dim,
            rotary: RotaryEmbedding {
                theta: self.config.rope_theta,
                dimensions: self.config.rotary_dim,
            },
            rms_epsilon: self.config.rms_epsilon,
            embedding: self.embedding,
            lm_head: self.lm_head,
            mtp: self.mtp,
            identity: self.identity,
        }
    }
}

/// Fixed-shape packed decoder-only causal language-model graph.
///
/// Prompt graphs set `past_tokens` to zero and consume token IDs only. Cached
/// decode graphs consume one `past_k.{layer}` and `past_v.{layer}` tensor per
/// layer and return complete `present_k.{layer}`/`present_v.{layer}` tensors.
#[derive(Debug, Clone, Copy)]
pub struct CausalLmModel<'a> {
    /// Query token count fixed into this graph.
    pub tokens: usize,
    /// Prefix-cache token count fixed into this graph.
    pub past_tokens: usize,
    /// Query attention head count.
    pub n_head: usize,
    /// Key/value attention head count.
    pub n_kv_head: usize,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Optional rotary position embedding applied to Q and newly produced K.
    pub rotary: Option<RotaryEmbedding>,
    /// Positive finite RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Interpret every preserved RMSNorm parameter as a zero-centered offset.
    ///
    /// When true, the effective scale is `1 + weight`, matching Qwen3.5.
    pub zero_centered_norm: bool,
    /// Tied token embedding and language-model head table.
    pub embedding: PackedTernaryMatrix<'a>,
    /// Optional untied language-model head; `None` reuses [`Self::embedding`].
    pub lm_head: Option<PackedTernaryMatrix<'a>>,
    /// Ordered decoder layers; at least one is required.
    pub layers: &'a [CausalLmDecoderLayer<'a>],
    /// Final RMSNorm weight.
    pub final_norm: &'a [f32],
    /// Complete artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// SmolLM2/Hugging Face geometry supported by the homogeneous causal exporter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmolLm2Config {
    /// Decoder layer count.
    pub layers: usize,
    /// Residual-stream width.
    pub hidden: usize,
    /// SwiGLU intermediate width.
    pub intermediate: usize,
    /// Token vocabulary size.
    pub vocab: usize,
    /// Query attention head count.
    pub n_head: usize,
    /// Key/value attention head count.
    pub n_kv_head: usize,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Rotary embedding base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Whether the language-model head aliases the token embedding.
    pub tied_embeddings: bool,
    /// Whether attention or MLP projections carry additive biases.
    pub projection_bias: bool,
    /// Decoder activation semantics.
    pub activation: CausalActivation,
    /// Rotary-position semantics.
    pub rotary_mode: RotaryMode,
}

/// Activation families relevant to admitted causal architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalActivation {
    /// Gated SiLU (`SiLU(gate) * up`).
    SwiGlu,
    /// Squared ReLU used by BitNet-family decoders.
    Relu2,
}

/// Rotary-position coverage/scaling modes relevant to causal architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryMode {
    /// Unscaled RoPE over the entire attention head.
    Full,
    /// RoPE over only a prefix of each attention head.
    Partial,
    /// Any position-dependent or context-scaled RoPE variant.
    Scaled,
}

/// Borrowed canonical tensor source used by the SmolLM2 ONNX adapter.
pub trait SmolLm2TensorProvider<'a> {
    /// Enumerate the complete canonical source tensor set, including unsupported extras.
    fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError>;
    /// Resolve one canonical Hugging Face rank-two tensor as packed ternary data.
    fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError>;
    /// Resolve one canonical preserved fp32 vector.
    fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError>;
}

/// Borrowed SmolLM2 graph after canonical tensor-name mapping.
#[derive(Debug)]
pub struct MappedSmolLm2<'a> {
    config: SmolLm2Config,
    embedding: PackedTernaryMatrix<'a>,
    layers: Vec<CausalLmDecoderLayer<'a>>,
    final_norm: &'a [f32],
    identity: OnnxArtifactIdentityV2<'a>,
}

impl<'a> MappedSmolLm2<'a> {
    /// Canonically ordered decoder layers.
    #[must_use]
    pub fn layers(&self) -> &[CausalLmDecoderLayer<'a>] {
        &self.layers
    }

    /// Borrow this mapped architecture as a fixed prompt/decode graph.
    #[must_use]
    pub fn model(&self, tokens: usize, past_tokens: usize) -> CausalLmModel<'_> {
        CausalLmModel {
            tokens,
            past_tokens,
            n_head: self.config.n_head,
            n_kv_head: self.config.n_kv_head,
            head_dim: self.config.head_dim,
            rotary: Some(RotaryEmbedding {
                theta: self.config.rope_theta,
                dimensions: self.config.head_dim,
            }),
            rms_epsilon: self.config.rms_epsilon,
            zero_centered_norm: false,
            embedding: self.embedding,
            lm_head: None,
            layers: &self.layers,
            final_norm: self.final_norm,
            identity: self.identity,
        }
    }
}

/// Map canonical SmolLM2 Hugging Face tensors into the causal ONNX contract.
///
/// # Errors
/// Returns [`OnnxModelError`] for missing tensors or any geometry/packed-data
/// mismatch. This adapter deliberately represents only tied-head, bias-free,
/// full-RoPE SwiGLU SmolLM2 graphs.
pub fn map_smollm2_causal_lm<'a>(
    source: &'a impl SmolLm2TensorProvider<'a>,
    config: SmolLm2Config,
    identity: OnnxArtifactIdentityV2<'a>,
) -> Result<MappedSmolLm2<'a>, OnnxModelError> {
    validate_smollm2_config(config)?;
    let expected_names = smollm2_tensor_names(config.layers)?;
    validate_exact_tensor_set("SmolLM2", source.tensor_names()?, &expected_names)?;
    let embedding = source.matrix("model.embed_tokens.weight")?;
    let final_norm = source.vector("model.norm.weight")?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers)
        .map_err(|_| OnnxModelError::ShapeOverflow("SmolLM2 layer allocation"))?;
    for index in 0..config.layers {
        let prefix = format!("model.layers.{index}");
        layers.push(CausalLmDecoderLayer {
            attention_norm: source.vector(&format!("{prefix}.input_layernorm.weight"))?,
            query_norm: None,
            key_norm: None,
            query: CausalQueryProjection::Separate {
                query: source.matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
                gate: None,
            },
            key: source.matrix(&format!("{prefix}.self_attn.k_proj.weight"))?,
            value: source.matrix(&format!("{prefix}.self_attn.v_proj.weight"))?,
            attention_output: source.matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
            attention_sub_norm: None,
            ffn_norm: source.vector(&format!("{prefix}.post_attention_layernorm.weight"))?,
            gate: source.matrix(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up: source.matrix(&format!("{prefix}.mlp.up_proj.weight"))?,
            ffn_sub_norm: None,
            activation: CausalActivation::SwiGlu,
            down: source.matrix(&format!("{prefix}.mlp.down_proj.weight"))?,
        });
    }
    let mapped = MappedSmolLm2 {
        config,
        embedding,
        layers,
        final_norm,
        identity,
    };
    validate_causal_lm(&mapped.model(1, 0))?;
    if mapped.embedding.rows != config.vocab
        || mapped.embedding.columns != config.hidden
        || mapped.layers.iter().any(|layer| {
            layer.gate.rows != config.intermediate || layer.up.rows != config.intermediate
        })
    {
        return Err(OnnxModelError::InvalidModel(
            "SmolLM2 tensor geometry differs from architecture config".to_owned(),
        ));
    }
    Ok(mapped)
}

fn validate_smollm2_config(config: SmolLm2Config) -> Result<(), OnnxModelError> {
    validate_causal_adapter_geometry(CausalAdapterGeometry {
        family: "SmolLM2",
        layers: config.layers,
        hidden: config.hidden,
        intermediate: config.intermediate,
        vocab: config.vocab,
        n_head: config.n_head,
        n_kv_head: config.n_kv_head,
        head_dim: config.head_dim,
        rope_theta: config.rope_theta,
        rms_epsilon: config.rms_epsilon,
    })?;
    if !config.tied_embeddings
        || config.projection_bias
        || config.activation != CausalActivation::SwiGlu
        || config.rotary_mode != RotaryMode::Full
    {
        return Err(OnnxModelError::InvalidModel(
            "SmolLM2 adapter requires tied embeddings, bias-free SwiGLU and full unscaled RoPE"
                .to_owned(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct CausalAdapterGeometry {
    family: &'static str,
    layers: usize,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_theta: f32,
    rms_epsilon: f32,
}

fn validate_causal_adapter_geometry(geometry: CausalAdapterGeometry) -> Result<(), OnnxModelError> {
    if geometry.layers == 0
        || geometry.hidden == 0
        || geometry.intermediate == 0
        || geometry.vocab == 0
        || geometry.n_head == 0
        || geometry.n_kv_head == 0
        || geometry.head_dim == 0
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{} config dimensions must be nonzero",
            geometry.family
        )));
    }
    let projected_hidden =
        geometry
            .n_head
            .checked_mul(geometry.head_dim)
            .ok_or(OnnxModelError::ShapeOverflow(
                "causal adapter attention width",
            ))?;
    if projected_hidden != geometry.hidden || !geometry.n_head.is_multiple_of(geometry.n_kv_head) {
        return Err(OnnxModelError::InvalidModel(format!(
            "{} attention geometry differs from hidden width or GQA grouping",
            geometry.family
        )));
    }
    if geometry.head_dim < 2 || !geometry.head_dim.is_multiple_of(2) {
        return Err(OnnxModelError::InvalidModel(format!(
            "{} full RoPE requires a positive even head dimension",
            geometry.family
        )));
    }
    if !geometry.rope_theta.is_finite()
        || geometry.rope_theta <= 0.0
        || !geometry.rms_epsilon.is_finite()
        || geometry.rms_epsilon <= 0.0
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{} RoPE theta and RMS epsilon must be positive and finite",
            geometry.family
        )));
    }
    Ok(())
}

fn smollm2_tensor_names(layers: usize) -> Result<Vec<String>, OnnxModelError> {
    let count = layers
        .checked_mul(9)
        .and_then(|count| count.checked_add(2))
        .ok_or(OnnxModelError::ShapeOverflow("SmolLM2 tensor-name count"))?;
    let mut names = Vec::new();
    names
        .try_reserve_exact(count)
        .map_err(|_| OnnxModelError::ShapeOverflow("SmolLM2 tensor-name allocation"))?;
    names.push("model.embed_tokens.weight".to_owned());
    names.push("model.norm.weight".to_owned());
    for index in 0..layers {
        let prefix = format!("model.layers.{index}");
        names.extend([
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
        ]);
    }
    Ok(names)
}

fn validate_exact_tensor_set(
    family: &str,
    actual: &[String],
    expected: &[String],
) -> Result<(), OnnxModelError> {
    let mut actual_sorted = Vec::new();
    actual_sorted
        .try_reserve_exact(actual.len())
        .map_err(|_| OnnxModelError::ShapeOverflow("adapter source manifest allocation"))?;
    actual_sorted.extend(actual.iter().map(String::as_str));
    actual_sorted.sort_unstable();
    if actual_sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(OnnxModelError::InvalidModel(format!(
            "{family} source tensor manifest contains duplicate names"
        )));
    }
    let mut expected_sorted = Vec::new();
    expected_sorted
        .try_reserve_exact(expected.len())
        .map_err(|_| OnnxModelError::ShapeOverflow("adapter expected manifest allocation"))?;
    expected_sorted.extend(expected.iter().map(String::as_str));
    expected_sorted.sort_unstable();
    if let Some(name) = expected_sorted
        .iter()
        .find(|name| actual_sorted.binary_search(name).is_err())
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{family} source is missing tensor {name}"
        )));
    }
    if let Some(name) = actual_sorted
        .iter()
        .find(|name| expected_sorted.binary_search(name).is_err())
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{family} source contains unsupported tensor {name}"
        )));
    }
    Ok(())
}

const QWEN35_EMBEDDING_TENSOR: &str = "model.language_model.embed_tokens.weight";
const QWEN35_FINAL_NORM_TENSOR: &str = "model.language_model.norm.weight";
const QWEN35_LM_HEAD_TENSOR: &str = "lm_head.weight";

struct Qwen35LayerTensorNames {
    attention_norm: String,
    ffn_norm: String,
    gate: String,
    up: String,
    down: String,
    mixer: Qwen35MixerTensorNames,
}

enum Qwen35MixerTensorNames {
    DeltaNet {
        qkv: String,
        z: String,
        beta: String,
        decay: String,
        output: String,
        conv_weight: String,
        norm_weight: String,
        dt_bias: String,
        a_log: String,
    },
    FullAttention {
        fused_query_gate: String,
        key: String,
        value: String,
        output: String,
        query_norm: String,
        key_norm: String,
    },
}

impl Qwen35LayerTensorNames {
    fn new(prefix: &str, layer_type: Qwen35LayerType) -> Self {
        let name = |suffix: &str| format!("{prefix}.{suffix}");
        Self {
            attention_norm: name("input_layernorm.weight"),
            ffn_norm: name("post_attention_layernorm.weight"),
            gate: name("mlp.gate_proj.weight"),
            up: name("mlp.up_proj.weight"),
            down: name("mlp.down_proj.weight"),
            mixer: match layer_type {
                Qwen35LayerType::DeltaNet => Qwen35MixerTensorNames::DeltaNet {
                    qkv: name("linear_attn.in_proj_qkv.weight"),
                    z: name("linear_attn.in_proj_z.weight"),
                    beta: name("linear_attn.in_proj_b.weight"),
                    decay: name("linear_attn.in_proj_a.weight"),
                    output: name("linear_attn.out_proj.weight"),
                    conv_weight: name("linear_attn.conv1d.weight"),
                    norm_weight: name("linear_attn.norm.weight"),
                    dt_bias: name("linear_attn.dt_bias"),
                    a_log: name("linear_attn.A_log"),
                },
                Qwen35LayerType::FullAttention => Qwen35MixerTensorNames::FullAttention {
                    fused_query_gate: name("self_attn.q_proj.weight"),
                    key: name("self_attn.k_proj.weight"),
                    value: name("self_attn.v_proj.weight"),
                    output: name("self_attn.o_proj.weight"),
                    query_norm: name("self_attn.q_norm.weight"),
                    key_norm: name("self_attn.k_norm.weight"),
                },
            },
        }
    }

    fn extend_manifest(self, names: &mut Vec<String>) {
        names.extend([
            self.attention_norm,
            self.ffn_norm,
            self.gate,
            self.up,
            self.down,
        ]);
        match self.mixer {
            Qwen35MixerTensorNames::DeltaNet {
                qkv,
                z,
                beta,
                decay,
                output,
                conv_weight,
                norm_weight,
                dt_bias,
                a_log,
            } => names.extend([
                qkv,
                z,
                beta,
                decay,
                output,
                conv_weight,
                norm_weight,
                dt_bias,
                a_log,
            ]),
            Qwen35MixerTensorNames::FullAttention {
                fused_query_gate,
                key,
                value,
                output,
                query_norm,
                key_norm,
            } => names.extend([fused_query_gate, key, value, output, query_norm, key_norm]),
        }
    }
}

struct Qwen35MtpTensorNames {
    pre_fc_norm_embedding: &'static str,
    pre_fc_norm_hidden: &'static str,
    fusion: &'static str,
    layer: Qwen35LayerTensorNames,
    final_norm: &'static str,
}

impl Qwen35MtpTensorNames {
    fn canonical() -> Self {
        Self {
            pre_fc_norm_embedding: "mtp.pre_fc_norm_embedding.weight",
            pre_fc_norm_hidden: "mtp.pre_fc_norm_hidden.weight",
            fusion: "mtp.fc.weight",
            layer: Qwen35LayerTensorNames::new("mtp.layers.0", Qwen35LayerType::FullAttention),
            final_norm: "mtp.norm.weight",
        }
    }

    fn extend_manifest(self, names: &mut Vec<String>) {
        names.extend([
            self.pre_fc_norm_embedding.to_owned(),
            self.pre_fc_norm_hidden.to_owned(),
            self.fusion.to_owned(),
        ]);
        self.layer.extend_manifest(names);
        names.push(self.final_norm.to_owned());
    }
}

/// Map exact Qwen3.5/Qwen3.6 language-plus-MTP names into packed ONNX contracts.
///
/// Vision tensors are outside this provider boundary and remain explicitly
/// represented by the artifact's deferred-coverage identity.
///
/// # Errors
/// Returns [`OnnxModelError`] for unsupported config semantics, incomplete or
/// extra tensor names, or any packed/preserved geometry mismatch.
pub fn map_qwen35_causal_lm<'a>(
    source: &'a impl Qwen35TensorProvider<'a>,
    config: Qwen35Config<'a>,
    identity: OnnxArtifactIdentityV2<'a>,
) -> Result<MappedQwen35<'a>, OnnxModelError> {
    validate_qwen35_config(config)?;
    let expected_names = qwen35_tensor_names(config.layer_types)?;
    validate_exact_tensor_set("Qwen3.5", source.tensor_names()?, &expected_names)?;

    let embedding = source.matrix(QWEN35_EMBEDDING_TENSOR)?;
    let lm_head = source.matrix(QWEN35_LM_HEAD_TENSOR)?;
    let final_norm = source.vector(QWEN35_FINAL_NORM_TENSOR)?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layer_types.len())
        .map_err(|_| OnnxModelError::ShapeOverflow("Qwen layer allocation"))?;
    for (index, layer_type) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let names = Qwen35LayerTensorNames::new(&prefix, layer_type);
        let attention_norm = source.vector(&names.attention_norm)?;
        let ffn_norm = source.vector(&names.ffn_norm)?;
        let gate = source.matrix(&names.gate)?;
        let up = source.matrix(&names.up)?;
        let down = source.matrix(&names.down)?;
        layers.push(match &names.mixer {
            Qwen35MixerTensorNames::DeltaNet {
                qkv,
                z,
                beta,
                decay,
                output,
                conv_weight,
                norm_weight,
                dt_bias,
                a_log,
            } => QwenCausalLmDecoderLayer::DeltaNet(QwenDeltaNetDecoderLayer {
                attention_norm,
                qkv: source.matrix(qkv)?,
                z: source.matrix(z)?,
                beta: source.matrix(beta)?,
                decay: source.matrix(decay)?,
                conv_weight: source.vector(conv_weight)?,
                norm_weight: source.vector(norm_weight)?,
                dt_bias: source.vector(dt_bias)?,
                a_log: source.vector(a_log)?,
                output: source.matrix(output)?,
                ffn_norm,
                gate,
                up,
                down,
            }),
            Qwen35MixerTensorNames::FullAttention { .. } => {
                QwenCausalLmDecoderLayer::FullAttention(map_qwen35_full_attention(
                    source,
                    &names,
                    attention_norm,
                    ffn_norm,
                    gate,
                    up,
                    down,
                )?)
            }
        });
    }

    let mtp_names = Qwen35MtpTensorNames::canonical();
    let mtp = Qwen35MtpDecoder {
        pre_fc_norm_embedding: source.vector(mtp_names.pre_fc_norm_embedding)?,
        pre_fc_norm_hidden: source.vector(mtp_names.pre_fc_norm_hidden)?,
        fusion: source.matrix(mtp_names.fusion)?,
        layer: map_qwen35_full_attention(
            source,
            &mtp_names.layer,
            source.vector(&mtp_names.layer.attention_norm)?,
            source.vector(&mtp_names.layer.ffn_norm)?,
            source.matrix(&mtp_names.layer.gate)?,
            source.matrix(&mtp_names.layer.up)?,
            source.matrix(&mtp_names.layer.down)?,
        )?,
        final_norm: source.vector(mtp_names.final_norm)?,
    };
    let mapped = MappedQwen35 {
        config,
        embedding,
        lm_head,
        layers,
        final_norm,
        mtp,
        identity,
    };
    validate_qwen_causal_lm(&mapped.model(1, 0))?;
    validate_qwen35_mapped_geometry(&mapped)?;
    Ok(mapped)
}

/// Map only pinned `Qwen/Qwen3.6-27B` language-plus-MTP geometry.
///
/// Family-sized fixtures use [`map_qwen35_causal_lm`], but cannot pass this
/// flagship admission contract.
///
/// # Errors
/// Returns [`OnnxModelError`] before tensor resolution when any pinned axis,
/// schedule position, RoPE value, DeltaNet geometry, or MTP semantic differs.
pub fn map_qwen36_27b_causal_lm<'a>(
    source: &'a impl Qwen35TensorProvider<'a>,
    config: Qwen35Config<'a>,
    identity: OnnxArtifactIdentityV2<'a>,
) -> Result<MappedQwen35<'a>, OnnxModelError> {
    validate_pinned_qwen36_27b_config(config)?;
    map_qwen35_causal_lm(source, config, identity)
}

fn map_qwen35_full_attention<'a>(
    source: &'a impl Qwen35TensorProvider<'a>,
    names: &Qwen35LayerTensorNames,
    attention_norm: &'a [f32],
    ffn_norm: &'a [f32],
    gate: PackedTernaryMatrix<'a>,
    up: PackedTernaryMatrix<'a>,
    down: PackedTernaryMatrix<'a>,
) -> Result<QwenFullAttentionDecoderLayer<'a>, OnnxModelError> {
    let Qwen35MixerTensorNames::FullAttention {
        fused_query_gate,
        key,
        value,
        output,
        query_norm,
        key_norm,
    } = &names.mixer
    else {
        return Err(OnnxModelError::InvalidModel(
            "Qwen full-attention resolver received DeltaNet tensor names".to_owned(),
        ));
    };
    Ok(QwenFullAttentionDecoderLayer {
        attention_norm,
        query_norm: source.vector(query_norm)?,
        key_norm: source.vector(key_norm)?,
        fused_query_gate: source.matrix(fused_query_gate)?,
        key: source.matrix(key)?,
        value: source.matrix(value)?,
        attention_output: source.matrix(output)?,
        ffn_norm,
        gate,
        up,
        down,
    })
}

fn validate_qwen35_config(config: Qwen35Config<'_>) -> Result<(), OnnxModelError> {
    if config.hidden == 0
        || config.intermediate == 0
        || config.vocab == 0
        || config.n_head == 0
        || config.n_kv_head == 0
        || config.head_dim == 0
        || config.rotary_dim == 0
        || config.full_attention_interval == 0
        || config.layer_types.is_empty()
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen3.5 config dimensions and schedule must be nonzero".to_owned(),
        ));
    }
    if !config.n_head.is_multiple_of(config.n_kv_head)
        || !config.head_dim.is_multiple_of(2)
        || !config.rotary_dim.is_multiple_of(2)
        || config.rotary_dim > config.head_dim
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen3.5 attention grouping and partial RoPE geometry are invalid".to_owned(),
        ));
    }
    if !config.rope_theta.is_finite()
        || config.rope_theta <= 0.0
        || !config.rms_epsilon.is_finite()
        || config.rms_epsilon <= 0.0
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen3.5 RoPE theta and RMS epsilon must be positive and finite".to_owned(),
        ));
    }
    if config.tied_embeddings || config.mtp_layers != 1 || config.mtp_dedicated_embeddings {
        return Err(OnnxModelError::InvalidModel(
            "Qwen3.5 adapter requires untied language head and one shared-embedding MTP layer"
                .to_owned(),
        ));
    }
    config
        .delta_geometry
        .dimensions()
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    if config
        .layer_types
        .iter()
        .copied()
        .enumerate()
        .any(|(index, layer)| {
            let expected = if (index + 1).is_multiple_of(config.full_attention_interval) {
                Qwen35LayerType::FullAttention
            } else {
                Qwen35LayerType::DeltaNet
            };
            layer != expected
        })
        || !config
            .layer_types
            .len()
            .is_multiple_of(config.full_attention_interval)
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen3.5 layer schedule differs from its declared full-attention interval".to_owned(),
        ));
    }
    Ok(())
}

fn validate_pinned_qwen36_27b_config(config: Qwen35Config<'_>) -> Result<(), OnnxModelError> {
    validate_qwen35_config(config)?;
    let delta = config.delta_geometry;
    if config.layer_types.len() != 64
        || config.full_attention_interval != 4
        || config.hidden != 5_120
        || config.intermediate != 17_408
        || config.vocab != 248_320
        || config.n_head != 24
        || config.n_kv_head != 4
        || config.head_dim != 256
        || config.rotary_dim != 64
        || config.rope_theta != 10_000_000.0
        || config.rms_epsilon != 1.0e-6
        || delta.conv_kernel_dim() != 4
        || delta.num_key_heads() != 16
        || delta.num_value_heads() != 48
        || delta.key_head_dim() != 128
        || delta.value_head_dim() != 128
    {
        return Err(OnnxModelError::InvalidModel(
            "configuration does not match pinned Qwen/Qwen3.6-27B geometry".to_owned(),
        ));
    }
    Ok(())
}

fn validate_qwen35_mapped_geometry(mapped: &MappedQwen35<'_>) -> Result<(), OnnxModelError> {
    let config = mapped.config;
    validate_matrix_shape(
        "Qwen embedding",
        mapped.embedding,
        config.vocab,
        config.hidden,
    )?;
    validate_matrix_shape("Qwen lm_head", mapped.lm_head, config.vocab, config.hidden)?;
    validate_vector("Qwen final_norm", mapped.final_norm, config.hidden)?;
    validate_vector(
        "Qwen MTP pre_fc_norm_embedding",
        mapped.mtp.pre_fc_norm_embedding,
        config.hidden,
    )?;
    validate_vector(
        "Qwen MTP pre_fc_norm_hidden",
        mapped.mtp.pre_fc_norm_hidden,
        config.hidden,
    )?;
    validate_matrix_shape(
        "Qwen MTP fusion",
        mapped.mtp.fusion,
        config.hidden,
        config
            .hidden
            .checked_mul(2)
            .ok_or(OnnxModelError::ShapeOverflow("Qwen MTP fusion width"))?,
    )?;
    validate_vector("Qwen MTP final_norm", mapped.mtp.final_norm, config.hidden)?;
    for (index, layer) in mapped.layers.iter().copied().enumerate() {
        match layer {
            QwenCausalLmDecoderLayer::DeltaNet(layer) => {
                for (suffix, matrix, rows, columns) in [
                    ("gate", layer.gate, config.intermediate, config.hidden),
                    ("up", layer.up, config.intermediate, config.hidden),
                    ("down", layer.down, config.hidden, config.intermediate),
                ] {
                    validate_matrix_shape(
                        &format!("Qwen layer {index} {suffix}"),
                        matrix,
                        rows,
                        columns,
                    )?;
                }
            }
            QwenCausalLmDecoderLayer::FullAttention(layer) => {
                validate_qwen35_full_attention_geometry(
                    &format!("Qwen layer {index}"),
                    layer,
                    config,
                )?;
            }
        }
    }
    validate_qwen35_full_attention_geometry("Qwen MTP layer", mapped.mtp.layer, config)
}

fn validate_qwen35_full_attention_geometry(
    name: &str,
    layer: QwenFullAttentionDecoderLayer<'_>,
    config: Qwen35Config<'_>,
) -> Result<(), OnnxModelError> {
    let query_width = config
        .n_head
        .checked_mul(config.head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("Qwen query width"))?;
    let kv_width = config
        .n_kv_head
        .checked_mul(config.head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("Qwen KV width"))?;
    validate_vector(
        &format!("{name} attention_norm"),
        layer.attention_norm,
        config.hidden,
    )?;
    validate_vector(
        &format!("{name} query_norm"),
        layer.query_norm,
        config.head_dim,
    )?;
    validate_vector(&format!("{name} key_norm"), layer.key_norm, config.head_dim)?;
    validate_matrix_shape(
        &format!("{name} fused_query_gate"),
        layer.fused_query_gate,
        query_width
            .checked_mul(2)
            .ok_or(OnnxModelError::ShapeOverflow("Qwen fused query width"))?,
        config.hidden,
    )?;
    validate_matrix_shape(&format!("{name} key"), layer.key, kv_width, config.hidden)?;
    validate_matrix_shape(
        &format!("{name} value"),
        layer.value,
        kv_width,
        config.hidden,
    )?;
    validate_matrix_shape(
        &format!("{name} attention_output"),
        layer.attention_output,
        config.hidden,
        query_width,
    )?;
    validate_vector(&format!("{name} ffn_norm"), layer.ffn_norm, config.hidden)?;
    validate_matrix_shape(
        &format!("{name} gate"),
        layer.gate,
        config.intermediate,
        config.hidden,
    )?;
    validate_matrix_shape(
        &format!("{name} up"),
        layer.up,
        config.intermediate,
        config.hidden,
    )?;
    validate_matrix_shape(
        &format!("{name} down"),
        layer.down,
        config.hidden,
        config.intermediate,
    )
}

fn qwen35_tensor_names(layer_types: &[Qwen35LayerType]) -> Result<Vec<String>, OnnxModelError> {
    let layer_count = layer_types.iter().try_fold(0_usize, |count, layer| {
        count
            .checked_add(match layer {
                Qwen35LayerType::DeltaNet => 14,
                Qwen35LayerType::FullAttention => 11,
            })
            .ok_or(OnnxModelError::ShapeOverflow("Qwen tensor-name count"))
    })?;
    let capacity = layer_count
        .checked_add(18)
        .ok_or(OnnxModelError::ShapeOverflow("Qwen tensor-name count"))?;
    let mut names = Vec::new();
    names
        .try_reserve_exact(capacity)
        .map_err(|_| OnnxModelError::ShapeOverflow("Qwen tensor-name allocation"))?;
    names.extend([
        QWEN35_EMBEDDING_TENSOR.to_owned(),
        QWEN35_FINAL_NORM_TENSOR.to_owned(),
        QWEN35_LM_HEAD_TENSOR.to_owned(),
    ]);
    for (index, layer_type) in layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        Qwen35LayerTensorNames::new(&prefix, layer_type).extend_manifest(&mut names);
    }
    Qwen35MtpTensorNames::canonical().extend_manifest(&mut names);
    debug_assert_eq!(names.len(), capacity);
    Ok(names)
}

/// BitNet GGUF geometry supported by the homogeneous causal exporter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitNetConfig {
    /// Decoder layer count.
    pub layers: usize,
    /// Residual-stream width.
    pub hidden: usize,
    /// ReLU2 intermediate width.
    pub intermediate: usize,
    /// Token vocabulary size.
    pub vocab: usize,
    /// Query attention head count.
    pub n_head: usize,
    /// Key/value attention head count.
    pub n_kv_head: usize,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Rotary embedding base.
    pub rope_theta: f32,
    /// RMSNorm epsilon shared by decoder norms.
    pub rms_epsilon: f32,
    /// Rotary-position semantics declared by GGUF metadata.
    pub rotary_mode: RotaryMode,
}

/// Complete packed-tensor view of a ternarized BitNet GGUF namespace.
pub trait BitNetGgufTensorProvider<'a> {
    /// Enumerate every tensor name in the GGUF, excluding metadata keys.
    fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError>;
    /// Resolve one GGUF rank-two tensor as packed ternary data.
    fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError>;
    /// Resolve one preserved GGUF fp32/fp16-widened norm vector.
    fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError>;
}

/// Borrowed BitNet graph after exact GGUF tensor-name mapping.
#[derive(Debug)]
pub struct MappedBitNet<'a> {
    config: BitNetConfig,
    embedding: PackedTernaryMatrix<'a>,
    layers: Vec<CausalLmDecoderLayer<'a>>,
    final_norm: &'a [f32],
    identity: OnnxArtifactIdentityV2<'a>,
}

impl<'a> MappedBitNet<'a> {
    /// Canonically ordered BitNet decoder layers.
    #[must_use]
    pub fn layers(&self) -> &[CausalLmDecoderLayer<'a>] {
        &self.layers
    }

    /// Borrow this mapping as a fixed prompt/decode graph.
    #[must_use]
    pub fn model(&self, tokens: usize, past_tokens: usize) -> CausalLmModel<'_> {
        CausalLmModel {
            tokens,
            past_tokens,
            n_head: self.config.n_head,
            n_kv_head: self.config.n_kv_head,
            head_dim: self.config.head_dim,
            rotary: Some(RotaryEmbedding {
                theta: self.config.rope_theta,
                dimensions: self.config.head_dim,
            }),
            rms_epsilon: self.config.rms_epsilon,
            zero_centered_norm: false,
            embedding: self.embedding,
            lm_head: None,
            layers: &self.layers,
            final_norm: self.final_norm,
            identity: self.identity,
        }
    }
}

/// Map exact BitNet GGUF tensor names into the packed causal ONNX contract.
///
/// This admits the tied-head, bias-free BitNet decoder: full RoPE, ReLU2,
/// attention-output subnorm and FFN-intermediate subnorm. Dense or untied GGUF
/// heads and extra tensors fail closed.
///
/// # Errors
/// Returns [`OnnxModelError`] for invalid config, incomplete/extra tensor sets,
/// or any preserved/packed tensor mismatch.
pub fn map_bitnet_gguf_causal_lm<'a>(
    source: &'a impl BitNetGgufTensorProvider<'a>,
    config: BitNetConfig,
    identity: OnnxArtifactIdentityV2<'a>,
) -> Result<MappedBitNet<'a>, OnnxModelError> {
    validate_bitnet_config(config)?;
    let expected_names = bitnet_gguf_tensor_names(config.layers)?;
    validate_exact_tensor_set("BitNet", source.tensor_names()?, &expected_names)?;
    let embedding = source.matrix("token_embd.weight")?;
    let final_norm = source.vector("output_norm.weight")?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers)
        .map_err(|_| OnnxModelError::ShapeOverflow("BitNet layer allocation"))?;
    for index in 0..config.layers {
        let prefix = format!("blk.{index}");
        layers.push(CausalLmDecoderLayer {
            attention_norm: source.vector(&format!("{prefix}.attn_norm.weight"))?,
            query_norm: None,
            key_norm: None,
            query: CausalQueryProjection::Separate {
                query: source.matrix(&format!("{prefix}.attn_q.weight"))?,
                gate: None,
            },
            key: source.matrix(&format!("{prefix}.attn_k.weight"))?,
            value: source.matrix(&format!("{prefix}.attn_v.weight"))?,
            attention_output: source.matrix(&format!("{prefix}.attn_output.weight"))?,
            attention_sub_norm: Some(source.vector(&format!("{prefix}.attn_sub_norm.weight"))?),
            ffn_norm: source.vector(&format!("{prefix}.ffn_norm.weight"))?,
            gate: source.matrix(&format!("{prefix}.ffn_gate.weight"))?,
            up: source.matrix(&format!("{prefix}.ffn_up.weight"))?,
            ffn_sub_norm: Some(source.vector(&format!("{prefix}.ffn_sub_norm.weight"))?),
            activation: CausalActivation::Relu2,
            down: source.matrix(&format!("{prefix}.ffn_down.weight"))?,
        });
    }
    let mapped = MappedBitNet {
        config,
        embedding,
        layers,
        final_norm,
        identity,
    };
    validate_causal_lm(&mapped.model(1, 0))?;
    if mapped.embedding.rows != config.vocab
        || mapped.embedding.columns != config.hidden
        || mapped.layers.iter().any(|layer| {
            layer.gate.rows != config.intermediate || layer.up.rows != config.intermediate
        })
    {
        return Err(OnnxModelError::InvalidModel(
            "BitNet tensor geometry differs from architecture config".to_owned(),
        ));
    }
    Ok(mapped)
}

fn validate_bitnet_config(config: BitNetConfig) -> Result<(), OnnxModelError> {
    validate_causal_adapter_geometry(CausalAdapterGeometry {
        family: "BitNet",
        layers: config.layers,
        hidden: config.hidden,
        intermediate: config.intermediate,
        vocab: config.vocab,
        n_head: config.n_head,
        n_kv_head: config.n_kv_head,
        head_dim: config.head_dim,
        rope_theta: config.rope_theta,
        rms_epsilon: config.rms_epsilon,
    })?;
    if config.rotary_mode != RotaryMode::Full {
        return Err(OnnxModelError::InvalidModel(
            "BitNet GGUF adapter requires full unscaled RoPE".to_owned(),
        ));
    }
    Ok(())
}

fn bitnet_gguf_tensor_names(layers: usize) -> Result<Vec<String>, OnnxModelError> {
    let count = layers
        .checked_mul(11)
        .and_then(|count| count.checked_add(2))
        .ok_or(OnnxModelError::ShapeOverflow("BitNet tensor-name count"))?;
    let mut names = Vec::new();
    names
        .try_reserve_exact(count)
        .map_err(|_| OnnxModelError::ShapeOverflow("BitNet tensor-name allocation"))?;
    names.push("token_embd.weight".to_owned());
    names.push("output_norm.weight".to_owned());
    for index in 0..layers {
        let prefix = format!("blk.{index}");
        names.extend([
            format!("{prefix}.attn_norm.weight"),
            format!("{prefix}.attn_q.weight"),
            format!("{prefix}.attn_k.weight"),
            format!("{prefix}.attn_v.weight"),
            format!("{prefix}.attn_output.weight"),
            format!("{prefix}.attn_sub_norm.weight"),
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.ffn_gate.weight"),
            format!("{prefix}.ffn_up.weight"),
            format!("{prefix}.ffn_sub_norm.weight"),
            format!("{prefix}.ffn_down.weight"),
        ]);
    }
    Ok(names)
}

impl<'a> TiedEmbeddingHeadModelV2<'a> {
    fn legacy(self) -> TiedEmbeddingHeadModel<'a> {
        TiedEmbeddingHeadModel {
            tokens: self.tokens,
            vocab: self.vocab,
            hidden: self.hidden,
            packed: self.packed,
            scales: self.scales,
            format: self.format,
            source_model_id: self.identity.source_model_id,
            recipe_id: self.identity.recipe_id,
            package_id: self.identity.package_id,
        }
    }
}

/// A deterministic ONNX model plus its content-bound external initializer file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalOnnxModel {
    /// Serialized `model.onnx` bytes.
    pub model_bytes: Vec<u8>,
    /// Serialized `weights.bin` bytes referenced by the model.
    pub weights_bytes: Vec<u8>,
    /// BLAKE3 digest of `weights_bytes`, also embedded in model metadata.
    pub weights_blake3: [u8; 32],
}

/// External Qwen language and MTP graphs sharing one deduplicated weight arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalQwen35Bundle {
    /// Canonical `language.onnx` bytes.
    pub language_model_bytes: Vec<u8>,
    /// Canonical `mtp.onnx` bytes.
    pub mtp_model_bytes: Vec<u8>,
    /// Shared `weights.bin`; embedding/head ranges are referenced by both graphs.
    pub weights_bytes: Vec<u8>,
    /// BLAKE3 digest of shared `weights.bin`, bound by both graphs.
    pub weights_blake3: [u8; 32],
}

/// Borrowed three-file Qwen language-plus-MTP package presented for admission.
#[derive(Debug, Clone, Copy)]
pub struct ExternalQwen35BundleFiles<'a> {
    /// Candidate `language.onnx` bytes.
    pub language_model_bytes: &'a [u8],
    /// Candidate `mtp.onnx` bytes.
    pub mtp_model_bytes: &'a [u8],
    /// Candidate shared `weights.bin` bytes.
    pub weights_bytes: &'a [u8],
}

/// Receipt from strict external ONNX model/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalOnnxModel {
    /// BLAKE3 digest of the canonical serialized ONNX protobuf.
    pub model_blake3: [u8; 32],
    /// BLAKE3 digest of the canonical external initializer bytes.
    pub weights_blake3: [u8; 32],
    /// Exact external initializer byte count.
    pub weights_bytes: usize,
    /// Fixed input token count.
    pub tokens: usize,
    /// Vocabulary row count.
    pub vocab: usize,
    /// Hidden column count.
    pub hidden: usize,
    /// Bound source model identity.
    pub source_model_id: String,
    /// Bound conversion recipe identity.
    pub recipe_id: String,
    /// Bound artifact package identity.
    pub package_id: String,
}

/// Owned identity receipt from schema-v2 external ONNX verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedOnnxArtifactIdentityV2 {
    /// Bound source model identity.
    pub source_model_id: String,
    /// Bound tokenizer identity.
    pub tokenizer_id: String,
    /// Bound conversion recipe identity.
    pub recipe_id: String,
    /// Bound Tritium build/source identity.
    pub tritium_build_id: String,
    /// Bound artifact package identity.
    pub package_id: String,
    /// Bound converted coverage-ledger identity.
    pub converted_coverage_id: String,
    /// Bound deferred/preserved coverage-ledger identity.
    pub deferred_coverage_id: String,
}

/// Receipt from strict schema-v2 external ONNX model/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalOnnxModelV2 {
    /// Verified graph, external data and geometry receipt.
    pub model: VerifiedExternalOnnxModel,
    /// Complete bound artifact identity.
    pub identity: VerifiedOnnxArtifactIdentityV2,
}

/// Receipt from strict external causal-LM graph/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalCausalLmModel {
    /// BLAKE3 digest of canonical `model.onnx` bytes.
    pub model_blake3: [u8; 32],
    /// BLAKE3 digest of authenticated `weights.bin` bytes.
    pub weights_blake3: [u8; 32],
    /// Exact external-data byte count.
    pub weights_bytes: usize,
    /// Fixed query-token count.
    pub tokens: usize,
    /// Fixed prefix-cache token count.
    pub past_tokens: usize,
    /// Decoder-layer count.
    pub layers: usize,
    /// Complete bound artifact identity.
    pub identity: VerifiedOnnxArtifactIdentityV2,
}

/// Exact causal-LM file identities copied from an independently trusted package manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmittedExternalCausalLmDigests {
    /// Admitted `model.onnx` BLAKE3.
    pub model_blake3: [u8; 32],
    /// Admitted `weights.bin` BLAKE3.
    pub weights_blake3: [u8; 32],
}

/// Three Qwen ONNX file digests copied from one authenticated package manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmittedExternalQwen35BundleDigests {
    /// Admitted `language.onnx` BLAKE3.
    pub language_model_blake3: [u8; 32],
    /// Admitted `mtp.onnx` BLAKE3.
    pub mtp_model_blake3: [u8; 32],
    /// Admitted shared `weights.bin` BLAKE3.
    pub weights_blake3: [u8; 32],
}

/// Receipt proving both external Qwen graphs belong to one admitted artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalQwen35Bundle {
    /// Verified heterogeneous language graph and external data.
    pub language: VerifiedExternalCausalLmModel,
    /// Verified one-layer MTP graph and external data.
    pub mtp: VerifiedExternalCausalLmModel,
}

/// Category of one unsupported ONNX graph item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedGraphItemKind {
    /// Unsupported node domain or operator.
    Node,
    /// Unsupported node attribute or attribute representation.
    Attribute,
    /// Unsupported tensor element type.
    Dtype,
    /// Unresolved conversion-coverage item.
    Coverage,
}

/// One typed, actionable unsupported-graph diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedGraphDiagnostic {
    /// Stable diagnostic category.
    pub kind: UnsupportedGraphItemKind,
    /// Node, attribute, tensor, or coverage path rejected.
    pub subject: String,
    /// Stable human-readable rejection reason.
    pub reason: String,
}

/// Validation or serialization errors from Tritium ONNX model encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnnxModelError {
    /// A required scalar dimension is zero.
    EmptyDimension(&'static str),
    /// A required identity string is empty.
    EmptyIdentity(&'static str),
    /// The format is not a portable packed ternary format.
    UnsupportedFormat(TernaryFormat),
    /// The scale count does not match the vocabulary.
    ScaleCount {
        /// Expected vocabulary row count.
        expected: usize,
        /// Actual scale count.
        got: usize,
    },
    /// A scale is negative or non-finite.
    InvalidScale {
        /// Index of the rejected scale.
        index: usize,
    },
    /// The packed byte count does not match the declared table.
    PackedBytes {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        got: usize,
    },
    /// A derived count cannot be represented by the host or ONNX `int64`.
    ShapeOverflow(&'static str),
    /// A packed row is malformed or carries a non-unit internal block scale.
    InvalidPackedRow {
        /// Rejected table row.
        row: usize,
        /// Stable diagnostic from the packed decoder.
        reason: String,
    },
    /// Serialized protobuf or Tritium metadata is malformed.
    InvalidModel(String),
    /// The supplied external data does not match its model binding.
    ExternalDataMismatch(String),
}

impl core::fmt::Display for OnnxModelError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDimension(name) => write!(formatter, "ONNX {name} must be positive"),
            Self::EmptyIdentity(name) => write!(formatter, "ONNX {name} must be non-empty"),
            Self::UnsupportedFormat(format) => {
                write!(formatter, "ONNX export does not support {format}")
            }
            Self::ScaleCount { expected, got } => write!(
                formatter,
                "ONNX scale count {got} does not match vocabulary {expected}"
            ),
            Self::InvalidScale { index } => write!(
                formatter,
                "ONNX scale {index} must be finite and nonnegative"
            ),
            Self::PackedBytes { expected, got } => write!(
                formatter,
                "ONNX packed table has {got} bytes, expected {expected}"
            ),
            Self::ShapeOverflow(name) => write!(formatter, "ONNX {name} exceeds int64/usize"),
            Self::InvalidPackedRow { row, reason } => {
                write!(formatter, "ONNX packed row {row} is invalid: {reason}")
            }
            Self::InvalidModel(reason) => write!(formatter, "invalid Tritium ONNX model: {reason}"),
            Self::ExternalDataMismatch(reason) => {
                write!(formatter, "Tritium ONNX external data mismatch: {reason}")
            }
        }
    }
}

impl std::error::Error for OnnxModelError {}

/// Diagnose every unsupported node, attribute, dtype, and unresolved coverage
/// item visible in a serialized Tritium ONNX graph.
///
/// Diagnostics use deterministic category order: nodes, attributes, dtypes,
/// then coverage. An empty result means these support dimensions are admitted;
/// strict graph identity and external-data admission remain verifier duties.
///
/// # Errors
/// [`OnnxModelError::InvalidModel`] if protobuf is malformed, oversized, or has
/// no graph.
pub fn diagnose_unsupported_graph(
    model_bytes: &[u8],
) -> Result<Vec<UnsupportedGraphDiagnostic>, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let graph = protobuf
        .graph
        .as_ref()
        .ok_or_else(|| OnnxModelError::InvalidModel("graph is missing".to_owned()))?;
    let mut tritium_opsets = protobuf
        .opset_import
        .iter()
        .filter(|opset| opset.domain == ONNX_DOMAIN);
    let tritium_opset = tritium_opsets.next().map(|opset| opset.version);
    if tritium_opsets.next().is_some() {
        return Err(OnnxModelError::InvalidModel(format!(
            "duplicate opset import for domain {ONNX_DOMAIN}"
        )));
    }
    let mut standard_opsets = protobuf
        .opset_import
        .iter()
        .filter(|opset| opset.domain.is_empty());
    let standard_opset = standard_opsets.next().map(|opset| opset.version);
    if standard_opsets.next().is_some() {
        return Err(OnnxModelError::InvalidModel(
            "duplicate opset import for standard ONNX domain".to_owned(),
        ));
    }
    let mut diagnostics = Vec::new();

    for node in &graph.node {
        let supported = node_supported(node, tritium_opset, standard_opset);
        if !supported {
            let reason = if node.domain == ONNX_DOMAIN
                && matches!(
                    node.op_type.as_str(),
                    ONNX_KV_ATTENTION_OP_NAME | crate::ONNX_QWEN_DELTANET_OP_NAME
                )
                && tritium_opset != Some(2)
            {
                format!(
                    "operator requires {ONNX_DOMAIN} opset 2, imported {}",
                    tritium_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else if node.domain == ONNX_DOMAIN && !matches!(tritium_opset, Some(1 | 2)) {
                format!(
                    "unsupported {ONNX_DOMAIN} opset {}",
                    tritium_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else if node.domain.is_empty() && standard_opset != Some(ONNX_OPSET) {
                format!(
                    "unsupported standard ONNX opset {}",
                    standard_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else {
                format!("unsupported operator {}::{}", node.domain, node.op_type)
            };
            diagnostics.push(UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Node,
                subject: diagnostic_subject(&node.name, "<unnamed node>"),
                reason,
            });
        }
    }
    for node in &graph.node {
        for attribute in &node.attribute {
            let subject = format!(
                "{}.{}",
                diagnostic_subject(&node.name, "<unnamed node>"),
                diagnostic_subject(&attribute.name, "<unnamed attribute>")
            );
            if !supported_attributes(node).contains(&attribute.name.as_str()) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: "unsupported attribute".to_owned(),
                });
            } else if attribute.kind != expected_attribute_kind(&attribute.name) {
                let expected = expected_attribute_kind(&attribute.name);
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "expected ONNX attribute type {expected}, got {}",
                        attribute.kind
                    ),
                });
            } else if attribute.name == ATTR_K && attribute.value <= 0 {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("K must be positive, got {}", attribute.value),
                });
            } else if attribute.name == ATTR_FORMAT && !matches!(attribute.value, 0 | 1) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("unsupported format code {}", attribute.value),
                });
            } else if matches!(
                attribute.name.as_str(),
                ATTR_N_HEAD
                    | ATTR_N_KV_HEAD
                    | ATTR_HEAD_DIM
                    | crate::ATTR_CONV_KERNEL_DIM
                    | crate::ATTR_NUM_KEY_HEADS
                    | crate::ATTR_NUM_VALUE_HEADS
                    | crate::ATTR_KEY_HEAD_DIM
                    | crate::ATTR_VALUE_HEAD_DIM
            ) && attribute.value <= 0
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "{} must be positive, got {}",
                        attribute.name, attribute.value
                    ),
                });
            } else if attribute.name == ATTR_PAST_TOKENS && attribute.value < 0 {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("past_tokens must be nonnegative, got {}", attribute.value),
                });
            } else if attribute.name == "perm" && !valid_rank_three_permutation(&attribute.ints) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "expected a rank-3 permutation of [0, 1, 2], got {:?}",
                        attribute.ints
                    ),
                });
            } else if node.op_type == "Softmax" && attribute.name == "axis" && attribute.value != -1
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("attention softmax axis must be -1, got {}", attribute.value),
                });
            } else if node.op_type == "Concat" && attribute.name == "axis" {
                match concat_role(node, graph) {
                    Some(role) if attribute.value != role.axis => {
                        diagnostics.push(UnsupportedGraphDiagnostic {
                            kind: UnsupportedGraphItemKind::Attribute,
                            subject,
                            reason: format!(
                                "{} concat axis must be {}, got {}",
                                role.label, role.axis, attribute.value
                            ),
                        });
                    }
                    None => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: "concat node has no supported semantic role".to_owned(),
                    }),
                    Some(_) => {}
                }
            } else if node.op_type == "ReduceMean"
                && attribute.name == "keepdims"
                && attribute.value != 1
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "RMSNorm reduction must keep dimensions, got {}",
                        attribute.value
                    ),
                });
            }
        }
        let supported = node_supported(node, tritium_opset, standard_opset);
        if supported {
            for &required in supported_attributes(node) {
                let count = node
                    .attribute
                    .iter()
                    .filter(|attribute| attribute.name == required)
                    .count();
                let subject = format!(
                    "{}.{}",
                    diagnostic_subject(&node.name, "<unnamed node>"),
                    required
                );
                match count {
                    0 => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: "missing required attribute".to_owned(),
                    }),
                    1 => {}
                    count => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: format!("duplicate attribute appears {count} times"),
                    }),
                }
            }
            let positive_value = |name: &str| {
                let mut attributes = node.attribute.iter().filter(|attribute| {
                    attribute.name == name && attribute.kind == ATTRIBUTE_INT && attribute.value > 0
                });
                let value = attributes.next().map(|attribute| attribute.value);
                if attributes.next().is_none() {
                    value
                } else {
                    None
                }
            };
            if node.op_type == ONNX_KV_ATTENTION_OP_NAME {
                if let (Some(n_head), Some(n_kv_head)) =
                    (positive_value(ATTR_N_HEAD), positive_value(ATTR_N_KV_HEAD))
                    && n_head % n_kv_head != 0
                {
                    diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject: format!(
                            "{}.{}",
                            diagnostic_subject(&node.name, "<unnamed node>"),
                            ATTR_N_KV_HEAD
                        ),
                        reason: format!(
                            "n_head {n_head} is not divisible by n_kv_head {n_kv_head}"
                        ),
                    });
                }
            } else if node.op_type == crate::ONNX_QWEN_DELTANET_OP_NAME
                && let (Some(num_key_heads), Some(num_value_heads)) = (
                    positive_value(crate::ATTR_NUM_KEY_HEADS),
                    positive_value(crate::ATTR_NUM_VALUE_HEADS),
                )
                && num_value_heads % num_key_heads != 0
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: format!(
                        "{}.{}",
                        diagnostic_subject(&node.name, "<unnamed node>"),
                        crate::ATTR_NUM_VALUE_HEADS
                    ),
                    reason: format!(
                        "num_value_heads {num_value_heads} is not divisible by num_key_heads {num_key_heads}"
                    ),
                });
            }
        }
    }
    for initializer in &graph.initializer {
        let expected = match initializer.name.as_str() {
            name if name == "tritium.packed" || name.ends_with(".packed") => Some(TENSOR_UINT8),
            name if name == "tritium.scales"
                || name == "attention.scale"
                || name.ends_with(".scales")
                || name.ends_with(".weight")
                || name.ends_with(".epsilon")
                || name.ends_with(".unit")
                || name.ends_with(".attention_scale")
                || name.ends_with(".attention_mask")
                || name.ends_with(".cos")
                || name.ends_with(".sin")
                || name.ends_with(".dt_bias")
                || name.ends_with(".a_log") =>
            {
                Some(TENSOR_FLOAT)
            }
            name if name.ends_with(".axes")
                || name.ends_with(".shape")
                || name.ends_with("_shape")
                || name.ends_with(".gqa_repeats")
                || name.ends_with(".first_start")
                || name.ends_with(".first_end")
                || name.ends_with(".second_start")
                || name.ends_with(".second_end")
                || name.ends_with(".tail_start")
                || name.ends_with(".tail_end")
                || name.ends_with(".steps") =>
            {
                Some(TENSOR_INT64)
            }
            _ => None,
        };
        let subject = format!("initializer {}", initializer.name);
        match expected {
            Some(expected) => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                Some(initializer.data_type),
                expected,
            ),
            None => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                Some(initializer.data_type),
            ),
        }
    }
    let is_deltanet_input = |name: &str| {
        graph.node.iter().any(|node| {
            node.domain == ONNX_DOMAIN
                && node.op_type == crate::ONNX_QWEN_DELTANET_OP_NAME
                && node.input.iter().any(|input| input == name)
        })
    };
    let is_deltanet_output = |name: &str| {
        graph.node.iter().any(|node| {
            node.domain == ONNX_DOMAIN
                && node.op_type == crate::ONNX_QWEN_DELTANET_OP_NAME
                && node.output.iter().any(|output| output == name)
        })
    };
    for input in &graph.input {
        let subject = format!("input {}", input.name);
        match input.name.as_str() {
            "tokens" | "shifted_tokens" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_INT64,
            ),
            "q" | "k_cache" | "v_cache" | "attention_mask" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_FLOAT,
            ),
            "hidden" | "target_hidden" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_FLOAT,
            ),
            name if is_deltanet_input(name) => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_FLOAT,
            ),
            name if name.starts_with("past_k.") || name.starts_with("past_v.") => {
                push_dtype_diagnostic(
                    &mut diagnostics,
                    subject,
                    tensor_elem_type(input),
                    TENSOR_FLOAT,
                )
            }
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
            ),
        }
    }
    for output in &graph.output {
        let subject = format!("output {}", output.name);
        match output.name.as_str() {
            "logits" | "context" | "next_hidden" | "mtp.logits" | "mtp.final_hidden" => {
                push_dtype_diagnostic(
                    &mut diagnostics,
                    subject,
                    tensor_elem_type(output),
                    TENSOR_FLOAT,
                )
            }
            name if is_deltanet_output(name) => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(output),
                TENSOR_FLOAT,
            ),
            name if name.starts_with("present_k.") || name.starts_with("present_v.") => {
                push_dtype_diagnostic(
                    &mut diagnostics,
                    subject,
                    tensor_elem_type(output),
                    TENSOR_FLOAT,
                )
            }
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(output),
            ),
        }
    }
    for value in &graph.value_info {
        let subject = format!("value_info {}", value.name);
        match value.name.as_str() {
            "hidden" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(value),
                TENSOR_FLOAT,
            ),
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(value),
            ),
        }
    }
    for entry in &protobuf.metadata_props {
        if let Some(subject) = entry.key.strip_prefix("tritium.coverage.unresolved.") {
            diagnostics.push(UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Coverage,
                subject: diagnostic_subject(subject, "<unnamed coverage item>"),
                reason: format!("coverage item is unresolved: {}", entry.value),
            });
        }
    }
    Ok(diagnostics)
}

fn supported_attributes(node: &NodeProto) -> &'static [&'static str] {
    match (node.domain.as_str(), node.op_type.as_str()) {
        (ONNX_DOMAIN, ONNX_OP_NAME | ONNX_EMBEDDING_OP_NAME) => &[ATTR_K, ATTR_FORMAT],
        (ONNX_DOMAIN, ONNX_KV_ATTENTION_OP_NAME) => {
            &[ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_HEAD_DIM, ATTR_PAST_TOKENS]
        }
        (ONNX_DOMAIN, crate::ONNX_QWEN_DELTANET_OP_NAME) => &[
            crate::ATTR_CONV_KERNEL_DIM,
            crate::ATTR_NUM_KEY_HEADS,
            crate::ATTR_NUM_VALUE_HEADS,
            crate::ATTR_KEY_HEAD_DIM,
            crate::ATTR_VALUE_HEAD_DIM,
        ],
        ("", "Transpose") => &["perm"],
        ("", "Softmax") => &["axis"],
        ("", "Concat") => &["axis"],
        ("", "ReduceMean") => &["keepdims"],
        (
            "",
            "MatMul" | "Mul" | "Add" | "Div" | "Sqrt" | "Sigmoid" | "Relu" | "Reshape" | "Tile"
            | "Identity" | "Slice" | "Neg",
        ) => &[],
        _ => &[],
    }
}

struct ConcatRole {
    axis: i64,
    label: &'static str,
}

fn concat_role(node: &NodeProto, graph: &GraphProto) -> Option<ConcatRole> {
    let producer = |value: &str| {
        graph
            .node
            .iter()
            .find(|candidate| candidate.output.iter().any(|output| output == value))
    };
    let rotary = node.input.len() == 2
        && producer(&node.input[0]).is_some_and(|candidate| candidate.op_type == "Neg")
        && producer(&node.input[1]).is_some_and(|candidate| candidate.op_type == "Slice");
    if rotary {
        return Some(ConcatRole {
            axis: -1,
            label: "RoPE",
        });
    }
    let rotary_prefix = node.input.len() == 2
        && node
            .input
            .iter()
            .all(|input| producer(input).is_some_and(|candidate| candidate.op_type == "Slice"));
    let rotary_tail = node.input.len() == 2
        && producer(&node.input[0]).is_some_and(|candidate| candidate.op_type == "Add")
        && producer(&node.input[1]).is_some_and(|candidate| candidate.op_type == "Slice");
    if rotary_prefix || rotary_tail {
        return Some(ConcatRole {
            axis: -1,
            label: "RoPE",
        });
    }
    let mtp_fusion = node.input.len() == 2
        && node.output.len() == 1
        && node
            .input
            .iter()
            .all(|input| producer(input).is_some_and(|candidate| candidate.op_type == "Mul"))
        && graph.node.iter().any(|candidate| {
            candidate.domain == ONNX_DOMAIN
                && candidate.op_type == ONNX_OP_NAME
                && candidate.input.first() == node.output.first()
        });
    if mtp_fusion {
        return Some(ConcatRole {
            axis: 1,
            label: "MTP fusion",
        });
    }
    let cache = node.input.len() == 2
        && node.output.len() == 1
        && graph.input.iter().any(|input| input.name == node.input[0])
        && graph
            .output
            .iter()
            .any(|output| output.name == node.output[0]);
    cache.then_some(ConcatRole {
        axis: 0,
        label: "KV cache",
    })
}

fn expected_attribute_kind(name: &str) -> i32 {
    if name == "perm" {
        ATTRIBUTE_INTS
    } else {
        ATTRIBUTE_INT
    }
}

fn valid_rank_three_permutation(values: &[i64]) -> bool {
    values.len() == 3 && values.iter().all(|axis| (0..3).contains(axis)) && {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted == [0, 1, 2]
    }
}

fn node_supported(
    node: &NodeProto,
    tritium_opset: Option<i64>,
    standard_opset: Option<i64>,
) -> bool {
    match (node.domain.as_str(), node.op_type.as_str()) {
        (ONNX_DOMAIN, ONNX_OP_NAME | ONNX_EMBEDDING_OP_NAME) => {
            matches!(tritium_opset, Some(1 | 2))
        }
        (ONNX_DOMAIN, ONNX_KV_ATTENTION_OP_NAME) => tritium_opset == Some(2),
        (ONNX_DOMAIN, crate::ONNX_QWEN_DELTANET_OP_NAME) => tritium_opset == Some(2),
        (
            "",
            "Transpose" | "MatMul" | "Mul" | "Add" | "Div" | "Sqrt" | "Sigmoid" | "Relu"
            | "Reshape" | "Tile" | "Identity" | "Slice" | "Neg" | "Concat" | "ReduceMean"
            | "Softmax",
        ) => standard_opset == Some(ONNX_OPSET),
        _ => false,
    }
}

fn diagnostic_subject(value: &str, fallback: &str) -> String {
    if value.is_empty() {
        fallback.to_owned()
    } else {
        value.to_owned()
    }
}

fn tensor_elem_type(value: &ValueInfoProto) -> Option<i32> {
    value
        .r#type
        .as_ref()
        .and_then(|kind| kind.tensor_type.as_ref())
        .map(|tensor| tensor.elem_type)
}

fn push_dtype_diagnostic(
    diagnostics: &mut Vec<UnsupportedGraphDiagnostic>,
    subject: String,
    actual: Option<i32>,
    expected: i32,
) {
    if actual != Some(expected) {
        diagnostics.push(UnsupportedGraphDiagnostic {
            kind: UnsupportedGraphItemKind::Dtype,
            subject,
            reason: actual.map_or_else(
                || format!("expected ONNX dtype {expected}, got missing type"),
                |actual| format!("expected ONNX dtype {expected}, got {actual}"),
            ),
        });
    }
}

fn push_uncontracted_dtype_diagnostic(
    diagnostics: &mut Vec<UnsupportedGraphDiagnostic>,
    subject: String,
    actual: Option<i32>,
) {
    diagnostics.push(UnsupportedGraphDiagnostic {
        kind: UnsupportedGraphItemKind::Dtype,
        subject,
        reason: actual.map_or_else(
            || "missing type has no supported contract for this tensor".to_owned(),
            |actual| format!("dtype {actual} has no supported contract for this tensor"),
        ),
    });
}

/// Encode a tied packed embedding/head graph as a deterministic ONNX model.
///
/// # Errors
/// [`OnnxModelError`] if dimensions, identities, scales, format, or packed byte
/// count violate the frozen `com.tritium` opset-1 contract.
pub fn encode_tied_embedding_head(
    model: TiedEmbeddingHeadModel<'_>,
) -> Result<Vec<u8>, OnnxModelError> {
    validate(&model)?;
    let scale_bytes = scale_bytes(model.scales);
    let packed_bytes = as_i64(model.packed.len(), "packed byte count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    encode_model(
        model,
        vec![
            inline_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_bytes],
                model.packed.to_vec(),
            ),
            inline_tensor("tritium.scales", TENSOR_FLOAT, vec![vocab], scale_bytes),
        ],
        Vec::new(),
        "1",
    )
}

/// Encode a schema-v2 tied embedding/head graph with complete artifact identity.
///
/// # Errors
/// [`OnnxModelError`] if graph geometry, payload, or any identity is invalid.
pub fn encode_tied_embedding_head_v2(
    model: TiedEmbeddingHeadModelV2<'_>,
) -> Result<Vec<u8>, OnnxModelError> {
    validate_v2(&model)?;
    let legacy = model.legacy();
    let scale_bytes = scale_bytes(legacy.scales);
    let packed_bytes = as_i64(legacy.packed.len(), "packed byte count")?;
    let vocab = as_i64(legacy.vocab, "vocabulary")?;
    encode_model(
        legacy,
        vec![
            inline_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_bytes],
                legacy.packed.to_vec(),
            ),
            inline_tensor("tritium.scales", TENSOR_FLOAT, vec![vocab], scale_bytes),
        ],
        identity_metadata_v2(model.identity),
        "2",
    )
}

/// Encode a packed decoder-only causal LM as a deterministic ONNX model.
///
/// Packed embedding/projection initializers remain compressed and execute via
/// `com.tritium` opset 1. RMSNorm, GQA expansion, causal attention, residuals,
/// SwiGLU, cache concatenation, and cache outputs use standard ONNX opset 21.
///
/// # Errors
/// [`OnnxModelError`] if model geometry, packed payloads, preserved vectors,
/// cache sizes, epsilon, or artifact identity violate the causal-LM contract.
pub fn encode_causal_lm(model: CausalLmModel<'_>) -> Result<Vec<u8>, OnnxModelError> {
    validate_causal_lm(&model)?;
    validate_causal_initializer_budget(&model)?;
    let (protobuf, weights) = build_causal_lm_graph(model, false)?;
    debug_assert!(weights.is_none());
    let encoded = protobuf.encode_to_vec();
    if encoded.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(encoded)
}

/// Encode one complete Qwen Gated DeltaNet decoder layer.
///
/// Packed QKV/Z/beta/decay/output/FFN projections execute through Tritium
/// mpGEMM nodes. The opset-2 DeltaNet node consumes explicit prior state and
/// publishes both next states; residual and SwiGLU paths remain standard ONNX.
/// Qwen zero-centered RMSNorm semantics are fixed into the graph.
///
/// # Errors
/// [`OnnxModelError`] if geometry, packed payloads, preserved parameters,
/// identities, or bounded inline initializer size violate the contract.
pub fn encode_qwen_deltanet_layer(
    model: QwenDeltaNetLayerModel<'_>,
) -> Result<Vec<u8>, OnnxModelError> {
    validate_qwen_deltanet_layer(&model)?;
    validate_qwen_deltanet_initializer_budget(&model)?;
    let tokens = as_i64(model.tokens, "Qwen DeltaNet token count")?;
    let hidden = as_i64(model.hidden, "Qwen DeltaNet hidden width")?;
    let mut graph = CausalGraphBuilder::default();
    let mut inputs = vec![tensor_value("hidden", TENSOR_FLOAT, &[tokens, hidden])];
    let mut state_outputs = Vec::with_capacity(2);
    let layer_output = graph.delta_net_block(
        0,
        "hidden",
        model.layer,
        DeltaNetGraphContext {
            geometry: model.geometry,
            state_scope: DeltaStateScope::Standalone,
            epsilon: model.rms_epsilon,
        },
        &mut inputs,
        &mut state_outputs,
    )?;
    graph.standard(
        "layer.0.output_identity".to_owned(),
        "Identity",
        &[&layer_output],
        &["next_hidden"],
        Vec::new(),
    );
    if let Some(error) = graph.failure {
        return Err(error);
    }
    let mut outputs = vec![tensor_value("next_hidden", TENSOR_FLOAT, &[tokens, hidden])];
    outputs.extend(state_outputs);
    let protobuf = ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: 2,
        graph: Some(GraphProto {
            node: graph.nodes,
            name: "tritium.qwen_deltanet_layer".to_owned(),
            initializer: graph.initializers,
            input: inputs,
            output: outputs,
            value_info: Vec::new(),
        }),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: 2,
            },
        ],
        metadata_props: vec![
            metadata("tritium.schema_version", "2"),
            metadata("tritium.graph_kind", "qwen-deltanet-layer"),
            metadata("tritium.source_model_id", model.identity.source_model_id),
            metadata("tritium.tokenizer_id", model.identity.tokenizer_id),
            metadata("tritium.recipe_id", model.identity.recipe_id),
            metadata("tritium.build_id", model.identity.tritium_build_id),
            metadata("tritium.package_id", model.identity.package_id),
            metadata(
                "tritium.coverage.converted_id",
                model.identity.converted_coverage_id,
            ),
            metadata(
                "tritium.coverage.deferred_id",
                model.identity.deferred_coverage_id,
            ),
            metadata("tritium.tokens", &model.tokens.to_string()),
            metadata("tritium.hidden", &model.hidden.to_string()),
            metadata("tritium.layers", "1"),
            metadata("tritium.rms_norm_weight_semantics", "zero-centered-offset"),
        ],
    };
    let encoded = protobuf.encode_to_vec();
    if encoded.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(encoded)
}

/// Encode a packed heterogeneous Qwen causal language model.
///
/// # Errors
/// [`OnnxModelError`] if schedule or model contract is invalid.
pub fn encode_qwen_causal_lm(model: QwenCausalLmModel<'_>) -> Result<Vec<u8>, OnnxModelError> {
    validate_qwen_causal_lm(&model)?;
    build_qwen_causal_lm_graph(model, CausalGraphBuilder::counting())?;
    let (protobuf, _) = build_qwen_causal_lm_graph(model, CausalGraphBuilder::default())?;
    let encoded = protobuf.encode_to_vec();
    if encoded.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(encoded)
}

/// Encode packed Qwen MTP fusion, decoder, cache, final hidden rows, and logits.
///
/// Inputs are `shifted_tokens` and `target_hidden`, followed by `past_k.0` and
/// `past_v.0` for cached decode. Token shifting/alignment remains an explicit
/// caller contract instead of hidden graph mutation.
///
/// # Errors
/// [`OnnxModelError`] if model geometry, initializer admission, or encoding fails.
pub fn encode_qwen35_mtp(model: Qwen35MtpModel<'_>) -> Result<Vec<u8>, OnnxModelError> {
    validate_qwen35_mtp_model(&model)?;
    build_qwen35_mtp_graph(model, CausalGraphBuilder::counting())?;
    let (protobuf, _) = build_qwen35_mtp_graph(model, CausalGraphBuilder::default())?;
    let encoded = protobuf.encode_to_vec();
    if encoded.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(encoded)
}

/// Encode heterogeneous Qwen language graph with authenticated external data.
///
/// # Errors
/// [`OnnxModelError`] if model validation, external allocation, or protobuf
/// encoding fails.
pub fn encode_external_qwen_causal_lm(
    model: QwenCausalLmModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_qwen_causal_lm(&model)?;
    let (protobuf, weights) = build_qwen_causal_lm_graph(model, CausalGraphBuilder::external())?;
    finish_external_model(protobuf, weights)
}

/// Encode Qwen MTP graph with authenticated external data.
///
/// # Errors
/// [`OnnxModelError`] if model validation, external allocation, or protobuf
/// encoding fails.
pub fn encode_external_qwen35_mtp(
    model: Qwen35MtpModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_qwen35_mtp_model(&model)?;
    let (protobuf, weights) = build_qwen35_mtp_graph(model, CausalGraphBuilder::external())?;
    finish_external_model(protobuf, weights)
}

/// Encode matching Qwen language and MTP graphs as one three-file bundle.
///
/// # Errors
/// [`OnnxModelError`] if graph identities, geometry, shared embedding/head, or
/// external-data encoding differ from bundle contract.
pub fn encode_external_qwen35_bundle(
    language: QwenCausalLmModel<'_>,
    mtp: Qwen35MtpModel<'_>,
) -> Result<ExternalQwen35Bundle, OnnxModelError> {
    validate_qwen35_bundle_pair(&language, &mtp)?;
    let (language_protobuf, language_weights) =
        build_qwen_causal_lm_graph(language, CausalGraphBuilder::external())?;
    let language_weights = language_weights.ok_or_else(|| {
        OnnxModelError::InvalidModel("Qwen language bundle used inline storage".to_owned())
    })?;
    let aliases = qwen_shared_external_aliases(&language_protobuf)?;
    let (mtp_protobuf, weights) = build_qwen35_mtp_graph(
        mtp,
        CausalGraphBuilder::external_reusing(language_weights, aliases),
    )?;
    let weights_bytes = weights.ok_or_else(|| {
        OnnxModelError::InvalidModel("Qwen MTP bundle used inline storage".to_owned())
    })?;
    let weights_blake3 = *blake3::hash(&weights_bytes).as_bytes();
    Ok(ExternalQwen35Bundle {
        language_model_bytes: bind_external_metadata(language_protobuf, &weights_bytes)?,
        mtp_model_bytes: bind_external_metadata(mtp_protobuf, &weights_bytes)?,
        weights_bytes,
        weights_blake3,
    })
}

fn qwen_shared_external_aliases(
    protobuf: &ModelProto,
) -> Result<BTreeMap<String, TensorProto>, OnnxModelError> {
    let graph = protobuf
        .graph
        .as_ref()
        .ok_or_else(|| OnnxModelError::InvalidModel("Qwen language graph is absent".to_owned()))?;
    let mut aliases = BTreeMap::new();
    for name in QWEN_SHARED_EXTERNAL_INITIALIZERS {
        let tensor = graph
            .initializer
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| {
                OnnxModelError::InvalidModel(format!(
                    "Qwen language graph lacks shared initializer {name}"
                ))
            })?;
        aliases.insert(name.to_owned(), tensor.clone());
    }
    Ok(aliases)
}

/// Encode a causal LM with weight/value initializers in authenticated `weights.bin`.
///
/// Initializers retain packed ternary representation; no dense weight shadow is
/// emitted. Shape-driving `int64` constants remain inline for ONNX shape
/// inference. External ranges are 64-byte aligned and graph metadata binds
/// exact byte length plus BLAKE3 digest.
///
/// # Errors
/// [`OnnxModelError`] if model validation, range arithmetic, allocation, or
/// protobuf bounds fail.
pub fn encode_external_causal_lm(
    model: CausalLmModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_causal_lm(&model)?;
    let (protobuf, weights) = build_causal_lm_graph(model, true)?;
    finish_external_model(protobuf, weights)
}

fn finish_external_model(
    protobuf: ModelProto,
    weights: Option<Vec<u8>>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    let weights_bytes = weights.ok_or_else(|| {
        OnnxModelError::InvalidModel("external graph builder returned inline storage".to_owned())
    })?;
    let weights_blake3 = *blake3::hash(&weights_bytes).as_bytes();
    let model_bytes = bind_external_metadata(protobuf, &weights_bytes)?;
    Ok(ExternalOnnxModel {
        model_bytes,
        weights_bytes,
        weights_blake3,
    })
}

fn bind_external_metadata(
    mut protobuf: ModelProto,
    weights_bytes: &[u8],
) -> Result<Vec<u8>, OnnxModelError> {
    let digest = blake3::hash(weights_bytes).to_hex().to_string();
    protobuf.metadata_props.extend([
        metadata("tritium.external_data.file", EXTERNAL_WEIGHTS_FILE),
        metadata(
            "tritium.external_data.bytes",
            &weights_bytes.len().to_string(),
        ),
        metadata("tritium.external_data.blake3", &digest),
    ]);
    let model_bytes = protobuf.encode_to_vec();
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(model_bytes)
}

/// Verify an external causal-LM model before ONNX Runtime session creation.
///
/// # Security
/// `admitted` must come from an independently authenticated package manifest.
/// Never compute it from `model_bytes` or `weights_bytes`: doing so removes the
/// trust root and turns integrity verification into a self-consistency check.
///
/// # Errors
/// [`OnnxModelError`] for malformed metadata, unsupported graph semantics,
/// noncanonical/overlapping ranges, nonzero padding, or manifest digest/length
/// mismatch.
pub fn verify_external_causal_lm(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    admitted: AdmittedExternalCausalLmDigests,
) -> Result<VerifiedExternalCausalLmModel, OnnxModelError> {
    Ok(verify_external_decoder_graph(
        model_bytes,
        weights_bytes,
        admitted,
        "causal-lm",
        ExternalRangePolicy::Exclusive,
    )?
    .receipt)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExternalRangePolicy {
    Exclusive,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifiedExternalInitializer {
    name: String,
    data_type: i32,
    dimensions: Vec<i64>,
    offset: usize,
    length: usize,
}

struct VerifiedExternalDecoderGraph {
    receipt: VerifiedExternalCausalLmModel,
    metadata: BTreeMap<String, String>,
    external_initializers: Vec<VerifiedExternalInitializer>,
}

fn verify_external_decoder_graph(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    admitted: AdmittedExternalCausalLmDigests,
    expected_graph_kind: &str,
    range_policy: ExternalRangePolicy,
) -> Result<VerifiedExternalDecoderGraph, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let actual_model_hash = blake3::hash(model_bytes);
    if actual_model_hash.as_bytes() != &admitted.model_blake3 {
        return Err(OnnxModelError::ExternalDataMismatch(
            "model BLAKE3 differs from admitted package manifest".to_owned(),
        ));
    }
    let actual_hash = blake3::hash(weights_bytes);
    if actual_hash.as_bytes() != &admitted.weights_blake3 {
        return Err(OnnxModelError::ExternalDataMismatch(
            "weights BLAKE3 differs from admitted package manifest".to_owned(),
        ));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let mut metadata = BTreeMap::new();
    for entry in &protobuf.metadata_props {
        if metadata
            .insert(entry.key.clone(), entry.value.clone())
            .is_some()
        {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate metadata key {}",
                entry.key
            )));
        }
    }
    require_metadata(&metadata, "tritium.schema_version", "2")?;
    require_metadata(&metadata, "tritium.graph_kind", expected_graph_kind)?;
    require_metadata(
        &metadata,
        "tritium.external_data.file",
        EXTERNAL_WEIGHTS_FILE,
    )?;
    let declared_weights = parse_usize(&metadata, "tritium.external_data.bytes")?;
    if declared_weights != weights_bytes.len() {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "byte length {} differs from declared {declared_weights}",
            weights_bytes.len()
        )));
    }
    if metadata_value(&metadata, "tritium.external_data.blake3")?
        != actual_hash.to_hex().to_string()
    {
        return Err(OnnxModelError::ExternalDataMismatch(
            "BLAKE3 digest differs from model metadata".to_owned(),
        ));
    }
    let diagnostics = diagnose_unsupported_graph(model_bytes)?;
    if !diagnostics.is_empty() {
        return Err(OnnxModelError::InvalidModel(format!(
            "graph has {} unsupported items",
            diagnostics.len()
        )));
    }
    let graph = protobuf
        .graph
        .as_ref()
        .ok_or_else(|| OnnxModelError::InvalidModel("model has no graph".to_owned()))?;
    let mut cursor = 0_usize;
    let mut names = BTreeMap::new();
    let mut external_initializers = Vec::new();
    for initializer in &graph.initializer {
        if names.insert(initializer.name.as_str(), ()).is_some() {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate initializer {}",
                initializer.name
            )));
        }
        let element_bytes = match initializer.data_type {
            TENSOR_UINT8 => 1,
            TENSOR_FLOAT => core::mem::size_of::<f32>(),
            TENSOR_INT64 => core::mem::size_of::<i64>(),
            other => {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} has unsupported dtype {other}",
                    initializer.name
                )));
            }
        };
        let expected_length =
            initializer
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dimension| {
                    let dimension = usize::try_from(dimension).map_err(|_| {
                        OnnxModelError::ExternalDataMismatch(format!(
                            "initializer {} has negative dimension",
                            initializer.name
                        ))
                    })?;
                    bytes
                        .checked_mul(dimension)
                        .ok_or(OnnxModelError::ShapeOverflow(
                            "external initializer byte count",
                        ))
                })?;
        if initializer.data_type == TENSOR_INT64 {
            if initializer.data_location != 0
                || !initializer.external_data.is_empty()
                || initializer.raw_data.len() != expected_length
            {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "shape initializer {} is not canonical inline data",
                    initializer.name
                )));
            }
            continue;
        }
        let (offset, length) = encoded_external_range(initializer)?;
        if length != expected_length {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} length {length} differs from shape-derived {expected_length}",
                initializer.name
            )));
        }
        if offset % EXTERNAL_ALIGNMENT != 0 {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} offset {offset} is not {EXTERNAL_ALIGNMENT}-byte aligned",
                initializer.name
            )));
        }
        let end = offset
            .checked_add(length)
            .ok_or(OnnxModelError::ShapeOverflow("external initializer range"))?;
        if end > weights_bytes.len() {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} range exceeds weights.bin",
                initializer.name
            )));
        }
        if range_policy == ExternalRangePolicy::Exclusive {
            let expected_offset = align_up(cursor, EXTERNAL_ALIGNMENT)?;
            if offset != expected_offset {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} offset {offset} is not canonical {expected_offset}",
                    initializer.name
                )));
            }
            if weights_bytes[cursor..offset].iter().any(|&byte| byte != 0) {
                return Err(OnnxModelError::ExternalDataMismatch(
                    "alignment padding is not zero".to_owned(),
                ));
            }
            cursor = end;
        }
        external_initializers.push(VerifiedExternalInitializer {
            name: initializer.name.clone(),
            data_type: initializer.data_type,
            dimensions: initializer.dims.clone(),
            offset,
            length,
        });
    }
    if range_policy == ExternalRangePolicy::Exclusive && cursor != weights_bytes.len() {
        return Err(OnnxModelError::ExternalDataMismatch(
            "weights.bin has unreferenced trailing bytes".to_owned(),
        ));
    }
    let identity = VerifiedOnnxArtifactIdentityV2 {
        source_model_id: metadata_value(&metadata, "tritium.source_model_id")?.to_owned(),
        tokenizer_id: metadata_value(&metadata, "tritium.tokenizer_id")?.to_owned(),
        recipe_id: metadata_value(&metadata, "tritium.recipe_id")?.to_owned(),
        tritium_build_id: metadata_value(&metadata, "tritium.build_id")?.to_owned(),
        package_id: metadata_value(&metadata, "tritium.package_id")?.to_owned(),
        converted_coverage_id: metadata_value(&metadata, "tritium.coverage.converted_id")?
            .to_owned(),
        deferred_coverage_id: metadata_value(&metadata, "tritium.coverage.deferred_id")?.to_owned(),
    };
    Ok(VerifiedExternalDecoderGraph {
        receipt: VerifiedExternalCausalLmModel {
            model_blake3: *actual_model_hash.as_bytes(),
            weights_blake3: *actual_hash.as_bytes(),
            weights_bytes: weights_bytes.len(),
            tokens: parse_usize(&metadata, "tritium.tokens")?,
            past_tokens: parse_usize(&metadata, "tritium.past_tokens")?,
            layers: parse_usize(&metadata, "tritium.layers")?,
            identity,
        },
        metadata,
        external_initializers,
    })
}

/// Verify one externally admitted Qwen language-plus-MTP three-file bundle.
///
/// # Security
/// `admitted` must come from one independently authenticated package manifest,
/// never from candidate file metadata or the encoder's returned digests.
///
/// # Errors
/// [`OnnxModelError`] if any digest, external range, graph role, identity, or
/// shared execution geometry differs.
pub fn verify_external_qwen35_bundle(
    files: ExternalQwen35BundleFiles<'_>,
    admitted: AdmittedExternalQwen35BundleDigests,
) -> Result<VerifiedExternalQwen35Bundle, OnnxModelError> {
    let require_digest = |bytes: &[u8], expected: [u8; 32], label: &str| {
        if blake3::hash(bytes).as_bytes() != &expected {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "{label} BLAKE3 differs from admitted package manifest"
            )));
        }
        Ok(())
    };
    require_digest(
        files.language_model_bytes,
        admitted.language_model_blake3,
        "language model",
    )?;
    require_digest(
        files.mtp_model_bytes,
        admitted.mtp_model_blake3,
        "MTP model",
    )?;
    require_digest(
        files.weights_bytes,
        admitted.weights_blake3,
        "bundle weights",
    )?;
    let language = verify_external_decoder_graph(
        files.language_model_bytes,
        files.weights_bytes,
        AdmittedExternalCausalLmDigests {
            model_blake3: admitted.language_model_blake3,
            weights_blake3: admitted.weights_blake3,
        },
        "qwen-causal-lm",
        ExternalRangePolicy::Shared,
    )?;
    let mtp = verify_external_decoder_graph(
        files.mtp_model_bytes,
        files.weights_bytes,
        AdmittedExternalCausalLmDigests {
            model_blake3: admitted.mtp_model_blake3,
            weights_blake3: admitted.weights_blake3,
        },
        "qwen35-mtp",
        ExternalRangePolicy::Shared,
    )?;
    verify_shared_qwen_ranges(&language, &mtp, files.weights_bytes)?;
    require_metadata(
        &language.metadata,
        "tritium.rms_norm_weight_semantics",
        "zero-centered-offset",
    )?;
    require_metadata(&language.metadata, "tritium.tied_embedding_head", "false")?;
    require_metadata(
        &mtp.metadata,
        "tritium.input_alignment",
        "caller-shifted-target-aligned",
    )?;
    require_metadata(
        &mtp.metadata,
        "tritium.rms_norm_weight_semantics",
        "zero-centered-offset",
    )?;
    require_metadata(&mtp.metadata, "tritium.tied_embedding_head", "false")?;
    if mtp.receipt.layers != 1 {
        return Err(OnnxModelError::InvalidModel(format!(
            "Qwen MTP bundle graph must contain one layer, got {}",
            mtp.receipt.layers
        )));
    }
    parse_positive_f32(&language.metadata, "tritium.rms_epsilon")?;
    parse_positive_f32(&mtp.metadata, "tritium.rms_epsilon")?;
    let interval = parse_usize(&language.metadata, "tritium.full_attention_interval")?;
    let schedule = metadata_value(&language.metadata, "tritium.layer_schedule")?
        .split(',')
        .collect::<Vec<_>>();
    if interval == 0
        || !language.receipt.layers.is_multiple_of(interval)
        || schedule.len() != language.receipt.layers
        || schedule.iter().enumerate().any(|(index, layer)| {
            let expected = if (index + 1).is_multiple_of(interval) {
                "full_attention"
            } else {
                "linear_attention"
            };
            *layer != expected
        })
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language layer schedule metadata is not canonical".to_owned(),
        ));
    }
    if language.receipt.identity != mtp.receipt.identity
        || language.receipt.tokens != mtp.receipt.tokens
        || language.receipt.past_tokens != mtp.receipt.past_tokens
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language and MTP admission identity or execution shape differs".to_owned(),
        ));
    }
    for key in [
        "tritium.hidden",
        "tritium.vocab",
        "tritium.n_head",
        "tritium.n_kv_head",
        "tritium.head_dim",
        "tritium.rotary_dim",
        "tritium.rope_theta",
        "tritium.rms_epsilon",
    ] {
        if metadata_value(&language.metadata, key)? != metadata_value(&mtp.metadata, key)? {
            return Err(OnnxModelError::InvalidModel(format!(
                "Qwen language and MTP metadata {key} differs"
            )));
        }
    }
    Ok(VerifiedExternalQwen35Bundle {
        language: language.receipt,
        mtp: mtp.receipt,
    })
}

fn verify_shared_qwen_ranges(
    language: &VerifiedExternalDecoderGraph,
    mtp: &VerifiedExternalDecoderGraph,
    weights: &[u8],
) -> Result<(), OnnxModelError> {
    let shared_names = QWEN_SHARED_EXTERNAL_INITIALIZERS;
    fn find<'a>(
        graph: &'a VerifiedExternalDecoderGraph,
        name: &str,
    ) -> Result<&'a VerifiedExternalInitializer, OnnxModelError> {
        graph
            .external_initializers
            .iter()
            .find(|initializer| initializer.name == name)
            .ok_or_else(|| {
                OnnxModelError::ExternalDataMismatch(format!(
                    "Qwen bundle graph lacks shared initializer {name}"
                ))
            })
    }
    for name in shared_names {
        if find(language, name)? != find(mtp, name)? {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "Qwen shared initializer {name} has different descriptors"
            )));
        }
    }
    let mut ranges = language
        .external_initializers
        .iter()
        .chain(&mtp.external_initializers)
        .cloned()
        .collect::<Vec<_>>();
    ranges.sort_by(|left, right| {
        (left.offset, left.length, left.name.as_str()).cmp(&(
            right.offset,
            right.length,
            right.name.as_str(),
        ))
    });
    let mut cursor = 0_usize;
    let mut index = 0_usize;
    while index < ranges.len() {
        let range = &ranges[index];
        let expected_offset = align_up(cursor, EXTERNAL_ALIGNMENT)?;
        if range.offset != expected_offset {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "shared initializer {} offset {} is not canonical {expected_offset}",
                range.name, range.offset
            )));
        }
        if weights[cursor..range.offset].iter().any(|&byte| byte != 0) {
            return Err(OnnxModelError::ExternalDataMismatch(
                "shared weights alignment padding is not zero".to_owned(),
            ));
        }
        let mut next = index + 1;
        while next < ranges.len() && ranges[next].offset == range.offset {
            if ranges[next].length != range.length
                || ranges[next].name != range.name
                || !shared_names.contains(&range.name.as_str())
            {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} has noncanonical overlapping shared range",
                    ranges[next].name
                )));
            }
            next += 1;
        }
        if next - index > 2 {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} shared range is referenced more than twice",
                range.name
            )));
        }
        cursor = range
            .offset
            .checked_add(range.length)
            .ok_or(OnnxModelError::ShapeOverflow(
                "shared external initializer range",
            ))?;
        index = next;
    }
    if cursor != weights.len() {
        return Err(OnnxModelError::ExternalDataMismatch(
            "shared weights.bin has unreferenced trailing bytes".to_owned(),
        ));
    }
    Ok(())
}

/// Encode a tied packed embedding/head graph with 64-byte-aligned initializers
/// in the fixed relative file `weights.bin`.
///
/// The model metadata binds the external filename, exact byte length and BLAKE3
/// digest. A loader must verify those fields before creating an ORT session.
///
/// # Errors
/// [`OnnxModelError`] if the graph contract is invalid or external offsets and
/// lengths overflow ONNX `int64`/host `usize`.
pub fn encode_external_tied_embedding_head(
    model: TiedEmbeddingHeadModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate(&model)?;
    encode_external_model(model, Vec::new(), "1")
}

/// Encode a schema-v2 tied embedding/head graph with content-bound external data.
///
/// # Errors
/// [`OnnxModelError`] if graph geometry, payload, external layout, or any
/// identity is invalid.
pub fn encode_external_tied_embedding_head_v2(
    model: TiedEmbeddingHeadModelV2<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_v2(&model)?;
    encode_external_model(model.legacy(), identity_metadata_v2(model.identity), "2")
}

fn encode_external_model(
    model: TiedEmbeddingHeadModel<'_>,
    mut identity_metadata: Vec<StringStringEntryProto>,
    schema_version: &'static str,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    let scale_bytes = scale_bytes(model.scales);
    let scale_offset = align_up(model.packed.len(), EXTERNAL_ALIGNMENT)?;
    let weights_len = scale_offset
        .checked_add(scale_bytes.len())
        .ok_or(OnnxModelError::ShapeOverflow("external weight byte count"))?;
    let mut weights_bytes = Vec::new();
    weights_bytes
        .try_reserve_exact(weights_len)
        .map_err(|_| OnnxModelError::ShapeOverflow("external weight allocation"))?;
    weights_bytes.extend_from_slice(model.packed);
    weights_bytes.resize(scale_offset, 0);
    weights_bytes.extend_from_slice(&scale_bytes);
    let weights_blake3 = *blake3::hash(&weights_bytes).as_bytes();
    let digest_hex = blake3::Hash::from_bytes(weights_blake3)
        .to_hex()
        .to_string();
    let packed_len = as_i64(model.packed.len(), "packed byte count")?;
    let scale_offset_i64 = as_i64(scale_offset, "external scale offset")?;
    let scale_len = as_i64(scale_bytes.len(), "external scale byte count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    let model_bytes = encode_model(
        model,
        vec![
            external_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_len],
                0,
                packed_len,
            ),
            external_tensor(
                "tritium.scales",
                TENSOR_FLOAT,
                vec![vocab],
                scale_offset_i64,
                scale_len,
            ),
        ],
        {
            identity_metadata.extend([
                metadata("tritium.external_data.file", EXTERNAL_WEIGHTS_FILE),
                metadata("tritium.external_data.bytes", &weights_len.to_string()),
                metadata("tritium.external_data.blake3", &digest_hex),
            ]);
            identity_metadata
        },
        schema_version,
    )?;
    Ok(ExternalOnnxModel {
        model_bytes,
        weights_bytes,
        weights_blake3,
    })
}

/// Verify a serialized external-data graph before it is handed to ONNX Runtime.
///
/// Verification is fail-closed: metadata keys must be unique, the fixed
/// filename and geometry bindings must be canonical, length and BLAKE3 must
/// match, padding must be zero, scales and packed rows must validate, and a
/// deterministic re-encode must reproduce both files byte-for-byte.
///
/// # Errors
/// [`OnnxModelError`] for malformed protobuf/metadata, a digest or length
/// mismatch, invalid packed/scales data, or any non-canonical graph mutation.
pub fn verify_external_tied_embedding_head(
    model_bytes: &[u8],
    weights_bytes: &[u8],
) -> Result<VerifiedExternalOnnxModel, OnnxModelError> {
    let verified = verify_external_parts(model_bytes, weights_bytes, "1")?;
    let source_model_id = metadata_value(&verified.metadata, "tritium.source_model_id")?.to_owned();
    let recipe_id = metadata_value(&verified.metadata, "tritium.recipe_id")?.to_owned();
    let package_id = metadata_value(&verified.metadata, "tritium.package_id")?.to_owned();
    let specification = TiedEmbeddingHeadModel {
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        packed: &weights_bytes[..verified.packed_len],
        scales: &verified.scales,
        format: verified.format,
        source_model_id: &source_model_id,
        recipe_id: &recipe_id,
        package_id: &package_id,
    };
    let expected = encode_external_tied_embedding_head(specification)?;
    verify_canonical_external(&expected, model_bytes, weights_bytes)?;
    Ok(VerifiedExternalOnnxModel {
        model_blake3: *blake3::hash(model_bytes).as_bytes(),
        weights_blake3: verified.weights_blake3,
        weights_bytes: weights_bytes.len(),
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        source_model_id,
        recipe_id,
        package_id,
    })
}

/// Verify a schema-v2 external-data graph and its complete artifact identity.
///
/// Schema-v1 remains available through [`verify_external_tied_embedding_head`];
/// callers choose the expected contract explicitly, so neither version can be
/// silently interpreted as the other.
///
/// # Errors
/// [`OnnxModelError`] for malformed protobuf/metadata, an identity, digest or
/// length mismatch, invalid packed/scales data, or non-canonical mutation.
pub fn verify_external_tied_embedding_head_v2(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    expected_identity: OnnxArtifactIdentityV2<'_>,
) -> Result<VerifiedExternalOnnxModelV2, OnnxModelError> {
    validate_identity_v2(expected_identity)?;
    let verified = verify_external_parts(model_bytes, weights_bytes, "2")?;
    let identity = VerifiedOnnxArtifactIdentityV2 {
        source_model_id: metadata_value(&verified.metadata, "tritium.source_model_id")?.to_owned(),
        tokenizer_id: metadata_value(&verified.metadata, "tritium.tokenizer_id")?.to_owned(),
        recipe_id: metadata_value(&verified.metadata, "tritium.recipe_id")?.to_owned(),
        tritium_build_id: metadata_value(&verified.metadata, "tritium.build_id")?.to_owned(),
        package_id: metadata_value(&verified.metadata, "tritium.package_id")?.to_owned(),
        converted_coverage_id: metadata_value(&verified.metadata, "tritium.coverage.converted_id")?
            .to_owned(),
        deferred_coverage_id: metadata_value(&verified.metadata, "tritium.coverage.deferred_id")?
            .to_owned(),
    };
    for (name, actual, expected) in [
        (
            "source_model_id",
            identity.source_model_id.as_str(),
            expected_identity.source_model_id,
        ),
        (
            "tokenizer_id",
            identity.tokenizer_id.as_str(),
            expected_identity.tokenizer_id,
        ),
        (
            "recipe_id",
            identity.recipe_id.as_str(),
            expected_identity.recipe_id,
        ),
        (
            "tritium_build_id",
            identity.tritium_build_id.as_str(),
            expected_identity.tritium_build_id,
        ),
        (
            "package_id",
            identity.package_id.as_str(),
            expected_identity.package_id,
        ),
        (
            "converted_coverage_id",
            identity.converted_coverage_id.as_str(),
            expected_identity.converted_coverage_id,
        ),
        (
            "deferred_coverage_id",
            identity.deferred_coverage_id.as_str(),
            expected_identity.deferred_coverage_id,
        ),
    ] {
        if actual != expected {
            return Err(OnnxModelError::InvalidModel(format!(
                "artifact identity {name} does not match admission contract"
            )));
        }
    }
    let specification = TiedEmbeddingHeadModelV2 {
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        packed: &weights_bytes[..verified.packed_len],
        scales: &verified.scales,
        format: verified.format,
        identity: OnnxArtifactIdentityV2 {
            source_model_id: &identity.source_model_id,
            tokenizer_id: &identity.tokenizer_id,
            recipe_id: &identity.recipe_id,
            tritium_build_id: &identity.tritium_build_id,
            package_id: &identity.package_id,
            converted_coverage_id: &identity.converted_coverage_id,
            deferred_coverage_id: &identity.deferred_coverage_id,
        },
    };
    let expected = encode_external_tied_embedding_head_v2(specification)?;
    verify_canonical_external(&expected, model_bytes, weights_bytes)?;
    Ok(VerifiedExternalOnnxModelV2 {
        model: VerifiedExternalOnnxModel {
            model_blake3: *blake3::hash(model_bytes).as_bytes(),
            weights_blake3: verified.weights_blake3,
            weights_bytes: weights_bytes.len(),
            tokens: verified.tokens,
            vocab: verified.vocab,
            hidden: verified.hidden,
            source_model_id: identity.source_model_id.clone(),
            recipe_id: identity.recipe_id.clone(),
            package_id: identity.package_id.clone(),
        },
        identity,
    })
}

struct VerifiedExternalParts {
    metadata: BTreeMap<String, String>,
    weights_blake3: [u8; 32],
    tokens: usize,
    vocab: usize,
    hidden: usize,
    format: TernaryFormat,
    packed_len: usize,
    scales: Vec<f32>,
}

fn verify_external_parts(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    schema_version: &'static str,
) -> Result<VerifiedExternalParts, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let mut metadata = BTreeMap::new();
    for entry in protobuf.metadata_props {
        if metadata.insert(entry.key.clone(), entry.value).is_some() {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate metadata key {}",
                entry.key
            )));
        }
    }
    require_metadata(&metadata, "tritium.schema_version", schema_version)?;
    require_metadata(&metadata, "tritium.tied_embedding_head", "true")?;
    require_metadata(
        &metadata,
        "tritium.external_data.file",
        EXTERNAL_WEIGHTS_FILE,
    )?;
    let tokens = parse_usize(&metadata, "tritium.tokens")?;
    let vocab = parse_usize(&metadata, "tritium.vocab")?;
    let hidden = parse_usize(&metadata, "tritium.hidden")?;
    let declared_weights = parse_usize(&metadata, "tritium.external_data.bytes")?;
    let format = match metadata_value(&metadata, "tritium.weight_format")? {
        "tq2_0" => TernaryFormat::Tq2_0,
        "tq1_0" => TernaryFormat::Tq1_0,
        other => {
            return Err(OnnxModelError::InvalidModel(format!(
                "unsupported weight format {other}"
            )));
        }
    };
    let actual_weights_hash = blake3::hash(weights_bytes);
    let actual_weights_hex = actual_weights_hash.to_hex().to_string();
    if metadata_value(&metadata, "tritium.external_data.blake3")? != actual_weights_hex {
        return Err(OnnxModelError::ExternalDataMismatch(
            "BLAKE3 digest differs from model metadata".to_owned(),
        ));
    }
    if weights_bytes.len() != declared_weights {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "byte length {} differs from declared {declared_weights}",
            weights_bytes.len()
        )));
    }
    let block_bytes = match format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let packed_len = num_blocks(hidden)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(vocab))
        .ok_or(OnnxModelError::ShapeOverflow("packed byte count"))?;
    let scale_offset = align_up(packed_len, EXTERNAL_ALIGNMENT)?;
    let scale_len = vocab
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(OnnxModelError::ShapeOverflow("scale byte count"))?;
    let expected_weights = scale_offset
        .checked_add(scale_len)
        .ok_or(OnnxModelError::ShapeOverflow("external weight byte count"))?;
    if weights_bytes.len() != expected_weights {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "canonical geometry requires {expected_weights} bytes, got {}",
            weights_bytes.len()
        )));
    }
    if weights_bytes[packed_len..scale_offset]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(OnnxModelError::ExternalDataMismatch(
            "alignment padding is not zero".to_owned(),
        ));
    }
    let scales: Vec<f32> = weights_bytes[scale_offset..]
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    Ok(VerifiedExternalParts {
        metadata,
        weights_blake3: *actual_weights_hash.as_bytes(),
        tokens,
        vocab,
        hidden,
        format,
        packed_len,
        scales,
    })
}

fn verify_canonical_external(
    expected: &ExternalOnnxModel,
    model_bytes: &[u8],
    weights_bytes: &[u8],
) -> Result<(), OnnxModelError> {
    if expected.model_bytes != model_bytes {
        return Err(OnnxModelError::InvalidModel(
            "graph does not match the canonical bound Tritium graph".to_owned(),
        ));
    }
    if expected.weights_bytes != weights_bytes {
        return Err(OnnxModelError::ExternalDataMismatch(
            "initializer layout is not canonical".to_owned(),
        ));
    }
    Ok(())
}

fn metadata_value<'a>(
    metadata: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, OnnxModelError> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| OnnxModelError::InvalidModel(format!("missing metadata key {key}")))
}

fn require_metadata(
    metadata: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), OnnxModelError> {
    let value = metadata_value(metadata, key)?;
    if value != expected {
        return Err(OnnxModelError::InvalidModel(format!(
            "metadata {key} must be {expected:?}, got {value:?}"
        )));
    }
    Ok(())
}

fn parse_usize(metadata: &BTreeMap<String, String>, key: &str) -> Result<usize, OnnxModelError> {
    let value = metadata_value(metadata, key)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| OnnxModelError::InvalidModel(format!("metadata {key} is not usize")))?;
    if parsed.to_string() != value {
        return Err(OnnxModelError::InvalidModel(format!(
            "metadata {key} is not canonical decimal"
        )));
    }
    Ok(parsed)
}

fn parse_positive_f32(
    metadata: &BTreeMap<String, String>,
    key: &str,
) -> Result<f32, OnnxModelError> {
    let value = metadata_value(metadata, key)?;
    let parsed = value
        .parse::<f32>()
        .map_err(|_| OnnxModelError::InvalidModel(format!("metadata {key} is not f32")))?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed.to_string() != value {
        return Err(OnnxModelError::InvalidModel(format!(
            "metadata {key} is not canonical positive f32"
        )));
    }
    Ok(parsed)
}

fn encode_model(
    model: TiedEmbeddingHeadModel<'_>,
    initializer: Vec<TensorProto>,
    mut extra_metadata: Vec<StringStringEntryProto>,
    schema_version: &'static str,
) -> Result<Vec<u8>, OnnxModelError> {
    let tokens = as_i64(model.tokens, "token count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    let hidden = as_i64(model.hidden, "hidden size")?;
    let format = format_code(model.format)?;

    let graph = GraphProto {
        node: vec![
            NodeProto {
                input: strings(["tokens", "tritium.packed", "tritium.scales"]),
                output: strings(["hidden"]),
                name: "tritium.embedding".to_owned(),
                op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
                attribute: attributes(hidden, format),
                domain: ONNX_DOMAIN.to_owned(),
            },
            NodeProto {
                input: strings(["hidden", "tritium.packed", "tritium.scales"]),
                output: strings(["logits"]),
                name: "tritium.lm_head".to_owned(),
                op_type: ONNX_OP_NAME.to_owned(),
                attribute: attributes(hidden, format),
                domain: ONNX_DOMAIN.to_owned(),
            },
        ],
        name: "tritium.tied_embedding_head".to_owned(),
        initializer,
        input: vec![tensor_value("tokens", TENSOR_INT64, &[tokens])],
        output: vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])],
        value_info: Vec::new(),
    };
    let mut metadata_props = vec![
        metadata("tritium.schema_version", schema_version),
        metadata("tritium.source_model_id", model.source_model_id),
        metadata("tritium.recipe_id", model.recipe_id),
        metadata("tritium.package_id", model.package_id),
        metadata("tritium.weight_format", &model.format.to_string()),
        metadata("tritium.tied_embedding_head", "true"),
        metadata("tritium.tokens", &model.tokens.to_string()),
        metadata("tritium.vocab", &model.vocab.to_string()),
        metadata("tritium.hidden", &model.hidden.to_string()),
    ];
    metadata_props.append(&mut extra_metadata);
    Ok(ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: schema_version
            .parse()
            .expect("static schema version is numeric"),
        graph: Some(graph),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: TRITIUM_OPSET,
            },
        ],
        metadata_props,
    }
    .encode_to_vec())
}

struct CausalGraphBuilder {
    nodes: Vec<NodeProto>,
    initializers: Vec<TensorProto>,
    storage: CausalInitializerStorage,
    failure: Option<OnnxModelError>,
}

enum CausalInitializerStorage {
    Inline,
    External(ExternalInitializerStorage),
    Counting { bytes: usize },
}

struct ExternalInitializerStorage {
    weights: Vec<u8>,
    aliases: BTreeMap<String, TensorProto>,
}

fn encoded_external_range(tensor: &TensorProto) -> Result<(usize, usize), OnnxModelError> {
    let mut entries = BTreeMap::new();
    for entry in &tensor.external_data {
        if entries
            .insert(entry.key.as_str(), entry.value.as_str())
            .is_some()
        {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} has duplicate external-data key {}",
                tensor.name, entry.key
            )));
        }
    }
    if tensor.data_location != EXTERNAL_DATA
        || !tensor.raw_data.is_empty()
        || entries.len() != 3
        || entries.get("location") != Some(&EXTERNAL_WEIGHTS_FILE)
    {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "initializer {} external-data descriptor is not canonical",
            tensor.name
        )));
    }
    let parse = |key: &str| {
        let value = entries.get(key).ok_or_else(|| {
            OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} missing {key}",
                tensor.name
            ))
        })?;
        let parsed = value.parse::<usize>().map_err(|_| {
            OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} {key} is not usize",
                tensor.name
            ))
        })?;
        if parsed.to_string() != *value {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} {key} is not canonical decimal",
                tensor.name
            )));
        }
        Ok(parsed)
    };
    Ok((parse("offset")?, parse("length")?))
}

fn take_external_alias(
    storage: &mut ExternalInitializerStorage,
    name: &str,
    data_type: i32,
    dimensions: &[i64],
) -> Result<Option<(TensorProto, usize, usize)>, OnnxModelError> {
    let Some(tensor) = storage.aliases.remove(name) else {
        return Ok(None);
    };
    if tensor.data_type != data_type || tensor.dims != dimensions {
        return Err(OnnxModelError::InvalidModel(format!(
            "shared external initializer {name} dtype or shape differs"
        )));
    }
    let (offset, length) = encoded_external_range(&tensor)?;
    let end = offset
        .checked_add(length)
        .ok_or(OnnxModelError::ShapeOverflow(
            "shared external initializer range",
        ))?;
    if end > storage.weights.len() {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "shared external initializer {name} exceeds weight arena"
        )));
    }
    Ok(Some((tensor, offset, length)))
}

fn external_storage_result(
    storage: CausalInitializerStorage,
) -> Result<Option<Vec<u8>>, OnnxModelError> {
    match storage {
        CausalInitializerStorage::External(storage) => {
            if !storage.aliases.is_empty() {
                return Err(OnnxModelError::InvalidModel(format!(
                    "shared external initializers were not consumed: {}",
                    storage
                        .aliases
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            Ok(Some(storage.weights))
        }
        CausalInitializerStorage::Inline | CausalInitializerStorage::Counting { .. } => Ok(None),
    }
}

fn reserve_external_range(weights: &mut Vec<u8>, length: usize) -> Result<usize, OnnxModelError> {
    let offset = align_up(weights.len(), EXTERNAL_ALIGNMENT)?;
    let required = offset
        .checked_sub(weights.len())
        .and_then(|padding| padding.checked_add(length))
        .ok_or(OnnxModelError::ShapeOverflow(
            "external initializer allocation",
        ))?;
    weights
        .try_reserve_exact(required)
        .map_err(|_| OnnxModelError::ShapeOverflow("external initializer allocation"))?;
    weights.resize(offset, 0);
    Ok(offset)
}

impl Default for CausalGraphBuilder {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            initializers: Vec::new(),
            storage: CausalInitializerStorage::Inline,
            failure: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CausalGraphGeometry {
    past_tokens: i64,
    total_tokens: i64,
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    gqa_repeat: i64,
}

#[derive(Clone, Copy)]
struct FullAttentionGraphContext<'a> {
    tokens: i64,
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    head_dim_usize: usize,
    query_width: i64,
    geometry: CausalGraphGeometry,
    rotary_state: Option<&'a RotaryGraphState>,
    attention_mask: &'a str,
    epsilon: f32,
    zero_centered_norm: bool,
}

#[derive(Debug, Clone, Copy)]
enum DeltaStateScope {
    Standalone,
    LayerIndex(usize),
}

#[derive(Debug, Clone, Copy)]
struct DeltaNetGraphContext {
    geometry: QwenDeltaNetGeometry,
    state_scope: DeltaStateScope,
    epsilon: f32,
}

impl DeltaStateScope {
    fn names(self) -> [String; 4] {
        match self {
            Self::Standalone => [
                "conv_state".to_owned(),
                "recurrent_state".to_owned(),
                "next_conv".to_owned(),
                "next_recurrent".to_owned(),
            ],
            Self::LayerIndex(index) => [
                format!("conv_state.{index}"),
                format!("recurrent_state.{index}"),
                format!("next_conv.{index}"),
                format!("next_recurrent.{index}"),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RotaryGraphConfig {
    tokens: usize,
    head_dim: usize,
    rotary_dim: usize,
    past_tokens: usize,
    theta: f32,
}

struct RotaryGraphState {
    first_start: String,
    first_end: String,
    second_start: String,
    second_end: String,
    tail_start: String,
    tail_end: String,
    has_tail: bool,
    axes: String,
    steps: String,
    cos: String,
    sin: String,
}

fn rotary_tables(config: RotaryGraphConfig) -> Result<(Vec<f32>, Vec<f32>), OnnxModelError> {
    let tokens = config.tokens;
    let elements = tokens
        .checked_mul(config.rotary_dim)
        .ok_or(OnnxModelError::ShapeOverflow("RoPE table element count"))?;
    let table_bytes = elements
        .checked_mul(2)
        .and_then(|values| values.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(OnnxModelError::ShapeOverflow("RoPE table byte count"))?;
    if table_bytes > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "RoPE tables require {table_bytes} bytes, exceed bounded inline graph"
        )));
    }
    let mut cos = Vec::with_capacity(elements);
    let mut sin = Vec::with_capacity(elements);
    for token in 0..tokens {
        for lane in 0..config.rotary_dim {
            let (lane_cos, lane_sin) = rotary_pair(config, token, lane)?;
            cos.push(lane_cos as f32);
            sin.push(lane_sin as f32);
        }
    }
    Ok((cos, sin))
}

fn rotary_pair(
    config: RotaryGraphConfig,
    token: usize,
    lane: usize,
) -> Result<(f64, f64), OnnxModelError> {
    let half = config.rotary_dim / 2;
    let frequency_lane = lane % half;
    let position = (config.past_tokens + token) as f64;
    let angle = position
        * f64::from(config.theta).powf(-2.0 * frequency_lane as f64 / config.rotary_dim as f64);
    if !angle.is_finite() {
        return Err(OnnxModelError::InvalidModel(format!(
            "RoPE angle is non-finite at position {} frequency lane {frequency_lane}",
            config.past_tokens + token
        )));
    }
    let (sin, cos) = angle.sin_cos();
    Ok((cos, sin))
}

struct CacheGraphOutputs {
    expanded_k: String,
    expanded_v: String,
    declarations: [ValueInfoProto; 2],
}

impl CausalGraphBuilder {
    fn counting() -> Self {
        Self {
            storage: CausalInitializerStorage::Counting { bytes: 0 },
            ..Self::default()
        }
    }

    fn external() -> Self {
        Self {
            storage: CausalInitializerStorage::External(ExternalInitializerStorage {
                weights: Vec::new(),
                aliases: BTreeMap::new(),
            }),
            ..Self::default()
        }
    }

    fn external_reusing(weights: Vec<u8>, aliases: BTreeMap<String, TensorProto>) -> Self {
        Self {
            storage: CausalInitializerStorage::External(ExternalInitializerStorage {
                weights,
                aliases,
            }),
            ..Self::default()
        }
    }

    fn is_external(&self) -> bool {
        matches!(self.storage, CausalInitializerStorage::External(_))
    }

    fn storage_result(&self) -> Result<(), OnnxModelError> {
        self.failure.clone().map_or(Ok(()), Err)
    }

    fn count_inline_bytes(&mut self, bytes: usize) {
        let CausalInitializerStorage::Counting { bytes: total } = &mut self.storage else {
            return;
        };
        let Some(next) = total.checked_add(bytes) else {
            self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                "inline initializer byte count",
            ));
            return;
        };
        *total = next;
        if next > MAX_MODEL_BYTES {
            self.failure
                .get_or_insert(OnnxModelError::InvalidModel(format!(
                    "inline initializers require at least {next} bytes, exceed bounded graph"
                )));
        }
    }

    fn store_bytes(&mut self, name: &str, data_type: i32, dimensions: Vec<i64>, bytes: &[u8]) {
        if self.failure.is_some() {
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Counting { .. }) {
            self.count_inline_bytes(bytes.len());
            return;
        }
        let tensor = match &mut self.storage {
            CausalInitializerStorage::Inline => {
                inline_tensor(name, data_type, dimensions, bytes.to_vec())
            }
            CausalInitializerStorage::External(storage) => {
                match take_external_alias(storage, name, data_type, &dimensions) {
                    Ok(Some((tensor, offset, length))) => {
                        if length != bytes.len()
                            || storage.weights[offset..offset + length] != *bytes
                        {
                            self.failure
                                .get_or_insert(OnnxModelError::InvalidModel(format!(
                                    "shared external initializer {name} bytes differ"
                                )));
                            return;
                        }
                        tensor
                    }
                    Ok(None) => {
                        let offset = match reserve_external_range(&mut storage.weights, bytes.len())
                        {
                            Ok(offset) => offset,
                            Err(error) => {
                                self.failure.get_or_insert(error);
                                return;
                            }
                        };
                        storage.weights.extend_from_slice(bytes);
                        let offset = match i64::try_from(offset) {
                            Ok(offset) => offset,
                            Err(_) => {
                                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                                    "external initializer offset",
                                ));
                                return;
                            }
                        };
                        let length = match i64::try_from(bytes.len()) {
                            Ok(length) => length,
                            Err(_) => {
                                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                                    "external initializer length",
                                ));
                                return;
                            }
                        };
                        external_tensor(name, data_type, dimensions, offset, length)
                    }
                    Err(error) => {
                        self.failure.get_or_insert(error);
                        return;
                    }
                }
            }
            CausalInitializerStorage::Counting { .. } => unreachable!(),
        };
        self.initializers.push(tensor);
    }

    fn standard(
        &mut self,
        name: impl Into<String>,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attribute: Vec<AttributeProto>,
    ) {
        if self.failure.is_some() {
            return;
        }
        self.nodes.push(NodeProto {
            input: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output: outputs.iter().map(|value| (*value).to_owned()).collect(),
            name: name.into(),
            op_type: op_type.to_owned(),
            attribute,
            domain: String::new(),
        });
    }

    fn add_f32(&mut self, name: &str, dimensions: Vec<i64>, values: &[f32]) {
        if self.failure.is_some() {
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Counting { .. }) {
            let Some(bytes) = values.len().checked_mul(core::mem::size_of::<f32>()) else {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "inline f32 initializer byte count",
                ));
                return;
            };
            self.count_inline_bytes(bytes);
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Inline) {
            self.initializers.push(inline_tensor(
                name,
                TENSOR_FLOAT,
                dimensions,
                scale_bytes(values),
            ));
            return;
        }
        let length = match values.len().checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "external f32 initializer byte count",
                ));
                return;
            }
        };
        let CausalInitializerStorage::External(storage) = &mut self.storage else {
            unreachable!();
        };
        match take_external_alias(storage, name, TENSOR_FLOAT, &dimensions) {
            Ok(Some((tensor, offset, alias_length))) => {
                let bytes = &storage.weights[offset..offset + alias_length];
                if alias_length != length
                    || !bytes
                        .chunks_exact(core::mem::size_of::<f32>())
                        .zip(values)
                        .all(|(bytes, value)| bytes == value.to_le_bytes())
                {
                    self.failure
                        .get_or_insert(OnnxModelError::InvalidModel(format!(
                            "shared external initializer {name} values differ"
                        )));
                    return;
                }
                self.initializers.push(tensor);
                return;
            }
            Ok(None) => {}
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        }
        let offset = match reserve_external_range(&mut storage.weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        for value in values {
            storage.weights.extend_from_slice(&value.to_le_bytes());
        }
        let Ok(offset) = i64::try_from(offset) else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external initializer offset"));
            return;
        };
        let Ok(length) = i64::try_from(length) else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external initializer length"));
            return;
        };
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset,
            length,
        ));
    }

    fn add_causal_mask(
        &mut self,
        name: &str,
        tokens: usize,
        total_tokens: usize,
        past_tokens: usize,
        dimensions: Vec<i64>,
    ) {
        if self.failure.is_some() {
            return;
        }
        let elements = match tokens.checked_mul(total_tokens) {
            Some(elements) => elements,
            None => {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "attention mask element count",
                ));
                return;
            }
        };
        if matches!(self.storage, CausalInitializerStorage::Counting { .. }) {
            let Some(bytes) = elements.checked_mul(core::mem::size_of::<f32>()) else {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("attention mask byte count"));
                return;
            };
            self.count_inline_bytes(bytes);
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Inline) {
            let mut values = Vec::new();
            if values.try_reserve_exact(elements).is_err() {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("attention mask allocation"));
                return;
            }
            for query_index in 0..tokens {
                let last_visible = past_tokens + query_index;
                for key_index in 0..total_tokens {
                    values.push(if key_index <= last_visible {
                        0.0
                    } else {
                        -1.0e9
                    });
                }
            }
            self.add_f32(name, dimensions, &values);
            return;
        }
        let length = match elements.checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("attention mask byte count"));
                return;
            }
        };
        let CausalInitializerStorage::External(storage) = &mut self.storage else {
            unreachable!();
        };
        let offset = match reserve_external_range(&mut storage.weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        let (Ok(offset_i64), Ok(length_i64)) = (i64::try_from(offset), i64::try_from(length))
        else {
            self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                "external attention mask range",
            ));
            return;
        };
        for query_index in 0..tokens {
            let last_visible = past_tokens + query_index;
            for key_index in 0..total_tokens {
                let value = if key_index <= last_visible {
                    0.0_f32
                } else {
                    -1.0e9
                };
                storage.weights.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset_i64,
            length_i64,
        ));
    }

    fn add_external_rotary_table(
        &mut self,
        name: &str,
        config: RotaryGraphConfig,
        cosine: bool,
        dimensions: Vec<i64>,
    ) {
        if self.failure.is_some() {
            return;
        }
        let elements = match config.tokens.checked_mul(config.rotary_dim) {
            Some(elements) => elements,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("RoPE table element count"));
                return;
            }
        };
        let length = match elements.checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("RoPE table byte count"));
                return;
            }
        };
        let CausalInitializerStorage::External(storage) = &mut self.storage else {
            self.failure.get_or_insert(OnnxModelError::InvalidModel(
                "direct RoPE table emission requires external storage".to_owned(),
            ));
            return;
        };
        let offset = match reserve_external_range(&mut storage.weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        let (Ok(offset_i64), Ok(length_i64)) = (i64::try_from(offset), i64::try_from(length))
        else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external RoPE table range"));
            return;
        };
        for token in 0..config.tokens {
            for lane in 0..config.rotary_dim {
                let pair = match rotary_pair(config, token, lane) {
                    Ok(pair) => pair,
                    Err(error) => {
                        self.failure.get_or_insert(error);
                        return;
                    }
                };
                let value = if cosine { pair.0 } else { pair.1 } as f32;
                storage.weights.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset_i64,
            length_i64,
        ));
    }

    fn add_i64(&mut self, name: &str, dimensions: Vec<i64>, values: &[i64]) {
        if self.failure.is_some() {
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Counting { .. }) {
            let Some(bytes) = values.len().checked_mul(core::mem::size_of::<i64>()) else {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "inline i64 initializer byte count",
                ));
                return;
            };
            self.count_inline_bytes(bytes);
            return;
        }
        self.initializers.push(inline_tensor(
            name,
            TENSOR_INT64,
            dimensions,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        ));
    }

    fn add_matrix(&mut self, prefix: &str, matrix: PackedTernaryMatrix<'_>) {
        self.store_bytes(
            &format!("{prefix}.packed"),
            TENSOR_UINT8,
            vec![i64::try_from(matrix.packed.len()).expect("validated packed byte count")],
            matrix.packed,
        );
        self.add_f32(
            &format!("{prefix}.scales"),
            vec![i64::try_from(matrix.rows).expect("validated matrix rows")],
            matrix.scales,
        );
    }

    fn projection(
        &mut self,
        name: &str,
        input: &str,
        output: &str,
        matrix_prefix: &str,
        matrix: PackedTernaryMatrix<'_>,
    ) -> Result<(), OnnxModelError> {
        self.add_matrix(matrix_prefix, matrix);
        self.storage_result()?;
        self.nodes.push(NodeProto {
            input: vec![
                input.to_owned(),
                format!("{matrix_prefix}.packed"),
                format!("{matrix_prefix}.scales"),
            ],
            output: vec![output.to_owned()],
            name: name.to_owned(),
            op_type: ONNX_OP_NAME.to_owned(),
            attribute: attributes(
                as_i64(matrix.columns, "matrix columns")?,
                format_code(matrix.format)?,
            ),
            domain: ONNX_DOMAIN.to_owned(),
        });
        Ok(())
    }

    fn deinterleave_query_gate(
        &mut self,
        prefix: &str,
        fused: &str,
        tokens: i64,
        n_head: i64,
        head_dim: i64,
        query_width: i64,
    ) -> (String, String) {
        let fused_shape = format!("{prefix}.fused_shape");
        let flat_shape = format!("{prefix}.flat_shape");
        let first_start = format!("{prefix}.first_start");
        let first_end = format!("{prefix}.first_end");
        let second_start = format!("{prefix}.second_start");
        let second_end = format!("{prefix}.second_end");
        let axes = format!("{prefix}.axes");
        let steps = format!("{prefix}.steps");
        self.add_i64(&fused_shape, vec![4], &[tokens, n_head, 2, head_dim]);
        self.add_i64(&flat_shape, vec![2], &[tokens, query_width]);
        self.add_i64(&first_start, vec![1], &[0]);
        self.add_i64(&first_end, vec![1], &[1]);
        self.add_i64(&second_start, vec![1], &[1]);
        self.add_i64(&second_end, vec![1], &[2]);
        self.add_i64(&axes, vec![1], &[2]);
        self.add_i64(&steps, vec![1], &[1]);

        let heads = format!("{prefix}.heads");
        self.standard(
            format!("{prefix}.reshape"),
            "Reshape",
            &[fused, &fused_shape],
            &[&heads],
            Vec::new(),
        );
        let query_heads = format!("{prefix}.query_heads");
        let gate_heads = format!("{prefix}.gate_heads");
        self.standard(
            format!("{prefix}.slice_query"),
            "Slice",
            &[&heads, &first_start, &first_end, &axes, &steps],
            &[&query_heads],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.slice_gate"),
            "Slice",
            &[&heads, &second_start, &second_end, &axes, &steps],
            &[&gate_heads],
            Vec::new(),
        );
        let query = format!("{prefix}.query");
        let gate = format!("{prefix}.gate");
        self.standard(
            format!("{prefix}.flatten_query"),
            "Reshape",
            &[&query_heads, &flat_shape],
            &[&query],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.flatten_gate"),
            "Reshape",
            &[&gate_heads, &flat_shape],
            &[&gate],
            Vec::new(),
        );
        (query, gate)
    }

    fn rms_norm_with_semantics(
        &mut self,
        prefix: &str,
        input: &str,
        output: &str,
        weight: &[f32],
        epsilon: f32,
        zero_centered: bool,
    ) {
        let squared = format!("{prefix}.squared");
        let mean = format!("{prefix}.mean");
        let stabilized = format!("{prefix}.stabilized");
        let root = format!("{prefix}.root");
        let normalized = format!("{prefix}.normalized");
        let axes = format!("{prefix}.axes");
        let epsilon_name = format!("{prefix}.epsilon");
        let weight_name = format!("{prefix}.weight");
        self.add_i64(&axes, vec![1], &[-1]);
        self.add_f32(&epsilon_name, Vec::new(), &[epsilon]);
        self.add_f32(
            &weight_name,
            vec![i64::try_from(weight.len()).expect("validated norm width")],
            weight,
        );
        let effective_weight = if zero_centered {
            let unit_name = format!("{prefix}.unit");
            let effective_name = format!("{prefix}.effective_weight");
            self.add_f32(&unit_name, Vec::new(), &[1.0]);
            self.standard(
                format!("{prefix}.zero_centered_scale"),
                "Add",
                &[&weight_name, &unit_name],
                &[&effective_name],
                Vec::new(),
            );
            effective_name
        } else {
            weight_name
        };
        self.standard(
            format!("{prefix}.square"),
            "Mul",
            &[input, input],
            &[&squared],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.reduce"),
            "ReduceMean",
            &[&squared, &axes],
            &[&mean],
            vec![int_attribute("keepdims", 1)],
        );
        self.standard(
            format!("{prefix}.stabilize"),
            "Add",
            &[&mean, &epsilon_name],
            &[&stabilized],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.sqrt"),
            "Sqrt",
            &[&stabilized],
            &[&root],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.divide"),
            "Div",
            &[input, &root],
            &[&normalized],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.scale"),
            "Mul",
            &[&normalized, &effective_weight],
            &[output],
            Vec::new(),
        );
    }

    fn prepare_rotary(
        &mut self,
        config: RotaryGraphConfig,
    ) -> Result<RotaryGraphState, OnnxModelError> {
        let tokens = config.tokens;
        let half = config.rotary_dim / 2;
        let first_start = "rope.first_start".to_owned();
        let first_end = "rope.first_end".to_owned();
        let second_start = "rope.second_start".to_owned();
        let second_end = "rope.second_end".to_owned();
        let axes = "rope.axes".to_owned();
        let steps = "rope.steps".to_owned();
        self.add_i64(&first_start, vec![1], &[0]);
        self.add_i64(&first_end, vec![1], &[as_i64(half, "RoPE half dimension")?]);
        self.add_i64(
            &second_start,
            vec![1],
            &[as_i64(half, "RoPE half dimension")?],
        );
        self.add_i64(
            &second_end,
            vec![1],
            &[as_i64(config.rotary_dim, "RoPE rotary dimension")?],
        );
        let tail_start = "rope.tail_start".to_owned();
        let tail_end = "rope.tail_end".to_owned();
        self.add_i64(
            &tail_start,
            vec![1],
            &[as_i64(config.rotary_dim, "RoPE rotary dimension")?],
        );
        self.add_i64(
            &tail_end,
            vec![1],
            &[as_i64(config.head_dim, "RoPE head dimension")?],
        );
        self.add_i64(&axes, vec![1], &[-1]);
        self.add_i64(&steps, vec![1], &[1]);
        let cos_name = "rope.cos".to_owned();
        let sin_name = "rope.sin".to_owned();
        let dimensions = vec![
            as_i64(tokens, "RoPE token count")?,
            1,
            as_i64(config.rotary_dim, "RoPE rotary dimension")?,
        ];
        if matches!(self.storage, CausalInitializerStorage::Counting { .. }) {
            let bytes = config
                .tokens
                .checked_mul(config.rotary_dim)
                .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
                .ok_or(OnnxModelError::ShapeOverflow("RoPE table byte count"))?;
            self.count_inline_bytes(bytes);
            self.count_inline_bytes(bytes);
        } else if self.is_external() {
            self.add_external_rotary_table(&cos_name, config, true, dimensions.clone());
            self.add_external_rotary_table(&sin_name, config, false, dimensions);
        } else {
            let (cos, sin) = rotary_tables(config)?;
            self.add_f32(&cos_name, dimensions.clone(), &cos);
            self.add_f32(&sin_name, dimensions, &sin);
        }
        self.storage_result()?;
        Ok(RotaryGraphState {
            first_start,
            first_end,
            second_start,
            second_end,
            tail_start,
            tail_end,
            has_tail: config.rotary_dim < config.head_dim,
            axes,
            steps,
            cos: cos_name,
            sin: sin_name,
        })
    }

    fn rotary(&mut self, prefix: &str, input: &str, output: &str, state: &RotaryGraphState) {
        let first = format!("{prefix}.first");
        let second = format!("{prefix}.second");
        self.standard(
            format!("{prefix}.slice_first"),
            "Slice",
            &[
                input,
                &state.first_start,
                &state.first_end,
                &state.axes,
                &state.steps,
            ],
            &[&first],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.slice_second"),
            "Slice",
            &[
                input,
                &state.second_start,
                &state.second_end,
                &state.axes,
                &state.steps,
            ],
            &[&second],
            Vec::new(),
        );
        let negated_second = format!("{prefix}.negated_second");
        self.standard(
            format!("{prefix}.negate_second"),
            "Neg",
            &[&second],
            &[&negated_second],
            Vec::new(),
        );
        let rotated = format!("{prefix}.rotated");
        self.standard(
            format!("{prefix}.rotate_half"),
            "Concat",
            &[&negated_second, &first],
            &[&rotated],
            vec![int_attribute("axis", -1)],
        );
        let unrotated = format!("{prefix}.unrotated");
        self.standard(
            format!("{prefix}.unrotated_prefix"),
            "Concat",
            &[&first, &second],
            &[&unrotated],
            vec![int_attribute("axis", -1)],
        );
        let direct = format!("{prefix}.direct");
        let crossed = format!("{prefix}.crossed");
        self.standard(
            format!("{prefix}.direct_mul"),
            "Mul",
            &[&unrotated, &state.cos],
            &[&direct],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.cross_mul"),
            "Mul",
            &[&rotated, &state.sin],
            &[&crossed],
            Vec::new(),
        );
        let rotated_prefix = if state.has_tail {
            format!("{prefix}.rotated_prefix")
        } else {
            output.to_owned()
        };
        self.standard(
            format!("{prefix}.sum"),
            "Add",
            &[&direct, &crossed],
            &[&rotated_prefix],
            Vec::new(),
        );
        if state.has_tail {
            let tail = format!("{prefix}.tail");
            self.standard(
                format!("{prefix}.slice_tail"),
                "Slice",
                &[
                    input,
                    &state.tail_start,
                    &state.tail_end,
                    &state.axes,
                    &state.steps,
                ],
                &[&tail],
                Vec::new(),
            );
            self.standard(
                format!("{prefix}.append_tail"),
                "Concat",
                &[&rotated_prefix, &tail],
                &[output],
                vec![int_attribute("axis", -1)],
            );
        }
    }

    fn cache_and_expand_gqa(
        &mut self,
        index: usize,
        current_k: &str,
        current_v: &str,
        geometry: CausalGraphGeometry,
        inputs: &mut Vec<ValueInfoProto>,
    ) -> CacheGraphOutputs {
        let prefix = format!("layer.{index}");
        let present_k = format!("present_k.{index}");
        let present_v = format!("present_v.{index}");
        if geometry.past_tokens == 0 {
            self.standard(
                format!("{prefix}.key_identity"),
                "Identity",
                &[current_k],
                &[&present_k],
                Vec::new(),
            );
            self.standard(
                format!("{prefix}.value_identity"),
                "Identity",
                &[current_v],
                &[&present_v],
                Vec::new(),
            );
        } else {
            let past_k = format!("past_k.{index}");
            let past_v = format!("past_v.{index}");
            inputs.push(tensor_value(
                &past_k,
                TENSOR_FLOAT,
                &[geometry.past_tokens, geometry.n_kv_head, geometry.head_dim],
            ));
            inputs.push(tensor_value(
                &past_v,
                TENSOR_FLOAT,
                &[geometry.past_tokens, geometry.n_kv_head, geometry.head_dim],
            ));
            self.standard(
                format!("{prefix}.key_concat"),
                "Concat",
                &[&past_k, current_k],
                &[&present_k],
                vec![int_attribute("axis", 0)],
            );
            self.standard(
                format!("{prefix}.value_concat"),
                "Concat",
                &[&past_v, current_v],
                &[&present_v],
                vec![int_attribute("axis", 0)],
            );
        }
        let declarations = [
            tensor_value(
                &present_k,
                TENSOR_FLOAT,
                &[geometry.total_tokens, geometry.n_kv_head, geometry.head_dim],
            ),
            tensor_value(
                &present_v,
                TENSOR_FLOAT,
                &[geometry.total_tokens, geometry.n_kv_head, geometry.head_dim],
            ),
        ];
        let grouped_k_shape = format!("{prefix}.grouped_k_shape");
        let grouped_v_shape = format!("{prefix}.grouped_v_shape");
        let grouped_shape = [
            geometry.total_tokens,
            geometry.n_kv_head,
            1,
            geometry.head_dim,
        ];
        self.add_i64(&grouped_k_shape, vec![4], &grouped_shape);
        self.add_i64(&grouped_v_shape, vec![4], &grouped_shape);
        let grouped_k = format!("{prefix}.grouped_k");
        let grouped_v = format!("{prefix}.grouped_v");
        self.standard(
            format!("{prefix}.key_group_reshape"),
            "Reshape",
            &[&present_k, &grouped_k_shape],
            &[&grouped_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_group_reshape"),
            "Reshape",
            &[&present_v, &grouped_v_shape],
            &[&grouped_v],
            Vec::new(),
        );
        let repeats = format!("{prefix}.gqa_repeats");
        self.add_i64(&repeats, vec![4], &[1, 1, geometry.gqa_repeat, 1]);
        let tiled_k = format!("{prefix}.tiled_k");
        let tiled_v = format!("{prefix}.tiled_v");
        self.standard(
            format!("{prefix}.key_gqa_tile"),
            "Tile",
            &[&grouped_k, &repeats],
            &[&tiled_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_gqa_tile"),
            "Tile",
            &[&grouped_v, &repeats],
            &[&tiled_v],
            Vec::new(),
        );
        let expanded_shape = format!("{prefix}.expanded_kv_shape");
        self.add_i64(
            &expanded_shape,
            vec![3],
            &[geometry.total_tokens, geometry.n_head, geometry.head_dim],
        );
        let expanded_k = format!("{prefix}.expanded_k");
        let expanded_v = format!("{prefix}.expanded_v");
        self.standard(
            format!("{prefix}.key_gqa"),
            "Reshape",
            &[&tiled_k, &expanded_shape],
            &[&expanded_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_gqa"),
            "Reshape",
            &[&tiled_v, &expanded_shape],
            &[&expanded_v],
            Vec::new(),
        );
        CacheGraphOutputs {
            expanded_k,
            expanded_v,
            declarations,
        }
    }

    fn full_attention_block(
        &mut self,
        index: usize,
        hidden_name: &str,
        layer: CausalLmDecoderLayer<'_>,
        context: FullAttentionGraphContext<'_>,
        inputs: &mut Vec<ValueInfoProto>,
        cache_outputs: &mut Vec<ValueInfoProto>,
    ) -> Result<String, OnnxModelError> {
        let FullAttentionGraphContext {
            tokens,
            n_head,
            n_kv_head,
            head_dim,
            head_dim_usize,
            query_width,
            geometry,
            rotary_state,
            attention_mask,
            epsilon,
            zero_centered_norm,
        } = context;
        let prefix = format!("layer.{index}");
        let attention_input = format!("{prefix}.attention_input");
        self.rms_norm_with_semantics(
            &format!("{prefix}.attention_norm"),
            hidden_name,
            &attention_input,
            layer.attention_norm,
            epsilon,
            zero_centered_norm,
        );
        let current_k_flat = format!("{prefix}.current_k_flat");
        let current_v_flat = format!("{prefix}.current_v_flat");
        let (query_flat, attention_gate) = match layer.query {
            CausalQueryProjection::HeadInterleavedQueryGate { fused: weight } => {
                let fused = format!("{prefix}.fused_query_gate");
                self.projection(
                    &format!("{prefix}.fused_query_gate_projection"),
                    &attention_input,
                    &fused,
                    &format!("{prefix}.fused_query_gate"),
                    weight,
                )?;
                let (query, gate) = self.deinterleave_query_gate(
                    &format!("{prefix}.query_gate_split"),
                    &fused,
                    tokens,
                    n_head,
                    head_dim,
                    query_width,
                );
                (query, Some(gate))
            }
            CausalQueryProjection::Separate {
                query: weight,
                gate,
            } => {
                let query = format!("{prefix}.query_flat");
                self.projection(
                    &format!("{prefix}.query"),
                    &attention_input,
                    &query,
                    &format!("{prefix}.query"),
                    weight,
                )?;
                let gate = if let Some(weight) = gate {
                    let gate = format!("{prefix}.attention_gate");
                    self.projection(
                        &format!("{prefix}.attention_gate_projection"),
                        &attention_input,
                        &gate,
                        &format!("{prefix}.attention_gate"),
                        weight,
                    )?;
                    Some(gate)
                } else {
                    None
                };
                (query, gate)
            }
        };
        self.projection(
            &format!("{prefix}.key"),
            &attention_input,
            &current_k_flat,
            &format!("{prefix}.key"),
            layer.key,
        )?;
        self.projection(
            &format!("{prefix}.value"),
            &attention_input,
            &current_v_flat,
            &format!("{prefix}.value"),
            layer.value,
        )?;
        let query_shape = format!("{prefix}.query_shape");
        let kv_shape = format!("{prefix}.kv_shape");
        self.add_i64(&query_shape, vec![3], &[tokens, n_head, head_dim]);
        self.add_i64(&kv_shape, vec![3], &[tokens, n_kv_head, head_dim]);
        let query_tokens = format!("{prefix}.query_tokens");
        let query = format!("{prefix}.query_heads");
        let current_k = format!("{prefix}.current_k");
        let current_v = format!("{prefix}.current_v");
        self.standard(
            format!("{prefix}.query_reshape"),
            "Reshape",
            &[&query_flat, &query_shape],
            &[&query_tokens],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.key_reshape"),
            "Reshape",
            &[&current_k_flat, &kv_shape],
            &[&current_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_reshape"),
            "Reshape",
            &[&current_v_flat, &kv_shape],
            &[&current_v],
            Vec::new(),
        );
        let mut query_ready = query_tokens;
        if let Some(weight) = layer.query_norm {
            let normalized = format!("{prefix}.query_norm.output");
            self.rms_norm_with_semantics(
                &format!("{prefix}.query_norm"),
                &query_ready,
                &normalized,
                weight,
                epsilon,
                zero_centered_norm,
            );
            query_ready = normalized;
        }
        let mut key_ready = current_k;
        if let Some(weight) = layer.key_norm {
            let normalized = format!("{prefix}.key_norm.output");
            self.rms_norm_with_semantics(
                &format!("{prefix}.key_norm"),
                &key_ready,
                &normalized,
                weight,
                epsilon,
                zero_centered_norm,
            );
            key_ready = normalized;
        }
        if let Some(rotary) = rotary_state {
            let rotated_query = format!("{prefix}.query_rope.output");
            self.rotary(
                &format!("{prefix}.query_rope"),
                &query_ready,
                &rotated_query,
                rotary,
            );
            query_ready = rotated_query;
            let rotated_key = format!("{prefix}.key_rope.output");
            self.rotary(
                &format!("{prefix}.key_rope"),
                &key_ready,
                &rotated_key,
                rotary,
            );
            key_ready = rotated_key;
        }
        self.standard(
            format!("{prefix}.query_transpose"),
            "Transpose",
            &[&query_ready],
            &[&query],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let cache = self.cache_and_expand_gqa(index, &key_ready, &current_v, geometry, inputs);
        cache_outputs.extend(cache.declarations);
        let expanded_k = cache.expanded_k;
        let expanded_v = cache.expanded_v;
        let transposed_k = format!("{prefix}.transposed_k");
        let transposed_v = format!("{prefix}.transposed_v");
        self.standard(
            format!("{prefix}.key_transpose"),
            "Transpose",
            &[&expanded_k],
            &[&transposed_k],
            vec![ints_attribute("perm", &[1, 2, 0])],
        );
        self.standard(
            format!("{prefix}.value_transpose"),
            "Transpose",
            &[&expanded_v],
            &[&transposed_v],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let scores = format!("{prefix}.scores");
        self.standard(
            format!("{prefix}.score_matmul"),
            "MatMul",
            &[&query, &transposed_k],
            &[&scores],
            Vec::new(),
        );
        let attention_scale = format!("{prefix}.attention_scale");
        self.add_f32(
            &attention_scale,
            Vec::new(),
            &[1.0 / (head_dim_usize as f32).sqrt()],
        );
        let scaled_scores = format!("{prefix}.scaled_scores");
        self.standard(
            format!("{prefix}.score_scale"),
            "Mul",
            &[&scores, &attention_scale],
            &[&scaled_scores],
            Vec::new(),
        );
        let masked_scores = format!("{prefix}.masked_scores");
        self.standard(
            format!("{prefix}.mask"),
            "Add",
            &[&scaled_scores, attention_mask],
            &[&masked_scores],
            Vec::new(),
        );
        let probabilities = format!("{prefix}.probabilities");
        self.standard(
            format!("{prefix}.softmax"),
            "Softmax",
            &[&masked_scores],
            &[&probabilities],
            vec![int_attribute("axis", -1)],
        );
        let context_heads = format!("{prefix}.context_heads");
        self.standard(
            format!("{prefix}.context_matmul"),
            "MatMul",
            &[&probabilities, &transposed_v],
            &[&context_heads],
            Vec::new(),
        );
        let context_tokens = format!("{prefix}.context_tokens");
        self.standard(
            format!("{prefix}.context_transpose"),
            "Transpose",
            &[&context_heads],
            &[&context_tokens],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let context_shape = format!("{prefix}.context_shape");
        self.add_i64(&context_shape, vec![2], &[tokens, query_width]);
        let context = format!("{prefix}.context");
        self.standard(
            format!("{prefix}.context_reshape"),
            "Reshape",
            &[&context_tokens, &context_shape],
            &[&context],
            Vec::new(),
        );
        let mut output_input = context;
        if let Some(gate) = attention_gate {
            let sigmoid = format!("{prefix}.attention_gate_sigmoid");
            self.standard(
                format!("{prefix}.attention_gate_activation"),
                "Sigmoid",
                &[&gate],
                &[&sigmoid],
                Vec::new(),
            );
            let gated = format!("{prefix}.gated_attention");
            self.standard(
                format!("{prefix}.attention_gate_multiply"),
                "Mul",
                &[&output_input, &sigmoid],
                &[&gated],
                Vec::new(),
            );
            output_input = gated;
        }
        if let Some(weight) = layer.attention_sub_norm {
            let normalized = format!("{prefix}.attention_sub_norm.output");
            self.rms_norm_with_semantics(
                &format!("{prefix}.attention_sub_norm"),
                &output_input,
                &normalized,
                weight,
                epsilon,
                zero_centered_norm,
            );
            output_input = normalized;
        }
        let attention_output = format!("{prefix}.attention_output");
        self.projection(
            &format!("{prefix}.output"),
            &output_input,
            &attention_output,
            &format!("{prefix}.attention_output"),
            layer.attention_output,
        )?;
        let post_attention = format!("{prefix}.post_attention");
        self.standard(
            format!("{prefix}.attention_residual"),
            "Add",
            &[hidden_name, &attention_output],
            &[&post_attention],
            Vec::new(),
        );
        let layer_output = format!("layer.{}.input", index + 1);
        self.ffn_block(
            &prefix,
            &post_attention,
            &layer_output,
            layer.ffn(),
            epsilon,
            zero_centered_norm,
        )?;
        Ok(layer_output)
    }

    fn delta_net_block(
        &mut self,
        index: usize,
        hidden_name: &str,
        layer: QwenDeltaNetDecoderLayer<'_>,
        context: DeltaNetGraphContext,
        inputs: &mut Vec<ValueInfoProto>,
        state_outputs: &mut Vec<ValueInfoProto>,
    ) -> Result<String, OnnxModelError> {
        let DeltaNetGraphContext {
            geometry,
            state_scope,
            epsilon: epsilon_value,
        } = context;
        let dimensions = geometry
            .dimensions()
            .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
        let prefix = format!("layer.{index}");
        let attention_input = format!("{prefix}.attention_input");
        self.rms_norm_with_semantics(
            &format!("{prefix}.attention_norm"),
            hidden_name,
            &attention_input,
            layer.attention_norm,
            epsilon_value,
            true,
        );
        let raw_qkv = format!("{prefix}.raw_qkv");
        let z = format!("{prefix}.z");
        let beta = format!("{prefix}.beta_logits");
        let decay = format!("{prefix}.decay_logits");
        for (node, output, initializer, matrix) in [
            ("qkv_projection", &raw_qkv, "qkv", layer.qkv),
            ("z_projection", &z, "z", layer.z),
            ("beta_projection", &beta, "beta", layer.beta),
            ("decay_projection", &decay, "decay", layer.decay),
        ] {
            self.projection(
                &format!("{prefix}.{node}"),
                &attention_input,
                output,
                &format!("{prefix}.deltanet.{initializer}"),
                matrix,
            )?;
        }
        let conv_weight = format!("{prefix}.deltanet.conv.weight");
        let norm_weight = format!("{prefix}.deltanet.norm.weight");
        let dt_bias = format!("{prefix}.deltanet.dt_bias");
        let a_log = format!("{prefix}.deltanet.a_log");
        let epsilon = format!("{prefix}.deltanet.epsilon");
        self.add_f32(
            &conv_weight,
            vec![
                as_i64(dimensions.conv_width(), "DeltaNet convolution width")?,
                as_i64(
                    geometry.conv_kernel_dim(),
                    "DeltaNet convolution kernel width",
                )?,
            ],
            layer.conv_weight,
        );
        self.add_f32(
            &norm_weight,
            vec![as_i64(
                geometry.value_head_dim(),
                "DeltaNet value head width",
            )?],
            layer.norm_weight,
        );
        self.add_f32(
            &dt_bias,
            vec![as_i64(
                geometry.num_value_heads(),
                "DeltaNet value head count",
            )?],
            layer.dt_bias,
        );
        self.add_f32(
            &a_log,
            vec![as_i64(
                geometry.num_value_heads(),
                "DeltaNet value head count",
            )?],
            layer.a_log,
        );
        self.add_f32(&epsilon, Vec::new(), &[epsilon_value]);
        let [conv_state, recurrent_state, next_conv, next_recurrent] = state_scope.names();
        let conv_shape = [
            as_i64(dimensions.conv_width(), "DeltaNet convolution width")?,
            as_i64(
                geometry.conv_kernel_dim(),
                "DeltaNet convolution kernel width",
            )?,
        ];
        let recurrent_shape = [
            as_i64(geometry.num_value_heads(), "DeltaNet value head count")?,
            as_i64(geometry.key_head_dim(), "DeltaNet key head width")?,
            as_i64(geometry.value_head_dim(), "DeltaNet value head width")?,
        ];
        inputs.push(tensor_value(&conv_state, TENSOR_FLOAT, &conv_shape));
        inputs.push(tensor_value(
            &recurrent_state,
            TENSOR_FLOAT,
            &recurrent_shape,
        ));
        state_outputs.push(tensor_value(&next_conv, TENSOR_FLOAT, &conv_shape));
        state_outputs.push(tensor_value(
            &next_recurrent,
            TENSOR_FLOAT,
            &recurrent_shape,
        ));
        let normalized_core = format!("{prefix}.normalized_core");
        let node_inputs = crate::QwenDeltaNetInput::ALL.map(|slot| match slot {
            crate::QwenDeltaNetInput::RawQkv => raw_qkv.as_str(),
            crate::QwenDeltaNetInput::Z => z.as_str(),
            crate::QwenDeltaNetInput::BetaLogits => beta.as_str(),
            crate::QwenDeltaNetInput::DecayLogits => decay.as_str(),
            crate::QwenDeltaNetInput::ConvWeight => conv_weight.as_str(),
            crate::QwenDeltaNetInput::NormWeight => norm_weight.as_str(),
            crate::QwenDeltaNetInput::DtBias => dt_bias.as_str(),
            crate::QwenDeltaNetInput::ALog => a_log.as_str(),
            crate::QwenDeltaNetInput::ConvState => conv_state.as_str(),
            crate::QwenDeltaNetInput::RecurrentState => recurrent_state.as_str(),
            crate::QwenDeltaNetInput::Epsilon => epsilon.as_str(),
        });
        self.nodes.push(NodeProto {
            input: strings(node_inputs),
            output: strings([&normalized_core, &next_conv, &next_recurrent]),
            name: format!("{prefix}.deltanet"),
            op_type: crate::ONNX_QWEN_DELTANET_OP_NAME.to_owned(),
            attribute: vec![
                int_attribute(
                    crate::ATTR_CONV_KERNEL_DIM,
                    as_i64(
                        geometry.conv_kernel_dim(),
                        "DeltaNet convolution kernel width",
                    )?,
                ),
                int_attribute(
                    crate::ATTR_NUM_KEY_HEADS,
                    as_i64(geometry.num_key_heads(), "DeltaNet query/key head count")?,
                ),
                int_attribute(
                    crate::ATTR_NUM_VALUE_HEADS,
                    as_i64(geometry.num_value_heads(), "DeltaNet value head count")?,
                ),
                int_attribute(
                    crate::ATTR_KEY_HEAD_DIM,
                    as_i64(geometry.key_head_dim(), "DeltaNet key head width")?,
                ),
                int_attribute(
                    crate::ATTR_VALUE_HEAD_DIM,
                    as_i64(geometry.value_head_dim(), "DeltaNet value head width")?,
                ),
            ],
            domain: ONNX_DOMAIN.to_owned(),
        });
        let attention_output = format!("{prefix}.attention_output");
        self.projection(
            &format!("{prefix}.output_projection"),
            &normalized_core,
            &attention_output,
            &format!("{prefix}.deltanet.output"),
            layer.output,
        )?;
        let post_attention = format!("{prefix}.post_attention");
        self.standard(
            format!("{prefix}.attention_residual"),
            "Add",
            &[hidden_name, &attention_output],
            &[&post_attention],
            Vec::new(),
        );
        let layer_output = format!("layer.{}.input", index + 1);
        self.ffn_block(
            &prefix,
            &post_attention,
            &layer_output,
            layer.ffn(),
            epsilon_value,
            true,
        )?;
        Ok(layer_output)
    }

    fn ffn_block(
        &mut self,
        prefix: &str,
        input: &str,
        output: &str,
        layer: CausalFfnLayer<'_>,
        epsilon: f32,
        zero_centered_norm: bool,
    ) -> Result<(), OnnxModelError> {
        let ffn_input = format!("{prefix}.ffn_input");
        self.rms_norm_with_semantics(
            &format!("{prefix}.ffn_norm"),
            input,
            &ffn_input,
            layer.ffn_norm,
            epsilon,
            zero_centered_norm,
        );
        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        self.projection(
            &format!("{prefix}.gate_projection"),
            &ffn_input,
            &gate,
            &format!("{prefix}.gate"),
            layer.gate,
        )?;
        self.projection(
            &format!("{prefix}.up_projection"),
            &ffn_input,
            &up,
            &format!("{prefix}.up"),
            layer.up,
        )?;
        let activated_gate = format!("{prefix}.activated_gate");
        match layer.activation {
            CausalActivation::SwiGlu => {
                let sigmoid = format!("{prefix}.gate_sigmoid");
                self.standard(
                    format!("{prefix}.sigmoid"),
                    "Sigmoid",
                    &[&gate],
                    &[&sigmoid],
                    Vec::new(),
                );
                self.standard(
                    format!("{prefix}.silu"),
                    "Mul",
                    &[&gate, &sigmoid],
                    &[&activated_gate],
                    Vec::new(),
                );
            }
            CausalActivation::Relu2 => {
                let relu = format!("{prefix}.gate_relu");
                self.standard(
                    format!("{prefix}.relu"),
                    "Relu",
                    &[&gate],
                    &[&relu],
                    Vec::new(),
                );
                self.standard(
                    format!("{prefix}.relu_square"),
                    "Mul",
                    &[&relu, &relu],
                    &[&activated_gate],
                    Vec::new(),
                );
            }
        }
        let gated = format!("{prefix}.gated");
        self.standard(
            format!("{prefix}.gated_multiply"),
            "Mul",
            &[&activated_gate, &up],
            &[&gated],
            Vec::new(),
        );
        let mut down_input = gated;
        if let Some(weight) = layer.ffn_sub_norm {
            let normalized = format!("{prefix}.ffn_sub_norm.output");
            self.rms_norm_with_semantics(
                &format!("{prefix}.ffn_sub_norm"),
                &down_input,
                &normalized,
                weight,
                epsilon,
                zero_centered_norm,
            );
            down_input = normalized;
        }
        let ffn_output = format!("{prefix}.ffn_output");
        self.projection(
            &format!("{prefix}.down_projection"),
            &down_input,
            &ffn_output,
            &format!("{prefix}.down"),
            layer.down,
        )?;
        self.standard(
            format!("{prefix}.ffn_residual"),
            "Add",
            &[input, &ffn_output],
            &[output],
            Vec::new(),
        );
        Ok(())
    }
}

fn build_qwen35_mtp_graph(
    model: Qwen35MtpModel<'_>,
    mut graph: CausalGraphBuilder,
) -> Result<(ModelProto, Option<Vec<u8>>), OnnxModelError> {
    let tokens = as_i64(model.tokens, "MTP token count")?;
    let past_tokens = as_i64(model.past_tokens, "MTP past token count")?;
    let total_tokens = tokens
        .checked_add(past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("MTP total token count"))?;
    let total_tokens_usize = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("MTP total token count"))?;
    let hidden = as_i64(model.embedding.columns, "MTP hidden size")?;
    let vocab = as_i64(model.embedding.rows, "MTP vocabulary")?;
    let n_head = as_i64(model.n_head, "MTP attention head count")?;
    let n_kv_head = as_i64(model.n_kv_head, "MTP KV head count")?;
    let head_dim = as_i64(model.head_dim, "MTP head dimension")?;
    let query_width = n_head
        .checked_mul(head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("MTP query width"))?;
    let geometry = CausalGraphGeometry {
        past_tokens,
        total_tokens,
        n_head,
        n_kv_head,
        head_dim,
        gqa_repeat: as_i64(model.n_head / model.n_kv_head, "MTP GQA repeat")?,
    };
    let rotary_state = graph.prepare_rotary(RotaryGraphConfig {
        tokens: model.tokens,
        head_dim: model.head_dim,
        rotary_dim: model.rotary.dimensions,
        past_tokens: model.past_tokens,
        theta: model.rotary.theta,
    })?;
    let attention_mask = "mtp.attention_mask";
    graph.add_causal_mask(
        attention_mask,
        model.tokens,
        total_tokens_usize,
        model.past_tokens,
        vec![1, tokens, total_tokens],
    );

    graph.add_matrix("tok_embeddings", model.embedding);
    graph.nodes.push(NodeProto {
        input: strings([
            "shifted_tokens",
            "tok_embeddings.packed",
            "tok_embeddings.scales",
        ]),
        output: strings(["mtp.embedding"]),
        name: "mtp.embedding".to_owned(),
        op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.embedding.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    graph.rms_norm_with_semantics(
        "mtp.embedding_norm",
        "mtp.embedding",
        "mtp.embedding_norm.output",
        model.mtp.pre_fc_norm_embedding,
        model.rms_epsilon,
        true,
    );
    graph.rms_norm_with_semantics(
        "mtp.target_norm",
        "target_hidden",
        "mtp.target_norm.output",
        model.mtp.pre_fc_norm_hidden,
        model.rms_epsilon,
        true,
    );
    graph.standard(
        "mtp.fusion_concat",
        "Concat",
        &["mtp.embedding_norm.output", "mtp.target_norm.output"],
        &["mtp.fusion_input"],
        vec![int_attribute("axis", 1)],
    );
    graph.projection(
        "mtp.fusion",
        "mtp.fusion_input",
        "mtp.layer_input",
        "mtp.fusion",
        model.mtp.fusion,
    )?;

    let mut inputs = vec![
        tensor_value("shifted_tokens", TENSOR_INT64, &[tokens]),
        tensor_value("target_hidden", TENSOR_FLOAT, &[tokens, hidden]),
    ];
    let mut state_outputs = Vec::with_capacity(2);
    let layer_output = graph.full_attention_block(
        0,
        "mtp.layer_input",
        model.mtp.layer.causal(),
        FullAttentionGraphContext {
            tokens,
            n_head,
            n_kv_head,
            head_dim,
            head_dim_usize: model.head_dim,
            query_width,
            geometry,
            rotary_state: Some(&rotary_state),
            attention_mask,
            epsilon: model.rms_epsilon,
            zero_centered_norm: true,
        },
        &mut inputs,
        &mut state_outputs,
    )?;
    graph.rms_norm_with_semantics(
        "mtp.final_norm",
        &layer_output,
        "mtp.final_hidden",
        model.mtp.final_norm,
        model.rms_epsilon,
        true,
    );
    graph.add_matrix("lm_head", model.lm_head);
    graph.nodes.push(NodeProto {
        input: strings(["mtp.final_hidden", "lm_head.packed", "lm_head.scales"]),
        output: strings(["mtp.logits"]),
        name: "mtp.lm_head".to_owned(),
        op_type: ONNX_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.lm_head.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    if let Some(error) = graph.failure {
        return Err(error);
    }
    let mut outputs = vec![
        tensor_value("mtp.logits", TENSOR_FLOAT, &[tokens, vocab]),
        tensor_value("mtp.final_hidden", TENSOR_FLOAT, &[tokens, hidden]),
    ];
    outputs.extend(state_outputs);
    let external_weights = external_storage_result(graph.storage)?;
    Ok((
        ModelProto {
            ir_version: ONNX_IR_VERSION,
            producer_name: "tritium-onnx".to_owned(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            domain: ONNX_DOMAIN.to_owned(),
            model_version: 2,
            graph: Some(GraphProto {
                node: graph.nodes,
                name: "tritium.qwen35_mtp".to_owned(),
                initializer: graph.initializers,
                input: inputs,
                output: outputs,
                value_info: Vec::new(),
            }),
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: ONNX_OPSET,
                },
                OperatorSetIdProto {
                    domain: ONNX_DOMAIN.to_owned(),
                    version: 2,
                },
            ],
            metadata_props: vec![
                metadata("tritium.schema_version", "2"),
                metadata("tritium.graph_kind", "qwen35-mtp"),
                metadata("tritium.source_model_id", model.identity.source_model_id),
                metadata("tritium.tokenizer_id", model.identity.tokenizer_id),
                metadata("tritium.recipe_id", model.identity.recipe_id),
                metadata("tritium.build_id", model.identity.tritium_build_id),
                metadata("tritium.package_id", model.identity.package_id),
                metadata(
                    "tritium.coverage.converted_id",
                    model.identity.converted_coverage_id,
                ),
                metadata(
                    "tritium.coverage.deferred_id",
                    model.identity.deferred_coverage_id,
                ),
                metadata("tritium.tokens", &model.tokens.to_string()),
                metadata("tritium.past_tokens", &model.past_tokens.to_string()),
                metadata("tritium.layers", "1"),
                metadata("tritium.hidden", &model.embedding.columns.to_string()),
                metadata("tritium.vocab", &model.embedding.rows.to_string()),
                metadata("tritium.n_head", &model.n_head.to_string()),
                metadata("tritium.n_kv_head", &model.n_kv_head.to_string()),
                metadata("tritium.head_dim", &model.head_dim.to_string()),
                metadata("tritium.rotary_dim", &model.rotary.dimensions.to_string()),
                metadata("tritium.rms_epsilon", &model.rms_epsilon.to_string()),
                metadata("tritium.input_alignment", "caller-shifted-target-aligned"),
                metadata("tritium.rms_norm_weight_semantics", "zero-centered-offset"),
                metadata("tritium.tied_embedding_head", "false"),
                metadata("tritium.rope_theta", &model.rotary.theta.to_string()),
            ],
        },
        external_weights,
    ))
}

fn build_qwen_causal_lm_graph(
    model: QwenCausalLmModel<'_>,
    mut graph: CausalGraphBuilder,
) -> Result<(ModelProto, Option<Vec<u8>>), OnnxModelError> {
    let tokens = as_i64(model.tokens, "token count")?;
    let past_tokens = as_i64(model.past_tokens, "past token count")?;
    let total_tokens = tokens
        .checked_add(past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let total_tokens_usize = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let hidden = as_i64(model.embedding.columns, "hidden size")?;
    let vocab = as_i64(model.embedding.rows, "vocabulary")?;
    let n_head = as_i64(model.n_head, "attention head count")?;
    let n_kv_head = as_i64(model.n_kv_head, "KV head count")?;
    let head_dim = as_i64(model.head_dim, "head dimension")?;
    let query_width = n_head
        .checked_mul(head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("query width"))?;
    let geometry = CausalGraphGeometry {
        past_tokens,
        total_tokens,
        n_head,
        n_kv_head,
        head_dim,
        gqa_repeat: as_i64(model.n_head / model.n_kv_head, "GQA repeat")?,
    };
    let rotary_state = graph.prepare_rotary(RotaryGraphConfig {
        tokens: model.tokens,
        head_dim: model.head_dim,
        rotary_dim: model.rotary.dimensions,
        past_tokens: model.past_tokens,
        theta: model.rotary.theta,
    })?;
    let mask_elements =
        model
            .tokens
            .checked_mul(total_tokens_usize)
            .ok_or(OnnxModelError::ShapeOverflow(
                "attention mask element count",
            ))?;
    if mask_elements > MAX_MODEL_BYTES / core::mem::size_of::<f32>() {
        return Err(OnnxModelError::InvalidModel(format!(
            "attention mask requires {mask_elements} elements, exceeds bounded inline graph"
        )));
    }
    let attention_mask = "attention.attention_mask";
    graph.add_causal_mask(
        attention_mask,
        model.tokens,
        total_tokens_usize,
        model.past_tokens,
        vec![1, tokens, total_tokens],
    );
    graph.add_matrix("tok_embeddings", model.embedding);
    graph.nodes.push(NodeProto {
        input: strings(["tokens", "tok_embeddings.packed", "tok_embeddings.scales"]),
        output: strings(["layer.0.input"]),
        name: "tok_embeddings".to_owned(),
        op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.embedding.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut hidden_name = "layer.0.input".to_owned();
    let mut inputs = vec![tensor_value("tokens", TENSOR_INT64, &[tokens])];
    let mut state_outputs = Vec::with_capacity(model.layers.len() * 2);
    let full_context = FullAttentionGraphContext {
        tokens,
        n_head,
        n_kv_head,
        head_dim,
        head_dim_usize: model.head_dim,
        query_width,
        geometry,
        rotary_state: Some(&rotary_state),
        attention_mask,
        epsilon: model.rms_epsilon,
        zero_centered_norm: true,
    };
    for (index, layer) in model.layers.iter().copied().enumerate() {
        hidden_name = match layer {
            QwenCausalLmDecoderLayer::DeltaNet(layer) => graph.delta_net_block(
                index,
                &hidden_name,
                layer,
                DeltaNetGraphContext {
                    geometry: model.delta_geometry,
                    state_scope: DeltaStateScope::LayerIndex(index),
                    epsilon: model.rms_epsilon,
                },
                &mut inputs,
                &mut state_outputs,
            )?,
            QwenCausalLmDecoderLayer::FullAttention(layer) => graph.full_attention_block(
                index,
                &hidden_name,
                layer.causal(),
                full_context,
                &mut inputs,
                &mut state_outputs,
            )?,
        };
    }
    let final_hidden = "final_norm.output";
    graph.rms_norm_with_semantics(
        "final_norm",
        &hidden_name,
        final_hidden,
        model.final_norm,
        model.rms_epsilon,
        true,
    );
    let (head_packed, head_scales, head_format) = if let Some(head) = model.lm_head {
        graph.add_matrix("lm_head", head);
        ("lm_head.packed", "lm_head.scales", head.format)
    } else {
        (
            "tok_embeddings.packed",
            "tok_embeddings.scales",
            model.embedding.format,
        )
    };
    graph.nodes.push(NodeProto {
        input: strings([final_hidden, head_packed, head_scales]),
        output: strings(["logits"]),
        name: "lm_head".to_owned(),
        op_type: ONNX_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(head_format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut outputs = vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])];
    outputs.extend(state_outputs);
    let schedule = model
        .layers
        .iter()
        .map(|layer| match layer {
            QwenCausalLmDecoderLayer::DeltaNet(_) => "linear_attention",
            QwenCausalLmDecoderLayer::FullAttention(_) => "full_attention",
        })
        .collect::<Vec<_>>()
        .join(",");
    let full_attention_interval = qwen_full_attention_interval(model.layers)?;
    if let Some(error) = graph.failure {
        return Err(error);
    }
    let external_weights = external_storage_result(graph.storage)?;
    Ok((
        ModelProto {
            ir_version: ONNX_IR_VERSION,
            producer_name: "tritium-onnx".to_owned(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            domain: ONNX_DOMAIN.to_owned(),
            model_version: 2,
            graph: Some(GraphProto {
                node: graph.nodes,
                name: "tritium.qwen_causal_lm".to_owned(),
                initializer: graph.initializers,
                input: inputs,
                output: outputs,
                value_info: Vec::new(),
            }),
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: ONNX_OPSET,
                },
                OperatorSetIdProto {
                    domain: ONNX_DOMAIN.to_owned(),
                    version: 2,
                },
            ],
            metadata_props: vec![
                metadata("tritium.schema_version", "2"),
                metadata("tritium.graph_kind", "qwen-causal-lm"),
                metadata("tritium.source_model_id", model.identity.source_model_id),
                metadata("tritium.tokenizer_id", model.identity.tokenizer_id),
                metadata("tritium.recipe_id", model.identity.recipe_id),
                metadata("tritium.build_id", model.identity.tritium_build_id),
                metadata("tritium.package_id", model.identity.package_id),
                metadata(
                    "tritium.coverage.converted_id",
                    model.identity.converted_coverage_id,
                ),
                metadata(
                    "tritium.coverage.deferred_id",
                    model.identity.deferred_coverage_id,
                ),
                metadata("tritium.tokens", &model.tokens.to_string()),
                metadata("tritium.past_tokens", &model.past_tokens.to_string()),
                metadata("tritium.layers", &model.layers.len().to_string()),
                metadata("tritium.hidden", &model.embedding.columns.to_string()),
                metadata("tritium.vocab", &model.embedding.rows.to_string()),
                metadata("tritium.n_head", &model.n_head.to_string()),
                metadata("tritium.n_kv_head", &model.n_kv_head.to_string()),
                metadata("tritium.head_dim", &model.head_dim.to_string()),
                metadata("tritium.rotary_dim", &model.rotary.dimensions.to_string()),
                metadata("tritium.rms_epsilon", &model.rms_epsilon.to_string()),
                metadata(
                    "tritium.full_attention_interval",
                    &full_attention_interval.to_string(),
                ),
                metadata("tritium.layer_schedule", &schedule),
                metadata("tritium.rms_norm_weight_semantics", "zero-centered-offset"),
                metadata(
                    "tritium.tied_embedding_head",
                    if model.lm_head.is_none() {
                        "true"
                    } else {
                        "false"
                    },
                ),
                metadata("tritium.rope_theta", &model.rotary.theta.to_string()),
            ],
        },
        external_weights,
    ))
}

fn build_causal_lm_graph(
    model: CausalLmModel<'_>,
    external: bool,
) -> Result<(ModelProto, Option<Vec<u8>>), OnnxModelError> {
    let tokens = as_i64(model.tokens, "token count")?;
    let past_tokens = as_i64(model.past_tokens, "past token count")?;
    let total_tokens = tokens
        .checked_add(past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let total_tokens_usize = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let hidden = as_i64(model.embedding.columns, "hidden size")?;
    let vocab = as_i64(model.embedding.rows, "vocabulary")?;
    let n_head = as_i64(model.n_head, "attention head count")?;
    let n_kv_head = as_i64(model.n_kv_head, "KV head count")?;
    let head_dim = as_i64(model.head_dim, "head dimension")?;
    let query_width = n_head
        .checked_mul(head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("query width"))?;
    let gqa_repeat = as_i64(model.n_head / model.n_kv_head, "GQA repeat")?;
    let geometry = CausalGraphGeometry {
        past_tokens,
        total_tokens,
        n_head,
        n_kv_head,
        head_dim,
        gqa_repeat,
    };
    let mut graph = if external {
        CausalGraphBuilder::external()
    } else {
        CausalGraphBuilder::default()
    };
    let rotary_state = model
        .rotary
        .map(|rotary| {
            graph.prepare_rotary(RotaryGraphConfig {
                tokens: model.tokens,
                head_dim: model.head_dim,
                rotary_dim: rotary.dimensions,
                past_tokens: model.past_tokens,
                theta: rotary.theta,
            })
        })
        .transpose()?;
    let mask_elements =
        model
            .tokens
            .checked_mul(total_tokens_usize)
            .ok_or(OnnxModelError::ShapeOverflow(
                "attention mask element count",
            ))?;
    if !graph.is_external() && mask_elements > MAX_MODEL_BYTES / core::mem::size_of::<f32>() {
        return Err(OnnxModelError::InvalidModel(format!(
            "attention mask requires {mask_elements} elements, exceeds bounded inline graph"
        )));
    }
    let attention_mask = "attention.attention_mask";
    graph.add_causal_mask(
        attention_mask,
        model.tokens,
        total_tokens_usize,
        model.past_tokens,
        vec![1, tokens, total_tokens],
    );
    graph.add_matrix("tok_embeddings", model.embedding);
    graph.nodes.push(NodeProto {
        input: strings(["tokens", "tok_embeddings.packed", "tok_embeddings.scales"]),
        output: strings(["layer.0.input"]),
        name: "tok_embeddings".to_owned(),
        op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.embedding.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut hidden_name = "layer.0.input".to_owned();
    let mut inputs = vec![tensor_value("tokens", TENSOR_INT64, &[tokens])];
    let mut cache_outputs = Vec::with_capacity(model.layers.len() * 2);

    let full_attention_context = FullAttentionGraphContext {
        tokens,
        n_head,
        n_kv_head,
        head_dim,
        head_dim_usize: model.head_dim,
        query_width,
        geometry,
        rotary_state: rotary_state.as_ref(),
        attention_mask,
        epsilon: model.rms_epsilon,
        zero_centered_norm: model.zero_centered_norm,
    };
    for (index, layer) in model.layers.iter().copied().enumerate() {
        hidden_name = graph.full_attention_block(
            index,
            &hidden_name,
            layer,
            full_attention_context,
            &mut inputs,
            &mut cache_outputs,
        )?;
    }
    let final_hidden = "final_norm.output";
    graph.rms_norm_with_semantics(
        "final_norm",
        &hidden_name,
        final_hidden,
        model.final_norm,
        model.rms_epsilon,
        model.zero_centered_norm,
    );
    let (head_packed, head_scales, head_format) = if let Some(head) = model.lm_head {
        graph.add_matrix("lm_head", head);
        ("lm_head.packed", "lm_head.scales", head.format)
    } else {
        (
            "tok_embeddings.packed",
            "tok_embeddings.scales",
            model.embedding.format,
        )
    };
    graph.nodes.push(NodeProto {
        input: strings([final_hidden, head_packed, head_scales]),
        output: strings(["logits"]),
        name: "lm_head".to_owned(),
        op_type: ONNX_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(head_format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut outputs = vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])];
    outputs.extend(cache_outputs);
    let mut metadata_props = vec![
        metadata("tritium.schema_version", "2"),
        metadata("tritium.graph_kind", "causal-lm"),
        metadata("tritium.source_model_id", model.identity.source_model_id),
        metadata("tritium.tokenizer_id", model.identity.tokenizer_id),
        metadata("tritium.recipe_id", model.identity.recipe_id),
        metadata("tritium.build_id", model.identity.tritium_build_id),
        metadata("tritium.package_id", model.identity.package_id),
        metadata(
            "tritium.coverage.converted_id",
            model.identity.converted_coverage_id,
        ),
        metadata(
            "tritium.coverage.deferred_id",
            model.identity.deferred_coverage_id,
        ),
        metadata("tritium.tokens", &model.tokens.to_string()),
        metadata("tritium.past_tokens", &model.past_tokens.to_string()),
        metadata("tritium.layers", &model.layers.len().to_string()),
        metadata("tritium.hidden", &model.embedding.columns.to_string()),
        metadata("tritium.vocab", &model.embedding.rows.to_string()),
        metadata("tritium.n_head", &model.n_head.to_string()),
        metadata("tritium.n_kv_head", &model.n_kv_head.to_string()),
        metadata("tritium.head_dim", &model.head_dim.to_string()),
        metadata(
            "tritium.query_width",
            &model
                .n_head
                .checked_mul(model.head_dim)
                .expect("validated query width")
                .to_string(),
        ),
        metadata(
            "tritium.rms_norm_weight_semantics",
            if model.zero_centered_norm {
                "zero-centered-offset"
            } else {
                "scale"
            },
        ),
        metadata(
            "tritium.tied_embedding_head",
            if model.lm_head.is_none() {
                "true"
            } else {
                "false"
            },
        ),
    ];
    if let Some(rotary) = model.rotary {
        metadata_props.push(metadata("tritium.rope_theta", &rotary.theta.to_string()));
    }
    if let Some(error) = graph.failure {
        return Err(error);
    }
    let external_weights = external_storage_result(graph.storage)?;
    let protobuf = ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: 2,
        graph: Some(GraphProto {
            node: graph.nodes,
            name: "tritium.causal_lm".to_owned(),
            initializer: graph.initializers,
            input: inputs,
            output: outputs,
            value_info: Vec::new(),
        }),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: TRITIUM_OPSET,
            },
        ],
        metadata_props,
    };
    Ok((protobuf, external_weights))
}

#[cfg(test)]
pub(crate) fn encode_kv_attention_test_graph(
    query_tokens: usize,
    past_tokens: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Vec<u8> {
    let query_tokens = i64::try_from(query_tokens).unwrap();
    let past_tokens = i64::try_from(past_tokens).unwrap();
    let total_tokens = query_tokens.checked_add(past_tokens).unwrap();
    let n_head = i64::try_from(n_head).unwrap();
    let n_kv_head = i64::try_from(n_kv_head).unwrap();
    let head_dim = i64::try_from(head_dim).unwrap();
    let graph = GraphProto {
        node: vec![NodeProto {
            input: strings(["q", "k_cache", "v_cache"]),
            output: strings(["context"]),
            name: "tritium.kv_attention".to_owned(),
            op_type: crate::ONNX_KV_ATTENTION_OP_NAME.to_owned(),
            attribute: vec![
                AttributeProto {
                    name: crate::ATTR_N_HEAD.to_owned(),
                    value: n_head,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_N_KV_HEAD.to_owned(),
                    value: n_kv_head,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_HEAD_DIM.to_owned(),
                    value: head_dim,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_PAST_TOKENS.to_owned(),
                    value: past_tokens,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
            ],
            domain: ONNX_DOMAIN.to_owned(),
        }],
        name: "tritium.kv_attention.test".to_owned(),
        initializer: Vec::new(),
        input: vec![
            tensor_value("q", TENSOR_FLOAT, &[query_tokens, n_head, head_dim]),
            tensor_value(
                "k_cache",
                TENSOR_FLOAT,
                &[total_tokens, n_kv_head, head_dim],
            ),
            tensor_value(
                "v_cache",
                TENSOR_FLOAT,
                &[total_tokens, n_kv_head, head_dim],
            ),
        ],
        output: vec![tensor_value(
            "context",
            TENSOR_FLOAT,
            &[query_tokens, n_head, head_dim],
        )],
        value_info: Vec::new(),
    };
    ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx-test".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: 1,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: ONNX_DOMAIN.to_owned(),
            version: 2,
        }],
        metadata_props: Vec::new(),
    }
    .encode_to_vec()
}

#[cfg(all(test, feature = "onnx"))]
pub(crate) fn encode_qwen_deltanet_test_graph() -> Vec<u8> {
    let input_names = crate::QwenDeltaNetInput::ALL.map(|slot| slot.name().to_owned());
    let output_names = crate::QwenDeltaNetOutputSlot::ALL.map(|slot| slot.name().to_owned());
    let graph = GraphProto {
        node: vec![NodeProto {
            input: input_names.to_vec(),
            output: output_names.to_vec(),
            name: "tritium.qwen_deltanet".to_owned(),
            op_type: crate::ONNX_QWEN_DELTANET_OP_NAME.to_owned(),
            attribute: vec![
                int_attribute(crate::ATTR_CONV_KERNEL_DIM, 2),
                int_attribute(crate::ATTR_NUM_KEY_HEADS, 1),
                int_attribute(crate::ATTR_NUM_VALUE_HEADS, 1),
                int_attribute(crate::ATTR_KEY_HEAD_DIM, 1),
                int_attribute(crate::ATTR_VALUE_HEAD_DIM, 1),
            ],
            domain: ONNX_DOMAIN.to_owned(),
        }],
        name: "tritium.qwen_deltanet_test".to_owned(),
        initializer: Vec::new(),
        input: vec![
            tensor_value(
                crate::QwenDeltaNetInput::RawQkv.name(),
                TENSOR_FLOAT,
                &[1, 3],
            ),
            tensor_value(crate::QwenDeltaNetInput::Z.name(), TENSOR_FLOAT, &[1, 1]),
            tensor_value(
                crate::QwenDeltaNetInput::BetaLogits.name(),
                TENSOR_FLOAT,
                &[1, 1],
            ),
            tensor_value(
                crate::QwenDeltaNetInput::DecayLogits.name(),
                TENSOR_FLOAT,
                &[1, 1],
            ),
            tensor_value(
                crate::QwenDeltaNetInput::ConvWeight.name(),
                TENSOR_FLOAT,
                &[3, 2],
            ),
            tensor_value(
                crate::QwenDeltaNetInput::NormWeight.name(),
                TENSOR_FLOAT,
                &[1],
            ),
            tensor_value(crate::QwenDeltaNetInput::DtBias.name(), TENSOR_FLOAT, &[1]),
            tensor_value(crate::QwenDeltaNetInput::ALog.name(), TENSOR_FLOAT, &[1]),
            tensor_value(
                crate::QwenDeltaNetInput::ConvState.name(),
                TENSOR_FLOAT,
                &[3, 2],
            ),
            tensor_value(
                crate::QwenDeltaNetInput::RecurrentState.name(),
                TENSOR_FLOAT,
                &[1, 1, 1],
            ),
            tensor_value(crate::QwenDeltaNetInput::Epsilon.name(), TENSOR_FLOAT, &[]),
        ],
        output: vec![
            tensor_value(
                crate::QwenDeltaNetOutputSlot::NormalizedCore.name(),
                TENSOR_FLOAT,
                &[1, 1],
            ),
            tensor_value(
                crate::QwenDeltaNetOutputSlot::ConvState.name(),
                TENSOR_FLOAT,
                &[3, 2],
            ),
            tensor_value(
                crate::QwenDeltaNetOutputSlot::RecurrentState.name(),
                TENSOR_FLOAT,
                &[1, 1, 1],
            ),
        ],
        value_info: Vec::new(),
    };
    ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx-test".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: String::new(),
        model_version: 1,
        graph: Some(graph),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: 2,
            },
        ],
        metadata_props: Vec::new(),
    }
    .encode_to_vec()
}

#[cfg(test)]
pub(crate) fn encode_standard_attention_test_graph(
    query_tokens: usize,
    total_tokens: usize,
    head_dim: usize,
) -> Vec<u8> {
    let query_tokens = i64::try_from(query_tokens).unwrap();
    let total_tokens = i64::try_from(total_tokens).unwrap();
    let head_dim_i64 = i64::try_from(head_dim).unwrap();
    let transpose = |name: &str, input: &str, output: &str, permutation: &[i64]| NodeProto {
        input: strings([input]),
        output: strings([output]),
        name: name.to_owned(),
        op_type: "Transpose".to_owned(),
        attribute: vec![AttributeProto {
            name: "perm".to_owned(),
            value: 0,
            kind: ATTRIBUTE_INTS,
            ints: permutation.to_vec(),
        }],
        domain: String::new(),
    };
    let binary = |name: &str, op_type: &str, left: &str, right: &str, output: &str| NodeProto {
        input: strings([left, right]),
        output: strings([output]),
        name: name.to_owned(),
        op_type: op_type.to_owned(),
        attribute: Vec::new(),
        domain: String::new(),
    };
    let graph = GraphProto {
        node: vec![
            transpose(
                "attention.k_transpose",
                "k_cache",
                "k_transposed",
                &[1, 2, 0],
            ),
            transpose(
                "attention.v_transpose",
                "v_cache",
                "v_transposed",
                &[1, 0, 2],
            ),
            binary("attention.scores", "MatMul", "q", "k_transposed", "scores"),
            binary(
                "attention.scale",
                "Mul",
                "scores",
                "attention.scale",
                "scaled",
            ),
            binary(
                "attention.mask",
                "Add",
                "scaled",
                "attention_mask",
                "masked",
            ),
            NodeProto {
                input: strings(["masked"]),
                output: strings(["probabilities"]),
                name: "attention.softmax".to_owned(),
                op_type: "Softmax".to_owned(),
                attribute: vec![int_attribute("axis", -1)],
                domain: String::new(),
            },
            binary(
                "attention.context",
                "MatMul",
                "probabilities",
                "v_transposed",
                "context",
            ),
        ],
        name: "tritium.standard_attention.test".to_owned(),
        initializer: vec![inline_tensor(
            "attention.scale",
            TENSOR_FLOAT,
            Vec::new(),
            (1.0_f32 / (head_dim as f32).sqrt()).to_le_bytes().to_vec(),
        )],
        input: vec![
            tensor_value("q", TENSOR_FLOAT, &[query_tokens, 1, head_dim_i64]),
            tensor_value("k_cache", TENSOR_FLOAT, &[total_tokens, 1, head_dim_i64]),
            tensor_value("v_cache", TENSOR_FLOAT, &[total_tokens, 1, head_dim_i64]),
            tensor_value(
                "attention_mask",
                TENSOR_FLOAT,
                &[query_tokens, 1, total_tokens],
            ),
        ],
        output: vec![tensor_value(
            "context",
            TENSOR_FLOAT,
            &[query_tokens, 1, head_dim_i64],
        )],
        value_info: Vec::new(),
    };
    ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx-test".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: String::new(),
        model_version: 1,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET,
        }],
        metadata_props: Vec::new(),
    }
    .encode_to_vec()
}

fn validate(model: &TiedEmbeddingHeadModel<'_>) -> Result<(), OnnxModelError> {
    for (name, dimension) in [
        ("token count", model.tokens),
        ("vocabulary", model.vocab),
        ("hidden size", model.hidden),
    ] {
        if dimension == 0 {
            return Err(OnnxModelError::EmptyDimension(name));
        }
    }
    for (name, identity) in [
        ("source_model_id", model.source_model_id),
        ("recipe_id", model.recipe_id),
        ("package_id", model.package_id),
    ] {
        if identity.is_empty() {
            return Err(OnnxModelError::EmptyIdentity(name));
        }
    }
    if model.scales.len() != model.vocab {
        return Err(OnnxModelError::ScaleCount {
            expected: model.vocab,
            got: model.scales.len(),
        });
    }
    if let Some((index, _)) = model
        .scales
        .iter()
        .enumerate()
        .find(|(_, scale)| !scale.is_finite() || **scale < 0.0)
    {
        return Err(OnnxModelError::InvalidScale { index });
    }
    let block_bytes = match model.format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let expected = num_blocks(model.hidden)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(model.vocab))
        .ok_or(OnnxModelError::ShapeOverflow("packed byte count"))?;
    if model.packed.len() != expected {
        return Err(OnnxModelError::PackedBytes {
            expected,
            got: model.packed.len(),
        });
    }
    validate_packed_payload(model)?;
    for (name, dimension) in [
        ("token count", model.tokens),
        ("vocabulary", model.vocab),
        ("hidden size", model.hidden),
        ("packed byte count", model.packed.len()),
    ] {
        as_i64(dimension, name)?;
    }
    Ok(())
}

fn validate_causal_lm(model: &CausalLmModel<'_>) -> Result<(), OnnxModelError> {
    if model.layers.is_empty() {
        return Err(OnnxModelError::InvalidModel(
            "causal LM requires at least one decoder layer".to_owned(),
        ));
    }
    for (name, value) in [
        ("token count", model.tokens),
        ("attention head count", model.n_head),
        ("KV head count", model.n_kv_head),
        ("head dimension", model.head_dim),
        ("vocabulary", model.embedding.rows),
        ("hidden size", model.embedding.columns),
    ] {
        if value == 0 {
            return Err(OnnxModelError::EmptyDimension(name));
        }
        as_i64(value, name)?;
    }
    as_i64(model.past_tokens, "past token count")?;
    if !model.rms_epsilon.is_finite() || model.rms_epsilon <= 0.0 {
        return Err(OnnxModelError::InvalidModel(
            "RMSNorm epsilon must be finite and positive".to_owned(),
        ));
    }
    if !model.n_head.is_multiple_of(model.n_kv_head) {
        return Err(OnnxModelError::InvalidModel(format!(
            "n_head {} is not divisible by n_kv_head {}",
            model.n_head, model.n_kv_head
        )));
    }
    let query_width = model
        .n_head
        .checked_mul(model.head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("query width"))?;
    let hidden = model.embedding.columns;
    if let Some(rotary) = model.rotary {
        if !rotary.theta.is_finite() || rotary.theta <= 0.0 {
            return Err(OnnxModelError::InvalidModel(
                "RoPE theta must be finite and positive".to_owned(),
            ));
        }
        if rotary.dimensions < 2
            || !rotary.dimensions.is_multiple_of(2)
            || rotary.dimensions > model.head_dim
        {
            return Err(OnnxModelError::InvalidModel(format!(
                "RoPE dimensions must be positive, even, and at most head_dim; got {} for {}",
                rotary.dimensions, model.head_dim
            )));
        }
    }
    validate_matrix("embedding", model.embedding)?;
    if let Some(head) = model.lm_head {
        validate_matrix_shape("lm_head", head, model.embedding.rows, hidden)?;
    }
    validate_vector("final_norm", model.final_norm, hidden)?;
    for (index, layer) in model.layers.iter().copied().enumerate() {
        validate_vector(
            &format!("layer.{index}.attention_norm"),
            layer.attention_norm,
            hidden,
        )?;
        if let Some(weight) = layer.query_norm {
            validate_vector(&format!("layer.{index}.query_norm"), weight, model.head_dim)?;
        }
        if let Some(weight) = layer.key_norm {
            validate_vector(&format!("layer.{index}.key_norm"), weight, model.head_dim)?;
        }
        validate_vector(&format!("layer.{index}.ffn_norm"), layer.ffn_norm, hidden)?;
        let kv_width = model
            .n_kv_head
            .checked_mul(model.head_dim)
            .ok_or(OnnxModelError::ShapeOverflow("KV projection width"))?;
        match layer.query {
            CausalQueryProjection::HeadInterleavedQueryGate { fused } => {
                let fused_width = query_width
                    .checked_mul(2)
                    .ok_or(OnnxModelError::ShapeOverflow("fused query/gate width"))?;
                validate_matrix_shape(
                    &format!("layer.{index}.fused_query_gate"),
                    fused,
                    fused_width,
                    hidden,
                )?;
            }
            CausalQueryProjection::Separate { query, gate } => {
                validate_matrix_shape(&format!("layer.{index}.query"), query, query_width, hidden)?;
                if let Some(gate) = gate {
                    validate_matrix_shape(
                        &format!("layer.{index}.attention_gate"),
                        gate,
                        query_width,
                        hidden,
                    )?;
                }
            }
        }
        validate_matrix_shape(&format!("layer.{index}.key"), layer.key, kv_width, hidden)?;
        validate_matrix_shape(
            &format!("layer.{index}.value"),
            layer.value,
            kv_width,
            hidden,
        )?;
        validate_matrix_shape(
            &format!("layer.{index}.attention_output"),
            layer.attention_output,
            hidden,
            query_width,
        )?;
        if let Some(weight) = layer.attention_sub_norm {
            validate_vector(
                &format!("layer.{index}.attention_sub_norm"),
                weight,
                query_width,
            )?;
        }
        if layer.gate.rows == 0 || layer.gate.rows != layer.up.rows {
            return Err(OnnxModelError::InvalidModel(format!(
                "layer.{index} gate/up intermediate widths disagree"
            )));
        }
        let intermediate = layer.gate.rows;
        if let Some(weight) = layer.ffn_sub_norm {
            validate_vector(&format!("layer.{index}.ffn_sub_norm"), weight, intermediate)?;
        }
        validate_matrix_shape(
            &format!("layer.{index}.gate"),
            layer.gate,
            intermediate,
            hidden,
        )?;
        validate_matrix_shape(&format!("layer.{index}.up"), layer.up, intermediate, hidden)?;
        validate_matrix_shape(
            &format!("layer.{index}.down"),
            layer.down,
            hidden,
            intermediate,
        )?;
    }
    validate_identity_v2(model.identity)
}

fn validate_qwen_deltanet_layer(
    model: &QwenDeltaNetLayerModel<'_>,
) -> Result<crate::QwenDeltaNetDimensions, OnnxModelError> {
    for (name, value) in [
        ("Qwen DeltaNet token count", model.tokens),
        ("Qwen DeltaNet hidden width", model.hidden),
    ] {
        if value == 0 {
            return Err(OnnxModelError::EmptyDimension(name));
        }
        as_i64(value, name)?;
    }
    let dimensions = model
        .geometry
        .dimensions()
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    if !model.rms_epsilon.is_finite() || model.rms_epsilon <= 0.0 {
        return Err(OnnxModelError::InvalidModel(
            "Qwen DeltaNet RMSNorm epsilon must be finite and positive".to_owned(),
        ));
    }
    let value_width = dimensions.value_width();
    let conv_width = dimensions.conv_width();
    validate_vector(
        "layer.0.attention_norm",
        model.layer.attention_norm,
        model.hidden,
    )?;
    validate_matrix_shape("layer.0.qkv", model.layer.qkv, conv_width, model.hidden)?;
    validate_matrix_shape("layer.0.z", model.layer.z, value_width, model.hidden)?;
    validate_matrix_shape(
        "layer.0.beta",
        model.layer.beta,
        model.geometry.num_value_heads(),
        model.hidden,
    )?;
    validate_matrix_shape(
        "layer.0.decay",
        model.layer.decay,
        model.geometry.num_value_heads(),
        model.hidden,
    )?;
    let conv_weight_len = conv_width
        .checked_mul(model.geometry.conv_kernel_dim())
        .ok_or(OnnxModelError::ShapeOverflow(
            "DeltaNet convolution weights",
        ))?;
    validate_vector(
        "layer.0.conv_weight",
        model.layer.conv_weight,
        conv_weight_len,
    )?;
    validate_vector(
        "layer.0.norm_weight",
        model.layer.norm_weight,
        model.geometry.value_head_dim(),
    )?;
    validate_vector(
        "layer.0.dt_bias",
        model.layer.dt_bias,
        model.geometry.num_value_heads(),
    )?;
    validate_vector(
        "layer.0.a_log",
        model.layer.a_log,
        model.geometry.num_value_heads(),
    )?;
    validate_matrix_shape(
        "layer.0.output",
        model.layer.output,
        model.hidden,
        value_width,
    )?;
    validate_vector("layer.0.ffn_norm", model.layer.ffn_norm, model.hidden)?;
    if model.layer.gate.rows == 0 || model.layer.gate.rows != model.layer.up.rows {
        return Err(OnnxModelError::InvalidModel(
            "Qwen DeltaNet gate/up intermediate widths disagree".to_owned(),
        ));
    }
    let intermediate = model.layer.gate.rows;
    validate_matrix_shape("layer.0.gate", model.layer.gate, intermediate, model.hidden)?;
    validate_matrix_shape("layer.0.up", model.layer.up, intermediate, model.hidden)?;
    validate_matrix_shape("layer.0.down", model.layer.down, model.hidden, intermediate)?;
    validate_identity_v2(model.identity)?;
    Ok(dimensions)
}

fn validate_qwen35_mtp_model(model: &Qwen35MtpModel<'_>) -> Result<(), OnnxModelError> {
    let layer = [model.mtp.layer.causal()];
    validate_causal_lm(&CausalLmModel {
        tokens: model.tokens,
        past_tokens: model.past_tokens,
        n_head: model.n_head,
        n_kv_head: model.n_kv_head,
        head_dim: model.head_dim,
        rotary: Some(model.rotary),
        rms_epsilon: model.rms_epsilon,
        zero_centered_norm: true,
        embedding: model.embedding,
        lm_head: Some(model.lm_head),
        layers: &layer,
        final_norm: model.mtp.final_norm,
        identity: model.identity,
    })?;
    let hidden = model.embedding.columns;
    validate_vector(
        "Qwen MTP pre_fc_norm_embedding",
        model.mtp.pre_fc_norm_embedding,
        hidden,
    )?;
    validate_vector(
        "Qwen MTP pre_fc_norm_hidden",
        model.mtp.pre_fc_norm_hidden,
        hidden,
    )?;
    validate_matrix_shape(
        "Qwen MTP fusion",
        model.mtp.fusion,
        hidden,
        hidden
            .checked_mul(2)
            .ok_or(OnnxModelError::ShapeOverflow("Qwen MTP fusion width"))?,
    )
}

fn validate_qwen35_bundle_pair(
    language: &QwenCausalLmModel<'_>,
    mtp: &Qwen35MtpModel<'_>,
) -> Result<(), OnnxModelError> {
    validate_qwen_causal_lm(language)?;
    validate_qwen35_mtp_model(mtp)?;
    let same_identity = language.identity.source_model_id == mtp.identity.source_model_id
        && language.identity.tokenizer_id == mtp.identity.tokenizer_id
        && language.identity.recipe_id == mtp.identity.recipe_id
        && language.identity.tritium_build_id == mtp.identity.tritium_build_id
        && language.identity.package_id == mtp.identity.package_id
        && language.identity.converted_coverage_id == mtp.identity.converted_coverage_id
        && language.identity.deferred_coverage_id == mtp.identity.deferred_coverage_id;
    if !same_identity {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language and MTP artifact identities differ".to_owned(),
        ));
    }
    if language.tokens != mtp.tokens
        || language.past_tokens != mtp.past_tokens
        || language.n_head != mtp.n_head
        || language.n_kv_head != mtp.n_kv_head
        || language.head_dim != mtp.head_dim
        || language.rotary.dimensions != mtp.rotary.dimensions
        || language.rotary.theta.to_bits() != mtp.rotary.theta.to_bits()
        || language.rms_epsilon.to_bits() != mtp.rms_epsilon.to_bits()
    {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language and MTP execution geometry differs".to_owned(),
        ));
    }
    let same_matrix = |left: PackedTernaryMatrix<'_>, right: PackedTernaryMatrix<'_>| {
        left.rows == right.rows
            && left.columns == right.columns
            && left.format == right.format
            && left.packed == right.packed
            && left.scales == right.scales
    };
    if !same_matrix(language.embedding, mtp.embedding) {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language and MTP shared embeddings differ".to_owned(),
        ));
    }
    let Some(language_head) = language.lm_head else {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language-plus-MTP bundle requires an untied language head".to_owned(),
        ));
    };
    if !same_matrix(language_head, mtp.lm_head) {
        return Err(OnnxModelError::InvalidModel(
            "Qwen language and MTP language heads differ".to_owned(),
        ));
    }
    Ok(())
}

fn validate_qwen_causal_lm(model: &QwenCausalLmModel<'_>) -> Result<(), OnnxModelError> {
    if model.layers.is_empty() {
        return Err(OnnxModelError::InvalidModel(
            "Qwen causal LM requires at least one layer".to_owned(),
        ));
    }
    let mut has_delta = false;
    let mut has_full = false;
    for layer in model.layers.iter().copied() {
        match layer {
            QwenCausalLmDecoderLayer::DeltaNet(layer) => {
                has_delta = true;
                validate_qwen_deltanet_layer(&QwenDeltaNetLayerModel {
                    tokens: model.tokens,
                    hidden: model.embedding.columns,
                    rms_epsilon: model.rms_epsilon,
                    geometry: model.delta_geometry,
                    layer,
                    identity: model.identity,
                })?;
            }
            QwenCausalLmDecoderLayer::FullAttention(layer) => {
                has_full = true;
                let singleton = [layer.causal()];
                validate_causal_lm(&CausalLmModel {
                    tokens: model.tokens,
                    past_tokens: model.past_tokens,
                    n_head: model.n_head,
                    n_kv_head: model.n_kv_head,
                    head_dim: model.head_dim,
                    rotary: Some(model.rotary),
                    rms_epsilon: model.rms_epsilon,
                    zero_centered_norm: true,
                    embedding: model.embedding,
                    lm_head: model.lm_head,
                    layers: &singleton,
                    final_norm: model.final_norm,
                    identity: model.identity,
                })?;
            }
        }
    }
    if !has_delta || !has_full {
        return Err(OnnxModelError::InvalidModel(
            "Qwen heterogeneous schedule requires DeltaNet and full-attention layers".to_owned(),
        ));
    }
    qwen_full_attention_interval(model.layers)?;
    Ok(())
}

fn qwen_full_attention_interval(
    layers: &[QwenCausalLmDecoderLayer<'_>],
) -> Result<usize, OnnxModelError> {
    let interval = layers
        .iter()
        .position(|layer| matches!(layer, QwenCausalLmDecoderLayer::FullAttention(_)))
        .map(|index| index + 1)
        .ok_or_else(|| {
            OnnxModelError::InvalidModel("Qwen schedule lacks full attention".to_owned())
        })?;
    if !layers.len().is_multiple_of(interval)
        || layers.iter().enumerate().any(|(index, layer)| {
            let expected_full = (index + 1).is_multiple_of(interval);
            matches!(layer, QwenCausalLmDecoderLayer::FullAttention(_)) != expected_full
        })
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "Qwen layer schedule differs from canonical full-attention interval {interval}"
        )));
    }
    Ok(interval)
}

fn validate_qwen_deltanet_initializer_budget(
    model: &QwenDeltaNetLayerModel<'_>,
) -> Result<(), OnnxModelError> {
    let mut total = 0_usize;
    let mut add = |bytes: usize| -> Result<(), OnnxModelError> {
        total = total
            .checked_add(bytes)
            .ok_or(OnnxModelError::ShapeOverflow(
                "Qwen DeltaNet initializer byte count",
            ))?;
        if total > MAX_MODEL_BYTES {
            return Err(OnnxModelError::InvalidModel(format!(
                "Qwen DeltaNet initializers require at least {total} bytes, exceed bounded inline graph"
            )));
        }
        Ok(())
    };
    for matrix in [
        model.layer.qkv,
        model.layer.z,
        model.layer.beta,
        model.layer.decay,
        model.layer.output,
        model.layer.gate,
        model.layer.up,
        model.layer.down,
    ] {
        add(matrix.packed.len())?;
        add(matrix
            .scales
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(OnnxModelError::ShapeOverflow(
                "Qwen DeltaNet scale byte count",
            ))?)?;
    }
    for vector in [
        model.layer.attention_norm,
        model.layer.conv_weight,
        model.layer.norm_weight,
        model.layer.dt_bias,
        model.layer.a_log,
        model.layer.ffn_norm,
    ] {
        add(vector
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(OnnxModelError::ShapeOverflow(
                "Qwen DeltaNet preserved byte count",
            ))?)?;
    }
    Ok(())
}

fn validate_causal_initializer_budget(model: &CausalLmModel<'_>) -> Result<(), OnnxModelError> {
    fn add(total: &mut usize, bytes: usize) -> Result<(), OnnxModelError> {
        *total = total
            .checked_add(bytes)
            .ok_or(OnnxModelError::ShapeOverflow(
                "causal initializer byte count",
            ))?;
        if *total > MAX_MODEL_BYTES {
            return Err(OnnxModelError::InvalidModel(format!(
                "causal initializers require at least {total} bytes, exceed bounded inline graph"
            )));
        }
        Ok(())
    }
    fn add_vector(total: &mut usize, values: &[f32]) -> Result<(), OnnxModelError> {
        let bytes = values
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(OnnxModelError::ShapeOverflow("preserved vector byte count"))?;
        add(total, bytes)
    }
    fn add_matrix(
        total: &mut usize,
        matrix: PackedTernaryMatrix<'_>,
    ) -> Result<(), OnnxModelError> {
        add(total, matrix.packed.len())?;
        add_vector(total, matrix.scales)
    }

    let mut total = 0;
    add_matrix(&mut total, model.embedding)?;
    if let Some(head) = model.lm_head {
        add_matrix(&mut total, head)?;
    }
    add_vector(&mut total, model.final_norm)?;
    for layer in model.layers.iter().copied() {
        add_vector(&mut total, layer.attention_norm)?;
        if let Some(weight) = layer.query_norm {
            add_vector(&mut total, weight)?;
        }
        if let Some(weight) = layer.key_norm {
            add_vector(&mut total, weight)?;
        }
        if let Some(weight) = layer.attention_sub_norm {
            add_vector(&mut total, weight)?;
        }
        match layer.query {
            CausalQueryProjection::Separate { query, gate } => {
                add_matrix(&mut total, query)?;
                if let Some(gate) = gate {
                    add_matrix(&mut total, gate)?;
                }
            }
            CausalQueryProjection::HeadInterleavedQueryGate { fused } => {
                add_matrix(&mut total, fused)?;
            }
        }
        add_vector(&mut total, layer.ffn_norm)?;
        if let Some(weight) = layer.ffn_sub_norm {
            add_vector(&mut total, weight)?;
        }
        for matrix in [
            layer.key,
            layer.value,
            layer.attention_output,
            layer.gate,
            layer.up,
            layer.down,
        ] {
            add_matrix(&mut total, matrix)?;
        }
    }
    let total_tokens = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let mask_bytes = model
        .tokens
        .checked_mul(total_tokens)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(OnnxModelError::ShapeOverflow("attention mask byte count"))?;
    add(&mut total, mask_bytes)?;
    if let Some(rotary) = model.rotary {
        let rotary_bytes = model
            .tokens
            .checked_mul(rotary.dimensions)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(OnnxModelError::ShapeOverflow("RoPE table byte count"))?;
        add(&mut total, rotary_bytes)?;
    }
    Ok(())
}

fn validate_matrix_shape(
    name: &str,
    matrix: PackedTernaryMatrix<'_>,
    rows: usize,
    columns: usize,
) -> Result<(), OnnxModelError> {
    if matrix.rows != rows || matrix.columns != columns {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} shape [{}, {}] must be [{rows}, {columns}]",
            matrix.rows, matrix.columns
        )));
    }
    validate_matrix(name, matrix)
}

fn validate_matrix(name: &str, matrix: PackedTernaryMatrix<'_>) -> Result<(), OnnxModelError> {
    if matrix.rows == 0 || matrix.columns == 0 {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} matrix dimensions must be nonzero"
        )));
    }
    validate(&TiedEmbeddingHeadModel {
        tokens: 1,
        vocab: matrix.rows,
        hidden: matrix.columns,
        packed: matrix.packed,
        scales: matrix.scales,
        format: matrix.format,
        source_model_id: "matrix-validation",
        recipe_id: "matrix-validation",
        package_id: "matrix-validation",
    })
    .map_err(|error| OnnxModelError::InvalidModel(format!("{name}: {error}")))
}

fn validate_vector(name: &str, values: &[f32], expected: usize) -> Result<(), OnnxModelError> {
    if values.len() != expected {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name}[{index}] must be finite"
        )));
    }
    Ok(())
}

fn validate_v2(model: &TiedEmbeddingHeadModelV2<'_>) -> Result<(), OnnxModelError> {
    validate(&model.legacy())?;
    validate_identity_v2(model.identity)
}

fn validate_identity_v2(identity: OnnxArtifactIdentityV2<'_>) -> Result<(), OnnxModelError> {
    for (name, identity) in [
        ("source_model_id", identity.source_model_id),
        ("tokenizer_id", identity.tokenizer_id),
        ("recipe_id", identity.recipe_id),
        ("tritium_build_id", identity.tritium_build_id),
        ("package_id", identity.package_id),
        ("converted_coverage_id", identity.converted_coverage_id),
        ("deferred_coverage_id", identity.deferred_coverage_id),
    ] {
        if identity.is_empty() {
            return Err(OnnxModelError::EmptyIdentity(name));
        }
    }
    Ok(())
}

fn identity_metadata_v2(identity: OnnxArtifactIdentityV2<'_>) -> Vec<StringStringEntryProto> {
    vec![
        metadata("tritium.tokenizer_id", identity.tokenizer_id),
        metadata("tritium.build_id", identity.tritium_build_id),
        metadata(
            "tritium.coverage.converted_id",
            identity.converted_coverage_id,
        ),
        metadata(
            "tritium.coverage.deferred_id",
            identity.deferred_coverage_id,
        ),
    ]
}

fn validate_packed_payload(model: &TiedEmbeddingHeadModel<'_>) -> Result<(), OnnxModelError> {
    let block_bytes = match model.format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let blocks = num_blocks(model.hidden);
    let row_bytes = blocks
        .checked_mul(block_bytes)
        .ok_or(OnnxModelError::ShapeOverflow("packed row byte count"))?;
    let mut trits = vec![Trit::ZERO; model.hidden];
    let mut scales = vec![f16::ONE; blocks];
    for (row, packed) in model.packed.chunks_exact(row_bytes).enumerate() {
        let result = match model.format {
            TernaryFormat::Tq2_0 => unpack_tq2_0_row(packed, &mut trits, &mut scales),
            TernaryFormat::Tq1_0 => unpack_tq1_0_row(packed, &mut trits, &mut scales),
            other => return Err(OnnxModelError::UnsupportedFormat(other)),
        };
        result.map_err(|error| OnnxModelError::InvalidPackedRow {
            row,
            reason: error.to_string(),
        })?;
        if let Some((block, scale)) = scales
            .iter()
            .copied()
            .enumerate()
            .find(|(_, scale)| *scale != f16::ONE)
        {
            return Err(OnnxModelError::InvalidPackedRow {
                row,
                reason: format!("block {block} has non-unit scale {scale:?}"),
            });
        }
    }
    Ok(())
}

fn as_i64(value: usize, name: &'static str) -> Result<i64, OnnxModelError> {
    i64::try_from(value).map_err(|_| OnnxModelError::ShapeOverflow(name))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, OnnxModelError> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned / alignment * alignment)
        .ok_or(OnnxModelError::ShapeOverflow("external data alignment"))
}

fn scale_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|scale| scale.to_le_bytes())
        .collect()
}

fn format_code(format: TernaryFormat) -> Result<i64, OnnxModelError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(0),
        TernaryFormat::Tq1_0 => Ok(1),
        other => Err(OnnxModelError::UnsupportedFormat(other)),
    }
}

fn strings<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.into_iter().map(str::to_owned).collect()
}

fn attributes(k: i64, format: i64) -> Vec<AttributeProto> {
    vec![int_attribute(ATTR_K, k), int_attribute(ATTR_FORMAT, format)]
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        value,
        kind: ATTRIBUTE_INT,
        ints: Vec::new(),
    }
}

fn ints_attribute(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        value: 0,
        kind: ATTRIBUTE_INTS,
        ints: values.to_vec(),
    }
}

fn metadata(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.to_owned(),
        value: value.to_owned(),
    }
}

fn inline_tensor(name: &str, data_type: i32, dims: Vec<i64>, raw_data: Vec<u8>) -> TensorProto {
    TensorProto {
        dims,
        data_type,
        name: name.to_owned(),
        raw_data,
        external_data: Vec::new(),
        data_location: 0,
    }
}

fn external_tensor(
    name: &str,
    data_type: i32,
    dims: Vec<i64>,
    offset: i64,
    length: i64,
) -> TensorProto {
    TensorProto {
        dims,
        data_type,
        name: name.to_owned(),
        raw_data: Vec::new(),
        external_data: vec![
            metadata("location", EXTERNAL_WEIGHTS_FILE),
            metadata("offset", &offset.to_string()),
            metadata("length", &length.to_string()),
        ],
        data_location: EXTERNAL_DATA,
    }
}

fn tensor_value(name: &str, elem_type: i32, dimensions: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type,
                shape: Some(TensorShapeProto {
                    dim: dimensions
                        .iter()
                        .map(|&dim_value| TensorDimensionProto { dim_value })
                        .collect(),
                }),
            }),
        }),
    }
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(string, tag = "2")]
    producer_name: String,
    #[prost(string, tag = "3")]
    producer_version: String,
    #[prost(string, tag = "4")]
    domain: String,
    #[prost(int64, tag = "5")]
    model_version: i64,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, repeated, tag = "14")]
    metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    value_info: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    domain: String,
}

#[derive(Clone, PartialEq, Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(int64, tag = "3")]
    value: i64,
    #[prost(int32, tag = "20")]
    kind: i32,
    #[prost(int64, repeated, packed = "true", tag = "8")]
    ints: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    data_location: i32,
}

#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<TensorDimensionProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorDimensionProto {
    #[prost(int64, tag = "1")]
    dim_value: i64,
}

#[derive(Clone, PartialEq, Message)]
struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};

    fn unit_packed(format: TernaryFormat, rows: usize) -> Vec<u8> {
        unit_packed_shape(format, rows, 256)
    }

    fn unit_packed_shape(format: TernaryFormat, rows: usize, columns: usize) -> Vec<u8> {
        let blocks = num_blocks(columns);
        let trits = vec![Trit::ZERO; columns];
        let scales = vec![f16::ONE; blocks];
        let block_bytes = match format {
            TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
            TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
            other => panic!("unsupported test format {other}"),
        };
        let row_bytes = block_bytes * blocks;
        let mut packed = vec![0; row_bytes * rows];
        for row in packed.chunks_exact_mut(row_bytes) {
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(&trits, &scales, row).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(&trits, &scales, row).unwrap(),
                other => panic!("unsupported test format {other}"),
            }
        }
        packed
    }

    #[test]
    fn validation_is_fail_closed() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let base = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        assert!(encode_tied_embedding_head(base).is_ok());
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                scales: &[f32::NAN],
                ..base
            }),
            Err(OnnxModelError::InvalidScale { .. })
        ));
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                packed: &[],
                ..base
            }),
            Err(OnnxModelError::PackedBytes { .. })
        ));
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                source_model_id: "",
                ..base
            }),
            Err(OnnxModelError::EmptyIdentity("source_model_id"))
        ));
    }

    #[test]
    fn encoding_is_deterministic() {
        let packed = unit_packed(TernaryFormat::Tq1_0, 2);
        let model = TiedEmbeddingHeadModel {
            tokens: 3,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq1_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        assert_eq!(
            encode_tied_embedding_head(model).unwrap(),
            encode_tied_embedding_head(model).unwrap()
        );
    }

    #[test]
    fn external_encoding_binds_layout_and_digest() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 2);
        let model = TiedEmbeddingHeadModel {
            tokens: 2,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let encoded = encode_external_tied_embedding_head(model).unwrap();
        let receipt = verify_external_tied_embedding_head(
            encoded.model_bytes.as_slice(),
            encoded.weights_bytes.as_slice(),
        )
        .unwrap();
        assert_eq!(receipt.tokens, 2);
        assert_eq!(receipt.vocab, 2);
        assert_eq!(receipt.hidden, 256);
        assert_eq!(
            encoded.weights_blake3,
            *blake3::hash(&encoded.weights_bytes).as_bytes()
        );
        assert_eq!(encoded.weights_bytes.len() % 4, 0);
        let protobuf = ModelProto::decode(encoded.model_bytes.as_slice()).unwrap();
        let graph = protobuf.graph.unwrap();
        assert_eq!(graph.initializer.len(), 2);
        assert!(graph.initializer.iter().all(|tensor| {
            tensor.raw_data.is_empty()
                && tensor.data_location == EXTERNAL_DATA
                && tensor
                    .external_data
                    .iter()
                    .any(|entry| entry.key == "location" && entry.value == EXTERNAL_WEIGHTS_FILE)
        }));
        let expected_digest = blake3::Hash::from_bytes(encoded.weights_blake3)
            .to_hex()
            .to_string();
        assert!(protobuf.metadata_props.iter().any(|entry| {
            entry.key == "tritium.external_data.blake3" && entry.value == expected_digest
        }));

        let mut corrupted = encoded.weights_bytes.clone();
        corrupted[0] ^= 1;
        assert!(matches!(
            verify_external_tied_embedding_head(&encoded.model_bytes, &corrupted),
            Err(OnnxModelError::ExternalDataMismatch(_))
        ));
    }

    #[test]
    fn schema_v2_binds_complete_identity_without_breaking_v1() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 2);
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "source@revision",
            tokenizer_id: "tokenizer@revision",
            recipe_id: "recipe@digest",
            tritium_build_id: "tritium@git-sha",
            package_id: "package@digest",
            converted_coverage_id: "converted@digest",
            deferred_coverage_id: "deferred@digest",
        };
        let model = TiedEmbeddingHeadModelV2 {
            tokens: 2,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq2_0,
            identity,
        };
        let encoded = encode_external_tied_embedding_head_v2(model).unwrap();
        let receipt = verify_external_tied_embedding_head_v2(
            &encoded.model_bytes,
            &encoded.weights_bytes,
            identity,
        )
        .unwrap();
        assert_eq!(receipt.model.source_model_id, identity.source_model_id);
        assert_eq!(receipt.model.recipe_id, identity.recipe_id);
        assert_eq!(receipt.model.package_id, identity.package_id);
        assert_eq!(receipt.identity.tokenizer_id, identity.tokenizer_id);
        assert_eq!(receipt.identity.tritium_build_id, identity.tritium_build_id);
        assert_eq!(
            receipt.identity.converted_coverage_id,
            identity.converted_coverage_id
        );
        assert_eq!(
            receipt.identity.deferred_coverage_id,
            identity.deferred_coverage_id
        );
        assert!(
            verify_external_tied_embedding_head(&encoded.model_bytes, &encoded.weights_bytes)
                .is_err()
        );

        let mut protobuf = ModelProto::decode(encoded.model_bytes.as_slice()).unwrap();
        protobuf
            .metadata_props
            .retain(|entry| entry.key != "tritium.tokenizer_id");
        assert!(matches!(
            verify_external_tied_embedding_head_v2(
                &protobuf.encode_to_vec(),
                &encoded.weights_bytes,
                identity,
            ),
            Err(OnnxModelError::InvalidModel(_))
        ));

        let legacy = encode_external_tied_embedding_head(model.legacy()).unwrap();
        assert!(
            verify_external_tied_embedding_head(&legacy.model_bytes, &legacy.weights_bytes).is_ok()
        );
        assert!(
            verify_external_tied_embedding_head_v2(
                &legacy.model_bytes,
                &legacy.weights_bytes,
                identity,
            )
            .is_err()
        );
        assert!(matches!(
            verify_external_tied_embedding_head_v2(
                &encoded.model_bytes,
                &encoded.weights_bytes,
                OnnxArtifactIdentityV2 {
                    package_id: "different-package",
                    ..identity
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("package_id")
        ));
        assert!(matches!(
            encode_tied_embedding_head_v2(TiedEmbeddingHeadModelV2 {
                identity: OnnxArtifactIdentityV2 {
                    tokenizer_id: "",
                    ..identity
                },
                ..model
            }),
            Err(OnnxModelError::EmptyIdentity("tokenizer_id"))
        ));
    }

    #[test]
    fn unsupported_graph_diagnostics_are_typed_and_exhaustive() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModelV2 {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            identity: OnnxArtifactIdentityV2 {
                source_model_id: "source",
                tokenizer_id: "tokenizer",
                recipe_id: "recipe",
                tritium_build_id: "build",
                package_id: "package",
                converted_coverage_id: "converted",
                deferred_coverage_id: "deferred",
            },
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head_v2(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0].domain = "ai.onnx".to_owned();
        graph.node[0].op_type = "Gather".to_owned();
        graph.node[1].attribute.push(AttributeProto {
            name: "axis".to_owned(),
            value: 0,
            kind: ATTRIBUTE_INT,
            ints: Vec::new(),
        });
        graph.initializer[0].data_type = TENSOR_INT64;
        protobuf.metadata_props.push(metadata(
            "tritium.coverage.unresolved.language.layers.0.mlp",
            "preserved",
        ));

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Node,
                    subject: "tritium.embedding".to_owned(),
                    reason: "unsupported operator ai.onnx::Gather".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.format".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.axis".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "initializer tritium.packed".to_owned(),
                    reason: "expected ONNX dtype 2, got 7".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Coverage,
                    subject: "language.layers.0.mlp".to_owned(),
                    reason: "coverage item is unresolved: preserved".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_reject_invalid_attribute_values() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0]
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == ATTR_K)
            .unwrap()
            .value = 0;
        graph.node[1]
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == ATTR_FORMAT)
            .unwrap()
            .value = 9;

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "K must be positive, got 0".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.format".to_owned(),
                    reason: "unsupported format code 9".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_name_missing_and_duplicate_attributes() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0]
            .attribute
            .retain(|attribute| attribute.name != ATTR_K);
        let duplicate_format = graph.node[1]
            .attribute
            .iter()
            .find(|attribute| attribute.name == ATTR_FORMAT)
            .unwrap()
            .clone();
        graph.node[1].attribute.push(duplicate_format);

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "missing required attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.format".to_owned(),
                    reason: "duplicate attribute appears 2 times".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_cover_every_typed_tensor_declaration() {
        const ONNX_TENSOR_STRING: i32 = 8;
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        let mut extra_initializer = graph.initializer[0].clone();
        extra_initializer.name = "extra".to_owned();
        extra_initializer.data_type = ONNX_TENSOR_STRING;
        graph.initializer.push(extra_initializer);
        graph
            .input
            .push(tensor_value("extra_input", ONNX_TENSOR_STRING, &[1]));
        graph
            .output
            .push(tensor_value("extra_output", ONNX_TENSOR_STRING, &[1]));
        graph
            .value_info
            .push(tensor_value("hidden", ONNX_TENSOR_STRING, &[1, 256]));
        graph
            .value_info
            .push(tensor_value("scratch", ONNX_TENSOR_STRING, &[1]));

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "initializer extra".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "input extra_input".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "output extra_output".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "value_info hidden".to_owned(),
                    reason: "expected ONNX dtype 1, got 8".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "value_info scratch".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn supported_graph_has_no_unsupported_diagnostics() {
        let packed = unit_packed(TernaryFormat::Tq1_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq1_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let encoded = encode_tied_embedding_head(model).unwrap();
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn supported_kv_attention_graph_has_no_unsupported_diagnostics() {
        let encoded = encode_kv_attention_test_graph(1, 2, 2, 1, 4);
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn supported_standard_attention_graph_has_no_unsupported_diagnostics() {
        let encoded = encode_standard_attention_test_graph(2, 2, 1);
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn standard_attention_diagnostics_reject_invalid_permutation_and_softmax_axis() {
        let encoded = encode_standard_attention_test_graph(2, 2, 1);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0].attribute[0].ints = vec![0, 0, 2];
        graph
            .node
            .iter_mut()
            .find(|node| node.op_type == "Softmax")
            .unwrap()
            .attribute[0]
            .value = 0;

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "attention.k_transpose.perm".to_owned(),
                    reason: "expected a rank-3 permutation of [0, 1, 2], got [0, 0, 2]".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "attention.softmax.axis".to_owned(),
                    reason: "attention softmax axis must be -1, got 0".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn concat_diagnostics_enforce_cache_and_rotary_axes_by_role() {
        let concat = |inputs: &[&str], output: &str, axis: i64| NodeProto {
            input: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output: strings([output]),
            name: "renamed-cosmetic-node".to_owned(),
            op_type: "Concat".to_owned(),
            attribute: vec![int_attribute("axis", axis)],
            domain: String::new(),
        };
        let unary = |op_type: &str, input: &str, output: &str| NodeProto {
            input: strings([input]),
            output: strings([output]),
            name: String::new(),
            op_type: op_type.to_owned(),
            attribute: Vec::new(),
            domain: String::new(),
        };
        let unnamed_concat = |inputs: &[&str], output: &str| NodeProto {
            input: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output: strings([output]),
            name: String::new(),
            op_type: "Concat".to_owned(),
            attribute: vec![int_attribute("axis", -1)],
            domain: String::new(),
        };
        let protobuf = ModelProto {
            ir_version: ONNX_IR_VERSION,
            producer_name: "tritium-onnx-test".to_owned(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            domain: String::new(),
            model_version: 1,
            graph: Some(GraphProto {
                node: vec![
                    concat(&["past_k.0", "current_k"], "present_k.0", -1),
                    unary("Slice", "query", "first_half"),
                    unary("Neg", "second_half", "negated_second"),
                    concat(&["negated_second", "first_half"], "rotated", 0),
                    unary("Slice", "query", "partial_first"),
                    unary("Slice", "query", "partial_second"),
                    unnamed_concat(&["partial_first", "partial_second"], "unrotated"),
                    NodeProto {
                        input: strings(["direct", "crossed"]),
                        output: strings(["rotated_prefix"]),
                        name: String::new(),
                        op_type: "Add".to_owned(),
                        attribute: Vec::new(),
                        domain: String::new(),
                    },
                    unary("Slice", "query", "tail"),
                    unnamed_concat(&["rotated_prefix", "tail"], "partial_output"),
                    unary("Mul", "embedding", "normalized_embedding"),
                    unary("Mul", "target_hidden", "normalized_target"),
                    concat(
                        &["normalized_embedding", "normalized_target"],
                        "fusion_input",
                        0,
                    ),
                    NodeProto {
                        input: strings(["fusion_input", "fusion.packed", "fusion.scales"]),
                        output: strings(["fused_hidden"]),
                        name: "fusion".to_owned(),
                        op_type: ONNX_OP_NAME.to_owned(),
                        attribute: attributes(4, format_code(TernaryFormat::Tq2_0).unwrap()),
                        domain: ONNX_DOMAIN.to_owned(),
                    },
                ],
                name: "concat-axis-mutations".to_owned(),
                initializer: Vec::new(),
                input: vec![tensor_value("past_k.0", TENSOR_FLOAT, &[1, 1, 2])],
                output: vec![tensor_value("present_k.0", TENSOR_FLOAT, &[2, 1, 2])],
                value_info: Vec::new(),
            }),
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: ONNX_OPSET,
                },
                OperatorSetIdProto {
                    domain: ONNX_DOMAIN.to_owned(),
                    version: TRITIUM_OPSET,
                },
            ],
            metadata_props: Vec::new(),
        };
        let diagnostics = diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap();
        assert_eq!(diagnostics.len(), 3, "{diagnostics:#?}");
        assert!(
            diagnostics[0]
                .reason
                .contains("KV cache concat axis must be 0")
        );
        assert!(
            diagnostics[1]
                .reason
                .contains("RoPE concat axis must be -1")
        );
        assert!(
            diagnostics[2]
                .reason
                .contains("MTP fusion concat axis must be 1")
        );
    }

    #[test]
    fn kv_attention_diagnostics_reject_nondivisible_gqa_heads() {
        let encoded = encode_kv_attention_test_graph(1, 0, 3, 2, 4);
        assert_eq!(
            diagnose_unsupported_graph(&encoded).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Attribute,
                subject: "tritium.kv_attention.n_kv_head".to_owned(),
                reason: "n_head 3 is not divisible by n_kv_head 2".to_owned(),
            }]
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn qwen_deltanet_diagnostics_reject_nondivisible_head_groups() {
        let encoded = encode_qwen_deltanet_test_graph();
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        let node = &mut protobuf.graph.as_mut().unwrap().node[0];
        node.attribute
            .iter_mut()
            .find(|attribute| attribute.name == crate::ATTR_NUM_VALUE_HEADS)
            .unwrap()
            .value = 3;
        node.attribute
            .iter_mut()
            .find(|attribute| attribute.name == crate::ATTR_NUM_KEY_HEADS)
            .unwrap()
            .value = 2;

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Attribute,
                subject: "tritium.qwen_deltanet.num_value_heads".to_owned(),
                reason: "num_value_heads 3 is not divisible by num_key_heads 2".to_owned(),
            }]
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn qwen_deltanet_dtype_diagnostics_follow_topology_after_renaming() {
        let encoded = encode_qwen_deltanet_test_graph();
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        for (index, input) in graph.input.iter_mut().enumerate() {
            let name = format!("renamed_input_{index}");
            graph.node[0].input[index] = name.clone();
            input.name = name;
        }
        for (index, output) in graph.output.iter_mut().enumerate() {
            let name = format!("renamed_output_{index}");
            graph.node[0].output[index] = name.clone();
            output.name = name;
        }
        graph.input[0] = tensor_value("renamed_input_0", TENSOR_INT64, &[1, 3]);

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Dtype,
                subject: "input renamed_input_0".to_owned(),
                reason: "expected ONNX dtype 1, got 7".to_owned(),
            }]
        );
    }

    #[test]
    fn qwen_heterogeneous_encoder_rejects_empty_schedule() {
        let empty_matrix = PackedTernaryMatrix {
            rows: 0,
            columns: 0,
            packed: &[],
            scales: &[],
            format: TernaryFormat::Tq2_0,
        };
        let model = QwenCausalLmModel {
            tokens: 1,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 2,
            rotary: RotaryEmbedding {
                theta: 10_000.0,
                dimensions: 2,
            },
            rms_epsilon: 1.0e-6,
            delta_geometry: QwenDeltaNetGeometry::new(2, 1, 1, 1, 1).unwrap(),
            embedding: empty_matrix,
            lm_head: None,
            layers: &[],
            final_norm: &[],
            identity: OnnxArtifactIdentityV2 {
                source_model_id: "source",
                tokenizer_id: "tokenizer",
                recipe_id: "recipe",
                tritium_build_id: "build",
                package_id: "package",
                converted_coverage_id: "converted",
                deferred_coverage_id: "deferred",
            },
        };
        assert!(matches!(
            encode_qwen_causal_lm(model),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("at least one layer")
        ));
    }

    #[test]
    fn qwen35_tensor_adapter_rejects_incomplete_language_mtp_manifest() {
        struct EmptySource;

        impl<'a> Qwen35TensorProvider<'a> for EmptySource {
            fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
                Ok(&[])
            }

            fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError> {
                Err(OnnxModelError::InvalidModel(format!(
                    "unexpected matrix request {name}"
                )))
            }

            fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
                Err(OnnxModelError::InvalidModel(format!(
                    "unexpected vector request {name}"
                )))
            }
        }

        let schedule = [Qwen35LayerType::DeltaNet, Qwen35LayerType::FullAttention];
        let config = Qwen35Config {
            hidden: 256,
            intermediate: 512,
            vocab: 2,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 10_000.0,
            rms_epsilon: 1.0e-6,
            delta_geometry: QwenDeltaNetGeometry::new(4, 1, 1, 128, 128).unwrap(),
            layer_types: &schedule,
            full_attention_interval: 2,
            tied_embeddings: false,
            mtp_layers: 1,
            mtp_dedicated_embeddings: false,
        };
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "qwen-source",
            tokenizer_id: "qwen-tokenizer",
            recipe_id: "recipe",
            tritium_build_id: "build",
            package_id: "package",
            converted_coverage_id: "language-mtp",
            deferred_coverage_id: "vision",
        };
        assert!(matches!(
            map_qwen35_causal_lm(&EmptySource, config, identity),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("missing tensor")
                    && reason.contains("lm_head.weight")
        ));
    }

    #[test]
    fn qwen36_27b_adapter_rejects_short_reordered_and_wrong_cadence_schedules() {
        struct EmptySource;

        impl<'a> Qwen35TensorProvider<'a> for EmptySource {
            fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
                Ok(&[])
            }

            fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError> {
                Err(OnnxModelError::InvalidModel(format!(
                    "unexpected matrix request {name}"
                )))
            }

            fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
                Err(OnnxModelError::InvalidModel(format!(
                    "unexpected vector request {name}"
                )))
            }
        }

        fn config(schedule: &[Qwen35LayerType]) -> Qwen35Config<'_> {
            Qwen35Config {
                hidden: 5_120,
                intermediate: 17_408,
                vocab: 248_320,
                n_head: 24,
                n_kv_head: 4,
                head_dim: 256,
                rotary_dim: 64,
                rope_theta: 10_000_000.0,
                rms_epsilon: 1.0e-6,
                delta_geometry: QwenDeltaNetGeometry::new(4, 16, 48, 128, 128).unwrap(),
                layer_types: schedule,
                full_attention_interval: 4,
                tied_embeddings: false,
                mtp_layers: 1,
                mtp_dedicated_embeddings: false,
            }
        }

        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "Qwen/Qwen3.6-27B@6a9e13b",
            tokenizer_id: "Qwen/Qwen3.6-27B@6a9e13b",
            recipe_id: "recipe",
            tritium_build_id: "build",
            package_id: "package",
            converted_coverage_id: "language-mtp",
            deferred_coverage_id: "vision",
        };
        let canonical = (0..64)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::DeltaNet
                }
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            map_qwen36_27b_causal_lm(&EmptySource, config(&canonical), identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("missing tensor")
        ));
        assert!(matches!(
            map_qwen36_27b_causal_lm(&EmptySource, config(&canonical[..4]), identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("pinned")
        ));

        let mut reordered = canonical.clone();
        reordered.swap(0, 3);
        assert!(matches!(
            map_qwen36_27b_causal_lm(&EmptySource, config(&reordered), identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("interval")
        ));
        let mut wrong_cadence = canonical.clone();
        wrong_cadence[7] = Qwen35LayerType::DeltaNet;
        assert!(matches!(
            map_qwen36_27b_causal_lm(&EmptySource, config(&wrong_cadence), identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("interval")
        ));
    }

    #[test]
    fn qwen35_tensor_adapter_maps_exact_language_mtp_bundle_into_encodable_schedule() {
        struct Source {
            names: Vec<String>,
            matrix_2x256: Vec<u8>,
            matrix_384x256: Vec<u8>,
            matrix_128x256: Vec<u8>,
            matrix_1x256: Vec<u8>,
            matrix_256x128: Vec<u8>,
            matrix_512x256: Vec<u8>,
            matrix_256x256: Vec<u8>,
            matrix_256x512: Vec<u8>,
            scales_2: Vec<f32>,
            scales_384: Vec<f32>,
            scales_128: Vec<f32>,
            scales_1: Vec<f32>,
            scales_256: Vec<f32>,
            scales_512: Vec<f32>,
            hidden: Vec<f32>,
            conv: Vec<f32>,
            delta_norm: Vec<f32>,
            scalar: Vec<f32>,
        }

        impl<'a> Qwen35TensorProvider<'a> for Source {
            fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
                Ok(&self.names)
            }

            fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError> {
                let (rows, columns, packed, scales) = if matches!(
                    name,
                    "model.language_model.embed_tokens.weight" | "lm_head.weight"
                ) {
                    (2, 256, &self.matrix_2x256, &self.scales_2)
                } else if name.ends_with("linear_attn.in_proj_qkv.weight") {
                    (384, 256, &self.matrix_384x256, &self.scales_384)
                } else if name.ends_with("linear_attn.in_proj_z.weight") {
                    (128, 256, &self.matrix_128x256, &self.scales_128)
                } else if name.ends_with("linear_attn.in_proj_b.weight")
                    || name.ends_with("linear_attn.in_proj_a.weight")
                {
                    (1, 256, &self.matrix_1x256, &self.scales_1)
                } else if name.ends_with("linear_attn.out_proj.weight") {
                    (256, 128, &self.matrix_256x128, &self.scales_256)
                } else if name == "mtp.fc.weight" {
                    (256, 512, &self.matrix_256x512, &self.scales_256)
                } else if name.ends_with("self_attn.q_proj.weight") {
                    (512, 256, &self.matrix_512x256, &self.scales_512)
                } else if name.ends_with("self_attn.k_proj.weight")
                    || name.ends_with("self_attn.v_proj.weight")
                    || name.ends_with("self_attn.o_proj.weight")
                    || name.ends_with("mlp.gate_proj.weight")
                    || name.ends_with("mlp.up_proj.weight")
                    || name.ends_with("mlp.down_proj.weight")
                {
                    (256, 256, &self.matrix_256x256, &self.scales_256)
                } else {
                    return Err(OnnxModelError::InvalidModel(format!(
                        "unexpected Qwen matrix {name}"
                    )));
                };
                Ok(PackedTernaryMatrix {
                    rows,
                    columns,
                    packed,
                    scales,
                    format: TernaryFormat::Tq2_0,
                })
            }

            fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
                if name.ends_with("linear_attn.conv1d.weight") {
                    Ok(&self.conv)
                } else if name.ends_with("linear_attn.norm.weight") {
                    Ok(&self.delta_norm)
                } else if name.ends_with("linear_attn.dt_bias")
                    || name.ends_with("linear_attn.A_log")
                {
                    Ok(&self.scalar)
                } else if name.ends_with(".weight") {
                    Ok(&self.hidden)
                } else {
                    Err(OnnxModelError::InvalidModel(format!(
                        "unexpected Qwen vector {name}"
                    )))
                }
            }
        }

        let schedule = [Qwen35LayerType::DeltaNet, Qwen35LayerType::FullAttention];
        let mut source = Source {
            names: qwen35_tensor_names(&schedule).unwrap(),
            matrix_2x256: unit_packed_shape(TernaryFormat::Tq2_0, 2, 256),
            matrix_384x256: unit_packed_shape(TernaryFormat::Tq2_0, 384, 256),
            matrix_128x256: unit_packed_shape(TernaryFormat::Tq2_0, 128, 256),
            matrix_1x256: unit_packed_shape(TernaryFormat::Tq2_0, 1, 256),
            matrix_256x128: unit_packed_shape(TernaryFormat::Tq2_0, 256, 128),
            matrix_512x256: unit_packed_shape(TernaryFormat::Tq2_0, 512, 256),
            matrix_256x256: unit_packed_shape(TernaryFormat::Tq2_0, 256, 256),
            matrix_256x512: unit_packed_shape(TernaryFormat::Tq2_0, 256, 512),
            scales_2: vec![1.0; 2],
            scales_384: vec![1.0; 384],
            scales_128: vec![1.0; 128],
            scales_1: vec![1.0],
            scales_256: vec![1.0; 256],
            scales_512: vec![1.0; 512],
            hidden: vec![0.0; 256],
            conv: vec![0.0; 384 * 2],
            delta_norm: vec![1.0; 128],
            scalar: vec![0.0],
        };
        let config = Qwen35Config {
            hidden: 256,
            intermediate: 256,
            vocab: 2,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 10_000.0,
            rms_epsilon: 1.0e-6,
            delta_geometry: QwenDeltaNetGeometry::new(2, 1, 1, 128, 128).unwrap(),
            layer_types: &schedule,
            full_attention_interval: 2,
            tied_embeddings: false,
            mtp_layers: 1,
            mtp_dedicated_embeddings: false,
        };
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "qwen-source",
            tokenizer_id: "qwen-tokenizer",
            recipe_id: "recipe",
            tritium_build_id: "build",
            package_id: "package",
            converted_coverage_id: "language-mtp",
            deferred_coverage_id: "vision",
        };

        let mapped = map_qwen35_causal_lm(&source, config, identity).unwrap();
        assert!(matches!(
            mapped.layers(),
            [
                QwenCausalLmDecoderLayer::DeltaNet(_),
                QwenCausalLmDecoderLayer::FullAttention(_)
            ]
        ));
        assert_eq!(mapped.mtp().fusion.rows, 256);
        assert_eq!(mapped.mtp().fusion.columns, 512);
        let reordered_schedule = [mapped.layers()[1], mapped.layers()[0]];
        assert!(matches!(
            encode_qwen_causal_lm(QwenCausalLmModel {
                layers: &reordered_schedule,
                ..mapped.model(1, 0)
            }),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("canonical full-attention interval")
        ));
        assert!(encode_qwen_causal_lm(mapped.model(1, 0)).is_ok());
        assert!(encode_qwen35_mtp(mapped.mtp_model(1, 0)).is_ok());
        let bundle =
            encode_external_qwen35_bundle(mapped.model(1, 0), mapped.mtp_model(1, 0)).unwrap();
        let standalone_language = encode_external_qwen_causal_lm(mapped.model(1, 0)).unwrap();
        let standalone_mtp = encode_external_qwen35_mtp(mapped.mtp_model(1, 0)).unwrap();
        assert!(
            bundle.weights_bytes.len()
                < standalone_language.weights_bytes.len() + standalone_mtp.weights_bytes.len(),
            "shared embedding/head ranges must reduce physical package bytes"
        );
        assert_eq!(
            bundle.weights_blake3,
            *blake3::hash(&bundle.weights_bytes).as_bytes()
        );
        let admitted = AdmittedExternalQwen35BundleDigests {
            language_model_blake3: *blake3::hash(&bundle.language_model_bytes).as_bytes(),
            mtp_model_blake3: *blake3::hash(&bundle.mtp_model_bytes).as_bytes(),
            weights_blake3: *blake3::hash(&bundle.weights_bytes).as_bytes(),
        };
        let files = ExternalQwen35BundleFiles {
            language_model_bytes: &bundle.language_model_bytes,
            mtp_model_bytes: &bundle.mtp_model_bytes,
            weights_bytes: &bundle.weights_bytes,
        };
        let receipt = verify_external_qwen35_bundle(files, admitted).unwrap();
        assert_eq!(receipt.language.layers, 2);
        assert_eq!(receipt.mtp.layers, 1);
        assert_eq!(receipt.language.tokens, receipt.mtp.tokens);
        assert_eq!(receipt.language.past_tokens, receipt.mtp.past_tokens);
        assert_eq!(receipt.language.identity, receipt.mtp.identity);

        let mut corrupted_weights = bundle.weights_bytes.clone();
        corrupted_weights[0] ^= 1;
        assert!(matches!(
            verify_external_qwen35_bundle(
                ExternalQwen35BundleFiles {
                    weights_bytes: &corrupted_weights,
                    ..files
                },
                admitted,
            ),
            Err(OnnxModelError::ExternalDataMismatch(reason))
                if reason.contains("bundle weights")
                    && reason.contains("package manifest")
        ));
        let language_proto = ModelProto::decode(bundle.language_model_bytes.as_slice()).unwrap();
        let mut mtp_proto = ModelProto::decode(bundle.mtp_model_bytes.as_slice()).unwrap();
        for name in QWEN_SHARED_EXTERNAL_INITIALIZERS {
            let descriptor = |protobuf: &ModelProto| {
                protobuf
                    .graph
                    .as_ref()
                    .unwrap()
                    .initializer
                    .iter()
                    .find(|initializer| initializer.name == name)
                    .unwrap()
                    .external_data
                    .clone()
            };
            assert_eq!(descriptor(&language_proto), descriptor(&mtp_proto));
        }
        mtp_proto
            .metadata_props
            .iter_mut()
            .find(|entry| entry.key == "tritium.package_id")
            .unwrap()
            .value = "different-package".to_owned();
        let mismatched_mtp = mtp_proto.encode_to_vec();
        assert!(matches!(
            verify_external_qwen35_bundle(
                ExternalQwen35BundleFiles {
                    mtp_model_bytes: &mismatched_mtp,
                    ..files
                },
                AdmittedExternalQwen35BundleDigests {
                    mtp_model_blake3: *blake3::hash(&mismatched_mtp).as_bytes(),
                    ..admitted
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("identity or execution shape differs")
        ));
        let mut mismatched_epsilon = ModelProto::decode(bundle.mtp_model_bytes.as_slice()).unwrap();
        mismatched_epsilon
            .metadata_props
            .iter_mut()
            .find(|entry| entry.key == "tritium.rms_epsilon")
            .unwrap()
            .value = "0.000002".to_owned();
        let mismatched_epsilon = mismatched_epsilon.encode_to_vec();
        assert!(matches!(
            verify_external_qwen35_bundle(
                ExternalQwen35BundleFiles {
                    mtp_model_bytes: &mismatched_epsilon,
                    ..files
                },
                AdmittedExternalQwen35BundleDigests {
                    mtp_model_blake3: *blake3::hash(&mismatched_epsilon).as_bytes(),
                    ..admitted
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("metadata tritium.rms_epsilon differs")
        ));
        let mut bad_interval = ModelProto::decode(bundle.language_model_bytes.as_slice()).unwrap();
        bad_interval
            .metadata_props
            .iter_mut()
            .find(|entry| entry.key == "tritium.full_attention_interval")
            .unwrap()
            .value = "1".to_owned();
        let bad_interval = bad_interval.encode_to_vec();
        assert!(matches!(
            verify_external_qwen35_bundle(
                ExternalQwen35BundleFiles {
                    language_model_bytes: &bad_interval,
                    ..files
                },
                AdmittedExternalQwen35BundleDigests {
                    language_model_blake3: *blake3::hash(&bad_interval).as_bytes(),
                    ..admitted
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("layer schedule metadata is not canonical")
        ));
        let mut overlapping_mtp = ModelProto::decode(bundle.mtp_model_bytes.as_slice()).unwrap();
        let initializer = overlapping_mtp
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .iter_mut()
            .find(|initializer| {
                initializer.data_location == EXTERNAL_DATA
                    && !QWEN_SHARED_EXTERNAL_INITIALIZERS.contains(&initializer.name.as_str())
            })
            .unwrap();
        initializer
            .external_data
            .iter_mut()
            .find(|entry| entry.key == "offset")
            .unwrap()
            .value = "0".to_owned();
        let overlapping_mtp = overlapping_mtp.encode_to_vec();
        assert!(matches!(
            verify_external_qwen35_bundle(
                ExternalQwen35BundleFiles {
                    mtp_model_bytes: &overlapping_mtp,
                    ..files
                },
                AdmittedExternalQwen35BundleDigests {
                    mtp_model_blake3: *blake3::hash(&overlapping_mtp).as_bytes(),
                    ..admitted
                },
            ),
            Err(OnnxModelError::ExternalDataMismatch(reason))
                if reason.contains("overlapping shared range")
                    || reason.contains("not canonical")
        ));
        assert!(matches!(
            encode_external_qwen35_bundle(
                mapped.model(1, 0),
                Qwen35MtpModel {
                    tokens: 2,
                    ..mapped.mtp_model(1, 0)
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("execution geometry differs")
        ));

        source
            .names
            .push("model.language_model.layers.2.input_layernorm.weight".to_owned());
        assert!(matches!(
            map_qwen35_causal_lm(&source, config, identity),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("unsupported tensor") && reason.contains("layers.2")
        ));
    }

    #[test]
    fn kv_attention_diagnostics_reject_opset_one_import() {
        let encoded = encode_kv_attention_test_graph(1, 0, 2, 1, 4);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        protobuf.opset_import[0].version = 1;
        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Node,
                subject: "tritium.kv_attention".to_owned(),
                reason: "operator requires com.tritium opset 2, imported 1".to_owned(),
            }]
        );
    }

    #[test]
    fn diagnostics_reject_duplicate_tritium_opset_imports() {
        let encoded = encode_kv_attention_test_graph(1, 0, 2, 1, 4);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        protobuf.opset_import.push(OperatorSetIdProto {
            domain: ONNX_DOMAIN.to_owned(),
            version: 3,
        });
        assert!(matches!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("duplicate opset import")
        ));
    }

    #[test]
    fn encoder_rejects_non_unit_internal_scales() {
        let mut packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let scale_offset = TQ2_0_BLOCK_BYTES - core::mem::size_of::<f16>();
        packed[scale_offset..].copy_from_slice(&f16::ZERO.to_le_bytes());
        let result = encode_tied_embedding_head(TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        });
        assert!(matches!(
            result,
            Err(OnnxModelError::InvalidPackedRow { .. })
        ));
    }

    #[test]
    fn rotary_tables_remain_finite_at_extreme_valid_theta_and_position() {
        let (cos, sin) = rotary_tables(RotaryGraphConfig {
            tokens: 2,
            head_dim: 128,
            rotary_dim: 128,
            past_tokens: 10_000_000,
            theta: f32::MIN_POSITIVE,
        })
        .unwrap();
        assert_eq!(cos.len(), 256);
        assert_eq!(sin.len(), 256);
        assert!(cos.iter().chain(&sin).all(|value| value.is_finite()));
    }

    #[test]
    fn rotary_tables_reject_oversized_allocation_before_building_values() {
        let result = rotary_tables(RotaryGraphConfig {
            tokens: MAX_MODEL_BYTES,
            head_dim: 2,
            rotary_dim: 2,
            past_tokens: 0,
            theta: 10_000.0,
        });
        assert!(matches!(
            result,
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("RoPE tables require")
        ));
    }

    #[test]
    fn causal_initializer_budget_rejects_aggregate_payload_before_cloning() {
        let empty_matrix = PackedTernaryMatrix {
            rows: 1,
            columns: 1,
            packed: &[],
            scales: &[],
            format: TernaryFormat::Tq2_0,
        };
        let model = CausalLmModel {
            tokens: 4097,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 1,
            rotary: None,
            rms_epsilon: 1.0e-5,
            zero_centered_norm: false,
            embedding: empty_matrix,
            lm_head: None,
            final_norm: &[],
            layers: &[],
            identity: OnnxArtifactIdentityV2 {
                source_model_id: "source",
                tokenizer_id: "tokenizer",
                recipe_id: "recipe",
                tritium_build_id: "build",
                package_id: "package",
                converted_coverage_id: "converted",
                deferred_coverage_id: "deferred",
            },
        };
        assert!(matches!(
            validate_causal_initializer_budget(&model),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("causal initializers require at least")
        ));
    }

    #[test]
    fn counting_builder_rejects_generated_initializer_aggregate_without_materializing() {
        let mut graph = CausalGraphBuilder::counting();
        graph.add_causal_mask("mask", 4096, 4096, 0, vec![1, 4096, 4096]);
        assert!(graph.failure.is_none());

        graph.add_i64("shape", vec![1], &[1]);
        assert!(matches!(
            graph.failure,
            Some(OnnxModelError::InvalidModel(ref reason))
                if reason.contains("inline initializers require at least")
        ));
        assert!(graph.initializers.is_empty());
    }

    #[test]
    fn smollm2_tensor_adapter_builds_encodable_canonical_graph() {
        use std::cell::RefCell;

        struct Source {
            names: Vec<String>,
            vocab_packed: Vec<u8>,
            hidden_packed: Vec<u8>,
            kv_packed: Vec<u8>,
            intermediate_packed: Vec<u8>,
            down_packed: Vec<u8>,
            vocab_scales: Vec<f32>,
            hidden_scales: Vec<f32>,
            kv_scales: Vec<f32>,
            intermediate_scales: Vec<f32>,
            down_scales: Vec<f32>,
            norm: Vec<f32>,
            requested: RefCell<Vec<String>>,
        }
        impl<'a> SmolLm2TensorProvider<'a> for Source {
            fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
                Ok(&self.names)
            }

            fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError> {
                self.requested.borrow_mut().push(name.to_owned());
                let (rows, columns, packed, scales) = if name == "model.embed_tokens.weight" {
                    (
                        2,
                        256,
                        self.vocab_packed.as_slice(),
                        self.vocab_scales.as_slice(),
                    )
                } else if matches!(
                    name,
                    "model.layers.0.self_attn.q_proj.weight"
                        | "model.layers.0.self_attn.o_proj.weight"
                ) {
                    (
                        256,
                        256,
                        self.hidden_packed.as_slice(),
                        self.hidden_scales.as_slice(),
                    )
                } else if matches!(
                    name,
                    "model.layers.0.self_attn.k_proj.weight"
                        | "model.layers.0.self_attn.v_proj.weight"
                ) {
                    (
                        128,
                        256,
                        self.kv_packed.as_slice(),
                        self.kv_scales.as_slice(),
                    )
                } else if matches!(
                    name,
                    "model.layers.0.mlp.gate_proj.weight" | "model.layers.0.mlp.up_proj.weight"
                ) {
                    (
                        512,
                        256,
                        self.intermediate_packed.as_slice(),
                        self.intermediate_scales.as_slice(),
                    )
                } else if name == "model.layers.0.mlp.down_proj.weight" {
                    (
                        256,
                        512,
                        self.down_packed.as_slice(),
                        self.down_scales.as_slice(),
                    )
                } else {
                    return Err(OnnxModelError::InvalidModel(format!(
                        "unexpected matrix {name}"
                    )));
                };
                Ok(PackedTernaryMatrix {
                    rows,
                    columns,
                    packed,
                    scales,
                    format: TernaryFormat::Tq2_0,
                })
            }

            fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
                self.requested.borrow_mut().push(name.to_owned());
                match name {
                    "model.norm.weight"
                    | "model.layers.0.input_layernorm.weight"
                    | "model.layers.0.post_attention_layernorm.weight" => Ok(&self.norm),
                    _ => Err(OnnxModelError::InvalidModel(format!(
                        "unexpected vector {name}"
                    ))),
                }
            }
        }
        let mut source = Source {
            names: smollm2_tensor_names(1).unwrap(),
            vocab_packed: unit_packed(TernaryFormat::Tq2_0, 2),
            hidden_packed: unit_packed(TernaryFormat::Tq2_0, 256),
            kv_packed: unit_packed(TernaryFormat::Tq2_0, 128),
            intermediate_packed: unit_packed(TernaryFormat::Tq2_0, 512),
            down_packed: unit_packed_shape(TernaryFormat::Tq2_0, 256, 512),
            vocab_scales: vec![1.0; 2],
            hidden_scales: vec![1.0; 256],
            kv_scales: vec![1.0; 128],
            intermediate_scales: vec![1.0; 512],
            down_scales: vec![1.0; 256],
            norm: vec![1.0; 256],
            requested: RefCell::new(Vec::new()),
        };
        let config = SmolLm2Config {
            layers: 1,
            hidden: 256,
            intermediate: 512,
            vocab: 2,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 64,
            rope_theta: 10_000.0,
            rms_epsilon: 1.0e-5,
            tied_embeddings: true,
            projection_bias: false,
            activation: CausalActivation::SwiGlu,
            rotary_mode: RotaryMode::Full,
        };
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "smollm2-source@revision",
            tokenizer_id: "smollm2-tokenizer@revision",
            recipe_id: "recipe",
            tritium_build_id: "build",
            package_id: "package",
            converted_coverage_id: "converted",
            deferred_coverage_id: "deferred",
        };
        assert!(matches!(
            map_smollm2_causal_lm(
                &source,
                SmolLm2Config {
                    activation: CausalActivation::Relu2,
                    ..config
                },
                identity,
            ),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("bias-free SwiGLU")
        ));
        assert!(source.requested.borrow().is_empty());
        source
            .names
            .push("model.layers.0.self_attn.q_proj.bias".to_owned());
        assert!(matches!(
            map_smollm2_causal_lm(&source, config, identity),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("unsupported tensor") && reason.contains("q_proj.bias")
        ));
        source.names.pop();
        let missing = source.names.pop().unwrap();
        assert!(matches!(
            map_smollm2_causal_lm(&source, config, identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("missing tensor")
        ));
        source.names.push(missing);
        source.names.push(source.names[0].clone());
        assert!(matches!(
            map_smollm2_causal_lm(&source, config, identity),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("duplicate names")
        ));
        source.names.pop();
        assert!(source.requested.borrow().is_empty());
        assert!(matches!(
            map_smollm2_causal_lm(
                &source,
                SmolLm2Config {
                    intermediate: 128,
                    ..config
                },
                identity,
            ),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("architecture config")
        ));
        source.requested.borrow_mut().clear();
        let mapped = map_smollm2_causal_lm(&source, config, identity).unwrap();
        assert_eq!(mapped.layers().len(), 1);
        assert!(encode_external_causal_lm(mapped.model(1, 0)).is_ok());
        assert_eq!(
            source.requested.into_inner(),
            [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
        );
    }

    #[test]
    fn bitnet_gguf_adapter_maps_relu2_subnorm_graph_and_rejects_extra_head() {
        use std::cell::RefCell;

        struct Source {
            names: Vec<String>,
            embedding: Vec<u8>,
            hidden: Vec<u8>,
            kv: Vec<u8>,
            intermediate: Vec<u8>,
            down: Vec<u8>,
            embedding_scales: Vec<f32>,
            hidden_scales: Vec<f32>,
            kv_scales: Vec<f32>,
            intermediate_scales: Vec<f32>,
            norm_hidden: Vec<f32>,
            norm_intermediate: Vec<f32>,
            requested: RefCell<Vec<String>>,
        }

        impl<'a> BitNetGgufTensorProvider<'a> for Source {
            fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
                self.requested.borrow_mut().push("<manifest>".to_owned());
                Ok(&self.names)
            }

            fn matrix(&'a self, name: &str) -> Result<PackedTernaryMatrix<'a>, OnnxModelError> {
                self.requested.borrow_mut().push(name.to_owned());
                let (rows, columns, packed, scales) = match name {
                    "token_embd.weight" => (
                        2,
                        256,
                        self.embedding.as_slice(),
                        self.embedding_scales.as_slice(),
                    ),
                    "blk.0.attn_q.weight" | "blk.0.attn_output.weight" => (
                        256,
                        256,
                        self.hidden.as_slice(),
                        self.hidden_scales.as_slice(),
                    ),
                    "blk.0.attn_k.weight" | "blk.0.attn_v.weight" => {
                        (128, 256, self.kv.as_slice(), self.kv_scales.as_slice())
                    }
                    "blk.0.ffn_gate.weight" | "blk.0.ffn_up.weight" => (
                        512,
                        256,
                        self.intermediate.as_slice(),
                        self.intermediate_scales.as_slice(),
                    ),
                    "blk.0.ffn_down.weight" => (
                        256,
                        512,
                        self.down.as_slice(),
                        self.hidden_scales.as_slice(),
                    ),
                    _ => {
                        return Err(OnnxModelError::InvalidModel(format!(
                            "unexpected BitNet matrix {name}"
                        )));
                    }
                };
                Ok(PackedTernaryMatrix {
                    rows,
                    columns,
                    packed,
                    scales,
                    format: TernaryFormat::Tq2_0,
                })
            }

            fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
                self.requested.borrow_mut().push(name.to_owned());
                match name {
                    "output_norm.weight"
                    | "blk.0.attn_norm.weight"
                    | "blk.0.attn_sub_norm.weight"
                    | "blk.0.ffn_norm.weight" => Ok(&self.norm_hidden),
                    "blk.0.ffn_sub_norm.weight" => Ok(&self.norm_intermediate),
                    _ => Err(OnnxModelError::InvalidModel(format!(
                        "unexpected BitNet vector {name}"
                    ))),
                }
            }
        }

        let mut source = Source {
            names: [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.attn_sub_norm.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_sub_norm.weight",
                "blk.0.ffn_down.weight",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            embedding: unit_packed(TernaryFormat::Tq2_0, 2),
            hidden: unit_packed(TernaryFormat::Tq2_0, 256),
            kv: unit_packed(TernaryFormat::Tq2_0, 128),
            intermediate: unit_packed(TernaryFormat::Tq2_0, 512),
            down: unit_packed_shape(TernaryFormat::Tq2_0, 256, 512),
            embedding_scales: vec![1.0; 2],
            hidden_scales: vec![1.0; 256],
            kv_scales: vec![1.0; 128],
            intermediate_scales: vec![1.0; 512],
            norm_hidden: vec![1.0; 256],
            norm_intermediate: vec![1.0; 512],
            requested: RefCell::new(Vec::new()),
        };
        let config = BitNetConfig {
            layers: 1,
            hidden: 256,
            intermediate: 512,
            vocab: 2,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 64,
            rope_theta: 10_000.0,
            rms_epsilon: 1.0e-5,
            rotary_mode: RotaryMode::Full,
        };
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "bitnet-gguf@digest",
            tokenizer_id: "bitnet-tokenizer@digest",
            recipe_id: "recipe",
            tritium_build_id: "build",
            package_id: "package",
            converted_coverage_id: "converted",
            deferred_coverage_id: "deferred",
        };
        assert!(matches!(
            map_bitnet_gguf_causal_lm(
                &source,
                BitNetConfig {
                    rotary_mode: RotaryMode::Partial,
                    ..config
                },
                identity,
            ),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("full unscaled RoPE")
        ));
        assert!(matches!(
            map_bitnet_gguf_causal_lm(
                &source,
                BitNetConfig {
                    hidden: 252,
                    head_dim: 63,
                    ..config
                },
                identity,
            ),
            Err(OnnxModelError::InvalidModel(reason)) if reason.contains("even head dimension")
        ));
        assert!(source.requested.borrow().is_empty());
        let mapped = map_bitnet_gguf_causal_lm(&source, config, identity).unwrap();
        assert_eq!(mapped.layers().len(), 1);
        assert_eq!(mapped.layers()[0].activation, CausalActivation::Relu2);
        assert!(mapped.layers()[0].attention_sub_norm.is_some());
        assert!(mapped.layers()[0].ffn_sub_norm.is_some());
        assert!(encode_external_causal_lm(mapped.model(1, 0)).is_ok());

        source.names.push("output.weight".to_owned());
        assert!(matches!(
            map_bitnet_gguf_causal_lm(&source, config, identity),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("unsupported tensor") && reason.contains("output.weight")
        ));
    }
}
