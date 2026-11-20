//! Borrowed shared-weight graph and optional backend-owned decoder execution.
use crate::{BackendError, DeltaMixerGeometry, DeltaMixerWeights, DeviceBuffer};
use std::sync::Arc;

/// Actual execution selected by the common language runner, not qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecoderExecution {
    /// Canonical shared host graph.
    Host,
    /// Explicit device-f32 language graph.
    ResidentDeviceF32,
}

/// Primitive geometry/settings compiled from the authoritative architecture.
#[derive(Clone, Copy, Debug)]
pub struct DecoderGeometry {
    /// Residual width.
    pub hidden: usize,
    /// Feed-forward width.
    pub intermediate: usize,
    /// Query heads.
    pub query_heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// Actual per-head width, independent of residual width.
    pub head_dim: usize,
    /// Vocabulary rows.
    pub vocabulary: usize,
    /// Standard RMSNorm epsilon.
    pub rms_eps: f32,
    /// Standard NeoX RoPE base.
    pub rope_theta: f32,
}

/// Admitted stream bounds for resident graph construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderLimits {
    /// Selected context capacity.
    pub max_context: usize,
    /// Maximum new tokens in a forward segment.
    pub max_segment_tokens: usize,
}

/// Actual standard feed-forward activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecoderActivation {
    /// SiLU(gate) times up.
    SwiGlu,
    /// Squared ReLU(gate) times up.
    Relu2,
}

/// Borrowed immutable layer bindings, never copies of matrix payloads.
#[allow(missing_debug_implementations)]
pub struct DecoderLayer<'a> {
    /// Pre-attention RMSNorm.
    pub attention_norm: &'a [f32],
    /// Query projection.
    pub query: &'a Arc<dyn DeviceBuffer>,
    /// Key projection.
    pub key: &'a Arc<dyn DeviceBuffer>,
    /// Value projection.
    pub value: &'a Arc<dyn DeviceBuffer>,
    /// Attention output projection.
    pub attention_output: &'a Arc<dyn DeviceBuffer>,
    /// Optional attention-output RMSNorm.
    pub attention_sub_norm: &'a [f32],
    /// Optional query bias.
    pub query_bias: &'a [f32],
    /// Optional key bias.
    pub key_bias: &'a [f32],
    /// Optional value bias.
    pub value_bias: &'a [f32],
    /// Optional per-head query RMSNorm.
    pub query_norm: &'a [f32],
    /// Optional per-head key RMSNorm.
    pub key_norm: &'a [f32],
    /// Pre-MLP RMSNorm.
    pub ffn_norm: &'a [f32],
    /// MLP gate projection.
    pub gate: &'a Arc<dyn DeviceBuffer>,
    /// MLP up projection.
    pub up: &'a Arc<dyn DeviceBuffer>,
    /// MLP down projection.
    pub down: &'a Arc<dyn DeviceBuffer>,
    /// Actual gated activation.
    pub activation: DecoderActivation,
    /// Optional gated-body RMSNorm before the down projection.
    pub ffn_sub_norm: &'a [f32],
    /// Actual gated-body RMSNorm epsilon.
    pub ffn_sub_eps: f32,
}

/// Runtime compilation view of existing admitted shared weights.
#[allow(missing_debug_implementations)]
pub struct DecoderGraph<'a> {
    /// Immutable primitive geometry/settings.
    pub geometry: DecoderGeometry,
    /// Shared token table.
    pub embedding: &'a Arc<dyn DeviceBuffer>,
    /// Shared output projection; tied heads use the same Arc as embedding.
    pub head: &'a Arc<dyn DeviceBuffer>,
    /// Final RMSNorm.
    pub final_norm: &'a [f32],
    /// Ordered layers.
    pub layers: DecoderLayers<'a>,
    /// Normalization and RoPE interpretation; never inferred from tensor names.
    pub semantics: DecoderSemantics,
    /// Retained, nonexecuted matrices (e.g. unverified MTP), included in admission.
    pub retained_matrices: &'a [&'a Arc<dyn DeviceBuffer>],
}

/// Immutable numerical interpretation compiled from the authoritative graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderSemantics {
    /// Decoder/QK norms use one plus preserved weight, instead of direct weight.
    pub zero_centered_norm: bool,
    /// Active even rotary prefix per head.
    pub rotary_dim: usize,
    /// Partial RoPE tables use the Qwen fp32 power/reciprocal construction.
    pub qwen_rope_f32: bool,
}
impl DecoderSemantics {
    /// Standard direct norms and full f64 table construction.
    #[must_use]
    pub const fn standard(head_dim: usize) -> Self {
        Self {
            zero_centered_norm: false,
            rotary_dim: head_dim,
            qwen_rope_f32: false,
        }
    }
}
/// Actual ordered graph layer family, with no placeholder mixer fields.
#[allow(missing_debug_implementations)]
#[non_exhaustive]
pub enum DecoderLayers<'a> {
    /// Standard attention blocks with their existing variation settings.
    Standard(&'a [DecoderLayer<'a>]),
    /// Explicit hybrid schedule.
    Hybrid(&'a [DecoderHybridLayer<'a>]),
}
/// Shared hybrid decoder norms/MLP and an explicit token mixer.
#[allow(missing_debug_implementations)]
pub struct DecoderHybridLayer<'a> {
    /// Input decoder norm.
    pub input_norm: &'a [f32],
    /// Post-mixer decoder norm.
    pub post_norm: &'a [f32],
    /// SwiGLU gate projection.
    pub gate: &'a Arc<dyn DeviceBuffer>,
    /// SwiGLU up projection.
    pub up: &'a Arc<dyn DeviceBuffer>,
    /// SwiGLU down projection.
    pub down: &'a Arc<dyn DeviceBuffer>,
    /// Actual mixer kind/bindings.
    pub mixer: DecoderMixer<'a>,
}
/// Typed hybrid mixer bindings.
#[allow(missing_debug_implementations)]
#[non_exhaustive]
pub enum DecoderMixer<'a> {
    /// Qwen full attention with fused per-head query/gate rows.
    FullAttention(DecoderFullAttention<'a>),
    /// Qwen causal convolution and gated DeltaNet recurrence.
    DeltaNet(DecoderDeltaNet<'a>),
}
/// Qwen full-attention immutable shared handles and preserved norms.
#[allow(missing_debug_implementations)]
pub struct DecoderFullAttention<'a> {
    /// `[2 * query_width, hidden]` in per-head query/gate order.
    pub query_gate: &'a Arc<dyn DeviceBuffer>,
    /// Key projection.
    pub key: &'a Arc<dyn DeviceBuffer>,
    /// Value projection.
    pub value: &'a Arc<dyn DeviceBuffer>,
    /// Output projection.
    pub output: &'a Arc<dyn DeviceBuffer>,
    /// Zero-centered per-head query norm.
    pub query_norm: &'a [f32],
    /// Zero-centered per-head key norm.
    pub key_norm: &'a [f32],
}
/// Qwen DeltaNet immutable shared handles and preserved coefficients.
#[allow(missing_debug_implementations)]
pub struct DecoderDeltaNet<'a> {
    /// Real recurrence/convolution/gated normalization settings.
    pub geometry: DeltaMixerGeometry,
    /// Preserved coefficients.
    pub weights: DeltaMixerWeights<'a>,
    /// Raw QKV projection.
    pub qkv: &'a Arc<dyn DeviceBuffer>,
    /// Output gate projection.
    pub z: &'a Arc<dyn DeviceBuffer>,
    /// Beta projection.
    pub beta: &'a Arc<dyn DeviceBuffer>,
    /// Decay projection.
    pub decay: &'a Arc<dyn DeviceBuffer>,
    /// Output projection.
    pub output: &'a Arc<dyn DeviceBuffer>,
}

/// Backend-owned graph execution with atomic language-stream publication.
pub trait ResidentDecoder: Send {
    /// Selected stream bounds.
    fn limits(&self) -> DecoderLimits;
    /// Unique matrix/vector/KV/workspace device payload, not physical VRAM.
    fn payload_bytes(&self) -> usize;
    /// Committed visible language cursor.
    fn len(&self) -> usize;
    /// Whether the committed language stream is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Forward a contiguous segment and return final-row logits.
    /// Cancellation returns None without publication; errors preserve committed
    /// cache prefixes. Only successful finite readback followed by cancellation
    /// preflight may commit the graph.
    ///
    /// # Errors
    /// Input/extent/capacity, allocation, numerical and driver failures.
    fn forward(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, BackendError>;
    /// Clear visibility while retaining bounded device capacity.
    fn reset(&mut self);
}
