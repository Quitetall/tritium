//! Borrowed shared-weight graph and optional backend-owned decoder execution.
use crate::{BackendError, DeviceBuffer};
use std::sync::Arc;

/// Actual execution selected by the common language runner, not qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecoderExecution {
    /// Canonical shared host graph.
    Host,
    /// Explicit device-f32 whole standard decoder graph.
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
    pub layers: &'a [DecoderLayer<'a>],
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
