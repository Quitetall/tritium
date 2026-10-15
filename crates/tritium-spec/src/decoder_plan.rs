//! Structural resident allocation inputs; no weights, reservations or device work.
use crate::{
    BackendError, DecoderActivation, DecoderGeometry, DecoderSemantics, DeltaMixerGeometry,
    DeltaMixerWeightLengths,
};

/// Authoritative graph family, never inferred from coefficient names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecoderAllocationKind {
    /// Standard attention and SwiGLU/Relu2 layers.
    Standard,
    /// Ordered Qwen gated-attention/DeltaNet layers.
    Hybrid,
}

/// Structural mixer coefficients, without reading numerical values.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum DecoderAllocationMixer {
    /// Standard or fused-query-gate attention.
    Attention {
        /// Fused query/gate rows and mandatory per-head Q/K normalization.
        gated: bool,
        /// Lengths in order: output sub-norm, Q/K/V biases, Q/K head norms.
        vectors: [usize; 6],
    },
    /// Causal convolution, gated recurrence and preserved coefficient lengths.
    DeltaNet {
        /// Actual convolution/recurrence geometry.
        geometry: DeltaMixerGeometry,
        /// Declared coefficient lengths, not finite-value validation.
        weights: DeltaMixerWeightLengths,
    },
}

/// One ordered layer's structural allocation input.
#[derive(Clone, Copy, Debug)]
pub struct DecoderAllocationLayer {
    /// Input normalization coefficient count.
    pub input_norm: usize,
    /// Post-mixer normalization coefficient count.
    pub post_norm: usize,
    /// Actual MLP activation.
    pub activation: DecoderActivation,
    /// Optional gated-body normalization coefficient count.
    pub ffn_sub_norm: usize,
    /// Gated-body normalization epsilon.
    pub ffn_sub_eps: f32,
    /// Actual ordered mixer.
    pub mixer: DecoderAllocationMixer,
}

/// Metadata supplied by a source plan or actual checked graph bindings.
/// Matrix shapes, ownership and values need their separate actual-path checks.
/// This describes logical payload only, not physical fit or model support.
#[derive(Clone, Copy, Debug)]
pub struct DecoderAllocationMetadata<'a> {
    /// Authoritative primitive geometry.
    pub geometry: DecoderGeometry,
    /// Standard versus hybrid interpretation.
    pub kind: DecoderAllocationKind,
    /// Actual normalization and RoPE interpretation.
    pub semantics: DecoderSemantics,
    /// Final normalization coefficient count.
    pub final_norm: usize,
    /// Ordered layer metadata, without a second tensor/count inventory.
    pub layers: &'a [DecoderAllocationLayer],
    /// Native matrix payload summed once per owner, including retained MTP.
    pub matrix_payload_bytes: usize,
    /// Maximum columns of an executed nonidentity-basis projection, or zero.
    /// Retained nonexecuted matrices do not require rotation workspace.
    pub rotation_workspace_width: usize,
}

/// Checked logical resident payload categories, not a reservation or measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderPayloadPlan {
    matrices: usize,
    preserved: usize,
    state: usize,
    workspace: usize,
    total: usize,
}
impl DecoderPayloadPlan {
    /// Construct a backend declaration with checked category addition.
    /// # Errors
    /// The total exceeds the platform's addressable extent.
    pub fn new(
        matrices: usize,
        preserved: usize,
        state: usize,
        workspace: usize,
    ) -> Result<Self, BackendError> {
        let total = matrices
            .checked_add(preserved)
            .and_then(|n| n.checked_add(state))
            .and_then(|n| n.checked_add(workspace))
            .ok_or_else(|| BackendError::InvalidInput("decoder payload overflow".into()))?;
        Ok(Self {
            matrices,
            preserved,
            state,
            workspace,
            total,
        })
    }
    /// Unique retained matrix payload, including unexecuted retained matrices.
    pub const fn matrix_payload_bytes(self) -> usize {
        self.matrices
    }
    /// Decoder/attention/MLP preserved vectors outside mixer-owned state.
    pub const fn preserved_payload_bytes(self) -> usize {
        self.preserved
    }
    /// KV caches and DeltaNet preserved coefficients plus state pairs.
    pub const fn state_payload_bytes(self) -> usize {
        self.state
    }
    /// Shared graph workspace, token IDs, constants and output buffers.
    pub const fn workspace_payload_bytes(self) -> usize {
        self.workspace
    }
    /// Sum of all four categories, excluding allocator/driver/host overhead.
    pub const fn payload_bytes(self) -> usize {
        self.total
    }
}
