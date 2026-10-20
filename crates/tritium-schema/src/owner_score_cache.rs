//! Candidate immutable scalar-window owner observations (ADR 0093).
use crate::{BlobId, OriginalSourceId, OwnerScoreCacheId, QuantSourceTensor};
use alloc::{string::String, vec::Vec};

/// Estimator unit; no Fisher/GGN interpretation is inferred from scalar adjoints.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum OwnerScoreEstimator {
    /// One declared scalar objective's complete owner-row gradient.
    ScalarWindowGradient,
}

/// Ordinary owner operation, in canonical source coordinates.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum OwnerScoreRole {
    /// `z = x W^T`, without omitted source uses.
    Matmul,
    /// Ordinary lookup, without padding/frequency/custom backward rules.
    Gather,
}

/// Context scalar interpretation; payload bytes are contiguous logical values.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum OwnerContextDtype {
    /// IEEE binary16.
    F16,
    /// Brain float16.
    Bf16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Boolean bytes, exactly 0/1.
    Bool,
    /// Complex pair of binary32 components.
    C64,
    /// Complex pair of binary64 components.
    C128,
}
impl OwnerContextDtype {
    /// Logical bytes per scalar, excluding metadata/allocator overhead.
    #[must_use]
    pub const fn width(self) -> u64 {
        match self {
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }
}

/// Declared logical tensor context, not proof that a callback used it.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerContextMeta {
    /// Unique nonempty context key.
    pub name: String,
    /// Exact original scalar dtype.
    pub dtype: OwnerContextDtype,
    /// Original logical dimensions; scalar rank zero is allowed.
    pub shape: Vec<u64>,
    /// Original conjugate-view flag; stored values are resolved.
    pub conjugated: bool,
    /// Original negative-view flag; stored values are resolved.
    pub negated: bool,
}
impl OwnerContextMeta {
    /// Checked logical byte count; not an allocation or RSS admission.
    pub fn byte_len(&self) -> Option<u64> {
        if self.name.is_empty() || self.shape.len() > 16 {
            return None;
        }
        self.shape
            .iter()
            .try_fold(self.dtype.width(), |n, d| n.checked_mul(*d))
    }
}

/// One declared actual invocation; bytes/coverage are verified separately.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerInvocationMeta {
    /// Global operator-entry ordinal, not renumbered per owner.
    pub ordinal: u64,
    /// Sorted unique alias paths of one actual module; root path may be empty.
    pub paths: Vec<String>,
    /// Actual ordinary operation.
    pub role: OwnerScoreRole,
    /// Positive flattened invocation sample count.
    pub samples: u64,
}
impl OwnerInvocationMeta {
    /// Structural interpretation only; does not prove source/graph execution.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.samples > 0 && !self.paths.is_empty() && self.paths.windows(2).all(|p| p[0] < p[1])
    }
}

/// Complete context/invocation metadata supplied alongside actual buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreWindowMeta {
    /// Sorted logical context keys.
    pub context: Vec<OwnerContextMeta>,
    /// Strictly ordered invocations for this owner.
    pub invocations: Vec<OwnerInvocationMeta>,
}

/// Actual context-byte binding.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerContextBinding {
    /// Original context interpretation.
    pub meta: OwnerContextMeta,
    /// Exact logical-value byte digest.
    pub content: BlobId,
}

/// Actual invocation input/adjoint binding.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerInvocationBinding {
    /// Declared invocation interpretation.
    pub meta: OwnerInvocationMeta,
    /// Exact f32 input or i64 index byte digest.
    pub input: BlobId,
    /// Exact f32 output-adjoint byte digest.
    pub adjoint: BlobId,
}

/// Logical identity preimage; excludes physical package layout and ceilings.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreCacheContent {
    /// Original complete source, not fitted model/package identity.
    pub source: OriginalSourceId,
    /// Exact original matrix interpretation.
    pub tensor: QuantSourceTensor,
    /// Explicit scalar-window unit; no Fisher/GGN fallback.
    pub estimator: OwnerScoreEstimator,
    /// Actual tokenizer bytes.
    pub tokenizer: BlobId,
    /// Actual declared objective bytes, not independent execution proof.
    pub objective: BlobId,
    /// Sorted exact context descriptors and content.
    pub context: Vec<OwnerContextBinding>,
    /// Ordered exact invocation descriptors and numerical content.
    pub invocations: Vec<OwnerInvocationBinding>,
}

/// Positive logical snapshot admission; excludes copies/metadata/tape/full RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreCacheLimits {
    /// Total tokenizer/objective/context/numerical payload bytes.
    pub max_payload_bytes: u64,
    /// Total context bytes, excluding recipe/tokenizer.
    pub max_context_bytes: u64,
    /// Number of distinct context tensors.
    pub max_context_tensors: u64,
    /// Number of actual invocations.
    pub max_invocations: u64,
    /// Sum of invocation samples, not token/window count.
    pub max_samples: u64,
}
impl OwnerScoreCacheLimits {
    /// All logical ceilings must be positive.
    #[must_use]
    pub const fn valid(self) -> bool {
        self.max_payload_bytes > 0
            && self.max_context_bytes > 0
            && self.max_context_tensors > 0
            && self.max_invocations > 0
            && self.max_samples > 0
    }
}

/// Actual immutable identity/counts; recomputed by the reader.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreCacheBinding {
    /// Complete logical content preimage.
    pub content: OwnerScoreCacheContent,
    /// Derived actual content identity.
    pub cache: OwnerScoreCacheId,
    /// Derived total payload bytes.
    pub payload_bytes: u64,
    /// Derived context bytes.
    pub context_bytes: u64,
    /// Derived sum of invocation samples.
    pub samples: u64,
}

/// Typed one-owner/one-window profile in the common .trit envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreCacheManifest {
    /// Claimed content/counts, untrusted until reconstructed.
    pub binding: OwnerScoreCacheBinding,
    /// Producer logical ceilings; excluded from content identity.
    pub limits: OwnerScoreCacheLimits,
}
impl OwnerScoreCacheManifest {
    /// Structural counts/geometry admission before decoded payload allocation.
    /// Content identity/numerical/source truth must be checked separately.
    #[must_use]
    pub fn valid(&self, receiving: OwnerScoreCacheLimits) -> bool {
        valid_owner_score_binding(&self.binding, self.limits, receiving)
    }
}

pub(crate) fn valid_owner_score_binding(
    b: &OwnerScoreCacheBinding,
    l: OwnerScoreCacheLimits,
    receiving: OwnerScoreCacheLimits,
) -> bool {
    let c = &b.content;
    if !l.valid()
        || !receiving.valid()
        || c.source.as_bytes() == &[0; 32]
        || c.tensor.shape.len() != 2
        || c.tensor.byte_len().is_err()
        || c.tokenizer.as_bytes() == &[0; 32]
        || c.objective.as_bytes() == &[0; 32]
        || b.cache.as_bytes() == &[0; 32]
        || c.context.is_empty()
        || c.invocations.is_empty()
        || c.context.len() as u64 > l.max_context_tensors.min(receiving.max_context_tensors)
        || c.invocations.len() as u64 > l.max_invocations.min(receiving.max_invocations)
        || !c
            .context
            .windows(2)
            .all(|p| p[0].meta.name < p[1].meta.name)
        || !c
            .invocations
            .windows(2)
            .all(|p| p[0].meta.ordinal < p[1].meta.ordinal)
        || c.invocations.iter().any(|i| !i.meta.valid())
    {
        return false;
    }
    let context = c
        .context
        .iter()
        .try_fold(0u64, |n, t| n.checked_add(t.meta.byte_len()?));
    let samples = c
        .invocations
        .iter()
        .try_fold(0u64, |n, i| n.checked_add(i.meta.samples));
    let tokens = c.context.iter().find(|t| t.meta.name == "input_ids");
    context == Some(b.context_bytes)
        && samples == Some(b.samples)
        && b.context_bytes <= l.max_context_bytes.min(receiving.max_context_bytes)
        && b.payload_bytes <= l.max_payload_bytes.min(receiving.max_payload_bytes)
        && b.payload_bytes > b.context_bytes
        && b.samples <= l.max_samples.min(receiving.max_samples)
        && tokens.is_some_and(|t| {
            matches!(
                t.meta.dtype,
                OwnerContextDtype::I32 | OwnerContextDtype::I64
            ) && t.meta.byte_len().is_some_and(|n| n > 0)
        })
}
