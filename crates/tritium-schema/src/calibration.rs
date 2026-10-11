//! Candidate source-bound calibration requests and factor views (ADR 0060).
use crate::{BlobId, FitInputDigest, OriginalSourceId, QuantSourceTensor};
use alloc::vec::Vec;

/// Declared identities of the already finalized calibration source/cache.
/// The collector binds them; cache/source producers must independently verify them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationSource {
    /// Original source content identity, not a fitted model/package identity.
    pub model: FitInputDigest,
    /// Finalized activation/gradient cache manifest and content identity.
    pub cache: FitInputDigest,
    /// Ordered token stream and tokenizer provenance identity.
    pub tokens: FitInputDigest,
}

/// Input statistics partition, independent of scale groups and basis blocks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum CalibrationPartition {
    /// Full original-input covariance, including cross-group correlations.
    Full,
    /// Explicit block-diagonal approximation; final group may be ragged.
    Groups {
        /// Positive block width, not a weight-format admission constraint.
        columns: u32,
    },
}

/// Declared output factor encoding; no gradients are inferred from activations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum CalibrationOutput {
    /// Input Gram only, without a Fisher claim.
    None,
    /// One dense output-gradient vector per sample.
    DenseFisher,
    /// One named output-row factor per sample, zeros in other rows.
    IndexedFisher,
}

/// Consumer whose original-source values are being calibrated.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum CalibrationConsumer {
    /// Raw original-input activations of a linear projection.
    Matmul,
    /// Gather with the explicitly declared hidden-space identity approximation.
    /// Requires indexed output factors; does not materialize vocabulary one-hots.
    GatherIdentity,
}

/// Candidate request for one tensor's original-basis calibration factors.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationPlan {
    /// Exact original matrix interpretation, not fitted weights.
    pub tensor: QuantSourceTensor,
    /// Declared finalized source/cache/token binding.
    pub source: CalibrationSource,
    /// Explicit statistics partition.
    pub partition: CalibrationPartition,
    /// Actual output encoding required for every batch.
    pub output: CalibrationOutput,
    /// Actual consuming operation and approximation.
    pub consumer: CalibrationConsumer,
    /// Canonical global ordinal of the first sample in this shard.
    pub sample_start: u64,
    /// Absolute diagonal damping after output-factor scaling.
    pub damping: f64,
    /// Bound on numerical reducer/transaction/finalization workspace bytes.
    /// Excludes caller buffers, metadata/allocator overhead and full process RSS.
    pub max_numeric_bytes: u64,
}

/// Invalid candidate calibration request.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationPlanError {
    /// Invalid original source name/precision/shape or not a matrix.
    Source,
    /// At least one declared provenance digest is missing.
    MissingSource,
    /// Group width is zero.
    Partition,
    /// An unsupported consumer/output-factor combination.
    Consumer,
    /// Nonfinite/negative damping, or unused damping for input-only statistics.
    Damping,
    /// Numerical byte budget is zero.
    Budget,
}

impl CalibrationPlan {
    /// Validate the candidate interpretation before numerical allocation.
    /// This does not prove the declared source/cache identities or full RSS.
    pub fn validate(&self) -> Result<(), CalibrationPlanError> {
        if self.tensor.shape.len() != 2 || self.tensor.byte_len().is_err() {
            return Err(CalibrationPlanError::Source);
        }
        if [self.source.model, self.source.cache, self.source.tokens]
            .iter()
            .any(|digest| digest.0 == [0; 32])
        {
            return Err(CalibrationPlanError::MissingSource);
        }
        if matches!(self.partition, CalibrationPartition::Groups { columns: 0 }) {
            return Err(CalibrationPlanError::Partition);
        }
        if self.consumer == CalibrationConsumer::GatherIdentity
            && self.output != CalibrationOutput::IndexedFisher
        {
            return Err(CalibrationPlanError::Consumer);
        }
        if !self.damping.is_finite()
            || self.damping < 0.0
            || (self.output == CalibrationOutput::None && self.damping != 0.0)
        {
            return Err(CalibrationPlanError::Damping);
        }
        if self.max_numeric_bytes == 0 {
            return Err(CalibrationPlanError::Budget);
        }
        Ok(())
    }
}

/// View of one actual normalized factor and its existing numerical digest domain.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationFactor {
    /// Gram input width or Fisher output-row count.
    pub dimension: u64,
    /// Observed samples, including masked samples.
    pub samples: u64,
    /// Selected samples, including selected zero-weight samples.
    pub selected: u64,
    /// Sum of selected weights used to normalize the factor.
    pub total_weight: f64,
    /// Exact source/ordinal/weight/mask selection digest.
    pub selection: FitInputDigest,
    /// Existing InputGram or OutputFisher content digest, not a raw Gram digest.
    pub content: FitInputDigest,
}

/// Actual collector output view, not a registered receipt or release verdict.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationFactors {
    /// Request actually used by this collector.
    pub plan: CalibrationPlan,
    /// Original-input group order, with each actual width recorded.
    pub inputs: Vec<CalibrationFactor>,
    /// Actual output factor, absent for input-only statistics.
    pub output: Option<CalibrationFactor>,
}

/// Logical retained cache bounds, separate from reducer workspace or process RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCacheLimits {
    /// Exact retained scalar/token/mask and recipe bytes; excludes metadata/capacity overhead.
    pub max_payload_bytes: u64,
    /// Maximum observed samples, including masked samples.
    pub max_samples: u64,
    /// Maximum declared model-context windows, not reduction replay chunks.
    pub max_windows: u64,
}

/// Context for collecting one tensor's actual original-basis paired cache.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCacheRequest {
    /// Exact original interpretation, not fitted weights.
    pub tensor: QuantSourceTensor,
    /// Declared original identity; the engine must verify actual byte equality.
    pub model: OriginalSourceId,
    /// Actual operation/explicit approximation producing the inputs.
    pub consumer: CalibrationConsumer,
    /// Required actual gradient encoding; absent is explicitly input-only.
    pub output: CalibrationOutput,
    /// First canonical sample ordinal, retained across replay/shards.
    pub sample_start: u64,
    /// Logical cache admission, not numerical reduction admission.
    pub limits: CalibrationCacheLimits,
}
impl CalibrationCacheRequest {
    /// Validate context without creating fake cache/token identities.
    /// # Errors
    /// Invalid original geometry, missing model identity, consumer or logical budget.
    pub fn validate(&self) -> Result<(), CalibrationPlanError> {
        if self.tensor.shape.len() != 2 || self.tensor.byte_len().is_err() {
            return Err(CalibrationPlanError::Source);
        }
        if self.model.as_bytes() == &[0; 32] {
            return Err(CalibrationPlanError::MissingSource);
        }
        if self.consumer == CalibrationConsumer::GatherIdentity
            && self.output != CalibrationOutput::IndexedFisher
        {
            return Err(CalibrationPlanError::Consumer);
        }
        if self.limits.max_payload_bytes == 0
            || self.limits.max_samples == 0
            || self.limits.max_windows == 0
        {
            return Err(CalibrationPlanError::Budget);
        }
        Ok(())
    }
}

/// Immutable finalized cache context, not a registered/audited observation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCacheBinding {
    /// Context actually used, including declared logical limits.
    pub request: CalibrationCacheRequest,
    /// Actual finalized cache/token identities; model identity remains declared until engine admission.
    pub source: CalibrationSource,
    /// BLAKE3 of actual supplied tokenizer bytes, not a name.
    pub tokenizer: BlobId,
    /// BLAKE3 of exact supplied objective/teacher/recovery recipe bytes.
    /// Byte identity does not independently establish that this recipe was executed.
    pub objective: BlobId,
    /// Observed sample count, including masked samples.
    pub samples: u64,
    /// Declared context-window count, not the number of reduction chunks.
    pub windows: u64,
    /// Actual retained payload bytes, excluding metadata/capacity overhead and full RSS.
    pub payload_bytes: u64,
}

/// Candidate typed manifest carried by the existing `.trit` cache transport.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCacheManifest {
    /// Claimed binding; readers must reconstruct it from actual payload bytes.
    pub binding: CalibrationCacheBinding,
    /// Ordered declared model-context lengths, not physical replay chunks.
    pub window_samples: Vec<u64>,
}
impl CalibrationCacheManifest {
    /// Check structural counts and both producer and receiving logical admission.
    /// This does not verify payloads, identities or full process RSS.
    pub fn validate(&self, limits: CalibrationCacheLimits) -> Result<(), CalibrationPlanError> {
        self.binding.request.validate()?;
        let b = &self.binding;
        let declared = b.request.limits;
        if self.window_samples.is_empty()
            || self.window_samples.len() as u64 != b.windows
            || self.window_samples.contains(&0)
            || self
                .window_samples
                .iter()
                .try_fold(0u64, |sum, n| sum.checked_add(*n))
                != Some(b.samples)
            || b.request.sample_start.checked_add(b.samples).is_none()
        {
            return Err(CalibrationPlanError::Source);
        }
        if [declared, limits].iter().any(|l| {
            l.max_payload_bytes == 0
                || l.max_samples == 0
                || l.max_windows == 0
                || b.payload_bytes > l.max_payload_bytes
                || b.samples > l.max_samples
                || b.windows > l.max_windows
        }) {
            return Err(CalibrationPlanError::Budget);
        }
        Ok(())
    }
}
