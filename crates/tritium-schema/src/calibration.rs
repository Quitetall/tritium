//! Candidate source-bound calibration requests and factor views (ADR 0060).
use crate::{FitInputDigest, QuantSourceTensor};
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
