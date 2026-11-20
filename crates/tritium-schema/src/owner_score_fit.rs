//! Candidate owner-window fit vocabulary, distinct from per-token K-FAC.
use crate::{
    DirectFitOutput, DirectFitPlan, FitInputDigest, OwnerMetricRowsId, OwnerMetricStreamId,
    OwnerRowMetricId, OwnerScoreCacheBinding, OwnerScoreCacheId, OwnerScoreCacheLimits,
    OwnerScoreEstimator,
};
use alloc::vec::Vec;

/// One explicitly ordered expected scalar-loss window; repetition is intentional.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct WeightedOwnerWindow {
    /// Finalized cache content ID, not a declared bootstrap identity.
    pub cache: OwnerScoreCacheId,
    /// Nonnegative finite scalar-window weight, normalized by their sum.
    pub weight: f64,
}

/// Logical admission only, excluding source/tapes/metadata/solver/allocator/RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreMetricLimits {
    /// Maximum ordered window count, including repeats and zero weights.
    pub max_windows: u64,
    /// Sum of logical cache payload bytes, counting each list position.
    pub max_payload_bytes: u64,
    /// Maximum one reconstructed row's f64 output bytes.
    pub max_row_bytes: u64,
    /// Conservative bound on reducer numerical f64 entries and row output.
    pub max_reduction_entries: u64,
}
impl OwnerScoreMetricLimits {
    /// Shared checked numerical admission; not downstream fitting or RSS.
    pub fn admits(&self, columns: u64, windows: u64, payload: u64) -> bool {
        if columns == 0
            || windows == 0
            || self.max_windows == 0
            || windows > self.max_windows
            || self.max_payload_bytes == 0
            || payload > self.max_payload_bytes
            || self.max_row_bytes == 0
            || self.max_reduction_entries == 0
        {
            return false;
        }
        let depth = u64::from(u64::BITS - windows.leading_zeros()) + 1;
        let entries = columns
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(4 * depth + 8))
            .and_then(|n| n.checked_add(columns));
        columns
            .checked_mul(8)
            .is_some_and(|n| n <= self.max_row_bytes)
            && entries.is_some_and(|n| n <= self.max_reduction_entries)
    }
}

/// A complete immutable cache interpretation and its scalar-window weight.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct WeightedOwnerCache {
    /// Expected complete binding, verified against actual cache content at execution.
    pub binding: OwnerScoreCacheBinding,
    /// Nonnegative finite window weight.
    pub weight: f64,
}
impl WeightedOwnerCache {
    /// The same ordered numerical window request used by the metric consumer.
    pub fn window(&self) -> WeightedOwnerWindow {
        WeightedOwnerWindow {
            cache: self.binding.cache,
            weight: self.weight,
        }
    }
}

/// Expected actual cache stream and numerical request, not source custody proof.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreFitPlan {
    /// Common original-row fitter and complete independent level requests.
    pub direct: DirectFitPlan,
    /// Ordered complete immutable scalar-loss window bindings.
    pub windows: Vec<WeightedOwnerCache>,
    /// Absolute source-basis row-metric diagonal damping.
    pub damping: f64,
    /// Logical cache/reducer limits, independent of direct fitter limits.
    pub limits: OwnerScoreMetricLimits,
    /// Positive maximum recorded row identities; not event bytes or process RSS.
    pub max_row_bindings: u64,
}

/// Invalid owner-fit structure or logical admission; numerical truth is separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnerScorePlanError {
    /// Invalid or differing cache/source/tensor/recipe interpretation.
    Binding,
    /// Invalid damping/window weights, including zero or overflowing total.
    Configuration,
    /// Missing or exceeded logical limits, or arithmetic overflow.
    Limit,
}
impl OwnerScoreFitPlan {
    /// Structural cross-field validation; cache identity/bytes are checked separately.
    pub fn validate(&self) -> Result<(), OwnerScorePlanError> {
        use OwnerScorePlanError as E;
        let first = self.windows.first().ok_or(E::Binding)?;
        let content = &first.binding.content;
        if self.direct.layout.validate().is_err()
            || content.tensor.shape.as_slice() != [self.direct.layout.rows, self.direct.layout.cols]
            || self.max_row_bindings == 0
        {
            return Err(E::Binding);
        }
        if self.direct.layout.rows > self.max_row_bindings {
            return Err(E::Limit);
        }
        let mut payload = 0_u64;
        let mut total = 0.0_f64;
        for window in &self.windows {
            let b = &window.binding;
            let c = &b.content;
            if c.source != content.source
                || c.tensor != content.tensor
                || c.tokenizer != content.tokenizer
                || c.objective != content.objective
                || c.estimator != OwnerScoreEstimator::ScalarWindowGradient
                || c.context
                    .iter()
                    .any(|value| value.content.as_bytes() == &[0; 32])
                || c.invocations.iter().any(|value| {
                    value.input.as_bytes() == &[0; 32] || value.adjoint.as_bytes() == &[0; 32]
                })
            {
                return Err(E::Binding);
            }
            let structural = OwnerScoreCacheLimits {
                max_payload_bytes: self.limits.max_payload_bytes,
                max_context_bytes: self.limits.max_payload_bytes,
                max_context_tensors: c.context.len() as u64,
                max_invocations: c.invocations.len() as u64,
                max_samples: b.samples,
            };
            if !crate::owner_score_cache::valid_owner_score_binding(b, structural, structural) {
                return Err(E::Binding);
            }
            if !window.weight.is_finite() || window.weight < 0.0 {
                return Err(E::Configuration);
            }
            total += window.weight;
            payload = payload.checked_add(b.payload_bytes).ok_or(E::Limit)?;
        }
        if !total.is_finite() || total <= 0.0 || !self.damping.is_finite() || self.damping < 0.0 {
            return Err(E::Configuration);
        }
        if !self
            .limits
            .admits(self.direct.layout.cols, self.windows.len() as u64, payload)
        {
            return Err(E::Limit);
        }
        Ok(())
    }
}

/// One actual completed source-basis row metric; its matrix is not stored here.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerRowMetricBinding {
    /// Canonical source row ordinal, including zero-score rows.
    pub row: u64,
    /// Finalized actual row metric identity, not an opaque raw-Gram label.
    pub content: OwnerRowMetricId,
}

/// Actual owner-window numerical execution, not model Direct or qualification.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerScoreLevelsFitted {
    /// Executed controller, complete actual window bindings and logical limits.
    pub plan: OwnerScoreFitPlan,
    /// Actual original f32 input bits, not original BF16 file content.
    pub source_f32: FitInputDigest,
    /// Actual ordered weighted cache stream and damping identity.
    pub stream: OwnerMetricStreamId,
    /// Complete source-row metric inventory, recorded during fitting.
    pub rows: Vec<OwnerRowMetricBinding>,
    /// Content identity of ordered actual row metric references and stream.
    pub rows_content: OwnerMetricRowsId,
    /// Actual executing Rayon pool size.
    pub rayon_threads: u32,
    /// Actual fitted levels in requested order.
    pub outputs: Vec<DirectFitOutput>,
}
