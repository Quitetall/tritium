//! Actual durable windows -> f64 row Gram -> unchanged common direct fitter.
use super::calibrated_fit::fit_row_metrics;
use super::{DirectFitError, DirectTensorFit, OwnerScoreCache, OwnerScoreCacheError};
use crate::{CurvatureError, InputGramAccumulator};
use tritium_schema::{DirectFitPlan, OriginalSourceId, OwnerScoreCacheId, QuantSourceTensor};

/// One explicitly ordered expected scalar-loss window. Repetition is intentional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedOwnerWindow {
    /// Actual finalized cache content ID, never a bootstrap/declared placeholder.
    pub cache: OwnerScoreCacheId,
    /// Nonnegative finite window weight; normalization uses the sum of weights.
    pub weight: f64,
}

/// Logical admission only, excluding source/tapes/metadata/solver/allocator/RSS.
#[derive(Clone, Copy, Debug)]
pub struct OwnerScoreMetricLimits {
    /// Maximum ordered window count, including repeats and zero weights.
    pub max_windows: u64,
    /// Sum of logical cache payload bytes, counting each list position.
    pub max_payload_bytes: u64,
    /// Maximum one reconstructed row's f64 output bytes.
    pub max_row_bytes: u64,
    /// Conservative bound on live reducer numerical f64 entries and row output.
    pub max_reduction_entries: u64,
}

/// No identity/fitted success is returned on invalid context, limits or numerics.
#[derive(Debug)]
#[non_exhaustive]
pub enum OwnerScoreMetricError {
    /// Expected IDs or common source/tensor/tokenizer/objective differ.
    Binding,
    /// Empty/invalid window weights or damping, including zero/overflowed total.
    Configuration,
    /// Logical admission, checked geometry or allocation failed.
    Limit,
    /// Shared cache row reconstruction failed.
    Cache(OwnerScoreCacheError),
    /// Shared finite deterministic Gram reduction failed.
    Reduction(CurvatureError),
    /// Common direct fitting/admission/output failed.
    Fit(DirectFitError),
}
impl core::fmt::Display for OwnerScoreMetricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "owner row metric rejected: {self:?}")
    }
}
impl std::error::Error for OwnerScoreMetricError {}
impl From<DirectFitError> for OwnerScoreMetricError {
    fn from(value: DirectFitError) -> Self {
        Self::Fit(value)
    }
}

/// Completed full row-block metric, not Fisher/GGN or a qualification receipt.
#[derive(Debug)]
pub struct OwnerRowMetric {
    values: Vec<f64>,
    digest: [u8; 32],
}
impl OwnerRowMetric {
    /// Canonical source-basis row-major f64 matrix.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
    /// Finalized row identity binding actual caches, weights, damping and values.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    /// Transfer the completed numerical matrix without another data copy.
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }
}

/// Validated actual immutable cache stream; does not materialize all row scores.
#[derive(Debug)]
pub struct OwnerScoreMetrics<'a> {
    caches: &'a [&'a OwnerScoreCache],
    windows: Vec<WeightedOwnerWindow>,
    damping: f64,
    limits: OwnerScoreMetricLimits,
    stream: [u8; 32],
    columns: usize,
}
impl<'a> OwnerScoreMetrics<'a> {
    /// Bind an ordered weighted list against actual complete cache identities.
    /// Common source/tensor/estimator/tokenizer/objective must match; contexts may
    /// differ. Limits are checked before any numerical matrix or row allocation.
    /// No declared source hash is independent proof of source/callback custody.
    pub fn new(
        caches: &'a [&'a OwnerScoreCache],
        expected: &[WeightedOwnerWindow],
        damping: f64,
        limits: OwnerScoreMetricLimits,
    ) -> Result<Self, OwnerScoreMetricError> {
        use OwnerScoreMetricError as E;
        if caches.is_empty() || caches.len() != expected.len() {
            return Err(E::Binding);
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(E::Configuration);
        }
        if limits.max_windows == 0
            || limits.max_payload_bytes == 0
            || limits.max_row_bytes == 0
            || limits.max_reduction_entries == 0
            || caches.len() as u64 > limits.max_windows
        {
            return Err(E::Limit);
        }
        let content = &caches[0].binding().content;
        let columns = usize::try_from(content.tensor.shape[1]).map_err(|_| E::Limit)?;
        let row_bytes = (columns as u64).checked_mul(8).ok_or(E::Limit)?;
        // Transactional updates and finalization clone canonical segment stacks.
        // Four stacks plus eight temporary matrices conservatively cover the
        // common reducer, returned Gram and row score, not the downstream fitter.
        let depth = u64::from(usize::BITS - caches.len().leading_zeros()) + 1;
        let entries = (columns as u64)
            .checked_mul(columns as u64)
            .and_then(|n| n.checked_mul(4 * depth + 8))
            .and_then(|n| n.checked_add(columns as u64))
            .ok_or(E::Limit)?;
        if row_bytes > limits.max_row_bytes || entries > limits.max_reduction_entries {
            return Err(E::Limit);
        }
        let mut payload = 0_u64;
        let mut total_weight = 0.0_f64;
        let mut hash = blake3::Hasher::new_derive_key("tritium.owner-window.metric-stream.v1");
        hash.update(&(caches.len() as u64).to_le_bytes());
        for (cache, window) in caches.iter().zip(expected) {
            let actual = &cache.binding().content;
            if cache.id() != window.cache
                || actual.source != content.source
                || actual.tensor != content.tensor
                || actual.estimator != content.estimator
                || actual.tokenizer != content.tokenizer
                || actual.objective != content.objective
            {
                return Err(E::Binding);
            }
            if !window.weight.is_finite() || window.weight < 0.0 {
                return Err(E::Configuration);
            }
            total_weight += window.weight;
            payload = payload
                .checked_add(cache.binding().payload_bytes)
                .ok_or(E::Limit)?;
            hash.update(cache.id().as_bytes());
            hash.update(&bits(window.weight).to_le_bytes());
        }
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(E::Configuration);
        }
        if payload > limits.max_payload_bytes {
            return Err(E::Limit);
        }
        hash.update(&bits(damping).to_le_bytes());
        let mut windows = Vec::new();
        windows
            .try_reserve_exact(expected.len())
            .map_err(|_| E::Limit)?;
        windows.extend_from_slice(expected);
        Ok(Self {
            caches,
            windows,
            damping,
            limits,
            stream: *hash.finalize().as_bytes(),
            columns,
        })
    }

    /// Source observed when constructing the actual caches, not audited custody.
    pub fn source(&self) -> OriginalSourceId {
        self.caches[0].binding().content.source
    }
    /// Common canonical tensor interpretation.
    pub fn tensor(&self) -> &QuantSourceTensor {
        &self.caches[0].binding().content.tensor
    }
    /// Reduce a single row via the shared f64-capable canonical Gram accumulator.
    /// Every admitted window is replayed, including zero-weight observations.
    pub fn row_metric(&self, row: usize) -> Result<OwnerRowMetric, OwnerScoreMetricError> {
        use OwnerScoreMetricError as E;
        let mut reducer = InputGramAccumulator::new(self.columns).map_err(E::Reduction)?;
        for (cache, window) in self.caches.iter().zip(&self.windows) {
            let score = cache
                .row_score(row, self.limits.max_row_bytes)
                .map_err(E::Cache)?;
            reducer
                .accumulate_owner_row(&score, window.weight)
                .map_err(E::Reduction)?;
        }
        let gram = reducer.finish().map_err(E::Reduction)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(gram.as_slice().len())
            .map_err(|_| E::Limit)?;
        for (index, &value) in gram.as_slice().iter().enumerate() {
            let value = value
                + if index / self.columns == index % self.columns {
                    self.damping
                } else {
                    0.0
                };
            if !value.is_finite() {
                return Err(E::Reduction(CurvatureError::NonFiniteAccumulation));
            }
            values.push(if value == 0.0 { 0.0 } else { value });
        }
        let mut hash = blake3::Hasher::new_derive_key("tritium.owner-window.row-metric.v1");
        hash.update(&self.stream);
        hash.update(&(row as u64).to_le_bytes());
        hash.update(&(self.columns as u64).to_le_bytes());
        for &value in &values {
            hash.update(&bits(value).to_le_bytes());
        }
        Ok(OwnerRowMetric {
            values,
            digest: *hash.finalize().as_bytes(),
        })
    }
}

/// Fit every owner row/level from actual windows through the common controller.
/// Caller f32 weights must be source-verified by the full model engine; this
/// runtime interface does not assign model Direct provenance or a fit receipt.
pub fn fit_owner_score_levels(
    plan: &DirectFitPlan,
    original: &QuantSourceTensor,
    model: OriginalSourceId,
    weights: &[f32],
    metrics: &OwnerScoreMetrics<'_>,
) -> Result<Vec<DirectTensorFit>, OwnerScoreMetricError> {
    if metrics.source() != model
        || metrics.tensor() != original
        || original.shape.as_slice() != [plan.layout.rows, plan.layout.cols]
    {
        return Err(OwnerScoreMetricError::Binding);
    }
    fit_row_metrics(plan, weights, |row, _, destination| {
        let metric = metrics.row_metric(row)?;
        destination.copy_from_slice(metric.as_slice());
        Ok(())
    })
}

fn bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}
