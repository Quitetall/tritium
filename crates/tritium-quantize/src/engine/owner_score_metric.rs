//! Actual durable windows -> f64 row Gram -> unchanged common direct fitter.
use super::calibrated_fit::{RowMetricSource, fit_row_metrics};
use super::direct::reserved_vec;
use super::direct_metric::{DirectMetric, MetricBlock};
use super::{DirectFitError, DirectTensorFit, OwnerScoreCache, OwnerScoreCacheError};
use crate::salt_v2_curvature::OwnerGramStrip;
use crate::{CurvatureError, InputGramAccumulator};
use std::borrow::Cow;
use tritium_schema::{
    Basis, DirectFitPlan, DirectFitProfile, OriginalSourceId, OwnerMetricStreamId,
    QuantSourceTensor,
};
pub use tritium_schema::{OwnerScoreMetricLimits, WeightedOwnerWindow};

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
        // Retain v1's conservative scalar bound even when consuming finalization
        // and ownership transfer need fewer matrices. This is not downstream
        // solver/source/process-RSS admission and does not widen any old limit.
        if !limits.admits(columns as u64, caches.len() as u64, 0) {
            return Err(E::Limit);
        }
        let mut payload = 0_u64;
        let mut total_weight = 0.0_f64;
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
        }
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(E::Configuration);
        }
        if payload > limits.max_payload_bytes {
            return Err(E::Limit);
        }
        let stream = tritium_evidence::owner_metric_stream_id(expected, damping)
            .map_err(|_| E::Configuration)?;
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
            stream: *stream.as_bytes(),
            columns,
        })
    }

    /// Source observed when constructing the actual caches, not audited custody.
    pub fn source(&self) -> OriginalSourceId {
        self.caches[0].binding().content.source
    }
    /// Actual admitted ordered window/weight/damping identity.
    pub fn stream_id(&self) -> OwnerMetricStreamId {
        OwnerMetricStreamId::from_bytes(self.stream)
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
        let mut values = reducer.finish_owned().map_err(E::Reduction)?.into_values();
        damp_values(&mut values, self.columns, 0, self.damping).map_err(E::Reduction)?;
        let mut hash = self.row_hasher(row);
        hash_values(&mut hash, &values);
        Ok(OwnerRowMetric {
            values,
            digest: *hash.finalize().as_bytes(),
        })
    }

    fn row_hasher(&self, row: usize) -> blake3::Hasher {
        let mut hash = blake3::Hasher::new_derive_key("tritium.owner-window.row-metric.v1");
        hash.update(&self.stream);
        hash.update(&(row as u64).to_le_bytes());
        hash.update(&(self.columns as u64).to_le_bytes());
        hash
    }

    // Keep v1 admission intact. The private storage choice must additionally
    // bound retained scores, transactional strips and solver-local matrices.
    fn streaming_group(&self, plan: &DirectFitPlan) -> Option<usize> {
        let group = usize::from(plan.layout.group);
        if plan.layout.basis != Basis::Identity
            || !matches!(plan.profile, DirectFitProfile::JointIndependentGroups(_))
            || group == 0
            || group >= self.columns
        {
            return None;
        }
        streaming_storage_fits(
            self.columns as u64,
            self.windows.len() as u64,
            group as u64,
            self.limits.max_reduction_entries,
        )
        .then_some(group)
    }

    fn streamed_row(
        &self,
        row: usize,
        group: usize,
    ) -> Result<(DirectMetric<'static>, [u8; 32]), OwnerScoreMetricError> {
        use OwnerScoreMetricError as E;
        let mut scores = reserved_vec(self.caches.len())?;
        for cache in self.caches {
            scores.push(
                cache
                    .row_score(row, self.limits.max_row_bytes)
                    .map_err(E::Cache)?,
            );
        }
        let mut hash = self.row_hasher(row);
        let mut blocks = reserved_vec(self.columns.div_ceil(group))?;
        for start in (0..self.columns).step_by(group) {
            let end = (start + group).min(self.columns);
            let mut reducer =
                OwnerGramStrip::new(self.columns, start..end).map_err(E::Reduction)?;
            for (score, window) in scores.iter().zip(&self.windows) {
                reducer
                    .accumulate(score, window.weight)
                    .map_err(E::Reduction)?;
            }
            let mut values = reducer.finish().map_err(E::Reduction)?;
            damp_values(&mut values, self.columns, start, self.damping).map_err(E::Reduction)?;
            // Hash every complete source-basis strip, not only local entries.
            hash_values(&mut hash, &values);
            let width = end - start;
            let mut local = reserved_vec(width * width)?;
            for local_row in 0..width {
                local.extend_from_slice(
                    &values[local_row * self.columns + start..local_row * self.columns + end],
                );
            }
            blocks.push(MetricBlock {
                width,
                values: Cow::Owned(local),
            });
        }
        Ok((
            DirectMetric::IndependentGroups { group, blocks },
            *hash.finalize().as_bytes(),
        ))
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
    fit_owner_score_levels_observed(plan, original, model, weights, metrics, |_, _| Ok(()))
}

pub(super) fn fit_owner_score_levels_observed(
    plan: &DirectFitPlan,
    original: &QuantSourceTensor,
    model: OriginalSourceId,
    weights: &[f32],
    metrics: &OwnerScoreMetrics<'_>,
    observe: impl FnMut(usize, [u8; 32]) -> Result<(), OwnerScoreMetricError>,
) -> Result<Vec<DirectTensorFit>, OwnerScoreMetricError> {
    if metrics.source() != model
        || metrics.tensor() != original
        || original.shape.as_slice() != [plan.layout.rows, plan.layout.cols]
    {
        return Err(OwnerScoreMetricError::Binding);
    }
    fit_row_metrics(
        plan,
        weights,
        OwnerRows {
            metrics,
            observe,
            group: metrics.streaming_group(plan),
        },
    )
}

struct OwnerRows<'a, 'b, F> {
    metrics: &'a OwnerScoreMetrics<'b>,
    observe: F,
    group: Option<usize>,
}
impl<F: FnMut(usize, [u8; 32]) -> Result<(), OwnerScoreMetricError>> RowMetricSource
    for OwnerRows<'_, '_, F>
{
    type Error = OwnerScoreMetricError;
    fn metric(&mut self, row: usize, _: usize) -> Result<DirectMetric<'_>, Self::Error> {
        if let Some(group) = self.group {
            let (metric, digest) = self.metrics.streamed_row(row, group)?;
            (self.observe)(row, digest)?;
            return Ok(metric);
        }
        let metric = self.metrics.row_metric(row)?;
        (self.observe)(row, metric.digest())?;
        Ok(DirectMetric::Dense(Cow::Owned(metric.into_values())))
    }
}

fn damp_values(
    values: &mut [f64],
    columns: usize,
    start_row: usize,
    damping: f64,
) -> Result<(), CurvatureError> {
    for (index, value) in values.iter_mut().enumerate() {
        let damped = *value
            + if start_row + index / columns == index % columns {
                damping
            } else {
                0.
            };
        if !damped.is_finite() {
            return Err(CurvatureError::NonFiniteAccumulation);
        }
        *value = if damped == 0. { 0. } else { damped };
    }
    Ok(())
}

fn hash_values(hash: &mut blake3::Hasher, values: &[f64]) {
    for &value in values {
        hash.update(&bits(value).to_le_bytes());
    }
}

fn bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

// Extra private storage admission, never a replacement for the unchanged v1
// full-metric limits checked by OwnerScoreMetrics::new and the direct controller.
fn streaming_storage_fits(cols: u64, windows: u64, group: u64, limit: u64) -> bool {
    if cols == 0 || windows == 0 || group == 0 || group >= cols {
        return false;
    }
    let counts = || {
        let factor = 4 * (u64::from(u64::BITS - windows.leading_zeros()) + 1) + 8;
        let strip = cols.checked_mul(group)?;
        let required = cols
            .checked_mul(windows)?
            .checked_add(strip.checked_mul(factor)?)?
            .checked_add(strip)?
            .checked_add(cols)?;
        let dense = cols
            .checked_mul(cols)?
            .checked_mul(factor)?
            .checked_add(cols)?;
        Some((required, dense))
    };
    counts().is_some_and(|(required, dense)| required <= limit && required < dense)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_storage_is_additionally_bounded_and_never_forces_worse_retention() {
        assert!(streaming_storage_fits(512, 1, 32, 279_552));
        assert!(!streaming_storage_fits(512, 1, 32, 279_551));
        assert!(streaming_storage_fits(35, 5, 32, 100_000));
        assert!(!streaming_storage_fits(35, 1000, 32, u64::MAX));
        for (cols, windows, group) in [
            (0, 1, 32),
            (64, 0, 32),
            (64, 1, 0),
            (32, 1, 32),
            (u64::MAX, 1, 32),
            (512, u64::MAX, 32),
        ] {
            assert!(!streaming_storage_fits(cols, windows, group, u64::MAX));
        }
    }
}
