//! Same-pass actual row metric observation around the common owner fitter.
use super::direct::{admit_payload, admit_plan, reserved_vec};
use super::owner_score_metric::fit_owner_score_levels_observed;
use super::recorded::{fit_outputs, source_digest};
use super::{
    DirectFitError, DirectTensorFit, OwnerScoreCache, OwnerScoreMetricError, OwnerScoreMetrics,
};
use tritium_evidence::{
    EvidenceError, EvidenceRecorder, OwnerScoreFitReceiptView, owner_metric_rows_id,
};
use tritium_format::trit_model::TritModelError;
use tritium_schema::{
    OwnerRowMetricBinding, OwnerRowMetricId, OwnerScoreFitPlan, OwnerScoreLevelsFitted,
    OwnerScorePlanError,
};

/// Actual independent level outputs and immutable initial log-derived view.
#[derive(Debug)]
pub struct RecordedOwnerScoreFit {
    /// Actual owned fitted levels, not another level's prefix.
    pub levels: Vec<DirectTensorFit>,
    /// Initial execution observation, not complete model or release provenance.
    pub observation: OwnerScoreFitReceiptView,
}

/// No fitted success/event on context, admission, numerical or recording failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum RecordedOwnerScoreFitError {
    /// Invalid structural request or logical row/reducer admission.
    Plan(OwnerScorePlanError),
    /// Actual complete immutable cache bindings differ from the request.
    CacheBinding,
    /// Common direct controller admission failed before row reconstruction.
    Admission(DirectFitError),
    /// Shared owner metric/fitter rejected the numerical problem.
    Fit(OwnerScoreMetricError),
    /// Shared fitted output semantic identity failed.
    Semantic(TritModelError),
    /// Recorder/typed/canonical wire binding failed.
    Evidence(EvidenceError),
    /// Actual Rayon pool size cannot be represented.
    ThreadCountOverflow,
}
impl core::fmt::Display for RecordedOwnerScoreFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "recorded owner fit rejected: {self:?}")
    }
}
impl std::error::Error for RecordedOwnerScoreFitError {}

/// Execute actual immutable owner windows and record every consumed row identity.
/// Complete expected bindings, geometry, direct admission and recorder context
/// are checked first. Metrics are observed during the existing fit pass, never
/// reconstructed for recording. Source/output digests use existing shared code.
/// Original source/callback custody and complete model/package qualification are
/// separate; limits do not admit all metadata/allocator/tape/solver/process RSS.
pub fn fit_owner_score_levels_recorded(
    plan: &OwnerScoreFitPlan,
    weights: &[f32],
    caches: &[&OwnerScoreCache],
    recorder: &mut EvidenceRecorder,
    span: &str,
    parent: Option<&str>,
    logical_time: u64,
) -> Result<RecordedOwnerScoreFit, RecordedOwnerScoreFitError> {
    use RecordedOwnerScoreFitError as E;
    plan.validate().map_err(E::Plan)?;
    if caches.len() != plan.windows.len()
        || caches
            .iter()
            .zip(&plan.windows)
            .any(|(cache, window)| cache.binding() != &window.binding)
    {
        return Err(E::CacheBinding);
    }
    recorder
        .check_owner_score_fit_context(span, parent, logical_time, plan)
        .map_err(E::Evidence)?;
    let (rows, cols, count) = admit_plan(&plan.direct, true).map_err(E::Admission)?;
    let entries = cols
        .checked_mul(cols)
        .ok_or(E::Admission(DirectFitError::ShapeOverflow))?;
    admit_payload(weights, None, cols, count, entries).map_err(E::Admission)?;
    let mut windows = reserved_vec(plan.windows.len()).map_err(E::Admission)?;
    windows.extend(plan.windows.iter().map(|window| window.window()));
    let metrics =
        OwnerScoreMetrics::new(caches, &windows, plan.damping, plan.limits).map_err(E::Fit)?;
    let mut bindings = reserved_vec(rows).map_err(E::Admission)?;
    let rayon_threads =
        u32::try_from(rayon::current_num_threads()).map_err(|_| E::ThreadCountOverflow)?;
    let levels = fit_owner_score_levels_observed(
        &plan.direct,
        metrics.tensor(),
        metrics.source(),
        weights,
        &metrics,
        |row, digest| {
            bindings.push(OwnerRowMetricBinding {
                row: row as u64,
                content: OwnerRowMetricId::from_bytes(digest),
            });
            Ok(())
        },
    )
    .map_err(E::Fit)?;
    let observation = recorder
        .emit_owner_score_levels_fitted(
            span,
            parent,
            logical_time,
            &OwnerScoreLevelsFitted {
                plan: plan.clone(),
                source_f32: source_digest(
                    plan.direct.layout.rows,
                    plan.direct.layout.cols,
                    weights,
                ),
                stream: metrics.stream_id(),
                rows_content: owner_metric_rows_id(
                    metrics.stream_id(),
                    plan.direct.layout.cols,
                    &bindings,
                )
                .map_err(E::Evidence)?,
                rows: bindings,
                rayon_threads,
                outputs: fit_outputs(&levels).map_err(E::Semantic)?,
            },
        )
        .map_err(E::Evidence)?;
    Ok(RecordedOwnerScoreFit {
        levels,
        observation,
    })
}
