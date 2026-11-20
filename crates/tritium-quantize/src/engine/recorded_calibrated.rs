//! Actual-cache execution evidence around the shared calibrated numerical consumer.

use super::direct::{admit_payload, admit_plan};
use super::recorded::{fit_outputs, source_digest};
use super::{
    CalibratedFitError, CalibrationCache, CalibrationCacheError, DirectFitError, DirectTensorFit,
    fit_calibrated_levels,
};
use tritium_evidence::{CalibratedFitReceiptView, EvidenceError, EvidenceRecorder};
use tritium_format::trit_model::TritModelError;
use tritium_schema::{CalibratedFitPlan, CalibratedLevelsFitted, CalibrationPlanError};

/// Actual fitted outputs and immutable initial evidence; not model Direct provenance.
#[derive(Debug)]
pub struct RecordedCalibratedFit {
    /// Owned outputs; later mutation cannot change the logged observation.
    pub levels: Vec<DirectTensorFit>,
    /// Log-derived typed view, not a completed model or release receipt.
    pub observation: CalibratedFitReceiptView,
}
/// No success event or fitted return on binding, reduction, fitting or recording failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum RecordedCalibratedFitError {
    /// Invalid expected cache/reduction context or geometry.
    Plan(CalibrationPlanError),
    /// Actual immutable cache binding differs from the expected complete binding.
    CacheBinding,
    /// Existing aggregate direct controller admission failed before reduction.
    Admission(DirectFitError),
    /// Actual common cache replay/reduction failed.
    Cache(CalibrationCacheError),
    /// Existing calibrated direct controller rejected numerical fitting.
    Fit(CalibratedFitError),
    /// Existing semantic output identity could not be calculated.
    Semantic(TritModelError),
    /// Recorder context or typed emission failed.
    Evidence(EvidenceError),
    /// Actual Rayon pool size cannot be represented in the observation.
    ThreadCountOverflow,
}
impl core::fmt::Display for RecordedCalibratedFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "recorded calibrated fit rejected: {self:?}")
    }
}
impl std::error::Error for RecordedCalibratedFitError {}

/// Reduce an actual immutable cache, fit original rows, and bind actual execution.
///
/// Complete expected/actual cache binding, geometry, original payload, aggregate
/// direct-fit admission and recorder context are checked before numerical work.
/// Calls the existing cache collector and calibrated controller; source bit and
/// output semantic digests use existing shared implementations. No reducer, hash,
/// fitter or forward is copied. Input-only remains input-only; no fallback.
///
/// Cache byte ownership is not proof of declared recipe execution or raw model
/// equality. Source-f32 identity is not BF16 file identity. Whole-model source,
/// lifecycle and sealed package admission still require engine integration.
/// Logical limits exclude complete fitting/reduction metadata/copies/process RSS.
/// # Errors
/// Typed context/admission, actual reducer/controller, output hash or recorder failure.
pub fn fit_calibrated_levels_recorded(
    plan: &CalibratedFitPlan,
    weights: &[f32],
    cache: &CalibrationCache,
    recorder: &mut EvidenceRecorder,
    span: &str,
    parent: Option<&str>,
    logical_time: u64,
) -> Result<RecordedCalibratedFit, RecordedCalibratedFitError> {
    plan.calibration_plan()
        .map_err(RecordedCalibratedFitError::Plan)?;
    if cache.binding() != &plan.cache {
        return Err(RecordedCalibratedFitError::CacheBinding);
    }
    recorder
        .check_emit::<CalibratedLevelsFitted>(span, parent, logical_time)
        .map_err(RecordedCalibratedFitError::Evidence)?;
    let (_, cols, count) =
        admit_plan(&plan.direct, true).map_err(RecordedCalibratedFitError::Admission)?;
    let entries = cols
        .checked_mul(cols)
        .ok_or(RecordedCalibratedFitError::Admission(
            DirectFitError::ShapeOverflow,
        ))?;
    admit_payload(weights, None, cols, count, entries)
        .map_err(RecordedCalibratedFitError::Admission)?;
    let rayon_threads = u32::try_from(rayon::current_num_threads())
        .map_err(|_| RecordedCalibratedFitError::ThreadCountOverflow)?;
    let calibrated = cache
        .collect(plan.partition, plan.damping, plan.max_numeric_bytes)
        .map_err(RecordedCalibratedFitError::Cache)?;
    let levels = fit_calibrated_levels(
        &plan.direct,
        &plan.cache.request.tensor,
        plan.cache.request.model,
        weights,
        &calibrated,
    )
    .map_err(RecordedCalibratedFitError::Fit)?;
    let observation = recorder
        .emit_calibrated_levels_fitted(
            span,
            parent,
            logical_time,
            &CalibratedLevelsFitted {
                plan: plan.clone(),
                source_f32: source_digest(
                    plan.direct.layout.rows,
                    plan.direct.layout.cols,
                    weights,
                ),
                factors: calibrated.factors().clone(),
                rayon_threads,
                outputs: fit_outputs(&levels).map_err(RecordedCalibratedFitError::Semantic)?,
            },
        )
        .map_err(RecordedCalibratedFitError::Evidence)?;
    Ok(RecordedCalibratedFit {
        levels,
        observation,
    })
}
