//! Candidate actual-cache fitting observations (ADR 0061), not release verdicts.

use crate::{
    CalibrationCacheBinding, CalibrationFactors, CalibrationPartition, CalibrationPlan,
    CalibrationPlanError, DirectFitOutput, DirectFitPlan, FitInputDigest,
};
use alloc::vec::Vec;

/// Expected immutable cache context and numerical request for a calibrated fit.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedFitPlan {
    /// Existing original-row controller profile, outputs and logical limits.
    pub direct: DirectFitPlan,
    /// Expected complete actual cache binding; model truth requires engine admission.
    pub cache: CalibrationCacheBinding,
    /// Explicit input statistics partition, independent of weight scale groups.
    pub partition: CalibrationPartition,
    /// Existing absolute row-metric damping after Fisher scaling.
    pub damping: f64,
    /// Logical reducer budget, not caller buffers, fitter workspace or total RSS.
    pub max_numeric_bytes: u64,
}

impl CalibratedFitPlan {
    /// Validate context and return the same reduction request used by cache replay.
    /// Does not verify cache bytes, original model truth or controller numerics.
    /// # Errors
    /// Invalid cache/source context, geometry, partition, damping or logical budget.
    pub fn calibration_plan(&self) -> Result<CalibrationPlan, CalibrationPlanError> {
        let cache = &self.cache;
        cache.request.validate()?;
        if cache.request.tensor.shape.as_slice()
            != [self.direct.layout.rows, self.direct.layout.cols]
            || self.direct.layout.validate().is_err()
            || cache.source.model.0 != *cache.request.model.as_bytes()
            || cache.samples == 0
            || cache.windows == 0
            || cache.windows > cache.samples
            || cache
                .request
                .sample_start
                .checked_add(cache.samples)
                .is_none()
        {
            return Err(CalibrationPlanError::Source);
        }
        let limits = cache.request.limits;
        if cache.payload_bytes == 0
            || cache.payload_bytes > limits.max_payload_bytes
            || cache.samples > limits.max_samples
            || cache.windows > limits.max_windows
        {
            return Err(CalibrationPlanError::Budget);
        }
        let plan = cache.reduction_plan(self.partition, self.damping, self.max_numeric_bytes);
        plan.validate()?;
        Ok(plan)
    }
}

/// Actual per-tensor calibrated execution evidence, not model Direct provenance.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedLevelsFitted {
    /// Executed numerical request and actual immutable cache context.
    pub plan: CalibratedFitPlan,
    /// Existing bit digest of actual original f32 inputs, not original BF16 files.
    pub source_f32: FitInputDigest,
    /// Actual shared collector factor identities, dimensions and selection.
    pub factors: CalibrationFactors,
    /// Actual Rayon pool size.
    pub rayon_threads: u32,
    /// Actual independent outputs in requested order, never plane prefixes.
    pub outputs: Vec<DirectFitOutput>,
}
