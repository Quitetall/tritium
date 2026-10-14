//! Actual calibration factors consumed by the existing row-local direct controller.
//! Runtime context checks are not audited raw-source or model/package provenance.

use super::direct::{admit_payload, admit_plan, reserved_vec};
use super::{
    CalibratedTensor, CalibrationError, DirectFitError, DirectTensorFit, fit_direct_levels,
};
use tritium_core::Trit;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    AdditiveLayout, DirectFitPlan, OriginalSourceId, PlaneRelation, QuantSourceTensor,
};

/// No fallback on expected context, actual factor, direct-fit or output failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum CalibratedFitError {
    /// Original tensor/model context or requested fit geometry differs from factors.
    Binding,
    /// The existing direct controller rejected admission, numerics or output.
    Fit(DirectFitError),
    /// The existing source/selection-bound K-FAC builder rejected a row metric.
    Calibration(CalibrationError),
}
impl core::fmt::Display for CalibratedFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "calibrated direct fit rejected: {self:?}")
    }
}
impl std::error::Error for CalibratedFitError {}
impl From<DirectFitError> for CalibratedFitError {
    fn from(error: DirectFitError) -> Self {
        Self::Fit(error)
    }
}
impl From<CalibrationError> for CalibratedFitError {
    fn from(error: CalibrationError) -> Self {
        Self::Calibration(error)
    }
}

/// Fit every named level from actual original rows with their actual calibration factors.
///
/// Checks expected tensor/model context and aggregate direct-fit admission first.
/// Full inputs retain cross-group covariance; explicitly grouped statistics form
/// a block-diagonal approximation. Fisher uses the existing K-FAC builder for
/// each output row and group, including its damping and zero-Fisher behavior.
/// Input-only factors are never described as Fisher. Basis rotation, numerical
/// fitting and stored scale rounding stay in the unchanged direct controller.
///
/// Only one assembled row metric is live; scalar source, calibration factors,
/// outputs, row controller copies and solver state still consume memory. Existing
/// logical element limits are not aggregate RSS or GPU/out-of-core admission.
/// Expected context equality does not independently prove that caller f32 values
/// are the raw model identified by `model`. The complete engine must enforce this
/// and record calibration/log/package binding before assigning audited provenance.
/// No existing v1 fit observation or release verdict is created or relabeled.
/// # Errors
/// Context/geometry mismatch, direct-fit limits/numerics/output or actual K-FAC failure.
pub fn fit_calibrated_levels(
    plan: &DirectFitPlan,
    original: &QuantSourceTensor,
    model: OriginalSourceId,
    weights: &[f32],
    calibration: &CalibratedTensor,
) -> Result<Vec<DirectTensorFit>, CalibratedFitError> {
    let binding = &calibration.factors().plan;
    if binding.tensor != *original
        || binding.source.model.0 != *model.as_bytes()
        || original.shape.as_slice() != [plan.layout.rows, plan.layout.cols]
    {
        return Err(CalibratedFitError::Binding);
    }
    fit_row_metrics(plan, weights, |row, cols, metric| {
        if row == 0 || calibration.output().is_some() {
            metric.fill(0.0);
            assemble_metric(calibration, row, cols, metric)?;
        }
        Ok(())
    })
}

// Common row controller: admit the full tensor before any row work and install
// each unchanged direct fitter output. Consumers provide source-basis metrics,
// not another solver or basis transform. Errors never publish partial levels.
pub(super) fn fit_row_metrics<E: From<DirectFitError>>(
    plan: &DirectFitPlan,
    weights: &[f32],
    mut assemble: impl FnMut(usize, usize, &mut [f64]) -> Result<(), E>,
) -> Result<Vec<DirectTensorFit>, E> {
    let (rows, cols, count) = admit_plan(plan, true)?;
    let metric_count = cols
        .checked_mul(cols)
        .ok_or(DirectFitError::ShapeOverflow)?;
    admit_payload(weights, None, cols, count, metric_count)?;
    let mut metric = reserved_vec(metric_count)?;
    metric.resize(metric_count, 0.0);
    let mut pending = reserved_vec(plan.levels.len())?;
    for level in &plan.levels {
        pending.push(PendingLevel::new(
            AdditiveLayout {
                max_planes: level.planes,
                ..plan.layout
            },
            count,
        )?);
    }
    let mut row_plan = plan.clone();
    row_plan.layout.rows = 1;
    for row in 0..rows {
        assemble(row, cols, &mut metric)?;
        let fitted = fit_direct_levels(
            &row_plan,
            &weights[row * cols..(row + 1) * cols],
            Some(&metric),
        )?;
        for (target, fitted) in pending.iter_mut().zip(fitted) {
            target.install(row, cols, count, &fitted.tensor);
        }
    }
    let mut output = reserved_vec(pending.len())?;
    for (level, pending) in plan.levels.iter().zip(pending) {
        output.push(DirectTensorFit {
            level: level.name.clone(),
            tensor: AdditiveTensor::new(
                pending.layout,
                level.planes,
                pending.trits,
                pending.scales,
            )
            .map_err(DirectFitError::Output)?,
        });
    }
    Ok(output)
}

fn assemble_metric(
    calibration: &CalibratedTensor,
    row: usize,
    cols: usize,
    destination: &mut [f64],
) -> Result<(), CalibratedFitError> {
    let mut start = 0;
    for (group, input) in calibration.inputs().iter().enumerate() {
        let width = input.dimension();
        let kfac;
        let values = if calibration.output().is_some() {
            kfac = calibration.kfac_metric(group, row)?;
            kfac.metric().as_slice()
        } else {
            input.as_slice()
        };
        let end = start + width;
        if end > cols {
            return Err(CalibratedFitError::Binding);
        }
        for local in 0..width {
            destination[(start + local) * cols + start..(start + local) * cols + end]
                .copy_from_slice(&values[local * width..(local + 1) * width]);
        }
        start = end;
    }
    if start != cols {
        return Err(CalibratedFitError::Binding);
    }
    Ok(())
}

struct PendingLevel {
    layout: AdditiveLayout,
    trits: Vec<Trit>,
    scales: Vec<f32>,
    scale_planes: usize,
}
impl PendingLevel {
    fn new(layout: AdditiveLayout, count: usize) -> Result<Self, DirectFitError> {
        let planes = usize::from(layout.max_planes);
        let scale_planes = if layout.law.relation == PlaneRelation::Free {
            planes
        } else {
            1
        };
        let units =
            (layout.rows as usize) * (layout.cols as usize).div_ceil(usize::from(layout.group));
        let mut trits = reserved_vec(count * planes)?;
        trits.resize(count * planes, Trit::ZERO);
        let mut scales = reserved_vec(units * scale_planes)?;
        scales.resize(units * scale_planes, 0.0);
        Ok(Self {
            layout,
            trits,
            scales,
            scale_planes,
        })
    }
    fn install(&mut self, row: usize, cols: usize, count: usize, tensor: &AdditiveTensor) {
        for (plane, digits) in tensor.trits().chunks_exact(cols).enumerate() {
            let start = plane * count + row * cols;
            self.trits[start..start + cols].copy_from_slice(digits);
        }
        let per_row = cols.div_ceil(usize::from(self.layout.group));
        let units = self.layout.rows as usize * per_row;
        for plane in 0..self.scale_planes {
            let start = plane * units + row * per_row;
            self.scales[start..start + per_row]
                .copy_from_slice(&tensor.scales()[plane * per_row..(plane + 1) * per_row]);
        }
    }
}
