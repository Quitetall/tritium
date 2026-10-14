//! Typed direct levels over the real numerical fitters, without prefix slicing.

use super::{ActivationAwareConfig, ScaleRefitMode, fit_tensor_with_scale_refit_mode};
use crate::{DensePsdMetric, JointFitConfig, JointFitError, JointFitMetric, RelayBasins};
use half::f16;
use std::collections::BTreeSet;
use tritium_core::{AdditiveError, Trit, apply_basis, apply_basis_f64};
use tritium_format::{AdditiveTensor, AdditiveTensorError};
use tritium_schema::{
    AdditiveLayout, Basis, DirectFitPlan, DirectFitProfile, JointFeedbackFitOptions,
    JointGroupFitOptions, LadderFitOptions, LadderScaleRefit, LayoutError, PlaneAllocation,
    PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision, Transport, admitted_execution_group,
};

/// One directly fitted level; no release verdict or manifest provenance is fabricated.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectTensorFit {
    /// Level name from the request.
    pub level: String,
    /// Shared owned tensor with exactly rounded stored scales and bound basis.
    pub tensor: AdditiveTensor,
}

/// Invalid direct-fit admission or a numerical/output failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DirectFitError {
    /// Invalid semantic layout.
    Layout(LayoutError),
    /// Unsupported candidate profile/law/basis/codec/transport/allocation combination.
    Unsupported(&'static str),
    /// Empty level list, empty/duplicate name or an invalid plane count.
    InvalidLevels,
    /// A numerical configuration value is outside its declared domain.
    InvalidOptions(&'static str),
    /// Dimensions or aggregate element counts overflow.
    ShapeOverflow,
    /// A pre-allocation logical count limit was exceeded.
    LimitExceeded {
        /// Logical resource being bounded, not process RSS.
        resource: &'static str,
        /// Required element count.
        required: u64,
        /// Supplied maximum element count.
        limit: u64,
    },
    /// Original source weights have the wrong row-major length.
    WeightLength {
        /// Required number of source scalars.
        expected: usize,
        /// Actual source scalar count.
        actual: usize,
    },
    /// An original weight is not finite.
    NonFiniteWeight {
        /// Row-major source index.
        index: usize,
    },
    /// A finite source overflowed while being transformed into the fitted basis.
    NonFiniteBasisWeight {
        /// Row-major transformed source index.
        index: usize,
    },
    /// A finite Gram overflowed during its two-axis basis transformation.
    NonFiniteBasisMetric {
        /// Row-major transformed metric index.
        index: usize,
    },
    /// Supplied Gram has the wrong row-major square length.
    GramLength {
        /// Required Gram entry count.
        expected: usize,
        /// Supplied Gram entry count.
        actual: usize,
    },
    /// Gram entry is not finite, or a diagonal is negative.
    InvalidGram {
        /// Row-major metric index.
        index: usize,
    },
    /// Supplied Gram is asymmetric beyond the existing relative metric tolerance.
    AsymmetricGram,
    /// Basis transformation failed.
    Basis(AdditiveError),
    /// Damped ladder factorization/feedback could not produce a fit.
    LadderFactorization,
    /// The existing joint solver rejected the numerical problem.
    Joint(JointFitError),
    /// Full-Gram inverse factorization failed; no identity fallback is attempted.
    JointFactorization,
    /// Validated sequential propagation rejected the numerical problem or update.
    Feedback(crate::FeedbackError),
    /// A finite feedback working value cannot be represented by the f32-input joint fitter.
    FitterNarrowing {
        /// Output row.
        row: usize,
        /// Input column.
        column: usize,
    },
    /// The shared owned tensor rejected the fitted output.
    Output(AdditiveTensorError),
    /// A preflighted owned buffer could not be reserved.
    AllocationFailed,
}

impl core::fmt::Display for DirectFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "direct tensor fit rejected: {self:?}")
    }
}
impl std::error::Error for DirectFitError {}

// Real closed-form and joint adapters share the law-bearing layout interface.
// Keep this seam internal until another production fitter actually needs to plug in.
trait LevelFitter {
    fn fit(
        &self,
        layout: AdditiveLayout,
        source: &[f32],
        gram: Option<&[f64]>,
    ) -> Result<AdditiveTensor, DirectFitError>;
}
struct ClosedFormFitter(LadderFitOptions);
impl LevelFitter for ClosedFormFitter {
    fn fit(
        &self,
        layout: AdditiveLayout,
        source: &[f32],
        gram: Option<&[f64]>,
    ) -> Result<AdditiveTensor, DirectFitError> {
        fit_ladder(
            layout,
            source,
            layout.rows as usize,
            layout.cols as usize,
            gram.expect("ladder metric built"),
            self.0,
        )
    }
}
struct JointFitter {
    options: JointGroupFitOptions,
    feedback: Option<JointFeedbackFitOptions>,
}
impl LevelFitter for JointFitter {
    fn fit(
        &self,
        layout: AdditiveLayout,
        source: &[f32],
        gram: Option<&[f64]>,
    ) -> Result<AdditiveTensor, DirectFitError> {
        if let Some(options) = self.feedback {
            super::joint_feedback::fit(
                layout,
                source,
                gram.expect("feedback metric built"),
                options,
            )
        } else {
            fit_joint(
                layout,
                source,
                layout.rows as usize,
                layout.cols as usize,
                gram,
                self.options,
            )
        }
    }
}

/// Fit every named level from the same original row-major f32 weights.
///
/// `gram` is optional input curvature in the *source* basis. `None` explicitly
/// requests identity curvature. The selected profile retains its own existing
/// arithmetic and does not silently switch precision, fallback or fitter.
/// Hadamard block width is independent of scale group width. Caller inputs
/// remain unchanged, and no fitted level is another level's prefix/source.
///
/// This is the per-tensor implementation used by the model source/sink/evidence
/// engine, not a model controller or allocator itself. It grants no new backend
/// admission and emits no quality, memory, performance or release verdict.
///
/// # Errors
/// Admission rejects unsupported combinations/options/levels, wrong shapes,
/// nonfinite/asymmetric metrics, overflows and logical count limits before
/// numerical allocation. Numerical/output failures never become fallback fits.
pub fn fit_direct_levels(
    plan: &DirectFitPlan,
    weights: &[f32],
    gram: Option<&[f64]>,
) -> Result<Vec<DirectTensorFit>, DirectFitError> {
    let (_, cols, count) = admit(plan, weights, gram)?;
    let mut source = reserved_vec(count)?;
    source.extend_from_slice(weights);
    for row in source.chunks_mut(cols) {
        apply_basis(row, plan.layout.basis).map_err(DirectFitError::Basis)?;
    }
    if let Some(index) = source.iter().position(|value| !value.is_finite()) {
        return Err(DirectFitError::NonFiniteBasisWeight { index });
    }

    let metric = match (gram, plan.profile) {
        (Some(values), _) => {
            let mut owned = reserved_vec(values.len())?;
            owned.extend_from_slice(values);
            Some(owned)
        }
        (
            None,
            DirectFitProfile::ActivationAwareLadder(_) | DirectFitProfile::JointFeedbackGroups(_),
        ) => {
            let mut identity = reserved_vec(cols * cols)?;
            identity.resize(cols * cols, 0.0);
            for col in 0..cols {
                identity[col * cols + col] = 1.0;
            }
            Some(identity)
        }
        _ => None,
    };
    let mut metric = metric;
    if let Some(values) = &mut metric {
        match plan.layout.basis {
            Basis::Hadamard { block } => {
                super::activation_aware::rotate_gram(values, cols, block as usize);
            }
            Basis::SignedRht { .. } => rotate_metric(values, cols, plan.layout.basis)?,
            _ => {}
        }
    }
    if let Some(values) = &metric
        && let Some(index) = values.iter().position(|value| !value.is_finite())
    {
        return Err(DirectFitError::NonFiniteBasisMetric { index });
    }

    let ladder;
    let joint;
    let fitter: &dyn LevelFitter = match plan.profile {
        DirectFitProfile::ActivationAwareLadder(options) => {
            ladder = ClosedFormFitter(options);
            &ladder
        }
        DirectFitProfile::JointIndependentGroups(options) => {
            joint = JointFitter {
                options,
                feedback: None,
            };
            &joint
        }
        DirectFitProfile::JointFeedbackGroups(options) => {
            joint = JointFitter {
                options: options.fitter,
                feedback: Some(options),
            };
            &joint
        }
        _ => return Err(DirectFitError::Unsupported("numerical profile")),
    };
    let mut output = reserved_vec(plan.levels.len())?;
    for level in &plan.levels {
        let layout = AdditiveLayout {
            max_planes: level.planes,
            ..plan.layout
        };
        let tensor = fitter.fit(layout, &source, metric.as_deref())?;
        output.push(DirectTensorFit {
            level: level.name.clone(),
            tensor,
        });
    }
    Ok(output)
}

fn admit(
    plan: &DirectFitPlan,
    weights: &[f32],
    gram: Option<&[f64]>,
) -> Result<(usize, usize, usize), DirectFitError> {
    let (rows, cols, count) = admit_plan(plan, gram.is_some())?;
    let metric_count = cols
        .checked_mul(cols)
        .ok_or(DirectFitError::ShapeOverflow)?;
    admit_payload(weights, gram, cols, count, metric_count)?;
    Ok((rows, cols, count))
}

pub(super) fn admit_plan(
    plan: &DirectFitPlan,
    has_gram: bool,
) -> Result<(usize, usize, usize), DirectFitError> {
    let layout = plan.layout;
    layout.validate().map_err(DirectFitError::Layout)?;
    if !admitted_execution_group(layout.law, layout.group) {
        return Err(DirectFitError::Unsupported("unadmitted execution group"));
    }
    if layout.allocation != PlaneAllocation::Uniform {
        return Err(DirectFitError::Unsupported("per-tile allocation"));
    }
    if !matches!(layout.codec, PlaneCodec::D2 | PlaneCodec::B3) {
        return Err(DirectFitError::Unsupported("codec requires sparse fitting"));
    }
    if layout.transport != Transport::Raw {
        return Err(DirectFitError::Unsupported("entropy transport"));
    }
    if !matches!(
        layout.basis,
        Basis::Identity | Basis::Hadamard { .. } | Basis::SignedRht { .. }
    ) {
        return Err(DirectFitError::Unsupported("basis metric transform"));
    }
    if layout.law.anchor != ScaleAnchor::Group {
        return Err(DirectFitError::Unsupported("scale anchor"));
    }
    match plan.profile {
        DirectFitProfile::ActivationAwareLadder(options) => {
            if layout.law.relation != (PlaneRelation::Tied { num: 1, den: 3 })
                || layout.law.precision != ScalePrecision::F16
            {
                return Err(DirectFitError::Unsupported("ladder scale law"));
            }
            if !options.damp.is_finite() || options.damp < 0.0 {
                return Err(DirectFitError::InvalidOptions("ladder damping"));
            }
            tritium_core::FeedbackDecay::new(options.decay, options.decay_ramp)
                .map_err(|_| DirectFitError::InvalidOptions("feedback decay"))?;
            if !matches!(
                options.scale_refit,
                LadderScaleRefit::None | LadderScaleRefit::PostPass | LadderScaleRefit::InLoop
            ) {
                return Err(DirectFitError::Unsupported("scale refit placement"));
            }
        }
        DirectFitProfile::JointIndependentGroups(options)
        | DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
            fitter: options, ..
        }) => {
            if layout.law.relation != PlaneRelation::Free
                || !matches!(
                    layout.law.precision,
                    ScalePrecision::F16 | ScalePrecision::F32
                )
            {
                return Err(DirectFitError::Unsupported("joint scale law"));
            }
            if let DirectFitProfile::JointFeedbackGroups(feedback) = plan.profile {
                if !feedback.damp.is_finite() || feedback.damp < 0.0 {
                    return Err(DirectFitError::InvalidOptions("joint feedback damping"));
                }
                if !matches!(
                    feedback.precision,
                    tritium_schema::WorkPrecision::F32Accum64 | tritium_schema::WorkPrecision::F64
                ) {
                    return Err(DirectFitError::Unsupported("feedback precision"));
                }
                tritium_core::FeedbackDecay::new(feedback.decay, feedback.decay_ramp)
                    .map_err(|_| DirectFitError::InvalidOptions("feedback decay"))?;
            }
            if options.max_iterations == 0
                || options.em_restarts == 0
                || !options.ridge.is_finite()
                || options.ridge <= 0.0
                || !options.ridge_condition_limit.is_finite()
                || options.ridge_condition_limit <= 1.0
            {
                return Err(DirectFitError::InvalidOptions("joint solve options"));
            }
        }
        _ => return Err(DirectFitError::Unsupported("numerical profile")),
    }
    if plan.levels.is_empty() {
        return Err(DirectFitError::InvalidLevels);
    }
    let mut names = BTreeSet::new();
    let mut total_planes = 0u64;
    for level in &plan.levels {
        if level.name.is_empty()
            || !names.insert(&level.name)
            || level.planes == 0
            || level.planes > layout.max_planes
        {
            return Err(DirectFitError::InvalidLevels);
        }
        total_planes = total_planes
            .checked_add(u64::from(level.planes))
            .ok_or(DirectFitError::ShapeOverflow)?;
    }
    let count = layout
        .rows
        .checked_mul(layout.cols)
        .ok_or(DirectFitError::ShapeOverflow)?;
    let metric_count = layout
        .cols
        .checked_mul(layout.cols)
        .ok_or(DirectFitError::ShapeOverflow)?;
    let output_count = count
        .checked_mul(total_planes)
        .ok_or(DirectFitError::ShapeOverflow)?;
    limit("source weights", count, plan.limits.max_weights)?;
    if has_gram
        || matches!(
            plan.profile,
            DirectFitProfile::ActivationAwareLadder(_) | DirectFitProfile::JointFeedbackGroups(_)
        )
    {
        limit("input Gram", metric_count, plan.limits.max_metric_entries)?;
    }
    limit("output trits", output_count, plan.limits.max_output_trits)?;
    let rows = usize::try_from(layout.rows).map_err(|_| DirectFitError::ShapeOverflow)?;
    let cols = usize::try_from(layout.cols).map_err(|_| DirectFitError::ShapeOverflow)?;
    let count = usize::try_from(count).map_err(|_| DirectFitError::ShapeOverflow)?;
    usize::try_from(metric_count).map_err(|_| DirectFitError::ShapeOverflow)?;
    usize::try_from(output_count).map_err(|_| DirectFitError::ShapeOverflow)?;
    Ok((rows, cols, count))
}

// B G B^T: row-wise forward transforms right-multiply B^T; then columns
// left-multiply B. Core owns signs, global input-axis indices and precision.
fn rotate_metric(values: &mut [f64], cols: usize, basis: Basis) -> Result<(), DirectFitError> {
    for row in values.chunks_exact_mut(cols) {
        apply_basis_f64(row, basis).map_err(DirectFitError::Basis)?;
    }
    let mut column = reserved_vec(cols)?;
    column.resize(cols, 0.0);
    for col in 0..cols {
        for (row, value) in column.iter_mut().enumerate() {
            *value = values[row * cols + col];
        }
        apply_basis_f64(&mut column, basis).map_err(DirectFitError::Basis)?;
        for (row, value) in column.iter().enumerate() {
            values[row * cols + col] = *value;
        }
    }
    Ok(())
}

pub(super) fn admit_payload(
    weights: &[f32],
    gram: Option<&[f64]>,
    cols: usize,
    count: usize,
    metric_count: usize,
) -> Result<(), DirectFitError> {
    if weights.len() != count {
        return Err(DirectFitError::WeightLength {
            expected: count,
            actual: weights.len(),
        });
    }
    if let Some(index) = weights.iter().position(|value| !value.is_finite()) {
        return Err(DirectFitError::NonFiniteWeight { index });
    }
    if let Some(values) = gram {
        if values.len() != metric_count {
            return Err(DirectFitError::GramLength {
                expected: metric_count,
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().enumerate().position(|(index, value)| {
            !value.is_finite() || (index / cols == index % cols && *value < 0.0)
        }) {
            return Err(DirectFitError::InvalidGram { index });
        }
        let scale = values
            .iter()
            .fold(0.0f64, |m, value| m.max(value.abs()))
            .max(f64::MIN_POSITIVE);
        for row in 0..cols {
            for col in row + 1..cols {
                if (values[row * cols + col] / scale - values[col * cols + row] / scale).abs()
                    > 1e-10
                {
                    return Err(DirectFitError::AsymmetricGram);
                }
            }
        }
    }
    Ok(())
}

fn limit(resource: &'static str, required: u64, maximum: u64) -> Result<(), DirectFitError> {
    if required > maximum {
        Err(DirectFitError::LimitExceeded {
            resource,
            required,
            limit: maximum,
        })
    } else {
        Ok(())
    }
}

pub(super) fn reserved_vec<T>(count: usize) -> Result<Vec<T>, DirectFitError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| DirectFitError::AllocationFailed)?;
    Ok(values)
}

fn fit_ladder(
    layout: AdditiveLayout,
    source: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f64],
    options: LadderFitOptions,
) -> Result<AdditiveTensor, DirectFitError> {
    let cfg = ActivationAwareConfig {
        planes: usize::from(layout.max_planes),
        group: usize::from(layout.group),
        grid: options.grid as usize,
        damp: options.damp,
        search_sweeps: options.search_sweeps as usize,
        refit_scale: options.scale_refit != LadderScaleRefit::None,
        rotate: false,
        decay: options.decay,
        decay_ramp: options.decay_ramp,
    };
    let placement = if options.scale_refit == LadderScaleRefit::InLoop {
        ScaleRefitMode::InLoop
    } else {
        ScaleRefitMode::PostPass
    };
    let fits = fit_tensor_with_scale_refit_mode(source, rows, cols, gram, &cfg, placement)
        .ok_or(DirectFitError::LadderFactorization)?;
    let per_row = cols.div_ceil(cfg.group);
    let mut trits = reserved_vec(rows * cols * cfg.planes)?;
    trits.resize(rows * cols * cfg.planes, Trit::ZERO);
    let mut scales = reserved_vec(fits.len())?;
    for (index, (anchor, planes)) in fits.into_iter().enumerate() {
        scales.push(f16::from_f32(anchor).to_f32());
        let row = index / per_row;
        let col = index % per_row * cfg.group;
        for (plane, digits) in planes.into_iter().enumerate() {
            let start = plane * rows * cols + row * cols + col;
            for (offset, digit) in digits.into_iter().enumerate() {
                trits[start + offset] = Trit::from_i8(digit).expect("ladder digits are ternary");
            }
        }
    }
    AdditiveTensor::new(layout, layout.max_planes, trits, scales).map_err(DirectFitError::Output)
}

fn fit_joint(
    layout: AdditiveLayout,
    source: &[f32],
    rows: usize,
    cols: usize,
    gram: Option<&[f64]>,
    options: JointGroupFitOptions,
) -> Result<AdditiveTensor, DirectFitError> {
    let group = usize::from(layout.group);
    let per_row = cols.div_ceil(group);
    let mut output = JointOutput::new(layout, options)?;
    for block in 0..per_row {
        let start = block * group;
        let end = (start + group).min(cols);
        let width = end - start;
        let metric = if let Some(values) = gram {
            let mut local = reserved_vec(width * width)?;
            for row in start..end {
                local.extend_from_slice(&values[row * cols + start..row * cols + end]);
            }
            Some(DensePsdMetric::new(width, &local).map_err(DirectFitError::Joint)?)
        } else {
            None
        };
        for row in 0..rows {
            let metric = metric
                .as_ref()
                .map_or(JointFitMetric::Identity, JointFitMetric::Dense);
            output.fit_row(
                row,
                block,
                &source[row * cols + start..row * cols + end],
                metric,
            )?;
        }
    }
    output.finish()
}

// One solver invocation/scale rounding/output installation for independent and feedback fits.
pub(super) struct JointOutput {
    layout: AdditiveLayout,
    config: JointFitConfig,
    trits: Vec<Trit>,
    scales: Vec<f32>,
}
impl JointOutput {
    pub(super) fn new(
        layout: AdditiveLayout,
        options: JointGroupFitOptions,
    ) -> Result<Self, DirectFitError> {
        let rows = layout.rows as usize;
        let cols = layout.cols as usize;
        let planes = usize::from(layout.max_planes);
        let units = rows * cols.div_ceil(usize::from(layout.group));
        let mut trits = reserved_vec(rows * cols * planes)?;
        trits.resize(rows * cols * planes, Trit::ZERO);
        let mut scales = reserved_vec(units * planes)?;
        scales.resize(units * planes, 0.0);
        Ok(Self {
            layout,
            config: joint_config(layout, options),
            trits,
            scales,
        })
    }
    pub(super) fn fit_row(
        &mut self,
        row: usize,
        block: usize,
        input: &[f32],
        metric: JointFitMetric<'_>,
    ) -> Result<Vec<f32>, DirectFitError> {
        let fit =
            crate::fit_joint_ternary(input, metric, self.config).map_err(DirectFitError::Joint)?;
        let rows = self.layout.rows as usize;
        let cols = self.layout.cols as usize;
        let group = usize::from(self.layout.group);
        let per_row = cols.div_ceil(group);
        let units = rows * per_row;
        for plane in 0..self.config.planes {
            self.scales[plane * units + row * per_row + block] = fit.scales[plane];
            let offset = plane * rows * cols + row * cols + block * group;
            for (index, &digit) in fit.trits[plane].iter().enumerate() {
                self.trits[offset + index] =
                    Trit::from_i8(digit).expect("joint digits are ternary");
            }
        }
        Ok(fit.reconstruction)
    }
    pub(super) fn finish(self) -> Result<AdditiveTensor, DirectFitError> {
        AdditiveTensor::new(self.layout, self.layout.max_planes, self.trits, self.scales)
            .map_err(DirectFitError::Output)
    }
}

fn joint_config(layout: AdditiveLayout, options: JointGroupFitOptions) -> JointFitConfig {
    JointFitConfig {
        planes: usize::from(layout.max_planes),
        max_iterations: options.max_iterations as usize,
        ridge: options.ridge,
        em_restarts: options.em_restarts as usize,
        ridge_condition_limit: options.ridge_condition_limit,
        scale_precision: match layout.law.precision {
            ScalePrecision::F16 => crate::ScalePrecision::F16,
            _ => crate::ScalePrecision::F32,
        },
        relay_basins: RelayBasins {
            softened: options.softened_relay,
            modulated: options.modulated_relay,
        },
    }
}
