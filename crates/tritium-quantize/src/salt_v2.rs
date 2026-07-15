//! Joint additive-ternary fitting for SALT V2.

use half::f16;

/// Precision used when scoring fitted scales.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalePrecision {
    /// Score the fitted `f32` scales directly.
    #[default]
    F32,
    /// Round every fitted scale through the deployment `f16` representation before scoring.
    F16,
}

/// Configuration for [`fit_joint_ternary`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointFitConfig {
    /// Number of additive ternary planes. Must be in `1..=3`.
    pub planes: usize,
    /// Maximum number of alternating scale/assignment updates.
    pub max_iterations: usize,
    /// Positive diagonal ridge added to the weighted scale normal equations.
    pub ridge: f64,
    /// Number of deterministic output-aware initialization basins to evaluate.
    pub em_restarts: usize,
    /// Maximum accepted condition number for the ridge-regularized scale system.
    pub ridge_condition_limit: f64,
    /// Precision at which scales are scored and returned.
    pub scale_precision: ScalePrecision,
}

impl Default for JointFitConfig {
    fn default() -> Self {
        Self {
            planes: 1,
            max_iterations: 16,
            ridge: 1e-8,
            em_restarts: 4,
            ridge_condition_limit: 1e6,
            scale_precision: ScalePrecision::F32,
        }
    }
}

/// Owned dense symmetric positive-semidefinite curvature for one weight group.
///
/// Values are row-major `f64`. Construction validates finiteness, symmetry, and a numerically
/// tolerant Cholesky factorization, so fit-time code may rely on the PSD contract.
#[derive(Clone, Debug, PartialEq)]
pub struct DensePsdMetric {
    dimension: usize,
    values: Vec<f64>,
}

impl DensePsdMetric {
    /// Validate and copy a row-major dense PSD matrix.
    ///
    /// # Errors
    /// Rejects zero dimension, wrong storage length, non-finite/asymmetric entries, and matrices
    /// with a negative pivot beyond the numerical PSD tolerance.
    pub fn new(dimension: usize, values: &[f64]) -> Result<Self, JointFitError> {
        if dimension == 0 {
            return Err(JointFitError::InvalidDenseMetricDimension);
        }
        let expected = dimension.saturating_mul(dimension);
        if values.len() != expected {
            return Err(JointFitError::DenseMetricLengthMismatch {
                expected,
                got: values.len(),
            });
        }
        for row in 0..dimension {
            for col in 0..dimension {
                if !values[row * dimension + col].is_finite() {
                    return Err(JointFitError::NonFiniteDenseMetric { row, col });
                }
            }
        }
        let matrix_scale = values
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        if matrix_scale == 0.0 {
            return Err(JointFitError::ZeroMetric);
        }

        let mut canonical_values = values.to_vec();
        for row in 0..dimension {
            for col in row + 1..dimension {
                let upper = values[row * dimension + col];
                let lower = values[col * dimension + row];
                if (upper / matrix_scale - lower / matrix_scale).abs() > 1e-10 {
                    return Err(JointFitError::AsymmetricDenseMetric { row, col });
                }
                // Accepted near-symmetry must not leak into dense coordinate deltas, which assume
                // one exact symmetric quadratic. Averaging half-products avoids overflow for two
                // same-sign finite values near f64::MAX.
                let symmetric = upper * 0.5 + lower * 0.5;
                canonical_values[row * dimension + col] = symmetric;
                canonical_values[col * dimension + row] = symmetric;
            }
            // Every PSD diagonal is non-negative. Reject an explicitly negative stored diagonal
            // at any scale; Schur-complement roundoff is handled separately below.
            if canonical_values[row * dimension + row] < 0.0 {
                return Err(JointFitError::NonPositiveSemidefiniteMetric { pivot: row });
            }
        }

        // Semidefinite-aware Cholesky. For a PSD Schur complement, a zero pivot implies the
        // remainder of that column is zero; a material residual there is therefore also non-PSD.
        // Normalize by the matrix's own scale so a tiny-but-material negative matrix is not hidden
        // by an absolute tolerance derived from 1.0.
        let normalized: Vec<f64> = canonical_values
            .iter()
            .map(|value| value / matrix_scale)
            .collect();
        let mut lower = vec![0.0_f64; expected];
        let tolerance_factor = f64::EPSILON * (dimension as f64).max(1.0) * 64.0;
        for pivot in 0..dimension {
            let prior_sq: f64 = (0..pivot)
                .map(|col| {
                    let value = lower[pivot * dimension + col];
                    value * value
                })
                .sum();
            let diagonal = normalized[pivot * dimension + pivot] - prior_sq;
            let pivot_scale = normalized[pivot * dimension + pivot]
                .abs()
                .max(prior_sq.abs())
                .max(1.0);
            let pivot_tolerance = pivot_scale * tolerance_factor;
            if diagonal < -pivot_tolerance {
                return Err(JointFitError::NonPositiveSemidefiniteMetric { pivot });
            }
            if diagonal <= pivot_tolerance {
                for row in pivot + 1..dimension {
                    let prior: f64 = (0..pivot)
                        .map(|col| lower[row * dimension + col] * lower[pivot * dimension + col])
                        .sum();
                    let entry = normalized[row * dimension + pivot];
                    let residual_tolerance =
                        entry.abs().max(prior.abs()).max(1.0) * tolerance_factor;
                    if (entry - prior).abs() > residual_tolerance {
                        return Err(JointFitError::NonPositiveSemidefiniteMetric { pivot });
                    }
                }
                continue;
            }
            let root = diagonal.sqrt();
            lower[pivot * dimension + pivot] = root;
            for row in pivot + 1..dimension {
                let prior: f64 = (0..pivot)
                    .map(|col| lower[row * dimension + col] * lower[pivot * dimension + col])
                    .sum();
                lower[row * dimension + pivot] =
                    (normalized[row * dimension + pivot] - prior) / root;
            }
        }

        Ok(Self {
            dimension,
            values: canonical_values,
        })
    }

    /// Build `output_weight * input_gram`, the per-output-row K-FAC curvature block.
    ///
    /// `output_weight` is the positive scalar output-gradient curvature for this row/group.
    ///
    /// # Errors
    /// Rejects a non-positive/non-finite output weight and propagates [`Self::new`] validation.
    pub fn from_kfac_input_gram(
        dimension: usize,
        input_gram: &[f64],
        output_weight: f64,
    ) -> Result<Self, JointFitError> {
        if !output_weight.is_finite() || output_weight <= 0.0 {
            return Err(JointFitError::InvalidKfacOutputWeight);
        }
        let scaled: Vec<f64> = input_gram
            .iter()
            .map(|value| value * output_weight)
            .collect();
        Self::new(dimension, &scaled)
    }

    /// Matrix dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Row-major matrix values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
}

/// Reconstruction curvature used by [`fit_joint_ternary`].
#[derive(Clone, Copy, Debug, Default)]
pub enum JointFitMetric<'a> {
    /// Identity curvature, equivalent to ordinary squared reconstruction error.
    #[default]
    Identity,
    /// Non-negative diagonal curvature, one entry per weight.
    Diagonal(&'a [f32]),
    /// Dense symmetric PSD group curvature, scored as `error^T H error`.
    Dense(&'a DensePsdMetric),
}

/// Alternating-optimization phase that produced an accepted objective reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointFitUpdatePhase {
    /// The joint scale M step.
    Scale,
    /// The exact-state coordinate E step.
    Assignment,
}

/// Evidence for one accepted E or M update.
#[derive(Clone, Debug, PartialEq)]
pub struct JointFitUpdateReceipt {
    /// Zero-based alternating-optimization iteration.
    pub iteration: usize,
    /// Update phase.
    pub phase: JointFitUpdatePhase,
    /// Objective before the phase.
    pub objective_before: f64,
    /// Strictly lower objective after the phase.
    pub objective_after: f64,
}

/// Numerical evidence from one conditioned joint scale solve.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleSolveTelemetry {
    /// Spectral condition number before ridge regularization.
    pub condition_before: f64,
    /// Spectral condition number after ridge regularization.
    pub condition_after: f64,
    /// Diagonal ridge actually used.
    pub ridge_used: f64,
    /// Whether the condition limit increased the configured minimum ridge.
    pub adaptive_ridge: bool,
}

/// Evidence for one attempted M step.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleSolveReceipt {
    /// Zero-based alternating-optimization iteration.
    pub iteration: usize,
    /// Numerical solve telemetry.
    pub telemetry: ScaleSolveTelemetry,
    /// Whether the unregularized reconstruction objective strictly improved and was accepted.
    pub accepted: bool,
}

/// Source of a deterministic initialization basin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointFitStartKind {
    /// Output-aware EM basin with the given deterministic restart index.
    DeterministicRestart(usize),
    /// Embedded best solution with one fewer active plane and a zero final plane.
    LowerPlaneFallback,
}

/// Complete optimization evidence for one initialization basin.
#[derive(Clone, Debug, PartialEq)]
pub struct JointFitRestartReceipt {
    /// Initialization source.
    pub kind: JointFitStartKind,
    /// Objective before alternating updates.
    pub initial_objective: f64,
    /// Objective after the final accepted update.
    pub final_objective: f64,
    /// Every accepted E and M phase, in execution order.
    pub accepted_updates: Vec<JointFitUpdateReceipt>,
    /// Every attempted conditioned scale solve.
    pub scale_solves: Vec<ScaleSolveReceipt>,
}

/// Result of a joint additive-ternary fit.
#[derive(Clone, Debug, PartialEq)]
pub struct JointTernaryFit {
    /// Non-negative scale for each plane.
    pub scales: Vec<f32>,
    /// Plane-major trits. Every value is one of `-1`, `0`, or `+1`.
    pub trits: Vec<Vec<i8>>,
    /// Dense reconstruction `sum_p scales[p] * trits[p]`.
    pub reconstruction: Vec<f32>,
    /// Final reconstruction error under the selected curvature metric.
    pub objective: f64,
    /// Selected start's initial objective followed by every strictly improved accepted E/M phase.
    pub accepted_objectives: Vec<f64>,
    /// Optimization evidence for every evaluated initialization basin.
    pub restart_receipts: Vec<JointFitRestartReceipt>,
    /// Index into [`Self::restart_receipts`] for the returned basin.
    pub selected_start: usize,
}

/// Why a joint additive-ternary fit could not be produced.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum JointFitError {
    /// The weight group was empty.
    EmptyWeights,
    /// Plane count was outside the supported `1..=3` range.
    InvalidPlaneCount {
        /// Rejected plane count.
        got: usize,
    },
    /// `max_iterations` was zero.
    InvalidMaxIterations,
    /// No deterministic EM initialization was requested.
    InvalidRestartCount,
    /// The ridge coefficient was not finite and strictly positive.
    InvalidRidge,
    /// The requested regularized condition limit was not finite and greater than one.
    InvalidConditionLimit,
    /// An input weight was NaN or infinite.
    NonFiniteWeight {
        /// Index of the rejected weight.
        index: usize,
    },
    /// The diagonal metric did not have one entry per weight.
    MetricLengthMismatch {
        /// Required metric length.
        expected: usize,
        /// Supplied metric length.
        got: usize,
    },
    /// A diagonal metric entry was negative, NaN, or infinite.
    InvalidMetric {
        /// Index of the rejected metric entry.
        index: usize,
    },
    /// Every diagonal metric entry was zero, leaving no scored objective.
    ZeroMetric,
    /// A fixed-scale assignment requested a scale count outside `1..=3`.
    InvalidScaleCount {
        /// Rejected scale count.
        got: usize,
    },
    /// A fixed assignment scale was negative, NaN, or infinite.
    InvalidScale {
        /// Index of the rejected scale.
        index: usize,
    },
    /// A dense metric used a zero matrix dimension.
    InvalidDenseMetricDimension,
    /// Dense row-major storage did not contain `dimension²` entries.
    DenseMetricLengthMismatch {
        /// Required entry count.
        expected: usize,
        /// Supplied entry count.
        got: usize,
    },
    /// A dense metric entry was NaN or infinite.
    NonFiniteDenseMetric {
        /// Matrix row.
        row: usize,
        /// Matrix column.
        col: usize,
    },
    /// A dense metric was not symmetric within the validation tolerance.
    AsymmetricDenseMetric {
        /// First mismatched row.
        row: usize,
        /// First mismatched column.
        col: usize,
    },
    /// A dense symmetric metric had a materially negative semidefinite factorization pivot.
    NonPositiveSemidefiniteMetric {
        /// First rejected pivot.
        pivot: usize,
    },
    /// K-FAC output-gradient curvature was not finite and strictly positive.
    InvalidKfacOutputWeight,
    /// Dense metric dimension did not match the fitted weight group.
    DenseMetricDimensionMismatch {
        /// Number of fitted weights.
        expected: usize,
        /// Dense matrix dimension.
        got: usize,
    },
    /// The ridge-regularized scale system could not be solved to finite scales.
    ScaleSolveFailed,
    /// Reconstruction objective arithmetic overflowed or otherwise became non-finite.
    NonFiniteObjective,
    /// A finite fitted scale overflowed the selected deployment representation.
    ScaleNotRepresentable {
        /// Plane containing the rejected scale.
        plane: usize,
    },
}

impl core::fmt::Display for JointFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyWeights => write!(f, "joint ternary fit requires at least one weight"),
            Self::InvalidPlaneCount { got } => {
                write!(f, "joint ternary plane count must be in 1..=3, got {got}")
            }
            Self::InvalidMaxIterations => write!(f, "max_iterations must be greater than zero"),
            Self::InvalidRestartCount => write!(f, "em_restarts must be greater than zero"),
            Self::InvalidRidge => write!(f, "ridge must be finite and greater than zero"),
            Self::InvalidConditionLimit => {
                write!(
                    f,
                    "ridge_condition_limit must be finite and greater than one"
                )
            }
            Self::NonFiniteWeight { index } => write!(f, "weight at index {index} is not finite"),
            Self::MetricLengthMismatch { expected, got } => {
                write!(f, "metric length mismatch: expected {expected}, got {got}")
            }
            Self::InvalidMetric { index } => {
                write!(f, "diagonal metric at index {index} is invalid")
            }
            Self::ZeroMetric => write!(f, "metric must contain a positive scored direction"),
            Self::InvalidScaleCount { got } => {
                write!(f, "exact assignment requires 1..=3 scales, got {got}")
            }
            Self::InvalidScale { index } => {
                write!(f, "assignment scale at index {index} is invalid")
            }
            Self::InvalidDenseMetricDimension => {
                write!(f, "dense metric dimension must be greater than zero")
            }
            Self::DenseMetricLengthMismatch { expected, got } => {
                write!(
                    f,
                    "dense metric length mismatch: expected {expected}, got {got}"
                )
            }
            Self::NonFiniteDenseMetric { row, col } => {
                write!(f, "dense metric entry ({row}, {col}) is not finite")
            }
            Self::AsymmetricDenseMetric { row, col } => {
                write!(f, "dense metric is asymmetric at ({row}, {col})")
            }
            Self::NonPositiveSemidefiniteMetric { pivot } => {
                write!(
                    f,
                    "dense metric is not positive semidefinite at pivot {pivot}"
                )
            }
            Self::InvalidKfacOutputWeight => {
                write!(f, "K-FAC output curvature must be finite and positive")
            }
            Self::DenseMetricDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dense metric dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::ScaleSolveFailed => write!(f, "ridge scale solve failed"),
            Self::NonFiniteObjective => {
                write!(f, "reconstruction objective is not finite")
            }
            Self::ScaleNotRepresentable { plane } => {
                write!(f, "scale for plane {plane} is not deployment-representable")
            }
        }
    }
}

impl std::error::Error for JointFitError {}

/// Find the exact ternary codes for fixed non-negative scales.
///
/// Every weight independently enumerates all `3^P` additive states, for `P` in `1..=3`, and
/// selects the state with minimum squared reconstruction error. Ties prefer zero, then `-1`, then
/// `+1` in plane order, providing deterministic sparse canonicalization.
///
/// # Errors
/// Rejects empty/non-finite weights, a scale count outside `1..=3`, or invalid scales.
pub fn exact_ternary_assignment(
    weights: &[f32],
    scales: &[f32],
) -> Result<Vec<Vec<i8>>, JointFitError> {
    if weights.is_empty() {
        return Err(JointFitError::EmptyWeights);
    }
    if let Some(index) = weights.iter().position(|weight| !weight.is_finite()) {
        return Err(JointFitError::NonFiniteWeight { index });
    }
    if !(1..=3).contains(&scales.len()) {
        return Err(JointFitError::InvalidScaleCount { got: scales.len() });
    }
    if let Some(index) = scales
        .iter()
        .position(|scale| !scale.is_finite() || *scale < 0.0)
    {
        return Err(JointFitError::InvalidScale { index });
    }

    const CODES: [i8; 3] = [0, -1, 1];
    let states = 3_usize.pow(scales.len() as u32);
    let mut trits = vec![vec![0_i8; weights.len()]; scales.len()];
    for (weight_index, &weight) in weights.iter().enumerate() {
        let mut best_error = f64::INFINITY;
        let mut best_codes = [0_i8; 3];
        for state in 0..states {
            let mut encoded = state;
            let mut reconstruction = 0.0_f32;
            let mut candidate = [0_i8; 3];
            for plane in 0..scales.len() {
                let trit = CODES[encoded % 3];
                encoded /= 3;
                candidate[plane] = trit;
                reconstruction += scales[plane] * f32::from(trit);
            }
            let error = f64::from(weight) - f64::from(reconstruction);
            let squared = error * error;
            if squared < best_error {
                best_error = squared;
                best_codes = candidate;
            }
        }
        for plane in 0..scales.len() {
            trits[plane][weight_index] = best_codes[plane];
        }
    }
    Ok(trits)
}

/// Jointly fit up to three zero-point-free additive ternary planes.
///
/// The metric may be identity, non-negative diagonal curvature, or a validated dense PSD group
/// curvature. The representation contains only per-plane scales and trits: no residual offset or
/// arbitrary codebook is introduced.
///
/// # Errors
/// Rejects empty/non-finite weights, invalid configuration, and malformed metrics.
pub fn fit_joint_ternary(
    weights: &[f32],
    fit_metric: JointFitMetric<'_>,
    config: JointFitConfig,
) -> Result<JointTernaryFit, JointFitError> {
    if !(1..=3).contains(&config.planes) {
        return Err(JointFitError::InvalidPlaneCount { got: config.planes });
    }
    if weights.is_empty() {
        return Err(JointFitError::EmptyWeights);
    }
    if config.max_iterations == 0 {
        return Err(JointFitError::InvalidMaxIterations);
    }
    if config.em_restarts == 0 {
        return Err(JointFitError::InvalidRestartCount);
    }
    if !config.ridge.is_finite() || config.ridge <= 0.0 {
        return Err(JointFitError::InvalidRidge);
    }
    if !config.ridge_condition_limit.is_finite() || config.ridge_condition_limit <= 1.0 {
        return Err(JointFitError::InvalidConditionLimit);
    }
    if let Some(index) = weights.iter().position(|weight| !weight.is_finite()) {
        return Err(JointFitError::NonFiniteWeight { index });
    }
    let metric_diagonal: Vec<f64> = match fit_metric {
        JointFitMetric::Identity => vec![1.0; weights.len()],
        JointFitMetric::Diagonal(values) => {
            if values.len() != weights.len() {
                return Err(JointFitError::MetricLengthMismatch {
                    expected: weights.len(),
                    got: values.len(),
                });
            }
            if let Some(index) = values
                .iter()
                .position(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(JointFitError::InvalidMetric { index });
            }
            if !values.iter().any(|value| *value > 0.0) {
                return Err(JointFitError::ZeroMetric);
            }
            values.iter().map(|value| f64::from(*value)).collect()
        }
        JointFitMetric::Dense(dense) => {
            if dense.dimension != weights.len() {
                return Err(JointFitError::DenseMetricDimensionMismatch {
                    expected: weights.len(),
                    got: dense.dimension,
                });
            }
            (0..dense.dimension)
                .map(|index| dense.values[index * dense.dimension + index].max(0.0))
                .collect()
        }
    };
    let metric_sum: f64 = metric_diagonal.iter().sum();
    if metric_sum <= 0.0 {
        return Err(JointFitError::ZeroMetric);
    }
    if !metric_sum.is_finite() {
        return Err(JointFitError::ScaleSolveFailed);
    }
    let mut starts = Vec::with_capacity(config.em_restarts + usize::from(config.planes > 1));
    for restart in 0..config.em_restarts {
        let scales = deterministic_initial_scales(weights, &metric_diagonal, config, restart)?;
        starts.push(optimize_start(
            weights,
            fit_metric,
            config,
            scales,
            JointFitStartKind::DeterministicRestart(restart),
        )?);
    }

    // A lower-plane embedding is an additional basin, not one of the configured OA-EM restarts.
    // It guarantees P-monotonicity without pretending the non-convex solver is globally optimal.
    if config.planes > 1 {
        let lower = fit_joint_ternary(
            weights,
            fit_metric,
            JointFitConfig {
                planes: config.planes - 1,
                ..config
            },
        )?;
        let lower_receipt = lower.restart_receipts[lower.selected_start].clone();
        let lower_accepted_objectives = lower.accepted_objectives;
        let mut scales = lower.scales;
        scales.push(0.0);
        let mut trits = lower.trits;
        trits.push(vec![0; weights.len()]);
        starts.push(FitState {
            scales,
            trits,
            reconstruction: lower.reconstruction,
            objective: lower.objective,
            accepted_objectives: lower_accepted_objectives,
            receipt: JointFitRestartReceipt {
                kind: JointFitStartKind::LowerPlaneFallback,
                initial_objective: lower_receipt.initial_objective,
                final_objective: lower_receipt.final_objective,
                accepted_updates: lower_receipt.accepted_updates,
                scale_solves: lower_receipt.scale_solves,
            },
        });
    }

    let selected_start = starts
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.objective.total_cmp(&right.objective))
        .map(|(index, _)| index)
        .expect("validated positive restart count");
    let restart_receipts = starts.iter().map(|state| state.receipt.clone()).collect();
    let selected = starts.swap_remove(selected_start);
    Ok(JointTernaryFit {
        scales: selected.scales,
        trits: selected.trits,
        reconstruction: selected.reconstruction,
        objective: selected.objective,
        accepted_objectives: selected.accepted_objectives,
        restart_receipts,
        selected_start,
    })
}

#[derive(Clone, Debug)]
struct FitState {
    scales: Vec<f32>,
    trits: Vec<Vec<i8>>,
    reconstruction: Vec<f32>,
    objective: f64,
    accepted_objectives: Vec<f64>,
    receipt: JointFitRestartReceipt,
}

fn optimize_start(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    config: JointFitConfig,
    scales: Vec<f32>,
    kind: JointFitStartKind,
) -> Result<FitState, JointFitError> {
    let trits = assignment_for_metric(weights, &scales, metric)?;
    let reconstruction = reconstruct_planes(&scales, &trits, weights.len());
    let objective = metric_objective(weights, &reconstruction, metric)?;
    let mut state = FitState {
        scales,
        trits,
        reconstruction,
        objective,
        accepted_objectives: vec![objective],
        receipt: JointFitRestartReceipt {
            kind,
            initial_objective: objective,
            final_objective: objective,
            accepted_updates: Vec::new(),
            scale_solves: Vec::new(),
        },
    };

    for iteration in 0..config.max_iterations {
        let mut improved = false;
        let scale_outcome = solve_scales(
            weights,
            &state.trits,
            metric,
            config.ridge,
            config.ridge_condition_limit,
            config.scale_precision,
        )?;
        let scale_reconstruction =
            reconstruct_planes(&scale_outcome.scales, &scale_outcome.trits, weights.len());
        let scale_objective = metric_objective(weights, &scale_reconstruction, metric)?;
        let scale_accepted = scale_objective < state.objective;
        state.receipt.scale_solves.push(ScaleSolveReceipt {
            iteration,
            telemetry: scale_outcome.telemetry,
            accepted: scale_accepted,
        });
        if scale_accepted {
            let objective_before = state.objective;
            state.scales = scale_outcome.scales;
            state.trits = scale_outcome.trits;
            state.reconstruction = scale_reconstruction;
            state.objective = scale_objective;
            state.accepted_objectives.push(scale_objective);
            state.receipt.accepted_updates.push(JointFitUpdateReceipt {
                iteration,
                phase: JointFitUpdatePhase::Scale,
                objective_before,
                objective_after: scale_objective,
            });
            improved = true;
        }

        let assignment = assignment_for_metric(weights, &state.scales, metric)?;
        let assignment_reconstruction =
            reconstruct_planes(&state.scales, &assignment, weights.len());
        let assignment_objective = metric_objective(weights, &assignment_reconstruction, metric)?;
        if assignment_objective < state.objective {
            let objective_before = state.objective;
            state.trits = assignment;
            state.reconstruction = assignment_reconstruction;
            state.objective = assignment_objective;
            state.accepted_objectives.push(assignment_objective);
            state.receipt.accepted_updates.push(JointFitUpdateReceipt {
                iteration,
                phase: JointFitUpdatePhase::Assignment,
                objective_before,
                objective_after: assignment_objective,
            });
            improved = true;
        }
        if !improved {
            break;
        }
    }
    state.receipt.final_objective = state.objective;
    Ok(state)
}

fn deterministic_initial_scales(
    weights: &[f32],
    metric_diagonal: &[f64],
    config: JointFitConfig,
    restart: usize,
) -> Result<Vec<f32>, JointFitError> {
    let metric_sum: f64 = metric_diagonal.iter().sum();
    let mut scales = Vec::with_capacity(config.planes);
    if config.planes == 2 && restart + 1 == config.em_restarts {
        // Reserve one deterministic P2 basin for a max-minus-min decomposition. This exactly
        // represents groups such as [small, -large, -large] with scales
        // [large, large - small], avoiding the dead second plane that residual-mean starts can
        // produce. Other restarts and the lower-plane fallback remain available for general data.
        let mut minimum_positive = f32::INFINITY;
        let mut maximum = 0.0_f32;
        for (&weight, &metric_weight) in weights.iter().zip(metric_diagonal) {
            if metric_weight <= 0.0 {
                continue;
            }
            let magnitude = weight.abs();
            maximum = maximum.max(magnitude);
            if magnitude > 0.0 {
                minimum_positive = minimum_positive.min(magnitude);
            }
        }
        let difference = if minimum_positive.is_finite() {
            maximum - minimum_positive
        } else {
            0.0
        };
        scales.push(deployment_scale(maximum, config.scale_precision, 0)?);
        scales.push(deployment_scale(difference, config.scale_precision, 1)?);
    } else if restart == 0 {
        let mut residual = weights.to_vec();
        for plane in 0..config.planes {
            let weighted_abs: f64 = residual
                .iter()
                .zip(metric_diagonal)
                .map(|(value, weight)| f64::from(value.abs()) * weight)
                .sum();
            let scale = deployment_scale(
                (weighted_abs / metric_sum) as f32,
                config.scale_precision,
                plane,
            )?;
            scales.push(scale);
            if scale > 0.0 {
                for value in &mut residual {
                    let trit = (*value / scale).round().clamp(-1.0, 1.0);
                    *value -= scale * trit;
                }
            }
        }
    } else {
        let quantile = 0.5 + 0.45 * (restart as f64 / config.em_restarts as f64);
        let anchor = weighted_abs_quantile(weights, metric_diagonal, quantile);
        for plane in 0..config.planes {
            let divisor = 2_f64.powi(plane as i32);
            let modulation = 1.0 + 0.125 * (((restart + plane) % 3) as f64 - 1.0);
            scales.push(deployment_scale(
                (anchor * modulation / divisor) as f32,
                config.scale_precision,
                plane,
            )?);
        }
    }
    scales.sort_by(|left, right| right.total_cmp(left));
    Ok(scales)
}

fn weighted_abs_quantile(weights: &[f32], metric_diagonal: &[f64], quantile: f64) -> f64 {
    let mut values: Vec<(f32, f64, usize)> = weights
        .iter()
        .zip(metric_diagonal)
        .enumerate()
        .map(|(index, (value, weight))| (value.abs(), *weight, index))
        .collect();
    values.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.2.cmp(&right.2))
    });
    let total: f64 = values.iter().map(|value| value.1).sum();
    let target = total * quantile.clamp(0.0, 1.0);
    let mut cumulative = 0.0;
    for (value, weight, _) in &values {
        cumulative += weight;
        if cumulative >= target {
            return f64::from(*value);
        }
    }
    values.last().map_or(0.0, |value| f64::from(value.0))
}

fn metric_entry(metric: JointFitMetric<'_>, row: usize, col: usize) -> f64 {
    match metric {
        JointFitMetric::Identity => f64::from(row == col),
        JointFitMetric::Diagonal(values) => {
            if row == col {
                f64::from(values[row])
            } else {
                0.0
            }
        }
        JointFitMetric::Dense(dense) => dense.values[row * dense.dimension + col],
    }
}

#[derive(Clone, Debug)]
struct ScaleSolveOutcome {
    scales: Vec<f32>,
    trits: Vec<Vec<i8>>,
    telemetry: ScaleSolveTelemetry,
}

fn solve_scales(
    weights: &[f32],
    trits: &[Vec<i8>],
    metric: JointFitMetric<'_>,
    ridge: f64,
    condition_limit: f64,
    precision: ScalePrecision,
) -> Result<ScaleSolveOutcome, JointFitError> {
    let planes = trits.len();
    let mut normal = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for plane in 0..planes {
        for other in 0..planes {
            for row in 0..weights.len() {
                let left = f64::from(trits[plane][row]);
                if left == 0.0 {
                    continue;
                }
                for (col, &other_trit) in trits[other].iter().enumerate() {
                    normal[plane][other] +=
                        left * metric_entry(metric, row, col) * f64::from(other_trit);
                }
            }
        }
        for (row, plane_trit) in trits[plane].iter().enumerate() {
            let left = f64::from(*plane_trit);
            if left == 0.0 {
                continue;
            }
            for (col, &weight) in weights.iter().enumerate() {
                rhs[plane] += left * metric_entry(metric, row, col) * f64::from(weight);
            }
        }
    }

    // Floating accumulation can leave a few ulps of asymmetry. The mathematical normal matrix is
    // symmetric PSD, so use the deterministic average before spectral conditioning and solving.
    for row in [0_usize, 1, 2].into_iter().take(planes) {
        for col in [0_usize, 1, 2].into_iter().take(planes).skip(row + 1) {
            let symmetric = 0.5 * (normal[row][col] + normal[col][row]);
            normal[row][col] = symmetric;
            normal[col][row] = symmetric;
        }
    }
    let (minimum_eigenvalue, maximum_eigenvalue) = symmetric_eigen_extrema(normal, planes);
    let spectral_tolerance = maximum_eigenvalue.abs().max(1.0) * f64::EPSILON * 64.0;
    if minimum_eigenvalue < -spectral_tolerance || !maximum_eigenvalue.is_finite() {
        return Err(JointFitError::ScaleSolveFailed);
    }
    let minimum_eigenvalue = minimum_eigenvalue.max(0.0);
    let condition_before = spectral_condition(minimum_eigenvalue, maximum_eigenvalue);
    let condition_ridge = if condition_before <= condition_limit || maximum_eigenvalue == 0.0 {
        0.0
    } else {
        ((maximum_eigenvalue - condition_limit * minimum_eigenvalue) / (condition_limit - 1.0))
            .max(0.0)
    };
    let ridge_used = ridge.max(condition_ridge);
    let adaptive_ridge = ridge_used > ridge;
    for (plane, row) in normal.iter_mut().enumerate().take(planes) {
        row[plane] += ridge_used;
    }
    let condition_after = spectral_condition(
        minimum_eigenvalue + ridge_used,
        maximum_eigenvalue + ridge_used,
    );

    for pivot in 0..planes {
        let selected = (pivot..planes)
            .max_by(|a, b| normal[*a][pivot].abs().total_cmp(&normal[*b][pivot].abs()))
            .expect("non-empty pivot range");
        normal.swap(pivot, selected);
        rhs.swap(pivot, selected);
        let divisor = normal[pivot][pivot];
        if !divisor.is_finite() || divisor.abs() <= f64::EPSILON {
            return Err(JointFitError::ScaleSolveFailed);
        }
        for value in &mut normal[pivot][pivot..planes] {
            *value /= divisor;
        }
        rhs[pivot] /= divisor;
        let normalized_pivot = normal[pivot];
        for row in 0..planes {
            if row == pivot {
                continue;
            }
            let factor = normal[row][pivot];
            for (value, pivot_value) in normal[row][pivot..planes]
                .iter_mut()
                .zip(&normalized_pivot[pivot..planes])
            {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    if rhs[..planes].iter().any(|scale| !scale.is_finite()) {
        return Err(JointFitError::ScaleSolveFailed);
    }

    // Plane signs are a representation symmetry. Canonicalize each negative solved coefficient
    // by flipping that plane's trits, then sort scales and trit planes with the same permutation.
    let mut canonical_trits = trits.to_vec();
    let mut signed_scales = rhs[..planes].to_vec();
    for (scale, plane_trits) in signed_scales.iter_mut().zip(&mut canonical_trits) {
        if *scale < 0.0 {
            *scale = -*scale;
            for trit in plane_trits {
                *trit = -*trit;
            }
        }
    }
    let mut order: Vec<usize> = (0..planes).collect();
    order.sort_by(|left, right| {
        signed_scales[*right]
            .total_cmp(&signed_scales[*left])
            .then_with(|| left.cmp(right))
    });
    let mut scales = Vec::with_capacity(planes);
    let mut ordered_trits = Vec::with_capacity(planes);
    for (plane, source) in order.into_iter().enumerate() {
        scales.push(deployment_scale(
            signed_scales[source] as f32,
            precision,
            plane,
        )?);
        ordered_trits.push(canonical_trits[source].clone());
    }
    Ok(ScaleSolveOutcome {
        scales,
        trits: ordered_trits,
        telemetry: ScaleSolveTelemetry {
            condition_before,
            condition_after,
            ridge_used,
            adaptive_ridge,
        },
    })
}

fn spectral_condition(minimum: f64, maximum: f64) -> f64 {
    if maximum == 0.0 || minimum <= 0.0 {
        f64::INFINITY
    } else {
        maximum / minimum
    }
}

fn symmetric_eigen_extrema(mut matrix: [[f64; 3]; 3], dimension: usize) -> (f64, f64) {
    if dimension == 1 {
        return (matrix[0][0], matrix[0][0]);
    }
    for _ in 0..32 {
        let mut pivot = (0, 1);
        for row in 0..dimension {
            for col in row + 1..dimension {
                if matrix[row][col].abs() > matrix[pivot.0][pivot.1].abs() {
                    pivot = (row, col);
                }
            }
        }
        let (row, col) = pivot;
        let off_diagonal = matrix[row][col];
        let scale = matrix[row][row].abs().max(matrix[col][col].abs()).max(1.0);
        if off_diagonal.abs() <= scale * f64::EPSILON * 16.0 {
            break;
        }
        let tau = (matrix[col][col] - matrix[row][row]) / (2.0 * off_diagonal);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let row_diagonal = matrix[row][row];
        let col_diagonal = matrix[col][col];
        matrix[row][row] = cosine * cosine * row_diagonal - 2.0 * sine * cosine * off_diagonal
            + sine * sine * col_diagonal;
        matrix[col][col] = sine * sine * row_diagonal
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * col_diagonal;
        matrix[row][col] = 0.0;
        matrix[col][row] = 0.0;
        for other in [0_usize, 1, 2].into_iter().take(dimension) {
            if other == row || other == col {
                continue;
            }
            let other_row = matrix[other][row];
            let other_col = matrix[other][col];
            matrix[other][row] = cosine * other_row - sine * other_col;
            matrix[row][other] = matrix[other][row];
            matrix[other][col] = sine * other_row + cosine * other_col;
            matrix[col][other] = matrix[other][col];
        }
    }
    let mut minimum = matrix[0][0];
    let mut maximum = matrix[0][0];
    for (index, row) in matrix.iter().enumerate().take(dimension).skip(1) {
        minimum = minimum.min(row[index]);
        maximum = maximum.max(row[index]);
    }
    (minimum, maximum)
}

fn deployment_scale(
    scale: f32,
    precision: ScalePrecision,
    plane: usize,
) -> Result<f32, JointFitError> {
    let stored = match precision {
        ScalePrecision::F32 => scale,
        ScalePrecision::F16 => f16::from_f32(scale).to_f32(),
    };
    if stored.is_finite() {
        Ok(stored)
    } else {
        Err(JointFitError::ScaleNotRepresentable { plane })
    }
}

fn assignment_for_metric(
    weights: &[f32],
    scales: &[f32],
    metric: JointFitMetric<'_>,
) -> Result<Vec<Vec<i8>>, JointFitError> {
    let mut trits = exact_ternary_assignment(weights, scales)?;
    let JointFitMetric::Dense(dense) = metric else {
        return Ok(trits);
    };

    const CODES: [i8; 3] = [0, -1, 1];
    let states = 3_usize.pow(scales.len() as u32);
    let mut reconstruction = reconstruct_planes(scales, &trits, weights.len());
    let mut error: Vec<f64> = weights
        .iter()
        .zip(&reconstruction)
        .map(|(weight, fitted)| f64::from(*weight) - f64::from(*fitted))
        .collect();
    let mut h_error = vec![0.0_f64; weights.len()];
    for (row, value) in h_error.iter_mut().enumerate() {
        *value = (0..weights.len())
            .map(|col| dense.values[row * dense.dimension + col] * error[col])
            .sum();
    }

    // Deterministic coordinate descent. Each coordinate update is the exact 3^P minimizer with
    // every other coordinate fixed, and the full dense quadratic is updated analytically.
    for _ in 0..8 {
        let mut changed = false;
        for index in 0..weights.len() {
            let old_reconstruction = reconstruction[index];
            let mut best_delta = 0.0_f64;
            let mut best_reconstruction = old_reconstruction;
            let mut best_codes: Vec<i8> = trits.iter().map(|plane| plane[index]).collect();
            for state in 0..states {
                let mut encoded = state;
                let mut candidate_reconstruction = 0.0_f32;
                let mut candidate_codes = vec![0_i8; scales.len()];
                for plane in 0..scales.len() {
                    let trit = CODES[encoded % 3];
                    encoded /= 3;
                    candidate_codes[plane] = trit;
                    candidate_reconstruction += scales[plane] * f32::from(trit);
                }
                let error_delta = f64::from(old_reconstruction - candidate_reconstruction);
                let objective_delta = 2.0 * error_delta * h_error[index]
                    + error_delta * error_delta * dense.values[index * dense.dimension + index];
                if objective_delta < best_delta {
                    best_delta = objective_delta;
                    best_reconstruction = candidate_reconstruction;
                    best_codes = candidate_codes;
                }
            }
            if best_delta < 0.0 {
                let error_delta = f64::from(old_reconstruction - best_reconstruction);
                reconstruction[index] = best_reconstruction;
                error[index] += error_delta;
                for (row, value) in h_error.iter_mut().enumerate() {
                    *value += dense.values[row * dense.dimension + index] * error_delta;
                }
                for (plane, values) in trits.iter_mut().enumerate() {
                    values[index] = best_codes[plane];
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(trits)
}

fn reconstruct_planes(scales: &[f32], trits: &[Vec<i8>], len: usize) -> Vec<f32> {
    let mut reconstruction = vec![0.0_f32; len];
    for (scale, plane) in scales.iter().zip(trits) {
        for (value, trit) in reconstruction.iter_mut().zip(plane) {
            *value += *scale * f32::from(*trit);
        }
    }
    reconstruction
}

fn metric_objective(
    weights: &[f32],
    reconstruction: &[f32],
    metric: JointFitMetric<'_>,
) -> Result<f64, JointFitError> {
    let error: Vec<f64> = weights
        .iter()
        .zip(reconstruction)
        .map(|(weight, fitted)| f64::from(*weight) - f64::from(*fitted))
        .collect();
    let mut objective = 0.0_f64;
    match metric {
        JointFitMetric::Identity => {
            for value in error {
                accumulate_objective_term(&mut objective, value * value)?;
            }
        }
        JointFitMetric::Diagonal(diagonal) => {
            for (value, weight) in error.iter().zip(diagonal) {
                let squared = value * value;
                accumulate_objective_term(&mut objective, squared * f64::from(*weight))?;
            }
        }
        JointFitMetric::Dense(dense) => {
            for row in 0..dense.dimension {
                for col in 0..dense.dimension {
                    let weighted = error[row] * dense.values[row * dense.dimension + col];
                    accumulate_objective_term(&mut objective, weighted * error[col])?;
                }
            }
        }
    }
    Ok(objective.max(0.0))
}

fn accumulate_objective_term(objective: &mut f64, term: f64) -> Result<(), JointFitError> {
    if !term.is_finite() {
        return Err(JointFitError::NonFiniteObjective);
    }
    *objective += term;
    if !objective.is_finite() {
        return Err(JointFitError::NonFiniteObjective);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(scales: &[f32], trits: &[Vec<i8>], len: usize) -> Vec<f32> {
        let mut out = vec![0.0; len];
        for (scale, plane) in scales.iter().zip(trits) {
            for (value, trit) in out.iter_mut().zip(plane) {
                *value += *scale * f32::from(*trit);
            }
        }
        out
    }

    fn squared_error(weights: &[f32], reconstruction: &[f32]) -> f64 {
        weights
            .iter()
            .zip(reconstruction)
            .map(|(weight, fitted)| {
                let error = f64::from(*weight) - f64::from(*fitted);
                error * error
            })
            .sum()
    }

    fn greedy_residual_reconstruction(weights: &[f32], planes: usize) -> Vec<f32> {
        let mut residual = weights.to_vec();
        let mut reconstruction = vec![0.0_f32; weights.len()];
        for _ in 0..planes {
            let scale =
                residual.iter().map(|value| value.abs()).sum::<f32>() / residual.len() as f32;
            if scale == 0.0 {
                continue;
            }
            for ((source, fitted), remainder) in
                weights.iter().zip(&mut reconstruction).zip(&mut residual)
            {
                let trit = (*remainder / scale).round().clamp(-1.0, 1.0);
                *fitted += scale * trit;
                *remainder = *source - *fitted;
            }
        }
        reconstruction
    }

    fn joint_grid_oracle(weights: &[f32], planes: usize, scale_grid: &[f32]) -> f64 {
        let codes = [-1_i8, 0, 1];
        let scale_state_count = scale_grid.len().pow(planes as u32);
        let trit_state_count = 3_usize.pow((weights.len() * planes) as u32);
        let mut best = f64::INFINITY;

        for mut scale_state in 0..scale_state_count {
            let mut scales = vec![0.0_f32; planes];
            for scale in &mut scales {
                *scale = scale_grid[scale_state % scale_grid.len()];
                scale_state /= scale_grid.len();
            }
            for mut trit_state in 0..trit_state_count {
                let mut trits = vec![vec![0_i8; weights.len()]; planes];
                for plane in &mut trits {
                    for trit in plane {
                        *trit = codes[trit_state % 3];
                        trit_state /= 3;
                    }
                }
                best = best.min(squared_error(
                    weights,
                    &reconstruct(&scales, &trits, weights.len()),
                ));
            }
        }
        best
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let weights = [1.0, -2.0];

        let zero_planes = fit_joint_ternary(
            &weights,
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 0,
                ..JointFitConfig::default()
            },
        );
        assert_eq!(
            zero_planes,
            Err(JointFitError::InvalidPlaneCount { got: 0 })
        );

        let too_many_planes = fit_joint_ternary(
            &weights,
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 4,
                ..JointFitConfig::default()
            },
        );
        assert_eq!(
            too_many_planes,
            Err(JointFitError::InvalidPlaneCount { got: 4 })
        );

        assert_eq!(
            fit_joint_ternary(&[], JointFitMetric::Identity, JointFitConfig::default()),
            Err(JointFitError::EmptyWeights)
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    max_iterations: 0,
                    ..JointFitConfig::default()
                },
            ),
            Err(JointFitError::InvalidMaxIterations)
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    ridge: 0.0,
                    ..JointFitConfig::default()
                },
            ),
            Err(JointFitError::InvalidRidge)
        );
        assert_eq!(
            fit_joint_ternary(
                &[1.0, f32::NAN],
                JointFitMetric::Identity,
                JointFitConfig::default(),
            ),
            Err(JointFitError::NonFiniteWeight { index: 1 })
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Diagonal(&[1.0]),
                JointFitConfig::default(),
            ),
            Err(JointFitError::MetricLengthMismatch {
                expected: 2,
                got: 1,
            })
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Diagonal(&[1.0, -1.0]),
                JointFitConfig::default(),
            ),
            Err(JointFitError::InvalidMetric { index: 1 })
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Diagonal(&[0.0, 0.0]),
                JointFitConfig::default(),
            ),
            Err(JointFitError::ZeroMetric)
        );

        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    em_restarts: 0,
                    ..JointFitConfig::default()
                },
            ),
            Err(JointFitError::InvalidRestartCount)
        );
        assert_eq!(
            fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    ridge_condition_limit: 1.0,
                    ..JointFitConfig::default()
                },
            ),
            Err(JointFitError::InvalidConditionLimit)
        );
    }

    #[test]
    fn exact_assignment_matches_global_exhaustive_oracle() {
        let weights = [0.3, -1.4, 2.1];
        let scales = [1.0, 0.4];
        let got = exact_ternary_assignment(&weights, &scales).expect("valid assignment");
        let got_error = squared_error(&weights, &reconstruct(&scales, &got, weights.len()));

        // Independent global oracle: enumerate all 3^(planes*weights) complete ternary matrices.
        let codes = [-1_i8, 0, 1];
        let state_count = 3_usize.pow((weights.len() * scales.len()) as u32);
        let mut oracle_error = f64::INFINITY;
        for mut state in 0..state_count {
            let mut trits = vec![vec![0_i8; weights.len()]; scales.len()];
            for plane in &mut trits {
                for trit in plane {
                    *trit = codes[state % 3];
                    state /= 3;
                }
            }
            oracle_error = oracle_error.min(squared_error(
                &weights,
                &reconstruct(&scales, &trits, weights.len()),
            ));
        }

        assert_eq!(got_error.to_bits(), oracle_error.to_bits());
    }

    #[test]
    fn fitting_is_bitwise_deterministic() {
        let weights = [-2.4, -1.1, -0.2, 0.0, 0.35, 0.9, 1.8, 3.2];
        let metric = [1.0, 4.0, 0.5, 2.0, 1.0, 3.0, 0.25, 5.0];
        let config = JointFitConfig {
            planes: 2,
            max_iterations: 12,
            ridge: 1e-7,
            scale_precision: ScalePrecision::F32,
            ..JointFitConfig::default()
        };

        let first = fit_joint_ternary(&weights, JointFitMetric::Diagonal(&metric), config)
            .expect("first fit");
        let second = fit_joint_ternary(&weights, JointFitMetric::Diagonal(&metric), config)
            .expect("second fit");

        assert_eq!(first, second);
        assert!(first.scales.iter().any(|scale| *scale > 0.0));
        assert!(first.reconstruction.iter().any(|weight| *weight != 0.0));
        assert_eq!(first.restart_receipts.len(), config.em_restarts + 1);
        assert!(first.selected_start < first.restart_receipts.len());
    }

    #[test]
    fn default_p2_uses_both_planes_for_exact_difference_solution() {
        let fit = fit_joint_ternary(
            &[0.25, -3.0, -3.0],
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 2,
                ..JointFitConfig::default()
            },
        )
        .expect("valid default P2 fit");

        assert_eq!(fit.objective.to_bits(), 0.0_f64.to_bits());
        assert_eq!(fit.reconstruction, vec![0.25, -3.0, -3.0]);
        assert!(fit.scales.iter().all(|scale| *scale > 0.0));
    }

    #[test]
    fn oa_em_evaluates_every_configured_restart_for_p1_through_p3() {
        let weights = [-2.9, -1.1, -0.18, 0.33, 0.95, 2.4];
        for planes in 1..=3 {
            let config = JointFitConfig {
                planes,
                em_restarts: 3,
                ..JointFitConfig::default()
            };
            let fit = fit_joint_ternary(&weights, JointFitMetric::Identity, config)
                .expect("deterministic multi-start fit");
            let restart_kinds: Vec<_> = fit
                .restart_receipts
                .iter()
                .filter_map(|receipt| match receipt.kind {
                    JointFitStartKind::DeterministicRestart(index) => Some(index),
                    JointFitStartKind::LowerPlaneFallback => None,
                })
                .collect();
            assert_eq!(restart_kinds, vec![0, 1, 2]);
            assert_eq!(
                fit.restart_receipts.len(),
                config.em_restarts + usize::from(planes > 1)
            );
            assert!(
                fit.restart_receipts
                    .iter()
                    .all(|receipt| receipt.final_objective <= receipt.initial_objective)
            );
        }
    }

    #[test]
    fn scale_sign_and_plane_order_canonicalization_preserve_reconstruction() {
        let weights = [-1.75, 1.75, -1.25, 1.25];
        let trits = vec![vec![1, -1, 1, -1], vec![-1, 1, 1, -1]];
        let outcome = solve_scales(
            &weights,
            &trits,
            JointFitMetric::Identity,
            1e-12,
            1e6,
            ScalePrecision::F32,
        )
        .expect("well-conditioned solve");

        assert!(outcome.scales.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(outcome.scales.iter().all(|scale| *scale >= 0.0));
        assert_eq!(outcome.trits[0], vec![-1, 1, -1, 1]);
        assert_eq!(outcome.trits[1], vec![-1, 1, 1, -1]);
        let fitted = reconstruct(&outcome.scales, &outcome.trits, weights.len());
        assert!(squared_error(&weights, &fitted) < 1e-20);
    }

    #[test]
    fn singular_scale_system_uses_reported_adaptive_ridge() {
        let weights = [-1.0, 0.5, 1.0];
        let trits = vec![vec![-1, 0, 1], vec![-1, 0, 1]];
        let outcome = solve_scales(
            &weights,
            &trits,
            JointFitMetric::Identity,
            1e-10,
            1e4,
            ScalePrecision::F32,
        )
        .expect("adaptive ridge makes the system solvable");

        assert!(outcome.telemetry.adaptive_ridge);
        assert!(outcome.telemetry.condition_before.is_infinite());
        assert!(outcome.telemetry.condition_after <= 1e4 * (1.0 + 1e-10));
        assert!(outcome.telemetry.ridge_used > 1e-10);
    }

    #[test]
    fn tiny_joint_fit_matches_full_trit_and_scale_grid_oracle() {
        let cases: [(&[f32], usize, &[f32]); 3] = [
            (&[-1.0, 0.0, 1.0], 1, &[0.0, 0.5, 1.0]),
            (&[-1.5, -0.5, 0.5, 1.5], 2, &[0.0, 0.5, 1.0]),
            (&[-1.75, 0.25], 3, &[0.0, 0.25, 0.5, 1.0]),
        ];

        for (weights, planes, scale_grid) in cases {
            let oracle = joint_grid_oracle(weights, planes, scale_grid);
            let fit = fit_joint_ternary(
                weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    planes,
                    ridge: 1e-12,
                    ..JointFitConfig::default()
                },
            )
            .expect("joint fit");
            assert!(
                (fit.objective - oracle).abs() <= 1e-12,
                "P{planes}: fit={}, oracle={oracle}",
                fit.objective
            );
        }
    }

    #[test]
    fn every_accepted_e_and_m_update_is_strictly_monotone() {
        let fit = fit_joint_ternary(
            &[-3.1, -1.4, -0.45, 0.15, 0.8, 2.2, 4.0],
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 3,
                ..JointFitConfig::default()
            },
        )
        .expect("joint fit");

        for restart in &fit.restart_receipts {
            for update in &restart.accepted_updates {
                assert!(update.objective_after < update.objective_before);
            }
            assert!(
                restart
                    .accepted_updates
                    .windows(2)
                    .all(|pair| pair[1].objective_before <= pair[0].objective_after)
            );
            for solve in &restart.scale_solves {
                assert!(solve.telemetry.ridge_used >= 1e-8);
            }
        }
        assert!(
            fit.accepted_objectives
                .windows(2)
                .all(|pair| pair[1] < pair[0])
        );
    }

    #[test]
    fn dense_metric_constructor_rejects_invalid_curvature() {
        let valid = DensePsdMetric::from_kfac_input_gram(2, &[2.0, 1.0, 1.0, 2.0], 3.0)
            .expect("positive scaled Gram");
        assert_eq!(valid.dimension(), 2);
        assert_eq!(valid.as_slice(), &[6.0, 3.0, 3.0, 6.0]);

        assert_eq!(
            DensePsdMetric::new(2, &[1.0, 0.2, 0.1, 1.0]),
            Err(JointFitError::AsymmetricDenseMetric { row: 0, col: 1 })
        );
        assert_eq!(
            DensePsdMetric::new(2, &[1.0, 2.0, 2.0, 1.0]),
            Err(JointFitError::NonPositiveSemidefiniteMetric { pivot: 1 })
        );
        assert_eq!(
            DensePsdMetric::new(2, &[-1.0, 0.0, 0.0, 1.0e20]),
            Err(JointFitError::NonPositiveSemidefiniteMetric { pivot: 0 })
        );
        assert_eq!(
            DensePsdMetric::new(2, &[1.0, f64::NAN, f64::NAN, 1.0]),
            Err(JointFitError::NonFiniteDenseMetric { row: 0, col: 1 })
        );
        assert_eq!(
            DensePsdMetric::new(2, &[0.0; 4]),
            Err(JointFitError::ZeroMetric)
        );
        assert_eq!(
            DensePsdMetric::from_kfac_input_gram(2, &[1.0, 0.0, 0.0, 1.0], 0.0),
            Err(JointFitError::InvalidKfacOutputWeight)
        );
    }

    #[test]
    fn dense_metric_psd_validation_is_relative_to_matrix_scale() {
        let tiny_spd = DensePsdMetric::new(2, &[2.0e-20, 1.0e-20, 1.0e-20, 2.0e-20])
            .expect("valid tiny SPD matrix");
        assert_eq!(tiny_spd.as_slice(), &[2.0e-20, 1.0e-20, 1.0e-20, 2.0e-20]);

        assert_eq!(
            DensePsdMetric::new(1, &[-1.0e-20]),
            Err(JointFitError::NonPositiveSemidefiniteMetric { pivot: 0 })
        );
        assert_eq!(
            DensePsdMetric::new(2, &[1.0e-20, 0.0, 0.0, -1.0e-20]),
            Err(JointFitError::NonPositiveSemidefiniteMetric { pivot: 1 })
        );
    }

    #[test]
    fn accepted_near_symmetric_metric_is_stored_as_one_exact_quadratic() {
        let metric = DensePsdMetric::new(2, &[1.0, 4.0e-11, -4.0e-11, 1.0e-20])
            .expect("near-symmetric PSD metric");

        assert_eq!(metric.as_slice(), &[1.0, 0.0, 0.0, 1.0e-20]);

        // This is the scale/reconstruction regime that used to make the asymmetric coordinate
        // delta predict an improvement while the public quadratic objective actually increased.
        let fit = fit_joint_ternary(
            &[0.0, -1.0e10],
            JointFitMetric::Dense(&metric),
            JointFitConfig::default(),
        )
        .expect("fit against canonical quadratic");
        assert!(
            fit.accepted_objectives
                .windows(2)
                .all(|pair| pair[1] < pair[0])
        );
    }

    #[test]
    fn dense_metric_scores_the_full_quadratic_form() {
        let weights = [1.0, 0.2];
        let dense = DensePsdMetric::new(2, &[2.0, 1.0, 1.0, 2.0]).expect("PSD metric");
        let fit = fit_joint_ternary(
            &weights,
            JointFitMetric::Dense(&dense),
            JointFitConfig::default(),
        )
        .expect("dense fit");
        let error = [
            f64::from(weights[0]) - f64::from(fit.reconstruction[0]),
            f64::from(weights[1]) - f64::from(fit.reconstruction[1]),
        ];
        let expected =
            2.0 * error[0] * error[0] + 2.0 * error[0] * error[1] + 2.0 * error[1] * error[1];

        assert!((fit.objective - expected).abs() <= 1e-14);
    }

    #[test]
    fn non_finite_objective_accumulation_returns_a_typed_error() {
        let dense =
            DensePsdMetric::new(2, &[1.0e250, 0.0, 0.0, 1.0e250]).expect("finite PSD metric");
        let fit = fit_joint_ternary(
            &[f32::MAX, 1.0],
            JointFitMetric::Dense(&dense),
            JointFitConfig::default(),
        );

        assert_eq!(fit, Err(JointFitError::NonFiniteObjective));
    }

    #[test]
    fn accepted_iterations_are_monotone_and_p2_dominates_baselines() {
        let weights = [-3.0, -1.7, -0.8, -0.15, 0.2, 0.65, 1.4, 2.8, 4.1];
        let p1 = fit_joint_ternary(
            &weights,
            JointFitMetric::Identity,
            JointFitConfig::default(),
        )
        .expect("P1 fit");
        let p2 = fit_joint_ternary(
            &weights,
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 2,
                ..JointFitConfig::default()
            },
        )
        .expect("P2 fit");
        let greedy = greedy_residual_reconstruction(&weights, 2);
        let greedy_error = squared_error(&weights, &greedy);

        assert!(p2.objective <= p1.objective);
        assert!(p2.objective <= greedy_error);
        assert!(
            p2.accepted_objectives.len() >= 2,
            "worked sample must exercise an update"
        );
        assert!(
            p2.accepted_objectives
                .windows(2)
                .all(|pair| pair[1] < pair[0])
        );
    }

    #[test]
    fn f16_scoring_returns_deployment_representable_scales() {
        let fit = fit_joint_ternary(
            &[-2.73, -0.91, -0.13, 0.37, 1.42, 3.19],
            JointFitMetric::Identity,
            JointFitConfig {
                planes: 2,
                scale_precision: ScalePrecision::F16,
                ..JointFitConfig::default()
            },
        )
        .expect("f16-scored fit");

        assert!(
            fit.scales
                .iter()
                .all(|scale| { half::f16::from_f32(*scale).to_f32().to_bits() == scale.to_bits() })
        );
    }
}
