//! Joint additive-ternary fitting for SALT V2.

use std::borrow::Cow;

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

/// Extra deterministic CAT-Q relay initialization basins appended after the OA-EM restarts.
///
/// Both basins default to off, which reproduces the historical restart set exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelayBasins {
    /// Softened-relay basin: fixed `0.5` normalized threshold, scale-only soft descent.
    pub softened: bool,
    /// Modulated basin: scale, threshold, and shift all soft-descended.
    pub modulated: bool,
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
    /// Extra deterministic CAT-Q relay initialization basins.
    pub relay_basins: RelayBasins,
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
            relay_basins: RelayBasins::default(),
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
    diagonal: Option<Vec<f64>>,
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
        let mut is_diagonal = true;
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
                is_diagonal &= symmetric == 0.0;
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
            diagonal: is_diagonal.then(|| {
                (0..dimension)
                    .map(|index| values[index * dimension + index])
                    .collect()
            }),
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

    /// Materialize `scale * self + diagonal_shift * I` without repeating PSD factorization.
    ///
    /// `self` already passed the full constructor. Non-negative scaling and a
    /// non-negative diagonal shift preserve symmetry and positive
    /// semidefiniteness, so only scalar validity, finite arithmetic, and a
    /// nonzero scored direction remain to check.
    pub(crate) fn scaled_with_diagonal(&self, scale: f64, diagonal_shift: f64) -> Option<Self> {
        if !scale.is_finite() || scale < 0.0 || !diagonal_shift.is_finite() || diagonal_shift < 0.0
        {
            return None;
        }
        let mut values = Vec::new();
        values.try_reserve_exact(self.values.len()).ok()?;
        let mut any_positive = false;
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                let value = self.values[row * self.dimension + column] * scale
                    + if row == column { diagonal_shift } else { 0.0 };
                if !value.is_finite() {
                    return None;
                }
                any_positive |= row == column && value > 0.0;
                values.push(value);
            }
        }
        any_positive.then_some(Self {
            dimension: self.dimension,
            values,
            diagonal: self.diagonal.as_ref().map(|diagonal| {
                diagonal
                    .iter()
                    .map(|value| value * scale + diagonal_shift)
                    .collect()
            }),
        })
    }

    /// Return exact diagonal storage when all off-diagonal entries are zero.
    ///
    /// This is intentionally exact: callers may replace dense quadratic scoring with the
    /// mathematically equivalent diagonal path only when no curvature information is discarded.
    pub(crate) fn exact_diagonal(&self) -> Option<&[f64]> {
        self.diagonal.as_deref()
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
    /// Non-negative diagonal curvature stored without narrowing f64 evidence.
    DiagonalF64(&'a [f64]),
    /// Affine diagonal curvature over f64 evidence without materializing a scaled vector.
    DiagonalAffine {
        /// Base input-side diagonal.
        values: &'a [f64],
        /// Output-side K-FAC scalar.
        scale: f64,
        /// Non-negative diagonal damping.
        shift: f64,
    },
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
    /// CAT-Q softened-relay basin: fixed normalized threshold, scale-only soft descent.
    SoftenedRelayBasin,
    /// CAT-Q modulated relay basin: scale, threshold, and shift soft-descended.
    ModulatedRelayBasin,
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
    let mut codebook = [(0.0_f32, [0_i8; 3], 0_usize); 27];
    for (state, entry) in codebook.iter_mut().take(states).enumerate() {
        let mut encoded = state;
        let mut reconstruction = 0.0_f32;
        let mut candidate = [0_i8; 3];
        for plane in 0..scales.len() {
            let trit = CODES[encoded % 3];
            encoded /= 3;
            candidate[plane] = trit;
            reconstruction += scales[plane] * f32::from(trit);
        }
        *entry = (reconstruction, candidate, state);
    }
    let codebook = &mut codebook[..states];
    // State is unique, making this a deterministic total order without a heap-backed sort buffer.
    codebook.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.2.cmp(&right.2))
    });
    // Equal reconstructions have identical error for every weight. Retain only
    // the lowest state for each total-ordered value so the assignment loop does
    // not need a second binary search to find the first lower duplicate.
    let mut unique_states = 0;
    for index in 0..states {
        let entry = codebook[index];
        if unique_states == 0 || codebook[unique_states - 1].0.total_cmp(&entry.0).is_ne() {
            codebook[unique_states] = entry;
            unique_states += 1;
        }
    }
    let codebook = &codebook[..unique_states];
    // Adjacent-codebook midpoints partition the real line into nearest-code
    // regions. Compute them in f64 so they are exact for the f32 endpoints.
    // The codebook has at most 27 entries, so keep its at most 26 boundaries on
    // the stack. This path runs once per fitted row; a heap allocation here
    // compounds across every row and accepted solver iteration.
    let mut midpoint_storage = [0.0_f64; 26];
    for (index, pair) in codebook.windows(2).enumerate() {
        midpoint_storage[index] = (f64::from(pair[0].0) + f64::from(pair[1].0)) * 0.5;
    }
    let midpoints = &midpoint_storage[..unique_states.saturating_sub(1)];
    let max_reconstruction = codebook
        .iter()
        .map(|entry| entry.0.abs())
        .fold(0.0_f32, f32::max);
    let min_reconstruction_gap = codebook
        .windows(2)
        .filter_map(|pair| {
            let gap = f64::from(pair[1].0) - f64::from(pair[0].0);
            (gap > 0.0).then_some(gap)
        })
        .fold(f64::INFINITY, f64::min);

    let mut trits = vec![vec![0_i8; weights.len()]; scales.len()];
    for (weight_index, &weight) in weights.iter().enumerate() {
        let mut best_codes = [0_i8; 3];
        // At extreme dynamic ranges, distinct f32 reconstructions can collapse
        // to the same f64 squared error. Preserve the original first-state tie
        // behavior there; ordinary values use the exact nearest-code fast path.
        let ill_conditioned = !weight.is_finite()
            || !max_reconstruction.is_finite()
            || f64::from(weight.abs()) > f64::from(max_reconstruction.max(1.0)) * 67_108_864.0
            || min_reconstruction_gap
                <= (f64::from(weight.abs()) + f64::from(max_reconstruction)) * (1.0 / 67_108_864.0);
        if ill_conditioned {
            let mut best_error = f64::INFINITY;
            let mut best_state = usize::MAX;
            for &(reconstruction, candidate, state) in codebook.iter() {
                let error = f64::from(weight) - f64::from(reconstruction);
                let squared = error * error;
                if squared < best_error || (squared == best_error && state < best_state) {
                    best_error = squared;
                    best_codes = candidate;
                    best_state = state;
                }
            }
            for plane in 0..scales.len() {
                trits[plane][weight_index] = best_codes[plane];
            }
            continue;
        }
        let value = f64::from(weight);
        let upper = midpoints.partition_point(|midpoint| *midpoint < value);
        let candidate_index = if upper < midpoints.len() && midpoints[upper] == value {
            // At an exact midpoint, retain the original exhaustive oracle's
            // deterministic lower-state tie break.
            if codebook[upper].2 < codebook[upper + 1].2 {
                upper
            } else {
                upper + 1
            }
        } else if upper == 0 {
            0
        } else if upper == midpoints.len() {
            codebook.len() - 1
        } else {
            upper
        };
        let (_, candidate, _) = codebook[candidate_index];
        best_codes = candidate;
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
    #[cfg(test)]
    let validation_started = std::time::Instant::now();
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
    let metric_diagonal: Cow<'_, [f64]> = match fit_metric {
        JointFitMetric::Identity => Cow::Owned(vec![1.0; weights.len()]),
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
            Cow::Owned(values.iter().map(|value| f64::from(*value)).collect())
        }
        JointFitMetric::DiagonalF64(values) => {
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
            Cow::Borrowed(values)
        }
        JointFitMetric::DiagonalAffine {
            values,
            scale,
            shift,
        } => {
            if !scale.is_finite() || scale < 0.0 || !shift.is_finite() || shift < 0.0 {
                return Err(JointFitError::InvalidMetric { index: 0 });
            }
            if values.len() != weights.len() {
                return Err(JointFitError::MetricLengthMismatch {
                    expected: weights.len(),
                    got: values.len(),
                });
            }
            let mut diagonal = Vec::with_capacity(values.len());
            for (index, value) in values.iter().enumerate() {
                let value = *value * scale + shift;
                if !value.is_finite() || value < 0.0 {
                    return Err(JointFitError::InvalidMetric { index });
                }
                diagonal.push(value);
            }
            if !diagonal.iter().any(|value| *value > 0.0) {
                return Err(JointFitError::ZeroMetric);
            }
            Cow::Owned(diagonal)
        }
        JointFitMetric::Dense(dense) => {
            if dense.dimension != weights.len() {
                return Err(JointFitError::DenseMetricDimensionMismatch {
                    expected: weights.len(),
                    got: dense.dimension,
                });
            }
            Cow::Owned(
                (0..dense.dimension)
                    .map(|index| dense.values[index * dense.dimension + index].max(0.0))
                    .collect(),
            )
        }
    };
    let metric_sum: f64 = metric_diagonal.iter().sum();
    if metric_sum <= 0.0 {
        return Err(JointFitError::ZeroMetric);
    }
    if !metric_sum.is_finite() {
        return Err(JointFitError::ScaleSolveFailed);
    }
    #[cfg(test)]
    record_solver_phase(3, validation_started);
    #[cfg(test)]
    let order_started = std::time::Instant::now();
    let weighted_abs_order = if config.em_restarts > 1 {
        weighted_abs_order(weights, &metric_diagonal)
    } else {
        WeightedAbsOrder::default()
    };
    #[cfg(test)]
    record_solver_phase(4, order_started);
    fit_joint_ternary_prepared(
        weights,
        fit_metric,
        config,
        &metric_diagonal,
        metric_sum,
        &weighted_abs_order,
    )
}

fn fit_joint_ternary_prepared(
    weights: &[f32],
    fit_metric: JointFitMetric<'_>,
    config: JointFitConfig,
    metric_diagonal: &[f64],
    metric_sum: f64,
    weighted_abs_order: &WeightedAbsOrder,
) -> Result<JointTernaryFit, JointFitError> {
    let relay_starts =
        usize::from(config.relay_basins.softened) + usize::from(config.relay_basins.modulated);
    let mut starts =
        Vec::with_capacity(config.em_restarts + relay_starts + usize::from(config.planes > 1));
    for restart in 0..config.em_restarts {
        #[cfg(test)]
        let initialization_started = std::time::Instant::now();
        let scales = deterministic_initial_scales(
            weights,
            metric_diagonal,
            metric_sum,
            weighted_abs_order,
            config,
            restart,
        )?;
        #[cfg(test)]
        record_solver_phase(5, initialization_started);
        starts.push(optimize_start(
            weights,
            fit_metric,
            config,
            scales,
            JointFitStartKind::DeterministicRestart(restart),
        )?);
    }

    // Relay basins are extra candidates appended after the configured OA-EM restarts, so the
    // deterministic restart indices and receipts are byte-identical when both basins are off.
    for (enabled, kind) in [
        (
            config.relay_basins.softened,
            JointFitStartKind::SoftenedRelayBasin,
        ),
        (
            config.relay_basins.modulated,
            JointFitStartKind::ModulatedRelayBasin,
        ),
    ] {
        if !enabled {
            continue;
        }
        #[cfg(test)]
        let relay_started = std::time::Instant::now();
        let scales = relay::basin_scales(
            weights,
            config.planes,
            kind == JointFitStartKind::ModulatedRelayBasin,
            config.scale_precision,
        )?;
        #[cfg(test)]
        record_solver_phase(6, relay_started);
        starts.push(optimize_start(weights, fit_metric, config, scales, kind)?);
    }

    // A lower-plane embedding is an additional basin, not one of the configured OA-EM restarts.
    // It guarantees P-monotonicity without pretending the non-convex solver is globally optimal.
    if config.planes > 1 {
        let lower = fit_joint_ternary_prepared(
            weights,
            fit_metric,
            JointFitConfig {
                planes: config.planes - 1,
                ..config
            },
            metric_diagonal,
            metric_sum,
            weighted_abs_order,
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
            receipt: Some(JointFitRestartReceipt {
                kind: JointFitStartKind::LowerPlaneFallback,
                initial_objective: lower_receipt.initial_objective,
                final_objective: lower_receipt.final_objective,
                accepted_updates: lower_receipt.accepted_updates,
                scale_solves: lower_receipt.scale_solves,
            }),
        });
    }

    let selected_start = starts
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.objective.total_cmp(&right.objective))
        .map(|(index, _)| index)
        .expect("validated positive restart count");
    // These receipts are returned once; cloning their nested vectors only to
    // drop the originals needlessly doubles per-fit allocations and copies.
    let restart_receipts = starts
        .iter_mut()
        .map(|state| {
            state
                .receipt
                .take()
                .expect("fit-state receipt is present before result assembly")
        })
        .collect();
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
    receipt: Option<JointFitRestartReceipt>,
}

#[cfg(test)]
std::thread_local! {
    static ASSIGNMENT_FOR_METRIC_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SOLVER_PHASE_NANOS: std::cell::Cell<[u128; 7]> = const { std::cell::Cell::new([0; 7]) };
}

#[cfg(test)]
fn record_solver_phase(phase: usize, started: std::time::Instant) {
    SOLVER_PHASE_NANOS.with(|elapsed| {
        let mut totals = elapsed.get();
        totals[phase] = totals[phase].saturating_add(started.elapsed().as_nanos());
        elapsed.set(totals);
    });
}

fn optimize_start(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    config: JointFitConfig,
    scales: Vec<f32>,
    kind: JointFitStartKind,
) -> Result<FitState, JointFitError> {
    #[cfg(test)]
    let assignment_started = std::time::Instant::now();
    let trits = assignment_for_metric(weights, &scales, metric)?;
    #[cfg(test)]
    record_solver_phase(0, assignment_started);

    #[cfg(test)]
    let reconstruction_started = std::time::Instant::now();
    let (reconstruction, objective) =
        reconstruct_planes_and_objective(weights, &scales, &trits, metric)?;
    #[cfg(test)]
    record_solver_phase(2, reconstruction_started);
    let mut state = FitState {
        scales,
        trits,
        reconstruction,
        objective,
        accepted_objectives: vec![objective],
        receipt: Some(JointFitRestartReceipt {
            kind,
            initial_objective: objective,
            final_objective: objective,
            accepted_updates: Vec::new(),
            scale_solves: Vec::new(),
        }),
    };

    // The initial trits came from assignment_for_metric at the initial scales.
    let mut assignment_checked_for_current_scales = true;
    for iteration in 0..config.max_iterations {
        let mut improved = false;
        #[cfg(test)]
        let scale_solve_started = std::time::Instant::now();
        let scale_outcome = solve_scales(
            weights,
            &state.trits,
            metric,
            config.ridge,
            config.ridge_condition_limit,
            config.scale_precision,
        )?;
        #[cfg(test)]
        record_solver_phase(1, scale_solve_started);

        #[cfg(test)]
        let reconstruction_started = std::time::Instant::now();
        let (scale_reconstruction, scale_objective) =
            reconstruct_planes_and_objective_with_transform(
                weights,
                &scale_outcome.scales,
                &state.trits,
                metric,
                &scale_outcome.transform,
            )?;
        #[cfg(test)]
        record_solver_phase(2, reconstruction_started);
        let scale_accepted = scale_objective < state.objective;
        state
            .receipt
            .as_mut()
            .expect("fit-state receipt is present during optimization")
            .scale_solves
            .push(ScaleSolveReceipt {
                iteration,
                telemetry: scale_outcome.telemetry,
                accepted: scale_accepted,
            });
        if scale_accepted {
            let objective_before = state.objective;
            apply_scale_solve_transform(&mut state.trits, &scale_outcome.transform);
            state.scales = scale_outcome.scales;
            state.reconstruction = scale_reconstruction;
            state.objective = scale_objective;
            state.accepted_objectives.push(scale_objective);
            state
                .receipt
                .as_mut()
                .expect("fit-state receipt is present during optimization")
                .accepted_updates
                .push(JointFitUpdateReceipt {
                    iteration,
                    phase: JointFitUpdatePhase::Scale,
                    objective_before,
                    objective_after: scale_objective,
                });
            improved = true;
            assignment_checked_for_current_scales = false;
        }

        // Re-evaluate assignments only after an accepted scale change. A rejected
        // scale candidate leaves the current scales unchanged, and the assignment
        // step for those scales was already evaluated in the prior iteration.
        if !assignment_checked_for_current_scales {
            #[cfg(test)]
            let assignment_started = std::time::Instant::now();
            let assignment = assignment_for_metric(weights, &state.scales, metric)?;
            #[cfg(test)]
            record_solver_phase(0, assignment_started);
            assignment_checked_for_current_scales = true;
            // The current reconstruction/objective already correspond to these exact
            // scales and trits. Avoid rebuilding and rescoring the row when the
            // assignment step rediscovers the same state.
            if assignment != state.trits {
                #[cfg(test)]
                let reconstruction_started = std::time::Instant::now();
                let (assignment_reconstruction, assignment_objective) =
                    reconstruct_planes_and_objective(weights, &state.scales, &assignment, metric)?;
                #[cfg(test)]
                record_solver_phase(2, reconstruction_started);
                if assignment_objective < state.objective {
                    let objective_before = state.objective;
                    state.trits = assignment;
                    state.reconstruction = assignment_reconstruction;
                    state.objective = assignment_objective;
                    state.accepted_objectives.push(assignment_objective);
                    state
                        .receipt
                        .as_mut()
                        .expect("fit-state receipt is present during optimization")
                        .accepted_updates
                        .push(JointFitUpdateReceipt {
                            iteration,
                            phase: JointFitUpdatePhase::Assignment,
                            objective_before,
                            objective_after: assignment_objective,
                        });
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    state
        .receipt
        .as_mut()
        .expect("fit-state receipt is present during optimization")
        .final_objective = state.objective;
    Ok(state)
}

fn deterministic_initial_scales(
    weights: &[f32],
    metric_diagonal: &[f64],
    metric_sum: f64,
    weighted_abs_order: &WeightedAbsOrder,
    config: JointFitConfig,
    restart: usize,
) -> Result<Vec<f32>, JointFitError> {
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
        let anchor = weighted_abs_quantile(weighted_abs_order, quantile);
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

type WeightedAbsEntry = (f32, f64, usize);

#[derive(Default)]
struct WeightedAbsOrder {
    entries: Vec<WeightedAbsEntry>,
    total_weight: f64,
}

fn weighted_abs_order(weights: &[f32], metric_diagonal: &[f64]) -> WeightedAbsOrder {
    let mut values: Vec<WeightedAbsEntry> = weights
        .iter()
        .zip(metric_diagonal)
        .enumerate()
        .map(|(index, (value, weight))| (value.abs(), *weight, index))
        .collect();
    // The original index is a unique tiebreaker, so this total order is identical
    // to stable sorting while avoiding a temporary allocation for every fitted row.
    values.sort_unstable_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.2.cmp(&right.2))
    });
    // The scale-start schedule asks several weighted quantiles from this same
    // order. Cache the exact sorted-order total so every restart avoids
    // re-summing the same group; keep each prefix scan allocation-free.
    let total_weight = values.iter().map(|value| value.1).sum();
    WeightedAbsOrder {
        entries: values,
        total_weight,
    }
}

fn weighted_abs_quantile(order: &WeightedAbsOrder, quantile: f64) -> f64 {
    let target = order.total_weight * quantile.clamp(0.0, 1.0);
    let mut cumulative = 0.0;
    for (value, weight, _) in &order.entries {
        cumulative += weight;
        if cumulative >= target {
            return f64::from(*value);
        }
    }
    order.entries.last().map_or(0.0, |value| f64::from(value.0))
}

#[derive(Clone, Debug)]
struct ScaleSolveOutcome {
    scales: Vec<f32>,
    transform: ScaleSolveTransform,
    telemetry: ScaleSolveTelemetry,
}

#[derive(Clone, Copy, Debug)]
struct ScaleSolveTransform {
    trit_order: [usize; 3],
    trit_signs: [i8; 3],
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
    match metric {
        JointFitMetric::Identity => {
            for plane in 0..planes {
                for other in 0..planes {
                    normal[plane][other] = trits[plane]
                        .iter()
                        .zip(&trits[other])
                        .map(|(left, right)| f64::from(*left) * f64::from(*right))
                        .sum();
                }
                rhs[plane] = trits[plane]
                    .iter()
                    .zip(weights)
                    .map(|(left, weight)| f64::from(*left) * f64::from(*weight))
                    .sum();
            }
        }
        JointFitMetric::Diagonal(diagonal) => {
            for plane in 0..planes {
                for other in 0..planes {
                    normal[plane][other] = trits[plane]
                        .iter()
                        .zip(&trits[other])
                        .zip(diagonal)
                        .map(|((left, right), weight)| {
                            f64::from(*left) * f64::from(*right) * f64::from(*weight)
                        })
                        .sum();
                }
                rhs[plane] = trits[plane]
                    .iter()
                    .zip(weights)
                    .zip(diagonal)
                    .map(|((left, weight), curvature)| {
                        f64::from(*left) * f64::from(*weight) * f64::from(*curvature)
                    })
                    .sum();
            }
        }
        JointFitMetric::DiagonalF64(diagonal) => {
            // Each normal-matrix entry and rhs component must still accumulate in row order,
            // but all of them can share one pass over the weights and curvature. The previous
            // plane-major implementation traversed this same data P² + P times per scale solve.
            for index in 0..weights.len() {
                let curvature = diagonal[index];
                let weight = f64::from(weights[index]);
                for plane in 0..planes {
                    let left = f64::from(trits[plane][index]);
                    rhs[plane] += left * weight * curvature;
                    // The normal matrix is a weighted Gram matrix. Accumulate
                    // only one triangle, then mirror it; each mirrored entry
                    // sees the same row order and the same commutative trit
                    // product as the former full-matrix loop.
                    for other in plane..planes {
                        normal[plane][other] += left * f64::from(trits[other][index]) * curvature;
                    }
                }
            }
            for plane in 0..planes {
                let (previous_rows, current_and_after) = normal.split_at_mut(plane);
                let current_row = &mut current_and_after[0];
                for (lower, previous_row) in current_row[..plane].iter_mut().zip(previous_rows) {
                    *lower = previous_row[plane];
                }
            }
        }
        JointFitMetric::DiagonalAffine {
            values: diagonal,
            scale,
            shift,
        } => {
            for plane in 0..planes {
                for other in 0..planes {
                    normal[plane][other] = trits[plane]
                        .iter()
                        .zip(&trits[other])
                        .zip(diagonal)
                        .map(|((left, right), weight)| {
                            f64::from(*left) * f64::from(*right) * (*weight * scale + shift)
                        })
                        .sum();
                }
                rhs[plane] = trits[plane]
                    .iter()
                    .zip(weights)
                    .zip(diagonal)
                    .map(|((left, weight), curvature)| {
                        f64::from(*left) * f64::from(*weight) * (*curvature * scale + shift)
                    })
                    .sum();
            }
        }
        JointFitMetric::Dense(dense) => {
            for plane in 0..planes {
                for other in 0..planes {
                    for row in 0..weights.len() {
                        let left = f64::from(trits[plane][row]);
                        if left == 0.0 {
                            continue;
                        }
                        for (col, &other_trit) in trits[other].iter().enumerate() {
                            normal[plane][other] += left
                                * dense.values[row * dense.dimension + col]
                                * f64::from(other_trit);
                        }
                    }
                }
                for (row, plane_trit) in trits[plane].iter().enumerate() {
                    let left = f64::from(*plane_trit);
                    if left == 0.0 {
                        continue;
                    }
                    for (col, &weight) in weights.iter().enumerate() {
                        rhs[plane] +=
                            left * dense.values[row * dense.dimension + col] * f64::from(weight);
                    }
                }
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

    // Plane signs are a representation symmetry. Record the sign and permutation
    // rather than cloning each trit vector; the caller applies the transform only
    // if this scale candidate is accepted.
    let mut signed_planes = Vec::with_capacity(planes);
    for (source, scale) in rhs[..planes].iter().enumerate() {
        let mut scale = *scale;
        let mut sign = 1_i8;
        if scale < 0.0 {
            scale = -scale;
            sign = -1;
        }
        signed_planes.push((scale, source, sign));
    }
    signed_planes.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut scales = Vec::with_capacity(planes);
    let mut trit_order = [0_usize; 3];
    let mut trit_signs = [1_i8; 3];
    for (plane, (scale, source, sign)) in signed_planes.into_iter().enumerate() {
        scales.push(deployment_scale(scale as f32, precision, plane)?);
        trit_order[plane] = source;
        trit_signs[plane] = sign;
    }
    Ok(ScaleSolveOutcome {
        scales,
        transform: ScaleSolveTransform {
            trit_order,
            trit_signs,
        },
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
    #[cfg(test)]
    ASSIGNMENT_FOR_METRIC_CALLS.with(|calls| calls.set(calls.get() + 1));

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

/// Reconstruct additive planes and evaluate diagonal objectives in one ordered pass.
///
/// The reconstructed f32 values are accumulated in exactly the same plane order as
/// [`reconstruct_planes`]. For diagonal metrics, this avoids writing then rereading
/// the full reconstruction solely to compute the objective. Dense metrics retain the
/// existing quadratic evaluator because their objective depends on off-diagonal terms.
fn reconstruct_planes_and_objective(
    weights: &[f32],
    scales: &[f32],
    trits: &[Vec<i8>],
    metric: JointFitMetric<'_>,
) -> Result<(Vec<f32>, f64), JointFitError> {
    if matches!(metric, JointFitMetric::Dense(_)) {
        let reconstruction = reconstruct_planes(scales, trits, weights.len());
        let objective = metric_objective(weights, &reconstruction, metric)?;
        return Ok((reconstruction, objective));
    }

    let mut reconstruction = vec![0.0_f32; weights.len()];
    let mut objective = 0.0_f64;
    for index in 0..weights.len() {
        let mut fitted = 0.0_f32;
        for (scale, plane) in scales.iter().zip(trits) {
            fitted += *scale * f32::from(plane[index]);
        }
        reconstruction[index] = fitted;
        let error = f64::from(weights[index]) - f64::from(fitted);
        let curvature = match metric {
            JointFitMetric::Identity => 1.0,
            JointFitMetric::Diagonal(values) => f64::from(values[index]),
            JointFitMetric::DiagonalF64(values) => values[index],
            JointFitMetric::DiagonalAffine {
                values,
                scale,
                shift,
            } => values[index] * scale + shift,
            JointFitMetric::Dense(_) => unreachable!("dense metric returned above"),
        };
        accumulate_objective_term(&mut objective, error * error * curvature)?;
    }
    Ok((reconstruction, objective.max(0.0)))
}

fn reconstruct_planes_and_objective_with_transform(
    weights: &[f32],
    scales: &[f32],
    trits: &[Vec<i8>],
    metric: JointFitMetric<'_>,
    transform: &ScaleSolveTransform,
) -> Result<(Vec<f32>, f64), JointFitError> {
    if matches!(metric, JointFitMetric::Dense(_)) {
        let mut reconstruction = vec![0.0_f32; weights.len()];
        for index in 0..weights.len() {
            for (plane, scale) in scales.iter().enumerate() {
                let source = transform.trit_order[plane];
                let trit = trits[source][index] * transform.trit_signs[plane];
                reconstruction[index] += *scale * f32::from(trit);
            }
        }
        let objective = metric_objective(weights, &reconstruction, metric)?;
        return Ok((reconstruction, objective));
    }

    let mut reconstruction = vec![0.0_f32; weights.len()];
    let mut objective = 0.0_f64;
    for index in 0..weights.len() {
        let mut fitted = 0.0_f32;
        for (plane, scale) in scales.iter().enumerate() {
            let source = transform.trit_order[plane];
            let trit = trits[source][index] * transform.trit_signs[plane];
            fitted += *scale * f32::from(trit);
        }
        reconstruction[index] = fitted;
        let error = f64::from(weights[index]) - f64::from(fitted);
        let curvature = match metric {
            JointFitMetric::Identity => 1.0,
            JointFitMetric::Diagonal(values) => f64::from(values[index]),
            JointFitMetric::DiagonalF64(values) => values[index],
            JointFitMetric::DiagonalAffine {
                values,
                scale,
                shift,
            } => values[index] * scale + shift,
            JointFitMetric::Dense(_) => unreachable!("dense metric returned above"),
        };
        accumulate_objective_term(&mut objective, error * error * curvature)?;
    }
    Ok((reconstruction, objective.max(0.0)))
}

fn apply_scale_solve_transform(trits: &mut Vec<Vec<i8>>, transform: &ScaleSolveTransform) {
    let planes = trits.len();
    let mut sources: [Option<Vec<i8>>; 3] = std::array::from_fn(|_| None);
    for (slot, plane) in sources.iter_mut().zip(std::mem::take(trits)) {
        *slot = Some(plane);
    }
    let mut ordered = Vec::with_capacity(planes);
    for plane in 0..planes {
        let mut values = sources[transform.trit_order[plane]]
            .take()
            .expect("scale-solve permutation selects each source plane once");
        if transform.trit_signs[plane] < 0 {
            for trit in &mut values {
                *trit = -*trit;
            }
        }
        ordered.push(values);
    }
    *trits = ordered;
}

fn metric_objective(
    weights: &[f32],
    reconstruction: &[f32],
    metric: JointFitMetric<'_>,
) -> Result<f64, JointFitError> {
    let mut objective = 0.0_f64;
    match metric {
        JointFitMetric::Identity => {
            for (&weight, &fitted) in weights.iter().zip(reconstruction) {
                let value = f64::from(weight) - f64::from(fitted);
                accumulate_objective_term(&mut objective, value * value)?;
            }
        }
        JointFitMetric::Diagonal(diagonal) => {
            for ((&weight, &fitted), curvature) in weights.iter().zip(reconstruction).zip(diagonal)
            {
                let value = f64::from(weight) - f64::from(fitted);
                let squared = value * value;
                accumulate_objective_term(&mut objective, squared * f64::from(*curvature))?;
            }
        }
        JointFitMetric::DiagonalF64(diagonal) => {
            for ((&weight, &fitted), curvature) in weights.iter().zip(reconstruction).zip(diagonal)
            {
                let value = f64::from(weight) - f64::from(fitted);
                let squared = value * value;
                accumulate_objective_term(&mut objective, squared * *curvature)?;
            }
        }
        JointFitMetric::DiagonalAffine {
            values: diagonal,
            scale,
            shift,
        } => {
            for ((&weight, &fitted), curvature) in weights.iter().zip(reconstruction).zip(diagonal)
            {
                let value = f64::from(weight) - f64::from(fitted);
                let squared = value * value;
                accumulate_objective_term(&mut objective, squared * (*curvature * scale + shift))?;
            }
        }
        JointFitMetric::Dense(dense) => {
            let error: Vec<f64> = weights
                .iter()
                .zip(reconstruction)
                .map(|(weight, fitted)| f64::from(*weight) - f64::from(*fitted))
                .collect();
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

/// CAT-Q softened two-sided-relay initialization basins (arXiv 2606.26650).
///
/// Each basin soft-fits sequential residual planes with the smooth relay surrogate and hands the
/// exact E/M solver only the fitted per-plane scale magnitudes as one extra restart candidate.
/// The surrogate's normalized threshold `delta` and (modulated variant) shift `mu` are
/// basin-internal: they shape which scales come out and are neither stored nor returned, so the
/// emitted representation stays pure scales-and-trits and the ADR 0028 zero-point ban holds.
///
/// The soft objective is plain squared error in absmean-normalized units — the basin is only an
/// initializer; the exact solver applies the configured curvature metric. Everything is
/// bit-repeatable: no RNG, fixed iteration counts, fixed step size, and a fixed sharpness
/// schedule.
mod relay {
    use super::{JointFitError, ScalePrecision, deployment_scale};

    /// Fixed number of analytic gradient-descent steps per plane.
    const STEPS: usize = 12;
    /// Fixed descent step size applied to mean gradients in normalized units.
    const STEP_SIZE: f64 = 0.05;
    /// Initial relay sharpness `s0`. The schedule doubles it every four steps:
    /// steps 0-3 use 30, steps 4-7 use 60, steps 8-11 use 120.
    const INITIAL_SHARPNESS: f64 = 30.0;
    /// Normalized scale bounds. The residual absmean is 1 by construction, so these bound the
    /// fitted scale to a sane multiple of it and keep `u = c / scale` finite.
    const SCALE_BOUNDS: (f64, f64) = (1e-3, 8.0);
    /// Normalized threshold bounds inside the open interval `(0, 1)`.
    const THRESHOLD_BOUNDS: (f64, f64) = (0.05, 0.95);
    /// Normalized shift bounds for the modulated variant.
    const SHIFT_BOUNDS: (f64, f64) = (-2.0, 2.0);
    /// `tanh(±20)` rounds exactly to `±1` in binary64, so the shortcut preserves bits.
    const TANH_SATURATION: f64 = 20.0;

    /// Deployment-signature wrapper over the f64 relay core, exercised by the property tests;
    /// the descent evaluates the core directly.
    #[cfg(test)]
    pub(super) fn two_sided_relay(v: f32, sharpness: f32, delta: f32) -> f32 {
        relay(f64::from(v), f64::from(sharpness), f64::from(delta)) as f32
    }

    /// CAT-Q softened ternarization
    /// `(tanh(s * (v - delta)) + tanh(s * (v + delta))) / (2 * tanh(s))`.
    ///
    /// Odd-symmetric with `f(0) = 0`, bounded by `|f| <= 1` on `|v| <= 1` for
    /// `delta` in `[0, 1]`, and approaching the hard ternary indicator with
    /// threshold `delta` as `sharpness -> inf`.
    #[cfg(test)]
    fn relay(v: f64, sharpness: f64, delta: f64) -> f64 {
        (((v - delta) * sharpness).tanh() + ((v + delta) * sharpness).tanh())
            / (2.0 * sharpness.tanh())
    }

    // `1 - tanh^2` saturates to exactly zero for large inputs instead of overflowing like
    // `cosh`-based forms, which keeps every descent step finite.
    #[cfg(test)]
    fn sech_squared(x: f64) -> f64 {
        let tanh = x.tanh();
        1.0 - tanh * tanh
    }

    const fn sharpness_at(step: usize) -> f64 {
        INITIAL_SHARPNESS * (1 << (step / 4)) as f64
    }

    /// Fitted soft-plane parameters in absmean-normalized units.
    struct PlaneFit {
        scale: f64,
        threshold: f64,
        shift: f64,
    }

    /// Sequential per-plane residual soft fit returning `planes` deployment-rounded scale
    /// magnitudes, canonicalized descending like every other deterministic basin.
    ///
    /// After each plane the HARD projection of the soft fit — the exact ternary assignment of
    /// the shift-centered residual at the fitted scale and threshold — is subtracted, so only
    /// the `scale * trit` contribution ever leaves the basin.
    pub(super) fn basin_scales(
        weights: &[f32],
        planes: usize,
        modulated: bool,
        precision: ScalePrecision,
    ) -> Result<Vec<f32>, JointFitError> {
        let mut residual = weights.to_vec();
        let mut scales = Vec::with_capacity(planes);
        // Every plane's descent needs the residual normalized by that plane's
        // absmean. Reuse one buffer instead of allocating a new f64 vector for
        // each of the three deterministic relay basins.
        let mut normalized = Vec::with_capacity(residual.len());
        for plane in 0..planes {
            let absmean = residual
                .iter()
                .map(|value| f64::from(value.abs()))
                .sum::<f64>()
                / residual.len() as f64;
            if absmean <= 0.0 {
                scales.push(deployment_scale(0.0, precision, plane)?);
                continue;
            }
            normalized.clear();
            normalized.extend(residual.iter().map(|value| f64::from(*value) / absmean));
            let fit = descend(&normalized, modulated);
            let scale = deployment_scale((fit.scale * absmean) as f32, precision, plane)?;
            scales.push(scale);
            if scale > 0.0 {
                let shift = (fit.shift * absmean) as f32;
                let threshold = scale * fit.threshold as f32;
                for value in &mut residual {
                    let centered = *value - shift;
                    let trit = if centered > threshold {
                        1.0
                    } else if centered < -threshold {
                        -1.0
                    } else {
                        0.0
                    };
                    *value -= scale * trit;
                }
            }
        }
        scales.sort_by(|left, right| right.total_cmp(left));
        Ok(scales)
    }

    /// Minimize `L = mean_i (c_i - a * relay(c_i / a, s_k, delta))^2` with `c_i = w_i - mu` by
    /// `STEPS` analytic gradient steps of fixed size `STEP_SIZE` under the documented sharpness
    /// schedule. The softened variant descends only `a`; the modulated variant also descends
    /// `delta` and `mu`. Every parameter is clamped to its bounds after each step.
    fn descend(normalized: &[f64], modulated: bool) -> PlaneFit {
        let count = normalized.len() as f64;
        let mut scale = 1.0_f64;
        let mut threshold = 0.5_f64;
        let mut shift = if modulated {
            (normalized.iter().sum::<f64>() / count).clamp(SHIFT_BOUNDS.0, SHIFT_BOUNDS.1)
        } else {
            0.0
        };
        for step in 0..STEPS {
            let sharpness = sharpness_at(step);
            // Every scheduled sharpness is >= 30, where binary64 tanh is exactly one.
            let norm = 2.0;
            let mut grad_scale = 0.0_f64;
            let mut grad_threshold = 0.0_f64;
            let mut grad_shift = 0.0_f64;
            for &value in normalized {
                let centered = value - shift;
                let u = centered / scale;
                let lower = (u - threshold) * sharpness;
                let upper = (u + threshold) * sharpness;
                let tanh_lower = relay_tanh(lower);
                let tanh_upper = relay_tanh(upper);
                let soft = (tanh_lower + tanh_upper) / norm;
                let soft_du = sharpness
                    * ((1.0 - tanh_lower * tanh_lower) + (1.0 - tanh_upper * tanh_upper))
                    / norm;
                let soft_dthreshold = sharpness
                    * ((1.0 - tanh_upper * tanh_upper) - (1.0 - tanh_lower * tanh_lower))
                    / norm;
                let error = centered - scale * soft;
                grad_scale += error * (u * soft_du - soft);
                grad_threshold -= error * scale * soft_dthreshold;
                grad_shift += error * (soft_du - 1.0);
            }
            let step_factor = 2.0 * STEP_SIZE / count;
            scale = (scale - step_factor * grad_scale).clamp(SCALE_BOUNDS.0, SCALE_BOUNDS.1);
            if modulated {
                threshold = (threshold - step_factor * grad_threshold)
                    .clamp(THRESHOLD_BOUNDS.0, THRESHOLD_BOUNDS.1);
                shift = (shift - step_factor * grad_shift).clamp(SHIFT_BOUNDS.0, SHIFT_BOUNDS.1);
            }
        }
        PlaneFit {
            scale,
            threshold,
            shift,
        }
    }

    #[inline]
    fn relay_tanh(value: f64) -> f64 {
        if value >= TANH_SATURATION {
            1.0
        } else if value <= -TANH_SATURATION {
            -1.0
        } else {
            value.tanh()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn saturation_shortcut_matches_binary64_tanh_at_boundary_and_extremes() {
            let positive_boundary = TANH_SATURATION.to_bits();
            let negative_boundary = (-TANH_SATURATION).to_bits();
            let values = [
                f64::from_bits(positive_boundary - 1),
                TANH_SATURATION,
                f64::from_bits(positive_boundary + 1),
                f64::from_bits(negative_boundary - 1),
                -TANH_SATURATION,
                f64::from_bits(negative_boundary + 1),
                -100.0,
                100.0,
            ];
            for value in values {
                assert_eq!(
                    relay_tanh(value).to_bits(),
                    value.tanh().to_bits(),
                    "{value}"
                );
            }
        }

        fn descend_reference(normalized: &[f64], modulated: bool) -> PlaneFit {
            let count = normalized.len() as f64;
            let mut scale = 1.0_f64;
            let mut threshold = 0.5_f64;
            let mut shift = if modulated {
                (normalized.iter().sum::<f64>() / count).clamp(SHIFT_BOUNDS.0, SHIFT_BOUNDS.1)
            } else {
                0.0
            };
            for step in 0..STEPS {
                let sharpness = sharpness_at(step);
                let norm = 2.0 * sharpness.tanh();
                let mut grad_scale = 0.0_f64;
                let mut grad_threshold = 0.0_f64;
                let mut grad_shift = 0.0_f64;
                for &value in normalized {
                    let centered = value - shift;
                    let u = centered / scale;
                    let lower = (u - threshold) * sharpness;
                    let upper = (u + threshold) * sharpness;
                    let soft = relay(u, sharpness, threshold);
                    let soft_du = sharpness * (sech_squared(lower) + sech_squared(upper)) / norm;
                    let soft_dthreshold =
                        sharpness * (sech_squared(upper) - sech_squared(lower)) / norm;
                    let error = centered - scale * soft;
                    grad_scale += error * (u * soft_du - soft);
                    grad_threshold -= error * scale * soft_dthreshold;
                    grad_shift += error * (soft_du - 1.0);
                }
                let step_factor = 2.0 * STEP_SIZE / count;
                scale = (scale - step_factor * grad_scale).clamp(SCALE_BOUNDS.0, SCALE_BOUNDS.1);
                if modulated {
                    threshold = (threshold - step_factor * grad_threshold)
                        .clamp(THRESHOLD_BOUNDS.0, THRESHOLD_BOUNDS.1);
                    shift =
                        (shift - step_factor * grad_shift).clamp(SHIFT_BOUNDS.0, SHIFT_BOUNDS.1);
                }
            }
            PlaneFit {
                scale,
                threshold,
                shift,
            }
        }

        #[test]
        fn fused_relay_descent_matches_reference_bits() {
            for length in [1, 3, 16, 64, 128] {
                for seed in 0..8 {
                    let normalized = (0..length)
                        .map(|index| {
                            let value = (index * 37 + seed * 19) % 101;
                            (value as f64 - 50.0) / 13.0
                        })
                        .collect::<Vec<_>>();
                    for modulated in [false, true] {
                        let expected = descend_reference(&normalized, modulated);
                        let actual = descend(&normalized, modulated);
                        assert_eq!(actual.scale.to_bits(), expected.scale.to_bits());
                        assert_eq!(actual.threshold.to_bits(), expected.threshold.to_bits());
                        assert_eq!(actual.shift.to_bits(), expected.shift.to_bits());
                    }
                }
            }
        }
    }
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

    #[test]
    fn fused_reconstruction_objective_is_bit_identical_to_reference_paths() {
        let weights = [0.75, -0.5, 0.25, -1.25];
        let scales = [0.625, 0.1875];
        let trits = [vec![1, -1, 0, 1], vec![-1, 0, 1, -1]];
        let diagonal_f32 = [0.5, 1.25, 2.0, 0.75];
        let diagonal_f64 = [0.5, 1.25, 2.0, 0.75];

        let assert_same = |metric| {
            let expected_reconstruction = reconstruct_planes(&scales, &trits, weights.len());
            let expected_objective =
                metric_objective(&weights, &expected_reconstruction, metric).unwrap();
            let (actual_reconstruction, actual_objective) =
                reconstruct_planes_and_objective(&weights, &scales, &trits, metric).unwrap();
            assert_eq!(actual_reconstruction, expected_reconstruction);
            assert_eq!(actual_objective.to_bits(), expected_objective.to_bits());
        };

        assert_same(JointFitMetric::Identity);
        assert_same(JointFitMetric::Diagonal(&diagonal_f32));
        assert_same(JointFitMetric::DiagonalF64(&diagonal_f64));
        assert_same(JointFitMetric::DiagonalAffine {
            values: &diagonal_f64,
            scale: 0.75,
            shift: 0.125,
        });
        let dense = DensePsdMetric::new(
            weights.len(),
            &[
                2.0, 0.25, 0.0, 0.0, 0.25, 1.5, 0.0, 0.0, 0.0, 0.0, 0.75, 0.125, 0.0, 0.0, 0.125,
                1.0,
            ],
        )
        .unwrap();
        assert_same(JointFitMetric::Dense(&dense));
    }

    #[test]
    fn cached_weighted_abs_order_preserves_quantile_bits() {
        let weights = [0.0, -2.0, 0.5, 1.5, -2.0, 0.25, 8.0, -0.75];
        let diagonal = [3.0, 0.25, 1.0, 4.0, 2.0, 0.5, 0.0, 7.0];
        let cached = weighted_abs_order(&weights, &diagonal);
        for quantile in [0.0_f64, 0.25, 0.5, 0.75, 0.95, 1.0] {
            let mut reference: Vec<WeightedAbsEntry> = weights
                .iter()
                .zip(&diagonal)
                .enumerate()
                .map(|(index, (value, weight))| (value.abs(), *weight, index))
                .collect();
            reference.sort_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.2.cmp(&right.2))
            });
            assert_eq!(
                cached.entries, reference,
                "quantile ordering must remain canonical"
            );
            let total: f64 = reference.iter().map(|value| value.1).sum();
            assert_eq!(cached.total_weight.to_bits(), total.to_bits());
            let target = total * quantile.clamp(0.0, 1.0);
            let mut cumulative = 0.0;
            let expected = reference
                .iter()
                .find_map(|value| {
                    cumulative += value.1;
                    (cumulative >= target).then_some(f64::from(value.0))
                })
                .or_else(|| reference.last().map(|value| f64::from(value.0)))
                .unwrap_or(0.0);
            assert_eq!(
                weighted_abs_quantile(&cached, quantile).to_bits(),
                expected.to_bits(),
                "quantile {quantile}"
            );
        }
    }

    #[test]
    #[ignore = "manual G64/P3 solver phase profile"]
    fn profile_g64_p3_solver_phases() {
        let rows: Vec<Vec<f32>> = (0..256)
            .map(|row| {
                (0..64)
                    .map(|column| {
                        let value = (row * 64 + column) * 37 % 101;
                        (value as f32 - 50.0) / 37.0
                    })
                    .collect()
            })
            .collect();
        let diagonal: Vec<f64> = (0..64)
            .map(|column| 0.25 + ((column * 17 % 31) as f64 / 31.0))
            .collect();
        let config = JointFitConfig {
            planes: 3,
            max_iterations: 16,
            ridge: 1e-8,
            em_restarts: 4,
            ridge_condition_limit: 1e6,
            scale_precision: ScalePrecision::F16,
            relay_basins: RelayBasins {
                softened: true,
                modulated: true,
            },
        };

        SOLVER_PHASE_NANOS.with(|elapsed| elapsed.set([0; 7]));
        let started = std::time::Instant::now();
        for weights in &rows {
            fit_joint_ternary(weights, JointFitMetric::DiagonalF64(&diagonal), config)
                .expect("profile G64/P3 row fit");
        }
        let total = started.elapsed().as_nanos();
        let phases = SOLVER_PHASE_NANOS.with(std::cell::Cell::get);
        assert!(phases.iter().sum::<u128>() <= total);
        eprintln!(
            "G64/P3 256-row profile: total={:.3}ms assignment={:.3}ms scale_solve={:.3}ms reconstruction={:.3}ms validate_metric={:.3}ms weighted_order={:.3}ms init_scales={:.3}ms relay_scales={:.3}ms other={:.3}ms",
            total as f64 / 1_000_000.0,
            phases[0] as f64 / 1_000_000.0,
            phases[1] as f64 / 1_000_000.0,
            phases[2] as f64 / 1_000_000.0,
            phases[3] as f64 / 1_000_000.0,
            phases[4] as f64 / 1_000_000.0,
            phases[5] as f64 / 1_000_000.0,
            phases[6] as f64 / 1_000_000.0,
            (total - phases.iter().sum::<u128>()) as f64 / 1_000_000.0,
        );
    }

    #[test]
    #[ignore = "manual G64/P2 compact PTQ solver phase profile"]
    fn profile_g64_p2_compact_solver_phases() {
        let rows: Vec<Vec<f32>> = (0..256)
            .map(|row| {
                (0..64)
                    .map(|column| {
                        let value = (row * 64 + column) * 37 % 101;
                        (value as f32 - 50.0) / 37.0
                    })
                    .collect()
            })
            .collect();
        let diagonal: Vec<f64> = (0..64)
            .map(|column| 0.25 + ((column * 17 % 31) as f64 / 31.0))
            .collect();
        report_compact_g64_p2_phase_profile(
            &rows,
            &diagonal,
            "synthetic relay-on",
            RelayBasins {
                softened: true,
                modulated: true,
            },
        );
    }

    #[test]
    #[ignore = "requires fixture from scripts/profile-smollm2-ptq-solver.py"]
    fn profile_smollm2_g64_p2_solver_phases() {
        const HEADER_BYTES: usize = 16;
        let fixture_path = std::env::var_os("TRITIUM_SMOLLM2_PROFILE_FIXTURE")
            .expect("set TRITIUM_SMOLLM2_PROFILE_FIXTURE to a generated fixture path");
        let bytes = std::fs::read(fixture_path).expect("read SmolLM2 profile fixture");
        assert!(
            bytes.len() >= HEADER_BYTES,
            "fixture is shorter than its header"
        );
        assert_eq!(&bytes[..8], b"TRIPRF01", "unknown fixture format");
        let rows = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let columns = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(rows, 256, "fixture row count must be 256");
        assert_eq!(columns, 64, "fixture group width must be 64");
        let weight_bytes = rows.checked_mul(columns).unwrap().checked_mul(4).unwrap();
        let expected_bytes = HEADER_BYTES + weight_bytes + columns * 8;
        assert_eq!(
            bytes.len(),
            expected_bytes,
            "fixture has an invalid byte length"
        );
        let weight_data = &bytes[HEADER_BYTES..HEADER_BYTES + weight_bytes];
        let weights = weight_data
            .chunks(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("validated f32 chunk")))
            .collect::<Vec<_>>()
            .chunks_exact(columns)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        let diagonal_data = &bytes[HEADER_BYTES + weight_bytes..];
        let diagonal = diagonal_data
            .chunks(8)
            .map(|value| f64::from_le_bytes(value.try_into().expect("validated f64 chunk")))
            .collect::<Vec<_>>();
        let baseline_objective = report_compact_g64_p2_phase_profile(
            &weights,
            &diagonal,
            "pinned SmolLM2 relay-off",
            RelayBasins::default(),
        );
        let softened_objective = report_compact_g64_p2_phase_profile(
            &weights,
            &diagonal,
            "pinned SmolLM2 softened-only",
            RelayBasins {
                softened: true,
                modulated: false,
            },
        );
        let modulated_objective = report_compact_g64_p2_phase_profile(
            &weights,
            &diagonal,
            "pinned SmolLM2 modulated-only",
            RelayBasins {
                softened: false,
                modulated: true,
            },
        );
        let dual_objective = report_compact_g64_p2_phase_profile(
            &weights,
            &diagonal,
            "pinned SmolLM2 dual-relay",
            RelayBasins {
                softened: true,
                modulated: true,
            },
        );
        assert!(softened_objective <= baseline_objective);
        assert!(modulated_objective <= baseline_objective);
        assert!(dual_objective <= softened_objective);
        assert!(dual_objective <= modulated_objective);
    }

    fn report_compact_g64_p2_phase_profile(
        rows: &[Vec<f32>],
        diagonal: &[f64],
        fixture_kind: &str,
        relay_basins: RelayBasins,
    ) -> f64 {
        assert_eq!(rows.len(), 256);
        assert!(rows.iter().all(|row| row.len() == 64));
        assert_eq!(diagonal.len(), 64);
        let config = JointFitConfig {
            planes: 2,
            max_iterations: 16,
            ridge: 1e-8,
            em_restarts: 4,
            ridge_condition_limit: 1e6,
            scale_precision: ScalePrecision::F16,
            relay_basins,
        };

        const REPEATS: usize = 5;
        SOLVER_PHASE_NANOS.with(|elapsed| elapsed.set([0; 7]));
        let started = std::time::Instant::now();
        let mut objective_sum = 0.0;
        for _ in 0..REPEATS {
            for row in rows {
                let fit = fit_joint_ternary(row, JointFitMetric::DiagonalF64(diagonal), config)
                    .expect("profile compact G64/P2 row fit");
                objective_sum += fit.objective;
            }
        }
        let total = started.elapsed().as_nanos();
        let phases = SOLVER_PHASE_NANOS.with(std::cell::Cell::get);
        assert!(phases.iter().sum::<u128>() <= total);
        let divisor = REPEATS as f64 * 1_000_000.0;
        eprintln!(
            "G64/P2 compact {fixture_kind} 256-row profile over {REPEATS} repeats: total={:.3}ms assignment={:.3}ms scale_solve={:.3}ms reconstruction={:.3}ms validate_metric={:.3}ms weighted_order={:.3}ms init_scales={:.3}ms relay_scales={:.3}ms other={:.3}ms objective_sum={:.9}",
            total as f64 / divisor,
            phases[0] as f64 / divisor,
            phases[1] as f64 / divisor,
            phases[2] as f64 / divisor,
            phases[3] as f64 / divisor,
            phases[4] as f64 / divisor,
            phases[5] as f64 / divisor,
            phases[6] as f64 / divisor,
            (total - phases.iter().sum::<u128>()) as f64 / divisor,
            objective_sum / REPEATS as f64,
        );
        objective_sum / REPEATS as f64
    }

    #[test]
    fn cached_start_context_preserves_existing_three_plane_output() {
        let weights: Vec<f32> = (0..128)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 37.0)
            .collect();
        let diagonal = [1.0_f64; 128];
        let fit = fit_joint_ternary(
            &weights,
            JointFitMetric::DiagonalF64(&diagonal),
            JointFitConfig {
                planes: 3,
                max_iterations: 16,
                ridge: 1e-8,
                em_restarts: 4,
                ridge_condition_limit: 1e6,
                scale_precision: ScalePrecision::F16,
                relay_basins: RelayBasins {
                    softened: true,
                    modulated: true,
                },
            },
        )
        .expect("three-plane row fit");
        let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
        for &scale in &fit.scales {
            for byte in scale.to_bits().to_le_bytes() {
                fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        }
        for plane in &fit.trits {
            for &trit in plane {
                fingerprint = (fingerprint ^ u64::from(trit as u8)).wrapping_mul(0x100_0000_01b3);
            }
        }
        assert_eq!(fingerprint, 0xd20d_9b32_8141_ebad);
    }

    #[test]
    fn exact_diagonal_metric_fast_path_matches_dense_fit() {
        let dense = DensePsdMetric::new(
            4,
            &[
                2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 7.0,
            ],
        )
        .expect("diagonal metric");
        let diagonal = dense.exact_diagonal().expect("exact diagonal");
        let weights = [0.9, -0.6, 0.25, -0.15];
        let config = JointFitConfig {
            planes: 3,
            max_iterations: 4,
            em_restarts: 3,
            ..JointFitConfig::default()
        };
        let dense_fit =
            fit_joint_ternary(&weights, JointFitMetric::Dense(&dense), config).expect("dense fit");
        let diagonal_fit =
            fit_joint_ternary(&weights, JointFitMetric::DiagonalF64(diagonal), config)
                .expect("diagonal fit");
        let affine_fit = fit_joint_ternary(
            &weights,
            JointFitMetric::DiagonalAffine {
                values: diagonal,
                scale: 1.0,
                shift: 0.0,
            },
            config,
        )
        .expect("affine diagonal fit");
        assert_eq!(dense_fit.scales, diagonal_fit.scales);
        assert_eq!(dense_fit.trits, diagonal_fit.trits);
        assert_eq!(dense_fit.reconstruction, diagonal_fit.reconstruction);
        assert_eq!(dense_fit.objective, diagonal_fit.objective);
        assert_eq!(dense_fit.scales, affine_fit.scales);
        assert_eq!(dense_fit.trits, affine_fit.trits);
        assert_eq!(dense_fit.reconstruction, affine_fit.reconstruction);
        assert_eq!(dense_fit.objective, affine_fit.objective);
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
    fn exact_assignment_preserves_exhaustive_codes_and_tie_order() {
        fn reference(weights: &[f32], scales: &[f32]) -> Vec<Vec<i8>> {
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
            trits
        }

        let weights = [-3.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, f32::MAX];
        let cases: &[&[f32]] = &[
            &[1.0],
            &[1.0, 0.5],
            &[1.0, 1.0],
            &[1.0, 0.5, 0.25],
            &[0.0, 0.0, 0.0],
            &[f32::MAX, f32::MAX, f32::MAX],
            &[1.0e30, 1.0e30, f32::from_bits(1)],
        ];
        for scales in cases {
            assert_eq!(
                exact_ternary_assignment(&weights, scales).expect("valid assignment"),
                reference(&weights, scales),
                "scale set {scales:?}",
            );
        }
    }

    #[test]
    fn exact_assignment_matches_exhaustive_oracle_around_every_codebook_midpoint() {
        fn reference(weights: &[f32], scales: &[f32]) -> Vec<Vec<i8>> {
            const CODES: [i8; 3] = [0, -1, 1];
            let states = 3_usize.pow(scales.len() as u32);
            let mut trits = vec![vec![0_i8; weights.len()]; scales.len()];
            for (weight_index, &weight) in weights.iter().enumerate() {
                let mut best_error = f64::INFINITY;
                let mut best_state = usize::MAX;
                let mut best_codes = [0_i8; 3];
                for state in 0..states {
                    let mut encoded = state;
                    let mut reconstruction = 0.0_f32;
                    let mut codes = [0_i8; 3];
                    for plane in 0..scales.len() {
                        let code = CODES[encoded % 3];
                        encoded /= 3;
                        codes[plane] = code;
                        reconstruction += scales[plane] * f32::from(code);
                    }
                    let error = f64::from(weight) - f64::from(reconstruction);
                    let squared = error * error;
                    if squared < best_error || (squared == best_error && state < best_state) {
                        best_error = squared;
                        best_state = state;
                        best_codes = codes;
                    }
                }
                for plane in 0..scales.len() {
                    trits[plane][weight_index] = best_codes[plane];
                }
            }
            trits
        }

        fn next_up(value: f32) -> f32 {
            if value == 0.0 {
                return f32::from_bits(1);
            }
            if value.is_sign_positive() {
                f32::from_bits(value.to_bits() + 1)
            } else {
                f32::from_bits(value.to_bits() - 1)
            }
        }

        fn next_down(value: f32) -> f32 {
            if value == 0.0 {
                return -f32::from_bits(1);
            }
            if value.is_sign_positive() {
                f32::from_bits(value.to_bits() - 1)
            } else {
                f32::from_bits(value.to_bits() + 1)
            }
        }

        for scales in [
            &[0.75][..],
            &[1.0, 0.5],
            &[1.0, 1.0],
            &[1.0, 0.5, 0.25],
            &[0.125, 0.0625, 0.03125],
        ] {
            const CODES: [i8; 3] = [0, -1, 1];
            let mut reconstructions = (0..3_usize.pow(scales.len() as u32))
                .map(|mut state| {
                    let mut reconstruction = 0.0_f32;
                    for scale in scales {
                        let code = CODES[state % 3];
                        state /= 3;
                        reconstruction += *scale * f32::from(code);
                    }
                    reconstruction
                })
                .collect::<Vec<_>>();
            reconstructions.sort_by(f32::total_cmp);
            reconstructions.dedup_by(|left, right| left.total_cmp(right).is_eq());

            let mut weights = reconstructions.clone();
            for pair in reconstructions.windows(2) {
                let midpoint = ((f64::from(pair[0]) + f64::from(pair[1])) * 0.5) as f32;
                weights.extend([next_down(midpoint), midpoint, next_up(midpoint)]);
            }
            assert_eq!(
                exact_ternary_assignment(&weights, scales).unwrap(),
                reference(&weights, scales),
                "scale set {scales:?}",
            );
        }
    }

    #[test]
    fn fitting_is_bitwise_deterministic() {
        let weights = [-2.4, -1.1, -0.2, 0.0, 0.35, 0.9, 1.8, 3.2];
        let metric = [1.0, 4.0, 0.5, 2.0, 1.0, 3.0, 0.25, 5.0];
        for relay_basins in [
            RelayBasins::default(),
            RelayBasins {
                softened: true,
                modulated: true,
            },
        ] {
            let config = JointFitConfig {
                planes: 2,
                max_iterations: 12,
                ridge: 1e-7,
                scale_precision: ScalePrecision::F32,
                relay_basins,
                ..JointFitConfig::default()
            };

            let first = fit_joint_ternary(&weights, JointFitMetric::Diagonal(&metric), config)
                .expect("first fit");
            let second = fit_joint_ternary(&weights, JointFitMetric::Diagonal(&metric), config)
                .expect("second fit");

            assert_eq!(first, second);
            assert!(first.scales.iter().any(|scale| *scale > 0.0));
            assert!(first.reconstruction.iter().any(|weight| *weight != 0.0));
            let relay_starts =
                usize::from(relay_basins.softened) + usize::from(relay_basins.modulated);
            assert_eq!(
                first.restart_receipts.len(),
                config.em_restarts + relay_starts + 1
            );
            assert!(first.selected_start < first.restart_receipts.len());
        }
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
        for relay_basins in [
            RelayBasins::default(),
            RelayBasins {
                softened: true,
                modulated: false,
            },
            RelayBasins {
                softened: false,
                modulated: true,
            },
            RelayBasins {
                softened: true,
                modulated: true,
            },
        ] {
            for planes in 1..=3 {
                let config = JointFitConfig {
                    planes,
                    em_restarts: 3,
                    relay_basins,
                    ..JointFitConfig::default()
                };
                let fit = fit_joint_ternary(&weights, JointFitMetric::Identity, config)
                    .expect("deterministic multi-start fit");
                let restart_kinds: Vec<_> = fit
                    .restart_receipts
                    .iter()
                    .filter_map(|receipt| match receipt.kind {
                        JointFitStartKind::DeterministicRestart(index) => Some(index),
                        JointFitStartKind::SoftenedRelayBasin
                        | JointFitStartKind::ModulatedRelayBasin
                        | JointFitStartKind::LowerPlaneFallback => None,
                    })
                    .collect();
                assert_eq!(restart_kinds, vec![0, 1, 2]);
                let relay_kinds: Vec<_> = fit
                    .restart_receipts
                    .iter()
                    .filter(|receipt| {
                        matches!(
                            receipt.kind,
                            JointFitStartKind::SoftenedRelayBasin
                                | JointFitStartKind::ModulatedRelayBasin
                        )
                    })
                    .map(|receipt| receipt.kind)
                    .collect();
                let mut expected_relay_kinds = Vec::new();
                if relay_basins.softened {
                    expected_relay_kinds.push(JointFitStartKind::SoftenedRelayBasin);
                }
                if relay_basins.modulated {
                    expected_relay_kinds.push(JointFitStartKind::ModulatedRelayBasin);
                }
                assert_eq!(relay_kinds, expected_relay_kinds);
                let relay_starts =
                    usize::from(relay_basins.softened) + usize::from(relay_basins.modulated);
                assert_eq!(
                    fit.restart_receipts.len(),
                    config.em_restarts + relay_starts + usize::from(planes > 1)
                );
                assert!(
                    fit.restart_receipts
                        .iter()
                        .all(|receipt| receipt.final_objective <= receipt.initial_objective)
                );
            }
        }
    }

    #[test]
    fn converged_scale_candidate_does_not_recompute_current_assignment() {
        ASSIGNMENT_FOR_METRIC_CALLS.with(|calls| calls.set(0));

        let state = optimize_start(
            &[1.0, -1.0],
            JointFitMetric::Identity,
            JointFitConfig::default(),
            vec![1.0],
            JointFitStartKind::DeterministicRestart(0),
        )
        .expect("exact one-plane fit");

        assert_eq!(state.objective, 0.0);
        ASSIGNMENT_FOR_METRIC_CALLS.with(|calls| {
            assert_eq!(
                calls.get(),
                1,
                "initial assignment remains valid because the scale candidate was not accepted"
            );
        });
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
        let mut canonical_trits = trits.clone();
        apply_scale_solve_transform(&mut canonical_trits, &outcome.transform);
        assert_eq!(canonical_trits[0], vec![-1, 1, -1, 1]);
        assert_eq!(canonical_trits[1], vec![-1, 1, 1, -1]);
        let fitted = reconstruct(&outcome.scales, &canonical_trits, weights.len());
        assert!(squared_error(&weights, &fitted) < 1e-20);
        let reference = reconstruct_planes_and_objective(
            &weights,
            &outcome.scales,
            &canonical_trits,
            JointFitMetric::Identity,
        )
        .expect("canonical reference objective");
        let mapped = reconstruct_planes_and_objective_with_transform(
            &weights,
            &outcome.scales,
            &trits,
            JointFitMetric::Identity,
            &outcome.transform,
        )
        .expect("mapped candidate objective");
        assert_eq!(mapped, reference);
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
        for relay_basins in [
            RelayBasins::default(),
            RelayBasins {
                softened: true,
                modulated: true,
            },
        ] {
            let fit = fit_joint_ternary(
                &[-3.1, -1.4, -0.45, 0.15, 0.8, 2.2, 4.0],
                JointFitMetric::Identity,
                JointFitConfig {
                    planes: 3,
                    relay_basins,
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
        for relay_basins in [
            RelayBasins::default(),
            RelayBasins {
                softened: true,
                modulated: true,
            },
        ] {
            let p1 = fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    relay_basins,
                    ..JointFitConfig::default()
                },
            )
            .expect("P1 fit");
            let p2 = fit_joint_ternary(
                &weights,
                JointFitMetric::Identity,
                JointFitConfig {
                    planes: 2,
                    relay_basins,
                    ..JointFitConfig::default()
                },
            )
            .expect("P2 fit");
            let greedy = greedy_residual_reconstruction(&weights, 2);
            let greedy_error = squared_error(&weights, &greedy);

            assert!(p2.objective <= p1.objective);
            assert!(p2.objective <= greedy_error);
            if relay_basins == RelayBasins::default() {
                assert!(
                    p2.accepted_objectives.len() >= 2,
                    "worked sample must exercise an update"
                );
            }
            assert!(
                p2.accepted_objectives
                    .windows(2)
                    .all(|pair| pair[1] < pair[0])
            );
        }
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

    // Deterministic modular pseudo-random groups, the same seedless synthetic-weight pattern the
    // model-fit tests use.
    fn deterministic_group(seed: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let value = (index * (2 * seed + 3) + 5 * seed) % 31;
                (value as f32 - 15.0) / (4.0 + seed as f32)
            })
            .collect()
    }

    #[test]
    fn two_sided_relay_is_odd_zero_at_zero_and_bounded() {
        for sharpness in [5.0_f32, 30.0, 120.0] {
            for delta in [0.1_f32, 0.5, 0.9] {
                assert!(relay::two_sided_relay(0.0, sharpness, delta).abs() <= 1e-7);
                for step in 0..=40 {
                    let v = -1.0 + 0.05 * step as f32;
                    let forward = relay::two_sided_relay(v, sharpness, delta);
                    let mirrored = relay::two_sided_relay(-v, sharpness, delta);
                    assert!(
                        (forward + mirrored).abs() <= 1e-6,
                        "odd symmetry broken at v = {v}, s = {sharpness}, delta = {delta}"
                    );
                    assert!(
                        forward.abs() <= 1.0 + 1e-6,
                        "|f| > 1 at v = {v}, s = {sharpness}, delta = {delta}: {forward}"
                    );
                }
            }
        }
    }

    #[test]
    fn two_sided_relay_sharp_limit_matches_hard_ternary_off_threshold() {
        // Every probe sits at least 0.05 from the |v| = 0.5 threshold, so the tanh terms
        // saturate at sharpness 4096 and the relay collapses to the hard ternary indicator.
        for v in [-0.95_f32, -0.6, -0.2, 0.0, 0.3, 0.55, 0.95] {
            let hard = if v > 0.5 {
                1.0
            } else if v < -0.5 {
                -1.0
            } else {
                0.0
            };
            let soft = relay::two_sided_relay(v, 4096.0, 0.5);
            assert!(
                (soft - hard).abs() <= 1e-6,
                "v = {v}: soft {soft} vs hard {hard}"
            );
        }
    }

    #[test]
    fn relay_basin_scales_are_deterministic_descending_and_non_negative() {
        for seed in [1_usize, 4, 7] {
            let weights = deterministic_group(seed, 24);
            for modulated in [false, true] {
                let first = relay::basin_scales(&weights, 3, modulated, ScalePrecision::F32)
                    .expect("first basin fit");
                let second = relay::basin_scales(&weights, 3, modulated, ScalePrecision::F32)
                    .expect("second basin fit");
                let first_bits: Vec<u32> = first.iter().map(|scale| scale.to_bits()).collect();
                let second_bits: Vec<u32> = second.iter().map(|scale| scale.to_bits()).collect();
                if seed == 1 {
                    let expected = if modulated {
                        [0x3fd5_81fe, 0x3f2f_4664, 0x3e81_d4e6]
                    } else {
                        [0x3fd8_decd, 0x3f1b_3413, 0x3e66_d7aa]
                    };
                    assert_eq!(first_bits, expected);
                }
                assert_eq!(first_bits, second_bits);
                assert_eq!(first.len(), 3);
                assert!(first.iter().all(|scale| *scale >= 0.0));
                assert!(first.windows(2).all(|pair| pair[0] >= pair[1]));
            }
        }
    }

    #[test]
    fn relay_basins_never_worsen_the_final_objective() {
        // Extra accept-only-if-improves basins can only widen the minimized start set, so the
        // selected objective with basins enabled is bounded by the baseline. Exact determinism
        // makes the f64 comparison stable.
        for seed in 0..10 {
            let weights = deterministic_group(seed, 16);
            for planes in 1..=3 {
                let baseline = fit_joint_ternary(
                    &weights,
                    JointFitMetric::Identity,
                    JointFitConfig {
                        planes,
                        ..JointFitConfig::default()
                    },
                )
                .expect("baseline fit");
                let relayed = fit_joint_ternary(
                    &weights,
                    JointFitMetric::Identity,
                    JointFitConfig {
                        planes,
                        relay_basins: RelayBasins {
                            softened: true,
                            modulated: true,
                        },
                        ..JointFitConfig::default()
                    },
                )
                .expect("relay-basin fit");
                assert!(
                    relayed.objective <= baseline.objective,
                    "seed {seed} P{planes}: relay {} > baseline {}",
                    relayed.objective,
                    baseline.objective
                );
            }
        }
    }
}

#[cfg(test)]
mod prefix_stability {
    use super::*;

    /// **A joint fit is not prefix-stable, and the master format depends on believing it is.**
    ///
    /// `salt_v2_master` stores "ordered Pmax planes ... every lower artifact slices this prefix",
    /// and `salt_v2_model` builds that master from a single [`fit_joint_ternary`] at three planes.
    /// That inherits a property greedy residual expansion has and joint optimization does not:
    /// plane 0 of a three-plane fit is whatever best serves the trio, not the best single plane.
    ///
    /// Measured on the Qwen3.6-27B fp master, the gap is 60% on `down_proj` and 123% on
    /// `embed_tokens` in relative Frobenius error, and it reproduces what the shipped artifact
    /// actually decodes to. This keeps a deterministic, dependency-free version of that fact in the
    /// test suite so the prefix assumption cannot be reintroduced as if it held.
    ///
    /// The assertion is deliberately one-directional: it requires the prefix to be strictly worse,
    /// so it fails the moment someone makes the fit prefix-stable — at which point this test should
    /// be deleted along with the workaround it documents, not weakened.
    #[test]
    fn plane_zero_of_a_joint_fit_is_not_the_best_single_plane() {
        // A group whose mass sits in two well-separated magnitudes: a three-plane fit can spend its
        // planes cooperatively, so its first plane has no reason to be a good standalone quantizer.
        let weights: Vec<f32> = (0..128)
            .map(|index| {
                let base = if index % 4 == 0 { 1.0 } else { 0.05 };
                base * if index % 2 == 0 { 1.0 } else { -1.0 } + 0.01 * ((index % 7) as f32 - 3.0)
            })
            .collect();
        let config = |planes| JointFitConfig {
            planes,
            ..JointFitConfig::default()
        };
        let squared = |reconstruction: &[f32]| -> f64 {
            weights
                .iter()
                .zip(reconstruction)
                .map(|(want, got)| f64::from(want - got).powi(2))
                .sum()
        };

        let single = fit_joint_ternary(&weights, JointFitMetric::Identity, config(1))
            .expect("single-plane fit");
        let triple = fit_joint_ternary(&weights, JointFitMetric::Identity, config(3))
            .expect("three-plane fit");
        let pruned: Vec<f32> = triple.trits[0]
            .iter()
            .map(|trit| triple.scales[0] * f32::from(*trit))
            .collect();

        let (direct, prefix) = (squared(&single.reconstruction), squared(&pruned));
        assert!(
            prefix > direct,
            "the joint fit's first plane matched a direct single-plane fit \
             (prefix {prefix:.6e}, direct {direct:.6e}); if fit_joint_ternary is now \
             prefix-stable, delete this test and the pruning workarounds it documents"
        );
    }
}
