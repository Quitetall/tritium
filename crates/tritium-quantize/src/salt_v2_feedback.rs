//! Reference second-order sequential feedback for SALT V2.
//!
//! This module deliberately uses `f64` and dense matrices. It is the correctness oracle for
//! GPTQ/BlockLDLQ-style feedback around a caller-supplied additive-ternary group fitter, not a
//! deployment kernel. Groups are visited in the exact order supplied, and each group sees the
//! working weights left by all earlier groups.

use core::fmt;

/// One contiguous column group, expressed as a half-open range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnGroup {
    /// First column in the group.
    pub start: usize,
    /// First column after the group.
    pub end: usize,
}

/// Second-order information used to propagate reconstruction residuals.
#[derive(Clone, Copy, Debug)]
pub enum FeedbackMetric<'a> {
    /// Dense symmetric positive-definite inverse Hessian in row-major order.
    InverseHessian(&'a [f64]),
    /// `L D L^T` factors of the inverse Hessian, with full row-major unit-lower `L`.
    InverseHessianLdl {
        /// Full square unit-lower factor in row-major order; its upper triangle must be zero.
        unit_lower: &'a [f64],
        /// Strictly positive diagonal factor.
        diagonal: &'a [f64],
    },
}

/// Inputs to a deterministic sequential-feedback fit.
#[derive(Clone, Copy, Debug)]
pub struct FeedbackProblem<'a> {
    /// Number of output rows in the weight matrix.
    pub rows: usize,
    /// Number of input columns in the weight matrix.
    pub columns: usize,
    /// Original weights in row-major order.
    pub weights: &'a [f64],
    /// Ordered, non-empty, contiguous groups that partition all columns.
    pub groups: &'a [ColumnGroup],
    /// Second-order information, validated before the first fitter callback.
    pub metric: FeedbackMetric<'a>,
}

/// Public group-fitting request passed to the caller.
#[derive(Clone, Copy, Debug)]
pub struct GroupFitRequest<'a> {
    /// Deterministic group ordinal.
    pub group_index: usize,
    /// First source-matrix column represented by this compact block.
    pub column_start: usize,
    /// Number of columns in this compact block.
    pub columns: usize,
    /// Number of output rows in this compact block.
    pub rows: usize,
    /// Current feedback-adjusted weights, compact row-major by `rows x columns`.
    pub working_weights: &'a [f64],
}

/// A malformed feedback problem or reconstruction.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FeedbackError {
    /// The matrix had no rows or no columns.
    EmptyMatrix,
    /// A matrix dimension product overflowed `usize`.
    ShapeOverflow,
    /// The row-major weight slice had the wrong length.
    WeightLengthMismatch {
        /// Required number of scalars.
        expected: usize,
        /// Supplied number of scalars.
        actual: usize,
    },
    /// An original weight was NaN or infinite.
    NonFiniteWeight {
        /// Row-major scalar index.
        index: usize,
    },
    /// No column groups were supplied.
    EmptyGroups,
    /// A group was empty, out of bounds, overlapping, or out of order.
    InvalidGroupRange {
        /// Group ordinal.
        group: usize,
        /// Required start column at this point in the partition.
        expected_start: usize,
        /// Supplied first column.
        start: usize,
        /// Supplied exclusive end column.
        end: usize,
        /// Total matrix columns.
        columns: usize,
    },
    /// The supplied groups ended before covering all columns.
    IncompleteGroupPartition {
        /// Number of columns covered by the supplied prefix.
        covered: usize,
        /// Total matrix columns.
        columns: usize,
    },
    /// Dense inverse-Hessian storage was not `columns x columns`.
    MetricLengthMismatch {
        /// Required scalar count.
        expected: usize,
        /// Supplied scalar count.
        actual: usize,
    },
    /// An inverse-Hessian or factor entry was NaN or infinite.
    NonFiniteMetric {
        /// Row-major scalar index.
        index: usize,
    },
    /// A dense inverse Hessian was not symmetric within numerical tolerance.
    AsymmetricMetric {
        /// First mismatched row.
        row: usize,
        /// First mismatched column.
        column: usize,
    },
    /// An inverse-Hessian or `D` diagonal entry was not finite and strictly positive.
    NonPositiveDiagonal {
        /// Rejected diagonal index.
        index: usize,
    },
    /// The dense inverse Hessian was not positive definite.
    NonPositiveDefinite {
        /// First rejected Cholesky pivot.
        pivot: usize,
    },
    /// Block-LDL elimination produced a non-finite coefficient or Schur entry.
    NonFiniteBlockElimination {
        /// Group being eliminated.
        group: usize,
        /// Full-matrix row of the rejected intermediate.
        row: usize,
        /// Full-matrix column of the rejected intermediate.
        column: usize,
    },
    /// The LDL diagonal did not contain one value per column.
    LdlDiagonalLengthMismatch {
        /// Required diagonal length.
        expected: usize,
        /// Supplied diagonal length.
        actual: usize,
    },
    /// A diagonal entry of the LDL `L` factor was not one.
    NonUnitLdlDiagonal {
        /// Rejected diagonal index.
        index: usize,
    },
    /// The nominally lower-triangular LDL `L` factor had a nonzero upper entry.
    NonZeroLdlUpper {
        /// Upper-triangle row.
        row: usize,
        /// Upper-triangle column.
        column: usize,
    },
    /// A requested group ordinal did not exist.
    GroupOutOfRange {
        /// Rejected group ordinal.
        group: usize,
        /// Number of available groups.
        group_count: usize,
    },
    /// The caller returned the wrong number of reconstruction scalars.
    ReconstructionLengthMismatch {
        /// Group whose reconstruction was rejected.
        group: usize,
        /// Required scalar count.
        expected: usize,
        /// Supplied scalar count.
        actual: usize,
    },
    /// A caller reconstruction contained NaN or infinity.
    NonFiniteReconstruction {
        /// Group whose reconstruction was rejected.
        group: usize,
        /// Compact row-major scalar index.
        index: usize,
    },
    /// Applying a finite residual correction overflowed `f64`.
    NonFiniteFeedback {
        /// Full-matrix row receiving the rejected update.
        row: usize,
        /// Full-matrix column receiving the rejected update.
        column: usize,
    },
}

impl fmt::Display for FeedbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => formatter.write_str("feedback matrix must have rows and columns"),
            Self::ShapeOverflow => formatter.write_str("feedback matrix shape overflow"),
            Self::WeightLengthMismatch { expected, actual } => write!(
                formatter,
                "weight matrix needs {expected} values, received {actual}"
            ),
            Self::NonFiniteWeight { index } => {
                write!(formatter, "weight {index} is not finite")
            }
            Self::EmptyGroups => formatter.write_str("feedback needs at least one column group"),
            Self::InvalidGroupRange {
                group,
                expected_start,
                start,
                end,
                columns,
            } => write!(
                formatter,
                "group {group} is {start}..{end}; expected a non-empty range at {expected_start} within 0..{columns}"
            ),
            Self::IncompleteGroupPartition { covered, columns } => write!(
                formatter,
                "column groups cover 0..{covered}, not the full 0..{columns}"
            ),
            Self::MetricLengthMismatch { expected, actual } => write!(
                formatter,
                "inverse Hessian needs {expected} values, received {actual}"
            ),
            Self::NonFiniteMetric { index } => {
                write!(formatter, "inverse-Hessian data {index} is not finite")
            }
            Self::AsymmetricMetric { row, column } => write!(
                formatter,
                "inverse Hessian differs at ({row}, {column}) and ({column}, {row})"
            ),
            Self::NonPositiveDiagonal { index } => write!(
                formatter,
                "second-order diagonal {index} must be finite and positive"
            ),
            Self::NonPositiveDefinite { pivot } => {
                write!(
                    formatter,
                    "inverse Hessian is not positive definite at pivot {pivot}"
                )
            }
            Self::NonFiniteBlockElimination { group, row, column } => write!(
                formatter,
                "block-LDL elimination for group {group} overflowed at ({row}, {column})"
            ),
            Self::LdlDiagonalLengthMismatch { expected, actual } => write!(
                formatter,
                "LDL diagonal needs {expected} values, received {actual}"
            ),
            Self::NonUnitLdlDiagonal { index } => {
                write!(formatter, "LDL L diagonal {index} is not one")
            }
            Self::NonZeroLdlUpper { row, column } => {
                write!(
                    formatter,
                    "LDL L entry ({row}, {column}) is above the diagonal"
                )
            }
            Self::GroupOutOfRange { group, group_count } => write!(
                formatter,
                "group {group} is outside the available 0..{group_count}"
            ),
            Self::ReconstructionLengthMismatch {
                group,
                expected,
                actual,
            } => write!(
                formatter,
                "group {group} reconstruction needs {expected} values, received {actual}"
            ),
            Self::NonFiniteReconstruction { group, index } => write!(
                formatter,
                "group {group} reconstruction value {index} is not finite"
            ),
            Self::NonFiniteFeedback { row, column } => write!(
                formatter,
                "feedback update overflowed at matrix entry ({row}, {column})"
            ),
        }
    }
}

impl std::error::Error for FeedbackError {}

/// Either framework validation or a caller fitter failure.
#[derive(Clone, Debug, PartialEq)]
pub enum FeedbackRunError<E> {
    /// Sequential-feedback validation failed.
    Feedback(FeedbackError),
    /// The caller-provided group fitter failed.
    Fitter(E),
}

impl<E: fmt::Display> fmt::Display for FeedbackRunError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Feedback(error) => error.fmt(formatter),
            Self::Fitter(error) => write!(formatter, "group fitter failed: {error}"),
        }
    }
}

impl<E> From<FeedbackError> for FeedbackRunError<E> {
    fn from(error: FeedbackError) -> Self {
        Self::Feedback(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FeedbackBlock {
    group: ColumnGroup,
    later_columns: usize,
    // group-column major, then later-column minor: A^-1 B.
    coefficients: Vec<f64>,
}

/// Owned result and mutable state of a sequential-feedback pass.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackState {
    rows: usize,
    columns: usize,
    groups: Vec<ColumnGroup>,
    blocks: Vec<FeedbackBlock>,
    working_weights: Vec<f64>,
    reconstruction: Vec<f64>,
    fit_inputs: Vec<Vec<f64>>,
    fitted: Vec<bool>,
}

impl FeedbackState {
    /// Number of output rows in the full matrix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns in the full matrix.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Deterministic column-group partition used by this state.
    #[must_use]
    pub fn groups(&self) -> &[ColumnGroup] {
        &self.groups
    }

    /// Current feedback-adjusted full matrix in row-major order.
    #[must_use]
    pub fn working_weights(&self) -> &[f64] {
        &self.working_weights
    }

    /// Current fitted reconstruction in full row-major matrix order.
    #[must_use]
    pub fn reconstruction(&self) -> &[f64] {
        &self.reconstruction
    }

    /// Compact working block last presented to a fitted group.
    #[must_use]
    pub fn group_fit_input(&self, group: usize) -> Option<&[f64]> {
        self.fitted
            .get(group)
            .copied()
            .filter(|fitted| *fitted)
            .map(|_| self.fit_inputs[group].as_slice())
    }

    /// Refit one group against its current working block and correct every later working column.
    ///
    /// The propagated correction is based on the exact change in residual, including both a new
    /// reconstruction and any upstream change to this group's working input. Refitting an earlier
    /// group therefore makes later fitted reconstructions stale; call [`Self::refit_suffix`] on the
    /// following group to refresh the dependency chain.
    pub fn refit_group<F, E>(
        &mut self,
        group: usize,
        mut fitter: F,
    ) -> Result<(), FeedbackRunError<E>>
    where
        F: FnMut(GroupFitRequest<'_>) -> Result<Vec<f64>, E>,
    {
        self.fit_one(group, &mut fitter)
    }

    /// Refit all groups from `first_group` to the end in deterministic order.
    ///
    /// If the caller fitter fails, already completed groups in this suffix remain updated, while
    /// the failing group and all groups after it remain untouched.
    pub fn refit_suffix<F, E>(
        &mut self,
        first_group: usize,
        mut fitter: F,
    ) -> Result<(), FeedbackRunError<E>>
    where
        F: FnMut(GroupFitRequest<'_>) -> Result<Vec<f64>, E>,
    {
        if first_group >= self.groups.len() {
            return Err(FeedbackError::GroupOutOfRange {
                group: first_group,
                group_count: self.groups.len(),
            }
            .into());
        }
        for group in first_group..self.groups.len() {
            self.fit_one(group, &mut fitter)?;
        }
        Ok(())
    }

    /// Install a refitted reconstruction and apply its exact residual delta downstream.
    ///
    /// For unchanged group input, if `delta_q = q_new - q_old` and
    /// `C = K_GG^-1 K_GR` for this group's active Schur-complement inverse block `K`, later
    /// working weights receive `delta_q C`. The implementation uses the more general
    /// residual-delta form so it also remains exact after upstream refits.
    pub fn replace_group_reconstruction(
        &mut self,
        group: usize,
        reconstruction: &[f64],
    ) -> Result<(), FeedbackError> {
        let Some(block) = self.blocks.get(group) else {
            return Err(FeedbackError::GroupOutOfRange {
                group,
                group_count: self.groups.len(),
            });
        };
        let width = block.group.end - block.group.start;
        let expected = self.rows * width;
        if reconstruction.len() != expected {
            return Err(FeedbackError::ReconstructionLengthMismatch {
                group,
                expected,
                actual: reconstruction.len(),
            });
        }
        if let Some(index) = reconstruction.iter().position(|value| !value.is_finite()) {
            return Err(FeedbackError::NonFiniteReconstruction { group, index });
        }

        let current_input =
            compact_block(&self.working_weights, self.rows, self.columns, block.group);
        let old_reconstruction =
            compact_block(&self.reconstruction, self.rows, self.columns, block.group);
        let mut residual_delta = vec![0.0; expected];
        for index in 0..expected {
            let new_residual = current_input[index] - reconstruction[index];
            let old_residual = if self.fitted[group] {
                self.fit_inputs[group][index] - old_reconstruction[index]
            } else {
                0.0
            };
            residual_delta[index] = new_residual - old_residual;
        }

        // Build and validate all updates before mutating state.
        let mut updates = vec![0.0; self.rows * block.later_columns];
        for row in 0..self.rows {
            for later in 0..block.later_columns {
                let mut correction = 0.0;
                for local_column in 0..width {
                    correction += residual_delta[row * width + local_column]
                        * block.coefficients[local_column * block.later_columns + later];
                }
                let column = block.group.end + later;
                let updated = self.working_weights[row * self.columns + column] - correction;
                if !updated.is_finite() {
                    return Err(FeedbackError::NonFiniteFeedback { row, column });
                }
                updates[row * block.later_columns + later] = updated;
            }
        }

        for row in 0..self.rows {
            for later in 0..block.later_columns {
                let column = block.group.end + later;
                self.working_weights[row * self.columns + column] =
                    updates[row * block.later_columns + later];
            }
            for local_column in 0..width {
                let compact = row * width + local_column;
                let full = row * self.columns + block.group.start + local_column;
                self.reconstruction[full] = reconstruction[compact];
            }
        }
        self.fit_inputs[group] = current_input;
        self.fitted[group] = true;
        Ok(())
    }

    fn fit_one<F, E>(&mut self, group: usize, fitter: &mut F) -> Result<(), FeedbackRunError<E>>
    where
        F: FnMut(GroupFitRequest<'_>) -> Result<Vec<f64>, E>,
    {
        let Some(range) = self.groups.get(group).copied() else {
            return Err(FeedbackError::GroupOutOfRange {
                group,
                group_count: self.groups.len(),
            }
            .into());
        };
        let input = compact_block(&self.working_weights, self.rows, self.columns, range);
        let reconstruction = fitter(GroupFitRequest {
            group_index: group,
            column_start: range.start,
            columns: range.end - range.start,
            rows: self.rows,
            working_weights: &input,
        })
        .map_err(FeedbackRunError::Fitter)?;
        self.replace_group_reconstruction(group, &reconstruction)
            .map_err(FeedbackRunError::Feedback)
    }
}

/// Fit every column group in order and propagate each residual into later columns.
///
/// For group `G` and remaining columns `R`, let `K` be the active inverse-Hessian Schur block after
/// eliminating every earlier group. The dense reference update is
/// `W_R <- W_R - (W_G - Q_G) K_GG^-1 K_GR`. The callback can invoke `fit_joint_ternary` or any
/// other fitter that returns a finite reconstruction with the same compact shape as its request.
pub fn fit_with_feedback<F, E>(
    problem: FeedbackProblem<'_>,
    mut fitter: F,
) -> Result<FeedbackState, FeedbackRunError<E>>
where
    F: FnMut(GroupFitRequest<'_>) -> Result<Vec<f64>, E>,
{
    let inverse_hessian = validate_problem(problem).map_err(FeedbackRunError::Feedback)?;
    let blocks = build_feedback_blocks(problem.columns, problem.groups, &inverse_hessian)
        .map_err(FeedbackRunError::Feedback)?;
    let mut state = FeedbackState {
        rows: problem.rows,
        columns: problem.columns,
        groups: problem.groups.to_vec(),
        blocks,
        working_weights: problem.weights.to_vec(),
        reconstruction: vec![0.0; problem.weights.len()],
        fit_inputs: vec![Vec::new(); problem.groups.len()],
        fitted: vec![false; problem.groups.len()],
    };
    for group in 0..state.groups.len() {
        state.fit_one(group, &mut fitter)?;
    }
    Ok(state)
}

fn validate_problem(problem: FeedbackProblem<'_>) -> Result<Vec<f64>, FeedbackError> {
    if problem.rows == 0 || problem.columns == 0 {
        return Err(FeedbackError::EmptyMatrix);
    }
    let weight_count = problem
        .rows
        .checked_mul(problem.columns)
        .ok_or(FeedbackError::ShapeOverflow)?;
    if problem.weights.len() != weight_count {
        return Err(FeedbackError::WeightLengthMismatch {
            expected: weight_count,
            actual: problem.weights.len(),
        });
    }
    if let Some(index) = problem.weights.iter().position(|value| !value.is_finite()) {
        return Err(FeedbackError::NonFiniteWeight { index });
    }
    validate_groups(problem.columns, problem.groups)?;
    materialize_inverse_hessian(problem.columns, problem.metric)
}

fn validate_groups(columns: usize, groups: &[ColumnGroup]) -> Result<(), FeedbackError> {
    if groups.is_empty() {
        return Err(FeedbackError::EmptyGroups);
    }
    let mut expected_start = 0;
    for (group, range) in groups.iter().copied().enumerate() {
        if range.start != expected_start || range.end <= range.start || range.end > columns {
            return Err(FeedbackError::InvalidGroupRange {
                group,
                expected_start,
                start: range.start,
                end: range.end,
                columns,
            });
        }
        expected_start = range.end;
    }
    if expected_start != columns {
        return Err(FeedbackError::IncompleteGroupPartition {
            covered: expected_start,
            columns,
        });
    }
    Ok(())
}

fn materialize_inverse_hessian(
    columns: usize,
    metric: FeedbackMetric<'_>,
) -> Result<Vec<f64>, FeedbackError> {
    let square = columns
        .checked_mul(columns)
        .ok_or(FeedbackError::ShapeOverflow)?;
    let mut dense = match metric {
        FeedbackMetric::InverseHessian(values) => {
            validate_square_length(square, values.len())?;
            values.to_vec()
        }
        FeedbackMetric::InverseHessianLdl {
            unit_lower,
            diagonal,
        } => {
            validate_square_length(square, unit_lower.len())?;
            if diagonal.len() != columns {
                return Err(FeedbackError::LdlDiagonalLengthMismatch {
                    expected: columns,
                    actual: diagonal.len(),
                });
            }
            if let Some(index) = unit_lower.iter().position(|value| !value.is_finite()) {
                return Err(FeedbackError::NonFiniteMetric { index });
            }
            let tolerance = 64.0 * f64::EPSILON;
            for index in 0..columns {
                if !diagonal[index].is_finite() || diagonal[index] <= 0.0 {
                    return Err(FeedbackError::NonPositiveDiagonal { index });
                }
                if (unit_lower[index * columns + index] - 1.0).abs() > tolerance {
                    return Err(FeedbackError::NonUnitLdlDiagonal { index });
                }
                for column in index + 1..columns {
                    if unit_lower[index * columns + column].abs() > tolerance {
                        return Err(FeedbackError::NonZeroLdlUpper { row: index, column });
                    }
                }
            }
            let mut reconstructed = vec![0.0; square];
            for row in 0..columns {
                for column in 0..columns {
                    let through = row.min(column);
                    let value = (0..=through)
                        .map(|factor| {
                            unit_lower[row * columns + factor]
                                * diagonal[factor]
                                * unit_lower[column * columns + factor]
                        })
                        .sum();
                    reconstructed[row * columns + column] = value;
                }
            }
            reconstructed
        }
    };

    if let Some(index) = dense.iter().position(|value| !value.is_finite()) {
        return Err(FeedbackError::NonFiniteMetric { index });
    }
    let magnitude = dense
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let symmetry_tolerance = magnitude * 128.0 * f64::EPSILON;
    for row in 0..columns {
        if dense[row * columns + row] <= 0.0 {
            return Err(FeedbackError::NonPositiveDiagonal { index: row });
        }
        for column in row + 1..columns {
            let upper_index = row * columns + column;
            let lower_index = column * columns + row;
            let upper = dense[upper_index];
            let lower = dense[lower_index];
            if (upper - lower).abs() > symmetry_tolerance {
                return Err(FeedbackError::AsymmetricMetric { row, column });
            }
            // Accepted near-symmetry has one canonical meaning. Halving before addition avoids
            // overflowing when both finite entries are close to `f64::MAX`.
            let symmetric = upper * 0.5 + lower * 0.5;
            if !symmetric.is_finite() {
                return Err(FeedbackError::NonFiniteMetric { index: upper_index });
            }
            dense[upper_index] = symmetric;
            dense[lower_index] = symmetric;
        }
    }
    cholesky(&dense, columns)?;
    Ok(dense)
}

fn validate_square_length(expected: usize, actual: usize) -> Result<(), FeedbackError> {
    if actual != expected {
        return Err(FeedbackError::MetricLengthMismatch { expected, actual });
    }
    Ok(())
}

fn cholesky(matrix: &[f64], dimension: usize) -> Result<Vec<f64>, FeedbackError> {
    let magnitude = matrix
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let tolerance = magnitude * 128.0 * f64::EPSILON;
    let mut lower = vec![0.0; matrix.len()];
    for row in 0..dimension {
        for column in 0..=row {
            let prior: f64 = (0..column)
                .map(|factor| lower[row * dimension + factor] * lower[column * dimension + factor])
                .sum();
            if row == column {
                let pivot = matrix[row * dimension + row] - prior;
                if !pivot.is_finite() || pivot <= tolerance {
                    return Err(FeedbackError::NonPositiveDefinite { pivot: row });
                }
                lower[row * dimension + column] = pivot.sqrt();
            } else {
                lower[row * dimension + column] =
                    (matrix[row * dimension + column] - prior) / lower[column * dimension + column];
            }
        }
    }
    Ok(lower)
}

fn build_feedback_blocks(
    columns: usize,
    groups: &[ColumnGroup],
    inverse_hessian: &[f64],
) -> Result<Vec<FeedbackBlock>, FeedbackError> {
    let mut active = inverse_hessian.to_vec();
    let mut blocks = Vec::with_capacity(groups.len());

    for (group_index, group) in groups.iter().copied().enumerate() {
        let width = group.end - group.start;
        let active_columns = columns - group.start;
        let later_columns = active_columns - width;
        let mut diagonal_block = vec![0.0; width * width];
        for row in 0..width {
            for column in 0..width {
                diagonal_block[row * width + column] = active[row * active_columns + column];
            }
        }
        let lower = cholesky(&diagonal_block, width)?;
        let mut coefficients = vec![0.0; width * later_columns];
        for later in 0..later_columns {
            let right_hand_side: Vec<f64> = (0..width)
                .map(|row| active[row * active_columns + width + later])
                .collect();
            let solution = solve_cholesky(&lower, width, &right_hand_side);
            for row in 0..width {
                if !solution[row].is_finite() {
                    return Err(FeedbackError::NonFiniteBlockElimination {
                        group: group_index,
                        row: group.start + row,
                        column: group.end + later,
                    });
                }
                coefficients[row * later_columns + later] = solution[row];
            }
        }

        // Continue the block LDL elimination on the Schur complement. Every later group's
        // feedback must be derived from this active inverse block, rather than from the matching
        // principal block of the original inverse Hessian.
        let mut next_active = vec![0.0; later_columns * later_columns];
        for row in 0..later_columns {
            for column in row..later_columns {
                let correction: f64 = (0..width)
                    .map(|factor| {
                        active[factor * active_columns + width + row]
                            * coefficients[factor * later_columns + column]
                    })
                    .sum();
                let value = active[(width + row) * active_columns + width + column] - correction;
                if !correction.is_finite() || !value.is_finite() {
                    return Err(FeedbackError::NonFiniteBlockElimination {
                        group: group_index,
                        row: group.end + row,
                        column: group.end + column,
                    });
                }
                next_active[row * later_columns + column] = value;
                next_active[column * later_columns + row] = value;
            }
        }
        active = next_active;
        blocks.push(FeedbackBlock {
            group,
            later_columns,
            coefficients,
        });
    }

    Ok(blocks)
}

fn solve_cholesky(lower: &[f64], dimension: usize, right_hand_side: &[f64]) -> Vec<f64> {
    let mut intermediate = vec![0.0; dimension];
    for row in 0..dimension {
        let prior: f64 = (0..row)
            .map(|column| lower[row * dimension + column] * intermediate[column])
            .sum();
        intermediate[row] = (right_hand_side[row] - prior) / lower[row * dimension + row];
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let prior: f64 = (row + 1..dimension)
            .map(|column| lower[column * dimension + row] * solution[column])
            .sum();
        solution[row] = (intermediate[row] - prior) / lower[row * dimension + row];
    }
    solution
}

fn compact_block(full: &[f64], rows: usize, columns: usize, group: ColumnGroup) -> Vec<f64> {
    let width = group.end - group.start;
    let mut compact = Vec::with_capacity(rows * width);
    for row in 0..rows {
        compact.extend_from_slice(&full[row * columns + group.start..row * columns + group.end]);
    }
    compact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_worked_scalar_groups_apply_inverse_hessian_feedback() {
        let weights = [2.0, 3.0];
        let inverse_hessian = [2.0, 1.0, 1.0, 2.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];
        let mut seen = Vec::new();

        let result = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&inverse_hessian),
            },
            |request| {
                seen.push(request.working_weights.to_vec());
                Ok::<_, ()>(if request.group_index == 0 {
                    vec![1.0]
                } else {
                    vec![2.0]
                })
            },
        )
        .unwrap();

        assert_eq!(seen, vec![vec![2.0], vec![2.5]]);
        assert_eq!(result.working_weights(), &[2.0, 2.5]);
        assert_eq!(result.reconstruction(), &[1.0, 2.0]);
    }

    #[test]
    fn hand_worked_block_group_solves_joint_feedback_coefficients() {
        // For G=0..2, A=[[2,1],[1,2]] and B=[1,0], so A^-1 B=[2/3,-1/3].
        // Residual [1,0] must therefore move the last working weight from 3 to 7/3.
        let weights = [2.0, 1.0, 3.0];
        let inverse_hessian = [2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0];
        let groups = [
            ColumnGroup { start: 0, end: 2 },
            ColumnGroup { start: 2, end: 3 },
        ];
        let mut last_group_input = None;

        let result = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 3,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&inverse_hessian),
            },
            |request| {
                Ok::<_, ()>(if request.group_index == 0 {
                    vec![1.0, 1.0]
                } else {
                    last_group_input = Some(request.working_weights[0]);
                    vec![2.0]
                })
            },
        )
        .unwrap();

        assert!((last_group_input.unwrap() - 7.0 / 3.0).abs() < 1e-12);
        assert_slices_close(result.working_weights(), &[2.0, 1.0, 7.0 / 3.0]);
        assert_eq!(result.reconstruction(), &[1.0, 1.0, 2.0]);
    }

    #[test]
    fn sequential_scalar_groups_use_schur_complement_feedback() {
        // K is the inverse Hessian. Eliminating column zero changes the active inverse block for
        // columns one and two from [[2, 0], [0, 2]] to [[3/2, -1/2], [-1/2, 3/2]]. The second
        // feedback coefficient is therefore -1/3, not K[1,2] / K[1,1] = 0.
        let weights = [1.0, 1.0, 0.0];
        let inverse_hessian = [2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
            ColumnGroup { start: 2, end: 3 },
        ];

        let result = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 3,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&inverse_hessian),
            },
            |request| {
                Ok::<_, ()>(if request.group_index < 2 {
                    vec![0.0]
                } else {
                    request.working_weights.to_vec()
                })
            },
        )
        .unwrap();

        assert_slices_close(result.reconstruction(), &[0.0, 0.0, -1.0 / 3.0]);

        // H = K^-1, worked independently for this fixed matrix. The resulting quadratic loss is
        // exactly 2/3 (without a conventional factor of one half).
        let error = [1.0, 1.0, 1.0 / 3.0];
        let hessian = [1.0, -0.5, -0.5, -0.5, 0.75, 0.25, -0.5, 0.25, 0.75];
        let loss: f64 = (0..3)
            .map(|row| {
                error[row]
                    * (0..3)
                        .map(|column| hessian[row * 3 + column] * error[column])
                        .sum::<f64>()
            })
            .sum();
        assert!((loss - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn feedback_beats_independent_rounding_on_correlated_columns() {
        let weights = [0.49, 0.60];
        let inverse_hessian = [1.0, 0.9, 0.9, 1.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];

        let feedback = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&inverse_hessian),
            },
            |request| {
                Ok::<_, ()>(
                    request
                        .working_weights
                        .iter()
                        .map(|value| value.round())
                        .collect(),
                )
            },
        )
        .unwrap();

        // Independent nearest-integer fitting chooses [0, 1]. The feedback-adjusted second
        // column is 0.159, so the sequential fit instead chooses the correlated optimum [0, 0].
        let independent = [0.0, 1.0];
        assert_eq!(feedback.reconstruction(), &[0.0, 0.0]);
        let hessian = [100.0 / 19.0, -90.0 / 19.0, -90.0 / 19.0, 100.0 / 19.0];
        let feedback_loss = quadratic_loss(&weights, feedback.reconstruction(), &hessian);
        let independent_loss = quadratic_loss(&weights, &independent, &hessian);
        assert!((feedback_loss - 0.373_157_894_736_842).abs() < 1e-12);
        assert!((independent_loss - 3.962_631_578_947_37).abs() < 1e-12);
        assert!(feedback_loss < independent_loss);
    }

    #[test]
    fn scale_delta_correction_uses_schur_coefficients_and_matches_clean_recomputation() {
        let weights = [0.74, 0.74, 0.74];
        let inverse_hessian = [1.0, 0.5, 0.25, 0.5, 1.0, 0.4, 0.25, 0.4, 1.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
            ColumnGroup { start: 2, end: 3 },
        ];
        let problem = FeedbackProblem {
            rows: 1,
            columns: 3,
            weights: &weights,
            groups: &groups,
            metric: FeedbackMetric::InverseHessian(&inverse_hessian),
        };
        let mut corrected = fit_with_feedback(problem, |request| {
            Ok::<_, ()>(
                request
                    .working_weights
                    .iter()
                    .map(|value| value.round())
                    .collect(),
            )
        })
        .unwrap();

        // A stale refit that changes scales but does not repair downstream working weights would
        // fit the old [0.74, 0.87, 0.8527] inputs and keep [0.5, 1.0, 1.0].
        let stale_without_delta: Vec<f64> = (0..groups.len())
            .map(|group| round_to_half(corrected.group_fit_input(group).unwrap()[0]))
            .collect();
        assert_eq!(stale_without_delta, vec![0.5, 1.0, 1.0]);

        corrected
            .refit_group(0, |request| {
                Ok::<_, ()>(
                    request
                        .working_weights
                        .iter()
                        .copied()
                        .map(round_to_half)
                        .collect(),
                )
            })
            .unwrap();
        assert_slices_close(
            corrected.working_weights(),
            &[0.74, 0.62, 0.727_666_666_666_666_6],
        );
        corrected
            .refit_suffix(1, |request| {
                Ok::<_, ()>(
                    request
                        .working_weights
                        .iter()
                        .copied()
                        .map(round_to_half)
                        .collect(),
                )
            })
            .unwrap();

        let clean = fit_with_feedback(problem, |request| {
            Ok::<_, ()>(
                request
                    .working_weights
                    .iter()
                    .copied()
                    .map(round_to_half)
                    .collect(),
            )
        })
        .unwrap();
        assert_eq!(corrected.reconstruction(), &[0.5, 0.5, 0.5]);
        assert_ne!(stale_without_delta, clean.reconstruction());
        assert_slices_close(corrected.reconstruction(), clean.reconstruction());
        assert_slices_close(corrected.working_weights(), clean.working_weights());
    }

    #[test]
    fn visits_uneven_groups_and_rows_in_stable_column_order() {
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let inverse_hessian = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let groups = [
            ColumnGroup { start: 0, end: 2 },
            ColumnGroup { start: 2, end: 3 },
            ColumnGroup { start: 3, end: 4 },
        ];
        let mut visits = Vec::new();
        let mut result = fit_with_feedback(
            FeedbackProblem {
                rows: 2,
                columns: 4,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&inverse_hessian),
            },
            |request| {
                visits.push((
                    request.group_index,
                    request.column_start,
                    request.columns,
                    request.rows,
                    request.working_weights.to_vec(),
                ));
                Ok::<_, ()>(request.working_weights.to_vec())
            },
        )
        .unwrap();

        assert_eq!(
            visits,
            vec![
                (0, 0, 2, 2, vec![1.0, 2.0, 5.0, 6.0]),
                (1, 2, 1, 2, vec![3.0, 7.0]),
                (2, 3, 1, 2, vec![4.0, 8.0]),
            ]
        );
        let mut suffix_visits = Vec::new();
        result
            .refit_suffix(1, |request| {
                suffix_visits.push(request.group_index);
                Ok::<_, ()>(request.working_weights.to_vec())
            })
            .unwrap();
        assert_eq!(suffix_visits, vec![1, 2]);
        assert_eq!(result.reconstruction(), weights);
    }

    #[test]
    fn ldl_factors_match_dense_inverse_hessian_feedback() {
        let weights = [2.0, 3.0];
        let dense = [2.0, 1.0, 1.0, 2.0];
        let unit_lower = [1.0, 0.0, 0.5, 1.0];
        let diagonal = [2.0, 1.5];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];
        let fit = |request: GroupFitRequest<'_>| {
            Ok::<_, ()>(if request.group_index == 0 {
                vec![1.0]
            } else {
                vec![2.0]
            })
        };
        let from_dense = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&dense),
            },
            fit,
        )
        .unwrap();
        let from_ldl = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessianLdl {
                    unit_lower: &unit_lower,
                    diagonal: &diagonal,
                },
            },
            fit,
        )
        .unwrap();

        assert_eq!(from_ldl, from_dense);
    }

    #[test]
    fn dense_feedback_is_scale_invariant_and_rejects_tiny_indefinite_metrics() {
        let weights = [1.0, 2.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];
        for scale in [1e-20, 1.0, 1e20] {
            let inverse_hessian = [2.0 * scale, 0.5 * scale, 0.5 * scale, scale];
            let result = fit_with_feedback(
                FeedbackProblem {
                    rows: 1,
                    columns: 2,
                    weights: &weights,
                    groups: &groups,
                    metric: FeedbackMetric::InverseHessian(&inverse_hessian),
                },
                |request| {
                    Ok::<_, ()>(if request.group_index == 0 {
                        vec![0.0]
                    } else {
                        request.working_weights.to_vec()
                    })
                },
            )
            .unwrap();
            assert_slices_close(result.reconstruction(), &[0.0, 1.75]);
        }

        let tiny_indefinite = [1e-20, 2e-20, 2e-20, 1e-20];
        let error = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &weights,
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&tiny_indefinite),
            },
            |request| Ok::<_, ()>(request.working_weights.to_vec()),
        )
        .unwrap_err();
        assert_eq!(
            error,
            FeedbackRunError::Feedback(FeedbackError::NonPositiveDefinite { pivot: 1 })
        );
    }

    #[test]
    fn dense_asymmetry_rejection_is_scale_invariant() {
        let weights = [1.0, 2.0];
        let groups = [ColumnGroup { start: 0, end: 2 }];
        for scale in [1e-20, 1.0, 1e20] {
            let asymmetric = [scale, 0.1 * scale, 0.2 * scale, scale];
            let error = fit_with_feedback(
                FeedbackProblem {
                    rows: 1,
                    columns: 2,
                    weights: &weights,
                    groups: &groups,
                    metric: FeedbackMetric::InverseHessian(&asymmetric),
                },
                |request| Ok::<_, ()>(request.working_weights.to_vec()),
            )
            .unwrap_err();
            assert_eq!(
                error,
                FeedbackRunError::Feedback(FeedbackError::AsymmetricMetric { row: 0, column: 1 })
            );
        }
    }

    #[test]
    fn accepted_near_asymmetry_is_canonicalized_to_its_exact_symmetric_part() {
        let weights = [1.0, 2.0];
        let groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];
        for scale in [1e-20, 1.0, 1e20] {
            let perturbation = 8.0 * f64::EPSILON * scale;
            let upper = 0.5 * scale + perturbation;
            let lower = 0.5 * scale - perturbation;
            assert_ne!(upper, lower);
            let symmetric = upper * 0.5 + lower * 0.5;
            let near_asymmetric = [2.0 * scale, upper, lower, scale];
            let exact_symmetric = [2.0 * scale, symmetric, symmetric, scale];
            let run = |metric: &[f64]| {
                fit_with_feedback(
                    FeedbackProblem {
                        rows: 1,
                        columns: 2,
                        weights: &weights,
                        groups: &groups,
                        metric: FeedbackMetric::InverseHessian(metric),
                    },
                    |request| {
                        Ok::<_, ()>(if request.group_index == 0 {
                            vec![0.0]
                        } else {
                            request.working_weights.to_vec()
                        })
                    },
                )
                .unwrap()
            };

            let canonicalized = run(&near_asymmetric);
            let exact = run(&exact_symmetric);
            assert_eq!(canonicalized.working_weights(), exact.working_weights());
            assert_eq!(canonicalized.reconstruction(), exact.reconstruction());
        }
    }

    #[test]
    fn rejects_malformed_shapes_metrics_factors_and_reconstructions() {
        let groups = [ColumnGroup { start: 0, end: 2 }];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let run = |problem| {
            fit_with_feedback(problem, |request| {
                Ok::<_, ()>(request.working_weights.to_vec())
            })
        };

        assert_eq!(
            run(FeedbackProblem {
                rows: 0,
                columns: 2,
                weights: &[],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&identity),
            }),
            Err(FeedbackRunError::Feedback(FeedbackError::EmptyMatrix))
        );
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&identity),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::WeightLengthMismatch {
                    expected: 2,
                    actual: 1,
                }
            ))
        );
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, f64::NAN],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&identity),
            }),
            Err(FeedbackRunError::Feedback(FeedbackError::NonFiniteWeight {
                index: 1
            }))
        );

        let gap = [ColumnGroup { start: 1, end: 2 }];
        assert!(matches!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &gap,
                metric: FeedbackMetric::InverseHessian(&identity),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::InvalidGroupRange { .. }
            ))
        ));
        assert!(matches!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&[1.0, 0.0, 0.0]),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::MetricLengthMismatch { .. }
            ))
        ));
        assert!(matches!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&[1.0, f64::NAN, f64::NAN, 1.0]),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonFiniteMetric { .. }
            ))
        ));
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&[1.0, 0.2, 0.1, 1.0]),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::AsymmetricMetric { row: 0, column: 1 }
            ))
        );
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&[0.0, 0.0, 0.0, 1.0]),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonPositiveDiagonal { index: 0 }
            ))
        );
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&[1.0, 2.0, 2.0, 1.0]),
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonPositiveDefinite { pivot: 1 }
            ))
        );

        assert!(matches!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessianLdl {
                    unit_lower: &[1.0, 0.25, 0.0, 1.0],
                    diagonal: &[1.0, 1.0],
                },
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonZeroLdlUpper { .. }
            ))
        ));
        assert_eq!(
            run(FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessianLdl {
                    unit_lower: &[1.0, 0.0, 0.0, 1.0],
                    diagonal: &[1.0, -1.0],
                },
            }),
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonPositiveDiagonal { index: 1 }
            ))
        );

        let bad_length = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&identity),
            },
            |_| Ok::<_, ()>(vec![0.0]),
        );
        assert_eq!(
            bad_length,
            Err(FeedbackRunError::Feedback(
                FeedbackError::ReconstructionLengthMismatch {
                    group: 0,
                    expected: 2,
                    actual: 1,
                }
            ))
        );
        let non_finite = fit_with_feedback(
            FeedbackProblem {
                rows: 1,
                columns: 2,
                weights: &[1.0, 2.0],
                groups: &groups,
                metric: FeedbackMetric::InverseHessian(&identity),
            },
            |_| Ok::<_, ()>(vec![0.0, f64::INFINITY]),
        );
        assert_eq!(
            non_finite,
            Err(FeedbackRunError::Feedback(
                FeedbackError::NonFiniteReconstruction { group: 0, index: 1 }
            ))
        );
    }

    fn round_to_half(value: f64) -> f64 {
        (value * 2.0).round() / 2.0
    }

    fn assert_slices_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    fn quadratic_loss(weights: &[f64], reconstruction: &[f64], hessian: &[f64]) -> f64 {
        let error = [
            weights[0] - reconstruction[0],
            weights[1] - reconstruction[1],
        ];
        error[0] * (hessian[0] * error[0] + hessian[1] * error[1])
            + error[1] * (hessian[2] * error[0] + hessian[3] * error[1])
    }
}
