//! Deterministic native hard projections for matched ternary baselines.

use tritium_core::Trit;

/// One dense ternary plane with one reconstruction scale per matrix row.
#[derive(Clone, Debug, PartialEq)]
pub struct BaselineTernaryPlane {
    trits: Vec<Trit>,
    row_scales: Vec<f32>,
}

impl BaselineTernaryPlane {
    /// Row-major ternary codes.
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }

    /// Non-negative scale for each output row.
    pub fn row_scales(&self) -> &[f32] {
        &self.row_scales
    }
}

/// Hard baseline projection with explicit physical planes.
#[derive(Clone, Debug, PartialEq)]
pub struct TernaryBaselineProjection {
    rows: usize,
    columns: usize,
    planes: Vec<BaselineTernaryPlane>,
}

impl TernaryBaselineProjection {
    /// Output-row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Input-column count.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Physical ternary planes. TTQ uses two because sign magnitudes differ.
    pub fn planes(&self) -> &[BaselineTernaryPlane] {
        &self.planes
    }

    /// Decode exact f32 hard projection for reference evaluation.
    pub fn decode(&self) -> Vec<f32> {
        let mut decoded = vec![0.0; self.rows * self.columns];
        for plane in &self.planes {
            for (index, (output, trit)) in decoded.iter_mut().zip(&plane.trits).enumerate() {
                *output += trit.to_f32() * plane.row_scales[index / self.columns];
            }
        }
        decoded
    }
}

/// Ternary Weight Networks hard-projection recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwnConfig {
    threshold_factor: f32,
}

impl TwnConfig {
    /// Construct finite positive threshold multiplier.
    ///
    /// # Errors
    /// Rejects zero, negative, NaN, and infinite factors.
    pub fn new(threshold_factor: f32) -> Result<Self, TernaryBaselineError> {
        if !threshold_factor.is_finite() || threshold_factor <= 0.0 {
            return Err(TernaryBaselineError::InvalidThresholdFactor);
        }
        Ok(Self { threshold_factor })
    }

    /// Multiplier applied to each row's absolute mean.
    pub const fn threshold_factor(self) -> f32 {
        self.threshold_factor
    }
}

/// Learned hard state required to export Trained Ternary Quantization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TtqState {
    positive_scale: f32,
    negative_scale: f32,
    threshold_ratio: f32,
}

impl TtqState {
    /// Construct finite positive sign scales and open-interval threshold ratio.
    ///
    /// # Errors
    /// Rejects non-positive/non-finite scales or ratio outside `(0, 1)`.
    pub fn new(
        positive_scale: f32,
        negative_scale: f32,
        threshold_ratio: f32,
    ) -> Result<Self, TernaryBaselineError> {
        if !positive_scale.is_finite() || positive_scale <= 0.0 {
            return Err(TernaryBaselineError::InvalidPositiveScale);
        }
        if !negative_scale.is_finite() || negative_scale <= 0.0 {
            return Err(TernaryBaselineError::InvalidNegativeScale);
        }
        if !threshold_ratio.is_finite() || !(0.0..1.0).contains(&threshold_ratio) {
            return Err(TernaryBaselineError::InvalidThresholdRatio);
        }
        Ok(Self {
            positive_scale,
            negative_scale,
            threshold_ratio,
        })
    }

    /// Learned magnitude for positive codes.
    pub const fn positive_scale(self) -> f32 {
        self.positive_scale
    }

    /// Learned magnitude for negative codes.
    pub const fn negative_scale(self) -> f32 {
        self.negative_scale
    }

    /// Learned ratio applied to each row's absolute mean.
    pub const fn threshold_ratio(self) -> f32 {
        self.threshold_ratio
    }
}

/// Deterministically project a row-major matrix with TWN thresholding.
///
/// Nonzeros use strict `abs(weight) > factor * row_absmean`; each row scale is
/// selected nonzero absolute mean. An all-zero row has zero scale and codes.
///
/// # Errors
/// Rejects empty/overflowing/mismatched shapes, non-finite weights, or derived
/// values outside finite f32 range.
pub fn project_twn(
    weights: &[f32],
    rows: usize,
    columns: usize,
    config: TwnConfig,
) -> Result<TernaryBaselineProjection, TernaryBaselineError> {
    validate_matrix(weights, rows, columns)?;
    let mut trits = Vec::new();
    trits
        .try_reserve_exact(weights.len())
        .map_err(|_| TernaryBaselineError::AllocationFailed)?;
    let mut row_scales = Vec::new();
    row_scales
        .try_reserve_exact(rows)
        .map_err(|_| TernaryBaselineError::AllocationFailed)?;
    for row in weights.chunks_exact(columns) {
        let mean = row_absmean(row)?;
        let threshold = mean * f64::from(config.threshold_factor);
        let mut selected_sum = 0.0_f64;
        let mut selected_count = 0_u64;
        for &weight in row {
            let selected = f64::from(weight.abs()) > threshold;
            let trit = if selected {
                selected_sum += f64::from(weight.abs());
                selected_count += 1;
                if weight > 0.0 { Trit::POS } else { Trit::NEG }
            } else {
                Trit::ZERO
            };
            trits.push(trit);
        }
        row_scales.push(finite_f32(if selected_count == 0 {
            0.0
        } else {
            selected_sum / selected_count as f64
        })?);
    }
    Ok(TernaryBaselineProjection {
        rows,
        columns,
        planes: vec![BaselineTernaryPlane { trits, row_scales }],
    })
}

/// Export TTQ hard state as separate positive and negative ternary planes.
///
/// Separating signs preserves learned asymmetric magnitudes and prices TTQ as
/// two physical planes instead of pretending one ternary plane has two scales.
///
/// # Errors
/// Rejects empty/overflowing/mismatched shapes, non-finite weights, or derived
/// thresholds outside finite f32 range.
pub fn project_ttq(
    weights: &[f32],
    rows: usize,
    columns: usize,
    state: TtqState,
) -> Result<TernaryBaselineProjection, TernaryBaselineError> {
    validate_matrix(weights, rows, columns)?;
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    positive
        .try_reserve_exact(weights.len())
        .map_err(|_| TernaryBaselineError::AllocationFailed)?;
    negative
        .try_reserve_exact(weights.len())
        .map_err(|_| TernaryBaselineError::AllocationFailed)?;
    for row in weights.chunks_exact(columns) {
        let threshold = row_absmean(row)? * f64::from(state.threshold_ratio);
        for &weight in row {
            positive.push(if f64::from(weight) > threshold {
                Trit::POS
            } else {
                Trit::ZERO
            });
            negative.push(if f64::from(weight) < -threshold {
                Trit::NEG
            } else {
                Trit::ZERO
            });
        }
    }
    Ok(TernaryBaselineProjection {
        rows,
        columns,
        planes: vec![
            BaselineTernaryPlane {
                trits: positive,
                row_scales: vec![state.positive_scale; rows],
            },
            BaselineTernaryPlane {
                trits: negative,
                row_scales: vec![state.negative_scale; rows],
            },
        ],
    })
}

fn validate_matrix(
    weights: &[f32],
    rows: usize,
    columns: usize,
) -> Result<(), TernaryBaselineError> {
    let Some(expected) = rows.checked_mul(columns) else {
        return Err(TernaryBaselineError::ShapeOverflow { rows, columns });
    };
    if expected == 0 {
        return Err(TernaryBaselineError::EmptyShape { rows, columns });
    }
    if weights.len() != expected {
        return Err(TernaryBaselineError::ShapeMismatch {
            expected,
            got: weights.len(),
        });
    }
    if let Some(index) = weights.iter().position(|weight| !weight.is_finite()) {
        return Err(TernaryBaselineError::NonFiniteWeight { index });
    }
    Ok(())
}

fn row_absmean(row: &[f32]) -> Result<f64, TernaryBaselineError> {
    let sum = row
        .iter()
        .try_fold(0.0_f64, |sum, weight| {
            let next = sum + f64::from(weight.abs());
            next.is_finite().then_some(next)
        })
        .ok_or(TernaryBaselineError::NonFiniteDerived)?;
    Ok(sum / row.len() as f64)
}

fn finite_f32(value: f64) -> Result<f32, TernaryBaselineError> {
    let value = value as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or(TernaryBaselineError::NonFiniteDerived)
}

/// Failure from native TWN or TTQ hard projection.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TernaryBaselineError {
    /// TWN threshold multiplier is not finite and positive.
    InvalidThresholdFactor,
    /// TTQ positive scale is not finite and positive.
    InvalidPositiveScale,
    /// TTQ negative scale is not finite and positive.
    InvalidNegativeScale,
    /// TTQ threshold ratio is not finite and inside `(0, 1)`.
    InvalidThresholdRatio,
    /// Matrix element count overflowed addressable size.
    ShapeOverflow {
        /// Output rows.
        rows: usize,
        /// Input columns.
        columns: usize,
    },
    /// Matrix has a zero dimension.
    EmptyShape {
        /// Output rows.
        rows: usize,
        /// Input columns.
        columns: usize,
    },
    /// Weight payload does not match declared shape.
    ShapeMismatch {
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        got: usize,
    },
    /// Input weight is NaN or infinite.
    NonFiniteWeight {
        /// Row-major weight index.
        index: usize,
    },
    /// Derived threshold or scale exceeded finite f32 range.
    NonFiniteDerived,
    /// Bounded output allocation failed.
    AllocationFailed,
}

impl core::fmt::Display for TernaryBaselineError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidThresholdFactor => {
                formatter.write_str("TWN threshold factor must be finite and positive")
            }
            Self::InvalidPositiveScale => {
                formatter.write_str("TTQ positive scale must be finite and positive")
            }
            Self::InvalidNegativeScale => {
                formatter.write_str("TTQ negative scale must be finite and positive")
            }
            Self::InvalidThresholdRatio => {
                formatter.write_str("TTQ threshold ratio must be finite and inside (0, 1)")
            }
            Self::ShapeOverflow { rows, columns } => {
                write!(
                    formatter,
                    "baseline matrix shape {rows}x{columns} overflows"
                )
            }
            Self::EmptyShape { rows, columns } => {
                write!(formatter, "baseline matrix shape {rows}x{columns} is empty")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(
                    formatter,
                    "baseline matrix needs {expected} weights, got {got}"
                )
            }
            Self::NonFiniteWeight { index } => {
                write!(formatter, "baseline matrix weight {index} is non-finite")
            }
            Self::NonFiniteDerived => {
                formatter.write_str("baseline projection derived a non-finite value")
            }
            Self::AllocationFailed => formatter.write_str("baseline projection allocation failed"),
        }
    }
}

impl std::error::Error for TernaryBaselineError {}
