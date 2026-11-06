//! Shared column-order decay semantics for additive quantization feedback.

use core::fmt;

/// Validated error-propagation fraction, optionally ramped over source columns.
///
/// The policy is independent of group boundaries, fitter and working precision.
/// Default is undecayed GPTQ. It does not select a model recipe or admit quality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeedbackDecay {
    target: f64,
    ramp: bool,
}

/// Invalid decay policy or source-column geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeedbackDecayError {
    /// The requested fraction was non-finite or outside `[0, 1]`.
    InvalidTarget,
    /// A column did not belong to a nonempty source width.
    InvalidColumn {
        /// Requested zero-based source column.
        column: usize,
        /// Total source width.
        columns: usize,
    },
}

impl fmt::Display for FeedbackDecayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTarget => formatter.write_str("feedback decay must be finite in [0, 1]"),
            Self::InvalidColumn { column, columns } => write!(
                formatter,
                "feedback column {column} is outside nonempty width {columns}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FeedbackDecayError {}

impl Default for FeedbackDecay {
    fn default() -> Self {
        Self {
            target: 1.0,
            ramp: false,
        }
    }
}

impl FeedbackDecay {
    /// Construct a finite fraction in `[0, 1]`, rejecting rather than clamping.
    pub fn new(target: f64, ramp: bool) -> Result<Self, FeedbackDecayError> {
        if !target.is_finite() || !(0.0..=1.0).contains(&target) {
            return Err(FeedbackDecayError::InvalidTarget);
        }
        Ok(Self { target, ramp })
    }

    /// Fraction at a source column, not a group-local or callback ordinal.
    ///
    /// A ramp follows the shipping arithmetic `1 - (1-target)*column/(columns-1)`.
    /// A one-column source uses `target`; a constant policy always uses `target`.
    pub fn coefficient(self, column: usize, columns: usize) -> Result<f64, FeedbackDecayError> {
        if column >= columns {
            return Err(FeedbackDecayError::InvalidColumn { column, columns });
        }
        Ok(if self.ramp && columns > 1 {
            1.0 - (1.0 - self.target) * column as f64 / (columns - 1) as f64
        } else {
            self.target
        })
    }
}

/// Historical calibration-density heuristic for the propagation fraction.
///
/// This preserves the existing SmolLM2-derived rule: interpolate in log
/// `tokens/width` between `(2.7, 0.5)` and `(10.7, 0.75)`, clamping outside.
/// Zero tokens or width select `0.5`. It is not a universal quality guarantee.
/// Natural logarithms require `std`; the explicit policy remains `no_std`.
#[cfg(feature = "std")]
#[must_use]
pub fn auto_feedback_decay(calibration_tokens: usize, columns: usize) -> f64 {
    const LOW: (f64, f64) = (2.7, 0.5);
    const HIGH: (f64, f64) = (10.7, 0.75);
    if calibration_tokens == 0 || columns == 0 {
        return LOW.1;
    }
    let ratio = calibration_tokens as f64 / columns as f64;
    let t = ((ratio.ln() - LOW.0.ln()) / (HIGH.0.ln() - LOW.0.ln())).clamp(0.0, 1.0);
    LOW.1 + t * (HIGH.1 - LOW.1)
}
