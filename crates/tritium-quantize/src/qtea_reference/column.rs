// Copyright (c) 2026 The QTEA Authors. Derived from qtea.py at
// e39bd8c0297e3427daeb404427c74cb6e202b683. MIT notice in LICENSE-MIT.

use tritium_core::Trit;

use super::{
    ArithmeticError, GroupParameters, budget, check_input, check_iterations, filled, finite,
};

/// A refined column's actual trits, multiplier and affine reconstructed values.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinedColumn {
    pub(super) trits: Vec<Trit>,
    pub(super) multiplier: f32,
    pub(super) values: Vec<f32>,
}

impl RefinedColumn {
    /// One ternary assignment per output row.
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }

    /// Shared column-v multiplier, not a per-row scale re-solve.
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }

    /// Native f32 `(alpha*v)*trit+beta` in original row order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

impl GroupParameters {
    /// Refine an original-order weight column using this group's fixed parameters.
    ///
    /// Uses the pinned code's selected absolute mean / **all-row** mean alpha,
    /// not the paper's selected mean of normalized magnitudes. Zero iterations
    /// keeps strict dead-zone assignments and v=1. Reassignment ties go outward.
    /// Charges `(8+sizeof(Trit))*rows` live new array bytes, including returned
    /// trits/values and centered scratch; borrowed group/input bytes are excluded.
    /// No complete Gram feedback or solver registration is performed.
    ///
    /// # Errors
    /// Refuses mismatched shapes, more than64 rounds, insufficient budget,
    /// failed allocation and nonfinite inputs/intermediates.
    pub fn refine_column(
        &self,
        column: &[f32],
        iterations: u32,
        max_work_bytes: usize,
    ) -> Result<RefinedColumn, ArithmeticError> {
        if column.len() != self.alpha.len() {
            return Err(ArithmeticError::InvalidShape);
        }
        check_iterations(iterations)?;
        budget(
            column.len().checked_mul(8 + size_of::<Trit>()),
            max_work_bytes,
        )?;
        check_input(column)?;
        let mut centered = filled(column.len(), 0.0)?;
        let mut trits = filled(column.len(), Trit::ZERO)?;
        let mut values = filled(column.len(), 0.0)?;
        for (row, weight) in column.iter().enumerate() {
            centered[row] = finite(*weight - self.beta[row])?;
            trits[row] = super::group::assign(centered[row], self.delta[row]);
        }
        // Group admission bounds the row count to2^24 (exact f32 integers).
        // The upstream zero-round path does not compute this reduction at all.
        let alpha_mean = if iterations == 0 {
            1.0
        } else {
            let (alpha_sum, count) =
                self.alpha.iter().try_fold((0.0, 0.0), |(sum, n), alpha| {
                    Ok::<_, ArithmeticError>((finite(sum + alpha)?, n + 1.0))
                })?;
            finite(alpha_sum / count)?
        };
        if alpha_mean <= 0.0 {
            return Err(ArithmeticError::NonFiniteArithmetic);
        }
        let mut multiplier = 1.0;
        for _ in 0..iterations {
            let (mut sum, mut selected) = (0.0, 0.0);
            for (value, code) in centered.iter().zip(&trits) {
                if !code.is_zero() {
                    sum = finite(sum + value.abs())?;
                    selected += 1.0;
                }
            }
            multiplier = if selected == 0.0 {
                1.0
            } else {
                finite((sum / selected) / alpha_mean)?.max(0.01)
            };
            for (row, code) in trits.iter_mut().enumerate() {
                let level = finite(self.alpha[row] * multiplier)?;
                let pos = finite(centered[row] - level)?.abs();
                let neg = finite(centered[row] + level)?.abs();
                let zero = centered[row].abs();
                *code = if pos <= neg && pos <= zero {
                    Trit::POS
                } else if neg < pos && neg <= zero {
                    Trit::NEG
                } else {
                    Trit::ZERO
                };
            }
        }
        for (row, value) in values.iter_mut().enumerate() {
            let level = finite(self.alpha[row] * multiplier)?;
            *value = finite(finite(level * trits[row].to_f32())? + self.beta[row])?;
        }
        Ok(RefinedColumn {
            trits,
            multiplier,
            values,
        })
    }
}
