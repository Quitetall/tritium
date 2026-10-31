// Copyright (c) 2026 The QTEA Authors. Derived from quantizer.py at
// e39bd8c0297e3427daeb404427c74cb6e202b683. MIT notice in LICENSE-MIT.

use tritium_core::Trit;

use super::{ArithmeticError, budget, check_input, check_iterations, filled, finite};

/// Affine per-row parameters fitted on one original column group.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupParameters {
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) delta: Vec<f32>,
    columns: usize,
}

impl GroupParameters {
    /// Fit one nonempty row-major group, with at most2^24 rows,4096 columns and0..=64 ITF rounds.
    ///
    /// `max_work_bytes` charges new live arrays:12*rows+sizeof(Trit)*columns,
    /// including returned parameters and reused row scratch, excluding borrowed
    /// weights/allocator overhead. Not a physical-memory or whole-fit limit.
    /// f32 scalar reductions are left-to-right. No caller input is mutated.
    ///
    /// # Errors
    /// Refuses geometry/config/budget/allocation errors, nonfinite input or
    /// intermediates, and initial rows without a surviving threshold assignment.
    pub fn fit(
        weights: &[f32],
        rows: usize,
        columns: usize,
        delta_coefficient: f32,
        iterations: u32,
        max_work_bytes: usize,
    ) -> Result<Self, ArithmeticError> {
        if rows == 0
            || rows > (1 << 24)
            || columns == 0
            || columns > 4096
            || rows.checked_mul(columns) != Some(weights.len())
        {
            return Err(ArithmeticError::InvalidShape);
        }
        check_iterations(iterations)?;
        if !delta_coefficient.is_finite() || delta_coefficient <= 0.0 {
            return Err(ArithmeticError::InvalidConfig);
        }
        budget(
            rows.checked_mul(12).and_then(|bytes| {
                columns
                    .checked_mul(size_of::<Trit>())
                    .and_then(|scratch| bytes.checked_add(scratch))
            }),
            max_work_bytes,
        )?;
        check_input(weights)?;
        let mut result = Self {
            alpha: filled(rows, 0.0)?,
            beta: filled(rows, 0.0)?,
            delta: filled(rows, 0.0)?,
            columns,
        };
        let mut codes = filled(columns, Trit::ZERO)?;
        for (row, weights) in weights.chunks_exact(columns).enumerate() {
            let (alpha, beta, delta) = fit_row(weights, &mut codes, delta_coefficient, iterations)
                .map_err(|error| match error {
                    ArithmeticError::DegenerateRow { .. } => ArithmeticError::DegenerateRow { row },
                    other => other,
                })?;
            result.alpha[row] = alpha;
            result.beta[row] = beta;
            result.delta[row] = delta;
        }
        Ok(result)
    }

    /// One positive magnitude per output row.
    pub fn alpha(&self) -> &[f32] {
        &self.alpha
    }

    /// One affine center per output row.
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// One strict initial-assignment dead zone per output row.
    pub fn delta(&self) -> &[f32] {
        &self.delta
    }

    /// Original fitted column count (this is not a complete tensor recipe).
    pub const fn columns(&self) -> usize {
        self.columns
    }
}

pub(super) fn assign(centered: f32, delta: f32) -> Trit {
    if centered > delta {
        Trit::POS
    } else if centered < -delta {
        Trit::NEG
    } else {
        Trit::ZERO
    }
}

fn fit_row(
    weights: &[f32],
    codes: &mut [Trit],
    coefficient: f32,
    iterations: u32,
) -> Result<(f32, f32, f32), ArithmeticError> {
    let n = weights.len() as f32; // At most4096: exactly representable.
    let mean = finite(weights.iter().copied().sum::<f32>() / n)?;
    let deviation = weights.iter().try_fold(0.0, |sum, weight| {
        finite(sum + finite(*weight - mean)?.abs())
    })?;
    let mut delta = finite(coefficient * (deviation / n))?;
    let mut selected_sum = 0.0;
    let mut selected_count = 0.0;
    for (weight, code) in weights.iter().zip(codes.iter_mut()) {
        let centered = finite(*weight - mean)?;
        *code = assign(centered, delta);
        if !code.is_zero() {
            selected_sum = finite(selected_sum + centered.abs())?;
            selected_count += 1.0;
        }
    }
    if selected_count == 0.0 {
        return Err(ArithmeticError::DegenerateRow { row: 0 });
    }
    let mut alpha = finite(selected_sum / selected_count)?;
    let mut shift = 0.0;
    for _ in 0..iterations {
        (alpha, shift) = least_squares(weights, codes, mean, alpha)?;
        delta = finite(0.5 * alpha)?;
        let mut changed = false;
        for (weight, code) in weights.iter().zip(codes.iter_mut()) {
            let updated = assign(finite(finite(*weight - mean)? - shift)?, delta);
            changed |= updated != *code;
            *code = updated;
        }
        if !changed {
            break;
        }
    }
    if alpha <= 0.0 {
        return Err(ArithmeticError::NonFiniteArithmetic);
    }
    Ok((alpha, finite(mean + shift)?, delta))
}

fn least_squares(
    weights: &[f32],
    codes: &[Trit],
    mean: f32,
    fallback: f32,
) -> Result<(f32, f32), ArithmeticError> {
    let (mut cc, mut c1, mut rc, mut r1) = (0.0, 0.0, 0.0, 0.0);
    for (weight, code) in weights.iter().zip(codes) {
        let code = code.to_f32();
        let centered = finite(*weight - mean)?;
        cc += code * code;
        c1 += code;
        rc = finite(rc + finite(centered * code)?)?;
        r1 = finite(r1 + centered)?;
    }
    let n = weights.len() as f32;
    let det = finite(finite(cc * n)? - finite(c1 * c1)?)?;
    if det.abs() <= 1e-8 {
        return Ok((fallback.abs().max(1e-8), 0.0));
    }
    let alpha = finite(finite(finite(n * rc)? - finite(c1 * r1)?)? / det)?;
    let shift = finite(finite(finite(cc * r1)? - finite(c1 * rc)?)? / det)?;
    Ok((alpha.abs().max(1e-8), shift))
}
