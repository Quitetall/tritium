// Copyright (c) 2026 The QTEA Authors. Derived from qtea.py at
// e39bd8c0297e3427daeb404427c74cb6e202b683. MIT notice in LICENSE-MIT.

use super::{ArithmeticError, budget, check_input, filled, finite};

/// One1:4 semi-sparse residual column with actual finite E4M3FN bytes.
///
/// Indices and codes are unpacked u8 arrays, not a compressed artifact codec.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseResidualColumn {
    indices: Vec<u8>,
    codes: Vec<u8>,
    scale: f32,
}

impl SparseResidualColumn {
    /// Select first maximum-magnitude row per4 and round its scaled value to FP8.
    ///
    /// Nonempty length must be a multiple of4. Scale is448/max(peak,1e-10);
    /// rounding is nearest/ties-to-even and preserves the selected zero's sign.
    /// Charges2*(rows/4) new live array-element bytes, excluding borrowed input
    /// and allocator overhead. No dense scratch column is allocated.
    ///
    /// # Errors
    /// Refuses shape/budget/allocation errors and nonfinite input/intermediates.
    pub fn fit(residual: &[f32], max_work_bytes: usize) -> Result<Self, ArithmeticError> {
        if residual.is_empty() || !residual.len().is_multiple_of(4) {
            return Err(ArithmeticError::InvalidShape);
        }
        let groups = residual.len() / 4;
        budget(groups.checked_mul(2), max_work_bytes)?;
        check_input(residual)?;
        let peak = residual.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
        let scale = finite(448.0 / peak.max(1e-10))?;
        if scale <= 0.0 {
            return Err(ArithmeticError::NonFiniteArithmetic);
        }
        let mut indices = filled(groups, 0_u8)?;
        let mut codes = filled(groups, 0_u8)?;
        for (group, values) in residual.as_chunks::<4>().0.iter().enumerate() {
            let mut index = 0_u8;
            for candidate in 1_u8..4 {
                if values[usize::from(candidate)].abs() > values[usize::from(index)].abs() {
                    index = candidate;
                }
            }
            indices[group] = index;
            codes[group] = encode(finite(values[usize::from(index)] * scale)?);
            finite(decode(codes[group]) / scale)?;
        }
        Ok(Self {
            indices,
            codes,
            scale,
        })
    }

    /// Original output-row count.
    pub fn rows(&self) -> usize {
        self.indices.len() * 4
    }

    /// Selected0..=3 row position for each consecutive4-row group.
    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    /// Actual E4M3FN bytes, including a sign bit for selected negative zero.
    pub fn fp8_codes(&self) -> &[u8] {
        &self.codes
    }

    /// Multiply residuals by this before encoding; divide decoded FP8 by it.
    pub const fn scale_multiplier(&self) -> f32 {
        self.scale
    }

    /// Reconstruct into exact caller-sized dense storage without allocating.
    ///
    /// # Errors
    /// A wrong output length is rejected before any output mutation.
    pub fn decode_into(&self, output: &mut [f32]) -> Result<(), ArithmeticError> {
        if output.len() != self.rows() {
            return Err(ArithmeticError::InvalidShape);
        }
        output.fill(0.0);
        for (group, (index, code)) in self.indices.iter().zip(&self.codes).enumerate() {
            output[group * 4 + usize::from(*index)] = decode(*code) / self.scale;
        }
        Ok(())
    }
}

pub(super) fn decode(code: u8) -> f32 {
    let magnitude = code & 0x7f;
    let exponent = i32::from(magnitude >> 3);
    let fraction = f32::from(magnitude & 7);
    let value = if exponent == 0 {
        fraction * (1.0 / 512.0)
    } else {
        (8.0 + fraction) * 2.0_f32.powi(exponent - 10)
    };
    if code & 0x80 == 0 { value } else { -value }
}

fn encode(value: f32) -> u8 {
    // Scaled finite peaks are448; multiplication rounding can go one ulp above
    // that. Nearest finite value is still448. Never emit the0x7f NaN encoding.
    let magnitude = value.abs().min(448.0);
    let (mut low, mut high) = (0_u8, 126_u8);
    while low < high {
        let middle = low + (high - low) / 2;
        if decode(middle) < magnitude {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    let lower = low.saturating_sub(1);
    let lower_distance = magnitude - decode(lower);
    let upper_distance = decode(low) - magnitude;
    let code = if lower_distance < upper_distance
        || (lower_distance == upper_distance && lower & 1 == 0)
    {
        lower
    } else {
        low
    };
    code | if value.is_sign_negative() { 0x80 } else { 0 }
}
