// Copyright (c) 2026 The QTEA Authors. Derived from qtea.py at
// e39bd8c0297e3427daeb404427c74cb6e202b683. MIT notice in LICENSE-MIT.

use super::{ArithmeticError, budget, check_input, filled, finite};

/// Salient-first importance ranking, not the later non-salient LDL pivot order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SalientRanking {
    columns: Vec<usize>,
    salient_count: usize,
}

impl SalientRanking {
    /// Rank by max_row(W²*H_diag²), descending; equal scores use original index.
    ///
    /// Rows/columns must be nonzero and at most2^24. Group size is1..=4096
    /// and divides columns. Ratio is a finite f64 in0..=1. Positive ratios
    /// select at least one complete group, rounded ties-to-even in f64 as in
    /// Python round; zero selects none. Remaining columns stay in score order,
    /// **not** LDL order. Diagonal nonnegativity is not full Gram/SPD admission.
    ///
    /// `max_work_bytes` charges `(sizeof(usize)+4)*columns` new live array
    /// bytes (returned indices and temporary scores). Unstable in-place sorting
    /// uses the explicit unique index tie-break, with no extra heap arrays.
    ///
    /// # Errors
    /// Refuses geometry, ratio, budget/allocation, invalid input/diagonal, and
    /// overflow in squared saliency scores. No fallback ranking is returned.
    pub fn rank(
        weights: &[f32],
        rows: usize,
        columns: usize,
        diagonal: &[f32],
        group_size: usize,
        ratio: f64,
        max_work_bytes: usize,
    ) -> Result<Self, ArithmeticError> {
        if rows == 0
            || columns == 0
            || rows > (1 << 24)
            || columns > (1 << 24)
            || rows.checked_mul(columns) != Some(weights.len())
            || diagonal.len() != columns
            || group_size == 0
            || group_size > 4096
            || !columns.is_multiple_of(group_size)
        {
            return Err(ArithmeticError::InvalidShape);
        }
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(ArithmeticError::InvalidConfig);
        }
        budget(columns.checked_mul(size_of::<usize>() + 4), max_work_bytes)?;
        check_input(weights)?;
        if let Some(column) = diagonal.iter().position(|x| !x.is_finite() || *x < 0.0) {
            return Err(ArithmeticError::InvalidDiagonal { column });
        }
        let mut scores = filled(columns, 0.0_f32)?;
        let mut order = filled(columns, 0_usize)?;
        for (column, value) in order.iter_mut().enumerate() {
            *value = column;
            let h_squared = finite(diagonal[column] * diagonal[column])?;
            for row in weights.chunks_exact(columns) {
                let squared = finite(row[column] * row[column])?;
                scores[column] = scores[column].max(finite(squared * h_squared)?);
            }
        }
        order.sort_unstable_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                .then_with(|| left.cmp(right))
        });
        // Geometry bounds blocks<=2^24, so f64 represents all integer counts.
        let blocks = columns / group_size;
        let salient_count = if ratio == 0.0 {
            0
        } else {
            let selected = ((blocks as f64 * ratio).round_ties_even() as usize).max(1);
            selected.min(blocks) * group_size
        };
        Ok(Self {
            columns: order,
            salient_count,
        })
    }

    /// All original column indices, sorted salient-first by score/index.
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Length of the salient prefix; always a whole group count.
    pub const fn salient_count(&self) -> usize {
        self.salient_count
    }
}
