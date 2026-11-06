//! Paired salience numerics shared with the model-aware graph adapter (ADR0064).
//! Graph eligibility and executed capture/source lineage are separate obligations.

/// Rejection before any paired caller-owned tensor is changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SalienceFoldError {
    /// No observed samples or channels.
    EmptyStatistics,
    /// Alpha is nonfinite or outside `[0,1]`.
    Alpha,
    /// Second moments are negative or nonfinite.
    Moments,
    /// Empty, overflowing or incompatible matrix/partner geometry.
    Geometry,
    /// The actual stored f32 scale is not finite and positive.
    Scale,
    /// A source weight/partner is not finite.
    NonFiniteSource,
    /// A transformed f32 value would not be finite.
    NonFiniteOutput,
}
impl core::fmt::Display for SalienceFoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "paired salience fold rejected: {self:?}")
    }
}
impl std::error::Error for SalienceFoldError {}

/// One graph-admitted paired operation on row-major f32 tensors.
///
/// Projections have one input column per second-moment channel. The inverse
/// partner has one output row per channel; a norm is a one-column partner.
/// The caller supplies actual graph partners, not inferred naming or eligibility.
#[derive(Debug)]
pub struct SaliencePair<'a> {
    /// Matrices whose input columns are multiplied by the salience scales.
    pub projections: &'a mut [&'a mut [f32]],
    /// Preceding norm or linear partner whose output rows are divided.
    pub inverse: &'a mut [f32],
    /// Row width of the inverse partner (one for a norm).
    pub inverse_columns: usize,
}
impl SaliencePair<'_> {
    /// Validate then apply both halves, returning the actual stored f32 scales.
    ///
    /// Uses the legacy ordered f64 statistics and f32 multiply/divide profile.
    /// It does not combine factors or assert bit-identical model logits.
    /// # Errors
    /// Bad statistics/alpha/geometry, nonfinite scales/inputs/outputs. On error
    /// every supplied tensor is unchanged. Allocation/RSS admission is separate.
    pub fn apply(
        &mut self,
        second_moments: &[f64],
        samples: usize,
        alpha: f64,
    ) -> Result<Vec<f32>, SalienceFoldError> {
        if second_moments.is_empty() || samples == 0 {
            return Err(SalienceFoldError::EmptyStatistics);
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(SalienceFoldError::Alpha);
        }
        if second_moments.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(SalienceFoldError::Moments);
        }
        let channels = second_moments.len();
        if self.projections.is_empty()
            || self.inverse_columns == 0
            || channels.checked_mul(self.inverse_columns) != Some(self.inverse.len())
            || self
                .projections
                .iter()
                .any(|w| w.is_empty() || !w.len().is_multiple_of(channels))
        {
            return Err(SalienceFoldError::Geometry);
        }
        let scales = smooth_scales(second_moments, samples, alpha);
        if scales.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(SalienceFoldError::Scale);
        }
        // Preflight the exact operations without mutation. Do not clamp or
        // silently retain one half if the other cannot be represented.
        for matrix in self.projections.iter() {
            for row in matrix.chunks(channels) {
                for (&value, &scale) in row.iter().zip(&scales) {
                    validate_value(value, value * scale)?;
                }
            }
        }
        for (row, &scale) in self.inverse.chunks(self.inverse_columns).zip(&scales) {
            for &value in row {
                validate_value(value, value / scale)?;
            }
        }
        for matrix in self.projections.iter_mut() {
            scale_cols(matrix, channels, &scales);
        }
        divide_rows(self.inverse, self.inverse_columns, &scales);
        Ok(scales)
    }
}

fn validate_value(source: f32, output: f32) -> Result<(), SalienceFoldError> {
    if !source.is_finite() {
        Err(SalienceFoldError::NonFiniteSource)
    } else if !output.is_finite() {
        Err(SalienceFoldError::NonFiniteOutput)
    } else {
        Ok(())
    }
}

/// Legacy unchecked salience scale helper, retained for existing research callers.
/// New conversion paths should use the checked paired operation instead.
pub fn smooth_scales(acc: &[f64], rows: usize, alpha: f64) -> Vec<f32> {
    let rms: Vec<f64> = acc
        .iter()
        .map(|&s| (s / rows as f64).sqrt().max(1e-8))
        .collect();
    let gm = (rms.iter().map(|r| r.ln()).sum::<f64>() / rms.len() as f64).exp();
    rms.iter().map(|&r| (r / gm).powf(alpha) as f32).collect()
}

/// Legacy unchecked row-major column multiplication; no graph claim.
pub fn scale_cols(w: &mut [f32], cols: usize, s: &[f32]) {
    for row in w.chunks_mut(cols) {
        for (v, &sj) in row.iter_mut().zip(s) {
            *v *= sj;
        }
    }
}

/// Legacy unchecked row-major partner row division; no graph claim.
pub fn divide_rows(w: &mut [f32], cols: usize, s: &[f32]) {
    for (r, row) in w.chunks_mut(cols).enumerate() {
        for v in row {
            *v /= s[r];
        }
    }
}
