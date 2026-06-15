//! RMSNorm — the only normalization BitNet (and the llama family) uses.

use crate::error::NnError;

/// RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + eps) · w[i]`.
///
/// `x`, `w`, and `out` must have equal length. Computed in `f32`.
///
/// # Errors
/// [`NnError::Shape`] if the buffer lengths disagree.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) -> Result<(), NnError> {
    if x.len() != w.len() {
        return Err(NnError::Shape {
            expected: x.len(),
            got: w.len(),
        });
    }
    if out.len() != x.len() {
        return Err(NnError::Shape {
            expected: x.len(),
            got: out.len(),
        });
    }
    let n = x.len();
    if n == 0 {
        return Ok(());
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * inv * wi;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weights() {
        // x = [3, 4]; mean(x²) = 12.5; sqrt = 3.5355; out = x / 3.5355.
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rmsnorm(&x, &w, 0.0, &mut out).unwrap();
        let inv = 1.0 / (12.5f32).sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_shape_error() {
        let mut out = [0.0; 2];
        assert!(matches!(
            rmsnorm(&[1.0, 2.0], &[1.0], 0.0, &mut out),
            Err(NnError::Shape { .. })
        ));
    }
}
