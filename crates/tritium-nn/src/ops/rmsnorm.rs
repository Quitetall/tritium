//! RMSNorm — the only normalization BitNet (and the llama family) uses.

use crate::error::NnError;

/// Slots in the canonical reduction order (ADR 0018). Mirrors the 256-thread
/// blocks every backend rmsnorm kernel launches with — slot `t` on the host is
/// thread `t` on the device, so the fold order is identical by construction.
pub(crate) const CANONICAL_REDUCE_SLOTS: usize = 256;

/// Sum of squares in the **canonical cross-backend order** (ADR 0018):
/// slot `t` folds `x[t], x[t+256], …` in ascending index order (f32 multiply
/// then add, no FMA), then a power-of-two tree combines the 256 slots
/// (`off = 128, 64, …, 1`). Every backend implements this exact order, so
/// cross-backend results are bit-identical by construction — and, unlike the
/// pre-1.x sequential fold, the order is parallel- and SIMD-friendly on every
/// target (it removed a hard latency floor from GPU decode; see ADR 0018 for
/// the measurements). The tree sum is also *more* accurate than the
/// sequential fold (shorter accumulation chains round less).
#[inline]
pub(crate) fn sum_squares_canonical(x: &[f32]) -> f32 {
    let mut part = [0.0f32; CANONICAL_REDUCE_SLOTS];
    for (i, &v) in x.iter().enumerate() {
        let t = i % CANONICAL_REDUCE_SLOTS;
        part[t] += v * v;
    }
    let mut off = CANONICAL_REDUCE_SLOTS / 2;
    while off > 0 {
        for t in 0..off {
            part[t] += part[t + off];
        }
        off >>= 1;
    }
    part[0]
}

/// RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + eps) · w[i]`.
///
/// `x`, `w`, and `out` must have equal length. Computed in `f32`; the `mean(x²)`
/// reduction folds in the canonical cross-backend order (ADR 0018).
///
/// # Errors
/// [`NnError::Shape`] if the buffer lengths disagree.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) -> Result<(), NnError> {
    rmsnorm_with_scale(x, w, eps, out, |weight| weight)
}

/// Zero-centered RMSNorm used by Qwen3.5:
/// `out[i] = x[i] / sqrt(mean(x²) + eps) · (1 + w[i])`.
///
/// Qwen3.5 stores zero as the identity scale, unlike ordinary [`rmsnorm`],
/// which stores one. Inputs, weights, reduction, and output are all `f32`.
///
/// # Errors
/// [`NnError::Shape`] if the buffer lengths disagree.
pub fn rmsnorm_zero_centered(
    x: &[f32],
    w: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<(), NnError> {
    rmsnorm_with_scale(x, w, eps, out, |weight| 1.0 + weight)
}

#[inline]
fn rmsnorm_with_scale(
    x: &[f32],
    w: &[f32],
    eps: f32,
    out: &mut [f32],
    scale: impl Fn(f32) -> f32,
) -> Result<(), NnError> {
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
    let mean_sq = sum_squares_canonical(x) / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * inv * scale(wi);
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
