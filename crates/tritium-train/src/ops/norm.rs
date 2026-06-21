//! RMSNorm forward + backward for the autograd tape.
//!
//! Forward (per row `r` of length `cols`, weight `w` shared across rows):
//! ```text
//! inv_r   = 1 / sqrt(mean_i(x[r,i]²) + eps)
//! y[r,i]  = x[r,i] · inv_r · w[i]
//! ```
//! `x` is `[rows, cols]`, `w` is `[cols]`, `y` is `[rows, cols]`. This is the BitNet /
//! llama RMSNorm — matches `tritium_nn::ops::rmsnorm` (the inference forward) so the
//! gradient is taken of the same function the model evaluates.
//!
//! Backward (derived from `inv_r` depending on every `x[r,·]`): with
//! `c_r = Σ_i g[r,i]·w[i]·x[r,i]`,
//! ```text
//! gx[r,k] = inv_r·g[r,k]·w[k] − inv_r³·x[r,k]·c_r / cols
//! gw[i]   = Σ_r g[r,i]·x[r,i]·inv_r
//! ```

/// RMSNorm forward: `y[r,i] = x[r,i] · inv_r · w[i]`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(x: &[f32], w: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    debug_assert!(
        cols > 0,
        "rmsnorm cols must be > 0 (mean over 0 elements is NaN)"
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let xr = &x[r * cols..r * cols + cols];
        let mean_sq = xr.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..cols {
            out[r * cols + i] = xr[i] * inv * w[i];
        }
    }
    out
}

/// vjp returning `[gx, gw]` (shapes of `x` and `w`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(
    x: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    debug_assert!(cols > 0, "rmsnorm cols must be > 0");
    let mut gx = vec![0.0f32; rows * cols];
    let mut gw = vec![0.0f32; cols];
    for r in 0..rows {
        let xr = &x[r * cols..r * cols + cols];
        let gr = &grad_out[r * cols..r * cols + cols];
        let mean_sq = xr.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let mut c = 0.0f32; // Σ_i g·w·x
        for i in 0..cols {
            c += gr[i] * w[i] * xr[i];
        }
        let inv3_c_over_n = inv * inv * inv * c / cols as f32;
        for k in 0..cols {
            gx[r * cols + k] = inv * gr[k] * w[k] - inv3_c_over_n * xr[k];
            gw[k] += gr[k] * xr[k] * inv;
        }
    }
    vec![gx, gw]
}
