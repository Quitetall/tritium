//! Bias add: `Y[m,n] = X[m,n] + b[n]` (b broadcast over the `rows` rows).
//!
//! Backward: `gX = gY`; `gb[n] = Σ_m gY[m,n]`.

/// Forward: add per-column bias `b` to each row of `x` (`[rows, cols]`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(x: &[f32], b: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = x[r * cols + c] + b[c];
        }
    }
    out
}

/// vjp returning `[gX, gb]` (shapes `[rows,cols]` and `[cols]`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(_x: &[f32], _b: &[f32], rows: usize, cols: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_x = grad_out.to_vec();
    let mut g_b = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            g_b[c] += grad_out[r * cols + c];
        }
    }
    vec![g_x, g_b]
}
