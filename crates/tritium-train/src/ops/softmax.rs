//! Row-wise softmax forward + backward for the autograd tape (attention probabilities).
//!
//! Forward (per row `r` of length `cols`, numerically stable):
//! ```text
//! m_r    = max_j x[r,j]
//! p[r,i] = exp(x[r,i] − m_r) / Σ_j exp(x[r,j] − m_r)
//! ```
//! Backward (Jacobian of softmax): with `dot_r = Σ_j p[r,j]·g[r,j]`,
//! ```text
//! gx[r,i] = p[r,i] · (g[r,i] − dot_r)
//! ```
//! Distinct from the fused [`softmax_xent`](super::loss) in `loss.rs`: this returns the
//! probability vector itself (attention needs `p`, then `p·V`), not a scalar loss.

/// Row-wise softmax forward: `[rows, cols]` → `[rows, cols]`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert!(
        cols > 0,
        "softmax cols must be > 0 (empty row has no distribution)"
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let xr = &x[r * cols..r * cols + cols];
        let m = xr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..cols {
            let e = (xr[i] - m).exp();
            out[r * cols + i] = e;
            sum += e;
        }
        for i in 0..cols {
            out[r * cols + i] /= sum;
        }
    }
    out
}

/// Large finite negative used for masked attention scores. Finite (not `−∞`) so a
/// finite-difference probe of a masked position yields a clean `0`, not `NaN`; after
/// softmax it underflows to probability `~0` just like `−∞` would.
pub const MASK_NEG: f32 = -1e30;

/// Additive causal mask over `[rows=queries, cols=keys]` scores: key `j` is visible to
/// query `i` iff `j <= i` (aligned positions); masked entries become [`MASK_NEG`].
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn causal_mask_forward(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = if j <= i { x[i * cols + j] } else { MASK_NEG };
        }
    }
    out
}

/// vjp of the causal mask: `gx[i,j] = g[i,j]` if `j <= i` else `0` (masked output is a
/// constant, independent of the input).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn causal_mask_vjp(rows: usize, cols: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let mut gx = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            if j <= i {
                gx[i * cols + j] = grad_out[i * cols + j];
            }
        }
    }
    vec![gx]
}

/// vjp returning `[gx]` (shape of `x`): `gx[r,i] = p[r,i]·(g[r,i] − Σ_j p[r,j]·g[r,j])`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(x: &[f32], rows: usize, cols: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let p = forward(x, rows, cols);
    let mut gx = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let pr = &p[r * cols..r * cols + cols];
        let gr = &grad_out[r * cols..r * cols + cols];
        let dot: f32 = (0..cols).map(|j| pr[j] * gr[j]).sum();
        for i in 0..cols {
            gx[r * cols + i] = pr[i] * (gr[i] - dot);
        }
    }
    vec![gx]
}
