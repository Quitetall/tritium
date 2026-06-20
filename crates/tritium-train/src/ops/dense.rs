//! Plain dense matmul forward + backward for the autograd tape.
//!
//! Forward:  `Y[m,n] = Σ_k X[m,k]·W[n,k]`   (`X:[M,K]`, `W:[N,K]` → `Y:[M,N]`).
//! Backward: `gX[m,k] = Σ_n gY[m,n]·W[n,k]`,  `gW[n,k] = Σ_m gY[m,n]·X[m,k]`.
//!
//! This is the ternary [`matmul`](super::matmul) op with neither the per-row `scale` nor
//! the ternary weight semantics — a real `f32` contraction differentiable in *both*
//! inputs. It is the building block the LoRA delta composes from (`ΔY = (act·Aᵀ)·Bᵀ`):
//! the two factors are continuous trainable weights, so the plain contraction is exactly
//! the right primitive (the ternary path's sign/scale machinery would be wrong here).

/// Forward: `Y[m,n] = Σ_k X[m,k]·W[n,k]`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

/// vjp returning `[gX, gW]` (same shapes as `x`, `w`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(x: &[f32], w: &[f32], m: usize, n: usize, k: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let mut g_x = vec![0.0f32; m * k];
    let mut g_w = vec![0.0f32; n * k];
    for mi in 0..m {
        for ni in 0..n {
            let gy = grad_out[mi * n + ni];
            for ki in 0..k {
                g_x[mi * k + ki] += gy * w[ni * k + ki];
                g_w[ni * k + ki] += gy * x[mi * k + ki];
            }
        }
    }
    vec![g_x, g_w]
}
