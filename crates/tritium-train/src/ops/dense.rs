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
//!
//! The forward and vjp are row-parallel (rayon) above [`PAR_THRESHOLD`] flops; each output row is
//! independent and its inner accumulation order is preserved, so the parallel result is
//! **bit-identical** to the serial loop (the tape gradchecks + the ModelRunner conformance gate at
//! rel 1.85e-6 both still hold). Below the threshold it runs serial to dodge rayon's fork overhead.

use rayon::prelude::*;

/// Parallelize a matmul only when `m·n·k` reaches this many multiply-adds (small matmuls — the toy
/// tests, per-head attention — stay serial, where rayon's task overhead would dominate).
const PAR_THRESHOLD: usize = 1 << 18;

/// Forward: `Y[m,n] = Σ_k X[m,k]·W[n,k]`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    // Each output row `mi` is independent; the inner `ki` accumulation order is preserved, so the
    // parallel and serial results are bit-identical. Serial below the threshold (rayon overhead).
    let row = |mi: usize, dst: &mut [f32]| {
        let xr = &x[mi * k..mi * k + k];
        for (ni, o) in dst.iter_mut().enumerate() {
            let wr = &w[ni * k..ni * k + k];
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += xr[ki] * wr[ki];
            }
            *o = acc;
        }
    };
    if m * n * k >= PAR_THRESHOLD {
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(mi, dst)| row(mi, dst));
    } else {
        out.chunks_mut(n)
            .enumerate()
            .for_each(|(mi, dst)| row(mi, dst));
    }
    out
}

/// Transpose `[rows, cols] → [cols, rows]`. Needed for attention's `P·V` (which contracts
/// the key dim, unlike `dense`'s last-dim contraction): `attn = dense(P, transpose(V))`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn transpose_forward(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// vjp of transpose: `gx[r,c] = g[c,r]` (transpose the cotangent back).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn transpose_vjp(rows: usize, cols: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let mut gx = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            gx[r * cols + c] = grad_out[c * rows + r];
        }
    }
    vec![gx]
}

/// vjp returning `[gX, gW]` (same shapes as `x`, `w`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(x: &[f32], w: &[f32], m: usize, n: usize, k: usize, grad_out: &[f32]) -> Vec<Vec<f32>> {
    let mut g_x = vec![0.0f32; m * k];
    let mut g_w = vec![0.0f32; n * k];
    // Split the fused loop into two race-free passes, each parallel over ITS output's rows with the
    // same accumulation order as the serial fused loop → bit-identical. g_x row mi sums over ni
    // (increasing); g_w row ni sums over mi (increasing) — exactly the original nesting.
    let gx_row = |mi: usize, dst: &mut [f32]| {
        for ni in 0..n {
            let gy = grad_out[mi * n + ni];
            let wr = &w[ni * k..ni * k + k];
            for ki in 0..k {
                dst[ki] += gy * wr[ki];
            }
        }
    };
    let gw_row = |ni: usize, dst: &mut [f32]| {
        for mi in 0..m {
            let gy = grad_out[mi * n + ni];
            let xr = &x[mi * k..mi * k + k];
            for ki in 0..k {
                dst[ki] += gy * xr[ki];
            }
        }
    };
    if m * n * k >= PAR_THRESHOLD {
        g_x.par_chunks_mut(k)
            .enumerate()
            .for_each(|(mi, dst)| gx_row(mi, dst));
        g_w.par_chunks_mut(k)
            .enumerate()
            .for_each(|(ni, dst)| gw_row(ni, dst));
    } else {
        g_x.chunks_mut(k)
            .enumerate()
            .for_each(|(mi, dst)| gx_row(mi, dst));
        g_w.chunks_mut(k)
            .enumerate()
            .for_each(|(ni, dst)| gw_row(ni, dst));
    }
    vec![g_x, g_w]
}
