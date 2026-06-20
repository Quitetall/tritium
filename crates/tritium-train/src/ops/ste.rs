//! STE-quantize. The **QAT forward** is `trit = round(clamp(Wf/s_q, -1, 1))`
//! ([`quantize_forward`]); the **straight-through backward** passes the gradient
//! through `1/s_q` only where `|Wf/s_q| < 1` ([`quantize_vjp`]).
//!
//! `round` is piecewise-constant, so its true derivative is 0 almost everywhere and
//! it cannot be finite-difference-checked. By the straight-through definition the
//! backward is instead the *exact* gradient of the differentiable surrogate
//! `clamp(Wf/s_q, -1, 1)` ([`quantize_surrogate`]) — and that surrogate is what the
//! Gate-C gradient check finite-differences against.
//!
//! `s_q` is per-row AbsMean (BitNet b1.58), recomputed each step but treated as a
//! constant of the forward (stop-gradient on the quantizer scale).

use tritium_core::absmean;

/// Per-row AbsMean quantizer scale for `[rows, cols]` latent weights.
#[must_use]
pub fn absmean_scale_per_row(wf: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| absmean(&wf[r * cols..r * cols + cols]))
        .collect()
}

/// Forward: trits as `f32` in `{-1,0,+1}`. `s_q[r]==0` (degenerate row) ⇒ all-zero.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn quantize_forward(wf: &[f32], s_q: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = s_q[r];
        for c in 0..cols {
            let i = r * cols + c;
            out[i] = if s == 0.0 {
                0.0
            } else {
                (wf[i] / s).round().clamp(-1.0, 1.0)
            };
        }
    }
    out
}

/// Straight-through differentiable surrogate: `clamp(Wf/s_q, -1, 1)` (no `round`).
/// Its exact gradient w.r.t. `Wf` IS [`quantize_vjp`], so this is the finite-difference
/// oracle for Gate C. `s_q[r]==0` (degenerate row) ⇒ all-zero. The real QAT forward
/// [`quantize_forward`] applies `round` on top; that `round` is invisible to the STE
/// backward by construction (its true gradient is 0 a.e.).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn quantize_surrogate(wf: &[f32], s_q: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = s_q[r];
        for c in 0..cols {
            let i = r * cols + c;
            out[i] = if s == 0.0 {
                0.0
            } else {
                (wf[i] / s).clamp(-1.0, 1.0)
            };
        }
    }
    out
}

/// vjp: `gWf[i] = g[i] · (1/s_q[r]) · 1[ |Wf[i]/s_q[r]| < 1 ]`. Returns one grad
/// buffer per input `[gWf, g_sq]`; `g_sq` is all-zero (stop-gradient on the scale).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn quantize_vjp(
    wf: &[f32],
    s_q: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut g_wf = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = s_q[r];
        if s == 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            if (wf[i] / s).abs() < 1.0 {
                g_wf[i] = grad_out[i] / s;
            }
        }
    }
    let g_sq = vec![0.0f32; rows];
    vec![g_wf, g_sq]
}
