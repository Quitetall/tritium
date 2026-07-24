//! Scalar losses. Each is a forward (one-element output) + a `vjp` w.r.t. the
//! prediction/logits only — the target is data, a constant of the forward.

/// Mean squared error over every element: `L = mean((pred - target)^2)`.
#[must_use]
pub fn mse_forward(pred: &[f32], target: &[f32]) -> Vec<f32> {
    let n = pred.len() as f32;
    let s: f32 = pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum();
    vec![s / n]
}

/// vjp returning `[gPred]`: `gPred = gOut · (2/N)·(pred - target)`.
///
/// # Panics
/// Panics if `grad_out` is empty (the scalar loss cotangent is read at index 0).
#[must_use]
pub fn mse_vjp(pred: &[f32], target: &[f32], grad_out: &[f32]) -> Vec<Vec<f32>> {
    let n = pred.len() as f32;
    let g = grad_out[0];
    let g_pred = pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| g * 2.0 * (p - t) / n)
        .collect();
    vec![g_pred]
}

/// Numerically-stable row softmax (subtracts the row max before exp).
fn softmax_row(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Softmax cross-entropy averaged over `rows`: `L = mean_r(-Σ_c target·ln softmax(logits))`.
/// `target` is a per-row weighting (typically a one-hot or probability distribution).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn softmax_xent_forward(logits: &[f32], target: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut loss = 0.0f32;
    for r in 0..rows {
        let p = softmax_row(&logits[r * cols..r * cols + cols]);
        for c in 0..cols {
            // ln(p) is finite for finite logits; clamp guards the degenerate p==0.
            loss -= target[r * cols + c] * p[c].max(f32::MIN_POSITIVE).ln();
        }
    }
    vec![loss / rows as f32]
}

/// vjp returning `[gLogits]`: `gLogits[r,c] = gOut/rows · (softmax(logits)[r,c]·Σ_c target[r,·] − target[r,c])`.
/// For a normalized target (Σ_c = 1) this is the familiar `(softmax − target)/rows`;
/// the `Σ target` factor keeps it the exact gradient for any (even unnormalized) target.
///
/// # Panics
/// Panics if `grad_out` is empty (the scalar loss cotangent is read at index 0).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn softmax_xent_vjp(
    logits: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let g = grad_out[0] / rows as f32;
    let mut g_logits = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let p = softmax_row(&logits[r * cols..r * cols + cols]);
        let sum_t: f32 = target[r * cols..r * cols + cols].iter().sum();
        for c in 0..cols {
            g_logits[r * cols + c] = g * (p[c] * sum_t - target[r * cols + c]);
        }
    }
    vec![g_logits]
}

/// Top-k knowledge-distillation loss (Lever 3): softmax cross-entropy against a **sparse** teacher
/// target given as per-row top-`k` `(index, probability)` pairs — `L = mean_r(−Σ_{j<k}
/// prob[r,j]·ln softmax(logits[r])[idx[r,j]])`. Identical to [`softmax_xent_forward`] with the top-k
/// expanded to a dense target, but the teacher only has to store/stream `k` probs+indices per position
/// instead of the full vocabulary (the distillation teacher cache shrinks by `vocab/k`, e.g. ~770× at
/// `k=64`, `vocab≈49k`). `idx[r,j] ∈ [0, cols)`; `prob` need not sum to 1. Student `logits` are dense
/// `[rows, cols]`; `idx`/`prob` are `[rows, k]`, row-major.
///
/// # Panics
/// Panics if any `idx` is out of `[0, cols)` (indexing the softmax row) or the slices are shorter than
/// the declared shape.
#[must_use]
pub fn topk_kd_forward(
    logits: &[f32],
    idx: &[u32],
    prob: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
) -> Vec<f32> {
    let mut loss = 0.0f32;
    for r in 0..rows {
        let p = softmax_row(&logits[r * cols..r * cols + cols]);
        for j in 0..k {
            let c = idx[r * k + j] as usize;
            // ln(p) is finite for finite logits; clamp guards the degenerate p==0.
            loss -= prob[r * k + j] * p[c].max(f32::MIN_POSITIVE).ln();
        }
    }
    vec![loss / rows as f32]
}

/// vjp of [`topk_kd_forward`] returning `[gLogits]`. The full softmax normalizer keeps the gradient
/// **dense** — `gLogits[r,c] = gOut/rows · (softmax[r,c]·Σ_j prob[r,j] − t[r,c])` where `t[r,c]` is the
/// teacher mass landing on column `c` (0 for the `cols − k` untouched columns). Duplicate indices in a
/// row accumulate correctly. This is exactly [`softmax_xent_vjp`] with the sparse target; Lever 3's win
/// is the teacher-cache size, not backward FLOPs (the lm-head gradient stays dense).
///
/// # Panics
/// Panics if `grad_out` is empty, or any `idx` is out of `[0, cols)`.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn topk_kd_vjp(
    logits: &[f32],
    idx: &[u32],
    prob: &[f32],
    rows: usize,
    cols: usize,
    k: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let g = grad_out[0] / rows as f32;
    let mut g_logits = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let p = softmax_row(&logits[r * cols..r * cols + cols]);
        let sum_prob: f32 = prob[r * k..r * k + k].iter().sum();
        for c in 0..cols {
            g_logits[r * cols + c] = g * p[c] * sum_prob;
        }
        // Subtract the sparse teacher mass at the top-k columns (accumulates on duplicate indices).
        for j in 0..k {
            let c = idx[r * k + j] as usize;
            g_logits[r * cols + c] -= g * prob[r * k + j];
        }
    }
    vec![g_logits]
}
