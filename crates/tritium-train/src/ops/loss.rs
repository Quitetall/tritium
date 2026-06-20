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
