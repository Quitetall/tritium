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

use rayon::prelude::*;
use tritium_core::absmean;

/// Parallelize a weight's SALT quantization above this element count (per-row work is independent
/// ⇒ bit-identical to the serial loop). Set high: quantization runs on every weight every step
/// (T planes each), so rayon's per-call fork overhead only pays off on the very large tensors
/// (embeddings, and every weight at 32B scale) — small weights at 135M-scale stay serial. ~1M.
const PAR_MIN_ELEMS: usize = 1 << 20;

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

/// Multi-plane **SALT** residual quantize (round), `t` ternary planes with per-row AbsMean
/// scales — the SALT student's forward. Returns the **dense reconstruction**
/// `Ŵ = Σ_p s_p·trit_p` (`[rows, cols]` row-major). Greedy residual expansion: each plane fits
/// the AbsMean of the running residual, subtracts its contribution, and the next plane fits
/// what's left (ADR 0001 §1). `t == 1` is [`quantize_forward`] scaled back to weight space.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn salt_quantize_forward(wf: &[f32], rows: usize, cols: usize, t: usize) -> Vec<f32> {
    let mut residual = wf.to_vec();
    let mut recon = vec![0.0f32; rows * cols];
    let par = rows * cols >= PAR_MIN_ELEMS;
    // One row's quantize: subtract this plane's ternary contribution from the residual and add it to
    // the reconstruction. Rows are independent within a plane, so parallel == serial bit-for-bit.
    let quant_row = |rec: &mut [f32], res: &mut [f32], sr: f32| {
        if sr == 0.0 {
            return;
        }
        for c in 0..cols {
            let contrib = sr * (res[c] / sr).round().clamp(-1.0, 1.0);
            rec[c] += contrib;
            res[c] -= contrib;
        }
    };
    for _plane in 0..t {
        let s = absmean_scale_per_row(&residual, rows, cols);
        if par {
            recon
                .par_chunks_mut(cols)
                .zip(residual.par_chunks_mut(cols))
                .zip(s.par_iter())
                .for_each(|((rec, res), &sr)| quant_row(rec, res, sr));
        } else {
            recon
                .chunks_mut(cols)
                .zip(residual.chunks_mut(cols))
                .zip(s.iter())
                .for_each(|((rec, res), &sr)| quant_row(rec, res, sr));
        }
    }
    recon
}

/// Straight-through backward for [`salt_quantize_forward`]: the `t`-plane reconstruction tracks
/// the latent `Wf` closely (that is what the residual planes buy — `Ŵ → Wf` as `t` grows, so
/// `dŴ/dWf → I`), so the estimator passes the output gradient straight to the latent —
/// `gWf = grad_out`. Masking out-of-base-range elements (as the single-plane [`quantize_vjp`]
/// does) would under-train exactly the large weights the residual planes exist to represent.
///
/// **Caveat:** although [`salt_quantize_forward`] at `t == 1` equals `s·trit` (the single-plane
/// forward), this identity backward is *not* the single-plane [`quantize_vjp`] — it passes the
/// gradient in the saturated region the mask would kill. `salt_ste(…, 1)` is therefore a stricter
/// (more lenient) estimator than [`quantize_surrogate`]; use the latter for pure single-plane QAT.
/// Returns `[gWf]`.
#[must_use]
pub fn salt_quantize_vjp(
    _wf: &[f32],
    _rows: usize,
    _cols: usize,
    _t: usize,
    grad_out: &[f32],
) -> Vec<f32> {
    grad_out.to_vec()
}

// ── Tequila: deadzone bias (leaky STE) ───────────────────────────────────────────────────────────
// The plain STE masks the saturated region: a weight with `|Wf/s| >= 1` receives EXACTLY zero
// gradient, so once it saturates nothing can ever pull it back — it is dead for the rest of training.
// "Tequila" leaks a fraction `leak` of the incoming gradient through that region (the STE analogue of
// LeakyReLU), letting saturated weights recover. `leak = 0` reproduces the hard mask exactly;
// `leak = 1` is the fully-transparent estimator [`salt_quantize_vjp`] already uses.
//
// NOTE: this only changes the *masked* estimators — [`quantize_vjp`] (single-plane QAT) and
// [`lsq_vjp`] (the LamQuant learned-scale path). The multi-plane [`salt_quantize_vjp`] is already
// identity, so SALT distillation is unaffected by any `leak`.

/// Leaky straight-through surrogate: `clamp(x,-1,1) + leak·(x − clamp(x,-1,1))` with `x = Wf/s_q`,
/// scaled back by `s_q`. Slope 1 in-band, `leak` in the saturated region — the exact
/// finite-difference oracle for [`quantize_vjp_leaky`]. `leak = 0` is [`quantize_surrogate`].
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn quantize_surrogate_leaky(
    wf: &[f32],
    s_q: &[f32],
    rows: usize,
    cols: usize,
    leak: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = s_q[r];
        if s == 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            let x = wf[i] / s;
            let clamped = x.clamp(-1.0, 1.0);
            out[i] = clamped + leak * (x - clamped);
        }
    }
    out
}

/// vjp of [`quantize_surrogate_leaky`]: `gWf[i] = g[i]/s_q[r] · (1 in-band, else `leak`)`.
/// Returns `[gWf, g_sq]` with `g_sq` all-zero (stop-gradient on the scale), matching
/// [`quantize_vjp`], which this generalizes (`leak = 0`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn quantize_vjp_leaky(
    wf: &[f32],
    s_q: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
    leak: f32,
) -> Vec<Vec<f32>> {
    let mut g_wf = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = s_q[r];
        if s == 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            let slope = if (wf[i] / s).abs() < 1.0 { 1.0 } else { leak };
            g_wf[i] = grad_out[i] * slope / s;
        }
    }
    let g_sq = vec![0.0f32; rows];
    vec![g_wf, g_sq]
}

/// [`lsq_vjp`] with the Tequila deadzone leak on the **weight** gradient. The `α` gradient is the
/// unchanged LSQ estimator (it is already defined in the saturated region — that is where `α` gets
/// its signal), so only `gWf` differs: `leak·g` instead of `0` outside the clamp band. `leak = 0`
/// reproduces [`lsq_vjp`] exactly.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn lsq_vjp_leaky(
    wf: &[f32],
    alpha: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
    leak: f32,
) -> Vec<Vec<f32>> {
    let mut g_wf = vec![0.0f32; rows * cols];
    let mut g_a = vec![0.0f32; rows];
    let grad_scale = 1.0 / (cols as f32).sqrt();
    for r in 0..rows {
        let a = alpha[r];
        if a <= 0.0 {
            continue;
        }
        let mut ga = 0.0f32;
        for c in 0..cols {
            let i = r * cols + c;
            let v = wf[i] / a;
            let g = grad_out[i];
            if v.abs() < 1.0 {
                g_wf[i] = g;
                ga += g * (v.round() - v);
            } else {
                g_wf[i] = leak * g; // Tequila: saturated weights stay reachable
                ga += g * v.signum();
            }
        }
        g_a[r] = ga * grad_scale;
    }
    vec![g_wf, g_a]
}

// ── Sherry: cosine-annealed fp residual ──────────────────────────────────────────────────────────
// Quantizing hard from step 0 homogenizes gradients: every weight in a row is forced onto the same
// coarse ternary grid through one shared scale, so the loss surface the optimizer sees early is far
// rougher than the fp one. "Sherry" blends a *decaying* fraction of the untouched fp weight into the
// reconstruction — `(1-α)·Ŵ + α·Wf` — so training starts near the smooth fp landscape and anneals to
// pure ternary. At `α = 0` the forward is bit-identical to [`salt_quantize_forward`], so a run that
// finishes its anneal ends fully ternary with nothing left to remove.

/// Blend the fp weight into a SALT reconstruction: `(1-α)·Ŵ + α·Wf`, `α ∈ [0,1]`.
/// `α = 0` ⇒ exactly [`salt_quantize_forward`]; `α = 1` ⇒ the fp weight untouched.
///
/// The straight-through backward is unchanged (identity, as in [`salt_quantize_vjp`]): the blend is a
/// *forward-only* smoothing of the landscape, and both terms already pass gradient with slope 1.
#[must_use]
pub fn salt_quantize_forward_sherry(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    alpha: f32,
) -> Vec<f32> {
    let mut recon = salt_quantize_forward(wf, rows, cols, t);
    if alpha != 0.0 {
        for (r, &w) in recon.iter_mut().zip(wf) {
            *r += alpha * (w - *r); // (1-α)·recon + α·wf, one rounding
        }
    }
    recon
}

/// Cosine anneal of the Sherry fp-mix from `start` at step 0 down to 0 at `total` (held at 0 after).
/// Mirrors the [`LrSchedule`](crate::LrSchedule) shape so a campaign can drive both off the step index.
#[must_use]
pub fn sherry_alpha(start: f32, step: u64, total: u64) -> f32 {
    if total == 0 || step >= total {
        return 0.0;
    }
    let progress = step as f32 / total as f32;
    start * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
}

// ── LSQ (Learned Step-Size Quantization) ─────────────────────────────────────────────────────────
// A *trainable* per-row step size `α` replacing the fixed AbsMean scale (Esser et al. 2020,
// specialized to the ternary grid `Qn=Qp=1`). Both the latent weight `Wf` and `α` receive gradients:
// `Wf` through the standard round-clamp STE, `α` through the LSQ step-size estimator. LamQuant uses
// this to calibrate the quantizer scale end-to-end instead of pinning AbsMean (ADR 0030 Tier 1).

/// LSQ QAT forward: `q = round(clamp(Wf/α, -1, 1))·α` — the ternary reconstruction with a **learned**
/// per-row scale `α` (`[rows]`). A degenerate `α[r] <= 0` yields an all-zero row.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn lsq_forward(wf: &[f32], alpha: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let a = alpha[r];
        if a <= 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            out[i] = (wf[i] / a).round().clamp(-1.0, 1.0) * a;
        }
    }
    out
}

/// The straight-through **weight-surrogate** `clamp(Wf/α, -1, 1)·α`. Its exact gradient w.r.t. `Wf` is
/// the LSQ `gWf` (`grad·1[|Wf/α|<1]`), so it is the finite-difference oracle for the *weight* gradient.
/// Its `α` gradient is **not** the LSQ `α` estimator (LSQ's uses the rounded value and is not the
/// gradient of any smooth surrogate) — validate `gAlpha` by its closed form + a descent test instead.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn lsq_surrogate(wf: &[f32], alpha: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let a = alpha[r];
        if a <= 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            out[i] = (wf[i] / a).clamp(-1.0, 1.0) * a;
        }
    }
    out
}

/// LSQ backward → `[gWf, gAlpha]`. `gWf[i] = grad·1[|v|<1]` (STE through round, `v=Wf/α`).
/// `gAlpha[r] = (1/√cols)·Σ_i grad_i·( round(v_i)−v_i  if |v_i|<1  else  sign(v_i) )` — the LSQ
/// step-size estimator summed over the row, with the paper's `1/√(Qp·features)` gradient scale
/// (`Qp=1`, `features=cols`) so `α`'s update magnitude tracks the weight grad.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn lsq_vjp(
    wf: &[f32],
    alpha: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut g_wf = vec![0.0f32; rows * cols];
    let mut g_a = vec![0.0f32; rows];
    let grad_scale = 1.0 / (cols as f32).sqrt();
    for r in 0..rows {
        let a = alpha[r];
        if a <= 0.0 {
            continue;
        }
        let mut ga = 0.0f32;
        for c in 0..cols {
            let i = r * cols + c;
            let v = wf[i] / a;
            let g = grad_out[i];
            if v.abs() < 1.0 {
                g_wf[i] = g; // STE: dq/dWf = 1 in-band
                ga += g * (v.round() - v);
            } else {
                ga += g * v.signum(); // saturated: dq/dα = clamped level ±1
            }
        }
        g_a[r] = ga * grad_scale;
    }
    vec![g_wf, g_a]
}
