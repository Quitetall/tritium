//! HESTIA-style differentiable ternarization (arXiv 2601.20745; ADR 0035 WS-C1).
//!
//! Unlike the STE family ([`ste`](super::ste)), which pairs a rounded forward with a
//! surrogate-gradient backward, the HESTIA forward is itself smooth: each latent weight
//! maps to the **expected trit** under a temperature-controlled softmax over the ternary
//! grid `{-1, 0, +1}`. Per element (row `r`, `z = Wf[i]/s[r]`): logits `a_q = -(z-q)²/τ`,
//! `π = softmax(a)` (max-subtracted), `E = π₊₁ - π₋₁`, `out[i] = s[r]·E`. As `τ → 0` the
//! softmax concentrates on the nearest trit and the forward converges to the hard
//! [`quantize_forward`](super::ste::quantize_forward); as `τ → ∞` it goes uniform and the
//! output vanishes. Because the forward is smooth everywhere, the backward is its *exact*
//! gradient — finite-difference-checkable at every point in both `Wf` and `τ`, with no
//! kink placement (the Gate-C strengthening ADR 0035 records).
//!
//! `s` is per-row AbsMean ([`absmean_scale_per_row`](super::ste::absmean_scale_per_row)),
//! recomputed each step but treated as a constant of the forward — stop-gradient on the
//! quantizer scale, exactly as [`quantize_vjp`](super::ste::quantize_vjp) returns an
//! all-zero `g_sq`. `τ` (shared, `[1]`) IS trainable: it receives the exact temperature
//! gradient so the anneal can be learned rather than imposed.
//!
//! Caller contract: `τ >= MIN_DIFFERENTIABLE_TAU`. Smaller temperatures cannot square in
//! binary32 without underflow and therefore cannot represent the exact temperature VJP. An
//! out-of-contract temperature yields an all-zero output with zero gradients, and `s[r] == 0`
//! zeroes that row the same way — the LSQ convention for invalid `α`
//! ([`lsq_forward`](super::ste::lsq_forward)).

/// Smallest binary32 temperature whose square remains normal and supports finite VJP algebra.
///
/// This is exactly `sqrt(f32::MIN_POSITIVE) == 2^-63`, matching Python HESTIA admission.
pub const MIN_DIFFERENTIABLE_TAU: f32 = f32::from_bits(0x2000_0000);

/// Softmax over the ternary grid at normalized weight `z`: `[π₋₁, π₀, π₊₁]`.
/// Max-subtracted (small `τ` makes the logits large-magnitude negatives). The partition
/// sum is `(e₋₁ + e₊₁) + e₀`: negating `z` swaps the ±1 lanes bit-exactly and f32 `+` is
/// commutative, so the sum — and hence `π` — is bit-identical under `z → -z`, making the
/// forward's odd symmetry `out(-Wf) = -out(Wf)` exact.
fn trit_softmax(z: f32, tau: f32) -> [f32; 3] {
    let d = [z + 1.0, z, z - 1.0]; // z - q for q ∈ {-1, 0, +1}
    let a = [-d[0] * d[0] / tau, -d[1] * d[1] / tau, -d[2] * d[2] / tau];
    let m = a[0].max(a[1]).max(a[2]);
    let e = [(a[0] - m).exp(), (a[1] - m).exp(), (a[2] - m).exp()];
    let sum = (e[0] + e[2]) + e[1];
    [e[0] / sum, e[1] / sum, e[2] / sum]
}

/// Forward: `out[i] = s[r]·(π₊₁ - π₋₁)` — the softmax-expected trit scaled back to weight
/// space. `s[r] == 0` (degenerate row) or `τ <= 0` ⇒ all-zero (see module docs).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn hestia_forward(wf: &[f32], s: &[f32], tau: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let t = tau[0];
    if t < MIN_DIFFERENTIABLE_TAU {
        return out;
    }
    for r in 0..rows {
        let sr = s[r];
        if sr == 0.0 {
            continue;
        }
        for c in 0..cols {
            let i = r * cols + c;
            let pi = trit_softmax(wf[i] / sr, t);
            out[i] = sr * (pi[2] - pi[0]);
        }
    }
    out
}

/// Exact vjp of [`hestia_forward`] → `[gWf, gS, gTau]`.
///
/// `dE/dz` and `dE/dτ` use the pairwise covariance identity. This evaluates the
/// temperature numerator before its single division by `τ²`, avoiding the undefined
/// `0·∞` produced by separately materializing `(z-q)²/τ²` at the admitted floor.
/// `d out/d Wf = s·(dE/dz)·(1/s) = dE/dz`, so the scale cancels. `gS` is all-zero
/// (stop-gradient on the scale, see module docs). Degenerate rows/`τ` contribute nothing.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn hestia_vjp(
    wf: &[f32],
    s: &[f32],
    tau: &[f32],
    rows: usize,
    cols: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut g_wf = vec![0.0f32; rows * cols];
    let g_s = vec![0.0f32; rows];
    let mut g_tau = 0.0f32;
    let t = tau[0];
    if t >= MIN_DIFFERENTIABLE_TAU {
        for r in 0..rows {
            let sr = s[r];
            if sr == 0.0 {
                continue;
            }
            for c in 0..cols {
                let i = r * cols + c;
                let z = wf[i] / sr;
                let pi = trit_softmax(z, t);
                let edge_mass = pi[0] * pi[1] + pi[1] * pi[2];
                let span_mass = pi[0] * pi[2];
                let d_e_d_z = (2.0 / t) * (edge_mass + 4.0 * span_mass);
                g_wf[i] = grad_out[i] * d_e_d_z;

                let minus_zero_mass = pi[0] * pi[1];
                let zero_plus_mass = pi[1] * pi[2];
                let minus_plus_mass = pi[0] * pi[2];
                let minus_zero = if minus_zero_mass == 0.0 {
                    0.0
                } else {
                    -minus_zero_mass * (2.0 * z + 1.0)
                };
                let zero_plus = if zero_plus_mass == 0.0 {
                    0.0
                } else {
                    -zero_plus_mass * (2.0 * z - 1.0)
                };
                let minus_plus = if minus_plus_mass == 0.0 {
                    0.0
                } else {
                    -8.0 * minus_plus_mass * z
                };
                let d_e_d_tau = ((minus_zero + zero_plus) + minus_plus) / (t * t);
                g_tau += grad_out[i] * sr * d_e_d_tau;
            }
        }
    }
    vec![g_wf, g_s, vec![g_tau]]
}
