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
use tritium_format::salt_v2::SaltV2Codec;

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

// ── Hadamard rotation front end + finer scale groups ─────────────────────────────────────────────
// SALT's weakness is **outliers**, and it is structural rather than a bug: each plane can add at most
// `±s_p` where `s_p = mean|residual|`, so a scale derived from the bulk can never reach a heavy tail.
// Measured on a 128-wide group with three 6σ outliers, `T=3` leaves error 8.20 on a weight of 9.88
// while the rest of the group sits at 0.12.
//
// A Hadamard rotation fixes the *distribution* rather than the fitter. `H` is orthogonal, so
// `‖H·q − H·w‖ = ‖q − w‖`: rotating before quantizing and un-rotating after preserves the error norm
// exactly, while mixing every coordinate into every other — which spreads one large weight's energy
// across the whole group and makes the residual far more Gaussian. Measured reconstruction SSE on that
// heavy-tailed group: `T=1` 116→57, `T=2` 87→22, **`T=3` 76→13 (5.7×)**. On already-Gaussian data it is
// a no-op (1.00–1.03×), as it should be.
//
// For a linear layer `y = x·Wᵀ`, insert `H·Hᵀ = I`: `y = (x·H)·(W·H)ᵀ`. The rotation therefore runs
// along the **input** dimension — the same axis SALT already fits along — so a deployment folds one
// Hadamard transform into the activations and quantizes `W·H`. The transform is `O(n log n)` with only
// adds and subtracts, so it does not reintroduce multiplies into the ternary path.

/// In-place normalized fast Walsh–Hadamard transform. `v.len()` must be a power of two. The transform
/// is its own inverse at this normalization (`H·H = I`), and preserves the L2 norm.
///
/// # Panics
/// Panics if `v.len()` is not a power of two.
pub fn fast_hadamard(v: &mut [f32]) {
    let n = v.len();
    assert!(
        n.is_power_of_two(),
        "Hadamard needs a power-of-two length, got {n}"
    );
    let mut len = 1;
    while len < n {
        for start in (0..n).step_by(len * 2) {
            for i in start..start + len {
                let (a, b) = (v[i], v[i + len]);
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= scale;
    }
}

/// Whether the Hadamard front end is applied to a group.
///
/// Rotation is **not** universally good: its value depends entirely on the group's tail weight.
/// Measured reconstruction SSE at `T=3` on 128-wide groups — sub-Gaussian data gets *worse*:
///
/// | distribution            | plain | rotated | effect      |
/// |-------------------------|-------|---------|-------------|
/// | uniform (sub-Gaussian)  |  0.59 |    2.98 | 5× **worse**|
/// | Gaussian                |  8.97 |    8.81 | neutral     |
/// | Laplace (heavy)         | 40.6  |   17.5  | 2.3× better |
/// | Gaussian + 3 8σ outliers|137.5  |   15.5  | 8.9× better |
///
/// So [`Auto`](Self::Auto) is the useful policy: fit the group both ways and keep the better one. That
/// is provably never worse than not rotating, and costs **one bit per group** to record the choice
/// (1/128 bits per weight ≈ 0.008 bpw) plus one extra fit at quantization time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RotationPolicy {
    /// Never rotate — the plain grouped fit.
    #[default]
    Never,
    /// Always rotate (diagnostic; hurts sub-Gaussian groups).
    Always,
    /// Rotate a group only when it measurably reduces that group's reconstruction error.
    Auto,
}

/// SALT with an explicit scale **group size** and a **Hadamard rotation** policy.
///
/// - `group`: weights per scale. SALT's default is one scale per output row (576–1536 weights on a
///   135M model), which is *coarser than the TQ2_0 format it deploys into* (256-trit blocks). Passing
///   `group = 256` matches the deployed format; `128` matches the usual PTQ reporting convention.
///   Finer groups cost `16/group` extra bits per weight per plane for the f16 scale.
/// - `rotation`: see [`RotationPolicy`]. Groups whose length is not a power of two (a ragged final
///   block) are never rotated rather than zero-padded, so no phantom weights are introduced.
/// - `iters`: [`salt_quantize_forward_itf`] alternations (`0` = the greedy AbsMean fit).
///
/// `group >= cols`, [`RotationPolicy::Never`], `iters = 0` reproduces [`salt_quantize_forward`].
#[must_use]
pub fn salt_quantize_forward_grouped(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    iters: usize,
    rotation: RotationPolicy,
) -> Vec<f32> {
    let group = group.max(1);
    let mut out = vec![0.0f32; wf.len()];
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (bs, bd) in src.chunks(group).zip(dst.chunks_mut(group)) {
            let (fit, _) = fit_group(bs, t, iters, rotation, &mut buf);
            bd.copy_from_slice(&fit);
        }
    }
    out
}

/// [`salt_quantize_forward_grouped`] with the rotation decisions supplied as a **fixed mask** rather
/// than re-decided per call.
///
/// This is what a *trainer* needs, and what evaluating a trained model needs.
/// [`RotationPolicy::Auto`] picks per group by comparing both fits, so it re-decides every time it
/// is called: run it on a master that training has moved and it can choose differently than the
/// trainer did, quantizing the checkpoint in a basis the student never optimised for. A frozen mask
/// removes that degree of freedom — the same reason `DeviceTrainer` freezes its mask at construction.
///
/// `mask` is one byte per group (`1` = rotate) in row-major group order,
/// `row * cols.div_ceil(group) + block` — exactly [`rotation_mask`]'s layout. Groups whose length is
/// not a power of two are never rotated regardless of their bit, matching the unmasked path.
#[must_use]
pub fn salt_quantize_forward_grouped_masked(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    iters: usize,
    mask: &[u8],
) -> Vec<f32> {
    let group = group.max(1);
    let per_row = cols.div_ceil(group);
    assert_eq!(
        mask.len(),
        rows * per_row,
        "rotation mask must carry one byte per group"
    );
    let mut out = vec![0.0f32; wf.len()];
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (b, (bs, bd)) in src.chunks(group).zip(dst.chunks_mut(group)).enumerate() {
            // Always/Never rather than Auto: the bit already IS the decision.
            let policy = if mask[r * per_row + b] == 1 {
                RotationPolicy::Always
            } else {
                RotationPolicy::Never
            };
            let (fit, _) = fit_group(bs, t, iters, policy, &mut buf);
            bd.copy_from_slice(&fit);
        }
    }
    out
}

/// Fit one scale group, returning `(reconstruction, was_rotated)`.
///
/// This is the **single** place the rotation decision is made. Both
/// [`salt_quantize_forward_grouped`] and [`rotation_mask`] go through it, so a mask shipped to a
/// device backend cannot disagree with the reconstruction the host fitter would have produced —
/// a divergence that would be silent and would corrupt training rather than fail a gate.
fn fit_group(
    bs: &[f32],
    t: usize,
    iters: usize,
    rotation: RotationPolicy,
    buf: &mut Vec<f32>,
) -> (Vec<f32>, bool) {
    let sse = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
            .sum()
    };
    // A ragged final block is never rotated rather than zero-padded, so no phantom weights appear.
    let rotatable = bs.len().is_power_of_two() && bs.len() > 1;
    let rotated = (rotation != RotationPolicy::Never && rotatable).then(|| {
        buf.clear();
        buf.extend_from_slice(bs);
        fast_hadamard(buf); // into the rotated basis
        let q = salt_quantize_forward_itf(buf, 1, buf.len(), t, iters);
        buf.copy_from_slice(&q);
        fast_hadamard(buf); // H is its own inverse: back to the original basis
        buf.clone()
    });
    match (rotation, rotated) {
        (RotationPolicy::Always, Some(rot)) => (rot, true),
        (RotationPolicy::Auto, Some(rot)) => {
            let plain = salt_quantize_forward_itf(bs, 1, bs.len(), t, iters);
            // Keep whichever candidate actually fits this group better.
            if sse(&rot, bs) < sse(&plain, bs) {
                (rot, true)
            } else {
                (plain, false)
            }
        }
        _ => (salt_quantize_forward_itf(bs, 1, bs.len(), t, iters), false),
    }
}

/// The per-group rotation decisions [`salt_quantize_forward_grouped`] would make, as one byte per
/// group (`1` = rotate), in row-major group order — the layout a device kernel indexes by
/// `row * cols.div_ceil(group) + block`.
///
/// A GPU trainer decides rotation **once on the host** from the initial weights rather than
/// re-deciding every step: a flipping rotation bit would make the loss surface discontinuous, and
/// the deployed format needs one fixed bit per group anyway.
///
/// `iters` must match the fitter that will consume the mask — a mask chosen under one fitter and
/// applied to another is worse than either used consistently. `DeviceTrainer` passes its
/// `SaltGrouping::iters` here and to the device kernel, so the two always agree.
#[must_use]
pub fn rotation_mask(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    iters: usize,
    rotation: RotationPolicy,
) -> Vec<u8> {
    let group = group.max(1);
    let mut mask = Vec::with_capacity(rows * cols.div_ceil(group));
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        for bs in wf[r * cols..(r + 1) * cols].chunks(group) {
            let (_, rotated) = fit_group(bs, t, iters, rotation, &mut buf);
            mask.push(u8::from(rotated));
        }
    }
    mask
}

/// Packed bits per weight for `t` ternary planes at scale-group size `group` under the **D2** codec:
/// 2 bits per trit plus one f16 scale per group, per plane. `group = 256` gives TQ2_0's 2.0625 bpw
/// per plane.
///
/// D2 wastes ~20% of its trit bits: a trit carries `log2(3) = 1.585` bits of entropy but D2 spends 2.
/// See [`ternary_bits_per_weight_codec`] for the rate an actual packer achieves.
#[must_use]
pub fn ternary_bits_per_weight(t: usize, group: usize) -> f64 {
    ternary_bits_per_weight_codec(t, group, SaltV2Codec::D2, group)
}

/// Packed bits per weight for `t` planes, taken from the **packer's own byte ledger** rather than a
/// hand-derived constant — so a reported bpw cannot drift from what the artifact actually costs.
///
/// `run` is the number of trits packed as one contiguous codec run (for a row-major weight this is
/// the row length, or `group` when each group is packed independently). It matters because every
/// codec pads its final unit: B3 packs five trits per byte, so a 128-trit run costs
/// `ceil(128/5) = 26` bytes = 1.625 bits/trit, not the asymptotic 1.6.
///
/// | codec | bits/trit (asymptotic) | T=3 g128 |
/// |---|---|---|
/// | `D2` | 2.0 | 6.38 bpw |
/// | `B3` | 1.6 | **5.20 bpw** |
/// | `S34` | 1.25 | 4.13 bpw (structured: one forced zero per four trits) |
///
/// The `16/group` scale term is unchanged — codecs compress trits, not the f16 group scales.
#[must_use]
pub fn ternary_bits_per_weight_codec(
    t: usize,
    group: usize,
    codec: SaltV2Codec,
    run: usize,
) -> f64 {
    let group = group.max(1);
    let run = run.max(1);
    // Ask the packer, do not re-derive: `ledger` is the same code path `pack_b3` uses to size its
    // output buffer, so this figure is the artifact's real cost including terminal padding.
    let bits_per_trit = match codec.ledger(run) {
        Ok(l) => (l.physical_bytes * 8) as f64 / run as f64,
        // A length the codec cannot account for; fall back to the dense rate rather than lie low.
        Err(_) => 2.0,
    };
    t as f64 * (bits_per_trit + 16.0 / group as f64)
}

/// Packed bits per weight for the **balanced-ternary ladder**, which stores one `s₀` per group for
/// the whole plane stack instead of one f16 scale per plane per group.
///
/// The trit payload is identical to [`ternary_bits_per_weight_codec`] — same `T` planes, same codec —
/// so the entire difference is `(T-1)·16/group` bits of scale that the ladder does not need. At
/// `T=3`, `group=128`, B3 that is 5.25 → **5.00 bpw**, before any quality change. (B3 over a
/// 128-trit run costs `ceil(128/5) = 26` bytes = 1.625 bits/trit, not the asymptotic 1.6.)
#[must_use]
pub fn ternary_bits_per_weight_geometric(
    t: usize,
    group: usize,
    codec: SaltV2Codec,
    run: usize,
) -> f64 {
    let group = group.max(1);
    ternary_bits_per_weight_codec(t, group, codec, run)
        - (t.max(1) - 1) as f64 * 16.0 / group as f64
}

// ── The balanced-ternary scale ladder ────────────────────────────────────────────────────────────
// The greedy expansion below picks `s_p` from the residual plane `p-1` left behind. Measured on real
// weight groups, the resulting ratios `s_{p+1}/s_p` run ≈0.41, 0.42, 0.58, 0.70 and eventually exceed
// 1.0 — a later plane taking a LARGER scale than its predecessor (`fit_plane_itf`'s accept guard will
// take that trade whenever chasing a few outliers lowers SSE). Each added plane then buys ~1 dB of
// reconstruction against the 9.54 dB (= 1.585 bits × 6.02) the rate allows.
//
// Note what is NOT the problem: the reachable levels do not collide. Enumerating `Σ ε_p s_p` for the
// greedy ladder yields 27/27, 81/81, 729/729, 6561/6561 DISTINCT values on Gaussian/Laplace/t₃ groups.
// The levels are distinct but badly spaced — new planes land beside existing levels instead of
// subdividing the gaps.
//
// Pinning the ratio to exactly 1/3 is balanced ternary. Then
//
//     Σ_p ε_p s_p = Δ · k,   Δ = s₀·3^-(T-1),   k = Σ_p ε_p 3^(T-p)
//
// and `k` ranges over EVERY integer in ±(3^T-1)/2, bijectively. So the levels are a uniform grid, the
// exact minimum-SSE assignment is one `round` followed by base-3 digit extraction, and:
//
//   * the fit is O(T) per weight instead of 3^T enumeration — cheap enough to run every training step;
//   * everything after the single `round` is integer, so a device mirror can be bit-exact rather than
//     tolerance-checked (ADR 0018);
//   * one scale is stored per group instead of T.
//
// This is a forward reconstruction rule only. `salt_quantize_vjp` is untouched — still identity STE.
//
// SAME PROXY-GAP CAVEAT as ITF above: this is guaranteed to improve weight-space MSE, and that does
// not automatically mean better perplexity. It ships on held-out ppl or not at all.

/// `3^n` for `n ≤ 8`, exact in `f32` (3^8 = 6561 ≪ 2^24).
fn pow3(n: usize) -> f32 {
    let mut v = 1.0f32;
    for _ in 0..n {
        v *= 3.0;
    }
    v
}

/// Largest representable multiple of `Δ` at `t` planes: `(3^t − 1)/2`.
fn ladder_kmax(t: usize) -> i32 {
    (3_i32.saturating_pow(t.min(9) as u32) - 1) / 2
}

/// `2^(-j/4)`, the grid's step factor, built so a device mirror reproduces it **exactly**.
///
/// The obvious `(-(j as f32) / 4.0).exp2()` is not usable across host and device: CUDA's `exp2f` and
/// Rust's `f32::exp2` are each accurate to a couple of ULP but not to the same couple, and a 1-ULP
/// difference in `Δ` changes which grid candidate wins, which changes the whole fit. So the factor is
/// assembled from four exact quarter-octave constants and repeated exact halving — no transcendental
/// on either side, and every operation correctly rounded.
fn ladder_grid_factor(j: usize) -> f32 {
    /// `2^(-r/4)` for `r ∈ 0..4`, as f32.
    // These four f32 values must stay BIT-IDENTICAL to `QUARTER_OCTAVE` in
    // `crates/tritium-cuda/kernels/train_grad.cu`, or the device fitter picks a different grid
    // candidate than the host and the parity gate fails. `2^-1/2` is spelled as the std constant
    // because clippy recognises the literal; it is the same f32 (0x3f3504f3) either way.
    const QUARTER: [f32; 4] = [1.0, 0.8408964, core::f32::consts::FRAC_1_SQRT_2, 0.59460356];
    let mut f = QUARTER[j % 4];
    for _ in 0..(j / 4) {
        f *= 0.5; // exact: halving only decrements the exponent
    }
    f
}

/// Assign one group to the ladder with step `delta`, returning plane-major trits.
///
/// `k = clamp(round(w/Δ))` is the nearest lattice point — and because the ladder is a uniform grid,
/// nearest-lattice-point IS the minimum-SSE ternary assignment, exactly. No search, no iteration.
fn ladder_digits(bs: &[f32], t: usize, delta: f32) -> Vec<Vec<i8>> {
    let kmax = ladder_kmax(t);
    let mut planes = vec![vec![0i8; bs.len()]; t];
    for (i, &w) in bs.iter().enumerate() {
        let mut k = if delta > 0.0 && delta.is_finite() {
            (w / delta).round().clamp(-(kmax as f32), kmax as f32) as i32
        } else {
            0
        };
        // Balanced-ternary digits, most significant plane first: plane 0 carries 3^(t-1).
        for p in (0..t).rev() {
            let d = (k + 1).rem_euclid(3) - 1;
            planes[p][i] = d as i8;
            k = (k - d) / 3;
        }
    }
    planes
}

/// Sum of squared error of the ladder reconstruction at step `delta`, ascending order (ADR 0018).
fn ladder_sse(bs: &[f32], t: usize, delta: f32) -> f32 {
    let kmax = ladder_kmax(t) as f32;
    let mut e = 0.0f32;
    for &w in bs {
        let k = if delta > 0.0 && delta.is_finite() {
            (w / delta).round().clamp(-kmax, kmax)
        } else {
            0.0
        };
        let d = w - k * delta;
        e += d * d;
    }
    e
}

/// Fit one group to the ladder, returning `(s₀, plane-major trits)`.
///
/// The only free parameter is the step `Δ`, searched over a **deterministic** log grid:
/// `Δ_j = Δ₀·2^(-j/4)` for `j ∈ 0..grid`, where `Δ₀ = max|w| / kmax` is the exactly clipping-free
/// step. Increasing `j` trades range for resolution monotonically. Fixed candidate set, argmin SSE,
/// lowest `j` wins ties — so there is no data-dependent early exit to mirror on a device, unlike
/// [`fit_plane_itf`]'s accept guard.
///
/// `grid = 0` (or 1) uses `j = 0` alone: the clipping-free ladder, the cheapest useful configuration.
fn fit_group_geometric(bs: &[f32], t: usize, grid: usize) -> (f32, Vec<Vec<i8>>) {
    let t = t.max(1);
    let kmax = ladder_kmax(t) as f32;
    let peak = bs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak <= 0.0 || !peak.is_finite() {
        // Dead group: zero scale, zero trits — mirrors the greedy path's dead-row early-out.
        return (0.0, vec![vec![0i8; bs.len()]; t]);
    }
    let delta0 = peak / kmax;

    let mut best_delta = delta0;
    let mut best_sse = ladder_sse(bs, t, delta0);
    for j in 1..grid {
        let delta = delta0 * ladder_grid_factor(j);
        let sse = ladder_sse(bs, t, delta);
        if sse < best_sse {
            best_sse = sse;
            best_delta = delta;
        }
    }
    // s₀ is plane 0's scale; the stored anchor. Δ = s₀·3^-(T-1).
    (best_delta * pow3(t - 1), ladder_digits(bs, t, best_delta))
}

/// Reconstruct a group from `(s₀, trits)` — `Σ_p s₀·3^-p · trit_p`, ascending plane order.
fn ladder_reconstruct(s0: f32, planes: &[Vec<i8>], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (p, plane) in planes.iter().enumerate() {
        let s = s0 / pow3(p);
        for (o, &trit) in out.iter_mut().zip(plane) {
            *o += s * f32::from(trit);
        }
    }
    out
}

/// Fit one group with the ladder under `rotation`, returning `(reconstruction, s₀, trits)`.
///
/// Mirrors [`fit_group`]'s rotation handling exactly — including that a ragged (non-power-of-two)
/// group is never rotated rather than zero-padded, so no phantom weights are introduced. The trits
/// returned for a rotated group are the ones fitted **in the rotated basis**; that is what a packer
/// stores, and the reconstruction is rotated back.
fn fit_group_geometric_rotated(
    bs: &[f32],
    t: usize,
    grid: usize,
    rotation: RotationPolicy,
    buf: &mut Vec<f32>,
) -> (Vec<f32>, f32, Vec<Vec<i8>>, bool) {
    let sse = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
            .sum()
    };
    let rotatable = bs.len().is_power_of_two() && bs.len() > 1;
    let plain = || {
        let (s0, planes) = fit_group_geometric(bs, t, grid);
        let recon = ladder_reconstruct(s0, &planes, bs.len());
        (recon, s0, planes, false)
    };
    if rotation == RotationPolicy::Never || !rotatable {
        return plain();
    }

    buf.clear();
    buf.extend_from_slice(bs);
    fast_hadamard(buf);
    let (s0, planes) = fit_group_geometric(buf, t, grid);
    let mut back = ladder_reconstruct(s0, &planes, buf.len());
    fast_hadamard(&mut back); // H is its own inverse at normalized scale

    match rotation {
        RotationPolicy::Always => (back, s0, planes, true),
        _ => {
            let (p_recon, p_s0, p_planes, _) = plain();
            if sse(&back, bs) < sse(&p_recon, bs) {
                (back, s0, planes, true)
            } else {
                (p_recon, p_s0, p_planes, false)
            }
        }
    }
}

/// [`salt_quantize_forward_grouped`] on the **balanced-ternary ladder** — same shape, same output
/// layout, same rotation semantics; only the scale rule differs.
///
/// `grid` is the number of deterministic `Δ` candidates (see [`fit_group_geometric`]); `0` or `1`
/// means the single clipping-free step. `16` spans a 16× range and is the usual choice — the search
/// is load-bearing on heavy-tailed groups, where a clipping-free ladder wastes resolution on outliers.
#[must_use]
pub fn salt_quantize_forward_grouped_geometric(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    grid: usize,
    rotation: RotationPolicy,
) -> Vec<f32> {
    let group = group.max(1);
    let mut out = vec![0.0f32; wf.len()];
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (bs, bd) in src.chunks(group).zip(dst.chunks_mut(group)) {
            let (fit, _, _, _) = fit_group_geometric_rotated(bs, t, grid, rotation, &mut buf);
            bd.copy_from_slice(&fit);
        }
    }
    out
}

/// The per-group rotation decisions [`salt_quantize_forward_grouped_geometric`] would make, as one
/// byte per group (`1` = rotate), in the layout a device kernel indexes by
/// `row * cols.div_ceil(group) + block` — the ladder's counterpart to [`rotation_mask`].
///
/// A trainer must freeze the mask with the SAME fitter the kernel runs. Deriving it from the ITF
/// fitter and applying it to a ladder fit is the train/eval quantizer mismatch that invalidated a
/// published conclusion once already; the two fitters make genuinely different rotation choices
/// because they have different error curves.
#[must_use]
pub fn geometric_rotation_mask(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    grid: usize,
    rotation: RotationPolicy,
) -> Vec<u8> {
    let group = group.max(1);
    let mut out = Vec::with_capacity(rows * cols.div_ceil(group));
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        for bs in src.chunks(group) {
            let (_, _, _, rotated) = fit_group_geometric_rotated(bs, t, grid, rotation, &mut buf);
            out.push(u8::from(rotated));
        }
    }
    out
}

/// [`salt_quantize_forward_grouped_geometric`] with a **per-group plane count**.
///
/// The uniform-`T` fitter spends the same bits on every group regardless of how much the loss cares
/// about it. Rate-distortion allocation (`tritium_quantize::allocate_with_curves`) instead hands
/// each group a `T_g` by loss-drop-per-bit, so sensitive groups get resolution and inert ones do
/// not. This is the fitter side of that: the allocator decides `T_g`, this reconstructs it.
///
/// `planes` is one count per group in row-major group order — `row * cols.div_ceil(group) + block`,
/// the same layout as [`rotation_mask`]. A count of `0` reconstructs the group as all-zero (the
/// allocator is permitted to prune a tile to nothing); counts are otherwise unbounded here, with
/// the ladder's own `T ≤ 9` limit applying.
///
/// Passing a uniform count reproduces [`salt_quantize_forward_grouped_geometric`] bit-for-bit,
/// which is what makes an allocated arm comparable to a uniform baseline.
///
/// # Panics
/// If `planes.len()` is not `rows * cols.div_ceil(group)`.
#[must_use]
pub fn salt_quantize_forward_grouped_geometric_alloc(
    wf: &[f32],
    rows: usize,
    cols: usize,
    planes: &[u8],
    group: usize,
    grid: usize,
    rotation: RotationPolicy,
) -> Vec<f32> {
    let group = group.max(1);
    let per_row = cols.div_ceil(group);
    assert_eq!(
        planes.len(),
        rows * per_row,
        "geometric allocation must carry one plane count per group"
    );
    let mut out = vec![0.0f32; wf.len()];
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (b, (bs, bd)) in src.chunks(group).zip(dst.chunks_mut(group)).enumerate() {
            let t = usize::from(planes[r * per_row + b]);
            if t == 0 {
                continue; // pruned tile: already zero
            }
            let (fit, _, _, _) = fit_group_geometric_rotated(bs, t, grid, rotation, &mut buf);
            bd.copy_from_slice(&fit);
        }
    }
    out
}

/// The `(s₀, plane-major trits)` [`salt_quantize_forward_grouped_geometric`] produces, per group, in
/// row-major group order (`row * cols.div_ceil(group) + block`) — the layout a packer and the
/// diagnostics index by.
///
/// Goes through the same [`fit_group_geometric_rotated`] as the dense forward, so the symbols a
/// packer writes cannot drift from the reconstruction training optimised against.
#[must_use]
pub fn geometric_ladder_fit(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    group: usize,
    grid: usize,
    rotation: RotationPolicy,
) -> Vec<(f32, Vec<Vec<i8>>)> {
    let group = group.max(1);
    let mut out = Vec::with_capacity(rows * cols.div_ceil(group));
    let mut buf: Vec<f32> = Vec::with_capacity(group);
    for r in 0..rows {
        let src = &wf[r * cols..(r + 1) * cols];
        for bs in src.chunks(group) {
            let (_, s0, planes, _) = fit_group_geometric_rotated(bs, t, grid, rotation, &mut buf);
            out.push((s0, planes));
        }
    }
    out
}

// ── ITF: iterative ternary fitting ───────────────────────────────────────────────────────────────
// [`salt_quantize_forward`] fits each plane in ONE greedy pass: take `s = AbsMean(residual)`, then
// `t = clamp(round(residual/s))`. AbsMean is a heuristic (the flat BitNet b1.58 contract) — it is *not*
// the scale that minimizes the plane's reconstruction error.
//
// For a fixed trit vector `t`, the least-squares optimal scalar is the projection
//
//     s* = <r, t> / <t, t>        (NOT mean|r|)
//
// and for a fixed `s`, the optimal ternary assignment is exactly `clamp(round(r/s))`. Both half-steps
// are therefore *exact* minimizations of the same objective `||r − s·t||²`, so alternating them is
// monotone non-increasing — the PT²-LLM "Iterative Ternary Fitting" idea, specialized to a SALT plane.
// The output is an ordinary `(scale, trits)` pair, so the packed format is unchanged: this is a better
// fit at identical bits, not a new representation.
//
// NOTE: better weight-space reconstruction does not automatically mean better end-task quality (the
// documented "proxy gap": per-layer MSE can anti-correlate with perplexity). ITF must be judged on
// downstream ppl, not on the MSE it is guaranteed to improve.

/// Least-squares optimal scale for a fixed ternary assignment: `<r,t>/<t,t>`, or `None` if `t` is
/// all-zero (no information) or the projection is non-positive (degenerate).
/// Both sums accumulate in ascending order so a device mirror can reproduce them exactly (ADR 0018).
fn ls_optimal_scale(residual: &[f32], trits: &[f32]) -> Option<f32> {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (&r, &t) in residual.iter().zip(trits) {
        num += r * t;
        den += t * t;
    }
    if den <= 0.0 {
        return None;
    }
    let s = num / den;
    (s > 0.0 && s.is_finite()).then_some(s)
}

/// Sum of squared reconstruction error for one plane, `Σ (r − s·t)²`, ascending order.
fn plane_sse(residual: &[f32], trits: &[f32], scale: f32) -> f32 {
    let mut e = 0.0f32;
    for (&r, &t) in residual.iter().zip(trits) {
        let d = r - scale * t;
        e += d * d;
    }
    e
}

/// Fit ONE plane to `residual` by iterative ternary fitting, returning `(scale, trits)`.
///
/// Starts from the AbsMean fit (so `iters == 0` is exactly the greedy behaviour) and alternates
/// `scale ← <r,t>/<t,t>` with `trits ← clamp(round(r/scale))`, keeping a candidate **only when it
/// strictly reduces** the plane's squared error. That accept-on-improvement guard makes the result
/// provably never worse than AbsMean, even under float noise.
fn fit_plane_itf(residual: &[f32], iters: usize) -> (f32, Vec<f32>) {
    let quantize = |s: f32| -> Vec<f32> {
        residual
            .iter()
            .map(|&r| (r / s).round().clamp(-1.0, 1.0))
            .collect()
    };
    let mut scale = residual.iter().map(|&v| v.abs()).sum::<f32>() / residual.len() as f32;
    if scale <= 0.0 || !scale.is_finite() {
        return (0.0, vec![0.0; residual.len()]); // dead row: matches the greedy path's early-out
    }
    let mut trits = quantize(scale);
    let mut best_sse = plane_sse(residual, &trits, scale);
    for _ in 0..iters {
        let Some(cand_scale) = ls_optimal_scale(residual, &trits) else {
            break;
        };
        let cand_trits = quantize(cand_scale);
        let cand_sse = plane_sse(residual, &cand_trits, cand_scale);
        if cand_sse >= best_sse {
            break; // converged (or float noise) — keep the better pair
        }
        scale = cand_scale;
        trits = cand_trits;
        best_sse = cand_sse;
    }
    (scale, trits)
}

/// [`salt_quantize_forward`] with **iterative ternary fitting** per plane. `iters == 0` reproduces the
/// greedy AbsMean expansion bit-for-bit; higher values refine each plane's scale toward the
/// least-squares optimum before the residual is passed to the next plane.
///
/// Returns the same dense reconstruction `Ŵ = Σ_p s_p·trit_p` in the same layout, so every downstream
/// consumer (packing, the STE backward, the device kernels) is unchanged.
#[must_use]
pub fn salt_quantize_forward_itf(
    wf: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    iters: usize,
) -> Vec<f32> {
    if iters == 0 {
        return salt_quantize_forward(wf, rows, cols, t);
    }
    let mut residual = wf.to_vec();
    let mut recon = vec![0.0f32; rows * cols];
    for _plane in 0..t {
        for r in 0..rows {
            let lo = r * cols;
            let hi = lo + cols;
            let (scale, trits) = fit_plane_itf(&residual[lo..hi], iters);
            if scale == 0.0 {
                continue;
            }
            for (c, tr) in trits.iter().enumerate() {
                let contribution = scale * tr;
                recon[lo + c] += contribution;
                residual[lo + c] -= contribution;
            }
        }
    }
    recon
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
