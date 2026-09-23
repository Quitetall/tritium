//! Activation-aware SALT fitting: GPTQ error compensation, plus an optional discrete search over
//! the trit move set and closed-form step refits.
//!
//! The shipping converter rounds each group to the nearest ladder point, minimizing `‖W − Ŵ‖²`. The
//! loss does not see `W`; it sees `W·x`. Fitting in the activation metric `‖(W − Ŵ)·X‖²` — and, more
//! importantly, *acting* on its off-diagonal by pushing each quantized column's residual into the
//! columns not yet quantized — is worth **−0.49%** perplexity at identical bits on SmolLM2-135M, and
//! **−0.64%** with the discrete search and step refits on top.
//!
//! Everything here emits `(s₀, plane-major trits)` per group: the same shape
//! [`tritium_train::ops::ste::geometric_ladder_fit`] returns, so the existing packer consumes it
//! unchanged and the artifact is byte-for-byte the same container.
//!
//! # What the measurements say about using this
//!
//! - **Calibration size binds it.** GPTQ reaches 0.236× of round-to-nearest's objective on the Gram
//!   it fitted and only 0.459× on unseen tokens. Give it as many tokens as can be afforded.
//! - **The discrete search alone overfits** (−0.37%, worse than GPTQ). It only pays with
//!   [`ActivationAwareConfig::refit_scale`], where continuous parameters absorb what discrete moves
//!   chase.
//! - **A Gram narrower than the input width is rank-deficient by construction.** `down_proj` is 1536
//!   wide; a Gram from 1,024 tokens made GPTQ *lose* by 2.98%.

use half::f16;
use rayon::prelude::*;
use tritium_train::Tape;
use tritium_train::ops::ste::{fast_hadamard, group_is_rotatable, ladder_quantize_at, ladder_step};
use tritium_train::tape::ValueId;

use crate::calibrate::{Arch, Tap, forward_aq};

/// How to fit one tensor in the activation metric.
#[derive(Clone, Copy, Debug)]
pub struct ActivationAwareConfig {
    /// Ternary planes per weight.
    pub planes: usize,
    /// Weights sharing one ladder anchor.
    pub group: usize,
    /// Step candidates the ladder searches per group.
    pub grid: usize,
    /// Tikhonov damping as a fraction of `mean(diag H)`. A calibration Gram is routinely singular;
    /// 0.01 is the GPTQ paper's value and needs raising when tokens are scarce.
    pub damp: f64,
    /// Coordinate-descent sweeps over the trit moves. `0` is plain GPTQ.
    pub search_sweeps: usize,
    /// Alternate the search with closed-form refits of each group's step. Without this the search
    /// overfits the calibration Gram and loses to plain GPTQ.
    pub refit_scale: bool,
    /// Fit in the Hadamard-rotated basis, which is what the shipping artifact stores.
    pub rotate: bool,
    /// Fraction of each column's rounding error that GPTQ pushes into the columns after it.
    /// `1.0` is plain GPTQ. See [`auto_decay`] for why anything less can be better.
    pub decay: f64,
    /// Ramp the decay over the column order: the first column propagates in full and `decay` is
    /// reached only at the last, where a short remaining suffix has to absorb everything pushed
    /// into it (ADR 0043 L-B, after QTEA). Off, the same `decay` applies to every column.
    pub decay_ramp: bool,
}

impl Default for ActivationAwareConfig {
    fn default() -> Self {
        Self {
            planes: 3,
            group: 256,
            grid: 16,
            damp: 0.01,
            search_sweeps: 8,
            refit_scale: true,
            rotate: true,
            decay: 1.0,
            decay_ramp: false,
        }
    }
}

/// Propagation decay for a Gram estimated from `calibration_tokens` samples of a `cols`-wide input.
///
/// GPTQ trusts `H⁻¹` completely: every column's rounding error is pushed in full into the columns
/// after it. A Gram from few samples relative to its width is rank-deficient, and then what gets
/// pushed is mostly the damping's guess, so trusting it less helps. Measured on SmolLM2-135M over
/// WikiText-2 (folded, T=2/T=3, held-out perplexity against the plain fit), the best fraction
/// falls as tokens fall:
///
/// ```text
/// tokens   width   tokens/width   best λ    plain GPTQ → decayed
/// 16,384   1,536      10.7        0.75      −13.1% → −13.8%   (T=2)
///  4,096   1,536       2.7        ≤0.5       −2.5% →  −9.8%
///  2,048   1,536       1.3        ≤0.5       +4.6% →  −7.4%
/// ```
///
/// This interpolates `λ` linearly in `log(tokens/width)` between those two measured ends, `0.5` at
/// a ratio of 2.7 and `0.75` at 10.7, and clamps outside them. It is a rule fitted to three points
/// on one model, not a derivation; a wider sweep should replace the constants, not the shape.
#[must_use]
pub fn auto_decay(calibration_tokens: usize, cols: usize) -> f64 {
    const LOW: (f64, f64) = (2.7, 0.5);
    const HIGH: (f64, f64) = (10.7, 0.75);
    if calibration_tokens == 0 || cols == 0 {
        return LOW.1;
    }
    let ratio = calibration_tokens as f64 / cols as f64;
    let t = ((ratio.ln() - LOW.0.ln()) / (HIGH.0.ln() - LOW.0.ln())).clamp(0.0, 1.0);
    LOW.1 + t * (HIGH.1 - LOW.1)
}

/// Input Gram `E[x·xᵀ]` at each of the four projection inputs, per layer.
///
/// `q`, `k` and `v` share the attention-norm output; `gate` and `up` share the FFN-norm output;
/// `down` sees the FFN intermediate; `o` sees the concatenated attention heads. The tied head has no
/// Gram here — it is an embedding as well as a projection, so a fit that suits one corrupts the
/// other.
#[derive(Debug, Clone)]
pub struct TapGrams {
    /// Per layer, `n_embd × n_embd`.
    pub attn: Vec<Vec<f64>>,
    /// Per layer, `n_embd × n_embd`.
    pub ffn: Vec<Vec<f64>>,
    /// Per layer, `ff × ff`.
    pub down: Vec<Vec<f64>>,
    /// Per layer, `(n_head·head_dim)²`.
    pub o: Vec<Vec<f64>>,
    /// Tokens accumulated.
    pub rows: usize,
}

fn accumulate(h: &mut [f64], k: usize, act: &[f32], seq: usize) {
    h.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for r in 0..seq {
            let x = &act[r * k..(r + 1) * k];
            let xi = f64::from(x[i]);
            if xi == 0.0 {
                continue;
            }
            for (j, hij) in row.iter_mut().enumerate().skip(i) {
                *hij += xi * f64::from(x[j]);
            }
        }
    });
}

fn finish(h: &mut [f64], k: usize, rows: usize) {
    for i in 0..k {
        for j in 0..i {
            h[i * k + j] = h[j * k + i];
        }
    }
    let n = rows.max(1) as f64;
    for v in h.iter_mut() {
        *v /= n;
    }
}

impl TapGrams {
    /// Collect every tap's Gram from `weights`, one forward per window.
    ///
    /// The `O(seq·k²)` accumulation is the cost here, not the forwards, which is why it is parallel:
    /// the sample counts this fit needs are otherwise out of reach.
    #[must_use]
    pub fn collect(weights: &[Vec<f32>], arch: &Arch, windows: &[&[u32]]) -> Self {
        let q_width = arch.n_head * arch.head_dim;
        let mut out = Self {
            attn: vec![vec![0.0; arch.n_embd * arch.n_embd]; arch.n_layers],
            ffn: vec![vec![0.0; arch.n_embd * arch.n_embd]; arch.n_layers],
            down: vec![vec![0.0; arch.ff * arch.ff]; arch.n_layers],
            o: vec![vec![0.0; q_width * q_width]; arch.n_layers],
            rows: 0,
        };
        for tokens in windows {
            let mut tape = Tape::new();
            let wids: Vec<ValueId> = weights.iter().map(|w| tape.leaf(w.clone())).collect();
            forward_aq(
                &mut tape,
                &wids,
                arch,
                tokens,
                &mut |kind, li, v, seq, cols| match kind {
                    Tap::AttnIn => accumulate(&mut out.attn[li], cols, v, seq),
                    Tap::FfnIn => accumulate(&mut out.ffn[li], cols, v, seq),
                    Tap::DownIn => accumulate(&mut out.down[li], cols, v, seq),
                    Tap::OProjIn => accumulate(&mut out.o[li], cols, v, seq),
                    Tap::Head => {}
                },
            );
            out.rows += tokens.len();
        }
        for li in 0..arch.n_layers {
            finish(&mut out.attn[li], arch.n_embd, out.rows);
            finish(&mut out.ffn[li], arch.n_embd, out.rows);
            finish(&mut out.down[li], arch.ff, out.rows);
            finish(&mut out.o[li], q_width, out.rows);
        }
        out
    }

    /// The Gram for slot `slot` of layer `li`, in `extract`'s order: q, k, v, o, gate, up, down.
    #[must_use]
    pub fn for_slot(&self, li: usize, slot: usize) -> &[f64] {
        match slot {
            0..=2 => &self.attn[li],
            3 => &self.o[li],
            4..=5 => &self.ffn[li],
            _ => &self.down[li],
        }
    }
}

fn rotate_row(row: &mut [f32], group: usize) {
    for slice in row.chunks_mut(group) {
        if group_is_rotatable(slice.len()) {
            fast_hadamard(slice);
        }
    }
}

/// `fast_hadamard` in f64. The Gram is a metric and its inverse is Cholesky-factored, so rounding it
/// to f32 to reuse the f32 transform would throw away half the digits of the very matrix being
/// factorized.
fn hadamard_f64(v: &mut [f64]) {
    let n = v.len();
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
    let scale = 1.0 / (n as f64).sqrt();
    for x in v.iter_mut() {
        *x *= scale;
    }
}

fn rotate_row_f64(row: &mut [f64], group: usize) {
    for slice in row.chunks_mut(group) {
        if group_is_rotatable(slice.len()) {
            hadamard_f64(slice);
        }
    }
}

/// `H ← R·H·Rᵀ`, `R` block-diagonal Hadamard over scale groups, so the metric is expressed in the
/// basis the codes are stored in.
fn rotate_gram(h: &mut [f64], k: usize, group: usize) {
    h.par_chunks_mut(k)
        .for_each(|row| rotate_row_f64(row, group));
    let mut col = vec![0.0f64; k];
    for c in 0..k {
        for (r, v) in col.iter_mut().enumerate() {
            *v = h[r * k + c];
        }
        rotate_row_f64(&mut col, group);
        for (r, &v) in col.iter().enumerate() {
            h[r * k + c] = v;
        }
    }
}

/// Lower Cholesky `A = L·Lᵀ` of `H + λI`, `λ = damp·mean(diag H)`.
fn damped_cholesky(h: &[f64], k: usize, damp: f64) -> Option<Vec<f64>> {
    let mean: f64 = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    let lambda = damp * mean.max(1e-12);
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = h[i * k + j] + if i == j { lambda } else { 0.0 };
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum <= 0.0 || sum.is_nan() {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    Some(l)
}

/// `H⁻¹` from `H + λI`, and the lower Cholesky of that inverse. GPTQ's compensation reads the second.
fn damped_inverse_cholesky(h: &[f64], k: usize, damp: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let l = damped_cholesky(h, k, damp)?;
    let mut inv_l = vec![0.0f64; k * k];
    for i in 0..k {
        inv_l[i * k + i] = 1.0 / l[i * k + i];
        for j in 0..i {
            let mut sum = 0.0;
            for p in j..i {
                sum += l[i * k + p] * inv_l[p * k + j];
            }
            inv_l[i * k + j] = -sum / l[i * k + i];
        }
    }
    let mut hinv = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in i.max(j)..k {
                sum += inv_l[p * k + i] * inv_l[p * k + j];
            }
            hinv[i * k + j] = sum;
            hinv[j * k + i] = sum;
        }
    }
    let chol = damped_cholesky(&hinv, k, 0.0)?;
    Some((hinv, chol))
}

/// Balanced-ternary digits of `k`, most significant plane first — the ladder's own encoding.
fn digits_of(mut value: i64, planes: usize) -> Vec<i8> {
    let mut out = vec![0i8; planes];
    for p in (0..planes).rev() {
        let d = (value + 1).rem_euclid(3) - 1;
        out[p] = d as i8;
        value = (value - d) / 3;
    }
    out
}

/// Fit one tensor in the activation metric, returning `(s₀, plane-major trits)` per group in
/// row-major group order — the shape the ladder packer already consumes.
///
/// `gram` is the tensor's input Gram in the model's basis, `cols × cols`.
///
/// Returns `None` if the Gram stays singular after damping, so a caller can fall back to the plain
/// ladder rather than propagate a bad fit.
#[must_use]
pub fn fit_tensor(
    w: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f64],
    cfg: &ActivationAwareConfig,
) -> Option<Vec<(f32, Vec<Vec<i8>>)>> {
    let group = cfg.group.max(1);
    let per_row = cols.div_ceil(group);
    let kmax = (3i64.pow(cfg.planes as u32) - 1) / 2;

    let mut work: Vec<f32> = w.to_vec();
    if cfg.rotate {
        for row in work.chunks_mut(cols) {
            rotate_row(row, group);
        }
    }
    let mut h = gram.to_vec();
    if cfg.rotate {
        rotate_gram(&mut h, cols, group);
    }
    let mean: f64 = (0..cols).map(|a| h[a * cols + a]).sum::<f64>() / cols as f64;
    let mut h_damped = h.clone();
    for a in 0..cols {
        h_damped[a * cols + a] += cfg.damp * mean.max(1e-12);
    }
    let (_, chol) = damped_inverse_cholesky(&h, cols, cfg.damp)?;

    // ── GPTQ: quantize each column, push its residual into the columns still to come.
    let mut quantized = vec![0.0f32; rows * cols];
    let mut steps = vec![0.0f32; rows * per_row];
    for j in 0..cols {
        let block = j / group;
        if j % group == 0 {
            let end = ((block + 1) * group).min(cols);
            steps
                .par_chunks_mut(per_row)
                .zip(work.par_chunks(cols))
                .for_each(|(d, wr)| d[block] = ladder_step(&wr[j..end], cfg.planes, cfg.grid));
        }
        let d_jj = chol[j * cols + j];
        if d_jj <= 0.0 || !d_jj.is_finite() {
            return None;
        }
        // Decay on the propagated error: a multiplier on `err`, so `lambda == 1.0` is plain GPTQ
        // to the bit. Under the ramp it runs from 1 at the first column to `cfg.decay` at the last.
        let lambda = if cfg.decay_ramp && cols > 1 {
            1.0 - (1.0 - cfg.decay) * j as f64 / (cols - 1) as f64
        } else {
            cfg.decay
        };
        quantized
            .par_chunks_mut(cols)
            .zip(work.par_chunks_mut(cols))
            .zip(steps.par_chunks(per_row))
            .for_each(|((q, wr), d)| {
                let value = ladder_quantize_at(wr[j], cfg.planes, d[block]);
                q[j] = value;
                let err = lambda * f64::from(wr[j] - value) / d_jj;
                for j2 in (j + 1)..cols {
                    wr[j2] -= (err * chol[j2 * cols + j]) as f32;
                }
            });
    }

    // ── Codes, and the optional discrete search over them.
    let moves: Vec<i64> = (0..cfg.planes)
        .flat_map(|p| {
            let m = 3i64.pow(p as u32);
            [m, -m]
        })
        .collect();
    let mut w_rot: Vec<f32> = w.to_vec();
    if cfg.rotate {
        for row in w_rot.chunks_mut(cols) {
            rotate_row(row, group);
        }
    }
    let planes = cfg.planes;
    let sweeps = cfg.search_sweeps;
    let refit = cfg.refit_scale;
    let out: Vec<Vec<(f32, Vec<Vec<i8>>)>> = w_rot
        .par_chunks(cols)
        .zip(quantized.par_chunks(cols))
        .zip(steps.par_chunks(per_row))
        .map(|((wr, qr), dr)| {
            let mut d: Vec<f64> = dr.iter().map(|&v| f64::from(v)).collect();
            let mut codes: Vec<i64> = (0..cols)
                .map(|j| {
                    let dj = d[j / group];
                    if dj > 0.0 {
                        (f64::from(qr[j]) / dj).round() as i64
                    } else {
                        0
                    }
                })
                .collect();
            if sweeps > 0 {
                search_row(
                    wr, &mut codes, &mut d, cols, group, &h_damped, &moves, kmax, sweeps, refit,
                );
            }
            (0..per_row)
                .map(|b| {
                    let (lo, hi) = (b * group, ((b + 1) * group).min(cols));
                    let mut trits = vec![vec![0i8; hi - lo]; planes];
                    for (i, j) in (lo..hi).enumerate() {
                        for (p, digit) in digits_of(codes[j], planes).into_iter().enumerate() {
                            trits[p][i] = digit;
                        }
                    }
                    // s₀ = Δ·3^(T−1): plane 0 carries the most significant digit.
                    let mut s0 = d[b];
                    for _ in 0..planes.saturating_sub(1) {
                        s0 *= 3.0;
                    }
                    (s0 as f32, trits)
                })
                .collect()
        })
        .collect();
    Some(out.into_iter().flatten().collect())
}

/// Coordinate descent over one row's codes, priced exactly from `g = H·r`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn search_row(
    w: &[f32],
    codes: &mut [i64],
    d: &mut [f64],
    cols: usize,
    group: usize,
    h: &[f64],
    moves: &[i64],
    kmax: i64,
    sweeps: usize,
    refit: bool,
) {
    let per_row = cols.div_ceil(group);
    let mut r: Vec<f64> = (0..cols)
        .map(|j| f64::from(w[j]) - d[j / group] * codes[j] as f64)
        .collect();
    let mut g: Vec<f64> = (0..cols)
        .map(|a| {
            h[a * cols..(a + 1) * cols]
                .iter()
                .zip(&r)
                .map(|(x, y)| x * y)
                .sum()
        })
        .collect();
    for _ in 0..sweeps {
        let mut improved = false;
        for j in 0..cols {
            let dj = d[j / group];
            if dj <= 0.0 {
                continue;
            }
            let hjj = h[j * cols + j];
            let mut best = (0.0f64, 0i64);
            for &m in moves {
                if (codes[j] + m).abs() > kmax {
                    continue;
                }
                let delta = -dj * m as f64;
                let change = 2.0 * delta * g[j] + delta * delta * hjj;
                if change < best.0 {
                    best = (change, m);
                }
            }
            if best.1 != 0 && best.0 < -1e-15 {
                let delta = -dj * best.1 as f64;
                codes[j] += best.1;
                r[j] += delta;
                for (a, ga) in g.iter_mut().enumerate() {
                    *ga += delta * h[a * cols + j];
                }
                improved = true;
            }
        }
        if refit {
            for b in 0..per_row {
                let (lo, hi) = (b * group, ((b + 1) * group).min(cols));
                let num: f64 = (lo..hi).map(|j| codes[j] as f64 * g[j]).sum();
                let mut den = 0.0f64;
                for a in lo..hi {
                    if codes[a] == 0 {
                        continue;
                    }
                    for c in lo..hi {
                        den += codes[a] as f64 * h[a * cols + c] * codes[c] as f64;
                    }
                }
                if den <= 0.0 {
                    continue;
                }
                let eps = num / den;
                if d[b] + eps <= 0.0 {
                    continue;
                }
                d[b] += eps;
                for j in lo..hi {
                    r[j] -= eps * codes[j] as f64;
                }
                for (a, ga) in g.iter_mut().enumerate() {
                    let hrow = &h[a * cols..(a + 1) * cols];
                    let s: f64 = (lo..hi).map(|c| hrow[c] * codes[c] as f64).sum();
                    *ga -= eps * s;
                }
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
}

/// Dense reconstruction of a fit, for scoring. `f16` rounding of the stored scales is applied, so
/// this is what the artifact will decode to, not what the fitter intended.
#[must_use]
pub fn fit_to_dense(
    fits: &[(f32, Vec<Vec<i8>>)],
    rows: usize,
    cols: usize,
    group: usize,
    rotate: bool,
) -> Vec<f32> {
    let per_row = cols.div_ceil(group);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for b in 0..per_row {
            let (s0, trits) = &fits[r * per_row + b];
            let lo = b * group;
            for (p, plane) in trits.iter().enumerate() {
                let mut scale = f64::from(f16::from_f32(*s0).to_f32());
                for _ in 0..p {
                    scale /= 3.0;
                }
                for (i, &t) in plane.iter().enumerate() {
                    out[r * cols + lo + i] += (scale * f64::from(t)) as f32;
                }
            }
        }
        if rotate {
            rotate_row(&mut out[r * cols..(r + 1) * cols], group);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_train::ops::ste::{self, RotationPolicy};

    /// The degenerate control. With `H = I` the compensation term vanishes — the Cholesky of a
    /// scaled identity is diagonal — so GPTQ reduces to rounding each column on the ladder's own
    /// grid, which is exactly what the plain fitter does. The codes must match it group for group.
    #[test]
    fn an_identity_gram_reproduces_the_plain_ladder_fit() {
        let (rows, cols, planes, group) = (3usize, 256usize, 3usize, 128usize);
        let mut s = 0x1357_9BDFu64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        for i in 0..cols {
            gram[i * cols + i] = 1.0;
        }
        for rotate in [false, true] {
            let cfg = ActivationAwareConfig {
                planes,
                group,
                grid: 16,
                damp: 1e-9,
                search_sweeps: 0,
                refit_scale: false,
                rotate,
                decay: 1.0,
                decay_ramp: false,
            };
            let fits = fit_tensor(&w, rows, cols, &gram, &cfg).expect("fit");
            let oracle = ste::geometric_ladder_fit(
                &w,
                rows,
                cols,
                planes,
                group,
                16,
                if rotate {
                    RotationPolicy::Always
                } else {
                    RotationPolicy::Never
                },
            );
            assert_eq!(fits.len(), oracle.len());
            for (i, ((s_a, t_a), (s_b, t_b))) in fits.iter().zip(&oracle).enumerate() {
                assert!(
                    (s_a - s_b).abs() <= 1e-5 * s_b.abs().max(1e-6),
                    "rotate={rotate} group {i}: anchor {s_a} vs {s_b}"
                );
                assert_eq!(t_a, t_b, "rotate={rotate} group {i}: digits differ");
            }
        }
    }

    /// The search may never raise the objective it minimizes, and on a real (anisotropic) Gram it
    /// must find something to lower.
    #[test]
    fn the_search_lowers_the_activation_objective() {
        let (rows, cols, planes, group) = (4usize, 256usize, 3usize, 128usize);
        let mut s = 0x2468_ACE0u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f64 / 8_388_608.0) - 1.0
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| next() as f32).collect();
        let m = 800;
        let x: Vec<f32> = (0..m * cols)
            .map(|i| (next() * (1.0 + 5.0 * f64::from(u8::from(i % cols % 9 == 0)))) as f32)
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        accumulate(&mut gram, cols, &x, m);
        finish(&mut gram, cols, m);
        let mut hd = gram.clone();
        let mean: f64 = (0..cols).map(|a| hd[a * cols + a]).sum::<f64>() / cols as f64;
        for a in 0..cols {
            hd[a * cols + a] += 0.01 * mean;
        }
        let objective = |q: &[f32]| -> f64 {
            (0..rows)
                .map(|r| {
                    let e: Vec<f64> = (0..cols)
                        .map(|j| f64::from(w[r * cols + j] - q[r * cols + j]))
                        .collect();
                    (0..cols)
                        .map(|a| e[a] * (0..cols).map(|b| hd[a * cols + b] * e[b]).sum::<f64>())
                        .sum::<f64>()
                })
                .sum()
        };
        let base = ActivationAwareConfig {
            planes,
            group,
            grid: 16,
            damp: 0.01,
            search_sweeps: 0,
            refit_scale: false,
            rotate: false,
            decay: 1.0,
            decay_ramp: false,
        };
        let gptq = fit_to_dense(
            &fit_tensor(&w, rows, cols, &gram, &base).unwrap(),
            rows,
            cols,
            group,
            false,
        );
        let searched = fit_to_dense(
            &fit_tensor(
                &w,
                rows,
                cols,
                &gram,
                &ActivationAwareConfig {
                    search_sweeps: 8,
                    refit_scale: true,
                    ..base
                },
            )
            .unwrap(),
            rows,
            cols,
            group,
            false,
        );
        let (a, b) = (objective(&gptq), objective(&searched));
        println!("activation objective: gptq {a:.6e} search+refit {b:.6e}");
        assert!(
            b < a,
            "the search did not lower the objective: {b:.6e} vs {a:.6e}"
        );
    }

    /// A tap that saw no signal must degrade to the plain ladder, not to a wrong fit: with an
    /// all-zero Gram the damping leaves a multiple of the identity, whose compensation term is zero.
    /// A Gram with non-finite entries is a different matter and must be refused.
    #[test]
    fn a_dead_gram_degrades_to_the_ladder_and_a_broken_one_is_refused() {
        let (rows, cols, planes, group) = (2usize, 256usize, 3usize, 128usize);
        let mut s = 0xFACE_B00Cu64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let cfg = ActivationAwareConfig {
            planes,
            group,
            grid: 16,
            damp: 0.01,
            search_sweeps: 0,
            refit_scale: false,
            rotate: false,
            decay: 1.0,
            decay_ramp: false,
        };
        let dead = fit_tensor(&w, rows, cols, &vec![0.0f64; cols * cols], &cfg).expect("dead gram");
        let oracle =
            ste::geometric_ladder_fit(&w, rows, cols, planes, group, 16, RotationPolicy::Never);
        for (i, ((s_a, t_a), (s_b, t_b))) in dead.iter().zip(&oracle).enumerate() {
            assert!(
                (s_a - s_b).abs() <= 1e-5 * s_b.abs().max(1e-6),
                "group {i} anchor"
            );
            assert_eq!(t_a, t_b, "group {i} digits");
        }
        let mut broken = vec![0.0f64; cols * cols];
        broken[0] = f64::NAN;
        assert!(fit_tensor(&w, rows, cols, &broken, &cfg).is_none());
    }
}

#[cfg(test)]
mod decay_tests {
    use super::*;

    fn synthetic(rows: usize, cols: usize) -> (Vec<f32>, Vec<f64>) {
        // Deterministic heavy-tailed weights and a full-rank, well-conditioned Gram.
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let x = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
                x * x * x + 0.05 * x
            })
            .collect();
        let mut gram = vec![0.0f64; cols * cols];
        for a in 0..cols {
            for b in 0..cols {
                gram[a * cols + b] =
                    0.3f64.powi((a as i32 - b as i32).abs()) * (1.0 + a as f64 / cols as f64);
            }
        }
        (w, gram)
    }

    #[test]
    fn a_decay_of_one_is_plain_gptq_with_or_without_the_ramp() {
        let (w, gram) = synthetic(4, 128);
        let base = ActivationAwareConfig {
            planes: 2,
            group: 64,
            grid: 8,
            search_sweeps: 0,
            rotate: false,
            ..ActivationAwareConfig::default()
        };
        let plain = fit_tensor(&w, 4, 128, &gram, &base).expect("fit");
        let ramped = fit_tensor(
            &w,
            4,
            128,
            &gram,
            &ActivationAwareConfig {
                decay_ramp: true,
                ..base
            },
        )
        .expect("fit");
        assert_eq!(plain, ramped, "at decay 1.0 the ramp must change nothing");
    }

    #[test]
    fn a_decay_below_one_changes_the_propagation() {
        let (w, gram) = synthetic(4, 128);
        let base = ActivationAwareConfig {
            planes: 2,
            group: 64,
            grid: 8,
            search_sweeps: 0,
            rotate: false,
            ..ActivationAwareConfig::default()
        };
        let plain = fit_tensor(&w, 4, 128, &gram, &base).expect("fit");
        let decayed = fit_tensor(
            &w,
            4,
            128,
            &gram,
            &ActivationAwareConfig { decay: 0.5, ..base },
        )
        .expect("fit");
        assert_ne!(
            plain, decayed,
            "decay 0.5 must alter the codes GPTQ produces"
        );
    }

    #[test]
    fn auto_decay_follows_the_measured_ends_and_clamps_outside_them() {
        // The two measured ends, on down_proj's 1,536-wide input.
        assert!((auto_decay(4_096, 1_536) - 0.5).abs() < 0.01);
        assert!((auto_decay(16_384, 1_536) - 0.75).abs() < 0.01);
        // Clamped past them.
        assert_eq!(auto_decay(1_024, 1_536), 0.5);
        assert_eq!(auto_decay(1 << 20, 1_536), 0.75);
        // Monotone non-decreasing in tokens at fixed width.
        let mut last = 0.0;
        for tokens in [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768] {
            let d = auto_decay(tokens, 576);
            assert!(d >= last, "auto_decay must not fall as tokens grow");
            last = d;
        }
        // Degenerate inputs pick the cautious end rather than NaN.
        assert_eq!(auto_decay(0, 576), 0.5);
        assert_eq!(auto_decay(4_096, 0), 0.5);
    }
}
