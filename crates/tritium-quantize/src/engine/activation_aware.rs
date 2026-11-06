//! Shipping f32 working-weight / f64 metric activation-aware fitting.
//! Legacy group-tail rotation is not admission of a typed Hadamard basis.

use super::ladder::{ladder_quantize_at, ladder_step};
use half::f16;
use rayon::prelude::*;
use tritium_core::{fast_hadamard, fast_hadamard_f64, group_is_rotatable};

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

/// Placement of the optional one-scalar-per-group ladder scale refit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleRefitMode {
    /// Keep the existing coordinate-search refit after GPTQ propagation.
    PostPass,
    /// Refit each completed group before later columns are propagated.
    InLoop,
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
    tritium_core::auto_feedback_decay(calibration_tokens, cols)
}

fn rotate_row(row: &mut [f32], group: usize) {
    for slice in row.chunks_mut(group) {
        if group_is_rotatable(slice.len()) {
            fast_hadamard(slice);
        }
    }
}

fn rotate_row_f64(row: &mut [f64], group: usize) {
    for slice in row.chunks_mut(group) {
        if group_is_rotatable(slice.len()) {
            fast_hadamard_f64(slice);
        }
    }
}

/// `H ← R·H·Rᵀ`, `R` block-diagonal Hadamard over scale groups, so the metric is expressed in the
/// basis the codes are stored in.
pub(super) fn rotate_gram(h: &mut [f64], k: usize, group: usize) {
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
    let hinv = damped_inverse(h, k, damp)?;
    let chol = damped_cholesky(&hinv, k, 0.0)?;
    Some((hinv, chol))
}

/// Shared full-Gram inverse; group feedback does not need the ladder's extra Cholesky.
pub(super) fn damped_inverse(h: &[f64], k: usize, damp: f64) -> Option<Vec<f64>> {
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
    Some(hinv)
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
    fit_tensor_with_scale_refit_mode(w, rows, cols, gram, cfg, ScaleRefitMode::PostPass)
}

/// Fit a tensor with an explicit placement for the group-scale refit.
///
/// `InLoop` refits one completed group at a time and pushes the scale change through the same
/// inverse-Cholesky compensation used by GPTQ. The public [`ActivationAwareConfig`] and the
/// behavior of [`fit_tensor`] remain source-compatible and unchanged.
#[must_use]
pub fn fit_tensor_with_scale_refit_mode(
    w: &[f32],
    rows: usize,
    cols: usize,
    gram: &[f64],
    cfg: &ActivationAwareConfig,
    refit_mode: ScaleRefitMode,
) -> Option<Vec<(f32, Vec<Vec<i8>>)>> {
    let decay = tritium_core::FeedbackDecay::new(cfg.decay, cfg.decay_ramp).ok()?;
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
    let mut group_targets = vec![vec![0.0f32; group]; rows];
    for j in 0..cols {
        let block = j / group;
        if j % group == 0 {
            let end = ((block + 1) * group).min(cols);
            if cfg.refit_scale && refit_mode == ScaleRefitMode::InLoop {
                group_targets
                    .par_iter_mut()
                    .zip(work.par_chunks(cols))
                    .for_each(|(target, wr)| target[..end - j].copy_from_slice(&wr[j..end]));
            }
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
        let lambda = decay.coefficient(j, cols).ok()?;
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
        let end = ((block + 1) * group).min(cols);
        if cfg.refit_scale && refit_mode == ScaleRefitMode::InLoop && j + 1 == end {
            work.par_chunks_mut(cols)
                .zip(quantized.par_chunks_mut(cols))
                .zip(steps.par_chunks_mut(per_row))
                .zip(group_targets.par_iter())
                .for_each(|(((wr, qr), d), target)| {
                    refit_completed_group(
                        wr,
                        qr,
                        d,
                        target,
                        block * group,
                        end,
                        block,
                        cols,
                        &h_damped,
                        &chol,
                    );
                });
        }
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
    let refit = cfg.refit_scale && refit_mode == ScaleRefitMode::PostPass;
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

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn refit_completed_group(
    work: &mut [f32],
    quantized: &mut [f32],
    steps: &mut [f32],
    target: &[f32],
    start: usize,
    end: usize,
    block: usize,
    cols: usize,
    h: &[f64],
    chol: &[f64],
) {
    let old_step = f64::from(steps[block]);
    if old_step <= 0.0 || !old_step.is_finite() {
        return;
    }
    let codes: Vec<i64> = (start..end)
        .map(|j| (f64::from(quantized[j]) / old_step).round() as i64)
        .collect();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for a in start..end {
        let ca = codes[a - start] as f64;
        let hrow = &h[a * cols..(a + 1) * cols];
        for b in start..end {
            let cb = codes[b - start] as f64;
            numerator += ca * hrow[b] * f64::from(target[b - start]);
            denominator += ca * hrow[b] * cb;
        }
    }
    if denominator <= 0.0 || !denominator.is_finite() {
        return;
    }
    let new_step = (numerator / denominator).max(0.0);
    if !new_step.is_finite() {
        return;
    }
    let delta = new_step - old_step;
    if delta.abs() <= f64::EPSILON * old_step.max(1.0) {
        return;
    }
    steps[block] = new_step as f32;
    for j in start..end {
        let new_value = new_step * codes[j - start] as f64;
        let change = new_value - f64::from(quantized[j]);
        quantized[j] = new_value as f32;
        let diagonal = chol[j * cols + j];
        if diagonal <= 0.0 || !diagonal.is_finite() {
            continue;
        }
        let error = change / diagonal;
        for j2 in end..cols {
            // GPTQ applies `work -= (weight - quantized) / diagonal * chol`. Increasing the
            // quantized value lowers that error, so the corresponding future-work correction is
            // an addition.
            work[j2] += (error * chol[j2 * cols + j]) as f32;
        }
    }
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
