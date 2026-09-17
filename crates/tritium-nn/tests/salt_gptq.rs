//! **Lever 1 — change the metric the fit minimizes.**
//!
//! SALT minimizes `‖W − Ŵ‖²`. The loss does not see `W`; it sees `W·x`. Those two are not the same
//! objective, and this repo has measured them moving in *opposite* directions: the AWQ salience fold
//! makes weight-space error **51% worse** (0.0252 → 0.0380) and perplexity **better**.
//!
//! The fold is the **diagonal** approximation of the right thing. It rescales each input channel by
//! `(rms_j/gm)^α`, which is exactly a diagonal reweighting of the error metric. The full version
//! uses the input Gram `H = E[x xᵀ]` and minimizes `‖(W − Ŵ)·X‖²` — and crucially, it can *act* on
//! the off-diagonal, by pushing the error of each quantized column into the columns that are not
//! yet quantized. That is OBQ/GPTQ, and it is the known-best PTQ technique nobody here has run.
//!
//! # Why this is the lever, and allocation was not
//!
//! Allocation is closed: ~50 arms, four campaigns, four signals up to exact task-loss gradients,
//! and uniform won every time — because uniform `T` already *is* allocation by relative precision,
//! and a plane is a fixed 9× step too coarse to express anything finer.
//!
//! Each plane already buys **9.54 dB**, which is `log2(3) × 6.02 dB/bit` — the information-theoretic
//! maximum for 1.585 bits. The ladder is *on* the scalar rate-distortion bound. No redistribution of
//! planes and no better plane can beat it, because the plane is already optimal.
//!
//! What is left is to leave the setting. The ladder is a **scalar**, **memoryless**, **Euclidean**
//! quantizer; this file relaxes the third word. It changes no bits, no container, and no allocation
//! — only which reconstruction the same bits encode.
//!
//! # What this does
//!
//! Per projection, in the rotated basis the artifact actually stores:
//!
//! 1. Collect the input Gram `H` at that projection's tap, on the **folded** model, so `H` is the
//!    distribution the deployed weights see.
//! 2. Rotate `H` to match: `H' = R·H·Rᵀ`, `R` block-diagonal Hadamard over scale groups. Skipping
//!    this would compensate error in one basis using a metric expressed in another.
//! 3. `H'⁻¹` with Tikhonov damping (a calibration Gram is routinely singular), then its Cholesky.
//! 4. Walk the input columns in order. Round each on the ladder's own grid, then push the residual
//!    into the columns still to come, weighted by `H⁻¹`.
//!
//! The step `Δ` per (row, group) is taken with the fitter's own grid search **when the column walk
//! reaches that group**, on the already-compensated weights — the standard group-wise GPTQ order.
//! Both arms therefore encode the identical container at the identical bit count; only the digits
//! differ.
//!
//! # Not covered, and why it is structural
//!
//! The tied embedding/head is excluded. As an embedding it is a gather — there is no contraction
//! over an input axis for a Gram to describe. As a head it is a projection and would be eligible,
//! but the weights are the same tensor, so compensating it as a head corrupts it as a gather. This
//! is the same tie that makes it unfoldable, and it is the most sensitive tensor in the model.
//!
//! # Measured 2026-09-16 — it works; the first run was under-sampled
//!
//! The first version collected the Gram from **1,024 tokens** and lost by +2.98%. GPTQ losing to
//! plain rounding at 4.75 bits is a red flag for the experiment, not a result, and the cause was
//! visible in the shapes: `down_proj`'s input is **1,536 wide**, so a Gram from fewer tokens than
//! that is rank-deficient by construction. In its null directions `H⁻¹` is enormous, so
//! compensation dumps error there freely — and on held-out data those directions carry activation.
//!
//! ```text
//! Gram tokens   damp 0.01   damp 0.1
//!     1,024       +3.02%      +1.31%
//!     4,096       +0.73%      -0.15%
//!    12,288       -0.49%      -0.46%     <- beats the shipping Euclidean fit
//! ```
//!
//! Three things pin the diagnosis. The loss falls **monotonically** with sample count at either
//! damping. More damping helps a *lot* when samples are scarce (+3.02% → +1.31%) — it is
//! regularizing a badly-estimated matrix. And at 12,288 tokens the two dampings **converge** to
//! within 0.03 points: once the Gram is well estimated there is nothing left for damping to fix.
//!
//! Weight-space error rises (0.0556 → 0.0674) while perplexity improves — the proxy-gap signature
//! again, and the reason nothing in this project ranks recipes by reconstruction fidelity.
//!
//! **Not yet converged.** 4,096 → 12,288 still bought 1.22 points; standard GPTQ calibrates on
//! ~262K tokens, 21× more than the largest arm here. −0.49% is a floor, not the number.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_gptq -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, damped_inverse, extract, fold, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::{Tap, forward_aq};
use tritium_train::Tape;
use tritium_train::ops::ste::{
    self, RotationPolicy, fast_hadamard, group_is_rotatable, ladder_quantize_at, ladder_step,
};
use tritium_train::tape::ValueId;

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
const GRAM_SEQ: usize = 256;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_list_usize(key: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn env_list_f64(key: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_owned()
    });
    let text = std::fs::read_to_string(&path).expect("corpus");
    let v: serde_json::Value = serde_json::from_str(&text).expect("corpus json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Apply the fitter's per-group Hadamard across one `cols`-wide row, in place.
fn rotate_row(row: &mut [f32], group: usize) {
    for slice in row.chunks_mut(group) {
        if group_is_rotatable(slice.len()) {
            fast_hadamard(slice);
        }
    }
}

/// `fast_hadamard` in f64: the same unnormalized Walsh–Hadamard recursion, then `1/√n`.
///
/// The Gram is a metric, and a Cholesky of its inverse amplifies rounding. The first version of this
/// file round-tripped `H` through f32 to reuse `fast_hadamard`, which throws away half the digits of
/// the very quantity being factorized. The basis is identical — only the precision differs.
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

/// `H ← R·H·Rᵀ`, `R` block-diagonal Hadamard over scale groups.
///
/// Two passes: rotate every row's groups (that is `H·Rᵀ`, since `R` is symmetric), then every
/// column's. Uses the same group rule as the fitter, so the metric is expressed in exactly the basis
/// the stored codes are in.
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

/// Lower Cholesky `A = L·Lᵀ`, row-major, lower triangle filled.
fn lower_cholesky(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
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

/// GPTQ one tensor in the rotated basis. Returns the dense reconstruction in the ORIGINAL basis,
/// so it drops into the same evaluator as the plain fit.
///
/// `w` is row-major `[rows, cols]`; `gram` is `cols × cols` in the original basis.
fn gptq_tensor(
    w: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    gram: &[f64],
    damp: f64,
) -> Option<Vec<f32>> {
    let (mut out, _) = gptq_codes(w, rows, cols, t, gram, damp)?;
    // Back to the model's basis.
    for row in out.chunks_mut(cols) {
        rotate_row(row, GROUP);
    }
    Some(out)
}

/// GPTQ in the rotated basis, returning the reconstruction STILL IN THAT BASIS plus the step `Δ`
/// per (row, group) — everything a discrete search needs to continue from where GPTQ stopped.
fn gptq_codes(
    w: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    gram: &[f64],
    damp: f64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    // Everything happens in the basis the codes are stored in.
    let mut work: Vec<f32> = w.to_vec();
    for row in work.chunks_mut(cols) {
        rotate_row(row, GROUP);
    }
    let mut h = gram.to_vec();
    rotate_gram(&mut h, cols, GROUP);
    let hinv = damped_inverse(&h, cols, damp)?;
    let chol = lower_cholesky(&hinv, cols)?;

    let per_row = cols.div_ceil(GROUP);
    let mut out = vec![0.0f32; rows * cols];
    // One step per (row, group), taken when the walk first reaches that group.
    let mut delta = vec![0.0f32; rows * per_row];

    for j in 0..cols {
        let block = j / GROUP;
        if j % GROUP == 0 {
            let end = ((block + 1) * GROUP).min(cols);
            // The step comes from the CURRENT compensated weights — group-wise GPTQ order.
            delta
                .par_chunks_mut(per_row)
                .zip(work.par_chunks(cols))
                .for_each(|(d, wr)| {
                    d[block] = ladder_step(&wr[j..end], t, GRID);
                });
        }
        let d_jj = chol[j * cols + j];
        if d_jj <= 0.0 || !d_jj.is_finite() {
            return None;
        }
        // Quantize column j, then push its residual into the columns still to come.
        out.par_chunks_mut(cols)
            .zip(work.par_chunks_mut(cols))
            .zip(delta.par_chunks(per_row))
            .for_each(|((o, wr), d)| {
                let q = ladder_quantize_at(wr[j], t, d[block]);
                o[j] = q;
                let err = f64::from(wr[j] - q) / d_jj;
                for j2 in (j + 1)..cols {
                    wr[j2] -= (err * chol[j2 * cols + j]) as f32;
                }
            });
    }

    Some((out, delta))
}

/// **Discrete local search over the trit move set**, continuing from GPTQ's codes.
///
/// Per row the problem is a closest-vector problem: choose integers `k` (each `|k_j| ≤ (3^T−1)/2`)
/// minimizing `(w − Δ⊙k)ᵀ·H·(w − Δ⊙k)`. GPTQ is one greedy sequential-rounding pass. This improves
/// on it by coordinate descent with the moves the ternary representation makes natural: change
/// `k_j` by `±1`, or by `±3^p` — a flip of a higher-plane trit, a jump rounding never considers.
///
/// Every move is priced exactly in O(1) from `g = H·r`: changing `r_j` by `δ` changes the objective by
/// `2δ·g_j + δ²·H_jj`, and an accepted move updates `g` in O(k). So the objective falls
/// monotonically and no move is accepted on an estimate.
///
/// With `refit_scale`, each sweep also re-solves every group's step `Δ` in closed form for its
/// current codes (`ε = kᵀg / kᵀHk`) — the continuous half, alternating with the discrete half.
///
/// Returns the reconstruction in the model's basis. `H` here is the damped, rotated Gram.
// Indexing `d` by group while `k`, `r` and `g` are indexed by column is the clearest form of the
// closed-form step refit; an iterator over `d` alone would hide which ranges belong together.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn search_codes(
    w: &[f32],
    cols: usize,
    t: usize,
    h_rot: &[f64],
    gptq_rot: &[f32],
    delta: &[f32],
    sweeps: usize,
    refit_scale: bool,
) -> Vec<f32> {
    let per_row = cols.div_ceil(GROUP);
    let kmax = (3i64.pow(t as u32) - 1) / 2;
    let moves: Vec<i64> = (0..t)
        .flat_map(|p| {
            let m = 3i64.pow(p as u32);
            [m, -m]
        })
        .collect();
    let mut w_rot: Vec<f32> = w.to_vec();
    for row in w_rot.chunks_mut(cols) {
        rotate_row(row, GROUP);
    }
    let mut out = vec![0.0f32; w.len()];
    out.par_chunks_mut(cols)
        .zip(w_rot.par_chunks(cols))
        .zip(gptq_rot.par_chunks(cols))
        .zip(delta.par_chunks(per_row))
        .for_each(|(((o, wr), qr), dr)| {
            let mut d: Vec<f64> = dr.iter().map(|&v| f64::from(v)).collect();
            let mut k: Vec<i64> = (0..cols)
                .map(|j| {
                    let dj = d[j / GROUP];
                    if dj > 0.0 {
                        (f64::from(qr[j]) / dj).round() as i64
                    } else {
                        0
                    }
                })
                .collect();
            let mut r: Vec<f64> = (0..cols)
                .map(|j| f64::from(wr[j]) - d[j / GROUP] * k[j] as f64)
                .collect();
            let mut g: Vec<f64> = (0..cols)
                .map(|a| {
                    h_rot[a * cols..(a + 1) * cols]
                        .iter()
                        .zip(&r)
                        .map(|(h, x)| h * x)
                        .sum()
                })
                .collect();
            for _ in 0..sweeps {
                let mut improved = false;
                for j in 0..cols {
                    let dj = d[j / GROUP];
                    if dj <= 0.0 {
                        continue;
                    }
                    let hjj = h_rot[j * cols + j];
                    let mut best = (0.0f64, 0i64);
                    for &m in &moves {
                        let nk = k[j] + m;
                        if nk.abs() > kmax {
                            continue;
                        }
                        let delta_r = -dj * m as f64;
                        let change = 2.0 * delta_r * g[j] + delta_r * delta_r * hjj;
                        if change < best.0 {
                            best = (change, m);
                        }
                    }
                    if best.1 != 0 && best.0 < -1e-15 {
                        let delta_r = -dj * best.1 as f64;
                        k[j] += best.1;
                        r[j] += delta_r;
                        for (a, ga) in g.iter_mut().enumerate() {
                            *ga += delta_r * h_rot[a * cols + j];
                        }
                        improved = true;
                    }
                }
                if refit_scale {
                    for b in 0..per_row {
                        let (lo, hi) = (b * GROUP, ((b + 1) * GROUP).min(cols));
                        let num: f64 = (lo..hi).map(|j| k[j] as f64 * g[j]).sum();
                        let mut den = 0.0f64;
                        for a in lo..hi {
                            if k[a] == 0 {
                                continue;
                            }
                            for c in lo..hi {
                                den += k[a] as f64 * h_rot[a * cols + c] * k[c] as f64;
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
                        // r_b -= eps·k_b, so g -= eps·H[:, b]·k_b.
                        for j in lo..hi {
                            r[j] -= eps * k[j] as f64;
                        }
                        for (a, ga) in g.iter_mut().enumerate() {
                            let hrow = &h_rot[a * cols..(a + 1) * cols];
                            let s: f64 = (lo..hi).map(|c| hrow[c] * k[c] as f64).sum();
                            *ga -= eps * s;
                        }
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
            for j in 0..cols {
                o[j] = (d[j / GROUP] * k[j] as f64) as f32;
            }
            rotate_row(o, GROUP);
        });
    out
}

/// `tr((W − Ŵ)·H·(W − Ŵ)ᵀ)` — the layer objective, in the model's basis.
fn layer_objective(w: &[f32], q: &[f32], cols: usize, h: &[f64]) -> f64 {
    w.par_chunks(cols)
        .zip(q.par_chunks(cols))
        .map(|(wr, qr)| {
            let e: Vec<f64> = wr.iter().zip(qr).map(|(&a, &b)| f64::from(a - b)).collect();
            (0..cols)
                .map(|a| {
                    if e[a] == 0.0 {
                        return 0.0;
                    }
                    e[a] * h[a * cols..(a + 1) * cols]
                        .iter()
                        .zip(&e)
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum()
}

/// Accumulate `Σ x xᵀ` (upper triangle) from a `[seq, k]` activation block.
///
/// Parallel over rows of `H`, which are independent. This is the hot loop — `O(seq·k²)` per tap, and
/// the FFN intermediate is 1536 wide — so it has to be, or the sample counts GPTQ actually needs are
/// out of reach.
fn accumulate_gram(h: &mut [f64], k: usize, act: &[f32], seq: usize) {
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

fn mirror_and_scale(h: &mut [f64], k: usize, rows: usize) {
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

#[test]
#[ignore = "needs SmolLM2-135M; collects per-tap Grams and runs GPTQ over every projection"]
fn activation_metric_fit_against_the_euclidean_one() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_GPTQ_T", 3);
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);

    let n_layers = arch.n_layers;
    let q_width = arch.n_head * arch.head_dim;

    // ── Baseline: the shipping fit. Same container, same bits, Euclidean metric. Scored once.
    let plain: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                r,
                c,
                t_ref,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();
    let werr = |q: &[Vec<f32>]| -> f64 {
        let (mut se, mut sw) = (0.0f64, 0.0f64);
        for (a, b) in fp.iter().zip(q) {
            for (&x, &y) in a.iter().zip(b) {
                se += f64::from(x - y) * f64::from(x - y);
                sw += f64::from(x) * f64::from(x);
            }
        }
        (se / sw).sqrt()
    };
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_plain = perplexity_windowed(&plain, &arch, &eval, EVAL_WINDOW);

    let windows_list = env_list_usize("TRITIUM_GPTQ_WINDOWS", &[4, 16, 48]);
    let damp_list = env_list_f64("TRITIUM_GPTQ_DAMP", &[0.01, 0.1]);
    let max_windows = *windows_list
        .iter()
        .max()
        .expect("at least one window count");
    assert!(
        (max_windows * GRAM_SEQ) <= train.len(),
        "{max_windows} × {GRAM_SEQ} calibration tokens exceeds the training split"
    );

    println!(
        "SmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | T={t_ref} | rotation always\n\
         down_proj's input is {} wide, so a Gram from fewer tokens than that is rank-deficient by\n\
         construction.\n",
        eval.len(),
        arch.ff
    );
    println!(
        "{:<44} {:>11} {:>10} {:>12} {:>10}",
        "fit", "ppl", "× fp", "vs Eucl.", "wt err"
    );
    println!("{}", "-".repeat(92));
    println!(
        "{:<44} {ppl_fp:>11.4} {:>9.4}× {:>12} {:>10}",
        "fp master", 1.0, "—", "—"
    );
    println!(
        "{:<44} {ppl_plain:>11.4} {:>9.4}× {:>12} {:>10.4}",
        "Euclidean ‖W−Ŵ‖² (SHIPPING)",
        ppl_plain / ppl_fp,
        "—",
        werr(&plain)
    );

    // ── Grams accumulate as a running sum, so every smaller window count is a PREFIX of the larger
    // one and can be snapshotted on the way past for free. That makes sample size a clean variable:
    // the same tokens, in the same order, just more of them.
    let mut g_attn: Vec<Vec<f64>> = vec![vec![0.0; arch.n_embd * arch.n_embd]; n_layers];
    let mut g_ffn: Vec<Vec<f64>> = vec![vec![0.0; arch.n_embd * arch.n_embd]; n_layers];
    let mut g_down: Vec<Vec<f64>> = vec![vec![0.0; arch.ff * arch.ff]; n_layers];
    let mut g_o: Vec<Vec<f64>> = vec![vec![0.0; q_width * q_width]; n_layers];
    let mut seen = 0usize;
    let mut best: Option<(f64, String)> = None;

    for wnd in 0..max_windows {
        let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
        let mut t = Tape::new();
        let wids: Vec<ValueId> = fp.iter().map(|w| t.leaf(w.clone())).collect();
        forward_aq(&mut t, &wids, &arch, toks, &mut |kind, li, v, seq, cols| {
            // Identity: this pass is a tap, not a quantizer.
            match kind {
                Tap::AttnIn => accumulate_gram(&mut g_attn[li], cols, v, seq),
                Tap::FfnIn => accumulate_gram(&mut g_ffn[li], cols, v, seq),
                Tap::DownIn => accumulate_gram(&mut g_down[li], cols, v, seq),
                Tap::OProjIn => accumulate_gram(&mut g_o[li], cols, v, seq),
                // The tied head is excluded — see the module docs.
                Tap::Head => {}
            }
        });
        seen += toks.len();

        if !windows_list.contains(&(wnd + 1)) {
            continue;
        }
        // Snapshot: finalize a COPY so the running sum keeps accumulating untouched.
        let snap = |g: &[Vec<f64>], k: usize| -> Vec<Vec<f64>> {
            g.iter()
                .map(|h| {
                    let mut c = h.clone();
                    mirror_and_scale(&mut c, k, seen);
                    c
                })
                .collect()
        };
        let (sa, sf, sd, so) = (
            snap(&g_attn, arch.n_embd),
            snap(&g_ffn, arch.n_embd),
            snap(&g_down, arch.ff),
            snap(&g_o, q_width),
        );

        for &damp in &damp_list {
            let mut gptq = plain.clone();
            let (mut done, mut skipped) = (0usize, 0usize);
            for li in 0..n_layers {
                let base = 1 + 7 * li;
                for (slot, gram) in [
                    (0usize, &sa[li]), // q
                    (1, &sa[li]),      // k
                    (2, &sa[li]),      // v
                    (3, &so[li]),      // o
                    (4, &sf[li]),      // gate
                    (5, &sf[li]),      // up
                    (6, &sd[li]),      // down
                ] {
                    let i = base + slot;
                    let (rows, cols) = shapes[i];
                    match gptq_tensor(&fp[i], rows, cols, t_ref, gram, damp) {
                        Some(q) => {
                            gptq[i] = q;
                            done += 1;
                        }
                        // A Gram still singular after damping saw no signal; fall back rather than
                        // propagate garbage.
                        None => skipped += 1,
                    }
                }
            }
            let ppl = perplexity_windowed(&gptq, &arch, &eval, EVAL_WINDOW);
            let label = format!("activation metric, {seen:>5} tok, damp {damp}");
            println!(
                "{label:<44} {ppl:>11.4} {:>9.4}× {:>11.2}% {:>10.4}{}",
                ppl / ppl_fp,
                100.0 * (ppl - ppl_plain) / ppl_plain,
                werr(&gptq),
                if skipped > 0 {
                    format!("  ({skipped} fell back)")
                } else {
                    String::new()
                }
            );
            assert!(
                done > 0,
                "no projection was fitted at {seen} tokens / damp {damp} — the arm is the \
                 baseline under another name"
            );
            if best.as_ref().is_none_or(|(b, _)| ppl < *b) {
                best = Some((ppl, label));
            }
        }
    }

    let (best_ppl, best_label) = best.expect("at least one arm ran");
    println!(
        "\nbest activation-metric arm: {best_label} at {best_ppl:.4} ({:+.2}% vs shipping {ppl_plain:.4})",
        100.0 * (best_ppl - ppl_plain) / ppl_plain
    );
    println!(
        "\nRead DOWN the token column at fixed damping. If perplexity falls as samples grow, the first\n\
         run's +2.98% was an under-sampled Gram overfitting the calibration batch, not the method."
    );
}

/// **Stage 3 of the standard additive-PTQ recipe: calibrate each layer on the QUANTIZED model's
/// inputs, in order.**
///
/// The sweep above collects every Gram from the fp model. Reference GPTQ runs block by block —
/// quantize block `L`, push calibration data through the quantized prefix, collect block `L+1`'s
/// inputs there.
///
/// **Measured 2026-09-17: no gain, and the reason corrects what this doc first claimed.**
///
/// ```text
/// round-to-nearest (SHIPPING)   24.2783            —
/// GPTQ, fp Grams                24.1586      -0.49%
/// GPTQ, sequential Grams        24.1732      -0.43%
/// ```
///
/// The first version of this doc said sequential calibration makes every block "absorb the error of
/// every block before it". It does not. Plain sequential GPTQ minimizes `‖(W − Ŵ)·X̂‖`: the fp weights
/// applied to the QUANTIZED inputs. That refits the layer to the shifted input distribution, but the
/// target is still `W·X̂`, not the fp model's actual output `W·X`, so inherited error is never
/// corrected — only not compounded. At T=3, where rounding error is small, that buys nothing.
///
/// Absorbing upstream error needs the asymmetric target `‖W·X − Ŵ·X̂‖`, which is GPTQ run on
/// `W' = W·C·Ĥ⁻¹` with the cross-covariance `C = E[x·x̂ᵀ]` — see
/// `asymmetric_calibration_absorbs_inherited_error`.
///
/// Three arms at identical bits, all with the Gram at the full 12,288 tokens and damping 0.01:
/// the shipping round-to-nearest fit, GPTQ on fp Grams (reproducing the sweep's best arm), and
/// GPTQ on sequential Grams. The tied embedding is round-to-nearest in all three, and it is
/// quantized before layer 0's Gram is collected, so the sequential arm's inputs are the quantized
/// model's inputs from the first token on.
#[test]
#[ignore = "needs SmolLM2-135M; one full forward per layer per calibration window"]
fn sequential_calibration_against_fp_calibration() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_GPTQ_T", 3);
    let windows = env_usize("TRITIUM_GPTQ_SEQ_WINDOWS", 48);
    let damp = 0.01f64;
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();
    assert!(windows * GRAM_SEQ <= train.len());

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);
    let n_layers = arch.n_layers;
    let q_width = arch.n_head * arch.head_dim;

    let plain: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                r,
                c,
                t_ref,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();

    // Grams for ONE layer, collected by running `weights` forward. Other layers' taps are ignored.
    let layer_grams = |weights: &[Vec<f32>], li: usize| -> [Vec<f64>; 4] {
        let mut ga = vec![0.0f64; arch.n_embd * arch.n_embd];
        let mut gf = vec![0.0f64; arch.n_embd * arch.n_embd];
        let mut gd = vec![0.0f64; arch.ff * arch.ff];
        let mut go = vec![0.0f64; q_width * q_width];
        let mut seen = 0usize;
        for wnd in 0..windows {
            let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
            let mut t = Tape::new();
            let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
            forward_aq(&mut t, &wids, &arch, toks, &mut |kind, l, v, seq, cols| {
                if l != li {
                    return;
                }
                match kind {
                    Tap::AttnIn => accumulate_gram(&mut ga, cols, v, seq),
                    Tap::FfnIn => accumulate_gram(&mut gf, cols, v, seq),
                    Tap::DownIn => accumulate_gram(&mut gd, cols, v, seq),
                    Tap::OProjIn => accumulate_gram(&mut go, cols, v, seq),
                    Tap::Head => {}
                }
            });
            seen += toks.len();
        }
        mirror_and_scale(&mut ga, arch.n_embd, seen);
        mirror_and_scale(&mut gf, arch.n_embd, seen);
        mirror_and_scale(&mut gd, arch.ff, seen);
        mirror_and_scale(&mut go, q_width, seen);
        [ga, gf, gd, go]
    };

    let quantize_layer = |target: &mut [Vec<f32>], li: usize, g: &[Vec<f64>; 4]| -> usize {
        let base = 1 + 7 * li;
        let mut fell_back = 0;
        for (slot, gram) in [
            (0usize, &g[0]),
            (1, &g[0]),
            (2, &g[0]),
            (3, &g[3]),
            (4, &g[1]),
            (5, &g[1]),
            (6, &g[2]),
        ] {
            let i = base + slot;
            let (rows, cols) = shapes[i];
            match gptq_tensor(&fp[i], rows, cols, t_ref, gram, damp) {
                Some(q) => target[i] = q,
                None => {
                    target[i] = plain[i].clone();
                    fell_back += 1;
                }
            }
        }
        fell_back
    };

    // ── GPTQ on fp Grams: every layer's inputs come from the fp model.
    println!("GPTQ on fp Grams ({} tokens)…", windows * GRAM_SEQ);
    // fp Grams do not depend on quantization order, so every layer's come from one pass.
    let fp_grams: Vec<[Vec<f64>; 4]> = {
        let mut all: Vec<[Vec<f64>; 4]> = (0..n_layers)
            .map(|_| {
                [
                    vec![0.0f64; arch.n_embd * arch.n_embd],
                    vec![0.0f64; arch.n_embd * arch.n_embd],
                    vec![0.0f64; arch.ff * arch.ff],
                    vec![0.0f64; q_width * q_width],
                ]
            })
            .collect();
        let mut seen = 0usize;
        for wnd in 0..windows {
            let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
            let mut t = Tape::new();
            let wids: Vec<ValueId> = fp.iter().map(|w| t.leaf(w.clone())).collect();
            forward_aq(
                &mut t,
                &wids,
                &arch,
                toks,
                &mut |kind, l, v, seq, cols| match kind {
                    Tap::AttnIn => accumulate_gram(&mut all[l][0], cols, v, seq),
                    Tap::FfnIn => accumulate_gram(&mut all[l][1], cols, v, seq),
                    Tap::DownIn => accumulate_gram(&mut all[l][2], cols, v, seq),
                    Tap::OProjIn => accumulate_gram(&mut all[l][3], cols, v, seq),
                    Tap::Head => {}
                },
            );
            seen += toks.len();
        }
        for g in &mut all {
            mirror_and_scale(&mut g[0], arch.n_embd, seen);
            mirror_and_scale(&mut g[1], arch.n_embd, seen);
            mirror_and_scale(&mut g[2], arch.ff, seen);
            mirror_and_scale(&mut g[3], q_width, seen);
        }
        all
    };
    let mut on_fp = plain.clone();
    let mut fb_fp = 0;
    for (li, g) in fp_grams.iter().enumerate() {
        fb_fp += quantize_layer(&mut on_fp, li, g);
    }
    drop(fp_grams);

    // ── Sequential: layer L's Grams come from a model whose embedding and layers < L are already
    // quantized. `seq` IS that model at every step.
    println!("GPTQ on sequential Grams…");
    let mut seq = fp.clone();
    seq[0] = plain[0].clone();
    let mut fb_seq = 0;
    for li in 0..n_layers {
        let g = layer_grams(&seq, li);
        fb_seq += quantize_layer(&mut seq, li, &g);
        if li % 10 == 0 {
            println!("  layer {li}/{n_layers}");
        }
    }

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_plain = perplexity_windowed(&plain, &arch, &eval, EVAL_WINDOW);
    let ppl_on_fp = perplexity_windowed(&on_fp, &arch, &eval, EVAL_WINDOW);
    let ppl_seq = perplexity_windowed(&seq, &arch, &eval, EVAL_WINDOW);

    println!(
        "\nSmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | T={t_ref} | rotation\n\
         Gram {} tokens, damp {damp} | fell back: fp {fb_fp}, sequential {fb_seq}\n",
        eval.len(),
        windows * GRAM_SEQ
    );
    println!(
        "{:<40} {:>11} {:>10} {:>12}",
        "fit", "ppl", "× fp", "vs RTN"
    );
    println!("{}", "-".repeat(78));
    for (label, ppl) in [
        ("fp master", ppl_fp),
        ("round-to-nearest (SHIPPING)", ppl_plain),
        ("GPTQ, fp Grams", ppl_on_fp),
        ("GPTQ, sequential Grams", ppl_seq),
    ] {
        println!(
            "{label:<40} {ppl:>11.4} {:>9.4}× {:>11.2}%",
            ppl / ppl_fp,
            100.0 * (ppl - ppl_plain) / ppl_plain
        );
    }
    println!(
        "\nsequential vs fp Grams: {:+.2}%  |  share of RTN's excess over fp recovered: fp Grams {:.1}%, \
         sequential {:.1}%",
        100.0 * (ppl_seq - ppl_on_fp) / ppl_on_fp,
        100.0 * (ppl_plain - ppl_on_fp) / (ppl_plain - ppl_fp),
        100.0 * (ppl_plain - ppl_seq) / (ppl_plain - ppl_fp)
    );
    assert!(
        ppl_seq.is_finite() && ppl_on_fp.is_finite(),
        "a GPTQ arm did not produce a usable model"
    );
}

/// **Does discrete search over the trit moves beat GPTQ's greedy rounding — and does it generalize?**
///
/// Four arms at identical bits and container: round-to-nearest, GPTQ, GPTQ + trit search, and GPTQ +
/// trit search + closed-form step refits. Grams are fp (the sweep's configuration) at 12,288 tokens.
///
/// A stronger optimizer overfits a Gram more readily, so every arm's layer objective is reported on
/// the calibration Gram AND on a held-out Gram built from the next 12,288 tokens. If the search
/// wins on calibration and loses held-out, that is overfitting, and perplexity should agree.
#[test]
#[ignore = "needs SmolLM2-135M; two Gram sets, a search per projection, and four evaluations"]
fn discrete_search_over_trit_moves_against_gptq() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_GPTQ_T", 3);
    let windows = 48usize;
    let damp = 0.01f64;
    let sweeps = env_usize("TRITIUM_SEARCH_SWEEPS", 8);
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();
    assert!(2 * windows * GRAM_SEQ <= train.len());

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);
    let n_layers = arch.n_layers;
    let q_width = arch.n_head * arch.head_dim;

    // [attn, ffn, down, o] per layer, from windows [first, first + windows).
    let grams = |first: usize| -> Vec<[Vec<f64>; 4]> {
        let mut all: Vec<[Vec<f64>; 4]> = (0..n_layers)
            .map(|_| {
                [
                    vec![0.0f64; arch.n_embd * arch.n_embd],
                    vec![0.0f64; arch.n_embd * arch.n_embd],
                    vec![0.0f64; arch.ff * arch.ff],
                    vec![0.0f64; q_width * q_width],
                ]
            })
            .collect();
        let mut seen = 0usize;
        for wnd in first..first + windows {
            let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
            let mut t = Tape::new();
            let wids: Vec<ValueId> = fp.iter().map(|w| t.leaf(w.clone())).collect();
            forward_aq(
                &mut t,
                &wids,
                &arch,
                toks,
                &mut |kind, l, v, seq, cols| match kind {
                    Tap::AttnIn => accumulate_gram(&mut all[l][0], cols, v, seq),
                    Tap::FfnIn => accumulate_gram(&mut all[l][1], cols, v, seq),
                    Tap::DownIn => accumulate_gram(&mut all[l][2], cols, v, seq),
                    Tap::OProjIn => accumulate_gram(&mut all[l][3], cols, v, seq),
                    Tap::Head => {}
                },
            );
            seen += toks.len();
        }
        for g in &mut all {
            mirror_and_scale(&mut g[0], arch.n_embd, seen);
            mirror_and_scale(&mut g[1], arch.n_embd, seen);
            mirror_and_scale(&mut g[2], arch.ff, seen);
            mirror_and_scale(&mut g[3], q_width, seen);
        }
        all
    };
    println!("calibration Grams…");
    let cal = grams(0);
    println!("held-out Grams…");
    let held = grams(windows);

    let plain: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                r,
                c,
                t_ref,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();
    let mut gptq = plain.clone();
    let mut search_v = plain.clone();
    let mut search_vp = plain.clone();
    // Objective totals: [rtn, gptq, V, V+P] × [cal, held].
    let mut obj = [[0.0f64; 2]; 4];
    println!("fitting (sweeps ≤ {sweeps})…");
    for li in 0..n_layers {
        let base = 1 + 7 * li;
        for (slot, gi) in [
            (0usize, 0usize),
            (1, 0),
            (2, 0),
            (3, 3),
            (4, 1),
            (5, 1),
            (6, 2),
        ] {
            let i = base + slot;
            let (rows, cols) = shapes[i];
            let Some((g_rot, delta)) = gptq_codes(&fp[i], rows, cols, t_ref, &cal[li][gi], damp)
            else {
                continue;
            };
            let mut g_dense = g_rot.clone();
            for row in g_dense.chunks_mut(cols) {
                rotate_row(row, GROUP);
            }
            // The search's objective: the same damped, rotated Gram GPTQ used.
            let mut h = cal[li][gi].clone();
            let mean: f64 = (0..cols).map(|a| h[a * cols + a]).sum::<f64>() / cols as f64;
            for a in 0..cols {
                h[a * cols + a] += damp * mean.max(1e-12);
            }
            rotate_gram(&mut h, cols, GROUP);
            let v = search_codes(&fp[i], cols, t_ref, &h, &g_rot, &delta, sweeps, false);
            let vp = search_codes(&fp[i], cols, t_ref, &h, &g_rot, &delta, sweeps, true);
            for (a, q) in [&plain[i], &g_dense, &v, &vp].into_iter().enumerate() {
                obj[a][0] += layer_objective(&fp[i], q, cols, &cal[li][gi]);
                obj[a][1] += layer_objective(&fp[i], q, cols, &held[li][gi]);
            }
            gptq[i] = g_dense;
            search_v[i] = v;
            search_vp[i] = vp;
        }
        if li % 10 == 0 {
            println!("  layer {li}/{n_layers}");
        }
    }
    drop((cal, held));

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppls = [
        perplexity_windowed(&plain, &arch, &eval, EVAL_WINDOW),
        perplexity_windowed(&gptq, &arch, &eval, EVAL_WINDOW),
        perplexity_windowed(&search_v, &arch, &eval, EVAL_WINDOW),
        perplexity_windowed(&search_vp, &arch, &eval, EVAL_WINDOW),
    ];

    println!(
        "\nSmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | T={t_ref} | fp Grams {} tok\n\
         layer objective summed over 210 projections, relative to round-to-nearest\n",
        eval.len(),
        windows * GRAM_SEQ
    );
    println!(
        "{:<34} {:>12} {:>12} {:>11} {:>10}",
        "fit", "obj (cal)", "obj (held)", "ppl", "vs RTN"
    );
    println!("{}", "-".repeat(84));
    println!("{:<34} {:>12} {:>12} {ppl_fp:>11.4}", "fp master", "—", "—");
    for (a, label) in [
        "round-to-nearest (SHIPPING)",
        "GPTQ",
        "GPTQ + trit search",
        "GPTQ + trit search + Δ refit",
    ]
    .into_iter()
    .enumerate()
    {
        println!(
            "{label:<34} {:>11.3}× {:>11.3}× {:>11.4} {:>+9.2}%",
            obj[a][0] / obj[0][0],
            obj[a][1] / obj[0][1],
            ppls[a],
            100.0 * (ppls[a] - ppls[0]) / ppls[0]
        );
    }
    // The search minimizes the DAMPED objective, whose monotonicity the unit test below proves. The
    // undamped calibration objective reported here can move slightly the other way, so this only
    // guards against gross failure.
    assert!(
        obj[3][0] <= obj[1][0] * 1.05,
        "the search raised the calibration objective by more than 5% over GPTQ"
    );
}

/// The search's contract on a small case: every move is exactly priced, so the damped objective it
/// minimizes can never rise above GPTQ's starting point, with or without step refits.
#[test]
fn trit_search_never_raises_the_damped_objective() {
    let (rows, cols, t) = (4usize, 256usize, 3usize);
    let mut s = 0xABCD_EF01u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / 8_388_608.0) - 1.0
    };
    let w: Vec<f32> = (0..rows * cols).map(|_| next() as f32).collect();
    let m = 600;
    let x: Vec<f64> = (0..m * cols)
        .map(|i| next() * (1.0 + 4.0 * ((i % cols) % 7 == 0) as u8 as f64))
        .collect();
    let mut gram = vec![0.0f64; cols * cols];
    accumulate_gram(
        &mut gram,
        cols,
        &x.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        m,
    );
    mirror_and_scale(&mut gram, cols, m);
    let damp = 0.01;
    let (g_rot, delta) = gptq_codes(&w, rows, cols, t, &gram, damp).expect("gptq");
    let mut g_dense = g_rot.clone();
    for row in g_dense.chunks_mut(cols) {
        rotate_row(row, GROUP);
    }
    let mut hd = gram.clone();
    let mean: f64 = (0..cols).map(|a| hd[a * cols + a]).sum::<f64>() / cols as f64;
    for a in 0..cols {
        hd[a * cols + a] += damp * mean;
    }
    let mut h_rot = hd.clone();
    rotate_gram(&mut h_rot, cols, GROUP);

    let base = layer_objective(&w, &g_dense, cols, &hd);
    let v = search_codes(&w, cols, t, &h_rot, &g_rot, &delta, 8, false);
    let vp = search_codes(&w, cols, t, &h_rot, &g_rot, &delta, 8, true);
    let (ov, ovp) = (
        layer_objective(&w, &v, cols, &hd),
        layer_objective(&w, &vp, cols, &hd),
    );
    println!("damped objective: gptq {base:.6e}  search {ov:.6e}  search+refit {ovp:.6e}");
    assert!(
        ov <= base * (1.0 + 1e-6),
        "trit search raised the objective"
    );
    assert!(
        ovp <= base * (1.0 + 1e-6),
        "trit search + refit raised the objective"
    );
    assert!(
        ov < base,
        "the search found nothing to improve on a random problem"
    );
}

/// **Asymmetric calibration: target the fp model's output, from the quantized model's input.**
///
/// Sequential GPTQ minimizes `‖(W − Ŵ)·X̂‖` and so never corrects error inherited from upstream (see
/// `sequential_calibration_against_fp_calibration`). The target that does is `‖W·X − Ŵ·X̂‖`: layer
/// `L`, fed what the quantized prefix actually produces, reproducing what the fp model produced.
///
/// Summed over samples that objective is, up to a constant, `tr((Ŵ − W')·Ĥ·(Ŵ − W')ᵀ)` with
///
/// ```text
/// Ĥ = Σ x̂·x̂ᵀ      C = Σ x·x̂ᵀ      W' = W·C·Ĥ⁻¹
/// ```
///
/// so it is ordinary GPTQ with `Ĥ` as the metric and `W'` — the least-squares map from quantized
/// inputs to fp outputs — as the target. `x` and `x̂` are the same tap on the same tokens, from the fp
/// model and from the model whose embedding and layers < L are already quantized.
///
/// Arms: round-to-nearest (re-measured) and asymmetric sequential GPTQ. GPTQ on fp Grams (−0.49%)
/// and plain sequential (−0.43%) were measured with this same configuration and are quoted.
#[test]
#[ignore = "needs SmolLM2-135M; two full forwards per layer per calibration window"]
fn asymmetric_calibration_absorbs_inherited_error() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_GPTQ_T", 3);
    let windows = env_usize("TRITIUM_GPTQ_SEQ_WINDOWS", 48);
    let damp = 0.01f64;
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);
    let n_layers = arch.n_layers;
    let q_width = arch.n_head * arch.head_dim;
    let widths = [arch.n_embd, arch.n_embd, arch.ff, q_width]; // attn, ffn, down, o

    let plain: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                r,
                c,
                t_ref,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();

    // Capture layer `li`'s four tap activations for one window.
    let capture = |weights: &[Vec<f32>], toks: &[u32], li: usize| -> [Vec<f32>; 4] {
        let mut out: [Vec<f32>; 4] = Default::default();
        let mut t = Tape::new();
        let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
        forward_aq(
            &mut t,
            &wids,
            &arch,
            toks,
            &mut |kind, l, v, _seq, _cols| {
                if l != li {
                    return;
                }
                let slot = match kind {
                    Tap::AttnIn => 0,
                    Tap::FfnIn => 1,
                    Tap::DownIn => 2,
                    Tap::OProjIn => 3,
                    Tap::Head => return,
                };
                out[slot] = v.to_vec();
            },
        );
        out
    };

    let mut seq = fp.clone();
    seq[0] = plain[0].clone();
    println!(
        "asymmetric sequential GPTQ over {} tokens per layer…",
        windows * GRAM_SEQ
    );
    for li in 0..n_layers {
        let mut hhat: Vec<Vec<f64>> = widths.iter().map(|&k| vec![0.0; k * k]).collect();
        let mut cross: Vec<Vec<f64>> = widths.iter().map(|&k| vec![0.0; k * k]).collect();
        let mut seen = 0usize;
        for wnd in 0..windows {
            let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
            let x = capture(&fp, toks, li);
            let xh = capture(&seq, toks, li);
            let n = toks.len();
            for s in 0..4 {
                let k = widths[s];
                accumulate_gram(&mut hhat[s], k, &xh[s], n);
                // C = Σ x·x̂ᵀ, full (not symmetric).
                let (xs, xhs) = (&x[s], &xh[s]);
                cross[s].par_chunks_mut(k).enumerate().for_each(|(a, row)| {
                    for r in 0..n {
                        let xa = f64::from(xs[r * k + a]);
                        if xa == 0.0 {
                            continue;
                        }
                        let xr = &xhs[r * k..(r + 1) * k];
                        for (c, v) in row.iter_mut().enumerate() {
                            *v += xa * f64::from(xr[c]);
                        }
                    }
                });
            }
            seen += n;
        }
        for s in 0..4 {
            let k = widths[s];
            mirror_and_scale(&mut hhat[s], k, seen);
            for v in cross[s].iter_mut() {
                *v /= seen as f64;
            }
        }

        // Transform matrix per tap: C·Ĥ_d⁻¹.
        let transforms: Vec<Option<Vec<f64>>> = (0..4)
            .map(|s| {
                let k = widths[s];
                let hinv = damped_inverse(&hhat[s], k, damp)?;
                Some(mat_mul(&cross[s], &hinv, k))
            })
            .collect();

        let base = 1 + 7 * li;
        for (slot, s) in [
            (0usize, 0usize),
            (1, 0),
            (2, 0),
            (3, 3),
            (4, 1),
            (5, 1),
            (6, 2),
        ] {
            let i = base + slot;
            let (rows, cols) = shapes[i];
            let Some(m) = &transforms[s] else {
                seq[i] = plain[i].clone();
                continue;
            };
            let target = apply_right(&fp[i], cols, m);
            seq[i] = gptq_tensor(&target, rows, cols, t_ref, &hhat[s], damp)
                .unwrap_or_else(|| plain[i].clone());
        }
        if li % 5 == 0 {
            println!("  layer {li}/{n_layers}");
        }
    }

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_plain = perplexity_windowed(&plain, &arch, &eval, EVAL_WINDOW);
    let ppl_asym = perplexity_windowed(&seq, &arch, &eval, EVAL_WINDOW);
    println!(
        "\nSmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | T={t_ref} | {} tok, damp {damp}\n",
        eval.len(),
        windows * GRAM_SEQ
    );
    println!(
        "{:<44} {:>11} {:>10} {:>10}",
        "fit", "ppl", "× fp", "vs RTN"
    );
    println!("{}", "-".repeat(78));
    println!("{:<44} {ppl_fp:>11.4}", "fp master");
    println!(
        "{:<44} {ppl_plain:>11.4} {:>9.4}× {:>10}",
        "round-to-nearest (SHIPPING)",
        ppl_plain / ppl_fp,
        "—"
    );
    println!(
        "{:<44} {:>11} {:>10} {:>10}",
        "GPTQ, fp Grams (quoted)", "24.1586", "", "-0.49%"
    );
    println!(
        "{:<44} {:>11} {:>10} {:>10}",
        "GPTQ, sequential (quoted)", "24.1732", "", "-0.43%"
    );
    println!(
        "{:<44} {ppl_asym:>11.4} {:>9.4}× {:>+9.2}%",
        "GPTQ, asymmetric sequential",
        ppl_asym / ppl_fp,
        100.0 * (ppl_asym - ppl_plain) / ppl_plain
    );
    assert!(
        ppl_asym.is_finite(),
        "the asymmetric arm did not produce a usable model"
    );
}

/// `A·B` for `k × k` row-major matrices.
fn mat_mul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; k * k];
    m.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let arow = &a[i * k..(i + 1) * k];
        for (j, v) in row.iter_mut().enumerate() {
            *v = arow
                .iter()
                .enumerate()
                .map(|(c, &x)| x * b[c * k + j])
                .sum();
        }
    });
    m
}

/// `W·M` for `W` row-major `[rows, cols]` and `M` `cols × cols`.
fn apply_right(w: &[f32], cols: usize, m: &[f64]) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    out.par_chunks_mut(cols)
        .zip(w.par_chunks(cols))
        .for_each(|(o, wr)| {
            for (b, ob) in o.iter_mut().enumerate() {
                *ob = wr
                    .iter()
                    .enumerate()
                    .map(|(a, &wa)| f64::from(wa) * m[a * cols + b])
                    .sum::<f64>() as f32;
            }
        });
    out
}

/// The asymmetric objective's defining identity, on a small case with no damping:
/// `Σ_s ‖W·x_s − Ŵ·x̂_s‖²  =  tr((Ŵ − W')·Ĥ·(Ŵ − W')ᵀ) + const`, `W' = W·C·Ĥ⁻¹`. The constant must not
/// depend on `Ŵ`, so the difference between the two sides is checked to be identical for two
/// unrelated `Ŵ`.
#[test]
fn asymmetric_target_reproduces_the_fp_output_objective() {
    let (rows, k, n) = (3usize, 6usize, 400usize);
    let mut s = 0x5EED_1234u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / 8_388_608.0) - 1.0
    };
    let w: Vec<f32> = (0..rows * k).map(|_| next() as f32).collect();
    let x: Vec<f64> = (0..n * k).map(|_| next()).collect();
    // x̂: a perturbed, mixed copy of x — what an upstream quantizer would do.
    let xh: Vec<f64> = (0..n * k)
        .map(|i| x[i] + 0.3 * x[(i / k) * k + (i % k + 1) % k] + 0.1 * next())
        .collect();
    let (mut hh, mut c) = (vec![0.0f64; k * k], vec![0.0f64; k * k]);
    for r in 0..n {
        for a in 0..k {
            for b in 0..k {
                hh[a * k + b] += xh[r * k + a] * xh[r * k + b];
                c[a * k + b] += x[r * k + a] * xh[r * k + b];
            }
        }
    }
    // Undamped inverse (tiny damping keeps the helper's contract; 1e-12 is below every term here).
    let hinv = damped_inverse(&hh, k, 1e-12).unwrap();
    let wp = apply_right(&w, k, &mat_mul(&c, &hinv, k));

    let lhs = |q: &[f32]| -> f64 {
        (0..n)
            .map(|r| {
                (0..rows)
                    .map(|i| {
                        let (mut yx, mut yq) = (0.0f64, 0.0f64);
                        for a in 0..k {
                            yx += f64::from(w[i * k + a]) * x[r * k + a];
                            yq += f64::from(q[i * k + a]) * xh[r * k + a];
                        }
                        (yx - yq) * (yx - yq)
                    })
                    .sum::<f64>()
            })
            .sum()
    };
    let rhs = |q: &[f32]| -> f64 {
        (0..rows)
            .map(|i| {
                let e: Vec<f64> = (0..k)
                    .map(|a| f64::from(q[i * k + a]) - f64::from(wp[i * k + a]))
                    .collect();
                (0..k)
                    .map(|a| e[a] * (0..k).map(|b| hh[a * k + b] * e[b]).sum::<f64>())
                    .sum::<f64>()
            })
            .sum()
    };
    let q1: Vec<f32> = (0..rows * k).map(|_| next() as f32).collect();
    let q2: Vec<f32> = (0..rows * k).map(|_| (next() * 0.2) as f32).collect();
    let (c1, c2) = (lhs(&q1) - rhs(&q1), lhs(&q2) - rhs(&q2));
    assert!(
        (c1 - c2).abs() <= 1e-4 * lhs(&q1).abs().max(1.0),
        "objective and GPTQ form differ by a Ŵ-dependent amount: {c1} vs {c2}"
    );
}
