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

    // Back to the model's basis.
    for row in out.chunks_mut(cols) {
        rotate_row(row, GROUP);
    }
    Some(out)
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
