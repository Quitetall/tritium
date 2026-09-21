//! **GPTQ sequential error compensation, finally fed.**
//!
//! `tritium_quantize::fit_with_feedback` is a complete GPTQ/BlockLDLQ implementation in f64 —
//! quantize column groups in order, and after each one push the residual it induced onto the
//! not-yet-quantized columns through `H⁻¹`. It has never run, because nothing produced the `H`.
//!
//! `common::GramSet` now does. This wires the two together and measures whether the off-diagonal
//! curvature — 34–68% of `‖H‖²`, per `salt_gram.rs` — is worth what it costs.
//!
//! Coverage is six of seven projections per block: q/k/v (attn tap), gate/up (ffn tap), down (down
//! tap). `o_proj` is skipped because its input is the attention concat, where GQA has query heads
//! sharing kv dims — the same reason the salience fold skips it. The tied embed/head is skipped
//! because it has no single input to take a Gram over.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_gptq -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, GramSet, calibrate, damped_inverse, extract, fold, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_quantize::{ColumnGroup, FeedbackMetric, FeedbackProblem, fit_with_feedback};
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
/// Calibration windows, overridable with `TRITIUM_GPTQ_CALIB_WINDOWS`.
///
/// The Gram is a `k × k` covariance — 576² for the attention and FFN taps, 1,536² for `down`. Four
/// windows is 2,048 tokens against a 1,536-wide input, fewer samples than the covariance has
/// dimensions, so `damped_inverse` returns mostly damping rather than data. Measured on
/// SmolLM2-135M over WikiText-2, GPTQ's effect on held-out perplexity versus the plain fitter:
///
/// ```text
/// cal tokens      T=2       T=3
///      2,048   -18.3%    +5.6%
///      4,096   -19.3%    +4.0%
///      8,192   -22.7%    +0.4%
///     16,384   -29.3%    -0.6%
///     32,768   -31.1%    -2.3%
/// ```
///
/// At T=3 the sign flips between 8,192 and 16,384 tokens: below that, sequential compensation
/// against an under-sampled Hessian is worse than not compensating at all. The old default of four
/// windows sat on the wrong side of that crossing, which is why GPTQ read as marginal here. Neither
/// column has plateaued at 32,768, so this default is a floor, not an optimum.
///
/// T=1 is omitted deliberately: it runs at 10⁴–10⁵× fp in every arm and its deltas bounce
/// (-91.8/-97.1/-79.2/-90.0/-94.5) with no trend. Differences between destroyed models are noise.
///
/// Repeating the sweep with `TRITIUM_GPTQ_SMOOTH=0.75` — the shipped configuration — the crossing
/// lands in the same place, and GPTQ still pays on top of the fold:
///
/// ```text
/// cal tokens      T=2       T=3
///      2,048    +4.8%    +3.6%
///      4,096    -5.9%    +1.1%
///      8,192   -11.2%    -1.7%
///     16,384   -13.1%    -4.3%
///     32,768   -17.0%    -4.9%
/// ```
///
/// Two readings follow. The crossing sits near 8k tokens folded and unfolded alike, which is the
/// Gram's sample count against its dimension — the fold changes neither, so it cannot move it. And
/// the fold and GPTQ overlap without being redundant: the fold alone takes T=2 from 6.40× fp to
/// 2.79×, more than unfolded GPTQ ever recovers, yet GPTQ still buys a further 17% on top.
///
/// One caveat on the folded numbers: `calibrate` for the fold reads the same window loop as the
/// Gram, so this knob moves both at once and the folded `plain` baseline drifts between arms
/// (T=3: 31.822, 31.501, 31.686, 32.361, 32.155 — unordered). Each delta compares plain against
/// GPTQ at an identical fold and is sound; cross-arm comparisons of absolute folded perplexity are
/// not. Those tables predate the split below: the fold now reads its own
/// `TRITIUM_GPTQ_FOLD_WINDOWS`, pinned at 32 by default, so sweeping the Gram no longer moves the
/// fold and the folded `plain` baseline is identical in every arm.
///
/// **Error-decayed propagation (ADR 0043 L-B), fold pinned at 32 windows.** GPTQ's change against
/// the plain fit, by Gram size and decay `λ` on the propagated error:
///
/// ```text
///                       T=2                          T=3
/// tokens    λ=1    0.9    0.75    0.5     λ=1    0.9    0.75    0.5
///  2,048   +4.6   -0.4   -5.6   -7.4     +2.1   +1.2   -0.2   -2.6     constant
///  4,096   -2.5   -5.2   -8.6   -9.8     -1.2   -1.8   -3.1   -3.6     constant
///  4,096          -3.6   -5.5   -7.4            -1.4   -2.2   -2.4     ramp
/// 16,384  -13.1  -13.7  -13.8  -11.0     -4.3   -4.7   -5.1   -4.8     constant
/// 16,384         -14.2  -14.5  -13.6            -4.6   -4.8   -5.3     ramp
/// ```
///
/// Decay pays everywhere and pays most where the Gram is starved: at 2,048 tokens it turns a
/// harmful GPTQ into a helpful one, and `λ = 0.5` there beats undecayed GPTQ on twice the tokens.
/// The best `λ` falls as tokens fall (0.75 at 16k, 0.5 or lower at 4k and 2k), so a fixed `λ` is
/// the wrong object and a schedule in tokens-per-dimension is the natural next step. The ramp
/// shape wins at 16k and loses at 4k; its mean decay is milder than the constant's at equal `λ`, so
/// that comparison does not yet separate *where* the decay lands from *how much* there is.
///
/// With the fold pinned, the folded sign flip is the Gram's alone and crosses between 2,048 and
/// 4,096 tokens. The ~8k crossing in the folded table above was partly the under-calibrated fold.
const DEFAULT_CALIB_WINDOWS: usize = 32;
const DEFAULT_FOLD_WINDOWS: usize = 32;

fn calib_windows(train_len: usize) -> usize {
    windows_from_env(
        "TRITIUM_GPTQ_CALIB_WINDOWS",
        DEFAULT_CALIB_WINDOWS,
        train_len,
    )
}

fn fold_windows(train_len: usize) -> usize {
    windows_from_env("TRITIUM_GPTQ_FOLD_WINDOWS", DEFAULT_FOLD_WINDOWS, train_len)
}

fn windows_from_env(name: &str, default: usize, train_len: usize) -> usize {
    let requested = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .max(1);
    let available = train_len / CALIB_SEQ;
    assert!(
        available > 0,
        "corpus is shorter than one calibration window"
    );
    requested.min(available)
}
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const ITERS: usize = 5;
/// GPTQ's standard ridge, as a fraction of `mean(diag H)`. A calibration Gram is routinely
/// singular (a dead channel gives an exactly-zero row), so without damping the Cholesky fails.
const DAMP: f64 = 0.01;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    let ids = |k: &str| -> Vec<u32> {
        j[k].as_array()
            .expect(k)
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// The plain fitter: what every SALT number so far used.
fn plain(w: &[f32], rows: usize, cols: usize, t: usize) -> Vec<f32> {
    ste::salt_quantize_forward_grouped(w, rows, cols, t, GROUP, ITERS, RotationPolicy::Auto)
}

/// The same fitter, driven through GPTQ sequential feedback against a real inverse Hessian.
///
/// Column groups are exactly `GROUP` wide so each feedback block is one scale group per row — the
/// identical partition the plain fitter uses, which keeps this an ablation of the FEEDBACK rather
/// than of the grouping.
///
/// `decay` scales the rounding error each group hands to the columns after it (ADR 0043 L-B, after
/// QTEA). The library propagates `working − returned`, so the callback reports
/// `working − λ·(working − fit)` and keeps the true fit aside; `λ = 1` is plain GPTQ bit for bit.
/// Under [`DecayShape::Ramp`] early groups propagate in full and `λ` is reached only at the last
/// group, which is where a short remaining suffix has to absorb everything pushed into it.
fn gptq(
    w: &[f32],
    rows: usize,
    cols: usize,
    t: usize,
    h_inv: &[f64],
    decay: f64,
    shape: DecayShape,
) -> Option<Vec<f32>> {
    let weights: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
    let groups: Vec<ColumnGroup> = (0..cols.div_ceil(GROUP))
        .map(|g| ColumnGroup {
            start: g * GROUP,
            end: ((g + 1) * GROUP).min(cols),
        })
        .collect();
    let problem = FeedbackProblem {
        rows,
        columns: cols,
        weights: &weights,
        groups: &groups,
        metric: FeedbackMetric::InverseHessian(h_inv),
    };
    let group_count = groups.len();
    let mut shipped = vec![0.0f32; rows * cols];
    fit_with_feedback(problem, |req: tritium_quantize::GroupFitRequest<'_>| {
        // The block arrives feedback-adjusted: earlier groups' rounding error has already been
        // pushed into it. Quantize it exactly as the plain fitter would.
        let block: Vec<f32> = req.working_weights.iter().map(|&v| v as f32).collect();
        let fit = ste::salt_quantize_forward_grouped(
            &block,
            req.rows,
            req.columns,
            t,
            GROUP,
            ITERS,
            RotationPolicy::Auto,
        );
        for row in 0..req.rows {
            let from = row * req.columns;
            let to = row * cols + req.column_start;
            shipped[to..to + req.columns].copy_from_slice(&fit[from..from + req.columns]);
        }
        let lambda = match shape {
            DecayShape::Constant => decay,
            DecayShape::Ramp if group_count > 1 => {
                1.0 - (1.0 - decay) * req.group_index as f64 / (group_count - 1) as f64
            }
            DecayShape::Ramp => decay,
        };
        Ok::<Vec<f64>, std::convert::Infallible>(
            req.working_weights
                .iter()
                .zip(&fit)
                .map(|(&working, &got)| working - lambda * (working - f64::from(got)))
                .collect(),
        )
    })
    .ok()?;
    Some(shipped)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum DecayShape {
    Constant,
    Ramp,
}

/// `TRITIUM_GPTQ_DECAYS=1.0,0.75,0.5` and `TRITIUM_GPTQ_DECAY_SHAPE=const|ramp`. Every listed decay
/// reuses the one set of Grams, so a sweep costs one collection rather than one per arm.
fn decay_arms() -> (Vec<f64>, DecayShape) {
    let decays = std::env::var("TRITIUM_GPTQ_DECAYS")
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|value| value.trim().parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|list| !list.is_empty())
        .unwrap_or_else(|| vec![1.0]);
    let shape = match std::env::var("TRITIUM_GPTQ_DECAY_SHAPE").as_deref() {
        Ok("ramp") => DecayShape::Ramp,
        _ => DecayShape::Constant,
    };
    (decays, shape)
}

fn planes_under_test() -> Vec<usize> {
    std::env::var("TRITIUM_GPTQ_PLANES")
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|list| !list.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3])
}

/// Does sequential compensation against real curvature beat the plain fitter on held-out ppl?
#[test]
#[ignore = "slow GPTQ sweep; needs SmolLM2-135M; run explicitly"]
fn gptq_feedback_against_real_curvature() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let calib_windows = calib_windows(train.len());
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // TRITIUM_GPTQ_SMOOTH=<alpha> composes the salience fold with GPTQ. The two address DISJOINT
    // parts of the curvature -- the fold rescales columns by E[x_j²] (the diagonal), GPTQ propagates
    // rounding residuals through H⁻¹ (the off-diagonal) -- so whether they add is a real question.
    //
    // Order matters: fold FIRST, then collect the Gram, because folding divides attn_norm/ffn_norm
    // by s and therefore changes the very activations H is a covariance of. Collecting H on the
    // unfolded model and applying it to folded weights would be measuring the wrong curvature.
    let (arch, fp) = match std::env::var("TRITIUM_GPTQ_SMOOTH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        None => (arch, fp),
        Some(alpha) => {
            let mut calib = Calib::new(&arch);
            let fold_windows = fold_windows(train.len());
            for w in 0..fold_windows {
                calibrate(
                    &fp,
                    &arch,
                    &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
                    &mut calib,
                );
            }
            let (folded, farch) = fold(&fp, &shapes, &arch, &calib, alpha);
            println!("salience fold applied first: alpha={alpha}, {fold_windows} windows");
            (farch, folded)
        }
    };
    let ppl_fp_check = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    assert!(
        (ppl_fp_check - ppl_fp).abs() < 1e-3 * ppl_fp,
        "the fold must be function-preserving: {ppl_fp_check} vs {ppl_fp}"
    );

    // One pass per calibration window taps every layer.
    let mut grams = GramSet::new(&arch);
    for w in 0..calib_windows {
        grams.accumulate_forward(&fp, &arch, &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ]);
    }
    let n_layers = arch.n_layers;
    let attn: Vec<Vec<f64>> = std::mem::take(&mut grams.attn)
        .into_iter()
        .map(|g| g.finish())
        .collect();
    let ffn: Vec<Vec<f64>> = std::mem::take(&mut grams.ffn)
        .into_iter()
        .map(|g| g.finish())
        .collect();
    let down: Vec<Vec<f64>> = std::mem::take(&mut grams.down)
        .into_iter()
        .map(|g| g.finish())
        .collect();
    println!(
        "grams collected: {n_layers} layers x 3 taps, {calib_windows} windows = {} tokens\n",
        calib_windows * CALIB_SEQ
    );

    // Invert once per tap (q/k/v share attn; gate/up share ffn).
    let inv = |h: &[f64], k: usize| damped_inverse(h, k, DAMP);
    let attn_inv: Vec<Option<Vec<f64>>> = attn.iter().map(|h| inv(h, arch.n_embd)).collect();
    let ffn_inv: Vec<Option<Vec<f64>>> = ffn.iter().map(|h| inv(h, arch.n_embd)).collect();
    let down_inv: Vec<Option<Vec<f64>>> = down.iter().map(|h| inv(h, arch.ff)).collect();
    let failed = attn_inv.iter().filter(|v| v.is_none()).count()
        + ffn_inv.iter().filter(|v| v.is_none()).count()
        + down_inv.iter().filter(|v| v.is_none()).count();
    println!(
        "inverses: {} ok, {failed} not positive-definite after damping\n",
        3 * n_layers - failed
    );

    println!(
        "{:<22} {:>8} {:>13} {:>10}",
        "configuration", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(58));
    let (decays, shape) = decay_arms();
    for t in planes_under_test() {
        let bpw = ste::ternary_bits_per_weight(t, GROUP) + 1.0 / GROUP as f64;

        let base: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| plain(w, n, k, t))
            .collect();
        let p_base = perplexity_windowed(&base, &arch, &eval, EVAL_WINDOW);
        println!(
            "{:<22} {bpw:>8.2} {p_base:>13.3} {:>9.2}×",
            format!("T={t} plain"),
            p_base / ppl_fp
        );

        // Same weights, but the six tapped projections go through GPTQ feedback.
        for &decay in &decays {
            let mut fed = base.clone();
            for li in 0..n_layers {
                let b = 1 + 7 * li;
                let jobs: [(usize, &Option<Vec<f64>>); 6] = [
                    (b, &attn_inv[li]),
                    (b + 1, &attn_inv[li]),
                    (b + 2, &attn_inv[li]),
                    (b + 4, &ffn_inv[li]),
                    (b + 5, &ffn_inv[li]),
                    (b + 6, &down_inv[li]),
                ];
                for (idx, h_inv) in jobs {
                    if let Some(h) = h_inv {
                        let (n, k) = shapes[idx];
                        if let Some(q) = gptq(&fp[idx], n, k, t, h, decay, shape) {
                            fed[idx] = q;
                        }
                    }
                }
            }
            let p_fed = perplexity_windowed(&fed, &arch, &eval, EVAL_WINDOW);
            println!(
                "{:<22} {bpw:>8.2} {p_fed:>13.3} {:>9.2}×   ({:+.1}% vs plain)",
                if decay == 1.0 {
                    format!("T={t} +GPTQ")
                } else {
                    format!("T={t} +GPTQ {shape:?} λ={decay}")
                },
                p_fed / ppl_fp,
                (p_fed / p_base - 1.0) * 100.0
            );
        }
    }
    println!(
        "\nGPTQ covers 6 of 7 projections per block (q/k/v, gate/up, down). o_proj and the tied \
         embed/head keep the plain fit, so this understates what full coverage would give."
    );
}
