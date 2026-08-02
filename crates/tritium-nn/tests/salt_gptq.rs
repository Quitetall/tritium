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

use common::{GramSet, damped_inverse, extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_quantize::{ColumnGroup, FeedbackMetric, FeedbackProblem, fit_with_feedback};
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
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
fn gptq(w: &[f32], rows: usize, cols: usize, t: usize, h_inv: &[f64]) -> Option<Vec<f32>> {
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
    let state = fit_with_feedback(problem, |req: tritium_quantize::GroupFitRequest<'_>| {
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
        Ok::<Vec<f64>, std::convert::Infallible>(fit.into_iter().map(f64::from).collect())
    })
    .ok()?;
    Some(state.reconstruction().iter().map(|&v| v as f32).collect())
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
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // One pass per calibration window taps every layer.
    let mut grams = GramSet::new(&arch);
    for w in 0..CALIB_WINDOWS {
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
    println!("grams collected: {n_layers} layers x 3 taps, {CALIB_WINDOWS} windows\n");

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
    for t in [1usize, 2, 3] {
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
                    if let Some(q) = gptq(&fp[idx], n, k, t, h) {
                        fed[idx] = q;
                    }
                }
            }
        }
        let p_fed = perplexity_windowed(&fed, &arch, &eval, EVAL_WINDOW);
        println!(
            "{:<22} {bpw:>8.2} {p_fed:>13.3} {:>9.2}×   ({:+.1}% vs plain)",
            format!("T={t} +GPTQ"),
            p_fed / ppl_fp,
            (p_fed / p_base - 1.0) * 100.0
        );
    }
    println!(
        "\nGPTQ covers 6 of 7 projections per block (q/k/v, gate/up, down). o_proj and the tied \
         embed/head keep the plain fit, so this understates what full coverage would give."
    );
}
