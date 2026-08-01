//! **Activation-aware SALT fitting** — the half of PT²-LLM we were missing.
//!
//! SALT minimises `‖W − Ŵ‖²`. Every SOTA PTQ method since GPTQ minimises `‖(W − Ŵ)X‖²` instead,
//! because a weight column that multiplies a large activation deserves more of the quantization
//! budget than one that multiplies noise. PT²-LLM is exactly two halves — Iterative Ternary Fitting
//! (landed) and Activation-aware Grid Alignment (this file). The full-stack sweep measured the cost
//! of the missing half: 1.74× fp at 6.38 bpw, *worse* than plain RTN int4 at 4.25 bpw.
//!
//! For weight-only quantization the diagonal-Hessian objective has an exact, inference-free
//! implementation (AWQ's): scale weight column `j` by `s_j = (rms_j / geomean)^α` derived from the
//! input channel's calibration RMS, and push `1/s_j` back into a tensor that costs nothing. In this
//! architecture that fold is *exact and free* for six of the seven projections per layer:
//!
//! | projection | `1/s` absorbed by | why it is free |
//! |---|---|---|
//! | q, k, v | `attn_norm[j]` | the norm gain is fp and per-channel, never quantized |
//! | gate, up | `ffn_norm[j]` | same |
//! | down | `up`'s output **row** `j` | rows already carry a free per-row scale in the fitter |
//!
//! `o_proj` is skipped: under GQA several query heads share one kv head, so its input channels are
//! not independently scalable without a group-wise compromise. The tied embed/head is skipped
//! because the same matrix is read row-wise as an embedding — scaling its columns would perturb the
//! residual stream, and no compensating tensor exists while the weights are tied.
//!
//! [`fold_is_exact_in_fp`] is the load-bearing gate: with quantization off, folding must leave the
//! model's perplexity **bit-identical**. If that fails, every number below is meaningless.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_smoothing_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{
    Arch, Calib, calibrate, extract, fold, perplexity_windowed, quantize, quantize_split, split_bpw,
};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste;

const EVAL_WINDOW: usize = 512;
/// Calibration windows drawn from the *training* split — never from the held-out set.
const CALIB_WINDOWS: usize = 8;

/// `TRITIUM_MODEL_DIR` selects the model, so the same sweep runs across a scale curve. Every SOTA
/// ternary number is reported at 7B+, where there is far more redundancy to spend on quantization
/// error; SmolLM2-135M is the hardest case anyone benchmarks. The SmolLM2 family shares one
/// tokenizer (vocab 49152) and ties its head at every size, so 135M/360M/1.7B are directly
/// comparable on the same corpus token ids with no harness change.
fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TRITIUM_MODEL_DIR") {
        return PathBuf::from(dir);
    }
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
    let ids = |key: &str| -> Vec<u32> {
        j[key]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

type Fixture = (Arch, Vec<Vec<f32>>, Vec<(usize, usize)>, Calib, Vec<u32>);

fn setup() -> Option<Fixture> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return None;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    assert!(
        train.len() >= CALIB_WINDOWS * EVAL_WINDOW,
        "need a training split for calibration; got {} ids",
        train.len()
    );
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * EVAL_WINDOW..(w + 1) * EVAL_WINDOW],
            &mut calib,
        );
    }
    Some((arch, fp, shapes, calib, eval))
}

/// **The gate.** The fold is an algebraic reparameterisation: with quantization off it must leave
/// the model's output unchanged. Anything else means the inverse landed in the wrong tensor, and
/// every quality number measured on top of it would be measuring that bug instead.
///
/// The tolerance is `1e-6` relative — tight enough that a swapped `*`/`÷`, a fold applied to the
/// wrong axis, or an inverse dropped on one of the six projections all fail it, while still leaving
/// room for f32 rounding accumulated through 30 layers of scaled norms. The observed drift is ~4e-8
/// relative, two orders of magnitude inside the bound.
#[test]
#[ignore = "needs SmolLM2-135M; run explicitly"]
fn fold_is_exact_in_fp() {
    let Some((arch, fp, shapes, calib, eval)) = setup() else {
        return;
    };
    let reference = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    for alpha in [0.5f64, 1.0] {
        let (folded, farch) = fold(&fp, &shapes, &arch, &calib, alpha);
        let got = perplexity_windowed(&folded, &farch, &eval, EVAL_WINDOW);
        println!("α={alpha}: fp {reference:.9} → folded {got:.9}");
        assert!(
            (got - reference).abs() < 1e-6 * reference,
            "the fold must be function-preserving in fp: {got} vs {reference}"
        );
    }
}

/// The measurement: does weighting the fit by activation salience close the gap to int4 — and does
/// it rescue T=1, the 2.13 bpw configuration that is the only apples-to-apples comparison against
/// shipping ternary formats?
#[test]
#[ignore = "slow activation-aware PTQ sweep; needs SmolLM2-135M; run explicitly"]
fn activation_aware_fitting_closes_the_gap() {
    let Some((arch, fp, shapes, calib, eval)) = setup() else {
        return;
    };
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    println!(
        "fp reference {ppl_fp:.3} ppl | calibrated on {} train tokens\n",
        calib.rows
    );
    println!(
        "{:<44} {:>8} {:>14} {:>11}",
        "configuration", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(80));

    for t in 1..=3usize {
        for group in [128usize] {
            for alpha in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
                let (folded, farch) = fold(&fp, &shapes, &arch, &calib, alpha);
                let q = quantize(&folded, &shapes, t, group);
                let p = perplexity_windowed(&q, &farch, &eval, EVAL_WINDOW);
                let bpw = ste::ternary_bits_per_weight(t, group) + 1.0 / group as f64;
                println!(
                    "{:<44} {:>8.2} {:>14.3} {:>10.2}×",
                    format!("T={t} g{group} +ITF +Had  α={alpha:.2}"),
                    bpw,
                    p,
                    p / ppl_fp
                );
            }
            println!();
        }
    }
    println!(
        "α=0 reproduces the committed fitter exactly, so each block reads as a pure ablation of \
         activation salience. The fold is free at inference: every inverse lands in an fp norm gain \
         or in a row scale the fitter already carries."
    );
}

/// **Where the remaining T=1 error lives.** The tied embed/head is 21% of the parameters, and as the
/// LM head it is a 49k×576 matmul whose error lands directly on the logits — with a single plane,
/// that is very nearly a random projection. The earlier ablation showed it carrying 2.2× even at
/// T=3, and salience folding cannot reach it: rescaling the residual stream per channel is not
/// invertible through RMSNorm (`rmsnorm(x·s) ≠ rmsnorm(x)·s`), so no compensating tensor exists.
///
/// The fix that preserves the thesis is **more planes, not more bits**. Keeping the head in int8
/// would break multiply-freeness on the single largest decode matmul; giving it extra ternary planes
/// keeps every row multiply-free and still lands under int4 overall.
///
/// `TRITIUM_SMOOTH_ALPHA` selects the fold strength, and **it must be calibrated per model** — the
/// optimum shifts down as the model grows, so inheriting another model's value is a real error, not
/// a rounding one:
///
/// | model | α* | penalty at 135M's α=0.75 |
/// |---|---|---|
/// | SmolLM2-135M | 0.75 | — |
/// | SmolLM2-360M | 0.50 | 4.2× worse at T=1 (2915× → 12296×) |
///
/// Both curves are U-shaped for the same reason — past the optimum, salience weighting inflates the
/// weight matrix's own dynamic range faster than it buys resolution where the activations are — but
/// the turning point is a property of the model, not a constant. The 0.75 default below is correct
/// **only for the default 135M**; set the variable when pairing this with `TRITIUM_MODEL_DIR`.
#[test]
#[ignore = "slow head-allocation sweep; needs SmolLM2-135M; run explicitly"]
fn planes_where_they_matter_rescue_the_head() {
    let Some((arch, fp, shapes, calib, eval)) = setup() else {
        return;
    };
    let alpha: f64 = std::env::var("TRITIUM_SMOOTH_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.75);
    let group = 128usize;
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let (folded, farch) = fold(&fp, &shapes, &arch, &calib, alpha);
    println!("fp reference {ppl_fp:.3} ppl | salience fold α={alpha}\n");
    println!(
        "{:<44} {:>8} {:>14} {:>11}",
        "configuration", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(80));

    for (t_head, t_body) in [
        (1usize, 1usize),
        (2, 1),
        (3, 1),
        (4, 1),
        (3, 2),
        (4, 2),
        (4, 3),
    ] {
        let q = quantize_split(&folded, &shapes, t_head, t_body, group);
        let p = perplexity_windowed(&q, &farch, &eval, EVAL_WINDOW);
        println!(
            "{:<44} {:>8.2} {:>14.3} {:>10.2}×",
            format!("head T={t_head}, body T={t_body}  g{group} α={alpha:.2}"),
            split_bpw(&shapes, t_head, t_body, group),
            p,
            p / ppl_fp
        );
    }
    println!(
        "\nEvery row is fully ternary — no int8 island, no multipliers. `head T=1, body T=1` is the \
         uniform 2.13 bpw baseline; the rest spend their extra bits only where the logits see them."
    );
}
