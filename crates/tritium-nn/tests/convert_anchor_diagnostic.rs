//! **Why does `tritium convert --fold-alpha 0` score 19.263 when the ladder anchor is 15.268?**
//!
//! `convert_roundtrip` compares a converted artifact against a number the research harness
//! produced. Those two numbers travel through *different machinery* in two independent ways, and
//! the failure does not say which one moved:
//!
//! | | anchor (15.268) | `convert_roundtrip` (19.263) |
//! |---|---|---|
//! | weights scored | the fitter's dense f32 reconstruction | TQ2_0-packed planes, f16 block scales |
//! | forward | `common::perplexity_windowed` (tape, whole window at once) | `ModelRunner` teacher-forced, token by token |
//! | eval tokens | all 32,768 | the first 8,192 |
//!
//! Any of the three could account for a 26% gap, so this measures a 2×2 on **one fixed slice**:
//!
//! - **A** fp weights, tape forward → the fp reference for this evaluator
//! - **B** fp weights, `ModelRunner` forward → A vs B isolates the **evaluator**
//! - **C** ladder fit, tape forward → reproduces the anchor's method; C vs the 15.268 anchor
//!   isolates the **eval slice**
//! - **D** the converted artifact, `ModelRunner` forward → C vs D isolates **fit vs artifact**,
//!   which is the only cell where a bug in `tritium convert` could live
//!
//! Ratios are what matter. Comparing D against an absolute number measured on a different slice
//! with a different forward was the mistake that produced the confusing failure in the first place.
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//! TRITIUM_CONVERTED_DIR=/tmp/tritium-convert-rt-3799551/unfolded \
//!   cargo test -p tritium-nn --release --test convert_anchor_diagnostic -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::{ModelRunner, teacher_forced_perplexity_windows};
use tritium_train::ops::ste::{self, RotationPolicy};

const WINDOW: usize = 512;
const TOKENS: usize = 8192;
const PLANES: usize = 4;
const GROUP: usize = 256;
const GRID: usize = 16;

/// The published anchor this diagnostic exists to explain: SmolLM2-360M, ladder T=4, g256, no
/// fold, no rotation, measured by the research harness on the full eval split.
const ANCHOR: f64 = 15.268;
/// The fp master under the same harness and split.
const ANCHOR_FP: f64 = 14.909;

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

fn eval_tokens() -> Vec<u32> {
    let bytes = std::fs::read(env_path("TRITIUM_CORPUS")).expect("read corpus");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("parse corpus");
    value["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .take(TOKENS)
        .collect()
}

fn runner_ppl(load: impl FnOnce() -> ModelRunner, tokens: &[u32]) -> f64 {
    let mut runner = load();
    teacher_forced_perplexity_windows(&mut runner, tokens, WINDOW)
        .expect("score")
        .perplexity
}

#[test]
#[ignore = "four full evaluations of a 360M model on CPU"]
fn locate_the_gap_between_the_artifact_and_the_anchor() {
    let model = env_path("TRITIUM_MODEL_DIR");
    let converted = env_path("TRITIUM_CONVERTED_DIR");
    let tokens = eval_tokens();
    assert_eq!(tokens.len(), TOKENS);

    let runner = ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
        .expect("load fp master");
    let (arch, fp, shapes) = extract(&runner);
    drop(runner);

    let a = perplexity_windowed(&fp, &arch, &tokens, WINDOW);
    println!("A  fp        / tape    : {a:.3}");

    let b = runner_ppl(
        || {
            ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
                .expect("load fp master")
        },
        &tokens,
    );
    println!("B  fp        / runner  : {b:.3}");

    // The anchor's own recipe, dense: no fold, no rotation, geometric ladder at T=4/g256.
    let fitted: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(n, k))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                n,
                k,
                PLANES,
                GROUP,
                GRID,
                RotationPolicy::Never,
            )
        })
        .collect();
    let c = perplexity_windowed(&fitted, &arch, &tokens, WINDOW);
    println!("C  ladder    / tape    : {c:.3}");

    let d = runner_ppl(
        || {
            ModelRunner::from_salt(
                &converted,
                &converted.join("model.tslb"),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .expect("load converted artifact")
        },
        &tokens,
    );
    println!("D  artifact  / runner  : {d:.3}");

    println!("\n--- attribution ---");
    println!("evaluator  (B/A)           : {:.4}x", b / a);
    println!("slice      (C vs {ANCHOR:.3})    : {:.4}x", c / ANCHOR);
    println!("fit->artifact (D/C)        : {:.4}x", d / c);
    println!(
        "degradation vs fp, tape    : {:.4}x  (anchor: {:.4}x)",
        c / a,
        ANCHOR / ANCHOR_FP
    );
    println!("degradation vs fp, runner  : {:.4}x", d / b);
    println!(
        "\nThe quantity that must match the anchor is the RATIO to fp under the SAME evaluator: \
         D/B = {:.4}x versus the anchor's {:.4}x.",
        d / b,
        ANCHOR / ANCHOR_FP
    );
}
