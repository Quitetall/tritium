//! **Does `tritium convert` produce a model that is correct, not merely loadable?**
//!
//! `convert` already reloads its own artifact before reporting success, which catches the whole
//! class of "wrote a structurally valid file the loader never resolves" bugs — a tensor name the
//! loader does not look up, a shape the bundle disagrees with. That check cannot catch a *numeric*
//! error, and this command has one specific way to be numerically wrong:
//!
//! The salience fold scales projection columns by `s` and divides the preceding norm by `s`. Both
//! halves land in different files — the weights in `model.tslb`, the norms in `model.safetensors`.
//! If the norm half were dropped, mis-keyed, or written for the wrong layer, the artifact would
//! load perfectly and evaluate as a **badly damaged model**. Nothing structural distinguishes that
//! from a good run.
//!
//! So this scores the converted model on held-out text and compares it against measured anchors.
//!
//! # Anchors (SmolLM2-360M, WikiText-2 held-out, g256, no rotation)
//!
//! | configuration | perplexity | ×fp |
//! |---|---|---|
//! | fp16 master | 14.909 | 1.000 |
//! | ladder T=4, **no fold** | 15.268 | 1.024 |
//!
//! `convert --fold-alpha 0` must reproduce the no-fold row: it runs the identical fitter on
//! identical weights, so a difference means the *writer* changed the values.
//!
//! Fold-without-rotation has never been measured — the published numbers are fold **and** rotation.
//! The test therefore asserts a bound rather than a target: the folded model must not be *worse*
//! than the unfolded one by more than noise. A dropped norm half would blow through that by orders
//! of magnitude, which is the failure this exists to catch.
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-cli --release --test convert_roundtrip -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use tritium_nn::{ModelRunner, teacher_forced_perplexity_windows};

/// Window length for scoring. Matches the research harness so the numbers are comparable.
const EVAL_WINDOW: usize = 512;
/// Held-out tokens. The full 32,768-token eval split takes far longer than this test is worth;
/// 8,192 is stable to well under the margins asserted here.
const EVAL_TOKENS: usize = 8192;

/// Measured perplexity of the fp16 master, for the ×fp column in the printout.
const FP_PPL: f64 = 14.909;
/// Measured perplexity of ladder T=4 / g256 / no fold / no rotation.
const UNFOLDED_ANCHOR: f64 = 15.268;
/// How far the unfolded conversion may drift from its anchor. The fit is deterministic, so this is
/// only absorbing the difference between this eval slice and the harness's.
const ANCHOR_TOLERANCE: f64 = 0.10;

fn tritium_bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("tritium")
}

fn eval_tokens(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("parse corpus json");
    value["eval_ids"]
        .as_array()
        .expect("corpus has eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .take(EVAL_TOKENS)
        .collect()
}

fn convert(model: &Path, out: &Path, corpus: &Path, alpha: f64) {
    let _ = std::fs::remove_dir_all(out);
    let mut cmd = Command::new(tritium_bin());
    cmd.args([
        "convert",
        "--model",
        model.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--planes",
        "4",
        "--group",
        "256",
        "--fold-alpha",
        &alpha.to_string(),
    ]);
    // alpha = 0 is the identity fold, so a corpus would only cost time. Passing one anyway would
    // also make the two arms differ in more than the knob under test.
    if alpha != 0.0 {
        cmd.args(["--calib", corpus.to_str().unwrap()]);
    }
    let status = cmd.status().expect("run tritium convert");
    assert!(status.success(), "convert failed at alpha={alpha}");
}

fn perplexity(dir: &Path, tokens: &[u32]) -> f64 {
    let mut runner = ModelRunner::from_salt(
        dir,
        &dir.join("model.tslb"),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap_or_else(|e| panic!("load converted model {}: {e}", dir.display()));
    teacher_forced_perplexity_windows(&mut runner, tokens, EVAL_WINDOW)
        .expect("score converted model")
        .perplexity
}

#[test]
#[ignore = "needs a real fp master and corpus; minutes, not seconds"]
fn converted_model_scores_like_the_fitter_that_made_it() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let corpus = PathBuf::from(
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a corpus json"),
    );
    let tokens = eval_tokens(&corpus);
    assert_eq!(tokens.len(), EVAL_TOKENS, "corpus is too small");

    let root = std::env::temp_dir().join(format!("tritium-convert-rt-{}", std::process::id()));
    let unfolded_dir = root.join("unfolded");
    let folded_dir = root.join("folded");

    convert(&model, &unfolded_dir, &corpus, 0.0);
    let unfolded = perplexity(&unfolded_dir, &tokens);
    println!(
        "convert --fold-alpha 0   : {unfolded:.3} ({:.3}x fp)",
        unfolded / FP_PPL
    );

    convert(&model, &folded_dir, &corpus, 0.75);
    let folded = perplexity(&folded_dir, &tokens);
    println!(
        "convert --fold-alpha 0.75: {folded:.3} ({:.3}x fp)",
        folded / FP_PPL
    );
    println!(
        "fold delta: {:+.3}% (fold-without-rotation; the published numbers are fold AND rotation)",
        100.0 * (folded - unfolded) / unfolded
    );

    // The unfolded arm runs the same fitter on the same weights as the measured anchor, so any
    // difference is the writer, not the method.
    let drift = (unfolded - UNFOLDED_ANCHOR).abs() / UNFOLDED_ANCHOR;
    assert!(
        drift <= ANCHOR_TOLERANCE,
        "convert --fold-alpha 0 scored {unfolded:.3}, but ladder T=4/g256/no-fold/no-rotation \
         measures {UNFOLDED_ANCHOR:.3} ({:.1}% off). The fit is deterministic, so this is the \
         writer changing values, not the method.",
        100.0 * drift
    );

    // The fold's whole point is that it is an exact reparameterisation: weights scaled by `s`,
    // preceding norm divided by `s`. If the norm half were dropped or mis-keyed the artifact would
    // still load and would evaluate as a wrecked model. Bounding the regression catches that with
    // enormous margin, without asserting a number nobody has measured.
    assert!(
        folded <= unfolded * 1.05,
        "folding made the model {:.1}% WORSE ({folded:.3} vs {unfolded:.3}). The fold is an exact \
         reparameterisation, so a real regression this size means one half of it is missing — \
         most likely the folded norms in model.safetensors are not reaching the loader.",
        100.0 * (folded - unfolded) / unfolded
    );

    let _ = std::fs::remove_dir_all(&root);
}
