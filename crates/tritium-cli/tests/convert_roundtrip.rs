//! **Does `tritium convert` produce a model that is correct, not merely loadable?**
//!
//! `convert` already reloads its own artifact before reporting success, which catches the whole
//! class of "wrote a structurally valid file the loader never resolves" bugs — a tensor name the
//! loader does not look up, a shape the bundle disagrees with. That check cannot catch a *numeric*
//! error, and this command has one specific way to be numerically wrong:
//!
//! The salience fold scales projection columns by `s` and divides the preceding norm by `s`. The
//! two halves land in different files — the weights in `model.tslb`, the norms in
//! `model.safetensors`. If the norm half were dropped, mis-keyed, or written for the wrong layer,
//! the artifact would load perfectly and evaluate as a **badly damaged model**. Nothing structural
//! distinguishes that from a good run.
//!
//! # Everything here is a ratio, and that is the point
//!
//! The first version of this test asserted an absolute perplexity against 15.268, a number the
//! research harness measured. It failed by 26%, and **none of that was a defect in `convert`**.
//! `convert_anchor_diagnostic` decomposed it:
//!
//! | step | factor | cause |
//! |---|---|---|
//! | eval slice | 1.223× | the first 8,192 tokens are harder than the full 32,768 |
//! | evaluator | 1.000× | the tape and `ModelRunner` agree exactly on fp weights |
//! | fit → artifact bytes | 1.0003× | f16 block-scale rounding, at the floor |
//! | artifact → runtime | 1.0276× | **int8 activations** |
//!
//! That last row is not a bug and not this test's business, but it is why an anchor from the
//! research harness cannot be asserted here: `SaltLinear::forward` calls
//! `quantize_activation_int8` before every projection, while the harness scores dense
//! reconstructions through `Tape::dense_matmul` in fp32. The two measure genuinely different
//! systems. **Every published SALT perplexity is W-ternary/A-fp32; this path is W-ternary/A-int8.**
//!
//! So the fp reference is measured **here**, through the same runner and the same slice, and every
//! assertion is a ratio against it. A basis that is re-measured in-process cannot silently drift.
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-cli --release --test convert_roundtrip -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use tritium_nn::{ModelRunner, teacher_forced_perplexity_windows};

/// Window length for scoring. Matches the research harness so the shapes are comparable.
const EVAL_WINDOW: usize = 512;
/// Held-out tokens. The full 32,768-token split costs far more than this test is worth; because
/// every assertion is a ratio measured on this same slice, the slice only has to be *consistent*,
/// not representative.
const EVAL_TOKENS: usize = 8192;

/// Measured degradation of a `convert`-written T=4/g256 artifact relative to the fp master,
/// **through the shipping runtime** (so int8 activations are included): 19.263 / 18.240.
///
/// Deliberately not the research anchor's 15.268/14.909 = 1.0241, which is the same weights with
/// fp32 activations. This constant guards what a user actually runs.
const RUNTIME_RATIO_T4: f64 = 1.0560;
/// Tolerance on that ratio. The fit is deterministic and the slice is fixed, so this only absorbs
/// floating-point ordering — a real writer defect moves the ratio by percent, not by 0.5%.
const RATIO_TOLERANCE: f64 = 0.005;

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
    // alpha = 0 is the identity fold, so a corpus would only cost time — and passing one anyway
    // would make the two arms differ in more than the knob under test.
    if alpha != 0.0 {
        cmd.args(["--calib", corpus.to_str().unwrap()]);
    }
    let status = cmd.status().expect("run tritium convert");
    assert!(status.success(), "convert failed at alpha={alpha}");
}

fn score(mut runner: ModelRunner, tokens: &[u32]) -> f64 {
    teacher_forced_perplexity_windows(&mut runner, tokens, EVAL_WINDOW)
        .expect("score model")
        .perplexity
}

fn score_converted(dir: &Path, tokens: &[u32]) -> f64 {
    let runner = ModelRunner::from_salt(
        dir,
        &dir.join("model.tslb"),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap_or_else(|e| panic!("load converted model {}: {e}", dir.display()));
    score(runner, tokens)
}

#[test]
#[ignore = "three full evaluations of a real model on CPU"]
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

    // The reference, measured here rather than quoted: same runner, same slice, same window.
    let fp = score(
        ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load fp master"),
        &tokens,
    );
    println!("fp master                : {fp:.3}");

    convert(&model, &unfolded_dir, &corpus, 0.0);
    let unfolded = score_converted(&unfolded_dir, &tokens);
    println!(
        "convert --fold-alpha 0   : {unfolded:.3} ({:.4}x fp)",
        unfolded / fp
    );

    convert(&model, &folded_dir, &corpus, 0.75);
    let folded = score_converted(&folded_dir, &tokens);
    println!(
        "convert --fold-alpha 0.75: {folded:.3} ({:.4}x fp)",
        folded / fp
    );
    println!(
        "fold delta: {:+.3}% (fold WITHOUT rotation, through int8 activations — a configuration \
         no published number covers)",
        100.0 * (folded - unfolded) / unfolded
    );

    // The unfolded arm is a deterministic fit of unmodified weights, so its degradation against fp
    // is a fixed property of the pipeline. Any movement is the writer.
    let ratio = unfolded / fp;
    assert!(
        (ratio - RUNTIME_RATIO_T4).abs() <= RATIO_TOLERANCE,
        "convert --fold-alpha 0 degraded fp by {ratio:.4}x, expected {RUNTIME_RATIO_T4:.4}x \
         (+/-{RATIO_TOLERANCE}). The fit is deterministic and fp is re-measured in this same run, \
         so this is the conversion pipeline changing values."
    );

    // The fold is an exact reparameterisation: weights scaled by `s`, preceding norm divided by
    // `s`. If the norm half were dropped or mis-keyed the artifact would still load and would
    // evaluate as a wrecked model. Bounding the regression catches that with enormous margin,
    // without asserting a fold-without-rotation number nobody has measured.
    assert!(
        folded <= unfolded * 1.05,
        "folding made the model {:.1}% WORSE ({folded:.3} vs {unfolded:.3}). The fold is an exact \
         reparameterisation, so a regression this size means one half of it is missing — most \
         likely the folded norms in model.safetensors are not reaching the loader.",
        100.0 * (folded - unfolded) / unfolded
    );

    let _ = std::fs::remove_dir_all(&root);
}
