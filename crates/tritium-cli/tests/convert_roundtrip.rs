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

fn convert_with(model: &Path, out: &Path, corpus: &Path, alpha: f64, rotate: bool) {
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
    if !rotate {
        cmd.arg("--no-rotation");
    }
    if alpha != 0.0 {
        cmd.args(["--calib", corpus.to_str().unwrap()]);
    }
    let status = cmd.status().expect("run tritium convert");
    assert!(
        status.success(),
        "convert failed at alpha={alpha} rotate={rotate}"
    );
}

fn convert(model: &Path, out: &Path, corpus: &Path, alpha: f64) {
    // Historic arms predate rotation and must stay in the original basis to remain comparable.
    // `alpha = 0` is the identity fold, so a corpus would only cost time — and passing one anyway
    // would make the two arms differ in more than the knob under test.
    convert_with(model, out, corpus, alpha, false);
}

/// The Hadamard group an artifact declares, or `None` for an unrotated (version-1) bundle.
fn bundle_rotation_group(dir: &Path) -> Option<usize> {
    let file = std::fs::File::open(dir.join("model.tslb")).expect("open bundle");
    tritium_format::SaltBundleReader::new_strict(std::io::BufReader::new(file))
        .expect("parse bundle")
        .rotation_group()
        .map(usize::from)
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

/// **Does rotation survive into the artifact, and is it worth what the tape said?**
///
/// The tape measured rotation at **21.1% at T=3** and 1.6% at T=4 on SmolLM2-135M. That number was
/// unreachable from any artifact until the bundle learned to record a rotation group: a rotated fit
/// stores codes in the rotated basis, and a runtime that does not rotate the activation computes
/// `W·H·x` instead of `W·x` — silently.
///
/// This asserts the loop is closed end to end. Both arms are converted the same way and scored
/// through the same `ModelRunner`, so the int8-activation factor that makes an absolute tape anchor
/// unusable here (see the header) divides out: what is left is rotation alone.
///
/// Deliberately a **ratio with a loose floor**, not an anchor. `convert` runs at `--planes 4
/// --group 256` where the tape measured rotation worth only ~1.6%, and the artifact adds A8 on top,
/// so demanding the tape's T=3 figure would be asserting a number nobody measured in this
/// configuration. The claim under test is directional and falsifiable: rotation must not make the
/// artifact worse, and the bundle must round-trip as a rotated one.
#[test]
#[ignore = "two full evaluations of a real model on CPU"]
fn rotation_reaches_the_artifact_and_does_not_cost_quality() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let corpus = PathBuf::from(
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a corpus json"),
    );
    let tokens = eval_tokens(&corpus);
    assert_eq!(tokens.len(), EVAL_TOKENS, "corpus is too small");
    let tmp = std::env::temp_dir().join(format!("tritium-rot-{}", std::process::id()));
    let plain = tmp.join("plain");
    let rotated = tmp.join("rotated");

    // Same fold on both arms so rotation is the only difference.
    convert_with(&model, &plain, &corpus, 0.75, false);
    convert_with(&model, &rotated, &corpus, 0.75, true);

    // The artifact must SAY it is rotated, or the runtime will not rotate and the codes are wrong.
    // Asked through the reader rather than by indexing a header byte, so a future header field
    // cannot leave this assertion quietly reading something else.
    assert_eq!(
        bundle_rotation_group(&rotated),
        Some(256),
        "a rotated conversion must record its Hadamard group; without it the runtime reconstructs \
         W·H and the model is silently wrong"
    );
    assert_eq!(
        bundle_rotation_group(&plain),
        None,
        "--no-rotation must still write a bundle that declares no rotation, which is what every \
         reader predating version 2 can load"
    );

    let ppl_plain = score_converted(&plain, &tokens);
    let ppl_rotated = score_converted(&rotated, &tokens);
    let _ = std::fs::remove_dir_all(&tmp);

    println!(
        "convert --planes 4 --group 256 --fold-alpha 0.75\n  \
         unrotated (v1) {ppl_plain:.4}\n  rotated   (v2) {ppl_rotated:.4}   ({:+.2}%)",
        100.0 * (ppl_rotated - ppl_plain) / ppl_plain
    );

    assert!(
        ppl_rotated.is_finite() && ppl_rotated > 0.0,
        "rotated artifact scored {ppl_rotated}, which means the runtime rotation and the fit \
         disagree — the codes are in one basis and the activation in the other"
    );
    // A 1% tolerance, not equality: rotation changes the fit, and at T=4/g256 the tape put its
    // worth at ~1.6%, which A8 can plausibly mask. A wrong-basis runtime would be off by orders of
    // magnitude, not percent, so this still catches the failure it exists to catch.
    assert!(
        ppl_rotated <= ppl_plain * 1.01,
        "rotation made the artifact WORSE ({ppl_plain:.4} -> {ppl_rotated:.4}). The tape says \
         rotation helps, so this means the runtime is not applying the same transform the fitter \
         did — check group width and the order of rotate-then-quantize."
    );
}

/// **How much of the int8-activation tax do per-group scales recover in the shipping runtime?**
///
/// `quantize_activation_int8` takes one absmax per token over the whole row, so one outlier sets
/// the step for every value in it. The research tape says moving to per-group scales recovers
/// **exactly 64%** of the A8 tax at both weight settings tested (+1.19% → +0.43% at `T=3`,
/// +1.08% → +0.43% at `T=4`) for no additional bits — activation scales are transient.
///
/// That was measured through `forward_aq`, which is not what anybody runs. This measures it where
/// it would ship: one converted artifact, four evaluations through `ModelRunner`, only the
/// activation granularity moving.
///
/// Prediction, recorded before the first run so being wrong is visible: per-group is better than
/// per-token by a few tenths of a percent, and `g128` is at least as good as `g256`.
#[test]
#[ignore = "four full evaluations of a real model on CPU"]
fn per_group_activations_recover_part_of_the_a8_tax() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let corpus = PathBuf::from(
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a corpus json"),
    );
    let tokens = eval_tokens(&corpus);
    assert_eq!(tokens.len(), EVAL_TOKENS, "corpus is too small");

    let dir = std::env::temp_dir().join(format!("tritium-a8g-{}", std::process::id()));
    convert_with(&model, &dir, &corpus, 0.75, true);

    let fp = score(
        ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load fp master"),
        &tokens,
    );

    // Same artifact every time; only `set_salt_activation_group` differs.
    let score_at = |group: Option<usize>| -> f64 {
        let mut runner = ModelRunner::from_salt(
            &dir,
            &dir.join("model.tslb"),
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .expect("load converted model");
        let touched = runner.weights.set_salt_activation_group(group);
        assert!(
            touched > 0,
            "no SALT projection was configured — the model is not on the path under test, so a \
             flat result here would mean nothing"
        );
        score(runner, &tokens)
    };

    let per_token = score_at(None);
    let g256 = score_at(Some(256));
    let g128 = score_at(Some(128));
    let g64 = score_at(Some(64));
    let _ = std::fs::remove_dir_all(&dir);

    println!("\nSmolLM2-135M | convert --planes 4 --group 256, fold 0.75, rotated | fp {fp:.4}");
    println!(
        "{:<34} {:>11} {:>9} {:>14}",
        "activation scales", "ppl", "x fp", "vs per-token"
    );
    println!("{}", "-".repeat(72));
    for (label, ppl) in [
        ("per token (SHIPPING)", per_token),
        ("per group g256", g256),
        ("per group g128", g128),
        ("per group g64", g64),
    ] {
        let delta = if (ppl - per_token).abs() < 1e-12 {
            "—".to_owned()
        } else {
            format!("{:+.2}%", 100.0 * (ppl - per_token) / per_token)
        };
        println!("{label:<34} {ppl:>11.4} {:>8.4}x {delta:>14}", ppl / fp);
    }

    assert!(
        [g256, g128, g64].iter().all(|p| p.is_finite() && *p > 0.0),
        "a per-group arm did not produce a usable model"
    );
    // The directional claim. Narrower groups can only reduce quantization error (each group's
    // absmax is at most the row's), so a per-group arm scoring WORSE means the plumbing is wrong,
    // not that the idea failed.
    assert!(
        g128 <= per_token,
        "per-group g128 ({g128:.4}) is worse than per-token ({per_token:.4}). Each group's absmax \
         is bounded by the row's, so this cannot be a property of the quantizer — check that the \
         dequantized values are reaching the GEMM and that the per-token scale fold is not being \
         applied twice"
    );
}
