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
    convert_full(model, out, corpus, alpha, rotate, 4, false);
}

fn convert_full(
    model: &Path,
    out: &Path,
    corpus: &Path,
    alpha: f64,
    rotate: bool,
    planes: usize,
    dense: bool,
) {
    let _ = std::fs::remove_dir_all(out);
    let mut cmd = Command::new(tritium_bin());
    cmd.args([
        "convert",
        "--model",
        model.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--planes",
        &planes.to_string(),
        "--group",
        "256",
        "--fold-alpha",
        &alpha.to_string(),
    ]);
    if !rotate {
        cmd.arg("--no-rotation");
    }
    if dense {
        cmd.arg("--dense-container");
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
    let path = dir.join("model.tslb");
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(
        &mut std::fs::File::open(&path).expect("open bundle"),
        &mut magic,
    )
    .expect("read magic");
    let file = std::io::BufReader::new(std::fs::File::open(&path).expect("open bundle"));
    // `convert` writes TSLJ by default and TQ2_0 under --dense-container; both record the group.
    if magic == tritium_format::salt_joint_bundle::SALT_JOINT_BUNDLE_MAGIC {
        tritium_format::salt_joint_bundle::JointSaltBundleReader::new_strict(file)
            .expect("parse TSLJ bundle")
            .rotation_group()
    } else {
        tritium_format::SaltBundleReader::new_strict(file)
            .expect("parse TSLB bundle")
            .rotation_group()
    }
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

/// **Do per-group activation scales still help once the activation is rotated?**
///
/// The A8 headroom measurement found **5.41 dB** available on the FFN intermediate from per-group
/// scales, and through the research tape that converted to recovering **exactly 64% of the A8 tax**
/// at both weight settings (+1.19% → +0.43% at `T=3`, +1.08% → +0.43% at `T=4`). On that evidence
/// this was the highest-ranked open lever in the plan: a quality win for zero bits.
///
/// Measured here, in the shipping runtime, it is **worth nothing** — and the reason is mechanical.
///
/// # The two are substitutes, and rotation got there first
///
/// Per-group scales pay off in proportion to how *concentrated* the outliers are: `gain =
/// (cols·γ_row²) / Σ_g(width_g·γ_g²)`, which is `1` for a flat row and large only when the row's
/// maximum lives in one group. The Hadamard does exactly the thing that flattens that ratio — it
/// mixes every coordinate into every other, which is why rotation helps the ladder in the first
/// place — and `SaltLinear::forward` applies it to the activation **before** quantizing.
///
/// `forward_aq`, where the +1.19% → +0.43% was measured, does **not** rotate activations. So the
/// tape measured per-group scales against an unrotated distribution with its outliers intact. Both
/// numbers are right; they are measurements of different activations.
///
/// # Design
///
/// Two artifacts, identical but for `--no-rotation`, four activation granularities each, all eight
/// through the same `ModelRunner`. Ratios within an arm, so the comparison is exact.
///
/// Prediction, recorded before the run: the unrotated artifact reproduces something like the tape's
/// gap; the rotated one stays flat. If **both** are flat the hypothesis is wrong and the tape's
/// number needs a different explanation.
///
/// # Measured 2026-09-16 — the direction holds, the magnitude does not
///
/// ```text
/// SmolLM2-135M | --planes 4 --group 256 --fold-alpha 0.75 | fp 27.8762
/// activation scales          rotated       vs A8   unrotated       vs A8
/// per token (SHIPPING)       28.2470           —     29.3795           —
/// per group g256             28.2521      +0.02%     29.3739      -0.02%
/// per group g128             28.2679      +0.07%     29.3192      -0.21%
/// per group g64              28.2445      -0.01%     29.3096      -0.24%
/// ```
///
/// Unrotated, per-group scales help **monotonically** as the group narrows — the signature of the
/// mechanism actually operating. Rotated, the three arms scatter within ±0.07% of the baseline,
/// which is noise. The substitution account is confirmed.
///
/// But the size settles the lever: **0.24% at best**, against **3.85%** for rotation on the same
/// artifact (29.3795 → 28.2470). Rotation is ~16× larger and collects the same win by the same
/// mechanism, so there is no configuration where this is the thing to reach for. The plan ranked it
/// the top open lever on the tape's +1.19% → +0.43%; measured where it would ship, it is dead.
#[test]
#[ignore = "ten full evaluations of a real model on CPU; the per-group arms leave the AVX2 integer path"]
fn per_group_activation_scales_are_a_substitute_for_rotation() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let corpus = PathBuf::from(
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a corpus json"),
    );
    let tokens = eval_tokens(&corpus);
    assert_eq!(tokens.len(), EVAL_TOKENS, "corpus is too small");

    let root = std::env::temp_dir().join(format!("tritium-a8g-{}", std::process::id()));
    let fp = score(
        ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load fp master"),
        &tokens,
    );
    println!("\nSmolLM2-135M | --planes 4 --group 256 --fold-alpha 0.75 | fp {fp:.4}");
    println!(
        "{:<22} {:>11} {:>11} {:>11} {:>11}",
        "activation scales", "rotated", "vs A8", "unrotated", "vs A8"
    );
    println!("{}", "-".repeat(70));

    // [rotated, unrotated] x [per-token, g256, g128, g64].
    let mut table = [[0.0f64; 4]; 2];
    for (arm, rotate) in [(0usize, true), (1usize, false)] {
        let dir = root.join(if rotate { "rot" } else { "plain" });
        convert_with(&model, &dir, &corpus, 0.75, rotate);
        for (slot, group) in [None, Some(256), Some(128), Some(64)]
            .into_iter()
            .enumerate()
        {
            let mut runner = ModelRunner::from_salt(
                &dir,
                &dir.join("model.tslb"),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .expect("load converted model");
            let touched = runner.weights.set_salt_activation_group(group);
            assert!(
                touched > 0,
                "no SALT projection was configured — the model is not on the path under test, so \
                 a flat result here would mean nothing"
            );
            table[arm][slot] = score(runner, &tokens);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    let _ = std::fs::remove_dir_all(&root);

    let pct = |v: f64, base: f64| {
        if (v - base).abs() < 1e-12 {
            "—".to_owned()
        } else {
            format!("{:+.2}%", 100.0 * (v - base) / base)
        }
    };
    for (slot, label) in [
        "per token (SHIPPING)",
        "per group g256",
        "per group g128",
        "per group g64",
    ]
    .into_iter()
    .enumerate()
    {
        println!(
            "{label:<22} {:>11.4} {:>11} {:>11.4} {:>11}",
            table[0][slot],
            pct(table[0][slot], table[0][0]),
            table[1][slot],
            pct(table[1][slot], table[1][0]),
        );
    }

    let best = |row: [f64; 4]| row[1].min(row[2]).min(row[3]);
    let gain_rot = 100.0 * (table[0][0] - best(table[0])) / table[0][0];
    let gain_plain = 100.0 * (table[1][0] - best(table[1])) / table[1][0];
    println!(
        "\nbest per-group gain over per-token:  rotated {gain_rot:+.2}%  |  unrotated {gain_plain:+.2}%"
    );

    assert!(
        table.iter().flatten().all(|p| p.is_finite() && *p > 0.0),
        "an arm did not produce a usable model"
    );
    // The claim. Rotation whitens the activation, so it should leave per-group scales far less to
    // recover than they find in the original basis. A failure here does NOT mean the plumbing is
    // broken — it means the substitution account is wrong and the tape's +1.19% -> +0.43% needs
    // another explanation.
    assert!(
        gain_plain > gain_rot,
        "per-group scales bought {gain_plain:+.2}% unrotated and {gain_rot:+.2}% rotated. The \
         substitution account predicts the first is clearly larger; it is not, so the account is \
         wrong and the tape's recovery figure has some other cause"
    );
}

/// **The unpadded container changes the file and nothing else.**
///
/// `convert` writes `TSLJ` by default: joint-symbol Huffman, no block padding, decoded back to
/// TQ2_0 rows at load. That is only safe if the loaded model is *exactly* the one the padded file
/// would have produced — not close, identical — because every published number was measured
/// against TQ2_0 rows.
///
/// So the same model is converted twice, and three things are asserted:
///
/// 1. every tensor decodes to **byte-identical** rows from both files;
/// 2. both load through `ModelRunner` and score **bit-identical** perplexity;
/// 3. the `TSLJ` file is materially smaller.
///
/// Row identity alone would not be enough: it proves the format, not the loader's `TSLJ` branch.
/// Perplexity through the runtime proves the whole path.
#[test]
#[ignore = "two conversions and two evaluations of a real model on CPU"]
fn unpadded_container_changes_the_file_and_nothing_else() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let corpus = PathBuf::from(
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a corpus json"),
    );
    // A short slice suffices: the claim is equality, and equality on any window is exact.
    let tokens: Vec<u32> = eval_tokens(&corpus).into_iter().take(1024).collect();

    let root = std::env::temp_dir().join(format!("tritium-tslj-{}", std::process::id()));
    let joint = root.join("joint");
    let dense = root.join("dense");
    convert_full(&model, &joint, &corpus, 0.75, true, 3, false);
    convert_full(&model, &dense, &corpus, 0.75, true, 3, true);

    let jb = std::fs::read(joint.join("model.tslb")).expect("read TSLJ");
    let db = std::fs::read(dense.join("model.tslb")).expect("read TSLB");
    assert_eq!(
        &jb[..4],
        b"TSLJ",
        "default convert must write the unpadded container"
    );
    assert_eq!(&db[..4], b"TSLB", "--dense-container must write TQ2_0");

    use tritium_format::salt_joint_bundle::read_any_salt_bundle;
    let jt = read_any_salt_bundle(&jb).expect("decode TSLJ");
    let dt = read_any_salt_bundle(&db).expect("decode TSLB");
    assert_eq!(jt.len(), dt.len(), "tensor count");
    let params: usize = dt.iter().map(|t| t.rows * t.k).sum();
    for (a, b) in jt.iter().zip(&dt) {
        assert_eq!(
            a, b,
            "tensor `{}` decodes differently from the two containers",
            b.name
        );
    }

    let ppl_joint = score_converted(&joint, &tokens);
    let ppl_dense = score_converted(&dense, &tokens);
    let _ = std::fs::remove_dir_all(&root);

    let bpw = |n: usize| n as f64 * 8.0 / params as f64;
    println!(
        "T=3 g256 rotated | {} tensors, {params} weights, all rows byte-identical\n  \
         TSLB (padded TQ2_0) {:>12} bytes  {:.4} bpw  ppl {ppl_dense:.6}\n  \
         TSLJ (joint, unpadded) {:>9} bytes  {:.4} bpw  ppl {ppl_joint:.6}\n  \
         file {:+.2}%",
        dt.len(),
        db.len(),
        bpw(db.len()),
        jb.len(),
        bpw(jb.len()),
        100.0 * (jb.len() as f64 - db.len() as f64) / db.len() as f64
    );

    assert_eq!(
        ppl_joint.to_bits(),
        ppl_dense.to_bits(),
        "the two containers scored {ppl_joint} and {ppl_dense}. Rows were identical, so the \
         loader's TSLJ branch is building the matrix differently from the TSLB one"
    );
    assert!(
        (jb.len() as f64) < 0.7 * db.len() as f64,
        "TSLJ is {} bytes against TSLB's {}; measured at 4.58 vs 7.97 bpw, anything above 70% means \
         the coder is not doing its job",
        jb.len(),
        db.len()
    );
}
