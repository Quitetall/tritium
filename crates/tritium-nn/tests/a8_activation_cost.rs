//! **What do int8 activations cost, and does it depend on the plane count?**
//!
//! Every SALT perplexity this repo has published is *weight*-quantized only: the harnesses score
//! `ste::salt_quantize_forward_grouped*` dense reconstructions through `Tape::dense_matmul`, which
//! consumes fp32 activations. The shipping runtime does not — `SaltLinear::forward`
//! (`crates/tritium-nn/src/layers/salt.rs:114`) calls `quantize_activation_int8` before every
//! projection, and `Projection::Salt` declares `ProjectionActivationMode::A8` where
//! `Projection::Dense` declares `F32`.
//!
//! `convert_anchor_diagnostic` measured that difference at **1.0276×** for T=4 on SmolLM2-360M,
//! having first ruled out the evaluator (fp scores identically through both paths) and the packer
//! (bytes decode to the fit at the f16 floor). So the fp-parity claim is a statement about weight
//! quantization, and a user running the artifact pays more.
//!
//! One number is not a result. This measures the A8 cost **as a function of plane count**, which is
//! the question that decides how it should be reported:
//!
//! - If the cost is roughly constant in `T`, it is an additive activation tax and can be quoted
//!   once alongside any weight-quantization figure.
//! - If it *grows* with `T`, then the high-`T` fp-parity numbers are the most overstated ones, and
//!   the parity claim degrades exactly where it is currently strongest.
//! - If it *shrinks* with `T`, more planes buy tolerance to int8 activations, which is itself an
//!   argument for the ladder.
//!
//! Both arms per `T` use the same weights, slice and window; only the activation precision differs.
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test a8_activation_cost -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::{extract, perplexity_windowed};
use tritium_nn::{ModelRunner, teacher_forced_perplexity_windows};
use tritium_train::ops::ste::{self, RotationPolicy};

const WINDOW: usize = 512;
const TOKENS: usize = 8192;
const GROUP: usize = 256;
const GRID: usize = 16;
/// `tritium convert` refuses fewer than 3 planes without rotation (323× fp at T=2 measured), so the
/// shipping range is the range worth measuring.
const PLANE_COUNTS: [usize; 2] = [3, 4];

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
}

fn tritium_bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("tritium")
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

#[test]
#[ignore = "several full evaluations of a real model on CPU"]
fn int8_activation_cost_across_plane_counts() {
    let model = env_path("TRITIUM_MODEL_DIR");
    let tokens = eval_tokens();
    assert_eq!(tokens.len(), TOKENS);

    let runner = ModelRunner::from_hf(&model, Box::new(tritium_cpu::CpuBackend::new()))
        .expect("load fp master");
    let (arch, fp, shapes) = extract(&runner);
    drop(runner);

    // fp reference under BOTH evaluators. They are known to agree exactly; re-measuring keeps every
    // ratio below anchored to a number produced in this same run.
    let fp_tape = perplexity_windowed(&fp, &arch, &tokens, WINDOW);
    println!("fp / tape (A-fp32)   : {fp_tape:.3}");

    let root = std::env::temp_dir().join(format!("tritium-a8-{}", std::process::id()));
    let mut rows: Vec<(usize, f64, f64)> = Vec::new();

    for planes in PLANE_COUNTS {
        // A-fp32: the fitter's dense reconstruction on the tape — the basis of every published
        // SALT number.
        let fitted: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped_geometric(
                    w,
                    n,
                    k,
                    planes,
                    GROUP,
                    GRID,
                    RotationPolicy::Never,
                )
            })
            .collect();
        let a_fp32 = perplexity_windowed(&fitted, &arch, &tokens, WINDOW);
        drop(fitted);

        // A-int8: the same fit, written by `tritium convert` and executed by the shipping runtime.
        let out = root.join(format!("t{planes}"));
        let _ = std::fs::remove_dir_all(&out);
        let status = Command::new(tritium_bin())
            .args([
                "convert",
                "--model",
                model.to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
                "--planes",
                &planes.to_string(),
                "--group",
                &GROUP.to_string(),
                "--grid",
                &GRID.to_string(),
                "--fold-alpha",
                "0",
            ])
            .status()
            .expect("run tritium convert");
        assert!(status.success(), "convert failed at T={planes}");

        let mut r = ModelRunner::from_salt(
            &out,
            &out.join("model.tslb"),
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .expect("load converted artifact");
        let a_int8 = teacher_forced_perplexity_windows(&mut r, &tokens, WINDOW)
            .expect("score artifact")
            .perplexity;
        drop(r);
        let _ = std::fs::remove_dir_all(&out);

        println!(
            "T={planes}: A-fp32 {a_fp32:.3} ({:.4}x fp) | A-int8 {a_int8:.3} ({:.4}x fp) | A8 tax \
             {:+.2}%",
            a_fp32 / fp_tape,
            a_int8 / fp_tape,
            100.0 * (a_int8 - a_fp32) / a_fp32
        );
        rows.push((planes, a_fp32, a_int8));
    }

    println!("\n--- int8 activation tax by plane count ---");
    for (planes, a_fp32, a_int8) in &rows {
        println!(
            "T={planes}  weights-only {:.4}x fp  ->  deployed {:.4}x fp   (tax {:+.2}%)",
            a_fp32 / fp_tape,
            a_int8 / fp_tape,
            100.0 * (a_int8 - a_fp32) / a_fp32
        );
    }
    println!(
        "\nEvery published SALT perplexity is the 'weights-only' column. The 'deployed' column is \
         what `tritium convert` + `ModelRunner` produce."
    );
    let _ = std::fs::remove_dir_all(&root);
}
