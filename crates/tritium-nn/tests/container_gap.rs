//! **How much worse is the artifact a user downloads than the numbers we publish?**
//!
//! Three configurations exist in this repo and only one of them is published:
//!
//! | path | fold | rotation |
//! |---|---|---|
//! | every published number, including the whole 2026-09 campaign | yes | yes |
//! | `tritium convert` | yes | **no** |
//! | `tritium quantize` | **no** | **no** |
//!
//! `cli/convert.rs` says it outright: *"the published numbers are fold **and** rotation, and
//! `quantize`'s numbers are neither."* The AWQ fold escapes the container because it is absorbed
//! into the stored weights **and** the stored norms — the product is identity, so a folded model is
//! self-describing. Rotation is not absorbed: the fit reconstructs `W·H`, so the runtime must rotate
//! activations to match, and `nn/src/layers/salt.rs` has no rotation path at all. Hence
//! `cli/quantize_ladder.rs` hardcodes [`RotationPolicy::Never`].
//!
//! Rotation is not incidental. Without it the ladder **loses to the old ITF fitter by 14.7× at
//! T=2**, because the fixed 1/3 spacing assumes a well-conditioned distribution and the Hadamard is
//! what supplies it.
//!
//! **Nothing in the repo states the size of this gap.** Every table is quoted in a configuration no
//! shipped artifact can express. This measures it, so the footnote can be written whether or not
//! the container is ever fixed.
//!
//! # Design
//!
//! `α = 0` reproduces no-fold exactly (`s_j = (rms_j/gm)^0 = 1`), so both fold settings run the
//! same code path and the comparison cannot drift between them. Rotation is the
//! [`RotationPolicy`] the fitter is given. The grid is therefore 2 × 2 × |T|, with one fp baseline.
//!
//! A control runs first: the fold is *function-preserving* by construction, so fp perplexity must
//! be unchanged by it. If that fails, every folded number in this repo is suspect and nothing below
//! is worth reading.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test container_gap -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;

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

#[test]
#[ignore = "needs SmolLM2-135M; run explicitly"]
fn fold_and_rotation_gap_between_research_and_artifact() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
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

    // CONTROL. The fold scales weight columns by `s` and divides the preceding norm's rows by the
    // same `s`; the product is identity, so it cannot change what the fp model computes. If this
    // drifts by more than float reassociation, every folded number in this repo is suspect.
    let ppl_fp_unfolded = perplexity_windowed(&fp0, &arch0, &eval, EVAL_WINDOW);
    let (folded_fp, folded_arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);
    let ppl_fp_folded = perplexity_windowed(&folded_fp, &folded_arch, &eval, EVAL_WINDOW);
    let drift = (ppl_fp_folded - ppl_fp_unfolded).abs() / ppl_fp_unfolded;
    println!(
        "control — the fold is function-preserving:\n  \
         fp unfolded {ppl_fp_unfolded:.4} | fp folded {ppl_fp_folded:.4} | drift {:.3e}\n",
        drift
    );
    assert!(
        drift < 1e-3,
        "the fold changed the fp model by {drift:.3e} — it is supposed to be exactly \
         function-preserving, so either scale_cols/divide_rows disagree or the norms were not \
         folded with the weights"
    );

    println!(
        "SmolLM2-135M | WikiText-2 {} held-out | g{GROUP} | ladder\n\
         `tritium quantize` ships no-fold/no-rotation. `tritium convert` ships fold-only.\n\
         Every published number is fold+rotation. This is the size of that gap.\n",
        eval.len()
    );
    println!(
        "{:<10} {:>8} {:>10} {:>11} {:>9} {:>16}",
        "planes", "fold α", "rotation", "ppl", "× fp", "vs published"
    );
    println!("{}", "-".repeat(72));

    for t in [3usize, 4] {
        let mut published = f64::NAN;
        // fold+rotation first so every later row can be quoted against it.
        for (alpha, rot, label) in [
            (0.75, RotationPolicy::Always, "always"),
            (0.75, RotationPolicy::Never, "never"),
            (0.0, RotationPolicy::Always, "always"),
            (0.0, RotationPolicy::Never, "never"),
        ] {
            // `α = 0` yields `s_j = 1` exactly, so this is the no-fold path through identical code.
            let (weights, arch) = fold(&fp0, &shapes, &arch0, &calib, alpha);
            let q: Vec<Vec<f32>> = weights
                .iter()
                .zip(&shapes)
                .map(|(w, &(r, c))| {
                    ste::salt_quantize_forward_grouped_geometric(w, r, c, t, GROUP, GRID, rot)
                })
                .collect();
            let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
            if published.is_nan() {
                published = ppl;
            }
            let note = if (ppl - published).abs() < 1e-9 {
                "— (published)".to_owned()
            } else {
                format!("{:+.2}%", 100.0 * (ppl - published) / published)
            };
            let ships = match (alpha, rot) {
                (0.75, RotationPolicy::Never) => "  <- convert",
                (0.0, RotationPolicy::Never) => "  <- quantize",
                _ => "",
            };
            println!(
                "T={t:<8} {alpha:>8} {label:>10} {ppl:>11.3} {:>8.3}× {note:>16}{ships}",
                ppl / ppl_fp_unfolded
            );
        }
        println!();
    }

    println!(
        "The `convert` and `quantize` rows are what a user actually runs. The gap between them and\n\
         the published row is the footnote every table in this repo is currently missing — and the\n\
         rotation half of it is a CONTAINER limitation, not an algorithmic one: `fast_hadamard` is\n\
         parameterless, so the bundle needs one bit per group, not a matrix."
    );
}
