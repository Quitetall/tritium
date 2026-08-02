//! **The calibration tap.** Nothing in this repo has ever produced input activations for the
//! curvature-driven solvers, which is why `salt_v2_feedback.rs` (GPTQ/BlockLDLQ),
//! `fit_joint_ternary` (exact 3^P joint plane assignment) and `build_kfac_metric` are all written,
//! correct, and never executed. `InputGramAccumulator` and `OutputFisherAccumulator` have no
//! non-test callers anywhere.
//!
//! This is the missing producer, at the smallest honest scope: the full input Gram `H = E[x xᵀ]` at
//! the four distinct projection-input points of a real SmolLM2 block, taken from the same validated
//! tape forward the perplexity harness uses.
//!
//! Why the full matrix rather than the diagonal we already collect: the salience fold only needs
//! `E[x_j²]` per channel, but sequential error compensation needs to know how columns COVARY — when
//! column `j` is rounded, GPTQ pushes its error onto the not-yet-quantized columns through `H⁻¹`.
//! The off-diagonal is the entire mechanism.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_gram -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, Gram, calibrate, calibrate_gram, extract};
use tritium_nn::ModelRunner;

const CALIB_SEQ: usize = 512;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn train_ids() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["train_ids"]
        .as_array()
        .expect("train_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

/// The Gram must be a real covariance of the same activations the cheap collector sees:
/// symmetric, positive semi-definite, and diagonal-consistent with `Calib`.
///
/// The diagonal cross-check is the load-bearing one — it proves the tap is reading the tensor it
/// claims to. `Calib` accumulates `Σx²` at the same three points via a completely separate code
/// path, so agreement is independent corroboration rather than a tautology.
#[test]
#[ignore = "needs SmolLM2-135M; run explicitly"]
fn input_gram_is_psd_and_agrees_with_the_diagonal_collector() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, _shapes) = extract(&runner);
    let toks = &train_ids()[..CALIB_SEQ];

    // The cheap diagonal collector, for cross-checking.
    let mut calib = Calib::new(&arch);
    calibrate(&fp, &arch, toks, &mut calib);

    for (layer, role, k, want_diag) in [
        (0usize, "attn", arch.n_embd, &calib.attn_in[0]),
        (0, "ffn", arch.n_embd, &calib.ffn_in[0]),
        (0, "down", arch.ff, &calib.down_in[0]),
        (7, "attn", arch.n_embd, &calib.attn_in[7]),
    ] {
        let mut g = Gram::new(k);
        calibrate_gram(&fp, &arch, toks, layer, role, &mut g);
        assert_eq!(g.rows, CALIB_SEQ, "every token must be counted");

        let diag_raw = g.diagonal();
        let h = g.finish();

        // 1. Symmetric.
        for &(i, j) in &[(0usize, 1usize), (3, 17), (k / 2, k - 1)] {
            let (a, b) = (h[i * k + j], h[j * k + i]);
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                "H must be symmetric at ({i},{j}) for L{layer}/{role}: {a} vs {b}"
            );
        }

        // 2. PSD along a few random directions: vᵀHv >= 0.
        let mut s = 0x9E37u64 ^ (layer as u64);
        for _ in 0..8 {
            let v: Vec<f64> = (0..k)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 2000) as f64 / 1000.0 - 1.0
                })
                .collect();
            let mut q = 0.0f64;
            for i in 0..k {
                let row = &h[i * k..(i + 1) * k];
                let vi = v[i];
                for (j, &hij) in row.iter().enumerate() {
                    q += vi * hij * v[j];
                }
            }
            assert!(
                q >= -1e-6,
                "H must be PSD for L{layer}/{role}, got vᵀHv = {q}"
            );
        }

        // 3. Diagonal agrees with the independent Σx² collector.
        for (i, (&got, &want)) in diag_raw.iter().zip(want_diag).enumerate().take(k) {
            let tol = 1e-6 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "L{layer}/{role} channel {i}: Gram diagonal {got} != Calib {want}"
            );
        }

        // 4. There is real off-diagonal mass — otherwise GPTQ has nothing to propagate and the
        //    diagonal fold we already have would be the whole story.
        let total: f64 = h.iter().map(|v| v * v).sum();
        let diag: f64 = (0..k).map(|i| h[i * k + i] * h[i * k + i]).sum();
        let off = 1.0 - diag / total;
        println!(
            "L{layer}/{role:<5} k={k:<5} off-diagonal energy {:.1}%  (trace {:.4e})",
            off * 100.0,
            (0..k).map(|i| h[i * k + i]).sum::<f64>()
        );
        assert!(
            off > 0.05,
            "L{layer}/{role}: only {:.2}% off-diagonal energy — sequential compensation would buy \
             nothing over the diagonal fold",
            off * 100.0
        );
    }
}
