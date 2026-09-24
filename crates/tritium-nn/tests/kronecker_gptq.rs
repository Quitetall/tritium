//! **Does output-side curvature help GPTQ? (H6, first-principles plan)**
//!
//! H1 showed the cost of a ternary error is a stable quadratic form and that the cheap proxies rank
//! it (Gram/Fisher ~0.8), while the MEASURED per-unit cost is swamped by a first-order sample term.
//! So the one objective worth optimizing harder is the curvature itself. Shipping GPTQ uses only its
//! input half: it minimises `tr(E·A·Eᵀ)` with `A = E[x·xᵀ]`, treating every output row as equally
//! and independently important. The Kronecker (K-FAC) curvature is `A ⊗ G` with `G = E[δ·δᵀ]` over
//! the output gradient, and a non-diagonal `G` couples the rows' rounding decisions.
//! [`tritium_nn::salt_fit::fit_tensor_kronecker`] rounds on that lattice (nearest-plane over
//! `chol(A⁻¹) ⊗ chol(G⁻¹)`); with `G = I` it is shipping GPTQ bit for bit.
//!
//! `G` is the TRUE Fisher: labels are sampled from the fp model's own softmax, not taken from the
//! text, so it is a curvature and not a gradient-noise estimate.
//!
//! Scoring uses the antithetic split H1 suggested. Besides `Q = W + E`, each arm is also scored at
//! `W − E`: `even = (ΔL(+E) + ΔL(−E))/2` is the curvature cost with the first-order sample term
//! cancelled, `odd` is that sample term. Differences in `even` are what a method changes; `odd` is
//! what the evaluation set happens to reward.
//!
//! # Measured 2026-09-24 — SmolLM2-135M, A and G from 16,384 tokens, full WikiText-2 eval
//!
//! ```text
//!                                   T=2 ppl     even      odd      T=3 ppl     even      odd
//! ladder (RTN)                       46.405  +0.71623  -0.00011     24.278  +0.07746  -0.00915
//! GPTQ input-only, auto decay        40.263  +0.61521  -0.04106     24.177  +0.06620  -0.00206
//! Kronecker GPTQ, λ_G=1              39.124  +0.60297  -0.05751     24.064  +0.06323  -0.00379
//! Kronecker GPTQ, λ_G=0.5            38.918  +0.58743  -0.04726     24.027  +0.06220  -0.00430
//! ```
//!
//! **The output side pays: H6 promoted.** Against shipping GPTQ at matched calibration, λ_G=0.5 is
//! -3.3% perplexity at T=2 (plan bar: >1%) and cuts the curvature (even) cost 4.5% at T=2 and 6.0%
//! at T=3; at T=3 the perplexity change (-0.6%) is inside the configuration noise floor, the even
//! part is not. Partial trust in G (λ_G=0.5) beats full trust at both T — the same shape as the
//! input-side decay, for the same reason (a 16k-token estimate of a 1,536-wide covariance).
//!
//! The odd column is new information: RTN's error is uncorrelated with the held-out gradient
//! (odd ≈ 0), but every CALIBRATED fit carries a real negative first-order term on held-out text
//! (-0.04 at T=2) — the calibration aligns `E` with the domain's gradient, and it generalizes.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test kronecker_gptq -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H6_TS` (default `2,3`), `TRITIUM_H6_CALIB` (512-token windows for A and G,
//! default 32), `TRITIUM_H6_ROW_DECAYS` (default `1,0.5`), `TRITIUM_H6_EVAL_WINDOWS` (default 64).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{
    Calib, calibrate, extract, fold, forward_with_outputs, output_fishers, perplexity_windowed,
};
use tritium_nn::ModelRunner;
use tritium_nn::salt_fit::{
    ActivationAwareConfig, OutputMetric, TapGrams, auto_decay, fit_tensor, fit_tensor_kronecker,
    fit_to_dense,
};
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const WINDOW: usize = 512;
const FOLD_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS (WikiText-2 basis)");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus")).expect("json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

#[derive(Clone, Copy)]
enum Arm {
    Plain,
    Gptq,
    Kronecker(f64),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Plain => "ladder (RTN)".into(),
            Arm::Gptq => "GPTQ input-only, auto decay".into(),
            Arm::Kronecker(d) => format!("Kronecker GPTQ, λ_G={d}"),
        }
    }
}

#[test]
#[ignore = "H6 Kronecker GPTQ; needs SmolLM2-135M; run explicitly"]
fn output_side_curvature_against_input_only_gptq() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts: Vec<usize> = env_str("TRITIUM_H6_TS", "2,3")
        .split(',')
        .map(|s| s.trim().parse().expect("T"))
        .collect();
    let calib_windows: usize = env_str("TRITIUM_H6_CALIB", "32").parse().expect("calib");
    let row_decays: Vec<f64> = env_str("TRITIUM_H6_ROW_DECAYS", "1,0.5")
        .split(',')
        .map(|s| s.trim().parse().expect("decay"))
        .collect();
    let n_eval: usize = env_str("TRITIUM_H6_EVAL_WINDOWS", "64")
        .parse()
        .expect("eval");

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let eval = &eval[..(n_eval * WINDOW).min(eval.len())];
    let mut calib = Calib::new(&arch);
    for w in 0..FOLD_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    // The recorded forward must be the library forward, or G is some other model's curvature.
    {
        let tokens = &train[..64];
        let mut t1 = Tape::new();
        let w1: Vec<ValueId> = folded.iter().map(|x| t1.leaf(x.clone())).collect();
        let lib = common::forward(&mut t1, &w1, &arch, tokens);
        let mut t2 = Tape::new();
        let w2: Vec<ValueId> = folded.iter().map(|x| t2.leaf(x.clone())).collect();
        let (ours, _) = forward_with_outputs(&mut t2, &w2, &arch, tokens);
        assert_eq!(
            t1.value(lib),
            t2.value(ours),
            "recorded forward drifted from the library"
        );
    }

    let started = Instant::now();
    let windows: Vec<&[u32]> = (0..calib_windows)
        .map(|w| &train[w * WINDOW..(w + 1) * WINDOW])
        .collect();
    let grams = TapGrams::collect(&folded, &arch, &windows);
    let fishers = output_fishers(&folded, &arch, &windows, 0xF15E_A5ED);
    let tokens = calib_windows * WINDOW;
    let nll = |w: &[Vec<f32>]| perplexity_windowed(w, &arch, eval, WINDOW).ln();
    let base = nll(&folded);
    println!(
        "SmolLM2-135M | fold α=0.75 | ladder g{GROUP} rotated | A and G from {tokens} tokens (G: sampled-label Fisher) \
         | {} eval tokens | fp NLL {base:.5} | curvature {:.0}s\n",
        eval.len(),
        started.elapsed().as_secs_f64()
    );

    let mut arms = vec![Arm::Plain, Arm::Gptq];
    arms.extend(row_decays.iter().map(|&d| Arm::Kronecker(d)));
    for &t in &ts {
        println!("T={t}");
        println!(
            "{:<32} {:>9} {:>10} {:>10} {:>10} {:>10}",
            "arm", "ppl", "ΔL(+E)", "ΔL(−E)", "even", "odd"
        );
        for &arm in &arms {
            let started = Instant::now();
            let q: Vec<Vec<f32>> = folded
                .iter()
                .zip(&shapes)
                .enumerate()
                .map(|(i, (w, &(rows, cols)))| {
                    let ladder = || {
                        ste::salt_quantize_forward_grouped_geometric(
                            w,
                            rows,
                            cols,
                            t,
                            GROUP,
                            GRID,
                            RotationPolicy::Always,
                        )
                    };
                    // The tied embed/head has no single input Gram: every arm keeps it on the ladder.
                    if i == 0 || matches!(arm, Arm::Plain) {
                        return ladder();
                    }
                    let (li, slot) = ((i - 1) / 7, (i - 1) % 7);
                    let cfg = ActivationAwareConfig {
                        planes: t,
                        group: GROUP,
                        grid: GRID,
                        damp: 0.01,
                        search_sweeps: 0,
                        refit_scale: false,
                        rotate: true,
                        decay: auto_decay(tokens, cols),
                        decay_ramp: false,
                    };
                    let a = grams.for_slot(li, slot);
                    let fits = match arm {
                        Arm::Gptq => fit_tensor(w, rows, cols, a, &cfg),
                        Arm::Kronecker(d) => fit_tensor_kronecker(
                            w,
                            rows,
                            cols,
                            a,
                            &OutputMetric {
                                gram: &fishers[li * 7 + slot],
                                damp: 0.01,
                                decay: d,
                            },
                            &cfg,
                        ),
                        Arm::Plain => unreachable!(),
                    };
                    fits.map_or_else(ladder, |f| fit_to_dense(&f, rows, cols, GROUP, true))
                })
                .collect();
            let anti: Vec<Vec<f32>> = folded
                .iter()
                .zip(&q)
                .map(|(w, qv)| w.iter().zip(qv).map(|(a, b)| 2.0 * a - b).collect())
                .collect();
            let (plus, minus) = (nll(&q) - base, nll(&anti) - base);
            println!(
                "{:<32} {:>9.3} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5}   ({:.0}s)",
                arm.label(),
                (base + plus).exp(),
                plus,
                minus,
                0.5 * (plus + minus),
                0.5 * (plus - minus),
                started.elapsed().as_secs_f64()
            );
        }
        println!();
    }
}
