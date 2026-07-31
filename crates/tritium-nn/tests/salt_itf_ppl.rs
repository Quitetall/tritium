//! Does ITF's better *reconstruction* actually buy better *perplexity*?
//!
//! [`ste::salt_quantize_forward_itf`] cuts weight-space reconstruction error by ~9× at `T=3` versus the
//! greedy AbsMean fit, at identical bits. That is a weight-space claim. The SOTA survey
//! (`docs/research-ternary-sota-mid2026.md`) documents a **proxy gap** — per-layer MSE can
//! *anti-correlate* with end-task quality (BCJR-QAT's negative result) — so the reconstruction win
//! does not automatically transfer.
//!
//! This is the cheap, decisive check: pure PTQ, no training. Quantize the same fp model both ways and
//! score held-out perplexity on the disjoint WikiText-2 split. If ITF does not help here, it is not
//! worth wiring into the CUDA training path.
//!
//! `#[ignore]`d (needs SmolLM2-135M + a corpus); run:
//!
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_itf_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste;

const EVAL_WINDOW: usize = 512;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn eval_ids() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

#[test]
#[ignore = "slow PTQ perplexity comparison; needs SmolLM2-135M; run explicitly"]
fn itf_ptq_beats_greedy_ptq_on_held_out_perplexity() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let iters: usize = std::env::var("TRITIUM_ITF_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let eval = eval_ids();

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    println!(
        "fp reference: {ppl_fp:.3} ppl over {} held-out tokens",
        eval.len()
    );

    // Same weights, same bits, same plane count — only the per-plane fitter differs.
    for t in 1..=3 {
        let greedy: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| ste::salt_quantize_forward(w, n, k, t))
            .collect();
        let itf: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| ste::salt_quantize_forward_itf(w, n, k, t, iters))
            .collect();

        // Weight-space error, the thing ITF provably improves.
        let sse = |q: &[Vec<f32>]| -> f64 {
            q.iter()
                .zip(&fp)
                .flat_map(|(a, b)| a.iter().zip(b))
                .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
                .sum()
        };
        let (sse_g, sse_i) = (sse(&greedy), sse(&itf));

        // End-task quality, the thing that actually matters.
        let ppl_g = perplexity_windowed(&greedy, &arch, &eval, EVAL_WINDOW);
        let ppl_i = perplexity_windowed(&itf, &arch, &eval, EVAL_WINDOW);

        println!(
            "T={t}: SSE {sse_g:.4e} → {sse_i:.4e} ({:.1}% better) | \
             PTQ ppl {ppl_g:.3} → {ppl_i:.3} ({:.1}% {}) | gap to fp {:.2}× → {:.2}×",
            (sse_g - sse_i) / sse_g * 100.0,
            (ppl_g - ppl_i) / ppl_g * 100.0,
            if ppl_i < ppl_g { "better" } else { "WORSE" },
            ppl_g / ppl_fp,
            ppl_i / ppl_fp,
        );

        assert!(
            sse_i <= sse_g * (1.0 + 1e-6),
            "ITF must not worsen reconstruction (T={t})"
        );
        assert!(ppl_i.is_finite(), "ITF perplexity must be finite (T={t})");
    }
    println!(
        "\nProxy-gap read: if ppl tracks SSE downward, ITF is worth wiring into the CUDA \
         training path; if ppl moves the other way, the reconstruction win does not transfer."
    );
}
