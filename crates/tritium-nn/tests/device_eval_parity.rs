//! **Does GPU evaluation agree with the host, and by how much?**
//!
//! Perplexity sweeps are the campaign's bottleneck: ~10 min per evaluation at SmolLM2-360M and
//! ~49 min at 1.7B, all on CPU, while the GPU idles.
//! [`perplexity_windowed_device`](common::perplexity_windowed_device) moves the forward pass onto
//! the device tape — but that is a **change of measurement basis**, and this repo has twice come
//! close to publishing a wrong number by changing one silently (a 2,048-token curve subsample that
//! reads ~7% high against the full held-out set; a baseline harness that priced SALT at TQ2_0's
//! 2.0625 bpw/plane instead of B3's 1.625).
//!
//! So the rule is: **the host path stays authoritative for any number that gets quoted**, the device
//! path is for sweeps, and this test measures the gap instead of assuming it.
//!
//! The device tape reproduces the host op sequence within ~1e-4 — transcendentals (`exp`, `silu`,
//! `softmax`) are not bit-identical across backends — so perplexity cannot be expected to match
//! exactly. What matters is that the disagreement is far smaller than any effect we draw
//! conclusions from. For calibration: the ladder's win over folded RTN int5 is **1.0%**, and the
//! fold is worth **5.7%** at int4. A basis error above ~0.1% could reorder those.
//!
//! Both perplexities run through the same `score_window` helper, so a delta here can only come from
//! the forward pass, never from the scoring arithmetic.
//!
//! `#[ignore]`d (needs a GPU and the cached model); run:
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --features cuda --test device_eval_parity \
//!   -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{extract, perplexity_windowed, perplexity_windowed_device};
use tritium_cuda::CudaBackend;
use tritium_nn::ModelRunner;

const EVAL_WINDOW: usize = 512;
/// Relative perplexity agreement required between the host and device paths.
///
/// An order of magnitude tighter than the smallest effect the campaign reports (the 1.0% ladder-vs-
/// int5 margin), so a basis change can never masquerade as a result.
const MAX_REL_DELTA: f64 = 1e-3;
/// Held-out tokens to score. Enough windows to average out per-window noise without spending an
/// hour on a gate; the full corpus is what the research runs use.
const GATE_TOKENS: usize = 4096;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m"));
    PathBuf::from(dir)
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
#[ignore = "needs a CUDA device + the cached model; run explicitly"]
fn device_eval_matches_host_ppl() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() && !dir.join("model.safetensors.index.json").exists()
    {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, _shapes) = extract(&runner);

    let all = eval_ids();
    let eval = &all[..GATE_TOKENS.min(all.len())];

    let t0 = Instant::now();
    let host = perplexity_windowed(&fp, &arch, eval, EVAL_WINDOW);
    let host_secs = t0.elapsed().as_secs_f64();

    let backend = CudaBackend::new(0).expect("open CUDA device");
    let t1 = Instant::now();
    let device = perplexity_windowed_device(&backend, &fp, &arch, eval, EVAL_WINDOW);
    let device_secs = t1.elapsed().as_secs_f64();

    let rel = (device - host).abs() / host;
    println!(
        "{} | {} tokens | window {EVAL_WINDOW}\n\
         host   ppl {host:.6}   ({host_secs:.1}s)\n\
         device ppl {device:.6}   ({device_secs:.1}s)\n\
         relative delta {rel:.3e}  (bound {MAX_REL_DELTA:.0e})\n\
         speedup {:.2}x",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        eval.len(),
        host_secs / device_secs.max(1e-9),
    );
    println!(
        "\nFor scale: the ladder's margin over folded RTN int5 is 1.0% and the salience fold is \
         worth 5.7% at int4. A delta approaching those is not a rounding detail — it would mean \
         device-measured sweeps cannot be compared against host-measured baselines at all."
    );

    assert!(
        rel <= MAX_REL_DELTA,
        "host/device perplexity disagree by {rel:.3e} (> {MAX_REL_DELTA:.0e}): \
         host {host:.6}, device {device:.6}. GPU eval must not be used for sweeps until this is \
         understood — a basis error of this size can reorder the campaign's conclusions."
    );
}
