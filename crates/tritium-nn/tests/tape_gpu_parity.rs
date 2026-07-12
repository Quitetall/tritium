//! GPU training path — Phase 1 parity gate (plan 0043). Runs the same whole-model tape forward
//! (embed → 30 GQA blocks → SwiGLU → tied head) two ways on SmolLM2-135M: the CPU tape, and a tape
//! whose matmuls are dispatched to the GPU (`Tape::with_gemm(GpuGemm)` → the bit-exact `train_grad.cu`
//! kernels). Asserts the GPU-backed logits match the CPU tape — validating the pluggable `TrainGemm`
//! seam end-to-end on a real model — and prints CPU-vs-GPU forward time. `#[ignore]`d (needs
//! SmolLM2-135M + a CUDA device); run:
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test tape_gpu_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use common::{extract, forward, logits_of};
use tritium_cuda::train::GpuGemm;
use tritium_nn::ModelRunner;
use tritium_train::{Tape, TrainGemm};

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

#[test]
#[ignore = "needs SmolLM2-135M + a CUDA device; run explicitly"]
fn gpu_tape_matches_cpu_tape_on_smollm2() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (a, fp, _shapes) = extract(&runner);
    let tokens: Vec<u32> = vec![1, 338, 263, 1243, 310, 278, 4086, 29889];
    let (seq, vocab) = (tokens.len(), a.vocab);

    // CPU tape (the None path — the validated reference).
    let t0 = Instant::now();
    let cpu_logits = logits_of(&fp, &a, &tokens);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    // GPU tape: every dense_matmul dispatched to the device via GpuGemm.
    let gemm: Rc<dyn TrainGemm> = Rc::new(GpuGemm::new(0).expect("open CUDA device"));
    let t1 = Instant::now();
    let mut t = Tape::with_gemm(gemm);
    let wids: Vec<_> = fp.iter().map(|w| t.leaf(w.clone())).collect();
    let out = forward(&mut t, &wids, &a, &tokens);
    let gpu_logits = t.value(out).to_vec();
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

    // Compare the last-token logits (the row the runner conformance also checks).
    let base = (seq - 1) * vocab;
    let (cl, gl) = (
        &cpu_logits[base..base + vocab],
        &gpu_logits[base..base + vocab],
    );
    let max_abs = cl
        .iter()
        .zip(gl)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let range = cl.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - cl.iter().copied().fold(f32::INFINITY, f32::min);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    println!(
        "0043 P1 GPU-tape parity (SmolLM2, {} layers, seq {seq}): argmax cpu {} vs gpu {}; \
         max|Δlogit| {max_abs:.4e} / range {range:.2} = {:.2e} rel. Fwd time CPU {cpu_ms:.0}ms | GPU {gpu_ms:.0}ms \
         (host-orchestrated; per-matmul round-trips — device-resident is Phase 2).",
        a.n_layers,
        argmax(cl),
        argmax(gl),
        max_abs / range,
    );

    assert_eq!(
        argmax(cl),
        argmax(gl),
        "GPU tape must predict the same token as the CPU tape"
    );
    assert!(
        max_abs / range < 5e-3,
        "GPU tape logits must match the CPU tape within 0.5% of the logit range: max|Δ| {max_abs:.4e}"
    );
}
