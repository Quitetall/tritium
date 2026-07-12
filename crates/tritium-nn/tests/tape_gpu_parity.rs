//! GPU training path — Phase 1 parity gate (plan 0043). Runs the same whole-model tape step
//! (embed → 30 GQA blocks → SwiGLU → tied head → scalar loss → backward) two ways on SmolLM2-135M:
//! the CPU tape, and a tape whose matmuls dispatch to the GPU (`Tape::with_gemm(GpuGemm)` → the
//! bit-exact `train_grad.cu` kernels). Asserts the GPU-backed **logits and gradients** match the CPU
//! tape — validating the pluggable `TrainGemm` seam (forward AND backward) end-to-end on a real
//! model — and prints CPU-vs-GPU step time. `#[ignore]`d (needs SmolLM2-135M + a CUDA device); run:
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test tape_gpu_parity -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use common::{extract, forward};
use tritium_cuda::train::GpuGemm;
use tritium_nn::ModelRunner;
use tritium_train::{Tape, TrainGemm};

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1000) as f32 / 500.0 - 1.0
        })
        .collect()
}

fn max_rel(cpu: &[f32], gpu: &[f32]) -> (f32, f32) {
    let max_abs = cpu
        .iter()
        .zip(gpu)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - cpu.iter().copied().fold(f32::INFINITY, f32::min);
    (max_abs, max_abs / range.max(1e-9))
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
    // A realistic distillation sequence length (64) so the per-layer matmuls are big enough to
    // characterize where host-orchestrated GPU dispatch overtakes the CPU tape — not just the
    // tiny-matmul, transfer-bound regime. Parity holds at any length; the timing is what varies.
    let base = [1u32, 338, 263, 1243, 310, 278, 4086, 29889];
    let tokens: Vec<u32> = (0..64).map(|i| base[i % base.len()]).collect();
    let (seq, vocab) = (tokens.len(), a.vocab);
    let cot = seeded(9, seq * vocab); // fixed cotangent for the scalar loss L = Σ out·r

    // One full step (forward → scalar loss → backward) on the given tape. Returns the last-token
    // logits and the tied-embedding gradient (index 0), plus wall time.
    let run = |mut t: Tape| -> (Vec<f32>, Vec<f32>, f64) {
        let start = Instant::now();
        let wids: Vec<_> = fp.iter().map(|w| t.leaf(w.clone())).collect();
        let out = forward(&mut t, &wids, &a, &tokens); // [seq, vocab]
        let r = t.leaf(cot.clone());
        let scalar = t.dense_matmul(out, r, 1, 1, seq * vocab); // Σ out·r
        let grads = t.backward(scalar);
        let last = t.value(out)[(seq - 1) * vocab..seq * vocab].to_vec();
        let g_embd = grads[wids[0]].clone();
        (last, g_embd, start.elapsed().as_secs_f64() * 1e3)
    };

    let (cpu_logits, cpu_gembd, cpu_ms) = run(Tape::new());
    let gemm: Rc<dyn TrainGemm> = Rc::new(GpuGemm::new(0).expect("open CUDA device"));
    let (gpu_logits, gpu_gembd, gpu_ms) = run(Tape::with_gemm(gemm));

    let (logit_abs, logit_rel) = max_rel(&cpu_logits, &gpu_logits);
    let (grad_abs, grad_rel) = max_rel(&cpu_gembd, &gpu_gembd);
    println!(
        "0043 P1 GPU-tape parity (SmolLM2, {} layers, seq {seq}): \
         logits max|Δ| {logit_abs:.3e} ({logit_rel:.2e} rel) | embd-grad max|Δ| {grad_abs:.3e} ({grad_rel:.2e} rel). \
         Full fwd+bwd step: CPU {cpu_ms:.0}ms | GPU {gpu_ms:.0}ms (host-orchestrated; device-resident is Phase 2).",
        a.n_layers,
    );

    assert!(logit_rel < 5e-3, "forward logits GPU vs CPU: {logit_abs:.3e}");
    assert!(grad_rel < 5e-3, "backward embd-grad GPU vs CPU: {grad_abs:.3e}");
}
