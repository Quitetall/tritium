//! GPU training path — P2.5c capstone gate (plan 0043). Assembles the WHOLE SmolLM2-135M model
//! (embed → 30 GQA blocks → SwiGLU → tied head) on the device-resident `DeviceTape` and runs a full
//! forward+backward entirely in VRAM, then compares its logits and the tied-embedding gradient to
//! the `tritium-train` CPU tape (`common::forward`) under a softmax-xent distillation loss. This is
//! the DeviceTape validated on a REAL 30-layer model — the proof that the device-resident engine
//! (P2.1–P2.5b) trains an actual transformer, not just synthetic blocks. `#[ignore]`d (needs
//! SmolLM2-135M + a CUDA device); run:
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test device_tape_real_model -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{device_forward, extract, forward};
use tritium_cuda::CudaBackend;
use tritium_cuda::train::DeviceTape;
use tritium_nn::ModelRunner;
use tritium_train::Tape;

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

fn row_softmax(logits: &[f32], vocab: usize) -> Vec<f32> {
    let mut p = logits.to_vec();
    for row in p.chunks_mut(vocab) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    p
}

fn max_rel(cpu: &[f32], dev: &[f32]) -> f32 {
    let max_abs = cpu
        .iter()
        .zip(dev)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let range = cpu.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - cpu.iter().copied().fold(f32::INFINITY, f32::min);
    max_abs / range.max(1e-9)
}

#[test]
#[ignore = "needs SmolLM2-135M + a CUDA device; run explicitly"]
fn device_tape_trains_smollm2_matching_cpu_tape() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (a, fp, _shapes) = extract(&runner);
    let tokens: Vec<u32> = vec![1, 338, 263, 1243, 310, 278, 4086, 29889];
    let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let (seq, vocab) = (tokens.len(), a.vocab);
    // Distillation target: the fp model's own next-token distribution (a real soft target).
    let target = row_softmax(&seeded(7, seq * vocab), vocab);

    // ── CPU tape reference (the validated common::forward + softmax-xent) ──
    let cpu_start = Instant::now();
    let mut t = Tape::new();
    let cwids: Vec<_> = fp.iter().map(|w| t.leaf(w.clone())).collect();
    let clogits = forward(&mut t, &cwids, &a, &tokens);
    let cpu_logits = t.value(clogits).to_vec();
    let ctg = t.leaf(target.clone());
    let closs = t.softmax_xent(clogits, ctg, seq, vocab);
    let cgrads = t.backward(closs);
    let cpu_embd_grad = cgrads[cwids[0]].clone();
    let cpu_wd0_grad = cgrads[cwids[7]].clone(); // layer-0 down-proj: an intermediate leaf-sink grad_w
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

    // ── DeviceTape: whole model fwd+bwd on the GPU ──
    let backend = CudaBackend::new(0).expect("open CUDA device");
    // Warm up (CUDA context + kernel JIT + allocator) so the timing below reflects steady state, not
    // a cold first launch — a single cold measurement is unreliable in either direction.
    {
        let mut warm = DeviceTape::new(&backend, vocab).expect("device tape");
        let (wl, ww) = device_forward(&mut warm, &a, &fp, &tokens_i32, seq);
        let _ = warm
            .xent_backward(wl, &target, seq, vocab, &[ww[0]])
            .unwrap();
    }
    let dev_start = Instant::now();
    let mut dt = DeviceTape::new(&backend, vocab).expect("device tape");
    let (logits, wids) = device_forward(&mut dt, &a, &fp, &tokens_i32, seq);
    let dev_logits = dt.value(logits).unwrap();
    // Request the tied embed (wids[0]) AND layer-0's wd (wids[7]) — the latter is an intermediate
    // grad_w sink that the tied-embed grad alone would not exercise.
    let grads = dt
        .xent_backward(logits, &target, seq, vocab, &[wids[0], wids[7]])
        .unwrap();
    let (dev_embd_grad, dev_wd0_grad) = (&grads[0], &grads[1]);
    let dev_ms = dev_start.elapsed().as_secs_f64() * 1e3;

    let logit_rel = max_rel(&cpu_logits, &dev_logits);
    let grad_rel = max_rel(&cpu_embd_grad, dev_embd_grad);
    let wd0_rel = max_rel(&cpu_wd0_grad, dev_wd0_grad);
    println!(
        "0043 P2.5c DeviceTape ON REAL SmolLM2-135M ({} layers, seq {seq}, vocab {vocab}): \
         logits rel {logit_rel:.2e} | tied-embd grad rel {grad_rel:.2e} | layer0 wd grad rel {wd0_rel:.2e} \
         vs CPU tape. whole-model fwd+bwd: device {dev_ms:.0}ms | CPU tape {cpu_ms:.0}ms ({:.1}× faster).",
        a.n_layers,
        cpu_ms / dev_ms.max(1e-9)
    );

    assert!(
        logit_rel < 1e-4,
        "device logits vs CPU tape: {logit_rel:.3e}"
    );
    assert!(
        grad_rel < 1e-4,
        "device tied-embed grad vs CPU tape: {grad_rel:.3e}"
    );
    assert!(
        wd0_rel < 1e-4,
        "device layer0 wd grad vs CPU tape: {wd0_rel:.3e}"
    );
}
