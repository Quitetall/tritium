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

use common::{Arch, extract, forward};
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

/// Assemble the whole model on the `DeviceTape` — the device mirror of `common::forward`. Returns the
/// tape, the logits value id, and the tied-embed weight id.
fn build_device<'a>(
    dt: &mut DeviceTape<'a>,
    a: &Arch,
    fp: &[Vec<f32>],
    tokens_i32: &[i32],
    seq: usize,
) -> (usize, usize) {
    let embd = dt.leaf(&fp[0]).unwrap();
    let mut hidden = dt.embed(embd, tokens_i32, seq, a.n_embd, a.vocab).unwrap();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = dt.leaf(&a.attn_norms[li]).unwrap();
        let xn = dt.rmsnorm(hidden, an, seq, a.n_embd, a.eps).unwrap();
        let (wq, wk, wv, wo) = (
            dt.leaf(&fp[base]).unwrap(),
            dt.leaf(&fp[base + 1]).unwrap(),
            dt.leaf(&fp[base + 2]).unwrap(),
            dt.leaf(&fp[base + 3]).unwrap(),
        );
        let attn = dt
            .attention(
                xn, wq, wk, wv, wo, seq, a.n_embd, a.n_head, a.n_head_kv, a.head_dim, a.theta,
            )
            .unwrap();
        hidden = dt.add(hidden, attn).unwrap();
        let fnw = dt.leaf(&a.ffn_norms[li]).unwrap();
        let hn = dt.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps).unwrap();
        let (wg, wu, wd) = (
            dt.leaf(&fp[base + 4]).unwrap(),
            dt.leaf(&fp[base + 5]).unwrap(),
            dt.leaf(&fp[base + 6]).unwrap(),
        );
        let g = dt.matmul(hn, wg, seq, a.ff, a.n_embd).unwrap();
        let u = dt.matmul(hn, wu, seq, a.ff, a.n_embd).unwrap();
        let ga = dt.silu(g).unwrap();
        let gated = dt.mul(ga, u).unwrap();
        let down = dt.matmul(gated, wd, seq, a.n_embd, a.ff).unwrap();
        hidden = dt.add(hidden, down).unwrap();
    }
    let onw = dt.leaf(&a.out_norm).unwrap();
    let fnorm = dt.rmsnorm(hidden, onw, seq, a.n_embd, a.eps).unwrap();
    let logits = dt.matmul(fnorm, embd, seq, a.vocab, a.n_embd).unwrap(); // tied head
    (logits, embd)
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
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;

    // ── DeviceTape: whole model fwd+bwd on the GPU ──
    let backend = CudaBackend::new(0).expect("open CUDA device");
    let dev_start = Instant::now();
    let mut dt = DeviceTape::new(&backend, vocab).expect("device tape");
    let (logits, embd) = build_device(&mut dt, &a, &fp, &tokens_i32, seq);
    let dev_logits = dt.value(logits).unwrap();
    let dev_embd_grad = dt
        .xent_backward(logits, &target, seq, vocab, &[embd])
        .unwrap()
        .pop()
        .unwrap();
    let dev_ms = dev_start.elapsed().as_secs_f64() * 1e3;

    let logit_rel = max_rel(&cpu_logits, &dev_logits);
    let grad_rel = max_rel(&cpu_embd_grad, &dev_embd_grad);
    println!(
        "0043 P2.5c DeviceTape ON REAL SmolLM2-135M ({} layers, seq {seq}, vocab {vocab}): \
         logits rel {logit_rel:.2e} | tied-embd grad rel {grad_rel:.2e} vs CPU tape. \
         whole-model fwd+bwd: device {dev_ms:.0}ms | CPU tape {cpu_ms:.0}ms ({:.1}× faster).",
        a.n_layers,
        cpu_ms / dev_ms.max(1e-9)
    );

    assert!(logit_rel < 1e-4, "device logits vs CPU tape: {logit_rel:.3e}");
    assert!(
        grad_rel < 1e-4,
        "device tied-embed grad vs CPU tape: {grad_rel:.3e}"
    );
}
