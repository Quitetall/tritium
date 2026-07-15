//! Whole-model differentiable forward vs the inference `ModelRunner` (plan 0040 step 3b).
//!
//! Assembles a full standard-transformer forward on the autograd `Tape` — embedding gather → N
//! blocks (rmsnorm → GQA attention via `nn::attention` → residual → rmsnorm → SwiGLU → residual) →
//! final rmsnorm → tied lm-head — from SmolLM2-135M's real fp weights (un-quantized leaves), and
//! asserts its last-token logits match `ModelRunner::from_hf` (the deployed inference forward). This
//! is the INDEPENDENT cross-check that the tape forward is faithful — the prerequisite for running
//! the SALT-distillation loop on a real model (step 4). `#[ignore]`d; run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test tape_model_conformance -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_nn::{Mlp, ModelRunner, Projection};
use tritium_train::Tape;
use tritium_train::nn::attention;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn dense(p: &Projection) -> (&[f32], usize, usize) {
    match p {
        Projection::Dense(d) => (&d.weights, d.n_out, d.k_in),
        Projection::Salt(_) | Projection::Ternary(_) => {
            panic!("from_hf builds Dense projections")
        }
    }
}

#[test]
#[ignore = "needs SmolLM2-135M under ~/.cache/tritium-models/smollm2-135m; run explicitly"]
fn tape_model_matches_modelrunner_on_smollm2() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let cpu = || Box::new(tritium_cpu::CpuBackend::new());
    let mut runner = ModelRunner::from_hf(&dir, cpu()).expect("from_hf");

    // A short prompt (keep the seq×vocab head matmul cheap).
    let tokens: Vec<u32> = vec![1, 338, 263, 1243, 310, 278, 4086, 29889];
    let seq = tokens.len();
    let positions: Vec<usize> = (0..seq).collect();

    // Reference: the inference runner's last-token logits.
    runner.reset();
    let ref_logits = runner.forward(&tokens, &positions).expect("runner forward");

    let cfg = &runner.config;
    let (n_embd, n_head, n_head_kv, head_dim) = (
        cfg.n_embd as usize,
        cfg.n_head as usize,
        cfg.n_head_kv as usize,
        cfg.head_dim() as usize,
    );
    let (eps, theta) = (cfg.rms_eps, cfg.rope_theta);
    let w = &runner.weights;
    let (vocab, n_layers) = (w.vocab, w.layers.len());
    assert!(w.lm_head.is_none(), "test assumes SmolLM2's tied lm-head");

    // Build the differentiable forward on the tape from the fp weights (leaves = un-quantized).
    let mut t = Tape::new();
    let embd = t.leaf(
        w.token_embd
            .as_dense()
            .expect("conformance fixture requires dense token embedding")
            .to_vec(),
    );
    let mut hidden = t.embed_gather(embd, &tokens, vocab, n_embd);

    for li in 0..n_layers {
        let b = &w.layers[li];
        let an = t.leaf(b.attn_norm.clone());
        let xn = t.rmsnorm(hidden, an, seq, n_embd, eps);
        let wq = t.leaf(dense(&b.q_proj).0.to_vec());
        let wk = t.leaf(dense(&b.k_proj).0.to_vec());
        let wv = t.leaf(dense(&b.v_proj).0.to_vec());
        let wo = t.leaf(dense(&b.o_proj).0.to_vec());
        let attn = attention(
            &mut t, xn, wq, wk, wv, wo, seq, n_embd, n_head, n_head_kv, head_dim, theta,
        );
        hidden = t.add(hidden, attn);

        let (gate, up, down) = match &b.mlp {
            Mlp::SwiGlu(m) => (&m.gate, &m.up, &m.down),
            Mlp::Relu2(_) => panic!("SmolLM2 is SwiGLU"),
        };
        let ff = dense(gate).1;
        let fnw = t.leaf(b.ffn_norm.clone());
        let hn = t.rmsnorm(hidden, fnw, seq, n_embd, eps);
        let wg = t.leaf(dense(gate).0.to_vec());
        let wu = t.leaf(dense(up).0.to_vec());
        let wd = t.leaf(dense(down).0.to_vec());
        let g = t.dense_matmul(hn, wg, seq, ff, n_embd);
        let u = t.dense_matmul(hn, wu, seq, ff, n_embd);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let down_out = t.dense_matmul(gated, wd, seq, n_embd, ff);
        hidden = t.add(hidden, down_out);
    }

    let onw = t.leaf(w.output_norm.clone());
    let fnorm = t.rmsnorm(hidden, onw, seq, n_embd, eps);
    // Tied head: logits = fnorm · token_embdᵀ. Reuse the embedding leaf (same weight).
    let logits = t.dense_matmul(fnorm, embd, seq, vocab, n_embd);
    let all = t.value(logits);
    let last = seq - 1;
    let tape_last = &all[last * vocab..last * vocab + vocab];

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    let (a_tape, a_ref) = (argmax(tape_last), argmax(&ref_logits));
    let max_abs = tape_last
        .iter()
        .zip(&ref_logits)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ref_range = ref_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - ref_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "0040 tape-model conformance (SmolLM2, {n_layers} layers, seq {seq}): argmax tape {a_tape} vs runner {a_ref}; \
         max|Δlogit| {max_abs:.4e}, logit range {ref_range:.2}, rel {:.2e}",
        max_abs / ref_range
    );

    assert_eq!(
        a_tape, a_ref,
        "tape forward must predict the same next token as the runner"
    );
    assert!(
        max_abs / ref_range < 5e-3,
        "tape logits must match the runner within 0.5% of the logit range: max|Δ| {max_abs:.4e}"
    );
}
