#![allow(dead_code, unreachable_pub)]
//! Shared differentiable-model helpers for the SALT-distillation tests (the ModelRunner-validated
//! tape forward, ppl, and weight extraction). Kept in `tests/common` so multiple test binaries
//! reuse the one validated forward instead of copying it.

use tritium_nn::{Mlp, ModelRunner, Projection};
use tritium_train::Tape;
use tritium_train::nn::attention;
use tritium_train::tape::ValueId;

/// Dims + fp 1D norms for the tape forward (the parts held fp, not quantized/trained).
pub struct Arch {
    pub attn_norms: Vec<Vec<f32>>,
    pub ffn_norms: Vec<Vec<f32>>,
    pub out_norm: Vec<f32>,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
    pub n_layers: usize,
}

pub fn dense(p: &Projection) -> (Vec<f32>, usize, usize) {
    match p {
        Projection::Dense(d) => (d.weights.clone(), d.n_out, d.k_in),
        Projection::Ternary(_) => panic!("from_hf builds Dense projections"),
    }
}

/// Extract the tape `Arch`, the flat fp weights (index 0 = tied token_embd, then per layer
/// q,k,v,o,gate,up,down), and their `(n_out, k_in)` shapes from a loaded runner.
pub fn extract(runner: &ModelRunner) -> (Arch, Vec<Vec<f32>>, Vec<(usize, usize)>) {
    let cfg = &runner.config;
    let w = &runner.weights;
    assert!(w.lm_head.is_none(), "assumes tied lm-head");
    let arch = Arch {
        attn_norms: w.layers.iter().map(|b| b.attn_norm.clone()).collect(),
        ffn_norms: w.layers.iter().map(|b| b.ffn_norm.clone()).collect(),
        out_norm: w.output_norm.clone(),
        n_embd: cfg.n_embd as usize,
        n_head: cfg.n_head as usize,
        n_head_kv: cfg.n_head_kv as usize,
        head_dim: cfg.head_dim() as usize,
        ff: match &w.layers[0].mlp {
            Mlp::SwiGlu(m) => dense(&m.gate).1,
            Mlp::Relu2(_) => panic!("SwiGLU"),
        },
        vocab: w.vocab,
        eps: cfg.rms_eps,
        theta: cfg.rope_theta,
        n_layers: w.layers.len(),
    };
    let mut fp: Vec<Vec<f32>> = vec![w.token_embd.clone()];
    let mut shapes: Vec<(usize, usize)> = vec![(w.vocab, arch.n_embd)];
    for b in w.layers.iter() {
        let (gate, up, down) = match &b.mlp {
            Mlp::SwiGlu(m) => (&m.gate, &m.up, &m.down),
            Mlp::Relu2(_) => unreachable!(),
        };
        for p in [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, gate, up, down] {
            let (wv, n, k) = dense(p);
            fp.push(wv);
            shapes.push((n, k));
        }
    }
    (arch, fp, shapes)
}

/// The whole-model forward on the tape (validated bit-exact vs `ModelRunner` in
/// `tape_model_conformance`): embed_gather → N GQA blocks → SwiGLU → tied head. `wids[0]` is the
/// tied embed/head; then per layer `q,k,v,o,gate,up,down`. Returns `[tokens.len(), vocab]` logits.
pub fn forward(t: &mut Tape, wids: &[ValueId], a: &Arch, tokens: &[u32]) -> ValueId {
    let seq = tokens.len();
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        let attn = attention(
            t,
            xn,
            wids[base],
            wids[base + 1],
            wids[base + 2],
            wids[base + 3],
            seq,
            a.n_embd,
            a.n_head,
            a.n_head_kv,
            a.head_dim,
            a.theta,
        );
        hidden = t.add(hidden, attn);
        let fnw = t.leaf(a.ffn_norms[li].clone());
        let hn = t.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps);
        let g = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let u = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        hidden = t.add(hidden, down);
    }
    let onw = t.leaf(a.out_norm.clone());
    let fnorm = t.rmsnorm(hidden, onw, seq, a.n_embd, a.eps);
    t.dense_matmul(fnorm, wids[0], seq, a.vocab, a.n_embd) // tied head
}

/// Forward once with a fixed weight set (leaves), returning `[seq, vocab]` logits values.
pub fn logits_of(weights: &[Vec<f32>], a: &Arch, tokens: &[u32]) -> Vec<f32> {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    let out = forward(&mut t, &wids, a, tokens);
    t.value(out).to_vec()
}

/// Teacher-forced perplexity from `[seq, vocab]` logits over `tokens`.
pub fn perplexity(logits: &[f32], tokens: &[u32], vocab: usize) -> f64 {
    let seq = tokens.len();
    let mut nll = 0.0f64;
    for tpos in 0..seq - 1 {
        let row = &logits[tpos * vocab..tpos * vocab + vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = m + row
            .iter()
            .map(|&x| (f64::from(x) - m).exp())
            .sum::<f64>()
            .ln();
        nll += lse - f64::from(row[tokens[tpos + 1] as usize]);
    }
    (nll / (seq - 1) as f64).exp()
}
