#![allow(dead_code, unreachable_pub)]
//! Shared differentiable-model helpers for the SALT-distillation tests (the ModelRunner-validated
//! tape forward, ppl, and weight extraction). Kept in `tests/common` so multiple test binaries
//! reuse the one validated forward instead of copying it.

use tritium_nn::{Mlp, ModelRunner, Projection};
use tritium_train::nn::attention;
use tritium_train::tape::ValueId;
use tritium_train::Tape;

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

/// The device mirror of [`forward`] on the GPU [`DeviceTape`](tritium_cuda::train::DeviceTape)
/// (plan 0043 P2.5): assembles the whole model on-device using `weights` as the 2D-weight leaves
/// (fp order: `[0]` = tied embed, then per layer `q,k,v,o,gate,up,down`) and the `Arch` fp norms.
/// Returns the logits value id and the weight-leaf ids **in `weights` order** (so a caller can map
/// downloaded grads straight back to the master weights). The op sequence is identical to
/// [`forward`], so the device tape reproduces it within 1e-4.
#[cfg(feature = "cuda")]
pub fn device_forward<'backend, 'leaf>(
    dt: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    a: &Arch,
    weights: &[Vec<f32>],
    tokens_i32: &[i32],
    seq: usize,
) -> (usize, Vec<usize>) {
    device_forward_with(dt, a, tokens_i32, seq, |dt, index| {
        dt.leaf(&weights[index]).unwrap()
    })
}

/// The zero-copy variant of [`device_forward`]: each trainable weight is an
/// already-resident tensor borrowed from `DeviceTrainer`.
#[cfg(feature = "cuda")]
pub fn device_forward_resident<'backend, 'leaf>(
    dt: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    a: &Arch,
    weights: &[&'leaf tritium_cuda::train::DeviceTensor],
    tokens_i32: &[i32],
    seq: usize,
) -> (usize, Vec<usize>) {
    device_forward_with(dt, a, tokens_i32, seq, |dt, index| {
        dt.leaf_device(weights[index]).unwrap()
    })
}

#[cfg(feature = "cuda")]
fn device_forward_with<'backend, 'leaf>(
    dt: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    a: &Arch,
    tokens_i32: &[i32],
    seq: usize,
    mut weight_leaf: impl FnMut(&mut tritium_cuda::train::DeviceTape<'backend, 'leaf>, usize) -> usize,
) -> (usize, Vec<usize>) {
    let embd = weight_leaf(dt, 0);
    let mut wids = vec![embd];
    let mut hidden = dt.embed(embd, tokens_i32, seq, a.n_embd, a.vocab).unwrap();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = dt.leaf(&a.attn_norms[li]).unwrap();
        let xn = dt.rmsnorm(hidden, an, seq, a.n_embd, a.eps).unwrap();
        let wq = weight_leaf(dt, base);
        let wk = weight_leaf(dt, base + 1);
        let wv = weight_leaf(dt, base + 2);
        let wo = weight_leaf(dt, base + 3);
        let attn = dt
            .attention(
                xn,
                wq,
                wk,
                wv,
                wo,
                seq,
                a.n_embd,
                a.n_head,
                a.n_head_kv,
                a.head_dim,
                a.theta,
            )
            .unwrap();
        hidden = dt.add(hidden, attn).unwrap();
        let fnw = dt.leaf(&a.ffn_norms[li]).unwrap();
        let hn = dt.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps).unwrap();
        let wg = weight_leaf(dt, base + 4);
        let wu = weight_leaf(dt, base + 5);
        let wd = weight_leaf(dt, base + 6);
        let g = dt.matmul(hn, wg, seq, a.ff, a.n_embd).unwrap();
        let u = dt.matmul(hn, wu, seq, a.ff, a.n_embd).unwrap();
        let ga = dt.silu(g).unwrap();
        let gated = dt.mul(ga, u).unwrap();
        let down = dt.matmul(gated, wd, seq, a.n_embd, a.ff).unwrap();
        hidden = dt.add(hidden, down).unwrap();
        wids.extend([wq, wk, wv, wo, wg, wu, wd]);
    }
    let onw = dt.leaf(&a.out_norm).unwrap();
    let fnorm = dt.rmsnorm(hidden, onw, seq, a.n_embd, a.eps).unwrap();
    let logits = dt.matmul(fnorm, embd, seq, a.vocab, a.n_embd).unwrap(); // tied head
    (logits, wids)
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
