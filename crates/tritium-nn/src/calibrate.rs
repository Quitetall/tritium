//! Activation calibration and the AWQ-style salience fold.
//!
//! **Promoted out of `crates/tritium-nn/tests/common/mod.rs`, where it lived for the whole SALT
//! campaign.** Every published ladder number was measured under this fold at α = 0.75, yet nothing
//! in the shipping tree could call it — `tritium quantize` reads a bare safetensors file and has no
//! way to compute activation statistics. That gap is why the CLI's artifacts are a *different
//! configuration* from the research runs (measured: the fold is worth 2.3× at T=2, 8.5% at T=3, and
//! 1.1% at T=4). This module is the prerequisite for closing it.
//!
//! # Why this is model-aware, not a tensor op
//!
//! [`fold`] takes an [`Arch`] and **returns a modified one**: it scales weight columns by the
//! per-channel salience and divides the *preceding norm* by the same factor, so the product is
//! unchanged while the weights become easier to quantize. That requires knowing which norm feeds
//! which projection — a property of the layer graph, not of any single tensor. Any command offering
//! calibration therefore has to load a model, not a file of tensors.
//!
//! # Scope
//!
//! The Gram/curvature machinery (`Gram`, `GramSet`, `damped_inverse`, `calibrate_gram`) stays in the
//! test tree deliberately: per-group curvature allocation was measured **negative** — 12.4% lower
//! weight SSE and 12% worse perplexity at identical bits — so it is a research artifact, not a
//! shipping path.
//!
//! `o_proj` is not calibrated. Its input is the attention concat, which under GQA has query heads
//! sharing kv dims — exactly the case the salience fold also skips.

use tritium_train::Tape;
use tritium_train::nn::attention;
use tritium_train::tape::ValueId;

use crate::{Mlp, ModelRunner, Projection};

/// Dims + fp 1D norms for the tape forward (the parts held fp, not quantized/trained).
#[derive(Debug, Clone)]
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

/// Unwrap a `Dense` projection to `(weights, n_out, k_in)`. Panics on any compact projection:
/// calibration reads the fp master, so a quantized model is a caller error, not a runtime case.
pub fn dense(p: &Projection) -> (Vec<f32>, usize, usize) {
    match p {
        Projection::Dense(d) => (d.weights.clone(), d.n_out, d.k_in),
        Projection::Salt(_)
        | Projection::HostSaltV2(_)
        | Projection::Ternary(_)
        | Projection::Q2(_) => {
            panic!("from_hf builds Dense projections")
        }
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => panic!("from_hf builds Dense projections"),
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
    let mut fp: Vec<Vec<f32>> = vec![
        w.token_embd
            .as_dense()
            .expect("training reference requires dense token embedding")
            .to_vec(),
    ];
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

/// Per-input-channel second moments for the projections whose scale is foldable.
#[derive(Debug, Clone)]
pub struct Calib {
    pub attn_in: Vec<Vec<f64>>, // [layer][n_embd] — input to q, k, v
    pub ffn_in: Vec<Vec<f64>>,  // [layer][n_embd] — input to gate, up
    pub down_in: Vec<Vec<f64>>, // [layer][ff]     — input to down
    pub rows: usize,
}

impl Calib {
    pub fn new(a: &Arch) -> Self {
        Self {
            attn_in: vec![vec![0.0; a.n_embd]; a.n_layers],
            ffn_in: vec![vec![0.0; a.n_embd]; a.n_layers],
            down_in: vec![vec![0.0; a.ff]; a.n_layers],
            rows: 0,
        }
    }
}

/// Accumulate `Σ x²` per column of a `[seq, cols]` tape value.
pub fn accumulate(t: &Tape, id: ValueId, seq: usize, cols: usize, acc: &mut [f64]) {
    let v = t.value(id);
    for r in 0..seq {
        for (c, a) in acc.iter_mut().enumerate() {
            let x = f64::from(v[r * cols + c]);
            *a += x * x;
        }
    }
}

/// One calibration forward. Mirrors [`forward`] exactly, tapping the three foldable inputs.
///
/// Only the diagonal (`Σ x²` per channel) is accumulated, which is all the salience fold needs. The
/// full Gram `Σ x xᵀ` that sequential error compensation would require stays in the test tree with
/// the rest of the curvature machinery — see the module docs for why.
pub fn calibrate(weights: &[Vec<f32>], a: &Arch, tokens: &[u32], c: &mut Calib) {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    let seq = tokens.len();
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        accumulate(&t, xn, seq, a.n_embd, &mut c.attn_in[li]);
        let attn = attention(
            &mut t,
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
        accumulate(&t, hn, seq, a.n_embd, &mut c.ffn_in[li]);
        let g = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let u = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        accumulate(&t, gated, seq, a.ff, &mut c.down_in[li]);
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        hidden = t.add(hidden, down);
    }
    c.rows += seq;
}

/// AWQ's salience scale: `s_j = (rms_j / geomean(rms))^α`, normalised by the geometric mean so the
/// fold neither inflates nor shrinks the matrix overall (α=0 ⇒ all ones ⇒ exactly today's fitter).
pub fn smooth_scales(acc: &[f64], rows: usize, alpha: f64) -> Vec<f32> {
    let rms: Vec<f64> = acc
        .iter()
        .map(|&s| (s / rows as f64).sqrt().max(1e-8))
        .collect();
    let gm = (rms.iter().map(|r| r.ln()).sum::<f64>() / rms.len() as f64).exp();
    rms.iter().map(|&r| (r / gm).powf(alpha) as f32).collect()
}

/// Multiply every column `j` of a row-major `[rows, cols]` matrix by `s[j]` — the weight half of
/// the fold.
pub fn scale_cols(w: &mut [f32], cols: usize, s: &[f32]) {
    for row in w.chunks_mut(cols) {
        for (v, &sj) in row.iter_mut().zip(s) {
            *v *= sj;
        }
    }
}

/// Divide every row `r` by `s[r]` — the norm half of the fold, applied to the *preceding* norm so
/// the product with [`scale_cols`] is the identity.
pub fn divide_rows(w: &mut [f32], cols: usize, s: &[f32]) {
    for (r, row) in w.chunks_mut(cols).enumerate() {
        for v in row {
            *v /= s[r];
        }
    }
}

/// Apply the salience fold. Returns the rebalanced weights plus the `Arch` whose fp norms absorb
/// the inverse — together an exact reparameterisation of the same function.
pub fn fold(
    fp: &[Vec<f32>],
    shapes: &[(usize, usize)],
    a: &Arch,
    c: &Calib,
    alpha: f64,
) -> (Vec<Vec<f32>>, Arch) {
    let mut w = fp.to_vec();
    let mut arch = a.clone();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;

        // q, k, v — inverse into attn_norm.
        let sa = smooth_scales(&c.attn_in[li], c.rows, alpha);
        for k in 0..3 {
            scale_cols(&mut w[base + k], shapes[base + k].1, &sa);
        }
        for (n, &s) in arch.attn_norms[li].iter_mut().zip(&sa) {
            *n /= s;
        }

        // gate, up — inverse into ffn_norm.
        let sf = smooth_scales(&c.ffn_in[li], c.rows, alpha);
        for k in 4..6 {
            scale_cols(&mut w[base + k], shapes[base + k].1, &sf);
        }
        for (n, &s) in arch.ffn_norms[li].iter_mut().zip(&sf) {
            *n /= s;
        }

        // down — inverse into up's output rows (`gated = silu(gate) ⊙ up` is linear in up).
        let sd = smooth_scales(&c.down_in[li], c.rows, alpha);
        scale_cols(&mut w[base + 6], shapes[base + 6].1, &sd);
        divide_rows(&mut w[base + 5], shapes[base + 5].1, &sd);
    }
    (w, arch)
}
