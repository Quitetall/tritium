#![allow(dead_code, unreachable_pub)]
//! Shared differentiable-model helpers for the SALT-distillation tests (the ModelRunner-validated
//! tape forward, ppl, and weight extraction). Kept in `tests/common` so multiple test binaries
//! reuse the one validated forward instead of copying it.

use tritium_nn::{Mlp, ModelRunner, Projection};
use tritium_train::Tape;
use tritium_train::nn::attention;
use tritium_train::ops::ste::{self, RotationPolicy};
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
        Projection::Salt(_) | Projection::HostSaltV2(_) | Projection::Ternary(_) => {
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

/// Packed-SALT device forward used by the real-model training gate. Each
/// compact weight gets exactly one gradient-only latent-master leaf; the tied
/// embedding/head deliberately reuse master zero. Block boundaries expose only
/// the residual stream so checkpoint policies can evict and replay internals.
#[cfg(feature = "cuda")]
pub fn device_forward_packed<'backend, 'leaf>(
    dt: &mut tritium_cuda::train::DeviceTape<'backend, 'leaf>,
    a: &Arch,
    weights: &'leaf [tritium_cuda::train::DevicePackedSaltWeight],
    tokens_i32: &[i32],
    seq: usize,
) -> (usize, Vec<usize>) {
    assert_eq!(seq, tokens_i32.len());
    assert_eq!(weights.len(), 1 + 7 * a.n_layers);
    let masters: Vec<usize> = weights
        .iter()
        .map(|weight| {
            let len = weight
                .rows()
                .checked_mul(weight.cols())
                .expect("packed master shape overflow");
            dt.gradient_leaf(len).unwrap()
        })
        .collect();

    let mut hidden = dt.salt_embed(masters[0], &weights[0], tokens_i32).unwrap();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = dt.leaf(&a.attn_norms[li]).unwrap();
        let xn = dt.rmsnorm(hidden, an, seq, a.n_embd, a.eps).unwrap();
        let attn = dt
            .salt_attention(
                xn,
                masters[base],
                &weights[base],
                masters[base + 1],
                &weights[base + 1],
                masters[base + 2],
                &weights[base + 2],
                masters[base + 3],
                &weights[base + 3],
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
        let gate = dt
            .salt_matmul(hn, masters[base + 4], &weights[base + 4], seq)
            .unwrap();
        let up = dt
            .salt_matmul(hn, masters[base + 5], &weights[base + 5], seq)
            .unwrap();
        let activated_gate = dt.silu(gate).unwrap();
        let gated = dt.mul(activated_gate, up).unwrap();
        let down = dt
            .salt_matmul(gated, masters[base + 6], &weights[base + 6], seq)
            .unwrap();
        hidden = dt.add(hidden, down).unwrap();
        dt.checkpoint_keep(&[hidden]).unwrap();
    }

    let onw = dt.leaf(&a.out_norm).unwrap();
    let fnorm = dt.rmsnorm(hidden, onw, seq, a.n_embd, a.eps).unwrap();
    let logits = dt.salt_matmul(fnorm, masters[0], &weights[0], seq).unwrap();
    (logits, masters)
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

/// Teacher-forced perplexity over a held-out set evaluated in **non-overlapping windows** of
/// `window` tokens — the standard windowed-perplexity protocol, and the only tractable one here.
///
/// [`logits_of`] runs the whole token slice through one autograd [`Tape`], which retains a
/// `[seq, seq]` attention matrix **per head per layer**: memory is *quadratic* in the evaluated
/// length. At SmolLM2-135M (30 layers × 9 heads) a 4096-token held-out set costs
/// `30·9·4096²·4 B ≈ 18 GB` and OOMs the machine, while 256 tokens costs 71 MB. Windowing bounds
/// the peak at `30·9·window²·4 B` (≈283 MB at `window = 512`) no matter how large the held-out set
/// is, because each window's tape is dropped before the next is built.
///
/// NLL is accumulated across windows and exponentiated once at the end, so the result is the
/// token-averaged perplexity over every scored position (each window scores its own `len-1`
/// next-token predictions). Applying the identical windowing to the fp, PTQ, and distilled models
/// keeps their comparison exact.
pub fn perplexity_windowed(weights: &[Vec<f32>], a: &Arch, eval_ids: &[u32], window: usize) -> f64 {
    assert!(window >= 2, "a window must score at least one next-token");
    let mut nll = 0.0f64;
    let mut scored = 0usize;
    for chunk in eval_ids.chunks(window) {
        if chunk.len() < 2 {
            continue; // a 1-token tail scores nothing
        }
        // Build, score, and drop one window's tape before the next allocates.
        let logits = logits_of(weights, a, chunk);
        for tpos in 0..chunk.len() - 1 {
            let row = &logits[tpos * a.vocab..tpos * a.vocab + a.vocab];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = m + row
                .iter()
                .map(|&x| (f64::from(x) - m).exp())
                .sum::<f64>()
                .ln();
            nll += lse - f64::from(row[chunk[tpos + 1] as usize]);
        }
        scored += chunk.len() - 1;
    }
    assert!(scored > 0, "held-out set scored no positions");
    (nll / scored as f64).exp()
}

/// ITF alternations used by the shared fitter helpers (the value every PTQ sweep reported with).
pub const ITERS: usize = 5;

// ── Activation-aware (AWQ-style) salience fold ──────────────────────────────────────────────────
// Shared by the PTQ sweeps and the distillation runs so both fold identically. SALT minimises
// `‖W − Ŵ‖²`; every SOTA PTQ method since GPTQ minimises `‖(W − Ŵ)X‖²`, weighting each input
// channel by how much the activations actually use it. For weight-only quantization that objective
// has an exact, inference-free implementation: scale weight column `j` by `s_j` from calibration
// and push `1/s_j` into a tensor that is already free.
/// Per-input-channel second moments for the projections whose scale is foldable.
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

/// Full input **Gram** `Σ x xᵀ` for one tap point — the curvature GPTQ actually needs.
///
/// [`Calib`] keeps only the diagonal (`Σ x²` per channel), which is all the AWQ-style salience fold
/// requires. Sequential error compensation needs the OFF-DIAGONAL terms too: when column `j` is
/// rounded, the error it induces is pushed onto the not-yet-quantized columns through `H⁻¹`, and
/// `H = Σ x xᵀ` is what says how those columns covary.
///
/// Stored as a dense lower-triangle-symmetric `k×k` in f64. At `k = 1536` that is 18.9 MB per tap
/// point, which is why this is a separate opt-in structure rather than something [`Calib`] always
/// carries.
pub struct Gram {
    pub k: usize,
    /// Row-major `k×k`, symmetric.
    pub h: Vec<f64>,
    pub rows: usize,
}

impl Gram {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            h: vec![0.0; k * k],
            rows: 0,
        }
    }

    /// Accumulate every row of a `[seq, k]` activation block.
    pub fn accumulate(&mut self, t: &Tape, id: ValueId, seq: usize) {
        let v = t.value(id);
        for r in 0..seq {
            let x = &v[r * self.k..(r + 1) * self.k];
            // Upper triangle only, mirrored at the end — halves the work on a hot O(seq·k²) loop.
            for i in 0..self.k {
                let xi = f64::from(x[i]);
                if xi == 0.0 {
                    continue;
                }
                let row = &mut self.h[i * self.k..(i + 1) * self.k];
                for (j, hij) in row.iter_mut().enumerate().skip(i) {
                    *hij += xi * f64::from(x[j]);
                }
            }
        }
        self.rows += seq;
    }

    /// Mirror the upper triangle into the lower and divide by the sample count, giving `E[x xᵀ]`.
    pub fn finish(mut self) -> Vec<f64> {
        for i in 0..self.k {
            for j in 0..i {
                self.h[i * self.k + j] = self.h[j * self.k + i];
            }
        }
        let n = self.rows.max(1) as f64;
        for v in &mut self.h {
            *v /= n;
        }
        self.h
    }

    /// The diagonal, which must agree with [`Calib`]'s cheap collector — the cross-check that says
    /// this tap sees the same activations the fold does.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.k).map(|i| self.h[i * self.k + i]).collect()
    }
}

/// Collect the input Gram at ONE tap point, named by layer and role.
///
/// Roles map to the four distinct projection inputs per block (q/k/v share one, gate/up share one):
/// `"attn"` = the attn-norm output, `"ffn"` = the ffn-norm output, `"down"` = `silu(gate)⊙up`,
/// `"o"` = the attention concat.
pub fn calibrate_gram(
    weights: &[Vec<f32>],
    a: &Arch,
    tokens: &[u32],
    layer: usize,
    role: &str,
    g: &mut Gram,
) {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    let seq = tokens.len();
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        if li == layer && role == "attn" {
            g.accumulate(&t, xn, seq);
        }
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
        if li == layer && role == "ffn" {
            g.accumulate(&t, hn, seq);
        }
        let gate = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let up = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        let ga = t.silu(gate);
        let gated = t.mul(ga, up);
        if li == layer && role == "down" {
            g.accumulate(&t, gated, seq);
        }
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        hidden = t.add(hidden, down);
    }
}

/// One calibration forward. Mirrors `common::forward` exactly, tapping the three foldable inputs.
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

pub fn scale_cols(w: &mut [f32], cols: usize, s: &[f32]) {
    for row in w.chunks_mut(cols) {
        for (v, &sj) in row.iter_mut().zip(s) {
            *v *= sj;
        }
    }
}

pub fn divide_rows(w: &mut [f32], cols: usize, s: &[f32]) {
    for (r, row) in w.chunks_mut(cols).enumerate() {
        for v in row {
            *v /= s[r];
        }
    }
}

pub fn clone_arch(a: &Arch) -> Arch {
    Arch {
        attn_norms: a.attn_norms.clone(),
        ffn_norms: a.ffn_norms.clone(),
        out_norm: a.out_norm.clone(),
        n_embd: a.n_embd,
        n_head: a.n_head,
        n_head_kv: a.n_head_kv,
        head_dim: a.head_dim,
        ff: a.ff,
        vocab: a.vocab,
        eps: a.eps,
        theta: a.theta,
        n_layers: a.n_layers,
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
    let mut arch = clone_arch(a);
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

pub fn quantize(
    w: &[Vec<f32>],
    shapes: &[(usize, usize)],
    t: usize,
    group: usize,
) -> Vec<Vec<f32>> {
    w.iter()
        .zip(shapes)
        .map(|(v, &(n, k))| {
            ste::salt_quantize_forward_grouped(v, n, k, t, group, ITERS, RotationPolicy::Auto)
        })
        .collect()
}

/// Quantize with a **separate plane count for the tied embed/head** (index 0) and for the body.
pub fn quantize_split(
    w: &[Vec<f32>],
    shapes: &[(usize, usize)],
    t_head: usize,
    t_body: usize,
    group: usize,
) -> Vec<Vec<f32>> {
    w.iter()
        .zip(shapes)
        .enumerate()
        .map(|(i, (v, &(n, k)))| {
            let t = if i == 0 { t_head } else { t_body };
            ste::salt_quantize_forward_grouped(v, n, k, t, group, ITERS, RotationPolicy::Auto)
        })
        .collect()
}

/// Parameter-weighted bpw when the tied embed/head runs at `t_head` planes and the body at `t_body`.
///
/// The two costs scale differently and must not be merged: trits plus their f16 scale are charged
/// **per plane**, but `RotationPolicy::Auto`'s choice bit is decided once per group for the whole
/// stack of planes (see `ste::salt_quantize_forward_grouped`), so it is charged **once**.
pub fn split_bpw(shapes: &[(usize, usize)], t_head: usize, t_body: usize, group: usize) -> f64 {
    let n: Vec<usize> = shapes.iter().map(|&(a, b)| a * b).collect();
    let total: usize = n.iter().sum();
    let body: usize = n[1..].iter().sum();
    let planes = t_head as f64 * n[0] as f64 + t_body as f64 * body as f64;
    ste::ternary_bits_per_weight(1, group) * planes / total as f64 + 1.0 / group as f64
}
