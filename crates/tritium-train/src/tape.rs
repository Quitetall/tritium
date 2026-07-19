//! Reverse-mode autograd tape: a flat list of ops over an `f32` value arena.
//!
//! Each op runs its forward eagerly (storing the output in the arena) and records a
//! backward closure that accumulates gradients directly into the per-value grad
//! buffers. [`Tape::backward`] seeds the scalar loss with cotangent `1`, walks the
//! recorded ops in reverse, and each closure adds its contribution to the input
//! grads — so a value feeding more than one consumer sums its incoming grads,
//! exactly as the chain rule requires.
//!
//! The quantizer slot uses the STE *surrogate* ([`crate::ops::ste::quantize_surrogate`]),
//! not the rounded QAT forward: `round` is piecewise-constant and not finite-difference-
//! checkable, while its straight-through backward is the surrogate's exact gradient (see
//! the `ste` module). The graph this tape differentiates is therefore the smooth
//! surrogate model; `round` is a forward-only QAT detail layered on later (ADR 0007).

use std::rc::Rc;

use crate::gemm::TrainGemm;
pub use crate::ops::conv1d::Conv1dCfg;
pub use crate::ops::fsq::{FsqBound, FsqCfg, FsqSte};
use crate::ops::{
    act, bias, conv1d, dense, elementwise, embed, fsq, loss, matmul, norm, rope, shape, softmax,
    ste,
};

/// Index of a value buffer in a [`Tape`]'s arena.
pub type ValueId = usize;

/// Backward closure: receives (input_slices, grad_output, grads_array, input_ids)
/// and accumulates gradients directly into `grads[input_id]` for each input.
/// No return value — zero intermediate allocations.
type Backward = Box<dyn Fn(&[&[f32]], &[f32], &mut [Vec<f32>], &[ValueId])>;

struct Node {
    inputs: Vec<ValueId>,
    output: ValueId,
    backward: Backward,
}

/// A recording autograd tape. Register inputs with [`Tape::leaf`], compose with the op
/// methods, then call [`Tape::backward`] on the scalar loss id to get per-leaf grads.
///
/// Contract: every [`ValueId`] passed to an op method must have been minted by *this*
/// tape, and the `rows`/`cols`/`m`/`n`/`k` dims must match the buffers behind those ids.
/// Violating either panics (out-of-bounds index) — ids are plain `usize`, not checked
/// across tapes.
#[derive(Default)]
pub struct Tape {
    values: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    /// Optional device GEMM engine (plan 0043). `None` ⇒ the built-in CPU matmul path (bit-for-bit
    /// the pre-0043 behaviour). `Some` ⇒ `dense_matmul`/`matmul` dispatch to the engine (e.g. GPU).
    gemm: Option<Rc<dyn TrainGemm>>,
}

impl core::fmt::Debug for Tape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape")
            .field("values", &self.values.len())
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

impl Tape {
    /// An empty tape (CPU matmuls).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty tape whose `dense_matmul`/`matmul` run on the given engine (e.g. a GPU backend)
    /// instead of the built-in CPU path — the training hot path offloaded to the device (plan 0043).
    #[must_use]
    pub fn with_gemm(gemm: Rc<dyn TrainGemm>) -> Self {
        Self {
            gemm: Some(gemm),
            ..Self::default()
        }
    }

    /// Register a leaf (parameter or input) buffer; returns its [`ValueId`].
    pub fn leaf(&mut self, v: Vec<f32>) -> ValueId {
        let id = self.values.len();
        self.values.push(v);
        id
    }

    /// Borrow a value buffer by id.
    #[must_use]
    pub fn value(&self, id: ValueId) -> &[f32] {
        &self.values[id]
    }

    /// Store an op output and record its inputs + backward closure; returns the output id.
    fn record(&mut self, inputs: Vec<ValueId>, output: Vec<f32>, backward: Backward) -> ValueId {
        let out_id = self.values.len();
        self.values.push(output);
        self.nodes.push(Node {
            inputs,
            output: out_id,
            backward,
        });
        out_id
    }

    /// Reverse-accumulate grads from the scalar `loss`. Returns one grad buffer per value
    /// id (same shapes as the arena); untouched leaves stay zero.
    #[must_use]
    pub fn backward(&self, loss: ValueId) -> Vec<Vec<f32>> {
        let mut grads: Vec<Vec<f32>> = self.values.iter().map(|v| vec![0.0f32; v.len()]).collect();
        // seed dLoss/dLoss = 1 over the (scalar) loss buffer.
        for g in &mut grads[loss] {
            *g = 1.0;
        }
        // Reusable buffer for input slice references (avoids 270 allocs).
        let mut input_buf: Vec<&[f32]> = Vec::new();
        for node in self.nodes.iter().rev() {
            input_buf.clear();
            input_buf.extend(node.inputs.iter().map(|&i| self.values[i].as_slice()));
            // Split grads at node.output so we can immutably borrow g_out while
            // mutably borrowing the rest. Since inputs are always < output (ops
            // append to the arena), grads_lo contains all input grad slots.
            let (grads_lo, grads_hi) = grads.split_at_mut(node.output);
            let g_out = &grads_hi[0]; // grads[node.output]
            (node.backward)(&input_buf, g_out, grads_lo, &node.inputs);
        }
        grads
    }

    // ── op-recording methods (forward eager + captured backward) ───────────────

    /// STE surrogate `clamp(Wf/s_q, -1, 1)` with straight-through backward.
    pub fn ste_surrogate(
        &mut self,
        wf: ValueId,
        s_q: ValueId,
        rows: usize,
        cols: usize,
    ) -> ValueId {
        let out = ste::quantize_surrogate(&self.values[wf], &self.values[s_q], rows, cols);
        self.record(
            vec![wf, s_q],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = ste::quantize_vjp(ins[0], ins[1], rows, cols, g);
                for (k, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k]][j] += v;
                    }
                }
            }),
        )
    }

    /// Multi-plane **SALT** STE: forward is the `t`-plane residual quantize (the dense
    /// reconstruction `Ŵ`, `[rows, cols]`), backward is straight-through to the latent. Use the
    /// output as a dense weight (e.g. via [`dense_matmul`](Self::dense_matmul)) for the student.
    pub fn salt_ste(&mut self, wf: ValueId, rows: usize, cols: usize, t: usize) -> ValueId {
        let out = ste::salt_quantize_forward(&self.values[wf], rows, cols, t);
        self.record(
            vec![wf],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gwf = ste::salt_quantize_vjp(ins[0], rows, cols, t, g);
                for (j, &v) in gwf.iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// LSQ STE: the ternary reconstruction `round(clamp(Wf/α))·α` with a **learned** per-row step size
    /// `α` (`[rows]`), replacing the fixed AbsMean scale. Both `wf` and `alpha` are trainable — the
    /// straight-through weight grad routes to `wf`, the LSQ step-size estimator to `alpha`. Use the
    /// output as a dense weight (e.g. via [`conv1d`](Self::conv1d) or [`dense_matmul`](Self::dense_matmul)).
    pub fn lsq_ste(&mut self, wf: ValueId, alpha: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = ste::lsq_forward(&self.values[wf], &self.values[alpha], rows, cols);
        self.record(
            vec![wf, alpha],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = ste::lsq_vjp(ins[0], ins[1], rows, cols, g);
                for (k, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k]][j] += v;
                    }
                }
            }),
        )
    }

    /// Plain dense matmul `Y[m,n] = Σ_k X[m,k]·W[n,k]` (no scale, real `f32`).
    pub fn dense_matmul(
        &mut self,
        x: ValueId,
        w: ValueId,
        m: usize,
        n: usize,
        k: usize,
    ) -> ValueId {
        // A device engine (e.g. GPU) runs the fp matmul fwd/bwd on-device; the CPU default (no engine)
        // keeps calling `dense::{forward,vjp}` verbatim, so every existing gate is bit-for-bit unchanged.
        let out = match &self.gemm {
            Some(eng) => eng.dense_forward(&self.values[x], &self.values[w], m, n, k),
            None => dense::forward(&self.values[x], &self.values[w], m, n, k),
        };
        let gemm = self.gemm.clone();
        self.record(
            vec![x, w],
            out,
            Box::new(move |ins, g, grads, ids| match &gemm {
                Some(eng) => {
                    let (ga, gw) = eng.dense_backward(g, ins[0], ins[1], m, n, k);
                    for (j, &v) in ga.iter().enumerate() {
                        grads[ids[0]][j] += v;
                    }
                    for (j, &v) in gw.iter().enumerate() {
                        grads[ids[1]][j] += v;
                    }
                }
                None => {
                    let gs = dense::vjp(ins[0], ins[1], m, n, k, g);
                    for (k_idx, gv) in gs.into_iter().enumerate() {
                        for (j, &v) in gv.iter().enumerate() {
                            grads[ids[k_idx]][j] += v;
                        }
                    }
                }
            }),
        )
    }

    /// Transpose `[rows, cols] → [cols, rows]` (for attention's `P·V`).
    pub fn transpose(&mut self, x: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = dense::transpose_forward(&self.values[x], rows, cols);
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gs = dense::transpose_vjp(rows, cols, g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Gather `[seq, n_embd]` embedding rows from `weight [vocab, n_embd]` by token id — the
    /// differentiable input embedding (grad scatters back to the rows read; tied heads train it).
    pub fn embed_gather(
        &mut self,
        weight: ValueId,
        tokens: &[u32],
        vocab: usize,
        n_embd: usize,
    ) -> ValueId {
        let out = embed::gather_forward(&self.values[weight], tokens, n_embd);
        let tokens = tokens.to_vec();
        self.record(
            vec![weight],
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gw = embed::gather_vjp(vocab, &tokens, n_embd, g);
                for (j, &v) in gw.iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Extract a contiguous column range `[rows, len]` (cols `start..start+len`) from a
    /// `[rows, cols]` matrix. The multi-head/GQA reshape that splits a projection into per-head
    /// slices (plan 0040).
    pub fn slice_cols(
        &mut self,
        x: ValueId,
        rows: usize,
        cols: usize,
        start: usize,
        len: usize,
    ) -> ValueId {
        let out = shape::slice_cols_forward(&self.values[x], rows, cols, start, len);
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gx = shape::slice_cols_vjp(rows, cols, start, len, g);
                for (j, &v) in gx.iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Concatenate `parts` (each `[rows, lens[i]]`) along columns → `[rows, Σ lens]`. Reassembles
    /// per-head attention outputs (plan 0040). `parts.len()` must equal `lens.len()`.
    pub fn concat_cols(&mut self, parts: &[ValueId], rows: usize, lens: &[usize]) -> ValueId {
        let bufs: Vec<&[f32]> = parts.iter().map(|&p| self.values[p].as_slice()).collect();
        let out = shape::concat_cols_forward(&bufs, rows, lens);
        let lens = lens.to_vec();
        self.record(
            parts.to_vec(),
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gs = shape::concat_cols_vjp(rows, &lens, g);
                for (i, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[i]][j] += v;
                    }
                }
            }),
        )
    }

    /// Stop-gradient: forwards `x` unchanged, but blocks the backward pass (its input
    /// receives zero gradient). This is how a frozen base is held constant — the leaves
    /// behind a `detach` train nothing.
    pub fn detach(&mut self, x: ValueId) -> ValueId {
        let out = self.values[x].clone();
        self.record(
            vec![x],
            out,
            Box::new(|_ins, _g, _grads, _ids| {
                // Zero gradient — nothing to accumulate.
            }),
        )
    }

    /// Multiply by a compile-time-constant scalar `c`: `Y = c·X`, `vjp = c·g`.
    pub fn scale_const(&mut self, x: ValueId, c: f32) -> ValueId {
        let out: Vec<f32> = self.values[x].iter().map(|&v| v * c).collect();
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g, grads, ids| {
                for (j, &gv) in g.iter().enumerate() {
                    grads[ids[0]][j] += gv * c;
                }
            }),
        )
    }

    /// Ternary matmul `Y = s · (A · Tᵀ)`.
    pub fn matmul(
        &mut self,
        act_: ValueId,
        trits: ValueId,
        scale: ValueId,
        m: usize,
        n: usize,
        k: usize,
    ) -> ValueId {
        let out = matmul::forward(
            &self.values[act_],
            &self.values[trits],
            &self.values[scale],
            m,
            n,
            k,
        );
        self.record(
            vec![act_, trits, scale],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = matmul::vjp(ins[0], ins[1], ins[2], m, n, k, g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// Ternary 1-D convolution `Y = s ⊙ conv1d(X, W)` (im2col → ternary matmul → col2im).
    ///
    /// `x` is `[batch, C_in, L_in]`, `w` is the flattened `[C_out, (C_in/groups)·K]` weight, `scale`
    /// is `[C_out]`. Differentiable in `x`, `w`, and `scale`. Compose the ternary student exactly as
    /// the linear path does — `s_q = leaf(absmean_scale_per_row(wf, C_out, K_g))`,
    /// `w_surr = ste_surrogate(wf, s_q, C_out, K_g)`, `conv1d(x, w_surr, s_q, cfg)` — so the STE
    /// weight grad and the per-output-channel scale fold match [`matmul`](Self::matmul).
    pub fn conv1d(&mut self, x: ValueId, w: ValueId, scale: ValueId, cfg: Conv1dCfg) -> ValueId {
        let out = conv1d::forward(&self.values[x], &self.values[w], &self.values[scale], &cfg);
        self.record(
            vec![x, w, scale],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = conv1d::vjp(ins[0], ins[1], ins[2], &cfg, g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// Finite scalar quantization: round `x` `[channels, len]` to each channel's `L`-level grid, with a
    /// straight-through backward. The stored value is the rounded (QAT/deploy) activation — this is the
    /// trainable activation-quant node — while the backward is the exact gradient of the chosen
    /// surrogate (`Hard`/`Stochastic` → `bound'`; `SoftRound` → annealed). The rate knob for the codec
    /// latent.
    pub fn fsq(&mut self, x: ValueId, cfg: FsqCfg, ste: FsqSte) -> ValueId {
        let out = fsq::forward(&self.values[x], &cfg, ste);
        self.record(
            vec![x],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = fsq::vjp(ins[0], &cfg, ste, g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Bias add `Y[m,n] = X[m,n] + b[n]`.
    pub fn bias(&mut self, x: ValueId, b: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = bias::forward(&self.values[x], &self.values[b], rows, cols);
        self.record(
            vec![x, b],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = bias::vjp(ins[0], ins[1], rows, cols, g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// Squared-ReLU activation.
    pub fn relu2(&mut self, x: ValueId) -> ValueId {
        let out = act::relu2_forward(&self.values[x]);
        self.record(
            vec![x],
            out,
            Box::new(|ins, g, grads, ids| {
                let gs = act::relu2_vjp(ins[0], g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// SiLU / swish activation `Y = x·σ(x)` — the SwiGLU gate.
    pub fn silu(&mut self, x: ValueId) -> ValueId {
        let out = act::silu_forward(&self.values[x]);
        self.record(
            vec![x],
            out,
            Box::new(|ins, g, grads, ids| {
                let gs = act::silu_vjp(ins[0], g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Element-wise add `Y = A + B`.
    pub fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let out = elementwise::add_forward(&self.values[a], &self.values[b]);
        self.record(
            vec![a, b],
            out,
            Box::new(|ins, g, grads, ids| {
                let gs = elementwise::add_vjp(ins[0], ins[1], g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// Element-wise multiply `Y = A ⊙ B`.
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let out = elementwise::mul_forward(&self.values[a], &self.values[b]);
        self.record(
            vec![a, b],
            out,
            Box::new(|ins, g, grads, ids| {
                let gs = elementwise::mul_vjp(ins[0], ins[1], g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// RMSNorm `y[r,i] = x[r,i]·inv_r·w[i]` (per-row, weight shared across rows).
    pub fn rmsnorm(
        &mut self,
        x: ValueId,
        w: ValueId,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> ValueId {
        let out = norm::forward(&self.values[x], &self.values[w], rows, cols, eps);
        self.record(
            vec![x, w],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = norm::vjp(ins[0], ins[1], rows, cols, eps, g);
                for (k_idx, gv) in gs.into_iter().enumerate() {
                    for (j, &v) in gv.iter().enumerate() {
                        grads[ids[k_idx]][j] += v;
                    }
                }
            }),
        )
    }

    /// Row-wise softmax `[rows, cols] → [rows, cols]` (attention probabilities).
    pub fn softmax(&mut self, x: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = softmax::forward(&self.values[x], rows, cols);
        self.record(
            vec![x],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gs = softmax::vjp(ins[0], rows, cols, g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Additive causal mask over `[rows=queries, cols=keys]` scores (`j <= i` visible).
    pub fn causal_mask(&mut self, x: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = softmax::causal_mask_forward(&self.values[x], rows, cols);
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gs = softmax::causal_mask_vjp(rows, cols, g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// RoPE over a `[n_token, n_head, head_dim]` flat buffer (orthogonal rotation;
    /// `positions`/`theta` are data, only `x` is differentiated).
    pub fn rope(
        &mut self,
        x: ValueId,
        positions: Vec<usize>,
        n_head: usize,
        head_dim: usize,
        theta: f32,
    ) -> ValueId {
        let out = rope::forward(&self.values[x], &positions, n_head, head_dim, theta);
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g, grads, ids| {
                let gs = rope::vjp(&positions, n_head, head_dim, theta, g);
                for (j, &v) in gs[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Mean-squared-error loss against a constant `target` (scalar output).
    pub fn mse(&mut self, pred: ValueId, target: ValueId) -> ValueId {
        let out = loss::mse_forward(&self.values[pred], &self.values[target]);
        self.record(
            vec![pred, target],
            out,
            Box::new(|ins, g, grads, ids| {
                let gp = loss::mse_vjp(ins[0], ins[1], g);
                // Accumulate pred gradient; target is data (zero grad).
                for (j, &v) in gp[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }

    /// Softmax cross-entropy loss (scalar output).
    pub fn softmax_xent(
        &mut self,
        logits: ValueId,
        target: ValueId,
        rows: usize,
        cols: usize,
    ) -> ValueId {
        let out =
            loss::softmax_xent_forward(&self.values[logits], &self.values[target], rows, cols);
        self.record(
            vec![logits, target],
            out,
            Box::new(move |ins, g, grads, ids| {
                let gp = loss::softmax_xent_vjp(ins[0], ins[1], rows, cols, g);
                // Accumulate logits gradient; target is data (zero grad).
                for (j, &v) in gp[0].iter().enumerate() {
                    grads[ids[0]][j] += v;
                }
            }),
        )
    }
}
