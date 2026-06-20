//! Reverse-mode autograd tape: a flat list of ops over an `f32` value arena.
//!
//! Each op runs its forward eagerly (storing the output in the arena) and records a
//! backward closure `(input_values, grad_output) -> grad_per_input`. [`Tape::backward`]
//! seeds the scalar loss with cotangent `1`, walks the recorded ops in reverse, and
//! **accumulates** each input's grad — so a value feeding more than one consumer sums
//! its incoming grads, exactly as the chain rule requires.
//!
//! The quantizer slot uses the STE *surrogate* ([`crate::ops::ste::quantize_surrogate`]),
//! not the rounded QAT forward: `round` is piecewise-constant and not finite-difference-
//! checkable, while its straight-through backward is the surrogate's exact gradient (see
//! the `ste` module). The graph this tape differentiates is therefore the smooth
//! surrogate model; `round` is a forward-only QAT detail layered on later (ADR 0007).

use crate::ops::{act, bias, dense, elementwise, loss, matmul, ste};

/// Index of a value buffer in a [`Tape`]'s arena.
pub type ValueId = usize;

type Backward = Box<dyn Fn(&[&[f32]], &[f32]) -> Vec<Vec<f32>>>;

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
    /// An empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
        for node in self.nodes.iter().rev() {
            let input_vals: Vec<&[f32]> = node
                .inputs
                .iter()
                .map(|&i| self.values[i].as_slice())
                .collect();
            let g_out = grads[node.output].clone();
            let g_ins = (node.backward)(&input_vals, &g_out);
            for (k, &in_id) in node.inputs.iter().enumerate() {
                for (j, &gv) in g_ins[k].iter().enumerate() {
                    grads[in_id][j] += gv;
                }
            }
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
            Box::new(move |ins, g| ste::quantize_vjp(ins[0], ins[1], rows, cols, g)),
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
        let out = dense::forward(&self.values[x], &self.values[w], m, n, k);
        self.record(
            vec![x, w],
            out,
            Box::new(move |ins, g| dense::vjp(ins[0], ins[1], m, n, k, g)),
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
            Box::new(|ins, _g| vec![vec![0.0; ins[0].len()]]),
        )
    }

    /// Multiply by a compile-time-constant scalar `c`: `Y = c·X`, `vjp = c·g`.
    pub fn scale_const(&mut self, x: ValueId, c: f32) -> ValueId {
        let out: Vec<f32> = self.values[x].iter().map(|&v| v * c).collect();
        self.record(
            vec![x],
            out,
            Box::new(move |_ins, g| vec![g.iter().map(|&gv| gv * c).collect()]),
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
            Box::new(move |ins, g| matmul::vjp(ins[0], ins[1], ins[2], m, n, k, g)),
        )
    }

    /// Bias add `Y[m,n] = X[m,n] + b[n]`.
    pub fn bias(&mut self, x: ValueId, b: ValueId, rows: usize, cols: usize) -> ValueId {
        let out = bias::forward(&self.values[x], &self.values[b], rows, cols);
        self.record(
            vec![x, b],
            out,
            Box::new(move |ins, g| bias::vjp(ins[0], ins[1], rows, cols, g)),
        )
    }

    /// Squared-ReLU activation.
    pub fn relu2(&mut self, x: ValueId) -> ValueId {
        let out = act::relu2_forward(&self.values[x]);
        self.record(vec![x], out, Box::new(|ins, g| act::relu2_vjp(ins[0], g)))
    }

    /// Element-wise add `Y = A + B`.
    pub fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let out = elementwise::add_forward(&self.values[a], &self.values[b]);
        self.record(
            vec![a, b],
            out,
            Box::new(|ins, g| elementwise::add_vjp(ins[0], ins[1], g)),
        )
    }

    /// Element-wise multiply `Y = A ⊙ B`.
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let out = elementwise::mul_forward(&self.values[a], &self.values[b]);
        self.record(
            vec![a, b],
            out,
            Box::new(|ins, g| elementwise::mul_vjp(ins[0], ins[1], g)),
        )
    }

    /// Mean-squared-error loss against a constant `target` (scalar output).
    pub fn mse(&mut self, pred: ValueId, target: ValueId) -> ValueId {
        let out = loss::mse_forward(&self.values[pred], &self.values[target]);
        self.record(
            vec![pred, target],
            out,
            Box::new(|ins, g| {
                let gp = loss::mse_vjp(ins[0], ins[1], g);
                // target is data: its grad slot stays zero.
                vec![
                    gp.into_iter().next().unwrap_or_default(),
                    vec![0.0; ins[1].len()],
                ]
            }),
        )
    }
}
