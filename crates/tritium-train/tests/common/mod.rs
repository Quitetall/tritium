//! Shared toy-training scaffolding for the optimizer + checkpoint integration tests.
//!
//! A small, smooth, linear-in-the-band model so convergence isolates the *optimizer*,
//! not STE saturation: `z = bias(matmul(act, surrogate(Wf, s_q), s))`, with `s_q` set
//! large enough that `|Wf/s_q| < 1` everywhere (the surrogate's linear region, all
//! grads flow). The loss is MSE to a target.

// Shared across test binaries via `mod common;`, so not every helper is used by each
// (dead_code) and `pub` items look unreachable from a single binary (unreachable_pub).
#![allow(dead_code, unreachable_pub)]

use tritium_train::optim::{AdamState, AdamW, Optimizer};
use tritium_train::tape::Tape;

/// Toy layer dims: act `[M,K]`, Wf `[N,K]`, s/s_q/b `[N]`, output/target `[M,N]`.
pub const M: usize = 2;
pub const N: usize = 3;
pub const K: usize = 4;

/// Deterministic xorshift64 fixture in `[lo, hi)`.
#[must_use]
pub fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

/// Forward to the `[M,N]` prediction vector.
#[must_use]
pub fn forward(wf: &[f32], s_q: &[f32], act: &[f32], s: &[f32], b: &[f32]) -> Vec<f32> {
    use tritium_train::ops::{bias, matmul, ste};
    let t = ste::quantize_surrogate(wf, s_q, N, K);
    let y = matmul::forward(act, &t, s, M, N, K);
    bias::forward(&y, b, M, N)
}

/// Scalar MSE loss of the forward against `target`.
#[must_use]
pub fn forward_loss(
    wf: &[f32],
    s_q: &[f32],
    act: &[f32],
    s: &[f32],
    b: &[f32],
    target: &[f32],
) -> f32 {
    use tritium_train::ops::loss;
    loss::mse_forward(&forward(wf, s_q, act, s, b), target)[0]
}

/// The frozen (non-trained) inputs of the toy problem: the quant scale `s_q`, the
/// activations, the matmul scale `s`, and the regression target.
#[derive(Clone, Debug)]
pub struct ToyData {
    pub s_q: Vec<f32>,
    pub act: Vec<f32>,
    pub s: Vec<f32>,
    pub target: Vec<f32>,
}

impl ToyData {
    /// Scalar loss of `p` against this data.
    #[must_use]
    pub fn loss(&self, p: &ToyParams) -> f32 {
        forward_loss(&p.wf, &self.s_q, &self.act, &self.s, &p.b, &self.target)
    }
}

/// Trainable parameters of the toy layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ToyParams {
    pub wf: Vec<f32>,
    pub b: Vec<f32>,
}

/// AdamW state for the two toy parameter groups.
#[derive(Clone, Debug, PartialEq)]
pub struct ToyState {
    pub wf: AdamState,
    pub b: AdamState,
}

impl ToyState {
    /// Fresh zeroed state sized to a [`ToyParams`].
    #[must_use]
    pub fn init(opt: &AdamW) -> Self {
        Self {
            wf: opt.init_state(N * K),
            b: opt.init_state(N),
        }
    }
}

/// One AdamW step over `(wf, b)` at 1-based step `t`. Rebuilds the tape, backprops, and
/// updates both parameter groups in place. Returns the scalar loss *before* the step.
pub fn train_step(t: u64, opt: &AdamW, p: &mut ToyParams, st: &mut ToyState, d: &ToyData) -> f32 {
    let mut tape = Tape::new();
    let wf_id = tape.leaf(p.wf.clone());
    let sq_id = tape.leaf(d.s_q.clone());
    let act_id = tape.leaf(d.act.clone());
    let s_id = tape.leaf(d.s.clone());
    let b_id = tape.leaf(p.b.clone());
    let tg_id = tape.leaf(d.target.clone());

    let tq = tape.ste_surrogate(wf_id, sq_id, N, K);
    let y = tape.matmul(act_id, tq, s_id, M, N, K);
    let z = tape.bias(y, b_id, M, N);
    let l = tape.mse(z, tg_id);

    let loss = tape.value(l)[0];
    let grads = tape.backward(l);
    opt.step(t, &mut p.wf, &grads[wf_id], &mut st.wf);
    opt.step(t, &mut p.b, &grads[b_id], &mut st.b);
    loss
}

/// A frozen `s_q` large enough to keep the surrogate in its linear region for weights
/// in `[-1.x, 1.x]` (so the toy problem is smooth and [`forward`] is exact-linear).
#[must_use]
pub fn linear_region_s_q() -> Vec<f32> {
    vec![2.0; N]
}
