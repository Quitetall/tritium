//! First-order optimizers over flat `f32` parameter buffers.
//!
//! The [`Optimizer`] trait is deliberately minimal: it expresses exactly what the
//! optimizer that exists today ([`AdamW`]) needs. Shape is *not* in the signature — a
//! future 2D-aware optimizer (e.g. Muon) holds its own `rows`/`cols` as struct fields,
//! configured per parameter group, rather than threading a `Shape` through every step
//! that the scalar optimizers would ignore. The codebase is young; widening the trait
//! when a second optimizer actually arrives is a cheap, clean edit (plan 0008).
//!
//! Optimizers are tape-agnostic: they consume the grad [`Tape::backward`](crate::Tape::backward)
//! produced for a leaf and update that leaf's parameter buffer in place. Parameters
//! live outside the tape; the QAT loop rebuilds the forward graph each step, extracts
//! grads, and calls [`Optimizer::step`].

use rayon::prelude::*;

use crate::checkpoint::{CheckpointError, Cursor};

/// Parallelize a leaf's AdamW update above this element count (element-wise ⇒ bit-identical to the
/// serial loop). Set high: the optimizer is called on every leaf every step, so rayon's per-call
/// fork overhead only pays off on the very large tensors (embeddings, and every weight at 32B
/// scale) — small leaves at 135M-scale stay serial. ~1M elements.
const PAR_MIN_ELEMS: usize = 1 << 20;

/// A first-order optimizer over flat `f32` parameter buffers, with per-parameter state
/// that can be serialized into a checkpoint.
pub trait Optimizer {
    /// Per-parameter persistent state (the unit of checkpointing).
    type State: Clone;

    /// Fresh, zeroed state for a leaf of `len` elements.
    fn init_state(&self, len: usize) -> Self::State;

    /// One update step at 1-based step index `t`: read `grad`, update `state`, then
    /// write the new parameter values into `param` in place. `param`, `grad`, and the
    /// state buffers all have the same length.
    fn step(&self, t: u64, param: &mut [f32], grad: &[f32], state: &mut Self::State);

    /// Append this state's bytes (little-endian) to `out`.
    fn write_state(&self, state: &Self::State, out: &mut Vec<u8>);

    /// Read state for a leaf of `len` elements from `cursor`.
    ///
    /// # Errors
    /// [`CheckpointError`] if the cursor is truncated.
    fn read_state(&self, len: usize, cursor: &mut Cursor) -> Result<Self::State, CheckpointError>;
}

/// Decoupled-weight-decay Adam (Loshchilov & Hutter, 2019). Weight decay is applied to
/// the parameter directly (`w·(1 − lr·wd)`), never folded into the moment buffers, and
/// `eps` sits outside the square root (`√v̂ + eps`) — matching `torch.optim.AdamW`.
#[derive(Clone, Copy, Debug)]
pub struct AdamW {
    /// Learning rate.
    pub lr: f32,
    /// First-moment (mean) decay `β₁`.
    pub beta1: f32,
    /// Second-moment (variance) decay `β₂`.
    pub beta2: f32,
    /// Numerical floor added to `√v̂` in the denominator.
    pub eps: f32,
    /// Decoupled weight-decay coefficient.
    pub weight_decay: f32,
}

impl AdamW {
    /// `torch.optim.AdamW` defaults — `β = (0.9, 0.999)`, `eps = 1e-8`, `wd = 0.01` — at
    /// learning rate `lr`.
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW per-parameter state: first moment `m` and second moment `v`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamState {
    /// First-moment (mean of gradients) EMA, one entry per parameter.
    pub m: Vec<f32>,
    /// Second-moment (mean of squared gradients) EMA, one entry per parameter.
    pub v: Vec<f32>,
}

impl Optimizer for AdamW {
    type State = AdamState;

    fn init_state(&self, len: usize) -> AdamState {
        AdamState {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    fn step(&self, t: u64, param: &mut [f32], grad: &[f32], state: &mut AdamState) {
        debug_assert!(t >= 1, "step index t is 1-based; t=0 gives bc=0 ⇒ NaN");
        assert_eq!(param.len(), grad.len(), "param/grad length mismatch");
        assert_eq!(param.len(), state.m.len(), "param/state m length mismatch");
        assert_eq!(param.len(), state.v.len(), "param/state v length mismatch");
        // Bias-correction denominators 1 − βᵗ at the 1-based step t. For t > i32::MAX
        // the exponent saturates, which is harmless: βᵗ has already underflowed to ~0
        // there, so 1 − βᵗ → 1 (the correct limit).
        let exp = i32::try_from(t).unwrap_or(i32::MAX);
        let bc1 = 1.0 - self.beta1.powi(exp);
        let bc2 = 1.0 - self.beta2.powi(exp);
        // Decoupled weight decay: a multiplicative shrink on the param, applied outside
        // the adaptive denominator (the "W" in AdamW).
        let shrink = 1.0 - self.lr * self.weight_decay;
        // Each element updates independently, so the parallel and serial results are bit-identical.
        // Big leaves (every trained 2D weight) go parallel; tiny ones stay serial (rayon overhead).
        let elem = |p: &mut f32, g: f32, mi: &mut f32, vi: &mut f32| {
            let m = self.beta1 * *mi + (1.0 - self.beta1) * g;
            let v = self.beta2 * *vi + (1.0 - self.beta2) * g * g;
            *mi = m;
            *vi = v;
            *p = *p * shrink - self.lr * (m / bc1 / ((v / bc2).sqrt() + self.eps));
        };
        let (m, v) = (&mut state.m, &mut state.v);
        if param.len() >= PAR_MIN_ELEMS {
            param
                .par_iter_mut()
                .zip(grad.par_iter())
                .zip(m.par_iter_mut())
                .zip(v.par_iter_mut())
                .for_each(|(((p, &g), mi), vi)| elem(p, g, mi, vi));
        } else {
            for (((p, &g), mi), vi) in param
                .iter_mut()
                .zip(grad)
                .zip(m.iter_mut())
                .zip(v.iter_mut())
            {
                elem(p, g, mi, vi);
            }
        }
    }

    fn write_state(&self, state: &AdamState, out: &mut Vec<u8>) {
        for &x in &state.m {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &state.v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn read_state(&self, len: usize, cursor: &mut Cursor) -> Result<AdamState, CheckpointError> {
        let m = cursor.f32_vec(len)?;
        let v = cursor.f32_vec(len)?;
        Ok(AdamState { m, v })
    }
}
