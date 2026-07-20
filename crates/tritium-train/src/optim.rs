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
use crate::ops::dense;

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

/// Stateless plain stochastic gradient descent.
///
/// This portable reference intentionally excludes momentum: momentum SGD is a
/// distinct future optimizer identity rather than an ambiguous stateful mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f32,
}

impl Sgd {
    /// Plain SGD at learning rate `lr`, without weight decay.
    #[must_use]
    pub const fn new(lr: f32) -> Self {
        Self { lr }
    }
}

/// Zero-sized persistent state for stateless [`Sgd`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SgdState;

impl Optimizer for Sgd {
    type State = SgdState;

    fn init_state(&self, _len: usize) -> Self::State {
        SgdState
    }

    fn step(&self, t: u64, param: &mut [f32], grad: &[f32], _state: &mut Self::State) {
        debug_assert!(t >= 1, "step index t is 1-based");
        assert_eq!(param.len(), grad.len(), "param/grad length mismatch");
        for (parameter, &gradient) in param.iter_mut().zip(grad) {
            *parameter -= self.lr * gradient;
        }
    }

    fn write_state(&self, _state: &Self::State, _out: &mut Vec<u8>) {}

    fn read_state(
        &self,
        _len: usize,
        _cursor: &mut Cursor,
    ) -> Result<Self::State, CheckpointError> {
        Ok(SgdState)
    }
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

/// Cautious AdamW (Liang et al. 2024, "Cautious Optimizers"): AdamW whose adaptive update is applied
/// **only where it agrees with the current gradient sign**, rescaled by `n/(aligned+1)` so the average
/// step magnitude is preserved; the decoupled weight decay is unchanged. Elements where the momentum-
/// driven update opposes the gradient are held (they still get the WD shrink). Exposed as a distinct
/// type wrapping [`AdamW`] — same [`AdamState`] and serialization — so **no existing AdamW call site
/// changes** (ADR 0030 Tier 1: an optimizer hook LamQuant can toggle; STE stays optimizer-agnostic).
///
/// Serial by construction: the alignment count is a per-parameter reduction. It is opt-in, so the
/// standard [`AdamW`] hot path (and its parallelism) is untouched.
#[derive(Clone, Copy, Debug)]
pub struct CautiousAdamW(pub AdamW);

impl CautiousAdamW {
    /// Cautious AdamW at learning rate `lr` with `torch.optim.AdamW` defaults.
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self(AdamW::new(lr))
    }
}

impl Optimizer for CautiousAdamW {
    type State = AdamState;

    fn init_state(&self, len: usize) -> AdamState {
        self.0.init_state(len)
    }

    fn step(&self, t: u64, param: &mut [f32], grad: &[f32], state: &mut AdamState) {
        debug_assert!(t >= 1, "step index t is 1-based");
        assert_eq!(param.len(), grad.len(), "param/grad length mismatch");
        assert_eq!(param.len(), state.m.len(), "param/state m length mismatch");
        assert_eq!(param.len(), state.v.len(), "param/state v length mismatch");
        let a = &self.0;
        let exp = i32::try_from(t).unwrap_or(i32::MAX);
        let bc1 = 1.0 - a.beta1.powi(exp);
        let bc2 = 1.0 - a.beta2.powi(exp);
        let shrink = 1.0 - a.lr * a.weight_decay;
        let n = param.len();
        // Pass 1: update the moments, form the masked adaptive update, count aligned elements.
        let mut upd = vec![0.0f32; n];
        let mut aligned = 0usize;
        for i in 0..n {
            let g = grad[i];
            let mi = a.beta1 * state.m[i] + (1.0 - a.beta1) * g;
            let vi = a.beta2 * state.v[i] + (1.0 - a.beta2) * g * g;
            state.m[i] = mi;
            state.v[i] = vi;
            let u = (mi / bc1) / ((vi / bc2).sqrt() + a.eps);
            if u * g > 0.0 {
                upd[i] = u;
                aligned += 1;
            }
        }
        // Pass 2: rescale to preserve the step magnitude, apply decoupled WD to every element.
        let rescale = n as f32 / (aligned as f32 + 1.0);
        for i in 0..n {
            param[i] = param[i] * shrink - a.lr * upd[i] * rescale;
        }
    }

    fn write_state(&self, state: &AdamState, out: &mut Vec<u8>) {
        self.0.write_state(state, out);
    }

    fn read_state(&self, len: usize, cursor: &mut Cursor) -> Result<AdamState, CheckpointError> {
        self.0.read_state(len, cursor)
    }
}

/// Block size for block-wise int8 moment quantization (Dettmers et al. 2022, "8-bit Optimizers via
/// Block-wise Quantization"): each contiguous run of this many elements shares one f32 absmax scale,
/// so a spike in one block never inflates another's quantization grid. 256 matches the bitsandbytes
/// default and keeps a block's requantization reduction inside one CUDA block for the device mirror.
pub const INT8_ADAM_BLOCK: usize = 256;

/// [`Int8AdamW`] per-parameter state: the two AdamW moments stored **block-wise int8** rather than
/// f32 — a 4× shrink of the optimizer state, which is the dominant resident VRAM cost at ≥1.7B
/// (`m`+`v` in f32 is ~2× the model itself). `m` is signed (`i8`, symmetric absmax); `v ≥ 0` so it
/// uses the full unsigned range (`u8`). Each block carries its own scale; every step dequantizes to
/// f32, runs the exact AdamW update, and requantizes with a fresh per-block absmax.
#[derive(Clone, Debug, PartialEq)]
pub struct Int8AdamState {
    /// First-moment EMA, signed int8 (dequantize `m = m_q · m_scale[block]`).
    pub m_q: Vec<i8>,
    /// Second moment stored in **sqrt-space**, unsigned int8: `√v = v_q · v_scale[block]`, so
    /// `v = (v_q · v_scale)²`. Quantizing `√v` (the quantity the denominator actually uses) halves
    /// the second moment's exponent range, so a small `v` no longer underflows to 0 while `m` keeps
    /// its history — the asymmetry that makes a naive linear-`v` quantizer produce exploding steps.
    pub v_q: Vec<u8>,
    /// Per-block absmax scale for `m` (`len = ceil(n / INT8_ADAM_BLOCK)`).
    pub m_scale: Vec<f32>,
    /// Per-block absmax scale for `√v` (sqrt-space; dequantize `v = (v_q · v_scale)²`).
    pub v_scale: Vec<f32>,
    /// Parameter element count (the final block may be shorter than `INT8_ADAM_BLOCK`).
    pub len: usize,
}

/// AdamW with **block-wise int8 moment state** (Dettmers et al. 2022). Numerically identical to
/// [`AdamW`] except the two moments are held quantized between steps: dequantize → exact f32 update →
/// requantize, block by block. The parameter buffer itself stays f32 (bf16-master storage is a
/// separate, device-side lever). Exposed as its own type over the same [`AdamW`] config so no
/// existing call site changes; the CUDA `adamw_step_8bit` kernel mirrors this reference exactly
/// (same block size, same round-half-away requantization) and is gated against it.
#[derive(Clone, Copy, Debug)]
pub struct Int8AdamW(pub AdamW);

impl Int8AdamW {
    /// Block-wise int8 AdamW at learning rate `lr` with `torch.optim.AdamW` defaults.
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self(AdamW::new(lr))
    }
}

impl Optimizer for Int8AdamW {
    type State = Int8AdamState;

    fn init_state(&self, len: usize) -> Int8AdamState {
        let nblocks = len.div_ceil(INT8_ADAM_BLOCK);
        Int8AdamState {
            m_q: vec![0; len],
            v_q: vec![0; len],
            m_scale: vec![0.0; nblocks],
            v_scale: vec![0.0; nblocks],
            len,
        }
    }

    fn step(&self, t: u64, param: &mut [f32], grad: &[f32], state: &mut Int8AdamState) {
        debug_assert!(t >= 1, "step index t is 1-based; t=0 gives bc=0 ⇒ NaN");
        assert_eq!(param.len(), grad.len(), "param/grad length mismatch");
        assert_eq!(param.len(), state.len, "param/state length mismatch");
        assert_eq!(state.m_q.len(), state.len, "state m_q length mismatch");
        assert_eq!(state.v_q.len(), state.len, "state v_q length mismatch");
        let a = &self.0;
        let exp = i32::try_from(t).unwrap_or(i32::MAX);
        let bc1 = 1.0 - a.beta1.powi(exp);
        let bc2 = 1.0 - a.beta2.powi(exp);
        let shrink = 1.0 - a.lr * a.weight_decay;
        let n = param.len();
        // One block at a time: dequantize its moments, run the exact AdamW element update collecting
        // the new f32 moments, then requantize the block against its own fresh absmax. The device
        // mirror does the identical dequant→update→requant with a per-block reduction.
        for (b, base) in (0..n).step_by(INT8_ADAM_BLOCK).enumerate() {
            let hi = (base + INT8_ADAM_BLOCK).min(n);
            let (ms, vs) = (state.m_scale[b], state.v_scale[b]);
            let mut m_new = vec![0.0f32; hi - base];
            let mut v_new = vec![0.0f32; hi - base];
            for (k, i) in (base..hi).enumerate() {
                let g = grad[i];
                let m_old = f32::from(state.m_q[i]) * ms;
                // v is stored in sqrt-space: dequantize √v then square back to v.
                let root_old = f32::from(state.v_q[i]) * vs;
                let v_old = root_old * root_old;
                let mi = a.beta1 * m_old + (1.0 - a.beta1) * g;
                let vi = a.beta2 * v_old + (1.0 - a.beta2) * g * g;
                param[i] = param[i] * shrink - a.lr * (mi / bc1 / ((vi / bc2).sqrt() + a.eps));
                m_new[k] = mi;
                v_new[k] = vi.sqrt(); // requantize √v, not v
            }
            // Requantize: signed absmax grid for m (÷127); unsigned absmax grid for √v (÷255, ≥0). A
            // zero block keeps scale 0 and codes 0 (dequantizes back to 0 — no division by zero).
            let m_absmax = m_new.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let root_absmax = v_new.iter().fold(0.0f32, |acc, &x| acc.max(x));
            let new_ms = if m_absmax > 0.0 {
                m_absmax / 127.0
            } else {
                0.0
            };
            let new_vs = if root_absmax > 0.0 {
                root_absmax / 255.0
            } else {
                0.0
            };
            state.m_scale[b] = new_ms;
            state.v_scale[b] = new_vs;
            for (k, i) in (base..hi).enumerate() {
                state.m_q[i] = if new_ms > 0.0 {
                    (m_new[k] / new_ms).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                // Floor a nonzero √v at code 1: it must never dequantize to 0, or a later quiet step
                // (g≈0, residual m) collapses AdamW's denominator to eps and the step explodes. Only
                // an exactly-zero √v (a dead block) keeps code 0.
                state.v_q[i] = if new_vs > 0.0 && v_new[k] > 0.0 {
                    (v_new[k] / new_vs).round().clamp(1.0, 255.0) as u8
                } else {
                    0
                };
            }
        }
    }

    fn write_state(&self, state: &Int8AdamState, out: &mut Vec<u8>) {
        out.extend(state.m_q.iter().map(|&q| q as u8));
        out.extend_from_slice(&state.v_q);
        for &x in &state.m_scale {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &state.v_scale {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn read_state(
        &self,
        len: usize,
        cursor: &mut Cursor,
    ) -> Result<Int8AdamState, CheckpointError> {
        let mut m_q = Vec::with_capacity(len);
        for _ in 0..len {
            m_q.push(cursor.u8()? as i8);
        }
        let mut v_q = Vec::with_capacity(len);
        for _ in 0..len {
            v_q.push(cursor.u8()?);
        }
        let nblocks = len.div_ceil(INT8_ADAM_BLOCK);
        let m_scale = cursor.f32_vec(nblocks)?;
        let v_scale = cursor.f32_vec(nblocks)?;
        Ok(Int8AdamState {
            m_q,
            v_q,
            m_scale,
            v_scale,
            len,
        })
    }
}

/// Newton–Schulz quintic orthogonalization (Muon): drives the singular values of a `[rows, cols]`
/// matrix toward 1 (returns `≈ U·Vᵀ`), using ONLY matmuls — no SVD — so it runs on the same device
/// as training. The iteration operates on the smaller Gram matrix (transpose if `rows > cols`), so
/// its cost is `O(min(rows,cols)² · max)` per step; the quintic coefficients are the Muon paper's.
#[must_use]
pub fn newton_schulz(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f32> {
    const A: f32 = 3.4445;
    const B: f32 = -4.7750;
    const C: f32 = 2.0315;
    let fnorm = g.iter().map(|&v| v * v).sum::<f32>().sqrt() + 1e-7;
    let normed: Vec<f32> = g.iter().map(|&v| v / fnorm).collect();
    // Work with `x` shaped `[r, c]` where `r ≤ c`, so the `X·Xᵀ` Gram is `[r, r]` (the smaller one).
    let (mut x, r, c, transposed) = if rows > cols {
        (
            dense::transpose_forward(&normed, rows, cols),
            cols,
            rows,
            true,
        )
    } else {
        (normed, rows, cols, false)
    };
    for _ in 0..steps {
        let a_mat = dense::forward(&x, &x, r, r, c); // A = X·Xᵀ   [r,r]
        let a2 = dense::forward(&a_mat, &a_mat, r, r, r); // A·A (A symmetric ⇒ A·Aᵀ = A·A)  [r,r]
        let b_mat: Vec<f32> = a_mat
            .iter()
            .zip(&a2)
            .map(|(&av, &a2v)| B * av + C * a2v)
            .collect();
        let xt = dense::transpose_forward(&x, r, c); // [c,r]
        let bx = dense::forward(&b_mat, &xt, r, c, r); // B·X   [r,c]
        for (xi, &bxi) in x.iter_mut().zip(&bx) {
            *xi = A * *xi + bxi;
        }
    }
    if transposed {
        dense::transpose_forward(&x, r, c) // back to [rows, cols]
    } else {
        x
    }
}

/// **Muon** (Momentum Orthogonalized by Newton–Schulz) — a memory-lean optimizer for 2D hidden
/// weights: **one** momentum buffer (SGD-momentum memory, half of AdamW's `m`+`v`) with the update
/// spectrally orthogonalized ([`newton_schulz`]) to match AdamW-class convergence. At 32B this
/// halves optimizer-state RAM vs AdamW (128 GB vs 256 GB). Not for embeddings/heads/1D norms —
/// those keep [`AdamW`] (their Gram matrix is huge and orthogonalization is ill-posed). One `Muon`
/// per weight *shape* (it holds `rows`/`cols`); the momentum ↔ update math is bias-correction-free.
#[derive(Clone, Copy, Debug)]
pub struct Muon {
    /// Learning rate (the RMS-match factor `√max(rows,cols)` is applied internally).
    pub lr: f32,
    /// Momentum decay `μ`.
    pub momentum: f32,
    /// Decoupled weight decay (as in AdamW).
    pub weight_decay: f32,
    /// Output rows of the weight this instance steps.
    pub rows: usize,
    /// Input cols of the weight this instance steps.
    pub cols: usize,
    /// Newton–Schulz iterations (5 is the standard; the quintic converges fast).
    pub ns_steps: usize,
}

impl Muon {
    /// Muon at learning rate `lr` for a `[rows, cols]` weight — `μ = 0.95`, `wd = 0.01`, 5 NS steps.
    #[must_use]
    pub fn new(lr: f32, rows: usize, cols: usize) -> Self {
        Self {
            lr,
            momentum: 0.95,
            weight_decay: 0.01,
            rows,
            cols,
            ns_steps: 5,
        }
    }
}

/// Muon's per-weight state: a single momentum buffer (vs AdamW's two).
#[derive(Clone, Debug)]
pub struct MuonState {
    /// Momentum accumulator `M = μ·M + g`.
    pub momentum: Vec<f32>,
}

impl Optimizer for Muon {
    type State = MuonState;

    fn init_state(&self, len: usize) -> MuonState {
        MuonState {
            momentum: vec![0.0; len],
        }
    }

    fn step(&self, _t: u64, param: &mut [f32], grad: &[f32], state: &mut MuonState) {
        assert_eq!(
            param.len(),
            self.rows * self.cols,
            "Muon param shape mismatch"
        );
        assert_eq!(param.len(), grad.len(), "param/grad length mismatch");
        assert_eq!(
            param.len(),
            state.momentum.len(),
            "param/state length mismatch"
        );
        for (mi, &g) in state.momentum.iter_mut().zip(grad) {
            *mi = self.momentum * *mi + g;
        }
        let ortho = newton_schulz(&state.momentum, self.rows, self.cols, self.ns_steps);
        // The orthogonalized update has unit singular values; scale by √max(rows,cols) so its
        // per-element RMS matches an AdamW step (Muon's standard RMS match). Decoupled weight decay.
        let scale = self.lr * (self.rows.max(self.cols) as f32).sqrt();
        let shrink = 1.0 - self.lr * self.weight_decay;
        for (p, &o) in param.iter_mut().zip(&ortho) {
            *p = *p * shrink - scale * o;
        }
    }

    fn write_state(&self, state: &MuonState, out: &mut Vec<u8>) {
        for &x in &state.momentum {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn read_state(&self, len: usize, cursor: &mut Cursor) -> Result<MuonState, CheckpointError> {
        Ok(MuonState {
            momentum: cursor.f32_vec(len)?,
        })
    }
}

#[cfg(test)]
mod muon_tests {
    use super::*;

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

    /// Newton–Schulz drives the singular values toward 1: for the orthogonalized `O` (`r ≤ c`),
    /// `O·Oᵀ → I_r`. Muon's quintic coefficients aim for a *band* around 1 (fast, not exact), so we
    /// check (a) it lands close-ish (`< 0.4`), (b) it SUBSTANTIALLY improves over the raw input's
    /// `X·Xᵀ` deviation, and (c) it preserves energy (`trace(O·Oᵀ) = Σσ² ≈ r`).
    #[test]
    fn newton_schulz_orthogonalizes() {
        let (rows, cols) = (6usize, 10usize);
        let g = seeded(7, rows * cols);
        let dev = |m: &[f32]| {
            let mut e = 0.0f32;
            for i in 0..rows {
                for j in 0..rows {
                    let t = if i == j { 1.0 } else { 0.0 };
                    e = e.max((m[i * rows + j] - t).abs());
                }
            }
            e
        };
        // Normalize the raw input the same way NS does, for a fair before/after comparison.
        let fnorm = g.iter().map(|&v| v * v).sum::<f32>().sqrt() + 1e-7;
        let normed: Vec<f32> = g.iter().map(|&v| v / fnorm).collect();
        let before = dev(&dense::forward(&normed, &normed, rows, rows, cols));

        let o = newton_schulz(&g, rows, cols, 5);
        let oot = dense::forward(&o, &o, rows, rows, cols); // O·Oᵀ  [rows,rows]
        let after = dev(&oot);
        let trace: f32 = (0..rows).map(|i| oot[i * rows + i]).sum();

        assert!(
            after < 0.4,
            "O·Oᵀ should be near I after NS; deviation {after}"
        );
        assert!(
            after < 0.5 * before,
            "NS must improve orthogonality: {before} → {after}"
        );
        assert!(
            (trace - rows as f32).abs() < 0.6 * rows as f32,
            "singular values should be ~1 (trace≈r={rows}); trace {trace}"
        );
    }

    /// Muon descends a quadratic `½‖W − target‖²` (grad = W − target) to near-zero loss.
    #[test]
    fn muon_minimizes_a_quadratic() {
        let (rows, cols) = (8usize, 8usize);
        let target = seeded(3, rows * cols);
        let mut w = seeded(99, rows * cols);
        let opt = Muon::new(0.02, rows, cols);
        let mut st = opt.init_state(w.len());
        let loss = |w: &[f32]| {
            w.iter()
                .zip(&target)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        let l0 = loss(&w);
        for t in 1..=200u64 {
            let grad: Vec<f32> = w.iter().zip(&target).map(|(&a, &b)| a - b).collect();
            opt.step(t, &mut w, &grad, &mut st);
        }
        let l1 = loss(&w);
        assert!(
            l1 < 0.05 * l0,
            "Muon must reduce the quadratic: {l0} → {l1}"
        );
    }

    #[test]
    fn cautious_holds_updates_that_oppose_the_gradient() {
        // wd=0 so a masked element does not move at all. Momentum opposes the gradient on elems 0 & 2.
        let no_wd = AdamW {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let grad = vec![-1.0f32, 1.0, 1.0, -1.0];
        let m0 = vec![3.0f32, 3.0, -3.0, -3.0];

        let opt = CautiousAdamW(no_wd);
        let mut p = vec![1.0f32; 4];
        let mut st = AdamState {
            m: m0.clone(),
            v: vec![0.04; 4],
        };
        opt.step(1, &mut p, &grad, &mut st);
        assert_eq!(p[0], 1.0, "elem 0 opposes the grad → held");
        assert_eq!(p[2], 1.0, "elem 2 opposes the grad → held");
        assert!(p[1] < 1.0, "elem 1 aligned (grad>0) → decreases: {}", p[1]);
        assert!(p[3] > 1.0, "elem 3 aligned (grad<0) → increases: {}", p[3]);

        // Standard AdamW moves the opposing element instead of holding it.
        let mut p2 = vec![1.0f32; 4];
        let mut st2 = AdamState {
            m: m0,
            v: vec![0.04; 4],
        };
        no_wd.step(1, &mut p2, &grad, &mut st2);
        assert!((p2[0] - 1.0).abs() > 1e-6, "standard AdamW moves elem 0");
    }

    #[test]
    fn cautious_minimizes_a_quadratic() {
        // On a clean descent the update aligns with the gradient, so Cautious still converges.
        let n = 64usize;
        let target = seeded(5, n);
        let mut w = seeded(77, n);
        let opt = CautiousAdamW::new(0.05);
        let mut st = opt.init_state(n);
        let loss = |w: &[f32]| {
            w.iter()
                .zip(&target)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        let l0 = loss(&w);
        for t in 1..=300u64 {
            let grad: Vec<f32> = w.iter().zip(&target).map(|(&a, &b)| a - b).collect();
            opt.step(t, &mut w, &grad, &mut st);
        }
        let l1 = loss(&w);
        assert!(
            l1 < 0.05 * l0,
            "Cautious AdamW must reduce the quadratic: {l0} → {l1}"
        );
    }

    /// After one step the stored int8 moments must sit within one quantization grid step of the true
    /// f32 AdamW moments — block by block, including the short ragged final block (300 = 256 + 44).
    /// (Step 1 also leaves the *parameters* bit-identical to f32 AdamW: the zero initial state
    /// dequantizes to 0, so the update uses the same full-precision mᵢ,vᵢ; quantization only diverges
    /// the trajectory from step 2.)
    #[test]
    fn int8_adam_moments_track_f32_within_grid() {
        let n = 300usize;
        let grad = seeded(11, n);
        let opt8 = Int8AdamW::new(0.01);
        let optf = AdamW::new(0.01);
        let mut p8 = seeded(22, n);
        let mut pf = p8.clone();
        let mut s8 = opt8.init_state(n);
        let mut sf = optf.init_state(n);
        opt8.step(1, &mut p8, &grad, &mut s8);
        optf.step(1, &mut pf, &grad, &mut sf);
        assert_eq!(
            p8, pf,
            "step-1 params must match f32 (zero initial moments)"
        );
        for b in 0..n.div_ceil(INT8_ADAM_BLOCK) {
            let lo = b * INT8_ADAM_BLOCK;
            let hi = (lo + INT8_ADAM_BLOCK).min(n);
            let (ms, vs) = (s8.m_scale[b], s8.v_scale[b]);
            for i in lo..hi {
                let m_dq = f32::from(s8.m_q[i]) * ms;
                // v is stored in sqrt-space, so compare √v to √v_true within one sqrt-grid step.
                let root_dq = f32::from(s8.v_q[i]) * vs;
                assert!(
                    (m_dq - sf.m[i]).abs() <= ms + 1e-9,
                    "m off-grid at {i}: {m_dq} vs {} (step {ms})",
                    sf.m[i]
                );
                assert!(
                    (root_dq - sf.v[i].sqrt()).abs() <= vs + 1e-9,
                    "√v off-grid at {i}: {root_dq} vs {} (step {vs})",
                    sf.v[i].sqrt()
                );
            }
        }
    }

    /// The 4×-smaller int8 moment state must still drive a quadratic to near-zero, comparably to full
    /// f32 AdamW — the quality bar that makes the VRAM shrink usable, not just cheap.
    #[test]
    fn int8_adam_descends_like_f32() {
        let n = 512usize;
        let target = seeded(5, n);
        let loss = |w: &[f32]| {
            w.iter()
                .zip(&target)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        let opt8 = Int8AdamW::new(0.05);
        let mut w8 = seeded(77, n);
        let mut s8 = opt8.init_state(n);
        let optf = AdamW::new(0.05);
        let mut wf = seeded(77, n);
        let mut sf = optf.init_state(n);
        for t in 1..=300u64 {
            let g8: Vec<f32> = w8.iter().zip(&target).map(|(&a, &b)| a - b).collect();
            opt8.step(t, &mut w8, &g8, &mut s8);
            let gf: Vec<f32> = wf.iter().zip(&target).map(|(&a, &b)| a - b).collect();
            optf.step(t, &mut wf, &gf, &mut sf);
        }
        let (l8, lf) = (loss(&w8), loss(&wf));
        assert!(l8 < 1e-3, "int8 Adam must converge the quadratic: {l8}");
        assert!(
            l8 < 3.0 * lf.max(1e-12) + 1e-4,
            "int8 Adam must stay within a small factor of f32: {l8} vs {lf}"
        );
    }

    /// Regression for the real-135M int8 divergence, reproduced at unit scale. A block's `m` absmax and
    /// `√v` absmax can be dominated by DIFFERENT coordinates: a huge-magnitude **oscillating** coord
    /// (coord 0 here, ±1000) has `m ≈ 0` (sign cancellation) but a huge `√v` (RMS), so it dominates the
    /// √v grid but NOT the m grid. A steady neighbour (coord 1) then has its `v` rounded to int8 code 0
    /// while its `m` survives; when coord 1 goes quiet, `vi` collapses to 0 and AdamW's `m/(√v+eps)`
    /// step explodes. The code-1 floor on a nonzero `√v` keeps the denominator bounded.
    #[test]
    fn int8_adam_quiet_coord_in_wide_block_does_not_explode() {
        let n = 2usize;
        let opt = Int8AdamW::new(0.01);
        let mut w = vec![0.0f32, 0.0];
        let mut st = opt.init_state(n);
        for step in 1..=20u64 {
            // Coord 0: huge OSCILLATING gradient → m cancels toward 0, √v stays ~1000 (dominates the
            // √v grid but not the m grid). Coord 1: builds moment for 4 steps, then goes silent.
            let osc = if step % 2 == 0 { -1000.0f32 } else { 1000.0 };
            let g = if step <= 4 {
                vec![osc, 1.0]
            } else {
                vec![osc, 0.0]
            };
            opt.step(step, &mut w, &g, &mut st);
            assert!(
                w[1].is_finite() && w[1].abs() < 100.0,
                "quiet coord exploded at step {step}: w[1] = {}",
                w[1]
            );
        }
    }

    /// Block-wise int8 state (signed m, unsigned v, per-block scales, ragged tail) round-trips through
    /// the checkpoint serializer byte-for-byte.
    #[test]
    fn int8_adam_state_survives_checkpoint() {
        let n = 300usize;
        let opt = Int8AdamW::new(0.01);
        let mut s = opt.init_state(n);
        let mut w = seeded(1, n);
        let g = seeded(2, n);
        opt.step(1, &mut w, &g, &mut s);
        opt.step(2, &mut w, &g, &mut s);
        let mut bytes = Vec::new();
        opt.write_state(&s, &mut bytes);
        let mut cur = Cursor::new(&bytes);
        let s2 = opt.read_state(n, &mut cur).expect("read int8 adam state");
        assert_eq!(s, s2, "int8 adam state must round-trip");
        assert_eq!(cur.remaining(), 0, "no trailing bytes");
    }
}
