//! BitNet feed-forward: a gated ReLU² MLP.
//!
//! `down(relu(gate(x))² ⊙ up(x))` — `gate`, `up`, and `down` are all
//! [`TernaryLinear`] (no biases), with `gate`/`up` projecting `n_embd → n_ff` and
//! `down` projecting `n_ff → n_embd`. The squared-ReLU activation
//! (`relu(z)² = max(z, 0)²`) is BitNet's, in place of SwiGLU. Forward lands in
//! WF-3.

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::TernaryLinear;

/// A gated squared-ReLU MLP block.
#[allow(missing_debug_implementations)]
pub struct Relu2Mlp {
    /// Gate projection `n_embd → n_ff` (the squared-ReLU branch).
    pub gate: TernaryLinear,
    /// Up projection `n_embd → n_ff` (the linear branch).
    pub up: TernaryLinear,
    /// Down projection `n_ff → n_embd`.
    pub down: TernaryLinear,
}

impl Relu2Mlp {
    /// Forward over `m` tokens: `out = down(relu(gate(x))² ⊙ up(x))`.
    ///
    /// `x` is `[m, n_embd]`; `out` is `[m, n_embd]`, overwritten.
    ///
    /// # Errors
    /// [`NnError::Shape`] on buffer-length mismatch, or [`NnError::Backend`] if a
    /// projection's backend GEMM fails.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        let _ = (backend, x, m, out);
        todo!("WF-3: gated ReLU² MLP forward")
    }
}
