//! BitNet feed-forward: a gated ReLU² MLP.
//!
//! `down(relu(gate(x))² ⊙ up(x))` — `gate`, `up`, and `down` are all
//! [`TernaryLinear`] (no biases), with `gate`/`up` projecting `n_embd → n_ff` and
//! `down` projecting `n_ff → n_embd`. The squared-ReLU activation
//! (`relu(z)² = max(z, 0)²`) is BitNet's, in place of SwiGLU.
//!
//! # Reference (`transformers` `modeling_bitnet.BitNetMLP`)
//!
//! The shipped model is `down_proj(ffn_sub_norm(act_fn(gate_proj(x)) *
//! up_proj(x)))` with `act_fn = relu2` (i.e. `relu(z)²`). This struct implements
//! the **gated body** `down(relu(gate(x))² ⊙ up(x))` exactly; the intermediate
//! `ffn_sub_norm` (a `BitNetRMSNorm` over `n_ff`, unit-weight by default) lives in
//! the BitLinear path and is wired in WF-4 with the rest of the sub-norm/quant
//! placement, not here.

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
        let n_embd = self.gate.k_in;
        let n_ff = self.gate.n_out;
        // Shape contract: gate/up share `[n_embd → n_ff]`; down is `[n_ff → n_embd]`.
        if x.len() != m * n_embd {
            return Err(NnError::Shape {
                expected: m * n_embd,
                got: x.len(),
            });
        }
        if out.len() != m * n_embd {
            return Err(NnError::Shape {
                expected: m * n_embd,
                got: out.len(),
            });
        }

        // gate(x) and up(x): both `[m, n_ff]`.
        let mut gate = vec![0.0f32; m * n_ff];
        let mut up = vec![0.0f32; m * n_ff];
        self.gate.forward(backend, x, m, &mut gate)?;
        self.up.forward(backend, x, m, &mut up)?;

        // Gated activation in place into `gate`: relu(g)² ⊙ u.
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            let r = g.max(0.0);
            *g = r * r * u;
        }

        // down(·): `[m, n_ff] → [m, n_embd]`.
        self.down.forward(backend, &gate, m, out)
    }
}
