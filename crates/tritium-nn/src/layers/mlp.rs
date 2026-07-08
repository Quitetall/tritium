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
//! that exactly: the gated body `relu(gate(x))² ⊙ up(x)`, then the intermediate
//! `ffn_sub_norm` (a `BitNetRMSNorm` over `n_ff`) applied to that product, then
//! `down`. The sub-norm is load-bearing for layer-0 numeric parity.

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::Projection;
use crate::ops::rmsnorm;

/// A gated squared-ReLU MLP block with the BitNet intermediate sub-norm.
#[allow(missing_debug_implementations)]
pub struct Relu2Mlp {
    /// Gate projection `n_embd → n_ff` (the squared-ReLU branch).
    pub gate: Projection,
    /// Up projection `n_embd → n_ff` (the linear branch).
    pub up: Projection,
    /// Down projection `n_ff → n_embd`.
    pub down: Projection,
    /// `ffn_sub_norm` (`BitNetRMSNorm` over `n_ff`) applied to the gated product
    /// before `down`; length `n_ff`.
    pub ffn_sub_norm: Vec<f32>,
    /// RMSNorm epsilon for `ffn_sub_norm`.
    pub rms_eps: f32,
}

impl Relu2Mlp {
    /// Forward over `m` tokens:
    /// `out = down(ffn_sub_norm(relu(gate(x))² ⊙ up(x)))`.
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
        let n_embd = self.gate.k_in();
        let n_ff = self.gate.n_out();
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

        // BitNet `ffn_sub_norm` over the gated product, row by row, before `down`.
        if self.ffn_sub_norm.len() == n_ff {
            let mut normed = vec![0.0f32; m * n_ff];
            for t in 0..m {
                let src = &gate[t * n_ff..t * n_ff + n_ff];
                let dst = &mut normed[t * n_ff..t * n_ff + n_ff];
                rmsnorm(src, &self.ffn_sub_norm, self.rms_eps, dst)?;
            }
            return self.down.forward(backend, &normed, m, out);
        }

        // down(·): `[m, n_ff] → [m, n_embd]`.
        self.down.forward(backend, &gate, m, out)
    }
}

/// A gated SwiGLU MLP: `down( silu(gate(x)) ⊙ up(x) )`, `silu(z) = z·σ(z)`. The
/// Llama/Qwen feed-forward — no sub-norm (unlike [`Relu2Mlp`]). `gate`/`up` project
/// `n_embd → n_ff`; `down` projects `n_ff → n_embd`.
#[allow(missing_debug_implementations)]
pub struct SwiGluMlp {
    /// Gate projection `n_embd → n_ff` (the SiLU branch).
    pub gate: Projection,
    /// Up projection `n_embd → n_ff` (the linear branch).
    pub up: Projection,
    /// Down projection `n_ff → n_embd`.
    pub down: Projection,
}

impl SwiGluMlp {
    /// Forward over `m` tokens: `out = down( silu(gate(x)) ⊙ up(x) )`.
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
        let n_embd = self.gate.k_in();
        let n_ff = self.gate.n_out();
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

        // Gated activation in place into `gate`: silu(g) ⊙ u, silu(z) = z·σ(z).
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            let z = *g;
            let silu = z / (1.0 + (-z).exp());
            *g = silu * u;
        }

        // down(·): `[m, n_ff] → [m, n_embd]`.
        self.down.forward(backend, &gate, m, out)
    }
}

/// The block's feed-forward, dispatched by architecture ([`crate::MlpKind`]).
#[allow(missing_debug_implementations)]
pub enum Mlp {
    /// BitNet squared-ReLU with sub-norm.
    Relu2(Relu2Mlp),
    /// Llama/Qwen SwiGLU.
    SwiGlu(SwiGluMlp),
}

impl Mlp {
    /// Dispatch the feed-forward forward pass. `x`/`out` are `[m, n_embd]`.
    ///
    /// # Errors
    /// Propagates the underlying MLP's [`NnError`].
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        match self {
            Mlp::Relu2(mlp) => mlp.forward(backend, x, m, out),
            Mlp::SwiGlu(mlp) => mlp.forward(backend, x, m, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::DenseLinear;

    fn dense(w: &[f32], n_out: usize, k_in: usize) -> Projection {
        // Exact fp32 — the general-inference path (matches the fp64 hand reference).
        Projection::Dense(DenseLinear::new_exact(w.to_vec(), n_out, k_in).unwrap())
    }

    #[test]
    fn swiglu_matches_hand_reference() {
        let backend = tritium_cpu::CpuBackend::new();
        let (n_embd, n_ff) = (2usize, 3usize);
        // Row-major [n_out, k_in].
        let gate_w = [0.5f32, -0.3, 0.1, 0.2, -0.4, 0.6]; // [3, 2]
        let up_w = [0.2f32, 0.1, 0.3, -0.5, 0.7, 0.2]; // [3, 2]
        let down_w = [0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6]; // [2, 3]
        let x = [1.0f32, -2.0];

        let mlp = SwiGluMlp {
            gate: dense(&gate_w, n_ff, n_embd),
            up: dense(&up_w, n_ff, n_embd),
            down: dense(&down_w, n_embd, n_ff),
        };
        let mut out = vec![0.0f32; n_embd];
        mlp.forward(&backend, &x, 1, &mut out).unwrap();

        // fp64 hand reference.
        let dot = |w: &[f32], r: usize| -> f64 {
            (0..n_embd)
                .map(|c| f64::from(w[r * n_embd + c]) * f64::from(x[c]))
                .sum()
        };
        let silu = |z: f64| z / (1.0 + (-z).exp());
        let h: Vec<f64> = (0..n_ff)
            .map(|r| silu(dot(&gate_w, r)) * dot(&up_w, r))
            .collect();
        let expect: Vec<f64> = (0..n_embd)
            .map(|o| {
                (0..n_ff)
                    .map(|r| f64::from(down_w[o * n_ff + r]) * h[r])
                    .sum()
            })
            .collect();
        for (o, &e) in expect.iter().enumerate() {
            assert!(
                (f64::from(out[o]) - e).abs() < 1e-5,
                "out[{o}]={} expect={e}",
                out[o]
            );
        }
    }
}
