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
use crate::layers::{Projection, ProjectionActivationMode};
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
    /// Bind three projections to one validated SwiGLU geometry.
    ///
    /// `gate` and `up` must both map `n_embd → n_ff`; `down` must map
    /// `n_ff → n_embd`. All projections must consume activations through one
    /// arithmetic mode.
    ///
    /// Public fields remain available for checkpoint loaders. Model binders must
    /// repeat the one-time parameter scan after any such mutation; forward keeps
    /// only O(1) geometry validation on its hot path.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] for zero or contradictory projection geometry,
    /// [`NnError::MissingConfig`] when activation arithmetic modes differ, or
    /// [`NnError::Backend`] when a projection parameter is non-finite.
    pub fn new(gate: Projection, up: Projection, down: Projection) -> Result<Self, NnError> {
        let mlp = Self { gate, up, down };
        mlp.activation_mode()?;
        validate_projection_parameters(&mlp.gate)?;
        validate_projection_parameters(&mlp.up)?;
        validate_projection_parameters(&mlp.down)?;
        Ok(mlp)
    }

    /// Validate projection geometry and return their shared activation mode.
    ///
    /// This is an O(1) hot-path check. It validates retained buffer geometry but
    /// deliberately does not rescan matrix values; constructors and model binders
    /// perform the one-time finiteness scan.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] for zero or contradictory projection geometry,
    /// [`NnError::MissingConfig`] when activation arithmetic modes differ, or
    /// [`NnError::Backend`] when retained device geometry is invalid.
    pub fn activation_mode(&self) -> Result<ProjectionActivationMode, NnError> {
        let n_embd = self.gate.k_in();
        let n_ff = self.gate.n_out();
        if n_embd == 0 || n_ff == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: n_embd.min(n_ff),
            });
        }
        validate_projection(&self.gate, n_ff, n_embd)?;
        validate_projection(&self.up, n_ff, n_embd)?;
        validate_projection(&self.down, n_embd, n_ff)?;

        let mode = self.gate.activation_mode();
        if self.up.activation_mode() != mode || self.down.activation_mode() != mode {
            return Err(NnError::MissingConfig(
                "SwiGLU projections must use one activation arithmetic mode".to_owned(),
            ));
        }
        Ok(mode)
    }

    /// Forward over `m` tokens: `out = down( silu(gate(x)) ⊙ up(x) )`.
    /// `x` is `[m, n_embd]`; `out` is `[m, n_embd]`, overwritten.
    ///
    /// # Errors
    /// [`NnError::Shape`] on projection or buffer mismatch,
    /// [`NnError::MissingConfig`] when projection activation modes differ, or
    /// [`NnError::Backend`] if scratch allocation or a projection fails.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        self.activation_mode()?;
        let n_embd = self.gate.k_in();
        let n_ff = self.gate.n_out();
        let input_len = checked_buffer_len(m, n_embd, x.len())?;
        if x.len() != input_len {
            return Err(NnError::Shape {
                expected: input_len,
                got: x.len(),
            });
        }
        let output_len = checked_buffer_len(m, n_embd, out.len())?;
        if out.len() != output_len {
            return Err(NnError::Shape {
                expected: output_len,
                got: out.len(),
            });
        }
        let hidden_len = checked_buffer_len(m, n_ff, x.len())?;

        // gate(x) and up(x): both `[m, n_ff]`.
        let mut gate = zeroed_scratch(hidden_len, "SwiGLU gate")?;
        let mut up = zeroed_scratch(hidden_len, "SwiGLU up")?;
        let mut staged_out = zeroed_scratch(output_len, "SwiGLU output")?;
        self.gate.forward(backend, x, m, &mut gate)?;
        self.up.forward(backend, x, m, &mut up)?;

        // Gated activation in place into `gate`: silu(g) ⊙ u, silu(z) = z·σ(z).
        for (g, &u) in gate.iter_mut().zip(up.iter()) {
            let z = *g;
            let silu = z / (1.0 + (-z).exp());
            *g = silu * u;
        }

        // down(·): `[m, n_ff] → [m, n_embd]`.
        self.down.forward(backend, &gate, m, &mut staged_out)?;
        out.copy_from_slice(&staged_out);
        Ok(())
    }
}

fn validate_projection(
    projection: &Projection,
    expected_n_out: usize,
    expected_k_in: usize,
) -> Result<(), NnError> {
    if projection.n_out() != expected_n_out {
        return Err(NnError::Shape {
            expected: expected_n_out,
            got: projection.n_out(),
        });
    }
    if projection.k_in() != expected_k_in {
        return Err(NnError::Shape {
            expected: expected_k_in,
            got: projection.k_in(),
        });
    }
    projection.validate_retained_geometry()
}

fn validate_projection_parameters(projection: &Projection) -> Result<(), NnError> {
    projection.validate_retained_parameters()
}

fn checked_buffer_len(rows: usize, width: usize, got: usize) -> Result<usize, NnError> {
    rows.checked_mul(width).ok_or(NnError::Shape {
        expected: usize::MAX,
        got,
    })
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate {name} scratch for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
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

    /// The inner [`Relu2Mlp`] if this is a BitNet MLP, else `None`. Used by the
    /// BitNet-only paths (the CUDA resident decoder, accuracy diagnostics) that read
    /// `gate`/`up`/`down`/`ffn_sub_norm` directly.
    #[must_use]
    pub fn as_relu2(&self) -> Option<&Relu2Mlp> {
        match self {
            Mlp::Relu2(mlp) => Some(mlp),
            Mlp::SwiGlu(_) => None,
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
