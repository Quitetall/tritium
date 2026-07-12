//! Pluggable GEMM backend for the Tape's matmuls (plan 0043 — GPU training path).
//!
//! The CPU tape computes `dense_matmul`/`matmul` with the built-in `ops` free-functions. Injecting a
//! `TrainGemm` (e.g. tritium-cuda's GPU engine, backed by the bit-exact `train_grad.cu` kernels)
//! routes those matmuls — the training hot path (~95% of the FLOPs) — onto the device instead, while
//! the cheaper elementwise/norm ops stay on the host. `s = ones` ⇒ a plain fp `dense_matmul`; a real
//! per-row scale ⇒ the ternary `matmul` (`Y = s·(A·Wᵀ)`). A `Tape` with no engine (the default) runs
//! the built-in CPU path bit-for-bit, so every existing gate is untouched.

use tritium_core::GemmShape;

/// A device (or host) GEMM engine the [`Tape`](crate::Tape) can dispatch its matmuls to.
///
/// Contract: the result must match the CPU `ops::{dense,matmul}` forward/vjp within the training
/// tolerance (the reference GPU kernels are compiled `--fmad=false` with a fixed sequential
/// reduction so they reproduce the host rounding — see `tritium-cuda/kernels/train_grad.cu`).
pub trait TrainGemm {
    /// `Y[m,n] = s[n]·Σ_k A[m,k]·W[n,k]`  (`A:[M,K]`, `W:[N,K]`, `s:[N]`, `Y:[M,N]`).
    fn forward(&self, a: &[f32], w: &[f32], s: &[f32], shape: GemmShape) -> Vec<f32>;

    /// `(g_a[M,K], g_w[N,K], g_s[N])` for `gy[M,N]` — the three matmul gradients.
    fn backward(
        &self,
        gy: &[f32],
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>);
}
