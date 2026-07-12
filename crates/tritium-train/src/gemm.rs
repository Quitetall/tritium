//! Pluggable GEMM backend for the Tape's matmuls (plan 0043 — GPU training path).
//!
//! The CPU tape computes `dense_matmul`/`matmul` with the built-in `ops` free-functions. Injecting a
//! `TrainGemm` (e.g. tritium-cuda's GPU engine, backed by the bit-exact `train_grad.cu` kernels)
//! routes those matmuls — the training hot path (~95% of the FLOPs) — onto the device instead, while
//! the cheaper elementwise/norm ops stay on the host. `s = ones` ⇒ a plain fp `dense_matmul`; a real
//! per-row scale ⇒ the ternary `matmul` (`Y = s·(A·Wᵀ)`). A `Tape` with no engine (the default) runs
//! the built-in CPU path bit-for-bit, so every existing gate is untouched.

/// A device (or host) GEMM engine the [`Tape`](crate::Tape) can dispatch its `dense_matmul` to.
///
/// Contract: the result must match the CPU `ops::dense` forward/vjp within the training tolerance
/// (the reference GPU kernels are compiled `--fmad=false` with a fixed sequential reduction so they
/// reproduce the host rounding — see `tritium-cuda/kernels/train_grad.cu`).
pub trait TrainGemm {
    /// `Y[m,n] = Σ_k X[m,k]·W[n,k]`  (`X:[M,K]`, `W:[N,K]`, `Y:[M,N]`).
    fn dense_forward(&self, x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32>;

    /// `(g_x[M,K], g_w[N,K])` for `gy[M,N]` — the two fp-matmul gradients. (No per-row scale, so no
    /// `grad_s` — a real saving vs the ternary path, since the `grad_s` kernel costs a full GEMM.)
    fn dense_backward(
        &self,
        gy: &[f32],
        x: &[f32],
        w: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<f32>);
}
