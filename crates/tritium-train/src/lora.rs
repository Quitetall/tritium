//! LoRA adapter over a frozen base (ADR 0007, plan 0009).
//!
//! A low-rank update `ΔW = (α/r)·B·A` (`A:[r,K]`, `B:[N,r]` → `ΔW:[N,K]`) added to a frozen
//! base weight. Only `A` and `B` train; the base is held constant (in the autograd graph the
//! base matmul is wrapped in [`Tape::detach`](crate::Tape::detach), so the base leaves get
//! zero gradient).
//!
//! This struct is pure data + the merge math; the differentiable layer forward
//! `Y = detach(base) + (α/r)·(act·Aᵀ)·Bᵀ` is composed on the tape from the `dense_matmul`,
//! `detach`, and `scale_const` primitives (it is not a fused op).

/// A LoRA adapter: low-rank factors `A` (`[rank, k]`) and `B` (`[n, rank]`), row-major, plus
/// the scaling hyper-parameter `alpha`. The effective update is `(alpha/rank)·B·A`.
#[derive(Clone, Debug, PartialEq)]
pub struct Lora {
    /// Down-projection factor, shape `[rank, k]` (row-major).
    pub a: Vec<f32>,
    /// Up-projection factor, shape `[n, rank]` (row-major).
    pub b: Vec<f32>,
    /// Adapter rank `r` (inner dimension of `B·A`).
    pub rank: usize,
    /// Output channels `N` (rows of the base / of `B`).
    pub n: usize,
    /// Input features `K` (cols of the base / of `A`).
    pub k: usize,
    /// LoRA scaling numerator; the applied scale is `alpha / rank`.
    pub alpha: f32,
}

impl Lora {
    /// The applied LoRA scale `α/r`.
    #[must_use]
    pub fn scaling(&self) -> f32 {
        debug_assert!(self.rank > 0, "LoRA rank must be ≥ 1");
        self.alpha / self.rank as f32
    }

    /// The dense delta-weight `ΔW[n,k] = (α/r)·Σ_j B[n,j]·A[j,k]`, shape `[n, k]`.
    #[must_use]
    #[allow(clippy::needless_range_loop)]
    pub fn delta_weights(&self) -> Vec<f32> {
        assert_eq!(
            self.a.len(),
            self.rank * self.k,
            "lora.a shape mismatch (expected rank*k)"
        );
        assert_eq!(
            self.b.len(),
            self.n * self.rank,
            "lora.b shape mismatch (expected n*rank)"
        );
        let s = self.scaling();
        let mut dw = vec![0.0f32; self.n * self.k];
        for ni in 0..self.n {
            for ki in 0..self.k {
                let mut acc = 0.0f32;
                for j in 0..self.rank {
                    acc += self.b[ni * self.rank + j] * self.a[j * self.k + ki];
                }
                dw[ni * self.k + ki] = s * acc;
            }
        }
        dw
    }

    /// Fold the adapter into a dense base weight for inference:
    /// `W_merged[n,k] = base_dense[n,k] + ΔW[n,k]`. `base_dense` is the dequantized base
    /// (`scale[n]·trits[n,k]`), length `n*k`.
    #[must_use]
    pub fn merge(&self, base_dense: &[f32]) -> Vec<f32> {
        assert_eq!(
            base_dense.len(),
            self.n * self.k,
            "base_dense shape mismatch"
        );
        let dw = self.delta_weights();
        base_dense.iter().zip(&dw).map(|(&w, &d)| w + d).collect()
    }
}
