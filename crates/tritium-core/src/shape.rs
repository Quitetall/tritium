//! GEMM problem geometry.

/// Dimensions of a matmul `C[M,N] = A[M,K] · Wᵀ`, where `W` is the `[N, K]`
/// ternary weight (row-major, output-major) and `A` is the activation `[M, K]`.
///
/// - `m` — activation rows (batch × sequence).
/// - `n` — output features (weight rows).
/// - `k` — contraction / input features (weight + activation columns).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct GemmShape {
    /// Activation rows (batch × sequence).
    pub m: usize,
    /// Output features (weight rows).
    pub n: usize,
    /// Contraction / input features (weight + activation columns).
    pub k: usize,
}

impl GemmShape {
    /// Construct a shape from its `m`, `n`, `k` dimensions.
    #[inline]
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    /// Multiply-accumulate count `M·N·K`. For ternary this is the count of
    /// add/sub/skip ops, not true FMAs — the metric backends report throughput on.
    #[inline]
    pub const fn macs(&self) -> u64 {
        (self.m as u64) * (self.n as u64) * (self.k as u64)
    }

    /// Whether the operand/output buffer lengths are internally consistent.
    #[inline]
    pub const fn buffers_fit(&self, act_len: usize, weight_len: usize, out_len: usize) -> bool {
        act_len == self.m * self.k && weight_len == self.n * self.k && out_len == self.m * self.n
    }
}
