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

/// Geometry of a ternary 1-D convolution `Y[B, C_out, L_out] = scale ⊙ conv1d(X, W)` — the codec's
/// conv op (ADR 0030). The weight is packed 2-D `[C_out, (C_in/groups)·K]` (the per-output-channel
/// ternary reshape), so `k_g()` is the matmul contraction and `n_g()` the per-group output count.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ConvShape {
    /// Batch size.
    pub batch: usize,
    /// Input channels (divisible by `groups`).
    pub c_in: usize,
    /// Output channels (divisible by `groups`).
    pub c_out: usize,
    /// Input length.
    pub l_in: usize,
    /// Kernel size.
    pub k: usize,
    /// Stride (≥ 1).
    pub stride: usize,
    /// Dilation (≥ 1).
    pub dilation: usize,
    /// Left zero-padding.
    pub pad_left: usize,
    /// Right zero-padding.
    pub pad_right: usize,
    /// Convolution groups (≥ 1).
    pub groups: usize,
}

impl ConvShape {
    /// Output length, or `0` if the dilated kernel is wider than the padded input (or the geometry is
    /// degenerate).
    #[inline]
    pub const fn l_out(&self) -> usize {
        if self.k == 0 || self.stride == 0 {
            return 0;
        }
        let eff = self.dilation * (self.k - 1) + 1;
        let padded = self.l_in + self.pad_left + self.pad_right;
        if padded < eff {
            return 0;
        }
        (padded - eff) / self.stride + 1
    }

    /// Input channels per group `C_in/groups`.
    #[inline]
    pub const fn c_in_pg(&self) -> usize {
        self.c_in / self.groups
    }

    /// Output channels per group `N_g = C_out/groups`.
    #[inline]
    pub const fn n_g(&self) -> usize {
        self.c_out / self.groups
    }

    /// Flattened per-output-channel weight width `K_g = (C_in/groups)·K`.
    #[inline]
    pub const fn k_g(&self) -> usize {
        self.c_in_pg() * self.k
    }

    /// Whether the geometry is well-formed and the buffers are the right length (`x=B·C_in·L_in`,
    /// `weights=C_out·K_g`, `scale=C_out`, `out=B·C_out·L_out`).
    #[inline]
    pub fn buffers_fit(
        &self,
        x_len: usize,
        weight_len: usize,
        scale_len: usize,
        out_len: usize,
    ) -> bool {
        self.groups != 0
            && self.k != 0
            && self.stride != 0
            && self.dilation != 0
            && self.c_in % self.groups == 0
            && self.c_out % self.groups == 0
            && self.l_out() > 0
            && x_len == self.batch * self.c_in * self.l_in
            && weight_len == self.c_out * self.k_g()
            && scale_len == self.c_out
            && out_len == self.batch * self.c_out * self.l_out()
    }
}
