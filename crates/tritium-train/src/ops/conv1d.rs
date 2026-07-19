//! Ternary 1-D convolution forward + backward for the autograd tape.
//!
//! Forward:  `Y[b, co, l] = s[co] · Σ_{ci,kk} X[b, ci, l·stride + kk·dilation − pad_left] · W[co, ci', kk]`
//! (grouped: `ci` ranges over this group's input channels; `ci'` is its group-local index).
//!
//! The convolution is lowered to **im2col → [`matmul::forward`](super::matmul) → col2im**: for each
//! `(batch, group)` the input patches are gathered into a `[L_out, K_g]` matrix (`K_g = (C_in/groups)·K`)
//! and contracted against the group's reshaped weight `[N_g, K_g]` (`N_g = C_out/groups`) by the exact
//! same scale-folded ternary contraction the transformer linears use. This buys conv the matmul's
//! bit-identical forward and its already-gradchecked `vjp` for free; the only conv-specific code is the
//! gather (im2col) and its transpose (col2im), which are pure index arithmetic — no float accumulation
//! in the forward, and a **canonically ordered** accumulation in the backward's input grad so the
//! overlapping-window sums stay byte-identical across CPU/CUDA/MCU (the ADR 0018 discipline).
//!
//! The weight is laid out `[C_out, C_in/groups, K]` → flat `[C_out, K_g]`, which is exactly the reshape
//! the ternary path wants: `ste::absmean_scale_per_row` on `[C_out, K_g]` yields a **per-output-channel**
//! AbsMean scale, and because `K_g` spans only this output channel's own group inputs the per-row scale
//! is automatically per-group-correct. The fp (decoder) path is the same op with `scale = [1.0; C_out]`.

use super::matmul;

/// Geometry of a 1-D convolution. Explicit `pad_left`/`pad_right` (not a single symmetric pad) so
/// even-kernel "same" padding and byte-exact MCU deploy are expressible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dCfg {
    /// Batch size `B`.
    pub batch: usize,
    /// Input channels `C_in` (must be divisible by `groups`).
    pub c_in: usize,
    /// Output channels `C_out` (must be divisible by `groups`).
    pub c_out: usize,
    /// Input length `L_in`.
    pub l_in: usize,
    /// Kernel size `K`.
    pub k: usize,
    /// Stride (≥ 1).
    pub stride: usize,
    /// Dilation (≥ 1).
    pub dilation: usize,
    /// Zero-padding prepended to the left.
    pub pad_left: usize,
    /// Zero-padding appended to the right.
    pub pad_right: usize,
    /// Convolution groups (≥ 1): `groups == C_in == C_out` is depthwise, `groups == 1` is dense.
    pub groups: usize,
}

impl Conv1dCfg {
    /// Output length `L_out = ⌊(L_in + pad_left + pad_right − dilation·(K−1) − 1)/stride⌋ + 1`,
    /// or `0` when the (dilated) kernel is wider than the padded input.
    #[must_use]
    pub fn l_out(&self) -> usize {
        let eff = self.dilation * (self.k - 1) + 1; // dilated kernel span
        let padded = self.l_in + self.pad_left + self.pad_right;
        if self.k == 0 || self.stride == 0 || padded < eff {
            return 0;
        }
        (padded - eff) / self.stride + 1
    }

    /// Input channels per group `C_in/groups`.
    #[must_use]
    pub fn c_in_pg(&self) -> usize {
        self.c_in / self.groups
    }

    /// Output channels per group `N_g = C_out/groups` (the matmul `n`).
    #[must_use]
    pub fn n_g(&self) -> usize {
        self.c_out / self.groups
    }

    /// Flattened per-output-channel weight width `K_g = (C_in/groups)·K` (the matmul `k`).
    #[must_use]
    pub fn k_g(&self) -> usize {
        self.c_in_pg() * self.k
    }

    /// Whether the geometry is well-formed and the supplied buffers are the right length.
    #[must_use]
    pub fn buffers_fit(
        &self,
        x_len: usize,
        w_len: usize,
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
            && w_len == self.c_out * self.k_g()
            && scale_len == self.c_out
            && out_len == self.batch * self.c_out * self.l_out()
    }
}

/// im2col for one `(batch, group)`: gather the `[L_out, K_g]` patch matrix. Column `j = ci'·K + kk`;
/// out-of-range taps (padding) are zero. `cfg` fields are read directly for indexing.
fn im2col(x: &[f32], cfg: &Conv1dCfg, b: usize, g: usize) -> Vec<f32> {
    let (l_out, k, k_g) = (cfg.l_out(), cfg.k, cfg.k_g());
    let c_in_pg = cfg.c_in_pg();
    let mut cols = vec![0.0f32; l_out * k_g];
    for l in 0..l_out {
        for ci_local in 0..c_in_pg {
            let ci = g * c_in_pg + ci_local;
            let base = (b * cfg.c_in + ci) * cfg.l_in;
            for kk in 0..k {
                let p = l as isize * cfg.stride as isize + kk as isize * cfg.dilation as isize
                    - cfg.pad_left as isize;
                if p >= 0 && (p as usize) < cfg.l_in {
                    cols[l * k_g + (ci_local * k + kk)] = x[base + p as usize];
                }
            }
        }
    }
    cols
}

/// Forward `Y[B, C_out, L_out] = s ⊙ conv1d(X, W)`. `scale` is `[C_out]` (ternary: per-output-channel
/// AbsMean; fp/decoder: all ones). Row-major `X:[B, C_in, L_in]`, `W:[C_out, (C_in/groups)·K]`.
#[must_use]
pub fn forward(x: &[f32], w: &[f32], scale: &[f32], cfg: &Conv1dCfg) -> Vec<f32> {
    let (l_out, n_g, k_g) = (cfg.l_out(), cfg.n_g(), cfg.k_g());
    debug_assert!(cfg.buffers_fit(x.len(), w.len(), scale.len(), cfg.batch * cfg.c_out * l_out));
    let mut out = vec![0.0f32; cfg.batch * cfg.c_out * l_out];
    for b in 0..cfg.batch {
        for g in 0..cfg.groups {
            let cols = im2col(x, cfg, b, g);
            let w_g = &w[g * n_g * k_g..(g * n_g + n_g) * k_g];
            let s_g = &scale[g * n_g..g * n_g + n_g];
            // Y_bg[L_out, N_g] = s ⊙ (patches · W_gᵀ).
            let y_bg = matmul::forward(&cols, w_g, s_g, l_out, n_g, k_g);
            // Scatter to [b, C_out, L_out]; output positions are disjoint (no accumulation).
            for n in 0..n_g {
                let co = g * n_g + n;
                let dst = (b * cfg.c_out + co) * l_out;
                for l in 0..l_out {
                    out[dst + l] = y_bg[l * n_g + n];
                }
            }
        }
    }
    out
}

/// vjp returning `[gX, gW, gScale]` (same shapes as `x`, `w`, `scale`).
///
/// The input grad is a **col2im scatter** — the transpose of im2col — whose overlapping windows
/// accumulate into shared `gX` cells in the pinned order `l → ci_local → kk`, so the sum is
/// bit-reproducible on every backend. `gW`/`gScale` sum over batch (the matmul vjp already sums over
/// the `L_out` positions within a group).
#[must_use]
pub fn vjp(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    cfg: &Conv1dCfg,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let (l_out, n_g, k_g) = (cfg.l_out(), cfg.n_g(), cfg.k_g());
    let c_in_pg = cfg.c_in_pg();
    let mut g_x = vec![0.0f32; cfg.batch * cfg.c_in * cfg.l_in];
    let mut g_w = vec![0.0f32; cfg.c_out * k_g];
    let mut g_s = vec![0.0f32; cfg.c_out];
    for b in 0..cfg.batch {
        for g in 0..cfg.groups {
            let cols = im2col(x, cfg, b, g);
            let w_g = &w[g * n_g * k_g..(g * n_g + n_g) * k_g];
            let s_g = &scale[g * n_g..g * n_g + n_g];
            // Gather the cotangent for this (b, g): gY_bg[L_out, N_g].
            let mut gy_bg = vec![0.0f32; l_out * n_g];
            for n in 0..n_g {
                let co = g * n_g + n;
                let src = (b * cfg.c_out + co) * l_out;
                for l in 0..l_out {
                    gy_bg[l * n_g + n] = grad_out[src + l];
                }
            }
            let grads = matmul::vjp(&cols, w_g, s_g, l_out, n_g, k_g, &gy_bg);
            let (gcols, gw_g, gs_g) = (&grads[0], &grads[1], &grads[2]);
            // gW / gScale accumulate over batch.
            for n in 0..n_g {
                let co = g * n_g + n;
                g_s[co] += gs_g[n];
                for j in 0..k_g {
                    g_w[co * k_g + j] += gw_g[n * k_g + j];
                }
            }
            // gX = col2im(gcols): scatter each patch element back to its input position, accumulating
            // overlaps in the canonical order l → ci_local → kk. Padding taps (p out of range) drop.
            for l in 0..l_out {
                for ci_local in 0..c_in_pg {
                    let ci = g * c_in_pg + ci_local;
                    let base = (b * cfg.c_in + ci) * cfg.l_in;
                    for kk in 0..cfg.k {
                        let p = l as isize * cfg.stride as isize
                            + kk as isize * cfg.dilation as isize
                            - cfg.pad_left as isize;
                        if p >= 0 && (p as usize) < cfg.l_in {
                            g_x[base + p as usize] += gcols[l * k_g + (ci_local * cfg.k + kk)];
                        }
                    }
                }
            }
        }
    }
    vec![g_x, g_w, g_s]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    #[test]
    fn pointwise_is_a_matmul() {
        // K=1, groups=1 ⇒ conv is a [C_out, C_in] matmul applied at every position.
        let cfg = Conv1dCfg {
            batch: 1,
            c_in: 2,
            c_out: 1,
            l_in: 3,
            k: 1,
            stride: 1,
            dilation: 1,
            pad_left: 0,
            pad_right: 0,
            groups: 1,
        };
        // x[2,3] = [[1,2,3],[4,5,6]], w[1, 2*1] = [10, 100] ⇒ y[l] = 10·x0[l] + 100·x1[l].
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [10.0, 100.0];
        let y = forward(&x, &w, &ones(1), &cfg);
        assert_eq!(y, vec![410.0, 520.0, 630.0]);
    }

    #[test]
    fn depthwise_causal_conv_matches_hand_calc() {
        // Depthwise K=3, left-pad 2 (causal), 1 channel: y[l] = Σ_kk w[kk]·x[l-2+kk].
        let cfg = Conv1dCfg {
            batch: 1,
            c_in: 1,
            c_out: 1,
            l_in: 4,
            k: 3,
            stride: 1,
            dilation: 1,
            pad_left: 2,
            pad_right: 0,
            groups: 1,
        };
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 2.0, 3.0]; // taps
        // l=0: w0·x[-2]+w1·x[-1]+w2·x0 = 3·1 = 3
        // l=1: w1·x0+w2·x1 = 2·1+3·2 = 8
        // l=2: w0·x0+w1·x1+w2·x2 = 1+4+9 = 14
        // l=3: w0·x1+w1·x2+w2·x3 = 2+6+12 = 20
        let y = forward(&x, &w, &ones(1), &cfg);
        assert_eq!(y, vec![3.0, 8.0, 14.0, 20.0]);
    }

    #[test]
    fn grouped_splits_channels() {
        // groups=2, C_in=2, C_out=2, K=1: each output channel sees only its own input channel.
        let cfg = Conv1dCfg {
            batch: 1,
            c_in: 2,
            c_out: 2,
            l_in: 2,
            k: 1,
            stride: 1,
            dilation: 1,
            pad_left: 0,
            pad_right: 0,
            groups: 2,
        };
        let x = [1.0, 2.0, 3.0, 4.0]; // ch0=[1,2], ch1=[3,4]
        let w = [5.0, 7.0]; // co0 uses ci0 ·5, co1 uses ci1 ·7
        let y = forward(&x, &w, &ones(2), &cfg);
        assert_eq!(y, vec![5.0, 10.0, 21.0, 28.0]);
    }

    #[test]
    fn l_out_and_buffers_fit() {
        let cfg = Conv1dCfg {
            batch: 2,
            c_in: 4,
            c_out: 6,
            l_in: 10,
            k: 3,
            stride: 2,
            dilation: 2,
            pad_left: 1,
            pad_right: 1,
            groups: 2,
        };
        // eff = 2*(3-1)+1 = 5; padded = 12; (12-5)/2+1 = 4.
        assert_eq!(cfg.l_out(), 4);
        assert_eq!(cfg.k_g(), (4 / 2) * 3);
        assert!(cfg.buffers_fit(2 * 4 * 10, 6 * cfg.k_g(), 6, 2 * 6 * 4));
        assert!(!cfg.buffers_fit(0, 0, 0, 0));
    }
}
