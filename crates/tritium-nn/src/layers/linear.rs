//! `TernaryLinear`: a bias-free linear layer whose weight is ternary `{-1,0,+1}`.
//!
//! The weight is stored on-device (packed, uploaded once via the backend) with a
//! per-output-channel scale; the forward pass quantizes activations to int8
//! ([`crate::ops::quantize_activation_int8`]) and calls
//! [`tritium_spec::TernaryBackend::mpgemm`]. BitNet has no biases.
//!
//! # W1.58A8 dequant fold
//!
//! [`new`](TernaryLinear::new) re-packs the `[N, K]` ternary weight to TQ2_0 with
//! a **block scale of `1.0`** (the per-channel weight scale is carried separately
//! in [`scales`](TernaryLinear::scales), matching the backend's
//! packing-vs-scaling split) and uploads it once. The forward pass then realises
//!
//! ```text
//! y[m, n] = weight_scale · act_scale[m] · Σ_k q_act[m, k] · trit[n, k]
//! ```
//!
//! in two steps: the backend [`mpgemm`](tritium_spec::TernaryBackend::mpgemm)
//! folds `weight_scale` (it is `scales[n]`), and the forward multiplies each
//! output row by its per-token activation dequant scale `act_scale[m]`.

use half::f16;
use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};
use tritium_spec::{DeviceBuffer, TernaryBackend};

use crate::error::NnError;
use crate::ops::quantize_activation_int8;

/// A ternary linear projection `y = (act_q · Wᵀ) · scale`, `W` shape `[n_out, k_in]`.
///
/// Owns the uploaded device weight handle and the per-output-channel scales; the
/// caller supplies the backend at construction and forward time.
#[allow(missing_debug_implementations)]
pub struct TernaryLinear {
    /// Output feature count (`N`, rows of the weight).
    pub n_out: usize,
    /// Input feature count (`K`, columns of the weight).
    pub k_in: usize,
    /// Per-output-channel scale `scales[n]` (the I2_S row scale × the A8 fold);
    /// length `n_out`.
    pub scales: Vec<f32>,
    /// Opaque device handle to the uploaded packed ternary weight.
    pub weights: Box<dyn DeviceBuffer>,
}

impl TernaryLinear {
    /// Build a layer from in-memory ternary weights and a single per-tensor scale.
    ///
    /// `trits` is the `[n_out, k_in]` (output-major, row-major) ternary weight and
    /// `weight_scale` is the one scalar that dequantizes every output channel
    /// (BitNet's per-tensor weight scale). Each row is re-packed to TQ2_0 with a
    /// block scale of `1.0` via [`pack_tq2_0_row`], uploaded once through
    /// `backend`, and `weight_scale` is broadcast into
    /// [`scales`](TernaryLinear::scales) (length `n_out`) so the backend GEMM
    /// applies it per output channel.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `trits.len() != n_out * k_in`, or [`NnError::Backend`]
    /// if packing or the backend upload fails.
    pub fn new(
        backend: &dyn TernaryBackend,
        trits: &[Trit],
        n_out: usize,
        k_in: usize,
        weight_scale: f32,
    ) -> Result<Self, NnError> {
        let expected = n_out.checked_mul(k_in).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: trits.len(),
        })?;
        if trits.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: trits.len(),
            });
        }

        // Per-row TQ2_0: one f16 block scale per 256-trit block, all 1.0 so the
        // unpacked trits are the raw weights (the per-channel scale is applied by
        // the backend from `scales`, never from the block scale).
        let nb = num_blocks(k_in);
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let block_scales = vec![f16::ONE; nb];
        let mut packed = vec![0u8; n_out * row_bytes];
        for (row, out_row) in trits
            .chunks_exact(k_in.max(1))
            .zip(packed.chunks_mut(row_bytes))
        {
            // `k_in == 0` would make `chunks_exact` ill-defined; reject it as a
            // shape error rather than packing an empty layer.
            if k_in == 0 {
                return Err(NnError::Shape {
                    expected: 1,
                    got: 0,
                });
            }
            pack_tq2_0_row(row, &block_scales, out_row)
                .map_err(|e| NnError::Backend(e.to_string()))?;
        }

        let shape = GemmShape::new(0, n_out, k_in);
        let weights = backend.upload_weights(&packed, shape, TernaryFormat::Tq2_0)?;

        Ok(Self {
            n_out,
            k_in,
            scales: vec![weight_scale; n_out],
            weights,
        })
    }

    /// GEMM shape for an `m`-row activation batch through this layer.
    #[must_use]
    pub fn shape(&self, m: usize) -> GemmShape {
        GemmShape::new(m, self.n_out, self.k_in)
    }

    /// Forward: quantize `act` (`[m, k_in]`) to int8 per token, run the ternary
    /// GEMM on `backend`, and write `[m, n_out]` into `out`.
    ///
    /// # Errors
    /// [`NnError::Shape`] on buffer-length mismatch, or [`NnError::Backend`] if
    /// the backend GEMM fails.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        act: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        let act_len = m * self.k_in;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m * self.n_out;
        if out.len() != out_len {
            return Err(NnError::Shape {
                expected: out_len,
                got: out.len(),
            });
        }

        // A8: per-token int8 absmax quant. `q_act` is the int8 values kept as f32
        // (the f32 mpGEMM consumes them directly); `act_scale[r]` is the per-token
        // dequant multiplier folded into the output below.
        let mut q_act = vec![0.0f32; act_len];
        let mut act_scale = vec![0.0f32; m];
        quantize_activation_int8(act, m, self.k_in, &mut q_act, &mut act_scale)?;

        // Ternary GEMM: out[r, n] = scales[n] · Σ_k q_act[r, k] · trit[n, k].
        // `scales` already carries the per-tensor weight scale.
        backend.mpgemm(
            &q_act,
            &*self.weights,
            &self.scales,
            self.shape(m),
            TernaryFormat::Tq2_0,
            out,
        )?;

        // Fold the per-token activation dequant scale: y · act_scale · weight_scale
        // (weight_scale already applied by the GEMM).
        for r in 0..m {
            let s = act_scale[r];
            for slot in &mut out[r * self.n_out..r * self.n_out + self.n_out] {
                *slot *= s;
            }
        }

        Ok(())
    }
}
