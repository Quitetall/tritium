//! `TernaryLinear`: a bias-free linear layer whose weight is ternary `{-1,0,+1}`.
//!
//! The weight is stored on-device (packed, uploaded once via the backend) with a
//! per-output-channel scale; the forward pass quantizes activations to int8
//! ([`crate::ops::quantize_activation_int8`]) and calls
//! [`tritium_spec::TernaryBackend::mpgemm`]. BitNet has no biases. Real wiring
//! lands in WF-3.

use tritium_core::GemmShape;
use tritium_spec::{DeviceBuffer, TernaryBackend};

use crate::error::NnError;

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
        let _ = (backend, act, m, out);
        todo!("WF-3: A8 quant + ternary mpgemm forward")
    }
}
