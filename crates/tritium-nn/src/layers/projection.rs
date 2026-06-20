//! `Projection`: a linear projection that is either **ternary** (the deployed
//! BitNet weight) or **dense fp32** (the SALT / fp reference for the accuracy
//! curve, ADR 0006).
//!
//! Both variants expose the same `forward(backend, act, m, out)`, so a
//! [`TransformerBlock`](crate::layers::TransformerBlock) runs either without
//! changing its forward body. Only an all-ternary model can build the
//! device-resident CUDA decoder — [`Projection::as_ternary`] gates that, and a
//! model carrying any dense projection falls back to the host-orchestrated
//! forward (correct, just not graph-accelerated).

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{DenseLinear, TernaryLinear};

/// A linear projection: ternary (on-device, packed) or dense fp32 (host).
#[allow(missing_debug_implementations)] // `TernaryLinear` holds `&dyn DeviceBuffer`
pub enum Projection {
    /// The deployed ternary weight (TQ2_0 on device + per-channel scales).
    Ternary(TernaryLinear),
    /// A dense fp32 weight (fp master, or a SALT plane-stack dequantized).
    Dense(DenseLinear),
}

impl Projection {
    /// Forward through whichever projection this is. The `backend` is used only by
    /// the ternary GEMM; the dense path is a host matmul and ignores it.
    ///
    /// # Errors
    /// Propagates the underlying projection's [`NnError`].
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        act: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        match self {
            Projection::Ternary(l) => l.forward(backend, act, m, out),
            Projection::Dense(l) => l.forward(act, m, out),
        }
    }

    /// Output feature count `N`.
    pub fn n_out(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.n_out,
            Projection::Dense(l) => l.n_out,
        }
    }

    /// Input feature count `K`.
    pub fn k_in(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.k_in,
            Projection::Dense(l) => l.k_in,
        }
    }

    /// The ternary projection, or `None` if dense. The device-resident decoder
    /// needs every projection to be ternary; a single dense one disables it.
    pub fn as_ternary(&self) -> Option<&TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Dense(_) => None,
        }
    }

    /// Mutable access to the ternary projection, or `None` if dense. Used by QAT
    /// healing to swap a re-trained ternary weight back in (plan 0010).
    pub fn as_ternary_mut(&mut self) -> Option<&mut TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Dense(_) => None,
        }
    }
}
