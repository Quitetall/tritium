//! `Projection`: a linear projection backed by a deployed BitNet ternary weight,
//! packed additive SALT planes, or a dense fp32 reference weight.
//!
//! All variants expose the same `forward(backend, act, m, out)`, so a
//! [`TransformerBlock`](crate::layers::TransformerBlock) runs either without
//! changing its forward body. Only an all-ternary model can build the
//! device-resident CUDA decoder — [`Projection::as_ternary`] gates that, and a
//! model carrying any SALT or dense projection falls back to the host-orchestrated
//! forward (correct, just not graph-accelerated).

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{DenseLinear, SaltLinear, TernaryLinear};

/// A linear projection: deployed ternary, additive SALT, or dense fp32.
#[allow(missing_debug_implementations)] // `TernaryLinear` holds `&dyn DeviceBuffer`
pub enum Projection {
    /// The deployed ternary weight (TQ2_0 on device + per-channel scales).
    Ternary(TernaryLinear),
    /// Packed additive SALT planes, executed without a retained fp32 matrix.
    Salt(SaltLinear),
    /// A dense fp32 weight (normally the fp master/reference path).
    Dense(DenseLinear),
}

impl Projection {
    /// Forward through whichever projection this is. The `backend` is used only by
    /// the deployed ternary GEMM; the SALT and dense paths run on the host.
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
            Projection::Salt(l) => l.forward(act, m, out),
            Projection::Dense(l) => l.forward(act, m, out),
        }
    }

    /// Output feature count `N`.
    pub fn n_out(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.n_out,
            Projection::Salt(l) => l.n_out(),
            Projection::Dense(l) => l.n_out,
        }
    }

    /// Input feature count `K`.
    pub fn k_in(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.k_in,
            Projection::Salt(l) => l.k_in(),
            Projection::Dense(l) => l.k_in,
        }
    }

    /// The deployed ternary projection, or `None` for SALT/dense weights. The
    /// device-resident decoder needs every projection in this form.
    pub fn as_ternary(&self) -> Option<&TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Salt(_) | Projection::Dense(_) => None,
        }
    }

    /// Mutable access to the deployed ternary projection, or `None` for SALT/dense
    /// weights. Used by QAT healing to swap a re-trained weight back in (plan 0010).
    pub fn as_ternary_mut(&mut self) -> Option<&mut TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Salt(_) | Projection::Dense(_) => None,
        }
    }
}
