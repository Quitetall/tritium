//! `Projection`: a linear projection backed by a deployed BitNet ternary weight,
//! host-packed additive SALT planes, a physically encoded resident SALT V2
//! allocation, or a dense fp32 reference weight.
//!
//! All variants expose the same `forward(backend, act, m, out)`, so a
//! [`TransformerBlock`](crate::layers::TransformerBlock) runs either without
//! changing its forward body. Only an all-ternary model can build the
//! device-resident CUDA decoder — [`Projection::as_ternary`] gates that, and a
//! model carrying any SALT or dense projection falls back to the host-orchestrated
//! forward (correct, just not graph-accelerated).

#[cfg(feature = "cuda")]
use std::sync::Arc;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{DenseLinear, SaltLinear, TernaryLinear};

/// Activation arithmetic performed before a projection's weight contraction.
///
/// These variants name implemented numeric paths, not campaign evidence aliases:
/// [`F32`](Self::F32) must never be reported as the planned A16 rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionActivationMode {
    /// Consume fp32 activations without per-token quantization.
    F32,
    /// Quantize each activation row to signed int8 with an absmax scale.
    A8,
}

/// A linear projection: deployed ternary, additive SALT, or dense fp32.
#[allow(missing_debug_implementations)] // `TernaryLinear` holds `&dyn DeviceBuffer`
pub enum Projection {
    /// The deployed ternary weight (TQ2_0 on device + per-channel scales).
    Ternary(TernaryLinear),
    /// Packed additive SALT planes, executed without a retained fp32 matrix.
    Salt(SaltLinear),
    /// A SALT V2 matrix retained in its physical codec on a CUDA device.
    #[cfg(feature = "cuda")]
    SaltV2(Arc<tritium_cuda::SaltV2ResidentTensor>),
    /// A dense fp32 weight (normally the fp master/reference path).
    Dense(DenseLinear),
}

impl Projection {
    /// Activation arithmetic used by this projection.
    #[must_use]
    pub fn activation_mode(&self) -> ProjectionActivationMode {
        match self {
            Projection::Ternary(_) | Projection::Salt(_) => ProjectionActivationMode::A8,
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => ProjectionActivationMode::F32,
            Projection::Dense(linear) => {
                if linear.quantizes_activations() {
                    ProjectionActivationMode::A8
                } else {
                    ProjectionActivationMode::F32
                }
            }
        }
    }

    /// Validate retained fp32 projection parameter geometry and finiteness.
    ///
    /// Packed SALT constructors validate every retained scale before publishing
    /// their immutable storage. Resident SALT V2 handles are likewise validated
    /// when uploaded, so only dense coefficients and deployed ternary row scales
    /// require inspection here.
    pub(crate) fn validate_retained_parameters(&self) -> Result<(), NnError> {
        match self {
            Projection::Ternary(linear) => {
                linear.validate_retained_geometry()?;
                if linear.scales.iter().any(|value| !value.is_finite()) {
                    return Err(NnError::Backend(
                        "ternary projection scales contain a non-finite value".to_owned(),
                    ));
                }
            }
            Projection::Dense(linear) => {
                linear.validate_retained_geometry()?;
                if linear.weights.iter().any(|value| !value.is_finite()) {
                    return Err(NnError::Backend(
                        "dense projection weights contain a non-finite value".to_owned(),
                    ));
                }
            }
            Projection::Salt(_) => {}
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => {}
        }
        Ok(())
    }

    /// Forward through whichever projection this is. Deployed ternary and resident
    /// SALT V2 weights use `backend`; host-packed SALT and dense weights run on the host.
    ///
    /// # Errors
    /// Propagates the underlying projection's [`NnError`]. Resident SALT V2 returns
    /// [`NnError::Backend`] if `backend` is not the owning CUDA context.
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
            #[cfg(feature = "cuda")]
            Projection::SaltV2(tensor) => salt_v2_forward_exact(backend, tensor, act, m, out),
            Projection::Dense(l) => l.forward(act, m, out),
        }
    }

    /// Output feature count `N`.
    pub fn n_out(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.n_out,
            Projection::Salt(l) => l.n_out(),
            #[cfg(feature = "cuda")]
            Projection::SaltV2(tensor) => tensor.rows(),
            Projection::Dense(l) => l.n_out,
        }
    }

    /// Input feature count `K`.
    pub fn k_in(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.k_in,
            Projection::Salt(l) => l.k_in(),
            #[cfg(feature = "cuda")]
            Projection::SaltV2(tensor) => tensor.columns(),
            Projection::Dense(l) => l.k_in,
        }
    }

    /// The deployed ternary projection, or `None` for SALT/dense weights. The
    /// device-resident decoder needs every projection in this form.
    pub fn as_ternary(&self) -> Option<&TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Salt(_) | Projection::Dense(_) => None,
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => None,
        }
    }

    /// Mutable access to the deployed ternary projection, or `None` for SALT/dense
    /// weights. Used by QAT healing to swap a re-trained weight back in (plan 0010).
    pub fn as_ternary_mut(&mut self) -> Option<&mut TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Salt(_) | Projection::Dense(_) => None,
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => None,
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn salt_v2_cuda_backend(
    backend: &dyn TernaryBackend,
) -> Result<&tritium_cuda::CudaBackend, NnError> {
    backend
        .as_concrete()
        .and_then(|concrete| concrete.downcast_ref::<tritium_cuda::CudaBackend>())
        .ok_or_else(|| {
            NnError::Backend(
                "resident SALT V2 weights require a backend-aware CUDA execution path".into(),
            )
        })
}

#[cfg(feature = "cuda")]
pub(crate) fn salt_v2_forward_exact(
    backend: &dyn TernaryBackend,
    tensor: &tritium_cuda::SaltV2ResidentTensor,
    act: &[f32],
    m: usize,
    out: &mut [f32],
) -> Result<(), NnError> {
    let cuda = salt_v2_cuda_backend(backend)?;
    match cuda.salt_v2_forward_exact_into(tensor, act, m, out) {
        Ok(_receipt) => Ok(()),
        Err(tritium_spec::BackendError::ShapeMismatch { expected, got }) => {
            Err(NnError::Shape { expected, got })
        }
        Err(error) => Err(NnError::Backend(error.to_string())),
    }
}
