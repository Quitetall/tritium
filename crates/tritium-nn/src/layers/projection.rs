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

use std::sync::Arc;

use tritium_format::salt_v2_package::SaltV2ScaleUpdate;
use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{DenseLinear, HostSaltV2Linear, Q2Linear, SaltLinear, TernaryLinear};

/// Activation arithmetic performed before a projection's weight contraction.
///
/// These variants name implemented numeric paths, not campaign evidence aliases:
/// [`F32`](Self::F32) must never be reported as the planned A16 rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProjectionActivationMode {
    /// Consume fp32 activations without per-token quantization.
    F32,
    /// Quantize each activation row to signed int8 with an absmax scale.
    A8,
}

/// A linear projection: deployed ternary, additive SALT, or dense fp32.
///
/// `#[non_exhaustive]`: the `SaltV2` variant only exists under the `cuda`
/// feature, so a downstream exhaustive match would compile cpu-only and
/// break the moment feature unification enables `cuda` anywhere in the dep
/// graph — always carry a wildcard arm.
#[allow(missing_debug_implementations)] // `TernaryLinear` holds `&dyn DeviceBuffer`
#[non_exhaustive]
pub enum Projection {
    /// The deployed ternary weight (TQ2_0 on device + per-channel scales).
    Ternary(TernaryLinear),
    /// Standard GGUF Q2_0 retained packed with one scale per 64 coefficients.
    Q2(Q2Linear),
    /// Packed additive SALT planes, executed without a retained fp32 matrix.
    Salt(SaltLinear),
    /// Compact SALT V2 codec payload executed on portable host arithmetic.
    HostSaltV2(Arc<HostSaltV2Linear>),
    /// A SALT V2 matrix retained in its physical codec on a CUDA device.
    #[cfg(feature = "cuda")]
    SaltV2(Arc<tritium_cuda::SaltV2ResidentTensor>),
    /// A dense fp32 weight (normally the fp master/reference path).
    Dense(DenseLinear),
}

impl Projection {
    /// Apply scale-only candidate updates to a uniquely owned host SALT V2
    /// resident matrix. Packed trits stay in place; CUDA residents and shared
    /// host handles fail closed until their mutation paths are explicit.
    ///
    /// # Errors
    /// Returns an error for non-host-SALT projections, shared resident handles,
    /// or malformed updates. Validation occurs before the resident scales change.
    pub fn apply_host_salt_v2_scale_updates(
        &mut self,
        tensor_index: usize,
        updates: &[SaltV2ScaleUpdate],
    ) -> Result<(), NnError> {
        match self {
            Projection::HostSaltV2(matrix) => {
                if matrix.tensor_index() != tensor_index {
                    return Err(NnError::Backend(format!(
                        "SALT V2 scale update targets tensor {tensor_index}, but this projection is tensor {}",
                        matrix.tensor_index()
                    )));
                }
                Arc::get_mut(matrix)
                    .ok_or_else(|| {
                        NnError::Backend(
                            "cannot update shared host SALT V2 resident scales in place".into(),
                        )
                    })?
                    .apply_scale_updates(tensor_index, updates)
            }
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => Err(NnError::Backend(
                "in-place candidate scale updates are not implemented for CUDA SALT V2".into(),
            )),
            _ => Err(NnError::Backend(
                "scale updates require a host SALT V2 projection".into(),
            )),
        }
    }

    /// Activation arithmetic used by this projection.
    #[must_use]
    pub fn activation_mode(&self) -> ProjectionActivationMode {
        match self {
            Projection::Ternary(_) | Projection::Q2(_) | Projection::Salt(_) => {
                ProjectionActivationMode::A8
            }
            Projection::HostSaltV2(_) => ProjectionActivationMode::F32,
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

    /// Validate retained projection geometry without scanning parameter values.
    pub(crate) fn validate_retained_geometry(&self) -> Result<(), NnError> {
        match self {
            Projection::Ternary(linear) => linear.validate_retained_geometry(),
            Projection::Q2(linear) => linear.validate_retained_geometry(),
            Projection::Dense(linear) => linear.validate_retained_geometry(),
            Projection::Salt(_) | Projection::HostSaltV2(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => Ok(()),
        }
    }

    /// Validate retained fp32 projection parameter geometry and finiteness.
    ///
    /// Packed SALT constructors validate every retained scale before publishing
    /// their immutable storage. Resident SALT V2 handles are likewise validated
    /// when uploaded, so only dense coefficients and deployed ternary row scales
    /// require inspection here.
    pub(crate) fn validate_retained_parameters(&self) -> Result<(), NnError> {
        self.validate_retained_geometry()?;
        match self {
            Projection::Ternary(linear) => {
                if linear.scales.iter().any(|value| !value.is_finite()) {
                    return Err(NnError::Backend(
                        "ternary projection scales contain a non-finite value".to_owned(),
                    ));
                }
            }
            Projection::Q2(_) => {
                // Constructor validates every packed code and scale.
            }
            Projection::Dense(linear) => {
                if linear.weights.iter().any(|value| !value.is_finite()) {
                    return Err(NnError::Backend(
                        "dense projection weights contain a non-finite value".to_owned(),
                    ));
                }
            }
            Projection::Salt(_) => {
                // The packed constructor validates every retained row scale.
            }
            Projection::HostSaltV2(_) => {
                // Strict package decode validates payload and scales before construction.
            }
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => {
                // Upload validates the immutable resident allocation.
            }
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
            Projection::Q2(l) => l.forward(act, m, out),
            Projection::Salt(l) => l.forward(act, m, out),
            Projection::HostSaltV2(l) => l.forward(act, m, out),
            #[cfg(feature = "cuda")]
            Projection::SaltV2(tensor) => salt_v2_forward_exact(backend, tensor, act, m, out),
            Projection::Dense(l) => l.forward(act, m, out),
        }
    }

    /// Output feature count `N`.
    pub fn n_out(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.n_out,
            Projection::Q2(l) => l.n_out(),
            Projection::Salt(l) => l.n_out(),
            Projection::HostSaltV2(l) => l.rows(),
            #[cfg(feature = "cuda")]
            Projection::SaltV2(tensor) => tensor.rows(),
            Projection::Dense(l) => l.n_out,
        }
    }

    /// Input feature count `K`.
    pub fn k_in(&self) -> usize {
        match self {
            Projection::Ternary(l) => l.k_in,
            Projection::Q2(l) => l.k_in(),
            Projection::Salt(l) => l.k_in(),
            Projection::HostSaltV2(l) => l.columns(),
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
            Projection::Q2(_)
            | Projection::Salt(_)
            | Projection::HostSaltV2(_)
            | Projection::Dense(_) => None,
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => None,
        }
    }

    /// Mutable access to the deployed ternary projection, or `None` for SALT/dense
    /// weights. Used by QAT healing to swap a re-trained weight back in (plan 0010).
    pub fn as_ternary_mut(&mut self) -> Option<&mut TernaryLinear> {
        match self {
            Projection::Ternary(l) => Some(l),
            Projection::Q2(_)
            | Projection::Salt(_)
            | Projection::HostSaltV2(_)
            | Projection::Dense(_) => None,
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => None,
        }
    }
}

/// Whether the tolerance-gated SALT V2 forward is enabled.
#[cfg(feature = "cuda")]
fn salt_v2_fast_enabled() -> bool {
    match std::env::var("TRITIUM_SALT_V2_FAST") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Ok(value) => {
            eprintln!(
                "tritium-nn: TRITIUM_SALT_V2_FAST={value:?} - use 1 or 0 (unset = 0); reading as 0"
            );
            false
        }
        Err(_) => false,
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
    // `TRITIUM_SALT_V2_FAST=1` reduces each row by warp shuffle instead of
    // replaying the scalar kernel's addition order. That reassociates the K-sum,
    // so it is opt-in: the exact kernel stays the default while the fast one's
    // effect on model output is being measured, and it is gated on relative
    // error against the CPU reference rather than equality.
    let result = if salt_v2_fast_enabled() {
        cuda.salt_v2_forward_fast_into(tensor, act, m, out)
    } else {
        cuda.salt_v2_forward_exact_into(tensor, act, m, out)
    };
    match result {
        Ok(_receipt) => Ok(()),
        Err(tritium_spec::BackendError::ShapeMismatch { expected, got }) => {
            Err(NnError::Shape { expected, got })
        }
        Err(error) => Err(NnError::Backend(error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use half::f16;
    use tritium_format::{
        salt_v2::SaltV2Codec,
        salt_v2_package::{
            SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2ScaleUpdate, SaltV2Tensor,
            SaltV2Tile, write_salt_v2_package,
        },
    };

    use super::Projection;
    use crate::layers::HostSaltV2Linear;

    fn host_matrix() -> HostSaltV2Linear {
        let tensor = SaltV2Tensor::new(
            "weight",
            vec![2, 128],
            vec![
                SaltV2Tile::new(vec![
                    SaltV2Plane::new(vec![1; 256], vec![f16::ONE, f16::ONE]).unwrap(),
                ])
                .unwrap(),
            ],
        )
        .unwrap();
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).unwrap();
        let encoded = write_salt_v2_package(&package).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        HostSaltV2Linear::from_reader(&mut reader, "weight").unwrap()
    }

    #[test]
    fn resident_scale_updates_mutate_only_unique_host_projection() {
        let mut projection = Projection::HostSaltV2(Arc::new(host_matrix()));
        let update = SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::from_f32(2.0), f16::ONE]).unwrap();
        projection
            .apply_host_salt_v2_scale_updates(0, std::slice::from_ref(&update))
            .unwrap();
        let matrix = match &projection {
            Projection::HostSaltV2(matrix) => matrix,
            _ => unreachable!(),
        };
        let mut output = [0.0; 2];
        matrix.forward(&vec![1.0; 128], 1, &mut output).unwrap();
        assert_eq!(output, [256.0, 128.0]);

        let mismatched = SaltV2ScaleUpdate::new(1, 0, 0, vec![f16::ONE, f16::ONE]).unwrap();
        assert!(
            projection
                .apply_host_salt_v2_scale_updates(1, std::slice::from_ref(&mismatched))
                .is_err()
        );

        let shared = Arc::new(host_matrix());
        let _other_owner = Arc::clone(&shared);
        let mut shared_projection = Projection::HostSaltV2(shared);
        assert!(
            shared_projection
                .apply_host_salt_v2_scale_updates(3, std::slice::from_ref(&update))
                .is_err()
        );
    }
}
