//! # tritium-spec
//!
//! Versioned contracts every Tritium execution backend implements. This crate
//! owns [`TernaryBackend`], its buffer/capability vocabulary, and the frozen
//! versioned portable-training registries. No implementations live
//! here — device crates provide adapters and `tritium-runtime` selects them.
//!
//! ## Object safety
//!
//! [`TernaryBackend`] is deliberately **object-safe** so the runtime can hold a
//! `Box<dyn TernaryBackend>` registry of heterogeneous devices. That rules out an
//! associated `Buffer` type (which would make `dyn TernaryBackend` impossible), so
//! device memory is carried as a boxed [`DeviceBuffer`] trait object and each
//! backend downcasts it to its concrete buffer via [`core::any::Any`].
//!
//! ## Packing
//!
//! Weight packing (TQ1_0/TQ2_0) is **not** part of this trait — it lives host-side
//! in `tritium-format`, the single source of truth. A backend receives
//! already-packed bytes through [`TernaryBackend::upload_weights`].
#![forbid(unsafe_code)]
// v0.90 hardening: every public item must carry a doc comment.
#![deny(missing_docs)]

use core::any::Any;
use core::fmt;

// Re-export the core vocabulary types the contract speaks so a backend author
// needs only depend on `tritium-spec` to implement [`TernaryBackend`].
pub use tritium_core::{DType, GemmShape, TernaryFormat, TritError};

mod caps;
pub use caps::DeviceCaps;
mod train_backend;
pub use train_backend::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainLimitsV1, TrainNamedBufferMutV1, TrainNamedBufferRefV1,
    TrainOperationErrorV1, TrainOutputV1, TrainOwnedBufferDataV1, TrainOwnedBufferV1,
    TrainReceiptV1, TrainRequestError, TrainRequestV1, train_output_digest_v1,
    train_request_digest_v1,
};
mod training;
pub use training::{
    TrainingManifestError, TrainingOpCategoryV1, TrainingOpDescriptorV1, TrainingOpManifestV1,
    TrainingOpManifestV2, TrainingOpManifestV3, TrainingVjpV1,
};
mod training_vectors;
pub use training_vectors::{
    TrainingToleranceV1, TrainingVectorAttributeV1, TrainingVectorAttributeValueV1,
    TrainingVectorBufferDataV1, TrainingVectorBufferV1, TrainingVectorCaseV1, TrainingVectorError,
    TrainingVectorErrorCategoryV1, TrainingVectorExpectedV1, TrainingVectorSetV1,
    TrainingVectorSetV2, TrainingVectorSetV3,
};

/// Parameters for a ternary mixed-precision GEMM call.
///
/// Groups the operands of [`TernaryBackend::mpgemm`] (and
/// [`TernaryBackend::mpgemm_with_act_quant`]) into one value so the call site is
/// a single struct rather than six positional arguments. All slices are
/// row-major and follow the [`GemmShape`] convention: `act` is `[M, K]`, the
/// `weights` buffer is `[N, K]` (output-major) packed in `format`, `scales` is
/// `[N]` per-output-channel, and `out` is `[M, N]` and is overwritten.
pub struct MpGemm<'a> {
    /// `[M, K]` row-major activations.
    pub act: &'a [f32],
    /// Device handle for the packed `[N, K]` ternary weights (from
    /// [`TernaryBackend::upload_weights`]).
    pub weights: &'a dyn DeviceBuffer,
    /// `[N]` per-output-channel scales.
    pub scales: &'a [f32],
    /// Problem geometry `(M, N, K)`.
    pub shape: GemmShape,
    /// Packing scheme of `weights`.
    pub format: TernaryFormat,
    /// `[M, N]` output, overwritten by the call.
    pub out: &'a mut [f32],
}

impl fmt::Debug for MpGemm<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpGemm")
            .field("act_len", &self.act.len())
            .field("weights_bytes", &self.weights.len_bytes())
            .field("scales_len", &self.scales.len())
            .field("shape", &self.shape)
            .field("format", &self.format)
            .field("out_len", &self.out.len())
            .finish()
    }
}

/// Inputs and outputs for the VJP of one packed ternary mpGEMM.
///
/// `grad_projected_weight` is the gradient with respect to the dense projected
/// weight `P[n,k] = scales[n] * trit[n,k]`. Framework adapters apply their
/// estimator-specific STE after this backend-neutral primitive.
pub struct MpGemmProjectedVjp<'a> {
    /// Forward activation `[M, K]`.
    pub act: &'a [f32],
    /// Packed ternary weight `[N, K]`.
    pub weights: &'a dyn DeviceBuffer,
    /// Per-output-channel scale `[N]`.
    pub scales: &'a [f32],
    /// Upstream gradient `[M, N]`.
    pub grad_output: &'a [f32],
    /// Shared matrix dimensions.
    pub shape: GemmShape,
    /// Packed ternary format.
    pub format: TernaryFormat,
    /// Output gradient with respect to `act`, shape `[M, K]`.
    pub grad_act: &'a mut [f32],
    /// Output gradient with respect to projected weight, shape `[N, K]`.
    pub grad_projected_weight: &'a mut [f32],
    /// Optional output bias gradient `[N]`.
    pub grad_bias: Option<&'a mut [f32]>,
}

impl fmt::Debug for MpGemmProjectedVjp<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MpGemmProjectedVjp")
            .field("act_len", &self.act.len())
            .field("weights_len_bytes", &self.weights.len_bytes())
            .field("scales_len", &self.scales.len())
            .field("grad_output_len", &self.grad_output.len())
            .field("shape", &self.shape)
            .field("format", &self.format)
            .field("grad_act_len", &self.grad_act.len())
            .field(
                "grad_projected_weight_len",
                &self.grad_projected_weight.len(),
            )
            .field(
                "grad_bias_len",
                &self.grad_bias.as_ref().map(|values| values.len()),
            )
            .finish()
    }
}

/// Opaque handle to device-resident memory owned by a backend.
///
/// The runtime treats this as an opaque token; the owning backend downcasts it
/// back to its concrete type with [`DeviceBuffer::as_any`]. `'static` is required
/// for `Any`; buffers own their storage so this is no constraint in practice.
pub trait DeviceBuffer: Any + Send + Sync {
    /// Size of the buffer in bytes.
    fn len_bytes(&self) -> usize;

    /// True if the buffer holds no bytes.
    fn is_empty(&self) -> bool {
        self.len_bytes() == 0
    }

    /// Upcast for downcasting back to the concrete buffer type.
    fn as_any(&self) -> &dyn Any;
}

/// A ternary execution backend: uploads packed weights, runs mixed-precision GEMM.
///
/// Implementations must match [`tritium_core::reference_mpgemm`] within the
/// tolerance documented for [`mpgemm`](TernaryBackend::mpgemm). Correctness against
/// the reference is verified by `tritium-testkit`'s conformance harness, so every
/// backend is held to the identical bar.
pub trait TernaryBackend: Send + Sync {
    /// Stable identifier for this backend instance, e.g. `"cpu"` or `"cuda:0"`.
    fn device_id(&self) -> &str;

    /// Physical device identity for release evidence and cross-process admission.
    ///
    /// Backends without a stronger hardware identity use [`Self::device_id`].
    /// CUDA implementations override this with the driver-reported GPU UUID.
    fn physical_device_id(&self) -> &str {
        self.device_id()
    }

    /// What this device can do — used by the runtime to pick a backend.
    fn capabilities(&self) -> DeviceCaps;

    /// Upload host-side packed weight bytes (`format`, shape `[N, K]`) to device
    /// memory, returning an opaque handle for reuse across `mpgemm` calls.
    ///
    /// # Errors
    /// [`BackendError::UnsupportedFormat`] if the backend cannot consume `format`;
    /// [`BackendError::OutOfMemory`] on allocation failure;
    /// [`BackendError::InvalidInput`] if `packed.len()` disagrees with `shape`/`format`.
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError>;

    /// Compute `out[M,N] = scale[n] · Σ_k act[m,k] · w[n,k]` for ternary `w`.
    ///
    /// Operands are carried in [`MpGemm`]: `act` is `[M, K]` row-major, `weights`
    /// is the handle from [`upload_weights`](TernaryBackend::upload_weights),
    /// `scales` is `[N]` per-output-channel, `out` is `[M, N]` and is overwritten.
    ///
    /// Result must match [`tritium_core::reference_mpgemm`] with relative error
    /// `≤ 1e-4` (fp32 accumulation reorders across backends; bit-exactness is not
    /// required for the float path, only for packing).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if buffer lengths disagree with `shape`;
    /// [`BackendError::Backend`] for device-specific failures.
    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError>;

    /// Compute activation, dense projected-weight, and optional bias VJPs for
    /// one packed ternary mpGEMM.
    ///
    /// Backends with a native training path override this method. Default
    /// returns [`BackendError::Backend`], preserving source compatibility for
    /// inference-only third-party backends.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] for invalid operand lengths;
    /// [`BackendError::Backend`] when this backend has no packed VJP path or its
    /// device execution fails.
    fn mpgemm_projected_vjp(&self, _p: MpGemmProjectedVjp<'_>) -> Result<(), BackendError> {
        Err(BackendError::Backend(format!(
            "backend `{}` does not implement packed mpGEMM VJP",
            self.device_id()
        )))
    }

    /// Ternary mpGEMM with **W1.58A8** activation quantization fused in: quantize
    /// `act` to per-token int8 (absmax, `Qp = 127`), contract against the ternary
    /// weights, and fold both the per-token activation scale and the per-channel
    /// `weight_scales` into the `f32` output.
    ///
    /// This is the BitNet linear-layer primitive. `act` is `[M, K]` row-major,
    /// `weights` the handle from [`upload_weights`](TernaryBackend::upload_weights),
    /// `weight_scales` is `[N]` per-output-channel, `out` is `[M, N]` (overwritten).
    /// The result is
    /// `out[m,n] = act_scale[m] · weight_scale[n] · Σ_k q[m,k] · w[n,k]`, where
    /// `q[m,k]` is the int8 quant of `act[m,k]` and `act_scale[m] = γ_m / 127`
    /// (`γ_m = max_k |act[m,k]|`).
    ///
    /// ## Default impl — host path
    ///
    /// The provided default does the quant **on the host** (matching
    /// `tritium-nn`'s `ops::act_quant` / `transformers` `ActQuant`: round-half-to-
    /// even, range `[-128, 127]`, zero-row → zero scale), then delegates to
    /// [`mpgemm`](TernaryBackend::mpgemm) and folds the per-token scale. So every
    /// backend supports this method for free, identically to the v0.20 caller-side
    /// path. A GPU backend **overrides** it to quantize on-device and feed an IMMA
    /// int8 kernel directly, dropping a host pass + an H2D round-trip; the override
    /// must reproduce this default's output within the `mpgemm` tolerance (the
    /// "fused == host-A8" gate of ADR 0005).
    //
    // WF-D: the host quant here is duplicated from `tritium-nn::ops::act_quant`
    // (which cannot be a dependency — it sits above this crate). If the two ever
    // need to share, lift the quant into `tritium-core` as the single numeric
    // truth. The `act_quant_golden` test below pins this copy to the same oracle
    // value the nn copy asserts, so they cannot silently diverge.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if `act`/`out`/`scales` lengths
    /// disagree with `shape`; otherwise whatever
    /// [`mpgemm`](TernaryBackend::mpgemm) returns.
    ///
    /// In the [`MpGemm`], `scales` is the `[N]` per-output-channel **weight**
    /// scale; the per-token activation scale is computed internally.
    fn mpgemm_with_act_quant(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales: weight_scales,
            shape,
            format,
            out,
        } = p;
        let GemmShape { m, n, k } = shape;
        if act.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: act.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }
        if weight_scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: weight_scales.len(),
            });
        }

        // Per-token int8 absmax quant: int8 values kept in f32 (the f32 `mpgemm`
        // consumes them directly), plus the per-token dequant multiplier.
        let mut q = vec![0.0_f32; m * k];
        let mut act_scale = vec![0.0_f32; m];
        quantize_act_int8(act, m, k, &mut q, &mut act_scale);

        // out[m,n] = weight_scale[n] · Σ_k q[m,k] · w[n,k]
        self.mpgemm(MpGemm {
            act: &q,
            weights,
            scales: weight_scales,
            shape,
            format,
            out: &mut *out,
        })?;

        // Fold the per-token activation scale: out[m,n] *= act_scale[m].
        for (row, &s) in out.chunks_exact_mut(n).zip(act_scale.iter()) {
            for v in row {
                *v *= s;
            }
        }
        Ok(())
    }

    /// Downcast hook for a backend-specific fast path. Returns `None` by default;
    /// a backend that exposes a concrete accelerated forward (e.g. the CUDA
    /// device-resident decode, [`tritium_cuda::CudaDecodeModel`]) overrides this to
    /// return `Some(self)` so a caller holding a `&dyn TernaryBackend` can recover
    /// the concrete type with `downcast_ref`. Keeping it defaulted means the generic
    /// trait stays object-safe and host-slice-oriented; only the one backend that has
    /// a device-resident path opts in.
    ///
    /// Named `as_concrete` (not `as_any`) to avoid clashing with
    /// [`DeviceBuffer::as_any`].
    fn as_concrete(&self) -> Option<&dyn core::any::Any> {
        None
    }
}

/// Symmetric int8 activation-quant positive cap (`Qp`): the int8 range is
/// `[-128, A8_QB]`. Matches BitNet's `ActQuant`/`BitLinear` quant in
/// `transformers` and `tritium-nn::ops::act_quant::QB`.
const A8_QB: f32 = 127.0;

/// Per-token int8 absmax activation quant for the W1.58A8 path (the host default
/// of [`TernaryBackend::mpgemm_with_act_quant`]).
///
/// For each of the `m` rows of the `[m, k]` activation tensor: `γ = max_c |act|`;
/// `q = clamp(round_ties_even(act · 127 / γ), -128, 127)` kept in `f32`;
/// `scale = γ / 127` (the dequant multiplier). A fully-zero row yields all-zero
/// quants and a `0` scale (its dequantized contribution is `0` either way). This
/// replicates `transformers` `ActQuant.forward` — round-half-to-even is
/// load-bearing for greedy token parity, so it is matched exactly. `q_out` is
/// `[m, k]`, `scale_out` is `[m]`; both are assumed correctly sized (the trait
/// method validates before calling).
fn quantize_act_int8(act: &[f32], m: usize, k: usize, q_out: &mut [f32], scale_out: &mut [f32]) {
    for r in 0..m {
        let row = &act[r * k..r * k + k];
        let mut gamma = 0.0_f32;
        for &v in row {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        let out_row = &mut q_out[r * k..r * k + k];
        if gamma == 0.0 {
            for q in out_row.iter_mut() {
                *q = 0.0;
            }
            scale_out[r] = 0.0;
            continue;
        }
        let s = A8_QB / gamma;
        // Asymmetric int8 range: -128 is reachable but the positive cap is +127
        // (Qp = A8_QB), so a value rounding to +128 saturates to +127 — matching
        // transformers' `ActQuant` clamp.
        for (q, &v) in out_row.iter_mut().zip(row) {
            *q = (v * s).round_ties_even().clamp(-128.0, A8_QB);
        }
        scale_out[r] = gamma / A8_QB;
    }
}

/// Errors a backend can return. Backends never panic on bad input — they return
/// one of these.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BackendError {
    /// A buffer length disagrees with the [`GemmShape`].
    ShapeMismatch {
        /// Length the shape implies.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// The backend cannot consume this packing format.
    UnsupportedFormat(TernaryFormat),
    /// Device allocation failed for the requested byte count.
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
    },
    /// Input failed validation before reaching the device.
    InvalidInput(String),
    /// Device-specific failure (e.g. a CUDA driver error), stringified.
    Backend(String),
    /// An error from the foundation layer.
    Core(TritError),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            BackendError::UnsupportedFormat(fmt_) => {
                write!(f, "unsupported ternary format: {fmt_:?}")
            }
            BackendError::OutOfMemory { requested } => {
                write!(f, "out of device memory: requested {requested} bytes")
            }
            BackendError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            BackendError::Backend(msg) => write!(f, "backend error: {msg}"),
            BackendError::Core(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            BackendError::Core(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TritError> for BackendError {
    fn from(e: TritError) -> Self {
        BackendError::Core(e)
    }
}

// Compile-time guarantee that both traits stay object-safe — the runtime holds
// `Box<dyn TernaryBackend>` and passes `&dyn DeviceBuffer`. If a future method
// breaks object safety, this fails to compile.
#[allow(dead_code)]
fn _assert_object_safe(backend: &dyn TernaryBackend, buffer: &dyn DeviceBuffer) {
    let _ = (backend, buffer);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host-quant default must match `transformers` `ActQuant` on the same
    /// golden row `tritium-nn::ops::act_quant` pins, so the two copies cannot
    /// drift: `[1, -2, 0.5, -0.5, 2, -1]` (γ = 2) →
    /// `[64, -127, 32, -32, 127, -64]`, scale `2/127`. (`63.5 → 64` and
    /// `-63.5 → -64` exercise round-half-to-even.)
    #[test]
    fn act_quant_golden() {
        let act = [1.0_f32, -2.0, 0.5, -0.5, 2.0, -1.0];
        let mut q = [f32::NAN; 6];
        let mut scale = [f32::NAN; 1];
        quantize_act_int8(&act, 1, 6, &mut q, &mut scale);
        assert_eq!(q, [64.0, -127.0, 32.0, -32.0, 127.0, -64.0]);
        assert!((scale[0] - 2.0 / 127.0).abs() < 1e-9);
    }

    /// A zero row quantizes to zeros with a zero scale (isolated to its own row).
    #[test]
    fn act_quant_zero_row() {
        let act = [0.0_f32, 0.0, 3.0, -6.0];
        let mut q = [f32::NAN; 4];
        let mut scale = [f32::NAN; 2];
        quantize_act_int8(&act, 2, 2, &mut q, &mut scale);
        assert_eq!(&q[0..2], &[0.0, 0.0]);
        assert_eq!(scale[0], 0.0);
        assert!((scale[1] - 6.0 / 127.0).abs() < 1e-9);
    }

    /// Minimal backend whose `mpgemm` is the literal reference contraction
    /// `out[m,n] = scales[n] · Σ_k act[m,k] · w[n,k]`, used to check the
    /// `mpgemm_with_act_quant` default folds the per-token scale correctly.
    struct MockBuffer {
        trits: Vec<i8>, // [N, K] row-major, values in {-1,0,1}; dims come from `shape`
    }
    impl DeviceBuffer for MockBuffer {
        fn len_bytes(&self) -> usize {
            self.trits.len()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct MockBackend;
    impl TernaryBackend for MockBackend {
        fn device_id(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> DeviceCaps {
            DeviceCaps::new("mock", "spec test backend")
        }
        fn upload_weights(
            &self,
            _packed: &[u8],
            _shape: GemmShape,
            _format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Err(BackendError::InvalidInput("mock: unused".into()))
        }
        fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
            let MpGemm {
                act,
                weights,
                scales,
                shape,
                format: _format,
                out,
            } = p;
            let buf = weights
                .as_any()
                .downcast_ref::<MockBuffer>()
                .ok_or_else(|| BackendError::Backend("mock: bad buffer".into()))?;
            let GemmShape { m, n, k } = shape;
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0_f32;
                    for ki in 0..k {
                        acc += act[mi * k + ki] * f32::from(buf.trits[ni * k + ki]);
                    }
                    out[mi * n + ni] = scales[ni] * acc;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn inference_backend_default_rejects_packed_vjp_explicitly() {
        let backend = MockBackend;
        let buffer = MockBuffer { trits: vec![1] };
        let mut grad_act = [0.0];
        let mut grad_weight = [0.0];
        let error = backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: &[1.0],
                weights: &buffer,
                scales: &[1.0],
                grad_output: &[1.0],
                shape: GemmShape::new(1, 1, 1),
                format: TernaryFormat::Tq2_0,
                grad_act: &mut grad_act,
                grad_projected_weight: &mut grad_weight,
                grad_bias: None,
            })
            .unwrap_err();

        assert_eq!(
            error,
            BackendError::Backend("backend `mock` does not implement packed mpGEMM VJP".to_owned())
        );
    }

    /// `mpgemm_with_act_quant` = quantize → contract → fold per-token scale.
    /// `M=1, K=2, N=1`: act `[3, -1]` (γ=3) → q `[127, -42]`, act_scale `3/127`;
    /// w `[1, -1]`, weight_scale `2.0` → raw `2·(127·1 + (-42)·(-1)) = 338`;
    /// folded `338 · 3/127`.
    #[test]
    fn fused_folds_per_token_scale() {
        let backend = MockBackend;
        let buf = MockBuffer { trits: vec![1, -1] };
        let act = [3.0_f32, -1.0];
        let weight_scales = [2.0_f32];
        let shape = GemmShape { m: 1, n: 1, k: 2 };
        let mut out = [f32::NAN; 1];
        backend
            .mpgemm_with_act_quant(MpGemm {
                act: &act,
                weights: &buf,
                scales: &weight_scales,
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            })
            .expect("fused mpgemm");
        let expected = 338.0_f32 * 3.0 / 127.0;
        // Tight bound: the only slack is the f32 fold-order difference between
        // `(338*3)/127` and `338*(3/127)` (~1 ULP).
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "got {}, want {expected}",
            out[0]
        );
    }

    /// Shape validation fires before any compute.
    #[test]
    fn fused_rejects_bad_shapes() {
        let backend = MockBackend;
        let buf = MockBuffer { trits: vec![1, -1] };
        let shape = GemmShape { m: 1, n: 1, k: 2 };
        let mut out = [0.0_f32; 1];
        // act too short
        assert!(matches!(
            backend.mpgemm_with_act_quant(MpGemm {
                act: &[3.0],
                weights: &buf,
                scales: &[2.0],
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            }),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }
}
