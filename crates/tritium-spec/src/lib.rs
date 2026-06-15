//! # tritium-spec
//!
//! The contract every Tritium execution backend implements. One trait,
//! [`TernaryBackend`], plus the buffer and capability types it speaks. No
//! implementations live here — `tritium-cpu`, `tritium-cuda`, and future backends
//! implement this trait; `tritium-runtime` stores them as trait objects.
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

use core::any::Any;
use core::fmt;

use tritium_core::{GemmShape, TernaryFormat, TritError};

mod caps;
pub use caps::DeviceCaps;

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
    /// `act` is `[M, K]` row-major, `weights` is the handle from
    /// [`upload_weights`](TernaryBackend::upload_weights), `scales` is `[N]`
    /// per-output-channel, `out` is `[M, N]` and is overwritten.
    ///
    /// Result must match [`tritium_core::reference_mpgemm`] with relative error
    /// `≤ 1e-4` (fp32 accumulation reorders across backends; bit-exactness is not
    /// required for the float path, only for packing).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if buffer lengths disagree with `shape`;
    /// [`BackendError::Backend`] for device-specific failures.
    fn mpgemm(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError>;
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

impl std::error::Error for BackendError {}

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
