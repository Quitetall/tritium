//! GPU host side for the ROCm/HIP backend. Compiled only with `--features rocm`.
//!
//! This module owns a HIP device binding + loaded module, resolves the
//! addition-only TQ2_0 mpGEMM kernel from the code object emitted by `build.rs`,
//! and drives it. It maps every HIP error to a [`BackendError`] so the backend
//! never panics on a device failure, and reports allocation failures as
//! [`BackendError::OutOfMemory`].
//!
//! ## HIP runtime FFI
//!
//! Host↔device work goes through the raw HIP runtime FFI in [`crate::ffi`]:
//! `hipInit` / `hipSetDevice` open device 0, `hipModuleLoadData` loads the embedded
//! code object, `hipModuleGetFunction` resolves the `tq2_0_add_mpgemm` symbol,
//! `hipMalloc` / `hipMemcpy` move bytes, and `hipModuleLaunchKernel` launches the
//! kernel (with `hipStreamSynchronize` on the default stream before reading back).
//!
//! The only `unsafe` here is the FFI calls; each is in a narrowly scoped block with
//! a `SAFETY:` justification, exactly the pattern tritium-cuda uses for its kernel
//! launch.

use core::any::Any;
use core::ffi::{c_char, c_void};
use std::ffi::CString;
use std::sync::Arc;

use tritium_core::{GemmShape, TernaryFormat};
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

use crate::ffi;

/// Kernel entry point — must match the `extern "C"` symbol in the `.hip` file.
const KERNEL_NAME: &str = "tq2_0_add_mpgemm";

/// HIP threads per block for the 1-D launch grid (one thread per output element).
/// 256 is a safe, occupancy-friendly default on every supported AMD arch.
const THREADS_PER_BLOCK: u32 = 256;

/// The code object produced by `build.rs` (`hipcc --genco`). Embedded at compile
/// time so the backend needs no `.co` file on disk at runtime — the analogue of
/// tritium-cuda's `include_str!` of the PTX.
const TQ2_0_ADD_CO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tq2_0_add.co"));

/// Wrap a HIP error code into a [`BackendError::Backend`] with context, unless it is
/// success. Returns `Ok(())` on `hipSuccess`.
fn hip_check(context: &str, code: ffi::hipError_t) -> Result<(), BackendError> {
    if code == ffi::HIP_SUCCESS {
        Ok(())
    } else {
        Err(BackendError::Backend(format!(
            "{context}: {} (hip {code})",
            ffi::hip_get_error_string(code)
        )))
    }
}

/// Classify a HIP allocation failure as OOM (with the requested byte count) or a
/// generic backend error — mirrors tritium-cuda's `alloc_or_backend`.
fn alloc_or_backend(context: &str, code: ffi::hipError_t, requested: usize) -> BackendError {
    if code == ffi::HIP_ERROR_OUT_OF_MEMORY {
        BackendError::OutOfMemory { requested }
    } else {
        // Reuse hip_check's formatting for the non-OOM case.
        hip_check(context, code).expect_err("non-success code")
    }
}

/// An RAII owner of a `hipMalloc`'d device allocation. `hipFree`s on drop so a
/// failed `mpgemm` mid-sequence never leaks VRAM.
#[derive(Debug)]
struct DeviceAlloc {
    ptr: ffi::hipDeviceptr_t,
    bytes: usize,
}

// SAFETY: `DeviceAlloc` owns a HIP device pointer. HIP device memory is process-
// global and not tied to a thread, and the backend serializes its own access; the
// pointer is only ever used through the HIP runtime, which is internally
// thread-safe for these calls. So the handle is safe to send/share across threads.
unsafe impl Send for DeviceAlloc {}
unsafe impl Sync for DeviceAlloc {}

impl DeviceAlloc {
    /// Allocate `bytes` of device memory. `bytes == 0` yields a null/empty alloc
    /// (HIP rejects zero-size `hipMalloc`), which the callers never launch against.
    fn new(bytes: usize, requested_for_oom: usize) -> Result<Self, BackendError> {
        if bytes == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                bytes: 0,
            });
        }
        let mut ptr: ffi::hipDeviceptr_t = core::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer; `bytes > 0`. On success HIP writes a
        // device pointer into `ptr`; on failure it is left null and we propagate the
        // error (classified as OOM where applicable).
        let code = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
        if code != ffi::HIP_SUCCESS {
            return Err(alloc_or_backend("hipMalloc", code, requested_for_oom));
        }
        Ok(Self { ptr, bytes })
    }

    /// Host→device copy of `src` into this allocation. `src.len()` must be
    /// `<= self.bytes`.
    fn copy_from_host<T: Copy>(&self, src: &[T]) -> Result<(), BackendError> {
        let n = core::mem::size_of_val(src);
        debug_assert!(n <= self.bytes, "h2d overrun");
        if n == 0 {
            return Ok(());
        }
        // SAFETY: `self.ptr` is a live device alloc of `self.bytes >= n` bytes;
        // `src` is a host slice of exactly `n` bytes. Synchronous copy, H2D kind.
        let code = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                n,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        hip_check("hipMemcpy h2d", code)
    }

    /// Device→host copy filling `dst` from this allocation. `dst.len()` (bytes) must
    /// be `<= self.bytes`.
    fn copy_to_host<T: Copy>(&self, dst: &mut [T]) -> Result<(), BackendError> {
        let n = core::mem::size_of_val(dst);
        debug_assert!(n <= self.bytes, "d2h overrun");
        if n == 0 {
            return Ok(());
        }
        // SAFETY: `self.ptr` is a live device alloc of `self.bytes >= n` bytes;
        // `dst` is a host slice of exactly `n` bytes. Synchronous copy, D2H kind.
        let code = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                n,
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        hip_check("hipMemcpy d2h", code)
    }
}

impl Drop for DeviceAlloc {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` came from `hipMalloc` and is freed exactly once
            // (here, on drop). The return code is ignored — there is nothing
            // actionable to do in a destructor, and a double-free is impossible
            // because we null nothing else holds this pointer.
            unsafe {
                let _ = ffi::hipFree(self.ptr);
            }
        }
    }
}

/// Device-resident packed TQ2_0 weights for one matmul operand: the htod copy of
/// the host-packed bytes plus the `[N, K]` geometry and per-row byte stride.
///
/// Internal to the crate: it crosses the [`TernaryBackend`] boundary only as a
/// `Box<dyn DeviceBuffer>`, downcast back here via [`core::any::Any`].
#[derive(Debug)]
pub struct RocmBuffer {
    /// Device allocation holding the packed TQ2_0 bytes. `Arc` so a buffer can be
    /// cheaply shared (mirrors tritium-cuda's `Arc<CudaSlice<u8>>`).
    device: Arc<DeviceAlloc>,
    /// Output channels (`N`), unpadded.
    n: usize,
    /// Contraction dimension (`K`), unpadded.
    k: usize,
    /// Packed bytes per weight row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    row_bytes: usize,
    /// Total bytes uploaded, cached for [`DeviceBuffer::len_bytes`].
    bytes: usize,
}

impl DeviceBuffer for RocmBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// AMD ROCm/HIP ternary mpGEMM backend.
///
/// Opens HIP device `ordinal`, loads the add-only TQ2_0 kernel once in
/// [`RocmBackend::new`], and shares the device binding + module + resolved function
/// across calls. `upload_weights` / `mpgemm` are synchronous (the trait is sync):
/// each `mpgemm` uploads, launches, `hipStreamSynchronize`s the default stream, and
/// copies the result back.
#[derive(Debug)]
pub struct RocmBackend {
    /// Loaded code-object module (kept alive so `func` stays valid; unloaded on drop).
    module: ffi::hipModule_t,
    /// The resolved `tq2_0_add_mpgemm` kernel handle (one thread per output).
    func: ffi::hipFunction_t,
    /// Backend identifier, e.g. `"rocm:0"`.
    device_id: String,
    /// Human-readable device name reported by the driver, e.g. `"AMD Instinct MI300X"`.
    device_name: String,
}

// SAFETY: `RocmBackend` holds opaque HIP module/function handles. These are process-
// global runtime objects, not thread-bound; the HIP runtime is thread-safe for the
// launch + module calls we make, and the backend does not mutate the handles after
// construction. So the backend is safe to send/share across threads (the spec trait
// requires `Send + Sync`).
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Open HIP device `ordinal`, load the TQ2_0 add kernel, and return a backend.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if `hipInit` fails (no AMD device / no ROCm
    /// driver), the device cannot be selected, the module fails to load, or the
    /// kernel symbol is missing. The "no device" case is the expected self-skip on
    /// cpu-only machines that still link this crate.
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        // SAFETY: `hipInit(0)` is the documented HIP runtime entry point; flags must
        // be 0. It is safe to call repeatedly. A non-zero code means no usable AMD
        // device — surfaced as an error (the registry/test self-skip on it).
        let code = unsafe { ffi::hipInit(0) };
        hip_check("hipInit", code)?;

        let mut count: i32 = 0;
        // SAFETY: `&mut count` is a valid out-pointer for the device count.
        let code = unsafe { ffi::hipGetDeviceCount(&mut count) };
        hip_check("hipGetDeviceCount", code)?;
        if count <= 0 {
            return Err(BackendError::Backend("no HIP devices present".into()));
        }
        let dev = i32::try_from(ordinal).map_err(|_| {
            BackendError::InvalidInput(format!("device ordinal {ordinal} too large"))
        })?;
        if dev >= count {
            return Err(BackendError::Backend(format!(
                "HIP device {dev} out of range (count {count})"
            )));
        }

        // SAFETY: `dev` is in `[0, count)`. Binds this thread to the device.
        let code = unsafe { ffi::hipSetDevice(dev) };
        hip_check("hipSetDevice", code)?;

        let device_name = query_device_name(dev);

        // Load the embedded code object and resolve the kernel symbol.
        let mut module: ffi::hipModule_t = core::ptr::null_mut();
        // SAFETY: `TQ2_0_ADD_CO` is a valid, NUL-free code-object image embedded at
        // build time; `&mut module` is a valid out-pointer. HIP copies the image, so
        // the borrow need not outlive the call.
        let code =
            unsafe { ffi::hipModuleLoadData(&mut module, TQ2_0_ADD_CO.as_ptr() as *const c_void) };
        hip_check("hipModuleLoadData", code)?;

        let name = CString::new(KERNEL_NAME).expect("kernel name has no interior NUL");
        let mut func: ffi::hipFunction_t = core::ptr::null_mut();
        // SAFETY: `module` was just loaded successfully; `name` is a valid NUL-
        // terminated C string; `&mut func` is a valid out-pointer.
        let code =
            unsafe { ffi::hipModuleGetFunction(&mut func, module, name.as_ptr() as *const c_char) };
        if let Err(e) = hip_check("hipModuleGetFunction", code) {
            // SAFETY: `module` is a live module from the successful load above; we
            // unload it on the error path so we do not leak it. Ignore the code.
            unsafe {
                let _ = ffi::hipModuleUnload(module);
            }
            return Err(e);
        }

        Ok(Self {
            module,
            func,
            device_id: format!("rocm:{ordinal}"),
            device_name,
        })
    }

    /// Packed bytes per weight row for `k` trits in TQ2_0.
    fn row_bytes(k: usize) -> usize {
        num_blocks(k) * TQ2_0_BLOCK_BYTES
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: `self.module` came from a successful `hipModuleLoadData` and is
            // unloaded exactly once (here). The resolved `func` belongs to the module
            // and is invalidated by the unload; nothing uses it after drop.
            unsafe {
                let _ = ffi::hipModuleUnload(self.module);
            }
        }
    }
}

/// Read the device name via `hipDeviceGetName`, defaulting to a placeholder if the
/// query fails (the name is cosmetic — it only populates [`DeviceCaps`]).
fn query_device_name(dev: i32) -> String {
    let mut buf = [0i8; 256];
    // SAFETY: `buf` is a 256-byte stack buffer; we pass its capacity. HIP writes a
    // NUL-terminated name (or fails, leaving `buf` zeroed, which reads as empty).
    let code = unsafe { ffi::hipDeviceGetName(buf.as_mut_ptr() as *mut c_char, 256, dev) };
    if code != ffi::HIP_SUCCESS {
        return "unknown AMD device".to_owned();
    }
    // SAFETY: HIP wrote a NUL-terminated string into `buf` on success.
    let cstr = unsafe { core::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char) };
    let name = cstr.to_string_lossy().into_owned();
    if name.is_empty() {
        "unknown AMD device".to_owned()
    } else {
        name
    }
}

impl TernaryBackend for RocmBackend {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn capabilities(&self) -> DeviceCaps {
        // No fp8/IMMA matrix-core path yet; the fused W1.58A8 path degrades via the
        // trait default (host quant → mpgemm → fold). Only TQ2_0 is advertised.
        DeviceCaps::new("rocm", self.device_name.clone())
            .with_features(vec!["hip".to_owned(), "tq2_0".to_owned()])
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let GemmShape { n, k, .. } = shape;
        let row_bytes = Self::row_bytes(k);
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} ({format:?})",
                packed.len()
            )));
        }

        let device = DeviceAlloc::new(packed.len(), expected)?;
        device.copy_from_host(packed)?;

        Ok(Box::new(RocmBuffer {
            device: Arc::new(device),
            n,
            k,
            row_bytes,
            bytes: packed.len(),
        }))
    }

    fn mpgemm(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a RocmBuffer".into()))?;

        let GemmShape { m, n, k } = shape;
        if buf.n != n || buf.k != k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: n * k,
            });
        }
        if act.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: act.len(),
            });
        }
        if scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: scales.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }
        // Empty output: nothing to launch (matches the reference's empty result).
        // Mirrors tritium-cuda's `mpgemm_kernel` early-out.
        if m == 0 || n == 0 {
            return Ok(());
        }

        // Reject shapes whose flat output count would overflow the kernel's `int`
        // index arithmetic / the `u32` launch grid (a silent truncation would give a
        // wrong answer; BitNet shapes are orders of magnitude below this).
        let total = m
            .checked_mul(n)
            .filter(|&t| t <= i32::MAX as usize && k <= i32::MAX as usize)
            .ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "shape {shape:?} exceeds the rocm kernel's int index range"
                ))
            })?;

        let d_act = DeviceAlloc::new(core::mem::size_of_val(act), act.len() * 4)?;
        d_act.copy_from_host(act)?;
        let d_scales = DeviceAlloc::new(core::mem::size_of_val(scales), scales.len() * 4)?;
        d_scales.copy_from_host(scales)?;
        let d_out = DeviceAlloc::new(core::mem::size_of_val(out), m * n * 4)?;

        // Launch geometry: one thread per output element (the simple kernel).
        let grid = (total as u32).div_ceil(THREADS_PER_BLOCK);

        // Kernel scalar args (the four trailing `int`s). Held in locals so their
        // addresses are stable for the duration of the launch.
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let row_bytes_i = buf.row_bytes as i32;

        // `hipModuleLaunchKernel` takes an array of `void*`, one per kernel argument
        // in declaration order: (act, weights, scales, out, m, n, k, row_bytes). The
        // device pointers are passed BY VALUE, so each entry points at the *variable
        // holding the pointer* (i.e. `&d_act.ptr`), exactly as `cuLaunchKernel`
        // expects — the same convention cudarc's launch builder encodes.
        let act_ptr = d_act.ptr;
        let w_ptr = buf.device.ptr;
        let scales_ptr = d_scales.ptr;
        let out_ptr = d_out.ptr;
        let mut params: [*mut c_void; 8] = [
            (&act_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&w_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&scales_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&out_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&m_i as *const i32 as *mut c_void),
            (&n_i as *const i32 as *mut c_void),
            (&k_i as *const i32 as *mut c_void),
            (&row_bytes_i as *const i32 as *mut c_void),
        ];

        // SAFETY: `self.func` is the resolved `tq2_0_add_mpgemm` kernel, whose
        // signature is (const float*, const unsigned char*, const float*, float*,
        // int, int, int, int) — matched here by `params` in that exact order (four
        // device-pointer-by-value entries then four `int` scalars). The device
        // allocations are sized against `shape` above; the grid covers ceil(M*N/256)
        // blocks of 256 threads and the kernel bounds-checks `idx >= M*N`. Shared
        // memory is 0, stream is the default (null). All argument locals outlive the
        // launch (the synchronize below joins before they drop). `extra` is null.
        let code = unsafe {
            ffi::hipModuleLaunchKernel(
                self.func,
                grid,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                core::ptr::null_mut(),
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        hip_check("hipModuleLaunchKernel", code)?;

        // Join the default stream before reading back (the launch is async).
        // SAFETY: null stream is the default stream; blocks until the launch done.
        let code = unsafe { ffi::hipStreamSynchronize(core::ptr::null_mut()) };
        hip_check("hipStreamSynchronize", code)?;

        d_out.copy_to_host(out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! GPU conformance test. Runs only with `--features rocm` AND a working AMD/HIP
    //! device, so it is exercised on the ROCm CI lane, never on cpu-only lanes. When
    //! no device is present the test self-skips (constructing the backend returns
    //! `Err`) rather than failing — exactly like tritium-cuda / tritium-wgpu.
    //!
    //! `run_conformance` itself packs each frozen vector's trits to TQ2_0 (block
    //! scale 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
    //! scales, and grades against `reference_mpgemm`.

    use super::*;
    use tritium_testkit::{Tolerance, frozen_vectors, run_conformance};

    /// Frozen-set conformance on the AMD/HIP device, or a clean self-skip when no
    /// device is present (mirrors the CUDA / wgpu conformance tests). The frozen set
    /// mixes TQ1_0 and TQ2_0 vectors; this kernel only handles TQ2_0, so we filter to
    /// the TQ2_0 subset the kernel is responsible for.
    #[test]
    fn conformance_or_skip() {
        let backend = match RocmBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping rocm conformance: no device ({e})");
                return;
            }
        };
        let vectors: Vec<_> = frozen_vectors()
            .into_iter()
            .filter(|v| v.format == "tq2_0")
            .collect();
        assert!(!vectors.is_empty(), "expected some tq2_0 frozen vectors");

        let report = run_conformance(&backend, &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} rocm conformance failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all tq2_0 vectors must pass");
    }
}
