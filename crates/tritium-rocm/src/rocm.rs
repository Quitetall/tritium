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
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

use crate::ffi;

/// Kernel entry point — must match the `extern "C"` symbol in the `.hip` file.
const KERNEL_NAME: &str = "tq2_0_add_mpgemm";

/// v3 prefill-attention kernel entry point — must match the `extern "C"`
/// symbol in `kernels/gqa_attention_v3.hip` (Track E2).
const ATTN_KERNEL_NAME: &str = "gqa_attention_batch_v3_f32";

/// Wavefront-width probe entry point (same code object as the attention
/// kernel): writes the device `warpSize` so the host can gate the v3 dispatch
/// on a wave width the kernel's analysis covers (32 or 64).
const WAVE_PROBE_NAME: &str = "attn_v3_wave_probe";

/// HIP threads per block for the 1-D launch grid (one thread per output element).
/// 256 is a safe, occupancy-friendly default on every supported AMD arch.
const THREADS_PER_BLOCK: u32 = 256;

/// The code object produced by `build.rs` (`hipcc --genco`). Embedded at compile
/// time so the backend needs no `.co` file on disk at runtime — the analogue of
/// tritium-cuda's `include_str!` of the PTX.
const TQ2_0_ADD_CO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tq2_0_add.co"));

/// The v3 prefill-attention code object (plus the wave probe), emitted by
/// `build.rs` alongside the mpGEMM one and embedded the same way.
const GQA_ATTENTION_V3_CO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gqa_attention_v3.co"));

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
// SAFETY: same reasoning as `Send` above — shared references only ever reach the
// internally thread-safe HIP runtime.
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
    /// Loaded attention code-object module (kept alive so `attn_func` stays
    /// valid; unloaded on drop).
    attn_module: ffi::hipModule_t,
    /// The resolved `gqa_attention_batch_v3_f32` kernel handle (Track E2).
    attn_func: ffi::hipFunction_t,
    /// Physical wavefront width reported by the `attn_v3_wave_probe` kernel at
    /// init (64 on CDNA, 32 on RDNA). The v3 attention dispatch refuses the
    /// device rung unless it is 32 or 64 — the analogue of tritium-metal's
    /// `thread_execution_width == 32` check.
    wave_size: i32,
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
// SAFETY: same reasoning as `Send` above — the handles are immutable after
// construction and every use goes through the thread-safe HIP runtime.
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

        // Load the embedded code objects and resolve the kernel symbols. Each
        // error path unloads whatever was already loaded so `new` never leaks
        // a module.
        let module = load_module(TQ2_0_ADD_CO)?;
        let func = match get_function(module, KERNEL_NAME) {
            Ok(f) => f,
            Err(e) => {
                unload_module(module);
                return Err(e);
            }
        };

        let attn_module = match load_module(GQA_ATTENTION_V3_CO) {
            Ok(m) => m,
            Err(e) => {
                unload_module(module);
                return Err(e);
            }
        };
        let attn_resolved = get_function(attn_module, ATTN_KERNEL_NAME)
            .and_then(|attn_func| Ok((attn_func, get_function(attn_module, WAVE_PROBE_NAME)?)))
            .and_then(|(attn_func, probe)| Ok((attn_func, probe_wave_size(probe)?)));
        let (attn_func, wave_size) = match attn_resolved {
            Ok(v) => v,
            Err(e) => {
                unload_module(attn_module);
                unload_module(module);
                return Err(e);
            }
        };

        Ok(Self {
            module,
            func,
            attn_module,
            attn_func,
            wave_size,
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
        // Both modules came from successful `hipModuleLoadData` calls and are
        // unloaded exactly once (here). The resolved functions belong to their
        // modules and are invalidated by the unload; nothing uses them after
        // drop.
        unload_module(self.module);
        unload_module(self.attn_module);
    }
}

/// Load an embedded code-object image into a HIP module.
fn load_module(image: &[u8]) -> Result<ffi::hipModule_t, BackendError> {
    let mut module: ffi::hipModule_t = core::ptr::null_mut();
    // SAFETY: `image` is a valid code-object image embedded at build time;
    // `&mut module` is a valid out-pointer. HIP copies the image, so the
    // borrow need not outlive the call.
    let code = unsafe { ffi::hipModuleLoadData(&mut module, image.as_ptr() as *const c_void) };
    hip_check("hipModuleLoadData", code)?;
    Ok(module)
}

/// Unload a module if non-null, ignoring the return code (used on error paths
/// and in `Drop`, where there is nothing actionable to do with a failure).
fn unload_module(module: ffi::hipModule_t) {
    if !module.is_null() {
        // SAFETY: `module` came from a successful `hipModuleLoadData` and each
        // call site unloads it exactly once.
        unsafe {
            let _ = ffi::hipModuleUnload(module);
        }
    }
}

/// Resolve kernel symbol `name` in `module`.
fn get_function(module: ffi::hipModule_t, name: &str) -> Result<ffi::hipFunction_t, BackendError> {
    let cname = CString::new(name).expect("kernel name has no interior NUL");
    let mut func: ffi::hipFunction_t = core::ptr::null_mut();
    // SAFETY: `module` is a live module; `cname` is a valid NUL-terminated C
    // string; `&mut func` is a valid out-pointer.
    let code =
        unsafe { ffi::hipModuleGetFunction(&mut func, module, cname.as_ptr() as *const c_char) };
    hip_check(&format!("hipModuleGetFunction({name})"), code)?;
    Ok(func)
}

/// Run the 1-thread `attn_v3_wave_probe` kernel and return the device's
/// physical wavefront width. A probe kernel is used instead of
/// `hipDeviceGetAttribute` / `hipGetDeviceProperties` because the attribute
/// enum values and the props struct layout have both changed across ROCm
/// major releases, while module launch + memcpy is ABI this crate already
/// depends on (see the kernel banner).
fn probe_wave_size(probe: ffi::hipFunction_t) -> Result<i32, BackendError> {
    let d_out = DeviceAlloc::new(core::mem::size_of::<i32>(), core::mem::size_of::<i32>())?;
    let out_ptr = d_out.ptr;
    let mut params: [*mut c_void; 1] = [(&out_ptr as *const ffi::hipDeviceptr_t as *mut c_void)];
    // SAFETY: `probe` is the resolved `attn_v3_wave_probe` kernel, whose only
    // argument is an `int*` sized for one value (allocated just above). One
    // 1×1×1 block of one thread; shared memory 0; default stream; `extra`
    // null. The argument local outlives the launch (the synchronize below
    // joins before it drops).
    let code = unsafe {
        ffi::hipModuleLaunchKernel(
            probe,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            core::ptr::null_mut(),
            params.as_mut_ptr(),
            core::ptr::null_mut(),
        )
    };
    hip_check("hipModuleLaunchKernel(wave probe)", code)?;
    // SAFETY: null stream is the default stream; blocks until the probe done.
    let code = unsafe { ffi::hipStreamSynchronize(core::ptr::null_mut()) };
    hip_check("hipStreamSynchronize(wave probe)", code)?;
    let mut out = [0i32; 1];
    d_out.copy_to_host(&mut out)?;
    Ok(out[0])
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

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format,
            out,
        } = p;
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

impl RocmBackend {
    /// v3 Q-blocked prefill GQA attention (Track E2: the HIP port of
    /// tritium-cuda's `gqa_attention_batch_v3_f32`) — `M >= 1` query rows
    /// against an f32 KV arena, causal, GQA.
    ///
    /// STATUS: the device kernel is compile/lint-verified only until the
    /// budgeted MI300X session runs `rocm::tests::
    /// attn_v3_matches_pinned_host_reference_or_skip` (runbook: the
    /// [`crate::attn`] module docs).
    ///
    /// Layouts (row-major): `q`/`out` `[m, n_head, head_dim]`; `k`/`v` KV
    /// arenas `[>= causal_offset + m, n_head_kv, head_dim]`. `ctx_max` is the
    /// scores-scratch stride (`>= causal_offset + m`; the arena capacity, in
    /// the CUDA runner's terms).
    ///
    /// Dispatch priority — CUDA runs v3 → v2 → rev-1; this backend has no v2
    /// or rev-1 device kernel, so the ladder is v3 (device) → the pinned-order
    /// HOST reference [`crate::attn::gqa_attention_prefill_ref`] (same
    /// summation orders — and, with HIP's f64 `exp_f32`, expected bit-equal;
    /// see the `attn` module docs). The host rung serves when:
    /// * `TRITIUM_ATTN_V3=0` (kill switch; the tritium-cuda env contract —
    ///   any value other than `0`/`1` is a loud reject);
    /// * `head_dim > ATTN_V3_HDMAX` (the kernel's LDS staging bound);
    /// * the probed device `warpSize` is neither 32 nor 64 (the kernel's
    ///   width-32 logical-lane mapping is only argued for those — the
    ///   analogue of Metal's `thread_execution_width` refusal);
    /// * the row-block count exceeds the conservative 65535 grid-dimension
    ///   ceiling (the bound the Metal twin also uses).
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on shape/length violations or a
    /// malformed `TRITIUM_ATTN_V3`; [`BackendError::Backend`] on HIP
    /// dispatch failures.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_prefill(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        ctx_max: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        causal_offset: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        crate::attn::validate_v3_launch(
            q.len(),
            k.len(),
            v.len(),
            out.len(),
            ctx_max,
            n_head,
            n_head_kv,
            head_dim,
            causal_offset,
            m,
        )
        .map_err(BackendError::InvalidInput)?;

        let v3_enabled =
            crate::attn::parse_attn_v3(std::env::var("TRITIUM_ATTN_V3").ok().as_deref())
                .map_err(BackendError::InvalidInput)?;
        let (grid_x, grid_y) = crate::attn::v3_grid(m, n_head);
        let use_device = v3_enabled
            && head_dim <= crate::attn::ATTN_V3_HDMAX
            && (self.wave_size == 32 || self.wave_size == 64)
            && grid_x <= 65535
            && grid_y <= 65535;
        if !use_device {
            crate::attn::gqa_attention_prefill_ref(
                q,
                k,
                v,
                out,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset,
                m,
            );
            return Ok(());
        }
        self.gqa_attention_prefill_v3_device(
            q,
            k,
            v,
            out,
            ctx_max,
            n_head,
            n_head_kv,
            head_dim,
            scale,
            causal_offset,
            m,
            (grid_x, grid_y),
        )
    }

    /// Launch the v3 kernel: grid `(n_head, ceil(m/ATTN_V3_BQ))` blocks of
    /// `ATTN_V3_THREADS`, plus the global `[m, n_head, ctx_max]` scores
    /// scratch (device-allocated per call, like the CUDA prefill's per-call
    /// `d_scores`). Caller has validated shapes and the dispatch bounds.
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention_prefill_v3_device(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        ctx_max: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        causal_offset: usize,
        m: usize,
        (grid_x, grid_y): (u32, u32),
    ) -> Result<(), BackendError> {
        let scores_len = crate::attn::v3_scores_len(m, n_head, ctx_max)
            .ok_or_else(|| BackendError::InvalidInput("scores scratch overflows".to_owned()))?;

        let d_q = DeviceAlloc::new(core::mem::size_of_val(q), q.len() * 4)?;
        d_q.copy_from_host(q)?;
        let d_k = DeviceAlloc::new(core::mem::size_of_val(k), k.len() * 4)?;
        d_k.copy_from_host(k)?;
        let d_v = DeviceAlloc::new(core::mem::size_of_val(v), v.len() * 4)?;
        d_v.copy_from_host(v)?;
        let d_out = DeviceAlloc::new(core::mem::size_of_val(out), out.len() * 4)?;
        // checked *4: v3_scores_len checked the ELEMENT product, but an
        // element count in (2^62, 2^64) would wrap the byte conversion toward
        // 0 and DeviceAlloc::new(0, _) returns a null alloc the kernel would
        // write through (review 5593fda F1; unreachable at real shapes).
        let scores_bytes = scores_len
            .checked_mul(4)
            .ok_or_else(|| BackendError::Backend("attn v3 scores byte size overflows".into()))?;
        let d_scores = DeviceAlloc::new(scores_bytes, scores_bytes)?;

        // Kernel scalar args, in locals so their addresses are stable for the
        // duration of the launch (validate_v3_launch capped each at i32::MAX).
        let ctx_max_i = ctx_max as i32;
        let n_head_i = n_head as i32;
        let n_head_kv_i = n_head_kv as i32;
        let head_dim_i = head_dim as i32;
        let causal_offset_i = causal_offset as i32;
        let m_i = m as i32;

        let q_ptr = d_q.ptr;
        let k_ptr = d_k.ptr;
        let v_ptr = d_v.ptr;
        let out_ptr = d_out.ptr;
        let scores_ptr = d_scores.ptr;
        let mut params: [*mut c_void; 12] = [
            (&q_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&k_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&v_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&out_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&scores_ptr as *const ffi::hipDeviceptr_t as *mut c_void),
            (&ctx_max_i as *const i32 as *mut c_void),
            (&n_head_i as *const i32 as *mut c_void),
            (&n_head_kv_i as *const i32 as *mut c_void),
            (&head_dim_i as *const i32 as *mut c_void),
            (&scale as *const f32 as *mut c_void),
            (&causal_offset_i as *const i32 as *mut c_void),
            (&m_i as *const i32 as *mut c_void),
        ];

        // SAFETY: `self.attn_func` is the resolved `gqa_attention_batch_v3_f32`
        // kernel, whose signature is (const float*, const float*, const
        // float*, float*, float*, int, int, int, int, float, int, int) —
        // matched here by `params` in that exact order (five device-pointer-
        // by-value entries then the scalars). The device allocations are sized
        // against the validated shapes above; the grid is
        // (n_head, ceil(m/BQ)) blocks of ATTN_V3_THREADS, the CUDA launch
        // geometry the kernel is written for, and the kernel bounds-checks its
        // row block (`nrows <= 0` early-out). Shared memory is 0 (static LDS
        // only), stream is the default (null). All argument locals outlive the
        // launch (the synchronize below joins before they drop). `extra` null.
        let code = unsafe {
            ffi::hipModuleLaunchKernel(
                self.attn_func,
                grid_x,
                grid_y,
                1,
                crate::attn::ATTN_V3_THREADS,
                1,
                1,
                0,
                core::ptr::null_mut(),
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        hip_check("hipModuleLaunchKernel(attn v3)", code)?;

        // Join the default stream before reading back (the launch is async).
        // SAFETY: null stream is the default stream; blocks until done.
        let code = unsafe { ffi::hipStreamSynchronize(core::ptr::null_mut()) };
        hip_check("hipStreamSynchronize(attn v3)", code)?;

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
            .filter(|v| v.format == tritium_core::TernaryFormat::Tq2_0)
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

    // ---- v3 prefill attention (Track E2) — MI300X-lane conformance gate ----

    use crate::attn;

    /// RAII guard for `TRITIUM_ATTN_V3` (the tritium-metal / tritium-nn env
    /// pattern): sets the value and restores the previous one on drop
    /// (panic-safe), so the wrapper smoke below cannot inherit a CI-exported
    /// `TRITIUM_ATTN_V3=0` and silently compare ref-vs-ref.
    struct AttnV3Env(Option<String>);

    impl AttnV3Env {
        fn set(v: &str) -> Self {
            let prev = std::env::var("TRITIUM_ATTN_V3").ok();
            // SAFETY: the only reader of this env var in this test binary is
            // `gqa_attention_prefill`'s kill-switch lookup, which runs on this
            // thread inside the guard's scope; no other test in the crate
            // reads or writes the environment concurrently.
            unsafe {
                std::env::set_var("TRITIUM_ATTN_V3", v);
            }
            Self(prev)
        }
    }

    impl Drop for AttnV3Env {
        fn drop(&mut self) {
            // SAFETY: same no-concurrent-env-access argument as `set`.
            unsafe {
                match &self.0 {
                    Some(v) => std::env::set_var("TRITIUM_ATTN_V3", v),
                    None => std::env::remove_var("TRITIUM_ATTN_V3"),
                }
            }
        }
    }

    /// Deterministic q/k/v for an attention shape (xorshift64, the same
    /// generator the Metal twin's gate uses — no external rng).
    fn attn_case(
        m: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        causal_offset: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx_top = causal_offset + m;
        let mut s: u64 = 0xA076_1D64_78BD_642F
            ^ ((m as u64) << 3)
            ^ ((n_head as u64) << 21)
            ^ ((head_dim as u64) << 37)
            ^ (causal_offset as u64);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..m * n_head * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..ctx_top * n_head_kv * head_dim)
            .map(|_| next())
            .collect();
        let v: Vec<f32> = (0..ctx_top * n_head_kv * head_dim)
            .map(|_| next())
            .collect();
        (q, k, v)
    }

    /// The v3 device kernel vs the pinned-order host reference, across the
    /// CUDA gate's regimes: staircase (m spanning multiple BQ row-blocks with
    /// a tail), pure tail (m < BQ), deep ctx (causal_offset > 0), GQA + MHA
    /// head groupings, head_dim at/below the HDMAX cap and not a multiple of
    /// 32, and a ctx_max wider than ctx_top (scores-stride vs arena split).
    ///
    /// TWO-TIER assertion (see the `attn` module docs):
    /// * Tier 1 (hard, always): per-element agreement at rel 1e-5 / abs 1e-6.
    ///   Every summation ORDER is pinned identically on both sides, so any
    ///   drift past this exp-only band is a REAL order/mapping bug.
    /// * Tier 2 (hard by default): `to_bits` equality — HIP's `exp_f32`
    ///   keeps CUDA's f64-round-once spelling, so bit-parity with the host's
    ///   glibc `expf` is expected. If ONLY this tier fires, the deviation is
    ///   exp-attributable (ocml f64 exp ULP): set
    ///   `TRITIUM_ROCM_ATTN_STRICT_BITS=0` to demote it to a printed report,
    ///   record the count, and file the follow-up (the Metal twin's
    ///   documented-tolerance precedent) — do not loosen Tier 1.
    ///
    /// Calls the device path directly (not the dispatch wrapper), so a
    /// wave-size fallback cannot turn this gate into a vacuous ref-vs-ref
    /// pass; skips (loudly) when no AMD device exists and on a wave width
    /// the kernel's analysis does not cover.
    #[test]
    fn attn_v3_matches_pinned_host_reference_or_skip() {
        let backend = match RocmBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                // A load-time .co rejection (missing gfx slice, bad module)
                // arrives here identically to "no device" — on the one
                // budgeted MI300X session that must NOT read as a skip:
                // TRITIUM_ROCM_REQUIRE_DEVICE=1 turns it into a hard failure
                // (review 5593fda F2; the runbook exports it).
                if std::env::var_os("TRITIUM_ROCM_REQUIRE_DEVICE").is_some_and(|v| v == "1") {
                    panic!("TRITIUM_ROCM_REQUIRE_DEVICE=1 but backend init failed: {e}");
                }
                eprintln!("skipping rocm attn v3 gate: no device ({e})");
                return;
            }
        };
        if backend.wave_size != 32 && backend.wave_size != 64 {
            eprintln!(
                "skipping rocm attn v3 gate: probed warpSize {} is neither 32 nor 64",
                backend.wave_size
            );
            return;
        }
        let strict_bits = std::env::var("TRITIUM_ROCM_ATTN_STRICT_BITS").map_or(true, |v| v != "0");

        // (m, n_head, n_head_kv, head_dim, causal_offset, ctx_slack)
        for &(m, n_head, n_head_kv, head_dim, causal_offset, ctx_slack) in &[
            (1usize, 4usize, 4usize, 64usize, 0usize, 0usize), // single row, MHA
            (5, 8, 2, 64, 0, 0),                               // tail-only block
            (20, 8, 2, 128, 0, 0),                             // staircase + tail, HDMAX
            (11, 8, 2, 80, 37, 0),                             // deep ctx, hd % 32 != 0
            (16, 4, 1, 64, 3, 5),                              // ctx_max > ctx_top
            (67, 8, 4, 64, 129, 0),                            // multi-chunk deep ctx
        ] {
            let ctx_max = causal_offset + m + ctx_slack;
            let (q, k, v) = attn_case(m, n_head, n_head_kv, head_dim, causal_offset);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut got = vec![0.0f32; m * n_head * head_dim];
            let mut want = vec![0.0f32; m * n_head * head_dim];

            attn::validate_v3_launch(
                q.len(),
                k.len(),
                v.len(),
                got.len(),
                ctx_max,
                n_head,
                n_head_kv,
                head_dim,
                causal_offset,
                m,
            )
            .expect("gate shape must satisfy the launch contract");
            backend
                .gqa_attention_prefill_v3_device(
                    &q,
                    &k,
                    &v,
                    &mut got,
                    ctx_max,
                    n_head,
                    n_head_kv,
                    head_dim,
                    scale,
                    causal_offset,
                    m,
                    attn::v3_grid(m, n_head),
                )
                .expect("v3 device launch");
            attn::gqa_attention_prefill_ref(
                &q,
                &k,
                &v,
                &mut want,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset,
                m,
            );

            let mut bit_mismatches = 0usize;
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                // Tier 1: any drift past the exp-only band is a real bug.
                let diff = (g - w).abs();
                let ok = diff <= 1e-6 || diff <= 1e-5 * w.abs();
                assert!(
                    ok,
                    "[{i}] device {g} vs host {w} (m={m} n_head={n_head} kv={n_head_kv} \
                     hd={head_dim} co={causal_offset} ctx_max={ctx_max}) — beyond the \
                     exp-only tolerance: a pinned-order or thread-mapping bug"
                );
                if g.to_bits() != w.to_bits() {
                    bit_mismatches += 1;
                    if bit_mismatches <= 5 {
                        eprintln!(
                            "attn v3 bit mismatch [{i}]: device {g} ({:#010x}) vs host {w} \
                             ({:#010x}) (m={m} n_head={n_head} kv={n_head_kv} hd={head_dim} \
                             co={causal_offset})",
                            g.to_bits(),
                            w.to_bits()
                        );
                    }
                }
            }
            // Tier 2: bit parity (expected via the f64-round-once exp).
            if strict_bits {
                assert_eq!(
                    bit_mismatches,
                    0,
                    "attn v3: {bit_mismatches}/{} elements differ in bits while ALL pass \
                     the exp-only tolerance (m={m} n_head={n_head} kv={n_head_kv} \
                     hd={head_dim} co={causal_offset}). This is an ocml-exp ULP finding, \
                     not an order bug — re-run with TRITIUM_ROCM_ATTN_STRICT_BITS=0, \
                     record the counts, and file the documented-tolerance follow-up.",
                    got.len()
                );
            } else if bit_mismatches > 0 {
                eprintln!(
                    "attn v3 REPORT (strict bits off): {bit_mismatches}/{} bit mismatches \
                     at m={m} n_head={n_head} kv={n_head_kv} hd={head_dim} co={causal_offset}",
                    got.len()
                );
            }
        }

        // Smoke the public dispatch wrapper once (env parsing + priority +
        // validation glue; the loop above pinned the device kernel itself).
        // Pin the kill switch ON for this leg: a CI-exported TRITIUM_ATTN_V3=0
        // would otherwise route the wrapper to the host reference and turn the
        // comparison below into a vacuous ref-vs-ref pass.
        let _attn_env = AttnV3Env::set("1");
        let (m, n_head, n_head_kv, head_dim, causal_offset) = (5, 8, 2, 64, 0);
        let (q, k, v) = attn_case(m, n_head, n_head_kv, head_dim, causal_offset);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut got = vec![0.0f32; m * n_head * head_dim];
        let mut want = vec![0.0f32; m * n_head * head_dim];
        backend
            .gqa_attention_prefill(
                &q,
                &k,
                &v,
                &mut got,
                causal_offset + m,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset,
                m,
            )
            .expect("dispatch wrapper");
        attn::gqa_attention_prefill_ref(
            &q,
            &k,
            &v,
            &mut want,
            n_head,
            n_head_kv,
            head_dim,
            scale,
            causal_offset,
            m,
        );
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let diff = (g - w).abs();
            assert!(
                diff <= 1e-6 || diff <= 1e-5 * w.abs(),
                "wrapper [{i}] device {g} vs host {w}"
            );
        }
    }
}
