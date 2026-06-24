//! Raw `extern "C"` FFI to the HIP runtime. Compiled only with `--features rocm`.
//!
//! Declares the minimal, ABI-stable HIP surface the backend uses — the same shapes
//! the C `hip/hip_runtime_api.h` exposes — so we link against `libamdhip64` without
//! an external binding crate. Every symbol here is part of the public HIP runtime
//! ABI and stable across ROCm 5.x/6.x; the types mirror the C declarations exactly:
//!
//! - `hipError_t` is a C `enum` (`int`); `0` is `hipSuccess`.
//! - `hipModule_t` / `hipFunction_t` / `hipStream_t` are opaque pointer handles.
//! - `hipDeviceptr_t` is `void*` device memory.
//! - `hipMemcpyKind` is a C enum; we only need H2D (`1`) and D2H (`2`).
//!
//! The `#[link(name = "amdhip64")]` block names the HIP runtime shared library
//! (`libamdhip64.so`), which ships with every ROCm install. Linking is resolved at
//! build time on the ROCm lane (the toolkit is present there); on the default Linux
//! build this whole module is `#[cfg]`-compiled out, so the linker never sees it.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_uint, c_void};

/// HIP error code (`hipError_t`); `0` == `hipSuccess`. Other values stringify via
/// [`hip_get_error_string`].
pub type hipError_t = c_int;

/// `hipSuccess`.
pub const HIP_SUCCESS: hipError_t = 0;
/// `hipErrorOutOfMemory` (a.k.a. `hipErrorMemoryAllocation`) — the value the HIP
/// runtime returns when a device allocation cannot be satisfied. Stable at 2 across
/// ROCm releases (it mirrors CUDA's `cudaErrorMemoryAllocation`).
pub const HIP_ERROR_OUT_OF_MEMORY: hipError_t = 2;

/// Opaque device-memory pointer (`hipDeviceptr_t` == `void*`).
pub type hipDeviceptr_t = *mut c_void;

/// Opaque module handle (`hipModule_t`).
pub type hipModule_t = *mut c_void;
/// Opaque kernel handle (`hipFunction_t`).
pub type hipFunction_t = *mut c_void;
/// Opaque stream handle (`hipStream_t`); the null pointer is the default stream.
pub type hipStream_t = *mut c_void;

/// `hipMemcpyKind::hipMemcpyHostToDevice`.
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `hipMemcpyKind::hipMemcpyDeviceToHost`.
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// SAFETY: these are the public, ABI-stable HIP runtime entry points (the same
// declarations as `hip/hip_runtime_api.h`). Each `unsafe extern` import is sound to
// declare; *calls* are wrapped in narrowly scoped `unsafe` blocks in `rocm.rs` with
// `SAFETY:` justifications. `libamdhip64` ships with every ROCm install and is only
// linked when the `rocm` feature compiles this module.
#[link(name = "amdhip64")]
unsafe extern "C" {
    /// Initialize the HIP runtime. `flags` must be 0. Fails (non-zero) when no AMD
    /// device / ROCm driver is present — the self-skip signal for the conformance
    /// test and the registry.
    pub fn hipInit(flags: c_uint) -> hipError_t;

    /// Number of HIP devices into `*count`.
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;

    /// Bind the calling thread to device `device_id`.
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;

    /// Write the device name (NUL-terminated) into `name` (capacity `len`).
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;

    /// Allocate `size` bytes of device memory, storing the pointer into `*ptr`.
    pub fn hipMalloc(ptr: *mut hipDeviceptr_t, size: usize) -> hipError_t;

    /// Free device memory previously returned by [`hipMalloc`].
    pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;

    /// Synchronous memcpy of `size` bytes between host and device per `kind`.
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;

    /// Load a code-object image (the `build.rs`-emitted bytes) into `*module`.
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;

    /// Unload a module loaded by [`hipModuleLoadData`].
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;

    /// Resolve a kernel symbol `name` in `module` into `*function`.
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;

    /// Launch `f` with the given grid/block geometry. `kernel_params` is an array of
    /// `void*`, each pointing at one kernel argument, in declaration order (the HIP
    /// analogue of CUDA's `cuLaunchKernel` param array). `extra` is null here.
    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    /// Block until all work on `stream` (null = default) completes.
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    /// Human-readable string for a `hipError_t` (never null).
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}

/// Safe-ish wrapper around [`hipGetErrorString`]: returns an owned `String` for an
/// error code. Used to build [`tritium_spec::BackendError`] messages.
pub fn hip_get_error_string(error: hipError_t) -> String {
    // SAFETY: `hipGetErrorString` returns a pointer to a static, NUL-terminated C
    // string for any `hipError_t` (including unknown codes → "unknown error"), and
    // never null. We only read it (copy to an owned String) and do not free it.
    unsafe {
        let ptr = hipGetErrorString(error);
        if ptr.is_null() {
            return format!("hip error {error} (no string)");
        }
        core::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}
