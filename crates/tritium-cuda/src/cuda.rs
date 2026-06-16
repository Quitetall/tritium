//! GPU host side for the CUDA backend. Compiled only with `--features cuda`.
//!
//! This module owns a [`cudarc`] context + default stream, loads the PTX emitted
//! by `build.rs`, and drives the addition-only TQ2_0 mpGEMM kernel. It maps every
//! `cudarc` driver error to a [`BackendError`] so the backend never panics on a
//! device failure, and reports allocation failures as
//! [`BackendError::OutOfMemory`].
//!
//! ## cudarc 0.19 API
//!
//! Ported from the 0.13 device API to the 0.19 context/stream API:
//! - [`cudarc::driver::CudaContext::new`] returns an `Arc<CudaContext>`; memory
//!   and launches go through its [`default_stream`](cudarc::driver::CudaContext::default_stream).
//! - PTX is loaded with [`CudaContext::load_module`] (taking a
//!   [`cudarc::nvrtc::Ptx`] built from our pre-compiled string) and the kernel is
//!   fetched with [`CudaModule::load_function`].
//! - Host↔device copies use the stream's `clone_htod` / `clone_dtoh` /
//!   `memcpy_dtoh`, and launches use the `launch_builder(...).arg(...).launch(cfg)`
//!   builder. cudarc's `fallback-dynamic-loading` feature dlopen's `libcuda` at
//!   runtime, so there is no build-time CUDA-toolkit-version pin (which is what
//!   lets this crate build against CUDA 13.3).
//!
//! The crate-level `#![deny(unsafe_code)]` stands; the only `unsafe` here is the
//! kernel launch (`launch_builder(...).launch` is an `unsafe fn`), behind a
//! narrowly scoped `#[allow(unsafe_code)]` with a `SAFETY:` justification — exactly
//! the pattern `tritium-runtime` uses for its `distributed_slice` statics.

use core::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use tritium_core::{GemmShape, TernaryFormat};
use tritium_format::{
    IMMA_K, IMMA_N, IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks,
};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

use crate::autotune::{
    CacheKey, CandidateResult, ShapeBucket, TileConfig, cache_dir, tune_or_load,
};
use crate::codegen::{JIT_KERNEL_NAME, compile_imma};

/// Kernel entry point — must match the `extern "C"` symbol in the `.cu` file.
/// (cudarc 0.19 keys modules by the returned [`CudaModule`] handle, not by a
/// registered module name, so only the function symbol is needed.)
const KERNEL_NAME: &str = "tq2_0_add_mpgemm";
/// The decode-oriented tiled add-only kernel (v0.30 WF-A): one warp per output,
/// one block per row with the activation row staged in shared memory.
const KERNEL_NAME_TILED: &str = "tq2_0_add_mpgemm_tiled";
/// The IMMA int8 tensor-core prefill kernel (v0.30 WF-A part 2): one warp per
/// 16×8 output tile, `mma.m16n8k32` int32 accumulate, ternary weights in the
/// [`TernaryFormat::I2sInt8`] tile interleave.
const KERNEL_NAME_IMMA: &str = "tq2_0_imma_mpgemm";
/// On-device per-token int8 absmax activation quant (W1.58A8), the first step of
/// the fused `mpgemm_with_act_quant` override.
const KERNEL_NAME_ACT_QUANT: &str = "act_quant_int8_per_token";
/// The device-resident RMSNorm decode kernel (v0.3.1) — bit-matches the host
/// `tritium_nn::ops::rmsnorm` (sequential f32 sum-of-squares, no FMA).
const KERNEL_NAME_RMSNORM: &str = "rmsnorm_f32";
/// The device-resident RoPE decode kernel (v0.3.1) — bit-matches the host
/// `tritium_nn::ops::rope_apply` (precomputed f64→f32 trig, f32 rotation, no FMA).
const KERNEL_NAME_ROPE: &str = "rope_apply_f32";
/// The device-resident softmax decode kernel (v0.3.1) — matches
/// `tritium_nn::ops::softmax_rows`; only `expf` may differ from host libm by ~1 ULP.
const KERNEL_NAME_SOFTMAX: &str = "softmax_f32";
/// Threads per block for `act_quant_int8_per_token` — must match the kernel's
/// `ACT_QUANT_THREADS` (its shared reduction is sized for this, a power of two).
const ACT_QUANT_THREADS: u32 = 256;
/// CUDA threads per block for the 1-D launch grid (simple kernel).
const THREADS_PER_BLOCK: u32 = 256;
/// Warps per block for the tiled kernel — each warp computes one output column,
/// so a block covers this many `N` at once (8 warps = 256 threads).
const WARPS_PER_BLOCK: u32 = 8;
/// Largest `K` the tiled kernel accepts: it stages `K` f32 activations in shared
/// memory (`K * 4` bytes = 32 KiB at the cap), comfortably under the 48 KiB
/// default dynamic-shared budget and covering every BitNet shape (max K = 6912).
const TILED_K_MAX: usize = 8_192;
/// Largest `M` routed to the tiled (decode) kernel. Above this the problem is
/// prefill-shaped and the one-thread-per-output kernel is the better default
/// until the IMMA tensor-core kernel lands (WF-A part 2).
const TILED_M_MAX: usize = 64;

/// The PTX produced by `build.rs` (`nvcc -ptx`). Embedded at compile time so the
/// backend needs no PTX file on disk at runtime.
const TQ2_0_ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_add.ptx"));
/// The IMMA prefill kernel + the on-device act-quant kernel, compiled by `build.rs`
/// to a SECOND PTX target at `compute_80` (the `mma.m16n8k32` int8 shape needs
/// sm_80+, above the add kernel's sm_75 floor). Embedded the same way.
const TQ2_0_IMMA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_imma.ptx"));
/// The device-resident decode kernels (v0.3.1), compiled `--fmad=false` so they
/// reproduce the host f32 ops bit-for-bit. Embedded the same way as the others.
const DECODE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode.ptx"));

/// Map a `cudarc` driver error to a [`BackendError`]. Allocation failures surface
/// as [`BackendError::OutOfMemory`]; everything else is stringified into
/// [`BackendError::Backend`] so the device error text survives.
fn driver_err(context: &str, err: &DriverError) -> BackendError {
    BackendError::Backend(format!("{context}: {err}"))
}

/// Device-resident packed ternary weights for one matmul operand.
///
/// Wraps a [`CudaSlice<u8>`] (the htod copy of the host-packed bytes) plus the
/// `[N, K]` geometry, the packing [`TernaryFormat`], and a format-specific stride
/// ([`Stride`]), so `mpgemm` / `mpgemm_with_act_quant` can validate and launch
/// without re-deriving them.
///
/// Internal to the crate: it crosses the [`TernaryBackend`] boundary only as a
/// `Box<dyn DeviceBuffer>`, downcast back here via [`core::any::Any`].
#[derive(Debug)]
pub(crate) struct CudaBuffer {
    /// Device allocation holding the packed bytes (TQ2_0 rows or the I2sInt8 tiles).
    device: CudaSlice<u8>,
    /// Output channels (`N`), unpadded.
    n: usize,
    /// Contraction dimension (`K`), unpadded.
    k: usize,
    /// The packing this buffer holds — gates which kernel may consume it.
    format: TernaryFormat,
    /// Format-specific addressing stride (TQ2_0 per-row bytes vs IMMA k-tile count).
    stride: Stride,
    /// Total bytes uploaded (`device.len()`), cached for [`DeviceBuffer::len_bytes`].
    bytes: usize,
}

/// Format-specific addressing metadata cached alongside a [`CudaBuffer`].
#[derive(Debug, Clone, Copy)]
enum Stride {
    /// TQ2_0: packed bytes per weight row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    Tq2_0 { row_bytes: usize },
    /// I2sInt8: the packed K-tile count (`ceil(k / IMMA_K)`), the kernel's
    /// `num_ktiles` launch argument and the per-n-tile k-tile stride.
    I2sInt8 { num_ktiles: usize },
}

impl DeviceBuffer for CudaBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A CUDA execution backend bound to a single device ordinal.
///
/// Construct with [`CudaBackend::new`]; it opens the context, loads the PTX module,
/// resolves the kernel, and caches a friendly `device_id` like `"cuda:0"`. The
/// underlying [`CudaContext`], [`CudaStream`], and [`CudaModule`] are all
/// reference-counted (`Arc`) by `cudarc`.
#[derive(Debug)]
pub struct CudaBackend {
    /// The context's default stream — all memory ops and launches go through it.
    /// The stream holds its own `Arc<CudaContext>`, so the context stays alive for
    /// as long as the backend does without a separate field.
    stream: Arc<CudaStream>,
    /// Loaded add-only PTX module (kept alive so `func`/`func_tiled` stay valid).
    _module: Arc<CudaModule>,
    /// Loaded IMMA PTX module (kept alive so `func_imma`/`func_act_quant` stay
    /// valid). A separate `compute_80` image, distinct from `_module`'s `compute_75`.
    _imma_module: Arc<CudaModule>,
    /// The resolved `tq2_0_add_mpgemm` kernel (one thread per output).
    func: CudaFunction,
    /// The resolved `tq2_0_add_mpgemm_tiled` kernel (warp per output, shared-mem
    /// staged activations) — the decode path.
    func_tiled: CudaFunction,
    /// The resolved `tq2_0_imma_mpgemm` kernel (IMMA int8 tensor-core prefill).
    func_imma: CudaFunction,
    /// The resolved `act_quant_int8_per_token` kernel (on-device W1.58A8 quant).
    func_act_quant: CudaFunction,
    /// Loaded decode PTX module (v0.3.1 device-resident forward), kept alive so its
    /// functions stay valid. Compiled `--fmad=false` to bit-match the host f32 ops.
    _decode_module: Arc<CudaModule>,
    /// The resolved `rmsnorm_f32` decode kernel (bit-matches `ops::rmsnorm`).
    // Read only by `rmsnorm` (test-exercised today); `forward_device` wires it into
    // the per-token decode next. W1-in-progress.
    #[allow(dead_code)]
    func_rmsnorm: CudaFunction,
    /// The resolved `rope_apply_f32` decode kernel (bit-matches `ops::rope_apply`).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    func_rope: CudaFunction,
    /// The resolved `softmax_f32` decode kernel (matches `ops::softmax_rows`; expf may
    /// differ ~1 ULP).
    #[allow(dead_code)] // wired into `forward_device` (attention) next; W1-in-progress.
    func_softmax: CudaFunction,
    /// Backend identifier, e.g. `"cuda:0"`.
    device_id: String,
    /// Human-readable device name reported by the driver, e.g. `"NVIDIA H100"`.
    device_name: String,
    /// The device's SM arch tag (`"sm_89"` on the 4090), part of the autotune
    /// [`CacheKey`] so a tuned tile is never reused across architectures.
    sm_arch: String,
    /// The CUDA driver version (`cuDriverGetVersion`, e.g. `13030`), the cache
    /// invalidation axis — a driver bump can change the JIT'd SASS, so a stale tuned
    /// entry keyed under the old version is ignored.
    cuda_version: u32,
    /// Process-lifetime cache of JIT-compiled IMMA functions, keyed by the exact
    /// [`TileConfig`] they were rendered for. Compiling a kernel via nvrtc is
    /// expensive, so each distinct tile is compiled at most once per process; the
    /// owning [`CudaModule`] is held alongside so the [`CudaFunction`] stays valid.
    /// A `Mutex` makes the backend `Sync` (the spec trait does not require it, but
    /// the runtime may share a backend across threads). Determinism is unaffected:
    /// the same tile always renders the same source → the same SASS → the same
    /// numerics, whether read from this cache or freshly compiled.
    imma_jit: Mutex<HashMap<TileConfig, (Arc<CudaModule>, CudaFunction)>>,
    /// Per-(arch,dtype,shape-bucket,version) resolved winning tile, memoised in
    /// memory so a repeated shape does not re-hit the on-disk cache / re-tune. Seeded
    /// from the on-disk cache via [`tune_or_load`] on first use of a bucket.
    tuned_tiles: Mutex<HashMap<CacheKey, TileConfig>>,
}

impl CudaBackend {
    /// Open CUDA device `ordinal`, load the TQ2_0 add kernel, and return a backend.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if the device cannot be opened, the PTX module
    /// fails to load, or the kernel symbol is missing (no driver, no GPU, malformed
    /// PTX, …).
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| driver_err("open cuda device", &e))?;
        let stream = ctx.default_stream();

        let module = ctx
            .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
            .map_err(|e| driver_err("load tq2_0_add ptx", &e))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| driver_err("resolve tq2_0_add kernel", &e))?;
        let func_tiled = module
            .load_function(KERNEL_NAME_TILED)
            .map_err(|e| driver_err("resolve tq2_0_add_tiled kernel", &e))?;

        // The IMMA kernels live in their own compute_80 PTX (the add kernel's
        // compute_75 PTX cannot assemble `mma.m16n8k32`). Loading it JITs to the
        // present device, which the GPU lane guarantees is sm_80+ (the 4090 is
        // sm_89). A pre-sm_80 device would fail here, which is the correct error.
        let imma_module = ctx
            .load_module(Ptx::from_src(TQ2_0_IMMA_PTX))
            .map_err(|e| driver_err("load tq2_0_imma ptx", &e))?;
        let func_imma = imma_module
            .load_function(KERNEL_NAME_IMMA)
            .map_err(|e| driver_err("resolve tq2_0_imma kernel", &e))?;
        let func_act_quant = imma_module
            .load_function(KERNEL_NAME_ACT_QUANT)
            .map_err(|e| driver_err("resolve act_quant kernel", &e))?;

        // The v0.3.1 device-resident decode kernels (their own `--fmad=false` PTX).
        let decode_module = ctx
            .load_module(Ptx::from_src(DECODE_PTX))
            .map_err(|e| driver_err("load decode ptx", &e))?;
        let func_rmsnorm = decode_module
            .load_function(KERNEL_NAME_RMSNORM)
            .map_err(|e| driver_err("resolve rmsnorm kernel", &e))?;
        let func_rope = decode_module
            .load_function(KERNEL_NAME_ROPE)
            .map_err(|e| driver_err("resolve rope kernel", &e))?;
        let func_softmax = decode_module
            .load_function(KERNEL_NAME_SOFTMAX)
            .map_err(|e| driver_err("resolve softmax kernel", &e))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_owned());

        // SM arch tag for the autotune cache key (e.g. "sm_89" on the 4090). Read the
        // device's compute capability via the driver attributes; default to the IMMA
        // floor `sm_80` if the query fails (the kernel requires sm_80+ anyway).
        let sm_arch = query_sm_arch(&ctx);
        // CUDA driver version for cache invalidation (e.g. 13030 for 13.3).
        let cuda_version = query_driver_version();

        Ok(Self {
            stream,
            _module: module,
            _imma_module: imma_module,
            func,
            func_tiled,
            func_imma,
            func_act_quant,
            _decode_module: decode_module,
            func_rmsnorm,
            func_rope,
            func_softmax,
            device_id: format!("cuda:{ordinal}"),
            device_name,
            sm_arch,
            cuda_version,
            imma_jit: Mutex::new(HashMap::new()),
            tuned_tiles: Mutex::new(HashMap::new()),
        })
    }

    /// Packed bytes per weight row for `k` trits in TQ2_0.
    fn row_bytes(k: usize) -> usize {
        num_blocks(k) * TQ2_0_BLOCK_BYTES
    }

    /// Device RMSNorm on host slices (htod → launch → dtoh), **bit-matching**
    /// `tritium_nn::ops::rmsnorm`. A building block for the v0.3.1 device-resident
    /// forward (which calls the same kernel on already-resident buffers, no copies)
    /// and the target of the bit-exact golden.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if `x`/`w`/`out` lengths disagree; device
    /// failures via the cudarc mapping.
    // Test-exercised (the bit-match golden) until `forward_device` calls it on
    // resident buffers; W1-in-progress.
    #[allow(dead_code)]
    pub(crate) fn rmsnorm(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let n = x.len();
        if w.len() != n || out.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: w.len().min(out.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| driver_err("rmsnorm htod x", &e))?;
        let d_w = self
            .stream
            .clone_htod(w)
            .map_err(|e| driver_err("rmsnorm htod w", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| driver_err("rmsnorm alloc out", &e))?;

        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_rmsnorm);
        launch
            .arg(&d_x)
            .arg(&d_w)
            .arg(&eps)
            .arg(&n_i)
            .arg(&mut d_out);

        // SAFETY: kernel signature is `(const float* x, const float* w, const float
        // eps, const int n, float* out)`; the args are pushed in that exact order and
        // type. `d_x`/`d_w` are length-`n` device buffers, `d_out` the single length-`n`
        // output, `eps`/`n_i` scalars by value. One block; the kernel bounds its loops
        // by `n`, so no thread reads past any buffer.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch rmsnorm", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("rmsnorm dtoh", &e))?;
        Ok(())
    }

    /// Device RoPE for one token (M=1 decode) on host slices, **bit-matching**
    /// `tritium_nn::ops::rope_apply`. `cos_t`/`sin_t` are the `head_dim/2` precomputed
    /// f32 trig values for the token's absolute position (built host-side identically
    /// to the host op). In-place on `x` (`[n_head * head_dim]`). A building block for
    /// the device-resident forward.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on a length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn rope(
        &self,
        x: &mut [f32],
        cos_t: &[f32],
        sin_t: &[f32],
        n_head: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let half = head_dim / 2;
        if x.len() != n_head * head_dim || cos_t.len() != half || sin_t.len() != half {
            return Err(BackendError::ShapeMismatch {
                expected: n_head * head_dim,
                got: x.len(),
            });
        }
        let total = n_head * half;
        if total == 0 {
            return Ok(());
        }
        let mut d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| driver_err("rope htod x", &e))?;
        let d_cos = self
            .stream
            .clone_htod(cos_t)
            .map_err(|e| driver_err("rope htod cos", &e))?;
        let d_sin = self
            .stream
            .clone_htod(sin_t)
            .map_err(|e| driver_err("rope htod sin", &e))?;

        let n_head_i = n_head as i32;
        let head_dim_i = head_dim as i32;
        let threads = 256u32;
        let grid = (total as u32).div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_rope);
        launch
            .arg(&mut d_x)
            .arg(&d_cos)
            .arg(&d_sin)
            .arg(&n_head_i)
            .arg(&head_dim_i);

        // SAFETY: kernel signature `(float* x, const float* cos, const float* sin,
        // int n_head, int head_dim)`; args pushed in that order/type. `d_x` is the
        // length-`n_head*head_dim` in-place buffer, `d_cos`/`d_sin` are length-`half`;
        // the grid covers `n_head*half` threads each guarded by `idx >= total`, and
        // every (head,j) pair is owned by exactly one thread (race-free in-place).
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch rope", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_x, x)
            .map_err(|e| driver_err("rope dtoh", &e))?;
        Ok(())
    }

    /// Device row-wise softmax on host slices, in-place on `x` (`[rows, row_len]`).
    /// Matches `tritium_nn::ops::softmax_rows` except possibly `expf` (device libm vs
    /// host glibc). A building block for the device-resident attention.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on a length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` (attention) next; W1-in-progress.
    pub(crate) fn softmax(
        &self,
        x: &mut [f32],
        row_len: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        if row_len == 0 || x.len() != rows * row_len {
            return Err(BackendError::ShapeMismatch {
                expected: rows * row_len,
                got: x.len(),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let mut d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| driver_err("softmax htod", &e))?;
        let row_len_i = row_len as i32;
        let rows_i = rows as i32;
        let threads = 64u32;
        let grid = (rows as u32).div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_softmax);
        launch.arg(&mut d_x).arg(&row_len_i).arg(&rows_i);

        // SAFETY: kernel signature `(float* x, int row_len, int rows)`; args pushed in
        // that order/type. `d_x` is the `[rows*row_len]` in-place buffer; one thread
        // per row, guarded by `row >= rows`, each touching only its own row.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch softmax", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_x, x)
            .map_err(|e| driver_err("softmax dtoh", &e))?;
        Ok(())
    }

    /// Pick the add-only kernel for this problem shape. The tiled (decode) kernel
    /// wins for small `M` and is bounded by its shared-memory activation stage
    /// (`K * 4` ≤ 48 KiB); everything else uses the one-thread-per-output kernel.
    fn select_add_kernel(m: usize, k: usize) -> AddKernel {
        if m > 0 && m <= TILED_M_MAX && k <= TILED_K_MAX {
            AddKernel::Tiled
        } else {
            AddKernel::Simple
        }
    }

    /// Run one TQ2_0 add-only mpGEMM through the chosen kernel. Shared by the
    /// public [`TernaryBackend::mpgemm`] (which auto-selects) and the tests (which
    /// force each kernel so both stay gated against the reference).
    ///
    /// # Errors
    /// Validation [`BackendError::ShapeMismatch`] / [`BackendError::UnsupportedFormat`]
    /// as documented on [`TernaryBackend::mpgemm`]; device failures via the cudarc
    /// error mapping.
    fn mpgemm_kernel(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
        kernel: AddKernel,
    ) -> Result<(), BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        // The add path consumes TQ2_0 rows; an I2sInt8 buffer would mis-address.
        let Stride::Tq2_0 { row_bytes } = buf.stride else {
            return Err(BackendError::UnsupportedFormat(buf.format));
        };

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
        if m == 0 || n == 0 {
            return Ok(());
        }

        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let d_scales = self
            .stream
            .clone_htod(scales)
            .map_err(|e| alloc_or_backend("upload scales (htod)", &e, scales.len() * 4))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        // Kernel-specific launch geometry. Both kernels take the identical argument
        // list, so only the function handle and the grid/shared config differ.
        let (func, cfg) = match kernel {
            AddKernel::Simple => {
                let total = (m * n) as u32;
                let grid = total.div_ceil(THREADS_PER_BLOCK);
                (
                    &self.func,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
            AddKernel::Tiled => {
                // `select_add_kernel` only routes K within the shared budget here;
                // assert it for direct callers (the tests) so an oversized K fails
                // loudly rather than as a cryptic CUDA shared-mem launch error.
                debug_assert!(
                    k <= TILED_K_MAX,
                    "tiled kernel K={k} exceeds the {TILED_K_MAX} shared-mem cap"
                );
                // One warp per output column → a block covers WARPS_PER_BLOCK of N;
                // one block-row per M. Shared memory stages this row's K acts.
                let grid_n = (n as u32).div_ceil(WARPS_PER_BLOCK);
                (
                    &self.func_tiled,
                    LaunchConfig {
                        grid_dim: (grid_n, m as u32, 1),
                        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                        shared_mem_bytes: (k * 4) as u32,
                    },
                )
            }
        };

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let row_bytes_i = row_bytes as i32;

        let mut launch = self.stream.launch_builder(func);
        launch
            .arg(&d_act)
            .arg(&buf.device)
            .arg(&d_scales)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&row_bytes_i);

        // SAFETY: `LaunchArgs::launch` is `unsafe` because the kernel signature is
        // not type-checked against the pushed args. Both `tq2_0_add_mpgemm` and
        // `tq2_0_add_mpgemm_tiled` declare the identical parameter list
        // (`const float*`, `const unsigned char*`, `const float*`, `float*`, then
        // four `int`s), pushed here in that exact order/type. Only `d_out` is
        // mutable (the single `float* out`). Device buffers were sized against
        // `shape` above / in `upload_weights`; the tiled grid covers `M` rows ×
        // `ceil(N / WARPS_PER_BLOCK)` warp-columns with bounds checks (`mi >= m`,
        // `ni >= n`) inside the kernel, and the shared request is `K * 4` bytes,
        // matching the kernel's `extern __shared__ float[K]`. All host scalars
        // outlive the launch.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch tq2_0_add", &e))?;
        }

        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }

    /// The fused W1.58A8 path on the IMMA int8 tensor cores: quantize `act` to
    /// per-token int8 **on device**, contract against the [`TernaryFormat::I2sInt8`]
    /// weights with `mma.m16n8k32`, and fold both the per-token activation scale and
    /// the per-channel `weight_scales` into the `f32` output — all without an extra
    /// host pass or an H2D round-trip of the quantized activations.
    ///
    /// Validation mirrors the spec default's: `act`/`out`/`weight_scales` lengths vs
    /// `shape`, the buffer's `[N, K]` geometry, and that the buffer is an I2sInt8
    /// packing. Empty `M`/`N` is a no-op.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] / [`BackendError::UnsupportedFormat`] on bad
    /// inputs; device failures via the cudarc error mapping.
    fn imma_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        let Stride::I2sInt8 { num_ktiles } = buf.stride else {
            return Err(BackendError::UnsupportedFormat(buf.format));
        };

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
        if weight_scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: weight_scales.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }
        if m == 0 || n == 0 {
            return Ok(());
        }

        // --- Upload activations; alloc the on-device int8 quant + per-token scale. ---
        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let mut d_qact = self
            .stream
            .alloc_zeros::<i8>(m * k)
            .map_err(|e| alloc_or_backend("alloc qact", &e, m * k))?;
        let mut d_act_scale = self
            .stream
            .alloc_zeros::<f32>(m)
            .map_err(|e| alloc_or_backend("alloc act_scale", &e, m * 4))?;
        let d_wscale = self
            .stream
            .clone_htod(weight_scales)
            .map_err(|e| alloc_or_backend("upload weight_scales (htod)", &e, n * 4))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let num_ktiles_i = num_ktiles as i32;

        // --- Step 1: per-token int8 absmax quant (one block per row). ---
        {
            let cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (ACT_QUANT_THREADS, 1, 1),
                shared_mem_bytes: 0, // the kernel's reduction buffer is static __shared__
            };
            let mut launch = self.stream.launch_builder(&self.func_act_quant);
            launch
                .arg(&d_act)
                .arg(&mut d_qact)
                .arg(&mut d_act_scale)
                .arg(&m_i)
                .arg(&k_i);
            // SAFETY: `launch` is `unsafe` — args are not type-checked against the
            // kernel. `act_quant_int8_per_token(const float*, signed char*, float*,
            // int, int)` is fed exactly: `d_act` (f32 [M,K], read), `d_qact` (i8
            // [M,K], written), `d_act_scale` (f32 [M], written), then `m`, `k`. The
            // grid is `M` blocks of `ACT_QUANT_THREADS` (matching the kernel's
            // `ACT_QUANT_THREADS` static shared reduction). All buffers were sized
            // against `shape`; host scalars outlive the launch.
            #[allow(unsafe_code)]
            unsafe {
                launch
                    .launch(cfg)
                    .map_err(|e| driver_err("launch act_quant", &e))?;
            }
        }

        // --- Step 2: IMMA int8 contraction, folding both scales into `out`. ---
        // Resolve the tuned tile for this shape (on-disk autotune cache → JIT), and
        // launch its kernel with the tile's geometry. The AOT-equivalent tile uses
        // the embedded AOT cubin; every other tile is JIT-compiled (and cached). All
        // tiles are numerically identical (exact int32 accumulate + one scale fold).
        let tile = self.resolve_imma_tile(shape);
        let func = self.imma_function_for_tile(tile)?;
        self.launch_imma_tile(
            &func, tile, &d_qact, &buf.device, &d_act_scale, &d_wscale, &mut d_out, m_i, n_i, k_i,
            num_ktiles_i,
        )?;

        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }

    /// Launch the IMMA contraction kernel `func` (rendered/compiled for `tile`) with
    /// the launch geometry `tile` dictates: a grid of `ceil(N/tile_n) ×
    /// ceil(M/tile_m)` blocks of `tile.warps · 32` threads. The kernel's staging is
    /// static `__shared__`, so `shared_mem_bytes` stays 0. Argument order/types are
    /// identical for the AOT and every JIT tile (the codegen pins the signature).
    ///
    /// # Errors
    /// The cudarc launch error, mapped to [`BackendError::Backend`].
    #[allow(clippy::too_many_arguments)]
    fn launch_imma_tile(
        &self,
        func: &CudaFunction,
        tile: TileConfig,
        d_qact: &CudaSlice<i8>,
        d_weights: &CudaSlice<u8>,
        d_act_scale: &CudaSlice<f32>,
        d_wscale: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        m_i: i32,
        n_i: i32,
        k_i: i32,
        num_ktiles_i: i32,
    ) -> Result<(), BackendError> {
        let grid_n = (n_i as u32).div_ceil(tile.tile_n as u32);
        let grid_m = (m_i as u32).div_ceil(tile.tile_m as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid_n, grid_m, 1),
            block_dim: (tile.block_threads(), 1, 1),
            shared_mem_bytes: 0, // the kernel's staging is static __shared__
        };
        let mut launch = self.stream.launch_builder(func);
        launch
            .arg(d_qact)
            .arg(d_weights)
            .arg(d_act_scale)
            .arg(d_wscale)
            .arg(d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&num_ktiles_i);
        // SAFETY: `launch` is `unsafe`. Both the AOT `tq2_0_imma_mpgemm` and every
        // JIT-rendered tile declare the identical signature `(const signed char*,
        // const unsigned char*, const float*, const float*, float*, int, int, int,
        // int)` (the codegen pins it), fed here in that exact order/type: `d_qact`
        // (i8 [M,K]), `d_weights` (I2sInt8 tile bytes, validated to the packed length
        // in `upload_weights`), `d_act_scale` (f32 [M]), `d_wscale` (f32 [N]),
        // `d_out` (f32 [M,N], the only mutable operand), then `m,n,k,num_ktiles`. The
        // grid covers `ceil(M/tile_m)·ceil(N/tile_n)` blocks; the kernel
        // bounds-checks every global (`gm<m`, `gn<n`, `kt<num_ktiles`) and addresses
        // the weight tiles with `num_ktiles`, the stride cached in the buffer. Host
        // scalars outlive the launch.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch tq2_0_imma (tiled)", &e))?;
        }
        Ok(())
    }

    /// The autotune cache key for an IMMA launch of `shape` on this device.
    fn imma_cache_key(&self, shape: GemmShape) -> CacheKey {
        CacheKey {
            arch: self.sm_arch.clone(),
            dtype: IMMA_DTYPE_TAG,
            bucket: ShapeBucket::from_shape(shape),
            cuda_version: self.cuda_version,
        }
    }

    /// Resolve the winning [`TileConfig`] for `shape`: memoised in memory, seeded from
    /// the on-disk autotune cache (and a device tile search on a cold cache) via
    /// [`tune_or_load`].
    ///
    /// On any device-side hiccup during the search the policy falls back to
    /// [`TileConfig::AOT_EQUIVALENT`] (the guaranteed-correct anchor) without caching
    /// it, so correctness is never at risk — only performance. Because every tile is
    /// numerically identical, the resolved tile choice never affects the result.
    fn resolve_imma_tile(&self, shape: GemmShape) -> TileConfig {
        let key = self.imma_cache_key(shape);
        if let Some(t) = self.tuned_tiles.lock().expect("tuned_tiles poisoned").get(&key) {
            return *t;
        }
        let dir = cache_dir();
        // The device-side evaluation half: JIT + launch each candidate on this shape,
        // validate it against a host reference, and time it.
        let tile = tune_or_load(
            &dir,
            &key,
            |cand| self.evaluate_candidate(cand, shape),
            |e| eprintln!("tritium-cuda: autotune cache write failed ({e}); continuing un-cached"),
        );
        self.tuned_tiles
            .lock()
            .expect("tuned_tiles poisoned")
            .insert(key, tile);
        tile
    }

    /// Evaluate one candidate `tile` on `shape`: JIT-compile it, run it on a small
    /// deterministic problem of this shape, check the result against the host
    /// reference (the same exact-int contraction the kernel computes), and time it.
    /// A compile/launch failure or an out-of-tolerance result marks the candidate
    /// incorrect (rejected), never aborting the search.
    fn evaluate_candidate(&self, tile: TileConfig, shape: GemmShape) -> CandidateResult {
        match self.try_evaluate_candidate(tile, shape) {
            Ok(r) => r,
            Err(_) => CandidateResult { correct: false, seconds: f64::INFINITY },
        }
    }

    /// Fallible inner body of [`evaluate_candidate`]; any `Err` is folded to an
    /// "incorrect" candidate by the caller.
    fn try_evaluate_candidate(
        &self,
        tile: TileConfig,
        shape: GemmShape,
    ) -> Result<CandidateResult, BackendError> {
        let GemmShape { m, n, k } = shape;
        // Guard against degenerate buckets (M=0 etc.): use at least one row/col/tile.
        let m = m.max(1);
        let n = n.max(1);
        let k = k.div_ceil(IMMA_K).max(1) * IMMA_K; // round K up to a whole k-tile

        // Deterministic int8 activations, ternary weights, and scales for the probe.
        let qact: Vec<i8> = (0..m * k).map(|i| ((i % 5) as i8) - 2).collect();
        let act_scale: Vec<f32> = (0..m).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
        let wscale: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();
        let trits: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();

        // Host reference: exact int32 contraction folded by the two scales — the same
        // arithmetic the kernel performs, so a correct tile matches to the bit modulo
        // the single f32 fold's rounding (within the IMMA tolerance).
        let mut reference = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc: i64 = 0;
                for ki in 0..k {
                    acc += qact[mi * k + ki] as i64 * trits[ni * k + ki] as i64;
                }
                reference[mi * n + ni] = act_scale[mi] * wscale[ni] * acc as f32;
            }
        }

        // Pack the weights into the I2sInt8 tile layout the kernel expects.
        let packed = pack_i2s_int8_tiles(&trits, n, k);

        // Upload operands.
        let d_qact = self
            .stream
            .clone_htod(&qact)
            .map_err(|e| driver_err("autotune upload qact", &e))?;
        let d_weights = self
            .stream
            .clone_htod(&packed)
            .map_err(|e| driver_err("autotune upload weights", &e))?;
        let d_act_scale = self
            .stream
            .clone_htod(&act_scale)
            .map_err(|e| driver_err("autotune upload act_scale", &e))?;
        let d_wscale = self
            .stream
            .clone_htod(&wscale)
            .map_err(|e| driver_err("autotune upload wscale", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| driver_err("autotune alloc out", &e))?;

        let num_ktiles_i = (k / IMMA_K) as i32;
        let func = self.imma_function_for_tile(tile)?;

        // One correctness launch + readback.
        self.launch_imma_tile(
            &func, tile, &d_qact, &d_weights, &d_act_scale, &d_wscale, &mut d_out, m as i32,
            n as i32, k as i32, num_ktiles_i,
        )?;
        let mut got = vec![0.0f32; m * n];
        self.stream
            .memcpy_dtoh(&d_out, &mut got)
            .map_err(|e| driver_err("autotune dtoh", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("autotune sync", &e))?;

        let correct = got.iter().zip(&reference).all(|(&g, &r)| imma_close(g, r));
        if !correct {
            return Ok(CandidateResult { correct: false, seconds: f64::INFINITY });
        }

        // Time a few repetitions (median of the wall-clock per-launch time). This is a
        // coarse but deterministic-enough metric for tile selection; the gate only
        // requires that the *winner* be correct, not that timings be reproducible to
        // the nanosecond.
        const REPS: usize = 8;
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let start = std::time::Instant::now();
            self.launch_imma_tile(
                &func, tile, &d_qact, &d_weights, &d_act_scale, &d_wscale, &mut d_out, m as i32,
                n as i32, k as i32, num_ktiles_i,
            )?;
            self.stream
                .synchronize()
                .map_err(|e| driver_err("autotune timing sync", &e))?;
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let seconds = times[times.len() / 2];
        Ok(CandidateResult { correct: true, seconds })
    }

    /// Resolve the IMMA kernel [`CudaFunction`] for `tile`.
    ///
    /// The [`TileConfig::AOT_EQUIVALENT`] tile is served by the **embedded AOT cubin**
    /// (`func_imma`, built by `build.rs`) — the common-shape fast path that needs no
    /// nvrtc at all. Every other tile is JIT-compiled via nvrtc on first use and
    /// cached for the process lifetime. This is the cold-cache (JIT) == warm-cache
    /// (AOT) contract made concrete: the AOT-equivalent tile's two realisations (the
    /// embedded cubin here, and a fresh JIT compile via [`imma_jit_function`]) are
    /// bit-identical, which the determinism test asserts explicitly.
    ///
    /// The returned [`CudaFunction`] is `Arc`-backed and holds its owning
    /// [`CudaModule`] alive internally, so it stays valid after the cache lock is
    /// dropped.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] for an invalid tile; [`BackendError::Backend`]
    /// on an nvrtc compile or module-load failure.
    fn imma_function_for_tile(&self, tile: TileConfig) -> Result<CudaFunction, BackendError> {
        // Common-shape fast path: the AOT-equivalent tile is the embedded cubin.
        if tile == TileConfig::AOT_EQUIVALENT {
            return Ok(self.func_imma.clone());
        }
        if let Some((_, f)) = self.imma_jit.lock().expect("imma_jit poisoned").get(&tile) {
            return Ok(f.clone());
        }
        let (module, func) = self.imma_jit_function(tile)?;
        let mut guard = self.imma_jit.lock().expect("imma_jit poisoned");
        // Another thread may have raced us between the unlock above and here; keep
        // whichever entry is present (both are numerically identical), and return its
        // function so every caller observes the same handle.
        let (_, f) = guard.entry(tile).or_insert((module, func));
        Ok(f.clone())
    }

    /// JIT-compile + load the IMMA kernel for `tile` (no process cache); returns the
    /// owning module plus the freshly resolved [`CudaFunction`]. Used by
    /// [`imma_function_for_tile`] and, directly, by the cold-vs-warm determinism test.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] / [`BackendError::Backend`] as
    /// [`compile_imma`] / module load report.
    fn imma_jit_function(
        &self,
        tile: TileConfig,
    ) -> Result<(Arc<CudaModule>, CudaFunction), BackendError> {
        let ptx = compile_imma(tile, IMMA_JIT_ARCH)?;
        let module = self
            .stream
            .context()
            .load_module(ptx)
            .map_err(|e| driver_err("load JIT IMMA module", &e))?;
        let func = module
            .load_function(JIT_KERNEL_NAME)
            .map_err(|e| driver_err("resolve JIT IMMA kernel", &e))?;
        Ok((module, func))
    }
}

/// The weight-dtype tag the IMMA path keys its autotune cache under (the I2sInt8
/// packing). Distinct from the add-only `tq2_0` path so the two never share a tuned
/// tile (they have different kernels entirely).
const IMMA_DTYPE_TAG: &str = "i2sint8";

/// The virtual arch the JIT path renders for: `compute_80`, matching the AOT IMMA
/// PTX target (`build.rs`'s `IMMA_MIN_ARCH`). nvrtc emits forward-compatible PTX
/// that the driver JITs to the present device, exactly as the AOT PTX is loaded —
/// so a JIT'd tile and the AOT cubin for the *same* tile go through the identical
/// driver back-end and produce bit-identical SASS/output (the cold==warm gate).
const IMMA_JIT_ARCH: &str = "compute_80";

/// Read the device compute capability and format it as an `sm_XY` tag for the
/// autotune cache key. Falls back to the IMMA floor `sm_80` if the query fails (the
/// kernel needs sm_80+ regardless, so this is the most conservative correct default).
fn query_sm_arch(ctx: &Arc<CudaContext>) -> String {
    match ctx.compute_capability() {
        Ok((major, minor)) => format!("sm_{major}{minor}"),
        Err(_) => "sm_80".to_owned(),
    }
}

/// Query the CUDA driver version via `cuDriverGetVersion` (e.g. `13030` for 13.3),
/// used as the autotune cache invalidation axis. Returns `0` if the query fails,
/// which still keys a stable (if uninformative) cache slot.
fn query_driver_version() -> u32 {
    let mut version: core::ffi::c_int = 0;
    // SAFETY: `cuDriverGetVersion` writes a single `int` through the supplied
    // pointer and returns a `CUresult`; `version` is a live local `c_int` that
    // outlives the call. cudarc's `fallback-dynamic-loading` resolves the symbol from
    // the dlopen'd `libcuda` (the same driver the rest of the backend uses). The crate
    // forbids unsafe by default; this scoped allow mirrors the kernel-launch sites.
    #[allow(unsafe_code)]
    let res = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
    if res == cudarc::driver::sys::CUresult::CUDA_SUCCESS && version >= 0 {
        version as u32
    } else {
        0
    }
}

/// Which add-only kernel a launch should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddKernel {
    /// One thread per output element — the v0.10 kernel; general fallback.
    Simple,
    /// One warp per output, shared-mem staged activations — the decode path.
    Tiled,
}

impl TernaryBackend for CudaBackend {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn capabilities(&self) -> DeviceCaps {
        // total_memory is left at its default (unknown): the contract permits 0 and
        // the runtime does not rely on the figure here. Both packings this backend
        // consumes are advertised so the runtime can route either kernel here.
        DeviceCaps::new("cuda", self.device_name.clone())
            .with_features(vec!["tq2_0".to_owned(), "i2s_int8".to_owned()])
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;

        // Each format has its own packed byte length and addressing stride; TQ1_0
        // is not a GPU packing here.
        let (expected, stride) = match format {
            TernaryFormat::Tq2_0 => {
                let row_bytes = Self::row_bytes(k);
                (n * row_bytes, Stride::Tq2_0 { row_bytes })
            }
            TernaryFormat::I2sInt8 => {
                // The IMMA tile interleave (see `tritium_format::convert_i2s_to_int8`):
                // `ceil(N/IMMA_N) · ceil(K/IMMA_K) · IMMA_WTILE_BYTES` bytes.
                let num_ntiles = n.div_ceil(IMMA_N);
                let num_ktiles = k.div_ceil(IMMA_K);
                (
                    num_ntiles * num_ktiles * IMMA_WTILE_BYTES,
                    Stride::I2sInt8 { num_ktiles },
                )
            }
            other => return Err(BackendError::UnsupportedFormat(other)),
        };

        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} ({format:?})",
                packed.len()
            )));
        }

        // htod copy of the packed bytes. A driver OOM here is reported as such.
        let device = self.stream.clone_htod(packed).map_err(|e| {
            if is_oom(&e) {
                BackendError::OutOfMemory {
                    requested: expected,
                }
            } else {
                driver_err("upload weights (htod)", &e)
            }
        })?;

        Ok(Box::new(CudaBuffer {
            device,
            n,
            k,
            format,
            stride,
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
        // Auto-select the add-only kernel by shape (decode → tiled), then run it.
        // All validation + the launch live in `mpgemm_kernel`.
        let kernel = Self::select_add_kernel(shape.m, shape.k);
        self.mpgemm_kernel(act, weights, scales, shape, format, out, kernel)
    }

    /// CUDA override of the fused W1.58A8 path. For an [`TernaryFormat::I2sInt8`]
    /// buffer this drives the on-device quant + IMMA int8 kernel
    /// ([`imma_with_act_quant`](CudaBackend::imma_with_act_quant)), dropping the
    /// host quant pass + the H2D round-trip the spec default would do. For a TQ2_0
    /// buffer (no int8 tensor-core path) it defers to the spec default — host
    /// per-token quant → [`mpgemm`](TernaryBackend::mpgemm) (add-only kernel) → fold
    /// — so both packings are served and the override stays within the `mpgemm`
    /// tolerance of that default (ADR 0005's "fused == host-A8" gate).
    ///
    /// # Errors
    /// As [`imma_with_act_quant`](CudaBackend::imma_with_act_quant) /
    /// [`mpgemm`](TernaryBackend::mpgemm) document, plus
    /// [`BackendError::InvalidInput`] if the buffer is not a CUDA buffer.
    fn mpgemm_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        match buf.format {
            TernaryFormat::I2sInt8 => {
                self.imma_with_act_quant(act, weights, weight_scales, shape, out)
            }
            // TQ2_0 (and anything else uploadable) → the host-quant default, which
            // delegates to the add-only `mpgemm`. `format` must agree with the
            // buffer; the default's `mpgemm` re-checks it.
            _ => self.default_mpgemm_with_act_quant(act, weights, weight_scales, shape, format, out),
        }
    }
}

/// Free function holding the spec's default `mpgemm_with_act_quant` body, so the
/// CUDA override can fall back to it for non-IMMA (TQ2_0) buffers without losing
/// the on-device IMMA path for I2sInt8. It is the literal host-A8 reference:
/// per-token int8 absmax quant on the host, then [`TernaryBackend::mpgemm`], then
/// fold the per-token scale — identical to [`tritium_spec`]'s provided default and
/// to the v0.20 caller-side quant.
impl CudaBackend {
    fn default_mpgemm_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
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

        // Per-token int8 absmax quant kept in f32 (the f32 `mpgemm` consumes it
        // directly), plus the per-token dequant multiplier. Matches the spec default
        // / `tritium-nn::ops::act_quant` (round-half-to-even, clamp [-128, 127]).
        let mut q = vec![0.0_f32; m * k];
        let mut act_scale = vec![0.0_f32; m];
        quantize_act_int8_host(act, m, k, &mut q, &mut act_scale);

        self.mpgemm(&q, weights, weight_scales, shape, format, out)?;

        // Fold the per-token activation scale: out[m,n] *= act_scale[m].
        for (row, &s) in out.chunks_exact_mut(n).zip(act_scale.iter()) {
            for v in row {
                *v *= s;
            }
        }
        Ok(())
    }
}

/// Symmetric int8 activation-quant positive cap (`Qp`). Matches
/// `tritium_spec`'s `A8_QB` / `tritium-nn::ops::act_quant::QB` and the device
/// `act_quant_int8_per_token` kernel.
const A8_QB: f32 = 127.0;

/// Host per-token int8 absmax quant — the fallback used by
/// [`CudaBackend::default_mpgemm_with_act_quant`] for TQ2_0 buffers. A verbatim copy
/// of `tritium_spec::quantize_act_int8` (which cannot be imported — it is private to
/// that crate). The fused-vs-default gate keeps the two from drifting: the IMMA
/// device quant, this host copy, and the spec copy all reproduce the same int8
/// values, pinned by `act_quant` parity in the spec's own `act_quant_golden` test.
fn quantize_act_int8_host(act: &[f32], m: usize, k: usize, q_out: &mut [f32], scale_out: &mut [f32]) {
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
        for (q, &v) in out_row.iter_mut().zip(row) {
            *q = (v * s).round_ties_even().clamp(-128.0, A8_QB);
        }
        scale_out[r] = gamma / A8_QB;
    }
}

/// The IMMA conformance tolerance the autotune correctness check uses (relative,
/// matching ADR 0002's `1e-4`). A candidate tile whose output lands outside this band
/// is a launch-geometry bug (the int contraction is exact), so it is rejected.
const IMMA_AUTOTUNE_REL_TOL: f32 = 1e-4;

/// Relative-or-absolute closeness check for the autotune correctness gate: accept if
/// the absolute error is within `IMMA_AUTOTUNE_REL_TOL · max(1, |reference|)`. Mirrors
/// the testkit's `Tolerance::accepts` shape so the in-tree probe matches the
/// committed conformance gate.
fn imma_close(got: f32, reference: f32) -> bool {
    let diff = (got - reference).abs();
    diff <= IMMA_AUTOTUNE_REL_TOL * reference.abs().max(1.0)
}

/// Pack an `[N, K]` ternary weight matrix (`i8` codes in `{-1,0,+1}`, row-major) into
/// the IMMA `I2sInt8` tile layout the kernel reads (the same interleave
/// `tritium_format::convert_i2s_to_int8` produces): `ceil(N/8) · ceil(K/32)` tiles of
/// 64 bytes, tile `(nt, kt)` at flat byte offset `(nt·num_ktiles + kt)·64`, each tile
/// 8×32 codes in `(n_in_tile, k_in_tile)` row-major, 4 codes/byte (low pair = first
/// element), `code = trit + 1`. Padding rows/cols past `n`/`k` carry trit 0 (code 1),
/// which contributes nothing to the int32 sum.
///
/// Used only by the autotune probe (it generates its own weights), so it packs the
/// tile layout directly rather than round-tripping the block-striped I2_S payload.
fn pack_i2s_int8_tiles(trits: &[i8], n: usize, k: usize) -> Vec<u8> {
    let num_ntiles = n.div_ceil(IMMA_N);
    let num_ktiles = k.div_ceil(IMMA_K);
    let mut out = vec![0u8; num_ntiles * num_ktiles * IMMA_WTILE_BYTES];
    for nt in 0..num_ntiles {
        for kt in 0..num_ktiles {
            let tile0 = (nt * num_ktiles + kt) * IMMA_WTILE_BYTES;
            // 256 codes per tile in (n_in_tile, k_in_tile) row-major order.
            for e in 0..(IMMA_N * IMMA_K) {
                let n_in_tile = e / IMMA_K;
                let k_in_tile = e % IMMA_K;
                let gn = nt * IMMA_N + n_in_tile;
                let gk = kt * IMMA_K + k_in_tile;
                // In-range → real trit; padding → 0.
                let trit = if gn < n && gk < k { trits[gn * k + gk] } else { 0 };
                let code = (trit + 1) as u8; // {-1,0,1} -> {0,1,2}
                let byte = tile0 + e / 4;
                let shift = 2 * (e % 4);
                out[byte] |= code << shift;
            }
        }
    }
    out
}

/// Heuristic: did this driver error come from an allocation running out of memory?
fn is_oom(err: &DriverError) -> bool {
    // `DriverError`'s Display includes the CUDA status string; the out-of-memory
    // status renders as "out of memory". This keeps us off the unstable numeric
    // status value while still classifying the common case.
    format!("{err}")
        .to_ascii_lowercase()
        .contains("out of memory")
}

/// Classify an allocation/copy failure as OOM (with the requested byte count) or a
/// generic backend error.
fn alloc_or_backend(context: &str, err: &DriverError, requested: usize) -> BackendError {
    if is_oom(err) {
        BackendError::OutOfMemory { requested }
    } else {
        driver_err(context, err)
    }
}

/// Construct the backend on device 0 for the runtime registry.
///
/// Returns `Err` (which the registry logs and skips) when no CUDA device is
/// available — the expected case on cpu-only machines that still link this crate.
fn init_cuda() -> Result<Box<dyn TernaryBackend>, BackendError> {
    Ok(Box::new(CudaBackend::new(0)?))
}

// Self-register into the runtime's distributed slice, but only with the `cuda`
// feature. `linkme`'s `distributed_slice` expands to a `#[link_section]` static
// that trips the `unsafe_code` lint, hence the scoped allow (same pattern as
// `tritium-runtime`'s own registrations).
#[allow(unsafe_code)]
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static CUDA: BackendEntry = BackendEntry {
    name: "cuda",
    init: init_cuda,
};

#[cfg(test)]
mod tests {
    //! GPU conformance + CPU↔CUDA parity tests. Run only with `--features cuda` AND
    //! a working CUDA device, so they are exercised on the Wave D GPU CI lane, never
    //! on cpu-only lanes. When no device is present the tests self-skip
    //! (constructing the backend returns `Err`) rather than failing.
    //!
    //! `run_conformance` itself packs each vector's trits to TQ2_0 (block scale
    //! 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
    //! scales, and grades against `reference_mpgemm` — so the test only has to
    //! supply the TQ2_0 vectors this kernel supports.

    use super::*;
    use tritium_cpu::CpuBackend;
    use tritium_testkit::{ConformanceVector, Tolerance, generate_vectors, run_conformance};

    /// The full conformance set this kernel is responsible for: every TQ2_0 vector
    /// from the committed generator (the kernel does not handle TQ1_0).
    fn tq2_vectors() -> Vec<ConformanceVector> {
        let v: Vec<_> = generate_vectors(0xC0FFEE, 16)
            .into_iter()
            .filter(|v| v.format == "tq2_0")
            .collect();
        assert!(!v.is_empty(), "expected some tq2_0 conformance vectors");
        v
    }

    #[test]
    fn cuda_matches_reference_within_tolerance() {
        // Skip cleanly when no GPU is present (cpu-only dev box / wrong CI lane).
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cuda conformance: no device ({e})");
                return;
            }
        };

        let tq2 = tq2_vectors();
        let report = run_conformance(&backend, &tq2, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} cuda conformance cases failed: {:?}",
            report.failed.len(),
            report.failed
        );
    }

    /// ADR 0002 U2: CPU↔CUDA parity. The *same* committed TQ2_0 vectors run through
    /// both [`CpuBackend`] and [`CudaBackend`]; every output element must agree
    /// within `1e-4` relative. This is the load-bearing cross-backend gate — it
    /// catches a backend that is internally self-consistent (passes conformance)
    /// but disagrees with the other backend on shared inputs.
    #[test]
    fn cuda_matches_cpu_within_tolerance() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cpu<->cuda parity: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // Run both backends over the identical TQ2_0 vector set.
        let cpu_report = run_conformance(&cpu, &tq2_vectors(), tol);
        assert!(
            cpu_report.is_ok(),
            "cpu backend failed its own conformance, parity is moot: {:?}",
            cpu_report.failed
        );

        // Replay each vector through both backends and compare outputs directly,
        // rather than only against the shared reference, so any CPU/CUDA divergence
        // surfaces even within the reference tolerance band.
        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("vector weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);

            let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);
            let cuda_out = run_backend(&cuda, &packed, &v.activation, &v.scales, shape);

            assert_eq!(
                cpu_out.len(),
                cuda_out.len(),
                "{}: output len mismatch",
                v.id
            );
            for (i, (&c, &g)) in cpu_out.iter().zip(&cuda_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "{}: cpu/cuda disagree at [{i}]: cpu={c} cuda={g}",
                    v.id
                );
            }
        }
    }

    /// Pack an `[N, K]` trit matrix to TQ2_0 rows, block scale fixed to `1.0` (the
    /// testkit convention), ready for `upload_weights`.
    fn pack_tq2_0(trits: &[tritium_core::Trit], shape: GemmShape) -> Vec<u8> {
        use tritium_format::pack_tq2_0_row;
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
        }
        packed
    }

    /// Upload weights + run one TQ2_0 mpGEMM through any backend, returning `[M, N]`.
    fn run_backend<B: TernaryBackend>(
        backend: &B,
        packed: &[u8],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
    ) -> Vec<f32> {
        let buf = backend
            .upload_weights(packed, shape, TernaryFormat::Tq2_0)
            .expect("upload weights");
        let mut out = vec![0.0f32; shape.m * shape.n];
        backend
            .mpgemm(
                act,
                buf.as_ref(),
                scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .expect("mpgemm");
        out
    }

    /// Upload weights + run one TQ2_0 mpGEMM through a *forced* add kernel, so a
    /// test can gate each path independently of the shape-based auto-selection.
    fn run_kernel(
        cuda: &CudaBackend,
        packed: &[u8],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
        kernel: AddKernel,
    ) -> Vec<f32> {
        let buf = cuda
            .upload_weights(packed, shape, TernaryFormat::Tq2_0)
            .expect("upload weights");
        let mut out = vec![0.0f32; shape.m * shape.n];
        cuda.mpgemm_kernel(
            act,
            buf.as_ref(),
            scales,
            shape,
            TernaryFormat::Tq2_0,
            &mut out,
            kernel,
        )
        .expect("mpgemm_kernel");
        out
    }

    /// Both add kernels must match the CPU reference (within tolerance) on the full
    /// committed TQ2_0 conformance set. This gates the new tiled kernel directly,
    /// and re-gates the simple kernel, regardless of which one auto-selection picks.
    #[test]
    fn both_add_kernels_match_reference() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping both-kernel gate: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);
            let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);

            let simple = run_kernel(
                &cuda,
                &packed,
                &v.activation,
                &v.scales,
                shape,
                AddKernel::Simple,
            );
            for (i, (&g, &c)) in simple.iter().zip(&cpu_out).enumerate() {
                assert!(tol.accepts(g, c), "{}: simple vs cpu [{i}] {g} {c}", v.id);
            }

            // The tiled kernel only accepts K within its shared-memory budget.
            if v.k <= TILED_K_MAX {
                let tiled = run_kernel(
                    &cuda,
                    &packed,
                    &v.activation,
                    &v.scales,
                    shape,
                    AddKernel::Tiled,
                );
                for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                    assert!(tol.accepts(g, c), "{}: tiled vs cpu [{i}] {g} {c}", v.id);
                }
            }
        }
    }

    /// The tiled kernel must be correct on boundary shapes: tail `K` (not a 256
    /// multiple, so a partial final TQ2_0 block), partial warps (`N` not a multiple
    /// of `WARPS_PER_BLOCK`), partial grids (`M`/`N` of 1), and `K` at the cap.
    #[test]
    fn tiled_handles_tail_shapes() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tiled tail-shape gate: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // (M, N, K) — tail K, partial warps/blocks, single rows/cols, K at the cap.
        let shapes = [
            (1usize, 1usize, 1usize),
            (1, 7, 300),
            (5, 130, 257),
            (64, 3, 2560),
            (3, 33, 6912),
            (1, 1, TILED_K_MAX),
        ];

        for (m, n, k) in shapes {
            assert!(k <= TILED_K_MAX, "test shape K exceeds the tiled cap");
            let shape = GemmShape::new(m, n, k);

            // Deterministic ternary weights, activations, and per-channel scales.
            let trits: Vec<_> = (0..n * k)
                .map(|i| tritium_core::Trit::from_i8(((i % 3) as i8) - 1).unwrap())
                .collect();
            let act: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
            let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.25).collect();

            let packed = pack_tq2_0(&trits, shape);
            let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);
            let tiled = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

            assert_eq!(tiled.len(), cpu_out.len(), "shape {shape:?}: len");
            for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "shape {shape:?}: tiled vs cpu [{i}] tiled={g} cpu={c}"
                );
            }
        }
    }

    #[test]
    fn rejects_tq1_0_format() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => return, // no device: nothing to assert about format handling
        };
        let shape = GemmShape { m: 1, n: 1, k: 256 };
        // The format gate runs before any length check, so the bytes need not be a
        // valid TQ1_0 length. `Box<dyn DeviceBuffer>` is not `Debug`, so `unwrap_err`
        // is unavailable — match on the result instead (same idiom as tritium-cpu).
        match backend.upload_weights(&[0u8; 66], shape, TernaryFormat::Tq1_0) {
            Err(BackendError::UnsupportedFormat(_)) => {}
            other => panic!(
                "expected UnsupportedFormat, got {:?}",
                other.map(|_| "ok-buffer")
            ),
        }
    }

    // ---- IMMA int8 tensor-core path (v0.30 WF-A part 2) ------------------------
    //
    // Tolerance: the conformance default (`relative = 1e-4`, ADR 0002). The IMMA
    // kernel contracts in **int32**, which is *exact* for int8×ternary (no overflow
    // for any BitNet K — see `kernels/tq2_0_imma.cu`), so the only float rounding is
    // the single per-output `act_scale·weight_scale·acc`. The 1e-4 band is therefore
    // the *reference's* own f32-accumulate rounding, not a defect of this kernel —
    // no widened reduction bar is needed (cf. the tiled add-only kernel, which sums
    // in double to stay inside the band; the IMMA integer accumulate is exact).

    /// Build an I2_S tensor payload (`N·K/4` quant bytes + one trailing `f32` scale)
    /// from an `[N, K]` row-major trit matrix, inverting the 32-byte block striping
    /// (`code = trit + 1`, element `pos` of a 128-block at byte `pos%32`, shift
    /// `6 - 2*(pos/32)`). `n*k` must be a multiple of 128 (the conformance shapes
    /// all are: K ∈ {256, 512}).
    fn build_i2s_payload(trits: &[i8], scale: f32) -> Vec<u8> {
        let n_elements = trits.len();
        assert!(
            n_elements.is_multiple_of(128),
            "i2s payload needs 128-multiple elems"
        );
        let mut quants = vec![0u8; n_elements / 4];
        for (global, &t) in trits.iter().enumerate() {
            let block = global / 128;
            let pos = global % 128;
            let group = pos / 32;
            let gp = pos % 32;
            let code = (t + 1) as u8; // {-1,0,1} -> {0,1,2}
            quants[block * 32 + gp] |= code << (6 - 2 * group);
        }
        let mut payload = quants;
        payload.extend_from_slice(&scale.to_le_bytes());
        payload
    }

    /// Pack an `[N, K]` trit matrix into the IMMA `I2sInt8` layout by routing it
    /// through the *real* converter (`build_i2s_payload` → `convert_i2s_to_int8`),
    /// so the test exercises exactly the bytes the kernel will see in production.
    /// Returns the packed bytes (block scale folded into the per-tensor `scale`,
    /// which the test keeps separate as the per-channel scale, so pass `scale = 1`).
    fn pack_i2s_int8(trits: &[i8], shape: GemmShape) -> Vec<u8> {
        let GemmShape { n, k, .. } = shape;
        let payload = build_i2s_payload(trits, 1.0);
        let w = tritium_format::convert_i2s_to_int8(&payload, GemmShape { m: 0, n, k })
            .expect("convert i2s -> int8");
        w.bytes
    }

    /// IMMA == reference within tolerance over the conformance set. The vectors'
    /// weights are converted to `I2sInt8`, uploaded, and run through the fused
    /// `mpgemm_with_act_quant` (which routes I2sInt8 → on-device quant + IMMA). The
    /// reference is `mpgemm_with_act_quant`'s contract on the *same f32 activations*:
    /// `out[m,n] = act_scale[m]·weight_scale[n]·Σ q[m,k]·w[n,k]`, which the testkit
    /// CPU path computes via the spec default — so this gates IMMA == host-A8 == ref
    /// in one shot.
    #[test]
    fn imma_matches_reference_within_tolerance() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping imma conformance: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);

            // Reference: the host-A8 default path on the CPU backend over the SAME
            // f32 activations + per-channel weight scales.
            let cpu_buf = {
                let trits: Vec<_> = v
                    .weights
                    .iter()
                    .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                    .collect();
                let packed = pack_tq2_0(&trits, shape);
                cpu.upload_weights(&packed, shape, TernaryFormat::Tq2_0)
                    .expect("cpu upload")
            };
            let mut ref_out = vec![0.0f32; shape.m * shape.n];
            cpu.mpgemm_with_act_quant(
                &v.activation,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut ref_out,
            )
            .expect("cpu host-A8 reference");

            // IMMA: upload the I2sInt8 weights, run the fused override (on-device
            // quant + tensor-core contraction).
            let imma_bytes = pack_i2s_int8(&v.weights, shape);
            let imma_buf = cuda
                .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut imma_out = vec![0.0f32; shape.m * shape.n];
            cuda.mpgemm_with_act_quant(
                &v.activation,
                imma_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut imma_out,
            )
            .expect("imma fused mpgemm");

            assert_eq!(imma_out.len(), ref_out.len(), "{}: len", v.id);
            for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "{}: imma vs host-A8 ref [{i}] imma={g} ref={c}",
                    v.id
                );
            }
        }
    }

    /// The CUDA fused override (IMMA) == the spec host-A8 default == the v0.20
    /// caller-side quant, all within tolerance — the "fused == host-A8" gate of ADR
    /// 0005. Three independently-derived results over the same inputs:
    ///   1. `cuda.mpgemm_with_act_quant` on an I2sInt8 buffer → on-device quant + IMMA.
    ///   2. The spec *default* `mpgemm_with_act_quant` (host quant → `mpgemm`) run on
    ///      the CPU backend (a TQ2_0 buffer).
    ///   3. The v0.20 caller-side quant: quantize on the host, then call plain
    ///      `mpgemm` and fold the per-token scale by hand.
    #[test]
    fn imma_fused_equals_host_a8_and_caller_quant() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping fused parity: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let GemmShape { m, n, k } = shape;
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let tq2 = pack_tq2_0(&trits, shape);

            // (1) CUDA fused override on I2sInt8.
            let imma_bytes = pack_i2s_int8(&v.weights, shape);
            let imma_buf = cuda
                .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut fused = vec![0.0f32; m * n];
            cuda.mpgemm_with_act_quant(
                &v.activation,
                imma_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut fused,
            )
            .expect("cuda fused");

            // (2) Spec host-A8 default on the CPU backend (TQ2_0).
            let cpu_buf = cpu
                .upload_weights(&tq2, shape, TernaryFormat::Tq2_0)
                .expect("cpu upload");
            let mut host_a8 = vec![0.0f32; m * n];
            cpu.mpgemm_with_act_quant(
                &v.activation,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut host_a8,
            )
            .expect("cpu host-A8");

            // (3) v0.20 caller-side quant: host quant → plain `mpgemm` → fold.
            let mut q = vec![0.0f32; m * k];
            let mut act_scale = vec![0.0f32; m];
            quantize_act_int8_host(&v.activation, m, k, &mut q, &mut act_scale);
            let mut caller = vec![0.0f32; m * n];
            cpu.mpgemm(
                &q,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut caller,
            )
            .expect("cpu plain mpgemm");
            for (row, &s) in caller.chunks_exact_mut(n).zip(act_scale.iter()) {
                for x in row {
                    *x *= s;
                }
            }

            for i in 0..m * n {
                assert!(
                    tol.accepts(fused[i], host_a8[i]),
                    "{}: fused vs host-A8 [{i}] {} {}",
                    v.id,
                    fused[i],
                    host_a8[i]
                );
                assert!(
                    tol.accepts(fused[i], caller[i]),
                    "{}: fused vs caller-quant [{i}] {} {}",
                    v.id,
                    fused[i],
                    caller[i]
                );
            }
        }
    }

    /// IMMA tail/boundary shapes: M not a multiple of 16, N not a multiple of 8, and
    /// single rows/cols — the padding in the I2sInt8 tiles and the kernel's global
    /// bounds checks must keep every covered output correct. K stays a 256-multiple
    /// (the I2_S converter needs a 128-multiple element count); the M/N tails are the
    /// interesting axes for the 16×8 tile.
    #[test]
    fn imma_handles_tail_shapes() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping imma tail shapes: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // (M, N, K): single row/col, partial 16-row tile, partial 8-col tile.
        let shapes = [
            (1usize, 1usize, 256usize),
            (1, 8, 256),
            (3, 5, 256),
            (16, 8, 512),
            (17, 9, 256),
            (33, 13, 512),
        ];
        for (m, n, k) in shapes {
            let shape = GemmShape::new(m, n, k);
            // Deterministic ternary weights, activations, per-channel scales.
            let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
            let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
            let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

            // Reference: host-A8 default on the CPU backend.
            let trits: Vec<_> = raw
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
                .collect();
            let cpu_buf = cpu
                .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
                .expect("cpu upload");
            let mut ref_out = vec![0.0f32; m * n];
            cpu.mpgemm_with_act_quant(
                &act,
                cpu_buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut ref_out,
            )
            .expect("cpu host-A8");

            let imma_buf = cuda
                .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut imma_out = vec![0.0f32; m * n];
            cuda.mpgemm_with_act_quant(
                &act,
                imma_buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut imma_out,
            )
            .expect("imma fused");

            for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "shape {shape:?}: imma vs ref [{i}] imma={g} ref={c}"
                );
            }
        }
    }

    // ---- WF-B: autotune + nvrtc JIT determinism (ADR 0005) ---------------------
    //
    // These gate the WF-B contract: a JIT-compiled tile is BIT-IDENTICAL to the AOT
    // cubin for the same tile (cold-cache == warm-cache), and any tuned tile matches
    // the reference within the IMMA tolerance. Both are guaranteed by construction —
    // every tile does the same exact int32 mma accumulate + one f32 scale fold — but
    // these tests prove it on-device across tile shapes.

    /// Deterministic int8 activations / ternary weights / scales for a WF-B probe.
    fn jit_probe_inputs(m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>, Vec<i8>) {
        let qact: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
        let act_scale: Vec<f32> = (0..m).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
        let wscale: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();
        let trits: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
        (qact, act_scale, wscale, trits)
    }

    /// Run one IMMA contraction with an explicit `func`/`tile` (host-quantised int8
    /// inputs already supplied), returning the `[M, N]` f32 output. Drives
    /// `launch_imma_tile` directly so a test can force a specific tile + kernel image
    /// (AOT cubin vs a freshly JIT-compiled module).
    fn run_imma_tile(
        cuda: &CudaBackend,
        func: &CudaFunction,
        tile: TileConfig,
        qact: &[i8],
        packed_weights: &[u8],
        act_scale: &[f32],
        wscale: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let num_ktiles = k.div_ceil(IMMA_K);
        let d_qact = cuda.stream.clone_htod(qact).expect("htod qact");
        let d_weights = cuda.stream.clone_htod(packed_weights).expect("htod weights");
        let d_act_scale = cuda.stream.clone_htod(act_scale).expect("htod act_scale");
        let d_wscale = cuda.stream.clone_htod(wscale).expect("htod wscale");
        let mut d_out = cuda.stream.alloc_zeros::<f32>(m * n).expect("alloc out");
        cuda.launch_imma_tile(
            func,
            tile,
            &d_qact,
            &d_weights,
            &d_act_scale,
            &d_wscale,
            &mut d_out,
            m as i32,
            n as i32,
            k as i32,
            num_ktiles as i32,
        )
        .expect("launch imma tile");
        let mut out = vec![0.0f32; m * n];
        cuda.stream.memcpy_dtoh(&d_out, &mut out).expect("dtoh out");
        cuda.stream.synchronize().expect("sync");
        out
    }

    /// COLD-CACHE (JIT) == WARM-CACHE (AOT) BIT-IDENTICAL for a fixed tile.
    ///
    /// The AOT-equivalent tile has two realisations: the embedded AOT cubin
    /// (`func_imma`, the warm/default path) and a fresh nvrtc JIT compile of the
    /// rendered source (the cold path). For a range of shapes their outputs must be
    /// **bit-for-bit equal** (`==` on the raw `f32`, not a tolerance) — the load-bearing
    /// WF-B determinism gate. If they ever diverge, JIT and AOT are not interchangeable
    /// and the autotune cache could change numerics, which ADR 0005 forbids.
    #[test]
    fn jit_aot_equivalent_is_bit_identical() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping JIT==AOT bit-identity: no device ({e})");
                return;
            }
        };

        // Freshly JIT-compile the AOT-equivalent tile (the cold path). The AOT side
        // is the embedded cubin resolved by `imma_function_for_tile`.
        let tile = TileConfig::AOT_EQUIVALENT;
        let (_jit_mod, jit_func) = cuda
            .imma_jit_function(tile)
            .expect("JIT-compile AOT-equivalent tile");
        let aot_func = cuda
            .imma_function_for_tile(tile)
            .expect("resolve AOT cubin");

        // Tail + clean shapes; K a 32-multiple (one whole k-tile minimum).
        let shapes = [
            (1usize, 1usize, 32usize),
            (3, 5, 64),
            (16, 8, 256),
            (17, 9, 96),
            (33, 13, 512),
            (64, 40, 2560 % 8192), // a realistic-ish K, still a 32-multiple below
        ];
        for (m, n, k) in shapes {
            let k = k.max(IMMA_K); // never zero k-tiles
            let k = k.div_ceil(IMMA_K) * IMMA_K; // snap to a whole k-tile
            let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
            let packed = pack_i2s_int8_tiles(&trits, n, k);

            let aot = run_imma_tile(
                &cuda, &aot_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
            );
            let jit = run_imma_tile(
                &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
            );

            assert_eq!(aot.len(), jit.len(), "shape ({m},{n},{k}): len");
            for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
                // Bit-identical: compare the raw IEEE-754 bit patterns so even a
                // signed-zero or NaN-payload difference would fail (none expected).
                assert_eq!(
                    a.to_bits(),
                    j.to_bits(),
                    "shape ({m},{n},{k}): JIT vs AOT diverge at [{i}] aot={a} jit={j}"
                );
            }
        }
    }

    /// A NON-TRIVIAL JIT tile (wider M/N, deeper K, multi-warp) is ALSO bit-identical
    /// to the AOT cubin. This proves the determinism guarantee holds across the tile
    /// shapes the autotune search actually considers, not just the AOT-equivalent
    /// anchor — the int32 accumulate is order-independent, so a 32×16/4-warp tile that
    /// splits the work differently still lands on the same bits.
    #[test]
    fn jit_wide_tile_matches_aot_bit_identical() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping wide-tile JIT==AOT: no device ({e})");
                return;
            }
        };
        let aot_func = cuda
            .imma_function_for_tile(TileConfig::AOT_EQUIVALENT)
            .expect("AOT cubin");

        // A representative spread of the search's candidate tiles.
        let tiles = [
            TileConfig { tile_m: 16, tile_n: 8, tile_k: 128, warps: 1, stages: 2 },
            TileConfig { tile_m: 16, tile_n: 16, tile_k: 64, warps: 2, stages: 2 },
            TileConfig { tile_m: 32, tile_n: 16, tile_k: 64, warps: 4, stages: 2 },
            TileConfig { tile_m: 64, tile_n: 16, tile_k: 32, warps: 8, stages: 3 },
        ];
        let (m, n, k) = (40usize, 24usize, 256usize);
        let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
        let packed = pack_i2s_int8_tiles(&trits, n, k);

        let aot = run_imma_tile(
            &cuda,
            &aot_func,
            TileConfig::AOT_EQUIVALENT,
            &qact,
            &packed,
            &act_scale,
            &wscale,
            m,
            n,
            k,
        );

        for tile in tiles {
            assert!(tile.is_valid(), "test tile {tile:?} invalid");
            let (_m, jit_func) = cuda
                .imma_jit_function(tile)
                .unwrap_or_else(|e| panic!("JIT-compile {tile:?}: {e:?}"));
            let jit = run_imma_tile(
                &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
            );
            for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    j.to_bits(),
                    "tile {tile:?}: JIT vs AOT diverge at [{i}] aot={a} jit={j}"
                );
            }
        }
    }

    /// The TUNED config (resolved through the on-disk autotune cache + tile search)
    /// matches the reference within the IMMA tolerance. Drives the full public fused
    /// path (`mpgemm_with_act_quant`), which now consults the cache via
    /// `resolve_imma_tile`, on a prefill-shaped problem — so this exercises the tuner
    /// end-to-end (cold cache → search → winner) and gates the winner vs the CPU
    /// host-A8 reference. A second call (warm cache) must agree bit-for-bit with the
    /// first, since a cached tile is numerically identical to the freshly-tuned one.
    #[test]
    fn tuned_config_matches_reference_and_is_stable() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tuned-config gate: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // A prefill-shaped problem so the search has something to chew on. K is a
        // 256-multiple (the I2_S converter the reference path uses needs a
        // 128-multiple); N/M exercise partial tiles.
        let (m, n, k) = (40usize, 24usize, 256usize);
        let shape = GemmShape::new(m, n, k);
        let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

        // Reference: host-A8 default on the CPU backend (TQ2_0).
        let trits: Vec<_> = raw
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
            .collect();
        let cpu_buf = cpu
            .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
            .expect("cpu upload");
        let mut ref_out = vec![0.0f32; m * n];
        cpu.mpgemm_with_act_quant(
            &act,
            cpu_buf.as_ref(),
            &scales,
            shape,
            TernaryFormat::Tq2_0,
            &mut ref_out,
        )
        .expect("cpu host-A8 reference");

        // Tuned path: upload I2sInt8, run the fused override (which resolves + tunes
        // the tile). Run it twice; the second call hits the in-memory + on-disk cache.
        let imma_buf = cuda
            .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut tuned1 = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(
            &act,
            imma_buf.as_ref(),
            &scales,
            shape,
            TernaryFormat::I2sInt8,
            &mut tuned1,
        )
        .expect("tuned fused (cold)");
        let mut tuned2 = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(
            &act,
            imma_buf.as_ref(),
            &scales,
            shape,
            TernaryFormat::I2sInt8,
            &mut tuned2,
        )
        .expect("tuned fused (warm)");

        // Tuned == reference within tolerance.
        for (i, (&g, &c)) in tuned1.iter().zip(&ref_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "tuned vs ref [{i}] tuned={g} ref={c}"
            );
        }
        // Cold vs warm cache: bit-for-bit identical (same tile → same numerics).
        for (i, (&a, &b)) in tuned1.iter().zip(&tuned2).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cold vs warm tuned output diverges at [{i}] cold={a} warm={b}"
            );
        }
    }

    /// v0.3.1 de-risk: the device `rmsnorm_f32` decode kernel must reproduce the host
    /// `tritium_nn::ops::rmsnorm` **bit-for-bit** (`to_bits` equal), so the fully
    /// device-resident forward keeps greedy 256/256. This is the proof that a
    /// sequential-f32 + FMA-disabled device kernel can match host f32 exactly; the
    /// rest of the decode kernels follow the same discipline.
    #[test]
    fn rmsnorm_bit_matches_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping rmsnorm bit-match: no device ({e})");
                return;
            }
        };
        // Host reference — identical to `tritium_nn::ops::rmsnorm` (this crate does
        // not depend on tritium-nn, so the 4-line formula is replicated verbatim).
        fn host_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
            let n = x.len();
            let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect()
        }
        // BitNet hidden/ffn sizes + a few edge lengths; deterministic xorshift inputs.
        for &n in &[2560usize, 6912, 1, 17, 256, 2559] {
            let mut s = 0x1234_5678_9abc_def0u64 ^ (n as u64).wrapping_mul(0x9E37_79B9);
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
            };
            let x: Vec<f32> = (0..n).map(|_| next()).collect();
            let w: Vec<f32> = (0..n).map(|_| next()).collect();
            let eps = 1e-5f32;

            let want = host_rmsnorm(&x, &w, eps);
            let mut got = vec![0.0f32; n];
            backend.rmsnorm(&x, &w, eps, &mut got).expect("device rmsnorm");

            for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    h.to_bits(),
                    "rmsnorm bit mismatch n={n} i={i}: got {g} ({:#010x}) want {h} ({:#010x})",
                    g.to_bits(),
                    h.to_bits()
                );
            }
        }
    }

    /// The device `rope_apply_f32` kernel must reproduce `tritium_nn::ops::rope_apply`
    /// **bit-for-bit** for one token (M=1 decode). The trig is computed exactly as the
    /// host op (f64 `sin_cos` → f32, data-independent) and the f32 rotation has no FMA.
    #[test]
    fn rope_bit_matches_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping rope bit-match: no device ({e})");
                return;
            }
        };
        // BitNet 2B4T uses head_dim=128, n_head 20(Q)/5(KV), theta=500000.
        for &(n_head, head_dim) in &[(20usize, 128usize), (5, 128), (1, 8), (3, 64)] {
            let half = head_dim / 2;
            let theta = 500_000.0f32;
            for &pos in &[0usize, 1, 7, 255, 4095] {
                // Trig tables, identical to the host op (f64 sin_cos cast to f32).
                let theta_f64 = f64::from(theta);
                let inv_hd = 1.0 / head_dim as f64;
                let mut cos_t = vec![0.0f32; half];
                let mut sin_t = vec![0.0f32; half];
                for j in 0..half {
                    let inv_freq = theta_f64.powf(-2.0 * j as f64 * inv_hd);
                    let (s, c) = (pos as f64 * inv_freq).sin_cos();
                    cos_t[j] = c as f32;
                    sin_t[j] = s as f32;
                }
                // Deterministic input.
                let mut st =
                    0xDEAD_BEEF_CAFE_F00Du64 ^ ((pos as u64) * 131 + n_head as u64 * 17 + head_dim as u64);
                let mut next = || {
                    st ^= st << 13;
                    st ^= st >> 7;
                    st ^= st << 17;
                    ((st >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
                };
                let x0: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();

                // Host rope (replicated; Rust does not auto-contract a*c - b*s to FMA).
                let mut want = x0.clone();
                for head in 0..n_head {
                    let base = head * head_dim;
                    for j in 0..half {
                        let a = x0[base + j];
                        let b = x0[base + j + half];
                        want[base + j] = a * cos_t[j] - b * sin_t[j];
                        want[base + j + half] = b * cos_t[j] + a * sin_t[j];
                    }
                }

                let mut got = x0.clone();
                backend
                    .rope(&mut got, &cos_t, &sin_t, n_head, head_dim)
                    .expect("device rope");

                for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        h.to_bits(),
                        "rope bit mismatch (n_head={n_head} head_dim={head_dim} pos={pos}) i={i}: got {g} want {h}"
                    );
                }
            }
        }
    }

    /// Measure device softmax vs host `softmax_rows`. The reductions are bit-matched;
    /// the open question is `expf` (device CUDA libm vs host glibc). Reports the max
    /// ULP difference + whether bit-exact, and asserts a tight relative tolerance so
    /// the result is informative without spuriously failing on a ~1-ULP exp delta.
    /// This is the gate-deciding measurement: bit-exact ⇒ strict greedy 256/256 is
    /// reachable; otherwise the forward uses the perplexity+lockstep fallback.
    #[test]
    fn softmax_vs_host_exp_divergence() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping softmax divergence: no device ({e})");
                return;
            }
        };
        fn host_softmax(x: &mut [f32], row_len: usize) {
            for row in x.chunks_mut(row_len) {
                let mut m = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    let e = (*v - m).exp();
                    *v = e;
                    sum += e;
                }
                let inv = 1.0f32 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        let (rows, row_len) = (20usize, 1024usize); // decode-ish: n_head × ctx
        let mut s = 0x5151_5151_2727_2727u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 16.0 - 8.0
        };
        let x0: Vec<f32> = (0..rows * row_len).map(|_| next()).collect();
        let mut want = x0.clone();
        host_softmax(&mut want, row_len);
        let mut got = x0.clone();
        backend.softmax(&mut got, row_len, rows).expect("device softmax");

        let (mut max_ulp, mut n_diff, mut max_rel) = (0i64, 0usize, 0.0f64);
        for (&g, &h) in got.iter().zip(&want) {
            let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
            if du != 0 {
                n_diff += 1;
            }
            max_ulp = max_ulp.max(du);
            if h != 0.0 {
                max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
            }
        }
        eprintln!(
            "softmax device-vs-host: max_ulp={max_ulp} n_diff={n_diff}/{} max_rel={max_rel:.3e} bit_exact={}",
            got.len(),
            n_diff == 0
        );
        assert!(
            max_rel < 1e-5,
            "device softmax exp diverges too far from host: max_rel={max_rel:.3e}"
        );
    }
}
