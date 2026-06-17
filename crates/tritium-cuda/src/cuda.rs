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
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr,
    DriverError, LaunchConfig, PushKernelArg, result, sys,
};
use cudarc::nvrtc::Ptx;
use std::ffi::{CString, c_void};

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
/// Residual add `x += y` (exact f32 add).
const KERNEL_NAME_RESIDUAL: &str = "residual_add_f32";
/// Embedding-table row gather (exact copy).
const KERNEL_NAME_EMBED: &str = "embedding_gather_f32";
/// Tied LM head `logits[v] = <h, embd[v]>` (sequential dot, bit-matches host).
const KERNEL_NAME_LM_HEAD: &str = "lm_head_f32";
/// GQA attention, M=1 decode (v0.3.1) — matches `ops::gqa_attention`; dots/weighted
/// sums bit-match, the inline softmax `expf` differs ≤3 ULP.
const KERNEL_NAME_ATTN: &str = "gqa_attention_decode_f32";
/// On-device int8 activation quant for the tiled (TQ2_0) decode GEMM (v0.3.1) —
/// bit-matches `ops::quantize_activation_int8` (int8 kept as f32 + per-token scale).
const KERNEL_NAME_ACT_QUANT_TILED: &str = "act_quant_tiled_f32";
/// Per-token activation-scale fold `out *= act_scale` (v0.3.1) — the device half of
/// the W1.58A8 dequant the host applies after the GEMM.
const KERNEL_NAME_SCALE_MUL: &str = "scale_mul_f32";
/// BitNet squared-ReLU FFN gate `g = relu(g)² ⊙ u` (v0.3.1) — bit-matches the host
/// `mlp` gating loop (`r = g.max(0); g = r*r*u`).
const KERNEL_NAME_RELU2_GATE: &str = "relu2_gate_f32";
/// v0.3.2 graph variants reading the per-token control block (token/pos/cache_len).
const KERNEL_NAME_EMBED_G: &str = "embedding_gather_f32_g";
const KERNEL_NAME_ROPE_G: &str = "rope_apply_f32_g";
const KERNEL_NAME_KV_APPEND: &str = "kv_append_f32";
const KERNEL_NAME_ATTN_G: &str = "gqa_attention_decode_f32_g";
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
    /// `Arc` so the device-resident decode model ([`CudaDecodeModel`]) can share the
    /// already-uploaded weight with the [`TernaryLinear`] that owns it — the resident
    /// forward references the same VRAM as the prefill path, with no re-upload.
    device: Arc<CudaSlice<u8>>,
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

impl CudaBuffer {
    /// Share the uploaded device allocation (cheap `Arc` clone). Used by
    /// [`CudaDecodeModel`] to reference the prefill path's weights without re-upload.
    pub(crate) fn device_arc(&self) -> Arc<CudaSlice<u8>> {
        Arc::clone(&self.device)
    }

    /// `(N, K)` channel dims of the packed weight.
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// The TQ2_0 per-row byte stride, or `None` if this buffer is not TQ2_0
    /// (the resident decode path is TQ2_0-only; IMMA stays the prefill format).
    pub(crate) fn tq2_0_row_bytes(&self) -> Option<usize> {
        match self.stride {
            Stride::Tq2_0 { row_bytes } => Some(row_bytes),
            Stride::I2sInt8 { .. } => None,
        }
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
    /// Resolved `residual_add_f32` / `embedding_gather_f32` / `lm_head_f32` kernels.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    func_residual: CudaFunction,
    #[allow(dead_code)]
    func_embed: CudaFunction,
    #[allow(dead_code)]
    func_lm_head: CudaFunction,
    /// The resolved `gqa_attention_decode_f32` kernel (M=1 decode attention).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    func_attn: CudaFunction,
    /// The resolved `act_quant_tiled_f32` kernel (on-device A8 quant for the tiled GEMM).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    func_act_quant_tiled: CudaFunction,
    /// The resolved `scale_mul_f32` kernel (per-token act-scale fold).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    func_scale_mul: CudaFunction,
    /// The resolved `relu2_gate_f32` kernel (BitNet squared-ReLU FFN gate).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    func_relu2_gate: CudaFunction,
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
        let func_residual = decode_module
            .load_function(KERNEL_NAME_RESIDUAL)
            .map_err(|e| driver_err("resolve residual kernel", &e))?;
        let func_embed = decode_module
            .load_function(KERNEL_NAME_EMBED)
            .map_err(|e| driver_err("resolve embed kernel", &e))?;
        let func_lm_head = decode_module
            .load_function(KERNEL_NAME_LM_HEAD)
            .map_err(|e| driver_err("resolve lm_head kernel", &e))?;
        let func_attn = decode_module
            .load_function(KERNEL_NAME_ATTN)
            .map_err(|e| driver_err("resolve attention kernel", &e))?;
        let func_act_quant_tiled = decode_module
            .load_function(KERNEL_NAME_ACT_QUANT_TILED)
            .map_err(|e| driver_err("resolve act_quant_tiled kernel", &e))?;
        let func_scale_mul = decode_module
            .load_function(KERNEL_NAME_SCALE_MUL)
            .map_err(|e| driver_err("resolve scale_mul kernel", &e))?;
        let func_relu2_gate = decode_module
            .load_function(KERNEL_NAME_RELU2_GATE)
            .map_err(|e| driver_err("resolve relu2_gate kernel", &e))?;

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
            func_residual,
            func_embed,
            func_lm_head,
            func_attn,
            func_act_quant_tiled,
            func_scale_mul,
            func_relu2_gate,
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

    /// Device residual add `x += y` (exact) on host slices. Building block.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn residual_add(&self, x: &mut [f32], y: &[f32]) -> Result<(), BackendError> {
        let n = x.len();
        if y.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: y.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let mut d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| driver_err("residual htod x", &e))?;
        let d_y = self
            .stream
            .clone_htod(y)
            .map_err(|e| driver_err("residual htod y", &e))?;
        let n_i = n as i32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_residual);
        launch.arg(&mut d_x).arg(&d_y).arg(&n_i);
        // SAFETY: signature `(float* x, const float* y, int n)`; pushed in that
        // order; `d_x`/`d_y` are length `n`, one thread per element guarded by `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch residual", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_x, x)
            .map_err(|e| driver_err("residual dtoh", &e))?;
        Ok(())
    }

    /// Device BitNet squared-ReLU FFN gate `gate = relu(gate)² ⊙ up` (in place into
    /// `gate`), bit-matching the host `mlp` gating loop. Building block.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn relu2_gate(&self, gate: &mut [f32], up: &[f32]) -> Result<(), BackendError> {
        let n = gate.len();
        if up.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: up.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let mut d_gate = self
            .stream
            .clone_htod(gate)
            .map_err(|e| driver_err("relu2_gate htod gate", &e))?;
        let d_up = self
            .stream
            .clone_htod(up)
            .map_err(|e| driver_err("relu2_gate htod up", &e))?;
        let n_i = n as i32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_relu2_gate);
        launch.arg(&mut d_gate).arg(&d_up).arg(&n_i);
        // SAFETY: signature `(float* gate, const float* up, int n)`; pushed in that
        // order; both buffers are length `n`, one thread per element guarded by `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch relu2_gate", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_gate, gate)
            .map_err(|e| driver_err("relu2_gate dtoh", &e))?;
        Ok(())
    }

    /// Device embedding-row gather `out = table[tok]` (exact copy) on host slices.
    /// (Test/building block — uploads `table`; the forward uses the resident table.)
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn embedding_gather(
        &self,
        table: &[f32],
        tok: usize,
        n_embd: usize,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if out.len() != n_embd || table.len() < tok.saturating_add(1) * n_embd {
            return Err(BackendError::ShapeMismatch {
                expected: n_embd,
                got: out.len(),
            });
        }
        if n_embd == 0 {
            return Ok(());
        }
        let d_table = self
            .stream
            .clone_htod(table)
            .map_err(|e| driver_err("embed htod table", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(n_embd)
            .map_err(|e| driver_err("embed alloc out", &e))?;
        let tok_i = tok as i32;
        let n_embd_i = n_embd as i32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_embd as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_embed);
        launch
            .arg(&d_table)
            .arg(&tok_i)
            .arg(&n_embd_i)
            .arg(&mut d_out);
        // SAFETY: signature `(const float* table, int tok, int n_embd, float* out)`;
        // pushed in order; `table` covers row `tok`, `out` is length `n_embd`, one
        // thread per element guarded by `i < n_embd`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch embed", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("embed dtoh", &e))?;
        Ok(())
    }

    /// Device tied LM head `logits[v] = <h, embd[v]>` on host slices, **bit-matching**
    /// the host's sequential dot. (Test/building block — uploads `embd`; the forward
    /// uses the resident embedding table.)
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn lm_head(
        &self,
        h: &[f32],
        embd: &[f32],
        n_embd: usize,
        vocab: usize,
        logits: &mut [f32],
    ) -> Result<(), BackendError> {
        if h.len() != n_embd || embd.len() != vocab * n_embd || logits.len() != vocab {
            return Err(BackendError::ShapeMismatch {
                expected: vocab,
                got: logits.len(),
            });
        }
        if vocab == 0 {
            return Ok(());
        }
        let d_h = self
            .stream
            .clone_htod(h)
            .map_err(|e| driver_err("lm_head htod h", &e))?;
        let d_embd = self
            .stream
            .clone_htod(embd)
            .map_err(|e| driver_err("lm_head htod embd", &e))?;
        let mut d_logits = self
            .stream
            .alloc_zeros::<f32>(vocab)
            .map_err(|e| driver_err("lm_head alloc logits", &e))?;
        let n_embd_i = n_embd as i32;
        let vocab_i = vocab as i32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((vocab as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_lm_head);
        launch
            .arg(&d_h)
            .arg(&d_embd)
            .arg(&n_embd_i)
            .arg(&vocab_i)
            .arg(&mut d_logits);
        // SAFETY: signature `(const float* h, const float* embd, int n_embd, int vocab,
        // float* logits)`; pushed in order; `h` length `n_embd`, `embd` length
        // `vocab*n_embd`, `logits` length `vocab`; one thread per vocab row guarded by
        // `v >= vocab`, each reading only its own `embd` row.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch lm_head", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_logits, logits)
            .map_err(|e| driver_err("lm_head dtoh", &e))?;
        Ok(())
    }

    /// Device GQA attention for the M=1 decode token on host slices. Matches
    /// `tritium_nn::ops::gqa_attention` (seq=1) except the inline softmax `expf`
    /// (≤3 ULP). `q` is `[n_head, head_dim]`; `k`/`v` are `[ctx, n_head_kv, head_dim]`;
    /// `limit` is the last visible key index. A building block for `forward_device`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(crate) fn gqa_attention_decode(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        ctx: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        limit: usize,
    ) -> Result<(), BackendError> {
        let q_len = n_head * head_dim;
        let kv_len = ctx * n_head_kv * head_dim;
        if n_head_kv == 0
            || !n_head.is_multiple_of(n_head_kv)
            || q.len() != q_len
            || k.len() != kv_len
            || v.len() != kv_len
            || out.len() != q_len
        {
            return Err(BackendError::ShapeMismatch {
                expected: q_len,
                got: q.len(),
            });
        }
        if n_head == 0 || ctx == 0 {
            return Ok(());
        }
        let d_q = self
            .stream
            .clone_htod(q)
            .map_err(|e| driver_err("attn htod q", &e))?;
        let d_k = self
            .stream
            .clone_htod(k)
            .map_err(|e| driver_err("attn htod k", &e))?;
        let d_v = self
            .stream
            .clone_htod(v)
            .map_err(|e| driver_err("attn htod v", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(q_len)
            .map_err(|e| driver_err("attn alloc out", &e))?;
        let mut d_scores = self
            .stream
            .alloc_zeros::<f32>(n_head * ctx)
            .map_err(|e| driver_err("attn alloc scores", &e))?;

        let ctx_i = ctx as i32;
        let n_head_i = n_head as i32;
        let n_head_kv_i = n_head_kv as i32;
        let head_dim_i = head_dim as i32;
        let limit_i = limit as i32;
        let threads = 64u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_head as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_attn);
        launch
            .arg(&d_q)
            .arg(&d_k)
            .arg(&d_v)
            .arg(&mut d_out)
            .arg(&mut d_scores)
            .arg(&ctx_i)
            .arg(&n_head_i)
            .arg(&n_head_kv_i)
            .arg(&head_dim_i)
            .arg(&scale)
            .arg(&limit_i);

        // SAFETY: kernel signature `(const float* q, const float* k, const float* v,
        // float* out, float* scores, int ctx, int n_head, int n_head_kv, int head_dim,
        // float scale, int limit)`; the 11 args are pushed in that exact order/type.
        // `q`/`out` are length `n_head*head_dim`, `k`/`v` length `ctx*n_head_kv*head_dim`,
        // `scores` length `n_head*ctx`; one thread per head guarded by `h >= n_head`,
        // each touching only its own `q`/`out`/`scores` row and `kv`-selected k/v rows.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch attention", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("attn dtoh", &e))?;
        Ok(())
    }

    /// On-device int8 activation quant for the tiled GEMM (one row), on host slices,
    /// **bit-matching** `tritium_nn::ops::quantize_activation_int8`. Writes the
    /// int8-as-f32 values to `q_out` and returns the per-token dequant scale. A
    /// building block for the device-resident decode GEMM.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if `q_out.len() != act.len()`; device errors.
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    pub(crate) fn act_quant_tiled(
        &self,
        act: &[f32],
        q_out: &mut [f32],
    ) -> Result<f32, BackendError> {
        let k = act.len();
        if q_out.len() != k {
            return Err(BackendError::ShapeMismatch {
                expected: k,
                got: q_out.len(),
            });
        }
        if k == 0 {
            return Ok(0.0);
        }
        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| driver_err("act_quant htod", &e))?;
        let mut d_q = self
            .stream
            .alloc_zeros::<f32>(k)
            .map_err(|e| driver_err("act_quant alloc q", &e))?;
        let mut d_scale = self
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| driver_err("act_quant alloc scale", &e))?;
        let k_i = k as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_act_quant_tiled);
        launch.arg(&d_act).arg(&k_i).arg(&mut d_q).arg(&mut d_scale);
        // SAFETY: kernel signature `(const float* act, int k, float* q_out, float*
        // act_scale)`; args pushed in that order/type. `act`/`q_out` are length `k`,
        // `act_scale` length 1; a single block, loops bounded by `k`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch act_quant_tiled", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_q, q_out)
            .map_err(|e| driver_err("act_quant dtoh q", &e))?;
        let mut scale = [0.0f32; 1];
        self.stream
            .memcpy_dtoh(&d_scale, &mut scale)
            .map_err(|e| driver_err("act_quant dtoh scale", &e))?;
        Ok(scale[0])
    }

    /// Device-resident TQ2_0 mpGEMM for the M=1 decode token (host-slice wrapper):
    /// on-device A8 quant → tiled add-only **f64** GEMM (the reference-matching decode
    /// kernel) → per-token scale fold, chained on device buffers. **Bit-matches** the
    /// host `quantize_activation_int8` + tiled `mpgemm` + fold. `forward_device` keeps
    /// `normed`/`out` resident; this wrapper htod's/dtoh's only for testing.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] if `m != 1`, `k` exceeds the tiled cap, or the
    /// buffer is not a TQ2_0 `CudaBuffer`; [`BackendError::ShapeMismatch`] otherwise.
    #[allow(dead_code)] // the resident chain feeds `forward_device` next; W1-in-progress.
    pub(crate) fn mpgemm_device(
        &self,
        normed: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if m != 1 {
            return Err(BackendError::InvalidInput(
                "mpgemm_device is M=1 decode only".into(),
            ));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        if buf.n != n || buf.k != k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: n * k,
            });
        }
        if normed.len() != k || scales.len() != n || out.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: k,
                got: normed.len(),
            });
        }
        if k > TILED_K_MAX {
            return Err(BackendError::InvalidInput(format!(
                "mpgemm_device k={k} exceeds the tiled cap {TILED_K_MAX}"
            )));
        }
        let row_bytes = match buf.stride {
            Stride::Tq2_0 { row_bytes } => row_bytes,
            Stride::I2sInt8 { .. } => {
                return Err(BackendError::UnsupportedFormat(TernaryFormat::I2sInt8));
            }
        };

        let d_normed = self
            .stream
            .clone_htod(normed)
            .map_err(|e| driver_err("gemm_dev htod normed", &e))?;
        let d_scales = self
            .stream
            .clone_htod(scales)
            .map_err(|e| driver_err("gemm_dev htod scales", &e))?;
        let mut d_q = self
            .stream
            .alloc_zeros::<f32>(k)
            .map_err(|e| driver_err("gemm_dev alloc q", &e))?;
        let mut d_act_scale = self
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| driver_err("gemm_dev alloc scale", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| driver_err("gemm_dev alloc out", &e))?;

        let k_i = k as i32;
        let n_i = n as i32;
        let m_i = 1i32;
        let rb_i = row_bytes as i32;

        // 1. On-device A8 quant: normed -> int8-as-f32 q + per-token act_scale.
        {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = self.stream.launch_builder(&self.func_act_quant_tiled);
            l.arg(&d_normed).arg(&k_i).arg(&mut d_q).arg(&mut d_act_scale);
            // SAFETY: `act_quant_tiled_f32(const float* act, int k, float* q, float*
            // scale)`; args in order; `d_normed`/`d_q` length `k`, `d_act_scale` 1.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch gemm_dev quant", &e))?;
            }
        }
        // 2. Tiled add-only f64 GEMM (M=1): folds the per-channel weight scale.
        {
            let grid_n = (n as u32).div_ceil(WARPS_PER_BLOCK);
            let cfg = LaunchConfig {
                grid_dim: (grid_n, 1, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: (k * 4) as u32,
            };
            let mut l = self.stream.launch_builder(&self.func_tiled);
            l.arg(&d_q)
                .arg(buf.device.as_ref())
                .arg(&d_scales)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled(act, weights, scales, out, m, n, k,
            // row_bytes)`; args in that order. `d_q` length `k`, `buf.device` the
            // uploaded TQ2_0 weight, `d_scales` length `n`, `d_out` length `n` (M=1);
            // grid covers `ceil(n/8)` warp-columns × 1 row; shared `k*4` ≤ 32 KiB.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch gemm_dev tiled", &e))?;
            }
        }
        // 3. Per-token activation-scale fold: out *= act_scale.
        {
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = self.stream.launch_builder(&self.func_scale_mul);
            l.arg(&mut d_out).arg(&d_act_scale).arg(&n_i);
            // SAFETY: `scale_mul_f32(float* out, const float* s, int n)`; args in
            // order; `d_out` length `n`, `d_act_scale` the length-1 scalar.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch gemm_dev fold", &e))?;
            }
        }
        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("gemm_dev dtoh", &e))?;
        Ok(())
    }

    /// Build a fully device-resident decoder ([`CudaDecodeModel`]) from `spec`:
    /// upload the dense fp32 weights + precompute the RoPE table once, share the
    /// already-uploaded ternary weights (`Arc` clone, no re-upload), and allocate the
    /// per-layer KV arenas + the reused scratch. The returned model owns all of this,
    /// so it outlives the borrowed `spec` (the runner can drop its borrow afterwards).
    ///
    /// # Errors
    /// [`BackendError`] if a ternary weight is not a TQ2_0 [`CudaBuffer`], a shape is
    /// inconsistent, or a device allocation/upload fails.
    pub fn build_decode_model(
        &self,
        spec: &DecodeModelSpec,
    ) -> Result<CudaDecodeModel, BackendError> {
        let s = &self.stream;
        let DecodeModelSpec {
            n_embd, n_head, n_head_kv, head_dim, n_ff, vocab, max_ctx, rope_theta, rms_eps, ..
        } = *spec;
        let q_width = n_head * head_dim;
        let kv_width = n_head_kv * head_dim;
        let half = head_dim / 2;

        // Validate the dense shapes the kernels assume.
        if spec.token_embd.len() != vocab * n_embd {
            return Err(BackendError::ShapeMismatch { expected: vocab * n_embd, got: spec.token_embd.len() });
        }
        if spec.output_norm.len() != n_embd {
            return Err(BackendError::ShapeMismatch { expected: n_embd, got: spec.output_norm.len() });
        }

        // RoPE table: cos/sin for every (pos, lane), computed exactly as the host
        // `ops::rope_apply` (f64 inv_freq + sin_cos, cast to f32), so the device
        // rotation bit-matches. `inv_freq[j] = theta^(-2j/head_dim)`.
        let theta = f64::from(rope_theta);
        let inv_head_dim = 1.0f64 / head_dim as f64;
        let inv_freq: Vec<f64> = (0..half).map(|j| theta.powf(-2.0 * j as f64 * inv_head_dim)).collect();
        let mut cos_t = vec![0.0f32; max_ctx * half];
        let mut sin_t = vec![0.0f32; max_ctx * half];
        for pos in 0..max_ctx {
            let p = pos as f64;
            for j in 0..half {
                let (sin, cos) = (p * inv_freq[j]).sin_cos();
                cos_t[pos * half + j] = cos as f32;
                sin_t[pos * half + j] = sin as f32;
            }
        }

        let upload = |v: &[f32], what: &str| -> Result<CudaSlice<f32>, BackendError> {
            s.clone_htod(v).map_err(|e| driver_err(what, &e))
        };
        let alloc = |n: usize, what: &str| -> Result<CudaSlice<f32>, BackendError> {
            s.alloc_zeros::<f32>(n).map_err(|e| driver_err(what, &e))
        };

        let d_token_embd = upload(spec.token_embd, "decode token_embd htod")?;
        let d_output_norm = upload(spec.output_norm, "decode output_norm htod")?;
        let d_cos = upload(&cos_t, "decode rope cos htod")?;
        let d_sin = upload(&sin_t, "decode rope sin htod")?;

        // Per-layer weights + KV arenas.
        let mut layers = Vec::with_capacity(spec.layers.len());
        let mut kv_k = Vec::with_capacity(spec.layers.len());
        let mut kv_v = Vec::with_capacity(spec.layers.len());
        for ls in &spec.layers {
            let opt_norm = |w: &[f32], width: usize, what: &str| -> Result<Option<CudaSlice<f32>>, BackendError> {
                if w.is_empty() {
                    Ok(None)
                } else if w.len() == width {
                    Ok(Some(upload(w, what)?))
                } else {
                    Err(BackendError::ShapeMismatch { expected: width, got: w.len() })
                }
            };
            if ls.attn_norm.len() != n_embd || ls.ffn_norm.len() != n_embd {
                return Err(BackendError::ShapeMismatch {
                    expected: n_embd,
                    got: ls.attn_norm.len().min(ls.ffn_norm.len()),
                });
            }
            layers.push(ResidentLayer {
                attn_norm: upload(ls.attn_norm, "decode attn_norm htod")?,
                attn_sub_norm: opt_norm(ls.attn_sub_norm, q_width, "decode attn_sub_norm htod")?,
                ffn_norm: upload(ls.ffn_norm, "decode ffn_norm htod")?,
                ffn_sub_norm: opt_norm(ls.ffn_sub_norm, n_ff, "decode ffn_sub_norm htod")?,
                q: ResidentLinear::build(s, &ls.q)?,
                k: ResidentLinear::build(s, &ls.k)?,
                v: ResidentLinear::build(s, &ls.v)?,
                o: ResidentLinear::build(s, &ls.o)?,
                gate: ResidentLinear::build(s, &ls.gate)?,
                up: ResidentLinear::build(s, &ls.up)?,
                down: ResidentLinear::build(s, &ls.down)?,
            });
            kv_k.push(alloc(max_ctx * kv_width, "decode kv_k alloc")?);
            kv_v.push(alloc(max_ctx * kv_width, "decode kv_v alloc")?);
        }

        // Resolve the kernels from the resident modules (decode + the add module's tiled GEMM).
        let f = |m: &Arc<CudaModule>, name: &str| -> Result<CudaFunction, BackendError> {
            m.load_function(name).map_err(|e| driver_err("resolve decode kernel", &e))
        };
        let dm = &self._decode_module;

        Ok(CudaDecodeModel {
            stream: Arc::clone(&self.stream),
            f_rmsnorm: f(dm, KERNEL_NAME_RMSNORM)?,
            f_rope: f(dm, KERNEL_NAME_ROPE)?,
            f_attn: f(dm, KERNEL_NAME_ATTN)?,
            f_residual: f(dm, KERNEL_NAME_RESIDUAL)?,
            f_embed: f(dm, KERNEL_NAME_EMBED)?,
            f_lm_head: f(dm, KERNEL_NAME_LM_HEAD)?,
            f_relu2: f(dm, KERNEL_NAME_RELU2_GATE)?,
            f_quant: f(dm, KERNEL_NAME_ACT_QUANT_TILED)?,
            f_tiled: f(&self._module, KERNEL_NAME_TILED)?,
            f_scale: f(dm, KERNEL_NAME_SCALE_MUL)?,
            d_token_embd,
            d_output_norm,
            d_cos,
            d_sin,
            layers,
            kv_k,
            kv_v,
            cache_len: 0,
            d_x: alloc(n_embd, "decode d_x")?,
            d_normed: alloc(n_embd, "decode d_normed")?,
            d_q: alloc(q_width, "decode d_q")?,
            d_knew: alloc(kv_width, "decode d_knew")?,
            d_vnew: alloc(kv_width, "decode d_vnew")?,
            d_attn: alloc(q_width, "decode d_attn")?,
            d_attn_sn: alloc(q_width, "decode d_attn_sn")?,
            d_proj_out: alloc(n_embd, "decode d_proj_out")?,
            d_gate: alloc(n_ff, "decode d_gate")?,
            d_up: alloc(n_ff, "decode d_up")?,
            d_gate_sn: alloc(n_ff, "decode d_gate_sn")?,
            d_scores: alloc(n_head * max_ctx, "decode d_scores")?,
            d_logits: alloc(vocab, "decode d_logits")?,
            d_qact: alloc(TILED_K_MAX.min(n_ff.max(q_width).max(n_embd)), "decode d_qact")?,
            d_act_scale: alloc(1, "decode d_act_scale")?,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            half,
            q_width,
            kv_width,
            n_ff,
            vocab,
            max_ctx,
            rms_eps,
            attn_scale: 1.0f32 / (head_dim as f32).sqrt(),
            d_ctrl: s
                .alloc_zeros::<i32>(4)
                .map_err(|e| driver_err("decode d_ctrl", &e))?,
            cap_stream: self
                .stream
                .context()
                .new_stream()
                .map_err(|e| driver_err("decode capture stream", &e))?,
            graph: None,
            raw: None,
        })
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
    #[allow(clippy::too_many_arguments)] // act + weights + scales + shape + format + out + kernel
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
            .arg(buf.device.as_ref())
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
            &func, tile, &d_qact, buf.device.as_ref(), &d_act_scale, &d_wscale, &mut d_out, m_i, n_i, k_i,
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

    /// Opt into the downcast hook so the runner can recover `&CudaBackend` from its
    /// `&dyn TernaryBackend` and build a [`CudaDecodeModel`] (the device-resident
    /// decode fast path, v0.3.1).
    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
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
            device: Arc::new(device),
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

// ===========================================================================
// v0.3.1 (ADR 0013): the device-resident M=1 decode forward.
//
// `CudaDecodeModel` keeps the residual stream, the KV cache, the RoPE table, and
// every weight (dense + the shared ternary `Arc`s) in VRAM across all layers, so a
// decode step crosses the host boundary exactly **twice**: one token-id H2D
// (implicit in `embedding_gather`'s scalar arg) and one logits D2H. The old
// host-orchestrated forward did ~210 synchronous round-trips/token; this does ~1.
//
// Every kernel it launches is the same bit-matching decode kernel the goldens pin
// (`decode.cu`, `--fmad=false`), so moving the math onto the GPU does not change a
// rounding — greedy parity with the host (hence transformers) is preserved, modulo
// the softmax/attention `expf` (≤3 ULP, the documented fallback-gated op).
// ===========================================================================

/// Borrowed spec of one ternary projection for [`CudaBackend::build_decode_model`]:
/// the already-uploaded device weight (shared, not re-uploaded) + its per-channel
/// scales. The weight must be a TQ2_0 [`CudaBuffer`] (the decode GEMM is TQ2_0-only).
#[allow(missing_debug_implementations)] // holds `&dyn DeviceBuffer` (no Debug)
pub struct DecodeLinearSpec<'a> {
    /// The uploaded ternary weight (downcast to [`CudaBuffer`] internally).
    pub weights: &'a dyn DeviceBuffer,
    /// Per-output-channel f32 weight scales, length `N`.
    pub scales: &'a [f32],
}

/// Borrowed spec of one transformer block (dense norm weights + the 7 ternary
/// projections). `attn_sub_norm`/`ffn_sub_norm` may be empty to skip the BitNet
/// sub-norms (matches the host `if len == width` guard).
#[allow(missing_debug_implementations)] // transitively holds `&dyn DeviceBuffer`
pub struct DecodeLayerSpec<'a> {
    /// `input_layernorm` weight, length `n_embd`.
    pub attn_norm: &'a [f32],
    /// `attn_sub_norm` weight (length `q_width`), or empty to skip.
    pub attn_sub_norm: &'a [f32],
    /// `post_attention_layernorm` weight, length `n_embd`.
    pub ffn_norm: &'a [f32],
    /// `ffn_sub_norm` weight (length `n_ff`), or empty to skip.
    pub ffn_sub_norm: &'a [f32],
    /// Attention q/k/v/o projections.
    pub q: DecodeLinearSpec<'a>,
    /// See [`q`](Self::q).
    pub k: DecodeLinearSpec<'a>,
    /// See [`q`](Self::q).
    pub v: DecodeLinearSpec<'a>,
    /// See [`q`](Self::q).
    pub o: DecodeLinearSpec<'a>,
    /// MLP gate/up/down projections.
    pub gate: DecodeLinearSpec<'a>,
    /// See [`gate`](Self::gate).
    pub up: DecodeLinearSpec<'a>,
    /// See [`gate`](Self::gate).
    pub down: DecodeLinearSpec<'a>,
}

/// Borrowed spec of a whole model for [`CudaBackend::build_decode_model`]. Carries
/// the dense fp32 weights (uploaded once) + per-layer specs + the head geometry.
#[allow(missing_debug_implementations)] // transitively holds `&dyn DeviceBuffer`
pub struct DecodeModelSpec<'a> {
    /// Token embedding `[vocab, n_embd]` row-major (also the tied LM head).
    pub token_embd: &'a [f32],
    /// Final `output_norm` weight, length `n_embd`.
    pub output_norm: &'a [f32],
    /// Per-layer specs, length `n_layers`.
    pub layers: Vec<DecodeLayerSpec<'a>>,
    /// Hidden size.
    pub n_embd: usize,
    /// Query head count.
    pub n_head: usize,
    /// KV head count (GQA).
    pub n_head_kv: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// FFN inner size.
    pub n_ff: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Maximum context (KV arena rows).
    pub max_ctx: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
}

/// One ternary projection, resident: the shared device weight + device scales.
#[derive(Debug)]
struct ResidentLinear {
    device: Arc<CudaSlice<u8>>,
    scales: CudaSlice<f32>,
    n: usize,
    k: usize,
    row_bytes: usize,
}

impl ResidentLinear {
    /// Build from a borrowed spec: share the device weight (`Arc` clone, no
    /// re-upload) and upload the per-channel scales once.
    fn build(stream: &Arc<CudaStream>, spec: &DecodeLinearSpec) -> Result<Self, BackendError> {
        let buf = spec
            .weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("decode weight is not a CudaBuffer".into()))?;
        let row_bytes = buf.tq2_0_row_bytes().ok_or(BackendError::UnsupportedFormat(
            TernaryFormat::I2sInt8,
        ))?;
        let (n, k) = buf.dims();
        if spec.scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: spec.scales.len(),
            });
        }
        if k > TILED_K_MAX {
            return Err(BackendError::InvalidInput(format!(
                "decode K={k} exceeds the tiled cap {TILED_K_MAX}"
            )));
        }
        let scales = stream
            .clone_htod(spec.scales)
            .map_err(|e| driver_err("decode scales htod", &e))?;
        Ok(Self {
            device: buf.device_arc(),
            scales,
            n,
            k,
            row_bytes,
        })
    }
}

/// One resident transformer block: dense norms + the 7 projections (no KV — the KV
/// arenas live model-side so a layer's weights and its KV can be borrowed disjointly).
#[derive(Debug)]
struct ResidentLayer {
    attn_norm: CudaSlice<f32>,
    attn_sub_norm: Option<CudaSlice<f32>>,
    ffn_norm: CudaSlice<f32>,
    ffn_sub_norm: Option<CudaSlice<f32>>,
    q: ResidentLinear,
    k: ResidentLinear,
    v: ResidentLinear,
    o: ResidentLinear,
    gate: ResidentLinear,
    up: ResidentLinear,
    down: ResidentLinear,
}

/// A fully device-resident BitNet decoder. One [`step`](CudaDecodeModel::step) is a
/// single-token (M=1) forward run entirely on the GPU. See the section banner above.
#[allow(missing_debug_implementations)]
pub struct CudaDecodeModel {
    stream: Arc<CudaStream>,
    // Decode kernels (loaded from the resident modules at build).
    f_rmsnorm: CudaFunction,
    f_rope: CudaFunction,
    f_attn: CudaFunction,
    f_residual: CudaFunction,
    f_embed: CudaFunction,
    f_lm_head: CudaFunction,
    f_relu2: CudaFunction,
    f_quant: CudaFunction,
    f_tiled: CudaFunction,
    f_scale: CudaFunction,
    // Dense device weights (uploaded once).
    d_token_embd: CudaSlice<f32>,
    d_output_norm: CudaSlice<f32>,
    d_cos: CudaSlice<f32>,
    d_sin: CudaSlice<f32>,
    layers: Vec<ResidentLayer>,
    // Per-layer KV arenas `[max_ctx * kv_width]`, model-side for disjoint borrows.
    kv_k: Vec<CudaSlice<f32>>,
    kv_v: Vec<CudaSlice<f32>>,
    cache_len: usize,
    // Reused scratch (sized once).
    d_x: CudaSlice<f32>,
    d_normed: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_knew: CudaSlice<f32>,
    d_vnew: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_attn_sn: CudaSlice<f32>,
    d_proj_out: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_gate_sn: CudaSlice<f32>,
    d_scores: CudaSlice<f32>,
    d_logits: CudaSlice<f32>,
    d_qact: CudaSlice<f32>,
    d_act_scale: CudaSlice<f32>,
    // Geometry.
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    half: usize,
    q_width: usize,
    kv_width: usize,
    n_ff: usize,
    vocab: usize,
    max_ctx: usize,
    rms_eps: f32,
    attn_scale: f32,
    // --- v0.3.2 CUDA-graph decode path ---
    /// Device control block `[token, pos, cache_len, _]`, rewritten per token so one
    /// captured graph replays across tokens (the `_g` kernels read it).
    d_ctrl: CudaSlice<i32>,
    /// Dedicated capture/replay stream (capturing the default stream is disallowed).
    cap_stream: Arc<CudaStream>,
    /// The captured decode graph, built lazily on first graph step. Declared **before**
    /// `raw` so it drops (graph-exec destroyed) before the modules it references unload.
    graph: Option<CudaGraph>,
    /// Raw-loaded PTX modules + `CUfunction` handles for the captured raw launches
    /// (the safe `CudaFunction` hides `cu_function`). `None` until the graph path is used.
    raw: Option<RawGraphKernels>,
}

impl CudaDecodeModel {
    /// Reset the KV cache (start a fresh sequence). The arena bytes are left as-is;
    /// only the watermark moves, exactly like [`crate`]'s host `KvCache::reset`.
    pub fn reset(&mut self) {
        self.cache_len = 0;
    }

    /// Number of tokens currently cached (the decode position the next `step` writes).
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    /// Run one M=1 decode step for `token` at absolute position `pos`, returning the
    /// next-token logits `[vocab]`. The whole forward stays on the device; only the
    /// logits are copied back. `pos` must equal [`cache_len`](Self::cache_len).
    ///
    /// # Errors
    /// [`BackendError`] on a device failure, capacity overflow, or shape mismatch.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, BackendError> {
        if self.cache_len >= self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "decode context overflow: cache_len={} max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }
        // `pos` must track the KV watermark: RoPE indexes the table by `pos` while KV
        // append + the attention range use `cache_len`. A mismatch would silently apply
        // the wrong rotation (and could index the RoPE table out of bounds), so this is
        // a hard runtime guard, not a debug assert.
        if pos != self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "decode pos={pos} must equal the KV watermark cache_len={}",
                self.cache_len
            )));
        }
        // The tied embedding/LM-head table has `vocab` rows; an out-of-range token id
        // would make `embedding_gather` read past `d_token_embd`.
        if token as usize >= self.vocab {
            return Err(BackendError::InvalidInput(format!(
                "decode token id {token} out of range (vocab={})",
                self.vocab
            )));
        }
        let n_embd = self.n_embd;
        let eps = self.rms_eps;

        // Embedding gather: d_x = token_embd[token].
        Self::launch_embed(&self.stream, &self.f_embed, &self.d_token_embd, token, n_embd, &mut self.d_x)?;

        for li in 0..self.layers.len() {
            self.layer(li, pos)?;
        }

        // Final norm over the (single) last token, then the tied LM head.
        Self::launch_rmsnorm(
            &self.stream, &self.f_rmsnorm, &self.d_x, &self.d_output_norm, eps, n_embd, &mut self.d_normed,
        )?;
        Self::launch_lm_head(
            &self.stream, &self.f_lm_head, &self.d_normed, &self.d_token_embd, n_embd, self.vocab, &mut self.d_logits,
        )?;

        let mut logits = vec![0.0f32; self.vocab];
        self.stream
            .memcpy_dtoh(&self.d_logits, &mut logits)
            .map_err(|e| driver_err("decode logits dtoh", &e))?;
        // Bump the watermark only after a fully successful step. A failure mid-step
        // leaves this token's KV rows written at offset `cache_len` but the watermark
        // unmoved; a retry at the same `pos` overwrites them in place (idempotent), so
        // the cache is never left half-advanced.
        self.cache_len += 1;
        Ok(logits)
    }

    /// One transformer block, fully on-device, into/out of the resident residual `d_x`.
    fn layer(&mut self, li: usize, pos: usize) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let (n_embd, q_width, kv_width) = (self.n_embd, self.q_width, self.kv_width);
        let (n_head, n_head_kv, head_dim, half) = (self.n_head, self.n_head_kv, self.head_dim, self.half);

        // --- pre-norm attention ---
        Self::launch_rmsnorm(&self.stream, &self.f_rmsnorm, &self.d_x, &self.layers[li].attn_norm, eps, n_embd, &mut self.d_normed)?;
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, &self.d_normed, &self.layers[li].q, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_q)?;
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, &self.d_normed, &self.layers[li].k, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_knew)?;
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, &self.d_normed, &self.layers[li].v, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_vnew)?;

        // RoPE on q and the new k (this token's position row of the precomputed table).
        let base = pos * half;
        {
            let cos_v = self.d_cos.slice(base..base + half);
            let sin_v = self.d_sin.slice(base..base + half);
            Self::launch_rope(&self.stream, &self.f_rope, &mut self.d_q, &cos_v, &sin_v, n_head, head_dim)?;
        }
        {
            let cos_v = self.d_cos.slice(base..base + half);
            let sin_v = self.d_sin.slice(base..base + half);
            Self::launch_rope(&self.stream, &self.f_rope, &mut self.d_knew, &cos_v, &sin_v, n_head_kv, head_dim)?;
        }

        // Append the new k/v to this layer's KV arena at the watermark, dtod.
        let off = self.cache_len * kv_width;
        {
            let mut dst = self.kv_k[li].slice_mut(off..off + kv_width);
            self.stream.memcpy_dtod(&self.d_knew, &mut dst).map_err(|e| driver_err("kv append k", &e))?;
        }
        {
            let mut dst = self.kv_v[li].slice_mut(off..off + kv_width);
            self.stream.memcpy_dtod(&self.d_vnew, &mut dst).map_err(|e| driver_err("kv append v", &e))?;
        }

        // Attention over the cached prefix (ctx = watermark+1, last visible = watermark).
        let ctx = self.cache_len + 1;
        Self::launch_attention(
            &self.stream, &self.f_attn, &self.d_q, &self.kv_k[li], &self.kv_v[li], &mut self.d_attn, &mut self.d_scores,
            ctx, n_head, n_head_kv, head_dim, self.attn_scale, self.cache_len,
        )?;

        // BitNet attn_sub_norm before o_proj (over q_width == n_embd), then o_proj +
        // the first residual into d_x.
        let attn_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].attn_sub_norm.as_ref() {
            Self::launch_rmsnorm(&self.stream, &self.f_rmsnorm, &self.d_attn, sn, eps, q_width, &mut self.d_attn_sn)?;
            &self.d_attn_sn
        } else {
            &self.d_attn
        };
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, attn_in, &self.layers[li].o, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_proj_out)?;
        Self::launch_residual(&self.stream, &self.f_residual, &mut self.d_x, &self.d_proj_out, n_embd)?;

        // --- pre-norm ReLU² MLP ---
        Self::launch_rmsnorm(&self.stream, &self.f_rmsnorm, &self.d_x, &self.layers[li].ffn_norm, eps, n_embd, &mut self.d_normed)?;
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, &self.d_normed, &self.layers[li].gate, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_gate)?;
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, &self.d_normed, &self.layers[li].up, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_up)?;
        Self::launch_relu2(&self.stream, &self.f_relu2, &mut self.d_gate, &self.d_up, self.n_ff)?;
        let down_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
            Self::launch_rmsnorm(&self.stream, &self.f_rmsnorm, &self.d_gate, sn, eps, self.n_ff, &mut self.d_gate_sn)?;
            &self.d_gate_sn
        } else {
            &self.d_gate
        };
        Self::gemm(&self.stream, &self.f_quant, &self.f_tiled, &self.f_scale, down_in, &self.layers[li].down, &mut self.d_qact, &mut self.d_act_scale, &mut self.d_proj_out)?;
        Self::launch_residual(&self.stream, &self.f_residual, &mut self.d_x, &self.d_proj_out, n_embd)?;
        Ok(())
    }

    // --- launch helpers (associated fns so step()/layer() can pass disjoint field
    //     borrows of `self` without going through a `&self`/`&mut self` method). ---

    fn launch_rmsnorm(
        stream: &Arc<CudaStream>, func: &CudaFunction, x: &CudaSlice<f32>, w: &CudaSlice<f32>,
        eps: f32, n: usize, out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(w).arg(&eps).arg(&n_i).arg(out);
        // SAFETY: `rmsnorm_f32(const float* x, const float* w, float eps, int n, float* out)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident rmsnorm", &e))?; }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        stream: &Arc<CudaStream>, f_quant: &CudaFunction, f_tiled: &CudaFunction, f_scale: &CudaFunction,
        d_in: &CudaSlice<f32>, lin: &ResidentLinear,
        d_qact: &mut CudaSlice<f32>, d_act_scale: &mut CudaSlice<f32>, d_out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (n, k) = (lin.n, lin.k);
        let (n_i, k_i, m_i, rb_i) = (n as i32, k as i32, 1i32, lin.row_bytes as i32);
        // 1. on-device A8 quant of the activation row. (Reborrow the `&mut` scratch so
        //    the same bindings can be reused by the later launches.)
        {
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut l = stream.launch_builder(f_quant);
            l.arg(d_in).arg(&k_i).arg(&mut *d_qact).arg(&mut *d_act_scale);
            // SAFETY: `act_quant_tiled_f32(const float* act, int k, float* q, float* scale)`.
            #[allow(unsafe_code)]
            unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident gemm quant", &e))?; }
        }
        // 2. tiled add-only f64 GEMM (M=1), folds the per-channel weight scale.
        {
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: (k * 4) as u32,
            };
            let mut l = stream.launch_builder(f_tiled);
            l.arg(&*d_qact).arg(lin.device.as_ref()).arg(&lin.scales).arg(&mut *d_out).arg(&m_i).arg(&n_i).arg(&k_i).arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled(act, weights, scales, out, m, n, k, row_bytes)`.
            #[allow(unsafe_code)]
            unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident gemm tiled", &e))?; }
        }
        // 3. per-token activation-scale fold: out *= act_scale.
        {
            let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut l = stream.launch_builder(f_scale);
            l.arg(&mut *d_out).arg(&*d_act_scale).arg(&n_i);
            // SAFETY: `scale_mul_f32(float* out, const float* s, int n)`.
            #[allow(unsafe_code)]
            unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident gemm fold", &e))?; }
        }
        Ok(())
    }

    fn launch_rope(
        stream: &Arc<CudaStream>, func: &CudaFunction, x: &mut CudaSlice<f32>,
        cos_t: &CudaView<f32>, sin_t: &CudaView<f32>, n_head: usize, head_dim: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i) = (n_head as i32, head_dim as i32);
        let total = (n_head * (head_dim / 2)) as u32;
        let cfg = LaunchConfig { grid_dim: (total.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(cos_t).arg(sin_t).arg(&nh_i).arg(&hd_i);
        // SAFETY: `rope_apply_f32(float* x, const float* cos_t, const float* sin_t, int n_head, int head_dim)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident rope", &e))?; }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention(
        stream: &Arc<CudaStream>, func: &CudaFunction, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>, scores: &mut CudaSlice<f32>,
        ctx: usize, n_head: usize, n_head_kv: usize, head_dim: usize, scale: f32, limit: usize,
    ) -> Result<(), BackendError> {
        let (ctx_i, nh_i, nhkv_i, hd_i, lim_i) = (ctx as i32, n_head as i32, n_head_kv as i32, head_dim as i32, limit as i32);
        let threads = 64u32;
        let cfg = LaunchConfig { grid_dim: ((n_head as u32).div_ceil(threads), 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(q).arg(k).arg(v).arg(out).arg(scores).arg(&ctx_i).arg(&nh_i).arg(&nhkv_i).arg(&hd_i).arg(&scale).arg(&lim_i);
        // SAFETY: `gqa_attention_decode_f32(q, k, v, out, scores, ctx, n_head, n_head_kv, head_dim, scale, limit)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident attention", &e))?; }
        Ok(())
    }

    fn launch_residual(
        stream: &Arc<CudaStream>, func: &CudaFunction, x: &mut CudaSlice<f32>, y: &CudaSlice<f32>, n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(y).arg(&n_i);
        // SAFETY: `residual_add_f32(float* x, const float* y, int n)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident residual", &e))?; }
        Ok(())
    }

    fn launch_relu2(
        stream: &Arc<CudaStream>, func: &CudaFunction, gate: &mut CudaSlice<f32>, up: &CudaSlice<f32>, n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(gate).arg(up).arg(&n_i);
        // SAFETY: `relu2_gate_f32(float* gate, const float* up, int n)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident relu2", &e))?; }
        Ok(())
    }

    fn launch_embed(
        stream: &Arc<CudaStream>, func: &CudaFunction, table: &CudaSlice<f32>, tok: u32, n_embd: usize, out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (tok_i, ne_i) = (tok as i32, n_embd as i32);
        let cfg = LaunchConfig { grid_dim: ((n_embd as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(table).arg(&tok_i).arg(&ne_i).arg(out);
        // SAFETY: `embedding_gather_f32(const float* table, int tok, int n_embd, float* out)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident embed", &e))?; }
        Ok(())
    }

    fn launch_lm_head(
        stream: &Arc<CudaStream>, func: &CudaFunction, h: &CudaSlice<f32>, embd: &CudaSlice<f32>,
        n_embd: usize, vocab: usize, logits: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i) = (n_embd as i32, vocab as i32);
        let cfg = LaunchConfig { grid_dim: ((vocab as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut l = stream.launch_builder(func);
        l.arg(h).arg(embd).arg(&ne_i).arg(&v_i).arg(logits);
        // SAFETY: `lm_head_f32(const float* h, const float* embd, int n_embd, int vocab, float* logits)`.
        #[allow(unsafe_code)]
        unsafe { l.launch(cfg).map_err(|e| driver_err("launch resident lm_head", &e))?; }
        Ok(())
    }

    // ===================== v0.3.2 CUDA-graph decode path =====================

    /// Run one decode step through the **captured CUDA graph** (built lazily on first
    /// call): rewrite the device control block `[token, pos, cache_len]`, replay the
    /// whole-forward graph with a single launch, and read back the logits. Crosses the
    /// host boundary with two tiny transfers (ctrl H2D + logits D2H) and **one** graph
    /// launch — replacing ~930 per-token kernel launches. Numerically identical to
    /// [`step`](Self::step) (the `_g` kernels read the control block but do the same
    /// math). `pos` must equal [`cache_len`](Self::cache_len).
    ///
    /// # Errors
    /// [`BackendError`] on capacity overflow, a `pos`/token guard, or a device failure.
    pub fn step_graph(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, BackendError> {
        if self.cache_len >= self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "decode context overflow: cache_len={} max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }
        if pos != self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "decode pos={pos} must equal the KV watermark cache_len={}",
                self.cache_len
            )));
        }
        if token as usize >= self.vocab {
            return Err(BackendError::InvalidInput(format!(
                "decode token id {token} out of range (vocab={})",
                self.vocab
            )));
        }
        if self.graph.is_none() {
            self.capture_graph()?;
        }

        // Rewrite the control block, then replay. Both are on `cap_stream`, so the H2D
        // is ordered before the graph reads `d_ctrl`.
        let ctrl = [token as i32, pos as i32, self.cache_len as i32, 0i32];
        self.cap_stream
            .memcpy_htod(&ctrl, &mut self.d_ctrl)
            .map_err(|e| driver_err("decode ctrl htod", &e))?;
        self.graph
            .as_ref()
            .expect("graph captured above")
            .launch()
            .map_err(|e| driver_err("decode graph launch", &e))?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("decode graph sync", &e))?;

        let mut logits = vec![0.0f32; self.vocab];
        self.cap_stream
            .memcpy_dtoh(&self.d_logits, &mut logits)
            .map_err(|e| driver_err("decode graph logits dtoh", &e))?;
        self.cache_len += 1;
        Ok(logits)
    }

    /// Load the raw kernels (once) and record + instantiate the decode graph.
    fn capture_graph(&mut self) -> Result<(), BackendError> {
        if self.raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            self.raw = Some(RawGraphKernels::load(&ctx)?);
        }
        let graph = self.record_graph()?;
        self.graph = Some(graph);
        Ok(())
    }

    fn raw(&self) -> &RawGraphKernels {
        self.raw.as_ref().expect("raw kernels loaded before record")
    }

    /// Extract every buffer's stable device pointer (guards dropped here, outside
    /// capture), then capture the full forward via raw launches on `cap_stream`.
    fn record_graph(&self) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            n: l.n,
            k: l.k,
            rb: l.row_bytes,
        };
        let layers: Vec<LayerPtrs> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| LayerPtrs {
                attn_norm: dptr(&l.attn_norm, s),
                attn_sub_norm: l.attn_sub_norm.as_ref().map(|b| dptr(b, s)),
                ffn_norm: dptr(&l.ffn_norm, s),
                ffn_sub_norm: l.ffn_sub_norm.as_ref().map(|b| dptr(b, s)),
                q: lin(&l.q),
                k: lin(&l.k),
                v: lin(&l.v),
                o: lin(&l.o),
                gate: lin(&l.gate),
                up: lin(&l.up),
                down: lin(&l.down),
                kv_k: dptr(&self.kv_k[li], s),
                kv_v: dptr(&self.kv_v[li], s),
            })
            .collect();
        let p = GraphPtrs {
            d_x: dptr(&self.d_x, s),
            d_normed: dptr(&self.d_normed, s),
            d_q: dptr(&self.d_q, s),
            d_knew: dptr(&self.d_knew, s),
            d_vnew: dptr(&self.d_vnew, s),
            d_attn: dptr(&self.d_attn, s),
            d_attn_sn: dptr(&self.d_attn_sn, s),
            d_proj_out: dptr(&self.d_proj_out, s),
            d_gate: dptr(&self.d_gate, s),
            d_up: dptr(&self.d_up, s),
            d_gate_sn: dptr(&self.d_gate_sn, s),
            d_scores: dptr(&self.d_scores, s),
            d_logits: dptr(&self.d_logits, s),
            d_qact: dptr(&self.d_qact, s),
            d_act_scale: dptr(&self.d_act_scale, s),
            d_ctrl: dptr(&self.d_ctrl, s),
            d_cos: dptr(&self.d_cos, s),
            d_sin: dptr(&self.d_sin, s),
            d_token_embd: dptr(&self.d_token_embd, s),
            d_output_norm: dptr(&self.d_output_norm, s),
        };
        // Drain the events the device_ptr extraction recorded, so the capture (which
        // uses only raw launches) carries no pre-capture dependency.
        s.synchronize().map_err(|e| driver_err("pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("decode begin_capture", &e))?;

        // The exact op order of `step` + `layer`, all raw-launched on `cap_stream`.
        self.g_embed(p.d_token_embd, p.d_ctrl, p.d_x)?;
        for lp in &layers {
            self.g_layer(&p, lp)?;
        }
        self.g_rmsnorm(p.d_x, p.d_output_norm, self.n_embd, p.d_normed)?;
        self.g_lm_head(p.d_normed, p.d_token_embd, p.d_logits)?;

        let graph = s
            .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| driver_err("decode end_capture", &e))?
            .ok_or_else(|| BackendError::Backend("decode graph capture produced no graph".into()))?;
        Ok(graph)
    }

    /// One transformer block, raw-launched into the capture. Mirrors [`layer`](Self::layer).
    fn g_layer(&self, p: &GraphPtrs, l: &LayerPtrs) -> Result<(), BackendError> {
        let (n_embd, q_width, kv_width) = (self.n_embd, self.q_width, self.kv_width);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);

        // pre-norm attention
        self.g_rmsnorm(p.d_x, l.attn_norm, n_embd, p.d_normed)?;
        self.g_gemm(p.d_normed, &l.q, p.d_qact, p.d_act_scale, p.d_q)?;
        self.g_gemm(p.d_normed, &l.k, p.d_qact, p.d_act_scale, p.d_knew)?;
        self.g_gemm(p.d_normed, &l.v, p.d_qact, p.d_act_scale, p.d_vnew)?;
        self.g_rope(p.d_q, p.d_cos, p.d_sin, p.d_ctrl, n_head, head_dim)?;
        self.g_rope(p.d_knew, p.d_cos, p.d_sin, p.d_ctrl, n_head_kv, head_dim)?;
        self.g_kv_append(p.d_knew, l.kv_k, p.d_ctrl, kv_width)?;
        self.g_kv_append(p.d_vnew, l.kv_v, p.d_ctrl, kv_width)?;
        self.g_attn(p.d_q, l.kv_k, l.kv_v, p.d_attn, p.d_scores, p.d_ctrl)?;
        let attn_in = if let Some(sn) = l.attn_sub_norm {
            self.g_rmsnorm(p.d_attn, sn, q_width, p.d_attn_sn)?;
            p.d_attn_sn
        } else {
            p.d_attn
        };
        self.g_gemm(attn_in, &l.o, p.d_qact, p.d_act_scale, p.d_proj_out)?;
        self.g_residual(p.d_x, p.d_proj_out, n_embd)?;

        // pre-norm ReLU² MLP
        self.g_rmsnorm(p.d_x, l.ffn_norm, n_embd, p.d_normed)?;
        self.g_gemm(p.d_normed, &l.gate, p.d_qact, p.d_act_scale, p.d_gate)?;
        self.g_gemm(p.d_normed, &l.up, p.d_qact, p.d_act_scale, p.d_up)?;
        self.g_relu2(p.d_gate, p.d_up, self.n_ff)?;
        let down_in = if let Some(sn) = l.ffn_sub_norm {
            self.g_rmsnorm(p.d_gate, sn, self.n_ff, p.d_gate_sn)?;
            p.d_gate_sn
        } else {
            p.d_gate
        };
        self.g_gemm(down_in, &l.down, p.d_qact, p.d_act_scale, p.d_proj_out)?;
        self.g_residual(p.d_x, p.d_proj_out, n_embd)?;
        Ok(())
    }

    // Raw-launch helpers (build the kernel_params array from pre-extracted device
    // pointers + scalar locals; only `raw_launch` is unsafe). `cs` = capture stream.

    fn g_rmsnorm(&self, x: sys::CUdeviceptr, w: sys::CUdeviceptr, n: usize, out: sys::CUdeviceptr) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let n_i = n as i32;
        let mut params = [pp(&x), pp(&w), pp(&eps), pp(&n_i), pp(&out)];
        raw_launch(self.raw().rmsnorm, (1, 1, 1), (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_gemm(&self, d_in: sys::CUdeviceptr, lin: &LinPtrs, d_qact: sys::CUdeviceptr, d_act_scale: sys::CUdeviceptr, d_out: sys::CUdeviceptr) -> Result<(), BackendError> {
        let cs = self.cap_stream.cu_stream();
        let (n_i, k_i, m_i, rb_i) = (lin.n as i32, lin.k as i32, 1i32, lin.rb as i32);
        // 1. act quant
        {
            let mut params = [pp(&d_in), pp(&k_i), pp(&d_qact), pp(&d_act_scale)];
            raw_launch(self.raw().act_quant, (1, 1, 1), (256, 1, 1), 0, cs, &mut params)?;
        }
        // 2. tiled GEMM
        {
            let grid = ((lin.n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1);
            let smem = (lin.k * 4) as u32;
            let mut params = [pp(&d_qact), pp(&lin.w), pp(&lin.sc), pp(&d_out), pp(&m_i), pp(&n_i), pp(&k_i), pp(&rb_i)];
            raw_launch(self.raw().tiled, grid, (WARPS_PER_BLOCK * 32, 1, 1), smem, cs, &mut params)?;
        }
        // 3. scale fold
        {
            let grid = ((lin.n as u32).div_ceil(256), 1, 1);
            let mut params = [pp(&d_out), pp(&d_act_scale), pp(&n_i)];
            raw_launch(self.raw().scale, grid, (256, 1, 1), 0, cs, &mut params)?;
        }
        Ok(())
    }

    fn g_rope(&self, x: sys::CUdeviceptr, cos_t: sys::CUdeviceptr, sin_t: sys::CUdeviceptr, ctrl: sys::CUdeviceptr, n_head: usize, head_dim: usize) -> Result<(), BackendError> {
        let (nh_i, hd_i) = (n_head as i32, head_dim as i32);
        let total = (n_head * (head_dim / 2)) as u32;
        let grid = (total.div_ceil(256), 1, 1);
        let mut params = [pp(&x), pp(&cos_t), pp(&sin_t), pp(&ctrl), pp(&nh_i), pp(&hd_i)];
        raw_launch(self.raw().rope_g, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_kv_append(&self, src: sys::CUdeviceptr, kv_base: sys::CUdeviceptr, ctrl: sys::CUdeviceptr, kv_width: usize) -> Result<(), BackendError> {
        let kw_i = kv_width as i32;
        let grid = ((kv_width as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&src), pp(&kv_base), pp(&ctrl), pp(&kw_i)];
        raw_launch(self.raw().kv_append, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_attn(&self, q: sys::CUdeviceptr, k: sys::CUdeviceptr, v: sys::CUdeviceptr, out: sys::CUdeviceptr, scores: sys::CUdeviceptr, ctrl: sys::CUdeviceptr) -> Result<(), BackendError> {
        let (mc_i, nh_i, nhkv_i, hd_i) = (self.max_ctx as i32, self.n_head as i32, self.n_head_kv as i32, self.head_dim as i32);
        let scale = self.attn_scale;
        let threads = 64u32;
        let grid = ((self.n_head as u32).div_ceil(threads), 1, 1);
        let mut params = [pp(&q), pp(&k), pp(&v), pp(&out), pp(&scores), pp(&ctrl), pp(&mc_i), pp(&nh_i), pp(&nhkv_i), pp(&hd_i), pp(&scale)];
        raw_launch(self.raw().attn_g, grid, (threads, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_residual(&self, x: sys::CUdeviceptr, y: sys::CUdeviceptr, n: usize) -> Result<(), BackendError> {
        let n_i = n as i32;
        let grid = ((n as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&x), pp(&y), pp(&n_i)];
        raw_launch(self.raw().residual, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_relu2(&self, gate: sys::CUdeviceptr, up: sys::CUdeviceptr, n: usize) -> Result<(), BackendError> {
        let n_i = n as i32;
        let grid = ((n as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&gate), pp(&up), pp(&n_i)];
        raw_launch(self.raw().relu2, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_embed(&self, table: sys::CUdeviceptr, ctrl: sys::CUdeviceptr, out: sys::CUdeviceptr) -> Result<(), BackendError> {
        let ne_i = self.n_embd as i32;
        let grid = ((self.n_embd as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&table), pp(&ctrl), pp(&ne_i), pp(&out)];
        raw_launch(self.raw().embed_g, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }

    fn g_lm_head(&self, h: sys::CUdeviceptr, embd: sys::CUdeviceptr, logits: sys::CUdeviceptr) -> Result<(), BackendError> {
        let (ne_i, v_i) = (self.n_embd as i32, self.vocab as i32);
        let grid = ((self.vocab as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&h), pp(&embd), pp(&ne_i), pp(&v_i), pp(&logits)];
        raw_launch(self.raw().lm_head, grid, (256, 1, 1), 0, self.cap_stream.cu_stream(), &mut params)
    }
}

/// A kernel param: a pointer to the arg value `v`. For a pointer arg, `v` is the
/// `CUdeviceptr`; for a by-value arg, `v` is the scalar. Casting a reference to a raw
/// pointer is safe (the deref happens inside `cuLaunchKernel`); the caller keeps `v`
/// alive across the launch (it is a local that outlives `raw_launch`, and graph capture
/// snapshots the value into the kernel node).
fn pp<T>(v: &T) -> *mut c_void {
    (v as *const T) as *mut c_void
}

/// Extract a buffer's stable device address, dropping the `SyncOnDrop` guard
/// immediately (outside any capture — its drop records an event, forbidden inside a
/// capture). The `CUdeviceptr` is valid for the buffer's lifetime, so it is safe to
/// bake into a captured graph that the buffer outlives.
fn dptr<T>(buf: &CudaSlice<T>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (ptr, guard) = buf.device_ptr(stream);
    drop(guard);
    ptr
}

/// Launch a kernel via the RAW driver entry point, bypassing cudarc's safe launch
/// (whose per-buffer event waits trip `STREAM_CAPTURE_ISOLATION` during capture).
fn raw_launch(
    func: sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    stream: sys::CUstream,
    params: &mut [*mut c_void],
) -> Result<(), BackendError> {
    // SAFETY: `func` is a valid `CUfunction` from a loaded `RawGraphKernels` module;
    // `params` holds exactly one pointer per kernel arg in declaration order, each
    // pointing to a live value (a `CUdeviceptr` for a pointer arg, the scalar for a
    // by-value arg) that outlives this call. The kernel signatures are pinned by the
    // `g_*` callers against `decode.cu`. Graph capture snapshots the arg values.
    #[allow(unsafe_code)]
    unsafe {
        result::launch_kernel(func, grid, block, smem, stream, params)
    }
    .map_err(|e| driver_err("raw graph launch", &e))
}

/// Pre-extracted device pointers for one ternary projection.
#[derive(Clone, Copy)]
struct LinPtrs {
    w: sys::CUdeviceptr,
    sc: sys::CUdeviceptr,
    n: usize,
    k: usize,
    rb: usize,
}

/// Pre-extracted device pointers for one transformer block.
struct LayerPtrs {
    attn_norm: sys::CUdeviceptr,
    attn_sub_norm: Option<sys::CUdeviceptr>,
    ffn_norm: sys::CUdeviceptr,
    ffn_sub_norm: Option<sys::CUdeviceptr>,
    q: LinPtrs,
    k: LinPtrs,
    v: LinPtrs,
    o: LinPtrs,
    gate: LinPtrs,
    up: LinPtrs,
    down: LinPtrs,
    kv_k: sys::CUdeviceptr,
    kv_v: sys::CUdeviceptr,
}

/// Pre-extracted device pointers for the shared dense weights + scratch buffers.
struct GraphPtrs {
    d_x: sys::CUdeviceptr,
    d_normed: sys::CUdeviceptr,
    d_q: sys::CUdeviceptr,
    d_knew: sys::CUdeviceptr,
    d_vnew: sys::CUdeviceptr,
    d_attn: sys::CUdeviceptr,
    d_attn_sn: sys::CUdeviceptr,
    d_proj_out: sys::CUdeviceptr,
    d_gate: sys::CUdeviceptr,
    d_up: sys::CUdeviceptr,
    d_gate_sn: sys::CUdeviceptr,
    d_scores: sys::CUdeviceptr,
    d_logits: sys::CUdeviceptr,
    d_qact: sys::CUdeviceptr,
    d_act_scale: sys::CUdeviceptr,
    d_ctrl: sys::CUdeviceptr,
    d_cos: sys::CUdeviceptr,
    d_sin: sys::CUdeviceptr,
    d_token_embd: sys::CUdeviceptr,
    d_output_norm: sys::CUdeviceptr,
}

/// Raw-loaded PTX modules + `CUfunction` handles for the v0.3.2 graph-captured decode.
/// Raw (not the safe `CudaModule`/`CudaFunction`) because the captured launch needs the
/// `sys::CUfunction` handle, which the safe `CudaFunction` keeps `pub(crate)`. These are
/// a SECOND JIT of the same PTX the backend already loaded (a few MB of extra SASS);
/// the modules are unloaded on drop.
struct RawGraphKernels {
    modules: Vec<sys::CUmodule>,
    embed_g: sys::CUfunction,
    rope_g: sys::CUfunction,
    kv_append: sys::CUfunction,
    attn_g: sys::CUfunction,
    rmsnorm: sys::CUfunction,
    residual: sys::CUfunction,
    relu2: sys::CUfunction,
    lm_head: sys::CUfunction,
    act_quant: sys::CUfunction,
    scale: sys::CUfunction,
    tiled: sys::CUfunction,
}

// SAFETY: the raw `CUmodule`/`CUfunction` are process-valid device handles, used only on
// the owning `CudaDecodeModel`'s single capture stream (never concurrently across
// threads — `CudaGraph` is itself documented not-thread-safe, so the whole graph path is
// single-threaded by construction).
#[allow(unsafe_code)]
unsafe impl Send for RawGraphKernels {}

impl RawGraphKernels {
    fn load(ctx: &Arc<CudaContext>) -> Result<Self, BackendError> {
        ctx.bind_to_thread()
            .map_err(|e| driver_err("raw kernels bind", &e))?;
        let load_mod = |ptx: &str| -> Result<sys::CUmodule, BackendError> {
            let c = CString::new(ptx)
                .map_err(|_| BackendError::InvalidInput("PTX has an interior NUL".into()))?;
            // SAFETY: `c` is a valid NUL-terminated PTX image; `load_data` JIT-compiles it.
            #[allow(unsafe_code)]
            unsafe { result::module::load_data(c.as_ptr() as *const c_void) }
                .map_err(|e| driver_err("raw module load_data", &e))
        };
        let get = |m: sys::CUmodule, name: &str| -> Result<sys::CUfunction, BackendError> {
            let c = CString::new(name)
                .map_err(|_| BackendError::InvalidInput("kernel name has a NUL".into()))?;
            // SAFETY: `m` is a loaded module; `name` is one of its `extern "C"` entry points.
            #[allow(unsafe_code)]
            unsafe { result::module::get_function(m, c) }
                .map_err(|e| driver_err("raw get_function", &e))
        };
        let dm = load_mod(DECODE_PTX)?;
        let am = load_mod(TQ2_0_ADD_PTX)?;
        Ok(Self {
            embed_g: get(dm, KERNEL_NAME_EMBED_G)?,
            rope_g: get(dm, KERNEL_NAME_ROPE_G)?,
            kv_append: get(dm, KERNEL_NAME_KV_APPEND)?,
            attn_g: get(dm, KERNEL_NAME_ATTN_G)?,
            rmsnorm: get(dm, KERNEL_NAME_RMSNORM)?,
            residual: get(dm, KERNEL_NAME_RESIDUAL)?,
            relu2: get(dm, KERNEL_NAME_RELU2_GATE)?,
            lm_head: get(dm, KERNEL_NAME_LM_HEAD)?,
            act_quant: get(dm, KERNEL_NAME_ACT_QUANT_TILED)?,
            scale: get(dm, KERNEL_NAME_SCALE_MUL)?,
            tiled: get(am, KERNEL_NAME_TILED)?,
            modules: vec![dm, am],
        })
    }
}

impl Drop for RawGraphKernels {
    fn drop(&mut self) {
        for &m in &self.modules {
            if !m.is_null() {
                // SAFETY: each module was loaded by `load` and is unloaded exactly once
                // here; the owning model's `graph` field is declared before `raw`, so the
                // graph-exec referencing these functions is destroyed first, and nothing
                // launches the graph after this point.
                #[allow(unsafe_code)]
                unsafe {
                    let _ = result::module::unload(m);
                }
            }
        }
    }
}

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
    #[allow(clippy::too_many_arguments)] // a test driver mirroring the kernel's operands
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
            (64, 40, 2560), // a realistic-ish K (a 32-multiple, below the tiled cap)
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

    /// `residual_add` / `embedding_gather` / `lm_head` must match host bit-for-bit:
    /// the first two are exact (add / copy), the LM head reproduces the host's
    /// sequential dot in k-order (no FMA).
    #[test]
    fn residual_embed_lmhead_bit_match_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping residual/embed/lm_head bit-match: no device ({e})");
                return;
            }
        };
        let mut s = 0xABCD_1234_5678_9876u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };

        // residual_add: x += y (exact).
        {
            let n = 2560usize;
            let x0: Vec<f32> = (0..n).map(|_| next()).collect();
            let y: Vec<f32> = (0..n).map(|_| next()).collect();
            let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| a + b).collect();
            let mut got = x0.clone();
            backend.residual_add(&mut got, &y).expect("residual");
            for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), h.to_bits(), "residual_add mismatch [{i}]");
            }
        }

        // embedding_gather: out = table[tok] (exact copy).
        {
            let (vocab, n_embd) = (64usize, 256usize);
            let table: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
            let tok = 37usize;
            let want = &table[tok * n_embd..tok * n_embd + n_embd];
            let mut got = vec![0.0f32; n_embd];
            backend
                .embedding_gather(&table, tok, n_embd, &mut got)
                .expect("embed");
            for (i, (&g, &h)) in got.iter().zip(want).enumerate() {
                assert_eq!(g.to_bits(), h.to_bits(), "embedding_gather mismatch [{i}]");
            }
        }

        // lm_head: sequential dot, bit-exact.
        {
            let (vocab, n_embd) = (128usize, 2560usize);
            let h: Vec<f32> = (0..n_embd).map(|_| next()).collect();
            let embd: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
            let mut want = vec![0.0f32; vocab];
            for (v, slot) in want.iter_mut().enumerate() {
                let row = &embd[v * n_embd..v * n_embd + n_embd];
                let mut acc = 0.0f32;
                for k in 0..n_embd {
                    acc += h[k] * row[k];
                }
                *slot = acc;
            }
            let mut got = vec![0.0f32; vocab];
            backend
                .lm_head(&h, &embd, n_embd, vocab, &mut got)
                .expect("lm_head");
            for (v, (&g, &hh)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), hh.to_bits(), "lm_head mismatch [{v}]: got {g} want {hh}");
            }
        }
    }

    /// `relu2_gate` must reproduce the host BitNet squared-ReLU FFN gate `r =
    /// g.max(0); g = r*r*u` **bit-for-bit**. The input deliberately straddles zero so
    /// the `max(.,0)` clamp (and the gate's hard zero on negatives) is exercised.
    #[test]
    fn relu2_gate_bit_matches_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping relu2_gate bit-match: no device ({e})");
                return;
            }
        };
        let mut s = 0x51A7_3C9E_2D6B_8F40u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Range [-4, 4): ~half the gate values negative, hitting the ReLU clamp.
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let n = 6912usize; // BitNet 2B4T n_ff
        let gate0: Vec<f32> = (0..n).map(|_| next()).collect();
        let up: Vec<f32> = (0..n).map(|_| next()).collect();
        // Host reference: identical to layers::mlp's gating loop.
        let want: Vec<f32> = gate0
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let r = g.max(0.0);
                r * r * u
            })
            .collect();
        let mut got = gate0.clone();
        backend.relu2_gate(&mut got, &up).expect("relu2_gate");
        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "relu2_gate mismatch [{i}]: got {g} want {h}");
        }
    }

    /// Device GQA attention (M=1 decode) vs host `gqa_attention`. The dots + weighted
    /// sums bit-match; the inline softmax `expf` gives a ≤3-ULP / ~1e-7 divergence, so
    /// this measures the max rel error (reported) and asserts it stays tiny — the
    /// attention output is the only forward op carrying the exp difference.
    #[test]
    fn gqa_attention_decode_matches_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping attention match: no device ({e})");
                return;
            }
        };
        // BitNet 2B4T attention dims; a modest cached context for the decode token.
        let (n_head, n_head_kv, head_dim, ctx) = (20usize, 5usize, 128usize, 96usize);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let limit = ctx - 1; // steady-state decode: all cached keys visible
        let n_rep = n_head / n_head_kv;

        let mut s = 0x0BAD_F00D_1357_2468u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
        };
        let q: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();

        // Host reference — replicates ops::gqa_attention for seq=1.
        let mut want = vec![0.0f32; n_head * head_dim];
        let mut scores = vec![0.0f32; ctx];
        for h in 0..n_head {
            let kv = h / n_rep;
            let q_row = &q[h * head_dim..h * head_dim + head_dim];
            for (j, sc) in scores.iter_mut().enumerate() {
                if j > limit {
                    *sc = f32::NEG_INFINITY;
                    continue;
                }
                let k_row = &k[(j * n_head_kv + kv) * head_dim..][..head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                *sc = dot * scale;
            }
            let mut m = f32::NEG_INFINITY;
            for &sc in &scores {
                if sc > m {
                    m = sc;
                }
            }
            let mut sum = 0.0f32;
            for sc in scores.iter_mut() {
                let e = (*sc - m).exp();
                *sc = e;
                sum += e;
            }
            let inv = 1.0f32 / sum;
            for sc in scores.iter_mut() {
                *sc *= inv;
            }
            let o = &mut want[h * head_dim..h * head_dim + head_dim];
            for (j, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let v_row = &v[(j * n_head_kv + kv) * head_dim..][..head_dim];
                for d in 0..head_dim {
                    o[d] += w * v_row[d];
                }
            }
        }

        let mut got = vec![0.0f32; n_head * head_dim];
        backend
            .gqa_attention_decode(&q, &k, &v, &mut got, ctx, n_head, n_head_kv, head_dim, scale, limit)
            .expect("device attention");

        let (mut max_ulp, mut n_diff, mut max_rel, mut max_abs) = (0i64, 0usize, 0.0f64, 0.0f64);
        for (&g, &h) in got.iter().zip(&want) {
            let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
            if du != 0 {
                n_diff += 1;
            }
            max_ulp = max_ulp.max(du);
            max_abs = max_abs.max(f64::from((g - h).abs()));
            if h != 0.0 {
                max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
            }
        }
        eprintln!(
            "attention device-vs-host: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={max_ulp} n_diff={n_diff}/{}",
            got.len()
        );
        // The dots + weighted sum bit-match; the sole divergence is the softmax `expf`
        // (≤3 ULP, ~1e-6 ABSOLUTE), which inflates to a larger *relative* error only on
        // near-zero (cancellation) outputs. The meaningful metric is the absolute error,
        // which must stay tiny (it propagates into the residual stream as a small add).
        assert!(
            max_abs < 1e-3,
            "device attention absolute error too large (likely a real bug): max_abs={max_abs:.3e}"
        );
    }

    /// `act_quant_tiled` must reproduce `ops::quantize_activation_int8` **bit-for-bit**
    /// (the int8-as-f32 values and the per-token scale), including the zero-row case.
    #[test]
    fn act_quant_tiled_bit_matches_host() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping act_quant bit-match: no device ({e})");
                return;
            }
        };
        fn host_quant(act: &[f32]) -> (Vec<f32>, f32) {
            let mut gamma = 0.0f32;
            for &v in act {
                let a = v.abs();
                if a > gamma {
                    gamma = a;
                }
            }
            if gamma == 0.0 {
                return (vec![0.0; act.len()], 0.0);
            }
            let s = 127.0f32 / gamma;
            (
                act.iter()
                    .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                    .collect(),
                gamma / 127.0,
            )
        }
        for &k in &[2560usize, 6912, 17, 1] {
            let mut s = 0x9999_AAAA_BBBB_CCCCu64 ^ k as u64;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
            };
            let act: Vec<f32> = (0..k).map(|_| next()).collect();
            let (q_want, scale_want) = host_quant(&act);
            let mut q_got = vec![f32::NAN; k];
            let scale_got = backend.act_quant_tiled(&act, &mut q_got).expect("act_quant");
            assert_eq!(scale_got.to_bits(), scale_want.to_bits(), "scale mismatch k={k}");
            for (i, (&g, &h)) in q_got.iter().zip(&q_want).enumerate() {
                assert_eq!(g.to_bits(), h.to_bits(), "act_quant q mismatch k={k} i={i}");
            }
        }
        // Zero row → zeros + zero scale.
        let act = vec![0.0f32; 64];
        let mut q = vec![1.0f32; 64];
        let sc = backend.act_quant_tiled(&act, &mut q).expect("act_quant zero");
        assert_eq!(sc, 0.0);
        assert!(q.iter().all(|&x| x == 0.0), "zero row must quantize to zeros");
    }

    /// The device GEMM chain (`mpgemm_device`: on-device quant → tiled f64 GEMM →
    /// scale fold) must reproduce the host path (`quantize_activation_int8` → tiled
    /// `mpgemm` → `out *= act_scale`) **bit-for-bit** — same quant, same kernel, same
    /// fold, just resident. This is the GEMM half of the device-resident decode.
    #[test]
    fn mpgemm_device_bit_matches_host_path() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping mpgemm_device match: no device ({e})");
                return;
            }
        };
        let (n, k) = (640usize, 2560usize); // BitNet attn_k projection shape
        let shape = GemmShape::new(1, n, k);

        let mut st = 0x1357_9BDF_2468_ACE0u64;
        let trits: Vec<tritium_core::Trit> = (0..n * k)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                tritium_core::Trit::from_i8(((st >> 33) % 3) as i8 - 1).unwrap()
            })
            .collect();
        let packed = pack_tq2_0(&trits, shape);
        let weights = cuda
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .expect("upload");

        let mut sf = 0x2468_ACE0_1357_9BDFu64;
        let mut nf = || {
            sf ^= sf << 13;
            sf ^= sf >> 7;
            sf ^= sf << 17;
            ((sf >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
        };
        let normed: Vec<f32> = (0..k).map(|_| nf()).collect();
        let scales: Vec<f32> = (0..n).map(|_| 0.5 + nf().abs()).collect();

        // Host path: quantize_activation_int8 + tiled mpgemm + per-token fold.
        let (q_host, act_scale) = {
            let mut gamma = 0.0f32;
            for &v in &normed {
                let a = v.abs();
                if a > gamma {
                    gamma = a;
                }
            }
            if gamma == 0.0 {
                (vec![0.0f32; k], 0.0f32)
            } else {
                let s = 127.0f32 / gamma;
                (
                    normed
                        .iter()
                        .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                        .collect::<Vec<_>>(),
                    gamma / 127.0,
                )
            }
        };
        let mut out_host = run_kernel(&cuda, &packed, &q_host, &scales, shape, AddKernel::Tiled);
        for v in out_host.iter_mut() {
            *v *= act_scale;
        }

        // Device chain.
        let mut out_dev = vec![0.0f32; n];
        cuda.mpgemm_device(&normed, weights.as_ref(), &scales, shape, &mut out_dev)
            .expect("mpgemm_device");

        for (i, (&g, &h)) in out_dev.iter().zip(&out_host).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "mpgemm_device mismatch [{i}]: got {g} want {h}");
        }
    }

    /// CUDA-graph capture spike (v0.3.1 W2) — documents a hard cudarc-0.19 limitation.
    ///
    /// Capturing the decode forward into a replayable graph would collapse the ~390
    /// per-token kernel launches into one `graph.launch()`, the biggest remaining decode
    /// win (the launch path is the wall at M=1). But cudarc 0.19's **safe** launch
    /// (`LaunchArgs::launch`) waits on each buffer's read/write `CudaEvent` before the
    /// kernel — and those events were recorded by the pre-capture uploads, so the very
    /// first captured launch trips `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` ("dependency
    /// created on uncaptured work"). RELAXED capture mode does not help (the dependency is
    /// real, not a mode artifact). The raw escape — `result::launch_kernel`, which does no
    /// event tracking — needs the `sys::CUfunction` handle, but cudarc keeps
    /// `CudaFunction::cu_function` `pub(crate)`, so the only way through is a *parallel*
    /// raw-FFI module/function/launch path (load the PTX via `result::module::load_data`,
    /// `get_function`, hand-pack params), bypassing cudarc's safe layer entirely.
    ///
    /// That raw path is the deferred W2 work (it materially expands the `unsafe` surface
    /// of this `#![deny(unsafe_code)]` crate, so it is its own gated change). This test is
    /// `#[ignore]`d: it asserts the limitation still holds, so if a future cudarc makes the
    /// safe launch capture-compatible, this starts passing and flags that the raw path is
    /// no longer needed.
    #[test]
    #[ignore = "cudarc 0.19 safe launch is capture-incompatible; W2 needs the raw-FFI path"]
    fn cuda_graph_capture_blocked_by_cudarc_safe_launch() {
        use cudarc::driver::sys;
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cuda graph spike: no device ({e})");
                return;
            }
        };
        let n = 256usize;
        let x0 = vec![1.0f32; n];
        let y = vec![2.0f32; n];
        let cap = backend.stream.context().new_stream().expect("capture stream");
        let mut d_x = cap.clone_htod(&x0).expect("htod x");
        let d_y = cap.clone_htod(&y).expect("htod y");
        cap.synchronize().expect("sync");

        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin_capture");
        let mut l = cap.launch_builder(&backend.func_residual);
        l.arg(&mut d_x).arg(&d_y).arg(&n_i);
        // SAFETY: `residual_add_f32(float* x, const float* y, int n)`.
        #[allow(unsafe_code)]
        let launched = unsafe { l.launch(cfg) };
        // The capture launch trips STREAM_CAPTURE_ISOLATION on cudarc 0.19. If this ever
        // succeeds, the safe launch became capture-compatible — revisit the raw-FFI plan.
        assert!(
            launched.is_err(),
            "cudarc safe launch unexpectedly captured cleanly — the raw-FFI W2 path may be unnecessary now"
        );
        let _ = cap.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
    }

    /// CUDA-graph **raw-FFI** capture spike (v0.3.2) — the path that works where the
    /// safe launch trips isolation. Pre-extract each buffer's stable `CUdeviceptr`
    /// *before* `begin_capture` (dropping the `SyncOnDrop` guard outside capture), raw-
    /// load the decode PTX for a raw `CUfunction`, then capture two `residual_add_f32`
    /// launches via `result::launch_kernel` (no cudarc event waits → no isolation), and
    /// assert the single graph replay is **bit-identical** to the host reference. This
    /// pins the v0.3.2 mechanic before the full decode forward is captured.
    #[test]
    fn cuda_graph_raw_launch_replay_bit_identical() {
        use cudarc::driver::{DevicePtr, DevicePtrMut, result, sys};
        use std::ffi::{CString, c_void};

        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping raw-graph spike: no device ({e})");
                return;
            }
        };
        let ctx = backend.stream.context().clone();
        ctx.bind_to_thread().expect("bind ctx");

        // Raw-load the decode PTX → a raw CUfunction (the safe CudaFunction hides
        // `cu_function`, so the captured launch needs this raw handle).
        let ptx_c = CString::new(DECODE_PTX).expect("ptx cstring");
        #[allow(unsafe_code)]
        let cu_module =
            unsafe { result::module::load_data(ptx_c.as_ptr() as *const c_void).expect("load_data") };
        let fname = CString::new("residual_add_f32").expect("fn cstring");
        #[allow(unsafe_code)]
        let cu_func =
            unsafe { result::module::get_function(cu_module, fname).expect("get_function") };

        let n = 2560usize;
        let x0 = vec![1.0f32; n];
        let y = vec![2.0f32; n];
        // residual_add applied twice: ((x0 + y) + y), the kernel's single-f32-add order.
        let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| (a + b) + b).collect();

        let cap = ctx.new_stream().expect("capture stream");
        let mut d_x = cap.clone_htod(&x0).expect("htod x");
        let d_y = cap.clone_htod(&y).expect("htod y");
        cap.synchronize().expect("pre-extract sync");

        // Pre-extract stable device pointers; drop the SyncOnDrop guards OUTSIDE capture
        // (their drop records an event, which is forbidden inside a capture).
        let px: sys::CUdeviceptr = {
            let (p, g) = d_x.device_ptr_mut(&cap);
            drop(g);
            p
        };
        let py: sys::CUdeviceptr = {
            let (p, g) = d_y.device_ptr(&cap);
            drop(g);
            p
        };
        cap.synchronize().expect("post-extract sync");

        let n_i = n as i32;
        let grid = ((n as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);

        cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .expect("begin_capture");
        for _ in 0..2 {
            // kernel_params: each entry points to the arg VALUE (a CUdeviceptr for a
            // `float*`, the i32 for `int n`); these locals outlive the launch call, and
            // graph capture snapshots the values into the kernel node.
            let mut params: [*mut c_void; 3] = [
                (&px) as *const sys::CUdeviceptr as *mut c_void,
                (&py) as *const sys::CUdeviceptr as *mut c_void,
                (&n_i) as *const i32 as *mut c_void,
            ];
            // SAFETY: raw `residual_add_f32(float* x, const float* y, int n)`; params in
            // declaration order; `px`/`py` are valid device addresses (extracted above,
            // `d_x`/`d_y` alive for the test), `n_i` matches the buffer length.
            #[allow(unsafe_code)]
            unsafe {
                result::launch_kernel(cu_func, grid, block, 0, cap.cu_stream(), &mut params)
                    .expect("raw capture launch");
            }
        }
        let graph = cap
            .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .expect("end_capture")
            .expect("non-empty graph");

        // d_x is still x0 (capture did not execute). One replay runs both adds.
        graph.launch().expect("graph launch");
        cap.synchronize().expect("post-replay sync");
        let mut got = vec![0.0f32; n];
        cap.memcpy_dtoh(&d_x, &mut got).expect("dtoh");
        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "raw graph replay mismatch [{i}]: got {g} want {h}");
        }

        // The captured graph holds the raw CUfunction; unload only after a final sync.
        cap.synchronize().expect("final sync");
        drop(graph);
        #[allow(unsafe_code)]
        unsafe {
            result::module::unload(cu_module).expect("unload");
        }
    }
}
