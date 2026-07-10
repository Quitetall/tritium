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
use tritium_format::{IMMA_K, IMMA_N, IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

use crate::autotune::{
    CacheKey, CandidateResult, ShapeBucket, TileConfig, cache_dir, tune_or_load,
};
use crate::codegen::{JIT_KERNEL_NAME, compile_imma};

mod consts;
mod kv;

use consts::*;
use kv::*;

/// Map a `cudarc` driver error to a [`BackendError`]. Allocation failures surface
/// as [`BackendError::OutOfMemory`]; everything else is stringified into
/// [`BackendError::Backend`] so the device error text survives.
fn driver_err(context: &str, err: &DriverError) -> BackendError {
    BackendError::Backend(format!("{context}: {err}"))
}

/// [`CudaGraph`] with ownership-transfer `Send`.
///
/// cudarc's `CudaGraph` holds raw `CUgraph`/`CUgraphExec` handles and is not
/// `Send` (it is not thread-SAFE — no concurrent use). Tritium only ever MOVES
/// a decode model (and its captured graph) to a single owning thread and
/// replays it there exclusively (`tritium-serve`'s decode worker) — the same
/// single-owner argument the `unsafe impl Send for RawGraphKernels` documents.
/// Driver handles are context-scoped, not thread-scoped, so crossing threads
/// by ownership transfer is sound. One caveat the soundness argument relies
/// on: cudarc's `CudaGraph::Drop` does not re-bind the context first, so the
/// graph should be dropped on a thread that has touched the context — true
/// for every current holder (the owning thread both replays and drops).
struct SendGraph(CudaGraph);
#[allow(unsafe_code)]
// SAFETY: see the type doc — exclusive single-owner use only; the wrapper is
// never shared (`&SendGraph` never crosses threads while a replay is live).
unsafe impl Send for SendGraph {}

impl std::ops::Deref for SendGraph {
    type Target = CudaGraph;
    fn deref(&self) -> &CudaGraph {
        &self.0
    }
}

/// Run a graph-capture body; on error, TERMINATE the capture (best-effort)
/// before propagating. Without this, a mid-capture failure leaves the capture
/// stream in capture mode and every later operation on it fails with
/// confusing `STREAM_CAPTURE_*` errors (reviewer-accepted deferred item from
/// three capture sites; fixed for all of them here).
fn capture_body<T>(
    s: &Arc<CudaStream>,
    body: impl FnOnce() -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    match body() {
        Ok(v) => Ok(v),
        Err(e) => {
            // End the wedged capture; any partial graph is dropped. The
            // primary error wins over secondary termination errors.
            let _ = s.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            Err(e)
        }
    }
}

/// Opt the warp-attention kernel into `max_ctx * 4` dynamic shared bytes via
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. `set` applies the attribute
/// to the caller's function handle (safe `CudaFunction` or raw `CUfunction` — both
/// launch the same kernel; a fresh JIT of the module needs its own opt-in).
///
/// Set UNCONDITIONALLY, not only above the 48 KiB default cap: the default
/// dynamic budget is 48 KiB *minus the kernel's static shared* (its `__shared__
/// float s_inv` rounds up to 16 B), so a threshold check would leave the
/// boundary `max_ctx` values (12285..=12288) failing at launch. Setting a value
/// at or below the default is legal and free. A device that cannot grant the
/// request (context_length past its opt-in shared limit, ≈ 25K on Ada) surfaces
/// HERE as an actionable model-build error, not a launch failure mid-decode.
fn attn_shared_opt_in(
    max_ctx: usize,
    set: impl FnOnce(i32) -> Result<(), DriverError>,
) -> Result<(), BackendError> {
    let bytes = max_ctx * 4;
    set(bytes as i32).map_err(|e| {
        BackendError::Backend(format!(
            "decode attention needs {bytes} B of dynamic shared memory for \
             context_length={max_ctx} (scores are staged per head-block), which this \
             device rejected: {e}. Reduce the model's context_length or run this model \
             on a GPU with a larger opt-in shared-memory limit."
        ))
    })
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

/// A SALT multi-plane projection, resident in VRAM: the plane-major TQ2_0 planes
/// uploaded once, plus the geometry the `salt_mpgemm_tiled_f32` kernel needs.
///
/// Unlike [`ResidentLinear`] (the W1.58A8 path: int8-quantized activation + a
/// per-channel scale), a SALT linear feeds the kernel the **raw f32 activation**
/// and the per-block scales packed in the planes — the contract
/// [`tritium_format::dequant_salt_row`] defines. Built by [`CudaBackend::upload_salt`]
/// and run by [`CudaBackend::salt_forward`]; the kernel itself is gated bit-for-bit
/// against the dequant reference by `salt_mpgemm_matches_dequant_reference`.
#[derive(Debug)]
pub struct SaltResidentLinear {
    /// Plane-major packed weight `[T, N, row_bytes]`: plane `p`, row `ni` at
    /// `p*N*row_bytes + ni*row_bytes`. Uploaded once; reused across decode steps.
    device: Arc<CudaSlice<u8>>,
    /// Output channels (`N`).
    n: usize,
    /// Contraction dimension (`K`).
    k: usize,
    /// TQ2_0 bytes per plane-row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    row_bytes: usize,
    /// Realized plane count `T` (max over rows; ragged rows zero-padded on upload).
    t_planes: usize,
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
    /// Fused-scaled variant: folds `act_scale` into the epilogue, eliminating the
    /// separate `scale_mul_f32` launch (v0.6.0 opt #15).
    func_tiled_scaled: CudaFunction,
    /// The resolved `salt_mpgemm_tiled_f32` kernel (v0.4.0 SALT multi-plane GEMM),
    /// driven by the resident [`CudaBackend::salt_forward`] path.
    func_salt: CudaFunction,
    /// Sparse-aware f32-tiled kernel: bitmap skip for zero blocks (P1 opt).
    #[allow(dead_code)]
    // validated sparse mpgemm kernel; auto-dispatch integration is future (1.x) work
    func_tiled_f32_sparse: CudaFunction,
    /// Sparse-aware fused-scaled f32-tiled kernel: bitmap skip + act_scale fold.
    #[allow(dead_code)]
    // validated sparse fused-scaled kernel; auto-dispatch integration is future (1.x) work
    func_tiled_i8_scaled_sparse: CudaFunction,
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
    /// The resolved `rmsnorm_quant_f32` decode kernel (fused RMSNorm + int8 act-quant).
    /// Exercised by `rmsnorm_quant_bit_matches_host`; the resident decode path launches
    /// it via its own `f_rmsnorm_quant` handle.
    #[allow(dead_code)]
    func_rmsnorm_quant: CudaFunction,
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
    /// `gqa_attention_split_partial_f32` + `gqa_attention_combine_f32` — the
    /// flash-decoding (split-KV) attention pair (low-N occupancy fix). Resolved here
    /// and exercised by the `attn_split_kv_matches_direct_attention` equivalence gate;
    /// the resident decode wiring (`md_attn`/`gb_attn`) lands next and makes them load-bearing.
    #[allow(dead_code)]
    func_attn_split_partial: CudaFunction,
    #[allow(dead_code)]
    func_attn_combine: CudaFunction,
    /// The resolved `act_quant_tiled_f32` kernel (on-device A8 quant for the tiled GEMM).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    func_act_quant_tiled: CudaFunction,
    /// The resolved `scale_mul_f32` kernel (per-token act-scale fold).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    func_scale_mul: CudaFunction,
    /// The resolved `relu2_gate_f32` kernel (BitNet squared-ReLU FFN gate).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    func_relu2_gate: CudaFunction,
    /// Loaded v0.50 training backward PTX module (`--fmad=false`), kept alive so its
    /// gradient functions stay valid.
    _grad_module: Arc<CudaModule>,
    /// The resolved `ternary_matmul_grad_a/_w/_s` kernels (f32 backward, ADR 0007).
    /// The f32 forward companion to the grad kernels (`Y = s·(A·Wᵀ)`), used by the QAT training
    /// step ([`super::train`]).
    func_train_forward: CudaFunction,
    /// Gradient-checked against the `tritium-train` CPU vjp oracle; wired into the QAT training
    /// step ([`super::train`]).
    func_grad_a: CudaFunction,
    func_grad_w: CudaFunction,
    func_grad_s: CudaFunction,
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
        let func_tiled_scaled = module
            .load_function(KERNEL_NAME_TILED_SCALED)
            .map_err(|e| driver_err("resolve tq2_0_add_tiled_scaled kernel", &e))?;
        let func_salt = module
            .load_function(KERNEL_NAME_SALT)
            .map_err(|e| driver_err("resolve salt_mpgemm kernel", &e))?;
        let func_tiled_f32_sparse = module
            .load_function(KERNEL_NAME_TILED_F32_SPARSE)
            .map_err(|e| driver_err("resolve tq2_0_add_tiled_f32_sparse kernel", &e))?;
        let func_tiled_i8_scaled_sparse = module
            .load_function(KERNEL_NAME_TILED_I8_SCALED_SPARSE)
            .map_err(|e| driver_err("resolve tq2_0_add_tiled_i8_scaled_sparse kernel", &e))?;

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
        let func_rmsnorm_quant = decode_module
            .load_function(KERNEL_NAME_RMSNORM_QUANT)
            .map_err(|e| driver_err("resolve rmsnorm_quant kernel", &e))?;
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
        let func_attn_split_partial = decode_module
            .load_function(KERNEL_NAME_ATTN_SPLIT_PARTIAL)
            .map_err(|e| driver_err("resolve attn split-partial kernel", &e))?;
        let func_attn_combine = decode_module
            .load_function(KERNEL_NAME_ATTN_COMBINE)
            .map_err(|e| driver_err("resolve attn combine kernel", &e))?;
        let func_act_quant_tiled = decode_module
            .load_function(KERNEL_NAME_ACT_QUANT_TILED)
            .map_err(|e| driver_err("resolve act_quant_tiled kernel", &e))?;
        let func_scale_mul = decode_module
            .load_function(KERNEL_NAME_SCALE_MUL)
            .map_err(|e| driver_err("resolve scale_mul kernel", &e))?;
        let func_relu2_gate = decode_module
            .load_function(KERNEL_NAME_RELU2_GATE)
            .map_err(|e| driver_err("resolve relu2_gate kernel", &e))?;

        // The v0.50 training backward kernels (their own `--fmad=false` PTX, compute_75).
        let grad_module = ctx
            .load_module(Ptx::from_src(TRAIN_GRAD_PTX))
            .map_err(|e| driver_err("load train_grad ptx", &e))?;
        let func_train_forward = grad_module
            .load_function(KERNEL_NAME_TRAIN_FWD)
            .map_err(|e| driver_err("resolve train_forward kernel", &e))?;
        let func_grad_a = grad_module
            .load_function(KERNEL_NAME_GRAD_A)
            .map_err(|e| driver_err("resolve grad_a kernel", &e))?;
        let func_grad_w = grad_module
            .load_function(KERNEL_NAME_GRAD_W)
            .map_err(|e| driver_err("resolve grad_w kernel", &e))?;
        let func_grad_s = grad_module
            .load_function(KERNEL_NAME_GRAD_S)
            .map_err(|e| driver_err("resolve grad_s kernel", &e))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_owned());

        // SM arch tag for the autotune cache key (e.g. "sm_89" on the 4090). Read the
        // device's compute capability via the driver attributes; default to the IMMA
        // floor `sm_80` if the query fails (the kernel requires sm_80+ anyway).
        let sm_arch = query_sm_arch(&ctx);
        // CUDA driver version for cache invalidation (e.g. 13030 for 13.3).
        let cuda_version = query_driver_version();
        warn_if_cuda_driver_outside_bound_major(cuda_version);

        Ok(Self {
            stream,
            _module: module,
            _imma_module: imma_module,
            func,
            func_tiled,
            func_tiled_scaled,
            func_salt,
            func_tiled_f32_sparse,
            func_tiled_i8_scaled_sparse,
            func_imma,
            func_act_quant,
            _decode_module: decode_module,
            func_rmsnorm,
            func_rmsnorm_quant,
            func_rope,
            func_softmax,
            func_residual,
            func_embed,
            func_lm_head,
            func_attn,
            func_attn_split_partial,
            func_attn_combine,
            func_act_quant_tiled,
            func_scale_mul,
            func_relu2_gate,
            _grad_module: grad_module,
            func_train_forward,
            func_grad_a,
            func_grad_w,
            func_grad_s,
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

    /// Reject training-backward shapes whose flat-index products would overflow the
    /// device's `int` index arithmetic (the kernels form `m*n`, `n*k`, `m*k` as int32),
    /// which also bounds the `u32` launch grid. A silent `as` truncation would otherwise
    /// give a wrong answer; BitNet shapes are orders of magnitude below this limit.
    fn check_grad_launch_bounds(m: usize, n: usize, k: usize) -> Result<(), BackendError> {
        let lim = i32::MAX as usize;
        if m.checked_mul(n).is_none_or(|v| v > lim)
            || n.checked_mul(k).is_none_or(|v| v > lim)
            || m.checked_mul(k).is_none_or(|v| v > lim)
        {
            return Err(BackendError::InvalidInput(format!(
                "grad shape {m}x{n}x{k}: a flat-index product exceeds i32::MAX (device index overflow)"
            )));
        }
        Ok(())
    }

    // ── v0.50 training: f32 ternary-matmul forward + backward (ADR 0007 Gate C) ──
    // The backward kernels mirror `tritium_train::ops::matmul::vjp` and the forward mirrors
    // `::forward`, gradient-checked against them on the GPU lane. htod → launch (one thread per
    // output, sequential reduction, no atomics) → dtoh. Wired into the QAT training step
    // (`super::train`, plan 0013); a future resident engine keeps these buffers on-device.

    /// Device `Y[m,n] = s[n]·Σ_k A[m,k]·W[n,k]`. `a`:[M,K], `w`:[N,K] (STE-quantized weight as f32),
    /// `s`:[N], `y`:[M,N]. Reproduces [`tritium_train::ops::matmul::forward`] (same reduction order +
    /// `--fmad=false`).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    pub(crate) fn train_forward(
        &self,
        a: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
        y: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if a.len() != m * k || w.len() != n * k || s.len() != n || y.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: y.len(),
            });
        }
        Self::check_grad_launch_bounds(m, n, k)?;
        if m * n == 0 {
            return Ok(());
        }
        if k == 0 {
            // Empty contraction: Y[m,n] = s[n]·0 = 0. Avoid a zero-length htod and launch.
            y.fill(0.0);
            return Ok(());
        }
        let d_a = self
            .stream
            .clone_htod(a)
            .map_err(|e| driver_err("train_forward htod a", &e))?;
        let d_w = self
            .stream
            .clone_htod(w)
            .map_err(|e| driver_err("train_forward htod w", &e))?;
        let d_s = self
            .stream
            .clone_htod(s)
            .map_err(|e| driver_err("train_forward htod s", &e))?;
        let mut d_y = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| driver_err("train_forward alloc", &e))?;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((m * n) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_train_forward);
        launch
            .arg(&d_a)
            .arg(&d_w)
            .arg(&d_s)
            .arg(&mut d_y)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: signature `(const float* a[M,K], const float* w[N,K], const float* s[N],
        // float* y[M,N], int m, int n, int k)`; args pushed in that order; one thread per Y
        // element, guarded by `idx < m*n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch train_forward", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_y, y)
            .map_err(|e| driver_err("train_forward dtoh", &e))?;
        Ok(())
    }

    /// Device `gA[m,k] = Σ_n gy[m,n]·s[n]·W[n,k]`. `gy`:[M,N], `w`:[N,K], `s`:[N], `ga`:[M,K].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    pub(crate) fn grad_a(
        &self,
        gy: &[f32],
        w: &[f32],
        s: &[f32],
        shape: GemmShape,
        ga: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if gy.len() != m * n || w.len() != n * k || s.len() != n || ga.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: ga.len(),
            });
        }
        Self::check_grad_launch_bounds(m, n, k)?;
        if m * k == 0 {
            return Ok(());
        }
        let d_gy = self
            .stream
            .clone_htod(gy)
            .map_err(|e| driver_err("grad_a htod gy", &e))?;
        let d_w = self
            .stream
            .clone_htod(w)
            .map_err(|e| driver_err("grad_a htod w", &e))?;
        let d_s = self
            .stream
            .clone_htod(s)
            .map_err(|e| driver_err("grad_a htod s", &e))?;
        let mut d_ga = self
            .stream
            .alloc_zeros::<f32>(m * k)
            .map_err(|e| driver_err("grad_a alloc", &e))?;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((m * k) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_grad_a);
        launch
            .arg(&d_gy)
            .arg(&d_w)
            .arg(&d_s)
            .arg(&mut d_ga)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: signature `(const float* gy[M,N], const float* w[N,K], const float*
        // s[N], float* ga[M,K], int m, int n, int k)`; args pushed in that order; one
        // thread per gA element, guarded by `idx < m*k`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch grad_a", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_ga, ga)
            .map_err(|e| driver_err("grad_a dtoh", &e))?;
        Ok(())
    }

    /// Device `gW[n,k] = Σ_m gy[m,n]·s[n]·A[m,k]`. `gy`:[M,N], `a`:[M,K], `s`:[N], `gw`:[N,K].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    pub(crate) fn grad_w(
        &self,
        gy: &[f32],
        a: &[f32],
        s: &[f32],
        shape: GemmShape,
        gw: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if gy.len() != m * n || a.len() != m * k || s.len() != n || gw.len() != n * k {
            return Err(BackendError::ShapeMismatch {
                expected: n * k,
                got: gw.len(),
            });
        }
        Self::check_grad_launch_bounds(m, n, k)?;
        if n * k == 0 {
            return Ok(());
        }
        let d_gy = self
            .stream
            .clone_htod(gy)
            .map_err(|e| driver_err("grad_w htod gy", &e))?;
        let d_a = self
            .stream
            .clone_htod(a)
            .map_err(|e| driver_err("grad_w htod a", &e))?;
        let d_s = self
            .stream
            .clone_htod(s)
            .map_err(|e| driver_err("grad_w htod s", &e))?;
        let mut d_gw = self
            .stream
            .alloc_zeros::<f32>(n * k)
            .map_err(|e| driver_err("grad_w alloc", &e))?;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((n * k) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_grad_w);
        launch
            .arg(&d_gy)
            .arg(&d_a)
            .arg(&d_s)
            .arg(&mut d_gw)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: signature `(const float* gy[M,N], const float* a[M,K], const float*
        // s[N], float* gw[N,K], int m, int n, int k)`; args pushed in that order; one
        // thread per gW element, guarded by `idx < n*k`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch grad_w", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_gw, gw)
            .map_err(|e| driver_err("grad_w dtoh", &e))?;
        Ok(())
    }

    /// Device `gs[n] = Σ_m gy[m,n]·(Σ_k A[m,k]·W[n,k])`. `gy`:[M,N], `a`:[M,K], `w`:[N,K], `gs`:[N].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on length mismatch; device errors via cudarc.
    pub(crate) fn grad_s(
        &self,
        gy: &[f32],
        a: &[f32],
        w: &[f32],
        shape: GemmShape,
        gs: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if gy.len() != m * n || a.len() != m * k || w.len() != n * k || gs.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: gs.len(),
            });
        }
        Self::check_grad_launch_bounds(m, n, k)?;
        if n == 0 {
            return Ok(());
        }
        let d_gy = self
            .stream
            .clone_htod(gy)
            .map_err(|e| driver_err("grad_s htod gy", &e))?;
        let d_a = self
            .stream
            .clone_htod(a)
            .map_err(|e| driver_err("grad_s htod a", &e))?;
        let d_w = self
            .stream
            .clone_htod(w)
            .map_err(|e| driver_err("grad_s htod w", &e))?;
        let mut d_gs = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| driver_err("grad_s alloc", &e))?;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_grad_s);
        launch
            .arg(&d_gy)
            .arg(&d_a)
            .arg(&d_w)
            .arg(&mut d_gs)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: signature `(const float* gy[M,N], const float* a[M,K], const float*
        // w[N,K], float* gs[N], int m, int n, int k)`; args pushed in that order; one
        // thread per gs element, guarded by `ni < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch grad_s", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_gs, gs)
            .map_err(|e| driver_err("grad_s dtoh", &e))?;
        Ok(())
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

    /// Device fused RMSNorm + int8 activation-quant (`rmsnorm_quant_f32`) on host
    /// slices. `q_out` receives the int8 values stored as f32 (matching
    /// `act_quant_tiled`); returns the per-tensor activation scale. One block of 256
    /// (a power of two, as the absmax tree reduction requires) with `n*4` dynamic shared
    /// bytes. A standalone regression guard for the fused decode kernel.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on a length mismatch; device errors via cudarc.
    #[allow(dead_code)] // the resident decode path uses its own `f_rmsnorm_quant`; this
    // host-slice wrapper exists for the conformance gate.
    pub(crate) fn rmsnorm_quant(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        q_out: &mut [f32],
    ) -> Result<f32, BackendError> {
        let n = x.len();
        if w.len() != n || q_out.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: w.len().min(q_out.len()),
            });
        }
        if n == 0 {
            return Ok(0.0);
        }
        let d_x = self
            .stream
            .clone_htod(x)
            .map_err(|e| driver_err("rmsnorm_quant htod x", &e))?;
        let d_w = self
            .stream
            .clone_htod(w)
            .map_err(|e| driver_err("rmsnorm_quant htod w", &e))?;
        let mut d_q = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| driver_err("rmsnorm_quant alloc q", &e))?;
        let mut d_scale = self
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| driver_err("rmsnorm_quant alloc scale", &e))?;

        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let mut launch = self.stream.launch_builder(&self.func_rmsnorm_quant);
        launch
            .arg(&d_x)
            .arg(&d_w)
            .arg(&eps)
            .arg(&n_i)
            .arg(&mut d_q)
            .arg(&mut d_scale);

        // SAFETY: kernel signature is `(const float* x, const float* w, const float eps,
        // const int n, float* q_out, float* act_scale)`; args are pushed in that exact
        // order and type. `d_x`/`d_w`/`d_q` are length-`n` device buffers, `d_scale` a
        // single f32, `eps`/`n_i` scalars by value. One block of 256; `n*4` dynamic shared
        // bytes hold `s_x`; the kernel bounds its loops by `n`, so no thread reads past
        // any buffer.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch rmsnorm_quant", &e))?;
        }
        self.stream
            .memcpy_dtoh(&d_q, q_out)
            .map_err(|e| driver_err("rmsnorm_quant dtoh q", &e))?;
        let mut scale = [0.0f32; 1];
        self.stream
            .memcpy_dtoh(&d_scale, &mut scale)
            .map_err(|e| driver_err("rmsnorm_quant dtoh scale", &e))?;
        Ok(scale[0])
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

    /// Flash-decoding (split-KV) attention over ONE sequence (`n=1`), for the
    /// equivalence gate: contract `q [n_head*head_dim]` against `k`/`v`
    /// `[ctx, n_head_kv, head_dim]` via the `gqa_attention_split_partial_f32` +
    /// `gqa_attention_combine_f32` pair (`n_split` chunks of `chunk` keys), returning
    /// `out [n_head*head_dim]`. Must match [`Self::gqa_attention_decode`] within tol.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn attn_split_dense(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        ctx: usize,
        n_split: usize,
        chunk: usize,
    ) -> Result<Vec<f32>, BackendError> {
        // head_dim/32 must fit the kernel's per-lane acc[SPLIT_MAX_HD_PER_LANE=8].
        assert!(
            head_dim <= 256,
            "split-KV attention: head_dim={head_dim} > 256 overflows acc[8] in the partial kernel"
        );
        let q_len = n_head * head_dim;
        let d_q = self
            .stream
            .clone_htod(q)
            .map_err(|e| driver_err("split htod q", &e))?;
        let d_k = self
            .stream
            .clone_htod(k)
            .map_err(|e| driver_err("split htod k", &e))?;
        let d_v = self
            .stream
            .clone_htod(v)
            .map_err(|e| driver_err("split htod v", &e))?;
        let positions = vec![(ctx - 1) as i32];
        let d_pos = self
            .stream
            .clone_htod(&positions)
            .map_err(|e| driver_err("split htod pos", &e))?;
        let mut d_part = self
            .stream
            .alloc_zeros::<f32>(n_head * n_split * (head_dim + 2))
            .map_err(|e| driver_err("split alloc partials", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(q_len)
            .map_err(|e| driver_err("split alloc out", &e))?;

        let (max_ctx_i, nh_i, nhkv_i, hd_i) =
            (ctx as i32, n_head as i32, n_head_kv as i32, head_dim as i32);
        let (n_i, ns_i, chunk_i) = (1i32, n_split as i32, chunk as i32);

        // Partial: one warp (32 threads) per (row, head, split).
        let cfg_p = LaunchConfig {
            grid_dim: ((n_head * n_split) as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lp = self.stream.launch_builder(&self.func_attn_split_partial);
        lp.arg(&d_q)
            .arg(&d_k)
            .arg(&d_v)
            .arg(&mut d_part)
            .arg(&d_pos)
            .arg(&max_ctx_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&n_i)
            .arg(&ns_i)
            .arg(&chunk_i);
        // SAFETY: matches `gqa_attention_split_partial_f32(q, k, v, partials, positions,
        // max_ctx, n_head, n_head_kv, head_dim, scale, n, n_split, chunk)` — order/types
        // exact; only `partials` mutable. q is n_head*head_dim, k/v are ctx*n_head_kv*head_dim
        // (max_ctx=ctx), partials is n_head*n_split*(head_dim+2); grid covers n_head*n_split
        // warps with the in-kernel `warp >= n*n_head*n_split` guard.
        #[allow(unsafe_code)]
        unsafe {
            lp.launch(cfg_p)
                .map_err(|e| driver_err("launch split partial", &e))?;
        }

        // Combine: one warp per (row, head).
        let cfg_c = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lc = self.stream.launch_builder(&self.func_attn_combine);
        lc.arg(&d_part)
            .arg(&mut d_out)
            .arg(&nh_i)
            .arg(&hd_i)
            .arg(&n_i)
            .arg(&ns_i);
        // SAFETY: matches `gqa_attention_combine_f32(partials, out, n_head, head_dim, n,
        // n_split)`; only `out` mutable; out is n_head*head_dim; grid covers n_head warps.
        #[allow(unsafe_code)]
        unsafe {
            lc.launch(cfg_c)
                .map_err(|e| driver_err("launch split combine", &e))?;
        }

        let mut out = vec![0.0f32; q_len];
        self.stream
            .memcpy_dtoh(&d_out, &mut out)
            .map_err(|e| driver_err("split dtoh out", &e))?;
        Ok(out)
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
            l.arg(&d_normed)
                .arg(&k_i)
                .arg(&mut d_q)
                .arg(&mut d_act_scale);
            // SAFETY: `act_quant_tiled_f32(const float* act, int k, float* q, float*
            // scale)`; args in order; `d_normed`/`d_q` length `k`, `d_act_scale` 1.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch gemm_dev quant", &e))?;
            }
        }
        // 2. Tiled add-only f64 GEMM (M=1) with fused act_scale fold.
        //    Epilogue: out[mi,ni] = acc * scales[ni] * act_scale[mi].
        //    Eliminates the separate scale_mul_f32 launch + its memory pass.
        {
            let grid_n = (n as u32).div_ceil(WARPS_PER_BLOCK);
            let cfg = LaunchConfig {
                grid_dim: (grid_n, 1, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: (k * 4) as u32,
            };
            let mut l = self.stream.launch_builder(&self.func_tiled_scaled);
            l.arg(&d_q)
                .arg(buf.device.as_ref())
                .arg(&d_scales)
                .arg(&d_act_scale)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled_scaled(act, weights, scales, act_scale,
            // out, m, n, k, row_bytes)`; args in that order.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch gemm_dev tiled scaled", &e))?;
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
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            vocab,
            max_ctx,
            rope_theta,
            rms_eps,
            ..
        } = *spec;
        let q_width = n_head * head_dim;
        let kv_width = n_head_kv * head_dim;
        // ADR 0020 rung 1: f16 KV opt-in. The f16 attention twins are float4-
        // shaped (half4 loads), so the same geometry gate as the split
        // attention applies.
        let kv_dtype = kv_dtype_from_env()?;
        if kv_dtype != KvDtype::F32 && !head_dim.is_multiple_of(4) {
            return Err(BackendError::InvalidInput(format!(
                "TRITIUM_KV={kv_dtype:?} requires head_dim % 4 == 0 (got {head_dim})"
            )));
        }
        if kv_dtype.has_scales() && !head_dim.is_multiple_of(KV_QGROUP) {
            return Err(BackendError::InvalidInput(format!(
                "TRITIUM_KV=i8 requires head_dim % {KV_QGROUP} == 0 (got {head_dim})"
            )));
        }
        if kv_dtype.has_scales() && kv_width > 64 * KV_QGROUP {
            // kv_quant_row_q8's shared absmax array holds 64 groups, and the
            // fused-rope twin's dynamic shared (2·kv_width·4 B) must stay
            // under the 48 KiB default the raw handle never opts past.
            return Err(BackendError::InvalidInput(format!(
                "TRITIUM_KV=i8 supports kv_width <= {} (got {kv_width})",
                64 * KV_QGROUP
            )));
        }
        let kv_elem = kv_dtype.elem();
        let half = head_dim / 2;

        // Validate the dense shapes the kernels assume.
        // The i8 dp4a GEMMs read activation rows (`qact + mi*k`) as 4-byte words,
        // so every dimension that can be a batched row stride must be a multiple
        // of 4 — otherwise odd rows misalign and the first prefill/batch decode
        // dies with a cryptic CUDA misaligned-address fault. Reject at build.
        for (dim, name) in [(n_embd, "n_embd"), (q_width, "q_width"), (n_ff, "n_ff")] {
            if dim % 4 != 0 {
                return Err(BackendError::InvalidInput(format!(
                    "decode model {name}={dim} is not a multiple of 4; the int8 \
                     dp4a GEMMs require 4-byte-aligned activation rows"
                )));
            }
        }
        // RoPE (and the fused rope+kv graph kernel) rotates (j, j+half) pairs;
        // an odd head_dim would leave the last arena element unwritten. The CPU
        // op rejects it too — reject here so the CUDA path can't load one.
        if head_dim % 2 != 0 {
            return Err(BackendError::InvalidInput(format!(
                "decode model head_dim={head_dim} must be even (RoPE pairs)"
            )));
        }
        if spec.token_embd.len() != vocab * n_embd {
            return Err(BackendError::ShapeMismatch {
                expected: vocab * n_embd,
                got: spec.token_embd.len(),
            });
        }
        if spec.output_norm.len() != n_embd {
            return Err(BackendError::ShapeMismatch {
                expected: n_embd,
                got: spec.output_norm.len(),
            });
        }

        // RoPE table: cos/sin for every (pos, lane), computed exactly as the host
        // `ops::rope_apply` (f64 inv_freq + sin_cos, cast to f32), so the device
        // rotation bit-matches. `inv_freq[j] = theta^(-2j/head_dim)`.
        let theta = f64::from(rope_theta);
        let inv_head_dim = 1.0f64 / head_dim as f64;
        let inv_freq: Vec<f64> = (0..half)
            .map(|j| theta.powf(-2.0 * j as f64 * inv_head_dim))
            .collect();
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
        // f16 copy for the graph LM head. `token_embd` is f16 in the GGUF (widened to f32
        // losslessly on load), so `f16::from_f32` round-trips exactly — the LM head reads
        // identical values from half the bytes.
        let embd_f16: Vec<u16> = spec
            .token_embd
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let d_token_embd_f16 = s
            .clone_htod(&embd_f16)
            .map_err(|e| driver_err("decode token_embd f16 htod", &e))?;
        let d_output_norm = upload(spec.output_norm, "decode output_norm htod")?;
        let d_cos = upload(&cos_t, "decode rope cos htod")?;
        let d_sin = upload(&sin_t, "decode rope sin htod")?;

        // Per-layer weights + KV arenas.
        let mut layers = Vec::with_capacity(spec.layers.len());
        let mut kv_k = Vec::with_capacity(spec.layers.len());
        let mut kv_k_scales = Vec::new();
        let mut kv_v_scales = Vec::new();
        let mut kv_v = Vec::with_capacity(spec.layers.len());
        for ls in &spec.layers {
            let opt_norm = |w: &[f32],
                            width: usize,
                            what: &str|
             -> Result<Option<CudaSlice<f32>>, BackendError> {
                if w.is_empty() {
                    Ok(None)
                } else if w.len() == width {
                    Ok(Some(upload(w, what)?))
                } else {
                    Err(BackendError::ShapeMismatch {
                        expected: width,
                        got: w.len(),
                    })
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
                qkv: ResidentLinear::build_fused(s, &[&ls.q, &ls.k, &ls.v])?,
                gateup: ResidentLinear::build_fused(s, &[&ls.gate, &ls.up])?,
            });
            kv_k.push(
                s.alloc_zeros::<u8>(max_ctx * kv_width * kv_elem)
                    .map_err(|e| driver_err("decode kv_k alloc", &e))?,
            );
            kv_v.push(
                s.alloc_zeros::<u8>(max_ctx * kv_width * kv_elem)
                    .map_err(|e| driver_err("decode kv_v alloc", &e))?,
            );
            if kv_dtype.has_scales() {
                let n_scales = max_ctx * n_head_kv * (head_dim / KV_QGROUP);
                kv_k_scales.push(
                    s.alloc_zeros::<f32>(n_scales)
                        .map_err(|e| driver_err("decode kv_k scales alloc", &e))?,
                );
                kv_v_scales.push(
                    s.alloc_zeros::<f32>(n_scales)
                        .map_err(|e| driver_err("decode kv_v scales alloc", &e))?,
                );
            }
        }

        // Resolve the kernels from the resident modules (decode + the add module's tiled GEMM).
        let f = |m: &Arc<CudaModule>, name: &str| -> Result<CudaFunction, BackendError> {
            m.load_function(name)
                .map_err(|e| driver_err("resolve decode kernel", &e))
        };
        let dm = &self._decode_module;

        // v0.50 opt: warp-parallel attention (1 warp/head, 32 threads) instead of
        // single-thread-per-head. The warp kernel is already loaded by the graph path.
        // It stages `max_ctx` f32 scores in dynamic shared memory; above the 48 KiB
        // default per-launch cap that needs an explicit opt-in, and past the device
        // limit the model is rejected HERE with an actionable error instead of a
        // cryptic launch failure on the first decode step.
        let f_attn = f(dm, KERNEL_NAME_ATTN_WARP)?;
        attn_shared_opt_in(max_ctx, |bytes| {
            f_attn.set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        })?;
        // v1.x split attention: the reduce kernel stages the same max_ctx scores in
        // dynamic shared, so it needs the identical opt-in on its own handle.
        let sel =
            |f32_name: &'static str, h_name: &'static str, q8_name: &'static str| match kv_dtype {
                KvDtype::F32 => f32_name,
                KvDtype::F16 => h_name,
                // T2 rides the i8 lattice: every CONSUMER uses the q8 kernels;
                // only the append selections below override to the _t2 names.
                KvDtype::I8 | KvDtype::T2 => q8_name,
            };
        let f_attn_scores = f(
            dm,
            sel(
                KERNEL_NAME_ATTN_SCORES,
                KERNEL_NAME_ATTN_SCORES_H,
                KERNEL_NAME_ATTN_SCORES_Q8,
            ),
        )?;
        let f_attn_reduce = f(
            dm,
            sel(
                KERNEL_NAME_ATTN_REDUCE,
                KERNEL_NAME_ATTN_REDUCE_H,
                KERNEL_NAME_ATTN_REDUCE_Q8,
            ),
        )?;
        attn_shared_opt_in(max_ctx, |bytes| {
            f_attn_reduce.set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        })?;
        let f_attn_tree_scores = f(
            dm,
            sel(
                KERNEL_NAME_ATTN_TREE_SCORES,
                KERNEL_NAME_ATTN_TREE_SCORES_H,
                KERNEL_NAME_ATTN_TREE_SCORES_Q8,
            ),
        )?;
        let f_attn_tree_reduce = f(
            dm,
            sel(
                KERNEL_NAME_ATTN_TREE_REDUCE,
                KERNEL_NAME_ATTN_TREE_REDUCE_H,
                KERNEL_NAME_ATTN_TREE_REDUCE_Q8,
            ),
        )?;
        attn_shared_opt_in(max_ctx, |bytes| {
            f_attn_tree_reduce.set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        })?;

        Ok(CudaDecodeModel {
            stream: Arc::clone(&self.stream),
            f_attn_scores,
            f_attn_reduce,
            tree_scratch: None,
            tree_graphs: None,
            pending_tree: None,
            kv_elem,
            kv_dtype,
            kv_k_scales,
            kv_v_scales,
            f_rmsnorm: f(dm, KERNEL_NAME_RMSNORM)?,
            f_rope: f(dm, KERNEL_NAME_ROPE)?,
            f_attn,
            f_residual: f(dm, KERNEL_NAME_RESIDUAL)?,
            f_embed: f(dm, KERNEL_NAME_EMBED)?,
            f_lm_head: f(dm, KERNEL_NAME_LM_HEAD)?,
            f_relu2: f(dm, KERNEL_NAME_RELU2_GATE)?,
            // v1.x: decode-path quants emit PACKED INT8 for the dp4a GEMMs; the
            // f32 quant kernels remain for the public host-facing helpers.
            f_quant: f(dm, KERNEL_NAME_ACT_QUANT_TILED_I8)?,
            f_rmsnorm_quant: f(dm, KERNEL_NAME_RMSNORM_QUANT_I8)?,
            // v0.50 opt: f32-accumulate tiled GEMM (same kernel the graph path uses).
            // f64 is 1/64-rate on the 4090 — the original decode bottleneck. The f32
            // variant stays within the 1e-4 conformance bar (graph path already gated).
            f_tiled: f(&self._module, KERNEL_NAME_TILED_F32)?,
            f_scale: f(dm, KERNEL_NAME_SCALE_MUL)?,
            f_tiled_scaled: f(&self._module, KERNEL_NAME_TILED_I8_SCALED)?,
            f_tiled_scaled_residual: f(&self._module, KERNEL_NAME_TILED_I8_SCALED_RESIDUAL)?,
            f_rmsnorm_batch: f(dm, KERNEL_NAME_RMSNORM_BATCH)?,
            f_embed_batch: f(dm, KERNEL_NAME_EMBED_BATCH)?,
            f_rope_batch: f(dm, KERNEL_NAME_ROPE_BATCH)?,
            f_quant_batch: f(dm, KERNEL_NAME_ACT_QUANT_BATCH_I8)?,
            f_scale_batch: f(dm, KERNEL_NAME_SCALE_BATCH)?,
            f_kv_append_batch: f(
                dm,
                if kv_dtype == KvDtype::T2 {
                    KERNEL_NAME_KV_APPEND_BATCH_T2
                } else {
                    sel(
                        KERNEL_NAME_KV_APPEND_BATCH,
                        KERNEL_NAME_KV_APPEND_BATCH_H,
                        KERNEL_NAME_KV_APPEND_BATCH_Q8,
                    )
                },
            )?,
            f_attn_batch: f(
                dm,
                sel(
                    KERNEL_NAME_ATTN_BATCH,
                    KERNEL_NAME_ATTN_BATCH_H,
                    KERNEL_NAME_ATTN_BATCH_Q8,
                ),
            )?,
            f_attn_tree: f(dm, KERNEL_NAME_ATTN_TREE)?,
            f_attn_tree_scores,
            f_attn_tree_reduce,
            f_argmax_partial: f(dm, KERNEL_NAME_ARGMAX_PARTIAL)?,
            f_argmax_combine: f(dm, KERNEL_NAME_ARGMAX_COMBINE)?,
            f_lm_head_tiled: f(dm, KERNEL_NAME_LM_HEAD_TILED_F16)?,
            f_lm_head_f16: f(dm, KERNEL_NAME_LM_HEAD_WARP_F16)?,
            f_kv_append_mdecode: f(dm, KERNEL_NAME_KV_APPEND_MDECODE)?,
            f_attn_mdecode: f(dm, KERNEL_NAME_ATTN_MDECODE)?,
            f_attn_split_partial: f(dm, KERNEL_NAME_ATTN_SPLIT_PARTIAL)?,
            f_attn_combine: f(dm, KERNEL_NAME_ATTN_COMBINE)?,
            d_token_embd,
            d_token_embd_f16,
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
            d_qkv: alloc(q_width + 2 * kv_width, "decode d_qkv")?,
            d_gateup: alloc(2 * n_ff, "decode d_gateup")?,
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
            // v1.x: packed int8 A8 activations for the dp4a GEMMs (4× smaller than
            // the old int8-as-f32 buffer; the GEMMs read it as 4-byte words).
            d_qact: s
                .alloc_zeros::<i8>(TILED_K_MAX.min(n_ff.max(q_width).max(n_embd)))
                .map_err(|e| driver_err("decode d_qact", &e))?,
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
            batch_raw: None,
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

    /// Sparse-aware tiled mpGEMM: same as [`mpgemm_kernel`] with `AddKernel::Tiled`
    /// but passes a pre-computed zero-block bitmap to the sparse kernel variant,
    /// allowing it to skip all-zero 256-trit blocks entirely.
    ///
    /// `bitmap` is the output of [`tritium_format::compute_zero_bitmaps`] — one
    /// u32 per 32 blocks, per row, concatenated in row order. `words_per_row` is
    /// `ceil(num_blocks(k) / 32)`.
    ///
    /// Falls back to the non-sparse tiled kernel if `bitmap` is empty.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // sparse mpgemm host path; test-exercised, not yet in auto-dispatch (1.x)
    fn mpgemm_kernel_with_bitmap(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        bitmap: &[u32],
        words_per_row: usize,
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
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

        if bitmap.is_empty() {
            return self.mpgemm_kernel(act, weights, scales, shape, format, out, AddKernel::Tiled);
        }

        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let d_scales = self
            .stream
            .clone_htod(scales)
            .map_err(|e| alloc_or_backend("upload scales (htod)", &e, scales.len() * 4))?;
        let d_bitmap = self
            .stream
            .clone_htod(bitmap)
            .map_err(|e| alloc_or_backend("upload bitmap (htod)", &e, bitmap.len() * 4))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        debug_assert!(
            k <= TILED_K_MAX,
            "sparse tiled kernel K={k} exceeds the {TILED_K_MAX} shared-mem cap"
        );
        let grid_n = (n as u32).div_ceil(WARPS_PER_BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (grid_n, m as u32, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: (k * 4) as u32,
        };

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let row_bytes_i = row_bytes as i32;
        let wpr_i = words_per_row as i32;

        let mut launch = self.stream.launch_builder(&self.func_tiled_f32_sparse);
        launch
            .arg(&d_act)
            .arg(buf.device.as_ref())
            .arg(&d_scales)
            .arg(&d_bitmap)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&row_bytes_i)
            .arg(&wpr_i);

        // SAFETY: args match the `tq2_0_add_*_sparse` kernel signature `(const float*
        // act, const u8* weights, const float* scales, const u32* bitmap, float* out,
        // int m, int n, int k, int row_bytes, int words_per_row)`, pushed in that exact
        // order/type. `d_act` is `[m,k]`, the weight buffer is the uploaded `[n,k]` TQ2_0
        // rows, `d_scales` is `[n]`, `d_bitmap` is `[n*words_per_row]`, `d_out` is the
        // `[m,n]` output. The grid is `(grid_n, m)`, `k*4` dynamic shared bytes; the
        // kernel bounds every loop by the int dims, so no thread reads past a buffer.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch tq2_0_add_sparse", &e))?;
        }

        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }

    /// Upload a SALT tensor's rows into VRAM as one plane-major buffer, ready for
    /// [`Self::salt_forward`]. Each [`SaltRow`] contributes its `T` planes; a row
    /// with fewer planes than the tensor max is zero-padded (a zeroed TQ2_0 plane
    /// dequantizes to 0, so it is a no-op in the accumulate). The weight is uploaded
    /// once and reused across decode steps — this is the resident-decode wiring of
    /// the SALT kernel, vs the test-only [`Self::salt_mpgemm_dense`] which re-uploads.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if `rows.len() != n` or a row's `k` disagrees;
    /// [`BackendError::WrongBlockLen`]-style [`BackendError::InvalidInput`] if a plane
    /// is the wrong byte length or `k` exceeds the tiled shared-memory cap.
    pub fn upload_salt(
        &self,
        rows: &[tritium_format::SaltRow],
        n: usize,
        k: usize,
    ) -> Result<SaltResidentLinear, BackendError> {
        if rows.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: rows.len(),
            });
        }
        if k > TILED_K_MAX {
            return Err(BackendError::InvalidInput(format!(
                "SALT decode K={k} exceeds the tiled cap {TILED_K_MAX}"
            )));
        }
        let row_bytes = num_blocks(k) * TQ2_0_BLOCK_BYTES;
        let t_planes = rows.iter().map(|r| r.planes.len()).max().unwrap_or(0);
        for r in rows {
            if r.k != k {
                return Err(BackendError::ShapeMismatch {
                    expected: k,
                    got: r.k,
                });
            }
            for plane in &r.planes {
                if plane.len() != row_bytes {
                    return Err(BackendError::InvalidInput(format!(
                        "SALT plane is {} bytes, expected {row_bytes}",
                        plane.len()
                    )));
                }
            }
        }

        // Plane-major assembly: plane p, then row ni; ragged rows zero-padded.
        let zero_plane = vec![0u8; row_bytes];
        let mut weights = Vec::with_capacity(t_planes * n * row_bytes);
        for p in 0..t_planes {
            for r in rows {
                match r.planes.get(p) {
                    Some(plane) => weights.extend_from_slice(plane),
                    None => weights.extend_from_slice(&zero_plane),
                }
            }
        }
        let device = self
            .stream
            .clone_htod(&weights)
            .map_err(|e| driver_err("htod salt resident weights", &e))?;
        Ok(SaltResidentLinear {
            device: Arc::new(device),
            n,
            k,
            row_bytes,
            t_planes,
        })
    }

    /// Run a resident SALT projection: contract `act` `[M, K]` against the
    /// VRAM-resident plane-major planes of `lin`, returning `[M, N]` row-major.
    /// Same launch as [`Self::salt_mpgemm_dense`] but against the pre-uploaded
    /// weight — no per-call H2D of the planes.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if `act.len() != m * lin.k`; a driver error
    /// from the allocation, launch, or readback otherwise.
    pub fn salt_forward(
        &self,
        lin: &SaltResidentLinear,
        act: &[f32],
        m: usize,
    ) -> Result<Vec<f32>, BackendError> {
        if act.len() != m * lin.k {
            return Err(BackendError::ShapeMismatch {
                expected: m * lin.k,
                got: act.len(),
            });
        }
        let (n, k) = (lin.n, lin.k);
        if m == 0 || n == 0 {
            return Ok(vec![0.0f32; m * n]);
        }
        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| driver_err("htod salt act", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| driver_err("alloc salt out", &e))?;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), m as u32, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: (k * 4) as u32,
        };
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let rb_i = lin.row_bytes as i32;
        let tp_i = lin.t_planes as i32;
        let plane_stride = (n * lin.row_bytes) as i64;

        let mut launch = self.stream.launch_builder(&self.func_salt);
        launch
            .arg(&d_act)
            .arg(lin.device.as_ref())
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&rb_i)
            .arg(&tp_i)
            .arg(&plane_stride);
        // SAFETY: `salt_mpgemm_tiled_f32(const float*, const unsigned char*, float*, int,
        // int, int, int, int, long long)` — args pushed in that order/type; only `d_out`
        // is mutable. `d_act` is `M*K`, the resident weight is `T*N*row_bytes` (built by
        // `upload_salt`), `d_out` is `M*N`; the grid covers `M` rows × `ceil(N/WARPS_PER_BLOCK)`
        // warp-columns with in-kernel `mi>=m`/`ni>=n` guards, and `K*4` shared matches the
        // kernel's `extern __shared__ float[K]`. Host scalars outlive the launch.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch salt resident", &e))?;
        }
        let mut out = vec![0.0f32; m * n];
        self.stream
            .memcpy_dtoh(&d_out, &mut out)
            .map_err(|e| driver_err("dtoh salt resident out", &e))?;
        Ok(out)
    }

    /// SALT multi-plane dense GEMM (v0.4.0 P1): contract `act` `[M, K]` against
    /// `t_planes` stacked TQ2_0 planes (`weights`, **plane-major** — plane `p`, row
    /// `ni` at `p*N*row_bytes + ni*row_bytes`), reading each block's f16 scale and
    /// summing `Σ_p scale_p·tmatmul(t_p)`. Matches [`tritium_format::dequant_salt_row`]
    /// → fp32 matmul within 1e-4. This is the kernel-correctness entry point;
    /// [`Self::salt_forward`] wires the same launch into the resident decode path.
    #[cfg(test)]
    fn salt_mpgemm_dense(
        &self,
        act: &[f32],
        weights: &[u8],
        m: usize,
        n: usize,
        k: usize,
        t_planes: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let row_bytes = num_blocks(k) * TQ2_0_BLOCK_BYTES;
        assert_eq!(
            weights.len(),
            t_planes * n * row_bytes,
            "weights len mismatch"
        );
        assert_eq!(act.len(), m * k, "act len mismatch");

        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| driver_err("htod act", &e))?;
        let d_w = self
            .stream
            .clone_htod(weights)
            .map_err(|e| driver_err("htod salt weights", &e))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| driver_err("alloc salt out", &e))?;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), m as u32, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: (k * 4) as u32,
        };
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let rb_i = row_bytes as i32;
        let tp_i = t_planes as i32;
        let plane_stride = (n * row_bytes) as i64;

        let mut launch = self.stream.launch_builder(&self.func_salt);
        launch
            .arg(&d_act)
            .arg(&d_w)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&rb_i)
            .arg(&tp_i)
            .arg(&plane_stride);
        // SAFETY: `salt_mpgemm_tiled_f32` declares (const float*, const unsigned char*,
        // float*, int, int, int, int, int, long long) — pushed here in that exact order
        // and type. Only `d_out` is mutable. Buffers are sized `M*K` / `T*N*row_bytes` /
        // `M*N` above; the grid covers `M` rows × `ceil(N/WARPS_PER_BLOCK)` warp-columns
        // with in-kernel `mi>=m`/`ni>=n` guards, and the `K*4` shared request matches the
        // kernel's `extern __shared__ float[K]`. All host scalars outlive the launch.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch salt_mpgemm", &e))?;
        }
        let mut out = vec![0.0f32; m * n];
        self.stream
            .memcpy_dtoh(&d_out, &mut out)
            .map_err(|e| driver_err("dtoh salt out", &e))?;
        Ok(out)
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
            &func,
            tile,
            &d_qact,
            buf.device.as_ref(),
            &d_act_scale,
            &d_wscale,
            &mut d_out,
            m_i,
            n_i,
            k_i,
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
        if let Some(t) = self
            .tuned_tiles
            .lock()
            .expect("tuned_tiles poisoned")
            .get(&key)
        {
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
            Err(_) => CandidateResult {
                correct: false,
                seconds: f64::INFINITY,
            },
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
            &func,
            tile,
            &d_qact,
            &d_weights,
            &d_act_scale,
            &d_wscale,
            &mut d_out,
            m as i32,
            n as i32,
            k as i32,
            num_ktiles_i,
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
            return Ok(CandidateResult {
                correct: false,
                seconds: f64::INFINITY,
            });
        }

        // Time a few repetitions and take the MINIMUM: external contention (a
        // desktop compositor, another process) only ever inflates a launch's
        // wall-clock, so min-of-N is the noise-robust estimator of a tile's true
        // cost — a median can crown the wrong winner on a busy GPU and the cache
        // then pins that mistake. The gate only requires that the *winner* be
        // correct, not that timings be reproducible to the nanosecond.
        const REPS: usize = 8;
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let start = std::time::Instant::now();
            self.launch_imma_tile(
                &func,
                tile,
                &d_qact,
                &d_weights,
                &d_act_scale,
                &d_wscale,
                &mut d_out,
                m as i32,
                n as i32,
                k as i32,
                num_ktiles_i,
            )?;
            self.stream
                .synchronize()
                .map_err(|e| driver_err("autotune timing sync", &e))?;
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let seconds = times[0];
        Ok(CandidateResult {
            correct: true,
            seconds,
        })
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
const CUDARC_BINDING_CUDA_MAJOR: u32 = 13;

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

fn cuda_driver_major(version: u32) -> Option<u32> {
    if version == 0 {
        None
    } else {
        Some(version / 1000)
    }
}

fn warn_if_cuda_driver_outside_bound_major(version: u32) {
    let Some(major) = cuda_driver_major(version) else {
        eprintln!(
            "tritium-cuda: warning: could not query CUDA driver version; cudarc is bound to cuda-13020"
        );
        return;
    };
    if major != CUDARC_BINDING_CUDA_MAJOR {
        eprintln!(
            "tritium-cuda: warning: CUDA driver version {version} is outside CUDA {CUDARC_BINDING_CUDA_MAJOR}.x; cudarc is bound to cuda-13020, so driver ABI drift may surface as launch/load errors"
        );
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
    fn as_concrete(&self) -> Option<&dyn Any> {
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

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format,
            out,
        } = p;
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
    fn mpgemm_with_act_quant(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales: weight_scales,
            shape,
            format,
            out,
        } = p;
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
            _ => {
                self.default_mpgemm_with_act_quant(act, weights, weight_scales, shape, format, out)
            }
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

        self.mpgemm(MpGemm {
            act: &q,
            weights,
            scales: weight_scales,
            shape,
            format,
            out,
        })?;

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
fn quantize_act_int8_host(
    act: &[f32],
    m: usize,
    k: usize,
    q_out: &mut [f32],
    scale_out: &mut [f32],
) {
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
                let trit = if gn < n && gk < k {
                    trits[gn * k + gk]
                } else {
                    0
                };
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
            .ok_or_else(|| {
                BackendError::InvalidInput("decode weight is not a CudaBuffer".into())
            })?;
        let row_bytes = buf
            .tq2_0_row_bytes()
            .ok_or(BackendError::UnsupportedFormat(TernaryFormat::I2sInt8))?;
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

    /// Build a **fused** projection by concatenating `parts` along the output dim `N`
    /// (e.g. q‖k‖v, gate‖up). The parts must share `K` + `row_bytes` (they all project
    /// the same input). The packed weight rows are copied (dtod) into one arena and the
    /// scales concatenated, so one tiled GEMM produces all parts' outputs in one launch
    /// (fewer graph nodes, better M=1 occupancy). Bit-identical: each output row's
    /// warp-reduce is unchanged, only the grouping into one kernel differs. Owns a fresh
    /// arena (not the shared prefill weight), so `+Σ row_bytes·n` VRAM.
    fn build_fused(
        stream: &Arc<CudaStream>,
        parts: &[&DecodeLinearSpec],
    ) -> Result<Self, BackendError> {
        let bufs: Vec<&CudaBuffer> = parts
            .iter()
            .map(|p| {
                p.weights
                    .as_any()
                    .downcast_ref::<CudaBuffer>()
                    .ok_or_else(|| {
                        BackendError::InvalidInput("fused decode weight not a CudaBuffer".into())
                    })
            })
            .collect::<Result<_, _>>()?;
        let (_, k) = bufs[0].dims();
        let row_bytes = bufs[0]
            .tq2_0_row_bytes()
            .ok_or(BackendError::UnsupportedFormat(TernaryFormat::I2sInt8))?;
        let mut total_n = 0usize;
        for (b, p) in bufs.iter().zip(parts) {
            let (n, bk) = b.dims();
            if bk != k || b.tq2_0_row_bytes() != Some(row_bytes) || p.scales.len() != n {
                return Err(BackendError::InvalidInput(
                    "fused decode parts disagree on K/rb/scales".into(),
                ));
            }
            total_n += n;
        }
        let total_bytes = total_n * row_bytes;
        let mut device = stream
            .alloc_zeros::<u8>(total_bytes)
            .map_err(|e| driver_err("fused decode weight alloc", &e))?;
        let mut scales: Vec<f32> = Vec::with_capacity(total_n);
        let mut off = 0usize;
        for (b, p) in bufs.iter().zip(parts) {
            let bytes = b.dims().0 * row_bytes;
            let src = b.device_arc();
            let mut dst = device.slice_mut(off..off + bytes);
            stream
                .memcpy_dtod(src.as_ref(), &mut dst)
                .map_err(|e| driver_err("fused decode weight dtod", &e))?;
            scales.extend_from_slice(p.scales);
            off += bytes;
        }
        let d_scales = stream
            .clone_htod(&scales)
            .map_err(|e| driver_err("fused decode scales htod", &e))?;
        Ok(Self {
            device: Arc::new(device),
            scales: d_scales,
            n: total_n,
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
    /// Fused q‖k‖v projection (one GEMM for all three) — the graph path uses this; the
    /// eager `layer` keeps the separate q/k/v. Output `[q_width + 2·kv_width]`.
    qkv: ResidentLinear,
    /// Fused gate‖up projection. Output `[2·n_ff]`.
    gateup: ResidentLinear,
}

/// A fully device-resident BitNet decoder. One [`step`](CudaDecodeModel::step) is a
/// single-token (M=1) forward run entirely on the GPU. See the section banner above.
#[allow(missing_debug_implementations)]
/// Reusable device scratch for [`CudaDecodeModel::tree_verify_greedy`] —
/// allocated on first use for the requested node count (and re-grown if a
/// later tree is larger), then reused: a spec-decode loop calls verify every
/// few tokens, and 15 alloc/free round-trips + a zeroed `[m, vocab]` logits
/// buffer per call were measured to swallow the speculative gains entirely.
/// Every buffer is fully overwritten by the kernels that read it (the scores
/// scratch per live (row, head, key), the logits by the row-tiled LM head), so
/// reuse needs no zeroing.
struct TreeScratch {
    m_cap: usize,
    d_x: CudaSlice<f32>,
    d_normed: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_attn_sn: CudaSlice<f32>,
    d_proj: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_up: CudaSlice<f32>,
    d_gate_sn: CudaSlice<f32>,
    d_qact: CudaSlice<i8>,
    d_act_scale: CudaSlice<f32>,
    d_scores: CudaSlice<f32>,
    d_logits_all: CudaSlice<f32>,
    d_norm_all: CudaSlice<f32>,
    d_ids: CudaSlice<i32>,
    d_tok: CudaSlice<i32>,
    d_pos: CudaSlice<i32>,
    d_anc: CudaSlice<i32>,
    d_nanc: CudaSlice<i32>,
    d_amax_val: CudaSlice<f32>,
    d_amax_idx: CudaSlice<i32>,
}

/// Padded tree sizes with a captured verify-trunk graph. A verify with
/// `m <= TREE_BUCKET_MAX` nodes pads to the smallest bucket and replays that
/// bucket's graph (~1 launch instead of ~420 eager launches); larger trees
/// (the HTTP tree API allows them) use the eager path.
const TREE_BUCKETS: [usize; 5] = [8, 16, 24, 32, 48];
const TREE_BUCKET_MAX: usize = 48;

/// Captured verify-trunk graphs + their [prefix_len, real_m] ctrl buffer.
/// Graphs bake the TreeScratch buffer pointers — invalidated (dropped) if the
/// scratch ever re-grows.
struct TreeGraphs {
    d_ctrl: CudaSlice<i32>,
    graphs: HashMap<usize, SendGraph>,
    /// Keeps the raw modules the captured graphs reference alive (read only
    /// by Drop — the ownership IS the point).
    #[allow(dead_code)]
    raw_keepalive: Option<Arc<BatchRawKernels>>,
}

pub struct CudaDecodeModel {
    stream: Arc<CudaStream>,
    // Decode kernels (loaded from the resident modules at build).
    f_rmsnorm: CudaFunction,
    f_rope: CudaFunction,
    f_attn: CudaFunction,
    /// v1.x split attention pair (preferred when head_dim % 4 == 0).
    f_attn_scores: CudaFunction,
    f_attn_reduce: CudaFunction,
    /// Lazily grown scratch for `tree_verify_greedy` (see [`TreeScratch`]).
    tree_scratch: Option<TreeScratch>,
    /// The uncommitted tree from [`Self::tree_verify_logits`]: (node count,
    /// forward-time cache_len, parents). Consumed by [`Self::tree_commit`];
    /// invalidated by any new tree forward, any cache-advancing decode op
    /// (`step`/`step_graph`/`prefill`) or [`Self::reset`] — and the stamped
    /// cache_len makes commit REFUSE even if an invalidation site is missed.
    pending_tree: Option<(usize, usize, Vec<i32>)>,
    /// Captured tree-verify trunk graphs, keyed by bucket (see [`TreeGraphs`]).
    tree_graphs: Option<TreeGraphs>,
    f_residual: CudaFunction,
    f_embed: CudaFunction,
    f_lm_head: CudaFunction,
    f_relu2: CudaFunction,
    f_quant: CudaFunction,
    f_rmsnorm_quant: CudaFunction,
    #[allow(dead_code)] // unfused tiled handle; layer() now uses f_tiled_scaled
    f_tiled: CudaFunction,
    #[allow(dead_code)] // standalone scale-fold; unused since the fused scaled path
    f_scale: CudaFunction,
    f_tiled_scaled: CudaFunction,
    #[allow(dead_code)] // fused scaled+residual handle; wired into the residual path next
    f_tiled_scaled_residual: CudaFunction,
    // v0.3.6 batched (M>1) prefill kernels.
    f_rmsnorm_batch: CudaFunction,
    f_embed_batch: CudaFunction,
    f_rope_batch: CudaFunction,
    f_quant_batch: CudaFunction,
    #[allow(dead_code)]
    f_scale_batch: CudaFunction,
    f_kv_append_batch: CudaFunction,
    f_attn_batch: CudaFunction,
    /// Tree-verify attention (ADR 0014) + the batched LM head/argmax it shares
    /// with the batch-decode graph, resolved eagerly for `tree_verify_greedy`.
    f_attn_tree: CudaFunction,
    /// Split tree-verify attention pair (preferred when head_dim % 4 == 0):
    /// the single-kernel `f_attn_tree` is latency-bound per (node, head) warp
    /// and was the dominant per-verify cost (~150µs/layer at verify context).
    f_attn_tree_scores: CudaFunction,
    f_attn_tree_reduce: CudaFunction,
    f_argmax_partial: CudaFunction,
    f_argmax_combine: CudaFunction,
    f_lm_head_tiled: CudaFunction,
    f_lm_head_f16: CudaFunction,
    f_kv_append_mdecode: CudaFunction,
    #[allow(dead_code)] // kept for fallback; split-KV replaced it in the M=N path
    f_attn_mdecode: CudaFunction,
    f_attn_split_partial: CudaFunction,
    f_attn_combine: CudaFunction,
    // Dense device weights (uploaded once).
    d_token_embd: CudaSlice<f32>,
    /// f16 copy of `token_embd` (the GGUF's native precision) for the graph LM head — it
    /// reads the whole table per token, so f16 halves that 1.3 GB read bit-identically.
    d_token_embd_f16: CudaSlice<u16>,
    d_output_norm: CudaSlice<f32>,
    d_cos: CudaSlice<f32>,
    d_sin: CudaSlice<f32>,
    layers: Vec<ResidentLayer>,
    // Per-layer KV arenas, model-side for disjoint borrows. Stored as BYTES
    // (`max_ctx * kv_width * kv_elem`): elements are f32 (kv_elem = 4) or,
    // under the ADR 0020 f16 rung, __half (kv_elem = 2) — the dtype-selected
    // kernels interpret them; host code only ever slices by byte offsets.
    kv_k: Vec<CudaSlice<u8>>,
    kv_v: Vec<CudaSlice<u8>>,
    /// KV element size in bytes: 4 (f32, default), 2 (f16) or 1 (i8 rung).
    kv_elem: usize,
    /// The selected KV dtype (drives kernel-handle selection and gates).
    kv_dtype: KvDtype,
    /// i8-rung per-group scales, `[max_ctx, n_head_kv, head_dim/KV_QGROUP]`
    /// f32 per layer per direction; empty vecs on other rungs.
    kv_k_scales: Vec<CudaSlice<f32>>,
    kv_v_scales: Vec<CudaSlice<f32>>,
    cache_len: usize,
    // Reused scratch (sized once).
    d_x: CudaSlice<f32>,
    d_normed: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    /// Fused q‖k‖v GEMM output `[q_width + 2·kv_width]` (q/k/v are offset slices of it).
    d_qkv: CudaSlice<f32>,
    /// Fused gate‖up GEMM output `[2·n_ff]`.
    d_gateup: CudaSlice<f32>,
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
    d_qact: CudaSlice<i8>,
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
    graph: Option<SendGraph>,
    /// Raw-loaded PTX modules + `CUfunction` handles for the captured raw launches
    /// (the safe `CudaFunction` hides `cu_function`). `None` until the graph path is used.
    raw: Option<RawGraphKernels>,
    /// Raw batch (M=N) kernels for the graph-captured `decode_batch_graph`. A second JIT
    /// of the batch entry points, loaded lazily on first batched-graph decode. The per-N
    /// graph itself lives on the [`BatchKv`] (its buffers are N-specific). Held behind an
    /// `Arc` that each `BatchKv` clones when it captures a graph, so the modules these
    /// `CUfunction` handles come from stay loaded as long as *any* captured graph still
    /// references them — even if this model is dropped before the batch (the structural
    /// drop-order guarantee the M=1 `graph`/`raw` fields rely on does not exist for the
    /// caller-owned `BatchKv`).
    batch_raw: Option<Arc<BatchRawKernels>>,
}

impl CudaDecodeModel {
    /// Reset the KV cache (start a fresh sequence). The arena bytes are left as-is;
    /// only the watermark moves, exactly like [`crate`]'s host `KvCache::reset`.
    pub fn reset(&mut self) {
        self.pending_tree = None;
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
        // A cache-advancing op invalidates any uncommitted tree.
        self.pending_tree = None;
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

        // Populate the control block BEFORE the layer loop: the attention kernels
        // (`launch_attention` → the ctrl-driven split/warp kernels) read
        // `cache_len` from `d_ctrl[2]`. This was silently missing — the eager step
        // attended against whatever ctrl the last graph/batch call left behind
        // (ctx = 1 on a fresh model), masked because the runner always routes
        // decode through `step_graph`; caught by the eager↔graph cross-check in
        // `cuda_batch_and_graph_single_token_bit_identical`.
        let ctrl = [token as i32, pos as i32, self.cache_len as i32, 0i32];
        self.stream
            .memcpy_htod(&ctrl, &mut self.d_ctrl)
            .map_err(|e| driver_err("decode eager ctrl htod", &e))?;

        // Embedding gather: d_x = token_embd[token].
        Self::launch_embed(
            &self.stream,
            &self.f_embed,
            &self.d_token_embd,
            token,
            n_embd,
            &mut self.d_x,
        )?;

        for li in 0..self.layers.len() {
            self.layer(li, pos)?;
        }

        // Final norm over the (single) last token, then the tied LM head.
        Self::launch_rmsnorm(
            &self.stream,
            &self.f_rmsnorm,
            &self.d_x,
            &self.d_output_norm,
            eps,
            n_embd,
            &mut self.d_normed,
        )?;
        Self::launch_lm_head(
            &self.stream,
            &self.f_lm_head,
            &self.d_normed,
            &self.d_token_embd,
            n_embd,
            self.vocab,
            &mut self.d_logits,
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
        let (n_head, n_head_kv, head_dim, half) =
            (self.n_head, self.n_head_kv, self.head_dim, self.half);

        // --- pre-norm attention ---
        // Fused rmsnorm + quant (v0.7.0): skips intermediate d_normed + 1 launch.
        Self::launch_rmsnorm_quant(
            &self.stream,
            &self.f_rmsnorm_quant,
            &self.d_x,
            &self.layers[li].attn_norm,
            eps,
            n_embd,
            &mut self.d_qact,
            &mut self.d_act_scale,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.layers[li].q,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_q,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.layers[li].k,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_knew,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.layers[li].v,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_vnew,
        )?;

        // RoPE on q and the new k (this token's position row of the precomputed table).
        let base = pos * half;
        {
            let cos_v = self.d_cos.slice(base..base + half);
            let sin_v = self.d_sin.slice(base..base + half);
            Self::launch_rope(
                &self.stream,
                &self.f_rope,
                &mut self.d_q,
                &cos_v,
                &sin_v,
                n_head,
                head_dim,
            )?;
        }
        {
            let cos_v = self.d_cos.slice(base..base + half);
            let sin_v = self.d_sin.slice(base..base + half);
            Self::launch_rope(
                &self.stream,
                &self.f_rope,
                &mut self.d_knew,
                &cos_v,
                &sin_v,
                n_head_kv,
                head_dim,
            )?;
        }

        // Append the new k/v to this layer's KV arena at the watermark via the
        // dtype-selected append kernel (a plain copy for f32; converts under
        // the f16 rung — a dtod byte copy can't change element type).
        {
            let (d_knew, d_vnew) = (&self.d_knew, &self.d_vnew);
            let s = &self.stream;
            Self::bl_kv_append(
                s,
                &self.f_kv_append_batch,
                d_knew,
                &mut self.kv_k[li],
                self.cache_len,
                kv_width,
                1,
                if self.kv_dtype.has_scales() {
                    Some(&mut self.kv_k_scales[li])
                } else {
                    None
                },
            )?;
            Self::bl_kv_append(
                s,
                &self.f_kv_append_batch,
                d_vnew,
                &mut self.kv_v[li],
                self.cache_len,
                kv_width,
                1,
                if self.kv_dtype.has_scales() {
                    Some(&mut self.kv_v_scales[li])
                } else {
                    None
                },
            )?;
        }

        // Attention over the cached prefix (ctx = watermark+1, last visible = watermark).
        // The warp kernel reads cache_len from d_ctrl[2] (already populated by step()).
        Self::launch_attention(
            &self.stream,
            &self.f_attn,
            &self.f_attn_scores,
            &self.f_attn_reduce,
            &self.d_q,
            &self.kv_k[li],
            &self.kv_v[li],
            if self.kv_dtype.has_scales() {
                Some(&self.kv_k_scales[li])
            } else {
                None
            },
            if self.kv_dtype.has_scales() {
                Some(&self.kv_v_scales[li])
            } else {
                None
            },
            &mut self.d_attn,
            &mut self.d_scores,
            &self.d_ctrl,
            self.max_ctx,
            n_head,
            n_head_kv,
            head_dim,
            self.attn_scale,
        )?;

        // BitNet attn_sub_norm before o_proj (over q_width == n_embd), then o_proj +
        // the first residual into d_x.
        let attn_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].attn_sub_norm.as_ref() {
            Self::launch_rmsnorm(
                &self.stream,
                &self.f_rmsnorm,
                &self.d_attn,
                sn,
                eps,
                q_width,
                &mut self.d_attn_sn,
            )?;
            &self.d_attn_sn
        } else {
            &self.d_attn
        };
        Self::gemm(
            &self.stream,
            &self.f_quant,
            &self.f_tiled_scaled,
            attn_in,
            &self.layers[li].o,
            &mut self.d_qact,
            &mut self.d_act_scale,
            &mut self.d_proj_out,
        )?;
        Self::launch_residual(
            &self.stream,
            &self.f_residual,
            &mut self.d_x,
            &self.d_proj_out,
            n_embd,
        )?;

        // --- pre-norm ReLU² MLP ---
        // Fused rmsnorm + quant (v0.7.0).
        Self::launch_rmsnorm_quant(
            &self.stream,
            &self.f_rmsnorm_quant,
            &self.d_x,
            &self.layers[li].ffn_norm,
            eps,
            n_embd,
            &mut self.d_qact,
            &mut self.d_act_scale,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.layers[li].gate,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_gate,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.layers[li].up,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_up,
        )?;
        Self::launch_relu2(
            &self.stream,
            &self.f_relu2,
            &mut self.d_gate,
            &self.d_up,
            self.n_ff,
        )?;
        let down_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
            Self::launch_rmsnorm(
                &self.stream,
                &self.f_rmsnorm,
                &self.d_gate,
                sn,
                eps,
                self.n_ff,
                &mut self.d_gate_sn,
            )?;
            &self.d_gate_sn
        } else {
            &self.d_gate
        };
        Self::gemm(
            &self.stream,
            &self.f_quant,
            &self.f_tiled_scaled,
            down_in,
            &self.layers[li].down,
            &mut self.d_qact,
            &mut self.d_act_scale,
            &mut self.d_proj_out,
        )?;
        Self::launch_residual(
            &self.stream,
            &self.f_residual,
            &mut self.d_x,
            &self.d_proj_out,
            n_embd,
        )?;
        Ok(())
    }

    // --- launch helpers (associated fns so step()/layer() can pass disjoint field
    //     borrows of `self` without going through a `&self`/`&mut self` method). ---

    fn launch_rmsnorm(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        eps: f32,
        n: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(w).arg(&eps).arg(&n_i).arg(out);
        // SAFETY: `rmsnorm_f32(const float* x, const float* w, float eps, int n, float* out)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident rmsnorm", &e))?;
        }
        Ok(())
    }

    /// Fused RMSNorm + A8 quant (v0.7.0): reads `x`, writes `d_qact` + `d_act_scale`
    /// directly. Eliminates the intermediate `d_normed` buffer and one kernel launch.
    #[allow(clippy::too_many_arguments)]
    fn launch_rmsnorm_quant(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        eps: f32,
        n: usize,
        d_qact: &mut CudaSlice<i8>,
        d_act_scale: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (RMSNORM_QUANT_THREADS, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let mut l = stream.launch_builder(func);
        l.arg(x)
            .arg(w)
            .arg(&eps)
            .arg(&n_i)
            .arg(&mut *d_qact)
            .arg(&mut *d_act_scale);
        // SAFETY: `rmsnorm_quant_i8(x, w, eps, n, q_out /* i8 */, act_scale)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident rmsnorm_quant", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        stream: &Arc<CudaStream>,
        f_quant: &CudaFunction,
        f_tiled_scaled: &CudaFunction,
        d_in: &CudaSlice<f32>,
        lin: &ResidentLinear,
        d_qact: &mut CudaSlice<i8>,
        d_act_scale: &mut CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (n, k) = (lin.n, lin.k);
        let (n_i, k_i, m_i, rb_i) = (n as i32, k as i32, 1i32, lin.row_bytes as i32);
        // 1. on-device A8 quant of the activation row.
        {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(f_quant);
            l.arg(d_in)
                .arg(&k_i)
                .arg(&mut *d_qact)
                .arg(&mut *d_act_scale);
            // SAFETY: `act_quant_tiled_i8(const float* act, int k, signed char* q, float* scale)`.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch resident gemm quant", &e))?;
            }
        }
        // 2. Tiled GEMM with fused act_scale fold (v0.6.0 opt #15).
        //    Epilogue: out[mi,ni] = acc * scales[ni] * act_scale[mi].
        //    Single launch replaces the former GEMM + scale_mul_f32 pair.
        //    DP4A i8 kernel: reads the packed-int8 row directly — no dynamic shared.
        {
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(f_tiled_scaled);
            l.arg(&*d_qact)
                .arg(lin.device.as_ref())
                .arg(&lin.scales)
                .arg(&*d_act_scale)
                .arg(&mut *d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled_i8_scaled(qact /* i8 */, weights, scales,
            // act_scale, out, m, n, k, row_bytes)`.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch resident gemm tiled scaled", &e))?;
            }
        }
        Ok(())
    }

    /// Run just the A8 quantization step (act → q_act + act_scale).
    #[allow(dead_code)] // standalone quant launch; superseded by the fused path
    fn launch_quant(
        stream: &Arc<CudaStream>,
        f_quant: &CudaFunction,
        d_in: &CudaSlice<f32>,
        k: usize,
        d_qact: &mut CudaSlice<i8>,
        d_act_scale: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let k_i = k as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(f_quant);
        l.arg(d_in)
            .arg(&k_i)
            .arg(&mut *d_qact)
            .arg(&mut *d_act_scale);
        // SAFETY: `act_quant_tiled_i8(const float* act, int k, signed char* q, float* scale)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident quant", &e))?;
        }
        Ok(())
    }

    /// Run a tiled GEMM + scale fold on already-quantized activations.
    /// Skips the quant step — use when multiple projections share the same input.
    #[allow(clippy::too_many_arguments)]
    fn gemm_prequantized(
        stream: &Arc<CudaStream>,
        f_tiled_scaled: &CudaFunction,
        lin: &ResidentLinear,
        d_qact: &CudaSlice<i8>,
        d_act_scale: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (n, k) = (lin.n, lin.k);
        let (n_i, k_i, m_i, rb_i) = (n as i32, k as i32, 1i32, lin.row_bytes as i32);
        // Fused tiled GEMM (M=1) with act_scale fold in the epilogue.
        // DP4A i8 kernel: reads the packed-int8 row directly — no dynamic shared.
        {
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(f_tiled_scaled);
            l.arg(d_qact)
                .arg(lin.device.as_ref())
                .arg(&lin.scales)
                .arg(d_act_scale)
                .arg(&mut *d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled_i8_scaled(qact /* i8 */, weights, scales,
            // act_scale, out, m, n, k, row_bytes)`.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch resident gemm tiled scaled", &e))?;
            }
        }
        Ok(())
    }

    fn launch_rope(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        x: &mut CudaSlice<f32>,
        cos_t: &CudaView<f32>,
        sin_t: &CudaView<f32>,
        n_head: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i) = (n_head as i32, head_dim as i32);
        let total = (n_head * (head_dim / 2)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(cos_t).arg(sin_t).arg(&nh_i).arg(&hd_i);
        // SAFETY: `rope_apply_f32(float* x, const float* cos_t, const float* sin_t, int n_head, int head_dim)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident rope", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention(
        stream: &Arc<CudaStream>,
        func_legacy: &CudaFunction,
        func_scores: &CudaFunction,
        func_reduce: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        k_scales: Option<&CudaSlice<f32>>,
        v_scales: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        ctrl: &CudaSlice<i32>,
        max_ctx: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(), BackendError> {
        let (mc_i, nh_i, nhkv_i, hd_i) = (
            max_ctx as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
        );
        if head_dim.is_multiple_of(4) {
            // v1.x split pair (bit-identical, ctx-parallel — see decode.cu).
            // 1) scores fan-out: grid (n_head, ceil(max_ctx/SCORE_CHUNK)); blocks
            //    past the live ctx early-exit via ctrl, so the static grid replays
            //    at every context length.
            {
                let cfg = LaunchConfig {
                    grid_dim: (
                        n_head as u32,
                        (max_ctx.div_ceil(ATTN_SCORE_CHUNK)) as u32,
                        1,
                    ),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut l = stream.launch_builder(func_scores);
                l.arg(q)
                    .arg(k)
                    .arg(&mut *scores)
                    .arg(ctrl)
                    .arg(&mc_i)
                    .arg(&nh_i)
                    .arg(&nhkv_i)
                    .arg(&hd_i)
                    .arg(&scale);
                if let Some(sc) = k_scales {
                    // The `_q8` twin takes the K scales as a trailing param.
                    l.arg(sc);
                }
                // SAFETY: `gqa_attention_scores_g(q, k, scores, ctrl, max_ctx, n_head, n_head_kv, head_dim, scale)`.
                #[allow(unsafe_code)]
                unsafe {
                    l.launch(cfg)
                        .map_err(|e| driver_err("launch resident attention scores", &e))?;
                }
            }
            // 2) per-head softmax + weighted reduce: 128-thread block, stages the
            //    head's scores in dynamic shared (same over-48KiB opt-in).
            {
                let cfg = LaunchConfig {
                    grid_dim: (n_head as u32, 1, 1),
                    block_dim: (ATTN_REDUCE_THREADS, 1, 1),
                    shared_mem_bytes: (max_ctx * 4) as u32,
                };
                let mut l = stream.launch_builder(func_reduce);
                l.arg(v)
                    .arg(&*scores)
                    .arg(out)
                    .arg(ctrl)
                    .arg(&mc_i)
                    .arg(&nh_i)
                    .arg(&nhkv_i)
                    .arg(&hd_i);
                if let Some(sc) = v_scales {
                    // The `_q8` twin takes the V scales as a trailing param.
                    l.arg(sc);
                }
                // SAFETY: `gqa_attention_reduce_g(v, scores, out, ctrl, max_ctx, n_head, n_head_kv, head_dim)`.
                #[allow(unsafe_code)]
                unsafe {
                    l.launch(cfg)
                        .map_err(|e| driver_err("launch resident attention reduce", &e))?;
                }
            }
            return Ok(());
        }
        // Legacy geometry (head_dim % 4 != 0): one warp-block per head; the kernel
        // stages the head's scores in dynamic shared memory (`max_ctx * 4` bytes).
        // `scores` is the legacy global scratch, unused by the warp kernel.
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: (max_ctx * 4) as u32,
        };
        let mut l = stream.launch_builder(func_legacy);
        l.arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(scores)
            .arg(ctrl)
            .arg(&mc_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale);
        // SAFETY: `gqa_attention_decode_warp_g(q, k, v, out, scores, ctrl, max_ctx, n_head, n_head_kv, head_dim, scale)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident attention", &e))?;
        }
        Ok(())
    }

    fn launch_residual(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(x).arg(y).arg(&n_i);
        // SAFETY: `residual_add_f32(float* x, const float* y, int n)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident residual", &e))?;
        }
        Ok(())
    }

    fn launch_relu2(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(gate).arg(up).arg(&n_i);
        // SAFETY: `relu2_gate_f32(float* gate, const float* up, int n)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident relu2", &e))?;
        }
        Ok(())
    }

    fn launch_embed(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        table: &CudaSlice<f32>,
        tok: u32,
        n_embd: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (tok_i, ne_i) = (tok as i32, n_embd as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n_embd as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(table).arg(&tok_i).arg(&ne_i).arg(out);
        // SAFETY: `embedding_gather_f32(const float* table, int tok, int n_embd, float* out)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident embed", &e))?;
        }
        Ok(())
    }

    fn launch_lm_head(
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        h: &CudaSlice<f32>,
        embd: &CudaSlice<f32>,
        n_embd: usize,
        vocab: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i) = (n_embd as i32, vocab as i32);
        let cfg = LaunchConfig {
            grid_dim: ((vocab as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(func);
        l.arg(h).arg(embd).arg(&ne_i).arg(&v_i).arg(logits);
        // SAFETY: `lm_head_f32(const float* h, const float* embd, int n_embd, int vocab, float* logits)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch resident lm_head", &e))?;
        }
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
        // A cache-advancing op invalidates any uncommitted tree.
        self.pending_tree = None;
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

        // Drain any pending work on the default stream before the graph (on `cap_stream`)
        // reads/writes the shared buffers. The graph-only runner path never leaves work
        // there (so this is a no-op sync on an idle stream), but if a caller interleaves
        // the eager `step` (which runs on `self.stream`) with `step_graph`, this closes the
        // cross-stream race on the residual stream + KV arenas.
        self.stream
            .synchronize()
            .map_err(|e| driver_err("decode pre-graph default sync", &e))?;

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

    /// **Batched (M>1) prefill** — process all `tokens` (`positions[r]` absolute) in ONE
    /// device-resident M=P forward, seeding the KV cache and advancing the watermark by P,
    /// then return the last token's logits. Replaces the O(P) sequential `step_graph` loop
    /// (the TTFT cliff). Bit-identical to that loop per row (the batch kernels share the
    /// M=1 reduction order). Eager (safe) launches — prefill is one-shot, not replayed, so
    /// it needs no graph. `positions[0]` must equal [`cache_len`](Self::cache_len) and the
    /// positions must be contiguous.
    ///
    /// # Errors
    /// [`BackendError`] on capacity overflow, a `pos` guard, an out-of-range token, or a
    /// device failure.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        // A cache-advancing op invalidates any uncommitted tree.
        self.pending_tree = None;
        let m = tokens.len();
        if m == 0 || positions.len() != m {
            return Err(BackendError::InvalidInput(
                "prefill: empty or mismatched positions".into(),
            ));
        }
        if positions[0] != self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "prefill pos[0]={} must equal cache_len={}",
                positions[0], self.cache_len
            )));
        }
        if self.cache_len + m > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "prefill overflow: cache_len={} + {m} > max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }
        for (r, (&t, &p)) in tokens.iter().zip(positions).enumerate() {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "prefill token {t} out of range"
                )));
            }
            if p != self.cache_len + r {
                return Err(BackendError::InvalidInput(
                    "prefill positions not contiguous".into(),
                ));
            }
        }

        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let causal_offset = self.cache_len;
        let ctx_max = self.cache_len + m;

        // Upload token ids + positions as i32.
        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let d_tokens = s
            .clone_htod(&tok_i)
            .map_err(|e| driver_err("prefill tokens htod", &e))?;
        let d_positions = s
            .clone_htod(&pos_i)
            .map_err(|e| driver_err("prefill positions htod", &e))?;

        // M=P scratch (allocated for this prefill; freed on return).
        let alloc =
            |n: usize, what: &str| s.alloc_zeros::<f32>(n).map_err(|e| driver_err(what, &e));
        let mut d_x = alloc(m * n_embd, "prefill d_x")?;
        let mut d_normed = alloc(m * n_embd, "prefill d_normed")?;
        let mut d_q = alloc(m * q_width, "prefill d_q")?;
        let mut d_k = alloc(m * kv_width, "prefill d_k")?;
        let mut d_v = alloc(m * kv_width, "prefill d_v")?;
        let mut d_attn = alloc(m * q_width, "prefill d_attn")?;
        let mut d_attn_sn = alloc(m * q_width, "prefill d_attn_sn")?;
        let mut d_proj = alloc(m * n_embd, "prefill d_proj")?;
        let mut d_gate = alloc(m * n_ff, "prefill d_gate")?;
        let mut d_up = alloc(m * n_ff, "prefill d_up")?;
        let mut d_gate_sn = alloc(m * n_ff, "prefill d_gate_sn")?;
        let mut d_qact = s
            .alloc_zeros::<i8>(m * n_ff)
            .map_err(|e| driver_err("prefill d_qact", &e))?;
        let mut d_act_scale = alloc(m, "prefill d_act_scale")?;
        let mut d_scores = alloc(m * n_head * ctx_max, "prefill d_scores")?;
        let mut d_logits = alloc(self.vocab, "prefill d_logits")?;

        // Embedding gather (m rows).
        Self::bl_embed(
            s,
            &self.f_embed_batch,
            &self.d_token_embd,
            &d_tokens,
            n_embd,
            m,
            &mut d_x,
        )?;

        for li in 0..self.layers.len() {
            // --- attention ---
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &d_x,
                &self.layers[li].attn_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut d_normed,
            )?;
            // q/k/v share one quant of d_normed.
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &d_normed,
                n_embd,
                m,
                &mut d_qact,
                &mut d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].q,
                &d_act_scale,
                m,
                &mut d_q,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].k,
                &d_act_scale,
                m,
                &mut d_k,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].v,
                &d_act_scale,
                m,
                &mut d_v,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut d_q,
                &self.d_cos,
                &self.d_sin,
                &d_positions,
                n_head,
                head_dim,
                m,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut d_k,
                &self.d_cos,
                &self.d_sin,
                &d_positions,
                n_head_kv,
                head_dim,
                m,
            )?;
            Self::bl_kv_append(
                s,
                &self.f_kv_append_batch,
                &d_k,
                &mut self.kv_k[li],
                causal_offset,
                kv_width,
                m,
                if self.kv_dtype.has_scales() {
                    Some(&mut self.kv_k_scales[li])
                } else {
                    None
                },
            )?;
            Self::bl_kv_append(
                s,
                &self.f_kv_append_batch,
                &d_v,
                &mut self.kv_v[li],
                causal_offset,
                kv_width,
                m,
                if self.kv_dtype.has_scales() {
                    Some(&mut self.kv_v_scales[li])
                } else {
                    None
                },
            )?;
            Self::bl_attn(
                s,
                &self.f_attn_batch,
                &d_q,
                &self.kv_k[li],
                &self.kv_v[li],
                if self.kv_dtype.has_scales() {
                    Some(&self.kv_k_scales[li])
                } else {
                    None
                },
                if self.kv_dtype.has_scales() {
                    Some(&self.kv_v_scales[li])
                } else {
                    None
                },
                &mut d_attn,
                &mut d_scores,
                ctx_max,
                n_head,
                n_head_kv,
                head_dim,
                self.attn_scale,
                causal_offset,
                m,
            )?;
            let attn_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].attn_sub_norm.as_ref()
            {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &d_attn,
                    sn,
                    self.rms_eps,
                    q_width,
                    m,
                    &mut d_attn_sn,
                )?;
                &d_attn_sn
            } else {
                &d_attn
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                attn_in,
                q_width,
                m,
                &mut d_qact,
                &mut d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].o,
                &d_act_scale,
                m,
                &mut d_proj,
            )?;
            Self::bl_residual(s, &self.f_residual, &mut d_x, &d_proj, m * n_embd)?;

            // --- ReLU² MLP ---
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &d_x,
                &self.layers[li].ffn_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut d_normed,
            )?;
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &d_normed,
                n_embd,
                m,
                &mut d_qact,
                &mut d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].gate,
                &d_act_scale,
                m,
                &mut d_gate,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].up,
                &d_act_scale,
                m,
                &mut d_up,
            )?;
            Self::bl_relu2(s, &self.f_relu2, &mut d_gate, &d_up, m * n_ff)?;
            let down_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &d_gate,
                    sn,
                    self.rms_eps,
                    n_ff,
                    m,
                    &mut d_gate_sn,
                )?;
                &d_gate_sn
            } else {
                &d_gate
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                down_in,
                n_ff,
                m,
                &mut d_qact,
                &mut d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &d_qact,
                &self.layers[li].down,
                &d_act_scale,
                m,
                &mut d_proj,
            )?;
            Self::bl_residual(s, &self.f_residual, &mut d_x, &d_proj, m * n_embd)?;
        }

        // Final norm over the LAST token only, then the tied LM head (f16 table).
        let mut d_last = alloc(n_embd, "prefill d_last")?;
        {
            let last_row = d_x.slice((m - 1) * n_embd..m * n_embd);
            s.memcpy_dtod(&last_row, &mut d_last)
                .map_err(|e| driver_err("prefill last row", &e))?;
        }
        let mut d_last_norm = alloc(n_embd, "prefill d_last_norm")?;
        Self::bl_rmsnorm(
            s,
            &self.f_rmsnorm_batch,
            &d_last,
            &self.d_output_norm,
            self.rms_eps,
            n_embd,
            1,
            &mut d_last_norm,
        )?;
        Self::bl_lm_head_f16(
            s,
            &self.f_lm_head_f16,
            &d_last_norm,
            &self.d_token_embd_f16,
            n_embd,
            self.vocab,
            &mut d_logits,
        )?;

        let mut logits = vec![0.0f32; self.vocab];
        s.memcpy_dtoh(&d_logits, &mut logits)
            .map_err(|e| driver_err("prefill logits dtoh", &e))?;
        self.cache_len += m;
        Ok(logits)
    }

    /// **BASTION greedy tree-verify** (ADR 0014) — verify a draft token tree in ONE
    /// batched forward and commit the longest greedy-accepted path.
    ///
    /// `tokens[i]` / `parents[i]` describe the tree: node 0 is the single root
    /// (`parents[0] == -1`) and MUST be the token the caller was about to `step`
    /// (greedy target semantics: the root is already committed by the caller's
    /// previous argmax); every other node has `parents[i] < i`. Node `i` is a
    /// draft candidate for the position after its parent. Duplicate sibling
    /// tokens are allowed; the first matching child wins.
    ///
    /// The whole tree runs as an M=N batched forward (rows = nodes, RoPE at
    /// `cache_len + depth(i)`, K/V written provisionally at arena rows
    /// `cache_len + i`) with the tree-masked attention
    /// (`gqa_attention_tree_f32`): each node attends the committed prefix plus
    /// its own ancestor chain. Greedy acceptance walks from the root taking the
    /// child whose token equals the target argmax at the current node; the
    /// accepted path's K/V rows are then promoted (compacted) into
    /// `cache_len..cache_len+L` and the watermark advances by `L` — rejected
    /// rows sit past the watermark and are dead (O(1) rollback).
    ///
    /// Returns the `L` newly determined tokens: `out[k]` = target argmax at the
    /// k-th accepted node. `out[k] == tokens[path[k+1]]` for the accepted
    /// drafts and `out[L-1]` is the bonus token (feed it back as the next
    /// root). `L >= 1` always: a full draft reject degenerates to exactly one
    /// plain greedy step — losslessness is by construction, since every
    /// returned token IS the target's greedy argmax at its position.
    ///
    /// Intended tree sizes are small (BASTION budgets N at the roofline knee,
    /// typically ≲ 64 nodes): the ancestor table is O(N²) and the scores
    /// scratch O(N · n_head · (cache_len + N)) — both per-call allocations.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a malformed tree (root not at 0,
    /// non-topological parents, out-of-range token) or capacity overflow;
    /// device errors otherwise.
    ///
    /// This is the shared FORWARD half: it validates the tree, appends
    /// provisional K/V at arena rows [cache_len, cache_len + m) and leaves
    /// every node's logits in `tree_scratch.d_logits_all` WITHOUT committing.
    /// [`Self::tree_verify_greedy`] adds the device argmax + greedy walk;
    /// [`Self::tree_verify_logits`] hands the logits to the host for the
    /// speculative-sampling accept rule, with [`Self::tree_commit`] promoting
    /// the host-chosen path.
    fn tree_forward(&mut self, tokens: &[u32], parents: &[i32]) -> Result<usize, BackendError> {
        // A new forward invalidates any uncommitted previous tree.
        self.pending_tree = None;
        let m = tokens.len();
        if m == 0 || parents.len() != m {
            return Err(BackendError::InvalidInput(
                "tree_verify: empty or mismatched parents".into(),
            ));
        }
        if parents[0] != -1 {
            return Err(BackendError::InvalidInput(
                "tree_verify: node 0 must be the root (parent -1)".into(),
            ));
        }
        for (i, &p) in parents.iter().enumerate().skip(1) {
            if p < 0 || p as usize >= i {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify: parents[{i}]={p} is not topological (0 <= parent < i)"
                )));
            }
        }
        for &t in tokens {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify token {t} out of range"
                )));
            }
        }
        if self.cache_len + m > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify overflow: cache_len={} + {m} nodes > max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }

        // Graph route: pad to the smallest captured bucket and replay ONE
        // graph instead of ~420 eager launches. Requirements: the split
        // attention geometry (the ctrl twins are float4 kernels), a bucketable
        // size, room for the PADDED tree in the arena, and a context whose
        // score staging fits the default shared-memory limit (the raw-handle
        // capture path doesn't carry the opt-in attribute the safe handles
        // get at load).
        let bucket = if self.head_dim.is_multiple_of(4)
            && m <= TREE_BUCKET_MAX
            && self.max_ctx * 4 <= 48 * 1024
            && self.kv_elem == 4 // non-f32 KV: eager tree (no ctrl twins; graph measured ≈ no win)
            && std::env::var_os("TRITIUM_TREE_EAGER").is_none()
        {
            TREE_BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= m && self.cache_len + b <= self.max_ctx)
        } else {
            None
        };
        // mb = padded node count (bucket) or the real m on the eager path;
        // every host array below is built at stride/length mb. Pad rows are
        // root-token duplicates at depth 1 — valid math whose results only
        // the pads themselves ever see.
        let mb = bucket.unwrap_or(m);

        // Depths, RoPE positions, and per-node ancestor slot lists (root-first,
        // including self). Ancestors are arena slots (cache_len + node index).
        let mut depth = vec![0usize; m];
        let mut anc: Vec<i32> = vec![0; mb * mb]; // [mb, max_anc=mb], row-major
        let mut n_anc = vec![0i32; mb];
        for i in 0..m {
            if parents[i] >= 0 {
                let p = parents[i] as usize;
                depth[i] = depth[p] + 1;
                let (dst_off, src_off) = (i * mb, p * mb);
                let np = n_anc[p] as usize;
                // anc[i] = anc[parent] ++ [slot(i)] (rows are disjoint: p < i).
                anc.copy_within(src_off..src_off + np, dst_off);
                anc[dst_off + np] = (self.cache_len + i) as i32;
                n_anc[i] = n_anc[p] + 1;
            } else {
                anc[i * mb] = (self.cache_len + i) as i32;
                n_anc[i] = 1;
            }
        }
        for i in m..mb {
            // Pad: a root child (its ancestor list is the root's slot + its own).
            anc[i * mb] = self.cache_len as i32;
            anc[i * mb + 1] = (self.cache_len + i) as i32;
            n_anc[i] = 2;
        }
        let mut positions: Vec<i32> = depth.iter().map(|&d| (self.cache_len + d) as i32).collect();
        positions.resize(mb, (self.cache_len + 1) as i32);

        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let prefix_len = self.cache_len;
        let ctx_max = self.cache_len + m;

        // Reusable M=N scratch: allocated on first use (or re-grown for a larger
        // tree), then cached on the model — per-call alloc/free measurably ate
        // the speculative gains. Scores are sized by `max_ctx` (the largest any
        // verify can need) so growth is driven by `m` alone.
        // Capacity covers the graph buckets from the start so the captured
        // graphs' baked pointers stay valid; an oversized eager tree (> the
        // bucket max) re-grows the scratch and must drop every graph.
        let m_cap_want = mb.max(TREE_BUCKET_MAX);
        if self
            .tree_scratch
            .as_ref()
            .is_none_or(|t| t.m_cap < m_cap_want)
        {
            // Unconditional: an error-path `?` between take and put-back drops
            // the scratch while captured graphs keep its baked pointers — on
            // the next call the scratch is None but stale graphs would replay
            // into freed memory if this drop were guarded on `is_some()`.
            self.tree_graphs = None;
            let m = m_cap_want;
            let alloc =
                |n: usize, what: &str| s.alloc_zeros::<f32>(n).map_err(|e| driver_err(what, &e));
            self.tree_scratch = Some(TreeScratch {
                m_cap: m,
                d_x: alloc(m * n_embd, "tree d_x")?,
                d_normed: alloc(m * n_embd, "tree d_normed")?,
                d_q: alloc(m * q_width, "tree d_q")?,
                d_k: alloc(m * kv_width, "tree d_k")?,
                d_v: alloc(m * kv_width, "tree d_v")?,
                d_attn: alloc(m * q_width, "tree d_attn")?,
                d_attn_sn: alloc(m * q_width, "tree d_attn_sn")?,
                d_proj: alloc(m * n_embd, "tree d_proj")?,
                d_gate: alloc(m * n_ff, "tree d_gate")?,
                d_up: alloc(m * n_ff, "tree d_up")?,
                d_gate_sn: alloc(m * n_ff, "tree d_gate_sn")?,
                d_qact: s
                    .alloc_zeros::<i8>(m * n_ff)
                    .map_err(|e| driver_err("tree d_qact", &e))?,
                d_act_scale: alloc(m, "tree d_act_scale")?,
                d_scores: alloc(m * n_head * self.max_ctx, "tree d_scores")?,
                d_logits_all: alloc(m * self.vocab, "tree d_logits")?,
                d_norm_all: alloc(m * n_embd, "tree d_norm_all")?,
                d_ids: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_ids", &e))?,
                d_tok: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_tok", &e))?,
                d_pos: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_pos", &e))?,
                d_anc: s
                    .alloc_zeros::<i32>(m * m)
                    .map_err(|e| driver_err("tree d_anc", &e))?,
                d_nanc: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_nanc", &e))?,
                d_amax_val: alloc(m * ARGMAX_CHUNKS, "tree d_amax_val")?,
                d_amax_idx: s
                    .alloc_zeros::<i32>(m * ARGMAX_CHUNKS)
                    .map_err(|e| driver_err("tree d_amax_idx", &e))?,
            });
        }
        // Move the scratch out for disjoint borrows vs `self.kv_*` below; put
        // it back once the device work completes.
        let mut ts = self.tree_scratch.take().expect("tree scratch just ensured");

        // Uploads go into the cached buffers too (oversized is fine — kernels
        // read exactly the first m / m·m entries; `max_anc == m` is a stride
        // into linear memory, not a buffer shape).
        let mut tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        tok_i.resize(mb, tok_i[0]); // pads embed the root token (valid, unread)
        s.memcpy_htod(&tok_i, &mut ts.d_tok)
            .map_err(|e| driver_err("tree tokens htod", &e))?;
        s.memcpy_htod(&positions, &mut ts.d_pos)
            .map_err(|e| driver_err("tree positions htod", &e))?;
        s.memcpy_htod(&anc, &mut ts.d_anc)
            .map_err(|e| driver_err("tree anc htod", &e))?;
        s.memcpy_htod(&n_anc, &mut ts.d_nanc)
            .map_err(|e| driver_err("tree n_anc htod", &e))?;

        if let Some(bucket) = bucket {
            // ── Graph route: replay the captured trunk (1 launch), then the
            // eager tail at the REAL node count (a padded LM head would read
            // the 656 MB f16 table once per extra 8-row tile).
            if self.batch_raw.is_none() {
                let ctx = self.cap_stream.context().clone();
                self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
            }
            if self.tree_graphs.is_none() {
                let d_ctrl = self
                    .cap_stream
                    .alloc_zeros::<i32>(2)
                    .map_err(|e| driver_err("tree ctrl alloc", &e))?;
                self.tree_graphs = Some(TreeGraphs {
                    d_ctrl,
                    graphs: HashMap::new(),
                    raw_keepalive: self.batch_raw.clone(),
                });
            }
            let have = self
                .tree_graphs
                .as_ref()
                .expect("tree graphs just ensured")
                .graphs
                .contains_key(&bucket);
            if !have {
                if std::env::var_os("TRITIUM_TREE_DEBUG").is_some() {
                    eprintln!("tree-graph: capturing bucket {bucket}");
                }
                let g = self.record_graph_tree(&ts, bucket)?;
                self.tree_graphs
                    .as_mut()
                    .expect("tree graphs just ensured")
                    .graphs
                    .insert(bucket, SendGraph(g));
            }
            // The uploads above ran on the default stream; the graph replays
            // on the capture stream — order them before the ctrl write.
            s.synchronize()
                .map_err(|e| driver_err("tree pre-replay sync", &e))?;
            let ctrl = [prefix_len as i32, m as i32];
            let tg = self.tree_graphs.as_mut().expect("tree graphs just ensured");
            self.cap_stream
                .memcpy_htod(&ctrl, &mut tg.d_ctrl)
                .map_err(|e| driver_err("tree ctrl htod", &e))?;
            self.tree_graphs
                .as_ref()
                .expect("tree graphs just ensured")
                .graphs
                .get(&bucket)
                .expect("tree graph just inserted")
                .launch()
                .map_err(|e| driver_err("tree graph launch", &e))?;
            let cs = &self.cap_stream;
            Self::bl_rmsnorm(
                cs,
                &self.f_rmsnorm_batch,
                &ts.d_x,
                &self.d_output_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut ts.d_norm_all,
            )?;
            Self::bl_lm_head_tiled(
                cs,
                &self.f_lm_head_tiled,
                &ts.d_norm_all,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                m,
                &mut ts.d_logits_all,
            )?;
            // The verify's writes (KV appends included) must be visible to the
            // default-stream consumers that follow (argmax/logits dtoh/promote).
            cs.synchronize()
                .map_err(|e| driver_err("tree post-replay sync", &e))?;
        } else {
            Self::bl_embed(
                s,
                &self.f_embed_batch,
                &self.d_token_embd,
                &ts.d_tok,
                n_embd,
                m,
                &mut ts.d_x,
            )?;

            for li in 0..self.layers.len() {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &ts.d_x,
                    &self.layers[li].attn_norm,
                    self.rms_eps,
                    n_embd,
                    m,
                    &mut ts.d_normed,
                )?;
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    &ts.d_normed,
                    n_embd,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].q,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_q,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].k,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_k,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].v,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_v,
                )?;
                Self::bl_rope(
                    s,
                    &self.f_rope_batch,
                    &mut ts.d_q,
                    &self.d_cos,
                    &self.d_sin,
                    &ts.d_pos,
                    n_head,
                    head_dim,
                    m,
                )?;
                Self::bl_rope(
                    s,
                    &self.f_rope_batch,
                    &mut ts.d_k,
                    &self.d_cos,
                    &self.d_sin,
                    &ts.d_pos,
                    n_head_kv,
                    head_dim,
                    m,
                )?;
                // Provisional K/V at arena rows [cache_len, cache_len + m) — node i's
                // row is cache_len + i regardless of its depth (attention resolves
                // rows through the ancestor table, not contiguity).
                Self::bl_kv_append(
                    s,
                    &self.f_kv_append_batch,
                    &ts.d_k,
                    &mut self.kv_k[li],
                    prefix_len,
                    kv_width,
                    m,
                    if self.kv_dtype.has_scales() {
                        Some(&mut self.kv_k_scales[li])
                    } else {
                        None
                    },
                )?;
                Self::bl_kv_append(
                    s,
                    &self.f_kv_append_batch,
                    &ts.d_v,
                    &mut self.kv_v[li],
                    prefix_len,
                    kv_width,
                    m,
                    if self.kv_dtype.has_scales() {
                        Some(&mut self.kv_v_scales[li])
                    } else {
                        None
                    },
                )?;
                if head_dim.is_multiple_of(4) {
                    Self::bl_attn_tree_split(
                        s,
                        &self.f_attn_tree_scores,
                        &self.f_attn_tree_reduce,
                        &ts.d_q,
                        &self.kv_k[li],
                        &self.kv_v[li],
                        if self.kv_dtype.has_scales() {
                            Some(&self.kv_k_scales[li])
                        } else {
                            None
                        },
                        if self.kv_dtype.has_scales() {
                            Some(&self.kv_v_scales[li])
                        } else {
                            None
                        },
                        &mut ts.d_attn,
                        &mut ts.d_scores,
                        &ts.d_anc,
                        &ts.d_nanc,
                        ctx_max,
                        n_head,
                        n_head_kv,
                        head_dim,
                        self.attn_scale,
                        prefix_len,
                        m,
                    )?;
                } else {
                    Self::bl_attn_tree(
                        s,
                        &self.f_attn_tree,
                        &ts.d_q,
                        &self.kv_k[li],
                        &self.kv_v[li],
                        &mut ts.d_attn,
                        &mut ts.d_scores,
                        &ts.d_anc,
                        &ts.d_nanc,
                        ctx_max,
                        n_head,
                        n_head_kv,
                        head_dim,
                        self.attn_scale,
                        prefix_len,
                        m,
                    )?;
                }
                let attn_in: &CudaSlice<f32> =
                    if let Some(sn) = self.layers[li].attn_sub_norm.as_ref() {
                        Self::bl_rmsnorm(
                            s,
                            &self.f_rmsnorm_batch,
                            &ts.d_attn,
                            sn,
                            self.rms_eps,
                            q_width,
                            m,
                            &mut ts.d_attn_sn,
                        )?;
                        &ts.d_attn_sn
                    } else {
                        &ts.d_attn
                    };
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    attn_in,
                    q_width,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].o,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_proj,
                )?;
                Self::bl_residual(s, &self.f_residual, &mut ts.d_x, &ts.d_proj, m * n_embd)?;

                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &ts.d_x,
                    &self.layers[li].ffn_norm,
                    self.rms_eps,
                    n_embd,
                    m,
                    &mut ts.d_normed,
                )?;
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    &ts.d_normed,
                    n_embd,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].gate,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_gate,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].up,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_up,
                )?;
                Self::bl_relu2(s, &self.f_relu2, &mut ts.d_gate, &ts.d_up, m * n_ff)?;
                let down_in: &CudaSlice<f32> =
                    if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
                        Self::bl_rmsnorm(
                            s,
                            &self.f_rmsnorm_batch,
                            &ts.d_gate,
                            sn,
                            self.rms_eps,
                            n_ff,
                            m,
                            &mut ts.d_gate_sn,
                        )?;
                        &ts.d_gate_sn
                    } else {
                        &ts.d_gate
                    };
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    down_in,
                    n_ff,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].down,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_proj,
                )?;
                Self::bl_residual(s, &self.f_residual, &mut ts.d_x, &ts.d_proj, m * n_embd)?;
            }

            // Final norm over ALL rows, batched LM head, per-row greedy argmax.
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &ts.d_x,
                &self.d_output_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut ts.d_norm_all,
            )?;
            Self::bl_lm_head_tiled(
                s,
                &self.f_lm_head_tiled,
                &ts.d_norm_all,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                m,
                &mut ts.d_logits_all,
            )?;
        }
        // Forward complete: logits for every tree node sit in
        // `tree_scratch.d_logits_all[0..m*vocab]`; provisional K/V occupy arena
        // rows [cache_len, cache_len + m). Nothing is committed yet.
        self.tree_scratch = Some(ts);
        Ok(m)
    }

    /// Greedy tree verify (ADR 0014): forward the draft tree, device-argmax
    /// every node, walk the accepted path and commit it. Returns the target's
    /// greedy tokens along the accepted path (+ the bonus token).
    pub fn tree_verify_greedy(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<u32>, BackendError> {
        let m = self.tree_forward(tokens, parents)?;
        let s = &self.stream;
        let mut ts = self
            .tree_scratch
            .take()
            .expect("tree scratch after forward");
        Self::bl_argmax_rows_chunked(
            s,
            &self.f_argmax_partial,
            &self.f_argmax_combine,
            &ts.d_logits_all,
            self.vocab,
            m,
            &mut ts.d_amax_val,
            &mut ts.d_amax_idx,
            &mut ts.d_ids,
        )?;
        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut ids = vec![0i32; m];
        // The cached buffer may exceed this call's `m` — copy exactly m ids.
        let ids_view = ts.d_ids.slice(0..m);
        s.memcpy_dtoh(&ids_view, &mut ids)
            .map_err(|e| driver_err("tree ids dtoh", &e))?;

        // Device work is done — return the scratch to the cache. (An early `?`
        // above drops it instead; the next call simply re-allocates.)
        self.tree_scratch = Some(ts);

        // Greedy accept walk: from the root, descend into the (first) child whose
        // draft token equals the target argmax at the current node.
        let mut path = vec![0usize];
        loop {
            let cur = *path.last().expect("path non-empty");
            let want = ids[cur];
            let next = (cur + 1..m).find(|&c| parents[c] as usize == cur && tok_i[c] == want);
            match next {
                Some(c) => path.push(c),
                None => break,
            }
        }
        self.tree_promote(&path)?;

        Ok(path.iter().map(|&n| ids[n] as u32).collect())
    }

    /// Forward a draft tree and return every node's logits `[m, vocab]`
    /// row-major on the host, for a HOST-side accept rule (speculative
    /// sampling). Provisional K/V occupy arena rows [cache_len, cache_len+m);
    /// nothing is committed until [`Self::tree_commit`]. Any other decode
    /// operation (or another tree forward) in between invalidates the
    /// provisional rows — commit refuses once the pending tree is gone.
    pub fn tree_verify_logits(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<f32>, BackendError> {
        let m = self.tree_forward(tokens, parents)?;
        let s = &self.stream;
        let ts = self
            .tree_scratch
            .take()
            .expect("tree scratch after forward");
        let mut logits = vec![0.0f32; m * self.vocab];
        let view = ts.d_logits_all.slice(0..m * self.vocab);
        s.memcpy_dtoh(&view, &mut logits)
            .map_err(|e| driver_err("tree logits dtoh", &e))?;
        self.tree_scratch = Some(ts);
        self.pending_tree = Some((m, self.cache_len, parents.to_vec()));
        Ok(logits)
    }

    /// Commit the host-chosen accepted path of the pending tree (from
    /// [`Self::tree_verify_logits`]): promote its K/V rows and advance the
    /// cache. `path` holds tree-node indices, starting at the root (0), each
    /// subsequent node a child of the previous.
    pub fn tree_commit(&mut self, path: &[usize]) -> Result<(), BackendError> {
        let Some((m, fwd_cache_len, parents)) = self.pending_tree.take() else {
            return Err(BackendError::InvalidInput(
                "tree_commit: no pending tree (call tree_verify_logits first; any \
                 intervening decode operation invalidates the provisional rows)"
                    .into(),
            ));
        };
        if fwd_cache_len != self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "tree_commit: the cache moved since the tree forward ({} -> {}) — the \
                 provisional rows were overwritten by an intervening decode operation; \
                 re-run tree_verify_logits",
                fwd_cache_len, self.cache_len
            )));
        }
        if path.is_empty() || path[0] != 0 {
            return Err(BackendError::InvalidInput(
                "tree_commit: path must start at the root (node 0)".into(),
            ));
        }
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b >= m || parents[b] as usize != a {
                return Err(BackendError::InvalidInput(format!(
                    "tree_commit: node {b} is not a child of {a} (m={m})"
                )));
            }
        }
        self.tree_promote(path)
    }

    /// Debug/test access: start a capture on the capture stream and fail it
    /// through [`capture_body`] — exercises the mid-capture error recovery
    /// (the stream must be usable afterwards).
    #[doc(hidden)]
    pub fn debug_fail_capture(&mut self) -> Result<(), BackendError> {
        let s = &self.cap_stream;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("debug begin_capture", &e))?;
        let r: Result<(), BackendError> = capture_body(s, || {
            Err(BackendError::InvalidInput(
                "injected capture failure".into(),
            ))
        });
        match r {
            Err(_) => Ok(()), // the injected error propagated; capture terminated
            Ok(()) => {
                // Unreachable with the hardcoded Err body, but end the capture
                // anyway so even a misuse of this debug hook can't wedge the
                // stream.
                let _ = s.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                Err(BackendError::Backend(
                    "debug_fail_capture: injected error vanished".into(),
                ))
            }
        }
    }

    /// Debug/test access: dtoh one K/V row of the single-sequence arena (bytes).
    #[doc(hidden)]
    pub fn debug_kv_row(&self, li: usize, row: usize, v: bool) -> Result<Vec<u8>, BackendError> {
        let rb = self.kv_width * self.kv_elem;
        let arena = if v { &self.kv_v[li] } else { &self.kv_k[li] };
        let view = arena.slice(row * rb..(row + 1) * rb);
        let mut out = vec![0u8; rb];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug kv row dtoh", &e))?;
        Ok(out)
    }

    /// Debug/test access: dtoh one K row of the single-sequence arena (bytes).
    #[doc(hidden)]
    pub fn debug_kv_k_row(&self, li: usize, row: usize) -> Result<Vec<u8>, BackendError> {
        let rb = self.kv_width * self.kv_elem;
        let view = self.kv_k[li].slice(row * rb..(row + 1) * rb);
        let mut out = vec![0u8; rb];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug kv row dtoh", &e))?;
        Ok(out)
    }

    /// Debug/test access: dtoh one K row of a batch slot (f32 bytes).
    #[doc(hidden)]
    pub fn debug_batch_kv_row(
        &self,
        batch: &BatchKv,
        li: usize,
        row_slot: usize,
        row: usize,
        v: bool,
    ) -> Result<Vec<u8>, BackendError> {
        let kw = self.kv_width;
        let off = (row_slot * batch.max_ctx + row) * kw;
        let arena = if v { &batch.kv_v[li] } else { &batch.kv_k[li] };
        let view = arena.slice(off..off + kw);
        let mut out = vec![0f32; kw];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug batch kv row dtoh", &e))?;
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Debug/test access: dtoh one K row of a batch slot (f32 bytes).
    #[doc(hidden)]
    pub fn debug_batch_kv_k_row(
        &self,
        batch: &BatchKv,
        li: usize,
        row_slot: usize,
        row: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let kw = self.kv_width;
        let off = (row_slot * batch.max_ctx + row) * kw;
        let view = batch.kv_k[li].slice(off..off + kw);
        let mut out = vec![0f32; kw];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug batch kv row dtoh", &e))?;
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Continuous-batching admission: copy this model's single-sequence KV
    /// rows `[0, len)` (every layer, K and V) into batch slot `row`'s arena.
    /// The caller prefills the prompt through the SINGLE-sequence path (the
    /// optimized prefill), then adopts the cache into the slot and
    /// [`BatchKv::set_position`]s it — zero new kernels.
    ///
    /// Phase-1 constraint: batch arenas are f32, so this requires the f32 KV
    /// rung (`kv_elem == 4`); other rungs are rejected loudly.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a bad row/len or a non-f32 KV rung.
    pub fn copy_kv_into_batch_row(
        &self,
        batch: &mut BatchKv,
        row: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        if self.kv_elem != 4 {
            return Err(BackendError::InvalidInput(
                "continuous batching requires the f32 KV rung (batch arenas are f32); \
                 unset TRITIUM_KV"
                    .into(),
            ));
        }
        if row >= batch.n {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: row {row} >= batch n {}",
                batch.n
            )));
        }
        if len > batch.max_ctx || len > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: len {len} exceeds max_ctx {}",
                batch.max_ctx.min(self.max_ctx)
            )));
        }
        if len > self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: len {len} > cache_len {} — prefill the \
                 prompt through the single-sequence path first",
                self.cache_len
            )));
        }
        let s = &self.stream;
        let bytes = len * self.kv_width * 4;
        for li in 0..self.layers.len() {
            for (src, dst) in [
                (&self.kv_k[li], &mut batch.kv_k[li]),
                (&self.kv_v[li], &mut batch.kv_v[li]),
            ] {
                let (src_ptr, sg) = src.device_ptr(s);
                let dst_off = row * batch.max_ctx * self.kv_width;
                // f32 elements → byte pointer offset ×4.
                let (dst_base, dg) = dst.device_ptr(s);
                let dst_ptr = dst_base + (dst_off * 4) as sys::CUdeviceptr;
                // SAFETY: raw byte copy between live device allocations on this
                // model's stream: src holds `len·kv_width` f32 rows from
                // position 0 (single-seq arena, byte-typed) and dst is the
                // slot's leading `len·kv_width` f32 span; sizes checked above.
                #[allow(unsafe_code)]
                unsafe { result::memcpy_dtod_async(dst_ptr, src_ptr, bytes, s.cu_stream()) }
                    .map_err(|e| driver_err("batch row adopt dtod", &e))?;
                drop(sg);
                drop(dg);
            }
        }
        // Belt-and-braces ordering: decode_batch_graph's replay already
        // syncs the default stream first, so this is redundant for the
        // current caller — kept so future capture-stream consumers can't
        // read a half-landed adoption.
        s.synchronize()
            .map_err(|e| driver_err("batch row adopt sync", &e))?;
        Ok(())
    }

    /// Promote the accepted path: node path[k] (arena slot cache_len + path[k])
    /// moves to arena row cache_len + k, then the cache advances by the path
    /// length. `path` is strictly increasing (children follow parents), so
    /// src >= dst and no promoted row is overwritten before it is read.
    fn tree_promote(&mut self, path: &[usize]) -> Result<(), BackendError> {
        let s = &self.stream;
        // Arena rows are addressed in BYTES (kv_elem = 4/2/1); a row copy is
        // dtype-agnostic. Under the i8 rung each token also owns a scale row.
        let row_bytes = self.kv_width * self.kv_elem;
        let sc_row = if self.kv_dtype.has_scales() {
            self.n_head_kv * (self.head_dim / KV_QGROUP)
        } else {
            0
        };
        for (k, &node) in path.iter().enumerate() {
            if node == k {
                continue; // already in place (chain prefix)
            }
            let src = (self.cache_len + node) * row_bytes;
            let dst = (self.cache_len + k) * row_bytes;
            for li in 0..self.layers.len() {
                for arena in [&mut self.kv_k[li], &mut self.kv_v[li]] {
                    // Copy via a device temporary: src/dst rows never overlap (path is
                    // strictly increasing, src >= dst), but the tmp keeps the copy
                    // trivially safe for any future path shape.
                    let row = {
                        let src_slice = arena.slice(src..src + row_bytes);
                        let mut tmp = s
                            .alloc_zeros::<u8>(row_bytes)
                            .map_err(|e| driver_err("tree promote tmp", &e))?;
                        s.memcpy_dtod(&src_slice, &mut tmp)
                            .map_err(|e| driver_err("tree promote read", &e))?;
                        tmp
                    };
                    let mut dst_slice = arena.slice_mut(dst..dst + row_bytes);
                    s.memcpy_dtod(&row, &mut dst_slice)
                        .map_err(|e| driver_err("tree promote write", &e))?;
                }
                if sc_row > 0 {
                    let s_src = (self.cache_len + node) * sc_row;
                    let s_dst = (self.cache_len + k) * sc_row;
                    for arena in [&mut self.kv_k_scales[li], &mut self.kv_v_scales[li]] {
                        let row = {
                            let src_slice = arena.slice(s_src..s_src + sc_row);
                            let mut tmp = s
                                .alloc_zeros::<f32>(sc_row)
                                .map_err(|e| driver_err("tree promote sc tmp", &e))?;
                            s.memcpy_dtod(&src_slice, &mut tmp)
                                .map_err(|e| driver_err("tree promote sc read", &e))?;
                            tmp
                        };
                        let mut dst_slice = arena.slice_mut(s_dst..s_dst + sc_row);
                        s.memcpy_dtod(&row, &mut dst_slice)
                            .map_err(|e| driver_err("tree promote sc write", &e))?;
                    }
                }
            }
        }
        self.cache_len += path.len();
        Ok(())
    }

    // --- batched (M>1) prefill launch helpers (safe launches; eager one-shot path) ---

    /// Tree-verify attention launch (ADR 0014): one warp per (node, head).
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_tree(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        anc: &CudaSlice<i32>,
        n_anc: &CudaSlice<i32>,
        ctx_max: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        prefix_len: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (cm_i, nh_i, nhkv_i, hd_i) = (
            ctx_max as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
        );
        let (pl_i, ma_i, m_i) = (prefix_len as i32, m as i32, m as i32);
        let total_warps = (m * n_head) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total_warps.div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(scores)
            .arg(anc)
            .arg(n_anc)
            .arg(&cm_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&pl_i)
            .arg(&ma_i)
            .arg(&m_i);
        // SAFETY: `gqa_attention_tree_f32(q, k, v, out, scores, anc, n_anc, ctx_max,
        // n_head, n_head_kv, head_dim, scale, prefix_len, max_anc, m)`; max_anc == m
        // (the host builds the ancestor table [m, m]).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree attention", &e))?;
        }
        Ok(())
    }

    /// Split tree-verify attention (head_dim % 4 == 0): a context-parallel
    /// scores fan-out (one warp per (node, head, 128-key chunk), float4 d-order
    /// chains) followed by a per-(node, head) 128-thread softmax + weighted-sum
    /// reduce. Bit-identical to `bl_attn_tree` — every rounded f32 fold keeps
    /// its order (per-key dots in d-order, the softmax sum in j-order on one
    /// thread, per-dim weighted chains in j-order).
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_tree_split(
        s: &Arc<CudaStream>,
        f_scores: &CudaFunction,
        f_reduce: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        k_scales: Option<&CudaSlice<f32>>,
        v_scales: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        anc: &CudaSlice<i32>,
        n_anc: &CudaSlice<i32>,
        ctx_max: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        prefix_len: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        // Keep in sync with `#define TREE_SCORE_CHUNK` in decode.cu.
        const TREE_SCORE_CHUNK: usize = 128;
        let (cm_i, nh_i, nhkv_i, hd_i) = (
            ctx_max as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
        );
        let (pl_i, ma_i, m_i) = (prefix_len as i32, m as i32, m as i32);
        // Upper bound over rows: every row's context is prefix_len + n_anc[row]
        // with n_anc[row] <= m; the kernels guard per-row.
        let ctx_bound = prefix_len + m;
        let cfg = LaunchConfig {
            grid_dim: (
                (m * n_head) as u32,
                (ctx_bound.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f_scores);
        l.arg(q)
            .arg(k)
            .arg(&mut *scores)
            .arg(anc)
            .arg(n_anc)
            .arg(&cm_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&pl_i)
            .arg(&ma_i)
            .arg(&m_i);
        if let Some(sc) = k_scales {
            l.arg(sc);
        }
        // SAFETY: `gqa_attention_tree_scores_g(q, k, scores, anc, n_anc, ctx_max,
        // n_head, n_head_kv, head_dim, scale, prefix_len, max_anc, m)` or the
        // `_q8` twin with trailing `k_scales`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree attention scores", &e))?;
        }
        let cfg = LaunchConfig {
            grid_dim: ((m * n_head) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (ctx_bound * 4) as u32,
        };
        let mut l = s.launch_builder(f_reduce);
        l.arg(v)
            .arg(&*scores)
            .arg(out)
            .arg(anc)
            .arg(n_anc)
            .arg(&cm_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&pl_i)
            .arg(&ma_i)
            .arg(&m_i);
        if let Some(sc) = v_scales {
            l.arg(sc);
        }
        // SAFETY: `gqa_attention_tree_reduce_g(v, scores, out, anc, n_anc, ctx_max,
        // n_head, n_head_kv, head_dim, prefix_len, max_anc, m)` (or the `_q8` twin
        // with trailing `v_scales`) with ctx_bound·4 B of dynamic shared (opt-in
        // set at load for up to max_ctx·4).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree attention reduce", &e))?;
        }
        Ok(())
    }

    /// Batched tied LM head over `m` rows (f16 table, row-tiled).
    #[allow(clippy::too_many_arguments)]
    fn bl_lm_head_tiled(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        h: &CudaSlice<f32>,
        embd_f16: &CudaSlice<u16>, // f16 bits (the kernel reinterprets as __half)
        n_embd: usize,
        vocab: usize,
        m: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i, m_i) = (n_embd as i32, vocab as i32, m as i32);
        let warps = vocab as u32;
        let cfg = LaunchConfig {
            grid_dim: (
                (warps * 32).div_ceil(256),
                (m as u32).div_ceil(LMHEAD_ROW_TILE),
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(h)
            .arg(embd_f16)
            .arg(&ne_i)
            .arg(&v_i)
            .arg(&m_i)
            .arg(logits);
        // SAFETY: `lm_head_tiled_f16(const float* h, const __half* embd, int n_embd,
        // int vocab, int m, float* logits)` — one warp per vocab row, grid.y tiles rows.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree lm_head", &e))?;
        }
        Ok(())
    }

    /// Chunked per-row argmax: the single-kernel `argmax_rows_f32` spawns only
    /// `m` blocks (129µs measured at m=13); the partial/combine pair fans each
    /// row over `ARGMAX_CHUNKS` blocks with the identical tie rule, so the
    /// result is exactly the single kernel's.
    #[allow(clippy::too_many_arguments)]
    fn bl_argmax_rows_chunked(
        s: &Arc<CudaStream>,
        f_partial: &CudaFunction,
        f_combine: &CudaFunction,
        logits: &CudaSlice<f32>,
        vocab: usize,
        m: usize,
        pvals: &mut CudaSlice<f32>,
        pidx: &mut CudaSlice<i32>,
        out: &mut CudaSlice<i32>,
    ) -> Result<(), BackendError> {
        let (v_i, m_i) = (vocab as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (m as u32, ARGMAX_CHUNKS as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f_partial);
        l.arg(logits)
            .arg(&v_i)
            .arg(&m_i)
            .arg(&mut *pvals)
            .arg(&mut *pidx);
        // SAFETY: `argmax_rows_partial_f32(logits, vocab, m, pvals, pidx)`;
        // pvals/pidx are [m, ARGMAX_CHUNKS].
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch argmax partial", &e))?;
        }
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f_combine);
        l.arg(&*pvals).arg(&*pidx).arg(&m_i).arg(out);
        // SAFETY: `argmax_rows_combine_f32(pvals, pidx, m, out)` (one warp/row).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch argmax combine", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_rmsnorm(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        eps: f32,
        n: usize,
        m: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (n_i, m_i) = (n as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let mut l = s.launch_builder(f);
        l.arg(x).arg(w).arg(&eps).arg(&n_i).arg(&m_i).arg(out);
        // SAFETY: `rmsnorm_batch_f32(const float* x, const float* w, float eps, int n, int m, float* out)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill rmsnorm", &e))?;
        }
        Ok(())
    }

    fn bl_embed(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        table: &CudaSlice<f32>,
        tokens: &CudaSlice<i32>,
        n_embd: usize,
        m: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (ne_i, m_i) = (n_embd as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (((m * n_embd) as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(table).arg(tokens).arg(&ne_i).arg(&m_i).arg(out);
        // SAFETY: `embedding_gather_batch_f32(const float* table, const int* tokens, int n_embd, int m, float* out)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill embed", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_rope(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        x: &mut CudaSlice<f32>,
        cos_t: &CudaSlice<f32>,
        sin_t: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        n_head: usize,
        head_dim: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i, m_i) = (n_head as i32, head_dim as i32, m as i32);
        let total = (m * n_head * (head_dim / 2)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(x)
            .arg(cos_t)
            .arg(sin_t)
            .arg(positions)
            .arg(&nh_i)
            .arg(&hd_i)
            .arg(&m_i);
        // SAFETY: `rope_apply_batch_f32(float* x, const float* cos, const float* sin, const int* pos, int n_head, int head_dim, int m)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill rope", &e))?;
        }
        Ok(())
    }

    fn bl_quant(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        d_in: &CudaSlice<f32>,
        k: usize,
        m: usize,
        qact: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (k_i, m_i) = (k as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(d_in).arg(&k_i).arg(&m_i).arg(qact).arg(scale);
        // SAFETY: `act_quant_batch_i8(const float* act, int k, int m, signed char* q, float* scale)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill quant", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_matmul(
        s: &Arc<CudaStream>,
        f_tiled_scaled: &CudaFunction,
        qact: &CudaSlice<i8>,
        lin: &ResidentLinear,
        scale: &CudaSlice<f32>,
        m: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (n, k, rb) = (lin.n, lin.k, lin.row_bytes);
        let (m_i, n_i, k_i, rb_i) = (m as i32, n as i32, k as i32, rb as i32);
        // Fused tiled GEMM + per-token act_scale fold (v0.6.0 opt #15).
        // grid.y = m dispatches one block-row per output row; the kernel reads
        // act_scale[mi] per row in the epilogue.
        // DP4A i8 kernel: reads each packed-int8 row directly — no dynamic shared.
        {
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(WARPS_PER_BLOCK), m as u32, 1),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(f_tiled_scaled);
            l.arg(qact)
                .arg(lin.device.as_ref())
                .arg(&lin.scales)
                .arg(scale)
                .arg(&mut *out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: `tq2_0_add_mpgemm_tiled_scaled(act, weights, scales,
            // act_scale, out, m, n, k, row_bytes)` (grid.y = m).
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch prefill tiled scaled", &e))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_kv_append(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        src: &CudaSlice<f32>,
        kv_base: &mut CudaSlice<u8>,
        cache_len: usize,
        kv_width: usize,
        m: usize,
        scales: Option<&mut CudaSlice<f32>>,
    ) -> Result<(), BackendError> {
        let (cl_i, kw_i, m_i) = (cache_len as i32, kv_width as i32, m as i32);
        let cfg = LaunchConfig {
            // The i8 kernel reduces per-group absmax in shared memory, one
            // block per row; the f32/f16 copies are elementwise. One block
            // per row is correct for all three (grid.x = m).
            grid_dim: if scales.is_some() {
                (m as u32, 1, 1)
            } else {
                (((m * kv_width) as u32).div_ceil(256), 1, 1)
            },
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(src).arg(kv_base).arg(&cl_i).arg(&kw_i).arg(&m_i);
        if let Some(sc) = scales {
            // The `_q8` twin takes the scales arena as a trailing param.
            l.arg(sc);
        }
        // SAFETY: `kv_append_batch_f32(const float* src, float* kv_base, int cache_len, int kv_width, int m)`
        // or the `_q8` twin with a trailing `float* scales` (grid = m blocks).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch kv_append (prefill/step/tree)", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_attn<KV: cudarc::driver::DeviceRepr>(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<KV>,
        v: &CudaSlice<KV>,
        k_scales: Option<&CudaSlice<f32>>,
        v_scales: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        ctx_max: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        causal_offset: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (cm_i, nh_i, nhkv_i, hd_i, co_i, m_i) = (
            ctx_max as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            causal_offset as i32,
            m as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (((m * n_head) as u32).div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(scores)
            .arg(&cm_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&co_i)
            .arg(&m_i);
        if let Some(sc) = k_scales {
            l.arg(sc);
        }
        if let Some(sc) = v_scales {
            l.arg(sc);
        }
        // SAFETY: `gqa_attention_batch_f32(q, k, v, out, scores, ctx_max, n_head, n_head_kv,
        // head_dim, scale, causal_offset, m)` or the `_q8` twin with trailing
        // `k_scales, v_scales`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill attn", &e))?;
        }
        Ok(())
    }

    fn bl_residual(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        total: usize,
    ) -> Result<(), BackendError> {
        let n_i = total as i32;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(x).arg(y).arg(&n_i);
        // SAFETY: `residual_add_f32(float* x, const float* y, int n)` over m·n_embd elements.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill residual", &e))?;
        }
        Ok(())
    }

    fn bl_relu2(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        total: usize,
    ) -> Result<(), BackendError> {
        let n_i = total as i32;
        let cfg = LaunchConfig {
            grid_dim: ((total as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(gate).arg(up).arg(&n_i);
        // SAFETY: `relu2_gate_f32(float* gate, const float* up, int n)` over m·n_ff elements.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill relu2", &e))?;
        }
        Ok(())
    }

    fn bl_lm_head_f16(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        h: &CudaSlice<f32>,
        embd_f16: &CudaSlice<u16>,
        n_embd: usize,
        vocab: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i) = (n_embd as i32, vocab as i32);
        let cfg = LaunchConfig {
            grid_dim: ((vocab as u32).div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(h).arg(embd_f16).arg(&ne_i).arg(&v_i).arg(logits);
        // SAFETY: `lm_head_warp_f16(const float* h, const __half* embd, int n_embd, int vocab, float* logits)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill lm_head", &e))?;
        }
        Ok(())
    }

    // ===================== v0.3.7 batched M=N decode =====================

    /// Allocate batched-decode state for `n` concurrent sequences: a per-sequence KV arena
    /// (`[n, max_ctx, kv_width]` per layer) + the M=N scratch, all starting empty.
    ///
    /// # Errors
    /// [`BackendError`] on a device allocation failure.
    pub fn new_batch(&self, n: usize) -> Result<BatchKv, BackendError> {
        let s = &self.stream;
        let alloc =
            |k: usize, what: &str| s.alloc_zeros::<f32>(k).map_err(|e| driver_err(what, &e));
        let mut kv_k = Vec::with_capacity(self.layers.len());
        let mut kv_v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            kv_k.push(alloc(n * self.max_ctx * self.kv_width, "batch kv_k")?);
            kv_v.push(alloc(n * self.max_ctx * self.kv_width, "batch kv_v")?);
        }
        Ok(BatchKv {
            n,
            max_ctx: self.max_ctx,
            kv_k,
            kv_v,
            positions: vec![0; n],
            d_tokens: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_tokens", &e))?,
            d_positions: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_positions", &e))?,
            d_x: alloc(n * self.n_embd, "batch d_x")?,
            d_normed: alloc(n * self.n_embd, "batch d_normed")?,
            d_q: alloc(n * self.q_width, "batch d_q")?,
            d_k: alloc(n * self.kv_width, "batch d_k")?,
            d_v: alloc(n * self.kv_width, "batch d_v")?,
            d_attn: alloc(n * self.q_width, "batch d_attn")?,
            d_attn_sn: alloc(n * self.q_width, "batch d_attn_sn")?,
            d_proj: alloc(n * self.n_embd, "batch d_proj")?,
            d_gate: alloc(n * self.n_ff, "batch d_gate")?,
            d_up: alloc(n * self.n_ff, "batch d_up")?,
            d_gate_sn: alloc(n * self.n_ff, "batch d_gate_sn")?,
            d_qact: s
                .alloc_zeros::<i8>(n * self.n_ff)
                .map_err(|e| driver_err("batch d_qact", &e))?,
            d_act_scale: alloc(n, "batch d_act_scale")?,
            d_scores: alloc(n * self.n_head * self.max_ctx, "batch d_scores")?,
            d_attn_partials: alloc(
                n * self.n_head * self.max_ctx.div_ceil(ATTN_SPLIT_CHUNK) * (self.head_dim + 2),
                "batch d_attn_partials",
            )?,
            d_h: alloc(self.n_embd, "batch d_h")?,
            d_logits: alloc(self.vocab, "batch d_logits")?,
            d_logits_batch: alloc(n * self.vocab, "batch d_logits_batch")?,
            d_argmax: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_argmax", &e))?,
            graph: None,
            graph_argmax: None,
            raw_keepalive: None,
        })
    }

    /// One batched decode step: `tokens[r]` is sequence `r`'s next token (at its own
    /// position `batch.positions[r]`). Runs the M=N forward — each row attends its OWN KV
    /// slice — appends `n` k/v rows, advances every position by 1, and returns each
    /// sequence's next-token logits `[vocab]`. Bit-identical per row to a single-sequence
    /// `step_graph` (the batch kernels share the M=1 reduction order).
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch context overflow".into(),
                ));
            }
        }
        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim, max_ctx) =
            (self.n_head, self.n_head_kv, self.head_dim, self.max_ctx);

        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        s.memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch tokens htod", &e))?;
        s.memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch pos htod", &e))?;

        Self::bl_embed(
            s,
            &self.f_embed_batch,
            &self.d_token_embd,
            &batch.d_tokens,
            n_embd,
            n,
            &mut batch.d_x,
        )?;

        for li in 0..self.layers.len() {
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &batch.d_x,
                &self.layers[li].attn_norm,
                self.rms_eps,
                n_embd,
                n,
                &mut batch.d_normed,
            )?;
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &batch.d_normed,
                n_embd,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].q,
                &batch.d_act_scale,
                n,
                &mut batch.d_q,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].k,
                &batch.d_act_scale,
                n,
                &mut batch.d_k,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].v,
                &batch.d_act_scale,
                n,
                &mut batch.d_v,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut batch.d_q,
                &self.d_cos,
                &self.d_sin,
                &batch.d_positions,
                n_head,
                head_dim,
                n,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut batch.d_k,
                &self.d_cos,
                &self.d_sin,
                &batch.d_positions,
                n_head_kv,
                head_dim,
                n,
            )?;
            Self::md_kv_append(
                s,
                &self.f_kv_append_mdecode,
                &batch.d_k,
                &mut batch.kv_k[li],
                &batch.d_positions,
                max_ctx,
                kv_width,
                n,
            )?;
            Self::md_kv_append(
                s,
                &self.f_kv_append_mdecode,
                &batch.d_v,
                &mut batch.kv_v[li],
                &batch.d_positions,
                max_ctx,
                kv_width,
                n,
            )?;
            Self::md_attn(
                s,
                &self.f_attn_split_partial,
                &self.f_attn_combine,
                &batch.d_q,
                &batch.kv_k[li],
                &batch.kv_v[li],
                &mut batch.d_attn,
                &mut batch.d_attn_partials,
                &batch.d_positions,
                max_ctx,
                n_head,
                n_head_kv,
                head_dim,
                self.attn_scale,
                n,
            )?;
            let attn_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].attn_sub_norm.as_ref()
            {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &batch.d_attn,
                    sn,
                    self.rms_eps,
                    q_width,
                    n,
                    &mut batch.d_attn_sn,
                )?;
                &batch.d_attn_sn
            } else {
                &batch.d_attn
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                attn_in,
                q_width,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].o,
                &batch.d_act_scale,
                n,
                &mut batch.d_proj,
            )?;
            Self::bl_residual(
                s,
                &self.f_residual,
                &mut batch.d_x,
                &batch.d_proj,
                n * n_embd,
            )?;

            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &batch.d_x,
                &self.layers[li].ffn_norm,
                self.rms_eps,
                n_embd,
                n,
                &mut batch.d_normed,
            )?;
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &batch.d_normed,
                n_embd,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].gate,
                &batch.d_act_scale,
                n,
                &mut batch.d_gate,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].up,
                &batch.d_act_scale,
                n,
                &mut batch.d_up,
            )?;
            Self::bl_relu2(s, &self.f_relu2, &mut batch.d_gate, &batch.d_up, n * n_ff)?;
            let down_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &batch.d_gate,
                    sn,
                    self.rms_eps,
                    n_ff,
                    n,
                    &mut batch.d_gate_sn,
                )?;
                &batch.d_gate_sn
            } else {
                &batch.d_gate
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                down_in,
                n_ff,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].down,
                &batch.d_act_scale,
                n,
                &mut batch.d_proj,
            )?;
            Self::bl_residual(
                s,
                &self.f_residual,
                &mut batch.d_x,
                &batch.d_proj,
                n * n_embd,
            )?;
        }

        // Final norm (all n rows) then per-row LM head.
        Self::bl_rmsnorm(
            s,
            &self.f_rmsnorm_batch,
            &batch.d_x,
            &self.d_output_norm,
            self.rms_eps,
            n_embd,
            n,
            &mut batch.d_normed,
        )?;
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            {
                let row = batch.d_normed.slice(r * n_embd..(r + 1) * n_embd);
                s.memcpy_dtod(&row, &mut batch.d_h)
                    .map_err(|e| driver_err("batch row copy", &e))?;
            }
            Self::bl_lm_head_f16(
                s,
                &self.f_lm_head_f16,
                &batch.d_h,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                &mut batch.d_logits,
            )?;
            let mut logits = vec![0.0f32; self.vocab];
            s.memcpy_dtoh(&batch.d_logits, &mut logits)
                .map_err(|e| driver_err("batch logits dtoh", &e))?;
            out.push(logits);
        }
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn md_kv_append(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        src: &CudaSlice<f32>,
        kv_base: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        max_ctx: usize,
        kv_width: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (mc_i, kw_i, n_i) = (max_ctx as i32, kv_width as i32, n as i32);
        let cfg = LaunchConfig {
            grid_dim: (((n * kv_width) as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(src)
            .arg(kv_base)
            .arg(positions)
            .arg(&mc_i)
            .arg(&kw_i)
            .arg(&n_i);
        // SAFETY: `kv_append_mdecode_f32(const float* src, float* kv_base, const int* pos, int max_ctx, int kv_width, int n)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch mdecode kv_append", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn md_attn(
        s: &Arc<CudaStream>,
        f_partial: &CudaFunction,
        f_combine: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        partials: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        max_ctx: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_split = max_ctx.div_ceil(ATTN_SPLIT_CHUNK);
        let (mc_i, nh_i, nhkv_i, hd_i, n_i, ns_i, ck_i) = (
            max_ctx as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            n as i32,
            n_split as i32,
            ATTN_SPLIT_CHUNK as i32,
        );
        // Partial: one warp (32 threads) per (row, head, split).
        {
            let cfg = LaunchConfig {
                grid_dim: ((n * n_head * n_split) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(f_partial);
            l.arg(q)
                .arg(k)
                .arg(v)
                .arg(&mut *partials)
                .arg(positions)
                .arg(&mc_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&scale)
                .arg(&n_i)
                .arg(&ns_i)
                .arg(&ck_i);
            // SAFETY: matches `gqa_attention_split_partial_f32(q, k, v, partials, positions,
            // max_ctx, n_head, n_head_kv, head_dim, scale, n, n_split, chunk)`; only `partials`
            // mutable; partials is `n·n_head·n_split·(head_dim+2)`; grid covers n·n_head·n_split warps.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch md split partial", &e))?;
            }
        }
        // Combine: one warp per (row, head).
        {
            let cfg = LaunchConfig {
                grid_dim: ((n * n_head) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(f_combine);
            l.arg(&*partials)
                .arg(out)
                .arg(&nh_i)
                .arg(&hd_i)
                .arg(&n_i)
                .arg(&ns_i);
            // SAFETY: matches `gqa_attention_combine_f32(partials, out, n_head, head_dim, n, n_split)`;
            // only `out` mutable; out is `n·n_head·head_dim`; grid covers n·n_head warps.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch md split combine", &e))?;
            }
        }
        Ok(())
    }

    /// **Graph-captured batched (M=N) decode** — the Track-2 perf sibling of
    /// [`decode_batch`](Self::decode_batch). The device-resident M=N body is recorded once
    /// into a CUDA graph (per batch, since the capture bakes in *these* buffers' pointers)
    /// and replayed per step, eliminating the per-kernel launch overhead that left the
    /// eager M=N path slower than the M=1 [`step_graph`](Self::step_graph). Bit-identical
    /// to `decode_batch` per row — the graph replays the exact same kernels in the same
    /// order over the same buffers; only the launch mechanism differs — gated by
    /// `cuda_batch_decode_graph_matches_eager`. The LM head stays eager (per-row), like
    /// `decode_batch`.
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch_graph(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch_graph expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch_graph token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch_graph context overflow".into(),
                ));
            }
        }

        // Lazily load the raw batch kernels, then capture this batch's graph (per-N).
        if self.batch_raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
        }
        if batch.graph.is_none() {
            let g = self.record_graph_batch(batch, false)?;
            batch.graph = Some(SendGraph(g));
            // Keep the modules the captured graph references alive for as long as this
            // batch lives (see `BatchKv::raw_keepalive`).
            batch.raw_keepalive = self.batch_raw.clone();
        }

        // Drain any pending default-stream work before the graph (on `cap_stream`) touches
        // the shared batch buffers, exactly as `step_graph` does for the M=1 path.
        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch graph pre default sync", &e))?;

        // Upload this step's tokens + positions on the capture stream, ordered before the
        // replay (the captured embed/rope/kv/attn read them as stable pointers — the M=N
        // analogue of the M=1 `d_ctrl`).
        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        self.cap_stream
            .memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch graph tokens htod", &e))?;
        self.cap_stream
            .memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch graph pos htod", &e))?;
        batch
            .graph
            .as_ref()
            .expect("graph captured above")
            .launch()
            .map_err(|e| driver_err("batch graph launch", &e))?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("batch graph sync", &e))?;

        // Final norm landed in `d_normed`; run the per-row LM head eagerly (one warp head
        // per row over the f16 token table), mirroring `decode_batch`'s tail bit-for-bit.
        // The tail stays on `cap_stream` — the same stream the graph ran on — so the read
        // of `d_normed` is plain stream-ordered after the graph's write, with no
        // cross-stream handoff (the M=1 `step_graph` keeps its post-graph dtoh on
        // `cap_stream` for the same reason).
        let s = &self.cap_stream;
        let n_embd = self.n_embd;
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            {
                let row = batch.d_normed.slice(r * n_embd..(r + 1) * n_embd);
                s.memcpy_dtod(&row, &mut batch.d_h)
                    .map_err(|e| driver_err("batch graph row copy", &e))?;
            }
            Self::bl_lm_head_f16(
                s,
                &self.f_lm_head_f16,
                &batch.d_h,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                &mut batch.d_logits,
            )?;
            let mut logits = vec![0.0f32; self.vocab];
            s.memcpy_dtoh(&batch.d_logits, &mut logits)
                .map_err(|e| driver_err("batch graph logits dtoh", &e))?;
            out.push(logits);
        }
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(out)
    }

    /// **On-device-sampling batched decode** — the serving fast path. Same M=N forward as
    /// [`decode_batch_graph`](Self::decode_batch_graph), but the captured graph also runs a
    /// batched LM head + greedy argmax, so only `n` token ids (`n·4` bytes) come back instead
    /// of `n·vocab·4` bytes of logits (the readback that caps the eager-tail path). Each
    /// returned token equals the host `sample_greedy` of the logits the logits-path would
    /// produce (the batched LM head is bit-identical per row to the single-row kernel; the
    /// argmax tie rule matches `max_by`), gated by `cuda_batch_decode_graph_argmax_matches_greedy`.
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch_graph_argmax(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<u32>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch_graph_argmax expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch_graph_argmax token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch_graph_argmax context overflow".into(),
                ));
            }
        }

        if self.batch_raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
        }
        if batch.graph_argmax.is_none() {
            let g = self.record_graph_batch(batch, true)?;
            batch.graph_argmax = Some(g);
            batch.raw_keepalive = self.batch_raw.clone();
        }

        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch argmax pre default sync", &e))?;

        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        self.cap_stream
            .memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch argmax tokens htod", &e))?;
        self.cap_stream
            .memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch argmax pos htod", &e))?;
        batch
            .graph_argmax
            .as_ref()
            .expect("argmax graph captured above")
            .launch()
            .map_err(|e| driver_err("batch argmax graph launch", &e))?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("batch argmax graph sync", &e))?;

        // The graph wrote the n greedy token ids into d_argmax; copy back just those.
        let mut ids = vec![0i32; n];
        self.cap_stream
            .memcpy_dtoh(&batch.d_argmax, &mut ids)
            .map_err(|e| driver_err("batch argmax dtoh", &e))?;
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(ids.into_iter().map(|t| t as u32).collect())
    }

    /// Capture the tree-verify trunk (embed → 30 layers, NO final norm / LM
    /// head — those run eagerly at the real node count) for a padded tree of
    /// `bucket` nodes, raw-launched on `cap_stream`. Mirrors the eager
    /// `tree_verify_greedy` trunk OP-FOR-OP: every kernel is the same function
    /// with the same geometry, except the three ctrl-driven twins
    /// (`kv_append_tree_g`, `gqa_attention_tree_{scores,reduce}_ctrl_g`) that
    /// read [prefix_len, real_m] from `TreeGraphs::d_ctrl` at replay. Real
    /// rows' math is row-independent, so their results are bit-identical to
    /// the eager path (gated by `cuda_tree_verify_greedy_lossless`).
    fn record_graph_tree(
        &self,
        ts: &TreeScratch,
        bucket: usize,
    ) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let mb = bucket;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let raw = self.batch_raw();
        let (f_kv, f_sc, f_rd) = (
            raw.kv_append_tree,
            raw.attn_tree_scores_ctrl,
            raw.attn_tree_reduce_ctrl,
        );

        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            n: l.n,
            k: l.k,
            rb: l.row_bytes,
        };
        struct TreeLayerPtrs {
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
        let layers: Vec<TreeLayerPtrs> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| TreeLayerPtrs {
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
        let tg = self
            .tree_graphs
            .as_ref()
            .expect("TreeGraphs created before record");
        let d_ctrl = dptr(&tg.d_ctrl, s);
        let (d_tok, d_pos, d_anc, d_nanc) = (
            dptr(&ts.d_tok, s),
            dptr(&ts.d_pos, s),
            dptr(&ts.d_anc, s),
            dptr(&ts.d_nanc, s),
        );
        let (d_x, d_normed, d_q, d_k, d_v) = (
            dptr(&ts.d_x, s),
            dptr(&ts.d_normed, s),
            dptr(&ts.d_q, s),
            dptr(&ts.d_k, s),
            dptr(&ts.d_v, s),
        );
        let (d_attn, d_attn_sn, d_proj, d_gate, d_up, d_gate_sn) = (
            dptr(&ts.d_attn, s),
            dptr(&ts.d_attn_sn, s),
            dptr(&ts.d_proj, s),
            dptr(&ts.d_gate, s),
            dptr(&ts.d_up, s),
            dptr(&ts.d_gate_sn, s),
        );
        let (d_qact, d_act_scale, d_scores) = (
            dptr(&ts.d_qact, s),
            dptr(&ts.d_act_scale, s),
            dptr(&ts.d_scores, s),
        );
        let (d_cos, d_sin, d_token_embd) = (
            dptr(&self.d_cos, s),
            dptr(&self.d_sin, s),
            dptr(&self.d_token_embd, s),
        );

        // The ctrl-driven launches, closed over the baked dims.
        let cs = self.cap_stream.cu_stream();
        let (kw_i, mb_i) = (kv_width as i32, mb as i32);
        let (stride_i, nh_i, nhkv_i, hd_i, ma_i) = (
            self.max_ctx as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            mb as i32,
        );
        let scale = self.attn_scale;
        let kv_append = |src: sys::CUdeviceptr, base: sys::CUdeviceptr| {
            let grid = (((mb * kv_width) as u32).div_ceil(256), 1, 1);
            let mut params = [pp(&src), pp(&base), pp(&d_ctrl), pp(&kw_i), pp(&mb_i)];
            raw_launch(f_kv, grid, (256, 1, 1), 0, cs, &mut params)
        };
        let attn = |kv_k: sys::CUdeviceptr, kv_v: sys::CUdeviceptr| {
            const TREE_SCORE_CHUNK: usize = 128; // keep in sync with decode.cu
            let grid = (
                (mb * n_head) as u32,
                (self.max_ctx.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            );
            let mut params = [
                pp(&d_q),
                pp(&kv_k),
                pp(&d_scores),
                pp(&d_anc),
                pp(&d_nanc),
                pp(&d_ctrl),
                pp(&stride_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&scale),
                pp(&ma_i),
                pp(&mb_i),
            ];
            raw_launch(f_sc, grid, (32, 1, 1), 0, cs, &mut params)?;
            let grid = ((mb * n_head) as u32, 1, 1);
            let smem = (self.max_ctx * 4) as u32;
            let mut params = [
                pp(&kv_v),
                pp(&d_scores),
                pp(&d_attn),
                pp(&d_anc),
                pp(&d_nanc),
                pp(&d_ctrl),
                pp(&stride_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&ma_i),
                pp(&mb_i),
            ];
            raw_launch(f_rd, grid, (128, 1, 1), smem, cs, &mut params)
        };

        // Drain the device_ptr events so the capture carries no pre-capture deps.
        s.synchronize()
            .map_err(|e| driver_err("tree pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("tree pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("tree begin_capture", &e))?;

        capture_body(s, || {
            self.gb_embed(d_token_embd, d_tok, d_x, mb)?;
            for lp in &layers {
                self.gb_rmsnorm(d_x, lp.attn_norm, n_embd, d_normed, mb)?;
                self.gb_quant(d_normed, n_embd, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.q, d_qact, d_act_scale, d_q, mb)?;
                self.gb_matmul(&lp.k, d_qact, d_act_scale, d_k, mb)?;
                self.gb_matmul(&lp.v, d_qact, d_act_scale, d_v, mb)?;
                self.gb_rope(d_q, d_cos, d_sin, d_pos, n_head, head_dim, mb)?;
                self.gb_rope(d_k, d_cos, d_sin, d_pos, n_head_kv, head_dim, mb)?;
                kv_append(d_k, lp.kv_k)?;
                kv_append(d_v, lp.kv_v)?;
                attn(lp.kv_k, lp.kv_v)?;
                let attn_in = if let Some(sn) = lp.attn_sub_norm {
                    self.gb_rmsnorm(d_attn, sn, q_width, d_attn_sn, mb)?;
                    d_attn_sn
                } else {
                    d_attn
                };
                self.gb_quant(attn_in, q_width, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.o, d_qact, d_act_scale, d_proj, mb)?;
                self.gb_residual(d_x, d_proj, mb * n_embd)?;

                self.gb_rmsnorm(d_x, lp.ffn_norm, n_embd, d_normed, mb)?;
                self.gb_quant(d_normed, n_embd, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.gate, d_qact, d_act_scale, d_gate, mb)?;
                self.gb_matmul(&lp.up, d_qact, d_act_scale, d_up, mb)?;
                self.gb_relu2(d_gate, d_up, mb * n_ff)?;
                let down_in = if let Some(sn) = lp.ffn_sub_norm {
                    self.gb_rmsnorm(d_gate, sn, n_ff, d_gate_sn, mb)?;
                    d_gate_sn
                } else {
                    d_gate
                };
                self.gb_quant(down_in, n_ff, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.down, d_qact, d_act_scale, d_proj, mb)?;
                self.gb_residual(d_x, d_proj, mb * n_embd)?;
            }
            Ok(())
        })?;

        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| driver_err("tree end_capture", &e))?
            .ok_or_else(|| BackendError::Backend("tree graph capture produced no graph".into()))?;
        Ok(graph)
    }

    fn batch_raw(&self) -> &BatchRawKernels {
        self.batch_raw
            .as_ref()
            .expect("batch raw kernels loaded before record")
    }

    /// Extract every batch + weight buffer's stable device pointer (guards dropped here,
    /// outside capture), then capture the full M=N forward via raw launches on
    /// `cap_stream`. Mirrors [`record_graph`](Self::record_graph) for the batched path,
    /// reading the **per-batch** KV arenas and the **unfused** q/k/v/gate/up projections
    /// (the eager `decode_batch` is unfused — fusing is the follow-on).
    ///
    /// `with_head`: when true, the capture also runs the batched LM head + greedy argmax
    /// after the final RMSNorm (the on-device-sampling graph, ending at `d_argmax`); when
    /// false it ends at `d_normed` and the LM head is the eager per-row tail.
    fn record_graph_batch(
        &self,
        batch: &BatchKv,
        with_head: bool,
    ) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let n = batch.n;
        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            n: l.n,
            k: l.k,
            rb: l.row_bytes,
        };
        let layers: Vec<BatchLayerPtrs> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| BatchLayerPtrs {
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
                kv_k: dptr(&batch.kv_k[li], s),
                kv_v: dptr(&batch.kv_v[li], s),
            })
            .collect();
        let p = BatchPtrs {
            d_tokens: dptr(&batch.d_tokens, s),
            d_positions: dptr(&batch.d_positions, s),
            d_x: dptr(&batch.d_x, s),
            d_normed: dptr(&batch.d_normed, s),
            d_q: dptr(&batch.d_q, s),
            d_k: dptr(&batch.d_k, s),
            d_v: dptr(&batch.d_v, s),
            d_attn: dptr(&batch.d_attn, s),
            d_attn_sn: dptr(&batch.d_attn_sn, s),
            d_proj: dptr(&batch.d_proj, s),
            d_gate: dptr(&batch.d_gate, s),
            d_up: dptr(&batch.d_up, s),
            d_gate_sn: dptr(&batch.d_gate_sn, s),
            d_qact: dptr(&batch.d_qact, s),
            d_act_scale: dptr(&batch.d_act_scale, s),
            d_scores: dptr(&batch.d_scores, s),
            d_attn_partials: dptr(&batch.d_attn_partials, s),
            d_cos: dptr(&self.d_cos, s),
            d_sin: dptr(&self.d_sin, s),
            d_token_embd: dptr(&self.d_token_embd, s),
            d_output_norm: dptr(&self.d_output_norm, s),
            d_token_embd_f16: dptr(&self.d_token_embd_f16, s),
            d_logits_batch: dptr(&batch.d_logits_batch, s),
            d_argmax: dptr(&batch.d_argmax, s),
        };
        // Drain the events the device_ptr extraction recorded, so the capture (raw
        // launches only) carries no pre-capture dependency.
        s.synchronize()
            .map_err(|e| driver_err("batch pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("batch begin_capture", &e))?;

        capture_body(s, || {
            // The exact op order of `decode_batch`, all raw-launched on `cap_stream`.
            self.gb_embed(p.d_token_embd, p.d_tokens, p.d_x, n)?;
            for lp in &layers {
                self.gb_layer(&p, lp, n)?;
            }
            self.gb_rmsnorm(p.d_x, p.d_output_norm, self.n_embd, p.d_normed, n)?;
            if with_head {
                // Batched LM head over all n rows → d_logits_batch, then per-row greedy argmax →
                // d_argmax. Both raw-launched into the capture; only d_argmax is read back.
                self.gb_lm_head_batch(p.d_normed, p.d_token_embd_f16, p.d_logits_batch, n)?;
                self.gb_argmax(p.d_logits_batch, p.d_argmax, n)?;
            }
            Ok(())
        })?;

        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| driver_err("batch end_capture", &e))?
            .ok_or_else(|| BackendError::Backend("batch graph capture produced no graph".into()))?;
        Ok(graph)
    }

    /// One transformer block of the M=N forward, raw-launched into the capture. Mirrors
    /// the per-layer body of [`decode_batch`](Self::decode_batch) op-for-op.
    fn gb_layer(&self, p: &BatchPtrs, l: &BatchLayerPtrs, n: usize) -> Result<(), BackendError> {
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);

        // pre-norm attention. q/k/v share ONE quant of d_normed, then three unfused GEMMs.
        self.gb_rmsnorm(p.d_x, l.attn_norm, n_embd, p.d_normed, n)?;
        self.gb_quant(p.d_normed, n_embd, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.q, p.d_qact, p.d_act_scale, p.d_q, n)?;
        self.gb_matmul(&l.k, p.d_qact, p.d_act_scale, p.d_k, n)?;
        self.gb_matmul(&l.v, p.d_qact, p.d_act_scale, p.d_v, n)?;
        self.gb_rope(p.d_q, p.d_cos, p.d_sin, p.d_positions, n_head, head_dim, n)?;
        self.gb_rope(
            p.d_k,
            p.d_cos,
            p.d_sin,
            p.d_positions,
            n_head_kv,
            head_dim,
            n,
        )?;
        self.gb_kv_append(p.d_k, l.kv_k, p.d_positions, kv_width, n)?;
        self.gb_kv_append(p.d_v, l.kv_v, p.d_positions, kv_width, n)?;
        self.gb_attn(
            p.d_q,
            l.kv_k,
            l.kv_v,
            p.d_attn,
            p.d_attn_partials,
            p.d_positions,
            n,
        )?;
        let attn_in = if let Some(sn) = l.attn_sub_norm {
            self.gb_rmsnorm(p.d_attn, sn, q_width, p.d_attn_sn, n)?;
            p.d_attn_sn
        } else {
            p.d_attn
        };
        self.gb_quant(attn_in, q_width, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.o, p.d_qact, p.d_act_scale, p.d_proj, n)?;
        self.gb_residual(p.d_x, p.d_proj, n * n_embd)?;

        // pre-norm ReLU² MLP. gate/up unfused; relu2 writes gate = relu(gate)² ⊙ up.
        self.gb_rmsnorm(p.d_x, l.ffn_norm, n_embd, p.d_normed, n)?;
        self.gb_quant(p.d_normed, n_embd, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.gate, p.d_qact, p.d_act_scale, p.d_gate, n)?;
        self.gb_matmul(&l.up, p.d_qact, p.d_act_scale, p.d_up, n)?;
        self.gb_relu2(p.d_gate, p.d_up, n * n_ff)?;
        let down_in = if let Some(sn) = l.ffn_sub_norm {
            self.gb_rmsnorm(p.d_gate, sn, n_ff, p.d_gate_sn, n)?;
            p.d_gate_sn
        } else {
            p.d_gate
        };
        self.gb_quant(down_in, n_ff, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.down, p.d_qact, p.d_act_scale, p.d_proj, n)?;
        self.gb_residual(p.d_x, p.d_proj, n * n_embd)?;
        Ok(())
    }

    // Raw-launch helpers for the batched capture (`gb_*`): each mirrors the matching safe
    // `bl_*`/`md_*` helper 1:1 — same grid/block/smem and the same kernel-param order —
    // but builds the params from pre-extracted device pointers and raw-launches on
    // `cap_stream`. `n` is the batch (row) count.

    fn gb_embed(
        &self,
        table: sys::CUdeviceptr,
        tokens: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (ne_i, n_i) = (self.n_embd as i32, n as i32);
        let grid = (((n * self.n_embd) as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&table), pp(&tokens), pp(&ne_i), pp(&n_i), pp(&out)];
        raw_launch(
            self.batch_raw().embed,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn gb_rmsnorm(
        &self,
        x: sys::CUdeviceptr,
        w: sys::CUdeviceptr,
        dim: usize,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let (dim_i, n_i) = (dim as i32, n as i32);
        let smem = (dim * 4) as u32;
        let mut params = [pp(&x), pp(&w), pp(&eps), pp(&dim_i), pp(&n_i), pp(&out)];
        raw_launch(
            self.batch_raw().rmsnorm,
            (n as u32, 1, 1),
            (256, 1, 1),
            smem,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn gb_quant(
        &self,
        d_in: sys::CUdeviceptr,
        k: usize,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (k_i, n_i) = (k as i32, n as i32);
        let mut params = [pp(&d_in), pp(&k_i), pp(&n_i), pp(&d_qact), pp(&d_act_scale)];
        raw_launch(
            self.batch_raw().quant,
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn gb_matmul(
        &self,
        lin: &LinPtrs,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let cs = self.cap_stream.cu_stream();
        let (m_i, n_out_i, k_i, rb_i) = (n as i32, lin.n as i32, lin.k as i32, lin.rb as i32);
        // v1.x: one fused i8 dp4a launch — the `_scaled` epilogue folds the per-row
        // act_scale, replacing the former tiled + scale_mul_batch pair. The multiply
        // order is unchanged ((acc·weight_scale)·act_scale) and the int32 contraction
        // is exact, so the batch-graph argmax lockstep gate is unaffected.
        let grid = ((lin.n as u32).div_ceil(WARPS_PER_BLOCK), n as u32, 1);
        let mut params = [
            pp(&d_qact),
            pp(&lin.w),
            pp(&lin.sc),
            pp(&d_act_scale),
            pp(&d_out),
            pp(&m_i),
            pp(&n_out_i),
            pp(&k_i),
            pp(&rb_i),
        ];
        raw_launch(
            self.batch_raw().tiled_scaled,
            grid,
            (WARPS_PER_BLOCK * 32, 1, 1),
            0,
            cs,
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gb_rope(
        &self,
        x: sys::CUdeviceptr,
        cos_t: sys::CUdeviceptr,
        sin_t: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        n_head: usize,
        head_dim: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i, n_i) = (n_head as i32, head_dim as i32, n as i32);
        let total = (n * n_head * (head_dim / 2)) as u32;
        let grid = (total.div_ceil(256), 1, 1);
        let mut params = [
            pp(&x),
            pp(&cos_t),
            pp(&sin_t),
            pp(&positions),
            pp(&nh_i),
            pp(&hd_i),
            pp(&n_i),
        ];
        raw_launch(
            self.batch_raw().rope,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn gb_kv_append(
        &self,
        src: sys::CUdeviceptr,
        kv_base: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        kv_width: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (mc_i, kw_i, n_i) = (self.max_ctx as i32, kv_width as i32, n as i32);
        let grid = (((n * kv_width) as u32).div_ceil(256), 1, 1);
        let mut params = [
            pp(&src),
            pp(&kv_base),
            pp(&positions),
            pp(&mc_i),
            pp(&kw_i),
            pp(&n_i),
        ];
        raw_launch(
            self.batch_raw().kv_append,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gb_attn(
        &self,
        q: sys::CUdeviceptr,
        k: sys::CUdeviceptr,
        v: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        partials: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_split = self.max_ctx.div_ceil(ATTN_SPLIT_CHUNK);
        let (mc_i, nh_i, nhkv_i, hd_i, n_i, ns_i, ck_i) = (
            self.max_ctx as i32,
            self.n_head as i32,
            self.n_head_kv as i32,
            self.head_dim as i32,
            n as i32,
            n_split as i32,
            ATTN_SPLIT_CHUNK as i32,
        );
        let scale = self.attn_scale;
        let cs = self.cap_stream.cu_stream();
        {
            let grid = ((n * self.n_head * n_split) as u32, 1, 1);
            let mut params = [
                pp(&q),
                pp(&k),
                pp(&v),
                pp(&partials),
                pp(&positions),
                pp(&mc_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&scale),
                pp(&n_i),
                pp(&ns_i),
                pp(&ck_i),
            ];
            raw_launch(
                self.batch_raw().attn_split_partial,
                grid,
                (32, 1, 1),
                0,
                cs,
                &mut params,
            )?;
        }
        {
            let grid = ((n * self.n_head) as u32, 1, 1);
            let mut params = [
                pp(&partials),
                pp(&out),
                pp(&nh_i),
                pp(&hd_i),
                pp(&n_i),
                pp(&ns_i),
            ];
            raw_launch(
                self.batch_raw().attn_combine,
                grid,
                (32, 1, 1),
                0,
                cs,
                &mut params,
            )?;
        }
        Ok(())
    }

    fn gb_residual(
        &self,
        x: sys::CUdeviceptr,
        y: sys::CUdeviceptr,
        total: usize,
    ) -> Result<(), BackendError> {
        let total_i = total as i32;
        let grid = ((total as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&x), pp(&y), pp(&total_i)];
        raw_launch(
            self.batch_raw().residual,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn gb_relu2(
        &self,
        gate: sys::CUdeviceptr,
        up: sys::CUdeviceptr,
        total: usize,
    ) -> Result<(), BackendError> {
        let total_i = total as i32;
        let grid = ((total as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&gate), pp(&up), pp(&total_i)];
        raw_launch(
            self.batch_raw().relu2,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Batched LM head over all `n` rows: `d_normed[n, n_embd] · token_embd_f16 → d_logits[n, vocab]`.
    /// One warp per vocab row, computing `LMHEAD_ROW_TILE` output rows per launch so the embd
    /// table is read once per row-tile (not once per row); `grid.y = ceil(n / TILE)`.
    /// Bit-identical per row to the single-row head.
    fn gb_lm_head_batch(
        &self,
        h: sys::CUdeviceptr,
        embd: sys::CUdeviceptr,
        d_logits: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i, n_i) = (self.n_embd as i32, self.vocab as i32, n as i32);
        let grid = (
            (self.vocab as u32).div_ceil(8),
            (n as u32).div_ceil(LMHEAD_ROW_TILE),
            1,
        );
        let mut params = [
            pp(&h),
            pp(&embd),
            pp(&ne_i),
            pp(&v_i),
            pp(&n_i),
            pp(&d_logits),
        ];
        raw_launch(
            self.batch_raw().lm_head_tiled,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Per-row greedy argmax `d_logits[n, vocab] → d_out[n]` (i32). One block per row.
    fn gb_argmax(
        &self,
        d_logits: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (v_i, n_i) = (self.vocab as i32, n as i32);
        let grid = (n as u32, 1, 1);
        let mut params = [pp(&d_logits), pp(&v_i), pp(&n_i), pp(&d_out)];
        raw_launch(
            self.batch_raw().argmax,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Load the raw kernels (once) and record + instantiate the decode graph.
    fn capture_graph(&mut self) -> Result<(), BackendError> {
        if self.raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            let raw = RawGraphKernels::load(&ctx, self.kv_dtype)?;
            // The warp attention kernel's dynamic shared (`max_ctx` f32 scores) needs
            // the same over-48-KiB opt-in on this second JIT's function handle as the
            // eager path's `f_attn` (see `build_decode_model`).
            // NOTE: the attribute is a CAP, not a floor — sizing it below a
            // launch's dynamic-shared request makes that launch INVALID_VALUE
            // (bisected the hard way; keep per-function sizing if a kernel
            // ever stages more than one score bank).
            for (func, units) in [
                (raw.attn_g, self.max_ctx),
                (raw.attn_reduce_g, self.max_ctx),
            ] {
                attn_shared_opt_in(units, |bytes| {
                    // SAFETY: `func` is a valid function handle from the module just
                    // loaded above; setting a function attribute is not a launch.
                    #[allow(unsafe_code)]
                    unsafe {
                        result::function::set_function_attribute(
                            func,
                            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes,
                        )
                    }
                })?;
            }
            self.raw = Some(raw);
        }
        let graph = self.record_graph()?;
        self.graph = Some(SendGraph(graph));
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
                o: lin(&l.o),
                down: lin(&l.down),
                qkv: lin(&l.qkv),
                gateup: lin(&l.gateup),
                kv_k: dptr(&self.kv_k[li], s),
                kv_v: dptr(&self.kv_v[li], s),
                kv_k_sc: if self.kv_dtype.has_scales() {
                    dptr(&self.kv_k_scales[li], s)
                } else {
                    0
                },
                kv_v_sc: if self.kv_dtype.has_scales() {
                    dptr(&self.kv_v_scales[li], s)
                } else {
                    0
                },
            })
            .collect();
        let p = GraphPtrs {
            d_x: dptr(&self.d_x, s),
            d_normed: dptr(&self.d_normed, s),
            d_qkv: dptr(&self.d_qkv, s),
            d_gateup: dptr(&self.d_gateup, s),
            d_attn: dptr(&self.d_attn, s),
            d_attn_sn: dptr(&self.d_attn_sn, s),
            d_proj_out: dptr(&self.d_proj_out, s),
            d_gate_sn: dptr(&self.d_gate_sn, s),
            d_scores: dptr(&self.d_scores, s),
            d_logits: dptr(&self.d_logits, s),
            d_qact: dptr(&self.d_qact, s),
            d_act_scale: dptr(&self.d_act_scale, s),
            d_ctrl: dptr(&self.d_ctrl, s),
            d_cos: dptr(&self.d_cos, s),
            d_sin: dptr(&self.d_sin, s),
            d_token_embd: dptr(&self.d_token_embd, s),
            d_token_embd_f16: dptr(&self.d_token_embd_f16, s),
            d_output_norm: dptr(&self.d_output_norm, s),
        };
        // Drain the events the device_ptr extraction recorded, so the capture (which
        // uses only raw launches) carries no pre-capture dependency.
        s.synchronize()
            .map_err(|e| driver_err("pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("decode begin_capture", &e))?;

        capture_body(s, || {
            // The exact op order of `step` + `layer`, all raw-launched on `cap_stream`.
            self.g_embed(p.d_token_embd, p.d_ctrl, p.d_x)?;
            for lp in &layers {
                self.g_layer(&p, lp)?;
            }
            self.g_rmsnorm(p.d_x, p.d_output_norm, self.n_embd, p.d_normed)?;
            self.g_lm_head(p.d_normed, p.d_token_embd_f16, p.d_logits)?;
            Ok(())
        })?;

        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| driver_err("decode end_capture", &e))?
            .ok_or_else(|| {
                BackendError::Backend("decode graph capture produced no graph".into())
            })?;
        Ok(graph)
    }

    /// One transformer block, raw-launched into the capture. Mirrors [`layer`](Self::layer).
    fn g_layer(&self, p: &GraphPtrs, l: &LayerPtrs) -> Result<(), BackendError> {
        let (n_embd, q_width, kv_width) = (self.n_embd, self.q_width, self.kv_width);

        // pre-norm attention. q/k/v are FUSED into one GEMM over `d_normed`; q/k/v are then
        // offset slices of the `[q_width + 2·kv_width]` output `d_qkv` (f32 → 4 bytes/elt).
        // Fused rmsnorm+quant (v0.7.0): eliminates intermediate d_normed + 1 launch.
        self.g_rmsnorm_quant(p.d_x, l.attn_norm, n_embd, p.d_qact, p.d_act_scale)?;
        self.g_matmul(&l.qkv, p.d_qact, p.d_act_scale, p.d_qkv)?;
        let q_ptr = p.d_qkv;
        let knew_ptr = p.d_qkv + (q_width * 4) as sys::CUdeviceptr;
        let vnew_ptr = p.d_qkv + ((q_width + kv_width) * 4) as sys::CUdeviceptr;
        // Fused rope(q) + rope(k) + append(k) + append(v): one node, not four
        // (v1.x node-count opt; values bit-identical to the unfused sequence).
        self.g_rope_kv(
            q_ptr, knew_ptr, vnew_ptr, l.kv_k, l.kv_v, l.kv_k_sc, l.kv_v_sc, p.d_cos, p.d_sin,
            p.d_ctrl,
        )?;
        self.g_attn(
            q_ptr, l.kv_k, l.kv_v, l.kv_k_sc, l.kv_v_sc, p.d_attn, p.d_scores, p.d_ctrl,
        )?;
        if let Some(sn) = l.attn_sub_norm {
            // Fused sub-norm + quant: skips intermediate attn_sn buffer.
            self.g_rmsnorm_quant(p.d_attn, sn, q_width, p.d_qact, p.d_act_scale)?;
        } else {
            self.g_quant(p.d_attn, q_width, p.d_qact, p.d_act_scale)?;
        }
        // Fused O proj GEMM + residual add (v0.7.0 Phase 2).
        // Pass d_x as both residual and output → in-place: d_x += GEMM.
        self.g_matmul_residual(&l.o, p.d_qact, p.d_act_scale, p.d_x, p.d_x)?;

        // pre-norm ReLU² MLP. gate/up are FUSED; gate/up are halves of `d_gateup` [2·n_ff].
        let n_ff = self.n_ff;
        // Fused rmsnorm+quant (v0.7.0).
        self.g_rmsnorm_quant(p.d_x, l.ffn_norm, n_embd, p.d_qact, p.d_act_scale)?;
        self.g_matmul(&l.gateup, p.d_qact, p.d_act_scale, p.d_gateup)?;
        let gate_ptr = p.d_gateup;
        let up_ptr = p.d_gateup + (n_ff * 4) as sys::CUdeviceptr;
        self.g_relu2(gate_ptr, up_ptr, n_ff)?; // gate = relu(gate)² ⊙ up, in place
        if let Some(sn) = l.ffn_sub_norm {
            // Fused sub-norm + quant.
            self.g_rmsnorm_quant(gate_ptr, sn, n_ff, p.d_qact, p.d_act_scale)?;
        } else {
            self.g_quant(gate_ptr, n_ff, p.d_qact, p.d_act_scale)?;
        }
        // Fused down proj GEMM + residual add (v0.7.0 Phase 2).
        // Pass d_x as both residual and output → in-place: d_x += GEMM.
        self.g_matmul_residual(&l.down, p.d_qact, p.d_act_scale, p.d_x, p.d_x)?;
        Ok(())
    }

    // Raw-launch helpers (build the kernel_params array from pre-extracted device
    // pointers + scalar locals; only `raw_launch` is unsafe). `cs` = capture stream.

    fn g_rmsnorm(
        &self,
        x: sys::CUdeviceptr,
        w: sys::CUdeviceptr,
        n: usize,
        out: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let n_i = n as i32;
        // `rmsnorm_shared_f32` stages the `n`-float input row into dynamic shared memory.
        let smem = (n * 4) as u32;
        let mut params = [pp(&x), pp(&w), pp(&eps), pp(&n_i), pp(&out)];
        raw_launch(
            self.raw().rmsnorm,
            (1, 1, 1),
            (256, 1, 1),
            smem,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// On-device A8 quant of an activation row (`k`-wide) → `d_qact` + `d_act_scale`. Split
    /// out from the GEMM so projections sharing an input (q/k/v and gate/up both read
    /// `d_normed`) quantize it **once** instead of per-GEMM — fewer graph nodes.
    fn g_quant(
        &self,
        d_in: sys::CUdeviceptr,
        k: usize,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let k_i = k as i32;
        let mut params = [pp(&d_in), pp(&k_i), pp(&d_qact), pp(&d_act_scale)];
        raw_launch(
            self.raw().act_quant,
            (1, 1, 1),
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Fused RMSNorm + A8 quant (v0.7.0): reads `x`, writes `d_qact` + `d_act_scale`
    /// directly. Eliminates the intermediate `d_normed` buffer and one kernel launch.
    /// Dynamic shared memory = n * 4 bytes.
    fn g_rmsnorm_quant(
        &self,
        x: sys::CUdeviceptr,
        w: sys::CUdeviceptr,
        n: usize,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let n_i = n as i32;
        let smem = (n * 4) as u32;
        let mut params = [
            pp(&x),
            pp(&w),
            pp(&eps),
            pp(&n_i),
            pp(&d_qact),
            pp(&d_act_scale),
        ];
        raw_launch(
            self.raw().rmsnorm_quant,
            (1, 1, 1),
            (RMSNORM_QUANT_THREADS, 1, 1),
            smem,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Tiled add-only GEMM + the per-token scale fold, consuming a pre-quantized `d_qact`
    /// (+ its `d_act_scale`). `g_quant` must have run on the matching input first.
    fn g_matmul(
        &self,
        lin: &LinPtrs,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let cs = self.cap_stream.cu_stream();
        let (n_i, k_i, m_i, rb_i) = (lin.n as i32, lin.k as i32, 1i32, lin.rb as i32);
        // Fused tiled GEMM + act_scale fold (v0.6.0 opt #15).
        // Single launch replaces the former tiled + scale_mul pair.
        // DP4A i8 kernel: reads the packed-int8 row directly — no dynamic shared.
        {
            let grid = ((lin.n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1);
            let smem = 0u32;
            let mut params = [
                pp(&d_qact),
                pp(&lin.w),
                pp(&lin.sc),
                pp(&d_act_scale),
                pp(&d_out),
                pp(&m_i),
                pp(&n_i),
                pp(&k_i),
                pp(&rb_i),
            ];
            raw_launch(
                self.raw().tiled_scaled,
                grid,
                (WARPS_PER_BLOCK * 32, 1, 1),
                smem,
                cs,
                &mut params,
            )?;
        }
        Ok(())
    }

    /// Fused tiled GEMM + act_scale fold + residual add (v0.7.0 Phase 2).
    /// Epilogue: `out[mi,ni] = residual[mi,ni] + acc * scales[ni] * act_scale[mi]`.
    /// Eliminates the separate `residual_add_f32` launch + its memory pass.
    fn g_matmul_residual(
        &self,
        lin: &LinPtrs,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        d_residual: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let cs = self.cap_stream.cu_stream();
        let (n_i, k_i, m_i, rb_i) = (lin.n as i32, lin.k as i32, 1i32, lin.rb as i32);
        let grid = ((lin.n as u32).div_ceil(WARPS_PER_BLOCK), 1, 1);
        // DP4A i8 kernel: reads the packed-int8 row directly — no dynamic shared.
        let smem = 0u32;
        let mut params = [
            pp(&d_qact),
            pp(&lin.w),
            pp(&lin.sc),
            pp(&d_act_scale),
            pp(&d_residual),
            pp(&d_out),
            pp(&m_i),
            pp(&n_i),
            pp(&k_i),
            pp(&rb_i),
        ];
        raw_launch(
            self.raw().tiled_scaled_residual,
            grid,
            (WARPS_PER_BLOCK * 32, 1, 1),
            smem,
            cs,
            &mut params,
        )
    }

    /// Fused rope(q)+rope(k)+append(k)+append(v) — one graph node per layer.
    #[allow(clippy::too_many_arguments)]
    fn g_rope_kv(
        &self,
        q: sys::CUdeviceptr,
        knew: sys::CUdeviceptr,
        vnew: sys::CUdeviceptr,
        kv_k: sys::CUdeviceptr,
        kv_v: sys::CUdeviceptr,
        kv_k_sc: sys::CUdeviceptr,
        kv_v_sc: sys::CUdeviceptr,
        cos_t: sys::CUdeviceptr,
        sin_t: sys::CUdeviceptr,
        ctrl: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let (nh_i, nhkv_i, hd_i, kw_i) = (
            self.n_head as i32,
            self.n_head_kv as i32,
            self.head_dim as i32,
            self.kv_width as i32,
        );
        let half = self.head_dim / 2;
        let total = (self.n_head * half + self.n_head_kv * half + self.kv_width) as u32;
        // The i8 twin reduces per-group absmax over the STAGED rotated k (+ v)
        // in shared memory, so it runs block 0 = q rotation, block 1 = k/v
        // stage + quantize + store (dynamic shared = 2·kv_width·4 B); the
        // f32/f16 kernels are elementwise over `total` threads.
        let (grid, smem) = if self.kv_dtype.has_scales() {
            ((2u32, 1, 1), (2 * self.kv_width * 4) as u32)
        } else {
            ((total.div_ceil(256), 1, 1), 0u32)
        };
        let mut params = vec![
            pp(&q),
            pp(&knew),
            pp(&vnew),
            pp(&kv_k),
            pp(&kv_v),
            pp(&cos_t),
            pp(&sin_t),
            pp(&ctrl),
            pp(&nh_i),
            pp(&nhkv_i),
            pp(&hd_i),
            pp(&kw_i),
        ];
        if self.kv_dtype.has_scales() {
            params.push(pp(&kv_k_sc));
            params.push(pp(&kv_v_sc));
        }
        // SAFETY: `rope_kv_fused_g(q, k, v, kv_k_base, kv_v_base, cos, sin, ctrl,
        // n_head, n_head_kv, head_dim, kv_width)` or the `_q8` twin with trailing
        // `k_scales, v_scales`.
        raw_launch(
            self.raw().rope_kv,
            grid,
            (256, 1, 1),
            smem,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    #[allow(dead_code)] // superseded by g_rope_kv in the capture; kept for non-fused debugging
    fn g_rope(
        &self,
        x: sys::CUdeviceptr,
        cos_t: sys::CUdeviceptr,
        sin_t: sys::CUdeviceptr,
        ctrl: sys::CUdeviceptr,
        n_head: usize,
        head_dim: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i) = (n_head as i32, head_dim as i32);
        let total = (n_head * (head_dim / 2)) as u32;
        let grid = (total.div_ceil(256), 1, 1);
        let mut params = [
            pp(&x),
            pp(&cos_t),
            pp(&sin_t),
            pp(&ctrl),
            pp(&nh_i),
            pp(&hd_i),
        ];
        raw_launch(
            self.raw().rope_g,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    // Superseded by g_rope_kv in the capture; kept for non-fused debugging.
    // NOTE: f32-ONLY — under i8 `raw().kv_append` is the 5-param one-block-
    // per-row `kv_append_q8`; this helper's 4-param elementwise launch would
    // be params-array UB. Assert kv_dtype == F32 before resurrecting.
    #[allow(dead_code)]
    fn g_kv_append(
        &self,
        src: sys::CUdeviceptr,
        kv_base: sys::CUdeviceptr,
        ctrl: sys::CUdeviceptr,
        kv_width: usize,
    ) -> Result<(), BackendError> {
        let kw_i = kv_width as i32;
        let grid = ((kv_width as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&src), pp(&kv_base), pp(&ctrl), pp(&kw_i)];
        raw_launch(
            self.raw().kv_append,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn g_attn(
        &self,
        q: sys::CUdeviceptr,
        k: sys::CUdeviceptr,
        v: sys::CUdeviceptr,
        k_sc: sys::CUdeviceptr,
        v_sc: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        scores: sys::CUdeviceptr,
        ctrl: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let (mc_i, nh_i, nhkv_i, hd_i) = (
            self.max_ctx as i32,
            self.n_head as i32,
            self.n_head_kv as i32,
            self.head_dim as i32,
        );
        let scale = self.attn_scale;
        let cs = self.cap_stream.cu_stream();
        if self.head_dim.is_multiple_of(4) {
            // v1.x split pair — see `launch_attention` for the geometry rationale.
            {
                let grid = (
                    self.n_head as u32,
                    (self.max_ctx.div_ceil(ATTN_SCORE_CHUNK)) as u32,
                    1,
                );
                let mut params = vec![
                    pp(&q),
                    pp(&k),
                    pp(&scores),
                    pp(&ctrl),
                    pp(&mc_i),
                    pp(&nh_i),
                    pp(&nhkv_i),
                    pp(&hd_i),
                    pp(&scale),
                ];
                if self.kv_dtype.has_scales() {
                    params.push(pp(&k_sc));
                }
                raw_launch(
                    self.raw().attn_scores_g,
                    grid,
                    (32, 1, 1),
                    0,
                    cs,
                    &mut params,
                )?;
            }
            let grid = (self.n_head as u32, 1, 1);
            let smem = (self.max_ctx * 4) as u32;
            let mut params = vec![
                pp(&v),
                pp(&scores),
                pp(&out),
                pp(&ctrl),
                pp(&mc_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
            ];
            if self.kv_dtype.has_scales() {
                params.push(pp(&v_sc));
            }
            return raw_launch(
                self.raw().attn_reduce_g,
                grid,
                (ATTN_REDUCE_THREADS, 1, 1),
                smem,
                cs,
                &mut params,
            );
        }
        // Legacy geometry (head_dim % 4 != 0): one warp-block per head, scores in
        // dynamic shared (`max_ctx * 4` bytes); the global `scores` arg is unused.
        let grid = (self.n_head as u32, 1, 1);
        let smem = (self.max_ctx * 4) as u32;
        let mut params = [
            pp(&q),
            pp(&k),
            pp(&v),
            pp(&out),
            pp(&scores),
            pp(&ctrl),
            pp(&mc_i),
            pp(&nh_i),
            pp(&nhkv_i),
            pp(&hd_i),
            pp(&scale),
        ];
        raw_launch(self.raw().attn_g, grid, (32, 1, 1), smem, cs, &mut params)
    }

    #[allow(dead_code)] // graph-decode residual; wired in as the CUDA Graphs path lands
    fn g_residual(
        &self,
        x: sys::CUdeviceptr,
        y: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let grid = ((n as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&x), pp(&y), pp(&n_i)];
        raw_launch(
            self.raw().residual,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn g_relu2(
        &self,
        gate: sys::CUdeviceptr,
        up: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_i = n as i32;
        let grid = ((n as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&gate), pp(&up), pp(&n_i)];
        raw_launch(
            self.raw().relu2,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn g_embed(
        &self,
        table: sys::CUdeviceptr,
        ctrl: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let ne_i = self.n_embd as i32;
        let grid = ((self.n_embd as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&table), pp(&ctrl), pp(&ne_i), pp(&out)];
        raw_launch(
            self.raw().embed_g,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    fn g_lm_head(
        &self,
        h: sys::CUdeviceptr,
        embd: sys::CUdeviceptr,
        logits: sys::CUdeviceptr,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i) = (self.n_embd as i32, self.vocab as i32);
        // One WARP per vocab row: 256-thread block = 8 warps, so grid covers ceil(vocab/8).
        let grid = ((self.vocab as u32).div_ceil(8), 1, 1);
        let mut params = [pp(&h), pp(&embd), pp(&ne_i), pp(&v_i), pp(&logits)];
        raw_launch(
            self.raw().lm_head,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }
}

mod graph_raw;
use graph_raw::*;
pub use graph_raw::BatchKv;

#[cfg(test)]
mod tests;
