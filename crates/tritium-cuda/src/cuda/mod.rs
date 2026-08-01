//! GPU host side for the CUDA backend. Compiled only with `--features cuda`.
//!
//! ## Module map (P2a split of the former 11.8k-line cuda.rs)
//!
//! - **mod.rs (here)**: `CudaDecodeModel` core — the eager M=1 step, prefill,
//!   single-sequence graph capture (`g_*` builders), the shared raw-launch
//!   helpers (`bl_*`, used by prefill + tree + batch), `CudaBuffer`, the
//!   decode-model specs and resident structs.
//! - **backend.rs**: `CudaBackend` — context/stream ownership, PTX loading,
//!   mpgemm dispatch (tiled/IMMA/sparse/SALT), uploads, autotune glue,
//!   registration. Every `cudarc` driver error maps to [`BackendError`]
//!   (allocation failures as [`BackendError::OutOfMemory`]) so the backend
//!   never panics on a device failure.
//! - **tree.rs / batch.rs**: sibling `impl CudaDecodeModel` blocks — BASTION
//!   tree-verify and M=N continuous batching. Tree graph capture reuses
//!   batch's `gb_*` builders (pub(super); recorded in ADR 0022).
//! - **graph_raw.rs**: `cuLaunchKernel` capture plumbing + `BatchKv`.
//! - **consts.rs / kv.rs**: kernel symbols + launch geometry + embedded PTX;
//!   the `KvDtype` rung ladder and its single `pick` dispatch point.
//! - **telemetry.rs**: synchronized device-memory samples and async-pool
//!   high-water evidence tied to one backend and CUDA device identity.
//! - **tests.rs**: the GPU conformance/parity suite.
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
#[cfg(feature = "device-loss-qualification")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr,
    DriverError, LaunchConfig, PushKernelArg, result, sys,
};
use cudarc::nvrtc::Ptx;
use std::ffi::{CString, c_void};

use tritium_core::{GemmShape, TernaryFormat};
use tritium_format::{IMMA_K, IMMA_N, IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks};
use tritium_runtime::BackendEntry;
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, MpGemm, MpGemmProjectedVjp, TernaryBackend,
};

use crate::autotune::{
    CacheKey, CandidateResult, ShapeBucket, TileConfig, cache_dir, tune_or_load,
};
use crate::codegen::{JIT_KERNEL_NAME, compile_imma};

mod consts;
pub use consts::KV_PAGE_TOKENS;
mod kv;

use consts::*;
use kv::*;

/// Process-local, one-shot arm for destructive CUDA context-loss qualification.
/// Only `tritium-serve`'s explicitly enabled SIGUSR2 listener can set it.
#[cfg(feature = "device-loss-qualification")]
static QUALIFICATION_CONTEXT_LOSS_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Arm one destructive context-loss injection for release qualification.
///
/// This function does not touch CUDA and is safe from an async signal listener:
/// the next SALT V2 execution on this process consumes the arm and launches a
/// real device `trap`, making the CUDA context unusable until process restart.
/// Returns `true` only for the first outstanding request.
#[cfg(feature = "device-loss-qualification")]
pub fn request_destructive_context_loss_for_qualification() -> bool {
    QUALIFICATION_CONTEXT_LOSS_REQUESTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

#[cfg(feature = "device-loss-qualification")]
fn take_destructive_context_loss_qualification_request() -> bool {
    QUALIFICATION_CONTEXT_LOSS_REQUESTED.swap(false, Ordering::AcqRel)
}

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
    /// TQ1_0 (A2): packed bytes per row (`num_blocks(k) * TQ1_0_BLOCK_BYTES`),
    /// consumed natively by the tq1 decode kernels.
    Tq1_0 { row_bytes: usize },
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
    /// `(row_bytes, is_tq1)` for the decode-resident formats; `None` for IMMA.
    pub(crate) fn decode_row_bytes(&self) -> Option<(usize, bool)> {
        match self.stride {
            Stride::Tq2_0 { row_bytes } => Some((row_bytes, false)),
            Stride::Tq1_0 { row_bytes } => Some((row_bytes, true)),
            Stride::I2sInt8 { .. } => None,
        }
    }
}

mod backend;
// Lib code no longer references backend items directly (post-P2a/A2 churn),
// but cuda::tests reaches them through `use super::*` via this glob.
#[cfg_attr(not(test), allow(unused_imports))]
use backend::*;
pub use backend::{
    CudaBackend, SaltResidentLinear, SaltV2Forward, SaltV2ForwardMode, SaltV2ForwardReceipt,
    SaltV2GatherReceipt, SaltV2ResidentAllocationReceipt, SaltV2ResidentTensor,
};
#[allow(unused_imports)] // TrainingSaltLinear is the Track D tape seam.
pub(crate) use backend::{EmbedSegments, TrainingSaltLinear};
mod telemetry;
pub use telemetry::{CudaDeviceIdentity, CudaMemorySnapshot, CudaMemoryTelemetry};

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

/// ADR 0026 Track P: the I2sInt8 tile-interleaved twin of a linear's TQ2_0
/// rows, consumed by the IMMA tensor-core mpGEMM at prefill M ≥ the tuned
/// crossover. ~0.25 B/weight beside the ~0.258 B/weight TQ2_0 rows; built
/// only for the per-projection linears the prefill path launches.
#[derive(Debug)]
struct ImmaShadow {
    device: CudaSlice<u8>,
    /// `ceil(k / IMMA_K)` — the packed k-tile stride the kernel takes.
    num_ktiles: i32,
}

/// One ternary projection, resident: the shared device weight + device scales.
#[derive(Debug)]
struct ResidentLinear {
    device: Arc<CudaSlice<u8>>,
    scales: CudaSlice<f32>,
    n: usize,
    k: usize,
    row_bytes: usize,
    /// Zero-block skip bitmap (A1b): one bit per 256-trit block per row, SET
    /// = all-zero (skipped by the `_sparse` decode GEMM). Uploaded at build
    /// only when the tensor's zero-block fraction clears
    /// [`BITMAP_MIN_ZERO_BLOCK_FRAC`]; `None` ⇒ the kernel gets a NULL
    /// pointer (bit-identical dense behavior by the kernel's contract).
    bitmap: Option<CudaSlice<u32>>,
    /// A2: true = TQ1_0-packed rows (the tq1 kernel twins); false = TQ2_0.
    tq1: bool,
    /// ADR 0026 Track P: IMMA twin of the rows (prefill tensor-core path).
    imma: Option<ImmaShadow>,
}

/// Minimum all-zero-block fraction before a skip bitmap is worth carrying
/// (census: BitNet ffn gate/up ≈1%, attention 0%; SALT students higher).
const BITMAP_MIN_ZERO_BLOCK_FRAC: f64 = 0.005;

/// dtoh the packed rows and build the zero-block bitmap when it clears the
/// threshold (load-time only; one pass over the packed bytes).
fn maybe_bitmap_from_host(
    stream: &Arc<CudaStream>,
    host: &[u8],
    n: usize,
    k: usize,
) -> Result<Option<CudaSlice<u32>>, BackendError> {
    let row_bytes = host.len().checked_div(n).unwrap_or(0);
    let bm = tritium_format::compute_zero_bitmaps(host, n, k, row_bytes)
        .map_err(|e| BackendError::InvalidInput(format!("zero bitmap: {e}")))?;
    // Counting over n*words_per_row words vs blocks = n*nb is exact because
    // compute_zero_bitmap never sets padding bits (loop bound block_idx < nb).
    let set: u64 = bm.iter().map(|w| u64::from(w.count_ones())).sum();
    let blocks = (n * k.div_ceil(256)) as u64;
    if blocks == 0 || (set as f64 / blocks as f64) < BITMAP_MIN_ZERO_BLOCK_FRAC {
        return Ok(None);
    }
    stream
        .clone_htod(&bm)
        .map(Some)
        .map_err(|e| driver_err("bitmap htod", &e))
}

/// TQ2_0 packed rows → the I2sInt8 tile interleave (the IMMA kernel's weight
/// layout). Trits-level: block scales are DISCARDED — both GEMM paths read
/// the per-channel `scales` array as the only weight scale, so the shadow
/// carries codes only. Host-side, load-time only.
fn imma_shadow_bytes(
    host_rows: &[u8],
    n: usize,
    k: usize,
    row_bytes: usize,
) -> Result<Vec<u8>, BackendError> {
    let nb = k.div_ceil(256);
    let mut trits = vec![0i8; n * k];
    let mut row = vec![tritium_core::Trit::ZERO; k];
    let mut block_scales = vec![half::f16::ZERO; nb];
    for ni in 0..n {
        tritium_format::unpack_tq2_0_row(
            &host_rows[ni * row_bytes..(ni + 1) * row_bytes],
            &mut row,
            &mut block_scales,
        )
        .map_err(|e| BackendError::InvalidInput(format!("imma shadow unpack: {e}")))?;
        for (dst, t) in trits[ni * k..(ni + 1) * k].iter_mut().zip(&row) {
            *dst = t.get();
        }
    }
    Ok(pack_i2s_int8_tiles(&trits, n, k))
}

impl ResidentLinear {
    /// Build from a borrowed spec: share the device weight (`Arc` clone, no
    /// re-upload) and upload the per-channel scales once.
    /// `enable_bitmap`: compute the A1b zero-block skip bitmap. Only linears
    /// launched through the decode graph's `g_matmul` consume it (today: the
    /// FUSED qkv/gateup) — o/down go through the dense residual kernel and
    /// the individual linears feed dense batch/tree paths, so building
    /// bitmaps for them is dead dtoh + scan work (review N1).
    /// `enable_imma` (ADR 0026 Track P): also build the I2sInt8 IMMA shadow
    /// (TQ2 rows only; TQ1 prefill stays dp4a). One dtoh serves both the
    /// bitmap scan and the shadow repack.
    fn build(
        stream: &Arc<CudaStream>,
        spec: &DecodeLinearSpec,
        enable_bitmap: bool,
        enable_imma: bool,
    ) -> Result<Self, BackendError> {
        let buf = spec
            .weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| {
                BackendError::InvalidInput("decode weight is not a CudaBuffer".into())
            })?;
        let (row_bytes, tq1) = buf
            .decode_row_bytes()
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
        let device = buf.device_arc();
        // The skip bitmap and the IMMA shadow are TQ2 geometries; TQ1 rows go
        // dense/dp4a. One dtoh of the packed rows serves both consumers.
        let (bitmap, imma) = if (enable_bitmap || enable_imma) && !tq1 {
            let mut host = vec![0u8; n * row_bytes];
            stream
                .memcpy_dtoh(device.as_ref(), &mut host)
                .map_err(|e| driver_err("resident rows dtoh", &e))?;
            let bitmap = if enable_bitmap {
                maybe_bitmap_from_host(stream, &host, n, k)?
            } else {
                None
            };
            let imma = if enable_imma {
                let bytes = imma_shadow_bytes(&host, n, k, row_bytes)?;
                Some(ImmaShadow {
                    device: stream
                        .clone_htod(&bytes)
                        .map_err(|e| driver_err("imma shadow htod", &e))?,
                    num_ktiles: k.div_ceil(tritium_format::IMMA_K) as i32,
                })
            } else {
                None
            };
            (bitmap, imma)
        } else {
            (None, None)
        };
        Ok(Self {
            device,
            scales,
            n,
            k,
            row_bytes,
            bitmap,
            tq1,
            imma,
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
        let (row_bytes, tq1) = bufs[0]
            .decode_row_bytes()
            .ok_or(BackendError::UnsupportedFormat(TernaryFormat::I2sInt8))?;
        let mut total_n = 0usize;
        for (b, p) in bufs.iter().zip(parts) {
            let (n, bk) = b.dims();
            if bk != k || b.decode_row_bytes() != Some((row_bytes, tq1)) || p.scales.len() != n {
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
        let bitmap = if tq1 {
            None // skip bitmaps are TQ2-geometry; TQ1 rows run dense
        } else {
            let mut host = vec![0u8; total_n * row_bytes];
            stream
                .memcpy_dtoh(&device, &mut host)
                .map_err(|e| driver_err("fused rows dtoh", &e))?;
            maybe_bitmap_from_host(stream, &host, total_n, k)?
        };
        Ok(Self {
            device: Arc::new(device),
            scales: d_scales,
            n: total_n,
            k,
            row_bytes,
            bitmap,
            tq1,
            // The fused pair feeds the M=1 decode graph only — no IMMA twin
            // (prefill launches the per-projection linears).
            imma: None,
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
// L2 (ADR 0032): 4 and 12 added — adaptive-k drafts live at m 5..9, where
// the old ladder padded m=9 to 16 rows (1.78x trunk FLOPs off one extra
// draft token). Capture cost is once per bucket actually used.
const TREE_BUCKETS: [usize; 7] = [4, 8, 12, 16, 24, 32, 48];
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

/// A fully device-resident BitNet decoder. One [`step`](CudaDecodeModel::step) is a
/// single-token (M=1) forward run entirely on the GPU. See the section banner above.
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
    /// A2: TQ1-native twin picked when a linear's rows are TQ1-packed.
    f_tq1_tiled_scaled: CudaFunction,
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
    /// v2 order-preserving prefill attention (one block per (row, head),
    /// shared-staged K + shared scores in place of the global scratch;
    /// bit-identical per (row, head) to `f_attn_batch`). `None` when the KV
    /// dtype has no v2 twin (i8/t2 rungs) or `TRITIUM_ATTN_V2=0`; dispatch
    /// additionally requires `head_dim <= ATTN_V2_HDMAX` and
    /// `causal_offset + m <= ATTN_V2_MAX_CTX` (the kernel's shared budget).
    f_attn_batch_v2: Option<CudaFunction>,
    /// v3 Q-blocked prefill attention (ATTN_V3_BQ rows share each staged K/V
    /// chunk; scores in the global scratch so no ctx bound; bit-identical to
    /// `f_attn_batch`/v2 per (row, head)). Preferred over v2 when present;
    /// `None` for i8/t2 KV or `TRITIUM_ATTN_V3=0`.
    f_attn_batch_v3: Option<CudaFunction>,
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
    /// Lazy m=1 argmax scratch for `step_graph_argmax`:
    /// (pvals `[1, ARGMAX_CHUNKS]`, pidx `[1, ARGMAX_CHUNKS]`, out `[1]`).
    am_scratch: Option<(CudaSlice<f32>, CudaSlice<i32>, CudaSlice<i32>)>,
    /// L1' chained-draft glue (ADR 0032): the `draft_chain_advance` kernel and
    /// its lazy scratch — (`chain_out` `[DRAFT_CHAIN_MAX]` i32, `halt` `[1]` i32).
    f_draft_chain_advance: CudaFunction,
    chain_scratch: Option<(CudaSlice<i32>, CudaSlice<i32>)>,
    f_lm_head_tiled: CudaFunction,
    f_lm_head_f16: CudaFunction,
    f_kv_append_mdecode: CudaFunction,
    f_kv_append_mdecode_paged: CudaFunction,
    f_attn_split_partial: CudaFunction,
    f_attn_split_partial_paged: CudaFunction,
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
    /// ADR 0026 Track P: build-time-resolved IMMA tile functions, keyed
    /// `(n, k, m_log2 bucket)`. The `CudaFunction`s are Arc-backed and hold
    /// their modules alive; a lookup miss at dispatch falls back to dp4a.
    /// Empty when shadows are disabled (`TRITIUM_IMMA_PREFILL=0` /
    /// `TRITIUM_IMMA_TUNE=off` / TQ1 models).
    imma_funcs: HashMap<(usize, usize, u32), (TileConfig, CudaFunction)>,
    /// Prefill M at/above which the IMMA path dispatches
    /// (`TRITIUM_IMMA_MIN_M`, default 32 — the tuned dp4a/IMMA crossover).
    imma_min_m: usize,
}

impl core::fmt::Debug for CudaDecodeModel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaDecodeModel")
            .field("n_embd", &self.n_embd)
            .field("n_head", &self.n_head)
            .field("n_head_kv", &self.n_head_kv)
            .field("head_dim", &self.head_dim)
            .field("n_ff", &self.n_ff)
            .field("vocab", &self.vocab)
            .field("max_ctx", &self.max_ctx)
            .field("cache_len", &self.cache_len)
            .field("layers", &self.layers.len())
            .field("kv_dtype", &self.kv_dtype)
            .finish_non_exhaustive()
    }
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
            &self.f_tq1_tiled_scaled,
            &self.layers[li].q,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_q,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.f_tq1_tiled_scaled,
            &self.layers[li].k,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_knew,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.f_tq1_tiled_scaled,
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
            &self.f_tq1_tiled_scaled,
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
            &self.f_tq1_tiled_scaled,
            &self.layers[li].gate,
            &self.d_qact,
            &self.d_act_scale,
            &mut self.d_gate,
        )?;
        Self::gemm_prequantized(
            &self.stream,
            &self.f_tiled_scaled,
            &self.f_tq1_tiled_scaled,
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
            &self.f_tq1_tiled_scaled,
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
        f_tq1_tiled_scaled: &CudaFunction,
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
            let mut l = stream.launch_builder(if lin.tq1 {
                f_tq1_tiled_scaled
            } else {
                f_tiled_scaled
            });
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
        f_tq1_tiled_scaled: &CudaFunction,
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
            let mut l = stream.launch_builder(if lin.tq1 {
                f_tq1_tiled_scaled
            } else {
                f_tiled_scaled
            });
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
        self.step_graph_replay(token, pos)?;
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

    /// One graph decode step returning the **argmax token id** instead of the
    /// logits: replay the captured graph, run the chunked device argmax on the
    /// resident `d_logits` (the partial/combine pair whose tie rule the batch
    /// argmax gate pins equal to the host `sample_greedy` scan), and cross the
    /// host boundary with 4 bytes instead of `vocab * 4` (~513 KB at the
    /// LLaMA-3 vocab). The drafter's in-loop greedy steps are the consumer
    /// (round 25): per drafted token the logits download + host scan + Vec
    /// alloc dominated the graph replay itself.
    ///
    /// # Errors
    /// As [`step_graph`](Self::step_graph); an error after the graph launch
    /// leaves `cache_len` unadvanced, so the next step rewrites the same KV
    /// row (state stays consistent).
    pub fn step_graph_argmax(&mut self, token: u32, pos: usize) -> Result<u32, BackendError> {
        self.step_graph_replay(token, pos)?;
        self.ensure_am_scratch()?;
        let (pvals, pidx, out) = self.am_scratch.as_mut().expect("just seeded");
        // Same stream as the graph replay, so ordering needs no sync.
        Self::bl_argmax_rows_chunked(
            &self.cap_stream,
            &self.f_argmax_partial,
            &self.f_argmax_combine,
            &self.d_logits,
            self.vocab,
            1,
            pvals,
            pidx,
            out,
        )?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("decode graph argmax sync", &e))?;
        let mut id = [0i32; 1];
        self.cap_stream
            .memcpy_dtoh(&*out, &mut id)
            .map_err(|e| driver_err("decode graph argmax dtoh", &e))?;
        self.cache_len += 1;
        Ok(id[0] as u32)
    }

    /// The shared [`step_graph`]/[`step_graph_argmax`] core: guards, lazy
    /// capture, ctrl rewrite, and the graph launch — WITHOUT the trailing
    /// sync/readback or the `cache_len` advance (callers own both).
    fn step_graph_replay(&mut self, token: u32, pos: usize) -> Result<(), BackendError> {
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
        Ok(())
    }

    /// L1' chained k-step greedy draft (ADR 0032): replay the captured M=1
    /// decode graph `k` times back-to-back on `cap_stream`, feeding each
    /// step's device argmax into the next step's control block ON DEVICE
    /// (`draft_chain_advance`) — one host round-trip (a single trailing sync
    /// plus a `k`-int readback) per CHAIN instead of one per token. The
    /// measured per-token host round-trip (~1.2 ms of two syncs, ctrl H2D and
    /// readback) was the dominant drafter-step term; this collapses it to ~1/k.
    ///
    /// Returns the drafted ids, truncated at the first EOS **inclusive** —
    /// exactly the host loop's "the draft believes the turn ends here"
    /// semantics (the EOS is drafted, never fed; post-halt replays rewrite
    /// the same KV row with identical values, unobserved past the watermark).
    /// The KV watermark advances by the returned length. Drafts are
    /// bit-identical to `k` calls of [`step_graph_argmax`] (same graph, same
    /// argmax kernels, same inputs — gated by the acceptance test).
    ///
    /// # Errors
    /// As [`step_graph`](Self::step_graph), plus `k` guards; `k` must fit the
    /// remaining KV room in FULL (`cache_len + k <= max_ctx`) so no replay
    /// can overflow the arena mid-chain.
    pub fn draft_chain(
        &mut self,
        token: u32,
        pos: usize,
        k: usize,
        eos: u32,
    ) -> Result<Vec<u32>, BackendError> {
        if k == 0 || k > DRAFT_CHAIN_MAX {
            return Err(BackendError::InvalidInput(format!(
                "draft_chain k={k} out of range (1..={DRAFT_CHAIN_MAX})"
            )));
        }
        if self.cache_len + k > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "draft_chain overflow: cache_len={} + k={k} > max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }
        // Guards + capture + pre-graph drain + ctrl H2D + replay 0 (consumes
        // `token` at `pos`). Includes the pos == cache_len invariant.
        self.step_graph_replay(token, pos)?;
        self.ensure_am_scratch()?;
        if self.chain_scratch.is_none() {
            let chain = self
                .cap_stream
                .alloc_zeros::<i32>(DRAFT_CHAIN_MAX)
                .map_err(|e| driver_err("draft chain alloc", &e))?;
            let halt = self
                .cap_stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| driver_err("draft halt alloc", &e))?;
            self.chain_scratch = Some((chain, halt));
        }
        {
            // Reset the halt flag (stream-ordered before replay 0's argmax
            // consumers — it is only read by chain_advance, launched below).
            let (_, halt) = self.chain_scratch.as_mut().expect("just seeded");
            self.cap_stream
                .memcpy_htod(&[0i32], halt)
                .map_err(|e| driver_err("draft halt htod", &e))?;
        }

        for step in 0..k {
            if step > 0 {
                // Replays 1..k read the ctrl that chain_advance(step-1) wrote
                // on-device; all stream-ordered on cap_stream.
                self.graph
                    .as_ref()
                    .expect("graph captured in step_graph_replay")
                    .launch()
                    .map_err(|e| driver_err("draft chain graph launch", &e))?;
            }
            let (pvals, pidx, out) = self.am_scratch.as_mut().expect("seeded above");
            Self::bl_argmax_rows_chunked(
                &self.cap_stream,
                &self.f_argmax_partial,
                &self.f_argmax_combine,
                &self.d_logits,
                self.vocab,
                1,
                pvals,
                pidx,
                out,
            )?;
            let (chain, halt) = self.chain_scratch.as_mut().expect("seeded above");
            let (step_i, eos_i) = (step as i32, eos as i32);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let out_ref: &CudaSlice<i32> = &*out;
            let mut l = self.cap_stream.launch_builder(&self.f_draft_chain_advance);
            l.arg(&mut self.d_ctrl)
                .arg(out_ref)
                .arg(&mut *chain)
                .arg(&mut *halt)
                .arg(&step_i)
                .arg(&eos_i);
            // SAFETY: `draft_chain_advance(int* ctrl, const int* am_out,
            // int* chain_out, int* halt, int step, int eos)` — d_ctrl is the
            // 4-int control block, am_out the 1-int argmax result, chain_out
            // DRAFT_CHAIN_MAX ints with step < k <= DRAFT_CHAIN_MAX, halt one
            // int. Single thread; scalars outlive the launch.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch draft_chain_advance", &e))?;
            }
        }

        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("draft chain sync", &e))?;
        let mut ids = vec![0i32; k];
        {
            let (chain, _) = self.chain_scratch.as_ref().expect("seeded above");
            let view = chain
                .try_slice(0..k)
                .ok_or_else(|| BackendError::Backend("draft chain slice".into()))?;
            self.cap_stream
                .memcpy_dtoh(&view, &mut ids)
                .map_err(|e| driver_err("draft chain dtoh", &e))?;
        }
        // -1 sentinels mark frozen (post-EOS) steps; everything before the
        // first sentinel was genuinely fed/produced.
        let out: Vec<u32> = ids
            .iter()
            .take_while(|&&x| x >= 0)
            .map(|&x| x as u32)
            .collect();
        self.cache_len += out.len();
        Ok(out)
    }

    /// Seed the lazy m=1 argmax scratch (shared by [`step_graph_argmax`] and
    /// [`draft_chain`]).
    fn ensure_am_scratch(&mut self) -> Result<(), BackendError> {
        if self.am_scratch.is_none() {
            let pvals = self
                .cap_stream
                .alloc_zeros::<f32>(ARGMAX_CHUNKS)
                .map_err(|e| driver_err("argmax pvals alloc", &e))?;
            let pidx = self
                .cap_stream
                .alloc_zeros::<i32>(ARGMAX_CHUNKS)
                .map_err(|e| driver_err("argmax pidx alloc", &e))?;
            let out = self
                .cap_stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| driver_err("argmax out alloc", &e))?;
            self.am_scratch = Some((pvals, pidx, out));
        }
        Ok(())
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
        // Attention dispatch for this prefill — invariant across layers
        // (head_dim, causal_offset, m are fixed), so decide once. Priority:
        // v3 (Q-blocked, needs the global scores scratch, no ctx bound) ->
        // v2 (shared scores, ctx-capped) -> rev-1. Only the pure-v2 case
        // skips the [m, n_head, ctx_max] scratch (review 98ab046 nit N1).
        let attn_v3: Option<CudaFunction> = self
            .f_attn_batch_v3
            .clone()
            .filter(|_| head_dim <= consts::ATTN_V2_HDMAX);
        let attn_v2: Option<CudaFunction> = if attn_v3.is_some() {
            None
        } else {
            self.f_attn_batch_v2.clone().filter(|_| {
                head_dim <= consts::ATTN_V2_HDMAX && causal_offset + m <= consts::ATTN_V2_MAX_CTX
            })
        };
        let mut d_scores = if attn_v2.is_some() {
            alloc(1, "prefill d_scores (v2 stub)")?
        } else {
            alloc(m * n_head * ctx_max, "prefill d_scores")?
        };
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
            self.matmul_m(s, &d_qact, &self.layers[li].q, &d_act_scale, m, &mut d_q)?;
            self.matmul_m(s, &d_qact, &self.layers[li].k, &d_act_scale, m, &mut d_k)?;
            self.matmul_m(s, &d_qact, &self.layers[li].v, &d_act_scale, m, &mut d_v)?;
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
            // v3/v2 attention when their bounds cover this launch (both
            // bit-identical to the rev-1 kernel per (row, head) — to_bits
            // gated).
            if let Some(f_v3) = attn_v3.as_ref() {
                Self::bl_attn_v3(
                    s,
                    f_v3,
                    &d_q,
                    &self.kv_k[li],
                    &self.kv_v[li],
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
            } else if let Some(f_v2) = attn_v2.as_ref() {
                Self::bl_attn_v2(
                    s,
                    f_v2,
                    &d_q,
                    &self.kv_k[li],
                    &self.kv_v[li],
                    &mut d_attn,
                    n_head,
                    n_head_kv,
                    head_dim,
                    self.attn_scale,
                    causal_offset,
                    m,
                )?;
            } else {
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
            }
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
            self.matmul_m(s, &d_qact, &self.layers[li].o, &d_act_scale, m, &mut d_proj)?;
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
            self.matmul_m(
                s,
                &d_qact,
                &self.layers[li].gate,
                &d_act_scale,
                m,
                &mut d_gate,
            )?;
            self.matmul_m(s, &d_qact, &self.layers[li].up, &d_act_scale, m, &mut d_up)?;
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
            self.matmul_m(
                s,
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

    /// Debug/test access: drop every build-time-resolved IMMA tile function
    /// so `matmul_m` falls back to dp4a — the bit-identity gate prefills the
    /// same prompt with and without the dispatch on ONE model (no env games,
    /// no second 2.5 GB build).
    #[doc(hidden)]
    pub fn debug_disable_imma(&mut self) {
        self.imma_funcs.clear();
    }

    /// Prefill GEMM dispatch (ADR 0026 Track P): route to the IMMA
    /// tensor-core path when `m` clears the tuned crossover and this linear
    /// carries a shadow + a build-time-resolved tile function; else the dp4a
    /// `bl_matmul`, byte-for-byte unchanged. The two paths are BIT-IDENTICAL
    /// (gated: `imma_matches_dp4a_tiled_scaled_bit_exact`), so this branch is
    /// purely a performance decision — no numerics seam. Called ONLY from the
    /// eager prefill; the graph-captured batch/tree paths keep calling
    /// `bl_matmul` directly (a capture-time branch would bake one kernel and
    /// silently change replay semantics).
    fn matmul_m(
        &self,
        s: &Arc<CudaStream>,
        qact: &CudaSlice<i8>,
        lin: &ResidentLinear,
        scale: &CudaSlice<f32>,
        m: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        if m >= self.imma_min_m
            && !lin.tq1
            && let Some(shadow) = lin.imma.as_ref()
        {
            // floor(log2(m)), clamped to the resolved bucket range.
            let bucket = (usize::BITS - 1 - m.leading_zeros()).clamp(5, 11);
            if let Some((tile, func)) = self.imma_funcs.get(&(lin.n, lin.k, bucket)) {
                return backend::launch_imma_tile_on(
                    s,
                    func,
                    *tile,
                    qact,
                    &shadow.device,
                    scale,
                    &lin.scales,
                    out,
                    m as i32,
                    lin.n as i32,
                    lin.k as i32,
                    shadow.num_ktiles,
                );
            }
        }
        Self::bl_matmul(
            s,
            &self.f_tiled_scaled,
            &self.f_tq1_tiled_scaled,
            qact,
            lin,
            scale,
            m,
            out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bl_matmul(
        s: &Arc<CudaStream>,
        f_tiled_scaled: &CudaFunction,
        f_tq1_tiled_scaled: &CudaFunction,
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
            // A2: pick the kernel matching the row packing.
            let mut l = s.launch_builder(if lin.tq1 {
                f_tq1_tiled_scaled
            } else {
                f_tiled_scaled
            });
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

    /// Launch the v3 Q-blocked prefill attention: grid
    /// `(n_head, ceil(m/ATTN_V3_BQ))`, `ATTN_V3_THREADS` threads. Takes the
    /// full `[m, n_head, ctx_max]` scores scratch (unlike v2). Caller
    /// guarantees `head_dim <= ATTN_V2_HDMAX` and that `f` matches the KV
    /// dtype.
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_v3(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
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
            grid_dim: (
                n_head as u32,
                (m as u32).div_ceil(consts::ATTN_V3_BQ as u32),
                1,
            ),
            block_dim: (consts::ATTN_V3_THREADS, 1, 1),
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
        // SAFETY: `gqa_attention_batch_v3_{f32,h}(q, k, v, out, scores,
        // ctx_max, n_head, n_head_kv, head_dim, scale, causal_offset, m)` —
        // KV arena bytes reinterpret as the dtype the function was built for
        // (paired in build_decode_model), scores is the caller's full
        // [m, n_head, ctx_max] scratch, and the head_dim cap keeps every
        // shared index in range.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill attn v3", &e))?;
        }
        Ok(())
    }

    /// Launch the v2 prefill attention: grid `(n_head, m)`, one
    /// `ATTN_V2_THREADS`-thread block per (row, head). Caller guarantees
    /// `head_dim <= ATTN_V2_HDMAX` and `causal_offset + m <= ATTN_V2_MAX_CTX`
    /// (the kernel's static shared sizing) and that `f` matches the KV dtype.
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_v2(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        causal_offset: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, nhkv_i, hd_i, co_i, m_i) = (
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            causal_offset as i32,
            m as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, m as u32, 1),
            block_dim: (consts::ATTN_V2_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&co_i)
            .arg(&m_i);
        // SAFETY: `gqa_attention_batch_v2_{f32,h}(q, k, v, out, n_head,
        // n_head_kv, head_dim, scale, causal_offset, m)` — the KV arena bytes
        // reinterpret as the dtype the resolved function was compiled for
        // (build_decode_model pairs f_attn_batch_v2 with kv_dtype), and the
        // caller-enforced ATTN_V2 bounds keep every shared index in range.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch prefill attn v2", &e))?;
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
            // NULL bitmap = dense-identical (the sparse kernel's contract).
            bm: l.bitmap.as_ref().map_or(0, |b| dptr(b, s)),
            wpr: l.k.div_ceil(256).div_ceil(32) as i32,
            tq1: l.tq1,
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
            if lin.tq1 {
                // TQ1-native twin: dense 9-arg signature (no bitmap).
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
                    self.raw().tq1_tiled_scaled,
                    grid,
                    (WARPS_PER_BLOCK * 32, 1, 1),
                    smem,
                    cs,
                    &mut params,
                )?;
            } else {
                let mut params = [
                    pp(&d_qact),
                    pp(&lin.w),
                    pp(&lin.sc),
                    pp(&d_act_scale),
                    pp(&lin.bm),
                    pp(&d_out),
                    pp(&m_i),
                    pp(&n_i),
                    pp(&k_i),
                    pp(&rb_i),
                    pp(&lin.wpr),
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
            if lin.tq1 {
                self.raw().tq1_tiled_scaled_residual
            } else {
                self.raw().tiled_scaled_residual
            },
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

mod framework_external;
use framework_external::{CurrentContextRestore, ExternalCudaKernels};
pub use framework_external::{
    ExternalLinearBackward, ExternalLinearForward, ExternalLinearGeometry, ExternalLinearPack,
    ExternalLinearScalar,
};
mod graph_raw;
pub use graph_raw::BatchKv;
use graph_raw::*;
mod batch;
mod tree;

#[cfg(test)]
mod tests;
