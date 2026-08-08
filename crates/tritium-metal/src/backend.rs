//! The Apple Metal ternary mpGEMM backend. TQ2_0 weights stay PACKED on device and
//! are decoded in-kernel (`mpgemm_tq2_0` — ~2.06 bit/trit, device-memory parity with
//! the cuda/rocm backends); TQ1_0 weights are host-unpacked and widened to one `i32`
//! per trit (`mpgemm`). Both live in a shared-storage `MTLBuffer` (unified memory on
//! Apple Silicon); the MSL kernel runs and the result is read back.
//!
//! Compiled only on macOS with `--features metal` (see the crate docs).

use core::any::Any;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

/// Threadgroup width for the 2-D dispatch (must match the threadgroup size the
/// kernel is launched with; the MSL kernel itself is agnostic to the value).
const TG_SIZE: u64 = 64;

/// `[M, N, K]` dims + the 2-D dispatch x-extent, passed to the kernel by value as
/// a small constant buffer (`set_bytes` at buffer index 4). `repr(C)` so the
/// field order/offsets match the MSL `struct Dims` exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    /// `threadgroups_x * TG_SIZE`: the kernel flattens `(gid.x, gid.y)` to a
    /// linear output index as `gid.y * lane_stride + gid.x`, so M*N can exceed any
    /// single grid dimension's ceiling.
    lane_stride: u32,
    /// Packed bytes per weight row — read only by the `mpgemm_tq2_0` (packed-decode)
    /// kernel to stride into the on-device TQ2_0 bytes. 0 for the widened `mpgemm`
    /// (i32) kernel, which ignores it.
    row_bytes: u32,
}

// Pin the size + field offsets so a field reorder (which preserves size) cannot
// silently land `n` where the kernel reads `k`. MSL packs this as 5×u32 = 20
// bytes with natural alignment, matching `repr(C)`.
const _: () = assert!(core::mem::size_of::<Dims>() == 20);
const _: () = assert!(core::mem::offset_of!(Dims, m) == 0);
const _: () = assert!(core::mem::offset_of!(Dims, n) == 4);
const _: () = assert!(core::mem::offset_of!(Dims, k) == 8);
const _: () = assert!(core::mem::offset_of!(Dims, lane_stride) == 12);
const _: () = assert!(core::mem::offset_of!(Dims, row_bytes) == 16);

/// Scalar params for the v3 prefill attention (`attention.metal`), passed by
/// value via `set_bytes` at buffer index 5. `repr(C)` so the field
/// order/offsets match the MSL `struct AttnV3Params` exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttnV3Params {
    ctx_max: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    scale: f32,
    causal_offset: u32,
    m: u32,
}

// Pin size + offsets against the MSL struct (7×4-byte fields, natural
// alignment) — the Dims discipline above, so a reorder cannot land `m` where
// the kernel reads `ctx_max`.
const _: () = assert!(core::mem::size_of::<AttnV3Params>() == 28);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, ctx_max) == 0);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, n_head) == 4);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, n_head_kv) == 8);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, head_dim) == 12);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, scale) == 16);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, causal_offset) == 20);
const _: () = assert!(core::mem::offset_of!(AttnV3Params, m) == 24);

/// Send+Sync wrapper around the non-`Send` metal-rs handles.
///
/// metal-rs types are `foreign-types` pointer wrappers and are neither `Send` nor
/// `Sync`, but [`TernaryBackend`] requires both. Apple documents `MTLDevice`,
/// `MTLCommandQueue`, and `MTLComputePipelineState` as safe for concurrent use
/// across threads, so sharing these handles between threads is sound.
struct Handles {
    device: Device,
    queue: CommandQueue,
    /// `mpgemm` — widened-i32 path (TQ1_0).
    pipeline: ComputePipelineState,
    /// `mpgemm_tq2_0` — packed in-kernel-decode path (TQ2_0).
    pipeline_tq2: ComputePipelineState,
    /// `gqa_attention_batch_v3_f32` — v3 Q-blocked prefill attention
    /// (attention.metal, compiled as its OWN library with fast math disabled
    /// so the pinned f32 orders stay IEEE round-to-nearest; mpgemm keeps its
    /// validated default-options compile untouched).
    pipeline_attn_v3: ComputePipelineState,
}

// SAFETY: the wrapped handles are `MTLDevice` / `MTLCommandQueue` /
// `MTLComputePipelineState`, all documented by Apple as thread-safe for
// concurrent use. The backend only ever *reads* through them (creating command
// buffers, encoders, and buffers — each itself confined to the calling `mpgemm`
// invocation), never mutating shared state, so moving/sharing `Handles` across
// threads cannot introduce a data race.
#[allow(unsafe_code)]
unsafe impl Send for Handles {}
// SAFETY: see the `Send` impl above — the same Apple thread-safety guarantee
// covers shared (`&`) concurrent access.
#[allow(unsafe_code)]
unsafe impl Sync for Handles {}

/// How the ternary weights are laid out in the device `MTLBuffer`.
#[derive(Clone, Copy, Debug)]
enum WeightRepr {
    /// TQ2_0 packed bytes, decoded in-kernel by `mpgemm_tq2_0`. `row_bytes` is the
    /// packed stride per weight row. Device memory == packed size (~2.06 bit/trit),
    /// matching the cuda/rocm backends so large models fit unified memory.
    PackedTq2_0 { row_bytes: usize },
    /// One `i32` per trit (host-unpacked + widened), consumed by the `mpgemm`
    /// kernel. Used for TQ1_0 — 32 bit/trit on device, so for small weights only.
    WidenedI32,
}

/// Device buffer: ternary weights resident in a shared-storage `MTLBuffer` (unified
/// memory on Apple Silicon — the CPU fill and the GPU read share physical pages,
/// with no discrete host→device copy), plus the `[N, K]` dims, the original packed
/// byte count, and how the bytes are laid out ([`WeightRepr`]).
pub struct MetalBuffer {
    weights: WeightBuf,
    repr: WeightRepr,
    n: usize,
    k: usize,
    bytes: usize, // original packed byte count, for len_bytes()
}

/// Send+Sync wrapper around the weights `MTLBuffer`.
struct WeightBuf(Buffer);

// SAFETY: `MTLBuffer` is documented as safe to access from multiple threads; the
// buffer is filled once at upload time (before any sharing) and thereafter only
// read by the kernel. See the `Handles` SAFETY note.
#[allow(unsafe_code)]
unsafe impl Send for WeightBuf {}
// SAFETY: see the `Send` impl above.
#[allow(unsafe_code)]
unsafe impl Sync for WeightBuf {}

impl core::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("repr", &self.repr)
            .field("n", &self.n)
            .field("k", &self.k)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl DeviceBuffer for MetalBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Apple-Silicon GPU backend: MSL ternary mpGEMM over Metal.
///
/// The device, queue, and compiled pipeline are built once in
/// [`MetalBackend::new`] and shared across calls. `upload_weights`/`mpgemm` are
/// synchronous (the trait is sync); each `mpgemm` commits one command buffer and
/// blocks on `wait_until_completed`.
pub struct MetalBackend {
    handles: Handles,
    device_name: String,
}

impl core::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_name", &self.device_name)
            .finish()
    }
}

/// Packed bytes per block for a format this backend supports.
fn block_bytes(format: TernaryFormat) -> Result<usize, BackendError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(BackendError::UnsupportedFormat(other)),
    }
}

impl MetalBackend {
    /// Acquire the system-default Metal device, build a command queue, and
    /// compile the mpGEMM pipeline from the embedded MSL source.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if no Metal device is present
    /// (`MTLCreateSystemDefaultDevice()` is nil), the MSL fails to compile, the
    /// `mpgemm` function is missing, or the pipeline state cannot be created — the
    /// registry logs and skips it, and the conformance test self-skips.
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or_else(|| {
            BackendError::Backend("no Metal device (system default is nil)".into())
        })?;
        let device_name = device.name().to_owned();

        let queue = device.new_command_queue();

        // Compile the MSL kernel at runtime. `newLibraryWithSource:` returns the
        // compiler diagnostics as the `Err` string on failure.
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(include_str!("mpgemm.metal"), &options)
            .map_err(|e| BackendError::Backend(format!("MSL compile: {e}")))?;
        // Both kernels compile from the one embedded MSL source: `mpgemm` (widened
        // i32, for TQ1_0) and `mpgemm_tq2_0` (packed in-kernel decode, for TQ2_0).
        let f_i32 = library
            .get_function("mpgemm", None)
            .map_err(|e| BackendError::Backend(format!("missing `mpgemm` function: {e}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&f_i32)
            .map_err(|e| BackendError::Backend(format!("pipeline state for `mpgemm`: {e}")))?;
        let f_tq2 = library
            .get_function("mpgemm_tq2_0", None)
            .map_err(|e| BackendError::Backend(format!("missing `mpgemm_tq2_0` function: {e}")))?;
        let pipeline_tq2 = device
            .new_compute_pipeline_state_with_function(&f_tq2)
            .map_err(|e| {
                BackendError::Backend(format!("pipeline state for `mpgemm_tq2_0`: {e}"))
            })?;

        // v3 prefill attention: a SEPARATE library so it can compile with fast
        // math disabled (IEEE round-to-nearest +,*,/ — the MSL meaning of
        // CUDA's __fadd_rn/__fmul_rn discipline) without perturbing the
        // validated mpgemm codegen above.
        let attn_options = CompileOptions::new();
        attn_options.set_fast_math_enabled(false);
        let attn_library = device
            .new_library_with_source(include_str!("attention.metal"), &attn_options)
            .map_err(|e| BackendError::Backend(format!("attention MSL compile: {e}")))?;
        let f_attn_v3 = attn_library
            .get_function("gqa_attention_batch_v3_f32", None)
            .map_err(|e| {
                BackendError::Backend(format!("missing `gqa_attention_batch_v3_f32` function: {e}"))
            })?;
        let pipeline_attn_v3 = device
            .new_compute_pipeline_state_with_function(&f_attn_v3)
            .map_err(|e| {
                BackendError::Backend(format!("pipeline state for `gqa_attention_batch_v3_f32`: {e}"))
            })?;

        Ok(MetalBackend {
            handles: Handles {
                device,
                queue,
                pipeline,
                pipeline_tq2,
                pipeline_attn_v3,
            },
            device_name,
        })
    }
}

/// Create a shared-storage (unified-memory) `MTLBuffer` initialised from `data`.
///
/// `StorageModeShared` keeps the buffer in memory both the CPU and the GPU
/// address directly — the Apple-Silicon path — so there is no separate upload
/// copy. A zero-length buffer would be rejected by Metal, so callers must avoid
/// empty allocations (the kernel is never dispatched for zero-size shapes).
fn shared_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let len_bytes = core::mem::size_of_val(data) as u64;
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        len_bytes,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Read `count` `f32`s out of a shared-storage `MTLBuffer` into `out`.
///
/// On `StorageModeShared` the GPU writes are visible to the CPU once the command
/// buffer has completed (the caller blocks on `wait_until_completed` first), so
/// this is a plain memcpy out of the unified-memory contents pointer.
#[allow(unsafe_code)]
fn read_f32(buf: &Buffer, out: &mut [f32]) {
    // SAFETY: `buf` was allocated with `out.len() * 4` bytes (the caller sizes the
    // output buffer to exactly `m * n` f32s), is shared-storage so its `contents`
    // pointer is valid on the CPU, and the GPU work that wrote it has completed
    // (the caller blocked on `wait_until_completed`). The source is a raw byte
    // region holding `out.len()` little-endian f32s; we copy them out without
    // creating an aliasing &[f32] over Metal-owned memory.
    unsafe {
        let src = buf.contents().cast::<f32>();
        core::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
    }
}

impl TernaryBackend for MetalBackend {
    fn device_id(&self) -> &str {
        "metal"
    }

    fn capabilities(&self) -> DeviceCaps {
        // No fp8/IMMA; the fused W1.58A8 path degrades via the trait default.
        DeviceCaps::new("metal", self.device_name.clone())
            .with_features(vec!["metal".to_owned()])
            .with_memory(self.handles.device.recommended_max_working_set_size())
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let row_bytes = nb * block_bytes(format)?;
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} format {format:?}",
                packed.len()
            )));
        }

        // TQ2_0 keeps the PACKED bytes on device (decoded in-kernel by
        // `mpgemm_tq2_0`) so device memory stays at the packed ~2.06 bit/trit —
        // 16× less than widening to i32 — matching the cuda/rocm backends so large
        // models fit unified memory. TQ1_0 (the small/rare format) is host-unpacked
        // and widened to one i32 per trit for the `mpgemm` kernel. A zero-element
        // buffer would be an invalid Metal allocation, so degenerate shapes get a
        // 1-byte placeholder (mpgemm short-circuits zero-dim cases before touching
        // the buffer).
        let dev = &self.handles.device;
        let (weights, repr) = match format {
            TernaryFormat::Tq2_0 => {
                let buf = if packed.is_empty() {
                    dev.new_buffer(1, MTLResourceOptions::StorageModeShared)
                } else {
                    shared_buffer(dev, packed)
                };
                (buf, WeightRepr::PackedTq2_0 { row_bytes })
            }
            TernaryFormat::Tq1_0 => {
                let mut trits = vec![Trit::ZERO; n * k];
                let mut scratch = vec![half::f16::ONE; nb];
                for ni in 0..n {
                    let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
                    let trits_row = &mut trits[ni * k..ni * k + k];
                    unpack_tq1_0_row(row, trits_row, &mut scratch)
                        .map_err(|e| BackendError::Backend(format!("unpack row {ni}: {e}")))?;
                }
                let widened: Vec<i32> = trits.iter().map(|t| i32::from(t.get())).collect();
                let buf = if widened.is_empty() {
                    dev.new_buffer(1, MTLResourceOptions::StorageModeShared)
                } else {
                    shared_buffer(dev, &widened)
                };
                (buf, WeightRepr::WidenedI32)
            }
            other => return Err(BackendError::UnsupportedFormat(other)),
        };

        Ok(Box::new(MetalBuffer {
            weights: WeightBuf(weights),
            repr,
            n,
            k,
            bytes: packed.len(),
        }))
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
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a MetalBuffer".to_owned()))?;
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

        // Zero-dim shapes: the reference returns all-zeros (empty output, or an
        // empty K-contraction → every out is scale·0 = 0). Mirror it without
        // touching the GPU — a zero-size buffer/dispatch is a Metal error.
        if m * n == 0 || k == 0 {
            out.fill(0.0);
            return Ok(());
        }

        // The kernel indexes in u32 (M*N flattened, K loop). Reject shapes that
        // would silently wrap the cast rather than dispatching a truncated grid.
        if m > u32::MAX as usize
            || n > u32::MAX as usize
            || k > u32::MAX as usize
            || m * n > u32::MAX as usize
        {
            return Err(BackendError::InvalidInput(format!(
                "shape {shape:?} exceeds the metal kernel's u32 index range"
            )));
        }

        // 2-D dispatch grid: split the ceil(M*N / TG_SIZE) threadgroups across x
        // and y so a single dimension never exceeds Metal's per-dimension
        // threadgroup ceiling. The kernel reconstructs the linear index via
        // `gid.y * lane_stride + gid.x`. We dispatch full threadgroups (the kernel
        // guards the tail by `idx >= total`), so this never under-covers M*N.
        let total = (m * n) as u64;
        let total_groups = total.div_ceil(TG_SIZE).max(1);
        // Metal guarantees at least 65535 threadgroups per grid dimension; use the
        // same conservative split the wgpu backend uses so very large M*N still
        // dispatch within a single dimension's limit.
        const MAX_DIM: u64 = 65535;
        let (gx, gy) = if total_groups <= MAX_DIM {
            (total_groups, 1)
        } else {
            (MAX_DIM, total_groups.div_ceil(MAX_DIM))
        };
        let lane_stride = gx * TG_SIZE;
        // Select the kernel + per-row stride from how the weights are laid out:
        // packed TQ2_0 (decoded in-kernel) vs widened-i32 (TQ1_0).
        let (pipeline, row_bytes) = match buf.repr {
            WeightRepr::PackedTq2_0 { row_bytes } => (&self.handles.pipeline_tq2, row_bytes as u32),
            WeightRepr::WidenedI32 => (&self.handles.pipeline, 0u32),
        };
        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lane_stride: lane_stride as u32,
            row_bytes,
        };

        // Shared-storage input buffers (unified memory — filled by the CPU, read
        // by the kernel with no discrete copy).
        let act_buf = shared_buffer(&self.handles.device, act);
        let scales_buf = shared_buffer(&self.handles.device, scales);
        let out_bytes = (m * n * core::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .handles
            .device
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.handles.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&act_buf), 0);
        encoder.set_buffer(1, Some(&buf.weights.0), 0);
        encoder.set_buffer(2, Some(&scales_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);
        // Dims is a small by-value struct passed via `set_bytes` (no buffer alloc).
        encoder.set_bytes(
            4,
            core::mem::size_of::<Dims>() as u64,
            core::ptr::from_ref(&dims).cast(),
        );
        // Dispatch full threadgroups (gx·gy of them), each TG_SIZE×1×1 threads;
        // the kernel's `idx >= total` guard discards the padding threads.
        let threadgroups = MTLSize::new(gx, gy, 1);
        let threads_per_group = MTLSize::new(TG_SIZE, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Shared storage → the GPU's writes are visible to the CPU now that the
        // command buffer has completed. Copy them into `out`.
        read_f32(&out_buf, out);
        Ok(())
    }
}

impl MetalBackend {
    /// v3 Q-blocked prefill GQA attention (Track E1: the MSL port of
    /// tritium-cuda's `gqa_attention_batch_v3_f32`) — `M >= 1` query rows
    /// against an f32 KV arena, causal, GQA.
    ///
    /// STATUS: the device kernel is compile-verified only until the
    /// self-hosted Apple-Silicon lane runs `backend::tests::
    /// attn_v3_matches_pinned_host_reference_or_skip`.
    ///
    /// Layouts (row-major): `q`/`out` `[m, n_head, head_dim]`; `k`/`v` KV
    /// arenas `[>= causal_offset + m, n_head_kv, head_dim]`. `ctx_max` is the
    /// scores-scratch stride (`>= causal_offset + m`; the arena capacity, in
    /// the CUDA runner's terms).
    ///
    /// Dispatch priority — CUDA runs v3 → v2 → rev-1; this backend has no v2
    /// or rev-1 device kernel, so the ladder is v3 (device) → the pinned-order
    /// HOST reference [`crate::attn::gqa_attention_prefill_ref`] (same
    /// summation orders, so the two rungs agree to within the documented
    /// `precise::exp` deviation). The host rung serves when:
    /// * `TRITIUM_ATTN_V3=0` (kill switch; the tritium-cuda env contract —
    ///   any value other than `0`/`1` is a loud reject);
    /// * `head_dim > ATTN_V3_HDMAX` (the kernel's threadgroup staging bound);
    /// * the compiled pipeline's `thread_execution_width != 32` (the kernel's
    ///   warp→simdgroup mapping assumes 32-lane simdgroups);
    /// * the row-block count exceeds Metal's guaranteed 65535 threadgroups
    ///   per grid dimension.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on shape/length violations or a
    /// malformed `TRITIUM_ATTN_V3`; [`BackendError::Backend`] on Metal
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
        let (tg_x, tg_y) = crate::attn::v3_threadgroups(m, n_head);
        // Metal guarantees at least 65535 threadgroups per grid dimension
        // (same conservative ceiling the mpgemm dispatch uses).
        let use_device = v3_enabled
            && head_dim <= crate::attn::ATTN_V3_HDMAX
            && self.handles.pipeline_attn_v3.thread_execution_width()
                == crate::attn::ATTN_V3_SIMD_WIDTH as u64
            && tg_x <= 65535
            && tg_y <= 65535;
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
            (tg_x, tg_y),
        )
    }

    /// Launch the v3 kernel: grid `(n_head, ceil(m/ATTN_V3_BQ))` threadgroups
    /// of `ATTN_V3_THREADS`, plus the global `[m, n_head, ctx_max]` scores
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
        (tg_x, tg_y): (u64, u64),
    ) -> Result<(), BackendError> {
        let scores_len = crate::attn::v3_scores_len(m, n_head, ctx_max)
            .ok_or_else(|| BackendError::InvalidInput("scores scratch overflows".to_owned()))?;

        let dev = &self.handles.device;
        let q_buf = shared_buffer(dev, q);
        let k_buf = shared_buffer(dev, k);
        let v_buf = shared_buffer(dev, v);
        let out_bytes = core::mem::size_of_val(out) as u64;
        let out_buf = dev.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);
        let scores_bytes = (scores_len * core::mem::size_of::<f32>()) as u64;
        let scores_buf = dev.new_buffer(scores_bytes, MTLResourceOptions::StorageModeShared);

        let params = AttnV3Params {
            ctx_max: ctx_max as u32,
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            scale,
            causal_offset: causal_offset as u32,
            m: m as u32,
        };

        let cmd = self.handles.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.handles.pipeline_attn_v3);
        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&k_buf), 0);
        encoder.set_buffer(2, Some(&v_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_buffer(4, Some(&scores_buf), 0);
        encoder.set_bytes(
            5,
            core::mem::size_of::<AttnV3Params>() as u64,
            core::ptr::from_ref(&params).cast(),
        );
        let threadgroups = MTLSize::new(tg_x, tg_y, 1);
        let threads_per_group = MTLSize::new(u64::from(crate::attn::ATTN_V3_THREADS), 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        read_f32(&out_buf, out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MetalBackend;
    use tritium_testkit::{
        Tolerance, frozen_vectors, run_conformance, run_fused_fallback_contract,
    };

    /// Frozen-set conformance + fused-fallback on the Metal device, or a clean
    /// self-skip when no Metal device is present (mirrors the CUDA/wgpu tests:
    /// `MTLCreateSystemDefaultDevice()` is nil → `MetalBackend::new` errors → skip).
    #[test]
    fn conformance_and_fused_fallback_or_skip() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping metal conformance: no Metal device ({e})");
                return;
            }
        };
        let vectors = frozen_vectors();

        let report = run_conformance(&backend, &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} metal conformance failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all vectors must pass");

        let fused = run_fused_fallback_contract(&backend, &vectors, Tolerance::default());
        assert!(
            fused.is_ok(),
            "{} metal fused-fallback failures: {:?}",
            fused.failed.len(),
            fused.failed
        );
        assert_eq!(
            fused.passed,
            vectors.len(),
            "fused path must degrade cleanly"
        );
    }

    // ---- coverage beyond the frozen set (large shapes, zero dims) -------------

    use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};
    use tritium_spec::{MpGemm, TernaryBackend};

    /// Build a deterministic random `[M,K]` activation + packed `[N,K]` tq2_0
    /// weights + `[N]` scales, plus the `reference_mpgemm` oracle output.
    fn random_case(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>) {
        // xorshift64 — no external rng, deterministic across runs/platforms.
        let mut s: u64 =
            0x9E37_79B9_7F4A_7C15 ^ ((m as u64) << 1) ^ ((n as u64) << 17) ^ (k as u64);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let act: Vec<f32> = (0..m * k)
            .map(|_| (next() as f32 / u64::MAX as f32) * 2.0 - 1.0)
            .collect();
        let trits: Vec<Trit> = (0..n * k)
            .map(|_| {
                let v = match next() % 3 {
                    0 => 0i8,
                    1 => 1,
                    _ => -1,
                };
                Trit::from_i8(v).expect("valid trit")
            })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|_| (next() as f32 / u64::MAX as f32) + 0.5)
            .collect();

        let nb = num_blocks(k);
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let unit = vec![half::f16::ONE; nb];
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
        }

        let shape = GemmShape::new(m, n, k);
        let mut expected = vec![0.0f32; m * n];
        reference_mpgemm(&act, &trits, &scales, shape, &mut expected).expect("reference");
        (act, packed, scales, expected)
    }

    fn run_shape(backend: &MetalBackend, m: usize, n: usize, k: usize) {
        let (act, packed, scales, expected) = random_case(m, n, k);
        let shape = GemmShape::new(m, n, k);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .expect("upload");
        let mut out = vec![0.0f32; m * n];
        backend
            .mpgemm(MpGemm {
                act: &act,
                weights: buf.as_ref(),
                scales: &scales,
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            })
            .expect("mpgemm");
        let tol = Tolerance::default();
        for (i, (&g, &w)) in out.iter().zip(&expected).enumerate() {
            assert!(
                tol.accepts(g, w),
                "[{i}] got {g} want {w} (shape {m}x{n}x{k})"
            );
        }
    }

    /// A GEMM whose output count (M*N) exceeds the single-dimension 65535-
    /// threadgroup ceiling (65535*64 = 4_194_240), exercising the 2-D grid split.
    #[test]
    fn large_shape_exceeding_single_dim_dispatch() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no Metal device ({e})");
                return;
            }
        };
        run_shape(&backend, 1024, 4096, 256); // M*N = 4_194_304 > 4_194_240
    }

    /// Zero-dimension shapes match the reference (empty output when M=0, or K=0 →
    /// all-zeros) without a Metal zero-size-buffer/dispatch error.
    #[test]
    fn zero_dims_match_reference() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no Metal device ({e})");
                return;
            }
        };
        run_shape(&backend, 0, 4, 256); // M=0 → empty output
        run_shape(&backend, 2, 3, 0); // K=0 → all-zeros (each out = scale·empty-sum)
    }

    // ---- v3 prefill attention (Track E1) — Mac-lane conformance gate --------

    use crate::attn;

    /// Deterministic q/k/v for an attention shape (xorshift64, like
    /// `random_case` above — no external rng).
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
        let k: Vec<f32> = (0..ctx_top * n_head_kv * head_dim).map(|_| next()).collect();
        let v: Vec<f32> = (0..ctx_top * n_head_kv * head_dim).map(|_| next()).collect();
        (q, k, v)
    }

    /// The v3 device kernel vs the pinned-order host reference, across the
    /// CUDA gate's regimes: staircase (m spanning multiple BQ row-blocks with
    /// a tail), pure tail (m < BQ), deep ctx (causal_offset > 0), GQA + MHA
    /// head groupings, head_dim at/below the HDMAX cap and not a multiple of
    /// 32, and a ctx_max wider than ctx_top (scores-stride vs arena split).
    ///
    /// TOLERANCE, not to_bits: every summation ORDER is pinned identically on
    /// both sides, but the kernel's exponential is `precise::exp` (Metal has
    /// no f64 to reproduce CUDA's exp_f32) vs the host's glibc-correctly-
    /// rounded `f32::exp` — a few ULP per weight, so per-element agreement is
    /// asserted at rel 1e-5 / abs 1e-6. A drift beyond that means a REAL
    /// order/mapping bug, not exp noise. Calls the device path directly (not
    /// the dispatch wrapper), so an exec-width fallback cannot turn this gate
    /// into a vacuous ref-vs-ref pass; skips (loudly) off-32-lane hardware
    /// and when no Metal device exists.
    #[test]
    fn attn_v3_matches_pinned_host_reference_or_skip() {
        let backend = match MetalBackend::new() {
            Ok(b) => b,
            // Skip ONLY the genuinely-no-device case; an MSL compile error in
            // attention.metal must FAIL here, not masquerade as a skip (this
            // is the first place the kernel source ever meets a Metal
            // compiler — it is authored on Linux).
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("no Metal device"),
                    "metal attn v3 gate: backend init failed (not a missing device): {msg}"
                );
                eprintln!("skipping metal attn v3 gate: no Metal device ({msg})");
                return;
            }
        };
        let width = backend.handles.pipeline_attn_v3.thread_execution_width();
        if width != attn::ATTN_V3_SIMD_WIDTH as u64 {
            eprintln!("skipping metal attn v3 gate: thread_execution_width {width} != 32");
            return;
        }

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
                    attn::v3_threadgroups(m, n_head),
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

            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let diff = (g - w).abs();
                let ok = diff <= 1e-6 || diff <= 1e-5 * w.abs();
                assert!(
                    ok,
                    "[{i}] device {g} vs host {w} (m={m} n_head={n_head} kv={n_head_kv} \
                     hd={head_dim} co={causal_offset} ctx_max={ctx_max}) — beyond the \
                     exp-only tolerance: a pinned-order or thread-mapping bug"
                );
            }
        }

        // Smoke the public dispatch wrapper once (env parsing + priority +
        // validation glue; the loop above pinned the device kernel itself).
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
                "wrapper [{i}] {g} vs {w}"
            );
        }
    }
}
