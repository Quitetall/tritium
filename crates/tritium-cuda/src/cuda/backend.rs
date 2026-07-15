//! The `CudaBackend`: mpgemm dispatch (tiled/IMMA/sparse/SALT), weight upload,
//! autotune glue, device queries and backend registration
//! (P2a split: move-only from `cuda/mod.rs`).

use super::*;
use cudarc::driver::{HostSlice, PinnedHostSlice, SyncOnDrop};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2IndexedRuntimeLedger, SaltV2Tensor, SaltV2Transform, pack_salt_v2_plane,
};

/// A logical prefix of cudarc page-locked memory.
///
/// cudarc 0.19 does not expose pinned host views. Delegating the synchronization
/// guard to the owner keeps the page-locked allocation's event semantics while
/// allowing bounded copies for unequal parameter leaves.
struct PinnedPrefix<'a, T> {
    owner: &'a mut PinnedHostSlice<T>,
    len: usize,
}

#[allow(unsafe_code)]
// SAFETY: both methods delegate to the owning `PinnedHostSlice`, retain its
// `SyncOnDrop`, and only narrow the returned slice to a validated prefix.
impl<T> HostSlice<T> for PinnedPrefix<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        // SAFETY: the delegated slice is used only with `stream`; `new`
        // validates that `len` is within the owner allocation.
        let (slice, sync) = unsafe { self.owner.stream_synced_slice(stream) };
        (&slice[..self.len], sync)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        // SAFETY: the delegated slice is used only with `stream`; `new`
        // validates that `len` is within the owner allocation.
        let (slice, sync) = unsafe { self.owner.stream_synced_mut_slice(stream) };
        (&mut slice[..self.len], sync)
    }
}

impl<'a, T> PinnedPrefix<'a, T> {
    fn new(owner: &'a mut PinnedHostSlice<T>, len: usize) -> Result<Self, BackendError> {
        if len > owner.len() {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: owner.len(),
            });
        }
        Ok(Self { owner, len })
    }
}

/// Compact Track D rows keep only TQ2's 2-bit `qs` payload; scales are f32 sidecars.
const TRAINING_SALT_QK: usize = tritium_format::QK_K;
const TRAINING_SALT_QS_BYTES: usize = TRAINING_SALT_QK / 4;
const TRAINING_SALT_TILE_X: u32 = 32;
const TRAINING_SALT_TILE_M: u32 = 4;
const TRAINING_SALT_EXACT_TILE_M: u32 = 16;
const KERNEL_NAME_SALT_TRAINING_FORWARD_TILED: &str = "salt_training_forward_tiled";
const KERNEL_NAME_SALT_TRAINING_GRAD_A_TILED: &str = "salt_training_grad_a_tiled";

#[derive(Clone, Copy)]
enum TrainingSaltDispatch {
    Exact,
    Fast,
    // Referenced by non-test dispatch selection even though only tests request
    // it directly; cfg(test) would remove a variant needed by production code.
    #[allow(dead_code)]
    ScalarExact,
    #[cfg(test)]
    ScalarFast,
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
    pub(super) device: Arc<CudaSlice<u8>>,
    /// Output channels (`N`).
    pub(super) n: usize,
    /// Contraction dimension (`K`).
    pub(super) k: usize,
    /// TQ2_0 bytes per plane-row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    pub(super) row_bytes: usize,
    /// Realized plane count `T` (max over rows; ragged rows zero-padded on upload).
    pub(super) t_planes: usize,
}

/// Exact requested `CudaSlice` byte ledger for one resident SALT V2 rank-2 tensor.
///
/// This counts logical buffer lengths requested by this handle, not allocator
/// rounding, retained pool capacity, CUDA context/module memory, or unrelated
/// allocations. `dense_weight_bytes()` is structurally zero because upload
/// retains codec payloads and reconstructs coefficients only inside the
/// contraction kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2ResidentAllocationReceipt {
    codec: SaltV2Codec,
    runtime: SaltV2IndexedRuntimeLedger,
}

impl SaltV2ResidentAllocationReceipt {
    fn new(codec: SaltV2Codec, runtime: SaltV2IndexedRuntimeLedger) -> Self {
        Self { codec, runtime }
    }

    /// Codec used by every resident plane.
    #[must_use]
    pub fn codec(self) -> SaltV2Codec {
        self.codec
    }

    /// Encoded D2/B3/S34 bytes resident on the device.
    #[must_use]
    pub fn payload_bytes(self) -> u64 {
        self.runtime.payload_bytes()
    }

    /// Group128 f16 scale bytes resident on the device.
    #[must_use]
    pub fn scale_bytes(self) -> u64 {
        self.runtime.scale_bytes()
    }

    /// Complete two-bit allocation-map bytes resident on the device.
    #[must_use]
    pub fn map_bytes(self) -> u64 {
        self.runtime.allocation_map_bytes()
    }

    /// Coarse plane-rank prefix bytes resident on the device.
    #[must_use]
    pub fn rank_prefix_bytes(self) -> u64 {
        self.runtime.rank_prefix_bytes()
    }

    /// Logical two-bit allocation-map size, including scalar-carried tail bits.
    #[must_use]
    pub fn allocation_map_bits(self) -> u64 {
        self.runtime.allocation_map_bits()
    }

    /// Allocation-map tail bits carried in the resident handle/kernel scalar.
    #[must_use]
    pub fn allocation_map_embedded_bits(self) -> u64 {
        self.runtime.allocation_map_embedded_bits()
    }

    /// Dense dequantized weight bytes, always zero.
    #[must_use]
    pub fn dense_weight_bytes(self) -> u64 {
        self.runtime.dense_shadow_bytes()
    }

    /// Sum of payload, scales, complete map bytes, and coarse rank prefixes.
    #[must_use]
    pub fn steady_resident_bytes(self) -> u64 {
        self.runtime.steady_resident_bytes()
    }

    /// Shared checked indexed-runtime plan verified against the uploaded buffers.
    #[must_use]
    pub fn runtime_ledger(self) -> SaltV2IndexedRuntimeLedger {
        self.runtime
    }
}

/// Device-resident SALT V2 tensor retaining physical codec bytes, a compact
/// allocation map, and bounded-scan rank prefixes without a dense shadow.
#[derive(Debug)]
pub struct SaltV2ResidentTensor {
    payload: CudaSlice<u8>,
    scales: CudaSlice<u16>,
    index_metadata: Option<CudaSlice<u8>>,
    rows: usize,
    columns: usize,
    tile_count: usize,
    plane_count: usize,
    codec_tag: u32,
    allocation_map_bytes: u32,
    rank_prefix_count: u32,
    terminal_map_value: u32,
    receipt: SaltV2ResidentAllocationReceipt,
}

impl SaltV2ResidentTensor {
    /// Exact persistent device-allocation receipt for this tensor.
    #[must_use]
    pub fn allocation_receipt(&self) -> SaltV2ResidentAllocationReceipt {
        self.receipt
    }

    /// Output rows in the row-major semantic matrix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Contraction columns in the row-major semantic matrix.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }
}

/// Execution label written into every SALT V2 CUDA forward receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2ForwardMode {
    /// Scalar deterministic kernel with CPU-reference reduction order.
    Exact,
    /// Public fast entry point currently dispatching the exact kernel unchanged.
    FastAliasesExact,
}

/// Checked requested-`CudaSlice` ledger for one SALT V2 forward.
///
/// These byte totals cover the handle plus this call's activation/output
/// buffers. They are not a physical CUDA pool/context high-water measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2ForwardReceipt {
    mode: SaltV2ForwardMode,
    resident: SaltV2ResidentAllocationReceipt,
    activation_bytes: u64,
    output_bytes: u64,
    dense_weight_bytes: u64,
    peak_resident_bytes: u64,
}

impl SaltV2ForwardReceipt {
    fn new(
        mode: SaltV2ForwardMode,
        resident: SaltV2ResidentAllocationReceipt,
        activation_elements: usize,
        output_elements: usize,
    ) -> Result<Self, BackendError> {
        let bytes = |elements: usize, field: &str| {
            elements
                .checked_mul(core::mem::size_of::<f32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| {
                    BackendError::InvalidInput(format!("SALT V2 {field} byte count overflows u64"))
                })
        };
        let activation_bytes = bytes(activation_elements, "activation")?;
        let output_bytes = bytes(output_elements, "output")?;
        let peak_resident_bytes = resident
            .steady_resident_bytes()
            .checked_add(activation_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                BackendError::InvalidInput("SALT V2 peak resident byte count overflows u64".into())
            })?;
        Ok(Self {
            mode,
            resident,
            activation_bytes,
            output_bytes,
            dense_weight_bytes: 0,
            peak_resident_bytes,
        })
    }

    /// Exact dispatch label; fast aliasing cannot be mistaken for an optimized kernel.
    #[must_use]
    pub fn mode(self) -> SaltV2ForwardMode {
        self.mode
    }

    /// Persistent encoded-weight and metadata bytes.
    #[must_use]
    pub fn steady_resident_bytes(self) -> u64 {
        self.resident.steady_resident_bytes()
    }

    /// Component-level persistent allocation receipt used by this launch.
    #[must_use]
    pub fn resident_allocation(self) -> SaltV2ResidentAllocationReceipt {
        self.resident
    }

    /// Per-call activation upload bytes.
    #[must_use]
    pub fn activation_bytes(self) -> u64 {
        self.activation_bytes
    }

    /// Per-call device output allocation bytes.
    #[must_use]
    pub fn output_bytes(self) -> u64 {
        self.output_bytes
    }

    /// Dense dequantized weight bytes, always zero.
    #[must_use]
    pub fn dense_weight_bytes(self) -> u64 {
        self.dense_weight_bytes
    }

    /// Persistent bytes plus the activation and output allocations live at launch.
    #[must_use]
    pub fn peak_resident_bytes(self) -> u64 {
        self.peak_resident_bytes
    }
}

/// Host-visible output and checked allocation evidence from a SALT V2 forward.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2Forward {
    /// Row-major `[M, N]` output copied from the device.
    pub output: Vec<f32>,
    /// Dispatch label and device allocation accounting.
    pub receipt: SaltV2ForwardReceipt,
}

fn pack_salt_v2_cuda_plane(
    codec: SaltV2Codec,
    trits: &[tritium_core::Trit],
) -> Result<Vec<u8>, BackendError> {
    pack_salt_v2_plane(codec, trits)
        .map_err(|error| BackendError::InvalidInput(format!("SALT V2 codec: {error}")))
}

/// Training-specific resident SALT planes produced directly from a latent f32
/// master. Codes use TQ2's canonical 2-bit addressing but omit its per-block
/// f16 scale; Track A's exact f32 AbsMean lives in the separate `scales` buffer.
/// This is intentionally distinct from [`SaltResidentLinear`]'s inference and
/// on-disk-compatible 66-byte blocks.
#[derive(Debug)]
pub(crate) struct TrainingSaltLinear {
    codes: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    n: usize,
    k: usize,
    row_bytes: usize,
    planes: usize,
}

#[allow(dead_code)] // accounting is load-bearing in the Track D GPU gate before tape wiring
impl TrainingSaltLinear {
    pub(crate) fn rows(&self) -> usize {
        self.n
    }

    pub(crate) fn cols(&self) -> usize {
        self.k
    }

    pub(crate) fn planes(&self) -> usize {
        self.planes
    }

    /// Compact 2-bit code payload bytes, excluding external scales.
    pub(crate) fn packed_bytes(&self) -> usize {
        self.codes.len()
    }

    /// External Track A f32 scale bytes.
    pub(crate) fn scale_bytes(&self) -> usize {
        self.scales.len() * core::mem::size_of::<f32>()
    }

    /// Total resident bytes owned by this packed handle.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.packed_bytes() + self.scale_bytes()
    }

    fn kernel_dims(&self, m: usize) -> Result<(i32, i32, i32, i32, i32), BackendError> {
        let convert = |name: &str, value: usize| {
            i32::try_from(value)
                .map_err(|_| BackendError::InvalidInput(format!("SALT {name} exceeds i32::MAX")))
        };
        Ok((
            convert("batch", m)?,
            convert("rows", self.n)?,
            convert("cols", self.k)?,
            convert("planes", self.planes)?,
            convert("row bytes", self.row_bytes)?,
        ))
    }
}

/// Device metadata for deterministic segmented embedding gradients. Equal
/// token ids are grouped while `positions` stays ascending inside each group.
#[derive(Debug)]
pub(crate) struct EmbedSegments {
    metadata: CudaSlice<i32>,
    unique_rows: usize,
    seq: usize,
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
    pub(super) stream: Arc<CudaStream>,
    /// Loaded add-only PTX module (kept alive so `func`/`func_tiled` stay valid).
    pub(super) _module: Arc<CudaModule>,
    /// Loaded IMMA PTX module (kept alive so `func_imma`/`func_act_quant` stay
    /// valid). A separate `compute_80` image, distinct from `_module`'s `compute_75`.
    pub(super) _imma_module: Arc<CudaModule>,
    /// The resolved `tq2_0_add_mpgemm` kernel (one thread per output).
    pub(super) func: CudaFunction,
    /// The resolved `tq2_0_add_mpgemm_tiled` kernel (warp per output, shared-mem
    /// staged activations) — the decode path.
    pub(super) func_tiled: CudaFunction,
    /// Fused-scaled variant: folds `act_scale` into the epilogue, eliminating the
    /// separate `scale_mul_f32` launch (v0.6.0 opt #15).
    pub(super) func_tiled_scaled: CudaFunction,
    /// The resolved `salt_mpgemm_tiled_f32` kernel (v0.4.0 SALT multi-plane GEMM),
    /// driven by the resident [`CudaBackend::salt_forward`] path.
    pub(super) func_salt: CudaFunction,
    /// Direct SALT V2 codec PTX kept alive for `func_salt_v2_exact`.
    pub(super) _salt_v2_module: Arc<CudaModule>,
    /// Scalar-correct D2/B3/S34 forward. The first fast API aliases this handle.
    pub(super) func_salt_v2_exact: CudaFunction,
    /// Sparse-aware f32-tiled kernel: bitmap skip for zero blocks (P1 opt).
    #[allow(dead_code)]
    // validated sparse mpgemm kernel; auto-dispatch integration is future (1.x) work
    pub(super) func_tiled_f32_sparse: CudaFunction,
    /// Sparse-aware fused-scaled f32-tiled kernel: bitmap skip + act_scale fold.
    #[allow(dead_code)]
    // validated sparse fused-scaled kernel; auto-dispatch integration is future (1.x) work
    pub(super) func_tiled_i8_scaled_sparse: CudaFunction,
    /// The resolved `tq2_0_imma_mpgemm` kernel (IMMA int8 tensor-core prefill).
    pub(super) func_imma: CudaFunction,
    /// The resolved `act_quant_int8_per_token` kernel (on-device W1.58A8 quant).
    pub(super) func_act_quant: CudaFunction,
    /// Loaded decode PTX module (v0.3.1 device-resident forward), kept alive so its
    /// functions stay valid. Compiled `--fmad=false` to bit-match the host f32 ops.
    pub(super) _decode_module: Arc<CudaModule>,
    /// The resolved `rmsnorm_f32` decode kernel (bit-matches `ops::rmsnorm`).
    // Read only by `rmsnorm` (test-exercised today); `forward_device` wires it into
    // the per-token decode next. W1-in-progress.
    #[allow(dead_code)]
    pub(super) func_rmsnorm: CudaFunction,
    /// The resolved `rmsnorm_quant_f32` decode kernel (fused RMSNorm + int8 act-quant).
    /// Exercised by `rmsnorm_quant_bit_matches_host`; the resident decode path launches
    /// it via its own `f_rmsnorm_quant` handle.
    #[allow(dead_code)]
    pub(super) func_rmsnorm_quant: CudaFunction,
    /// The resolved `rope_apply_f32` decode kernel (bit-matches `ops::rope_apply`).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(super) func_rope: CudaFunction,
    /// The resolved `softmax_f32` decode kernel (matches `ops::softmax_rows`; expf may
    /// differ ~1 ULP).
    #[allow(dead_code)] // wired into `forward_device` (attention) next; W1-in-progress.
    pub(super) func_softmax: CudaFunction,
    /// Resolved `residual_add_f32` / `embedding_gather_f32` / `lm_head_f32` kernels.
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(super) func_residual: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_embed: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_lm_head: CudaFunction,
    /// The resolved `gqa_attention_decode_f32` kernel (M=1 decode attention).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(super) func_attn: CudaFunction,
    /// `gqa_attention_split_partial_f32` + `gqa_attention_combine_f32` — the
    /// flash-decoding (split-KV) attention pair (low-N occupancy fix). Resolved here
    /// and exercised by the `attn_split_kv_matches_direct_attention` equivalence gate;
    /// the resident decode wiring (`md_attn`/`gb_attn`) lands next and makes them load-bearing.
    #[allow(dead_code)]
    pub(super) func_attn_split_partial: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_attn_combine: CudaFunction,
    /// The resolved `act_quant_tiled_f32` kernel (on-device A8 quant for the tiled GEMM).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    pub(super) func_act_quant_tiled: CudaFunction,
    /// The resolved `scale_mul_f32` kernel (per-token act-scale fold).
    #[allow(dead_code)] // wired into the device GEMM next; W1-in-progress.
    pub(super) func_scale_mul: CudaFunction,
    /// The resolved `relu2_gate_f32` kernel (BitNet squared-ReLU FFN gate).
    #[allow(dead_code)] // wired into `forward_device` next; W1-in-progress.
    pub(super) func_relu2_gate: CudaFunction,
    /// Loaded v0.50 training backward PTX module (`--fmad=false`), kept alive so its
    /// gradient functions stay valid.
    pub(super) _grad_module: Arc<CudaModule>,
    /// The resolved `ternary_matmul_grad_a/_w/_s` kernels (f32 backward, ADR 0007).
    /// The f32 forward companion to the grad kernels (`Y = s·(A·Wᵀ)`), used by the QAT training
    /// step ([`super::train`]).
    pub(super) func_train_forward: CudaFunction,
    /// Gradient-checked against the `tritium-train` CPU vjp oracle; wired into the QAT training
    /// step ([`super::train`]).
    pub(super) func_grad_a: CudaFunction,
    pub(super) func_grad_w: CudaFunction,
    pub(super) func_grad_s: CudaFunction,
    /// ADR 0027 Track A: per-row SALT residual quantization on resident f32 buffers.
    #[allow(dead_code)] // consumed by DeviceTrainer in the next Track A slice
    pub(super) func_salt_quantize_fwd: CudaFunction,
    /// ADR 0027 Track A: fused AdamW update on resident master/moment buffers.
    #[allow(dead_code)] // consumed by DeviceTrainer in the next Track A slice
    pub(super) func_adamw_step: CudaFunction,
    /// ADR 0027 Track D: compact SALT pack, forward, and activation gradient.
    #[allow(dead_code)] // consumed by the Track D backend entry points below
    pub(super) func_salt_pack_training: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_forward: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_grad_a: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_forward_exact: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_grad_a_exact: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_forward_exact_tiled: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_grad_a_exact_tiled: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_forward_tiled: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_grad_a_tiled: CudaFunction,
    #[allow(dead_code)]
    pub(super) func_salt_training_embed: CudaFunction,
    /// plan 0043 P2.2 device-resident glue: elementwise silu/mul/add fwd/bwd + grad accumulate.
    pub(super) func_silu_fwd: CudaFunction,
    pub(super) func_silu_bwd: CudaFunction,
    pub(super) func_ew_mul_fwd: CudaFunction,
    pub(super) func_ew_mul_bwd: CudaFunction,
    pub(super) func_ew_add_fwd: CudaFunction,
    pub(super) func_accumulate: CudaFunction,
    /// plan 0043 P2.3 device-resident training RMSNorm (sequential order; forward, per-row inv, grads).
    pub(super) func_rmsnorm_train_fwd: CudaFunction,
    pub(super) func_rmsnorm_train_inv: CudaFunction,
    pub(super) func_rmsnorm_train_grad_x: CudaFunction,
    pub(super) func_rmsnorm_train_grad_w: CudaFunction,
    /// plan 0043 P2.4 device-resident attention glue.
    pub(super) func_softmax_fwd: CudaFunction,
    pub(super) func_softmax_bwd: CudaFunction,
    pub(super) func_causal_mask_fwd: CudaFunction,
    pub(super) func_causal_mask_bwd: CudaFunction,
    pub(super) func_rope_apply: CudaFunction,
    pub(super) func_slice_cols_fwd: CudaFunction,
    pub(super) func_copy_into_cols: CudaFunction,
    pub(super) func_transpose_fwd: CudaFunction,
    pub(super) func_embed_gather_fwd: CudaFunction,
    pub(super) func_embed_gather_bwd: CudaFunction,
    /// ADR 0027 Track B: deterministic segmented embedding gradient.
    pub(super) func_embed_gather_bwd_segmented: CudaFunction,
    pub(super) func_softmax_xent_bwd: CudaFunction,
    pub(super) func_scale_const: CudaFunction,
    /// Backend identifier, e.g. `"cuda:0"`.
    pub(super) device_id: String,
    /// Human-readable device name reported by the driver, e.g. `"NVIDIA H100"`.
    pub(super) device_name: String,
    /// The device's SM arch tag (`"sm_89"` on the 4090), part of the autotune
    /// [`CacheKey`] so a tuned tile is never reused across architectures.
    pub(super) sm_arch: String,
    /// The CUDA driver version (`cuDriverGetVersion`, e.g. `13030`), the cache
    /// invalidation axis — a driver bump can change the JIT'd SASS, so a stale tuned
    /// entry keyed under the old version is ignored.
    pub(super) cuda_version: u32,
    /// Process-lifetime cache of JIT-compiled IMMA functions, keyed by the exact
    /// [`TileConfig`] they were rendered for. Compiling a kernel via nvrtc is
    /// expensive, so each distinct tile is compiled at most once per process; the
    /// owning [`CudaModule`] is held alongside so the [`CudaFunction`] stays valid.
    /// A `Mutex` makes the backend `Sync` (the spec trait does not require it, but
    /// the runtime may share a backend across threads). Determinism is unaffected:
    /// the same tile always renders the same source → the same SASS → the same
    /// numerics, whether read from this cache or freshly compiled.
    pub(super) imma_jit: Mutex<HashMap<TileConfig, (Arc<CudaModule>, CudaFunction)>>,
    /// Per-(arch,dtype,shape-bucket,version) resolved winning tile, memoised in
    /// memory so a repeated shape does not re-hit the on-disk cache / re-tune. Seeded
    /// from the on-disk cache via [`tune_or_load`] on first use of a bucket.
    pub(super) tuned_tiles: Mutex<HashMap<CacheKey, TileConfig>>,
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
        let func_salt_quantize_fwd = grad_module
            .load_function(KERNEL_NAME_SALT_QUANTIZE_FWD)
            .map_err(|e| driver_err("resolve salt_quantize_forward kernel", &e))?;
        let func_adamw_step = grad_module
            .load_function(KERNEL_NAME_ADAMW_STEP)
            .map_err(|e| driver_err("resolve adamw_step kernel", &e))?;
        let func_salt_pack_training = grad_module
            .load_function(KERNEL_NAME_SALT_PACK_TRAINING)
            .map_err(|e| driver_err("resolve salt_pack_training kernel", &e))?;
        let func_salt_training_forward = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_FORWARD)
            .map_err(|e| driver_err("resolve salt_training_forward kernel", &e))?;
        let func_salt_training_grad_a = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_GRAD_A)
            .map_err(|e| driver_err("resolve salt_training_grad_a kernel", &e))?;
        let func_salt_training_forward_exact = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_FORWARD_EXACT)
            .map_err(|e| driver_err("resolve salt_training_forward_exact kernel", &e))?;
        let func_salt_training_grad_a_exact = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_GRAD_A_EXACT)
            .map_err(|e| driver_err("resolve salt_training_grad_a_exact kernel", &e))?;
        let func_salt_training_forward_exact_tiled = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_FORWARD_EXACT_TILED)
            .map_err(|e| driver_err("resolve salt_training_forward_exact_tiled kernel", &e))?;
        let func_salt_training_grad_a_exact_tiled = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_GRAD_A_EXACT_TILED)
            .map_err(|e| driver_err("resolve salt_training_grad_a_exact_tiled kernel", &e))?;
        let func_salt_training_forward_tiled = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_FORWARD_TILED)
            .map_err(|e| driver_err("resolve salt_training_forward_tiled kernel", &e))?;
        let func_salt_training_grad_a_tiled = grad_module
            .load_function(KERNEL_NAME_SALT_TRAINING_GRAD_A_TILED)
            .map_err(|e| driver_err("resolve salt_training_grad_a_tiled kernel", &e))?;
        let func_salt_training_embed =
            grad_module
                .load_function(KERNEL_NAME_SALT_TRAINING_EMBED)
                .map_err(|e| driver_err("resolve salt_training_embed_gather kernel", &e))?;
        // P2.2 device-resident glue ops (same `--fmad=false` train_grad.ptx module).
        let func_silu_fwd = grad_module
            .load_function(KERNEL_NAME_SILU_FWD)
            .map_err(|e| driver_err("resolve silu_forward kernel", &e))?;
        let func_silu_bwd = grad_module
            .load_function(KERNEL_NAME_SILU_BWD)
            .map_err(|e| driver_err("resolve silu_backward kernel", &e))?;
        let func_ew_mul_fwd = grad_module
            .load_function(KERNEL_NAME_EW_MUL_FWD)
            .map_err(|e| driver_err("resolve ew_mul_forward kernel", &e))?;
        let func_ew_mul_bwd = grad_module
            .load_function(KERNEL_NAME_EW_MUL_BWD)
            .map_err(|e| driver_err("resolve ew_mul_backward kernel", &e))?;
        let func_ew_add_fwd = grad_module
            .load_function(KERNEL_NAME_EW_ADD_FWD)
            .map_err(|e| driver_err("resolve ew_add_forward kernel", &e))?;
        let func_accumulate = grad_module
            .load_function(KERNEL_NAME_ACCUMULATE)
            .map_err(|e| driver_err("resolve accumulate kernel", &e))?;
        let func_rmsnorm_train_fwd = grad_module
            .load_function(KERNEL_NAME_RMSNORM_TRAIN_FWD)
            .map_err(|e| driver_err("resolve rmsnorm_train_forward kernel", &e))?;
        let func_rmsnorm_train_inv = grad_module
            .load_function(KERNEL_NAME_RMSNORM_TRAIN_INV)
            .map_err(|e| driver_err("resolve rmsnorm_train_inv kernel", &e))?;
        let func_rmsnorm_train_grad_x = grad_module
            .load_function(KERNEL_NAME_RMSNORM_TRAIN_GRAD_X)
            .map_err(|e| driver_err("resolve rmsnorm_train_grad_x kernel", &e))?;
        let func_rmsnorm_train_grad_w = grad_module
            .load_function(KERNEL_NAME_RMSNORM_TRAIN_GRAD_W)
            .map_err(|e| driver_err("resolve rmsnorm_train_grad_w kernel", &e))?;
        let func_softmax_fwd = grad_module
            .load_function(KERNEL_NAME_SOFTMAX_FWD)
            .map_err(|e| driver_err("resolve softmax_forward kernel", &e))?;
        let func_softmax_bwd = grad_module
            .load_function(KERNEL_NAME_SOFTMAX_BWD)
            .map_err(|e| driver_err("resolve softmax_backward kernel", &e))?;
        let func_causal_mask_fwd = grad_module
            .load_function(KERNEL_NAME_CAUSAL_MASK_FWD)
            .map_err(|e| driver_err("resolve causal_mask_forward kernel", &e))?;
        let func_causal_mask_bwd = grad_module
            .load_function(KERNEL_NAME_CAUSAL_MASK_BWD)
            .map_err(|e| driver_err("resolve causal_mask_backward kernel", &e))?;
        let func_rope_apply = grad_module
            .load_function(KERNEL_NAME_ROPE_APPLY)
            .map_err(|e| driver_err("resolve rope_apply kernel", &e))?;
        let func_slice_cols_fwd = grad_module
            .load_function(KERNEL_NAME_SLICE_COLS_FWD)
            .map_err(|e| driver_err("resolve slice_cols_forward kernel", &e))?;
        let func_copy_into_cols = grad_module
            .load_function(KERNEL_NAME_COPY_INTO_COLS)
            .map_err(|e| driver_err("resolve copy_into_cols kernel", &e))?;
        let func_transpose_fwd = grad_module
            .load_function(KERNEL_NAME_TRANSPOSE_FWD)
            .map_err(|e| driver_err("resolve transpose_forward kernel", &e))?;
        let func_embed_gather_fwd = grad_module
            .load_function(KERNEL_NAME_EMBED_GATHER_FWD)
            .map_err(|e| driver_err("resolve embed_gather_forward kernel", &e))?;
        let func_embed_gather_bwd = grad_module
            .load_function(KERNEL_NAME_EMBED_GATHER_BWD)
            .map_err(|e| driver_err("resolve embed_gather_backward kernel", &e))?;
        let func_embed_gather_bwd_segmented = grad_module
            .load_function(KERNEL_NAME_EMBED_GATHER_BWD_SEGMENTED)
            .map_err(|e| driver_err("resolve embed_gather_backward_segmented kernel", &e))?;
        let func_softmax_xent_bwd = grad_module
            .load_function(KERNEL_NAME_SOFTMAX_XENT_BWD)
            .map_err(|e| driver_err("resolve softmax_xent_backward kernel", &e))?;
        let func_scale_const = grad_module
            .load_function(KERNEL_NAME_SCALE_CONST)
            .map_err(|e| driver_err("resolve scale_const kernel", &e))?;

        let salt_v2_module = ctx
            .load_module(Ptx::from_src(SALT_V2_PTX))
            .map_err(|e| driver_err("load salt_v2 ptx", &e))?;
        let func_salt_v2_exact = salt_v2_module
            .load_function(KERNEL_NAME_SALT_V2_EXACT)
            .map_err(|e| driver_err("resolve salt_v2_forward_exact kernel", &e))?;

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
            _salt_v2_module: salt_v2_module,
            func_salt_v2_exact,
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
            func_salt_quantize_fwd,
            func_adamw_step,
            func_salt_pack_training,
            func_salt_training_forward,
            func_salt_training_grad_a,
            func_salt_training_forward_exact,
            func_salt_training_grad_a_exact,
            func_salt_training_forward_exact_tiled,
            func_salt_training_grad_a_exact_tiled,
            func_salt_training_forward_tiled,
            func_salt_training_grad_a_tiled,
            func_salt_training_embed,
            func_silu_fwd,
            func_silu_bwd,
            func_ew_mul_fwd,
            func_ew_mul_bwd,
            func_ew_add_fwd,
            func_accumulate,
            func_rmsnorm_train_fwd,
            func_rmsnorm_train_inv,
            func_rmsnorm_train_grad_x,
            func_rmsnorm_train_grad_w,
            func_softmax_fwd,
            func_softmax_bwd,
            func_causal_mask_fwd,
            func_causal_mask_bwd,
            func_rope_apply,
            func_slice_cols_fwd,
            func_copy_into_cols,
            func_transpose_fwd,
            func_embed_gather_fwd,
            func_embed_gather_bwd,
            func_embed_gather_bwd_segmented,
            func_softmax_xent_bwd,
            func_scale_const,
            device_id: format!("cuda:{ordinal}"),
            device_name,
            sm_arch,
            cuda_version,
            imma_jit: Mutex::new(HashMap::new()),
            tuned_tiles: Mutex::new(HashMap::new()),
        })
    }

    /// Packed bytes per weight row for `k` trits in TQ2_0.
    pub(super) fn row_bytes(k: usize) -> usize {
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

    // ───────────────────────── device-resident training (plan 0043 P2) ─────────────────────────
    // The methods above round-trip host↔device per call (Phase 1). For the device-resident tape,
    // activations and gradients live in VRAM across the whole fwd+bwd step: upload the leaves ONCE,
    // chain the kernels below on the resident buffers (no per-op copies), download only the final
    // result. Each kernel is the SAME `train_grad.cu` code as its host-slice sibling, so the resident
    // path stays bit-exact vs the CPU tape.

    /// Upload a host slice to a resident device buffer (htod). Leaves of the device tape.
    ///
    /// # Errors
    /// Device failures via the cudarc mapping.
    // Test-exercised (`resident_matmul_chain_matches_cpu_tape`) until the `DeviceTape` wires these
    // resident primitives into the full model forward+backward (plan 0043 P2.5).
    #[allow(dead_code)]
    pub(crate) fn dev_upload(&self, host: &[f32]) -> Result<CudaSlice<f32>, BackendError> {
        self.stream
            .clone_htod(host)
            .map_err(|e| driver_err("dev_upload htod", &e))
    }

    /// Create the non-blocking transfer stream used by host-offloaded AdamW.
    /// This must happen before its persistent device slots are allocated so
    /// cudarc installs per-buffer event tracking for cross-stream ordering.
    pub(crate) fn offload_transfer_stream(&self) -> Result<Arc<CudaStream>, BackendError> {
        self.stream
            .context()
            .new_stream()
            .map_err(|e| driver_err("create host-offload transfer stream", &e))
    }

    /// Allocate initialized page-locked staging memory.
    #[allow(unsafe_code)]
    pub(crate) fn offload_alloc_pinned_zeros(
        &self,
        len: usize,
    ) -> Result<PinnedHostSlice<f32>, BackendError> {
        // SAFETY: every `f32` bit pattern is valid and the full allocation is
        // initialized immediately, before it can be observed by a copy.
        let mut pinned = unsafe { self.stream.context().alloc_pinned::<f32>(len) }
            .map_err(|e| driver_err("allocate host-offload pinned staging", &e))?;
        pinned
            .as_mut_slice()
            .map_err(|e| driver_err("initialize host-offload pinned staging", &e))?
            .fill(0.0);
        Ok(pinned)
    }

    /// Enqueue one logical pinned-host prefix on the transfer stream.
    pub(crate) fn offload_htod_prefix(
        &self,
        transfer: &Arc<CudaStream>,
        host: &mut PinnedHostSlice<f32>,
        len: usize,
        device: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        if transfer.context() != self.stream.context() || !self.same_context(device) {
            return Err(BackendError::InvalidInput(
                "host-offload transfer buffers belong to a different CUDA context".into(),
            ));
        }
        if len > device.len() {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: device.len(),
            });
        }
        let prefix = PinnedPrefix::new(host, len)?;
        transfer
            .memcpy_htod(&prefix, device)
            .map_err(|e| driver_err("enqueue host-offload htod", &e))
    }

    /// Enqueue one logical device prefix into page-locked host staging.
    pub(crate) fn offload_dtoh_prefix(
        &self,
        transfer: &Arc<CudaStream>,
        device: &CudaSlice<f32>,
        len: usize,
        host: &mut PinnedHostSlice<f32>,
    ) -> Result<(), BackendError> {
        if transfer.context() != self.stream.context() || !self.same_context(device) {
            return Err(BackendError::InvalidInput(
                "host-offload transfer buffers belong to a different CUDA context".into(),
            ));
        }
        if len > device.len() {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: device.len(),
            });
        }
        let source = device.slice(..len);
        let mut prefix = PinnedPrefix::new(host, len)?;
        transfer
            .memcpy_dtoh(&source, &mut prefix)
            .map_err(|e| driver_err("enqueue host-offload dtoh", &e))
    }

    /// Drain both streams before exposing updated host state.
    pub(crate) fn offload_synchronize(
        &self,
        transfer: &Arc<CudaStream>,
    ) -> Result<(), BackendError> {
        if transfer.context() != self.stream.context() {
            return Err(BackendError::InvalidInput(
                "host-offload transfer stream belongs to a different CUDA context".into(),
            ));
        }
        self.stream
            .synchronize()
            .map_err(|e| driver_err("synchronize host-offload compute stream", &e))?;
        transfer
            .synchronize()
            .map_err(|e| driver_err("synchronize host-offload transfer stream", &e))
    }

    /// Allocate a zeroed resident device buffer of `n` f32 (fresh activation / grad accumulator).
    ///
    /// # Errors
    /// Device failures via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn dev_alloc_zeros(&self, n: usize) -> Result<CudaSlice<f32>, BackendError> {
        self.stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| driver_err("dev_alloc_zeros", &e))
    }

    /// Download a resident device buffer to host (dtoh). The single copy at the end of the step.
    ///
    /// # Errors
    /// Device failures via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn dev_download(
        &self,
        d: &CudaSlice<f32>,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        self.stream
            .memcpy_dtoh(d, out)
            .map_err(|e| driver_err("dev_download dtoh", &e))
    }

    /// Whether `buffer` belongs to this backend's CUDA context. Context value
    /// equality deliberately permits a different stream (or backend handle)
    /// retaining the same primary context.
    pub(crate) fn same_context<T>(&self, buffer: &CudaSlice<T>) -> bool {
        self.stream.context() == buffer.context()
    }

    /// The backend stream used to join an NCCL communicator. Keeping this
    /// accessor crate-private prevents callers from bypassing the backend's
    /// stream ordering while allowing the sibling NCCL module to enqueue
    /// collectives directly against resident training buffers.
    #[cfg(feature = "nccl")]
    pub(crate) fn nccl_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }

    #[cfg(test)]
    pub(crate) fn dev_synchronize(&self) -> Result<(), BackendError> {
        self.stream
            .synchronize()
            .map_err(|e| driver_err("synchronize resident training stream", &e))
    }

    #[cfg(test)]
    pub(crate) fn dev_name(&self) -> &str {
        &self.device_name
    }

    /// Greedy multi-plane SALT reconstruction on resident f32 buffers, matching
    /// [`tritium_train::ops::ste::salt_quantize_forward`]. Scale reduction is
    /// sequential within each row; rows execute independently in parallel.
    /// `d_residual` is caller-owned scratch and is fully overwritten.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] for unsupported plane counts or dimensions;
    /// [`BackendError::ShapeMismatch`] for undersized buffers; device failures
    /// through the cudarc mapping.
    #[allow(dead_code)] // kernel parity gate lands before DeviceTrainer wiring
    pub(crate) fn salt_quantize_forward_dev(
        &self,
        d_master: &CudaSlice<f32>,
        d_residual: &mut CudaSlice<f32>,
        d_quantized: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        planes: usize,
    ) -> Result<(), BackendError> {
        if !(1..=3).contains(&planes) {
            return Err(BackendError::InvalidInput(format!(
                "SALT plane count must be in 1..=3, got {planes}"
            )));
        }
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| BackendError::InvalidInput("SALT shape overflows usize".into()))?;
        let rows_i = i32::try_from(rows)
            .map_err(|_| BackendError::InvalidInput("SALT rows exceed i32::MAX".into()))?;
        let cols_i = i32::try_from(cols)
            .map_err(|_| BackendError::InvalidInput("SALT cols exceed i32::MAX".into()))?;
        let planes_i = i32::try_from(planes)
            .map_err(|_| BackendError::InvalidInput("SALT planes exceed i32::MAX".into()))?;
        if d_master.len() < len || d_residual.len() < len || d_quantized.len() < len {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: d_master.len().min(d_residual.len()).min(d_quantized.len()),
            });
        }
        if len == 0 {
            return Ok(());
        }

        let mut launch = self.stream.launch_builder(&self.func_salt_quantize_fwd);
        launch
            .arg(d_master)
            .arg(d_residual)
            .arg(d_quantized)
            .arg(&rows_i)
            .arg(&cols_i)
            .arg(&planes_i);
        // SAFETY: kernel receives three buffers of at least rows*cols and one
        // thread per row; all dimensions fit its signed 32-bit ABI.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch salt_quantize_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Quantize a resident latent master into compact training-only SALT planes.
    /// The returned handle owns plane-major TQ2-addressed codes and external f32
    /// per-row scales; `d_residual` is reusable caller-owned scratch.
    ///
    /// # Errors
    /// Invalid plane counts, dimensions, cross-context buffers, allocation or
    /// launch failures, and undersized master/scratch buffers are typed errors.
    #[allow(dead_code)] // Track D gate precedes DeviceTape's SaltMatmul wiring
    pub(crate) fn pack_training_salt(
        &self,
        d_master: &CudaSlice<f32>,
        d_residual: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
        planes: usize,
    ) -> Result<TrainingSaltLinear, BackendError> {
        if !(1..=3).contains(&planes) {
            return Err(BackendError::InvalidInput(format!(
                "SALT plane count must be in 1..=3, got {planes}"
            )));
        }
        Self::check_grad_launch_bounds(1, n, k)?;
        if !self.same_context(d_master) || !self.same_context(d_residual) {
            return Err(BackendError::InvalidInput(
                "SALT pack buffer belongs to a different CUDA context".into(),
            ));
        }
        let dense_len = n
            .checked_mul(k)
            .ok_or_else(|| BackendError::InvalidInput("SALT shape overflows usize".into()))?;
        if d_master.len() < dense_len || d_residual.len() < dense_len {
            return Err(BackendError::ShapeMismatch {
                expected: dense_len,
                got: d_master.len().min(d_residual.len()),
            });
        }
        let row_bytes = k
            .div_ceil(TRAINING_SALT_QK)
            .checked_mul(TRAINING_SALT_QS_BYTES)
            .ok_or_else(|| BackendError::InvalidInput("SALT row bytes overflow usize".into()))?;
        let packed_len = planes
            .checked_mul(n)
            .and_then(|v| v.checked_mul(row_bytes))
            .ok_or_else(|| BackendError::InvalidInput("SALT packed bytes overflow usize".into()))?;
        let scale_len = planes
            .checked_mul(n)
            .ok_or_else(|| BackendError::InvalidInput("SALT scale count overflows usize".into()))?;
        let codes = self
            .stream
            .alloc_zeros::<u8>(packed_len)
            .map_err(|e| driver_err("alloc compact SALT codes", &e))?;
        let scales = self
            .stream
            .alloc_zeros::<f32>(scale_len)
            .map_err(|e| driver_err("alloc compact SALT scales", &e))?;
        let mut weight = TrainingSaltLinear {
            codes,
            scales,
            n,
            k,
            row_bytes,
            planes,
        };
        self.repack_training_salt(d_master, d_residual, &mut weight)?;
        Ok(weight)
    }

    /// Refresh an existing compact SALT handle after its latent master changes.
    /// No allocation occurs; code and scale buffers are overwritten in place.
    ///
    /// # Errors
    /// Shape/context violations and device launch failures are typed errors.
    #[allow(dead_code)] // Track D gate precedes DeviceTrainer's step wiring
    pub(crate) fn repack_training_salt(
        &self,
        d_master: &CudaSlice<f32>,
        d_residual: &mut CudaSlice<f32>,
        weight: &mut TrainingSaltLinear,
    ) -> Result<(), BackendError> {
        Self::check_grad_launch_bounds(1, weight.n, weight.k)?;
        self.validate_training_salt(weight)?;
        if !self.same_context(d_master) || !self.same_context(d_residual) {
            return Err(BackendError::InvalidInput(
                "SALT pack buffer belongs to a different CUDA context".into(),
            ));
        }
        let dense_len = weight.n * weight.k;
        if d_master.len() < dense_len || d_residual.len() < dense_len {
            return Err(BackendError::ShapeMismatch {
                expected: dense_len,
                got: d_master.len().min(d_residual.len()),
            });
        }
        if dense_len == 0 {
            return Ok(());
        }
        let (_, ni, ki, pi, rbi) = weight.kernel_dims(1)?;
        let mut launch = self.stream.launch_builder(&self.func_salt_pack_training);
        launch
            .arg(d_master)
            .arg(d_residual)
            .arg(&mut weight.codes)
            .arg(&mut weight.scales)
            .arg(&ni)
            .arg(&ki)
            .arg(&pi)
            .arg(&rbi);
        // SAFETY: one thread owns each row; all flat products and scalar
        // dimensions were checked, and every buffer covers the full shape.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(weight.n))
                .map_err(|e| driver_err("launch salt_pack_training", &e))?;
        }
        Ok(())
    }

    /// Exact packed SALT forward. It reconstructs one weight in plane order,
    /// then contracts in K order, matching dense SALT materialization without
    /// allocating a dense `[N,K]` buffer.
    ///
    /// # Errors
    /// Shape/context violations and device launch failures are typed errors.
    pub(crate) fn training_salt_forward(
        &self,
        d_a: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_forward_impl(d_a, weight, m, d_y, TrainingSaltDispatch::Exact)
    }

    /// Plane-order packed SALT forward optimized for throughput. This may
    /// differ from dense-order arithmetic by ordinary f32 reassociation.
    pub(crate) fn training_salt_forward_fast(
        &self,
        d_a: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_forward_impl(d_a, weight, m, d_y, TrainingSaltDispatch::Fast)
    }

    /// Scalar dense-order oracle for the shared-memory exact kernel.
    #[cfg(test)]
    pub(crate) fn training_salt_forward_exact_scalar(
        &self,
        d_a: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_forward_impl(d_a, weight, m, d_y, TrainingSaltDispatch::ScalarExact)
    }

    #[cfg(test)]
    pub(crate) fn training_salt_forward_scalar(
        &self,
        d_a: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_forward_impl(d_a, weight, m, d_y, TrainingSaltDispatch::ScalarFast)
    }

    pub(super) fn training_salt_forward_tiled_supported(m: usize, n: usize, k: usize) -> bool {
        // Four M rows amortize the cooperative activation/code load. Keep
        // skinny and sub-block contractions on the scalar oracle.
        m >= TRAINING_SALT_TILE_M as usize && n >= TRAINING_SALT_TILE_X as usize && k >= 256
    }

    pub(super) fn training_salt_forward_exact_tiled_supported(
        _m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        // Cooperative reconstruction wins once K spans half a reduction tile
        // and either N fills a warp or the long K reduction amortizes a tail
        // block. Tiny contractions retain the scalar oracle.
        k >= TRAINING_SALT_TILE_X as usize && (n >= TRAINING_SALT_TILE_X as usize || k >= 256)
    }

    fn training_salt_forward_impl(
        &self,
        d_a: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_y: &mut CudaSlice<f32>,
        dispatch: TrainingSaltDispatch,
    ) -> Result<(), BackendError> {
        Self::check_grad_launch_bounds(m, weight.n, weight.k)?;
        self.validate_training_salt(weight)?;
        if !self.same_context(d_a) || !self.same_context(d_y) {
            return Err(BackendError::InvalidInput(
                "SALT forward buffer belongs to a different CUDA context".into(),
            ));
        }
        let act_len = m * weight.k;
        let out_len = m * weight.n;
        if d_a.len() < act_len || d_y.len() < out_len {
            return Err(BackendError::ShapeMismatch {
                expected: if d_a.len() < act_len {
                    act_len
                } else {
                    out_len
                },
                got: if d_a.len() < act_len {
                    d_a.len()
                } else {
                    d_y.len()
                },
            });
        }
        if out_len == 0 {
            return Ok(());
        }
        let (mi, ni, ki, pi, rbi) = weight.kernel_dims(m)?;
        let use_exact = matches!(
            dispatch,
            TrainingSaltDispatch::Exact | TrainingSaltDispatch::ScalarExact
        );
        let use_exact_tiled = matches!(dispatch, TrainingSaltDispatch::Exact)
            && Self::training_salt_forward_exact_tiled_supported(m, weight.n, weight.k);
        let use_fast_tiled = matches!(dispatch, TrainingSaltDispatch::Fast)
            && Self::training_salt_forward_tiled_supported(m, weight.n, weight.k);
        let function = if use_exact_tiled {
            &self.func_salt_training_forward_exact_tiled
        } else if use_exact {
            &self.func_salt_training_forward_exact
        } else if use_fast_tiled {
            &self.func_salt_training_forward_tiled
        } else {
            &self.func_salt_training_forward
        };
        let cfg = if use_exact_tiled || use_fast_tiled {
            let tile_m = if use_exact_tiled {
                TRAINING_SALT_EXACT_TILE_M
            } else {
                TRAINING_SALT_TILE_M
            };
            LaunchConfig {
                grid_dim: (
                    (weight.n as u32).div_ceil(TRAINING_SALT_TILE_X),
                    (m as u32).div_ceil(tile_m),
                    1,
                ),
                block_dim: (TRAINING_SALT_TILE_X, tile_m, 1),
                shared_mem_bytes: 0,
            }
        } else {
            Self::elementwise_cfg(out_len)
        };
        let mut launch = self.stream.launch_builder(function);
        launch
            .arg(d_a)
            .arg(&weight.codes)
            .arg(&weight.scales)
            .arg(d_y)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki)
            .arg(&pi)
            .arg(&rbi);
        // SAFETY: validated handle/input/output buffers cover MxK, packed
        // TxNxrow_bytes, TxN and MxN; flat indices fit the kernel ABI.
        #[allow(unsafe_code)]
        unsafe {
            launch.launch(cfg).map_err(|e| {
                driver_err(
                    if use_exact_tiled {
                        "launch salt_training_forward_exact_tiled"
                    } else if use_exact {
                        "launch salt_training_forward_exact"
                    } else if use_fast_tiled {
                        "launch salt_training_forward_tiled"
                    } else {
                        "launch salt_training_forward"
                    },
                    &e,
                )
            })?;
        }
        Ok(())
    }

    /// Exact activation VJP through packed SALT planes. Each weight is
    /// reconstructed in plane order before the N-order contraction. The
    /// latent-master weight gradient remains the ordinary fp STE path and is
    /// intentionally not stored in this packed handle.
    ///
    /// # Errors
    /// Shape/context violations and device launch failures are typed errors.
    pub(crate) fn training_salt_grad_a(
        &self,
        d_gy: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_grad_a_impl(d_gy, weight, m, d_ga, TrainingSaltDispatch::Exact)
    }

    /// Plane-order packed SALT activation VJP optimized for throughput. This
    /// may differ from dense-order arithmetic by ordinary f32 reassociation.
    pub(crate) fn training_salt_grad_a_fast(
        &self,
        d_gy: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_grad_a_impl(d_gy, weight, m, d_ga, TrainingSaltDispatch::Fast)
    }

    /// Scalar dense-order oracle for the shared-memory exact kernel.
    #[cfg(test)]
    pub(crate) fn training_salt_grad_a_exact_scalar(
        &self,
        d_gy: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_grad_a_impl(d_gy, weight, m, d_ga, TrainingSaltDispatch::ScalarExact)
    }

    #[cfg(test)]
    pub(crate) fn training_salt_grad_a_scalar(
        &self,
        d_gy: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        self.training_salt_grad_a_impl(d_gy, weight, m, d_ga, TrainingSaltDispatch::ScalarFast)
    }

    pub(super) fn training_salt_grad_a_tiled_supported(m: usize, n: usize, k: usize) -> bool {
        m >= TRAINING_SALT_TILE_M as usize && n >= TRAINING_SALT_TILE_X as usize && k >= 128
    }

    pub(super) fn training_salt_grad_a_exact_tiled_supported(
        _m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        n >= TRAINING_SALT_TILE_X as usize && k >= TRAINING_SALT_TILE_X as usize
    }

    fn training_salt_grad_a_impl(
        &self,
        d_gy: &CudaSlice<f32>,
        weight: &TrainingSaltLinear,
        m: usize,
        d_ga: &mut CudaSlice<f32>,
        dispatch: TrainingSaltDispatch,
    ) -> Result<(), BackendError> {
        Self::check_grad_launch_bounds(m, weight.n, weight.k)?;
        self.validate_training_salt(weight)?;
        if !self.same_context(d_gy) || !self.same_context(d_ga) {
            return Err(BackendError::InvalidInput(
                "SALT grad_a buffer belongs to a different CUDA context".into(),
            ));
        }
        let gy_len = m * weight.n;
        let ga_len = m * weight.k;
        if d_gy.len() < gy_len || d_ga.len() < ga_len {
            return Err(BackendError::ShapeMismatch {
                expected: if d_gy.len() < gy_len { gy_len } else { ga_len },
                got: if d_gy.len() < gy_len {
                    d_gy.len()
                } else {
                    d_ga.len()
                },
            });
        }
        if ga_len == 0 {
            return Ok(());
        }
        let (mi, ni, ki, pi, rbi) = weight.kernel_dims(m)?;
        let use_exact = matches!(
            dispatch,
            TrainingSaltDispatch::Exact | TrainingSaltDispatch::ScalarExact
        );
        let use_exact_tiled = matches!(dispatch, TrainingSaltDispatch::Exact)
            && Self::training_salt_grad_a_exact_tiled_supported(m, weight.n, weight.k);
        let use_fast_tiled = matches!(dispatch, TrainingSaltDispatch::Fast)
            && Self::training_salt_grad_a_tiled_supported(m, weight.n, weight.k);
        let function = if use_exact_tiled {
            &self.func_salt_training_grad_a_exact_tiled
        } else if use_exact {
            &self.func_salt_training_grad_a_exact
        } else if use_fast_tiled {
            &self.func_salt_training_grad_a_tiled
        } else {
            &self.func_salt_training_grad_a
        };
        let cfg = if use_exact_tiled || use_fast_tiled {
            let tile_m = if use_exact_tiled {
                TRAINING_SALT_EXACT_TILE_M
            } else {
                TRAINING_SALT_TILE_M
            };
            LaunchConfig {
                grid_dim: (
                    (weight.k as u32).div_ceil(TRAINING_SALT_TILE_X),
                    (m as u32).div_ceil(tile_m),
                    1,
                ),
                block_dim: (TRAINING_SALT_TILE_X, tile_m, 1),
                shared_mem_bytes: 0,
            }
        } else {
            Self::elementwise_cfg(ga_len)
        };
        let mut launch = self.stream.launch_builder(function);
        launch
            .arg(d_gy)
            .arg(&weight.codes)
            .arg(&weight.scales)
            .arg(d_ga)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki)
            .arg(&pi)
            .arg(&rbi);
        // SAFETY: validated buffers cover MxN, packed TxNxrow_bytes, TxN and
        // MxK; one thread writes each activation-gradient element.
        #[allow(unsafe_code)]
        unsafe {
            launch.launch(cfg).map_err(|e| {
                driver_err(
                    if use_exact_tiled {
                        "launch salt_training_grad_a_exact_tiled"
                    } else if use_exact {
                        "launch salt_training_grad_a_exact"
                    } else if use_fast_tiled {
                        "launch salt_training_grad_a_tiled"
                    } else {
                        "launch salt_training_grad_a"
                    },
                    &e,
                )
            })?;
        }
        Ok(())
    }

    /// Gather token rows directly from compact training SALT planes. Token ids
    /// must already have been host-validated against `weight.rows()`; the
    /// kernel retains a defensive bounds guard before any packed-row read.
    ///
    /// # Errors
    /// Shape/context violations and device launch failures are typed errors.
    pub(crate) fn training_salt_embed_forward(
        &self,
        weight: &TrainingSaltLinear,
        d_tokens: &CudaSlice<i32>,
        seq: usize,
        d_out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        Self::check_grad_launch_bounds(seq, weight.n, weight.k)?;
        self.validate_training_salt(weight)?;
        if !self.same_context(d_tokens) || !self.same_context(d_out) {
            return Err(BackendError::InvalidInput(
                "SALT embedding buffer belongs to a different CUDA context".into(),
            ));
        }
        let out_len = seq.checked_mul(weight.k).ok_or_else(|| {
            BackendError::InvalidInput("SALT embedding shape overflows usize".into())
        })?;
        if d_tokens.len() < seq || d_out.len() < out_len {
            return Err(BackendError::ShapeMismatch {
                expected: if d_tokens.len() < seq { seq } else { out_len },
                got: if d_tokens.len() < seq {
                    d_tokens.len()
                } else {
                    d_out.len()
                },
            });
        }
        if out_len == 0 {
            return Ok(());
        }
        let (seq_i, vocab_i, dim_i, planes_i, row_bytes_i) = weight.kernel_dims(seq)?;
        let mut launch = self.stream.launch_builder(&self.func_salt_training_embed);
        launch
            .arg(&weight.codes)
            .arg(&weight.scales)
            .arg(d_tokens)
            .arg(d_out)
            .arg(&seq_i)
            .arg(&vocab_i)
            .arg(&dim_i)
            .arg(&planes_i)
            .arg(&row_bytes_i);
        // SAFETY: validated buffers cover the packed handle, seq token ids and
        // seq*dim outputs; the kernel guards each token before row addressing.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(out_len))
                .map_err(|e| driver_err("launch salt_training_embed_gather", &e))?;
        }
        Ok(())
    }

    fn validate_training_salt(&self, weight: &TrainingSaltLinear) -> Result<(), BackendError> {
        if !self.same_context(&weight.codes) || !self.same_context(&weight.scales) {
            return Err(BackendError::InvalidInput(
                "packed SALT handle belongs to a different CUDA context".into(),
            ));
        }
        let expected_row_bytes = weight
            .k
            .div_ceil(TRAINING_SALT_QK)
            .checked_mul(TRAINING_SALT_QS_BYTES)
            .ok_or_else(|| BackendError::InvalidInput("SALT row bytes overflow usize".into()))?;
        let expected_codes = weight
            .planes
            .checked_mul(weight.n)
            .and_then(|v| v.checked_mul(expected_row_bytes))
            .ok_or_else(|| BackendError::InvalidInput("SALT packed bytes overflow usize".into()))?;
        let expected_scales = weight
            .planes
            .checked_mul(weight.n)
            .ok_or_else(|| BackendError::InvalidInput("SALT scale count overflows usize".into()))?;
        if !(1..=3).contains(&weight.planes)
            || weight.row_bytes != expected_row_bytes
            || weight.codes.len() != expected_codes
            || weight.scales.len() != expected_scales
        {
            return Err(BackendError::InvalidInput(
                "packed SALT handle metadata does not match its buffers".into(),
            ));
        }
        Ok(())
    }

    /// Apply one fused AdamW update to resident parameter and moment buffers.
    /// Bias-correction scalars are computed on the host with the CPU optimizer's
    /// exact `powi`/saturating-step contract; the kernel only performs the
    /// independent element updates in matching operation order.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] when `step == 0` or the element count
    /// exceeds the kernel ABI; [`BackendError::ShapeMismatch`] when buffer
    /// lengths differ; device failures through the cudarc mapping.
    #[allow(dead_code)] // kernel parity gate lands before DeviceTrainer wiring
    pub(crate) fn adamw_step_dev(
        &self,
        d_param: &mut CudaSlice<f32>,
        d_grad: &CudaSlice<f32>,
        d_m: &mut CudaSlice<f32>,
        d_v: &mut CudaSlice<f32>,
        step: u64,
        opt: &tritium_train::AdamW,
    ) -> Result<(), BackendError> {
        if step == 0 {
            return Err(BackendError::InvalidInput(
                "AdamW step index is 1-based".into(),
            ));
        }
        let len = d_param.len();
        for got in [d_grad.len(), d_m.len(), d_v.len()] {
            if got != len {
                return Err(BackendError::ShapeMismatch { expected: len, got });
            }
        }
        if len == 0 {
            return Ok(());
        }
        let len_i = i32::try_from(len)
            .map_err(|_| BackendError::InvalidInput("AdamW length exceeds i32::MAX".into()))?;
        let exp = i32::try_from(step).unwrap_or(i32::MAX);
        let bc1 = 1.0 - opt.beta1.powi(exp);
        let bc2 = 1.0 - opt.beta2.powi(exp);
        let shrink = 1.0 - opt.lr * opt.weight_decay;
        let one_minus_beta1 = 1.0 - opt.beta1;
        let one_minus_beta2 = 1.0 - opt.beta2;

        let mut launch = self.stream.launch_builder(&self.func_adamw_step);
        launch
            .arg(d_param)
            .arg(d_grad)
            .arg(d_m)
            .arg(d_v)
            .arg(&len_i)
            .arg(&opt.lr)
            .arg(&opt.beta1)
            .arg(&opt.beta2)
            .arg(&one_minus_beta1)
            .arg(&one_minus_beta2)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&opt.eps)
            .arg(&shrink);
        // SAFETY: every device buffer has exactly `len` elements and every
        // scalar matches the kernel ABI; one thread owns one element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(len))
                .map_err(|e| driver_err("launch adamw_step_dev", &e))?;
        }
        Ok(())
    }

    /// Apply AdamW to a logical prefix of persistent offload slots. The views
    /// retain the parent buffers' cudarc events, so the default compute stream
    /// waits for the transfer-stream H2D and a later D2H waits for this kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adamw_step_dev_prefix(
        &self,
        d_param: &mut CudaSlice<f32>,
        d_grad: &CudaSlice<f32>,
        d_m: &mut CudaSlice<f32>,
        d_v: &mut CudaSlice<f32>,
        len: usize,
        step: u64,
        opt: &tritium_train::AdamW,
    ) -> Result<(), BackendError> {
        if step == 0 {
            return Err(BackendError::InvalidInput(
                "AdamW step index is 1-based".into(),
            ));
        }
        if d_grad.len() != len {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: d_grad.len(),
            });
        }
        for got in [d_param.len(), d_m.len(), d_v.len()] {
            if got < len {
                return Err(BackendError::ShapeMismatch { expected: len, got });
            }
        }
        if !self.same_context(d_param)
            || !self.same_context(d_grad)
            || !self.same_context(d_m)
            || !self.same_context(d_v)
        {
            return Err(BackendError::InvalidInput(
                "host-offload AdamW buffers belong to a different CUDA context".into(),
            ));
        }
        if len == 0 {
            return Ok(());
        }
        let len_i = i32::try_from(len)
            .map_err(|_| BackendError::InvalidInput("AdamW length exceeds i32::MAX".into()))?;
        let exp = i32::try_from(step).unwrap_or(i32::MAX);
        let bc1 = 1.0 - opt.beta1.powi(exp);
        let bc2 = 1.0 - opt.beta2.powi(exp);
        let shrink = 1.0 - opt.lr * opt.weight_decay;
        let one_minus_beta1 = 1.0 - opt.beta1;
        let one_minus_beta2 = 1.0 - opt.beta2;
        let mut param = d_param.slice_mut(..len);
        let mut first_moment = d_m.slice_mut(..len);
        let mut second_moment = d_v.slice_mut(..len);

        let mut launch = self.stream.launch_builder(&self.func_adamw_step);
        launch
            .arg(&mut param)
            .arg(d_grad)
            .arg(&mut first_moment)
            .arg(&mut second_moment)
            .arg(&len_i)
            .arg(&opt.lr)
            .arg(&opt.beta1)
            .arg(&opt.beta2)
            .arg(&one_minus_beta1)
            .arg(&one_minus_beta2)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&opt.eps)
            .arg(&shrink);
        // SAFETY: each view covers exactly `len` initialized elements and the
        // gradient has the same length; one thread owns one element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(len))
                .map_err(|e| driver_err("launch host-offload adamw prefix", &e))?;
        }
        Ok(())
    }

    /// `Y[m,n] = s[n]·Σ_k A[m,k]·W[n,k]` on ALREADY-RESIDENT buffers — the device-resident companion
    /// to [`train_forward`](Self::train_forward): same kernel, same `--fmad=false` sequential
    /// reduction, no htod/dtoh. `d_y` must be preallocated `m*n` (e.g. via [`dev_alloc_zeros`]).
    ///
    /// # Errors
    /// [`BackendError`] on launch-bound violation or device failure.
    #[allow(dead_code)]
    pub(crate) fn matmul_forward_dev(
        &self,
        d_a: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        d_s: &CudaSlice<f32>,
        shape: GemmShape,
        d_y: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        Self::check_grad_launch_bounds(m, n, k)?;
        // Guard undersized resident buffers: the kernel indexes inputs unbounded and writes the
        // whole [0, m*n) output, so a too-small buffer is device UB (OOB read / VRAM corruption),
        // not a clean error. `CudaSlice::len()` is a host-side value — free to check. `<` (not `==`)
        // so an oversized `s` (e.g. one shared ones buffer across shapes) is still allowed.
        if d_a.len() < m * k || d_w.len() < n * k || d_s.len() < n || d_y.len() < m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: d_y.len(),
            });
        }
        if m * n == 0 || k == 0 {
            return Ok(()); // k==0 ⇒ Y = s·0 = 0, and d_y is caller-zeroed.
        }
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((m * n) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_train_forward);
        launch
            .arg(d_a)
            .arg(d_w)
            .arg(d_s)
            .arg(d_y)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: same signature/contract as `train_forward`, on resident buffers.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch matmul_forward_dev", &e))?;
        }
        Ok(())
    }

    /// `gA[m,k] = Σ_n gy[m,n]·s[n]·W[n,k]` on resident buffers (device-resident companion to
    /// [`grad_a`](Self::grad_a)). `d_ga` preallocated `m*k`; the kernel writes every element.
    ///
    /// # Errors
    /// [`BackendError`] on launch-bound violation or device failure.
    #[allow(dead_code)]
    pub(crate) fn grad_a_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        d_s: &CudaSlice<f32>,
        shape: GemmShape,
        d_ga: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        Self::check_grad_launch_bounds(m, n, k)?;
        if d_gy.len() < m * n || d_w.len() < n * k || d_s.len() < n || d_ga.len() < m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: d_ga.len(),
            });
        }
        if m * k == 0 {
            return Ok(());
        }
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((m * k) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_grad_a);
        launch
            .arg(d_gy)
            .arg(d_w)
            .arg(d_s)
            .arg(d_ga)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: same signature/contract as `grad_a`, on resident buffers.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch grad_a_dev", &e))?;
        }
        Ok(())
    }

    /// `gW[n,k] = Σ_m gy[m,n]·s[n]·A[m,k]` on resident buffers (device-resident companion to
    /// [`grad_w`](Self::grad_w)). `d_gw` preallocated `n*k`; the kernel writes every element.
    ///
    /// # Errors
    /// [`BackendError`] on launch-bound violation or device failure.
    #[allow(dead_code)]
    pub(crate) fn grad_w_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        d_a: &CudaSlice<f32>,
        d_s: &CudaSlice<f32>,
        shape: GemmShape,
        d_gw: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        Self::check_grad_launch_bounds(m, n, k)?;
        if d_gy.len() < m * n || d_a.len() < m * k || d_s.len() < n || d_gw.len() < n * k {
            return Err(BackendError::ShapeMismatch {
                expected: n * k,
                got: d_gw.len(),
            });
        }
        if n * k == 0 {
            return Ok(());
        }
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let threads = THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (((n * k) as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_grad_w);
        launch
            .arg(d_gy)
            .arg(d_a)
            .arg(d_s)
            .arg(d_gw)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: same signature/contract as `grad_w`, on resident buffers.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch grad_w_dev", &e))?;
        }
        Ok(())
    }

    /// Launch config for a 1-thread-per-element elementwise kernel over `n` f32.
    fn elementwise_cfg(n: usize) -> LaunchConfig {
        let threads = THREADS_PER_BLOCK;
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// SiLU forward `y[i] = x[i]·σ(x[i])` on resident buffers (`ops::act::silu_forward`).
    /// `expf` may differ from host libm by ~1 ULP ⇒ gated device==CPU within rel 1e-4.
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn silu_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < n || d_y.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_y.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_silu_fwd);
        launch.arg(d_x).arg(d_y).arg(&ni);
        // SAFETY: signature `(const float* x, float* y, int n)`; one thread per element, `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch silu_forward_dev", &e))?;
        }
        Ok(())
    }

    /// SiLU backward `gx[i] = gy[i]·(s + x·s·(1−s))`, `s=σ(x[i])` (`ops::act::silu_vjp`).
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn silu_backward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_gy: &CudaSlice<f32>,
        d_gx: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < n || d_gy.len() < n || d_gx.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_gx.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_silu_bwd);
        launch.arg(d_x).arg(d_gy).arg(d_gx).arg(&ni);
        // SAFETY: signature `(const float* x, const float* gy, float* gx, int n)`; `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch silu_backward_dev", &e))?;
        }
        Ok(())
    }

    /// Elementwise multiply forward `y[i] = a[i]·b[i]` on resident buffers
    /// (`ops::elementwise::mul_forward`; bit-exact).
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn ew_mul_forward_dev(
        &self,
        d_a: &CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_a.len() < n || d_b.len() < n || d_y.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_y.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_ew_mul_fwd);
        launch.arg(d_a).arg(d_b).arg(d_y).arg(&ni);
        // SAFETY: signature `(const float* a, const float* b, float* y, int n)`; `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch ew_mul_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Elementwise multiply backward, one factor: `g_out[i] = gy[i]·other[i]` (`ops::elementwise::mul_vjp`;
    /// bit-exact). Call with `other = b` for `gA`, then `other = a` for `gB`.
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn ew_mul_backward_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        d_other: &CudaSlice<f32>,
        d_gout: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_gy.len() < n || d_other.len() < n || d_gout.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_gout.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_ew_mul_bwd);
        launch.arg(d_gy).arg(d_other).arg(d_gout).arg(&ni);
        // SAFETY: signature `(const float* gy, const float* other, float* g_out, int n)`; `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch ew_mul_backward_dev", &e))?;
        }
        Ok(())
    }

    /// Elementwise add forward `y[i] = a[i]+b[i]` on resident buffers (`ops::elementwise::add_forward`;
    /// bit-exact). Its backward is just [`accumulate_dev`] of `gy` into each input's grad.
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn ew_add_forward_dev(
        &self,
        d_a: &CudaSlice<f32>,
        d_b: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_a.len() < n || d_b.len() < n || d_y.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_y.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_ew_add_fwd);
        launch.arg(d_a).arg(d_b).arg(d_y).arg(&ni);
        // SAFETY: signature `(const float* a, const float* b, float* y, int n)`; `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch ew_add_forward_dev", &e))?;
        }
        Ok(())
    }

    /// In-place gradient accumulate `dst[i] += src[i]` on resident buffers — how the device tape sums
    /// a value's grad across its consumers (residuals, tied embedding), matching `grads[id] += v`.
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn accumulate_dev(
        &self,
        d_dst: &mut CudaSlice<f32>,
        d_src: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_dst.len() < n || d_src.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_dst.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_accumulate);
        launch.arg(d_dst).arg(d_src).arg(&ni);
        // SAFETY: signature `(float* dst, const float* src, int n)`; in-place read+write, `i < n`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch accumulate_dev", &e))?;
        }
        Ok(())
    }

    /// Training RMSNorm forward on resident buffers (`ops::norm::forward`, sequential order —
    /// distinct from the inference tree-order [`rmsnorm`](Self::rmsnorm)). `x`:[rows,cols],
    /// `w`:[cols], `y`:[rows,cols]. One thread per row. Bit-exact vs the CPU op (only +,*,/,sqrt).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn rmsnorm_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        if d_x.len() < rows * cols || d_w.len() < cols || d_y.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_y.len(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_rmsnorm_train_fwd);
        launch
            .arg(d_x)
            .arg(d_w)
            .arg(d_y)
            .arg(&ri)
            .arg(&ci)
            .arg(&eps);
        // SAFETY: signature `(const float* x, const float* w, float* y, int rows, int cols, float
        // eps)`; one thread per row, guarded by `r < rows`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch rmsnorm_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Training RMSNorm backward on resident buffers (`ops::norm::vjp`): writes `gx`:[rows,cols] and
    /// `gw`:[cols]. Allocates the per-row `inv[rows]` once (`rmsnorm_train_inv`) and shares it across
    /// `grad_x` (thread/row) and `grad_w` (thread/col). Bit-exact vs the CPU vjp.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // resident buffers + shape: a kernel-launch wrapper
    pub(crate) fn rmsnorm_backward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_w: &CudaSlice<f32>,
        d_gy: &CudaSlice<f32>,
        d_gx: &mut CudaSlice<f32>,
        d_gw: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        if d_x.len() < rows * cols
            || d_gy.len() < rows * cols
            || d_gx.len() < rows * cols
            || d_w.len() < cols
            || d_gw.len() < cols
        {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_gx.len(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut d_inv = self
            .stream
            .alloc_zeros::<f32>(rows)
            .map_err(|e| driver_err("rmsnorm_backward_dev alloc inv", &e))?;
        // inv[r] = 1/sqrt(mean_sq_r + eps) — one thread per row.
        let mut l_inv = self.stream.launch_builder(&self.func_rmsnorm_train_inv);
        l_inv.arg(d_x).arg(&mut d_inv).arg(&ri).arg(&ci).arg(&eps);
        // SAFETY: `(const float* x, float* inv, int rows, int cols, float eps)`; one thread/row.
        #[allow(unsafe_code)]
        unsafe {
            l_inv
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch rmsnorm_train_inv", &e))?;
        }
        // gx[r,·] — one thread per row.
        let mut l_gx = self.stream.launch_builder(&self.func_rmsnorm_train_grad_x);
        l_gx.arg(d_x)
            .arg(d_w)
            .arg(d_gy)
            .arg(&d_inv)
            .arg(d_gx)
            .arg(&ri)
            .arg(&ci);
        // SAFETY: `(const float* x, w, gy, inv, float* gx, int rows, int cols)`; one thread/row.
        #[allow(unsafe_code)]
        unsafe {
            l_gx.launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch rmsnorm_train_grad_x", &e))?;
        }
        // gw[i] = Σ_r … — one thread per column.
        let mut l_gw = self.stream.launch_builder(&self.func_rmsnorm_train_grad_w);
        l_gw.arg(d_x)
            .arg(d_gy)
            .arg(&d_inv)
            .arg(d_gw)
            .arg(&ri)
            .arg(&ci);
        // SAFETY: `(const float* x, gy, inv, float* gw, int rows, int cols)`; one thread/col.
        #[allow(unsafe_code)]
        unsafe {
            l_gw.launch(Self::elementwise_cfg(cols))
                .map_err(|e| driver_err("launch rmsnorm_train_grad_w", &e))?;
        }
        Ok(())
    }

    /// Upload an i32 host slice to a resident device buffer (positions / token ids for rope/gather).
    ///
    /// # Errors
    /// Device failure via the cudarc mapping.
    #[allow(dead_code)]
    pub(crate) fn dev_upload_i32(&self, host: &[i32]) -> Result<CudaSlice<i32>, BackendError> {
        self.stream
            .clone_htod(host)
            .map_err(|e| driver_err("dev_upload_i32 htod", &e))
    }

    /// Multiply a resident buffer by a constant scalar: `y[i] = x[i]·c` (attention's `1/√head_dim`;
    /// bit-exact). Its own vjp shape (backward = `scale_const_dev(gy, c)`).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn scale_const_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        c: f32,
        n: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < n || d_y.len() < n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: d_y.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let ni = n as i32;
        let mut launch = self.stream.launch_builder(&self.func_scale_const);
        launch.arg(d_x).arg(d_y).arg(&c).arg(&ni);
        // SAFETY: `(const float* x, float* y, float c, int n)`; one thread per element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(n))
                .map_err(|e| driver_err("launch scale_const_dev", &e))?;
        }
        Ok(())
    }

    /// Row-softmax forward on resident buffers (`ops::softmax::forward`; expf ⇒ within 1e-4).
    /// `x`,`y`:[rows,cols].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn softmax_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < rows * cols || d_y.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_y.len(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_softmax_fwd);
        launch.arg(d_x).arg(d_y).arg(&ri).arg(&ci);
        // SAFETY: `(const float* x, float* y, int rows, int cols)`; one thread per row.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch softmax_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Softmax backward from the saved probs `p` (`ops::softmax::vjp`). `p`,`gy`,`gx`:[rows,cols].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn softmax_backward_dev(
        &self,
        d_p: &CudaSlice<f32>,
        d_gy: &CudaSlice<f32>,
        d_gx: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if d_p.len() < rows * cols || d_gy.len() < rows * cols || d_gx.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_gx.len(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_softmax_bwd);
        launch.arg(d_p).arg(d_gy).arg(d_gx).arg(&ri).arg(&ci);
        // SAFETY: `(const float* p, const float* gy, float* gx, int rows, int cols)`; thread/row.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch softmax_backward_dev", &e))?;
        }
        Ok(())
    }

    /// Additive causal mask forward on resident scores [rows=queries, cols=keys]
    /// (`ops::softmax::causal_mask_forward`; bit-exact).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn causal_mask_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < rows * cols || d_y.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_y.len(),
            });
        }
        if rows * cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_causal_mask_fwd);
        launch.arg(d_x).arg(d_y).arg(&ri).arg(&ci);
        // SAFETY: `(const float* x, float* y, int rows, int cols)`; one thread per element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows * cols))
                .map_err(|e| driver_err("launch causal_mask_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Causal-mask backward (`ops::softmax::causal_mask_vjp`; bit-exact). `gy`,`gx`:[rows,cols].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn causal_mask_backward_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        d_gx: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if d_gy.len() < rows * cols || d_gx.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_gx.len(),
            });
        }
        if rows * cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_causal_mask_bwd);
        launch.arg(d_gy).arg(d_gx).arg(&ri).arg(&ci);
        // SAFETY: `(const float* gy, float* gx, int rows, int cols)`; one thread per element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows * cols))
                .map_err(|e| driver_err("launch causal_mask_backward_dev", &e))?;
        }
        Ok(())
    }

    /// RoPE apply on a resident [n_token, n_head, head_dim] buffer (`ops::rope`; sin/cos ⇒ within
    /// 1e-4). `sign = +1.0` forward, `-1.0` for the vjp (inverse rotation). `positions` is a resident
    /// i32 buffer of length `n_token`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer / odd head_dim; device failure via cudarc.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)] // resident buffers + rope geometry: a kernel-launch wrapper
    pub(crate) fn rope_apply_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        d_positions: &CudaSlice<i32>,
        n_head: usize,
        head_dim: usize,
        theta: f32,
        n_token: usize,
        sign: f32,
    ) -> Result<(), BackendError> {
        let total = n_token * n_head * head_dim;
        if !head_dim.is_multiple_of(2)
            || d_x.len() < total
            || d_y.len() < total
            || d_positions.len() < n_token
        {
            return Err(BackendError::ShapeMismatch {
                expected: total,
                got: d_y.len(),
            });
        }
        let half = head_dim / 2;
        let pairs = n_token * n_head * half;
        if pairs == 0 {
            return Ok(());
        }
        let (nh, hd, nt) = (n_head as i32, head_dim as i32, n_token as i32);
        let mut launch = self.stream.launch_builder(&self.func_rope_apply);
        launch
            .arg(d_x)
            .arg(d_y)
            .arg(d_positions)
            .arg(&nh)
            .arg(&hd)
            .arg(&theta)
            .arg(&nt)
            .arg(&sign);
        // SAFETY: `(const float* x, float* y, const int* positions, int n_head, int head_dim,
        // float theta, int n_token, float sign)`; one thread per rotation pair.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(pairs))
                .map_err(|e| driver_err("launch rope_apply_dev", &e))?;
        }
        Ok(())
    }

    /// Extract columns [start, start+len) from a resident [rows,cols] buffer into [rows,len]
    /// (`ops::shape::slice_cols_forward`; bit-exact).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized/out-of-range buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn slice_cols_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        start: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        if start + len > cols || d_x.len() < rows * cols || d_y.len() < rows * len {
            return Err(BackendError::ShapeMismatch {
                expected: rows * len,
                got: d_y.len(),
            });
        }
        if rows * len == 0 {
            return Ok(());
        }
        let (ri, ci, si, li) = (rows as i32, cols as i32, start as i32, len as i32);
        let mut launch = self.stream.launch_builder(&self.func_slice_cols_fwd);
        launch.arg(d_x).arg(d_y).arg(&ri).arg(&ci).arg(&si).arg(&li);
        // SAFETY: `(const float* x, float* y, int rows, int cols, int start, int len)`; thread per
        // output element (rows*len).
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows * len))
                .map_err(|e| driver_err("launch slice_cols_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Insert a resident [rows,len] block into columns [start,start+len) of a [rows,total] buffer
    /// (`dst[r,start+c]=src[r,c]`; bit-exact). Builds `concat_cols` (N inserts) and `slice_cols`'s vjp
    /// (insert into a zeroed buffer).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized/out-of-range buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn copy_into_cols_dev(
        &self,
        d_src: &CudaSlice<f32>,
        d_dst: &mut CudaSlice<f32>,
        rows: usize,
        total: usize,
        start: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        if start + len > total || d_src.len() < rows * len || d_dst.len() < rows * total {
            return Err(BackendError::ShapeMismatch {
                expected: rows * total,
                got: d_dst.len(),
            });
        }
        if rows * len == 0 {
            return Ok(());
        }
        let (ri, ti, si, li) = (rows as i32, total as i32, start as i32, len as i32);
        let mut launch = self.stream.launch_builder(&self.func_copy_into_cols);
        launch
            .arg(d_src)
            .arg(d_dst)
            .arg(&ri)
            .arg(&ti)
            .arg(&si)
            .arg(&li);
        // SAFETY: `(const float* src, float* dst, int rows, int total, int start, int len)`; thread
        // per source element (rows*len).
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows * len))
                .map_err(|e| driver_err("launch copy_into_cols_dev", &e))?;
        }
        Ok(())
    }

    /// Transpose a resident [rows,cols] buffer → [cols,rows] (`ops::dense::transpose_forward`; also
    /// its own vjp; bit-exact).
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn transpose_forward_dev(
        &self,
        d_x: &CudaSlice<f32>,
        d_y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), BackendError> {
        if d_x.len() < rows * cols || d_y.len() < rows * cols {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_y.len(),
            });
        }
        if rows * cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_transpose_fwd);
        launch.arg(d_x).arg(d_y).arg(&ri).arg(&ci);
        // SAFETY: `(const float* x, float* y, int rows, int cols)`; one thread per element.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows * cols))
                .map_err(|e| driver_err("launch transpose_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Embedding gather forward `y[t,:] = w[tokens[t],:]` on resident buffers
    /// (`ops::embed::gather_forward`; bit-exact). `tokens` is a resident i32 buffer of length `seq`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn embed_gather_forward_dev(
        &self,
        d_w: &CudaSlice<f32>,
        d_tokens: &CudaSlice<i32>,
        d_y: &mut CudaSlice<f32>,
        seq: usize,
        dim: usize,
    ) -> Result<(), BackendError> {
        if d_tokens.len() < seq || d_y.len() < seq * dim {
            return Err(BackendError::ShapeMismatch {
                expected: seq * dim,
                got: d_y.len(),
            });
        }
        if seq * dim == 0 {
            return Ok(());
        }
        let (si, di) = (seq as i32, dim as i32);
        let mut launch = self.stream.launch_builder(&self.func_embed_gather_fwd);
        launch.arg(d_w).arg(d_tokens).arg(d_y).arg(&si).arg(&di);
        // SAFETY: `(const float* w, const int* tokens, float* y, int seq, int dim)`; thread per
        // output element (seq*dim). w indexed by tokens[t]∈[0,vocab); caller supplies valid ids.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(seq * dim))
                .map_err(|e| driver_err("launch embed_gather_forward_dev", &e))?;
        }
        Ok(())
    }

    /// Embedding gather backward: `gw[v,:] = Σ_{t:tokens[t]==v} gy[t,:]`, summed ascending-t (bit-exact
    /// vs `ops::embed::gather_vjp`; one thread per gw element, no atomics). `gw`:[vocab,dim].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn embed_gather_backward_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        d_tokens: &CudaSlice<i32>,
        d_gw: &mut CudaSlice<f32>,
        seq: usize,
        dim: usize,
        vocab: usize,
    ) -> Result<(), BackendError> {
        if d_gy.len() < seq * dim || d_tokens.len() < seq || d_gw.len() < vocab * dim {
            return Err(BackendError::ShapeMismatch {
                expected: vocab * dim,
                got: d_gw.len(),
            });
        }
        if vocab * dim == 0 {
            return Ok(());
        }
        let (si, di, vi) = (seq as i32, dim as i32, vocab as i32);
        let mut launch = self.stream.launch_builder(&self.func_embed_gather_bwd);
        launch
            .arg(d_gy)
            .arg(d_tokens)
            .arg(d_gw)
            .arg(&si)
            .arg(&di)
            .arg(&vi);
        // SAFETY: `(const float* gy, const int* tokens, float* gw, int seq, int dim, int vocab)`;
        // one thread per gw element (vocab*dim).
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(vocab * dim))
                .map_err(|e| driver_err("launch embed_gather_backward_dev", &e))?;
        }
        Ok(())
    }

    /// Validate, stably group, and upload embedding segment metadata once for
    /// reuse by the tape's backward pass.
    pub(crate) fn prepare_embed_segments(
        &self,
        tokens: &[i32],
        seq: usize,
        vocab: usize,
    ) -> Result<EmbedSegments, BackendError> {
        if tokens.len() != seq {
            return Err(BackendError::ShapeMismatch {
                expected: seq,
                got: tokens.len(),
            });
        }
        let _ = i32::try_from(seq)
            .map_err(|_| BackendError::InvalidInput("embedding seq exceeds i32::MAX".into()))?;
        let vocab_i = i32::try_from(vocab)
            .map_err(|_| BackendError::InvalidInput("embedding vocab exceeds i32::MAX".into()))?;
        for (position, &token) in tokens.iter().enumerate() {
            if token < 0 || token >= vocab_i {
                return Err(BackendError::InvalidInput(format!(
                    "token {token} at position {position} is outside 0..{vocab}"
                )));
            }
        }
        // Stable sort: equal token ids retain ascending original position.
        let mut order: Vec<usize> = (0..seq).collect();
        order.sort_by_key(|&position| tokens[position]);
        let mut rows = Vec::new();
        let mut offsets = vec![0i32];
        let mut positions = Vec::with_capacity(seq);
        for position in order {
            let token = tokens[position];
            if rows.last().copied() != Some(token) {
                if !rows.is_empty() {
                    offsets.push(positions.len() as i32);
                }
                rows.push(token);
            }
            positions.push(position as i32);
        }
        offsets.push(seq as i32);

        let unique_rows = rows.len();
        rows.extend(offsets);
        rows.extend(positions);
        Ok(EmbedSegments {
            unique_rows,
            metadata: self.dev_upload_i32(&rows)?,
            seq,
        })
    }

    /// Deterministic segmented embedding backward using metadata already
    /// uploaded by [`Self::prepare_embed_segments`].
    pub(crate) fn embed_gather_backward_segmented_prepared_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        segments: &EmbedSegments,
        d_gw: &mut CudaSlice<f32>,
        seq: usize,
        dim: usize,
        vocab: usize,
    ) -> Result<(), BackendError> {
        if segments.seq != seq {
            return Err(BackendError::ShapeMismatch {
                expected: segments.seq,
                got: seq,
            });
        }
        let gy_len = seq.checked_mul(dim).ok_or_else(|| {
            BackendError::InvalidInput("embedding gradient shape overflows usize".into())
        })?;
        let gw_len = vocab.checked_mul(dim).ok_or_else(|| {
            BackendError::InvalidInput("embedding weight shape overflows usize".into())
        })?;
        if d_gy.len() < gy_len || d_gw.len() < gw_len {
            return Err(BackendError::ShapeMismatch {
                expected: gw_len,
                got: d_gw.len(),
            });
        }
        let dim_i = i32::try_from(dim)
            .map_err(|_| BackendError::InvalidInput("embedding dim exceeds i32::MAX".into()))?;
        let unique_i = i32::try_from(segments.unique_rows).map_err(|_| {
            BackendError::InvalidInput("unique embedding rows exceed i32::MAX".into())
        })?;
        self.stream
            .memset_zeros(d_gw)
            .map_err(|e| driver_err("zero segmented embedding gradient", &e))?;
        if segments.unique_rows == 0 || dim == 0 || vocab == 0 {
            return Ok(());
        }
        let work = segments.unique_rows.checked_mul(dim).ok_or_else(|| {
            BackendError::InvalidInput("segmented embedding work size overflows usize".into())
        })?;
        let mut launch = self
            .stream
            .launch_builder(&self.func_embed_gather_bwd_segmented);
        launch
            .arg(d_gy)
            .arg(&segments.metadata)
            .arg(d_gw)
            .arg(&unique_i)
            .arg(&dim_i);
        // SAFETY: kernel args match the declaration; metadata was validated
        // and constructed by `prepare_embed_segments`.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(work))
                .map_err(|e| driver_err("launch embed_gather_backward_segmented_dev", &e))?;
        }
        Ok(())
    }

    /// Total-cost wrapper used by the correctness/performance gate: host
    /// grouping, metadata upload, zeroing, and segmented kernel.
    #[allow(dead_code)]
    pub(crate) fn embed_gather_backward_segmented_dev(
        &self,
        d_gy: &CudaSlice<f32>,
        tokens: &[i32],
        d_gw: &mut CudaSlice<f32>,
        seq: usize,
        dim: usize,
        vocab: usize,
    ) -> Result<(), BackendError> {
        let segments = self.prepare_embed_segments(tokens, seq, vocab)?;
        self.embed_gather_backward_segmented_prepared_dev(d_gy, &segments, d_gw, seq, dim, vocab)
    }

    /// Softmax cross-entropy backward on resident logits (`ops::loss::softmax_xent_vjp`; expf ⇒ within
    /// 1e-4). `gscale = grad_out/rows`. `logits`,`target`,`g_logits`:[rows,cols].
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] on undersized buffer; device failure via cudarc.
    #[allow(dead_code)]
    pub(crate) fn softmax_xent_backward_dev(
        &self,
        d_logits: &CudaSlice<f32>,
        d_target: &CudaSlice<f32>,
        d_glogits: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        gscale: f32,
    ) -> Result<(), BackendError> {
        if d_logits.len() < rows * cols
            || d_target.len() < rows * cols
            || d_glogits.len() < rows * cols
        {
            return Err(BackendError::ShapeMismatch {
                expected: rows * cols,
                got: d_glogits.len(),
            });
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let (ri, ci) = (rows as i32, cols as i32);
        let mut launch = self.stream.launch_builder(&self.func_softmax_xent_bwd);
        launch
            .arg(d_logits)
            .arg(d_target)
            .arg(d_glogits)
            .arg(&ri)
            .arg(&ci)
            .arg(&gscale);
        // SAFETY: `(const float* logits, const float* target, float* g_logits, int rows, int cols,
        // float gscale)`; one thread per row.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(Self::elementwise_cfg(rows))
                .map_err(|e| driver_err("launch softmax_xent_backward_dev", &e))?;
        }
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
    pub(super) fn attn_split_dense(
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
            // Host mpgemm has no TQ1 kernel path (v1: the resident decoder is
            // TQ1's only consumer).
            Stride::Tq1_0 { .. } => {
                return Err(BackendError::UnsupportedFormat(TernaryFormat::Tq1_0));
            }
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
        // ADR 0026 Track P: build I2sInt8 IMMA shadows for the prefill
        // tensor-core path (~0.25 B/weight beside the TQ2 rows). Kill switch:
        // TRITIUM_IMMA_PREFILL=0 skips the shadows AND the dispatch; the tune
        // policy `off` ALSO skips the shadows (review N3 — half a GB of
        // shadow that can never dispatch is waste, not caution).
        let imma_tune_policy = match std::env::var("TRITIUM_IMMA_TUNE") {
            Err(std::env::VarError::NotPresent) => ImmaTunePolicy::Tune,
            Ok(v) => match v.as_str() {
                "tune" | "" => ImmaTunePolicy::Tune,
                "load" => ImmaTunePolicy::Load,
                "off" => ImmaTunePolicy::Off,
                other => {
                    return Err(BackendError::InvalidInput(format!(
                        "TRITIUM_IMMA_TUNE={other:?} — use tune, load or off"
                    )));
                }
            },
            Err(e) => {
                return Err(BackendError::InvalidInput(format!(
                    "TRITIUM_IMMA_TUNE: {e}"
                )));
            }
        };
        let enable_imma = match std::env::var("TRITIUM_IMMA_PREFILL") {
            Err(std::env::VarError::NotPresent) => true,
            Ok(v) if v == "0" => false,
            Ok(v) if v == "1" || v.is_empty() => true,
            Ok(v) => {
                return Err(BackendError::InvalidInput(format!(
                    "TRITIUM_IMMA_PREFILL={v:?} — use 1 (default) or 0"
                )));
            }
            Err(e) => {
                return Err(BackendError::InvalidInput(format!(
                    "TRITIUM_IMMA_PREFILL: {e}"
                )));
            }
        } && imma_tune_policy != ImmaTunePolicy::Off;
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
                // The seven per-projection linears are what prefill launches
                // — they carry the IMMA shadow (ADR 0026 Track P) unless the
                // kill switch is set.
                q: ResidentLinear::build(s, &ls.q, false, enable_imma)?,
                k: ResidentLinear::build(s, &ls.k, false, enable_imma)?,
                v: ResidentLinear::build(s, &ls.v, false, enable_imma)?,
                o: ResidentLinear::build(s, &ls.o, false, enable_imma)?,
                gate: ResidentLinear::build(s, &ls.gate, false, enable_imma)?,
                up: ResidentLinear::build(s, &ls.up, false, enable_imma)?,
                down: ResidentLinear::build(s, &ls.down, false, enable_imma)?,
                // Only the fused pair launches through the sparse decode-graph
                // slot — bitmaps computed here alone (review N1).
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
        let sel = |a, b, c| kv_dtype.pick(a, b, c);
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

        // ADR 0026 Track P: dispatch threshold (default 32 = the tuned
        // dp4a/IMMA crossover). Loud-reject selector.
        let imma_min_m = match std::env::var("TRITIUM_IMMA_MIN_M") {
            Err(std::env::VarError::NotPresent) => 32usize,
            Ok(v) => match v.trim().parse::<usize>() {
                Ok(t) if t >= 2 => t,
                _ => {
                    return Err(BackendError::InvalidInput(format!(
                        "TRITIUM_IMMA_MIN_M={v:?} — use an integer >= 2"
                    )));
                }
            },
            Err(e) => {
                return Err(BackendError::InvalidInput(format!(
                    "TRITIUM_IMMA_MIN_M: {e}"
                )));
            }
        };

        // ADR 0026 Track P step 3: resolve the IMMA tile functions at BUILD
        // time for every shadowed (N, K) shape × the prefill M buckets, so
        // the serving path never touches nvrtc or the tune sweep. Policy:
        //   TRITIUM_IMMA_TUNE=tune (default) — sweep on a cold disk cache
        //     (one-time, printed notice; winners persist on disk);
        //   load — disk-cache-or-AOT, never sweeps;
        //   off  — no IMMA functions (dispatch falls back to dp4a).
        let imma_funcs = {
            let policy = if enable_imma {
                imma_tune_policy
            } else {
                ImmaTunePolicy::Off
            };
            let mut map = HashMap::new();
            if policy != ImmaTunePolicy::Off {
                let mut shapes = std::collections::BTreeSet::new();
                for l in &layers {
                    for lin in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
                        if lin.imma.is_some() {
                            shapes.insert((lin.n, lin.k));
                        }
                    }
                }
                // Prefill M buckets: 32.. (the dp4a/IMMA crossover) up to
                // 2048+-token one-shot prompts; log2 floor matches
                // ShapeBucket::from_shape, so the disk keys line up.
                for &(n, k) in &shapes {
                    for m_log2 in 5u32..=11 {
                        let shape = GemmShape {
                            m: 1usize << m_log2,
                            n,
                            k,
                        };
                        let tile = match policy {
                            ImmaTunePolicy::Tune => self.resolve_imma_tile(shape),
                            ImmaTunePolicy::Load => crate::autotune::load_cached(
                                &cache_dir(),
                                &self.imma_cache_key(shape),
                            )
                            .unwrap_or(TileConfig::AOT_EQUIVALENT),
                            ImmaTunePolicy::Off => unreachable!(),
                        };
                        // Review N5: a JIT failure for a cached non-AOT tile
                        // degrades to the embedded AOT cubin (correct, slower)
                        // instead of aborting the whole model build. The TILE
                        // must fall back WITH the function — the launch derives
                        // its grid/block geometry from the stored TileConfig,
                        // and the AOT kernel under a big tile's geometry would
                        // compute garbage.
                        let (tile, func) = match self.imma_function_for_tile(tile) {
                            Ok(f) => (tile, f),
                            Err(e) => {
                                eprintln!(
                                    "tritium-cuda: IMMA tile {tile:?} JIT failed ({e}); \
                                     falling back to the AOT tile for ({n},{k},m2^{m_log2})"
                                );
                                (
                                    TileConfig::AOT_EQUIVALENT,
                                    self.imma_function_for_tile(TileConfig::AOT_EQUIVALENT)?,
                                )
                            }
                        };
                        map.insert((n, k, m_log2), (tile, func));
                    }
                }
                if !map.is_empty() {
                    eprintln!(
                        "tritium-cuda: IMMA prefill ready — {} tile functions over {} \
                         weight shapes (TRITIUM_IMMA_TUNE policy applied)",
                        map.len(),
                        shapes.len(),
                    );
                }
            }
            map
        };

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
            f_tq1_tiled_scaled: f(&self._module, KERNEL_NAME_TQ1_TILED_I8_SCALED)?,
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
            f_kv_append_mdecode_paged: f(dm, KERNEL_NAME_KV_APPEND_MDECODE_PAGED)?,
            f_attn_split_partial: f(dm, KERNEL_NAME_ATTN_SPLIT_PARTIAL)?,
            f_attn_split_partial_paged: f(dm, KERNEL_NAME_ATTN_SPLIT_PARTIAL_PAGED)?,
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
            imma_funcs,
            imma_min_m,
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
    pub(super) fn mpgemm_kernel(
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
    pub(super) fn mpgemm_kernel_with_bitmap(
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

    /// Upload one validated rank-2 SALT V2 tensor without materializing dense weights.
    ///
    /// The selected codec is applied independently to every present tile plane.
    /// Device memory retains the exact encoded payload, group128 f16 scales,
    /// complete two-bit allocation-map bytes, and one coarse u32 plane-rank
    /// prefix per 256-tile block after the first. Terminal map bits ride in the
    /// mandatory launch scalar. Missing planes consume no payload or scale bytes.
    ///
    /// # Errors
    /// Rejects non-matrices, dimensions or offsets that exceed the kernel's u32
    /// ABI, codec-incompatible S34 groups, accounting overflow, and driver
    /// allocation/upload failures.
    pub fn upload_salt_v2(
        &self,
        tensor: &SaltV2Tensor,
        codec: SaltV2Codec,
    ) -> Result<SaltV2ResidentTensor, BackendError> {
        if !matches!(tensor.transform(), SaltV2Transform::None) {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 CUDA does not implement tensor transform {:?}; only None is accepted",
                tensor.transform()
            )));
        }
        if tensor.dims().len() != 2 {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 CUDA requires rank 2, got rank {}",
                tensor.dims().len()
            )));
        }
        let rows = usize::try_from(tensor.dims()[0]).map_err(|_| {
            BackendError::InvalidInput(format!(
                "SALT V2 row dimension {} exceeds host usize",
                tensor.dims()[0]
            ))
        })?;
        let columns = usize::try_from(tensor.dims()[1]).map_err(|_| {
            BackendError::InvalidInput(format!(
                "SALT V2 column dimension {} exceeds host usize",
                tensor.dims()[1]
            ))
        })?;
        let coefficients = rows.checked_mul(columns).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 matrix dimension product overflows usize".into())
        })?;
        if coefficients != tensor.logical_coefficients() {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 shape has {coefficients} coefficients but tensor has {}",
                tensor.logical_coefficients()
            )));
        }
        let codec_tag = match codec {
            SaltV2Codec::D2 => 0,
            SaltV2Codec::B3 => 1,
            SaltV2Codec::S34 => 2,
            _ => {
                return Err(BackendError::InvalidInput(format!(
                    "unsupported SALT V2 CUDA codec {codec:?}"
                )));
            }
        };
        let to_u32 = |value: usize, field: &str| {
            u32::try_from(value).map_err(|_| {
                BackendError::InvalidInput(format!("SALT V2 {field} exceeds the u32 kernel ABI"))
            })
        };
        to_u32(rows, "row count")?;
        to_u32(columns, "column count")?;
        to_u32(tensor.tiles().len(), "tile count")?;

        let plane_count = tensor.tiles().iter().try_fold(0usize, |total, tile| {
            total.checked_add(tile.planes().len()).ok_or_else(|| {
                BackendError::InvalidInput("SALT V2 present plane count overflows usize".into())
            })
        })?;
        to_u32(plane_count, "present plane count")?;
        let planned = SaltV2IndexedRuntimeLedger::for_tensor(tensor, codec).map_err(|error| {
            BackendError::InvalidInput(format!("SALT V2 indexed-runtime planning failed: {error}"))
        })?;

        let mut payload = Vec::<u8>::new();
        let mut scale_bits = Vec::<u16>::new();
        let map_bytes = usize::try_from(planned.allocation_map_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 map bytes exceed host usize".into())
        })?;
        let rank_prefix_bytes = usize::try_from(planned.rank_prefix_bytes()).map_err(|_| {
            BackendError::InvalidInput("SALT V2 rank-prefix bytes exceed host usize".into())
        })?;
        let index_bytes = map_bytes.checked_add(rank_prefix_bytes).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 index bytes overflow usize".into())
        })?;
        let mut allocation_map = Vec::<u8>::new();
        allocation_map
            .try_reserve_exact(map_bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: map_bytes,
            })?;
        allocation_map.resize(map_bytes, 0);
        let mut rank_prefixes = Vec::<u32>::new();
        rank_prefixes
            .try_reserve_exact(rank_prefix_bytes / core::mem::size_of::<u32>())
            .map_err(|_| BackendError::OutOfMemory {
                requested: rank_prefix_bytes,
            })?;
        let mut terminal_map_value = 0_u32;
        let mut planes_before_tile = 0usize;

        for (tile_index, tile) in tensor.tiles().iter().enumerate() {
            if tile_index != 0
                && tile_index.is_multiple_of(
                    tritium_format::salt_v2_package::SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES,
                )
            {
                rank_prefixes.push(to_u32(planes_before_tile, "rank prefix")?);
            }
            let map_code = u8::try_from(tile.planes().len() - 1)
                .map_err(|_| BackendError::InvalidInput("SALT V2 plane count underflow".into()))?;
            let map_bit = tile_index.checked_mul(2).ok_or_else(|| {
                BackendError::InvalidInput("SALT V2 map bit offset overflows usize".into())
            })?;
            if map_bit < map_bytes * 8 {
                allocation_map[map_bit / 8] |= map_code << (map_bit % 8);
            } else {
                terminal_map_value |= u32::from(map_code) << (map_bit - map_bytes * 8);
            }
            for plane in tile.planes() {
                let packed = pack_salt_v2_cuda_plane(codec, plane.trits())?;
                payload.extend_from_slice(&packed);
                scale_bits.extend(plane.scales().iter().map(|scale| scale.to_bits()));
            }
            planes_before_tile = planes_before_tile
                .checked_add(tile.planes().len())
                .ok_or_else(|| {
                    BackendError::InvalidInput("SALT V2 plane rank overflows usize".into())
                })?;
        }
        debug_assert_eq!(planes_before_tile, plane_count);
        to_u32(payload.len(), "payload byte count")?;
        to_u32(scale_bits.len(), "scale count")?;
        let mut index_metadata = allocation_map;
        index_metadata
            .try_reserve_exact(rank_prefix_bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: index_bytes,
            })?;
        for prefix in &rank_prefixes {
            index_metadata.extend_from_slice(&prefix.to_le_bytes());
        }
        debug_assert_eq!(index_metadata.len(), index_bytes);

        let scale_bytes = scale_bits
            .len()
            .checked_mul(core::mem::size_of::<u16>())
            .ok_or_else(|| {
                BackendError::InvalidInput("SALT V2 scale bytes overflow usize".into())
            })?;
        let actual_matches_plan = u64::try_from(payload.len()).ok()
            == Some(planned.payload_bytes())
            && u64::try_from(scale_bytes).ok() == Some(planned.scale_bytes())
            && u64::try_from(map_bytes).ok() == Some(planned.allocation_map_bytes())
            && u64::try_from(rank_prefix_bytes).ok() == Some(planned.rank_prefix_bytes())
            && planned.dense_shadow_bytes() == 0;
        if !actual_matches_plan {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 uploaded buffers disagree with indexed-runtime plan: actual payload={}; scale={scale_bytes}; map={map_bytes}; rank-prefix={rank_prefix_bytes}, planned={planned:?}",
                payload.len()
            )));
        }
        let receipt = SaltV2ResidentAllocationReceipt::new(codec, planned);

        let d_payload = self
            .stream
            .clone_htod(&payload)
            .map_err(|error| alloc_or_backend("upload SALT V2 payload", &error, payload.len()))?;
        let d_scales = self
            .stream
            .clone_htod(&scale_bits)
            .map_err(|error| alloc_or_backend("upload SALT V2 scales", &error, scale_bytes))?;
        let d_index_metadata = if index_metadata.is_empty() {
            None
        } else {
            Some(self.stream.clone_htod(&index_metadata).map_err(|error| {
                alloc_or_backend("upload SALT V2 compact index", &error, index_bytes)
            })?)
        };
        Ok(SaltV2ResidentTensor {
            payload: d_payload,
            scales: d_scales,
            index_metadata: d_index_metadata,
            rows,
            columns,
            tile_count: tensor.tiles().len(),
            plane_count,
            codec_tag,
            allocation_map_bytes: to_u32(map_bytes, "allocation map bytes")?,
            rank_prefix_count: to_u32(rank_prefixes.len(), "rank prefix count")?,
            terminal_map_value,
            receipt,
        })
    }

    /// Execute the deterministic scalar SALT V2 kernel.
    ///
    /// The result streams resident trits through group128/plane add-sub-skip
    /// activation accumulators, applies each f16 scale once, then accumulates the
    /// output exactly like the CPU semantic oracle. No dense weight allocation is
    /// made.
    ///
    /// # Errors
    /// Rejects a handle from another CUDA context, an activation length mismatch,
    /// non-finite input/output, launch-bound overflow, or a driver failure.
    pub fn salt_v2_forward_exact(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
    ) -> Result<SaltV2Forward, BackendError> {
        self.salt_v2_forward_impl(tensor, activation, m, SaltV2ForwardMode::Exact)
    }

    /// Execute the SALT V2 fast entry point.
    ///
    /// This correctness-first implementation deliberately aliases
    /// [`Self::salt_v2_forward_exact`]. The returned receipt is labeled
    /// [`SaltV2ForwardMode::FastAliasesExact`]; no performance claim is implied.
    ///
    /// # Errors
    /// Returns the errors documented by [`Self::salt_v2_forward_exact`].
    pub fn salt_v2_forward_fast(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
    ) -> Result<SaltV2Forward, BackendError> {
        self.salt_v2_forward_impl(tensor, activation, m, SaltV2ForwardMode::FastAliasesExact)
    }

    fn salt_v2_forward_impl(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
        m: usize,
        mode: SaltV2ForwardMode,
    ) -> Result<SaltV2Forward, BackendError> {
        if !self.same_context(&tensor.payload)
            || !self.same_context(&tensor.scales)
            || tensor
                .index_metadata
                .as_ref()
                .is_some_and(|metadata| !self.same_context(metadata))
        {
            return Err(BackendError::InvalidInput(
                "SALT V2 resident tensor belongs to a different CUDA context".into(),
            ));
        }
        let activation_elements = m.checked_mul(tensor.columns).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 activation length overflows usize".into())
        })?;
        if activation.len() != activation_elements {
            return Err(BackendError::ShapeMismatch {
                expected: activation_elements,
                got: activation.len(),
            });
        }
        if let Some((index, value)) = activation
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 activation {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }
        let output_elements = m.checked_mul(tensor.rows).ok_or_else(|| {
            BackendError::InvalidInput("SALT V2 output length overflows usize".into())
        })?;
        let receipt =
            SaltV2ForwardReceipt::new(mode, tensor.receipt, activation_elements, output_elements)?;
        if output_elements == 0 {
            return Ok(SaltV2Forward {
                output: Vec::new(),
                receipt,
            });
        }

        let m_u32 = u32::try_from(m).map_err(|_| {
            BackendError::InvalidInput("SALT V2 batch rows exceed the u32 kernel ABI".into())
        })?;
        let n_u32 = u32::try_from(tensor.rows).map_err(|_| {
            BackendError::InvalidInput("SALT V2 output rows exceed the u32 kernel ABI".into())
        })?;
        let k_u32 = u32::try_from(tensor.columns).map_err(|_| {
            BackendError::InvalidInput("SALT V2 columns exceed the u32 kernel ABI".into())
        })?;
        let tile_count = u32::try_from(tensor.tile_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 tile count exceeds the u32 kernel ABI".into())
        })?;
        let plane_count = u32::try_from(tensor.plane_count).map_err(|_| {
            BackendError::InvalidInput("SALT V2 plane count exceeds the u32 kernel ABI".into())
        })?;
        let payload_bytes = tensor.receipt.payload_bytes();
        let scale_count = tensor.receipt.scale_bytes() / core::mem::size_of::<u16>() as u64;
        let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
        let total_outputs = u32::try_from(output_elements).map_err(|_| {
            BackendError::InvalidInput("SALT V2 output elements exceed the u32 launch grid".into())
        })?;

        let d_activation = self.stream.clone_htod(activation).map_err(|error| {
            alloc_or_backend(
                "upload SALT V2 activation",
                &error,
                activation_elements * core::mem::size_of::<f32>(),
            )
        })?;
        let mut d_output = self
            .stream
            .alloc_zeros::<f32>(output_elements)
            .map_err(|error| {
                alloc_or_backend(
                    "allocate SALT V2 output",
                    &error,
                    output_elements * core::mem::size_of::<f32>(),
                )
            })?;
        let cfg = LaunchConfig {
            grid_dim: (total_outputs.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.func_salt_v2_exact);
        launch
            .arg(&d_activation)
            .arg(&tensor.payload)
            .arg(&tensor.scales)
            .arg(index_metadata)
            .arg(&mut d_output)
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&k_u32)
            .arg(&tensor.codec_tag)
            .arg(&tile_count)
            .arg(&plane_count)
            .arg(&payload_bytes)
            .arg(&scale_count)
            .arg(&tensor.allocation_map_bytes)
            .arg(&tensor.rank_prefix_count)
            .arg(&tensor.terminal_map_value);
        // SAFETY: `salt_v2_forward_exact` receives the arguments above in the
        // exact pointer/scalar order. The private resident handle owns validated
        // payload/scales/compact index metadata from this context. `d_activation` is
        // M*K, `d_output` is M*N, and all derived payload/scale ranks were checked
        // against the u32 ABI and complete host allocations before upload.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch SALT V2 exact forward", &error))?;
        }
        let mut output = vec![0.0f32; output_elements];
        self.stream
            .memcpy_dtoh(&d_output, &mut output)
            .map_err(|error| driver_err("download SALT V2 output", &error))?;
        if let Some((index, value)) = output
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(BackendError::InvalidInput(format!(
                "SALT V2 output {index} is non-finite ({:#010x})",
                value.to_bits()
            )));
        }
        Ok(SaltV2Forward { output, receipt })
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
    pub(super) fn salt_mpgemm_dense(
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
    pub(super) fn imma_with_act_quant(
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
    pub(super) fn launch_imma_tile(
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
        launch_imma_tile_on(
            &self.stream,
            func,
            tile,
            d_qact,
            d_weights,
            d_act_scale,
            d_wscale,
            d_out,
            m_i,
            n_i,
            k_i,
            num_ktiles_i,
        )
    }
}

/// Free-function IMMA tile launch (ADR 0026 Track P): callable from the
/// resident model's prefill dispatch (`CudaDecodeModel` holds no backend
/// handle, only the shared stream).
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_imma_tile_on(
    stream: &Arc<CudaStream>,
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
    {
        let grid_n = (n_i as u32).div_ceil(tile.tile_n as u32);
        let grid_m = (m_i as u32).div_ceil(tile.tile_m as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid_n, grid_m, 1),
            block_dim: (tile.block_threads(), 1, 1),
            shared_mem_bytes: 0, // the kernel's staging is static __shared__
        };
        let mut launch = stream.launch_builder(func);
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
}

impl CudaBackend {
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
    pub(super) fn resolve_imma_tile(&self, shape: GemmShape) -> TileConfig {
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
        // The candidate-invariant probe state (operands, host reference, device
        // uploads) is built ONCE per shape — review M2: recomputing the O(m·n·k)
        // host reference per candidate made a cold 2B4T tune ~17× more expensive
        // (~12 min release / hours in a dev-profile test binary).
        if crate::autotune::load_cached(&dir, &key).is_none() {
            eprintln!(
                "tritium-cuda: autotune sweep for {}x{}xk{} (m-bucket {}) — one-time, \
                 winners persist in {}",
                shape.m,
                shape.n,
                shape.k,
                ShapeBucket::from_shape(shape).m_log2,
                dir.display(),
            );
        }
        let probe = match self.build_imma_probe(shape) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("tritium-cuda: autotune probe setup failed ({e}); using the AOT tile");
                return TileConfig::AOT_EQUIVALENT;
            }
        };
        // The device-side evaluation half: JIT + launch each candidate on this shape,
        // validate it against the (pre-computed) host reference, and time it.
        let tile = tune_or_load(
            &dir,
            &key,
            |cand| self.evaluate_candidate(cand, &probe),
            |e| eprintln!("tritium-cuda: autotune cache write failed ({e}); continuing un-cached"),
        );
        self.tuned_tiles
            .lock()
            .expect("tuned_tiles poisoned")
            .insert(key, tile);
        tile
    }

    /// Evaluate one candidate `tile` against a pre-built [`ImmaProbe`]:
    /// JIT-compile it, launch once for correctness against the probe's host
    /// reference, and time it. A compile/launch failure or a bit-mismatch
    /// marks the candidate incorrect (rejected), never aborting the search.
    fn evaluate_candidate(&self, tile: TileConfig, probe: &ImmaProbe) -> CandidateResult {
        match self.try_evaluate_candidate(tile, probe) {
            Ok(r) => r,
            Err(_) => CandidateResult {
                correct: false,
                seconds: f64::INFINITY,
            },
        }
    }

    /// Build the candidate-INVARIANT half of a tune sweep once per shape:
    /// deterministic operands, the exact host reference, and the device
    /// uploads (review M2 — this used to run per candidate, ~17× the cost).
    fn build_imma_probe(&self, shape: GemmShape) -> Result<ImmaProbe, BackendError> {
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

        // Host reference: exact int32 contraction folded by the two scales in the
        // kernel's EXACT association ((float)acc * wscale * act_scale — unified with
        // the dp4a family, ADR 0026 bit-identity contract), so a correct tile matches
        // bit-for-bit (the acceptance check is to_bits-strict). Threaded over rows:
        // the largest bucket (m=2048, n=6912, k=2560) is ~36G MACs — ~8 s single-
        // threaded in release, minutes in a dev-profile test binary.
        let mut reference = vec![0.0f32; m * n];
        let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
        let rows_per = m.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (chunk_idx, out_chunk) in reference.chunks_mut(rows_per * n).enumerate() {
                let qact = &qact;
                let trits = &trits;
                let wscale = &wscale;
                let act_scale = &act_scale;
                scope.spawn(move || {
                    let mi0 = chunk_idx * rows_per;
                    for (r, out_row) in out_chunk.chunks_mut(n).enumerate() {
                        let mi = mi0 + r;
                        for (ni, out) in out_row.iter_mut().enumerate() {
                            let mut acc: i64 = 0;
                            for ki in 0..k {
                                acc += qact[mi * k + ki] as i64 * trits[ni * k + ki] as i64;
                            }
                            *out = acc as f32 * wscale[ni] * act_scale[mi];
                        }
                    }
                });
            }
        });

        // Pack the weights into the I2sInt8 tile layout the kernel expects.
        let packed = pack_i2s_int8_tiles(&trits, n, k);

        // Upload operands once; every candidate launches over the same buffers.
        Ok(ImmaProbe {
            m,
            n,
            k,
            reference,
            d_qact: self
                .stream
                .clone_htod(&qact)
                .map_err(|e| driver_err("autotune upload qact", &e))?,
            d_weights: self
                .stream
                .clone_htod(&packed)
                .map_err(|e| driver_err("autotune upload weights", &e))?,
            d_act_scale: self
                .stream
                .clone_htod(&act_scale)
                .map_err(|e| driver_err("autotune upload act_scale", &e))?,
            d_wscale: self
                .stream
                .clone_htod(&wscale)
                .map_err(|e| driver_err("autotune upload wscale", &e))?,
        })
    }

    /// Fallible inner body of [`evaluate_candidate`]; any `Err` is folded to an
    /// "incorrect" candidate by the caller.
    fn try_evaluate_candidate(
        &self,
        tile: TileConfig,
        probe: &ImmaProbe,
    ) -> Result<CandidateResult, BackendError> {
        let ImmaProbe {
            m,
            n,
            k,
            reference,
            d_qact,
            d_weights,
            d_act_scale,
            d_wscale,
        } = probe;
        let (m, n, k) = (*m, *n, *k);
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
            d_qact,
            d_weights,
            d_act_scale,
            d_wscale,
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

        // Bit-strict since the epilogue unification (ADR 0026 Track P step 1):
        // every rendered tile computes the exact i32 contraction + the one
        // canonical float fold, so a correct candidate matches the host
        // reference to the bit — imma_close's tolerance band retired here.
        let correct = got
            .iter()
            .zip(reference)
            .all(|(&g, &r)| g.to_bits() == r.to_bits());
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
                d_qact,
                d_weights,
                d_act_scale,
                d_wscale,
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
    pub(super) fn imma_function_for_tile(
        &self,
        tile: TileConfig,
    ) -> Result<CudaFunction, BackendError> {
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
    pub(super) fn imma_jit_function(
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

pub(super) fn cuda_driver_major(version: u32) -> Option<u32> {
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
pub(super) enum AddKernel {
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
            // A2: entropy-dense rows consumed natively by the tq1 decode
            // kernels (the host mpgemm path rejects this format — the
            // resident decoder is TQ1's only consumer in v1).
            TernaryFormat::Tq1_0 => {
                let row_bytes = num_blocks(k) * tritium_format::TQ1_0_BLOCK_BYTES;
                (n * row_bytes, Stride::Tq1_0 { row_bytes })
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
    pub(super) fn default_mpgemm_with_act_quant(
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
pub(super) fn quantize_act_int8_host(
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
/// `TRITIUM_IMMA_TUNE` policy (ADR 0026 Track P step 3).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ImmaTunePolicy {
    Tune,
    Load,
    Off,
}

/// Candidate-invariant tune-probe state: deterministic operands uploaded
/// once, plus the exact host reference every candidate is bit-compared to
/// (review M2 — previously rebuilt per candidate at O(m·n·k)).
struct ImmaProbe {
    m: usize,
    n: usize,
    k: usize,
    reference: Vec<f32>,
    d_qact: CudaSlice<i8>,
    d_weights: CudaSlice<u8>,
    d_act_scale: CudaSlice<f32>,
    d_wscale: CudaSlice<f32>,
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
pub(super) fn pack_i2s_int8_tiles(trits: &[i8], n: usize, k: usize) -> Vec<u8> {
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
pub(super) fn is_oom(err: &DriverError) -> bool {
    // `DriverError`'s Display includes the CUDA status string; the out-of-memory
    // status renders as "out of memory". This keeps us off the unstable numeric
    // status value while still classifying the common case.
    format!("{err}")
        .to_ascii_lowercase()
        .contains("out of memory")
}

/// Classify an allocation/copy failure as OOM (with the requested byte count) or a
/// generic backend error.
pub(super) fn alloc_or_backend(context: &str, err: &DriverError, requested: usize) -> BackendError {
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
pub(super) fn init_cuda() -> Result<Box<dyn TernaryBackend>, BackendError> {
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
