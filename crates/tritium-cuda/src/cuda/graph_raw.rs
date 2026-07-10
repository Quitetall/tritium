//! Raw kernel-launch plumbing for CUDA-graph capture (pre-extracted device
//! pointers, `cuLaunchKernel` bridging) + the batched decode's `BatchKv` pool
//! (P2a split: move-only from `cuda/mod.rs`).

use super::*;

/// A kernel param: a pointer to the arg value `v`. For a pointer arg, `v` is the
/// `CUdeviceptr`; for a by-value arg, `v` is the scalar. Casting a reference to a raw
/// pointer is safe (the deref happens inside `cuLaunchKernel`); the caller keeps `v`
/// alive across the launch (it is a local that outlives `raw_launch`, and graph capture
/// snapshots the value into the kernel node).
pub(super) fn pp<T>(v: &T) -> *mut c_void {
    (v as *const T) as *mut c_void
}

/// Extract a buffer's stable device address, dropping the `SyncOnDrop` guard
/// immediately (outside any capture — its drop records an event, forbidden inside a
/// capture). The `CUdeviceptr` is valid for the buffer's lifetime, so it is safe to
/// bake into a captured graph that the buffer outlives.
pub(super) fn dptr<T>(buf: &CudaSlice<T>, stream: &CudaStream) -> sys::CUdeviceptr {
    let (ptr, guard) = buf.device_ptr(stream);
    drop(guard);
    ptr
}

/// Launch a kernel via the RAW driver entry point, bypassing cudarc's safe launch
/// (whose per-buffer event waits trip `STREAM_CAPTURE_ISOLATION` during capture).
pub(super) fn raw_launch(
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
    unsafe { result::launch_kernel(func, grid, block, smem, stream, params) }.map_err(|e| {
        driver_err(
            &format!(
                "raw graph launch (grid {grid:?} block {block:?} smem {smem} nparams {})",
                params.len()
            ),
            &e,
        )
    })
}

/// Pre-extracted device pointers for one ternary projection.
#[derive(Clone, Copy)]
pub(super) struct LinPtrs {
    pub(super) w: sys::CUdeviceptr,
    pub(super) sc: sys::CUdeviceptr,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) rb: usize,
}

/// Pre-extracted device pointers for one transformer block.
pub(super) struct LayerPtrs {
    pub(super) attn_norm: sys::CUdeviceptr,
    pub(super) attn_sub_norm: Option<sys::CUdeviceptr>,
    pub(super) ffn_norm: sys::CUdeviceptr,
    pub(super) ffn_sub_norm: Option<sys::CUdeviceptr>,
    pub(super) o: LinPtrs,
    pub(super) down: LinPtrs,
    pub(super) qkv: LinPtrs,
    pub(super) gateup: LinPtrs,
    pub(super) kv_k: sys::CUdeviceptr,
    pub(super) kv_v: sys::CUdeviceptr,
    /// i8-rung per-group scale arenas (0 on other rungs; never launched then).
    pub(super) kv_k_sc: sys::CUdeviceptr,
    pub(super) kv_v_sc: sys::CUdeviceptr,
}

/// Pre-extracted device pointers for the shared dense weights + scratch buffers.
#[allow(dead_code)] // CUDA-Graphs decode scaffolding; ptrs wired in as that path lands
pub(super) struct GraphPtrs {
    pub(super) d_x: sys::CUdeviceptr,
    pub(super) d_normed: sys::CUdeviceptr,
    pub(super) d_qkv: sys::CUdeviceptr,
    pub(super) d_gateup: sys::CUdeviceptr,
    pub(super) d_attn: sys::CUdeviceptr,
    pub(super) d_attn_sn: sys::CUdeviceptr,
    pub(super) d_proj_out: sys::CUdeviceptr,
    pub(super) d_gate_sn: sys::CUdeviceptr,
    pub(super) d_scores: sys::CUdeviceptr,
    pub(super) d_logits: sys::CUdeviceptr,
    pub(super) d_qact: sys::CUdeviceptr,
    pub(super) d_act_scale: sys::CUdeviceptr,
    pub(super) d_ctrl: sys::CUdeviceptr,
    pub(super) d_cos: sys::CUdeviceptr,
    pub(super) d_sin: sys::CUdeviceptr,
    pub(super) d_token_embd: sys::CUdeviceptr,
    pub(super) d_token_embd_f16: sys::CUdeviceptr,
    pub(super) d_output_norm: sys::CUdeviceptr,
}

/// Raw-loaded PTX modules + `CUfunction` handles for the v0.3.2 graph-captured decode.
/// Raw (not the safe `CudaModule`/`CudaFunction`) because the captured launch needs the
/// `sys::CUfunction` handle, which the safe `CudaFunction` keeps `pub(crate)`. These are
/// a SECOND JIT of the same PTX the backend already loaded (a few MB of extra SASS);
/// the modules are unloaded on drop.
#[allow(dead_code)] // CUDA-Graphs decode scaffolding; handles wired in as that path lands
pub(super) struct RawGraphKernels {
    pub(super) modules: Vec<sys::CUmodule>,
    pub(super) embed_g: sys::CUfunction,
    pub(super) rope_g: sys::CUfunction,
    pub(super) kv_append: sys::CUfunction,
    /// Fused rope(q)+rope(k)+append(k)+append(v) (v1.x node-count opt).
    pub(super) rope_kv: sys::CUfunction,
    pub(super) attn_g: sys::CUfunction,
    /// v1.x split attention pair (preferred when head_dim % 4 == 0).
    pub(super) attn_scores_g: sys::CUfunction,
    pub(super) attn_reduce_g: sys::CUfunction,
    pub(super) rmsnorm: sys::CUfunction,
    pub(super) rmsnorm_quant: sys::CUfunction,
    pub(super) residual: sys::CUfunction,
    pub(super) relu2: sys::CUfunction,
    pub(super) lm_head: sys::CUfunction,
    pub(super) act_quant: sys::CUfunction,
    pub(super) scale: sys::CUfunction,
    pub(super) tiled: sys::CUfunction,
    pub(super) tiled_scaled: sys::CUfunction,
    pub(super) tiled_scaled_residual: sys::CUfunction,
}

// SAFETY: the raw `CUmodule`/`CUfunction` are process-valid device handles, used only on
// the owning `CudaDecodeModel`'s single capture stream (never concurrently across
// threads — `CudaGraph` is itself documented not-thread-safe, so the whole graph path is
// single-threaded by construction).
#[allow(unsafe_code)]
unsafe impl Send for RawGraphKernels {}

impl RawGraphKernels {
    pub(super) fn load(ctx: &Arc<CudaContext>, kv_dtype: KvDtype) -> Result<Self, BackendError> {
        let sel = |a, b, c| kv_dtype.pick(a, b, c);
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
            kv_append: get(
                dm,
                if kv_dtype == KvDtype::T2 {
                    KERNEL_NAME_KV_APPEND_T2
                } else {
                    sel(
                        KERNEL_NAME_KV_APPEND,
                        KERNEL_NAME_KV_APPEND_H,
                        KERNEL_NAME_KV_APPEND_Q8,
                    )
                },
            )?,
            rope_kv: get(
                dm,
                if kv_dtype == KvDtype::T2 {
                    KERNEL_NAME_ROPE_KV_FUSED_T2
                } else {
                    sel(
                        KERNEL_NAME_ROPE_KV_FUSED,
                        KERNEL_NAME_ROPE_KV_FUSED_H,
                        KERNEL_NAME_ROPE_KV_FUSED_Q8,
                    )
                },
            )?,
            // Warp-per-head attention (bit-identical to the one-thread `_g`, just parallel).
            attn_g: get(dm, KERNEL_NAME_ATTN_WARP)?,
            attn_scores_g: get(
                dm,
                sel(
                    KERNEL_NAME_ATTN_SCORES,
                    KERNEL_NAME_ATTN_SCORES_H,
                    KERNEL_NAME_ATTN_SCORES_Q8,
                ),
            )?,
            attn_reduce_g: get(
                dm,
                sel(
                    KERNEL_NAME_ATTN_REDUCE,
                    KERNEL_NAME_ATTN_REDUCE_H,
                    KERNEL_NAME_ATTN_REDUCE_Q8,
                ),
            )?,
            // Shared-staged rmsnorm: BIT-IDENTICAL to the sequential rmsnorm_f32 (same sum
            // order, from shared), just ~8× faster — so greedy 256/256 holds. (A *parallel*
            // tree-reduction rmsnorm would reach ~132 tok/s but reorders the sum and breaks
            // the gate; this gets most of the win while staying bit-exact.)
            rmsnorm: get(dm, KERNEL_NAME_RMSNORM_SHARED)?,
            // Fused rmsnorm + quant: eliminates one global read+write per norm (v0.7.0).
            rmsnorm_quant: get(dm, KERNEL_NAME_RMSNORM_QUANT_I8)?,
            residual: get(dm, KERNEL_NAME_RESIDUAL)?,
            relu2: get(dm, KERNEL_NAME_RELU2_GATE)?,
            act_quant: get(dm, KERNEL_NAME_ACT_QUANT_TILED_I8)?,
            scale: get(dm, KERNEL_NAME_SCALE_MUL)?,
            // Coalesced warp-per-row LM head reading the f16 token_embd (bit-identical to
            // the f32 warp head — f16 is the GGUF's native precision — at half the read).
            lm_head: get(dm, KERNEL_NAME_LM_HEAD_WARP_F16)?,
            // f32-accumulate GEMM (the f64 one is 1/64-rate on consumer GPUs).
            tiled: get(am, KERNEL_NAME_TILED_F32)?,
            // DP4A fused-scaled variant: packed-int8 activations, act_scale folded
            // into the epilogue (v0.6.0 opt #15 → v1.x i8).
            tiled_scaled: get(am, KERNEL_NAME_TILED_I8_SCALED)?,
            // Fused-scaled + residual: GEMM epilogue adds to residual (v0.7.0 Phase 2).
            tiled_scaled_residual: get(am, KERNEL_NAME_TILED_I8_SCALED_RESIDUAL)?,
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

/// Raw-loaded PTX modules + `CUfunction` handles for the graph-captured **batched** (M=N)
/// decode ([`CudaDecodeModel::decode_batch_graph`]). The batched analogue of
/// [`RawGraphKernels`]: a second JIT of the same `DECODE_PTX` / `TQ2_0_ADD_PTX` the backend
/// already loaded, exposing the raw `sys::CUfunction` handles the captured launches need.
/// The `tiled` GEMM is the f32-accumulate variant — bit-identical to the eager batch path's
/// double-accumulate GEMM here because the activations are int8 and the weights ternary, so
/// each partial product is an exact integer and the running sum never leaves f32's exact
/// integer range (the same equivalence the M=1 graph relies on).
pub(super) struct BatchRawKernels {
    pub(super) modules: Vec<sys::CUmodule>,
    pub(super) embed: sys::CUfunction,
    pub(super) rmsnorm: sys::CUfunction,
    pub(super) quant: sys::CUfunction,
    pub(super) rope: sys::CUfunction,
    pub(super) kv_append: sys::CUfunction,
    #[allow(dead_code)] // kept for fallback; split-KV replaced it
    pub(super) attn: sys::CUfunction,
    pub(super) attn_split_partial: sys::CUfunction,
    pub(super) attn_combine: sys::CUfunction,
    pub(super) residual: sys::CUfunction,
    pub(super) relu2: sys::CUfunction,
    /// i8 dp4a fused-scaled GEMM (v1.x): replaces the former `tiled` + separate
    /// `scale_mul_batch` pair — the epilogue folds the per-row act_scale.
    pub(super) tiled_scaled: sys::CUfunction,
    pub(super) lm_head_tiled: sys::CUfunction,
    pub(super) argmax: sys::CUfunction,
    /// Ctrl-driven tree-verify trunk kernels (graph-capturable twins — read
    /// [prefix_len, real_m] from a device buffer instead of baked scalars).
    pub(super) kv_append_tree: sys::CUfunction,
    pub(super) attn_tree_scores_ctrl: sys::CUfunction,
    pub(super) attn_tree_reduce_ctrl: sys::CUfunction,
}

// SAFETY: same contract as `RawGraphKernels` — the raw handles are process-valid device
// handles used only on the owning model's single capture stream, never concurrently across
// threads (the graph path is single-threaded by construction). `Sync` (which `RawGraphKernels`
// does not need, being owned directly) is additionally required here because the model holds
// these behind an `Arc` (shared with each `BatchKv`), and `Arc<T>: Send` needs `T: Send + Sync`;
// it is sound for the same reason — the handles are immutable after `load` and never touched
// off the single capture stream.
#[allow(unsafe_code)]
unsafe impl Send for BatchRawKernels {}
// SAFETY: as above — the handles are immutable after `load` and only ever used on the single
// capture stream, so concurrent shared (`&`) access across threads observes nothing mutable.
#[allow(unsafe_code)]
unsafe impl Sync for BatchRawKernels {}

impl BatchRawKernels {
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<Self, BackendError> {
        ctx.bind_to_thread()
            .map_err(|e| driver_err("batch raw kernels bind", &e))?;
        let load_mod = |ptx: &str| -> Result<sys::CUmodule, BackendError> {
            let c = CString::new(ptx)
                .map_err(|_| BackendError::InvalidInput("PTX has an interior NUL".into()))?;
            // SAFETY: `c` is a valid NUL-terminated PTX image; `load_data` JIT-compiles it.
            #[allow(unsafe_code)]
            unsafe { result::module::load_data(c.as_ptr() as *const c_void) }
                .map_err(|e| driver_err("batch raw module load_data", &e))
        };
        let get = |m: sys::CUmodule, name: &str| -> Result<sys::CUfunction, BackendError> {
            let c = CString::new(name)
                .map_err(|_| BackendError::InvalidInput("kernel name has a NUL".into()))?;
            // SAFETY: `m` is a loaded module; `name` is one of its `extern "C"` entry points.
            #[allow(unsafe_code)]
            unsafe { result::module::get_function(m, c) }
                .map_err(|e| driver_err("batch raw get_function", &e))
        };
        let dm = load_mod(DECODE_PTX)?;
        let am = load_mod(TQ2_0_ADD_PTX)?;
        Ok(Self {
            embed: get(dm, KERNEL_NAME_EMBED_BATCH)?,
            rmsnorm: get(dm, KERNEL_NAME_RMSNORM_BATCH)?,
            quant: get(dm, KERNEL_NAME_ACT_QUANT_BATCH_I8)?,
            rope: get(dm, KERNEL_NAME_ROPE_BATCH)?,
            kv_append: get(dm, KERNEL_NAME_KV_APPEND_MDECODE)?,
            attn: get(dm, KERNEL_NAME_ATTN_MDECODE)?,
            attn_split_partial: get(dm, KERNEL_NAME_ATTN_SPLIT_PARTIAL)?,
            attn_combine: get(dm, KERNEL_NAME_ATTN_COMBINE)?,
            residual: get(dm, KERNEL_NAME_RESIDUAL)?,
            relu2: get(dm, KERNEL_NAME_RELU2_GATE)?,
            tiled_scaled: get(am, KERNEL_NAME_TILED_I8_SCALED)?,
            lm_head_tiled: get(dm, KERNEL_NAME_LM_HEAD_TILED_F16)?,
            argmax: get(dm, KERNEL_NAME_ARGMAX_ROWS)?,
            kv_append_tree: get(dm, KERNEL_NAME_KV_APPEND_TREE)?,
            attn_tree_scores_ctrl: get(dm, KERNEL_NAME_ATTN_TREE_SCORES_CTRL)?,
            attn_tree_reduce_ctrl: get(dm, KERNEL_NAME_ATTN_TREE_REDUCE_CTRL)?,
            modules: vec![dm, am],
        })
    }
}

impl Drop for BatchRawKernels {
    fn drop(&mut self) {
        for &m in &self.modules {
            if !m.is_null() {
                // SAFETY: each module was loaded by `load` and is unloaded exactly once
                // here; a `BatchKv`'s captured graph must be dropped before the model whose
                // `batch_raw` these are, so nothing launches that graph after this point.
                #[allow(unsafe_code)]
                unsafe {
                    let _ = result::module::unload(m);
                }
            }
        }
    }
}

/// Pre-extracted device pointers for one transformer block of the batched capture. Like
/// [`LayerPtrs`] but the projections are **unfused** (separate q/k/v and gate/up, matching
/// the eager `decode_batch`) and the KV arenas are the **per-batch** ones.
pub(super) struct BatchLayerPtrs {
    pub(super) attn_norm: sys::CUdeviceptr,
    pub(super) attn_sub_norm: Option<sys::CUdeviceptr>,
    pub(super) ffn_norm: sys::CUdeviceptr,
    pub(super) ffn_sub_norm: Option<sys::CUdeviceptr>,
    pub(super) q: LinPtrs,
    pub(super) k: LinPtrs,
    pub(super) v: LinPtrs,
    pub(super) o: LinPtrs,
    pub(super) gate: LinPtrs,
    pub(super) up: LinPtrs,
    pub(super) down: LinPtrs,
    pub(super) kv_k: sys::CUdeviceptr,
    pub(super) kv_v: sys::CUdeviceptr,
}

/// Pre-extracted device pointers for the batched capture's scratch + shared dense weights.
pub(super) struct BatchPtrs {
    pub(super) d_tokens: sys::CUdeviceptr,
    pub(super) d_positions: sys::CUdeviceptr,
    pub(super) d_x: sys::CUdeviceptr,
    pub(super) d_normed: sys::CUdeviceptr,
    pub(super) d_q: sys::CUdeviceptr,
    pub(super) d_k: sys::CUdeviceptr,
    pub(super) d_v: sys::CUdeviceptr,
    pub(super) d_attn: sys::CUdeviceptr,
    pub(super) d_attn_sn: sys::CUdeviceptr,
    pub(super) d_proj: sys::CUdeviceptr,
    pub(super) d_gate: sys::CUdeviceptr,
    pub(super) d_up: sys::CUdeviceptr,
    pub(super) d_gate_sn: sys::CUdeviceptr,
    pub(super) d_qact: sys::CUdeviceptr,
    pub(super) d_act_scale: sys::CUdeviceptr,
    #[allow(dead_code)] // kept for fallback; split-KV partials replaced it
    pub(super) d_scores: sys::CUdeviceptr,
    pub(super) d_attn_partials: sys::CUdeviceptr,
    pub(super) d_cos: sys::CUdeviceptr,
    pub(super) d_sin: sys::CUdeviceptr,
    pub(super) d_token_embd: sys::CUdeviceptr,
    pub(super) d_output_norm: sys::CUdeviceptr,
    // On-device-sampling head (only used when the argmax graph is captured).
    pub(super) d_token_embd_f16: sys::CUdeviceptr,
    pub(super) d_logits_batch: sys::CUdeviceptr,
    pub(super) d_argmax: sys::CUdeviceptr,
}

/// State for batched M=N decode of `n` concurrent sequences (v0.3.7): per-sequence KV
/// arenas + the M=N forward scratch. Built by [`CudaDecodeModel::new_batch`] and advanced
/// by [`CudaDecodeModel::decode_batch`]. Each sequence has its own KV slice
/// (`kv_k[layer]` is `[n, max_ctx, kv_width]`) and its own [`positions`](Self::positions).
#[allow(missing_debug_implementations)]
pub struct BatchKv {
    pub(super) n: usize,
    /// Per-sequence KV capacity (the arena stride, in tokens).
    pub(super) max_ctx: usize,
    pub(super) kv_k: Vec<CudaSlice<f32>>,
    pub(super) kv_v: Vec<CudaSlice<f32>>,
    /// Per-sequence current position (next KV write slot); length `n`.
    pub(super) positions: Vec<usize>,
    pub(super) d_tokens: CudaSlice<i32>,
    pub(super) d_positions: CudaSlice<i32>,
    pub(super) d_x: CudaSlice<f32>,
    pub(super) d_normed: CudaSlice<f32>,
    pub(super) d_q: CudaSlice<f32>,
    pub(super) d_k: CudaSlice<f32>,
    pub(super) d_v: CudaSlice<f32>,
    pub(super) d_attn: CudaSlice<f32>,
    pub(super) d_attn_sn: CudaSlice<f32>,
    pub(super) d_proj: CudaSlice<f32>,
    pub(super) d_gate: CudaSlice<f32>,
    pub(super) d_up: CudaSlice<f32>,
    pub(super) d_gate_sn: CudaSlice<f32>,
    pub(super) d_qact: CudaSlice<i8>,
    pub(super) d_act_scale: CudaSlice<f32>,
    pub(super) d_scores: CudaSlice<f32>,
    /// Split-KV attention partials `[n · n_head · S · (head_dim+2)]`, `S = ceil(max_ctx/ATTN_SPLIT_CHUNK)`.
    pub(super) d_attn_partials: CudaSlice<f32>,
    pub(super) d_h: CudaSlice<f32>,
    pub(super) d_logits: CudaSlice<f32>,
    /// `[n, vocab]` logits scratch for the on-device-sampling graph (the batched LM head
    /// writes here, then the argmax reduces it). Separate from the per-row `d_logits` used
    /// by the eager/logits tail.
    pub(super) d_logits_batch: CudaSlice<f32>,
    /// `[n]` greedy token ids the on-device argmax writes; the only thing the argmax graph
    /// copies back (N·4 bytes vs N·vocab·4).
    pub(super) d_argmax: CudaSlice<i32>,
    /// The captured M=N decode graph for this batch, built lazily on the first
    /// [`CudaDecodeModel::decode_batch_graph`]. Tied to *these* buffers (the capture bakes
    /// their device pointers in), so it lives on the batch, not the model. Ends at the final
    /// RMSNorm (the LM head is the eager tail).
    ///
    /// Declared **before** `raw_keepalive` so the `CUgraphExec` is destroyed before the
    /// module ref it holds is released (the kernel nodes reference functions in those
    /// modules).
    pub(super) graph: Option<SendGraph>,
    /// The captured on-device-sampling graph (built lazily on the first
    /// [`CudaDecodeModel::decode_batch_graph_argmax`]): the same forward plus the batched LM
    /// head + greedy argmax, ending at `d_argmax`. Also declared before `raw_keepalive`.
    pub(super) graph_argmax: Option<CudaGraph>,
    /// A clone of the model's [`batch_raw`](CudaDecodeModel::batch_raw) `Arc`, taken when
    /// either graph is captured. Keeps the `CUfunction` modules the graphs reference loaded
    /// for as long as this batch lives, regardless of whether the producing model is dropped
    /// first — closing the use-after-free the bare doc-convention couldn't enforce. Held,
    /// never read.
    pub(super) raw_keepalive: Option<Arc<BatchRawKernels>>,
}

impl BatchKv {
    /// Number of concurrent sequences.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the batch holds no sequences.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Per-sequence positions (the number of tokens each sequence has decoded).
    /// Set one sequence slot's position (continuous-batching admission: the
    /// slot's KV rows `[0, pos)` must already hold that sequence's cache;
    /// see [`CudaDecodeModel::copy_kv_into_batch_row`]).
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a bad row or a position beyond the
    /// arena.
    pub fn set_position(&mut self, row: usize, pos: usize) -> Result<(), BackendError> {
        if row >= self.n {
            return Err(BackendError::InvalidInput(format!(
                "set_position: row {row} >= batch n {}",
                self.n
            )));
        }
        if pos > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "set_position: pos {pos} > max_ctx {}",
                self.max_ctx
            )));
        }
        self.positions[row] = pos;
        Ok(())
    }

    /// Per-sequence positions (the next KV write slot of each sequence).
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
}
