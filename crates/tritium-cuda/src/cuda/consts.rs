//! Kernel entry-point symbols, launch-geometry constants and embedded PTX
//! for the CUDA backend (P2a split: move-only from `cuda/mod.rs`; every
//! symbol must match its `extern "C"` twin in the `.cu` sources).

/// Kernel entry point — must match the `extern "C"` symbol in the `.cu` file.
/// (cudarc 0.19 keys modules by the returned [`CudaModule`] handle, not by a
/// registered module name, so only the function symbol is needed.)
pub(super) const KERNEL_NAME: &str = "tq2_0_add_mpgemm";
/// The decode-oriented tiled add-only kernel (v0.30 WF-A): one warp per output,
/// one block per row with the activation row staged in shared memory.
pub(super) const KERNEL_NAME_TILED: &str = "tq2_0_add_mpgemm_tiled";
/// f32-accumulate tiled GEMM for the v0.3.2 graph (perf) path — f64 is 1/64-rate on the
/// 4090, the decode bottleneck. Not bit-exact; perplexity-gated.
pub(super) const KERNEL_NAME_TILED_F32: &str = "tq2_0_add_mpgemm_tiled_f32";
pub(super) const KERNEL_NAME_TILED_SCALED: &str = "tq2_0_add_mpgemm_tiled_scaled";
/// DP4A fused-scaled decode GEMMs (v1.x): consume PACKED INT8 activations (the
/// `_i8` quant kernels' output) read directly from global — no shared staging.
/// Exact int32 accumulate; bit-identical to the old f32 fold on this path.
pub(super) const KERNEL_NAME_TILED_I8_SCALED: &str = "tq2_0_add_mpgemm_tiled_i8_scaled";
pub(super) const KERNEL_NAME_TILED_I8_SCALED_RESIDUAL: &str =
    "tq2_0_add_mpgemm_tiled_i8_scaled_residual";
/// Sparse-aware f32-tiled kernel: same as `tiled_f32` but accepts a per-row
/// zero-block bitmap to skip all-zero 256-trit blocks.
pub(super) const KERNEL_NAME_TILED_F32_SPARSE: &str = "tq2_0_add_mpgemm_tiled_f32_sparse";
/// Sparse-aware fused-scaled i8 dp4a kernel: same as `tiled_i8_scaled` but
/// A2: TQ1_0-native decode GEMM twins (entropy-dense weights read natively).
pub(super) const KERNEL_NAME_TQ1_TILED_I8_SCALED: &str = "tq1_0_add_mpgemm_tiled_i8_scaled";
pub(super) const KERNEL_NAME_TQ1_TILED_I8_SCALED_RESIDUAL: &str =
    "tq1_0_add_mpgemm_tiled_i8_scaled_residual";
/// with bitmap skip for zero blocks.
pub(super) const KERNEL_NAME_TILED_I8_SCALED_SPARSE: &str =
    "tq2_0_add_mpgemm_tiled_i8_scaled_sparse";
/// SALT multi-plane accumulate (v0.4.0): sums `Σ_p scale_p·tmatmul(t_p)` over `T`
/// stacked TQ2_0 planes, reading each block's f16 scale. Matches
/// [`tritium_format::dequant_salt_row`] → fp32 matmul within 1e-4.
pub(super) const KERNEL_NAME_SALT: &str = "salt_mpgemm_tiled_f32";
/// The IMMA int8 tensor-core prefill kernel (v0.30 WF-A part 2): one warp per
/// 16×8 output tile, `mma.m16n8k32` int32 accumulate, ternary weights in the
/// [`TernaryFormat::I2sInt8`] tile interleave.
pub(super) const KERNEL_NAME_IMMA: &str = "tq2_0_imma_mpgemm";
/// On-device per-token int8 absmax activation quant (W1.58A8), the first step of
/// the fused `mpgemm_with_act_quant` override.
pub(super) const KERNEL_NAME_ACT_QUANT: &str = "act_quant_int8_per_token";
/// The device-resident RMSNorm decode kernel (v0.3.1) — bit-matches the host
/// `tritium_nn::ops::rmsnorm` (ADR 0018 canonical tree sum-of-squares, no FMA).
pub(super) const KERNEL_NAME_RMSNORM: &str = "rmsnorm_f32";
/// The device-resident RoPE decode kernel (v0.3.1) — bit-matches the host
/// `tritium_nn::ops::rope_apply` (precomputed f64→f32 trig, f32 rotation, no FMA).
pub(super) const KERNEL_NAME_ROPE: &str = "rope_apply_f32";
/// The device-resident softmax decode kernel (v0.3.1) — matches
/// `tritium_nn::ops::softmax_rows`; only `expf` may differ from host libm by ~1 ULP.
pub(super) const KERNEL_NAME_SOFTMAX: &str = "softmax_f32";
/// Residual add `x += y` (exact f32 add).
pub(super) const KERNEL_NAME_RESIDUAL: &str = "residual_add_f32";
/// Embedding-table row gather (exact copy).
pub(super) const KERNEL_NAME_EMBED: &str = "embedding_gather_f32";
/// Tied LM head `logits[v] = <h, embd[v]>` (sequential dot, bit-matches host).
pub(super) const KERNEL_NAME_LM_HEAD: &str = "lm_head_f32";
/// GQA attention, M=1 decode (v0.3.1) — matches `ops::gqa_attention`; dots/weighted
/// sums bit-match, the inline softmax `expf` differs ≤3 ULP.
pub(super) const KERNEL_NAME_ATTN: &str = "gqa_attention_decode_f32";
/// Flash-decoding (split-KV) attention pair — the low-N occupancy fix.
pub(super) const KERNEL_NAME_ATTN_SPLIT_PARTIAL: &str = "gqa_attention_split_partial_f32";
pub(super) const KERNEL_NAME_ATTN_COMBINE: &str = "gqa_attention_combine_f32";
/// Keys per split-KV attention chunk. Fixed (not ctx-dependent) so the captured
/// graph's grid `n·n_head·ceil(max_ctx/CHUNK)` is valid for every decode step.
// WIRED BY: mimo 2.5 pro — split-KV attention into M=N decode (plans/0001)
pub(super) const ATTN_SPLIT_CHUNK: usize = 64;
/// On-device int8 activation quant for the tiled (TQ2_0) decode GEMM (v0.3.1) —
/// bit-matches `ops::quantize_activation_int8` (int8 kept as f32 + per-token scale);
/// its rmsnorm-fused sibling folds the sum in the ADR 0018 canonical order.
pub(super) const KERNEL_NAME_ACT_QUANT_TILED: &str = "act_quant_tiled_f32";
pub(super) const KERNEL_NAME_RMSNORM_QUANT: &str = "rmsnorm_quant_f32";
/// i8-emitting siblings of the two quants above + the batch quant (v1.x): same
/// math, `q_out` is packed int8 for the `tiled_i8_scaled*` dp4a GEMMs. The f32
/// originals stay for the public host-facing quant helpers.
pub(super) const KERNEL_NAME_ACT_QUANT_TILED_I8: &str = "act_quant_tiled_i8";
pub(super) const KERNEL_NAME_RMSNORM_QUANT_I8: &str = "rmsnorm_quant_i8";
pub(super) const KERNEL_NAME_ACT_QUANT_BATCH_I8: &str = "act_quant_batch_i8";
/// Threads for the `rmsnorm_quant_i8` launches. The kernel pins its canonical
/// fold to 256 slots explicitly, so extra threads (ncu: the kernel is
/// single-block latency-bound at ~1% of every throughput ceiling with 256)
/// legally accelerate its elementwise passes. Must be a multiple of 32, ≥ 256,
/// ≤ 1024 (`s_red[1024]`).
pub(super) const RMSNORM_QUANT_THREADS: u32 = 512;
/// Per-token activation-scale fold `out *= act_scale` (v0.3.1) — the device half of
/// the W1.58A8 dequant the host applies after the GEMM.
pub(super) const KERNEL_NAME_SCALE_MUL: &str = "scale_mul_f32";
/// BitNet squared-ReLU FFN gate `g = relu(g)² ⊙ u` (v0.3.1) — bit-matches the host
/// `mlp` gating loop (`r = g.max(0); g = r*r*u`).
pub(super) const KERNEL_NAME_RELU2_GATE: &str = "relu2_gate_f32";
/// v0.3.2 graph variants reading the per-token control block (token/pos/cache_len).
pub(super) const KERNEL_NAME_EMBED_G: &str = "embedding_gather_f32_g";
pub(super) const KERNEL_NAME_ROPE_G: &str = "rope_apply_f32_g";
pub(super) const KERNEL_NAME_KV_APPEND: &str = "kv_append_f32";
/// f16-KV twins (ADR 0020 rung 1): selected into the SAME function handles at
/// build when the model runs with `TRITIUM_KV_F16=1`, so launch sites don't
/// change. Suffix `_h` = `__half` KV element type; math identical.
pub(super) const KERNEL_NAME_KV_APPEND_H: &str = "kv_append_h";
pub(super) const KERNEL_NAME_KV_APPEND_BATCH_H: &str = "kv_append_batch_h";
pub(super) const KERNEL_NAME_ROPE_KV_FUSED_H: &str = "rope_kv_fused_h";
pub(super) const KERNEL_NAME_ATTN_SCORES_H: &str = "gqa_attention_scores_h";
pub(super) const KERNEL_NAME_ATTN_REDUCE_H: &str = "gqa_attention_reduce_h";
pub(super) const KERNEL_NAME_ATTN_BATCH_H: &str = "gqa_attention_batch_h";
pub(super) const KERNEL_NAME_ATTN_TREE_SCORES_H: &str = "gqa_attention_tree_scores_h";
pub(super) const KERNEL_NAME_ATTN_TREE_REDUCE_H: &str = "gqa_attention_tree_reduce_h";
pub(super) const KERNEL_NAME_KV_APPEND_Q8: &str = "kv_append_q8";
pub(super) const KERNEL_NAME_KV_APPEND_BATCH_Q8: &str = "kv_append_batch_q8";
pub(super) const KERNEL_NAME_ROPE_KV_FUSED_Q8: &str = "rope_kv_fused_q8";
pub(super) const KERNEL_NAME_ATTN_SCORES_Q8: &str = "gqa_attention_scores_q8";
pub(super) const KERNEL_NAME_ATTN_REDUCE_Q8: &str = "gqa_attention_reduce_q8";
pub(super) const KERNEL_NAME_ATTN_BATCH_Q8: &str = "gqa_attention_batch_q8";
pub(super) const KERNEL_NAME_ATTN_TREE_SCORES_Q8: &str = "gqa_attention_tree_scores_q8";
pub(super) const KERNEL_NAME_ATTN_TREE_REDUCE_Q8: &str = "gqa_attention_tree_reduce_q8";
pub(super) const KERNEL_NAME_KV_APPEND_T2: &str = "kv_append_t2";
pub(super) const KERNEL_NAME_KV_APPEND_BATCH_T2: &str = "kv_append_batch_t2";
pub(super) const KERNEL_NAME_ROPE_KV_FUSED_T2: &str = "rope_kv_fused_t2";

/// Fused q-RoPE + k-RoPE + K/V append for the decode graph (v1.x): one launch
/// replaces four, bit-identical values (see the kernel doc).
pub(super) const KERNEL_NAME_ROPE_KV_FUSED: &str = "rope_kv_fused_g";
/// Warp-per-head GQA attention for the graph (v0.3.3 perf) — bit-identical to the
/// one-thread `_g` kernel (parallel across keys/output-dims, no reduction reorder).
/// v1.x: legacy/fallback — the split scores+reduce pair below is preferred when
/// `head_dim % 4 == 0` (float4 K rows); this stays for other geometries.
pub(super) const KERNEL_NAME_ATTN_WARP: &str = "gqa_attention_decode_warp_g";
/// Split decode attention (v1.x): ctx-parallel scores fan-out + per-head 128-thread
/// softmax/weighted reduce. Bit-identical to the warp kernel (no f32 sum reordered);
/// measured 2×–9× over it across context lengths. `SCORE_CHUNK` mirrors decode.cu.
pub(super) const KERNEL_NAME_ATTN_SCORES: &str = "gqa_attention_scores_g";
pub(super) const KERNEL_NAME_ATTN_REDUCE: &str = "gqa_attention_reduce_g";
/// BASTION tree-verify attention (ADR 0014): batch attention where row `i`
/// attends the shared prefix + its ancestor slots instead of a contiguous range.
pub(super) const KERNEL_NAME_ATTN_TREE: &str = "gqa_attention_tree_f32";
pub(super) const KERNEL_NAME_ATTN_TREE_SCORES: &str = "gqa_attention_tree_scores_g";
pub(super) const KERNEL_NAME_KV_APPEND_TREE: &str = "kv_append_tree_g";
pub(super) const KERNEL_NAME_ARGMAX_PARTIAL: &str = "argmax_rows_partial_f32";
pub(super) const KERNEL_NAME_ARGMAX_COMBINE: &str = "argmax_rows_combine_f32";
/// Keep in sync with `ARGMAX_CHUNKS` in decode.cu.
pub(super) const ARGMAX_CHUNKS: usize = 16;
pub(super) const KERNEL_NAME_ATTN_TREE_SCORES_CTRL: &str = "gqa_attention_tree_scores_ctrl_g";
pub(super) const KERNEL_NAME_ATTN_TREE_REDUCE_CTRL: &str = "gqa_attention_tree_reduce_ctrl_g";
pub(super) const KERNEL_NAME_ATTN_TREE_REDUCE: &str = "gqa_attention_tree_reduce_g";
/// Keys per scores-block — keep in sync with `SCORE_CHUNK` in decode.cu.
pub(super) const ATTN_SCORE_CHUNK: usize = 128;
/// Threads per reduce block — keep ≤ its `s_red[256]` scratch and a power of two.
pub(super) const ATTN_REDUCE_THREADS: u32 = 128;
/// Shared-staged rmsnorm for the graph (v0.3.4 perf) — bit-identical to rmsnorm_f32
/// (same ADR 0018 canonical tree order, just reading a shared stage instead of global).
pub(super) const KERNEL_NAME_RMSNORM_SHARED: &str = "rmsnorm_shared_f32";
/// f16-`token_embd` warp LM head for the graph (v0.3.4 perf) — bit-identical to the f32
/// warp head (f16 is the GGUF's native precision), halves the 1.3 GB/token table read.
pub(super) const KERNEL_NAME_LM_HEAD_WARP_F16: &str = "lm_head_warp_f16";
/// v0.3.6 batched (M>1) prefill kernels — process the whole prompt in one forward.
pub(super) const KERNEL_NAME_RMSNORM_BATCH: &str = "rmsnorm_batch_f32";
pub(super) const KERNEL_NAME_EMBED_BATCH: &str = "embedding_gather_batch_f32";
pub(super) const KERNEL_NAME_ROPE_BATCH: &str = "rope_apply_batch_f32";
// (act_quant_batch_f32 remains in decode.cu but has no host consumer since the
// batch paths moved to the i8 quant + dp4a GEMMs — see KERNEL_NAME_ACT_QUANT_BATCH_I8.)
pub(super) const KERNEL_NAME_SCALE_BATCH: &str = "scale_mul_batch_f32";
pub(super) const KERNEL_NAME_KV_APPEND_BATCH: &str = "kv_append_batch_f32";
pub(super) const KERNEL_NAME_ATTN_BATCH: &str = "gqa_attention_batch_f32";
/// v0.3.7 batched M=N decode (N concurrent sequences, per-sequence KV).
/// The direct M=N attention (`gqa_attention_mdecode_f32`) was retired in ADR
/// 0025 step 2 — the split partial+combine pair is the only batch attention
/// (a loaded-but-never-launched dense-indexed fallback would have silently
/// bypassed paging).
pub(super) const KERNEL_NAME_KV_APPEND_MDECODE: &str = "kv_append_mdecode_f32";
/// ADR 0025 paged-KV twins: kv pools + per-slot page table instead of the
/// dense `[n, max_ctx, kv_width]` arenas.
pub(super) const KERNEL_NAME_KV_APPEND_MDECODE_PAGED: &str = "kv_append_mdecode_paged_f32";
pub(super) const KERNEL_NAME_ATTN_SPLIT_PARTIAL_PAGED: &str =
    "gqa_attention_split_partial_paged_f32";
/// Tokens per KV page (ADR 0025). MUST equal decode.cu's KV_PAGE_TOKENS —
/// the paged==dense bit-equality gate breaks instantly on a mismatch.
pub const KV_PAGE_TOKENS: usize = 256;

/// v0.3.8 on-device sampling for the batched decode graph.
pub(super) const KERNEL_NAME_LM_HEAD_TILED_F16: &str = "lm_head_tiled_f16";
pub(super) const KERNEL_NAME_ARGMAX_ROWS: &str = "argmax_rows_f32";
/// v0.50 (ADR 0007) f32 training backward kernels for the ternary matmul.
pub(super) const KERNEL_NAME_TRAIN_FWD: &str = "ternary_matmul_forward";
pub(super) const KERNEL_NAME_GRAD_A: &str = "ternary_matmul_grad_a";
pub(super) const KERNEL_NAME_GRAD_W: &str = "ternary_matmul_grad_w";
pub(super) const KERNEL_NAME_GRAD_S: &str = "ternary_matmul_grad_s";
/// ADR 0027 Track A: resident per-row multi-plane SALT quantization.
pub(super) const KERNEL_NAME_SALT_QUANTIZE_FWD: &str = "salt_quantize_forward";
/// ADR 0027 Track A: fused resident AdamW parameter/moment update.
pub(super) const KERNEL_NAME_ADAMW_STEP: &str = "adamw_step";
/// ADR 0027 Track D: master-to-compact-plane pack and fused packed contractions.
pub(super) const KERNEL_NAME_SALT_PACK_TRAINING: &str = "salt_pack_training";
pub(super) const KERNEL_NAME_SALT_TRAINING_FORWARD: &str = "salt_training_forward";
pub(super) const KERNEL_NAME_SALT_TRAINING_GRAD_A: &str = "salt_training_grad_a";
pub(super) const KERNEL_NAME_SALT_TRAINING_FORWARD_EXACT: &str = "salt_training_forward_exact";
pub(super) const KERNEL_NAME_SALT_TRAINING_GRAD_A_EXACT: &str = "salt_training_grad_a_exact";
pub(super) const KERNEL_NAME_SALT_TRAINING_FORWARD_EXACT_TILED: &str =
    "salt_training_forward_exact_tiled";
pub(super) const KERNEL_NAME_SALT_TRAINING_GRAD_A_EXACT_TILED: &str =
    "salt_training_grad_a_exact_tiled";
pub(super) const KERNEL_NAME_SALT_TRAINING_EMBED: &str = "salt_training_embed_gather";
/// plan 0043 P2.2 device-resident glue ops (elementwise fwd/bwd + grad accumulate).
pub(super) const KERNEL_NAME_SILU_FWD: &str = "silu_forward";
pub(super) const KERNEL_NAME_SILU_BWD: &str = "silu_backward";
pub(super) const KERNEL_NAME_EW_MUL_FWD: &str = "ew_mul_forward";
pub(super) const KERNEL_NAME_EW_MUL_BWD: &str = "ew_mul_backward";
pub(super) const KERNEL_NAME_EW_ADD_FWD: &str = "ew_add_forward";
pub(super) const KERNEL_NAME_ACCUMULATE: &str = "accumulate";
/// plan 0043 P2.3 device-resident training RMSNorm (sequential order, mirrors ops::norm).
pub(super) const KERNEL_NAME_RMSNORM_TRAIN_FWD: &str = "rmsnorm_train_forward";
pub(super) const KERNEL_NAME_RMSNORM_TRAIN_INV: &str = "rmsnorm_train_inv";
pub(super) const KERNEL_NAME_RMSNORM_TRAIN_GRAD_X: &str = "rmsnorm_train_grad_x";
pub(super) const KERNEL_NAME_RMSNORM_TRAIN_GRAD_W: &str = "rmsnorm_train_grad_w";
/// plan 0043 P2.4 device-resident attention glue (softmax/mask/rope/reshape/gather/xent).
pub(super) const KERNEL_NAME_SOFTMAX_FWD: &str = "softmax_forward";
pub(super) const KERNEL_NAME_SOFTMAX_BWD: &str = "softmax_backward";
pub(super) const KERNEL_NAME_CAUSAL_MASK_FWD: &str = "causal_mask_forward";
pub(super) const KERNEL_NAME_CAUSAL_MASK_BWD: &str = "causal_mask_backward";
pub(super) const KERNEL_NAME_ROPE_APPLY: &str = "rope_apply";
pub(super) const KERNEL_NAME_SLICE_COLS_FWD: &str = "slice_cols_forward";
pub(super) const KERNEL_NAME_COPY_INTO_COLS: &str = "copy_into_cols";
pub(super) const KERNEL_NAME_TRANSPOSE_FWD: &str = "transpose_forward";
pub(super) const KERNEL_NAME_EMBED_GATHER_FWD: &str = "embed_gather_forward";
pub(super) const KERNEL_NAME_EMBED_GATHER_BWD: &str = "embed_gather_backward";
pub(super) const KERNEL_NAME_EMBED_GATHER_BWD_SEGMENTED: &str = "embed_gather_backward_segmented";
pub(super) const KERNEL_NAME_SOFTMAX_XENT_BWD: &str = "softmax_xent_backward";
pub(super) const KERNEL_NAME_SCALE_CONST: &str = "scale_const";
/// Plan 0043 Stage 6: direct scalar-correct SALT V2 D2/B3/S34 execution.
pub(super) const KERNEL_NAME_SALT_V2_EXACT: &str = "salt_v2_forward_exact";
/// Plan 0043 Stage 6: exact selected-row reconstruction for token embeddings.
pub(super) const KERNEL_NAME_SALT_V2_GATHER: &str = "salt_v2_gather_rows";
/// Row-tile of [`KERNEL_NAME_LM_HEAD_TILED_F16`] — keep in sync with `LMHEAD_ROW_TILE` in decode.cu.
pub(super) const LMHEAD_ROW_TILE: u32 = 8;
/// Threads per block for `act_quant_int8_per_token` — must match the kernel's
/// `ACT_QUANT_THREADS` (its shared reduction is sized for this, a power of two).
pub(super) const ACT_QUANT_THREADS: u32 = 256;
/// CUDA threads per block for the 1-D launch grid (simple kernel).
pub(super) const THREADS_PER_BLOCK: u32 = 256;
/// Warps per block for the tiled kernel — each warp computes one output column,
/// so a block covers this many `N` at once (8 warps = 256 threads). ncu note
/// (2026-07-07): the small decode GEMMs run at 0.42–0.62 waves / 44–48% DRAM —
/// warp-count-starved by the shape itself (N columns < machine warp capacity
/// at M=1). Halving block width (4 warps) changes nothing (same total warps),
/// and three split-K variants measured slower; the architectural fix for the
/// underfill is batching rows (M=N decode / BASTION tree-verify), not a finer
/// launch geometry.
pub(super) const WARPS_PER_BLOCK: u32 = 8;
/// Largest `K` the tiled kernel accepts: it stages `K` f32 activations in shared
/// memory (`K * 4` bytes = 32 KiB at the cap), comfortably under the 48 KiB
/// default dynamic-shared budget and covering every BitNet shape (max K = 6912).
pub(super) const TILED_K_MAX: usize = 8_192;
/// Largest `M` routed to the tiled (decode) kernel. Above this the problem is
/// prefill-shaped and the one-thread-per-output kernel is the better default
/// until the IMMA tensor-core kernel lands (WF-A part 2).
pub(super) const TILED_M_MAX: usize = 64;

/// The PTX produced by `build.rs` (`nvcc -ptx`). Embedded at compile time so the
/// backend needs no PTX file on disk at runtime.
pub(super) const TQ2_0_ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_add.ptx"));
/// The IMMA prefill kernel + the on-device act-quant kernel, compiled by `build.rs`
/// to a SECOND PTX target at `compute_80` (the `mma.m16n8k32` int8 shape needs
/// sm_80+, above the add kernel's sm_75 floor). Embedded the same way.
pub(super) const TQ2_0_IMMA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_imma.ptx"));
/// The device-resident decode kernels (v0.3.1), compiled `--fmad=false` so they
/// reproduce the host f32 ops bit-for-bit. Embedded the same way as the others.
pub(super) const DECODE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode.ptx"));
/// The v0.50 training backward kernels (gA/gW/gs), compiled `--fmad=false` so they
/// match the host CPU vjp oracle bit-for-bit. Embedded the same way as the others.
pub(super) const TRAIN_GRAD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/train_grad.ptx"));
/// Direct SALT V2 D2/B3/S34 kernel, compiled with FMA contraction disabled.
pub(super) const SALT_V2_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/salt_v2.ptx"));
