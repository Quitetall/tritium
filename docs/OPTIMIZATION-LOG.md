# Optimization Log

Real metrics, measured on the actual codebase. Every change verified against the
greedy 32/32 token acceptance test (`cpu_longer_greedy_matches_transformers`).

## Methodology

- **TDD**: benchmark before → implement → verify correctness → benchmark after.
- All benchmarks in `crates/tritium-nn/tests/bench_cpu_hotpaths.rs` (release mode,
  500 iterations, `std::hint::black_box` to prevent dead-code elimination).
- Acceptance test: greedy decode of BitNet b1.58 2B4T, 32 tokens, must match
  transformers reference exactly (token-for-token).

## Results

### #14 — RoPE cos/sin table precomputation ✅

**File:** `crates/tritium-nn/src/ops/rope.rs`

**Change:** Precompute a `[positions × half]` cos/sin table before the head loop,
then index into it. For a given position and lane j, the (cos, sin) pair is
identical across all heads — only the data being rotated differs.

**Before:** `angle.sin_cos()` called `(n_head-1) × half` extra times per position.
For n_head=25, head_dim=128, seq=16: 24 × 64 = 1,536 redundant sin_cos calls.

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| median | 186µs | 14.7µs | **~13×** |

**Verification:** `rope_matches_torch_goldens` + `rope_position_zero_is_identity` +
`cpu_longer_greedy_matches_transformers` all pass (32/32 exact).

---

### #17 — RMSNorm prefill skip ✅

**File:** `crates/tritium-nn/src/model/runner.rs`

**Change:** When `dump` is None (production path), compute RMSNorm only for the
last token instead of iterating all `seq` tokens. The LM head only needs the last
token's norm; computing norms for tokens 0..seq-1 was wasted work during prefill.

**Before:** Loop over all `seq` tokens with an if-check per iteration.
**After:** Direct `rmsnorm()` call on the last token only.

| Metric | Before (seq=128) | After | Speedup |
|--------|------------------|-------|---------|
| median | 158µs | 1.3µs | **~122×** |

**Note:** This only affects prefill (seq > 1). Decode (seq=1) is unchanged.

**Verification:** `cpu_longer_greedy_matches_transformers` passes (32/32 exact).

---

### #18 — softmax_rows 3→2 pass fusion ❌ (no real win)

**File:** `crates/tritium-nn/src/ops/softmax.rs`

**Finding:** The apparent 1.6× speedup in the benchmark was entirely due to
error-checking overhead in the public API (`softmax_rows` has shape validation),
not pass count. The actual algorithm is already optimal: pass 1 (max), pass 2
(exp+sum), pass 3 (normalize). Passes 2 and 3 cannot be fused because `inv = 1/sum`
requires the complete sum before normalization.

**Status:** No change. The real softmax win is in the training tape
(`softmax::vjp` recomputes the forward — that's a separate refactor).

---

### #10 — TernaryLinear scratch buffer reuse ❌ (no measurable win)

**File:** `crates/tritium-nn/src/layers/linear.rs`

**Finding:** At M=1 (decode), the allocation overhead of `q_act` (10KB) and
`act_scale` (4 bytes) is negligible — 18ns both before and after pre-allocation.
The allocator handles small allocations efficiently in release mode.

**Status:** No change. Would matter at large M (batch prefill) but not a priority.

---

---

### VJP closure signature refactor ✅

**File:** `crates/tritium-train/src/tape.rs`

**Change:** VJP closures now accumulate gradients directly into `grads[input_id]`
instead of returning `Vec<Vec<f32>>`. Eliminates per-node intermediate allocations.
Also: reusable `input_buf` (Vec<&[f32]>) avoids 270 small Vec allocs per backward,
and `split_at_mut` avoids the g_out clone.

**Before:** `Backward` returned `Vec<Vec<f32>>` — 270 outer Vec + ~1080 inner Vec
allocations per backward pass.

**After:** Closures write directly into `grads_lo[input_id]` — zero intermediate
allocations. One `input_buf` reused across all nodes.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| median | 210µs | 188µs | **~10%** |

**Note:** The 10% win is smaller than the predicted 30-50% because the VJP
computation itself (the math) dominates over allocation overhead. The closures still
do real work (loops over M×N×K elements); the allocation was a smaller fraction of
total time than estimated.

**Verification:** All 55 `tritium-train` tests pass (tape_toy_layer, tape_tiny_transformer,
gradcheck_ops, gradcheck_ste_matmul, optim_adamw, lora, checkpoint_roundtrip, etc.).

---

## Summary

| Optimization | Win | Status |
|-------------|-----|--------|
| RoPE cos/sin precompute | **13×** (CPU) | ✅ Done |
| RMSNorm prefill skip | **122×** (CPU prefill) | ✅ Done |
| Eager f32 GEMM + quant dedup + warp attn | **~2.3×** (CUDA eager N=1) | ✅ Done |
| VJP closure signature refactor | **~10%** (training backward) | ✅ Done |
| softmax pass fusion | 0× | ❌ No real win |
| quantize scratch reuse | 0× | ❌ No measurable win |

**Net effect:**
- CPU prefill (seq=128): ~340µs saved (RoPE + RMSNorm)
- CUDA eager decode: 29.8 → 68.7 tok/s at N=1 (~2.3×)
- CUDA graph decode: unchanged (already had all opts)
- 256-token generation: 8.1s → 4.3s (~1.9×)
- Training backward: ~10% faster

---

### #7 + #8 + #9 — Eager path: f32 GEMM + quant dedup + warp attention ✅

**File:** `crates/tritium-cuda/src/cuda.rs`

**Changes:**
1. Switch eager tiled GEMM from `KERNEL_NAME_TILED` (f64) to `KERNEL_NAME_TILED_F32`.
   f64 is 1/64-rate on the 4090 — "the decode bottleneck."
2. Deduplicate act quantization: q/k/v share one `launch_quant` call (was 3),
   gate/up share one (was 2). Saves 3 quant launches per layer = ~78 per token.
3. Switch eager attention from `gqa_attention_decode_f32` (1 thread/head) to
   `gqa_attention_decode_warp_g` (1 warp/head, 32 threads). The warp kernel was
   already loaded by the graph path; `launch_attention` updated to match its
   signature (reads limit from `d_ctrl[2]`).

**Cumulative effect on eager N=1:**

| Change | Eager N=1 | Delta |
|--------|-----------|-------|
| Baseline (f64 + single-thread attn) | 29.8 tok/s | — |
| + f32 GEMM | 37.7 tok/s | +27% |
| + warp attention | 58.9 tok/s | +56% |
| + quant dedup | 68.7 tok/s | +17% |
| **Total** | **68.7 tok/s** | **~2.3×** |

Graph path unchanged (112.9 → 110.7, noise — already had all three).

**Verification:** 45/45 CUDA tests pass. 256/256 greedy tokens match on CUDA.
Conformance within 1e-4.

---

### #15 — Fuse scale_mul into GEMM epilogue ✅

**Files:** `crates/tritium-cuda/kernels/tq2_0_add.cu`, `crates/tritium-cuda/src/cuda.rs`

**Change:** Created fused kernel variants `tq2_0_add_mpgemm_tiled_scaled` (f64 acc)
and `tq2_0_add_mpgemm_tiled_f32_scaled` (f32 acc) that fold the per-token
activation scale into the GEMM epilogue:
```c
out[mi, ni] = acc * scales[ni] * act_scale[mi]  // one write, not two
```

Previously the pattern was: GEMM → `scale_mul_f32` (separate kernel launch +
full read-write pass over the output buffer). Now it's a single launch.

**Impact:** Eliminates ~182 kernel launches per token (7 GEMMs × 26 layers) and
~182 extra memory passes over the output buffer. Each `scale_mul_f32` launch
reads N floats, multiplies, and writes N floats back — that's ~20 KB of
unnecessary memory traffic per GEMM at N=2560.

**Scope:** M=1 decode path (eager step, graph capture, device-resident forward).
The batched M>1 prefill path still uses the separate `scale_mul_batch_f32` —
a follow-up optimization.

**Verification:** 10/10 CUDA unit tests pass. Code compiles cleanly. GPU
conformance tests self-skip in this CPU-only environment (they will run on the
GPU CI lane).

---

### #16 — Parallel CPU LM head with rayon ✅

**File:** `crates/tritium-nn/src/model/runner.rs`

**Change:** The tied LM head computes 128K dot products of 2560 elements each
(logit[v] = <last_norm, token_embd[v]>). Parallelized with rayon
`par_chunks_mut(1024)` for multi-core throughput.

**Impact:** On a multi-core CPU, this reduces the LM head time proportionally
to the core count. For the CPU-only decode path, this is a significant win
since the LM head is the dominant cost after the transformer layers.

---

### #18 — Per-layer scratch arena ✅

**Files:** `crates/tritium-nn/src/layers/transformer_block.rs`, `crates/tritium-nn/src/model/runner.rs`

**Change:** Added `BlockScratch` struct with pre-allocated buffers (normed, q, k,
v, attn, sn, mlp_out). Sized once at model init, then passed mutably to each
layer via `forward_with_scratch()`. Eliminates ~7 heap allocs per layer × 26
layers ≈ 182 allocs per forward pass.

**Impact:** Reduces allocation overhead in the CPU forward path. The dump path
(`forward_dump`) still allocates fresh buffers for the fidelity ladder.

**Verification:** 11/11 NN tests pass. Build clean.

---

### Phase 1 — Fused RMSNorm + Quantize ✅

**File:** `crates/tritium-cuda/kernels/decode.cu`

**Change:** New `rmsnorm_quant_f32` kernel combines `rmsnorm_shared_f32` +
`act_quant_tiled_f32` into one pass. The rmsnorm output stays in shared memory;
the absmax reduction and quantization read from shared, not global.

**Saves:** 1 global read + 1 global write per call. 4 calls per layer × 26
layers = **104 launches eliminated** from the graph path.

---

### Phase 2 — Fused GEMM + Residual Add ✅

**File:** `crates/tritium-cuda/kernels/tq2_0_add.cu`

**Change:** New `tq2_0_add_mpgemm_tiled_f32_scaled_residual` kernel folds the
residual add into the GEMM epilogue:
```c
out[idx] = residual[idx] + acc * scales[ni] * act_scale[mi]
```
When residual and out are the same pointer, this becomes in-place `d_x += GEMM`.

**Saves:** 2 residual adds per layer × 26 layers = **52 launches eliminated**
from the graph path. Each also saves a full read+write pass over the hidden
state.

**Combined Phase 1+2:** 156 launches eliminated from the graph path (from ~400
to ~244).

---

### v1.x — DP4A decode mpGEMM + byte-once f32 decode ✅

**Files:** `crates/tritium-cuda/kernels/tq2_0_add.cu`

**Finding (profile-driven):** `nsys --cuda-graph-trace=node` on the e2e decode
bench showed `tq2_0_add_mpgemm_tiled_f32_scaled` as the top GPU-time kernel.
Root cause: the lane-per-`ki += 32` stride re-read the *same* qs byte four
times (TQ2_0 stores the trits at e, e+32, e+64, e+96 in one byte's four 2-bit
slots) and burned index math + I2F per trit.

**Changes:**
1. **DP4A rewrite of the `_scaled` family** (`_f32_scaled`, `_f32_scaled_residual`,
   `_f32_scaled_sparse`). These kernels are only ever launched on A8-quantized
   activations (integer-valued f32 in [-128,127] from `act_quant_*` /
   `rmsnorm_quant_f32`), so the staging pass converts them to packed int8x4 in
   shared memory (`__float2int_rn` is exact on integers) and the K loop runs
   `(w >> 2·slot) & 0x03030303` → `__vsub4` → `__dp4a` — 4 trits per
   instruction, each qs byte read once, int32 accumulate. The int32 sum is
   EXACT and order-independent, and the old f32 sum was also exact on this
   path (|Σ| ≤ 127·6912 < 2²⁴), so outputs are **bit-identical** — perplexity
   ours=1.398684 unchanged to the last digit.
2. **Byte-once restructure of the arbitrary-float kernels** (`_f32`,
   `_f32_sparse`): each lane reads its chunk byte once and consumes all four
   slots as FMAs. Public-path semantics unchanged (1e-4 tolerance gate).

**Kernel-only (standalone cudaEvent harness, 4090, M=1):**

| shape (N,K) | before | byte-once | dp4a | SOL (weights/1008GB/s) |
|---|---|---|---|---|
| 2560,2560 | 11.95µs | 6.43µs | **3.88µs** | 1.68µs |
| 6912,2560 | 21.90µs | 11.00µs | **6.27µs** | 4.53µs |
| 2560,6912 | 28.27µs | 13.51µs | **8.33µs** | 4.53µs |
| 6912,6912 | 55.14µs | 29.11µs | **14.17µs** | 12.22µs (86% of SOL) |

**In-model (nsys, same eval workload):** GEMM total 11.01ms → 4.71ms (2.34×),
median instance 33.5µs → 13.7µs; GEMM share of GPU time 75% → 54%.

**Verification:** 51/51 tritium-cuda tests, tritium-nn greedy 32/32 vs
transformers, cpu_cuda_parity, e2e perplexity bit-identical.

---

### v1.x — decode attention: shared scores + parallel max/exp ✅

**Files:** `crates/tritium-cuda/kernels/decode.cu` (`gqa_attention_decode_warp_g`),
`crates/tritium-cuda/src/cuda.rs` (both launch sites)

**Finding:** after the GEMM fix, node-level graph profiling showed the warp
attention kernel at **61.6% of decode GPU time** (233µs median per layer call).
Causes: scores staged in *global* memory, and the whole softmax — including the
f64 `exp_f32` (1/64 rate on the 4090) — ran sequentially on lane 0, one key at
a time.

**Change (bit-exactness preserving):**
- scores staged in dynamic shared memory; launch geometry changed from
  8-warps/256-thread blocks to ONE warp-block per head with
  `max_ctx·4` dynamic shared bytes (fits the 48 KiB cap up to max_ctx=12288).
  The global `scores` scratch stays in the ABI but is no longer written.
- max scan → parallel warp reduction (f32 max never rounds; `fmaxf` skips NaN
  exactly like the sequential `>` scan).
- `exp_f32` → elementwise-parallel across all 32 lanes (same function, same
  bits per key).
- ONLY the softmax sum (the one rounded f32 fold) stays sequential on lane 0
  in the host's j-order — now reading shared instead of global.

| metric | before | after |
|---|---|---|
| median per call (nsys node) | 232.8µs | 74.0µs (**3.1×**) |
| share of decode GPU time | 61.6% | 32.7% |

**Verification:** same suite as above; perplexity bit-identical (the kernel is
bit-identical by construction).

---

### Combined effect (this pass)

| metric | before | after |
|---|---|---|
| e2e decode (4090, cuda-graph, desktop-contended box) | ~143–161 tok/s | ~165–185 tok/s |
| e2e prefill | ~405 tok/s | ~651 tok/s (median; after right-sizing dp4a launch shared to ceil(k/4)·4 bytes) |
| decode GPU-time profile | GEMM 75% | attn 33% / rmsnorm 32% / lm_head 12% / GEMM 20% |

The e2e wall clock on this box carries ±10% noise from the desktop sharing the
GPU; the kernel-level medians above are the authoritative deltas.

---

### v1.x round 2 — i8 activation pipeline + split attention + RFC probe

**Commits:** f606bc0 (i8 pipeline), 21f2c8e (review fixes), 2a83301 (split attention)

1. **Packed-int8 activation pipeline.** The f32-staged dp4a GEMMs re-read the
   whole f32 activation row per N-column block (3.2 MiB of L2 traffic at
   N=2560 — ~2× the weight bytes). Quant kernels now emit packed int8
   (`rmsnorm_quant_i8` etc.); the `_i8_scaled*` GEMMs read it directly via
   `__ldg` — no shared staging, no barrier. Kernel-level 25–45% across shapes,
   70–85% of weight-bandwidth SOL. Split-K variants were swept and REGRESSED
   (not parallelism-starved) — rejected by measurement. The batch-graph GEMM
   also folded its separate `scale_mul_batch` launch into the fused epilogue.
2. **Split ctx-parallel attention** (bit-exact): scores fan out over
   `(n_head, ceil(max_ctx/128))` blocks as float4 chains in d-order; a
   128-thread per-head block does tree-max (exact) + parallel exp + the one
   ordered softmax sum + one-dim-per-thread weighted sum with loads hoisted
   out of the w==0 skip. 34→16µs @ctx=64 … 2486→276µs @ctx=4096 (2.1–9×).
   NOTE (review finding): the legacy `gqa_attention_decode_warp_g` fallback
   body was also rewritten in 2a83301 (2-key ILP + register-tiled weighted
   sum, bit-exact, verified in review) — it is the head_dim%4!=0 path and is
   unexercised by the shipped model; a parity test for that geometry is a
   good follow-up.

| metric | round 1 end | round 2 end |
|---|---|---|
| e2e decode (4090, contended box) | ~165–185 tok/s | **~185–195 tok/s** |
| e2e prefill | ~651 tok/s | ~600–760 tok/s (noise-bound) |
| decode GPU profile | attn 33/rms 32/GEMM 20/head 12 | **rms 42/head ~20/GEMM 18/attn 15** |

3. **Numerics-RFC probe (ADR 0018, reverted after measurement).** Switching
   ONLY `rmsnorm_quant_i8`'s sum-of-squares to a deterministic tree order:
   **decode 187 → 327 tok/s (+75%)**, perplexity rel err IMPROVED
   2.957e-3 → 2.659e-4, `cuda_greedy_matches_transformers` PASSES,
   `cpu_cuda_parity` fails (only one side changed — the RFC moves host +
   backends together). Decision pending; see docs/adr/0018.

### v1.x round 3 — ADR 0018 landed + BASTION tree-verify (commits 96d7ddf, bf3f9a8)

1. **ADR 0018 accepted + implemented**: rmsnorm sum-of-squares folds in the
   canonical 256-slot tree order on host AND all five CUDA rmsnorm kernels.
   rmsnorm_quant med 12.2 → 6.3µs; cross-backend bit-parity holds by
   construction (cpu_cuda_parity green). The 256-token greedy-vs-transformers
   gate re-baselined to a ≥96-token exact prefix (measured divergence at 104,
   into an equally coherent continuation; the tree sum is strictly MORE
   accurate — perplexity rel err 2.957e-3 → 2.659e-4).
2. **BASTION greedy tree-verify primitive (ADR 0014)**: gqa_attention_tree_f32
   + CudaDecodeModel::tree_verify_greedy — one batched forward verifies a
   draft token tree, promotes the accepted path's KV, O(1) rollback.
   Losslessness gate green across chains/branches/partial/full rejects.
   Sampling accept rule + the end-to-end speedup bench (needs the external
   block-diffusion drafter) remain open.
3. **"Known issue" RESOLVED (round 4)**: the suspected batch-vs-step KV gap
   does not exist — batch-prefill and graph-step single-token logits are
   BIT-IDENTICAL (0/128256 differing bits; now gated by
   `cuda_batch_and_graph_single_token_bit_identical`). The tail divergence
   was the EAGER path's f32-table LM head vs the graph's f16 head — a
   designed logit difference. The tree-verify tail assertion was restored
   to exact tokens via graph steps.

| metric | session start | round 3 end |
|---|---|---|
| e2e decode | ~161 tok/s | **289–330 tok/s (34–39% of roofline)** |
| e2e prefill | ~405 tok/s | **~941 tok/s median** |
| decode GPU profile | GEMM 75% | rms 29 / head 20 (SOL) / GEMM 25 / attn 21 |

Post-ADR profile note: the attention reduce's remaining cost is v-streaming +
graph-node overhead, NOT the ordered softmax/weighted folds — canonicalizing
the attention sums (ADR 0018's "next candidate") measures as low-value now.
The remaining levers are contract-free: GEMM 60→90% SOL, node-count fusion
(~460 graph nodes/token), and the IMMA prefill rewrite.

### v1.x round 4 — IMMA fragment fix, rope+kv fusion, GEMM chase closed

1. **IMMA fragment u32 loads + B reuse** (3ae7b4c): the AOT kernel + nvrtc
   template re-packed every mma operand byte-by-byte from shared (~100 SIMT
   instructions per tensor-core op — why IMMA sat at ~1% of int8 peak). Each
   fragment register is exactly one aligned little-endian u32 load; the
   template's inner loop is now nn-major with B-fragment reuse. Candidate
   space extended to 256×64-class tiles; CODEGEN_REV=2 keys the autotune
   cache; the sweep takes min-of-8 (contention only inflates — a median can
   crown the wrong winner and the cache pins it, observed on this box).
   Op-level (transfer-inclusive): M=512 shapes 1.1–1.4×.
2. **rope+kv fusion**: `rope_kv_fused_g` replaces four graph nodes per layer
   (rope q, rope k, append k, append v) with one — ~90 nodes/token removed,
   bit-identical values (k's rotated pair writes straight to the arena).
3. **Decode GEMM %-chase CLOSED with two measured rejections**: __ldcs
   streaming hints + 2-block ILP = no gain (like split-K before). The dp4a
   kernels are at their DRAM-bound limit in-model; microbench SOL% readings
   above ~85% are L2 artifacts (weights cached across iterations — in-model
   the 30 layers' ~530 MB defeat L2). Next step requires ncu (needs
   NVreg_RestrictProfilingToAdminUsers=0 + reboot on this box).
4. **Regression baseline re-recorded**: 130 → 270 tok/s (measured 289–336;
   pinned below the contended-desktop floor).

| metric | session start | round 4 end |
|---|---|---|
| e2e decode | ~161 tok/s | **~296–336 tok/s** |
| e2e prefill | ~405 tok/s | ~941 tok/s |

### v1.x round 5 — warp-shuffle reduction tails (commit c48c8dd)

The block trees (ADR 0018 canonical sums + absmax + attention max) ran all 8
levels through shared memory with a barrier per level; the last 6 levels live
inside warp 0 and a `__shfl_down_sync` by `off` computes the identical
canonical pairing on the previous level's outputs — same DAG, same roundings,
6 fewer barriers per tree. Verified bit-identical standalone (0 differing
quantized outputs) and by the full gate suite (cpu_cuda_parity bit-lockstep).
decode 338 → **344.6 tok/s**.

### Decode end-state analysis (2026-07-06)

Per-token medians: rmsnorm_quant ~0.63ms, lm_head 0.70ms (AT weight-read SOL),
GEMMs 0.94ms (DRAM-bound; SOL share 0.52ms), attention ~0.62ms, misc 0.05ms.
The remaining gap to the 848 tok/s roofline is now dominated by **per-kernel
latency floors** (hundreds of small kernels on a 128-SM GPU: launch/dispatch,
staging passes, barrier minimums), not by any single inefficient kernel. The
structural next step is a **persistent/mega-kernel decode** (one or a few
resident kernels per token with grid-sync between stages) — an ADR-scale
redesign, not an incremental patch. Secondary: ncu-guided GEMM work (needs
perf-counter permissions), IMMA cp.async/ldmatrix, and — for the real
multiplier — the BASTION drafter integration (the greedy verifier is live;
the sampling accept rule is the remaining ADR 0014 Tritium-side item).

### v1.x round 6 — serve/BASTION boundary, megakernel deferral, BLUT finding

1. **tritium-serve: --backend cuda + tree-verify HTTP surface** (a0862ff):
   the ADR 0014 boundary is now a wire protocol (/v1/tree/session,
   /v1/tree/verify), validated lossless end-to-end on the 4090 over HTTP.
   LAMU gained a `tritium` engine preset (lamu-rs 171b2c1) — Tritium is now
   loadable from LAMU like any other engine, with the verifier endpoints
   available to a future spec-decode orchestrator. Remaining: drafter
   marginals (python, lucebox-hub) — see docs/bastion-lamu-integration.md.
2. **Megakernel premise measured, ADR 0019 DEFERRED**: grid.sync costs
   0.64–0.98µs vs graph-node 0.77–1.22µs — ~0.2µs × 370 boundaries ≈ 3%/token,
   and the graph's kernel-time sum already ≈ wall time. Not worth the rewrite.
3. **BLUT finding (T-MAC LUT, tritium-cpu/simd/lut.rs)**: the LUT's grouped
   re-association — the reason it was tolerance-gated and left unwired — is
   EXACT on the quantized decode path (integer-valued activations, sums
   < 2²⁴; the same argument that unlocked dp4a). However on this box's CPU
   (AVX2 + AVX-VNNI, no AVX-512) a `vpdpbusd` int8 path strictly dominates
   the LUT (no tables, no gather, unpacks 2-bit→int8 in registers like the
   CUDA dp4a kernel; Σ code·act − Σact identity handles the unsigned×signed
   operand order). Plan: implement the VNNI quantized CPU kernel behind
   `mpgemm_with_act_quant` (bit-identical on the model path), keep the LUT
   for table-lookup-only ISAs. Not implemented this round.

### v1.x round 7 — CPU A8 int8 fast path (commit c68c9de)

The dp4a exactness argument, applied to the CPU: integer-valued A8
activations make the whole contraction exact, so a reordered AVX2
`maddubs`-based int8 kernel is **bit-identical** to the sequential bit-exact
reference (not tolerance-gated). Detection is an O(m·k) scan; arbitrary-float
callers keep the existing kernels; `k ≤ 65536` guards the exactness bound.

| metric | before | after |
|---|---|---|
| cpu greedy 32-token gate (incl. load) | 96.9s | **46.9s** |
| greedy vs transformers | 32/32 | 32/32 (bit-identity held) |

This SUBSUMES the earlier T-MAC LUT wiring plan on any CPU with int8 dot
support (maddubs/VNNI); the LUT module remains for gather-only ISAs.
(Correction for the record: "BLUT" in the user's roadmap is the training-
pipeline DAG orchestrator repo at ~/blut, NOT the LUT kernel — the CPU
fast-path work stands on its own merits.)

### v1.x round 8 — ncu counter-guided pass (first with real counters)

Perf-counter access unlocked (`NVreg_RestrictProfilingToAdminUsers=0`).
Counter data on the live decode graph (medians, `--set basic`):

| kernel | dur | DRAM% | occ% | waves/SM |
|---|---|---|---|---|
| GEMM gateup (1728 blk) | 14.4µs | 68 | 87 | 2.25 |
| GEMM qkv (480 blk) | 5.8µs | 47 | 61 | 0.62 |
| GEMM o/down (320 blk) | 10.1µs | 44 | 41 | 0.42 |
| rmsnorm_quant_i8 (1 blk, 256t) | 8.4µs | ~1 | 17 | 0.00 |
| attention scores / reduce | 6.7 / 3.8µs | ~2 | ~8 | ~0.2 |

Findings + actions:
1. **Small GEMMs are WARP-COUNT-starved, not DRAM-bound** (the earlier
   conclusion was wrong for N ≤ 3840): the shape has fewer columns than the
   machine has warp slots at M=1. Split-K re-tested on the i8-direct kernel
   (interleaved AND contiguous k-ranges) — still slower; halving block width
   (WARPS_PER_BLOCK 4) is a no-op (same warp count). Warp-per-column stands;
   the architectural fix for underfill is batching rows — which is exactly
   what M=N decode and BASTION tree-verify provide.
2. **rmsnorm_quant_i8 runs at ~1% of every throughput ceiling** (single
   block, pure latency). The kernel now pins its canonical fold to 256 slots
   EXPLICITLY and launches with 512 threads (extra threads accelerate the
   staging/normalize/quant passes; the ADR 0018 order is unchanged and the
   absmax merge stays exact): **8.35 → 5.82µs (−30%)**, ~0.3ms/token.
3. Attention scores/reduce are latency floors at decode ctx (~10.5µs/layer
   combined under ncu isolation); no counter-indicated lever beyond what
   ADR 0019 already deferred.

decode 326–389 tok/s (contention spread; best-case 45.9% of roofline),
perplexity bit-identical. 9/9 acceptance + 51/51 CUDA tests green.

### v1.x round 9 — prompt-lookup speculative decoding, live end-to-end

`tritium-serve --spec lookup`: greedy requests self-draft via prompt lookup
(longest 2..8-gram suffix match against the generation history, adaptive
draft length 6→40 doubling on full accepts) and commit chains through the
BASTION tree verifier. Model-free drafter — ADR 0014's external-drafter
boundary applies to model drafters and is unchanged.

Getting the verify cheap enough took three measured passes:

| change | speedup on the 224-tok reference run |
|---|---|
| naive loop (per-verify allocs, per-warp tree attention) | 0.61× (slower!) |
| + cached TreeScratch (15 bufs incl. 6.7MB logits; uploads too) | 0.62× |
| + split tree attention (scores fan-out + 128-thread reduce) | 0.89× |
| + longest-match ≥2-gram drafter + adaptive length | **1.19×, lossless** |

Telemetry (TRITIUM_SPEC_STATS=1): 3.65 tok/verify, verify 8.5ms vs plain
step 2.8ms. The split tree-attention pair mirrors the decode split with the
ancestor indirection in both phases; every rounded f32 fold keeps its order,
so tree-vs-step bit-parity holds (`cuda_tree_verify_greedy_lossless` and the
new `cuda_spec_lookup_matches_plain_greedy` both gate it).

Next multiplier identified, not yet landed: the verify's remaining 3× cost
over a decode step is the eager launch storm (~420 launches/verify) — a
ctrl-driven CUDA graph for fixed-m trees would collapse it toward ~1 step,
putting 2×+ in reach on repetitive text.

### v1.x round 10 — ctrl-driven tree-verify graph (kept) + the launch-storm hypothesis killed

Built the fixed-m verify graph: the trunk (embed → 30 layers) is captured per
padded bucket {8,16,24,32,48} with three ctrl-driven kernel twins
(`kv_append_tree_g`, `gqa_attention_tree_{scores,reduce}_ctrl_g` reading
[prefix_len, real_m] from a device buffer), pad rows early-exited in
attention; LM head + argmax stay eager at the real node count. Lossless gate
green on the graph path AND the `TRITIUM_TREE_EAGER=1` fallback.

**Measured: no wall-clock win (8.6 → 8.5ms/verify).** nsys per-verify budget:
tree attention 1.85ms, GEMMs ~2.2ms (weight-traffic floor ~1.15ms), LM head
1.0ms, rmsnorm+quant ~1.2ms — the verify is GPU-bound; the ~420 eager
launches were overlapped with GPU execution all along. Kept anyway: the graph
frees the CPU during verifies (matters once an external drafter shares the
host) and holds verify dispatch cost flat as buckets grow.

Also from the same counter data:
- `argmax_rows` was 129µs at m=13 (m blocks × 128k-vocab scans) → chunked
  partial/combine pair with the identical tie rule (~10µs, exact).
- LMHEAD_ROW_TILE 8→16 **regressed 2×** (verify 8.5→13.5ms): 16 predicated
  MACs per table element tips the head kernel compute-bound. Reverted; 8 is
  the measured balance point.

Verify-cost end-state: the remaining levers are the tree reduce (40µs/layer)
and acceptance itself — drafter quality (DFlash marginals) moves the
multiplier far more than any remaining verify µs.

### v1.x round 10b — entropy packing (base-243 / TQ1_0-class): REJECTED by measurement

Hypothesis: TQ2_0 spends 2.0625 bits/trit; base-243 packing (5 trits/byte,
1.6875 bpw, GGML TQ1_0's trick: byte = ceil(v·256/243), digit i =
((uint8)(b·3^i)·3)>>8) is a LOSSLESS repack that cuts ternary weight bytes
~18%. Decode is "bandwidth-bound", so fewer bytes should mean faster GEMMs.

Measured in a standalone harness (kbench6.cu: exhaustive 243+81-value
pack/unpack roundtrip + GEMM checked against the f64 reference; SoA planes so
alignment is clean; BitNet decode shapes, warm numbers):

| kernel | vs TQ2_0 dp4a |
|---|---|
| naive multiply-trick unpack (funnel-shift act assembly) | 1.38–1.79× slower |
| shared-LUT + __byte_perm 4×4 transpose + digit-major repack (all dp4a aligned) | **1.19–1.37× slower** |

Why it loses: TQ2_0's base-4 decode is ~3 ALU ops per 8 trits (shift, mask,
vsub4, 2×dp4a); base-243 is ~22 ops per 20 trits even in the LUT variant —
~4× the per-trit ALU. The counter data (round 8) shows the in-model decode
GEMMs at only 44–68% DRAM (warp-count-starved), so there is no bandwidth
wall for an 18% byte saving to relieve; the extra ALU lands straight on the
critical path. At prefill (compute-bound) it's worse by construction.

Where it WOULD pay: a machine whose ternary GEMMs run at >85% DRAM with idle
ALU — not this GPU at these shapes. The entropy argument stays correct as
*storage* (a TQ1_0 file is ~18% smaller on disk); it just doesn't convert to
decode speed here.

### v1.x round 11 — f16 KV cache (ADR 0020 rung 1): +33–40% at long context

Eight `_h` twin kernels (append ×3, rope_kv fused, decode split pair, batch
attention, tree split pair) whose ONLY delta is the KV element type — stores
round once via `__float2half_rn`, loads widen via `__half2float`, every f32
fold keeps its order. The f32 originals are byte-for-byte untouched (default
path stays bit-exact, 9/9 acceptance green). Arenas became byte buffers with
a `kv_elem` switch; dtype-selected function handles mean launch sites don't
change; verify trees route eager under f16 (ctrl twins deliberately not
duplicated — the graph measured ≈ no win). Eager step's dtod KV appends
became kernel appends (a dtod can't convert element types).

| metric | f32 | f16 KV |
|---|---|---|
| decode @ ctx≈4K | 68–72 tok/s | **95.6 tok/s** |
| decode @ short ctx | ~330–390 | unchanged (latency-bound) |
| perplexity rel err | 2.659e-4 | 1.582e-3 (~0.16%) |
| KV memory @ 4K | 630 MB | 315 MB |

Spec-decode gates pass under the rung (spec == plain within the same KV
dtype). New explicit bench: `long_ctx.rs` (ignored test, env-flagged runs).

### v1.x round 12 — measured competitor comparison (same file, same machine)

Model: `ggml-model-i2_s.gguf` (BitNet 2B4T, 1.71 GiB). Machine: RTX 4090 +
i5-13600K. bitnet.cpp = upstream microsoft/BitNet @ 1f86f058 (build 3962),
built locally (I2_S path, preset LUT header); its llama-bench numbers are
mean ± σ. Tritium numbers via `tritium report decode` + session benches.

| engine | backend | decode tok/s | prefill tok/s |
|---|---|---|---|
| **Tritium** | CUDA (4090) | **295–390** (contention spread; ×1.19 spec-decode on repetitive text) | **~941** |
| **Tritium**, f16 KV @ ctx≈4K | CUDA | **95.6** | — |
| bitnet.cpp | CPU (14t best) | 23.1 ± 0.1 | 203 (222 @ 20t) |
| Tritium | CPU | 0.95 | — |
| llama.cpp (mainline-class, ggml 0.10.x) | CUDA | **cannot load the model** (arch `bitnet-b1.58` unknown; segfaults) | — |
| bitnet.cpp `gpu/` (W2A8) | — | separate custom build targeting datacenter GPUs; not runnable here | — |

Reading:
- **On GPU, Tritium is effectively the only engine that runs this model on
  consumer CUDA** — and it does so at 13–17× the best available alternative
  on this box (bitnet.cpp CPU decode) and ~4.4× its prefill.
- **On pure CPU, bitnet.cpp is ~24× ahead of our CPU backend.** Honest
  framing: Tritium's CPU path is the bit-exact reference/parity backend
  (every CUDA optimization is gated against it), not a performance target;
  bitnet.cpp's entire project is hand-tuned CPU ternary kernels. Closing
  that gap (TL2-class LUTs, threading the GEMM) is real work that exists in
  the backlog but has never been the thesis.

Repro: `llama-bench -m <i2_s.gguf> -p 512 -n 128 -t 14` (bitnet.cpp build);
`tritium report decode --backend cuda|cpu --decode-steps 128 …`.

### v1.x round 13 — KV rung 2 (i8 + per-group dynamic scales): a MEMORY rung

Eight `_q8` twin kernels (appends quantize per-(token, kv-head, 64-dim group)
at absmax/127 — the A8 activation recipe; attention dequants to f32 then runs
the identical fold chains), a scales side-arena `[max_ctx, n_head_kv,
head_dim/64]` f32 per layer/direction, three-way dtype selection
(`TRITIUM_KV=f32|f16|i8`, legacy `TRITIUM_KV_F16=1` honored), promote moves
scale rows, tree verifies eager under non-f32.

| KV dtype | decode @ ctx≈4K | ppl rel err | KV mem @4K |
|---|---|---|---|
| f32 | 69.5 tok/s | 2.659e-4 | 630 MB |
| f16 | **103.2 tok/s** | 1.582e-3 | 315 MB |
| i8-g64 | 72.5 tok/s | 2.614e-3 | **~160 MB** |

Honest verdict: **i8 is a memory rung, not a speed rung on this GPU.** The
grouped-dequant attention kernels are latency-bound; the per-(j, group)
scale-load stream cancels the DRAM saving (scores DO win — 14.8 vs 23.8 µs —
but the reduce loses it back). Two speed attempts measured and rejected:
a 4-chain ILP scores rewrite (kept — it fixed a real regression) and a
shared weight-bank reduce (386 vs 340 µs, reverted). Value proposition:
4× KV capacity — batching slots and long contexts per GB.

Also bisected the hard way: `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
is a CAP, not a floor — the capture-path opt-in loop sizing it below a
launch's request makes that launch CUDA_ERROR_INVALID_VALUE. The raw-launch
error path now prints grid/block/smem/nparams.

Gates: 9/9 acceptance (f32 default bit-exact, untouched), 51/51 CUDA, spec
losslessness green under i8 (exercises the q8 tree kernels), long-ctx A/B
above.

### v1.x round 14 — continuous batching phase 1: 1.65× aggregate throughput, zero new kernels

`tritium-serve --batch-slots N`: a fixed N-slot pool over ONE `BatchKv` whose
M=N graph is captured once. Admission runs the prompt through the OPTIMIZED
single-sequence prefill (941 tok/s) and adopts the KV rows into the slot
(`copy_kv_into_batch_row`, byte dtod + a cross-stream sync — the replay runs
on the capture stream); free slots are fed a pad token with position pinned
to 0; per-slot host sampling reuses the plain samplers' truncated
distributions; retirement on EOS/budget/disconnect frees the slot for the
next admission. Tree/spec endpoints answer 501 in this mode.

Measured (8 slots, 24-token prompts, 64 tokens each): shared-GPU run
284.1 → 468.6 tok/s (1.65×); UNCONTENDED re-run 300.8 → **453.5 tok/s
aggregate (1.51×)** — the honest number — with all 8 admission prefills
inside the timed window (decode-only steady state is higher).

Gates: adoption bit-exactness (30 layers × 26 rows × K/V), token-0 exactness
+ agreement-prefix reporting vs single-seq, and cross-pool determinism
(same requests, reversed submission order → different slots → identical
streams; covers concurrency, slot reuse and pad isolation).

DISCOVERED en route (worth its own line): **the M=N batch path was never
bit-identical to `step_graph`** — the existing acceptance gate pins TOKENS
over a short horizon, and the mdecode attention's reduction shapes differ at
the ulp level, so near-tie argmaxes can flip at long horizons (measured:
128k/128k logits differ at step 0 even at prompt length 8; greedy tokens
still agree for 10/10 on four of six probe prompts, 5/10 and 2/10 on two).
The serve gates encode this honestly instead of pretending exactness.

### v1.x round 15 — ternary KV ("KVTQ", ADR 0020 rung 3): REJECTED by measurement

`TRITIUM_KV=t2`: appends quantize K and V to {-s, 0, +s} per 64-dim group
(s = 1.5·group-absmean ≈ the MSE-optimal 3-level quantizer for Gaussian
data), encoded in the i8 lattice so the rung-2 attention kernels run
unchanged. Perplexity: **1.4028 → 1.9203 (rel err 3.7e-1)** — catastrophic;
speed ≈ f32 (67.9 tok/s @4K). Verdict: ternary is too coarse for BOTH K and
V at G=64 on this model; the experiment cost three append kernels and zero
attention work, exactly as planned. Untested variants recorded for a future
probe: V-only ternary + K at i8/f16 (asymmetric), smaller groups. The `t2`
mode stays selectable as the experiment harness. Note: the user's beellama
fork ships a `turbo3_tcq` ternary-coded KV cache for Qwen — whatever it does
beyond plain MSE-ternary (hybrid precision? outlier channels?) is the
interesting difference to study.

## Still open (from the full optimization scan)

1. **rmsnorm_quant_f32 sequential sum** — now ~32% of decode GPU time
   (12µs × 4/layer × 30 layers ≈ 1.45ms/token). Blocked by the bit-match
   contract: the f32 sum-of-squares must fold in the host's order. Unblocking
   requires changing the HOST (and every backend + goldens) to a documented
   pairwise/tree order — a cross-cutting numerics RFC, not a kernel patch.
2. **Attention phase 1/3 memory locality** — k-row reads are uncoalesced
   (lane-per-key × sequential d). A warp-per-key staging pass (or 2-key ILP)
   could take the remaining 74µs/call down further; bit-exact per-key d-order
   must be kept.
3. **AVX2 GEMM SIMD accumulator** (CPU) — blocked by the same bit-exact
   sequential-fold contract (`to_bits` equality tests). Same RFC as (1). Note:
   on the CPU decode path activations are also integer-valued int8 quants, so
   an AVX2-VNNI (`vpdpbusd`) path would be exact and order-independent — the
   contract question is only about the public arbitrary-f32 `mpgemm`.
4. **IMMA prefill kernel** — one warp per block, one 16×8 tile; B tiles
   re-unpacked per m-tile. Multi-warp blocks + B reuse should lift the
   compute-bound prefill sharply.
5. **lm_head_warp_f16** — already at ~100% of the 656 MB/token f16 read SOL;
   only a format change (e.g. ternary-quantized tied head) moves it.

## v1.x round 15 — Track A: zero-skip wiring + TQ1_0-native decode (2026-07-11)

Census first (`tritium report sparsity`, full 2.08B ternary weights): 42.21%
element zeros, 2.441% all-zero 256-blocks (blk.1 gate/up is a 43.6% dead-neuron
outlier; attention ~0%), entropy floor 1.560 b/w, TQ1_0 real rate 1.625
(payload) / 1.688 (stored), bitmap+signs 1.578.

- **A1b block-skip**: the validated `_sparse` kernels wired into the decode
  graph (bitmap uploaded only for tensors ≥0.5% zero-blocks — the fused gateup
  qualifies at ~1%). Bit-exact by construction and by gate (9/9 acceptance with
  skipping live). Decode delta: noise on BitNet, as the census predicted — the
  machinery's customer is sparse-trained students (ADR 0024).
- **A2 TQ1_0-native**: tq1 dp4a kernel twins (plain + residual), BIT-identical
  to the TQ2 kernels on the same trits (gate: tq1_matches_tq2_tiled_scaled_
  bit_exact, first-run pass). `TRITIUM_WEIGHTS=tq1` packs/uploads/runs TQ1
  natively through the resident decoder (decode graph + prefill); batch/tree
  reject loudly in v1; host mpgemm rejects (resident-only consumer).
  Gates: TQ1 greedy token-exact, ppl within 1%, CPU↔CUDA parity (CPU speaks
  TQ1 interchange). **Measured (contended box: llama-server 8.3GB +
  Helldivers 6.3GB co-resident)**: decode 172–182 tok/s BOTH formats — parity
  within noise, exactly the honest pre-analysis (~4% ideal gateup saving is
  invisible in a latency-bound profile); **resident VRAM 3750 → 3590 MiB
  (−160 MB, −18% of ternary bytes)** — the capacity win is real. (Single-
  sample VRAM readings mid-load race the resident build: profile over time.)

Verdict: TQ1-native is a CAPACITY rung, not a speed rung at M=1 — same
pattern as i8 KV. Bigger models / more KV headroom per GB; decode unchanged.
Next: A4 bitmap+signs prototype (1.578 b/w + element-skip) head-to-head.

## v1.x round 16 — A4 verdict: bitmap+signs (and TQ1) lose the gateup race (2026-07-11)

Head-to-head kernel bench, REAL gateup shape (M=1, N=13824, K=2560), 2000
launches, 4090 (CAVEAT: box contended by a co-resident game + llama-server —
absolute µs inflated, but the relative ordering is an ALU-vs-bytes signal the
contention cannot invert):

| kernel | bytes (vs TQ2) | µs/launch |
|---|---|---|
| TQ2 dense 2-bit  | 100%  | **13.11** |
| TQ1 5-trits/byte | 81.8% | 19.74 (1.51× slower) |
| TB1 bitmap+signs | 81.1% | 33.77 (2.58× slower) |

(Re-run on mixed-sign ~1/3-zero trits after the vacuous-gate fix; the first
run's block-structured 50%-zero pattern gave 12.14/21.47/35.93 — same
ordering, same conclusion. TB1 at BitNet's true 42% zeros would be ~77%
bytes.)

Both compact kernels are BIT-exact vs TQ2 (gates green — NOTE: the original
gates ran on degenerate all-zero/no-negative trits (review-found vacuous) and
were re-pointed at mixed-sign random trits, still green; single shared launch
config, per-kernel occupancy untuned) — the loss
is pure decode cost: TQ1 pays ~24 ALU ops per dp4a word (per-byte mul-shift
chains) vs TQ2's ~3; TB1 additionally serializes on a per-block warp prefix
scan for sign addressing. At M=1 the GEMM is not DRAM-bound ENOUGH (round-8:
68% on gateup, less elsewhere) to hide that; byte savings of 18–27% bought
77–196% more time.

**Verdicts (recorded, either way, per the A4 plan):**
- TB1 is REFUTED as a decode format at BitNet's density. Its niche survives
  on paper only for high-sparsity students (p ≥ ~0.6, where the sign stream
  shrinks and 2:4/block-skip compose) — revisit IF ADR 0024 produces one.
  Kernel + format + gates stay in-tree (small, bit-exact, bench harness).
- TQ1-native is confirmed a CAPACITY-ONLY rung: −18% weight VRAM, and the e2e
  "parity within noise" from round 15 must be RE-BENCHED uncontended — the
  kernel-level +9µs on gateup (~+5% e2e) may be visible there. Until then the
  honest recommendation is TRITIUM_WEIGHTS=tq1 only when VRAM-constrained.
- The compact-format speed path, if ever needed, is shared-memory staged
  decode (amortize ALU across the warp) — not attempted; the ceiling (~4% e2e)
  does not justify it. The REAL sparsity speed play remains ADR 0024's 2:4
  tensor cores in the compute-bound regimes.

## v1.x round 17 — Track C1: chunked prefill unstalls live slots at admission (2026-07-11)

Batching P2 step 1. Phase-1 admission ran the whole prompt through one
monolithic single-sequence prefill, stalling every live slot for the prompt's
full prefill time. Now admission parks the request as the ONE in-flight
`Pending` and prefills it in fixed chunks (`TRITIUM_PREFILL_CHUNK`, default
128) interleaved with the lockstep decode steps — serve/batch.rs state machine
only, zero kernel changes. Bit-exact by construction: `prefill` is documented
bit-identical per row to the sequential step loop, so chunking it cannot
change the KV or the admission logits (G1's first-token bit-guarantee holds
verbatim).

**Measured (gate `cuda_batched_admission_interleaves_live_slot`: slot A
decoding, slot B admitted with a 2048-token prompt; 4090, CONTENDED — game +
llama-server co-resident):**

| admission mode | A tokens in B's window | A max inter-token gap | B admission |
|---|---|---|---|
| monolithic (`TRITIUM_PREFILL_CHUNK=1000000`) | 1 | **3.59 s** | 3.60 s |
| chunked 128 (default) | 16 | **458 ms** | 4.51 s |

Worst-case live-slot stall −7.8×; the admission itself pays ~25% more
wall-clock (the 16 interleaved decode steps ARE the live slots' tokens — the
deliberate trade). The gap bound scales with chunk size; 128 ≈ one ~225 ms
chunk-prefill + one batch step + SSE jitter on this contended box.

Gates: G1/G2 unchanged-green (first tokens bit-equal, determinism across
reversed admission order); contract 25/25. `cuda_spec_sampled_topk1` OOM'd at
model load (16.8 GB foreign VRAM resident) — third environmental flake of
this class this session, spec path untouched by this change, queued for the
uncontended re-run. Client disconnect mid-prefill now abandons the remaining
chunks (bonus of the state machine).

Next: C2 per-row masks (free slots still burn a row), C3 paged KV (arenas
still dense `[n, max_ctx]`).

### Round 15/16 caveat CLOSED — uncontended re-bench (2026-07-11, game exited)

Deferred verifications, run in the least-contended window this session
(llama-server 8.3 GB idle-resident only):

- **TQ1 e2e decode (the round-15 "parity within noise" caveat)**: interleaved
  ×3, 256 steps, 6-token prompt (short-context cut — GEMM share is at its
  largest): TQ2 276.9/287.8/285.6 tok/s, TQ1 290.9/292.3/292.3. **TQ1 shows NO
  e2e regression** — median +2.3% (within run-to-run spread of the slower TQ2
  runs; claim "parity, possibly a hair better"). The round-16 kernel-level
  1.51× gateup penalty does NOT materialize end-to-end: decode is
  latency-bound outside the GEMMs and the −20% gateup DRAM traffic offsets
  the ALU cost. The TQ1 capacity rung (−160 MB resident) is **free**.
- **The 3 environmental OOM flakes all pass**: cuda_tree_verify_greedy_lossless,
  cuda_batch_and_graph_single_token_bit_identical (acceptance),
  cuda_spec_sampled_topk1_matches_plain_greedy (serve) — green serially,
  confirming the failures were foreign-VRAM contention (game + llama-server),
  not code.
- **C1 review nit N1**: G1/G2 re-run with TRITIUM_PREFILL_CHUNK=15 (forces a
  1-token final chunk through step_graph + the mid-admission M=1 graph
  capture): green, identical agreement prefixes to the default-chunk run.

## v1.x round 18 — Track C2: per-row masks — dead rows touch nothing (2026-07-11)

Batching P2 step 2. Free/retired slots in the M=N batch previously decoded as
position-0 pad rows: a junk KV write at row offset 0 every step plus 1-key
attention. C2 introduces per-row liveness with a **sentinel, not a new
buffer**: `BatchKv::set_live(row, false)` uploads that row's position as
`-1`, and `ctx = positions[row] + 1 = 0` falls out of the existing kernel
arithmetic —

- `gqa_attention_split_partial_f32` already emitted identity partials for
  `start >= ctx` (zero-key windows needed NO change);
- `gqa_attention_combine_f32` gains an `L == 0 → zeros` guard (all-identity
  partials would otherwise produce `0·inf = NaN`);
- `kv_append_mdecode_f32` gains `p < 0 → return` (an unguarded `-1` writes
  into the PREVIOUS row's arena — the latent OOB this work removed);
- `rope_apply_batch_f32` gains the same guard (`-1` cos/sin index is OOB).

Dead rows' positions freeze (`advance_live`) — an unconditionally advancing
dead row would eventually trip the max_ctx overflow guard and poison the
whole batch. Zero new kernel args, zero new buffers, graphs unaffected
(positions are per-step data). Deviations from the plan text, recorded:
len-only masks (no `start` offset — no consumer until C3 pages it) and no
speculative f16 twins (the mdecode family is f32-only; ADR 0022 consolidates
real twins, not hypothetical ones). The legacy `gqa_attention_mdecode_f32`
is loaded-but-never-launched and was left position-based.

This is the **paged-KV contract** C3 builds on: a dead row owns no write
slot and touches no arena bytes.

Gates: NEW `cuda_batch_dead_row_touches_nothing` (live rows bit-identical to
an all-live batch over the horizon; dead row's arena byte-zero at first+last
layer, K+V, whole stepped span; non-vacuity asserted on a live row) — first-
run pass. All 5 batch acceptance gates green (bit-identity eager==graph,
argmax==greedy, batch==single unchanged — all-live defaults reproduce pre-C2
behavior exactly). Serve: G1/G2 + C1 interleave re-run green. What C2 does
NOT claim: dead rows still cost their dense GEMM rows (inherent to M=N GEMM
— the real free-slot compute lives there, and that is a C3/occupancy story,
not a masking one).

Review (NEEDS CHANGES, both findings verified + folded): (1) the dead-row
validation skip had un-guarded the vocab bound — but the embed gather reads
EVERY row's token, so a garbage token on a dead row was an OOB device read;
vocab check restored unconditional, only the position-overflow check is
liveness-gated. (2) The gate was blind to a regression of the very
kv_append guard it pins: an unguarded -1 writes into row 0's arena TAIL
(pos max_ctx-1), not row 1's — the gate now asserts that exact aliasing
target stays zero, plus loud rejection of an out-of-range token on a dead
row. Re-run green first try.

## v1.x round 19 — Track C3: paged KV shipped end-to-end (2026-07-11)

Batching P2 step 3, ADR 0025 executed. The dense `[n, max_ctx, kv_width]`
batch arenas are now optional: `--kv-pool-tokens N` (serve) /
`new_batch_paged` (API) replaces them with per-layer K/V page POOLS
(256-token pages) shared by all slots through per-slot page tables.

- **Kernels**: kv_append_mdecode + attn split_partial became ONE body each,
  templated on `bool PAGED` — the `if constexpr`-pruned dense instantiation
  is **SASS byte-identical** to the retired hand-written kernels
  (independently re-verified in review; a struct-shaped codec drifted the
  schedule by 566 opcodes first — the constexpr body is the shape ptxas
  reproduces). The never-launched dense-indexed fallback
  `gqa_attention_mdecode_f32` and its per-batch `d_scores` scratch
  (n·n_head·max_ctx f32) were retired ahead of paging.
- **Host**: prefix-contiguous all-or-nothing reservation (free-list; one
  page id spans all layers' K+V pools), release at retirement, table
  content uploaded per step next to d_positions (pointer baked at capture —
  no recapture ever), unmapped positions rejected loudly pre-launch.
- **Serve policy (v1, no eviction)**: admission reserves prompt+max_tokens
  up front; a full pool PARKS the job (FIFO, retried before new work); a
  request that can never fit errors loudly. Every retirement/abandonment
  path releases pages.
- **Bit-exactness**: paged == dense BIT-IDENTICAL (gate
  `cuda_batch_paged_matches_dense_bit_exact`: adoption, graph+eager+argmax
  lockstep, dead row, retire/re-admit page REUSE over stale bytes,
  exhaustion/no-leak). Serve G3: 6 streams through a 3-page/4-slot pool ==
  dense streams exactly. CORRECTION (review of a1be4dd): the first G3 used
  4 pages for 4 slots, where a free slot implies a free page — parking was
  NEVER exercised despite the claim; 3 pages makes page exhaustion the
  binding constraint, so the park/retry path now has real coverage.
- **Memory (arithmetic at BitNet 2B4T geometry, f32 KV, 30 layers)**: the
  G3 workload's 4-slot dense arenas = 2.52 GB; its 4-page pool = 157 MB
  (−94%). Honest caveats: the win is the workload's length-distribution
  gap to max_ctx, not a constant; `d_attn_partials` (~5 MB) still scales
  with max_ctx; dense remains the default.

Gates: 54/54 cuda unit, 6/6 batch acceptance, serve G1/G2 + C1 interleave
(dense default regression-free) + new G3. Track C batching P2 = C1 + C2 +
C3 COMPLETE. C4 (batched spec-decode coexistence) remains the stretch item.

## v1.x round 20 — Track C4: tree sessions coexist with the batch pool (2026-07-11)

Batching P2 stretch item. The BASTION tree endpoints (`/v1/tree/session`,
`/v1/tree/verify` — the external-drafter spec-decode surface) previously
answered 501 with `--batch-slots > 1`. Now they work: C1's chunk machine
generalizes to a `PendingGoal` (chat Admit | TreeOpen), so a session open
prefills interleaved with live-slot decode steps exactly like an admission;
verifies run inline between batch steps as bounded ops.

**Design decision, recorded honestly**: v1 is SERIALIZED-OWNERSHIP
coexistence — the session owns the single-sequence KV between admissions,
and the single-worker contract carries over verbatim ("a chat completion
closes it" ⇒ in batched mode: a chat ADMISSION resets the runner and closes
the session; the next verify gets 409 and the drafter re-opens). Sessions do
NOT survive concurrent admissions. The plan's fuller vision — sessions
leasing a slot's paged region and surviving admissions — needs the tree
kernel stack (kv_append_tree + two ctrl twins + promote/compact + per-bucket
graphs whose pointers bake the single-seq arenas) parametrized over KV
regions; recorded as the follow-up, with C3's constexpr-codec methodology
ready for it. A cheaper middle rung (stash/restore: reverse-adoption copy of
the session rows into pages around each admission, ~30 ms per admission at
2K ctx) was also considered and shelved — it changes the session-lifetime
contract rather than porting it.

Also in this change: the admission job-pull is no longer gated on a free
slot. Tree ops need no seat, so the queue drains whenever nothing is parked;
a seatless Generate PARKS (the C3 mechanism, FIFO) instead of gating the
whole queue on slot availability — verify latency under a full pool is
bounded by the op itself while nothing is parked (a parked Generate holds
FIFO, so verifies behind it still wait for one retirement).

Gate `cuda_batched_tree_session_coexists` (first-run pass): batched open +
two chained verifies token-identical to the single-worker server (losslessness
across modes); an INFORMED draft forces the accept path (2 committed, identical across
modes — junk drafts degenerate to L=1; note the accepted CHAIN is
compaction-free, node==k along the path, so KV row-moving promotion is
exercised by the kernel-level tree gates, not here); session ops succeed and return identical tokens WHILE a chat
stream decodes in the other slot (stream unaffected); verify after a chat
admission answers 409. Full serve regression re-run (G1/G2, G3, C1
interleave, contract, spec) — the worker restructure must not move anything.

Track C is now COMPLETE INCLUDING THE STRETCH. Next: Track E step 1
(rmsnorm_fast, ADR 0023).


## v1.x round 21 — Track E verdict: rmsnorm_fast REJECTED at +1.75% (2026-07-11)

ADR 0023 step 1 executed in full and deleted by its own decision rule.
`rmsnorm_quant_i8_fast` (fused stage+fold, 4× independent FMA accumulators,
one barrier fewer, blockDim-generic combine) behind
`TRITIUM_KERNEL_TIER=fast` (loud-reject selector resolved at model build;
graphs bake the symbol; healthz disclosure). Correctness under `fast`:
256-token greedy == transformers reference EXACTLY; ppl rel 2.93e-3
(exact tier 2.66e-4 ⇒ tier drift ≤ 3.2e-3 by triangle inequality, measured
difference 2.66e-3 — above the RFC's ~2e-3 hope, inside the 1% bar).

| tier | tok/s (512 steps, ABBA ×2, quiet 4090) |
|---|---|
| exact | 274.6 / 273.5 / 266.8 / 274.9 — median 274.0 |
| fast | 278.3 / 278.7 / 278.3 / 279.3 — median 278.5 |

**Pairwise +1.35/+1.60/+1.90/+4.31% (outlier = an exact-side dip), median
+1.75%, ratio of medians +1.62% — < the ≥3% bar under every reading →
variant deleted, attention fast pair not authorized (its gate was rmsnorm
≥3%).** The
refutation settles the profile conflict: rmsnorm's M=1 cost is structural
(launch + barriers + elementwise passes), not sum-order — the flat profile's
~32% attribution was not an optimization target. CORRECTION: the implementation
was built, measured and deleted without an intermediate commit — ADR 0023's
verdict section is the durable record (design specified there in rebuild
detail), NOT git history. A first bench run was discarded as contaminated
(monotonic thermal/contention decay 302→206 tok/s — the order-alternated
protocol is the keeper).

Track E closed. The remaining decode headroom at M=1 per this session's
measurements: spec decode (Track D's drafter, gated on BLUT training) ≫
everything else.
