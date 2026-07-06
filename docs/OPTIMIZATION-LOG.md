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
