# Plan 0020 — Decode Architectural Optimization

**Goal:** Push decode throughput from 110 tok/s toward the ~500-1000 tok/s range by
addressing the architectural bottlenecks, not just per-kernel tuning.

**Current state:** 110 tok/s (graph path), ~10% of realistic memory-bandwidth
peak. The gap is dominated by: (1) weight re-reads across layers, (2) kernel
launch overhead, (3) non-weight memory traffic.

**Approach:** Three parallel tracks, each independently valuable. No track
blocks another.

---

## Track A — Persistent Kernels (L2 Residency)

**Problem:** Per-layer weights are ~17.4 MB. L2 cache is 72 MB. In principle,
4 layers fit in L2. But between GEMMs, other kernels (RMSNorm, RoPE, softmax)
evict the weights, forcing re-reads from DRAM on the next GEMM.

**Solution:** Use CUDA **cooperative groups** to launch a persistent kernel that
processes all 7 GEMMs + elementwise ops for one layer in a single kernel
invocation. Weights are loaded into shared memory or L2 once and reused across
all GEMMs.

**Implementation:**

1. **Single-layer persistent kernel** — one kernel launch per layer instead of
   ~15. The kernel:
   - Loads the layer's 7 weight matrices into shared memory / registers
   - Runs rmsnorm → quant → Q GEMM → K GEMM → V GEMM → RoPE → attn → O GEMM
     → residual → rmsnorm → gate GEMM → up GEMM → relu2 → down GEMM → residual
   - All within one launch, keeping weights resident

2. **Weight tiling** — the 7 weight matrices total ~17.4 MB. Shared memory is
   max 48-164 KB per SM. Strategy:
   - Stream each weight matrix through shared memory in tiles
   - Use L2 residency hints (`cudaFuncSetAttribute` for preferred L2 cache
     size) to keep the current weight in L2 while processing
   - For the GEMM itself: stage K-tiles of activations in shared memory
     (already done), stream weight tiles from L2

3. **Cooperative launch** — `cudaLaunchCooperativeKernel` to ensure all SMs
   participate in the layer, preventing one SM from finishing early and
   evicting weights others still need.

**Expected win:** 2-4× (from reducing DRAM re-reads to L2-resident reads).
Theoretical ceiling with perfect L2 residency: ~11,000 tok/s.

**Complexity:** High. Requires rewriting the layer loop as a single CUDA kernel.
The biggest single win, but also the biggest implementation effort.

**Verification:** Greedy 256/256 must still match. Perplexity within 1%.

---

## Track B — Kernel Fusion (Reduce Launches + Memory Traffic)

**Problem:** ~400 kernel launches per token. Each launch has 5-10 µs overhead.
Each kernel reads inputs from and writes outputs to global memory.

**Solution:** Fuse adjacent kernels that share data.

### B1 — Fused RMSNorm + Quantize

**Current:** `rmsnorm_f32` → `act_quant_tiled_f32` (2 launches, 1 global
write + 1 global read between them)

**Fused:** `rmsnorm_quant_f32` — compute rmsnorm, then quantize in the same
kernel. The rmsnorm output stays in registers/shared memory.

**Savings:** 26 launches/token eliminated (one per norm). ~26 × 10 KB global
write+read saved.

### B2 — Fused GEMM + Residual Add

**Current:** `tq2_0_add_mpgemm_tiled_scaled` → `residual_add_f32` (2 launches)

**Fused:** Add an `add_residual` parameter to the GEMM epilogue. After
`out = acc * scales * act_scale`, do `out += residual[mi * n + ni]`.

**Savings:** 52 launches/token eliminated (2 residual adds per layer × 26).
~52 × 10 KB global read+write saved.

### B3 — Fused GEMM + RMSNorm

**Current:** `residual_add_f32` → `rmsnorm_f32` (2 launches, but rmsnorm
reads the residual output)

**Fused:** The GEMM epilogue writes the residual sum directly, then computes
rmsnorm over the result in the same kernel. This requires a block-wide
reduction for the rmsnorm (sum of squares), which is a second pass over the
output — but it's in shared memory, not global.

**Savings:** 52 launches/token (norms before attention + MLP).

### B4 — Fused Q/K/V GEMM (Already Done)

The Q/K/V projections are already fused into a single GEMM with output
`[q_width + 2·kv_width]`. ✓

**Total B savings:** ~130 launches eliminated, ~2-3 ms saved.

**Complexity:** Medium. Each fusion is a new kernel variant. B1 is easiest,
B3 is hardest (requires cooperative reduction in the epilogue).

---

## Track C — Batched Decode (M>1)

**Problem:** Single-token decode (M=1) is memory-bound. Weight reads dominate.
Tensor cores sit idle (they need M≥8-16 for efficiency).

**Solution:** Process multiple independent decode requests in parallel (M>1).

### C1 — Multi-Request Batching

Already partially implemented: `decode_batch` and `decode_batch_graph` handle
M>1. But the current batched path uses the f64 tiled GEMM (slower) and
doesn't fuse scale_mul.

**Fix:** Switch batched path to f32 fused-scaled GEMM (already done in #15b).
Verify the batched graph path is competitive.

### C2 — Speculative Decoding (M>1 from a Single Request)

ADR 0014 already defines this. A block-diffusion drafter produces B candidate
tokens; Tritium verifies all B in one M=B forward pass. This amortizes weight
reads across B tokens.

**Implementation:** Requires the tree-masked verify attention + shared-prefix
KV from ADR 0014. This is the **highest-impact single optimization** — up to
6.6× reported.

**Sequencing:** After v0.60 (needs the drafter, which is external).

### C3 — Tensor Core Decode

For M≥8, the IMMA int8 tensor-core kernel becomes efficient. Currently only
used for prefill.

**Idea:** Batch 8 independent decode requests and run them through the IMMA
kernel instead of the TQ2_0 tiled kernel. The IMMA kernel is ~4× faster per
element at M≥8.

**Requires:** A batching layer that groups requests by position (so they can
share the same kernel launch). This is a scheduler concern, not a kernel
concern.

---

## Track D — Weight Compression (Diminishing Returns)

**Current:** TQ2_0 at ~1.58 bits/trit. Already near the information-theoretic
minimum for ternary.

**Possible:**
- **Block-level pruning:** zero out entire 256-trit blocks that contribute
  little (sparse TQ2_0). Saves bandwidth but requires sparse GEMM.
- **Mixed precision:** use TQ1_0 (1-bit) for less sensitive layers. Requires
  per-layer format selection.

**Expected win:** 10-20% bandwidth reduction at best. Not worth the complexity
until A-C are exhausted.

---

## Recommended Sequencing

| Phase | Track | What | Expected Win | Effort |
|-------|-------|------|-------------|--------|
| **Phase 1** | B1 | Fused RMSNorm + Quantize | ~0.5 ms/token | Low |
| **Phase 2** | B2 | Fused GEMM + Residual | ~1 ms/token | Medium |
| **Phase 3** | A | Persistent layer kernel | ~3-5 ms/token | High |
| **Phase 4** | C2 | Speculative decode (ADR 0014) | ~2-4× end-to-end | High (external dep) |
| **Phase 5** | B3 | Fused GEMM + RMSNorm | ~1 ms/token | High |
| **Phase 6** | C3 | Tensor core decode batching | ~2× for batched | Medium |

Phases 1-2 are low-hanging fruit. Phase 3 is the biggest single win but
requires a kernel rewrite. Phase 4 needs an external drafter.

---

## Measuring Progress

Each phase must pass these gates:

1. **Correctness:** Greedy 256/256 token match (bit-exact for f64 path,
   within 1e-4 for f32 path)
2. **Performance:** `step()` latency measured with `Instant::now()` on a
   warm cache (100 iterations, take median)
3. **Conformance:** `mpgemm_device_bit_matches_host_path` test passes
4. **Perplexity:** ≤1% degradation vs baseline (for f32 path changes)

**Target:** 500 tok/s by end of Phase 3. 1000+ tok/s with speculative decode.
