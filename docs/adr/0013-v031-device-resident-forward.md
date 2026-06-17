# ADR 0013 — v0.3.1 End-to-End Performance: the Device-Resident Forward

- **Status:** Done (v0.3.1) — device-resident decode shipped + numerics-gated; the
  CUDA graph (W2) and the `≥1.2×` competitor gate are **deferred to v0.3.2** (see the
  Outcome below).
- **Date:** 2026-06-16
- **Relates:** closes the slipped **Pe** (performance) exit gate of
  [ADR 0005](./0005-v030-performance.md) (v0.3.0); precedes
  [ADR 0006](./0006-v040-quantization.md) (v0.40 SALT), which needs a fast forward to
  validate accuracy-vs-bpw at scale.

> **Roadmap note.** v0.3.1 is a **performance point-release** inserted between v0.3.0
> and v0.4.0. It adds **no new numerics and no new API** — it makes the v0.3.0 forward
> fast *end-to-end*, closing the `≥1.2×` bitnet.cpp gate ADR 0005 deferred. v0.4.0
> (SALT, ADR 0006) is unchanged; SALT then rides this fast forward. ADR numbers are
> creation-order, so this is 0013 even though 0006–0012 already map to 0.40–1.0.

## Context

v0.3.0 shipped fast CUDA kernels (tiled decode, IMMA int8 prefill, fused on-device
A8), autotune, and a bench/roofline harness — all correctness-verified, zero numerics
change. **But the model forward never got faster**: it is still the v0.20
host-orchestrated pipeline. A 3-agent research pass (2026-06-16) pinned why, with
file:line evidence:

- **Residual stream is host-resident.** `forward_inner` keeps `hidden`/`next` as host
  `Vec<f32>` for all 30 layers (`runner.rs:148/164/195`); each `TransformerBlock` takes
  `x: &[f32]` and writes `out: &mut [f32]`.
- **Every ternary linear round-trips.** `TernaryLinear::forward` host-quantizes
  (`linear.rs:153`) then calls `backend.mpgemm` (`linear.rs:157`); `mpgemm_kernel` does
  `clone_htod(act)` + `clone_htod(scales)` + `alloc_zeros(out)` + `launch` +
  `memcpy_dtoh(out)` **per call** (`cuda.rs:331-417`). **7 GEMMs/layer × 30 = ~210
  synchronous H2D→kernel→D2H round-trips per token.**
- **All non-GEMM ops run on the host CPU in f32** — rmsnorm, RoPE, GQA attention,
  softmax, both residual adds, the squared-ReLU gate, the embedding gather, the tied
  LM head — so the activations never stay resident. The KV cache is host `Vec<f32>`.
- **The v0.3.0 fused IMMA int8 path is UNWIRED from the forward.** `grep
  mpgemm_with_act_quant crates/tritium-nn/src` returns nothing — nn only calls
  `backend.mpgemm`, so the per-token path runs only the **add-only TQ2_0** decode
  kernel. The IMMA tensor-core kernel + on-device act-quant (`imma_with_act_quant`,
  `cuda.rs:435`) are dead code from the model's perspective.

Result: **~2.4 s/token (~0.4 tok/s), ~2000× off the 848.6 tok/s memory roofline.** The
bottleneck is launch / round-trip overhead, not the kernels. ADR 0005's Pe gate
(`≥1.2×` bitnet.cpp at unchanged perplexity + a live `ncu` run) is `[~]` partial;
v0.3.1 closes it.

## Decision

Make the per-token forward **device-resident**: the residual stream stays in VRAM
across all 30 layers, intermediate ops run on-device, the IMMA int8 path is wired in,
and the M=1 decode forward is captured as a **CUDA graph** replayed with one launch —
**without changing any numerics** (relocate the math, do not rewrite it; the greedy
256/256 + perplexity 2.81e-3 acceptance is the gate).

### A. Persistent device tensors + wire the fused path
- Hold the residual stream, KV cache, and per-layer scratch as long-lived
  `CudaSlice` fields — `alloc_zeros` once, reuse across launches (`core.rs:1559`);
  device-to-device via `memcpy_dtod` (`core.rs:1657`).
- **Wire the forward to the fused path**: convert weights to `I2sInt8` at load and call
  `mpgemm_with_act_quant` (the IMMA int8 kernel) instead of host-quant + add-only
  `mpgemm`. This activates the v0.3.0 tensor-core path that is currently dead code.
- **Move the non-GEMM ops on-device**: rmsnorm, RoPE, the v0.20 naive masked GQA
  attention, softmax, residual adds, squared-ReLU, the final norm + tied LM head — as
  small CUDA kernels (fused where cheap), each **numerically identical** to the host
  f32 op it replaces.

### B. CUDA graph capture (decode)
cudarc 0.19.7 already exposes this (no FFI, no version bump): `CudaGraph` with
`begin_capture`/`end_capture`/`launch`/`upload` (`safe/graph.rs`).
- Record the **M=1 decode** forward **once** on a **dedicated `new_stream`** — NOT the
  default stream (the NULL stream is not capturable: `cuStreamBeginCapture` rejects it;
  `core.rs:654` vs `core.rs:674`). Replay with one `cuGraphLaunch` per token: write only
  the new token's embedding into the resident input slice, `graph.launch()`, read back
  only the final logits — replacing all 210 round-trips with one.
- **Pre-allocate every buffer before capture**; keep all syncs / readbacks / autotune
  timing **outside** the capture region (`end_capture` fails if the stream did anything
  non-capturable); consider `disable_event_tracking` (`core.rs:493`).
- **Spike first**: a trivial two-kernel capture asserting `graph.launch` output ==
  eager output, before wiring the full forward (no in-tree precedent).
- **Prefill** (variable M) stays **eager** in v0.3.1 (a graph-per-shape-bucket is a
  later option); decode (M=1, shape-static) is the hot path that captures cleanly.

### C. Numerics preservation (the hard constraint)
- The math **does not change** — ops move from host f32 to device f32 (or the
  already-verified IMMA int8 path), bit-for-bit where a kernel reproduces the host
  reduction order, else within the documented per-op tolerance (ADR 0002/0004).
- **Gate:** the v0.20 acceptance — greedy **256/256 exact**, perplexity **≤1%
  (2.81e-3)**, CPU↔CUDA parity — must **still pass** after the refactor. Per-op golden
  tests vs the host f32 op for every new device kernel.

### D. Make `≥1.2×` real (the measurement was a fiction)
Research found the committed baseline is meaningless: `BITNET_CPP_2B4T_DECODE = 28.0`
and `LLAMA_CPP_2B4T_DECODE = 18.0` (`benches/src/lib.rs:273/303`) are **CPU**,
`Published` (admitted extrapolations, not measured), used as the denominator for a
**CUDA/4090 decode** gate — and **no published bitnet.cpp 4090 decode tok/s exists**
(the official GPU kernel is W2A8, validated only on A100; the 2B4T report gives only
CPU ~34.5 tok/s). So `≥1.2×` against a 28-tok/s CPU floor proves nothing.
- **BuiltOnBox baseline:** measure bitnet.cpp on the **same 4090 + same 2B4T weights**.
  Mainline llama.cpp **cannot load** this repo's I2_S GGUF (type-id 36 collision with
  the removed `IQ4_NL_4_4`; `benches/src/lib.rs:289-299`), so either build the
  **bitnet.cpp fork GPU path** (`conda` + `bitnet_kernels/compile.sh`; W2A8; may reject
  `sm_89`) or **re-quantize to TQ2_0** for mainline. Record the number **and** its
  command in `Baseline.source`. If the fork won't run on the 4090, **record that
  verbatim** and fall back to the strongest defensible same-HW number — **never** a CPU
  figure as a GPU denominator. (W2A8 ≠ W1.58A8: pin perplexity on both sides.)
- **Make the gate bite:** add an **absolute floor** assert (`decode_tps ≥ 1.2 ×
  baseline_4090`) distinct from the `>5%` relative-drop check; retarget
  `REGRESSION_DROP_THRESHOLD`'s denominator to the BuiltOnBox 4090 number.
- **Roofline proof:** a **recorded live `ncu` run** on the pinned 4090, committed as an
  artifact — decode `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` ≥ ~80–90%
  of the 848.6 ceiling; prefill `sm__pipe_tensor_op_imma.avg.pct_of_peak_sustained_active`
  high + achieved int8 TOPS vs 660.6 (metrics verified vs Nsight Compute docs;
  `benches/README.md:68-105`). `ncu` is a **separate** pass, never folded into the timed
  tok/s bench (it serializes kernels and inflates wall-clock).
- **Enable CI:** flip the `perf-regression` + `gpu` lanes off `if: false`
  (`.github/workflows/ci.yml`) on the self-hosted 4090 once the model + bitnet.cpp
  baseline are pinned at the cache path.

## Validation (exit gates / testing blockers)

| Gate | Tag | How tested |
|------|-----|------------|
| Numerics unchanged end-to-end | C, Pe | the v0.20 acceptance re-run: greedy **256/256 exact** + perplexity **2.81e-3** + CPU↔CUDA parity, after the device-resident refactor |
| Each new device op == the host f32 op | C | per-op golden (rmsnorm / RoPE / GQA attention / softmax / residual / squared-ReLU / LM head) vs the host reference |
| CUDA-graph replay == eager forward | C, D | capture-vs-eager golden; first-capture vs replay bit-identical; replay is deterministic |
| Round-trips eliminated in decode | E | assert ≤ a small constant `dtoh`/token (only the final logits), measured from the instrumented backend |
| The IMMA int8 path is actually exercised | C | assert the fused kernel is launched in the forward (not the add-only fallback) |
| `decode tok/s ≥ 1.2 × BuiltOnBox bitnet.cpp 4090` at unchanged perplexity | Pe | e2e bench: **absolute floor** assert + the `>5%` regression, denominator = the measured 4090 baseline |
| Effective utilization at the roofline | Pe | recorded **live `ncu`** %-of-SOL artifact: decode ≥ ~80–90% of peak HBM; prefill tensor-op-active high |
| No memory errors / races in the new kernels + graph | C | compute-sanitizer memcheck + racecheck |

## Outcome — what v0.3.1 actually shipped

- [x] **Device-resident decode forward** (`CudaDecodeModel`): the residual stream + the
  per-layer KV cache stay in one set of `CudaSlice`s across all 30 layers; **every** op
  runs on-device (embedding gather, rmsnorm, q/k/v/o + gate/up/down GEMMs, RoPE, GQA
  attention, ReLU² gate, sub-norms, residuals, tied LM head). A decode step crosses the
  host boundary **once** (logits D2H) — down from ~210 synchronous round-trips/token.
  The runner downcasts its `dyn TernaryBackend` to `CudaBackend` (defaulted `as_any`
  hook) and drives one `step` per token; the host path stays the golden oracle.
  - Decode keeps the **tiled add-only f64** GEMM (the reference-matching kernel), **not**
    IMMA. The v0.30 finding stands: IMMA accelerates *compute*, but M=1 decode is
    memory/launch-bound, so IMMA gives zero decode speedup and its f32 scale-fold broke
    greedy at token 75. IMMA stays a **prefill** optimization (deferred, #28).
- [x] **greedy 256/256 exact vs transformers + perplexity 2.96e-3 (≤1%) + CPU↔CUDA
  parity** (identical IDs / 32 lockstep steps, worst logit rel 6.3e-7) — all green on the
  device-resident path. Bit-match was *achieved* (not the perplexity fallback): the
  softmax exp is computed in f64 then rounded to f32 (`exp_f32`), matching glibc `expf`
  closely enough that the greedy argmax never flips.
- [x] Each device op has a **golden** vs its host f32 reference (bit-exact except the
  softmax `expf`, ≤2 ULP). 37 cuda tests green.
- [x] **BuiltOnBox decode baseline** committed: Tritium **~27.6 tok/s** on the 4090
  (3.3% of the 848.6 tok/s roofline), ~6× the v0.20 host path. The e2e gate keys on this
  (conservative 25.0 floor) + the `>5%` regression.
- [ ] **`≥1.2×` competitor gate — deferred.** No same-HW GPU *ternary* baseline is
  obtainable: llama.cpp's CUDA backend has no TQ/I2_S mul-mat kernel (ternary GEMM is
  CPU-only) and cannot load the I2_S artifact (type-id 36 = removed `IQ4_NL_4_4`); the HF
  weights to re-quantize are absent. bitnet.cpp's numbers are CPU. So there is no GPU
  ternary engine to race; the gate awaits a measurable competitor or the v0.3.2 graph.
- [x] **CUDA graph (W2) — shipped in v0.3.2** (via the raw-FFI capture path cudarc 0.19's
  safe launch couldn't do). But it was the WRONG lever: collapsing ~930 launches/token into
  one replay gave **no speedup** (26.6 vs 27.6 tok/s) — proving launches were never the
  wall. The graph is the experiment that found the real bottleneck (see v0.3.2 below).
- [ ] A live `ncu` %-of-SOL artifact + the self-hosted GPU CI lanes — follow-on (no
  `ncu`/`nsys` on the box; the graph-vs-eager equivalence served as the bottleneck probe).

## v0.3.2 — what the graph actually unlocked

The CUDA graph itself gave no speedup, but building it pinpointed the real decode
bottleneck on the (perf) path:
- **f32-accumulate GEMM** — the tiled kernel accumulates the K contraction in **double**
  for the 1e-4 conformance bar; the 4090 runs f64 at **1/64 the f32 rate** and decode does
  ~210 GEMMs/token, so the double reduction was dominant. The graph path swaps in an
  f32-accumulate variant (the eager `mpgemm` keeps the double kernel for conformance).
- **Coalesced warp-per-row LM head** — the by-thread head read the 1.3 GB `token_embd`
  fully uncoalesced; a warp-per-row layout coalesces it.
- Net **~1.66× decode (27.6 → 45.9 tok/s)** with **zero numerics regression** — greedy
  256/256, perplexity 2.96e-3, parity identical all hold, because the real activations
  never hit the cancellation-heavy worst case the f64 guarded.

## Deferred (→ v0.3.3)

- More decode headroom (we are at ~5.4% of the memory roofline): parallelize the remaining
  sequential single-thread bit-match kernels (rmsnorm's thread-0 sum, the one-thread-per-
  head GQA attention), and an f16 `token_embd` for the LM head read.
- The IMMA **prefill** path (#28); a batched device prefill (today's prefill is sequential
  per-token decode); a live `ncu` artifact + GPU CI lanes; the `≥1.2×` competitor gate
  (still no GPU ternary competitor — see `benches/src/lib.rs`).
