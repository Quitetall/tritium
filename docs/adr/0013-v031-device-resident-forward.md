# ADR 0013 — v0.3.1 End-to-End Performance: the Device-Resident Forward

- **Status:** Planned
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

## Definition of done — tag `v0.3.1`

- [ ] Device-resident forward: residual stream + KV cache stay in VRAM; the non-GEMM ops run on-device; **the IMMA int8 path is wired into the runner**.
- [ ] The M=1 decode forward is captured as a CUDA graph and replayed with one launch; **eager == graph** bit-identical.
- [ ] **greedy 256/256 exact + perplexity 2.81e-3 + CPU↔CUDA parity still green** after the refactor.
- [ ] Each new device op golden vs the host f32 reference; compute-sanitizer clean.
- [ ] A **BuiltOnBox** bitnet.cpp 4090 baseline measured + committed (or the failure recorded verbatim + a defensible same-HW fallback) — no CPU figure as a GPU denominator.
- [ ] `decode tok/s ≥ 1.2×` that baseline; the absolute floor **and** the `>5%` regression both enforced; a **live `ncu`** %-of-SOL artifact recorded.
- [ ] The `perf-regression` + `gpu` CI lanes enabled on the self-hosted 4090.
- [ ] All of U1–U9 green on CPU + CUDA. Tag `v0.3.1`.

## Open decisions

- **Baseline source:** the bitnet.cpp **fork GPU** path (W2A8, A100-validated — may
  reject `sm_89`) vs **re-quantizing to TQ2_0** for mainline llama.cpp. Whichever gives
  the fairest same-HW 4090 number with perplexity pinned on both sides.
- **Attention on device:** naive masked GQA first (correctness-first, mirrors v0.20),
  or jump to a fused/flash attention now.
- **Prefill:** stay eager in v0.3.1, or add a graph-per-shape-bucket.
- **How much stays host:** the embedding gather + LM head are large `Vec<f32>` host
  loops; decide whether they move on-device in v0.3.1 or stay (they bracket the per-token
  graph, so one H2D of the input row + one D2H of the logits may be acceptable).
