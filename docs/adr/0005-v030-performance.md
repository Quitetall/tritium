# ADR 0005 — v0.30 Performance

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.30 milestone of [ADR 0002](./0002-release-roadmap.md); builds on [ADR 0004](./0004-v020-inference-spine.md), which builds on [ADR 0003](./0003-v010-implementation.md)

## Status

Planned — **not started**. v0.30 makes the working inference spine *fast* without
moving any numerics: add-only **and** IMMA int8 CUDA kernels, nvrtc JIT with an
on-disk autotune cache, AVX-512 + NEON CPU paths, and a `benches/` harness.

Prerequisite **met**: v0.20 is tagged (`v0.20.0`) and green — BitNet b1.58 2B4T greedy
matches transformers (256/256 exact, perplexity 2.81e-3, CPU↔CUDA bit-identical). There
is now a correct, reference-matched forward pass to make faster; this milestone trades
**no** accuracy for speed. Scope below folds the v0.30 research workflow (kernel,
codegen, CPU, bench, contract-evolution).

Hard blockers: a **real GPU** (Turing+ / `mma.m16n8k32` for the IMMA path) with a
working nvrtc toolchain; an **NEON host** (aarch64) and an **AVX-512 host** for the
cross-ISA parity lane; the **BitNet b1.58 2B4T** model download already pinned in
v0.20; and recorded **llama.cpp + bitnet.cpp** baseline numbers on the same
hardware/model committed as the comparison point before any gate can be judged.

## Scope

Performance tier on the v0.20 spine — **zero numerics change**. The designs below come
from the v0.30 research workflow.

**Contract evolution (additive, non-breaking).** Add an *optional* trait method
`mpgemm_with_act_quant` (a default-impl sibling; v0.10/v0.20 `mpgemm` and all its
conformance/parity gates stay untouched). It fuses the W1.58A8 per-token int8 absmax
quant — done caller-side in `tritium-nn` today — **on-device**, so the IMMA path takes
int8 activations directly, dropping an extra host pass + H2D round-trip.

**I2_S → GPU layouts at load.** Convert the I2_S weights once at load into the
GPU-optimal packings — TQ2_0 for the add-only kernel, an interleaved int8 layout
(`I2sInt8`) for IMMA — validated against the reference at conversion time (one-time, not
per-matmul). Adds `TernaryFormat::I2sInt8` + converters to `tritium-format`.

**CUDA — two kernels + a crossover.**
- *Add-only* (CUDA cores): tile the v0.10 one-output-per-thread kernel (shared-mem
  staging, warp reduction) — wins **memory-bound decode** (batch=1).
- *IMMA int8* (`mma.m16n8k32` tensor cores, Ada `sm_89`): int8-act × ternary-weight,
  16×8×32 tiles, double-buffered shared-mem unpack — wins **compute-bound prefill**
  (large M). Mirrors bitnet.cpp's W1.58A8 GPU kernel / BitBLAS / Marlin.
- Runtime selects by (batch, shape); crossover ~M=32, autotuned.

**Autotune + nvrtc JIT.** Template the IMMA kernel over tile (M/N/K, warps, stages);
search per (arch, shape-bucket); cache the winner on disk (`~/.cache/tritium`, keyed by
arch+dtype+shape-bucket, invalidated on CUDA/driver version). AOT default cubins for
common shapes + JIT for the long tail. Autotuning **never** changes numerics (tuned ==
reference; cold-cache == warm-cache bit-exact).

**CPU — AVX-512 + NEON + the T-MAC LUT.** Bring up the LUT-based mpGEMM deferred from
0.10 (precompute partial sums, table lookup via `vpermb`/`vpshufb` on x86, `vqtbl` on
NEON) + AVX-512/VNNI + ARM NEON, behind the existing `is_x86_feature_detected!` dispatch;
AMX/VNNI for the int8 activations. Cross-ISA bit-parity-or-tolerance vs the scalar reference.

**Bench harness** (`benches/`): divan microbenches for mpGEMM + an end-to-end tokens/sec
bench on BitNet (reuse the v0.20 runner), each coupled to an **unchanged-perplexity**
check, with a `>5%` regression gate, vs committed `llama.cpp`/`bitnet.cpp`/`Marlin`
baselines, reported as **% of roofline ceiling** (below).

Touches `tritium-cuda` (IMMA + JIT + autotune), `tritium-cpu` (AVX-512, NEON, LUT),
`tritium-format` (`I2sInt8` + converters), `tritium-spec` (the optional fused method),
`tritium-nn` (call the fused path), + a `benches/` crate + perf-regression CI.

## Workflow decomposition (preliminary)

Mostly parallel by backend, then integrate + bench — one `Workflow` per wave:
- **WF-A CUDA** — tiled add-only ‖ IMMA int8 kernel (+ cudarc/build.rs wiring). Gate: IMMA==add-only==reference; tail shapes.
- **WF-B Autotune/JIT** — nvrtc codegen + tile search + on-disk cache. Gate: tuned==reference, cold==warm bit-exact.
- **WF-C CPU** — AVX-512 ‖ NEON ‖ T-MAC LUT. Gate: all-ISA parity.
- **WF-D Contract+layout** — `mpgemm_with_act_quant` (fused A8) + I2_S→GPU converters. Gate: fused==host-A8; converted==reference.
- **WF-E Bench+regression** — divan + e2e tok/s + roofline/SOL + the regression CI lane. Gate: the perf numbers below.

(GPU+nvcc for A/B/D; AVX-512 + aarch64 hosts for C; recorded competitor baselines for E.)

## Open decisions

- AOT default cubins vs JIT-only (build time + binary size vs portability).
- Build the competitors (llama.cpp/bitnet.cpp) in CI vs commit published numbers on the pinned 4090.
- Keep the add-only kernel once IMMA lands? (Likely yes — decode is memory-bound; IMMA doesn't help batch=1.)

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| IMMA path == add-only path within tolerance; both == reference | C, P | vs-reference + conformance harness (two kernels, one truth) | GPU |
| Autotuning never changes numerics beyond tolerance; tuned-from-cache reproduces same output; cold-cache == warm-cache | C, D | golden + determinism replay (cold vs warm) | GPU |
| AVX2 vs AVX-512 vs NEON vs scalar all agree | P | cross-ISA conformance vectors (vs-reference) | per-platform |
| Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant | C, E | proptest boundary suite over all kernel variants | GPU |
| tokens/sec ≥ parity with bitnet.cpp (floor `1.0×`, target `≥1.2×`) at unchanged perplexity | Pe | bench vs llama.cpp / bitnet.cpp / Marlin on same HW/model | GPU |
| No `>5%` tokens/sec drop vs recorded baseline | Pe | bench with failing regression threshold | scheduled |
| Fused on-device A8 path == the v0.20 host-side A8 path | C, P | vs the v0.20 caller-side quant (one truth) | GPU |
| I2_S→GPU layout (TQ2_0 / `I2sInt8`) == reference dequant | C | golden vs reference at load | cpu / GPU |
| **Effective GPU utilization at the roofline** | Pe | `ncu` **% of SOL** — decode: achieved HBM BW / peak; prefill: tensor-op-active% + int8 TOPS / peak — plus tok/s vs the `bandwidth / model_bytes` ceiling (not `nvidia-smi` GPU-Util) | GPU |

## Definition of done — tag v0.30.0

- [ ] **C/P** IMMA int8 path result == add-only path result within tolerance; both == reference over the randomized + boundary suites.
- [ ] **C/D** Autotuning never changes numerics beyond tolerance; a tuned config from cache reproduces the same output; cold-cache vs warm-cache identical.
- [ ] **P** AVX2 vs AVX-512 vs NEON vs scalar all agree (cross-ISA parity).
- [ ] **C/E** Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant.
- [ ] **Pe** tokens/sec ≥ parity with bitnet.cpp on the same hardware/model (floor `1.0×`, target `≥1.2×`) at **unchanged perplexity** (no accuracy traded for speed).
- [ ] **Pe** Perf-regression job fails on a `>5%` tokens/sec drop vs the recorded baseline (llama.cpp + bitnet.cpp committed as the comparison point).
- [ ] **C/P** The fused on-device A8 path (`mpgemm_with_act_quant`) matches the v0.20 host-side A8 path.
- [ ] **C** The I2_S→GPU layout converters (TQ2_0 / `I2sInt8`) validate against the reference at load.
- [ ] **Pe** Effective-utilization gate: decode within ~10% of peak HBM bandwidth (`ncu` % of SOL), prefill tensor-op-active high; end-to-end tok/s within ~80–90% of the roofline ceiling.
- [ ] All of U1–U9 green on CPU + CUDA. Tag `v0.30`.
