# ADR 0005 — v0.3.0 Performance

- **Status:** Done — tagged **`v0.3.0`** (2026-06-16)
- **Date:** 2026-06-15 (planned) / 2026-06-16 (shipped)
- **Relates:** executes the 0.3.0 milestone of [ADR 0002](./0002-release-roadmap.md); builds on [ADR 0004](./0004-v020-inference-spine.md), which builds on [ADR 0003](./0003-v010-implementation.md)

> **Versioning note (2026-06-16):** the project switched from the `0.x0` milestone
> staircase to **SemVer** (`MAJOR.MINOR.PATCH`). The milestone formerly called
> "0.30" is **0.3.0**, tagged `v0.3.0`. Earlier tags `v0.10.0` / `v0.20.0` are
> immutable (conceptually 0.1.0 / 0.2.0). "v0.30" in older notes = this milestone.

## Status

**Done.** v0.3.0 makes the v0.20 inference spine *fast* with **zero numerics
change**, verified end-to-end. Delivered (one commit per wave on `main`):

- **WF-A tiled add-only decode kernel** (`tq2_0_add_mpgemm_tiled`) — warp-per-output
  + shared-mem-staged activations + warp reduction, f64 accumulate to stay within
  1e-4 of the sequential reference. Auto-selected for decode (M≤64, K≤8192).
- **WF-A2 IMMA int8 prefill kernel** (`tq2_0_imma_mpgemm`) — `mma.m16n8k32`
  (`s32.s8.s8.s32`) tensor cores, exact int32 accumulate, double-buffered shared
  unpack of the 2-bit weights, `compute_80` second PTX target. Fragment layout
  audited against the PTX ISA + CUTLASS.
- **WF-D fused on-device A8** — `CudaBackend::mpgemm_with_act_quant` override:
  per-token int8 absmax quant on device → IMMA → scale fold. Plus the
  `I2sInt8` `TernaryFormat` + `convert_i2s_to_int8` tile interleave matching the
  kernel byte-for-byte, and `convert_i2s_to_tq2_0` for the add-only path.
- **WF-B autotune + nvrtc JIT** — `codegen::render_imma_source(TileConfig)` +
  `compile_imma` (nvrtc), a deterministic budget-pruned tile sweep, and an on-disk
  cache keyed by arch+dtype+shape-bucket+CUDA-version. JIT == AOT **bit-identical**
  by construction (exact int32 accumulate; only launch geometry varies).
- **WF-C CPU SIMD** — AVX-512 + NEON per-element kernels (bit-exact with scalar via
  shared k-order fold) behind feature dispatch, and the ISA-agnostic T-MAC LUT
  (implemented + unit-tested, held off the hot path until its SIMD gather lands).
- **WF-E bench + roofline** — divan CPU/GPU microbenches (20 BitNet shapes), an
  end-to-end tokens/sec bench coupled to a perplexity check, the roofline ceiling
  (`peak_HBM / model_bytes` = 848.6 tok/s decode; 660.6 int8 TOPS prefill) + an
  `ncu` %-of-SOL recipe, and a `>5%` regression CI lane vs committed baselines.

Prerequisite was met (v0.20 tagged + green). The GPU lane = an **RTX 4090** (sm_89,
nvcc 13.3). The dev box is AVX2-only x86_64, so the wider-ISA kernels were validated
by **emulation**: **NEON** under `qemu-aarch64-static` (cross-compiled aarch64) and
**AVX-512** under **Intel SDE** (`sde64 -spr`) — both bit-exact vs scalar across the
full `tritium-cpu` suite (see the cross-ISA exit gate). Native SIMD *throughput* on
real AVX-512 / aarch64 silicon is still unmeasured. The competitor baseline is
committed as **published bitnet.cpp numbers** (the same-HW build is a follow-on).

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

## Definition of done — tagged `v0.3.0`

Correctness + infrastructure (**all met, verified on the RTX 4090 + independent re-run**):

- [x] **C/P** IMMA int8 path == reference within tolerance over the randomized + boundary suites; the int32 accumulate is exact, so the 1e-4 band is the reference's own f32 rounding, not a kernel defect (`imma_matches_reference_within_tolerance`, `imma_handles_tail_shapes`). Fragment layout audited against the PTX ISA + CUTLASS.
- [x] **C/D** Autotuning never changes numerics: JIT == AOT **bit-identical** for a fixed tile, every candidate validated vs reference, cold-cache == warm-cache (`jit_aot_equivalent_is_bit_identical`, `jit_wide_tile_matches_aot_bit_identical`, `tuned_config_matches_reference_and_is_stable`).
- [x] **C/E** Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant (add-only simple + tiled + IMMA).
- [x] **C/P** Fused on-device A8 (`mpgemm_with_act_quant`) == the v0.20 host-side A8 path == the spec host default (`imma_fused_equals_host_a8_and_caller_quant`).
- [x] **C** I2_S→GPU converters (`convert_i2s_to_tq2_0` / `convert_i2s_to_int8`) validate against the reference decode at conversion time; the `I2sInt8` interleave is byte-for-byte the kernel's B operand.
- [x] **C** **Zero numerics change end-to-end**: BitNet 2B4T greedy still **256/256 exact** vs transformers, perplexity **2.81e-3**, CPU↔CUDA bit-identical — re-run with the tiled decode kernel as the default and the refactored backend.
- [x] compute-sanitizer memcheck / racecheck / synccheck **0 errors** on the new GPU kernels; `cargo build`/`clippy` workspace **0 warnings**; full cpu + `--features cuda` suites green.

Partial / lane-deferred (**honestly not fully closed here**):

- [x] **P** Cross-ISA parity **validated by execution** — AVX2 == AVX-512 == NEON ==
  scalar, all bit-exact:
  - **AVX2 == scalar == LUT** natively on the dev box (AVX2 host).
  - **NEON** under `qemu-aarch64-static` (cross-compiled `aarch64-unknown-linux-gnu`):
    the full `tritium-cpu` suite **31/31**, incl. `simd::neon::tests::neon_matches_scalar`
    + the conformance set, running slow (real TCG) so the NEON kernel genuinely executes.
  - **AVX-512** under **Intel SDE** (`sde64 -spr`, Sapphire Rapids): the full suite
    **32/32** with **no skip** — `simd::avx512::tests::avx512_matches_scalar_when_available`
    runs (SDE reports `avx512f/bw/vl`) + every conformance vector dispatches through the
    AVX-512 kernel. (QEMU-user's x86 path can't do this — it doesn't expose AVX-512 via
    CPUID — so SDE is the tool; recorded for reproducibility.)

  **Caveat:** emulation validates **correctness**, not native SIMD **speed** — AVX-512 /
  NEON *throughput* on real silicon is still unmeasured.
- [~] **Pe** The bench harness, roofline ceiling (848.6 tok/s decode), `ncu` %-of-SOL recipe, and the `>5%`-drop regression CI lane are all in place; the **headline `≥1.2×` bitnet.cpp tok/s target is NOT yet hit** — the e2e pipeline is still the v0.20 correctness-first one (per-matmul H2D/D2H round-trips; the IMMA kernel is conformance-verified + microbenched but **not yet wired into the model forward**). The competitor baseline is committed as **published** bitnet.cpp numbers, not a same-HW build. A live `ncu` run is not recorded. **Tracked as the remaining e2e-performance work (follow-on 0.3.x / 0.4.0).**

`v0.3.0` ships the **verified, zero-numerics-change performance kernels + autotune +
bench/roofline infrastructure**. Turning the fast kernels into a fast *end-to-end*
forward (wire IMMA into the runner, eliminate per-matmul round-trips, persistent
device tensors) + the same-HW `≥1.2×` measurement is the next perf milestone.
