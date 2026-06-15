# ADR 0005 — v0.30 Performance

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.30 milestone of [ADR 0002](./0002-release-roadmap.md); builds on [ADR 0004](./0004-v020-inference-spine.md), which builds on [ADR 0003](./0003-v010-implementation.md)

## Status

Planned — **not started**. v0.30 makes the working inference spine *fast* without
moving any numerics: add-only **and** IMMA int8 CUDA kernels, nvrtc JIT with an
on-disk autotune cache, AVX-512 + NEON CPU paths, and a `benches/` harness.

What must land first: **v0.20 must be tagged and green** — exact greedy token match
and perplexity parity for **BitNet b1.58 2B4T** on CPU + CUDA. The performance work
is meaningless until there is a correct, reference-matched forward pass to make
faster; this milestone explicitly trades **no** accuracy for speed.

Hard blockers: a **real GPU** (Turing+ / `mma.m16n8k32` for the IMMA path) with a
working nvrtc toolchain; an **NEON host** (aarch64) and an **AVX-512 host** for the
cross-ISA parity lane; the **BitNet b1.58 2B4T** model download already pinned in
v0.20; and recorded **llama.cpp + bitnet.cpp** baseline numbers on the same
hardware/model committed as the comparison point before any gate can be judged.

## Scope

Delivers the performance tier on the v0.20 spine: add-only **and** IMMA int8 CUDA
paths (two kernels, one truth), **nvrtc JIT** compilation with an **on-disk autotune
cache**, **AVX-512 + NEON** CPU kernel variants alongside the existing AVX2/scalar,
and a `benches/` harness for tokens/sec vs **llama.cpp / bitnet.cpp / Marlin**.
Touches `tritium-cuda` (IMMA + JIT + autotune), `tritium-cpu` (AVX-512, NEON), and a
new `benches/` crate plus a perf-regression CI job. No new ops, no numerics changes.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| IMMA path == add-only path within tolerance; both == reference | C, P | vs-reference + conformance harness (two kernels, one truth) | GPU |
| Autotuning never changes numerics beyond tolerance; tuned-from-cache reproduces same output; cold-cache == warm-cache | C, D | golden + determinism replay (cold vs warm) | GPU |
| AVX2 vs AVX-512 vs NEON vs scalar all agree | P | cross-ISA conformance vectors (vs-reference) | per-platform |
| Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant | C, E | proptest boundary suite over all kernel variants | GPU |
| tokens/sec ≥ parity with bitnet.cpp (floor `1.0×`, target `≥1.2×`) at unchanged perplexity | Pe | bench vs llama.cpp / bitnet.cpp / Marlin on same HW/model | GPU |
| No `>5%` tokens/sec drop vs recorded baseline | Pe | bench with failing regression threshold | scheduled |

## Definition of done — tag v0.30.0

- [ ] **C/P** IMMA int8 path result == add-only path result within tolerance; both == reference over the randomized + boundary suites.
- [ ] **C/D** Autotuning never changes numerics beyond tolerance; a tuned config from cache reproduces the same output; cold-cache vs warm-cache identical.
- [ ] **P** AVX2 vs AVX-512 vs NEON vs scalar all agree (cross-ISA parity).
- [ ] **C/E** Tail shapes (non-tile-multiple `M`/`N`/`K`) correct under every kernel variant.
- [ ] **Pe** tokens/sec ≥ parity with bitnet.cpp on the same hardware/model (floor `1.0×`, target `≥1.2×`) at **unchanged perplexity** (no accuracy traded for speed).
- [ ] **Pe** Perf-regression job fails on a `>5%` tokens/sec drop vs the recorded baseline (llama.cpp + bitnet.cpp committed as the comparison point).
- [ ] All of U1–U9 green on CPU + CUDA. Tag `v0.30`.
