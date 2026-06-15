# ADR 0003 — v0.10 Foundation: implementation record

- **Status:** Accepted (v0.10.0 — all U1–U9 gates closed, GPU-validated)
- **Date:** 2026-06-14
- **Relates:** executes the 0.10 milestone of [ADR 0002](./0002-release-roadmap.md)

## Context

The v0.10 plan (per-crate specs, dependency waves, validation matrix) was produced
by a 5-agent research workflow, reviewed/corrected, and approved. This ADR records
what actually shipped and which gates remain open, so the audit trail lives in the
repo rather than an ephemeral plan file.

## What shipped

Eight crates ship in rc1: `tritium-core` (foundation, built first inline) plus
seven built in dependency waves (depth-first; backends parallel):

- **Wave A (inline):** `tritium-spec` (object-safe `TernaryBackend`), `tritium-format`
  (TQ1_0/TQ2_0 pack/unpack, ggml port).
- **Wave B (workflow, 2∥→1):** `tritium-format` extended with the GGUF reader + row
  wrappers, `tritium-runtime` (linkme registry), `tritium-testkit` (conformance harness).
- **Wave C (workflow, 2∥→1):** `tritium-cpu` (AVX2+scalar), `tritium-cuda`
  (feature-gated kernel), `tritium-cli` (inspect, list-backends).

Key decisions realized (some refined from the plan):
- Trait is **object-safe** with a boxed `dyn DeviceBuffer` (+`Any`), not an
  associated `Buffer` type — required for the `Box<dyn TernaryBackend>` registry.
- Packing lives in `tritium-format`; the CPU backend unpacks via it. `linkme` over
  `inventory` for registration (crates `deny(unsafe_code)` + a scoped allow on the
  `distributed_slice` static, since it emits a custom `link_section`).
- The AVX2 kernel folds SIMD-decoded add/sub/skip contributions **sequentially in
  f32** to match `reference_mpgemm` bit-for-bit — a reordered/f64 reduction is more
  accurate but drifts ~1e-4 from the reference at K=512 (past the conformance floor).
- CUDA is entirely behind `--features cuda`; the default build needs no toolkit.

## Validation status (vs ADR 0002 taxonomy)

Green locally (cpu-only, `clippy -D warnings`, fmt, ~95 tests + doctests):
- **C/E** pack roundtrip, golden TQ1/TQ2 bytes, row tail-padding, GGUF parse.
- **C** CPU mpGEMM vs reference — conformance harness, zero failures.
- **E/F/S** GGUF total parser: truncated/bad-magic/overflow → typed errors; the
  adversarial-prealloc DoS is fixed + regression-tested.
- **D** determinism; **Do** docs + doctests + `cli inspect`/`list-backends`.
- End-to-end: `list-backends` discovers the linkme-registered CPU backend.

Closed for `v0.10.0` (validated on an RTX 4090 + CUDA 13.3):
- **U2 / CUDA** ✓: CUDA kernel vs reference and CPU↔CUDA parity (cudarc 0.19, ≤1e-4).
- **U5** ✓: GGUF fuzz — 550,816,129 runs / 1h, 0 crashes, RSS flat.
- **U7** ✓: `compute-sanitizer` memcheck 0 errors; `miri` N/A for AVX2 intrinsics, so
  the unsafe kernel rests on audit + reviewer sign-off + bit-exact scalar parity.
- Real GGUF ✓: reader pinned to the official `gguf` writer's output (`tests/real_gguf.rs`).

## Consequence

`v0.10.0` is tagged. Both the CPU and CUDA paths are real, conformant, reviewed, and
GPU-validated; every U1–U9 gate is green. The foundation is ready for v0.20.
