# ADR 0003 — v0.10 Foundation: implementation record

- **Status:** Accepted (rc1)
- **Date:** 2026-06-14
- **Relates:** executes the 0.10 milestone of [ADR 0002](./0002-release-roadmap.md)

## Context

The v0.10 plan (per-crate specs, dependency waves, validation matrix) was produced
by a 5-agent research workflow, reviewed/corrected, and approved. This ADR records
what actually shipped and which gates remain open, so the audit trail lives in the
repo rather than an ephemeral plan file.

## What shipped

Seven crates, built in dependency waves (depth-first; backends parallel):

- **Wave A (inline):** `tritium-spec` (object-safe `TernaryBackend`), `tritium-format`
  (TQ1_0/TQ2_0 ggml port + GGUF reader).
- **Wave B (workflow, 2∥→1):** `tritium-format` GGUF+rows, `tritium-runtime`
  (linkme registry), `tritium-testkit` (conformance harness).
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

Open — blocking `v0.10.0` final, tracked on dedicated CI lanes (`.github/workflows/ci.yml`):
- **U2 / CUDA**: kernel vs reference and CPU↔CUDA parity — need a GPU + nvcc.
- **U5**: GGUF fuzz ≥1h (target exists; `cargo-fuzz` not run here).
- **U7**: `compute-sanitizer` (GPU); `miri` cannot execute AVX2 intrinsics, so the
  unsafe kernel is covered by manual audit + reviewer sign-off + scalar bit-parity.
- Real llama.cpp `.gguf` fixture + golden-dump comparison.

## Consequence

`v0.10.0-rc1` is tagged: the CPU path is real, conformant, and reviewed; the GPU
path compiles and is written against cudarc 0.13 but is unverified until a GPU lane
runs. Promote to `v0.10.0` once the open gates pass.
