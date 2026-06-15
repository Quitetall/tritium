# Changelog

All notable changes to Tritium. Format loosely follows Keep a Changelog; this is
pre-1.0, so APIs may break between `0.x0` milestones.

## [0.10.0-rc1] — 2026-06-14 — Foundation

First milestone (ADR 0002 roadmap). A ternary mpGEMM runs bit-exact against the
reference on CPU, end to end through the backend contract, registry, and CLI.

### Added
- **tritium-core** — `Trit` (`{-1,0,+1}`, `repr(transparent)` i8), `DType`,
  `TernaryFormat`, `ScaleGranularity`/`absmean`, `GemmShape`, `reference_mpgemm`
  (the add/sub/skip ground truth), `TritError`. `no_std`-able, zero deps.
- **tritium-spec** — object-safe `TernaryBackend` trait (boxed `dyn DeviceBuffer`
  + `Any` downcast for runtime dispatch), `DeviceCaps`, `BackendError`.
- **tritium-format** — TQ1_0/TQ2_0 pack/unpack (faithful ggml port, golden +
  roundtrip tested), row-level wrappers (tail zero-pad), and a total, bounds-checked
  GGUF v2/v3 reader (`read_gguf`). cargo-fuzz target for the parser.
- **tritium-runtime** — `linkme` distributed-slice backend registry; a failing
  backend `init` is skipped, never fatal.
- **tritium-testkit** — `ConformanceVector` + `run_conformance<B: TernaryBackend>`
  graded against `reference_mpgemm`; JSONL persistence. Self-validated.
- **tritium-cpu** — AVX2 + scalar ternary mpGEMM, runtime-dispatched, rayon over
  rows. AVX2 reproduces the reference accumulation bit-for-bit. Conformance: zero
  failures.
- **tritium-cuda** — feature-gated CUDA backend (`--features cuda`): add-only
  `tq2_0_add.cu` kernel + `build.rs` nvcc→PTX + cudarc host side. Default build inert.
- **tritium-cli** — `tritium inspect <gguf>` and `tritium list-backends`.

### Security
- Bounded GGUF tensor/dimension preallocation against adversarial counts (a
  declared `n_dims` could otherwise drive a ~34 GB allocation and abort). Found by
  the commit-review policy; fixed with regression tests.

### Validated on GPU (RTX 4090, CUDA 13.3 / nvcc 13.3) — 2026-06-14
- CUDA kernel vs reference and **CPU↔CUDA parity (U2)** ✓ — ported to cudarc 0.19;
  both backends agree ≤1e-4 on the conformance set.
- `compute-sanitizer` memcheck: **0 errors** (U7) ✓.

### Still open before `0.10.0` final
- GGUF parser fuzz ≥1h (U5) — scheduled CI lane (`cargo-fuzz` target exists);
  `miri` is N/A (cannot execute AVX2 intrinsics).
- Real llama.cpp `.gguf` fixture load + golden-dump comparison.

[0.10.0-rc1]: https://github.com/Quitetall/tritium/releases/tag/v0.10.0-rc1
