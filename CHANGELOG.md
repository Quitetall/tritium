# Changelog

All notable changes to Tritium. Format loosely follows Keep a Changelog; this is
pre-1.0, so APIs may break between `0.x0` milestones.

## [0.20.0] — 2026-06-15 — Inference Spine

End-to-end token generation: **BitNet b1.58 2B4T** loads from its I2_S GGUF and
decodes tokens that match HF transformers, on CPU **and** CUDA (ADR 0004).

### Added
- **tritium-format** — I2_S decoder (`unpack_i2s_block`/`unpack_i2s_tensor`): ggml
  type-36, per-tensor f32 scale, `trit = code-1`, plain `[N,K]`; bit-exact vs the HF
  checkpoint on every layer-0 projection shape.
- **tritium-nn** — ops (RoPE NeoX, GQA attention, softmax, top-k/p sampling) vs torch
  goldens; W1.58**A8** int8 activation quant (Qb=127, round-half-to-even); paged KV
  cache (incremental==full); `TernaryLinear`/`Relu2Mlp`/`TransformerBlock` with the
  `attn_sub_norm`/`ffn_sub_norm` sub-LN; `ModelRunner::{load,forward,generate}` + a
  fidelity-ladder debug hook; tied LM head.
- **tritium-py** — PyO3 0.23 + maturin abi3 wheel: `Model.load/generate` (GIL released),
  `ternary_matmul`; every error → a Python exception.
- **tritium-cli** — `generate` subcommand.

### Validated
- **Forward fidelity** — vs transformers fp32: embedding bit-exact, per-op rungs ~1e-6,
  final-logit **argmax exact**.
- **Acceptance (RTX 4090)** — CUDA greedy **256/256 tokens exact**; **perplexity 2.81e-3**
  (≤1%); **CPU↔CUDA parity** bit-identical over 32 steps.
- **Python binding** — shape/dtype errors raise, GIL release proven, 6-thread no deadlock.

### Notes
- Tokenizer is Python-side (HF) for the acceptance harness; a native Rust tokenizer is
  deferred to v0.80. Big-model tests are gated (model download + GPU), not on cpu-CI.

## [0.10.0] — 2026-06-15 — Foundation

First milestone (ADR 0002 roadmap). A ternary mpGEMM runs bit-exact against the
reference on **CPU and CUDA**, end to end through the backend contract, registry,
and CLI. All v0.10 exit gates (U1–U9) closed.

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

### Gates closed for `0.10.0`
- **GPU (RTX 4090, CUDA 13.3)** — CUDA kernel vs reference and **CPU↔CUDA parity
  (U2)** ✓ (cudarc 0.19, both backends ≤1e-4); `compute-sanitizer` memcheck **0
  errors** (U7) ✓.
- **Fuzz (U5)** — GGUF parser, **550,816,129 runs / 1h, 0 crashes**, RSS flat.
- **Real GGUF (0.10.5)** — reader pinned to the official `gguf` writer's output
  (TQ2_0/TQ1_0/F16/F32 tensors + metadata), fixture committed.
- `miri` is N/A (cannot execute AVX2 intrinsics); the unsafe AVX2 kernel is covered
  by audit + reviewer sign-off + bit-exact scalar parity + `compute-sanitizer`.

[0.10.0]: https://github.com/Quitetall/tritium/releases/tag/v0.10.0
