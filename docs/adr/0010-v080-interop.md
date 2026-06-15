# ADR 0010 — v0.80 Interop

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.80 milestone of [ADR 0002](./0002-release-roadmap.md); builds on the 0.70 backend-breadth milestone ([ADR 0009](./0009-v070-backend-breadth.md)); precedes the 0.90 hardening milestone ([ADR 0011](./0011-v090-hardening.md))

## Status

Planned — not started. No `tritium-onnx`, `tritium-candle`, `tritium-burn`,
`tritium-ffi`, or `tritium-serve` crate exists yet.

Must land first: **v0.70 (Backend breadth)** — every backend green on the same
conformance suite + reference-model parity. The frontends in this milestone wrap
the stabilized `tritium-spec` contract across all backends; they cannot ship an
honest "matches the native reference" gate until that reference is itself final.

Hard blockers: a **real acceptance model** (BitNet b1.58 2B4T) is needed so
candle/burn/ONNX/serve outputs can be matched against the native path —
**model-download** CI. ONNX needs **ONNX Runtime** installed in CI to load and
run the custom op / EP. serve and FFI gates need no GPU but exercise CPU+CUDA
behind the surface.

## Scope

Deliver the interop / frontend surface over the stabilized backend set:
`tritium-onnx` (custom op → then ONNX Runtime EP), `tritium-candle` and
`tritium-burn` (ternary ops exposed inside each framework), `tritium-ffi`
(C ABI + generated header), and `tritium-serve` (OpenAI-compatible HTTP server).
Each surface wraps the existing `TernaryBackend` / `tritium-nn` path; none adds
new kernels. Every surface ships with an end-to-end test matching the native
reference.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|-----------|---------|
| ONNX `TritiumMatMul` graph output equals the native path; ORT loads + runs it; unsupported-op fallback defined | C | golden + vs-reference (native path) | model-download |
| candle model + burn model each run and match the reference | C | vs-reference (per framework) | model-download |
| FFI header compiles under C **and** C++; C round-trip equals the Rust result; ABI version checked; null/invalid args ⇒ error code, never crash; calls thread-safe | F, Co | contract test (C/C++ compile + round-trip) + fuzz on null/invalid args + sanitizer | cpu-only |
| serve: OpenAI-schema contract test; streaming correctness; concurrent requests + backpressure; graceful shutdown mid-stream | C, Co, F | contract test (OpenAI schema) + concurrency stress | cpu-only |

## Definition of done — tag `v0.80.0`

- [ ] ONNX: a graph with `TritiumMatMul` produces output equal to the native path; ORT loads/runs it; unsupported-op fallback defined.
- [ ] candle/burn: a model built in each framework runs and matches the reference.
- [ ] FFI: header compiles under C and C++; a C test round-trips and equals the Rust result; ABI version checked; null/invalid args ⇒ error code (never crash); calls thread-safe.
- [ ] serve: OpenAI-schema contract test passes; streaming correctness; concurrent requests; backpressure; graceful shutdown mid-stream.
- [ ] Every interop surface has an end-to-end test matching the native reference; U1–U9 green. Tag `v0.80.0`.
