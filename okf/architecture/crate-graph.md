---
type: Reference
title: Crate graph
description: Every Tritium crate by layer, with current vs planned status.
tags: [architecture, crates, workspace]
timestamp: 2026-06-14T00:00:00Z
---

# Crate graph

Cargo workspace. No CMake — `build.rs` + cargo features gate every backend.

Status as of **v0.10.0-rc1**. See [release roadmap](/concepts/release-roadmap.md).

## L0 — foundation (pure, no_std-able)
- `tritium-core` — types, dtypes, scaling, reference math. **Landed (v0.10).** See [tritium-core](/crates/tritium-core.md).
- `tritium-format` — TQ1_0/TQ2_0 packing + GGUF reader + row wrappers. **Landed (v0.10).**
- `tritium-spec` — object-safe backend trait, no impls. **Landed (v0.10).**
- `tritium-testkit` — conformance vectors + harness. **Landed (v0.10).**

## L1 — backends (each implements `tritium-spec`)
- `tritium-cpu` — AVX2 + scalar ternary mpGEMM, runtime dispatch, rayon. **Landed (v0.10).**
- `tritium-cuda` — cudarc host + add-only `.cu` kernel (feature-gated). **Landed (v0.10), GPU path unverified.**
- `tritium-metal`, `tritium-rocm`, `tritium-wgpu` — *planned (0.70).*

## L2 — runtime + functional
- `tritium-runtime` — linkme registry + dispatch. **Landed (v0.10).**
- `tritium-quantize` — fp→ternary; owns [SALT](/concepts/salt-quantization.md). *Planned (0.40).*
- `tritium-nn` — mixed-precision ops + model runner. *Planned (0.20).*
- `tritium-train` — QAT, STE autograd, optimizer. *Planned (0.50+).*

## L3 — frontends / bindings
- `tritium-ffi` (C ABI), `tritium-py` (PyO3+maturin), `tritium-onnx` (ort custom op),
  `tritium-candle`, `tritium-burn`, `tritium-wasm`. *Planned (0.80).*

## L4 — apps
- `tritium-cli` — `tritium` binary (`inspect`, `list-backends`). **Landed (v0.10).**
- `tritium-serve` — OpenAI-compatible server. *Planned (0.80).*
