---
type: Reference
title: Crate graph
description: Every Tritium crate by layer, with current vs planned status.
tags: [architecture, crates, workspace]
timestamp: 2026-06-14T00:00:00Z
---

# Crate graph

Cargo workspace. No CMake — `build.rs` + cargo features gate every backend.

## L0 — foundation (pure, no_std-able)
- `tritium-core` — types, dtypes, scaling, reference math. **Exists.** See [tritium-core](/crates/tritium-core.md).
- `tritium-format` — TQ1_0/TQ2_0 packing, GGUF + safetensors I/O. *Planned next.*
- `tritium-spec` — backend trait contract, no impls. *Planned.*
- `tritium-testkit` — reference vectors + conformance harness. *Planned.*

## L1 — backends (each implements `tritium-spec`)
- `tritium-cpu` — `std::arch` SIMD: x86 (AVX2/AVX-512/AMX), ARM (NEON/SVE). *Planned.*
- `tritium-cuda` — cudarc host + `.cu` kernels (nvcc in build.rs) + nvrtc JIT. *Planned.*
- `tritium-metal`, `tritium-rocm`, `tritium-wgpu` — *planned.*

## L2 — runtime + functional
- `tritium-runtime` — registry, device discovery, dispatch, mem pools, autotune cache. *Planned.*
- `tritium-quantize` — fp→ternary; owns [SALT](/concepts/salt-quantization.md). *Planned.*
- `tritium-nn` — mixed-precision ops + model runner. *Planned.*
- `tritium-train` — QAT, STE autograd, optimizer. *Planned.*

## L3 — frontends / bindings
- `tritium-ffi` (C ABI), `tritium-py` (PyO3+maturin), `tritium-onnx` (ort custom op),
  `tritium-candle`, `tritium-burn`, `tritium-wasm`. *Planned.*

## L4 — apps
- `tritium-cli` — `tritium` binary. *Planned.*
- `tritium-serve` — OpenAI-compatible server. *Planned.*
