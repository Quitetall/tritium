# Tritium

Foundational library for **ternary-model inference and training**, GPU + CPU
first-class. Ternary weights live in `{-1, 0, +1}` (~1.58 bits/weight) — matmul
collapses to add / subtract / skip, no multiplies. Tritium makes that fast
everywhere: CUDA (addition-only **and** int8 tensor-core paths), CPU SIMD with
LUT lookup (AVX2 / AVX-512 / NEON), with ONNX and PyTorch bindings.

Consolidates prior art — BitNet/bitnet.cpp, T-MAC, BitBLAS, llama.cpp's
TQ1_0/TQ2_0, GPTQ-Marlin, ExLlamaV2 — behind one Rust workspace.

> Status: **pre-alpha**, scaffolding. `tritium-core` is the first real crate.

## Architecture

Hexagonal / ports-and-adapters. Dependencies point **inward** only:

```
foundation  (core · format · spec · testkit)
   ↑
backends    (cpu · cuda · metal · rocm · wgpu)   — each impls tritium-spec
   ↑
runtime     (runtime · quantize · nn · train)
   ↑
frontends   (ffi · py · onnx · candle · burn · wasm · cli · serve)
```

A frontend never depends on a concrete backend. A backend never depends on
another backend. Every backend passes the **same** conformance vectors from
`tritium-testkit`, so cross-backend bit-exactness is structural.

## Build

Cargo only — no CMake. CUDA crates shell `nvcc` from `build.rs` and load
PTX/cubin at runtime via `cudarc`; feature flags gate every backend.

```sh
cargo build                     # cpu-only foundation
cargo test  -p tritium-core     # reference math + roundtrip
```

## License

`MIT OR Apache-2.0`. Vendored upstreams attributed in [NOTICE](./NOTICE).
