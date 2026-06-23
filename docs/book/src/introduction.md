# Introduction

**Tritium** is a from-scratch library for **ternary-model inference and
training**, GPU- and CPU-first. It is written in Rust as a single Cargo
workspace (no CMake): the CUDA backend shells `nvcc` from `build.rs` and loads
PTX/cubin at runtime via `cudarc`, and every backend is gated behind a feature
flag so the default build is CPU-only.

## What "ternary" means here

Ternary weights live in `{-1, 0, +1}` — about **1.58 bits per weight**, the
[BitNet b1.58](https://arxiv.org/abs/2402.17764) regime. Because every weight is
`-1`, `0`, or `+1`, a matrix–vector product collapses from multiply-accumulate
to **add / subtract / skip**: there are no multiplies in the weight contraction.
Tritium's job is to make that multiply-free kernel fast everywhere — CUDA
(an addition-only path **and** an int8 tensor-core path), CPU SIMD
(AVX2 / AVX-512 / NEON with a scalar fallback), WebGPU (WGSL over Vulkan), and
WebAssembly (a scalar reference path) — and to do so while staying
bit-exact for the on-disk packing and within a documented tolerance for the
floating-point accumulation.

The linear primitive is **W1.58A8**: ternary weights with int8-quantized
activations. The activation is quantized per-token (absmax, `Qp = 127`), the
ternary contraction runs add/sub/skip, and both the per-token activation scale
and the per-channel weight scale are folded into the `f32` output. This is the
BitNet linear layer.

## What is in the workspace

Tritium consolidates prior art — BitNet / `bitnet.cpp`, T-MAC, BitBLAS,
llama.cpp's `TQ1_0`/`TQ2_0`, GPTQ-Marlin, ExLlamaV2 — behind one Rust workspace.
Each crate has a single responsibility:

| Crate | Responsibility |
|-------|----------------|
| `tritium-core` | Foundational ternary types, dtypes, scaling schemes, and the **reference math** every backend is graded against. |
| `tritium-spec` | The object-safe `TernaryBackend` trait — the contract every backend implements. No implementations. |
| `tritium-format` | Ternary weight packing (`TQ1_0`/`TQ2_0`) and GGUF I/O — the single source of truth for on-disk layout. |
| `tritium-testkit` | Conformance vectors + a generic harness any backend runs to prove it matches the reference. |
| `tritium-runtime` | Backend registry and dispatch; backends self-register via `linkme` with no central edit. |
| `tritium-cpu` | CPU backend: runtime-dispatched ternary mpGEMM — AVX-512 → AVX2 on x86-64, NEON on aarch64, scalar fallback. |
| `tritium-cuda` | CUDA backend: `cudarc` host side + an addition-only `.cu` kernel built by `build.rs`/`nvcc`. |
| `tritium-wgpu` | Cross-platform GPU backend: WGSL ternary mpGEMM over `wgpu` (validated on Vulkan). |
| `tritium-wasm` | Scalar `wasm32-wasip1` backend: the reference mpGEMM with no host-only deps. |
| `tritium-quantize` | SALT — sensitivity-allocated layered ternary quantization (residual planes + rate-distortion allocation). |
| `tritium-nn` | Inference layer over the backend: nn ops, KV cache, and the BitNet model runner. |
| `tritium-train` | STE autograd, QAT, an optimizer, and LoRA for ternary BitNet models (single-node). |
| `tritium-cli` | The `tritium` command-line tool. |
| `tritium-serve` | OpenAI-compatible HTTP/SSE inference server (axum, feature-gated). |
| `tritium-ffi` | C ABI (`cdylib` + `staticlib`) for inference from C/C++/any language. |
| `tritium-candle` | A candle `CustomOp1` running Tritium's ternary mpGEMM, bit-exact with the reference. |
| `tritium-burn` | A backend-generic burn op running Tritium's ternary mpGEMM, bit-exact with the reference. |
| `tritium-onnx` | A bit-exact ternary mpGEMM kernel + an `ort` 2.x custom ONNX operator. |
| `tritium-py` | PyO3 bindings (maturin wheel). |

## How this book is organized

- **[Architecture](./architecture.md)** — the inward-pointing crate graph, the
  `TernaryBackend` trait and `DeviceCaps`, and the frozen-vector conformance
  model.
- **[Quickstart](./quickstart.md)** — build the workspace and drive the
  `tritium` CLI.
- **[Backends](./backends.md)** — what each backend does and how capability
  fallback works.
- **[Quantization](./quantization.md)** — the SALT pipeline.
- **[Training](./training.md)** — QAT / STE / LoRA.
- **[Interop](./interop.md)** — serve, the C ABI, candle/burn ops, and the ONNX
  custom op, with real entry points.
- **[Conformance](./conformance.md)** and **[Contributing](./contributing.md)**.

> **Project status.** Tritium is pre-1.0 and developed milestone-by-milestone
> (see the [release roadmap ADR](../../adr/0002-release-roadmap.md) and
> [`docs/ROADMAP.md`](../../ROADMAP.md)). v0.80 interop is complete on the 0.6.x
> line; v0.90 hardening (this book is part of it) and v1.0 are in progress. APIs
> may break between minor versions until 1.0.

## License

Tritium is [Apache-2.0](https://github.com/Quitetall/tritium/blob/main/LICENSE).
Vendored upstreams are attributed in `NOTICE`.
