# Introduction

**Tritium** is a from-scratch library for **ternary-model inference and
training**, GPU- and CPU-first. It is written in Rust as a single Cargo
workspace (no CMake): the CUDA backend shells `nvcc` from `build.rs` and loads
PTX/cubin at runtime via `cudarc`, and every backend is gated behind a feature
flag so the default build is CPU-only.

## What "ternary" means here

Ternary weights live in `{-1, 0, +1}`. A uniformly distributed trit has
`log2(3) ≈ 1.585` bits of mathematical entropy, but that is not an artifact,
resident-memory, or whole-model compression claim. Physical rate also includes
packing, scales, indexes, metadata, preserved tensors, alignment, and codec
overhead.

Tritium uses one to three additive ternary planes. The compact base plane is
augmented only where a recipe and measured budget admit residual planes, so
physical bytes and quality form an explicit tradeoff. Each plane turns the
weight contraction into **add / subtract / skip**; scales and any activation
quantization remain part of the numerical contract.

W1.58A8 is one important native inference profile: activations are quantized
per token, the ternary contraction uses add/subtract/skip, and activation and
weight scales are folded into the floating output. It is not the only training
or PTQ path. PyTorch research graphs retain floating latent masters during QAT,
and additive PTQ/refinement recipes own their own precision and rate evidence.

Implementations exist for CPU SIMD, CUDA, Metal, ROCm, native `wgpu`, WASI/WASM,
and browser WebGPU at different qualification levels. Native `wgpu` evidence on
Vulkan is not browser evidence, and browser WebGPU may map to Vulkan, Metal, or
D3D12. A backend is supported only when its exact compatibility cell has a
physical receipt.

## What is in the workspace

Tritium consolidates prior art — BitNet / `bitnet.cpp`, T-MAC, BitBLAS,
llama.cpp's `TQ1_0`/`TQ2_0`, GPTQ-Marlin, ExLlamaV2 — behind one Rust workspace.
Each component has a single responsibility:

| Component | Responsibility |
|-------|----------------|
| `tritium-core` | Foundational ternary types, dtypes, scaling schemes, and the **reference math** every backend is graded against. |
| `tritium-spec` | The object-safe `TernaryBackend` trait — the contract every backend implements. No implementations. |
| `tritium-format` | Ternary weight packing (`TQ1_0`/`TQ2_0`) and GGUF I/O — the single source of truth for on-disk layout. |
| `tritium-testkit` | Conformance vectors + a generic harness any backend runs to prove it matches the reference. |
| `tritium-runtime` | Backend registry and dispatch; backends self-register via `linkme` with no central edit. |
| `tritium-cpu` | CPU backend: runtime-dispatched ternary mpGEMM — AVX-512 → AVX2 on x86-64, NEON on aarch64, scalar fallback. |
| `tritium-cuda` | CUDA backend: `cudarc` host side + an addition-only `.cu` kernel built by `build.rs`/`nvcc`. |
| `tritium-metal` / `tritium-rocm` | Feature-gated Apple Metal and AMD HIP backends. |
| `tritium-wgpu` | Native cross-platform WGSL backend over `wgpu`; target qualification is receipt-specific. |
| `tritium-wasm` | Scalar `wasm32-wasip1` backend: the reference mpGEMM with no host-only deps. |
| `tritium-mcu` | Fixed-arena `no_std` ternary-codec execution for constrained targets. |
| `tritium-quantize` | SALT — sensitivity-allocated layered ternary quantization (residual planes + rate-distortion allocation). |
| `tritium-salt` | SALT V2 orchestration and resumable campaign pipeline. |
| `tritium-nn` | Inference layer over the backend: nn ops, KV cache, and the BitNet model runner. |
| `tritium-train` | STE autograd, QAT, optimizers, LoRA and distributed-training/checkpoint substrates. |
| `tritium-cli` | The `tritium` command-line tool. |
| `tritium-serve` | OpenAI-compatible HTTP/SSE inference server (axum, feature-gated). |
| `tritium-ffi` | C ABI (`cdylib` + `staticlib`) for inference from C/C++/any language. |
| `tritium-candle` | A candle `CustomOp1` running Tritium's ternary mpGEMM, bit-exact with the reference. |
| `tritium-burn` | A backend-generic burn op running Tritium's ternary mpGEMM, bit-exact with the reference. |
| `tritium-onnx` | A bit-exact ternary mpGEMM kernel + an `ort` 2.x custom ONNX operator. |
| `tritium-py` | PyO3 bindings (maturin wheel). |
| `@tritium-ai/web` | Strict-TypeScript compiled training session with WASM and WebGPU adapters. |

## How this book is organized

- **[Architecture](./architecture.md)** — the inward-pointing crate graph, the
  `TernaryBackend` trait and `DeviceCaps`, and the frozen-vector conformance
  model.
- **[Quickstart](./quickstart.md)** — build the workspace and drive the
  `tritium` CLI.
- **[Backends](./backends.md)** — what each backend does and how capability
  fallback works.
- **[Quantization](./quantization.md)** — additive SALT PTQ and physical rate.
- **[Training](./training.md)** — QAT, estimators, refinement and distributed substrate.
- **[Interop](./interop.md)** — serve, the C ABI, candle/burn ops, and the ONNX
  custom op, with real entry points.
- **[Model Zoo](./model-zoo.md)** and **[Benchmarks](./benchmarks.md)** — admitted
  artifacts, evidence rules and current blockers.
- **[Conformance](./conformance.md)** and **[Contributing](./contributing.md)**.

> **Project status.** The repository uses the `1.1.0-rc.1` candidate version; it
> is neither `LOCAL_RC_READY` nor a qualified public v1.1 release. Stable-core
> compatibility follows the v1.0 tier policy; evolving training, backend and
> interop APIs retain the 1.x runway documented in
> ADR 0033 (see the [research repository](https://github.com/Quitetall/tritium-research)).
> Package, flagship-model, browser, deployment and second-machine evidence gates
> remain open. The generated [compatibility matrix](../../compatibility.md) is
> authoritative: a pending cell is not support.

## License

Tritium is [Apache-2.0](https://github.com/Quitetall/tritium/blob/main/LICENSE).
Vendored upstreams are attributed in `NOTICE`.
