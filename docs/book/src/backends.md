# Backends

Every backend implements the same `TernaryBackend` trait from `tritium-spec` and
is graded against the same frozen conformance vectors from `tritium-testkit`
(see [Conformance](./conformance.md)). That is what makes cross-backend
bit-exactness — for the packing paths — and tolerance-bounded agreement — for
the float path — *structural*. The shipped backends:

## `tritium-cpu` — x86-64 + aarch64 CPU

A runtime-dispatched SIMD kernel that picks the widest path the host supports: on
x86-64 **AVX-512** (`avx512f`+`avx512bw`+`avx512vl`), then **AVX2**; on aarch64
**NEON**; with a scalar fallback for hosts without a SIMD path. The scalar path
delegates to `tritium_core::reference_mpgemm`, so it is correct by construction;
the SIMD paths are graded bit-for-bit against it. The backend self-registers with the runtime
through the `BACKENDS` `linkme` distributed slice, so linking the crate into a
binary makes a `"cpu"` backend appear in the registry with no central edit.

The path: `upload_weights` validates the packed byte length against the `[N, K]`
shape and the format's block size and stores the bytes verbatim in a
`CpuBuffer`; `mpgemm` downcasts the buffer, unpacks the `[N, K]` weights, and
runs the contraction. This is the default backend in every build.

## `tritium-cuda` — NVIDIA GPU

`cudarc` host side plus an addition-only ternary mpGEMM `.cu` kernel. There is
no CMake: `build.rs` shells `nvcc` to compile the kernel, and the PTX/cubin is
loaded at runtime via `cudarc`. The crate is behind the `cuda` feature, so the
default workspace build never needs a CUDA toolkit. CUDA exposes two compute
paths — an addition-only kernel and an int8 tensor-core (IMMA) kernel — selected
from `DeviceCaps`.

```sh
cargo test -p tritium-cuda --features cuda     # needs nvcc + an NVIDIA GPU
```

## `tritium-wgpu` — cross-platform GPU (Vulkan)

A WGSL ternary mpGEMM over `wgpu`, validated on a Vulkan adapter (an RTX 4090).
The shader is the same add/sub/skip form, dispatched in 2-D to get past the
65535-per-dimension workgroup cap, and error-scoped so an adapter failure is a
returned error rather than a panic. The `wgpu` dependency is pinned exactly
(`=23.0.1`) so the supply-chain `wildcards = "deny"` gate stays meaningful. The
backend self-skips when no Vulkan adapter is present.

## `tritium-wasm` — WebAssembly (`wasm32-wasip1`)

A scalar `TernaryBackend` for `wasm32-wasip1`: the reference ternary mpGEMM with
**no host-only dependencies** — no `rayon`, no `linkme`, no SIMD intrinsics. It
depends only on the foundation crates (`spec`/`core`/`format`), because
`linkme`'s `distributed_slice` is unavailable on `wasm32`; the backend is
constructed explicitly instead of self-registering. Its conformance suite runs
**inside wasmtime** (Cranelift) in CI.

## Planned

`tritium-metal` (Apple) and `tritium-rocm` (AMD) are planned platform backends
(see the backend-breadth ADR (see the [research repository](https://github.com/Quitetall/tritium-research))); they are
fenced behind the per-platform hardware they need.

## Capability fallback

`DeviceCaps` (`supports_imma`, `supports_fp8`, the `features` flags) lets the
runtime pick a path, but a backend that *lacks* a capability must still return a
**correct** result. The fused W1.58A8 primitive is the case that matters: a
backend with no fp8 path must degrade gracefully instead of panicking or
silently producing garbage.

`tritium-testkit` pins this with the **fused-fallback contract**,
`run_fused_fallback_contract`. Where `run_conformance` exercises the plain
`mpgemm`, the fallback contract exercises the fused path on backends that cannot
take the hardware-accelerated route, asserts the degraded result still lands
within a scale-aware tolerance floor, and fails a backend that either refuses to
degrade or degrades to a wrong answer. The CPU, `wgpu`, and `wasm` backends are
all run through this contract.

This contract is the v0.70 capability-fallback gate: it makes "no fp8 here" a
*correctness-preserving* fallback, not a missing feature that breaks callers.
