# Interop

The v0.80 interop milestone (ADR 0010 (see the [research repository](https://github.com/Quitetall/tritium-research))) exposes
Tritium through the seams other ecosystems already speak: an OpenAI-compatible
HTTP server, a C ABI, candle/burn ops, and an ONNX custom operator. Each is
feature-gated so the default workspace build stays free of the heavy deps; each
interop op runs the **same reference kernel**, so a frontend layer is bit-exact
with what every backend is graded against.

## `tritium-serve` — OpenAI HTTP/SSE

An axum server speaking the OpenAI wire protocol (chat completions, buffered and
streamed via SSE), behind the `serve` feature. It is LAMU-compatible by virtue of
that wire fidelity.

Entry points (`tritium-serve` crate root):

- `build_router(config: ServeConfig) -> Router` (feature `serve`) — build the
  axum router.
- `ServeConfig` — server configuration (`Default` provided).
- `Generator` — the seam the server decodes through, with two implementations:
  `RunnerGenerator` (a real model) and `MockGenerator` (model-free, for the
  contract tests). Supporting types: `GenRequest`, `Step`, `FinishReason`,
  `Sampling`, `GenError`.
- `IdPassthroughTokenizer` — a token-ID passthrough tokenizer.

The server runs one decode thread behind a bounded queue (concurrency,
backpressure, graceful shutdown). The OpenAI wire contract — schema, SSE framing,
`stream == buffered` equivalence, `finish_reason`, stop strings, concurrency,
backpressure, shutdown — is proven **model-free** through `MockGenerator`, so it
runs on CPU in CI with no model. The real-model end-to-end path is a manual lane
gated on a GGUF being present.

```sh
cargo test -p tritium-serve --features serve     # the model-free contract suite
```

## `tritium-ffi` — C ABI

A C ABI (`cdylib` + `staticlib`) for inference from C/C++/any language. It is
**panic-safe** — panics never unwind across the boundary (they are caught) — and
every `unsafe extern "C"` entry point is null-checked. A `cbindgen`-generated
header (`include/tritium.h`) is kept honest by a drift test plus a C11/C++17
compile check.

The exported functions:

- `tritium_abi_version() -> u32` — the C ABI version (`TRITIUM_ABI_VERSION`,
  currently `1`).
- `tritium_version() -> *const c_char` — the crate version as a static,
  NUL-terminated string (never null).
- `tritium_model_load_file(path, out_status) -> *mut TritiumModel` — load a GGUF
  model on the CPU backend; returns null on failure and writes a `TritiumStatus`
  if `out_status` is non-null.
- `tritium_generate(...)` — single-pass / size-then-fill greedy generation;
  `*out_len` is always defined.
- `tritium_model_free(model)` — free the handle.

This crate unblocks the v1.0 C-ABI freeze.

## `tritium-candle` — candle `CustomOp1`

A candle-native op so a candle model graph can use BitNet ternary weights, behind
the `candle` feature.

- `ternary_mpgemm(...)` — applied via `apply_op1_no_bwd`; takes an `[M, K]` f32
  activation tensor, `[N, K]` packed ternary weights (`TQ2_0`/`TQ1_0`), and `[N]`
  per-output-channel scales, producing `[M, N]` f32. `N` is taken from
  `scales.len()`.
- `TernaryMpGemm<'a>` — the `CustomOp1` type it wraps.

The kernel is `tritium_core::reference_mpgemm` itself, so a candle BitNet layer is
**bit-exact** with the reference. It validates dtype / contiguity / `K` / packed
length and returns an error (never panics) on mismatch.

```sh
cargo test -p tritium-candle --features candle
```

## `tritium-burn` — backend-generic burn op

A burn op that runs the ternary mpGEMM on a burn `Tensor` for any `Backend`,
behind the `burn` feature.

- `ternary_mpgemm<B: Backend>(...)` — a host round-trip (read →
  `reference_mpgemm` → rebuild) over `[M, K]` f32 activations × `[N, K]` packed
  ternary weights × `[N]` scales → `[M, N]` f32, pinned to `DType::F32`,
  **bit-exact** with the reference. Works on any burn backend (NdArray, wgpu,
  cuda) in f32.
- `BurnTernaryError` — a lazy-backend read failure (e.g. deferred execution) is
  returned as this error, not a panic.

```sh
cargo test -p tritium-burn --features burn
```

## `tritium-onnx` — ONNX kernel + custom op

Two layers, so the always-on CI needs no native library:

- **Layer 1 (default, zero external deps):** `ternary_mpgemm_kernel(...)`, a plain
  bit-exact kernel whose conformance test is the default-feature gate. No `ort`,
  no `onnxruntime`.
- **Layer 2 (feature `onnx`, pulls `ort`):** `TritiumTernaryMpGemmOp`, an `ort`
  2.x custom operator exposing the kernel as the ONNX node `ONNX_OP_NAME`
  (`"TritiumTernaryMpGemm"`) under domain `ONNX_DOMAIN` (`"tritium"`), built with
  `tritium_operator_domain()`. `ort = 2.0.0-rc.12` (default-features off +
  `download-binaries` + `tls-rustls`) fetches a prebuilt onnxruntime at build, so
  a networked CI lane builds + tests `--features onnx` with no system library.
  The full native session dispatch is the `#[ignore]`d end-to-end test; the
  `run` kernel logic and operator registration are tested bit-exact.

```sh
cargo test -p tritium-onnx                  # Layer 1, always-on bit-exact kernel
cargo test -p tritium-onnx --features onnx  # Layer 2, the ort custom operator
```

## `tritium-py` — Python (PyO3 / maturin)

PyO3 bindings (a maturin `cdylib` wheel): load a ternary GGUF model and generate
tokens from Python. It is exercised via maturin + pytest rather than `cargo test`.
