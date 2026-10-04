# tritium-wasm

wasm32 ternary inference: no-registry build for WASI and browser targets.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Portable-training receipt bundle

`examples/seal_wasi_training_receipts.rs` runs the frozen v2 portable-training
corpus inside a `wasm32-wasip1` guest and writes its content-addressed receipt
bundle. The guest does not infer its Wasmtime identity. Set
`TRITIUM_WASM_PHYSICAL_DEVICE` from the observed host runtime version and
architecture when compiling, then retain the host command/version output with
the bundle. For example:

```sh
WASMTIME_VERSION="$(wasmtime --version | awk '{print $2}')"
TRITIUM_WASM_PHYSICAL_DEVICE="wasmtime:${WASMTIME_VERSION}:$(uname -m)" \
  cargo run --locked --release --target wasm32-wasip1 -p tritium-wasm \
  --example seal_wasi_training_receipts -- release/v1.1/wasi-training-receipts
```

The example requires a non-placeholder `wasmtime:` identity. This is an
operator-supplied identity, not cryptographic runtime attestation. Run it only
from the clean candidate revision intended for evidence; the receipt embeds
the build identity, and the release verifier rejects dirty or mismatched
source revisions. A generated bundle is backend evidence only; candidate
registration and independent release gates remain separate.
