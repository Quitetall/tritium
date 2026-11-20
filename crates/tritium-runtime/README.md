# tritium-runtime

Backend registry and runtime dispatch for ternary execution.

`Registry::init()` always provides the `reference` semantic backend, after
available host linker-discovered hardware adapters. The reserved name cannot be
overridden. WebAssembly uses `Registry::with_entries(&entries)` for explicit
adapters; its default list has the same reference, not host linker sections.

Reference upload owns additive trits, exact widened stored scales and compact
tile indexes, or dense f32 values. It delegates matmul/gather to the shared core
and accepts the same native-only and checked/logged shared-payload admission as
other adapters. Payload bytes include logical owned count/index buffers; they
are not process RSS, allocator capacity, KV memory or a compression/speed claim.
Legacy packed TQ operations are explicitly unsupported here; the testkit's
unit-block conformance adapter is not a production GGUF decoder.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.
