# tritium-cpu

CPU backend: SIMD ternary mpGEMM kernels (AVX2/AVX-512/NEON) and the bit-exact reference path.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Immutable packed scale adjoints

`CpuBackend::additive_stored_scale_count` reports the actual stored-scale length
of an uploaded additive handle. `accumulate_additive_matmul_scale_vjp` and
`accumulate_additive_embed_scale_vjp` add continuous scale gradients into that
many caller-owned f64 values, using one columns-wide f32 basis workspace.
They read admitted D2/B3/S34 bytes directly through the same checked core plan as
the decoded reference. No dense/trit master or hidden heap scratch is created.
Repeated uses and duplicate embedding indices accumulate; shape/index refusals
precede buffer mutation. Dense, foreign and legacy handles are rejected.

These scalar CPU operations do not update scales/codes, enable mutable Python
autograd or qualify a complete training backend. Existing upload payloads,
inference capabilities and forward results are unchanged. Process memory,
performance and model-quality acceptance require their separate release gates.
