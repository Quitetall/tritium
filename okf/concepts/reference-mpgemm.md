---
type: Contract
title: Reference mpGEMM
description: The slow, obviously-correct mixed-precision GEMM every backend must match.
resource: https://github.com/2BrianCells/tritium/blob/main/crates/tritium-core/src/reference.rs
tags: [tritium-core, conformance, ground-truth, mpgemm]
timestamp: 2026-06-14T00:00:00Z
---

# Reference mpGEMM

The correctness ground truth in [tritium-core](/crates/tritium-core.md):

```text
out[m, n] = scale[n] · Σ_k  act[m, k] · w[n, k]
```

- `act` `[M,K]` f32 activations · `weights` `[N,K]` ternary (output-major) ·
  `scale_n` `[N]` per-channel scales · `out` `[M,N]`.
- Deliberately written in **add / subtract / skip** form (no multiply) — the same
  contract a real kernel optimizes, here for clarity over speed.
- Validates buffer lengths against [GemmShape](/crates/tritium-core.md); returns
  `TritError::ShapeMismatch` otherwise.

Role: every backend kernel (CUDA, CPU SIMD, future Metal/ROCm/WGPU) must match this
within tolerance. Conformance vectors derived from it (in `tritium-testkit`) make
cross-backend bit-exactness structural — see
[hexagonal layering](/architecture/hexagonal-layering.md).

A property test asserts the multiply-free form equals a plain f32 matmul that
treats each trit as a coefficient; divergence means the decomposition is wrong.
