---
type: Crate
title: tritium-core
description: Foundation crate — ternary types, dtypes, scaling schemes, and reference math.
resource: https://github.com/2BrianCells/tritium/tree/main/crates/tritium-core
tags: [crate, foundation, no_std, l0]
timestamp: 2026-06-14T00:00:00Z
---

# tritium-core

The L0 foundation crate. Pure, dependency-free, `no_std`-able (`std`/`alloc`
features). Holds the shared vocabulary and the correctness ground truth; depends
on nothing, everything depends on it.

## Public surface

- **[`Trit`](/concepts/trit-type.md)** — the `{-1,0,+1}` scalar.
- **`DType`** — precision lattice: ternary, I4/I8/U8, F8E4M3/F8E5M2, F4E2M1,
  F16/BF16/F32. Mixed precision because ternary weights pair with higher-precision
  activations (W1.58A8) and GPU fp8/fp4 paths.
- **`TernaryFormat`** — [TQ1_0 / TQ2_0](/concepts/tq-formats.md) vocabulary + bpw +
  block size.
- **`ScaleGranularity`** (PerTensor / PerChannel / PerGroup) and **`absmean`** —
  the BitNet b1.58 scaling primitive.
- **`GemmShape`** — `(M, N, K)` geometry + buffer-fit checks.
- **[`reference_mpgemm`](/concepts/reference-mpgemm.md)** — the ground-truth GEMM.
- **`TritError`** — hand-rolled, no_std (no `thiserror`, zero deps).
- Constant `TERNARY_IDEAL_BITS = 1.5849625`.

## Status

Exists and tested: unit tests + a proptest asserting the multiply-free reference
equals a float matmul. `clippy` clean; builds with `--no-default-features` (no_std).

Design rule: this crate stays zero-dependency. It is the only thing every backend
links — keep it minimal.
