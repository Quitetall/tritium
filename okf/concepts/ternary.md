---
type: Concept
title: Ternary weights
description: Weights in {-1, 0, +1} turn matmul into add / subtract / skip, no multiplies.
tags: [ternary, bitnet, mpgemm, multiply-free]
timestamp: 2026-06-14T00:00:00Z
---

# Ternary weights

A ternary weight is one of `{-1, 0, +1}` — ~1.585 bits of information
(`log2 3`). In a dot product against an activation vector, such a weight needs no
multiplier:

- `+1` → **add** the activation.
- `-1` → **subtract** the activation.
- `0`  → **skip**.

So a full matmul `C = A · Wᵀ` becomes accumulation by sign. This is the entire
premise of Tritium: ~10× smaller weights, no multiply units, memory-bandwidth-bound
instead of compute-bound. The information-theoretic floor is the constant
`TERNARY_IDEAL_BITS = 1.5849625`; real packed width depends on the
[TQ format](/concepts/tq-formats.md).

Related: the [Trit type](/concepts/trit-type.md) encodes one such value; the
[reference mpGEMM](/concepts/reference-mpgemm.md) expresses the add/sub/skip form
as the correctness contract; [SALT](/concepts/salt-quantization.md) adds capacity
back where the model is sensitive while staying multiply-free.

Prior art: BitNet b1.58, T-MAC, llama.cpp TQ formats.
