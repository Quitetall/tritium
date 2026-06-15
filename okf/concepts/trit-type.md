---
type: Type
title: Trit
description: The ternary scalar — repr(transparent) over i8, invariant {-1, 0, +1}.
resource: https://github.com/Quitetall/tritium/blob/main/crates/tritium-core/src/trit.rs
tags: [tritium-core, type, ternary]
timestamp: 2026-06-14T00:00:00Z
---

# Trit

The ternary scalar in [tritium-core](/crates/tritium-core.md).

- `#[repr(transparent)]` over `i8`, so `&[Trit]` is bit-compatible with an `&[i8]`
  whose elements are all in `{-1, 0, 1}`. Backends transmute packed buffers once
  the invariant holds.
- Invariant `{-1, 0, +1}` is upheld by every constructor: `from_i8` (validating),
  `from_sign` (collapse any integer to its sign), `TryFrom<i8>`.
- Constants `Trit::NEG`, `Trit::ZERO`, `Trit::POS`. `is_zero()` marks the pruned
  state backends skip in the inner loop.

Underpins [ternary weights](/concepts/ternary.md) and the
[reference mpGEMM](/concepts/reference-mpgemm.md).
