---
type: Format
title: TQ1_0 and TQ2_0
description: The two canonical ternary packing schemes, inherited from llama.cpp/ggml.
tags: [format, packing, tq1_0, tq2_0, ggml]
timestamp: 2026-06-14T00:00:00Z
---

# TQ1_0 and TQ2_0

Canonical ternary packing schemes. `tritium-core` fixes the shared vocabulary
(`TernaryFormat`); byte-exact pack/unpack lives in `tritium-format`. Both follow
ggml's 256-element super-block with one fp16 block scale.

| Format | bits/weight | Packing | Unpack cost | Orientation |
|--------|-------------|---------|-------------|-------------|
| **TQ1_0** | 1.6875 | 5 trits/byte (`3^5 = 243 < 256`) | base-3 division | compact; CPU / edge |
| **TQ2_0** | 2.0625 | 2 bits/trit, 4/byte | shift + mask | GPU (matches BitNet int8 packing) |

Design rule: **TQ2_0 for GPU compute, TQ1_0 for edge/storage**; document the
conversion path as one source of truth in `tritium-core` /
[tritium-core](/crates/tritium-core.md). `TernaryFormat::is_bit_aligned()`
distinguishes the cheap-unpack (TQ2_0) case.

[SALT](/concepts/salt-quantization.md) extends TQ2_0 with a residual-plane sidecar.
