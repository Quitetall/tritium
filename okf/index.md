---
okf_version: "0.1"
type: Knowledge Bundle
title: Tritium
description: Foundational ternary-model inference and training library — OKF knowledge bundle.
resource: https://github.com/Quitetall/tritium
tags: [tritium, ternary, quantization, inference, cuda, bitnet]
timestamp: 2026-06-14T00:00:00Z
---

# Tritium — Knowledge Bundle

Agent- and human-readable knowledge for the Tritium project, in Google's
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
v0.1.

Tritium is a foundational library for **ternary-model inference and training**.
Weights live in `{-1, 0, +1}` (~1.58 bits) so matmul collapses to add / subtract
/ skip — no multiplies. GPU (CUDA) and CPU (SIMD + LUT) are first-class; ONNX and
PyTorch bindings bridge into existing stacks.

## Sections

- [Architecture](/architecture/index.md) — hexagonal layering and the crate graph.
- [Concepts](/concepts/index.md) — ternary math, packing formats, SALT quantization, the reference contract.
- [Crates](/crates/index.md) — per-crate knowledge.

History in [log](/log.md).
