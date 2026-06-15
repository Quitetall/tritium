---
type: Design Principle
title: Hexagonal layering
description: Dependencies point inward only; backends and frontends are interchangeable adapters.
tags: [architecture, ports-and-adapters, conformance]
timestamp: 2026-06-14T00:00:00Z
---

# Hexagonal layering

Tritium is ports-and-adapters. Dependencies point **inward** only:

```
foundation  (core · format · spec · testkit)
   ↑
backends    (cpu · cuda · metal · rocm · wgpu)   — each implements the spec
   ↑
runtime     (runtime · quantize · nn · train)
   ↑
frontends   (ffi · py · onnx · candle · burn · wasm · cli · serve)
```

Rules:

- A **frontend never depends on a concrete backend.** It targets the runtime/spec.
- A **backend never depends on another backend.** Each is an isolated adapter.
- The **contract lives in `tritium-spec`**; the correctness ground truth lives in
  [reference mpGEMM](/concepts/reference-mpgemm.md).
- Every backend runs the **same conformance vectors** from `tritium-testkit`, so
  cross-backend bit-exactness is structural, not hoped-for.

Consequence: a new chip is one new backend crate that cannot break anyone; a new
consumer is one new frontend crate that cannot reach past the contract. See the
[crate graph](/architecture/crate-graph.md).
