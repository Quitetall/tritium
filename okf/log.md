---
type: Update Log
title: Change Log
description: Chronological history of the Tritium knowledge bundle.
timestamp: 2026-06-14T00:00:00Z
---

# Log

- **2026-06-14** — Shipped **v0.10.0-rc1** Foundation: 8 crates (core, spec, format,
  testkit, runtime, cpu, cuda, cli). CPU ternary mpGEMM bit-exact vs reference;
  end-to-end backend registry + CLI. GPU/fuzz/sanitizer gates deferred to CI lanes
  (ADR 0003).
- **2026-06-14** — Added the release roadmap concept (ADR 0002): depth-first
  v0.x0 → v1.0 staircase with the validation taxonomy and per-milestone exit gates.
- **2026-06-14** — Bundle created. Captured hexagonal architecture, crate graph,
  ternary thesis, TQ1_0/TQ2_0 formats, the SALT quantization scheme, the reference
  mpGEMM contract, and `tritium-core`.
