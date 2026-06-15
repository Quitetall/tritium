---
type: Update Log
title: Change Log
description: Chronological history of the Tritium knowledge bundle.
timestamp: 2026-06-14T00:00:00Z
---

# Log

- **2026-06-15** — Tagged **v0.10.0** (final). All U1–U9 gates closed: CUDA kernel +
  CPU↔CUDA parity (U2) validated on an RTX 4090, GGUF fuzz 550M runs/1h 0 crashes
  (U5), compute-sanitizer 0 errors (U7), real-GGUF fixture vs the official writer.
  v0.20 (Inference Spine) started — ADR 0004, `tritium-nn` foundation.
- **2026-06-14** — Shipped **v0.10.0-rc1** Foundation: 8 crates (core, spec, format,
  testkit, runtime, cpu, cuda, cli). CPU ternary mpGEMM bit-exact vs reference;
  end-to-end backend registry + CLI. GPU/fuzz/sanitizer gates deferred to CI lanes
  (ADR 0003).
- **2026-06-14** — Added the release roadmap concept (ADR 0002): depth-first
  v0.x0 → v1.0 staircase with the validation taxonomy and per-milestone exit gates.
- **2026-06-14** — Bundle created. Captured hexagonal architecture, crate graph,
  ternary thesis, TQ1_0/TQ2_0 formats, the SALT quantization scheme, the reference
  mpGEMM contract, and `tritium-core`.
