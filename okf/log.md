---
type: Update Log
title: Change Log
description: Chronological history of the Tritium knowledge bundle.
timestamp: 2026-06-14T00:00:00Z
---

# Log

- **2026-06-15** — Tagged **v0.20.0** (Inference Spine). BitNet b1.58 2B4T loads from
  its I2_S GGUF and decodes tokens matching HF transformers on CPU + CUDA: CUDA greedy
  256/256 exact, perplexity 2.81e-3 (≤1%), CPU↔CUDA bit-identical. New: tritium-nn
  (ops/layers/KV/ModelRunner + W1.58A8 quant), tritium-format I2_S reader, tritium-py
  wheel, tritium-cli generate. ADR 0004.
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
