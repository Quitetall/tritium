---
type: Concept
title: Release roadmap (v0.x0 → v1.0)
description: Depth-first staircase of shippable milestones to a real, usable v1.0, each with airtight exit gates.
resource: https://github.com/Quitetall/tritium/blob/main/docs/adr/0002-release-roadmap.md
tags: [roadmap, milestones, validation, release]
timestamp: 2026-06-14T00:00:00Z
---

# Release roadmap (v0.x0 → v1.0)

v1.0 is the full vision — all backends, inference + [SALT](/concepts/salt-quantization.md)
+ distributed training, interop — **real and usable**. Reached via shippable
`0.x0` milestones, **Approach A: capability depth-first, backend breadth second**,
CPU+CUDA first. Full detail + per-milestone exit gates in
[ADR 0002](https://github.com/Quitetall/tritium/blob/main/docs/adr/0002-release-roadmap.md).

| Milestone | Theme |
|---|---|
| 0.10 | Foundation (format·spec·testkit·cpu·cuda·runtime·cli) |
| 0.20 | Inference spine (nn ops·GGUF load·py·end-to-end tokens) |
| 0.30 | Performance (add-only + IMMA·autotune·AVX-512/NEON·benches) |
| 0.40 | SALT quantization |
| 0.50 | Training core (STE·QAT·LoRA) |
| 0.60 | Pretraining + distributed (FSDP/DDP) |
| 0.70 | Backend breadth (Metal·ROCm·WGPU/WASM) |
| 0.80 | Interop (ONNX·candle/burn·FFI·serve) |
| 0.90 | Hardening (fuzz·sanitizers·CI matrix·packaging·security) |
| 1.0 | Release (API/ABI freeze·docs·reproducible benches) |

Each milestone is gated by a uniform **validation taxonomy** — correctness,
cross-backend parity, edge/boundary, failure/invalid input, untrusted-input
safety, determinism, performance, memory, concurrency, docs — engineered so that
passing the gate means the milestone genuinely works, not just the happy path.
Gates are blocking: milestone N+1 does not start until N is green and tagged.
The [reference contract](/concepts/reference-mpgemm.md) and the conformance vectors
derived from it are what make the parity gates enforceable across
[every backend](/architecture/hexagonal-layering.md).
