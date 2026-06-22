# ADR 0015 — Knowledge Graph Audit of the Tritium Codebase

- **Status:** Accepted
- **Date:** 2026-06-21
- **Deciders:** Brian Lam
- **Relates:** graphify automated extraction over 222 files (164 code, 58 docs)

## Context

An automated knowledge graph was built from the full Tritium repository using
graphify (AST extraction for code, LLM semantic extraction for docs). The graph
contains **2,426 nodes**, **5,422 edges**, and **136 communities** (20 labeled).
This ADR records the structural findings and what they reveal about the
codebase's architecture, coupling, and documentation gaps.

## Findings

### 1. God Nodes — The Core Abstractions

The 10 most-connected nodes by edge count:

| Rank | Node | Edges | Role |
|------|------|-------|------|
| 1 | `BackendError` | 98 | Error type threaded through every backend operation |
| 2 | `Result` | 96 | Return type of every fallible function |
| 3 | `CudaDecodeModel` | 74 | The device-resident decoder — largest single struct |
| 4 | `driver_err()` | 68 | CUDA driver error mapping helper |
| 5 | `CudaBackend` | 52 | The GPU backend entry point |
| 6 | `num_blocks()` | 44 | Block-size calculation utility |
| 7 | `Arc` | 36 | Shared ownership across async boundaries |
| 8 | `CudaSlice` | 32 | Device memory wrapper |
| 9 | `raw_launch()` | 30 | Low-level kernel launch primitive |
| 10 | `CudaFunction` | 29 | Resolved kernel handle |

**Interpretation:** The codebase is error-centric (BackendError + Result dominate)
and CUDA-centric (6 of the top 10 are CUDA types). The `CudaDecodeModel` is the
single largest coupling point — it owns the KV cache, scratch buffers, weight
pointers, and kernel function handles. This is expected for a single-backend
system but would become a problem if a second GPU backend (ROCm, Metal) is added.

### 2. `num_blocks()` — The Hidden Bridge

`num_blocks()` has the **highest betweenness centrality** (0.135) in the entire
graph, bridging **16 communities**. It is called from:

- TQ2_0 format packing (`tritium-format`)
- GGUF weight loading (`tritium-format`)
- Conformance test harness (`tritium-testkit`)
- CUDA autotuning (`tritium-cuda`)
- CUDA kernel launch sites (`tritium-cuda`)
- Layer/projection code (`tritium-nn`)
- Benchmark harness (`benches`)

**Interpretation:** `num_blocks()` is the universal "how many 256-trit blocks
does this shape need?" function. Its ubiquity is correct — TQ2_0 block geometry
is a fundamental invariant — but it also means any change to block size (e.g.,
TQ1_0 with 512-trit blocks) would ripple through the entire codebase. Consider
making it a method on a `BlockGeometry` trait rather than a free function.

### 3. Surprising Connections

- **SALT Quantization ↔ LoRA Adapters** — both manipulate frozen ternary bases.
  SALT adds quantization planes on top; LoRA adds low-rank adapters. They share
  the "modify a frozen base" pattern but have no structural link.

- **BitNet 2B4T ↔ mpGEMM** — the acceptance model and its core operation are
  deeply linked through the conformance gate (greedy 256/256 token match).

- **IMMA tensor-core kernel ↔ mpGEMM** — two paths to the same ternary multiply
  (TQ2_0 tiled for decode, IMMA int8 for prefill). The graph found a
  `conceptually_related_to` edge between them.

- **Distributed Checkpoint ↔ Training Pipeline** — the DCP system (plans
  0014-0017) is tightly coupled to the training core (plans 0005-0010) through
  the optimizer state serialization path.

### 4. Community Structure

The 20 labeled communities cluster into 5 functional areas:

| Area | Communities | Nodes |
|------|------------|-------|
| **Core types & error handling** | Core Types & Traits, Error Handling, Format & Encoding | ~190 |
| **CUDA backend** | CUDA Stream & Launch, CUDA Kernel Functions, CUDA Autotuning, Batched Decode Graph | ~200 |
| **Inference pipeline** | Model Runner & Inference, KV Cache & Attention, Neural Network Ops, Layer & Projection | ~250 |
| **Training & distributed** | Training Pipeline, Distributed Checkpoint | ~80 |
| **Architecture & docs** | Architecture & Concepts, Conformance & Testing, Backend Abstraction | ~150 |

**Observation:** The CUDA backend communities are the densest (highest internal
edge count), reflecting the tight coupling between kernel launch, autotuning, and
memory management. The inference pipeline communities are more loosely coupled,
which is healthy — the ops layer is independent of the model runner.

### 5. Cohesion Warnings

Three communities have low cohesion scores (< 0.08), suggesting they may need
refactoring:

- **Core Types & Traits** (0.066) — too many unrelated types lumped together
- **Ternary Math & GEMM** (0.058) — mixing format packing with reference math
- **Model Runner & Inference** (0.071) — runner logic interleaved with weight
  loading

These are not urgent but should be addressed before v1.0 to keep the crate
graph navigable.

### 6. Import Cycles

The graph found several 1-file self-import cycles (test files importing
themselves). These are benign — Rust's module system allows `mod tests` within
the same file — but the graph correctly flags them as structural anomalies.

## Decision

Record these findings as an ADR for future reference. No immediate action
required — the structural health is acceptable for a v0.5.x codebase. The
following items should be tracked:

1. **`num_blocks()` extraction** — consider a `BlockGeometry` trait (low priority,
   only needed if block size changes)
2. **`CudaDecodeModel` decomposition** — if a second GPU backend is added, split
   the model into backend-agnostic and backend-specific parts
3. **Cohesion refactoring** — address before v1.0 (see ADR 0012)

## Consequences

- **Positive:** the knowledge graph provides a queryable map of the codebase.
  `graphify query "How does the GEMM kernel work?"` can now trace from the CUDA
  kernel through the launch code to the conformance tests.
- **Negative:** the graph is a snapshot — it will drift as code changes. Run
  `graphify --update` periodically to keep it current.
- **Neutral:** the 136 communities suggest the codebase is well-modularized. The
  20 labeled communities cover the main functional areas. The remaining 116 are
  small clusters (2-5 nodes) that will either grow or merge as the codebase
  evolves.
