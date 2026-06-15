# ADR 0004 — v0.20 Inference Spine

- **Status:** Accepted (scope + decisions approved 2026-06-14)
- **Relates:** executes the 0.20 milestone of [ADR 0002](./0002-release-roadmap.md); builds on [ADR 0003](./0003-v010-implementation.md)

## Context

v0.10 gives a conformant ternary mpGEMM on CPU + CUDA behind the `TernaryBackend`
contract. v0.20 turns that primitive into **end-to-end token generation**: load
**BitNet b1.58 2B4T** from GGUF and decode tokens that match a reference impl.
Plan synthesized by a 5-agent research workflow; this ADR records the approved
scope and the three decisions made on it.

## Decision

Build two crates plus a CLI command, with the BitNet 2B4T greedy/perplexity match
as the exit gate.

### Approved decisions

1. **Tokenizer — HuggingFace now, native later.** Use the HF `tokenizers` crate (or
   the GGUF-embedded vocab) to tokenize for the acceptance test now; a native
   Tritium tokenizer is deferred to v0.80. `tritium-nn` keeps a `Tokenizer` *trait*
   only this milestone.
2. **Attention — naive.** Correctness-first masked GQA attention; flash/fused
   attention is a 0.30 performance item.
3. **CUDA parity — loosened for non-ternary ops.** Ternary mpGEMM stays ≤`1e-4`
   relative; fp16/fp32 attention + softmax + norm paths use ≤`2e-3` (ADR 0002 fp16
   convention) because CPU/GPU reductions reorder. **Greedy token-IDs must still
   match exactly.**

### Crates

- **`tritium-nn`** — inference layer over `TernaryBackend`:
  - `ops/` — `rmsnorm`, `rope`, `attention` (GQA + causal mask), `softmax`, `sampling` (greedy/top-k/top-p/temperature).
  - `layers/` — `transformer_block`, `mlp` (ReLU² FFN, bias-free).
  - `kv_cache/` — paged K/V cache; incremental decode must equal full recompute.
  - `model/` — `ModelConfig` (from GGUF metadata), `ModelRunner::{load, generate}`, `Tokenizer` trait.
  - Ternary linears call `backend.mpgemm`; norms/softmax run in fp32.
- **`tritium-py`** — PyO3 0.23 + maturin wheel: `Model.load/generate` + a low-level
  ternary matmul op; numpy/DLPack interop; GIL released during compute.
- **`tritium-cli`** — add a `generate` subcommand.

### BitNet b1.58 2B4T target (from research)

~2.4B params; **bias-free ternary** linear weights (TQ) with fp16 norms/embeddings/
lm_head; **GQA** (K/V 8 heads, 4:1 group); RoPE; **ReLU² (squared ReLU) FFN**;
sub-LN / RMSNorm; context 4096.

### Parallelization waves

`foundation + nn-testkit` → `per-op kernels (3-wide parallel: rmsnorm/rope/attention/
sampling vs numpy/torch refs)` → `integration (model runner, sequential)` →
`tritium-py (parallel with integration)` → `cli generate`. Gate-driven, not dated.

## Validation (exit gates)

- **C** each op vs a numpy/torch reference (rmsnorm, rope, GQA attention, softmax, sampling).
- **C** KV-cache incremental decode == full recompute (across page boundaries).
- **C (acceptance)** greedy decode of BitNet 2B4T produces token-IDs **exactly
  matching** the reference (bitnet.cpp / HF) for committed prompts, ≥256 tokens, on
  **CPU + CUDA**; perplexity ≤`1%` of reference on a fixed eval set.
- **C** sampling distribution matches reference within a χ²/KL bound at fixed seed.
- **P** CPU↔CUDA: ternary ≤1e-4, non-ternary ≤2e-3, greedy token-match exact.
- **F/Co** Python binding: dtype/shape errors raise Python exceptions (no segfault);
  GIL released; ≥4 threads no deadlock.
- **E** seq-len 1 decode, max context, batch>1, large-M prefill.

CI lanes: `nn-ops` (cpu, fast), `nn-integration` (model download + cache),
`nn-cuda` (GPU, reuses the 4090 lane), `py-binding`.

## Definition of done — tag `v0.20.0`

- [ ] `tritium-nn` ops + layers + KV cache + `ModelRunner`; `tritium-py` wheel; `cli generate`.
- [ ] BitNet 2B4T greedy token-match + perplexity ≤1% on CPU **and** CUDA.
- [ ] All per-op + KV + edge + py gates green; cpu-only CI green; GPU lane green.
- [ ] HF tokenizer wired for the acceptance test; committed pre-tokenized prompts as fallback.
- [ ] Model fetched + cached in CI (gated, not on every push).

## Open questions

- Model-download/caching strategy for CI (HF hub vs vendored small fixture).
- Where the HF tokenizer runs (Rust `tokenizers` crate vs Python-side in the binding).
- ReLU² vs SiLU confirm from the exact 2B4T GGUF metadata at load time.
