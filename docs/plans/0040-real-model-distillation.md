# 0040 — Real-model SALT distillation at scale (serves: ADR 0020 capstone; follows 0038b)

## Goal
Run the end-to-end SALT-distillation loop (proven on toys in 0038b: 98% recovery) on a REAL model —
SmolLM2-135M first (CPU-validatable), then scale to 32B on rented GPU (0041 capstone). The blocker
is a **differentiable whole-model forward**: `ModelRunner::forward` is inference-only; distillation
needs the same architecture on the autograd `Tape` with every 2D weight a SALT-STE latent.

## The gap vs the toy (0038b)
The toy is **single-head, no-GQA, tied assumptions**. The real Llama/Qwen forward adds:
1. **Multi-head + GQA attention** — Q is `n_head·head_dim` wide, K/V are `n_kv_head·head_dim`
   (grouped). Per-head scaled-dot-product-attention, KV heads shared across query-head groups.
   ← needs column **slice/concat** on the tape to split heads and reassemble. **THIS PLAN'S step 1.**
2. **Token embedding lookup** (differentiable gather) + **tied lm-head**.
3. Real dims (n_embd 576, 30 layers, vocab 49152), real RoPE θ, real config plumbing.
4. **Memory**: 135M params × (latent + Adam m,v + grads) fp32 ≈ 2–3 GB — fine on CPU for a few
   steps; 32B needs CPU-offload AdamW (ADR 0016) + gradient checkpointing (recompute forward in
   backward) + streaming quantize. GPU for throughput.

## Steps
### Step 1 — Tape column slice/concat primitives (this commit) — the multi-head enabler
`ops::shape::{slice_cols, concat_cols}` (forward + vjp) + `Tape::{slice_cols, concat_cols}`. Slice
extracts a column range `[rows, len]` from `[rows, cols]`; concat reassembles. These are exactly
what per-head attention needs: slice head `h`'s Q/K/V, run SDPA, concat the head outputs.
**Gate (TDD):** finite-difference gradcheck of both vjps; a tape round-trip (split a matrix into
column blocks and concat back == identity, gradient == ones) — the reshape is differentiable.

### Step 2 — Multi-head + GQA attention on the tape (reusable helper)
`tape_attention(t, x, wq,wk,wv,wo, n_head, n_kv_head, head_dim, θ)` using step 1 + the existing
rope/softmax/causal_mask/dense_matmul. Gate: matches a reference multi-head SDPA forward within
1e-4, and gradchecks end-to-end on a tiny multi-head config.

### Step 3 — Differentiable real-model forward + teacher-logit cache
A `TapeModel` mirroring `build_standard_model` (embeddings→N blocks→final-norm→lm-head), every 2D
weight a SALT-STE latent; token-embedding gather + tied head. Teacher-logit cache: run the fp
`ModelRunner` over a calibration set ONCE, store soft logits (top-k) to disk. Gate: `TapeModel`
forward (latents un-quantized) matches `ModelRunner::from_hf` logits within 1e-4 on SmolLM2 — the
differentiable forward is faithful.

### Step 4 — SmolLM2 distillation run (CPU, few steps) — the real-model recovery
Distill SmolLM2's SALT latents against the cached teacher logits; show ppl recovers vs PTQ on the
real model (the thing 0038 step 5 said needs e2e, now delivered at real scale). Gate: `ppl_distilled
≪ ppl_ptq`, approaching `ppl_fp`.

### Step 5 (0041 territory) — scale to 32B on rented GPU
CPU-offload AdamW + gradient checkpointing + streaming quantize writer; GPU throughput; the ≤1% ppl
capstone gate on a standard-transformer 32B, Qwen3.6-27B headline.

## Verification
Each step: `cargo test -p tritium-train` (+ `-p tritium-nn` for step 3/4) green incl. the new gate,
clippy `-D`/fmt clean, code-reviewer subagent, explicit-path commit (shared tree), push when clean.
