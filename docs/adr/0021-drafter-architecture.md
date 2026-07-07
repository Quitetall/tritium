# ADR 0021 — Drafter architecture: a tiny ternary AR model in the same engine

Status: **PROPOSED** (2026-07-07) — verifier side complete (greedy +
sampling accept rule, HTTP tree endpoints, in-process spec loop); blocked on
training the draft model (a BLUT project, recipe below).

## Context

The BASTION verifier is done: `tree_verify_greedy` / `tree_verify_logits` +
`tree_commit`, lossless greedy and lossless-in-distribution sampling, wired
through tritium-serve (`--spec lookup` ships the model-free prompt-lookup
drafter at 1.19× greedy). The remaining 2–3× lives in drafter quality:
3.65 tok/verify with n-gram lookup vs 8–12 with a trained drafter, against
a verify cost (~8.5 ms) that amortizes over accepted tokens.

Candidate architectures, assessed:

| architecture | acceptance | why not / why |
|---|---|---|
| DFlash block diffusion (external python) | best-in-class published | drafters are per-target-family; ours target Qwen-27B — tokenizer-incompatible with BitNet; training block-diffusion from scratch is a research project; cross-process orchestration + VRAM co-residency |
| EAGLE-style draft head | high | needs target hidden states at draft time → invasive engine surgery inside the decode graph |
| Medusa multi-head | medium | modifies the target model; retrains heads |
| **tiny ternary AR model, same engine** | medium-high | zero new engine code — it IS a Tritium model; ternary → tiny VRAM + very fast decode; trained by BLUT (its purpose); the whole loop stays in tritium-serve |

## Decision

Draft with a **~100–200M-parameter BitNet-style (ternary) autoregressive
model on the target's tokenizer (LLaMA-3, 128256)**, served by the *same*
tritium-serve process as a second `ModelRunner` (its own KV, its own decode
graphs, same GPU). The spec loop already in `generator.rs` swaps
`lookup_draft` for K greedy draft steps; everything downstream
(tree_verify_logits / accept rule / tree_commit) is unchanged.

Why this shape wins here:

- **On-thesis**: ternary drafter for a ternary target — the drafter decodes
  at several thousand tok/s on the same kernels this repo already optimized,
  so drafting 8–12 tokens costs ~2–4 ms against an 8.5 ms verify.
- **No new seams**: no python sidecar, no HTTP hop, no marginals protocol —
  the DFlash/LAMU external path (docs/bastion-lamu-integration.md) remains
  valid for lucebox-hub targets, but BitNet's drafter shouldn't pay
  cross-process costs the engine can absorb.
- **Trainable with what exists**: BLUT's typed DAG (SFT/distill →
  ConvertGguf → RegisterModel) is exactly this pipeline.
- Chain drafts first (the accept rule is live for chains); top-k marginals
  from the drafter's logits enable branchy trees later — `tree_verify_*`
  already accepts arbitrary trees.

## The BLUT recipe (the blocking work)

1. **Student**: 6–8 layers, n_embd 512–768, GQA matching a small head count,
   BitNet b1.58 recipe (ternary linears + A8 activations), LLaMA-3 tokenizer
   (embeddings can be f16-tied like the target; the embedding table dominates
   the student's size — ~130M params of it — so the *ternary* body stays
   almost free).
2. **Data**: distillation from BitNet 2B4T — sample the target's own
   generations (temperature ~0.8–1.0 over a broad prompt mix) and train the
   student on next-token cross-entropy against the target's sampled streams
   (sequence-level KD; logit-KD optional later). Acceptance is measured
   against the *target*, so the target's own distribution IS the training
   signal.
3. **Gate**: offline acceptance-rate probe — draft K=8 chains on held-out
   prompts, verify with the target, report tok/verify. Ship threshold:
   ≥6 tok/verify on prose (≈2× decode at current verify cost).
4. **Ship**: ConvertGguf (I2_S or TQ2_0 — the loader takes either) →
   `tritium-serve --model target.gguf --draft-model student.gguf`.

## Consequences

- tritium-serve grows `--draft-model <gguf>` (second runner; drafts via its
  own decode graphs; falls back to prompt-lookup when absent). The plumbing
  is validated *today* by self-speculation (target drafts for itself:
  acceptance ≈ 100%, wall-clock < 1× — a correctness rig, not a win).
- VRAM: student ≈ 0.4–0.5 GB (mostly the f16 embedding) + its KV — fits
  beside the 1.2 GB target with room to spare.
- The drafter inherits every engine improvement automatically (f16 KV,
  future rungs, kernel work).
