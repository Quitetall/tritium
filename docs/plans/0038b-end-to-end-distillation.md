# 0038b — End-to-end SALT distillation across depth (serves: ADR 0020, follows 0038 step 5)

## Why
0038 step 5 established, on the real SmolLM2, that ternary PTQ is catastrophic at the model level
(ppl 24→3.3M) *because error compounds across the 30 layers*, and that a purely **local** layerwise
heal cannot rescue it (each layer heals against clean fp activations the broken model never sees).
The fix it pointed to: **end-to-end** distillation — train ALL latents jointly against the teacher's
final output. The 1-block e2e test (`salt_distill_e2e.rs`) proved the mechanism at depth 1. The
missing link is **depth**: does end-to-end distillation defeat the multi-layer *compounding* that
sinks the local heal?

## Step 1 (this increment) — multi-block end-to-end distillation gate
`crates/tritium-train/tests/salt_distill_deep.rs`: an N-block (N≥3) tiny transformer
(rmsnorm→qkv→RoPE→causal attn→o→residual→SwiGLU→residual, ×N, →out-norm→lm-head) with **every 2D
weight** an fp32 latent SALT-quantized in the forward (STE, T=1 = aggressive). An fp teacher (same
graph, un-quantized) supplies soft logits; the ternary student is distilled by
`softmax_xent(student, softmax(teacher))` (= the KL gradient) over ALL latents jointly with AdamW.
**Gate:** end-to-end distillation recovers the large majority of the deep PTQ gap (measured above
the teacher's entropy floor) by flowing the gradient through ALL blocks jointly — the multi-layer
recovery the per-projection local heal (step 5) could not deliver. Observed **97.7%** recovery on a
3-block model (PTQ gap 0.064 → 0.0015). **Note:** the compounding *catastrophe* itself is a
trained-real-scale effect (step 5's 24→3.3M on 30 real layers); independent tiny random models
don't reproduce monotone growth-with-depth, so per-depth PTQ is reported but not gated.

Reuses the whole tape op set already exercised by `salt_distill_e2e.rs` (rmsnorm/rope/softmax/
causal_mask/dense_matmul/silu/salt_ste/softmax_xent) — just looped over blocks with a flat latent
list + one AdamW state per latent.

## Later (0040-scale, not this increment)
Teacher-logit cache + a **real-model** whole-model tape (or a differentiable ModelRunner) so the
same loop runs on SmolLM2/Qwen at scale, with CPU-offload AdamW + gradient checkpointing. That is
the capstone execution (0040→0041); step 1 here proves the depth claim cheaply first.
