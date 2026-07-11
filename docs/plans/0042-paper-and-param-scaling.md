# 0042 — Paper + parameter-scaling (byte-optimal ternary via SALT-distillation)

## Thesis
Ternary models are **Pareto-dominant over fp16 on quality-per-byte**: SALT-distillation recovers a
ternary model from catastrophic PTQ, and **parameter-scaling** (a *larger* ternary model, still
smaller in bytes) makes up the residual ternarization loss. The goal is not lossless 1:1
ternarization (hard) — it's the ternary frontier beating the fp frontier at equal or fewer bytes.

## Why it works (information-theoretic + empirical)
- A ternary weight carries `log₂3 ≈ 1.58` bits; fp16 wastes most of its 16. BitNet-family scaling
  laws: ternary needs only **~2–3× params** to match fp quality — and 2–3× params at ~1.58–2 bpw is
  still **~3–4× fewer bytes** than fp16. So "grow then ternarize" wins on memory while matching quality.
- Measured this session: SALT (T=2) starts ~670× less broken than single-plane and converges to a
  lower KL (better STE); end-to-end distillation recovers SmolLM2 from PTQ 1.6e5 → 26 ppl; Muon
  matches/beats AdamW convergence at half the optimizer state.

## Two levers for "more params"
1. **Bits — BUILT:** SALT Fisher-adaptive plane growth (0039). `T=1→2→3` adds ternary parameters
   (planes) where the loss is sensitive. Fine-grained; the adaptive-bpw knob.
2. **Neurons — NEW (grow-then-ternarize):** function-preserving width/depth expansion (net2net /
   layer-stacking / LiGO) turns fp-`N` → fp-`kN` *warm-started* (quality preserved), then the
   **existing 1:1 SALT-distillation** ternarizes it. Cheap (warm start, not from-scratch), reuses
   the whole 0040 stack, sidesteps random-init big-student training. **The key new experiment.**

## Contributions (paper)
1. SALT — sensitivity-allocated layered ternary (residual planes, per-block AbsMean).
2. Fisher-adaptive plane growth — allocate bits by loss curvature, not magnitude (beats Energy≈Uniform).
3. Distillation-recovery — end-to-end SALT-STE distillation defeats catastrophic ternary PTQ + depth.
4. Grow-then-ternarize — parameter-scaling for byte-optimality; the ternary frontier dominates fp.
5. Systems — the differentiable tape (ModelRunner-validated), lossless row-parallel training, Muon
   at half the optimizer memory, CPU-offload/streaming for 32B.

## Experimental arc (small → up); the ASAP figure first
**Money plot = quality vs BYTES: ternary frontier vs fp frontier.** Draw it small first.
1. **NOW (CPU/small):** held-out corpus eval + multi-sequence distillation (the blocker for ANY
   publishable number — everything so far is in-sample toy). Then the **SmolLM2 family curve**
   (135M/360M/1.7B fp, all on disk): SALT-distill each; show a *bigger ternary* model beating a
   *smaller/equal-byte fp* model on held-out ppl (e.g. ternary-360M ≈ 68 MB vs fp-135M ≈ 270 MB).
2. **Grow-then-ternarize prototype (small):** expand a small fp model 2×, SALT-distill, show it
   recovers quality and beats the un-grown ternary at equal-ish bytes.
3. **Standard dense 32B (rented GPU, ~$1–8k):** the "SOTA 32B" milestone → ≤1% ppl at adaptive bpw.
4. **Stretch — Qwen3.6-27B:** needs linear-attention (SSM) layer support (48/64 layers) + a 54 GB
   fp master fetch. A real arch project; future-work / v2, not the headline.

## The one gap blocking every real number
All results to date are **in-sample** (one 12-token sequence). Step 1's held-out corpus + multi-seq
distillation is prerequisite for the paper. Needs: a tokenized held-out set (Python tokenizer →
JSON token ids, like `smollm2_ref.json`) + a train/eval split + multi-sequence distillation loop.

## Risks
- Grow-then-ternarize init: does function-preserving growth + SALT-distill converge cleanly? (Step 2
  tests it small.) Main research risk — if it lands, the whole param-scaling story lands.
- Qwen3.6 linear-attention is a separate build; don't gate the paper on it.
- Held-out generalization vs in-sample: the real token budget (0041) is still unmeasured.
