# 0041 — Capstone: scale SALT distillation to a 32B standard transformer (ADR 0020 binding gate)

## Where we are
The **method is complete and proven on a real model** (0035→0040 step 4): the ModelRunner-validated
differentiable forward + SALT-STE latents + Fisher sensitivity + AdamW distillation took SmolLM2-135M
from catastrophic PTQ (ppl 1.6e5) back to ~27 in-sample. Nothing algorithmic remains. **0041 is a
scale + hardware task**: run the same loop on a standard-transformer **32B** against a real corpus,
on rented GPU, to the ADR-0020 gate: **≤1% perplexity vs fp16 at adaptive bpw**.

## The gate
Standard-transformer 32B (dense Qwen3-32B / Llama / Mistral class — NOT Qwen3.6, which is
linear-attention; see salt-recon-tooling-and-qwen36-reality). Held-out ppl within 1% of the fp16
teacher, at an adaptive (Fisher-allocated) average bpw. Qwen3.6-27B stays a stretch headline (needs
linear-attention layers + a fetched fp master).

## What scale demands (and the memory math)
For 32B params, fp32 state is the wall:
- latent master (fp32): **128 GB**; AdamW `m`+`v` (fp32): **256 GB**; gradients (fp32): **128 GB**.
- The *quantized* weights (what the forward reads) are tiny: ternary at 1.58–4.75 bpw ≈ **6–19 GB**.

So the ~512 GB of latent+optimizer+grad state cannot live on one 80 GB GPU. The design (ADR 0016):
**keep latent + Adam state on the host**, stream one layer at a time to the GPU, quantize on-device,
forward/backward on-device with gradient checkpointing, send the wgrad back to host, AdamW-step on
host. GPU holds only the resident layer + activations for a microbatch.

## Lossless training-phase optimizations (the "don't lose anything" ask)
Everything here is **bit-identical / exact** — no accuracy traded:

1. **Row-parallel dense matmul — DONE this session** (`ops::dense` forward+vjp, rayon over output
   rows; each element's k-accumulation order preserved → bit-identical, verified by the ModelRunner
   conformance gate at unchanged 1.85e-6). The training hot path; ~N-core on CPU, and the pattern
   the GPU kernels already follow.
2. **Gradient checkpointing** — recompute each block's forward during backward instead of storing all
   activations. Exact same gradients; turns activation memory from O(layers) into O(1) resident. The
   current tape stores every value — 0041 adds a checkpointed block boundary.
3. **CPU-offload AdamW** (ADR 0016) — latent + `m`,`v` on host, streamed per layer; the optimizer
   math is unchanged. This is what makes 32B fit at all.
4. **Multiply-free / fused SALT GEMM** — the forward `Ŵ·x` and the dgrad `gY·Ŵ` go *through* the
   ternary weight `Ŵ = Σ_p s_p·t_p`. Computing `Σ_p s_p·(t_p · x)` directly is (a) **bit-identical**
   (the scale distributes exactly over the plane sum), (b) avoids materializing the dense `Ŵ`
   intermediate (huge at 32B — GBs per weight), and (c) **multiply-free** (add/sub/skip; DP4A/IMMA on
   GPU). Only the wgrad `gYᵀ·x` stays fp (activations are continuous). This is the core ternary
   *compute* advantage; the current tape dequants-to-dense and does an fp matmul, leaving it on the
   table. (Three-regimes: training is compute-bound, ~2× in the backward from the multiply-free dgrad.)
5. **Zero-skip sparsity** — ternary planes carry ~⅓+ zeros; a sparse ternary GEMM skips them (0·x=0,
   exact). Tritium's kernels don't yet (ternary-superiority-conditional-on-sparsity) — a real,
   lossless kernel win for both forward and dgrad.
6. **Streaming quantize writer** — quantize + shard-write without holding the whole model in RAM
   (the mmap-streaming report path already proves the pattern).
7. **Teacher-logit cache** — run the fp teacher over the corpus once, store soft logits (full or
   the exact top-k that covers ≥1−ε mass — the *exact* tail can be kept if losslessness is required).

Explicitly **excluded** (they DO lose something): top-k distillation that drops tail mass, bf16/fp16
*latent* master (less gradient precision), stochastic-rounding the master, gradient compression.

## SALT vs a lesser STE (why multi-plane helps convergence, not just capacity)
Single-plane STE (T=1, plain BitNet b1.58) quantizes `W→s·trit(W/s)` — a coarse staircase with a
large `|Ŵ−W|`. SALT's residual planes (T≥2) halve the error per plane, so:
- **the forward starts far less broken** — measured: SmolLM2 PTQ ppl **T=1 ≈ 1.09e8 vs T=2 ≈ 1.6e5**
  (~670× less catastrophic before any training);
- **the STE surrogate is less biased** — the straight-through `∂Ŵ/∂W≈I` is exact only where `Ŵ=W`;
  a smaller `|Ŵ−W|` means the surrogate gradient is applied to a weight closer to the one actually
  used, so updates are less biased → faster, more stable convergence;
- **at a fixed step/bpw budget**, Fisher-adaptive SALT (0039) spends the extra planes only where the
  loss is sensitive — beating both uniform-T and single-plane at equal average bits.
**Empirical sweep** (SmolLM2, distill 60 steps, in-sample 12-tok calib):

| planes | PTQ ppl (pre-train) | final distill KL (xent) | distilled ppl |
|---|---|---|---|
| T=1 (lesser STE) | **1.086e8** | 3.90 | 27.30 |
| T=2 (SALT) | **1.616e5** | 3.49 | 25.12 |
| T=3 (SALT) | 1.783e5 | **3.19** | 38.64 |

Reading it: (a) single-plane PTQ is **~670× more catastrophic** than 2-plane SALT before any training
(1e8 vs 1e5) — the clearest SALT win; (b) the training KL is **monotone-better with more planes**
(3.90→3.49→3.19) — the STE is a better estimator as `|Ŵ−W|` shrinks; (c) final in-sample hard-ppl at
60 steps is noisy (T=2 best; T=3's lower KL doesn't map to lower hard-ppl on one short sequence — the
soft-KL objective and hard-token ppl diverge under overfit). Net: **yes, SALT beats a lesser STE** —
better starting point + better convergence — at a bit cost the Fisher allocator (0039) then spends
where the loss is sensitive.

## Steps
1. **Gradient checkpointing** on the tape (block-boundary recompute) — gate: identical grads vs the
   non-checkpointed tape on SmolLM2, lower peak memory.
2. **Multiply-free fused SALT GEMM** op (forward + STE dgrad; wgrad fp) — gate: bit-identical to
   `dense_matmul(salt_quantize(latent), x)`, no dense-`Ŵ` allocation; benchmark the multiply-free win.
3. **CPU-offload AdamW + per-layer streaming** driver — gate: a multi-layer model trains with the
   optimizer state on host, loss-parity vs the all-resident run.
4. **GPU port + provisioning** (Thunder Compute) — the resident forward/backward on device; wire the
   real corpus + teacher-logit cache; short SmolLM2 GPU run for parity.
5. **The 32B capstone run** — distill a standard-transformer 32B against the corpus to ≤1% held-out
   ppl at adaptive bpw. Report bpw, ppl, VRAM, wall-clock.

## Verification
Each infra step gated by bit-identity / loss-parity vs the validated CPU path before it touches the
GPU. Every commit reviewed (code-reviewer subagent), explicit-path staging (shared tree), pushed.
