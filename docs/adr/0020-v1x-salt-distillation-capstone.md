# ADR 0020 — v1.x Capstone: SALT-Distillation Ternarization of a SOTA Model (binding public-release gate)

- **Status:** Proposed — **binding gate on the public v1.0 launch**
- **Date:** 2026-07-07
- **Deciders:** Brian Lam
- **Relates:** elevates the acceptance bar of [ADR 0012](./0012-v100-release.md) (v1.0 Release); realizes the "heal" step of [ADR 0001](./0001-salt-quantization.md) (SALT §6) as a full loop; executes the training economics of [ADR 0016](./0016-ternary-training-methods.md); consumes the 1.x reconstruction tooling (`tritium report salt-model`).
- **Reconciliation:** the tagged `v1.0.0` (2026-06-28) stands as the **frozen-API / infrastructure** milestone. It is **not** the public launch. This ADR makes the *public* release (crates.io publish + 1.0 announcement, task #45) **blocked** until the capstone below passes.

## Context

`v1.0.0` froze the API and proved the *infrastructure* on **BitNet-2B4T** — a model trained
ternary **from scratch**. It did **not** prove Tritium's actual thesis: that you can take an
existing fp16 **SOTA** model and turn it into a **ternary** model with **near-fp16 quality**
at a fraction of the VRAM. A ternary framework that can only run a model someone else trained
ternary has not earned a public "1.0".

The 1.x reconstruction tooling (`report salt-model`, sharded + mmap-streamed, parallel) shipped
and established the honest gap:

- Post-training quantization alone loses **~0.5 relative-Frobenius** at the ternary floor
  (measured on a real 27B bf16 master: `frob_rel ≈ 0.52`, `cosine ≈ 0.88` on the embeddings).
- Magnitude-based sensitivity (`Energy`) ≈ `Uniform` — a real **loss** signal is needed.

The missing piece is **training**: using the fp model as an oracle to distill quality back into
a SALT-parameterized ternary student. **Native from-scratch ternary training** of a 27B+ model
is out of reach at this stage (frontier-scale; see ADR 0016 §1 for the economics). **Distillation
from an fp oracle is not** — and is exactly what SALT's greedy residual planes + STE were built to
support.

## Decision

**The public v1.0 launch is BLOCKED until Tritium ternarizes a 27–35B SOTA model to near-fp16
quality via SALT-aware distillation.**

### Method — SALT-distillation

- **Oracle (teacher):** the frozen fp16 model. Its logits (top-k) and optionally hidden states
  are the distillation target, **cached offline** — the teacher may run in an external stack
  (HF/PyTorch), so Tritium need not host teacher-quality fp inference to *train* the student.
- **Ternary student:** the **same architecture**, every 2D weight held as an fp32 **latent
  master** `θ`. The forward **SALT-quantizes** `θ` → residual ternary planes `Σ_p s_p·t_p` and
  runs the multiply-free student. **SALT is the STE**: the backward treats the quantizer as
  identity, so gradients flow to `θ`; AdamW updates `θ` (optimizer states **CPU-offloaded** per
  ADR 0016 — for ternary, weights are 10× smaller than optimizer state, which flips the offload
  economics).
- **Distillation loss:** logit `KL(teacher‖student)` + hidden-state matching. **Layerwise first**
  (cheap, embarrassingly parallel — the `qat_heal_gate` pattern already drives a BitNet layer's
  ternary distillation loss down ≥90%), then **end-to-end** logit KL for final polish.
- **Adaptive capacity (the SALT lever):** periodically re-score **per-tile loss sensitivity**
  (grad-based diagonal Fisher) and **grow planes** where `error × sensitivity` stays high — spend
  bits only where accuracy degrades most. The student's **effective** ternary-plane count may grow
  well past the base parameter count (the "up to 50–70B" target — *effective* ternary params,
  still multiply-free, still ~1/5–1/8 the VRAM), while **average bpw stays low**.

### Goal (the promise)

**Near-fp16 quality at ternary cost.** Binding number: **≤ 1% perplexity delta** vs the fp16
teacher (and KL below a small bound) on a held-out set, at whatever **adaptive average bpw** SALT
needs — accept `~2.5–3.0` where accuracy demands it. Report the **full quality-vs-bpw curve**;
the ≤1% point is the headline. VRAM stays **~1/5–1/8 of fp16**.

## Scope of the target (decision, this ADR)

- **Binding gate arch = a standard-transformer 32B-class model** (dense SwiGLU / GQA / RoPE /
  untied head — a tractable, well-understood forward + STE backward). The method must hit ≤1% here.
- **Qwen3.6-27B (multimodal Mamba/SSM hybrid) is the headline extension**, pursued *after* the
  standard-arch gate passes; its `linear_attn` (A_log/dt_bias/conv1d) + attn-output-gate forward
  and STE backward are the larger, follow-on lift. (35B-a3b MoE is a further option.)
- **Rationale:** the exotic arch is the schedule risk. Proving the *method* on a standard SOTA
  model is ~90% of the value and de-risks the release; the hybrid then becomes a *demonstration*,
  not a *blocker*.

## Build cascade (milestone → gate)

1. **KEYSTONE — general inference engine.** A config-driven **architecture registry** + weight-name
   mapper for standard transformers (SwiGLU, GQA, RoPE variants, untied head, QK-norm). Loads fp
   *and* ternary; single-GPU (100B ternary ≈ 20 GB → fits one card). Unblocks running,
   teacher-caching, student forward, and real sensitivity. **This is the long pole (~1–3 mo).**
2. **SALT multi-plane inference loader.** Run what `quantize` emits (T=1..3 residual planes +
   multi-plane accumulate — the kernel exists; wire the model layer).
3. **SALT-aware distillation trainer.** Latent-master QAT with the SALT quantizer in the forward +
   STE; teacher-logit cache; KL + hidden loss; AdamW with CPU-offloaded optimizer (ADR 0016);
   gradient checkpointing to fit 32B on the available GPUs.
4. **Real sensitivity + adaptive plane growth.** Grad-based diagonal Fisher per tile →
   `Sensitivity::Custom`; a periodic plane-growth policy under a bpw/quality target.
5. **Scale plumbing.** Streaming `quantize` **writer** (today it buffers all rows in RAM — a wall
   past ~100B); streaming GPU weight load; multi-GPU only when the target exceeds one card.
6. **CAPSTONE run + gate.** Distill the 32B target to ≤1% ppl; report the curve + measured VRAM
   reduction + an end-to-end `load → infer` of the ternary student in Tritium. Then the headline
   **Qwen3.6-27B** demo.

## Testability (exit gates)

| Gate | How tested |
|------|------------|
| General inference: a standard 32B fp model loads + runs in Tritium; greedy matches HF (token-exact or logit rel < 1e-3) | conformance vs HF logits |
| SALT student loads + runs; multi-plane accumulate == dequant | kernel parity test |
| Distillation converges: student ppl within **1%** of fp16 teacher on a held-out set; KL below bound | e2e distill on a real 32B |
| VRAM: ternary student inference footprint ≤ ~**1/5** of fp16 (measured) | measured on GPU |
| Curve: quality-vs-bpw (1.58 → ~3) reported; headline ≤1% at ≤3 bpw | benchmark report |
| Headline: Qwen3.6-27B ternary demo runs (SSM hybrid, text backbone) | e2e (follow-on) |

## Definition of done — public v1.0 launch unblocked

- [ ] A standard 32B SOTA model ternarized via SALT-distillation to **≤1% ppl** vs its fp16 teacher, at adaptive bpw; full quality-vs-bpw curve reported.
- [ ] The ternary student runs **end-to-end in Tritium** (general inference engine + SALT multi-plane loader); VRAM reduction **measured** (~1/5–1/8).
- [ ] **Qwen3.6-27B** headline extension demonstrated (the stated goal; may trail the binding gate as a stretch demo).
- [ ] All prior v1.0 gates (ADR 0012) still green.
- [ ] **THEN**: crates.io publish + 1.0 announcement (task #45).

## Non-goals (explicit)

- **From-scratch ternary pretraining** at any scale — frontier-lab territory, not a v1 deliverable.
- **800B** as a gate — the tooling *reads/streams* toward it, but the binding target is 27–35B;
  800B serving (multi-GPU) and 800B distillation are post-capstone.
- **Full multimodal** (vision tower) of Qwen3.6 — text backbone only for the demo.

## Risks / long poles (honest)

- **The keystone gates everything** and is ~1–3 months of real inference-engine work.
- **Custom-autograd distillation of 32B** — memory (gradient checkpointing) + throughput; the
  CPU-offloaded optimizer (ADR 0016) is load-bearing to fit optimizer states.
- **1.58 → near-fp16 is aggressive.** The ≤1% bar likely lands at **~2.5–3 bpw** (still ~1/5 VRAM).
  The **curve is the honest artifact**; a single-point ≤1%-at-1.58-bpw promise would be dishonest.
- **Compute:** distilling a 32B for enough tokens = real GPU-hours — likely the rented boxes
  (Thunder / Hot Aisle MI300X / RunPod, per prior fenced sessions), not just the local 4090.
- **Qwen3.6 SSM/MoE** is a genuine research + systems lift; keeping it **off** the binding gate
  (headline, not blocker) is the schedule insurance.

## Amendment 2026-07-15 — Qwen3.6-27B is the active binding capstone

This amendment replaces the standard-transformer-32B-first execution order above. The earlier
scope and cascade remain as decision history, but a standard 32B is now an optional confirmation
model only: it is neither the active target nor a prerequisite for the public-launch gate.

The active binding capstone is `Qwen/Qwen3.6-27B` at immutable revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, as defined by its pinned
[model card](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md),
[configuration](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json),
and [weight index](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/model.safetensors.index.json).
Those sources supersede the earlier Mamba/SSM/MoE shorthand: the selected checkpoint has a dense
`qwen3_5` text core with a repeating three-Gated-DeltaNet-to-one-full-attention schedule and one
declared MTP hidden layer.
The first binding artifact covers the language core and bundled one-layer MTP drafter. The vision
encoder and multimodal integration remain the product end state, but are deferred until that
language-plus-MTP scope passes its architecture, identity, quality, accounting, and runtime gates.

PTQ and refined conversion are separate experiment tracks. They must retain separate artifacts,
recipes, costs, and claims; refined results must not be reported as PTQ. This target change does
not relax the inherited capstone gates: the qualifying result must still demonstrate **≤ 1%
perplexity delta** against the pinned source model, publish the full quality-versus-physical-bpw
curve, run end to end in Tritium, and report measured matched-fp16-versus-ternary inference memory
rather than a logical-bit estimate. All applicable provenance, parity, runtime, and publication gates in
[ADR 0028](./0028-salt-v2-additive-ternarization.md) also apply.

The active operational sequence is [plan 0043](../plans/0043-salt-v2-sota-campaign.md). Engineering
and evidence collection are local-first. This amendment authorizes no paid compute; any rental or
cloud run requires explicit future approval of a frozen recipe, cost ceiling, stop gate, and
validated resume path. It marks no empirical gate complete.
