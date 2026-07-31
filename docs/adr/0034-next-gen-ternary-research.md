# ADR 0034 — Next-generation ternary: research intake and method upgrades (July 2026 frontier)

Status: **PROPOSED** (2026-07-30)

- **Deciders:** Brian Lam
- **Research cutoff:** 2026-07-30, inclusive (extends ADR 0028's 2026-07-14 cutoff; nothing here
  reopens ADR 0028's frozen gates — see "Preregistration discipline" below)
- **Relates:** feeds the [ADR 0028](./0028-salt-v2-additive-ternarization.md) /
  [plan 0043](../plans/0043-salt-v2-sota-campaign.md) SALT V2 campaign (method upgrades for its
  PTQ and refined tracks, one new preflight gate); updates the competitive picture behind
  [ADR 0026](./0026-sota-campaign.md) and [ADR 0033](./0033-v11-full-public-release.md); revises
  the drafter urgency of [ADR 0021](./0021-drafter-architecture.md)/[ADR 0032](./0032-spec-decode-cost-model-and-next-levers.md);
  leaves [ADR 0019](./0019-persistent-megakernel-decode.md)'s deferral and
  [ADR 0024](./0024-structured-24-ternary.md)'s checkpoint gate intact.

> **Claim boundary.** This ADR records research findings and proposes method adoptions. It marks
> no empirical gate complete. Vendor-published numbers are labeled vendor-reported unless an
> independent reproduction is cited. Externally sourced claims went through a 108-agent
> adversarial-verification run (deep-research `wf_c798252e-234`, 3-vote per claim): 12 finding
> groups survived, one was refuted outright (MoTE's iso-memory number), and — notably — **no
> claim strictly newer than 2026-07-14 survived verification**; everything here qualifies under
> the "missed by prior cutoff" branch plus the direct-fetched July releases (Bonsai 27B,
> Neutrino-1, llama.cpp Q2_0, vLLM v0.26), whose *existence* is verified even where their
> vendor tables are not. Serving-stack and rate-distortion axes returned no surviving
> workflow claims and are covered here by direct primary-source fetches only.

## Context: six weeks moved the field

Since ADR 0028's cutoff (2026-07-14), four things happened that change Tritium's plans, and a
fifth that confirms them:

1. **Two ternary conversions of Tritium's exact flagship shipped.** PrismML's Bonsai 27B
   (2026-07-14) is a ternary + a 1-bit build of **Qwen3.6-27B** — the same pinned base as plan
   0043 — with vendor-reported 94.6% quality retention at 5.9 GB, and this time a first
   *independent* reproduction exists (see §2.1). Stage 8 no longer has only baselines to
   reproduce; it has a shipped competitor artifact on the same base model to beat.
2. **The ternary-drafter first was claimed.** Fermion Research's Neutrino-1 (2026-07-27) ships a
   natively-trained ternary 8B **plus a 0.6B certified ternary drafter** with bit-exact drafted
   decode (27,648 consecutive tokens verified; vendor-reported 396→763 tok/s on H100). The
   ternary-draft→ternary-target novelty window ADR 0021 targeted is now partially closed —
   what remains open is tree-verify + budget control (they do single-path argmax-match).
3. **The mainstream ternary GPU gap is closing.** llama.cpp adopted an official ternary
   **Q2_0 g64** format — CPU/Metal/Vulkan backends merged, CUDA in flight (PR #25707,
   July 2026). vLLM v0.26 merged Humming **w[2-7]a[4,8]** weight-only inference and mainlined
   DFlash/DSpark drafters. The serving-gap window ADR 0026 flagged is measurably shrinking.
4. **The method frontier moved past Tequila and PT²-LLM.** HESTIA (differentiable Hessian-guided
   QAT) beats Tequila by ~2.5-2.8 points at 10× fewer tokens; CAT-Q (Intel) reaches QAT-class
   ternary quality by PTQ with 512 calibration samples up to 235B-parameter MoE. Curvature
   estimation — the expensive step in SALT V2's capture pipeline — got ~10× cheaper via
   Kronecker factorization (KronQ, Fisher-Kronecker ACL 2026).
5. **Theory converged on SALT V2's bet.** Ordentlich–Polyanskiy's quantized-matmul series
   formalizes that the right objective is **matrix-product distortion, not weight MSE** — the
   information-theoretic frame for ADR 0028's output-aware objective — and locates exactly where
   ternary is near-optimal vs where lattices win.

## Findings (grouped, with adoption verdicts)

### 1. Quantization methods

#### 1.1 CAT-Q — PTQ ternarization at QAT quality (adopt techniques into the 0043 PTQ track)

[arXiv 2606.26650](https://arxiv.org/abs/2606.26650), Intel China; code
[IntelChina-AI/BitTern](https://github.com/IntelChina-AI/BitTern).

- **Learnable Modulation:** three learnable factors (δμ, δα, δΔ) refining mean/scale/threshold
  per group before ternarization — a distribution-alignment step SALT V2's Asymmetric fit lacks.
- **Softened Ternarization relay:** stage 1 optimizes a differentiable
  `f(W;s,Δ) = [tanh(s(W−Δ)) + tanh(s(W+Δ))]/[2·tanh(s)]` with annealed sharpness; stage 2
  (last 20%) hard-ternarizes. Optimal γ=0.8, s₀=30.
- **Sliding-layer output reconstruction:** optimizes multiple layers jointly against output
  error instead of layer-wise weight reconstruction — the same "neighboring layers absorb each
  other's error" argument as plan 0043's feedback stage, validated independently.
- Reported: Qwen3-4B ternary at 57.06% avg with ~1M tokens vs BitNet-class 54.16% at 100B
  tokens; scales to Qwen3-235B-A22B (60 h on 8×A100); W1.58A16 beats SliderQuant/OmniQuant/GPTQ
  W2A16 on Llama2-7B. Vendor-reported; code public.

**Verdict:** the three techniques compose with additive planes and are objective-compatible with
ADR 0028 (output-aware, zero-point-free — the learnable μ folds into the pre-quant transform, not
the stored format). Adopt as *candidate refinements inside the 0043 PTQ track's existing stages*,
each behind its own ablation gate. CAT-Q replaces PT²-LLM as the strongest reproduced-baseline
candidate for Stage 8's baseline set.

#### 1.2 HESTIA — differentiable Hessian-guided QAT (adopt for the 0043 refined track)

[arXiv 2601.20745](https://arxiv.org/html/2601.20745); code
[hestia2026/Hestia](https://github.com/hestia2026/Hestia).

Replaces STE with a temperature-controlled softmax relaxation over {−1,0,+1} (weights become
expectations over soft state assignments; τ→0 recovers hard ternary), and — the actual
innovation — schedules per-tensor temperature by **Hessian-trace sensitivity** (Hutch++):
sensitive tensors stay soft longer. Llama-3.2-1B/3B: 0.547/0.601 avg vs Tequila's 0.519/0.576
at the same 10B tokens; competitive with 100B-token ternary baselines. Also validated on the
Fairy2i complex-ternary codebook — evidence the scheduling transfers across codebooks,
including plausibly SALT's plane-sum.

**Verdict:** the 0043 refined track's polish stage currently specifies STE-style hard-trit
refinement. Adopt HESTIA-style relaxation + sensitivity-scheduled annealing as the *candidate
mechanism* for that stage (the curvature artifacts Stage 2 already captures are exactly the
sensitivity input it needs). Frozen quality gates unchanged; this is a mechanism swap behind
the same gates. Tequila's deadzone-bias framing (ADR 0026 Tier 1 item) is superseded — the
deadzone is handled by the relaxation itself.

#### 1.3 Kronecker-factored curvature — 10× cheaper capture (adopt in the capture pipeline)

KronQ ([arXiv 2607.07964](https://arxiv.org/abs/2607.07964)) and Fisher-Guided Quantization via
Kronecker Factorization ([ACL 2026](https://aclanthology.org/2026.acl-long.1805/)) both replace
YAQA-style power iteration with Kronecker-factored Hessian/Fisher approximations at ~10× lower
cost, with near-baseline 4-bit and ~5-6% degradation at 2-bit quality retention.

**Verdict:** SALT V2's Qwen curvature-capture pipeline (the July 22 commit series) is the direct
consumer. Evaluate Kronecker factorization as the capture backend before the 27B campaign run —
at 27B, a 10× curvature-cost reduction materially changes what fits on local hardware
(plan 0043's local-first spend policy).

#### 1.4 Compute-optimal QAT scheduling (adopt as planning input, refined track)

Apple's Compute-Optimal QAT ([arXiv 2509.22935](https://arxiv.org/abs/2509.22935)): the optimal
QAT fraction of a training budget is not fixed — it **grows with tokens-per-parameter-byte** and
is predictable from that statistic; fusing LR cooldown with QAT saves redundant FP updates.
Schedule×bit-width interaction mapped for sub-100M models in
[arXiv 2605.25966](https://arxiv.org/abs/2605.25966).

**Verdict:** adopt the tokens-per-parameter-byte statistic as the budget-planning input for the
0043 refined track and any future BLUT drafter training run. No gate change.

#### 1.5 Watch items (no action)

- **Bit-by-Bit** progressive QAT + outlier channel splitting reaching ternary
  ([arXiv 2604.07888](https://arxiv.org/abs/2604.07888)) — overlapping coverage with 1.1/1.2.
- **BitRL** ([arXiv 2604.24273](https://arxiv.org/abs/2604.24273)) — RL post-training works
  directly on 1-bit models; relevant only when a Tritium-trained ternary model needs alignment.
- **LittleBit** ([arXiv 2506.13771](https://arxiv.org/abs/2506.13771), NeurIPS 2025) — 0.1 bpw
  via low-rank binarized factorization; the sub-ternary frontier. Violates the additive-ternary
  execution invariant (dense low-rank matmuls); not adoptable, but the matched-bytes comparison
  belongs in the paper's related work.
- **D2Quant / BWLA / RobuQ / VBQ** — adjacent PTQ advances (2-bit dual-scale + LayerNorm bias
  correction; W1AX binary; W1.58A2 for DiTs via Hadamard-normality; learned per-group
  precision). Mine for the Stage 8 baseline set as needed.

### 2. Competitive landscape

#### 2.1 Bonsai 27B — the competitor on Tritium's flagship (new Stage-8 reference)

[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf),
2026-07-14, Apache 2.0. Ternary (1.71 true bpw, 5.9 GB, Q2_0 g128) + 1-bit (1.125 bpw, 3.9 GB)
builds of Qwen3.6-27B via QAT + reasoning-trace distillation. Vendor-reported: thinking-avg
80.49 = 94.6% of FP16 (ternary); H100 decode 98 tok/s, +1.34× with a bundled Q4_1 **DSpark
drafter**; 4-bit KV on the 16 full-attention layers; vision tower HQQ-4-bit; **the ternary
representation is applied uniformly across the hybrid backbone including the 48 Gated-DeltaNet
layers**.

Unlike the Bonsai-8B case (whose vendor table failed verification in the mid-2026 sweep), a
first independent reproduction exists:
[Astezelex/bonsai-27b-16gb-bench](https://github.com/Astezelex/bonsai-27b-16gb-bench) — on a
16 GB RTX 5060 Ti, ternary Bonsai beats Qwen3.6-27B UD-IQ2_XXS at matched VRAM (AIME26@60k
0.867 vs 0.633; MMLU-Redux 0.871 vs 0.860; LCB 0.520 vs 0.300; 44.4 vs 35.8 tok/s; ~2.5× less
energy per solved problem), with raw eval JSONs published. Its **score@budget** methodology
(accuracy + cap-rate + stated token budget, because thinking models fail by not-knowing OR
not-converging) is the right evaluation frame for thinking-mode ternary models.

**Verdicts:**
- Stage 8's baseline set gains a mandatory member: reproduce Bonsai-27B-ternary under the 0043
  harness. The SALT V2 quality gates stay frozen; "beat/tie Bonsai at no greater artifact
  bytes" is *additionally* what the additive-ternary SOTA gate now concretely means on this
  flagship.
- Adopt score@budget reporting (accuracy, cap-rate, budget) for all 0043 thinking-mode evals.
- Bonsai's uniform ternarization of the GDN layers is an existence proof that **QAT/distillation
  absorbs linear-attention quantization error that PTQ practice avoids** — see §3.1.

#### 2.2 Fermion Neutrino-1 — the ternary-drafter first, and entropy-coded transport

[fermionresearch.com/research/neutrino-8b](https://www.fermionresearch.com/research/neutrino-8b/),
2026-07-27. Natively-trained ternary family (8B + 0.6B); proprietary coded container: ternary
lane losslessly compresses to ~55% of raw bytes (62.6% zeros measured), 3.88 GB disk / 2.56 GB
download; weights decoded inside the matmul kernels, never materialized fp; **0.6B certified
drafter** with argmax-match acceptance, drafted decode verified bit-exact over 27,648 tokens;
vendor-reported MMLU 72.1, H100 396→763 tok/s drafted.

**Verdicts:**
- ADR 0021's novelty claim narrows: "first ternary draft-of-ternary-target" is taken. What
  remains defensible and unclaimed: **tree-verified** (BASTION budget-controlled) ternary
  spec-decode, and the acceptance-theory-grounded drafter objective (§4.2). Update ADR
  0021/0032 framing accordingly.
- Entropy-coded *transport* is now field-proven at ternary sparsity levels. Plan 0043 already
  scopes ANS/rANS as seekable outer transport with expanded-bytes accounting — promote that
  from "benchmark candidate" to a planned deliverable (the ~45% download saving is free
  distribution value and does not touch the runtime format or the bpw accounting rules).

#### 2.3 Ecosystem: the format is standardizing without Tritium

- llama.cpp official ternary **Q2_0 g64** ([discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019),
  CPU [PR #24448](https://github.com/ggml-org/llama.cpp/pull/24448) merged, CUDA
  [PR #25707](https://github.com/ggml-org/llama.cpp/pull/25707) in flight). Group-64 chosen
  over 128 by maintainer for shape coverage + quality at <6% memory cost. The ecosystem now has
  THREE ternary GGUF layouts in the wild: TQ1_0/TQ2_0 (g256), Q2_0 g64 (official), Q2_0 g128
  (Prism fork).
- vLLM v0.26: Humming w[2-7]a[4,8] weight-only merged ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.26.0))
  — W2 storage covers ternary values without ternary-aware kernels; DSpark spec-decode
  mainlined; hybrid (SWA+full) DFlash drafters.
- CPU: Litespark ([arXiv 2605.06485](https://arxiv.org/abs/2605.06485)) joins Vec-LUT/T-MAC
  (NEON/VNNI/AMX). Microsoft shipped BitNet-*embedding* models (0.6B/270M, 2026-07-20) and
  VibeASR.cpp — BitNet is expanding sideways (embeddings/ASR), not upward; no larger native
  BitNet.

**Verdicts:**
- Add a **Q2_0 g64 read path** (import) and an **export path** for SALT V2 CompactV1 artifacts
  → Q2_0 g64 where the profile permits (it is a plain grouped ternary layout — the additive
  P=1 prefix maps directly). Rationale: every quality result Tritium publishes becomes
  runnable in the ecosystem's standard runtime, which is how method wins convert to adoption.
  Group-size crosswalk (256↔64) is scale re-fitting, already solved machinery.
- The ADR 0026 assumption "no mainstream ternary CUDA" expires when PR #25707 merges. The
  defensible moat shifts fully to: quality method (SALT V2), training, tree-verify spec-decode,
  and the evidence-ledger discipline.

### 3. Architecture co-design

#### 3.1 Linear-attention (GDN) quantization sensitivity — new preflight gate for 0043

Four independent signals, and they **conflict** — which is exactly why this must be a measured
gate rather than an assumption:
- **Ternary Mamba** ([arXiv 2606.18114](https://arxiv.org/abs/2606.18114)): W1.58 QAT of
  Mamba-2 works (744 MB from 2,687 MB, 48.1% avg, 102M tokens), but (a) **post-hoc corrections
  that work on Transformers fail on SSMs — quantization error accumulates through the
  recurrence**; (b) "zero-ratio collapse," a learnable-scale instability specific to
  QAT-from-pretrained-checkpoint (SALT V2's exact regime).
- **Industry PTQ practice on Qwen3.6:** NVIDIA-ecosystem NVFP4 conversions
  ([vrfai/Qwen3.6-27B-NVFP4](https://huggingface.co/vrfai/Qwen3.6-27B-NVFP4)) quantize FFN +
  full-attention projections but **keep all 48 DeltaNet projections in BF16**, explicitly for
  recurrence-precision reasons.
- **Bonsai 27B** ternarizes those same layers uniformly — with QAT + distillation — at 94.6%
  vendor-reported retention (independently supported at the benchmark level, §2.1).
- **VBQ's learned-allocation probe** ([arXiv 2607.02893](https://arxiv.org/abs/2607.02893),
  verified 3-0, medium confidence): on a Qwen3.5-style hybrid, the learned per-group allocation
  finds **DeltaNet projections the MOST compressible tensors in the model** — `in_proj_qkv`
  selects 1 bit for 95.7% of its groups — while value and MLP up/down projections are the
  precision-hungry classes. (Caveat: a deliberately short 1B from-scratch probe, not a
  27B conversion.) Same paper: a frequency-tiered mixed-precision LM head (mean 1.18 bits) was
  the single most impactful allocation choice (+0.6 PPL if made uniform) — relevant to SALT
  V2's allocation stage, where the embedding/head are in scope.

Synthesis: the *weight matrices* of GDN blocks may be highly ternarizable (VBQ, Bonsai) even
while the *recurrence dynamics* punish PTQ-style per-layer correction (Ternary Mamba) and
industry PTQ practice avoids them entirely (NVFP4 conversions). These are compatible if the
risk lives in error *propagation through the state*, not in weight representation — precisely
what a recurrence-divergence probe (not per-layer MSE) distinguishes. This is currently
invisible to plan 0043's uniform 506-matrix scope.

**Verdict (the one new gate this ADR proposes):** add a **GDN-sensitivity preflight** to plan
0043 before Stage 8: sliced-layer PTQ probes on DeltaNet-block matrices vs full-attention-block
matrices at matched bpw, reporting divergence growth *along the recurrence* (not just per-layer
output MSE, which Ternary Mamba shows is the wrong proxy here). Outcome routes the campaign: if
PTQ divergence on GDN blocks exceeds the frozen threshold, the PTQ track records it honestly
(per "no is a valid result") and the refined track carries those tensors; the unqualified
language+MTP ternarization claim is unaffected either way because coverage policy is unchanged
— only the *evidence expectations* per tensor class are.
Watch: zero-ratio collapse mitigation when learnable scales enter the refined track.

#### 3.2 MoE: ternary routed experts (architecture recipe only — quantitative claim REFUTED)

MoTE ([arXiv 2506.14435](https://arxiv.org/abs/2506.14435), lead author Hongyu Wang of BitNet):
during sparse up-cycling, keep the pre-trained FFN as a full-precision shared expert and train
ALL routed experts ternary — more low-precision experts instead of fewer high-precision ones.
The architectural recipe verified 3-0; **its headline iso-memory win (+4.3% avg over
MoE-LLaVA at matched 3.4 GB) was REFUTED 0-3 in adversarial verification — do not cite it**.
The recipe (not the number) aligns with the grow-then-ternarize thesis and is citable as
convergent architecture-level thinking in plan 0042's paper. Changes nothing in 0043.

#### 3.4 Structured sparsity: 2:4 is the wrong ratio — SlideSparse re-targets ADR 0024

SlideSparse (MSR BitNet group — Huang, Dong, Wei; [arXiv 2603.05232](https://arxiv.org/abs/2603.05232),
**ICML 2026, peer-reviewed**; code [bcacdwk/vllmbench](https://github.com/bcacdwk/vllmbench)),
verified 3-0 on method and speedups:

- Strict 2:4 (50% pruning) collapses BF16 Qwen3 reasoning 54%→15% under identical fine-tuning;
  6:8 retains 51.6%. (2-1 on the collapse framing; measured on BF16, not ternary.)
- Companion Sparse-BitNet ([arXiv 2603.05168](https://arxiv.org/abs/2603.05168), same group):
  **ternary tolerates 2:4 far better than BF16 (+5.7% vs +18.8% degradation) but still
  degrades — and the BitNet team themselves chose 6:8 over 2:4** for sparse training.
- The enabling trick: Sliding Window Decomposition **losslessly** (their Theorem 1) re-expresses
  any (2N−2):2N block as N−1 overlapping 2:4-compliant windows — milder ratios like 6:8 (25%
  pruning) execute on existing NVIDIA sparse tensor cores. Measured 1.33× end-to-end on
  Qwen2.5-7B at 6:8 (A100 INT8, matching the N/(N−1) bound); 1.18-1.19× on RTX 4090 FP8;
  validated across 4090/5080/A100/H100/B200 with vLLM integration including BitNet models.

**Verdict:** ADR 0024's target ratio changes from 2:4 to **6:8-via-SlideSparse**. The measured
BitNet-2B4T 4-group census in ADR 0024 (only 12.9% of groups have 0 zeros) is *more* compatible
with 6:8's 25% pruning than with 2:4's 50%. The checkpoint gate stays: no kernel work before a
sparsity-trained checkpoint exists; the plan-0039 trainer hook should now target 6:8 masks.
Note this partially rehabilitates the *direction* of the refuted spbitnet claims — with a
peer-reviewed mechanism and honest ratios.

#### 3.5 KV/state precision: structure-aware allocation, and the SSM-state frontier

- **Block-GTQ** ([arXiv 2606.24033](https://arxiv.org/abs/2606.24033), CUHK, code released;
  verified 3-0): at a *matched* 2-bit KV budget on Llama-3.1-8B, **RoPE-aware non-uniform bit
  allocation** raises NIAH from 70.6→97.4 (fp16 99.6) and LongBench-EN 36.87→53.31 vs uniform
  allocation. Structure-aware allocation, not more bits, is what makes ~2-bit KV viable —
  directly applicable to ADR 0020's KV ladder rungs (the rejected uniform ternary-KV rung t2 is
  consistent with this: uniformity, not bit count, was plausibly the failure).
- **DF-SSM** ([arXiv 2606.10932](https://arxiv.org/abs/2606.10932); medium confidence): proposes
  **binarizing the SSM hidden state itself** (K×K binary field per state element, σ-δ error
  feedback, popcount readout at chunk boundaries) — the recurrent-state analogue of KV
  quantization for DeltaNet-style layers. Watch item for the Qwen3.6 GDN state memory story.

#### 3.3 Gated DeltaNet-2 (watch)

NVIDIA's GDN-2 ([NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)) decouples
erase/write into channel-wise gates; no shipped model uses it yet. Watch for Qwen3.7-class
adoption; no action.

### 4. Theory

#### 4.1 Quantized-matmul rate-distortion (adopt as the paper's formal frame)

Ordentlich & Polyanskiy, High-Rate Quantized Matrix Multiplication
[I](https://arxiv.org/abs/2601.17187) / [II](https://arxiv.org/abs/2605.13768) (building on
their [Optimal Quantization for Matrix Multiplication](https://arxiv.org/abs/2410.13780)):
optimal quantizers for the *product* differ from optimal weight quantizers; universal
lattice-based quantizers achieve the (matching-lower-bounded) optimal distortion
`K(i,j)·2·2^(−2R)`; properly scaled ternary is competitive with lattice constructions in the
well-behaved-distribution regime, with lattices winning under heavy tails / fine-grained
sub-2-bit rate control. HyperQuant ([arXiv 2606.23406](https://arxiv.org/abs/2606.23406))
operationalizes the same frame (lattice + dithering + marginal-entropy calibration).

**Verdicts:** (a) cite this as the information-theoretic justification for ADR 0028's
output-aware objective — SALT V2 optimizes exactly the quantity the theory says matters;
(b) the incoherence/rotation preprocessing the theory implies (tails → well-behaved) is
already composable with SALT's front end and is where the remaining lattice advantage lives —
an explicit refined-track ablation, not a format change; (c) the paper's ternary-vs-2-bit
matched-bytes argument should be restated in product-distortion terms, where zero-skip and
scale-fit enter the constant.

#### 4.2 Speculative-decode acceptance theory (adopt: drafter objective + tree width)

[arXiv 2606.30265](https://arxiv.org/html/2606.30265): exact KL-divergence certificates for
draft acceptance. Greedy acceptance guaranteed when `KL(p||q) ≤ ε` and target margin
`γ_p > √(2ε)`; **tree branching with m candidates reduces the required margin to
√(4ε/(m+1))** — a closed-form account of *why* tree verify (ADR 0014/BASTION) beats
single-path at fixed drafter quality, and a direct objective for drafter training
(per-position KL to target + margin awareness, weighted by measured target margins).

**Verdict:** adopt into ADR 0021's training recipe (the BLUT run) and ADR 0032's cost model
(tree width m selectable from measured margins instead of grid search). This is also the
precise technical differentiator vs Neutrino's single-path argmax drafter.

#### 4.3 Model-dependent quantization robustness (evaluation policy)

Independent GGUF evals ([Kaitchup](https://kaitchup.substack.com/p/lessons-from-gguf-evaluations-ternary)):
TQ1_0 of Qwen3.5-397B-A17B tracks the original closely (~18.4% benchmark-error increase at
800→94 GB) while Minimax M2.5 degrades badly even at Q4 — quantization robustness is
model-family-specific and unpredictable from bpw alone. Reinforces plan 0043's
"confirmation model requires new preregistration" rule; generic bpw-quality claims without
per-model evidence are not publishable.

### 5. Kernels (small deltas only)

- **Ada-MK** ([arXiv 2605.11581](https://arxiv.org/abs/2605.11581)) automates megakernel
  construction via DAG search. It attacks the *engineering cost* side of ADR 0019's deferral;
  the *benefit* side (measured ~0.2 µs/boundary → ~3% ceiling on this decode) is unchanged.
  ADR 0019 stands; revisit only per its own reopen conditions.
- FlashAttention-4 (Blackwell tiles) and POD-Attention (fused prefill+decode) are the current
  attention references if/when the flash-numerics RFC (round 24's "both legs" note) proceeds.
- llama.cpp Q2_0-g64 CUDA (PR #25707) will be the first mainstream ternary CUDA baseline to
  add to `report compare` + BENCHMARKS.md the day it merges.

## Preregistration discipline

ADR 0028's frozen gates, tracks, and accounting rules are not modified. This ADR's adoptions
enter as: (a) *candidate mechanisms inside existing 0043 stages*, each behind its own ablation
(CAT-Q techniques → PTQ track; HESTIA relaxation → refined-track polish; Kronecker curvature →
capture backend); (b) *one new preflight gate* (GDN sensitivity, §3.1) inserted before Stage 8
spend — additive, spend-protective, and consistent with plan 0043's local-first policy;
(c) *evaluation-protocol adoptions* (score@budget, Bonsai-27B in the baseline set) that
strengthen rather than relax the evidence bar; (d) *export/interop work* (Q2_0 g64, entropy
transport) outside the claim-bearing measurement path. Anything that would change a frozen
threshold requires its own preregistration, per plan 0043.

## Decision summary

| # | Action | Lands in | Kind |
|---|--------|----------|------|
| 1 | CAT-Q techniques (learnable modulation, softened-ternarization relay, sliding-layer reconstruction) as ablation-gated candidates; CAT-Q into Stage-8 baseline set | 0043 PTQ track | method |
| 2 | HESTIA-style differentiable relaxation + curvature-scheduled annealing for the refined-track polish (two independent 2026 results — HESTIA, CAT-Q — converge on softened-ternarization > hard-STE) | 0043 refined track | method |
| 3 | Kronecker-factored curvature capture backend (KronQ/ACL-2026) evaluated before the 27B run | 0043 Stage 2 | infra |
| 4 | **GDN-sensitivity preflight gate** (recurrence-divergence probes, routes PTQ-vs-refined per tensor class; adjudicates the VBQ/Bonsai-vs-Ternary-Mamba/NVFP4 conflict) | 0043, pre-Stage-8 | new gate |
| 5 | Reproduce Bonsai-27B-ternary as a mandatory Stage-8 baseline; adopt score@budget reporting | 0043 Stage 8 | evaluation |
| 6 | Q2_0 g64 import + CompactV1→Q2_0 g64 export where profile-compatible | tritium-format | interop |
| 7 | Entropy-coded outer transport promoted to planned deliverable (expanded-bytes accounting unchanged) | 0043 / format | distribution |
| 8 | Acceptance-theory KL+margin objective into the drafter recipe; tree width from measured margins; reframe novelty as tree-verified ternary spec-decode | ADR 0021/0032 | method |
| 9 | Tokens-per-parameter-byte budget planning for refined-track/drafter training | 0043 / BLUT | planning |
| 10 | O-P product-distortion frame + matched-bytes restatement into plan 0042 paper; MoTE architecture cite (recipe only — its iso-memory number is refuted) | 0042 | paper |
| 11 | **Re-target ADR 0024 from 2:4 to 6:8 via SlideSparse sliding-window decomposition** (peer-reviewed, 1.33× measured, ternary-tolerance evidence from Sparse-BitNet); plan-0039 trainer hook targets 6:8 masks; checkpoint gate unchanged | ADR 0024 / plan 0039 | method |
| 12 | Structure-aware (RoPE-aware, non-uniform) bit allocation as the design frame for any future KV rung; note the rejected uniform t2 rung is consistent with Block-GTQ's uniformity-is-the-failure finding | ADR 0020 | design note |
| 13 | Watch: llama.cpp PR #25707 merge (→ BENCHMARKS.md competitor line), GDN-2 adoption, zero-ratio collapse in refined track, DF-SSM binary-state idea for GDN state memory, SharQ dual-GEMM pattern + Blackwell sparse-NVFP4 (E2M1 represents {−1,0,+1} exactly) for an eventual SM100/SM120 kernel target | — | watch |

## Consequences

- **Positive:** the 0043 campaign inherits the strongest known mechanisms at each stage without
  reopening its gates; the highest-risk tensor class (GDN) gets a spend-protective gate before
  the 27B run; the paper gains an information-theoretic frame that independently validates its
  central objective; the spec-decode program gets a theory-grounded objective and a sharpened
  novelty claim; artifacts become exportable to the emerging ecosystem standard.
- **Negative / risk:** adopting competitor-adjacent techniques (CAT-Q/HESTIA) means the SALT V2
  contribution must be argued as the *combination* (output-aware additive planes + allocation +
  native kernels), not any single mechanism — the paper's novelty section must be explicit.
  Bonsai-27B as a baseline raises the Stage-8 bar substantially; a loss there is a recorded
  negative result under plan 0043's rules. Two vendor claim-sets (Bonsai retention table,
  Neutrino throughput/quality) remain vendor-reported pending the verification ledger; nothing
  in this ADR's decisions depends on them being true — only the *existence* of the artifacts,
  which is verified.
- **Sequencing:** items 3 and 4 sit on the 27B critical path (before Stage 8); items 1-2 ride
  the existing 1.7B development rung; items 6-7 are parallel format work; items 8-9 wait on the
  BLUT run; item 10 is paper-time.

## Definition of done

- [ ] Items 1-2: ablation results recorded on the 1.7B rung (win, tie, or loss — all valid).
- [ ] Item 3: curvature-backend decision recorded with measured capture cost at ≥8B scale.
- [ ] Item 4: GDN preflight gate frozen (threshold + probe protocol) and executed; routing
      decision recorded before any Stage-8 spend.
- [ ] Item 5: Bonsai-27B reproduction numbers in the 0043 evidence pack, score@budget format.
- [ ] Item 6: Q2_0 g64 round-trip (import→export) gated bit-exact for P=1 profiles.
- [ ] Item 8: drafter objective documented in ADR 0021 revision; tree-width-from-margin in the
      ADR 0032 cost model.
- [ ] Item 11: ADR 0024 amended to 6:8/SlideSparse; plan-0039 trainer hook spec updated.
- [ ] Verification ledger from the deep-research run attached; vendor-reported claims labeled.
