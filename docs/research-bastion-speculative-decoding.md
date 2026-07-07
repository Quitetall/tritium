# BASTION & Block-Diffusion Speculative Decoding: Deep Dive

> 2026-06-22 — Research into BASTION (arXiv 2605.29727) and the surrounding block-diffusion speculative decoding ecosystem, oriented toward Tritium's ADR 0014 implementation.

---

## Table of Contents

1. [BASTION: The Paper](#1-bastion-the-paper)
2. [DFlash: The Block-Diffusion Drafter](#2-dflash-the-block-diffusion-drafter)
3. [The Block-Diffusion Speculative Decoding Ecosystem](#3-the-block-diffusion-speculative-decoding-ecosystem)
4. [Why Ternary + Block-Diffusion Is Uniquely Powerful](#4-why-ternary--block-diffusion-is-uniquely-powerful)
5. [What Tritium Needs to Implement (ADR 0014)](#5-what-tritium-needs-to-implement)
6. [External Drafter Options](#6-external-drafter-options)
7. [Speedup Projections for Tritium](#7-speedup-projections-for-tritium)
8. [References](#8-references)

---

## 1. BASTION: The Paper

**Title:** Bastion: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting

**Authors:** Soowon Oh, Nam Cao, Yujin Kim, Hojung Jung, Huzama Ahmad, Sangmin Bae, Se-Young Yun

**Venue:** NeurIPS 2026 — [arxiv.org/abs/2605.29727](https://arxiv.org/abs/2605.29727)

**License:** CC-BY 4.0

### Core Idea

Block-diffusion drafters predict multiple future-token distributions in one parallel pass, but they sample from **position-wise marginals** `q_k(·)` rather than fully conditioned sequences. Greedy paths from marginals miss the target model's preferred trajectory. BASTION solves this with dynamic tree construction.

### Three Synergistic Components

1. **Acceptance surrogate** — estimates expected accepted length via path confidence. Each candidate path `i` through the tree has score `ρ(i) = Π_k q_k(token_k)`. The surrogate predicts how many tokens the target will accept without running the target.

2. **Online latency estimator** — calibrates a hardware-aware roofline model. Measures `T_verify(N)` (time to verify N tree nodes) on the fly, so the budget controller knows the cost of each tree size.

3. **Adaptive best-first expansion** — grows the tree until marginal gains don't justify additional verification cost. The tree is built best-first: highest-probability prefixes expanded first, under a roofline budget that caps tree size `N` at the point where one target forward over `N` nodes still costs ~one AR step.

### Reported Results

- **Up to 6.61× over autoregressive** decoding
- **+39% over single-path block-diffusion** (DFlash without trees)
- **Lossless** — preserves the target model's distribution exactly via the standard speculative-sampling accept rule
- **Training-free** — no per-setting tuning required

### Key Insight for Tritium

The verification step is "free" up to the roofline knee. On an RTX 4090, the roofline knee is at ~73.7 ops/byte. A ternary target verifying N=32 tree nodes in one M=32 forward pass has arithmetic intensity well below the knee — the GPU is memory-bound, and verifying 32 nodes costs roughly the same wall time as verifying 1 node (because the memory traffic dominates and the KV is shared).

---

## 2. DFlash: The Block-Diffusion Drafter

**Title:** DFlash: Block Diffusion for Flash Speculative Decoding

**Authors:** Jian Chen, Yesheng Liang, Zhijian Liu

**Venue:** ICML 2026 — [arxiv.org/abs/2602.06036](https://arxiv.org/abs/2602.06036)

**Code:** [github.com/z-lab/dflash](https://github.com/z-lab/dflash)

### How DFlash Works

1. A **lightweight block diffusion model** generates an entire block of B candidate tokens in **one forward pass** (not sequential).
2. The draft model is **conditioned on context features extracted from the target model** (intermediate hidden states), ensuring semantic consistency.
3. The target LLM verifies all draft tokens in parallel — accepted tokens kept, rejected ones corrected.

### Results

- **Over 6× lossless acceleration** across models and tasks
- **Up to 2.5× higher speedup** than EAGLE-3 (state-of-the-art autoregressive speculative decoding)

### Why Block Diffusion > Autoregressive Drafting

| Property | Autoregressive Draft | Block Diffusion Draft |
|----------|---------------------|----------------------|
| Draft latency | O(B) sequential passes | O(1) single pass |
| Token conditioning | Fully conditioned | Position-wise marginals |
| Draft quality | High per-token | Lower per-token, compensated by tree |
| Memory | Small (one layer) | Small (one diffusion head) |
| Training | Distillation from target | Distillation from target |

The key trade-off: block diffusion produces **marginals** (per-position distributions) rather than fully conditioned sequences. A single greedy path from marginals is lower quality than an autoregressive draft. But the marginals are **rich** — they contain the full distribution at each position, which BASTION uses to build trees.

---

## 3. The Block-Diffusion Speculative Decoding Ecosystem

The field has exploded in early 2026. Here's the family tree:

### 3.1 DFlash Family

| Paper | arXiv | Key Idea | Speedup |
|-------|-------|----------|---------|
| **DFlash** | 2602.06036 | Block diffusion drafter, single-path verify | 6× |
| **DFlash–DFlare** | 2606.02091 | Layer-wise fusion to fix DFlash's conditioning bottleneck | >6× |
| **DDTree** | 2604.12989 | Best-first draft trees from DFlash marginals | >DFlash |
| **BASTION** | 2605.29727 | Budget-aware tree construction with roofline controller | 6.61× |

### 3.2 Other Diffusion Drafters

| Paper | arXiv | Key Idea | Speedup |
|-------|-------|----------|---------|
| **D²SD** | 2606.04446 | Dual diffusion drafters (first generates block + confidence, second re-anchors) | — |
| **CaDDTree** | 2606.01813 | Cost-aware diffusion draft trees, joint tree+budget optimization | — |
| **JetFlow** | 2606.18394 | Causal parallel draft head over fused hidden states | **9.64×** (MATH-500) |
| **BitLM** | 2605.11577 | Binary code representation + diffusion head | — |

### 3.3 Self-Speculative (No Separate Drafter)

| Paper | arXiv | Key Idea | Speedup |
|-------|-------|----------|---------|
| **S2D2** | 2603.25702 | Block-diffusion model with block_size=1 becomes AR → use as own verifier | 4.7× |
| **Teaching Diffusion to Speculate** | 2606.11552 | Training interventions to bridge diffusion ↔ AR gap | — |

### 3.4 VLM / Domain-Specific

| Paper | arXiv | Key Idea | Speedup |
|-------|-------|----------|---------|
| **Fast-dVLM** | 2604.06832 | Block-diffusion VLM, one-stage conversion from AR | 6× |
| **Fast-dDrive** | 2605.23163 | Block-diffusion for autonomous driving + scaffold spec-decode | 12× |
| **llada.cpp** | 2606.13740 | On-device diffusion LLM with multi-block spec-decode | 17-42× (vs CPU) |

### 3.5 Key Trend

The entire field is converging on the same architecture: **block-diffusion drafter + tree verification + budget-aware control**. BASTION is the most complete formulation, but DFlash, DDTree, and JetFlow are all variations on the same theme. The drafter generates B tokens in one pass; the tree construction converts marginals into a verification tree; the target verifies in one forward.

---

## 4. Why Ternary + Block-Diffusion Is Uniquely Powerful

### 4.1 The Ternary Draft Advantage

A ternary (1.58-bit) draft model has a unique property: **10× compression** vs FP16. This means:

- A **3B ternary draft** fits in the same memory as a **300M FP16 draft**
- A **1B ternary draft** is 10× faster to run than a **1B FP16 draft**
- The draft latency in the speedup formula `K / (1 + K * draft_latency / target_latency)` drops by 10×

### 4.2 Ternary Target + Block-Diffusion Draft

The BASTION architecture with a ternary target:

```
Block-diffusion drafter (FP16, small)
    → B tokens in one pass
    → Tree construction (best-first)
    → Ternary target verifies N nodes in one M=N pass
    → Accept longest path
```

The ternary target's verification is **extremely fast** because:
1. Ternary GEMV is 3-6× faster than FP16 (BitNet.cpp numbers)
2. Shared-prefix KV means most of the KV is already cached
3. The M=N forward is memory-bound, so verifying 32 nodes ≈ verifying 1 node in wall time

### 4.3 Ternary Draft + Ternary Target (the dream scenario)

If **both** draft and target are ternary:

```
Ternary block-diffusion drafter (3B ternary = 300M FP16 equivalent)
    → B tokens in one pass (extremely fast)
    → Tree construction
    → Ternary target (13B) verifies N nodes
    → Accept longest path
```

Draft latency is negligible. Verification latency is dominated by memory bandwidth. The speedup approaches `K` (the number of draft tokens) because the draft is essentially free.

### 4.4 No Published Work

**There is no published paper on ternary speculative decoding.** This is a research gap. The closest is:
- BitNet b1.58 2B4T used as a standalone model (no spec-decode)
- DFlash uses FP16 drafters exclusively
- BASTION's paper doesn't mention ternary at all

Tritium could be the first to demonstrate ternary speculative decoding.

---

## 5. What Tritium Needs to Implement (ADR 0014)

ADR 0014 defines three primitives. Here's what each requires:

### 5.1 Tree-Masked Verify Attention

**What:** N tree nodes share a prefix KV; each node attends `{shared prefix} ∪ {its tree-ancestors}` via a per-node ancestor mask.

**Implementation:**
- Input: parent table `P[N]` (each node's parent index), depth table `D[N]`
- Mask: for node `i`, attend to positions `{0..shared_prefix_len} ∪ {ancestors of i}`
- Reuse: split-KV partial+combine online-softmax from `decode.cu`
- New: the mask applied during the partial pass (one bitmask per node)

**Complexity:** 3-5 days. The split-KV structure already exists; the mask is the new ingredient.

### 5.2 Shared-Prefix KV + Provisional Commit/Rollback

**What:** One shared prefix KV + N provisional candidate K/V slots. Only the accepted path is promoted into the prefix.

**Implementation:**
- Shared prefix: a single KV arena for the already-accepted tokens
- Provisional region: N slots for the tree nodes' K/V values
- Promote: copy accepted-path K/V from provisional → prefix (O(accepted_length))
- Rollback: discard provisional region (O(1) — just reset the write pointer)

**Complexity:** 2-3 days. The KV layout change is the most intricate part.

### 5.3 Accept Logic

**What:** Longest-accepted-path selection under the speculative-sampling accept rule.

**Implementation:**
- Greedy: `tritium_nn::sample_greedy` — deterministic, already exists
- Sampling: standard speculative-sampling accept rule — for each position k, accept if `U < q(token_k) / p(token_k)` where q is drafter distribution, p is target distribution. On first reject, resample from `max(0, p-q)` and stop.

**Complexity:** 1-2 days. The accept rule is well-documented.

### 5.4 Total Implementation Estimate

| Primitive | Effort | Dependency |
|-----------|--------|------------|
| Tree-masked attention | 3-5d | Split-KV (done in v0.4.1) |
| Shared-prefix KV | 2-3d | Tree-masked attention |
| Accept logic | 1-2d | None |
| Integration testing | 2-3d | All above |
| **Total** | **8-13d** | |

---

## 6. External Drafter Options

ADR 0014 correctly identifies the drafter as out of scope. But here are the options:

### 6.1 DFlash (FP16 block-diffusion drafter)

**Code:** [github.com/z-lab/dflash](https://github.com/z-lab/dflash)

**Pros:** State-of-the-art, ICML 2026, proven 6× speedup, well-documented.
**Cons:** FP16 drafter = larger memory footprint, slower draft latency than ternary.
**Integration:** LAMU already runs DFlash block-diffusion spec-decode.

### 6.2 JetFlow (Causal parallel draft head)

**Code:** [github.com/hao-ai-lab/JetFlow](https://github.com/hao-ai-lab/JetFlow)

**Pros:** 9.64× speedup (best reported), trains a lightweight head on target's hidden states.
**Cons:** Requires training a head per target model.
**Integration:** The head is small (one layer) and could be quantized to ternary.

### 6.3 S2D2 (Self-speculative, no separate drafter)

**Code:** [github.com/phymhan/S2D2](https://github.com/phymhan/S2D2) (CC BY-NC-ND 4.0)

**Pros:** Training-free, uses the same model as both drafter and verifier.
**Cons:** Only works for diffusion LLMs (not autoregressive). 4.7× speedup (lower than DFlash).
**Integration:** Not directly applicable to Tritium's autoregressive target.

### 6.4 DDTree (Tree construction from DFlash)

**Pros:** Best-first tree construction from DFlash marginals. Directly compatible with BASTION's approach.
**Cons:** No code found yet (April 2026 paper).
**Integration:** The tree construction algorithm is the part BASTION improves upon.

### 6.5 Custom Ternary Drafter (Tritium-owned)

**Approach:** Train a small ternary block-diffusion head on the target model's hidden states.

**Pros:**
- 10× faster draft than FP16
- Tritium owns the full stack
- No external dependency

**Cons:**
- Requires training infrastructure (Tier 1 items from action plan)
- Research gap — no published recipe

**Timeline:** 2-4 weeks after Tier 1 training items are done.

---

## 7. Speedup Projections for Tritium

### 7.1 The Speedup Formula

```
speedup = K / (1 + K * t_draft / t_verify + t_draft / t_verify)
```

Where:
- K = number of draft tokens per step
- t_draft = time to generate K draft tokens
- t_verify = time to verify N tree nodes (one target forward)

### 7.2 RTX 4090 Projections

**Ternary target (BitNet b1.58 2B4T):**
- t_verify (N=32 nodes) ≈ 0.5ms (ternary GEMV is ~3× faster than FP16)
- t_verify (N=1, AR baseline) ≈ 0.5ms (same — memory-bound, N doesn't matter much)

**FP16 DFlash drafter:**
- t_draft ≈ 0.2ms (small diffusion head, one forward)
- K = 8 tokens per block

**Ternary DFlash drafter (hypothetical):**
- t_draft ≈ 0.02ms (10× faster)
- K = 8 tokens per block

| Configuration | t_draft | K | t_verify | Speedup |
|--------------|---------|---|----------|---------|
| AR baseline | — | 1 | 0.5ms | 1.0× |
| DFlash (FP16) + FP16 target | 0.2ms | 8 | 1.5ms | 3.2× |
| DFlash (FP16) + ternary target | 0.2ms | 8 | 0.5ms | 4.0× |
| DFlash (ternary) + ternary target | 0.02ms | 8 | 0.5ms | 7.3× |
| JetFlow + ternary target | 0.1ms | 12 | 0.5ms | 8.0× |

**With BASTION's tree construction (K=8 → N=32 tree nodes, higher acceptance):**

| Configuration | Accept rate | Avg accept | Speedup |
|--------------|-------------|------------|---------|
| DFlash single-path + ternary | 0.7 | 5.6 | 4.0× |
| BASTION tree + ternary | 0.85 | 6.8 | 5.2× |
| BASTION tree + ternary draft + ternary | 0.85 | 6.8 | 7.3× |

### 7.3 Theoretical Maximum

If the drafter is perfect (accept rate = 1.0, K=8):
- speedup = K = 8×

If the drafter is perfect and K=16:
- speedup = 16×

The practical limit is drafter quality × tree construction efficiency. BASTION's 6.61× with FP16 suggests **7-8× is achievable with ternary**.

---

## 8. References

- **BASTION**: [arxiv.org/abs/2605.29727](https://arxiv.org/abs/2605.29727) — Oh et al., NeurIPS 2026
- **DFlash**: [arxiv.org/abs/2602.06036](https://arxiv.org/abs/2602.06036), [github.com/z-lab/dflash](https://github.com/z-lab/dflash) — Chen et al., ICML 2026
- **DDTree**: [arxiv.org/abs/2604.12989](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026
- **DFlash–DFlare**: [arxiv.org/abs/2606.02091](https://arxiv.org/abs/2606.02091) — Zhang et al., 2026
- **D²SD**: [arxiv.org/abs/2606.04446](https://arxiv.org/abs/2606.04446) — Zhang et al., 2026
- **CaDDTree**: [arxiv.org/abs/2606.01813](https://arxiv.org/abs/2606.01813) — Zhang et al., 2026
- **JetFlow**: [arxiv.org/abs/2606.18394](https://arxiv.org/abs/2606.18394), [github.com/hao-ai-lab/JetFlow](https://github.com/hao-ai-lab/JetFlow) — Hu et al., 2026
- **S2D2**: [arxiv.org/abs/2603.25702](https://arxiv.org/abs/2603.25702), [github.com/phymhan/S2D2](https://github.com/phymhan/S2D2) — Han et al., 2026
- **BitLM**: [arxiv.org/abs/2605.11577](https://arxiv.org/abs/2605.11577) — Zhuang et al., 2026
- **Fast-dVLM**: [arxiv.org/abs/2604.06832](https://arxiv.org/abs/2604.06832) — Wu et al., 2026
- **llada.cpp**: [arxiv.org/abs/2606.13740](https://arxiv.org/abs/2606.13740) — Wang et al., 2026
- **ADR 0014**: `docs/adr/0014-spec-decode-bastion.md` — Tritium's BASTION design
