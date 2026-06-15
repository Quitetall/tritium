# ADR 0001 — SALT: Sensitivity-Allocated Layered Ternary quantization

- **Status:** Proposed
- **Date:** 2026-06-14
- **Deciders:** Brian Lam
- **Crates affected:** `tritium-quantize` (owner), `tritium-core`, `tritium-format`, `tritium-train` (heal step), CUDA/CPU backends (multi-plane accumulate)

## Context

Flat ternary quantization (every weight crushed to `{-1, 0, +1}` with one AbsMean
scale per tensor/channel, à la BitNet b1.58) is maximally cheap but loses accuracy
on weight groups that carry high information. We want a scheme that:

1. Keeps inference **multiply-free** (Tritium's whole premise — add/sub/skip kernels).
2. Spends extra capacity **only where the model is sensitive**, not uniformly.
3. Degrades gracefully along a single knob (average bits-per-weight, ~1.58 → ~3).
4. Reuses prior art rather than inventing new theory.

The original sketch ("find modes → multiply parameter count → prune to just
enough") conflated two separate levers. Mode-finding sets *where levels sit*
(the codebook); it does **not** set *how much capacity to add* — that is a
sensitivity + rate-distortion question. SALT separates them.

## Decision

Adopt **SALT** — a ternary-plane-native synthesis of four established techniques:

| Ingredient | Prior art |
|---|---|
| Sum of ternary planes (residual expansion) | ABC-Net (binary bases); AQLM (additive) |
| Non-uniform per-plane scales from weight modes | Deep Compression; SqueezeLLM (k-means) |
| Plane-count allocation by loss sensitivity | HAWQ (Hessian); SqueezeLLM (Fisher) |
| Sparse residual planes (dense base + sparse extra) | SpQR; SqueezeLLM dense-and-sparse |

The novelty is **engineering, not mathematical**: every plane is ternary, so a
T-plane weight runs as T add/sub/skip passes accumulated with per-plane scales —
the existing kernel, looped. No new hardware path.

### The pipeline

Per weight group `g` (granularity = a kernel tile: per output channel or per
128-element block, so compute stays regular):

**1. Residual ternary expansion**

```
W_g ≈ Σ_{p=1..T_g} s_{g,p} · t_{g,p},   t_{g,p} ∈ {-1,0,1}
```

Greedy fit, each plane consumes the prior residual:

```
R₀ = W_g
for p in 1..=T_g:
    s_p = absmean(R_{p-1})            # or mode-codebook scale (stage 2)
    t_p = ternary(R_{p-1} / s_p)      # threshold yields the 0 state
    R_p = R_{p-1} − s_p · t_p
```

T planes ≈ `1.585·T` effective bits; inference cost = T multiply-free passes.

**2. Mode codebook.** Replace the single per-plane `absmean` with a small
non-uniform scale set from k-means/GMM over the group's residual magnitudes.
Sharpens each plane's fit. Does **not** affect the budget.

**3. Sensitivity rank.** Per group `H_g` = Hessian diagonal / diagonal Fisher.
Reuse the Hessian GPTQ already computes — free signal.

**4. Plane allocation — rate-distortion water-filling** (the corrected "multiplier"):

```
minimize  Σ_g H_g · err_g(T_g)   subject to   Σ_g |g| · 1.585 · T_g ≤ Budget
```

Greedy: repeatedly assign the next plane to the group with the largest marginal
loss-drop-per-bit `H_g · [err_g(T_g) − err_g(T_g+1)]`. Low-sensitivity groups
settle at `T=1`; irrelevant groups at `T=0` (tile pruned/skipped).

**5. Prune to minimal.** Plane count already minimal from step 4. Residual planes
(`p ≥ 2`) are mostly zeros → store **sparse ternary** (nonzeros only),
magnitude-prune, then heal.

**6. Heal.** Short STE fine-tune (`tritium-train`) recovers residual loss.

## Hardware constraints (load-bearing)

- **Regular compute:** quantize `T_g` to `{1, 2, 3}` and allocate at tile
  granularity, so every tile runs a fixed number of add-only passes. One dense
  base plane everywhere; extra planes only on selected tiles.
- **Sparse-vs-dense residual:** GPU sparse-matmul overhead pays only below
  ~10% nonzero density. Above the threshold, keep the plane dense with
  **whole-tile skip**. Threshold is measured per arch, not assumed.
- **Storage:** `tritium-format` extends TQ2_0 with a residual sidecar —
  base plane + optional sparse planes + per-plane scales.

## Consequences

**Positive**
- Single knob (Budget) trades accuracy ↔ size along a smooth ~1.58–3 bpw curve.
- Kernel stays multiply-free; reuses the add/sub/skip path T times.
- Reuses GPTQ's Hessian; no extra calibration machinery.
- Each stage maps to a verified prior-art baseline → testable in isolation.

**Negative / risks**
- Variable plane count complicates kernel scheduling — mitigated by tile-uniform `{1,2,3}`.
- Sparse planes need a density gate to avoid being slower than dense.
- Allocation + residual fit + optional heal is a heavier pipeline than flat AbsMean.
- `err_g(T)` marginal-gain table must be cheap to compute or the allocator is slow.

## Alternatives considered

- **Flat ternary (BitNet AbsMean):** simplest, but no capacity targeting. SALT
  reduces to this at Budget = 1.58 bpw, so it is the `T=1` special case, not a rival.
- **AQLM as-is:** stronger codebooks but its sum-of-codebook lookups are not
  multiply-free and don't map onto add/sub/skip kernels. SALT trades some
  accuracy for kernel compatibility.
- **Uniform mixed-precision (some layers 2-bit, some 4-bit):** coarser than
  per-tile plane allocation and breaks the all-ternary kernel story.

## Open questions

- Best granularity for `g` (channel vs 128-block) — measure accuracy vs kernel regularity.
- Does mode-codebook (stage 2) earn its cost over plain AbsMean per plane?
- Sparse-plane density threshold per GPU arch (Ampere/Hopper/Blackwell).
- Heal (STE) required, or is post-training allocation enough at target bpw?

## References

- AQLM — Extreme Compression via Additive Quantization — https://arxiv.org/pdf/2401.06118
- ABC-Net — Towards Accurate Binary CNNs — https://arxiv.org/pdf/1711.11294
- SqueezeLLM — Dense-and-Sparse Quantization — https://arxiv.org/pdf/2306.07629
- HAWQ — Hessian AWare Quantization — https://arxiv.org/abs/1905.03696
- SpQR — Sparse-Quantized Representation — https://arxiv.org/pdf/2306.03078
