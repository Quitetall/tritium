---
type: Concept
title: SALT — Sensitivity-Allocated Layered Ternary
description: Add ternary planes only where the model is sensitive; prune the rest. Stays multiply-free.
resource: https://github.com/Quitetall/tritium/blob/main/docs/adr/0001-salt-quantization.md
tags: [quantization, salt, residual, additive, mixed-precision]
timestamp: 2026-06-14T00:00:00Z
---

# SALT — Sensitivity-Allocated Layered Ternary

Tritium's quantization scheme. Full rationale in
[ADR 0001](https://github.com/Quitetall/tritium/blob/main/docs/adr/0001-salt-quantization.md);
this is the agent-facing summary.

A weight group is approximated by a **sum of ternary planes**:

```
W_g ≈ Σ_{p=1..T_g} s_{g,p} · t_{g,p},   t ∈ {-1, 0, 1}
```

Inference runs T add/sub/skip passes accumulated with per-plane scales — still
[multiply-free](/concepts/ternary.md). The plane count `T_g` is allocated per
group, not uniform.

Pipeline:

1. **Residual expansion** — greedily fit planes; each eats the previous residual.
2. **Mode codebook** — non-uniform per-plane scales from k-means/GMM over residual
   magnitudes. Sets *where levels sit*.
3. **Sensitivity rank** — per-group Hessian/Fisher (reuse GPTQ's Hessian).
4. **Plane allocation** — rate-distortion water-filling under a bits-per-weight
   budget. Sets *how many planes*. This is the corrected "multiplier".
5. **Prune** — residual planes are sparse; store nonzeros only, magnitude-prune.
6. **Heal** — short STE fine-tune in `tritium-train`.

Key correction vs the original sketch: **modes shape the codebook, sensitivity
sets the budget** — two separate levers.

Lineage: ABC-Net (residual bases), AQLM (additive), HAWQ/SqueezeLLM (sensitivity),
SpQR (sparse outliers) — specialized to ternary planes for kernel compatibility.
Owned by `tritium-quantize`; storage extends [TQ2_0](/concepts/tq-formats.md).
