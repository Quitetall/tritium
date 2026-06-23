# Quantization (SALT)

`tritium-quantize` implements **SALT** — *Sensitivity-Allocated Layered Ternary*
quantization. SALT spends extra capacity **only where the model is sensitive**,
along a single accuracy↔size knob, while keeping inference multiply-free. It is
designed in [ADR 0001](../../adr/0001-salt-quantization.md) and scheduled by the
[v0.40 quantization ADR](../../adr/0006-v040-quantization.md).

## Why not flat ternary

Flat ternary (every weight crushed to `{-1, 0, +1}` with one AbsMean scale per
tensor/channel, à la BitNet b1.58) is maximally cheap but loses accuracy on
weight groups that carry high information. SALT keeps the multiply-free kernel
but adds capacity selectively. Flat ternary is exactly SALT's `T = 1` special
case, not a rival.

## The pipeline

SALT operates per weight group `g` (granularity = a kernel tile: per output
channel or per 128-element block, so compute stays regular):

> **Implemented today vs planned.** Steps **1, 3, and 4** are implemented in
> `tritium-quantize` and drive the `tritium quantize` CLI. Steps **2** (mode
> codebook), **5** (sparse-plane *application* in the quantizer — the sparse
> storage form exists in `tritium-format`, but the quantizer currently writes
> dense planes), and **6** (STE heal — the offline quantize path has no
> `tritium-train` dependency, so no automatic heal runs there) are scheduled but
> not yet wired into the offline pipeline. See
> [ADR 0006](../../adr/0006-v040-quantization.md).

1. **Residual ternary expansion.** Approximate the group as a sum of ternary
   planes, each fitting the previous residual:
   `W_g ≈ Σ_{p=1..T_g} s_{g,p} · t_{g,p}` with `t ∈ {-1, 0, +1}`. `T` planes ≈
   `1.585·T` effective bits, and inference cost is `T` multiply-free passes —
   the existing add/sub/skip kernel, looped.
2. **Mode codebook.** Replace the single per-plane `absmean` with a small
   non-uniform scale set from k-means/GMM over the group's residual magnitudes.
   Sharpens each plane's fit; does not change the budget.
3. **Sensitivity rank.** Per-group sensitivity `H_g` from the Hessian
   diagonal / diagonal Fisher — reusing the Hessian GPTQ already computes, so the
   signal is free.
4. **Plane allocation — rate-distortion water-filling.** Minimize
   `Σ_g H_g · err_g(T_g)` subject to a bit budget, greedily assigning the next
   plane to the group with the largest marginal loss-drop-per-bit. Low-sensitivity
   groups settle at `T = 1`; irrelevant tiles at `T = 0` (skipped).
5. **Prune to minimal.** Residual planes (`p ≥ 2`) are mostly zeros, so store them
   **sparse** (nonzeros only) and magnitude-prune.
6. **Heal.** A short STE fine-tune (via `tritium-train`) recovers residual loss —
   see [Training](./training.md).

## The single knob

The whole pipeline is driven by one knob — **average bits-per-weight** — which
trades accuracy against size along a smooth ~1.58–3 bpw curve. The CLI exposes
it directly:

```sh
tritium quantize --input model.safetensors --output model.tslb --bpw 2.0
```

`--bpw 1.585` is all-base ternary (the `T = 1` flat case); higher budgets buy
extra residual planes on the most sensitive tiles (up to `~4.75` bpw at `T = 3`).
You can preview the bpw/error tradeoff on a raw fp32 matrix without committing to
a full quantize via `tritium report salt`.

## Hardware constraints (load-bearing)

- **Regular compute.** Plane counts are quantized to `{1, 2, 3}` and allocated at
  tile granularity, so every tile runs a fixed number of add-only passes. One
  dense base plane everywhere; extra planes only on selected tiles.
- **Sparse-vs-dense residual.** GPU sparse-matmul overhead pays only below ~10%
  nonzero density; above that, the plane stays dense with **whole-tile skip**.
  The threshold is measured per architecture, not assumed.
- **Storage.** `tritium-format` extends `TQ2_0` with a residual sidecar — base
  plane + optional sparse planes + per-plane scales. The on-disk bundle is the
  `.tslb` SALT bundle (or a GGUF container holding the SALT rows).

SALT is an **engineering** synthesis of established techniques — residual ternary
expansion (ABC-Net, AQLM), non-uniform mode scales (Deep Compression,
SqueezeLLM), sensitivity allocation (HAWQ, SqueezeLLM), and sparse residual
planes (SpQR) — chosen so every plane is still ternary and runs on the existing
add/sub/skip kernel. There is no new hardware path. See
[ADR 0001](../../adr/0001-salt-quantization.md) for the full derivation and the
prior-art references.
