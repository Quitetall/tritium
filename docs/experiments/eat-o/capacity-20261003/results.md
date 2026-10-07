# Capacity levers for integer ternary training: scale, planes, width

2026-10-03..07. Branch `research/eat-residual-carry`, `examples/ternary_capacity.rs`. Exploratory;
same MNIST data order, splits and protocol as the EAT-O campaign; official test set untouched.

## Question

The carried-latent + averaging optimizer matches its fp32 STE counterpart and every reasonable
optimizer stalls near 92% (`../rich-20261003/results.md`) — a model ceiling. Which capacity lever
raises it: learned per-row scales, more trit planes per weight, or more width?

## Implementation

`784→H→10` integer ReLU net generalized with T trit planes per weight (integer weights in
±(3ᵀ−1)/2, real value W/3^(T−1); every dot product still add/subtract), per-row Q12 scales applied
once per row (fixed, a fixed global multiplier, or learned), and width H. Training: carried
latents (2¹⁶ per 1.0 of real weight) + EMA averaging 2⁻¹⁰, STE gradients as exact integer
rationals (2⁻²⁰). Unit tests: T=1/H=128/fixed scales is exactly `Model::forward`; its gradient is
exactly 2⁹ × the carry gradient; extra planes start at the same function. The base arm reproduces
the rich-averaging result exactly (seed 1: 0.923 / 19,483.82).

A defect found and fixed during tuning: a floor shift on the scale update turns every small
negative gradient into −1, so scales drift up one unit per step regardless of the signal
(results at shifts 26–32 were identical). The scale update now rounds to nearest; all reported
scale arms are after the fix.

## Results (validation 1000–4999, five seeds, 20,000 examples)

| arm | H | T | model | accuracy | loss | Δloss vs base (95% t) |
|---|---:|---:|---:|---:|---:|---:|
| base (fixed scales) | 128 | 1 | 19.7 KiB | 92.120% ± 0.274 | 19,503 | — |
| global scale ×1.75 (fixed) | 128 | 1 | 19.7 KiB | 95.520% ± 0.185 | 12,523 | −6,980 [−7,153, −6,807] |
| learned per-row scales (on ×1.5) | 128 | 1 | 19.9 KiB | 95.295% ± 0.114 | 14,569 | −4,934 [−5,128, −4,741] |
| planes | 128 | 2 | 39.3 KiB | 94.245% ± 0.220 | 14,492 | −5,012 |
| planes | 128 | 3 | 59.0 KiB | 94.795% ± 0.210 | 13,382 | −6,121 |
| width | 256 | 1 | 39.3 KiB | 94.920% ± 0.141 | 13,185 | −6,318 |
| width | 512 | 1 | 78.7 KiB | 96.595% ± 0.144 | 9,952 | −9,551 |
| ×1.5 + planes | 128 | 3 | 59.0 KiB | 96.365% ± 0.098 | 10,426 | −9,077 |
| ×1.5 + width | 384 | 1 | 59.0 KiB | 96.855% | 9,721 | — |
| ×1.5 + width + planes | 256 | 2 | 78.7 KiB | 96.940% | 9,172 | — |
| ×1.5 + width | 512 | 1 | 78.7 KiB | 97.030% ± 0.156 | 9,362 | −10,141 |
| ×1.5 + width + planes | 512 | 3 | 236.0 KiB | **97.245% ± 0.176** | **8,127** | −11,376 |

Matched bytes (paired, five seeds):
- 59.0 KiB: width 384/T=1 minus width 128/T=3: Δloss −705 [−949, −460], +0.49 pp — **width wins**.
- 78.7 KiB: width 256/T=2 minus width 512/T=1: Δloss −190 [−524, +144] — tie.

## Reading

1. **The cheapest lever is free: the fixed scales were wrong.** A fixed ×1.5–1.75 global scale (no
   extra bits, no learning) takes 92.12% → 95.52%. Learned per-row scales add nothing beyond a
   correct global scale (14,946 vs 14,556 tuning loss on top of ×1.5) — the earlier "learned
   scale" gain was this calibration.
2. **Width and planes both raise the ceiling, and they stack**, to 97.25% at T=3/H=512 — the
   ternary model the optimizer was capped by is now 5 pp better, still fully multiply-free in its
   dot products and trained integer-only.
3. **At equal bytes, prefer width at small budgets**: more ternary weights beat more planes per
   weight at 59 KiB (resolved), and tie at 78.7 KiB. This is the grow-then-ternarize result again
   (`matched-bytes-ternary-beats-fp`), now in integer-only training.

Bounds: one small classifier, five seeds, single-seed tuning (the two matched-byte arms reuse
the ×1.5 / shift-15 setting untuned), no fp32 reference for the enlarged models, no LM, no
hardware or wall-clock claim. Compute per step scales with weights × planes for both levers.
