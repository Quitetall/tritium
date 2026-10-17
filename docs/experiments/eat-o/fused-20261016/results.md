# Fused forward-only training: noise credit straight into the carry (ladder axes F and G)

2026-10-16. Branch `research/eat-residual-carry`. `examples/ternary_carry.rs --grad noise`
(estimator in `examples/ternary_carry/noise.rs`) and `--defer W`. Exploratory; same MNIST network,
data order, splits and protocol as the carry campaign. Official test set untouched. Quality
UNKNOWN for any release purpose.

## Question

Can the integer carry itself do the averaging that a noise-based (Dust-style) gradient estimate
normally gets from a large population of draws? If so, one noise draw per example, credited and
accumulated straight into the carry, should match K draws averaged per step at equal forward
compute. That is the fused rule: no backward pass, no master weight, no separate optimizer.

## Method

- **Gradient source:** the all-integer estimate from `noise.rs`.
  - Each layer is perturbed separately: antithetic noise on its raw pre-activation sums,
    credited by the exact integer loss difference ℓ(+a) − ℓ(−a).
  - Per-unit estimate Σ d·a / (2K·E[a²]), outer-producted with the layer input.
  - Expressed in the exact gradient's 2⁻¹¹ units, so the carry's lr shift keeps its meaning.
- **Defaults** (from the G1 diagnostic): ternary noise {−σ, 0, +σ} with σ = 2 quantization steps,
  and the integer credit magnitude, not just its sign.
- **Optimizer:** carry-sgd, unchanged.
- **Tests:**
  - the split gradient/update path equals the original step bit for bit;
  - `--defer 1` is the immediate step;
  - with many draws the noise estimate approaches the exact gradient's direction (cosine > 0.9
    on both layers, for ternary and Gaussian noise);
  - the exact-gradient default reproduces carry-sgd (seed 1: 0.9200 / 19,821.73).
- **Protocol:** tune each arm's lr shift on seed 101 at half its evaluation horizon (grid {2..9},
  extended to {0..12} where the edge was selected; the selections stood), then evaluate on five
  paired seeds.
- **Compute unit:** "noisy evaluations" counts partial forwards: 4 per draw per example (2 layers ×
  2 antithetic signs). Each recomputes only the layers after the perturbation.

## Results (validation 1000–4999, five paired seeds)

| Arm | Examples | Noisy evals | Accuracy | Mean loss | Δloss vs exact [95%] |
|---|---:|---:|---:|---:|---:|
| exact STE gradient (carry-sgd) | 20k | 0 | **92.05 ± 0.14** | 19,813 | — |
| exact, flips deferred 16 | 20k | 0 | 91.70 ± 0.17 | 20,145 | +332 [142, 522] |
| **noise K=1 (fused)** | 20k | 80k | 84.22 ± 2.43 | 33,829 | +14,015 |
| noise K=4 | 20k | 320k | 88.03 ± 0.21 | 28,273 | +8,460 |
| noise K=16 | 20k | 1.28M | 89.96 ± 0.26 | 23,783 | +3,970 |
| **noise K=1, 4× examples** | 80k | 320k | 87.44 ± 1.14 | 27,994 | +8,181 |
| **noise K=1, 16× examples** | 320k | 1.28M | **90.11 ± 0.29** | 23,246 | +3,433 |
| K=1, flips deferred 16 | 20k | 80k | 82.74 ± 1.41 | 35,307 | +15,494 |
| K=1, flips deferred 64 | 20k | 80k | 81.48 ± 2.10 | 35,359 | +15,545 |
| K=1, reset on flip (keep 0%) | 20k | 80k | 82.42 ± 2.69 | 34,961 | +15,148 |
| K=1, keep 50% | 20k | 80k | 80.36 ± 2.00 | 35,271 | +15,458 |
| K=1, leak 2⁻¹⁶ | 20k | 80k | 80.67 ± 1.92 | 35,461 | +15,648 |
| K=1, leak 2⁻¹² | 20k | 80k | 74.84 ± 3.40 | 39,384 | +19,571 |
| K=1, σ = 4 steps | 20k | 80k | 82.55 ± 1.85 | 34,963 | +15,150 |
| K=1, σ = 1 step | 20k | 80k | 82.54 ± 3.38 | 35,045 | +15,232 |
| K=1, Gaussian noise | 20k | 80k | 80.72 ± 1.87 | 35,422 | +15,609 |

Paired contrasts (Δloss, first minus second, 95% t):

| Contrast | Δloss | Reading |
|---|---:|---|
| K=1 at 80k − K=4 at 20k (equal noisy evals) | −279 [−1,032, 474] | **tie** |
| K=1 at 320k − K=16 at 20k (equal noisy evals) | **−537 [−988, −87]** | **fused wins** |
| K=4 − K=1 (equal examples) | −5,555 | more draws per example help at fixed data |
| deferred 16 − immediate (K=1) | +1,478 [−872, 3,829] | unresolved, no benefit |
| keep 50% − keep all (K=1) | +1,442 [194, 2,691] | hurts |
| leak 2⁻¹⁶ − none (K=1) | +1,632 [153, 3,112] | hurts |
| leak 2⁻¹² − none (K=1) | +5,555 [2,486, 8,625] | hurts |
| reset − keep all (K=1) | +1,133 [−716, 2,981] | unresolved |
| σ = 4 / σ = 1 / Gaussian − default | +1,135 / +1,217 / +1,593 | all unresolved, none better |

## Reading

1. **The carry does the population's job.** At equal forward compute, one ternary draw per
   example straight into the carry *ties* averaging 4 draws (−279 [−1,032, 474]) and *beats*
   averaging 16 (−537 [−988, −87]). This is the fused rule's central claim, and it holds here:
   averaging across steps in the integer carry is at least as good as averaging across draws
   within a step.
2. **The simplest form is the best one.** Every noise-control mechanism the design anticipated
   costs or does nothing:
   - deferring flips (a frozen window) does nothing;
   - partial reset and leak hurt;
   - full reset is unresolved;
   - σ = 1 or 4 steps and Gaussian noise are not better than ternary σ = 2.

   The winning rule is: ternary antithetic noise, integer credit, carry keeps everything, flip
   immediately. This parallels the exact-gradient rungs (rules-20261016), where forgetting also
   only cost. The anticipated noise random walk is not the binding problem at this scale; the
   estimator's variance is.
3. **The gap to backprop is data and compute, not a ceiling.** Forward-only reaches 90.1% with
   16× the examples and 1.28 M partial forward evaluations, against exact backprop's 92.05% at 20k.
   It is still improving with data (84.2 → 87.4 → 90.1% at 1× / 4× / 16×).
4. **Deferring exact-gradient flips costs a little** (+332 [142, 522]): minibatch-of-16 trits are
   slightly worse than per-example updates at the same examples.
5. **K=1 runs are noisy across seeds** (SD 1.1–3.4 pp against 0.2–0.3 for K ≥ 4 and for exact),
   and several single-seed tuning curves were non-monotone. The paired contrasts above control for
   seed. Arm rankings among the K=1 variants are within noise unless marked.

Bounds: one 2-layer MLP (128 hidden units), squared loss, five seeds, validation reused across
tuning and evaluation. Partial forward evaluations here are cheap (only later layers recompute),
so "noisy evaluations" is not a wall-clock or FLOP equivalence to backprop. G1 showed the cost
grows with layer width (the dimension tax), so this result does not transfer to transformers by
itself. No LM, no hardware claim.

## Files

`run.py`, `runs/protocol.json` (arms, grid, selections, binary hash), `runs/tune-*.json` and
`runs/eval-*.json` (every run, including the grid extensions for K1-keep0, K1-leak12, K1-sigma4).
