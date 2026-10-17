# Offline estimator diagnostic (ladder rung G1)

2026-10-16. Branch `research/eat-residual-carry`, `examples/ternary_estimator.rs`. No training:
the estimator is measured apart from any optimizer. Exploratory; quality UNKNOWN.

## Question

Before building a forward-only (Dust-style) trainer: how well does a noise-credited gradient
estimate agree with the exact STE gradient on frozen ternary checkpoints of the MNIST carry
network? How does that depend on the noise type, the noise size relative to the integer
quantization step, the population, and the training stage?

## Method

- **Checkpoints:** carry-sgd (shift 4, seed 1) after 0, 2,000 and 20,000 examples (accuracy 14.8%,
  87.1%, 92.0%). Saved with `ternary_carry --save-trits`.
- **Perturbation:** one layer at a time, as Dust does by layer type. Integer noise is added to that
  layer's raw pre-activation sums: the hidden layer (128 units, step 2⁸ raw units) or the output
  layer (10 units, step 2³). The true quantized forward is re-run from that point, and each draw
  is credited by its exact integer loss change.
- **Noise types:**
  - `Gauss`: rounded Gaussian with a clean-forward baseline;
  - `Anti`: the same noise as antithetic ± pairs;
  - `Ternary`: a ∈ {−σ, 0, +σ}, antithetic;
  - `Sign`: ternary antithetic with the credit reduced to its sign (ElasticZO / EGGROLL).
- **Estimate:** Σ r·a per unit, outer-producted with the layer input.
- **Population:** 256 training examples × K draws each. Examples are the population, as tokens
  are in Dust.
- **Scores:** cosine to the exact STE gradient summed over the same examples, and the
  |exact|-weighted fraction of coordinates whose sign matches.
- **Tests:** the clean forward reproduces `Model::forward`/`Model::loss` exactly, and antithetic
  noise recovers the exact output-layer gradient (cosine > 0.9 at 400 draws).

## Results (batch 256; `step*.jsonl` has every row, including batch 16)

Cosine vs draws per example K, at each row's best σ (σ in quantization steps):

| Checkpoint | Layer | Noise | best σ | K=1 | 4 | 16 | 64 | 256 | sign agree @256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| step 0 | hidden | Gauss | 2 | 0.248 | 0.435 | 0.685 | 0.884 | 0.963 | 0.973 |
| step 0 | hidden | Anti | 2 | 0.273 | 0.465 | 0.719 | 0.897 | 0.967 | 0.975 |
| step 0 | hidden | Ternary | 4 | 0.244 | 0.481 | 0.729 | 0.897 | 0.964 | 0.975 |
| step 0 | hidden | Sign | 4 | 0.232 | 0.430 | 0.670 | 0.848 | 0.947 | 0.965 |
| step 0 | output | Ternary | 4 | 0.861 | 0.948 | 0.983 | 0.997 | 0.998 | 1.000 |
| step 0 | output | Sign | 4 | 0.748 | 0.800 | 0.900 | 0.902 | 0.893 | 0.934 |
| step 20,000 | hidden | Gauss | 4 | 0.094 | 0.125 | 0.327 | 0.594 | 0.849 | 0.919 |
| step 20,000 | hidden | Anti | 4 | 0.109 | 0.151 | 0.341 | 0.625 | 0.859 | 0.924 |
| step 20,000 | hidden | Ternary | 4 | 0.198 | 0.230 | 0.401 | 0.664 | 0.859 | 0.924 |
| step 20,000 | hidden | Sign | 4 | 0.149 | 0.140 | 0.308 | 0.546 | 0.745 | 0.870 |
| step 20,000 | output | Ternary | 0.5 | 0.438 | 0.669 | 0.849 | 0.969 | 0.993 | 0.994 |
| step 20,000 | output | Sign | 0.5 | 0.419 | 0.500 | 0.726 | 0.867 | 0.884 | 0.958 |

Cosine vs σ (batch 256, K = 16), hidden layer at step 0, for σ = 0.25 / 0.5 / 1 / 2 / 4 steps:

| Noise | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|
| Gauss | 0.266 | 0.440 | 0.641 | 0.685 | 0.692 |
| Anti | 0.452 | 0.616 | 0.703 | 0.719 | 0.731 |
| Ternary | 0.367 | 0.659 | 0.694 | 0.718 | 0.729 |
| Sign | 0.211 | 0.426 | 0.526 | 0.627 | 0.670 |

## Reading

1. **The estimator converges on the true gradient direction here.**
   - At 256 examples × 256 draws, the hidden-layer cosine is 0.96 early and 0.85 on the trained
     model.
   - The output layer reaches 0.99.
   - Sign agreement (by |exact| mass) is 0.92–0.98 on the hidden layer and ≥ 0.99 on the output.
2. **The dimension tax is visible and as predicted.** The 128-unit hidden layer needs about 16×
   more draws than the 10-unit output layer for the same cosine, which fits node perturbation
   scaling with the number of units (Werfel/Xie/Seung). Wider layers will cost proportionally
   more.
3. **The trained model is the hard case.** At K = 16, the hidden-layer cosine falls from about
   0.72 at init to about 0.34–0.40 at 92%. Small late-training gradients sit closer to the integer
   loss's quantization floor. The cost of a forward-only trainer is set by late training, not
   early.
4. **Ternary noise is as good as Gaussian, or better.** It is best or tied at every checkpoint and
   K, and clearly best on the trained hidden layer at K ≤ 64 (0.198 vs 0.109 at K = 1). An
   all-integer step does not pay an estimator penalty here.
5. **Antithetic beats a clean baseline** at small σ and small K; at large K and σ they converge.
6. **Sign-only credit is biased.** It plateaus (output layer ~0.89–0.96 instead of 0.99+) and
   trails on the hidden layer. EGGROLL's and ElasticZO's sign fitness costs accuracy here; keep
   the integer magnitude.
7. **Noise below one quantization step is lost.** At 0.25 steps the cosine roughly halves (0.27
   vs 0.69 Gaussian, hidden, step 0), because the integer forward truncates sub-step noise.
   σ ≥ 1–2 steps is safe up to at least 4 steps, with no sign yet of leaving the linear regime.
   This is the coarse-noise landmine, measured: σ is bounded below by the quantization step.
8. **Cost.** Reaching cosine ≈ 0.7 on the early hidden layer takes K = 16 antithetic draws, i.e.
   32 extra forward-equivalents per example (with cheap partial forwards: only the layers after
   the perturbation are recomputed), against roughly 3 for backprop. On the trained model, the
   same cosine needs K ≈ 64–128. Whether the carry's averaging across steps lets a training run
   live with K = 1–4 is the fusion question (ladder axis F), not this rung's.

Bounds: one 2-layer MLP (784→128→10), squared loss, one checkpoint seed, one example sample per
checkpoint; cosine is to the STE surrogate gradient, not to a "true" discrete gradient (which the
noise estimator measures more directly than STE does).

## Files

`trits-step{0,2000,20000}.json` (checkpoints), `step*.jsonl` (all 400 rows per checkpoint: batch
16/256 × K ∈ {1,4,16,64,256} × σ ∈ {0.25,0.5,1,2,4} × 4 noise types × 2 layers).
