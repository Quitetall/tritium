# Rich optimizer state for integer ternary training

2026-10-03. Branch `research/eat-residual-carry`, `examples/ternary_carry.rs --optimizer rich`.
Exploratory; same MNIST network, data order, splits and protocol as the EAT-O campaign and the
residual-carry control (`../carry-20261003/results.md`).

## Question

Ternary weights cost ~0.2 B each, so an optimizer can hold far more per-weight state than a
mixed-precision AdamW (fp16 weight + fp32 master + two fp32 moments ≈ 16 B) without exceeding its
footprint. Which extra state, on top of the carried integer latent, buys quality?

Composable integer options (all on the i32 carried latent):

| option | state | rule |
|---|---|---|
| momentum | +4 B | EMA of the step, β = 1 − 2⁻ᵏ |
| curvature | +8 B | EMA of the EXACT per-example diagonal Gauss-Newton (output 2(h/8)², input 2p²Σw2²); step = Newton fraction 2⁻ˡʳ of g/(c+δ), δ = layer-mean curvature ≫ d |
| dampen | +1 B | halve a weight's step on every reversal; recover one level per 1024 steps |
| average | +4 B | EMA of the latent, 2⁻ᵏ; deployed trits = round(average) |

Unit tests: `rich` with no options is `carry-sgd` bit for bit; the output-layer Gauss-Newton
diagonal is exact.

## Tuning (seed 101, 10,000 steps, validation 0–999; carry-sgd reference 20,290)

| arm | best setting | tuning loss |
|---|---|---:|
| averaging + momentum | average 2⁻¹⁰, momentum 2⁻², lr shift 4 | 19,645 |
| averaging | average 2⁻¹⁰, lr shift 4 | 19,764 |
| momentum | 2⁻², lr shift 4 | 20,322 (neutral) |
| dampen | lr shift 4 | 22,205 (hurts) |
| diagonal Gauss-Newton | ema 2⁻⁴, δ = mean ≫ 9, Newton fraction 2⁻¹⁶ (interior) | 25,109 (hurts) |
| fp32 STE+SGD + EMA (control) | lr 2⁻⁹, decay 2⁻⁹ | 19,749 |

## Evaluation (validation 1000–4999, five paired seeds, 20,000 examples)

| optimizer | state B/weight | accuracy | mean loss | Δloss vs fp32 SGD+EMA (95% t) |
|---|---:|---:|---:|---:|
| rich: averaging + momentum | 12 | 92.205% ± 0.180 | 19,458 | −90 [−228, +48] |
| rich: averaging | 8 | 92.120% ± 0.274 | 19,503 | −45 [−186, +96] |
| fp32 STE+SGD + EMA | 4 + 4 | 92.205% ± 0.261 | 19,548 | — |
| carry-sgd | 4 | 92.050% ± 0.136 | 19,813 | +265 [+100, +430] |
| fp32 STE+SGD | 4 | 92.030% ± 0.163 | 19,881 | +333 [+166, +499] |

## Reading

1. **Averaging is the only extra state that pays, and it pays equally in fp32.** EMA of the
   latent cuts loss ~1.6% (resolved) for integer and fp32 alike; integer-with-averaging is
   indistinguishable from fp32-with-averaging. Momentum is neutral, oscillation damping hurts,
   and exact diagonal Gauss-Newton preconditioning hurts badly (it inflates steps on
   low-curvature weights such as input weights on near-black pixels).
2. **At every level tried, the integer ternary optimizer matches its fp32 counterpart.** More state
   does not beat fp32, because fp32 can hold the same state. What ternary changes is the
   footprint: carried latent + EMA is 8 B/weight of integer state plus ~0.2 B of trits, against
   ~16 B/weight for mixed-precision AdamW.
3. **All good optimizers converge to ~92%.** fp32 Adam 91.4, SGD 92.0, SGD+EMA 92.2. The ceiling
   here is the model (bias-free ternary 784→128→10 with fixed power-of-two scales and squared
   loss), not the optimizer; more optimizer state cannot move it. Raising it needs model capacity
   — learned scales, more planes per weight, width — not optimizer memory.

Bounds: one small classifier, five seeds, single-seed tuning with grid extensions, no LM, no
hardware or wall-clock claim.
