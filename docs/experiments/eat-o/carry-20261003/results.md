# Residual-carry control: the master weight moved into the optimizer

2026-10-03. Branch `research/eat-residual-carry`. Exploratory; quality UNKNOWN for any
release purpose; official MNIST test set untouched.

## Question

EAT-O accumulates per-coordinate evidence and flips a trit when it clears a threshold,
then clears that coordinate's evidence; its two EMAs also forget. A latent weight under
STE is the same accumulator without either loss: writing it as `trit·S + a`, an update
`a ← a − η·g` that pushes `|a|` past `S/2` moves the trit one step and **keeps the
remainder**. Does an integer accumulator that carries its remainder, living in optimizer
state with no float master tensor, recover STE quality?

## Implementation

`crates/tritium-train/examples/ternary_carry.rs`, reusing `ternary_lab`'s model, data,
initial trits, fresh-example order, splits and evaluation. Credit is `Model::sensitivity`
before its truncating division: the exact rational STE gradient in 2⁻¹¹ units (unit test
asserts `trunc(g/2¹¹) == sensitivity` for every coordinate). Every training operation is
an integer operation (integer Newton square root, Q30 bias correction, integer
stochastic rounding). Full coverage: every weight, every example, like the STE baseline;
counted work equals the STE baseline's by construction.

- `carry-sgd`: i32 latent, 65,536 units/trit, `latent −= g >> shift` (4 B/weight).
- `carry-sgd16`: i16 latent, 4,096 units/trit, round-to-nearest or stochastic (2 B/weight).
- `carry-adam`: i32 latent + i32 first moment + u32 second moment (12 B/weight).

The fp32 references come from `scripts/ternary-lab-baseline.py`, extended here with
`--optimizer sgd` (Adam unchanged) and given finer learning-rate grids than the
original campaign.

## Protocol

As the EAT-O classifier campaign: tune on seed 101 with 10,000 steps, minimum squared loss on
validation 0–999; then five paired seeds, 20,000 fresh examples each, evaluated on validation
1000–4999. carry-sgd's grid was extended adaptively (it first selected its edge, shift 12, then
shift 4 after extension); the fp32 grids were then made comparably fine. Selected settings:
carry-sgd shift 4 (≡ lr 2⁻⁹), carry-sgd16 shift 9 (≡ 2⁻¹⁰), carry-adam 655 units (≡ 0.01),
fp32 STE+SGD lr 2⁻⁹, fp32 STE+Adam lr 0.02.

## Results (validation 1000–4999, five seeds)

| Optimizer | State B/weight | Accuracy, mean ± SD | Mean loss |
|---|---:|---:|---:|
| carry-sgd (integer, i32 latent) | 4 | **92.050% ± 0.136** | **19,813** |
| fp32 STE + SGD | 4 (fp32 master) | 92.030% ± 0.163 | 19,881 |
| carry-sgd16, stochastic rounding | **2** | 91.605% ± 0.374 | 20,322 |
| carry-sgd16, round-to-nearest | 2 | 91.585% ± 0.378 | 20,385 |
| fp32 STE + Adam (lr 0.02) | 12 | 91.405% ± 0.187 | 20,321 |
| carry-adam (integer Adam) | 12 | 89.775% ± 0.486 | 23,986 |
| EAT-O best recipe (2026-10-01) | 12 | 73.300% ± 4.896 | 41,007 |
| original full EAT-O (2026-09-28) | 12 | 48.61% ± 5.45 | 57,819 |

Paired against fp32 STE+SGD (same seeds and evaluation examples):
- carry-sgd: Δloss −67.7, 95% t interval [−200.0, 64.6], Δaccuracy +0.02 pp — **indistinguishable**.
- carry-sgd16 stochastic: Δloss +441.5 [159.5, 723.6]; nearest: +503.7 [200.2, 807.2].

## Reading

1. **Master-free was never the obstacle; discarding evidence was.** An integer accumulator
   that carries its remainder IS a fixed-point master weight, and reproduces fp32 STE+SGD to
   within noise. EAT-O's gap (73% vs 92%) comes from what it does differently: clearing evidence
   on transition, forgetting through short EMAs, and vetoing 95–98% of candidates on a shared
   noise estimate (controller audit 2026-10-02).
2. **The state can be small.** A 2-byte integer latent — range for the trit plus the residual
   inside its rounding cell — matches fp32 STE+Adam's 12-byte state and trails STE+SGD by
   0.4 pp. Stochastic rounding barely matters at 4,096 units/trit.
3. **Integer Adam underperforms fp32 Adam here** (89.8% vs 91.4%). It was tuned only on the
   original 4-point grid and uses β1 = 0.875, β2 = 1−2⁻¹⁰ and a 2⁻¹¹-unit ε; not investigated.
   Plain SGD beats Adam for this squared-loss classifier in both fp32 and integer forms.

Bounds: one small classifier, five seeds, adaptive grid extension for carry-sgd, validation
reused across tuning (0–999) and evaluation (1000–4999) splits as in the earlier campaigns,
no LM, no hardware or wall-clock claim. Reversals are high (carry-sgd: ~2.08 M transitions,
~1.97 M reversals per run) — the latent dithers at cell boundaries, as STE masters do.

## Files

`summary.json` and `selected.json` (first campaign pass), `tune-*.json` / `eval-*.json` per run,
`ste/` (fp32 references), `sgd16/` (16-bit latent). `protocol.json` records the first pass's
grids; the later grid extensions are recorded by their `tune-*` files and this document.
