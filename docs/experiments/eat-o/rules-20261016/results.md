# Update-rule rungs on carry-sgd (ladder axis U)

2026-10-16. Branch `research/eat-residual-carry`, `examples/ternary_carry.rs` (`--keep-pct`,
`--leak`, `--round`, `--units-log2`, `--width`). Exploratory; same MNIST network, data order,
splits and protocol as the carry campaign (`../carry-20261003/results.md`). Official test set
untouched. Quality UNKNOWN for any release purpose.

## Question

The landscape survey (`docs/ternary-training-landscape.md` §3) left four open questions about the
carry. Each is tested here as a single change from `carry-sgd` with the exact STE gradient held
fixed:
- **Keep vs reset (U3):** does keeping the in-cell remainder on a flip matter, or is it
  "just error feedback"?
- **Leak (U4):** what does forgetting cost with exact gradients?
- **Rounding (U7):** does deterministic rounding hold up without the stochastic rounding the
  field relies on?
- **Width (U8):** does a narrow carry lose credit?

## Identity discipline

Every knob's default reproduces `carry-sgd` bit for bit. Unit tests assert:
- the generic rule path at identity settings equals `carry-sgd`;
- i16 storage at 2¹² units/trit with nearest rounding equals the existing `carry-sgd16`;
- keep 0% lands exactly on the new cell centre;
- the leak shrinks the remainder by `r >> k` and never flips a trit by itself;
- i8 storage saturates and counts it.

The base arm reproduces the recorded carry-sgd result exactly (seed 1: 0.9200 / 19,821.73).

**Defect found while writing the tests:** the shared test model `tiny()` had every ReLU dead on
every input, so every gradient was zero. The earlier `rich_without_options_is_carry_sgd` identity
test therefore passed vacuously. `tiny()` now uses fixed trits with live units, and a new test
asserts the stream produces nonzero gradients and transitions. All 20 tests pass non-vacuously,
including the old ones. The rich campaign's conclusions rested on measured runs, not on that test,
so they stand.

**Floor bias:** the original rule `latent −= g >> shift` floors, so any negative gradient moves
the latent by at least one unit, even at a huge shift. At 2¹⁶ units per trit this is invisible.
At 16 units per trit it is a 1/16-trit bias per step — see the i8 floor arms.

## Results (validation 1000–4999, five paired seeds, 20,000 examples)

Each arm's lr shift was tuned on seed 101 over {2,…,6}; the i8 2⁶-unit arms selected the edge (6),
were extended to {7, 8, 9}, and 6 stayed best. Δloss is paired against base, with a 95% t
interval (df = 4).

| Arm (one change from carry-sgd) | State B/w | Accuracy | Mean loss | Δloss [95%] | Transitions | Underflows | Saturations |
|---|---:|---:|---:|---:|---:|---:|---:|
| **base** (keep 100%, floor, 2¹⁶ units, i32) | 4 | **92.050 ± 0.14** | 19,813 | — | 2.08 M | 23 k | 0 |
| keep 50% | 4 | 92.080 ± 0.16 | 19,734 | −79 [−210, 52] | 0.62 M | 23 k | 0 |
| keep 0% (reset, as BOLD/Bop) | 4 | 91.725 ± 0.26 | 19,932 | +118 [−47, 284] | 0.20 M | 70 k | 0 |
| leak 2⁻¹⁶ | 4 | 91.655 ± 0.20 | 20,019 | +206 [96, 316] | 1.73 M | 70 k | 0 |
| leak 2⁻¹² | 4 | 91.755 ± 0.14 | 20,272 | +458 [251, 666] | 2.34 M | 22 k | 0 |
| leak 2⁻⁸ | 4 | 90.050 ± 0.16 | 24,217 | +4,404 [3,619, 5,189] | 2.29 M | 19 k | 0 |
| leak 2⁻⁴ | 4 | 57.865 ± 4.14 | 47,111 | +27,298 | 2.19 M | 2 k | 0 |
| round nearest | 4 | 91.880 ± 0.15 | 19,938 | +125 [−64, 315] | 1.75 M | 55 k | 0 |
| round stochastic | 4 | 91.665 ± 0.14 | 19,971 | +157 [19, 296] | 1.75 M | 61 k | 0 |
| i16, 2¹² units, floor | 2 | 91.555 ± 0.35 | 20,312 | +499 [187, 810] | 1.83 M | 2.1 M | 3.7 M |
| i16, 2¹² units, nearest | 2 | 91.585 ± 0.38 | 20,385 | +571 [219, 924] | 1.81 M | 1.9 M | 3.7 M |
| i16, 2¹² units, stochastic | 2 | 91.605 ± 0.37 | 20,322 | +509 [189, 830] | 1.82 M | 1.9 M | 3.7 M |
| i8, 2⁴ units, floor | 1 | 71.655 ± 8.15 | 39,779 | +19,966 | 6.14 M | 56 M | 4.3 M |
| i8, 2⁴ units, nearest | 1 | 90.890 ± 0.71 | 21,686 | +1,873 [530, 3,217] | 2.27 M | 112 M | 3.6 M |
| **i8, 2⁴ units, stochastic** | **1** | **91.445 ± 0.38** | 20,704 | +891 [450, 1,331] | 1.63 M | 166 M | 1.7 M |
| i8, 2⁶ units, floor | 1 | 65.080 ± 13.03 | 43,978 | +24,165 | 5.49 M | 71 M | 4.9 M |
| i8, 2⁶ units, nearest | 1 | 90.240 ± 0.53 | 22,767 | +2,953 [2,411, 3,496] | 1.52 M | 122 M | 4.2 M |
| i8, 2⁶ units, stochastic | 1 | 90.195 ± 0.34 | 22,679 | +2,866 [2,415, 3,316] | 1.60 M | 115 M | 4.3 M |

For reference (carry campaign): fp32 STE + SGD 92.030%, fp32 STE + Adam 91.405%, EAT-O best
recipe 73.300%.

## Reading

1. **Keeping the remainder is not a demonstrated advantage here.** Resetting to the cell centre
   on every flip costs +118 [−47, 284] in loss and 0.3 pp in accuracy, which is unresolved. Keeping
   half is indistinguishable from keeping all (−79 [−210, 52]).
   - What reset *does* change is dithering: transitions fall 10× (2.08 M → 0.20 M) and keep 50%
     falls 3×, at no measured cost.
   - The paper claim "keep-remainder beats reset (BOLD/Bop)" is therefore **not supported** at
     this scale. The defensible claim is narrower: the carry matches STE whether or not it keeps
     the remainder, and a partial reset suppresses boundary dithering for free.
   - This contradicts the carry campaign's reading that "discarding evidence was EAT-O's gap".
     Resetting on a flip alone cannot explain the 92 → 73% gap. The remaining suspects are
     EAT-O's other features: forgetting EMAs (see the leak rows), partial coverage, credit
     truncation, and the noise-estimate veto.
2. **Forgetting costs, monotonically.** With exact gradients the leak is pure loss: +206 at 2⁻¹⁶,
   +458 at 2⁻¹², +4,404 at 2⁻⁸, collapse at 2⁻⁴. This is the strongest explanation yet for
   EAT-O's gap, since EAT-O's short EMAs are a fast leak. For the forward-only rule the leak's job
   is noise control, so its price must be paid back there or not at all.
3. **Rounding mode does not matter at 2¹⁶ or 2¹² units per trit.** Floor, nearest and stochastic
   are within noise of each other at both widths; stochastic is marginally *worse* at i32
   (+157 [19, 296]). Deterministic rounding is safe whenever the carry resolves the step. This
   answers the field's SR objection for i32 and i16.
4. **At i8, stochastic rounding is required, and it works.**
   - Floor collapses at both unit sizes (65–72%, SD 8–13 pp). Its one-unit bias per negative
     gradient is 1/16 trit per step at 2⁴ units.
   - Nearest loses 1.9–3.0 k in loss.
   - Stochastic rounding at 2⁴ units reaches **91.445%, a 1-byte-per-weight integer carry
     0.6 pp below fp32 STE+SGD's 4-byte master**, with 166 M underflows that SR keeps unbiased.
   - This matches 2610.16298 and ECO: sub-unit updates need SR. With i8, the carry can be pitched
     on memory as well as on integer-only arithmetic.
5. **Saturation is heavy and apparently harmless at i16** (3.7 M clips per run at ±8 trit steps).
   At i8 it is under 5% of updates. It is not the limiting factor at either width.

Bounds: one small squared-loss classifier, five seeds, validation reused across tuning (0–999)
and evaluation (1000–4999), no LM, no hardware or wall-clock claim.

## Files

`run.py` (campaign driver), `runs/protocol.json` (arms, grid, selections, binary hash),
`runs/tune-*.json` and `runs/eval-*.json` (every run, including the i8 2⁶-unit grid extension
`tune-i8u6-{nearest,stochastic}-s{7,8,9}.json`).
