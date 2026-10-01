# EAT-O classifier campaign — 2026-09-28

**The current EAT-O implementation does not clear the advancement gate.** It
learns without continuous master weights, but the full variant reaches 48.61%
mean validation accuracy versus 91.19% for the controlled STE+Adam baseline,
while using 68.67 times as many counted contraction terms at equal fresh-example
count. This is evidence about this small implementation and recipe, not a verdict
on all evidence-accumulating optimizers. It provides no SOTA claim.

The experiment completed 36 tuning runs and 90 evaluation runs. All 80 integer
evaluation checkpoints passed the separately executed arithmetic/report verifier.
An independent integer evaluator reproduced accuracy for all 14 saved STE
checkpoints; aggregate FP32 loss roundoff was at most 6 summed-loss units.
An independent analysis reproduced the paired contrasts and advancement decision.
These checks establish bounded computational evidence, not release qualification.

## Protocol

- MNIST, bias-free 784→128→10 integer ReLU network, fixed scales and squared loss.
  First 55,000 training images supply fresh examples; the final 5,000 are validation.
  Tuning uses the first 1,000 validation images; evaluation uses the other 4,000.
  The official test set remains unused.
- Four settings per method, tuning seed 101, 10,000 fresh examples per setting.
  Thresholds are 0, 4, 16, 64; baseline learning rates are .0001, .001, .01, .1.
  Selection minimizes tuning loss. Selected threshold: 16 for simple, 0 for all
  other integer methods. Selected STE learning rate: .01.
- Five paired evaluation seeds, 1–5. Equal-data runs consume 20,000 fresh examples
  per seed. Work-limited runs target 4,090,880,000 counted contraction terms and
  stop after a complete step. A mismatch greater than 1% disqualifies comparison.
- The frozen primary gate compares full EAT-O with simple momentum: the upper
  endpoint of the paired 95% t interval for loss difference must be negative,
  with mean accuracy no more than one percentage point lower. Both regimes must
  pass, including work comparability. STE comparisons are reported separately.
- The 10,000-step tuning budget was chosen before evaluation because the old
  2,000-step default could not give every coordinate the required four visits.
  The runner now defaults to 10,000 and rejects non-smoke tuning below 3,176.

## Results

Accuracy is mean ± sample standard deviation across five seeds, in percentage
points. Work ratios compare equal-data training contraction counts with STE.
They do not measure elapsed time or complete computational cost.

| Method | Equal-data accuracy | Work-limited accuracy | Work comparison valid? | Data work / STE |
|---|---:|---:|:---:|---:|
| Simple momentum | 28.37% ± 5.93 | 26.48% ± 8.32 | Yes | 0.65× |
| Full EAT-O (`rich`) | 48.61% ± 5.45 | 12.70% ± 2.59 | **No** | 68.67× |
| Statistics only | 44.56% ± 2.34 | 39.83% ± 3.52 | Yes | 1.29× |
| Replay, cleared score history | 12.91% ± 2.50 | 12.09% ± 2.84 | Yes | 2.63× |
| No hysteresis | 66.59% ± 4.38 | 12.58% ± 2.57 | **No** | 554.60× |
| Fixed eight-trit accumulators | 46.67% ± 5.39 | 12.70% ± 2.59 | **No** | 68.60× |
| Integer32, larger state budget | 57.42% ± 1.87 | 12.27% ± 2.75 | **No** | 146.66× |
| Gradient-free probes | 40.69% ± 5.17 | 11.71% ± 2.46 | Yes | 254.25× |
| FP32 master weights, STE+Adam | 91.19% ± 0.26 | 91.19% ± 0.26 | Yes | 1.00× |

The invalid work comparisons are descriptive diagnostics only. Full EAT-O seeds
1 and 5 overshot by 2.906% and 1.459%; the other seeds stayed within tolerance.
Fixed8 had the same work-run results. No-hysteresis exceeded tolerance in all
five seeds, by up to 11.038%; integer32 exceeded it in three seeds. The gate was
not relaxed to accommodate these results.

### Gate disposition

- **PASS, equal-data primary comparison:** full EAT-O minus simple mean loss per
  example = −641.45, paired 95% t interval [−1155.34, −127.57]; accuracy +20.24 pp.
- **NOT QUALIFIED, work comparison:** full EAT-O violates work comparability.
  Its observed accuracy is 13.78 pp below simple, but this is not a valid matched
  comparison. The advancement flag is false.
- **FAIL, combined advancement:** `advance_to_language_model=false`. No LM stage
  or production activation was performed.
- **Full EAT-O trails STE at equal data:** loss difference +37449.61, interval
  [37013.09, 37886.13]; accuracy −42.575 pp.
- **UNKNOWN, SOTA and scaling:** no modern language-model baseline, official-test
  qualification, large-model training, GPU implementation, or external SOTA
  reproduction was executed.

Statistics-only is much cheaper than replay-heavy EAT-O and has the best observed
work-limited accuracy among its variants. It still fails the frozen loss criterion
versus simple: loss difference +759.04, interval [−565.97, 2084.05], despite accuracy
being 13.345 pp higher. Accuracy and squared loss measure different properties.
The ablation comparisons are exploratory; five seeds and one tuning seed do not
justify broad optimizer rankings or an extensive hyperparameter-search claim.

## What this reveals

1. **Replay verification is expensive.** Full EAT-O scores candidates with full
   forward passes over cached examples. It averages 25,065.8 proposals in a
   20,000-example run. Cached baseline activations do not make candidate scoring
   incremental. At the work limit, full EAT-O sees only 2,972.4 fresh examples
   on average; simple sees 30,189.4 and STE sees 20,000.
2. **Update coverage is a major confound.** EAT-O visits a rotating tile of 128
   coordinates per fresh example; STE updates the full weight matrix. The
   experiment compares these implemented trainers, not isolated optimizer rules.
3. **The default allocation starves adaptive width growth.** All five full runs
   had zero precision-growth events. Replay occupies 96 examples and total state
   is 1,219,040 bytes against a 1,219,584-byte cap. Statistics-only had 35 growth
   events across the five data runs. This does not isolate the benefit of growth:
   it also removes replay. The full-versus-fixed8 comparison does not establish
   that extra trits help when the full variant never allocated them.
4. **Hysteresis needs revisiting.** Removing it improves equal-data accuracy in
   this recipe, but raises work dramatically. This supports examining its
   thresholds and observation clock; it does not establish that hysteresis is
   generally harmful.
5. **The integer32 comparison is confounded by history capacity.** Its 24-byte
   per-parameter budget retains 224 replay examples versus 96 under the default
   12-byte budget. It is not a pure precision ablation. The cleared-history
   variant also retains noise/scaling and observation metadata; its result does
   not rule out every replay-only design.

The next classifier iteration should stream evidence across all weights, exploit
cached activations for exact incremental candidate scoring, cap proposal work
before expensive evaluations, and reserve an explicit precision-growth budget.
Then rerun comparisons with equal update coverage and isolated precision/replay
budgets. Fix the work-limit boundary before treating those runs as matched.
Keep the LM stage deferred until a new, prospectively frozen classifier campaign
clears its gate. These are proposed next steps, not changes made during this run.

## Reproduction and evidence

Campaign directory: `target/eat-o-campaigns/full-20260928/` (local, Git-ignored).
It contains frozen executable/source snapshots, protocol, data/source hashes,
selected settings, per-run reports and checkpoints, 80 verification receipts,
`baseline-verification.json`, `summary.json`, `analysis.json`, and `metrics.csv`.
`provenance.json` records the initial command and machine. `orchestration-change.json`
records continuation with four workers after completed serial tuning runs were
validated and reused. Algorithm, settings, split, and gate stayed fixed. Timings
are subject to CPU contention and are not a speed benchmark.

- Machine: x86_64 Intel Core i5-13600K, CPU execution; no GPU claim.
- Checkout HEAD: `12e073281ca3ba7cc92aee77a99598fb64976b27`, with the implementation
  uncommitted. HEAD alone does not identify the experimental source.
- Compiled implementation BLAKE3:
  `ab99e32546ac6f072fe84609dbd9848ea70940246b378ae3d148777877af0c56`.
- Executable SHA256:
  `b8bac7dabdfa5cc2d5e78f9aa6b41c223b6986d0969f23f6a5a98242ead86fd8`.
- Baseline script SHA256:
  `d4744f17a4015e68ef1a64695a944d7960afc2ae6581910b299dc9e8491986a8`.
- Evaluation identity BLAKE3:
  `8d59857a3602e1ce07543cb3027562da9b3a5d90a0a1cf643f485a19ffa529d4`.

Verifier commands:

```sh
python target/eat-o-campaigns/full-20260928/verify_baselines.py
# analyze_results.py writes new analysis.json and metrics.csv; use a fresh copy
# of the campaign directory if rerunning, because those outputs already exist.
python target/eat-o-campaigns/full-20260928/analyze_results.py
python -m unittest discover -s scripts/tests -p test_ternary_lab.py
sh scripts/verify-gates.sh precommit
```

Five Python tests passed, including rejection of insufficient tuning before any
artifact directory is created. The precommit entrypoint passed with no staged
changes; it is not a staged implementation or full CI/release qualification.
Earlier scoped Rust tests and Clippy are recorded in the
[implementation guide](integer-ternary-lab.md). Integer source scanning and
reference evaluation do not prove machine-code purity or physical ternary storage.
Contraction counts include replay and prediction audits but exclude some other
work, including Adam's elementwise operations. They are not total FLOPs or wall
time. Optimizer-state allocation excludes dataset/runtime/temporary buffers.
