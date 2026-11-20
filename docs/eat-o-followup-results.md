# EAT-O coverage and hysteresis experiments

October 1, 2026. **The best exploratory recipe reached 73.30% mean validation
accuracy across five seeds, compared with 48.61% for the original full EAT-O.**
It used 13.90 times less counted contraction work, but still about 4.94 times
the STE baseline's work and fell below its 91.19% accuracy. No matched-work
advancement gate or SOTA claim follows from this round.

## Implementation change

The opt-in `--incremental yes` scorer computes exact candidate loss from current
activations. A first-layer change recomputes one hidden unit's raw sum and the
output sums; a second-layer change recomputes one output sum. Recomputing raw
sums preserves truncation remainders and ReLU behavior. It does not approximate
candidate loss, change evidence rules, or introduce continuous master weights.
Stale replay caches still require full forward evaluation; accepted transitions
still refresh current-example activations.

The original seed-1 full-EAT-O control reproduced the **entire non-work optimizer
state**, including weights, evidence, RNG and replay, exactly. Counted contraction
work fell from 278,559,793,486 to 34,843,224,302, a 7.995× reduction. This is one
MNIST control plus tests, not a universal speedup or wall-time benchmark. The
work proxy counts recomputed dot-product terms and existing sensitivity work;
it excludes some scalar corrections, loss arithmetic, allocation and other work.

The default remains full recomputation. Older private checkpoints deserialize
with incremental scoring disabled; it cannot be overridden during resume.

## Experiment and results

The initial protocol froze six recipes: statistics-only or 16-example replay,
each with 128, 1,024 or 4,096 coordinates per example, threshold zero, and no
hysteresis. Each consumed 10,000 training examples on seed 101. Selection used
minimum squared loss on validation examples 0–999. The selected recipe was
statistics-only with 4,096 coordinates. All five evaluation seeds consumed
20,000 training examples and used validation examples 1000–4999, matching the
previous campaign's split, network, scales, targets and sample order.

The selected recipe averaged 832,050 transitions and 596,602 reversals per run.
After observing this, a **separate exploratory ablation** was frozen: re-enable
hysteresis and change nothing else. All five seeds were reported. This was an
adaptive follow-up using already-observed validation data, not an untouched
confirmation set. The official MNIST test set remains unused.

| Recipe | Accuracy, mean ± sample SD | Mean loss per example | Counted work / STE |
|---|---:|---:|---:|
| Original full EAT-O | 48.61% ± 5.45 pp | 57,819.47 | 68.67× |
| Original no-hysteresis variant | 66.59% ± 4.38 pp | 50,433.98 | 554.60× |
| New statistics-only, 4,096 coordinates, no hysteresis | 47.45% ± 9.21 pp | 47,364.53 | 21.84× |
| New statistics-only, 4,096 coordinates, hysteresis | **73.30% ± 4.90 pp** | **41,007.37** | **4.94×** |
| Previous STE+Adam baseline | 91.19% ± 0.26 pp | 20,369.86 | 1.00× |

Hysteresis-on accuracies were 68.80%, 74.975%, 67.575%, 78.90%, and 76.25%.
Versus the identical new recipe without hysteresis, mean loss decreased by
6,357.16 (paired 95% t interval −9,200.44 to −3,513.89), accuracy increased
25.85 percentage points, and reversals fell to 93,620 per run on average.
Versus original full EAT-O, loss decreased by 16,812.11 (interval −19,926.09 to
−13,698.12), and accuracy increased 24.69 points. These small-sample, exploratory
intervals do not correct for adaptive experiment selection.

The lesson is that hysteresis interacts with observation frequency: disabling
it helped the sparse replay-heavy recipe, but hurt this denser statistics-only
recipe. Simply increasing coverage without controlling reversals did not improve
accuracy. Removing replay also freed capacity for precision growth: the new
hysteresis recipe averaged 792.8 growth events versus zero in original full EAT-O.
That confirms growth was exercised, not that growth itself caused the gain.

## Using the experimental recipe

```sh
env RUSTC_WRAPPER= CARGO_TARGET_DIR=target/eat-o-build CARGO_BUILD_JOBS=2 \
  cargo build --locked -p tritium-train --example ternary_lab --release
target/eat-o-build/release/examples/ternary_lab \
  --data target/eat-o-data/mnist --incremental yes \
  --history statistics --coordinates 4096 --hysteresis yes --threshold 0 \
  --seed 1 --steps 20000 --eval-offset 1000 --eval-limit 4000 \
  --checkpoint /tmp/eat-o-new.checkpoint.json --report /tmp/eat-o-new.json
python scripts/verify-ternary-lab.py \
  /tmp/eat-o-new.checkpoint.json /tmp/eat-o-new.json --data target/eat-o-data/mnist
```

Output paths must be new. `scripts/run-eat-o-followup.py` reproduces the initial
six-recipe search, five-seed selected evaluation and scoring-only control.
The separate hysteresis protocol and executable runner are retained with the
campaign artifacts. Defaults and production trainers were not changed.

The next experiment should prospectively freeze the hysteresis recipe, fix the
work-budget stopping boundary, and run a matched-work comparison. Isolate
precision growth with fixed-width controls, and test intermediate update coverage
and hysteresis timing. Even 4,096 coordinates is only about 4% of the matrix per
example; it is not dense evidence streaming or equal update coverage with STE.

## Evidence and limits

Artifacts: `target/eat-o-campaigns/coverage-20261001/`, Git-ignored. This includes
the frozen binary and source, `protocol.json`, `hysteresis-protocol.json`, selected
settings, 17 reports/checkpoints/independent verification receipts, `summary.json`,
and `independent-analysis.json`. `verify_followup_summary.py` independently checks
source/binary hashes, exact control state, and paired statistics. Machine, data
hashes, source hashes and binary identity are recorded in the protocol. The data
files were restored and matched both hashes from the original campaign.

Runtime checkout: `679a570cb113ea21a804387c13e054645735c826`, with experimental
changes uncommitted; the frozen source and binary hashes identify the actual
implementation. CPU execution used two experiment workers; shared-machine
timings do not establish a speed benchmark. The older baseline was reused,
not retrained under the current checkout.

Validation: 18 Rust tests passed, including rounded exact-candidate comparisons
and matching backprop/probe training trajectories; scoped Clippy passed with
warnings denied; five Python tests passed; the local precommit entrypoint passed
with no staged changes. All 17 checkpoint evaluations were independently
recomputed. Full CI, production/release qualification, GPU execution, physical
ternary packing and language-model training remain unexecuted for this change.

See the [original campaign](eat-o-classifier-results.md) and
[implementation guide](integer-ternary-lab.md) for the underlying contracts.
