# EAT-O experiment journal

This is the durable index and chronological log of EAT-O experiments. Each new
campaign freezes its protocol before execution. Every run records a start event,
command, report, verification result, and completion or failure event. Compact
reports, receipts and summaries live under `docs/experiments/eat-o/` so they can
be versioned with this repository. Large checkpoints and frozen executables live
under `target/eat-o-campaigns/`; their hashes and locations are recorded, but
that directory is disposable and is not a durable checkpoint archive.

All entries are research evidence. Reused validation data, independent arithmetic
verification, matched-work comparisons, and release qualification remain distinct.
No failed or inconclusive run should be silently dropped. Corrections should be
new entries referencing the original run, not edits to frozen receipts.

## Earlier campaigns

- September 28, 2026: [original classifier campaign](eat-o-classifier-results.md),
  36 tuning and 90 evaluation runs. Full EAT-O 48.61%, STE 91.19%; advancement blocked.
- October 1, 2026: [coverage and hysteresis follow-up](eat-o-followup-results.md),
  17 runs. Statistics-only, 4,096 coordinates and hysteresis reached 73.30%;
  scoring-only control preserved the original state with 7.995× less counted work.
  These older campaigns predate the per-run durable journal below.

## Precision ablation campaign

Campaign: `precision-20261001`. Compare adaptive at 12 bytes/parameter with
adaptive, fixed8, fixed24 and integer32 at a shared 24-byte/parameter allocation
cap. Each runs seeds 1–5 for 20,000 fresh examples. Fixed recipe: statistics-only,
no replay, 4,096 coordinates, hysteresis on, threshold zero, exact incremental
scoring. No hyperparameter tuning or test-set evaluation. All 25 runs are reported.

The common cap removes the old replay-capacity confound; actual allocations can
still differ. Fixed modes permit exponent coarsening; adaptive also periodically
refines/shrinks. These compare implemented precision policies, not mantissa width
alone. Counted contraction work is a proxy, not wall time or total operations.
Primary contrasts compare each fixed policy against adaptive24 using paired
five-seed loss differences and 95% t intervals. They are exploratory and not
multiplicity-adjusted. There is no language-model advancement decision here.

Protocol and complete data: [campaign directory](experiments/eat-o/precision-20261001/).

## Run events

- 2026-10-01T18:47:02.573580+00:00 — `precision-20261001/adaptive12-1` **STARTED**.
- 2026-10-01T18:47:02.573781+00:00 — `precision-20261001/adaptive12-2` **STARTED**.
- 2026-10-01T18:47:44.893808+00:00 — `precision-20261001/adaptive12-1` **VERIFIED**; accuracy 68.800%; mean loss 41047.507; work 19,646,963,470; state 1,031,840 bytes; growth 815; clips 0.
- 2026-10-01T18:47:44.898979+00:00 — `precision-20261001/adaptive12-3` **STARTED**.
- 2026-10-01T18:47:45.522112+00:00 — `precision-20261001/adaptive12-2` **VERIFIED**; accuracy 74.975%; mean loss 38972.272; work 20,119,937,814; state 1,030,176 bytes; growth 780; clips 0.
- 2026-10-01T18:47:45.525908+00:00 — `precision-20261001/adaptive12-4` **STARTED**.
- 2026-10-01T18:48:50.296212+00:00 — `precision-20261001/adaptive12-3` **VERIFIED**; accuracy 67.575%; mean loss 43378.439; work 19,969,699,253; state 1,031,008 bytes; growth 783; clips 0.
- 2026-10-01T18:48:50.302436+00:00 — `precision-20261001/adaptive12-5` **STARTED**.
- 2026-10-01T18:48:51.878593+00:00 — `precision-20261001/adaptive12-4` **VERIFIED**; accuracy 78.900%; mean loss 38616.311; work 21,733,992,005; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-01T18:48:51.885061+00:00 — `precision-20261001/adaptive24-1` **STARTED**.
- 2026-10-01T18:50:03.328229+00:00 — `precision-20261001/adaptive12-5` **VERIFIED**; accuracy 76.250%; mean loss 43022.302; work 19,593,908,549; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-01T18:50:03.338713+00:00 — `precision-20261001/adaptive24-2` **STARTED**.
- 2026-10-01T18:50:06.911031+00:00 — `precision-20261001/adaptive24-1` **VERIFIED**; accuracy 68.800%; mean loss 41047.507; work 19,646,963,470; state 1,031,840 bytes; growth 815; clips 0.
- 2026-10-01T18:50:06.913429+00:00 — `precision-20261001/adaptive24-3` **STARTED**.
- 2026-10-01T18:50:48.119667+00:00 — `precision-20261001/adaptive24-2` **VERIFIED**; accuracy 74.975%; mean loss 38972.272; work 20,119,937,814; state 1,030,176 bytes; growth 780; clips 0.
- 2026-10-01T18:50:48.122712+00:00 — `precision-20261001/adaptive24-4` **STARTED**.
- 2026-10-01T18:50:50.830064+00:00 — `precision-20261001/adaptive24-3` **VERIFIED**; accuracy 67.575%; mean loss 43378.439; work 19,969,699,253; state 1,031,008 bytes; growth 783; clips 0.
- 2026-10-01T18:50:50.832903+00:00 — `precision-20261001/adaptive24-5` **STARTED**.
- 2026-10-01T18:51:31.172591+00:00 — `precision-20261001/adaptive24-4` **VERIFIED**; accuracy 78.900%; mean loss 38616.311; work 21,733,992,005; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-01T18:51:31.175943+00:00 — `precision-20261001/fixed8-1` **STARTED**.
- 2026-10-01T18:51:33.556415+00:00 — `precision-20261001/adaptive24-5` **VERIFIED**; accuracy 76.250%; mean loss 43022.302; work 19,593,908,549; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-01T18:51:33.559034+00:00 — `precision-20261001/fixed8-2` **STARTED**.
- 2026-10-01T18:52:13.742532+00:00 — `precision-20261001/fixed8-1` **VERIFIED**; accuracy 71.300%; mean loss 39393.666; work 19,574,319,508; state 1,029,344 bytes; growth 0; clips 0.
- 2026-10-01T18:52:13.745390+00:00 — `precision-20261001/fixed8-3` **STARTED**.
- 2026-10-01T18:52:16.334348+00:00 — `precision-20261001/fixed8-2` **VERIFIED**; accuracy 71.700%; mean loss 39749.550; work 19,438,297,418; state 1,029,344 bytes; growth 0; clips 0.
- 2026-10-01T18:52:16.337518+00:00 — `precision-20261001/fixed8-4` **STARTED**.
- 2026-10-01T18:53:05.986513+00:00 — `precision-20261001/fixed8-3` **VERIFIED**; accuracy 71.550%; mean loss 41646.896; work 20,545,063,815; state 1,029,344 bytes; growth 0; clips 0.
- 2026-10-01T18:53:05.994941+00:00 — `precision-20261001/fixed8-5` **STARTED**.
- 2026-10-01T18:53:10.940613+00:00 — `precision-20261001/fixed8-4` **VERIFIED**; accuracy 77.950%; mean loss 39221.031; work 22,659,133,057; state 1,029,344 bytes; growth 0; clips 0.
- 2026-10-01T18:53:10.965893+00:00 — `precision-20261001/fixed24-1` **STARTED**.
- 2026-10-01T18:54:22.045878+00:00 — `precision-20261001/fixed8-5` **VERIFIED**; accuracy 80.750%; mean loss 39106.325; work 20,452,367,958; state 1,029,344 bytes; growth 0; clips 0.
- 2026-10-01T18:54:22.048951+00:00 — `precision-20261001/fixed24-2` **STARTED**.
- 2026-10-01T18:54:50.062981+00:00 — `precision-20261001/fixed24-1` **VERIFIED**; accuracy 68.800%; mean loss 41047.507; work 19,646,963,470; state 2,350,560 bytes; growth 0; clips 0.
- 2026-10-01T18:54:50.068799+00:00 — `precision-20261001/fixed24-3` **STARTED**.
- 2026-10-01T18:55:29.293145+00:00 — `precision-20261001/fixed24-2` **VERIFIED**; accuracy 74.975%; mean loss 38972.272; work 20,119,937,814; state 2,350,560 bytes; growth 0; clips 0.
- 2026-10-01T18:55:29.296018+00:00 — `precision-20261001/fixed24-4` **STARTED**.
- 2026-10-01T18:55:57.044101+00:00 — `precision-20261001/fixed24-3` **VERIFIED**; accuracy 67.575%; mean loss 43378.439; work 19,969,699,253; state 2,350,560 bytes; growth 0; clips 0.
- 2026-10-01T18:55:57.048769+00:00 — `precision-20261001/fixed24-5` **STARTED**.
- 2026-10-01T18:56:34.431261+00:00 — `precision-20261001/fixed24-4` **VERIFIED**; accuracy 78.900%; mean loss 38616.311; work 21,733,992,005; state 2,350,560 bytes; growth 0; clips 0.
- 2026-10-01T18:56:34.436539+00:00 — `precision-20261001/integer32-1` **STARTED**.
- 2026-10-01T18:57:04.841145+00:00 — `precision-20261001/fixed24-5` **VERIFIED**; accuracy 76.250%; mean loss 43022.302; work 19,593,908,549; state 2,350,560 bytes; growth 0; clips 0.
- 2026-10-01T18:57:04.846937+00:00 — `precision-20261001/integer32-2` **STARTED**.
- 2026-10-01T18:57:27.414881+00:00 — `precision-20261001/integer32-1` **VERIFIED**; accuracy 68.800%; mean loss 41047.507; work 19,646,963,470; state 1,994,848 bytes; growth 0; clips 0.
- 2026-10-01T18:57:27.418859+00:00 — `precision-20261001/integer32-3` **STARTED**.
- 2026-10-01T18:57:56.288015+00:00 — `precision-20261001/integer32-2` **VERIFIED**; accuracy 74.975%; mean loss 38972.272; work 20,119,937,814; state 1,994,848 bytes; growth 0; clips 0.
- 2026-10-01T18:57:56.292906+00:00 — `precision-20261001/integer32-4` **STARTED**.
- 2026-10-01T18:58:18.549276+00:00 — `precision-20261001/integer32-3` **VERIFIED**; accuracy 67.575%; mean loss 43378.439; work 19,969,699,253; state 1,994,848 bytes; growth 0; clips 0.
- 2026-10-01T18:58:18.554752+00:00 — `precision-20261001/integer32-5` **STARTED**.
- 2026-10-01T18:58:46.331639+00:00 — `precision-20261001/integer32-4` **VERIFIED**; accuracy 78.900%; mean loss 38616.311; work 21,733,992,005; state 1,994,848 bytes; growth 0; clips 0.
- 2026-10-01T18:59:07.764884+00:00 — `precision-20261001/integer32-5` **VERIFIED**; accuracy 76.250%; mean loss 43022.302; work 19,593,908,549; state 1,994,848 bytes; growth 0; clips 0.
- 2026-10-01T18:59:07.770021+00:00 — `precision-20261001/campaign` **COMPLETE**.

## Precision campaign conclusion

All 25 runs completed, with independently recomputed checkpoint evaluations,
zero accumulator clipping, and an independently checked statistical summary.
The two adaptive budgets, fixed24, and integer32 produced identical final models
for every paired seed. Additional permanent accumulator width did not improve
this recipe at 20,000 examples.

| Policy | Five-seed accuracy | Mean final optimizer state |
|---|---:|---:|
| Adaptive, 12-byte cap | 73.30% ± 4.90 pp | 1,031,008 bytes |
| Adaptive, 24-byte cap | 73.30% ± 4.90 pp | 1,031,008 bytes |
| Fixed8, 24-byte cap | 74.65% ± 4.41 pp | 1,029,344 bytes |
| Fixed24, 24-byte cap | 73.30% ± 4.90 pp | 2,350,560 bytes |
| Integer32, 24-byte cap | 73.30% ± 4.90 pp | 1,994,848 bytes |

Adaptive used about 56.1% less final state than fixed24 while matching its final
models. The original 12-byte cap was not limiting these runs. Fixed8's mean loss
was 1,183.87 lower than adaptive24, but the paired 95% interval was
[−3,590.78, +1,223.03], so the apparent gain is inconclusive. Its mean counted
work was slightly higher, not lower. State figures exclude model weights,
datasets, transient buffers, allocator overhead and runtime memory.

Decision: retain adaptive12 as the reference recipe; do not increase its budget
or permanent accumulator width on this evidence. Fixed8 remains a candidate for
confirmation rather than a demonstrated winner. These results isolate implemented
precision policies, including their different exponent handling, not mantissa
width in every possible setting. They do not settle activation precision, other
training objectives, larger models, matched-work quality or SOTA.

- [Complete results with every seed](experiments/eat-o/precision-20261001/results.md)
- [Per-run machine-readable data](experiments/eat-o/precision-20261001/runs.csv)
- [Independent analysis](experiments/eat-o/precision-20261001/independent-analysis.json)
- Reproduction runner: `scripts/run-eat-o-precision.py`; analysis:
  `scripts/analyze-eat-o-precision.py`. Frozen copies accompany the records.
- Validation this round: five Python tests, Python compilation checks, and local
  precommit checks passed. No Rust training code changed in this precision round;
  the previous frozen, tested executable was used.

For subsequent experiments, create a new campaign ID, record the hypothesis and
controls before launching, and retain commands, failures and verified results.
A STARTED event without a terminal event is incomplete, never a pass. Use a new
output directory for reruns; do not overwrite existing reports or receipts.
Large local checkpoints need a separate backup if they must survive target-directory
cleanup; the compact data and checkpoint hashes are retained in this repository.

## Exact discrete credit campaign

Campaign `credit-20261001`: freeze the previous adaptive12, statistics-only,
4,096-coordinate, hysteresis-on, threshold-zero recipe. Compare approximate
backprop sensitivity with exact finite-change loss evidence, seeds 1–5,
20,000 fresh examples each, validation examples 1000–4999. Both routes use raw
sum caches for exact candidate scoring and accepted updates. No signal compression,
replay, tuning, official-test evaluation, or work-matched claim. Report every seed.
Primary contrast: probe minus backprop mean squared loss, paired 95% t interval;
accuracy and counted work reported separately. Validation remains reused.
A probe seed-1 control without raw caching must reproduce the same final model.
Each backprop seed must reproduce the prior hysteresis model. These are cache
correctness checks, separate from whether exact credit improves learning.
Raw sums are transient per-example working memory (138 i64 values for MNIST),
not checkpointed optimizer history. The work proxy counts scalar weight/activation
correction products as well as full contractions, but not all arithmetic.

- 2026-10-01T20:56:03.445324+00:00 — `credit-20261001/backprop-1` **STARTED**.
- 2026-10-01T20:56:03.445958+00:00 — `credit-20261001/backprop-2` **STARTED**.
- 2026-10-01T20:56:03.451118+00:00 — `credit-20261001/backprop-1` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'backprop', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '1', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/backprop-1.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/backprop-1.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.453200+00:00 — `credit-20261001/backprop-3` **STARTED**.
- 2026-10-01T20:56:03.453773+00:00 — `credit-20261001/backprop-2` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'backprop', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '2', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/backprop-2.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/backprop-2.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.457340+00:00 — `credit-20261001/backprop-3` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'backprop', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '3', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/backprop-3.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/backprop-3.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.457811+00:00 — `credit-20261001/backprop-4` **STARTED**.
- 2026-10-01T20:56:03.461721+00:00 — `credit-20261001/backprop-5` **STARTED**.
- 2026-10-01T20:56:03.465328+00:00 — `credit-20261001/backprop-4` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'backprop', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '4', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/backprop-4.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/backprop-4.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.467120+00:00 — `credit-20261001/backprop-5` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'backprop', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '5', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/backprop-5.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/backprop-5.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.467660+00:00 — `credit-20261001/probe-1` **STARTED**.
- 2026-10-01T20:56:03.471390+00:00 — `credit-20261001/probe-2` **STARTED**.
- 2026-10-01T20:56:03.475201+00:00 — `credit-20261001/probe-1` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '1', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe-1.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe-1.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.477032+00:00 — `credit-20261001/probe-3` **STARTED**.
- 2026-10-01T20:56:03.477307+00:00 — `credit-20261001/probe-2` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '2', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe-2.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe-2.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.479972+00:00 — `credit-20261001/probe-3` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '3', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe-3.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe-3.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.482007+00:00 — `credit-20261001/probe-4` **STARTED**.
- 2026-10-01T20:56:03.483468+00:00 — `credit-20261001/probe-5` **STARTED**.
- 2026-10-01T20:56:03.488104+00:00 — `credit-20261001/probe-4` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '4', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe-4.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe-4.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.490775+00:00 — `credit-20261001/probe_reference-1` **STARTED**.
- 2026-10-01T20:56:03.489661+00:00 — `credit-20261001/probe-5` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'yes', '--precision', 'adaptive', '--budget', '1219584', '--seed', '5', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe-5.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe-5.checkpoint.json']' returned non-zero exit status 1..
- 2026-10-01T20:56:03.494176+00:00 — `credit-20261001/probe_reference-1` **FAILED**; Command '['/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/ternary_lab', '--data', '/home/brianklam/Desktop/Tritium/target/eat-o-data/mnist', '--history', 'statistics', '--replay', '0', '--coordinates', '4096', '--hysteresis', 'yes', '--threshold', '0', '--incremental', 'yes', '--route', 'probe', '--raw-cache', 'no', '--precision', 'adaptive', '--budget', '1219584', '--seed', '1', '--steps', '20000', '--eval-offset', '1000', '--eval-limit', '4000', '--report', '/home/brianklam/Desktop/Tritium/docs/experiments/eat-o/credit-20261001/probe_reference-1.json', '--checkpoint', '/home/brianklam/Desktop/Tritium/target/eat-o-campaigns/credit-20261001/probe_reference-1.checkpoint.json']' returned non-zero exit status 1..

## Exact credit launch correction

`credit-20261001` failed before training: all 11 attempts used a stale binary
copied while the new release build was still running, and rejected `--raw-cache`.
The FAILED events and commands are retained. No quality result came from that
campaign. The corrected `credit-20261002` uses the completed executable and a
preflight that rejects binaries missing the option before creating run records.
The experimental recipe and comparison remain unchanged.

- 2026-10-02T19:05:06.665754+00:00 — `credit-20261002/backprop-1` **STARTED**.
- 2026-10-02T19:05:06.667431+00:00 — `credit-20261002/backprop-2` **STARTED**.
- 2026-10-02T19:06:01.995968+00:00 — `credit-20261002/backprop-2` **VERIFIED**; accuracy 74.975%; mean loss 38972.272; work 3,204,926,422; state 1,030,176 bytes; growth 780; clips 0.
- 2026-10-02T19:06:02.365451+00:00 — `credit-20261002/backprop-3` **STARTED**.
- 2026-10-02T19:06:02.946971+00:00 — `credit-20261002/backprop-1` **VERIFIED**; accuracy 68.800%; mean loss 41047.507; work 3,203,190,434; state 1,031,840 bytes; growth 815; clips 0.
- 2026-10-02T19:06:02.972589+00:00 — `credit-20261002/backprop-4` **STARTED**.
- 2026-10-02T19:06:45.264322+00:00 — `credit-20261002/backprop-4` **VERIFIED**; accuracy 78.900%; mean loss 38616.311; work 3,205,539,815; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-02T19:06:45.299998+00:00 — `credit-20261002/backprop-3` **VERIFIED**; accuracy 67.575%; mean loss 43378.439; work 3,205,656,365; state 1,031,008 bytes; growth 783; clips 0.
- 2026-10-02T19:06:45.311807+00:00 — `credit-20261002/backprop-5` **STARTED**.
- 2026-10-02T19:06:45.314929+00:00 — `credit-20261002/probe-1` **STARTED**.
- 2026-10-02T19:07:24.633053+00:00 — `credit-20261002/backprop-5` **VERIFIED**; accuracy 76.250%; mean loss 43022.302; work 3,204,368,301; state 1,031,008 bytes; growth 793; clips 0.
- 2026-10-02T19:07:24.641073+00:00 — `credit-20261002/probe-2` **STARTED**.
- 2026-10-02T19:07:28.339044+00:00 — `credit-20261002/probe-1` **VERIFIED**; accuracy 65.875%; mean loss 45707.821; work 3,218,231,688; state 1,031,008 bytes; growth 779; clips 0.
- 2026-10-02T19:07:28.341028+00:00 — `credit-20261002/probe-3` **STARTED**.
- 2026-10-02T19:08:07.322787+00:00 — `credit-20261002/probe-2` **VERIFIED**; accuracy 67.075%; mean loss 46920.287; work 3,217,787,402; state 1,031,008 bytes; growth 749; clips 0.
- 2026-10-02T19:08:07.353780+00:00 — `credit-20261002/probe-4` **STARTED**.
- 2026-10-02T19:08:10.830114+00:00 — `credit-20261002/probe-3` **VERIFIED**; accuracy 65.500%; mean loss 46312.090; work 3,217,906,658; state 1,031,008 bytes; growth 732; clips 0.
- 2026-10-02T19:08:10.833533+00:00 — `credit-20261002/probe-5` **STARTED**.
- 2026-10-02T19:09:07.559503+00:00 — `credit-20261002/probe-4` **VERIFIED**; accuracy 64.975%; mean loss 47099.586; work 3,219,449,162; state 1,031,008 bytes; growth 765; clips 0.
- 2026-10-02T19:09:07.575460+00:00 — `credit-20261002/probe_reference-1` **STARTED**.
- 2026-10-02T19:09:10.840867+00:00 — `credit-20261002/probe-5` **VERIFIED**; accuracy 68.975%; mean loss 46197.267; work 3,217,402,530; state 1,031,840 bytes; growth 759; clips 0.
- 2026-10-02T19:11:21.600237+00:00 — `credit-20261002/probe_reference-1` **VERIFIED**; accuracy 65.875%; mean loss 45707.821; work 233,401,937,952; state 1,031,008 bytes; growth 779; clips 0.
- 2026-10-02T19:11:21.677219+00:00 — `credit-20261002/campaign` **COMPLETE**.

## Exact credit conclusion and commit checkpoint

The corrected `credit-20261002` campaign completed all 11 runs. Every checkpoint
passed independent integer evaluation, with zero clipping; an independent analysis
recomputed the paired comparison. Approximate evidence achieved 73.30% ± 4.90 pp;
exact finite-change evidence achieved 66.48% ± 1.60 pp. Exact minus approximate
mean loss was +5,440.04, paired 95% t interval [+2,180.13, +8,699.96]. Exact credit
therefore did not improve this recipe. This is reused-validation research, not a
universal statement about discrete optimization or a matched-work gate.

The raw-sum cache reproduced all five prior backprop models. Mean counted work
fell from 20,212,900,218.2 to 3,204,736,267.4 terms, about 6.3× less. The exact-probe
seed-1 cache control also reproduced its uncached model, with 3,218,231,688 versus
233,401,937,952 counted terms. These are arithmetic-work proxies, not measured
wall-time speedups or hardware qualification. The cache uses transient integer
sums; optimizer-state accounting does not include those buffers.

Decision: retain approximate evidence with hysteresis as the reference. Keep exact
finite-change scoring available for audits and future proposal calibration.
Quantization plateaus, example-specific benefits, and interactions among weights
are hypotheses for the poorer exact-credit result; this experiment did not isolate
them. A hybrid proposal/verification controller remains untested.

- [Exact-credit results and every seed](experiments/eat-o/credit-20261002/results.md)
- [Independent analysis](experiments/eat-o/credit-20261002/independent-analysis.json)
- [Failed launch explanation](experiments/eat-o/credit-20261001/README.md)

Current-source checks before the local commit: 19 Rust tests, scoped Clippy with
warnings denied, and five Python tests passed. The Spark test profile could not
run with this account, so the parent executed the named checks directly. The
commit hook checks the exact staged tree; full workspace CI, hardware performance,
release qualification and production integration are not claimed.

The future credit runner now preflights both the CLI option and the executable's
compiled source digest before creating a campaign. The corrected campaign itself
used the earlier option check; a separate source-binding receipt verifies its
frozen executable reports against its frozen sources. Existing receipts are not
rewritten to imply a stronger preflight than actually ran.
