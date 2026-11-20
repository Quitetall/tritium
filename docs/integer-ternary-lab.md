# EAT-O — Evidence-Accumulating Ternary Optimizer

Status: experimental implementation; the completed classifier campaign did not
clear advancement. See [results and limitations](eat-o-classifier-results.md).
Scaling and SOTA claims remain **UNKNOWN**.
The [experiment journal](eat-o-experiment-log.md) records campaign protocols and
individual run outcomes. New experiments must log starts, failures, and verified
results there, with compact reports and receipts under `docs/experiments/eat-o/`;
temporary console output alone is not the experiment record.
The [October follow-up](eat-o-followup-results.md) adds opt-in exact incremental
scoring and reports an exploratory 73.30% five-seed recipe; matched-work
qualification remains outstanding.
This is a private example, not a new production trainer, public API, release gate,
or checkpoint contract. Production adoption needs an ADR and separate qualification.

## Question

Can richer optimizer memory improve direct ternary learning without a continuous
master weight? The harness separates the credit signal, accumulated memory, and
decision to change a trit. Quality takes priority over memory savings, with an
explicit default budget to prevent invisible state growth.

The implementation lives in `crates/tritium-train/examples/ternary_lab.rs` and its
private `ternary_lab/` modules. The optimizer type is `EatOptimizer`; the laboratory
command remains `ternary_lab`. No existing trainer behavior changes.

Related primary work:

- [Bop](https://papers.neurips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization.pdf): binary transitions driven by gradient inertia.
- [BitBop](https://github.com/ValerioDolci/bitbop): direct ternary transitions with floating-point momentum/computation; research baseline, not a scaling guarantee.
- [NITI](https://phwl.org/assets/papers/niti_tpds22.pdf): integer-only training arithmetic.

This implementation does not reproduce those papers' complete training recipes.

## Numerical and learning contract

Model matrices are signed-byte trits, initialized directly in {-1,0,+1}. There is
no master weight, weight quantizer, or continuous proposal array. Byte storage
is intentional in this correctness reference; no 1.58-bit physical claim is made.
The MNIST model is bias-free 784→128→10, with integer ReLU, division by 256 after
the first matrix multiplication and by 8 after the second, truncating toward zero.
Targets are 256 for the correct class and zero otherwise; loss is summed squared
error. Every operation in the experimental training path uses integers. Model
shape/input bounds bound matrix accumulations; loss and evidence calculations
use i128. Checked reference tests cover the small state spaces.

Two credit routes share the optimizer:

- `backprop`: sensitivity of the unrounded linear extension at the current trits.
  Weight sensitivities are calculated for a rotating tile, without allocating a
  whole dense gradient. Integer activation rounding has a surrogate derivative;
  **this is not an exact derivative of the discrete computation**. It eliminates
  the weight-quantizer STE, not all approximation or all gradient computation.
- `probe`: measure loss before and after an adjacent candidate on the same
  training example, then restore the trit. No backpropagation. Additional forward
  evaluations are counted, including rejected proposals and prediction audits.

Each direction has fast (1/8) and slow (1/64) integer EMAs of estimated loss
improvement. A block-shared EMA of absolute signal disagreement measures noise.
After four observations, both timescales must exceed threshold plus noise.
Hysteresis adds a two-visit cooldown and doubles the threshold for a reversal
within 16 visits. Ages count visits to the coordinate, not global training steps.
Accepted transitions reset that coordinate's evidence. The `simple` ablation
stores only fast evidence; the campaign also disables its hysteresis.

Adjacent transitions are -1↔0↔+1. Acceptance is sequential. Activations are
refreshed after every accepted transition; later decisions use the changed model.
Accumulated scores describe historical observations, not promises about current
loss. That distinction is measured using prediction-error diagnostics.

### Representation and allocation

An accumulator represents `2^e * Σ d[k] 3^k`, with d in {-1,0,+1}. The canonical
balanced-ternary integer is offset-coded into 13/26/39 packed bits for 8/16/24
trits. This is integer emulation, not native ternary hardware. Per-128-weight
blocks share the exponent and width. Additional digits provide range; decreasing
the exponent provides resolution. Changes preserve represented values except
when explicit stochastic rounding coarsens them.

Adaptive banks grow when range or sub-quantum samples require more precision and
the allocation budget permits. Otherwise they coarsen with seeded stochastic
rounding. Every 64 steps they can shrink when all values fit exactly with a 4×
margin, or refine exactly when range permits. This heuristic is a hypothesis,
not an optimal precision allocator. Reports include growth, shrink, rescale,
lost-small-signal, and clipping counts. Clipping fails independent verification.

`fixed8` and `fixed24` keep mantissa width fixed but can rescale. `integer32` uses
fixed-width offset-coded integer accumulators as a numerical comparison. Its
larger state needs an explicit larger budget; the campaign labels it a reference
and grants 24 bytes/parameter. It is **not** a memory-matched result against the
12-byte default.

### History and budget

`statistics` retains scores only. `both` combines scores with replay verification.
`replay` clears score history after each observation, retaining hysteresis metadata
and using replay to verify proposals. All variants retain the observation count.

Replay stores up to 256 past training examples and their versioned hidden/output
activations. Its capacity can be smaller under the budget. Stale activations are
recomputed before scoring; temporary candidates never populate the cache.
Examples are admitted while space permits, then replaced by reservoir-style
sampling. Changing capacity makes this a bounded history sampler, not a claim of
uniform reservoir sampling across all training history. Validation never enters
optimizer state.

With `--incremental yes`, finite-change probes, prediction audits, and replay
candidate scores recompute only affected raw sums from current activations.
This preserves exact candidate loss despite activation truncation and ReLU.
Full recomputation remains the default and stale replay caches still require
refreshing. The flag is checkpointed and cannot override a resumed experiment.

`--raw-cache yes` additionally retains the current example's raw hidden/output
sums. Candidate scoring and accepted transitions update only the affected sums,
preserving integer rounding and ReLU exactly. This uses 138 additional i64 values
for the MNIST shape, plus container metadata, as transient working memory outside
the optimizer-state cap. It is not replay history. The option defaults to off and
cannot override a resumed checkpoint. Both backprop and exact-probe routes can
use it; the latter supplies actual finite-change loss benefits to the evidence
banks, without compressing incoming signals to one trit.

MNIST defaults to 12 bytes/parameter for optimizer buffers, metadata, containers,
and replay. Growth uses spare capacity; replay cannot evict evidence to grow.
Reported allocation uses actual Vec capacities plus struct sizes. Temporary
buffers, allocator overhead, input dataset, model weights, and process runtime
are outside this **state** cap, and are not claimed to fit within it. Process peak
RSS is reported separately. Dataset loading differs from PyTorch, so RSS is not a
controlled memory-efficiency comparison. Tiny oracles use an explicit 64-KiB
budget because container overhead dominates a 16-weight model.

## Running

```sh
cargo test --locked -p tritium-train --example ternary_lab
cargo clippy --locked -p tritium-train --example ternary_lab -- -D warnings
cargo build --locked -p tritium-train --example ternary_lab --release
cargo run --locked -p tritium-train --example ternary_lab --release -- \
  --route probe --history both --steps 1000 \
  --checkpoint /tmp/toy-checkpoint.json --report /tmp/toy-report.json
python scripts/verify-ternary-lab.py /tmp/toy-checkpoint.json /tmp/toy-report.json
```

If a local `RUSTC_WRAPPER` points at an unavailable `sccache`, prefix Cargo commands
with `env RUSTC_WRAPPER=`. Do not change shared configuration to fix a local run.

For MNIST, provide the uncompressed canonical IDX files from
[the CVDF mirror](https://storage.googleapis.com/cvdf-datasets/mnist/):
`train-images-idx3-ubyte` and `train-labels-idx1-ubyte`. The loader enforces the
60,000-example train set and 28×28 shape. The first 55,000 examples are training;
the final 5,000 are validation. `--split test` additionally requires the t10k IDX
files and must be reserved for a selected final configuration.

```sh
# Find the executable under Cargo's configured target directory.
python scripts/run-ternary-lab.py --binary /path/to/release/examples/ternary_lab \
  --data /path/to/mnist --output /tmp/ternary-campaign --smoke
# Remove --smoke for the 10,000-step tuning / 20,000-step campaign.
```

Non-smoke tuning must run at least 3,176 steps: the controller requires four
observations per weight, and only 128 coordinates are visited per step. The old
2,000-step default could not exercise the controller across the model. The default
is now 10,000; the full campaign selected this budget before evaluation.

The Python tools need PyTorch (baseline only) and `blake3`. The campaign freezes
its binary and baseline script. It searches four thresholds for each integer
variant and four learning rates for the FP32 STE+Adam baseline, all on seed 101.
Tuning uses validation examples 0..999; five paired evaluation seeds use the
remaining 4,000. It records the protocol before running and writes each result
without overwriting an existing path. No network downloads happen inside training.

Two regimes match fresh-example count and counted linear contraction terms.
Replay exposures are additional and counted. Contraction terms are an arithmetic
work proxy, **not** FLOPs, hardware instructions, or wall-clock equivalence. A step
can overshoot the work budget; inspect the reported counts. Work comparisons
with more than 1% mismatch cannot pass advancement. Timing is reported
separately. Smoke runs force `advance=false` regardless of apparent differences.
The richer method advances only with a negative upper endpoint of the paired
95% t interval for loss difference versus simple momentum and no more than one
percentage point lower accuracy. The STE gap is reported separately. Five seeds
are a small research comparison, not release qualification.

Checkpoint resume accepts the same dataset and saved configuration; optimizer
options cannot override it. `--steps` adds steps. All model state, evidence,
precision, replay/cache versions, counters, and RNG state are saved. A separate
sample seed makes fresh-example order independent of optimizer RNG consumption.
Resumed allocations are normalized to preserve budget-dependent decisions.

## Verification and limits

Rust tests cover packed carries/range, signed stochastic rounding, integer
comparators, exact enumeration of all 27 tiny-network states, stale caches,
noise/hysteresis, zero recovery, budget pressure, corruption, and bit-exact resume.
The separately executed Python verifier decodes the accumulators independently,
reconstructs balanced-ternary digits, recalculates evaluation predictions/loss,
checks report/checkpoint identity, and rejects clipping. It does not approve
learning quality. Source scanning for float types supplements, rather than proves,
the integer-only arithmetic contract.

The external FP32 baseline uses explicit weight and activation STEs, identical
initial trits and fresh-example sampling, fixed scales, and squared loss. It is
intentionally outside the integer harness. This is a controlled small classifier,
not an optimized production MNIST recipe or a BitNet language-model baseline.

No language-model implementation, CUDA speedup, physical packed-weight claim,
quality qualification, or production activation follows from these tests. Those
remain separate work after the learning hypothesis survives the controlled ladder.

## Executed local evidence (2026-09-28, before the EAT-O naming change)

- `env RUSTC_WRAPPER= cargo test --locked -p tritium-train --example ternary_lab`: 16 tests passed.
- `env RUSTC_WRAPPER= cargo clippy --locked -p tritium-train --example ternary_lab -- -D warnings`: passed.
- `python -m unittest discover -s scripts/tests -p test_ternary_lab.py`: four verifier/campaign tests passed.
- `sh scripts/verify-gates.sh precommit`: passed on the working tree with no staged changes. Full workspace/CI/release gates were not run.
- Full smoke protocol: four tuning settings for nine variants, then five seeds in each of two comparison regimes. All 80 integer evaluation checkpoints were separately verified. Smoke runs do not satisfy a learning-quality gate.
- A 10,000-step MNIST diagnostic (`--history both --eval-limit 128`, seed 1) performed 16 accepted transitions with zero clipping. Optimizer/replay allocation was 1,219,040 bytes against a 1,219,584-byte cap, with 96 cached examples. Independent Python evaluation reproduced loss sum 8,142,899 and 28/128 correct. This small, single-seed slice is not a tuned quality comparison.
- Continuing that diagnostic for 100 steps produced a byte-identical checkpoint to a fresh uninterrupted 10,100-step run.

Diagnostic compiled-implementation BLAKE3:
`1f796758732ee78843c3501eecb37eabb0ef4955389dfbcb345d01cb56e2aeb3`.
Checkpoint BLAKE3:
`3536c88d6b508020a09c9e526b8a16f11a62503785b0d82ed4257bd33f20e413`.
Runtime checkout revision: `12e073281ca3ba7cc92aee77a99598fb64976b27`, with this
implementation uncommitted; the implementation digest binds the actual compiled
source rather than claiming it was present at HEAD. Temporary run artifacts live
under `/tmp/tritium-ternary-lab-campaign-final-smoke-v2` and
`/tmp/ternary-lab-mnist-10000{.json,.checkpoint.json}`. The full 20,000-step,
five-seed quality campaign was still unexecuted at this diagnostic stage.
The subsequent [full campaign](eat-o-classifier-results.md) completed all runs
and independent arithmetic checks, but did not clear advancement.
