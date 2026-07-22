# 0048 — Estimator catalog and SALT refinement

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Parallel dependency:** [plan 0043](./0043-salt-v2-sota-campaign.md)

## Goal

Make Tritium the authoritative implementation surface for ternary estimators
and production additive PTQ/refinement. Every algorithm has a hard exportable
forward, explicit backward policy, versioned identity, coverage/state
accounting and matched-byte evaluation.

## Public seams under test

- `tritium.torch.create_estimator`
- `tritium.torch.register_estimator`
- `tritium.torch.registered_estimators`
- `AnnealedSTE`, `LSQEstimator`, `TWNEstimator`, `TTQEstimator`, and
  `SparseTernaryEstimator`
- `TernaryConfig.qat(estimator=...)`
- `RefinementConfig.scale_only(...)` and `RefinementConfig.hard_pv(...)`
- `tritium.torch.refine`, `RefinementResult`, and artifact ancestry

## Step 1 — differentiable estimator catalog and plugins

- Built-in AbsMean/SALT STE remain the conformance base.
- Add annealed STE, LSQ, TWN, TTQ and explicit sparse-ternary estimators.
- Hard detached forward values must decode exactly from `{-1,0,+1}` trits and
  finite nonnegative scales; only backward surrogates may differ.
- Learned scale/threshold state is an ordinary `Parameter`, included exactly
  once in coverage and safetensors state.
- Modules sharing a latent master share one estimator instance.
- External factories register explicitly; duplicate registration fails closed.

Gate: literal hard decode, finite gradients for masters and estimator
parameters, config conversion, tied estimator identity, state/HF round-trip,
coverage and duplicate-plugin tests.

## Step 2 — production SALT reconstruction

- [x] Consume guided-Fisher/input-Hessian/forward-KL Kronecker factors without
  row-wise dense expansion; prove bit-exact solver and tensor-master parity with
  the materialized metric.
- [x] Canonically persist and bounded-reopen those per-tensor factors with
  source/evidence identity and adversarial corruption/count rejection.
- [x] Connect complete Qwen evidence namespaces to admitted checkpoint matrices
  and the resumable pure-PTQ master campaign with campaign-wide token-stream
  consistency.
- [x] Preserve standard SwiGLU QKV bias and conventional per-head Q/K RMSNorm
  constants through semantic identity, exact dense reconstruction, and
  replayable resident/packed CUDA training graphs. Qwen3.6 hybrid DeltaNet and
  gated-attention training remain a separate open integration.
- [x] Add a bounded-memory block/sliding-window evaluator with exact teacher and
  student stream identities, final-logit teacher CE/KL, complete deterministic
  multi-start selection and a strict canonical `TSV2OUT` v1 receipt.
  - Edit `crates/tritium-quantize/src/salt_v2_output.rs` for the public frozen
    schedule/spec, streamed accumulator, metric calculation and deterministic
    restart-selection seams.
  - Edit `crates/tritium-quantize/src/salt_v2_output/codec.rs` for the bounded,
    canonical receipt encoder and fail-closed strict reopen path.
  - Edit `crates/tritium-quantize/tests/output_reconstruction.rs` for literal
    metric, traversal, teacher-drift, corruption, allocation-amplification and
    unreachable-state coverage.
  - Expected output: `cargo test -p tritium-quantize --test
    output_reconstruction` reports six passed tests; `cargo clippy -p
    tritium-quantize --all-targets -- -D warnings` exits zero.
- Bind that receipt's selected candidate to its immutable tensor-master set in
  the resumable plan-0043 driver; execute it on checkpoint-scale evidence.
- Keep calibration, reconstruction and validation datasets source-bound and
  disjoint where the recipe requires.

## Step 3 — separate refinement tracks

Status: **IN PROGRESS** — `convert(prepared_qat)` now emits a separately typed
inference-only `QatHardResult`, preflights every unique projection before graph
mutation, packs tied Linear/Embedding masters once and binds source-checkpoint,
recipe and compact-state identities. Built-in hard forwards now decode exact
stored f16 scales. TTQ's asymmetric positive/negative values are represented as
two honest row-scale planes and therefore require `planes=2`; the prior
per-element-scale pseudo-plane could not enter SALT. Generic QAT-hard state now
exports through the shared `export` seam into a canonical SHA-256-ledgered
safetensors bundle, strict-reloads evidence without a source model, or binds to
an explicit dense/QAT shell with tied compact storage and exact hard-state
parity. Rehashed ancestry, latent masters, estimator state, invalid B3/scales,
unknown files, corrupt payloads and wrong shells fail closed. Convolution
lowering and public HF checkpoint registration remain open; none are relabeled
as PTQ. The post-PTQ refinement lifecycle is implemented: versioned scale-only
and hard-PV/S34 recipes consume an immutable PTQ/refinement parent, use bounded
row-wise second-order fitting, optimize hard candidates against held-out
teacher loss, and emit a strict child conversion plus complete ancestry.
Training/validation batches are individually hashed, aggregate-bound, and
cross-split overlap is rejected. Strict reload and atomic export rehash the
parent, evidence, child algorithm/recipe, S34 topology, payloads, and directory
publication. Every deployment-aligned G128 child additionally carries a native
seek-backed SALT package bound to the child conversion identity; deliberately
unaligned research fixtures retain the hard conversion but cannot claim a SALT
package.

- [x] Remove refinement from `TernaryConfig.ptq`; reject ambiguous legacy non-none
  refinement values instead of silently migrating them.
- [x] Implement `refine(parent, teacher, training, validation, config, work_dir)`.
  Scale-only, true alternating hard PV and S34 refinement are distinct child
  work identities and result discriminants.
- [x] Bind every child to its immediate parent and complete ancestry. Scale-only
  freezes trits/allocation; hard PV may change assignments only under its
  declared structure and frozen nested-prefix constraints.
- [x] Export the G128 hard package and prove reload parity; no latent residual may
  enter a PTQ claim. QAT checkpoint, QAT export, PTQ, scale-only and hard-PV
  artifacts remain separately typed. Unaligned research-only children are
  explicitly non-package artifacts.
- Run matched-physical-byte ablations against RTN/AbsMean, GPTQ/AWQ-style
  second-order baselines and the frozen SALT variants.

Release admission now has three separate fail-closed contracts.
`tritium.estimator-catalog-qualification.v1` binds the exact candidate wheel and
all seven built-ins, hard trits/scales, gradients/state/ties/coverage and
external-plugin rejection behavior. The installed-wheel worker emits the
retained `tritium.estimator-catalog-execution.v1` trace; the qualifier binds its
wheel hash, CPU environment and exact results into the release receipt.
`tritium.refinement-qualification.v1`
requires disjoint source-bound splits and distinct scale-only, hard-PV and S34
children with exact ancestry, held-out hard candidates, G128 native SALT
packages, strict reload and no latent residuals; it must descend from the
flagship NearLossless PTQ artifact and the estimator receipt.
`tritium.baseline-ablation-qualification.v1` freezes RTN/AbsMean,
GPTQ/AWQ-style, SALT V1 and three mechanism ablations, recomputes matched-byte
claim eligibility, and binds the exact refined model/evaluation lineage. These
contracts and producers are implemented. Refinement receipts are derived from
retained dataset/candidate ledgers, hard-structure deltas, package identities,
reload samples and validation losses. Ablation receipts are derived from
retained matched-byte timing/residency samples and frozen recipe identities.
Executing them against the final candidate wheel and checkpoint-scale campaign
artifacts remains open.

## Verification

```bash
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
python -m unittest scripts.tests.test_qualify_estimator_catalog
python -m unittest scripts.tests.test_qualify_refinement_campaign
python -m unittest scripts.tests.test_qualify_baseline_ablation
python -m unittest scripts.tests.test_verify_estimator_refinement_receipt
python -m compileall -q crates/tritium-py/python/tritium
cargo fmt --check
git diff --check
```

## Done criterion

All catalog/plugin and production reconstruction/refinement gates pass; hard
artifact receipts feed plan 0043 without relabeling refined results as PTQ.
