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
- Bind block/sliding reconstruction, output-aware initialization and final
  teacher-logit loss into the resumable plan-0043 driver.
- Keep calibration, reconstruction and validation datasets source-bound and
  disjoint where the recipe requires.

## Step 3 — separate refinement tracks

Status: **IN PROGRESS** — `convert(prepared_qat)` now emits a separately typed
inference-only `QatHardResult`, preflights every unique projection before graph
mutation, packs tied Linear/Embedding masters once and binds source-checkpoint,
recipe and compact-state identities. Built-in hard forwards now decode exact
stored f16 scales. TTQ's asymmetric positive/negative values are represented as
two honest row-scale planes and therefore require `planes=2`; the prior
per-element-scale pseudo-plane could not enter SALT. Generic QAT-hard durable
export/reload, convolution lowering and public HF checkpoint registration remain
open and are not relabeled as PTQ.

- Remove refinement from `TernaryConfig.ptq`; reject ambiguous legacy non-none
  refinement values instead of silently migrating them.
- Implement `refine(parent, teacher, training, validation, config, work_dir)`.
  Scale-only, true alternating hard PV and S34 refinement are distinct child
  work identities and result discriminants.
- Bind every child to its immediate parent and complete ancestry. Scale-only
  freezes trits/allocation; hard PV may change assignments only under its
  declared structure and frozen nested-prefix constraints.
- Export the hard package and prove reload parity; no latent residual may enter
  a PTQ claim. QAT checkpoint, QAT export, PTQ, scale-only and hard-PV artifacts
  remain separately typed.
- Run matched-physical-byte ablations against RTN/AbsMean, GPTQ/AWQ-style
  second-order baselines and the frozen SALT variants.

## Verification

```bash
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests
python -m compileall -q crates/tritium-py/python/tritium
cargo fmt --check
git diff --check
```

## Done criterion

All catalog/plugin and production reconstruction/refinement gates pass; hard
artifact receipts feed plan 0043 without relabeling refined results as PTQ.
