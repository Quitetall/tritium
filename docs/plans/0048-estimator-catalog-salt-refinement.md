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
- Bind block/sliding reconstruction, output-aware initialization and final
  teacher-logit loss into the resumable plan-0043 driver.
- Keep calibration, reconstruction and validation datasets source-bound and
  disjoint where the recipe requires.

## Step 3 — separate refinement tracks

- Implement scale-only, true alternating hard PV and S34 refinement as distinct
  work identities.
- Export the hard package and prove reload parity; no latent residual may enter
  a PTQ claim.
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
