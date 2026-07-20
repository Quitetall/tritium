# 0045 — PyTorch reference ternary module and conversion contract

Status: **DONE** (2026-07-20; commit `67fb0c9`)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Scope:** pure-PyTorch reference only; optimized dispatcher kernels are plan 0046

## Goal

Ship the stable reference seam that later native kernels optimize:
`TernaryConfig`, `Estimator`, `TernaryProjection`, `TernaryLinear`,
`prepare_qat`, structured errors and exact coverage reports. Success means a
normal PyTorch model converts transactionally, preserves parameters/tied
weights/state, performs a differentiable hard-ternary forward, and accounts for
every parameter without using Tritium's list bridge.

## Preconditions

- Branch `main` contains ADR 0033 commit `20431c5`.
- Worktree is clean before this plan starts.
- PyTorch is available to the Python test environment.
- Existing Conv1d/FSQ wrappers and tests remain unchanged.

## Public seams under test

- `tritium.nn.TernaryLinear`
- `tritium.torch.TernaryConfig`
- `tritium.torch.prepare_qat`
- `tritium.torch.inspect`
- `tritium.torch.Estimator` / `TernaryProjection`
- `tritium.torch.TritiumError` / `CoverageReport`

Tests observe only these seams. Optimized dispatcher choice, module-plan
internals and cache implementation are not test contracts.

## Steps

### Step 1 — Reference projection and `TernaryLinear`

- Add frozen configuration/error/coverage/projection types under
  `crates/tritium-py/python/tritium/torch/`.
- Add `AbsMeanSTE` using device-resident PyTorch operations. Forward values are
  exact per-row `{-scale, 0, +scale}`; backward uses masked STE with detached
  AbsMean scales.
- Add `tritium.nn.TernaryLinear` with `from_float`, ordinary `weight`/`bias`
  parameter names and estimator extra-state validation.
- Gate: fixed literal projection example, forward parity, activation/weight/bias
  gradients and state-dict round-trip.

### Step 2 — Transactional recursive conversion

- `prepare_qat` performs a read-only inspection pass before any mutation.
- Convert selected `nn.Linear` leaves; reject unknown targets and target matches
  with incompatible shapes/types.
- Preserve shared `Parameter` identity and qualified module paths.
- Account for every unique parameter once; record aliases, converted/preserved
  disposition, exact logical bytes and reason.
- Attach immutable coverage to the returned model; `inspect` returns it.
- Gate: nested model conversion, tied weights, root Linear, preserved norms,
  unknown-target rollback and coverage JSON round-trip.

### Step 3 — Package surface and regression

- Export `tritium.torch` and `tritium.nn` without making PyTorch mandatory for
  inference-only imports.
- Keep existing `tritium.autograd` surface compatible.
- Run focused Python suite and format/static checks.

## Gate

```bash
PYTHONPATH=crates/tritium-py/python pytest -q \
  crates/tritium-py/tests/test_torch_api.py \
  crates/tritium-py/tests/test_autograd.py
cargo fmt --check
git diff --check
```

PASS: all tests pass; no `.tolist()` appears in the new `tritium.torch` or
`tritium.nn` implementation; existing autograd tests remain green.

## Review

After commit, call lamu `review_commit` with this plan as `plan_file`. Verify
each finding at the cited line before changing code. Provider failure is an
infrastructure blocker and must be reported verbatim; it is not a PASS verdict.

## Commit

```text
feat(torch): add reference ternary linear conversion
```

## Done criterion

Focused gates pass, commit review has a verdict or an explicitly recorded
provider blocker, worktree is clean, and plan 0046 can replace the reference
execution without changing public behavior.

## Result

- Python package suite: **34 passed**.
- New device-resident reference path contains no `.tolist()` bridge.
- Hard forward, strict masked STE, root/nested conversion, shared modules,
  tied parameters, state identity, structured estimator errors and coverage
  round-trip are gated.
- Mandatory lamu `review_commit` was invoked after commit but its cloud provider
  returned HTTP 402 credit exhaustion before a verdict. Local lamu fallback
  returned `PASS`; all eight emitted nits were verified false positives or
  intentional contracts. Cloud review remains an infrastructure limitation,
  not a substituted PASS claim.
