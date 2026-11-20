# Installed estimator candidate admission

## Repair and evidence boundary

Source `5ca0c91c28930197cffb35949fe065ec155db19c` applies the shared installed
candidate guard before the estimator worker executes its catalog. Both public
`run()` and CLI paths now bind native source, installed release, wheel archive
and installed payload bytes, and the executing qualifier origin. The existing
seven-estimator catalog, numerical conditions and v1 execution trace are unchanged.

The worker previously checked argument shape and hashed whatever file the
caller supplied. The outer qualifier's clean-source check did not bind the
installed worker to that source. It was therefore insufficient protection for
either standalone invocation or a stale installed package.

This batch verifies a tiny CPU catalog and plugin lifecycle, not representative
checkpoint reconstruction/refinement, algorithmic superiority, distributed/GPU
training, flagship Qwen quality or public-release clearance. No candidate
manifest or release qualification receipt was invented or sealed here.

## Reproduce before repair

Baseline: the retained official CPU wheel for source
`65d3b3c1509a3f6aa8b9e80ac0a1a53d05da8872`, SHA-256
`c49068a17efc2594fd56649a5a631120ddcfca0d510617e9af474ffb8027eb1d`.
Eight entry-point tests failed twice: wrong source, wrong release, opaque wheel
bytes and foreign qualifier origin each reached forbidden model execution via
both public run and CLI. The real CLI additionally ran all seven estimators and
emitted `result=pass` traces for a false source and opaque non-wheel bytes.
The negative harness then failed its rejection assertion.

Those baseline outputs are named `baseline-INADMISSIBLE-*.json`. Their pass
labels demonstrate the bug; they are **not admissible release evidence**.
They were never fed into the release registry or sealed as qualification.

The direct CLI negative probe now rejects both cases before writing output.
The regression command uses the executing candidate archive:

```sh
TRITIUM_TEST_INSTALLED_WHEEL=1 TRITIUM_TEST_CANDIDATE_WHEEL_DIR=dist \
  python -I -m pytest -q \
  crates/tritium-py/tests/test_estimator_candidate_provenance.py
```

Developer runs without that directory are explicitly skipped. Hosted installed
tests set it. The existing opaque-wheel algorithm fixture now explicitly mocks
admission; it tests algorithms, not package provenance or qualification.

## Clean candidate results

Fresh local wheel checks passed **56 tests** across estimator catalog/worker,
estimator admission and related HF/observability candidate checks. The actual
hosted-wheel estimator suite passed **28 tests**. Nine qualifier/verifier and
workflow contract unit tests, actionlint, staged-tree formatting/Python
compilation and normal pre-push gates passed without bypasses.

| CPU wheel | Bytes | SHA-256 |
| --- | ---: | --- |
| Local `linux_x86_64` | 10,582,818 | `d6655a3c6edad07be2874578517a30b8ce894708bc381e6f6c21b718333efb15` |
| Hosted `manylinux_2_28_x86_64` | 10,573,932 | `dbedb830cf88bf5fc0339cba0f3c3f2e4969c528b76b6402b4f6ff732c15996d` |

Both stamp exact source `5ca0c91c...` and version `1.1.0rc2`. The local Linux
tag is not a publishable PyPI wheel. Local CPython 3.14.7/Torch 2.11.0+cu130
venvs reused existing framework dependencies without a Tritium source overlay.
These are not clean-dependency or compiler-free environments.

Actual CLI execution against each candidate passed all seven built-ins:
AbsMean STE, annealed STE, LSQ, SALT STE, sparse ternary, TTQ and TWN. All frozen
hard-trit/scale, gradient/state, tied-identity and coverage conditions passed.
External registration, duplicate rejection, contract validation, purity opt-in
and invalid-projection rejection also passed. Separate processes rechecked each
trace using the official trace loader plus literal-true result checks. This is
trace validation, not sealing the catalog's final release qualification receipt.

Hosted workflow
[38049581623](https://github.com/Quitetall/tritium/actions/runs/38049581623)
binds exact source `5ca0c91c...`. Linux artifact `11669450470` has ZIP SHA-256
`3812dc370c8d47e44f78810a10f97d6629851924055ee86784175bb454809411`.
The source/compiler-free job now executes the actual estimator worker and
retains `estimator-clean/trace.json`; the installed job runs the new admission
and worker tests. The complete workflow subsequently passed at that exact
source: Linux/macOS/Windows wheels, the ABI matrix, installed-wheel tests,
source-free execution and the pinned SmolLM2 job. CUDA was skipped, not qualified.

The installed-wheel job `114206463208` reports **172 passed, four skipped**.
Source/compiler-free job `114206463184` passed and retained the actual seven-case
catalog trace in artifact `11669400803`, ZIP SHA-256
`066249d3da7cbc658a12e2f9823faac563a1c3daff23cfb9a610556091757242`.
Its 2,676-byte `estimator-clean/trace.json` has SHA-256
`9525e862a235630d631f5b3f21133ce25261812a0e6ab6a03c86c3000895fbc3`.
All three retained estimator traces (local candidate, hosted candidate locally,
hosted source-free execution) pass separate identity/result validation after
relocation. Other retained tutorial receipts were not independently requalified
by this estimator batch. Workflow success is not final release admission.

## Custody and outstanding work

Evidence custody:
`/home/brianklam/Projects/Tritium/archive/verification/estimator-binding-5ca0c91c-20261010`.
Retain both candidates and hosted metadata/SBOM, original red and green logs,
the clearly marked inadmissible baseline traces, valid catalog traces, hosted
job metadata/logs and separate trace validation. Verify archive checksums and
relocated trace checks before removing this task's venv, pytest trees and
scratch. Generated payloads are not committed to Git. Old campaigns and
unapproved August caches are not part of cleanup. Foreign staged diff identity
remains `c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`.

Next: distributed-worker candidate admission, source-bound diagnostic
measurements replacing fixed caller samples, and final-candidate estimator
qualification with an authentic candidate inventory. Representative refinement,
matched baselines, physical hardware/browser, Qwen language/MTP quality/runtime,
serving/deployment, model-zoo and independent final-release gates remain open.
