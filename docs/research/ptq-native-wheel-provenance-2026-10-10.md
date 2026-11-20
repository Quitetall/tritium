# Public PTQ probe: exact native wheel provenance — 2026-10-10

Scope: local software and small-matrix CPU regression checks. This is not Qwen
quality, model-level performance, release approval, or a claim of SOTA status.

## Diagnosis

The previous final script sweep failed the public PTQ 1.5x parallelism gate
while importing the global `tritium` 1.0.0 / `tritium-torch` 1.1.0rc1 installation.
Its native extension lacks the current `fit_joint_ternary_*` entrypoints. ADR
0033 names **pytritium** as the distribution and retains `tritium` as its import
namespace; an importable namespace alone does not prove the correct package.

An isolated repetition against that global installation measured
0.298s serial / 0.276s parallel (1.08x), below the unchanged 1.5x threshold.
It did not measure this checkout's native fitter.

The exact-source CI wheel was available, so no duplicate local Rust build was
needed. GitHub run `38030600128`, artifact `11661124830` (`wheel-linux-x86_64`),
binds source `644af423ca771259cfaabbed7a80fa0f6ea9d90f`.

- Wheel: `pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`
- Bytes: `10,572,176`
- SHA-256: `5f1ff4c34f940dc67f931a4bdc871dbb6b1f2f5e0e2d634df7335554ca7fa933`
- Embedded identity: `source-git:644af423ca771259cfaabbed7a80fa0f6ea9d90f`

`scripts/verify-wheel.py` passed structural/version/RECORD verification, without
claiming an install smoke from that invocation. The downloaded CI compatibility
receipt and SBOM were retained unchanged alongside the wheel. A temporary venv
installed only that exact local wheel, using `--no-index --no-deps`; it reused
existing system/user-site dependencies and therefore is **not** a clean-install
qualification. No global package was installed, removed or replaced.

Host: Intel Core i9-14900K, Linux x86_64, 32 logical CPUs. Python 3.14.7,
PyTorch 2.11.0+cu130, CPU fitting only. Rayon used one/four threads, with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in each fresh probe subprocess.

The unchanged original probe against the correct wheel passed in 8.664s:
1.413s serial / 0.388s parallel, **3.64x**, with identical fitted artifact bytes
and weighted MSE. This identifies the wrong-install problem separately from
native fitting throughput.

## Test hardening, not a relaxed gate

`scripts/tests/test_ptq_parallelism.py` now checks installation before timing:

- the `pytritium` distribution must exist;
- the objective-returning native grouped fitter must exist;
- native source identity must be clean, and match `TRITIUM_SOURCE_REVISION`
  when the caller supplies it (as the wheel workflow does);
- package init, PTQ implementation and native extension must belong to that
  distribution's RECORD and match their recorded sizes and SHA-256-or-stronger
  hashes; shadowed namespaces and symlinked members fail closed;
- the standalone child embeds the same guard without importing the repository
  scripts or injecting source Python modules into its import path;
- the actual public `convert()` must call the native fitter and fit all 2048
  rows, rather than merely having an unused native capability available;
- serial and parallel runs must retain the same installation, native work,
  algorithm, fitted plane/scale digest and weighted MSE.

The existing 2048x256 public-conversion workload, seed, one/four Rayon threads,
90s subprocess deadline, deployment recipe and **1.5x** speedup predicate remain.
Call instrumentation only forwards to the original native entrypoint in the
owned subprocess; it is not a fake fitter or a native-only microbenchmark.
No product API, artifact format, or qualification-receipt schema was changed.

The new contract tests first failed on the absent guard. They now cover missing
distribution/current fitter, dirty/unverified/wrong source, invalid expected
revision, namespace shadowing, missing/weak RECORD integrity, installed-byte
tampering, symlinks, exact child-guard reuse, Python fallback, and partial row
coverage. These fixture tests are not empirical qualification.

## Observed checks and limitations

With the guard, a correct-wheel probe passed at 3.30x. A later correct-wheel
instrumented attempt failed at **0.95x** (25.078s / 26.330s; 68.460s for 11 tests).
That slowdown is not explained or discarded. Other work was running on the
machine, but load is not established as its cause. Do not call all attempts
passing or claim stable model-level throughput from selected successful trials.

The first full exact-wheel script sweep passed **601 tests in 57.407s**. Its
native probe measured 1.195s / 0.446s, 2.68x, with one native call covering 2048
rows. After the final two contract tests were added, the focused 12-test
contract suite passed in 0.054s, and the native probe passed in 8.706s:
1.178s / 0.344s, **3.43x**, with the same one-call/2048-row coverage.

The fitted plane/scale SHA-256 for the traced successful runs is
`78154db0ec95e67aa646ffb4642c271462903c37de46977c3fa92182a91c1efb`.
The old global installation now fails before timing with
`RuntimeError: PTQ parallelism requires an installed pytritium wheel` (2.974s).
This is an intentional negative control, not a passing performance result.

The full script sweep emitted Python 3.14's warning about fork in a
multithreaded process. It was not suppressed. The separately tested OCI worker
execution bound remains fail-closed; this warning is not a production receipt.

All baseline hosted runs for `644af423` completed successfully: CI
`38030600083`, wheels `38030600128`, CodeQL `38030600079`, capstone
`38030600111`, and docs `38030600089`. The wheel workflow's installed-Torch
probe, source-free tutorial and complete abi3 interpreter matrix passed.
Its CUDA lane was skipped by runner policy, not qualified by these CPU results.
These hosted results belong to that baseline, not the new test-hardening commit.

## Durable artifact and reproduction

The wheel, original CI compatibility receipt and SBOM are saved outside scratch:

`/home/brianklam/Projects/Tritium/archive/verification/ptq-wheel-644af423-20261010/wheel/`

To reproduce without replacing global Tritium, create a temporary
system-site-packages venv under `/mnt/4tb/tmp`, install the retained wheel with
`pip --isolated --no-index --no-deps --ignore-installed`, then run from the repo:

```text
env PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 TMPDIR=<owned-scratch> \
  TRITIUM_SOURCE_REVISION=644af423ca771259cfaabbed7a80fa0f6ea9d90f \
  <venv>/bin/python -m unittest scripts.tests.test_ptq_parallelism -v
```

Use explicit timeouts and record dependency versions/CPU load. Final-source
script-sweep and staged-tree hook results are recorded separately below.
The temporary venv/scratch must be removed after recording evidence. Existing
August artifacts and all unrelated source/staged work remain untouched.

Final-source script sweep:

```text
timeout 300s env PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 TMPDIR=<owned-scratch> \
  TRITIUM_SOURCE_REVISION=644af423ca771259cfaabbed7a80fa0f6ea9d90f \
  <venv>/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'
Ran 603 tests in 39.596s — OK
```

Its public probe measured 1.246s serial / 0.342s parallel, 3.65x, with the
recorded exact source identity, identical fitted digest/MSE, and one native
call covering all 2048 rows. This final software-suite result does not erase
the failed timing trial above or create a release qualification receipt.

Installed-wheel artifact/custody tests also passed without source-package
injection:

```text
timeout 300s env PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 TMPDIR=<owned-scratch> \
  TRITIUM_TEST_INSTALLED_WHEEL=1 <venv>/bin/python -m pytest -q \
  crates/tritium-py/tests/test_ptq_artifacts.py \
  crates/tritium-py/tests/test_bundle_binding.py \
  crates/tritium-py/tests/test_qualify_onnx_worker.py
123 passed in 2.44s
```

These are small-fixture artifact/custody tests, not whole-Qwen ONNX parity or
quality evidence. `scripts/verify-gates.sh precommit` passed staged whitespace,
staged-tree Rust formatting and compilation of both changed Python files.
Normal commit/push hooks remain enabled. No release gate or performance
threshold was bypassed.
