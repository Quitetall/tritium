# Installed observability candidate-binding repair

## Result and scope

Source `65d3b3c1509a3f6aa8b9e80ac0a1a53d05da8872` checks the shared native
source/release/wheel guard before the observability producer executes a model
or the CLI replay processes a receipt. The executing qualifier's origin must
also belong to the candidate wheel. Observability's existing stricter installed
inventory, telemetry parsers, receipt schema, and numerical gates are retained.
This repairs implementation of ADR 0033's exact-candidate contract; it does not
create a new receipt kind or approve release qualification.

The cause was specific: the old helper matched installed files to wheel payloads
but did not compare the supplied source/release with the native package, and
did not check the executing qualifier's own origin. The producer and replay
both shared those omissions. The existing opaque-wheel rejection already worked.

## Red-capable tests, then clean candidate checks

Baseline: the archived official wheel for source
`8cf96e6bf39bad385353daf7d81cfa945e9073d4`, SHA-256
`0f1d5e1be1ed25c904e03ca366b02ebdef87ac68780fa85dab2caae21b9c3d80`.
Four source/release call-site tests failed twice in 0.56/0.49 seconds by reaching
forbidden model/receipt execution. Expanding to producer/replay origin and wheel
cases yielded six failures and two passes before the repair. An initial replay
attempt encountered missing telemetry dependency metadata; its result is not
counted as evidence of the admission bug. The minimized replay tests isolate
that unrelated runtime-version lookup. These tests are admission probes, not
mocked empirical qualification.

The regression command uses an actual executing candidate wheel, not invented
wheel bytes for source/release cases:

```sh
TRITIUM_TEST_INSTALLED_WHEEL=1 TRITIUM_TEST_CANDIDATE_WHEEL_DIR=dist \
  python -I -m pytest -q \
  crates/tritium-py/tests/test_observability_candidate_provenance.py
```

Missing candidate-directory configuration is explicitly skipped in developer
runs; CI sets the directory and runs all eight cases. Its Tritium install uses
`--no-compile`, preserving the source-free observability inventory contract.

Fresh local native-wheel checks passed **69 tests and 9 subtests** across the
observability, tutorial and HF provenance/lifecycle/export suites. The actual
hosted-wheel observability suite passed **30 tests**. Ten workflow/source
contract unit tests, actionlint, staged-tree formatting/Python compilation,
and normal pre-push gates passed. No bypass was used.

## Candidate identities and real adapter execution

| CPU wheel | Bytes | SHA-256 |
| --- | ---: | --- |
| Local `linux_x86_64` | 10,582,791 | `9ce6af6cdd19c2c50d968e8b638886af49444aa0cd0310b07018496a6022c9c5` |
| Hosted `manylinux_2_28_x86_64` | 10,573,908 | `c49068a17efc2594fd56649a5a631120ddcfca0d510617e9af474ffb8027eb1d` |

Both stamp exact source `65d3b3c1...` and version `1.1.0rc2`. The local Linux
tag is not a publishable PyPI wheel.

Real TensorBoard 2.21.0, offline W&B 0.28.1 and OpenTelemetry API/SDK 1.44.0
executed the fixed tied-weight QAT fixture against both candidates, followed
by separate installed CLI replay. Local CPython 3.14.7/Torch 2.11.0+cu130
execution hid CUDA and restricted PATH to an empty compiler directory. The
venv reused existing framework dependencies. This is not evidence that the
host has no compiler installed or that its dependencies came from a clean
container. A dependency install exceeded its 90-second deadline; after that
process was confirmed terminal, a bounded `--no-compile` retry completed.

Local-wheel receipt:
`sha256:1edb2f5cbf3d8236014dc3e7f21d2f11da7a6255e5a20cfdc5f0c94fc94a5bc9`.
Hosted-wheel local receipt:
`sha256:b2753005d5c3af12152a1de0e551285a74f3808cdcd21a2ba86bf7c8928c1e2a`.
Both retain five real telemetry files. Actual unmocked producer and replay CLI
calls reject false source/release before execution, without creating outputs.

[Hosted workflow 38048521591](https://github.com/Quitetall/tritium/actions/runs/38048521591)
binds the same source. Installed-wheel job `114203533034` succeeded with **161
passed, 4 skipped**. Source/compiler-free Python 3.13 job `114203533029`
succeeded, including actual telemetry production and installed replay. The abi3
matrix admission succeeded. CUDA was skipped, not qualified. At this record's
observation the overall workflow was still running; completed jobs are not a
claim that every remaining job or the full release passed.

Source-free artifact `11668587221` has ZIP SHA-256
`646b0e5589277ecef0d32e50643af492b325e72c31b4d1184a8e22d9fa37d9b7`.
Its observability receipt is
`sha256:9a0855d5cd72b84d14c4fe74905961a5e9eb985905779e8b38b627522c10189b`,
binding five telemetry files/8,781 bytes and the `c49068a1...` wheel.
Separate local portable validation checks its receipt and every telemetry byte,
plus both local receipt trees and the three retained hosted QAT/HF trees.
This is six portable checks, not installed replay of a relocated CI runtime.

## Remaining empirical limitation

The qualifier currently supplies `runtime/decode_ms=3.5`,
`memory/resident_bytes=4096` and `teacher_kl=0.125` as fixed caller samples.
Their actual adapter encoding/decoding is exercised, but their values are not
measured performance, residency or teacher quality. This repair does not turn
them into empirical evidence. ADR 0033's prohibition on copied counters still
applies: complete empirical diagnostics clearance remains unproven. Source-bound
measurements and the necessary governed contract migration must precede claiming
that full obligation is satisfied. Do not weaken an existing numerical gate or
rewrite these immutable v1 receipts to manufacture that result.

This batch does not qualify distributed training, representative estimator or
refinement campaigns, native kernel performance, physical backend/browser
coverage, Qwen language/MTP quality, serving/deployment, or public activation.

## Custody and next work

Evidence is retained outside scratch at
`/home/brianklam/Projects/Tritium/archive/verification/observability-binding-65d3b3c1-20261010`:
both wheels, original metadata/SBOM and ZIP identities, red/green and dependency
failure logs, actual telemetry/receipt trees, hosted job metadata/logs, and
portable verification commands. Checksums and relocated portable checks pass
before deleting this task's venv, pytest trees and scratch. Generated payloads
are not committed to Git. Old campaign outputs and unapproved August caches
are not part of this cleanup. Foreign staged diff identity remains
`c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`.

Next audit targets are the estimator worker and distributed worker's actual
installed-candidate admission. The fixed diagnostic samples above are a
separate measurement/evidence-contract task, not a reason to relabel this
software integration pass as full empirical release clearance.
