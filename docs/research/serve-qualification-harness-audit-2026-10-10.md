# Serving qualification execution-bound audit — 2026-10-10

Scope: **software verification only**. No Docker image, production Qwen model,
Kubernetes cluster, or cloud workload was run for this change. No qualification
receipt was produced. The release gates remain open.

Source baseline: `1865d4498876f9556deac282add82f5fdd0ec030`.
Decision: private ADR 0051, decision 14; serving work order: plan 0052.

## Finding and repair

`scripts/qualify-oci-runtime.py` previously executed its queue saturation and
slow-SSE/disconnect workload in the qualifier's process. HTTP socket timeouts
are not absolute workload deadlines. A callback that never returns strands
`future.result()` and `ThreadPoolExecutor` teardown, including response cleanup.
A peer delivering bytes slowly can also keep a read alive beyond the intended
total budget. The parent cannot safely stop an arbitrary Python thread.

The existing workload now runs inside an owned fork-isolated process. The
parent owns one monotonic deadline established before worker startup. New
`--queue-workload-timeout` defaults to 600 seconds, independently of each HTTP
`--request-timeout`. The workload includes startup, metrics, HTTP opens/reads,
threaded flooding, slow-reader hold, response closure, telemetry settlement,
and the post-disconnect generation. Its response objects remain in the child.

On timeout or interrupted observation, the parent terminates and joins the
child; after a one-second cleanup grace it escalates to kill and joins for one
more second. These cleanup graces are not extra successful-workload time or
model-cancellation latency thresholds. A missing, malformed, failed, late, or
unreapable worker cannot produce passing evidence. Successful evidence also
requires successful worker exit before the deadline.

IPC uses bounded JSON bytes, never external pickle data. A worker message is
limited to 4092 bytes and further reduced to the pipe's atomic-write bound
minus its four-byte framing header. Result fields must exactly match the 16
existing nonnegative integer fields; booleans are not integers. Failure text
redacts bearer credentials and the prompt before truncation. All qualifier
duration inputs must be positive and finite before Docker or model work.
Slow-reader hold still requires at least one second.

The qualifier fails closed where fork isolation is unavailable. This is a
requirement of this execution tool, not withdrawal of product platform support.

## Compatibility and empirical limitations

OCI runtime receipt schema `tritium.oci-runtime-qualification.v4`, required
checks, identity bindings, and acceptance predicates are unchanged. Existing
receipts are not edited or regenerated. The execution safeguard neither adds
an empirical success criterion nor relaxes one.

The existing v4 workload checks queue saturation, strict rejection envelopes,
token production, queue/disconnect counter movement, worker liveness, and a
bounded one-token recovery. It does **not** by itself establish phase-specific
HTTP deadline/disconnect latency, idle-worker recovery, KV/resident-resource
baseline restoration, or all concurrency/failure combinations. Those require
a separately versioned evidence contract and real candidate-bound runs. Other
qualifier phases are not claimed to share this process-isolation safeguard.

## Reproduction and local evidence

Environment: Linux, Python 3.14. Commands used `PYTHONDONTWRITEBYTECODE=1` and
`TMPDIR=/mnt/4tb/tmp`. Tests use local fixtures/mocks; no result here qualifies a
model, a physical backend, a production image, or serving performance.

Before the repair:

```text
timeout 15s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python scripts/tests/test_qualify_oci_runtime.py \
  QualifyOciRuntimeTests.test_queue_workload_deadline_reaps_noncooperative_http_thread
Ran 1 test in 2.007s — FAILED
AssertionError: True is not false : queue qualification stranded after wall deadline
```

The test's outer process owns a separate safety deadline and kills/reaps the
stranded test process. After repair, the same regression passed in 0.156s.

Additional tests cover successful workload response closure/reaping, SIGTERM
resistance requiring SIGKILL, interrupt and startup-failure cleanup, redaction,
oversized/malformed/missing IPC, invalid fields, failed exit after a message,
a message followed by a stall, unsupported fork, and invalid finite-duration
inputs before any worker or Docker invocation.

```text
timeout 60s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python scripts/tests/test_qualify_oci_runtime.py
25 tests passed in 2.607s (including construction-failure cleanup)

timeout 120s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python scripts/tests/test_release_evidence_status.py
32 tests passed in 0.401s

timeout 120s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python scripts/tests/test_qualify_kubernetes_deployment.py
71 tests passed in 0.310s

timeout 120s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python -m unittest scripts.tests.test_release_status
12 tests passed in 1.976s
```

An initial direct invocation of `test_release_status.py` failed with
`ModuleNotFoundError: No module named 'scripts'`; its supported module invocation
above passed without a source change.

The first complete script-suite sweep passed all 590 tests in 95.733s, before
the construction-failure cleanup test was added. Final-source results are
recorded separately below; this older sweep is not substituted for them.

The final-source sweep used:

```text
timeout 300s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python -m unittest discover -s scripts/tests -p 'test_*.py'
Ran 591 tests in 34.208s — FAILED (failures=1)
test_public_convert_parallelizes_rows_without_changing_fitted_artifact:
AssertionError: 1.0836769983041694 not greater than or equal to 1.5
```

All other 590 tests passed. The PTQ assertion was not changed. Its isolated
reproduction also failed: `python -m unittest scripts.tests.test_ptq_parallelism`
ran one test in 6.806s, measuring 0.070s/0.073s (0.96x), with identical fitted
outputs but below the required 1.5x speedup.

The isolated probe imports `/home/brianklam/.local/lib/python3.14/site-packages`
distributions `tritium` 1.0.0 and `tritium-torch` 1.1.0rc1. The loaded
`tritium/_tritium.abi3.so` has no `fit_joint_ternary_dense` or other current
`fit_joint_ternary_*` functions; its observed fit/quantization exports are only
the STE quantization helpers. That installation cannot qualify this checkout's
native Rayon fitter. The earlier sweep's different timing/path remains
unexplained; it is not proof that the ambient installation is reliable.
Exact-current-wheel PTQ verification remains open. Neither the passing earlier
sweep nor this provenance diagnosis converts the final sweep into PASS.

The shared sweep emitted Python 3.14's warning about fork in a multithreaded
process. The owned worker still has a bounded parent deadline and fails closed
if forked work cannot finish; this warning is not suppressed or treated as a
production qualification result.

The canonical staged-tree gate `scripts/verify-gates.sh precommit` passed:
staged whitespace checks, staged-tree Rust formatting, and compilation of both
changed Python files. It did not format or commit unrelated working-tree WIP.

The prior baseline's hosted runs were rechecked: CI `38029358979`, wheels
`38029358983`, CodeQL `38029358986`, capstone `38029358987`, and docs
`38029358989` all completed successfully. These results belong to the baseline,
not to this new change and not to empirical release qualification.

No persistent per-run scratch was created. Test-owned temporary files/processes
are cleaned by their owners. Existing build caches, unrelated staged work, and
the August campaign/offload directories remain untouched.
