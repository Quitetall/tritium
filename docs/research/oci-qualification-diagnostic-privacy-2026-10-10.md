# OCI qualification failure-diagnostic privacy

Date: 2026-10-10. Parent source:
`e83832eb721f0e2a1cbd3f3da20d458bde45b026`.

This is scoped software-verification evidence for plan 0052 and ADR 0033's
hardened-serving work. It is not an OCI runtime/security receipt, model-quality
result, independent security clearance, or release qualification.

## Reproduced failure and repair

The OCI qualification harness previously rendered complete command arguments,
raw subprocess stderr, HTTP URLs, unexpected response bodies, and chained
transport/subprocess exceptions. Compose receives transient bearer credentials
through its environment; a failed tool can echo them into captured diagnostics.
Failure formatting must therefore not trust those inputs.

Regression tests were written before the respective repairs, using only dummy
markers. The initial command reproduction failed four assertions, twice. The
HTTP URL/body/transport reproduction failed five assertions. A further SSE
rejection test failed because Python's implicit exception context still
rendered the server's HTTP reason. These were failures at the actual helper
seams, not tests of a separate redaction utility.

The existing helpers now report a fixed failure category, a fixed allowlisted
tool label where applicable, and a numeric subprocess exit code. They do not
publish the raw diagnostic inputs described above. Exception chaining is
suppressed on the repaired paths. HTTP error bodies are closed even when their
reads fail. Successful subprocess output remains unchanged.

The nine diagnostic tests cover real child-process failure and timeout, mocked
launch/subprocess exceptions, successful output, allowlisted tool labeling,
HTTP mismatch and transport failures, failed error-body reads and closure,
and implicit HTTP exception context on SSE rejection. The tests inspect the
formatted traceback, not only the final exception message. No real credential
or private payload is used.

Receipt v4, `CHECKS`, `QUEUE_RESULT_FIELDS`, and `REQUIRED_STARTUP` are unchanged.
The 429 status, canonical error envelope and `Retry-After: 1` predicates remain
exact comparisons. No acceptance threshold, public schema, model admission,
workload, or MTP promotion rule was weakened.

## Observed checks

Run from the main checkout with bytecode writes disabled and temporary files
on the existing SSD scratch volume:

```bash
timeout 120 env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python3 -m unittest scripts.tests.test_qualify_oci_runtime \
  scripts.tests.test_release_evidence_status
```

Result: **PASS**, 66 tests in 12.949s. A subsequent in-process loop loaded and
ran `scripts.tests.test_qualify_oci_runtime.CommandDiagnosticsTests` twenty
times: **PASS**, 180 tests. The real timeout regression is bounded at 0.1s and
checks that the helper returns within 3s; `subprocess.run` owns termination and
reaping of that child.

An AST comparison against the parent commit confirmed the four frozen receipt
constant values above are unchanged. Scoped `git diff --check` passed.

Adjacent admission/contract checks:

```bash
timeout 120 env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python3 -m unittest scripts.tests.test_qualify_kubernetes_deployment \
  scripts.tests.test_release_status scripts.tests.test_qualify_oci_security \
  scripts.tests.test_oci_contract scripts.tests.test_verify_oci_archive
```

Result: **PASS**, 119 tests in 2.316s. These are software/fixture checks, not
executed Kubernetes, image-scan or deployment qualification.

Verified implementation bytes:

| File | SHA-256 |
| --- | --- |
| `scripts/qualify-oci-runtime.py` | `19fde495db10b65681cff5d3033c9362b8e279573910f1c9d1a6cb4e19e327e5` |
| `scripts/tests/test_qualify_oci_runtime.py` | `37733d45ef92dacf3b6657dc4faf6a160ce857ef974e59be0c946e95d9c12ce9` |

## Boundaries and remaining work

This does not establish whole-repository secret safety or all possible logging
paths. Successful tool output is still parsed by existing qualification code.
The subprocess/HTTP tests do not execute Docker, a real model, Kubernetes or a
physical GPU. Real exact-candidate OCI security/runtime and deployment gates
remain open, as do candidate-bound serving latency/resource/failure evidence.

Whole-Qwen ONNX parity is separately blocked on an admitted production MTP
oracle and promotion path. The compiled authorization ledger currently has a
synthetic fixture, not production approval. The packed model's unverified
drafter is not substituted for a qualified reference and `mtp_verified` is not
enabled by this work.

The full release still requires PyTorch/HF empirical qualification, physical
backend/browser matrices, complete exact-source package/tutorial/Colab
evidence, authorized recipe/Qwen PTQ and refined quality/runtime/reproduction,
production deployment qualification, an audited four-model zoo and community
deliverables, independent clearance/replay, signed release and explicit human
activation/publication. No release verdict is changed.

No persistent per-run scratch was created for these checks. Test-managed
temporary directories and child processes were cleaned up. The shared build
cache, unrelated worktree edits and August campaign artifacts were preserved;
the proposed August cleanup still has no owner approval.

## Security scanner follow-up

Parent source: `07879b5fe6b596f180d2d31f9a672215845a1b43`. This follow-up repairs
the analogous diagnostic boundary in `scripts/qualify-oci-security.py` and
rejects non-finite/nonpositive scan limits before input access or execution.

The original scanner helper copied raw stderr and chained subprocess exceptions
into failures. Its timeout and producer database-age checks also admitted NaN
or infinite values through comparison-only validation. The six minimized
regressions failed 16 assertions on two runs before repair. No scanner, network,
image, real secret or GPU was used to construct that signal. The real child
failure/timeout tests use synthetic markers; launch failures and invalid-limit
preflight are exercised with injected failures at the actual helper seams.

The repair keeps fixed scanner failure categories and numeric exit status,
withholds raw subprocess diagnostics and suppresses rendered exception context.
Both direct scanner execution and qualifier preflight require finite positive
numeric timeout; database age remains finite, positive and at most 24 hours.
Successful scanner metadata and report contents are unchanged. The final seven
execution tests passed twenty repetitions (140 tests). No temporary debug
instrumentation or separate redaction framework was added.

Final command:

```bash
timeout 120 env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python3 -m unittest scripts.tests.test_qualify_oci_security \
  scripts.tests.test_qualify_oci_runtime scripts.tests.test_release_evidence_status \
  scripts.tests.test_qualify_kubernetes_deployment scripts.tests.test_release_status \
  scripts.tests.test_oci_contract scripts.tests.test_verify_oci_archive
```

Result: **PASS**, 192 tests in 5.258s. The intermediate run before adding the
successful report-preservation test also passed (191 tests in 58.870s); neither
duration is a model/scanner throughput measurement. Scoped `git diff --check`
passed. AST comparison with the parent confirms unchanged receipt admission,
finding counts, archive staging, atomic publication, schema/version/report-size
constants and Trivy minimum version. No scanner command, zero-finding rule,
database freshness gate or receipt acceptance threshold was weakened.

| File | SHA-256 |
| --- | --- |
| `scripts/qualify-oci-security.py` | `251835475b50bd34531583e2a2f28895649eff601cc8443fb207abf7ffc72fd1` |
| `scripts/tests/test_qualify_oci_security.py` | `dfed07dbca396c8a98502af1f9485aa49590d7dd8fac5e9aca4e37e0ac8f9da9` |

This is execution-bound/diagnostic regression evidence, not an actual Trivy
vulnerability or secret scan, trusted security clearance, or a release receipt.
The earlier scope limitations and full-release obligations remain binding.
Test-owned children and temporary report files were cleaned up. No campaign
artifact, shared cache, unrelated WIP, running GPU workload or global install
was changed.
