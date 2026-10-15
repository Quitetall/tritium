# Qualification Git context — 2026-10-10

Status: **source-admission regressions repaired; empirical gates remain open**.
Repair source: `d9b59414412c2860f50b1cabbdbbda477ac08081`.

## Actual failure pattern

Twelve qualification frontends ran Git in a requested directory without clearing
inherited repository-local selectors. `GIT_DIR`, `GIT_WORK_TREE`, index/common
directory and object-directory selectors could redirect the command into a
clean foreign checkout. A dirty requested linked worktree with the same HEAD
was admitted; a different foreign revision could be accepted, while the correct
requested revision was rejected. No receipt-format forgery was needed to bypass
that source preflight. This is related to, but distinct from, the earlier
pre-push hook isolation repair.

Two real disposable-repository runs each report 36 failed rejection subchecks
and 12 erroneous valid-checkout rejections. The full matrix has 96 subchecks
across twelve frontends, staged/unstaged edits, inherited selectors/config and
valid/foreign revision cases. The minimized refinement probe shows:

```text
inherited-foreign-selectors dirty-checkout-admitted=True
only-git-selectors-removed dirty-checkout-admitted=False
```

Only selector removal changes that probe's verdict. Independent Git calls prove
the requested worktree is dirty, the foreign checkout clean and the two linked
checkouts initially share the claimed commit. These are source-admission tests,
not training, model-quality or physical-performance evidence.

## Repair and validation

`scripts/_qualification_git.py` obtains Git's local-variable inventory without
inheriting Git context, checks its required selectors, then removes those local
selectors and config-injection pairs only from each child environment. It uses
the requested directory, leaves the parent environment unchanged, preserves
global operator context and bounds both Git subprocesses to 30 seconds. Missing
Git, timeout, malformed discovery or command failure rejects admission.
`GIT_NO_REPLACE_OBJECTS=1` additionally binds original source objects rather than
a local replacement-ref view under an unchanged advertised revision.

The same helper now covers estimator catalog, PTQ refinement, matched-byte
ablation, ONNX inference, CPU/CUDA dispatch, training backends/performance,
browser training, model-zoo/community, installed API and HF distributed source
checks. Public arguments, receipt schemas, numerical/workload gates and existing
tracked-clean policy are unchanged. No receipt or release activation is emitted
by these tests. This is not a hostile-Git/Python sandbox or an audit of every Git
consumer; workflow-source, Stage-7 recipe and reproduction call sites still need
their own source-context checks.

Final focused suite: **63 tests passed in 2.438s**. Six new tests include the
96-case matrix, three real frontend CLI rejections before output, discovery and
timeout negatives, and twelve replacement-ref rejection subchecks. The minimized
probe now rejects the dirty requested checkout in both environments. Parent
environment, foreign HEAD/index/files and clean requested linked checkouts remain
unchanged. Normal commit hooks pass.

The first post-fix run hit 10-second Git fixture-setup timeouts during shared-host
memory pressure; those logs are retained, not called behavioral passes. Fixture
setup is now explicitly bounded at 30 seconds. A mistaken nonexistent API-test
module name also failed one invocation; the corrected actual module passes.
Earlier focused suites passed 61/62 tests; the final suite above is authoritative.
No production numerical/time gate was relaxed.

Previous diagnostic-record source `c102b7a9f5032c01621109cfbfaf31402ce24a1e` has
successful completed wheel workflow `38053892865` and CI `38053892876`.
Those results do not qualify this later repair or clear proposed diagnostics v2.

Durable logs, marked counterexamples, source snapshots and probe:
`/home/brianklam/Projects/Tritium/archive/verification/qualification-git-context-d9b59414-20261010`.
Owned per-run scratch is removed after verification; no campaign output,
unapproved old cache or foreign dirty/staged work is deleted.

## Release work still open

Measured diagnostics migration/adoption and catalog clearance; representative
PTQ/refinement and matched baselines; physical multi-device/backend/browser and
performance; flagship Qwen language/MTP quality/runtime/residency/reproduction;
serving performance/faults; production OCI/security/deployment evidence; final
RC/model zoo/community; second-machine/operator clearance and human activation.
