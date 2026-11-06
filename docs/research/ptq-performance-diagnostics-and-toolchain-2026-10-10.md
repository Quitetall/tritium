# PTQ timing diagnostics and pinned foundation target — 2026-10-10

Code: `def8b6eefbf4a23a2fb5b4894745be91b36507ce` (includes `18380800`).
Base: `a776c7311f80eaefbf6891849dd8be1c6955d756`.

Scope: CI setup repair, benchmark observability and script-fixture repair.
No fitter, weight format, model artifact, numerical precision default or
performance threshold changed. This is not completion of ADR 0044 or the
v1.1 release.

## Performance failure remains unexplained, not erased

The previous wheel run `38061136994` at `4e42b029` failed the actual public
2048x256 `convert()` speedup gate: **1.388913829x versus 1.5x required**.
The exact CPU wheel was downloaded from that run's `wheel-linux-x86_64`
artifact, ID `11673365549`, and retained with its original receipt and SBOM.

- Wheel: `pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`.
- Bytes: 10,578,895.
- SHA-256: `260d7d016ca6dbd935f03b95ea3029c52564b899b701f34cf34ce4ea15457472`.
- Native identity: `source-git:4e42b029a3c93b75d6f953b8ea48ae2a8bf1c826`.

Structural/RECORD verification passed. A temporary system-site-packages venv
installed only that wheel with `--isolated --no-index --no-deps
--ignore-installed`. This reused existing dependencies and is not a
clean-install qualification. Global packages were not changed.

The unchanged original gate passed locally three times: **3.37x, 2.82x,
3.38x**, with identical fitted bytes. Local host: i9-14900K, Linux x86_64,
32 logical CPUs, Python 3.14.7, PyTorch 2.11.0+cu130; fitting used CPU only.
Separately, the ordinary new-source hosted wheel run `38063273525` at
`a776c731` passed the same gate at **2.54x** (2.060s / 0.811s). That run was
triggered by the source push, not a restart of the failed run.

These successes do not explain the earlier failure. The diagnosis loop reached
an agent-runnable, red-capable public-conversion test, but did not reproduce
the hosted timing failure locally. Causal tuning/bisection is therefore deferred
until another failing measurement has discriminating evidence. No speculative
performance fix, gate relaxation or claim of stable model-level speed is made.

## Diagnostics on the same public path

The existing objective-returning native fitter is still called directly.
Counters now reset immediately before `convert()`, excluding preparation or
calibration work from the native-coverage assertion. The measured total remains
the complete public conversion, not a native-only microbenchmark.

Additional diagnostic output includes native wall/process CPU time, total
conversion CPU time, remaining conversion wall time, Python/Torch versions and
bounded cgroup-v2 quota observations. Quota reads are at most 4096 bytes per
file and at most 64 membership components, with parent limits observed. Private
group paths are not printed. Missing/malformed/partial observations remain
explicitly incomplete, not asserted unlimited capacity or hardware admission.

The instrumented small-matrix gate passed at **2.09x**, with one native call
covering all 2048 rows. Native wall time was 1.112s serial / 0.344s parallel;
other conversion wall time was 0.424s / 0.392s. This distinguishes components
in that successful local sample; it is not the cause of the earlier hosted
failure. The fit digest stays
`78154db0ec95e67aa646ffb4642c271462903c37de46977c3fa92182a91c1efb`.

Fixture tests explicitly retain failure at 1.39x under the unchanged 1.5x
predicate, along with provenance and native row-coverage rejection tests.

## Actual hosted CI setup defect

CI `38063273509` at `a776c731` failed the required bare-metal check on all three
OS jobs. The Ubuntu log reports:

```text
error[E0463]: can't find crate for `core`
note: the thumbv7em-none-eabihf target may not be installed
```

The action installed the target for floating `stable`; Cargo resolves the
repository's pinned **1.98.0** toolchain. This is the same target-selection
class already documented for wasm in `rust-toolchain.toml`. Local
`cargo +stable check ... --target thumbv7em-none-eabihf` reproduced the exact
missing-core error in 0.05s; the repository toolchain check passes.

The mandatory foundation target now lives in `rust-toolchain.toml`, alongside
the required wasm targets. The erroneous floating-action target input is
removed. The pinned compiler version and required bare-metal command remain
unchanged. The configuration regression first failed on the absent target and
then passed; the real warnings-as-errors bare-metal check passed in 0.23s.
This proves compilation only, not MCU execution or physical qualification.
Fresh same-source hosted OS results are still required.

## Checks and retained failures

On the clean code commit:

```sh
env PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 TMPDIR=<owned-scratch> \
  TRITIUM_SOURCE_REVISION=4e42b029a3c93b75d6f953b8ea48ae2a8bf1c826 \
  <venv>/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'
```

**655 tests passed in 88.760s.** The controller is the new code commit; the
native benchmark component is the explicitly bound older `4e42b029` wheel,
not an installed wheel built from the new commit. Its public probe passed at
2.00x; this does not qualify a new-source release wheel.

The first sweep failed three cache-hook fixture tests: their all-success Git
stub returned an empty repository root after the production hook's existing
Git-context isolation change. The fixture now answers the root/context queries
and rejects unexpected command classes. Production hook behavior was not
weakened. The three cache tests and the real-Git origin/index isolation test
all pass (4 tests, 0.187s). Original red sweep output is retained.

Two early diagnostic-fixture invocations hit their 30s deadlines. A later
isolated unchanged case passed in 2.246s, and the complete 17-test diagnostic
contract suite passed in 0.083s. The timeout cause is unproven and is not
discarded as a passing attempt. Combined toolchain/diagnostic contracts pass
18 tests. The full sweep retains Python's multithreaded-fork warning rather
than suppressing it. Fixture `STAGE 7 PASS` text is not a Qwen measurement.

Canonical `scripts/verify-gates.sh prepush` completed with `MainPID=0`,
`SubState=exited`, `Result=success`, `ExecMainStatus=0`: formatting, projections,
bare-metal, default/all-features Clippy and Windows GNU cross-check pass.
Actionlint passes; local ShellCheck is absent, explicitly warned by the script.
Shell run-block lint is therefore not claimed locally. No shell block changed;
the hosted workflow lint remains authoritative for the new candidate.

Archive: `/home/brianklam/Projects/Tritium/archive/verification/ptq-diagnostics-def8b6ee-20261010`.
Retained-byte/source custody checking is not numerical or release approval.

## Remaining

Keep the unexplained timing failure open until discriminating hosted evidence
supports a measured fix or an accepted benchmark-method amendment. Complete
ADR 0044's remaining schema/projection/identity, basis-bound gather, bounded
evidence, streaming/sharded format, all-backend conformance, unified engine,
loader/runner/surface and migration/deletion work. Full Stage-7, Qwen
language/MTP quality/runtime/physical-byte/reproduction, physical distributed
and backend training, serving/deployment/security, candidate packaging/model
zoo/docs/community, independent clearance and human activation remain open.
