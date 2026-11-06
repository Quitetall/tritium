# Serving tree and solo speculative cancellation

Date: 2026-10-09 (EDT). Base: `13f8e0e3c4e0eff886f222bd1cb24aff782c472d`.
Decision: private research ADR 0051, decision 9; plan 0052.

## Change and scope

The runtime-free Generator interface now accepts a borrowed cancellation query
for tree-session opening and verification. Compatibility defaults check entry
only; they cannot interrupt arbitrary legacy work. The native RunnerGenerator
adapter forwards the query to controlled prefill and tree verification. An
entered, cancelled session open retires the new session; cancelled verification
preserves the committed session for retry. Pre-entry cancellation preserves the
prior session.

The real single worker queries response closure or drain, skips closed queued
tree jobs, and classifies cancellation separately from backend errors. Drain
retains typed `Draining`; a disconnected client receives no manufactured result
and cancellation does not latch a backend fault. Continuous-batch tree jobs use
the same native verification query. Solo speculative cycles forward stream
closure/drain into target work and retire request-owned target/draft state.

Greedy and sampled single-request speculation poll between draft/target work,
inside native target forward/verification, and before sampled host commit.
Cancelled generation resets request-owned target/draft state and closes solo
tree authority. Existing lookup/model drafting, graph eligibility, sampled
acceptance and grouped multi-slot dispatch remain enabled.

## Reproduction and tests

Three separate red loops preceded their respective fixes. Each failed twice,
once through Cargo and once through its built test binary:

- Real worker open/verify disconnect and drain: entered work was not stopped
  (`left: 1, right: 0`) or a drain published a result. Forwarding the query at
  worker dispatch and handling `None` separately fixed both tests. Each test
  covers session open and verification, then a successful queued recovery.
- Native serving adapters: session open and verification did not forward the
  native query. Their entry-only compatibility defaults failed the probe;
  native overrides restored transactional behavior.
- Actual greedy/top-k/top-p speculation: cancellation after the first emitted
  token still captured another target tree (`left: 1, right: 0`). Loop/query
  propagation avoids that work. The never-cancelled baseline must still emit
  four tokens and capture a tree, preventing a plain-decode dispatch bypass.

Additional physical adapter tests sweep every observed query for session open
and verification, assert committed KV bytes bit-for-bit, and require exact
ordinary recovery. Context 16 exercises captured trees; context 12289 naturally
exercises eager trees. A speculative sweep covers greedy/top-k/top-p with both
lookup and model drafting, checking emitted prefixes, target/draft retirement,
and exact recovery at every observed exposed query. This does not claim queries
inside the unchanged drafter chain.

Continuous-batch software tests cover preclosed/draining solo cycles and
disconnect/drain triggered inside a real host target projection, with empty and
nonempty committed prefixes. They require no output, unchanged history/counters,
bitwise committed KV rollback and exact recovery. The last-token budget naturally
selects a plain target step; it does not qualify batched CUDA tree propagation.
The prior physical paged-KV retirement test still runs using the shared fixture.

## Validation record

Build environment: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`,
`RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`. The existing cache
resolves to `/mnt/4tb/build/cargo/70/e33462ce184cce`.
Hardware: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
These are deterministic two-layer, hidden-64, vocabulary-8 physical fixtures,
not a downloaded model or the release candidate.

The first managed validation invocation
`10ee77fa27b045cea4bb560b16d585b1` passed 112 serving tests and three
runtime-free compatibility checks, then failed CUDA test compilation with
`error[E0433]: cannot find type Duration in this scope` at batch.rs. Qualifying
both test-only uses as `std::time::Duration::ZERO` fixed the compile error.
No later command from that failed sequential job is counted as passed.

Corrected invocation `8ed703dc243844819a2a76af70dcc2bc` completed scoped
warnings-denied CUDA all-target Clippy, three physical tree-adapter tests,
two physical speculative tests, two solo batch software tests, formatting and
whitespace checks. It used required-device f32 KV. The worker pair separately
passed 20 bounded repetitions (40 executed tests, 80 open/verify scenarios).

A separate unisolated f16 adapter-trio run failed two tests with
`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` and
`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. The identical trio passed with
`--test-threads=1`; another unisolated parallel run also passed, establishing a
race rather than a stable f16 mismatch. Source inspection found that independent
CudaBackend instances use the same device primary context and default stream.
This implicates cross-fixture capture interference; the exact driver interleave
was not instrumented and a general production fix is not claimed.

The shared fixture now holds a test-only device-ownership mutex for each entire
physical fixture test, including destruction. All graph/eager and model-draft
dispatch remains enabled; the normal parallel Rust harness still runs. These
tests qualify sequential worker ownership only. Concurrent independently owned
models sharing a primary context remain an explicit backend investigation and
qualification obligation, not a green result from this fixture isolation.

The first final-source repetition invocation
`ac043bb364fe4ea18397c5d085e762a2` also reproduced that capture interference in
both speculative tests with f32 KV (zero passed, two failed). It terminated
with exit 101 after passing the serving/default checks and scoped Clippy;
remaining repetitions and formatting from that sequential job are not passes.
This second observation confirms the issue is not limited to f16. The
subsequent fixture-isolated run uses a rebuilt binary, not that failing binary.

Fixture-isolated invocation `d42422cddb5f481ab2dfb36c9d1b7631` passed the
initial required-device f32 adapter trio, then three complete rounds per dtype
with f32 and f16 KV: tree adapters (3), speculation (2), solo-cycle software
(2) and pending/retirement (4). This is 69 executed CUDA-feature checks total,
including 39 tiny physical checks and 30 host software checks. Rust's default
parallel test harness remained enabled; the fixture ownership guard isolates
only physical fixture lifetimes. Test binary SHA-256:
`5bc0731427bf8040a824722642226b889d400727cf272a89409ab58b1fa9c82c`.

Exact commands (all with the build environment above; each Cargo command has
a 300/900-second timeout and each direct test call a 45-second timeout):

```sh
cargo test --locked -p tritium-serve --features serve
cargo test --locked -p tritium-serve --lib generator::tests::default_cancellable_
cargo clippy --locked -p tritium-serve --features cuda --all-targets -- -D warnings
TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f32 cargo test --locked \
  -p tritium-serve --features cuda --lib generator::tests::runner_tree_
```

The repetition body uses the built CUDA-feature lib-test binary above with
`TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=<f32|f16>`, `--quiet` and each of
`generator::tests::runner_tree_`, `generator::tests::speculative_loop_`,
`batch::tests::solo_spec_cycle_`, `batch::tests::pending_`, three rounds per
dtype. Formatting uses `cargo fmt --all -- --check`; whitespace uses
`git diff --check`. That managed invocation terminated successfully at
2026-10-09 23:33:05 EDT (`SubState=exited`, `ExecMainStatus=0`,
`Result=success`), also passing 112 serving tests, three runtime-free
compatibility checks, scoped CUDA all-target Clippy, formatting and whitespace
checks. The final binary digest matched the one above.

Two exploratory whole-lib CUDA sweeps, one per dtype, exceeded their initial
45-second limits and are not passes. A named f32 sweep with a 60-second limit
then completed all 75 tests in 53.21 seconds. Its last completed case was the
pre-existing model-backed `truncate_reconcile_pins`, which loads the existing
8-layer drafter three times; it was not an in-operation cancellation hang.
The model-backed check is still not a candidate-bound qualification receipt.

Hosted follow-up (2026-10-10): all five workflows for the exact implementation
commit `5ba8a79a5584d08c0df597abb2d296f481dceb8c` completed successfully:
CI (`38021878463`), wheels (`38021879596`), docs (`38021878528`), CodeQL
(`38021878457`) and CPU capstone (`38021878407`). Observed with
`gh run list --commit 5ba8a79a5584d08c0df597abb2d296f481dceb8c --json
name,status,conclusion,databaseId --limit 8`. This confirms the hosted software
checks for that commit; it does not close any candidate-bound empirical gate or
qualify the subsequent uncommitted CUDA ownership investigation.

## Limits and remaining gates

Local software and tiny physical checks do not qualify cancellation latency,
real-model resource bounds, model quality, SOTA, a release candidate or public
activation. Suite wall time is not a cancellation-latency measurement. A running
GPU kernel or graph is not preempted; cancellation can race the final commit.

Atomic multi-slot native verification, in-operation drafter chain/reconcile
and batched enrollment queries remain separate rollout obligations. Candidate-
bound worker/KV/resource/failure-matrix, exact CPU/CUDA OCI/security/Kubernetes
and independent release gates remain open. No large fitting/capture, cloud
spend, publication or campaign deletion was performed.

Only owned serving source, shared test fixture and this note are part of the
main-repository change. Unrelated source/docs edits and staging are preserved.
No new scratch directory, model payload or target directory was created; build
outputs reuse the existing project cache and normal hook scratch is automatic.
