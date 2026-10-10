# Ordinary lockstep batch cancellation

Date: 2026-10-10. Decision: private ADR 0051 decision 13, plan 0052.
Base tree: `924121e8fb46304085211af41c71e4c9d79939f6`.

## Implementation

Native eager logits, graph logits and graph argmax expose borrowed-query,
optional-output controlled decode. Ordinary methods share those implementations
with never-cancel queries. Mathematical operations, graph eligibility, capture,
replay and readbacks remain unchanged. No sequential, solo or host decode is
substituted to make cancellation easier.

Entry cancellation avoids validation, capture and work. Entered eager work
checks after uploads/embedding, at layer/attention/MLP and head-row boundaries,
then before publication. Graph paths query only outside capture/replay, after
uploads and replay, and before publishing their whole result; graph-logit heads
also check between rows. No callback executes inside capture. Cancellation
settles both owned producing streams; a driver failure remains an error. Host
positions/liveness are not changed until the single `advance_live()` after the
last checkpoint. Thus `None` publishes no output, preserves all entry committed
prefixes and pages, and leaves provisional suffix writes to be overwritten on
retry. Unrelated single-sequence tree authorization remains intact. This is
cooperative cancellation, not kernel preemption or atomic coordination with an
external cancellation sender after the final checkpoint.

The ModelRunner facade distinguishes native unavailability from cancellation
and avoids host work. The actual continuous-batch loop calls one private round
implementation with its drain query. That round combines drain with closure
of any selected live response and forwards it into the existing graph-logit
path. `None` updates no salt/history/token budget, emits no row result and
returns to the next worker iteration without fallback or fault classification.
Normal retirement releases disconnected rows; connected peers retry from their
unchanged target prefixes. Successful prior admission in the same worker tick
is not undone: the transaction starts at decode-call entry.

Declared batch generation/decode checkpoints now report
`cooperative_boundaries`. The existing schema and all seven wire mode values
are retained. The envelope still says `kernel_preemption=false` and
`qualification="not_assessed"`; this declaration does not clear a release gate.

## Regression and scope

An entry-only native implementation failed the required-device f32 regression:
`path 0 must poll after entry`, one failure, 0.39s (compile 10.03s). After
checkpoints, that exact check passed (0.39s, compile 30.69s). An expanded test
initially referenced `BackendError` from `tritium_core`; compilation failed
with E0425/E0433 and was corrected to its owner, `tritium_spec`.

The serving round first retained ordinary graph dispatch without forwarding
the query. Two exact built-binary red runs failed at
`cancel_at=1, drain=false` (0.56s/0.40s). An earlier bare-name `--exact` Cargo
selection executed zero tests and is not evidence of this regression passing.
Changing forwarding alone made the round test pass (2.74s, compile 8.19s).
Native expanded f32 checks passed three tests in 0.75s before adding the final
guard checks. Exploratory outcomes are not substituted for the final lane.

Final checks sweep each observed native query point on all three routes,
dense/paged KV, empty/nonempty/distinct prefixes, dead/live rows and cold/warm
graphs. They compare ordinary/controlled outputs bitwise, preserve prefix/page
bytes and free counts, retain independent solo-tree authorization and recover
exactly against ordinary references. A hidden debug accessor distinguishes
actual captured logit/argmax graphs from eager substitution. Capacity, unmapped
pages, invalid lengths and even dead-row invalid tokens remain errors; dead
rows at capacity stay frozen. The facade covers cold capture, no host adoption,
pre-entry cancellation and native unavailability.

The serving round tests close an actual selected Tokio response or set drain
at entry, after device uploads, after replay and before batch commit. They
observe no output/history/budget/salt progress, unchanged peer bytes/pages,
zero early releases, exactly one disconnected-row retirement (or both on
drain), zero release failures and ordinary-greedy-equivalent live-peer retry.
These cross the same round function the worker loop calls, but explicitly
perform retirement in the fixture; they are not model-bound full HTTP or
destructive deployment qualification. The real batched-router capability
fixture additionally observes bounded worker startup and retirement.

## Final managed validation

Existing SSD cache: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`;
observed binary directory `/mnt/4tb/build/cargo/70/e33462ce184cce/debug/deps`.
`RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`,
`TRITIUM_REQUIRE_CUDA=1`. Unit
`tritium-lockstep-cancel-final-20261010-v1.service`, invocation
`d7f4f55c1a8a4f288d7347e2fe7a3b76`: terminal success at
2026-10-10 01:55:36 EDT (`SubState=exited`, `MainPID=0`,
`ExecMainStatus=0`, `Result=success`).

The primary unit's CUDA-library command omitted its opt-in `cuda` feature and
executed only ten runtime-free tests. That result is not physical coverage.
Separate required-device unit
`tritium-lockstep-cancel-cuda-library-20261010-v1.service`, invocation
`d0bb4e41b61f44258fc9411682a14e62`, runs the explicit CUDA-feature library
command below with f32 KV. This supplemented, rather than restarted, the
then-live final unit.

The supplement exited successfully at 2026-10-10 01:55:47 EDT with
`SubState=exited`, `MainPID=0`, `ExecMainStatus=0`, `Result=success`.
Device: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Quiescent final-source outcomes:

- both complete 18-test native runner suites (f32 5.65s, f16 5.61s);
- both complete 92-test CUDA-feature serving library suites (f32 50.90s,
  f16 51.93s);
- five additional four-test native and three-test serving lockstep filters per
  dtype: 70 extra executions, all successful;
- explicit CUDA-feature library: 155 passed, 12 ignored, 11.67s; ignored
  performance and capture-incompatibility probes are not passes;
- ten runtime-free CUDA-crate tests, 118 host serving tests (66 library, four
  binary, two CLI, 45 contract, one telemetry), 27 default serving tests;
- scoped CUDA/NN/serving all-target warnings-denied Clippy (59.25s), whole-
  workspace formatting and working-tree whitespace checks.

No failed test, panic or CUDA deinitialization error appeared in either final
invocation. Binary SHA-256 identities:

```text
runner_cancellation_cuda-f804ff2ad2409381
f74acc6c1dc7c67c023534e76883ea096ea143c68a4ba3d6c17d5d182dee2767
tritium_serve-8174e1b89dd2f7e4
f383e65b3eb4f8c9a66bd17cdcfe7019d9c23d0b23d9ba7fc36f2cf55ed96afe
```

```sh
for kv in f32 f16; do
  export TRITIUM_KV="$kv"
  cargo test --locked -p tritium-nn --features cuda --test runner_cancellation_cuda
  cargo test --locked -p tritium-serve --features cuda --lib
  # Five extra built-binary lockstep-filter runs per dtype for each suite.
done
cargo test --locked -p tritium-cuda --lib
# Separate required-device supplement:
cargo test --locked -p tritium-cuda --features cuda --lib
cargo test --locked -p tritium-serve --features serve
cargo test --locked -p tritium-serve --lib
cargo clippy --locked -p tritium-cuda -p tritium-nn -p tritium-serve \
  --features tritium-nn/cuda,tritium-serve/cuda --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

Commands were timeout-bounded in fail-fast managed units; terminal state,
counts, device and binary identities above bind the local outcome. Missing
required CUDA fails, rather than silently converting an absent device to pass.

## Remaining release obligations and custody

Candidate-bound model/HTTP cancellation latency, KV/resident-resource baselines,
concurrency/failure receipts and real-model overhead remain unqualified. Tiny
fixtures do not qualify Qwen, SOTA performance or the public release. Framework/
PTQ/refinement/distributed evidence, the full physical backend/browser matrix,
whole-model ONNX and source-free packaging, Stage-7/Qwen language/MTP quality/
runtime/reproduction, OCI/security/Kubernetes, audited zoo/docs/governance and
independent second-machine review/human activation remain required.

The note is retained project evidence. Existing SSD cache is reused, no new
persistent scratch directory is created, and managed units are retired after
their terminal proofs are recorded. Unrelated staged/WIP changes are preserved.
No old campaign scratch deletion, capture/fitting/cloud job or public activation
is authorized or performed by this milestone.
