# Cooperative native Qwen cancellation — 2026-10-09

Source baseline: `5b116626728a63af21668e2e4c8e693b0dc3ab37`.
Contract: ADR 0033 / plan 0052 and ADR 0051 (private planning repository).
Status: local software validation passed for the first strict-runner rollout.
Remaining adapter rollout and empirical release qualification remain open.

## Red-capable feedback loop

```sh
timeout 900 cargo test --locked -p tritium-serve --features serve --lib \
  active_prefill_ -- --nocapture
```

A synchronized cooperative generator pauses its first request inside prefill.
The test closes the receiving channel or sets drain **before** releasing that
operation, then requires the next checkpoint to skip subsequent work. A later
request must recover without latching a backend fault. These tests exercise
the actual worker dispatch, not native model latency.

Repro unit `tritium-serve-active-prefill-repro-20261009.service`, invocation
`3b9168f9c8544159821361cc84b062ba`, exited 101 at 21:27:36 local time: both
tests failed in 0.00 seconds. Disconnect reported:

```text
prefill cancellation must skip the next native-operation checkpoint
left: 1
right: 0
```

Drain received a token instead of the expected terminal stop. A cached repeat
reproduced both failures. Ranked hypotheses were legacy dispatch bypassing the
query, closure/drain visibility, and incorrect barrier ordering. The test's
causal ordering rules out cancellation being sent after operation release.

Changing only worker dispatch to the cancellable interface made both tests
pass in 0.00 seconds. Probe unit
`tritium-serve-active-prefill-dispatch-probe-20261009.service`, invocation
`1a8c2370d3c248788fe60b213c14074a`, completed at 21:29:19. This isolates the
dispatch defect; native checkpoints were implemented afterwards.

## Implementation and guarantees

`Generator::generate_cancellable` accepts a borrowed, runtime-free cancellation
query. Its compatibility default avoids pre-cancelled work and suppresses
token delivery, but cannot interrupt a legacy prefill implementation. The
single-request worker queries receiver closure or drain; the existing stream
deadline watchdog closes the receiver even when the response is not polled.

The strict Qwen generator overrides the interface and uses
`Qwen35TextRunner::forward_cancellable` for prefill and decode. Checkpoints
cover entry, embedding completion, mixer/MLP boundaries, language-head entry
and pre-commit completion. Cancellation reuses the existing transaction:
staged DeltaNet state is discarded and provisional full-attention KV cursors
are rolled back. It returns no output, keeps the committed cache reusable and
does not turn cancellation into a backend failure. Ordinary forward/capture
paths retain their existing observers and numerical operations.

Queries must be cheap, nonblocking and non-panicking. Callback count is not a
contract. Already-running operations cannot be preempted, and cancellation
after the final checkpoint can race with commit. This is not a promise of
bounded native cancellation latency.

## Local validation

Managed unit `tritium-cooperative-prefill-validation-20261009.service`,
invocation `dedd159abf8e469c88ab303a5a768f52`, uses the existing SSD cache,
`RUSTC_WRAPPER=`, two build jobs and `/mnt/4tb/tmp` as TMPDIR:

```sh
timeout 900 cargo test --locked -p tritium-nn --test qwen35_text_runner
timeout 900 cargo test --locked -p tritium-serve --features serve
timeout 900 cargo clippy --locked -p tritium-nn -p tritium-serve \
  --features tritium-serve/serve --all-targets -- -D warnings
timeout 120 cargo test --locked -p tritium-serve --lib default_cancellable_generator_
timeout 120 cargo fmt --all --check
```

The first command passed all 14 tests at 21:33:54, in 0.06 seconds. The new
runner cases cancel at every checkpoint observed in an uncancelled forward,
covering empty/nonempty hybrid DeltaNet and full-attention caches, dense and
HostSALT QKV fixtures. They compare hidden states, logits and committed state
values by float bits, and prove recovery after cancellation. Empty-token and
out-of-vocabulary runtime errors remain distinguishable from cancellation.
The bitwise assertions were saved while the first build was already in
progress, so that initial verdict alone is not attributed to the final
assertions. A final-source rerun in managed unit
`tritium-cooperative-prefill-final-source-20261009.service`, invocation
`a5ee8194ae974c8fadac3808b6d902b9`, passed all 14 tests at 21:39:19 in 0.22
seconds after the final source was quiescent.

The complete serving suite passed 108 executed tests: 58 library, four binary,
two CLI, 43 contract and one OpenTelemetry test. Scoped warnings-denied Clippy
finished successfully at 21:39:14. Both default-feature generator seam tests
passed at 21:39:25. Workspace formatting and `git diff --check` also passed.
Disabled CUDA/batch/spec/e2e and zero-case doc lanes are not empirical evidence.

The repetition unit's journal retains only 13 paired verdicts, so it is not
used to claim all twenty. A separate bounded foreground loop then completed
successfully with explicit terminal output:

```text
PASS: 20 worker-pair repetitions and 20 native-trio repetitions
```

Each iteration runs the two `active_prefill_` worker checks and the three
`cancellable_forward_` native checks through Cargo, stops on any nonzero exit,
and bounds each command to 20 seconds. The outer command is bounded to 180
seconds and concludes with a separately executed workspace formatting check.
These are developer-worktree checks; commit-tree hooks and hosted CI are
separate gates, and no independent empirical release receipt is created.

## Remaining rollout and qualification

- Extend declared cancellation/reclamation behavior to legacy ModelRunner and
  device-resident adapters, batched/chunked prefill and tree work.
- Add finer operation/row/kernel checkpoints where physical measurements show
  an individual operation exceeds the cancellation budget.
- Complete capability reporting, candidate-bound KV/resource receipts, native
  CPU/CUDA cancellation latency and the complete serving failure matrix.
- Run exact-candidate image/cluster qualification and independent release gates.

The worker fixtures and tiny CPU native tests do not qualify a 27B model,
GPU behavior, resource high-water marks, model quality, SOTA or public release.
No per-run scratch directory was created; existing build caches were reused.
Source, this evidence note and the private ADR are intentionally retained.
Historical August campaign data and unrelated staged/optimizer work remain
untouched.
