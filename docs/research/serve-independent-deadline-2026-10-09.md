# Serving deadline independent of SSE polling

Date: 2026-10-09
Source baseline: `c6bd75e9b7b8d70fc590fec7600b68b4081bb54a`
Contract: ADR 0033 / plan 0052, bounded streaming lifetime from queue admission.
Evidence class: local developer software checks, not release qualification.

## Confirmed defect

A one-second stream deadline did not stop cooperative decode when the caller
retained the response body without polling it. The deterministic router/worker
regression generates at most 60 events at 50-ms intervals, below the existing
64-event channel capacity, so ordinary full-buffer cancellation cannot create a
false pass. It observes real generator entry, waits for callback cancellation,
requires one timeout counter, then consumes the typed timeout and `[DONE]` and
checks that a new buffered request succeeds.

Before the fix, the unpolled case failed after 2.00 seconds (exit 101). The
cached executable repeated that failure in 2.01 seconds. Reproduction invocation:
`945f735e145e4feda187f4c0d9d68cf2`. The identical helper with body consumption
enabled passed in 1.07 seconds, isolating body polling as the changed variable
(probe invocation `5d0a7db505514484a5a30a344290f5b0`).

## Implementation

A private `StreamEventReceiver` retains the same 64-event channel behind a
per-request async receiver lock. A weak-reference deadline watchdog closes that
receiver at expiry, independently of HTTP-body polling. The lock is not retained
across SSE yields. No relay queue, token copy or second token buffer is added.
The deadline is derived from queue-admission time; overflowing budgets expire
immediately. An explicit expired check prevents buffered events from bypassing
an already-elapsed `timeout_at` when body consumption resumes.

Completion and drop retire the watchdog; receiver drop closes delivery.
Expiry is counted once across the watchdog and SSE path. The disconnect guard
is created before the lazy body starts, so a never-polled body drop is observable.
Both outcome owners atomically claim state, preventing one request from being
counted as both a deadline timeout and a disconnect.

That final interleaving has its own production-function regression. Before
the atomic disconnect claim, disconnect-first followed by expiry failed with:
`disconnect must retire the deadline before it can claim a second outcome`.
Managed invocation `5c707dba4140489db3bcd7a583a69593` and a cached repeat both
exited 101, with test execution taking 0.00 seconds. The regression also checks
the reverse ordering. No public generator interface, model kernel, wire error
code, backend policy, token-buffer capacity or release threshold changes.

## Verification

Initial validation (`806ac6f92c2041fb95b06fc11d032608`) passed the three deadline
contract tests, the then-current 86-test serving package suite and scoped Clippy.
Lifecycle validation (`77077f1dff0c4e6081398d816ca96a84`) then passed 90 executed
package tests, scoped warnings-denied Clippy and workspace formatting. These
earlier results preceded the final atomic outcome claim and are not substituted
for validation of that change.

Final validation is tracked by `2484e070c5104767a59dcc7faaf9ab33`:

```sh
cargo test --locked -p tritium-serve --features serve
cargo clippy --locked -p tritium-serve --features serve --all-targets -- -D warnings
cargo fmt --all --check
```

Commands have explicit timeouts and reuse the shared SSD cache with
`RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2` and `TMPDIR=/mnt/4tb/tmp`.
The final package test command passed at 20:52:42 EDT: 49 library, 4 binary,
2 CLI, 35 HTTP contract and 1 OpenTelemetry tests, 91 executed tests in total.
The atomic outcome regression subsequently passed 100 consecutive cached
repeats; the polled and unpolled deadline/recovery variants each passed 10
consecutive cached repeats. Final warnings-denied Clippy completed at 20:55:19
EDT; formatting also passed. The managed job ended with terminal success and
exit 0 at 20:55:22 EDT. Implementation and regressions are saved in commit
`83cac86a`; these are dirty-worktree developer results, not clean candidate
qualification receipts. Unrelated source/document edits were preserved.
CUDA batch/speculative and real-model e2e feature lanes are not exercised by
these `serve`-feature software checks, and their zero-case targets do not count
as qualification.

## Remaining release obligations

Closing delivery cancels queued jobs and cooperative token callbacks. It does
not interrupt an in-flight kernel or establish bounded cancellation latency
during strict Qwen's full native prefill. Native cancellation interfaces and
model-bound latency/resource receipts remain required. Watchdog overhead is not
benchmarked here. Candidate-bound OCI/security/Kubernetes execution, physical
backend/browser coverage, fresh-package/ONNX/Colab gates, flagship language/MTP
quality and reproduction, audited zoo and independent release sign-off remain
open. No model fitting, paid compute, registry publication, public activation,
historical scratch deletion or unrelated process termination occurred.
