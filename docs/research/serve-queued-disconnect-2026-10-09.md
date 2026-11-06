# Serving queued-disconnect regression

Date: 2026-10-09
Source baseline: `d2532e0130d5e5cc7b66f4fb04c57638bfabf5cf`
Contract: ADR 0033 / plan 0052, request cancellation before expensive prefill.
Evidence class: local developer regression, not candidate qualification.

## Reproduction and cause

The single-request worker dequeued a generation job and entered the generator
even when its SSE receiver was already closed. Cancellation was noticed only
when the generator produced its first token. For the strict Qwen generator,
that is after allocating its request cache and forwarding the complete prompt.

The deterministic regression closes the receiver before enqueueing, submits the
cancelled job followed by a live job, observes the live job's token and terminal
event, and records actual `Generator::generate` entries. Before the fix:

```text
a queued disconnect must not enter the generator or prefill
left: [1, 2]
right: [2]
FAILED. 0 passed; 1 failed; 43 filtered out; finished in 0.00s
```

The original managed reproduction and direct cached-binary repeats failed with
exit 101 and the same assertion. A separate probe asserted that the sender
already observed receiver closure; that assertion passed while the model-entry
assertion still failed. Probe invocation:
`48fcf5e12fec43da97360f6d0a3455c9`, completed 20:26:05 EDT.
This falsifies delayed closure visibility as the cause. The recorder runs only
inside the generator, not at queue admission.

The fix checks `tx.is_closed()` before phase publication, telemetry for model
execution, or generator entry. It does not change a public API, wire format,
readiness identity, backend policy, frozen gate or numeric kernel.

## Checks

Managed validation invocation: `b6302ab6395d487eab976eaa8c7c82a4`.
Environment: shared SSD target, `RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`,
`TMPDIR=/mnt/4tb/tmp`.

```sh
cargo test --locked -p tritium-serve --features serve --lib \
  worker::tests::queued_disconnect_skips_generator_and_next_request_recovers \
  -- --exact --nocapture
cargo test --locked -p tritium-serve --features serve --test contract \
  queued_sse_disconnect_skips_prefill_and_recovers -- --exact --nocapture
cargo test --locked -p tritium-serve --features serve --lib
cargo test --locked -p tritium-serve --features serve --test contract
cargo clippy --locked -p tritium-serve --features serve --all-targets -- -D warnings
cargo fmt --all --check
```

Every managed command has an explicit timeout. Results so far:

- Worker regression: 1 passed, 0 failed, 0 ignored, 0.00 seconds, 20:31:58 EDT.
- HTTP/SSE regression: 1 passed, 0 failed, 0 ignored, 0.00 seconds, 20:32:16 EDT.
- Full library suite: 44 passed, 0 failed, 0 ignored, 0.36 seconds.
- Full HTTP contract suite: 33 passed, 0 failed, 0 ignored, 1.22 seconds.
- Workspace formatting: passed.
- Both new tests: 20 consecutive cached-executable repeats each, all passed.
- Scoped Clippy with `--all-targets -- -D warnings`: passed, completed
  20:35:18 EDT; the managed validation ended successfully with exit 0.

The complete `cargo test --locked -p tritium-serve --features serve` package
check also passed at 20:36:42 EDT (managed invocation
`0fd2956808c74e688a05f8a521dd4811`, terminal success/exit 0): 44 library,
4 binary, 2 CLI, 33 HTTP contract and 1 OpenTelemetry tests, 84 executed tests
in total. CUDA batch/speculative and real-model e2e feature lanes were disabled;
their zero-case targets and zero doc-tests are not hardware/model evidence.
Implementation and tests were saved in commit `3b19cc72`; this follow-up records
the checks that finished after that commit. The working tree also contained
unrelated edits, so these results remain developer checks, not clean candidate
qualification receipts.

The HTTP test holds the first request in a controlled generator, observes a
second request in the real router/worker queue, polls its SSE role frame, closes
its body before releasing the first request, and requires a third request to
complete. Its entry count must be two, not three, and queue depth must settle to
zero. Controlled generators make these software regression checks; they are
not physical GPU or real-model qualification.

## Packaging check boundary

A separate bounded developer check of the deployment, bundle, crate and npm
SBOM unit modules exited 124 after 120 seconds, without a complete suite
result. Verbose output reached bundle atomic publication after earlier cases
completed. Those partial cases are not an aggregate pass. No fsync or atomic
publication requirement was removed. Shared SSD I/O pressure remained high.
The test process was gone; its sole leftover fixture was a user-owned 55-byte
`bundle.cdx.json` containing only `bomFormat: CycloneDX` and `specVersion: 1.6`
under `/mnt/4tb/tmp/tmp0j9pmz22`. After checking the fixture identity and visible
file-descriptor/working-directory references, that newly created directory was
removed and its absence verified. No historical scratch was deleted.

## Still required

This prevents *already disconnected queued* jobs from starting model work.
It does not establish cancellation during a currently executing native prefill,
interrupt an in-flight device kernel, prove deadlines independently of SSE-body
polling, or seal the full resource/latency matrix. The current public generator
callback observes cancellation only at token boundaries; strict Qwen prefill
still runs before that callback. Any public cancellation-interface change needs
its own ADR and compatibility design.

Current candidate-bound OCI runtime/security and CPU/CUDA Kubernetes receipts,
the flagship language-plus-MTP artifact/quality/runtime/reproduction gates,
missing physical backend/browser lanes, fresh-environment package/ONNX/Colab
checks, audited zoo and independent release sign-off remain separate obligations.
No new model fitting, paid compute, publication, public activation, old-scratch
deletion or unrelated process termination occurred in this slice.
