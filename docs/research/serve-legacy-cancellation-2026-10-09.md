# Legacy/native-resident cancellation rollout — 2026-10-09

Source baseline: `e440c3371041dd5340e5e056c98fd2efe8fd40f2`.
Contract: ADR 0033 / plan 0052 and private ADR 0051's legacy extension.
Status: local software and tiny physical CUDA checks passed. Full serving
rollout and empirical release qualification remain open.

## Reproduction and causal probes

An entry-only implementation of the new native cancellation interface was
used as the minimal probe before internal checkpoints were implemented:

```sh
timeout 900 cargo test --locked -p tritium-nn --test training_dense_weights \
  legacy_cancellable_forward_stops_inside_prefill -- --exact --nocapture
```

Managed repro invocation `ddfa15b561564e738288d3df34ed2849` failed at 21:45:26
with `native prefill must poll after model entry` (0.00 seconds). A bounded
cached repeat also failed, exit 101. Ranked hypotheses were absence of an
internal checkpoint, cancellation/error conflation, and KV rollback failure.
The observed `Ok(Some(logits))` isolates the first, before rollback assertions.
Adding the post-embedding checkpoint made the same test pass in 0.00 seconds,
probe invocation `3c0db673ff674447850ae9b170d9d5ff` at 21:46:55.

The serving adapter needed a separate causal test after native checkpoints
existed. Its first managed invocation `886072068a624e43ac3ad2ad66449301`
used an unqualified filter with `--exact` and executed **zero tests**; that
result is not a pass. The corrected bounded cached command was:

```sh
timeout 120 cargo test --locked -p tritium-serve --features serve --lib \
  generator::tests::legacy_runner_generator_cancels_native_prefill_and_recovers \
  -- --exact --nocapture
```

It failed with `legacy adapter must cancel during native prefill`, actual two
delivered tokens versus expected zero, in 0.00 seconds (exit 101). The adapter
was still using the compatibility default, so its native prefill never received
the cancellation query. The implementation now passes it through explicitly.

## Implementation

`ModelRunner::forward_cancellable` returns `None` on cancellation, preserving
committed host KV prefixes through the existing per-layer rollback. Normal
forward and fidelity-dump paths share the same numerical operations with a
never-cancelled query. Host checkpoints surround blocks and the language head.

Native `CudaDecodeModel::prefill_cancellable` polls between embedding,
attention/MLP launch groups and head completion. On cancellation it waits for
submitted stream work before returning without advancing the cache watermark;
driver errors during that wait remain errors. M=1 graph replay is not killed:
the facade polls before/after replay and rewinds its watermark before returning
no output. A private typed resident outcome separates unavailable hardware
dispatch from cancellation; cancellation cannot trigger host fallback.

`RunnerGenerator` uses controlled prefill/plain decode, resets request-owned
runner/draft state on cancellation, and preserves speculative dispatch with
callback-level checks. It does not claim interruption inside a speculative
chain, tree operation or continuous-batch orchestration. Queries must be cheap,
nonblocking and non-panicking; cancellation after the final checkpoint can race
with completion. No bound on kernel or native cancellation latency is inferred.

## Completed checks and run identities

Host validation invocation `3c8b7733b3f744b29b52fcaf2f53559a` passed at
21:48:06:

```sh
timeout 900 cargo test --locked -p tritium-nn --test training_dense_weights
timeout 900 cargo test --locked -p tritium-nn --test qwen35_text_runner
```

All six host tests and 14 strict-Qwen tests passed. New host checks exercise
tied/untied heads, empty/nonempty prefixes, every observed checkpoint, bitwise
KV/logit recovery, and distinct runtime errors.

The serving suite passed 109 executed tests at 21:51:36: 59 library, four
binary, two CLI, 43 contract and one OpenTelemetry test. This includes the real
legacy CPU adapter's cancellation and recovery case. The managed verification
unit `tritium-legacy-cancellation-validation-20261009.service`, invocation
`0d4e92e2c1474716838f1150fb26ea64`, executed:

```sh
timeout 900 cargo clippy --locked -p tritium-nn -p tritium-serve \
  --features tritium-serve/serve --all-targets -- -D warnings
timeout 900 cargo check --locked -p tritium-serve --features cuda --all-targets
timeout 120 cargo fmt --all --check
```

The tiny in-memory CUDA test completed under
`tritium-legacy-cancellation-cuda-20261009.service`, invocation
`3bee269849644c968c96da316d4262c2`:

```sh
TRITIUM_REQUIRE_CUDA=1 timeout 1800 cargo test --locked -p tritium-nn \
  --features cuda --test runner_cancellation_cuda -- --nocapture --test-threads=1
```

It requires a CUDA device instead of admitting an unavailable-device skip.
The fixture has two ternary ReLU2 blocks, 64 hidden/FFN channels, GQA 2:1,
eight vocabulary entries and a 16-row cache. It tests native batched-prefill
checkpoint recovery and post-graph watermark rewind, not a real language model.
Preflight observed RTX 4090 `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`,
24,564 MiB total and 2,097 MiB in use. No model-serving compute process was
observed. Both physical tests passed at 21:52:59 in 2.26 seconds; the named
tests executed with required-device semantics, not unavailable-device skips.
This is local tiny-fixture correctness, not a candidate-bound GPU performance,
latency, resource or language-model qualification receipt.

The final combined repetition command completed 20 host-trio repetitions and
20 real legacy-adapter repetitions, then reached its 300-second outer timeout
(exit 124) before reporting its CUDA-pair or formatting verdict. Those missing
verdicts are not passes. After that command was terminal, a separate visible
required-device rerun with `TRITIUM_KV=f32` completed by 22:04 EDT: 69 seconds
of compilation followed by both CUDA tests passing in 1.12 seconds. This does
not establish the exact cause of the earlier combined timeout.

After the visible rerun, a separately completed foreground loop ran the CUDA
pair five times with `TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f32`; all ten test
executions passed (0.57–0.69 seconds per pair). The same command finished
`cargo fmt --all --check` and exited zero. Existing binaries/caches were reused:

```sh
timeout 180 env TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f32 bash -c '
  set -e
  for i in {1..5}; do
    timeout 30 cargo test --quiet --locked -p tritium-nn --features cuda \
      --test runner_cancellation_cuda -- --test-threads=1
  done
  timeout 120 cargo fmt --all --check
'
```

All commands used `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`,
`RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2` and `TMPDIR=/mnt/4tb/tmp`. The repeated
physical result is explicitly f32-KV coverage; it is not evidence for other KV
profiles or large-prompt IMMA.

## Remaining gates and cleanup

Scoped warnings-denied Clippy passed at 21:55:01. CUDA serving compilation
passed at 21:55:35; the combined managed unit completed at 21:55:37 after its
formatting check. These are local developer-worktree checks, not hosted CI
or an independent candidate qualification.
Continuous-batch/chunk/tree/speculative in-operation checkpoints,
capability reporting, finer host-block cancellation, native cancellation
latency, candidate-bound KV/resource receipts and the full serving failure
matrix remain open. Software tests and tiny physical fixtures cannot qualify
27B quality, SOTA, candidate deployment or public release.
The short native CUDA prompt exercises the small-M path; large-prompt IMMA,
other KV profiles and faulted-driver synchronization are not qualified here.

Existing SSD caches were reused; no per-run scratch directory or large model
artifact was created. The native backend warmed tuning entries in its existing
shared cache; shared backend caches are not removed while other work may use
them. Source, tiny fixtures and this evidence note are retained.
August campaign data and unrelated staged/optimizer work remain untouched.
