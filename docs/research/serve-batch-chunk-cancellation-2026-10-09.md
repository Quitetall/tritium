# Continuous-batch prompt cancellation — 2026-10-09

Source baseline: `77b5e657b4c717c5bad6e578e1e51ab817f2a383`.
Contract: ADR 0033 / plan 0052 and private ADR 0051 decision 7.
Status: native call-site, tiny physical retirement and scoped local validation
checks passed. Candidate-bound qualification remains open.

## Causal reproduction

The batch scheduler checked receiver closure between chunks, but called
ordinary native `forward` inside each chunk. A private `Pending::forward_chunk`
refactor initially retained that ordinary dispatch. A CPU backend delegates
real ternary projections and closes the receiver or sets drain from within the
first projection. No sleep, fabricated cancellation result, public fault hook
or model/backend substitution is used.

The exact regression command was:

```sh
timeout 180 cargo test --locked -p tritium-serve --features cuda --lib \
  batch::tests::pending_chunk_cancels_inside_native_forward -- --exact --nocapture
```

Its initial setup attempts failed before running the test: the dev-dependency
needed a lockfile edge, and the test used nonexistent KV accessor methods.
An offline Cargo run updated only the `tritium-serve -> tritium-core` dev edge;
the test was corrected to use the existing KV fields. These setup failures are
not cancellation reproductions.

The corrected command failed with `pending chunk must cancel inside native
forward (goal=0, drain=false, prefix=0)` in 0.00 seconds after compilation.
Running the same built binary directly reproduced the failure in 0.02 seconds
(both exit 101). Ranked hypotheses were missing query forwarding, receiver-close
visibility and native rollback failure. Changing **only** dispatch to
`forward_cancellable`, queried by drain or client closure, made the same test
pass in 0.07 seconds. It exercises 12 cases: ordinary/speculative admission and
tree-session open, disconnect/drain, and empty/nonempty committed prefixes.
The controlled native runner is executed on CPU; compiling with the `cuda`
feature does not turn those cases into physical GPU evidence.

## Scheduler integration

- A cancelled chunk publishes neither progress nor adopted rows or tokens.
- A private retirement operation consumes `Option<Pending>` once, resets its
  single-sequence staging KV and releases only its reserved batch row.
- Already-disconnected and draining pending requests share that retirement
  path. Drain preserves the existing error classification and returns to the
  loop's drain handling before another peer decode.
- Native runtime errors retain the existing error path. Normal chunk forwards
  retain positions, numerical operations and backend selection.
- Speculative and tree **prompt admission** use the same controlled chunk path;
  in-operation tree verification/speculative decode is not covered by this slice.

## Local checks

With source quiescent, this required-device command passed four named tests in
0.93 seconds after 79 seconds of compilation:

```sh
timeout 300 env TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f32 \
  cargo test --locked -p tritium-serve --features cuda --lib batch::tests:: \
  -- --nocapture --test-threads=1
```

Three checks use real native CPU forwards: in-operation cancellation/bitwise
rollback/recovery, normal-output/error preservation, and entry cancellation
before any projection. The fourth uses a real tiny two-layer ternary CUDA model
and paged batch KV. Retirement returns one reserved page, observes two
reservations/one release/zero release failures, preserves a live peer's raw K/V
bytes and position, and is idempotent even after newer staging work exists.
The same compiled four-test group passed ten separately completed bounded
repetitions with required-device and explicit f32-KV semantics.

Physical probe: RTX 4090 `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver
`615.71.09`. The CUDA model has two ReLU2 blocks, 64 hidden/FFN channels,
GQA 2:1, vocabulary eight and context sixteen. It is synthetic fixture data,
not a real language-model artifact. No missing-device skip was admitted.

Broader validation uses managed unit
`tritium-batch-chunk-validation-20261009.service`, invocation
`c0a3e1ec82434048a26dd74227e672eb`, with sequential commands:

```sh
timeout 900 cargo test --locked -p tritium-serve --features serve
timeout 900 cargo clippy --locked -p tritium-serve --features cuda \
  --all-targets -- -D warnings
timeout 300 cargo test --locked -p tritium-serve --lib \
  generator::tests::default_cancellable_generator_
timeout 120 cargo fmt --all --check
```

The serve suite passed 109 executed tests at 22:23:15 EDT (59 library, four
binary, two CLI, 43 contract and one OpenTelemetry). Scoped CUDA Clippy passed
at 22:24:42; both named runtime-free default-feature generator checks passed
at 22:24:46. The managed unit completed at 22:24:48 after formatting. A separate
foreground `cargo fmt --all --check` also exited zero. Disabled, zero-test
CUDA/model lanes are not qualification evidence. All Cargo commands reuse the SSD target cache with
`CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`, `RUSTC_WRAPPER=`,
`CARGO_BUILD_JOBS=2`, and `TMPDIR=/mnt/4tb/tmp`.

## Evidence limits and cleanup

This proves the native call-site behavior and the retirement helper's tiny
physical page/KV isolation. It is not a real-worker model/candidate-bound
failure-matrix receipt, a native cancellation-latency/resource high-water
qualification, a large-prompt/KV-profile qualification, or release approval.
Tree verification, speculative decode/drafter enrollment checkpoints,
capability reporting and the complete serving/deployment gates remain open.

Source/tests and this note are retained. Existing build and backend tuning
caches were reused; no per-run scratch directory or large model artifact was
created. Historical August scratch/campaign data and unrelated work are untouched.
