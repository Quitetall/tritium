# CUDA owner stream isolation

Date: 2026-10-10. Decision: private research ADR 0052; serving plan 0052.
Base implementation tree: `5ba8a79a5584d08c0df597abb2d296f481dceb8c`.

## Failure and diagnosis

The previous serving milestone exposed cross-fixture CUDA graph failures under
independent model ownership. Its test-only fixture mutex remained explicitly
unqualified as a production fix. This investigation uses a channel-controlled
two-owner physical regression, not a timing sleep or disabled graph route.

One owner holds a THREAD_LOCAL capture open on a nonblocking stream. A second
backend then creates an auxiliary stream, as resident decode initialization
does. The capture thread always receives completion and ends capture before
the main thread checks peer errors. An eight-byte raw memset node proves exact
graph replay; its allocation and pointer lifetime cover capture and download.

Probe sequence, before the production constructor change:

1. Peer upload and stream synchronization alone passed (one test, 0.37s).
2. Adding peer auxiliary stream creation failed with
   `DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted
   when stream is capturing")` (one test, 0.29s).
3. Removing upload/synchronization preserved the failure (0.40s and a separate
   cached repetition, 0.30s). The failing call was peer stream creation.
4. Keeping a peer auxiliary stream alive before capture changed that same call
   to PASS; the cold case still failed in the same two-test process.
5. Replacing cold `new_stream()` with safe cudarc `fork()` also passed; the cold
   case still failed in the same three-test process (two passed, one failed).

An initial unqualified name combined with `--exact` executed zero tests and is
not a pass. The original build handle was observed terminal before the direct
binary reproducer; no build was restarted because an observation expired.

The ranked hypotheses were first-stream context synchronization, driver
prohibition of stream creation itself, and legacy/default-stream coupling.
The one-variable controls discriminate the first: cudarc 0.19.9 maintains a
stream counter per context wrapper, although the wrappers retain the same
device primary context. Its first `new_stream()` synchronizes that context;
`fork()` establishes tracking and event ordering without the context-wide wait.
The previous backend retained only the null legacy stream, so its wrapper was
still in this first-stream state when resident decode requested a stream.

NVIDIA's [CUDA Graphs restrictions](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#prohibited-and-unhandled-operations)
prohibit synchronizing a context containing an active capture. The controls and
the inspected cudarc source identify this specific Tritium call pattern; they
do not establish that all possible CUDA capture failures have this cause.

## Change and regression surface

`CudaBackend::new` now creates and retains an owned nonblocking working stream
using `ctx.default_stream().fork()` before backend-owned allocations. This is a
fresh wrapper with no Tritium allocations to migrate. The safe fork joins prior
legacy-stream work by event and establishes stream/event tracking before later
decoder or host-offload streams are created. No private CUDA context, global
cache, production mutex, unsafe stream wrapper, graph/capture-mode change,
fallback or mathematical change was introduced.

Current-context restoration and primary device identity remain unchanged.
External framework operations retain their supplied raw stream and existing
buffer/`record_stream` contract; the working stream does not replace it.

The final regression exercises peer construction both before and during the
owner's open capture, requires distinct non-null working stream handles and a
distinct auxiliary handle, and verifies exact owner graph replay plus peer
upload/download recovery. Both passed (0.57s) after the constructor change.
Temporary prewarm/fork controls were removed after their results were recorded.

A separate native test builds serial references for two different prefixes,
then runs two independently owned resident runners on separate threads. Bounded
channel rendezvous coordinate prefill, decode and tree phases for eight rounds;
a failed peer disconnects the wait rather than trapping a barrier. It requires
bitwise serial-equivalent decode logits, identical accepted tree tokens and KV
bytes/watermarks, real captured tree buckets and no host-cache adoption. There
is no fixture ownership mutex. The initial full native CUDA suite passed all
nine tests (2.31s), including graph/eager cancellation and dense/paged peer
isolation. The six temporary serving fixture guards were subsequently removed.

## Validation identity and commands

Physical device: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Shared existing cache: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`
(resolves to `/mnt/4tb/build/cargo/70/e33462ce184cce`), `RUSTC_WRAPPER=`,
`CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`.

Named red/green command:

```sh
TRITIUM_REQUIRE_CUDA=1 cargo test --locked -p tritium-cuda --features cuda \
  --lib cuda_backend_stream_creation -- --nocapture --test-threads=1
```

Final two-owner pair includes the separate
`cuda_backend_construction_during_peer_capture_preserves_replay` case; the
`cuda_backend_` filter executes both. Required-device execution prevents
unavailable hardware from becoming a silent pass for these new checks.

Managed validation unit: `tritium-cuda-owner-final-20261010-v1.service`,
invocation `967b506d46e44e6eb491acfb1951ed13`. The first dispatch used unsupported
`systemd-run --set-environment` and failed before creating a unit. Corrected
`--setenv` dispatch was confirmed live; the absent unit's default status was not
treated as test evidence. The corrected invocation finished at
2026-10-10 00:16:08 EDT with `SubState=exited`, `ExecMainStatus=0` and
`Result=success`.

Initial validation binary digests:

- CUDA lib-test `tritium_cuda-408bd00b0052f5c0`:
  `40430b933394d3fe999d83c99792e7048746983f4e34aa86ca3f9db44b566b37`.
- Native integration-test `runner_cancellation_cuda-f804ff2ad2409381`:
  `04b367c72c14b458ac4d4651fd99daf6b09e285ba0f658623c744bb575c6c550`.

The managed script runs the complete CUDA lib-test binary serially under a
180-second bound; both f32/f16 native suites with normal test concurrency; five
additional two-owner repetitions per KV dtype; both f32/f16 complete serving
CUDA-feature lib suites with normal concurrency and no fixture guards; the
serve-feature suite; warnings-denied CUDA/NN/serving all-target Clippy;
workspace formatting; and whitespace checks. Individual commands are bounded
inside one 900-second parent deadline. No existing live job was stopped.

Results of that completed invocation:

- CUDA-feature library: 155 passed, 12 ignored, 0 failed (34.25s). The ignored
  tests include performance probes and the known safe-launch capture limitation;
  they are not passes or a performance qualification.
- Native physical suite: nine passed with f32 KV (2.20s), nine with f16 KV
  (2.11s), plus five additional two-owner repetitions per dtype, all passed.
- Unguarded CUDA-feature serving library: 75 passed with f32 KV (58.81s) and
  75 with f16 KV (65.78s). The pre-existing model-backed drafter reconciliation
  test executed rather than skipping; its existing 217,871,680-byte local GGUF
  was present. This is not candidate-bound drafter qualification.
- Serve-feature software suite: 112 executed tests passed, including 43 router
  contracts. Zero-case feature-gated binaries/doc tests are not extra passes.
- `cargo clippy --locked -p tritium-cuda -p tritium-nn -p tritium-serve
  --features cuda --all-targets -- -D warnings`, `cargo fmt --all -- --check`
  and `git diff --check` all passed through the fail-fast managed script.

The final CUDA serving binary `tritium_serve-8174e1b89dd2f7e4` has SHA-256
`f924981846a4f91b643c5a13037d720f4875d6f4b673f6eb567d568c927c86cf`.
Only CUDA working-stream documentation comments were corrected after the
initial CUDA/native binaries had been built; constructor behavior and test
bodies were unchanged. The subsequent serving rebuild and Clippy used those
comments. A final source-quiescent refresh explicitly rebuilds the CUDA pair
and both native suites and repeats unguarded serving tree/speculative checks.

Refresh unit: `tritium-cuda-owner-refresh-20261010-v1.service`, invocation
`138d1271fa934dad86f8847cfaa8083b`. It runs `cargo test --locked -p tritium-cuda
--features cuda --lib cuda_backend_ -- --nocapture`, both f32/f16
`cargo test --locked -p tritium-nn --features cuda --test
runner_cancellation_cuda`, then five rounds per dtype of the exact serving
binary's `generator::tests::runner_tree_` and
`generator::tests::speculative_loop_` filters with normal test concurrency.
Formatting and whitespace checks finish the 600-second fail-fast script.
This refresh finished at 2026-10-10 00:17:43 EDT with `SubState=exited`,
`ExecMainStatus=0`, `Result=success`: the final CUDA pair, both nine-test native
suites and all 50 additional unguarded serving checks passed, along with
formatting and whitespace checks. The three-test tree and two-test speculative
filters retain normal concurrent execution within each process.

Refreshed binary digests:

- CUDA lib-test: `f1666dc3481df079db50e8a4fcf44d4fb97a9611c2cf22052182832680a0f5f4`.
- Native integration-test: `ea19beaefa6265be0f1f3acc60fc2f1d1e64c05165073039244702960244a230`.
- CUDA serving lib-test: unchanged `f924981846a4f91b643c5a13037d720f4875d6f4b673f6eb567d568c927c86cf`.

## Scope and remaining obligations

Tiny physical concurrency and recovery are not candidate-bound real-model
performance/resource, quality, SOTA or independent release qualification.
Suite time is not a decode-speed or cancellation-latency measurement. Foreign
code can still invalidate a capture through prohibited context-wide operations;
models still share device capacity and fatal device/context state.
Cold framework initialization inside a foreign blocking/global capture is not
qualified by these Tritium-owned nonblocking THREAD_LOCAL checks; PyTorch/HF
capture interoperability still needs its own physical matrix.

Atomic grouped multi-slot cancellation, in-operation drafter/reconcile/
enrollment queries and capability reporting remain open. Candidate-bound
worker/KV/resource/failure-matrix, CPU/CUDA OCI/security/Kubernetes,
PyTorch/HF/PTQ/refinement, backend/browser qualification, audited Qwen/model zoo
and independent public-release gates remain open.

No new scratch directory, target directory, model payload, cloud compute or
campaign fitting/capture was created. Existing build cache was reused; no
August campaign/scratch deletion was authorized or performed. Unrelated
worktree edits and staging remain intact.
