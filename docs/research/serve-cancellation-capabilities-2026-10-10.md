# Cancellation capability diagnostics

Date: 2026-10-10. Decision: private ADR 0051, decision 12; serving plan 0052.
Base tree: `13c428325696922abdd0c6a409506e09e2edc1e5`.

## Implementation contract

The runtime-free Generator seam declares `CancellationCapabilitiesV1` using
seven frozen `CancellationCheckpoint` modes. Defaults are conservative:
entry/token delivery for generation, entry-only optional tree wrappers, unknown
hidden model drafting, and no configured continuous-batch routes. A default
method does not turn a third-party adapter into a natively cancellable one.
Known Mock, Runner and strict Qwen adapters declare their own configured routes.

Single-request Runner generation is cooperative on host and resident paths.
Optional native tree/drafter declarations are conditional; metadata only reads
the concrete backend type and configuration, not resident availability. CPU
Runner does not advertise a CUDA-only tree/drafter route. Strict Qwen generation
declares cooperative model boundaries, with no tree or model-draft route. An
artifact containing MTP weights does not enable MTP serving by itself.

Batch construction declares prompt/speculative work separately from ordinary
lockstep decode. Prompt work is cooperative; configured speculative work is
conditional. Ordinary lockstep decode currently reports `worker_iteration`,
not in-operation cooperative rollback. Metadata must not obscure this remaining
software gap or substitute a different dispatch path to improve its label.

Each router snapshots the declaration before worker handoff. `/healthz` adds
`cancellation` in healthy and all unhealthy diagnostic responses:

```json
{
  "schema": "tritium.cancellation-capabilities.v1",
  "checkpoints": {
    "generation": "entry_and_token_delivery",
    "tree_session_open": "entry_only",
    "tree_verify": "entry_only",
    "model_draft": "unknown",
    "batch_prompt": "not_enabled",
    "batch_speculation": "not_enabled",
    "batch_decode": "not_enabled"
  },
  "kernel_preemption": false,
  "qualification": "not_assessed"
}
```

This example is the legacy default, not a native route. The envelope always
fixes preemption to false and qualification to not assessed. It is descriptive
configuration, not evidence of operation availability, selected per-request
dispatch, hardware health, rollback, latency or readiness. Existing status,
auth/admission, error and `/readyz` behavior is unchanged. Third-party
declarations are not trusted qualification evidence. Metadata methods must be
cheap, side-effect-free, nonblocking and non-panicking.

## Checks and exploratory findings

Tests pin all seven serialized modes and the exact legacy checkpoint map;
prove default reporting does not invoke generation/context/vocabulary methods;
check CPU/native/attached-drafter declarations; verify snapshot-once behavior
through legacy and governed routers; and prove a cooperative declaration cannot
make a draining router ready. Healthy and each unhealthy health response retains
the fixed unqualified/non-preemptive envelope. Unhealthy-state rendering checks
use injected software flags, not destructive device-failure qualification.

A tiny real batched-worker constructor check waits for `/readyz=200`, checks the
actual HTTP snapshot with and without a drafter, drops the router, and waits for
the worker's existing liveness guard to confirm model retirement. The private
constructor returns that lifecycle signal to tests; public builder return types
are unchanged. No global hook, alternate decode route, readiness override or
new polling thread is introduced.

One exploratory CUDA compile failed because `TernaryBackend` uses
`as_concrete()`, not `as_any()`; the source was corrected to the existing
side-effect-free concrete hook. A later six-test exploratory run reported pass
but produced `CUDA_ERROR_DEINITIALIZED` while its unobserved worker was still
initializing at process teardown. That run is not accepted as clean physical
evidence. The fixture was corrected to observe bounded startup and retirement;
the subsequent six-test f32 run completed cleanly (0.71s, compile 44.21s).

## Final validation

Existing SSD cache: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`;
observed binaries reside in `/mnt/4tb/build/cargo/70/e33462ce184cce/debug/deps`.
Environment: `RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`,
and `TRITIUM_REQUIRE_CUDA=1` for CUDA lanes. Missing hardware must fail rather
than silently qualify a fixture.

Managed job: `tritium-cancel-capabilities-final-20261010-v1.service`, invocation
`d81c78b8e16a46de8dd55a3b2d9248a2`. Terminal success at
2026-10-10 01:37:52 EDT: `SubState=exited`, `MainPID=0`, `ExecMainStatus=0`,
`Result=success`. The unit retained this state via `RemainAfterExit=yes`;
a running unit's default result was not used as terminal evidence.

Device: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
The quiescent final source passed:

- 90 CUDA-feature library tests with f32 KV (55.72s), and the same 90 with
  f16 KV (55.34s), plus both two-test HTTP contract filters;
- five additional six-test CUDA capability filters per dtype: 60 extra
  executions, all successful with observed batch startup/retirement;
- the complete serve-feature software lane: 66 library, four binary, two
  CLI, 45 contract and one telemetry tests (118 executed, no failures);
- 27 runtime-free default-feature library tests and five additional two-test
  host HTTP contract filters (ten extra executions);
- both scoped all-target warnings-denied Clippy lanes (CUDA 8.95s, serve
  1.83s), whole-workspace format check and working-tree whitespace check.

No `CUDA_ERROR_DEINITIALIZED`, panic or failed test appeared in the final
invocation. These are local developer-worktree checks, not independent public
qualification. Final built binary SHA-256 identities:

```text
tritium_serve-8174e1b89dd2f7e4
9b6f0ecb6572dd365f59df5a93eab8e3d92793d4ebd962a2745ff689b38a23c9
contract-7fa46ccbb1aa35d2
5859c08a194bf6259740796d8b58748436890445cf7e0704b893326d1ee7f513
```

```sh
for kv in f32 f16; do
  export TRITIUM_KV="$kv"
  cargo test --locked -p tritium-serve --features cuda --lib
  cargo test --locked -p tritium-serve --features cuda \
    --test contract cancellation_capabilities
  # Five extra runs per dtype of the built CUDA-feature library binary's
  # cancellation_capabilities filter.
done
cargo test --locked -p tritium-serve --features serve
cargo test --locked -p tritium-serve --lib
# Five extra runs of the built serve-feature contract binary's
# cancellation_capabilities filter.
cargo clippy --locked -p tritium-serve --features cuda --all-targets -- -D warnings
cargo clippy --locked -p tritium-serve --features serve --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

Every command was timeout-bounded in a fail-fast managed job. Terminal state,
counts and binary identities above establish the local lane outcome.
These checks do not qualify a model-bound HTTP watchdog, cancellation latency,
KV/resource reclamation, SOTA quality/speed or the public release. Strict Qwen
declarations are source-level metadata, not a new Qwen model execution receipt.

## Remaining work and custody

Next cheap software work is optional-output native lockstep batch decode and
worker query adoption: preserve the actual graph/eager route, settle submitted
work on cancellation, publish no row output/partial promotion, preserve all
entry target prefixes/watermarks/pages, and recover connected peers on the next
tick. Dense/paged f32/f16 and real response/drain tests must prove it before
upgrading the `worker_iteration` declaration. No running kernel is preempted.

Candidate-bound worker/HTTP latency, KV/resource/concurrency/fault receipts,
the physical framework/backend/browser matrix, PTQ/refinement and Qwen gates,
packaging, deployment, zoo/docs and independent public activation remain open.
No fitting/capture/cloud job or old campaign scratch deletion was authorized or
performed. Existing cache was reused; no new persistent scratch directory was
created. This evidence note is a retained deliverable. Unrelated staged/WIP
changes remain outside the milestone.
