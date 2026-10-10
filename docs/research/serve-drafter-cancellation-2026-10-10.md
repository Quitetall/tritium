# Cooperative native drafter cancellation

Date: 2026-10-10. Decision: private ADR 0051, decision 11; serving plan 0052.
Base tree: `8924b3f2ada6b9a57f9f5373106b5db12dde5122`.

## Implementation

Native device-feedback chains, graph-argmax steps and batched drafting now
accept a borrowed cancellation query. Ordinary methods share a never-cancel
implementation. Queries run outside graph capture/replay; graphs remain
enabled. The solo chain retains device feedback and its single trailing chain
readback. Batched drafting retains its existing per-step batched argmax and
host-feedback path; this change does not revive the previously reverted
batched device-chain experiment.

Successful cancellation settles owned streams, publishes no draft output,
and preserves entry watermarks, committed prefix bytes and page mappings.
Batch cancellation also restores entry liveness, including dead rows and rows
halted by EOS. Provisional rows above a watermark are discarded, not restored
byte-for-byte. Entered solo decode invalidates pending tree authorization as
ordinary decode does; batch drafting preserves unrelated solo authorization.
Driver and validation failures remain errors with their existing partial-state
semantics. A cancellation wait failure is not reported as successful rollback.

ModelRunner's additive controlled interfaces distinguish absent native support
(`ResidentOpError::Unavailable`) from cancellation (`Ok(None)`). Historical
ordinary interfaces retain their fallback behavior. Cancellation cannot
accidentally trigger the per-step/host-logit ladder.

Serving passes response closure or drain through reconcile/gap prefill,
native chains, scalar drafting, full/delta enrollment and batched gap/draft
work. Entered request-owned reconciliation cancellation resets drafter state
for a fresh resync; entry-only cancellation leaves it untouched. Enrollment
checks before adoption and cannot publish cancelled work. A cancelled grouped
round releases only disconnected target rows, preserves connected peers,
discards overfed drafter enrollment and returns to worker retirement without
same-tick lockstep decode. Existing typed drain framing remains in the worker.

This is cooperative cancellation. A cancellation arriving after the final
checkpoint can race successful publication. Individual kernels and graph
replays are not preempted. The query must be cheap, nonblocking and non-panicking;
its invocation count is not an external contract.

## Verification surface

An entry-only helper prototype failed the host reconcile/fallback regression
before controlled forwarding was implemented (0 passed, 1 failed, exit 101).
The completed helper subsequently passed that regression. Native checks sweep
all observed chain and scalar checkpoints, empty/nonempty prefixes, EOS/no-EOS,
dense/paged batched KV, dead rows, unrelated solo tree authority, cold graph
capture through ModelRunner, and exact ordinary-path recovery with no host KV
adoption. Separate processes select f32 and f16 KV.

Serving checks cover real response closure and drain during reconcile and
native drafting; full/delta enrollment with a live peer; unavailable versus
cancelled facades; solo-cycle cancellation before target work; and every
observed checkpoint across an actual multi-slot round. The latter checks target
prefix bytes, page ownership/free counts, one-time disconnected-row release,
unchanged peer history/budget/no emission, and a next-tick peer recovery stream
against independent ordinary greedy decode. Actual grouped graph capture is
asserted. Tiny fixtures do not substitute plain decode for the native path.

Initial f32 checks passed: three native draft tests (4.73s) and the grouped-round
sweep (11.08s). These exploratory runs are not the final validation lane.

## Final validation

Device: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Existing cache: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`;
observed binaries are in `/mnt/4tb/build/cargo/70/e33462ce184cce/debug/deps`.
Environment: `RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`,
and `TRITIUM_REQUIRE_CUDA=1` (absent CUDA must fail, not silently skip).

Managed source-quiescent validation:
`tritium-drafter-cancel-final-20261010-v1.service`, invocation
`778c4e14a44449ea8a8ba41efc85c81d`. It completed at
2026-10-10 01:19:03 EDT with `SubState=exited`, `MainPID=0`,
`ExecMainStatus=0` and `Result=success`.

Completed lanes:

- 14 native cancellation tests per dtype (6.42s f32, 6.62s f16).
- 84 CUDA-feature serving tests per dtype (54.73s f32, 52.37s f16).
- Five extra repetitions per dtype of three native drafter and seven serving
  drafter checks: 100 additional passing executions.
- 155 CUDA library tests, 12 explicitly ignored, 0 failed (75.58s).
- 112 serve-feature software checks and three runtime-free compatibility checks.
- CUDA/NN/serving all-target warnings-denied Clippy (1m33s), formatting and
  whitespace checks. Zero-case gated binaries/doc-tests are not added checks.

Final physical binary SHA-256 identities:

- `runner_cancellation_cuda-f804ff2ad2409381`:
  `3a26cad95b6b21c693cfcca791bf3e1edb98831696424af9c8e3e1d9afd78dd6`.
- `tritium_serve-8174e1b89dd2f7e4`:
  `40a5be1b1e1a55167fd379ade0f321fe5851c418fdb86016f4cd33d2cb604e5a`.

```sh
for kv in f32 f16; do
  export TRITIUM_KV="$kv"
  cargo test --locked -p tritium-nn --features cuda \
    --test runner_cancellation_cuda -- --nocapture
  cargo test --locked -p tritium-serve --features cuda --lib -- --nocapture
  # Five further repetitions per dtype of each built binary's filter:
  # runner_cancellation_cuda resident_draft --nocapture
  # tritium_serve drafter_ --nocapture
done
cargo test --locked -p tritium-cuda --features cuda --lib -- --test-threads=1
cargo test --locked -p tritium-serve --features serve
cargo test --locked -p tritium-serve --lib default_cancellable
cargo clippy --locked -p tritium-cuda -p tritium-nn -p tritium-serve \
  --features cuda --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

Each command is individually timeout-bounded within the fail-fast managed job.
The tests validate implementation, not the release candidate. The completed
local lane does not independently clear an empirical public-release gate.

## Remaining release obligations

Precise adapter capability reporting remains a software obligation. Candidate-
bound HTTP cancellation latency, KV/resource high-water marks and fault
behavior remain empirical gates. These fixtures do not qualify Qwen quality,
physical artifact bytes, speed, foreign-framework capture, OCI/security,
Kubernetes, the hardware/browser matrix or independent public release.

The base commit's hosted CI, wheels, CodeQL, docs and capstone runs completed
successfully (38025463648, 38025463713, 38025463649, 38025463672,
38025463693). They do not validate this new source or inherit qualification.

No new persistent scratch directory, fitting/capture campaign or cloud job was created.
The existing shared build cache was reused. Old campaign scratch was not
deleted; the owner's cleanup confirmation remains outstanding. This note is a
retained deliverable, not temporary scratch.
