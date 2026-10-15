# Grouped native tree cancellation

Date: 2026-10-10. Decision: private ADR 0051, decision 10; serving plan 0052.
Base tree: `43ed455f40ee6d476dd12c9bab6c29b3f11dd096`.

## Change

The native grouped greedy verifier and ModelRunner facade now accept the same
borrowed, runtime-free cancellation query as the single-row interfaces.
Ordinary grouped verification shares the implementation with a never-cancel
query. It still runs one concatenated trunk and one head/argmax over the group;
this is not a sequential single-row or plain-decode replacement.

Eager launch groups have cooperative checkpoints; graph checkpoints are only
outside capture/replay. Cancellation settles both owned streams before batch
scratch becomes reusable or a paged row can retire. All accepted paths are
built before one final checkpoint and any promotion. There is deliberately no
query between individual promotions: that would create a partially committed
cancellation. `None` means no output or promotion in any selected row. Committed
prefixes, watermarks, page mappings and unrelated solo-tree authorization are
preserved. Provisional bytes above a watermark are disposable, not rolled back.
Driver/validation failures remain errors; driver errors after promotion begins
retain the existing non-atomic error semantics. Cancellation requested after
the final checkpoint can race a successful commit.

The continuous-batch speculative round queries drain or closure of any selected
response. A cancelled group releases only disconnected rows; connected peers
keep their target/host histories and budgets. The worker discards the overfed
drafter enrollment and re-enters its retirement/admission loop, rather than
running lockstep decode in the same tick. Existing drain framing stays with
the outer worker. This does not interrupt drafting or enrollment itself.

## Regression and evidence surface

An entry-only optional-output implementation failed the required-device
`resident_tree_group_cancellation_has_no_partial_commit` test: cancellation
requested at the second observed checkpoint returned output instead of `None`.
The one-test red run exited 101, with 0 passed/1 failed (0.57s); compilation
completed in 56.51s. After native checkpoints, the initial one-test green run
passed (0.73s; compilation 55.44s).

Expanded physical native checks cover every observed checkpoint on graph and
natural eager routes (`n_ctx=16` and `12289`), f32/f16 KV in separate processes,
dense/paged storage, empty/nonempty selected prefixes, two distinct tree
shapes in reverse slot order, and a third unrelated live row. Recovery compares
accepted tokens, committed KV bytes and positions with ordinary sequential
slot verification. They preserve page mappings/free counts and unrelated solo
pending-tree commit authority. Duplicate-row refusal remains an error with no
prefix change. Cold-capture facade cancellation/recovery requires a real
grouped captured bucket and no host-cache adoption. A hidden BatchKv debug
accessor makes graph/eager dispatch observable rather than inferred from a
successful token result.

Serving checks separately cover selected-response membership, drain, a real
response/drain transition injected after native device uploads, and a complete
multi-slot speculative round. The latter runs real batched drafting before
target cancellation, proves no peer emission/history/budget mutation, releases
only the disconnected row once, drops enrollment as the worker does, and
recovers the connected peer's emitted tokens against ordinary greedy decode.
These are tiny physical adapter/round fixtures, not a candidate-bound HTTP
watchdog, real-model latency or Kubernetes qualification.

## Validation

Device: NVIDIA GeForce RTX 4090,
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
Existing cache: `CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target`;
observed binaries are in `/mnt/4tb/build/cargo/70/e33462ce184cce/debug/deps`.
`RUSTC_WRAPPER=`, `CARGO_BUILD_JOBS=2`, `TMPDIR=/mnt/4tb/tmp`, and
`TRITIUM_REQUIRE_CUDA=1` prevent absent hardware from silently passing fixtures.

Source-quiescent validation is managed by
`tritium-group-cancel-final-20261010-v1.service`, invocation
`d6be425960bc409283b715d95c66f3b9`. It completed at
2026-10-10 00:43:21 EDT with `SubState=exited`, `MainPID=0`,
`ExecMainStatus=0` and `Result=success`.

Completed lanes:

- 11 native physical tests for each KV dtype (2.01s f32, 2.20s f16).
- 78 CUDA-feature serving library tests for each dtype (63.11s f32, 52.39s f16).
- Five extra repetitions per dtype of the two native grouped tests and three
  serving grouped tests: 50 additional passing test executions.
- 155 CUDA-feature library tests, 12 ignored, 0 failed (32.12s).
- 112 serve-feature software checks and three runtime-free compatibility checks.
- CUDA/NN/serving all-target warnings-denied Clippy, formatting and whitespace
  checks. Zero-case gated binaries/doc-tests are not extra qualifying checks.

Final physical binary SHA-256 identities:

- `runner_cancellation_cuda-f804ff2ad2409381`:
  `5a5da5e8285b00429c833acebcc596d2d9510c8d8620a0e66192662c57a359a6`.
- `tritium_serve-8174e1b89dd2f7e4`:
  `3dfe329234add53126a7db8b949cbe87eecab635c3a9516b93a57f44cf3f6b85`.

Commands (each individually timeout-bounded; the managed shell is fail-fast):

```sh
for kv in f32 f16; do
  export TRITIUM_KV="$kv"
  cargo test --locked -p tritium-nn --features cuda \
    --test runner_cancellation_cuda -- --nocapture
  cargo test --locked -p tritium-serve --features cuda --lib -- --nocapture
  # Five further repetitions per dtype of each built binary's grouped filter:
  # runner_cancellation_cuda resident_tree_group --nocapture
  # tritium_serve grouped_ --nocapture
done
cargo test --locked -p tritium-cuda --features cuda --lib -- --test-threads=1
cargo test --locked -p tritium-serve --features serve
cargo test --locked -p tritium-serve --lib default_cancellable
cargo clippy --locked -p tritium-cuda -p tritium-nn -p tritium-serve \
  --features cuda --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

## Scope and remaining work

This milestone does not qualify the release candidate, SOTA quality, speed,
physical model bytes, real-model cancellation latency, resource high-water
marks, foreign-framework capture, or the deployment failure matrix. In-operation
drafter chain/reconcile/enrollment cancellation and explicit capability
reporting remain next software obligations. Real-model worker/KV/resource,
CPU/CUDA OCI/security/Kubernetes and independent release gates remain open.

The previous base revision's hosted docs, capstone, wheels, CodeQL and CI runs
all completed successfully (38023929485, 38023929446, 38023929460,
38023929444 and 38023929439). Those runs do not validate this new source or
inherit physical/release qualification.

No new scratch directory, private model payload, fitting/capture campaign or
cloud job was created. Existing build cache was reused. No old scratch or
campaign output was deleted; cleanup still requires the owner's confirmation.
