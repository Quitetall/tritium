# Native tree cancellation — local implementation evidence

Date: 2026-10-09 (local EDT; checks continued on 2026-10-10 UTC).
Source base: `ea8325d5dc2d6869b04c6cf73b9b11429b15754d`.
Contract: research ADR 0051, decision 8; serving plan 0052.

## Outcome and scope

Native CUDA single-sequence greedy and host-logit tree verification, plus
single dense/paged batch-row greedy verification, now accept the same borrowed
runtime-free cancellation query as controlled prefill. ModelRunner exposes
those native interfaces without host fallback. Ordinary methods share the
implementation with a never-cancel query; mathematical operations, backend
selection and captured-graph eligibility are unchanged.

Cancellation before work is a no-op. Entered single-sequence tree work
invalidates older pending-tree authorization as before. Cancellation between
eager launch groups or outside graph capture/replay settles submitted stream
work and returns scratch to its owner, with committed KV watermarks unchanged.
Greedy verification checks before promotion; cancelled logit verification
cannot authorize a later commit. Single-row cancellation preserves peer rows,
page mappings, reservations and unrelated solo pending-tree authorization.
Driver/validation errors are not converted into successful cancellation.

This is **not** serving call-site propagation, multi-slot tree cancellation,
drafter-chain cancellation, Qwen qualification, measured cancellation latency,
resource high-water evidence, SOTA evidence or public release approval. Graph
replay remains enabled and is not preempted; long operations still delay the
next cooperative checkpoint. Cancellation arriving after the final check may
race with commit, as specified by ADR 0051.

## Red loop and diagnosis

An entry-only additive native interface first exercised the existing real
tree path. The regression requested cancellation on its second query and
required no output, unchanged cache length and bitwise prefix KV preservation.

```bash
env CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target RUSTC_WRAPPER= \
  CARGO_BUILD_JOBS=2 TMPDIR=/mnt/4tb/tmp TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f32 \
  timeout 300 cargo test --locked -p tritium-nn --features cuda \
  --test runner_cancellation_cuda \
  resident_tree_cancellation_stops_before_promotion -- --exact --nocapture
```

It failed with `entered tree verification ignored cancellation` twice:
0.64 seconds through Cargo and 0.54 seconds through the compiled test binary.
An earlier fixture `usize`/`u32` compile error was corrected before those
runs; it was not cancellation evidence. The one-variable semantic probe passed
the query to the shared greedy verification body and checked before promotion:
the same regression passed in 0.76 seconds. This supports the missing-query
propagation hypothesis rather than output suppression after promotion.

The final implementation adds launch-group checkpoints and graph-boundary
checks at that shared native seam, then extends the same forward contract to
host logits and a single batch row. An incorrect BackendError import in the
expanded test was separately corrected; that setup failure is not a runtime
verdict.

## Physical tiny-fixture checks

Machine: `onyx-maurader-BrianBigPC`; RTX 4090,
UUID `GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, driver `615.71.09`.
No large model, dataset or redistributed weights: deterministic synthetic
two-layer ternary ReLU2 fixture, 64 hidden/FF dimensions, GQA 2:1,
head dimension 32, vocabulary 8. Context 16 takes the existing captured graph
route; context 12289 naturally exceeds its 48 KiB context-smem eligibility
limit and takes eager. No graph/backend kill switch is used to pass the gate.
The solo graph-route assertion requires a real captured bucket.

`runner_cancellation_cuda` runs eight tests (the existing two prefill/decode
checks plus six tree checks):

- every observed solo greedy checkpoint, empty and nonempty prefixes, bitwise
  never-cancel parity, prefix rollback and committed-output/KV recovery;
- every observed host-logit checkpoint, bitwise logits recovery, revocation of
  entered authorization and preservation of pre-entry authorization;
- every observed single-row checkpoint on dense/paged and graph/eager paths,
  peer bytes/positions, page mappings/free-page count, unrelated solo authority
  and bitwise recovered committed KV;
- cancellation during a cold capture lifecycle, followed by safe graph replay;
- ModelRunner facade cancellation with no host-cache fallback;
- the original entered-verification regression.

Final strengthened source passed all eight through Cargo with required-device
f32 KV, single-threaded: 4.48 seconds. The same binary passed all eight with
required-device f16 KV: 5.07 seconds. Three further complete repetitions per KV
dtype passed (48 additional executed checks), in 4.24–4.53 seconds per suite.
These are suite durations, **not** cancellation latency measurements.

Compiled test binary SHA-256:
`260935b559470930280eef121a1f7e4f62d73e46ddfa49fec5cf5fe11dc1b0ab`.
Physical command (repeat three times per `TRITIUM_KV=f32|f16`):

```bash
TRITIUM_REQUIRE_CUDA=1 TRITIUM_KV=f16 timeout 15 \
  /mnt/4tb/build/cargo/70/e33462ce184cce/debug/deps/runner_cancellation_cuda-f804ff2ad2409381 \
  --test-threads=1
```

## Compatibility and local gates

Managed validation: `tritium-native-tree-validation-20261010.service`,
invocation `537b49d406e84dc2921b8b511e933e78`, using the existing Cargo cache,
empty Rust wrapper, two build jobs and `/mnt/4tb/tmp` temporary files.
Its commands are:

```bash
timeout 900 cargo clippy --locked -p tritium-cuda -p tritium-nn \
  --features tritium-cuda/cuda,tritium-nn/cuda --all-targets -- -D warnings
timeout 900 cargo test --locked -p tritium-serve --features serve
timeout 300 cargo test --locked -p tritium-nn \
  --test training_dense_weights --test qwen35_text_runner
timeout 120 cargo fmt --all -- --check
git diff --check
```

Warnings-denied CUDA/NN all-target Clippy, the complete serving suite, all 14
strict-Qwen and six host-runner fixture tests, formatting and diff whitespace
checks passed. The managed unit completed at 22:46:24 EDT with exit 0 after
1 minute 2.354 seconds; the CPU fixture summaries and terminal journal were
observed, not inferred from an unloaded unit's default result.

## Remaining and cleanup

Next: propagate controlled tree verification through real worker/Generator
tree jobs and speculative loops, add atomic multi-slot cancellation without
disabling that dispatch, and cover drafter reconciliation/enrollment/chains.
Capability reporting, finer checkpoints, real-model latency/KV/resource
receipts, exact OCI/Kubernetes qualification and independent release gates
remain open. Previous-head hosted CI was green in all five workflows; it is
not evidence for this new source.

No per-run scratch directory, large model output or campaign copy was created.
Existing build caches were reused. This note, source and regression tests are
retained intentionally. Historical scratch/campaigns and foreign worktree edits
and staging were not deleted or incorporated.
