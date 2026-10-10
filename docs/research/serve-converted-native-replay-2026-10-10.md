# Converted-model native serving replay — 2026-10-10

Scope: plan 0052 and ADR 0051, local compatibility-router coverage. This is
performer evidence, not an independently admitted production/release receipt.

## Implementation and inputs

- `20090af30fecbef3a8ab28d40d2e2558529d19e3` adds the actual converted-model
  SSE/disconnect/recovery test and makes explicitly selected real-model tests
  fail without `TRITIUM_SERVE_E2E=1`, instead of returning a false green skip.
- `a31628378386381d5ae1b9af28e2235a3c7bd86e` fixes a documentation-paragraph
  Clippy violation. It is the final clean source verified below.
- Intel Core i9-14900K, x86_64 Linux, native CPU backend; Rust 1.98.0
  (`88d9e12ae178fab0fb5cc050a94da85685d449ea`), unoptimized test profile.
- Existing local SALT SmolLM2-135M conversion with its actual HF tokenizer,
  configuration and preserved safetensors. No fitting/download was performed.
- `model.tslb`: 133,897,831 bytes; SHA-256
  `b97706476a5b57ac18a50a429a19d1dd7d0a5f442b488ffa5106963e0122c320`.
- Preserved `model.safetensors`: SHA-256
  `e4b569aad3be0d8bff551e81671ca27815eed43db4a2c3bbe0bd6c596cc03ce0`.
- `config.json`: SHA-256
  `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`.
- `tokenizer.json`: SHA-256
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
- `tokenizer_config.json`: SHA-256
  `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424`.
- Conversion `receipt.json`: SHA-256
  `659e6bc6a086b56f78fcdc8806cdf1a2746a6ed7eeb17612cfc517da2bb85845`.

## Checks and observed outcomes

The baseline clean `d50205eb` smoke passed: one real-model test, 21.97 seconds
including model loading. Running that same old executable explicitly without
opt-in returned exit 0 and reported one pass after printing a skip message.
That demonstrated the evidence flaw before the guard change.

The new lifecycle check uses no mock or substituted inference adapter. It:

1. Obtains an actual greedy four-token non-streaming reference.
2. Collects native SSE, checks role-first framing and `[DONE]`, and compares
   text, terminal finish reason and usage against that reference.
3. Drops one response after observing the worker prefill phase, and another
   after receiving actual native content and observing the decode phase.
4. Requires idle worker, empty queue, worker liveness and zero backend faults.
5. Requires identical subsequent choices and token accounting after each case,
   and exactly two recorded client disconnects.

The first draft failed with HTTP 400 because the test sent `stream_options`
on a non-streaming request. The existing server correctly rejected it; only
the test request was corrected. The subsequent dirty-tree run passed both
converted checks in 53.17 seconds. A clean `20090af3` run also passed both in
49.28 seconds. Those are separate runs, not relabeled final-source evidence.

The first software invocation passed 118 executed tests, with three real-model
tests explicitly ignored. Its subsequent scoped Clippy failed with
`doc list item without indentation` / `clippy::doc_lazy_continuation`.
The normal pre-push hook independently caught the same issue and blocked
`20090af3`; no hook bypass was used. The paragraph-only repair is `a3162837`.

Final clean-source invocation `bec2a256e14d48ed9440721d36873960` completed
successfully, executing these commands with separate 900/600-second limits:

```bash
TRITIUM_SERVE_E2E=1 TRITIUM_CONVERTED_PATH=/models/local-converted-model \
  cargo test --locked -p tritium-serve --features e2e --test e2e \
  serve_e2e_converted -- --ignored --nocapture --test-threads=1
cargo clippy --locked -p tritium-serve --features e2e --all-targets -- -D warnings
```

Both real-model checks passed in 61.30 seconds, with no ignored selected tests;
scoped warnings-denied Clippy passed. Final-source missing-opt-in execution
failed as required (exit 101), before loading the model. Raw final positive and
negative outputs are retained in
`docs/research/evidence/serve-smollm2-a3162837/`.
The negative raw output ends with a blank line; its deterministic gzip copy
preserves every byte while satisfying the staged whitespace gate. Decompression
was compared against the original; the uncompressed original remains archived.

Normal push invocation `e299890f71274e31bd01bc292c985a1b` completed with
success/exit 0 through the commit-tree formatting and workspace/optional-feature
warnings-denied Clippy hooks. Remote readback confirmed exact `a3162837`.
The unrelated staged diff remained byte-identical (SHA-256
`c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`).
Hosted CI and independent empirical clearance remain separate obligations.

Observed final-run disconnect-to-idle samples were 78.339456 ms for the prefill
case and 15.623691 ms for decode; subsequent full completion checks took
7.157022274 and 6.013440735 seconds. These are single noisy measurements on a
busy workstation, not latency bounds, representative benchmarks or GPU speed.
The phase gauge identifies worker phase, not an individual native projection
checkpoint. The 30/60-second assertions are test execution limits, not amended
release thresholds.

## Limits and remaining release fronts

This uses the in-process legacy compatibility router, not a public TCP listener
or schema-v3 admitted production artifact. Zero paged-KV gauges on this path
cannot prove KV reclamation. Identical recovered outputs do not measure
resident-memory return, prove every cancellation checkpoint, or qualify
kernel preemption, concurrency, deployment, SOTA or model quality. No Qwen,
MTP, paid compute, publication or human activation gate was cleared.

The full v1.1 objective remains open:

1. Candidate-bound native HTTP deadline/disconnect, latency, KV/resident-memory,
   concurrency and failure receipts.
2. PyTorch/HF lifecycle, estimator/baseline, PTQ/refinement and distributed
   empirical qualification.
3. Remaining physical backend qualification and the complete physical
   Chrome/Firefox/Safari training/lifecycle/fault matrix.
4. Whole-model ONNX, exact-candidate packages/compatibility, tutorials and Colab.
5. Authorized scalable recipe freeze; Qwen language+MTP PTQ/refined quality,
   physical bytes, runtime, memory and reproduction.
6. Exact-candidate OCI security/runtime and Kubernetes/serverless/observability
   failure qualification.
7. Audited four-model zoo, bounded claims/docs, governance and community.
8. Independent second-machine/operator replay and clearance, signed clean
   release, human activation, authorized publication and post-publication smoke.

Existing unrelated WIP and staged additions were preserved. The shared build
cache was reused. No August cache or campaign artifact was deleted. Per-run
logs, including failed attempts, are retained under the project's durable
`archive/verification/serve-smollm2-a3162837-20261010`, not left in scratch.
