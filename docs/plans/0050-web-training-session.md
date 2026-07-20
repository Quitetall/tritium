# 0050 — Compiled browser training session

Status: **IN PROGRESS** (2026-07-20)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Dependency:** [plan 0049](./0049-portable-training-manifest.md) — schema and
  vectors frozen; native backend parity continues in parallel
- **Package:** `@tritium-ai/web` (local archive only until publication is
  explicitly authorized)

## Goal

Ship a strict-TypeScript browser product that executes a compiled ternary
training recipe through the frozen `TrainingOpManifestV1`. WebGPU is the
accelerated implementation; deterministic WASM is a separately receipted
fallback. Neither path exposes an arbitrary JavaScript autograd graph.

The public lifecycle is:

```typescript
const session = await tritium.prepareTraining(model, config);
const result = await session.forward(batch);
await session.backward(result.loss);
await session.step();
const checkpoint = await session.checkpoint();
await session.resume(checkpoint);
await session.export("model.tsalt2");
```

## Frozen public contracts

Add `packages/tritium-web` with these exported, schema-versioned types:

- `TrainingRecipeV1`, `TrainingBatchV1`, `TrainingResultV1` and
  `TrainingErrorV1`;
- `WebTrainingConfigV1`, including explicit backend policy, memory ceiling,
  deterministic seed and capability requirements;
- `WebTrainingCapabilitiesV1` and `WebTrainingReceiptV1`, carrying the exact
  manifest/vector digests, implementation/build identity, browser adapter and
  physical GPU identity when disclosed;
- `WebTrainingSession`, with `forward`, `backward`, `step`, `checkpoint`,
  `resume`, `export`, `dispose` and read-only `capabilities`;
- `prepareTraining(model, config)` and `inspectWebArtifact(source)`.

Unknown fields, versions, operation IDs, dtypes and artifact formats fail
closed. `backend: "webgpu"` never falls back. `backend: "auto"` may select
WASM only when `allowWasmFallback` is true, and the resulting receipt must say
`wasm-fallback` rather than `webgpu`.

Session calls form a checked state machine:

```text
prepared -> forward-complete -> backward-complete -> stepped -> prepared
    |              |                  |
    +---------- checkpoint/resume/export ---------+
```

Invalid ordering is a typed error and cannot mutate model or optimizer state.
`dispose` is idempotent and terminal.

## Slice 1 — reproducible package and schema mirror

Create the package without a repository-relative runtime dependency:

- `packages/tritium-web/package.json`, strict `tsconfig` files and deterministic
  ESM build;
- generated copies of the canonical manifest, vector digest and shared public
  schemas, plus a generator that refuses drift from `spec/training/v1`;
- package exports for browser ESM, types and the WASM asset;
- tests for exact Rust/Python/TypeScript fixture parity and unknown-schema
  rejection.

`npm pack` must exclude sources, fixtures not required at runtime, temporary
receipts and absolute paths. The archive is inspected before installation.

Gate:

```bash
npm --prefix packages/tritium-web ci
npm --prefix packages/tritium-web run check
npm --prefix packages/tritium-web pack --dry-run
```

## Slice 2 — deterministic WASM controller

Compile `tritium-spec`, the bounded portable executor and SALT V2 strict reader
to `wasm32-unknown-unknown`. JavaScript owns only lifecycle orchestration and
typed-array views; Rust owns validation, request digests, execution semantics,
checkpoint encoding and package admission.

- Use a declared linear-memory ceiling and checked allocation before mutation.
- Do not read wall-clock, random globals, filesystem state or network state
  during execution.
- Seeded stochastic estimators receive the recipe seed explicitly.
- Repeated executions of every bounded vector produce byte-identical outputs,
  checkpoints, artifacts and receipts.
- Oversized shapes and memory growth return typed capacity errors.

The WASM lane executes the complete manifest at bounded shapes in Node and in
each release browser. It remains a fallback identity and cannot satisfy a
WebGPU gate.

## Slice 3 — compiled session and artifact lifecycle

Compile a `TrainingRecipeV1` into an immutable operation schedule, buffer plan
and optimizer-state layout. Preparation validates all roles, shapes, dtypes,
alias groups, memory ceilings and export targets before allocating device
state.

- Parameters, gradients, optimizer state and activations have explicit owners.
- Tied parameters occupy one allocation and receive one accumulated update.
- Forward/backward/step reuse prepared buffers; no allocator is used in the
  steady-state inner loop except capacity-bound batch staging.
- Checkpoint/resume uses the canonical plan-0049 lifecycle encoding.
- Export writes canonical SALT V2 bytes and immediately reloads them through
  the strict WASM reader before returning.
- Cancellation and device loss leave the session either reusable from its last
  complete step or terminal with a typed receipt; partial state is never
  presented as committed.

Reference tests compare every lifecycle transition and final artifact byte for
byte with the CPU adapter.

## Slice 4 — complete WebGPU backend

Implement all 35 manifest operations with WGSL compute pipelines and explicit
VJPs. Kernels may be fused after semantic parity, but the frozen public
operation IDs and vector behavior cannot change.

- F32 is mandatory. F16 is advertised only when the browser feature and every
  declared operation pass a separate vector set.
- Validate `maxComputeWorkgroupsPerDimension`, storage-buffer sizes and binding
  limits before dispatch.
- Keep parameters, gradients, optimizer state and intermediate tensors in GPU
  buffers from prepared-session setup through receipt finalization.
- No per-operation `mapAsync`, `readBuffer`, staging readback or queue fence is
  allowed. Read back only explicit public results, checkpoints, exports and
  final receipts.
- Pipeline caches are scoped to device, build identity, operation, dtype and
  specialization constants. Device loss invalidates them.
- Numerical tolerances come only from the canonical vectors. A browser-specific
  relaxation requires an ADR amendment.

GPU-capture or browser tracing must prove zero steady-state GPU-to-CPU tensor
readback and no hidden WASM execution when the receipt says `webgpu`.

## Slice 5 — real-browser conformance and fault injection

Run an installed package, not source imports, under current stable Chrome,
Firefox and Safari on physical WebGPU adapters. Each lane must:

1. execute every valid and invalid canonical vector;
2. run prepare through forward/backward/step/checkpoint/resume/export/reload;
3. compare the final artifact with the CPU reference;
4. record browser/version, OS, adapter/vendor/device description, limits,
   manifest/vector/build digests and peak buffer bytes;
5. inject device loss, allocation failure, malformed checkpoint, malformed SALT
   artifact, cancellation and out-of-order lifecycle calls.

Software adapters, headless no-GPU modes, emulators and WASM fallback are
structural evidence only. Missing physical identity, skipped tests or a zero
case count fail the release lane.

## Slice 6 — local archive and five-minute tutorial

- Install the exact `npm pack` archive into an empty strict-TypeScript project.
- Pass `tsc --noEmit`, production bundling and a browser smoke without network
  access after fixtures are staged.
- Publish a tiny SmolLM2-135M recipe demonstrating load, one PTQ calibration
  slice, one QAT optimizer step, checkpoint/resume, export/reload and generation.
- Finish within five minutes on the declared Colab/browser reference hardware,
  excluding the first model download.
- Generate capability and performance tables exclusively from admitted
  receipts.

Registry publication and npm name reservation are out of scope until Brian Lam
explicitly authorizes them.

## Verification cadence

```bash
npm --prefix packages/tritium-web run format:check
npm --prefix packages/tritium-web run lint
npm --prefix packages/tritium-web run typecheck
npm --prefix packages/tritium-web test
cargo test -p tritium-wasm -p tritium-wgpu
git diff --check
```

Browser lanes archive trace plus receipt artifacts. Final admission verifies
their manifest/vector/build digests against the candidate revision.

## Stop conditions

- Stop before weakening the complete-manifest, actual-browser, no-readback,
  artifact-parity or fail-closed gates.
- Stop if WebGPU execution requires a host implementation for an advertised
  operation; either implement the kernel or amend ADR 0033.
- Stop before npm publication, hosted browser purchases or paid test services
  without explicit authorization.

## Done criterion

The local npm archive installs in a clean project; strict types and production
bundle pass; deterministic WASM passes the bounded complete manifest under its
own identity; Chrome, Firefox and Safari pass the complete manifest and session
lifecycle on actual WebGPU; traces prove no steady-state readback; checkpoint
and SALT V2 artifacts reload identically through native Tritium; capability and
performance docs are receipt-generated.

## Commit sequence

```text
docs(plan-0050): freeze browser training work order
feat(web): scaffold strict training package
feat(wasm): add bounded portable training controller
feat(web): compile checked training sessions
feat(webgpu): conform portable training manifest
test(web): admit physical browser receipts
docs(web): publish receipted browser tutorial
```
