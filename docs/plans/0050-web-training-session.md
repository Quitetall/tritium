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
const artifact = await session.export();
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

Status: **DONE** — `@tritium-ai/web@1.1.0-rc.0` is a private local-RC package
with an exact lockfile, strict TypeScript 7 check, deterministic browser ESM
bundle, fail-closed manifest tests, BLAKE3-bound manifest/vector identities,
license/notice payloads and a clean-directory archive import. Publication
remains disabled.

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

Status: **IN PROGRESS** — `tritium-wasm` now builds as an actual
`wasm32-unknown-unknown` cdylib, the local npm archive bundles the bindgen guest,
and Node instantiation executes all 35 operations / 114 canonical cases twice
with zero failures under the guest's 64 MiB caller-buffer ceiling. The receipt
binds the exact guest bytes, embedded manifest/vector identities, deterministic
normalized execution digest and linker-enforced 192 MiB linear-memory maximum.
The guest now exposes a strict V1 request/response ABI using exact f32 bit
patterns, canonical request/output digests, structured stable errors and
pre-mutation output preservation. Its JSON transport is capped at 8 MiB, its
caller buffers at 64 MiB, and V1 `u64` fields are JavaScript-safe integers;
the JavaScript boundary maps release-profile guest aborts to `guest_trap`.
The wrapper limits itself to JSON serialization/size checks, validates success
receipts by recomputing canonical request/output digests, and leaves semantic
request validation plus stable error identity to Rust.
Typed lifecycle compilers now derive canonical checkpoint/resume layouts for
all five optimizers and strict SALT V2 export/reload requests, with exact-state
round trips through the admitted guest. A bounded lifecycle state owner copies
optimizer planes and commits and resumes only after canonical guest success; a
separate boundary admits caller-supplied SALT packages through strict reload.
The canonical corpus now generates the 31-operation / 57-execution role
registry used to compile immutable typed-array stores into exact forward, VJP
and optimizer-step portable requests. The package gate rejects registry drift,
unknown buffers, role drift, dtype/shape mismatches and malformed compiled
plans before guest entry. It remains explicitly `wasm-fallback`; session-owned
forward/VJP/optimizer execution now uses one admitted guest without per-dispatch
guest rehashing. A canonical 64 MiB-bounded `TRWEBP1` container initializes
root parameter and optimizer-state owners with BLAKE3 integrity and exact
little-endian lanes; aliases, gradients, activations and loss seeds are derived
from the compiled plan. Same-kind optimizer groups now checkpoint through
canonical plan-0049 bytes and resume atomically into fresh sessions with exact
step/plane recovery. Structural compilation, mixed-optimizer admission and
static portable buffer/request/lifecycle capacity checks fail before guest
creation. Lower-level schedule/lifecycle compiler failures normalize to the
stable session error surface. Receipts now carry planner-accounted phase peaks.
Forward, backward, step, checkpoint, resume and export accept an optional
`AbortSignal`; pre-dispatch and
cooperative in-flight cancellation return an identity-bound recoverable
failure receipt without advancing state. Adapters must commit cancellation
atomically and reject other recoverable typed errors before mutation. Typed
device loss during validation, preparation or a prepared
operation returns a nonrecoverable failure receipt, clears the active result,
makes the session terminal and disposes allocated adapter state exactly once.
Unknown failures or malformed post-dispatch results fail closed the same way.
Explicit disposal terminalizes before cleanup and permits only cleanup retry
after failure.
These structural receipts are not substitutes for the still-open physical
browser memory and device-loss evidence.
All 31 non-lifecycle operations now receive allocation-free canonical ABI,
forward-shape, attribute-domain and worst-case scratch validation before any
adapter allocation. Physical Chrome/Firefox runs remain open.

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

Status: **IN PROGRESS** — the strict public types, fail-closed adapter boundary,
backend/fallback policy, manifest coverage validation, immutable capability
snapshot, receipt/memory validation, serialized state machine, failure-stable
transitions, checkpoint/resume/export byte isolation and idempotent terminal
dispose are implemented and tested. Recipes now compile before `adapter.prepare`
may allocate into an immutable operation schedule and 16-byte-aligned static
buffer plan; tensor ownership, tied-parameter allocation, exact batch staging,
safe-integer byte accounting, preparation peak ceilings, and one-gradient /
one-optimizer ownership for each tied-parameter group fail closed.
The adapter boundary now separates allocation-free, non-mutating, non-retaining
structural/subset validation from persistent preparation; preparation may
allocate decoded state but may not mutate or retain its inputs. The complete
31-operation geometry/attribute catalog is checked against every canonical
forward/step success, output-shape drift and bounded attribute failures.
Preparation peak accounting includes the isolated validation and preparation
payloads. The compiler now derives the reachable reverse-mode VJP schedule from
the single declared loss, assigns canonical backend roles to saved inputs and
cotangents, reserves the loss seed, marks declared gradients for clearing at
each backward boundary, and emits deterministic `graph.add` fan-in reductions
into each tied parameter owner's sole gradient buffer. Generated canonical
bindings now lower every frozen forward, VJP, fan-in and optimizer step into a
typed portable dispatch with copied inputs and explicit output buffer IDs.
The bundled fallback now owns decoded buffers, stages checked batches, executes
that schedule through one prepared guest, resets reverse seeds, commits each
guest dispatch only after validated success and commits multi-group optimizer
steps atomically. Default `backend: "wasm"` and admitted `auto` fallback select
it without caller adapter wiring. Checkpoint and resume use canonical WASM
transactions directly; candidate planes and step commit only after every
returned plane validates. The built-in adapter now derives canonical B3 SALT V2
packages from the current owned parameters at every `graph.salt_ste` export
site, fits one to three additive planes using stored f16 group128 scales, and
passes the resulting bytes twice through the strict Rust reader/writer over a
direct binary WASM boundary before release. The compiled plan accounts the
package, fit scratch, doubled semantic storage, six simultaneous package
copies, a fixed metadata margin and returned artifacts in
`exportPackageBytes`, `exportPeakBytes` and the session peak. Required exports
fail before adapter allocation when no
target exists, a row boundary cannot share canonical group128 scales, the
plane count exceeds the SALT V2 container, or the exact package exceeds 8 MiB.
Physical WebGPU device-loss injection remains open; the public transaction and
failure-receipt contract is complete.

Compile a `TrainingRecipeV1` into an immutable operation schedule, buffer plan
and optimizer-state layout. Preparation validates all roles, shapes, dtypes,
alias groups, memory ceilings and export targets before allocating device
state.

- Parameters, gradients, optimizer state and activations have explicit owners.
- Tied parameters occupy one allocation and receive one accumulated update.
- Forward/backward/step reuse prepared tensor owners; transient request
  snapshots and atomic candidate outputs remain capacity-bound.
- Checkpoint/resume uses the canonical plan-0049 lifecycle encoding.
- Export writes canonical SALT V2 bytes, admits them over a binary WASM
  boundary, proves the guest output equals the live-state artifact, and
  immediately strict-reloads those bytes again before returning.
- Certified cancellation leaves the session reusable from its last complete
  step. Device loss, unknown post-dispatch failure, and malformed results make
  it terminal with a typed receipt; partial state is never presented as
  committed.

Reference tests compare every lifecycle transition and final artifact byte for
byte with the CPU adapter.

## Slice 4 — complete WebGPU backend

Status: **IN PROGRESS** — the npm archive now embeds a curated portable-training
candidate set copied byte-for-byte from `tritium-wgpu`, and binds those bytes
plus a candidate 31-operation module-dependency index into one SHA-256 identity.
Generation fails on manifest drift or missing candidate modules. The native
inference-only `mpgemm.wgsl` is intentionally outside this training candidate
set. The shared execution-aware 57-form catalog—including validated WGSL,
selectors, entry points, binding layouts, dispatch geometry and repeated
stages—remains open with browser device admission, resident buffer scheduling,
pipeline compilation/dispatch and physical conformance.

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
