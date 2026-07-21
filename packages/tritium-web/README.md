# `@tritium-ai/web`

Tritium's compiled browser-training package. The current release-candidate
contains the frozen language-neutral training identities and the checked
`WebTrainingSession` lifecycle. Session calls are serialized and fail closed on
schema, backend-policy, capability, receipt, memory, ordering and concurrency
violations. Recipes compile before `adapter.prepare` may allocate into an immutable,
16-byte-aligned buffer/schedule plan. Tensor owners are explicit, tied
parameters share one allocation and one compiled gradient/optimizer owner, and
batch names/dtypes/shapes are checked against the plan. Structural planning and
all 31 non-lifecycle operations pass allocation-free canonical ABI, shape,
attribute-domain and scratch-ceiling validation before adapter allocation. The
built-in adapter additionally applies optimizer-subset, portable-buffer,
worst-case request JSON and lifecycle-capacity gates before guest creation.
Model/batch bytes are isolated from adapter mutation. Failed
adapter calls do not advance session state. The archive also bundles a real
`wasm32-unknown-unknown` guest; `runPortableWasmConformance()` executes all 114
canonical cases twice inside that guest and returns a source-bound structural
receipt. `executePortableWasmRequest()` sends strict bit-pattern JSON through
the same admitted guest; Rust owns schema validation, typed capacity/error
codes, execution, request/output digests and atomic output mutation. Request
JSON is capped at 8 MiB before guest entry, caller buffers at 64 MiB, and V1
`u64` JSON values are restricted to non-negative JavaScript safe integers.
Expected failures are structured responses; a release-profile guest abort is
mapped by the JavaScript boundary to the stable `guest_trap` error code.
JavaScript gates only JSON serialization and size; Rust remains the semantic
validator, including safe-integer bounds. Successful receipts are checked against locally recomputed
canonical request/output digests, exact dtype, build identity and memory bounds.
Typed lifecycle compilers build canonical checkpoint/resume buffers for SGD,
AdamW, cautious AdamW, int8 AdamW and Muon, and strict SALT V2 export/reload
requests without exposing backend role names or encoded-size arithmetic.
The canonical vectors also generate all 31 non-lifecycle operation bindings
(57 forward/VJP/step signatures). Typed schedule compilers copy caller-owned
buffers into immutable portable requests, resolve tied tensors through their
canonical owner, preserve exact f32 bits, return explicit destination buffer
IDs and fail closed on binding, dtype, shape or compiled-plan drift. Generated
binding freshness is part of `npm run check`.

`encodeWebTrainingPayload(...)` writes canonical `TRWEBP1` bytes containing
root parameters and optimizer state. `decodeWebTrainingPayload(...)` verifies
BLAKE3 integrity, canonical ordering, exact dtype/byte lengths and compiled
ownership before materializing one mutable tensor per owner. `backend: "wasm"`
now selects a built-in session adapter; permitted `auto` fallback does likewise.
It prepares the guest once, then executes compiled forward, reverse and atomic
multi-group optimizer schedules without re-reading or rehashing guest bytes.
Same-kind optimizer groups checkpoint through canonical portable lifecycle
bytes and resume atomically with exact optimizer planes and step count.
Receipts report planner-accounted phase peaks; physical browser memory
receipts remain a separate acceptance gate.

The package bundles a curated portable-training WGSL candidate set copied
byte-for-byte from `tritium-wgpu`, with a SHA-256-bound candidate dependency
index for all 31 tensor operations. `webGpuKernelCandidateBundleV1()` exposes
those staged build inputs. It is not a validated source set or execution plan:
the shared 57-form phase/selector/entry-point/binding/dispatch catalog, device
admission, resident dispatch and physical-browser conformance are still release
gates. Candidate presence alone is not a WebGPU execution receipt.

Forward, backward, step, checkpoint, resume, and export accept `{ signal }`.
Cancellation before dispatch or a cooperative adapter cancellation returns a
typed, identity-bound recoverable failure receipt and leaves the previous
complete state reusable.
Adapters must reject cancellation only after rolling back partial writes and
must emit other recoverable typed failures before mutation. Typed device loss
during validation, preparation, or execution returns a nonrecoverable failure
receipt, clears active results, makes the session terminal and disposes adapter
state exactly once when allocation has begun. Unknown failures and malformed
results after dispatch also terminalize and dispose the adapter. Physical
WebGPU loss injection remains a browser acceptance gate.
Explicit disposal terminalizes the session before cleanup begins; failed
cleanup may be retried, but training never resumes on partially released state.

`PortableWasmLifecycleState.create(...)` owns copied optimizer planes, commits
and resumes atomically through that guest. Its separate `admitExport(...)`
boundary strict-reloads caller-supplied SALT packages before returning them.
The built-in training adapter separately derives a compact B3 SALT V2 package
from live `graph.salt_ste` parameter owners. Export targets are admitted before
allocation, use one to three additive planes with f16 group128 scales, and are
returned only after two direct-binary WASM reader/writer admissions, including
artifact-to-guest and reload byte equality. The compiled plan exposes exact
package bytes and a conservative export-phase peak covering fit scratch,
doubled semantic storage, six simultaneous package copies, a metadata margin
and returned artifacts. Multi-row
targets must align row boundaries to 128 coefficients; this prevents a browser
artifact from silently changing the training quantizer's scale domains.

The built-in adapter remains explicitly `wasm-fallback`; it never satisfies a
WebGPU gate. `createWebGpuTrainingAdapter(device, options)` accepts an
already-authorized WebGPU device and supplies resident forward/backward/step
execution to `prepareTraining`. The factory transfers exclusive device
ownership; disposing the adapter destroys that device. Submitted phases await
queue completion and fail terminally on device loss. It does not yet advertise GPU-native
checkpoint/resume/export. Automatic browser adapter acquisition is also still
open, so `backend: "webgpu"` returns `adapter_unavailable` unless this explicit
adapter is supplied.

This package is private while the local v1.1 release candidate is under
construction. Registry publication requires explicit release authorization.
Building the archive from source requires the pinned
`wasm32-unknown-unknown` Rust target and `wasm-bindgen-cli 0.2.126`; the build
fails on tool drift and verifies the guest's 192 MiB maximum-memory declaration.
Archive consumers need neither Rust nor wasm-bindgen.
