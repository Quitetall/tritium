# `@tritium-ai/web`

Tritium's compiled browser-training package. The current release-candidate
contains the frozen language-neutral training identities and the checked
`WebTrainingSession` lifecycle. Session calls are serialized and fail closed on
schema, backend-policy, capability, receipt, memory, ordering and concurrency
violations. Recipes compile before `adapter.prepare` may allocate into an immutable,
16-byte-aligned buffer/schedule plan. Tensor owners are explicit, tied
parameters share one allocation and one compiled gradient/optimizer owner, and
batch names/dtypes/shapes are checked against the plan. Structural planning and
the built-in adapter's optimizer-subset, portable-buffer, worst-case request
JSON and lifecycle-capacity gates run before guest creation; complete
operation-specific geometry/attribute admission remains a release gate.
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

`PortableWasmLifecycleState.create(...)` owns copied optimizer planes, commits
and resumes atomically through that guest. Its separate `admitExport(...)`
boundary strict-reloads caller-supplied SALT packages before returning them.
Complete pre-allocation operation geometry admission and state-derived model
export remain release gates.

The built-in adapter remains explicitly `wasm-fallback`; it never satisfies a
WebGPU gate. WebGPU implementation and state-derived SALT export remain under
construction. `backend: "webgpu"` still returns `adapter_unavailable` unless a
real WebGPU adapter is supplied.

This package is private while the local v1.1 release candidate is under
construction. Registry publication requires explicit release authorization.
Building the archive from source requires the pinned
`wasm32-unknown-unknown` Rust target and `wasm-bindgen-cli 0.2.126`; the build
fails on tool drift and verifies the guest's 192 MiB maximum-memory declaration.
Archive consumers need neither Rust nor wasm-bindgen.
