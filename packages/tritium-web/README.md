# `@tritium-ai/web`

Tritium's compiled browser-training package. The current release-candidate
contains the frozen language-neutral training identities and the checked
`WebTrainingSession` lifecycle. Session calls are serialized and fail closed on
schema, backend-policy, capability, receipt, memory, ordering and concurrency
violations. Recipes compile before `adapter.prepare` may allocate into an immutable,
16-byte-aligned buffer/schedule plan. Tensor owners are explicit, tied
parameters share one allocation and one compiled gradient/optimizer owner, batch names/dtypes/shapes are checked against
the plan, and allocation-free validation plus non-retaining preparation close
operation-specific geometry/attribute rules before persistent preparation.
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

The compiled lifecycle WASM controller and WebGPU adapter are still under construction;
the conformance guest is not mislabeled as either. Until a lifecycle adapter is
supplied, `prepareTraining(model, config)` returns a typed
`adapter_unavailable` error; this package never labels a JavaScript fallback as
WebGPU execution.

This package is private while the local v1.1 release candidate is under
construction. Registry publication requires explicit release authorization.
Building the archive from source requires the pinned
`wasm32-unknown-unknown` Rust target and `wasm-bindgen-cli 0.2.126`; the build
fails on tool drift and verifies the guest's 192 MiB maximum-memory declaration.
Archive consumers need neither Rust nor wasm-bindgen.
