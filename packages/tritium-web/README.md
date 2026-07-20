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
receipt.

The lifecycle WASM controller and WebGPU adapter are still under construction;
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
