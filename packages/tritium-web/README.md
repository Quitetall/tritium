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
adapter calls do not advance session state.

The generated WASM and WebGPU adapters are still under construction. Until one
is supplied, `prepareTraining(model, config)` returns a typed
`adapter_unavailable` error; this package never labels a JavaScript fallback as
WebGPU execution.

This package is private while the local v1.1 release candidate is under
construction. Registry publication requires explicit release authorization.
