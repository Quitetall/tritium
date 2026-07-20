# `@tritium-ai/web`

Tritium's compiled browser-training package. The current release-candidate
contains the frozen language-neutral training identities and the checked
`WebTrainingSession` lifecycle. Session calls are serialized and fail closed on
schema, backend-policy, capability, receipt, memory, ordering and concurrency
violations. Failed adapter calls do not advance session state.

The generated WASM and WebGPU adapters are still under construction. Until one
is supplied, `prepareTraining(model, config)` returns a typed
`adapter_unavailable` error; this package never labels a JavaScript fallback as
WebGPU execution.

This package is private while the local v1.1 release candidate is under
construction. Registry publication requires explicit release authorization.
