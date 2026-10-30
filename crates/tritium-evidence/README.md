# tritium-evidence

Typed, registered evidence events with per-span hash chains and deterministic
run roots. Diagnostics are not evidence unless emitted through this crate.

The log format and digest domains are governed by ADR 0044. This crate is the
partial P1 core. CLI integrity inspection exists; tracing macros, bounded/streaming
recording, environment capture, timing sidecars and complete cross-surface bindings
remain follow-up work.

Experimental text loading registers both `TENSOR_UPLOADED_REGISTRATION` and
`MODEL_LOADED_REGISTRATION`. A successful shared load returns an immutable
`ModelLoadReceiptView` of the recorded envelope, not another stored receipt file
or release verdict. Typed packages bind candidate package/model/level identities;
HF identity is explicitly unavailable. `ModelLoadReceiptView::fold` rechecks log
chains/root and strict load-event structure. Generic CLI integrity is not a full
event registry or independent qualification. Details and exact checks:
[`trit-model-load-evidence`](../../docs/research/trit-model-load-evidence-2026-10-10.md).
