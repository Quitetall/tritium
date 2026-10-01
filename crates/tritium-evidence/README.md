# tritium-evidence

Typed, registered evidence events with per-span hash chains and deterministic
run roots. Diagnostics are not evidence unless emitted through this crate.

The log format and digest domains are governed by ADR 0044. This crate is the
initial P1 core; CLI integration, tracing macros, environment capture, timing
sidecars, and cross-surface bindings are separate follow-up work.
