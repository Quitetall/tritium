# tritium-quantize

SALT quantization: additive ternary planes, exact joint fitting, curvature-guided allocation.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

`engine::DirectoryTensorSink::publish_spooled_model_with_companions` stages a
typed `TritModelSpool`, a sealed original log and bounded `TritModelCompanions`
through the same no-clobber transaction as ordinary models. Companion streams
have explicit portable leaf names, lengths and BlobIds; no companion-sized buffer
is retained. Exact container/manifest/directory caps are checked through the
owning encoder's layout inspector before creating publication files.

The admission callback sees the private typed reader and returns owned metadata
before terminal verification and commit. Retained companion handles are hashed
again after that callback. Source errors or pre-commit failure remove only owned
files; unexpected files are preserved. Generic companion bytes are not interpreted
as training checkpoints by this module. Source/directory custody, power-loss,
security, physical memory, model lineage and distribution approval remain external.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

The candidate typed `engine::Engine::run` fits original raw sources into direct
quality levels, seals the fitting log, verifies package/log binding and publishes
`.trit` through a no-overwrite sink. Its Rust-owned `DirectFitProfile` selects the
shipping tied-F16 activation-aware ladder, independent free F16/F32 joint groups,
or `JointFeedbackGroups` with full-Gram sequential residual propagation.

Joint feedback has explicit `WorkPrecision::{F32Accum64,F64}`, damping and
decay/ramp. F64 controls feedback arithmetic, not an f64 joint solver: the joint
fitter still accepts f32 inputs and rejects narrowing overflow. Working-state
buffers remain f64; this is not a memory reduction or full-RSS admission claim.
Rust model-plan builders use `QuantAction::Fit { plan: Box::new(plan), metric }`;
boxing does not alter the derived JSON representation. These candidate profiles
have software tests, not official Qwen quality, speed or SOTA qualification.
