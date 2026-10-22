# tritium-train

Ternary QAT and distillation: autograd tape, STE and differentiable estimators, optimizers.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Native scale recovery (CPU reference)

`pv_tuning::NativeScaleRefinementSession` runs a `ScaleOnly`
`NativeRefinementPlan` over every explicitly selected owner, in plan order.
Supply the admitted parent `AdditiveTensor`s and unweighted gradients of their
**actual stored scales**, not dense logical-weight gradients. The session applies
the declared window weight, the shared AdamW implementation, nonnegative
projection, and rounding to each tensor's declared scale precision. It preserves
ternary codes, compact allocation, law, codec, basis and transport; no f16 bridge
or dense/code master is introduced. Every update commits all owners atomically.

`step_from_consumers()` instead accepts ordered `NativeScaleContribution`
Matmul/Embed uses per owner. The shared core computes and reduces stored-scale
adjoints in f64 using the actual basis/allocation/law, then narrows once and calls
the same step. Repeated/tied uses and duplicate embedding indices contribute to
one owner update. An empty owner-use list means zero gradient, not skipped decay.
The consumer path checks its separate logical peak `max(4S+8M+4C,20S)` against
the workspace limit before allocation (total scales S, largest owner scales M,
largest columns C); direct steps retain their existing smaller ledger.

`checkpoint_bytes()` emits the distinct TNSR v1 state format; `resume()` requires
the same frozen plan and exact original parent tensors. Frozen codes and metadata
come from that parent, while the checkpoint retains exact scale bits, moments and
the completed-window cursor. This is not TPV1/TPVB1 and is not an inference file.
Logical size limits exclude caller gradients, graphs/activations, allocator slack
and process RSS. A valid checkpoint is not proof of gradient/objective execution.

The package-byte limit belongs to the future publisher, not this numerical state.
This API does not admit package custody, authenticate graph/objective/data or the
complete consumer inventory, implement native alternating-PV, emit publication
evidence or create a refined child. The separate `tritium.torch.native_scale_training`
adapter supplies eager CPU scale autograd; see `tritium-py/README.md`. Public
native refinement stays unavailable until those integrations and replay exist.

CPU packed scale snapshots can consume the session's updated exact scales while
preserving old forward/backward versions and sharing packed codes. The local
closed-loop fixture combines actual packed Matmul/Embed cotangents, the common
step, immutable snapshots and exact restart. The session still owns decoded
reference state: this does not establish production packed/out-of-core training,
an authenticated model objective or public native refinement.

`step_staged()` computes checked candidates, borrows their exact named scales to
one resource stager, and commits scales/moments/cursor only if staging succeeds.
Staging error or unwind preserves the old common state. Callbacks must keep
resources private; arbitrary external effects are not rolled back or admitted by
the numeric workspace ledger. Direct steps use the same candidate/commit path.

With `--features native-cpu`, `pv_tuning::NativeCpuScaleSession` owns both common
state and the complete named packed handle set. Binding pins current-state scales
over supplied code roots and verifies actual geometry/codes/scale bits. This
supports resume over original roots but does not authenticate original scale
lineage or model/source provenance. Every step privately stages all next handles,
then commits state and replaces the complete set; a late budget failure drops all
new resources and preserves old handles/checkpoints. Actual packed consumer
adjoints share the reference reducer. Budget for initial versions and simultaneous
old/new versions; retained old graph Arcs retain their scale payload admission.
The adapter still owns decoded reference state, not production packed-memory QAT.

## Recorded numerical replay

`pv_tuning::NativeScaleJournalWriter::new(stream, &session, limits)` starts an
optional journal over an empty caller-owned `Write + Seek` stream and a zero-step
session. Use `session.step_recorded(gradients, &mut journal)` for every declared
window. It uses the same optimizer/atomic commit path as `step`: write and flush
the exact admitted unweighted f32 scale gradients before committing state. Numeric
refusal writes nothing; write/flush failure or unwind preserves state and poisons
the writer. An incomplete/poisoned writer cannot finish. Generic flush does not
promise fsync or power-loss durability.

`journal.finish(&session)` returns the stream and exact journal/checkpoint byte
observations. Retain the journal BlobId outside the mutable stream. Then call
`replay_native_scale_journal(plan, original_parents, stream, journal_id,
expected_checkpoint, limits)`. It checks complete input identity before creating
optimizer state, replays every ordered frame through the common optimizer and
requires bit-identical checkpoint bytes, including moments. The returned
`VerifiedNativeScaleReplay` can recheck the current session; it is numerical
reproduction, not proof of the supplied gradients' model/objective/data origin.

`session.journal_size_ledger()` forecasts exact journal bytes and logical
writer/replay buffers without stream I/O. Journal limits separately cap exact
stream bytes and logical working payload.
The writer needs one serialized checkpoint plus fixed 8192-byte I/O scratch;
replay also admits reference state, update workspace and one f32 gradient window.
Streams, metadata, supplied expected-checkpoint bytes, allocator capacity and
external models/graphs are excluded. No whole journal is retained by the module.
With `native-cpu`, `NativeCpuScaleSession::step_recorded()` stages every next
snapshot before writing the journal, then commits common state and handles
together. A snapshot failure leaves the journal unchanged; a journal failure
drops all staged snapshots and preserves old handles/state. Finish/replay uses
the read-only `binding.session()` and the original parents.

This v1 profile does not append/resume partial journals, record Python steps,
publish children, authenticate backward execution or clear any
independent quality/performance/release gate. Existing unrecorded TNSR resume and
all prior numerical/inference profiles are unchanged.
