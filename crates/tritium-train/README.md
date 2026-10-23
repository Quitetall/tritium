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
native teacher/data refinement is exposed through `tritium.torch.refine` with
distinct scale-only and alternating-P/V plans; see `tritium-py/README.md`.

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

`journal.checkpoint(&session)` flushes and returns TNSR bytes with a schema-owned
`NativeScaleResumePoint` binding the exact emitted prefix, plan and cursor. Keep
both independently; this observation is neither numerical proof nor fsync.
`resume_native_scale_journal(request, empty_new_output, limits)` checks the prefix,
replays it from the actual original parents and requires an exact checkpoint
match before copying that prefix into the new stream and returning state/writer.
It never appends to, truncates or writes the recovery source. `ExactPrefix`
requires no trailing bytes; explicitly selecting `CheckpointPrefix` branches at
the saved boundary while leaving a later complete/partial tail untouched and
unvalidated. Failed output I/O can leave an unusable caller-owned destination;
the caller owns cleanup and exclusive source custody. Replay/copy use bounded
8192-byte chunks and the existing complete-journal/replay working limits.
The existing TNSJ v1/TNSR bytes are unchanged. Python recorded continuation
remains separate work. These interfaces do not
authenticate backward/objective/data origin or clear independent quality,
performance or release gates. Existing unrecorded TNSR resume is unchanged.

`stage_native_scale_resume_bundle(session, journal, empty_output, limits)` emits
and reopens a self-contained TNSB v1 recovery file: frozen plan, emitted resume
point, complete checkpoint and exact journal prefix. No prefix-sized Vec or
extraction file is retained. `publish_native_scale_resume_bundle` uses the same
stager in a new directory, fsyncs the pending file and commits `resume.tnsb` with
a no-clobber hard link. It cleans only owned precommit files; postcommit cleanup
or directory/parent-sync caveats are warnings. Retain the returned full BlobId
separately and use `NativeScaleResumeBundleReader::open(file, id, limits)` after
restart. Byte consistency is not numerical or quality qualification.

`bundle.resume(original_parents, empty_new_journal, journal_limits)` rechecks
the actual bundle and invokes strict common continuation through a bounded view
of its prefix. Original recording work need not survive, but original parent
tensors are still required. Saving restores the live journal append cursor;
destination-only failures permit retry if restoration succeeds, while source
read/hash/position uncertainty poisons that writer. Bundle limits admit exact
file/serialized manifest/checkpoint bytes and M+C+8192 logical working payload;
session/model, metadata heap, allocator, page cache, caller streams and RSS are
excluded. Generic staging flush is not fsync; no power-loss/security clearance
is inferred from the filesystem publisher.
Each explicit save copies the entire selected prefix; it is not an incremental
or compressed checkpoint. Set a disk ceiling and retention policy deliberately;
this module neither schedules saves nor deletes older recovery bundles.

## Immutable native scale children (Rust)

`pv_tuning::publish_native_scale_child` consumes a completed common session, its
actual `VerifiedNativeScaleReplay`, the original evidence-bound retained parent
reader and actual journal, explicit validation observations, an empty private
spool, declared limits, a new directory sink and a fresh registered recorder.
Register native start/checkpoint/completion plus
`NATIVE_SCALE_REPLAY_OBSERVED_REGISTRATION`. The producer records its compiled
source identity and CPU-reference profile; machine/validation values remain
caller observations, not independently authenticated quality or hardware proof.

The producer rechecks current state and actual original parent coefficients/scales
without retaining a second decoded model. It publishes exactly one schema1.1
Refined level, preserving unselected tensor bytes/storage and every original
asset. Selected tensors keep their layout/law/basis/allocation/codes and actual
stored scale precision. Selected encoding currently requires Raw transport.
`inspect_native_scale_child` forecasts logical spool/companion/builder payload,
not authenticity, complete container bytes or process RSS.

The common publisher writes `model.trit`, `evidence.jsonl`, `checkpoint.tnsr` and
`gradients.tnsj`. Companion lengths/identities and the original parent guard must
pass before the no-clobber model commit marker. Failed private spool tails belong
to the caller; failed publication cleans only owned files. Post-commit cleanup/
directory-sync caveats are explicit warnings. Training journals/checkpoints are
not deployment compression evidence and may require substantial disk space.

`NativeScaleChildReceiptView` folds original records against an admitted parent
and requested plan and verifies actual typed child bytes. It is record consistency,
not numerical replay or a quality/release capability. An unchanged child may keep
its semantic ModelId; package/provenance remain distinct. Standalone typed reopen
does not yet provide Python/HF shell loading: preserved catalogs still describe
the explicit parent level and need the separate refined-result loader. No old
completed-conversion gate is broadened. Python recording/public refine/export/load
and native alternating-P/V integration are available separately; objective/data
authentication and independent physical/model/security/release gates remain open.

## Native alternating P/V recovery (CPU reference)

`pv_tuning::NativePvRefinementSession` consumes an actual `AlternatingPv` plan,
separate stored-scale and present stored-weight adjoints, and immutable original
parents. It shares AdamW and the native code projector; it is not scale-only
training relabeled as P/V. All owners, moments, hard codes and the cursor commit
atomically. Recorded journals and durable recovery bundles preserve exact replay;
the optional `NativeCpuPvSession` stages every next packed snapshot before journal
append/flush and commit. See `tritium-py/README.md` for eager teacher/data training
and immutable child publication through `tritium.torch.refine`.

The code proposal reuses one transient present-weight buffer, maps window-weighted
gradients inside the shared optimizer, and decodes one row again for movement
ranking. A bounded heap retains exactly `K=ceil(fraction*U)` units; projection
still visits them in descending movement order with logical-index ties. No
coordinate moments are skipped, and no dense weight master persists. Logical
per-owner update workspace is `2T+I+20S+12A+sizeof(rank)*K+4C`: decoded coefficient
bytes T, auxiliary bytes I, stored scale count S, present weight count A,
code-unit count U and column count C. The session sums this conservative bound
over all owners; consumer and journal limits derive from the same ledger.
Persistent state remains `T+I+12S+8A`. Checkpoint/journal formats and recipes are
unchanged. These are payload bounds, excluding metadata, allocator slack, caller
gradients, external graphs and separately admitted packed versions—not process
RSS, VRAM, production GPU/paged fitting or model-quality qualification.

## Owner-paged native P/V state

`pv_tuning::NativePvPagedSession` initializes or strictly reopens the same TNPV v1
checkpoint from an ordered original-parent provider and a caller-owned seekable
stream. It retains no decoded tensor owner or optimizer payload between calls.
`step_staged` loads one owner, obtains separate unweighted scale/present-weight
adjoints, calls the same numerical update as the resident engine, writes one
owner into a new empty private checkpoint, and stages an optional owned runtime
resource. It returns the complete next session, observations and resources only
after all owners, original-parent identity, traversed old bytes, output identity
and flush pass. The old session/cursor remains unchanged, even on late failure
or unwind; private staged resources drop on refusal. Arbitrary callback effects
and partial destination files remain the caller's responsibility.

The original plan's aggregate logical limits still apply. Separately declare
`NativePvPagedLimits.max_resident_payload_bytes`; the conservative worst-owner
payload is state+update workspace+4*(stored scales+present weights)+16384 I/O
bytes. Returned adjoints are included; provider internals, staged runtime
resources, metadata, allocator slack, OS page cache and process RSS are not.
Underlying stream buffers are also excluded; use file-backed streams for
out-of-core behavior, not `Cursor<Vec<u8>>` containing the complete checkpoint.
Checkpoint I/O requests are at most8192bytes. The resident encoder uses the same
codec with four-byte chunks, preserving its existing small scratch envelope.

Owner paging is not within-owner paging: the largest owner and its moments,
gradients and candidates must still fit. Total on-disk state is not reduced:
TNPV stores decoded trits and f32 moments, not a compact inference package. An
update needs both complete old and new checkpoints, in addition to parent
artifacts and any journal. Set a disk ceiling and delete only obsolete owned
snapshots after their replacement is safely retained. Generic flush does not
provide fsync durability or immutable filesystem publication. Paged journal
recording, first-party source/runtime/Python wiring, within-owner moment paging,
GPU execution and independent model/physical/release qualification remain open.
