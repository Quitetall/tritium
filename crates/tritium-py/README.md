# tritium-py

PyO3 bindings for the pytritium Python package.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

The expert `tritium.convert_model_plan(plan_json, source_shards, output_dir,
run_id, *, asset_paths=None, input_grams=None, calibration_caches=None,
calibrated=False, max_plan_bytes=16777216)` runs the same Rust model conversion
engine as `tritium convert --plan`. Supply the existing Rust `QuantPlan` JSON,
the complete original safetensors shard inventory and exact named asset paths.
`calibrated=True` selects `CalibratedQuantPlan` and requires its bound `.trit`
cache files. `input_grams` binds named raw little-endian f64 square Gram files;
neither absent nor invalid metrics fall back to Identity. Computation releases
the GIL. The destination must be new and is never overwritten.

The readonly `NativeModelConversion` reports actual source/package/model/evidence
identities and publication warnings. `verify()` reopens `model.trit` and
`evidence.jsonl` and verifies their exact identity and binding.
`preserved_tensor(level, name, max_bytes=67108864)` returns precision, shape and
original dense bytes after verification; it does not decode additive tensors.
Revalidation reads whole files under declared bounds, not a streaming/RSS claim.
This converts the complete declared floating-point inventory, not arbitrary HF
architectures, and is not a model-quality or release-qualification receipt.
The existing torch PTQ lifecycle and its matrix-only packages are unchanged.

`tritium.torch.convert_native_module(prepared, output_dir, run_id, *, actions,
levels, limits, work_dir, asset_paths, architecture="torch-module",
input_grams=None)` is an opt-in persistent-state adapter to that engine. Supply
the existing Rust Fit actions for every selected coverage owner and the original
config/tokenizer assets required by QuantPlan. Its temporary source files live
under an existing caller work directory and are cleaned up after conversion.
Post-publication cleanup faults remain explicit warnings on the successful result.

The state catalog preserves exact aliases, biases, norms and buffers. Admitted
positive-shaped F16/BF16/F32 owners use native Fit/KeepDense. Scalar, empty,
float64, integer and boolean state is preserved as a safetensors asset, not
silently admitted as native executable tensors. The current state dtype profile
is F16/BF16/F32/F64, signed I8/I16/I32/I64, U8 and bool; other dtypes, custom
non-tensor state, meta/sparse/quantized tensors and overlapping views reject.
`result.asset(name, max_bytes=67108864)` reads bounded original asset bytes after
package/log verification. This is source packaging, not calibrated phase
migration or runtime/quality qualification.

`tritium.torch.load_native_module(shell, artifact, level="full",
load_run_id="load-1", inplace=False)` applies the version-2 state catalog to a
compatible fresh CPU Torch/HF shell. `artifact` is the returned native conversion
or its output directory, reopened through `NativeModelConversion.open(path)`.
The shared Rust reader verifies package/fit-log binding; the shared CPU backend
executes immutable packed Linear/Embedding owners without retaining a selected
dense weight. Original biases, norms and persistent buffers are restored with
exact aliases. F32 scale laws and Identity/Hadamard/SignedRht bases are preserved.
The loader checks state topology, selected consumer options and normalized HF
configuration before privately staging any changes. The version-1 preservation
catalog is not executable and is never silently upgraded.

The default adapter is eager CPU inference only. Explicit `input_gradients=True`
admits input-only eager autograd without training packed weights. The separate
scale-training path below does not change that default. Accelerator placement,
`torch.compile`, ordinary `state_dict`/`save_pretrained` export and arbitrary model
architectures are not supported or inferred. Save/reopen the native package, not
an incomplete framework checkpoint. Root dtype moves keep packed scale precision
unchanged and restore preserved aliases; descendant-only moves do not carry that
whole-model topology guarantee. Payload/workspace/output limits bound declared
native allocations, not aggregate RSS: the initial receiver retains one admitted
whole package byte image. Its sealed upload observations are not a model-quality,
physical-memory, performance or full-release qualification receipt.

### Explicit native scale training

`tritium.torch.native_scale_training` admits first-order CPU eager scale updates
through actual packed Linear/Embedding backward kernels and the common Rust
atomic optimizer. Each run selects **one explicit parent quality level**. Obtain
its exact parent tuple with `artifact.native_refinement_parent(level)` and supply
a complete Rust `NativeRefinementPlan` mapping or JSON: selected canonical owners,
`ScaleOnly` recipe, declared objective/teacher identities, ordered training and
validation windows, child level, and resource limits. The artifact must be an
actual `NativeModelConversion` or distinct `NativeModelRefinement`; the factory
revalidates the immediate parent's package/evidence and required companions.

```python
from tritium.torch import native_scale_training

trainer = native_scale_training(
    fresh_shell, artifact, level="full", plan=refinement_plan,
    load_run_id="scale-load-1", max_snapshot_bytes=snapshot_limit,
    max_parent_bytes=parent_limit, max_saved_input_bytes=saved_input_limit,
)
trainer.zero_grad()
loss = objective(trainer.model, training_window)
loss.backward()
trainer.step(refinement_plan["training"][0]["content"])
checkpoint = trainer.checkpoint()
unpublished_model = trainer.close()
```

The private shell leaves caller state untouched. Tied consumers share one f64
gradient-transport leaf; Rust owns deployed scales, moments and atomic updates.
Do not use `torch.optim` on these leaves. Pass `checkpoint=...` to resume the exact
same plan; changing parent level or plan rejects. `trainer.plan` is detached
metadata, not a way to change the frozen update schedule or owner order.

F16/BF16/F32 activations are supported; higher-order scale gradients, autocast,
concurrent backward/step, compile/vmap and accelerator/distributed training are
not admitted. Stale graphs and externally modified preserved state/owner leaves
reject. A failed backward requires `zero_grad()` before reuse. Each update needs
the next declared window identity and at least one owner gradient; unused owners
receive zero gradients. A rejected update retains Rust state and gradients.

Limits cover declared native state, transfer/workspace, snapshots and conservative
graph-lifetime saved inputs, not allocator capacity or total Torch/process RSS.
The common optimizer still retains decoded ternary reference parents; this is not
production packed/out-of-core training memory. Callback/data identities are
declarations, not authenticated replay or quality evidence. Closing exposes the
current private inference shell marked unpublished. It does **not** publish a
complete immutable child package, modify the parent, refine every quality level,
or run native public `refine` orchestration. The opt-in recorded publication path
below is separate. Code/PV updates and independent quality/performance/release
gates remain separate work.

To publish a child, configure `recording={"work_dir": existing_work_dir,
"max_journal_bytes": journal_cap, "max_working_bytes": recording_cap}` when
creating the trainer. A fresh run requires zero origin; recorded recovery uses
the explicit bundle path below, never in-place journal appending. Unrecorded
TNSR checkpoint/resume stays unchanged. Rust admits
exact stream/work bytes and writes the actual f32 gradients before atomic state
commit. Journal I/O failure never silently becomes an unrecorded update.

After all declared steps, call `trainer.publish(output_dir, run_id, machine=...,
validation=[...], limits=...)`. Validation entries use the existing Rust content,
parent_loss and child_loss fields. The exact positive limit keys are:
`max_spool_bytes`, `max_working_bytes`, `max_evidence_bytes`,
`max_companion_bytes`, `max_package_bytes`, `max_manifest_bytes`,
`max_directory_bytes`, `max_parent_bytes`, `max_replay_working_bytes`,
`max_journal_bytes`. Replay/reference buffers, current trainer state and Torch
graphs have separate scope; these ceilings are not total RSS. The stream can be
large: use its forecast/explicit limits, not deployment compression estimates.

The publisher reopens the actual original parent, replays the complete journal,
and uses the shared immutable writer. It returns `tritium.NativeModelRefinement`,
not a `NativeModelConversion`. The private shell stays unpublished. Load the
frozen result with `tritium.torch.load(result, model=fresh_shell,
level=child_level, load_run_id=..., inplace=False)`; exact native directory loading
and `export(result, new_directory)` also preserve this distinct result type.
`trainer.close()` closes/removes only its owned temporary recording work; the
published checkpoint/journal remain in the child directory.

### Recorded save and restart

```python
from tritium import NativeScaleResumeBundle

bundle_limits = dict(max_bundle_bytes=bundle_cap, max_manifest_bytes=metadata_cap,
                     max_checkpoint_bytes=checkpoint_cap, max_working_bytes=working_cap)
saved = trainer.save_resume(new_saved_directory, limits=bundle_limits)
saved_id = saved.bundle_id  # Retain this independently of the mutable file.
trainer.close()  # Its unique temporary journal work can now disappear.

saved = NativeScaleResumeBundle.open(new_saved_directory, bundle_id=saved_id, **bundle_limits)
trainer = native_scale_training(
    fresh_shell, original_parent, level="full", resume=saved,  # Plan is derived from the bundle.
    load_run_id="recovered-load", max_snapshot_bytes=snapshot_limit,
    max_parent_bytes=parent_limit, max_saved_input_bytes=saved_input_limit,
    recording=dict(work_dir=existing_work_dir, max_journal_bytes=journal_cap,
                   max_working_bytes=replay_cap),
)
```

The frozen descriptor exposes `completed_updates`, `plan_json`, `bundle_bytes`,
`warnings`, `verify()` and `admission="byte-consistent-unqualified"`. It is not an
inference artifact or numerical/quality approval. Rust reopens actual saved bytes,
checks the exact parent/request, decodes parents once, replays the saved history
and requires a byte-identical checkpoint before continuing in a new owned journal.
An explicit plan must match; checkpoint and bundle resume are mutually exclusive.
Only completed Rust scale/moment/cursor/history state is saved. Pending gradients,
autograd graphs, RNG, data-loader state and arbitrary nonpersistent Python buffers
are not saved; recompute pending windows and supply identical remaining inputs
for exact continuation. Objective/data/teacher authentication remains separate.

Save outside both the original parent and the trainer's owned temporary work.
New resume work cannot be under the parent or saved bundle. Bundle close/destruction
does not delete its durable directory; trainer close deletes only its own journal
work. Failed restoration never silently downgrades to unrecorded training. After
completion, the unchanged `publish` path replays the whole actual journal and
publishes an unqualified immutable child. Sealed/closed/failed writers cannot save.
Every explicit save copies the whole prefix under the supplied disk/working byte
ceilings. No automatic cadence, pruning, compressed storage, total RSS guarantee,
crash/security qualification or arbitrary-PyTorch checkpoint support is inferred.
The bundle checkpoint is included in decoded-parent working admission on resume;
Torch/packed handles, replay state and caller-owned buffers retain separate scopes.

### Public native refinement

`tritium.torch.refine(native_parent, teacher=original_dense_cpu_model, ..., config=...)`
uses a distinct `NativeRefinementConfig`. It never translates the native package
through legacy plane dictionaries or another optimizer. Specify the exact parent
level/native plan, new output and existing external work root. The source teacher's
persistent-state digest must match the immutable native Torch catalog; declare
`plan["teacher_source"] = plan["parent"]["source"]`. Official source authentication,
arbitrary Python forward-code attestation and independent quality remain separate.

```python
from tritium.torch import NativeRefinementConfig, refine

plan["objective"] = NativeRefinementConfig.objective_id(temperature=1.0)
for request, batch in zip(plan["training"], training_batches):
    request["content"] = NativeRefinementConfig.window_id(batch, limits=data_limits)
for request, batch in zip(plan["validation"], validation_batches):
    request["content"] = NativeRefinementConfig.window_id(batch, limits=data_limits)
config = NativeRefinementConfig(
    plan, level="full", run_id="native-kd-1", machine="local-cpu",
    training_limits=training_limits, publication_limits=publication_limits,
    data_limits=data_limits,
)
child = refine(parent, teacher=original_dense_cpu_model,
               training=iter(training_batches), validation=iter(validation_batches),
               config=config, work_dir=existing_work_root, output_dir=new_child_directory)
```

Exact positive `training_limits` keys: `max_snapshot_bytes`, `max_parent_bytes`,
`max_saved_input_bytes`, `max_package_bytes`, `max_payload_bytes`,
`max_workspace_bytes`, `max_output_bytes`, `max_evidence_bytes`. Publication keys
are the ten recorded-publication keys above. Data keys: `max_window_bytes`,
`max_validation_bytes`, `max_nodes`, `max_depth` (at most64), `max_loss_bytes`.
They bound logical data/copies and a 32-byte-per-logit loss-work forecast, not
framework/autograd/metadata heap, allocator, page cache or total process memory.

The built-in versioned objective uses f32 log-target KL with positive temperature
for multiple logits, scalar/one-logit MSE otherwise. Matching binary attention
masks and integer labels!=-100 exclude padded/ignored positions; empty support,
nonfinite output or mismatched geometry refuses. Window IDs use domain-separated,
typed length-framed hashing with bounded raw tensor traversal. Only ordinary CPU
tensor/scalar/container values are admitted. A protected copy of one training
window is held at a time; only bounded validation windows are retained. Streams
must match the full frozen schedule exactly; no cycling, missing or trailing data.

The actual terminal candidate must not regress against the original parent on
the same weighted validation snapshots before publication. No silent earlier-step
fallback rewrites the plan. Teacher modes are restored and original artifacts are
never overwritten; mutation checks refuse changes rather than repairing arbitrary
user code. `resume=saved_bundle` replays the exact plan and skips old training
computation while checking the complete supplied stream. Original-parent inference
provides the predecessor baseline on resume, not a throwaway optimizer.

Optional `checkpoint_steps=(...)` with explicit `checkpoint_limits` enables only
those cursors. Supply a durable existing `checkpoint_dir` outside work/parent/output
and optionally `on_checkpoint(saved_descriptor)`. Each new step directory is
no-clobber and copies the whole prefix; callback failure leaves that saved bundle
but no child. Older checkpoints are never pruned automatically. Postcommit trainer
cleanup caveats are operational `publication_warnings`, not altered evidence or a
false failed commit. Return type remains the unqualified `NativeModelRefinement`;
legacy `RefinementConfig` and its artifact/profile gates stay unchanged. Native
code/PV, objective/data attestation and accelerator/
distributed/physical/quality/release qualification remain open.

### Refining an existing native child

A `NativeModelRefinement` can supply actual hard starting values to another run
without being relabeled a PTQ conversion. Obtain its exact immediate tuple with
`child.native_refinement_parent(selected_level)`, build a new explicit plan, and
pass it to `native_scale_training` or public `refine`. The same level label may
appear in distinct packages; exact PackageId/model/source/evidence bindings, not
labels alone, identify the parent. New rounds start at cursor0 with fresh moments;
the previous TNSR is not silently imported as the new optimizer state. Use a saved
bundle only to resume the same new-round plan and immediate parent.

The parent descriptor admits actual model/strict log/checkpoint/journal bytes.
Source-catalog assets and original-source observation are inherited unchanged;
KD still uses the original dense teacher, while observed retention compares the
actual immediate parent's values. Old models need not remain merely to infer or
start another round, but audits of their earlier edges require those actual
parents separately. Each published child remains complete, immutable and distinct.

`child.verify_with_parent(immediate_parent, ...)` accepts either descriptor kind
and verifies that immediate numerical edge. Its JSON scope remains
`parent-backed-numerical-replay` and now explicitly reports
`lineage_scope="immediate-parent-only"` and the parent's unqualified admission kind.
It does not verify the entire chain or upgrade quality/objective flags. A new
successful edge never retroactively authenticates older history. Existing
completed-PTQ admission still rejects refined packages; standalone loading/export
preserve their distinct type, original logs and companions.

`NativeModelRefinement.open(directory)` independently checks actual typed child,
original log and exact checkpoint/journal identities without requiring the parent
for inference. Its admission is `record-consistent-unqualified`; it does not prove
original parent state, backward/objective origin or quality. The v2 catalog stays
unchanged and loads against the child's actual level/storage. `verify()` repeats
that same scope. `verify_with_parent(parent_conversion, max_parent_bytes=...,
max_replay_working_bytes=...)` separately checks original ancestry, actual common
numerical replay and encoded deployed owner bytes, not only checkpoint IDs. Even
that observation is not objective authentication or release qualification.

The opt-in calibrated lifecycle is separate from the legacy diagonal `calibrate`:
`prepare_native_module(prepared, actions=..., levels=..., limits=..., work_dir=...,
asset_paths=...)` observes the complete original source without fitting.
`calibrate_native_module(source, windows, objective, objective_bytes=...,
cache_limits=..., max_total_payload_bytes=...)` runs each actual objective once
and captures CPU Linear inputs plus loss gradients through `autograd.grad`,
without accumulating parameter gradients. Each window supplies integer
`input_ids` matching the flattened projection samples. The objective callable
returns a finite scalar; declared recipe bytes do not independently prove its
execution. Use `output="None"` explicitly for input-Gram-only capture.
Optional `calibration_weights` (finite nonnegative CPU floating values) and
`calibration_mask` (CPU bool values) must have exactly the input_ids shape.
These affect statistical reduction, not the objective loss. No broadcasting or
implicit use of attention_mask occurs. Weights are widened to native f64;
masked activations/gradients/weights must still be finite. Missing selection
preserves the unit-weight/true-mask cache bytes. Preparation binds relative
work/asset paths immediately, so later phases do not depend on the current
working directory; this is not immutable source custody.
Token and selection context is snapshotted before the objective; visible
changes during it reject rather than publish stale context. This does not prove
that arbitrary callback code used those tokens or freeze concurrent writers.

`convert_calibrated_native_module(calibration, output_dir, run_id, partition="Full",
damping=0.01, max_numeric_bytes=..., max_cache_package_bytes=...)` uses the shared
native calibrated plan, reducers, row-specific metrics and sealed lifecycle.
`partition={"Groups": {"columns": width}}` explicitly selects a block-diagonal
approximation. Input-only capture requires zero damping. The source is re-observed
at each phase; source/asset/cache changes fail rather than fall back to Identity.
The preparation JSON ceiling is 1 MiB. Existing direct plans, stored scale laws
and native loader semantics are unchanged.

`calibration.save(directory, max_package_bytes=...)` atomically publishes existing
`.trit` cache packages with their Rust-owned binding metadata. Reopen with
`NativeModuleCalibration.open(source, directory, max_package_bytes=...,
max_total_payload_bytes=...)`. Logical cache/numerical budgets do not cover the
framework tape, source copies or full RSS. Remaining aggregate cache payload
allowance is enforced by the shared decoder before decoded-payload copies.
This K-FAC capture adapter requires Linear-only consumers; mixed tied
embedding/head roles fail explicitly. The distinct owner-window path below
handles those roles without changing this estimator's meaning.
Source observation does not freeze concurrent writers,
and callback mutations are detected, not rolled back. No empirical gate is
cleared by this software lifecycle.

For shared embedding/head weights, call `source.capture_owner_window(window,
objective, objective_bytes=..., max_observation_bytes=..., max_context_bytes=...,
max_invocations=..., max_samples=..., max_graph_nodes=..., cache_limits=...,
max_total_cache_bytes=...)`. Supplying explicit `OwnerScoreCacheLimits` fields
produces immutable native caches of the actual scalar objective's full owner
gradients, including repeated gather/matmul uses. These are scalar-window score
observations, not K-FAC/Fisher estimates or qualified probe recipes.

`convert_owner_native_module(source, captures, output_dir, run_id,
window_weights=[...], damping=..., metric_limits=..., max_row_bindings=...,
max_total_cache_bytes=..., max_cache_package_bytes=...)` fits those windows
through the shared Rust owner-model engine. `captures` and weights must be
bounded lists/tuples in matching window order. `metric_limits` has positive
`max_windows`, `max_payload_bytes`, `max_row_bytes` and `max_reduction_entries`.
Every capture must cover all selected source owners; repeated captures remain
intentional observations. Distinct numerical payloads are exported once.
Optional `max_plan_bytes` defaults to 16 MiB for the complete owner plan.

The result uses the ordinary `load_native_module` and package-reopen interfaces.
Tied embedding/head handles remain tied; selected dense masters are not retained
by the loaded artifact. Temporary sources/caches are cleaned after success or
failure, with explicit warnings for post-publication cleanup errors. Capture,
source snapshots and cache encoding/decoding still overlap in memory. Full RSS,
GPU/out-of-core fitting, immutable recipe custody and qualified profile migration
remain separate work.

The public phase functions also admit this route explicitly:

```python
from tritium.torch import OwnerCalibrationRecipe, prepare, calibrate, convert, load

source = prepare(model, selection_config, inplace=False, native_options=native_options)
recipe = OwnerCalibrationRecipe(
    objective_bytes=b"my declared scalar objective",
    window_weights=[1.0, 1.0],
    capture_options=capture_options,
    fit_options=fit_options,
)
calibration = calibrate(source, [first_window, second_window],
                        objective=objective, owner_recipe=recipe)
result = convert(source, calibration, output_dir=output_dir, run_id="my-owner-fit")
loaded = load(result, model=fresh_model_shell, level="full",
              load_run_id="my-load", inplace=False)
```

`native_options` are the explicit `prepare_native_module` arguments above.
TernaryConfig selects coverage; the native actions and named levels select the
actual fit. `target_bpw` is not admitted on this explicit-limit route.
`capture_options` contain the five positive capture limits, `cache_limits` and
`max_total_cache_bytes` listed above, but not objective bytes. `fit_options` contain
the owner conversion options except `window_weights`. Recipe dictionaries are
frozen by value. Windows must be bounded lists/tuples in weight order, not arbitrary
iterators. The returned NativeOwnerCalibration is distinct from old calibration
receipts; its individual captures use the existing native save/open interface.
`inspect(source)` verifies source bytes before returning coverage. Legacy arguments
cannot silently enter owner conversion, and owner options cannot enter legacy
conversion. Qualified profile migration remains open; this explicit research
route does not certify a deployment profile.

`quantize` composes the same owner phases for a live module:

```python
from tritium.torch import quantize, export, load

result = quantize(
    model, selection_config, inplace=False, native_options=native_options,
    data=[first_window, second_window], objective=objective, owner_recipe=recipe,
    output_dir=output_dir, run_id="my-owner-fit",
)
portable = export(result, export_dir,
                  max_package_bytes=package_ceiling, max_evidence_bytes=log_ceiling)
reopened = load(portable.output_dir, format="trit",
                max_package_bytes=package_ceiling, max_evidence_bytes=log_ceiling)
loaded = load(reopened, model=fresh_model_shell, level="full",
              load_run_id="portable-load", inplace=False)
```

Export copies the original `model.trit` and canonical `evidence.jsonl` bytes
through the shared native verifier and no-clobber publisher. Source, package,
level/model, evidence and run identities stay unchanged. Unrelated source-folder
files are not exported. Existing destinations are never replaced; successful
publication with cleanup/sync trouble returns explicit warnings. The native
result also exposes `.export(...)` without importing the Torch frontend.

Optional export ceilings can tighten, not widen, the original result's bounds.
`load(path, format="trit")` explicitly admits a native conversion directory;
unknown formats and native limits on a legacy route fail closed. Converted
artifact verification does not promote imported/provisional models to completed
conversions. Export currently retains bounded whole-package/log buffers in Rust;
it does not claim streaming or complete-process RSS admission, immutable writer
custody, distribution authorization, model quality or release qualification.

Expert filesystem callers can use `convert_owner_model_plan` on the native
extension, or `tritium convert/quantize --owner-plan PLAN --source-shard SOURCE
--owner-cache TENSOR=LOWER_HEX_CACHE_ID=PATH --conversion-run-id RUN --out OUTPUT`.
Repeat source/cache/asset bindings for the exact plan inventory. Mixed owner
plans may also supply legacy `--calibration-cache TENSOR=PATH` bindings. No
missing cache silently falls back to a plain fit.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

Stage-7 campaign executors consume frozen token evidence through
`tritium.torch.Stage7CausalData.open`. It strictly binds the campaign pack and
model-tokenizer identities, reads only the selected sequence window, performs a
terminal same-handle payload check, and yields replayable PyTorch causal-LM
batches. A data receipt is input provenance only, not an execution or quality
result.

`tritium.torch.run_stage7_smoke_model(model, data, output_dir)` consumes that
validated data and transactionally executes or resumes the five concrete smoke
stages: capture, fit, allocate, package, and evaluate. Its result binds source,
tokens, conversion, allocation, package bytes, and measured causal loss. It is
an engineering receipt, not a release-qualified claim.

`tritium.torch.run_stage7_smollm2_smoke(campaign_path, model_dir, output_dir)`
is the frozen campaign wrapper. It admits only the pinned SmolLM2-135M revision,
exact tokenizer/files, source-derived rank-2 inventory, and first 128 C4 members
from the governed evidence pack, then emits the schemas consumed by the Stage-7
qualifier. The wrapper uses local files only and fails closed on unknown resume
artifacts. G64-shaped matrices are stored using explicit SALT V2 package-version
2 scale geometry; G128-only packages retain byte-identical version 1 encoding.
