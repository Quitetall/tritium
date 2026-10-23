# Training (QAT / STE)

`tritium-train` provides training for ternary BitNet models:
straight-through-estimator (STE) autograd, quantization-aware training (QAT),
backward kernels for the ternary ops, an optimizer with bit-exact save/restore,
and LoRA adapters on a frozen ternary base. It is specified in
ADR 0007 (see the [research-record access policy](./research-records.md)). Distributed training — FSDP /
ZeRO-3 (via `FlatShardPlan`), a `ProcessGroup` (with `SimProcessGroup` and NCCL
backends), and distributed checkpointing (`DistCheckpoint`) — shipped in
v0.60 (see the [research-record access policy](./research-records.md)) and is re-exported from
`tritium-train`; the NCCL transport lives in `tritium-cuda`.

## Straight-through estimator

Ternary quantization (`{-1, 0, +1}`) has a zero gradient almost everywhere, so
you cannot backprop through it directly. The **straight-through estimator**
passes the upstream gradient through the quantizer as if it were the identity
(within a clipping range), letting the full-precision shadow weights learn while
the forward pass uses the quantized weights. This is the core of QAT: train *with*
the quantization in the loop so the model adapts to it.

Every trainable op has a backward kernel gradient-checked against a
finite-difference numerical gradient within tolerance — that is the load-bearing
training gate (CPU-only, so it runs on every push). The backward path reuses the
forward kernels validated upstream by the inference spine, so the math is graded
once and shared.

## Optimizer + checkpoint

`tritium-train` ships an AdamW optimizer behind a minimal `Optimizer` trait, with
**bit-exact** save/restore of optimizer state: a resumed run equals the
uninterrupted run, and the checkpoint round-trip is a golden test. The
no-NaN/Inf-over-≥1k-steps and same-seed⇒same-loss-curve properties are the
mixed-precision conformance gate.

## Owner-paged native CPU P/V training

`tritium.torch.native_pv_paged_training` is a distinct first-order eager controller
over an actual `NativeModelConversion` or `NativeModelRefinement` parent. Select
one explicit parent level and a complete `NativeRefinementPlan`; obtain its exact
parent tuple with `parent.native_refinement_parent(level)`. It shares the packed
CPU kernels, optimizer, TNPJ journal, TNPB resume transport and immutable child
publisher with the resident path. It does not retain all decoded optimizer
owners or materialize checkpoint bytes. The ordinary resident factories and
high-level teacher/data `refine` facade are unchanged.

Small-fixture example (these caps are **not** Qwen sizing):

```python
from tritium import NativePvPagedTrainingLimits
from tritium.torch import native_pv_paged_training

caps = NativePvPagedTrainingLimits(
    max_owner_read_bytes=1_000_000,
    max_owner_resident_bytes=1_000_000,
    max_gradient_working_bytes=1_000_000,
    max_workspace_bytes=16_000_000,
    max_journal_bytes=2_000_000,
    max_frame_bytes=2_000_000,
    max_recording_working_bytes=16_384,
)
trainer = native_pv_paged_training(
    shell, parent, level="full", plan=plan, load_run_id="paged-load",
    workspace_dir="/path/to/new/private-work", limits=caps,
    max_snapshot_bytes=1_000_000, max_saved_input_bytes=10_000_000,
)
trainer.zero_grad()
loss = objective(trainer.model, batch)
loss.backward()
trainer.step(plan["training"][0]["content"])
print(trainer.workspace_report)  # logical reservations, not physical memory
```

The workspace must be new and outside parent, resume and output artifacts.
Anonymous current/next checkpoints, gradient frame, journal, replay and publication
spools share aggregate byte admission and at most six live files. Initial and
replayed checkpoints reserve the plan's checkpoint ceiling; subsequent updates
reserve exact checkpoint lengths. Journal and publication spools reserve their
declared ceilings. Allocation may refuse even when actual written files are
smaller. Budget real filesystem free space and I/O separately.

Use `save_resume(output_dir, limits=...)` with the same four positive bundle
fields as resident recovery (`max_bundle_bytes`, `max_manifest_bytes`,
`max_checkpoint_bytes`, `max_working_bytes`). The result is a
`NativePvPagedResumeBundle`, not a trained model or replay/quality approval.
Pass it as `resume=` to a fresh paged factory/workspace, omitting `plan` to use
its frozen request. Actual original-parent numerical replay precedes continuation;
pending gradients, Torch RNG and the external data iterator are not saved.
`checkpoint()` intentionally refuses rather than allocating a full byte buffer.

After completing every declared window, `publish` numerically replays and emits
an ordinary unqualified `NativeModelRefinement`. Its `limits` mapping has exactly
`max_spool_bytes`, `max_working_bytes`, `max_evidence_bytes`,
`max_companion_bytes`, `max_package_bytes`, `max_manifest_bytes` and
`max_directory_bytes`. Source-owner and replay/journal caps remain the frozen
controller limits. Publication working bytes cover one encoded owner plus fixed
publisher I/O, not a checkpoint buffer. Reload the child for frozen inference.

Call `close()` when finished. It preserves complete current packed weights on
the private shell, drops file leases and removes only the same empty owned
workspace. Unexpected files/replacement directories are preserved with warnings;
there is no recursive cleanup. If caller graph/state validation refuses close,
the original error remains primary, owned resources still retire and the invalid
graph stays closed rather than becoming an admitted inference model. Old graph
handles remain immutable and budgeted.
No concurrent backward/step, higher-order gradients, autocast, compile, vmap,
dtype/device mutation or `torch.optim` updates are admitted. Private f64 leaves
transport gradients; they are not latent weight values.

One complete numerical owner, current/old packed models and caller Torch
leaves/gradients still need memory. Source, numerical, gradient-transfer, recorded
scratch, packed and saved-input caps are separate; metadata, allocator/OS caches,
physical RSS/VRAM, source authenticity and objective/data/teacher attestation are
not established by this interface. Within-owner/GPU state and real-model
quality/performance qualification remain open.

## LoRA on a frozen ternary base

LoRA adapters train on top of a **frozen** ternary base. The contract, proptested:
the base weights receive **zero gradient** (they are frozen), the adapter merge is
correct, and the rank edges `r = 1` and `r = full` both behave. This is how you
fine-tune a quantized model cheaply without touching the base.

## QAT heal

The QAT heal loop is what closes SALT's residual loss (the "Heal" step of the
[SALT pipeline](./quantization.md)). It uses a `replace_weights` /
`invalidate_resident` bridge to swap healed weights back into the runtime and run
a short STE fine-tune. The convergence gate — a real ternary fine-tune recovering
≥90% of the accuracy gap vs the fp16 baseline — needs a GPU and a real source
model, so it is a documented manual gate on borrowed hardware rather than a
hosted-CI lane (gradient checks run CPU-only; the recovery gate cannot be
validated without GPU + a real model — see the
internal ADR (see the [research-record access policy](./research-records.md)).

## Ternary-specific training methods

The training methods that are specific to the ternary regime (and the choices
behind STE clipping, QAT scheduling, and the heal bridge) are written up in
ADR 0016 (see the [research-record access policy](./research-records.md)).
