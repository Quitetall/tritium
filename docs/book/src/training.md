# Training (QAT / STE)

`tritium-train` provides training for ternary BitNet models:
straight-through-estimator (STE) autograd, quantization-aware training (QAT),
backward kernels for the ternary ops, an optimizer with bit-exact save/restore,
and LoRA adapters on a frozen ternary base. It is specified in
ADR 0007 (see the [research repository](https://github.com/Quitetall/tritium-research)). Distributed training — FSDP /
ZeRO-3 (via `FlatShardPlan`), a `ProcessGroup` (with `SimProcessGroup` and NCCL
backends), and distributed checkpointing (`DistCheckpoint`) — shipped in
v0.60 (see the [research repository](https://github.com/Quitetall/tritium-research)) and is re-exported from
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
ADR (see the [research repository](https://github.com/Quitetall/tritium-research))).

## Ternary-specific training methods

The training methods that are specific to the ternary regime (and the choices
behind STE clipping, QAT scheduling, and the heal bridge) are written up in
ADR 0016 (see the [research repository](https://github.com/Quitetall/tritium-research)).
