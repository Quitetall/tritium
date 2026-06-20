# ADR 0007 — v0.50 Training Core

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.50 milestone of [ADR 0002](./0002-release-roadmap.md); depends on 0.40 SALT quantization ([ADR 0006](./0006-v040-salt-quantization.md)); precedes 0.60 pretraining + distributed ([ADR 0008](./0008-v060-pretraining-distributed.md))

## Status

Planned — **not started**. No `tritium-train` crate exists yet; this milestone is unblocked only once the inference spine (0.20) and SALT quantization (0.40) gates are green and tagged, since QAT heals SALT loss and the backward path reuses the forward kernels validated upstream. **Hard blocker:** the accuracy gate needs a GPU and a real fp16 source model plus a small fine-tune task with a known recoverable accuracy gap wired into CI — gradient checks run cpu-only, but the convergence/recovery gate cannot be validated without GPU and a real model.

## Scope

`tritium-train`: straight-through-estimator (STE) autograd, quantization-aware training (QAT), backward kernels for the ternary ops, an optimizer with save/restore state, and LoRA adapters on a frozen ternary base. Single-node only (distributed is 0.60). Touches: new `tritium-train` crate; backward additions to `tritium-cpu` / `tritium-cuda`; reuses `tritium-nn` forward ops and the SALT path from `tritium-quantize`.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| STE backward vs finite-difference numerical gradient within tolerance, every trainable op | C | gradient-check | cpu-only |
| Autograd graph reproduces analytic gradients on toy problems | C | vs-reference | cpu-only |
| LoRA: base weights receive zero gradient (frozen); adapter merge correct; rank edges `r=1` and `r=full` | C | proptest | cpu-only |
| Optimizer state save/restore bit-exact; resume == uninterrupted run | C/D | golden | cpu-only |
| No NaN/Inf over ≥1k steps; bf16-master mixed-precision path matches; same seed ⇒ same loss curve | E/D | conformance harness | GPU |
| Real ternary fine-tune recovers `≥90%` of the lost accuracy gap vs fp16 baseline; loss decreases (convergence smoke) | Pe | vs-reference | model-download |

## Definition of done — tag v0.5.0

- [ ] Gradient check: STE backward matches finite-difference numerical gradient within tolerance for **every** trainable op.
- [ ] Autograd graph reproduces analytic gradients on toy problems.
- [ ] LoRA: frozen base receives zero gradient; adapter merge is correct; rank edges `r=1` and `r=full` pass.
- [ ] Optimizer state save/restore is bit-exact; resumed run equals the uninterrupted run.
- [ ] No NaN/Inf over ≥1k steps; bf16-master mixed-precision path matches; same seed ⇒ same loss curve.
- [ ] A real ternary fine-tune recovers `≥90%` of the lost accuracy gap vs the fp16 baseline; loss decreases (convergence smoke).
- [ ] U1–U9 green on CPU+CUDA. Tag `v0.50`.
