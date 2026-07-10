# 0039 — Real Fisher sensitivity + adaptive plane growth  (serves: ADR 0020 / capstone-cascade)

## Goal
Replace the "Energy ≈ Uniform" sensitivity signal with a **real, loss-derived** one: the diagonal
Fisher `F_i = E[(∂L/∂w_i)²]` accumulated over data, reduced per allocator tile → `Sensitivity::Custom`.
Then a **periodic plane-growth** policy that adds planes where `error × Fisher` stays high under a
rising bpw target. This is the user's core thesis — *add ternary planes where the loss is most
sensitive* — and directly answers the shipped finding that Energy barely re-ranks vs Uniform.

## Why Energy ≈ Uniform (the finding this plan fixes)
The allocator ranks groups by `H_g · Δerr_g(T) / cost`. For AbsMean residual expansion, `Δerr_g` is
*already* ~proportional to group energy `‖w_g‖²`, so multiplying by `H_g = ‖w_g‖²` (Energy) mostly
re-scales an energy-correlated ranking — the plane order barely moves vs `H_g = 1` (Uniform). Only a
signal **decorrelated from raw magnitude** — the loss curvature `E[(∂L/∂w)²]` — changes the ranking.
Output/critical layers are loss-sensitive out of proportion to their weight magnitude; Fisher sees
that, Energy does not.

## What already exists (from exploration — most of the machinery is here)
- `quantize::Sensitivity::Custom(Vec<f64>)` — **fully plumbed** through `quantize_tensor`
  (length-validated `rows*nb`, consumed at group build; allocator needs *zero* change).
- Rate-distortion allocator `allocate(groups, cfg)` ranks by `H_g·Δerr/cost`; `higher_budget_never_reduces_planes` proves monotone growth.
- `tritium-train` reverse tape: `Tape::backward(loss) -> Vec<Vec<f32>>` — the gradient hook.
- Recon-fidelity report (`ReconAccum`/`report salt-model`) + the tiny-SwiGLU forward (`salt_distill_e2e.rs`) for the honest gate.

## Steps (one gated commit each)

### Step 1 — Diagonal-Fisher primitive: accumulator + tile reduction (TDD) ← FIRST COMMIT
- `tritium-train::fisher::FisherAccumulator` — `new(len)`, `accumulate(&grad)` (adds `g²`),
  `into_diag() -> Vec<f64>` (mean of squared grads = `E[(∂L/∂w)²]`). Trivial, deterministic.
- `tritium-quantize::fisher::tile_sensitivity(per_weight: &[f64], rows, k) -> Vec<f64>` — reduce a
  per-weight Fisher to **one `H_g` per (row, 256-block) group**, row-major `r·nb+b`, as the **mean**
  Fisher over the block's weights (so `H_g·Σ_iΔw_i² ≈ Σ_i F_iΔw_i²`, exact when F is flat on the tile).
  Length `rows·num_blocks(k)` — exactly `Sensitivity::Custom`'s contract.
- **Gate (TDD, unit):** known grads → accumulator = analytic mean-square; `tile_sensitivity` on a
  hand-checked buffer = correct per-tile means + correct length; feeding it as `Custom` into
  `quantize_tensor` allocates (length accepted, no panic). fmt+clippy `-D`.

### Step 2 — The honest gate: Fisher-allocation beats Energy/Uniform on FORWARD loss at fixed bpw
Reuse the tiny SwiGLU transformer. Compute per-weight diagonal Fisher of the task/KL loss via
`Tape::backward` over a small data batch (square-accumulate). Quantize every 2D weight **three ways
at the same budget_bpw** — `Uniform`, `Energy`, `Custom(tile_sensitivity(fisher))` — then **forward
each quantized model and measure the actual loss** (KL to the fp teacher / xent). 
**Gate:** `loss(Custom) < loss(Energy)` AND `loss(Custom) < loss(Uniform)` — Fisher-allocated
quantization hurts the model less at equal bits. This is NOT the weight-space objective the allocator
minimizes (that would be tautological); it tests that Fisher *predicts forward-loss sensitivity*
better than magnitude. Print all three losses + realized bpw. Lives where both crates are dev-deps.

### Step 3 — Adaptive plane growth (periodic re-allocation under a quality target)
An outer loop over the distill: every `N` steps, recompute Fisher from the running gradients and
re-allocate at a **rising** bpw (or grow tiles whose `Fisher·residual-error` is top-k) until a
quality target (KL/ppl) is met. Reuse the allocator's monotone-growth property so planes only add.
**Gate:** adaptive growth reaches the target KL at **lower average bpw** than uniform growth to the
same target (fewer bits for equal quality). Guard against oscillation (hysteresis / grow-only).

### Step 4 — CLI + report wiring for Custom/Fisher
`--sensitivity custom --fisher <sidecar>` (per-tensor Fisher buffers) for `tritium quantize` and
`report salt-model`; a `SaltSensitivityArg::Custom` arm + loader. **Gate:** CLI round-trip — quantize
with a Fisher sidecar, `report salt-model` shows Custom beats Energy frob/KL at fixed bpw on a real
tensor. (Producing the sidecar from a training run is the 0038b/0040 tie-in.)

## Verification
Each step: `cargo test -p <crate>` green incl. the new gate, `cargo clippy --all-targets -D warnings`
+ `cargo fmt --check` clean, code-reviewer subagent, then commit (explicit-path staging — shared
tree, see the no-`--amend` rule) + push via deploy key when the tree is clean and mine-only.

## Risks
Fisher estimation noise on tiny batches (mitigate: enough samples, or a constructed decorrelation);
growth oscillation (grow-only + hysteresis); re-alloc churn (Step 3 re-quantizes from the latent, not
incrementally — fine at small scale, revisit at 32B in 0040).
