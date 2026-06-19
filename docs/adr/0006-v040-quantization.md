# ADR 0006 — v0.40 SALT Quantization

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.40 milestone of [ADR 0002](./0002-release-roadmap.md); implements [ADR 0001 — SALT](./0001-salt-quantization.md); follows [ADR 0005](./0005-v030-performance.md); precedes [ADR 0007](./0007-v050-training-core.md)

## Status

Tag-ready. `tritium-quantize` (residual expansion + rate-distortion allocator +
`quantize_tensor`, with a per-tensor `ScaleGroup::Tensor` base for QAT-ternary masters), the
TQ2_0 residual sidecar + whole-model SALT bundle in `tritium-format`, the **GPU** multi-plane
accumulate kernel (`salt_mpgemm_tiled_f32`, matches the dequant reference within 1e-4), and
the `tritium quantize` CLI have all landed. **All CPU exit gates green**; the accuracy gate
is **reframed** (see below); SALT is validated both ways — `salt@1.585 == deployed I2_S` on
b1.58 and a monotone recon-error-vs-bpw curve on a normal fp model.

The three items previously scoped as deferrable have **all landed** (storage/IO/wiring, not
core correctness):

- **GGUF writer** (`tritium-format::write_gguf`, the reader's inverse) + a **SALT-in-GGUF
  container** (`write_salt_gguf`/`read_salt_gguf`), wired to `quantize --format gguf`. The
  sidecar bundle remains the canonical artifact; the GGUF is the single-container option.
- **Resident-GPU SALT decode** (`CudaBackend::upload_salt` + `salt_forward`,
  `SaltResidentLinear`): the `salt_mpgemm_tiled_f32` kernel now runs against a VRAM-resident,
  plane-major weight uploaded once — the building block a full SALT decode forward composes
  per projection (gated vs the host dequant reference + resident reuse).
- **Sparse residual plane** (`tritium-format::sparse`): the ADR 0001 §5 storage form +
  density switch (`choose_plane_repr`), round-tripping byte-identically to dense and gated
  for matmul-output equivalence (`sparse_dot` bit-exact vs the dense dot).

Remaining as explicit **v0.5+ follow-ons** (not v0.4.0 gates): a full SALT decode *forward*
composing the resident primitive across every projection, a Qwen-arch perplexity curve, and
the per-arch GPU **sparse-matmul** kernel (the compute win on top of the sparse storage form).

**Accuracy gate reframe (load-bearing).** The original "accuracy-vs-bpw within the stated
fp16 gap" gate is ill-defined for the only real model, BitNet b1.58: its bf16 "master" is
*latent* QAT weights, not a usable forward — raw-master perplexity is garbage and the SALT
curve *inverts* (more bpw reconstructs the latent master more faithfully → further from the
deployed ternary the model was trained for). Reframed to two gates that are actually
meaningful: (a) **`salt@1.585` (per-tensor base) reproduces the deployed I2_S** on b1.58
(proven: the per-tensor ternary matches the GGUF weights to f16); (b) a **smooth monotone
recon-error-vs-bpw curve on a normally-trained fp model** (gpt2), the validation b1.58
cannot give. A full Qwen-arch perplexity curve is the deferred "real accuracy" follow-on.

**Must land first:** v0.30 (performance) tagged green — SALT's multi-plane
accumulate (`Σ_p s_p·tmatmul`) rides the tuned mpGEMM kernels (add-only + IMMA,
all-ISA), so those must be conformant and benchmarked before residual planes
stack on top.

**Hard blocker:** an accuracy gate requires a real **fp16 source model** plus an
accuracy harness (perplexity / downstream task) wired into CI — the accuracy-vs-bpw
curve and the fp16-gap target cannot be validated against a synthetic model. This
implies a `model-download` CI lane and GPU time to run perplexity at scale.

## Scope

Ship `tritium-quantize` implementing [ADR 0001 SALT](./0001-salt-quantization.md):
residual planes, the mode codebook, sensitivity-driven plane allocation, and the
sparse residual plane. Add the **TQ2_0 residual sidecar** format (multi-plane
weights alongside legacy plain-TQ2 for backward-compat) to `tritium-format`, and a
`cli quantize` subcommand. Touches `tritium-quantize` (new), `tritium-format`
(sidecar), `tritium-cli`, and the runtime's multi-plane accumulate path.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| Multi-plane accumulate `Σ_p s_p·tmatmul` matches a SALT dequant→fp32 reference matmul within tolerance | C | vs-reference (dequant→fp32 reference matmul) | GPU |
| Residual reconstruction error decreases monotonically with plane count `T` | C/E | proptest over `T` (monotonicity property) | cpu-only |
| `T=1` reduces **exactly** to flat AbsMean (BitNet regression check) | C/E | golden (bit-exact vs flat AbsMean path) | cpu-only |
| Allocator respects the bpw budget exactly (`Σ|g|·1.585·T_g ≤ budget`); higher-sensitivity groups get ≥ planes than lower (ordering invariant) | C | proptest (budget + ordering invariants) | cpu-only |
| Sparse residual plane and dense residual plane produce identical matmul output; density-threshold switch correct on both sides | C/P | vs-reference / parity (sparse vs dense, both sides of threshold) | GPU |
| Sidecar roundtrips multi-plane weights; reads legacy plain-TQ2 (no residual); version field enforced; edge budgets (1.58=all base, very high=many planes), zero-variance group, outlier-heavy group | C/E | golden roundtrip + contract test (version/back-compat) | cpu-only |
| Same model+seed+budget ⇒ byte-identical packed output | D | single-vs-multi-run byte compare (determinism) | cpu-only |
| Accuracy-vs-bpw curve reported on the real model; at target bpw, within the stated gap of fp16 | Pe/C | accuracy harness vs fp16 reference (perplexity / downstream) | model-download |

## Definition of done — tag v0.40.0

- [x] Multi-plane accumulate kernel `Σ_p s_p·tmatmul` matches the SALT dequant→fp32 reference matmul within tolerance. *(tritium-cuda: `salt_mpgemm_tiled_f32` + `salt_mpgemm_matches_dequant_reference`, 1e-4)*
- [x] Residual reconstruction error decreases monotonically with plane count `T`; `T=1` reduces exactly to flat AbsMean (BitNet regression check). *(tritium-quantize: `plane.rs` gates)*
- [x] Allocator respects the bpw budget exactly (`Σ|g|·1.585·T_g ≤ budget`); higher-sensitivity groups receive ≥ planes than lower (ordering invariant). *(tritium-quantize: `allocate.rs` gates)*
- [x] Sparse residual plane and dense residual plane produce identical matmul output; the density-threshold switch is correct on both sides. *(tritium-format: `sparse.rs` — `sparse_from_tq2_0`/`sparse_to_tq2_0` byte-identical round-trip, `sparse_dot` bit-exact vs the dense dot, `choose_plane_repr` sparse@2.5%/dense@50%, malicious-input hardening. The per-arch GPU sparse-matmul kernel is the v0.5+ compute follow-on; the storage form + equivalence gate land here.)*
- [x] Format sidecar roundtrips multi-plane weights; reads legacy plain-TQ2 (no residual) for backward-compat; version field enforced; edge budgets, zero-variance group, and outlier-heavy group all handled. *(tritium-format: `salt.rs` + `salt_bundle.rs` gates, incl. malicious-input hardening)*
- [x] Same model+seed+budget ⇒ byte-identical packed output. *(determinism gates in `plane.rs`/`allocate.rs`/`quantize.rs`/`salt_bundle.rs`)*
- [x] **(reframed)** Accuracy validated: `salt@1.585 == deployed I2_S` on b1.58 *(tritium-nn `salt_accuracy` / `gguf_eval_perplexity`)* **and** a monotone recon-error-vs-bpw curve on a normal fp model *(tritium-quantize `recon_curve`, gpt2 0.540→0.387)*. The literal "within the fp16 gap" gate is dropped for a QAT-ternary master (no valid fp16 upper bound); a full Qwen-arch perplexity curve is the deferred follow-on.
- [x] `tritium quantize` CLI (fp safetensors → SALT bundle **or** GGUF container). *(tritium-cli `quantize.rs`; `--format gguf` now emits a SALT-in-GGUF container via `tritium-format::write_salt_gguf`, atop the new general `write_gguf`.)*
- [x] **GGUF writer** (`tritium-format::write_gguf`, inverse of the reader) + SALT-in-GGUF container (`write_salt_gguf`/`read_salt_gguf`). *Round-trips through `read_gguf`; gates for every value type, alignment, and malicious input.*
- [x] **Resident-GPU SALT decode** primitive (`CudaBackend::upload_salt` + `salt_forward`, `SaltResidentLinear`): the kernel runs against a VRAM-resident plane-major weight uploaded once. *(tritium-cuda: `salt_resident_forward_matches_dequant` — T=1/2/3 incl. ragged, vs host dequant + resident reuse.)*
- [ ] **Tag `v0.40`** — all DoD gates above green; pending U1–U9 + the final tag action.
