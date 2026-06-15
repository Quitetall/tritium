# ADR 0006 — v0.40 SALT Quantization

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.40 milestone of [ADR 0002](./0002-release-roadmap.md); implements [ADR 0001 — SALT](./0001-salt-quantization.md); follows [ADR 0005](./0005-v030-performance.md); precedes [ADR 0007](./0007-v050-training-core.md)

## Status

Planned — not started. No `tritium-quantize` crate, residual-sidecar format, or
`cli quantize` code exists yet.

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

- [ ] Multi-plane accumulate kernel `Σ_p s_p·tmatmul` matches the SALT dequant→fp32 reference matmul within tolerance.
- [ ] Residual reconstruction error decreases monotonically with plane count `T`; `T=1` reduces exactly to flat AbsMean (BitNet regression check).
- [ ] Allocator respects the bpw budget exactly (`Σ|g|·1.585·T_g ≤ budget`); higher-sensitivity groups receive ≥ planes than lower (ordering invariant).
- [ ] Sparse residual plane and dense residual plane produce identical matmul output; the density-threshold switch is correct on both sides.
- [ ] Format sidecar roundtrips multi-plane weights; reads legacy plain-TQ2 (no residual) for backward-compat; version field enforced; edge budgets, zero-variance group, and outlier-heavy group all handled.
- [ ] Same model+seed+budget ⇒ byte-identical packed output.
- [ ] Accuracy-vs-bpw curve reported on the real model; at target bpw, within the stated gap of fp16.
- [ ] Kernel matches dequant reference + sparse==dense + accuracy curve meets target — plus U1–U9. Tag `v0.40`.
