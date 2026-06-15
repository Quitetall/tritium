# ADR 0009 — v0.70 Backend Breadth

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.70 milestone of [ADR 0002](./0002-release-roadmap.md); follows [ADR 0008](./0008-v060-pretraining-distributed.md) (v0.60 Pretraining + distributed), precedes [ADR 0010](./0010-v080-interop-frontends.md) (v0.80 Interop / frontends)

## Status

Planned — not started. This is the depth→breadth pivot: it widens the
`tritium-spec` contract from CPU+CUDA to all remaining backends, so it cannot
begin until the spine is fully exercised on two backends. **Must land first:**
v0.60 (`v0.60.0` tagged) — the distributed/pretraining gate closes the last
single-backend capability, and the conformance vector set must be frozen and
versioned (the same one CPU/CUDA pass). **Hard blockers:** per-platform hardware
the team does not yet own outright — an Apple-silicon (Metal) box, an AMD ROCm
GPU, and a WebGPU/WASM runtime target; CI lanes (or documented emulation) for
each. Without real per-platform devices the parity gates below cannot be
validated.

## Scope

Ship the three remaining backend crates, each implementing `tritium-spec`:
`tritium-metal` (Apple GPU, unified memory), `tritium-rocm` (AMD), and
`tritium-wgpu`/`tritium-wasm` (WebGPU + browser/WASM). No new ops or contract
changes — these crates copy the now-stable spec the CPU/CUDA backends defined.
Each is registered in the runtime registry and selectable, and each must pass
the full conformance vector set. Touches: `tritium-metal`, `tritium-rocm`,
`tritium-wgpu`, `tritium-wasm`; `tritium-testkit` (per-platform harness lanes).

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| Each new backend passes the **full** conformance vector set (bit-exact integer paths, ≤ε float) | C/P | conformance harness | per-platform |
| Each new backend agrees with CPU/CUDA on identical input within tolerance | P | vs-reference (conformance set) | per-platform |
| Every backend reproduces the acceptance model's greedy token output (or sampling distribution) | C | vs-reference (acceptance model) | model-download |
| Platform edges: WASM memory ceiling; Metal unified-memory path; ROCm arch variants | E | conformance harness (boundary suite) | per-platform |
| Graceful, defined fallback when a backend lacks a capability (e.g., no fp8) | F | contract test | per-platform |
| Per-platform sanitizer/leak checks where tooling exists | M | sanitizer | per-platform |

## Definition of done — tag v0.70.0

- [ ] **C/P** Each new backend (`tritium-metal`, `tritium-rocm`, `tritium-wgpu`/`tritium-wasm`) passes the full conformance vector set — bit-exact integer paths, ≤ε float — the same set CPU/CUDA pass.
- [ ] **P** Cross-backend parity: every new backend agrees with CPU/CUDA within tolerance on the conformance set.
- [ ] **C** Every backend reproduces the acceptance model's greedy token output (or sampling distribution).
- [ ] **E/F** Platform edges handled: WASM memory ceiling; Metal unified-memory path; ROCm arch variants; graceful, defined fallback when a backend lacks a capability (e.g., no fp8).
- [ ] **U7/M** Per-platform sanitizer/leak checks pass where tooling exists.
- [ ] U1–U9 green on every new backend; CI lanes (or documented emulation) exist per platform. Tag `v0.70`.
