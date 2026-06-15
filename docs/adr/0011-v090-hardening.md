# ADR 0011 — v0.90 Hardening

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 0.90 milestone of [ADR 0002](./0002-release-roadmap.md); follows [ADR 0010](./0010-v080-interop.md) (v0.80 interop), precedes [ADR 0012](./0012-v100-release.md) (v1.0 release)

## Status

Planned — not started. No code yet.

Must land first: the 0.80 interop milestone (ADR 0010) must be tagged green —
this milestone hardens the *entire* accumulated surface (all crates, all
backends, all frontends), so every prior milestone gate must already be closed
before the full-matrix and whole-suite sanitizer/fuzz passes are meaningful.

Hard blockers:
- Full CI matrix needs **per-platform runners** for every supported target
  (manylinux / macOS / Windows) plus the GPU and multi-GPU lanes from prior
  milestones — without them the matrix gate cannot go green.
- `compute-sanitizer` over the whole suite needs a **real GPU** lane; multi-GPU
  paths need the **≥2-GPU** lane from 0.60.
- Acceptance-model parity examples in docs need the **external model download**
  (BitNet b1.58 2B4T) available in CI.

## Scope

Hardening pass over the v1.0 surface: no new capabilities, only depth on what
exists. Delivers mdbook documentation, fuzzing breadth across every parser, the
full CI build+test matrix, packaging (Python wheels + `crates.io` publish),
perf-regression enforcement on `main`, and a security review with threat model.
Touches every crate's docs/CI/packaging metadata; no kernel or API surface
changes.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| Every parser ≥ 24h cumulative fuzzing, zero open findings; corpora committed | S | fuzz | scheduled |
| ASan/UBSan/TSan/`miri`/`compute-sanitizer` clean across the whole suite | M, S | sanitizer | GPU |
| Full CI matrix builds + tests green on every target | M, S | per-platform build | per-platform |
| Wheels build for manylinux / macOS / Windows | Do | fresh-env e2e | per-platform |
| `cargo publish --dry-run` clean for every crate | Do | contract test | cpu-only |
| mdbook builds with no dead links; examples run in CI | Do | golden | model-download |
| Every public API documented | Do | doctest | cpu-only |
| Perf-regression gates enforced on `main` | Pe | bench | GPU |
| Security review completed; threat model for untrusted model files documented | S | manual review | manual |
| `cargo-deny` clean (licenses + CVEs); SBOM generated | S | contract test | scheduled |

## Definition of done — tag v0.90.0

- [ ] All parsers ≥ 24h cumulative fuzzing with zero open findings; corpora committed.
- [ ] ASan/UBSan/TSan/`miri`/`compute-sanitizer` clean across the whole suite.
- [ ] Full CI matrix builds + tests green on every target.
- [ ] Wheels build for manylinux / macOS / Windows.
- [ ] `cargo publish --dry-run` clean for every crate.
- [ ] mdbook builds with no dead links; every public API documented; examples run in CI.
- [ ] Perf-regression gates enforced on the main branch.
- [ ] Security review completed; threat model for untrusted model files documented.
- [ ] `cargo-deny` clean (licenses + CVEs); SBOM generated.
- [ ] Tag `v0.90`.
