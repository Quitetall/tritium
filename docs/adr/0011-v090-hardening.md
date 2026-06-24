# ADR 0011 — v0.90 Hardening

- **Status:** In progress — reachable hardening tooling shipped (0.5.x/0.6.x); GPU-matrix exit gate **amended 2026-06-24** (see Amendment below)
- **Date:** 2026-06-15
- **Relates:** executes the 0.90 milestone of [ADR 0002](./0002-release-roadmap.md); follows [ADR 0010](./0010-v080-interop.md) (v0.80 interop), precedes [ADR 0012](./0012-v100-release.md) (v1.0 release)

## Status

In progress. The reachable hardening tooling has shipped across the 0.5.x/0.6.x
line: cargo-deny, fuzz breadth + corpora, doc-coverage + semver baseline, mdbook +
dead-link lane, ASan/MSan/TSan + `miri` sanitizers, abi3 Python wheels, cpu-bench-
smoke, SBOM, and SECURITY/threat-model. The remaining gates depend on GPU/multi-GPU
hardware in CI; those are addressed by the **2026-06-24 amendment** below.

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

## Amendment (2026-06-24) — GPU-matrix exit gate

The "full CI matrix green on every push" gate assumed standing per-platform + GPU
runners. Continuous GPU runners cost $700–6000/mo per GPU; a research project has no
such budget, and parking the GPU lanes at `if: false` indefinitely is not a real
gate. The gate is therefore amended as follows:

1. **GPU-backend parity is validated by fenced manual sessions, with logged evidence**,
   rather than a per-push CI matrix:
   - **cuda** — Ampere / 2×A100 (51/51 + memcheck-clean, `v0.6.0`).
   - **wgpu** — 4090 Vulkan adapter (`v0.6.1`) + the Apple **Metal HAL** (`v0.6.9`).
   - **wasm** — wasmtime (`v0.6.1`).
   - **metal** — real Apple M1, bit-exact 89-vector conformance (`v0.6.9`).
   - **rocm** — real AMD Instinct MI300X (gfx942, ROCm 7.2.4), frozen-vector
     conformance ran on-GPU, 1e-4 (`v0.7.0`).
   Each is recorded in `CHANGELOG.md` and `docs/ROADMAP.md`.
2. **The GPU CI lanes are kept as reproducible, dispatchable recipes, not dead code.**
   `gpu` / `rocm` / `wgpu` / `perf-regression` / `serve-e2e` are gated on
   `workflow_dispatch` (`runs-on: [self-hosted, …]`): register a matching GPU runner,
   click *Run workflow*, and the exact validation re-runs on demand. The **`metal`**
   lane runs for free on the GitHub-hosted `macos-14` (Apple-Silicon) runner on every
   `main` push, self-skipping cleanly if that VM's Metal device is unavailable.
3. **"Green on every push" is waived for the paid-GPU backends** (cuda/rocm/wgpu)
   absent standing runners; the matrix gate is considered met by (1) + (2). If a
   continuous GPU-runner budget ever exists, flipping the `workflow_dispatch` guards
   back to `push` restores the always-on matrix with no other change.

This keeps the gate honest (the validation is real and reproducible) without
pretending a hobby-scale project can fund 24/7 GPU CI.

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
