# ADR 0012 — v1.0 Release

- **Status:** Planned
- **Date:** 2026-06-15
- **Relates:** executes the 1.0 milestone of [ADR 0002](./0002-release-roadmap.md); gated on the prior milestone [ADR 0011](./0011-v090-hardening.md); final step after [ADR 0004](./0004-v020-inference-spine.md) and the milestone chain it heads.

## Status

Planned — not started. v1.0 is the freeze milestone: it adds no new capability, only
locks what 0.10–0.90 built. It cannot begin until **0.90 (hardening) is tagged
`v0.90`** — full CI matrix green, zero sanitizer/fuzz findings, docs complete — since
v1.0 re-runs every prior gate on the release commit and freezes the surface 0.90
stabilized.

**Hard blocker:** the capstone requires a **real model** end-to-end (load → infer →
SALT-quantize → fine-tune) in a **fresh environment**, which needs an
**external model download** (BitNet b1.58 2B4T) and **GPU** (fine-tune/quantize at
real scale). A `cargo-semver-checks` baseline must be captured before the API/ABI can
be declared frozen.

## Scope

API/ABI freeze with `cargo-semver-checks` baseline + semver enforcement; final docs
(quickstart, model zoo) and a third-party-reproducible benchmark report; a capstone
fresh-env e2e that exercises install → inference → SALT quantize → fine-tune on a real
model. Touches no new crates — freezes the public API across all crates and the
`tritium-ffi` C ABI; adds CI lanes for semver checking and the fresh-env capstone.

## Testability (exit gates)

| Gate | Tag | How tested | CI lane |
|------|-----|------------|---------|
| `cargo-semver-checks` baseline set; public API + C ABI frozen | Do | contract test (`cargo-semver-checks` against baseline) | cpu-only |
| Every prior milestone gate re-runs green on the release commit (no regression) | C | conformance harness + full prior-gate suite vs-reference | GPU |
| Quickstart, model zoo, and benchmark report reproducible by a third party | Do | fresh-env e2e + golden benchmark report | model-download |
| Capstone: fresh env, public docs only — install, load model, infer, SALT-quantize, fine-tune | C | fresh-env e2e (single-vs-multi-GPU at real scale) | GPU |

## Definition of done — tag v1.0.0

- [ ] `cargo-semver-checks` baseline set; public API + C ABI frozen.
- [ ] Every prior milestone gate re-run green on the release commit (no regression) — full suite.
- [ ] Quickstart, model zoo, and a benchmark report reproducible by a third party.
- [ ] Real & usable capstone: in a fresh environment, following only public docs, a user can `pip`/`cargo` install, load a model, run inference, quantize with SALT, and fine-tune — validated by an end-to-end fresh-env test in CI.
- [ ] External reproduction of the quickstart confirmed. Tag `v1.0.0`.
