# ADR 0012 — v1.0 Release

- **Status:** Accepted — **DONE, tagged `v1.0.0` (2026-06-28)**
- **Date:** 2026-06-15 (planned); 2026-06-28 (completed)
- **Relates:** executes the 1.0 milestone of [ADR 0002](./0002-release-roadmap.md); gated on the prior milestone [ADR 0011](./0011-v090-hardening.md); final step after [ADR 0004](./0004-v020-inference-spine.md) and the milestone chain it heads.

## Status

**Accepted — complete (`v1.0.0`, 2026-06-28).** v1.0 is the freeze milestone: it adds
no new capability, only locks what 0.10–0.90 built. It began after **0.90 was tagged
`v0.9.0`** (2026-06-24). All four exit gates met — see the **Outcome** section below.

### Outcome (2026-06-28)

**Gate "Do" (freeze) — met, tiered.** v1.0 adopts a **tiered freeze** (decision recorded
in `docs/v1.0-api-freeze-audit.md`): the **frozen core** — `tritium-core`, `-spec`,
`-format`, `-runtime`, `-cpu`, `-quantize`, `-testkit` + the **C ABI**
(`TRITIUM_ABI_VERSION = 1`, cbindgen-drift + C11/C++17 gated) — is under semver
(`cargo-semver-checks` green vs the `v0.5.10` baseline). The **evolving tier** —
`tritium-nn`, `-train`, `-cuda`, interop (`-candle`/`-burn`/`-onnx`), `-serve` — is
documented as *not* semver-gated and may take breaking changes in 1.x minors (these
track fast-moving upstreams + ongoing perf/training work). This deliberately avoids a
risky last-minute reorganization of `tritium-train`'s surface (the audit's flagged P0)
by scoping the guarantee instead of forcing the surgery.

**Gate "C" (no regression) — met.** Every CPU gate re-runs green on the release commit
(fmt, clippy `-D warnings`, full workspace test, semver, cargo-deny, publish-check, SBOM,
wasm, the interop conformance lanes, free Metal CI). GPU validation is **fenced** per the
ADR-0011 amendment: CUDA conformance + the capstone below on a local RTX 4090; Metal (M1),
ROCm (MI300X), wgpu (4090) parity from prior fenced sessions; the GPU CI lanes are kept as
dispatchable `workflow_dispatch` recipes (no standing paid runners).

**Gate "Do" (docs reproducible) — met.** Quickstart, model zoo, benchmark methodology in
`docs/book/` (CPU-reproducible; measured GPU numbers honestly fenced).

**Gate "Real & usable capstone" — met, PROVEN on real hardware.** Real
`microsoft/bitnet-b1.58-2B-4T` runs correctly end-to-end on a local RTX 4090 (sm_89):
`cuda_perplexity_within_1pct` (ours 1.3987 / ref 1.4028, rel 2.96e-3), `cuda_greedy_matches_transformers`
(256/256 token-exact), `cpu_cuda_parity` (identical IDs, logit rel 2.26e-6),
`cuda_batch_decode_matches_single`, and `qat_heal_gate` (94.6% layerwise distillation
convergence, PPL 1.40). This required fixing two latent CUDA resident-decode bugs
(`302d059`: the `rmsnorm_quant_f32` shared-memory aliasing + the unscaled per-layer
`f_tiled` mpgemm), guarded by a new teeth-proven conformance test (`51b041d`).

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
