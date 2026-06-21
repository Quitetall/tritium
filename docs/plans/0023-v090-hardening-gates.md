# 0023 — v0.90 hardening gates: missing-docs lint + cargo-semver-checks  (serves: v0.90/v1.0; point release `v0.5.10`)

## Goal

Two reachable-now release-readiness gates, both CPU-only and clear of the user's GPU/training WIP:
1. **Doc-coverage gate** — `#![deny(missing_docs)]` on the stable foundation crates so every public
   item must be documented (U9 / ADR 0011).
2. **API-stability gate** — `cargo-semver-checks` over the stable public-API crates so no
   *unintentional* breaking change lands (ADR 0011 hardening → ADR 0012 freeze).

Success: the 6 doc-gated crates build clean under the lint; `cargo-semver-checks` reports the current
API as non-breaking vs the last release; both run in CI.

## Scope (deliberately bounded)

Both gates cover the **stable, GPU-free, non-binary library crates**: `tritium-core`, `tritium-spec`,
`tritium-format`, `tritium-runtime`, `tritium-cpu`*, `tritium-quantize`, `tritium-testkit`. **Excluded
for now** (and why): `tritium-cuda` (needs nvcc/GPU), `tritium-cli`/`tritium-benches` (binaries),
`tritium-py` (PyO3 cdylib — gated by v0.80 ffi/py work), `tritium-nn`/`tritium-train` (public surface
still in flux during the active perf/training WIP). Widen as those stabilize toward v1.0.
(\*`tritium-cpu` is in the semver set; the doc lint set is the other six — cpu's doc sweep is a
follow-up.)

## Changes
- **`#![deny(missing_docs)]`** added to `tritium-core`, `tritium-spec`, `tritium-format`,
  `tritium-runtime`, `tritium-testkit`, `tritium-quantize`. spec/runtime/testkit were already fully
  documented; core (4), format (2), quantize (11) had **17** undocumented public items (enum-variant
  struct fields + one constructor) — all now documented.
- **`scripts/check-semver.sh`** — runs `cargo semver-checks --baseline-rev <ref>` over the 7 stable
  crates; baseline defaults to the latest `v0.5.*` tag (`git describe`).
- **`.github/workflows/ci.yml`** — a `semver` lane (installs cargo-semver-checks via
  `taiki-e/install-action`, `fetch-depth: 0` for the baseline tag, runs the script). The doc gate
  needs no new lane: the existing `cargo build`/clippy lane enforces `#![deny(missing_docs)]`.

## Gate (verified locally)
```
cargo build  -p {core,spec,format,runtime,testkit,quantize}   # deny(missing_docs) passes
cargo clippy -p {...} --all-targets -- -D warnings            # clean
cargo test   -p {...}                                         # green
./scripts/check-semver.sh v0.5.6   # 7 crates: "no semver update required", exit 0
```

## Done criterion
Doc lint enforced on the 6 crates (17 items documented); semver script + CI lane added + verified;
version `0.5.10`; CHANGELOG + ROADMAP updated; reviewed; tagged `v0.5.10` + pushed.
