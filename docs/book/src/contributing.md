# Contributing

The authoritative contribution, governance, conduct, support, and security
policies live at the repository root:

- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)
- [`GOVERNANCE.md`](../../../GOVERNANCE.md)
- [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md)
- [`SUPPORT.md`](../../../SUPPORT.md)
- [`SECURITY.md`](../../../SECURITY.md)
- [`COMMUNITY.md`](../../../COMMUNITY.md)

Tritium is planned strategically and executed in small, gated steps. The map:

- **Strategic** — `docs/adr/`: Architecture Decision Records covering stable
  contracts and release gates. For current platform work, start at
  [ADR 0033](../../adr/0033-v11-full-public-release.md).
- **Index** — [`docs/ROADMAP.md`](../../ROADMAP.md): the living, ordered set of
  tactical plans from now to done, with status.
- **Tactical** — `docs/plans/NNNN-*.md`: one detailed, verification-gated plan
  per point-release or coherent feature. The v1.1 umbrella is
  [plan 0044](../../plans/0044-v11-full-public-release.md).

Milestone work is **gate-blocked, not date-blocked**. Independent work orders may
proceed in parallel, but no downstream claim becomes green until every declared
entry dependency and empirical gate passes.

## The conformance contract

If you add or change a backend, it must pass the **frozen conformance vector
set** (see [Conformance](./conformance.md)). Do not regenerate the set to make a
test pass — the set is immutable and protected by a drift gate; widening it is a
deliberate, reviewed re-freeze with a version bump.

## CI gates

The repository's CI (`.github/workflows/ci.yml`) is the authority on what must be
green. The shape of it:

- **`cpu-only-green`** — the required lane on every push: `cargo fmt --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test --workspace --exclude tritium-py`, across Linux / macOS / Windows.
  `RUSTFLAGS = -D warnings`.
- **`cargo-deny`** — supply-chain gate: licenses (allow-list + NOTICE),
  RustSec advisories, bans (no external wildcards), sources.
- **`semver`** — `cargo-semver-checks` API-stability gate vs the last release tag.
- **`wasm`** — `tritium-wasm` conformance inside wasmtime (`wasm32-wasip1`).
- **`serve-contract` / `candle-conformance` / `burn-conformance` / `onnx-op`** —
  the interop lanes, each feature-gated and CPU-only.
- **`fuzz`** — scheduled: every `tritium-format` parser, zero crashes.
- **`gpu` / `wgpu` / `perf-regression` / `serve-e2e`** — self-hosted, manual
  (`if: ${{ false }}`): they run only where a CUDA toolkit / Vulkan adapter / a
  pinned GPU box / a real model is present.
- **`publish-check` / `sbom`** — packaging-readiness + a CycloneDX SBOM.

Foundation crates carry `#![deny(missing_docs)]`, so every public item is
documented; the doctest sweep keeps the examples runnable.

## Documenting

This book is built with [mdBook](https://rust-lang.github.io/mdBook/) and gated by
`mdbook-linkcheck` (any dead internal link fails the build). To work on it:

```sh
mdbook serve docs/book      # live preview at http://localhost:3000
mdbook build docs/book      # one-shot build; runs the link checker
```

`mdbook build` runs the link checker as a backend (configured in
`docs/book/book.toml`), so a dead link fails locally exactly as it does in CI
(the `docs.yml` workflow). Cross-links to ADRs use relative paths
(`../../adr/NNNN-*.md`) that resolve to the real files under `docs/adr/`.

## License of contributions

Contributions are under [Apache-2.0](https://github.com/Quitetall/tritium/blob/main/LICENSE),
matching the project license.
