# API stability tiers

Tritium publishes 23 crates. They are not all the same kind of thing: some are a
contract other people build against, some are an engine that changes when the
measurements say it should, and some are backend adapters that track a moving
upstream. Freezing all three alike would freeze accidents as if they were design.

This document assigns every published crate to a tier and states what that tier
promises. It is the answer to a question the codebase previously left implicit:
`scripts/check-semver.sh:12` already excluded `tritium-nn` and `tritium-train`
as "the documented evolving 1.x tier" — a tier that was documented nowhere.

## Status of this document

Tiers are declared here as of 2026-09-03 and are being applied incrementally.
Until a crate's header states its tier, treat this table as the authority.

The starting surface was measured on 2026-09-03 by walking every `pub enum` and
`pub struct` under `crates/*/src/**` and checking whether `#[non_exhaustive]`
appears in the attribute block immediately preceding it: **130 public enums and
240 public structs with public fields carry no seal.** Treat that as an upper
bound — it counts definitions, and some sit in private modules where they are not
reachable from outside the crate. Sixty of the 130 unsealed enums live in
`tritium-cuda`, `tritium-nn`, `tritium-train` and `tritium-serve` — the four
crates the semver gate has historically excluded, so nothing has been watching
them. The sealing pass runs Tier 1 first.

## The tiers

### Tier 1 — Stable

`tritium-core`, `tritium-format`, `tritium-spec`, `tritium-runtime`,
`tritium-cpu`, `tritium-quantize`, `tritium-testkit`, `tritium-ffi`

The public API is a contract. Every public enum and every public struct with
public fields carries `#[non_exhaustive]`, so adding a variant or a field is not
a breaking change. Breaking changes require a major version and are announced in
`CHANGELOG.md`.

`tritium-ffi` is here for its **C ABI**, which is the real contract; its Rust
surface is incidental to that.

### Tier 2 — Evolving 1.x

`tritium-nn`, `tritium-serve`, `tritium-salt`, `tritium-train`, `tritium-cli`

These crates are the engine, and the engine changes when a measurement says it
should. Public **error enums** and published **configuration types** are sealed,
because those are what callers actually match on and construct. The rest of the
surface may change in a minor release, and such changes are recorded in
`CHANGELOG.md` rather than deferred to a major.

`tritium-cli` is a binary: its contract is the command-line surface — subcommand
names, flags and output formats — not a Rust API.

### Tier 3 — Backend and interop

`tritium-cuda`, `tritium-metal`, `tritium-rocm`, `tritium-wgpu`, `tritium-wasm`,
`tritium-onnx`, `tritium-burn`, `tritium-candle`, `tritium-mcu`,
`tritium-build-info`

**No API stability guarantee.** These crates exist to reach a platform or a
framework, and they follow whatever that platform or framework does. Their public
items are reachable but unsupported: depend on them only if you are willing to
follow the upstream they track. They receive no sealing pass, and a break in one
is expected rather than newsworthy.

### Outside the tiers

- `tritium-py` ships to PyPI, not crates.io. Its compatibility promise is the
  wheel/ABI matrix in [`docs/compatibility.md`](./compatibility.md).
- `tritium-benches` sets `publish = false` and is not part of any public surface.

## How this interacts with the semver gate

`scripts/check-semver.sh` runs `cargo-semver-checks`. During the 1.1.0 release
candidate window it **reports** rather than blocks: the breaking-change list is
recorded against the release-candidate section of `CHANGELOG.md` instead of
failing CI, because deliberate API changes are the point of a release candidate.
It returns to blocking when 1.1.0 ships.

Two properties matter more than the blocking behaviour:

- The baseline is a **published** version. Comparing against a git tag that was
  never published guards a contract that does not exist.
- Coverage is **every published library crate**, not a hand-picked subset. A
  crate excluded from the gate cannot break loudly; it can only break silently.
  A Tier 3 break is expected — but it should still be visible.

## Adding a crate

State the tier in the crate's `lib.rs` header and add it to the table above in
the same change. A crate with no stated tier is Tier 3 by default, because
promising less than you deliver is recoverable and the reverse is not.
