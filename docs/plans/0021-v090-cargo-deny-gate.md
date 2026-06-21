# 0021 — Supply-chain gate (cargo-deny) + publishable internal dep pins  (serves: v0.90 / ADR 0011; point release `v0.5.8`)

> Plan number is **0021**: `0020` was taken by `0020-decode-architectural-optimization.md` (a parallel
> core-optimization plan committed concurrently); this plan yielded the number to avoid the collision.

## Goal

Make `cargo deny check` pass and wire it as a CI lane — the supply-chain half of ADR 0011 hardening,
reachable now, zero hardware. Success: `cargo deny check licenses bans sources` is green locally, a
`cargo-deny` CI job runs the full check (incl. advisories) on every push/PR, and the fix that unblocks
the `bans` wildcard check also advances v1.0 publish-readiness.

## Context (verified)

`cargo-deny 0.16.4` is installed; `deny.toml` exists but `cargo deny check` **failed** on two real
gaps:
- **licenses**: `Unicode-3.0` (the `unicode-ident` transitive dep's license) was not in the allow-list.
- **bans**: the 10 internal `tritium-*` crates (used as workspace deps) tripped `wildcards = "deny"`. They are declared as bare
  `path` deps → wildcard version `*`. `allow-wildcard-paths = true` does **not** help: cargo-deny
  refuses it for *publishable* crates (crates.io forbids path deps). The correct fix is to pin each
  internal dep with a `version` — which is **also required** for the v1.0 `cargo publish` gate.

Advisories cannot be verified locally: cargo-deny 0.16.4 chokes parsing a CVSS-4.0 advisory
(`RUSTSEC-2026-0146`) in the shared local DB. The CI action pins a newer cargo-deny that handles it, so
CI is the authoritative advisory gate.

## Changes
- **`deny.toml`**: add `"Unicode-3.0"` to `[licenses] allow` (attributed in NOTICE like other upstreams).
- **`Cargo.toml` `[workspace.dependencies]`**: pin internal deps `{ path = "...", version = "0.5.7" }`.
  Caret `^0.5.7` resolves across the whole 0.5.x build-ahead line (zero churn; only a `0.60.0`-style
  minor bump updates it, in that release commit). Keeps `wildcards = "deny"` meaningful for *external*
  deps and unblocks `cargo publish`.
- **`.github/workflows/ci.yml`**: a `cargo-deny` job (`EmbarkStudios/cargo-deny-action@v2`,
  `command: check licenses bans sources advisories`) on every push + PR.

## Gate
```
cargo deny check licenses bans sources   # -> bans ok, licenses ok, sources ok (advisories: CI-only)
cargo check --workspace                  # resolves + builds clean with the version pins
```
(`license-not-encountered` warnings for unused forward-looking allowances are benign; the check exits 0.)

## Done criterion
`cargo deny check` green locally (sans advisories); CI lane added; internal deps version-pinned;
version `0.5.8`; CHANGELOG + ROADMAP updated; reviewed; tagged `v0.5.8` + pushed.
