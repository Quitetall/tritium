#!/usr/bin/env bash
# Publish-readiness gate (ADR 0011 / v0.90): every workspace crate packages cleanly
# into a valid `.crate` — manifest valid, all referenced files present, internal
# deps version-pinned and ordered.
#
# Why `cargo package --workspace` and not per-crate `cargo publish --dry-run`:
# this workspace has not been published to crates.io, so a per-crate
# `cargo publish --dry-run` / `cargo package -p <x>` fails to "prepare for upload"
# because each crate's internal deps (e.g. `tritium-core = "0.6.x"`) aren't on
# crates.io yet (a chicken-and-egg only resolvable by an ordered release-time
# publish). `cargo package --workspace` packages all members together and resolves
# the internal deps locally, validating packaging for every crate at once. The
# verify-build is already covered by the cpu-only-green lane. `--frozen` makes
# archive assembly fail on lockfile drift or a network dependency; clean offline
# installation of the resulting archives remains the stricter local-RC gate.
set -euo pipefail

./scripts/check-release-version.py
echo "publish-readiness: cargo package --workspace --frozen --no-verify"
cargo package --workspace --frozen --no-verify
echo "OK: every workspace crate packages cleanly (publish-ready manifests + files)."
