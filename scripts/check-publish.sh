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

cd "$(dirname "${BASH_SOURCE[0]}")/.."

./scripts/check-release-version.py
unpublished=$(
  cargo metadata --locked --no-deps --format-version 1 |
    # cargo metadata encodes `publish = false` as []; null means publishable.
    python3 -c 'import json,sys; print("\n".join(p["name"] for p in json.load(sys.stdin)["packages"] if p.get("publish") == []))'
)
exclude_args=()
while IFS= read -r package; do
  [[ -z "$package" ]] && continue
  exclude_args+=(--exclude "$package")
done <<< "$unpublished"
echo "publish-readiness: cargo package --workspace --frozen --no-verify (publishable crates only)"
cargo package --workspace "${exclude_args[@]}" --frozen --no-verify
echo "OK: every publishable workspace crate packages cleanly (manifests + files)."
