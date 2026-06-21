#!/usr/bin/env bash
# check-semver.sh — public-API stability gate (ADR 0011 hardening / ADR 0012 freeze).
#
# Asserts the stable library crates have no UNINTENTIONAL breaking API change vs a
# baseline git ref (default: the latest v0.5.* release tag reachable from HEAD).
# Pre-1.0 breaking changes are *allowed* but must be deliberate — this surfaces
# them so a reviewer signs off + bumps the version accordingly.
#
# Scope: the stable, GPU-free public-API crates. Deliberately EXCLUDED for now:
#   - tritium-cuda  (needs nvcc + a GPU to build the cuda feature)
#   - tritium-cli / tritium-benches (binaries — no library API to freeze)
#   - tritium-py    (PyO3 cdylib — its ABI is gated by the v0.80 ffi/py work)
#   - tritium-nn / tritium-train (API still in flux during the perf/training work)
# Widen this list as those crates' surfaces stabilize toward v1.0.
#
# Usage:  ./scripts/check-semver.sh [baseline-rev]
# Needs:  cargo-semver-checks (cargo install cargo-semver-checks --locked).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASELINE="${1:-$(git describe --tags --abbrev=0 --match 'v0.5.*' 2>/dev/null || echo v0.5.6)}"
echo "[check-semver] baseline rev: $BASELINE"

exec cargo semver-checks --baseline-rev "$BASELINE" \
  -p tritium-core \
  -p tritium-spec \
  -p tritium-format \
  -p tritium-runtime \
  -p tritium-cpu \
  -p tritium-quantize \
  -p tritium-testkit
