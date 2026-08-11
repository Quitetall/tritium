#!/usr/bin/env bash
# check-semver.sh — public-API stability gate (ADR 0011 hardening / ADR 0012 freeze).
#
# Asserts the stable library crates have no UNINTENTIONAL breaking API change vs a
# baseline git ref (default: the latest stable SemVer tag reachable from HEAD).
# Stable-line breaking changes fail this gate and require a major version.
#
# Scope: the stable, GPU-free public-API crates. Deliberately EXCLUDED for now:
#   - tritium-cuda  (needs nvcc + a GPU to build the cuda feature)
#   - tritium-cli / tritium-benches (binaries — no library API to freeze)
#   - tritium-py    (PyO3 cdylib — gated by wheel/API compatibility receipts)
#   - tritium-nn / tritium-train (the documented evolving 1.x tier)
#
# Usage:  ./scripts/check-semver.sh [baseline-rev]
#         ./scripts/check-semver.sh --print-baseline
# Needs:  cargo-semver-checks (cargo install cargo-semver-checks --locked).
# The checker can require a newer compiler than this workspace's declared MSRV;
# run it with the newest installed stable toolchain by default, while retaining
# an override for hermetic CI images (`TRITIUM_SEMVER_TOOLCHAIN=1.95`).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

latest_stable_baseline() {
  local tag
  while IFS= read -r tag; do
    if [[ "$tag" =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
      # A release-tag smoke must still compare with the previous stable API;
      # comparing a tag with itself would turn the gate into a no-op. Once HEAD
      # advances beyond the tag, that tag becomes the next development baseline.
      if [[ "$(git rev-parse "$tag^{commit}")" == "$(git rev-parse HEAD)" ]]; then
        continue
      fi
      printf '%s\n' "$tag"
      return 0
    fi
  done < <(git tag --merged HEAD --sort=-version:refname)
  echo "[check-semver] no stable SemVer baseline tag is reachable from HEAD" >&2
  return 1
}

main() {
  if [[ "${1:-}" == "--print-baseline" ]]; then
    latest_stable_baseline
    return 0
  fi
  local baseline="${1:-$(latest_stable_baseline)}"
  echo "[check-semver] baseline rev: $baseline"

  local toolchain="${TRITIUM_SEMVER_TOOLCHAIN:-stable}"
  echo "[check-semver] toolchain: $toolchain"
  exec cargo "+$toolchain" semver-checks --baseline-rev "$baseline" \
    -p tritium-core \
    -p tritium-spec \
    -p tritium-format \
    -p tritium-runtime \
    -p tritium-cpu \
    -p tritium-quantize \
    -p tritium-testkit
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
