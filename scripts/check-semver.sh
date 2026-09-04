#!/usr/bin/env bash
# check-semver.sh — public-API change report for the published library crates.
# (ADR 0011 hardening / ADR 0012 freeze; tiers in docs/STABILITY.md.)
#
# Compares each crate's public API against the version actually PUBLISHED on
# crates.io. It used to compare against the newest stable git tag, which was
# `v1.0.0` — a tag that was never published. crates.io has exactly one version of
# these crates, `1.1.0-rc.0`, so the tag baseline guarded a contract nobody could
# depend on while the real one went unchecked.
#
# MODE (`TRITIUM_SEMVER_MODE`):
#   report  default during the 1.1.0 release-candidate window. Runs the check,
#           prints the verdict, exits 0. Deliberate breaking changes are the
#           POINT of a release candidate; they belong in the CHANGELOG rc
#           section, not in a red CI lane.
#   block   exits non-zero on a breaking change. Make this the default again
#           when 1.1.0 ships, with 1.1.0 as the baseline.
#
# SCOPE: every published crate with a library API, per docs/STABILITY.md.
# Still EXCLUDED, each for a stated reason:
#   - tritium-cuda   the `cuda` feature needs nvcc, which CI does not have.
#                    Tier 3 promises no stability — but this is still the
#                    largest unwatched surface in the workspace.
#   - tritium-cli / tritium-benches  binaries; no library API to freeze.
#   - tritium-py     PyO3 cdylib — gated by wheel/API compatibility receipts.
#
# KNOWN GAP: `tritium-serve`'s surface sits behind the `serve`/`cuda` features,
# so a default-feature check sees almost none of it. That is a gap, not a pass.
#
# Usage:  ./scripts/check-semver.sh [baseline-rev]   # explicit git rev instead
#         ./scripts/check-semver.sh --print-baseline          # newest stable tag
#         ./scripts/check-semver.sh --print-baseline-version  # published baseline
# Env:    TRITIUM_SEMVER_BASELINE_VERSION (default 1.1.0-rc.0)
#         TRITIUM_SEMVER_MODE=report|block
#         TRITIUM_SEMVER_TOOLCHAIN=stable|1.95|...  (hermetic CI images)
# Needs:  cargo-semver-checks (cargo install cargo-semver-checks --locked).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Published baseline. `latest_stable_baseline` below is retained because an
# explicit git rev is still a supported comparison, and because the release-tag
# smoke depends on its tag-selection semantics.
DEFAULT_BASELINE_VERSION="1.1.0-rc.0"

CHECKED_CRATES=(
  tritium-core tritium-spec tritium-format tritium-runtime
  tritium-cpu tritium-quantize tritium-testkit tritium-ffi
  tritium-nn tritium-salt tritium-train tritium-serve
  tritium-burn tritium-candle tritium-onnx tritium-mcu
  tritium-wasm tritium-metal tritium-rocm tritium-wgpu
  tritium-build-info
)

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
  if [[ "${1:-}" == "--print-baseline-version" ]]; then
    printf '%s\n' "${TRITIUM_SEMVER_BASELINE_VERSION:-$DEFAULT_BASELINE_VERSION}"
    return 0
  fi

  local -a baseline_args
  if [[ -n "${1:-}" ]]; then
    baseline_args=(--baseline-rev "$1")
    echo "[check-semver] baseline rev: $1"
  else
    local version="${TRITIUM_SEMVER_BASELINE_VERSION:-$DEFAULT_BASELINE_VERSION}"
    baseline_args=(--baseline-version "$version")
    echo "[check-semver] baseline version (published on crates.io): $version"
  fi

  local toolchain="${TRITIUM_SEMVER_TOOLCHAIN:-stable}"
  local mode="${TRITIUM_SEMVER_MODE:-report}"
  echo "[check-semver] toolchain: $toolchain"
  echo "[check-semver] mode: $mode"
  echo "[check-semver] crates: ${#CHECKED_CRATES[@]}"

  local -a package_args=()
  local crate
  for crate in "${CHECKED_CRATES[@]}"; do
    package_args+=(-p "$crate")
  done

  if [[ "$mode" == "block" ]]; then
    exec cargo "+$toolchain" semver-checks "${baseline_args[@]}" "${package_args[@]}"
  fi
  if [[ "$mode" != "report" ]]; then
    echo "[check-semver] TRITIUM_SEMVER_MODE must be report or block, got '$mode'" >&2
    return 2
  fi

  local status=0
  cargo "+$toolchain" semver-checks "${baseline_args[@]}" "${package_args[@]}" || status=$?
  if (( status == 0 )); then
    echo "[check-semver] no breaking change against the published baseline."
    return 0
  fi
  echo "[check-semver] BREAKING CHANGES REPORTED (cargo-semver-checks exit $status)."
  echo "[check-semver] Not a failure during the release-candidate window: record"
  echo "[check-semver] them in the CHANGELOG rc section. Set TRITIUM_SEMVER_MODE=block"
  echo "[check-semver] to make this gate fail again."
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
