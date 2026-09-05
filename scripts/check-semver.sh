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
# FEATURE SET: --default-features on both sides. cargo-semver-checks defaults to
# --all-features, which turns on `cuda` / `rocm` and runs those build scripts. The
# BASELINE is the published 1.1.0-rc.0, whose tritium-cuda / tritium-rocm build.rs
# panic when nvcc / hipcc are absent and predate the TRITIUM_CHECK_ONLY escape --
# published code cannot be patched, so no environment variable can make the
# baseline build on a hosted runner. Default features compare the surface a
# default consumer sees, symmetrically, and the baseline builds anywhere.
# Revisit when the baseline is a version whose build scripts honour
# TRITIUM_CHECK_ONLY (1.1.0-rc.2 onward): --all-features becomes possible again.
# That revisit is ENFORCED, not remembered — see the lift check in main(). A
# workaround whose expiry depends on someone recalling why it exists is how a
# temporary narrowing becomes permanent.
#
# KNOWN GAP, and it is wider than one crate. Under --default-features every
# feature-gated surface goes unchecked, not just `tritium-serve`'s: `serve`,
# `cuda`, `rocm`, `nccl`, `e2e` and `device-loss-qualification` are all opt-in
# (`default = []`), so for tritium-{serve,cuda,rocm} the gate currently sees
# close to nothing. It is a gap rather than a pass, and — this is the part worth
# stating out loud — a NARROWED gate and a FULLY-COVERED one produce the same
# green lane. Five checks were found vacuous in this repository during the week
# this was written, every one of them a case where a failure or a non-result was
# indistinguishable from a pass.
#
# RELEASE TYPE: --release-type minor, and without it this gate checks NOTHING.
# cargo-semver-checks derives the comparison from the version pair, and it
# classifies 1.1.0-rc.0 -> 1.1.0-rc.2 as a MAJOR change — under SemVer a major
# bump permits any breaking change, so every lint is skipped as irrelevant and
# the tool correctly reports "no semver update required". Observed directly:
#
#   Checking tritium-core v1.1.0-rc.0 -> v1.1.0-rc.2 (major change)
#    Checked [0.000s] 0 checks: 0 pass, 253 skip
#    Summary no semver update required
#
# That is a green lane over zero executed lints, for EVERY crate, for the whole
# release-candidate series. Forcing the minor lint set asks the question we
# actually care about — "did anything break since the last published rc?" —
# and makes the checks run:
#
#   Checking tritium-core v1.1.0-rc.0 -> v1.1.0-rc.2 (assume minor change)
#    Checked [0.022s] 196 checks: 196 pass, 57 skip
#
# `minor` rather than `patch` on purpose: additions are expected during an rc,
# breakage is what we want surfaced. Note this interacts with the baseline
# choice — comparing against the old v1.0.0 git tag produced a minor delta and
# so ran lints by accident; moving the baseline onto the published rc.0 is what
# made the version pair major and silently emptied the gate.
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

# cargo-semver-checks builds rustdoc for every crate with ALL features, which
# drags in tritium-cuda's build script (nvcc) and tritium-rocm's (hipcc). On a
# machine without them the build script fails, rustdoc never runs, and
# cargo-semver-checks exits 101 — so this gate's answer would otherwise depend on
# whether the machine happens to have a CUDA toolkit installed. A gate that is
# not deterministic across machines is not a gate.
#
# Forcing this cannot change the verdict: semver-checks reads API signatures out
# of rustdoc JSON, never compiled kernel bytes. It is exported rather than
# defaulted precisely because an override would reintroduce the nondeterminism.
#
# This was not a theoretical concern. Before the failure/finding split below, a
# 101 here was reported as "BREAKING CHANGES REPORTED" and exited 0, so the CI
# lane passed while checking nothing — for as long as the lane had existed.
export TRITIUM_CHECK_ONLY=1

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
    # Expiry guard for --default-features. The narrowing exists only because the
    # published 1.1.0-rc.0 and -rc.1 build scripts predate the TRITIUM_CHECK_ONLY
    # escape and therefore cannot build without nvcc/hipcc. From 1.1.0-rc.2
    # onward they honour it, so the reason evaporates — and a narrowed gate looks
    # exactly like a covered one from CI. Fail loudly at that point rather than
    # trusting anyone to notice.
    if ! [[ "$version" =~ ^1\.1\.0-rc\.[01]$ ]]; then
      echo "[check-semver] baseline $version postdates the --default-features workaround." >&2
      echo "[check-semver] Its only justification was that 1.1.0-rc.0/rc.1 build scripts" >&2
      echo "[check-semver] predate TRITIUM_CHECK_ONLY and cannot build without nvcc/hipcc." >&2
      echo "[check-semver] Drop --default-features from both semver-checks invocations so" >&2
      echo "[check-semver] the feature-gated surfaces (serve, cuda, rocm) are checked again," >&2
      echo "[check-semver] then delete this guard and the FEATURE SET note in the header." >&2
      return 2
    fi
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
    exec cargo "+$toolchain" semver-checks --default-features --release-type minor "${baseline_args[@]}" "${package_args[@]}"
  fi
  if [[ "$mode" != "report" ]]; then
    echo "[check-semver] TRITIUM_SEMVER_MODE must be report or block, got '$mode'" >&2
    return 2
  fi

  local status=0
  cargo "+$toolchain" semver-checks --default-features --release-type minor "${baseline_args[@]}" "${package_args[@]}" || status=$?
  if (( status == 0 )); then
    echo "[check-semver] no breaking change against the published baseline."
    return 0
  fi
  # Report mode tolerates FINDINGS, never FAILURES. cargo-semver-checks exits 1
  # when it ran and found breaking changes; any other status means it could not
  # run at all (absent binary, a baseline version not on the registry, a crate
  # that will not build). Swallowing that would make this lane green while
  # checking nothing — the exact failure mode a report-only gate invites, and
  # indistinguishable from a clean pass to anyone reading CI.
  if (( status != 1 )); then
    echo "[check-semver] cargo-semver-checks FAILED TO RUN (exit $status)." >&2
    echo "[check-semver] This is a broken gate, not a clean result, so it fails" >&2
    echo "[check-semver] even in report mode. Check that cargo-semver-checks is" >&2
    echo "[check-semver] installed and that the baseline version exists on the" >&2
    echo "[check-semver] registry for every crate listed above." >&2
    return "$status"
  fi
  echo "[check-semver] BREAKING CHANGES REPORTED (cargo-semver-checks exit 1)."
  echo "[check-semver] Not a failure during the release-candidate window: record"
  echo "[check-semver] them in the CHANGELOG rc section. Set TRITIUM_SEMVER_MODE=block"
  echo "[check-semver] to make this gate fail again."
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
