#!/usr/bin/env bash
# Ordered crates.io publication for the Tritium workspace (plan 0053 / launch).
#
# IRREVERSIBLE: crates.io versions cannot be deleted, only yanked. Run only on
# the exact revision being released, after ./scripts/check-publish.sh passes
# and CI is green on that revision.
#
# Order is the dependency topological order (deps first) over ALL retained dep
# kinds — including dev-dependencies that carry a version requirement (cargo
# keeps those in the published manifest, and the registry rejects manifests
# naming crates it does not know). Verified against `cargo metadata`:
# tritium-testkit must precede tritium-mcu (mcu dev-depends on testkit).
#
# crates.io rate-limits NEW crate names (a small burst, then ~1 per 10
# minutes). A cold 23-crate launch therefore takes hours; the loop below waits
# out rate limits instead of aborting, and skips versions that are already
# live so the script is resumable after any interruption.
#
# Excluded: tritium-py (PyPI is its registry: tritium-torch wheels),
# tritium-benches (publish = false).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')"

ORDER=(
  tritium-build-info
  tritium-core
  tritium-format
  tritium-quantize
  tritium-spec
  tritium-testkit
  tritium-mcu
  tritium-train
  tritium-wasm
  tritium-burn
  tritium-candle
  tritium-runtime
  tritium-wgpu
  tritium-cpu
  tritium-cuda
  tritium-metal
  tritium-nn
  tritium-onnx
  tritium-rocm
  tritium-salt
  tritium-serve
  tritium-cli
  tritium-ffi
)

# True when $1@$VERSION is already live on crates.io (sparse index; the path
# scheme for names >= 4 chars is <first-2>/<next-2>/<name>).
already_published() {
  local name="$1"
  local prefix="${name:0:2}/${name:2:2}"
  curl -sf "https://index.crates.io/${prefix}/${name}" 2>/dev/null \
    | grep -q "\"vers\":\"${VERSION}\""
}

if [[ "${1:-}" != "--yes-publish" ]]; then
  echo "DRY RUN preflight: packaging every crate in publish order (no upload)."
  echo "To actually publish: $0 --yes-publish"
  cargo package --workspace --exclude tritium-py --exclude tritium-benches --frozen --no-verify
  echo "PREFLIGHT OK: all ${#ORDER[@]} crates package cleanly. Re-run with --yes-publish."
  exit 0
fi

# The published crate directories must be clean; unrelated dirt (e.g. tritium-py
# python sources, scripts/) is tolerated and reported.
DIRT="$(git status --porcelain)"
if [[ -n "${DIRT}" ]]; then
  BAD=""
  while IFS= read -r line; do
    f="${line:3}"
    case "$f" in
      crates/tritium-py/*|crates/tritium-benches/*|benches/*|scripts/*|docs/*) ;;
      *) BAD+="${line}"$'\n' ;;
    esac
  done <<< "${DIRT}"
  if [[ -n "${BAD}" ]]; then
    echo "REFUSING: uncommitted changes touch published crates:" >&2
    echo "${BAD}" >&2
    exit 1
  fi
  echo "note: tree has dirt outside the published crates (tolerated):"
  echo "${DIRT}"
fi

for crate in "${ORDER[@]}"; do
  if already_published "${crate}"; then
    echo "=== ${crate}@${VERSION} already live — skipping ==="
    continue
  fi
  echo "=== publishing ${crate}@${VERSION} ==="
  # The new-crate rate limit (burst of ~5, refill ~1 per 10 min) surfaces as a
  # CDN 503, not a labeled 429 — so after one quick retry, every further
  # failure waits out a refill interval. The cap only stops persistent
  # failures that survive multiple full refill windows.
  attempt=0
  until out="$(cargo publish -p "${crate}" --no-verify 2>&1)"; do
    echo "${out}" | grep -E "^error|Caused by|response, got" | head -3
    attempt=$((attempt + 1))
    if echo "${out}" | grep -qi "already uploaded\|already exists"; then
      echo "${crate}: version already on the registry — continuing."
      break
    fi
    if [[ "${attempt}" -ge 8 ]]; then
      echo "FAILED after ${attempt} attempts: ${crate} — stopping." >&2
      echo "Already-published crates stay live; re-run to resume from here." >&2
      exit 1
    fi
    # A CDN error page can mask an upload that actually landed — recheck.
    if already_published "${crate}"; then
      echo "${crate}: registry shows the version live despite the error — continuing."
      break
    fi
    if [[ "${attempt}" -eq 1 ]]; then
      echo "retry 1: waiting 30s..."
      sleep 30
    else
      echo "retry ${attempt}: waiting 620s (new-crate rate-limit refill)..."
      sleep 620
    fi
  done
  [[ -n "${out:-}" ]] && echo "${out}" | tail -2
done

echo "ALL ${#ORDER[@]} crates published at ${VERSION}. Record the revision in the release evidence."
