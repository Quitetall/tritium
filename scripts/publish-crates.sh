#!/usr/bin/env bash
# Ordered crates.io publication for the Tritium workspace (plan 0053 / launch).
#
# IRREVERSIBLE: crates.io versions cannot be deleted, only yanked. Run only on
# the exact revision being released, from a clean tree, after ./scripts/
# check-publish.sh passes and CI is green on that revision.
#
# Order is the dependency topological order (deps first) — a crate can only be
# published once every internal dependency it names is live on the registry.
# crates.io indexes new versions within seconds, but under load a downstream
# publish can race the index; the retry loop below absorbs that.
#
# Excluded: tritium-py (PyPI is its registry: tritium-torch wheels),
# tritium-benches (publish = false).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ORDER=(
  tritium-build-info
  tritium-core
  tritium-format
  tritium-mcu
  tritium-quantize
  tritium-spec
  tritium-testkit
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

if [[ "${1:-}" != "--yes-publish" ]]; then
  echo "DRY RUN preflight: packaging every crate in publish order (no upload)."
  echo "To actually publish: $0 --yes-publish"
  cargo package --workspace --exclude tritium-py --exclude tritium-benches --frozen --no-verify
  echo "PREFLIGHT OK: all ${#ORDER[@]} crates package cleanly. Re-run with --yes-publish."
  exit 0
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "REFUSING: working tree is dirty. Publish only from the release revision." >&2
  exit 1
fi

for crate in "${ORDER[@]}"; do
  echo "=== publishing ${crate} ==="
  for attempt in 1 2 3 4 5; do
    if cargo publish -p "${crate}" --no-verify; then
      break
    fi
    if [[ "${attempt}" == 5 ]]; then
      echo "FAILED after 5 attempts: ${crate} — stopping (already-published crates stay live)." >&2
      exit 1
    fi
    echo "retry ${attempt}: waiting for the registry index..."
    sleep 30
  done
done

echo "ALL ${#ORDER[@]} crates published. Record the revision in the release evidence."
