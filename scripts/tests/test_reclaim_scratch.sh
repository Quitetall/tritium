#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/tritium-reclaim-test.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT

mkdir "$scratch/stage7-campaign-preserve" \
    "$scratch/stage7-17b-feasibility" \
    "$scratch/qwen36-offload-capture" \
    "$scratch/ordinary-build-cache"
touch "$scratch/stage7-campaign-preserve/receipt" \
    "$scratch/stage7-17b-feasibility/receipt" \
    "$scratch/qwen36-offload-capture/receipt" \
    "$scratch/ordinary-build-cache/build"
touch -d '9 days ago' "$scratch/stage7-campaign-preserve" \
    "$scratch/stage7-17b-feasibility" \
    "$scratch/qwen36-offload-capture" \
    "$scratch/ordinary-build-cache"

default_report="$("$ROOT/scripts/reclaim-scratch.sh" --scratch "$scratch" --keep-days 7)"
grep -Fq 'KEEP (campaign/evidence preservation' <<<"$default_report"
grep -Fq 'stage7-campaign-preserve' <<<"$default_report"
grep -Fq 'stage7-17b-feasibility' <<<"$default_report"
grep -Fq 'qwen36-offload-capture' <<<"$default_report"
grep -Fq 'would remove' <<<"$default_report"
grep -Fq 'ordinary-build-cache' <<<"$default_report"

override_report="$("$ROOT/scripts/reclaim-scratch.sh" --scratch "$scratch" \
    --keep-days 7 --include-campaigns)"
grep -Fq 'would remove' <<<"$override_report"
grep -Fq 'stage7-campaign-preserve' <<<"$override_report"
grep -Fq 'stage7-17b-feasibility' <<<"$override_report"
grep -Fq 'qwen36-offload-capture' <<<"$override_report"

printf '%s\n' 'PASS: old campaign evidence is preserved by default and explicit override is visible.'
