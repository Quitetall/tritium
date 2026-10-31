#!/usr/bin/env bash
# Explicit external-reference replay of bounded synthetic native CPU output.
# No installation, downloads, catalog activation or qualification receipts.
set -euo pipefail
if test "$#" -ne 4 && test "$#" -ne 5; then
    echo "usage: $0 PYTHON REFERENCE_SCRIPT REFERENCE_SHA256 NEW_OUTPUT_DIRECTORY [tritium-cpu|tritium-salt]" >&2
    exit 2
fi
python=$1
reference=$2
expected_reference=$3
output=$4
producer=${5:-tritium-cpu}
case "$producer" in
    tritium-cpu|tritium-salt) ;;
    *) echo 'unsupported conformance producer; no fallback' >&2; exit 2 ;;
esac
repo=$(timeout 10 git rev-parse --show-toplevel)
test -x "$python"
test -f "$reference"
[[ "$expected_reference" =~ ^[a-f0-9]{64}$ ]]
test "$(sha256sum "$reference" | cut -d ' ' -f 1)" = "$expected_reference"
test ! -e "$output"
test ! -L "$output"
test -d "$(dirname "$output")"
test "$(realpath -m "$output")" = "$output"
test -z "$(timeout 10 git -C "$repo" status --porcelain)"
expected_product=$(timeout 10 git -C "$repo" rev-parse HEAD)
case "$output" in
    /mnt/4tb/tmp/tritium-*/?*) ;;
    *) echo 'output must be a new child of Tritium scratch on /mnt/4tb/tmp' >&2; exit 2 ;;
esac
disk=$(timeout 5 df -P /mnt/4tb | awk 'NR==2 {gsub(/%/,"",$5);print $5}')
test "$disk" -lt 90
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$repo"
timeout 600 cargo run --locked --offline -p "$producer" --example qtea_conformance -- "$output"
shopt -s nullglob dotglob
produced=("$output"/*)
test "${#produced[@]}" = 32
count=0
for case in 000 001 002 003 004 005 006 007; do
    tensor="$output/case-$case.qtea"
    test -f "$tensor" && test ! -L "$tensor"
    sha=$(sha256sum "$tensor" | cut -d ' ' -f 1)
    for batch in 0 1 3; do
        observation="$output/case-$case.batch-$batch.json"
        test -f "$observation" && test ! -L "$observation"
        timeout 30 "$python" -I -B "$reference" "$tensor" \
            --observation "$observation" --expected-sha256 "$sha" \
            > "$output/case-$case.batch-$batch.reference.json"
        count=$((count+1))
    done
done
test "$count" = 24
produced=("$output"/*)
test "${#produced[@]}" = 56
test "$(timeout 10 git rev-parse HEAD)" = "$expected_product"
test -z "$(timeout 10 git status --porcelain)"
test "$(sha256sum "$reference" | cut -d ' ' -f 1)" = "$expected_reference"
printf 'QTEA_NATIVE_REFERENCE_MATCH product=%s producer=%s checker_sha256=%s observations=%s (not qualification)\n' \
    "$expected_product" "$producer" "$expected_reference" "$count"
