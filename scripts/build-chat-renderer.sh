#!/usr/bin/env bash
# Rebuild only our fixed no-import guest; model templates are runtime data.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'usage: build-chat-renderer.sh [--check|--write]' >&2; exit 2;; esac
manifest="$repo/crates/tritium-nn/chat-renderer/Cargo.toml"
# Use the existing project's final-output cache, not a fresh target per run.
target="$repo/target/chat-renderer"
export RUSTFLAGS="--remap-path-prefix=$repo=/tritium --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo -C link-arg=-z -C link-arg=stack-size=262144 -C link-arg=--max-memory=33554432"
timeout 300s cargo build --locked --manifest-path "$manifest" --target wasm32-unknown-unknown --release --jobs 2 --target-dir "$target"
generated="$target/wasm32-unknown-unknown/release/tritium_chat_renderer_guest.wasm"
packaged="$repo/crates/tritium-nn/src/model/chat_renderer.wasm"
if [[ "$mode" == --write ]]; then
    /usr/bin/cp -- "$generated" "$packaged"
    chmod 644 "$packaged"
    timeout 90s python "$repo/scripts/generate-chat-renderer-record.py" --write
else
    cmp -- "$generated" "$packaged"
    timeout 90s python "$repo/scripts/generate-chat-renderer-record.py"
fi
sha256sum "$packaged"
