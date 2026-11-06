#!/usr/bin/env bash
# Rebuild only our fixed no-import guest; model templates are runtime data.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'usage: build-chat-renderer.sh [--check|--write]' >&2; exit 2;; esac
manifest="$repo/crates/tritium-nn/chat-renderer/Cargo.toml"
# Use the existing project's final-output cache, not a fresh target per run.
target="$repo/target/chat-renderer"
sysroot=$(timeout 15s rustc --print sysroot)
rustc_commit=$(timeout 15s rustc --version --verbose | sed -n 's/^commit-hash: //p')
[[ "$rustc_commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'rustc commit identity is unavailable' >&2; exit 1; }
# Generic std functions are monomorphized here, so their panic locations can
# carry the local rustup path even though the registry and checkout are remapped.
export RUSTFLAGS="--remap-path-prefix=$repo=/tritium --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$sysroot=/rust-toolchain --remap-path-prefix=$sysroot/lib/rustlib/src/rust=/rustc/$rustc_commit -C link-arg=-z -C link-arg=stack-size=262144 -C link-arg=--max-memory=33554432"
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
