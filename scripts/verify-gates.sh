#!/usr/bin/env sh
# Canonical local verification tiers. Hooks and release work call this entrypoint;
# CI keeps its platform-specific matrix commands explicit.

set -eu

tier=${1:-}
if [ -z "$tier" ]; then
    echo "usage: $0 <precommit|prepush|ci|release>" >&2
    exit 2
fi

run() {
    echo "+ $*" >&2
    "$@"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "required command missing: $1" >&2
        exit 1
    }
}

repo=$(git rev-parse --show-toplevel)
cd "$repo"

staged_snapshot=

cleanup_staged_snapshot() {
    if [ -n "$staged_snapshot" ]; then
        rm -rf "$staged_snapshot"
    fi
}

prepare_staged_snapshot() {
    if [ -n "$staged_snapshot" ]; then
        return
    fi
    staged_snapshot=$(mktemp -d "${TMPDIR:-/tmp}/tritium-precommit.XXXXXX")
    trap cleanup_staged_snapshot EXIT HUP INT TERM
    git archive --format=tar HEAD | tar -x -C "$staged_snapshot"
    git diff --cached --binary | git -C "$staged_snapshot" apply --binary -
}

verify_fmt() {
    # Commit hooks inspect staged content, not unrelated working-tree WIP.
    if git diff --cached --quiet; then
        run cargo fmt --all --check
        return
    fi

    prepare_staged_snapshot
    (cd "$staged_snapshot" && run cargo fmt --all --check)
}

verify_changed_sources() {
    git diff --cached --quiet && return
    prepare_staged_snapshot

    changed=$(git diff --cached --name-only --diff-filter=ACMR)
    python_files=
    shell_files=
    json_files=
    web_changed=0
    while IFS= read -r path; do
        [ -n "$path" ] || continue
        case "$path" in
            *.py) python_files="${python_files}
${path}" ;;
            *.sh) shell_files="${shell_files}
${path}" ;;
            *.json) json_files="${json_files}
${path}" ;;
        esac
        case "$path" in
            packages/tritium-web/*.json|packages/tritium-web/**/*.json|\
            packages/tritium-web/*.js|packages/tritium-web/**/*.js|\
            packages/tritium-web/*.mjs|packages/tritium-web/**/*.mjs|\
            packages/tritium-web/*.ts|packages/tritium-web/**/*.ts|\
            packages/tritium-web/*.tsx|packages/tritium-web/**/*.tsx)
                web_changed=1 ;;
        esac
    done <<CHANGED
$changed
CHANGED

    if [ -n "$python_files" ]; then
        require_command python
        while IFS= read -r path; do
            [ -n "$path" ] || continue
            (cd "$staged_snapshot" && run python -m py_compile "$path")
        done <<PYTHON
$python_files
PYTHON
    fi

    if [ -n "$shell_files" ]; then
        require_command bash
        while IFS= read -r path; do
            [ -n "$path" ] || continue
            (cd "$staged_snapshot" && run bash -n "$path")
        done <<SHELL
$shell_files
SHELL
    fi

    if [ -n "$json_files" ]; then
        require_command python
        while IFS= read -r path; do
            [ -n "$path" ] || continue
            (cd "$staged_snapshot" && run python -c \
                'import json, pathlib, sys; json.loads(pathlib.Path(sys.argv[1]).read_text())' \
                "$path")
        done <<JSON
$json_files
JSON
    fi

    if [ "$web_changed" -eq 1 ]; then
        require_command npm
        # The package's WASM release check requires a clean Git tree. Build an
        # independent temporary repository inside the staged snapshot; never
        # call git worktree while the caller's commit index is locked.
        git -C "$staged_snapshot" init --quiet
        printf '%s\n' 'packages/tritium-web/node_modules' >>"$staged_snapshot/.gitignore"
        git -C "$staged_snapshot" add -A
        git -C "$staged_snapshot" \
            -c user.name=tritium-precommit \
            -c user.email=precommit@invalid \
            commit --quiet --no-verify -m "pre-commit staged snapshot"
        if [ -d "$repo/packages/tritium-web/node_modules" ]; then
            ln -s "$repo/packages/tritium-web/node_modules" \
                "$staged_snapshot/packages/tritium-web/node_modules"
        fi
        (
            cd "$staged_snapshot"
            run npm --prefix packages/tritium-web run check
        )
    fi
}

case "$tier" in
    precommit)
        run git diff --cached --check
        verify_fmt
        verify_changed_sources
        ;;
    prepush)
        run cargo fmt --all --check
        run cargo clippy --locked --workspace --all-targets -- -D warnings
        ;;
    ci)
        run cargo fmt --all --check
        run cargo clippy --locked --workspace --all-targets -- -D warnings
        # tritium-py is a PyO3 cdylib; standalone cargo-test binaries cannot
        # link Python without a development lib. Its shipped surface is gated
        # by the native wheel/pytest path below, matching CI's maturin lane.
        run cargo test --locked --workspace --exclude tritium-py --no-fail-fast
        run python -m unittest discover -s scripts/tests -p 'test_*.py'
        run python scripts/check-community-contract.py --json
        ;;
    release)
        run cargo fmt --all --check
        # Type-check every feature-gated surface without requiring GPU toolkits.
        run env TRITIUM_CHECK_ONLY=1 cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
        run cargo test --locked --workspace --exclude tritium-py --no-fail-fast
        run python -m unittest discover -s scripts/tests -p 'test_*.py'
        run python scripts/check-community-contract.py --json
        require_command cargo-deny
        run cargo deny check
        run python scripts/check-release-version.py
        run ./scripts/check-semver.sh
        ;;
    *)
        echo "unknown verification tier: $tier" >&2
        exit 2
        ;;
esac
