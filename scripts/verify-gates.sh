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

verify_fmt() {
    # Commit hooks inspect staged content, not unrelated working-tree WIP.
    if git diff --cached --quiet; then
        run cargo fmt --all --check
        return
    fi

    snapshot=$(mktemp -d "${TMPDIR:-/tmp}/tritium-precommit.XXXXXX")
    trap 'rm -rf "$snapshot"' EXIT HUP INT TERM
    git archive --format=tar HEAD | tar -x -C "$snapshot"
    git diff --cached --binary | git -C "$snapshot" apply --binary -
    (cd "$snapshot" && run cargo fmt --all --check)
}

case "$tier" in
    precommit)
        run git diff --cached --check
        verify_fmt
        ;;
    prepush)
        run cargo fmt --all --check
        run cargo clippy --workspace --all-targets -- -D warnings
        ;;
    ci)
        run cargo fmt --all --check
        run cargo clippy --workspace --all-targets -- -D warnings
        # tritium-py is a PyO3 cdylib; standalone cargo-test binaries cannot
        # link Python without a development lib. Its shipped surface is gated
        # by the native wheel/pytest path below, matching CI's maturin lane.
        run cargo test --workspace --exclude tritium-py --no-fail-fast
        run python -m unittest discover -s scripts/tests -p 'test_*.py'
        run python scripts/check-community-contract.py --json
        ;;
    release)
        run cargo fmt --all --check
        # Type-check every feature-gated surface without requiring GPU toolkits.
        run env TRITIUM_CHECK_ONLY=1 cargo clippy --workspace --all-targets --all-features -- -D warnings
        run cargo test --workspace --exclude tritium-py --no-fail-fast
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
