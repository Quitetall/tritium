#!/usr/bin/env python3
"""Resolve a release tag to a commit already reachable from the default branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence


TAG_PATTERN = re.compile(
    r"v(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}"
    r"(?:-rc\.(?:0|[1-9][0-9]*))?"
)
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class ReleaseRefError(ValueError):
    """The requested release tag is invalid, absent, or not reviewed."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or "git command failed"
        raise ReleaseRefError(detail)
    return result


def resolve_release_ref(
    repo: Path,
    tag: str,
    default_branch: str,
    *,
    fetch_remote: str | None = "origin",
) -> dict[str, str]:
    """Resolve only a canonical release tag whose commit is reachable from main."""

    if not isinstance(tag, str) or TAG_PATTERN.fullmatch(tag) is None:
        raise ReleaseRefError("tag must be canonical vX.Y.Z or vX.Y.Z-rc.N")
    branch_check = _git(repo, "check-ref-format", "--branch", default_branch, check=False)
    if branch_check.returncode != 0:
        raise ReleaseRefError("default branch is not a valid Git branch name")

    if fetch_remote is not None:
        _git(
            repo,
            "fetch",
            "--no-tags",
            fetch_remote,
            f"+refs/heads/{default_branch}:refs/remotes/{fetch_remote}/{default_branch}",
        )
        _git(
            repo,
            "fetch",
            "--no-tags",
            fetch_remote,
            f"+refs/tags/{tag}:refs/tags/{tag}",
        )

    tag_ref = f"refs/tags/{tag}^{{commit}}"
    commit_result = _git(repo, "rev-parse", "--verify", tag_ref, check=False)
    if commit_result.returncode != 0:
        raise ReleaseRefError(f"release tag {tag!r} is absent or does not name a commit")
    revision = commit_result.stdout.strip()
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ReleaseRefError("release tag did not resolve to a full commit ID")

    default_ref = f"refs/remotes/{fetch_remote or 'origin'}/{default_branch}"
    ancestry = _git(repo, "merge-base", "--is-ancestor", revision, default_ref, check=False)
    if ancestry.returncode == 1:
        raise ReleaseRefError(
            f"release tag {tag!r} is not reachable from {default_branch!r}"
        )
    if ancestry.returncode != 0:
        raise ReleaseRefError(
            ancestry.stderr.strip() or "cannot verify release-tag ancestry"
        )
    return {"tag": tag, "revision": revision}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--default-branch", required=True)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="use existing local refs (for tests and offline verification)",
    )
    args = parser.parse_args(argv)
    try:
        identity = resolve_release_ref(
            args.repo.resolve(strict=True),
            args.tag,
            args.default_branch,
            fetch_remote=None if args.no_fetch else "origin",
        )
    except (OSError, ReleaseRefError) as error:
        print(f"resolve-release-ref: ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
