#!/usr/bin/env python3
"""Fail closed if a GitHub workflow checkout differs from its receipt identity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import runpy
import sys


REVISION = re.compile(r"^[0-9a-f]{40}$")
_run_git = runpy.run_path(Path(__file__).with_name("_qualification_git.py"))["run_git"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()

    expected = args.expected_revision.lower()
    if not REVISION.fullmatch(expected):
        parser.error("expected revision must be a full 40-character Git object ID")

    try:
        actual = _run_git(Path.cwd(), "rev-parse", "--verify", "HEAD^{commit}").lower()
    except ValueError as error:
        print(f"cannot resolve checked-out Git commit: {error}", file=sys.stderr)
        return 1

    if actual != expected:
        print(
            f"checked-out revision {actual} differs from receipt revision {expected}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: checked-out source revision is {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
