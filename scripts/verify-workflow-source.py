#!/usr/bin/env python3
"""Fail closed if a GitHub workflow checkout differs from its receipt identity."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


REVISION = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()

    expected = args.expected_revision.lower()
    if not REVISION.fullmatch(expected):
        parser.error("expected revision must be a full 40-character Git object ID")

    try:
        actual = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as error:
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
