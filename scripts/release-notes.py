#!/usr/bin/env python3
"""Render GitHub Release notes for one version from CHANGELOG.md.

The release workflow needs notes it can generate unattended. Two properties matter
more than prettiness:

* **A missing section is not an error.** Release candidates are cut between
  CHANGELOG entries -- `1.1.0-rc.0` shipped to crates.io with no section at all --
  so falling back to the Unreleased body keeps a tag publishable instead of failing
  the release on a documentation gap.
* **The heading is matched, not the whole line.** Entries carry a trailing date and
  title (`## [1.0.0] - 2026-06-28 - v1.0 Release`), and those drift.

Usage: release-notes.py <version>      # version without a leading `v`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
REPO = "https://github.com/Quitetall/tritium"


def sections(text: str) -> list[tuple[str, str]]:
    """Split CHANGELOG into (heading, body) pairs, in file order."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    out = []
    for n, i in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        out.append((lines[i], "\n".join(lines[i + 1 : end]).strip()))
    return out


def find(text: str, version: str) -> tuple[str | None, bool]:
    """Return (body, exact). `exact` is False when the Unreleased fallback was used."""
    want = re.escape(version)
    for heading, body in sections(text):
        if re.match(rf"^##\s*\[{want}\]", heading):
            return body, True
    for heading, body in sections(text):
        if re.match(r"^##\s*\[Unreleased\]", heading, flags=re.IGNORECASE):
            return body, False
    return None, False


def render(version: str, body: str | None, exact: bool) -> str:
    parts = []
    if not exact:
        parts.append(
            f"_No CHANGELOG section for {version} yet; these are the current "
            f"unreleased notes._\n"
        )
    if body:
        parts.append(body)
    else:
        parts.append(f"Release {version}.")
    parts.append(
        f"\n---\n\nFull changelog: [CHANGELOG.md]({REPO}/blob/main/CHANGELOG.md)"
    )
    return "\n".join(parts).strip() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[1].lstrip("v")
    body, exact = find(CHANGELOG.read_text(encoding="utf-8"), version)
    sys.stdout.write(render(version, body, exact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
