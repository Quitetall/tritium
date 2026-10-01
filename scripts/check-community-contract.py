#!/usr/bin/env python3
"""Check repository-owned governance and community routes without network access.

This gate proves local links, required contact routes, template evidence fields,
and the deliberate absence of unstaffed public channels. It does not turn an
external URL into a reachability claim; URL identity is checked syntactically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


GOVERNANCE_FILES = (
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "COMMUNITY.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/DISCUSSION_TEMPLATE/ideas.yml",
    ".github/DISCUSSION_TEMPLATE/q-a.yml",
    ".github/ISSUE_TEMPLATE/backend.yml",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/estimator.yml",
    ".github/ISSUE_TEMPLATE/model-evidence.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
)

REQUIRED_REPOSITORY_ROUTES = {
    "CONTRIBUTING.md": (
        "https://github.com/Quitetall/tritium-research",
    ),
    ".github/ISSUE_TEMPLATE/config.yml": (
        "https://github.com/Quitetall/tritium/security/policy",
        "https://github.com/Quitetall/tritium/blob/main/SUPPORT.md",
    ),
}
REQUIRED_CONTACTS = {
    "SECURITY.md": (("briankhanglam@gmail.com", "private security"),),
    "SUPPORT.md": (("GitHub issue tracker", "support route"),),
}
TEMPLATE_REQUIREMENTS = {
    ".github/PULL_REQUEST_TEMPLATE.md": ("Verification", "Evidence", "ADR"),
    ".github/DISCUSSION_TEMPLATE/ideas.yml": ("affected contracts", "Durable record"),
    ".github/DISCUSSION_TEMPLATE/q-a.yml": ("id: version", "environment", "artifact identity"),
    ".github/ISSUE_TEMPLATE/backend.yml": ("evidence_level", "Conformance and performance evidence"),
    ".github/ISSUE_TEMPLATE/bug.yml": ("id: version", "id: environment", "id: reproduce", "evidence_level"),
    ".github/ISSUE_TEMPLATE/estimator.yml": ("id: reference", "id: evidence", "evidence_level"),
    ".github/ISSUE_TEMPLATE/model-evidence.yml": ("id: identities", "id: measurements", "id: hardware", "evidence_level"),
    ".github/ISSUE_TEMPLATE/question.yml": ("id: version", "id: environment", "id: observed"),
}
UNSTAFFED_CHANNEL_RE = re.compile(
    r"(?:https?://)?(?:discord(?:\.gg|\.com/invite)|t\.me/|telegram\.me/|\b(?:x|twitter)\.com/|linkedin\.com/)",
    re.IGNORECASE,
)
INLINE_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\(\s*(?:<([^>]+)>|([^\s)]+))",
)
REFERENCE_LINK_RE = re.compile(
    r"^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))\s*$",
    re.MULTILINE,
)
MAX_FILE_BYTES = 2 * 1024 * 1024


class CommunityContractError(ValueError):
    """Repository governance contract is missing, stale, or unsafe."""


def _ordinary(repo: Path, relative: str) -> Path:
    path = repo / relative
    if path.is_symlink() or not path.is_file():
        raise CommunityContractError(f"required governance file missing: {relative}")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_FILE_BYTES:
        raise CommunityContractError(f"required governance file is unbounded: {relative}")
    return path


def _text(path: Path, relative: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise CommunityContractError(f"governance file is not UTF-8: {relative}") from error


def _targets(text: str):
    for match in INLINE_LINK_RE.finditer(text):
        yield match.group(1) or match.group(2)
    for match in REFERENCE_LINK_RE.finditer(text):
        yield match.group(1) or match.group(2)


def _github_anchors(text: str) -> set[str]:
    """Return GitHub-style Markdown heading slugs and explicit HTML IDs."""
    lines = text.splitlines()
    visible: list[str] = []
    fence: tuple[str, int] | None = None
    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            if (
                marker
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
            ):
                fence = None
            visible.append("")
            continue
        if marker:
            fence = (marker.group(1)[0], len(marker.group(1)))
            visible.append("")
            continue
        visible.append(line)

    anchors: set[str] = set()
    ids = re.finditer(
        r"\bid\s*=\s*(?:([\"'])([^\"']+)\1|([^\s>]+))",
        "\n".join(visible),
        re.IGNORECASE,
    )
    for match in ids:
        anchors.add((match.group(2) or match.group(3)).lower())

    used: set[str] = set(anchors)
    next_suffix: dict[str, int] = {}

    def add_heading(raw: str) -> None:
        raw = re.sub(r"<[^>]*>", " ", raw)
        raw = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", raw)
        raw = raw.replace("`", "").lower()
        slug = re.sub(r"[^\w\- ]", "", raw, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", slug.strip())
        if not slug:
            return
        candidate = slug
        suffix = next_suffix.get(slug, 0)
        while candidate in used:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        used.add(candidate)
        anchors.add(candidate)
        next_suffix[slug] = suffix

    index = 0
    while index < len(visible):
        line = visible[index]
        atx = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if atx:
            add_heading(atx.group(1))
        elif (
            line.strip()
            and index + 1 < len(visible)
            and re.fullmatch(r" {0,3}(?:=+|-+)\s*", visible[index + 1])
        ):
            add_heading(line.strip())
        index += 1
    return anchors


def _check_local_link(
    repo: Path, source: Path, source_relative: str, target: str
) -> bool:
    target = target.strip()
    if not target:
        return False
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return False
    if parsed.path.startswith("/"):
        raise CommunityContractError(
            f"absolute repository link is not portable: {source_relative}: {target}"
        )
    logical = unquote(parsed.path)
    resolved = (source.parent / logical).resolve() if logical else source.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise CommunityContractError(
            f"local link escapes repository: {source_relative}: {target}"
        ) from error
    if not resolved.exists():
        raise CommunityContractError(
            f"broken local governance link: {source_relative}: {target}"
        )
    fragment = unquote(parsed.fragment).lower()
    if fragment and resolved.suffix.lower() in {".md", ".markdown"}:
        try:
            anchors = _github_anchors(resolved.read_text(encoding="utf-8"))
        except UnicodeDecodeError as error:
            raise CommunityContractError(
                f"local Markdown link target is not UTF-8: {source_relative}: {target}"
            ) from error
        if fragment not in anchors:
            raise CommunityContractError(
                f"broken local Markdown anchor: {source_relative}: {target}"
            )
    return True


def _check_routes(relative: str, text: str) -> int:
    for route in REQUIRED_REPOSITORY_ROUTES.get(relative, ()):
        if route not in text:
            raise CommunityContractError(
                f"required repository route missing from {relative}: {route}"
            )
    for required, label in REQUIRED_CONTACTS.get(relative, ()):
        if required.lower() not in text.lower():
            raise CommunityContractError(
                f"required {label} route missing from {relative}: {required}"
            )
    return len(REQUIRED_REPOSITORY_ROUTES.get(relative, ()))


def check(repo: Path) -> dict[str, int | str]:
    """Validate governance files and return deterministic summary counters."""
    repo = repo.resolve(strict=True)
    local_links = 0
    repository_routes = 0
    for relative in GOVERNANCE_FILES:
        path = _ordinary(repo, relative)
        text = _text(path, relative)
        repository_routes += _check_routes(relative, text)
        for requirement in TEMPLATE_REQUIREMENTS.get(relative, ()):
            if requirement.lower() not in text.lower():
                raise CommunityContractError(
                    f"template evidence field missing from {relative}: {requirement}"
                )
        for target in _targets(text):
            if _check_local_link(repo, path, relative, target):
                local_links += 1

    community = _text(repo / "COMMUNITY.md", "COMMUNITY.md")
    channels = UNSTAFFED_CHANNEL_RE.findall(community)
    if channels:
        raise CommunityContractError(
            "unstaffed public community channel advertised in COMMUNITY.md"
        )
    if not re.search(r"only\s+active\s+project\s+discussion\s+surfaces", community, re.IGNORECASE):
        raise CommunityContractError(
            "COMMUNITY.md must declare active discussion surfaces explicitly"
        )
    if "security.md" not in community.lower():
        raise CommunityContractError("COMMUNITY.md must link private security route")
    return {
        "result": "pass",
        "governance_files": len(GOVERNANCE_FILES),
        "local_links": local_links,
        "repository_routes": repository_routes,
        "unstaffed_channels": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        report = check(args.repo)
    except (OSError, CommunityContractError) as error:
        print(f"community contract: FAIL: {error}")
        return 1
    print(json.dumps(report, sort_keys=True) if args.as_json else "community contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
