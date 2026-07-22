#!/usr/bin/env python3
"""Generate/check non-circular release claim blocks from frozen zoo policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy


ZOO = runpy.run_path(Path(__file__).with_name("verify-zoo-community-receipt.py"))
EXPECTED_MODELS = ZOO["EXPECTED_MODELS"]
COMPATIBILITY = runpy.run_path(Path(__file__).with_name("generate-compatibility.py"))

BEGIN = "<!-- BEGIN TRITIUM GENERATED RELEASE CLAIMS -->"
END = "<!-- END TRITIUM GENERATED RELEASE CLAIMS -->"
DOCUMENTS = (
    "README.md", "docs/book/src/model-zoo.md", "docs/book/src/benchmarks.md"
)


class ClaimGenerationError(ValueError):
    """Release claim documents are missing, malformed, or stale."""


def model_rows() -> list[str]:
    return [
        f"| `{tier}` | `{role}` | `{model_id}` | candidate artifact + admitted receipt ancestry |"
        for tier, role, model_id in EXPECTED_MODELS
    ]


def blocks() -> dict[str, str]:
    table = [
        "| Tier | Role | Frozen model | Admission rule |",
        "|---|---|---|---|",
        *model_rows(),
    ]
    return {
        "README.md": "\n".join([
            BEGIN,
            "### Audited v1.1 model ladder",
            "",
            *table,
            "",
            "This table is an admission inventory, not a support or SOTA claim. A row",
            "becomes releasable only when its exact candidate artifact, immutable source",
            "and tokenizer identity, model card, and required evidence ancestry validate.",
            END,
        ]),
        "docs/book/src/model-zoo.md": "\n".join([
            BEGIN,
            "## Audited v1.1 admission ladder",
            "",
            *table,
            "",
            "The generated ladder names release targets only. The model-zoo receipt",
            "binds exact revisions, tokenizer digests, cards, candidate artifacts and",
            "evidence receipts; absent evidence remains `MISSING`.",
            END,
        ]),
        "docs/book/src/benchmarks.md": "\n".join([
            BEGIN,
            "## Release-claim boundary",
            "",
            *table,
            "",
            "No row authorizes a measured quality, speed, memory, energy, compression or",
            "SOTA statement by itself. Such statements must be projected from admitted",
            "candidate receipts with matched artifacts and physical denominators.",
            END,
        ]),
    }


def replace_block(text: str, replacement: str, label: str) -> str:
    starts = text.count(BEGIN)
    ends = text.count(END)
    if starts != 1 or ends != 1:
        raise ClaimGenerationError(
            f"{label} must contain exactly one generated release-claim block"
        )
    prefix, remainder = text.split(BEGIN, 1)
    _, suffix = remainder.split(END, 1)
    return prefix + replacement + suffix


def generated_documents(repo: Path) -> dict[str, str]:
    rendered = blocks()
    result = {}
    for relative in DOCUMENTS:
        path = repo / relative
        try:
            current = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ClaimGenerationError(f"cannot read {relative}") from error
        result[relative] = replace_block(current, rendered[relative], relative)
    return result


def compatibility_markdown(repo: Path) -> str:
    source = repo / "release/compatibility-v1.1.json"
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        matrix = COMPATIBILITY["validate_matrix"](document, source)
        return COMPATIBILITY["render_markdown"](
            matrix, Path("release/compatibility-v1.1.json"),
            Path("docs/compatibility.md"),
        )
    except (OSError, ValueError) as error:
        raise ClaimGenerationError("compatibility projection is invalid") from error


def check(repo: Path) -> None:
    for relative, rendered in generated_documents(repo).items():
        if (repo / relative).read_text(encoding="utf-8") != rendered:
            raise ClaimGenerationError(f"{relative} release-claim block is stale")
    compatibility = repo / "docs/compatibility.md"
    if compatibility.read_text(encoding="utf-8") != compatibility_markdown(repo):
        raise ClaimGenerationError("docs/compatibility.md is stale")


def write(repo: Path) -> None:
    for relative, rendered in generated_documents(repo).items():
        (repo / relative).write_text(rendered, encoding="utf-8")
    compatibility = repo / "docs/compatibility.md"
    compatibility.write_text(compatibility_markdown(repo), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    try:
        check(repo) if args.check else write(repo)
    except ClaimGenerationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
