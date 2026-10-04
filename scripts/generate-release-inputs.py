#!/usr/bin/env python3
"""Generate frozen release-inputs from the SBOM-bound staged artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


INPUT_SCHEMA = "tritium.release-inputs.v1"
BUILD_TYPE = "https://tritium.ai/build/package/v1"
ARTIFACT_KINDS = {
    ".whl": "python-wheel",
    ".crate": "rust-crate",
    ".tgz": "npm-archive",
}
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
RELEASE_PATTERN = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")


class ReleaseInputsError(ValueError):
    """Staged package files and their SBOM bindings are not admissible."""


def _required(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseInputsError(f"{label} must be a non-empty string")
    return value


def _sbom_binding(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseInputsError(f"SBOM must be an ordinary file: {path.name}")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseInputsError(f"cannot read SBOM {path.name}: {error}") from error
    if not isinstance(document, dict) or document.get("bomFormat") != "CycloneDX":
        raise ReleaseInputsError(f"SBOM {path.name} is not CycloneDX JSON")
    metadata = document.get("metadata")
    component = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(component, dict):
        raise ReleaseInputsError(f"SBOM {path.name} lacks a root component")
    artifact_id = _required(component.get("bom-ref"), f"{path.name} bom-ref")
    if ID_PATTERN.fullmatch(artifact_id) is None:
        raise ReleaseInputsError(f"SBOM {path.name} has an unsafe artifact ID")
    properties = component.get("properties")
    if not isinstance(properties, list):
        raise ReleaseInputsError(f"SBOM {path.name} lacks artifact properties")
    bindings = [
        item.get("value")
        for item in properties
        if isinstance(item, dict) and item.get("name") == "tritium:artifact:file"
    ]
    if len(bindings) != 1:
        raise ReleaseInputsError(
            f"SBOM {path.name} must bind exactly one tritium:artifact:file"
        )
    filename = _required(bindings[0], f"{path.name} artifact filename")
    logical = PurePosixPath(filename)
    if (
        logical.is_absolute()
        or len(logical.parts) != 1
        or "\\" in filename
        or logical.name != filename
    ):
        raise ReleaseInputsError(f"SBOM {path.name} artifact path is not a basename")
    if Path(filename).suffix not in ARTIFACT_KINDS:
        raise ReleaseInputsError(f"SBOM {path.name} names an unsupported artifact")
    return artifact_id, filename


def build_inputs(
    staged: Path,
    *,
    release: str,
    source_revision: str,
    builder_id: str,
    invocation_id: str,
) -> dict[str, Any]:
    if staged.is_symlink() or not staged.is_dir():
        raise ReleaseInputsError("staged path must be an ordinary directory")
    if RELEASE_PATTERN.fullmatch(release) is None:
        raise ReleaseInputsError("release must be a canonical 1.1.0-rc.N version")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ReleaseInputsError("source revision must be a full lowercase Git object ID")
    builder_id = _required(builder_id, "builder ID")
    invocation_id = _required(invocation_id, "invocation ID")

    assets: dict[str, Path] = {}
    sboms: list[Path] = []
    for path in staged.rglob("*"):
        if path.is_symlink():
            raise ReleaseInputsError(
                f"staged release tree must not contain symlinks: {path}"
            )
        if path.is_dir():
            continue
        if path.name.endswith(".cdx.json"):
            sboms.append(path)
            continue
        if path.suffix not in ARTIFACT_KINDS:
            continue
        if not path.is_file():
            raise ReleaseInputsError(f"release artifact must be an ordinary file: {path}")
        relative = path.relative_to(staged).as_posix()
        assets[relative] = path
    if not assets:
        raise ReleaseInputsError("staged directory contains no release artifacts")

    artifacts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for sbom in sorted(sboms, key=lambda item: item.relative_to(staged).as_posix()):
        artifact_id, filename = _sbom_binding(sbom)
        sbom_relative = sbom.relative_to(staged).as_posix()
        colocated = (sbom.parent / filename).relative_to(staged).as_posix()
        matches = [path for path in assets if PurePosixPath(path).name == filename]
        if colocated in assets:
            artifact_path = colocated
        elif len(matches) == 1:
            artifact_path = matches[0]
        elif not matches:
            raise ReleaseInputsError(f"SBOM {sbom.name} names absent artifact {filename!r}")
        else:
            raise ReleaseInputsError(
                f"SBOM {sbom_relative} ambiguously names artifact {filename!r}; "
                "place the SBOM beside its artifact"
            )
        if artifact_id in seen_ids:
            raise ReleaseInputsError(f"duplicate artifact ID {artifact_id!r}")
        if artifact_path in seen_paths:
            raise ReleaseInputsError(f"duplicate SBOM binding for {artifact_path!r}")
        seen_ids.add(artifact_id)
        seen_paths.add(artifact_path)
        artifacts.append(
            {
                "id": artifact_id,
                "kind": ARTIFACT_KINDS[Path(artifact_path).suffix],
                "path": artifact_path,
                "sbom": sbom_relative,
            }
        )
    if seen_paths != set(assets):
        missing = sorted(set(assets) - seen_paths)
        raise ReleaseInputsError(
            "release artifacts lack unique SBOM bindings: " + ", ".join(missing)
        )
    if not artifacts:
        raise ReleaseInputsError("staged directory contains no SBOM-bound artifacts")

    return {
        "schema": INPUT_SCHEMA,
        "release": release,
        "source_revision": source_revision,
        "builder": {
            "id": builder_id,
            "build_type": BUILD_TYPE,
            "invocation_id": invocation_id,
        },
        "artifacts": sorted(artifacts, key=lambda item: item["id"]),
    }


def write_inputs(output: Path, document: dict[str, Any]) -> None:
    if output.is_symlink() or output.exists():
        raise ReleaseInputsError("release-inputs output already exists")
    if not output.parent.is_dir():
        raise ReleaseInputsError("release-inputs output parent must exist")
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--builder-id", required=True)
    parser.add_argument("--invocation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = build_inputs(
            args.staged,
            release=args.release,
            source_revision=args.source_revision,
            builder_id=args.builder_id,
            invocation_id=args.invocation_id,
        )
        write_inputs(args.output, document)
    except (OSError, ReleaseInputsError) as error:
        print(f"generate-release-inputs: ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifacts": len(document["artifacts"]),
                "source_revision": document["source_revision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
