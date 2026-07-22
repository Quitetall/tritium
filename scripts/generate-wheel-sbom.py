#!/usr/bin/env python3
"""Generate deterministic CycloneDX inventory for one verified Tritium wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


VERIFY_PATH = Path(__file__).with_name("verify-wheel.py")
SPEC = importlib.util.spec_from_file_location("tritium_verify_wheel", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)

ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
MAX_MEMBERS = 100_000


class SbomError(ValueError):
    """Wheel cannot produce a complete, canonical SBOM."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _component_ref(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def generate(wheel_input: Path, artifact_id: str) -> dict[str, Any]:
    if ID_PATTERN.fullmatch(artifact_id) is None:
        raise SbomError("artifact id must use lowercase portable identifier syntax")
    wheel = VERIFY.resolve_wheel(wheel_input)
    expected_version = VERIFY._workspace_version(Path(__file__).resolve().parent.parent)
    identity = VERIFY.inspect_wheel(wheel, expected_version)
    components: list[dict[str, Any]] = []
    dependency_refs: list[str] = []
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = VERIFY._safe_members(archive)
            if len(members) > MAX_MEMBERS:
                raise SbomError(f"wheel exceeds {MAX_MEMBERS} members")
            dist_info = f"tritium_torch-{expected_version}.dist-info"
            metadata = VERIFY._metadata(archive, members, f"{dist_info}/METADATA")
            for name, info in sorted(members.items()):
                if info.is_dir():
                    continue
                payload = archive.read(name)
                reference = _component_ref("wheel-file", name)
                dependency_refs.append(reference)
                components.append(
                    {
                        "type": "file",
                        "bom-ref": reference,
                        "name": name,
                        "hashes": [{"alg": "SHA-256", "content": _digest(payload)}],
                        "properties": [
                            {"name": "tritium:wheel:size", "value": str(len(payload))}
                        ],
                    }
                )
            for requirement in sorted(set(metadata.get_all("Requires-Dist", []))):
                match = REQUIREMENT_NAME.match(requirement)
                if match is None:
                    raise SbomError(f"invalid Requires-Dist value {requirement!r}")
                name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
                reference = _component_ref("pypi-requirement", requirement)
                dependency_refs.append(reference)
                components.append(
                    {
                        "type": "library",
                        "bom-ref": reference,
                        "name": name,
                        "properties": [
                            {"name": "tritium:python:requires-dist", "value": requirement}
                        ],
                    }
                )
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise SbomError(f"cannot inventory wheel: {error}") from error
    components.sort(key=lambda component: component["bom-ref"])
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "bom-ref": artifact_id,
                "name": "tritium-torch",
                "version": expected_version,
                "hashes": [{"alg": "SHA-256", "content": identity["sha256"]}],
                "properties": [
                    {"name": "tritium:artifact:file", "value": wheel.name},
                    {"name": "tritium:artifact:bytes", "value": str(identity["bytes"])},
                    {
                        "name": "tritium:wheel:platform-tag",
                        "value": str(identity["platform_tag"]),
                    },
                ],
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tritium-generate-wheel-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
        "dependencies": [{"ref": artifact_id, "dependsOn": sorted(dependency_refs)}],
    }


def write_sbom(document: dict[str, Any], output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise SbomError("SBOM output already exists")
    output.parent.resolve(strict=True)
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = generate(args.wheel, args.artifact_id)
        write_sbom(document, args.output)
    except (OSError, SbomError, VERIFY.WheelError) as error:
        print(f"generate-wheel-sbom: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"generate-wheel-sbom: OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
