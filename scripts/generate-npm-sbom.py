#!/usr/bin/env python3
"""Generate deterministic CycloneDX inventory for the Tritium web archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 100_000
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
PACKAGE_NAME = "@tritium-ai/web"


class NpmSbomError(ValueError):
    """Npm archive cannot produce a complete, canonical SBOM."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _component_ref(name: str) -> str:
    return f"npm-file:{hashlib.sha256(name.encode('utf-8')).hexdigest()}"


def _safe_member(name: str) -> str:
    logical = PurePosixPath(name)
    if (
        not name
        or logical.is_absolute()
        or "\\" in name
        or ".." in logical.parts
        or not logical.parts
        or logical.parts[0] != "package"
    ):
        raise NpmSbomError(f"unsafe npm archive member {name!r}")
    return name


def _package_json(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo]) -> dict[str, Any]:
    member = members.get("package/package.json")
    if member is None or not member.isfile():
        raise NpmSbomError("npm archive is missing package/package.json")
    stream = archive.extractfile(member)
    if stream is None or member.size > 1024 * 1024:
        raise NpmSbomError("npm package metadata is not a bounded regular file")
    try:
        value = json.loads(stream.read())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NpmSbomError("npm package metadata is invalid JSON") from error
    if not isinstance(value, dict):
        raise NpmSbomError("npm package metadata must be an object")
    if value.get("name") != PACKAGE_NAME:
        raise NpmSbomError("npm package name does not match Tritium web package")
    version = value.get("version")
    if not isinstance(version, str) or not version:
        raise NpmSbomError("npm package version is missing")
    return value


def inspect_archive(
    archive_path: Path,
    artifact_id: str,
    expected_version: str,
    source_revision: str,
) -> dict[str, Any]:
    if ID_PATTERN.fullmatch(artifact_id) is None:
        raise NpmSbomError("artifact id must use lowercase portable identifier syntax")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise NpmSbomError("source revision must be a full lowercase Git object ID")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise NpmSbomError("npm archive must be an ordinary file")
    if archive_path.stat().st_size <= 0 or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise NpmSbomError("npm archive size is outside release bounds")
    expected_name = f"tritium-ai-web-{expected_version}.tgz"
    if archive_path.name != expected_name:
        raise NpmSbomError("npm archive filename does not match package identity")
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            raw_members = archive.getmembers()
            if not raw_members or len(raw_members) > MAX_MEMBERS:
                raise NpmSbomError("npm archive member count is outside release bounds")
            members: dict[str, tarfile.TarInfo] = {}
            for member in raw_members:
                name = _safe_member(member.name)
                if name in members:
                    raise NpmSbomError(f"duplicate npm archive member {name!r}")
                if not (member.isfile() or member.isdir()):
                    raise NpmSbomError(f"npm archive member is not regular: {name!r}")
                members[name] = member
            metadata = _package_json(archive, members)
            if metadata.get("version") != expected_version:
                raise NpmSbomError("npm package version does not match archive identity")
            components: list[dict[str, Any]] = []
            dependency_refs: list[str] = []
            for name in sorted(members):
                member = members[name]
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise NpmSbomError(f"cannot read npm archive member {name!r}")
                digest = hashlib.sha256()
                size = 0
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                reference = _component_ref(name)
                dependency_refs.append(reference)
                components.append(
                    {
                        "type": "file",
                        "bom-ref": reference,
                        "name": name,
                        "hashes": [{"alg": "SHA-256", "content": digest.hexdigest()}],
                        "properties": [{"name": "tritium:npm:size", "value": str(size)}],
                    }
                )
    except (OSError, tarfile.TarError) as error:
        raise NpmSbomError(f"cannot inspect npm archive: {error}") from error
    component = {
        "type": "library",
        "bom-ref": artifact_id,
        "name": PACKAGE_NAME,
        "version": expected_version,
        "hashes": [{"alg": "SHA-256", "content": _sha256(archive_path)}],
        "properties": [
            {"name": "tritium:artifact:file", "value": archive_path.name},
            {"name": "tritium:artifact:bytes", "value": str(archive_path.stat().st_size)},
            {"name": "tritium:source:revision", "value": source_revision},
            {"name": "tritium:source:dirty", "value": "false"},
        ],
    }
    components.sort(key=lambda item: item["bom-ref"])
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": component,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tritium-generate-npm-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
        "dependencies": [{"ref": artifact_id, "dependsOn": sorted(dependency_refs)}],
    }


def _workspace_version(root: Path) -> str:
    package_path = root / "packages" / "tritium-web" / "package.json"
    try:
        value = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NpmSbomError(f"cannot read workspace package metadata: {error}") from error
    if not isinstance(value, dict) or value.get("name") != PACKAGE_NAME:
        raise NpmSbomError("workspace package metadata has the wrong name")
    version = value.get("version")
    if not isinstance(version, str) or not version:
        raise NpmSbomError("workspace package metadata lacks version")
    return version


def write_sbom(document: dict[str, Any], output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise NpmSbomError("SBOM output already exists")
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
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent.parent
        version = _workspace_version(root)
        document = inspect_archive(args.archive, args.artifact_id, version, args.source_revision)
        write_sbom(document, args.output)
    except (OSError, NpmSbomError) as error:
        print(f"generate-npm-sbom: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"generate-npm-sbom: OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
