#!/usr/bin/env python3
"""Bind packaged Rust crates to deterministic cargo-cyclonedx inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 200_000
HEX40 = re.compile(r"[0-9a-f]{40}")


class CrateSbomError(ValueError):
    """Crate archive or generated dependency inventory is not admissible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(path: Path, name: str, version: str, revision: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CrateSbomError("crate archive must be an ordinary file")
    if path.name != f"{name}-{version}.crate":
        raise CrateSbomError("crate filename does not match package identity")
    size = path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise CrateSbomError("crate archive size is outside release bounds")
    prefix = f"{name}-{version}"
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if not members or len(members) > MAX_MEMBERS:
                raise CrateSbomError("crate archive member count is outside release bounds")
            seen: set[str] = set()
            for member in members:
                logical = PurePosixPath(member.name)
                if (
                    logical.is_absolute()
                    or ".." in logical.parts
                    or "\\" in member.name
                    or not logical.parts
                    or logical.parts[0] != prefix
                ):
                    raise CrateSbomError(f"unsafe crate member {member.name!r}")
                if member.name in seen:
                    raise CrateSbomError(f"duplicate crate member {member.name!r}")
                seen.add(member.name)
                if not (member.isfile() or member.isdir()):
                    raise CrateSbomError(f"crate member is not a regular file: {member.name!r}")
            def read_member(relative: str) -> bytes:
                target = f"{prefix}/{relative}"
                try:
                    member = archive.getmember(target)
                except KeyError as error:
                    raise CrateSbomError(f"crate is missing {relative}") from error
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise CrateSbomError(f"crate {relative} is not a regular file")
                if member.size > 1024 * 1024:
                    raise CrateSbomError(f"crate {relative} exceeds metadata bounds")
                return extracted.read()

            manifest = tomllib.loads(read_member("Cargo.toml.orig").decode("utf-8"))
            package = manifest.get("package")
            if not isinstance(package, dict) or package.get("name") != name or package.get("version") != version:
                raise CrateSbomError("Cargo.toml.orig identity does not match archive")
            vcs = json.loads(read_member(".cargo_vcs_info.json"))
            git = vcs.get("git") if isinstance(vcs, dict) else None
            if (
                not isinstance(git, dict)
                or git.get("sha1") != revision
                or git.get("dirty", False) is not False
                or vcs.get("dirty", False) is not False
            ):
                raise CrateSbomError("crate VCS identity is dirty or differs from source revision")
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise CrateSbomError(f"cannot inspect crate archive: {error}") from error
    return {"bytes": size, "sha256": _sha256(path)}


def bind_sbom(
    value: Any,
    *,
    artifact_id: str,
    name: str,
    version: str,
    archive: Path,
    identity: dict[str, Any],
    revision: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("bomFormat") != "CycloneDX":
        raise CrateSbomError("cargo-cyclonedx output is not CycloneDX JSON")
    metadata = value.get("metadata")
    component = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(component, dict) or component.get("name") != name or component.get("version") != version:
        raise CrateSbomError("SBOM root does not match crate package")
    old_ref = component.get("bom-ref")
    if not isinstance(old_ref, str) or not old_ref:
        raise CrateSbomError("SBOM root lacks bom-ref")
    components = value.get("components", [])
    if not isinstance(components, list):
        raise CrateSbomError("SBOM components must be an array")
    reference_map = {old_ref: artifact_id}

    def normalize_component(item: Any) -> None:
        if not isinstance(item, dict):
            raise CrateSbomError("SBOM component must be an object")
        reference = item.get("bom-ref")
        if not isinstance(reference, str) or not reference:
            raise CrateSbomError("SBOM component lacks bom-ref")
        if reference != old_ref and (reference.startswith("path+file:") or "file://" in reference):
            local_name = item.get("name")
            local_version = item.get("version")
            if not isinstance(local_name, str) or not isinstance(local_version, str):
                raise CrateSbomError("local SBOM component lacks name/version")
            reference_map[reference] = (
                f"cargo-local:{local_name}@{local_version}:"
                f"{hashlib.sha256(reference.encode()).hexdigest()[:16]}"
            )
        item["bom-ref"] = reference_map.get(reference, reference)
        purl = item.get("purl")
        if isinstance(purl, str) and "file://" in purl:
            local_name = item.get("name")
            local_version = item.get("version")
            if not isinstance(local_name, str) or not isinstance(local_version, str):
                raise CrateSbomError("local SBOM purl lacks name/version")
            item["purl"] = f"pkg:cargo/{quote(local_name, safe='')}@{quote(local_version, safe='')}"
        nested = item.get("components", [])
        if not isinstance(nested, list):
            raise CrateSbomError("nested SBOM components must be an array")
        for child in nested:
            normalize_component(child)

    normalize_component(component)
    for item in components:
        normalize_component(item)
    if len(set(reference_map.values())) != len(reference_map):
        raise CrateSbomError("canonical SBOM component references collide")
    dependencies = value.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise CrateSbomError("SBOM dependencies must be an array")
    root_edges = 0
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise CrateSbomError("SBOM dependency must be an object")
        reference = dependency.get("ref")
        if reference in reference_map:
            dependency["ref"] = reference_map[reference]
        depends_on = dependency.get("dependsOn", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise CrateSbomError("SBOM dependency edges must be string arrays")
        dependency["dependsOn"] = [reference_map.get(item, item) for item in depends_on]
        if reference == old_ref:
            root_edges += 1
    if root_edges != 1:
        raise CrateSbomError("SBOM must contain exactly one root dependency edge")
    component["bom-ref"] = artifact_id
    component["hashes"] = [{"alg": "SHA-256", "content": identity["sha256"]}]
    properties = component.setdefault("properties", [])
    if not isinstance(properties, list):
        raise CrateSbomError("SBOM root properties must be an array")
    properties.extend(
        [
            {"name": "tritium:artifact:file", "value": archive.name},
            {"name": "tritium:artifact:bytes", "value": str(identity["bytes"])},
            {"name": "tritium:source:revision", "value": revision},
            {"name": "tritium:source:dirty", "value": "false"},
        ]
    )
    properties.sort(key=lambda item: (str(item.get("name")), str(item.get("value"))))
    encoded = json.dumps(value, sort_keys=True)
    if "file://" in encoded or "path+file:" in encoded:
        raise CrateSbomError("SBOM retains a local filesystem reference")
    return value


def generate_all(
    root: Path,
    archives: Path,
    output: Path,
    revision: str,
    source_date_epoch: int,
) -> list[Path]:
    if HEX40.fullmatch(revision) is None or source_date_epoch < 0:
        raise CrateSbomError("source revision or SOURCE_DATE_EPOCH is invalid")
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=root,
            text=True,
            timeout=120,
        )
    )
    packages = sorted(
        (package for package in metadata["packages"] if package.get("publish") != []),
        key=lambda package: package["name"],
    )
    written = []
    for package in packages:
        name = package["name"]
        version = package["version"]
        manifest = Path(package["manifest_path"])
        archive = archives / f"{name}-{version}.crate"
        identity = inspect_archive(archive, name, version, revision)
        temporary_name = f".tritium-release-{os.getpid()}-{name}.cdx"
        temporary = manifest.parent / f"{temporary_name}.json"
        if temporary.exists() or temporary.is_symlink():
            raise CrateSbomError("temporary cargo-cyclonedx output already exists")
        environment = {**os.environ, "SOURCE_DATE_EPOCH": str(source_date_epoch)}
        try:
            subprocess.run(
                [
                    "cargo",
                    "cyclonedx",
                    "--manifest-path",
                    str(manifest),
                    "--format",
                    "json",
                    "--spec-version",
                    "1.5",
                    "--override-filename",
                    temporary_name,
                ],
                cwd=root,
                env=environment,
                check=True,
                timeout=600,
            )
            value = json.loads(temporary.read_text(encoding="utf-8"))
        finally:
            temporary.unlink(missing_ok=True)
        artifact_id = f"crate-{name}"
        bound = bind_sbom(
            value,
            artifact_id=artifact_id,
            name=name,
            version=version,
            archive=archive,
            identity=identity,
            revision=revision,
        )
        destination = output / f"{artifact_id}.cdx.json"
        payload = json.dumps(bound, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        written.append(destination)
    if os.name != "nt":
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=Path("target/package"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    try:
        written = generate_all(
            root,
            args.archives.resolve(strict=True),
            args.output,
            args.source_revision,
            args.source_date_epoch,
        )
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        print(f"generate-crate-sboms: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"generate-crate-sboms: OK: {len(written)} crates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
