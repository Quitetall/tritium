#!/usr/bin/env python3
"""Assemble deterministic artifact identities and SLSA provenance into one candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


RELEASE_STATUS = runpy.run_path(Path(__file__).with_name("release-status"))
ReleaseError = RELEASE_STATUS["ReleaseError"]
ArtifactBinding = RELEASE_STATUS["ArtifactBinding"]
candidate_validate = RELEASE_STATUS["validate"]
file_identity = RELEASE_STATUS["_identity"]
contained_file = RELEASE_STATUS["_file"]
sha256_file = RELEASE_STATUS["_sha256"]
validate_sbom = RELEASE_STATUS["_validate_sbom"]
json_file = RELEASE_STATUS["_json_file"]
BUNDLE_SBOM = runpy.run_path(Path(__file__).with_name("generate-bundle-sbom.py"))
BUNDLE_KINDS = BUNDLE_SBOM["KINDS"]
BundleSbomError = BUNDLE_SBOM["BundleSbomError"]
generate_bundle_sbom = BUNDLE_SBOM["generate"]
write_bundle_sbom = BUNDLE_SBOM["write_sbom"]

INPUT_SCHEMA = "tritium.release-inputs.v1"
ENTRY_SCHEMA = "tritium.release-candidate.v1"
TOP_FIELDS = {"schema", "release", "source_revision", "builder", "artifacts"}
BUILDER_FIELDS = {"id", "build_type", "invocation_id"}
ARTIFACT_FIELDS = {"id", "kind", "path", "sbom"}
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
RC_PATTERN = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
HEX = frozenset("0123456789abcdef")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be an object")
    return value


def _exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ReleaseError(f"{label} fields do not match the frozen schema")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"{label} must be a non-empty string")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_fsynced(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def _new_contained_file(root: Path, value: str, label: str) -> Path:
    logical = PurePosixPath(value)
    if (
        logical.is_absolute()
        or not logical.parts
        or ".." in logical.parts
        or "\\" in value
    ):
        raise ReleaseError(f"{label} must be a contained POSIX path")
    cursor = root
    for part in logical.parts[:-1]:
        cursor /= part
        if cursor.is_symlink() or not cursor.is_dir():
            raise ReleaseError(f"{label} parent must be an ordinary directory")
    output = cursor / logical.parts[-1]
    if output.exists() or output.is_symlink():
        raise ReleaseError(f"{label} output already exists")
    return output


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_input(document: Any) -> tuple[str, str, dict[str, str], list[dict[str, str]]]:
    value = _object(document, "inputs")
    _exact(value, TOP_FIELDS, "inputs")
    if value.get("schema") != INPUT_SCHEMA:
        raise ReleaseError(f"inputs.schema must equal {INPUT_SCHEMA!r}")
    release = _string(value.get("release"), "inputs.release")
    if RC_PATTERN.fullmatch(release) is None:
        raise ReleaseError("inputs.release must be a canonical 1.1.0-rc.N version")
    revision = _string(value.get("source_revision"), "inputs.source_revision")
    if len(revision) != 40 or any(character not in HEX for character in revision):
        raise ReleaseError("inputs.source_revision must be a full lowercase Git object ID")
    builder = _object(value.get("builder"), "inputs.builder")
    _exact(builder, BUILDER_FIELDS, "inputs.builder")
    normalized_builder = {
        field: _string(builder.get(field), f"inputs.builder.{field}")
        for field in sorted(BUILDER_FIELDS)
    }
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ReleaseError("inputs.artifacts must be a non-empty array")
    artifacts: list[dict[str, str]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    portable_paths: set[str] = set()
    for ordinal, raw in enumerate(raw_artifacts):
        label = f"inputs.artifacts[{ordinal}]"
        artifact = _object(raw, label)
        _exact(artifact, ARTIFACT_FIELDS, label)
        artifact_id = _string(artifact.get("id"), f"{label}.id")
        if ID_PATTERN.fullmatch(artifact_id) is None:
            raise ReleaseError(f"{label}.id is not a safe artifact identifier")
        normalized = {
            "id": artifact_id,
            "kind": _string(artifact.get("kind"), f"{label}.kind"),
            "path": _string(artifact.get("path"), f"{label}.path"),
            "sbom": _string(artifact.get("sbom"), f"{label}.sbom"),
        }
        if artifact_id in ids:
            raise ReleaseError(f"duplicate artifact id {artifact_id!r}")
        portable_path = normalized["path"].casefold()
        if normalized["path"] in paths or portable_path in portable_paths:
            raise ReleaseError(f"duplicate artifact path {normalized['path']!r}")
        ids.add(artifact_id)
        paths.add(normalized["path"])
        portable_paths.add(portable_path)
        artifacts.append(normalized)
    artifacts.sort(key=lambda artifact: artifact["id"])
    return release, revision, normalized_builder, artifacts


def assemble(inputs: Path, output: Path, digest_tool: str) -> dict[str, Any]:
    if output.name != "manifest.json":
        raise ReleaseError("output filename must be manifest.json")
    root = output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise ReleaseError("candidate manifest already exists")
    document = json_file(inputs.resolve(strict=True), "inputs")
    release, revision, builder, artifacts = _validate_input(document)
    provenance_target = root / "provenance"
    if provenance_target.exists() or provenance_target.is_symlink():
        raise ReleaseError("candidate provenance directory already exists")

    manifest_artifacts = []
    provenance_payloads: list[tuple[str, bytes]] = []
    generated_sboms: list[Path] = []
    try:
        for artifact in artifacts:
            artifact_path = contained_file(root, artifact["path"], f"{artifact['id']}.path")
            requested_sbom = root / PurePosixPath(artifact["sbom"])
            generated_here = False
            if (
                artifact["kind"] in BUNDLE_KINDS
                and not requested_sbom.exists()
                and not requested_sbom.is_symlink()
            ):
                sbom_output = _new_contained_file(
                    root, artifact["sbom"], f"{artifact['id']}.sbom"
                )
                try:
                    generated = generate_bundle_sbom(
                        artifact_path,
                        artifact["id"],
                        artifact["kind"],
                        revision,
                        digest_tool,
                    )
                    write_bundle_sbom(generated, sbom_output)
                except (OSError, BundleSbomError, subprocess.SubprocessError) as error:
                    raise ReleaseError(
                        f"cannot generate {artifact['id']} bundle SBOM: {error}"
                    ) from error
                generated_sboms.append(sbom_output)
                generated_here = True
            sbom_path = contained_file(root, artifact["sbom"], f"{artifact['id']}.sbom")
            sbom = json_file(sbom_path, f"{artifact['id']}.sbom")
            identity = file_identity(artifact_path, digest_tool)
            if not generated_here:
                validate_sbom(
                    sbom,
                    artifact["id"],
                    sbom_path.name,
                    binding=ArtifactBinding(
                        filename=artifact_path.name,
                        bytes=identity["bytes"],
                        sha256=identity["sha256"],
                        kind=artifact["kind"],
                        path=artifact_path,
                        source_revision=revision,
                        digest_tool=digest_tool,
                    ),
                )
            provenance_relative = f"provenance/{artifact['id']}.intoto.json"
            provenance = {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": "https://slsa.dev/provenance/v1",
                "subject": [
                    {
                        "digest": {"sha256": identity["sha256"]},
                        "name": artifact["path"],
                    }
                ],
                "predicate": {
                    "buildDefinition": {
                        "buildType": builder["build_type"],
                        "externalParameters": {
                            "artifact_id": artifact["id"],
                            "artifact_kind": artifact["kind"],
                            "release": release,
                            "source_revision": revision,
                        },
                        "internalParameters": {},
                        "resolvedDependencies": [],
                    },
                    "runDetails": {
                        "builder": {"id": builder["id"]},
                        "metadata": {"invocationId": builder["invocation_id"]},
                    },
                },
            }
            provenance_payload = _canonical(provenance)
            provenance_payloads.append(
                (f"{artifact['id']}.intoto.json", provenance_payload)
            )
            manifest_artifacts.append(
                {
                    "id": artifact["id"],
                    "kind": artifact["kind"],
                    "path": artifact["path"],
                    "identity": identity,
                    "sbom": {
                        "path": sbom_path.relative_to(root).as_posix(),
                        "sha256": sha256_file(sbom_path),
                    },
                    "provenance": {
                        "path": provenance_relative,
                        "sha256": hashlib.sha256(provenance_payload).hexdigest(),
                    },
                }
            )
    except Exception:
        for path in generated_sboms:
            path.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": ENTRY_SCHEMA,
        "release": release,
        "source_revision": revision,
        "artifacts": manifest_artifacts,
    }
    published_provenance = False
    published_manifest = False
    try:
        provenance_target.mkdir()
        published_provenance = True
        for name, payload in provenance_payloads:
            _write_fsynced(provenance_target / name, payload)
        _fsync_directory(provenance_target)
        _fsync_directory(root)
        _write_fsynced(output, _canonical(manifest))
        published_manifest = True
        _fsync_directory(root)
        candidate_validate(output, digest_tool)
        return manifest
    except Exception:
        if published_manifest:
            output.unlink(missing_ok=True)
        if published_provenance:
            shutil.rmtree(provenance_target, ignore_errors=True)
        for path in generated_sboms:
            path.unlink(missing_ok=True)
        _fsync_directory(root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium"))
    args = parser.parse_args()
    try:
        manifest = assemble(args.inputs, args.output, args.digest_tool)
    except (OSError, ReleaseError) as error:
        print(f"assemble-release-candidate: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"assemble-release-candidate: OK: {manifest['release']} "
        f"({len(manifest['artifacts'])} artifacts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
