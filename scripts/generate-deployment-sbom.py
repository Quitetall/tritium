#!/usr/bin/env python3
"""Generate canonical CycloneDX inventories for deployment archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import runpy
import stat
import subprocess
import sys
import zlib
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE = runpy.run_path(Path(__file__).with_name("generate-bundle-sbom.py"))
PACKAGER = runpy.run_path(Path(__file__).with_name("package-helm-chart.py"))
OCI = runpy.run_path(Path(__file__).with_name("oci-archive.py"))
_open_artifact = BUNDLE["_open_artifact"]
_verify_path_chain = BUNDLE["_verify_path_chain"]
_read_member = BUNDLE["_read_member"]
_tar_number = BUNDLE["_tar_number"]
_tar_text = BUNDLE["_tar_text"]
_tar_zero_number = BUNDLE["_tar_zero_number"]
_read_exact = BUNDLE["_read_exact"]
_sha256_stream = BUNDLE["_sha256_stream"]
BundleSbomError = BUNDLE["BundleSbomError"]
_write_bundle_sbom = BUNDLE["write_sbom"]
_chart_metadata = PACKAGER["_chart_metadata"]
_gzip = PACKAGER["_gzip"]
_stat_signature = PACKAGER["_stat_signature"]
OciArchiveError = OCI["OciArchiveError"]
inspect_oci = OCI["inspect"]

KINDS = {"helm-chart", "oci-image"}
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
RELEASE_PATTERN = PACKAGER["RELEASE_PATTERN"]
MAX_ARCHIVE_BYTES = PACKAGER["MAX_CHART_BYTES"]
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_OCI_ARCHIVE_BYTES = 256 * 1024 * 1024 * 1024
CANONICAL_GZIP_HEADER = PACKAGER["CANONICAL_GZIP_HEADER"]


class DeploymentSbomError(ValueError):
    """Deployment archive cannot support exact canonical inventory."""


def write_sbom(document: dict[str, Any], output: Path) -> None:
    try:
        _write_bundle_sbom(document, output)
    except BundleSbomError as error:
        raise DeploymentSbomError(str(error)) from error


def _canonical_name(name: str) -> str:
    logical = PurePosixPath(name)
    if (
        logical.is_absolute()
        or len(logical.parts) < 2
        or logical.parts[0] != "tritium"
        or any(part in {"", ".", ".."} for part in logical.parts)
        or "\\" in name
        or len(name.encode("utf-8")) > 100
    ):
        raise DeploymentSbomError(f"chart member {name!r} is not canonical")
    return name


def _header(header: bytes) -> tuple[str, int]:
    if len(header) != 512:
        raise DeploymentSbomError("tar header is truncated")
    expected = _tar_number(header[148:156], "checksum")
    observed = sum(header[:148]) + 8 * ord(" ") + sum(header[156:])
    if expected != observed:
        raise DeploymentSbomError("tar header checksum differs")
    if header[257:263] != b"ustar\0" or header[263:265] != b"00":
        raise DeploymentSbomError("chart archive is not POSIX ustar")
    if header[345:500].rstrip(b"\0") or any(header[500:512]):
        raise DeploymentSbomError("chart tar prefix or padding is not canonical")
    if _tar_number(header[100:108], "mode") != 0o644:
        raise DeploymentSbomError("chart member mode is not canonical")
    if (
        _tar_number(header[108:116], "uid") != 0
        or _tar_number(header[116:124], "gid") != 0
        or _tar_number(header[136:148], "mtime") != 0
    ):
        raise DeploymentSbomError("chart tar identity or timestamp is not canonical")
    if header[156:157] not in {b"\0", b"0"} or any(header[157:257]):
        raise DeploymentSbomError("chart archive contains a link or non-file member")
    if _tar_text(header[265:297], "owner name") or _tar_text(
        header[297:329], "group name"
    ):
        raise DeploymentSbomError("chart tar owner names are not canonical")
    _tar_zero_number(header[329:337], "device fields")
    _tar_zero_number(header[337:345], "device fields")
    return _canonical_name(_tar_text(header[:100], "member name")), _tar_number(
        header[124:136], "member size"
    )


def _inventory(payload: bytes, digest_tool: str) -> tuple[list[dict[str, Any]], bytes]:
    stream = io.BytesIO(payload)
    records: list[dict[str, Any]] = []
    chart_metadata: bytes | None = None
    previous = ""
    portable: set[str] = set()
    total = 0
    while True:
        header = _read_exact(stream, 512, "tar header")
        if not any(header):
            if any(_read_exact(stream, 512, "second tar end block")):
                raise DeploymentSbomError("tar archive has one end block")
            trailing = stream.read()
            if any(trailing) or (1024 + len(trailing)) % 512:
                raise DeploymentSbomError("tar archive has noncanonical trailing padding")
            break
        name, size = _header(header)
        if name <= previous:
            raise DeploymentSbomError("chart members are not strictly sorted")
        folded = name.casefold()
        if folded in portable:
            raise DeploymentSbomError(f"chart contains duplicate portable member {name!r}")
        if len(records) == MAX_MEMBERS:
            raise DeploymentSbomError("chart member count exceeds release bounds")
        total += size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise DeploymentSbomError("chart contents exceed release bounds")
        member = _read_exact(stream, size, f"chart member {name!r}")
        try:
            identity, _ = _read_member(io.BytesIO(member), size, name, digest_tool)
        except Exception as error:
            raise DeploymentSbomError(f"cannot identify chart member {name!r}: {error}") from error
        padding = (-size) % 512
        if padding and any(_read_exact(stream, padding, f"padding for {name!r}")):
            raise DeploymentSbomError(f"chart member {name!r} has nonzero padding")
        if name == "tritium/Chart.yaml":
            chart_metadata = member
        records.append(
            {
                "name": name,
                "bytes": size,
                "sha256": identity["sha256"],
                "blake3": identity["blake3"],
                "package_id": identity["package_id"],
            }
        )
        previous = name
        portable.add(folded)
    names = {record["name"] for record in records}
    if (
        chart_metadata is None
        or "tritium/values.yaml" not in names
        or not any(name.startswith("tritium/templates/") for name in names)
    ):
        raise DeploymentSbomError("chart lacks metadata, values, or templates")
    return records, chart_metadata


def _decompress(payload: bytes) -> bytes:
    if payload[:10] != CANONICAL_GZIP_HEADER:
        raise DeploymentSbomError("chart archive lacks canonical gzip header")
    inflater = zlib.decompressobj(wbits=31)
    try:
        uncompressed = inflater.decompress(payload, MAX_UNCOMPRESSED_BYTES + 1)
        if inflater.unconsumed_tail or len(uncompressed) > MAX_UNCOMPRESSED_BYTES:
            raise DeploymentSbomError("chart gzip stream exceeds release bounds")
        uncompressed += inflater.flush()
    except zlib.error as error:
        raise DeploymentSbomError(f"chart gzip stream is invalid: {error}") from error
    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or len(uncompressed) > MAX_UNCOMPRESSED_BYTES
    ):
        raise DeploymentSbomError("chart gzip stream has trailing or excessive data")
    if _gzip(uncompressed) != payload:
        raise DeploymentSbomError("chart archive does not use canonical gzip encoding")
    return uncompressed


def _components(records: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "file",
            "bom-ref": (
                f"{domain}-file:"
                f"{hashlib.sha256(record['name'].encode('utf-8')).hexdigest()}"
            ),
            "name": record["name"],
            "hashes": [{"alg": "SHA-256", "content": record["sha256"]}],
            "properties": [
                {
                    "name": f"tritium:{domain}:member-bytes",
                    "value": str(record["bytes"]),
                },
                {
                    "name": f"tritium:{domain}:member-blake3",
                    "value": record["blake3"],
                },
                {
                    "name": f"tritium:{domain}:member-package-id",
                    "value": record["package_id"],
                },
            ],
        }
        for record in records
    ]


def _generate_oci(
    artifact: Path,
    artifact_id: str,
    kind: str,
    release: str,
    source_revision: str,
    digest_tool: str,
) -> dict[str, Any]:
    try:
        artifact, descriptor, directories = _open_artifact(artifact)
    except Exception as error:
        raise DeploymentSbomError(str(error)) from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise DeploymentSbomError("OCI artifact must be ordinary file")
            if opened.st_size <= 0 or opened.st_size > MAX_OCI_ARCHIVE_BYTES:
                raise DeploymentSbomError("OCI archive bytes exceed release bounds")
            archive_sha256 = _sha256_stream(stream)
            stream.seek(0)

            def identify(member: Any, size: int, name: str) -> dict[str, Any]:
                identity, _ = _read_member(member, size, name, digest_tool)
                return identity

            inspection = inspect_oci(
                stream,
                opened.st_size,
                release,
                source_revision,
                identify,
            )
            final = os.fstat(stream.fileno())
            if _stat_signature(final) != _stat_signature(opened):
                raise DeploymentSbomError(
                    "OCI artifact metadata changed while generating SBOM"
                )
            _verify_path_chain(artifact, stream.fileno(), directories)
    except DeploymentSbomError:
        raise
    except Exception as error:
        raise DeploymentSbomError(str(error)) from error
    finally:
        for directory in reversed(directories):
            os.close(directory)
    if not artifact.name.endswith(".oci.tar"):
        raise DeploymentSbomError("OCI artifact filename must end in .oci.tar")
    components = _components(inspection["records"], "oci")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "container",
                "bom-ref": artifact_id,
                "name": artifact_id,
                "version": release,
                "hashes": [{"alg": "SHA-256", "content": archive_sha256}],
                "properties": [
                    {"name": "tritium:artifact:file", "value": artifact.name},
                    {"name": "tritium:artifact:bytes", "value": str(opened.st_size)},
                    {"name": "tritium:artifact:kind", "value": kind},
                    {"name": "tritium:deployment:archive-format", "value": "oci-tar"},
                    {
                        "name": "tritium:oci:image-manifest",
                        "value": inspection["image_manifest_digest"],
                    },
                    {"name": "tritium:oci:platform", "value": inspection["platform"]},
                    {
                        "name": "tritium:oci:predicates",
                        "value": json.dumps(inspection["predicates"], separators=(",", ":")),
                    },
                    {
                        "name": "tritium:oci:builder-id",
                        "value": inspection["builder_id"],
                    },
                    {
                        "name": "tritium:oci:invocation-id",
                        "value": inspection["invocation_id"],
                    },
                    {"name": "tritium:release", "value": release},
                    {"name": "tritium:source:revision", "value": source_revision},
                ],
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tritium-generate-deployment-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
        "dependencies": [
            {"ref": artifact_id, "dependsOn": [item["bom-ref"] for item in components]}
        ],
    }


def generate(
    artifact: Path,
    artifact_id: str,
    kind: str,
    release: str,
    source_revision: str,
    digest_tool: str,
) -> dict[str, Any]:
    if ID_PATTERN.fullmatch(artifact_id) is None:
        raise DeploymentSbomError("artifact id must use portable lowercase syntax")
    if kind not in KINDS:
        raise DeploymentSbomError(f"artifact kind must be one of {sorted(KINDS)}")
    if RELEASE_PATTERN.fullmatch(release) is None:
        raise DeploymentSbomError("release must be canonical 1.1.0-rc.N")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        raise DeploymentSbomError("source revision must be full lowercase Git object ID")
    if not digest_tool:
        raise DeploymentSbomError("stream digest tool must be specified")
    if kind == "oci-image":
        return _generate_oci(
            artifact,
            artifact_id,
            kind,
            release,
            source_revision,
            digest_tool,
        )
    try:
        artifact, descriptor, directories = _open_artifact(artifact)
    except Exception as error:
        raise DeploymentSbomError(str(error)) from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise DeploymentSbomError("deployment artifact must be ordinary file")
            if opened.st_size <= 0 or opened.st_size > MAX_ARCHIVE_BYTES:
                raise DeploymentSbomError("deployment archive bytes exceed release bounds")
            payload = stream.read(MAX_ARCHIVE_BYTES + 1)
            if len(payload) != opened.st_size:
                raise DeploymentSbomError("deployment artifact changed while generating SBOM")
            records, chart = _inventory(_decompress(payload), digest_tool)
            stream.seek(0)
            if stream.read(MAX_ARCHIVE_BYTES + 1) != payload:
                raise DeploymentSbomError("deployment artifact changed while generating SBOM")
            final = os.fstat(stream.fileno())
            if _stat_signature(final) != _stat_signature(opened):
                raise DeploymentSbomError(
                    "deployment artifact metadata changed while generating SBOM"
                )
            _verify_path_chain(artifact, stream.fileno(), directories)
    except DeploymentSbomError:
        raise
    except Exception as error:
        raise DeploymentSbomError(str(error)) from error
    finally:
        for directory in reversed(directories):
            os.close(directory)
    expected_name = f"tritium-{release}.tgz"
    if artifact.name != expected_name:
        raise DeploymentSbomError(f"chart artifact filename must equal {expected_name!r}")
    try:
        _chart_metadata(chart, release)
    except Exception as error:
        raise DeploymentSbomError(str(error)) from error
    archive_sha256 = hashlib.sha256(payload).hexdigest()
    components = _components(records, "chart")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": artifact_id,
                "name": artifact_id,
                "version": release,
                "hashes": [{"alg": "SHA-256", "content": archive_sha256}],
                "properties": [
                    {"name": "tritium:artifact:file", "value": artifact.name},
                    {"name": "tritium:artifact:bytes", "value": str(len(payload))},
                    {"name": "tritium:artifact:kind", "value": kind},
                    {"name": "tritium:deployment:archive-format", "value": "helm-tgz"},
                    {"name": "tritium:release", "value": release},
                    {"name": "tritium:source:revision", "value": source_revision},
                ],
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tritium-generate-deployment-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
        "dependencies": [
            {"ref": artifact_id, "dependsOn": [item["bom-ref"] for item in components]}
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = generate(
            args.artifact.absolute(),
            args.artifact_id,
            args.kind,
            args.release,
            args.source_revision,
            args.digest_tool,
        )
        write_sbom(document, args.output)
    except (OSError, DeploymentSbomError, subprocess.SubprocessError) as error:
        print(f"generate-deployment-sbom: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"generate-deployment-sbom: OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
