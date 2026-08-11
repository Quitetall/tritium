#!/usr/bin/env python3
"""Verify Tritium OCI layout, image identity, and BuildKit attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any


OCI_ARCHIVE = runpy.run_path(Path(__file__).with_name("oci-archive.py"))
OciArchiveError = OCI_ARCHIVE["OciArchiveError"]
inspect_oci = OCI_ARCHIVE["inspect"]
BUILD_SCHEMA_V1 = "tritium.oci-build.v1"
BUILD_SCHEMA_V2 = "tritium.oci-build.v2"
BUILD_FIELDS_V1 = {
    "schema", "release", "flavor", "candidate_manifest_sha256",
    "source_revision", "source_created", "source_date_epoch", "source_archive",
    "source_archive_bytes", "source_archive_sha256", "platform", "archive",
    "archive_bytes", "archive_sha256", "builder_id",
}
BUILD_FIELDS_V2 = BUILD_FIELDS_V1 | {"package_inventory_sha256"}
PACKAGE_ARTIFACT_KINDS = frozenset({"oci-image", "helm-chart"})
HEX = frozenset("0123456789abcdef")
class OciError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OciError(f"{label} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OciError(f"{label} must be a JSON object")
    return value


def _ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OciError(f"{label} must be an ordinary file")
    return path.resolve(strict=True)


def package_inventory_sha256(candidate_document: dict[str, Any]) -> str:
    """Hash package artifacts while excluding deployment payloads.

    OCI builds need a stable identity that survives adding their own image and
    chart artifacts to the final release candidate. The inventory still binds
    every non-deployment artifact's id, kind, path and measured identity.
    """
    artifacts = candidate_document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise OciError("candidate artifacts must be a non-empty array")
    inventory = []
    for ordinal, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise OciError(f"candidate artifact {ordinal} must be an object")
        if artifact.get("kind") in PACKAGE_ARTIFACT_KINDS:
            continue
        identity = artifact.get("identity")
        if not isinstance(identity, dict):
            raise OciError(f"candidate artifact {ordinal} identity is malformed")
        inventory.append(
            {
                "id": artifact.get("id"),
                "kind": artifact.get("kind"),
                "path": artifact.get("path"),
                "identity": {
                    "bytes": identity.get("bytes"),
                    "sha256": identity.get("sha256"),
                    "blake3": identity.get("blake3"),
                },
            }
        )
    inventory.sort(key=lambda value: (value["id"], value["path"]))
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate(archive: Path, receipt_path: Path, candidate: Path) -> dict[str, Any]:
    archive = _ordinary(archive, "OCI archive")
    receipt = _json(_ordinary(receipt_path, "build receipt").read_bytes(), "build receipt")
    schema = receipt.get("schema")
    if schema == BUILD_SCHEMA_V1:
        if set(receipt) != BUILD_FIELDS_V1:
            raise OciError("build receipt fields do not match schema v1")
    elif schema == BUILD_SCHEMA_V2:
        if set(receipt) != BUILD_FIELDS_V2:
            raise OciError("build receipt fields do not match schema v2")
    else:
        raise OciError("unsupported OCI build receipt schema")
    candidate = _ordinary(candidate, "package candidate manifest")
    candidate_doc = _json(candidate.read_bytes(), "package candidate manifest")
    manifest_matches = receipt["candidate_manifest_sha256"] == _sha256(candidate)
    if schema == BUILD_SCHEMA_V1 and not manifest_matches:
        raise OciError("build receipt does not bind package candidate manifest")
    if schema == BUILD_SCHEMA_V2:
        if (
            not isinstance(receipt["package_inventory_sha256"], str)
            or len(receipt["package_inventory_sha256"]) != 64
            or any(c not in HEX for c in receipt["package_inventory_sha256"])
        ):
            raise OciError("build receipt package inventory digest is malformed")
        inventory = package_inventory_sha256(candidate_doc)
        if receipt["package_inventory_sha256"] != inventory:
            raise OciError("build receipt package inventory differs")
    if (receipt["release"], receipt["source_revision"]) != (
        candidate_doc.get("release"), candidate_doc.get("source_revision")
    ):
        raise OciError("build receipt lineage differs from package candidate")
    if receipt["archive"] != archive.name or receipt["archive_bytes"] != archive.stat().st_size:
        raise OciError("build receipt archive identity differs")
    if receipt["archive_sha256"] != _sha256(archive):
        raise OciError("build receipt archive SHA-256 differs")
    if receipt["flavor"] not in {"cpu", "cuda"} or receipt["platform"] != "linux/amd64":
        raise OciError("unsupported flavor or platform")

    try:
        with archive.open("rb") as stream:
            inspection = inspect_oci(
                stream,
                archive.stat().st_size,
                receipt["release"],
                receipt["source_revision"],
            )
    except OciArchiveError as error:
        raise OciError(str(error)) from error
    if receipt["builder_id"] != inspection["builder_id"]:
        raise OciError("build receipt builder identity differs from OCI provenance")
    return {
        "image_manifest_digest": inspection["image_manifest_digest"],
        "predicates": inspection["predicates"],
        "release": receipt["release"],
        "source_revision": receipt["source_revision"],
        "flavor": receipt["flavor"],
        "builder_id": inspection["builder_id"],
        "invocation_id": inspection["invocation_id"],
        "build_schema": schema,
        "manifest_matches": manifest_matches,
        "package_inventory_sha256": (
            receipt.get("package_inventory_sha256")
            if schema == BUILD_SCHEMA_V2
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--package-candidate", type=Path, required=True)
    parser.add_argument(
        "--print-package-inventory",
        action="store_true",
        help="print package inventory digest and exit without inspecting an OCI archive",
    )
    args = parser.parse_args()
    try:
        if args.print_package_inventory:
            candidate = _ordinary(args.package_candidate, "package candidate manifest")
            print(package_inventory_sha256(_json(candidate.read_bytes(), "package candidate manifest")))
            return 0
        if args.archive is None or args.build_receipt is None:
            parser.error("--archive and --build-receipt are required unless --print-package-inventory is used")
        result = validate(args.archive, args.build_receipt, args.package_candidate)
    except (OSError, OciError) as error:
        print(f"verify-oci-archive: BLOCKED: {error}")
        return 1
    print(f"verify-oci-archive: PASS: {result['image_manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
