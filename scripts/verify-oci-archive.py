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
BUILD_SCHEMA = "tritium.oci-build.v1"
BUILD_FIELDS = {
    "schema", "release", "flavor", "candidate_manifest_sha256",
    "source_revision", "source_created", "source_date_epoch", "source_archive",
    "source_archive_bytes", "source_archive_sha256", "platform", "archive",
    "archive_bytes", "archive_sha256", "builder_id",
}
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


def validate(archive: Path, receipt_path: Path, candidate: Path) -> dict[str, Any]:
    archive = _ordinary(archive, "OCI archive")
    receipt = _json(_ordinary(receipt_path, "build receipt").read_bytes(), "build receipt")
    if set(receipt) != BUILD_FIELDS or receipt.get("schema") != BUILD_SCHEMA:
        raise OciError("build receipt fields do not match schema")
    candidate = _ordinary(candidate, "package candidate manifest")
    candidate_doc = _json(candidate.read_bytes(), "package candidate manifest")
    if receipt["candidate_manifest_sha256"] != _sha256(candidate):
        raise OciError("build receipt does not bind package candidate manifest")
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--package-candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.archive, args.build_receipt, args.package_candidate)
    except (OSError, OciError) as error:
        print(f"verify-oci-archive: BLOCKED: {error}")
        return 1
    print(f"verify-oci-archive: PASS: {result['image_manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
