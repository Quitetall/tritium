#!/usr/bin/env python3
"""Verify Tritium OCI layout, image identity, and BuildKit attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


BUILD_SCHEMA = "tritium.oci-build.v1"
BUILD_FIELDS = {
    "schema", "release", "flavor", "candidate_manifest_sha256",
    "source_revision", "source_created", "source_date_epoch", "source_archive",
    "source_archive_bytes", "source_archive_sha256", "platform", "archive",
    "archive_bytes", "archive_sha256",
}
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
ATTESTATION = "application/vnd.in-toto+json"
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


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise OciError(f"{label} must be a SHA-256 descriptor")
    raw = value[7:]
    if len(raw) != 64 or any(character not in HEX for character in raw):
        raise OciError(f"{label} must be a SHA-256 descriptor")
    return raw


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

    files: dict[str, bytes | None] = {}
    sizes: dict[str, int] = {}
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            logical = PurePosixPath(member.name)
            if (logical.is_absolute() or ".." in logical.parts or "\\" in member.name
                    or logical.as_posix() != member.name or member.issym() or member.islnk()):
                raise OciError("OCI archive contains unsafe topology")
            if member.isdir():
                continue
            if not member.isfile() or member.name in files:
                raise OciError("OCI archive contains unsupported or duplicate entry")
            is_blob = member.name.startswith("blobs/sha256/")
            if member.size > 32 * 1024 * 1024 and not is_blob:
                raise OciError("OCI metadata exceeds size limit")
            stream = tar.extractfile(member)
            if stream is None:
                raise OciError("OCI archive member is unreadable")
            if is_blob:
                digest = hashlib.sha256()
                chunks: list[bytes] | None = [] if member.size <= 32 * 1024 * 1024 else None
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                if member.name != f"blobs/sha256/{digest.hexdigest()}":
                    raise OciError("OCI blob path does not match blob digest")
                files[member.name] = b"".join(chunks) if chunks is not None else None
            else:
                files[member.name] = stream.read()
            sizes[member.name] = member.size
    if _json(files.get("oci-layout", b""), "oci-layout").get("imageLayoutVersion") != "1.0.0":
        raise OciError("OCI layout version must equal 1.0.0")
    index = _json(files.get("index.json", b""), "index.json")

    def check_descriptor(descriptor: dict[str, Any], label: str) -> str:
        raw = _digest(descriptor.get("digest"), f"{label}.digest")
        name = f"blobs/sha256/{raw}"
        if name not in files:
            raise OciError(f"{label} blob is absent or digest-mismatched")
        if descriptor.get("size") != sizes[name]:
            raise OciError(f"{label} descriptor size differs")
        return name

    def blob(descriptor: dict[str, Any], label: str) -> bytes:
        name = check_descriptor(descriptor, label)
        payload = files[name]
        if payload is None:
            raise OciError(f"{label} metadata exceeds size limit")
        return payload

    descriptors = index.get("manifests")
    if not isinstance(descriptors, list):
        raise OciError("OCI index manifests must be an array")
    images = [item for item in descriptors if isinstance(item, dict) and item.get("mediaType") == MANIFEST and item.get("platform") == {"architecture": "amd64", "os": "linux"}]
    if len(images) != 1:
        raise OciError("OCI index must contain exactly one linux/amd64 image manifest")
    image_descriptor = images[0]
    image_digest = "sha256:" + _digest(image_descriptor["digest"], "image.digest")
    image = _json(blob(image_descriptor, "image manifest"), "image manifest")
    config_descriptor = image.get("config")
    if not isinstance(config_descriptor, dict):
        raise OciError("image manifest lacks config descriptor")
    config = _json(blob(config_descriptor, "image config"), "image config")
    layers = image.get("layers")
    if not isinstance(layers, list):
        raise OciError("image layers must be an array")
    for ordinal, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise OciError("image layer descriptor must be an object")
        check_descriptor(layer, f"image layer {ordinal}")
    runtime = config.get("config")
    labels = runtime.get("Labels") if isinstance(runtime, dict) else None
    expected_labels = {
        "org.opencontainers.image.revision": receipt["source_revision"],
        "org.opencontainers.image.version": receipt["release"],
        "io.tritium.artifact.schema": "3",
        "io.tritium.startup-receipt.schema": "1",
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected_labels.items()):
        raise OciError("image config labels do not bind release contracts")
    if runtime.get("User") in {None, "", "0", "0:0"} or runtime.get("Entrypoint") != ["/usr/local/bin/tritium-serve"]:
        raise OciError("image runtime identity is not hardened")

    predicates: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        annotations = descriptor.get("annotations", {})
        if annotations.get("vnd.docker.reference.type") != "attestation-manifest":
            continue
        if annotations.get("vnd.docker.reference.digest") != image_digest:
            raise OciError("attestation manifest does not bind image manifest")
        attestation = _json(blob(descriptor, "attestation manifest"), "attestation manifest")
        attestation_config = attestation.get("config")
        if not isinstance(attestation_config, dict):
            raise OciError("attestation manifest lacks config descriptor")
        check_descriptor(attestation_config, "attestation config")
        for layer in attestation.get("layers", []):
            if not isinstance(layer, dict) or layer.get("mediaType") != ATTESTATION:
                continue
            predicate = layer.get("annotations", {}).get("in-toto.io/predicate-type")
            payload = _json(blob(layer, "attestation layer"), "attestation layer")
            if payload.get("predicateType") != predicate:
                raise OciError("attestation predicate annotation differs from payload")
            subjects = payload.get("subject", [])
            if not any(isinstance(subject, dict) and subject.get("digest", {}).get("sha256") == image_digest[7:] for subject in subjects):
                raise OciError("attestation subject does not bind image manifest")
            predicates.add(predicate)
    if not any("spdx" in value.lower() for value in predicates) or not any("slsa" in value.lower() for value in predicates):
        raise OciError("OCI archive lacks image-bound SBOM and provenance attestations")
    return {"image_manifest_digest": image_digest, "predicates": sorted(predicates)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--package-candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.archive, args.build_receipt, args.package_candidate)
    except (OSError, tarfile.TarError, OciError) as error:
        print(f"verify-oci-archive: BLOCKED: {error}")
        return 1
    print(f"verify-oci-archive: PASS: {result['image_manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
