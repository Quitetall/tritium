#!/usr/bin/env python3
"""Independently bind measured Qwen source identity to the pinned Hub payload.

This verifier is intentionally separate from source admission: it never changes
that receipt's ``official_payload_authenticated`` field. It checks the local
snapshot against the immutable Hub revision, then requires the already measured
semantic identity to match the reviewed registration constants below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


REPOSITORY = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
API_URL = (
    f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
)
SCHEMA = "tritium.qwen36-official-source-identity.v1"
MANIFEST_SCHEMA = "tritium.qwen36-official-source-manifest.v1"
SOURCE_MODEL_ID = "126eb094f936c87bf7aeff60e57dadf5351ff082a48b8d63c7553919029cd3ca"
MANIFEST_CONTENT_ID = "tsc1_9553bf20975ed88ab3a673522930f9b585ae2e205959ea3dd00ee79c9587c0ba"
SOURCE_PROOF_ID = "tsc1_7e0c191fefc020e74bb0ea1da33d11f69a517a231970d6c9174ee66494e52aa1"
SOURCE_TENSOR_PAYLOAD_BYTES = 55_562_855_904
SNAPSHOT_FILE_COUNT = 29
SNAPSHOT_TOTAL_BYTES = 55_586_107_940
# Frozen from the canonical file inventory returned by the pinned Hub revision.
OFFICIAL_MANIFEST_SHA256 = "7911b682b615162590074c15baa429ff23c64b7c1d66bd2e134ef6fa3a2a3a3f"
MAX_API_BYTES = 8 * 1024 * 1024
HEX = frozenset("0123456789abcdef")
SOURCE_ADMISSION_MODULE = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-source-admission-receipt.py")
)
SOURCE_ADMISSION_VALIDATOR = SOURCE_ADMISSION_MODULE["validate"]
SOURCE_ADMISSION_ERROR = SOURCE_ADMISSION_MODULE["SourceAdmissionError"]


class OfficialIdentityError(ValueError):
    """The local snapshot or its measured identity differs from the registration."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _strict_json(data: bytes, label: str) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OfficialIdentityError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OfficialIdentityError(f"{label} must be a JSON object")
    return value


def _fetch_hub_metadata() -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/json", "User-Agent": "tritium-source-identity/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https" or final.hostname != "huggingface.co":
                raise OfficialIdentityError("Hub API redirected outside huggingface.co")
            body = response.read(MAX_API_BYTES + 1)
    except (OSError, urllib.error.URLError) as error:
        raise OfficialIdentityError("cannot fetch pinned Hugging Face revision metadata") from error
    if len(body) > MAX_API_BYTES:
        raise OfficialIdentityError("Hub API metadata exceeds the size limit")
    return _strict_json(body, "Hub API metadata"), hashlib.sha256(body).hexdigest()


def _registered_source_admission(path: Path) -> dict[str, Any]:
    try:
        validated = SOURCE_ADMISSION_VALIDATOR(path, REVISION, "1.1.0-rc.2", path)
    except SOURCE_ADMISSION_ERROR as error:
        raise OfficialIdentityError("source-admission receipt failed its validator") from error
    receipt = validated["receipt"]
    expected = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "identity_status": "measured-awaiting-official-registration",
        "official_payload_authenticated": False,
        "source_model_id": SOURCE_MODEL_ID,
        "manifest_content_id": MANIFEST_CONTENT_ID,
        "proof_id": SOURCE_PROOF_ID,
        "payload_bytes": SOURCE_TENSOR_PAYLOAD_BYTES,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise OfficialIdentityError(
                f"source-admission {field} does not match the frozen identity"
            )
    return validated


def _official_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("id") != REPOSITORY or metadata.get("sha") != REVISION:
        raise OfficialIdentityError("Hub API repository or immutable revision differs")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or len(siblings) != SNAPSHOT_FILE_COUNT:
        raise OfficialIdentityError("Hub API file inventory differs from the pinned snapshot")
    files: list[dict[str, Any]] = []
    names: set[str] = set()
    total_bytes = 0
    for entry in siblings:
        if not isinstance(entry, dict):
            raise OfficialIdentityError("Hub API file entry must be an object")
        name = entry.get("rfilename")
        if not isinstance(name, str):
            raise OfficialIdentityError("Hub API file name is invalid")
        logical = PurePosixPath(name)
        if logical.is_absolute() or len(logical.parts) != 1 or ".." in logical.parts:
            raise OfficialIdentityError("Hub API file path is not a root-level filename")
        if name in names:
            raise OfficialIdentityError("Hub API file inventory contains duplicate names")
        names.add(name)
        size = entry.get("size")
        if type(size) is not int or size < 0:
            raise OfficialIdentityError(f"Hub API size is invalid for {name}")
        lfs = entry.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise OfficialIdentityError(f"Hub API LFS identity is invalid for {name}")
            digest = lfs.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in HEX for char in digest)
                or lfs.get("size") != size
            ):
                raise OfficialIdentityError(f"Hub API SHA-256 is invalid for {name}")
            algorithm = "sha256"
        else:
            digest = entry.get("blobId")
            if (
                not isinstance(digest, str)
                or len(digest) != 40
                or any(char not in HEX for char in digest)
            ):
                raise OfficialIdentityError(f"Hub API Git blob identity is invalid for {name}")
            algorithm = "git-sha1"
        files.append({"name": name, "size": size, "digest": digest, "algorithm": algorithm})
        total_bytes += size
    if total_bytes != SNAPSHOT_TOTAL_BYTES:
        raise OfficialIdentityError("Hub API snapshot byte total differs from registration")
    return sorted(files, key=lambda item: item["name"])


def _manifest_sha256(files: list[dict[str, Any]]) -> str:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
    }
    return hashlib.sha256(canonical(manifest)).hexdigest()


def _hash_local_file(path: Path, algorithm: str) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OfficialIdentityError(f"cannot open ordinary source file {path.name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OfficialIdentityError(f"source entry {path.name} is not a regular file")
        if algorithm == "sha256":
            digest = hashlib.sha256()
        elif algorithm == "git-sha1":
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
        else:
            raise OfficialIdentityError("unsupported official digest algorithm")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or current.st_ino != after.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise OfficialIdentityError(f"source file {path.name} changed during verification")
        return after.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def verify_snapshot(
    model_dir: Path,
    admission_path: Path,
    metadata: dict[str, Any],
    metadata_sha256: str,
) -> dict[str, Any]:
    """Verify local bytes and return a receipt separate from source admission."""
    admission = _registered_source_admission(admission_path)
    expected_files = _official_files(metadata)
    manifest_sha256 = _manifest_sha256(expected_files)
    if manifest_sha256 != OFFICIAL_MANIFEST_SHA256:
        raise OfficialIdentityError("Hub file inventory differs from the frozen registration")
    root = model_dir
    if root.is_symlink() or not root.is_dir():
        raise OfficialIdentityError("model directory must be an ordinary directory")
    root = root.resolve(strict=True)
    actual_root_entries = {entry.name for entry in root.iterdir() if entry.name != ".cache"}
    expected_names = {item["name"] for item in expected_files}
    if actual_root_entries != expected_names:
        raise OfficialIdentityError("local root files differ from the pinned Hub inventory")
    for entry in root.iterdir():
        if entry.name == ".cache":
            if entry.is_symlink() or not entry.is_dir():
                raise OfficialIdentityError("local Hugging Face cache metadata is invalid")
        elif entry.name not in expected_names:
            raise OfficialIdentityError("local source has an unregistered extra entry")
    verified_files: list[dict[str, Any]] = []
    for expected in expected_files:
        local_path = root / expected["name"]
        size, digest = _hash_local_file(local_path, expected["algorithm"])
        if size != expected["size"] or digest != expected["digest"]:
            raise OfficialIdentityError(f"source bytes differ from pinned Hub file {expected['name']}")
        verified_files.append(expected)
    final_entries = {entry.name for entry in root.iterdir() if entry.name != ".cache"}
    if final_entries != expected_names:
        raise OfficialIdentityError("local source inventory changed during verification")
    if len(metadata_sha256) != 64 or any(char not in HEX for char in metadata_sha256):
        raise OfficialIdentityError("Hub API response digest is invalid")
    content = {
        "schema": SCHEMA,
        "result": "pass",
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_model_id": SOURCE_MODEL_ID,
        "manifest_content_id": MANIFEST_CONTENT_ID,
        "source_proof_id": SOURCE_PROOF_ID,
        "source_admission_receipt_id": admission["receipt_id"],
        "official_manifest_sha256": manifest_sha256,
        "hub_api_response_sha256": metadata_sha256,
        "verified_file_count": len(verified_files),
        "verified_total_bytes": sum(item["size"] for item in verified_files),
        "files": verified_files,
    }
    content["receipt_id"] = "sha256:" + hashlib.sha256(canonical(content)).hexdigest()
    return content


def validate_identity_receipt(path: Path) -> dict[str, Any]:
    """Validate a persisted identity receipt without access to the 55 GB source."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_API_BYTES:
        raise OfficialIdentityError("identity receipt must be a bounded ordinary file")
    value = _strict_json(path.read_bytes(), "official source identity receipt")
    required = {
        "schema", "result", "repository", "revision", "source_model_id",
        "manifest_content_id", "source_proof_id", "source_admission_receipt_id",
        "official_manifest_sha256", "hub_api_response_sha256", "verified_file_count",
        "verified_total_bytes", "files", "receipt_id",
    }
    if set(value) != required:
        raise OfficialIdentityError("official source identity receipt fields differ")
    if (
        value["schema"] != SCHEMA
        or value["result"] != "pass"
        or value["repository"] != REPOSITORY
        or value["revision"] != REVISION
        or value["source_model_id"] != SOURCE_MODEL_ID
        or value["manifest_content_id"] != MANIFEST_CONTENT_ID
        or value["source_proof_id"] != SOURCE_PROOF_ID
    ):
        raise OfficialIdentityError("official source identity receipt pin differs")
    for field in ("official_manifest_sha256", "hub_api_response_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in HEX for c in digest):
            raise OfficialIdentityError(f"{field} is not a lowercase SHA-256 digest")
    if value["official_manifest_sha256"] != OFFICIAL_MANIFEST_SHA256:
        raise OfficialIdentityError("official manifest digest differs from registration")
    admission_id = value["source_admission_receipt_id"]
    if (
        not isinstance(admission_id, str)
        or not admission_id.startswith("sha256:")
        or len(admission_id) != 71
        or any(char not in HEX for char in admission_id[7:])
    ):
        raise OfficialIdentityError("source-admission receipt ID is invalid")
    files = value["files"]
    if not isinstance(files, list) or len(files) != SNAPSHOT_FILE_COUNT:
        raise OfficialIdentityError("verified file list differs from registration")
    if any(
        not isinstance(item, dict)
        or set(item) != {"name", "size", "digest", "algorithm"}
        or not isinstance(item["name"], str)
        or type(item["size"]) is not int
        or item["size"] < 0
        or not isinstance(item["digest"], str)
        or not isinstance(item["algorithm"], str)
        or item["algorithm"] not in ("sha256", "git-sha1")
        or len(item["digest"]) != (64 if item["algorithm"] == "sha256" else 40)
        or any(char not in HEX for char in item["digest"])
        for item in files
    ):
        raise OfficialIdentityError("verified file entries are malformed")
    if files != sorted(files, key=lambda item: item["name"]):
        raise OfficialIdentityError("verified file entries are not canonically ordered")
    if _manifest_sha256(files) != OFFICIAL_MANIFEST_SHA256:
        raise OfficialIdentityError("verified file list does not match official Hub inventory")
    if (
        type(value["verified_file_count"]) is not int
        or value["verified_file_count"] != SNAPSHOT_FILE_COUNT
        or type(value["verified_total_bytes"]) is not int
        or value["verified_total_bytes"] != SNAPSHOT_TOTAL_BYTES
        or sum(item["size"] for item in files) != SNAPSHOT_TOTAL_BYTES
    ):
        raise OfficialIdentityError("verified file totals differ from registration")
    receipt_id = value.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    if receipt_id != expected_id:
        raise OfficialIdentityError("official source identity receipt digest differs")
    value["receipt_id"] = receipt_id
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise OfficialIdentityError("output receipt must be a new ordinary file") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise OfficialIdentityError("cannot durably publish official identity receipt") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source-admission-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata, metadata_sha256 = _fetch_hub_metadata()
        receipt = verify_snapshot(
            args.model_dir, args.source_admission_receipt, metadata, metadata_sha256
        )
        _write_new(args.output, receipt)
    except (OfficialIdentityError, OSError) as error:
        parser.error(str(error))
    print(f"PASS {receipt['receipt_id']} {receipt['verified_file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
