#!/usr/bin/env python3
"""Validate complete TrainingOpManifestV1 backend release evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.training-backend-qualification.v1"
MANIFEST_BLAKE3 = "aefb352d04db145e48394b392a106ab0ad831e09e62d8c76ceddedb36a564083"
VECTOR_BLAKE3 = "fcb250733b991aac165871f8c54b0b063337a3ed01bd1da02de220916887fbd6"
MANIFEST_SHA256 = "b6a2d6a77eb6b655c4392682b37ea0efa4c64b9da8cd380014bdc757b56dbad1"
VECTOR_SHA256 = "9ae03fbf2b9bdf39532906eeb1d370864f5c526c155d9a3427986f21b1f72a49"
FAMILIES = ("cpu", "cuda", "rocm", "metal", "wgpu", "wasi", "mcu")
BACKEND_PREFIXES = {
    "cpu": "cpu.reference.v1",
    "cuda": "cuda.portable.v1:",
    "rocm": "rocm.portable.v1:",
    "metal": "metal.portable.v1:",
    "wgpu": "wgpu.portable.v1:",
    "wasi": "wasm.portable.v1",
    "mcu": "mcu.portable.v1:",
}
FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "manifest_sha256", "vectors_sha256", "bundles",
}
BUNDLE_FIELDS = {"family", "artifact"}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256", "blake3"}
WIRE_FIELDS = {
    "schema_id", "schema_version", "backend_id", "backend_build", "physical_device",
    "manifest_digest", "vector_digest", "supported_operations", "dtypes", "limits",
    "device_resident", "cases",
}
LIMIT_FIELDS = {"max_rank", "max_elements", "max_bytes"}
CASE_FIELDS = {"case_id", "receipt"}
CASE_RECEIPT_FIELDS = {
    "operation", "execution", "dtype", "input_digest", "output_digest",
    "peak_resident_bytes", "scratch_bytes", "host_transfers", "device_resident",
}
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
FORBIDDEN_PHYSICAL = ("emulat", "simulat", "llvmpipe", "software", "fallback")


class TrainingBackendReceiptError(ValueError):
    """Portable backend evidence is partial, stale, emulated, or drifted."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise TrainingBackendReceiptError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingBackendReceiptError(f"{label} must be non-empty")
    return value


def hex_(value: Any, label: str) -> str:
    text = string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise TrainingBackendReceiptError(f"{label} must be 64 lowercase hexadecimal")
    return text


def nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingBackendReceiptError(f"{label} must be a nonnegative integer")
    return value


def contained(root: Path, value: Any, label: str) -> Path:
    text = string(value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise TrainingBackendReceiptError(f"{label} is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise TrainingBackendReceiptError(f"{label} traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise TrainingBackendReceiptError(f"{label} escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise TrainingBackendReceiptError(f"{label} must be an ordinary file")
    return path


def source_contract(repo: Path) -> tuple[list[str], list[dict[str, Any]]]:
    manifest_path = repo / "spec/training/v1/manifest.json"
    vectors_path = repo / "spec/training/v1/vectors/v1.json"
    if sha256(manifest_path) != MANIFEST_SHA256 or sha256(vectors_path) != VECTOR_SHA256:
        raise TrainingBackendReceiptError("frozen manifest or vector bytes drifted")
    try:
        manifest = json.loads(manifest_path.read_bytes())
        vectors = json.loads(vectors_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingBackendReceiptError("frozen training corpus is malformed") from error
    operations = manifest.get("operations") if isinstance(manifest, dict) else None
    cases = vectors.get("cases") if isinstance(vectors, dict) else None
    if not isinstance(operations, list) or not isinstance(cases, list):
        raise TrainingBackendReceiptError("frozen training corpus inventory is malformed")
    operation_ids = [item.get("id") for item in operations if isinstance(item, dict)]
    if len(operation_ids) != 35 or len(cases) != 114 or len(set(operation_ids)) != 35:
        raise TrainingBackendReceiptError("frozen operation/case counts drifted")
    if vectors.get("manifest_digest") != MANIFEST_BLAKE3:
        raise TrainingBackendReceiptError("vector manifest identity drifted")
    return operation_ids, cases


def candidate_inventory(candidate: Path) -> dict[str, tuple[tuple[Any, ...], Path, str]]:
    if candidate.is_symlink() or not candidate.is_file():
        raise TrainingBackendReceiptError("candidate manifest must be ordinary")
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingBackendReceiptError("candidate manifest is malformed") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise TrainingBackendReceiptError("candidate artifact inventory is malformed")
    result = {}
    for ordinal, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise TrainingBackendReceiptError(f"candidate artifact {ordinal} is malformed")
        artifact_id = string(value.get("id"), "candidate artifact id")
        path = contained(candidate.parent, value.get("path"), "candidate artifact path")
        identity = value["identity"]
        actual = (
            artifact_id, value.get("kind"), path.name, path.stat().st_size,
            sha256(path), identity.get("blake3"),
        )
        declared = (
            artifact_id, value.get("kind"), path.name, identity.get("bytes"),
            identity.get("sha256"), identity.get("blake3"),
        )
        if artifact_id in result or actual != declared:
            raise TrainingBackendReceiptError("candidate artifact identity is duplicate or drifted")
        result[artifact_id] = (actual, path, value.get("kind"))
    return result


def validate_bundle(
    family: str,
    path: Path,
    revision: str,
    operations: list[str],
    vectors: list[dict[str, Any]],
) -> tuple[str, str]:
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise TrainingBackendReceiptError("training bundle exceeds size limit")
    try:
        wire = object_(json.loads(path.read_bytes()), WIRE_FIELDS, f"{family} bundle")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingBackendReceiptError(f"{family} bundle is not UTF-8 JSON") from error
    backend = string(wire["backend_id"], "backend id")
    prefix = BACKEND_PREFIXES[family]
    if (prefix.endswith(":") and not backend.startswith(prefix)) or (
        not prefix.endswith(":") and backend != prefix
    ):
        raise TrainingBackendReceiptError(f"{family} backend identity differs from policy")
    build = string(wire["backend_build"], "backend build")
    if not build.endswith(f"+source-git:{revision}") or "+dirty-" in build:
        raise TrainingBackendReceiptError("backend build is not the clean candidate revision")
    physical = string(wire["physical_device"], "physical device")
    if family in {"cuda", "rocm", "metal", "wgpu", "mcu"} and any(
        marker in physical.casefold() for marker in FORBIDDEN_PHYSICAL
    ):
        raise TrainingBackendReceiptError(f"{family} evidence is emulated or fallback")
    if (
        wire["schema_id"] != "tritium.training_receipts"
        or wire["schema_version"] != 1
        or wire["manifest_digest"] != MANIFEST_BLAKE3
        or wire["vector_digest"] != VECTOR_BLAKE3
        or wire["supported_operations"] != operations
        or not isinstance(wire["dtypes"], list)
        or "f32" not in wire["dtypes"]
        or len(set(wire["dtypes"])) != len(wire["dtypes"])
        or any(dtype not in {"f32", "u32", "bytes"} for dtype in wire["dtypes"])
        or wire["device_resident"] is not True
    ):
        raise TrainingBackendReceiptError(f"{family} capability declaration is incomplete")
    limits = object_(wire["limits"], LIMIT_FIELDS, "limits")
    if any(nonnegative_integer(limits[field], f"limits.{field}") == 0 for field in LIMIT_FIELDS):
        raise TrainingBackendReceiptError("backend limits must be positive")
    cases = wire["cases"]
    if not isinstance(cases, list) or len(cases) != len(vectors):
        raise TrainingBackendReceiptError(f"{family} case coverage is incomplete")
    for ordinal, expected in enumerate(vectors):
        case = object_(cases[ordinal], CASE_FIELDS, f"{family} cases[{ordinal}]")
        if case["case_id"] != expected.get("case_id"):
            raise TrainingBackendReceiptError(f"{family} case order differs")
        success = expected.get("expected", {}).get("kind") == "success"
        if not success:
            if case["receipt"] is not None:
                raise TrainingBackendReceiptError("error case unexpectedly produced a receipt")
            continue
        receipt = object_(case["receipt"], CASE_RECEIPT_FIELDS, "case receipt")
        if (
            receipt["operation"] != expected.get("operation")
            or receipt["execution"] != expected.get("execution")
            or receipt["dtype"] not in wire["dtypes"]
            or hex_(receipt["input_digest"], "input digest") == ""
            or hex_(receipt["output_digest"], "output digest") == ""
            or nonnegative_integer(receipt["peak_resident_bytes"], "peak resident") == 0
            or nonnegative_integer(receipt["scratch_bytes"], "scratch bytes")
            > expected["expected"]["scratch_bytes_max"]
            or receipt["host_transfers"] != 0
            or receipt["device_resident"] is not True
        ):
            raise TrainingBackendReceiptError(f"{family} case receipt differs")
    return backend, physical


def validate(
    receipt_path: Path, revision: str, release: str, candidate: Path, repo: Path
) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise TrainingBackendReceiptError("receipt must be a bounded ordinary file")
    try:
        receipt = object_(json.loads(receipt_path.read_bytes()), FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingBackendReceiptError("receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise TrainingBackendReceiptError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise TrainingBackendReceiptError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise TrainingBackendReceiptError("expected revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise TrainingBackendReceiptError("receipt does not bind candidate manifest")
    if receipt["manifest_sha256"] != MANIFEST_SHA256 or receipt["vectors_sha256"] != VECTOR_SHA256:
        raise TrainingBackendReceiptError("receipt training corpus identity differs")
    operations, vectors = source_contract(repo)
    inventory = candidate_inventory(candidate)
    bundles = receipt["bundles"]
    if not isinstance(bundles, list) or len(bundles) != len(FAMILIES):
        raise TrainingBackendReceiptError("all seven backend bundles are required")
    identities = set()
    for ordinal, family in enumerate(FAMILIES):
        bundle = object_(bundles[ordinal], BUNDLE_FIELDS, f"bundles[{ordinal}]")
        if bundle["family"] != family:
            raise TrainingBackendReceiptError("backend family order differs from policy")
        artifact = object_(bundle["artifact"], ARTIFACT_FIELDS, "bundle artifact")
        candidate_entry = inventory.get(artifact["id"])
        declared = (
            artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"],
            artifact["sha256"], artifact["blake3"],
        )
        if (
            artifact["kind"] != "training-receipt-bundle"
            or candidate_entry is None
            or candidate_entry[0] != declared
        ):
            raise TrainingBackendReceiptError("backend bundle does not bind candidate bytes")
        identity = validate_bundle(
            family, candidate_entry[1], revision, operations, vectors
        )
        if identity in identities:
            raise TrainingBackendReceiptError("backend/device identity is duplicated")
        identities.add(identity)
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise TrainingBackendReceiptError("receipt identity mismatch")
    return receipt
