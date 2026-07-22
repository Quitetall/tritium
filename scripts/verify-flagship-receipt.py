#!/usr/bin/env python3
"""Validate candidate-bound Qwen3.6 flagship conversion evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.qwen36-conversion-refinement.v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SOURCE_FIELDS = {"model_id", "revision", "scope"}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
TRACK_FIELDS = {
    "track_id", "mode", "profile", "artifact", "parent_artifact_id",
    "work_id", "recipe_sha256", "package_id", "complete", "strict_reload",
}
COVERAGE_FIELDS = {
    "total_tensors", "additive_matrices", "preserved_tensors",
    "deferred_vision_tensors", "unknown_tensors", "duplicate_tensors",
    "missing_tensors", "vision_identity_bound",
}
PARITY_FIELDS = {
    "language_layers", "host_parity", "cuda_parity", "mtp_oracle_parity",
}
DETERMINISM_FIELDS = {"package_repeat_exact", "evaluation_repeat_exact"}
FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "source", "tracks", "coverage", "parity",
    "determinism",
}
EXPECTED_TRACKS = (
    ("compact-ptq", "ptq", "compact-v1"),
    ("near-lossless-ptq", "ptq", "near-lossless-v1"),
    ("near-lossless-refined", "refined", "near-lossless-v1"),
)
MAX_BYTES = 32 * 1024 * 1024


class FlagshipReceiptError(ValueError):
    """Flagship evidence is stale, incomplete, drifted, or mislabeled."""


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
        raise FlagshipReceiptError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FlagshipReceiptError(f"{label} must be non-empty")
    return value


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise FlagshipReceiptError(f"{label} must be a canonical SHA-256 digest")
    return text


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FlagshipReceiptError(f"{label} must be a positive integer")
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise FlagshipReceiptError("receipt must be a bounded ordinary file")
    try:
        return object_(json.loads(path.read_bytes()), FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipReceiptError("receipt must contain UTF-8 JSON") from error


def contained(root: Path, value: Any, label: str) -> Path:
    text = string(value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise FlagshipReceiptError(f"{label} is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise FlagshipReceiptError(f"{label} traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise FlagshipReceiptError(f"{label} escapes candidate root") from error
    if path.is_symlink() or not path.is_file():
        raise FlagshipReceiptError(f"{label} must be an ordinary file")
    return path


def candidate_artifacts(candidate: Path) -> dict[str, tuple[Any, ...]]:
    if candidate.is_symlink() or not candidate.is_file():
        raise FlagshipReceiptError("candidate manifest must be ordinary")
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipReceiptError("candidate manifest must contain UTF-8 JSON") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise FlagshipReceiptError("candidate artifact inventory is malformed")
    root = candidate.parent.resolve(strict=True)
    result = {}
    for ordinal, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise FlagshipReceiptError(f"candidate artifact {ordinal} is malformed")
        artifact_id = string(value.get("id"), f"candidate artifact {ordinal} id")
        if artifact_id in result:
            raise FlagshipReceiptError("candidate artifact identity is duplicated")
        path = contained(root, value.get("path"), f"candidate artifact {ordinal} path")
        actual = (
            artifact_id, value.get("kind"), path.name, path.stat().st_size, sha256(path)
        )
        declared = (
            artifact_id, value.get("kind"), path.name,
            value["identity"].get("bytes"), value["identity"].get("sha256"),
        )
        if actual != declared:
            raise FlagshipReceiptError("candidate artifact bytes contradict identity")
        result[artifact_id] = actual
    return result


def validate(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
) -> dict[str, Any]:
    receipt = load(receipt_path)
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise FlagshipReceiptError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise FlagshipReceiptError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise FlagshipReceiptError("expected source revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise FlagshipReceiptError("receipt does not bind candidate manifest")

    source = object_(receipt["source"], SOURCE_FIELDS, "source")
    if source != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "scope": "language+mtp",
    }:
        raise FlagshipReceiptError("source is not the pinned language-plus-MTP model")

    artifacts = candidate_artifacts(candidate)
    tracks = receipt["tracks"]
    if not isinstance(tracks, list) or len(tracks) != len(EXPECTED_TRACKS):
        raise FlagshipReceiptError("all three separately labeled tracks are required")
    seen_artifacts: set[str] = set()
    seen_work: set[str] = set()
    seen_packages: set[str] = set()
    for ordinal, expected in enumerate(EXPECTED_TRACKS):
        track = object_(tracks[ordinal], TRACK_FIELDS, f"tracks[{ordinal}]")
        if (track["track_id"], track["mode"], track["profile"]) != expected:
            raise FlagshipReceiptError("track order or label differs from frozen policy")
        artifact = object_(track["artifact"], ARTIFACT_FIELDS, "track artifact")
        candidate_identity = artifacts.get(artifact["id"])
        declared_identity = (
            artifact["id"], artifact["kind"], artifact["name"],
            artifact["bytes"], artifact["sha256"],
        )
        if artifact["kind"] != "model-bundle" or candidate_identity != declared_identity:
            raise FlagshipReceiptError("track does not bind a candidate model bundle")
        if artifact["id"] in seen_artifacts:
            raise FlagshipReceiptError("tracks must bind distinct artifacts")
        seen_artifacts.add(artifact["id"])
        work_id = digest(track["work_id"], "track work id")
        package_id = digest(track["package_id"], "track package id")
        digest(track["recipe_sha256"], "track recipe digest")
        if work_id in seen_work or package_id in seen_packages:
            raise FlagshipReceiptError("track work and package identities must be unique")
        seen_work.add(work_id)
        seen_packages.add(package_id)
        expected_parent = tracks[1]["artifact"]["id"] if ordinal == 2 else None
        if track["parent_artifact_id"] != expected_parent:
            raise FlagshipReceiptError("refined lineage must bind NearLossless PTQ parent")
        if track["complete"] is not True or track["strict_reload"] is not True:
            raise FlagshipReceiptError("track is incomplete or failed strict reload")

    coverage = object_(receipt["coverage"], COVERAGE_FIELDS, "coverage")
    expected_counts = {
        "total_tensors": 1199,
        "additive_matrices": 506,
        "preserved_tensors": 360,
        "deferred_vision_tensors": 333,
    }
    for field, expected in expected_counts.items():
        if positive_integer(coverage[field], f"coverage.{field}") != expected:
            raise FlagshipReceiptError("coverage differs from pinned tensor inventory")
    if any(coverage[field] != 0 for field in (
        "unknown_tensors", "duplicate_tensors", "missing_tensors"
    )) or coverage["vision_identity_bound"] is not True:
        raise FlagshipReceiptError("coverage is incomplete, duplicated, or unbound")

    parity = object_(receipt["parity"], PARITY_FIELDS, "parity")
    if parity["language_layers"] != 64 or any(
        parity[field] is not True
        for field in ("host_parity", "cuda_parity", "mtp_oracle_parity")
    ):
        raise FlagshipReceiptError("dense language/MTP parity is incomplete")
    determinism = object_(receipt["determinism"], DETERMINISM_FIELDS, "determinism")
    if any(determinism[field] is not True for field in DETERMINISM_FIELDS):
        raise FlagshipReceiptError("package or evaluation repetition was not exact")

    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise FlagshipReceiptError("receipt identity mismatch")
    return receipt
