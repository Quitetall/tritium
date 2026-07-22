#!/usr/bin/env python3
"""Validate physical portable-training performance evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.training-performance-qualification.v1"
FAMILIES = ("cpu", "cuda", "rocm", "metal", "wgpu", "wasi", "mcu")
FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "backend_manifest_receipt_id", "workload_id",
    "budget_id", "measurements",
}
MEASUREMENT_FIELDS = {
    "family", "tier", "artifact", "physical_device", "warmup_iterations",
    "sample_count", "cases_per_sample", "median_ms", "p95_ms", "cases_per_second",
    "cpu_relative_speed", "peak_resident_bytes", "peak_scratch_bytes", "host_transfers",
    "global_synchronizations", "native_execution", "budget_pass", "energy_joules",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256", "blake3"}
MAX_RECEIPT_BYTES = 32 * 1024 * 1024


class TrainingPerformanceError(ValueError):
    """Backend performance evidence is stale, synthetic, partial, or inconsistent."""


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
        raise TrainingPerformanceError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingPerformanceError(f"{label} must be non-empty")
    return value


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise TrainingPerformanceError(f"{label} must be a canonical SHA-256 digest")
    return text


def integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingPerformanceError(f"{label} must be an integer at least {minimum}")
    return value


def number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise TrainingPerformanceError(f"{label} must be finite and at least {minimum}")
    return float(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def contained(root: Path, value: Any) -> Path:
    text = string(value, "candidate artifact path")
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise TrainingPerformanceError("candidate artifact path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise TrainingPerformanceError("candidate artifact path traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise TrainingPerformanceError("candidate artifact escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise TrainingPerformanceError("candidate artifact must be ordinary")
    return path


def inventory(candidate: Path) -> dict[str, tuple[tuple[Any, ...], Path]]:
    if candidate.is_symlink() or not candidate.is_file():
        raise TrainingPerformanceError("candidate manifest must be ordinary")
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingPerformanceError("candidate manifest is malformed") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise TrainingPerformanceError("candidate artifact inventory is malformed")
    result = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise TrainingPerformanceError("candidate artifact is malformed")
        artifact_id = string(value.get("id"), "candidate artifact id")
        path = contained(candidate.parent, value.get("path"))
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
            raise TrainingPerformanceError("candidate artifact identity is duplicate or drifted")
        result[artifact_id] = (actual, path)
    return result


def validate(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise TrainingPerformanceError("receipt must be a bounded ordinary file")
    try:
        receipt = object_(json.loads(receipt_path.read_bytes()), FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingPerformanceError("receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise TrainingPerformanceError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise TrainingPerformanceError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise TrainingPerformanceError("expected revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise TrainingPerformanceError("receipt does not bind candidate manifest")
    digest(receipt["backend_manifest_receipt_id"], "backend manifest receipt id")
    if receipt["workload_id"] != "training-manifest-v1-full-114":
        raise TrainingPerformanceError("performance workload or budget is not frozen")
    digest(receipt["budget_id"], "budget id")
    artifacts = inventory(candidate)
    measurements = receipt["measurements"]
    if not isinstance(measurements, list) or len(measurements) != len(FAMILIES):
        raise TrainingPerformanceError("all seven backend measurements are required")
    cpu_median = None
    seen_devices = set()
    for ordinal, family in enumerate(FAMILIES):
        measurement = object_(
            measurements[ordinal], MEASUREMENT_FIELDS, f"measurements[{ordinal}]"
        )
        if measurement["family"] != family:
            raise TrainingPerformanceError("performance family order differs from policy")
        expected_tier = "throughput" if family in {"cpu", "cuda", "rocm", "metal", "wgpu"} else "bounded-latency"
        if measurement["tier"] != expected_tier:
            raise TrainingPerformanceError("performance tier differs from policy")
        artifact = object_(measurement["artifact"], ARTIFACT_FIELDS, "artifact")
        declared = (
            artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"],
            artifact["sha256"], artifact["blake3"],
        )
        candidate_artifact = artifacts.get(artifact["id"])
        if (
            artifact["kind"] != "training-receipt-bundle"
            or candidate_artifact is None
            or candidate_artifact[0] != declared
        ):
            raise TrainingPerformanceError("performance does not bind backend bundle")
        device = string(measurement["physical_device"], "physical device")
        try:
            backend_bundle = json.loads(candidate_artifact[1].read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingPerformanceError("backend bundle is not UTF-8 JSON") from error
        if not isinstance(backend_bundle, dict) or backend_bundle.get("physical_device") != device:
            raise TrainingPerformanceError("performance device differs from conformance bundle")
        if device in seen_devices:
            raise TrainingPerformanceError("physical performance device is duplicated")
        seen_devices.add(device)
        integer(measurement["warmup_iterations"], "warmup iterations", 10)
        integer(measurement["sample_count"], "sample count", 30)
        cases = integer(measurement["cases_per_sample"], "cases per sample", 114)
        if cases != 114:
            raise TrainingPerformanceError("performance sample is not the full corpus")
        median = number(measurement["median_ms"], "median milliseconds", 1e-12)
        p95 = number(measurement["p95_ms"], "p95 milliseconds", median)
        throughput = number(measurement["cases_per_second"], "cases per second", 1e-12)
        if not close(throughput, cases * 1000.0 / median):
            raise TrainingPerformanceError("performance throughput arithmetic differs")
        if family == "cpu":
            cpu_median = median
        if cpu_median is None:
            raise TrainingPerformanceError("CPU performance must be first")
        relative = number(measurement["cpu_relative_speed"], "CPU relative speed", 1e-12)
        if not close(relative, cpu_median / median):
            raise TrainingPerformanceError("CPU-relative performance arithmetic differs")
        integer(measurement["peak_resident_bytes"], "peak resident bytes", 1)
        integer(measurement["peak_scratch_bytes"], "peak scratch bytes", 0)
        energy = measurement["energy_joules"]
        if energy is not None:
            number(energy, "energy joules", 1e-12)
        if (
            measurement["host_transfers"] != 0
            or measurement["global_synchronizations"] != 0
            or measurement["native_execution"] is not True
            or measurement["budget_pass"] is not True
            or p95 < median
        ):
            raise TrainingPerformanceError("performance residency or budget gate failed")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise TrainingPerformanceError("receipt identity mismatch")
    return receipt
