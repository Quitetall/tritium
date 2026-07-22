#!/usr/bin/env python3
"""Validate physical portable-training performance evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
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
    "trace",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256", "blake3"}
FILE_FIELDS = {"path", "bytes", "sha256"}
TRACE_SCHEMA = "tritium.training-performance-samples.v1"
TRACE_FIELDS = {
    "schema", "family", "artifact_id", "physical_device", "workload_id",
    "budget_id", "warmups_ms", "samples",
}
SAMPLE_FIELDS = {
    "elapsed_ms", "cases", "peak_resident_bytes", "peak_scratch_bytes",
    "host_transfers", "global_synchronizations", "native_execution",
    "budget_pass", "energy_joules",
}
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


def support_file(root: Path, value: Any, label: str) -> Path:
    record = object_(value, FILE_FIELDS, label)
    path = contained(root, record["path"])
    if (
        path.stat().st_size > MAX_RECEIPT_BYTES
        or path.stat().st_size != integer(record["bytes"], f"{label}.bytes", 1)
        or sha256(path) != record["sha256"]
    ):
        raise TrainingPerformanceError(f"{label} bytes drifted")
    return path


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def validate_trace(
    path: Path, measurement: dict[str, Any], workload_id: str, budget_id: str
) -> None:
    try:
        trace = object_(json.loads(path.read_bytes()), TRACE_FIELDS, "performance trace")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingPerformanceError("performance trace is not UTF-8 JSON") from error
    if (
        trace["schema"] != TRACE_SCHEMA
        or trace["family"] != measurement["family"]
        or trace["artifact_id"] != measurement["artifact"]["id"]
        or trace["physical_device"] != measurement["physical_device"]
        or trace["workload_id"] != workload_id
        or trace["budget_id"] != budget_id
    ):
        raise TrainingPerformanceError("performance trace identity differs")
    warmups = trace["warmups_ms"]
    if not isinstance(warmups, list) or len(warmups) != measurement["warmup_iterations"]:
        raise TrainingPerformanceError("performance warmup trace is incomplete")
    for value in warmups:
        number(value, "warmup milliseconds", 1e-12)
    samples = trace["samples"]
    if not isinstance(samples, list) or len(samples) != measurement["sample_count"]:
        raise TrainingPerformanceError("performance sample trace is incomplete")
    elapsed = []
    resident = []
    scratch = []
    transfers = 0
    synchronizations = 0
    energies = []
    for ordinal, raw in enumerate(samples):
        sample = object_(raw, SAMPLE_FIELDS, f"samples[{ordinal}]")
        elapsed.append(number(sample["elapsed_ms"], "sample elapsed", 1e-12))
        if integer(sample["cases"], "sample cases", 114) != 114:
            raise TrainingPerformanceError("performance sample is not the full corpus")
        resident.append(integer(sample["peak_resident_bytes"], "sample resident", 1))
        scratch.append(integer(sample["peak_scratch_bytes"], "sample scratch", 0))
        transfers += integer(sample["host_transfers"], "sample host transfers", 0)
        synchronizations += integer(
            sample["global_synchronizations"], "sample synchronizations", 0
        )
        if sample["native_execution"] is not True or sample["budget_pass"] is not True:
            raise TrainingPerformanceError("performance sample is non-native or over budget")
        energy = sample["energy_joules"]
        if energy is not None:
            energies.append(number(energy, "sample energy", 1e-12))
    median = statistics.median(elapsed)
    p95 = percentile95(elapsed)
    expected_energy = sum(energies) if len(energies) == len(samples) else None
    if (
        not close(measurement["median_ms"], median)
        or not close(measurement["p95_ms"], p95)
        or measurement["peak_resident_bytes"] != max(resident)
        or measurement["peak_scratch_bytes"] != max(scratch)
        or measurement["host_transfers"] != transfers
        or measurement["global_synchronizations"] != synchronizations
        or measurement["energy_joules"] != expected_energy
    ):
        raise TrainingPerformanceError("performance aggregate differs from raw trace")


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
    support_root = receipt_path.parent.resolve(strict=True)
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
        trace_path = support_file(support_root, measurement["trace"], "performance trace")
        validate_trace(
            trace_path, measurement, receipt["workload_id"], receipt["budget_id"]
        )
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise TrainingPerformanceError("receipt identity mismatch")
    return receipt
