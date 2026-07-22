#!/usr/bin/env python3
"""Validate Qwen3.6 native-runtime and physical-byte release evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


RUNTIME_SCHEMA = "tritium.qwen36-runtime.v1"
PHYSICAL_SCHEMA = "tritium.qwen36-physical-bytes.v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TRACKS = ("compact-ptq", "near-lossless-ptq", "near-lossless-refined")
WORKLOADS = ((128, 1), (2048, 4))
COMMON_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "artifact", "model_id", "model_revision", "scope",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
DEVICE_FIELDS = {"backend", "physical", "uuid", "name", "driver", "total_bytes"}
RUNTIME_FIELDS = COMMON_FIELDS | {
    "device", "direct_ternary_kernel", "dense_materialization", "host_transfers",
    "measurements", "claimed_regime", "mtp",
}
MEASUREMENT_FIELDS = {
    "track_id", "phase", "context_tokens", "batch_size", "iterations",
    "median_ms", "tokens_per_second",
}
REGIME_FIELDS = {
    "context_tokens", "batch_size", "salt_v1_decode_ms", "ptq_decode_ms",
    "slowdown_pct",
}
MTP_FIELDS = {"acceptance_rate", "baseline_tokens_per_second", "mtp_tokens_per_second", "speedup"}
PHYSICAL_FIELDS = COMMON_FIELDS | {
    "quantized_weights", "dense_artifact_bytes", "dense_resident_bytes", "tracks",
}
PHYSICAL_TRACK_FIELDS = {
    "track_id", "artifact", "matrix_bytes", "matrix_bpw", "metadata_bpw",
    "whole_artifact_bytes", "whole_artifact_bpw", "resident_bytes", "resident_bpw",
    "peak_host_bytes", "peak_device_bytes", "peak_transient_bytes",
    "artifact_reduction", "resident_reduction",
}
MAX_RECEIPT_BYTES = 32 * 1024 * 1024


class FlagshipRuntimeError(ValueError):
    """Flagship runtime/physical evidence is stale, synthetic, or inconsistent."""


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
        raise FlagshipRuntimeError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FlagshipRuntimeError(f"{label} must be non-empty")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FlagshipRuntimeError(f"{label} must be an integer at least {minimum}")
    return value


def number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise FlagshipRuntimeError(f"{label} must be finite and at least {minimum}")
    return float(value)


def signed_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FlagshipRuntimeError(f"{label} must be finite")
    return float(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise FlagshipRuntimeError(f"{label} must be a bounded ordinary file")
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipRuntimeError(f"{label} must contain UTF-8 JSON") from error


def contained(root: Path, value: Any) -> Path:
    text = string(value, "candidate artifact path")
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise FlagshipRuntimeError("candidate artifact path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise FlagshipRuntimeError("candidate artifact path traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise FlagshipRuntimeError("candidate artifact escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise FlagshipRuntimeError("candidate artifact must be ordinary")
    return path


def candidate_inventory(candidate: Path) -> dict[str, tuple[Any, ...]]:
    try:
        document = json.loads(candidate.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipRuntimeError("candidate manifest is unreadable") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise FlagshipRuntimeError("candidate artifact inventory is malformed")
    result = {}
    for ordinal, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise FlagshipRuntimeError(f"candidate artifact {ordinal} is malformed")
        artifact_id = string(value.get("id"), "candidate artifact id")
        path = contained(candidate.parent, value.get("path"))
        actual = (artifact_id, value.get("kind"), path.name, path.stat().st_size, sha256(path))
        declared = (
            artifact_id, value.get("kind"), path.name,
            value["identity"].get("bytes"), value["identity"].get("sha256"),
        )
        if artifact_id in result or actual != declared:
            raise FlagshipRuntimeError("candidate artifact identity is duplicate or drifted")
        result[artifact_id] = actual
    return result


def bind(record: Any, inventory: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    artifact = object_(record, ARTIFACT_FIELDS, "artifact")
    declared = (
        artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"],
        artifact["sha256"],
    )
    if artifact["kind"] != "model-bundle" or inventory.get(artifact["id"]) != declared:
        raise FlagshipRuntimeError("receipt does not bind candidate model bundle")
    return artifact


def common(
    receipt: dict[str, Any], schema: str, revision: str, release: str, candidate: Path
) -> dict[str, tuple[Any, ...]]:
    if receipt["schema"] != schema or receipt["result"] != "pass":
        raise FlagshipRuntimeError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise FlagshipRuntimeError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise FlagshipRuntimeError("expected source revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise FlagshipRuntimeError("receipt does not bind candidate manifest")
    if (
        receipt["model_id"] != MODEL_ID
        or receipt["model_revision"] != MODEL_REVISION
        or receipt["scope"] != "language+mtp"
    ):
        raise FlagshipRuntimeError("receipt does not bind pinned language-plus-MTP scope")
    inventory = candidate_inventory(candidate)
    bind(receipt["artifact"], inventory)
    return inventory


def finish(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise FlagshipRuntimeError(f"{label} receipt identity mismatch")
    return receipt


def validate_runtime(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt = load(receipt_path, RUNTIME_FIELDS, "runtime receipt")
    common(receipt, RUNTIME_SCHEMA, revision, release, candidate)
    device = object_(receipt["device"], DEVICE_FIELDS, "device")
    if device["backend"] != "cuda" or device["physical"] is not True:
        raise FlagshipRuntimeError("runtime requires a physical CUDA device")
    for field in ("uuid", "name", "driver"):
        string(device[field], f"device.{field}")
    integer(device["total_bytes"], "device.total_bytes", minimum=1)
    if (
        receipt["direct_ternary_kernel"] is not True
        or receipt["dense_materialization"] is not False
        or receipt["host_transfers"] != 0
    ):
        raise FlagshipRuntimeError("runtime used dense materialization or host transfer")
    measurements = receipt["measurements"]
    expected = {
        (track, phase, context, batch)
        for track in TRACKS
        for phase in ("prefill", "decode")
        for context, batch in WORKLOADS
    }
    observed: dict[tuple[Any, ...], float] = {}
    if not isinstance(measurements, list) or len(measurements) != len(expected):
        raise FlagshipRuntimeError("runtime workload matrix is incomplete")
    for ordinal, value in enumerate(measurements):
        measurement = object_(value, MEASUREMENT_FIELDS, f"measurements[{ordinal}]")
        key = (
            measurement["track_id"], measurement["phase"],
            integer(measurement["context_tokens"], "context tokens", minimum=1),
            integer(measurement["batch_size"], "batch size", minimum=1),
        )
        integer(measurement["iterations"], "iterations", minimum=20)
        median = number(measurement["median_ms"], "median milliseconds", minimum=1e-12)
        throughput = number(
            measurement["tokens_per_second"], "tokens per second", minimum=1e-12
        )
        tokens = key[2] * key[3] if key[1] == "prefill" else key[3]
        if not close(throughput, tokens * 1000.0 / median):
            raise FlagshipRuntimeError("runtime throughput arithmetic differs")
        if key in observed:
            raise FlagshipRuntimeError("runtime workload is duplicated")
        observed[key] = median
    if set(observed) != expected:
        raise FlagshipRuntimeError("runtime workload matrix differs from policy")
    regime = object_(receipt["claimed_regime"], REGIME_FIELDS, "claimed regime")
    if (regime["context_tokens"], regime["batch_size"]) not in WORKLOADS:
        raise FlagshipRuntimeError("claimed regime is outside measured workloads")
    salt_v1 = number(regime["salt_v1_decode_ms"], "SALT V1 decode", minimum=1e-12)
    ptq = number(regime["ptq_decode_ms"], "PTQ decode", minimum=1e-12)
    slowdown = signed_number(regime["slowdown_pct"], "PTQ slowdown")
    ptq_key = (
        "near-lossless-ptq", "decode", regime["context_tokens"], regime["batch_size"]
    )
    if (
        not close(ptq, observed[ptq_key])
        or not close(slowdown, (ptq / salt_v1 - 1.0) * 100.0)
        or slowdown > 10.0
    ):
        raise FlagshipRuntimeError("PTQ is more than ten percent slower than SALT V1")
    mtp = object_(receipt["mtp"], MTP_FIELDS, "MTP")
    acceptance = number(mtp["acceptance_rate"], "MTP acceptance", minimum=1e-12)
    baseline = number(mtp["baseline_tokens_per_second"], "baseline throughput", minimum=1e-12)
    accelerated = number(mtp["mtp_tokens_per_second"], "MTP throughput", minimum=1e-12)
    speedup = number(mtp["speedup"], "MTP speedup", minimum=1e-12)
    if acceptance > 1.0 or not close(speedup, accelerated / baseline) or speedup <= 1.0:
        raise FlagshipRuntimeError("MTP acceptance or speedup gate failed")
    return finish(receipt, "runtime")


def validate_physical(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt = load(receipt_path, PHYSICAL_FIELDS, "physical-byte receipt")
    inventory = common(receipt, PHYSICAL_SCHEMA, revision, release, candidate)
    weights = integer(receipt["quantized_weights"], "quantized weights", minimum=1)
    dense_artifact = integer(receipt["dense_artifact_bytes"], "dense artifact bytes", minimum=1)
    dense_resident = integer(receipt["dense_resident_bytes"], "dense resident bytes", minimum=1)
    tracks = receipt["tracks"]
    if not isinstance(tracks, list) or len(tracks) != len(TRACKS):
        raise FlagshipRuntimeError("physical ledger requires three tracks")
    seen_artifacts = set()
    for ordinal, track_id in enumerate(TRACKS):
        track = object_(tracks[ordinal], PHYSICAL_TRACK_FIELDS, f"tracks[{ordinal}]")
        if track["track_id"] != track_id:
            raise FlagshipRuntimeError("physical track order differs from policy")
        artifact = bind(track["artifact"], inventory)
        if artifact["id"] in seen_artifacts:
            raise FlagshipRuntimeError("physical tracks must bind distinct artifacts")
        seen_artifacts.add(artifact["id"])
        matrix_bytes = integer(track["matrix_bytes"], "matrix bytes", minimum=1)
        whole = integer(track["whole_artifact_bytes"], "whole artifact bytes", minimum=1)
        resident = integer(track["resident_bytes"], "resident bytes", minimum=1)
        expected_matrix_bpw = matrix_bytes * 8.0 / weights
        expected_whole_bpw = whole * 8.0 / weights
        expected_resident_bpw = resident * 8.0 / weights
        if matrix_bytes > whole or whole != artifact["bytes"] or not all(
            close(actual, expected)
            for actual, expected in (
                (number(track["matrix_bpw"], "matrix bpw"), expected_matrix_bpw),
                (number(track["whole_artifact_bpw"], "whole bpw"), expected_whole_bpw),
                (number(track["resident_bpw"], "resident bpw"), expected_resident_bpw),
                (number(track["artifact_reduction"], "artifact reduction"), dense_artifact / whole),
                (number(track["resident_reduction"], "resident reduction"), dense_resident / resident),
            )
        ):
            raise FlagshipRuntimeError("physical-byte ledger arithmetic differs")
        cap = 2.25 if track_id == "compact-ptq" else 4.0
        if expected_matrix_bpw > cap or number(track["metadata_bpw"], "metadata bpw") > 0.01:
            raise FlagshipRuntimeError("matrix or metadata physical rate exceeds profile cap")
        artifact_reduction = number(track["artifact_reduction"], "artifact reduction")
        resident_reduction = number(track["resident_reduction"], "resident reduction")
        peak_device = integer(track["peak_device_bytes"], "peak_device_bytes", minimum=1)
        for field in ("peak_host_bytes", "peak_transient_bytes"):
            integer(track[field], field, minimum=1)
        if artifact_reduction <= 1.0 or resident_reduction <= 1.0 or peak_device < resident:
            raise FlagshipRuntimeError("physical reduction or peak residency is impossible")
    return finish(receipt, "physical-bytes")
