#!/usr/bin/env python3
"""Validate installed-wheel whole-Qwen ONNX Runtime qualification evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.onnx-inference-qualification.v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "wheel", "artifact", "model_artifact_id",
    "environment", "model", "runtime", "parity", "faults", "trace",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
ENVIRONMENT_FIELDS = {
    "python", "torch", "onnx", "onnxruntime", "tritium_distribution",
    "repository_absent", "compiler_absent",
}
MODEL_FIELDS = {
    "model_id", "revision", "scope", "profile", "conversion_mode", "package_id",
}
RUNTIME_FIELDS = {
    "provider", "physical_cpu", "bundle_schema", "sequence_mode",
    "standard_opset", "tritium_opsets", "custom_domain_executed",
    "external_data_authenticated", "dense_weight_initializers",
    "persistent_dense_shadows",
}
PARITY_FIELDS = {
    "prompt_cases", "cached_decode_cases", "generation_cases", "mtp_cases",
    "max_abs_error", "tolerance", "tokens_exact", "states_exact",
    "generation_exact", "mtp_exact",
}
FAULT_FIELDS = {
    "graph_corruption_rejected", "weights_corruption_rejected",
    "path_traversal_rejected", "unknown_operator_rejected",
    "trainable_export_rejected", "trainable_import_rejected",
}
FILE_FIELDS = {"file", "bytes", "sha256"}
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_TRACE_BYTES = 256 * 1024 * 1024


class OnnxReceiptError(ValueError):
    """Whole-model ONNX evidence is stale, incomplete, or structurally substituted."""


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
        raise OnnxReceiptError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OnnxReceiptError(f"{label} must be non-empty")
    return value


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OnnxReceiptError(f"{label} must be a positive integer")
    return value


def contained(root: Path, value: Any, label: str) -> Path:
    text = string(value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise OnnxReceiptError(f"{label} is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise OnnxReceiptError(f"{label} traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise OnnxReceiptError(f"{label} escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise OnnxReceiptError(f"{label} must be an ordinary file")
    return path


def inventory(candidate: Path) -> dict[str, tuple[Any, ...]]:
    if candidate.is_symlink() or not candidate.is_file():
        raise OnnxReceiptError("candidate manifest must be an ordinary file")
    try:
        document = json.loads(candidate.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OnnxReceiptError("candidate manifest is unreadable") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise OnnxReceiptError("candidate artifact inventory is malformed")
    result = {}
    for ordinal, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise OnnxReceiptError(f"candidate artifact {ordinal} is malformed")
        artifact_id = string(value.get("id"), "candidate artifact id")
        path = contained(candidate.parent, value.get("path"), "candidate artifact path")
        actual = (artifact_id, value.get("kind"), path.name, path.stat().st_size, sha256(path))
        declared = (
            artifact_id, value.get("kind"), path.name,
            value["identity"].get("bytes"), value["identity"].get("sha256"),
        )
        if artifact_id in result or actual != declared:
            raise OnnxReceiptError("candidate artifact identity is duplicate or drifted")
        result[artifact_id] = actual
    return result


def bind(record: Any, artifacts: dict[str, tuple[Any, ...]], kind: str, label: str) -> None:
    artifact = object_(record, ARTIFACT_FIELDS, label)
    declared = (
        artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"],
        artifact["sha256"],
    )
    if artifact["kind"] != kind or artifacts.get(artifact["id"]) != declared:
        raise OnnxReceiptError(f"{label} does not bind candidate bytes")


def validate(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise OnnxReceiptError("receipt must be a bounded ordinary file")
    try:
        receipt = object_(json.loads(receipt_path.read_bytes()), FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OnnxReceiptError("receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise OnnxReceiptError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise OnnxReceiptError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise OnnxReceiptError("expected source revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise OnnxReceiptError("receipt does not bind candidate manifest")
    artifacts = inventory(candidate)
    bind(receipt["wheel"], artifacts, "python-wheel", "wheel")
    bind(receipt["artifact"], artifacts, "onnx-bundle", "ONNX artifact")
    model_artifact_id = string(receipt["model_artifact_id"], "model artifact id")
    if artifacts.get(model_artifact_id, (None, None))[1] != "model-bundle":
        raise OnnxReceiptError("source model bundle is absent from candidate")

    environment = object_(receipt["environment"], ENVIRONMENT_FIELDS, "environment")
    for field in ("python", "torch", "onnx", "onnxruntime", "tritium_distribution"):
        string(environment[field], f"environment.{field}")
    expected_distribution = release.replace("-rc.", "rc")
    if (
        environment["onnx"] != "1.22.0"
        or environment["onnxruntime"] != "1.27.0"
        or environment["tritium_distribution"] != expected_distribution
        or environment["repository_absent"] is not True
        or environment["compiler_absent"] is not True
    ):
        raise OnnxReceiptError("ONNX environment differs from frozen source/compiler-free contract")
    model = object_(receipt["model"], MODEL_FIELDS, "model")
    if (
        model["model_id"] != MODEL_ID
        or model["revision"] != MODEL_REVISION
        or model["scope"] != "language+mtp"
        or model["profile"] not in {"compact-v1", "near-lossless-v1"}
        or model["conversion_mode"] not in {"ptq", "refined"}
        or re.fullmatch(r"sha256:[0-9a-f]{64}", string(model["package_id"], "package id")) is None
    ):
        raise OnnxReceiptError("model identity, scope, profile, or lineage is invalid")
    runtime = object_(receipt["runtime"], RUNTIME_FIELDS, "runtime")
    if (
        runtime["provider"] != "CPUExecutionProvider"
        or runtime["physical_cpu"] is not True
        or runtime["bundle_schema"] != "tritium-qwen35-onnx-bundle-v2"
        or runtime["sequence_mode"] != "dynamic-cache-v1"
        or runtime["standard_opset"] != 21
        or runtime["tritium_opsets"] != [1, 2]
        or runtime["custom_domain_executed"] is not True
        or runtime["external_data_authenticated"] is not True
        or runtime["dense_weight_initializers"] != 0
        or runtime["persistent_dense_shadows"] != 0
    ):
        raise OnnxReceiptError("real ORT/custom-domain execution contract failed")
    parity = object_(receipt["parity"], PARITY_FIELDS, "parity")
    for field in ("prompt_cases", "cached_decode_cases", "generation_cases", "mtp_cases"):
        positive_integer(parity[field], f"parity.{field}")
    maximum = parity["max_abs_error"]
    tolerance = parity["tolerance"]
    if (
        isinstance(maximum, bool) or not isinstance(maximum, (int, float))
        or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(maximum)) or not math.isfinite(float(tolerance))
        or not 0 <= float(maximum) <= float(tolerance)
        or float(tolerance) <= 0
        or any(parity[field] is not True for field in (
            "tokens_exact", "states_exact", "generation_exact", "mtp_exact"
        ))
    ):
        raise OnnxReceiptError("whole-model ONNX parity failed")
    faults = object_(receipt["faults"], FAULT_FIELDS, "faults")
    if any(faults[field] is not True for field in FAULT_FIELDS):
        raise OnnxReceiptError("ONNX corruption or trainable-graph fault gate failed")
    trace = object_(receipt["trace"], FILE_FIELDS, "trace")
    trace_path = contained(receipt_path.parent, trace["file"], "trace file")
    if (
        trace_path.stat().st_size <= 0
        or trace_path.stat().st_size > MAX_TRACE_BYTES
        or trace["bytes"] != trace_path.stat().st_size
        or trace["sha256"] != sha256(trace_path)
    ):
        raise OnnxReceiptError("execution trace bytes differ")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise OnnxReceiptError("receipt identity mismatch")
    return receipt
