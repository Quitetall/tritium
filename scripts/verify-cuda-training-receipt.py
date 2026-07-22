#!/usr/bin/env python3
"""Fail-closed verifier for one physical CUDA/fp16 training receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA = "tritium.cuda-training-qualification.v1"
TOP_FIELDS = {
    "schema", "receipt_id", "source_revision", "release", "run_id",
    "started_at_utc", "duration_ms", "command", "artifact", "machine",
    "environment", "device", "workload", "measurements", "invariants", "result",
}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
MACHINE_FIELDS = {"machine_id", "system", "architecture"}
ENVIRONMENT_FIELDS = {
    "python_version", "torch_version", "transformers_version",
    "accelerate_version", "cuda_runtime", "cuda_driver",
}
DEVICE_FIELDS = {
    "index", "uuid", "name", "compute_capability", "total_memory_bytes",
}
WORKLOAD_FIELDS = {
    "seed", "mixed_precision", "steps", "batch_size", "sequence_length",
    "model_config_sha256",
}
MEASUREMENT_FIELDS = {"elapsed_ms", "steps_per_second"}
INVARIANT_FIELDS = {
    "ternary_operator_host_transfers", "ternary_operator_dtype",
    "checkpoint_exact",
}


class ReceiptError(ValueError):
    """Receipt is malformed, stale, contradictory, or did not pass."""


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReceiptError(f"{label} fields do not match the frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label} must be a non-empty string")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ReceiptError(f"{label} must be a finite positive number")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    path: Path, source_revision: str, release: str, qualified_artifact: Path
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("receipt must be an ordinary file")
    try:
        document = _object(json.loads(path.read_bytes()), TOP_FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("receipt must contain UTF-8 JSON") from error
    if document["schema"] != SCHEMA:
        raise ReceiptError(f"receipt.schema must equal {SCHEMA!r}")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ReceiptError("expected source revision is not a full Git object ID")
    if document["source_revision"] != source_revision:
        raise ReceiptError("receipt source revision does not match the candidate")
    if document["release"] != release:
        raise ReceiptError("receipt release does not match the candidate")
    _string(document["run_id"], "receipt.run_id")
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", document["started_at_utc"]
    ) is None:
        raise ReceiptError("receipt.started_at_utc must be canonical UTC")
    _positive(document["duration_ms"], "receipt.duration_ms")
    command = document["command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(part, str) or not part for part in command
    ):
        raise ReceiptError("receipt.command must be a non-empty string array")

    if qualified_artifact.is_symlink() or not qualified_artifact.is_file():
        raise ReceiptError("qualified artifact must be an ordinary file")
    artifact = _object(document["artifact"], ARTIFACT_FIELDS, "receipt.artifact")
    _string(artifact["kind"], "receipt.artifact.kind")
    if artifact["name"] != qualified_artifact.name:
        raise ReceiptError("receipt artifact name does not match qualified artifact")
    if type(artifact["bytes"]) is not int or artifact["bytes"] < 0:
        raise ReceiptError("receipt artifact byte count is invalid")
    if artifact["bytes"] != qualified_artifact.stat().st_size:
        raise ReceiptError("receipt artifact byte count does not match qualified artifact")
    if artifact["sha256"] != _sha256(qualified_artifact):
        raise ReceiptError("receipt artifact SHA-256 does not match qualified artifact")

    machine = _object(document["machine"], MACHINE_FIELDS, "receipt.machine")
    if re.fullmatch(
        r"sha256:[0-9a-f]{64}", _string(machine["machine_id"], "machine_id")
    ) is None:
        raise ReceiptError("receipt.machine.machine_id must be a SHA-256 identity")
    _string(machine["system"], "receipt.machine.system")
    _string(machine["architecture"], "receipt.machine.architecture")

    environment = _object(
        document["environment"], ENVIRONMENT_FIELDS, "receipt.environment"
    )
    for field in ENVIRONMENT_FIELDS:
        _string(environment[field], f"receipt.environment.{field}")

    device = _object(document["device"], DEVICE_FIELDS, "receipt.device")
    if type(device["index"]) is not int or device["index"] < 0:
        raise ReceiptError("receipt.device.index must be nonnegative")
    for field in ("uuid", "name"):
        _string(device[field], f"receipt.device.{field}")
    if re.fullmatch(r"[1-9][0-9]*\.[0-9]+", device["compute_capability"]) is None:
        raise ReceiptError("receipt.device.compute_capability is invalid")
    if type(device["total_memory_bytes"]) is not int or device["total_memory_bytes"] <= 0:
        raise ReceiptError("receipt.device.total_memory_bytes must be positive")

    workload = _object(document["workload"], WORKLOAD_FIELDS, "receipt.workload")
    for field in ("seed", "steps", "batch_size", "sequence_length"):
        if type(workload[field]) is not int or workload[field] <= 0:
            raise ReceiptError(f"receipt.workload.{field} must be positive")
    if workload["mixed_precision"] != "fp16":
        raise ReceiptError("receipt workload is not fp16")
    if re.fullmatch(r"[0-9a-f]{64}", workload["model_config_sha256"]) is None:
        raise ReceiptError("receipt workload model digest is invalid")

    measurements = _object(
        document["measurements"], MEASUREMENT_FIELDS, "receipt.measurements"
    )
    elapsed_ms = _positive(
        measurements["elapsed_ms"], "receipt.measurements.elapsed_ms"
    )
    throughput = _positive(
        measurements["steps_per_second"], "receipt.measurements.steps_per_second"
    )
    expected_throughput = workload["steps"] * 1000.0 / elapsed_ms
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-9):
        raise ReceiptError("receipt throughput contradicts elapsed time and step count")

    invariants = _object(
        document["invariants"], INVARIANT_FIELDS, "receipt.invariants"
    )
    if invariants != {
        "ternary_operator_host_transfers": 0,
        "ternary_operator_dtype": "torch.float16",
        "checkpoint_exact": True,
    }:
        raise ReceiptError("receipt training invariants did not pass")
    if document["result"] != "pass":
        raise ReceiptError("receipt result is not pass")

    unsigned = dict(document)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise ReceiptError("receipt_id does not match canonical receipt bytes")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = validate(
            args.receipt, args.source_revision, args.release, args.artifact
        )
    except (OSError, ReceiptError) as error:
        print(f"verify-cuda-training-receipt: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "verify-cuda-training-receipt: PASS: "
        f"{document['device']['name']} {document['workload']['mixed_precision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
