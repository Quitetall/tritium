#!/usr/bin/env python3
"""Strict two-physical-device DDP/FSDP qualification receipt validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "tritium.hf-distributed-qualification.v1"
TOP_FIELDS = {
    "schema",
    "receipt_id",
    "source_revision",
    "release",
    "run_id",
    "started_at_utc",
    "duration_ms",
    "source_dirty",
    "command_contract",
    "artifact",
    "model_config_sha256",
    "model_parameters",
    "machine",
    "environment",
    "world_size",
    "devices",
    "modes",
    "result",
}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
MACHINE_FIELDS = {"machine_id", "system", "architecture"}
ENVIRONMENT_FIELDS = {
    "python_version",
    "torch_version",
    "transformers_version",
    "accelerate_version",
    "cuda_runtime",
    "cuda_driver",
    "nccl_version",
}
DEVICE_FIELDS = {
    "rank",
    "uuid",
    "name",
    "compute_capability",
    "total_memory_bytes",
}
MODE_FIELDS = {
    "name",
    "backend",
    "mixed_precision",
    "world_size",
    "steps",
    "global_batch_size",
    "sequence_length",
    "measured_tokens",
    "elapsed_ms",
    "tokens_per_second",
    "single_device_tokens_per_second",
    "scaling_efficiency",
    "peak_memory_bytes",
    "initial_loss",
    "final_loss",
    "checkpoint_exact",
    "rng_exact",
    "host_transfers",
    "global_state_sha256",
    "rank_checkpoint_sha256",
}
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 1024 * 1024


class ReceiptError(ValueError):
    """Distributed evidence is malformed, stale, or not physically distinct."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReceiptError(f"{label} fields do not match the frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReceiptError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ReceiptError(f"{label} must be finite and positive")
    return float(value)


def _hex(value: Any, length: int, label: str) -> str:
    text = _string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise ReceiptError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if not text.startswith("sha256:"):
        raise ReceiptError(f"{label} must be a sha256 digest")
    _hex(text.removeprefix("sha256:"), 64, label)
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_mode(raw: Any) -> dict[str, Any]:
    mode = _object(raw, MODE_FIELDS, "distributed mode")
    name = _string(mode["name"], "distributed mode.name")
    if name not in {"ddp", "fsdp"}:
        raise ReceiptError("distributed mode.name must be ddp or fsdp")
    if mode["backend"] != "nccl" or mode["mixed_precision"] != "fp16":
        raise ReceiptError(f"{name} must use NCCL fp16")
    if mode["world_size"] != 2:
        raise ReceiptError(f"{name} must use world_size=2")
    steps = _positive_int(mode["steps"], f"{name}.steps")
    if steps < 3:
        raise ReceiptError(f"{name} must measure at least three steps")
    batch = _positive_int(mode["global_batch_size"], f"{name}.global_batch_size")
    sequence = _positive_int(mode["sequence_length"], f"{name}.sequence_length")
    measured_tokens = _positive_int(mode["measured_tokens"], f"{name}.measured_tokens")
    if measured_tokens != steps * batch * sequence:
        raise ReceiptError(f"{name} measured-token arithmetic is inconsistent")
    elapsed_ms = _positive_float(mode["elapsed_ms"], f"{name}.elapsed_ms")
    throughput = _positive_float(mode["tokens_per_second"], f"{name}.tokens_per_second")
    expected_throughput = measured_tokens / (elapsed_ms / 1000.0)
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-9):
        raise ReceiptError(f"{name} throughput arithmetic is inconsistent")
    single_device = _positive_float(
        mode["single_device_tokens_per_second"],
        f"{name}.single_device_tokens_per_second",
    )
    scaling_efficiency = _positive_float(
        mode["scaling_efficiency"], f"{name}.scaling_efficiency"
    )
    expected_efficiency = throughput / (single_device * 2.0)
    if not math.isclose(
        scaling_efficiency, expected_efficiency, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ReceiptError(f"{name} scaling-efficiency arithmetic is inconsistent")
    minimum_efficiency = 0.70 if name == "ddp" else 0.55
    if scaling_efficiency < minimum_efficiency:
        raise ReceiptError(f"{name} scaling efficiency is below the frozen gate")
    _positive_int(mode["peak_memory_bytes"], f"{name}.peak_memory_bytes")
    initial_loss = _positive_float(mode["initial_loss"], f"{name}.initial_loss")
    final_loss = _positive_float(mode["final_loss"], f"{name}.final_loss")
    if final_loss > initial_loss:
        raise ReceiptError(f"{name} final loss exceeds initial loss")
    if mode["checkpoint_exact"] is not True or mode["rng_exact"] is not True:
        raise ReceiptError(f"{name} checkpoint and RNG continuation must be exact")
    if mode["host_transfers"] != 0:
        raise ReceiptError(f"{name} steady-state ternary ops performed host transfers")
    _digest(mode["global_state_sha256"], f"{name}.global_state_sha256")
    ranks = mode["rank_checkpoint_sha256"]
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ReceiptError(f"{name} must bind two rank checkpoints")
    for rank, digest in enumerate(ranks):
        _digest(digest, f"{name}.rank_checkpoint_sha256[{rank}]")
    return mode


def validate(
    receipt_path: Path,
    source_revision: str,
    release: str,
    artifact_path: Path,
) -> dict[str, Any]:
    """Validate exact candidate, hardware distinctness, and measurement semantics."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ReceiptError("distributed receipt must be an ordinary file")
    if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ReceiptError("distributed receipt exceeds metadata size limit")
    try:
        document = _object(
            json.loads(receipt_path.read_bytes()), TOP_FIELDS, "distributed receipt"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("distributed receipt must contain UTF-8 JSON") from error
    if document["schema"] != SCHEMA:
        raise ReceiptError(f"distributed receipt schema must equal {SCHEMA!r}")
    revision = _hex(document["source_revision"], 40, "source_revision")
    if revision != source_revision or document["release"] != release:
        raise ReceiptError("distributed receipt source or release is stale")
    _string(document["run_id"], "run_id")
    started = _string(document["started_at_utc"], "started_at_utc")
    if not started.endswith("Z") or "T" not in started:
        raise ReceiptError("started_at_utc must be an RFC3339 UTC timestamp")
    duration_ms = _positive_float(document["duration_ms"], "duration_ms")
    if document["source_dirty"] is not False:
        raise ReceiptError("distributed qualification requires a clean source revision")
    if document["command_contract"] != "torchrun-nproc2-ddp-then-fsdp-v1":
        raise ReceiptError("distributed command contract mismatch")
    _digest(document["model_config_sha256"], "model_config_sha256")
    _positive_int(document["model_parameters"], "model_parameters")
    artifact = _object(document["artifact"], ARTIFACT_FIELDS, "artifact")
    if artifact["kind"] != "python-wheel":
        raise ReceiptError("distributed artifact must be a Python wheel")
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ReceiptError("candidate wheel must be an ordinary file")
    artifact_path = artifact_path.resolve(strict=True)
    if (
        artifact["name"] != artifact_path.name
        or artifact["bytes"] != artifact_path.stat().st_size
        or artifact["sha256"] != _sha256(artifact_path)
    ):
        raise ReceiptError("distributed receipt does not bind candidate wheel bytes")
    machine = _object(document["machine"], MACHINE_FIELDS, "machine")
    _digest(machine["machine_id"], "machine.machine_id")
    _string(machine["system"], "machine.system")
    _string(machine["architecture"], "machine.architecture")
    environment = _object(document["environment"], ENVIRONMENT_FIELDS, "environment")
    for field, value in environment.items():
        _string(value, f"environment.{field}")
    if document["world_size"] != 2:
        raise ReceiptError("distributed qualification requires world_size=2")
    devices = document["devices"]
    if not isinstance(devices, list) or len(devices) != 2:
        raise ReceiptError("distributed qualification requires two device records")
    uuids = set()
    for rank, raw in enumerate(devices):
        device = _object(raw, DEVICE_FIELDS, f"devices[{rank}]")
        if device["rank"] != rank:
            raise ReceiptError("device ranks must be ordered 0,1")
        uuid = _string(device["uuid"], f"devices[{rank}].uuid")
        uuids.add(uuid)
        _string(device["name"], f"devices[{rank}].name")
        _string(device["compute_capability"], f"devices[{rank}].compute_capability")
        _positive_int(
            device["total_memory_bytes"], f"devices[{rank}].total_memory_bytes"
        )
    if len(uuids) != 2:
        raise ReceiptError("distributed ranks must use two distinct physical GPU UUIDs")
    modes_raw = document["modes"]
    if not isinstance(modes_raw, list):
        raise ReceiptError("distributed modes must be an array")
    modes = [_validate_mode(raw) for raw in modes_raw]
    if [mode["name"] for mode in modes] != ["ddp", "fsdp"]:
        raise ReceiptError("distributed modes must be exactly ordered ddp,fsdp")
    if duration_ms < sum(float(mode["elapsed_ms"]) for mode in modes):
        raise ReceiptError("distributed duration is shorter than measured mode time")
    if document["result"] != "pass":
        raise ReceiptError("distributed result must be pass")
    expected_id = (
        "sha256:"
        + hashlib.sha256(
            canonical(
                {key: value for key, value in document.items() if key != "receipt_id"}
            )
        ).hexdigest()
    )
    if document["receipt_id"] != expected_id:
        raise ReceiptError("distributed receipt identity mismatch")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.receipt, args.source_revision, args.release, args.artifact)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
