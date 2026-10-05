#!/usr/bin/env python3
"""Validate and score a Qwen36 DeltaNet recurrence-sensitivity receipt.

This verifier checks receipt structure, content identity, and the frozen routing
rule. It does not run the model or authenticate a measurement producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


SCHEMA = "tritium.qwen36-gdn-sensitivity-measurement.v1"
REPOSITORY = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
FAMILIES = ("deltanet", "full_attention")
TENSOR_CLASSES = ("qkv", "output", "gate_up", "down")
THRESHOLD_RATIO = 2.0
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 16 * 1024 * 1024


class ReceiptError(ValueError):
    """A sensitivity receipt is malformed or violates the frozen protocol."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _digest(value: Any, label: str, *, prefixed: bool = False) -> str:
    prefix = "sha256:" if prefixed else ""
    if not isinstance(value, str) or (prefixed and not value.startswith(prefix)):
        raise ReceiptError(f"{label} must be a SHA-256 digest")
    raw = value.removeprefix(prefix)
    if len(raw) != 64 or any(char not in HEX for char in raw):
        raise ReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return raw


def _finite_nonnegative(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReceiptError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ReceiptError(f"{label} is outside its allowed range")
    return result


def verify(value: Any) -> dict[str, Any]:
    """Validate the frozen eight-probe receipt and derive its routing decision."""
    required = {
        "schema", "repository", "revision", "source_model_id",
        "calibration_pack_receipt_id", "calibration_token_digest",
        "recipe_id", "matched_bpw", "runtime_adapter_sha256", "machine",
        "probes", "receipt_id",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ReceiptError("receipt fields differ from the frozen schema")
    if value["schema"] != SCHEMA or value["repository"] != REPOSITORY or value["revision"] != REVISION:
        raise ReceiptError("receipt is not bound to the pinned Qwen36 source")
    for field in ("source_model_id", "calibration_token_digest", "recipe_id"):
        _digest(value[field], field)
    _digest(value["calibration_pack_receipt_id"], "calibration_pack_receipt_id", prefixed=True)
    _digest(value["runtime_adapter_sha256"], "runtime_adapter_sha256")
    bpw = _finite_nonnegative(value["matched_bpw"], "matched_bpw", positive=True)
    if not isinstance(value["machine"], dict) or set(value["machine"]) != {
        "device", "device_name", "runtime_version", "tritium_revision"
    }:
        raise ReceiptError("machine identity fields differ from the frozen schema")
    if not all(isinstance(item, str) and item for item in value["machine"].values()):
        raise ReceiptError("machine identity fields must be non-empty strings")
    tritium_revision = value["machine"]["tritium_revision"]
    if len(tritium_revision) != 40 or any(char not in HEX for char in tritium_revision):
        raise ReceiptError("tritium_revision must be a lowercase Git commit SHA")
    probes = value["probes"]
    if not isinstance(probes, list) or len(probes) != 8:
        raise ReceiptError("the frozen probe set requires exactly eight matrices")

    seen_names: set[str] = set()
    classes_by_family: dict[str, set[str]] = {family: set() for family in FAMILIES}
    terminals: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for index, probe in enumerate(probes):
        fields = {
            "family", "tensor_class", "tensor_name", "tensor_index",
            "source_weight_sha256", "ternary_artifact_sha256", "bpw",
            "weight_mse", "sequence_positions", "output_divergence",
        }
        if not isinstance(probe, dict) or set(probe) != fields:
            raise ReceiptError(f"probe {index} fields differ from the frozen schema")
        family, tensor_class = probe["family"], probe["tensor_class"]
        if family not in FAMILIES or tensor_class not in TENSOR_CLASSES:
            raise ReceiptError(f"probe {index} has an unknown family or tensor class")
        if tensor_class in classes_by_family[family]:
            raise ReceiptError(f"probe {family} repeats tensor class {tensor_class}")
        classes_by_family[family].add(tensor_class)
        name = probe["tensor_name"]
        if not isinstance(name, str) or not name or name in seen_names:
            raise ReceiptError("probe tensor names must be non-empty and unique")
        seen_names.add(name)
        if isinstance(probe["tensor_index"], bool) or not isinstance(probe["tensor_index"], int) or probe["tensor_index"] < 0:
            raise ReceiptError(f"probe {index} tensor index is invalid")
        _digest(probe["source_weight_sha256"], f"probe {index} source weight")
        _digest(probe["ternary_artifact_sha256"], f"probe {index} ternary artifact")
        if _finite_nonnegative(probe["bpw"], f"probe {index} bpw", positive=True) != bpw:
            raise ReceiptError("all probes must use the same matched bpw")
        _finite_nonnegative(probe["weight_mse"], f"probe {index} weight MSE")
        positions, curve = probe["sequence_positions"], probe["output_divergence"]
        if not isinstance(positions, list) or not isinstance(curve, list) or len(positions) < 2 or len(positions) != len(curve):
            raise ReceiptError(f"probe {index} must contain a multi-depth divergence curve")
        previous = -1
        for depth in positions:
            if isinstance(depth, bool) or not isinstance(depth, int) or depth <= previous:
                raise ReceiptError(f"probe {index} sequence positions must increase")
            previous = depth
        for point, metric in enumerate(curve):
            _finite_nonnegative(metric, f"probe {index} divergence point {point}")
        terminals[family].append(float(curve[-1]))

    counts = {family: len(terminals[family]) for family in FAMILIES}
    if counts != {"deltanet": 4, "full_attention": 4}:
        raise ReceiptError("probe set must contain four matrices from each block family")
    if any(classes != set(TENSOR_CLASSES) for classes in classes_by_family.values()):
        raise ReceiptError("each family must cover the four frozen tensor classes")
    full = sorted(terminals["full_attention"])
    control_median = (full[1] + full[2]) / 2
    delta_max = max(terminals["deltanet"])
    routed_classes = sorted(
        {
            probe["tensor_class"]
            for probe in probes
            if probe["family"] == "deltanet"
            and (
                control_median == 0 and float(probe["output_divergence"][-1]) > 0
                or float(probe["output_divergence"][-1])
                > THRESHOLD_RATIO * control_median
            )
        }
    )

    declared_id = value["receipt_id"]
    _digest(declared_id, "receipt_id", prefixed=True)
    body = {key: item for key, item in value.items() if key != "receipt_id"}
    expected_id = "sha256:" + hashlib.sha256(canonical(body)).hexdigest()
    if declared_id != expected_id:
        raise ReceiptError("receipt_id does not match canonical receipt content")
    return {
        "schema": "tritium.qwen36-gdn-sensitivity-verification.v1",
        "receipt_id": declared_id,
        "result": "receipt-structure-valid",
        "matched_bpw": bpw,
        "deltanet_terminal_divergence_max": delta_max,
        "full_attention_terminal_divergence_median": control_median,
        "threshold_ratio": THRESHOLD_RATIO,
        "gate": "fail" if routed_classes else "pass",
        "route_to_refined_track": routed_classes,
        "evidence_scope": "receipt-structure-and-rule-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(args.receipt, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > MAX_RECEIPT_BYTES:
                raise ReceiptError("receipt must be a bounded ordinary file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(MAX_RECEIPT_BYTES + 1)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ) or len(payload) != before.st_size:
                raise ReceiptError("receipt changed while being read")
        finally:
            os.close(descriptor)
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=lambda item: (_ for _ in ()).throw(ReceiptError(f"invalid number {item}")),
        )
        result = verify(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ReceiptError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
