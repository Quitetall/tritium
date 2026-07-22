#!/usr/bin/env python3
"""Strict physical Chrome/Firefox/Safari WebGPU training receipt validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.browser-training-qualification.v1"
MANIFEST_DIGEST = "aefb352d04db145e48394b392a106ab0ad831e09e62d8c76ceddedb36a564083"
VECTOR_DIGEST = "fcb250733b991aac165871f8c54b0b063337a3ed01bd1da02de220916887fbd6"
TOP_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "artifact",
    "manifest_digest",
    "vector_digest",
    "lanes",
}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
LANE_FIELDS = {
    "engine",
    "browser_version",
    "os",
    "adapter",
    "limits",
    "case_counts",
    "lifecycle",
    "faults",
    "trace",
}
OS_FIELDS = {"name", "version", "architecture"}
ADAPTER_FIELDS = {"vendor", "architecture", "device", "description", "software"}
LIMIT_FIELDS = {
    "max_buffer_size",
    "max_storage_buffer_binding_size",
    "max_compute_workgroups_per_dimension",
    "max_storage_buffers_per_shader_stage",
}
CASE_FIELDS = {"valid", "invalid", "skipped"}
LIFECYCLE_FIELDS = {
    "prepare",
    "forward",
    "backward",
    "optimizer_step",
    "checkpoint_resume",
    "export_reload",
    "native_artifact_parity",
}
FAULT_FIELDS = {
    "device_loss",
    "allocation_failure",
    "malformed_checkpoint",
    "malformed_salt",
    "cancellation",
    "out_of_order",
}
TRACE_FIELDS = {
    "file",
    "bytes",
    "sha256",
    "steady_state_readbacks",
    "wasm_dispatches",
    "explicit_readbacks",
    "peak_buffer_bytes",
}
ENGINES = ("chrome", "firefox", "safari")
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


class BrowserReceiptError(ValueError):
    """Physical-browser evidence is malformed, stale, partial, or synthetic."""


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
        raise BrowserReceiptError(f"{label} fields do not match the frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BrowserReceiptError(f"{label} must be a non-empty string")
    return value


def hex_(value: Any, label: str) -> str:
    text = string(value, label)
    if len(text) != 64 or any(character not in HEX for character in text):
        raise BrowserReceiptError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return text


def positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BrowserReceiptError(f"{label} must be a positive integer")
    return value


def contained_file(root: Path, value: Any, label: str, maximum: int) -> Path:
    logical = PurePosixPath(string(value, label))
    if (
        logical.is_absolute()
        or "\\" in str(value)
        or "\0" in str(value)
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise BrowserReceiptError(f"{label} is unsafe")
    candidate = root.joinpath(*logical.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise BrowserReceiptError(f"{label} must be an ordinary file")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise BrowserReceiptError(f"{label} escapes the evidence directory") from error
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > maximum:
        raise BrowserReceiptError(f"{label} exceeds its byte ceiling")
    return resolved


def validate_lane(value: Any, ordinal: int, root: Path) -> dict[str, Any]:
    label = f"receipt.lanes[{ordinal}]"
    lane = object_(value, LANE_FIELDS, label)
    if lane["engine"] != ENGINES[ordinal]:
        raise BrowserReceiptError(
            "browser lanes must be ordered Chrome, Firefox, Safari"
        )
    if (
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+){1,3}",
            string(lane["browser_version"], f"{label}.browser_version"),
        )
        is None
    ):
        raise BrowserReceiptError(f"{label}.browser_version is invalid")
    os_value = object_(lane["os"], OS_FIELDS, f"{label}.os")
    for field in OS_FIELDS:
        string(os_value[field], f"{label}.os.{field}")
    if lane["engine"] == "safari" and os_value["name"].lower() not in {
        "macos",
        "darwin",
    }:
        raise BrowserReceiptError("Safari evidence must run on physical macOS")
    adapter = object_(lane["adapter"], ADAPTER_FIELDS, f"{label}.adapter")
    for field in ADAPTER_FIELDS - {"software"}:
        string(adapter[field], f"{label}.adapter.{field}")
    description = " ".join(
        str(adapter[field]) for field in ADAPTER_FIELDS - {"software"}
    )
    if adapter["software"] is not False or any(
        marker in description.lower()
        for marker in ("swiftshader", "llvmpipe", "software", "emulator")
    ):
        raise BrowserReceiptError(f"{label} does not identify a physical adapter")
    limits = object_(lane["limits"], LIMIT_FIELDS, f"{label}.limits")
    for field in LIMIT_FIELDS:
        positive(limits[field], f"{label}.limits.{field}")
    cases = object_(lane["case_counts"], CASE_FIELDS, f"{label}.case_counts")
    if cases != {"valid": 70, "invalid": 44, "skipped": 0}:
        raise BrowserReceiptError(f"{label} must execute all 114 canonical vectors")
    lifecycle = object_(lane["lifecycle"], LIFECYCLE_FIELDS, f"{label}.lifecycle")
    if any(lifecycle[field] is not True for field in LIFECYCLE_FIELDS):
        raise BrowserReceiptError(f"{label} lifecycle is incomplete")
    faults = object_(lane["faults"], FAULT_FIELDS, f"{label}.faults")
    if any(faults[field] is not True for field in FAULT_FIELDS):
        raise BrowserReceiptError(f"{label} fault injection is incomplete")
    trace = object_(lane["trace"], TRACE_FIELDS, f"{label}.trace")
    path = contained_file(root, trace["file"], f"{label}.trace.file", MAX_TRACE_BYTES)
    if trace["bytes"] != path.stat().st_size or hex_(
        trace["sha256"], f"{label}.trace.sha256"
    ) != sha256(path):
        raise BrowserReceiptError(f"{label} trace bytes differ")
    if trace["steady_state_readbacks"] != 0 or trace["wasm_dispatches"] != 0:
        raise BrowserReceiptError(
            f"{label} trace contains forbidden fallback or readback"
        )
    positive(trace["explicit_readbacks"], f"{label}.trace.explicit_readbacks")
    positive(trace["peak_buffer_bytes"], f"{label}.trace.peak_buffer_bytes")
    return lane


def validate(
    receipt_path: Path, revision: str, release: str, archive: Path
) -> dict[str, Any]:
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise BrowserReceiptError("browser receipt must be a bounded ordinary file")
    if (
        archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size <= 0
        or archive.stat().st_size > MAX_ARCHIVE_BYTES
    ):
        raise BrowserReceiptError("browser archive must be an ordinary file")
    try:
        receipt = object_(json.loads(receipt_path.read_bytes()), TOP_FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BrowserReceiptError("browser receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise BrowserReceiptError("browser receipt schema or result mismatch")
    if (
        len(revision) != 40
        or any(character not in HEX for character in revision)
        or not release
    ):
        raise BrowserReceiptError("expected browser source or release is invalid")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise BrowserReceiptError("browser receipt source or release is stale")
    string(receipt["run_id"], "receipt.run_id")
    artifact = object_(receipt["artifact"], ARTIFACT_FIELDS, "receipt.artifact")
    if (
        artifact["kind"] != "npm-archive"
        or artifact["name"] != archive.name
        or Path(str(artifact["name"])).name != artifact["name"]
        or not str(artifact["name"]).endswith(".tgz")
        or artifact["bytes"] != archive.stat().st_size
        or hex_(artifact["sha256"], "receipt.artifact.sha256") != sha256(archive)
    ):
        raise BrowserReceiptError("browser receipt does not bind the npm archive")
    if receipt["manifest_digest"] != MANIFEST_DIGEST:
        raise BrowserReceiptError("browser receipt manifest digest mismatch")
    if receipt["vector_digest"] != VECTOR_DIGEST:
        raise BrowserReceiptError("browser receipt vector digest mismatch")
    lanes = receipt["lanes"]
    if not isinstance(lanes, list) or len(lanes) != len(ENGINES):
        raise BrowserReceiptError("browser receipt must contain exactly three lanes")
    root = receipt_path.parent.resolve(strict=True)
    for ordinal, lane in enumerate(lanes):
        validate_lane(lane, ordinal, root)
    trace_files = [lane["trace"]["file"] for lane in lanes]
    if len(set(trace_files)) != len(trace_files):
        raise BrowserReceiptError("browser lanes must retain distinct trace files")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise BrowserReceiptError("browser receipt identity mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.receipt.absolute(),
        args.source_revision,
        args.release,
        args.artifact.absolute(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
