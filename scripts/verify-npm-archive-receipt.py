#!/usr/bin/env python3
"""Validate a release-bound npm archive qualification receipt."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "tritium.npm-archive-qualification.v1"
TOP_FIELDS = {
    "schema", "receipt_id", "release", "source_revision", "run_id",
    "started_at_utc", "duration_ms", "machine", "toolchain", "artifact",
    "evidence", "result",
}
MACHINE_FIELDS = {"machine_id", "system", "architecture"}
TOOLCHAIN_FIELDS = {"node", "npm"}
ARTIFACT_FIELDS = {"kind", "name", "package", "bytes", "sha256", "integrity"}
EVIDENCE_FIELDS = {
    "source_dirty", "entry_count", "source_free", "installed_offline",
    "strict_typescript", "wasm_build_id", "wasm_guest_digest",
}
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 1024 * 1024


class NpmReceiptError(ValueError):
    """The npm receipt is malformed, stale, dirty, or artifact-unbound."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise NpmReceiptError(f"{label} fields do not match the frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NpmReceiptError(f"{label} must be a non-empty string")
    return value


def _hex(value: Any, length: int, label: str) -> str:
    text = _string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise NpmReceiptError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def validate_receipt(
    path: Path, archive: Path, revision: str, release: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise NpmReceiptError("npm receipt must be a bounded ordinary file")
    if archive.is_symlink() or not archive.is_file():
        raise NpmReceiptError("npm archive must be an ordinary file")
    try:
        value = _object(json.loads(path.read_bytes()), TOP_FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NpmReceiptError("npm receipt must contain UTF-8 JSON") from error
    if value["schema"] != SCHEMA or value["result"] != "pass":
        raise NpmReceiptError("npm receipt schema or result mismatch")
    if value["release"] != release or value["source_revision"] != revision:
        raise NpmReceiptError("npm receipt release identity mismatch")
    _string(value["run_id"], "receipt.run_id")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(value["started_at_utc"])) is None:
        raise NpmReceiptError("npm receipt timestamp is invalid")
    duration = value["duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise NpmReceiptError("npm receipt duration is invalid")
    machine = _object(value["machine"], MACHINE_FIELDS, "receipt.machine")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(machine["machine_id"])) is None or any(
        not isinstance(machine[field], str) or not machine[field]
        for field in ("system", "architecture")
    ):
        raise NpmReceiptError("npm receipt machine identity is invalid")
    toolchain = _object(value["toolchain"], TOOLCHAIN_FIELDS, "receipt.toolchain")
    if re.fullmatch(r"v[0-9]+(?:\.[0-9]+){2}", str(toolchain["node"])) is None or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+){1,2}", str(toolchain["npm"])
    ) is None:
        raise NpmReceiptError("npm receipt toolchain is invalid")
    artifact = _object(value["artifact"], ARTIFACT_FIELDS, "receipt.artifact")
    if artifact["kind"] != "npm-archive" or artifact["name"] != archive.name:
        raise NpmReceiptError("npm receipt artifact identity mismatch")
    if (
        Path(str(artifact["name"])).name != artifact["name"]
        or "\\" in str(artifact["name"])
        or "\0" in str(artifact["name"])
        or not str(artifact["name"]).endswith(".tgz")
    ):
        raise NpmReceiptError("npm receipt archive name is unsafe")
    if artifact["package"] != f"@tritium-ai/web@{release}":
        raise NpmReceiptError("npm receipt package identity mismatch")
    if type(artifact["bytes"]) is not int or artifact["bytes"] != archive.stat().st_size:
        raise NpmReceiptError("npm receipt archive byte count mismatch")
    archive_sha256 = _sha256(archive)
    if _hex(artifact["sha256"], 64, "receipt.artifact.sha256") != archive_sha256:
        raise NpmReceiptError("npm receipt archive digest mismatch")
    expected_integrity = "sha512-" + base64.b64encode(
        hashlib.sha512(archive.read_bytes()).digest()
    ).decode("ascii")
    if artifact["integrity"] != expected_integrity:
        raise NpmReceiptError("npm receipt archive integrity mismatch")
    evidence = _object(value["evidence"], EVIDENCE_FIELDS, "receipt.evidence")
    if evidence["source_dirty"] is not False or any(
        evidence[field] is not True
        for field in ("source_free", "installed_offline", "strict_typescript")
    ):
        raise NpmReceiptError("npm receipt does not prove a clean qualified install")
    if type(evidence["entry_count"]) is not int or evidence["entry_count"] != 13:
        raise NpmReceiptError("npm receipt archive entry count is invalid")
    expected_build = f"tritium-wasm@{release}+source-git:{revision}"
    if evidence["wasm_build_id"] != expected_build:
        raise NpmReceiptError("npm receipt WASM build identity mismatch")
    _hex(evidence["wasm_guest_digest"], 64, "receipt.evidence.wasm_guest_digest")
    unsigned = dict(value)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise NpmReceiptError("npm receipt identity mismatch")
    return value
