#!/usr/bin/env python3
"""Validate a Qwen3.6 source-admission receipt for release evidence.

Source admission proves checkpoint inventory and proof identity only. It does
not qualify calibration, PTQ, refinement, runtime, or official payload
authentication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "tritium.qwen36-source-admission.v1"
PINNED_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
RECEIPT_FIELDS = {
    "proof_id",
    "manifest_content_id",
    "source_model_id",
    "repository",
    "revision",
    "identity_status",
    "official_payload_authenticated",
    "proof_bytes",
    "payload_bytes",
    "work_dir",
    "proof_path",
    "total_tensors",
    "total_coefficients",
    "language_tensors",
    "language_coefficients",
    "mtp_tensors",
    "mtp_coefficients",
    "vision_tensors",
    "vision_coefficients",
    "additive_tensors",
    "additive_coefficients",
    "preserved_tensors",
    "preserved_coefficients",
    "excluded_vision_tensors",
    "excluded_vision_coefficients",
}
TOP_FIELDS = {"schema", "result", "receipt", "proof_sha256"}
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


class SourceAdmissionError(ValueError):
    """Receipt is malformed, stale, or contradicts source inventory."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceAdmissionError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise SourceAdmissionError(f"{label} must be a nonnegative integer")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in HEX for character in text):
        raise SourceAdmissionError(f"{label} must be lowercase hexadecimal SHA-256")
    return text


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SourceAdmissionError("source-admission receipt must be an ordinary file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise SourceAdmissionError("source-admission receipt exceeds size bounds")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SourceAdmissionError("source-admission receipt must contain strict UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != TOP_FIELDS:
        raise SourceAdmissionError("source-admission receipt fields differ")
    return value


def validate(
    receipt_path: Path,
    expected_revision: str,
    _expected_release: str,
    _candidate: Path,
) -> dict[str, Any]:
    """Validate receipt and return registry-compatible identity-bearing value."""

    value = _load(receipt_path)
    if value["schema"] != SCHEMA or value["result"] != "pass":
        raise SourceAdmissionError("source-admission schema or result differs")
    receipt = value["receipt"]
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise SourceAdmissionError("source-admission receipt fields differ")
    if expected_revision != PINNED_REVISION or receipt["revision"] != PINNED_REVISION:
        raise SourceAdmissionError("source-admission revision is not the pinned Qwen revision")
    if receipt["repository"] != "Qwen/Qwen3.6-27B":
        raise SourceAdmissionError("source-admission repository differs")
    if receipt["official_payload_authenticated"] is not False:
        raise SourceAdmissionError("source-admission cannot claim official authentication")
    _string(receipt["identity_status"], "receipt.identity_status")
    for field in (
        "proof_id", "manifest_content_id", "source_model_id", "work_dir", "proof_path",
    ):
        _string(receipt[field], f"receipt.{field}")
    _digest(value["proof_sha256"], "proof_sha256")
    for field in RECEIPT_FIELDS - {
        "proof_id", "manifest_content_id", "source_model_id", "repository",
        "revision", "identity_status", "official_payload_authenticated", "work_dir",
        "proof_path",
    }:
        _nonnegative_int(receipt[field], f"receipt.{field}")
    if receipt["proof_bytes"] == 0 or receipt["payload_bytes"] == 0:
        raise SourceAdmissionError("source-admission proof and payload must be nonempty")
    tensor_totals = {
        "total_tensors": receipt["language_tensors"] + receipt["mtp_tensors"] + receipt["vision_tensors"],
        "total_coefficients": receipt["language_coefficients"] + receipt["mtp_coefficients"] + receipt["vision_coefficients"],
    }
    for field, expected in tensor_totals.items():
        if receipt[field] != expected:
            raise SourceAdmissionError(f"receipt.{field} does not equal source partition totals")
    if receipt["language_tensors"] + receipt["mtp_tensors"] != receipt["additive_tensors"] + receipt["preserved_tensors"]:
        raise SourceAdmissionError("language/MTP tensor coverage is incomplete")
    if receipt["language_coefficients"] + receipt["mtp_coefficients"] != receipt["additive_coefficients"] + receipt["preserved_coefficients"]:
        raise SourceAdmissionError("language/MTP coefficient coverage is incomplete")
    if receipt["total_tensors"] != receipt["additive_tensors"] + receipt["preserved_tensors"] + receipt["excluded_vision_tensors"]:
        raise SourceAdmissionError("tensor inventory decomposition differs")
    if receipt["total_coefficients"] != receipt["additive_coefficients"] + receipt["preserved_coefficients"] + receipt["excluded_vision_coefficients"]:
        raise SourceAdmissionError("coefficient inventory decomposition differs")
    identity = dict(value)
    identity["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return identity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = validate(args.receipt, args.revision, "not-applicable", args.receipt)
    print(json.dumps({"receipt_id": result["receipt_id"], "result": "pass"}, sort_keys=True))
