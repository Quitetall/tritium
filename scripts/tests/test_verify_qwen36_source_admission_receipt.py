from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "verify-qwen36-source-admission-receipt.py"
SPEC = importlib.util.spec_from_file_location("qwen36_source_admission_verify", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def record() -> dict:
    receipt = {
        "proof_id": "tsc1_" + "1" * 64,
        "manifest_content_id": "tsc1_" + "2" * 64,
        "source_model_id": "3" * 64,
        "repository": "Qwen/Qwen3.6-27B",
        "revision": MODULE.PINNED_REVISION,
        "identity_status": "measured-awaiting-official-registration",
        "official_payload_authenticated": False,
        "proof_bytes": 10,
        "payload_bytes": 100,
        "work_dir": "/tmp/work",
        "proof_path": "/tmp/work/ingest.tq36",
        "total_tensors": 12,
        "total_coefficients": 120,
        "language_tensors": 8,
        "language_coefficients": 80,
        "mtp_tensors": 2,
        "mtp_coefficients": 20,
        "vision_tensors": 2,
        "vision_coefficients": 20,
        "additive_tensors": 6,
        "additive_coefficients": 60,
        "preserved_tensors": 4,
        "preserved_coefficients": 40,
        "excluded_vision_tensors": 2,
        "excluded_vision_coefficients": 20,
    }
    value = {
        "schema": MODULE.SCHEMA,
        "result": "pass",
        "receipt": receipt,
        "proof_sha256": "4" * 64,
    }
    return value


def write(path: Path, value: dict) -> None:
    path.write_bytes(canonical(value) + b"\n")


def test_validates_inventory_and_returns_registry_identity(tmp_path: Path):
    path = tmp_path / "receipt.json"
    value = record()
    write(path, value)
    result = MODULE.validate(path, MODULE.PINNED_REVISION, "1.1.0-rc.1", path)
    assert result["receipt_id"] == "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("revision", "a" * 40, "pinned Qwen revision"),
        ("official_payload_authenticated", True, "official authentication"),
        ("vision_tensors", 3, "partition totals"),
        ("proof_bytes", 0, "proof and payload"),
    ],
)
def test_rejects_stale_or_inconsistent_receipt(
    tmp_path: Path, field: str, replacement: object, message: str
):
    value = record()
    value["receipt"][field] = replacement
    path = tmp_path / "receipt.json"
    write(path, value)
    with pytest.raises(MODULE.SourceAdmissionError, match=message):
        MODULE.validate(path, MODULE.PINNED_REVISION, "1.1.0-rc.1", path)


def test_rejects_duplicate_json_fields(tmp_path: Path):
    path = tmp_path / "receipt.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(MODULE.SourceAdmissionError, match="strict UTF-8 JSON"):
        MODULE.validate(path, MODULE.PINNED_REVISION, "1.1.0-rc.1", path)


def test_source_admission_is_a_distinct_release_gate():
    release_module = runpy.run_path(
        Path(__file__).resolve().parents[1] / "release-evidence-status.py"
    )
    assert ("qwen-source-admission", ("source-admission",)) in release_module["GATES"]
