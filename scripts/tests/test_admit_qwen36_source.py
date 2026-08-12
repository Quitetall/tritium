from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "admit-qwen36-source.py"
SPEC = importlib.util.spec_from_file_location("admit_qwen36_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = MODULE.RECEIPT_FIELDS


def fake_receipt(proof: Path):
    values = {field: 1 for field in FIELDS}
    for field in ("proof_id", "manifest_content_id", "source_model_id"):
        values[field] = "id"
    for field in ("identity_status", "repository", "revision", "work_dir", "proof_path"):
        values[field] = {
            "identity_status": "measured-awaiting-official-registration",
            "repository": "Qwen/Qwen3.6-27B",
            "revision": MODULE.PINNED_REVISION,
            "work_dir": str(proof.parent),
            "proof_path": str(proof),
        }[field]
    values["official_payload_authenticated"] = False
    values["proof_bytes"] = proof.stat().st_size
    return type("Receipt", (), values)()


def test_build_record_binds_proof_digest_and_candidate_status(tmp_path: Path):
    proof = tmp_path / "ingest.tq36"
    proof.write_bytes(b"proof")
    record = MODULE.build_record(fake_receipt(proof))

    assert record["schema"] == MODULE.SCHEMA
    assert record["result"] == "pass"
    assert record["receipt"]["proof_bytes"] == len(b"proof")
    assert record["receipt"]["official_payload_authenticated"] is False
    assert record["proof_sha256"] == hashlib.sha256(b"proof").hexdigest()


def test_write_new_is_canonical_and_never_replaces(tmp_path: Path):
    output = tmp_path / "receipt.json"
    MODULE._write_new(output, {"z": 1, "a": "x"})
    assert output.read_bytes() == b'{"a":"x","z":1}\n'
    with pytest.raises(MODULE.AdmissionReceiptError, match="replace existing"):
        MODULE._write_new(output, {"a": "different"})


def test_write_new_rejects_symlink_parent(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(MODULE.AdmissionReceiptError, match="symlink"):
        MODULE._write_new(link / "receipt.json", {"ok": True})


def test_receipt_value_rejects_authenticated_native_claim(tmp_path: Path):
    proof = tmp_path / "proof"
    proof.write_bytes(b"proof")
    receipt = fake_receipt(proof)
    receipt.official_payload_authenticated = True
    with pytest.raises(MODULE.AdmissionReceiptError, match="official payload"):
        MODULE.receipt_value(receipt)
