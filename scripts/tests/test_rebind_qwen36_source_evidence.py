from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "rebind-qwen36-source-evidence.py"
SPEC = importlib.util.spec_from_file_location("qwen36_source_evidence_rebind", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def setup_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    proof = b"deterministic source proof bytes"
    proof_path = tmp_path / "proof.tq36"
    proof_path.write_bytes(proof)
    admission_value = {
        "schema": "tritium.qwen36-source-admission.v1",
        "result": "pass",
        "receipt": {
            "proof_id": "tsc1_" + "1" * 64,
            "manifest_content_id": "tsc1_" + "2" * 64,
            "source_model_id": "3" * 64,
            "repository": "Qwen/Qwen3.6-27B",
            "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "proof_bytes": len(proof),
            "proof_path": "/tmp/old/ingest.tq36",
        },
        "proof_sha256": hashlib.sha256(proof).hexdigest(),
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_bytes(canonical(admission_value))
    admission_id = "sha256:" + hashlib.sha256(canonical(admission_value)).hexdigest()
    identity_value = {
        "schema": "tritium.qwen36-official-source-identity.v1",
        "source_admission_receipt_id": admission_id,
        "repository": admission_value["receipt"]["repository"],
        "revision": admission_value["receipt"]["revision"],
        "source_model_id": admission_value["receipt"]["source_model_id"],
        "manifest_content_id": admission_value["receipt"]["manifest_content_id"],
        "source_proof_id": admission_value["receipt"]["proof_id"],
        "receipt_id": "sha256:" + "a" * 64,
    }
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(canonical(identity_value))

    def validate_admission(path, *_args):
        value = json.loads(Path(path).read_bytes())
        return {"receipt_id": "sha256:" + hashlib.sha256(canonical(value)).hexdigest(), "receipt": value["receipt"]}

    def validate_identity(path):
        return json.loads(Path(path).read_bytes())

    monkeypatch.setattr(MODULE, "validate_source_admission", validate_admission)
    monkeypatch.setattr(MODULE, "validate_identity", validate_identity)
    return proof_path, admission_path, identity_path, admission_value, identity_value


def test_rebinds_receipts_for_byte_identical_relocated_proof(tmp_path: Path, monkeypatch):
    proof, admission, identity, old_admission, old_identity = setup_case(tmp_path, monkeypatch)
    output = tmp_path / "out" / "rebound"
    output.parent.mkdir()

    result = MODULE.rebind(admission, identity, proof, output)

    new_admission = json.loads((output / "source-admission.json").read_bytes())
    new_identity = json.loads((output / "official-source-identity.json").read_bytes())
    rebind = json.loads((output / "rebind.json").read_bytes())
    assert (output / "source-proof.tq36").read_bytes() == proof.read_bytes()
    assert new_admission["receipt"]["proof_path"] == str(output / "source-proof.tq36")
    assert new_identity["source_admission_receipt_id"] == result["source_admission_receipt_id"]
    assert result["source_admission_receipt_id"] != "sha256:" + hashlib.sha256(canonical(old_admission)).hexdigest()
    assert new_identity["repository"] == old_identity["repository"]
    assert rebind["source_admission_parent_receipt_id"] == "sha256:" + hashlib.sha256(canonical(old_admission)).hexdigest()
    assert rebind["official_identity_parent_receipt_id"] == old_identity["receipt_id"]
    assert rebind["changed_fields"] == [
        "source_admission.receipt.proof_path",
        "official_identity.source_admission_receipt_id",
    ]
    assert admission.read_bytes() == canonical(old_admission)
    assert identity.read_bytes() == canonical(old_identity)


def test_rejects_proof_hash_mismatch_without_creating_output(tmp_path: Path, monkeypatch):
    proof, admission, identity, *_ = setup_case(tmp_path, monkeypatch)
    proof.write_bytes(b"not the admitted proof")
    output = tmp_path / "out" / "rebound"
    output.parent.mkdir()

    with pytest.raises(MODULE.RebindError, match="differs"):
        MODULE.rebind(admission, identity, proof, output)
    assert not output.exists()


def test_refuses_to_replace_existing_output(tmp_path: Path, monkeypatch):
    proof, admission, identity, *_ = setup_case(tmp_path, monkeypatch)
    output = tmp_path / "out" / "rebound"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace")

    with pytest.raises(MODULE.RebindError, match="new path"):
        MODULE.rebind(admission, identity, proof, output)
    assert sentinel.read_text() == "do not replace"
