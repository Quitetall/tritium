from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/register-qwen36-source-identity.py"
SPEC = importlib.util.spec_from_file_location("register_qwen36_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ADMISSION_MODULE = runpy.run_path(ROOT / "scripts/verify-qwen36-source-admission-receipt.py")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def admission_record() -> dict:
    source = {
        "proof_id": "tsc1_" + "1" * 64,
        "manifest_content_id": "tsc1_" + "2" * 64,
        "source_model_id": "3" * 64,
        "repository": "Qwen/Qwen3.6-27B",
        "revision": ADMISSION_MODULE["PINNED_REVISION"],
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
    return {
        "schema": ADMISSION_MODULE["SCHEMA"],
        "result": "pass",
        "receipt": source,
        "proof_sha256": "4" * 64,
    }


def setup_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    admission = admission_record()
    admission_bytes = canonical(admission) + b"\n"
    candidate_admission = candidate_root / "source.json"
    candidate_admission.write_bytes(admission_bytes)
    candidate = candidate_root / "manifest.json"
    candidate_document = {
        "schema": "tritium.release-candidate.v1",
        "release": "1.1.0-rc.1",
        "source_revision": ADMISSION_MODULE["PINNED_REVISION"],
        "artifacts": [{
            "id": "qwen-source",
            "kind": "source-admission",
            "path": "source.json",
            "identity": {},
            "sbom": {},
            "provenance": {},
        }],
    }
    candidate_bytes = json.dumps(candidate_document, indent=2).encode() + b"\n"
    candidate.write_bytes(candidate_bytes)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    registered_admission = evidence / "source.json"
    registered_admission.write_bytes(admission_bytes)
    validated_admission = ADMISSION_MODULE["validate"](
        registered_admission,
        ADMISSION_MODULE["PINNED_REVISION"],
        "1.1.0-rc.1",
        candidate,
    )
    base = {
        "schema": "tritium.release-evidence-registry.v1",
        "release": candidate_document["release"],
        "source_revision": candidate_document["source_revision"],
        "candidate_manifest_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "receipts": [{
            "id": validated_admission["receipt_id"],
            "kind": "source-admission",
            "path": "source.json",
            "sha256": hashlib.sha256(admission_bytes).hexdigest(),
            "artifact_id": "qwen-source",
            "parents": [],
        }],
    }
    base_path = evidence / "registry.json"
    base_path.write_bytes(canonical(base) + b"\n")
    identity_path = tmp_path / "official-identity.json"
    identity_path.write_text("{}\n")
    identity = {
        "receipt_id": "sha256:" + "5" * 64,
        "source_admission_receipt_id": validated_admission["receipt_id"],
        "repository": admission["receipt"]["repository"],
        "revision": admission["receipt"]["revision"],
        "source_model_id": admission["receipt"]["source_model_id"],
        "manifest_content_id": admission["receipt"]["manifest_content_id"],
        "source_proof_id": admission["receipt"]["proof_id"],
    }
    monkeypatch.setitem(
        MODULE.register.__globals__, "validate_official_identity", lambda _: identity
    )
    monkeypatch.setitem(
        MODULE.RELEASE_STATUS["evaluate"].__globals__,
        "validate_official_source_identity",
        lambda _: identity,
    )
    output = evidence / "registry-with-identity.json"
    return base_path, candidate, registered_admission, identity_path, output, identity


def test_registers_identity_as_exact_admission_child_and_validates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base, candidate, admission, identity_path, output, identity = setup_case(
        tmp_path, monkeypatch
    )
    report = MODULE.register(base, candidate, admission, identity_path, output)

    assert output.is_file()
    gate = next(row for row in report["rows"] if row["id"] == "qwen-source-admission")
    assert gate["status"] == "PASS"
    registry = json.loads(output.read_bytes())
    registered = registry["receipts"][-1]
    assert registered["kind"] == "official-source-identity"
    assert registered["id"] == identity["receipt_id"]
    assert registered["parents"] == [identity["source_admission_receipt_id"]]
    assert (output.parent / registered["path"]).read_text() == "{}\n"


def test_refuses_mismatched_source_admission_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base, candidate, admission, identity_path, output, identity = setup_case(
        tmp_path, monkeypatch
    )
    identity["source_admission_receipt_id"] = "sha256:" + "0" * 64
    with pytest.raises(MODULE.RegistrationError, match="different admission"):
        MODULE.register(base, candidate, admission, identity_path, output)
    assert not output.exists()


def test_refuses_to_replace_registry_or_receipt_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base, candidate, admission, identity_path, output, _identity = setup_case(
        tmp_path, monkeypatch
    )
    output.write_text("preserve me")
    with pytest.raises(MODULE.RegistrationError, match="existing output"):
        MODULE.register(base, candidate, admission, identity_path, output)
    assert output.read_text() == "preserve me"
