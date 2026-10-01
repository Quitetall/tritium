from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "verify-qwen36-official-source-identity.py"
SPEC = importlib.util.spec_from_file_location("qwen36_official_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def setup_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    weights = b"small deterministic weights"
    config = b"official config fixture"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(weights)
    (model_dir / "config.json").write_bytes(config)

    monkeypatch.setattr(MODULE, "SNAPSHOT_FILE_COUNT", 2)
    monkeypatch.setattr(MODULE, "SNAPSHOT_TOTAL_BYTES", len(weights) + len(config))
    monkeypatch.setattr(MODULE, "SOURCE_MODEL_ID", "a" * 64)
    monkeypatch.setattr(MODULE, "MANIFEST_CONTENT_ID", "tsc1_" + "b" * 64)
    monkeypatch.setattr(MODULE, "SOURCE_PROOF_ID", "tsc1_" + "c" * 64)
    monkeypatch.setattr(MODULE, "SOURCE_TENSOR_PAYLOAD_BYTES", len(weights))

    receipt = {
        "proof_id": MODULE.SOURCE_PROOF_ID,
        "manifest_content_id": MODULE.MANIFEST_CONTENT_ID,
        "source_model_id": MODULE.SOURCE_MODEL_ID,
        "repository": MODULE.REPOSITORY,
        "revision": MODULE.REVISION,
        "identity_status": "measured-awaiting-official-registration",
        "official_payload_authenticated": False,
        "proof_bytes": 10,
        "payload_bytes": MODULE.SOURCE_TENSOR_PAYLOAD_BYTES,
        "work_dir": "/tmp/source-work",
        "proof_path": "/tmp/source-work/ingest.tq36",
        "total_tensors": 4,
        "total_coefficients": 32,
        "language_tensors": 2,
        "language_coefficients": 20,
        "mtp_tensors": 1,
        "mtp_coefficients": 5,
        "vision_tensors": 1,
        "vision_coefficients": 7,
        "additive_tensors": 2,
        "additive_coefficients": 20,
        "preserved_tensors": 1,
        "preserved_coefficients": 5,
        "excluded_vision_tensors": 1,
        "excluded_vision_coefficients": 7,
    }
    admission = {
        "schema": "tritium.qwen36-source-admission.v1",
        "result": "pass",
        "receipt": receipt,
        "proof_sha256": "d" * 64,
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_bytes(canonical(admission) + b"\n")

    metadata = {
        "id": MODULE.REPOSITORY,
        "sha": MODULE.REVISION,
        "siblings": [
            {
                "rfilename": "weights.safetensors",
                "size": len(weights),
                "lfs": {"sha256": hashlib.sha256(weights).hexdigest(), "size": len(weights)},
            },
            {"rfilename": "config.json", "size": len(config), "blobId": git_blob_sha1(config)},
        ],
    }
    monkeypatch.setattr(
        MODULE, "OFFICIAL_MANIFEST_SHA256", MODULE._manifest_sha256(MODULE._official_files(metadata))
    )
    return model_dir, admission_path, metadata


def test_verifies_lfs_and_git_blob_files_without_upgrading_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir, admission_path, metadata = setup_case(tmp_path, monkeypatch)
    verified = MODULE.verify_snapshot(model_dir, admission_path, metadata, "d" * 64)

    assert verified["schema"] == MODULE.SCHEMA
    assert verified["result"] == "pass"
    assert verified["source_model_id"] == MODULE.SOURCE_MODEL_ID
    assert verified["verified_file_count"] == 2
    assert verified["verified_total_bytes"] == MODULE.SNAPSHOT_TOTAL_BYTES
    assert {item["algorithm"] for item in verified["files"]} == {"sha256", "git-sha1"}
    assert "official_payload_authenticated" not in verified
    assert verified["receipt_id"].startswith("sha256:")
    receipt_path = tmp_path / "identity.json"
    receipt_path.write_bytes(json.dumps(verified).encode() + b"\n")
    assert MODULE.validate_identity_receipt(receipt_path)["receipt_id"] == verified["receipt_id"]


@pytest.mark.parametrize("change", ["contents", "extra-file", "revision", "admission"])
def test_rejects_snapshot_or_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
):
    model_dir, admission_path, metadata = setup_case(tmp_path, monkeypatch)
    if change == "contents":
        (model_dir / "config.json").write_bytes(b"altered config")
    elif change == "extra-file":
        (model_dir / "unexpected.bin").write_bytes(b"extra")
    elif change == "revision":
        metadata["sha"] = "0" * 40
    else:
        value = json.loads(admission_path.read_text())
        value["receipt"]["official_payload_authenticated"] = True
        admission_path.write_bytes(canonical(value) + b"\n")

    with pytest.raises(MODULE.OfficialIdentityError):
        MODULE.verify_snapshot(model_dir, admission_path, metadata, "d" * 64)


def test_rejects_symlinked_source_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir, admission_path, metadata = setup_case(tmp_path, monkeypatch)
    config = model_dir / "config.json"
    target = tmp_path / "target.json"
    target.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target)

    with pytest.raises(MODULE.OfficialIdentityError):
        MODULE.verify_snapshot(model_dir, admission_path, metadata, "d" * 64)


def test_identity_receipt_validator_rejects_a_forged_file_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir, admission_path, metadata = setup_case(tmp_path, monkeypatch)
    receipt = MODULE.verify_snapshot(model_dir, admission_path, metadata, "d" * 64)
    receipt["files"][0]["digest"] = "e" * len(receipt["files"][0]["digest"])
    path = tmp_path / "identity.json"
    path.write_bytes(json.dumps(receipt).encode())

    with pytest.raises(MODULE.OfficialIdentityError, match="official Hub inventory"):
        MODULE.validate_identity_receipt(path)
