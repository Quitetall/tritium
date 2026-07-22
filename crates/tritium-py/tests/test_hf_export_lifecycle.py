"""Whole-model Hugging Face hard export/reload qualification tests."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import tritium  # noqa: E402
from tritium.torch import hf_export_lifecycle as lifecycle  # noqa: E402


def _qualify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wheel = tmp_path / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"candidate wheel")
    monkeypatch.setattr(
        lifecycle,
        "_installed_distribution",
        lambda: ("1.1.0rc0", Path(tritium.__file__).resolve()),
    )
    output = tmp_path / "evidence"
    receipt = lifecycle.qualify_hf_export(
        output,
        wheel_artifact=wheel,
        source_revision="a" * 40,
        release="1.1.0-rc.0",
        run_id="test-export-1",
    )
    return wheel, output, receipt


def test_whole_model_export_reloads_exact_logits_and_generation(tmp_path, monkeypatch):
    wheel, output, receipt = _qualify(tmp_path, monkeypatch)
    assert receipt["passed"] is True
    assert receipt["converted_parameters"] == 8
    assert receipt["tied_after_reload"] is True
    assert receipt["no_dense_weight_shadows"] is True
    assert len(receipt["generated_ids"][0]) == 6
    assert (
        lifecycle.validate_hf_export_receipt(
            output / "receipt.json",
            expected_wheel=wheel,
            expected_source_revision="a" * 40,
            expected_release="1.1.0-rc.0",
        )
        == receipt
    )


def test_whole_model_export_rejects_artifact_tampering(tmp_path, monkeypatch):
    wheel, output, _ = _qualify(tmp_path, monkeypatch)
    state = output / "qat-hard" / "model.safetensors"
    payload = bytearray(state.read_bytes())
    payload[-1] ^= 1
    state.write_bytes(payload)
    with pytest.raises(ValueError, match="tree identity"):
        lifecycle.validate_hf_export_receipt(
            output / "receipt.json", expected_wheel=wheel
        )


def test_whole_model_export_rejects_rehashed_false_claim(tmp_path, monkeypatch):
    wheel, output, receipt = _qualify(tmp_path, monkeypatch)
    receipt["no_dense_weight_shadows"] = False
    receipt["receipt_id"] = lifecycle.receipt_id(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="no_dense_weight_shadows"):
        lifecycle.validate_hf_export_receipt(
            output / "receipt.json", expected_wheel=wheel
        )
