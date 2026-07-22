import hashlib
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from tritium.torch.tutorial_qat import (  # noqa: E402
    run_installed_qat_tutorial,
    validate_tutorial_receipt,
)


def _rehash(receipt):
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(payload).hexdigest()


def test_installed_qat_tutorial_result_strictly_reopens(tmp_path):
    output = tmp_path / "tutorial"
    receipt = run_installed_qat_tutorial(output, device_name="cpu")
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_tutorial_receipt(path, expected_device="cpu") == receipt


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("converted_parameters", 2, "coverage"),
        ("aliases", ["embed.weight"], "aliases"),
        ("algorithm_id", "tritium.salt-ste", "estimator"),
        ("planes", 1, "plane"),
        ("device", "cuda:0", "device"),
    ],
)
def test_installed_qat_tutorial_rejects_rehashed_claim_drift(
    tmp_path, field, value, message
):
    output = tmp_path / "tutorial"
    receipt = run_installed_qat_tutorial(output, device_name="cpu")
    receipt[field] = value
    _rehash(receipt)
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_tutorial_receipt(path, expected_device="cpu")
