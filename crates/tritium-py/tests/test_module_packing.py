"""Strict generic conversion to seek-backed SALT V2 package gates."""

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tritium.torch import TernaryConfig, calibrate, convert, prepare  # noqa: E402
from tritium.torch.module_artifacts import load_packed_module  # noqa: E402


def test_module_conversion_streams_strict_native_salt_package(tmp_path):
    model = torch.nn.Linear(128, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(
            torch.arange(256, dtype=torch.float32).reshape(2, 128) / 256 - 0.5
        )
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 128)],
        evidence_dir=tmp_path / "evidence",
    )
    converted = convert(prepared, calibration, work_dir=tmp_path / "work")
    packed = converted.pack_native(tmp_path / "packed", packing="b3")

    assert packed.conversion_artifact_id == converted.artifact_id
    assert packed.recipe_id == converted.recipe_id
    assert packed.packing == "b3"
    assert packed.tensors == 1
    assert packed.complete_model is False
    assert load_packed_module(packed.artifact_dir) == packed

    weights = packed.artifact_dir / "weights.tsalt2"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(ValueError):
        load_packed_module(packed.artifact_dir)


def test_module_conversion_native_pack_rejects_unaligned_g128_weights(tmp_path):
    prepared = prepare(
        torch.nn.Linear(3, 1, bias=False),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 3)],
        evidence_dir=tmp_path / "evidence",
    )
    converted = convert(prepared, calibration, work_dir=tmp_path / "work")
    with pytest.raises(ValueError, match="G128 alignment"):
        converted.pack_native(tmp_path / "packed")
