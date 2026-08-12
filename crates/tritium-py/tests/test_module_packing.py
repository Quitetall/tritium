"""Strict generic conversion to seek-backed SALT V2 package gates."""

import hashlib
import json

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from tritium.torch import TernaryConfig, calibrate, convert, prepare  # noqa: E402
from tritium.torch.module_artifacts import (  # noqa: E402
    load_module_conversion,
    load_packed_module,
)


def test_module_conversion_streams_strict_native_salt_package(tmp_path):
    model = torch.nn.Linear(384, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(
            torch.arange(1152, dtype=torch.float32).reshape(3, 384) / 1152 - 0.5
        )
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 384)],
        evidence_dir=tmp_path / "evidence",
    )
    converted = convert(prepared, calibration, work_dir=tmp_path / "work")
    fitted = converted.weight("weight")
    assert fitted.planes[0].group_size == 128
    assert fitted.planes[0].scales.shape == (3, 3)
    packed = converted.pack_native(tmp_path / "packed", packing="b3")

    assert packed.conversion_artifact_id == converted.artifact_id
    assert packed.recipe_id == converted.recipe_id
    assert packed.packing == "b3"
    assert packed.tensors == 1
    assert packed.complete_model is False
    assert load_packed_module(packed.artifact_dir) == packed

    manifest_path = packed.artifact_dir / "tritium.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["weights"]["tensors"] = 2
    identity = dict(manifest)
    del identity["artifact_id"]
    manifest["artifact_id"] = "sha256:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tensor count"):
        load_packed_module(packed.artifact_dir)
    manifest_path.write_bytes(original_manifest)

    weights = packed.artifact_dir / "weights.tsalt2"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(ValueError):
        load_packed_module(packed.artifact_dir)


def test_native_package_preserves_legacy_g64_scales_on_g128_width(tmp_path):
    model = torch.nn.Linear(256, 2, bias=False)
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 256)],
        evidence_dir=tmp_path / "evidence",
    )
    converted = convert(prepared, calibration, work_dir=tmp_path / "work")
    work = converted.artifact_dir

    # Reopen a valid conversion after widening its canonical G128 scales into
    # distinct legacy G64 values. Package admission must retain both values,
    # not silently coarsen them back to one G128 scale.
    receipt_path = work / "weight-00000.json"
    receipt = json.loads(receipt_path.read_text())
    for plane in receipt["planes"]:
        scales_path = work / plane["scales_file"]
        scales = torch.frombuffer(
            bytearray(scales_path.read_bytes()), dtype=torch.float16
        ).reshape(2, 2)
        payload = scales.repeat_interleave(2, dim=1).contiguous().numpy().tobytes()
        scales_path.write_bytes(payload)
        plane["scales_bytes"] = len(payload)
        plane["scales_digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
        plane["scales_shape"] = [2, 4]
        plane["group_size"] = 64
    receipt_payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    receipt_path.write_bytes(receipt_payload)
    manifest_path = work / "conversion.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["weight_receipts"][0] = {
        "file": "weight-00000.json",
        "digest": "sha256:" + hashlib.sha256(receipt_payload).hexdigest(),
        "bytes": len(receipt_payload),
    }
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = "sha256:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    widened = load_module_conversion(work)
    packed = widened.pack_native(tmp_path / "packed", packing="b3")
    assert packed.tensors == 1
    assert packed.serialized_bytes == 472
    assert load_packed_module(packed.artifact_dir) == packed


def test_module_conversion_native_pack_rejects_unaligned_g64_weights(tmp_path):
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
    with pytest.raises(ValueError, match="G64 alignment"):
        converted.pack_native(tmp_path / "packed")


def test_module_conversion_rejects_missing_selected_coverage_weight(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
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
    work = tmp_path / "work"
    convert(prepared, calibration, work_dir=work)

    manifest_path = work / "conversion.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["weight_receipts"].pop()
    identity = dict(manifest)
    del identity["artifact_id"]
    manifest["artifact_id"] = "sha256:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for path in work.glob("weight-00001*"):
        path.unlink()

    with pytest.raises(ValueError, match="selected coverage"):
        load_module_conversion(work)


def test_module_conversion_coverage_bijection_is_order_independent(tmp_path):
    class TiedEmbeddingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 4)
            self.head = torch.nn.Linear(4, 4, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, tokens):
            return self.head(self.embed(tokens))

    prepared = prepare(
        TiedEmbeddingHead(),
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Linear", "Embedding")
        ),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.tensor([[0, 1, 2]])],
        evidence_dir=tmp_path / "evidence",
    )
    converted = convert(prepared, calibration, work_dir=tmp_path / "work")

    assert converted.weights[0].path == "embed.weight"
    assert converted.weights[0].aliases == ("embed.weight", "head.weight")
    assert load_module_conversion(converted.artifact_dir) == converted
