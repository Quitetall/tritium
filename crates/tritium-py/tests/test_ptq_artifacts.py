"""Strict PTQ artifact and phased facade gates from plan 0047."""

import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

import tritium.torch.artifacts as artifacts  # noqa: E402
import tritium.torch.ptq as ptq  # noqa: E402
from tritium.torch import (  # noqa: E402
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    inspect,
    load,
    prepare,
    quantize,
)


def _write_bundle(root: Path) -> None:
    root.mkdir()
    (root / "compact.tsalt2").write_bytes(b"compact")
    (root / "near-lossless.tsalt2").write_bytes(b"near")
    (root / "preserved.safetensors").write_bytes(b"preserved")
    manifest = {
        "schema_version": 2,
        "artifact_kind": "qwen3.6-language-mtp-salt-v2-model-weights",
        "complete_model": False,
        "packing": "b3",
        "completion_id": "completion",
        "campaign_id": "campaign",
        "admission_id": "admission",
        "selection_id": "selection",
        "source_model_id": "source",
        "source_identity_status": "candidate",
        "official_payload_authenticated": False,
        "profiles": {
            "compact-v1": {
                "file": "compact.tsalt2",
                "package_id": "compact-id",
                "serialized_bytes": 7,
                "resident_bytes": 11,
            },
            "near-lossless-v1": {
                "file": "near-lossless.tsalt2",
                "package_id": "near-id",
                "serialized_bytes": 4,
                "resident_bytes": 13,
            },
        },
        "preserved": {
            "file": "preserved.safetensors",
            "package_id": "preserved-id",
            "tensors": 360,
            "payload_bytes": 5_343_232,
            "serialized_bytes": 9,
        },
    }
    (root / "tritium.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _fake_verify(path, package_id, serialized, resident):
    assert Path(path).stat().st_size == serialized
    return package_id, "b3", serialized, resident


def _fake_verify_preserved(path, package_id, tensors, payload, serialized):
    assert Path(path).stat().st_size == serialized
    return package_id, tensors, payload, serialized


def _upgrade_bundle_to_v3(root: Path) -> None:
    manifest_path = root / "tritium.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = []
    for filename in artifacts._HF_ASSET_FILES:
        payload = f"asset:{filename}".encode()
        (root / filename).write_bytes(payload)
        assets.append(
            {
                "file": filename,
                "package_id": f"id:{filename}",
                "bytes": len(payload),
            }
        )
    manifest.update(
        schema_version=3,
        artifact_kind="qwen3.6-language-mtp-salt-v2-hf-bundle",
        source_revision=artifacts._QWEN36_REVISION,
        hf_assets=assets,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_bundle_load_and_export_reverify_exact_profiles(monkeypatch, tmp_path):
    source = tmp_path / "source"
    _write_bundle(source)
    verified = []

    def tracking_verify(path, package_id, serialized, resident):
        verified.append(Path(path))
        return _fake_verify(path, package_id, serialized, resident)

    def tracking_verify_preserved(path, package_id, tensors, payload, serialized):
        verified.append(Path(path))
        return _fake_verify_preserved(path, package_id, tensors, payload, serialized)

    def publish_after_staged_verification(staging, target):
        staging = Path(staging)
        assert sum(path.parent == staging for path in verified) == 3
        staging.rename(target)

    monkeypatch.setattr(
        artifacts._tritium,
        "verify_salt_v2_package",
        tracking_verify,
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_preserved_safetensors",
        tracking_verify_preserved,
        raising=False,
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "publish_directory_noreplace",
        publish_after_staged_verification,
    )

    result = load(source)
    assert result.complete_model is False
    assert result.schema_version == 2
    assert result.preserved.tensors == 360
    assert result.artifact("compact-v1").serialized_bytes == 7
    assert result.near_lossless.resident_bytes == 13
    with pytest.raises(TritiumError) as caught:
        result.save_pretrained(tmp_path / "hf")
    assert caught.value.code == "incomplete_artifact"
    assert "preserved_bf16_tensors" not in caught.value.details["missing"]

    receipt = result.export(tmp_path / "copy")
    assert receipt.admission_id == "admission"
    assert (receipt.artifact_dir / "compact.tsalt2").read_bytes() == b"compact"
    assert receipt.preserved_package_id == "preserved-id"
    assert load(receipt.artifact_dir).compact.package_id == "compact-id"
    with pytest.raises(FileExistsError):
        result.export(receipt.artifact_dir)


def test_bundle_schema_rejects_unknown_fields_before_native_load(monkeypatch, tmp_path):
    source = tmp_path / "source"
    _write_bundle(source)
    manifest = json.loads((source / "tritium.json").read_text(encoding="utf-8"))
    manifest["unbound"] = True
    (source / "tritium.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_salt_v2_package",
        lambda *args: pytest.fail("native load must follow schema validation"),
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_preserved_safetensors",
        lambda *args: pytest.fail("native load must follow schema validation"),
        raising=False,
    )
    with pytest.raises(ValueError, match="fields"):
        load(source)


def test_v3_hf_assets_are_strictly_loaded_and_exported(monkeypatch, tmp_path):
    source = tmp_path / "source"
    _write_bundle(source)
    _upgrade_bundle_to_v3(source)

    monkeypatch.setattr(artifacts._tritium, "verify_salt_v2_package", _fake_verify)
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_preserved_safetensors",
        _fake_verify_preserved,
        raising=False,
    )

    verified_assets = []

    def verify_asset(path, package_id, byte_count):
        path = Path(path)
        assert path.stat().st_size == byte_count
        verified_assets.append(path)
        return package_id, byte_count

    monkeypatch.setattr(
        artifacts._tritium, "verify_hf_asset", verify_asset, raising=False
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "publish_directory_noreplace",
        lambda staging, target: Path(staging).rename(target),
    )

    result = load(source)
    assert result.schema_version == 3
    assert tuple(asset.file for asset in result.hf_assets) == artifacts._HF_ASSET_FILES
    result.save_pretrained(tmp_path / "hf")
    saved = load(tmp_path / "hf")
    assert saved.compact.package_id == result.compact.package_id
    assert tuple(asset.file for asset in saved.hf_assets) == artifacts._HF_ASSET_FILES

    receipt = result.export(tmp_path / "copy")
    assert receipt.schema_version == 3
    assert (receipt.artifact_dir / "tokenizer.json").read_bytes() == b"asset:tokenizer.json"
    assert sum(path.parent == receipt.artifact_dir for path in verified_assets) == 8


def test_v3_device_load_dispatches_governed_profile_to_native_runtime(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    _write_bundle(source)
    _upgrade_bundle_to_v3(source)
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_salt_v2_package",
        lambda *args: pytest.fail("native device load must not pre-hash both profiles"),
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_preserved_safetensors",
        lambda *args: pytest.fail("native device load owns preserved verification"),
        raising=False,
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_hf_asset",
        lambda *args: pytest.fail("native device load owns HF asset verification"),
        raising=False,
    )

    class FakeQwenModel:
        @staticmethod
        def load(path, *, profile, device):
            return Path(path), profile, device

    monkeypatch.setattr(artifacts._tritium, "QwenModel", FakeQwenModel)
    assert load(source, device="cpu", profile="near-lossless-v1") == (
        source.resolve(),
        "near-lossless-v1",
        "cpu",
    )


def test_v3_hf_asset_catalog_rejects_reordering_before_asset_io(monkeypatch, tmp_path):
    source = tmp_path / "source"
    _write_bundle(source)
    _upgrade_bundle_to_v3(source)
    manifest_path = source / "tritium.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hf_assets"][0], manifest["hf_assets"][1] = (
        manifest["hf_assets"][1],
        manifest["hf_assets"][0],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(artifacts._tritium, "verify_salt_v2_package", _fake_verify)
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_preserved_safetensors",
        _fake_verify_preserved,
        raising=False,
    )
    monkeypatch.setattr(
        artifacts._tritium,
        "verify_hf_asset",
        lambda *args: pytest.fail("asset verifier must follow catalog validation"),
        raising=False,
    )
    with pytest.raises(ValueError, match="canonical order"):
        load(source)


def test_local_ptq_prepare_and_calibration_are_explicit(monkeypatch, tmp_path):
    model = tmp_path / "model"
    evidence = tmp_path / "evidence"
    model.mkdir()
    evidence.mkdir()
    config = TernaryConfig.ptq(profile="compact-v1")
    prepared = prepare(model, config, inplace=False)
    assert prepared.model == model.resolve()
    assert prepared.coverage is None
    with pytest.raises(TritiumError) as caught:
        inspect(prepared)
    assert caught.value.code == "coverage_pending"
    with pytest.raises(TritiumError, match="never mutates"):
        prepare(model, config, inplace=True)

    monkeypatch.setattr(
        ptq._tritium,
        "inspect_qwen36_ptq_evidence",
        lambda *args, **kwargs: (
            "evidence-id",
            "guided-fisher",
            506,
            "source-id",
            "cache-id",
            "tokens-id",
        ),
    )
    receipt = calibrate(prepared, evidence_dir=evidence, max_evidence_bytes=123)
    assert receipt.evidence_id == "evidence-id"
    assert receipt.record_count == 506
    assert receipt.max_evidence_bytes == 123
    with pytest.raises(TritiumError, match="raw calibration"):
        calibrate(prepared, object(), evidence_dir=evidence)

    rate_prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_bpw=2.0),
        inplace=False,
    )
    with pytest.raises(TritiumError, match="exact byte ceilings") as caught:
        convert(
            rate_prepared,
            receipt,
            revision="revision",
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            compact_max_bytes=1,
            compact_max_resident_bytes=1,
            near_lossless_max_bytes=1,
            near_lossless_max_resident_bytes=1,
        )
    assert caught.value.code == "unsupported_recipe"


def test_live_module_calibration_streams_bounded_source_bound_curvature(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1, bias=False),
    )
    model.train()
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    batches = (
        torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]]),
        torch.tensor([[0.0, 1.0, 2.0]]),
    )
    evidence = tmp_path / "activation-evidence"
    receipt = calibrate(
        prepared,
        batches,
        evidence_dir=evidence,
        max_evidence_bytes=1024,
    )

    assert isinstance(receipt, ptq.ActivationCalibrationReceipt)
    assert receipt.curvature == "diagonal-second-moment-f64le"
    assert receipt.record_count == 2
    assert receipt.records[0].module == "0"
    assert receipt.records[0].samples == 3
    assert receipt.records[0].features == 3
    assert prepared.model.training is True
    values = struct.unpack("<3d", (evidence / receipt.records[0].file).read_bytes())
    assert values == pytest.approx((5.0, 5.0, 14.0))
    manifest = json.loads((evidence / "calibration.json").read_text())
    assert manifest["evidence_id"] == receipt.evidence_id
    assert manifest["source_model_digest"] == receipt.source_model_digest
    assert manifest["token_stream_digest"] == receipt.token_stream_digest
    assert receipt.evidence_id.startswith("sha256:")
    assert ptq.load_activation_calibration(evidence, max_evidence_bytes=1024) == receipt

    (evidence / receipt.records[0].file).write_bytes(b"\x00" * receipt.records[0].bytes)
    with pytest.raises(ValueError, match="digest mismatch"):
        ptq.load_activation_calibration(evidence)


def test_live_module_calibration_fails_closed_on_incomplete_or_oversize_data(tmp_path):
    class Conditional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(2, 2, bias=False)
            self.skipped = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value):
            return self.used(value)

    prepared = prepare(
        Conditional(),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    with pytest.raises(TritiumError) as caught:
        calibrate(
            prepared,
            [torch.ones(1, 2)],
            evidence_dir=tmp_path / "incomplete",
        )
    assert caught.value.code == "incomplete_coverage"
    assert not (tmp_path / "incomplete").exists()

    linear = prepare(
        torch.nn.Linear(3, 2, bias=False),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    with pytest.raises(TritiumError) as caught:
        calibrate(
            linear,
            [torch.ones(1, 3)],
            evidence_dir=tmp_path / "oversize",
            max_evidence_bytes=8,
        )
    assert caught.value.code == "evidence_too_large"
    assert not (tmp_path / "oversize").exists()

    unsupported = prepare(
        torch.nn.Sequential(
            torch.nn.Embedding(8, 3),
            torch.nn.Linear(3, 2, bias=False),
        ),
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Embedding", "Linear")
        ),
        inplace=False,
    )
    with pytest.raises(TritiumError) as caught:
        calibrate(
            unsupported,
            [torch.tensor([[1, 2]])],
            evidence_dir=tmp_path / "unsupported",
        )
    assert caught.value.code == "unsupported_module"
    assert caught.value.details["parameters"] == ["0.weight"]

    class SharedLinears(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(2, 2, bias=False)
            self.right = torch.nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight

        def forward(self, value):
            return self.left(value) + self.right(value)

    shared = prepare(
        SharedLinears(),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    with pytest.raises(TritiumError) as caught:
        calibrate(
            shared,
            [torch.ones(1, 2)],
            evidence_dir=tmp_path / "shared",
        )
    assert caught.value.code == "unsupported_shared_parameter"


def test_live_module_fit_consumes_bound_curvature_and_rejects_source_drift(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -0.4, 0.1], [-0.7, 0.2, 0.9]]))
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    receipt = calibrate(
        prepared,
        [torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])],
        evidence_dir=tmp_path / "fit-evidence",
    )
    result = ptq.fit(prepared, receipt)
    assert result.evidence_id == receipt.evidence_id
    assert result.algorithm_id == "tritium.diagonal-additive-3@1"
    assert len(result.weights) == 1
    assert len(result.weights[0].planes) == 3
    assert result.weights[0].weighted_mse >= 0
    for plane in result.weights[0].planes:
        assert set(plane.trits.unique().tolist()) <= {-1, 0, 1}
        assert torch.isfinite(plane.scales).all()
        assert (plane.scales >= 0).all()

    with torch.no_grad():
        prepared.model.weight[0, 0] += 1
    with pytest.raises(TritiumError) as caught:
        ptq.fit(prepared, receipt)
    assert caught.value.code == "source_changed"


def test_quantize_composes_the_three_public_phases(monkeypatch, tmp_path):
    sentinel_prepared = object()
    sentinel_calibration = object()
    sentinel_result = object()
    calls = []
    monkeypatch.setattr(
        ptq, "prepare", lambda *args, **kwargs: calls.append("prepare") or sentinel_prepared
    )
    monkeypatch.setattr(
        ptq,
        "calibrate",
        lambda *args, **kwargs: calls.append("calibrate") or sentinel_calibration,
    )
    monkeypatch.setattr(
        ptq, "convert", lambda *args, **kwargs: calls.append("convert") or sentinel_result
    )
    result = quantize(
        tmp_path / "model",
        TernaryConfig.ptq(profile="near-lossless-v1"),
        revision="revision",
        work_dir=tmp_path / "work",
        evidence_dir=tmp_path / "evidence",
        output_dir=tmp_path / "output",
        compact_max_bytes=1,
        compact_max_resident_bytes=2,
        near_lossless_max_bytes=3,
        near_lossless_max_resident_bytes=4,
    )
    assert result is sentinel_result
    assert calls == ["prepare", "calibrate", "convert"]
