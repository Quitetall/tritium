"""Strict PTQ artifact and phased facade gates from plan 0047."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

import tritium.torch.artifacts as artifacts  # noqa: E402
import tritium.torch.ptq as ptq  # noqa: E402
from tritium.nn import AdditiveTernaryEmbedding, AdditiveTernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    inspect,
    load,
    load_quantized_module,
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
    assert receipt.records[0].outputs == 2
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


def test_live_module_calibration_restores_mixed_submodule_training_modes(tmp_path):
    class MixedModes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
            self.right = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))

        def forward(self, value):
            return self.right(self.left(value))

    prepared = prepare(
        MixedModes(),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    prepared.model.train()
    prepared.model.left.eval()
    prepared.model.right.train()
    before = {
        name: module.training for name, module in prepared.model.named_modules()
    }
    calibrate(
        prepared,
        [torch.ones(2, 3)],
        evidence_dir=tmp_path / "mixed-mode-evidence",
    )
    after = {
        name: module.training for name, module in prepared.model.named_modules()
    }
    assert after == before
    assert before["left"] is False
    assert before["right"] is True


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
    receipt = calibrate(
        shared,
        [torch.ones(1, 2)],
        evidence_dir=tmp_path / "shared",
    )
    assert receipt.record_count == 1
    assert receipt.records[0].module == "left"
    assert set(receipt.records[0].weight_aliases) == {
        "left.weight",
        "right.weight",
    }
    # Both consumers run once; curvature records total observed rows, not
    # merely rows seen through the canonical module alias.
    assert receipt.records[0].samples == 2


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
    work_dir = tmp_path / "conversion-work"
    result = convert(prepared, receipt, work_dir=work_dir)
    reopened = ptq.load_module_conversion(result.artifact_dir)
    resumed = convert(prepared, receipt, work_dir=work_dir)
    assert reopened.recipe_id == result.recipe_id
    assert resumed == result
    assert result.artifact_dir == work_dir.resolve()
    assert result.evidence_id == receipt.evidence_id
    assert result.algorithm_id == "tritium.diagonal-additive-3@1"
    assert result.recipe_id.startswith("sha256:")
    assert result.config == prepared.config
    assert result.coverage == prepared.coverage
    fitted = result.weight("weight")
    assert result.weight_names == ("weight",)
    assert len(fitted.planes) == 3
    assert fitted.weighted_mse == pytest.approx(9.7121347e-5)
    assert fitted.planes[0].trits[0].tolist() == [1, -1, 0]
    assert fitted.planes[0].scales[0].item() == 0.7001953125
    assert fitted.planes[1].trits[0].tolist() == [1, 1, 0]
    assert fitted.planes[1].scales[0].item() == 0.300048828125
    assert fitted.planes[2].trits[0].tolist() == [0, 0, 1]
    assert fitted.planes[2].scales[0].item() == 0.0999755859375
    for plane in fitted.planes:
        assert set(plane.trits.unique().tolist()) <= {-1, 0, 1}
        assert torch.isfinite(plane.scales).all()
        assert (plane.scales >= 0).all()

    alternate = prepare(
        model,
        TernaryConfig.ptq(profile="near-lossless-v1", target_modules=("Linear",)),
        inplace=False,
    )
    alternate_result = convert(
        alternate, receipt, work_dir=tmp_path / "alternate-work"
    )
    assert alternate_result.recipe_id != result.recipe_id

    rate_limited = prepare(
        model,
        TernaryConfig.ptq(
            profile="compact-v1",
            target_modules=("Linear",),
            target_bpw=2.0,
        ),
        inplace=False,
    )
    adaptive_result = convert(
        rate_limited, receipt, work_dir=tmp_path / "rate-work"
    )
    assert adaptive_result.algorithm_id == "tritium.diagonal-additive-adaptive@1"
    assert adaptive_result.achieved_bpw <= 2.0
    assert len(adaptive_result.weight("weight").planes) == 1
    assert load_quantized_module(model, adaptive_result)(
        torch.ones(1, 3)
    ).shape == (1, 2)

    with torch.no_grad():
        prepared.model.weight[0, 0] += 1
    with pytest.raises(TritiumError) as caught:
        convert(prepared, receipt, work_dir=tmp_path / "drift-work")
    assert caught.value.code == "source_changed"


def test_adaptive_module_artifact_packs_variable_weight_plane_count(tmp_path):
    model = torch.nn.Linear(64, 2, bias=False)
    prepared = prepare(
        model,
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Linear",), target_bpw=2.0
        ),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 64)],
        evidence_dir=tmp_path / "adaptive-pack-evidence",
    )
    result = convert(
        prepared, calibration, work_dir=tmp_path / "adaptive-pack-work"
    )
    assert 1 <= len(result.weight("weight").planes) <= 3
    packed = result.pack_native(tmp_path / "adaptive-pack")
    assert packed.tensors == 1
    assert packed.serialized_bytes > 0


def test_live_module_convert_fits_later_planes_against_stored_low_precision_scale(
    tmp_path,
):
    weight = torch.tensor(
        [
            [
                0.599121,
                1.831055,
                -2.544922,
                -0.483154,
                0.517578,
                -2.634766,
                1.576172,
                2.402344,
            ]
        ],
        dtype=torch.float16,
    )
    curvature = torch.tensor(
        [
            2.540206,
            3.080229,
            3.783262,
            1.873168,
            2.096422,
            3.563944,
            2.739383,
            1.179104,
        ],
        dtype=torch.float32,
    )
    model = torch.nn.Linear(8, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        model.weight.copy_(weight)
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    receipt = calibrate(
        prepared,
        [curvature.sqrt().to(torch.float16).unsqueeze(0)],
        evidence_dir=tmp_path / "low-precision-evidence",
    )
    result = convert(prepared, receipt, work_dir=tmp_path / "low-precision-work")
    assert result.weight("weight").planes[2].trits[0].tolist() == [
        1,
        1,
        1,
        0,
        1,
        0,
        -1,
        1,
    ]


def test_live_module_convert_resumes_missing_weight_and_rejects_tampering(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    )
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    receipt = calibrate(
        prepared,
        [torch.tensor([[1.0, 2.0, 3.0]])],
        evidence_dir=tmp_path / "resume-evidence",
    )
    work = tmp_path / "resume-work"
    original = convert(prepared, receipt, work_dir=work)
    first_receipt = work / "weight-00000.json"
    first_receipt_time = first_receipt.stat().st_mtime_ns

    (work / "conversion.json").unlink()
    (work / "weight-00001.json").unlink()
    resumed = convert(prepared, receipt, work_dir=work)
    assert resumed.artifact_id == original.artifact_id
    assert first_receipt.stat().st_mtime_ns == first_receipt_time
    assert resumed.weight_names == ("0.weight", "1.weight")

    plane = resumed.weights[1].planes[0].trits_path
    plane.write_bytes(b"\x00" * plane.stat().st_size)
    with pytest.raises(ValueError, match="identity mismatch"):
        ptq.load_module_conversion(work)


def test_live_module_convert_rejects_self_consistent_wrong_resume_geometry(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 3)],
        evidence_dir=tmp_path / "geometry-evidence",
    )
    work = tmp_path / "geometry-work"
    convert(prepared, calibration, work_dir=work)
    (work / "conversion.json").unlink()
    receipt_path = work / "weight-00000.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["shape"] = [1, 6]
    receipt["fit_chunk_rows"] = 1
    for plane in receipt["planes"]:
        scale_path = work / plane["scales_file"]
        payload = scale_path.read_bytes()[:2]
        scale_path.write_bytes(payload)
        plane["scales_digest"] = "sha256:" + hashlib.sha256(payload).hexdigest()
        plane["scales_bytes"] = 2
        plane["scales_shape"] = [1, 1]
        plane["group_size"] = 6
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="identity differs from calibration"):
        convert(prepared, calibration, work_dir=work)


def test_live_module_convert_streams_deterministic_rows_under_working_ceiling(tmp_path):
    model = torch.nn.Linear(4, 5, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10)
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.tensor([[1.0, 2.0, 3.0, 4.0]])],
        evidence_dir=tmp_path / "tiled-evidence",
    )
    tiled = convert(
        prepared,
        calibration,
        work_dir=tmp_path / "tiled-work",
        max_working_bytes=1024,
    )
    roomy = convert(
        prepared,
        calibration,
        work_dir=tmp_path / "roomy-work",
        max_working_bytes=1024 * 1024,
    )
    assert tiled.weights[0].fit_chunk_rows < model.out_features
    assert tiled.weights[0].max_working_bytes == 1024
    assert [plane.trits_digest for plane in tiled.weights[0].planes] == [
        plane.trits_digest for plane in roomy.weights[0].planes
    ]
    assert [plane.scales_digest for plane in tiled.weights[0].planes] == [
        plane.scales_digest for plane in roomy.weights[0].planes
    ]


def test_live_module_convert_rejects_working_ceiling_below_one_row(tmp_path):
    prepared = prepare(
        torch.nn.Linear(4, 1, bias=False),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 4)],
        evidence_dir=tmp_path / "evidence",
    )
    with pytest.raises(TritiumError) as captured:
        convert(
            prepared,
            calibration,
            work_dir=tmp_path / "work",
            max_working_bytes=607,
        )
    assert captured.value.code == "working_set_too_small"
    assert captured.value.details == {
        "required_bytes": 608,
        "max_working_bytes": 607,
    }


def test_module_conversion_loader_keeps_v1_artifacts_readable(tmp_path):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "module_ptq_v1.json").read_text()
    )
    work = tmp_path / "work"
    work.mkdir()
    for name, payload_hex in fixture.items():
        (work / name).write_bytes(bytes.fromhex(payload_hex))

    reopened = ptq.load_module_conversion(work)
    assert reopened.schema_version == 1
    assert reopened.weights[0].fit_chunk_rows == 1
    assert reopened.weights[0].max_working_bytes is None
    assert reopened.weight("weight").planes[0].trits.tolist() == [[0, -1]]


def test_module_conversion_loader_rejects_cross_version_receipts(tmp_path):
    prepared = prepare(
        torch.nn.Linear(2, 1, bias=False),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 2)],
        evidence_dir=tmp_path / "evidence",
    )
    work = tmp_path / "work"
    convert(prepared, calibration, work_dir=work)

    manifest_path = work / "conversion.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["artifact_kind"] = "tritium.module-additive-ptq-v1"
    identity = dict(manifest)
    del identity["artifact_id"]
    manifest["artifact_id"] = "sha256:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="receipt fields differ from schema"):
        ptq.load_module_conversion(work)


def test_load_quantized_module_binds_compact_planes_without_dense_master(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(128, 2, bias=True))
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.arange(128, dtype=torch.float32).reshape(1, 128) / 128],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    loaded = load_quantized_module(prepared.model, artifact)

    assert isinstance(loaded[0], AdditiveTernaryLinear)
    assert loaded is not prepared.model
    assert dict(loaded.named_parameters()) == {}
    assert loaded[0].physical_bytes < prepared.model[0].weight.numel() * 2
    assert all(
        name.startswith(
            (
                "0._packed_weight.packed_trits_",
                "0._packed_weight.scales_",
                "0.bias",
            )
        )
        for name in loaded.state_dict()
    )
    inputs = torch.randn(3, 128)
    fitted = artifact.weight("0.weight")
    assert loaded[0].packed_weight.trit_counts() == tuple(
        tuple(int(torch.count_nonzero(plane.trits == value)) for value in (-1, 0, 1))
        for plane in fitted.planes
    )
    expected = sum(
        torch.nn.functional.linear(inputs, plane.trits.float())
        * plane.scales.float().squeeze(-1)
        for plane in fitted.planes
    ) + prepared.model[0].bias
    torch.testing.assert_close(loaded(inputs), expected)
    assert hasattr(prepared.model[0], "weight")
    inplace = load_quantized_module(prepared.model, artifact, inplace=True)
    assert inplace is prepared.model
    assert isinstance(inplace[0], AdditiveTernaryLinear)


def test_load_quantized_module_shares_tied_embedding_storage(tmp_path):
    class TiedEmbeddingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4, padding_idx=0)
            self.head = torch.nn.Linear(4, 8, bias=False)
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
    tokens = torch.tensor([[0, 1, 2]])
    calibration = calibrate(
        prepared,
        [tokens],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    loaded = load_quantized_module(prepared.model, artifact)

    assert isinstance(loaded.embed, AdditiveTernaryEmbedding)
    assert isinstance(loaded.head, AdditiveTernaryLinear)
    assert loaded.embed.packed_weight is loaded.head.packed_weight
    assert loaded.embed.padding_idx == 0
    assert dict(loaded.named_parameters()) == {}
    assert loaded.embed(tokens).shape == (1, 3, 4)
    assert loaded(tokens).shape == (1, 3, 8)


def test_load_quantized_module_rejects_source_drift(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    prepared = prepare(
        model,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 3)],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    with torch.no_grad():
        prepared.model.weight[0, 0] += 1
    with pytest.raises(TritiumError) as captured:
        load_quantized_module(prepared.model, artifact)
    assert captured.value.code == "source_changed"


def test_module_conversion_rejects_rebound_calibration_aliases(tmp_path):
    class TiedLinears(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(2, 2, bias=False)
            self.right = torch.nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight

        def forward(self, value):
            return self.left(value) + self.right(value)

    prepared = prepare(
        TiedLinears(),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    evidence = tmp_path / "evidence"
    calibration = calibrate(
        prepared,
        [torch.ones(1, 2)],
        evidence_dir=evidence,
    )
    manifest_path = evidence / "calibration.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["weight_aliases"] = [
        "right.weight",
        "left.weight",
    ]
    identity = dict(manifest)
    identity.pop("evidence_id")
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    manifest["evidence_id"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tampered = ptq.load_activation_calibration(evidence)
    with pytest.raises(TritiumError) as captured:
        convert(prepared, tampered, work_dir=tmp_path / "work")
    assert captured.value.code == "coverage_mismatch"


def test_load_quantized_module_rejects_linear_subclass_semantic_loss(tmp_path):
    class ShiftedLinear(torch.nn.Linear):
        def forward(self, value):
            return super().forward(value) + 1

    prepared = prepare(
        ShiftedLinear(3, 2, bias=False),
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [torch.ones(1, 3)],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    with pytest.raises(TritiumError) as captured:
        load_quantized_module(prepared.model, artifact)
    assert captured.value.code == "coverage_mismatch"


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
