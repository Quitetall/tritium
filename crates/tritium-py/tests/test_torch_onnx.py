import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import tritium.torch.onnx as onnx  # noqa: E402
from tritium.torch import TritiumError, export_onnx, load_onnx  # noqa: E402
from tritium.torch.artifacts import QuantizationResult  # noqa: E402
from tritium.torch.conversion import PreparedModel  # noqa: E402


def _write_onnx_bundle(root: Path):
    root.mkdir()
    (root / "language.onnx").write_bytes(b"language")
    (root / "mtp.onnx").write_bytes(b"mtp")
    (root / "weights.bin").write_bytes(b"weights")
    value = {
        "schema": "tritium-qwen35-onnx-bundle-v1",
        "language": {"file": "language.onnx", "blake3": "11" * 32},
        "mtp": {"file": "mtp.onnx", "blake3": "22" * 32},
        "weights": {"file": "weights.bin", "blake3": "33" * 32, "bytes": 7},
        "identity": {
            "source_model_id": "source",
            "tokenizer_id": "tokenizer",
            "recipe_id": "recipe",
            "package_id": "package",
            "converted_coverage_id": "converted",
            "deferred_coverage_id": "vision",
        },
        "conversion": {
            "mode": "ptq",
            "completion_id": "completion",
            "campaign_id": "campaign",
            "admission_id": "admission",
            "selection_id": "selection",
        },
    }
    (root / "tritium-onnx-manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def test_load_onnx_parses_exact_manifest_before_native_runtime(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    observed = {}

    class Runtime:
        @staticmethod
        def load(path, *, device):
            observed.update(path=path, device=device)
            return "runtime"

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    assert load_onnx(root, device="cuda:0") == "runtime"
    assert observed == {"path": str(root.resolve()), "device": "cuda:0"}


def test_load_onnx_rejects_unknown_or_corrupt_manifest_before_runtime(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    value = _write_onnx_bundle(root)
    value["unknown"] = True
    (root / "tritium-onnx-manifest.json").write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        onnx._tritium,
        "QwenOnnxModel",
        SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("must not load")),
        raising=False,
    )
    with pytest.raises(ValueError, match="top-level"):
        load_onnx(root)


def test_load_onnx_has_stable_capability_error(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    monkeypatch.delattr(onnx._tritium, "QwenOnnxModel", raising=False)
    with pytest.raises(TritiumError) as caught:
        load_onnx(root)
    assert caught.value.code == "onnx_runtime_unavailable"


def test_export_onnx_rejects_latent_training_graph():
    source = PreparedModel(model=object(), config=object(), coverage=None)
    with pytest.raises(TritiumError) as caught:
        export_onnx(source, "out")
    assert caught.value.code == "trainable_onnx_requires_v1_3"


def test_export_onnx_forwards_complete_ptq_ancestry(monkeypatch, tmp_path):
    source = QuantizationResult(
        artifact_dir=tmp_path / "source",
        packing="b3",
        completion_id="completion",
        campaign_id="campaign",
        admission_id="admission",
        selection_id="selection",
        source_model_id="source",
        source_identity_status="candidate",
        official_payload_authenticated=False,
        compact=SimpleNamespace(profile="compact-v1"),
        near_lossless=SimpleNamespace(profile="near-lossless-v1"),
        preserved=object(),
        hf_assets=(object(),),
        source_revision="revision",
        schema_version=3,
    )
    observed = {}

    def export(*args, **kwargs):
        observed.update(args=args, kwargs=kwargs)
        return "receipt"

    monkeypatch.setattr(onnx._tritium, "export_qwen35_onnx_bundle", export, raising=False)
    assert export_onnx(source, tmp_path / "out", tokens=2, past_tokens=3) == "receipt"
    assert observed["args"] == (str(source.artifact_dir), str(tmp_path / "out"))
    assert observed["kwargs"]["profile"] == "compact-v1"
    assert observed["kwargs"]["tokens"] == 2
    assert observed["kwargs"]["past_tokens"] == 3
