import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import tritium.torch.onnx as onnx  # noqa: E402
from tritium.torch import (  # noqa: E402
    QwenOnnxCausalLM,
    TritiumError,
    export_onnx,
    load_onnx,
)
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
        device = "cpu"

        @staticmethod
        def load(path, *, device):
            observed.update(path=path, device=device)
            return Runtime()

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    assert isinstance(load_onnx(root, device="cuda:0"), QwenOnnxCausalLM)
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


def test_load_onnx_rejects_symlinked_graph_before_runtime(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    (root / "language.onnx").unlink()
    (root / "language.onnx").symlink_to(root / "mtp.onnx")
    monkeypatch.setattr(
        onnx._tritium,
        "QwenOnnxModel",
        SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("must not load")),
        raising=False,
    )
    with pytest.raises(ValueError, match="ordinary file"):
        load_onnx(root)


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


def test_loaded_model_returns_batch_one_logits_and_flat_authenticated_state(
    monkeypatch, tmp_path
):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    observed = {}

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(path, *, device):
            assert path == str(root.resolve())
            assert device == "cpu"
            return Runtime()

        def forward_language(self, token_ids, states):
            observed.update(token_ids=token_ids, states=states)
            return SimpleNamespace(
                logits_shape=[2, 3],
                logits=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                state_names=["next_conv.0", "present_k.1"],
                state_shapes=[[2], [1, 1, 2]],
                states=[[7.0, 8.0], [9.0, 10.0]],
            )

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    result = model(torch.tensor([[4, 5]], dtype=torch.int64))
    assert result.logits.shape == (1, 2, 3)
    assert result.logits[0, -1].tolist() == [4.0, 5.0, 6.0]
    assert tuple(state.shape for state in result.past_key_values) == (
        torch.Size([2]),
        torch.Size([1, 1, 2]),
    )
    assert result.state_names == ("next_conv.0", "present_k.1")
    assert observed == {"token_ids": [4, 5], "states": None}


def test_loaded_model_validates_past_and_refuses_unqualified_generation(
    monkeypatch, tmp_path
):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    observed = {}

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(*args, **kwargs):
            return Runtime()

        def forward_language(self, token_ids, states):
            observed.update(token_ids=token_ids, states=states)
            return SimpleNamespace(
                logits_shape=[1, 2],
                logits=[0.0, 1.0],
                state_names=[],
                state_shapes=[],
                states=[],
            )

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    model(
        torch.tensor([1]),
        past_key_values=(torch.tensor([2.0], dtype=torch.float32),),
        return_dict=False,
    )
    assert observed["states"] == [[2.0]]
    with pytest.raises(TritiumError) as caught:
        model.generate(torch.tensor([[1]]))
    assert caught.value.code == "dynamic_onnx_generation_unavailable"
