"""Real ORT gates for packed generic module ONNX bundles."""

import json

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from tritium.nn import AdditiveTernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    export_module_onnx,
    load_quantized_module,
    load_module_onnx,
    prepare,
)


def _model():
    planes = []
    trits = torch.tensor(
        [[1, -1, 0, 1, -1, 0, 1, -1], [0, 1, 1, -1, 0, -1, 1, 0]],
        dtype=torch.int8,
    )
    for index in range(3):
        planes.append(
            SimpleNamespace(
                trits=trits,
                scales=torch.tensor(
                    [[0.5 / (index + 1)], [0.25 / (index + 1)]],
                    dtype=torch.float16,
                ),
                group_size=8,
            )
        )
    return AdditiveTernaryLinear(planes, torch.tensor([0.1, -0.2])).eval()


def _external_data_model():
    plane = SimpleNamespace(
        trits=torch.randint(-1, 2, (128, 128), dtype=torch.int8),
        scales=torch.full((128, 1), 0.25, dtype=torch.float16),
        group_size=128,
    )
    return AdditiveTernaryLinear((plane,)).eval()


def test_module_onnx_keeps_packed_state_runs_ort_and_supports_dynamic_batch(tmp_path):
    model = _model()
    example = torch.randn(2, 8)
    artifact = export_module_onnx(model, example, tmp_path / "bundle")
    assert artifact.checkpoint_digest.startswith("sha256:")
    assert load_module_onnx(artifact.artifact_dir, create_session=False) == artifact

    graph = onnx.load(artifact.artifact_dir / "model.onnx", load_external_data=False)
    initializers = {value.name: value for value in graph.graph.initializer}
    assert {
        f"_packed_weight.packed_trits_{index}" for index in range(3)
    } <= set(initializers)
    assert not any(
        value.data_type
        in {
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.BFLOAT16,
        }
        and tuple(value.dims) == (2, 8)
        for value in initializers.values()
    )
    runtime = load_module_onnx(artifact.artifact_dir)
    replay = torch.randn(5, 8)
    torch.testing.assert_close(runtime(replay), model(replay), rtol=1e-4, atol=1e-5)

    graph_path = artifact.artifact_dir / "model.onnx"
    payload = bytearray(graph_path.read_bytes())
    payload[-1] ^= 1
    graph_path.write_bytes(payload)
    with pytest.raises(ValueError, match="file identity mismatch"):
        load_module_onnx(artifact.artifact_dir)


def test_module_onnx_rejects_optimizer_dense_shadow_and_rolls_back(monkeypatch, tmp_path):
    original = torch.onnx.export

    def optimized(*args, **kwargs):
        kwargs["optimize"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", optimized)
    output = tmp_path / "bundle"
    with pytest.raises(TritiumError) as captured:
        export_module_onnx(_model(), torch.randn(2, 8), output)
    assert captured.value.code == "dense_shadow_detected"
    assert not output.exists()


def test_module_onnx_checks_external_data_from_graph_directory(tmp_path):
    artifact = export_module_onnx(
        _external_data_model(), torch.randn(1, 128), tmp_path / "bundle"
    )
    external = artifact.artifact_dir / "model.onnx.data"
    assert external.is_file()
    assert external.stat().st_size > 0
    runtime = load_module_onnx(artifact.artifact_dir)
    assert runtime(torch.randn(2, 128)).shape == (2, 128)


def test_huggingface_ptq_exports_tied_embedding_and_dynamic_sequence(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=True,
        use_cache=False,
    )
    source = transformers.LlamaForCausalLM(config).eval()
    prepared = prepare(
        source,
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Linear", "Embedding")
        ),
        inplace=False,
    )
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    calibration = calibrate(
        prepared,
        [{"input_ids": tokens, "use_cache": False}],
        evidence_dir=tmp_path / "evidence",
    )
    conversion = convert(prepared, calibration, work_dir=tmp_path / "work")
    model = load_quantized_module(prepared.model, conversion).eval()
    model.config.use_cache = False

    artifact = export_module_onnx(
        model,
        tokens,
        tmp_path / "bundle",
        input_names=("input_ids",),
        output_names=("logits",),
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
    )
    graph = onnx.load(artifact.artifact_dir / "model.onnx", load_external_data=False)
    initializer_names = {value.name for value in graph.graph.initializer}
    shared_prefix = "model.embed_tokens._packed_weight"
    assert f"{shared_prefix}.packed_trits_0" in initializer_names
    assert not any(name.startswith("lm_head._packed_weight") for name in initializer_names)
    manifest = json.loads(
        (artifact.artifact_dir / "tritium-module-onnx.json").read_text()
    )
    embedding = next(
        spec
        for spec in manifest["packed_modules"]
        if spec["path"] == "model.embed_tokens"
    )
    head = next(spec for spec in manifest["packed_modules"] if spec["path"] == "lm_head")
    assert embedding["storage_path"] == head["storage_path"] == shared_prefix
    assert embedding["packed_initializers"] == head["packed_initializers"]
    assert embedding["scale_initializers"] == head["scale_initializers"]

    replay = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    expected = model(replay).logits.detach()
    observed = load_module_onnx(artifact.artifact_dir)(replay)
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
