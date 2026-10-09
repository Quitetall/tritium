"""Real ORT gates for packed generic module ONNX bundles."""

import copy
import json

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from tritium.nn import (  # noqa: E402
    AdditiveTernaryLinear,
    AdditiveTernaryWeight,
    TernaryLinear,
)
from tritium.torch import (  # noqa: E402
    ModuleOnnxLineage,
    RefinementConfig,
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    export,
    export_module_onnx,
    export_onnx,
    load_onnx,
    load_quantized_module,
    load_module_onnx,
    prepare,
    refine,
)
from tritium.torch.module_onnx import _session_options  # noqa: E402
from tritium.torch.estimators import ProjectionContext  # noqa: E402


def _assert_exact_packed_initializer_parity(model, graph):
    """Hard ternary codes and narrowed scales must survive ONNX export exactly."""
    from onnx import numpy_helper

    initializers = {value.name: value for value in graph.graph.initializer}
    for path, packed_weight in model.named_modules():
        if not isinstance(packed_weight, AdditiveTernaryWeight):
            continue
        prefix = f"{path}." if path else ""
        for plane in range(packed_weight.plane_count):
            for field, expected in (
                ("packed_trits", getattr(packed_weight, f"packed_trits_{plane}")),
                ("scales", getattr(packed_weight, f"scales_{plane}")),
            ):
                name = f"{prefix}{field}_{plane}"
                assert name in initializers
                actual = numpy_helper.to_array(initializers[name])
                expected_array = expected.detach().cpu().numpy()
                assert actual.dtype == expected_array.dtype
                assert actual.shape == expected_array.shape
                assert actual.tobytes(order="C") == expected_array.tobytes(order="C")


def test_packed_onnx_runtime_disables_dense_constant_folding():
    options = _session_options(ort)
    assert (
        options.graph_optimization_level
        == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
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


def test_packed_linear_decodes_weights_in_bounded_chunks(monkeypatch):
    plane = SimpleNamespace(
        trits=torch.zeros((300_000, 16), dtype=torch.int8),
        scales=torch.ones((300_000, 1), dtype=torch.float16),
        group_size=16,
    )
    model = AdditiveTernaryLinear((plane,), bias=None).eval()
    decoded_rows = []
    decode = AdditiveTernaryWeight._dense_rows

    def record_chunk(weight, indices, *, dtype):
        decoded_rows.append(indices.numel())
        return decode(weight, indices, dtype=dtype)

    monkeypatch.setattr(AdditiveTernaryWeight, "_dense_rows", record_chunk)
    actual = model(torch.ones((1, 16)))

    assert actual.shape == (1, 300_000)
    assert decoded_rows == [262_144, 37_856]
    assert all(rows * model.in_features <= 1 << 22 for rows in decoded_rows)


def test_module_onnx_keeps_packed_state_runs_ort_and_supports_dynamic_batch(tmp_path):
    model = _model()
    example = torch.randn(2, 8)
    lineage = ModuleOnnxLineage(
        mode="qat-hard",
        artifact_id="sha256:" + "11" * 32,
        recipe_id="sha256:" + "22" * 32,
        source_model_digest="sha256:" + "33" * 32,
    )
    artifact = export_module_onnx(
        model, example, tmp_path / "bundle", lineage=lineage,
    )
    assert artifact.checkpoint_digest.startswith("sha256:")
    assert artifact.schema_version == 2
    assert artifact.lineage == lineage
    assert load_module_onnx(artifact.artifact_dir, create_session=False) == artifact

    graph = onnx.load(artifact.artifact_dir / "model.onnx", load_external_data=False)
    _assert_exact_packed_initializer_parity(model, graph)
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


def test_module_onnx_accepts_grouped_scale_initializers(tmp_path):
    trits = torch.tensor(
        [[1, -1, 0, 1, -1, 0, 1, -1], [0, 1, 1, -1, 0, -1, 1, 0]],
        dtype=torch.int8,
    )
    plane = SimpleNamespace(
        trits=trits,
        scales=torch.tensor([[0.5, 0.25], [0.25, 0.125]], dtype=torch.float16),
        group_size=4,
    )
    model = AdditiveTernaryLinear((plane,)).eval()
    artifact = export_module_onnx(
        model, torch.randn(2, 8), tmp_path / "grouped-scales"
    )
    graph = onnx.load(
        artifact.artifact_dir / "model.onnx", load_external_data=False
    )
    scales = next(
        value
        for value in graph.graph.initializer
        if value.name == "_packed_weight.scales_0"
    )
    assert tuple(scales.dims) == (2, 2)
    replay = torch.randn(3, 8)
    torch.testing.assert_close(
        load_module_onnx(artifact.artifact_dir)(replay), model(replay),
        rtol=1e-4, atol=1e-5,
    )


def test_public_facade_executes_qat_ptq_and_refinement_artifacts_in_ort(tmp_path):
    torch.manual_seed(131)
    example = torch.randn(2, 8)

    qat_source = torch.nn.Linear(8, 2).eval()
    qat_shell = copy.deepcopy(qat_source)
    qat = prepare(
        qat_source,
        TernaryConfig.qat(target_modules=("Linear",)),
        inplace=True,
    )
    qat.model.eval()
    qat_hard = convert(qat)
    qat_bundle = export_onnx(
        qat_hard,
        tmp_path / "qat-onnx",
        example_inputs=example,
    )
    qat_graph = onnx.load(
        qat_bundle.artifact_dir / "model.onnx", load_external_data=False
    )
    _assert_exact_packed_initializer_parity(qat_hard.model, qat_graph)
    qat_runtime = load_onnx(qat_bundle.artifact_dir)
    assert qat_runtime.artifact.lineage.mode == "qat-hard"
    torch.testing.assert_close(
        qat_runtime(example), qat_hard.model(example), rtol=1e-4, atol=1e-5
    )
    qat_artifact = export(qat_hard, tmp_path / "qat-hard")
    reopened_bundle = export_onnx(
        qat_artifact,
        tmp_path / "reopened-qat-onnx",
        model=qat_shell,
        example_inputs=example,
    )
    reopened_graph = onnx.load(
        reopened_bundle.artifact_dir / "model.onnx", load_external_data=False
    )
    _assert_exact_packed_initializer_parity(qat_hard.model, reopened_graph)
    reopened_runtime = load_onnx(reopened_bundle.artifact_dir)
    assert reopened_runtime.artifact.lineage.artifact_id == qat_artifact.artifact_id
    assert reopened_runtime.artifact.lineage.mode == "qat-hard"
    torch.testing.assert_close(
        reopened_runtime(example), qat_hard.model(example), rtol=1e-4, atol=1e-5
    )

    teacher = torch.nn.Linear(8, 2).eval()
    prepared = prepare(
        teacher,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        [example],
        evidence_dir=tmp_path / "calibration",
    )
    ptq = convert(prepared, calibration, work_dir=tmp_path / "ptq")
    ptq_bundle = export_onnx(
        ptq,
        tmp_path / "ptq-onnx",
        model=teacher,
        example_inputs=example,
    )
    ptq_runtime = load_onnx(ptq_bundle.artifact_dir)
    assert ptq_runtime.artifact.lineage.mode == "ptq"
    ptq_model = load_quantized_module(teacher, ptq)
    torch.testing.assert_close(
        ptq_runtime(example), ptq_model(example), rtol=1e-4, atol=1e-5
    )

    refined = refine(
        ptq,
        teacher=teacher,
        training=[torch.randn(2, 8)],
        validation=[torch.randn(3, 8)],
        config=RefinementConfig.scale_only(max_steps=1),
        work_dir=tmp_path / "refinement",
    )
    refined_bundle = export_onnx(
        refined,
        tmp_path / "refined-onnx",
        model=teacher,
        example_inputs=example,
    )
    refined_runtime = load_onnx(refined_bundle.artifact_dir)
    assert refined_runtime.artifact.lineage.mode == "scale-only"
    assert refined_runtime.artifact.lineage.ancestry == refined.ancestry
    refined_model = refined.load_model(teacher)
    torch.testing.assert_close(
        refined_runtime(example), refined_model(example), rtol=1e-4, atol=1e-5
    )

    hard_pv = refine(
        ptq,
        teacher=teacher,
        training=[torch.randn(2, 8)],
        validation=[torch.randn(3, 8)],
        config=RefinementConfig.hard_pv(max_steps=1, pv_iterations=1),
        work_dir=tmp_path / "hard-pv",
    )
    hard_pv_bundle = export_onnx(
        hard_pv,
        tmp_path / "hard-pv-onnx",
        model=teacher,
        example_inputs=example,
    )
    hard_pv_runtime = load_onnx(hard_pv_bundle.artifact_dir)
    assert hard_pv_runtime.artifact.lineage.mode == "hard-pv"
    assert hard_pv_runtime.artifact.lineage.ancestry == hard_pv.ancestry
    torch.testing.assert_close(
        hard_pv_runtime(example),
        hard_pv.load_model(teacher)(example),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_qat_threshold_codes_survive_hard_export_and_onnx_reload(
    tmp_path, source_dtype
):
    midpoint = torch.tensor(1.0, dtype=torch.float32)
    below = torch.nextafter(midpoint, torch.tensor(0.0))
    above = torch.nextafter(midpoint, torch.tensor(2.0))
    latent = torch.tensor(
        [[below.item(), 3.0], [1.0, 3.0], [above.item(), 3.0]],
        dtype=source_dtype,
    )
    dense = torch.nn.Linear(2, 3, bias=False, dtype=source_dtype)
    with torch.no_grad():
        dense.weight.copy_(latent)
    prepared = prepare(
        dense,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear",),
            planes=1,
        ),
        inplace=True,
    )
    assert isinstance(prepared.model, TernaryLinear)
    projection = prepared.model.estimator.project(
        prepared.model.weight,
        context=ProjectionContext(step=0, training=False, role="weight"),
    )
    expected_trits = projection.planes[0].trits.detach().clone()
    assert expected_trits[:, 1].tolist() == [1, 1, 1]
    if source_dtype == torch.float32:
        assert expected_trits[:, 0].tolist() == [0, 0, 1]
    else:
        # BF16 cannot distinguish the adjacent FP32 values, so all three land
        # exactly on torch.round's ties-to-even zero code.
        assert expected_trits[:, 0].tolist() == [0, 0, 0]

    prepared.model.eval()
    source_input = torch.ones((2, 2), dtype=source_dtype)
    live_hard_output = prepared.model(source_input)
    hard = convert(prepared)
    torch.testing.assert_close(
        hard.model(source_input), live_hard_output, rtol=0, atol=0
    )
    expected_packed = AdditiveTernaryWeight(projection.planes)
    for field in ("packed_trits_0", "scales_0"):
        expected_bytes = (
            getattr(expected_packed, field).detach().cpu().numpy().tobytes()
        )
        actual_bytes = (
            getattr(hard.model.packed_weight, field)
            .detach()
            .cpu()
            .numpy()
            .tobytes()
        )
        assert actual_bytes == expected_bytes
    bundle = export_onnx(
        hard,
        tmp_path / f"qat-boundary-{str(source_dtype).split('.')[-1]}",
        example_inputs=torch.ones((1, 2), dtype=torch.float32),
    )
    graph = onnx.load(bundle.artifact_dir / "model.onnx", load_external_data=False)
    _assert_exact_packed_initializer_parity(hard.model, graph)
    runtime = load_onnx(bundle.artifact_dir)
    runtime_input = torch.ones((2, 2), dtype=torch.float32)
    expected = hard.model(runtime_input)
    torch.testing.assert_close(
        runtime(runtime_input), expected, rtol=0, atol=0
    )


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


def test_packed_linear_onnx_keeps_large_flattened_indices_exact(tmp_path):
    rows, columns = 29128, 576
    trits = torch.zeros((rows, columns), dtype=torch.int8)
    trits[-1, 67] = 1
    plane = SimpleNamespace(
        trits=trits,
        scales=torch.ones((rows, 1), dtype=torch.float16),
        group_size=columns,
    )
    model = AdditiveTernaryLinear((plane,)).eval()
    hidden = torch.zeros((1, 1, columns), dtype=torch.float32)
    hidden[0, 0, 67] = 1.0

    artifact = export_module_onnx(
        model,
        hidden,
        tmp_path / "large-index-bundle",
        input_names=("hidden",),
        output_names=("logits",),
    )

    expected = model(hidden)
    observed = load_module_onnx(artifact.artifact_dir)(hidden)
    assert expected[0, 0, -1] == 1.0
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
