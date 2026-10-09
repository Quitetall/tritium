"""Real ORT gates for packed generic module ONNX bundles."""

import copy
import hashlib
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
from tritium.torch.module_onnx import (  # noqa: E402
    _capture_reference_terminal_outputs,
    _capture_terminal_intermediates,
    _first_decoder_attention_residual_name,
    _first_decoder_block_internal_names,
    _decoder_layer_residual_names,
    _session_options,
    _terminal_intermediate_names,
)
import tritium.torch.module_onnx as module_onnx  # noqa: E402
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


def test_terminal_intermediate_capture_replays_concat_shards_and_shared_input(
    tmp_path,
):
    from onnx import TensorProto, helper, numpy_helper

    x_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 4]
    )
    hidden_info = helper.make_tensor_value_info(
        "shared_hidden", TensorProto.FLOAT, [1, 1, 4]
    )
    shard0_info = helper.make_tensor_value_info(
        "shard0", TensorProto.FLOAT, [1, 1, 2]
    )
    shard1_info = helper.make_tensor_value_info(
        "shard1", TensorProto.FLOAT, [1, 1, 3]
    )
    logits_info = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 1, 5]
    )
    weight0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    weight1 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    weight2 = torch.ones((4, 6), dtype=torch.float32)
    residual_bias = torch.ones(4, dtype=torch.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["shared_hidden"]),
            helper.make_node(
                "Add", ["shared_hidden", "residual_bias"], ["residual_a"]
            ),
            helper.make_node("Identity", ["residual_a"], ["mlp_boundary"]),
            helper.make_node(
                "MatMul", ["residual_a", "weight2"], ["mlp_projection"]
            ),
            helper.make_node(
                "Add", ["mlp_boundary", "residual_bias"], ["residual_b"]
            ),
            helper.make_node("MatMul", ["shared_hidden", "weight0"], ["shard0"]),
            helper.make_node("MatMul", ["shared_hidden", "weight1"], ["shard1"]),
            helper.make_node("Concat", ["shard0", "shard1"], ["logits"], axis=2),
        ],
        "terminal-diagnostic",
        [x_info],
        [logits_info],
        initializer=[
            numpy_helper.from_array(weight0.numpy(), name="weight0"),
            numpy_helper.from_array(weight1.numpy(), name="weight1"),
            numpy_helper.from_array(weight2.numpy(), name="weight2"),
            numpy_helper.from_array(residual_bias.numpy(), name="residual_bias"),
        ],
    )
    residual_info = helper.make_tensor_value_info(
        "residual_a", TensorProto.FLOAT, [1, 1, 4]
    )
    residual_b_info = helper.make_tensor_value_info(
        "residual_b", TensorProto.FLOAT, [1, 1, 4]
    )
    mlp_boundary_info = helper.make_tensor_value_info(
        "mlp_boundary", TensorProto.FLOAT, [1, 1, 4]
    )
    mlp_projection_info = helper.make_tensor_value_info(
        "mlp_projection", TensorProto.FLOAT, [1, 1, 6]
    )
    graph.value_info.extend(
        [
            hidden_info,
            residual_info,
            residual_b_info,
            mlp_boundary_info,
            mlp_projection_info,
            shard0_info,
            shard1_info,
        ]
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        model_path,
    )
    assert _terminal_intermediate_names(graph, ["logits"]) == (
        "shard0",
        "shard1",
        "shared_hidden",
    )
    assert _decoder_layer_residual_names(graph, hidden_size=4, layer_count=1) == (
        "residual_b",
    )
    assert _first_decoder_attention_residual_name(
        graph, hidden_size=4, layer_count=1
    ) == "residual_a"
    assert _first_decoder_block_internal_names(
        graph, hidden_size=4, layer_count=1, intermediate_size=6
    ) == ("mlp_boundary", "mlp_projection")

    sample = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    captured = _capture_terminal_intermediates(
        tmp_path,
        ["input"],
        [sample],
        ["logits"],
        onnx,
        ort,
        hidden_size=4,
        layer_count=1,
        intermediate_size=6,
    )
    assert [(role, name) for role, name, _value in captured] == [
        ("terminal-output-replay", "logits"),
        ("terminal-intermediate", "shard0"),
        ("terminal-intermediate", "shard1"),
        ("terminal-intermediate", "shared_hidden"),
        ("terminal-layer-residual", "residual_b"),
        ("terminal-attention-residual", "residual_a"),
        ("terminal-first-block-internal", "mlp_boundary"),
        ("terminal-first-block-internal", "mlp_projection"),
    ]
    values = {name: value for _role, name, value in captured}
    assert torch.equal(
        torch.from_numpy(values["logits"]),
        torch.cat(
            [torch.from_numpy(values["shard0"]), torch.from_numpy(values["shard1"])],
            dim=-1,
        ),
    )
    assert torch.equal(torch.from_numpy(values["shared_hidden"]), sample)
    assert torch.equal(
        torch.from_numpy(values["residual_b"]), sample + 2 * residual_bias
    )
    assert torch.equal(
        torch.from_numpy(values["residual_a"]), sample + residual_bias
    )
    assert torch.equal(torch.from_numpy(values["mlp_boundary"]), sample + residual_bias)
    assert torch.equal(
        torch.from_numpy(values["mlp_projection"]),
        (sample + residual_bias) @ weight2,
    )
    assert torch.equal(
        torch.from_numpy(values["shard0"]), sample @ weight0
    )
    assert torch.equal(
        torch.from_numpy(values["shard1"]), sample @ weight1
    )
    assert not (tmp_path / ".terminal-diagnostic.onnx").exists()

    oversized = onnx.load(model_path)
    oversized.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 20_000_000
    onnx.save(oversized, model_path)
    assert (
        _capture_terminal_intermediates(
            tmp_path,
            ["input"],
            [sample],
            ["logits"],
            onnx,
            ort,
        )
        == ()
    )
    assert not (tmp_path / ".terminal-diagnostic.onnx").exists()
    assert [name for _role, name, _value in captured] == [
        "logits",
        "shard0",
        "shard1",
        "shared_hidden",
        "residual_b",
        "residual_a",
        "mlp_boundary",
        "mlp_projection",
    ]


def test_reference_terminal_capture_is_bounded_and_best_effort():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    logits = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)

    class TinyReference:
        config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=4,
            vocab_size=10,
        )

        def __call__(self, *args, **kwargs):
            assert kwargs == {"output_hidden_states": True, "use_cache": False}
            return SimpleNamespace(
                logits=logits, hidden_states=(hidden, hidden, hidden)
            )

    captured = _capture_reference_terminal_outputs(
        TinyReference(), ["input_ids"], [input_ids]
    )
    assert [(role, name) for role, name, _value in captured] == [
        ("reference-output-replay", "logits"),
        ("reference-hidden-state", "hidden_states[0]"),
        ("reference-hidden-state", "hidden_states[1]"),
        ("reference-terminal-hidden", "hidden_states[-1]"),
    ]
    assert torch.equal(captured[0][2], logits)
    assert all(torch.equal(item[2], hidden) for item in captured[1:])

    class ModernReference(TinyReference):
        def __call__(self, *args, **kwargs):
            assert kwargs == {"output_hidden_states": True, "use_cache": False}
            return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))

    modern_capture = _capture_reference_terminal_outputs(
        ModernReference(), ["input_ids"], [input_ids]
    )
    assert [(role, name) for role, name, _value in modern_capture] == [
        ("reference-output-replay", "logits"),
        ("reference-hidden-state", "hidden_states[0]"),
        ("reference-terminal-hidden", "hidden_states[-1]"),
    ]

    class OversizedReference(TinyReference):
        config = SimpleNamespace(
            num_hidden_layers=96,
            hidden_size=8192,
            vocab_size=10_000_000,
        )

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("oversized capture must not invoke the model")

    assert (
        _capture_reference_terminal_outputs(
            OversizedReference(), ["input_ids"], [input_ids]
        )
        == ()
    )


def test_reference_terminal_capture_includes_first_attention_residual():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    hidden = torch.ones((1, 3, 4), dtype=torch.float32)
    logits = torch.zeros((1, 3, 10), dtype=torch.float32)

    class TinyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.Identity()
            self.mlp = TinyMLP()

        def forward(self, value):
            return self.mlp(self.post_attention_layernorm(value + 1))

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Identity()
            self.up_proj = torch.nn.Identity()
            self.act_fn = torch.nn.SiLU()

        def forward(self, value):
            return self.act_fn(self.gate_proj(value)) * self.up_proj(value)

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyDecoderLayer()])

    class TinyHookedReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=4,
                intermediate_size=4,
                vocab_size=10,
            )
            self.model = TinyBackbone()

        def forward(self, _input_ids, *, output_hidden_states, use_cache):
            assert output_hidden_states is True
            assert use_cache is False
            self.model.layers[0](hidden)
            return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))

    captured = _capture_reference_terminal_outputs(
        TinyHookedReference(), ["input_ids"], [input_ids]
    )
    assert [(role, name) for role, name, _value in captured] == [
        ("reference-output-replay", "logits"),
        ("reference-attention-residual", "layers[0].attention_residual"),
        (
            "reference-mlp-input",
            "layers[0].post_attention_layernorm.output",
        ),
        (
            "reference-mlp-gate-projection",
            "layers[0].mlp.gate_proj.output",
        ),
        (
            "reference-mlp-up-projection",
            "layers[0].mlp.up_proj.output",
        ),
        (
            "reference-mlp-activation",
            "layers[0].mlp.act_fn.output",
        ),
        ("reference-mlp-output", "layers[0].mlp.output"),
        ("reference-hidden-state", "hidden_states[0]"),
        ("reference-terminal-hidden", "hidden_states[-1]"),
    ]
    torch.testing.assert_close(captured[1][2], hidden + 1, rtol=0, atol=0)
    torch.testing.assert_close(captured[2][2], hidden + 1, rtol=0, atol=0)
    torch.testing.assert_close(captured[3][2], hidden + 1, rtol=0, atol=0)
    torch.testing.assert_close(captured[4][2], hidden + 1, rtol=0, atol=0)
    torch.testing.assert_close(
        captured[5][2], torch.nn.functional.silu(hidden + 1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        captured[6][2], torch.nn.functional.silu(hidden + 1) * (hidden + 1),
        rtol=0,
        atol=0,
    )


def test_terminal_capture_failure_preserves_primary_parity_diagnostic(
    tmp_path, monkeypatch
):
    diagnostic_root = tmp_path / "diagnostics"
    monkeypatch.setenv("TRITIUM_ONNX_PARITY_FAILURE_DIR", str(diagnostic_root))
    monkeypatch.setenv("TRITIUM_ONNX_PARITY_CAPTURE_TERMINAL", "1")

    def fail_assert_close(*args, **kwargs):
        raise AssertionError("injected")

    def fail_terminal_capture(*args, **kwargs):
        raise RuntimeError("injected terminal capture failure")

    monkeypatch.setattr(torch.testing, "assert_close", fail_assert_close)
    monkeypatch.setattr(
        module_onnx, "_capture_terminal_intermediates", fail_terminal_capture
    )
    monkeypatch.setattr(
        module_onnx, "_capture_reference_terminal_outputs", fail_terminal_capture
    )
    model = _model()
    with pytest.raises(AssertionError, match="injected"):
        export_module_onnx(model, torch.randn(2, 8), tmp_path / "bundle")

    manifest = json.loads(
        (diagnostic_root / "bundle" / "diagnostic.json").read_text()
    )
    assert [item["role"] for item in manifest["replay_arrays"]] == [
        "input",
        "expected-output",
        "observed-output",
    ]


def test_terminal_intermediates_are_added_to_parity_diagnostic(tmp_path, monkeypatch):
    diagnostic_root = tmp_path / "diagnostics"
    monkeypatch.setenv("TRITIUM_ONNX_PARITY_FAILURE_DIR", str(diagnostic_root))
    monkeypatch.setenv("TRITIUM_ONNX_PARITY_CAPTURE_TERMINAL", "1")

    def fail_assert_close(*args, **kwargs):
        raise AssertionError("injected")

    terminal_value = torch.tensor([[[-0.25, 0.5]]]).numpy()
    monkeypatch.setattr(torch.testing, "assert_close", fail_assert_close)
    monkeypatch.setattr(
        module_onnx,
        "_capture_terminal_intermediates",
        lambda *_args, **_kwargs: (
            ("terminal-intermediate", "shard0", terminal_value),
        ),
    )
    with pytest.raises(AssertionError, match="injected"):
        export_module_onnx(_model(), torch.randn(2, 8), tmp_path / "bundle")

    retained = diagnostic_root / "bundle"
    manifest = json.loads((retained / "diagnostic.json").read_text())
    terminal_entry = next(
        item
        for item in manifest["replay_arrays"]
        if item["role"] == "terminal-intermediate"
    )
    assert terminal_entry["name"] == "shard0"
    assert (retained / terminal_entry["file"]).read_bytes() == terminal_value.tobytes()


@pytest.mark.parametrize("retain_diagnostics", [False, True])
def test_onnx_parity_failure_can_retain_opt_in_diagnostic_graph(
    tmp_path, monkeypatch, retain_diagnostics
):
    diagnostic_root = tmp_path / "diagnostics"
    if retain_diagnostics:
        monkeypatch.setenv("TRITIUM_ONNX_PARITY_FAILURE_DIR", str(diagnostic_root))
    else:
        monkeypatch.delenv("TRITIUM_ONNX_PARITY_FAILURE_DIR", raising=False)

    def fail_assert_close(*args, **kwargs):
        raise AssertionError("injected")

    monkeypatch.setattr(
        torch.testing, "assert_close", fail_assert_close
    )

    model = _model()
    replay = torch.randn(2, 8)
    with pytest.raises(AssertionError, match="injected"):
        export_module_onnx(model, replay, tmp_path / "bundle")

    assert not (tmp_path / "bundle").exists()
    if not retain_diagnostics:
        assert not diagnostic_root.exists()
        return
    retained = diagnostic_root / "bundle"
    graph = retained / "model.onnx"
    manifest = json.loads((retained / "diagnostic.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["checkpoint_digest"].startswith("sha256:")
    assert manifest["failure_type"] == "AssertionError"
    assert any(item["file"] == "model.onnx" for item in manifest["files"])
    for item in manifest["files"]:
        payload = (retained / item["file"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert "sha256:" + hashlib.sha256(payload).hexdigest() == item["sha256"]
    arrays = manifest["replay_arrays"]
    assert [item["role"] for item in arrays] == [
        "input", "expected-output", "observed-output"
    ]
    input_entry, expected_entry, observed_entry = arrays
    assert input_entry["shape"] == [2, 8]
    assert (retained / input_entry["file"]).read_bytes() == (
        replay.numpy().tobytes(order="C")
    )
    assert (retained / expected_entry["file"]).read_bytes() == (
        model(replay).detach().numpy().tobytes(order="C")
    )
    for item in arrays:
        payload = (retained / item["file"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert "sha256:" + hashlib.sha256(payload).hexdigest() == item["sha256"]
    assert observed_entry["shape"] == [2, 2]


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
