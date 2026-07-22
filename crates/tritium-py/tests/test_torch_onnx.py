import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import tritium.torch.onnx as onnx  # noqa: E402
from tritium.nn import AdditiveTernaryLinear, TernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    QatHardResult,
    QatHardArtifact,
    ModuleOnnxLineage,
    RefinementConfig,
    RefinementResult,
    QwenOnnxCausalLM,
    OnnxMtpOutput,
    TritiumError,
    export_onnx,
    export_module_onnx,
    load_onnx,
    load_module_onnx,
)
from tritium.torch.config import TernaryConfig  # noqa: E402
from tritium.torch.artifacts import QuantizationResult  # noqa: E402
from tritium.torch.conversion import PreparedModel  # noqa: E402
from tritium.torch.module_artifacts import ModuleQuantizationResult  # noqa: E402


def _write_onnx_bundle(root: Path, *, dynamic=False):
    root.mkdir()
    (root / "language.onnx").write_bytes(b"language")
    (root / "mtp.onnx").write_bytes(b"mtp")
    (root / "weights.bin").write_bytes(b"weights")
    value = {
        "schema": (
            "tritium-qwen35-onnx-bundle-v2"
            if dynamic
            else "tritium-qwen35-onnx-bundle-v1"
        ),
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
    if dynamic:
        value["sequence_mode"] = "dynamic-cache-v1"
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


def test_onnx_runtime_publishes_named_successful_operator_counts(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(path, *, device):
            return Runtime()

        @staticmethod
        def operator_counts():
            return SimpleNamespace(
                ternary_mpgemm=1,
                salt_v2_mpgemm=2,
                salt_v2_embedding=3,
                kv_attention=4,
                qwen_deltanet=5,
            )

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    assert model.operator_counts() == {
        "TritiumTernaryMpGemm": 1,
        "TritiumSaltV2MpGemm": 2,
        "TritiumSaltV2Embedding": 3,
        "TritiumKvAttention": 4,
        "TritiumQwenDeltaNet": 5,
    }


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


def test_export_onnx_routes_qat_hard_with_exact_lineage(monkeypatch, tmp_path):
    source = QatHardResult(
        model=torch.nn.Linear(2, 2).eval(),
        artifact_id="sha256:" + "11" * 32,
        source_checkpoint_digest="sha256:" + "22" * 32,
        hard_state_digest="sha256:" + "33" * 32,
        recipe_id="sha256:" + "44" * 32,
        config=TernaryConfig.qat(),
        source_coverage=SimpleNamespace(),
        weights=(),
    )
    observed = {}

    def export(*args, **kwargs):
        observed.update(args=args, kwargs=kwargs)
        return "qat-onnx"

    monkeypatch.setattr(onnx, "export_module_onnx", export, raising=False)
    example = torch.ones((1, 2))
    assert export_onnx(
        source,
        tmp_path / "out",
        example_inputs=example,
        input_names=("hidden",),
        output_names=("logits",),
        dynamic_axes={"hidden": {1: "sequence"}},
    ) == "qat-onnx"
    assert observed["args"] == (source.model, example, tmp_path / "out")
    assert observed["kwargs"]["lineage"].mode == "qat-hard"
    assert observed["kwargs"]["lineage"].artifact_id == source.artifact_id
    assert observed["kwargs"]["lineage"].recipe_id == source.recipe_id
    assert (
        observed["kwargs"]["lineage"].source_model_digest
        == source.source_checkpoint_digest
    )
    assert observed["kwargs"]["input_names"] == ("hidden",)
    assert observed["kwargs"]["output_names"] == ("logits",)
    assert observed["kwargs"]["dynamic_axes"] == {"hidden": {1: "sequence"}}

    with pytest.raises(ValueError, match="example_inputs"):
        export_onnx(source, tmp_path / "missing")


def test_export_onnx_routes_module_ptq_and_refinement_without_relabeling(
    monkeypatch, tmp_path
):
    ptq = ModuleQuantizationResult(
        artifact_dir=tmp_path / "ptq",
        artifact_id="sha256:" + "10" * 32,
        source_model_digest="sha256:" + "20" * 32,
        evidence_id="sha256:" + "30" * 32,
        algorithm_id="tritium.test@1",
        recipe_id="sha256:" + "40" * 32,
        config=TernaryConfig.ptq(profile="compact-v1"),
        coverage=SimpleNamespace(),
        weights=(),
    )
    refined = RefinementResult(
        artifact_dir=tmp_path / "refined",
        artifact_id="sha256:" + "50" * 32,
        parent_artifact_id=ptq.artifact_id,
        ancestry=(ptq.artifact_id,),
        source_model_digest=ptq.source_model_digest,
        teacher_digest="sha256:" + "60" * 32,
        training_digest="sha256:" + "70" * 32,
        training_batches=(),
        validation_digest="sha256:" + "80" * 32,
        validation_batches=(),
        config=RefinementConfig.scale_only(),
        conversion=SimpleNamespace(recipe_id="sha256:" + "90" * 32),
        packed=None,
        validation_loss_before=1.0,
        validation_loss_after=0.9,
        accepted_steps=1,
    )
    shell = torch.nn.Linear(2, 2).eval()
    converted = torch.nn.Linear(2, 2).eval()
    calls = []
    monkeypatch.setattr(
        onnx, "load_quantized_module", lambda model, source, inplace: converted,
        raising=False,
    )
    monkeypatch.setattr(
        RefinementResult, "load_model", lambda self, model, inplace: converted,
    )

    def export(model, inputs, output, **kwargs):
        calls.append((model, inputs, output, kwargs["lineage"]))
        return kwargs["lineage"].mode

    monkeypatch.setattr(onnx, "export_module_onnx", export, raising=False)
    example = torch.ones((1, 2))
    assert export_onnx(
        ptq, tmp_path / "ptq-onnx", model=shell, example_inputs=example,
    ) == "ptq"
    assert export_onnx(
        refined, tmp_path / "refined-onnx", model=shell, example_inputs=example,
    ) == "scale-only"
    assert calls[0][3].artifact_id == ptq.artifact_id
    assert calls[0][3].parent_artifact_id is None
    assert calls[1][3].artifact_id == refined.artifact_id
    assert calls[1][3].mode == "scale-only"
    assert calls[1][3].parent_artifact_id == ptq.artifact_id
    assert calls[1][3].ancestry == refined.ancestry


def test_load_onnx_routes_typed_module_bundle_through_public_facade(
    monkeypatch, tmp_path
):
    root = tmp_path / "module-onnx"
    root.mkdir()
    (root / "tritium-module-onnx.json").write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        onnx,
        "load_module_onnx",
        lambda path: observed.update(path=path) or "module-runtime",
        raising=False,
    )
    assert load_onnx(root) == "module-runtime"
    assert observed == {"path": root}
    with pytest.raises(TritiumError) as caught:
        load_onnx(root, device="cuda:0")
    assert caught.value.code == "onnx_device_unavailable"


def test_onnx_facade_rejects_trainable_and_checkpoint_state_with_stable_code(tmp_path):
    inputs = [torch.nn.Linear(2, 2), {"optimizer": {"state": {}}}]
    for source in inputs:
        with pytest.raises(TritiumError) as caught:
            export_onnx(source, tmp_path / "out")
        assert caught.value.code == "trainable_onnx_requires_v1_3"

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"state")
    with pytest.raises(TritiumError) as caught:
        load_onnx(checkpoint)
    assert caught.value.code == "trainable_onnx_requires_v1_3"


def test_module_export_rejects_latent_master_before_dependencies(tmp_path):
    plane = SimpleNamespace(
        trits=torch.ones((2, 2), dtype=torch.int8),
        scales=torch.ones((2, 1), dtype=torch.float16),
        group_size=2,
    )

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hard = AdditiveTernaryLinear((plane,)).eval()
            self.latent = TernaryLinear(2, 2).eval()

        def forward(self, value):
            return self.hard(value)

    with pytest.raises(TritiumError) as caught:
        export_module_onnx(Mixed().eval(), torch.ones((1, 2)), tmp_path / "out")
    assert caught.value.code == "trainable_onnx_requires_v1_3"


def test_module_onnx_lineage_rejects_forged_parentage_before_export(tmp_path):
    invalid = ModuleOnnxLineage(
        mode="scale-only",
        artifact_id="sha256:" + "11" * 32,
        recipe_id="sha256:" + "22" * 32,
        source_model_digest="sha256:" + "33" * 32,
        parent_artifact_id="sha256:" + "44" * 32,
        ancestry=("sha256:" + "55" * 32,),
    )
    with pytest.raises(ValueError, match="immediate parent"):
        export_module_onnx(
            torch.nn.Linear(2, 2).eval(),
            torch.ones((1, 2)),
            tmp_path / "forged",
            lineage=invalid,
        )


def test_module_onnx_loader_rejects_forged_lineage_before_ort(tmp_path):
    root = tmp_path / "module-onnx"
    root.mkdir()
    graph = root / "model.onnx"
    graph.write_bytes(b"not parsed before lineage admission")
    digest = lambda payload: "sha256:" + hashlib.sha256(payload).hexdigest()
    value = {
        "schema_version": 2,
        "artifact_kind": "tritium.packed-module-onnx-v2",
        "checkpoint_digest": "sha256:" + "11" * 32,
        "opset": 18,
        "input_names": ["input"],
        "output_names": ["output"],
        "packed_modules": [{
            "path": "linear",
            "storage_path": "linear._packed_weight",
            "rows": 1,
            "columns": 1,
            "planes": 1,
            "packed_initializers": ["packed"],
            "scale_initializers": ["scale"],
        }],
        "files": [{
            "file": "model.onnx",
            "sha256": digest(graph.read_bytes()),
            "bytes": graph.stat().st_size,
        }],
        "conversion": {
            "mode": "scale-only",
            "artifact_id": "sha256:" + "22" * 32,
            "recipe_id": "sha256:" + "33" * 32,
            "source_model_digest": "sha256:" + "11" * 32,
            "parent_artifact_id": "sha256:" + "44" * 32,
            "ancestry": ["sha256:" + "55" * 32],
        },
    }
    canonical = lambda item: json.dumps(
        item, sort_keys=True, separators=(",", ":")
    ).encode()
    value["artifact_id"] = digest(canonical(value))
    (root / "tritium-module-onnx.json").write_bytes(canonical(value))
    with pytest.raises(ValueError, match="immediate parent"):
        load_module_onnx(root, create_session=False)


def test_export_onnx_reopens_qat_hard_artifact_into_explicit_shell(
    monkeypatch, tmp_path
):
    source = QatHardArtifact(
        artifact_dir=tmp_path / "qat",
        artifact_id="sha256:" + "11" * 32,
        conversion_artifact_id="sha256:" + "22" * 32,
        source_checkpoint_digest="sha256:" + "33" * 32,
        hard_state_digest="sha256:" + "44" * 32,
        recipe_id="sha256:" + "55" * 32,
        config=TernaryConfig.qat(),
        source_coverage=SimpleNamespace(),
        weights=(),
        state_digest="sha256:" + "66" * 32,
        state_bytes=1,
        state_tensors=1,
        state_ledger=(),
    )
    shell = torch.nn.Linear(2, 2).eval()
    loaded = torch.nn.Linear(2, 2).eval()
    observed = {}
    monkeypatch.setattr(
        onnx,
        "load_qat_hard",
        lambda path, model, inplace: (
            observed.update(load=(path, model, inplace)) or loaded
        ),
        raising=False,
    )
    monkeypatch.setattr(
        onnx,
        "export_module_onnx",
        lambda model, inputs, output, **kwargs: observed.update(
            export=(model, inputs, output, kwargs["lineage"])
        ) or "artifact-onnx",
        raising=False,
    )
    example = torch.ones((1, 2))
    assert export_onnx(
        source, tmp_path / "out", model=shell, example_inputs=example,
    ) == "artifact-onnx"
    assert observed["load"] == (source.artifact_dir, shell, False)
    assert observed["export"][3].artifact_id == source.artifact_id
    assert observed["export"][3].mode == "qat-hard"


def test_export_onnx_forwards_dynamic_ptq_bundle(monkeypatch, tmp_path):
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
    assert export_onnx(source, tmp_path / "out") == "receipt"
    assert observed["args"] == (str(source.artifact_dir), str(tmp_path / "out"))
    assert observed["kwargs"]["profile"] == "compact-v1"
    assert observed["kwargs"]["tokens"] == 1
    assert observed["kwargs"]["past_tokens"] == 0

    with pytest.raises(ValueError, match="dynamic ONNX export"):
        export_onnx(source, tmp_path / "out", tokens=2)


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


def test_dynamic_model_greedy_generation_reuses_cache_and_stops_on_eos(
    monkeypatch, tmp_path
):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root, dynamic=True)
    calls = []

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(*args, **kwargs):
            return Runtime()

        def forward_language(self, token_ids, states):
            calls.append((token_ids, states))
            call = len(calls)
            next_token = {1: 2, 2: 1, 3: 3}[call]
            cache_rows = len(token_ids) if call == 1 else len(states[0]) + 1
            return SimpleNamespace(
                logits_shape=[len(token_ids), 4],
                logits=(
                    [0.0, 0.0, 0.0, 0.0] * (len(token_ids) - 1)
                    + [5.0 if index == next_token else 0.0 for index in range(4)]
                ),
                state_names=["present_k.0"],
                state_shapes=[[cache_rows, 1, 1]],
                states=[[float(index) for index in range(1, cache_rows + 1)]],
            )

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    generated = model.generate(torch.tensor([[4, 5]]), max_new_tokens=8, eos_token_id=3)
    assert generated.tolist() == [[4, 5, 2, 1, 3]]
    assert calls == [
        ([4, 5], None),
        ([2], [[1.0, 2.0]]),
        ([1], [[1.0, 2.0, 3.0]]),
    ]


def test_dynamic_generation_zero_budget_and_rejects_sampling(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root, dynamic=True)

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(*args, **kwargs):
            return Runtime()

        def forward_language(self, *args, **kwargs):
            pytest.fail("zero generation budget must not execute")

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    assert model.generate(torch.tensor([[7]]), max_new_tokens=0).tolist() == [[7]]
    with pytest.raises(TritiumError) as caught:
        model.generate(torch.tensor([[7]]), do_sample=True)
    assert caught.value.code == "onnx_sampling_unavailable"


def test_manifest_schema_and_sequence_mode_are_cross_bound(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    value = _write_onnx_bundle(root)
    value["sequence_mode"] = "dynamic-cache-v1"
    (root / "tritium-onnx-manifest.json").write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        onnx._tritium,
        "QwenOnnxModel",
        SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("must not load")),
        raising=False,
    )
    with pytest.raises(ValueError, match="schema v1"):
        load_onnx(root)

    value["schema"] = "tritium-qwen35-onnx-bundle-v2"
    value.pop("sequence_mode")
    (root / "tritium-onnx-manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="schema v2"):
        load_onnx(root)


def test_loaded_model_executes_authenticated_mtp_drafter(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    observed = {}

    class Runtime:
        device = "cpu"

        @staticmethod
        def load(*args, **kwargs):
            return Runtime()

        def forward_mtp(self, token_ids, target_hidden, states):
            observed.update(
                token_ids=token_ids,
                target_hidden=target_hidden,
                states=states,
            )
            return SimpleNamespace(
                logits_shape=[2, 3],
                logits=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                final_hidden_shape=[2, 2],
                final_hidden=[7.0, 8.0, 9.0, 10.0],
                state_names=["present_k.0", "present_v.0"],
                state_shapes=[[3, 1, 2], [3, 1, 2]],
                states=[list(range(6)), list(range(6, 12))],
            )

    monkeypatch.setattr(onnx._tritium, "QwenOnnxModel", Runtime, raising=False)
    model = load_onnx(root)
    result = model.draft(
        torch.tensor([[4, 5]], dtype=torch.int64),
        torch.tensor([[[0.5, 1.5], [2.5, 3.5]]], dtype=torch.float32),
        past_key_values=(
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ),
    )

    assert isinstance(result, OnnxMtpOutput)
    assert result.logits.shape == (1, 2, 3)
    assert result.final_hidden.shape == (1, 2, 2)
    assert tuple(state.shape for state in result.past_key_values) == (
        torch.Size([3, 1, 2]),
        torch.Size([3, 1, 2]),
    )
    assert result.state_names == ("present_k.0", "present_v.0")
    assert observed == {
        "token_ids": [4, 5],
        "target_hidden": [0.5, 1.5, 2.5, 3.5],
        "states": [[1.0, 2.0], [3.0, 4.0]],
    }


def test_loaded_model_rejects_mtp_target_hidden_token_drift(monkeypatch, tmp_path):
    root = tmp_path / "onnx"
    _write_onnx_bundle(root)
    monkeypatch.setattr(
        onnx._tritium,
        "QwenOnnxModel",
        SimpleNamespace(load=lambda *args, **kwargs: SimpleNamespace(device="cpu")),
        raising=False,
    )
    model = load_onnx(root)
    with pytest.raises(ValueError, match="target_hidden"):
        model.draft(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.zeros((1, 1, 4), dtype=torch.float32),
        )
