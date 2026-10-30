"""Actual objective/module capture into immutable shared Rust owner windows."""
from dataclasses import FrozenInstanceError
import json
import pytest
import torch
from torch import nn
from tritium.torch import TernaryConfig, prepare, prepare_native_module
from test_native_module_source import _fit


class TiedModel(nn.Module):
    def __init__(self, columns=32):
        super().__init__()
        self.embed = nn.Embedding(3, columns)
        self.embed_alias = self.embed
        self.head = nn.Linear(columns, 3, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        first = self.embed(ids)
        second = self.embed_alias(ids.flip(0))
        hidden = torch.tanh(first.cumsum(0)) + 0.1 * torch.sin(second)
        return self.head(hidden), self.head(hidden[:1])


def _source(root, *, frozen=False, dtype=torch.float32, columns=32):
    torch.manual_seed(512)
    model = TiedModel(columns).to(dtype=dtype)
    if frozen:
        model.requires_grad_(False)
    prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", planes=2,
                       target_modules=("Linear", "Embedding")), inplace=True)
    work = root / "work"; work.mkdir()
    for name in ("tokenizer.json", "config.json"):
        (root / name).write_bytes(b"{}")
    fit = _fit()
    plan = fit["Fit"]["plan"]
    plan["layout"]["rows"] = 3
    plan["layout"]["cols"] = columns
    plan["limits"]["max_weights"] = 3 * columns
    plan["limits"]["max_metric_entries"] = columns * columns
    plan["limits"]["max_output_trits"] = 9 * columns
    return prepare_native_module(prepared, actions={"embed.weight": fit}, levels=["compact", "full"],
        limits=dict(max_source_bytes=4096, max_asset_bytes=32768, max_retained_blob_bytes=65536,
                    max_package_bytes=100_000), work_dir=work,
        asset_paths={name: root / name for name in ("tokenizer.json", "config.json")})


def _window():
    return dict(input_ids=torch.tensor([0, 1, 0]), labels=torch.tensor([-100, 1, 2]),
                attention_mask=torch.tensor([True, True, True]))


def _objective(model, window):
    first, second = model(window["input_ids"])
    return first[1:].square().mean() + 0.3 * second.square().mean()


def _capture(source, window=None, objective=_objective, **overrides):
    options = dict(objective_bytes=b"selected squared means", max_observation_bytes=4096,
                   max_context_bytes=4096, max_invocations=8, max_samples=64, max_graph_nodes=512)
    options.update(overrides)
    return source.capture_owner_window(_window() if window is None else window, objective, **options)


def _hooks_empty(model):
    return all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())


@pytest.mark.parametrize("frozen", [False, True])
def test_real_tied_capture_deduplicates_aliases_and_matches_owner_autograd(tmp_path, frozen):
    source = _source(tmp_path, frozen=frozen)
    calls = []
    def objective(model, window):
        calls.append(1)
        return _objective(model, window)
    capture = _capture(source, objective=objective)
    assert calls == [1] and capture.source_id == source.observation.source_id
    assert len(capture.owners) == 1 and capture.owners[0].use_count == 4
    assert capture.owners[0].observation_bytes == 1376
    assert capture.invocations == (
        ("embed.weight", ("embed", "embed_alias"), "gather", 3),
        ("embed.weight", ("embed", "embed_alias"), "gather", 3),
        ("embed.weight", ("head",), "matmul", 3),
        ("embed.weight", ("head",), "matmul", 1))
    actual = torch.stack([capture.row_score("embed.weight", row, max_workspace_bytes=256)
                          for row in range(3)])
    model = source.prepared.model
    assert model.embed.weight.requires_grad is (not frozen) and model.embed.weight.grad is None
    model.embed.weight.requires_grad_(True)
    dense = torch.autograd.grad(_objective(model, _window()), model.embed.weight)[0]
    torch.testing.assert_close(actual, dense.double(), rtol=5e-6, atol=5e-7)
    assert _hooks_empty(model) and list((tmp_path / "work").iterdir()) == []
    with torch.no_grad():
        model.embed.weight.zero_()
    torch.testing.assert_close(capture.row_score("embed.weight", 0, max_workspace_bytes=256), actual[0])
    with pytest.raises((AttributeError, TypeError)):
        capture.owners[0].rows = 99
    with pytest.raises(FrozenInstanceError):
        capture.owners = ()
    with pytest.raises(ValueError, match="absent"):
        capture.row_score("missing", 0, max_workspace_bytes=256)


@pytest.mark.parametrize("variant", ["functional", "regularizer", "view", "detach"])
def test_unknown_owner_reads_fail_instead_of_being_omitted(tmp_path, variant):
    source = _source(tmp_path)
    def objective(model, window):
        loss = _objective(model, window)
        if variant == "functional":
            return loss + torch.nn.functional.linear(torch.ones(1, 32), model.head.weight).sum()
        if variant == "regularizer":
            return loss + model.head.weight.square().sum()
        if variant == "view":
            return loss + model.head.weight[:1].sum()
        return loss + model.head.weight.detach().sum()
    with pytest.raises(ValueError, match="unsupported read"):
        _capture(source, objective=objective)
    assert _hooks_empty(source.prepared.model) and source.prepared.model.embed.weight.grad is None


@pytest.mark.parametrize("limit,value", [("max_observation_bytes", 1375), ("max_context_bytes", 60),
    ("max_invocations", 3), ("max_samples", 9), ("max_graph_nodes", 1),
    ("max_observation_bytes", True), ("max_samples", 0)])
def test_admission_failures_cleanup_hooks(tmp_path, limit, value):
    source = _source(tmp_path)
    with pytest.raises(ValueError):
        _capture(source, **{limit: value})
    assert _hooks_empty(source.prepared.model)


@pytest.mark.parametrize("variant", ["padding", "frequency", "sparse", "max_norm", "forward", "hook"])
def test_unsupported_consumer_semantics_do_not_execute_objective(tmp_path, variant):
    source = _source(tmp_path)
    model = source.prepared.model
    handle = None
    if variant == "padding": model.embed.padding_idx = 0
    elif variant == "frequency": model.embed.scale_grad_by_freq = True
    elif variant == "sparse": model.embed.sparse = True
    elif variant == "max_norm": model.embed.max_norm = 1.0
    elif variant == "forward": model.embed.forward = lambda ids: torch.zeros(ids.shape + (32,))
    else: handle = model.embed.register_forward_hook(lambda *_: None)
    count = []
    try:
        with pytest.raises((ValueError, AttributeError)):
            _capture(source, objective=lambda *_: count.append(1))
        assert count == []
    finally:
        if handle is not None: handle.remove()


@pytest.mark.parametrize("variant", ["tokens", "labels", "attention", "added", "training", "changed_state"])
def test_visible_context_or_source_changes_reject(tmp_path, variant):
    source = _source(tmp_path)
    def objective(model, window):
        loss = _objective(model, window)
        if variant in {"tokens", "labels", "attention"}:
            name = {"tokens":"input_ids", "labels":"labels", "attention":"attention_mask"}[variant]
            window[name].zero_()
        elif variant == "added": window["extra"] = torch.tensor(1)
        elif variant == "training": model.eval()
        else: model.register_buffer("extra", torch.tensor(1))
        return loss
    with pytest.raises(ValueError):
        _capture(source, objective=objective)
    assert _hooks_empty(source.prepared.model)


def test_missing_disconnected_and_inference_paths_fail(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="did not execute"):
        _capture(source, objective=lambda m,w: m.embed(w["input_ids"]).sum())
    def disconnected(model, window):
        first, _ = model(window["input_ids"])
        return first.sum()
    with pytest.raises(RuntimeError, match="not have been used"):
        _capture(source, objective=disconnected)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference"):
        _capture(source)
    with pytest.raises(ValueError, match="statistical"):
        _capture(source, window={**_window(), "calibration_mask": torch.ones(3, dtype=torch.bool)})
    assert _hooks_empty(source.prepared.model)


def test_custom_backward_and_recomputation_fail(tmp_path):
    source = _source(tmp_path)
    class Custom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value): return value.clone()
        @staticmethod
        def backward(ctx, gradient): return gradient
    with pytest.raises(ValueError, match="custom autograd"):
        _capture(source, objective=lambda m,w: Custom.apply(_objective(m,w)))
    from torch.utils.checkpoint import checkpoint
    with pytest.raises(ValueError, match="recomputed|no-grad"):
        _capture(source, objective=lambda m,w: checkpoint(lambda x: _objective(m, {"input_ids":x}),
                 w["input_ids"], use_reentrant=False))
    assert _hooks_empty(source.prepared.model)


@pytest.mark.parametrize("kind", ["scaled", "inplace", "detached"])
def test_changed_global_linear_forward_cannot_relabel_operator_adjoints(tmp_path, monkeypatch, kind):
    source = _source(tmp_path)
    original = nn.Linear.forward
    def modified(module, value):
        result = original(module, value)
        if kind == "scaled": return result * 2
        if kind == "inplace": return result.mul_(2)
        return result.detach()
    monkeypatch.setattr(nn.Linear, "forward", modified)
    with pytest.raises(ValueError, match="output"):
        _capture(source)
    assert _hooks_empty(source.prepared.model)


@pytest.mark.parametrize("bias,noncontiguous", [(False,False), (True,False), (True,True)])
def test_batched_linear_views_and_bias_preserve_actual_row_adjoint(tmp_path, bias, noncontiguous):
    from test_native_module_calibration import _source as linear_source
    source = linear_source(tmp_path)
    model = source.prepared.model
    if not bias:
        # Observe the changed complete source before requesting any capture.
        model.a.bias = None; model.b.bias = None
        from test_native_module_source import _prepared
        source = prepare_native_module(_prepared(model), **json.loads(source.options_json))
    inputs = torch.randn(2, 3, 32)
    if noncontiguous: inputs = inputs.transpose(0, 1)
    window = dict(input_ids=torch.zeros(inputs.shape[:-1], dtype=torch.int64), inputs=inputs)
    capture = _capture(source, window, objective=lambda m,w: m(w["inputs"]).square().mean())
    dense = torch.autograd.grad(model(inputs).square().mean(), model.a.weight)[0]
    torch.testing.assert_close(capture.row_score("a.weight",0,max_workspace_bytes=256),
                               dense[0].double(), rtol=5e-6, atol=5e-7)


def _hf_source(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from tritium.torch.native_source import _inventory
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=16, tie_word_embeddings=True))
    prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", planes=2,
                       target_modules=("Linear","Embedding")), inplace=True)
    actions = {}
    for name, _, tensor, _, selected, _ in _inventory(prepared):
        if not selected: continue
        fit = _fit(); plan = fit["Fit"]["plan"]
        rows, columns = tensor.shape
        plan["layout"].update(rows=rows, cols=columns)
        plan["limits"].update(max_weights=rows*columns, max_metric_entries=columns*columns,
                              max_output_trits=3*rows*columns)
        actions[name] = fit
    work=tmp_path/"work"; work.mkdir()
    for name in ("config.json","tokenizer.json"): (tmp_path/name).write_bytes(b"{}")
    source = prepare_native_module(prepared, actions=actions, levels=["compact","full"],
        limits=dict(max_source_bytes=1_000_000,max_asset_bytes=100_000,max_retained_blob_bytes=1_000_000,
                    max_package_bytes=2_000_000),work_dir=work,
        asset_paths={name:tmp_path/name for name in ("config.json","tokenizer.json")})
    return source


def test_actual_tiny_hf_language_owner_capture(tmp_path):
    from tritium.torch.native_source import _inventory
    source=_hf_source(tmp_path)
    prepared=source.prepared; model=prepared.model
    actions={owner for owner,_,_,_,selected,_ in _inventory(prepared) if selected}
    window=dict(input_ids=torch.tensor([[1,2,3,4]]), labels=torch.tensor([[1,2,3,4]]))
    objective=lambda m,w: m(**w, use_cache=False).loss
    capture = _capture(source, window, objective=objective, max_observation_bytes=1_000_000,
                       objective_bytes=b"tiny HF next-token CE mean",
                       max_invocations=32,max_samples=256,max_graph_nodes=4096)
    owners = [tensor for _,_,tensor,_,selected,_ in _inventory(prepared) if selected]
    expected = torch.autograd.grad(objective(model, window), owners)
    assert [owner.tensor_name for owner in capture.owners] == sorted(actions)
    for owner, dense in zip(capture.owners,expected):
        for row in (0, owner.rows-1):
            torch.testing.assert_close(capture.row_score(owner.tensor_name,row,max_workspace_bytes=1024),
                                       dense[row].double(),rtol=1e-5,atol=1e-7)
    assert _hooks_empty(model) and list((tmp_path/"work").iterdir()) == []


def test_nonfinite_context_and_actual_nonfinite_adjoints_reject(tmp_path):
    source = _source(tmp_path)
    window = {**_window(), "unused": torch.tensor([float("nan")])}
    with pytest.raises(ValueError, match="nonfinite.*context"):
        _capture(source, window)
    def objective(model, window):
        first, second = model(window["input_ids"])
        return ((first.sum() + second.sum()) * 0).sqrt()
    with pytest.raises(ValueError, match="Nonfinite"):
        _capture(source, objective=objective)
    assert _hooks_empty(source.prepared.model)


def test_new_consumer_hook_is_rejected_but_not_removed_as_foreign_work(tmp_path):
    source = _source(tmp_path)
    created = []
    def objective(model, window):
        loss = _objective(model, window)
        created.append(model.head.register_forward_hook(lambda *_: None))
        return loss
    try:
        with pytest.raises(ValueError, match="hooks changed"):
            _capture(source, objective=objective)
        assert set(source.prepared.model.head._forward_hooks) == {created[0].id}
    finally:
        for handle in created: handle.remove()
    assert _hooks_empty(source.prepared.model)


def test_ephemeral_capture_is_not_a_durable_calibrated_fit(tmp_path):
    from tritium.torch import convert_calibrated_native_module
    source = _source(tmp_path)
    captured = _capture(source)
    with pytest.raises(TypeError, match="actual native calibration"):
        convert_calibrated_native_module(captured, tmp_path/"output", "not-a-fit",
            partition="Full", damping=0.01, max_numeric_bytes=100_000, max_cache_package_bytes=100_000)
    assert not (tmp_path/"output").exists()


def test_nested_ordinary_invocations_keep_operator_order(tmp_path, monkeypatch):
    source = _source(tmp_path)
    model = source.prepared.model
    original = nn.Linear.forward
    saved = []
    def forward(module, value):
        result = original(module, value)
        if module is model.head and value.shape[0] > 1:
            saved.append(module(value[:1]))
        return result
    monkeypatch.setattr(nn.Linear, "forward", forward)
    def objective(model, window):
        saved.clear()
        result = _objective(model, window)
        return result + 0.1 * saved[0].square().mean()
    capture = _capture(source, objective=objective)
    assert [(role,samples) for _,_,role,samples in capture.invocations] == [
        ("gather",3), ("gather",3), ("matmul",3), ("matmul",1), ("matmul",1)]
    dense = torch.autograd.grad(objective(model,_window()),model.head.weight)[0]
    torch.testing.assert_close(capture.row_score("embed.weight",0,max_workspace_bytes=256),
                               dense[0].double(),rtol=5e-6,atol=5e-7)
    assert _hooks_empty(model)
