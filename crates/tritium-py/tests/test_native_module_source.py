"""Actual Torch source state passes through the shared native engine, not a mock."""
from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from safetensors.torch import load

from tritium.torch import TernaryConfig, convert_native_module, prepare


class StateModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.a = nn.Linear(32, 1, dtype=dtype)
        self.b = nn.Linear(32, 1, dtype=dtype)
        self.b.weight = self.a.weight
        self.b.bias = self.a.bias
        self.norm = nn.LayerNorm(32, dtype=dtype)
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
        self.register_buffer("mask", torch.tensor([True, False]))
        self.register_buffer("mask_alias", self.mask)
        self.register_buffer("scalar", torch.tensor(-0.0))
        self.register_buffer("empty", torch.empty(0, dtype=torch.float64))
        self.register_buffer("not_persistent", torch.tensor(99), persistent=False)

    def forward(self, value):
        return self.a(self.norm(value)) + self.b(self.norm(value))


def _prepared(model=None):
    return prepare(StateModel() if model is None else model,
                   TernaryConfig.ptq(profile="compact-v1", planes=2, target_modules=("Linear",)),
                   inplace=True)


def _fit():
    return {"Fit": {"metric": "Identity", "plan": dict(
        layout=dict(rows=1, cols=32, tile=256, group=32, max_planes=2,
                    allocation="Uniform", codec="B3", basis="Identity", transport="Raw",
                    law=dict(anchor="Group", relation="Free", precision="F32")),
        profile={"JointIndependentGroups": dict(max_iterations=4, ridge=1e-8, em_restarts=1,
                  ridge_condition_limit=1e6, softened_relay=False, modulated_relay=False)},
        levels=[dict(name="compact", planes=1), dict(name="full", planes=2)],
        limits=dict(max_weights=32, max_metric_entries=1024, max_output_trits=96))}}


def _convert(root, prepared, **overrides):
    work = root / "work"
    work.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        (root / name).write_bytes(b"{}")
    options = dict(actions={"a.weight": _fit()}, levels=["compact", "full"],
                   limits=dict(max_source_bytes=4096, max_asset_bytes=32768,
                               max_retained_blob_bytes=65536, max_package_bytes=100_000),
                   work_dir=work,
                   asset_paths={name: root / name for name in ("config.json", "tokenizer.json")})
    if "asset_paths" in overrides:
        options["asset_paths"].update(overrides.pop("asset_paths"))
    options.update(overrides)
    return convert_native_module(prepared, root / "output", "torch-state-probe", **options)


def _raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("dtype,precision", [(torch.float16, "F16"),
                                             (torch.bfloat16, "Bf16"), (torch.float32, "F32")])
def test_actual_torch_state_is_complete_and_aliases_are_bound(tmp_path, dtype, precision):
    prepared = _prepared(StateModel(dtype))
    model = prepared.model
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    result = _convert(tmp_path, prepared)
    result.verify()
    assert list((tmp_path / "work").iterdir()) == []
    for name in before:
        assert torch.equal(model.state_dict()[name], before[name])
    assert model.a.weight is model.b.weight and model.a.bias is model.b.bias
    catalog = json.loads(result.asset("tritium-torch-state.json"))
    owners = {entry["owner"]: entry for entry in catalog["state"]}
    assert owners["a.weight"]["aliases"] == ["a.weight", "b.weight"]
    assert owners["a.bias"]["aliases"] == ["a.bias", "b.bias"]
    assert owners["mask"]["aliases"] == ["mask", "mask_alias"]
    assert {alias for entry in owners.values() for alias in entry["aliases"]} == set(before)
    assert "not_persistent" not in owners
    for name in ("a.bias", "norm.bias", "norm.weight"):
        assert result.preserved_tensor("compact", name) == (precision, list(before[name].shape), _raw(before[name]))
    with pytest.raises(ValueError, match="not dense-preserved"):
        result.preserved_tensor("compact", "a.weight")
    preserved = load(result.asset("tritium-torch-preserved.safetensors"))
    assert set(preserved) == {"counter", "mask", "scalar", "empty"}
    for name, value in preserved.items():
        assert value.dtype == before[name].dtype and value.shape == before[name].shape
        assert torch.equal(value, before[name])
    assert preserved["scalar"].view(torch.int32).item() == -2147483648
    for maximum in (0, 1):
        with pytest.raises(ValueError):
            result.asset("tritium-torch-state.json", max_bytes=maximum)


@pytest.mark.parametrize("variant", ["untie", "overlap", "unsupported", "meta", "actions", "plan", "budget", "reserved", "custom", "omitted", "negative_view", "cycle"])
def test_invalid_source_fails_before_copying(tmp_path, monkeypatch, variant):
    prepared = _prepared()
    options = {}
    if variant == "untie": prepared.model.b.weight = nn.Parameter(prepared.model.a.weight.detach().clone())
    elif variant == "overlap": prepared.model.register_buffer("slice", prepared.model.mask[:1])
    elif variant == "unsupported": prepared.model.register_buffer("complex", torch.ones(2, dtype=torch.complex128))
    elif variant == "meta": prepared.model.register_buffer("unmaterialized", torch.empty(2, device="meta"))
    elif variant == "actions": options["actions"] = {}
    elif variant == "plan":
        fit = _fit()
        fit["Fit"]["plan"]["layout"]["group"] = 999
        options["actions"] = {"a.weight": fit}
    elif variant == "budget":
        options["limits"] = dict(max_source_bytes=1, max_asset_bytes=32768,
                                max_retained_blob_bytes=65536, max_package_bytes=100_000)
    elif variant == "reserved": options["asset_paths"] = {"tritium-torch-state.json": "missing"}
    elif variant == "custom":
        class CustomState(StateModel):
            def get_extra_state(self): return {"arbitrary": "object"}
            def set_extra_state(self, value): pass
        prepared = _prepared(CustomState())
    elif variant == "omitted":
        def omit(module, state, prefix, metadata): state.pop(prefix + "counter")
        prepared.model._register_state_dict_hook(omit)
    elif variant == "negative_view":
        prepared.model.register_buffer("negative_scalar", torch._neg_view(prepared.model.scalar))
    else: prepared.model.add_module("cycle", prepared.model)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid source reached safetensors payload copying")
    monkeypatch.setattr("safetensors.torch.save_file", forbidden)
    with pytest.raises((ValueError, TypeError)):
        _convert(tmp_path, prepared, **options)
    assert not (tmp_path / "output").exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_source_mutation_during_capture_leaves_no_artifact_or_scratch(tmp_path, monkeypatch):
    import safetensors.torch
    prepared = _prepared()
    original = safetensors.torch.save_file
    def changed(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            prepared.model.norm.bias.add_(1)
    monkeypatch.setattr(safetensors.torch, "save_file", changed)
    with pytest.raises(ValueError, match="state changed during source capture"):
        _convert(tmp_path, prepared)
    assert not (tmp_path / "output").exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_caller_asset_and_output_no_clobber_are_preserved(tmp_path):
    prepared = _prepared()
    asset = tmp_path / "user.json"
    asset.write_bytes(b'{"actual":"source asset"}')
    result = _convert(tmp_path, prepared, asset_paths={"user.json": asset})
    assert result.asset("user.json") == asset.read_bytes()
    before = (tmp_path / "output/model.trit").read_bytes()
    with pytest.raises(FileExistsError):
        _convert(tmp_path, prepared)
    assert (tmp_path / "output/model.trit").read_bytes() == before
    assert list((tmp_path / "work").iterdir()) == []


def test_requested_metric_never_falls_back_and_cleans_source(tmp_path):
    fit = _fit()
    fit["Fit"]["metric"] = {"InputGramF64": {"content": [0] * 32}}
    with pytest.raises(ValueError, match="raw Gram input inventory"):
        _convert(tmp_path, _prepared(), actions={"a.weight": fit})
    assert not (tmp_path / "output").exists()
    assert list((tmp_path / "work").iterdir()) == []


def test_cleanup_error_after_publication_is_an_explicit_warning(tmp_path, monkeypatch):
    import tempfile
    original = tempfile.TemporaryDirectory.cleanup
    def failed_cleanup(self):
        original(self)
        raise OSError("injected source cleanup fault")
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", failed_cleanup)
    result = _convert(tmp_path, _prepared())
    result.verify()
    assert (tmp_path / "output/model.trit").is_file()
    assert result.publication_warnings == ["SourceCleanup: injected source cleanup fault"]


def test_actual_tied_huggingface_state_packages_all_owners(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(123)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, tie_word_embeddings=True)).eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = model(tokens).logits.clone()
    prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", planes=2,
                       target_modules=("Linear", "Embedding")), inplace=True)
    params = dict(model.named_parameters(remove_duplicate=False))
    actions = {}
    for entry in prepared.coverage.entries:
        if entry.disposition != "selected": continue
        rows, columns = params[entry.path].shape
        action = _fit()
        plan = action["Fit"]["plan"]
        plan["layout"].update(rows=rows, cols=columns)
        plan["limits"].update(max_weights=rows * columns, max_metric_entries=columns ** 2,
                               max_output_trits=3 * rows * columns)
        actions[entry.path] = action
    limits = dict(max_source_bytes=1_000_000, max_asset_bytes=65536,
                  max_retained_blob_bytes=1_000_000, max_package_bytes=2_000_000)
    result = _convert(tmp_path, prepared, actions=actions, limits=limits)
    result.verify()
    catalog = json.loads(result.asset("tritium-torch-state.json"))
    observed = {alias for entry in catalog["state"] for alias in entry["aliases"]}
    assert observed == set(model.state_dict())
    assert sum(entry["disposition"] == "fitted" for entry in catalog["state"]) == len(actions)
    embedding = next(entry for entry in catalog["state"] if "lm_head.weight" in entry["aliases"])
    assert embedding["aliases"] == ["lm_head.weight", "model.embed_tokens.weight"]
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with torch.no_grad():
        assert torch.equal(before, model(tokens).logits)
