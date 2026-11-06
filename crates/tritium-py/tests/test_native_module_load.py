"""Actual package-reader/backend execution in fresh Torch and HF shells."""
import copy
import json
import gc
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from torch import nn
from tritium import NativeModelConversion
from tritium.torch import load_native_module
from test_native_module_source import StateModel, _prepared, _convert, _fit


def test_fresh_shell_executes_native_and_restores_all_preserved_aliases(tmp_path):
    source = _prepared()
    result = _convert(tmp_path, source)
    original = {name: value.clone() for name, value in source.model.state_dict().items()}
    shell = StateModel()
    untouched = {name: value.clone() for name, value in shell.state_dict().items()}
    loaded = load_native_module(shell, result, level="full", load_run_id="load-probe", inplace=False)
    assert loaded is not shell
    for name, tensor in shell.state_dict().items():
        assert torch.equal(tensor, untouched[name])
    assert loaded.a.packed_weight is loaded.b.packed_weight
    assert loaded.a.bias is loaded.b.bias
    assert loaded.mask is loaded.mask_alias
    for name in ("norm.bias", "norm.weight", "counter", "mask", "scalar", "empty"):
        assert torch.equal(getattr(loaded, name.split(".")[0]).__getattr__(name.split(".")[1])
                           if "." in name else getattr(loaded, name), original[name])
    handle = loaded.a.packed_weight.handle
    assert json.loads(handle.storage_json)["Additive"]["layout"]["law"]["precision"] == "F32"
    assert handle.payload_bytes < 32 * 4
    value = torch.randn(2, 32)
    dense = torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows, dtype=torch.int64)))
    with torch.no_grad():
        normalized = loaded.norm(value)
        expected = 2 * torch.nn.functional.linear(normalized, dense, loaded.a.bias)
        assert torch.allclose(loaded(value), expected, atol=2e-6, rtol=2e-6)
    log = [json.loads(line) for line in loaded._tritium_native_upload_log.splitlines()]
    assert any(event.get("schema") == "tritium.runtime.tensor_uploaded" for event in log)
    with pytest.raises(RuntimeError, match="native package"):
        loaded.state_dict()
    with pytest.raises(ValueError, match="CPU-only"):
        loaded.to(device="meta")
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())


def test_package_reopen_and_inplace_staging_have_exact_outputs(tmp_path):
    result = _convert(tmp_path, _prepared())
    reopened = NativeModelConversion.open(result.output_dir)
    assert reopened.package_id == result.package_id and reopened.source_id == result.source_id
    one = load_native_module(StateModel(), result, level="compact", load_run_id="one", inplace=False)
    shell = StateModel()
    two = load_native_module(shell, result.output_dir, level="compact", load_run_id="two", inplace=True)
    assert two is shell
    value = torch.randn(3, 32)
    with torch.no_grad(): assert torch.equal(one(value), two(value))


def test_root_dtype_conversion_retains_preserved_alias_topology(tmp_path):
    result = _convert(tmp_path, _prepared())
    loaded = load_native_module(StateModel(), result, level="full", load_run_id="dtype", inplace=False)
    packed = loaded.a.packed_weight.handle
    assert loaded.to(dtype=torch.float64) is loaded
    assert loaded.a.bias is loaded.b.bias
    assert loaded.mask is loaded.mask_alias
    assert loaded.a.bias.dtype == torch.float64
    assert loaded.a.packed_weight.handle is packed
    assert json.loads(packed.storage_json)["Additive"]["layout"]["law"]["precision"] == "F32"
    with torch.no_grad(): assert loaded(torch.randn(2, 32, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.parametrize("basis", ["Identity", {"Hadamard": {"block": 32}},
                                  {"SignedRht": {"block": 32, "seed": 13, "domain": 29}}])
@pytest.mark.parametrize("codec", ["B3", "D2"])
@pytest.mark.parametrize("precision", ["F16", "F32"])
@pytest.mark.parametrize("level", ["compact", "full"])
def test_native_semantic_profiles_match_gathered_logical_weights(tmp_path, basis, codec, precision, level):
    action = _fit()
    layout = action["Fit"]["plan"]["layout"]
    layout.update(basis=basis, codec=codec)
    layout["law"]["precision"] = precision
    result = _convert(tmp_path, _prepared(), actions={"a.weight": action})
    loaded = load_native_module(StateModel(), result, level=level, load_run_id="semantic", inplace=False)
    handle = loaded.a.packed_weight.handle
    observed = json.loads(handle.storage_json)["Additive"]["layout"]
    assert observed["basis"] == basis and observed["codec"] == codec
    assert observed["law"]["precision"] == precision
    dense = torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
    # A singleton-axis Torch view can be contiguous but noncanonical for DLPack.
    value = torch.randn(3, 32).unsqueeze(0).transpose(0, 1)
    with torch.no_grad():
        normalized = loaded.norm(value)
        expected = 2 * torch.nn.functional.linear(normalized, dense, loaded.a.bias)
        assert torch.allclose(loaded(value), expected, atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize("variant", ["untied", "shape", "dtype", "extra", "budget", "package"])
def test_failed_load_preserves_source_shell(tmp_path, variant):
    result = _convert(tmp_path, _prepared())
    shell = StateModel()
    kwargs = {}
    if variant == "untied": shell.b.weight = nn.Parameter(shell.a.weight.detach().clone())
    elif variant == "shape": shell.norm = nn.LayerNorm(16)
    elif variant == "dtype": shell.norm = shell.norm.to(torch.float16)
    elif variant == "extra": shell.register_buffer("new", torch.ones(2))
    elif variant == "package": kwargs["max_package_bytes"] = 1
    else: kwargs["max_payload_bytes"] = 1
    before = {name: value.clone() for name, value in shell.state_dict().items()}
    identities = {name: id(value) for name, value in shell.named_parameters(remove_duplicate=False)}
    with pytest.raises(ValueError):
        load_native_module(shell, result, level="full", load_run_id="failed", inplace=True, **kwargs)
    for name, value in shell.state_dict().items(): assert torch.equal(value, before[name])
    assert identities == {name: id(value) for name, value in shell.named_parameters(remove_duplicate=False)}


def test_shared_consumer_module_identity_and_graph_contract_are_preserved(tmp_path):
    source = StateModel()
    source.b = source.a
    result = _convert(tmp_path, _prepared(source))
    mismatch = StateModel()
    with pytest.raises(ValueError, match="graph contract"):
        load_native_module(mismatch, result, level="full", load_run_id="split-module", inplace=False)
    shell = StateModel()
    shell.b = shell.a
    loaded = load_native_module(shell, result, level="full", load_run_id="shared-module", inplace=False)
    assert loaded.a is loaded.b
    assert shell.a is shell.b and type(shell.a) is nn.Linear


def test_shared_upload_budget_releases_with_handle_and_errors_are_explicit(tmp_path):
    result = _convert(tmp_path, _prepared())
    probe = result.open_level("full", "measure")
    measured = probe.packed_tensor("a.weight").payload_bytes
    del probe
    session = result.open_level("full", "budget", max_payload_bytes=measured)
    handle = session.packed_tensor("a.weight")
    assert session.owned_payload_bytes == handle.payload_bytes
    with pytest.raises(ValueError): session.packed_tensor("a.weight")
    del handle
    gc.collect()
    assert session.owned_payload_bytes == 0
    handle = session.packed_tensor("a.weight")
    with pytest.raises(ValueError): handle.matmul(torch.ones(1, 31))
    with pytest.raises(ValueError): handle.matmul(torch.ones(1, 32, dtype=torch.float16))
    with pytest.raises(ValueError): handle.gather(torch.tensor([-1]))
    with pytest.raises(ValueError): handle.gather(torch.tensor([1]))
    session.finish()
    with pytest.raises(ValueError, match="sealed"): session.packed_tensor("a.weight")


def test_logical_output_bounds_precede_activation_normalization(tmp_path, monkeypatch):
    from tritium.torch import native_load
    result = _convert(tmp_path, _prepared())
    session = result.open_level("full", "small-output", max_output_bytes=16)
    weight = native_load.NativeTernaryWeight(session.packed_tensor("a.weight"))
    def no_copy(value): raise AssertionError("normalization must not run before the byte guard")
    monkeypatch.setattr(native_load, "_compact", no_copy)
    with pytest.raises(ValueError, match="byte limit"):
        weight.matmul(torch.ones(1, 32, dtype=torch.float16))
    with pytest.raises(ValueError, match="byte limit"):
        weight.gather(torch.zeros(1, dtype=torch.int32), torch.float32)
    with pytest.raises(ValueError, match="byte limit"):
        weight.handle.gather(torch.zeros(1, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="inference-only"):
        weight.matmul(torch.ones(1, 32, requires_grad=True))


@pytest.mark.parametrize("file", ["model.trit", "evidence.jsonl"])
def test_corrupt_package_reopen_fails_without_mutating_shell(tmp_path, file):
    result = _convert(tmp_path, _prepared())
    path = Path(result.output_dir) / file
    data = bytearray(path.read_bytes())
    data[len(data)//2] ^= 1
    path.write_bytes(data)
    shell = StateModel()
    before = {name: value.clone() for name, value in shell.state_dict().items()}
    with pytest.raises(ValueError):
        load_native_module(shell, result.output_dir, level="full", load_run_id="corrupt", inplace=True)
    for name, value in shell.state_dict().items(): assert torch.equal(value, before[name])


@pytest.mark.parametrize("variant", ["v1", "mixed-aliases", "nonstr-owner", "nonstr-dtype"])
def test_catalog_type_errors_and_preservation_only_version_fail_closed(tmp_path, variant):
    from tritium.torch.native_load import _catalog
    result = _convert(tmp_path, _prepared())
    data = json.loads(result.asset("tritium-torch-state.json"))
    if variant == "v1": data["adapter"] = "tritium.torch.native-source@1"
    elif variant == "mixed-aliases": data["state"][0]["aliases"].append([])
    elif variant == "nonstr-owner": data["state"][0]["owner"] = []
    else: data["state"][0]["dtype"] = {}
    class CatalogProbe:
        def asset(self, *args, **kwargs): return json.dumps(data).encode()
    with pytest.raises(ValueError): _catalog(CatalogProbe())


def test_actual_tied_llama_loaded_native_generates_after_package_reopen(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from tritium.torch import prepare, TernaryConfig
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=32, tie_word_embeddings=True)
    source = LlamaForCausalLM(config).eval()
    prepared = prepare(source, TernaryConfig.ptq(profile="compact-v1", planes=2,
                       target_modules=("Linear", "Embedding")), inplace=True)
    parameters = dict(source.named_parameters(remove_duplicate=False))
    actions = {}
    for entry in prepared.coverage.entries:
        if entry.disposition != "selected": continue
        rows, cols = parameters[entry.path].shape
        fit = _fit()
        plan = fit["Fit"]["plan"]
        plan["layout"].update(rows=rows, cols=cols)
        plan["limits"].update(max_weights=rows*cols, max_metric_entries=cols*cols, max_output_trits=3*rows*cols)
        actions[entry.path] = fit
    result = _convert(tmp_path, prepared, actions=actions,
                      limits=dict(max_source_bytes=1_000_000, max_asset_bytes=65536,
                                  max_retained_blob_bytes=1_000_000, max_package_bytes=2_000_000))
    first = load_native_module(LlamaForCausalLM(copy.deepcopy(config)), result,
                               level="full", load_run_id="llama-first", inplace=False)
    second = load_native_module(LlamaForCausalLM(copy.deepcopy(config)), result.output_dir,
                                level="full", load_run_id="llama-reopen", inplace=False)
    assert first.model.embed_tokens.packed_weight is first.lm_head.packed_weight
    reference = copy.deepcopy(source)
    reference_modules = dict(reference.named_modules(remove_duplicate=False))
    for path, module in first.named_modules(remove_duplicate=False):
        if not hasattr(module, "packed_weight"): continue
        handle = module.packed_weight.handle
        dense = torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
        with torch.no_grad(): reference_modules[path].weight.copy_(dense)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(first(tokens).logits, second(tokens).logits)
        assert torch.allclose(first(tokens).logits, reference(tokens).logits, atol=5e-6, rtol=5e-6)
        a = first.generate(tokens, max_new_tokens=3, do_sample=False)
        b = second.generate(tokens, max_new_tokens=3, do_sample=False)
        assert torch.equal(a, b)
        assert torch.equal(a, reference.generate(tokens, max_new_tokens=3, do_sample=False))
    import tritium
    child = '''import json,sys,torch
sys.path.insert(0, sys.argv[1])
from transformers import LlamaConfig,LlamaForCausalLM
from tritium.torch import load_native_module
model=load_native_module(LlamaForCausalLM(LlamaConfig(**json.loads(sys.argv[3]))),
    sys.argv[2],level="full",load_run_id="restart",inplace=False)
with torch.no_grad():
    actual=model.generate(torch.tensor([[1,2,3]]),max_new_tokens=3,do_sample=False)
assert actual.tolist()==json.loads(sys.argv[4]), actual
print("fresh-process native generation parity")
'''
    # Installed-wheel qualification must stay isolated. Source tests use the
    # current interpreter's dependency environment (including user-site deps).
    isolation = ["-I"] if os.environ.get("TRITIUM_TEST_INSTALLED_WHEEL") == "1" else []
    restarted = subprocess.run([sys.executable, *isolation, "-c", child,
                                str(Path(tritium.__file__).resolve().parent.parent),
                                result.output_dir, json.dumps(config.to_dict()), json.dumps(a.tolist())],
                               env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                               capture_output=True, text=True, timeout=30)
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert "fresh-process native generation parity" in restarted.stdout
    assert not any("embed_tokens.weight" in name or "lm_head.weight" in name
                   for name, _ in first.named_parameters())
