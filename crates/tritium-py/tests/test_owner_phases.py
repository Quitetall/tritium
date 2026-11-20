"""Public phases exercise real shared-owner capture, native fitting and load."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from tritium import NativeModelConversion
from tritium.torch import (
    NativeModulePreparation, NativeOwnerCalibration, OwnerCalibrationRecipe,
    TernaryConfig, calibrate, convert, convert_owner_native_module, inspect, load, prepare,
)
from test_owner_capture import TiedModel, _source, _window, _objective, _hf_source
from test_owner_cache import _limits, _open
from test_owner_conversion import _options


def _recipe(**changes):
    fit = _options()
    weights = fit.pop("window_weights")
    options = dict(
        objective_bytes=b"selected squared means", window_weights=weights,
        capture_options=dict(max_observation_bytes=4096, max_context_bytes=4096,
            max_invocations=8, max_samples=64, max_graph_nodes=512,
            cache_limits=_limits(), max_total_cache_bytes=8192),
        fit_options=fit,
    )
    options.update(changes)
    return OwnerCalibrationRecipe(**options)


def _public_source(root, **options):
    expert = _source(root, **options)
    source = prepare(expert.prepared.model, expert.prepared.config, inplace=False,
                     native_options=json.loads(expert.options_json))
    assert isinstance(source, NativeModulePreparation)
    assert source.observation.source_id == expert.observation.source_id
    assert source.prepared.model is not expert.prepared.model
    return source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_phases_fit_real_tied_owner_and_cold_load(tmp_path, dtype):
    source = _public_source(tmp_path, dtype=dtype)
    before = source.prepared.model.embed.weight.detach().clone()
    assert inspect(source) == source.prepared.coverage
    recipe = _recipe()
    calls = []
    def objective(model, window):
        calls.append(1)
        return _objective(model, window)
    calibration = calibrate(source, [_window()], objective=objective, owner_recipe=recipe)
    assert isinstance(calibration, NativeOwnerCalibration) and calls == [1]
    first = convert(source, calibration, output_dir=tmp_path/"first", run_id="public-owner")
    assert isinstance(first, NativeModelConversion)
    assert load(first) is first
    stored = calibration.captures[0].save(tmp_path/"cache", max_package_bytes=65536)
    reopened = NativeOwnerCalibration(source, (_open(source, stored),), recipe)
    second = convert(source, reopened, output_dir=tmp_path/"second", run_id="public-owner")
    expert = convert_owner_native_module(source, reopened.captures, tmp_path/"expert", "public-owner",
        window_weights=recipe.window_weights, **recipe.fit_options)
    for name in ("model.trit", "evidence.jsonl"):
        assert (tmp_path/"first"/name).read_bytes() == (tmp_path/"second"/name).read_bytes()
        assert (tmp_path/"first"/name).read_bytes() == (tmp_path/"expert"/name).read_bytes()
    assert first.package_id == second.package_id
    assert first.package_id == expert.package_id
    loaded = load(first, model=TiedModel().to(dtype=dtype), level="full",
                  load_run_id="public-load", inplace=False)
    cold = load(second.output_dir, model=TiedModel().to(dtype=dtype), level="full",
                load_run_id="cold-load", inplace=False)
    assert loaded.embed is loaded.embed_alias
    assert loaded.embed.packed_weight is loaded.head.packed_weight
    assert not list(loaded.parameters())
    for actual, expected in zip(loaded(_window()["input_ids"]), cold(_window()["input_ids"])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(source.prepared.model.embed.weight, before)
    assert source.prepared.model.embed.weight.grad is None
    assert list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("variant", ["shell", "level", "run", "inplace", "device", "profile"])
def test_native_public_load_requires_explicit_execution_choices(tmp_path, variant):
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    result = convert(source, calibration, output_dir=tmp_path/"output", run_id="public-load-reject")
    options = dict(model=TiedModel(), level="full", load_run_id="rejected", inplace=False)
    if variant in {"shell", "level", "run", "inplace"}:
        del options[{"shell":"model", "level":"level", "run":"load_run_id", "inplace":"inplace"}[variant]]
    else: options[variant] = "cuda" if variant == "device" else "near-lossless-v1"
    with pytest.raises((TypeError, ValueError)):
        load(result, **options)
    assert not list(result.publication_warnings)


def test_aggregate_calibration_limit_stops_before_next_objective(tmp_path):
    source = _public_source(tmp_path)
    fit = _recipe().fit_options
    fit["max_total_cache_bytes"] = 8192
    recipe = _recipe(window_weights=[1., 1.], fit_options=fit)
    calls = []
    def objective(model, window):
        calls.append(1)
        return _objective(model, window)
    with pytest.raises(ValueError, match="capacity exceeds aggregate"):
        calibrate(source, [_window(), _window()], objective=objective, owner_recipe=recipe)
    assert calls == [1]
    assert source.prepared.model.embed.weight.grad is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in source.prepared.model.modules())
    assert list((tmp_path/"work").iterdir()) == []


def test_recipe_freezes_options_and_weights_without_relabeling_legacy(tmp_path):
    source = _public_source(tmp_path)
    capture = _recipe().capture_options
    fit = _recipe().fit_options
    weights = [0.5, 2.]
    recipe = _recipe(capture_options=capture, fit_options=fit, window_weights=weights)
    weights[0] = 99
    capture["cache_limits"]["max_payload_bytes"] = 1
    fit["metric_limits"]["max_windows"] = 1
    detached = recipe.fit_options
    detached["metric_limits"]["max_windows"] = 1
    assert recipe.window_weights == (0.5, 2.)
    assert recipe.capture_options["cache_limits"]["max_payload_bytes"] == 8192
    assert recipe.fit_options["metric_limits"]["max_windows"] == 8
    with pytest.raises(FrozenInstanceError):
        recipe.window_weights = ()
    other = _window(); other["input_ids"] = torch.tensor([2, 0, 1])
    calibrated = calibrate(source, [_window(), other], objective=_objective, owner_recipe=recipe)
    result = convert(source, calibrated, output_dir=tmp_path/"output", run_id="ordered-public")
    events = [json.loads(line) for line in (tmp_path/"output/evidence.jsonl").read_text().splitlines()]
    assert any(e.get("schema") == "tritium.quantize.owner_model_started" for e in events)
    assert not any(e.get("schema") == "tritium.quantize.calibrated_quantization_started" for e in events)
    assert result.source_id == source.observation.source_id


@pytest.mark.parametrize("variant", ["data", "iterator", "recipe", "objective", "evidence", "evidence_limit"])
def test_bad_public_calibration_admission_does_not_execute_objective(tmp_path, variant):
    source = _public_source(tmp_path)
    calls = []
    def objective(model, window):
        calls.append(1)
        return _objective(model, window)
    options = dict(objective=objective, owner_recipe=_recipe())
    data = [_window()]
    if variant == "data": data = []
    elif variant == "iterator": data = iter(data)
    elif variant == "recipe": options["owner_recipe"] = None
    elif variant == "objective": options["objective"] = None
    elif variant == "evidence": options["evidence_dir"] = tmp_path/"legacy"
    else: options["max_evidence_bytes"] = 1
    with pytest.raises((ValueError, TypeError)):
        calibrate(source, data, **options)
    assert calls == [] and list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("variant", ["unknown_capture", "unknown_fit", "missing_cache", "bool", "weights", "context"])
def test_bad_recipe_rejected_before_capture(variant):
    capture, fit = _recipe().capture_options, _recipe().fit_options
    changes = dict(capture_options=capture, fit_options=fit)
    if variant == "unknown_capture": capture["ignored"] = 7
    elif variant == "unknown_fit": fit["ignored"] = 7
    elif variant == "missing_cache": del capture["cache_limits"]
    elif variant == "bool": capture["max_graph_nodes"] = True
    elif variant == "weights": changes["window_weights"] = [True]
    else: changes["objective_bytes"] = b"x" * 4097
    with pytest.raises((TypeError, ValueError)):
        _recipe(**changes)


@pytest.mark.parametrize("argument,value", [
    ("revision", "other"), ("work_dir", "other"), ("packing", "q2"),
    ("max_working_bytes", 1), ("compact_max_bytes", 1),
    ("compact_max_resident_bytes", 1), ("near_lossless_max_bytes", 1),
    ("near_lossless_max_resident_bytes", 1), ("source_admission_receipt", "other"),
    ("official_identity_receipt", "other"),
])
def test_owner_convert_rejects_every_legacy_argument(tmp_path, argument, value):
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    with pytest.raises(TypeError, match="legacy arguments"):
        convert(source, calibration, output_dir=tmp_path/"rejected", run_id="rejected", **{argument: value})
    assert not (tmp_path/"rejected").exists()


@pytest.mark.parametrize("variant", ["source", "asset", "foreign", "legacy", "run", "output"])
def test_conversion_binding_failures_do_not_publish(tmp_path, variant):
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    options = dict(output_dir=tmp_path/"rejected", run_id="rejected")
    if variant == "source":
        with torch.no_grad(): source.prepared.model.embed.weight[0, 0] += 0.5
    elif variant == "asset": (tmp_path/"tokenizer.json").write_bytes(b"{ }")
    elif variant == "foreign": calibration = replace(calibration, source=replace(source))
    elif variant == "legacy": calibration = object()
    elif variant == "run": options["run_id"] = ""
    else: options["output_dir"] = None
    with pytest.raises((ValueError, TypeError)):
        convert(source, calibration, **options)
    assert not (tmp_path/"rejected").exists()
    assert list((tmp_path/"work").iterdir()) == []


def test_owner_arguments_do_not_fall_through_to_legacy_or_qat(tmp_path):
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    with pytest.raises(TypeError, match="native preparation"):
        calibrate(source.prepared, [_window()], objective=_objective, owner_recipe=_recipe())
    with pytest.raises(TypeError, match="native preparation"):
        convert(source.prepared, calibration)
    with pytest.raises(TypeError, match="native preparation"):
        convert(source.prepared, run_id="not-owner")
    with pytest.raises(TypeError, match="live PTQ"):
        prepare(TiedModel(), TernaryConfig.qat(), inplace=True, native_options=json.loads(source.options_json))
    with pytest.raises(ValueError, match="native_options"):
        prepare(TiedModel(), source.prepared.config, inplace=True, native_options={})
    config = replace(source.prepared.config, target_bpw=2.)
    with pytest.raises(ValueError, match="target_bpw"):
        prepare(TiedModel(), config, inplace=False, native_options=json.loads(source.options_json))


def test_real_hf_loss_through_public_phases_and_generation(tmp_path):
    from transformers import LlamaForCausalLM
    expert = _hf_source(tmp_path)
    source = prepare(expert.prepared.model, expert.prepared.config, inplace=False,
                     native_options=json.loads(expert.options_json))
    model = source.prepared.model.eval()
    capture = _recipe().capture_options
    capture.update(max_observation_bytes=1_000_000, max_invocations=32,
        max_samples=256, max_graph_nodes=4096, max_total_cache_bytes=8_000_000,
        cache_limits=dict(max_payload_bytes=1_000_000, max_context_bytes=4096,
            max_context_tensors=8, max_invocations=32, max_samples=256))
    fit = _recipe().fit_options
    fit.update(max_total_cache_bytes=8_000_000, max_cache_package_bytes=2_000_000,
        metric_limits=dict(max_windows=8, max_payload_bytes=2_000_000,
            max_row_bytes=4096, max_reduction_entries=1_000_000))
    recipe = _recipe(objective_bytes=b"tiny HF next-token CE mean", capture_options=capture, fit_options=fit)
    window = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), labels=torch.tensor([[1, 2, 3, 4]]))
    calibration = calibrate(source, [window], objective=lambda m, w: m(**w, use_cache=False).loss,
                            owner_recipe=recipe)
    result = convert(source, calibration, output_dir=tmp_path/"hf", run_id="public-hf")
    shell = LlamaForCausalLM(copy.deepcopy(model.config))
    loaded = load(result.output_dir, model=shell, level="full", load_run_id="hf-load", inplace=False).eval()
    assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    reference = copy.deepcopy(model)
    modules = dict(reference.named_modules(remove_duplicate=False))
    for path, module in loaded.named_modules(remove_duplicate=False):
        if hasattr(module, "packed_weight"):
            handle = module.packed_weight.handle
            with torch.no_grad(): modules[path].weight.copy_(torch.from_dlpack(handle.gather(torch.arange(handle.rows))))
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).logits, reference(tokens).logits, rtol=5e-6, atol=5e-6)
        generated = loaded.generate(tokens, max_new_tokens=3, do_sample=False)
        assert torch.equal(generated, reference.generate(tokens, max_new_tokens=3, do_sample=False))
        expected_logits = loaded(tokens).logits.tolist()
    model.config.to_json_file(tmp_path/"hf-config.json")
    replay = dict(tokens=tokens.tolist(), generated=generated.tolist(), logits=expected_logits)
    (tmp_path/"replay.json").write_text(json.dumps(replay))
    subprocess.run([sys.executable, "-I", "-c", """
import json,sys,torch
from pathlib import Path
from transformers import LlamaConfig,LlamaForCausalLM
from tritium.torch import load
p=Path(sys.argv[1]); expected=json.loads((p/'replay.json').read_text())
model=load(p/'hf',model=LlamaForCausalLM(LlamaConfig.from_json_file(p/'hf-config.json')),
           level='full',load_run_id='public-fresh-process',inplace=False).eval()
tokens=torch.tensor(expected['tokens'])
with torch.no_grad():
    torch.testing.assert_close(model(tokens).logits,torch.tensor(expected['logits']),rtol=5e-6,atol=5e-6)
    assert model.generate(tokens,max_new_tokens=3,do_sample=False).tolist()==expected['generated']
assert model.model.embed_tokens.packed_weight is model.lm_head.packed_weight
""", str(tmp_path)], check=True, capture_output=True, text=True, timeout=30)
    if location := os.environ.get("TRITIUM_OWNER_PHASE_ORACLE_DIR"):
        from safetensors.torch import save_file
        root = Path(location)
        root.mkdir()
        shutil.copytree(result.output_dir, root/"hf")
        for name in ("hf-config.json", "replay.json", "config.json", "tokenizer.json"):
            shutil.copyfile(tmp_path/name, root/name)
        save_file({name: value.detach().contiguous().clone() for name, value in model.state_dict().items()},
                  root/"original.safetensors")
        (root/"source-options.json").write_text(source.options_json)
        (root/"recipe.json").write_text(json.dumps(dict(
            objective_bytes_hex=recipe.objective_bytes.hex(), window_weights=recipe.window_weights,
            capture_options=recipe.capture_options, fit_options=recipe.fit_options)))
        calibration.captures[0].save(root/"cache", max_package_bytes=fit["max_cache_package_bytes"])
    assert all(parameter.grad is None for parameter in model.parameters())
    assert list((tmp_path/"work").iterdir()) == []
