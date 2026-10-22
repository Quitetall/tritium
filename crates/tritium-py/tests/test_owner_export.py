"""Public artifact portability and one-call conversion use actual native paths."""
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tritium import NativeModelConversion
from tritium.torch import (
    TernaryConfig, calibrate, calibrate_native_module, convert, convert_native_module,
    convert_calibrated_native_module, export, load, quantize,
)
from test_owner_capture import TiedModel, _window, _objective, _hf_source
from test_owner_phases import _public_source, _recipe
from test_native_module_calibration import (
    _source as _linear_source, _windows as _linear_windows, _objective as _linear_objective,
)


def _result(root, mode="owner", dtype=torch.float32):
    if mode == "calibrated":
        from test_native_module_source import StateModel, _prepared
        source = _linear_source(root, _prepared(StateModel(dtype)))
        windows = _linear_windows()
        for window in windows: window["inputs"] = window["inputs"].to(dtype=dtype)
        calibration = calibrate_native_module(source, windows, _linear_objective,
            objective_bytes=b"portable squared output", cache_limits=dict(
                max_payload_bytes=32768, max_samples=64, max_windows=8), max_total_payload_bytes=32768)
        cache = calibration.caches[0][1]
        return convert_calibrated_native_module(calibration, root/"original", "portable-native",
            partition="Full", damping=0.1, max_numeric_bytes=cache.required_numeric_bytes('"Full"', 0.1),
            max_cache_package_bytes=100_000)
    source = _public_source(root, dtype=dtype)
    if mode == "plain":
        return convert_native_module(source.prepared, root/"original", "portable-native",
                                     **json.loads(source.options_json))
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    return convert(source, calibration, output_dir=root/"original", run_id="portable-native")


def _identity(result):
    return (result.source_id, result.package_id, result.model_ids, result.evidence_root, result.run_id)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux address-space allowance")
def test_native_metadata_admission_does_not_allocate_a_package_image(tmp_path):
    import tritium
    from test_native_module_source import _prepared, _convert

    prepared = _prepared()
    prepared.model.register_buffer("large_preserved", torch.zeros(16 * 1024 * 1024))
    result = _convert(
        tmp_path, prepared,
        limits=dict(max_source_bytes=128 * 1024**2, max_asset_bytes=32768,
                    max_retained_blob_bytes=128 * 1024**2, max_package_bytes=128 * 1024**2),
    )
    assert (Path(result.output_dir) / "model.trit").stat().st_size > 64 * 1024**2
    program = '''import importlib.util,json,os,resource,sys
spec=importlib.util.spec_from_file_location("_tritium",sys.argv[1])
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
virtual=int(open("/proc/self/statm").read().split()[0])*os.sysconf("SC_PAGE_SIZE")
soft,hard=resource.getrlimit(resource.RLIMIT_AS)
limit=virtual+32*1024**2
if soft!=resource.RLIM_INFINITY: limit=min(limit,soft)
resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
opened=native.NativeModelConversion.open(sys.argv[2],max_package_bytes=128*1024**2)
assert opened.package_id==sys.argv[3]
opened.verify()
print("bounded native metadata reopen and verification")
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program,
         str(Path(tritium._tritium.__file__).resolve()), result.output_dir, result.package_id],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", RAYON_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("mode", ["plain", "calibrated", "owner"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_actual_native_exports_preserve_bytes_identities_and_standalone_load(tmp_path, mode, dtype):
    result = _result(tmp_path, mode, dtype)
    original = Path(result.output_dir)
    (original/"unreferenced.txt").write_bytes(b"not part of the artifact")
    expected = {name: (original/name).read_bytes() for name in ("model.trit", "evidence.jsonl")}
    result = result._with_source_cleanup_warning("original warning retained")
    copied = export(result, tmp_path/"exported", max_package_bytes=len(expected["model.trit"]),
                    max_evidence_bytes=len(expected["evidence.jsonl"]))
    assert isinstance(copied, NativeModelConversion) and _identity(copied) == _identity(result)
    assert copied.publication_warnings[:len(result.publication_warnings)] == result.publication_warnings
    assert {path.name for path in (tmp_path/"exported").iterdir()} == set(expected)
    for name, payload in expected.items():
        assert (tmp_path/"exported"/name).read_bytes() == payload
        assert (original/name).read_bytes() == payload
    shutil.rmtree(original)
    reopened = load(copied.output_dir, format="trit", max_package_bytes=len(expected["model.trit"]),
                    max_evidence_bytes=len(expected["evidence.jsonl"]))
    assert _identity(reopened) == _identity(copied)
    assert _identity(load(copied)) == _identity(copied)
    if mode == "calibrated":
        from test_native_module_source import StateModel
        shell = StateModel(dtype)
    else: shell = TiedModel().to(dtype=dtype)
    loaded = load(reopened, model=shell, level="full", load_run_id="portable-load", inplace=False,
                  max_package_bytes=len(expected["model.trit"]), max_evidence_bytes=len(expected["evidence.jsonl"]))
    if mode != "calibrated":
        assert loaded.embed.packed_weight is loaded.head.packed_weight
        assert loaded.embed is loaded.embed_alias
        assert all(torch.isfinite(value).all() for value in loaded(_window()["input_ids"]))
    else:
        assert loaded.a.packed_weight is loaded.b.packed_weight
        assert loaded.norm.weight.dtype == dtype
    assert list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("field", ["max_package_bytes", "max_evidence_bytes"])
@pytest.mark.parametrize("value", [0, 1])
def test_export_and_metadata_load_limits_refuse_before_publication(tmp_path, field, value):
    result = _result(tmp_path)
    with pytest.raises(ValueError):
        export(result, tmp_path/"refused", **{field: value})
    assert not (tmp_path/"refused").exists()
    with pytest.raises(ValueError): load(result, **{field: value})
    with pytest.raises(ValueError): load(result.output_dir, format="trit", **{field: value})
    with pytest.raises(ValueError):
        load(result, model=TiedModel(), level="full", load_run_id="refused", inplace=False, **{field: value})
    result.verify()


@pytest.mark.parametrize("variant", ["model", "log", "foreign"])
def test_native_export_rejects_changed_source_not_just_well_formed_replacement(tmp_path, variant):
    result = _result(tmp_path)
    root = Path(result.output_dir)
    if variant == "model":
        payload = bytearray((root/"model.trit").read_bytes()); payload[-1] ^= 1
        (root/"model.trit").write_bytes(payload)
    elif variant == "log": (root/"evidence.jsonl").write_bytes(b"corrupt")
    else:
        other_root = tmp_path/"other"; other_root.mkdir()
        other = _result(other_root, mode="plain")
        for name in ("model.trit", "evidence.jsonl"):
            shutil.copyfile(Path(other.output_dir)/name, root/name)
        NativeModelConversion.open(str(root)).verify()
    with pytest.raises(ValueError): export(result, tmp_path/"refused")
    assert not (tmp_path/"refused").exists()


def test_native_export_no_clobber_preserves_foreign_and_original_files(tmp_path):
    result = _result(tmp_path)
    target = tmp_path/"foreign"; target.mkdir(); (target/"keep").write_bytes(b"retain")
    with pytest.raises(FileExistsError): export(result, target)
    assert (target/"keep").read_bytes() == b"retain"
    with pytest.raises(FileExistsError): export(result, result.output_dir)
    result.verify()


def test_native_export_refuses_symlink_destination(tmp_path):
    result = _result(tmp_path)
    target = tmp_path/"foreign"; target.mkdir(); (target/"keep").write_bytes(b"retain")
    link = tmp_path/"link"
    try: link.symlink_to(target, target_is_directory=True)
    except OSError as error: pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(FileExistsError): export(result, link)
    assert link.is_symlink() and (target/"keep").read_bytes() == b"retain"


@pytest.mark.parametrize("inplace", [False, True])
def test_one_call_owner_fit_is_exact_phase_composition(tmp_path, inplace):
    source = _public_source(tmp_path)
    recipe = _recipe()
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=recipe)
    phased = convert(source, calibration, output_dir=tmp_path/"phased", run_id="same-composition")
    model = copy.deepcopy(source.prepared.model)
    del model._tritium_coverage
    before = model.embed.weight.detach().clone()
    calls = []
    def objective(m, window):
        calls.append(1)
        return _objective(m, window)
    composed = quantize(model, source.prepared.config, inplace=inplace,
        native_options=json.loads(source.options_json), data=[_window()], objective=objective,
        owner_recipe=recipe, output_dir=tmp_path/"one-call", run_id="same-composition")
    assert calls == [1] and _identity(composed) == _identity(phased)
    for name in ("model.trit", "evidence.jsonl"):
        assert (tmp_path/"phased"/name).read_bytes() == (tmp_path/"one-call"/name).read_bytes()
    assert torch.equal(model.embed.weight, before) and model.embed.weight.grad is None
    assert model.embed.weight is model.head.weight and model.embed is model.embed_alias
    assert hasattr(model, "_tritium_coverage") is inplace
    assert list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("variant", ["inplace", "recipe", "objective", "data", "run", "output", "exists", "legacy", "missing-native"])
def test_bad_one_call_request_does_not_execute_objective_or_publish(tmp_path, variant):
    source = _public_source(tmp_path)
    calls = []
    def objective(m, window): calls.append(1); return _objective(m, window)
    options = dict(inplace=False, native_options=json.loads(source.options_json), data=[_window()],
        objective=objective, owner_recipe=_recipe(), output_dir=tmp_path/"refused", run_id="refused")
    if variant in {"inplace", "recipe", "objective", "data", "run", "output"}:
        del options[{"recipe":"owner_recipe", "run":"run_id", "output":"output_dir"}.get(variant, variant)]
    elif variant == "exists": (tmp_path/"refused").mkdir()
    elif variant == "legacy": options["revision"] = "not-owner"
    else: del options["native_options"]
    before = source.prepared.model.embed.weight.detach().clone()
    with pytest.raises((TypeError, ValueError, FileExistsError)):
        quantize(source.prepared.model, source.prepared.config, **options)
    assert calls == [] and torch.equal(source.prepared.model.embed.weight, before)
    assert not (tmp_path/"refused").exists() or variant == "exists"
    assert list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("revision", "other"), ("work_dir", "other"), ("source_admission_receipt", "other"),
    ("official_identity_receipt", "other"), ("evidence_dir", "other"),
    ("compact_max_bytes", 1), ("compact_max_resident_bytes", 1),
    ("near_lossless_max_bytes", 1), ("near_lossless_max_resident_bytes", 1),
    ("packing", "q2"), ("max_evidence_bytes", 1),
])
def test_every_incompatible_one_call_legacy_option_refuses_before_callback(tmp_path, field, value):
    source = _public_source(tmp_path)
    calls = []
    def objective(m, window): calls.append(1); return _objective(m, window)
    options = dict(inplace=False, native_options=json.loads(source.options_json), data=[_window()],
        objective=objective, owner_recipe=_recipe(), output_dir=tmp_path/"refused", run_id="refused")
    options[field] = value
    with pytest.raises(TypeError, match="legacy arguments"):
        quantize(source.prepared.model, source.prepared.config, **options)
    assert calls == [] and not (tmp_path/"refused").exists()


def test_native_artifact_selectors_do_not_silently_enter_legacy_paths(tmp_path):
    with pytest.raises(TypeError, match="explicit native"):
        load(tmp_path, max_package_bytes=100)
    with pytest.raises(TypeError, match="NativeModelConversion"):
        export(object(), tmp_path/"refused", max_evidence_bytes=100)
    with pytest.raises(TypeError, match="missing required"):
        quantize(tmp_path/"missing", TernaryConfig.ptq(profile="compact-v1"))
    for format in ("salt", "guess", "", True):
        with pytest.raises(ValueError, match="format"):
            load(tmp_path, format=format)


def test_one_call_hf_export_fresh_process_greedy_generation_without_original(tmp_path):
    from transformers import LlamaForCausalLM
    from tritium.torch import OwnerCalibrationRecipe
    source = _hf_source(tmp_path)
    model = source.prepared.model.eval()
    capture = _recipe().capture_options
    capture.update(max_observation_bytes=1_000_000, max_invocations=32, max_samples=256,
        max_graph_nodes=4096, max_total_cache_bytes=8_000_000,
        cache_limits=dict(max_payload_bytes=1_000_000, max_context_bytes=4096,
            max_context_tensors=8, max_invocations=32, max_samples=256))
    fit = _recipe().fit_options
    fit.update(max_total_cache_bytes=8_000_000, max_cache_package_bytes=2_000_000,
        metric_limits=dict(max_windows=8, max_payload_bytes=2_000_000,
            max_row_bytes=4096, max_reduction_entries=1_000_000))
    recipe = OwnerCalibrationRecipe(objective_bytes=b"tiny HF next-token CE mean",
        window_weights=[1.], capture_options=capture, fit_options=fit)
    window = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), labels=torch.tensor([[1, 2, 3, 4]]))
    result = quantize(model, source.prepared.config, native_options=json.loads(source.options_json),
        inplace=False, data=[window], objective=lambda m, w: m(**w, use_cache=False).loss,
        owner_recipe=recipe, output_dir=tmp_path/"original", run_id="one-call-hf")
    copied = export(result, tmp_path/"exported")
    assert _identity(copied) == _identity(result)
    for name in ("model.trit", "evidence.jsonl"):
        assert (tmp_path/"original"/name).read_bytes() == (tmp_path/"exported"/name).read_bytes()
    model.config.to_json_file(tmp_path/"hf-config.json")
    loaded = load(copied.output_dir, format="trit", model=LlamaForCausalLM(copy.deepcopy(model.config)),
        level="full", load_run_id="exported-hf", inplace=False).eval()
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
        replay = dict(tokens=tokens.tolist(), logits=loaded(tokens).logits.tolist(), generated=generated.tolist())
    (tmp_path/"replay.json").write_text(json.dumps(replay))
    if location := os.environ.get("TRITIUM_OWNER_EXPORT_ORACLE_DIR"):
        from safetensors.torch import save_file
        root = Path(location); root.mkdir()
        shutil.copytree(copied.output_dir, root/"exported")
        for name in ("hf-config.json", "replay.json", "config.json", "tokenizer.json"):
            shutil.copyfile(tmp_path/name, root/name)
        save_file({name: value.detach().contiguous().clone() for name, value in model.state_dict().items()},
                  root/"original.safetensors")
        (root/"source-options.json").write_text(source.options_json)
        (root/"selection-config.json").write_text(json.dumps(source.prepared.config.to_dict()))
        (root/"recipe.json").write_text(json.dumps(dict(objective_bytes_hex=recipe.objective_bytes.hex(),
            window_weights=recipe.window_weights, capture_options=recipe.capture_options, fit_options=recipe.fit_options)))
        (root/"calibration.json").write_text(json.dumps({name: value.tolist() for name, value in window.items()}))
    shutil.rmtree(result.output_dir)
    for name in ("config.json", "tokenizer.json"): (tmp_path/name).unlink()
    subprocess.run([sys.executable, "-I", "-c", """
import json,sys,torch
from pathlib import Path
from transformers import LlamaConfig,LlamaForCausalLM
from tritium.torch import load
p=Path(sys.argv[1]); expected=json.loads((p/'replay.json').read_text())
model=load(p/'exported',format='trit',model=LlamaForCausalLM(LlamaConfig.from_json_file(p/'hf-config.json')),
           level='full',load_run_id='exported-fresh-process',inplace=False).eval()
tokens=torch.tensor(expected['tokens'])
with torch.no_grad():
    torch.testing.assert_close(model(tokens).logits,torch.tensor(expected['logits']),rtol=5e-6,atol=5e-6)
    assert model.generate(tokens,max_new_tokens=3,do_sample=False).tolist()==expected['generated']
assert model.model.embed_tokens.packed_weight is model.lm_head.packed_weight
""", str(tmp_path)], check=True, capture_output=True, text=True, timeout=30)
    assert list((tmp_path/"work").iterdir()) == []
