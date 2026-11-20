"""Real Torch objectives -> shared immutable cache -> calibrated engine -> native runtime."""
import copy
import json
import pytest
import torch
from torch import nn
from tritium import _tritium
from tritium.torch import (prepare_native_module, calibrate_native_module,
                          convert_calibrated_native_module, NativeModuleCalibration, load_native_module)
from test_native_module_source import StateModel, _prepared, _fit


def _source(root, prepared=None):
    work=root/"work"; work.mkdir(exist_ok=True)
    for name in ("config.json","tokenizer.json"): (root/name).write_bytes(b"{}")
    return prepare_native_module(_prepared() if prepared is None else prepared, actions={"a.weight":_fit()},
        levels=["compact","full"], limits=dict(max_source_bytes=4096,max_asset_bytes=32768,
            max_retained_blob_bytes=65536,max_package_bytes=100_000), work_dir=work,
        asset_paths={name:root/name for name in ("config.json","tokenizer.json")})


def _windows():
    generator=torch.Generator().manual_seed(83)
    return [dict(input_ids=torch.tensor([1,2,3,4]), inputs=torch.randn(4,32,generator=generator)),
            dict(input_ids=torch.tensor([5,6,7,8]), inputs=torch.randn(4,32,generator=generator))]


def _objective(model,window): return model(window["inputs"]).square().mean()


def _capture(source,**kwargs):
    options=dict(objective_bytes=b"declared squared-output mean v1",cache_limits=dict(
        max_payload_bytes=32768,max_samples=64,max_windows=8),max_total_payload_bytes=32768)
    options.update(kwargs)
    return calibrate_native_module(source,_windows(),_objective,**options)


@pytest.mark.parametrize("output,partition", [("DenseFisher","Full"),("None","Full"),
                                               ("DenseFisher",{"Groups":{"columns":13}})])
def test_real_objective_capture_restart_calibrated_native_execution(tmp_path,output,partition):
    source=_source(tmp_path)
    model=source.prepared.model
    before={name:value.clone() for name,value in model.state_dict().items()}
    for parameter in model.parameters(): parameter.grad=torch.ones_like(parameter)
    grad_ids={name:id(parameter.grad) for name,parameter in model.named_parameters()}
    count=0
    def objective(model,window):
        nonlocal count
        count+=1
        return _objective(model,window)
    calibration=calibrate_native_module(source,_windows(),objective,objective_bytes=b"mean squared-output v1",
        cache_limits=dict(max_payload_bytes=32768,max_samples=64,max_windows=8),
        max_total_payload_bytes=32768,output=output)
    assert count==2
    cache=calibration.caches[0][1]
    binding=json.loads(cache.binding_json)
    assert binding["samples"]==16 and binding["windows"]==4
    assert binding["request"]["output"]==output
    assert grad_ids=={name:id(parameter.grad) for name,parameter in model.named_parameters()}
    assert all(torch.equal(value,before[name]) for name,value in model.state_dict().items())
    assert all(not module._forward_hooks for module in model.modules())
    directory=calibration.save(tmp_path/"caches",max_package_bytes=100_000)
    reopened=NativeModuleCalibration.open(source,directory,max_package_bytes=100_000,max_total_payload_bytes=32768)
    assert reopened.caches[0][1].cache_id==cache.cache_id
    damping=0 if output=="None" else 0.01
    numeric=cache.required_numeric_bytes(json.dumps(partition),damping)
    result=convert_calibrated_native_module(reopened,tmp_path/"output","real-torch-calibration",
        partition=partition,damping=damping,max_numeric_bytes=numeric,max_cache_package_bytes=100_000)
    result.verify()
    loaded=load_native_module(StateModel(),result,level="full",load_run_id="calibrated-load",inplace=False)
    handle=loaded.a.packed_weight.handle
    dense=torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
    inputs=_windows()[0]["inputs"]
    with torch.no_grad():
        expected=2*torch.nn.functional.linear(loaded.norm(inputs),dense,loaded.a.bias)
        assert torch.allclose(loaded(inputs),expected,atol=4e-6,rtol=4e-6)
    log=[json.loads(line) for line in (tmp_path/"output/evidence.jsonl").read_text().splitlines()]
    assert any(event.get("schema")=="tritium.quantize.calibrated_levels_fitted" for event in log)
    assert not any(event.get("schema")=="tritium.quantize.direct_levels_fitted" for event in log)
    assert list((tmp_path/"work").iterdir())==[]
    with pytest.raises(FileExistsError): calibration.save(directory,max_package_bytes=100_000)


@pytest.mark.parametrize("variant",["weights","asset","missing","disconnected","nonfinite-gradient","tokens","budget","raise"])
def test_failed_capture_has_no_success_and_removes_hooks(tmp_path,variant):
    source=_source(tmp_path)
    model=source.prepared.model
    objective=_objective; windows=_windows()
    kwargs={}
    if variant=="weights":
        with torch.no_grad(): model.a.weight.add_(1)
    elif variant=="asset": (tmp_path/"tokenizer.json").write_bytes(b"changed")
    elif variant=="missing": objective=lambda m,w:m.a(m.norm(w["inputs"])).square().mean()
    elif variant=="disconnected":
        def objective(m,w):
            m(w["inputs"])
            return torch.ones((),requires_grad=True)
    elif variant=="nonfinite-gradient": objective=lambda m,w:(m(w["inputs"])*0).sqrt().sum()
    elif variant=="tokens": windows[0]["input_ids"][0]=-1
    elif variant=="budget": kwargs["cache_limits"]=dict(max_payload_bytes=128,max_samples=64,max_windows=8)
    else:
        def objective(m,w):
            m(w["inputs"])
            raise RuntimeError("objective failed")
    with pytest.raises((ValueError,RuntimeError)):
        calibrate_native_module(source,windows,objective,objective_bytes=b"test declared objective",
            cache_limits=kwargs.get("cache_limits",dict(max_payload_bytes=32768,max_samples=64,max_windows=8)),
            max_total_payload_bytes=32768)
    assert all(not module._forward_hooks for module in model.modules())
    assert list((tmp_path/"work").iterdir())==[]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cache_tamper_source_change_and_unsupported_consumers_are_not_rebound(tmp_path):
    source=_source(tmp_path)
    calibration=_capture(source)
    directory=calibration.save(tmp_path/"caches",max_package_bytes=100_000)
    package=directory/"cache-0.trit"
    data=package.read_bytes(); package.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(ValueError): NativeModuleCalibration.open(source,directory,max_package_bytes=100_000,max_total_payload_bytes=32768)
    with torch.no_grad(): source.prepared.model.norm.weight.add_(1)
    with pytest.raises(ValueError,match="source content changed"):
        convert_calibrated_native_module(calibration,tmp_path/"output","changed",partition="Full",damping=0.01,
            max_numeric_bytes=10_000_000,max_cache_package_bytes=100_000)
    assert not (tmp_path/"output").exists()


def test_native_builder_bounds_atomic_failure_seal_and_context(tmp_path):
    source=_source(tmp_path)
    with pytest.raises(ValueError,match="tokenizer"):
        source.observation.cache_builder("a.weight",b"wrong",b"objective",'"DenseFisher"',json.dumps(dict(max_payload_bytes=32768,max_samples=64,max_windows=8)))
    builder=source.observation.cache_builder("a.weight",b"{}",b"objective",'"DenseFisher"',json.dumps(dict(max_payload_bytes=32768,max_samples=64,max_windows=8)))
    tokens=torch.arange(4,dtype=torch.int64); inputs=torch.randn(4,32); grads=torch.randn(4,1)
    with pytest.raises(ValueError): builder.capture(tokens,inputs,gradients=torch.full((4,1),float("nan")))
    builder.capture(tokens,inputs,gradients=grads)
    cache=builder.finish()
    assert json.loads(cache.binding_json)["samples"]==4
    with pytest.raises(ValueError,match="sealed"): builder.capture(tokens,inputs,gradients=grads)
    with pytest.raises(ValueError,match="sealed"): builder.finish()
    with pytest.raises(AttributeError): source.observation.source_id="forged"
    with pytest.raises(TypeError): _tritium.NativeCalibrationCache()


def test_actual_untied_hf_objective_calibrates_all_selected_linear_owners(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM
    from tritium.torch import prepare,TernaryConfig
    config=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32,
                       tie_word_embeddings=False)
    model=LlamaForCausalLM(config).eval()
    prepared=prepare(model,TernaryConfig.ptq(profile="compact-v1",planes=2,target_modules=("Linear",)),inplace=True)
    actions={}
    parameters=dict(model.named_parameters(remove_duplicate=False))
    for entry in prepared.coverage.entries:
        if entry.disposition!="selected": continue
        rows,columns=parameters[entry.path].shape
        action=_fit(); plan=action["Fit"]["plan"]
        plan["layout"].update(rows=rows,cols=columns)
        plan["limits"].update(max_weights=rows*columns,max_metric_entries=columns**2,max_output_trits=3*rows*columns)
        actions[entry.path]=action
    work=tmp_path/"work"; work.mkdir()
    for name in ("config.json","tokenizer.json"): (tmp_path/name).write_bytes(b"{}")
    source=prepare_native_module(prepared,actions=actions,levels=["compact","full"],
        limits=dict(max_source_bytes=1_000_000,max_asset_bytes=65536,max_retained_blob_bytes=1_000_000,max_package_bytes=2_000_000),
        work_dir=work,asset_paths={name:tmp_path/name for name in ("config.json","tokenizer.json")})
    windows=[{"input_ids":torch.tensor([[1,2,3,4]])},{"input_ids":torch.tensor([[5,6,7,8]])}]
    calibration=calibrate_native_module(source,windows,lambda m,w:m(**w,labels=w["input_ids"]).loss,
        objective_bytes=b"declared HF next-token mean CE",cache_limits=dict(max_payload_bytes=32768,max_samples=16,max_windows=4),
        max_total_payload_bytes=len(actions)*32768)
    assert {name for name,_ in calibration.caches}==set(actions)
    assert all(json.loads(cache.binding_json)["samples"]==8 for _,cache in calibration.caches)
    numeric=max(cache.required_numeric_bytes('"Full"',0.01) for _,cache in calibration.caches)
    result=convert_calibrated_native_module(calibration,tmp_path/"output","hf-native-calibrated",partition="Full",
        damping=0.01,max_numeric_bytes=numeric,max_cache_package_bytes=100_000)
    loaded=load_native_module(LlamaForCausalLM(copy.deepcopy(config)),result.output_dir,level="full",load_run_id="hf-native-load",inplace=False)
    reference=copy.deepcopy(model)
    modules=dict(reference.named_modules(remove_duplicate=False))
    for path,module in loaded.named_modules(remove_duplicate=False):
        if not hasattr(module,"packed_weight"): continue
        handle=module.packed_weight.handle
        dense=torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
        with torch.no_grad(): modules[path].weight.copy_(dense)
    with torch.no_grad():
        tokens=windows[0]["input_ids"]
        assert torch.allclose(loaded(tokens).logits,reference(tokens).logits,atol=5e-6,rtol=5e-6)
        assert torch.equal(loaded.generate(tokens,max_new_tokens=2,do_sample=False),reference.generate(tokens,max_new_tokens=2,do_sample=False))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_mixed_gather_matmul_alias_is_not_silently_omitted(tmp_path):
    from tritium.torch import prepare,TernaryConfig
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__(); self.a=nn.Linear(32,32,bias=False); self.embed=nn.Embedding(32,32)
            self.embed.weight=self.a.weight
    model=Mixed()
    prepared=prepare(model,TernaryConfig.ptq(profile="compact-v1",planes=2,target_modules=("Linear","Embedding")),inplace=True)
    action=_fit(); action["Fit"]["plan"]["layout"]["rows"]=32
    action["Fit"]["plan"]["limits"].update(max_weights=1024,max_output_trits=3072)
    work=tmp_path/"work"; work.mkdir()
    for name in ("config.json","tokenizer.json"): (tmp_path/name).write_bytes(b"{}")
    source=prepare_native_module(prepared,actions={"a.weight":action},levels=["compact","full"],
        limits=dict(max_source_bytes=100_000,max_asset_bytes=32768,max_retained_blob_bytes=100_000,max_package_bytes=200_000),
        work_dir=work,asset_paths={name:tmp_path/name for name in ("config.json","tokenizer.json")})
    with pytest.raises(ValueError,match="mixed gather/matmul"):
        _capture(source)


def test_foreign_cache_and_numerical_budget_refuse_before_fitting(tmp_path):
    one=tmp_path/"one"; two=tmp_path/"two"; one.mkdir(); two.mkdir()
    source=_source(one); foreign=_source(two)
    cache=_capture(source)
    with pytest.raises(ValueError,match="observed source"):
        foreign.observation.validate_cache(cache.caches[0][1])
    with pytest.raises(ValueError,match="numerical workspace"):
        convert_calibrated_native_module(cache,one/"output","too-small",partition="Full",damping=0.01,
            max_numeric_bytes=1,max_cache_package_bytes=100_000)
    assert not (one/"output").exists() and list((one/"work").iterdir())==[]


def test_captured_gradients_match_analytic_objective_not_squared_activations(tmp_path):
    source=_source(tmp_path)
    actual=_capture(source)
    expected=source.observation.cache_builder("a.weight",b"{}",b"declared squared-output mean v1",'"DenseFisher"',
        json.dumps(dict(max_payload_bytes=32768,max_samples=64,max_windows=8)))
    model=source.prepared.model
    for window in _windows():
        with torch.no_grad():
            inputs=model.norm(window["inputs"])
            value=model(window["inputs"])
            # Each of the two additive Linear consumers has d mean(y^2)/d y_i = 2*y/N.
            gradient=2*value/value.numel()
        for _ in range(2): expected.capture(window["input_ids"],inputs,gradients=gradient)
    assert expected.finish().cache_id==actual.caches[0][1].cache_id


def test_postpublication_cache_cleanup_fault_returns_actual_result_with_warning(tmp_path,monkeypatch):
    import tempfile
    source=_source(tmp_path)
    calibration=_capture(source)
    original=tempfile.TemporaryDirectory.cleanup
    def cleanup(self):
        name=self.name
        original(self)
        if "tritium-native-fit-cache-" in name: raise OSError("injected cache cleanup fault")
    monkeypatch.setattr(tempfile.TemporaryDirectory,"cleanup",cleanup)
    cache=calibration.caches[0][1]
    result=convert_calibrated_native_module(calibration,tmp_path/"output","cleanup-warning",partition="Full",
        damping=0.01,max_numeric_bytes=cache.required_numeric_bytes('"Full"',0.01),max_cache_package_bytes=100_000)
    result.verify()
    assert any("CalibrationCacheCleanup" in warning for warning in result.publication_warnings)
    assert list((tmp_path/"work").iterdir())==[]
