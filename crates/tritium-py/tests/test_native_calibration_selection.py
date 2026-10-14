"""Actual weighted/masked capture and stable source paths through public native phases."""
import json
import pytest
import torch
from tritium.torch import (prepare_native_module, calibrate_native_module,
    convert_calibrated_native_module, NativeModuleCalibration)
from test_native_module_source import _prepared, _fit
from test_native_module_calibration import _source, _windows, _objective


def _selected_capture(source, windows):
    return calibrate_native_module(source, windows, _objective,
        objective_bytes=b"declared selected squared mean", cache_limits=dict(
            max_payload_bytes=32768, max_samples=64, max_windows=8),
        max_total_payload_bytes=32768)


def test_preparation_paths_are_bound_before_cwd_changes(tmp_path, monkeypatch):
    root=tmp_path/"initial"; root.mkdir(); (root/"work").mkdir()
    for name in ("config.json","tokenizer.json"): (root/name).write_bytes(b"{}")
    monkeypatch.chdir(root)
    source=prepare_native_module(_prepared(),actions={"a.weight":_fit()},levels=["compact","full"],
        limits=dict(max_source_bytes=4096,max_asset_bytes=32768,max_retained_blob_bytes=65536,max_package_bytes=100_000),
        work_dir="work",asset_paths={name:name for name in ("config.json","tokenizer.json")})
    changed=tmp_path/"changed"; changed.mkdir(); monkeypatch.chdir(changed)
    source.verify()
    captured=_selected_capture(source,_windows())
    captured.save(root/"caches",max_package_bytes=100_000)
    reopened=NativeModuleCalibration.open(source,root/"caches",max_package_bytes=100_000,max_total_payload_bytes=32768)
    cache=reopened.caches[0][1]
    result=convert_calibrated_native_module(reopened,root/"output","cwd-stable",partition="Full",damping=0.01,
        max_numeric_bytes=cache.required_numeric_bytes('"Full"',0.01),max_cache_package_bytes=100_000)
    result.verify()
    assert list((root/"work").iterdir())==[] and list(changed.iterdir())==[]


def test_weighted_masked_objective_matches_native_cache_and_reopens(tmp_path):
    source=_source(tmp_path)
    windows=_windows()
    for window in windows:
        window["calibration_weights"]=torch.tensor([1.,0.5,2.,0.],dtype=torch.float64)
        window["calibration_mask"]=torch.tensor([True,False,True,True])
    actual=_selected_capture(source,windows)
    builder=source.observation.cache_builder("a.weight",b"{}",b"declared selected squared mean",'"DenseFisher"',
        json.dumps(dict(max_payload_bytes=32768,max_samples=64,max_windows=8)))
    model=source.prepared.model
    for window in windows:
        with torch.no_grad():
            inputs=model.norm(window["inputs"])
            value=model(window["inputs"])
            gradient=2*value/value.numel()
        for _ in range(2):
            builder.capture(window["input_ids"],inputs,gradients=gradient,
                            weights=window["calibration_weights"],mask=window["calibration_mask"].to(torch.uint8))
    expected=builder.finish()
    cache=actual.caches[0][1]
    assert cache.cache_id==expected.cache_id
    unselected=_selected_capture(source,_windows()).caches[0][1]
    assert unselected.cache_id!=cache.cache_id
    directory=actual.save(tmp_path/"caches",max_package_bytes=100_000)
    reopened=NativeModuleCalibration.open(source,directory,max_package_bytes=100_000,max_total_payload_bytes=32768)
    assert reopened.caches[0][1].cache_id==cache.cache_id
    result=convert_calibrated_native_module(reopened,tmp_path/"output","selected-fit",partition="Full",damping=0.01,
        max_numeric_bytes=cache.required_numeric_bytes('"Full"',0.01),max_cache_package_bytes=100_000)
    result.verify()
    events=[json.loads(line) for line in (tmp_path/"output/evidence.jsonl").read_text().splitlines()]
    factors=next(event["payload"]["factors"] for event in events if event.get("schema")=="tritium.quantize.calibrated_levels_fitted")
    for factor in [*factors["inputs"], factors["output"]]:
        assert factor["samples"]==16 and factor["selected"]==12
        assert factor["total_weight"]==12.0
    assert factors["inputs"][0]["selection"]==factors["output"]["selection"]


def test_explicit_unit_selection_preserves_implicit_cache_bytes(tmp_path):
    source=_source(tmp_path)
    implicit=_selected_capture(source,_windows()).caches[0][1]
    windows=_windows()
    for window in windows:
        window["calibration_weights"]=torch.ones_like(window["input_ids"],dtype=torch.float32)
        window["calibration_mask"]=torch.ones_like(window["input_ids"],dtype=torch.bool)
    explicit=_selected_capture(source,windows).caches[0][1]
    assert explicit.cache_id==implicit.cache_id
    assert explicit.package(100_000)==implicit.package(100_000)


@pytest.mark.parametrize("variant",["negative","nan-masked","shape","integer","mask-integer","mask-shape"])
def test_invalid_selection_rejects_before_objective_and_cleans_hooks(tmp_path,variant):
    source=_source(tmp_path); windows=_windows(); calls=0
    window=windows[0]
    window["calibration_weights"]=torch.ones(4,dtype=torch.float64)
    window["calibration_mask"]=torch.ones(4,dtype=torch.bool)
    if variant=="negative": window["calibration_weights"][0]=-1
    elif variant=="nan-masked":
        window["calibration_weights"][0]=float("nan"); window["calibration_mask"][0]=False
    elif variant=="shape": window["calibration_weights"]=torch.ones(1,4)
    elif variant=="integer": window["calibration_weights"]=torch.ones(4,dtype=torch.int64)
    elif variant=="mask-integer": window["calibration_mask"]=torch.ones(4,dtype=torch.int64)
    else: window["calibration_mask"]=torch.ones(1,4,dtype=torch.bool)
    def objective(model,window):
        nonlocal calls
        calls+=1
        return _objective(model,window)
    with pytest.raises(ValueError):
        calibrate_native_module(source,windows,objective,objective_bytes=b"declared invalid probe",
            cache_limits=dict(max_payload_bytes=32768,max_samples=64,max_windows=8),max_total_payload_bytes=32768)
    assert calls==0 and all(not module._forward_hooks for module in source.prepared.model.modules())


def test_attention_mask_is_not_implicitly_used_for_calibration(tmp_path):
    source=_source(tmp_path); windows=_windows()
    for window in windows: window["attention_mask"]=torch.zeros(4,dtype=torch.int64)
    with_attention=_selected_capture(source,windows).caches[0][1]
    implicit=_selected_capture(source,_windows()).caches[0][1]
    assert with_attention.cache_id==implicit.cache_id


@pytest.mark.parametrize("variant",["weight-dtype","weight-shape","weight-nan-masked","mask-value","mask-shape","masked-input","masked-gradient"])
def test_native_selection_errors_are_atomic(tmp_path,variant):
    source=_source(tmp_path)
    builder=source.observation.cache_builder("a.weight",b"{}",b"native selection transaction",'"DenseFisher"',
        json.dumps(dict(max_payload_bytes=32768,max_samples=64,max_windows=8)))
    tokens=torch.arange(4,dtype=torch.int64); inputs=torch.ones(4,32); gradients=torch.ones(4,1)
    weights=torch.ones(4,dtype=torch.float64); mask=torch.tensor([0,1,1,1],dtype=torch.uint8)
    if variant=="weight-dtype": weights=weights.float()
    elif variant=="weight-shape": weights=weights.reshape(1,4)
    elif variant=="weight-nan-masked": weights[0]=float("nan")
    elif variant=="mask-value": mask[0]=2
    elif variant=="mask-shape": mask=mask.reshape(1,4)
    elif variant=="masked-input": inputs[0,0]=float("nan")
    else: gradients[0,0]=float("nan")
    with pytest.raises(ValueError): builder.capture(tokens,inputs,gradients=gradients,weights=weights,mask=mask)
    builder.capture(tokens,torch.ones(4,32),gradients=torch.ones(4,1),weights=torch.ones(4,dtype=torch.float64),
                    mask=torch.ones(4,dtype=torch.uint8))
    actual=builder.finish()
    assert json.loads(actual.binding_json)["samples"]==4


@pytest.mark.parametrize("field",["input_ids","calibration_weights","calibration_mask"])
def test_objective_cannot_mutate_captured_window_context(tmp_path,field):
    source=_source(tmp_path); windows=_windows()
    for window in windows:
        window["calibration_weights"]=torch.ones(4,dtype=torch.float64)
        window["calibration_mask"]=torch.ones(4,dtype=torch.bool)
    def objective(model,window):
        loss=_objective(model,window)
        window[field][0]=0
        return loss
    with pytest.raises(ValueError,match="window context changed"):
        calibrate_native_module(source,windows,objective,objective_bytes=b"declared mutation probe",
            cache_limits=dict(max_payload_bytes=32768,max_samples=64,max_windows=8),max_total_payload_bytes=32768)
    assert all(not module._forward_hooks for module in source.prepared.model.modules())


def test_cache_reopen_obeys_remaining_payload_limit_before_decode(tmp_path,monkeypatch):
    from tritium import _tritium
    source=_source(tmp_path)
    calibration=_selected_capture(source,_windows())
    directory=calibration.save(tmp_path/"caches",max_package_bytes=100_000)
    cache=calibration.caches[0][1]
    with pytest.raises(ValueError):
        _tritium.NativeCalibrationCache.open(str(directory/"cache-0.trit"),cache.binding_json,100_000,
                                             max_payload_bytes=cache.payload_bytes-1)
    original=_tritium.NativeCalibrationCache.open
    seen=[]
    def opened(path,binding,package,**kwargs):
        seen.append(kwargs["max_payload_bytes"])
        return original(path,binding,package,**kwargs)
    monkeypatch.setattr(_tritium.NativeCalibrationCache,"open",opened)
    reopened=NativeModuleCalibration.open(source,directory,max_package_bytes=100_000,
                                         max_total_payload_bytes=cache.payload_bytes)
    assert seen==[cache.payload_bytes] and reopened.caches[0][1].cache_id==cache.cache_id
