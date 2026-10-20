"""Real Torch objectives -> immutable native caches -> fitted package -> chat graph."""
import copy
from dataclasses import replace
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch
from tritium import NativeModelConversion, _tritium
from tritium.torch import convert_owner_native_module, load_native_module
from test_owner_capture import TiedModel, _source, _window, _capture, _hf_source
from test_owner_cache import _durable, _open


def _options():
    return dict(window_weights=[1.], damping=0.1,
        metric_limits=dict(max_windows=8, max_payload_bytes=65536, max_row_bytes=1024,
                           max_reduction_entries=200_000),
        max_row_bindings=64, max_total_cache_bytes=65536, max_cache_package_bytes=65536)


def _convert(source, captures, path, **changes):
    options = _options(); options.update(changes)
    return convert_owner_native_module(source, captures, path, "owner-python-model", **options)


@pytest.mark.parametrize("dtype,precision", [(torch.float16,"F16"), (torch.bfloat16,"Bf16"), (torch.float32,"F32")])
@pytest.mark.parametrize("columns", [32, 64])
def test_actual_owner_conversion_preserves_source_and_loads_packed_tied_handles(tmp_path, dtype, precision, columns):
    source = _source(tmp_path, dtype=dtype, columns=columns)
    captured = _durable(source)
    before = source.prepared.model.embed.weight.detach().clone()
    result = _convert(source, [captured], tmp_path/"output")
    result.verify()
    assert result.source_id == source.observation.source_id
    events = [json.loads(line) for line in (tmp_path/"output/evidence.jsonl").read_text().splitlines()]
    start = next(e["payload"]["plan"] for e in events if e.get("schema")=="tritium.quantize.owner_model_started")
    assert start["tensors"][0]["source"]["precision"] == precision
    assert any(e.get("schema")=="tritium.quantize.owner_score_levels_fitted" for e in events)
    assert not any(e.get("schema")=="tritium.quantize.calibrated_levels_fitted" for e in events)
    loaded = load_native_module(TiedModel(columns).to(dtype=dtype), result, level="full", load_run_id="owner-load", inplace=False)
    assert loaded.embed is loaded.embed_alias
    assert loaded.embed.packed_weight is loaded.head.packed_weight
    assert not list(loaded.parameters())
    handle = loaded.head.packed_weight.handle
    dense = torch.from_dlpack(handle.gather(torch.arange(3)))
    values = torch.arange(2 * columns,dtype=torch.float32).reshape(2,columns)/32
    actual = torch.from_dlpack(handle.matmul(values))
    torch.testing.assert_close(actual, torch.nn.functional.linear(values,dense), rtol=3e-6,atol=3e-6)
    assert torch.equal(before, source.prepared.model.embed.weight)
    assert source.prepared.model.embed.weight.grad is None
    assert list((tmp_path/"work").iterdir()) == []


@pytest.mark.parametrize("columns", [32, 64])
def test_weighted_ordered_windows_reopen_replay_and_no_clobber(tmp_path, columns):
    source = _source(tmp_path, columns=columns)
    first = _durable(source)
    window = _window(); window["input_ids"] = torch.tensor([2,0,1])
    second = _durable(source,window)
    stored = first.save(tmp_path/"cache",max_package_bytes=65536)
    reopened = _open(source,stored)
    one = _convert(source,[first,second,first],tmp_path/"one",window_weights=[1.,2.,0.5])
    two = _convert(source,[reopened,second,reopened],tmp_path/"two",window_weights=[1.,2.,0.5])
    assert one.package_id == two.package_id
    for name in ("model.trit","evidence.jsonl"):
        assert (tmp_path/"one"/name).read_bytes() == (tmp_path/"two"/name).read_bytes()
    with pytest.raises(FileExistsError):
        _convert(source,[first],tmp_path/"one")
    assert list((tmp_path/"work").iterdir()) == []
    changed = _convert(source,[first,second],tmp_path/"changed",window_weights=[2.,1.])
    assert changed.package_id != one.package_id
    assert NativeModelConversion.open(one.output_dir).evidence_root == one.evidence_root


@pytest.mark.parametrize("change", ["empty","weights","nan","negative","bool","damping","rows","row_bytes",
    "reducer","payload","aggregate","package","plan","windows","ephemeral","inventory","recipe","source","asset"])
def test_owner_adapter_refuses_bad_inputs_without_publication_or_source_changes(tmp_path, change):
    source = _source(tmp_path); captured = _durable(source)
    captures = [captured]; options = {}
    if change=="empty": captures=[]
    elif change=="weights": options["window_weights"]=[]
    elif change=="nan": options["window_weights"]=[float("nan")]
    elif change=="negative": options["window_weights"]=[-1.]
    elif change=="bool": options["window_weights"]=[True]
    elif change=="damping": options["damping"]=-0.1
    elif change=="rows": options["max_row_bindings"]=1
    elif change in {"row_bytes","reducer","payload","windows"}:
        metric=_options()["metric_limits"]
        metric[{"row_bytes":"max_row_bytes","reducer":"max_reduction_entries","payload":"max_payload_bytes","windows":"max_windows"}[change]]=1
        options["metric_limits"]=metric
        if change=="windows": captures=[captured,captured]; options["window_weights"]=[1.,1.]
    elif change=="aggregate": options["max_total_cache_bytes"]=1
    elif change=="package": options["max_cache_package_bytes"]=1
    elif change=="plan": options["max_plan_bytes"]=1
    elif change=="ephemeral": captures=[_capture(source)]
    elif change=="inventory": captures=[replace(captured,caches=(captured.caches[0],captured.caches[0]))]
    elif change=="recipe":
        captures.append(_capture(source,objective_bytes=b"different scalar recipe",
            cache_limits=dict(max_payload_bytes=8192,max_context_bytes=4096,max_context_tensors=8,max_invocations=16,max_samples=64),
            max_total_cache_bytes=8192))
        options["window_weights"]=[1.,1.]
    elif change=="source":
        with torch.no_grad(): source.prepared.model.embed.weight[0,0]+=0.25
    else: (tmp_path/"config.json").write_bytes(b"{ }")
    before = source.prepared.model.embed.weight.detach().clone()
    with pytest.raises((ValueError,TypeError)):
        _convert(source,captures,tmp_path/"rejected",**options)
    assert not (tmp_path/"rejected").exists()
    assert list((tmp_path/"work").iterdir()) == []
    assert torch.equal(before,source.prepared.model.embed.weight)
    assert source.prepared.model.embed.weight.grad is None


def test_real_tied_hf_objective_converts_cold_loads_and_generates(tmp_path):
    from transformers import LlamaForCausalLM
    torch.manual_seed(972)
    source=_hf_source(tmp_path)
    model=source.prepared.model.eval()
    window=dict(input_ids=torch.tensor([[1,2,3,4]]),labels=torch.tensor([[1,2,3,4]]))
    captured=_capture(source,window,objective=lambda m,w:m(**w,use_cache=False).loss,
        objective_bytes=b"tiny HF next-token CE mean",max_observation_bytes=1_000_000,
        max_invocations=32,max_samples=256,max_graph_nodes=4096,
        cache_limits=dict(max_payload_bytes=1_000_000,max_context_bytes=4096,max_context_tensors=8,max_invocations=32,max_samples=256),
        max_total_cache_bytes=8_000_000)
    result=_convert(source,[captured],tmp_path/"hf-output",max_total_cache_bytes=8_000_000,max_cache_package_bytes=2_000_000,
        metric_limits=dict(max_windows=8,max_payload_bytes=2_000_000,max_row_bytes=4096,max_reduction_entries=1_000_000))
    loaded=load_native_module(LlamaForCausalLM(copy.deepcopy(model.config)),result.output_dir,
        level="full",load_run_id="owner-hf",inplace=False).eval()
    assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    reference=copy.deepcopy(model)
    modules=dict(reference.named_modules(remove_duplicate=False))
    for path,module in loaded.named_modules(remove_duplicate=False):
        if not hasattr(module,"packed_weight"): continue
        handle=module.packed_weight.handle
        dense=torch.from_dlpack(handle.gather(torch.arange(handle.rows)))
        with torch.no_grad(): modules[path].weight.copy_(dense)
    tokens=torch.tensor([[1,2,3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).logits,reference(tokens).logits,rtol=5e-6,atol=5e-6)
        actual=loaded.generate(tokens,max_new_tokens=3,do_sample=False)
        assert torch.equal(actual,reference.generate(tokens,max_new_tokens=3,do_sample=False))
    child='''import json,sys,torch
from transformers import LlamaConfig,LlamaForCausalLM
from tritium.torch import load_native_module
m=load_native_module(LlamaForCausalLM(LlamaConfig(**json.loads(sys.argv[2]))),sys.argv[1],level="full",load_run_id="cold",inplace=False).eval()
with torch.no_grad(): result=m.generate(torch.tensor([[1,2,3]]),max_new_tokens=3,do_sample=False)
assert result.tolist()==json.loads(sys.argv[3])
print("owner model cold generation parity")
'''
    isolation=["-I"] if os.environ.get("TRITIUM_TEST_INSTALLED_WHEEL")=="1" else []
    process=subprocess.run([sys.executable,*isolation,"-c",child,result.output_dir,json.dumps(model.config.to_dict()),json.dumps(actual.tolist())],
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"),capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    assert "owner model cold generation parity" in process.stdout
    assert not any("embed_tokens.weight" in name or "lm_head.weight" in name for name,_ in loaded.named_parameters())
    assert list((tmp_path/"work").iterdir())==[]
    if destination := os.environ.get("TRITIUM_OWNER_FRONTEND_ORACLE_DIR"):
        from safetensors.torch import save_file
        from tritium.torch.native_source import _inventory
        root=Path(destination); root.mkdir(parents=True,exist_ok=True)
        shutil.copytree(result.output_dir,root/"model")
        captured.save(root/"caches",max_package_bytes=2_000_000)
        save_file({name:tensor.detach().cpu().contiguous() for name,_,tensor,_,_,_ in _inventory(source.prepared)},str(root/"original-fixture.safetensors"))
        (root/"hf-config.json").write_text(json.dumps(model.config.to_dict(),sort_keys=True))
        (root/"generation.json").write_text(json.dumps(dict(source=result.source_id,package=result.package_id,
            evidence_root=result.evidence_root,seed=972,tokens=actual.tolist()),sort_keys=True))


@pytest.mark.parametrize("fault", ["cache","source","identity","extra","io"])
def test_native_receiver_rechecks_actual_files_and_cleans_owned_sources(tmp_path, monkeypatch, fault):
    source=_source(tmp_path); capture=_durable(source)
    original=_tritium.convert_owner_model_plan
    before=source.prepared.model.embed.weight.detach().clone()
    def receiver(plan,shards,output,run,**kwargs):
        paths=kwargs["owner_cache_paths"]
        if fault=="cache":
            path=Path(next(iter(next(iter(paths.values())).values())))
            raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
        elif fault=="source":
            path=Path(shards[0]); raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
        elif fault=="identity":
            files=next(iter(paths.values())); identity=next(iter(files)); files[identity.upper()]=files.pop(identity)
        elif fault=="extra": paths["invented"]={"00"*32:"missing"}
        else: raise OSError("injected receiver I/O failure")
        return original(plan,shards,output,run,**kwargs)
    monkeypatch.setattr(_tritium,"convert_owner_model_plan",receiver)
    with pytest.raises((ValueError,OSError)):
        _convert(source,[capture],tmp_path/"output")
    assert not (tmp_path/"output").exists()
    assert list((tmp_path/"work").iterdir())==[]
    assert torch.equal(before,source.prepared.model.embed.weight)


def test_successful_publication_retains_cleanup_warning_without_false_failure(tmp_path, monkeypatch):
    source=_source(tmp_path); capture=_durable(source)
    cleanup=tempfile.TemporaryDirectory.cleanup
    def fault(self):
        name=Path(self.name).name
        cleanup(self)
        if name.startswith("tritium-owner-fit-cache-"):
            raise OSError("injected post-publication cleanup report")
    monkeypatch.setattr(tempfile.TemporaryDirectory,"cleanup",fault)
    result=_convert(source,[capture],tmp_path/"output")
    result.verify()
    assert any("OwnerCacheCleanup" in w for w in result.publication_warnings)
    assert list((tmp_path/"work").iterdir())==[]


def test_actual_native_inputs_replay_as_the_same_model(tmp_path, monkeypatch):
    source=_source(tmp_path); capture=_durable(source)
    root=tmp_path/"inputs"; root.mkdir()
    original=_tritium.convert_owner_model_plan
    def record(plan,shards,output,run,**kwargs):
        (root/"plan.json").write_text(plan)
        manifest=dict(run=run,shards=[],assets={},caches={})
        for index,path in enumerate(shards):
            name=f"shard-{index}.safetensors"; shutil.copyfile(path,root/name); manifest["shards"].append(name)
        for index,(key,path) in enumerate(kwargs["asset_paths"].items()):
            name=f"asset-{index}.bin"; shutil.copyfile(path,root/name); manifest["assets"][key]=name
        for index,(owner,files) in enumerate(kwargs["owner_cache_paths"].items()):
            manifest["caches"][owner]={}
            for identity,path in files.items():
                name=f"cache-{index}-{identity}.trit"; shutil.copyfile(path,root/name)
                manifest["caches"][owner][identity]=name
        (root/"inputs.json").write_text(json.dumps(manifest,sort_keys=True))
        return original(plan,shards,output,run,**kwargs)
    monkeypatch.setattr(_tritium,"convert_owner_model_plan",record)
    one=_convert(source,[capture],tmp_path/"one")
    inputs=json.loads((root/"inputs.json").read_text())
    two=original((root/"plan.json").read_text(),[str(root/p) for p in inputs["shards"]],str(tmp_path/"two"),inputs["run"],
        asset_paths={k:str(root/p) for k,p in inputs["assets"].items()},
        owner_cache_paths={owner:{identity:str(root/p) for identity,p in files.items()} for owner,files in inputs["caches"].items()})
    assert one.package_id==two.package_id and one.evidence_root==two.evidence_root
    if destination:=os.environ.get("TRITIUM_OWNER_FRONTEND_CLI_ORACLE_DIR"):
        dest=Path(destination)
        shutil.copytree(root,dest/"inputs")
        shutil.copytree(one.output_dir,dest/"python")
