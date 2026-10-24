"""Actual tiny HF capture -> shared native cache -> QTEA candidate, not qualification.

Set TRITIUM_QTEA_LIVE_FIXTURES to the complete Rust file-campaign corpus. No
model download or hand-written Gram encoder is used; original metrics are explicit.
"""
import copy
import json
import os
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn
from tritium.torch import prepare_qtea_source


@pytest.fixture
def live(tmp_path):
    location=os.environ.get("TRITIUM_QTEA_LIVE_FIXTURES")
    if not location: pytest.skip("requires complete Rust-generated QTEA source/metric corpus")
    def make(mode="tied"):
        from safetensors.torch import load_file
        from transformers import LlamaConfig,LlamaForCausalLM
        original=Path(location)/mode
        destination=tmp_path/mode;destination.mkdir()
        shutil.copytree(original/"inputs",destination/"inputs")
        request=json.loads((original/"request.json").read_text())
        files=json.loads((original/"paths.json").read_text())
        files["source_shards"]=[destination/"inputs"/Path(p).name for p in files["source_shards"]]
        for kind in ("assets","curvature"):
            files[kind]={n:destination/"inputs"/Path(p).name for n,p in files[kind].items()}
        config=LlamaConfig.from_dict(json.loads(files["assets"]["config.json"].read_text()))
        dtype={"F32":torch.float32,"F16":torch.float16,"Bf16":torch.bfloat16}[request["source"]["tensors"][0]["source"]["precision"]]
        model=LlamaForCausalLM(config).to(dtype=dtype);model.tie_weights();model.eval()
        state=load_file(str(files["source_shards"][0]))
        missing=model.load_state_dict(state,strict=False)
        assert not missing.unexpected_keys
        assert set(missing.missing_keys)==({"lm_head.weight"} if config.tie_word_embeddings else set())
        base={name:copy.deepcopy(request[name]) for name in ("campaign","source","model_profile","limits")}
        base["schema"]="tritium.qtea.source-preparation.v1"
        source=prepare_qtea_source(model,base,source_shards=files["source_shards"],asset_paths=files["assets"],max_live_tensor_bytes=1<<16)
        native={o["name"]:dict(spec=o["curvature"],path=files["curvature"][o["name"]])
                for o in request["owners"] if o["curvature"]["kind"]=="native"}
        options={o["name"]:{n:o[n] for n in ("recipe","profile","estimate")} for o in request["owners"]}
        work=destination/"work";work.mkdir()
        return source,native,options,request["run"],work,destination
    return make


def _windows():
    return [dict(input_ids=torch.tensor([[1,2,3,1]]),
        calibration_weights=torch.tensor([[0.25,1.,2.,1.]]),
        calibration_mask=torch.tensor([[True,True,False,True]]))]


def _objective(model,window):
    return model(input_ids=window["input_ids"],use_cache=False).logits.float().square().mean()


def _capture(source,native,objective=_objective):
    return source.calibrate(_windows(),objective,native_curvature=native,
        objective_bytes=b"actual tiny HF squared logits; synthetic, not a quality corpus",
        cache_limits=dict(max_payload_bytes=65536,max_samples=32,max_windows=4),
        max_total_payload_bytes=1<<22)


@pytest.mark.parametrize("mode",["tied","untied"])
def test_real_hf_full_source_capture_candidate_and_replay(live,mode):
    source,native,options,run,work,root=live(mode)
    before={n:t.detach().clone() for n,t in source.model.state_dict().items()}
    calibration=_capture(source,native)
    assert len(calibration.caches)+len(native)==len(options)
    for name,cache in calibration.caches:
        binding=json.loads(cache.binding_json)
        assert binding["request"]["consumer"]=="Matmul" and binding["request"]["output"]=="None"
        assert binding["samples"]==4 and binding["windows"]==1
        assert name not in native
    selected=calibration.inspect(run,options,work_dir=work,max_numeric_bytes=1<<20,max_batch_samples=2)
    selected_three=calibration.inspect(run,options,work_dir=work,max_numeric_bytes=1<<20,max_batch_samples=3)
    assert selected["owners"]==selected_three["owners"]
    one=calibration.run(run,options,selected,output_dir=root/"candidate",work_dir=work,max_numeric_bytes=1<<20,max_batch_samples=2)
    two=calibration.run(run,options,selected_three,output_dir=root/"replay",work_dir=work,max_numeric_bytes=1<<20,max_batch_samples=3)
    assert not one["qualification"] and one==two
    assert (root/"candidate/model.trit").read_bytes()==(root/"replay/model.trit").read_bytes()
    assert not (root/"candidate/evidence.jsonl").exists()
    assert not list(work.iterdir())
    assert all(torch.equal(before[n],t) for n,t in source.model.state_dict().items())
    assert all(t.grad is None for t in source.model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in source.model.modules())
    source.verify()
    output=os.environ.get("TRITIUM_QTEA_LIVE_OUTPUT")
    if output:
        dest=Path(output)/mode;shutil.copytree(root/"candidate",dest)
        for name,path in json.loads(source.files_json)["assets"].items():shutil.copyfile(path,dest/name)
        (dest/"selected.json").write_text(json.dumps(selected))
        (dest/"source-id.txt").write_text(source.observation.source_id)


@pytest.mark.parametrize("variant",["functional","view","regularizer","raise","tokens"])
def test_hidden_owner_reads_or_objective_failure_produce_no_caches_and_remove_hooks(live,variant):
    source,native,_,_,work,_=live()
    def objective(model,window):
        result=_objective(model,window)
        weight=model.model.layers[0].self_attn.q_proj.weight
        if variant=="functional":return result+torch.nn.functional.linear(torch.ones(1,weight.shape[1]),weight).sum()
        if variant=="view":return result+weight[:1].sum()
        if variant=="regularizer":return result+weight.square().sum()
        if variant=="tokens":window["input_ids"][0,0]=7;return result
        raise RuntimeError("caller objective failed")
    with pytest.raises((ValueError,RuntimeError)):_capture(source,native,objective)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in source.model.modules())
    assert not list(work.iterdir())


def test_missing_embedding_metric_and_wrong_native_metric_fail_before_objective(live):
    source,native,_,_,_,_=live();calls=[]
    for supplied in [{}, {name:dict(item,spec=dict(item["spec"],digest=[99]*32)) for name,item in native.items()}]:
        with pytest.raises(ValueError):_capture(source,supplied,lambda m,w:calls.append(1))
    assert not calls


@pytest.mark.parametrize("variant",["bytes","dtype","graph","extra","alias","files"])
def test_live_state_or_files_cannot_be_rebound_after_preparation(live,variant):
    source,_,_,_,_,_=live();model=source.model
    if variant=="bytes":
        with torch.no_grad():model.model.norm.weight.add_(1)
    elif variant=="dtype":model.float().double()
    elif variant=="graph":model.model.layers[0].self_attn.q_proj.in_features+=1
    elif variant=="extra":model.register_buffer("extra",torch.ones(1))
    elif variant=="alias":model.lm_head.weight=nn.Parameter(model.lm_head.weight.detach().clone())
    else:Path(json.loads(source.files_json)["assets"]["tokenizer_config.json"]).write_bytes(b"{}")
    with pytest.raises(ValueError):source.verify()


def test_source_request_is_explicit_and_bounded_before_source_work(tmp_path):
    with pytest.raises(ValueError):prepare_qtea_source(nn.Linear(2,2),{},source_shards=[tmp_path/"absent"],asset_paths={},max_live_tensor_bytes=16)


@pytest.mark.parametrize("model,cap",[(nn.Linear(4,4),1),(nn.Linear(2,2,device="meta"),16)])
def test_incompatible_live_state_refuses_before_original_file_reads(tmp_path,model,cap):
    with pytest.raises(ValueError,match="CPU|byte cap"):
        prepare_qtea_source(model,{},source_shards=[tmp_path/"absent"],asset_paths={},max_live_tensor_bytes=cap)
