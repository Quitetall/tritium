"""Actual immutable native rounds; only immediate-edge numerical scope is admitted."""
import json
from pathlib import Path
import shutil
import pytest
import torch
from tritium import NativeModelConversion,NativeModelRefinement
from tritium.torch import load,native_scale_training,refine
from test_native_module_source import StateModel,_prepared,_convert
from test_native_scale_training import plan,start
from test_native_refinement import recording,publish,update
from test_native_resume import limits as bundle_limits,reopen,recover


def first(root,dtype=torch.float32,level="full",seed=83):
    torch.manual_seed(seed);source=_prepared(StateModel(dtype));parent=_convert(root,source)
    x=torch.linspace(-1,1,64,dtype=dtype).reshape(2,32)
    trainer=start(StateModel(dtype),parent,plan(parent,level=level),level=level,recording=recording(root))
    for i in range(4):update(trainer,i,x)
    child=publish(trainer,root/"first");trainer.close()
    return parent,child,x,source.model


def files(result):
    return {p.name:p.read_bytes() for p in Path(result.output_dir).iterdir()}


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize("level",["compact","full"])
def test_three_actual_rounds_bind_exact_immediate_parent_and_preserve_ancestors(tmp_path,dtype,level):
    original,current,x,_=first(tmp_path,dtype,level)
    root_bytes=files(original);first_id=current.package_id
    for round in (2,3):
        before=files(current);binding=json.loads(current.native_refinement_parent("scale-child"))
        assert bytes(binding["package"]).hex()==current.package_id
        assert bytes(binding["source"]).hex()==original.source_id
        assert bytes(binding["evidence"]["root"]).hex()==current.evidence_root
        request=plan(current,level="scale-child")
        # Same level label in distinct immutable packages is deliberate.
        trainer=start(StateModel(dtype),current,request,level="scale-child",recording=recording(tmp_path))
        assert trainer.revision==0
        old=load(current,model=StateModel(dtype),level="scale-child",load_run_id=f"parent-{round}",inplace=False)
        with torch.no_grad():assert torch.equal(trainer.model(x),old(x))
        with pytest.raises(ValueError):
            start(StateModel(dtype),current,request,level="scale-child",checkpoint=(Path(current.output_dir)/"checkpoint.tnsr").read_bytes())
        for i in range(4):update(trainer,i,x)
        with torch.no_grad():expected=trainer.model(x)
        child=publish(trainer,tmp_path/f"round-{round}");trainer.close()
        assert isinstance(child,NativeModelRefinement) and not isinstance(child,NativeModelConversion)
        assert child.parent_package_id==current.package_id and child.parent_level=="scale-child"
        assert child.package_id not in {current.package_id,original.package_id,first_id}
        audit=json.loads(child.verify_with_parent(current,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000))
        assert audit["scope"]=="parent-backed-numerical-replay" and audit["lineage_scope"]=="immediate-parent-only"
        assert audit["parent_admission"]=="refined-record-consistent-unqualified" and not audit["quality_qualified"]
        with pytest.raises(ValueError):child.verify_with_parent(original,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
        with pytest.raises(TypeError):child.verify_with_parent(dict(output_dir=current.output_dir),max_parent_bytes=1,max_replay_working_bytes=1)
        loaded=load(child,model=StateModel(dtype),level="scale-child",load_run_id=f"child-{round}",inplace=False)
        with torch.no_grad():assert torch.equal(loaded(x),expected)
        assert loaded.a.packed_weight is loaded.b.packed_weight and loaded.mask is loaded.mask_alias
        assert child.asset("tritium-torch-state.json")==current.asset("tritium-torch-state.json")
        assert child.asset("tritium-torch-preserved.safetensors")==current.asset("tritium-torch-preserved.safetensors")
        assert files(current)==before and files(original)==root_bytes
        with pytest.raises(ValueError):NativeModelConversion.open(child.output_dir)
        assert not child.objective_authenticated and not child.quality_qualified
        current=child
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]


def test_refined_parent_snapshot_resume_and_export_work_after_older_model_removal(tmp_path):
    original,parent,x,_=first(tmp_path);shutil.rmtree(original.output_dir)
    parent=NativeModelRefinement.open(parent.output_dir)
    request=plan(parent,level="scale-child");before=files(parent)
    trainer=start(StateModel(),parent,request,level="scale-child",recording=recording(tmp_path))
    for i in range(2):update(trainer,i,x)
    saved=trainer.save_resume(tmp_path/"saved",limits=bundle_limits());checkpoint=trainer.checkpoint();trainer.close()
    resumed=recover(StateModel(),parent,reopen(saved),tmp_path)
    assert resumed.revision==2 and resumed.checkpoint()==checkpoint
    for i in range(2,4):update(resumed,i,x)
    child=publish(resumed,tmp_path/"grandchild");resumed.close()
    child.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    assert files(parent)==before
    exported=child.export(str(tmp_path/"exported"));assert exported.package_id==child.package_id
    shutil.rmtree(parent.output_dir);shutil.rmtree(child.output_dir)
    exported=NativeModelRefinement.open(exported.output_dir)
    request=plan(exported,level="scale-child");request["child_level"]="new-round"
    trainer=start(StateModel(),exported,request,level="scale-child",recording=recording(tmp_path))
    assert trainer.revision==0
    for i in range(4):update(trainer,i,x)
    result=publish(trainer,tmp_path/"fourth");trainer.close()
    result.verify_with_parent(exported,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    assert result.parent_package_id==exported.package_id and list(result.model_ids)==["new-round"]


@pytest.mark.parametrize("name",["checkpoint.tnsr","gradients.tnsj","evidence.jsonl"])
def test_changed_parent_companion_or_record_never_starts_chained_training(tmp_path,name):
    _,parent,_,_=first(tmp_path);request=plan(parent,level="scale-child")
    path=Path(parent.output_dir)/name;raw=path.read_bytes();changed=bytearray(raw);changed[len(changed)//2]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError):parent.native_refinement_parent("scale-child")
    with pytest.raises(ValueError):start(StateModel(),parent,request,level="scale-child",recording=recording(tmp_path))
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]
    path.write_bytes(raw);parent.verify()


def test_wrong_sibling_parent_and_wrong_level_cannot_rebind_same_shape(tmp_path):
    _,parent,x,_=first(tmp_path);request=plan(parent,level="scale-child")
    with pytest.raises(ValueError):parent.native_refinement_parent("full")
    sibling_root=tmp_path/"sibling";sibling_root.mkdir();_,sibling,_,_=first(sibling_root,seed=89)
    assert sibling.package_id!=parent.package_id
    with pytest.raises(ValueError):start(StateModel(),sibling,request,level="scale-child",recording=recording(tmp_path))
    trainer=start(StateModel(),parent,request,level="scale-child",recording=recording(tmp_path))
    for i in range(4):update(trainer,i,x)
    child=publish(trainer,tmp_path/"child");trainer.close()
    with pytest.raises(ValueError):child.verify_with_parent(sibling,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]


def test_public_kd_refinement_uses_immediate_refined_parent_as_predecessor(tmp_path):
    from test_native_refine import fixture,configuration
    teacher,original,training,validation,work=fixture(tmp_path)
    first_cfg=configuration(original,training,validation)
    parent=refine(original,teacher=teacher,training=training,validation=validation,config=first_cfg,work_dir=work,output_dir=tmp_path/"first")
    previous=files(parent);shutil.rmtree(original.output_dir)
    cfg=configuration(parent,training,validation,level="scale-child")
    cfg_plan=cfg.plan;cfg_plan["child_level"]="chained-kd"
    from tritium.torch import NativeRefinementConfig
    cfg=NativeRefinementConfig(cfg_plan,level=cfg.level,run_id=cfg.run_id,machine=cfg.machine,
        training_limits=cfg.resources["training"],publication_limits=cfg.resources["publication"],data_limits=cfg.resources["data"])
    child=refine(parent,teacher=teacher,training=iter(training),validation=iter(validation),config=cfg,work_dir=work,output_dir=tmp_path/"second")
    child.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    assert child.parent_package_id==parent.package_id and child.parent_level=="scale-child" and files(parent)==previous
    assert not child.quality_qualified and not child.objective_authenticated and list(work.iterdir())==[]


def test_tiny_hf_chained_child_keeps_tied_head_and_standalone_outputs(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    from tritium.torch import convert_native_module
    source=_hf_source(tmp_path);parent=convert_native_module(source.prepared,tmp_path/"hf-parent","hf-chain",**json.loads(source.options_json))
    owners=[v["owner"] for v in json.loads(parent.asset("tritium-torch-state.json"))["state"] if v["disposition"]=="fitted"]
    tokens=torch.tensor([[1,2,3]])
    for round in (1,2):
        level="full" if round==1 else "scale-child"
        trainer=start(LlamaForCausalLM(source.prepared.model.config),parent,plan(parent,owners,level=level),level=level,
            recording=recording(tmp_path),max_snapshot_bytes=1_000_000,max_saved_input_bytes=10_000_000,max_parent_bytes=4_000_000)
        assert trainer.revision==0 and trainer.model.model.embed_tokens.packed_weight is trainer.model.lm_head.packed_weight
        for i in range(4):
            trainer.zero_grad();trainer.model(tokens,labels=tokens,use_cache=False).loss.backward();trainer.step([10+i]*32)
        with torch.no_grad():expected=trainer.model(tokens,use_cache=False).logits
        child=publish(trainer,tmp_path/f"hf-{round}");trainer.close()
        child.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
        loaded=load(child,model=LlamaForCausalLM(source.prepared.model.config),level="scale-child",load_run_id=f"hf-loaded-{round}",inplace=False)
        with torch.no_grad():assert torch.equal(loaded(tokens,use_cache=False).logits,expected)
        parent=child
