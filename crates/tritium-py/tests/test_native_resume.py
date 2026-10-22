"""Installed-wheel actual backward/save/restart/replay; not quality qualification."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tritium import NativeScaleResumeBundle
from tritium.torch import native_scale_training, load, convert_native_module
from test_native_module_source import StateModel, _prepared, _convert
from test_native_scale_training import plan, start
from test_native_refinement import recording, publish, update


def limits():
    return dict(max_bundle_bytes=2_000_000,max_manifest_bytes=100_000,
                max_checkpoint_bytes=1_000_000,max_working_bytes=2_000_000)


def reopen(bundle, **overrides):
    options=limits();options.update(overrides)
    return NativeScaleResumeBundle.open(bundle.output_dir,bundle_id=bundle.bundle_id,**options)


def recover(shell,parent,bundle,root,**overrides):
    options=dict(level=json.loads(bundle.plan_json)["parent"]["level"],load_run_id="resumed-scale-load",
                 max_snapshot_bytes=65536,max_parent_bytes=2_000_000,max_saved_input_bytes=1_000_000,
                 recording=recording(root),resume=bundle)
    options.update(overrides)
    return native_scale_training(shell,parent,**options)


@pytest.mark.parametrize("level",["compact","full"])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_actual_restart_keeps_checkpoint_journal_outputs_ties_and_original_parent(tmp_path,level,dtype):
    torch.manual_seed(67);parent=_convert(tmp_path,_prepared(StateModel(dtype)))
    original=(tmp_path/"output/model.trit").read_bytes();request=plan(parent,level=level)
    x=torch.linspace(-1,1,64,dtype=dtype).reshape(2,32)
    baseline=start(StateModel(dtype),parent,request,level=level,recording=recording(tmp_path))
    for i in range(4):update(baseline,i,x)
    with torch.no_grad():expected=baseline.model(x)
    expected_checkpoint=baseline.checkpoint();reference=publish(baseline,tmp_path/"reference")
    baseline.close()
    trainer=start(StateModel(dtype),parent,request,level=level,recording=recording(tmp_path))
    for i in range(2):update(trainer,i,x)
    checkpoint=trainer.checkpoint();work=Path(trainer._recording_work.name)
    saved=trainer.save_resume(tmp_path/"saved",limits=limits())
    assert isinstance(saved,NativeScaleResumeBundle) and saved.completed_updates==2
    assert saved.admission=="byte-consistent-unqualified" and not saved.quality_qualified and not saved.objective_authenticated
    assert saved.warnings==[] and len(saved.bundle_id)==64 and saved.bundle_bytes==(tmp_path/"saved/resume.tnsb").stat().st_size
    assert sorted(p.name for p in (tmp_path/"saved").iterdir())==["resume.tnsb"]
    with pytest.raises(AttributeError):saved.bundle_id="0"*64
    detached=json.loads(saved.plan_json);detached["owners"].clear();assert json.loads(saved.plan_json)["owners"]
    trainer.close();assert not work.exists();saved.verify();saved=reopen(saved)
    shell=StateModel(dtype);before={n:t.clone() for n,t in shell.state_dict().items()}
    resumed=recover(shell,parent,saved,tmp_path)
    assert resumed.revision==2 and resumed.checkpoint()==checkpoint and resumed.plan==request
    assert resumed.model.a.packed_weight is resumed.model.b.packed_weight
    for i in range(2,4):update(resumed,i,x)
    assert resumed.checkpoint()==expected_checkpoint
    with torch.no_grad():assert torch.equal(resumed.model(x),expected)
    child=publish(resumed,tmp_path/"resumed-child")
    child.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    for name in ("gradients.tnsj","checkpoint.tnsr"):
        assert (Path(child.output_dir)/name).read_bytes()==(Path(reference.output_dir)/name).read_bytes()
    loaded=load(child,model=StateModel(dtype),level="scale-child",load_run_id="recovered-child",inplace=False)
    with torch.no_grad():assert torch.equal(loaded(x),expected)
    resumed.close();assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]
    for name,value in shell.state_dict().items():assert torch.equal(value,before[name])
    assert (tmp_path/"output/model.trit").read_bytes()==original


@pytest.mark.parametrize("cursor",[0,4])
def test_zero_and_complete_cursors_can_save_reopen_continue_and_publish(tmp_path,cursor):
    parent=_convert(tmp_path,_prepared());x=torch.linspace(-1,1,64).reshape(2,32)
    trainer=start(StateModel(),parent,recording=recording(tmp_path))
    for i in range(cursor):update(trainer,i,x)
    checkpoint=trainer.checkpoint();saved=trainer.save_resume(tmp_path/"saved",limits=limits());trainer.close()
    resumed=recover(StateModel(),parent,reopen(saved),tmp_path)
    assert resumed.revision==cursor and resumed.checkpoint()==checkpoint
    for i in range(cursor,4):update(resumed,i,x)
    publish(resumed,tmp_path/"child").verify();resumed.close()


def test_save_is_committed_state_not_pending_gradients_and_does_not_seal_live_writer(tmp_path):
    parent=_convert(tmp_path,_prepared());trainer=start(StateModel(),parent,recording=recording(tmp_path))
    x=torch.linspace(-1,1,64).reshape(2,32);before=trainer.checkpoint()
    trainer.model(x).float().square().mean().backward()
    saved=trainer.save_resume(tmp_path/"before-step",limits=limits());assert saved.completed_updates==0
    trainer.step([10]*32);expected=trainer.checkpoint()
    resumed=recover(StateModel(),parent,reopen(saved),tmp_path);assert resumed.checkpoint()==before
    update(resumed,0,x);assert resumed.checkpoint()==expected
    trainer.close();resumed.close()


def test_wrong_resume_inputs_caps_and_overlap_leave_no_owned_work_or_fallback(tmp_path):
    parent=_convert(tmp_path,_prepared());trainer=start(StateModel(),parent,recording=recording(tmp_path))
    update(trainer,0,torch.ones(2,32));saved=trainer.save_resume(tmp_path/"saved",limits=limits());cp=trainer.checkpoint();trainer.close()
    work=Path(recording(tmp_path)["work_dir"])
    wrong=plan(parent);wrong["child_level"]="wrong"
    for extra in [dict(plan=wrong),dict(level="compact"),dict(max_parent_bytes=1),dict(max_snapshot_bytes=1),
                  dict(recording={**recording(tmp_path),"max_journal_bytes":1}),
                  dict(recording={**recording(tmp_path),"max_working_bytes":1}),
                  dict(checkpoint=cp),dict(recording=None)]:
        with pytest.raises((ValueError,TypeError)):recover(StateModel(),parent,saved,tmp_path,**extra)
        assert list(work.iterdir())==[]
    with pytest.raises(ValueError,match="overlap"):
        recover(StateModel(),parent,saved,tmp_path,recording={**recording(tmp_path),"work_dir":saved.output_dir})
    with pytest.raises(TypeError,match="NativeScaleResumeBundle"):
        recover(StateModel(),parent,saved,tmp_path,resume="not-a-bundle")
    other_root=tmp_path/"other";other_root.mkdir();other=_convert(other_root,_prepared())
    assert other.package_id!=parent.package_id
    with pytest.raises(ValueError):recover(StateModel(),other,saved,tmp_path)
    assert list(work.iterdir())==[]
    for field in limits():
        with pytest.raises(ValueError):reopen(saved,**{field:1})
    for bad in ["0"*64,"A"*64,"not-hex"]:
        with pytest.raises(ValueError):NativeScaleResumeBundle.open(saved.output_dir,bundle_id=bad,**limits())
    saved.verify()


def test_save_retries_caps_refuses_overwrite_closed_unrecorded_failed_and_sealed_states(tmp_path):
    parent=_convert(tmp_path,_prepared());plain=start(StateModel(),parent)
    with pytest.raises(RuntimeError,match="recorded"):plain.save_resume(tmp_path/"plain",limits=limits())
    plain.close();trainer=start(StateModel(),parent,recording=recording(tmp_path));before=trainer.checkpoint()
    for field in limits():
        with pytest.raises(ValueError):trainer.save_resume(tmp_path/field,limits={**limits(),field:1})
        assert not (tmp_path/field).exists() and trainer.checkpoint()==before
    for destination in [Path(trainer._recording_work.name)/"nested",Path(parent.output_dir)/"nested"]:
        with pytest.raises(ValueError,match="overlap"):trainer.save_resume(destination,limits=limits())
    x=torch.ones(2,32);loss=trainer.model(x).float().square().mean()
    with pytest.raises(RuntimeError,match="first-order"):loss.backward(create_graph=True)
    with pytest.raises(RuntimeError,match="backward failed"):trainer.save_resume(tmp_path/"failed",limits=limits())
    trainer.zero_grad();saved=trainer.save_resume(tmp_path/"saved",limits=limits());contents=(tmp_path/"saved/resume.tnsb").read_bytes()
    with pytest.raises(ValueError):trainer.save_resume(tmp_path/"saved",limits=limits())
    assert (tmp_path/"saved/resume.tnsb").read_bytes()==contents
    for i in range(4):update(trainer,i,x)
    publish(trainer,tmp_path/"child")
    with pytest.raises(ValueError,match="live recorded writer"):trainer.save_resume(tmp_path/"sealed",limits=limits())
    trainer.close()
    with pytest.raises(RuntimeError,match="closed"):trainer.save_resume(tmp_path/"closed",limits=limits())
    saved.verify()


def test_modified_bundle_rejected_by_old_descriptor_and_recovery_cleans_new_work(tmp_path):
    parent=_convert(tmp_path,_prepared());trainer=start(StateModel(),parent,recording=recording(tmp_path))
    saved=trainer.save_resume(tmp_path/"saved",limits=limits());trainer.close()
    path=tmp_path/"saved/resume.tnsb";original=path.read_bytes();changed=bytearray(original);changed[len(changed)//2]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError):saved.verify()
    with pytest.raises(ValueError):reopen(saved)
    with pytest.raises(ValueError):recover(StateModel(),parent,saved,tmp_path)
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]
    path.write_bytes(original);saved.verify()


def test_real_separate_process_restart_uses_only_installed_wheel_parent_and_bundle(tmp_path):
    parent=_convert(tmp_path,_prepared());trainer=start(StateModel(),parent,recording=recording(tmp_path))
    x=torch.linspace(-1,1,64).reshape(2,32)
    for i in range(2):update(trainer,i,x)
    saved=trainer.save_resume(tmp_path/"saved",limits=limits());trainer.close()
    child_code='''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,tritium
from test_native_module_source import StateModel
from test_native_resume import limits,recover
from test_native_refinement import update,publish
assert tritium._tritium.source_identity()==sys.argv[4]
root=Path(sys.argv[2]);parent=tritium.NativeModelConversion.open(str(root/"output"))
saved=tritium.NativeScaleResumeBundle.open(str(root/"saved"),bundle_id=sys.argv[3],**limits())
trainer=recover(StateModel(),parent,saved,root)
assert trainer.revision==2
x=torch.linspace(-1,1,64).reshape(2,32)
for i in range(2,4):update(trainer,i,x)
result=publish(trainer,root/"process-child");result.verify_with_parent(parent,max_parent_bytes=4000000,max_replay_working_bytes=4000000)
trainer.close();print(json.dumps(dict(cursor=4,scope="parent-backed-numerical-replay",quality_qualified=result.quality_qualified)))
'''
    import tritium
    completed=subprocess.run([sys.executable,"-I","-B","-c",child_code,str(Path(__file__).parent),str(tmp_path),saved.bundle_id,
                              tritium._tritium.source_identity()],capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert json.loads(completed.stdout)["cursor"]==4 and not json.loads(completed.stdout)["quality_qualified"]
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir())==[]


def test_tiny_hf_multiple_owners_tied_head_resumes_real_backward_and_token_outputs(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    torch.manual_seed(71);source=_hf_source(tmp_path)
    parent=convert_native_module(source.prepared,tmp_path/"hf-parent","hf-parent-fit",**json.loads(source.options_json))
    catalog=json.loads(parent.asset("tritium-torch-state.json"));owners=[e["owner"] for e in catalog["state"] if e["disposition"]=="fitted"]
    request=plan(parent,owners);tokens=torch.tensor([[1,2,3]]);mask=torch.ones_like(tokens)
    def step(t,i):
        t.zero_grad();t.model(tokens,labels=tokens,use_cache=False).loss.backward();t.step([10+i]*32)
    baseline=start(LlamaForCausalLM(source.prepared.model.config),parent,request,recording=recording(tmp_path),max_snapshot_bytes=1_000_000,max_saved_input_bytes=10_000_000)
    for i in range(4):step(baseline,i)
    checkpoint=baseline.checkpoint();baseline.model.eval()
    with torch.no_grad():expected=baseline.model.generate(tokens,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0)
    reference=publish(baseline,tmp_path/"hf-reference");baseline.close()
    trainer=start(LlamaForCausalLM(source.prepared.model.config),parent,request,recording=recording(tmp_path),max_snapshot_bytes=1_000_000,max_saved_input_bytes=10_000_000)
    for i in range(2):step(trainer,i)
    saved=trainer.save_resume(tmp_path/"hf-saved",limits=limits());trainer.close()
    resumed=recover(LlamaForCausalLM(source.prepared.model.config),parent,reopen(saved),tmp_path,max_snapshot_bytes=1_000_000,max_parent_bytes=4_000_000,max_saved_input_bytes=10_000_000)
    assert resumed.model.model.embed_tokens.packed_weight is resumed.model.lm_head.packed_weight
    for i in range(2,4):step(resumed,i)
    assert resumed.checkpoint()==checkpoint
    result=publish(resumed,tmp_path/"hf-resumed");resumed.close()
    for name in ("checkpoint.tnsr","gradients.tnsj"):
        assert (Path(result.output_dir)/name).read_bytes()==(Path(reference.output_dir)/name).read_bytes()
    loaded=load(result,model=LlamaForCausalLM(source.prepared.model.config),level="scale-child",load_run_id="hf-recovered",inplace=False)
    with torch.no_grad():assert torch.equal(loaded.generate(tokens,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0),expected)
