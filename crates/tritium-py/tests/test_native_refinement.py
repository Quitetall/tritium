"""Installed Rust-backed record/train/publish/reopen, not quality/release acceptance."""
import json
from pathlib import Path
import shutil
import pytest
import torch
from tritium import NativeModelConversion, NativeModelRefinement
from tritium.torch import load, export, load_native_module, convert_native_module
from test_native_module_source import StateModel, _prepared, _convert, _fit
from test_native_scale_training import plan, start


def limits():
    return dict(max_spool_bytes=2_000_000,max_working_bytes=2_000_000,max_evidence_bytes=2_000_000,
                max_companion_bytes=2_000_000,max_package_bytes=2_000_000,max_manifest_bytes=1_000_000,
                max_directory_bytes=1_000_000,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000,
                max_journal_bytes=2_000_000)


def recording(tmp_path):
    work=tmp_path/"recording";work.mkdir(exist_ok=True)
    return dict(work_dir=work,max_journal_bytes=2_000_000,max_working_bytes=2_000_000)


def publish(trainer,directory,**kwargs):
    options=dict(machine="synthetic-installed-wheel",validation=[dict(content=[30]*32,parent_loss=2.,child_loss=3.)],limits=limits())
    options.update(kwargs)
    return trainer.publish(directory,"native-python-child",**options)


def update(trainer,i,inputs):
    trainer.zero_grad();trainer.model(inputs).float().square().mean().backward();trainer.step([10+i]*32)


@pytest.mark.parametrize("level",["compact","full"])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_record_publish_export_standalone_load_keeps_exact_outputs_aliases_and_state(tmp_path,level,dtype):
    torch.manual_seed(31);parent=_convert(tmp_path,_prepared(StateModel(dtype)));before=(tmp_path/"output/model.trit").read_bytes()
    shell=StateModel(dtype);source_state={n:t.clone() for n,t in shell.state_dict().items()};record=recording(tmp_path)
    trainer=start(shell,parent,plan(parent,level=level),level=level,recording=record)
    inputs=torch.linspace(-1,1,64,dtype=dtype).reshape(2,32)
    for i in range(4):update(trainer,i,inputs)
    with torch.no_grad():expected=trainer.model(inputs)
    result=publish(trainer,tmp_path/"child");assert isinstance(result,NativeModelRefinement) and not isinstance(result,NativeModelConversion)
    assert result.admission=="record-consistent-unqualified" and not result.quality_qualified and not result.objective_authenticated
    assert result.parent_level==level and result.parent_package_id==parent.package_id and result.package_id!=parent.package_id
    assert (tmp_path/"output/model.trit").read_bytes()==before
    for n,t in shell.state_dict().items():assert torch.equal(t,source_state[n])
    assert sorted(p.name for p in (tmp_path/"child").iterdir())==["checkpoint.tnsr","evidence.jsonl","gradients.tnsj","model.trit"]
    audit=json.loads(result.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000))
    assert audit["scope"]=="parent-backed-numerical-replay" and not audit["quality_qualified"]
    loaded=load(result,model=StateModel(dtype),level="scale-child",load_run_id="child-load",inplace=False)
    with torch.no_grad():assert torch.equal(loaded(inputs),expected)
    assert loaded.a.packed_weight is loaded.b.packed_weight and loaded.a.bias is loaded.b.bias
    copied=export(result,tmp_path/"exported");assert copied.package_id==result.package_id and copied.evidence_root==result.evidence_root
    trainer.close();assert list(Path(record["work_dir"]).iterdir())==[]
    shutil.rmtree(tmp_path/"child");shutil.rmtree(tmp_path/"output")
    reopened=NativeModelRefinement.open(str(tmp_path/"exported"));reopened.verify()
    with pytest.raises(ValueError):NativeModelConversion.open(str(tmp_path/"exported"))
    standalone=load_native_module(StateModel(dtype),reopened,level="scale-child",load_run_id="standalone",inplace=False)
    with torch.no_grad():assert torch.equal(standalone(inputs),expected)
    by_directory=load(tmp_path/"exported",model=StateModel(dtype),level="scale-child",load_run_id="directory",inplace=False)
    with torch.no_grad():assert torch.equal(by_directory(inputs),expected)
    assert not reopened.quality_qualified


def test_failed_publication_retries_same_sealed_journal_without_overwriting(tmp_path):
    parent=_convert(tmp_path,_prepared());record=recording(tmp_path);trainer=start(StateModel(),parent,recording=record)
    x=torch.linspace(-1,1,64).reshape(2,32)
    with pytest.raises(ValueError,match="incomplete"):publish(trainer,tmp_path/"early")
    for i in range(4):update(trainer,i,x)
    before=trainer.checkpoint();bad=limits();bad["max_spool_bytes"]=1
    with pytest.raises(ValueError,match="overlap"):publish(trainer,Path(trainer._recording_work.name)/"child")
    with pytest.raises(ValueError,match="overlap"):publish(trainer,Path(parent.output_dir)/"child")
    with pytest.raises(ValueError,match="limit"):publish(trainer,tmp_path/"small",limits=bad)
    assert trainer.checkpoint()==before and not (tmp_path/"small").exists()
    result=publish(trainer,tmp_path/"child");files={p.name:p.read_bytes() for p in (tmp_path/"child").iterdir()}
    with pytest.raises(ValueError,match="overlap"):result.export(str(tmp_path/"child"/"nested"))
    with pytest.raises(ValueError):publish(trainer,tmp_path/"child")
    assert {p.name:p.read_bytes() for p in (tmp_path/"child").iterdir()}==files
    assert not list(Path(trainer._recording_work.name).glob("publication-*.spool"));result.verify();trainer.close()


@pytest.mark.parametrize("name",["model.trit","evidence.jsonl","checkpoint.tnsr","gradients.tnsj"])
def test_result_rejects_changed_files_and_tight_companion_limits(tmp_path,name):
    parent=_convert(tmp_path,_prepared());trainer=start(StateModel(),parent,recording=recording(tmp_path));x=torch.ones(2,32)
    for i in range(4):update(trainer,i,x)
    result=publish(trainer,tmp_path/"child");trainer.close()
    with pytest.raises(ValueError):NativeModelRefinement.open(str(tmp_path/"child"),max_journal_bytes=1)
    with pytest.raises(ValueError):NativeModelRefinement.open(str(tmp_path/"child"),max_checkpoint_bytes=1)
    path=tmp_path/"child"/name;data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
    with pytest.raises(ValueError):result.verify()
    with pytest.raises(ValueError):result.export(str(tmp_path/"bad-export"))
    assert not (tmp_path/"bad-export").exists()


def test_recording_admission_and_cleanup_preserve_unrecorded_defaults(tmp_path):
    parent=_convert(tmp_path,_prepared());record=recording(tmp_path)
    with pytest.raises(ValueError,match="limit"):start(StateModel(),parent,recording={**record,"max_journal_bytes":1})
    assert list(Path(record["work_dir"]).iterdir())==[]
    plain=start(StateModel(),parent)
    with pytest.raises(RuntimeError,match="recorded"):publish(plain,tmp_path/"none")
    with pytest.raises(ValueError,match="zero origin"):start(StateModel(),parent,recording=record,checkpoint=plain.checkpoint())
    plain.close()
    trainer=start(StateModel(),parent,recording=record);work=Path(trainer._recording_work.name);assert work.is_dir()
    with pytest.raises(ValueError):publish(trainer,work/"must-not-delete-child")
    trainer.close();assert not work.exists()


def test_tiny_hf_multiple_owners_tied_head_real_backward_and_generated_token_parity(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    torch.manual_seed(43);source=_hf_source(tmp_path)
    parent=convert_native_module(source.prepared,tmp_path/"hf-parent","hf-parent-fit",**json.loads(source.options_json))
    catalog=json.loads(parent.asset("tritium-torch-state.json"));owners=[e["owner"] for e in catalog["state"] if e["disposition"]=="fitted"]
    assert len(owners)>1
    trainer=start(LlamaForCausalLM(source.prepared.model.config),parent,plan(parent,owners),recording=recording(tmp_path),max_snapshot_bytes=1_000_000,max_saved_input_bytes=10_000_000)
    tokens=torch.tensor([[1,2,3]]);mask=torch.ones_like(tokens)
    for i in range(4):
        trainer.zero_grad();trainer.model(tokens,labels=tokens,use_cache=False).loss.backward();trainer.step([10+i]*32)
    trainer.model.eval()
    with torch.no_grad():expected=trainer.model(tokens,use_cache=False).logits;expected_tokens=trainer.model.generate(tokens,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0)
    result=publish(trainer,tmp_path/"hf-child")
    result.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    loaded=load(result,model=LlamaForCausalLM(source.prepared.model.config),level="scale-child",load_run_id="hf-child-load",inplace=False)
    assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    with torch.no_grad():
        assert torch.equal(loaded(tokens,use_cache=False).logits,expected)
        assert torch.equal(loaded.generate(tokens,attention_mask=mask,max_new_tokens=3,do_sample=False,pad_token_id=0),expected_tokens)
    assert result.asset("tritium-torch-state.json")==parent.asset("tritium-torch-state.json")
    trainer.close()
