"""Public recorded KD orchestration fixtures; not model-quality qualification."""
from pathlib import Path
import json
import pytest
import torch
from tritium import NativeModelRefinement
from tritium.torch import NativeRefinementConfig as Config, NativeScaleTrainer, refine, load
from tritium.torch.errors import TritiumError
from tritium.torch import native_refinement as native
from test_native_module_source import StateModel, _prepared, _convert
from test_native_scale_training import plan as base_plan
from test_native_refinement import limits as publish_limits
from test_native_resume import limits as bundle_limits


def data_limits():
    return dict(max_window_bytes=100_000,max_validation_bytes=200_000,max_nodes=64,max_depth=8,max_loss_bytes=2_000_000)


def training_limits():
    return dict(max_snapshot_bytes=2_000_000,max_parent_bytes=4_000_000,max_saved_input_bytes=10_000_000,
                max_package_bytes=2_000_000,max_payload_bytes=2_000_000,max_workspace_bytes=2_000_000,
                max_output_bytes=2_000_000,max_evidence_bytes=2_000_000)


def configuration(parent,training,validation,*,level="full",owners="a.weight",lr=0.00001,steps=(),**overrides):
    request=base_plan(parent,owners,updates=len(training),level=level)
    request["teacher_source"]=request["parent"]["source"]
    request["objective"]=Config.objective_id()
    request["recipe"]["ScaleOnly"]["optimizer"]["lr"]=lr
    for declared,batch in zip(request["training"],training):declared["content"]=Config.window_id(batch,limits=data_limits())
    request["validation"]=[dict(content=Config.window_id(batch,limits=data_limits()),weight=float(i+1)) for i,batch in enumerate(validation)]
    options=dict(level=level,run_id="public-native-refine",machine="synthetic-installed-wheel",
                 training_limits=training_limits(),publication_limits=publish_limits(),data_limits=data_limits(),
                 checkpoint_steps=steps,checkpoint_limits=bundle_limits() if steps else None)
    options.update(overrides)
    return Config(request,**options)


def fixture(root,dtype=torch.float32):
    torch.manual_seed(79);prepared=_prepared(StateModel(dtype));parent=_convert(root,prepared)
    x=torch.linspace(-1,1,64,dtype=dtype).reshape(2,32)
    training=[x*(1+i/8) for i in range(4)];validation=[x+0.125,x-0.125]
    work=root/"runtime";work.mkdir()
    return prepared.model,parent,training,validation,work


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize("level",["compact","full"])
def test_public_native_refine_records_actual_backward_and_preserves_source(tmp_path,dtype,level):
    teacher,parent,training,validation,work=fixture(tmp_path,dtype)
    teacher.a.eval();flags=[m.training for m in teacher.modules()]
    state={n:t.clone() for n,t in teacher.state_dict().items()};original=(tmp_path/"output/model.trit").read_bytes()
    cfg=configuration(parent,training,validation,level=level)
    metadata=cfg.plan;metadata["owners"].clear();assert cfg.plan["owners"]
    resources=cfg.resources;resources["training"]["max_parent_bytes"]=1;assert cfg.resources["training"]["max_parent_bytes"]>1
    result=refine(parent,teacher=teacher,training=iter(training),validation=iter(validation),config=cfg,
                  work_dir=work,output_dir=tmp_path/"child")
    assert isinstance(result,NativeModelRefinement) and result.parent_level==level
    assert result.admission=="record-consistent-unqualified" and not result.quality_qualified and not result.objective_authenticated
    result.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    loaded=load(result,model=StateModel(dtype),level="scale-child",load_run_id="public-refine-load",inplace=False)
    with torch.no_grad():assert torch.isfinite(loaded(validation[0])).all()
    assert loaded.a.packed_weight is loaded.b.packed_weight
    assert [m.training for m in teacher.modules()]==flags
    assert all(torch.equal(t,state[n]) for n,t in teacher.state_dict().items())
    assert (tmp_path/"output/model.trit").read_bytes()==original and list(work.iterdir())==[]


def test_checkpoint_interruption_and_streamed_resume_match_uninterrupted_journal(tmp_path):
    teacher,parent,training,validation,work=fixture(tmp_path)
    cfg=configuration(parent,training,validation,steps=(2,4));checkpoints=tmp_path/"checkpoints";checkpoints.mkdir()
    saved=[];events=[]
    def stream():
        for i,batch in enumerate(training):events.append(("data",i));yield batch
    def interrupted(bundle):
        saved.append(bundle);events.append(("checkpoint",bundle.completed_updates));raise RuntimeError("intentional interrupt")
    with pytest.raises(RuntimeError,match="intentional interrupt"):
        refine(parent,teacher=teacher,training=stream(),validation=validation,config=cfg,work_dir=work,
               output_dir=tmp_path/"resumed",checkpoint_dir=checkpoints,on_checkpoint=interrupted)
    assert events==[("data",0),("data",1),("checkpoint",2)] and not (tmp_path/"resumed").exists()
    assert list(work.iterdir())==[];saved[0].verify()
    observed=[]
    result=refine(parent,teacher=teacher,training=iter(training),validation=validation,config=cfg,work_dir=work,
                  output_dir=tmp_path/"resumed",resume=saved[0],checkpoint_dir=checkpoints,on_checkpoint=lambda b:observed.append(b.completed_updates))
    assert observed==[4]
    reference=refine(parent,teacher=teacher,training=training,validation=validation,
                     config=configuration(parent,training,validation),work_dir=work,output_dir=tmp_path/"reference")
    for name in ("checkpoint.tnsr","gradients.tnsj"):
        assert (Path(result.output_dir)/name).read_bytes()==(Path(reference.output_dir)/name).read_bytes()
    assert list(work.iterdir())==[]


def test_typed_window_ids_are_unambiguous_bounded_and_strided_bit_exact():
    limits={**data_limits(),"max_window_bytes":8_000_000}
    tensor=torch.arange(1024*513,dtype=torch.float32).reshape(1024,513).t()
    from test_ptq_hashing import BoundedClone
    with BoundedClone():identity=Config.window_id(tensor,limits=limits)
    assert identity==Config.window_id(tensor.contiguous(),limits=limits)
    values=[[],(),{},None,False,0,0.0,{"a.b":torch.tensor(1)},{"a":{"b":torch.tensor(1)}}]
    assert len({tuple(Config.window_id(v,limits=data_limits())) for v in values})==len(values)
    cycle=[];cycle.append(cycle)
    for value,changed in [(cycle,{}),([[[[1]]]],{"max_depth":2}),([1,2,3],{"max_nodes":2}),
                          (torch.ones(32),{"max_window_bytes":1})]:
        with pytest.raises(ValueError):Config.window_id(value,limits={**data_limits(),**changed})


@pytest.mark.parametrize("cursor",[0,4])
def test_resume_at_scheduled_zero_or_final_cursor_does_not_overwrite_checkpoint(tmp_path,cursor):
    teacher,parent,training,validation,work=fixture(tmp_path);cfg=configuration(parent,training,validation,steps=(cursor,))
    checkpoints=tmp_path/"checkpoints";checkpoints.mkdir();saved=[]
    def interrupt(bundle):saved.append(bundle);raise RuntimeError("stop at cursor")
    with pytest.raises(RuntimeError,match="stop at cursor"):
        refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,
               output_dir=tmp_path/"child",checkpoint_dir=checkpoints,on_checkpoint=interrupt)
    original=(Path(saved[0].output_dir)/"resume.tnsb").read_bytes();calls=[]
    result=refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,
                  output_dir=tmp_path/"child",checkpoint_dir=checkpoints,resume=saved[0],on_checkpoint=calls.append)
    result.verify();assert calls==[]
    assert (Path(saved[0].output_dir)/"resume.tnsb").read_bytes()==original


def test_masked_kd_matches_independent_formula_and_zeros_excluded_gradients(tmp_path):
    teacher,parent,training,validation,_=fixture(tmp_path);cfg=configuration(parent,training,validation)
    logits=torch.tensor([[[.2,-.4,.7],[.4,.9,-.2],[.1,-.8,.5]]],requires_grad=True)
    target=torch.tensor([[[.7,.1,-.2],[.1,.4,.2],[.4,.8,-.1]]])
    batch=dict(attention_mask=torch.tensor([[1,0,1]]),labels=torch.tensor([[1,2,-100]]))
    loss=native._loss(logits,target,batch,cfg)
    logp=logits.log_softmax(-1);logq=target.log_softmax(-1)
    expected=(logq.exp()*(logq-logp)).sum(-1).clamp_min(0)[0,0]
    torch.testing.assert_close(loss,expected);loss.backward()
    assert torch.count_nonzero(logits.grad[0,0])>0 and torch.count_nonzero(logits.grad[0,1:])==0
    for mask in [torch.tensor([[0,0,0]]),torch.tensor([[1,2,1]]),torch.ones(3)]:
        with pytest.raises(ValueError):native._loss(logits.detach(),target,dict(attention_mask=mask),cfg)
    bad=configuration(parent,training,validation,data_limits={**data_limits(),"max_loss_bytes":1})
    with pytest.raises(ValueError,match="workspace"):native._loss(logits,target,{},bad)


@pytest.mark.parametrize("kind",["missing-train","extra-train","wrong-train","missing-validation","extra-validation","wrong-validation","validation-cap","window-cap"])
def test_data_refusals_publish_nothing_and_clean_only_owned_work(tmp_path,kind):
    teacher,parent,training,validation,work=fixture(tmp_path);cfg=configuration(parent,training,validation)
    if kind=="missing-train":training=training[:-1]
    elif kind=="extra-train":training=training+[training[0]]
    elif kind=="wrong-train":training=[torch.zeros_like(training[0]),*training[1:]]
    elif kind=="missing-validation":validation=validation[:-1]
    elif kind=="extra-validation":validation=validation+[validation[0]]
    elif kind=="wrong-validation":validation=[torch.zeros_like(validation[0]),*validation[1:]]
    elif kind=="validation-cap":cfg=configuration(parent,training,validation,data_limits={**data_limits(),"max_validation_bytes":1})
    else:cfg=configuration(parent,training,validation,data_limits={**data_limits(),"max_window_bytes":1})
    with pytest.raises(ValueError):refine(parent,teacher=teacher,training=iter(training),validation=iter(validation),config=cfg,work_dir=work,output_dir=tmp_path/"child")
    assert not (tmp_path/"child").exists() and list(work.iterdir())==[]


def test_regressed_terminal_candidate_is_not_published_or_replaced_by_another_plan(tmp_path):
    teacher,parent,training,validation,work=fixture(tmp_path);training=training[:1]
    cfg=configuration(parent,training,validation,lr=1000,steps=(1,));checkpoints=tmp_path/"checkpoints";checkpoints.mkdir()
    saved=[]
    with pytest.raises(TritiumError,match="predecessor retention"):
        refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,
               output_dir=tmp_path/"child",checkpoint_dir=checkpoints,on_checkpoint=saved.append)
    assert not (tmp_path/"child").exists() and list(work.iterdir())==[]
    assert saved[0].completed_updates==1 and not saved[0].quality_qualified;saved[0].verify()


def test_teacher_and_input_mutation_are_detected_before_publication(tmp_path):
    teacher,parent,training,validation,work=fixture(tmp_path);cfg=configuration(parent,training,validation)
    with torch.no_grad():teacher.a.weight.add_(1)
    with pytest.raises(ValueError,match="teacher differs"):
        refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=tmp_path/"wrong")
    assert list(work.iterdir())==[]
    root=tmp_path/"second";root.mkdir();teacher,parent,training,validation,work=fixture(root);cfg=configuration(parent,training,validation)
    before=[batch.clone() for batch in validation]
    hook=teacher.register_forward_pre_hook(lambda model,args:args[0].add_(0.25))
    try:
        with pytest.raises(ValueError,match="window mutated"):
            refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=root/"mutated")
    finally:hook.remove()
    assert all(torch.equal(a,b) for a,b in zip(validation,before)) and list(work.iterdir())==[]


def test_postcommit_cleanup_is_warning_and_precommit_error_is_not_masked(tmp_path,monkeypatch):
    teacher,parent,training,validation,work=fixture(tmp_path);cfg=configuration(parent,training,validation)
    close=NativeScaleTrainer.close
    def faulty(self):close(self);raise OSError("injected owned cleanup")
    monkeypatch.setattr(NativeScaleTrainer,"close",faulty)
    result=refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=tmp_path/"child")
    assert any("NativeRunCleanup" in warning for warning in result.publication_warnings);result.verify()
    wrong=[torch.zeros_like(training[0]),*training[1:]]
    with pytest.raises(ValueError,match="window identity"):
        refine(parent,teacher=teacher,training=wrong,validation=validation,config=cfg,work_dir=work,output_dir=tmp_path/"bad")
    assert list(work.iterdir())==[]


def test_config_and_output_refusals_are_early_and_native_children_are_not_legacy(tmp_path):
    teacher,parent,training,validation,work=fixture(tmp_path);cfg=configuration(parent,training,validation)
    output=tmp_path/"exists";output.mkdir();(output/"sentinel").write_text("keep")
    with pytest.raises(FileExistsError):refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=output)
    assert (output/"sentinel").read_text()=="keep"
    for target in [work/"child",Path(parent.output_dir)/"child"]:
        with pytest.raises(ValueError,match="overlap"):refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=target)
    request=cfg.plan;request["objective"]=[0]*32
    with pytest.raises(ValueError,match="objective"):Config(request,level="full",run_id="run",machine="cpu",training_limits=training_limits(),publication_limits=publish_limits(),data_limits=data_limits())
    result=refine(parent,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=tmp_path/"child")
    with pytest.raises(TypeError,match="chaining"):
        refine(result,teacher=teacher,training=training,validation=validation,config=cfg,work_dir=work,output_dir=tmp_path/"grandchild")


def test_tiny_hf_multiowner_public_kd_refinement_with_masks(tmp_path):
    from test_owner_capture import _hf_source
    from tritium.torch import convert_native_module
    source=_hf_source(tmp_path);teacher=source.prepared.model
    parent=convert_native_module(source.prepared,tmp_path/"hf-parent","hf-public-fit",**json.loads(source.options_json))
    owners=[v["owner"] for v in json.loads(parent.asset("tritium-torch-state.json"))["state"] if v["disposition"]=="fitted"]
    def batch(ids):return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.int64),use_cache=False)
    training=[batch([1,2,3]),batch([1,2,4]),batch([2,3,4]),batch([1,3,4])];validation=[batch([1,2,3,4])]
    cfg=configuration(parent,training,validation,owners=owners,lr=0.000001)
    work=tmp_path/"runtime";work.mkdir()
    child=refine(parent,teacher=teacher,training=iter(training),validation=iter(validation),config=cfg,work_dir=work,output_dir=tmp_path/"hf-child")
    child.verify_with_parent(parent,max_parent_bytes=4_000_000,max_replay_working_bytes=4_000_000)
    assert not child.objective_authenticated and list(work.iterdir())==[]
