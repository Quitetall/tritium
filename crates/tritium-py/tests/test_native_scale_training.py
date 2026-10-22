"""Installed Rust/Torch native scale training, not model/release acceptance."""
import gc
import json
import pytest
import torch
from tritium.torch import native_scale_training, load_native_module, convert_native_module
from test_native_module_source import StateModel, _prepared, _convert, _fit


def plan(result, owner="a.weight", updates=4, level="full"):
    return dict(parent=json.loads(result.native_refinement_parent(level)),child_level="scale-child",
                owners=[owner] if isinstance(owner,str) else sorted(owner),recipe={"ScaleOnly":{"optimizer":dict(lr=0.005,beta1=0.9,beta2=0.99,eps=1e-8,weight_decay=0.0)}},
                objective=[7]*32,teacher_source=None,
                training=[dict(content=[10+i]*32,weight=2.0 if i==0 else 1.0) for i in range(updates)],
                validation=[dict(content=[30]*32,weight=1.0)],
                limits=dict(max_state_bytes=1_000_000,max_workspace_bytes=1_000_000,
                            max_checkpoint_bytes=1_000_000,max_package_bytes=1_000_000))


def start(shell, result, request=None, **kwargs):
    options=dict(level="full",plan=plan(result) if request is None else request,load_run_id="scale-load",
                 max_snapshot_bytes=65536,max_parent_bytes=1_000_000,max_saved_input_bytes=1_000_000)
    options.update(kwargs)
    return native_scale_training(shell,result,**options)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize("basis",["Identity",{"SignedRht":dict(block=32,seed=13,domain=29)}])
def test_real_backward_reduces_tied_owner_and_matches_packed_vjp(tmp_path,dtype,basis):
    torch.manual_seed(17)
    action=_fit();action["Fit"]["plan"]["layout"]["basis"]=basis
    result=_convert(tmp_path,_prepared(StateModel(dtype)),actions={"a.weight":action})
    shell=StateModel(dtype);before={n:t.clone() for n,t in shell.state_dict().items()};derived=shell.not_persistent.clone()
    trainer=start(shell,result);model=trainer.model
    assert model.a.packed_weight is model.b.packed_weight
    x=torch.linspace(-0.5,0.5,64,dtype=dtype).reshape(2,32).requires_grad_()
    y=model(x);loss=y.float().square().mean();loss.backward()
    _,handle,leaf,_=trainer._cache["a.weight"]
    assert leaf.dtype==torch.float64 and leaf.grad is not None
    norm=model.norm(x).detach().float().contiguous()
    dy=(2*y.detach().float()/y.numel()).to(dtype).float().contiguous()
    expected=2*torch.utils.dlpack.from_dlpack(handle.scale_vjp(norm,dy))
    torch.testing.assert_close(leaf.grad,expected,rtol=3e-5,atol=3e-5)
    dense=torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
    rx=x.detach().clone().requires_grad_();rnorm=model.norm(rx)
    ry=2*(torch.nn.functional.linear(rnorm.float(),dense).to(dtype)+model.a.bias.to(dtype))
    ry.float().square().mean().backward()
    tolerance=3e-5 if dtype==torch.float32 else 5*torch.finfo(dtype).eps
    torch.testing.assert_close(x.grad,rx.grad,rtol=tolerance,atol=tolerance)
    old=trainer.checkpoint();trainer.step([10]*32);assert trainer.revision==1 and trainer.checkpoint()!=old
    for n,t in shell.state_dict().items():assert torch.equal(t,before[n])
    assert torch.equal(shell.not_persistent,derived)
    trained=trainer.close();assert trained is model and not trainer.active
    assert not hasattr(model,"_tritium_native_artifact") and model._tritium_native_unpublished_scales
    with torch.no_grad():assert torch.isfinite(model(x)).all()


def test_training_updates_loss_and_checkpoint_restart(tmp_path):
    torch.manual_seed(23);result=_convert(tmp_path,_prepared());request=plan(result)
    a=start(StateModel(),result,request);x=torch.linspace(-1,1,64).reshape(2,32)
    initial=a.model(x).square().mean().item();b=None
    for i in range(4):
        a.zero_grad();a.model(x).square().mean().backward();a.step([10+i]*32)
        if b is not None:
            b.zero_grad();b.model(x).square().mean().backward();b.step([10+i]*32)
            assert a.checkpoint()==b.checkpoint()
            with torch.no_grad():assert torch.equal(a.model(x),b.model(x))
        if i==1:b=start(StateModel(),result,request,checkpoint=a.checkpoint(),load_run_id="restart")
    with torch.no_grad():final=a.model(x).square().mean().item()
    assert final<initial
    print(f"NATIVE_TORCH_SCALE_LOSS initial={initial} final={final}")
    a.close();b.close();result.verify()


def test_tied_embedding_and_linear_backward_share_one_owner_leaf(tmp_path):
    from test_owner_capture import _source,TiedModel
    source=_source(tmp_path);result=convert_native_module(source.prepared,tmp_path/"tied-package","tied-fit",**json.loads(source.options_json))
    trainer=start(TiedModel(),result,plan(result,"embed.weight"));ids=torch.tensor([0,1,0])
    first,second=trainer.model(ids);(first.square().mean()+second.square().mean()).backward()
    assert len(trainer._cache)==1 and trainer.model.embed.packed_weight is trainer.model.head.packed_weight
    leaf=trainer._cache["embed.weight"][2];assert torch.isfinite(leaf.grad).all() and leaf.grad.numel()>0
    trainer.step([10]*32);assert trainer.revision==1
    trainer.close()


def test_stale_backward_higher_order_and_external_state_are_refused(tmp_path):
    result=_convert(tmp_path,_prepared());trainer=start(StateModel(),result);x=torch.ones(2,32,requires_grad=True)
    stale=trainer.model(x).sum();trainer.model(x).square().sum().backward();trainer.step([10]*32)
    checkpoint=trainer.checkpoint()
    with pytest.raises(RuntimeError,match="stale"):stale.backward()
    with pytest.raises(RuntimeError,match="backward failed"):trainer.step([11]*32)
    assert trainer.checkpoint()==checkpoint;trainer.zero_grad()
    loss=trainer.model(x).sum()
    with pytest.raises(RuntimeError,match="first-order"):torch.autograd.grad(loss,x,create_graph=True)
    trainer.zero_grad()
    with pytest.raises(RuntimeError,match="dtype/device"):trainer.model.half()
    trainer.model(x).sum().backward();leaf=trainer._cache["a.weight"][2]
    with torch.no_grad():leaf.add_(1)
    with pytest.raises(RuntimeError,match="leaf changed"):trainer.step([11]*32)
    assert trainer.checkpoint()==checkpoint


def test_budget_and_bad_window_refuse_before_update_and_keep_gradients(tmp_path):
    result=_convert(tmp_path,_prepared());request=plan(result)
    #One owner/two F32 scales: initial snapshot8bytes; next staging needs another8.
    trainer=start(StateModel(),result,request,max_snapshot_bytes=8);x=torch.linspace(-1,1,32).unsqueeze(0)
    trainer.model(x).sum().backward();before=trainer.checkpoint()
    with pytest.raises(ValueError,match="window differs"):trainer.step([99]*32)
    with pytest.raises(ValueError,match="budget"):trainer.step([10]*32)
    assert trainer.checkpoint()==before and trainer._cache["a.weight"][2].grad is not None
    trainer.close()


def test_saved_input_cap_precedes_copy_and_releases_with_graph(tmp_path,monkeypatch):
    import tritium.torch.native_training as module
    result=_convert(tmp_path,_prepared());trainer=start(StateModel(),result,max_saved_input_bytes=1)
    def no_copy(value):raise AssertionError("rejection must precede input copy")
    monkeypatch.setattr(module,"_compact",no_copy)
    with pytest.raises(ValueError,match="saved-input"):trainer.model(torch.ones(2,32))
    assert trainer.saved_input_bytes==0
    monkeypatch.undo();trainer.close()
    trainer=start(StateModel(),result);output=trainer.model(torch.ones(2,32));assert trainer.saved_input_bytes>0
    del output;gc.collect();assert trainer.saved_input_bytes==0;trainer.close()


@pytest.mark.parametrize("change",["owner","preserved","topology"])
def test_mutated_private_model_is_detected_without_touching_source(tmp_path,change):
    result=_convert(tmp_path,_prepared());shell=StateModel();trainer=start(shell,result);before=trainer.checkpoint()
    if change=="owner":trainer.model.b.packed_weight=trainer.model.a.packed_weight.__class__(trainer.model.a.packed_weight.handle)
    elif change=="preserved":trainer.model.counter.add_(1)
    else:trainer.model.extra=torch.nn.ReLU()
    with pytest.raises(RuntimeError,match="changed|rebound"):trainer.step([10]*32)
    assert trainer.checkpoint()==before


def test_invalid_request_rejects_before_loading_and_default_inference_unchanged(tmp_path):
    result=_convert(tmp_path,_prepared());request=plan(result);request["parent"]["level"]="compact"
    with pytest.raises(ValueError,match="level differs"):start(StateModel(),result,request)
    normal=load_native_module(StateModel(),result,level="full",load_run_id="default",inplace=False)
    with pytest.raises(RuntimeError,match="inference-only"):normal.a.packed_weight.matmul(torch.ones(1,32,requires_grad=True))
    with pytest.raises(TypeError):start(StateModel(),result,plan(result),inplace=True)


def test_embedding_padding_masks_only_embedding_use_and_handles_scalar_indices(tmp_path):
    from test_owner_capture import _source,TiedModel
    source=_source(tmp_path);result=convert_native_module(source.prepared,tmp_path/"padding-package","padding-fit",**json.loads(source.options_json))
    trainer=start(TiedModel(),result,plan(result,"embed.weight"));weight=trainer.model.embed.packed_weight
    # Native gather's consumer-specific padding rule, independent of any tied
    # head use; the same owner can still receive head gradients for that row.
    output=weight.gather(torch.tensor(0),torch.float32,padding=0);output.sum().backward()
    assert torch.equal(trainer._cache["embed.weight"][2].grad,torch.zeros_like(trainer._cache["embed.weight"][2]))
    trainer.zero_grad();output=weight.gather(torch.tensor([0,1,0]),torch.float32,padding=0);output.sum().backward()
    _,handle,leaf,_=trainer._cache["embed.weight"]
    expected=torch.utils.dlpack.from_dlpack(handle.embedding_scale_vjp(torch.tensor([0,1,0]),torch.tensor([[0.]*32,[1.]*32,[0.]*32])))
    assert torch.equal(leaf.grad,expected);trainer.close()


def test_empty_strided_inputs_and_training_dtype_refusals(tmp_path):
    result=_convert(tmp_path,_prepared());trainer=start(StateModel(),result);weight=trainer.model.a.packed_weight
    empty=torch.empty(0,32,requires_grad=True);weight.matmul(empty).sum().backward();assert empty.grad.shape==(0,32)
    trainer.zero_grad();value=torch.ones(2,64)[:,::2].requires_grad_();weight.matmul(value).sum().backward();assert value.grad.shape==(2,32)
    with pytest.raises(ValueError,match="CPU"):weight.matmul(torch.ones(1,32,dtype=torch.float64))
    with torch.autocast("cpu",dtype=torch.bfloat16):
        with pytest.raises(RuntimeError,match="autocast"):weight.matmul(torch.ones(1,32))
    trainer.close()


def test_tiny_hf_real_loss_backward_and_atomic_scale_update(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    source=_hf_source(tmp_path);result=convert_native_module(source.prepared,tmp_path/"hf-scale-package","hf-scale-fit",**json.loads(source.options_json))
    catalog=json.loads(result.asset("tritium-torch-state.json"));owners=[e["owner"] for e in catalog["state"] if e["disposition"]=="fitted"]
    trainer=start(LlamaForCausalLM(source.prepared.model.config),result,plan(result,owners),max_snapshot_bytes=1_000_000,max_saved_input_bytes=10_000_000)
    tokens=torch.tensor([[1,2,3]]);before=trainer.checkpoint();loss=trainer.model(tokens,labels=tokens).loss
    assert torch.isfinite(loss);loss.backward();assert all(item[2].grad is not None for item in trainer._cache.values())
    trainer.step([10]*32);assert trainer.checkpoint()!=before and trainer.revision==1;trainer.close()


@pytest.mark.parametrize("level",["compact","full"])
def test_one_explicit_parent_level_and_detached_plan_metadata(tmp_path,level):
    result=_convert(tmp_path,_prepared());parent=(tmp_path/"output/model.trit").read_bytes()
    trainer=start(StateModel(),result,plan(result,level=level),level=level)
    assert trainer.plan["parent"]["level"]==level
    metadata=trainer.plan;metadata["owners"].clear();metadata["parent"]["level"]="invented"
    trainer.model(torch.linspace(-1,1,32).unsqueeze(0)).sum().backward();trainer.step([10]*32)
    assert trainer.revision==1 and trainer.plan["parent"]["level"]==level
    other="full" if level=="compact" else "compact"
    with pytest.raises(ValueError):
        start(StateModel(),result,plan(result,level=other),level=other,checkpoint=trainer.checkpoint())
    trainer.close();result.verify()
    assert (tmp_path/"output/model.trit").read_bytes()==parent
    assert not (tmp_path/"output/scale-child").exists()  # No publication claim.


def test_forged_parent_identity_is_not_admitted(tmp_path):
    result=_convert(tmp_path,_prepared());request=plan(result)
    request["parent"]["package"]=[0]*32
    with pytest.raises(ValueError):start(StateModel(),result,request)
