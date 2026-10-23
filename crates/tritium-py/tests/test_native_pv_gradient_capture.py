"""Bounded pending adjoints with real Torch graphs, not model qualification."""
from pathlib import Path
import pytest
import torch
from tritium import NativePvPagedGradientLimits
from test_native_module_source import StateModel, _prepared, _convert, _fit
from test_native_pv_paged_training import start, publish, reopen, caps
from test_native_pv_training import plan
from test_native_resume import limits as bundle_limits


def capture_limits(**changes):
    args = dict(max_spool_bytes=1_000_000, max_owner_working_bytes=1_000_000, max_contributions=128)
    args.update(changes)
    return NativePvPagedGradientLimits(**args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("codec", ["D2", "B3"])
@pytest.mark.parametrize("basis", ["Identity", {"SignedRht":dict(block=32,seed=13,domain=29)}])
def test_scalar_tokens_real_backward_resume_and_publication_match_existing_paged_path(tmp_path,dtype,codec,basis):
    torch.manual_seed(83)
    action=_fit();action["Fit"]["plan"]["layout"].update(codec=codec,basis=basis)
    parent = _convert(tmp_path, _prepared(StateModel(dtype)),actions={"a.weight":action})
    a = start(StateModel(dtype), parent, tmp_path/"ordinary")
    b = start(StateModel(dtype), parent, tmp_path/"capture", gradient_capture=capture_limits())
    for i in range(3):
        outputs, gradients, stats = [], [], []
        for trainer in (a, b):
            trainer.zero_grad()
            x = torch.linspace(-1, 1, 64,dtype=dtype).reshape(2, 32).requires_grad_()
            y = trainer.model(x); y.float().square().mean().backward()
            outputs.append(y.detach()); gradients.append(x.grad)
            if trainer is b:
                assert b.workspace_report["pending_contributions"] == 2
                assert b.workspace_report["files"] == 3
                for _, _, token, _ in b._cache.values():
                    assert token.numel() == 1 and token.grad is None
            stats.append(trainer.step([10+i]*32))
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(gradients[0], gradients[1])
        assert stats[0] == stats[1]
        assert b.workspace_report["files"] == 2 and b.workspace_report["pending_gradient_bytes"] == 0
        if i == 0:
            expected = a.save_resume(tmp_path/"ordinary-prefix", limits=bundle_limits())
            saved = b.save_resume(tmp_path/"capture-prefix", limits=bundle_limits())
            assert expected.bundle_id == saved.bundle_id
            b.close(); assert not (tmp_path/"capture").exists()
            b = start(StateModel(dtype), parent, tmp_path/"resumed", plan=None, resume=reopen(saved), gradient_capture=capture_limits())
    reference = publish(a, tmp_path/"reference")
    child = publish(b, tmp_path/"child")
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(reference.output_dir)/name).read_bytes() == (Path(child.output_dir)/name).read_bytes()
    assert not child.quality_qualified and not child.objective_authenticated
    a.close(); b.close(); assert not (tmp_path/"resumed").exists()


@pytest.mark.parametrize("field", ["max_spool_bytes","max_owner_working_bytes","max_contributions"])
def test_capture_limits_are_positive_exact_and_immutable(field):
    with pytest.raises(ValueError):capture_limits(**{field:0})
    for value in (True,1.,"1",-1):
        with pytest.raises((TypeError,OverflowError)):capture_limits(**{field:value})
    value=capture_limits()
    with pytest.raises(AttributeError):setattr(value,field,1)


@pytest.mark.parametrize("failure", ["spool","owner","step-transfer","type","workspace"])
def test_capture_caps_refuse_without_committing_or_leaking_work(tmp_path,failure):
    parent=_convert(tmp_path,_prepared());work=tmp_path/"capture"
    extra={"gradient_capture":capture_limits()}
    if failure=="spool":extra["gradient_capture"]=capture_limits(max_spool_bytes=8)
    elif failure=="owner":extra["gradient_capture"]=capture_limits(max_owner_working_bytes=16384)
    elif failure=="step-transfer":extra["limits"]=caps(max_gradient_working_bytes=1)
    elif failure=="type":extra["gradient_capture"]={}
    else:extra["limits"]=caps(max_workspace_bytes=3_000_000)
    if failure=="workspace":
        b=start(StateModel(),parent,work,**extra)
        with pytest.raises(ValueError,match="workspace"):b.zero_grad()
        assert b.revision==0 and b.workspace_report["files"]==2
        b.close();assert not work.exists()
    else:
        with pytest.raises((TypeError,ValueError)):start(StateModel(),parent,work,**extra)
        assert not work.exists()
    parent.verify()


def test_failed_backward_poison_and_zero_grad_reset_do_not_advance_recording(tmp_path):
    parent=_convert(tmp_path,_prepared());b=start(StateModel(),parent,tmp_path/"capture",gradient_capture=capture_limits(max_contributions=1))
    b.zero_grad();x=torch.ones(2,32)
    with pytest.raises(ValueError,match="contribution cap"):b.model(x).square().mean().backward()
    assert b.workspace_report["pending_gradient_poisoned"] and b.revision==0
    with pytest.raises(RuntimeError,match="zero_grad"):b.step([10]*32)
    b.zero_grad();assert b.workspace_report["pending_contributions"]==0 and not b.workspace_report["pending_gradient_poisoned"]
    # One direct matrix use is a valid bounded observation, unlike the full tied graph.
    b.model.a(x).square().mean().backward();assert b.workspace_report["pending_contributions"]==1
    b.step([10]*32);assert b.revision==1 and b.workspace_report["pending_gradient_bytes"]==0
    b.close()


def test_window_source_refusal_and_pending_resume_omit_uncommitted_observations(tmp_path):
    parent=_convert(tmp_path,_prepared());work=tmp_path/"capture"
    b=start(StateModel(),parent,work,gradient_capture=capture_limits())
    b.zero_grad();b.model(torch.ones(2,32)).square().mean().backward()
    saved=b.save_resume(tmp_path/"pending-prefix",limits=bundle_limits());assert saved.completed_updates==0
    with pytest.raises(ValueError,match="window"):b.step([99]*32)
    assert b.workspace_report["pending_contributions"]==2 and b.revision==0
    path=Path(parent.output_dir)/"model.trit";original=path.read_bytes();changed=bytearray(original);changed[-1]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError):b.step([10]*32)
    assert b.revision==0 and b.workspace_report["files"]==3 and not b.workspace_report["pending_gradient_poisoned"]
    path.write_bytes(original);b.step([10]*32);b.close()
    c=start(StateModel(),parent,tmp_path/"resumed",plan=None,resume=reopen(saved),gradient_capture=capture_limits())
    assert c.revision==0 and c.workspace_report["pending_gradient_bytes"]==0
    with pytest.raises(ValueError,match="no captured gradients"):c.step([10]*32)
    c.close()


def test_public_convert_tied_gather_repeated_uses_and_unused_owners_match(tmp_path):
    from test_owner_phases import _public_source,_recipe
    from test_owner_capture import TiedModel,_window,_objective
    from tritium.torch import calibrate,convert
    source=_public_source(tmp_path);calibration=calibrate(source,[_window()],objective=_objective,owner_recipe=_recipe())
    parent=convert(source,calibration,output_dir=tmp_path/"parent",run_id="captured-parent")
    request=plan(parent,owners="embed.weight",updates=1)
    a=start(TiedModel(),parent,tmp_path/"ordinary",request)
    b=start(TiedModel(),parent,tmp_path/"capture",request,gradient_capture=capture_limits())
    results=[]
    for t in (a,b):
        t.zero_grad();_objective(t.model,_window()).backward();results.append(t.step([10]*32))
    assert results[0]==results[1]
    ref=publish(a,tmp_path/"reference");child=publish(b,tmp_path/"child")
    for name in ("checkpoint.tnpv","gradients.tnpj"):
        assert (Path(ref.output_dir)/name).read_bytes()==(Path(child.output_dir)/name).read_bytes()
    a.close();b.close()


@pytest.mark.parametrize("failure", ["stale","higher-order","input-version","leaf-version","preserved-state"])
def test_capture_retains_graph_guards_and_public_close_retires_pending(tmp_path,failure):
    parent=_convert(tmp_path,_prepared());work=tmp_path/"capture"
    b=start(StateModel(),parent,work,gradient_capture=capture_limits());b.zero_grad()
    x=torch.ones(2,32,requires_grad=True);y=b.model(x)
    if failure=="stale":
        y.square().mean().backward(retain_graph=True);b.step([10]*32)
        with pytest.raises(RuntimeError,match="stale"):y.sum().backward()
    elif failure=="higher-order":
        with pytest.raises(RuntimeError,match="first-order"):torch.autograd.grad(y.sum(),x,create_graph=True)
    elif failure=="input-version":
        with torch.no_grad():x.add_(1)
        with pytest.raises(RuntimeError,match="modified|version"):y.sum().backward()
    elif failure=="leaf-version":
        with torch.no_grad():b._cache["a.weight"][2].add_(1)
        with pytest.raises(RuntimeError,match="leaf changed"):y.sum().backward()
    else:
        with torch.no_grad():b.model.a.bias.add_(1)
        with pytest.raises(RuntimeError,match="preserved state"):y.sum().backward()
    if failure in {"leaf-version","preserved-state"}:
        with pytest.raises(RuntimeError):b.close()
    else:b.close()
    assert not b.active and not work.exists() and b.workspace_report["files"]==0


def test_unused_owner_zero_adjoints_and_decay_match_existing_path(tmp_path):
    class Unused(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Linear(32,1);self.b=torch.nn.Linear(32,1)
        def forward(self,x):return self.a(x)
    torch.manual_seed(97)
    parent=_convert(tmp_path,_prepared(Unused()),actions={"a.weight":_fit(),"b.weight":_fit()})
    request=plan(parent,owners=["a.weight","b.weight"],updates=1)
    a=start(Unused(),parent,tmp_path/"ordinary",request)
    b=start(Unused(),parent,tmp_path/"capture",request,gradient_capture=capture_limits())
    results=[]
    for t in (a,b):
        t.zero_grad();t.model(torch.ones(2,32)).square().mean().backward();results.append(t.step([10]*32))
    assert results[0]==results[1] and "b.weight" not in b._cache
    ref=publish(a,tmp_path/"reference");child=publish(b,tmp_path/"child")
    for name in ("checkpoint.tnpv","gradients.tnpj"):
        assert (Path(ref.output_dir)/name).read_bytes()==(Path(child.output_dir)/name).read_bytes()
    a.close();b.close()


def test_multiowner_hf_scalar_tokens_reload_and_generate_without_leaf_grads(tmp_path):
    import json
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    from tritium.torch import convert_native_module,load
    from test_native_pv_paged_verify import _limits as audit_limits
    source=_hf_source(tmp_path)
    parent=convert_native_module(source.prepared,tmp_path/"parent","capture-hf-parent",**json.loads(source.options_json))
    catalog=json.loads(parent.asset("tritium-torch-state.json"))
    owners=[e["owner"] for e in catalog["state"] if e["disposition"]=="fitted"]
    assert len(owners)>1
    request=plan(parent,owners=owners,updates=2)
    a=start(LlamaForCausalLM(source.prepared.model.config),parent,tmp_path/"ordinary",request)
    b=start(LlamaForCausalLM(source.prepared.model.config),parent,tmp_path/"capture",request,gradient_capture=capture_limits())
    tokens=torch.tensor([[1,2,3]])
    for i in range(2):
        results=[]
        for t in (a,b):
            t.zero_grad();t.model(tokens,labels=tokens,use_cache=False).loss.backward()
            if t is b:
                assert len(b._cache)>1
                assert all(token.numel()==1 and token.grad is None for _,_,token,_ in b._cache.values())
                assert b.workspace_report["pending_contributions"]>=len(owners)
            results.append(t.step([10+i]*32))
        assert results[0]==results[1]
    ref=publish(a,tmp_path/"reference");child=publish(b,tmp_path/"child")
    for name in ("checkpoint.tnpv","gradients.tnpj"):
        assert (Path(ref.output_dir)/name).read_bytes()==(Path(child.output_dir)/name).read_bytes()
    child.verify_paged_with_parent(parent,tmp_path/"audit",limits=audit_limits(child))
    loaded=load(child,model=LlamaForCausalLM(source.prepared.model.config),level="pv-child",load_run_id="capture-hf-load",inplace=False)
    generation=dict(max_new_tokens=2,min_new_tokens=2,do_sample=False,use_cache=False,attention_mask=torch.ones_like(tokens),pad_token_id=0)
    with torch.no_grad():
        assert torch.equal(loaded(tokens,use_cache=False).logits,b.model(tokens,use_cache=False).logits)
        assert torch.equal(loaded.generate(tokens,**generation),b.model.generate(tokens,**generation))
    a.close();b.close();assert not (tmp_path/"capture").exists()
