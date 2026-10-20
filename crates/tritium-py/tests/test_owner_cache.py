"""Actual owner capture -> common native .trit -> strict row replay/restart."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tritium import _tritium
from tritium.torch.owner_capture import NativeOwnerWindowCapture
from test_owner_capture import _source, _window, _objective, _capture


def _limits():
    return dict(max_payload_bytes=8192,max_context_bytes=4096,max_context_tensors=8,
                max_invocations=16,max_samples=64)


def _durable(source, window=None, objective=_objective, **options):
    return _capture(source, window, objective=objective, cache_limits=_limits(),
                    max_total_cache_bytes=8192, **options)


def _open(source, directory, **overrides):
    options=dict(limits=_limits(),max_package_bytes=65536,max_total_cache_bytes=8192)
    options.update(overrides)
    return NativeOwnerWindowCapture.open(source,directory,**options)


def _metric(caches, weights, row=0, damping=0.1, bounds=(8,65536,256,32768), bindings=None):
    values, identity = _tritium.NativeOwnerScoreCache.row_metric(
        caches, [c.binding_json for c in caches] if bindings is None else bindings,
        weights, row, damping, bounds)
    return torch.from_dlpack(values).reshape(caches[0].columns,caches[0].columns), identity


def test_actual_tied_windows_native_metric_matches_full_owner_autograd(tmp_path):
    source = _source(tmp_path)
    first = _window()
    second = _window(); second["input_ids"] = torch.tensor([2,0,1])
    captures = [_durable(source,w) for w in (first,second)]
    caches = [c.caches[0] for c in captures]
    for row in range(3):
        actual, identity = _metric(caches,[0.5,2.],row)
        assert actual.dtype == torch.float64 and len(identity)==64
        gradients = [torch.autograd.grad(_objective(source.prepared.model,w),
                     source.prepared.model.embed.weight)[0][row].double() for w in (first,second)]
        expected = (0.5*torch.outer(gradients[0],gradients[0])+2*torch.outer(gradients[1],gradients[1]))/2.5
        expected += 0.1*torch.eye(32,dtype=torch.float64)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        # Exact common-core row replay rather than framework f32 reduction order.
        rows = [torch.from_dlpack(c.row_score(row,256)) for c in caches]
        numerical = (0.5*torch.outer(rows[0],rows[0])+2*torch.outer(rows[1],rows[1]))/2.5
        numerical += 0.1*torch.eye(32,dtype=torch.float64)
        torch.testing.assert_close(actual,numerical,rtol=0,atol=0)
    assert source.prepared.model.embed.weight.grad is None


def test_native_metric_package_reopen_and_changed_semantics(tmp_path):
    source=_source(tmp_path)
    captured=_durable(source)
    directory=captured.save(tmp_path/"metric-caches",max_package_bytes=65536)
    reopened=_open(source,directory)
    original=captured.caches[0]; replay=reopened.caches[0]
    actual,identity=_metric([original],[1.],damping=0.)
    after,after_id=_metric([replay],[1.],damping=-0.,bounds=(16,131072,512,65536))
    torch.testing.assert_close(actual,after,rtol=0,atol=0)
    assert identity==after_id
    assert _metric([original],[2.],damping=0.)[1]!=identity
    assert _metric([original],[1.],row=1,damping=0.)[1]!=identity
    assert _metric([original],[1.],damping=0.2)[1]!=identity
    assert _metric([original,original],[1.,1.],damping=0.)[1]!=identity


@pytest.mark.parametrize("variant",["wrong_binding","recipe","weight","damping","rows","budget","inventory"])
def test_native_metric_rejects_wrong_context_and_admission(tmp_path,variant):
    source=_source(tmp_path); capture=_durable(source); cache=capture.caches[0]
    options={}
    caches=[cache]; weights=[1.]
    if variant=="wrong_binding":
        binding=json.loads(cache.binding_json); binding["samples"]+=1
        options["bindings"]=[json.dumps(binding)]
    elif variant=="recipe":
        another=_capture(source,objective_bytes=b"different recipe",cache_limits=_limits(),max_total_cache_bytes=8192)
        caches.append(another.caches[0]); weights.append(1.)
    elif variant=="weight": weights=[float("nan")]
    elif variant=="damping": options["damping"]=-1.
    elif variant=="rows": options["row"]=3
    elif variant=="budget": options["bounds"]=(8,65536,256,1)
    else: weights=[]
    with pytest.raises(ValueError): _metric(caches,weights,**options)


def test_real_owner_cache_saves_reopens_and_replays_exact_rows(tmp_path):
    source=_source(tmp_path)
    captured=_durable(source)
    assert len(captured.caches)==1
    cache=captured.caches[0]
    binding=json.loads(cache.binding_json)
    assert binding["content"]["estimator"]=="ScalarWindowGradient"
    assert binding["samples"]==10 and binding["context_bytes"]==51
    assert cache.payload_bytes==1376+51+2+len(captured.objective_bytes)
    assert cache.observation_bytes==1376
    directory=captured.save(tmp_path/"caches",max_package_bytes=65536)
    reopened=_open(source,directory)
    assert reopened.caches[0].cache_id==cache.cache_id
    assert reopened.invocations==captured.invocations
    assert reopened.objective_bytes==captured.objective_bytes
    for row in range(3):
        torch.testing.assert_close(reopened.row_score("embed.weight",row,max_workspace_bytes=256),
            captured.row_score("embed.weight",row,max_workspace_bytes=256),rtol=0,atol=0)
    assert reopened.caches[0].package(65536)==cache.package(65536)
    with pytest.raises(FileExistsError): captured.save(directory,max_package_bytes=65536)
    assert list((tmp_path/"work").iterdir())==[]


def test_new_process_native_reopen_has_exact_identity_and_row_bits(tmp_path):
    source=_source(tmp_path)
    captured=_durable(source)
    directory=captured.save(tmp_path/"caches",max_package_bytes=65536)
    expected=captured.caches[0]
    # The child uses the actual installed/source extension, never a copied codec.
    code="""import json,sys,torch
from tritium import _tritium
root=sys.argv[1]
with open(root+'/bindings.json') as f: binding=json.load(f)['bindings']['embed.weight']
limits=json.loads(sys.argv[2])
c=_tritium.NativeOwnerScoreCache.open(root+'/owner-0.trit',json.dumps(binding),json.dumps(limits),65536)
row=torch.from_dlpack(c.row_score(0,256))
print(json.dumps({'id':c.cache_id,'row':row.view(torch.int64).tolist()}))
"""
    result=subprocess.run([sys.executable,"-c",code,str(directory),json.dumps(_limits())],
        env=os.environ.copy(),check=True,capture_output=True,text=True,timeout=30)
    actual=json.loads(result.stdout)
    assert actual["id"]==expected.cache_id
    assert actual["row"]==captured.row_score("embed.weight",0,max_workspace_bytes=256).view(torch.int64).tolist()


@pytest.mark.parametrize("change",["objective","labels","attention","context_dtype","values","paths","ordinal","tokenizer"])
def test_actual_content_changes_do_not_reuse_cache_identity(tmp_path, change):
    source=_source(tmp_path)
    original=_durable(source)
    window=_window(); objective=_objective
    if change=="objective":
        changed=_capture(source,window,objective=objective,objective_bytes=b"another scalar recipe",
            cache_limits=_limits(),max_total_cache_bytes=8192)
        assert changed.caches[0].cache_id!=original.caches[0].cache_id
        return
    if change=="labels": window["labels"][1]=2
    elif change=="attention": window["attention_mask"][0]=False
    elif change=="context_dtype": window["labels"]=window["labels"].to(torch.int32)
    elif change=="values": objective=lambda m,w:2*_objective(m,w)
    if change in {"labels","attention","context_dtype","values"}:
        assert _durable(source,window,objective).caches[0].cache_id!=original.caches[0].cache_id
        return
    # Expert native receiver binds supplied invocation metadata too; source/recipe
    # execution remains a separate obligation, not asserted by these edits.
    binding=json.loads(original.caches[0].binding_json)
    meta=dict(context=[b["meta"] for b in binding["content"]["context"]],
              invocations=[b["meta"] for b in binding["content"]["invocations"]])
    if change=="paths": meta["invocations"][0]["paths"]=["another"]
    elif change=="ordinal":
        for invocation in meta["invocations"]: invocation["ordinal"]+=1
    context=[window[b["name"]].detach().reshape(-1).view(torch.uint8) for b in meta["context"]]
    tokenizer=b"different tokenizer" if change=="tokenizer" else b"{}"
    if change=="tokenizer":
        with pytest.raises(ValueError,match="tokenizer"):
            source.observation.owner_cache(original.owners[0],tokenizer,original.objective_bytes,
                                          json.dumps(meta),context,json.dumps(_limits()))
    else:
        changed=source.observation.owner_cache(original.owners[0],tokenizer,original.objective_bytes,
                                              json.dumps(meta),context,json.dumps(_limits()))
        assert changed.cache_id!=original.caches[0].cache_id


def test_receiving_and_producer_ceilings_do_not_change_content_identity(tmp_path):
    source=_source(tmp_path)
    first=_durable(source)
    larger=dict(_limits(),max_payload_bytes=16384,max_context_bytes=8192,max_samples=128)
    second=_capture(source,cache_limits=larger,max_total_cache_bytes=16384)
    assert first.caches[0].cache_id==second.caches[0].cache_id
    assert first.caches[0].package(65536)!=second.caches[0].package(65536)
    directory=second.save(tmp_path/"caches",max_package_bytes=65536)
    exact=dict(_limits(),max_payload_bytes=second.caches[0].payload_bytes)
    reopened=_open(source,directory,limits=exact,max_total_cache_bytes=exact["max_payload_bytes"])
    assert reopened.caches[0].cache_id==second.caches[0].cache_id
    with pytest.raises(ValueError):
        _open(source,directory,limits=dict(exact,max_payload_bytes=exact["max_payload_bytes"]-1))


@pytest.mark.parametrize("variant",["payload","binding","extra","profile","symlink","source","receiver","package"])
def test_corruption_source_inventory_and_receiving_fail_closed(tmp_path,variant):
    source=_source(tmp_path)
    directory=_durable(source).save(tmp_path/"caches",max_package_bytes=65536)
    options={}
    if variant=="payload":
        path=directory/"owner-0.trit"; data=bytearray(path.read_bytes()); data[-1]^=1; path.write_bytes(data)
    elif variant in {"binding","profile"}:
        path=directory/"bindings.json"; data=json.loads(path.read_text())
        if variant=="profile": data["profile"]="tritium.calibration-cache.v1"
        else: data["bindings"]["embed.weight"]["samples"]+=1
        path.write_text(json.dumps(data))
    elif variant=="extra": (directory/"extra").write_bytes(b"x")
    elif variant=="symlink":
        path=directory/"owner-0.trit"; moved=tmp_path/"moved.trit"; path.rename(moved); path.symlink_to(moved)
    elif variant=="source":
        with torch.no_grad(): source.prepared.model.embed.weight.add_(1)
    elif variant=="receiver": options["max_total_cache_bytes"]=1
    elif variant=="package": options["max_package_bytes"]=1
    with pytest.raises((ValueError,OSError)): _open(source,directory,**options)


def test_ephemeral_or_unadmitted_cache_publication_has_no_success(tmp_path):
    source=_source(tmp_path)
    ephemeral=_capture(source)
    with pytest.raises(ValueError,match="durable"):
        ephemeral.save(tmp_path/"no-cache",max_package_bytes=65536)
    with pytest.raises(ValueError,match="aggregate"):
        _capture(source,cache_limits=_limits(),max_total_cache_bytes=8191)
    durable=_durable(source)
    with pytest.raises(ValueError): durable.save(tmp_path/"too-small",max_package_bytes=1)
    assert not (tmp_path/"too-small").exists() and not list(tmp_path.glob(".tritium-owner-cache-*"))


def test_complete_tiny_hf_owner_inventory_reopens_with_shared_context(tmp_path):
    from test_owner_capture import _hf_source
    source=_hf_source(tmp_path)
    limits=dict(_limits(),max_payload_bytes=65536,max_context_bytes=8192,max_samples=256)
    capture=_capture(source,dict(input_ids=torch.tensor([[1,2,3,4]]),labels=torch.tensor([[1,2,3,4]])),
        objective=lambda m,w:m(**w,use_cache=False).loss,max_observation_bytes=1_000_000,
        objective_bytes=b"tiny HF next-token CE mean",
        max_context_bytes=8192,max_invocations=32,max_samples=256,max_graph_nodes=4096,
        cache_limits=limits,max_total_cache_bytes=1_000_000)
    directory=capture.save(tmp_path/"caches",max_package_bytes=200_000)
    reopened=_open(source,directory,limits=limits,max_package_bytes=200_000,max_total_cache_bytes=1_000_000)
    assert reopened.invocations==capture.invocations
    assert [c.cache_id for c in reopened.caches]==[c.cache_id for c in capture.caches]
    for owner in capture.owners:
        torch.testing.assert_close(reopened.row_score(owner.tensor_name,0,max_workspace_bytes=1024),
            capture.row_score(owner.tensor_name,0,max_workspace_bytes=1024),rtol=0,atol=0)
    total=sum(c.payload_bytes for c in capture.caches)
    with pytest.raises(ValueError):
        _open(source,directory,limits=limits,max_package_bytes=200_000,max_total_cache_bytes=total-1)


def test_complete_tiny_hf_two_window_metrics_match_actual_language_autograd(tmp_path):
    from test_owner_capture import _hf_source
    from tritium.torch.native_source import _inventory
    source=_hf_source(tmp_path)
    windows=[dict(input_ids=torch.tensor([tokens]),labels=torch.tensor([tokens]))
             for tokens in ([1,2,3,4],[4,3,2,1])]
    objective=lambda m,w:m(**w,use_cache=False).loss
    captures=[_capture(source,w,objective=objective,objective_bytes=b"tiny HF next-token CE mean",
        max_observation_bytes=1_000_000,max_context_bytes=8192,max_invocations=32,
        max_samples=256,max_graph_nodes=4096,
        cache_limits=dict(_limits(),max_payload_bytes=65536,max_context_bytes=8192,max_samples=256),
        max_total_cache_bytes=1_000_000) for w in windows]
    tensors=[tensor for _,_,tensor,_,selected,_ in _inventory(source.prepared) if selected]
    gradients=[torch.autograd.grad(objective(source.prepared.model,w),tensors) for w in windows]
    assert len(tensors)==len(captures[0].caches)==len(captures[1].caches)
    for index,(a,b) in enumerate(zip(captures[0].caches,captures[1].caches)):
        assert a.tensor_name==b.tensor_name
        for row in (0,a.rows-1):
            actual,_=_metric([a,b],[1.,2.],row,bounds=(8,200_000,1024,200_000))
            ga,gb=gradients[0][index][row].double(),gradients[1][index][row].double()
            expected=(torch.outer(ga,ga)+2*torch.outer(gb,gb))/3
            expected+=0.1*torch.eye(a.columns,dtype=torch.float64)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-7)
