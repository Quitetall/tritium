"""Installed real Torch/native P/V backward and publication, not model qualification."""
import gc
import json
from pathlib import Path
import pytest
import torch
from tritium import NativeModelRefinement
from tritium.torch import native_pv_training, native_scale_training, load_native_module, convert_native_module, load, export
from test_native_module_source import StateModel, _prepared, _convert, _fit
from test_native_scale_training import plan as scale_plan
from test_native_refinement import limits, recording


def plan(parent, owners="a.weight", updates=3, level="full"):
    result = scale_plan(parent, owners, updates, level)
    continuous = dict(lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.01)
    result["recipe"] = {"AlternatingPv": dict(continuous_optimizer=continuous,
        code_optimizer={**continuous, "lr": 0.15}, max_code_change_fraction=0.5, max_relative_code_change=1.0)}
    result["child_level"] = "pv-child"
    return result


def start(shell, parent, request=None, **kwargs):
    args = dict(level="full", plan=plan(parent) if request is None else request, load_run_id="pv-load",
                max_snapshot_bytes=1_000_000, max_parent_bytes=4_000_000, max_saved_input_bytes=10_000_000)
    args.update(kwargs)
    return native_pv_training(shell, parent, **args)


def update(trainer, ordinal, x):
    trainer.zero_grad()
    trainer.model(x).float().square().mean().backward()
    return trainer.step([10 + ordinal] * 32)


def publish(trainer, output, **kwargs):
    args = dict(machine="synthetic-installed-wheel", validation=[dict(content=[30]*32, parent_loss=1., child_loss=2.)], limits=limits())
    args.update(kwargs)
    return trainer.publish(output, "python-pv-child", **args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("basis", ["Identity", {"SignedRht": dict(block=32, seed=13, domain=29)}])
@pytest.mark.parametrize("precision", ["F16", "F32"])
@pytest.mark.parametrize("codec", ["D2", "B3"])
def test_real_backward_matches_packed_dual_adjoint_and_input_reference(tmp_path, dtype, basis, precision, codec):
    torch.manual_seed(17)
    action = _fit(); layout = action["Fit"]["plan"]["layout"]
    layout.update(basis=basis, codec=codec); layout["law"]["precision"] = precision
    parent = _convert(tmp_path, _prepared(StateModel(dtype)), actions={"a.weight": action})
    shell = StateModel(dtype); original = {n: t.clone() for n, t in shell.state_dict().items()}
    trainer = start(shell, parent); model = trainer.model
    x = torch.linspace(-0.5, 0.5, 64, dtype=dtype).reshape(2, 32).requires_grad_()
    y = model(x); y.float().square().mean().backward()
    _, handle, leaf, _ = trainer._cache["a.weight"]
    assert model.a.packed_weight is model.b.packed_weight
    assert leaf.numel() == handle.scale_count + handle.present_weight_count
    assert torch.count_nonzero(leaf[handle.scale_count:]) == 0
    norm = model.norm(x).detach().float().contiguous()
    dy = (2*y.detach().float()/y.numel()).to(dtype).float().contiguous()
    expected = 2*torch.utils.dlpack.from_dlpack(handle.pv_vjp(norm, dy))
    torch.testing.assert_close(leaf.grad, expected, rtol=3e-5, atol=3e-5)
    if basis == "Identity":
        torch.testing.assert_close(leaf.grad[handle.scale_count:].reshape(handle.rows, handle.columns),
                                   2*dy.double().t().mm(norm.double()), rtol=3e-5, atol=3e-5)
    dense = torch.utils.dlpack.from_dlpack(handle.gather(torch.arange(handle.rows)))
    rx = x.detach().clone().requires_grad_()
    reference = 2*(torch.nn.functional.linear(model.norm(rx).float(), dense).to(dtype) + model.a.bias.to(dtype))
    reference.float().square().mean().backward()
    tolerance = 3e-5 if dtype == torch.float32 else 5*torch.finfo(dtype).eps
    torch.testing.assert_close(x.grad, rx.grad, rtol=tolerance, atol=tolerance)
    before = trainer.checkpoint(); changed_scales, changed_codes = trainer.step([10]*32)
    assert trainer.revision == 1 and trainer.checkpoint() != before
    assert changed_scales > 0 and changed_codes > 0
    for n, t in shell.state_dict().items(): assert torch.equal(t, original[n])
    trainer.close(); assert model._tritium_native_unpublished_codes
    assert not hasattr(model, "_tritium_native_artifact")


def test_raw_checkpoint_restart_matches_full_codes_moments_and_outputs(tmp_path):
    torch.manual_seed(23); parent = _convert(tmp_path, _prepared()); request = plan(parent)
    a = start(StateModel(), parent, request); b = None
    x = torch.linspace(-1, 1, 64).reshape(2, 32)
    initial = a.model(x).square().mean().item(); movements = 0
    for i in range(3):
        movements += update(a, i, x)[1]
        if b is not None:
            update(b, i, x); assert a.checkpoint() == b.checkpoint()
            with torch.no_grad(): assert torch.equal(a.model(x), b.model(x))
        if i == 0: b = start(StateModel(), parent, request, checkpoint=a.checkpoint(), load_run_id="pv-restart")
    with torch.no_grad(): final = a.model(x).square().mean().item()
    assert movements > 0 and final < initial
    print(f"NATIVE_PYTHON_PV_LOSS initial={initial} final={final} changed_units={movements}")
    a.close(); b.close(); parent.verify()


def test_record_publish_standalone_export_and_pv_parent_chaining(tmp_path):
    parent = _convert(tmp_path, _prepared()); before = (Path(parent.output_dir)/"model.trit").read_bytes()
    record = recording(tmp_path); trainer = start(StateModel(), parent, recording=record)
    x = torch.linspace(-1, 1, 64).reshape(2, 32)
    with pytest.raises(ValueError, match="incomplete"): publish(trainer, tmp_path/"early")
    for i in range(3): update(trainer, i, x)
    with torch.no_grad(): expected = trainer.model(x)
    with pytest.raises(ValueError, match="overlap"): publish(trainer, Path(trainer._recording_work.name)/"bad-child")
    child = publish(trainer, tmp_path/"child"); assert isinstance(child, NativeModelRefinement)
    assert child.recovery_recipe == "alternating-pv" and not child.quality_qualified and not child.objective_authenticated
    assert sorted(p.name for p in (tmp_path/"child").iterdir()) == ["checkpoint.tnpv", "evidence.jsonl", "gradients.tnpj", "model.trit"]
    audit = json.loads(child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000))
    assert audit["scope"] == "parent-backed-numerical-replay"
    loaded = load(child, model=StateModel(), level="pv-child", load_run_id="pv-load-child", inplace=False)
    with torch.no_grad(): assert torch.equal(loaded(x), expected)
    assert loaded.a.packed_weight is loaded.b.packed_weight
    copied = export(child, tmp_path/"exported"); assert copied.package_id == child.package_id
    trainer.close(); assert list(Path(record["work_dir"]).iterdir()) == []
    assert (Path(parent.output_dir)/"model.trit").read_bytes() == before
    chained = start(StateModel(), child, plan(child, updates=1, level="pv-child"), level="pv-child", recording=record)
    update(chained, 0, x); second = publish(chained, tmp_path/"second"); chained.close()
    second.verify_with_parent(child, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    hidden_parent = Path(parent.output_dir).with_name("hidden-parent"); hidden_child = tmp_path/"hidden-child"
    Path(parent.output_dir).rename(hidden_parent); (tmp_path/"child").rename(hidden_child)
    try:
        standalone = load_native_module(StateModel(), copied, level="pv-child", load_run_id="standalone", inplace=False)
        with torch.no_grad(): assert torch.equal(standalone(x), expected)
    finally:
        hidden_parent.rename(parent.output_dir); hidden_child.rename(tmp_path/"child")


def test_tied_embedding_padding_and_head_share_dual_gradient_leaf(tmp_path):
    from test_owner_capture import _source, TiedModel
    source = _source(tmp_path); parent = convert_native_module(source.prepared, tmp_path/"parent", "pv-tied", **json.loads(source.options_json))
    trainer = start(TiedModel(), parent, plan(parent, "embed.weight")); weight = trainer.model.embed.packed_weight
    weight.gather(torch.tensor(0), torch.float32, padding=0).sum().backward()
    leaf = trainer._cache["embed.weight"][2]; assert torch.count_nonzero(leaf.grad) == 0
    trainer.zero_grad(); first, second = trainer.model(torch.tensor([0, 1, 0])); (first.square().mean()+second.square().mean()).backward()
    assert len(trainer._cache) == 1 and trainer.model.head.packed_weight is weight
    assert torch.isfinite(trainer._cache["embed.weight"][2].grad).all()
    assert trainer.step([10]*32)[1] > 0; trainer.close()


def test_stale_higher_order_budget_nonfinite_and_profile_refusals(tmp_path):
    parent = _convert(tmp_path, _prepared()); trainer = start(StateModel(), parent)
    x = torch.linspace(-1, 1, 64).reshape(2, 32).requires_grad_()
    stale = trainer.model(x).sum(); trainer.model(x).square().sum().backward(); trainer.step([10]*32)
    before = trainer.checkpoint()
    with pytest.raises(RuntimeError, match="stale"): stale.backward()
    with pytest.raises(RuntimeError, match="backward failed"): trainer.step([11]*32)
    trainer.zero_grad()
    with pytest.raises(RuntimeError, match="first-order"): torch.autograd.grad(trainer.model(x).sum(), x, create_graph=True)
    trainer.zero_grad(); trainer.model(x).sum().backward(); leaf = trainer._cache["a.weight"][2]
    leaf.grad[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"): trainer.step([11]*32)
    assert trainer.checkpoint() == before
    with pytest.raises(NotImplementedError, match="not implemented"): trainer.save_resume(tmp_path/"bundle", limits={})
    trainer.close()
    with pytest.raises(ValueError, match="AlternatingPv"): start(StateModel(), parent, scale_plan(parent))
    with pytest.raises(ValueError, match="ScaleOnly"):
        native_scale_training(StateModel(), parent, level="full", plan=plan(parent), load_run_id="bad", max_snapshot_bytes=10000, max_parent_bytes=1_000_000, max_saved_input_bytes=10000, _pv=True)
    tiny = start(StateModel(), parent, max_saved_input_bytes=1)
    with pytest.raises(ValueError, match="saved-input"): tiny.model(x)
    assert tiny.saved_input_bytes == 0; tiny.close()
    work = recording(tmp_path)
    with pytest.raises(ValueError, match="zero origin"): start(StateModel(), parent, recording=work, checkpoint=before)
    assert list(Path(work["work_dir"]).iterdir()) == []


def test_snapshot_cap_preserves_gradients_and_graph_lease_releases(tmp_path):
    parent = _convert(tmp_path, _prepared()); probe = start(StateModel(), parent)
    payload = probe._backend.frame("a.weight", 0, False)[0].payload_bytes; probe.close()
    trainer = start(StateModel(), parent, max_snapshot_bytes=payload)
    output = trainer.model(torch.linspace(-1, 1, 32).unsqueeze(0)); assert trainer.saved_input_bytes > 0
    output.sum().backward(); before = trainer.checkpoint()
    with pytest.raises(ValueError, match="budget"): trainer.step([10]*32)
    assert trainer.checkpoint() == before and trainer._cache["a.weight"][2].grad is not None
    del output; gc.collect(); assert trainer.saved_input_bytes == 0
    trainer.close()


def test_tiny_hf_real_pv_backward_publish_and_generated_token_parity(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    torch.manual_seed(43); source = _hf_source(tmp_path)
    parent = convert_native_module(source.prepared, tmp_path/"hf-parent", "hf-pv-parent", **json.loads(source.options_json))
    catalog = json.loads(parent.asset("tritium-torch-state.json")); owners = [e["owner"] for e in catalog["state"] if e["disposition"] == "fitted"]
    trainer = start(LlamaForCausalLM(source.prepared.model.config), parent, plan(parent, owners, updates=2), recording=recording(tmp_path))
    tokens = torch.tensor([[1, 2, 3]]); movements = 0
    for i in range(2):
        trainer.zero_grad(); trainer.model(tokens, labels=tokens, use_cache=False).loss.backward(); movements += trainer.step([10+i]*32)[1]
    assert movements > 0
    with torch.no_grad(): expected = trainer.model(tokens, use_cache=False).logits
    child = publish(trainer, tmp_path/"hf-child"); child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    loaded = load(child, model=LlamaForCausalLM(source.prepared.model.config), level="pv-child", load_run_id="hf-pv-load", inplace=False)
    assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    with torch.no_grad():
        assert torch.equal(loaded(tokens, use_cache=False).logits, expected)
        ids = loaded.generate(tokens, max_new_tokens=2, do_sample=False, use_cache=False)
    copied = export(child, tmp_path/"hf-export"); trainer.close()
    again = load(copied, model=LlamaForCausalLM(source.prepared.model.config), level="pv-child", load_run_id="hf-export-load", inplace=False)
    with torch.no_grad(): assert torch.equal(again.generate(tokens, max_new_tokens=2, do_sample=False, use_cache=False), ids)
