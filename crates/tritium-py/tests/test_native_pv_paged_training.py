"""Installed real paged CPU training/recovery/publication; not model qualification."""
import json
from pathlib import Path
import pytest
import torch
from tritium import NativePvPagedTrainingLimits, NativePvPagedResumeBundle
from tritium.torch import native_pv_paged_training, load, calibrate, convert
from test_native_module_source import StateModel, _prepared, _convert, _fit
from test_native_pv_training import plan, start as resident, update, publish as resident_publish
from test_native_refinement import recording, limits as old_publication_limits
from test_native_resume import limits as bundle_limits
from test_native_pv_paged_verify import _limits as verification_limits


def caps(**changes):
    args = dict(max_owner_read_bytes=1_000_000, max_owner_resident_bytes=1_000_000,
                max_gradient_working_bytes=1_000_000, max_workspace_bytes=16_000_000,
                max_journal_bytes=2_000_000, max_frame_bytes=2_000_000)
    args.update(changes)
    return NativePvPagedTrainingLimits(**args)


def start(shell, parent, work, request=None, **changes):
    args = dict(level="full", plan=plan(parent) if request is None else request,
                load_run_id="paged-load", workspace_dir=work, limits=caps(),
                max_snapshot_bytes=1_000_000, max_saved_input_bytes=10_000_000)
    args.update(changes)
    return native_pv_paged_training(shell, parent, **args)


def publish(trainer, output, **changes):
    limits = old_publication_limits()
    for name in ("max_parent_bytes", "max_replay_working_bytes", "max_journal_bytes"):
        limits.pop(name)
    args = dict(machine="synthetic-installed-wheel", validation=[dict(content=[30]*32, parent_loss=1., child_loss=2.)], limits=limits)
    args.update(changes)
    return trainer.publish(output, "python-pv-child", **args)


def reopen(bundle):
    return NativePvPagedResumeBundle.open(bundle.output_dir, bundle_id=bundle.bundle_id, **bundle_limits())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("codec", ["D2", "B3"])
@pytest.mark.parametrize("basis", ["Identity", {"SignedRht": dict(block=32, seed=13, domain=29)}])
def test_real_training_streamed_resume_and_child_match_resident(tmp_path, dtype, codec, basis):
    torch.manual_seed(59)
    action = _fit(); action["Fit"]["plan"]["layout"].update(codec=codec, basis=basis)
    parent = _convert(tmp_path, _prepared(StateModel(dtype)), actions={"a.weight": action})
    before = {p: p.read_bytes() for p in Path(parent.output_dir).iterdir()}
    request = plan(parent); x = torch.linspace(-1, 1, 64, dtype=dtype).reshape(2, 32)
    a = resident(StateModel(dtype), parent, request, recording=recording(tmp_path))
    shell = StateModel(dtype); shell_state = {n: t.clone() for n, t in shell.state_dict().items()}
    work = tmp_path/"paged"; b = start(shell, parent, work, request)
    assert b.model.a.packed_weight is b.model.b.packed_weight
    assert b.workspace_report["files"] == 2 and not list(work.iterdir())
    with pytest.raises(ValueError, match="save_resume"):
        b.checkpoint()
    with pytest.raises(ValueError, match="incomplete"):
        publish(b, tmp_path/"early")
    assert update(a, 0, x) == update(b, 0, x)
    expected = a.save_resume(tmp_path/"resident-prefix", limits=bundle_limits())
    saved = b.save_resume(tmp_path/"paged-prefix", limits=bundle_limits())
    assert isinstance(saved, NativePvPagedResumeBundle) and saved.bundle_id == expected.bundle_id
    assert not saved.quality_qualified and not saved.objective_authenticated and saved.warnings == []
    b.close(); assert not work.exists() and b.workspace_report["cleaned"] and not b._original_handles
    b = start(StateModel(dtype), parent, tmp_path/"resumed", request, plan=None, resume=reopen(saved))
    assert b.revision == 1
    for i in (1, 2):
        assert update(a, i, x) == update(b, i, x)
        with torch.no_grad():
            assert torch.equal(a.model(x), b.model(x))
    ref = resident_publish(a, tmp_path/"reference")
    child = publish(b, tmp_path/"child")
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(child.output_dir)/name).read_bytes() == (Path(ref.output_dir)/name).read_bytes()
    audit = json.loads(child.verify_paged_with_parent(parent, tmp_path/"audit", limits=verification_limits(child)))
    assert audit["workspace_cleaned"] and not audit["quality_qualified"] and not audit["objective_authenticated"]
    loaded = load(child, model=StateModel(dtype), level="pv-child", load_run_id="paged-child", inplace=False)
    with torch.no_grad(): assert torch.equal(loaded(x), b.model(x))
    assert b.workspace_report["peak_files"] <= 4
    a.close(); b.close(); assert not (tmp_path/"resumed").exists()
    assert all(torch.equal(t, shell_state[n]) for n, t in shell.state_dict().items())
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("name", ["max_owner_read_bytes", "max_owner_resident_bytes", "max_gradient_working_bytes",
                                  "max_workspace_bytes", "max_journal_bytes", "max_frame_bytes", "max_recording_working_bytes"])
def test_caps_are_exact_positive_integers_and_frozen(name):
    with pytest.raises(ValueError): caps(**{name: 0})
    for bad in (True, 1., "1", -1):
        with pytest.raises((TypeError, OverflowError)): caps(**{name: bad})
    value = caps()
    with pytest.raises(AttributeError): setattr(value, name, 1)


@pytest.mark.parametrize("change", ["owner-read", "owner-state", "workspace", "journal", "frame", "packed"])
def test_begin_caps_release_all_private_files_and_preserve_source(tmp_path, change):
    parent = _convert(tmp_path, _prepared()); work = tmp_path/"refused"
    name = {"owner-read":"max_owner_read_bytes", "owner-state":"max_owner_resident_bytes",
            "workspace":"max_workspace_bytes", "journal":"max_journal_bytes", "frame":"max_frame_bytes"}
    args = dict(max_snapshot_bytes=1) if change == "packed" else dict(limits=caps(**{name[change]: 1}))
    with pytest.raises(ValueError): start(StateModel(), parent, work, **args)
    assert not work.exists(); parent.verify()


def test_step_window_gradient_and_source_refusal_never_advance_or_leak(tmp_path):
    parent = _convert(tmp_path, _prepared()); work = tmp_path/"paged"
    b = start(StateModel(), parent, work); x = torch.linspace(-1, 1, 64).reshape(2, 32)
    zero = b.save_resume(tmp_path/"zero", limits=bundle_limits()).bundle_id
    b.model(x).square().mean().backward()
    with pytest.raises(ValueError, match="window"): b.step([99]*32)
    leaf = b._cache["a.weight"][2]; old = leaf.grad.clone(); leaf.grad.fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite|overflow"): b.step([10]*32)
    leaf.grad.copy_(old)
    path = Path(parent.output_dir)/"model.trit"; original = path.read_bytes()
    changed = bytearray(original); changed[-1] ^= 1; path.write_bytes(changed)
    with pytest.raises(ValueError): b.step([10]*32)
    assert path.read_bytes() == changed and b.revision == 0 and b.workspace_report["files"] == 2
    path.write_bytes(original)
    assert b.save_resume(tmp_path/"still-zero", limits=bundle_limits()).bundle_id == zero
    assert b.step([10]*32)[1] > 0
    b.close(); assert not work.exists()
    low = start(StateModel(), parent, tmp_path/"low", limits=caps(max_gradient_working_bytes=1))
    low.model(x).square().mean().backward()
    with pytest.raises(ValueError, match="working cap"): low.step([10]*32)
    assert low.revision == 0 and low.workspace_report["files"] == 2
    low.close()


def test_no_clobber_overlap_foreign_files_and_replaced_workspace_are_preserved(tmp_path):
    parent = _convert(tmp_path, _prepared())
    existing = tmp_path/"foreign"; existing.mkdir(); marker = existing/"marker"; marker.write_text("keep")
    with pytest.raises(FileExistsError): start(StateModel(), parent, existing)
    assert marker.read_text() == "keep"
    for work in (Path(parent.output_dir)/"nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"): start(StateModel(), parent, work)
    work = tmp_path/"paged"; b = start(StateModel(), parent, work)
    (work/"foreign").write_text("keep")
    with pytest.warns(RuntimeWarning, match="Cleanup"): b.close()
    assert (work/"foreign").read_text() == "keep" and not b.workspace_report["cleaned"]
    work = tmp_path/"replace"; moved = tmp_path/"moved"; b = start(StateModel(), parent, work)
    work.rename(moved); work.mkdir(); (work/"foreign").write_text("keep")
    x = torch.ones(2, 32); b.model(x).square().mean().backward()
    with pytest.raises(ValueError, match="identity"): b.step([10]*32)
    with pytest.warns(RuntimeWarning, match="identity"): b.close()
    assert moved.exists() and (work/"foreign").read_text() == "keep"


def test_actual_public_convert_tied_embedding_repeated_uses_and_unqualified_chaining(tmp_path):
    from test_owner_phases import _public_source, _recipe
    from test_owner_capture import TiedModel, _window, _objective
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    parent = convert(source, calibration, output_dir=tmp_path/"parent", run_id="public-paged-parent")
    request = plan(parent, owners="embed.weight", updates=2)
    a = resident(TiedModel(), parent, request, recording=recording(tmp_path))
    b = start(TiedModel(), parent, tmp_path/"tied", request)
    assert b.model.embed is b.model.embed_alias and b.model.embed.packed_weight is b.model.head.packed_weight
    for i in range(2):
        observed = []
        for t in (a, b):
            t.zero_grad(); _objective(t.model, _window()).backward(); observed.append(t.step([10+i]*32))
        assert observed[0] == observed[1]
    ref = resident_publish(a, tmp_path/"tied-reference"); child = publish(b, tmp_path/"tied-child")
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(ref.output_dir)/name).read_bytes() == (Path(child.output_dir)/name).read_bytes()
    a.close(); b.close()
    request = plan(child, owners="embed.weight", updates=1, level="pv-child")
    c = start(TiedModel(), child, tmp_path/"chained", request, level="pv-child")
    _objective(c.model, _window()).backward(); c.step([10]*32)
    second = publish(c, tmp_path/"second"); c.close()
    second.verify_paged_with_parent(child, tmp_path/"chain-audit", limits=verification_limits(second))
    assert not second.quality_qualified and not second.objective_authenticated


@pytest.mark.parametrize("cursor", [0, 3])
def test_zero_complete_prefixes_and_changed_bundle_or_plan_refuse_without_workspace_leaks(tmp_path, cursor):
    parent = _convert(tmp_path, _prepared()); b = start(StateModel(), parent, tmp_path/"paged-work")
    x = torch.linspace(-1, 1, 64).reshape(2, 32)
    for i in range(cursor): update(b, i, x)
    saved = b.save_resume(tmp_path/"saved", limits=bundle_limits()); b.close()
    changed_plan = plan(parent); changed_plan["child_level"] = "other"
    with pytest.raises(ValueError, match="plan differs"):
        start(StateModel(), parent, tmp_path/"bad-plan", changed_plan, resume=saved)
    assert not (tmp_path/"bad-plan").exists()
    path = Path(saved.output_dir)/"resume.tnpb"; original = path.read_bytes(); changed = bytearray(original); changed[-1] ^= 1; path.write_bytes(changed)
    with pytest.raises(ValueError): start(StateModel(), parent, tmp_path/"bad-source", resume=saved)
    assert path.read_bytes() == changed and not (tmp_path/"bad-source").exists()
    path.write_bytes(original)
    c = start(StateModel(), parent, tmp_path/"resumed", resume=reopen(saved), plan=None)
    assert c.revision == cursor
    for i in range(cursor, 3): update(c, i, x)
    child = publish(c, tmp_path/"child"); child.verify()
    with pytest.raises(ValueError): publish(c, tmp_path/"child")
    c.close(); assert not (tmp_path/"resumed").exists()


def test_unused_owner_decay_and_multilevel_inventory_match_resident(tmp_path):
    class Unused(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a = torch.nn.Linear(32, 1); self.b = torch.nn.Linear(32, 1)
        def forward(self, x): return self.a(x)
    torch.manual_seed(67)
    parent = _convert(tmp_path, _prepared(Unused()), actions={"a.weight": _fit(), "b.weight": _fit()})
    request = plan(parent, owners=["a.weight", "b.weight"], updates=1)
    a = resident(Unused(), parent, request, recording=recording(tmp_path))
    b = start(Unused(), parent, tmp_path/"paged", request)
    x = torch.ones(2, 32)
    assert update(a, 0, x) == update(b, 0, x) and "b.weight" not in b._cache
    reference = resident_publish(a, tmp_path/"reference"); child = publish(b, tmp_path/"child")
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(reference.output_dir)/name).read_bytes() == (Path(child.output_dir)/name).read_bytes()
    a.close(); b.close()


@pytest.mark.parametrize("failure", ["stale", "higher-order", "input-version", "preserved-state"])
def test_shared_torch_graph_guards_and_owned_work_retirement(tmp_path, failure):
    parent = _convert(tmp_path, _prepared()); work = tmp_path/"paged-work"; b = start(StateModel(), parent, work)
    x = torch.ones(2, 32, requires_grad=True); y = b.model(x)
    if failure == "stale":
        y.square().mean().backward(retain_graph=True); b.step([10]*32)
        with pytest.raises(RuntimeError, match="stale"): y.sum().backward()
    elif failure == "higher-order":
        with pytest.raises(RuntimeError, match="first-order"): torch.autograd.grad(y.sum(), x, create_graph=True)
    elif failure == "input-version":
        with torch.no_grad(): x.add_(1)
        with pytest.raises(RuntimeError, match="modified|version"): y.sum().backward()
    else:
        with torch.no_grad(): b.model.a.bias.add_(1)
        with pytest.raises(RuntimeError, match="preserved state"): b.step([10]*32)
    if failure == "preserved-state":
        with pytest.raises(RuntimeError, match="preserved state"): b.close()
    else:
        b.close()
    assert not b.active and not b._original_handles
    assert not work.exists() and b.workspace_report["cleaned"] and b.workspace_report["files"] == 0


def test_tiny_hf_multiowner_paged_backward_publish_reload_and_generation(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    from tritium.torch import convert_native_module
    source = _hf_source(tmp_path)
    parent = convert_native_module(source.prepared, tmp_path/"hf-parent", "hf-paged-parent", **json.loads(source.options_json))
    catalog = json.loads(parent.asset("tritium-torch-state.json"))
    owners = [e["owner"] for e in catalog["state"] if e["disposition"] == "fitted"]
    assert len(owners) > 1
    request = plan(parent, owners, updates=2)
    b = start(LlamaForCausalLM(source.prepared.model.config), parent, tmp_path/"hf-work", request)
    tokens = torch.tensor([[1, 2, 3]]); movements = 0
    for i in range(2):
        b.zero_grad(); b.model(tokens, labels=tokens, use_cache=False).loss.backward(); movements += b.step([10+i]*32)[1]
    assert movements > 0
    generation = dict(max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=False,
                      attention_mask=torch.ones_like(tokens), pad_token_id=0)
    with torch.no_grad():
        expected = b.model(tokens, use_cache=False).logits
        expected_ids = b.model.generate(tokens, **generation)
    child = publish(b, tmp_path/"hf-child")
    child.verify_paged_with_parent(parent, tmp_path/"hf-audit", limits=verification_limits(child))
    loaded = load(child, model=LlamaForCausalLM(source.prepared.model.config), level="pv-child", load_run_id="hf-paged-load", inplace=False)
    assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    with torch.no_grad():
        assert torch.equal(loaded(tokens, use_cache=False).logits, expected)
        ids = loaded.generate(tokens, **generation)
    assert ids.shape == (1, 5) and torch.equal(ids, expected_ids)
    b.close(); assert not (tmp_path/"hf-work").exists()


@pytest.mark.parametrize("name", ["checkpoint", "recording", "_paging", "max_parent_bytes", "unknown"])
def test_paged_factory_rejects_resident_or_private_options_without_loading(tmp_path, name):
    parent = _convert(tmp_path, _prepared()); work = tmp_path/"refused"
    value = {"checkpoint": b"not a checkpoint", "recording": None,
             "_paging": (str(work), caps()), "max_parent_bytes": 1, "unknown": 1}[name]
    with pytest.raises(TypeError, match="unknown native training load limit"):
        start(StateModel(), parent, work, **{name: value})
    assert not work.exists(); parent.verify()


@pytest.mark.parametrize("pv", [False, True])
def test_resident_factories_cannot_bind_private_paging_from_load_options(tmp_path, pv):
    from tritium.torch import native_scale_training, native_pv_training
    from test_native_scale_training import plan as scale_plan
    parent = _convert(tmp_path, _prepared()); work = tmp_path/"refused"
    begin = native_pv_training if pv else native_scale_training
    with pytest.raises(TypeError, match="unknown native training load limit"):
        begin(StateModel(), parent, level="full", plan=plan(parent) if pv else scale_plan(parent),
              load_run_id="refused-private-mode", max_parent_bytes=1, max_snapshot_bytes=1_000_000,
              max_saved_input_bytes=1_000_000, _paging=(str(work), caps()))
    assert not work.exists(); parent.verify()
