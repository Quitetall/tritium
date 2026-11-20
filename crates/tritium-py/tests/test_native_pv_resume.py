"""Installed P/V durable recovery, not model-quality or durability qualification."""
import json
from pathlib import Path
import pytest
import torch
from tritium import NativePvResumeBundle, NativeScaleResumeBundle
from tritium.torch import native_pv_training, native_scale_training, load, convert_native_module
from test_native_module_source import StateModel, _prepared, _convert
from test_native_pv_training import plan, start, update, publish
from test_native_refinement import recording
from test_native_resume import limits


def reopen(saved, **kwargs):
    return NativePvResumeBundle.open(saved.output_dir, bundle_id=saved.bundle_id, **{**limits(), **kwargs})


def recover(shell, parent, saved, root, **kwargs):
    options = dict(level=json.loads(saved.plan_json)["parent"]["level"], load_run_id="pv-recovery",
                   max_snapshot_bytes=1_000_000, max_parent_bytes=4_000_000, max_saved_input_bytes=10_000_000,
                   recording=recording(root), resume=saved)
    options.update(kwargs)
    return native_pv_training(shell, parent, **options)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("level", ["compact", "full"])
def test_recorded_pv_restart_has_exact_codes_moments_journal_outputs_and_parent(tmp_path, dtype, level):
    torch.manual_seed(67); parent = _convert(tmp_path, _prepared(StateModel(dtype)))
    request = plan(parent, level=level); x = torch.linspace(-1, 1, 64, dtype=dtype).reshape(2, 32)
    original = (Path(parent.output_dir)/"model.trit").read_bytes()
    baseline = start(StateModel(dtype), parent, request, level=level, recording=recording(tmp_path))
    for i in range(3): update(baseline, i, x)
    expected_cp = baseline.checkpoint()
    with torch.no_grad(): expected = baseline.model(x)
    reference = publish(baseline, tmp_path/"reference"); baseline.close()
    trainer = start(StateModel(dtype), parent, request, level=level, recording=recording(tmp_path))
    assert update(trainer, 0, x)[1] > 0
    checkpoint = trainer.checkpoint(); old_work = Path(trainer._recording_work.name)
    saved = trainer.save_resume(tmp_path/"saved", limits=limits())
    assert isinstance(saved, NativePvResumeBundle) and not isinstance(saved, NativeScaleResumeBundle)
    assert saved.completed_updates == 1 and saved.admission == "byte-consistent-unqualified"
    assert not saved.quality_qualified and not saved.objective_authenticated and saved.warnings == []
    assert sorted(p.name for p in (tmp_path/"saved").iterdir()) == ["resume.tnpb"]
    assert saved.bundle_bytes == (tmp_path/"saved/resume.tnpb").stat().st_size
    with pytest.raises(AttributeError): saved.bundle_id = "0"*64
    trainer.close(); assert not old_work.exists(); saved.verify()
    shell = StateModel(dtype); original_shell = {n: v.clone() for n, v in shell.state_dict().items()}
    resumed = recover(shell, parent, reopen(saved), tmp_path)
    assert resumed.revision == 1 and resumed.checkpoint() == checkpoint
    assert resumed.model.a.packed_weight is resumed.model.b.packed_weight
    for i in range(1, 3): update(resumed, i, x)
    assert resumed.checkpoint() == expected_cp
    with torch.no_grad(): assert torch.equal(resumed.model(x), expected)
    child = publish(resumed, tmp_path/"child"); child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    for name in ["checkpoint.tnpv", "gradients.tnpj"]:
        assert (Path(child.output_dir)/name).read_bytes() == (Path(reference.output_dir)/name).read_bytes()
    loaded = load(child, model=StateModel(dtype), level="pv-child", load_run_id="recovered-child", inplace=False)
    with torch.no_grad(): assert torch.equal(loaded(x), expected)
    resumed.close(); assert list(Path(recording(tmp_path)["work_dir"]).iterdir()) == []
    for name, value in shell.state_dict().items(): assert torch.equal(value, original_shell[name])
    assert (Path(parent.output_dir)/"model.trit").read_bytes() == original


@pytest.mark.parametrize("cursor", [0, 3])
def test_zero_and_complete_cursors_save_reopen_continue_without_old_work(tmp_path, cursor):
    parent = _convert(tmp_path, _prepared()); trainer = start(StateModel(), parent, recording=recording(tmp_path))
    x = torch.linspace(-1, 1, 64).reshape(2, 32)
    for i in range(cursor): update(trainer, i, x)
    cp = trainer.checkpoint(); saved = trainer.save_resume(tmp_path/"saved", limits=limits()); trainer.close()
    resumed = recover(StateModel(), parent, reopen(saved), tmp_path)
    assert resumed.revision == cursor and resumed.checkpoint() == cp
    for i in range(cursor, 3): update(resumed, i, x)
    publish(resumed, tmp_path/"child").verify(); resumed.close()


def test_pending_gradients_are_not_saved_and_save_keeps_live_recording_appendable(tmp_path):
    parent = _convert(tmp_path, _prepared()); trainer = start(StateModel(), parent, recording=recording(tmp_path))
    x = torch.linspace(-1, 1, 64).reshape(2, 32); before = trainer.checkpoint()
    trainer.model(x).square().mean().backward()
    saved = trainer.save_resume(tmp_path/"saved", limits=limits()); assert saved.completed_updates == 0
    trainer.step([10]*32); expected = trainer.checkpoint()
    resumed = recover(StateModel(), parent, reopen(saved), tmp_path); assert resumed.checkpoint() == before
    update(resumed, 0, x); assert resumed.checkpoint() == expected
    trainer.close(); resumed.close()


def test_wrong_profile_plan_checkpoint_caps_overlap_and_changed_bytes_fail_closed(tmp_path):
    parent = _convert(tmp_path, _prepared()); trainer = start(StateModel(), parent, recording=recording(tmp_path))
    x = torch.linspace(-1, 1, 64).reshape(2, 32); update(trainer, 0, x)
    saved = trainer.save_resume(tmp_path/"saved", limits=limits()); cp = trainer.checkpoint()
    for output in [Path(parent.output_dir)/"bad", Path(trainer._recording_work.name)/"bad"]:
        with pytest.raises(ValueError, match="overlap"): trainer.save_resume(output, limits=limits())
    with pytest.raises(ValueError): trainer.save_resume(tmp_path/"saved", limits=limits())
    with pytest.raises(ValueError): trainer.save_resume(tmp_path/"small", limits={**limits(), "max_bundle_bytes": 1})
    assert not (tmp_path/"small").exists(); trainer.close()
    with pytest.raises(ValueError): reopen(saved, max_working_bytes=1)
    wrong = plan(parent); wrong["child_level"] = "wrong"
    for override in [dict(plan=wrong), dict(checkpoint=cp), dict(max_snapshot_bytes=1), dict(recording=None)]:
        with pytest.raises((ValueError, RuntimeError)): recover(StateModel(), parent, saved, tmp_path, **override)
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir()) == []
    with pytest.raises(TypeError, match="NativeScaleResumeBundle"):
        native_scale_training(StateModel(), parent, level="full", resume=saved, load_run_id="wrong-family", max_snapshot_bytes=10000, max_parent_bytes=1_000_000, max_saved_input_bytes=10000, recording=recording(tmp_path))
    with pytest.raises(ValueError): NativeScaleResumeBundle.open(saved.output_dir, bundle_id=saved.bundle_id, **limits())
    path = tmp_path/"saved/resume.tnpb"; changed = bytearray(path.read_bytes()); changed[-40] ^= 1; path.write_bytes(changed)
    with pytest.raises(ValueError): saved.verify()
    with pytest.raises(ValueError): recover(StateModel(), parent, saved, tmp_path)
    assert list(Path(recording(tmp_path)["work_dir"]).iterdir()) == []


def test_tiny_tied_hf_recorded_resume_matches_uninterrupted_loss_trajectory_and_child(tmp_path):
    from transformers import LlamaForCausalLM
    from test_owner_capture import _hf_source
    torch.manual_seed(79); source = _hf_source(tmp_path)
    parent = convert_native_module(source.prepared, tmp_path/"hf-parent", "hf-resume-parent", **json.loads(source.options_json))
    catalog = json.loads(parent.asset("tritium-torch-state.json")); owners = [s["owner"] for s in catalog["state"] if s["disposition"] == "fitted"]
    request = plan(parent, owners, updates=2); tokens = torch.tensor([[1, 2, 3]])
    def advance(trainer, ordinal):
        trainer.zero_grad(); trainer.model(tokens, labels=tokens, use_cache=False).loss.backward(); trainer.step([10+ordinal]*32)
    baseline = start(LlamaForCausalLM(source.prepared.model.config), parent, request, recording=recording(tmp_path))
    for i in range(2): advance(baseline, i)
    expected = baseline.checkpoint()
    with torch.no_grad(): logits = baseline.model(tokens, use_cache=False).logits
    reference = publish(baseline, tmp_path/"reference"); baseline.close()
    partial = start(LlamaForCausalLM(source.prepared.model.config), parent, request, recording=recording(tmp_path))
    advance(partial, 0); saved = partial.save_resume(tmp_path/"saved", limits=limits()); partial.close()
    resumed = recover(LlamaForCausalLM(source.prepared.model.config), parent, reopen(saved), tmp_path)
    advance(resumed, 1); assert resumed.checkpoint() == expected
    with torch.no_grad(): assert torch.equal(resumed.model(tokens, use_cache=False).logits, logits)
    child = publish(resumed, tmp_path/"child"); resumed.close()
    for name in ["checkpoint.tnpv", "gradients.tnpj"]:
        assert (Path(child.output_dir)/name).read_bytes() == (Path(reference.output_dir)/name).read_bytes()
    child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
