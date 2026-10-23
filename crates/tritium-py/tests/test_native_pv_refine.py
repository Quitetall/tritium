"""Actual public teacher/data P/V facade; local fixtures, not quality qualification."""
import json
from pathlib import Path
import pytest
import torch
from tritium import NativePvResumeBundle, NativeScaleResumeBundle
from tritium.torch import NativeRefinementConfig as Config, NativePvTrainer, refine, load, native_pv_training
from tritium.torch import native_refinement as native
from tritium.torch.errors import TritiumError
from test_native_refine import (fixture, configuration as scale_configuration, data_limits,
                                training_limits, publish_limits, bundle_limits)


def configuration(parent, training, validation, *, level="full", owners="a.weight", code_lr=0.000001,
                  fraction=0.03125, lr=0.00001, steps=(), **kwargs):
    base = scale_configuration(parent, training, validation, level=level, owners=owners, lr=lr, steps=steps)
    plan = base.plan; optimizer = plan["recipe"]["ScaleOnly"]["optimizer"]
    plan["recipe"] = {"AlternatingPv": dict(continuous_optimizer=optimizer,
        code_optimizer={**optimizer, "lr": code_lr, "beta1": 0.0, "beta2": 0.0},
        max_code_change_fraction=fraction, max_relative_code_change=1.0)}
    plan["child_level"] = "pv-kd-child"
    resources = base.resources
    options = dict(level=level, run_id="public-pv-refine", machine="synthetic-installed-wheel",
                   training_limits=resources["training"], publication_limits=resources["publication"],
                   data_limits=resources["data"], checkpoint_steps=steps,
                   checkpoint_limits=bundle_limits() if steps else None)
    options.update(kwargs)
    return Config(plan, **options)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("level", ["compact", "full"])
def test_public_pv_recipe_uses_real_backward_retention_and_preserves_source(tmp_path, dtype, level):
    teacher, parent, training, validation, work = fixture(tmp_path, dtype)
    teacher.a.eval(); flags = [m.training for m in teacher.modules()]
    state = {n: t.clone() for n, t in teacher.state_dict().items()}; original = (Path(parent.output_dir)/"model.trit").read_bytes()
    cfg = configuration(parent, training, validation, level=level)
    result = refine(parent, teacher=teacher, training=iter(training), validation=iter(validation), config=cfg, work_dir=work, output_dir=tmp_path/"child")
    assert result.recovery_recipe == "alternating-pv" and result.parent_level == level
    assert not result.quality_qualified and not result.objective_authenticated
    assert sorted(p.name for p in (tmp_path/"child").iterdir()) == ["checkpoint.tnpv", "evidence.jsonl", "gradients.tnpj", "model.trit"]
    result.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    loaded = load(result, model=teacher, level="pv-kd-child", load_run_id="pv-kd-load", inplace=False)
    assert loaded.a.packed_weight is loaded.b.packed_weight
    assert [m.training for m in teacher.modules()] == flags
    assert all(torch.equal(t, state[n]) for n, t in teacher.state_dict().items())
    assert (Path(parent.output_dir)/"model.trit").read_bytes() == original and list(work.iterdir()) == []


def test_public_pv_matches_manual_recorded_driver_exact_checkpoint_and_journal(tmp_path):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = configuration(parent, training, validation)
    public = refine(parent, teacher=teacher, training=training, validation=validation, config=cfg, work_dir=work, output_dir=tmp_path/"public")
    guard = native._Teacher(teacher); trainer = None
    try:
        teacher.eval()
        windows = [native._snapshot(batch, cfg.resources["data"])[:2] for batch in validation]
        trainer = native_pv_training(teacher, parent, level=cfg.level, plan=cfg.plan, load_run_id="manual-pv",
            recording=dict(work_dir=work, max_journal_bytes=cfg.resources["publication"]["max_journal_bytes"],
                           max_working_bytes=cfg.resources["publication"]["max_replay_working_bytes"]), **cfg.resources["training"])
        trainer.model.eval(); before = native._evaluate(trainer.model, teacher, guard, windows, cfg)
        for batch, declared in zip(training, cfg.plan["training"]):
            copied, identity, _ = native._snapshot(batch, cfg.resources["data"])
            assert identity == declared["content"]; native._train(trainer, teacher, guard, copied, identity, cfg)
        after = native._evaluate(trainer.model, teacher, guard, windows, cfg)
        observed = [dict(content=w["content"], parent_loss=a, child_loss=b) for w, a, b in zip(cfg.plan["validation"], before, after)]
        manual = trainer.publish(tmp_path/"manual", cfg.run_id, machine=cfg.machine, validation=observed, limits=cfg.resources["publication"])
        for name in ["checkpoint.tnpv", "gradients.tnpj"]:
            assert (Path(public.output_dir)/name).read_bytes() == (Path(manual.output_dir)/name).read_bytes()
    finally:
        if trainer is not None: trainer.close()
        guard.restore()
    assert list(work.iterdir()) == []


def test_declared_pv_recipe_changes_real_codes_and_passes_observed_retention(tmp_path, monkeypatch):
    teacher, parent, training, validation, work = fixture(tmp_path)
    # A bounded fixture-only probe selected this explicit recipe. The production
    # guard and optimizer are unchanged; oversized discrete steps are tested below.
    cfg = configuration(parent, training, validation, code_lr=0.05, fraction=0.03125)
    changes = []; original_step = NativePvTrainer.step
    def observed_step(self, identity):
        observed = original_step(self, identity); changes.append(observed); return observed
    monkeypatch.setattr(NativePvTrainer, "step", observed_step)
    result = refine(parent, teacher=teacher, training=training, validation=validation,
                    config=cfg, work_dir=work, output_dir=tmp_path/"child")
    assert len(changes) == len(training) and sum(codes for _, codes in changes) > 0
    result.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    assert result.recovery_recipe == "alternating-pv" and not result.quality_qualified
    print(f"PUBLIC_PV_KD_CHANGED_UNITS={sum(codes for _, codes in changes)}")


@pytest.mark.parametrize("cursor", [0, 2, 4])
def test_public_pv_checkpoint_interrupt_resume_matches_uninterrupted_original_parent_baseline(tmp_path, cursor):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = configuration(parent, training, validation, steps=(cursor,)); checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir()
    saved = []
    def interrupt(bundle):
        assert isinstance(bundle, NativePvResumeBundle) and not isinstance(bundle, NativeScaleResumeBundle)
        saved.append(bundle); raise RuntimeError("intentional PV interrupt")
    with pytest.raises(RuntimeError, match="intentional PV interrupt"):
        refine(parent, teacher=teacher, training=iter(training), validation=validation, config=cfg,
               work_dir=work, output_dir=tmp_path/"resumed", checkpoint_dir=checkpoints, on_checkpoint=interrupt)
    assert list(work.iterdir()) == [] and not (tmp_path/"resumed").exists()
    original = (Path(saved[0].output_dir)/"resume.tnpb").read_bytes(); calls = []
    resumed = refine(parent, teacher=teacher, training=iter(training), validation=validation, config=cfg,
        work_dir=work, output_dir=tmp_path/"resumed", checkpoint_dir=checkpoints, resume=saved[0], on_checkpoint=calls.append)
    assert calls == [] and (Path(saved[0].output_dir)/"resume.tnpb").read_bytes() == original
    reference = refine(parent, teacher=teacher, training=training, validation=validation,
        config=configuration(parent, training, validation), work_dir=work, output_dir=tmp_path/"reference")
    for name in ["checkpoint.tnpv", "gradients.tnpj"]:
        assert (Path(resumed.output_dir)/name).read_bytes() == (Path(reference.output_dir)/name).read_bytes()
    resumed.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)


def test_regressed_pv_terminal_state_is_rejected_without_weakening_guard_or_erasing_bundle(tmp_path):
    teacher, parent, training, validation, work = fixture(tmp_path); training = training[:1]
    cfg = configuration(parent, training, validation, lr=1000, steps=(1,))
    checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir(); saved = []
    with pytest.raises(TritiumError) as caught:
        refine(parent, teacher=teacher, training=training, validation=validation, config=cfg,
            work_dir=work, output_dir=tmp_path/"child", checkpoint_dir=checkpoints, on_checkpoint=saved.append)
    assert caught.value.code == "predecessor_retention_failed"
    assert caught.value.details["candidate_loss"] > caught.value.details["predecessor_loss"]
    assert not (tmp_path/"child").exists() and list(work.iterdir()) == []
    assert saved[0].completed_updates == 1; saved[0].verify()


@pytest.mark.parametrize("change", ["wrong-train", "extra-validation", "window-cap", "teacher", "input-mutation"])
def test_pv_data_teacher_and_resource_refusals_publish_nothing_and_restore_flags(tmp_path, change):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = configuration(parent, training, validation); flags = [m.training for m in teacher.modules()]; hook = None
    if change == "wrong-train": training = [torch.zeros_like(training[0]), *training[1:]]
    elif change == "extra-validation": validation = validation+[validation[0]]
    elif change == "window-cap": cfg = configuration(parent, training, validation, data_limits={**data_limits(), "max_window_bytes": 1})
    elif change == "teacher":
        with torch.no_grad(): teacher.a.weight.add_(1)
    else: hook = teacher.register_forward_pre_hook(lambda _, args: args[0].add_(0.25))
    try:
        with pytest.raises(ValueError):
            refine(parent, teacher=teacher, training=training, validation=validation, config=cfg, work_dir=work, output_dir=tmp_path/"child")
    finally:
        if hook is not None: hook.remove()
    assert not (tmp_path/"child").exists() and list(work.iterdir()) == []
    assert [m.training for m in teacher.modules()] == flags


def test_profile_and_wrong_bundle_never_fall_back_to_scale(tmp_path, monkeypatch):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = configuration(parent, training, validation); request = cfg.plan
    request["recipe"]["AlternatingPv"]["code_optimizer"]["lr"] = -1
    def forbidden(_): raise AssertionError("P/V admission must never fall back to ScaleOnly")
    with monkeypatch.context() as patch:
        patch.setattr(native._tritium, "_validate_native_scale_plan", forbidden)
        with pytest.raises(ValueError):
            Config(request, level="full", run_id="bad", machine="cpu", training_limits=training_limits(), publication_limits=publish_limits(), data_limits=data_limits())
    from test_native_scale_training import start as scale_start
    from test_native_refinement import recording
    scale = scale_start(teacher, parent, recording=recording(tmp_path))
    saved = scale.save_resume(tmp_path/"scale-saved", limits=bundle_limits()); scale.close()
    with pytest.raises(TypeError, match="NativePvResumeBundle"):
        refine(parent, teacher=teacher, training=training, validation=validation, config=cfg, work_dir=work, output_dir=tmp_path/"child", resume=saved)
    assert not (tmp_path/"child").exists() and list(work.iterdir()) == []


def test_final_pv_teacher_digest_catches_data_write_without_version_change(tmp_path):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = configuration(parent, training, validation, steps=(0,)); checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir()
    version = teacher.a.weight._version; saved = []
    def mutate(bundle):
        saved.append(bundle); teacher.a.weight.data.add_(0.001); assert teacher.a.weight._version == version
    with pytest.raises(ValueError, match="teacher bytes changed"):
        refine(parent, teacher=teacher, training=training, validation=validation, config=cfg,
            work_dir=work, output_dir=tmp_path/"child", checkpoint_dir=checkpoints, on_checkpoint=mutate)
    assert not (tmp_path/"child").exists() and list(work.iterdir()) == []; saved[0].verify()


def test_tiny_hf_multiowner_masked_pv_teacher_refinement(tmp_path):
    from test_owner_capture import _hf_source
    from tritium.torch import convert_native_module
    source = _hf_source(tmp_path); teacher = source.prepared.model
    parent = convert_native_module(source.prepared, tmp_path/"hf-parent", "hf-pv-facade", **json.loads(source.options_json))
    owners = [e["owner"] for e in json.loads(parent.asset("tritium-torch-state.json"))["state"] if e["disposition"] == "fitted"]
    def batch(ids): return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.int64), use_cache=False)
    training = [batch([1, 2, 3]), batch([1, 2, 4])]; validation = [batch([1, 2, 3, 4])]
    cfg = configuration(parent, training, validation, owners=owners, lr=0.000001)
    work = tmp_path/"runtime"; work.mkdir()
    result = refine(parent, teacher=teacher, training=iter(training), validation=iter(validation), config=cfg, work_dir=work, output_dir=tmp_path/"child")
    assert result.recovery_recipe == "alternating-pv" and not result.quality_qualified
    result.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    assert list(work.iterdir()) == []
