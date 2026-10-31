"""Public owner-paged KD/refinement, not model or physical-memory qualification."""
from pathlib import Path
import pytest
import torch
from tritium import NativePvPagedResumeBundle
from tritium.torch import NativePvPagedRefinementConfig as Config, refine, load, calibrate, convert
from tritium.torch.errors import TritiumError
from test_native_refine import fixture, configuration as scale_configuration
from test_native_pv_refine import configuration
from test_native_pv_paged_training import caps
from test_native_pv_gradient_capture import capture_limits


def paged(base, **changes):
    resources = base.resources
    resources["training"].pop("max_parent_bytes")
    for name in ("max_parent_bytes", "max_replay_working_bytes", "max_journal_bytes"):
        resources["publication"].pop(name)
    args = dict(level=base.level, run_id=base.run_id, machine=base.machine,
        training_limits=resources["training"], publication_limits=resources["publication"],
        data_limits=resources["data"], temperature=base.temperature,
        checkpoint_steps=base.checkpoint_steps, checkpoint_limits=resources["checkpoint"],
        paging_limits=caps(), gradient_capture=capture_limits())
    args.update(changes)
    return Config(base.plan, **args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("level", ["compact", "full"])
def test_public_paged_refinement_matches_resident_recording_and_reload(tmp_path, dtype, level):
    teacher, parent, training, validation, work = fixture(tmp_path, dtype)
    teacher.a.eval(); flags = [m.training for m in teacher.modules()]
    original = {n: t.clone() for n, t in teacher.state_dict().items()}
    base = configuration(parent, training, validation, level=level)
    reference = refine(parent, teacher=teacher, training=training, validation=validation,
        config=base, work_dir=work, output_dir=tmp_path/"reference")
    cfg = paged(base)
    result = refine(parent, teacher=teacher, training=iter(training), validation=iter(validation),
        config=cfg, work_dir=work, output_dir=tmp_path/"child")
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(result.output_dir)/name).read_bytes() == (Path(reference.output_dir)/name).read_bytes()
    assert not result.objective_authenticated and not result.quality_qualified
    loaded = load(result, model=teacher, level="pv-kd-child", load_run_id="paged-kd-load", inplace=False)
    expected = load(reference, model=teacher, level="pv-kd-child", load_run_id="resident-kd-load", inplace=False)
    with torch.no_grad(): assert torch.equal(loaded(validation[0]), expected(validation[0]))
    assert loaded.a.packed_weight is loaded.b.packed_weight
    assert [m.training for m in teacher.modules()] == flags
    assert all(torch.equal(t, original[n]) for n, t in teacher.state_dict().items())
    assert list(work.iterdir()) == []


def test_config_freezes_distinct_resource_scopes_and_rejects_implicit_modes(tmp_path):
    teacher, parent, training, validation, work = fixture(tmp_path)
    base = configuration(parent, training, validation); cfg = paged(base)
    for field in ("paging_limits", "gradient_capture", "level"):
        with pytest.raises(AttributeError): setattr(cfg, field, None)
    values = cfg.resources; values["training"]["max_snapshot_bytes"] = 1
    assert cfg.resources["training"]["max_snapshot_bytes"] > 1
    assert "max_parent_bytes" not in cfg.resources["training"]
    assert len(cfg.resources["publication"]) == 7
    for changes in (dict(paging_limits={}), dict(gradient_capture=None),
                    dict(training_limits=base.resources["training"]),
                    dict(publication_limits=base.resources["publication"]), dict(_paging=True)):
        with pytest.raises((TypeError, ValueError)): paged(base, **changes)
    with pytest.raises(ValueError, match="AlternatingPv"):
        paged(scale_configuration(parent, training, validation))
    assert list(work.iterdir()) == []


@pytest.mark.parametrize("cursor", [0, 2, 4])
def test_paged_zero_intermediate_final_checkpoint_resume_matches_uninterrupted(tmp_path, cursor):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = paged(configuration(parent, training, validation, steps=(cursor,)))
    checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir(); saved = []
    def interrupted(bundle):
        assert isinstance(bundle, NativePvPagedResumeBundle)
        saved.append(bundle); raise RuntimeError("intentional paged interrupt")
    with pytest.raises(RuntimeError, match="intentional paged interrupt"):
        refine(parent, teacher=teacher, training=training, validation=validation, config=cfg,
            work_dir=work, output_dir=tmp_path/"resumed", checkpoint_dir=checkpoints, on_checkpoint=interrupted)
    assert list(work.iterdir()) == [] and not (tmp_path/"resumed").exists()
    original = (Path(saved[0].output_dir)/"resume.tnpb").read_bytes(); observed = []
    result = refine(parent, teacher=teacher, training=iter(training), validation=validation,
        config=cfg, work_dir=work, output_dir=tmp_path/"resumed", checkpoint_dir=checkpoints,
        resume=saved[0], on_checkpoint=observed.append)
    reference = refine(parent, teacher=teacher, training=training, validation=validation,
        config=paged(configuration(parent, training, validation)), work_dir=work, output_dir=tmp_path/"reference")
    assert observed == [] and (Path(saved[0].output_dir)/"resume.tnpb").read_bytes() == original
    for name in ("checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(result.output_dir)/name).read_bytes() == (Path(reference.output_dir)/name).read_bytes()
    assert list(work.iterdir()) == []


def test_wrong_resume_kinds_refuse_before_consuming_streams(tmp_path):
    teacher, parent, training, validation, work = fixture(tmp_path)
    cfg = paged(configuration(parent, training, validation)); events = []
    def stream():
        events.append("consumed"); yield training[0]
    for value in (b"raw checkpoint", {}, object()):
        with pytest.raises(TypeError, match="NativePvPagedResumeBundle"):
            refine(parent, teacher=teacher, training=stream(), validation=stream(), config=cfg,
                work_dir=work, output_dir=tmp_path/"child", resume=value)
    assert events == [] and list(work.iterdir()) == [] and not (tmp_path/"child").exists()


@pytest.mark.parametrize("failure", ["source", "missing", "trailing", "window", "spool", "owner", "held-out"])
def test_refusals_restore_teacher_and_retire_owned_workspace_without_child(tmp_path, failure):
    teacher, parent, training, validation, work = fixture(tmp_path)
    flags = [m.training for m in teacher.modules()]
    if failure == "held-out": training = training[:1]
    base = configuration(parent, training, validation, lr=1000 if failure == "held-out" else 0.00001)
    changes = dict(gradient_capture=capture_limits(max_spool_bytes=8)) if failure == "spool" else {}
    if failure == "owner": changes["paging_limits"] = caps(max_owner_resident_bytes=1)
    cfg = paged(base, **changes)
    if failure == "missing": training = training[:-1]
    elif failure == "trailing": training = training + training[:1]
    elif failure == "window": training = [torch.zeros_like(training[0]), *training[1:]]
    elif failure == "source":
        batches = training
        def changed():
            with torch.no_grad(): teacher.a.weight.add_(0.125)
            yield from batches
        training = changed()
    with pytest.raises((ValueError, TritiumError)) as caught:
        refine(parent, teacher=teacher, training=training, validation=validation, config=cfg,
            work_dir=work, output_dir=tmp_path/"child")
    if failure == "held-out": assert caught.value.code == "predecessor_retention_failed"
    assert not (tmp_path/"child").exists() and list(work.iterdir()) == []
    assert [m.training for m in teacher.modules()] == flags


def test_public_convert_tied_embedding_parent_uses_common_paged_refinement(tmp_path):
    from test_owner_phases import _public_source, _recipe
    from test_owner_capture import _window, _objective
    source = _public_source(tmp_path)
    calibration = calibrate(source, [_window()], objective=_objective, owner_recipe=_recipe())
    parent = convert(source, calibration, output_dir=tmp_path/"parent", run_id="paged-kd-parent")
    teacher = source.prepared.model
    training = [torch.tensor([0, 1, 0])]; validation = [torch.tensor([1, 0, 1])]
    base = configuration(parent, training, validation, owners="embed.weight", lr=0.000001)
    work = tmp_path/"runtime"; work.mkdir()
    result = refine(parent, teacher=teacher, training=training, validation=validation,
        config=paged(base), work_dir=work, output_dir=tmp_path/"child")
    loaded = load(result, model=teacher, level="pv-kd-child", load_run_id="paged-tied-load", inplace=False)
    assert loaded.embed.packed_weight is loaded.head.packed_weight
    assert list(work.iterdir()) == [] and not result.quality_qualified


def test_aggregate_code_moments_exceed_owner_cap_without_resident_fallback(tmp_path):
    from test_native_module_source import _prepared, _convert, _fit
    class Many(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 1, bias=False) for _ in range(72)])
        def forward(self, x): return torch.cat([layer(x) for layer in self.layers], dim=-1)
    torch.manual_seed(103); teacher = Many(); owners = [f"layers.{i}.weight" for i in range(72)]
    parent = _convert(tmp_path, _prepared(teacher), actions={n: _fit() for n in owners},
        limits=dict(max_source_bytes=65536, max_asset_bytes=262144, max_retained_blob_bytes=1_000_000, max_package_bytes=2_000_000))
    x = torch.linspace(-1, 1, 64).reshape(2, 32); training = [x]; validation = [x + 0.125]
    cfg = paged(configuration(parent, training, validation, owners=owners, lr=0.000001, steps=(1,)),
                paging_limits=caps(max_owner_resident_bytes=18000))
    # Two f32 code moments alone exceed the admitted largest-owner payload.
    assert sum(p.numel()*8 for p in teacher.parameters()) > cfg.paging_limits.max_owner_resident_bytes
    work = tmp_path/"runtime"; work.mkdir(); checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir(); saved = []
    result = refine(parent, teacher=teacher, training=training, validation=validation, config=cfg,
        work_dir=work, output_dir=tmp_path/"child", checkpoint_dir=checkpoints, on_checkpoint=saved.append)
    assert len(saved) == 1 and isinstance(saved[0], NativePvPagedResumeBundle)
    assert result.recovery_recipe == "alternating-pv" and list(work.iterdir()) == []


@pytest.mark.parametrize("fault", ["foreign", "replacement", "callback-error"])
def test_checkpoint_workspace_faults_preserve_foreign_bytes_and_primary_errors(tmp_path, fault):
    teacher, parent, training, validation, work = fixture(tmp_path)
    training = training[:1]; flags = [m.training for m in teacher.modules()]
    cfg = paged(configuration(parent, training, validation, steps=(1,)))
    checkpoints = tmp_path/"checkpoints"; checkpoints.mkdir(); observed = {}
    def inject(bundle):
        assert isinstance(bundle, NativePvPagedResumeBundle)
        entries = list(work.iterdir()); assert len(entries) == 1 and entries[0].is_dir()
        workspace = entries[0]; observed.update(workspace=workspace, bundle=bundle)
        if fault == "replacement":
            moved = workspace.with_name(workspace.name + "-moved")
            workspace.rename(moved); workspace.mkdir(); observed["moved"] = moved
        foreign = workspace/"caller-owned.txt"; foreign.write_text("preserve caller bytes")
        observed["foreign"] = foreign
        if fault == "callback-error": raise RuntimeError("intentional callback failure")
    arguments = dict(teacher=teacher, training=training, validation=validation, config=cfg,
        work_dir=work, output_dir=tmp_path/"child", checkpoint_dir=checkpoints, on_checkpoint=inject)
    with pytest.warns(RuntimeWarning):
        if fault == "foreign":
            result = refine(parent, **arguments)
            assert not result.quality_qualified and (tmp_path/"child").is_dir()
        else:
            failure = ValueError if fault == "replacement" else RuntimeError
            message = "identity" if fault == "replacement" else "intentional callback failure"
            with pytest.raises(failure, match=message): refine(parent, **arguments)
            assert not (tmp_path/"child").exists()
    assert observed["foreign"].read_text() == "preserve caller bytes"
    assert [m.training for m in teacher.modules()] == flags
    observed["bundle"].verify()
    assert set(observed["workspace"].iterdir()) == {observed["foreign"]}
    # The test alone owns these injected fixture bytes; production left them intact.
    observed["foreign"].unlink(); observed["workspace"].rmdir()
    if "moved" in observed:
        assert list(observed["moved"].iterdir()) == []
        observed["moved"].rmdir()
    assert list(work.iterdir()) == []
