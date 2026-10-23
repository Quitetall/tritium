"""Installed bounded artifact audit; not quality, physical-memory or release qualification."""
import json
from pathlib import Path
import pytest
import torch
from tritium import NativePvPagedVerificationLimits
from test_native_module_source import StateModel, _prepared, _convert, _fit
from test_native_pv_training import plan, start, update, publish
from test_native_refinement import recording


def _artifact(root, dtype=torch.float32, codec="B3", basis="Identity"):
    torch.manual_seed(37)
    action = _fit()
    action["Fit"]["plan"]["layout"].update(codec=codec, basis=basis)
    parent = _convert(root, _prepared(StateModel(dtype)), actions={"a.weight": action})
    trainer = start(StateModel(dtype), parent, recording=recording(root))
    x = torch.linspace(-1, 1, 64, dtype=dtype).reshape(2, 32)
    for i in range(3):
        update(trainer, i, x)
    child = publish(trainer, root/"child")
    trainer.close()
    return parent, child


def _limits(child, **kwargs):
    cp = (Path(child.output_dir)/"checkpoint.tnpv").stat().st_size
    args = dict(max_owner_read_bytes=1_000_000, max_owner_resident_bytes=1_000_000,
                max_encoded_owner_bytes=1_000_000, max_workspace_bytes=2*cp)
    args.update(kwargs)
    return NativePvPagedVerificationLimits(**args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("codec", ["D2", "B3"])
@pytest.mark.parametrize("basis", ["Identity", {"SignedRht": dict(block=32, seed=13, domain=29)}])
def test_installed_paged_and_resident_audit_match_and_sources_stay_unchanged(tmp_path, dtype, codec, basis):
    parent, child = _artifact(tmp_path, dtype, codec, basis)
    paths = [p for root in (parent.output_dir, child.output_dir) for p in Path(root).iterdir() if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    old = json.loads(child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000))
    work = tmp_path/"paged-work"
    result = json.loads(child.verify_paged_with_parent(parent, work, limits=_limits(child)))
    for key in ("scope", "package_id", "journal_id", "lineage_scope", "parent_admission"):
        assert result[key] == old[key]
    assert result["storage"] == "owner-paged" and result["lineage_scope"] == "immediate-parent-only"
    assert not result["quality_qualified"] and not result["objective_authenticated"]
    assert result["workspace_cleaned"] and result["workspace_cleanup_warnings"] == []
    assert result["workspace_peak_files"] == 2
    cp = (Path(child.output_dir)/"checkpoint.tnpv").stat().st_size
    assert result["workspace_peak_reserved_bytes"] == 2*cp
    assert not work.exists()
    assert before == {p: p.read_bytes() for p in paths}
    assert not child.quality_qualified and not child.objective_authenticated


@pytest.mark.parametrize("name", ["max_owner_read_bytes", "max_owner_resident_bytes", "max_encoded_owner_bytes", "max_workspace_bytes", "max_replay_working_bytes"])
def test_limits_are_explicit_positive_immutable_and_not_duck_typed(name):
    args = dict(max_owner_read_bytes=1_000_000, max_owner_resident_bytes=1_000_000,
                max_encoded_owner_bytes=1_000_000, max_workspace_bytes=1_000_000)
    with pytest.raises(ValueError, match="limits"):
        NativePvPagedVerificationLimits(**{**args, name: 0})
    with pytest.raises((TypeError, OverflowError)):
        NativePvPagedVerificationLimits(**{**args, name: -1})
    limits = NativePvPagedVerificationLimits(**args)
    assert limits.max_replay_working_bytes == 16384
    with pytest.raises(AttributeError):
        limits.max_workspace_bytes = 1


def test_caps_profile_types_overlap_and_existing_directory_fail_without_deleting_inputs(tmp_path):
    parent, child = _artifact(tmp_path)
    cp = (Path(child.output_dir)/"checkpoint.tnpv").stat().st_size
    for i, change in enumerate((dict(max_owner_read_bytes=1), dict(max_owner_resident_bytes=1),
                                dict(max_encoded_owner_bytes=1), dict(max_workspace_bytes=2*cp-1))):
        work = tmp_path/f"limited-{i}"
        with pytest.raises(ValueError):
            child.verify_paged_with_parent(parent, work, limits=_limits(child, **change))
        assert not work.exists()
    for work in (Path(parent.output_dir)/"nested", Path(child.output_dir)/"nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            child.verify_paged_with_parent(parent, work, limits=_limits(child))
    work = tmp_path/"foreign"
    work.mkdir(); (work/"marker").write_text("keep")
    with pytest.raises(FileExistsError):
        child.verify_paged_with_parent(parent, work, limits=_limits(child))
    assert (work/"marker").read_text() == "keep"
    with pytest.raises(TypeError):
        child.verify_paged_with_parent(dict(output_dir=parent.output_dir), tmp_path/"fake-parent", limits=_limits(child))
    with pytest.raises(TypeError):
        child.verify_paged_with_parent(parent, tmp_path/"fake-limits", limits={})
    child.verify(); parent.verify()


@pytest.mark.parametrize("name", ["model.trit", "evidence.jsonl", "checkpoint.tnpv", "gradients.tnpj"])
def test_changed_artifacts_refuse_without_workspace_or_source_repair(tmp_path, name):
    parent, child = _artifact(tmp_path)
    path = Path(child.output_dir)/name
    before = path.read_bytes(); changed = bytearray(before); changed[-1] ^= 1; path.write_bytes(changed)
    work = tmp_path/"work-audit"
    with pytest.raises(ValueError):
        child.verify_paged_with_parent(parent, work, limits=_limits(child))
    assert not work.exists() and path.read_bytes() == changed


def test_immediate_refined_parent_scope_and_scale_only_refusal(tmp_path):
    original, first = _artifact(tmp_path)
    trainer = start(StateModel(), first, plan(first, updates=1, level="pv-child"),
                    level="pv-child", recording=recording(tmp_path))
    update(trainer, 0, torch.linspace(-1, 1, 64).reshape(2, 32))
    second = publish(trainer, tmp_path/"second"); trainer.close()
    audit = json.loads(second.verify_paged_with_parent(first, tmp_path/"second-audit", limits=_limits(second)))
    assert audit["parent_admission"] == "refined-record-consistent-unqualified"
    assert audit["lineage_scope"] == "immediate-parent-only" and not audit["quality_qualified"]
    with pytest.raises(ValueError):
        second.verify_paged_with_parent(original, tmp_path/"wrong-parent", limits=_limits(second))
    from test_native_scale_training import start as scale_start
    from test_native_refinement import update as scale_update, publish as scale_publish
    scale = scale_start(StateModel(), original, recording=recording(tmp_path))
    for i in range(4):
        scale_update(scale, i, torch.ones(2, 32))
    child = scale_publish(scale, tmp_path/"scale"); scale.close()
    limits = NativePvPagedVerificationLimits(max_owner_read_bytes=1_000_000, max_owner_resident_bytes=1_000_000,
                max_encoded_owner_bytes=1_000_000, max_workspace_bytes=1_000_000)
    with pytest.raises(ValueError, match="alternating-pv"):
        child.verify_paged_with_parent(original, tmp_path/"scale-audit", limits=limits)
    assert not (tmp_path/"scale-audit").exists()
