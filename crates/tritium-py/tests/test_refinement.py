"""Typed scale-only refinement gates from ADR 0033 and plan 0048."""

import json

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import (  # noqa: E402
    RefinementConfig,
    RefinementResult,
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    export,
    load,
    prepare,
    refine,
)


def _parent(teacher, batches, tmp_path):
    prepared = prepare(
        teacher,
        TernaryConfig.ptq(profile="near-lossless-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(
        prepared,
        batches,
        evidence_dir=tmp_path / "calibration",
    )
    return convert(
        prepared,
        calibration,
        work_dir=tmp_path / "ptq",
        max_working_bytes=1024 * 1024,
    )


def test_scale_only_refinement_freezes_trits_and_binds_full_ancestry(tmp_path):
    torch.manual_seed(101)
    teacher = torch.nn.Linear(8, 5)
    calibration = [torch.randn(4, 8) for _ in range(3)]
    parent = _parent(teacher, calibration, tmp_path)
    training = [torch.randn(4, 8), torch.randn(3, 8)]
    validation = [torch.randn(5, 8)]
    config = RefinementConfig.scale_only(max_steps=12, learning_rate=0.05)

    result = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=config,
        work_dir=tmp_path / "refine",
    )

    assert isinstance(result, RefinementResult)
    assert result.mode == "scale-only"
    assert result.parent_artifact_id == parent.artifact_id
    assert result.ancestry == (parent.artifact_id,)
    assert result.validation_loss_after <= result.validation_loss_before
    assert 0 <= result.accepted_steps <= config.max_steps
    for name in parent.weight_names:
        before = parent.weight(name)
        after = result.weight(name)
        assert len(before.planes) == len(after.planes)
        for parent_plane, child_plane in zip(before.planes, after.planes):
            assert torch.equal(parent_plane.trits, child_plane.trits)
            assert parent_plane.group_size == child_plane.group_size

    hard = result.load_model(teacher)
    assert not any(parameter.requires_grad for parameter in hard.parameters())
    assert load(result.artifact_dir) == result
    published = export(result, tmp_path / "published")
    assert published.artifact_id == result.artifact_id

    resumed = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=config,
        work_dir=result.artifact_dir,
    )
    assert resumed == result


def test_low_precision_refinement_restores_teacher_dtype(tmp_path):
    torch.manual_seed(102)
    teacher = torch.nn.Linear(8, 5)
    parent = _parent(teacher, [torch.randn(4, 8)], tmp_path)
    result = refine(
        parent,
        teacher=teacher,
        training=[torch.randn(4, 8)],
        validation=[torch.randn(3, 8)],
        config=RefinementConfig.scale_only(max_steps=1, compute_dtype="float16"),
        work_dir=tmp_path / "low-precision",
    )
    assert result.validation_loss_after <= result.validation_loss_before
    assert teacher.weight.dtype is torch.float32
    assert teacher.bias.dtype is torch.float32


def test_refinement_rejects_overlap_hard_pv_and_rehashed_parent_claims(tmp_path):
    torch.manual_seed(103)
    teacher = torch.nn.Linear(4, 3)
    batches = [torch.randn(2, 4), torch.randn(3, 4)]
    parent = _parent(teacher, batches, tmp_path)

    with pytest.raises(TritiumError) as overlap:
        refine(
            parent,
            teacher=teacher,
            training=batches,
            validation=batches,
            config=RefinementConfig.scale_only(max_steps=1),
            work_dir=tmp_path / "overlap",
        )
    assert overlap.value.code == "dataset_overlap"

    hard_pv = refine(
        parent,
        teacher=teacher,
        training=[batches[0]],
        validation=[batches[1]],
        config=RefinementConfig.hard_pv(max_steps=1, pv_iterations=2),
        work_dir=tmp_path / "hard-pv",
    )
    assert hard_pv.mode == "hard-pv"
    assert hard_pv.validation_loss_after <= hard_pv.validation_loss_before

    result = refine(
        parent,
        teacher=teacher,
        training=[batches[0]],
        validation=[batches[1]],
        config=RefinementConfig.scale_only(max_steps=1),
        work_dir=tmp_path / "valid",
    )
    manifest_path = result.artifact_dir / "refinement.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_artifact_id"] = "sha256:" + "00" * 32
    manifest["ancestry"][-1] = manifest["parent_artifact_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ancestry|identity"):
        load(result.artifact_dir)


def test_g128_refinement_binds_seek_backed_salt_package(tmp_path):
    torch.manual_seed(107)
    teacher = torch.nn.Linear(128, 3)
    parent = _parent(teacher, [torch.randn(2, 128)], tmp_path)
    training = [torch.randn(2, 128)]
    validation = [torch.randn(3, 128)]
    config = RefinementConfig.scale_only(max_steps=1)
    result = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=config,
        work_dir=tmp_path / "refined",
    )

    assert result.schema_version == 2
    assert result.packed is not None
    assert result.packed.packing == "b3"
    assert result.packed.conversion_artifact_id == result.conversion.artifact_id
    assert result.packed.package_id

    manifest = result.artifact_dir / "refinement.json"
    manifest.unlink()
    resumed = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=config,
        work_dir=result.artifact_dir,
    )
    assert resumed.artifact_id == result.artifact_id

    value = json.loads(manifest.read_text(encoding="utf-8"))
    original = dict(value)
    value["packed_artifact_id"] = "sha256:" + "00" * 32
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="packed package"):
        load(result.artifact_dir)

    original["packing"] = "d2"
    manifest.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="packed package"):
        load(result.artifact_dir)
