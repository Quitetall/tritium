"""Typed refinement lifecycle and durable-lineage gates."""

import json

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import (  # noqa: E402
    RefinementConfig,
    TernaryConfig,
    TritiumError,
    calibrate,
    convert,
    export,
    load,
    prepare,
    refine,
)
from tritium.torch.refinement import _canonical, _digest, load_refinement  # noqa: E402


class _TinyLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 8)
        self.down = torch.nn.Linear(8, 4)

    def forward(self, value):
        return self.down(torch.nn.functional.silu(self.up(value)))


def _parent(tmp_path):
    torch.manual_seed(307)
    teacher = _TinyLogits().eval()
    calibration = (torch.randn(3, 8), torch.randn(2, 8))
    prepared = prepare(
        teacher,
        TernaryConfig.ptq(
            profile="compact-v1",
            target_modules=("Linear",),
        ),
        inplace=False,
    )
    evidence = calibrate(
        prepared,
        calibration,
        evidence_dir=tmp_path / "calibration",
    )
    parent = convert(
        prepared,
        evidence,
        work_dir=tmp_path / "parent",
    )
    return teacher, parent


def _datasets():
    return (
        (torch.full((2, 8), 0.25), torch.full((2, 8), -0.75)),
        (torch.full((2, 8), 1.25), torch.full((2, 8), -1.75)),
    )


def test_scale_only_refinement_freezes_trits_and_round_trips(tmp_path):
    teacher, parent = _parent(tmp_path)
    training, validation = _datasets()
    result = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=RefinementConfig.scale_only(max_steps=2, learning_rate=1e-2),
        work_dir=tmp_path / "refined",
    )

    assert result.mode == "scale-only"
    assert result.parent_artifact_id == parent.artifact_id
    assert result.ancestry == (parent.artifact_id,)
    assert result.validation_loss_after <= result.validation_loss_before
    for name in result.weight_names:
        before = parent.weight(name)
        after = result.weight(name)
        assert all(
            torch.equal(left.trits, right.trits)
            for left, right in zip(before.planes, after.planes)
        )
    assert load(result.artifact_dir) == result
    published = export(result, tmp_path / "published")
    assert load_refinement(published.artifact_dir) == published
    reloaded = result.load_model(teacher, inplace=False)
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())


def test_dense_hard_pv_has_distinct_identity_and_complete_ancestry(tmp_path):
    teacher, parent = _parent(tmp_path)
    training, validation = _datasets()
    result = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=RefinementConfig.hard_pv(
            max_steps=2,
            learning_rate=1e-2,
            pv_iterations=2,
        ),
        work_dir=tmp_path / "pv",
    )

    assert result.mode == "hard-pv"
    assert result.artifact_id != parent.artifact_id
    assert result.conversion.algorithm_id == "tritium.hard-pv-dense-refinement@1"
    assert result.validation_loss_after <= result.validation_loss_before
    assert load_refinement(result.artifact_dir) == result

    child = refine(
        result,
        teacher=teacher,
        training=training,
        validation=validation,
        config=RefinementConfig.scale_only(max_steps=1),
        work_dir=tmp_path / "child",
    )
    assert child.parent_artifact_id == result.artifact_id
    assert child.ancestry == (parent.artifact_id, result.artifact_id)


def test_refinement_rejects_any_cross_split_batch_overlap(tmp_path):
    teacher, parent = _parent(tmp_path)
    shared = torch.full((2, 8), 0.5)
    with pytest.raises(TritiumError) as captured:
        refine(
            parent,
            teacher=teacher,
            training=(shared, torch.ones(2, 8)),
            validation=(torch.zeros(2, 8), shared.clone()),
            config=RefinementConfig.scale_only(max_steps=1),
            work_dir=tmp_path / "overlap",
        )
    assert captured.value.code == "dataset_overlap"


def test_refinement_rejects_rehashed_child_ancestry(tmp_path):
    teacher, parent = _parent(tmp_path)
    training, validation = _datasets()
    result = refine(
        parent,
        teacher=teacher,
        training=training,
        validation=validation,
        config=RefinementConfig.scale_only(max_steps=1),
        work_dir=tmp_path / "refined",
    )
    manifest = result.artifact_dir / "refinement.json"
    value = json.loads(manifest.read_text())
    value["training_digest"] = "sha256:" + "0" * 64
    identity = dict(value)
    identity.pop("artifact_id")
    value["artifact_id"] = _digest(identity)
    manifest.write_bytes(_canonical(value))
    with pytest.raises(ValueError, match="dataset ledger"):
        load_refinement(result.artifact_dir)
