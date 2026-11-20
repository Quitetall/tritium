"""Refinement bookkeeping must not become an executable container layer."""
from __future__ import annotations

import pytest
import torch

from tritium.torch import RefinementConfig, TernaryConfig, calibrate, convert, export, load, prepare, refine


def parent_for(teacher, batch, directory):
    prepared = prepare(
        teacher,
        TernaryConfig.ptq(profile="near-lossless-v1", target_modules=("Linear", "Embedding")),
        inplace=False,
    )
    calibration = calibrate(prepared, [batch], evidence_dir=directory / "calibration")
    return convert(prepared, calibration, work_dir=directory / "parent", max_working_bytes=1024 * 1024)


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("kind", ["scale-only", "hard-pv"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_public_refinement_preserves_sequential_execution(tmp_path, container, kind, dtype):
    torch.manual_seed(160)
    linear = torch.nn.Linear(8, 3)
    teacher = torch.nn.Sequential(linear) if container else linear
    parent = parent_for(teacher, torch.ones(2, 8), tmp_path)
    kwargs = dict(
        teacher=teacher,
        training=[torch.ones(2, 8) * 0.25],
        validation=[torch.ones(3, 8) * 0.75],
        config=getattr(RefinementConfig, kind.replace("-", "_"))(max_steps=1, compute_dtype=dtype),
        work_dir=tmp_path / "child",
    )
    result = refine(parent, **kwargs)
    assert result.validation_loss_after <= result.validation_loss_before
    assert load(result.artifact_dir) == result
    hard = result.load_model(teacher)
    assert hard(torch.ones(2, 8)).shape == (2, 3)
    published = export(result, tmp_path / "published")
    reopened = load(published.artifact_dir)
    assert torch.equal(reopened.load_model(teacher)(torch.ones(2, 8)), hard(torch.ones(2, 8)))
    assert refine(parent, **kwargs) == result


@pytest.mark.parametrize("kind", ["nested", "tied", "shared-module", "embedding-head"])
def test_container_refinement_discovers_unique_scales_and_real_gradients(tmp_path, monkeypatch, kind):
    torch.manual_seed(161)
    if kind == "nested":
        teacher = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()), torch.nn.Linear(8, 3))
    elif kind == "embedding-head":
        embedding = torch.nn.Embedding(16, 8)
        head = torch.nn.Linear(8, 16, bias=False)
        head.weight = embedding.weight
        teacher = torch.nn.Sequential(embedding, head)
    else:
        first = torch.nn.Linear(8, 8)
        second = first if kind == "shared-module" else torch.nn.Linear(8, 8)
        second.weight = first.weight
        teacher = torch.nn.Sequential(first, torch.nn.ReLU(), second)
    teacher._container_fixture_root = True
    if kind == "embedding-head":
        calibration = torch.full((2, 3), 1, dtype=torch.long)
        training = torch.full((2, 3), 2, dtype=torch.long)
        validation = torch.full((2, 3), 3, dtype=torch.long)
    else:
        calibration = torch.ones(2, 8)
        training = calibration * 0.25
        validation = calibration * 0.75
    parent = parent_for(teacher, calibration, tmp_path)
    roots, optimizers, gradients = [], [], []
    root_names = tuple(name for name, _ in teacher.named_children())

    def observe(module, inputs):
        if getattr(module, "_container_fixture_root", False) and module is not teacher:
            assert tuple(name for name, _ in module.named_children()) == root_names
            roots.append(module)

    adamw = torch.optim.AdamW

    def observed_adamw(parameters, **kwargs):
        parameters = tuple(parameters)
        assert len({id(parameter) for parameter in parameters}) == len(parameters)
        optimizer = adamw(parameters, **kwargs)
        optimizers.append(parameters)
        step = optimizer.step

        def observed_step(*args, **kwargs):
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
            gradients.append(sum(float(parameter.grad.abs().sum()) for parameter in parameters))
            return step(*args, **kwargs)

        optimizer.step = observed_step
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", observed_adamw)
    handle = torch.nn.modules.module.register_module_forward_pre_hook(observe)
    try:
        result = refine(parent, teacher=teacher, training=[training], validation=[validation], config=RefinementConfig.scale_only(max_steps=2), work_dir=tmp_path / "child")
    finally:
        handle.remove()
    assert len(optimizers) == 1 and len(gradients) == 2 and any(value > 0 for value in gradients)
    identifiers = {id(parameter) for parameter in optimizers[0]}
    student = next(root for root in roots if {id(parameter) for parameter in root.parameters() if parameter.requires_grad} == identifiers)
    owners = {
        id(module) for module in student.modules()
        if any(id(parameter) in identifiers for parameter in module.parameters(recurse=False))
    }
    ownership_edges = [
        child for module in student.modules() for _, child in module.named_children()
        if id(child) in owners
    ]
    assert len(ownership_edges) == len(owners), "each scale owner is registered once"
    assert len(parent.weight_names) == (2 if kind == "nested" else 1)
    hard = result.load_model(teacher)
    assert torch.isfinite(hard(validation)).all()
    assert not any(parameter.requires_grad for parameter in hard.parameters())
    assert load(result.artifact_dir) == result


def test_sequential_s34_refinement_retains_native_package(tmp_path):
    teacher = torch.nn.Sequential(torch.nn.Linear(128, 3, bias=False))
    with torch.no_grad():
        teacher[0].weight.copy_(torch.tensor([1.0, -1.0, 0.0, 0.0]).repeat(3, 32) * torch.tensor([[0.5], [0.75], [1.0]]))
    parent = parent_for(teacher, torch.ones(2, 128), tmp_path)
    result = refine(
        parent,
        teacher=teacher,
        training=[torch.arange(128, dtype=torch.float32).repeat(2, 1) / 128],
        validation=[torch.arange(128, dtype=torch.float32).repeat(3, 1) / 256],
        config=RefinementConfig.hard_pv(structure="s34", max_steps=1),
        work_dir=tmp_path / "child",
    )
    assert result.packed is not None and result.packed.packing == "s34"
    assert result.validation_loss_before == result.validation_loss_after == 0.0
    assert load(result.artifact_dir) == result
    assert result.load_model(teacher)(torch.ones(2, 128)).shape == (2, 3)
