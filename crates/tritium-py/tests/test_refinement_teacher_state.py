"""Public refinement must preserve the caller's reference checkpoint."""
from __future__ import annotations

import pytest
import torch

from tritium.torch import RefinementConfig, TernaryConfig, calibrate, convert, prepare, refine


def parent_for(teacher, directory):
    prepared = prepare(
        teacher,
        TernaryConfig.ptq(profile="near-lossless-v1", target_modules=("Linear",)),
        inplace=False,
    )
    calibration = calibrate(prepared, [torch.ones(2, 8)], evidence_dir=directory / "calibration")
    return convert(prepared, calibration, work_dir=directory / "parent", max_working_bytes=1024 * 1024)


def snapshot(teacher):
    tensors = {**dict(teacher.named_parameters(remove_duplicate=False)), **dict(teacher.named_buffers(remove_duplicate=False))}
    return {
        name: (tensor, tensor.clone(), tensor.data_ptr(), tensor.grad, None if tensor.grad is None else tensor.grad.clone())
        for name, tensor in tensors.items()
    }, tuple((module, module.training) for module in teacher.modules())


def assert_restored(teacher, before):
    tensors, flags = before
    current = {**dict(teacher.named_parameters(remove_duplicate=False)), **dict(teacher.named_buffers(remove_duplicate=False))}
    for name, (original, value, pointer, gradient, gradient_value) in tensors.items():
        tensor = current[name]
        assert tensor is original, f"changed teacher binding {name}"
        assert tensor.dtype == value.dtype, f"changed teacher dtype {name}"
        assert torch.equal(tensor, value), f"changed teacher values {name}"
        assert tensor.data_ptr() == pointer, f"changed teacher storage {name}"
        assert tensor.grad is gradient, f"changed teacher gradient binding {name}"
        if gradient is not None:
            assert gradient.dtype == gradient_value.dtype
            assert torch.equal(gradient, gradient_value)
    assert all(module.training == training for module, training in flags)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("kind", ["scale-only", "hard-pv"])
@pytest.mark.parametrize("conversion_mode", ["default", "overwrite", "swap"])
def test_low_precision_refinement_preserves_teacher_checkpoint(tmp_path, dtype, kind, conversion_mode):
    torch.manual_seed(150)
    teacher = torch.nn.Linear(8, 3)
    teacher.register_buffer("reference", torch.randn(5))
    teacher.register_buffer("reference_alias", teacher.reference)
    teacher.add_module("frozen_dropout", torch.nn.Dropout(0.2))
    teacher.frozen_dropout.eval()
    parent = parent_for(teacher, tmp_path)
    for parameter in teacher.parameters():
        parameter.grad = torch.randn_like(parameter)
    before = snapshot(teacher)
    config = getattr(RefinementConfig, kind.replace("-", "_"))(max_steps=1, compute_dtype=dtype)
    kwargs = dict(teacher=teacher, training=[torch.ones(2, 8) * 0.25], validation=[torch.ones(3, 8) * 0.75], config=config, work_dir=tmp_path / "child")
    overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    swap = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(conversion_mode == "overwrite")
        torch.__future__.set_swap_module_params_on_conversion(conversion_mode == "swap")
        result = refine(parent, **kwargs)
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(overwrite)
        torch.__future__.set_swap_module_params_on_conversion(swap)
    assert result.parent_artifact_id == parent.artifact_id
    assert_restored(teacher, before)
    assert teacher.reference is teacher.reference_alias
    assert refine(parent, **kwargs) == result, "the unchanged teacher must admit sealed resume"


def test_refinement_preserves_mixed_teacher_training_flags(tmp_path):
    torch.manual_seed(151)
    teacher = torch.nn.Linear(8, 3)
    teacher.add_module("frozen_dropout", torch.nn.Dropout(0.2))
    teacher.train()
    teacher.frozen_dropout.eval()
    parent = parent_for(teacher, tmp_path)
    before = tuple(module.training for module in teacher.modules())
    refine(
        parent,
        teacher=teacher,
        training=[torch.ones(2, 8) * 0.25],
        validation=[torch.ones(3, 8) * 0.75],
        config=RefinementConfig.scale_only(max_steps=1),
        work_dir=tmp_path / "child",
    )
    assert tuple(module.training for module in teacher.modules()) == before


@pytest.mark.parametrize("failure", ["cast", "forward"])
def test_refinement_restores_teacher_after_execution_failure(tmp_path, monkeypatch, failure):
    torch.manual_seed(152)
    teacher = torch.nn.Linear(8, 3)
    teacher.register_buffer("reference", torch.randn(5))
    teacher.add_module("frozen_dropout", torch.nn.Dropout(0.2))
    teacher.frozen_dropout.eval()
    parent = parent_for(teacher, tmp_path)
    for parameter in teacher.parameters():
        parameter.grad = torch.randn_like(parameter)
    before = snapshot(teacher)
    if failure == "cast":
        convert_dtype = teacher.to

        def failing_cast(*args, **kwargs):
            convert_dtype(*args, **kwargs)
            raise RuntimeError("injected teacher cast failure")

        monkeypatch.setattr(teacher, "to", failing_cast)
        handle = None
    else:
        def failing_forward(module, inputs):
            raise RuntimeError("injected teacher forward failure")

        handle = teacher.register_forward_pre_hook(failing_forward)
    try:
        with pytest.raises(RuntimeError, match=f"injected teacher {failure} failure"):
            refine(parent, teacher=teacher, training=[torch.ones(2, 8) * 0.25], validation=[torch.ones(3, 8) * 0.75], config=RefinementConfig.scale_only(max_steps=1, compute_dtype="float16"), work_dir=tmp_path / "child")
    finally:
        if handle is not None:
            handle.remove()
    assert_restored(teacher, before)
    assert not (tmp_path / "child" / "refinement.json").exists()


@pytest.mark.parametrize("conversion_mode", ["default", "overwrite", "swap"])
def test_refinement_restores_tied_teacher_parameters(tmp_path, conversion_mode):
    class TiedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(8, 3)
            self.second = torch.nn.Linear(8, 3)
            self.second.weight = self.first.weight

        def forward(self, value):
            return self.first(value) + self.second(value)

    torch.manual_seed(153)
    teacher = TiedTeacher()
    parent = parent_for(teacher, tmp_path)
    for parameter in teacher.parameters():
        parameter.grad = torch.randn_like(parameter)
    before = snapshot(teacher)
    overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    swap = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(conversion_mode == "overwrite")
        torch.__future__.set_swap_module_params_on_conversion(conversion_mode == "swap")
        refine(parent, teacher=teacher, training=[torch.ones(2, 8) * 0.25], validation=[torch.ones(3, 8) * 0.75], config=RefinementConfig.scale_only(max_steps=1, compute_dtype="bfloat16"), work_dir=tmp_path / "child")
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(overwrite)
        torch.__future__.set_swap_module_params_on_conversion(swap)
    assert_restored(teacher, before)
    assert teacher.first.weight is teacher.second.weight
