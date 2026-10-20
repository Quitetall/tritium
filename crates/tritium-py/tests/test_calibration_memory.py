"""Moment capture must not materialize complete strided activation masters."""
from __future__ import annotations

import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tritium.torch import TernaryConfig, calibrate, prepare

ELEMENT_WINDOW = 1024 * 1024


class CaptureWindow(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.clone.default:
            assert args[0].numel() <= ELEMENT_WINDOW, (
                f"whole activation clone in calibration: {args[0].numel()} elements"
            )
        return func(*args, **(kwargs or {}))


def guard_capture(monkeypatch):
    register = torch.nn.Module.register_forward_pre_hook

    def guarded(module, hook, *args, **kwargs):
        def invoke(*values):
            with CaptureWindow():
                return hook(*values)
        return register(module, invoke, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "register_forward_pre_hook", guarded)


def legacy_moments(value):
    rows = value.reshape(-1, value.shape[-1])
    maximum = max(1, ELEMENT_WINDOW // rows.shape[-1])
    total = torch.zeros(rows.shape[-1], dtype=torch.float64)
    for start in range(0, rows.shape[0], maximum):
        chunk = rows[start:start + maximum].to(torch.float64)
        total.add_(chunk.square().sum(dim=0).cpu())
    return total


def run_capture(value, directory, monkeypatch):
    expected = legacy_moments(value)
    model = torch.nn.Linear(value.shape[-1], 2, bias=False, device=value.device, dtype=value.dtype)
    model.train()
    prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)), inplace=True)
    with monkeypatch.context() as patch:
        guard_capture(patch)
        receipt = calibrate(prepared, [value], evidence_dir=directory)
    actual = torch.frombuffer(bytearray((directory / receipt.records[0].file).read_bytes()), dtype=torch.float64)
    assert torch.equal(actual, expected), "retain the exact legacy reduction order"
    assert receipt.records[0].samples == value.numel() // value.shape[-1]
    assert prepared.model.training is True
    assert not prepared.model._forward_pre_hooks


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_public_calibration_bounds_strided_activation_copies(tmp_path, monkeypatch, dtype):
    value = (torch.arange(513 * 1024 * 3, dtype=torch.float32).reshape(513, 1024, 3).transpose(0, 1) / 1000000).to(dtype)
    assert not value.is_contiguous()
    run_capture(value.contiguous(), tmp_path / "control", monkeypatch)
    run_capture(value, tmp_path / "evidence", monkeypatch)


@pytest.mark.parametrize("kind", ["permuted", "wide", "offset", "expanded"])
def test_calibration_preserves_legacy_windows_for_other_strided_layouts(tmp_path, monkeypatch, kind):
    if kind == "permuted":
        value = torch.randn(3, 65, 2049, 7, generator=torch.Generator().manual_seed(42), dtype=torch.float64).permute(2, 0, 1, 3)
    elif kind == "wide":
        value = torch.randn(32, 2, 65537, generator=torch.Generator().manual_seed(43), dtype=torch.float64).transpose(0, 1)
    elif kind == "offset":
        value = torch.randn(700001, 6, generator=torch.Generator().manual_seed(44))[17:700000:2, ::2]
    else:
        value = torch.tensor([[[0.1, -0.2, 0.3]]], dtype=torch.float64).expand(1001, 500, 3)
    run_capture(value, tmp_path / "evidence", monkeypatch)


def test_physical_cuda_calibration_bounds_strided_activation_copies(tmp_path, monkeypatch):
    if os.environ.get("TRITIUM_REQUIRE_MOMENT_CUDA") != "1":
        pytest.skip("requires separately invoked physical CUDA moment gate")
    assert torch.cuda.is_available(), "required physical CUDA test cannot skip"
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        value = (torch.arange(513 * 1024 * 3, dtype=torch.float32, device="cuda").reshape(513, 1024, 3).transpose(0, 1) / 1000000).to(dtype)
        run_capture(value, tmp_path / str(dtype), monkeypatch)
    torch.cuda.synchronize()
