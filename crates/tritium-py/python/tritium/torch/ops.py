"""PyTorch dispatcher operators for device-resident ternary execution."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tritium import _tritium as _native


def _validate_linear_inputs(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> None:
    if input.ndim < 1:
        raise ValueError("ternary_linear input must have at least one dimension")
    if master.ndim != 2:
        raise ValueError("ternary_linear master weight must be rank 2")
    if input.shape[-1] != master.shape[1]:
        raise ValueError("ternary_linear input width does not match master weight")
    if not input.dtype.is_floating_point or not master.dtype.is_floating_point:
        raise TypeError("ternary_linear requires floating input and master weight")
    if input.dtype != master.dtype:
        raise TypeError("ternary_linear input and master weight must have the same dtype")
    if input.device != master.device:
        raise ValueError("ternary_linear input and master weight must share a device")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != master.shape[0]:
            raise ValueError("ternary_linear bias shape does not match output width")
        if bias.dtype != input.dtype or bias.device != input.device:
            raise TypeError("ternary_linear bias must match input dtype and device")


def _projection_components(master: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    accumulation_dtype = (
        torch.float32 if master.dtype in {torch.float16, torch.bfloat16} else master.dtype
    )
    scales = (
        master.detach()
        .to(accumulation_dtype)
        .abs()
        .mean(dim=1, keepdim=True)
        .to(master.dtype)
    )
    safe_scales = scales.clamp_min(torch.finfo(master.dtype).tiny)
    normalized = master / safe_scales
    hard = normalized.round().clamp(-1.0, 1.0)
    projected = hard * scales
    mask = ((normalized.abs() < 1.0) & (scales > 0)).to(master.dtype)
    return projected, mask


def reference_ternary_linear(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pure-Torch hard projection used as the dispatcher conformance oracle."""

    _validate_linear_inputs(input, master, bias)
    projected, mask = _projection_components(master)
    differentiable = projected.detach() + (master - master.detach()) * mask
    return F.linear(input, differentiable, bias)


def _native_cpu_supported(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    native = getattr(_native, "_ternary_linear_cpu_dlpack", None)
    return (
        native is not None
        and input.device.type == "cpu"
        and input.dtype == torch.float32
        and master.dtype == torch.float32
        and input.is_contiguous()
        and master.is_contiguous()
        and input.numel() > 0
        and master.shape[0] > 0
        and master.shape[1] > 0
        and (
            bias is None
            or (
                bias.dtype == torch.float32
                and bias.device.type == "cpu"
                and bias.is_contiguous()
            )
        )
    )


@torch.library.custom_op(
    "tritium::ternary_linear",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def ternary_linear(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Hard ternary Linear over tensors resident on the current CPU/CUDA device."""

    _validate_linear_inputs(input, master, bias)
    if _native_cpu_supported(input, master, bias):
        detached_input = input.detach()
        detached_master = master.detach()
        detached_bias = bias.detach() if bias is not None else None
        version = int(master._version)
        storage_identity = int(master.untyped_storage()._cdata)

        def call_native(scales: Optional[torch.Tensor]):
            return _native._ternary_linear_cpu_dlpack(
                detached_input,
                detached_master,
                detached_bias,
                scales,
                master,
                version,
                storage_identity,
            )

        capsule = call_native(None)
        if capsule is None:
            scales = detached_master.abs().mean(dim=1)
            capsule = call_native(scales)
            if capsule is None:
                raise RuntimeError("native ternary Linear cache fill did not publish")
        return torch.from_dlpack(capsule)
    projected, _mask = _projection_components(master)
    return F.linear(input, projected, bias)


@ternary_linear.register_fake
def _ternary_linear_fake(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    torch._check(input.dim() >= 1)
    torch._check(master.dim() == 2)
    torch._check(input.shape[-1] == master.shape[1])
    if bias is not None:
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == master.shape[0])
    return input.new_empty((*input.shape[:-1], master.shape[0]))


def _setup_ternary_linear_context(ctx, inputs, output) -> None:
    del output
    input, master, bias = inputs
    ctx.save_for_backward(input, master)
    ctx.has_bias = bias is not None


def _ternary_linear_backward(ctx, grad_output: torch.Tensor):
    input, master = ctx.saved_tensors
    projected, mask = _projection_components(master)
    input_2d = input.reshape(-1, input.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input = (grad_2d @ projected).reshape(input.shape)
    grad_master = (grad_2d.transpose(0, 1) @ input_2d) * mask
    grad_bias = grad_2d.sum(dim=0) if ctx.has_bias else None
    return grad_input, grad_master, grad_bias


torch.library.register_autograd(
    ternary_linear,
    _ternary_linear_backward,
    setup_context=_setup_ternary_linear_context,
)
torch.library.register_autocast(ternary_linear, "cpu", torch.bfloat16)
torch.library.register_autocast(ternary_linear, "cuda", torch.float16)
