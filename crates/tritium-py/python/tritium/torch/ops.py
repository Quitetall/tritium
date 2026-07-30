"""PyTorch dispatcher operators for device-resident ternary execution."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple
import weakref

import torch
import torch.nn.functional as F

from tritium import _tritium as _native

_CUDA_PACKED_CACHE_CAPACITY = 4096
_CUDA_PACKED_CACHE = OrderedDict()


class _CudaPackedLinear:
    __slots__ = (
        "owner",
        "version",
        "storage_identity",
        "data_ptr",
        "shape",
        "device",
        "scales",
        "packed",
        "row_bytes",
        "ready",
    )

    def __init__(
        self,
        owner,
        version,
        storage_identity,
        data_ptr,
        shape,
        device,
        scales,
        packed,
        row_bytes,
        ready,
    ):
        self.owner = owner
        self.version = version
        self.storage_identity = storage_identity
        self.data_ptr = data_ptr
        self.shape = shape
        self.device = device
        self.scales = scales
        self.packed = packed
        self.row_bytes = row_bytes
        self.ready = ready


def _drop_cuda_packed(owner_id: int, owner_ref) -> None:
    entry = _CUDA_PACKED_CACHE.get(owner_id)
    if entry is not None and entry.owner is owner_ref:
        _CUDA_PACKED_CACHE.pop(owner_id, None)


def _native_cuda_supported(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    return (
        getattr(_native, "_ternary_linear_cuda_pack", None) is not None
        and getattr(_native, "_ternary_linear_cuda_forward", None) is not None
        and input.device.type == "cuda"
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
                and bias.device == input.device
                and bias.is_contiguous()
            )
        )
    )


def _native_cuda_backward_supported(
    grad_output: torch.Tensor, input: torch.Tensor, master: torch.Tensor
) -> bool:
    return (
        getattr(_native, "_ternary_linear_cuda_backward", None) is not None
        and not torch.is_grad_enabled()
        and grad_output.device.type == "cuda"
        and grad_output.dtype == torch.float32
        and grad_output.is_contiguous()
        and _native_cuda_supported(input, master, None)
    )


def _cuda_device_ordinal(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else int(index)


def _cuda_stream(tensor: torch.Tensor):
    return torch.cuda.current_stream(tensor.device)


def _cuda_packed_linear(master: torch.Tensor) -> _CudaPackedLinear:
    owner_id = id(master)
    version = int(master._version)
    storage_identity = int(master.untyped_storage()._cdata)
    data_ptr = int(master.data_ptr())
    shape = tuple(master.shape)
    device = _cuda_device_ordinal(master)
    entry = _CUDA_PACKED_CACHE.pop(owner_id, None)
    if (
        entry is not None
        and entry.owner() is master
        and entry.version == version
        and entry.storage_identity == storage_identity
        and entry.data_ptr == data_ptr
        and entry.shape == shape
        and entry.device == device
    ):
        _cuda_stream(master).wait_event(entry.ready)
        _CUDA_PACKED_CACHE[owner_id] = entry
        return entry

    detached = master.detach()
    n, k = shape
    row_bytes = ((k + 255) // 256) * 66
    scales = detached.abs().mean(dim=1).contiguous()
    packed = torch.empty((n, row_bytes), dtype=torch.uint8, device=master.device)
    stream = _cuda_stream(master)
    _native._ternary_linear_cuda_pack(
        detached,
        scales,
        packed,
        n,
        k,
        row_bytes,
        stream,
        device,
    )
    ready = torch.cuda.Event()
    ready.record(stream)
    owner_ref = weakref.ref(
        master, lambda ref, key=owner_id: _drop_cuda_packed(key, ref)
    )
    entry = _CudaPackedLinear(
        owner_ref,
        version,
        storage_identity,
        data_ptr,
        shape,
        device,
        scales,
        packed,
        row_bytes,
        ready,
    )
    _CUDA_PACKED_CACHE[owner_id] = entry
    while len(_CUDA_PACKED_CACHE) > _CUDA_PACKED_CACHE_CAPACITY:
        _CUDA_PACKED_CACHE.popitem(last=False)
    return entry


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


def _native_cpu_backward_supported(
    grad_output: torch.Tensor, input: torch.Tensor, master: torch.Tensor
) -> bool:
    return (
        getattr(_native, "_ternary_linear_backward_cpu_dlpack", None) is not None
        and not torch.is_grad_enabled()
        and grad_output.device.type == "cpu"
        and grad_output.dtype == torch.float32
        and _native_cpu_supported(input, master, None)
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
    if _native_cuda_supported(input, master, bias):
        detached_input = input.detach()
        packed = _cuda_packed_linear(master)
        output = torch.empty(
            (*input.shape[:-1], master.shape[0]),
            dtype=input.dtype,
            device=input.device,
        )
        m = input.numel() // input.shape[-1]
        _native._ternary_linear_cuda_forward(
            detached_input,
            packed.packed,
            packed.scales,
            bias.detach() if bias is not None else None,
            output,
            m,
            master.shape[0],
            master.shape[1],
            packed.row_bytes,
            _cuda_stream(input),
            packed.device,
        )
        return output
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
        if capsule is not None:
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
    if _native_cuda_backward_supported(grad_output, input, master):
        detached_grad = grad_output.detach()
        detached_input = input.detach()
        detached_master = master.detach()
        packed = _cuda_packed_linear(master)
        grad_input = torch.empty_like(detached_input)
        grad_master = torch.empty_like(detached_master)
        grad_bias = (
            torch.empty(
                master.shape[0], dtype=master.dtype, device=master.device
            )
            if ctx.has_bias
            else None
        )
        m = input.numel() // input.shape[-1]
        _native._ternary_linear_cuda_backward(
            detached_grad,
            detached_input,
            detached_master,
            packed.packed,
            packed.scales,
            grad_input,
            grad_master,
            grad_bias,
            m,
            master.shape[0],
            master.shape[1],
            packed.row_bytes,
            _cuda_stream(input),
            packed.device,
        )
        return grad_input, grad_master, grad_bias
    if _native_cpu_backward_supported(grad_output, input, master):
        detached_grad = grad_output.detach().contiguous()
        detached_input = input.detach()
        detached_master = master.detach()
        version = int(master._version)
        storage_identity = int(master.untyped_storage()._cdata)

        def call_native(scales: Optional[torch.Tensor]):
            return _native._ternary_linear_backward_cpu_dlpack(
                detached_grad,
                detached_input,
                detached_master,
                scales,
                master,
                version,
                storage_identity,
                ctx.has_bias,
            )

        capsules = call_native(None)
        if capsules is None:
            scales = detached_master.abs().mean(dim=1)
            capsules = call_native(scales)
        if capsules is not None:
            grad_input, grad_master, grad_bias = capsules
            return (
                torch.from_dlpack(grad_input),
                torch.from_dlpack(grad_master),
                torch.from_dlpack(grad_bias) if grad_bias is not None else None,
            )

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
