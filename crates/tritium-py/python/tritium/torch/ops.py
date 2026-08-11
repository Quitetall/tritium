"""PyTorch dispatcher operators for device-resident ternary execution."""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tritium import _tritium as _native

_CUDA_PACKED_CACHE_CAPACITY = 4096
_CUDA_PACKED_CACHE = OrderedDict()
_CPU_AUTOCAST_DTYPE = torch.bfloat16


def _resolve_cuda_autocast_dtype() -> torch.dtype:
    """CUDA autocast dtype for the ternary ops. Default fp16 (fastest packed
    path); `TRITIUM_CUDA_AUTOCAST=bf16` preserves bf16 through the dispatch —
    required by bf16-with-fp32-escapes training loops (LamQuant's codec
    trainer, ADR 0037 Stage 4) where an fp16 downcast changes numerics.
    Resolved once at import, like the registration it feeds."""
    choice = os.environ.get("TRITIUM_CUDA_AUTOCAST", "fp16").strip().lower()
    if choice in ("fp16", "float16", ""):
        return torch.float16
    if choice in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(
        f"TRITIUM_CUDA_AUTOCAST must be fp16 or bf16, got {choice!r}"
    )


_CUDA_AUTOCAST_DTYPE = _resolve_cuda_autocast_dtype()
_NATIVE_CPU_FORWARD = getattr(_native, "_ternary_linear_cpu_dlpack", None)
_NATIVE_CPU_BACKWARD = getattr(
    _native, "_ternary_linear_backward_cpu_dlpack", None
)


def _functorch_transforms_active() -> bool:
    """Return whether public execution is inside a ``torch.func`` transform.

    PyTorch's generated autograd wrapper for ``torch.library.custom_op`` is not
    itself transformable on every supported Torch release. The public helper
    therefore uses the literal Torch reference while ``grad``/``vmap``/friends
    are active; eager and ``torch.compile`` execution still use the native
    dispatcher below. Keep this probe isolated so older Torch versions that do
    not expose the private predicate retain the native path.
    """

    probe = getattr(torch._C, "_are_functorch_transforms_active", None)
    return bool(probe()) if probe is not None else False


class _CudaPackedLinear:
    __slots__ = (
        "owner",
        "version",
        "storage_identity",
        "data_ptr",
        "shape",
        "dtype",
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
        dtype,
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
        self.dtype = dtype
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
    supported_pair = master.dtype == input.dtype or (
        input.dtype == torch.float16 and master.dtype == torch.float32
    )
    return (
        getattr(_native, "_ternary_linear_cuda_pack", None) is not None
        and getattr(_native, "_ternary_linear_cuda_forward", None) is not None
        and input.device.type == "cuda"
        and input.dtype in {torch.float16, torch.float32}
        and supported_pair
        and input.is_contiguous()
        and master.is_contiguous()
        and input.numel() > 0
        and master.shape[0] > 0
        and master.shape[1] > 0
        and (
            bias is None
            or (
                bias.dtype == input.dtype
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
        and grad_output.dtype == input.dtype
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
    dtype = master.dtype
    device = _cuda_device_ordinal(master)
    entry = _CUDA_PACKED_CACHE.pop(owner_id, None)
    if (
        entry is not None
        and entry.owner() is master
        and entry.version == version
        and entry.storage_identity == storage_identity
        and entry.data_ptr == data_ptr
        and entry.shape == shape
        and entry.dtype == dtype
        and entry.device == device
    ):
        _cuda_stream(master).wait_event(entry.ready)
        _CUDA_PACKED_CACHE[owner_id] = entry
        return entry

    detached = master.detach()
    n, k = shape
    row_bytes = ((k + 255) // 256) * 66
    scales = (
        detached.float()
        .abs()
        .mean(dim=1)
        .to(detached.dtype)
        .float()
        .contiguous()
    )
    packed = torch.empty((n, row_bytes), dtype=torch.uint8, device=master.device)
    stream = _cuda_stream(master)
    _native._ternary_linear_cuda_pack(
        detached,
        scales,
        packed,
        n,
        k,
        row_bytes,
        detached.element_size(),
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
        dtype,
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
    mixed_cuda_autocast = (
        input.device.type == "cuda"
        and input.dtype == torch.float16
        and master.dtype == torch.float32
    )
    if input.dtype != master.dtype and not mixed_cuda_autocast:
        raise TypeError(
            "ternary_linear input and master weight must have the same dtype, "
            "except fp16 CUDA input with an fp32 master"
        )
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
    if input.dtype != master.dtype:
        output = F.linear(
            input.float(),
            differentiable,
            bias.float() if bias is not None else None,
        )
        return output.to(input.dtype)
    return F.linear(input, differentiable, bias)


def _reference_ternary_linear_batched(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Batched reference path used only by the custom-op vmap rule."""

    if input.ndim < 2 or master.ndim != 3:
        raise ValueError("batched ternary_linear operands have invalid ranks")
    if input.shape[0] != master.shape[0] or input.shape[-1] != master.shape[-1]:
        raise ValueError("batched ternary_linear dimensions do not match")
    if input.device != master.device:
        raise ValueError("batched ternary_linear operands must share a device")
    mixed_cuda = (
        input.device.type == "cuda"
        and input.dtype == torch.float16
        and master.dtype == torch.float32
    )
    if input.dtype != master.dtype and not mixed_cuda:
        raise ValueError("batched ternary_linear operands have incompatible dtypes")
    if bias is not None and (
        bias.ndim != 2
        or bias.shape[0] != master.shape[0]
        or bias.shape[1] != master.shape[1]
        or bias.device != input.device
        or bias.dtype != input.dtype
    ):
        raise ValueError("batched ternary_linear bias does not match operands")

    accumulation_dtype = (
        torch.float32 if master.dtype in {torch.float16, torch.bfloat16} else master.dtype
    )
    scales = (
        master.detach()
        .to(accumulation_dtype)
        .abs()
        .mean(dim=-1, keepdim=True)
        .to(master.dtype)
    )
    safe_scales = scales.clamp_min(torch.finfo(master.dtype).tiny)
    normalized = master / safe_scales
    projected = normalized.round().clamp(-1.0, 1.0) * scales
    flat_input = input.reshape(input.shape[0], -1, input.shape[-1])
    bias_shape = (bias.shape[0], 1, bias.shape[1]) if bias is not None else None
    if input.dtype != master.dtype:
        output = torch.bmm(
            flat_input.float(), projected.float().transpose(-1, -2)
        )
        if bias is not None:
            output = output + bias.float().view(bias_shape)
        output = output.to(input.dtype)
    else:
        output = torch.bmm(flat_input, projected.transpose(-1, -2))
        if bias is not None:
            output = output + bias.view(bias_shape)
    return output.reshape(*input.shape[:-1], master.shape[1])


def _native_cpu_supported(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> bool:
    return (
        _NATIVE_CPU_FORWARD is not None
        and input.ndim >= 1
        and master.ndim == 2
        and input.shape[-1] == master.shape[1]
        and input.device.type == "cpu"
        and master.device.type == "cpu"
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
                bias.ndim == 1
                and bias.shape[0] == master.shape[0]
                and bias.dtype == torch.float32
                and bias.device.type == "cpu"
                and bias.is_contiguous()
            )
        )
    )


def _native_cpu_backward_supported(
    grad_output: torch.Tensor, input: torch.Tensor, master: torch.Tensor
) -> bool:
    return (
        _NATIVE_CPU_BACKWARD is not None
        and not torch.is_grad_enabled()
        and grad_output.device.type == "cpu"
        and grad_output.dtype == torch.float32
        and _native_cpu_supported(input, master, None)
    )


def _ternary_linear_cpu_native(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Run native CPU forward without entering the dispatcher shim.

    Inference callers do not need a graph-visible custom-op boundary. Keeping
    this fast path separate preserves autograd semantics while avoiding a
    measurable Python custom-op trampoline for decode-sized batches.
    """

    if not _native_cpu_supported(input, master, bias):
        return None
    detached_input = input.detach()
    detached_master = master.detach()
    detached_bias = bias.detach() if bias is not None else None
    version = int(master._version)
    storage_identity = int(master.untyped_storage()._cdata)
    capsule = _NATIVE_CPU_FORWARD(
        detached_input,
        detached_master,
        detached_bias,
        None,
        master,
        version,
        storage_identity,
    )
    if capsule is None:
        scales = detached_master.abs().mean(dim=1)
        capsule = _NATIVE_CPU_FORWARD(
            detached_input,
            detached_master,
            detached_bias,
            scales,
            master,
            version,
            storage_identity,
        )
    return torch.from_dlpack(capsule) if capsule is not None else None


@torch.library.custom_op(
    "tritium::ternary_linear",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def _ternary_linear_dispatch(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Hard ternary Linear over tensors resident on the current CPU/CUDA device."""

    native = _ternary_linear_cpu_native(input, master, bias)
    if native is not None:
        return native
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
            detached_input.element_size(),
            _cuda_stream(input),
            packed.device,
        )
        return output
    projected, _mask = _projection_components(master)
    if input.dtype != master.dtype:
        output = F.linear(
            input.float(), projected, bias.float() if bias is not None else None
        )
        return output.to(input.dtype)
    return F.linear(input, projected, bias)


@_ternary_linear_dispatch.register_fake
def _ternary_linear_fake(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if input.device != master.device:
        raise ValueError("ternary_linear input and master weight must share a device")
    torch._check(input.dim() >= 1)
    torch._check(master.dim() == 2)
    torch._check(input.shape[-1] == master.shape[1])
    if bias is not None:
        if bias.device != input.device:
            raise TypeError("ternary_linear bias must match input dtype and device")
        torch._check(bias.dim() == 1)
        torch._check(bias.shape[0] == master.shape[0])
    return input.new_empty((*input.shape[:-1], master.shape[0]))


def _ternary_linear_vmap(
    info, in_dims, input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
):
    """Batch the first-class op without falling through Python's generic vmap path."""

    # Dispatcher vmap metadata omits trailing optional arguments that were not
    # supplied at the call site. Normalize both ``(input, master)`` and
    # ``(input, master, bias)`` forms before applying batch dimensions.
    if len(in_dims) == 2:
        input_dim, master_dim = in_dims
        bias_dim = None
    elif len(in_dims) == 3:
        input_dim, master_dim, bias_dim = in_dims
    else:  # pragma: no cover - dispatcher owns this contract
        raise ValueError("ternary_linear vmap metadata has invalid arity")
    batch_size = info.batch_size

    def batched(value: torch.Tensor, dim: Optional[int], label: str) -> torch.Tensor:
        if dim is None:
            return value.unsqueeze(0).expand(batch_size, *value.shape)
        moved = value.movedim(dim, 0)
        if moved.shape[0] != batch_size:
            raise ValueError(f"ternary_linear {label} batch dimension differs from vmap")
        return moved

    batched_input = batched(input, input_dim, "input")
    batched_master = batched(master, master_dim, "master")
    batched_bias = (
        None if bias is None else batched(bias, bias_dim, "bias")
    )
    output = _reference_ternary_linear_batched(
        batched_input, batched_master, batched_bias
    )
    return output, 0


_register_vmap = getattr(torch.library, "register_vmap", None)
if _register_vmap is not None:
    _register_vmap(_ternary_linear_dispatch, _ternary_linear_vmap)


def _setup_ternary_linear_context(ctx, inputs, output) -> None:
    del output
    input, master, bias = inputs
    ctx.save_for_backward(input, master)
    ctx.has_bias = bias is not None


def _ternary_linear_backward_impl(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    master: torch.Tensor,
    has_bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _native_cuda_backward_supported(grad_output, input, master):
        detached_grad = grad_output.detach()
        detached_input = input.detach()
        detached_master = master.detach()
        packed = _cuda_packed_linear(master)
        grad_input = torch.empty_like(detached_input)
        grad_master = torch.empty_like(detached_master)
        grad_bias = (
            torch.empty(
                master.shape[0], dtype=input.dtype, device=master.device
            )
            if has_bias
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
            detached_input.element_size(),
            detached_master.element_size(),
            _cuda_stream(input),
            packed.device,
        )
        return (
            grad_input,
            grad_master,
            grad_bias if grad_bias is not None else input.new_empty((0,)),
        )
    if _native_cpu_backward_supported(grad_output, input, master):
        detached_grad = grad_output.detach().contiguous()
        detached_input = input.detach()
        detached_master = master.detach()
        version = int(master._version)
        storage_identity = int(master.untyped_storage()._cdata)
        capsules = _NATIVE_CPU_BACKWARD(
            detached_grad,
            detached_input,
            detached_master,
            None,
            master,
            version,
            storage_identity,
            has_bias,
        )
        if capsules is None:
            scales = detached_master.abs().mean(dim=1)
            capsules = _NATIVE_CPU_BACKWARD(
                detached_grad,
                detached_input,
                detached_master,
                scales,
                master,
                version,
                storage_identity,
                has_bias,
            )
        if capsules is not None:
            grad_input, grad_master, grad_bias = capsules
            return (
                torch.from_dlpack(grad_input),
                torch.from_dlpack(grad_master),
                torch.from_dlpack(grad_bias)
                if grad_bias is not None
                else input.new_empty((0,)),
            )

    projected, mask = _projection_components(master)
    input_2d = input.reshape(-1, input.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    if input.dtype != master.dtype:
        grad_2d_master = grad_2d.float()
        grad_input = (grad_2d_master @ projected).to(input.dtype).reshape(input.shape)
        grad_master = (
            grad_2d_master.transpose(0, 1) @ input_2d.float()
        ) * mask
        grad_bias = (
            grad_2d_master.sum(dim=0).to(input.dtype)
            if has_bias
            else input.new_empty((0,))
        )
        return grad_input, grad_master, grad_bias
    grad_input = (grad_2d @ projected).reshape(input.shape)
    grad_master = (grad_2d.transpose(0, 1) @ input_2d) * mask
    grad_bias = grad_2d.sum(dim=0) if has_bias else input.new_empty((0,))
    return grad_input, grad_master, grad_bias


def _ternary_linear_backward_dispatch(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    master: torch.Tensor,
    has_bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opaque first-order VJP boundary for compiled native/cache execution."""

    return torch.ops.tritium._ternary_linear_backward.default(
        grad_output, input, master, has_bias
    )


_BACKWARD_LIBRARY = torch.library.Library("tritium", "FRAGMENT")
_BACKWARD_LIBRARY.define(
    "_ternary_linear_backward(Tensor grad_output, Tensor input, Tensor master, bool has_bias) -> (Tensor, Tensor, Tensor)"
)
_BACKWARD_LIBRARY.impl(
    "_ternary_linear_backward", _ternary_linear_backward_impl, "CPU"
)
_BACKWARD_LIBRARY.impl(
    "_ternary_linear_backward", _ternary_linear_backward_impl, "CUDA"
)


@torch.library.register_fake("tritium::_ternary_linear_backward")
def _ternary_linear_backward_fake(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    master: torch.Tensor,
    has_bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output
    grad_bias = input.new_empty((master.shape[0],) if has_bias else (0,))
    return input.new_empty(input.shape), master.new_empty(master.shape), grad_bias


def _ternary_linear_backward(ctx, grad_output: torch.Tensor):
    input, master = ctx.saved_tensors
    if torch.is_grad_enabled():
        grad_input, grad_master, grad_bias = _ternary_linear_backward_impl(
            grad_output, input, master, ctx.has_bias
        )
    else:
        grad_input, grad_master, grad_bias = _ternary_linear_backward_dispatch(
            grad_output, input, master, ctx.has_bias
        )
    return grad_input, grad_master, grad_bias if ctx.has_bias else None


torch.library.register_autograd(
    _ternary_linear_dispatch,
    _ternary_linear_backward,
    setup_context=_setup_ternary_linear_context,
)
torch.library.register_autocast(_ternary_linear_dispatch, "cpu", _CPU_AUTOCAST_DTYPE)


def _ternary_linear_cuda_autocast(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Preserve fp32 master/optimizer state. Only activation-facing tensors use
    # fp16; disabling autocast around redispatch prevents recursion.
    with torch.autocast("cuda", enabled=False):
        cast_input = input.to(_CUDA_AUTOCAST_DTYPE)
        cast_bias = bias.to(_CUDA_AUTOCAST_DTYPE) if bias is not None else None
        native_mixed = (
            getattr(_native, "_ternary_linear_cuda_pack", None) is not None
            and getattr(_native, "_ternary_linear_cuda_forward", None) is not None
            and getattr(_native, "_ternary_linear_cuda_backward", None) is not None
            and master.dtype == torch.float32
            # The native mixed kernel is fp16-activation-only: under bf16
            # autocast the master must be cast down with the input, or the
            # dispatch validator rejects the bf16-input/fp32-master pair.
            and _CUDA_AUTOCAST_DTYPE == torch.float16
        )
        cast_master = master if native_mixed else master.to(_CUDA_AUTOCAST_DTYPE)
        return _ternary_linear_dispatch(cast_input, cast_master, cast_bias)


_CUDA_AUTOCAST_LIBRARY = torch.library.Library("tritium", "FRAGMENT")
_CUDA_AUTOCAST_LIBRARY.impl(
    "ternary_linear", _ternary_linear_cuda_autocast, "AutocastCUDA"
)


def ternary_linear(
    input: torch.Tensor, master: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Hard ternary Linear with graph-visible CPU/CUDA autocast semantics."""

    if _functorch_transforms_active():
        return reference_ternary_linear(input, master, bias)
    if input.device.type == "cpu" and not torch.is_grad_enabled():
        native = _ternary_linear_cpu_native(input, master, bias)
        if native is not None:
            return native
    if input.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return _ternary_linear_cuda_autocast(input, master, bias)
    if input.device.type == "cpu" and torch.is_autocast_enabled("cpu"):
        with torch.autocast("cpu", enabled=False):
            return _ternary_linear_dispatch(
                input.to(_CPU_AUTOCAST_DTYPE),
                master.to(_CPU_AUTOCAST_DTYPE),
                bias.to(_CPU_AUTOCAST_DTYPE) if bias is not None else None,
            )
    return _ternary_linear_dispatch(input, master, bias)
