"""Resumable phased Qwen3.6 PTQ lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn

from .. import _tritium
from ..salt import reconcile_qwen36_ptq_packages
from .artifacts import QuantizationResult, load
from .allocation import allocate_planes
from .config import TernaryConfig
from .conversion import PreparedModel, prepare
from .errors import TritiumError
from .module_artifacts import (
    FittedWeight,
    ModuleQuantizationResult,
    WeightCheckpointWriter,
    load_module_conversion,
    module_recipe_id,
    seal_module_conversion,
)
from .projection import (
    TernaryPlane,
    TernaryProjection,
    expand_plane_scales,
    validate_projection,
)

if TYPE_CHECKING:
    from .qat import QatHardResult

Pathish = Union[str, os.PathLike[str]]
# Conservative auxiliary-tensor ledger for the row-tiled float64 fitter.
# Source weights and framework allocator overhead are intentionally excluded.
_FIT_FIXED_BYTES_PER_FEATURE = 24
_FIT_BYTES_PER_COEFFICIENT = 128
_KRONECKER_OBJECTIVE_CONTEXT = b"tritium kronecker activation objective v1\0"


def bind_kronecker_activation_cache_digest(
    activation_cache_digest: str, objective_id: str
) -> str:
    """Bind an exact capture objective into a raw SHA-256 cache identity."""

    if (
        not isinstance(activation_cache_digest, str)
        or len(activation_cache_digest) != 64
        or any(byte not in "0123456789abcdefABCDEF" for byte in activation_cache_digest)
    ):
        raise ValueError("activation_cache_digest must be exactly 64 hexadecimal characters")
    if (
        not isinstance(objective_id, str)
        or not objective_id
        or len(objective_id.encode("utf-8")) > 1024
    ):
        raise ValueError("objective_id must be a nonempty UTF-8 string no larger than 1024 bytes")
    encoded = objective_id.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_KRONECKER_OBJECTIVE_CONTEXT)
    digest.update(bytes.fromhex(activation_cache_digest))
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationReceipt:
    """Content-bound admission of a complete canonical PTQ evidence namespace."""

    evidence_dir: Path
    evidence_id: str
    curvature: str
    record_count: int
    source_model_digest: str
    activation_cache_digest: str
    token_stream_digest: str
    max_evidence_bytes: int
    schema_version: int = 1


@dataclass(frozen=True)
class ActivationRecord:
    """One bounded diagonal-curvature record for a selected module input."""

    module: str
    weight_aliases: Tuple[str, ...]
    features: int
    outputs: int
    samples: int
    file: str
    digest: str
    bytes: int


@dataclass(frozen=True)
class ActivationCalibrationReceipt:
    """Strict receipt for streamed activation evidence from a live module."""

    evidence_dir: Path
    evidence_id: str
    curvature: str
    record_count: int
    source_model_digest: str
    activation_cache_digest: str
    token_stream_digest: str
    max_evidence_bytes: int
    records: Tuple[ActivationRecord, ...]
    schema_version: int = 2


@dataclass(frozen=True)
class _SelectedLinear:
    """One selected weight plus every distinct Linear consumer of it.

    Coverage stores aliases for one exact parameter, while PyTorch can expose
    that parameter through multiple distinct ``Linear`` modules.  Calibration
    must observe every consumer so the fitted curvature represents the model
    graph rather than whichever alias happens to be visited first.
    """

    path: str
    module: nn.Linear
    aliases: Tuple[str, ...]
    capture_modules: Tuple[nn.Linear, ...]


class KroneckerCalibrationWriter:
    """Stream one tensor's grouped curvature into the native S2KF producer.

    This is the low-level PyTorch runtime hook used by checkpoint-specific
    calibration drivers. It retains only one caller batch plus the native
    dyadic G128 state and atomically publishes only on :meth:`finish`.
    """

    def __init__(
        self,
        evidence_dir: Pathish,
        *,
        tensor_index: int,
        tensor_name: str,
        rows: int,
        columns: int,
        curvature: str,
        source_model_digest: str,
        activation_cache_digest: str,
        token_stream_digest: str,
        damping: float,
        objective_id: Optional[str] = None,
        indexed_output: bool = False,
        max_evidence_bytes: int = 64 * 1024 * 1024,
        max_batch_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._rows = rows
        self._columns = columns
        self._curvature = curvature
        self._objective_id = objective_id
        if type(indexed_output) is not bool:
            raise TypeError("indexed_output must be a boolean")
        self._indexed_output = indexed_output
        if objective_id is not None:
            activation_cache_digest = bind_kronecker_activation_cache_digest(
                activation_cache_digest, objective_id
            )
        self._activation_cache_digest = activation_cache_digest
        self._max_batch_bytes = max_batch_bytes
        self._native = _tritium.KroneckerEvidenceBuilder(
            os.fspath(evidence_dir),
            tensor_index,
            tensor_name,
            rows,
            columns,
            curvature,
            source_model_digest,
            activation_cache_digest,
            token_stream_digest,
            damping,
            max_evidence_bytes=max_evidence_bytes,
            max_batch_bytes=max_batch_bytes,
            indexed_output=indexed_output,
        )

    @staticmethod
    def _float_bytes(value: torch.Tensor, dtype: torch.dtype, numpy_dtype: str) -> bytes:
        return (
            value.detach()
            .to(device="cpu", dtype=dtype)
            .contiguous()
            .numpy()
            .astype(numpy_dtype, copy=False)
            .tobytes()
        )

    def append(
        self,
        activations: torch.Tensor,
        output_factors: Optional[torch.Tensor] = None,
        *,
        token_weights: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Append one canonical row-major batch and return retained segment counts."""

        if self._indexed_output:
            raise ValueError("indexed-output writers require append_indexed")

        if not isinstance(activations, torch.Tensor) or activations.ndim == 0:
            raise TypeError("activations must be a tensor with a feature dimension")
        if activations.shape[-1] != self._columns:
            raise ValueError(
                f"activations last dimension must be {self._columns}, got {activations.shape[-1]}"
            )
        _require_finite_tensor(activations, "calibration activations")
        sample_shape = tuple(activations.shape[:-1])
        samples = activations.numel() // self._columns
        required_bytes = activations.numel() * 4
        if output_factors is not None:
            if not isinstance(output_factors, torch.Tensor) or output_factors.ndim == 0:
                raise TypeError("output_factors must be a tensor with an output dimension")
            if (
                tuple(output_factors.shape[:-1]) != sample_shape
                or output_factors.shape[-1] != self._rows
            ):
                raise ValueError(
                    f"output_factors must have shape {sample_shape + (self._rows,)}"
                )
            _require_finite_tensor(output_factors, "calibration output factors")
            required_bytes += output_factors.numel() * 4
        if token_weights is not None:
            if (
                not isinstance(token_weights, torch.Tensor)
                or tuple(token_weights.shape) != sample_shape
            ):
                raise ValueError(f"token_weights must have shape {sample_shape}")
            _require_finite_tensor(token_weights, "calibration token weights")
            required_bytes += token_weights.numel() * 8
        if token_mask is not None:
            if (
                not isinstance(token_mask, torch.Tensor)
                or tuple(token_mask.shape) != sample_shape
            ):
                raise ValueError(f"token_mask must have shape {sample_shape}")
            required_bytes += token_mask.numel()
        if required_bytes > self._max_batch_bytes:
            raise ValueError(
                f"batch requires {required_bytes} bytes, limit is {self._max_batch_bytes}"
            )
        factors_bytes = (
            self._float_bytes(output_factors, torch.float32, "<f4")
            if output_factors is not None
            else None
        )
        weights_bytes = (
            self._float_bytes(token_weights, torch.float64, "<f8")
            if token_weights is not None
            else None
        )
        mask_bytes = None
        if token_mask is not None:
            mask = token_mask.detach().to(device="cpu").contiguous().view(-1)
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("token_mask values must be boolean or 0/1")
            mask_bytes = mask.to(torch.uint8).numpy().tobytes()
        return self._native.append_batch(
            self._float_bytes(activations, torch.float32, "<f4"),
            samples,
            output_factors_f32le=factors_bytes,
            token_weights_f64le=weights_bytes,
            token_mask_u8=mask_bytes,
        )

    def append_indexed(
        self,
        activations: torch.Tensor,
        output_indices: torch.Tensor,
        output_factors: Optional[torch.Tensor] = None,
        *,
        token_weights: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Append one sparse output row and factor per activation sample."""

        if not self._indexed_output:
            raise ValueError("dense-output writers require append")
        if not isinstance(activations, torch.Tensor) or activations.ndim == 0:
            raise TypeError("activations must be a tensor with a feature dimension")
        if activations.shape[-1] != self._columns:
            raise ValueError(
                f"activations last dimension must be {self._columns}, got {activations.shape[-1]}"
            )
        sample_shape = tuple(activations.shape[:-1])
        samples = activations.numel() // self._columns
        if (
            not isinstance(output_indices, torch.Tensor)
            or tuple(output_indices.shape) != sample_shape
        ):
            raise ValueError(f"output_indices must have shape {sample_shape}")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if output_indices.dtype not in integer_dtypes:
            raise TypeError("output_indices must use an integer dtype")
        if bool((output_indices < 0).any()) or bool(
            (output_indices >= self._rows).any()
        ):
            raise ValueError(f"output_indices values must be in [0, {self._rows})")
        if output_factors is not None and (
            not isinstance(output_factors, torch.Tensor)
            or tuple(output_factors.shape) != sample_shape
        ):
            raise ValueError(f"output_factors must have shape {sample_shape}")
        if output_factors is not None:
            _require_finite_tensor(output_factors, "calibration output factors")
        if token_weights is not None and (
            not isinstance(token_weights, torch.Tensor)
            or tuple(token_weights.shape) != sample_shape
        ):
            raise ValueError(f"token_weights must have shape {sample_shape}")
        if token_weights is not None:
            _require_finite_tensor(token_weights, "calibration token weights")
        if token_mask is not None and (
            not isinstance(token_mask, torch.Tensor)
            or tuple(token_mask.shape) != sample_shape
        ):
            raise ValueError(f"token_mask must have shape {sample_shape}")
        required_bytes = activations.numel() * 4 + samples * (8 + 4)
        if token_weights is not None:
            required_bytes += token_weights.numel() * 8
        if token_mask is not None:
            required_bytes += token_mask.numel()
        if required_bytes > self._max_batch_bytes:
            raise ValueError(
                f"batch requires {required_bytes} bytes, limit is {self._max_batch_bytes}"
            )
        indices_bytes = (
            output_indices.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
            .numpy()
            .astype("<u8", copy=False)
            .tobytes()
        )
        factors_bytes = (
            self._float_bytes(output_factors, torch.float32, "<f4")
            if output_factors is not None
            else None
        )
        weights_bytes = (
            self._float_bytes(token_weights, torch.float64, "<f8")
            if token_weights is not None
            else None
        )
        mask_bytes = None
        if token_mask is not None:
            mask = token_mask.detach().to(device="cpu").contiguous().view(-1)
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("token_mask values must be boolean or 0/1")
            mask_bytes = mask.to(torch.uint8).numpy().tobytes()
        return self._native.append_indexed_batch(
            self._float_bytes(activations, torch.float32, "<f4"),
            indices_bytes,
            samples,
            output_factors_f32le=factors_bytes,
            token_weights_f64le=weights_bytes,
            token_mask_u8=mask_bytes,
        )

    def append_indexed_identity(
        self,
        output_indices: torch.Tensor,
        *,
        token_weights: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int]:
        """Append sparse rows under indexed identity-output input-Hessian curvature."""

        if not self._indexed_output:
            raise ValueError("dense-output writers require append")
        if self._curvature != "input-hessian":
            raise ValueError("indexed identity evidence requires input-hessian curvature")
        if not isinstance(output_indices, torch.Tensor) or output_indices.ndim == 0:
            raise TypeError("output_indices must be a tensor with a sample dimension")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if output_indices.dtype not in integer_dtypes:
            raise TypeError("output_indices must use an integer dtype")
        sample_shape = tuple(output_indices.shape)
        samples = output_indices.numel()
        if bool((output_indices < 0).any()) or bool(
            (output_indices >= self._rows).any()
        ):
            raise ValueError(f"output_indices values must be in [0, {self._rows})")
        if token_weights is not None and (
            not isinstance(token_weights, torch.Tensor)
            or tuple(token_weights.shape) != sample_shape
        ):
            raise ValueError(f"token_weights must have shape {sample_shape}")
        if token_weights is not None:
            _require_finite_tensor(token_weights, "calibration token weights")
        if token_mask is not None and (
            not isinstance(token_mask, torch.Tensor)
            or tuple(token_mask.shape) != sample_shape
        ):
            raise ValueError(f"token_mask must have shape {sample_shape}")
        required_bytes = samples * 8
        if token_weights is not None:
            required_bytes += token_weights.numel() * 8
        if token_mask is not None:
            required_bytes += token_mask.numel()
        if required_bytes > self._max_batch_bytes:
            raise ValueError(
                f"batch requires {required_bytes} bytes, limit is {self._max_batch_bytes}"
            )
        indices_bytes = (
            output_indices.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
            .numpy()
            .astype("<u8", copy=False)
            .tobytes()
        )
        weights_bytes = (
            self._float_bytes(token_weights, torch.float64, "<f8")
            if token_weights is not None
            else None
        )
        mask_bytes = None
        if token_mask is not None:
            mask = token_mask.detach().to(device="cpu").contiguous().view(-1)
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("token_mask values must be boolean or 0/1")
            mask_bytes = mask.to(torch.uint8).numpy().tobytes()
        return self._native.append_indexed_identity_batch(
            indices_bytes,
            samples,
            token_weights_f64le=weights_bytes,
            token_mask_u8=mask_bytes,
        )

    def finish(self):
        """Finalize and atomically publish one canonical native evidence record."""

        return self._native.finish()

    def abort(self) -> None:
        """Drop unpublished native accumulator state."""

        self._native.abort()

    @property
    def active(self) -> bool:
        """Whether append/finalize operations are still admitted."""

        return self._native.active

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def curvature(self) -> str:
        return self._curvature

    @property
    def max_batch_bytes(self) -> int:
        return self._max_batch_bytes

    @property
    def objective_id(self) -> Optional[str]:
        return self._objective_id

    @property
    def indexed_output(self) -> bool:
        """Whether this writer accepts sparse indexed output factors."""

        return self._indexed_output

    @property
    def activation_cache_digest(self) -> str:
        return self._activation_cache_digest


@dataclass(frozen=True)
class KroneckerModuleCaptureReceipt:
    """Exact runtime coverage for one published module-curvature record."""

    module: str
    curvature: str
    objective: str
    batches: int
    module_calls: int
    samples: int
    selected_samples: int
    input_segments: int
    output_segments: int
    record: Any
    schema_version: int = 1


def _capture_output_tensor(output: Any, field: str) -> torch.Tensor:
    value = output.get(field) if isinstance(output, Mapping) else getattr(output, field, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"model output must expose tensor {field!r}")
    return value


def _require_finite_tensor(value: torch.Tensor, label: str) -> None:
    """Reject non-finite calibration values before native evidence mutation."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must contain only finite values")


def _freeze_capture_parameters(
    runner: nn.Module, *, curvature: str
) -> Tuple[Tuple[nn.Parameter, bool], ...]:
    """Freeze runner parameters while capturing output-side curvature.

    Capture requests gradients with respect to selected module outputs, not
    model parameters. Leaving checkpoint parameters trainable makes autograd
    retain parameter paths (and can allocate optimizer-sized graph state) for
    large models. Preserve and restore each flag so capture remains a
    non-mutating observation of caller training state.
    """

    if curvature == "input-hessian":
        return ()
    states = []
    seen: set[int] = set()
    for parameter in runner.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        states.append((parameter, parameter.requires_grad))
        parameter.requires_grad_(False)
    return tuple(states)


def _restore_capture_parameters(
    states: Tuple[Tuple[nn.Parameter, bool], ...]
) -> None:
    for parameter, requires_grad in states:
        parameter.requires_grad_(requires_grad)


def _capture_token_mask(
    mask: Optional[torch.Tensor], sample_shape: Tuple[int, ...]
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    samples = 1
    for dimension in sample_shape:
        samples *= dimension
    if mask.numel() != samples:
        raise ValueError(
            f"attention_mask has {mask.numel()} values for module sample shape {sample_shape}"
        )
    return mask.reshape(sample_shape)


def _causal_selection_mask(
    labels: torch.Tensor, token_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if labels.ndim == 0:
        raise ValueError("causal labels must have a sequence dimension")
    if token_mask is not None and token_mask.shape != labels.shape:
        raise ValueError("causal labels and attention_mask must have the same shape")
    selected = torch.zeros_like(labels, dtype=torch.bool)
    torch.ne(labels[..., 1:], -100, out=selected[..., :-1])
    return selected


def _forward_kl_factors(
    logits: torch.Tensor,
    sample_start: int,
    token_mask: Optional[torch.Tensor],
    max_objective_bytes: int,
) -> torch.Tensor:
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("forward-KL logits must have a nontrivial vocabulary dimension")
    _require_finite_tensor(logits, "forward-KL logits")
    elements = logits.numel()
    samples = elements // logits.shape[-1]
    vocabulary = logits.shape[-1]
    factor_bytes = elements * 4
    center_bytes = samples * 4
    fixed_bytes = factor_bytes + center_bytes
    required_bytes = fixed_bytes + 64
    if required_bytes > max_objective_bytes:
        raise ValueError(
            f"forward-KL factors require at least {required_bytes} bytes, "
            f"limit is {max_objective_bytes}"
        )
    # The explicit hash/centering workspace uses at most 64 bytes per element.
    # Keep it disjoint from the single retained f32 factor tensor.
    chunk_elements = min(elements, (max_objective_bytes - fixed_bytes) // 64)
    if chunk_elements == 0:
        raise AssertionError("objective preflight admitted an empty workspace")
    factors = torch.softmax(logits.detach(), dim=-1, dtype=torch.float32).sqrt_()
    flat = factors.view(-1)
    for offset in range(0, elements, chunk_elements):
        end = min(elements, offset + chunk_elements)
        indices = torch.arange(offset, end, device=logits.device, dtype=torch.int64)
        sample_ids = torch.div(indices, vocabulary, rounding_mode="floor").add_(sample_start)
        feature_ids = indices.remainder(vocabulary)
        mixed = feature_ids.mul(1_103_515_245).add_(sample_ids * 2_654_435_761)
        mixed.bitwise_xor_(torch.bitwise_right_shift(mixed, 16))
        signs = ((mixed & 1) * 2 - 1).to(torch.float32)
        flat[offset:end].mul_(signs)
    centers = factors.sum(dim=-1, keepdim=True)
    center_flat = centers.view(-1)
    for offset in range(0, elements, chunk_elements):
        end = min(elements, offset + chunk_elements)
        indices = torch.arange(offset, end, device=logits.device, dtype=torch.int64)
        sample_ids = torch.div(indices, vocabulary, rounding_mode="floor")
        correction = flat[offset:end].square()
        correction.mul_(center_flat[sample_ids])
        flat[offset:end].sub_(correction)
    mask = _capture_token_mask(token_mask, tuple(logits.shape[:-1]))
    if mask is not None:
        mask_flat = mask.reshape(-1)
        mask_chunk_samples = max(1, chunk_elements // vocabulary)
        for offset in range(0, samples, mask_chunk_samples):
            end = min(samples, offset + mask_chunk_samples)
            selected = mask_flat[offset:end].to(
                device=factors.device, dtype=factors.dtype
            )
            factors.view(samples, vocabulary)[offset:end].mul_(selected.view(-1, 1))
    return factors


def capture_kronecker_embedding(
    model: nn.Module,
    data: Iterable[Any],
    *,
    module: str,
    writer: KroneckerCalibrationWriter,
    curvature: str,
    execution_model: Optional[nn.Module] = None,
    guided_loss_reduction: Optional[str] = None,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
) -> KroneckerModuleCaptureReceipt:
    """Capture embedding-table Fisher/KL factors without dense vocabulary rows.

    ``execution_model`` can be a containing/oracle module when the selected
    embedding is not the module that owns the calibration forward. Hooks stay
    attached to ``model``; batches execute through ``execution_model``.

    For Fisher/KL curvature, the gradient with respect to each embedding
    output supplies the hidden-state factor and its token id supplies a sparse
    output row. For input-Hessian, lookup uses an explicit identity-output
    contract: each selected token contributes ``I`` in hidden space and its
    sparse row frequency. No dense one-hot vector is materialized.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if execution_model is not None and not isinstance(execution_model, nn.Module):
        raise TypeError("execution_model must be a torch.nn.Module")
    if not isinstance(writer, KroneckerCalibrationWriter):
        raise TypeError("writer must be a KroneckerCalibrationWriter")
    if curvature not in {"input-hessian", "guided-fisher", "forward-kl-kronecker"}:
        writer.abort()
        raise ValueError("unsupported embedding Kronecker curvature")
    if not writer.indexed_output:
        writer.abort()
        raise ValueError("embedding capture requires an indexed-output writer")
    if writer.curvature != curvature:
        writer.abort()
        raise ValueError("capture curvature differs from writer curvature")
    if curvature == "guided-fisher":
        if guided_loss_reduction not in {
            "sum",
            "mean-attention-mask",
            "mean-valid-causal-labels",
        }:
            writer.abort()
            raise ValueError(
                "guided-Fisher requires loss reduction sum, mean-attention-mask, "
                "or mean-valid-causal-labels"
            )
        objective = f"tritium.model-loss-guided-fisher.{guided_loss_reduction}@1"
    elif curvature == "forward-kl-kronecker":
        if guided_loss_reduction is not None:
            writer.abort()
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = "tritium.softmax-fisher-rademacher.single-probe@1"
    else:
        if guided_loss_reduction is not None:
            writer.abort()
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = "tritium.input-gram@1"
    if writer.objective_id != objective:
        writer.abort()
        raise ValueError("writer objective_id differs from the capture objective")
    if max_capture_bytes is None:
        max_capture_bytes = writer.max_batch_bytes
    if type(max_capture_bytes) is not int or max_capture_bytes <= 0:
        writer.abort()
        raise ValueError("max_capture_bytes must be a positive integer")
    if type(max_objective_bytes) is not int or max_objective_bytes <= 0:
        writer.abort()
        raise ValueError("max_objective_bytes must be a positive integer")

    modules = [
        value
        for path, value in model.named_modules(remove_duplicate=False)
        if path == module
    ]
    if len(modules) != 1:
        writer.abort()
        raise ValueError("module path must identify exactly one module")
    selected = modules[0]
    if not isinstance(selected, nn.Embedding) or tuple(selected.weight.shape) != (
        writer.rows,
        writer.columns,
    ):
        writer.abort()
        raise ValueError("embedding weight geometry differs from the evidence writer")

    pending: list[Tuple[torch.Tensor, torch.Tensor]] = []
    pending_index_bytes = 0
    token_mask: Optional[torch.Tensor] = None
    causal_selection_mask: Optional[torch.Tensor] = None

    def preflight_input(_selected, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("selected embedding must receive tensor token ids")
        if args[0].ndim == 0:
            raise ValueError("selected embedding token ids must have a sample dimension")

    def capture(_selected, args, output):
        nonlocal pending_index_bytes
        indices = args[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError("selected embedding must return one tensor")
        if tuple(output.shape) != tuple(indices.shape) + (writer.columns,):
            raise ValueError("selected embedding output geometry changed")
        if curvature != "input-hessian" and not output.requires_grad:
            output.requires_grad_(True)
        index_bytes = indices.numel() * indices.element_size()
        required_bytes = pending_index_bytes + index_bytes
        if required_bytes > max_capture_bytes:
            raise ValueError(
                f"capture snapshots require {required_bytes} bytes, limit is {max_capture_bytes}"
            )
        pending.append((indices.detach().clone(), output))
        pending_index_bytes = required_bytes

    pre_handle = selected.register_forward_pre_hook(preflight_input)
    handle = selected.register_forward_hook(capture)
    runner = model if execution_model is None else execution_model
    training_flags = tuple((component, component.training) for component in runner.modules())
    parameter_flags = _freeze_capture_parameters(runner, curvature=curvature)
    batches = 0
    module_calls = 0
    samples = 0
    selected_samples = 0
    input_segments = 0
    output_segments = 0
    finalizing = False
    objective_sample_start = 0
    try:
        runner.eval()
        for batch in data:
            pending.clear()
            raw_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            if raw_mask is not None and not isinstance(raw_mask, torch.Tensor):
                raise ValueError("attention_mask must be a tensor")
            if raw_mask is not None and not bool(((raw_mask == 0) | (raw_mask == 1)).all()):
                raise ValueError("attention_mask values must be boolean or 0/1")
            snapshot_bytes = 0 if raw_mask is None else raw_mask.numel() * raw_mask.element_size()
            raw_labels = None
            if guided_loss_reduction == "mean-valid-causal-labels":
                raw_labels = batch.get("labels") if isinstance(batch, Mapping) else None
                if not isinstance(raw_labels, torch.Tensor):
                    raise ValueError(
                        "mean-valid-causal-labels guided-Fisher requires tensor labels"
                    )
                snapshot_bytes += raw_labels.numel()
            if snapshot_bytes > max_capture_bytes:
                raise ValueError(
                    f"capture metadata requires {snapshot_bytes} bytes, limit is {max_capture_bytes}"
                )
            token_mask = None if raw_mask is None else raw_mask.detach().clone()
            causal_selection_mask = (
                None
                if raw_labels is None
                else _causal_selection_mask(raw_labels, token_mask)
            )
            pending_index_bytes = snapshot_bytes
            context = torch.no_grad() if curvature == "input-hessian" else torch.enable_grad()
            with context:
                output = _invoke_model(runner, batch)
                if not pending:
                    raise ValueError("selected embedding was not exercised by a calibration batch")
                if curvature == "input-hessian":
                    gradients = (None,) * len(pending)
                elif curvature == "guided-fisher":
                    loss = _capture_output_tensor(output, "loss")
                    if loss.numel() != 1 or not loss.isfinite().all():
                        raise ValueError("guided-Fisher model loss must be one finite scalar")
                    if guided_loss_reduction == "sum":
                        denominator = 1
                    elif guided_loss_reduction == "mean-attention-mask":
                        if token_mask is None:
                            raise ValueError(
                                "mean-attention-mask guided-Fisher requires attention_mask"
                            )
                        denominator = int(token_mask.count_nonzero().item())
                    else:
                        assert causal_selection_mask is not None
                        denominator = int(causal_selection_mask.count_nonzero().item())
                    if denominator <= 0:
                        raise ValueError("guided-Fisher loss denominator must be positive")
                    gradients = torch.autograd.grad(
                        loss * denominator, tuple(item[1] for item in pending)
                    )
                else:
                    logits = _capture_output_tensor(output, "logits")
                    factors = _forward_kl_factors(
                        logits,
                        objective_sample_start,
                        token_mask,
                        max_objective_bytes,
                    )
                    gradients = torch.autograd.grad(
                        logits,
                        tuple(item[1] for item in pending),
                        grad_outputs=factors,
                    )
                    objective_sample_start += logits.numel() // logits.shape[-1]
            for (indices, _), gradients_for_call in zip(pending, gradients):
                sample_shape = tuple(indices.shape)
                selected_mask = (
                    causal_selection_mask
                    if guided_loss_reduction == "mean-valid-causal-labels"
                    else token_mask
                )
                mask = _capture_token_mask(selected_mask, sample_shape)
                if curvature == "input-hessian":
                    input_segments, output_segments = writer.append_indexed_identity(
                        indices,
                        token_mask=mask,
                    )
                else:
                    input_segments, output_segments = writer.append_indexed(
                        gradients_for_call,
                        indices,
                        token_mask=mask,
                    )
                count = indices.numel()
                samples += count
                selected_samples += count if mask is None else int(mask.count_nonzero().item())
                module_calls += 1
            batches += 1
        if batches == 0:
            raise ValueError("calibration data must yield at least one batch")
        finalizing = True
        record = writer.finish()
    except BaseException as error:
        retryable_finish = finalizing and writer.active and isinstance(
            error,
            (
                _tritium.KroneckerPublicationError,
                _tritium.KroneckerResourceError,
            ),
        )
        if not retryable_finish:
            writer.abort()
        raise
    finally:
        pre_handle.remove()
        handle.remove()
        for component, training in training_flags:
            component.training = training
        _restore_capture_parameters(parameter_flags)
        pending.clear()
    return KroneckerModuleCaptureReceipt(
        module=module,
        curvature=curvature,
        objective=objective,
        batches=batches,
        module_calls=module_calls,
        samples=samples,
        selected_samples=selected_samples,
        input_segments=input_segments,
        output_segments=output_segments,
        record=record,
    )


def _qwen36_module_candidates(tensor_name: str, scope: str) -> Tuple[str, ...]:
    if not tensor_name.endswith(".weight"):
        raise ValueError("Qwen capture task must name a weight tensor")
    path = tensor_name.removesuffix(".weight")
    candidates = [path]
    if scope == "language" and path.startswith("model.language_model."):
        suffix = path.removeprefix("model.language_model.")
        candidates.extend((f"model.{suffix}", suffix))
    elif scope == "mtp-drafter" and path.startswith("mtp."):
        candidates.append(path.removeprefix("mtp."))
    return tuple(dict.fromkeys(candidates))


def _resolve_qwen36_capture_module(
    task: Any,
    language_model: nn.Module,
    mtp_model: Optional[nn.Module],
    output_head_model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, str, nn.Module]:
    if task.tensor_name == "lm_head.weight":
        if output_head_model is None and hasattr(language_model, "lm_head"):
            output_head_model = language_model
        if output_head_model is None:
            raise ValueError("Qwen output-head capture requires the root model")
        target = output_head_model
        candidates = ("lm_head",)
    elif task.scope == "language":
        target = language_model
    elif task.scope == "mtp-drafter":
        if mtp_model is None:
            raise ValueError("Qwen MTP capture task requires mtp_model")
        target = mtp_model
    else:
        raise ValueError(f"unsupported Qwen capture scope {task.scope!r}")
    modules = dict(target.named_modules(remove_duplicate=False))
    matches = [
        (candidate, modules[candidate])
        for candidate in _qwen36_module_candidates(task.tensor_name, task.scope)
        if candidate in modules
    ]
    unique = {id(value) for _, value in matches}
    if not matches:
        raise ValueError(f"cannot resolve Qwen capture module for {task.tensor_name}")
    if len(unique) != 1:
        raise ValueError(f"Qwen capture module aliases are ambiguous for {task.tensor_name}")
    return target, matches[0][0], matches[0][1]


def capture_qwen36_kronecker_evidence(
    language_model: nn.Module,
    data_factory: Callable[[Any], Iterable[Any]],
    *,
    model_dir: Pathish,
    declared_revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    curvature: str,
    activation_cache_digest: str,
    token_stream_digest: str,
    damping: float,
    mtp_model: Optional[nn.Module] = None,
    execution_model: Optional[nn.Module] = None,
    output_head_model: Optional[nn.Module] = None,
    guided_loss_reduction: Optional[str] = None,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    max_batch_bytes: int = 256 * 1024 * 1024,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
    max_shared_modules: int = 8,
    _session_factory=None,
):
    """Resume and complete the canonical pinned-Qwen grouped-evidence catalog.

    ``data_factory(task)`` must return a fresh replayable calibration iterable
    for each missing task window. Dense projections on the same model are
    captured in groups of at most ``max_shared_modules`` from one replay;
    ``task`` is the first canonical task in that window. Existing records are
    reused by the native session. Token embeddings remain sparse one-task
    captures. ``execution_model`` may be a full language-plus-MTP oracle that
    exercises selected modules nested under ``language_model`` or ``mtp_model``;
    this is required when those modules are not reached by ``language_model``'s
    own forward. The source checkpoint is admitted before any task is exposed.
    """

    if not isinstance(language_model, nn.Module):
        raise TypeError("language_model must be a torch.nn.Module")
    if mtp_model is not None and not isinstance(mtp_model, nn.Module):
        raise TypeError("mtp_model must be a torch.nn.Module")
    if execution_model is not None and not isinstance(execution_model, nn.Module):
        raise TypeError("execution_model must be a torch.nn.Module")
    if not callable(data_factory):
        raise TypeError("data_factory must be callable")
    if type(max_shared_modules) is not int or max_shared_modules <= 0:
        raise ValueError("max_shared_modules must be a positive integer")
    if curvature == "guided-fisher":
        if guided_loss_reduction not in {
            "sum",
            "mean-attention-mask",
            "mean-valid-causal-labels",
        }:
            raise ValueError("guided-Fisher requires an explicit supported loss reduction")
        objective = f"tritium.model-loss-guided-fisher.{guided_loss_reduction}@1"
    elif curvature == "forward-kl-kronecker":
        if guided_loss_reduction is not None:
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = "tritium.softmax-fisher-rademacher.single-probe@1"
    elif curvature == "input-hessian":
        if guided_loss_reduction is not None:
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = "tritium.input-gram@1"
    else:
        raise ValueError("unsupported Kronecker curvature")
    bound_cache_digest = bind_kronecker_activation_cache_digest(
        activation_cache_digest, objective
    )
    session_type = _tritium.Qwen36KroneckerCaptureSession
    if _session_factory is not None:
        session_type = _session_factory
    session = session_type(
        os.fspath(model_dir),
        declared_revision,
        os.fspath(work_dir),
        os.fspath(evidence_dir),
        curvature,
        bound_cache_digest,
        token_stream_digest,
        damping,
        max_evidence_bytes=max_evidence_bytes,
    )

    def validate_task(task: Any) -> None:
        if (
            task.curvature != curvature
            or task.activation_cache_digest != bound_cache_digest
            or task.token_stream_digest != token_stream_digest.lower()
            or task.damping != damping
        ):
            raise RuntimeError("native Qwen capture task drifted from the session contract")

    def writer_for(task: Any, *, indexed_output: bool) -> KroneckerCalibrationWriter:
        return KroneckerCalibrationWriter(
            evidence_dir,
            tensor_index=task.tensor_index,
            tensor_name=task.tensor_name,
            rows=task.rows,
            columns=task.columns,
            curvature=task.curvature,
            source_model_digest=task.source_model_digest,
            activation_cache_digest=activation_cache_digest,
            token_stream_digest=task.token_stream_digest,
            damping=task.damping,
            objective_id=objective,
            indexed_output=indexed_output,
            max_evidence_bytes=max_evidence_bytes,
            max_batch_bytes=max_batch_bytes,
        )

    def accept(task: Any) -> None:
        current = session.next_request()
        if current is None:
            # Last grouped record may have been validated while traversal
            # skipped from the first accepted record to a sealed catalog.
            return
        if current.tensor_index < task.tensor_index:
            raise RuntimeError("native Qwen capture acceptance order drifted")
        if current.tensor_index == task.tensor_index:
            if current.tensor_name != task.tensor_name:
                raise RuntimeError("native Qwen capture acceptance name drifted")
            if not session.accept_current():
                raise RuntimeError("Qwen evidence session refused the published record")
        # A grouped replay can publish later records before the first one is
        # accepted. Native session traversal validates those records and counts
        # them as reused while seeking the next missing task; no second
        # accept_current call is valid for them.

    while True:
        task = session.next_request()
        if task is None:
            receipt = session.finish()
            if receipt is None:
                raise RuntimeError("Qwen evidence session did not seal a complete catalog")
            return receipt
        validate_task(task)
        tasks = (task,)
        prefetch = getattr(session, "next_requests", None)
        if max_shared_modules > 1 and callable(prefetch):
            tasks = tuple(prefetch(max_shared_modules))
            if (
                not tasks
                or tasks[0].tensor_index != task.tensor_index
                or tasks[0].tensor_name != task.tensor_name
            ):
                raise RuntimeError("native Qwen capture prefetch order drifted")
        target, module_path, selected = _resolve_qwen36_capture_module(
            task, language_model, mtp_model, output_head_model
        )
        indexed_output = isinstance(selected, nn.Embedding)
        if not indexed_output:
            grouped = [(task, module_path, selected)]
            selected_ids = {id(selected)}
            for candidate in tasks[1:]:
                if candidate.scope != task.scope:
                    break
                validate_task(candidate)
                candidate_target, candidate_path, candidate_module = (
                    _resolve_qwen36_capture_module(
                        candidate, language_model, mtp_model, output_head_model
                    )
                )
                if (
                    candidate_target is not target
                    or isinstance(candidate_module, nn.Embedding)
                    or id(candidate_module) in selected_ids
                ):
                    break
                selected_ids.add(id(candidate_module))
                grouped.append((candidate, candidate_path, candidate_module))
            if len(grouped) > 1:
                writers = [writer_for(item, indexed_output=False) for item, _, _ in grouped]
                results = capture_kronecker_module_group(
                    target,
                    data_factory(grouped[0][0]),
                    modules=[path for _, path, _ in grouped],
                    writers=writers,
                    curvature=curvature,
                    execution_model=execution_model,
                    guided_loss_reduction=guided_loss_reduction,
                    max_capture_bytes=max_capture_bytes,
                    max_objective_bytes=max_objective_bytes,
                )
                for (expected, _, _), result in zip(grouped, results):
                    if result.record.tensor_index != expected.tensor_index:
                        raise RuntimeError("Qwen capture published the wrong tensor ordinal")
                    accept(expected)
                continue
        writer = writer_for(task, indexed_output=indexed_output)
        capture = capture_kronecker_embedding if indexed_output else capture_kronecker_module
        result = capture(
            target,
            data_factory(task),
            module=module_path,
            writer=writer,
            curvature=curvature,
            execution_model=execution_model,
            guided_loss_reduction=guided_loss_reduction,
            max_capture_bytes=max_capture_bytes,
            max_objective_bytes=max_objective_bytes,
        )
        if result.record.tensor_index != task.tensor_index:
            raise RuntimeError("Qwen capture published the wrong tensor ordinal")
        accept(task)


def capture_kronecker_module(
    model: nn.Module,
    data: Iterable[Any],
    *,
    module: str,
    writer: KroneckerCalibrationWriter,
    curvature: str,
    execution_model: Optional[nn.Module] = None,
    guided_loss_reduction: Optional[str] = None,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
) -> KroneckerModuleCaptureReceipt:
    """Capture one projection through a bounded model-aware PyTorch pass.

    ``execution_model`` can be a containing/oracle module when the selected
    projection is not the module that owns the calibration forward. Hooks stay
    attached to ``model``; batches execute through ``execution_model``.

    ``input-hessian`` accumulates input Gram factors without autograd.
    ``guided-fisher`` differentiates the model's exact scalar ``loss`` output.
    ``forward-kl-kronecker`` uses a stateless v1 Rademacher factor of the dense
    softmax Fisher. Only gradients with respect to the selected module outputs
    are requested, so parameter ``.grad`` buffers are never allocated.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if execution_model is not None and not isinstance(execution_model, nn.Module):
        raise TypeError("execution_model must be a torch.nn.Module")
    if not isinstance(writer, KroneckerCalibrationWriter):
        raise TypeError("writer must be a KroneckerCalibrationWriter")
    if curvature not in {
        "input-hessian",
        "guided-fisher",
        "forward-kl-kronecker",
    }:
        raise ValueError("unsupported Kronecker curvature")
    if writer.curvature != curvature:
        writer.abort()
        raise ValueError("capture curvature differs from writer curvature")
    if curvature == "guided-fisher":
        if guided_loss_reduction not in {
            "sum",
            "mean-attention-mask",
            "mean-valid-causal-labels",
        }:
            writer.abort()
            raise ValueError(
                "guided-Fisher requires loss reduction sum, mean-attention-mask, "
                "or mean-valid-causal-labels"
            )
        objective = f"tritium.model-loss-guided-fisher.{guided_loss_reduction}@1"
    else:
        if guided_loss_reduction is not None:
            writer.abort()
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = {
            "input-hessian": "tritium.input-gram@1",
            "forward-kl-kronecker": "tritium.softmax-fisher-rademacher.single-probe@1",
        }[curvature]
    if writer.objective_id != objective:
        writer.abort()
        raise ValueError("writer objective_id differs from the capture objective")
    if max_capture_bytes is None:
        max_capture_bytes = writer.max_batch_bytes
    if type(max_capture_bytes) is not int or max_capture_bytes <= 0:
        writer.abort()
        raise ValueError("max_capture_bytes must be a positive integer")
    if type(max_objective_bytes) is not int or max_objective_bytes <= 0:
        writer.abort()
        raise ValueError("max_objective_bytes must be a positive integer")

    modules = [
        value
        for path, value in model.named_modules(remove_duplicate=False)
        if path == module
    ]
    if len(modules) != 1:
        writer.abort()
        raise ValueError("module path must identify exactly one module")
    selected = modules[0]
    weight = getattr(selected, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or tuple(weight.shape) != (writer.rows, writer.columns)
    ):
        writer.abort()
        raise ValueError("module weight geometry differs from the evidence writer")

    pending: list[Tuple[torch.Tensor, torch.Tensor]] = []
    pending_activation_bytes = 0
    token_mask: Optional[torch.Tensor] = None
    causal_selection_mask: Optional[torch.Tensor] = None

    def preflight_input(_selected, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("selected module must receive a tensor first argument")
        activations = args[0]
        if activations.ndim == 0 or activations.shape[-1] != writer.columns:
            raise ValueError("selected module input feature geometry changed")

    def capture(_selected, args, output):
        nonlocal pending_activation_bytes
        activations = args[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError("selected module must return one tensor")
        if tuple(output.shape[:-1]) != tuple(activations.shape[:-1]) or output.shape[-1] != writer.rows:
            raise ValueError("selected module output geometry changed")
        if curvature != "input-hessian" and not output.requires_grad:
            output.requires_grad_(True)
        activation_bytes = activations.numel() * activations.element_size()
        required_bytes = pending_activation_bytes + activation_bytes
        if required_bytes > max_capture_bytes:
            raise ValueError(
                f"capture snapshots require {required_bytes} bytes, limit is {max_capture_bytes}"
            )
        pending.append((activations.detach().clone(), output))
        pending_activation_bytes = required_bytes

    pre_handle = selected.register_forward_pre_hook(preflight_input)
    handle = selected.register_forward_hook(capture)
    runner = model if execution_model is None else execution_model
    training_flags = tuple((component, component.training) for component in runner.modules())
    parameter_flags = _freeze_capture_parameters(runner, curvature=curvature)
    batches = 0
    module_calls = 0
    samples = 0
    selected_samples = 0
    input_segments = 0
    output_segments = 0
    finalizing = False
    objective_sample_start = 0
    try:
        runner.eval()
        for batch in data:
            pending.clear()
            raw_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            if raw_mask is not None and not isinstance(raw_mask, torch.Tensor):
                raise ValueError("attention_mask must be a tensor")
            if raw_mask is not None and not bool(
                ((raw_mask == 0) | (raw_mask == 1)).all()
            ):
                raise ValueError("attention_mask values must be boolean or 0/1")
            snapshot_bytes = (
                0 if raw_mask is None else raw_mask.numel() * raw_mask.element_size()
            )
            raw_labels = None
            if guided_loss_reduction == "mean-valid-causal-labels":
                raw_labels = batch.get("labels") if isinstance(batch, Mapping) else None
                if not isinstance(raw_labels, torch.Tensor):
                    raise ValueError(
                        "mean-valid-causal-labels guided-Fisher requires tensor labels"
                    )
                snapshot_bytes += raw_labels.numel()
            if snapshot_bytes > max_capture_bytes:
                raise ValueError(
                    f"capture metadata requires {snapshot_bytes} bytes, "
                    f"limit is {max_capture_bytes}"
                )
            token_mask = None if raw_mask is None else raw_mask.detach().clone()
            causal_selection_mask = (
                None
                if raw_labels is None
                else _causal_selection_mask(raw_labels, token_mask)
            )
            pending_activation_bytes = snapshot_bytes
            context = torch.no_grad() if curvature == "input-hessian" else torch.enable_grad()
            with context:
                output = _invoke_model(runner, batch)
                if not pending:
                    raise ValueError("selected module was not exercised by a calibration batch")
                if curvature == "guided-fisher":
                    loss = _capture_output_tensor(output, "loss")
                    if loss.numel() != 1 or not loss.isfinite().all():
                        raise ValueError("guided-Fisher model loss must be one finite scalar")
                    if guided_loss_reduction == "sum":
                        denominator = 1
                    elif guided_loss_reduction == "mean-attention-mask":
                        if token_mask is None:
                            raise ValueError(
                                "mean-attention-mask guided-Fisher requires attention_mask"
                            )
                        denominator = int(token_mask.count_nonzero().item())
                    else:
                        assert causal_selection_mask is not None
                        denominator = int(causal_selection_mask.count_nonzero().item())
                    if denominator <= 0:
                        raise ValueError("guided-Fisher loss denominator must be positive")
                    gradients = torch.autograd.grad(
                        loss * denominator, tuple(item[1] for item in pending)
                    )
                elif curvature == "forward-kl-kronecker":
                    logits = _capture_output_tensor(output, "logits")
                    factors = _forward_kl_factors(
                        logits,
                        objective_sample_start,
                        token_mask,
                        max_objective_bytes,
                    )
                    gradients = torch.autograd.grad(
                        logits,
                        tuple(item[1] for item in pending),
                        grad_outputs=factors,
                    )
                    objective_sample_start += logits.numel() // logits.shape[-1]
                else:
                    gradients = (None,) * len(pending)
            for (activations, _), factors in zip(pending, gradients):
                sample_shape = tuple(activations.shape[:-1])
                selected_mask = (
                    causal_selection_mask
                    if guided_loss_reduction == "mean-valid-causal-labels"
                    else token_mask
                )
                mask = _capture_token_mask(selected_mask, sample_shape)
                input_segments, output_segments = writer.append(
                    activations,
                    factors,
                    token_mask=mask,
                )
                count = activations.numel() // writer.columns
                samples += count
                selected_samples += count if mask is None else int(mask.count_nonzero().item())
                module_calls += 1
            batches += 1
        if batches == 0:
            raise ValueError("calibration data must yield at least one batch")
        finalizing = True
        record = writer.finish()
    except BaseException as error:
        retryable_finish = finalizing and writer.active and isinstance(
            error,
            (
                _tritium.KroneckerPublicationError,
                _tritium.KroneckerResourceError,
            ),
        )
        if not retryable_finish:
            writer.abort()
        raise
    finally:
        pre_handle.remove()
        handle.remove()
        for component, training in training_flags:
            component.training = training
        _restore_capture_parameters(parameter_flags)
        pending.clear()
    return KroneckerModuleCaptureReceipt(
        module=module,
        curvature=curvature,
        objective=objective,
        batches=batches,
        module_calls=module_calls,
        samples=samples,
        selected_samples=selected_samples,
        input_segments=input_segments,
        output_segments=output_segments,
        record=record,
    )


def capture_kronecker_module_group(
    model: nn.Module,
    data: Iterable[Any],
    *,
    modules: Sequence[str],
    writers: Sequence[KroneckerCalibrationWriter],
    curvature: str,
    execution_model: Optional[nn.Module] = None,
    guided_loss_reduction: Optional[str] = None,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
) -> Tuple[KroneckerModuleCaptureReceipt, ...]:
    """Capture several projections through one shared bounded PyTorch pass.

    ``execution_model`` has the same containing/oracle semantics as the
    single-module capture path.

    One forward pass per calibration batch feeds every target module's writer,
    replacing one full-calibration replay per tensor. Guided-Fisher and
    forward-KL obtain every module-output gradient in one backward via
    ``torch.autograd.grad``, so parameter ``.grad`` buffers are never
    allocated. Each writer receives exactly the activations, factors, masks,
    and global sample ordinals the per-tensor :func:`capture_kronecker_module`
    path would produce, so published records are byte-identical to per-tensor
    records for the same frozen inputs (ADR 0035 WS-A3).
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if execution_model is not None and not isinstance(execution_model, nn.Module):
        raise TypeError("execution_model must be a torch.nn.Module")
    if isinstance(modules, str) or not isinstance(modules, Sequence):
        raise TypeError("modules must be a sequence of module paths")
    if not isinstance(writers, Sequence):
        raise TypeError("writers must be a sequence of KroneckerCalibrationWriter")
    if any(not isinstance(writer, KroneckerCalibrationWriter) for writer in writers):
        raise TypeError("writers must be a sequence of KroneckerCalibrationWriter")

    def abort_all() -> None:
        for writer in writers:
            writer.abort()

    if len(modules) == 0 or len(modules) != len(writers):
        abort_all()
        raise ValueError("modules and writers must be equal-length and nonempty")
    if any(not isinstance(path, str) for path in modules):
        abort_all()
        raise ValueError("modules must be a sequence of module paths")
    if len(set(modules)) != len(modules):
        abort_all()
        raise ValueError("modules must not repeat one module path")
    if len({id(writer) for writer in writers}) != len(writers):
        abort_all()
        raise ValueError("writers must not repeat one writer")
    if curvature not in {
        "input-hessian",
        "guided-fisher",
        "forward-kl-kronecker",
    }:
        abort_all()
        raise ValueError("unsupported Kronecker curvature")
    if curvature == "guided-fisher":
        if guided_loss_reduction not in {
            "sum",
            "mean-attention-mask",
            "mean-valid-causal-labels",
        }:
            abort_all()
            raise ValueError(
                "guided-Fisher requires loss reduction sum, mean-attention-mask, "
                "or mean-valid-causal-labels"
            )
        objective = f"tritium.model-loss-guided-fisher.{guided_loss_reduction}@1"
    else:
        if guided_loss_reduction is not None:
            abort_all()
            raise ValueError("guided_loss_reduction is valid only for guided-fisher")
        objective = {
            "input-hessian": "tritium.input-gram@1",
            "forward-kl-kronecker": "tritium.softmax-fisher-rademacher.single-probe@1",
        }[curvature]
    for writer in writers:
        if writer.curvature != curvature:
            abort_all()
            raise ValueError("capture curvature differs from writer curvature")
        if writer.objective_id != objective:
            abort_all()
            raise ValueError("writer objective_id differs from the capture objective")
        if writer.indexed_output:
            abort_all()
            raise ValueError(
                "group capture supports dense writers; embeddings use "
                "capture_kronecker_embedding"
            )
    if max_capture_bytes is None:
        max_capture_bytes = min(writer.max_batch_bytes for writer in writers)
    if type(max_capture_bytes) is not int or max_capture_bytes <= 0:
        abort_all()
        raise ValueError("max_capture_bytes must be a positive integer")
    if type(max_objective_bytes) is not int or max_objective_bytes <= 0:
        abort_all()
        raise ValueError("max_objective_bytes must be a positive integer")

    selected_modules = []
    for path, writer in zip(modules, writers):
        candidates = [
            value
            for candidate, value in model.named_modules(remove_duplicate=False)
            if candidate == path
        ]
        if len(candidates) != 1:
            abort_all()
            raise ValueError("module path must identify exactly one module")
        selected = candidates[0]
        weight = getattr(selected, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or tuple(weight.shape) != (writer.rows, writer.columns)
        ):
            abort_all()
            raise ValueError("module weight geometry differs from the evidence writer")
        selected_modules.append(selected)
    if len({id(module) for module in selected_modules}) != len(selected_modules):
        abort_all()
        raise ValueError("module paths must identify distinct modules")

    pending: list[list[Tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in modules]
    pending_activation_bytes = 0
    token_mask: Optional[torch.Tensor] = None
    causal_selection_mask: Optional[torch.Tensor] = None

    def preflight_for(index: int):
        writer = writers[index]

        def preflight_input(_selected, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError("selected module must receive a tensor first argument")
            activations = args[0]
            if activations.ndim == 0 or activations.shape[-1] != writer.columns:
                raise ValueError("selected module input feature geometry changed")

        return preflight_input

    def capture_for(index: int):
        writer = writers[index]

        def capture(_selected, args, output):
            nonlocal pending_activation_bytes
            activations = args[0]
            if not isinstance(output, torch.Tensor):
                raise ValueError("selected module must return one tensor")
            if (
                tuple(output.shape[:-1]) != tuple(activations.shape[:-1])
                or output.shape[-1] != writer.rows
            ):
                raise ValueError("selected module output geometry changed")
            if curvature != "input-hessian" and not output.requires_grad:
                output.requires_grad_(True)
            activation_bytes = activations.numel() * activations.element_size()
            required_bytes = pending_activation_bytes + activation_bytes
            if required_bytes > max_capture_bytes:
                raise ValueError(
                    f"capture snapshots require {required_bytes} bytes, "
                    f"limit is {max_capture_bytes}"
                )
            pending[index].append((activations.detach().clone(), output))
            pending_activation_bytes = required_bytes

        return capture

    handles = []
    for index, selected in enumerate(selected_modules):
        handles.append(selected.register_forward_pre_hook(preflight_for(index)))
        handles.append(selected.register_forward_hook(capture_for(index)))
    runner = model if execution_model is None else execution_model
    training_flags = tuple((component, component.training) for component in runner.modules())
    parameter_flags = _freeze_capture_parameters(runner, curvature=curvature)
    batches = 0
    module_calls = [0] * len(modules)
    samples = [0] * len(modules)
    selected_samples = [0] * len(modules)
    segments = [(0, 0)] * len(modules)
    finalizing = False
    objective_sample_start = 0
    records = []
    try:
        runner.eval()
        for batch in data:
            for slot in pending:
                slot.clear()
            raw_mask = batch.get("attention_mask") if isinstance(batch, Mapping) else None
            if raw_mask is not None and not isinstance(raw_mask, torch.Tensor):
                raise ValueError("attention_mask must be a tensor")
            if raw_mask is not None and not bool(
                ((raw_mask == 0) | (raw_mask == 1)).all()
            ):
                raise ValueError("attention_mask values must be boolean or 0/1")
            snapshot_bytes = (
                0 if raw_mask is None else raw_mask.numel() * raw_mask.element_size()
            )
            raw_labels = None
            if guided_loss_reduction == "mean-valid-causal-labels":
                raw_labels = batch.get("labels") if isinstance(batch, Mapping) else None
                if not isinstance(raw_labels, torch.Tensor):
                    raise ValueError(
                        "mean-valid-causal-labels guided-Fisher requires tensor labels"
                    )
                snapshot_bytes += raw_labels.numel()
            if snapshot_bytes > max_capture_bytes:
                raise ValueError(
                    f"capture metadata requires {snapshot_bytes} bytes, "
                    f"limit is {max_capture_bytes}"
                )
            token_mask = None if raw_mask is None else raw_mask.detach().clone()
            causal_selection_mask = (
                None
                if raw_labels is None
                else _causal_selection_mask(raw_labels, token_mask)
            )
            pending_activation_bytes = snapshot_bytes
            context = torch.no_grad() if curvature == "input-hessian" else torch.enable_grad()
            with context:
                output = _invoke_model(runner, batch)
                if any(not slot for slot in pending):
                    raise ValueError("selected module was not exercised by a calibration batch")
                flat_outputs = tuple(item[1] for slot in pending for item in slot)
                if curvature == "guided-fisher":
                    loss = _capture_output_tensor(output, "loss")
                    if loss.numel() != 1 or not loss.isfinite().all():
                        raise ValueError("guided-Fisher model loss must be one finite scalar")
                    if guided_loss_reduction == "sum":
                        denominator = 1
                    elif guided_loss_reduction == "mean-attention-mask":
                        if token_mask is None:
                            raise ValueError(
                                "mean-attention-mask guided-Fisher requires attention_mask"
                            )
                        denominator = int(token_mask.count_nonzero().item())
                    else:
                        assert causal_selection_mask is not None
                        denominator = int(causal_selection_mask.count_nonzero().item())
                    if denominator <= 0:
                        raise ValueError("guided-Fisher loss denominator must be positive")
                    gradients = torch.autograd.grad(loss * denominator, flat_outputs)
                elif curvature == "forward-kl-kronecker":
                    logits = _capture_output_tensor(output, "logits")
                    factors = _forward_kl_factors(
                        logits,
                        objective_sample_start,
                        token_mask,
                        max_objective_bytes,
                    )
                    gradients = torch.autograd.grad(
                        logits,
                        flat_outputs,
                        grad_outputs=factors,
                    )
                    objective_sample_start += logits.numel() // logits.shape[-1]
                else:
                    gradients = (None,) * len(flat_outputs)
            cursor = 0
            for index, slot in enumerate(pending):
                writer = writers[index]
                for activations, _ in slot:
                    factors = gradients[cursor]
                    cursor += 1
                    sample_shape = tuple(activations.shape[:-1])
                    selected_mask = (
                        causal_selection_mask
                        if guided_loss_reduction == "mean-valid-causal-labels"
                        else token_mask
                    )
                    mask = _capture_token_mask(selected_mask, sample_shape)
                    segments[index] = writer.append(
                        activations,
                        factors,
                        token_mask=mask,
                    )
                    count = activations.numel() // writer.columns
                    samples[index] += count
                    selected_samples[index] += (
                        count if mask is None else int(mask.count_nonzero().item())
                    )
                    module_calls[index] += 1
            batches += 1
        if batches == 0:
            raise ValueError("calibration data must yield at least one batch")
        finalizing = True
        for writer in writers:
            records.append(writer.finish())
    except BaseException as error:
        retryable_finish = finalizing and isinstance(
            error,
            (
                _tritium.KroneckerPublicationError,
                _tritium.KroneckerResourceError,
            ),
        ) and any(writer.active for writer in writers)
        if not retryable_finish:
            abort_all()
        raise
    finally:
        for handle in handles:
            handle.remove()
        for component, training in training_flags:
            component.training = training
        _restore_capture_parameters(parameter_flags)
        for slot in pending:
            slot.clear()
    return tuple(
        KroneckerModuleCaptureReceipt(
            module=path,
            curvature=curvature,
            objective=objective,
            batches=batches,
            module_calls=module_calls[index],
            samples=samples[index],
            selected_samples=selected_samples[index],
            input_segments=segments[index][0],
            output_segments=segments[index][1],
            record=records[index],
        )
        for index, path in enumerate(modules)
    )


def _hash_field(digest: "hashlib._Hash", tag: str, payload: bytes) -> None:
    encoded = tag.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _hash_tensor(digest: "hashlib._Hash", tag: str, value: torch.Tensor) -> None:
    tensor = value.detach()
    _hash_field(digest, f"{tag}:dtype", str(tensor.dtype).encode("ascii"))
    _hash_field(
        digest,
        f"{tag}:shape",
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
    )
    flat = tensor.contiguous().view(-1)
    chunk_elements = max(1, (1024 * 1024) // max(1, flat.element_size()))
    for offset in range(0, flat.numel(), chunk_elements):
        chunk = (
            flat[offset : offset + chunk_elements]
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        _hash_field(digest, f"{tag}:chunk", chunk)


def _hash_value(digest: "hashlib._Hash", tag: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _hash_tensor(digest, tag, value)
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("calibration batch mappings require string keys")
        for key in sorted(value):
            _hash_value(digest, f"{tag}.{key}", value[key])
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _hash_value(digest, f"{tag}[{index}]", item)
    elif value is None or type(value) in {bool, int, float, str}:
        _hash_field(
            digest,
            tag,
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        )
    else:
        raise TypeError(
            f"unsupported calibration batch value at {tag}: {type(value).__name__}"
        )


def _source_model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        _hash_tensor(digest, f"state.{name}", value)
    return f"sha256:{digest.hexdigest()}"


def _selected_linear_modules(
    prepared: PreparedModel,
) -> Tuple[_SelectedLinear, ...]:
    assert isinstance(prepared.model, nn.Module)
    selected = {
        alias
        for entry in prepared.coverage.entries
        if entry.disposition == "selected"
        for alias in entry.aliases
    }
    records: Dict[int, dict[str, Any]] = {}
    for path, module in prepared.model.named_modules(remove_duplicate=False):
        weight_name = f"{path}.weight" if path else "weight"
        if not isinstance(module, nn.Linear) or weight_name not in selected:
            continue
        weight_key = id(module.weight)
        record = records.get(weight_key)
        if record is None:
            aliases = next(
                entry.aliases
                for entry in prepared.coverage.entries
                if weight_name in entry.aliases
            )
            record = {
                "path": path,
                "module": module,
                "aliases": tuple(aliases),
                "capture_modules": [],
            }
            records[weight_key] = record
        if all(id(candidate) != id(module) for candidate in record["capture_modules"]):
            record["capture_modules"].append(module)
    result = tuple(
        _SelectedLinear(
            path=record["path"],
            module=record["module"],
            aliases=record["aliases"],
            capture_modules=tuple(record["capture_modules"]),
        )
        for record in records.values()
    )
    if not result:
        raise TritiumError(
            "raw calibration currently requires at least one selected Linear module",
            code="unsupported_module",
            stage="calibrate",
        )
    covered = {alias for record in result for alias in record.aliases}
    unsupported = sorted(
        entry.path
        for entry in prepared.coverage.entries
        if entry.disposition == "selected"
        and not covered.intersection(entry.aliases)
    )
    if unsupported:
        raise TritiumError(
            "raw diagonal calibration cannot cover selected non-Linear weights",
            code="unsupported_module",
            stage="calibrate",
            details={"parameters": unsupported},
        )
    return result


def _invoke_model(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def _json_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate calibration manifest field {key!r}")
        value[key] = item
    return value


def load_activation_calibration(
    evidence_dir: Pathish, *, max_evidence_bytes: int = 64 * 1024 * 1024
) -> ActivationCalibrationReceipt:
    """Strictly reopen and rehash streamed diagonal-curvature evidence."""

    if max_evidence_bytes <= 0:
        raise ValueError("max_evidence_bytes must be positive")
    requested = Path(evidence_dir)
    if requested.is_symlink():
        raise ValueError("activation evidence directory must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("activation evidence must be an ordinary directory")
    manifest_path = directory / "calibration.json"
    metadata = manifest_path.lstat()
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or metadata.st_size > 1024 * 1024
    ):
        raise ValueError(
            "calibration.json must be an ordinary manifest no larger than 1 MiB"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_json_without_duplicates)
    fields = {
        "schema_version",
        "curvature",
        "source_model_digest",
        "activation_cache_digest",
        "token_stream_digest",
        "record_count",
        "records",
        "evidence_id",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("calibration manifest fields do not match schema version 1")
    if manifest["schema_version"] != 2:
        raise ValueError("unsupported activation calibration schema_version")
    if manifest["curvature"] != "diagonal-second-moment-f64le":
        raise ValueError("unsupported activation curvature representation")
    for field in (
        "source_model_digest",
        "activation_cache_digest",
        "token_stream_digest",
        "evidence_id",
    ):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
        ):
            raise ValueError(f"invalid {field}")
        try:
            bytes.fromhex(value[7:])
        except ValueError as error:
            raise ValueError(f"invalid {field}") from error
    values = manifest["records"]
    if (
        type(manifest["record_count"]) is not int
        or manifest["record_count"] <= 0
        or not isinstance(values, list)
        or len(values) != manifest["record_count"]
    ):
        raise ValueError("activation record_count does not match records")
    record_fields = {
        "module",
        "weight_aliases",
        "features",
        "outputs",
        "samples",
        "file",
        "digest",
        "bytes",
    }
    records = []
    cache_digest = hashlib.sha256()
    total_bytes = 0
    expected_files = {"calibration.json"}
    seen_aliases: set[str] = set()
    seen_modules: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != record_fields:
            raise ValueError("activation record fields do not match schema version 1")
        filename = f"curvature-{index:05d}.f64le"
        if item["file"] != filename:
            raise ValueError(
                "activation records are missing, duplicated, or out of order"
            )
        aliases = item["weight_aliases"]
        if (
            not isinstance(item["module"], str)
            or not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise ValueError("activation record module aliases are invalid")
        if len(set(aliases)) != len(aliases):
            raise ValueError("activation record aliases must be unique")
        if item["module"] in seen_modules:
            raise ValueError("activation record modules must be unique")
        if seen_aliases.intersection(aliases):
            raise ValueError("activation record aliases must be globally unique")
        seen_modules.add(item["module"])
        seen_aliases.update(aliases)
        if (
            type(item["features"]) is not int
            or item["features"] <= 0
            or type(item["outputs"]) is not int
            or item["outputs"] <= 0
            or type(item["samples"]) is not int
            or item["samples"] <= 0
            or type(item["bytes"]) is not int
            or item["bytes"] != item["features"] * 8
        ):
            raise ValueError("activation record geometry or byte ledger is invalid")
        path = directory / filename
        file_metadata = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or file_metadata.st_size != item["bytes"]
        ):
            raise ValueError("activation record is not an exact ordinary file")
        payload = path.read_bytes()
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if item["digest"] != digest:
            raise ValueError("activation record digest mismatch")
        _hash_field(cache_digest, filename, payload)
        total_bytes += len(payload)
        if total_bytes > max_evidence_bytes:
            raise ValueError("activation evidence exceeds max_evidence_bytes")
        expected_files.add(filename)
        records.append(
            ActivationRecord(
                module=item["module"],
                weight_aliases=tuple(aliases),
                features=item["features"],
                outputs=item["outputs"],
                samples=item["samples"],
                file=filename,
                digest=digest,
                bytes=item["bytes"],
            )
        )
    if {child.name for child in directory.iterdir()} != expected_files:
        raise ValueError("activation evidence directory contains unknown files")
    actual_cache_digest = f"sha256:{cache_digest.hexdigest()}"
    if manifest["activation_cache_digest"] != actual_cache_digest:
        raise ValueError("activation cache digest mismatch")
    identity = dict(manifest)
    evidence_id = identity.pop("evidence_id")
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if evidence_id != f"sha256:{hashlib.sha256(canonical).hexdigest()}":
        raise ValueError("activation evidence identity mismatch")
    return ActivationCalibrationReceipt(
        evidence_dir=directory,
        evidence_id=evidence_id,
        curvature=manifest["curvature"],
        record_count=len(records),
        source_model_digest=manifest["source_model_digest"],
        activation_cache_digest=actual_cache_digest,
        token_stream_digest=manifest["token_stream_digest"],
        max_evidence_bytes=max_evidence_bytes,
        records=tuple(records),
    )


def _collect_activations(
    prepared: PreparedModel,
    data: Iterable[Any],
    evidence_dir: Path,
    max_evidence_bytes: int,
) -> ActivationCalibrationReceipt:
    if max_evidence_bytes <= 0:
        raise ValueError("max_evidence_bytes must be positive")
    if evidence_dir.exists() or evidence_dir.is_symlink():
        raise FileExistsError(f"calibration evidence already exists: {evidence_dir}")
    modules = _selected_linear_modules(prepared)
    required_bytes = sum(record.module.in_features * 8 for record in modules)
    if required_bytes > max_evidence_bytes:
        raise TritiumError(
            "activation evidence exceeds max_evidence_bytes",
            code="evidence_too_large",
            stage="calibrate",
            details={
                "required_bytes": required_bytes,
                "max_evidence_bytes": max_evidence_bytes,
            },
        )
    accumulators: Dict[str, torch.Tensor] = {
        record.path: torch.zeros(record.module.in_features, dtype=torch.float64)
        for record in modules
    }
    samples = {record.path: 0 for record in modules}
    handles = []

    def hook_for(path: str):
        def capture(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise TritiumError(
                    "selected Linear module did not receive a tensor input",
                    code="invalid_calibration_batch",
                    stage="calibrate",
                    module=path,
                )
            value = args[0].detach()
            if value.shape[-1] != accumulators[path].numel():
                raise TritiumError(
                    "Linear calibration input width changed",
                    code="invalid_calibration_batch",
                    stage="calibrate",
                    module=path,
                )
            rows = value.reshape(-1, value.shape[-1])
            chunk_rows = max(1, (1024 * 1024) // rows.shape[-1])
            for offset in range(0, rows.shape[0], chunk_rows):
                chunk = rows[offset : offset + chunk_rows].to(torch.float64)
                accumulators[path].add_(chunk.square().sum(dim=0).cpu())
            samples[path] += rows.shape[0]

        return capture

    for record in modules:
        for module in record.capture_modules:
            handles.append(module.register_forward_pre_hook(hook_for(record.path)))

    model = prepared.model
    # Calibration temporarily forces eval semantics, but callers may keep
    # mixed train/eval islands (for example frozen normalization beside a
    # train-mode head). Restore every module's exact prior flag.
    training_flags = tuple(
        (component, component.training) for component in model.modules()
    )
    token_digest = hashlib.sha256()
    batches = 0
    try:
        model.eval()
        with torch.no_grad():
            for batches, batch in enumerate(data, 1):
                _hash_value(token_digest, f"batch[{batches - 1}]", batch)
                _invoke_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        for component, training in training_flags:
            component.training = training
    if batches == 0:
        raise TritiumError(
            "calibration data must yield at least one batch",
            code="invalid_calibration_batch",
            stage="calibrate",
        )
    empty = [path for path, count in samples.items() if count == 0]
    if empty:
        raise TritiumError(
            "selected modules were not all exercised by calibration data",
            code="incomplete_coverage",
            stage="calibrate",
            details={"modules": empty},
        )

    parent = evidence_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / evidence_dir.name
    staging = Path(
        tempfile.mkdtemp(prefix=f".{evidence_dir.name}.", dir=str(parent))
    )
    activation_digest = hashlib.sha256()
    record_values = []
    records = []
    try:
        for index, record in enumerate(modules):
            path = record.path
            payload = accumulators[path].numpy().astype("<f8", copy=False).tobytes()
            if len(payload) > max_evidence_bytes:
                raise TritiumError(
                    "activation evidence exceeds max_evidence_bytes",
                    code="evidence_too_large",
                    stage="calibrate",
                )
            filename = f"curvature-{index:05d}.f64le"
            (staging / filename).write_bytes(payload)
            digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            _hash_field(activation_digest, filename, payload)
            record = ActivationRecord(
                module=path,
                weight_aliases=record.aliases,
                features=accumulators[path].numel(),
                outputs=record.module.out_features,
                samples=samples[path],
                file=filename,
                digest=digest,
                bytes=len(payload),
            )
            records.append(record)
            record_values.append(
                {
                    "module": record.module,
                    "weight_aliases": list(record.weight_aliases),
                    "features": record.features,
                    "outputs": record.outputs,
                    "samples": record.samples,
                    "file": record.file,
                    "digest": record.digest,
                    "bytes": record.bytes,
                }
            )
        total_bytes = sum(record.bytes for record in records)
        if total_bytes > max_evidence_bytes:
            raise TritiumError(
                "activation evidence exceeds max_evidence_bytes",
                code="evidence_too_large",
                stage="calibrate",
                details={
                    "required_bytes": total_bytes,
                    "max_evidence_bytes": max_evidence_bytes,
                },
            )
        source_digest = _source_model_digest(model)
        cache_digest = f"sha256:{activation_digest.hexdigest()}"
        stream_digest = f"sha256:{token_digest.hexdigest()}"
        manifest = {
            "schema_version": 2,
            "curvature": "diagonal-second-moment-f64le",
            "source_model_digest": source_digest,
            "activation_cache_digest": cache_digest,
            "token_stream_digest": stream_digest,
            "record_count": len(records),
            "records": record_values,
        }
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        evidence_id = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        manifest["evidence_id"] = evidence_id
        (staging / "calibration.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _tritium.publish_directory_noreplace(str(staging), str(target))
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return load_activation_calibration(
        target, max_evidence_bytes=max_evidence_bytes
    )


def _scale_group_size(columns: int) -> int:
    """Choose SALT's canonical scale geometry for one matrix width."""

    if columns % 128 == 0:
        return 128
    if columns % 64 == 0:
        return 64
    return columns


def _diagonal_additive_projection(
    master: torch.Tensor, curvature: torch.Tensor, planes: int
) -> TernaryProjection:
    if master.ndim != 2 or curvature.ndim != 1 or curvature.numel() != master.shape[1]:
        raise TritiumError(
            "calibration curvature does not match the selected weight",
            code="evidence_geometry_mismatch",
            stage="convert",
        )
    master_f64 = master.detach().to(dtype=torch.float64, device="cpu")
    diagonal = curvature.to(dtype=torch.float64, device="cpu")
    mean = diagonal.mean()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal < 0).any()):
        raise TritiumError(
            "calibration curvature must be finite and nonnegative",
            code="invalid_evidence",
            stage="convert",
        )
    diagonal = (
        torch.ones_like(diagonal)
        if float(mean) == 0.0
        else diagonal + mean * 1e-4
    )
    group_size = _scale_group_size(master.shape[1])
    groups = (master.shape[1] + group_size - 1) // group_size
    grouped_master = master_f64.reshape(master.shape[0], groups, group_size)
    grouped_diagonal = diagonal.reshape(groups, group_size)
    trit_values = []
    scale_values = []
    residual = grouped_master
    for _ in range(planes):
        initial_scale = (residual.abs() * grouped_diagonal).sum(dim=2)
        initial_scale = initial_scale / grouped_diagonal.sum(dim=1).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        nonzero_scale = initial_scale.clamp_min(torch.finfo(torch.float64).tiny)
        trits = (
            (residual / nonzero_scale.unsqueeze(-1))
            .round()
            .clamp(-1, 1)
            .to(torch.int8)
        )
        trits_f64 = trits.to(torch.float64)
        denominator = (trits_f64.square() * grouped_diagonal).sum(dim=2)
        numerator = (residual * trits_f64 * grouped_diagonal).sum(dim=2)
        scale = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(numerator),
        ).clamp_min(0)
        trit_values.append(trits)
        stored_scale = (
            scale.to(torch.float16).to(torch.float64)
            if group_size == master.shape[1]
            else scale
        )
        scale_values.append(stored_scale)
        residual = residual - trits_f64 * stored_scale.unsqueeze(-1)

    # Greedy residual fitting is deterministic and cheap, but coordinate
    # refinement closes much of its additive-plane error without changing the
    # export contract. Keep all updates in FP64; round to stored FP16 only at
    # the final receipt boundary.
    if group_size < master.shape[1]:
        for _ in range(20):
            for index in range(planes):
                decoded_without = torch.zeros_like(grouped_master)
                for other, (other_trits, other_scale) in enumerate(
                    zip(trit_values, scale_values)
                ):
                    if other != index:
                        decoded_without = (
                            decoded_without
                            + other_trits.to(torch.float64) * other_scale.unsqueeze(-1)
                        )
                residual_without = grouped_master - decoded_without
                current_scale = scale_values[index].clamp_min(
                    torch.finfo(torch.float64).tiny
                )
                trits = (
                    (residual_without / current_scale.unsqueeze(-1))
                    .round()
                    .clamp(-1, 1)
                    .to(torch.int8)
                )
                trits_f64 = trits.to(torch.float64)
                denominator = (trits_f64.square() * grouped_diagonal).sum(dim=2)
                numerator = (residual_without * trits_f64 * grouped_diagonal).sum(
                    dim=2
                )
                scale = torch.where(
                    denominator > 0,
                    numerator
                    / denominator.clamp_min(torch.finfo(torch.float64).tiny),
                    torch.zeros_like(numerator),
                ).clamp_min(0)
                trit_values[index] = trits
                scale_values[index] = scale

    fitted_planes = [
        TernaryPlane(
            trits=trits.reshape_as(master),
            scales=scale.to(torch.float16),
            group_size=group_size,
        )
        for trits, scale in zip(trit_values, scale_values)
    ]
    decoded = torch.zeros_like(master_f64)
    for plane in fitted_planes:
        stored_scale_f64 = expand_plane_scales(
            plane.scales,
            rows=master.shape[0],
            columns=master.shape[1],
            group_size=group_size,
        ).to(torch.float64)
        decoded = decoded + plane.trits.to(torch.float64) * stored_scale_f64
    dense = torch.zeros_like(master, device="cpu")
    for plane in fitted_planes:
        dense = dense + plane.trits.to(master.dtype) * expand_plane_scales(
            plane.scales,
            rows=master.shape[0],
            columns=master.shape[1],
            group_size=plane.group_size,
        ).to(master.dtype)
    projection = TernaryProjection(
        dense=dense,
        planes=tuple(fitted_planes),
        algorithm_id=_diagonal_algorithm_id(planes),
        schema_version=1,
    )
    validate_projection(
        projection,
        master.detach().to(device="cpu"),
        algorithm_id=projection.algorithm_id,
        schema_version=1,
    )
    return projection


def _diagonal_algorithm_id(planes: int) -> str:
    return f"tritium.diagonal-additive-{planes}@1"


def _adaptive_diagonal_algorithm_id() -> str:
    """Identity for measured weight-level rate-distortion allocation."""

    return "tritium.diagonal-additive-adaptive@1"


def _fit_module(
    prepared: PreparedModel,
    calibration: ActivationCalibrationReceipt,
    work_dir: Pathish,
    max_working_bytes: int,
) -> ModuleQuantizationResult:
    """Fit hard additive planes from strict live-module calibration evidence."""

    if not isinstance(prepared, PreparedModel) or not isinstance(
        prepared.model, nn.Module
    ):
        raise TypeError("module conversion requires a live-module PreparedModel")
    if prepared.config.mode != "ptq":
        raise TritiumError(
            "module conversion requires PTQ preparation",
            code="invalid_phase",
            stage="convert",
        )
    if not isinstance(calibration, ActivationCalibrationReceipt):
        raise TypeError("module conversion requires an ActivationCalibrationReceipt")
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    if prepared.coverage is None:
        raise TritiumError(
            "module conversion requires exact prepared coverage",
            code="coverage_missing",
            stage="convert",
        )
    reopened = load_activation_calibration(
        calibration.evidence_dir,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    if reopened != calibration:
        raise TritiumError(
            "activation calibration changed after admission",
            code="evidence_changed",
            stage="convert",
        )
    source_digest = _source_model_digest(prepared.model)
    if source_digest != calibration.source_model_digest:
        raise TritiumError(
            "prepared model changed after calibration",
            code="source_changed",
            stage="convert",
        )
    parameters = dict(prepared.model.named_parameters(remove_duplicate=False))
    expected_modules = _selected_linear_modules(prepared)
    records = tuple(calibration.records)
    observed_by_aliases = {
        tuple(record.weight_aliases): record for record in records
    }
    expected_by_aliases = {
        tuple(record.aliases): record for record in expected_modules
    }
    if (
        len(observed_by_aliases) != len(records)
        or set(observed_by_aliases) != set(expected_by_aliases)
    ):
        raise TritiumError(
            "activation calibration aliases differ from prepared coverage",
            code="coverage_mismatch",
            stage="convert",
        )
    for aliases, expected in expected_by_aliases.items():
        record = observed_by_aliases[aliases]
        if record.module != expected.path:
            raise TritiumError(
                "activation calibration module differs from prepared coverage",
                code="coverage_mismatch",
                stage="convert",
                module=record.module,
            )
        if (
            record.features != expected.module.in_features
            or record.outputs != expected.module.out_features
        ):
            raise TritiumError(
                "activation calibration geometry differs from prepared coverage",
                code="coverage_mismatch",
                stage="convert",
                module=record.module,
            )
        canonical = parameters.get(aliases[0])
        if canonical is None or any(
            parameters.get(alias) is not canonical for alias in aliases
        ):
            raise TritiumError(
                "activation calibration aliases are not one shared parameter",
                code="coverage_mismatch",
                stage="convert",
                module=record.module,
            )
    adaptive = prepared.config.target_bpw is not None
    algorithm_id = (
        _adaptive_diagonal_algorithm_id()
        if adaptive
        else _diagonal_algorithm_id(prepared.config.planes)
    )
    recipe_id = module_recipe_id(
        source_digest,
        calibration.evidence_id,
        algorithm_id,
        prepared.config,
        prepared.coverage,
    )

    def chunk_rows(record: ActivationRecord) -> int:
        fixed_bytes = record.features * _FIT_FIXED_BYTES_PER_FEATURE
        per_row_bytes = record.features * _FIT_BYTES_PER_COEFFICIENT
        if max_working_bytes < fixed_bytes + per_row_bytes:
            raise TritiumError(
                "max_working_bytes cannot hold one fitted output row",
                code="working_set_too_small",
                stage="convert",
                module=record.module,
                details={
                    "required_bytes": fixed_bytes + per_row_bytes,
                    "max_working_bytes": max_working_bytes,
                },
            )
        return min(record.outputs, (max_working_bytes - fixed_bytes) // per_row_bytes)

    def measured_error_curve(record: ActivationRecord) -> tuple[float, ...]:
        """Measure weighted reconstruction error for every admissible plane count."""

        try:
            master = parameters[record.weight_aliases[0]]
        except KeyError as error:
            raise TritiumError(
                "calibration refers to a missing source parameter",
                code="evidence_geometry_mismatch",
                stage="convert",
                module=record.module,
            ) from error
        payload = (calibration.evidence_dir / record.file).read_bytes()
        curvature = torch.frombuffer(bytearray(payload), dtype=torch.float64)
        curvature = curvature / record.samples
        if curvature.numel() != master.shape[1]:
            raise TritiumError(
                "calibration curvature does not match the selected weight",
                code="evidence_geometry_mismatch",
                stage="convert",
                module=record.module,
            )
        objective_curvature = curvature
        if float(objective_curvature.sum()) <= 0.0:
            objective_curvature = torch.ones_like(objective_curvature)
        curves = [0.0] * (prepared.config.planes + 1)
        rows_per_chunk = chunk_rows(record)
        for start in range(0, master.shape[0], rows_per_chunk):
            stop = min(master.shape[0], start + rows_per_chunk)
            master_chunk = master[start:stop]
            dense = master_chunk.detach().cpu().to(torch.float64)
            grouped_curvature = objective_curvature.to(torch.float64).reshape(
                1, -1
            )
            curves[0] += float(
                (dense.square() * grouped_curvature).sum()
            )
            for planes in range(1, prepared.config.planes + 1):
                projection = _diagonal_additive_projection(
                    master_chunk, objective_curvature, planes
                )
                error = (
                    dense - projection.dense.to(torch.float64)
                ).square()
                curves[planes] += float((error * grouped_curvature).sum())
        return tuple(curves)

    plane_counts = None
    if adaptive:
        measured_curves = [measured_error_curve(record) for record in records]
        allocation = allocate_planes(
            [record.outputs * record.features for record in records],
            [1.0 for _ in records],
            measured_curves,
            prepared.config.target_bpw,
            t_min=1,
            t_max=prepared.config.planes,
        )
        plane_counts = {
            record.weight_aliases[0]: count
            for record, count in zip(records, allocation.plane_counts)
        }

    def fit_weight(
        record: ActivationRecord, writer: WeightCheckpointWriter
    ) -> float:
        try:
            master = parameters[record.weight_aliases[0]]
        except KeyError as error:
            raise TritiumError(
                "calibration refers to a missing source parameter",
                code="evidence_geometry_mismatch",
                stage="convert",
                module=record.module,
            ) from error
        payload = (calibration.evidence_dir / record.file).read_bytes()
        curvature = torch.frombuffer(bytearray(payload), dtype=torch.float64)
        curvature = curvature / record.samples
        weighted_error = 0.0
        rows_per_chunk = chunk_rows(record)
        for start in range(0, master.shape[0], rows_per_chunk):
            stop = min(master.shape[0], start + rows_per_chunk)
            master_chunk = master[start:stop]
            projection = _diagonal_additive_projection(
                master_chunk, curvature, writer.plane_count
            )
            error = (
                master_chunk.detach().cpu().to(torch.float64)
                - projection.dense.to(torch.float64)
            ).square()
            weighted_error += float((error * curvature).sum())
            writer.append(projection.planes)
        denominator = curvature.sum().clamp_min(1e-30) * master.shape[0]
        return weighted_error / float(denominator)

    return seal_module_conversion(
        work_dir,
        source_model_digest=source_digest,
        evidence_id=calibration.evidence_id,
        algorithm_id=algorithm_id,
        recipe_id=recipe_id,
        config=prepared.config,
        coverage=prepared.coverage,
        records=records,
        fit_weight=fit_weight,
        fit_chunk_rows=chunk_rows,
        max_working_bytes=max_working_bytes,
        plane_counts=plane_counts,
    )


def load_quantized_module(
    model: nn.Module,
    artifact: Union[ModuleQuantizationResult, Pathish],
    *,
    inplace: bool = False,
) -> nn.Module:
    """Bind one strict generic PTQ artifact to its source module graph."""

    if not isinstance(model, nn.Module):
        raise TypeError("load_quantized_module requires a torch.nn.Module")
    if not isinstance(inplace, bool):
        raise TypeError("load_quantized_module inplace must be a bool")
    if isinstance(artifact, ModuleQuantizationResult):
        admitted = load_module_conversion(artifact.artifact_dir)
        if admitted != artifact:
            raise TritiumError(
                "module conversion fields differ from strict artifact reload",
                code="artifact_changed",
                stage="load",
            )
    else:
        admitted = load_module_conversion(artifact)
    source_digest = _source_model_digest(model)
    if source_digest != admitted.source_model_digest:
        raise TritiumError(
            "source model differs from module conversion artifact",
            code="source_changed",
            stage="load",
            details={
                "expected": admitted.source_model_digest,
                "observed": source_digest,
            },
        )
    try:
        target = model if inplace else copy.deepcopy(model)
    except Exception as error:
        raise TritiumError(
            "source model could not be copied for module conversion",
            code="copy_failed",
            stage="load",
        ) from error

    from ..nn import (
        AdditiveTernaryEmbedding,
        AdditiveTernaryLinear,
        AdditiveTernaryWeight,
    )

    modules = dict(target.named_modules(remove_duplicate=False))
    owners = {}
    replacement_paths = {}
    group_sizes = {}
    for reference in admitted.weights:
        fitted = admitted.weight(reference.path)
        packed_weight = None
        replacement_by_module = {}
        for alias in reference.aliases:
            if alias == "weight":
                module_path = ""
            elif alias.endswith(".weight"):
                module_path = alias[: -len(".weight")]
            else:
                raise TritiumError(
                    "module conversion weight alias is not canonical",
                    code="coverage_mismatch",
                    stage="load",
                    module=alias,
                )
            module = modules.get(module_path)
            if type(module) not in {nn.Linear, nn.Embedding}:
                raise TritiumError(
                    "module conversion target is not an exact Linear or Embedding module",
                    code="coverage_mismatch",
                    stage="load",
                    module=module_path,
                )
            if tuple(module.weight.shape) != reference.shape:
                raise TritiumError(
                    "module conversion target geometry differs from artifact",
                    code="coverage_mismatch",
                    stage="load",
                    module=module_path,
                )
            prior = owners.get(id(module))
            if prior is not None and prior != reference.path:
                raise TritiumError(
                    "one module is bound to multiple conversion weights",
                    code="coverage_mismatch",
                    stage="load",
                    module=module_path,
                )
            owners[id(module)] = reference.path
            replacement = replacement_by_module.get(id(module))
            if replacement is None:
                if packed_weight is None:
                    packed_weight = AdditiveTernaryWeight(fitted.planes).to(
                        device=module.weight.device
                    )
                    for alias in reference.aliases:
                        group_sizes[alias] = packed_weight.group_size
                    owns_weight = True
                else:
                    owns_weight = False
                if type(module) is nn.Linear:
                    replacement = AdditiveTernaryLinear.from_packed_weight(
                        packed_weight,
                        module.bias,
                        owner=owns_weight,
                    )
                else:
                    if module.max_norm is not None:
                        raise TritiumError(
                            "module conversion cannot preserve mutating Embedding max_norm",
                            code="unsupported_module_option",
                            stage="load",
                            module=module_path,
                        )
                    replacement = AdditiveTernaryEmbedding(
                        packed_weight,
                        padding_idx=module.padding_idx,
                        max_norm=module.max_norm,
                        norm_type=module.norm_type,
                        scale_grad_by_freq=module.scale_grad_by_freq,
                        sparse=module.sparse,
                        dtype=module.weight.dtype,
                        owner=owns_weight,
                    )
                replacement_by_module[id(module)] = replacement
            replacement_paths[module_path] = replacement

    root = target
    for module_path, replacement in replacement_paths.items():
        if module_path == "":
            root = replacement
            continue
        parts = module_path.split(".")
        parent = root
        for part in parts[:-1]:
            parent = parent._modules[part]
        parent._modules[parts[-1]] = replacement
    root.requires_grad_(False)
    root.eval()
    if hasattr(root, "all_tied_weights_keys"):
        root.all_tied_weights_keys = {}
    if hasattr(root, "_tied_weights_keys"):
        root._tied_weights_keys = {}
    root._tritium_ptq_artifact_id = admitted.artifact_id
    root._tritium_coverage = admitted.coverage
    if hasattr(root, "config"):
        from .hf import attach_huggingface_recipe

        attach_huggingface_recipe(root, admitted.config)
        root.config.tritium_ptq_artifact_id = admitted.artifact_id
        root.config.tritium_ptq_source_digest = admitted.source_model_digest
        root.config.tritium_ptq_checkpoint_digest = _source_model_digest(root)
        root.config.tritium_ptq_group_sizes = dict(group_sizes)
    return root


def calibrate(
    prepared: PreparedModel,
    data: Any = None,
    *,
    evidence_dir: Pathish,
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> Union[CalibrationReceipt, ActivationCalibrationReceipt]:
    """Admit precomputed Qwen3.6 S2KF evidence for PTQ conversion.

    Local Qwen sources admit the canonical 506-record S2KF namespace. Live
    PyTorch modules instead stream bounded diagonal second moments from
    ``data`` into a separately typed evidence namespace.
    """

    if not isinstance(prepared, PreparedModel):
        raise TypeError("calibrate requires a PreparedModel")
    if prepared.config.mode != "ptq":
        raise TritiumError(
            "QAT preparation does not use the PTQ calibration phase",
            code="invalid_phase",
            stage="calibrate",
        )
    if isinstance(prepared.model, nn.Module):
        if data is None:
            raise TritiumError(
                "live-module calibration requires an iterable of batches",
                code="calibration_data_required",
                stage="calibrate",
            )
        return _collect_activations(
            prepared, data, Path(evidence_dir), max_evidence_bytes
        )
    if data is not None:
        raise TritiumError(
            "raw calibration collection is not available; provide evidence_dir only",
            code="evidence_required",
            stage="calibrate",
        )
    requested = Path(evidence_dir)
    if requested.is_symlink():
        raise TritiumError(
            "PTQ evidence directory must not be a symlink",
            code="invalid_evidence_path",
            stage="calibrate",
        )
    directory = requested.resolve(strict=True)
    values = _tritium.inspect_qwen36_ptq_evidence(
        str(directory), max_evidence_bytes=max_evidence_bytes
    )
    return CalibrationReceipt(
        evidence_dir=directory,
        evidence_id=values[0],
        curvature=values[1],
        record_count=values[2],
        source_model_digest=values[3],
        activation_cache_digest=values[4],
        token_stream_digest=values[5],
        max_evidence_bytes=max_evidence_bytes,
    )


def convert(
    prepared: PreparedModel,
    calibration: Optional[
        Union[CalibrationReceipt, ActivationCalibrationReceipt]
    ] = None,
    *,
    revision: Optional[str] = None,
    work_dir: Optional[Pathish] = None,
    output_dir: Optional[Pathish] = None,
    compact_max_bytes: Optional[int] = None,
    compact_max_resident_bytes: Optional[int] = None,
    near_lossless_max_bytes: Optional[int] = None,
    near_lossless_max_resident_bytes: Optional[int] = None,
    max_working_bytes: int = 64 * 1024 * 1024,
    packing: str = "b3",
) -> Union[QuantizationResult, ModuleQuantizationResult, "QatHardResult"]:
    """Freeze QAT or run resumable PTQ with explicit result discriminants."""

    if not isinstance(prepared, PreparedModel):
        raise TypeError("convert requires a PreparedModel")
    if prepared.config.mode == "qat":
        if calibration is not None:
            raise TypeError("QAT hard conversion does not accept calibration")
        supplied = {
            "revision": revision,
            "work_dir": work_dir,
            "output_dir": output_dir,
            "compact_max_bytes": compact_max_bytes,
            "compact_max_resident_bytes": compact_max_resident_bytes,
            "near_lossless_max_bytes": near_lossless_max_bytes,
            "near_lossless_max_resident_bytes": near_lossless_max_resident_bytes,
            "max_working_bytes": (
                max_working_bytes if max_working_bytes != 64 * 1024 * 1024 else None
            ),
            "packing": packing if packing != "b3" else None,
        }
        names = sorted(name for name, value in supplied.items() if value is not None)
        if names:
            raise TypeError(
                "QAT hard conversion does not accept PTQ arguments: "
                + ", ".join(names)
            )
        from .qat import convert_qat_hard

        return convert_qat_hard(prepared)
    if isinstance(prepared.model, nn.Module):
        if calibration is None:
            raise TypeError("live-module PTQ convert requires calibration")
        if work_dir is None:
            raise TypeError("live-module convert requires work_dir")
        qwen_arguments = {
            "revision": revision,
            "output_dir": output_dir,
            "compact_max_bytes": compact_max_bytes,
            "compact_max_resident_bytes": compact_max_resident_bytes,
            "near_lossless_max_bytes": near_lossless_max_bytes,
            "near_lossless_max_resident_bytes": near_lossless_max_resident_bytes,
        }
        supplied = sorted(
            name for name, value in qwen_arguments.items() if value is not None
        )
        if supplied:
            raise TypeError(
                "live-module convert does not accept Qwen package arguments: "
                + ", ".join(supplied)
            )
        return _fit_module(
            prepared,
            calibration,
            work_dir,
            max_working_bytes,
        )
    if not isinstance(calibration, CalibrationReceipt):
        raise TypeError("Qwen convert requires a CalibrationReceipt")
    if prepared.config.mode != "ptq" or not isinstance(prepared.model, Path):
        raise TritiumError(
            "Qwen3.6 PTQ conversion requires a prepared local source directory",
            code="invalid_phase",
            stage="convert",
        )
    if prepared.config.target_bpw is not None:
        raise TritiumError(
            "Qwen3.6 conversion currently requires exact byte ceilings; target_bpw is not silently approximated",
            code="unsupported_recipe",
            stage="convert",
            details={"target_bpw": prepared.config.target_bpw},
        )
    required = {
        "revision": revision,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "compact_max_bytes": compact_max_bytes,
        "compact_max_resident_bytes": compact_max_resident_bytes,
        "near_lossless_max_bytes": near_lossless_max_bytes,
        "near_lossless_max_resident_bytes": near_lossless_max_resident_bytes,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise TypeError(
            "Qwen convert missing required arguments: " + ", ".join(missing)
        )
    current = calibrate(
        prepared,
        evidence_dir=calibration.evidence_dir,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    if current != calibration:
        raise TritiumError(
            "calibration evidence changed after admission",
            code="evidence_changed",
            stage="convert",
        )
    native = reconcile_qwen36_ptq_packages(
        prepared.model,
        revision=revision,
        work_dir=work_dir,
        evidence_dir=calibration.evidence_dir,
        output_dir=output_dir,
        compact_max_bytes=compact_max_bytes,
        compact_max_resident_bytes=compact_max_resident_bytes,
        near_lossless_max_bytes=near_lossless_max_bytes,
        near_lossless_max_resident_bytes=near_lossless_max_resident_bytes,
        packing=packing,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    return load(native.artifact_dir)


def quantize(
    model_or_id: Pathish,
    config: TernaryConfig,
    *,
    revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    output_dir: Pathish,
    compact_max_bytes: int,
    compact_max_resident_bytes: int,
    near_lossless_max_bytes: int,
    near_lossless_max_resident_bytes: int,
    packing: str = "b3",
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> QuantizationResult:
    """Compose exact PTQ ``prepare`` → ``calibrate`` → ``convert`` phases."""

    prepared = prepare(model_or_id, config, inplace=False)
    calibration = calibrate(
        prepared,
        evidence_dir=evidence_dir,
        max_evidence_bytes=max_evidence_bytes,
    )
    return convert(
        prepared,
        calibration,
        revision=revision,
        work_dir=work_dir,
        output_dir=output_dir,
        compact_max_bytes=compact_max_bytes,
        compact_max_resident_bytes=compact_max_resident_bytes,
        near_lossless_max_bytes=near_lossless_max_bytes,
        near_lossless_max_resident_bytes=near_lossless_max_resident_bytes,
        packing=packing,
    )


__all__ = [
    "ActivationCalibrationReceipt",
    "ActivationRecord",
    "CalibrationReceipt",
    "FittedWeight",
    "ModuleQuantizationResult",
    "calibrate",
    "convert",
    "load_activation_calibration",
    "load_module_conversion",
    "load_quantized_module",
    "quantize",
]
