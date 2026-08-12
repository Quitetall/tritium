"""Strict Qwen3.6 language-plus-MTP calibration boundary.

Transformers can expose the language graph while silently dropping the bundled
MTP drafter from a checkpoint.  That is unsafe for plan-0043 evidence: a
language-only calibration must never be presented as language-plus-MTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from torch import nn

from .ptq import ActivationCalibrationReceipt, capture_qwen36_kronecker_evidence

Pathish = Union[str, Path]


class Qwen36ComponentError(ValueError):
    """The supplied graph does not satisfy the Qwen3.6 component contract."""


@dataclass(frozen=True)
class Qwen36Components:
    """Resolved language, MTP, and output-head modules from one graph."""

    root: nn.Module
    language_model: nn.Module
    mtp_model: Optional[nn.Module]
    lm_head: nn.Module
    language_path: str
    mtp_path: Optional[str]
    lm_head_path: str


def _module(root: nn.Module, path: str, label: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise Qwen36ComponentError(f"Qwen3.6 {label} module is missing: {path}")
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise Qwen36ComponentError(f"Qwen3.6 {label} is not a torch module: {path}")
    return current


def resolve_qwen36_components(
    model: nn.Module,
    *,
    require_mtp: bool = True,
) -> Qwen36Components:
    """Resolve canonical Qwen3.6 module paths without alias guessing.

    The supported Transformers layout is ``model.language_model`` under the
    top-level ``model`` field, with ``lm_head`` at the root.  The MTP drafter
    must be exposed as ``mtp`` by a loader that actually retained checkpoint
    tensors.  Missing MTP is an error by default, never an implicit downgrade.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    language = _module(model, "model.language_model", "language")
    lm_head = _module(model, "lm_head", "language output head")
    mtp = (
        _module(model, "mtp", "MTP drafter")
        if require_mtp or hasattr(model, "mtp")
        else None
    )
    if id(language) == id(lm_head) or (
        mtp is not None and id(language) == id(mtp)
    ) or (mtp is not None and id(lm_head) == id(mtp)):
        raise Qwen36ComponentError("Qwen3.6 component paths alias unexpectedly")
    return Qwen36Components(
        root=model,
        language_model=language,
        mtp_model=mtp,
        lm_head=lm_head,
        language_path="model.language_model",
        mtp_path="mtp" if mtp is not None else None,
        lm_head_path="lm_head",
    )


def capture_qwen36_components(
    model: nn.Module,
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
    guided_loss_reduction: Optional[str] = None,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    max_batch_bytes: int = 256 * 1024 * 1024,
    max_capture_bytes: Optional[int] = None,
    max_objective_bytes: int = 256 * 1024 * 1024,
    max_shared_modules: int = 8,
) -> ActivationCalibrationReceipt:
    """Capture strict language-plus-MTP Qwen3.6 evidence.

    Component resolution happens before native session creation or evidence
    mutation.  The existing model-aware capture loop remains the sole producer
    of canonical records and retains its resumability/provenance guarantees.
    """

    components = resolve_qwen36_components(model, require_mtp=True)
    if components.mtp_model is None:
        raise Qwen36ComponentError("Qwen3.6 MTP drafter is required for capture")
    return capture_qwen36_kronecker_evidence(
        components.language_model,
        data_factory,
        model_dir=model_dir,
        declared_revision=declared_revision,
        work_dir=work_dir,
        evidence_dir=evidence_dir,
        curvature=curvature,
        activation_cache_digest=activation_cache_digest,
        token_stream_digest=token_stream_digest,
        damping=damping,
        mtp_model=components.mtp_model,
        guided_loss_reduction=guided_loss_reduction,
        max_evidence_bytes=max_evidence_bytes,
        max_batch_bytes=max_batch_bytes,
        max_capture_bytes=max_capture_bytes,
        max_objective_bytes=max_objective_bytes,
        max_shared_modules=max_shared_modules,
    )


__all__ = [
    "Qwen36ComponentError",
    "Qwen36Components",
    "capture_qwen36_components",
    "resolve_qwen36_components",
]
