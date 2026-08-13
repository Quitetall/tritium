"""Tritium: ternary-model inference, conversion and differentiable ops from Python.

The compiled extension is ``tritium._tritium``; this package re-exports its surface and — when
PyTorch is installed — the :mod:`tritium.torch` research facade, :mod:`tritium.nn` modules and
:mod:`tritium.autograd` compatibility wrappers (ADR 0030 / ADR 0033).
"""

from ._tritium import (
    KroneckerConflictError,
    KroneckerContractError,
    KroneckerEvidenceBuilder,
    KroneckerEvidenceReceipt,
    KroneckerPublicationError,
    KroneckerResourceError,
    KroneckerSharedForwardGroup,
    KroneckerStateError,
    Model,
    QwenLoadReceipt,
    QwenModel,
    Qwen36KroneckerCaptureReceipt,
    Qwen36KroneckerCaptureSession,
    Qwen36KroneckerCaptureTask,
    QwenReferenceLanguageOutput,
    Stage7EvidenceContractError,
    Stage7EvidenceIoError,
    Stage7EvidenceStateError,
    Stage7TokenBatch,
    Stage7TokenEvidencePack,
    Stage7TokenEvidenceReceipt,
    compiled_backends,
    conv1d_forward,
    conv1d_vjp,
    fsq_forward,
    fsq_vjp,
    lsq_forward,
    lsq_vjp,
    ste_absmean_scale,
    ste_quantize_forward,
    ste_quantize_vjp,
    ternary_matmul,
)
from . import onnx, portable, salt

__all__ = [
    "KroneckerConflictError",
    "KroneckerContractError",
    "KroneckerEvidenceBuilder",
    "KroneckerEvidenceReceipt",
    "KroneckerPublicationError",
    "KroneckerResourceError",
    "KroneckerSharedForwardGroup",
    "KroneckerStateError",
    "Model",
    "QwenLoadReceipt",
    "QwenModel",
    "Qwen36KroneckerCaptureReceipt",
    "Qwen36KroneckerCaptureSession",
    "Qwen36KroneckerCaptureTask",
    "QwenReferenceLanguageOutput",
    "Stage7EvidenceContractError",
    "Stage7EvidenceIoError",
    "Stage7EvidenceStateError",
    "Stage7TokenBatch",
    "Stage7TokenEvidencePack",
    "Stage7TokenEvidenceReceipt",
    "compiled_backends",
    "ternary_matmul",
    "conv1d_forward",
    "conv1d_vjp",
    "fsq_forward",
    "fsq_vjp",
    "lsq_forward",
    "lsq_vjp",
    "ste_absmean_scale",
    "ste_quantize_forward",
    "ste_quantize_vjp",
    "salt",
    "onnx",
    "portable",
]

# The torch wrappers are optional: importing them requires PyTorch. Inference (Model/ternary_matmul)
# and the raw op primitives work without torch.
try:  # pragma: no cover - trivial import guard
    from . import autograd  # noqa: F401
    from . import torch  # noqa: F401
    # ``tritium.nn`` imports estimator/ops modules from ``tritium.torch``.
    # Load the facade first so the optional import guard cannot hide both
    # packages behind a circular-import failure.
    from . import nn  # noqa: F401

    __all__.extend(["autograd", "nn", "torch"])
except ImportError:
    pass
