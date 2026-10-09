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
# and the raw op primitives work without torch. Once PyTorch is installed, do not
# suppress ImportError from Tritium's facade: that would make a broken public
# API look like a successful core-only import.
import importlib.util as _importlib_util

try:
    from . import autograd  # noqa: F401
    from . import torch  # noqa: F401
    # ``tritium.nn`` imports estimator/ops modules from ``tritium.torch``;
    # initialize that package before exposing the higher-level facade.
    from . import nn  # noqa: F401

    __all__.extend(["autograd", "nn", "torch"])
except ModuleNotFoundError as error:
    # A missing optional dependency is allowed. If PyTorch is installed,
    # preserve missing-module failures from its own or Tritium's imports.
    if error.name != "torch" or _importlib_util.find_spec("torch") is not None:
        raise
