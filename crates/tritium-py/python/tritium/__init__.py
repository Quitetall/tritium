"""Tritium: ternary-model inference + autograd ops from Python.

The compiled extension is ``tritium._tritium``; this package re-exports its surface and — when
PyTorch is installed — the :mod:`tritium.autograd` wrappers that expose the ternary Conv1d / FSQ /
STE ops as ``torch.autograd.Function`` layers (ADR 0030).
"""

from ._tritium import (
    Model,
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

__all__ = [
    "Model",
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
]

# The torch wrappers are optional: importing them requires PyTorch. Inference (Model/ternary_matmul)
# and the raw op primitives work without torch.
try:  # pragma: no cover - trivial import guard
    from . import autograd  # noqa: F401

    __all__.append("autograd")
except ImportError:
    pass
