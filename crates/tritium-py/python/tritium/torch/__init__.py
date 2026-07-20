"""Differentiable PyTorch frontend for Tritium ternary models."""

from .config import TernaryConfig
from .conversion import inspect, prepare_qat
from .coverage import CoverageEntry, CoverageReport
from .errors import TritiumError
from .estimators import AbsMeanSTE, Estimator, SaltSTE
try:
    from .hf import HfTritiumConfig, register_huggingface
except ImportError:  # transformers is an optional integration dependency
    HfTritiumConfig = None
    register_huggingface = None
else:
    register_huggingface()
from .ops import reference_ternary_linear, ternary_linear
from .projection import ProjectionContext, TernaryProjection, validate_projection

__all__ = [
    "AbsMeanSTE",
    "CoverageEntry",
    "CoverageReport",
    "Estimator",
    "HfTritiumConfig",
    "ProjectionContext",
    "SaltSTE",
    "TernaryConfig",
    "TernaryProjection",
    "TritiumError",
    "inspect",
    "prepare_qat",
    "reference_ternary_linear",
    "register_huggingface",
    "ternary_linear",
    "validate_projection",
]
