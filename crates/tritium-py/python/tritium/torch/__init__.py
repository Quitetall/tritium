"""Differentiable PyTorch frontend for Tritium ternary models."""

from .config import RefinementConfig, TernaryConfig
from .conversion import PreparedModel, inspect, prepare, prepare_qat
from .coverage import CoverageEntry, CoverageReport
from .errors import TritiumError
from .estimators import (
    AbsMeanSTE,
    AdditiveEstimator,
    AnnealedSTE,
    Estimator,
    LSQEstimator,
    SaltSTE,
    SparseTernaryEstimator,
    TTQEstimator,
    TWNEstimator,
    create_estimator,
    register_estimator,
    registered_estimators,
)
try:
    from .hf import HfTritiumConfig, TritiumTrainer, register_huggingface
except ImportError:  # transformers is an optional integration dependency
    HfTritiumConfig = None
    TritiumTrainer = None
    register_huggingface = None
else:
    register_huggingface()
from .ops import reference_ternary_linear, ternary_linear
from .projection import (
    ProjectionContext,
    TernaryPlane,
    TernaryProjection,
    validate_projection,
)

__all__ = [
    "AbsMeanSTE",
    "AdditiveEstimator",
    "AnnealedSTE",
    "CoverageEntry",
    "CoverageReport",
    "Estimator",
    "HfTritiumConfig",
    "LSQEstimator",
    "ProjectionContext",
    "PreparedModel",
    "RefinementConfig",
    "SaltSTE",
    "SparseTernaryEstimator",
    "TernaryConfig",
    "TernaryPlane",
    "TernaryProjection",
    "TritiumError",
    "TritiumTrainer",
    "TTQEstimator",
    "TWNEstimator",
    "create_estimator",
    "inspect",
    "prepare",
    "prepare_qat",
    "reference_ternary_linear",
    "register_estimator",
    "register_huggingface",
    "registered_estimators",
    "ternary_linear",
    "validate_projection",
]
