"""Differentiable PyTorch frontend for Tritium ternary models."""

from .artifacts import (
    ArtifactRef,
    ExportReceipt,
    HfAssetRef,
    QuantizationResult,
    export,
    load,
)
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
from .onnx import (
    OnnxBundleManifest,
    OnnxCausalLMOutput,
    OnnxMtpOutput,
    QwenOnnxCausalLM,
    export_onnx,
    load_onnx,
)
from .ptq import (
    ActivationCalibrationReceipt,
    ActivationRecord,
    CalibrationReceipt,
    FittedWeight,
    ModuleQuantizationResult,
    calibrate,
    convert,
    fit,
    load_activation_calibration,
    quantize,
)
from .projection import (
    ProjectionContext,
    TernaryPlane,
    TernaryProjection,
    validate_projection,
)

__all__ = [
    "AbsMeanSTE",
    "ActivationCalibrationReceipt",
    "ActivationRecord",
    "AdditiveEstimator",
    "AnnealedSTE",
    "ArtifactRef",
    "CalibrationReceipt",
    "CoverageEntry",
    "CoverageReport",
    "Estimator",
    "ExportReceipt",
    "FittedWeight",
    "HfTritiumConfig",
    "HfAssetRef",
    "LSQEstimator",
    "ModuleQuantizationResult",
    "OnnxBundleManifest",
    "OnnxCausalLMOutput",
    "OnnxMtpOutput",
    "ProjectionContext",
    "PreparedModel",
    "QuantizationResult",
    "QwenOnnxCausalLM",
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
    "calibrate",
    "convert",
    "export",
    "export_onnx",
    "fit",
    "inspect",
    "load",
    "load_activation_calibration",
    "load_onnx",
    "prepare",
    "prepare_qat",
    "quantize",
    "reference_ternary_linear",
    "register_estimator",
    "register_huggingface",
    "registered_estimators",
    "ternary_linear",
    "validate_projection",
]
