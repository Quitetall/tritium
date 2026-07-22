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
from .observability import (
    OpenTelemetryDiagnostics,
    PlaneDiagnostics,
    ScaleStatistics,
    TensorDiagnostics,
    TernaryDiagnostics,
    TritHistogram,
    WandbDiagnostics,
    collect_diagnostics,
    log_opentelemetry,
    log_tensorboard,
    log_wandb,
)
from .module_onnx import (
    ModuleOnnxArtifact,
    ModuleOnnxLineage,
    OnnxModule,
    export_module_onnx,
    load_module_onnx,
)
from .onnx import (
    OnnxBundleManifest,
    OnnxCausalLMOutput,
    OnnxMtpOutput,
    QwenOnnxCausalLM,
    export_onnx,
    load_onnx,
)
from .qat import QatHardResult, QatHardWeight, convert_qat_hard
from .qat_artifacts import QatHardArtifact, export_qat_hard, load_qat_hard
from .ptq import (
    ActivationCalibrationReceipt,
    ActivationRecord,
    CalibrationReceipt,
    FittedWeight,
    KroneckerCalibrationWriter,
    KroneckerModuleCaptureReceipt,
    ModuleQuantizationResult,
    calibrate,
    bind_kronecker_activation_cache_digest,
    capture_kronecker_embedding,
    capture_kronecker_module,
    convert,
    load_activation_calibration,
    load_module_conversion,
    load_quantized_module,
    quantize,
)
from .tutorial import (
    SMOLLM2_MODEL_ID,
    SMOLLM2_REVISION,
    run_smollm2_release_demo,
)
from .projection import (
    ProjectionContext,
    TernaryPlane,
    TernaryProjection,
    validate_projection,
)
from .refinement import (
    RefinementResult,
    export_refinement,
    load_refinement,
    refine,
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
    "KroneckerCalibrationWriter",
    "KroneckerModuleCaptureReceipt",
    "ModuleQuantizationResult",
    "ModuleOnnxArtifact",
    "ModuleOnnxLineage",
    "OnnxBundleManifest",
    "OnnxCausalLMOutput",
    "OnnxMtpOutput",
    "OnnxModule",
    "OpenTelemetryDiagnostics",
    "PlaneDiagnostics",
    "ProjectionContext",
    "PreparedModel",
    "QatHardArtifact",
    "QatHardResult",
    "QatHardWeight",
    "QuantizationResult",
    "QwenOnnxCausalLM",
    "RefinementConfig",
    "RefinementResult",
    "SaltSTE",
    "ScaleStatistics",
    "SparseTernaryEstimator",
    "SMOLLM2_MODEL_ID",
    "SMOLLM2_REVISION",
    "TernaryConfig",
    "TernaryDiagnostics",
    "TernaryPlane",
    "TernaryProjection",
    "TritiumError",
    "TritiumTrainer",
    "TritHistogram",
    "TTQEstimator",
    "TWNEstimator",
    "WandbDiagnostics",
    "create_estimator",
    "calibrate",
    "bind_kronecker_activation_cache_digest",
    "capture_kronecker_embedding",
    "capture_kronecker_module",
    "convert",
    "convert_qat_hard",
    "collect_diagnostics",
    "export",
    "export_qat_hard",
    "export_refinement",
    "export_onnx",
    "export_module_onnx",
    "inspect",
    "load",
    "load_activation_calibration",
    "load_module_conversion",
    "load_module_onnx",
    "load_quantized_module",
    "load_refinement",
    "load_qat_hard",
    "load_onnx",
    "log_opentelemetry",
    "log_tensorboard",
    "log_wandb",
    "prepare",
    "prepare_qat",
    "quantize",
    "refine",
    "reference_ternary_linear",
    "register_estimator",
    "register_huggingface",
    "registered_estimators",
    "run_smollm2_release_demo",
    "ternary_linear",
    "TensorDiagnostics",
    "validate_projection",
]
