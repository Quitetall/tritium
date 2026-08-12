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
from .allocation import PlaneAllocation, allocate_planes
from .conversion import PreparedModel, inspect, prepare, prepare_qat
from .coverage import CoverageEntry, CoverageReport
from .errors import TritiumError
from .estimators import (
    AbsMeanSTE,
    AdditiveEstimator,
    AnnealedSTE,
    Estimator,
    HestiaEstimator,
    LSQEstimator,
    SaltSTE,
    SparseTernaryEstimator,
    TTQEstimator,
    TWNEstimator,
    create_estimator,
    hestia_soft_expectation,
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
from .qat import QatHardConsumer, QatHardResult, QatHardWeight, convert_qat_hard
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
    capture_kronecker_module_group,
    capture_qwen36_kronecker_evidence,
    convert,
    load_activation_calibration,
    load_module_conversion,
    load_quantized_module,
    quantize,
)
from .qwen36 import (
    Qwen36ComponentError,
    Qwen36Components,
    Qwen36MtpAdapter,
    Qwen36MtpLoadError,
    attach_qwen36_mtp,
    capture_qwen36_components,
    resolve_qwen36_components,
)
from .tutorial import (
    SMOLLM2_MODEL_ID,
    SMOLLM2_REVISION,
    run_smollm2_release_demo,
)
from .stage7 import Stage7CausalData, Stage7CausalDataReceipt
from .stage7_smoke import (
    SMOLLM2_135M_MODEL_ID,
    SMOLLM2_135M_REPO_ID,
    SMOLLM2_135M_REVISION,
    Stage7SmokeModelResult,
    Stage7SmolLM2SmokeResult,
    run_stage7_smoke_model,
    run_stage7_smollm2_smoke,
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
    "HestiaEstimator",
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
    "PlaneAllocation",
    "ProjectionContext",
    "PreparedModel",
    "QatHardArtifact",
    "QatHardConsumer",
    "QatHardResult",
    "QatHardWeight",
    "QuantizationResult",
    "QwenOnnxCausalLM",
    "Qwen36ComponentError",
    "Qwen36Components",
    "Qwen36MtpAdapter",
    "Qwen36MtpLoadError",
    "attach_qwen36_mtp",
    "RefinementConfig",
    "RefinementResult",
    "SaltSTE",
    "ScaleStatistics",
    "SparseTernaryEstimator",
    "SMOLLM2_MODEL_ID",
    "SMOLLM2_REVISION",
    "Stage7CausalData",
    "Stage7CausalDataReceipt",
    "SMOLLM2_135M_MODEL_ID",
    "SMOLLM2_135M_REPO_ID",
    "SMOLLM2_135M_REVISION",
    "Stage7SmokeModelResult",
    "Stage7SmolLM2SmokeResult",
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
    "hestia_soft_expectation",
    "calibrate",
    "allocate_planes",
    "bind_kronecker_activation_cache_digest",
    "capture_kronecker_embedding",
    "capture_kronecker_module",
    "capture_kronecker_module_group",
    "capture_qwen36_kronecker_evidence",
    "capture_qwen36_components",
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
    "resolve_qwen36_components",
    "reference_ternary_linear",
    "register_estimator",
    "register_huggingface",
    "registered_estimators",
    "run_smollm2_release_demo",
    "run_stage7_smoke_model",
    "run_stage7_smollm2_smoke",
    "ternary_linear",
    "TensorDiagnostics",
    "validate_projection",
]

# ADR 0037 Stage 4: importing the consolidated quantizers registers the
# "tequila-lsq" / "pareto-seq" estimator ids as a side effect — without this,
# create_estimator("tequila-lsq") only works if a caller happened to import
# the module first.
from . import quantizers as _quantizers  # noqa: E402,F401  (registry side effect)
