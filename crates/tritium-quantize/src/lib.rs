//! # tritium-quantize
//!
//! **SALT** — Sensitivity-Allocated Layered Ternary quantization (ADR 0001).
//! Turns fp weights into a stack of ternary planes `W ≈ Σ_p s_p · t_p` (each
//! `t_p ∈ {-1, 0, +1}`), spending extra planes only on the weight groups the
//! model is sensitive to, under a bits-per-weight budget. Inference stays
//! multiply-free: a `T`-plane weight is `T` add/sub/skip passes, scaled and
//! summed — the existing ternary mpGEMM kernel, looped.
//!
//! ## Pipeline (ADR 0001) and where each stage lives
//!
//! 1. **Residual ternary expansion** — [`residual_expand`], [`Plane`],
//!    [`PlaneStack`]. Greedy AbsMean per plane; `T = 1` is exactly flat BitNet
//!    b1.58. Reconstruction + error: [`PlaneStack::reconstruct`], [`recon_error`].
//! 2. Mode codebook — *(later v0.40 step)*.
//! 3. Sensitivity rank + **4. rate-distortion plane allocation** —
//!    [`allocate`], [`GroupInput`], [`AllocConfig`], [`Allocation`]. Greedy
//!    water-filling over per-group error curves under a bits-per-weight budget.
//! 5. Sparse residual planes / 6. heal — *(later; GPU + train).*
//!
//! The format sidecar (`tritium-format`) and the multi-plane accumulate kernel
//! (CUDA/CPU backends) consume what this crate produces; they land in their own
//! v0.40 steps (ADR 0006).
//!
//! ## CPU-only exit gates (ADR 0006), all enforced here
//!
//! - `T = 1` reduces **exactly** to flat AbsMean (BitNet regression golden).
//! - Reconstruction error is **monotonic** non-increasing in plane count `T`.
//! - Same input ⇒ **byte-identical** output (determinism).
//!
//! GPU gates (multi-plane accumulate matches dequant; sparse == dense) and the
//! model-level accuracy-vs-bpw curve gate are validated in their own lanes.
// v0.90 hardening: every public item must carry a doc comment.
#![deny(missing_docs)]

mod allocate;
mod architecture;
mod campaign;
mod conversion;
pub mod fisher;
mod plane;
mod quantize;
mod qwen35_coverage;
mod recon;
mod salt_v2;
mod salt_v2_activation;
mod salt_v2_allocator;
mod salt_v2_curvature;
mod salt_v2_evidence;
mod salt_v2_feedback;
mod salt_v2_model;
mod training_export;

pub use allocate::{AllocConfig, AllocError, Allocation, GroupInput, TRIT_BITS, allocate};
pub use architecture::{
    AdapterError, ArchitectureAdapter, ArchitectureFeature, ArchitectureRequirements,
    CapabilityGap, CapabilitySet, TensorDescriptor, TensorDisposition, TensorRole,
};
pub use campaign::{
    CalibrationId, CalibrationProvenance, CampaignError, CampaignId, CampaignLedger,
    CampaignMetrics, CampaignObjective, CampaignPoint, EvaluationId, EvaluationProvenance,
    ExactMillibpw, LogicalTritCount, LogicalTritRate, MeasuredPackage, PhysicalSizeReport,
    RecipeId, RecipeProvenance, ResidentSizeComponents, SerializedSizeComponents,
};
pub use conversion::{
    CONVERSION_STATE_MAGIC, CONVERSION_STATE_VERSION, ConversionError, ConversionRun,
    ConversionStage, RunStatus, StageAttempt, StageFailure, StageReceipt,
};
pub use plane::{
    Plane, PlaneStack, absmean_ternary, recon_error, residual_expand, ternary_at_scale,
};
pub use quantize::{
    BaseScaleScope, QuantConfig, QuantError, QuantizedTensor, Sensitivity, quantize_tensor,
};
pub use qwen35_coverage::{
    QWEN36_27B_COVERAGE_REVISION, Qwen35CoverageDisposition, Qwen35CoverageEntry,
    Qwen35CoverageError, Qwen35CoverageManifest, Qwen35CoverageSummary, Qwen35CoverageTotals,
    Qwen35LanguageLayerKind, Qwen35SourceDtype, Qwen35TensorMetadata, Qwen35TensorRole,
    Qwen35TensorScope,
};
pub use recon::{ReconAccum, ReconError, ReconStats, reconstruction_stats};
pub use salt_v2::{
    DensePsdMetric, JointFitConfig, JointFitError, JointFitMetric, JointFitRestartReceipt,
    JointFitStartKind, JointFitUpdatePhase, JointFitUpdateReceipt, JointTernaryFit, ScalePrecision,
    ScaleSolveReceipt, ScaleSolveTelemetry, exact_ternary_assignment, fit_joint_ternary,
};
pub use salt_v2_activation::{
    ActivationByteLedger, ActivationCache, ActivationCacheBuilder, ActivationCacheError,
    ActivationCacheSpec, ActivationChunk, ActivationDType, ActivationDigest, ActivationShard,
};
pub use salt_v2_allocator::{
    ByteDelta, GroupCandidates, NestedProfileAllocation, NestedProfileBudgets, PhysicalAllocError,
    PhysicalBytes, PlaneCandidate, ProfileAllocation, ProfileBudget, SaltV2Profile,
    allocate_nested_profiles,
};
pub use salt_v2_curvature::{
    CurvatureError, CurvatureSourceId, InputGram, InputGramAccumulator, KfacMetric, OutputFisher,
    OutputFisherAccumulator, build_kfac_metric,
};
pub use salt_v2_evidence::{
    SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceError, SaltV2KroneckerEvidenceReceipt,
};
pub use salt_v2_feedback::{
    ColumnGroup, FeedbackError, FeedbackMetric, FeedbackProblem, FeedbackRunError, FeedbackState,
    GroupFitRequest, fit_with_feedback,
};
pub use salt_v2_model::{
    CurvatureArtifact, KroneckerCurvature, PhysicalRateTarget, SaltV2Config, SaltV2Curvature,
    SaltV2DriverError, SaltV2Error, SaltV2ExternalStage, SaltV2ExternalStageRequest,
    SaltV2FeedbackArtifact, SaltV2FeedbackGroupReceipt, SaltV2FitConstraint, SaltV2FitTrack,
    SaltV2MasterFit, SaltV2MasterFitInput, SaltV2ModelFeedbackReceipt, SaltV2ModelFitInput,
    SaltV2ModelFitMetrics, SaltV2ModelFitReceipt, SaltV2ModelFitResult, SaltV2ModelPhysicalInput,
    SaltV2ModelStageDriver, SaltV2Packing, SaltV2PhysicalSize, SaltV2Refinement,
    SaltV2RestartableTensorMasterFitInput, SaltV2TensorFeedbackReceipt, SaltV2TensorFitInput,
    SaltV2TensorFitReceipt, SaltV2TensorMasterFitInput, SaltV2TensorMasterFitResult,
    SaltV2TileCandidateMetrics, allocate_and_pack_salt_v2_master,
    allocate_and_pack_salt_v2_master_with_packing, fit_salt_v2_master, fit_salt_v2_model,
    fit_salt_v2_restartable_tensor_master, fit_salt_v2_tensor_master,
    plan_salt_v2_restartable_tensor_master, plan_salt_v2_tensor_master,
};
pub use training_export::{
    TrainingSaltExportError, TrainingSaltExportStats, export_training_salt_row,
};
