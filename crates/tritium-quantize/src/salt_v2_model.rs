//! Whole-model reference fitting contract for SALT V2.

use core::fmt;
use std::collections::BTreeSet;

use half::f16;
use tritium_format::salt_v2::{SaltV2Codec, SaltV2CodecError};
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_PACKAGE_ALIGNMENT, SALT_V2_SCALE_GROUP_SIZE,
    SaltV2IndexedRuntimeLedger, SaltV2Package, SaltV2PackageError, SaltV2Plane, SaltV2Tensor,
    SaltV2Tile, write_salt_v2_package,
};
use tritium_format::{ModelId, PackageId};

use crate::salt_v2::{
    DensePsdMetric, JointFitConfig, JointFitError, JointFitMetric, ScalePrecision,
    fit_joint_ternary,
};
use crate::salt_v2_activation::ActivationCache;
use crate::salt_v2_allocator::{
    ByteDelta, GroupCandidates, NestedProfileBudgets, PhysicalAllocError, PhysicalBytes,
    PlaneCandidate, ProfileBudget, allocate_nested_profiles,
};
use crate::salt_v2_curvature::CurvatureSourceId;
use crate::salt_v2_feedback::{
    ColumnGroup, FeedbackError, FeedbackMetric, FeedbackProblem, FeedbackRunError, GroupFitRequest,
    fit_with_feedback,
};

const REFERENCE_SOLVER_VERSION: &str = "tritium-salt-v2-reference-model-fit-v2";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 model fit receipt v2";
const RECIPE_HASH_CONTEXT: &str = "tritium salt v2 model fit recipe v1";
const SOURCE_TENSOR_HASH_CONTEXT: &str = "tritium salt v2 source tensor v1";
const CURVATURE_HASH_CONTEXT: &str = "tritium salt v2 bound curvature artifact v2";
const FEEDBACK_HASH_CONTEXT: &str = "tritium salt v2 bound feedback artifact v1";
const FEEDBACK_RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 feedback receipt v1";
const MASTER_HASH_CONTEXT: &str = "tritium salt v2 ordered master fit v1";

/// Physical ternary codec selected for the complete SALT V2 package.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaltV2Packing {
    /// Aligned two-bit reference codec.
    #[default]
    D2,
    /// Dense radix-3 codec, five trits per byte.
    B3,
    /// Structured one-zero-per-four codec.
    S34,
}

impl SaltV2Packing {
    fn codec(self) -> SaltV2Codec {
        match self {
            Self::D2 => SaltV2Codec::D2,
            Self::B3 => SaltV2Codec::B3,
            Self::S34 => SaltV2Codec::S34,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::D2 => 1,
            Self::B3 => 2,
            Self::S34 => 3,
        }
    }
}

/// Curvature recipe whose precomputed evidence is supplied to the model fitter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaltV2Curvature {
    /// Per-weight empirical-Fisher diagonal.
    #[default]
    DiagonalFisher,
    /// Input activation Hessian.
    InputHessian,
    /// End-loss GuidedQuant-style Fisher curvature.
    GuidedFisher,
    /// Forward-KL Kronecker curvature.
    ForwardKlKronecker,
}

impl SaltV2Curvature {
    const fn tag(self) -> u8 {
        match self {
            Self::DiagonalFisher => 1,
            Self::InputHessian => 2,
            Self::GuidedFisher => 3,
            Self::ForwardKlKronecker => 4,
        }
    }
}

/// Model-recovery track. The reference fitter implements only [`Self::None`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaltV2Refinement {
    /// Pure PTQ: no parameter receives a gradient update.
    #[default]
    None,
    /// Fixed-allocation, fixed-trit scale-only refinement.
    ScaleOnly {
        /// Hard maximum number of refinement tokens.
        max_tokens: u64,
    },
    /// Smooth warmup followed by hard PV-style discrete/scale refinement.
    PvKl {
        /// Maximum soft-warmup tokens.
        warmup_tokens: u64,
        /// Maximum hard-trit tail tokens.
        hard_tokens: u64,
    },
}

/// Exact whole-model integer ceilings. A nominal bpw is never accepted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalRateTarget {
    /// Maximum exact serialized SALT matrix/package bytes.
    pub max_matrix_bytes: u64,
    /// Maximum SALT bytes plus preserved artifact bytes.
    pub max_artifact_bytes: u64,
    /// Optional maximum steady resident bytes, including preserved and shadow bytes.
    pub max_resident_bytes: Option<u64>,
}

impl Default for PhysicalRateTarget {
    fn default() -> Self {
        Self {
            max_matrix_bytes: u64::MAX,
            max_artifact_bytes: u64::MAX,
            max_resident_bytes: None,
        }
    }
}

/// Experimental model-level SALT V2 fitting recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaltV2Config {
    /// Coefficients sharing a deployment scale. The reference/package contract is G128.
    pub group_size: usize,
    /// Minimum plane count. The package's mandatory prefix is one plane.
    pub min_planes: usize,
    /// Maximum plane count, either two or three.
    pub max_planes: usize,
    /// One physical codec for the whole output package.
    pub packing: SaltV2Packing,
    /// Curvature recipe required from every tensor input.
    pub curvature: SaltV2Curvature,
    /// Optional signed-RHT seed. The reference fitter rejects it until a driver is linked.
    pub transform_seed: Option<u64>,
    /// Deterministic output-aware OA-EM restart count.
    pub em_restarts: usize,
    /// Maximum joint assignment/scale coordinate sweeps.
    pub coordinate_sweeps: usize,
    /// Finite positive condition threshold used to derive the reference solve ridge.
    pub ridge_condition_limit: f64,
    /// Exact physical ceilings.
    pub rate: PhysicalRateTarget,
    /// Explicit PTQ, scale-only, or short-PV track.
    pub refinement: SaltV2Refinement,
}

impl Default for SaltV2Config {
    fn default() -> Self {
        Self {
            group_size: SALT_V2_SCALE_GROUP_SIZE,
            min_planes: 1,
            max_planes: 3,
            packing: SaltV2Packing::D2,
            curvature: SaltV2Curvature::DiagonalFisher,
            transform_seed: None,
            em_restarts: 4,
            coordinate_sweeps: 10,
            ridge_condition_limit: 1_000_000.0,
            rate: PhysicalRateTarget::default(),
            refinement: SaltV2Refinement::None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CurvatureValues<'a> {
    Diagonal(&'a [f32]),
    DenseGroups(&'a [DensePsdMetric]),
}

/// Digest-bound, validated-at-use curvature evidence for one source tensor.
///
/// Construction requires immutable source-model, activation-cache, and token-stream provenance.
/// [`fit_salt_v2_model`] rejects an artifact unless all three identities match its fit input.
/// For that model-fit seam, construct [`CurvatureSourceId`] with the exact mapping
/// `source_model_id.as_bytes()`, `activations.digest()`, and
/// `activations.spec().source_digest()` respectively. The final value is the cache's canonical
/// calibration/token-stream provenance envelope; the complete cache digest separately binds its
/// values, masks, boundaries, and shard manifest.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureArtifact<'a> {
    kind: SaltV2Curvature,
    source_id: CurvatureSourceId,
    evidence_digest: [u8; 32],
    content_digest: [u8; 32],
    values: CurvatureValues<'a>,
}

impl<'a> CurvatureArtifact<'a> {
    /// Bind a per-weight empirical-Fisher diagonal to its canonical artifact digest.
    #[must_use]
    pub fn diagonal_fisher(
        source_id: CurvatureSourceId,
        evidence_digest: [u8; 32],
        diagonal: &'a [f32],
    ) -> Self {
        let values = CurvatureValues::Diagonal(diagonal);
        Self {
            kind: SaltV2Curvature::DiagonalFisher,
            source_id,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::DiagonalFisher,
                source_id,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise dense input-Hessian blocks to their canonical artifact digest.
    #[must_use]
    pub fn input_hessian(
        source_id: CurvatureSourceId,
        evidence_digest: [u8; 32],
        groups: &'a [DensePsdMetric],
    ) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::InputHessian,
            source_id,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::InputHessian,
                source_id,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise dense guided-Fisher blocks to their canonical artifact digest.
    #[must_use]
    pub fn guided_fisher(
        source_id: CurvatureSourceId,
        evidence_digest: [u8; 32],
        groups: &'a [DensePsdMetric],
    ) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::GuidedFisher,
            source_id,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::GuidedFisher,
                source_id,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise forward-KL Kronecker blocks to their canonical artifact digest.
    #[must_use]
    pub fn forward_kl_kronecker(
        source_id: CurvatureSourceId,
        evidence_digest: [u8; 32],
        groups: &'a [DensePsdMetric],
    ) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::ForwardKlKronecker,
            source_id,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::ForwardKlKronecker,
                source_id,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Curvature recipe represented by this artifact.
    #[must_use]
    pub const fn kind(self) -> SaltV2Curvature {
        self.kind
    }

    /// Immutable source-model, activation-cache, and token-stream provenance.
    #[must_use]
    pub const fn source_id(self) -> CurvatureSourceId {
        self.source_id
    }

    /// Upstream evidence digest supplied by the curvature builder.
    #[must_use]
    pub const fn evidence_digest(self) -> [u8; 32] {
        self.evidence_digest
    }

    /// Digest binding the upstream evidence ID to exact dimensions and curvature values.
    #[must_use]
    pub const fn digest(self) -> [u8; 32] {
        self.content_digest
    }
}

/// Digest-bound full input-column inverse Hessian and deterministic column partition.
///
/// This evidence is deliberately separate from the per-G128 fitting curvature. The latter scores
/// and refits additive planes; this artifact drives BlockLDLQ/GPTQ residual propagation across
/// natural input-column groups. Both are source-bound and independently receipted.
#[derive(Clone, Copy, Debug)]
pub struct SaltV2FeedbackArtifact<'a> {
    source_id: CurvatureSourceId,
    evidence_digest: [u8; 32],
    groups: &'a [ColumnGroup],
    inverse_hessian: &'a [f64],
    content_digest: [u8; 32],
}

impl<'a> SaltV2FeedbackArtifact<'a> {
    /// Bind a dense full input-column inverse Hessian to source and calibration provenance.
    #[must_use]
    pub fn inverse_hessian(
        source_id: CurvatureSourceId,
        evidence_digest: [u8; 32],
        groups: &'a [ColumnGroup],
        inverse_hessian: &'a [f64],
    ) -> Self {
        Self {
            source_id,
            evidence_digest,
            groups,
            inverse_hessian,
            content_digest: bound_feedback_digest(
                source_id,
                evidence_digest,
                groups,
                inverse_hessian,
            ),
        }
    }

    /// Immutable source-model, activation-cache, and token-stream provenance.
    #[must_use]
    pub const fn source_id(self) -> CurvatureSourceId {
        self.source_id
    }

    /// Upstream inverse-Hessian evidence digest.
    #[must_use]
    pub const fn evidence_digest(self) -> [u8; 32] {
        self.evidence_digest
    }

    /// Ordered natural input-column partition.
    #[must_use]
    pub const fn groups(self) -> &'a [ColumnGroup] {
        self.groups
    }

    /// Dense row-major full inverse Hessian.
    #[must_use]
    pub const fn values(self) -> &'a [f64] {
        self.inverse_hessian
    }

    /// Digest binding provenance, partition, dimensions, and exact binary64 values.
    #[must_use]
    pub const fn digest(self) -> [u8; 32] {
        self.content_digest
    }
}

/// One named row-major source tensor and its fitted curvature evidence.
#[derive(Clone, Copy, Debug)]
pub struct SaltV2TensorFitInput<'a> {
    /// Canonical tensor name.
    pub name: &'a str,
    /// Finite source weights in row-major order.
    pub weights: &'a [f32],
    /// Matrix output rows.
    pub rows: usize,
    /// Matrix input columns.
    pub cols: usize,
    /// Digest-bound curvature evidence selected by [`SaltV2Config::curvature`].
    pub curvature: CurvatureArtifact<'a>,
}

/// Whole-model preserved bytes that are outside the additive-ternary tensor package.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaltV2ModelPhysicalInput {
    /// Denominator for whole-artifact and resident bpw.
    pub total_model_parameters: u64,
    /// Serialized bytes for preserved tensors, configuration, and required side assets.
    pub preserved_artifact_bytes: u64,
    /// Steady resident bytes for preserved tensors and runtime metadata.
    pub preserved_resident_bytes: u64,
    /// Any mandatory runtime shadow representation.
    pub required_runtime_shadow_bytes: u64,
}

/// Complete whole-model reference-fit input.
#[derive(Clone, Copy, Debug)]
pub struct SaltV2ModelFitInput<'a> {
    /// Quantized two-dimensional tensors.
    pub tensors: &'a [SaltV2TensorFitInput<'a>],
    /// Canonical activation cache whose identity binds the calibration evidence.
    pub activations: &'a ActivationCache,
    /// Semantic identity of the source model.
    pub source_model_id: ModelId,
    /// Preserved and resident model geometry outside the SALT package.
    pub physical: SaltV2ModelPhysicalInput,
}

/// Search-stage input that requires one full feedback artifact per source tensor.
#[derive(Clone, Copy, Debug)]
pub struct SaltV2MasterFitInput<'a> {
    /// Ordinary source tensors, calibration cache, model identity, and physical geometry.
    pub model: SaltV2ModelFitInput<'a>,
    /// Feedback artifacts in exact tensor order.
    pub feedback: &'a [SaltV2FeedbackArtifact<'a>],
}

/// Recovery track stated by a completed result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaltV2FitTrack {
    /// Calibration-only post-training quantization.
    Ptq,
    /// Fixed-trit scale-only recovery.
    ScaleOnly,
    /// Short smooth-to-hard PV/KL recovery.
    PvKl,
}

/// Production stage deliberately not implemented by the small CPU reference fitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2ExternalStage {
    /// Signed randomized Hadamard transform and its online execution contract.
    SignedRht,
    /// Cached-activation block-output reconstruction.
    BlockOutputReconstruction,
    /// Fixed-trit scale-only teacher-KL refinement.
    ScaleOnlyRefinement,
    /// Smooth warmup and hard PV/KL refinement.
    PvKlRefinement,
}

/// Typed request boundary for an external production stage driver.
#[derive(Clone, Copy, Debug)]
pub struct SaltV2ExternalStageRequest<'a> {
    /// Stage the driver must execute without changing the recipe identity.
    pub stage: SaltV2ExternalStage,
    /// Frozen recipe.
    pub config: &'a SaltV2Config,
    /// Activation evidence identity.
    pub activation_digest: [u8; 32],
    /// Source-model identity.
    pub source_model_id: ModelId,
}

/// Driver boundary for production-only activation-output or refinement stages.
///
/// The reference fitter never invokes this trait implicitly. Callers must link a production
/// pipeline that validates and packages the driver's output, rather than treating a missing stage
/// as a local-objective fallback.
pub trait SaltV2ModelStageDriver {
    /// Execute one explicit external stage and return a content digest of its durable evidence.
    fn run_stage(
        &mut self,
        request: SaltV2ExternalStageRequest<'_>,
    ) -> Result<[u8; 32], SaltV2DriverError>;
}

/// Failure returned by a production stage driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2DriverError {
    /// Stable machine-readable driver code.
    pub code: String,
    /// Human-readable evidence-preserving failure detail.
    pub detail: String,
}

/// Exact serialized/resident whole-model physical accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaltV2PhysicalSize {
    /// Package/tensor headers and presence maps before payloads and final alignment.
    pub serialized_fixed_bytes: u64,
    /// Exact physical SALT package bytes, including final alignment padding.
    pub matrix_bytes: u64,
    /// Matrix plus preserved artifact bytes.
    pub artifact_bytes: u64,
    /// Package resident bytes plus preserved and mandatory shadow bytes.
    pub resident_bytes: u64,
    /// Whole-model peak residency, unavailable until a complete runtime measures it.
    pub peak_resident_bytes: Option<u64>,
    /// Exact serialized package padding.
    pub padding_bytes: u64,
    /// Encoded hard-trit payload bytes.
    pub trit_payload_bytes: u64,
    /// Non-negative f16 scale bytes.
    pub scale_bytes: u64,
    /// Optional-plane presence map bytes.
    pub allocation_map_bytes: u64,
    /// Logical allocation-map bits, including bits embedded in count words.
    pub allocation_map_bits: u64,
    /// Allocation-map bits embedded in mandatory count words.
    pub allocation_map_embedded_bits: u64,
    /// Package/tensor header bytes.
    pub header_bytes: u64,
    /// Serialized transform metadata bytes, disjoint from headers.
    pub transform_bytes: u64,
    /// Indexed-runtime tile-prefix map bytes.
    pub runtime_map_bytes: u64,
    /// Indexed-runtime coarse plane-rank prefix bytes.
    pub runtime_rank_prefix_bytes: u64,
    /// Dense indexed-runtime weight shadow bytes, structurally zero.
    pub runtime_dense_shadow_bytes: u64,
    /// Preserved serialized bytes supplied by the architecture adapter.
    pub preserved_artifact_bytes: u64,
    /// Preserved steady resident bytes supplied by the architecture adapter.
    pub preserved_resident_bytes: u64,
    /// Required runtime shadow bytes.
    pub required_runtime_shadow_bytes: u64,
}

/// One refitted tile candidate and its exact cumulative physical cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaltV2TileCandidateMetrics {
    /// Source tensor ordinal.
    pub tensor_index: usize,
    /// Allocation-tile ordinal within the tensor.
    pub tile_index: usize,
    /// Candidate prefix plane count.
    pub planes: u8,
    /// Exact cumulative payload-plus-scale bytes for this candidate.
    pub cumulative: PhysicalBytes,
    /// Full curvature-weighted reconstruction error.
    pub hessian_error: f64,
    /// Ordinary squared reconstruction error.
    pub frobenius_error: f64,
}

/// Whole-model quality and exact-rate telemetry for one selected reference fit.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2ModelFitMetrics {
    /// Number of source coefficients represented by hard ternary planes.
    pub quantized_parameter_count: u64,
    /// Coefficient count selected at P=1, P=2, and P=3 respectively.
    pub plane_histogram: [u64; 3],
    /// Selected plane count for every allocation tile in tensor order.
    pub selected_plane_counts: Vec<u8>,
    /// All refitted per-tile candidate points admitted by the recipe.
    pub tile_candidates: Vec<SaltV2TileCandidateMetrics>,
    /// Exact component and whole-model byte ledger.
    pub physical: SaltV2PhysicalSize,
    /// Exact number of hard ternary symbols selected by the allocator.
    pub logical_trits: u64,
    /// Information-theoretic hard-trit bits, reported only beside physical rates.
    pub logical_bits: f64,
    /// Logical bits per quantized coefficient.
    pub logical_bpw: f64,
    /// Exact SALT package bits per quantized coefficient.
    pub matrix_bpw: f64,
    /// Exact artifact bits per total model parameter.
    pub artifact_bpw: f64,
    /// Exact steady resident bits per total model parameter.
    pub resident_bpw: f64,
    /// Selected ordinary squared reconstruction error.
    pub frobenius_error: f64,
    /// Selected curvature-weighted reconstruction error.
    pub hessian_error: f64,
    /// Cached-activation block-output error, absent from the small reference fitter.
    pub block_output_error: Option<f64>,
    /// Teacher forward KL, absent from pure PTQ reference fitting.
    pub teacher_kl: Option<f64>,
}

/// Receipt binding one tensor's source and curvature evidence to selected planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2TensorFitReceipt {
    /// Canonical tensor name.
    pub name: String,
    /// Digest of name, shape, and exact source f32 bits.
    pub source_digest: [u8; 32],
    /// Digest of the precomputed curvature artifact.
    pub curvature_digest: [u8; 32],
    /// Selected plane count for each allocation tile.
    pub plane_counts: Vec<u8>,
}

/// Replay binding for one natural input-column feedback group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2FeedbackGroupReceipt {
    /// Stable group ordinal.
    pub group_index: usize,
    /// First source input column.
    pub column_start: usize,
    /// Exclusive source input column end.
    pub column_end: usize,
    /// Digest of the provisional feedback-adjusted block fitted before full refinement.
    pub provisional_fit_input_digest: [u8; 32],
    /// Digest of the provisional reconstruction installed into the feedback state.
    pub provisional_reconstruction_digest: [u8; 32],
    /// Digest of the final feedback-adjusted block consumed by the full joint refit.
    pub final_fit_input_digest: [u8; 32],
    /// Digest of the final additive reconstruction propagated downstream.
    pub final_reconstruction_digest: [u8; 32],
    /// Whether the full refit changed the installed reconstruction bits.
    pub nonzero_delta: bool,
}

/// Detailed BlockLDLQ/delta-correction receipt for one source tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2TensorFeedbackReceipt {
    /// Canonical tensor name.
    pub name: String,
    /// Digest of the full inverse-Hessian evidence and natural group partition.
    pub artifact_digest: [u8; 32],
    /// Per-group provisional/refined replay bindings in propagation order.
    pub groups: Vec<SaltV2FeedbackGroupReceipt>,
    /// Number of full refits installed through exact residual-delta propagation.
    pub delta_corrections: u64,
    /// Number of those refits whose reconstruction bits changed.
    pub nonzero_delta_corrections: u64,
    /// Digest of the complete final feedback-adjusted working matrix.
    pub final_working_digest: [u8; 32],
    /// Digest of the complete final additive reconstruction.
    pub final_reconstruction_digest: [u8; 32],
}

/// Model-wide second-order feedback receipt bound into the ordered master and final package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2ModelFeedbackReceipt {
    /// Tensor receipts in source order.
    pub tensors: Vec<SaltV2TensorFeedbackReceipt>,
    /// Digest over all tensor feedback receipts.
    pub receipt_id: [u8; 32],
}

/// Deterministic content-bound receipt for one reference whole-model fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2ModelFitReceipt {
    /// Receipt schema/solver identity.
    pub solver_version: &'static str,
    /// Semantic source-model identity.
    pub source_model_id: ModelId,
    /// Canonical activation-cache identity.
    pub activation_digest: [u8; 32],
    /// Canonical recipe identity.
    pub recipe_id: [u8; 32],
    /// Ordered Pmax master identity from which every selected prefix was sliced.
    pub master_id: [u8; 32],
    /// Source/curvature bindings in tensor order.
    pub tensors: Vec<SaltV2TensorFitReceipt>,
    /// Exact package-byte identity.
    pub package_id: PackageId,
    /// Whole-model parameter and preserved/runtime byte geometry used for authorization.
    pub physical: SaltV2ModelPhysicalInput,
    /// Digest over all receipt fields, including the exact package identity.
    pub receipt_id: [u8; 32],
    /// Explicit recovery track; the reference fitter returns only PTQ.
    pub track: SaltV2FitTrack,
    /// Whether second-order sequential feedback was applied.
    pub feedback_applied: bool,
    /// Detailed feedback evidence; present exactly when `feedback_applied` is true.
    pub feedback: Option<SaltV2ModelFeedbackReceipt>,
    /// Whether cached-activation output reconstruction was applied.
    pub output_reconstruction_applied: bool,
}

/// Successful whole-model SALT V2 reference fit.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2ModelFitResult {
    /// Canonical semantic tensors containing only scales and hard trits.
    pub tensors: Vec<SaltV2Tensor>,
    /// Exact canonical package bytes.
    pub package_bytes: Vec<u8>,
    /// Frozen recipe copied into the result for direct ceiling audit.
    pub config: SaltV2Config,
    /// Quality and exact physical telemetry.
    pub metrics: SaltV2ModelFitMetrics,
    /// Content-bound replay receipt.
    pub receipt: SaltV2ModelFitReceipt,
}

/// Why the reference whole-model SALT V2 fitter could not produce an artifact.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SaltV2Error {
    /// The semantic package supports G128 only in this reference path.
    UnsupportedReferenceGroupSize {
        /// Rejected group size.
        got: usize,
    },
    /// The semantic package's mandatory prefix requires `min_planes == 1`.
    UnsupportedMinimumPlanes {
        /// Rejected minimum plane count.
        got: usize,
    },
    /// Maximum planes was outside `2..=3`.
    InvalidMaximumPlanes {
        /// Rejected maximum plane count.
        got: usize,
    },
    /// Coordinate sweeps was zero.
    InvalidCoordinateSweeps,
    /// OA-EM restart count was zero.
    InvalidEmRestarts,
    /// Ridge condition limit was not finite and strictly greater than one.
    InvalidRidgeConditionLimit,
    /// A physical byte ceiling was zero or inconsistent.
    InvalidPhysicalRateTarget,
    /// A refinement token schedule was zero or violated the final 20% hard-tail rule.
    InvalidRefinementSchedule,
    /// No source tensors were supplied.
    EmptyModel,
    /// A tensor name was empty.
    EmptyTensorName {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Two tensor inputs had the same name.
    DuplicateTensorName(String),
    /// Tensor dimensions were zero or their product overflowed.
    InvalidTensorShape {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Tensor shape did not match its weight slice.
    TensorLengthMismatch {
        /// Tensor ordinal.
        tensor: usize,
        /// Shape-derived coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        got: usize,
    },
    /// A source weight was NaN or infinite.
    NonFiniteWeight {
        /// Tensor ordinal.
        tensor: usize,
        /// Flat coefficient index.
        index: usize,
    },
    /// Tensor curvature recipe differed from the frozen configuration.
    CurvatureKindMismatch {
        /// Tensor ordinal.
        tensor: usize,
        /// Required recipe.
        expected: SaltV2Curvature,
        /// Supplied recipe.
        got: SaltV2Curvature,
    },
    /// Curvature evidence was produced for a different semantic source model.
    CurvatureSourceModelMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Curvature evidence was produced from a different activation-cache artifact.
    CurvatureActivationCacheMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Curvature evidence used a different calibration/token-stream provenance envelope.
    CurvatureTokenStreamMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// A curvature artifact had the wrong diagonal or dense-block geometry.
    CurvatureGeometry {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Feedback artifact count differed from the source tensor count.
    FeedbackArtifactCountMismatch {
        /// Required number of artifacts.
        expected: usize,
        /// Supplied number of artifacts.
        got: usize,
    },
    /// Feedback evidence was produced for a different semantic source model.
    FeedbackSourceModelMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Feedback evidence was produced from a different activation-cache artifact.
    FeedbackActivationCacheMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Feedback evidence used a different calibration/token-stream provenance envelope.
    FeedbackTokenStreamMismatch {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Feedback groups cannot preserve the package's row-local G128 scale geometry.
    FeedbackScaleGeometry {
        /// Tensor ordinal.
        tensor: usize,
    },
    /// Full inverse-Hessian validation or propagation failed.
    Feedback {
        /// Tensor ordinal.
        tensor: usize,
        /// Underlying feedback error.
        source: FeedbackError,
    },
    /// Total model parameters were smaller than the quantized coefficient count.
    InvalidTotalModelParameters,
    /// Integer byte or coefficient accounting overflowed.
    AccountingOverflow,
    /// A required production stage has no safe reference implementation.
    ExternalStageRequired {
        /// Stage that must be supplied by a production driver.
        stage: SaltV2ExternalStage,
    },
    /// A joint group fit failed.
    JointFit {
        /// Tensor ordinal.
        tensor: usize,
        /// Allocation tile ordinal.
        tile: usize,
        /// G128 group ordinal within the tensor.
        group: usize,
        /// Candidate plane count.
        planes: usize,
        /// Underlying solver error.
        source: JointFitError,
    },
    /// A refitted P+1 candidate increased the active curvature objective.
    NonMonotoneCandidate {
        /// Tensor ordinal.
        tensor: usize,
        /// Allocation tile ordinal.
        tile: usize,
        /// Rejected plane count.
        planes: usize,
    },
    /// Exact codec payload pricing failed.
    Codec(SaltV2CodecError),
    /// Exact Pareto allocation failed.
    Allocation(PhysicalAllocError),
    /// Mandatory planes or preserved bytes left no point under all hard ceilings.
    NoFeasibleAllocation {
        /// Effective aligned package ceiling after artifact overhead.
        max_matrix_bytes: u64,
        /// Effective package resident ceiling after preserved/shadow bytes.
        max_resident_bytes: u64,
    },
    /// Canonical semantic package construction or writing failed.
    Package(SaltV2PackageError),
    /// Predicted additive accounting differed from the canonical package writer.
    PhysicalAccountingMismatch,
}

impl fmt::Display for SaltV2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedReferenceGroupSize { got } => {
                write!(
                    formatter,
                    "reference SALT V2 package requires group128, got {got}"
                )
            }
            Self::UnsupportedMinimumPlanes { got } => {
                write!(
                    formatter,
                    "reference SALT V2 requires min_planes=1, got {got}"
                )
            }
            Self::InvalidMaximumPlanes { got } => {
                write!(formatter, "SALT V2 max_planes must be 2 or 3, got {got}")
            }
            Self::InvalidCoordinateSweeps => {
                formatter.write_str("coordinate_sweeps must be nonzero")
            }
            Self::InvalidEmRestarts => formatter.write_str("em_restarts must be nonzero"),
            Self::InvalidRidgeConditionLimit => formatter
                .write_str("ridge_condition_limit must be finite and strictly greater than one"),
            Self::InvalidPhysicalRateTarget => {
                formatter.write_str("physical rate target is zero or inconsistent")
            }
            Self::InvalidRefinementSchedule => {
                formatter.write_str("refinement schedule is empty or violates the hard-tail rule")
            }
            Self::EmptyModel => formatter.write_str("SALT V2 model input is empty"),
            Self::EmptyTensorName { tensor } => {
                write!(formatter, "tensor {tensor} has an empty name")
            }
            Self::DuplicateTensorName(name) => write!(formatter, "duplicate tensor name `{name}`"),
            Self::InvalidTensorShape { tensor } => {
                write!(formatter, "tensor {tensor} has an invalid shape")
            }
            Self::TensorLengthMismatch {
                tensor,
                expected,
                got,
            } => write!(
                formatter,
                "tensor {tensor} shape needs {expected} weights, received {got}"
            ),
            Self::NonFiniteWeight { tensor, index } => {
                write!(formatter, "tensor {tensor} weight {index} is not finite")
            }
            Self::CurvatureKindMismatch {
                tensor,
                expected,
                got,
            } => write!(
                formatter,
                "tensor {tensor} curvature is {got:?}, expected {expected:?}"
            ),
            Self::CurvatureSourceModelMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} curvature source model does not match the fit input"
            ),
            Self::CurvatureActivationCacheMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} curvature activation cache does not match the fit input"
            ),
            Self::CurvatureTokenStreamMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} curvature token stream does not match the activation cache provenance"
            ),
            Self::CurvatureGeometry { tensor } => {
                write!(
                    formatter,
                    "tensor {tensor} curvature geometry is incompatible"
                )
            }
            Self::FeedbackArtifactCountMismatch { expected, got } => write!(
                formatter,
                "feedback needs {expected} tensor artifacts, received {got}"
            ),
            Self::FeedbackSourceModelMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} feedback source model does not match the fit input"
            ),
            Self::FeedbackActivationCacheMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} feedback activation cache does not match the fit input"
            ),
            Self::FeedbackTokenStreamMismatch { tensor } => write!(
                formatter,
                "tensor {tensor} feedback token stream does not match the activation cache provenance"
            ),
            Self::FeedbackScaleGeometry { tensor } => write!(
                formatter,
                "tensor {tensor} feedback groups do not preserve row-local group128 scales"
            ),
            Self::Feedback { tensor, source } => {
                write!(formatter, "tensor {tensor} feedback failed: {source}")
            }
            Self::InvalidTotalModelParameters => formatter
                .write_str("total model parameters are below the quantized coefficient count"),
            Self::AccountingOverflow => formatter.write_str("SALT V2 accounting overflow"),
            Self::ExternalStageRequired { stage } => {
                write!(formatter, "production stage {stage:?} is required")
            }
            Self::JointFit {
                tensor,
                tile,
                group,
                planes,
                source,
            } => write!(
                formatter,
                "tensor {tensor} tile {tile} group {group} P={planes} fit failed: {source}"
            ),
            Self::NonMonotoneCandidate {
                tensor,
                tile,
                planes,
            } => write!(
                formatter,
                "tensor {tensor} tile {tile} P={planes} increased the curvature objective"
            ),
            Self::Codec(source) => write!(formatter, "codec pricing failed: {source}"),
            Self::Allocation(source) => write!(formatter, "exact allocation failed: {source}"),
            Self::NoFeasibleAllocation {
                max_matrix_bytes,
                max_resident_bytes,
            } => write!(
                formatter,
                "no allocation fits package ceiling {max_matrix_bytes} and resident ceiling {max_resident_bytes}"
            ),
            Self::Package(source) => write!(formatter, "SALT V2 package failed: {source}"),
            Self::PhysicalAccountingMismatch => {
                formatter.write_str("predicted and canonical package byte ledgers differ")
            }
        }
    }
}

impl std::error::Error for SaltV2Error {}

impl From<SaltV2CodecError> for SaltV2Error {
    fn from(value: SaltV2CodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<SaltV2PackageError> for SaltV2Error {
    fn from(value: SaltV2PackageError) -> Self {
        Self::Package(value)
    }
}

#[derive(Clone, Debug)]
struct TileFitCandidate {
    tile: SaltV2Tile,
    metrics: SaltV2TileCandidateMetrics,
}

#[derive(Clone, Debug)]
struct TensorFitWork {
    name: String,
    rows: usize,
    cols: usize,
    tile_lengths: Vec<usize>,
    candidates: Vec<Vec<TileFitCandidate>>,
    source_digest: [u8; 32],
    curvature_digest: [u8; 32],
}

/// Owned, content-addressed Pmax search result reusable across exact physical rate targets.
///
/// Candidate curves and all lower-plane packages are derived from the same ordered master planes.
/// Allocation cannot mutate or refit this artifact.
#[derive(Clone, Debug)]
pub struct SaltV2MasterFit {
    work: Vec<TensorFitWork>,
    config: SaltV2Config,
    quantized_parameters: u64,
    source_model_id: ModelId,
    activation_digest: [u8; 32],
    physical: SaltV2ModelPhysicalInput,
    feedback: Option<SaltV2ModelFeedbackReceipt>,
    master_id: [u8; 32],
}

impl SaltV2MasterFit {
    /// Content identity of source, recipe-without-rate, evidence, and exact ordered planes.
    #[must_use]
    pub const fn master_id(&self) -> [u8; 32] {
        self.master_id
    }

    /// Fitting recipe. Its rate field is only the original request; allocation accepts a new rate.
    #[must_use]
    pub const fn config(&self) -> &SaltV2Config {
        &self.config
    }

    /// Detailed feedback evidence, absent only for the compatibility PTQ wrapper.
    #[must_use]
    pub const fn feedback_receipt(&self) -> Option<&SaltV2ModelFeedbackReceipt> {
        self.feedback.as_ref()
    }
}

/// Fit a small whole model with the deterministic CPU reference solver.
///
/// For D2/B3, every 256-coefficient allocation tile is jointly fit once at P=3 and a deterministic
/// plane order produces exact P=1/P=2 prefixes. S34 constructs and jointly refines one constrained
/// nested master frontier. Separately budgeted compact and near-lossless runs therefore slice
/// identical trits and scales instead of refitting them. Each plane retains one non-negative f16
/// scale per group128. The exact Pareto dynamic program in the shared allocator selects a
/// whole-model point under transformed integer package, artifact, and resident ceilings; the
/// canonical package writer then remeasures and verifies every component. The returned
/// representation has no zero point, bias, codebook, or floating residual.
///
/// This function intentionally implements only the pure-PTQ reference seam. S34 uses a
/// deterministic constrained CPU solver that preserves exact progressive prefixes. Signed RHT,
/// cached-output reconstruction, scale-only recovery, and short PV recovery return
/// [`SaltV2Error::ExternalStageRequired`] rather than silently falling back to a weaker algorithm.
/// The receipt reports feedback/output reconstruction as false.
///
/// # Errors
/// Rejects malformed recipes or tensors, missing/mismatched curvature evidence, unavailable
/// production stages, failed joint fits, non-monotone candidate curves, infeasible hard ceilings,
/// accounting overflow, and any canonical package validation failure.
pub fn fit_salt_v2_model(
    input: SaltV2ModelFitInput<'_>,
    config: &SaltV2Config,
) -> Result<SaltV2ModelFitResult, SaltV2Error> {
    let rate = config.rate;
    let master = fit_salt_v2_master_without_feedback(input, config)?;
    allocate_and_pack_salt_v2_master(&master, rate)
}

fn fit_salt_v2_master_without_feedback(
    input: SaltV2ModelFitInput<'_>,
    config: &SaltV2Config,
) -> Result<SaltV2MasterFit, SaltV2Error> {
    validate_config(config)?;
    validate_external_stages(config)?;
    let quantized_parameters = validate_model_input(&input, config)?;
    let mut work = Vec::new();
    work.try_reserve_exact(input.tensors.len())
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    for (tensor_index, tensor) in input.tensors.iter().enumerate() {
        let tensor_work = fit_tensor_candidates(tensor_index, tensor, config)?;
        work.push(tensor_work);
    }
    finish_master(input, *config, quantized_parameters, work, None)
}

/// Fit one content-addressed Pmax master with full BlockLDLQ feedback and delta correction.
///
/// A deterministic one-sweep provisional pass first installs every group reconstruction. The full
/// joint refit then revisits groups in natural column order through `FeedbackState::refit_suffix`;
/// every changed reconstruction is propagated with the exact residual delta before the next group
/// is refit. Allocation is deliberately deferred to [`allocate_and_pack_salt_v2_master`].
///
/// # Errors
/// Rejects all ordinary model-fit failures plus incomplete/mismatched feedback evidence, invalid
/// full inverse Hessians, non-G128-aligned natural groups, and failed provisional or final fits.
pub fn fit_salt_v2_master(
    input: SaltV2MasterFitInput<'_>,
    config: &SaltV2Config,
) -> Result<SaltV2MasterFit, SaltV2Error> {
    validate_config(config)?;
    validate_external_stages(config)?;
    let quantized_parameters = validate_model_input(&input.model, config)?;
    validate_feedback_artifacts(&input)?;

    let mut work = Vec::new();
    let mut feedback_receipts = Vec::new();
    work.try_reserve_exact(input.model.tensors.len())
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    feedback_receipts
        .try_reserve_exact(input.model.tensors.len())
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    for (tensor_index, (tensor, feedback)) in input
        .model
        .tensors
        .iter()
        .zip(input.feedback.iter())
        .enumerate()
    {
        let (tensor_work, receipt) =
            fit_tensor_candidates_with_feedback(tensor_index, tensor, *feedback, config)?;
        work.push(tensor_work);
        feedback_receipts.push(receipt);
    }
    let feedback = SaltV2ModelFeedbackReceipt {
        receipt_id: feedback_receipt_digest(&feedback_receipts),
        tensors: feedback_receipts,
    };
    finish_master(
        input.model,
        *config,
        quantized_parameters,
        work,
        Some(feedback),
    )
}

fn finish_master(
    input: SaltV2ModelFitInput<'_>,
    config: SaltV2Config,
    quantized_parameters: u64,
    work: Vec<TensorFitWork>,
    feedback: Option<SaltV2ModelFeedbackReceipt>,
) -> Result<SaltV2MasterFit, SaltV2Error> {
    let activation_digest = input.activations.digest().into_bytes();
    let master_id = master_digest(
        input.source_model_id,
        activation_digest,
        &config,
        input.physical,
        &work,
        feedback.as_ref(),
    )?;
    Ok(SaltV2MasterFit {
        work,
        config,
        quantized_parameters,
        source_model_id: input.source_model_id,
        activation_digest,
        physical: input.physical,
        feedback,
        master_id,
    })
}

/// Select exact-byte prefixes from an immutable ordered master and write a canonical package.
///
/// The supplied target may vary across calls. No candidate is refit, so compact artifacts remain
/// byte-exact prefixes of a less constrained package produced from the same master.
///
/// # Errors
/// Rejects malformed or infeasible physical ceilings, accounting mismatches, allocator failures,
/// and any canonical package validation failure.
pub fn allocate_and_pack_salt_v2_master(
    master: &SaltV2MasterFit,
    rate: PhysicalRateTarget,
) -> Result<SaltV2ModelFitResult, SaltV2Error> {
    let mut config = master.config;
    config.rate = rate;
    validate_config(&config)?;
    let work = &master.work;
    let physical = master.physical;
    let quantized_parameters = master.quantized_parameters;
    let mut tile_candidates = Vec::new();
    for tensor_work in work {
        for frontier in &tensor_work.candidates {
            tile_candidates.extend(frontier.iter().map(|candidate| candidate.metrics));
        }
    }
    let (serialized_fixed_bytes, resident_fixed_bytes) =
        fixed_package_bytes(work, config.packing.codec())?;

    let overhead_resident = physical
        .preserved_resident_bytes
        .checked_add(physical.required_runtime_shadow_bytes)
        .ok_or(SaltV2Error::AccountingOverflow)?;
    let package_artifact_ceiling = config
        .rate
        .max_artifact_bytes
        .checked_sub(physical.preserved_artifact_bytes)
        .ok_or_else(|| no_feasible(&config, 0))?;
    let package_matrix_ceiling = config.rate.max_matrix_bytes.min(package_artifact_ceiling);
    let aligned_package_ceiling = package_matrix_ceiling
        / u64::try_from(SALT_V2_PACKAGE_ALIGNMENT).map_err(|_| SaltV2Error::AccountingOverflow)?
        * u64::try_from(SALT_V2_PACKAGE_ALIGNMENT).map_err(|_| SaltV2Error::AccountingOverflow)?;
    let package_resident_ceiling = match config.rate.max_resident_bytes {
        Some(maximum) => maximum
            .checked_sub(overhead_resident)
            .ok_or_else(|| no_feasible(&config, aligned_package_ceiling))?,
        None => u64::MAX
            .checked_sub(overhead_resident)
            .ok_or(SaltV2Error::AccountingOverflow)?,
    };

    let allocator_frontiers = allocator_candidates(work, &config)?;
    let group_refs = allocator_frontiers
        .iter()
        .map(|candidates| GroupCandidates {
            candidates: candidates.as_slice(),
        })
        .collect::<Vec<_>>();
    let metadata = PhysicalBytes {
        serialized: serialized_fixed_bytes,
        resident: resident_fixed_bytes,
    };
    let maximum = PhysicalBytes {
        serialized: aligned_package_ceiling,
        resident: package_resident_ceiling,
    };
    let profile = ProfileBudget {
        maximum,
        metadata: ByteDelta::measured(metadata, metadata),
    };
    let allocation = allocate_nested_profiles(
        &group_refs,
        &NestedProfileBudgets {
            compact: profile,
            near_lossless: profile,
        },
    )
    .map_err(|error| match error {
        PhysicalAllocError::BudgetTooSmall { .. } => no_feasible(&config, aligned_package_ceiling),
        other => SaltV2Error::Allocation(other),
    })?
    .near_lossless;

    let mut selected_plane_counts = allocation.plane_counts;
    let mut frontier_index = 0usize;
    for tensor in work {
        for frontier in &tensor.candidates {
            let selected = selected_plane_counts
                .get_mut(frontier_index)
                .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
            while *selected > 1 {
                let current = frontier
                    .get(usize::from(*selected - 1))
                    .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
                let prefix = frontier
                    .get(usize::from(*selected - 2))
                    .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
                if current.metrics.hessian_error != prefix.metrics.hessian_error {
                    break;
                }
                *selected -= 1;
            }
            frontier_index += 1;
        }
    }
    let mut selected_index = 0usize;
    let mut tensors = Vec::new();
    let mut tensor_receipts = Vec::new();
    let mut plane_histogram = [0u64; 3];
    let mut hessian_error = 0.0f64;
    let mut frobenius_error = 0.0f64;
    let mut predicted_raw_serialized = serialized_fixed_bytes;
    let mut predicted_resident = resident_fixed_bytes;

    for tensor_work in work {
        let mut tiles = Vec::new();
        let mut tensor_plane_counts = Vec::new();
        for (tile_index, frontier) in tensor_work.candidates.iter().enumerate() {
            let planes = *selected_plane_counts
                .get(selected_index)
                .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
            selected_index += 1;
            let candidate = frontier
                .get(usize::from(planes - 1))
                .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
            tiles.push(candidate.tile.clone());
            tensor_plane_counts.push(planes);
            let logical = u64::try_from(tensor_work.tile_lengths[tile_index])
                .map_err(|_| SaltV2Error::AccountingOverflow)?;
            plane_histogram[usize::from(planes - 1)] = plane_histogram[usize::from(planes - 1)]
                .checked_add(logical)
                .ok_or(SaltV2Error::AccountingOverflow)?;
            hessian_error += candidate.metrics.hessian_error;
            frobenius_error += candidate.metrics.frobenius_error;
            if !(hessian_error.is_finite() && frobenius_error.is_finite()) {
                return Err(SaltV2Error::AccountingOverflow);
            }
            predicted_raw_serialized = predicted_raw_serialized
                .checked_add(candidate.metrics.cumulative.serialized)
                .ok_or(SaltV2Error::AccountingOverflow)?;
            predicted_resident = predicted_resident
                .checked_add(candidate.metrics.cumulative.resident)
                .ok_or(SaltV2Error::AccountingOverflow)?;
        }
        tensors.push(SaltV2Tensor::new(
            tensor_work.name.clone(),
            vec![
                u64::try_from(tensor_work.rows).map_err(|_| SaltV2Error::AccountingOverflow)?,
                u64::try_from(tensor_work.cols).map_err(|_| SaltV2Error::AccountingOverflow)?,
            ],
            tiles,
        )?);
        tensor_receipts.push(SaltV2TensorFitReceipt {
            name: tensor_work.name.clone(),
            source_digest: tensor_work.source_digest,
            curvature_digest: tensor_work.curvature_digest,
            plane_counts: tensor_plane_counts,
        });
    }
    if selected_index != selected_plane_counts.len() {
        return Err(SaltV2Error::PhysicalAccountingMismatch);
    }

    let package = SaltV2Package::new(config.packing.codec(), tensors.clone())?;
    let encoded = write_salt_v2_package(&package)?;
    let indexed_runtime = SaltV2IndexedRuntimeLedger::for_package(&package)?;
    let measured_raw = encoded
        .ledger
        .total_bytes
        .checked_sub(encoded.ledger.padding_bytes)
        .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
    if measured_raw != predicted_raw_serialized
        || indexed_runtime.steady_resident_bytes() != predicted_resident
        || encoded
            .ledger
            .headers_bytes
            .checked_add(encoded.ledger.transform_bytes)
            .and_then(|bytes| bytes.checked_add(encoded.ledger.maps_bytes))
            .ok_or(SaltV2Error::AccountingOverflow)?
            != serialized_fixed_bytes
    {
        return Err(SaltV2Error::PhysicalAccountingMismatch);
    }

    let artifact_bytes = encoded
        .ledger
        .total_bytes
        .checked_add(physical.preserved_artifact_bytes)
        .ok_or(SaltV2Error::AccountingOverflow)?;
    let resident_bytes = indexed_runtime
        .steady_resident_bytes()
        .checked_add(overhead_resident)
        .ok_or(SaltV2Error::AccountingOverflow)?;
    if encoded.ledger.total_bytes > config.rate.max_matrix_bytes
        || artifact_bytes > config.rate.max_artifact_bytes
        || config
            .rate
            .max_resident_bytes
            .is_some_and(|maximum| resident_bytes > maximum)
    {
        return Err(no_feasible(&config, aligned_package_ceiling));
    }

    let logical_trits = plane_histogram
        .iter()
        .enumerate()
        .try_fold(0u64, |total, (index, coefficients)| {
            total.checked_add(coefficients.checked_mul((index + 1) as u64)?)
        })
        .ok_or(SaltV2Error::AccountingOverflow)?;
    let logical_bits = logical_trits as f64 * crate::TRIT_BITS;
    let matrix_bpw = exact_bpw(encoded.ledger.total_bytes, quantized_parameters)?;
    let artifact_bpw = exact_bpw(artifact_bytes, physical.total_model_parameters)?;
    let resident_bpw = exact_bpw(resident_bytes, physical.total_model_parameters)?;
    let metrics = SaltV2ModelFitMetrics {
        quantized_parameter_count: quantized_parameters,
        plane_histogram,
        selected_plane_counts,
        tile_candidates,
        physical: SaltV2PhysicalSize {
            serialized_fixed_bytes,
            matrix_bytes: encoded.ledger.total_bytes,
            artifact_bytes,
            resident_bytes,
            peak_resident_bytes: None,
            padding_bytes: encoded.ledger.padding_bytes,
            trit_payload_bytes: encoded.ledger.payload_bytes,
            scale_bytes: encoded.ledger.scales_bytes,
            allocation_map_bytes: encoded.ledger.maps_bytes,
            allocation_map_bits: encoded.ledger.allocation_map_bits,
            allocation_map_embedded_bits: encoded.ledger.allocation_map_embedded_bits,
            header_bytes: encoded.ledger.headers_bytes,
            transform_bytes: encoded.ledger.transform_bytes,
            runtime_map_bytes: indexed_runtime.allocation_map_bytes(),
            runtime_rank_prefix_bytes: indexed_runtime.rank_prefix_bytes(),
            runtime_dense_shadow_bytes: indexed_runtime.dense_shadow_bytes(),
            preserved_artifact_bytes: physical.preserved_artifact_bytes,
            preserved_resident_bytes: physical.preserved_resident_bytes,
            required_runtime_shadow_bytes: physical.required_runtime_shadow_bytes,
        },
        logical_trits,
        logical_bits,
        logical_bpw: logical_bits / quantized_parameters as f64,
        matrix_bpw,
        artifact_bpw,
        resident_bpw,
        frobenius_error,
        hessian_error,
        block_output_error: None,
        teacher_kl: None,
    };
    let package_id = PackageId::from_package_bytes(&encoded.bytes);
    let recipe_id = recipe_digest(&config);
    let receipt_id = receipt_digest(ReceiptDigestInput {
        source_model_id: master.source_model_id,
        activation_digest: master.activation_digest,
        recipe_id,
        master_id: master.master_id,
        tensors: &tensor_receipts,
        package_id,
        physical,
        feedback: master.feedback.as_ref(),
    });
    let receipt = SaltV2ModelFitReceipt {
        solver_version: REFERENCE_SOLVER_VERSION,
        source_model_id: master.source_model_id,
        activation_digest: master.activation_digest,
        recipe_id,
        master_id: master.master_id,
        tensors: tensor_receipts,
        package_id,
        physical,
        receipt_id,
        track: SaltV2FitTrack::Ptq,
        feedback_applied: master.feedback.is_some(),
        feedback: master.feedback.clone(),
        output_reconstruction_applied: false,
    };

    Ok(SaltV2ModelFitResult {
        tensors,
        package_bytes: encoded.bytes,
        config,
        metrics,
        receipt,
    })
}

fn validate_config(config: &SaltV2Config) -> Result<(), SaltV2Error> {
    if config.group_size != SALT_V2_SCALE_GROUP_SIZE {
        return Err(SaltV2Error::UnsupportedReferenceGroupSize {
            got: config.group_size,
        });
    }
    if config.min_planes != 1 {
        return Err(SaltV2Error::UnsupportedMinimumPlanes {
            got: config.min_planes,
        });
    }
    if !(2..=3).contains(&config.max_planes) {
        return Err(SaltV2Error::InvalidMaximumPlanes {
            got: config.max_planes,
        });
    }
    if config.em_restarts == 0 {
        return Err(SaltV2Error::InvalidEmRestarts);
    }
    if config.coordinate_sweeps == 0 {
        return Err(SaltV2Error::InvalidCoordinateSweeps);
    }
    if !config.ridge_condition_limit.is_finite() || config.ridge_condition_limit <= 1.0 {
        return Err(SaltV2Error::InvalidRidgeConditionLimit);
    }
    if config.rate.max_matrix_bytes == 0
        || config.rate.max_artifact_bytes == 0
        || config.rate.max_resident_bytes == Some(0)
    {
        return Err(SaltV2Error::InvalidPhysicalRateTarget);
    }
    match config.refinement {
        SaltV2Refinement::None => {}
        SaltV2Refinement::ScaleOnly { max_tokens: 0 } => {
            return Err(SaltV2Error::InvalidRefinementSchedule);
        }
        SaltV2Refinement::PvKl {
            warmup_tokens,
            hard_tokens,
        } => {
            let total = u128::from(warmup_tokens) + u128::from(hard_tokens);
            if warmup_tokens == 0 || hard_tokens == 0 || u128::from(hard_tokens) * 5 < total {
                return Err(SaltV2Error::InvalidRefinementSchedule);
            }
        }
        SaltV2Refinement::ScaleOnly { .. } => {}
    }
    Ok(())
}

fn validate_external_stages(config: &SaltV2Config) -> Result<(), SaltV2Error> {
    if config.transform_seed.is_some() {
        return Err(SaltV2Error::ExternalStageRequired {
            stage: SaltV2ExternalStage::SignedRht,
        });
    }
    match config.refinement {
        SaltV2Refinement::None => Ok(()),
        SaltV2Refinement::ScaleOnly { .. } => Err(SaltV2Error::ExternalStageRequired {
            stage: SaltV2ExternalStage::ScaleOnlyRefinement,
        }),
        SaltV2Refinement::PvKl { .. } => Err(SaltV2Error::ExternalStageRequired {
            stage: SaltV2ExternalStage::PvKlRefinement,
        }),
    }
}

fn validate_model_input(
    input: &SaltV2ModelFitInput<'_>,
    config: &SaltV2Config,
) -> Result<u64, SaltV2Error> {
    if input.tensors.is_empty() {
        return Err(SaltV2Error::EmptyModel);
    }
    let source_model_digest = *input.source_model_id.as_bytes();
    let activation_cache_digest = input.activations.digest().into_bytes();
    // ActivationCacheSpec::source_digest is the canonical calibration-provenance envelope. Its
    // contract includes the ordered token stream, tokenizer, corpus revision, and sampling seed;
    // the cache digest above separately binds masks and boundaries. The solve boundary treats the
    // exact source envelope as its token-stream evidence.
    let token_stream_digest = input.activations.spec().source_digest().into_bytes();
    let mut names = BTreeSet::new();
    let mut quantized = 0u64;
    for (tensor_index, tensor) in input.tensors.iter().enumerate() {
        if tensor.name.is_empty() {
            return Err(SaltV2Error::EmptyTensorName {
                tensor: tensor_index,
            });
        }
        if !names.insert(tensor.name) {
            return Err(SaltV2Error::DuplicateTensorName(tensor.name.to_owned()));
        }
        let expected = tensor
            .rows
            .checked_mul(tensor.cols)
            .filter(|_| tensor.rows > 0 && tensor.cols > 0)
            .ok_or(SaltV2Error::InvalidTensorShape {
                tensor: tensor_index,
            })?;
        if expected != tensor.weights.len() {
            return Err(SaltV2Error::TensorLengthMismatch {
                tensor: tensor_index,
                expected,
                got: tensor.weights.len(),
            });
        }
        if let Some(index) = tensor.weights.iter().position(|weight| !weight.is_finite()) {
            return Err(SaltV2Error::NonFiniteWeight {
                tensor: tensor_index,
                index,
            });
        }
        let curvature_source = tensor.curvature.source_id();
        if curvature_source.source_model_digest() != source_model_digest {
            return Err(SaltV2Error::CurvatureSourceModelMismatch {
                tensor: tensor_index,
            });
        }
        if curvature_source.activation_cache_digest() != activation_cache_digest {
            return Err(SaltV2Error::CurvatureActivationCacheMismatch {
                tensor: tensor_index,
            });
        }
        if curvature_source.token_stream_digest() != token_stream_digest {
            return Err(SaltV2Error::CurvatureTokenStreamMismatch {
                tensor: tensor_index,
            });
        }
        if tensor.curvature.kind != config.curvature {
            return Err(SaltV2Error::CurvatureKindMismatch {
                tensor: tensor_index,
                expected: config.curvature,
                got: tensor.curvature.kind,
            });
        }
        validate_curvature_geometry(tensor_index, tensor)?;
        quantized = quantized
            .checked_add(u64::try_from(expected).map_err(|_| SaltV2Error::AccountingOverflow)?)
            .ok_or(SaltV2Error::AccountingOverflow)?;
    }
    if input.physical.total_model_parameters < quantized
        || input.physical.total_model_parameters == 0
    {
        return Err(SaltV2Error::InvalidTotalModelParameters);
    }
    Ok(quantized)
}

fn validate_feedback_artifacts(input: &SaltV2MasterFitInput<'_>) -> Result<(), SaltV2Error> {
    if input.feedback.len() != input.model.tensors.len() {
        return Err(SaltV2Error::FeedbackArtifactCountMismatch {
            expected: input.model.tensors.len(),
            got: input.feedback.len(),
        });
    }
    let source_model_digest = *input.model.source_model_id.as_bytes();
    let activation_cache_digest = input.model.activations.digest().into_bytes();
    let token_stream_digest = input.model.activations.spec().source_digest().into_bytes();
    for (tensor_index, (tensor, feedback)) in input
        .model
        .tensors
        .iter()
        .zip(input.feedback.iter())
        .enumerate()
    {
        let source = feedback.source_id();
        if source.source_model_digest() != source_model_digest {
            return Err(SaltV2Error::FeedbackSourceModelMismatch {
                tensor: tensor_index,
            });
        }
        if source.activation_cache_digest() != activation_cache_digest {
            return Err(SaltV2Error::FeedbackActivationCacheMismatch {
                tensor: tensor_index,
            });
        }
        if source.token_stream_digest() != token_stream_digest {
            return Err(SaltV2Error::FeedbackTokenStreamMismatch {
                tensor: tensor_index,
            });
        }
        let aligned_partition = tensor.cols.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE)
            && !feedback.groups().is_empty()
            && feedback.groups().iter().all(|group| {
                group.start.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE)
                    && group.end.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE)
            });
        if !aligned_partition {
            return Err(SaltV2Error::FeedbackScaleGeometry {
                tensor: tensor_index,
            });
        }
        // `fit_with_feedback` validates exact contiguity, full 0..cols coverage, dense metric
        // geometry, finiteness, symmetry, and positive definiteness before its first callback.
        // Keep that canonical validation in one place; this layer adds deployment-scale alignment.
    }
    Ok(())
}

fn validate_curvature_geometry(
    tensor_index: usize,
    tensor: &SaltV2TensorFitInput<'_>,
) -> Result<(), SaltV2Error> {
    match tensor.curvature.values {
        CurvatureValues::Diagonal(diagonal) => {
            if diagonal.len() != tensor.weights.len()
                || diagonal
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(SaltV2Error::CurvatureGeometry {
                    tensor: tensor_index,
                });
            }
        }
        CurvatureValues::DenseGroups(groups) => {
            let expected_groups = tensor.weights.len().div_ceil(SALT_V2_SCALE_GROUP_SIZE);
            if groups.len() != expected_groups {
                return Err(SaltV2Error::CurvatureGeometry {
                    tensor: tensor_index,
                });
            }
            for (group_index, group) in groups.iter().enumerate() {
                let start = group_index
                    .checked_mul(SALT_V2_SCALE_GROUP_SIZE)
                    .ok_or(SaltV2Error::AccountingOverflow)?;
                let expected = (tensor.weights.len() - start).min(SALT_V2_SCALE_GROUP_SIZE);
                if group.dimension() != expected {
                    return Err(SaltV2Error::CurvatureGeometry {
                        tensor: tensor_index,
                    });
                }
            }
        }
    }
    Ok(())
}

fn fixed_package_bytes(
    work: &[TensorFitWork],
    codec: SaltV2Codec,
) -> Result<(u64, u64), SaltV2Error> {
    let mut tensors = Vec::with_capacity(work.len());
    let mut base_plane_serialized_bytes = 0u64;
    let mut base_plane_resident_bytes = 0u64;
    for tensor in work {
        let mut tiles = Vec::with_capacity(tensor.candidates.len());
        for frontier in &tensor.candidates {
            let base = frontier
                .first()
                .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
            base_plane_serialized_bytes = base_plane_serialized_bytes
                .checked_add(base.metrics.cumulative.serialized)
                .ok_or(SaltV2Error::AccountingOverflow)?;
            base_plane_resident_bytes = base_plane_resident_bytes
                .checked_add(base.metrics.cumulative.resident)
                .ok_or(SaltV2Error::AccountingOverflow)?;
            tiles.push(base.tile.clone());
        }
        tensors.push(SaltV2Tensor::new(
            tensor.name.clone(),
            vec![
                u64::try_from(tensor.rows).map_err(|_| SaltV2Error::AccountingOverflow)?,
                u64::try_from(tensor.cols).map_err(|_| SaltV2Error::AccountingOverflow)?,
            ],
            tiles,
        )?);
    }
    let package = SaltV2Package::new(codec, tensors)?;
    let encoded = write_salt_v2_package(&package)?;
    let raw_serialized = encoded
        .ledger
        .total_bytes
        .checked_sub(encoded.ledger.padding_bytes)
        .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
    let serialized_fixed = raw_serialized
        .checked_sub(base_plane_serialized_bytes)
        .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
    let resident_fixed = SaltV2IndexedRuntimeLedger::for_package(&package)?
        .steady_resident_bytes()
        .checked_sub(base_plane_resident_bytes)
        .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
    Ok((serialized_fixed, resident_fixed))
}

#[derive(Clone, Debug)]
struct MasterTensorPlanes {
    trits: Vec<Vec<i8>>,
    scales: Vec<Vec<f16>>,
    populated_groups: Vec<bool>,
}

impl MasterTensorPlanes {
    fn new(coefficient_count: usize) -> Self {
        let scale_count = coefficient_count / SALT_V2_SCALE_GROUP_SIZE;
        Self {
            trits: (0..3).map(|_| vec![0; coefficient_count]).collect(),
            scales: (0..3)
                .map(|_| vec![f16::from_f32(0.0); scale_count])
                .collect(),
            populated_groups: vec![false; scale_count],
        }
    }
}

#[derive(Clone, Debug)]
struct FeedbackPlanePlacement {
    global_start: usize,
    trits: [Vec<i8>; 3],
    scales: [f16; 3],
}

#[derive(Clone, Debug)]
struct FeedbackGroupFit {
    reconstruction: Vec<f64>,
    placements: Vec<FeedbackPlanePlacement>,
}

#[derive(Clone, Copy, Debug)]
struct FeedbackPassRecord {
    column_start: usize,
    column_end: usize,
    fit_input_digest: [u8; 32],
    reconstruction_digest: [u8; 32],
}

fn fit_tensor_candidates_with_feedback(
    tensor_index: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    feedback: SaltV2FeedbackArtifact<'_>,
    config: &SaltV2Config,
) -> Result<(TensorFitWork, SaltV2TensorFeedbackReceipt), SaltV2Error> {
    let weights = tensor
        .weights
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let problem = FeedbackProblem {
        rows: tensor.rows,
        columns: tensor.cols,
        weights: &weights,
        groups: feedback.groups(),
        metric: FeedbackMetric::InverseHessian(feedback.values()),
    };

    let mut provisional_records = Vec::with_capacity(feedback.groups().len());
    let mut state = fit_with_feedback(problem, |request| {
        let fitted = fit_feedback_group(tensor_index, tensor, config, request, true)?;
        provisional_records.push(feedback_pass_record(request, &fitted.reconstruction));
        Ok::<_, SaltV2Error>(fitted.reconstruction)
    })
    .map_err(|error| map_feedback_run_error(tensor_index, error))?;

    let mut master_planes = MasterTensorPlanes::new(tensor.weights.len());
    let mut final_records = Vec::with_capacity(feedback.groups().len());
    state
        .refit_suffix(0, |request| {
            let fitted = fit_feedback_group(tensor_index, tensor, config, request, false)?;
            install_feedback_group(&mut master_planes, &fitted)?;
            final_records.push(feedback_pass_record(request, &fitted.reconstruction));
            Ok::<_, SaltV2Error>(fitted.reconstruction)
        })
        .map_err(|error| map_feedback_run_error(tensor_index, error))?;

    if provisional_records.len() != feedback.groups().len()
        || final_records.len() != feedback.groups().len()
        || master_planes
            .populated_groups
            .iter()
            .any(|populated| !*populated)
    {
        return Err(SaltV2Error::PhysicalAccountingMismatch);
    }
    let mut group_receipts = Vec::with_capacity(feedback.groups().len());
    let mut nonzero_delta_corrections = 0u64;
    for (group_index, ((group, provisional), final_record)) in feedback
        .groups()
        .iter()
        .zip(provisional_records.iter())
        .zip(final_records.iter())
        .enumerate()
    {
        if provisional.column_start != group.start
            || provisional.column_end != group.end
            || final_record.column_start != group.start
            || final_record.column_end != group.end
        {
            return Err(SaltV2Error::PhysicalAccountingMismatch);
        }
        let nonzero_delta = provisional.reconstruction_digest != final_record.reconstruction_digest;
        nonzero_delta_corrections = nonzero_delta_corrections
            .checked_add(u64::from(nonzero_delta))
            .ok_or(SaltV2Error::AccountingOverflow)?;
        group_receipts.push(SaltV2FeedbackGroupReceipt {
            group_index,
            column_start: group.start,
            column_end: group.end,
            provisional_fit_input_digest: provisional.fit_input_digest,
            provisional_reconstruction_digest: provisional.reconstruction_digest,
            final_fit_input_digest: final_record.fit_input_digest,
            final_reconstruction_digest: final_record.reconstruction_digest,
            nonzero_delta,
        });
    }
    let receipt = SaltV2TensorFeedbackReceipt {
        name: tensor.name.to_owned(),
        artifact_digest: feedback.digest(),
        delta_corrections: u64::try_from(group_receipts.len())
            .map_err(|_| SaltV2Error::AccountingOverflow)?,
        nonzero_delta_corrections,
        groups: group_receipts,
        final_working_digest: feedback_values_digest(3, state.working_weights()),
        final_reconstruction_digest: feedback_values_digest(4, state.reconstruction()),
    };
    let work = materialize_feedback_tensor_work(tensor_index, tensor, config, master_planes)?;
    Ok((work, receipt))
}

fn map_feedback_run_error(tensor: usize, error: FeedbackRunError<SaltV2Error>) -> SaltV2Error {
    match error {
        FeedbackRunError::Feedback(source) => SaltV2Error::Feedback { tensor, source },
        FeedbackRunError::Fitter(source) => source,
    }
}

fn feedback_pass_record(
    request: GroupFitRequest<'_>,
    reconstruction: &[f64],
) -> FeedbackPassRecord {
    FeedbackPassRecord {
        column_start: request.column_start,
        column_end: request.column_start + request.columns,
        fit_input_digest: feedback_values_digest(1, request.working_weights),
        reconstruction_digest: feedback_values_digest(2, reconstruction),
    }
}

fn fit_feedback_group(
    tensor_index: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    config: &SaltV2Config,
    request: GroupFitRequest<'_>,
    provisional: bool,
) -> Result<FeedbackGroupFit, SaltV2Error> {
    // Search always constructs the complete Pmax=3 master. `config.max_planes` limits the
    // allocation frontier later; consulting it here would make compact and near-lossless refit
    // different masters and violate the byte-prefix contract.
    const FULL_PLANES: usize = 3;
    let mut reconstruction = vec![0.0; request.working_weights.len()];
    let groups_per_row = request.columns / SALT_V2_SCALE_GROUP_SIZE;
    let mut placements = Vec::with_capacity(request.rows * groups_per_row);
    for row in 0..request.rows {
        for local_group in 0..groups_per_row {
            let local_start = local_group * SALT_V2_SCALE_GROUP_SIZE;
            let compact_start = row * request.columns + local_start;
            let compact_end = compact_start + SALT_V2_SCALE_GROUP_SIZE;
            let global_start = row * tensor.cols + request.column_start + local_start;
            let global_end = global_start + SALT_V2_SCALE_GROUP_SIZE;
            let working = request.working_weights[compact_start..compact_end]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            if working.iter().any(|value| !value.is_finite()) {
                return Err(SaltV2Error::AccountingOverflow);
            }
            let global_group = global_start / SALT_V2_SCALE_GROUP_SIZE;
            let tile_index = global_start / SALT_V2_ALLOCATION_TILE_SIZE;
            let metric = curvature_metric(tensor.curvature, global_start, global_end, global_group);
            let fit_config = JointFitConfig {
                planes: FULL_PLANES,
                max_iterations: if provisional {
                    1
                } else {
                    config.coordinate_sweeps
                },
                ridge: 1e-12,
                em_restarts: if provisional { 1 } else { config.em_restarts },
                ridge_condition_limit: config.ridge_condition_limit,
                scale_precision: ScalePrecision::F16,
            };
            let (scales, trits, order) = if config.packing == SaltV2Packing::S34 {
                let fitted =
                    fit_progressive_s34(&working, metric, fit_config).map_err(|source| {
                        SaltV2Error::JointFit {
                            tensor: tensor_index,
                            tile: tile_index,
                            group: global_group,
                            planes: FULL_PLANES,
                            source,
                        }
                    })?;
                (fitted.scales, fitted.trits, [0, 1, 2])
            } else {
                let fitted = fit_joint_ternary(&working, metric, fit_config).map_err(|source| {
                    SaltV2Error::JointFit {
                        tensor: tensor_index,
                        tile: tile_index,
                        group: global_group,
                        planes: FULL_PLANES,
                        source,
                    }
                })?;
                let order = progressive_plane_order(
                    &tensor.weights[global_start..global_end],
                    metric,
                    &fitted.scales,
                    &fitted.trits,
                )
                .map_err(|planes| SaltV2Error::NonMonotoneCandidate {
                    tensor: tensor_index,
                    tile: tile_index,
                    planes,
                })?;
                (fitted.scales, fitted.trits, order)
            };
            for local in 0..SALT_V2_SCALE_GROUP_SIZE {
                reconstruction[compact_start + local] = (0..FULL_PLANES)
                    .map(|plane| f64::from(scales[plane]) * f64::from(trits[plane][local]))
                    .sum();
            }
            placements.push(FeedbackPlanePlacement {
                global_start,
                trits: std::array::from_fn(|plane| trits[order[plane]].clone()),
                scales: std::array::from_fn(|plane| f16::from_f32(scales[order[plane]])),
            });
        }
    }
    Ok(FeedbackGroupFit {
        reconstruction,
        placements,
    })
}

fn install_feedback_group(
    master: &mut MasterTensorPlanes,
    fitted: &FeedbackGroupFit,
) -> Result<(), SaltV2Error> {
    for placement in &fitted.placements {
        let group = placement.global_start / SALT_V2_SCALE_GROUP_SIZE;
        let end = placement
            .global_start
            .checked_add(SALT_V2_SCALE_GROUP_SIZE)
            .ok_or(SaltV2Error::AccountingOverflow)?;
        let populated = master
            .populated_groups
            .get_mut(group)
            .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
        if *populated {
            return Err(SaltV2Error::PhysicalAccountingMismatch);
        }
        for plane in 0..3 {
            master.trits[plane][placement.global_start..end]
                .copy_from_slice(&placement.trits[plane]);
            master.scales[plane][group] = placement.scales[plane];
        }
        *populated = true;
    }
    Ok(())
}

fn materialize_feedback_tensor_work(
    tensor_index: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    config: &SaltV2Config,
    master: MasterTensorPlanes,
) -> Result<TensorFitWork, SaltV2Error> {
    let tile_count = tensor.weights.len().div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
    let mut candidates = Vec::with_capacity(tile_count);
    let mut tile_lengths = Vec::with_capacity(tile_count);
    for tile_index in 0..tile_count {
        let start = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2Error::AccountingOverflow)?;
        let end = (start + SALT_V2_ALLOCATION_TILE_SIZE).min(tensor.weights.len());
        let tile_len = end - start;
        tile_lengths.push(tile_len);
        candidates.push(materialize_feedback_tile_frontier(
            tensor_index,
            tile_index,
            start,
            end,
            tensor,
            config,
            &master,
            plane_physical_bytes(config.packing, tile_len)?,
        )?);
    }
    Ok(TensorFitWork {
        name: tensor.name.to_owned(),
        rows: tensor.rows,
        cols: tensor.cols,
        tile_lengths,
        candidates,
        source_digest: source_tensor_digest(tensor),
        curvature_digest: tensor.curvature.digest(),
    })
}

#[allow(clippy::too_many_arguments)]
fn materialize_feedback_tile_frontier(
    tensor_index: usize,
    tile_index: usize,
    tile_start: usize,
    tile_end: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    config: &SaltV2Config,
    master: &MasterTensorPlanes,
    per_plane_bytes: PhysicalBytes,
) -> Result<Vec<TileFitCandidate>, SaltV2Error> {
    const FULL_PLANES: usize = 3;
    let first_scale = tile_start / SALT_V2_SCALE_GROUP_SIZE;
    let scale_end = tile_end.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
    let semantic_planes = (0..FULL_PLANES)
        .map(|plane| {
            SaltV2Plane::new(
                master.trits[plane][tile_start..tile_end].to_vec(),
                master.scales[plane][first_scale..scale_end].to_vec(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut hessian_errors = [0.0; FULL_PLANES];
    let mut frobenius_errors = [0.0; FULL_PLANES];
    let mut group_start = tile_start;
    while group_start < tile_end {
        let group_end = (group_start + SALT_V2_SCALE_GROUP_SIZE).min(tile_end);
        let group = group_start / SALT_V2_SCALE_GROUP_SIZE;
        let metric = curvature_metric(tensor.curvature, group_start, group_end, group);
        let source = &tensor.weights[group_start..group_end];
        let mut reconstruction = vec![0.0f32; source.len()];
        for plane in 0..FULL_PLANES {
            let scale = master.scales[plane][group].to_f32();
            for (local, value) in reconstruction.iter_mut().enumerate() {
                *value += scale * f32::from(master.trits[plane][group_start + local]);
            }
            hessian_errors[plane] += if config.packing == SaltV2Packing::S34 {
                checked_s34_objective(source, &reconstruction, metric).map_err(|source| {
                    SaltV2Error::JointFit {
                        tensor: tensor_index,
                        tile: tile_index,
                        group,
                        planes: plane + 1,
                        source,
                    }
                })?
            } else {
                reconstruction_objective(source, &reconstruction, metric)
            };
            frobenius_errors[plane] += source
                .iter()
                .zip(reconstruction.iter())
                .map(|(left, right)| {
                    let residual = f64::from(*left) - f64::from(*right);
                    residual * residual
                })
                .sum::<f64>();
        }
        group_start = group_end;
    }

    let mut frontier = Vec::with_capacity(config.max_planes);
    for planes in 1..=config.max_planes {
        if planes > 1 {
            let prior = hessian_errors[planes - 2];
            if config.packing == SaltV2Packing::S34 && hessian_errors[planes - 1] >= prior {
                break;
            }
            let tolerance = 1e-12f64.max(prior.abs() * 1e-12);
            if hessian_errors[planes - 1] > prior + tolerance {
                return Err(SaltV2Error::NonMonotoneCandidate {
                    tensor: tensor_index,
                    tile: tile_index,
                    planes,
                });
            }
        }
        let cumulative = PhysicalBytes {
            serialized: per_plane_bytes
                .serialized
                .checked_mul(planes as u64)
                .ok_or(SaltV2Error::AccountingOverflow)?,
            resident: per_plane_bytes
                .resident
                .checked_mul(planes as u64)
                .ok_or(SaltV2Error::AccountingOverflow)?,
        };
        frontier.push(TileFitCandidate {
            tile: SaltV2Tile::new(semantic_planes[..planes].to_vec())?,
            metrics: SaltV2TileCandidateMetrics {
                tensor_index,
                tile_index,
                planes: planes as u8,
                cumulative,
                hessian_error: hessian_errors[planes - 1],
                frobenius_error: frobenius_errors[planes - 1],
            },
        });
    }
    Ok(frontier)
}

fn fit_tensor_candidates(
    tensor_index: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    config: &SaltV2Config,
) -> Result<TensorFitWork, SaltV2Error> {
    let tile_count = tensor.weights.len().div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
    let mut candidates = Vec::new();
    let mut tile_lengths = Vec::new();
    candidates
        .try_reserve_exact(tile_count)
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    tile_lengths
        .try_reserve_exact(tile_count)
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    for tile_index in 0..tile_count {
        let start = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2Error::AccountingOverflow)?;
        let end = (start + SALT_V2_ALLOCATION_TILE_SIZE).min(tensor.weights.len());
        let tile_len = end - start;
        tile_lengths.push(tile_len);
        candidates.push(fit_tile_frontier(
            tensor_index,
            tile_index,
            start,
            end,
            tensor,
            config,
            plane_physical_bytes(config.packing, tile_len)?,
        )?);
    }
    Ok(TensorFitWork {
        name: tensor.name.to_owned(),
        rows: tensor.rows,
        cols: tensor.cols,
        tile_lengths,
        candidates,
        source_digest: source_tensor_digest(tensor),
        curvature_digest: tensor.curvature.digest(),
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_tile_frontier(
    tensor_index: usize,
    tile_index: usize,
    tile_start: usize,
    tile_end: usize,
    tensor: &SaltV2TensorFitInput<'_>,
    config: &SaltV2Config,
    per_plane_bytes: PhysicalBytes,
) -> Result<Vec<TileFitCandidate>, SaltV2Error> {
    const FULL_PLANES: usize = 3;
    let tile_len = tile_end - tile_start;
    let mut plane_trits = (0..FULL_PLANES)
        .map(|_| Vec::with_capacity(tile_len))
        .collect::<Vec<_>>();
    let mut plane_scales = (0..FULL_PLANES)
        .map(|_| Vec::with_capacity(tile_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE)))
        .collect::<Vec<_>>();
    let mut hessian_errors = [0.0f64; FULL_PLANES];
    let mut frobenius_errors = [0.0f64; FULL_PLANES];
    let mut group_start = tile_start;
    while group_start < tile_end {
        let group_end = (group_start + SALT_V2_SCALE_GROUP_SIZE).min(tile_end);
        let group_index = group_start / SALT_V2_SCALE_GROUP_SIZE;
        let metric = curvature_metric(tensor.curvature, group_start, group_end, group_index);
        let weights = &tensor.weights[group_start..group_end];
        let (scales, trits, order) = if config.packing == SaltV2Packing::S34 {
            let fitted = fit_progressive_s34(
                weights,
                metric,
                JointFitConfig {
                    planes: FULL_PLANES,
                    max_iterations: config.coordinate_sweeps,
                    ridge: 1e-12,
                    em_restarts: config.em_restarts,
                    ridge_condition_limit: config.ridge_condition_limit,
                    scale_precision: ScalePrecision::F16,
                },
            )
            .map_err(|source| SaltV2Error::JointFit {
                tensor: tensor_index,
                tile: tile_index,
                group: group_index,
                planes: FULL_PLANES,
                source,
            })?;
            (fitted.scales, fitted.trits, [0, 1, 2])
        } else {
            let fitted = fit_joint_ternary(
                weights,
                metric,
                JointFitConfig {
                    planes: FULL_PLANES,
                    max_iterations: config.coordinate_sweeps,
                    ridge: 1e-12,
                    em_restarts: config.em_restarts,
                    ridge_condition_limit: config.ridge_condition_limit,
                    scale_precision: ScalePrecision::F16,
                },
            )
            .map_err(|source| SaltV2Error::JointFit {
                tensor: tensor_index,
                tile: tile_index,
                group: group_index,
                planes: FULL_PLANES,
                source,
            })?;
            let order = progressive_plane_order(weights, metric, &fitted.scales, &fitted.trits)
                .map_err(|planes| SaltV2Error::NonMonotoneCandidate {
                    tensor: tensor_index,
                    tile: tile_index,
                    planes,
                })?;
            (fitted.scales, fitted.trits, order)
        };
        let mut reconstruction = vec![0.0f32; weights.len()];
        for prefix in 0..FULL_PLANES {
            let source_plane = order[prefix];
            let scale = scales[source_plane];
            for (value, trit) in reconstruction.iter_mut().zip(&trits[source_plane]) {
                *value += scale * f32::from(*trit);
            }
            hessian_errors[prefix] += if config.packing == SaltV2Packing::S34 {
                checked_s34_objective(weights, &reconstruction, metric).map_err(|source| {
                    SaltV2Error::JointFit {
                        tensor: tensor_index,
                        tile: tile_index,
                        group: group_index,
                        planes: prefix + 1,
                        source,
                    }
                })?
            } else {
                reconstruction_objective(weights, &reconstruction, metric)
            };
            frobenius_errors[prefix] += weights
                .iter()
                .zip(&reconstruction)
                .map(|(source, reconstructed)| {
                    let residual = f64::from(*source) - f64::from(*reconstructed);
                    residual * residual
                })
                .sum::<f64>();
            plane_trits[prefix].extend_from_slice(&trits[source_plane]);
            plane_scales[prefix].push(f16::from_f32(scale));
        }
        group_start = group_end;
    }
    let semantic_planes = plane_trits
        .into_iter()
        .zip(plane_scales)
        .map(|(trits, scales)| SaltV2Plane::new(trits, scales))
        .collect::<Result<Vec<_>, _>>()?;
    let mut frontier = Vec::with_capacity(config.max_planes);
    for planes in 1..=config.max_planes {
        if planes > 1 {
            let prior = hessian_errors[planes - 2];
            if config.packing == SaltV2Packing::S34 && hessian_errors[planes - 1] >= prior {
                break;
            }
            let tolerance = 1e-12f64.max(prior.abs() * 1e-12);
            if hessian_errors[planes - 1] > prior + tolerance {
                return Err(SaltV2Error::NonMonotoneCandidate {
                    tensor: tensor_index,
                    tile: tile_index,
                    planes,
                });
            }
        }
        let cumulative = PhysicalBytes {
            serialized: per_plane_bytes
                .serialized
                .checked_mul(planes as u64)
                .ok_or(SaltV2Error::AccountingOverflow)?,
            resident: per_plane_bytes
                .resident
                .checked_mul(planes as u64)
                .ok_or(SaltV2Error::AccountingOverflow)?,
        };
        frontier.push(TileFitCandidate {
            tile: SaltV2Tile::new(semantic_planes[..planes].to_vec())?,
            metrics: SaltV2TileCandidateMetrics {
                tensor_index,
                tile_index,
                planes: planes as u8,
                cumulative,
                hessian_error: hessian_errors[planes - 1],
                frobenius_error: frobenius_errors[planes - 1],
            },
        });
    }
    Ok(frontier)
}

fn progressive_plane_order(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    scales: &[f32],
    trits: &[Vec<i8>],
) -> Result<[usize; 3], usize> {
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = None::<([usize; 3], [f64; 3])>;
    let mut has_monotone_two_plane_prefix = false;
    for order in ORDERS {
        let mut reconstruction = vec![0.0f32; weights.len()];
        let mut objectives = [0.0f64; 3];
        for (prefix, &plane) in order.iter().enumerate() {
            for (value, trit) in reconstruction.iter_mut().zip(&trits[plane]) {
                *value += scales[plane] * f32::from(*trit);
            }
            objectives[prefix] = reconstruction_objective(weights, &reconstruction, metric);
        }
        let monotone_two_plane_prefix = {
            let pair = &objectives[..2];
            let tolerance = 1e-12f64.max(pair[0].abs() * 1e-12);
            pair[1] <= pair[0] + tolerance
        };
        has_monotone_two_plane_prefix |= monotone_two_plane_prefix;
        let monotone = monotone_two_plane_prefix && {
            let pair = &objectives[1..];
            let tolerance = 1e-12f64.max(pair[0].abs() * 1e-12);
            pair[1] <= pair[0] + tolerance
        };
        if !monotone {
            continue;
        }
        if best.as_ref().is_none_or(|(best_order, best_objectives)| {
            objectives[0]
                .total_cmp(&best_objectives[0])
                .then_with(|| objectives[1].total_cmp(&best_objectives[1]))
                .then_with(|| order.cmp(best_order))
                .is_lt()
        }) {
            best = Some((order, objectives));
        }
    }
    best.map(|(order, _)| order)
        .ok_or(if has_monotone_two_plane_prefix { 3 } else { 2 })
}

#[derive(Clone, Debug)]
struct ProgressiveS34Fit {
    scales: Vec<f32>,
    trits: Vec<Vec<i8>>,
}

#[derive(Clone, Debug)]
struct S34PlaneFit {
    scale: f32,
    trits: Vec<i8>,
    objective: f64,
}

/// Fit structured planes in their deployment order so every returned P1/P2/P3 point is an exact
/// prefix. A sequential constrained fit establishes a monotone nested frontier, then joint
/// plane-coordinate refinement accepts only updates that do not worsen any affected prefix. S34
/// cannot pair a zero scale with its mandatory nonzero trits, so exhausted suffixes use the least
/// positive f16 scale and the tile frontier stops before any non-improving suffix.
fn fit_progressive_s34(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    config: JointFitConfig,
) -> Result<ProgressiveS34Fit, JointFitError> {
    let mut reconstruction = vec![0.0f32; weights.len()];
    let mut scales = Vec::with_capacity(config.planes);
    let mut trits = Vec::with_capacity(config.planes);

    for _ in 0..config.planes {
        let target = weights
            .iter()
            .zip(&reconstruction)
            .map(|(weight, reconstructed)| *weight - *reconstructed)
            .collect::<Vec<_>>();
        // The mature unconstrained P1 solver supplies a deterministic scale basin. The S34
        // assignment below never reuses its unconstrained trits.
        let seed = fit_joint_ternary(
            &target,
            metric,
            JointFitConfig {
                planes: 1,
                ..config
            },
        )?;
        let fitted = fit_s34_residual_plane(
            &target,
            metric,
            seed.scales[0],
            config.max_iterations,
            config.em_restarts,
        )?;
        let prior_objective = checked_s34_objective(weights, &reconstruction, metric)?;
        let mut candidate = reconstruction.clone();
        for (value, trit) in candidate.iter_mut().zip(&fitted.trits) {
            *value += fitted.scale * f32::from(*trit);
        }
        let candidate_objective = checked_s34_objective(weights, &candidate, metric)?;
        let tolerance = 1e-12f64.max(prior_objective.abs() * 1e-12);
        if candidate_objective <= prior_objective + tolerance {
            reconstruction = candidate;
            scales.push(fitted.scale);
            trits.push(fitted.trits);
        } else {
            // This is reachable only through floating accumulation disagreement between residual
            // and full-prefix scoring. Emit a package-valid minimum-scale plane; the tile-level
            // monotonicity gate truncates the frontier before this suffix.
            scales.push(minimum_s34_scale());
            trits.push(canonical_min_scale_s34(weights.len()));
        }
    }

    refine_nested_s34(
        weights,
        metric,
        &mut scales,
        &mut trits,
        config.max_iterations,
    )?;

    Ok(ProgressiveS34Fit { scales, trits })
}

/// Jointly revisit every structured plane while treating all other planes as fixed. The acceptance
/// rule is Pareto-safe over the nested prefixes: an update may improve one or more affected points,
/// but may not spend quality from P1 or P2 to improve P3.
fn refine_nested_s34(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    scales: &mut [f32],
    trits: &mut [Vec<i8>],
    max_iterations: usize,
) -> Result<(), JointFitError> {
    let mut objectives = s34_prefix_objectives(weights, metric, scales, trits)?;
    for _ in 0..max_iterations {
        let mut improved = false;
        for plane in 0..scales.len() {
            let mut target = weights.to_vec();
            for other in 0..scales.len() {
                if other == plane {
                    continue;
                }
                for (value, trit) in target.iter_mut().zip(&trits[other]) {
                    *value -= scales[other] * f32::from(*trit);
                }
            }
            let mut fitted = S34PlaneFit {
                scale: scales[plane],
                trits: trits[plane].clone(),
                objective: score_s34_plane(&target, metric, scales[plane], &trits[plane])?,
            };
            let candidate_scale = optimal_s34_scale(&target, &fitted.trits, metric)?;
            let scale_objective = score_s34_plane(&target, metric, candidate_scale, &fitted.trits)?;
            if scale_objective < fitted.objective
                || (scale_objective == fitted.objective && candidate_scale < fitted.scale)
            {
                fitted.scale = candidate_scale;
                fitted.objective = scale_objective;
            }
            let candidate_trits =
                coordinate_s34_assignment(&target, metric, fitted.scale, &fitted.trits)?;
            let assignment_objective =
                score_s34_plane(&target, metric, fitted.scale, &candidate_trits)?;
            if assignment_objective < fitted.objective
                || (assignment_objective == fitted.objective && candidate_trits < fitted.trits)
            {
                fitted.trits = candidate_trits;
            }

            let prior_scale = scales[plane];
            let prior_trits = std::mem::replace(&mut trits[plane], fitted.trits);
            scales[plane] = fitted.scale;
            let candidate_objectives = s34_prefix_objectives(weights, metric, scales, trits)?;
            if nested_s34_update_precedes(plane, &candidate_objectives, &objectives) {
                objectives = candidate_objectives;
                improved = true;
            } else {
                scales[plane] = prior_scale;
                trits[plane] = prior_trits;
            }
        }
        if !improved {
            break;
        }
    }
    Ok(())
}

fn s34_prefix_objectives(
    weights: &[f32],
    metric: JointFitMetric<'_>,
    scales: &[f32],
    trits: &[Vec<i8>],
) -> Result<Vec<f64>, JointFitError> {
    let mut reconstruction = vec![0.0f32; weights.len()];
    let mut objectives = Vec::with_capacity(scales.len());
    for (scale, plane) in scales.iter().zip(trits) {
        for (value, trit) in reconstruction.iter_mut().zip(plane) {
            *value += *scale * f32::from(*trit);
        }
        let objective = checked_s34_objective(weights, &reconstruction, metric)?;
        objectives.push(objective);
    }
    Ok(objectives)
}

fn nested_s34_update_precedes(plane: usize, candidate: &[f64], current: &[f64]) -> bool {
    let monotone = candidate.windows(2).all(|pair| {
        let tolerance = 1e-12f64.max(pair[0].abs() * 1e-12);
        pair[1] <= pair[0] + tolerance
    });
    if !monotone {
        return false;
    }
    let mut strict = false;
    for index in plane..candidate.len() {
        let tolerance = 1e-12f64.max(current[index].abs() * 1e-12);
        if candidate[index] > current[index] + tolerance {
            return false;
        }
        strict |= candidate[index] < current[index] - tolerance;
    }
    strict
}

fn fit_s34_residual_plane(
    target: &[f32],
    metric: JointFitMetric<'_>,
    unconstrained_scale: f32,
    max_iterations: usize,
    restarts: usize,
) -> Result<S34PlaneFit, JointFitError> {
    let fallback_trits = canonical_min_scale_s34(target.len());
    let fallback_scale = minimum_s34_scale();
    let fallback_objective = score_s34_plane(target, metric, fallback_scale, &fallback_trits)?;
    let mut best = S34PlaneFit {
        scale: fallback_scale,
        trits: fallback_trits,
        objective: fallback_objective,
    };

    // One physical quartet is small enough to solve exactly. This is also the executable oracle
    // for the general alternating constrained solver.
    if target.len() <= 4 {
        for pattern in s34_patterns(target.len()) {
            let scale = optimal_s34_scale(target, &pattern, metric)?;
            let objective = score_s34_plane(target, metric, scale, &pattern)?;
            let candidate = S34PlaneFit {
                scale,
                trits: pattern,
                objective,
            };
            if s34_plane_precedes(&candidate, &best) {
                best = candidate;
            }
        }
        return Ok(best);
    }

    let anchor = if unconstrained_scale > 0.0 {
        unconstrained_scale
    } else {
        diagonal_weighted_abs_mean(target, metric)?
    };
    for restart in 0..restarts {
        let scale = s34_restart_scale(anchor, restart, restarts)?;
        let trits = initial_s34_assignment(target, metric, scale, restart);
        let objective = score_s34_plane(target, metric, scale, &trits)?;
        let mut state = S34PlaneFit {
            scale,
            trits,
            objective,
        };

        for _ in 0..max_iterations {
            let mut changed = false;
            let scale = optimal_s34_scale(target, &state.trits, metric)?;
            let scale_objective = score_s34_plane(target, metric, scale, &state.trits)?;
            if scale_objective < state.objective
                || (scale_objective == state.objective && scale < state.scale)
            {
                state.scale = scale;
                state.objective = scale_objective;
                changed = true;
            }

            let assignment = coordinate_s34_assignment(target, metric, state.scale, &state.trits)?;
            let assignment_objective = score_s34_plane(target, metric, state.scale, &assignment)?;
            if assignment_objective < state.objective
                || (assignment_objective == state.objective && assignment < state.trits)
            {
                state.trits = assignment;
                state.objective = assignment_objective;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        if s34_plane_precedes(&state, &best) {
            best = state;
        }
    }
    Ok(best)
}

fn s34_plane_precedes(candidate: &S34PlaneFit, current: &S34PlaneFit) -> bool {
    candidate
        .objective
        .total_cmp(&current.objective)
        .then_with(|| candidate.scale.total_cmp(&current.scale))
        .then_with(|| candidate.trits.cmp(&current.trits))
        .is_lt()
}

fn canonical_min_scale_s34(logical_len: usize) -> Vec<i8> {
    let mut trits = vec![-1; logical_len];
    for start in (0..logical_len).step_by(4) {
        trits[start] = 0;
    }
    trits
}

/// Enumerate every logical pattern whose canonical shape completion has exactly one zero in its
/// physical quartet. Full groups require one logical zero; ragged groups permit zero or one because
/// the package's canonical completion inserts the missing zero before negative padding.
fn s34_patterns(logical_len: usize) -> Vec<Vec<i8>> {
    debug_assert!((1..=4).contains(&logical_len));
    const TRITS: [i8; 3] = [-1, 0, 1];
    let states = 3usize.pow(logical_len as u32);
    let mut patterns = Vec::with_capacity(32);
    for state in 0..states {
        let mut encoded = state;
        let mut pattern = Vec::with_capacity(logical_len);
        for _ in 0..logical_len {
            pattern.push(TRITS[encoded % 3]);
            encoded /= 3;
        }
        let zeros = pattern.iter().filter(|trit| **trit == 0).count();
        if (logical_len == 4 && zeros == 1) || (logical_len < 4 && zeros <= 1) {
            patterns.push(pattern);
        }
    }
    patterns.sort();
    patterns
}

fn diagonal_weighted_abs_mean(
    target: &[f32],
    metric: JointFitMetric<'_>,
) -> Result<f32, JointFitError> {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (index, value) in target.iter().enumerate() {
        let diagonal = metric_diagonal(metric, index);
        numerator += diagonal * f64::from(value.abs());
        denominator += diagonal;
    }
    if !(numerator.is_finite() && denominator.is_finite()) || denominator <= 0.0 {
        return Err(JointFitError::ScaleSolveFailed);
    }
    deploy_s34_scale(numerator / denominator)
}

fn s34_restart_scale(anchor: f32, restart: usize, restarts: usize) -> Result<f32, JointFitError> {
    if restart == 0 {
        return deploy_s34_scale(f64::from(anchor));
    }
    let span = restart as f64 / restarts as f64;
    let factor = 0.5 + span;
    deploy_s34_scale(f64::from(anchor) * factor)
}

fn initial_s34_assignment(
    target: &[f32],
    metric: JointFitMetric<'_>,
    scale: f32,
    restart: usize,
) -> Vec<i8> {
    let mut trits = Vec::with_capacity(target.len());
    for start in (0..target.len()).step_by(4) {
        let end = (start + 4).min(target.len());
        let mut ranked = s34_patterns(end - start)
            .into_iter()
            .map(|pattern| {
                let objective = pattern
                    .iter()
                    .enumerate()
                    .map(|(offset, trit)| {
                        let error =
                            f64::from(target[start + offset]) - f64::from(scale) * f64::from(*trit);
                        metric_diagonal(metric, start + offset) * error * error
                    })
                    .sum::<f64>();
                (objective, pattern)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let rank = restart.min(ranked.len() - 1);
        trits.extend_from_slice(&ranked[rank].1);
    }
    trits
}

fn coordinate_s34_assignment(
    target: &[f32],
    metric: JointFitMetric<'_>,
    scale: f32,
    initial: &[i8],
) -> Result<Vec<i8>, JointFitError> {
    let mut trits = initial.to_vec();
    let mut error = target
        .iter()
        .zip(&trits)
        .map(|(value, trit)| f64::from(*value) - f64::from(scale) * f64::from(*trit))
        .collect::<Vec<_>>();
    let mut metric_error = apply_metric(metric, &error);
    let mut objective = error
        .iter()
        .zip(&metric_error)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !objective.is_finite() {
        return Err(JointFitError::NonFiniteObjective);
    }

    for start in (0..target.len()).step_by(4) {
        let end = (start + 4).min(target.len());
        let current = trits[start..end].to_vec();
        let mut best_pattern = current.clone();
        let mut best_objective = objective;
        for pattern in s34_patterns(end - start) {
            let delta = pattern
                .iter()
                .zip(&current)
                .map(|(new, old)| *new - *old)
                .collect::<Vec<_>>();
            let linear = delta
                .iter()
                .enumerate()
                .map(|(offset, value)| f64::from(*value) * metric_error[start + offset])
                .sum::<f64>();
            let mut quadratic = 0.0f64;
            for (row_offset, row_delta) in delta.iter().enumerate() {
                for (column_offset, column_delta) in delta.iter().enumerate() {
                    quadratic += f64::from(*row_delta)
                        * metric_entry_local(metric, start + row_offset, start + column_offset)
                        * f64::from(*column_delta);
                }
            }
            let candidate_objective =
                objective - 2.0 * f64::from(scale) * linear + f64::from(scale).powi(2) * quadratic;
            if candidate_objective
                .total_cmp(&best_objective)
                .then_with(|| pattern.cmp(&best_pattern))
                .is_lt()
            {
                best_objective = candidate_objective;
                best_pattern = pattern;
            }
        }

        if best_pattern != current {
            let delta = best_pattern
                .iter()
                .zip(&current)
                .map(|(new, old)| *new - *old)
                .collect::<Vec<_>>();
            for (offset, value) in best_pattern.iter().enumerate() {
                trits[start + offset] = *value;
                error[start + offset] -= f64::from(scale) * f64::from(delta[offset]);
            }
            for (row, metric_value) in metric_error.iter_mut().enumerate() {
                let update = delta
                    .iter()
                    .enumerate()
                    .map(|(offset, value)| {
                        metric_entry_local(metric, row, start + offset) * f64::from(*value)
                    })
                    .sum::<f64>();
                *metric_value -= f64::from(scale) * update;
            }
            objective = best_objective;
        }
    }
    Ok(trits)
}

fn optimal_s34_scale(
    target: &[f32],
    trits: &[i8],
    metric: JointFitMetric<'_>,
) -> Result<f32, JointFitError> {
    let trits_f64 = trits
        .iter()
        .map(|trit| f64::from(*trit))
        .collect::<Vec<_>>();
    let target_f64 = target
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let metric_target = apply_metric(metric, &target_f64);
    let metric_trits = apply_metric(metric, &trits_f64);
    let numerator = trits_f64
        .iter()
        .zip(metric_target)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let denominator = trits_f64
        .iter()
        .zip(metric_trits)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !(numerator.is_finite() && denominator.is_finite()) {
        return Err(JointFitError::ScaleSolveFailed);
    }
    let optimum = if denominator <= 0.0 || numerator <= 0.0 {
        f64::from(minimum_s34_scale())
    } else {
        numerator / denominator
    };
    let rounded = deploy_s34_scale(optimum)?;
    let rounded_bits = f16::from_f32(rounded).to_bits();
    let mut best = None::<(f64, f32)>;
    for bits in rounded_bits.saturating_sub(1)..=rounded_bits.saturating_add(1) {
        if bits == 0 || bits > f16::MAX.to_bits() {
            continue;
        }
        let scale = f16::from_bits(bits).to_f32();
        let objective = score_s34_plane(target, metric, scale, trits)?;
        if best.as_ref().is_none_or(|(best_objective, best_scale)| {
            objective
                .total_cmp(best_objective)
                .then_with(|| scale.total_cmp(best_scale))
                .is_lt()
        }) {
            best = Some((objective, scale));
        }
    }
    best.map(|(_, scale)| scale)
        .ok_or(JointFitError::ScaleNotRepresentable { plane: 0 })
}

fn deploy_s34_scale(scale: f64) -> Result<f32, JointFitError> {
    if !scale.is_finite() || scale < 0.0 {
        return Err(JointFitError::ScaleSolveFailed);
    }
    let bounded = scale.clamp(f64::from(minimum_s34_scale()), f64::from(f16::MAX.to_f32())) as f32;
    let deployed = f16::from_f32(bounded).to_f32();
    if deployed.is_finite() {
        Ok(deployed)
    } else {
        Err(JointFitError::ScaleNotRepresentable { plane: 0 })
    }
}

fn minimum_s34_scale() -> f32 {
    f16::from_bits(1).to_f32()
}

fn score_s34_plane(
    target: &[f32],
    metric: JointFitMetric<'_>,
    scale: f32,
    trits: &[i8],
) -> Result<f64, JointFitError> {
    let reconstruction = trits
        .iter()
        .map(|trit| scale * f32::from(*trit))
        .collect::<Vec<_>>();
    checked_s34_objective(target, &reconstruction, metric)
}

fn checked_s34_objective(
    weights: &[f32],
    reconstruction: &[f32],
    metric: JointFitMetric<'_>,
) -> Result<f64, JointFitError> {
    let objective = reconstruction_objective(weights, reconstruction, metric);
    if !objective.is_finite() {
        return Err(JointFitError::NonFiniteObjective);
    }
    // `DensePsdMetric` validates the mathematical PSD contract. A negative result can therefore
    // only be cancellation in the direct quadratic accumulation.
    Ok(objective.max(0.0))
}

fn apply_metric(metric: JointFitMetric<'_>, values: &[f64]) -> Vec<f64> {
    match metric {
        JointFitMetric::Identity => values.to_vec(),
        JointFitMetric::Diagonal(diagonal) => values
            .iter()
            .zip(diagonal)
            .map(|(value, weight)| value * f64::from(*weight))
            .collect(),
        JointFitMetric::Dense(dense) => {
            let dimension = dense.dimension();
            let matrix = dense.as_slice();
            (0..dimension)
                .map(|row| {
                    values
                        .iter()
                        .enumerate()
                        .map(|(column, value)| matrix[row * dimension + column] * value)
                        .sum()
                })
                .collect()
        }
    }
}

fn metric_diagonal(metric: JointFitMetric<'_>, index: usize) -> f64 {
    metric_entry_local(metric, index, index)
}

fn metric_entry_local(metric: JointFitMetric<'_>, row: usize, column: usize) -> f64 {
    match metric {
        JointFitMetric::Identity => f64::from(row == column),
        JointFitMetric::Diagonal(diagonal) => {
            if row == column {
                f64::from(diagonal[row])
            } else {
                0.0
            }
        }
        JointFitMetric::Dense(dense) => dense.as_slice()[row * dense.dimension() + column],
    }
}

fn reconstruction_objective(
    weights: &[f32],
    reconstruction: &[f32],
    metric: JointFitMetric<'_>,
) -> f64 {
    match metric {
        JointFitMetric::Identity => weights
            .iter()
            .zip(reconstruction)
            .map(|(weight, reconstructed)| {
                let error = f64::from(*weight) - f64::from(*reconstructed);
                error * error
            })
            .sum(),
        JointFitMetric::Diagonal(diagonal) => weights
            .iter()
            .zip(reconstruction)
            .zip(diagonal)
            .map(|((weight, reconstructed), curvature)| {
                let error = f64::from(*weight) - f64::from(*reconstructed);
                f64::from(*curvature) * error * error
            })
            .sum(),
        JointFitMetric::Dense(dense) => {
            let errors = weights
                .iter()
                .zip(reconstruction)
                .map(|(weight, reconstructed)| f64::from(*weight) - f64::from(*reconstructed))
                .collect::<Vec<_>>();
            let dimension = dense.dimension();
            let values = dense.as_slice();
            let mut objective = 0.0f64;
            for row in 0..dimension {
                for column in 0..dimension {
                    objective += errors[row] * values[row * dimension + column] * errors[column];
                }
            }
            objective
        }
    }
}

fn curvature_metric<'a>(
    artifact: CurvatureArtifact<'a>,
    start: usize,
    end: usize,
    group_index: usize,
) -> JointFitMetric<'a> {
    match artifact.values {
        CurvatureValues::Diagonal(diagonal) => JointFitMetric::Diagonal(&diagonal[start..end]),
        CurvatureValues::DenseGroups(groups) => JointFitMetric::Dense(&groups[group_index]),
    }
}

fn plane_physical_bytes(
    packing: SaltV2Packing,
    logical_len: usize,
) -> Result<PhysicalBytes, SaltV2Error> {
    let codec_len = if packing == SaltV2Packing::S34 {
        logical_len
            .div_ceil(4)
            .checked_mul(4)
            .ok_or(SaltV2Error::AccountingOverflow)?
    } else {
        logical_len
    };
    let payload = packing.codec().ledger(codec_len)?.physical_bytes;
    let scales = logical_len
        .div_ceil(SALT_V2_SCALE_GROUP_SIZE)
        .checked_mul(2)
        .ok_or(SaltV2Error::AccountingOverflow)?;
    let total = payload
        .checked_add(scales)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(SaltV2Error::AccountingOverflow)?;
    Ok(PhysicalBytes {
        serialized: total,
        resident: total,
    })
}

fn allocator_candidates(
    work: &[TensorFitWork],
    config: &SaltV2Config,
) -> Result<Vec<Vec<PlaneCandidate>>, SaltV2Error> {
    let mut frontiers = Vec::new();
    for tensor in work {
        for candidates in &tensor.candidates {
            let per_plane = candidates
                .first()
                .map(|candidate| candidate.metrics.cumulative)
                .ok_or(SaltV2Error::PhysicalAccountingMismatch)?;
            let mut allocator = Vec::with_capacity(3);
            for plane_index in 0..3 {
                if let Some(candidate) = candidates.get(plane_index) {
                    allocator.push(PlaneCandidate {
                        planes: (plane_index + 1) as u8,
                        byte_delta: ByteDelta::measured(per_plane, per_plane),
                        distortion: candidate.metrics.hessian_error,
                    });
                } else {
                    let distortion = candidates
                        .last()
                        .ok_or(SaltV2Error::PhysicalAccountingMismatch)?
                        .metrics
                        .hessian_error;
                    // A structurally unavailable S34 suffix carries no distortion reduction and
                    // a real positive plane cost. The exact allocator therefore Pareto-drops it
                    // without an overflowing `u64::MAX` sentinel under unbounded recipes.
                    allocator.push(PlaneCandidate {
                        planes: (plane_index + 1) as u8,
                        byte_delta: ByteDelta::measured(per_plane, per_plane),
                        distortion,
                    });
                }
            }
            frontiers.push(allocator);
        }
    }
    debug_assert!(
        frontiers
            .iter()
            .all(|frontier| frontier.len() == 3 && config.max_planes <= 3)
    );
    Ok(frontiers)
}

fn source_tensor_digest(input: &SaltV2TensorFitInput<'_>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(SOURCE_TENSOR_HASH_CONTEXT);
    write_len_hash(&mut hasher, input.name.len());
    hasher.update(input.name.as_bytes());
    write_len_hash(&mut hasher, input.rows);
    write_len_hash(&mut hasher, input.cols);
    for weight in input.weights {
        hasher.update(&weight.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn bound_curvature_digest(
    kind: SaltV2Curvature,
    source_id: CurvatureSourceId,
    evidence_digest: [u8; 32],
    values: CurvatureValues<'_>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(CURVATURE_HASH_CONTEXT);
    hasher.update(&[kind.tag()]);
    hasher.update(&source_id.digest());
    hasher.update(&evidence_digest);
    match values {
        CurvatureValues::Diagonal(diagonal) => {
            hasher.update(&[1]);
            write_len_hash(&mut hasher, diagonal.len());
            for value in diagonal {
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
        CurvatureValues::DenseGroups(groups) => {
            hasher.update(&[2]);
            write_len_hash(&mut hasher, groups.len());
            for group in groups {
                write_len_hash(&mut hasher, group.dimension());
                for value in group.as_slice() {
                    hasher.update(&value.to_bits().to_le_bytes());
                }
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn bound_feedback_digest(
    source_id: CurvatureSourceId,
    evidence_digest: [u8; 32],
    groups: &[ColumnGroup],
    inverse_hessian: &[f64],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(FEEDBACK_HASH_CONTEXT);
    hasher.update(&source_id.digest());
    hasher.update(&evidence_digest);
    write_len_hash(&mut hasher, groups.len());
    for group in groups {
        write_len_hash(&mut hasher, group.start);
        write_len_hash(&mut hasher, group.end);
    }
    write_len_hash(&mut hasher, inverse_hessian.len());
    for value in inverse_hessian {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn feedback_values_digest(domain: u8, values: &[f64]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(FEEDBACK_RECEIPT_HASH_CONTEXT);
    hasher.update(&[domain]);
    write_len_hash(&mut hasher, values.len());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn feedback_receipt_digest(receipts: &[SaltV2TensorFeedbackReceipt]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(FEEDBACK_RECEIPT_HASH_CONTEXT);
    hasher.update(&[0]);
    write_len_hash(&mut hasher, receipts.len());
    for receipt in receipts {
        write_len_hash(&mut hasher, receipt.name.len());
        hasher.update(receipt.name.as_bytes());
        hasher.update(&receipt.artifact_digest);
        write_len_hash(&mut hasher, receipt.groups.len());
        for group in &receipt.groups {
            write_len_hash(&mut hasher, group.group_index);
            write_len_hash(&mut hasher, group.column_start);
            write_len_hash(&mut hasher, group.column_end);
            hasher.update(&group.provisional_fit_input_digest);
            hasher.update(&group.provisional_reconstruction_digest);
            hasher.update(&group.final_fit_input_digest);
            hasher.update(&group.final_reconstruction_digest);
            hasher.update(&[u8::from(group.nonzero_delta)]);
        }
        hasher.update(&receipt.delta_corrections.to_le_bytes());
        hasher.update(&receipt.nonzero_delta_corrections.to_le_bytes());
        hasher.update(&receipt.final_working_digest);
        hasher.update(&receipt.final_reconstruction_digest);
    }
    *hasher.finalize().as_bytes()
}

fn master_digest(
    source_model_id: ModelId,
    activation_digest: [u8; 32],
    config: &SaltV2Config,
    physical: SaltV2ModelPhysicalInput,
    work: &[TensorFitWork],
    feedback: Option<&SaltV2ModelFeedbackReceipt>,
) -> Result<[u8; 32], SaltV2Error> {
    let mut hasher = blake3::Hasher::new_derive_key(MASTER_HASH_CONTEXT);
    hasher.update(source_model_id.as_bytes());
    hasher.update(&activation_digest);
    let mut search_config = *config;
    search_config.rate = PhysicalRateTarget::default();
    hasher.update(&recipe_digest(&search_config));
    hasher.update(&physical.total_model_parameters.to_le_bytes());
    hasher.update(&physical.preserved_artifact_bytes.to_le_bytes());
    hasher.update(&physical.preserved_resident_bytes.to_le_bytes());
    hasher.update(&physical.required_runtime_shadow_bytes.to_le_bytes());
    match feedback {
        Some(receipt) => {
            hasher.update(&[1]);
            hasher.update(&receipt.receipt_id);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    write_len_hash(&mut hasher, work.len());
    for tensor in work {
        write_len_hash(&mut hasher, tensor.name.len());
        hasher.update(tensor.name.as_bytes());
        write_len_hash(&mut hasher, tensor.rows);
        write_len_hash(&mut hasher, tensor.cols);
        hasher.update(&tensor.source_digest);
        hasher.update(&tensor.curvature_digest);
        write_len_hash(&mut hasher, tensor.candidates.len());
        for (tile_length, frontier) in tensor.tile_lengths.iter().zip(tensor.candidates.iter()) {
            write_len_hash(&mut hasher, *tile_length);
            write_len_hash(&mut hasher, frontier.len());
            for candidate in frontier {
                let metrics = candidate.metrics;
                hasher.update(&[metrics.planes]);
                hasher.update(&metrics.cumulative.serialized.to_le_bytes());
                hasher.update(&metrics.cumulative.resident.to_le_bytes());
                hasher.update(&metrics.hessian_error.to_bits().to_le_bytes());
                hasher.update(&metrics.frobenius_error.to_bits().to_le_bytes());
                write_len_hash(&mut hasher, candidate.tile.planes().len());
                for plane in candidate.tile.planes() {
                    write_len_hash(&mut hasher, plane.trits().len());
                    for trit in plane.trits() {
                        hasher.update(&trit.get().to_le_bytes());
                    }
                    write_len_hash(&mut hasher, plane.scales().len());
                    for scale in plane.scales() {
                        hasher.update(&scale.to_bits().to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn recipe_digest(config: &SaltV2Config) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(RECIPE_HASH_CONTEXT);
    write_len_hash(&mut hasher, config.group_size);
    write_len_hash(&mut hasher, config.min_planes);
    write_len_hash(&mut hasher, config.max_planes);
    hasher.update(&[config.packing.tag(), config.curvature.tag()]);
    match config.transform_seed {
        Some(seed) => {
            hasher.update(&[1]);
            hasher.update(&seed.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    write_len_hash(&mut hasher, config.em_restarts);
    write_len_hash(&mut hasher, config.coordinate_sweeps);
    hasher.update(&config.ridge_condition_limit.to_bits().to_le_bytes());
    hasher.update(&config.rate.max_matrix_bytes.to_le_bytes());
    hasher.update(&config.rate.max_artifact_bytes.to_le_bytes());
    match config.rate.max_resident_bytes {
        Some(bytes) => {
            hasher.update(&[1]);
            hasher.update(&bytes.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match config.refinement {
        SaltV2Refinement::None => {
            hasher.update(&[0]);
        }
        SaltV2Refinement::ScaleOnly { max_tokens } => {
            hasher.update(&[1]);
            hasher.update(&max_tokens.to_le_bytes());
        }
        SaltV2Refinement::PvKl {
            warmup_tokens,
            hard_tokens,
        } => {
            hasher.update(&[2]);
            hasher.update(&warmup_tokens.to_le_bytes());
            hasher.update(&hard_tokens.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

#[derive(Clone, Copy)]
struct ReceiptDigestInput<'a> {
    source_model_id: ModelId,
    activation_digest: [u8; 32],
    recipe_id: [u8; 32],
    master_id: [u8; 32],
    tensors: &'a [SaltV2TensorFitReceipt],
    package_id: PackageId,
    physical: SaltV2ModelPhysicalInput,
    feedback: Option<&'a SaltV2ModelFeedbackReceipt>,
}

fn receipt_digest(input: ReceiptDigestInput<'_>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    write_len_hash(&mut hasher, REFERENCE_SOLVER_VERSION.len());
    hasher.update(REFERENCE_SOLVER_VERSION.as_bytes());
    hasher.update(input.source_model_id.as_bytes());
    hasher.update(&input.activation_digest);
    hasher.update(&input.recipe_id);
    hasher.update(&input.master_id);
    write_len_hash(&mut hasher, input.tensors.len());
    for tensor in input.tensors {
        write_len_hash(&mut hasher, tensor.name.len());
        hasher.update(tensor.name.as_bytes());
        hasher.update(&tensor.source_digest);
        hasher.update(&tensor.curvature_digest);
        write_len_hash(&mut hasher, tensor.plane_counts.len());
        hasher.update(&tensor.plane_counts);
    }
    hasher.update(input.package_id.as_bytes());
    hasher.update(&input.physical.total_model_parameters.to_le_bytes());
    hasher.update(&input.physical.preserved_artifact_bytes.to_le_bytes());
    hasher.update(&input.physical.preserved_resident_bytes.to_le_bytes());
    hasher.update(&input.physical.required_runtime_shadow_bytes.to_le_bytes());
    match input.feedback {
        Some(receipt) => {
            hasher.update(&[1]);
            hasher.update(&receipt.receipt_id);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[
        SaltV2FitTrack::Ptq as u8,
        u8::from(input.feedback.is_some()),
        0,
    ]);
    *hasher.finalize().as_bytes()
}

fn write_len_hash(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u128).to_le_bytes());
}

fn exact_bpw(bytes: u64, parameters: u64) -> Result<f64, SaltV2Error> {
    if parameters == 0 {
        return Err(SaltV2Error::InvalidTotalModelParameters);
    }
    Ok(bytes as f64 * 8.0 / parameters as f64)
}

fn no_feasible(config: &SaltV2Config, aligned_matrix: u64) -> SaltV2Error {
    SaltV2Error::NoFeasibleAllocation {
        max_matrix_bytes: aligned_matrix,
        max_resident_bytes: config.rate.max_resident_bytes.unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::salt_v2_activation::{
        ActivationCache, ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk,
        ActivationDType, ActivationDigest,
    };
    use tritium_format::{SemanticModelManifest, SemanticTensor};

    fn source_model_id() -> tritium_format::ModelId {
        SemanticModelManifest::new(
            "test-transformer",
            b"{}",
            vec![SemanticTensor::new("weight", vec![1, 1], &[0]).expect("semantic tensor")],
        )
        .expect("semantic manifest")
        .model_id()
    }

    fn activation_cache() -> ActivationCache {
        activation_cache_with_source([7; 32])
    }

    fn activation_cache_with_source(source_digest: [u8; 32]) -> ActivationCache {
        let spec = ActivationCacheSpec::new(
            0,
            "weight.input",
            1,
            1,
            ActivationDType::Float32,
            ActivationDigest::from_bytes(source_digest),
            1,
        )
        .expect("activation spec");
        let chunk = ActivationChunk::new(&spec, 0, 1, vec![1.0], vec![true], vec![1])
            .expect("activation chunk");
        let mut builder = ActivationCacheBuilder::new(spec);
        builder.ingest(chunk).expect("activation ingest");
        builder.finalize().expect("activation cache")
    }

    fn curvature_source(model_id: ModelId, cache: &ActivationCache) -> CurvatureSourceId {
        CurvatureSourceId::new(
            *model_id.as_bytes(),
            cache.digest().into_bytes(),
            cache.spec().source_digest().into_bytes(),
        )
        .expect("complete curvature provenance")
    }

    fn different_digest(mut digest: [u8; 32]) -> [u8; 32] {
        digest[0] = digest[0].wrapping_add(1);
        if digest == [0; 32] {
            digest[1] = 1;
        }
        digest
    }

    fn weights(tile_count: usize) -> Vec<f32> {
        (0..tile_count * 256)
            .map(|index| {
                let phase = (index % 23) as f32 - 11.0;
                phase / 13.0 + ((index / 128) % 3) as f32 * 0.03125
            })
            .collect()
    }

    fn config(matrix_bytes: u64) -> SaltV2Config {
        SaltV2Config {
            rate: PhysicalRateTarget {
                max_matrix_bytes: matrix_bytes,
                max_artifact_bytes: matrix_bytes + 97,
                max_resident_bytes: Some(matrix_bytes + 41),
            },
            ..SaltV2Config::default()
        }
    }

    fn fit(
        weights: &[f32],
        diagonal: &[f32],
        config: &SaltV2Config,
    ) -> Result<SaltV2ModelFitResult, SaltV2Error> {
        fit_with_curvature_evidence(weights, diagonal, [11; 32], config)
    }

    fn fit_with_curvature_evidence(
        weights: &[f32],
        diagonal: &[f32],
        evidence_digest: [u8; 32],
        config: &SaltV2Config,
    ) -> Result<SaltV2ModelFitResult, SaltV2Error> {
        fit_with_physical(
            weights,
            diagonal,
            evidence_digest,
            config,
            SaltV2ModelPhysicalInput {
                total_model_parameters: weights.len() as u64 + 10,
                preserved_artifact_bytes: 97,
                preserved_resident_bytes: 17,
                required_runtime_shadow_bytes: 24,
            },
        )
    }

    fn fit_with_physical(
        weights: &[f32],
        diagonal: &[f32],
        evidence_digest: [u8; 32],
        config: &SaltV2Config,
        physical: SaltV2ModelPhysicalInput,
    ) -> Result<SaltV2ModelFitResult, SaltV2Error> {
        let cache = activation_cache();
        let model_id = source_model_id();
        let source_id = curvature_source(model_id, &cache);
        fit_with_provenance(
            weights,
            diagonal,
            evidence_digest,
            config,
            physical,
            (model_id, &cache, source_id),
        )
    }

    fn fit_with_provenance(
        weights: &[f32],
        diagonal: &[f32],
        evidence_digest: [u8; 32],
        config: &SaltV2Config,
        physical: SaltV2ModelPhysicalInput,
        provenance: (ModelId, &ActivationCache, CurvatureSourceId),
    ) -> Result<SaltV2ModelFitResult, SaltV2Error> {
        let (model_id, cache, source_id) = provenance;
        let curvature = CurvatureArtifact::diagonal_fisher(source_id, evidence_digest, diagonal);
        let tensor = SaltV2TensorFitInput {
            name: "model.layers.0.mlp.down_proj.weight",
            weights,
            rows: 2,
            cols: weights.len() / 2,
            curvature,
        };
        fit_salt_v2_model(
            SaltV2ModelFitInput {
                tensors: &[tensor],
                activations: cache,
                source_model_id: model_id,
                physical,
            },
            config,
        )
    }

    fn feedback_inverse_hessian(columns: usize) -> Vec<f64> {
        let mut inverse = vec![0.0; columns * columns];
        for column in 0..columns {
            inverse[column * columns + column] = 1.0;
            if column + 1 < columns {
                inverse[column * columns + column + 1] = 0.125;
                inverse[(column + 1) * columns + column] = 0.125;
            }
        }
        inverse
    }

    #[test]
    fn feedback_master_is_reusable_and_compact_is_an_exact_prefix() {
        let source_weights = weights(2);
        let diagonal = vec![1.0; source_weights.len()];
        let cache = activation_cache();
        let model_id = source_model_id();
        let source_id = curvature_source(model_id, &cache);
        let curvature = CurvatureArtifact::diagonal_fisher(source_id, [31; 32], &diagonal);
        let tensor = SaltV2TensorFitInput {
            name: "model.layers.0.mlp.down_proj.weight",
            weights: &source_weights,
            rows: 2,
            cols: 256,
            curvature,
        };
        let groups = [
            ColumnGroup { start: 0, end: 128 },
            ColumnGroup {
                start: 128,
                end: 256,
            },
        ];
        let inverse_hessian = feedback_inverse_hessian(256);
        let feedback =
            SaltV2FeedbackArtifact::inverse_hessian(source_id, [41; 32], &groups, &inverse_hessian);
        let physical = SaltV2ModelPhysicalInput {
            total_model_parameters: source_weights.len() as u64 + 10,
            preserved_artifact_bytes: 97,
            preserved_resident_bytes: 17,
            required_runtime_shadow_bytes: 24,
        };
        let tensors = [tensor];
        let feedback_artifacts = [feedback];
        let recipe = config(10_000);
        let master_input = SaltV2MasterFitInput {
            model: SaltV2ModelFitInput {
                tensors: &tensors,
                activations: &cache,
                source_model_id: model_id,
                physical,
            },
            feedback: &feedback_artifacts,
        };
        let master = fit_salt_v2_master(master_input, &recipe).expect("feedback master");
        let replay = fit_salt_v2_master(master_input, &recipe).expect("feedback master replay");
        assert_eq!(replay.master_id(), master.master_id());
        assert_eq!(replay.feedback_receipt(), master.feedback_receipt());

        let near = allocate_and_pack_salt_v2_master(&master, recipe.rate)
            .expect("near-lossless allocation");
        assert!(near.receipt.feedback_applied);
        assert_eq!(near.receipt.master_id, master.master_id());
        let feedback_receipt = near.receipt.feedback.as_ref().expect("feedback receipt");
        assert_eq!(feedback_receipt.tensors.len(), 1);
        assert_eq!(feedback_receipt.tensors[0].groups.len(), 2);
        assert_eq!(feedback_receipt.tensors[0].delta_corrections, 2);
        assert!(feedback_receipt.tensors[0].nonzero_delta_corrections > 0);
        assert!(feedback_receipt.tensors[0].nonzero_delta_corrections <= 2);

        let one_plane_raw = near.metrics.physical.serialized_fixed_bytes
            + near
                .metrics
                .tile_candidates
                .iter()
                .filter(|candidate| candidate.planes == 1)
                .map(|candidate| candidate.cumulative.serialized)
                .sum::<u64>();
        let compact_ceiling = one_plane_raw.div_ceil(SALT_V2_PACKAGE_ALIGNMENT as u64)
            * SALT_V2_PACKAGE_ALIGNMENT as u64;
        let compact = allocate_and_pack_salt_v2_master(
            &master,
            PhysicalRateTarget {
                max_matrix_bytes: compact_ceiling,
                max_artifact_bytes: compact_ceiling + physical.preserved_artifact_bytes,
                max_resident_bytes: None,
            },
        )
        .expect("compact allocation");
        assert!(
            compact
                .metrics
                .selected_plane_counts
                .iter()
                .all(|planes| *planes == 1)
        );
        assert_eq!(compact.receipt.master_id, near.receipt.master_id);
        assert_eq!(compact.receipt.feedback, near.receipt.feedback);

        let near_package = SaltV2Package::new(SaltV2Codec::D2, near.tensors.clone())
            .expect("near semantic package");
        let requested = vec![
            compact
                .metrics
                .selected_plane_counts
                .iter()
                .map(|planes| usize::from(*planes))
                .collect::<Vec<_>>(),
        ];
        let derived = near_package
            .derive_prefix(&requested)
            .expect("derive exact compact prefix");
        assert_eq!(derived.tensors(), compact.tensors.as_slice());
    }

    #[test]
    fn feedback_artifact_digest_binds_groups_metric_and_provenance() {
        let cache = activation_cache();
        let model_id = source_model_id();
        let source_id = curvature_source(model_id, &cache);
        let groups = [ColumnGroup { start: 0, end: 2 }];
        let split_groups = [
            ColumnGroup { start: 0, end: 1 },
            ColumnGroup { start: 1, end: 2 },
        ];
        let left_metric = [1.0, 0.25, 0.25, 1.0];
        let right_metric = [1.0, 0.5, 0.5, 1.0];
        let left =
            SaltV2FeedbackArtifact::inverse_hessian(source_id, [51; 32], &groups, &left_metric);
        let changed_groups = SaltV2FeedbackArtifact::inverse_hessian(
            source_id,
            [51; 32],
            &split_groups,
            &left_metric,
        );
        let changed_metric =
            SaltV2FeedbackArtifact::inverse_hessian(source_id, [51; 32], &groups, &right_metric);
        let changed_source = SaltV2FeedbackArtifact::inverse_hessian(
            CurvatureSourceId::new(
                source_id.source_model_digest(),
                source_id.activation_cache_digest(),
                different_digest(source_id.token_stream_digest()),
            )
            .expect("different source"),
            [51; 32],
            &groups,
            &left_metric,
        );

        assert_ne!(left.digest(), changed_groups.digest());
        assert_ne!(left.digest(), changed_metric.digest());
        assert_ne!(left.digest(), changed_source.digest());
    }

    #[test]
    fn plane_order_failure_reports_the_earliest_impossible_prefix() {
        let trits = vec![vec![1], vec![1], vec![1]];
        let scales = [1.0, 1.0, 10.0];

        assert_eq!(
            progressive_plane_order(&[1.0], JointFitMetric::Identity, &scales, &trits).unwrap_err(),
            2
        );
        assert_eq!(
            progressive_plane_order(&[3.0], JointFitMetric::Identity, &scales, &trits).unwrap_err(),
            3
        );
    }

    #[test]
    fn feedback_master_fails_closed_on_missing_unbound_or_invalid_evidence() {
        let source_weights = weights(1);
        let diagonal = vec![1.0; source_weights.len()];
        let cache = activation_cache();
        let model_id = source_model_id();
        let source_id = curvature_source(model_id, &cache);
        let curvature = CurvatureArtifact::diagonal_fisher(source_id, [61; 32], &diagonal);
        let tensors = [SaltV2TensorFitInput {
            name: "model.layers.0.mlp.down_proj.weight",
            weights: &source_weights,
            rows: 2,
            cols: 128,
            curvature,
        }];
        let model = SaltV2ModelFitInput {
            tensors: &tensors,
            activations: &cache,
            source_model_id: model_id,
            physical: SaltV2ModelPhysicalInput {
                total_model_parameters: source_weights.len() as u64,
                ..SaltV2ModelPhysicalInput::default()
            },
        };
        let recipe = config(10_000);
        assert_eq!(
            fit_salt_v2_master(
                SaltV2MasterFitInput {
                    model,
                    feedback: &[],
                },
                &recipe,
            )
            .unwrap_err(),
            SaltV2Error::FeedbackArtifactCountMismatch {
                expected: 1,
                got: 0,
            }
        );

        let groups = [ColumnGroup { start: 0, end: 128 }];
        let inverse_hessian = feedback_inverse_hessian(128);
        let wrong_source = CurvatureSourceId::new(
            different_digest(source_id.source_model_digest()),
            source_id.activation_cache_digest(),
            source_id.token_stream_digest(),
        )
        .expect("different source");
        let wrong_source_feedback = [SaltV2FeedbackArtifact::inverse_hessian(
            wrong_source,
            [71; 32],
            &groups,
            &inverse_hessian,
        )];
        assert_eq!(
            fit_salt_v2_master(
                SaltV2MasterFitInput {
                    model,
                    feedback: &wrong_source_feedback,
                },
                &recipe,
            )
            .unwrap_err(),
            SaltV2Error::FeedbackSourceModelMismatch { tensor: 0 }
        );

        let gapped_groups = [ColumnGroup {
            start: 128,
            end: 256,
        }];
        let gapped_feedback = [SaltV2FeedbackArtifact::inverse_hessian(
            source_id,
            [71; 32],
            &gapped_groups,
            &inverse_hessian,
        )];
        assert_eq!(
            fit_salt_v2_master(
                SaltV2MasterFitInput {
                    model,
                    feedback: &gapped_feedback,
                },
                &recipe,
            )
            .unwrap_err(),
            SaltV2Error::Feedback {
                tensor: 0,
                source: FeedbackError::InvalidGroupRange {
                    group: 0,
                    expected_start: 0,
                    start: 128,
                    end: 256,
                    columns: 128,
                },
            }
        );

        let unaligned_groups = [
            ColumnGroup { start: 0, end: 64 },
            ColumnGroup {
                start: 64,
                end: 128,
            },
        ];
        let unaligned_feedback = [SaltV2FeedbackArtifact::inverse_hessian(
            source_id,
            [72; 32],
            &unaligned_groups,
            &inverse_hessian,
        )];
        assert_eq!(
            fit_salt_v2_master(
                SaltV2MasterFitInput {
                    model,
                    feedback: &unaligned_feedback,
                },
                &recipe,
            )
            .unwrap_err(),
            SaltV2Error::FeedbackScaleGeometry { tensor: 0 }
        );

        let mut invalid_inverse_hessian = inverse_hessian;
        invalid_inverse_hessian[0] = 0.0;
        let invalid_feedback = [SaltV2FeedbackArtifact::inverse_hessian(
            source_id,
            [73; 32],
            &groups,
            &invalid_inverse_hessian,
        )];
        assert_eq!(
            fit_salt_v2_master(
                SaltV2MasterFitInput {
                    model,
                    feedback: &invalid_feedback,
                },
                &recipe,
            )
            .unwrap_err(),
            SaltV2Error::Feedback {
                tensor: 0,
                source: FeedbackError::NonPositiveDiagonal { index: 0 },
            }
        );
    }

    #[test]
    fn receipt_identity_binds_every_physical_geometry_field() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let config = SaltV2Config::default();
        let base = SaltV2ModelPhysicalInput {
            total_model_parameters: weights.len() as u64 + 10,
            preserved_artifact_bytes: 7,
            preserved_resident_bytes: 11,
            required_runtime_shadow_bytes: 13,
        };
        let reference =
            fit_with_physical(&weights, &diagonal, [11; 32], &config, base).expect("reference fit");
        let variants = [
            SaltV2ModelPhysicalInput {
                total_model_parameters: base.total_model_parameters + 1,
                ..base
            },
            SaltV2ModelPhysicalInput {
                preserved_artifact_bytes: base.preserved_artifact_bytes + 1,
                ..base
            },
            SaltV2ModelPhysicalInput {
                preserved_resident_bytes: base.preserved_resident_bytes + 1,
                ..base
            },
            SaltV2ModelPhysicalInput {
                required_runtime_shadow_bytes: base.required_runtime_shadow_bytes + 1,
                ..base
            },
        ];
        for variant in variants {
            let changed = fit_with_physical(&weights, &diagonal, [11; 32], &config, variant)
                .expect("geometry variant fit");
            assert_eq!(changed.package_bytes, reference.package_bytes);
            assert_ne!(changed.receipt.receipt_id, reference.receipt.receipt_id);
        }
    }

    #[test]
    fn solve_boundary_rejects_each_curvature_provenance_mismatch() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let recipe = config(10_000);
        let physical = SaltV2ModelPhysicalInput {
            total_model_parameters: weights.len() as u64,
            ..SaltV2ModelPhysicalInput::default()
        };
        let model_id = source_model_id();
        let cache = activation_cache();
        let expected = curvature_source(model_id, &cache);

        let wrong_model = CurvatureSourceId::new(
            different_digest(expected.source_model_digest()),
            expected.activation_cache_digest(),
            expected.token_stream_digest(),
        )
        .expect("different model provenance");
        assert_eq!(
            fit_with_provenance(
                &weights,
                &diagonal,
                [12; 32],
                &recipe,
                physical,
                (model_id, &cache, wrong_model),
            )
            .unwrap_err(),
            SaltV2Error::CurvatureSourceModelMismatch { tensor: 0 }
        );

        let wrong_cache = CurvatureSourceId::new(
            expected.source_model_digest(),
            different_digest(expected.activation_cache_digest()),
            expected.token_stream_digest(),
        )
        .expect("different cache provenance");
        assert_eq!(
            fit_with_provenance(
                &weights,
                &diagonal,
                [12; 32],
                &recipe,
                physical,
                (model_id, &cache, wrong_cache),
            )
            .unwrap_err(),
            SaltV2Error::CurvatureActivationCacheMismatch { tensor: 0 }
        );

        let wrong_tokens = CurvatureSourceId::new(
            expected.source_model_digest(),
            expected.activation_cache_digest(),
            different_digest(expected.token_stream_digest()),
        )
        .expect("different token-stream provenance");
        assert_eq!(
            fit_with_provenance(
                &weights,
                &diagonal,
                [12; 32],
                &recipe,
                physical,
                (model_id, &cache, wrong_tokens),
            )
            .unwrap_err(),
            SaltV2Error::CurvatureTokenStreamMismatch { tensor: 0 }
        );
    }

    #[test]
    fn curvature_artifact_digest_binds_source_identity() {
        let cache = activation_cache();
        let model_id = source_model_id();
        let left_source = curvature_source(model_id, &cache);
        let right_source = CurvatureSourceId::new(
            left_source.source_model_digest(),
            left_source.activation_cache_digest(),
            different_digest(left_source.token_stream_digest()),
        )
        .expect("different token-stream provenance");
        let diagonal = [1.0, 2.0];
        let left = CurvatureArtifact::diagonal_fisher(left_source, [44; 32], &diagonal);
        let right = CurvatureArtifact::diagonal_fisher(right_source, [44; 32], &diagonal);

        assert_eq!(left.evidence_digest(), right.evidence_digest());
        assert_ne!(left.source_id(), right.source_id());
        assert_ne!(left.digest(), right.digest());
    }

    #[test]
    fn receipt_transitively_binds_curvature_provenance() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let recipe = config(10_000);
        let physical = SaltV2ModelPhysicalInput {
            total_model_parameters: weights.len() as u64,
            ..SaltV2ModelPhysicalInput::default()
        };
        let model_id = source_model_id();
        let left_cache = activation_cache_with_source([7; 32]);
        let right_cache = activation_cache_with_source([8; 32]);
        let left_source = curvature_source(model_id, &left_cache);
        let right_source = curvature_source(model_id, &right_cache);

        let left = fit_with_provenance(
            &weights,
            &diagonal,
            [44; 32],
            &recipe,
            physical,
            (model_id, &left_cache, left_source),
        )
        .expect("left provenance fit");
        let right = fit_with_provenance(
            &weights,
            &diagonal,
            [44; 32],
            &recipe,
            physical,
            (model_id, &right_cache, right_source),
        )
        .expect("right provenance fit");

        assert_eq!(left.package_bytes, right.package_bytes);
        assert_ne!(
            left.receipt.tensors[0].curvature_digest,
            right.receipt.tensors[0].curvature_digest
        );
        assert_ne!(left.receipt.receipt_id, right.receipt.receipt_id);
        assert_eq!(left.receipt.solver_version, REFERENCE_SOLVER_VERSION);
    }

    #[test]
    fn receipt_digest_directly_binds_tensor_curvature_digest() {
        let tensor = SaltV2TensorFitReceipt {
            name: "model.layers.0.mlp.down_proj.weight".to_owned(),
            source_digest: [31; 32],
            curvature_digest: [41; 32],
            plane_counts: vec![1],
        };
        let mut changed = tensor.clone();
        changed.curvature_digest = [42; 32];
        let model_id = source_model_id();
        let package_id = PackageId::from_package_bytes(b"same package");
        let physical = SaltV2ModelPhysicalInput {
            total_model_parameters: 1,
            ..SaltV2ModelPhysicalInput::default()
        };

        let left_tensors = [tensor];
        let right_tensors = [changed];
        let base = ReceiptDigestInput {
            source_model_id: model_id,
            activation_digest: [51; 32],
            recipe_id: [61; 32],
            master_id: [71; 32],
            tensors: &left_tensors,
            package_id,
            physical,
            feedback: None,
        };
        let left = receipt_digest(base);
        let right = receipt_digest(ReceiptDigestInput {
            tensors: &right_tensors,
            ..base
        });

        assert_ne!(left, right);
    }

    #[test]
    fn exact_model_allocator_matches_brute_force_for_two_tiles() {
        let weights = weights(2);
        let diagonal = vec![1.0; weights.len()];
        let unconstrained = fit(&weights, &diagonal, &config(10_000)).expect("candidate fit");
        let fixed = unconstrained.metrics.physical.serialized_fixed_bytes;
        let candidates = &unconstrained.metrics.tile_candidates;
        assert_eq!(candidates.len(), 6);

        // Permit a deliberately non-uniform two-tile frontier. Brute force all 3^2 choices
        // from the public exact byte/objective telemetry, then compare with a constrained fit.
        let ceiling = fixed
            + candidates
                .iter()
                .filter(|candidate| {
                    (candidate.tile_index == 0 && candidate.planes == 2)
                        || (candidate.tile_index == 1 && candidate.planes == 1)
                })
                .map(|candidate| candidate.cumulative.serialized)
                .sum::<u64>()
            + 7;
        let ceiling = ceiling - ceiling % 8;

        let mut brute = None::<(f64, Vec<u8>)>;
        for left in 1..=3 {
            for right in 1..=3 {
                let selected = [left, right];
                let raw = fixed
                    + selected
                        .iter()
                        .enumerate()
                        .map(|(tile, &planes)| {
                            candidates
                                .iter()
                                .find(|candidate| {
                                    candidate.tile_index == tile && candidate.planes == planes
                                })
                                .expect("candidate")
                                .cumulative
                                .serialized
                        })
                        .sum::<u64>();
                let encoded = raw.div_ceil(8) * 8;
                if encoded > ceiling {
                    continue;
                }
                let objective = selected
                    .iter()
                    .enumerate()
                    .map(|(tile, &planes)| {
                        candidates
                            .iter()
                            .find(|candidate| {
                                candidate.tile_index == tile && candidate.planes == planes
                            })
                            .expect("candidate")
                            .hessian_error
                    })
                    .sum::<f64>();
                let counts = selected.to_vec();
                if brute.as_ref().is_none_or(|(best, prior)| {
                    objective < *best || (objective == *best && counts < *prior)
                }) {
                    brute = Some((objective, counts));
                }
            }
        }

        let constrained = fit(&weights, &diagonal, &config(ceiling)).expect("constrained fit");
        let expected = brute.expect("feasible brute-force point");
        assert_eq!(constrained.metrics.selected_plane_counts, expected.1);
        assert!((constrained.metrics.hessian_error - expected.0).abs() <= 1e-10);
    }

    #[test]
    fn hard_integer_ceilings_are_never_crossed() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let fitted = fit(&weights, &diagonal, &config(10_000)).expect("fit");
        assert!(fitted.metrics.physical.matrix_bytes <= fitted.config.rate.max_matrix_bytes);
        assert!(fitted.metrics.physical.artifact_bytes <= fitted.config.rate.max_artifact_bytes);
        assert!(
            fitted.metrics.physical.resident_bytes
                <= fitted
                    .config
                    .rate
                    .max_resident_bytes
                    .expect("resident ceiling")
        );
        let decoded = tritium_format::salt_v2_package::read_salt_v2_package(&fitted.package_bytes)
            .expect("decode exact package ledger");
        let runtime = tritium_format::salt_v2_package::SaltV2IndexedRuntimeLedger::for_package(
            &decoded.package,
        )
        .expect("indexed runtime ledger");
        assert_eq!(
            fitted.metrics.physical.matrix_bytes,
            decoded.ledger.total_bytes
        );
        assert_eq!(
            fitted.metrics.physical.resident_bytes,
            runtime.steady_resident_bytes() + 41
        );
        assert_eq!(
            fitted.metrics.physical.runtime_map_bytes,
            runtime.allocation_map_bytes()
        );
        assert_eq!(
            fitted.metrics.physical.runtime_rank_prefix_bytes,
            runtime.rank_prefix_bytes()
        );
        assert_eq!(fitted.metrics.physical.peak_resident_bytes, None);
        assert_eq!(
            fitted.metrics.physical.trit_payload_bytes,
            decoded.ledger.payload_bytes
        );
        assert_eq!(
            fitted.metrics.physical.scale_bytes,
            decoded.ledger.scales_bytes
        );
        assert_eq!(
            fitted.metrics.physical.allocation_map_bytes,
            decoded.ledger.maps_bytes
        );
        assert_eq!(
            fitted.metrics.physical.allocation_map_bits,
            decoded.ledger.allocation_map_bits
        );
        assert_eq!(
            fitted.metrics.physical.allocation_map_embedded_bits,
            decoded.ledger.allocation_map_embedded_bits
        );
        assert_eq!(
            fitted.metrics.physical.header_bytes,
            decoded.ledger.headers_bytes
        );
        assert_eq!(
            fitted.metrics.physical.transform_bytes,
            decoded.ledger.transform_bytes
        );
        assert_eq!(
            fitted.metrics.physical.padding_bytes,
            decoded.ledger.padding_bytes
        );

        let mut impossible = config(1);
        impossible.rate.max_artifact_bytes = 98;
        impossible.rate.max_resident_bytes = Some(42);
        assert!(matches!(
            fit(&weights, &diagonal, &impossible),
            Err(SaltV2Error::NoFeasibleAllocation { .. })
        ));
    }

    #[test]
    fn same_content_and_recipe_are_byte_and_receipt_deterministic() {
        let weights = weights(2);
        let diagonal = vec![1.0; weights.len()];
        let config = config(10_000);
        let left = fit(&weights, &diagonal, &config).expect("left fit");
        let right = fit(&weights, &diagonal, &config).expect("right fit");
        assert_eq!(left.package_bytes, right.package_bytes);
        assert_eq!(left.tensors, right.tensors);
        assert_eq!(left.metrics, right.metrics);
        assert_eq!(left.receipt, right.receipt);
    }

    #[test]
    fn representation_round_trip_contains_only_hard_trits_and_nonnegative_scales() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let fitted = fit(&weights, &diagonal, &config(10_000)).expect("fit");
        let decoded = tritium_format::salt_v2_package::read_salt_v2_package(&fitted.package_bytes)
            .expect("canonical package");
        for tensor in decoded.package.tensors() {
            for tile in tensor.tiles() {
                for plane in tile.planes() {
                    assert!(plane.scales().iter().all(|scale| scale.to_f32() >= 0.0));
                    assert!(
                        plane
                            .trits()
                            .iter()
                            .all(|trit| (-1..=1).contains(&trit.get()))
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_or_unimplemented_contracts_fail_closed() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];

        let mut malformed = config(10_000);
        malformed.group_size = 64;
        assert!(matches!(
            fit(&weights, &diagonal, &malformed),
            Err(SaltV2Error::UnsupportedReferenceGroupSize { got: 64 })
        ));

        let mut scale_only = config(10_000);
        scale_only.refinement = SaltV2Refinement::ScaleOnly { max_tokens: 8 };
        assert!(matches!(
            fit(&weights, &diagonal, &scale_only),
            Err(SaltV2Error::ExternalStageRequired {
                stage: SaltV2ExternalStage::ScaleOnlyRefinement
            })
        ));

        let mut transformed = config(10_000);
        transformed.transform_seed = Some(9);
        assert!(matches!(
            fit(&weights, &diagonal, &transformed),
            Err(SaltV2Error::ExternalStageRequired {
                stage: SaltV2ExternalStage::SignedRht
            })
        ));
    }

    #[test]
    fn every_tile_candidate_is_monotone_and_selected_shape_is_ragged_compatible() {
        let weights = weights(3);
        let diagonal = vec![1.0; weights.len()];
        let fitted = fit(&weights, &diagonal, &config(10_000)).expect("fit");
        for tile in 0..3 {
            let frontier: Vec<_> = fitted
                .metrics
                .tile_candidates
                .iter()
                .filter(|candidate| candidate.tile_index == tile)
                .collect();
            assert_eq!(frontier.len(), 3);
            assert!(frontier.windows(2).all(|pair| {
                pair[1].hessian_error <= pair[0].hessian_error + 1e-12
                    && pair[1].cumulative.serialized > pair[0].cumulative.serialized
            }));
        }
        assert_eq!(
            fitted.tensors[0]
                .tiles()
                .iter()
                .map(|tile| tile.planes().len() as u8)
                .collect::<Vec<_>>(),
            fitted.metrics.selected_plane_counts
        );
    }

    #[test]
    fn compact_and_near_lossless_runs_share_exact_plane_prefixes() {
        let weights = weights(2);
        let diagonal = vec![1.0; weights.len()];
        let mut compact_config = config(10_000);
        compact_config.max_planes = 2;
        let compact = fit(&weights, &diagonal, &compact_config).expect("compact fit");
        let near = fit(&weights, &diagonal, &config(10_000)).expect("near fit");
        assert!(
            compact
                .metrics
                .selected_plane_counts
                .iter()
                .all(|planes| *planes == 2)
        );
        assert!(
            near.metrics
                .selected_plane_counts
                .iter()
                .all(|planes| *planes == 3)
        );

        let near_package = SaltV2Package::new(SaltV2Codec::D2, near.tensors.clone())
            .expect("near semantic package");
        let requested = vec![
            compact
                .metrics
                .selected_plane_counts
                .iter()
                .map(|planes| usize::from(*planes))
                .collect::<Vec<_>>(),
        ];
        let derived = near_package
            .derive_prefix(&requested)
            .expect("derive exact compact prefix");
        assert_eq!(derived.tensors(), compact.tensors.as_slice());
    }

    #[test]
    fn curvature_receipt_binds_values_as_well_as_upstream_evidence_id() {
        let weights = weights(1);
        let left_diagonal = vec![1.0; weights.len()];
        let mut right_diagonal = left_diagonal.clone();
        right_diagonal[17] = 1.25;
        let recipe = config(10_000);
        let left = fit_with_curvature_evidence(&weights, &left_diagonal, [44; 32], &recipe)
            .expect("left curvature fit");
        let right = fit_with_curvature_evidence(&weights, &right_diagonal, [44; 32], &recipe)
            .expect("right curvature fit");
        assert_ne!(
            left.receipt.tensors[0].curvature_digest,
            right.receipt.tensors[0].curvature_digest
        );
        assert_ne!(left.receipt.receipt_id, right.receipt.receipt_id);
    }

    #[test]
    fn b3_uses_the_same_semantic_fit_with_exact_codec_pricing() {
        let weights = weights(1);
        let diagonal = vec![1.0; weights.len()];
        let mut recipe = config(10_000);
        recipe.packing = SaltV2Packing::B3;
        let fitted = fit(&weights, &diagonal, &recipe).expect("B3 fit");
        let decoded = tritium_format::salt_v2_package::read_salt_v2_package(&fitted.package_bytes)
            .expect("B3 package");
        assert_eq!(decoded.package.codec(), SaltV2Codec::B3);
        assert_eq!(
            decoded.ledger.total_bytes,
            fitted.metrics.physical.matrix_bytes
        );
        assert_eq!(
            decoded.ledger.payload_bytes,
            fitted.metrics.physical.trit_payload_bytes
        );
    }

    #[test]
    fn tiny_s34_plane_matches_exhaustive_f16_scale_and_pattern_oracle() {
        let target = [0.9375, -0.21875, 0.40625, -1.15625];
        let fitted =
            fit_s34_residual_plane(&target, JointFitMetric::Identity, 0.5, 4, 3).expect("S34 fit");

        let mut oracle = S34PlaneFit {
            scale: 0.0,
            trits: vec![0, -1, -1, -1],
            objective: f64::INFINITY,
        };
        for pattern in s34_patterns(4) {
            for bits in 1..=f16::MAX.to_bits() {
                let scale = f16::from_bits(bits).to_f32();
                let reconstruction = pattern
                    .iter()
                    .map(|trit| scale * f32::from(*trit))
                    .collect::<Vec<_>>();
                let objective =
                    reconstruction_objective(&target, &reconstruction, JointFitMetric::Identity);
                let candidate = S34PlaneFit {
                    scale,
                    trits: pattern.clone(),
                    objective,
                };
                if s34_plane_precedes(&candidate, &oracle) {
                    oracle = candidate;
                }
            }
        }

        assert_eq!(fitted.scale.to_bits(), oracle.scale.to_bits());
        assert_eq!(fitted.trits, oracle.trits);
        assert_eq!(fitted.objective, oracle.objective);
    }

    #[test]
    fn s34_scale_selection_handles_midpoint_ties_and_double_rounding() {
        let midpoint = optimal_s34_scale(&[1.001_464_8], &[1], JointFitMetric::Identity)
            .expect("midpoint scale");
        assert_eq!(f16::from_f32(midpoint).to_bits(), 0x3c01);

        let target = [-0.500_732_4, -0.500_732_4, -0.500_732_36, 0.0];
        let double_round = optimal_s34_scale(&target, &[-1, -1, -1, 0], JointFitMetric::Identity)
            .expect("double-round scale");
        assert_eq!(f16::from_f32(double_round).to_bits(), 0x3801);
    }

    #[test]
    fn ragged_s34_solver_obeys_canonical_physical_quartet_semantics() {
        for logical_len in 1..=3 {
            let target = [0.75, -0.5, 0.25][..logical_len].to_vec();
            let fitted = fit_s34_residual_plane(&target, JointFitMetric::Identity, 0.5, 4, 2)
                .expect("ragged S34 fit");
            let logical_zeros = fitted.trits.iter().filter(|trit| **trit == 0).count();
            assert!(logical_zeros <= 1);

            let mut physical = fitted.trits;
            if logical_zeros == 0 {
                physical.push(0);
            }
            physical.resize(4, -1);
            assert_eq!(physical.iter().filter(|trit| **trit == 0).count(), 1);
        }
    }

    #[test]
    fn dense_curvature_s34_refinement_remains_monotone_and_deterministic() {
        let target = [0.8, -0.7, 0.3, -0.2, 0.9, -0.4, 0.1, -0.6];
        let mut matrix = vec![0.0; target.len() * target.len()];
        for row in 0..target.len() {
            matrix[row * target.len() + row] = 1.0;
            if row + 1 < target.len() {
                matrix[row * target.len() + row + 1] = 0.125;
                matrix[(row + 1) * target.len() + row] = 0.125;
            }
        }
        let dense = DensePsdMetric::new(target.len(), &matrix).expect("dense PSD metric");
        let config = JointFitConfig {
            planes: 3,
            max_iterations: 4,
            ridge: 1e-12,
            em_restarts: 3,
            ridge_condition_limit: 1_000_000.0,
            scale_precision: ScalePrecision::F16,
        };
        let left = fit_progressive_s34(&target, JointFitMetric::Dense(&dense), config)
            .expect("left dense S34 fit");
        let right = fit_progressive_s34(&target, JointFitMetric::Dense(&dense), config)
            .expect("right dense S34 fit");
        assert_eq!(left.scales, right.scales);
        assert_eq!(left.trits, right.trits);
        assert!(left.trits.iter().all(|plane| {
            plane
                .chunks_exact(4)
                .all(|group| group.iter().filter(|trit| **trit == 0).count() == 1)
        }));
        let objectives = s34_prefix_objectives(
            &target,
            JointFitMetric::Dense(&dense),
            &left.scales,
            &left.trits,
        )
        .expect("dense prefix objectives");
        assert!(
            objectives
                .windows(2)
                .all(|pair| { pair[1] <= pair[0] + 1e-12f64.max(pair[0].abs() * 1e-12) })
        );
    }

    #[test]
    fn s34_model_fit_is_deterministic_progressive_and_physically_exact() {
        let weights = (0..258)
            .map(|index| ((index % 29) as f32 - 14.0) / 17.0)
            .collect::<Vec<_>>();
        let diagonal = (0..weights.len())
            .map(|index| 0.5 + (index % 7) as f32 / 8.0)
            .collect::<Vec<_>>();
        let mut recipe = config(10_000);
        recipe.packing = SaltV2Packing::S34;
        recipe.em_restarts = 2;
        recipe.coordinate_sweeps = 4;

        let left = fit(&weights, &diagonal, &recipe).expect("left S34 fit");
        let right = fit(&weights, &diagonal, &recipe).expect("right S34 fit");
        assert_eq!(left.package_bytes, right.package_bytes);
        assert_eq!(left.metrics, right.metrics);
        assert_eq!(left.receipt, right.receipt);
        assert!(left.metrics.tile_candidates.chunks(3).all(|frontier| {
            frontier
                .windows(2)
                .all(|pair| pair[1].hessian_error <= pair[0].hessian_error + 1e-12)
        }));

        let decoded = tritium_format::salt_v2_package::read_salt_v2_package(&left.package_bytes)
            .expect("canonical S34 package");
        assert_eq!(decoded.package.codec(), SaltV2Codec::S34);
        assert_eq!(
            decoded.ledger.total_bytes,
            left.metrics.physical.matrix_bytes
        );
        assert_eq!(
            decoded.ledger.payload_bytes,
            left.metrics.physical.trit_payload_bytes
        );
        for tile in decoded.package.tensors()[0].tiles() {
            for plane in tile.planes() {
                for group in plane.trits().chunks(4) {
                    let logical_zeros = group.iter().filter(|trit| trit.is_zero()).count();
                    if group.len() == 4 {
                        assert_eq!(logical_zeros, 1);
                    } else {
                        assert!(logical_zeros <= 1);
                        let mut physical = group.iter().map(|trit| trit.get()).collect::<Vec<_>>();
                        if logical_zeros == 0 {
                            physical.push(0);
                        }
                        physical.resize(4, -1);
                        assert_eq!(physical.iter().filter(|trit| **trit == 0).count(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn all_zero_s34_model_uses_only_positive_scales_and_a_valid_frontier() {
        let weights = vec![0.0; 256];
        let diagonal = vec![1.0; weights.len()];
        let mut recipe = config(10_000);
        recipe.packing = SaltV2Packing::S34;
        let fitted = fit(&weights, &diagonal, &recipe).expect("zero S34 model fit");

        assert!(
            fitted.tensors[0].tiles()[0]
                .planes()
                .iter()
                .flat_map(|plane| plane.scales())
                .all(|scale| scale.to_f32() > 0.0)
        );
        assert!(fitted.metrics.tile_candidates.windows(2).all(|pair| {
            pair[1].hessian_error < pair[0].hessian_error
                && pair[1].cumulative.serialized > pair[0].cumulative.serialized
        }));
        tritium_format::salt_v2_package::read_salt_v2_package(&fitted.package_bytes)
            .expect("deployable zero S34 package");

        let unbounded = SaltV2Config {
            packing: SaltV2Packing::S34,
            ..SaltV2Config::default()
        };
        let unbounded_fit = fit(&weights, &diagonal, &unbounded)
            .expect("unbounded zero S34 model fit must not overflow sentinels");
        assert!(
            unbounded_fit
                .metrics
                .selected_plane_counts
                .iter()
                .all(|planes| *planes <= 2)
        );
    }

    #[test]
    fn zero_gain_planes_are_removed_from_the_selected_pareto_point() {
        let weights = vec![0.0; 256];
        let diagonal = vec![1.0; weights.len()];
        let fitted = fit(&weights, &diagonal, &config(10_000)).expect("zero fit");
        assert_eq!(fitted.metrics.selected_plane_counts, vec![1]);
        assert_eq!(fitted.metrics.hessian_error, 0.0);
    }
}
