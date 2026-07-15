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

const REFERENCE_SOLVER_VERSION: &str = "tritium-salt-v2-reference-model-fit-v1";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 model fit receipt v1";
const RECIPE_HASH_CONTEXT: &str = "tritium salt v2 model fit recipe v1";
const SOURCE_TENSOR_HASH_CONTEXT: &str = "tritium salt v2 source tensor v1";
const CURVATURE_HASH_CONTEXT: &str = "tritium salt v2 bound curvature artifact v1";

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
#[derive(Clone, Copy, Debug)]
pub struct CurvatureArtifact<'a> {
    kind: SaltV2Curvature,
    evidence_digest: [u8; 32],
    content_digest: [u8; 32],
    values: CurvatureValues<'a>,
}

impl<'a> CurvatureArtifact<'a> {
    /// Bind a per-weight empirical-Fisher diagonal to its canonical artifact digest.
    #[must_use]
    pub fn diagonal_fisher(evidence_digest: [u8; 32], diagonal: &'a [f32]) -> Self {
        let values = CurvatureValues::Diagonal(diagonal);
        Self {
            kind: SaltV2Curvature::DiagonalFisher,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::DiagonalFisher,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise dense input-Hessian blocks to their canonical artifact digest.
    #[must_use]
    pub fn input_hessian(evidence_digest: [u8; 32], groups: &'a [DensePsdMetric]) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::InputHessian,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::InputHessian,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise dense guided-Fisher blocks to their canonical artifact digest.
    #[must_use]
    pub fn guided_fisher(evidence_digest: [u8; 32], groups: &'a [DensePsdMetric]) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::GuidedFisher,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::GuidedFisher,
                evidence_digest,
                values,
            ),
            values,
        }
    }

    /// Bind groupwise forward-KL Kronecker blocks to their canonical artifact digest.
    #[must_use]
    pub fn forward_kl_kronecker(evidence_digest: [u8; 32], groups: &'a [DensePsdMetric]) -> Self {
        let values = CurvatureValues::DenseGroups(groups);
        Self {
            kind: SaltV2Curvature::ForwardKlKronecker,
            evidence_digest,
            content_digest: bound_curvature_digest(
                SaltV2Curvature::ForwardKlKronecker,
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
    /// S34-constrained discrete fitting.
    StructuredS34Fit,
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
    /// A curvature artifact had the wrong diagonal or dense-block geometry.
    CurvatureGeometry {
        /// Tensor ordinal.
        tensor: usize,
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
            Self::CurvatureGeometry { tensor } => {
                write!(
                    formatter,
                    "tensor {tensor} curvature geometry is incompatible"
                )
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

/// Fit a small whole model with the deterministic CPU reference solver.
///
/// Every 256-coefficient allocation tile is jointly fit once at P=3. A deterministic progressive
/// plane order produces exact P=1/P=2 prefixes, so separately budgeted compact and near-lossless
/// runs slice identical trits and scales instead of refitting them. Each plane retains one
/// non-negative f16 scale per group128. The exact Pareto dynamic program in the shared allocator
/// selects a whole-model point under transformed integer package, artifact, and resident ceilings;
/// the canonical package writer then remeasures and verifies every component. The returned
/// representation has no zero point, bias, codebook, or floating residual.
///
/// This function intentionally implements only the pure-PTQ reference seam. Signed RHT,
/// S34-constrained assignment, cached-output reconstruction, scale-only recovery, and short PV
/// recovery return [`SaltV2Error::ExternalStageRequired`] rather than silently falling back to a
/// weaker algorithm. The receipt reports feedback/output reconstruction as false.
///
/// # Errors
/// Rejects malformed recipes or tensors, missing/mismatched curvature evidence, unavailable
/// production stages, failed joint fits, non-monotone candidate curves, infeasible hard ceilings,
/// accounting overflow, and any canonical package validation failure.
pub fn fit_salt_v2_model(
    input: SaltV2ModelFitInput<'_>,
    config: &SaltV2Config,
) -> Result<SaltV2ModelFitResult, SaltV2Error> {
    validate_config(config)?;
    validate_external_stages(config)?;
    let quantized_parameters = validate_model_input(&input, config)?;
    let mut work = Vec::new();
    work.try_reserve_exact(input.tensors.len())
        .map_err(|_| SaltV2Error::AccountingOverflow)?;
    let mut tile_candidates = Vec::new();
    for (tensor_index, tensor) in input.tensors.iter().enumerate() {
        let tensor_work = fit_tensor_candidates(tensor_index, tensor, config)?;
        for frontier in &tensor_work.candidates {
            tile_candidates.extend(frontier.iter().map(|candidate| candidate.metrics));
        }
        work.push(tensor_work);
    }
    let (serialized_fixed_bytes, resident_fixed_bytes) =
        fixed_package_bytes(&work, config.packing.codec())?;

    let overhead_resident = input
        .physical
        .preserved_resident_bytes
        .checked_add(input.physical.required_runtime_shadow_bytes)
        .ok_or(SaltV2Error::AccountingOverflow)?;
    let package_artifact_ceiling = config
        .rate
        .max_artifact_bytes
        .checked_sub(input.physical.preserved_artifact_bytes)
        .ok_or_else(|| no_feasible(config, 0))?;
    let package_matrix_ceiling = config.rate.max_matrix_bytes.min(package_artifact_ceiling);
    let aligned_package_ceiling = package_matrix_ceiling
        / u64::try_from(SALT_V2_PACKAGE_ALIGNMENT).map_err(|_| SaltV2Error::AccountingOverflow)?
        * u64::try_from(SALT_V2_PACKAGE_ALIGNMENT).map_err(|_| SaltV2Error::AccountingOverflow)?;
    let package_resident_ceiling = match config.rate.max_resident_bytes {
        Some(maximum) => maximum
            .checked_sub(overhead_resident)
            .ok_or_else(|| no_feasible(config, aligned_package_ceiling))?,
        None => u64::MAX
            .checked_sub(overhead_resident)
            .ok_or(SaltV2Error::AccountingOverflow)?,
    };

    let allocator_frontiers = allocator_candidates(&work, config)?;
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
        PhysicalAllocError::BudgetTooSmall { .. } => no_feasible(config, aligned_package_ceiling),
        other => SaltV2Error::Allocation(other),
    })?
    .near_lossless;

    let mut selected_plane_counts = allocation.plane_counts;
    let mut frontier_index = 0usize;
    for tensor in &work {
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

    for tensor_work in &work {
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
        .checked_add(input.physical.preserved_artifact_bytes)
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
        return Err(no_feasible(config, aligned_package_ceiling));
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
    let artifact_bpw = exact_bpw(artifact_bytes, input.physical.total_model_parameters)?;
    let resident_bpw = exact_bpw(resident_bytes, input.physical.total_model_parameters)?;
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
            preserved_artifact_bytes: input.physical.preserved_artifact_bytes,
            preserved_resident_bytes: input.physical.preserved_resident_bytes,
            required_runtime_shadow_bytes: input.physical.required_runtime_shadow_bytes,
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
    let activation_digest = input.activations.digest().into_bytes();
    let recipe_id = recipe_digest(config);
    let receipt_id = receipt_digest(
        input.source_model_id,
        activation_digest,
        recipe_id,
        &tensor_receipts,
        package_id,
        input.physical,
    );
    let receipt = SaltV2ModelFitReceipt {
        solver_version: REFERENCE_SOLVER_VERSION,
        source_model_id: input.source_model_id,
        activation_digest,
        recipe_id,
        tensors: tensor_receipts,
        package_id,
        physical: input.physical,
        receipt_id,
        track: SaltV2FitTrack::Ptq,
        feedback_applied: false,
        output_reconstruction_applied: false,
    };

    Ok(SaltV2ModelFitResult {
        tensors,
        package_bytes: encoded.bytes,
        config: *config,
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
    if config.packing == SaltV2Packing::S34 {
        return Err(SaltV2Error::ExternalStageRequired {
            stage: SaltV2ExternalStage::StructuredS34Fit,
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
        let fitted = fit_joint_ternary(
            &tensor.weights[group_start..group_end],
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
        let weights = &tensor.weights[group_start..group_end];
        let order = progressive_plane_order(weights, metric, &fitted.scales, &fitted.trits).ok_or(
            SaltV2Error::NonMonotoneCandidate {
                tensor: tensor_index,
                tile: tile_index,
                planes: 2,
            },
        )?;
        let mut reconstruction = vec![0.0f32; weights.len()];
        for prefix in 0..FULL_PLANES {
            let source_plane = order[prefix];
            let scale = fitted.scales[source_plane];
            for (value, trit) in reconstruction.iter_mut().zip(&fitted.trits[source_plane]) {
                *value += scale * f32::from(*trit);
            }
            hessian_errors[prefix] += reconstruction_objective(weights, &reconstruction, metric);
            frobenius_errors[prefix] += weights
                .iter()
                .zip(&reconstruction)
                .map(|(source, reconstructed)| {
                    let residual = f64::from(*source) - f64::from(*reconstructed);
                    residual * residual
                })
                .sum::<f64>();
            plane_trits[prefix].extend_from_slice(&fitted.trits[source_plane]);
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
) -> Option<[usize; 3]> {
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = None::<([usize; 3], [f64; 3])>;
    for order in ORDERS {
        let mut reconstruction = vec![0.0f32; weights.len()];
        let mut objectives = [0.0f64; 3];
        for (prefix, &plane) in order.iter().enumerate() {
            for (value, trit) in reconstruction.iter_mut().zip(&trits[plane]) {
                *value += scales[plane] * f32::from(*trit);
            }
            objectives[prefix] = reconstruction_objective(weights, &reconstruction, metric);
        }
        let monotone = objectives.windows(2).all(|pair| {
            let tolerance = 1e-12f64.max(pair[0].abs() * 1e-12);
            pair[1] <= pair[0] + tolerance
        });
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
                    let prohibited = PhysicalBytes {
                        serialized: config.rate.max_matrix_bytes.saturating_add(1),
                        resident: config
                            .rate
                            .max_resident_bytes
                            .unwrap_or(config.rate.max_matrix_bytes)
                            .saturating_add(1),
                    };
                    allocator.push(PlaneCandidate {
                        planes: (plane_index + 1) as u8,
                        byte_delta: ByteDelta::measured(prohibited, prohibited),
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
    evidence_digest: [u8; 32],
    values: CurvatureValues<'_>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(CURVATURE_HASH_CONTEXT);
    hasher.update(&[kind.tag()]);
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

fn receipt_digest(
    source_model_id: ModelId,
    activation_digest: [u8; 32],
    recipe_id: [u8; 32],
    tensors: &[SaltV2TensorFitReceipt],
    package_id: PackageId,
    physical: SaltV2ModelPhysicalInput,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    hasher.update(source_model_id.as_bytes());
    hasher.update(&activation_digest);
    hasher.update(&recipe_id);
    write_len_hash(&mut hasher, tensors.len());
    for tensor in tensors {
        write_len_hash(&mut hasher, tensor.name.len());
        hasher.update(tensor.name.as_bytes());
        hasher.update(&tensor.source_digest);
        hasher.update(&tensor.curvature_digest);
        write_len_hash(&mut hasher, tensor.plane_counts.len());
        hasher.update(&tensor.plane_counts);
    }
    hasher.update(package_id.as_bytes());
    hasher.update(&physical.total_model_parameters.to_le_bytes());
    hasher.update(&physical.preserved_artifact_bytes.to_le_bytes());
    hasher.update(&physical.preserved_resident_bytes.to_le_bytes());
    hasher.update(&physical.required_runtime_shadow_bytes.to_le_bytes());
    hasher.update(&[SaltV2FitTrack::Ptq as u8, 0, 0]);
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
        let spec = ActivationCacheSpec::new(
            0,
            "weight.input",
            1,
            1,
            ActivationDType::Float32,
            ActivationDigest::from_bytes([7; 32]),
            1,
        )
        .expect("activation spec");
        let chunk = ActivationChunk::new(&spec, 0, 1, vec![1.0], vec![true], vec![1])
            .expect("activation chunk");
        let mut builder = ActivationCacheBuilder::new(spec);
        builder.ingest(chunk).expect("activation ingest");
        builder.finalize().expect("activation cache")
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
        let curvature = CurvatureArtifact::diagonal_fisher(evidence_digest, diagonal);
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
                activations: &cache,
                source_model_id: source_model_id(),
                physical,
            },
            config,
        )
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

        let mut s34 = config(10_000);
        s34.packing = SaltV2Packing::S34;
        assert!(matches!(
            fit(&weights, &diagonal, &s34),
            Err(SaltV2Error::ExternalStageRequired {
                stage: SaltV2ExternalStage::StructuredS34Fit
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
    fn zero_gain_planes_are_removed_from_the_selected_pareto_point() {
        let weights = vec![0.0; 256];
        let diagonal = vec![1.0; weights.len()];
        let fitted = fit(&weights, &diagonal, &config(10_000)).expect("zero fit");
        assert_eq!(fitted.metrics.selected_plane_counts, vec![1]);
        assert_eq!(fitted.metrics.hessian_error, 0.0);
    }
}
