# Generated from tritium-schema Rust types; do not edit.
from __future__ import annotations

from typing import Any, Literal, Union
from typing_extensions import NotRequired, TypeAlias, TypedDict

class BasisVariant1(TypedDict):
    Hadamard: BasisVariant1Hadamard

class BasisVariant1Hadamard(TypedDict):
    block: int

class BasisVariant2(TypedDict):
    SignedRht: BasisVariant2SignedRht

class BasisVariant2SignedRht(TypedDict):
    block: int
    domain: int
    seed: int

class CalibratedQuantActionVariant1(TypedDict):
    Fit: CalibratedQuantActionVariant1Fit

class CalibratedQuantActionVariant1Fit(TypedDict):
    metric: DirectFitMetric
    plan: DirectFitPlan

class CalibratedQuantActionVariant2(TypedDict):
    CalibratedFit: CalibratedQuantActionVariant2CalibratedFit

class CalibratedQuantActionVariant2CalibratedFit(TypedDict):
    plan: CalibratedFitPlan

class CalibrationCaptureInputVariant0(TypedDict):
    Tokens: CalibrationCaptureInputVariant0Tokens

class CalibrationCaptureInputVariant0Tokens(TypedDict):
    windows: list[list[int]]

class CalibrationCaptureInputVariant1(TypedDict):
    RawText: CalibrationCaptureInputVariant1RawText

class CalibrationCaptureInputVariant1RawText(TypedDict):
    windows: list[str]

class CalibrationPartitionVariant1(TypedDict):
    Groups: CalibrationPartitionVariant1Groups

class CalibrationPartitionVariant1Groups(TypedDict):
    columns: int

class DirectFitMetricVariant1(TypedDict):
    InputGramF64: DirectFitMetricVariant1InputGramF64

class DirectFitMetricVariant1InputGramF64(TypedDict):
    content: FitInputDigest

class DirectFitProfileVariant0(TypedDict):
    ActivationAwareLadder: LadderFitOptions

class DirectFitProfileVariant1(TypedDict):
    JointIndependentGroups: JointGroupFitOptions

class DirectFitProfileVariant2(TypedDict):
    JointFeedbackGroups: JointFeedbackFitOptions

class LevelFitVariant0(TypedDict):
    Direct: LevelFitVariant0Direct

class LevelFitVariant0Direct(TypedDict):
    evidence: list[EvidenceReference]

class LevelFitVariant1(TypedDict):
    ImportedPrefixOf: LevelFitVariant1ImportedPrefixOf

class LevelFitVariant1ImportedPrefixOf(TypedDict):
    level: str

class LoadedSourceIdentityVariant0(TypedDict):
    Trit: LoadedSourceIdentityVariant0Trit

class LoadedSourceIdentityVariant0Trit(TypedDict):
    level: str
    model_id: ModelId
    package_id: PackageId

class LoadedSourceIdentityVariant1(TypedDict):
    HuggingFace: LoadedSourceIdentityVariant1HuggingFace

class LoadedSourceIdentityVariant1HuggingFace(TypedDict):
    reason: UnknownReason

class OriginalSalienceCaptureInputVariant0(TypedDict):
    Tokens: OriginalSalienceCaptureInputVariant0Tokens

class OriginalSalienceCaptureInputVariant0Tokens(TypedDict):
    windows: list[list[int]]

class OriginalSalienceCaptureInputVariant1(TypedDict):
    RawText: OriginalSalienceCaptureInputVariant1RawText

class OriginalSalienceCaptureInputVariant1RawText(TypedDict):
    windows: list[str]

class OwnerQuantActionVariant1(TypedDict):
    Fit: OwnerQuantActionVariant1Fit

class OwnerQuantActionVariant1Fit(TypedDict):
    metric: DirectFitMetric
    plan: DirectFitPlan

class OwnerQuantActionVariant2(TypedDict):
    CalibratedFit: OwnerQuantActionVariant2CalibratedFit

class OwnerQuantActionVariant2CalibratedFit(TypedDict):
    plan: CalibratedFitPlan

class OwnerQuantActionVariant3(TypedDict):
    OwnerFit: OwnerQuantActionVariant3OwnerFit

class OwnerQuantActionVariant3OwnerFit(TypedDict):
    plan: OwnerScoreFitPlan

class PlaneCountsVariant0(TypedDict):
    Uniform: int

class PlaneCountsVariant1(TypedDict):
    PerTile: list[int]

class PlaneRelationVariant1(TypedDict):
    Tied: PlaneRelationVariant1Tied

class PlaneRelationVariant1Tied(TypedDict):
    den: int
    num: int

class QuantActionVariant1(TypedDict):
    Fit: QuantActionVariant1Fit

class QuantActionVariant1Fit(TypedDict):
    metric: DirectFitMetric
    plan: DirectFitPlan

class TensorSemanticMetaVariant0(TypedDict):
    Dense: TensorSemanticMetaVariant0Dense

class TensorSemanticMetaVariant0Dense(TypedDict):
    precision: ScalePrecision
    shape: list[int]

class TensorSemanticMetaVariant1(TypedDict):
    Additive: TensorSemanticMetaVariant1Additive

class TensorSemanticMetaVariant1Additive(TypedDict):
    basis: Basis
    cols: int
    group: int
    law: ScaleLaw
    planes: PlaneCounts
    rows: int
    tile: int

class TensorStorageVariant0(TypedDict):
    Dense: TensorStorageVariant0Dense

class TensorStorageVariant0Dense(TypedDict):
    precision: ScalePrecision
    shape: list[int]

class TensorStorageVariant1(TypedDict):
    Additive: TensorStorageVariant1Additive

class TensorStorageVariant1Additive(TypedDict):
    layout: AdditiveLayout
    planes: PlaneCounts

class TransportVariant1(TypedDict):
    JointHuffman: TransportVariant1JointHuffman

class TransportVariant1JointHuffman(TypedDict):
    block: int

class VerdictVariant2(TypedDict):
    Unknown: VerdictVariant2Unknown

class VerdictVariant2Unknown(TypedDict):
    reason: UnknownReason

class AdditiveLayout(TypedDict):
    allocation: PlaneAllocation
    basis: Basis
    codec: PlaneCodec
    cols: int
    group: int
    law: ScaleLaw
    max_planes: int
    rows: int
    tile: int
    transport: Transport

class AdmittedLaw(TypedDict):
    law: ScaleLaw
    max_planes: int

Basis: TypeAlias = Union[BasisVariant1, BasisVariant2, Literal["Identity"]]

BlobId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class BlobReference(TypedDict):
    id: BlobId
    shard: NotRequired[Union[None, str]]

class CalibratedFitPlan(TypedDict):
    cache: CalibrationCacheBinding
    damping: float
    direct: DirectFitPlan
    max_numeric_bytes: int
    partition: CalibrationPartition

class CalibratedLevelsFitted(TypedDict):
    factors: CalibrationFactors
    outputs: list[DirectFitOutput]
    plan: CalibratedFitPlan
    rayon_threads: int
    source_f32: FitInputDigest

CalibratedQuantAction: TypeAlias = Union[CalibratedQuantActionVariant1, CalibratedQuantActionVariant2, Literal["KeepDense"]]

class CalibratedQuantPlan(TypedDict):
    architecture: str
    assets: list[str]
    levels: list[str]
    limits: QuantLimits
    max_cache_package_bytes: int
    max_cache_payload_bytes: int
    original_source: OriginalSourceId
    tensors: list[CalibratedQuantTensorPlan]

class CalibratedQuantTensorPlan(TypedDict):
    action: CalibratedQuantAction
    source: QuantSourceTensor

class CalibratedQuantizationCompleted(TypedDict):
    original_source: OriginalSourceId
    output: QuantizationCompleted

class CalibratedQuantizationStarted(TypedDict):
    plan: CalibratedQuantPlan

class CalibrationCacheBinding(TypedDict):
    objective: BlobId
    payload_bytes: int
    request: CalibrationCacheRequest
    samples: int
    source: CalibrationSource
    tokenizer: BlobId
    windows: int

CalibrationCacheId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class CalibrationCacheLimits(TypedDict):
    max_payload_bytes: int
    max_samples: int
    max_windows: int

class CalibrationCacheManifest(TypedDict):
    binding: CalibrationCacheBinding
    window_samples: list[int]

class CalibrationCacheRequest(TypedDict):
    consumer: CalibrationConsumer
    limits: CalibrationCacheLimits
    model: OriginalSourceId
    output: CalibrationOutput
    sample_start: int
    tensor: QuantSourceTensor

class CalibrationCaptureCompleted(TypedDict):
    caches: list[CalibrationCacheBinding]
    windows: list[list[int]]

CalibrationCaptureInput: TypeAlias = Union[CalibrationCaptureInputVariant0, CalibrationCaptureInputVariant1]

class CalibrationCaptureLimits(TypedDict):
    max_config_bytes: int
    max_dense_bytes: int
    max_text_bytes: int
    max_tokenizer_config_bytes: int
    max_total_payload_bytes: int
    max_window_tokens: int
    per_cache: CalibrationCacheLimits

class CalibrationCapturePlan(TypedDict):
    input: CalibrationCaptureInput
    limits: CalibrationCaptureLimits
    profile: CalibrationCaptureProfile
    source: OriginalSourceId
    stage_package: PackageId
    tensors: list[QuantSourceTensor]
    tokenizer: BlobId
    tokenizer_config: NotRequired[Union[BlobId, None]]
    transform: EvidenceReference

CalibrationCaptureProfile: TypeAlias = Literal["NextTokenWindowMeanK1F32V1"]

class CalibrationCaptureStarted(TypedDict):
    plan: CalibrationCapturePlan

CalibrationConsumer: TypeAlias = Union[Literal["GatherIdentity"], Literal["Matmul"]]

class CalibrationFactor(TypedDict):
    content: FitInputDigest
    dimension: int
    samples: int
    selected: int
    selection: FitInputDigest
    total_weight: float

class CalibrationFactors(TypedDict):
    inputs: list[CalibrationFactor]
    output: NotRequired[Union[CalibrationFactor, None]]
    plan: CalibrationPlan

CalibrationOutput: TypeAlias = Union[Literal["DenseFisher"], Literal["IndexedFisher"], Literal["None"]]

CalibrationPartition: TypeAlias = Union[CalibrationPartitionVariant1, Literal["Full"]]

class CalibrationPlan(TypedDict):
    consumer: CalibrationConsumer
    damping: float
    max_numeric_bytes: int
    output: CalibrationOutput
    partition: CalibrationPartition
    sample_start: int
    source: CalibrationSource
    tensor: QuantSourceTensor

class CalibrationSource(TypedDict):
    cache: FitInputDigest
    model: FitInputDigest
    tokens: FitInputDigest

class CapturedSalienceConversionManifest(TypedDict):
    capture: EvidenceReference
    lineage: SalienceConversionManifest

class DirectFitLevel(TypedDict):
    name: str
    planes: int

class DirectFitLimits(TypedDict):
    max_metric_entries: int
    max_output_trits: int
    max_weights: int

DirectFitMetric: TypeAlias = Union[DirectFitMetricVariant1, Literal["Identity"]]

class DirectFitOutput(TypedDict):
    content: SemanticTensorDigest
    level: str
    planes: int

class DirectFitPlan(TypedDict):
    layout: AdditiveLayout
    levels: list[DirectFitLevel]
    limits: DirectFitLimits
    profile: DirectFitProfile

DirectFitProfile: TypeAlias = Union[DirectFitProfileVariant0, DirectFitProfileVariant1, DirectFitProfileVariant2]

class DirectLevelsFitted(TypedDict):
    metric: DirectFitMetric
    outputs: list[DirectFitOutput]
    plan: DirectFitPlan
    rayon_threads: int
    source_f32: FitInputDigest
    tensor: str

class EvidenceEnvelope(TypedDict):
    digest: str
    parent: NotRequired[Union[None, str]]
    payload: Any
    run: str
    schema: str
    seq: int
    span: str
    t: int
    v: int

class EvidenceReference(TypedDict):
    root: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]
    run: str
    span: str

FitInputDigest: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class JointFeedbackFitOptions(TypedDict):
    damp: float
    decay: float
    decay_ramp: bool
    fitter: JointGroupFitOptions
    precision: WorkPrecision

class JointGroupFitOptions(TypedDict):
    em_restarts: int
    max_iterations: int
    modulated_relay: bool
    ridge: float
    ridge_condition_limit: float
    softened_relay: bool

class LadderFitOptions(TypedDict):
    damp: float
    decay: float
    decay_ramp: bool
    grid: int
    scale_refit: LadderScaleRefit
    search_sweeps: int

LadderScaleRefit: TypeAlias = Union[Literal["InLoop"], Literal["None"], Literal["PostPass"]]

LayoutError: TypeAlias = Union[Literal["BasisBlockDoesNotDivideColumns"], Literal["EmptyDimensions"], Literal["EmptyPlaneStack"], Literal["EmptyTransportBlock"], Literal["InvalidBasisBlock"], Literal["InvalidScaleRatio"], Literal["PlaneCountExceedsLaw"], Literal["UnsupportedGroup"], Literal["UnsupportedLaw"], Literal["UnsupportedTileSize"]]

class LegacyModelIdentity(TypedDict):
    id: str
    level: str

class LegacyModelImportBoundPrepared(TypedDict):
    companion_assets: list[NamedAssetDigest]
    companion_source: OriginalSourceId
    companion_tensors: list[QuantSourceBinding]
    legacy_id: str
    manifest_bytes: int
    manifest_digest: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class LegacyModelImportPrepared(TypedDict):
    companion_assets: list[NamedAssetDigest]
    companion_source: OriginalSourceId
    companion_tensors: list[QuantSourceBinding]
    legacy_id: str
    manifest: ModelManifest

class LegacySaltV2AssetDescriptor(TypedDict):
    bytes: int
    file: str
    package_id: str

class LegacySaltV2BundleDescriptor(TypedDict):
    admission_id: str
    artifact_kind: str
    campaign_id: str
    complete_model: bool
    completion_id: str
    hf_assets: list[LegacySaltV2AssetDescriptor]
    official_payload_authenticated: bool
    packing: str
    preserved: LegacySaltV2PreservedDescriptor
    profiles: LegacySaltV2ProfileSet
    schema_version: int
    selection_id: str
    source_identity_status: str
    source_model_id: str
    source_revision: str

class LegacySaltV2PreservedDescriptor(TypedDict):
    file: str
    package_id: str
    payload_bytes: int
    serialized_bytes: int
    tensors: int

class LegacySaltV2ProfileDescriptor(TypedDict):
    file: str
    package_id: str
    resident_bytes: int
    serialized_bytes: int

LegacySaltV2ProfileSet = TypedDict("LegacySaltV2ProfileSet", {
    "compact-v1": LegacySaltV2ProfileDescriptor,
    "near-lossless-v1": LegacySaltV2ProfileDescriptor,
})

class Level(TypedDict):
    fit: LevelFit
    name: str
    tensors: list[TensorMeta]

LevelFit: TypeAlias = Union[LevelFitVariant0, LevelFitVariant1, Literal["Imported"]]

LoadedSourceIdentity: TypeAlias = Union[LoadedSourceIdentityVariant0, LoadedSourceIdentityVariant1]

class ManifestAsset(TypedDict):
    blob: BlobReference
    name: str

ModelId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class ModelLoaded(TypedDict):
    architecture: str
    backend: str
    layer_count: int
    physical_device: str
    source: LoadedSourceIdentity
    vocab_size: int

class ModelManifest(TypedDict):
    architecture: str
    assets: list[ManifestAsset]
    legacy_ids: list[LegacyModelIdentity]
    levels: list[Level]
    schema: SchemaId
    shards: list[ModelShard]

class ModelSemanticMeta(TypedDict):
    architecture: str
    assets: list[NamedAssetDigest]
    tensors: list[NamedTensorDigest]

class ModelShard(TypedDict):
    name: str
    package: PackageId

class NamedAssetDigest(TypedDict):
    content: BlobId
    name: str

class NamedTensorDigest(TypedDict):
    content: SemanticTensorDigest
    name: str

class OriginalCapturedSalienceConversionManifest(TypedDict):
    lineage: CapturedSalienceConversionManifest
    original_capture: EvidenceReference

class OriginalCapturedSalienceSourceStageManifest(TypedDict):
    lineage: SalienceSourceStageManifest
    original_capture: EvidenceReference

class OriginalSalienceCaptureCompleted(TypedDict):
    outputs: list[OriginalSalienceCaptureOutput]
    windows: list[list[int]]

OriginalSalienceCaptureInput: TypeAlias = Union[OriginalSalienceCaptureInputVariant0, OriginalSalienceCaptureInputVariant1]

class OriginalSalienceCaptureLimits(TypedDict):
    max_asset_bytes: int
    max_config_bytes: int
    max_dense_bytes: int
    max_moment_bytes: int
    max_source_bytes: int
    max_text_bytes: int
    max_tokenizer_config_bytes: int
    max_total_tokens: int
    max_window_tokens: int
    max_windows: int

class OriginalSalienceCaptureOutput(TypedDict):
    moments: SalienceMomentBinding
    operation: str

class OriginalSalienceCapturePair(TypedDict):
    channels: int
    inverse: str
    inverse_axis: SalienceInverseAxis
    name: str
    projections: list[str]

class OriginalSalienceCapturePlan(TypedDict):
    architecture: str
    assets: list[NamedAssetDigest]
    input: OriginalSalienceCaptureInput
    limits: OriginalSalienceCaptureLimits
    pairs: list[OriginalSalienceCapturePair]
    profile: OriginalSalienceCaptureProfile
    source: OriginalSourceId
    tensors: list[QuantSourceBinding]

OriginalSalienceCaptureProfile: TypeAlias = Literal["AllTokenF64SecondMomentV1"]

class OriginalSalienceCaptureStarted(TypedDict):
    plan: OriginalSalienceCapturePlan

OriginalSourceId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class OwnerContextBinding(TypedDict):
    content: BlobId
    meta: OwnerContextMeta

OwnerContextDtype: TypeAlias = Union[Literal["Bf16"], Literal["Bool"], Literal["C128"], Literal["C64"], Literal["F16"], Literal["F32"], Literal["F64"], Literal["I16"], Literal["I32"], Literal["I64"], Literal["I8"], Literal["U8"]]

class OwnerContextMeta(TypedDict):
    conjugated: bool
    dtype: OwnerContextDtype
    name: str
    negated: bool
    shape: list[int]

class OwnerInvocationBinding(TypedDict):
    adjoint: BlobId
    input: BlobId
    meta: OwnerInvocationMeta

class OwnerInvocationMeta(TypedDict):
    ordinal: int
    paths: list[str]
    role: OwnerScoreRole
    samples: int

OwnerMetricRowsId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

OwnerMetricStreamId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

OwnerQuantAction: TypeAlias = Union[Literal["KeepDense"], OwnerQuantActionVariant1, OwnerQuantActionVariant2, OwnerQuantActionVariant3]

class OwnerQuantPlan(TypedDict):
    architecture: str
    assets: list[str]
    levels: list[str]
    limits: QuantLimits
    max_cache_package_bytes: int
    max_cache_payload_bytes: int
    original_source: OriginalSourceId
    tensors: list[OwnerQuantTensorPlan]

class OwnerQuantTensorPlan(TypedDict):
    action: OwnerQuantAction
    source: QuantSourceTensor

class OwnerQuantizationCompleted(TypedDict):
    original_source: OriginalSourceId
    output: QuantizationCompleted

class OwnerQuantizationStarted(TypedDict):
    plan: OwnerQuantPlan

class OwnerRowMetricBinding(TypedDict):
    content: OwnerRowMetricId
    row: int

OwnerRowMetricId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class OwnerScoreCacheBinding(TypedDict):
    cache: OwnerScoreCacheId
    content: OwnerScoreCacheContent
    context_bytes: int
    payload_bytes: int
    samples: int

class OwnerScoreCacheContent(TypedDict):
    context: list[OwnerContextBinding]
    estimator: OwnerScoreEstimator
    invocations: list[OwnerInvocationBinding]
    objective: BlobId
    source: OriginalSourceId
    tensor: QuantSourceTensor
    tokenizer: BlobId

OwnerScoreCacheId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class OwnerScoreCacheLimits(TypedDict):
    max_context_bytes: int
    max_context_tensors: int
    max_invocations: int
    max_payload_bytes: int
    max_samples: int

class OwnerScoreCacheManifest(TypedDict):
    binding: OwnerScoreCacheBinding
    limits: OwnerScoreCacheLimits

OwnerScoreEstimator: TypeAlias = Literal["ScalarWindowGradient"]

class OwnerScoreFitPlan(TypedDict):
    damping: float
    direct: DirectFitPlan
    limits: OwnerScoreMetricLimits
    max_row_bindings: int
    windows: list[WeightedOwnerCache]

class OwnerScoreLevelsFitted(TypedDict):
    outputs: list[DirectFitOutput]
    plan: OwnerScoreFitPlan
    rayon_threads: int
    rows: list[OwnerRowMetricBinding]
    rows_content: OwnerMetricRowsId
    source_f32: FitInputDigest
    stream: OwnerMetricStreamId

class OwnerScoreMetricLimits(TypedDict):
    max_payload_bytes: int
    max_reduction_entries: int
    max_row_bytes: int
    max_windows: int

OwnerScoreRole: TypeAlias = Union[Literal["Gather"], Literal["Matmul"]]

class OwnerScoreWindowMeta(TypedDict):
    context: list[OwnerContextMeta]
    invocations: list[OwnerInvocationMeta]

PackageId: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

PlaneAllocation: TypeAlias = Union[Literal["PerTile"], Literal["Uniform"]]

PlaneCodec: TypeAlias = Union[Literal["B3"], Literal["D2"], Literal["S34"]]

PlaneCounts: TypeAlias = Union[PlaneCountsVariant0, PlaneCountsVariant1]

PlaneRelation: TypeAlias = Union[Literal["Free"], PlaneRelationVariant1]

QuantAction: TypeAlias = Union[Literal["KeepDense"], QuantActionVariant1]

class QuantLimits(TypedDict):
    max_asset_bytes: int
    max_package_bytes: int
    max_retained_blob_bytes: int
    max_source_bytes: int

class QuantOutputLevel(TypedDict):
    name: str
    tensors: list[TensorMeta]

class QuantPlan(TypedDict):
    architecture: str
    assets: list[str]
    levels: list[str]
    limits: QuantLimits
    tensors: list[QuantTensorPlan]

class QuantSourceBinding(TypedDict):
    content: BlobId
    source: QuantSourceTensor

class QuantSourceTensor(TypedDict):
    name: str
    precision: ScalePrecision
    shape: list[int]

class QuantTensorPlan(TypedDict):
    action: QuantAction
    source: QuantSourceTensor

class QuantizationCompleted(TypedDict):
    assets: list[NamedAssetDigest]
    levels: list[QuantOutputLevel]
    sources: list[QuantSourceBinding]

class QuantizationStarted(TypedDict):
    plan: QuantPlan

class SalienceConversionLimits(TypedDict):
    max_manifest_bytes: int
    max_package_bytes: int

class SalienceConversionManifest(TypedDict):
    conversion: EvidenceReference
    derived_source: OriginalSourceId
    model: ModelManifest
    original_source: OriginalSourceId
    schema: SchemaId
    stage_package: PackageId
    transform: EvidenceReference

SalienceInverseAxis: TypeAlias = Union[Literal["LinearOutputRows"], Literal["NormChannels"]]

class SalienceMomentBinding(TypedDict):
    channels: int
    content: BlobId
    rows: int

SalienceNumerics: TypeAlias = Literal["LegacyF64ScaleF32V1"]

class SalienceOperationOutput(TypedDict):
    operation: str
    scale_bits: list[tuple[int, int, int, int]]

class SalienceSourceStageManifest(TypedDict):
    observation: SalienceTransformObservation
    schema: SchemaId
    transform: EvidenceReference

class SalienceStageLimits(TypedDict):
    max_asset_bytes: int
    max_manifest_bytes: int
    max_package_bytes: int
    max_statistics_bytes: int
    max_tensor_bytes: int

SalienceStatisticsKind: TypeAlias = Literal["Declared"]

class SalienceTransformCompleted(TypedDict):
    observation: SalienceTransformObservation

class SalienceTransformLimits(TypedDict):
    max_asset_bytes: int
    max_derived_bytes: int
    max_pair_working_bytes: int
    max_source_bytes: int
    max_statistics_bytes: int

class SalienceTransformObservation(TypedDict):
    assets: list[NamedAssetDigest]
    derived_source: OriginalSourceId
    derived_tensors: list[QuantSourceBinding]
    operations: list[SalienceOperationOutput]
    original_source: OriginalSourceId
    original_tensors: list[QuantSourceBinding]
    plan: SalienceTransformPlan

class SalienceTransformOperation(TypedDict):
    alpha_bits: tuple[int, int, int, int, int, int, int, int]
    inverse: str
    inverse_axis: SalienceInverseAxis
    moments: SalienceMomentBinding
    name: str
    projections: list[str]

class SalienceTransformPlan(TypedDict):
    architecture: str
    assets: list[str]
    limits: SalienceTransformLimits
    numerics: SalienceNumerics
    operations: list[SalienceTransformOperation]
    original_source: OriginalSourceId
    statistics_kind: SalienceStatisticsKind
    tensors: list[QuantSourceTensor]

class SalienceTransformStarted(TypedDict):
    plan: SalienceTransformPlan

ScaleAnchor: TypeAlias = Union[Literal["Channel"], Literal["Group"], Literal["Row"], Literal["Tensor"]]

class ScaleLaw(TypedDict):
    anchor: ScaleAnchor
    precision: ScalePrecision
    relation: PlaneRelation

ScalePrecision: TypeAlias = Union[Literal["Bf16"], Literal["F16"], Literal["F32"]]

class SchemaId(TypedDict):
    major: int
    minor: int

SemanticTensorDigest: TypeAlias = tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]

class TensorCaps(TypedDict):
    execution: TensorExecution
    payload_bytes: int

TensorExecution: TypeAlias = Union[Literal["Emulated"], Literal["Native"]]

class TensorMeta(TypedDict):
    blob: BlobReference
    name: str
    storage: TensorStorage

TensorSemanticMeta: TypeAlias = Union[TensorSemanticMetaVariant0, TensorSemanticMetaVariant1]

TensorStorage: TypeAlias = Union[TensorStorageVariant0, TensorStorageVariant1]

class TensorUploadPolicy(TypedDict):
    max_payload_bytes: NotRequired[Union[None, int]]
    native_only: bool

class TensorUploaded(TypedDict):
    backend: str
    caps: TensorCaps
    decoded_source_bytes: int
    physical_device: str
    tensor: str

Transport: TypeAlias = Union[Literal["Raw"], TransportVariant1]

UnknownReason: TypeAlias = Union[Literal["AwaitingIndependentReview"], Literal["MissingCapability"], Literal["MissingEvidence"], Literal["NotEvaluated"], Literal["Other"]]

Verdict: TypeAlias = Union[Literal["Fail"], Literal["Pass"], VerdictVariant2]

class WeightedOwnerCache(TypedDict):
    binding: OwnerScoreCacheBinding
    weight: float

class WeightedOwnerWindow(TypedDict):
    cache: OwnerScoreCacheId
    weight: float

WorkPrecision: TypeAlias = Union[Literal["F32Accum64"], Literal["F64"]]
