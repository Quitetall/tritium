// Generated from tritium-schema Rust types; do not edit.
// JavaScript numbers cannot exactly represent every uint64 JSON integer.
// Use a lossless JSON parser when consuming values above Number.MAX_SAFE_INTEGER.

export interface AdditiveLayout {
  readonly allocation: PlaneAllocation;
  readonly basis: Basis;
  readonly codec: PlaneCodec;
  readonly cols: number;
  readonly group: number;
  readonly law: ScaleLaw;
  readonly max_planes: number;
  readonly rows: number;
  readonly tile: number;
  readonly transport: Transport;
}

export interface AdmittedLaw {
  readonly law: ScaleLaw;
  readonly max_planes: number;
}

export type Basis = "Identity" | { readonly Hadamard: { readonly block: number } } | { readonly SignedRht: { readonly block: number; readonly domain: number; readonly seed: number } };

export type BlobId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface BlobReference {
  readonly id: BlobId;
  readonly shard?: null | string;
}

export interface CalibratedFitPlan {
  readonly cache: CalibrationCacheBinding;
  readonly damping: number;
  readonly direct: DirectFitPlan;
  readonly max_numeric_bytes: number;
  readonly partition: CalibrationPartition;
}

export interface CalibratedLevelsFitted {
  readonly factors: CalibrationFactors;
  readonly outputs: ReadonlyArray<DirectFitOutput>;
  readonly plan: CalibratedFitPlan;
  readonly rayon_threads: number;
  readonly source_f32: FitInputDigest;
}

export type CalibratedQuantAction = "KeepDense" | { readonly CalibratedFit: { readonly plan: CalibratedFitPlan } } | { readonly Fit: { readonly metric: DirectFitMetric; readonly plan: DirectFitPlan } };

export interface CalibratedQuantPlan {
  readonly architecture: string;
  readonly assets: ReadonlyArray<string>;
  readonly levels: ReadonlyArray<string>;
  readonly limits: QuantLimits;
  readonly max_cache_package_bytes: number;
  readonly max_cache_payload_bytes: number;
  readonly original_source: OriginalSourceId;
  readonly tensors: ReadonlyArray<CalibratedQuantTensorPlan>;
}

export interface CalibratedQuantTensorPlan {
  readonly action: CalibratedQuantAction;
  readonly source: QuantSourceTensor;
}

export interface CalibratedQuantizationCompleted {
  readonly original_source: OriginalSourceId;
  readonly output: QuantizationCompleted;
}

export interface CalibratedQuantizationStarted {
  readonly plan: CalibratedQuantPlan;
}

export interface CalibrationCacheBinding {
  readonly objective: BlobId;
  readonly payload_bytes: number;
  readonly request: CalibrationCacheRequest;
  readonly samples: number;
  readonly source: CalibrationSource;
  readonly tokenizer: BlobId;
  readonly windows: number;
}

export type CalibrationCacheId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface CalibrationCacheLimits {
  readonly max_payload_bytes: number;
  readonly max_samples: number;
  readonly max_windows: number;
}

export interface CalibrationCacheManifest {
  readonly binding: CalibrationCacheBinding;
  readonly window_samples: ReadonlyArray<number>;
}

export interface CalibrationCacheRequest {
  readonly consumer: CalibrationConsumer;
  readonly limits: CalibrationCacheLimits;
  readonly model: OriginalSourceId;
  readonly output: CalibrationOutput;
  readonly sample_start: number;
  readonly tensor: QuantSourceTensor;
}

export interface CalibrationCaptureCompleted {
  readonly caches: ReadonlyArray<CalibrationCacheBinding>;
  readonly windows: ReadonlyArray<ReadonlyArray<number>>;
}

export type CalibrationCaptureInput = { readonly RawText: { readonly windows: ReadonlyArray<string> } } | { readonly Tokens: { readonly windows: ReadonlyArray<ReadonlyArray<number>> } };

export interface CalibrationCaptureLimits {
  readonly max_config_bytes: number;
  readonly max_dense_bytes: number;
  readonly max_text_bytes: number;
  readonly max_tokenizer_config_bytes: number;
  readonly max_total_payload_bytes: number;
  readonly max_window_tokens: number;
  readonly per_cache: CalibrationCacheLimits;
}

export interface CalibrationCapturePlan {
  readonly input: CalibrationCaptureInput;
  readonly limits: CalibrationCaptureLimits;
  readonly profile: CalibrationCaptureProfile;
  readonly source: OriginalSourceId;
  readonly stage_package: PackageId;
  readonly tensors: ReadonlyArray<QuantSourceTensor>;
  readonly tokenizer: BlobId;
  readonly tokenizer_config?: BlobId | null;
  readonly transform: EvidenceReference;
}

export type CalibrationCaptureProfile = "NextTokenWindowMeanK1F32V1";

export interface CalibrationCaptureStarted {
  readonly plan: CalibrationCapturePlan;
}

export type CalibrationConsumer = "GatherIdentity" | "Matmul";

export interface CalibrationFactor {
  readonly content: FitInputDigest;
  readonly dimension: number;
  readonly samples: number;
  readonly selected: number;
  readonly selection: FitInputDigest;
  readonly total_weight: number;
}

export interface CalibrationFactors {
  readonly inputs: ReadonlyArray<CalibrationFactor>;
  readonly output?: CalibrationFactor | null;
  readonly plan: CalibrationPlan;
}

export type CalibrationOutput = "DenseFisher" | "IndexedFisher" | "None";

export type CalibrationPartition = "Full" | { readonly Groups: { readonly columns: number } };

export interface CalibrationPlan {
  readonly consumer: CalibrationConsumer;
  readonly damping: number;
  readonly max_numeric_bytes: number;
  readonly output: CalibrationOutput;
  readonly partition: CalibrationPartition;
  readonly sample_start: number;
  readonly source: CalibrationSource;
  readonly tensor: QuantSourceTensor;
}

export interface CalibrationSource {
  readonly cache: FitInputDigest;
  readonly model: FitInputDigest;
  readonly tokens: FitInputDigest;
}

export interface CapturedSalienceConversionManifest {
  readonly capture: EvidenceReference;
  readonly lineage: SalienceConversionManifest;
}

export interface DirectFitLevel {
  readonly name: string;
  readonly planes: number;
}

export interface DirectFitLimits {
  readonly max_metric_entries: number;
  readonly max_output_trits: number;
  readonly max_weights: number;
}

export type DirectFitMetric = "Identity" | { readonly InputGramF64: { readonly content: FitInputDigest } };

export interface DirectFitOutput {
  readonly content: SemanticTensorDigest;
  readonly level: string;
  readonly planes: number;
}

export interface DirectFitPlan {
  readonly layout: AdditiveLayout;
  readonly levels: ReadonlyArray<DirectFitLevel>;
  readonly limits: DirectFitLimits;
  readonly profile: DirectFitProfile;
}

export type DirectFitProfile = { readonly ActivationAwareLadder: LadderFitOptions } | { readonly JointFeedbackGroups: JointFeedbackFitOptions } | { readonly JointIndependentGroups: JointGroupFitOptions };

export interface DirectLevelsFitted {
  readonly metric: DirectFitMetric;
  readonly outputs: ReadonlyArray<DirectFitOutput>;
  readonly plan: DirectFitPlan;
  readonly rayon_threads: number;
  readonly source_f32: FitInputDigest;
  readonly tensor: string;
}

export interface EvidenceEnvelope {
  readonly digest: string;
  readonly parent?: null | string;
  readonly payload: unknown;
  readonly run: string;
  readonly schema: string;
  readonly seq: number;
  readonly span: string;
  readonly t: number;
  readonly v: number;
}

export interface EvidenceReference {
  readonly root: [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
  readonly run: string;
  readonly span: string;
}

export type FitInputDigest = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface JointFeedbackFitOptions {
  readonly damp: number;
  readonly decay: number;
  readonly decay_ramp: boolean;
  readonly fitter: JointGroupFitOptions;
  readonly precision: WorkPrecision;
}

export interface JointGroupFitOptions {
  readonly em_restarts: number;
  readonly max_iterations: number;
  readonly modulated_relay: boolean;
  readonly ridge: number;
  readonly ridge_condition_limit: number;
  readonly softened_relay: boolean;
}

export interface LadderFitOptions {
  readonly damp: number;
  readonly decay: number;
  readonly decay_ramp: boolean;
  readonly grid: number;
  readonly scale_refit: LadderScaleRefit;
  readonly search_sweeps: number;
}

export type LadderScaleRefit = "InLoop" | "None" | "PostPass";

export type LayoutError = "BasisBlockDoesNotDivideColumns" | "EmptyDimensions" | "EmptyPlaneStack" | "EmptyTransportBlock" | "InvalidBasisBlock" | "InvalidScaleRatio" | "PlaneCountExceedsLaw" | "UnsupportedGroup" | "UnsupportedLaw" | "UnsupportedTileSize";

export interface LegacyModelIdentity {
  readonly id: string;
  readonly level: string;
}

export interface LegacyModelImportBoundPrepared {
  readonly companion_assets: ReadonlyArray<NamedAssetDigest>;
  readonly companion_source: OriginalSourceId;
  readonly companion_tensors: ReadonlyArray<QuantSourceBinding>;
  readonly legacy_id: string;
  readonly manifest_bytes: number;
  readonly manifest_digest: [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];
}

export interface LegacyModelImportPrepared {
  readonly companion_assets: ReadonlyArray<NamedAssetDigest>;
  readonly companion_source: OriginalSourceId;
  readonly companion_tensors: ReadonlyArray<QuantSourceBinding>;
  readonly legacy_id: string;
  readonly manifest: ModelManifest;
}

export interface LegacySaltV2BundleDescriptor {
  readonly admission_id: string;
  readonly artifact_kind: string;
  readonly campaign_id: string;
  readonly complete_model: boolean;
  readonly completion_id: string;
  readonly hf_assets: ReadonlyArray<LegacySaltV2AssetDescriptor>;
  readonly official_payload_authenticated: boolean;
  readonly packing: string;
  readonly preserved: LegacySaltV2PreservedDescriptor;
  readonly profiles: LegacySaltV2ProfileSet;
  readonly schema_version: number;
  readonly selection_id: string;
  readonly source_identity_status: string;
  readonly source_model_id: string;
  readonly source_revision: string;
}

export interface Level {
  readonly fit: LevelFit;
  readonly name: string;
  readonly tensors: ReadonlyArray<TensorMeta>;
}

export type LevelFit = "Imported" | { readonly Direct: { readonly evidence: ReadonlyArray<EvidenceReference> } } | { readonly ImportedPrefixOf: { readonly level: string } };

export type LoadedSourceIdentity = { readonly HuggingFace: { readonly reason: UnknownReason } } | { readonly Trit: { readonly level: string; readonly model_id: ModelId; readonly package_id: PackageId } };

export interface ManifestAsset {
  readonly blob: BlobReference;
  readonly name: string;
}

export type ModelId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface ModelLoaded {
  readonly architecture: string;
  readonly backend: string;
  readonly layer_count: number;
  readonly physical_device: string;
  readonly source: LoadedSourceIdentity;
  readonly vocab_size: number;
}

export interface ModelManifest {
  readonly architecture: string;
  readonly assets: ReadonlyArray<ManifestAsset>;
  readonly legacy_ids: ReadonlyArray<LegacyModelIdentity>;
  readonly levels: ReadonlyArray<Level>;
  readonly schema: SchemaId;
  readonly shards: ReadonlyArray<ModelShard>;
}

export interface ModelSemanticMeta {
  readonly architecture: string;
  readonly assets: ReadonlyArray<NamedAssetDigest>;
  readonly tensors: ReadonlyArray<NamedTensorDigest>;
}

export interface ModelShard {
  readonly name: string;
  readonly package: PackageId;
}

export interface NamedAssetDigest {
  readonly content: BlobId;
  readonly name: string;
}

export interface NamedTensorDigest {
  readonly content: SemanticTensorDigest;
  readonly name: string;
}

export interface OriginalCapturedSalienceConversionManifest {
  readonly lineage: CapturedSalienceConversionManifest;
  readonly original_capture: EvidenceReference;
}

export interface OriginalCapturedSalienceSourceStageManifest {
  readonly lineage: SalienceSourceStageManifest;
  readonly original_capture: EvidenceReference;
}

export interface OriginalSalienceCaptureCompleted {
  readonly outputs: ReadonlyArray<OriginalSalienceCaptureOutput>;
  readonly windows: ReadonlyArray<ReadonlyArray<number>>;
}

export type OriginalSalienceCaptureInput = { readonly RawText: { readonly windows: ReadonlyArray<string> } } | { readonly Tokens: { readonly windows: ReadonlyArray<ReadonlyArray<number>> } };

export interface OriginalSalienceCaptureLimits {
  readonly max_asset_bytes: number;
  readonly max_config_bytes: number;
  readonly max_dense_bytes: number;
  readonly max_moment_bytes: number;
  readonly max_source_bytes: number;
  readonly max_text_bytes: number;
  readonly max_tokenizer_config_bytes: number;
  readonly max_total_tokens: number;
  readonly max_window_tokens: number;
  readonly max_windows: number;
}

export interface OriginalSalienceCaptureOutput {
  readonly moments: SalienceMomentBinding;
  readonly operation: string;
}

export interface OriginalSalienceCapturePair {
  readonly channels: number;
  readonly inverse: string;
  readonly inverse_axis: SalienceInverseAxis;
  readonly name: string;
  readonly projections: ReadonlyArray<string>;
}

export interface OriginalSalienceCapturePlan {
  readonly architecture: string;
  readonly assets: ReadonlyArray<NamedAssetDigest>;
  readonly input: OriginalSalienceCaptureInput;
  readonly limits: OriginalSalienceCaptureLimits;
  readonly pairs: ReadonlyArray<OriginalSalienceCapturePair>;
  readonly profile: OriginalSalienceCaptureProfile;
  readonly source: OriginalSourceId;
  readonly tensors: ReadonlyArray<QuantSourceBinding>;
}

export type OriginalSalienceCaptureProfile = "AllTokenF64SecondMomentV1";

export interface OriginalSalienceCaptureStarted {
  readonly plan: OriginalSalienceCapturePlan;
}

export type OriginalSourceId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type OwnerContextDtype = "Bf16" | "Bool" | "C128" | "C64" | "F16" | "F32" | "F64" | "I16" | "I32" | "I64" | "I8" | "U8";

export type OwnerMetricRowsId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type OwnerMetricStreamId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type OwnerQuantAction = "KeepDense" | { readonly CalibratedFit: { readonly plan: CalibratedFitPlan } } | { readonly Fit: { readonly metric: DirectFitMetric; readonly plan: DirectFitPlan } } | { readonly OwnerFit: { readonly plan: OwnerScoreFitPlan } };

export interface OwnerQuantPlan {
  readonly architecture: string;
  readonly assets: ReadonlyArray<string>;
  readonly levels: ReadonlyArray<string>;
  readonly limits: QuantLimits;
  readonly max_cache_package_bytes: number;
  readonly max_cache_payload_bytes: number;
  readonly original_source: OriginalSourceId;
  readonly tensors: ReadonlyArray<OwnerQuantTensorPlan>;
}

export interface OwnerQuantTensorPlan {
  readonly action: OwnerQuantAction;
  readonly source: QuantSourceTensor;
}

export interface OwnerQuantizationCompleted {
  readonly original_source: OriginalSourceId;
  readonly output: QuantizationCompleted;
}

export interface OwnerQuantizationStarted {
  readonly plan: OwnerQuantPlan;
}

export interface OwnerRowMetricBinding {
  readonly content: OwnerRowMetricId;
  readonly row: number;
}

export type OwnerRowMetricId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface OwnerScoreCacheBinding {
  readonly cache: OwnerScoreCacheId;
  readonly content: OwnerScoreCacheContent;
  readonly context_bytes: number;
  readonly payload_bytes: number;
  readonly samples: number;
}

export interface OwnerScoreCacheContent {
  readonly context: ReadonlyArray<OwnerContextBinding>;
  readonly estimator: OwnerScoreEstimator;
  readonly invocations: ReadonlyArray<OwnerInvocationBinding>;
  readonly objective: BlobId;
  readonly source: OriginalSourceId;
  readonly tensor: QuantSourceTensor;
  readonly tokenizer: BlobId;
}

export type OwnerScoreCacheId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface OwnerScoreCacheLimits {
  readonly max_context_bytes: number;
  readonly max_context_tensors: number;
  readonly max_invocations: number;
  readonly max_payload_bytes: number;
  readonly max_samples: number;
}

export interface OwnerScoreCacheManifest {
  readonly binding: OwnerScoreCacheBinding;
  readonly limits: OwnerScoreCacheLimits;
}

export type OwnerScoreEstimator = "ScalarWindowGradient";

export interface OwnerScoreFitPlan {
  readonly damping: number;
  readonly direct: DirectFitPlan;
  readonly limits: OwnerScoreMetricLimits;
  readonly max_row_bindings: number;
  readonly windows: ReadonlyArray<WeightedOwnerCache>;
}

export interface OwnerScoreLevelsFitted {
  readonly outputs: ReadonlyArray<DirectFitOutput>;
  readonly plan: OwnerScoreFitPlan;
  readonly rayon_threads: number;
  readonly rows: ReadonlyArray<OwnerRowMetricBinding>;
  readonly rows_content: OwnerMetricRowsId;
  readonly source_f32: FitInputDigest;
  readonly stream: OwnerMetricStreamId;
}

export interface OwnerScoreMetricLimits {
  readonly max_payload_bytes: number;
  readonly max_reduction_entries: number;
  readonly max_row_bytes: number;
  readonly max_windows: number;
}

export type OwnerScoreRole = "Gather" | "Matmul";

export interface OwnerScoreWindowMeta {
  readonly context: ReadonlyArray<OwnerContextMeta>;
  readonly invocations: ReadonlyArray<OwnerInvocationMeta>;
}

export type PackageId = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export type PlaneAllocation = "PerTile" | "Uniform";

export type PlaneCodec = "B3" | "D2" | "S34";

export type PlaneCounts = { readonly PerTile: ReadonlyArray<number> } | { readonly Uniform: number };

export type PlaneRelation = "Free" | { readonly Tied: { readonly den: number; readonly num: number } };

export type QuantAction = "KeepDense" | { readonly Fit: { readonly metric: DirectFitMetric; readonly plan: DirectFitPlan } };

export interface QuantLimits {
  readonly max_asset_bytes: number;
  readonly max_package_bytes: number;
  readonly max_retained_blob_bytes: number;
  readonly max_source_bytes: number;
}

export interface QuantOutputLevel {
  readonly name: string;
  readonly tensors: ReadonlyArray<TensorMeta>;
}

export interface QuantPlan {
  readonly architecture: string;
  readonly assets: ReadonlyArray<string>;
  readonly levels: ReadonlyArray<string>;
  readonly limits: QuantLimits;
  readonly tensors: ReadonlyArray<QuantTensorPlan>;
}

export interface QuantSourceBinding {
  readonly content: BlobId;
  readonly source: QuantSourceTensor;
}

export interface QuantSourceTensor {
  readonly name: string;
  readonly precision: ScalePrecision;
  readonly shape: ReadonlyArray<number>;
}

export interface QuantTensorPlan {
  readonly action: QuantAction;
  readonly source: QuantSourceTensor;
}

export interface QuantizationCompleted {
  readonly assets: ReadonlyArray<NamedAssetDigest>;
  readonly levels: ReadonlyArray<QuantOutputLevel>;
  readonly sources: ReadonlyArray<QuantSourceBinding>;
}

export interface QuantizationStarted {
  readonly plan: QuantPlan;
}

export interface SalienceConversionLimits {
  readonly max_manifest_bytes: number;
  readonly max_package_bytes: number;
}

export interface SalienceConversionManifest {
  readonly conversion: EvidenceReference;
  readonly derived_source: OriginalSourceId;
  readonly model: ModelManifest;
  readonly original_source: OriginalSourceId;
  readonly schema: SchemaId;
  readonly stage_package: PackageId;
  readonly transform: EvidenceReference;
}

export type SalienceInverseAxis = "LinearOutputRows" | "NormChannels";

export interface SalienceMomentBinding {
  readonly channels: number;
  readonly content: BlobId;
  readonly rows: number;
}

export type SalienceNumerics = "LegacyF64ScaleF32V1";

export interface SalienceOperationOutput {
  readonly operation: string;
  readonly scale_bits: ReadonlyArray<[number, number, number, number]>;
}

export interface SalienceSourceStageManifest {
  readonly observation: SalienceTransformObservation;
  readonly schema: SchemaId;
  readonly transform: EvidenceReference;
}

export interface SalienceStageLimits {
  readonly max_asset_bytes: number;
  readonly max_manifest_bytes: number;
  readonly max_package_bytes: number;
  readonly max_statistics_bytes: number;
  readonly max_tensor_bytes: number;
}

export type SalienceStatisticsKind = "Declared";

export interface SalienceTransformCompleted {
  readonly observation: SalienceTransformObservation;
}

export interface SalienceTransformLimits {
  readonly max_asset_bytes: number;
  readonly max_derived_bytes: number;
  readonly max_pair_working_bytes: number;
  readonly max_source_bytes: number;
  readonly max_statistics_bytes: number;
}

export interface SalienceTransformObservation {
  readonly assets: ReadonlyArray<NamedAssetDigest>;
  readonly derived_source: OriginalSourceId;
  readonly derived_tensors: ReadonlyArray<QuantSourceBinding>;
  readonly operations: ReadonlyArray<SalienceOperationOutput>;
  readonly original_source: OriginalSourceId;
  readonly original_tensors: ReadonlyArray<QuantSourceBinding>;
  readonly plan: SalienceTransformPlan;
}

export interface SalienceTransformOperation {
  readonly alpha_bits: [number, number, number, number, number, number, number, number];
  readonly inverse: string;
  readonly inverse_axis: SalienceInverseAxis;
  readonly moments: SalienceMomentBinding;
  readonly name: string;
  readonly projections: ReadonlyArray<string>;
}

export interface SalienceTransformPlan {
  readonly architecture: string;
  readonly assets: ReadonlyArray<string>;
  readonly limits: SalienceTransformLimits;
  readonly numerics: SalienceNumerics;
  readonly operations: ReadonlyArray<SalienceTransformOperation>;
  readonly original_source: OriginalSourceId;
  readonly statistics_kind: SalienceStatisticsKind;
  readonly tensors: ReadonlyArray<QuantSourceTensor>;
}

export interface SalienceTransformStarted {
  readonly plan: SalienceTransformPlan;
}

export type ScaleAnchor = "Channel" | "Group" | "Row" | "Tensor";

export interface ScaleLaw {
  readonly anchor: ScaleAnchor;
  readonly precision: ScalePrecision;
  readonly relation: PlaneRelation;
}

export type ScalePrecision = "Bf16" | "F16" | "F32";

export interface SchemaId {
  readonly major: number;
  readonly minor: number;
}

export type SemanticTensorDigest = [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number];

export interface TensorCaps {
  readonly execution: TensorExecution;
  readonly payload_bytes: number;
}

export type TensorExecution = "Emulated" | "Native";

export interface TensorMeta {
  readonly blob: BlobReference;
  readonly name: string;
  readonly storage: TensorStorage;
}

export type TensorSemanticMeta = { readonly Additive: { readonly basis: Basis; readonly cols: number; readonly group: number; readonly law: ScaleLaw; readonly planes: PlaneCounts; readonly rows: number; readonly tile: number } } | { readonly Dense: { readonly precision: ScalePrecision; readonly shape: ReadonlyArray<number> } };

export type TensorStorage = { readonly Additive: { readonly layout: AdditiveLayout; readonly planes: PlaneCounts } } | { readonly Dense: { readonly precision: ScalePrecision; readonly shape: ReadonlyArray<number> } };

export interface TensorUploadPolicy {
  readonly max_payload_bytes?: null | number;
  readonly native_only: boolean;
}

export interface TensorUploaded {
  readonly backend: string;
  readonly caps: TensorCaps;
  readonly decoded_source_bytes: number;
  readonly physical_device: string;
  readonly tensor: string;
}

export type Transport = "Raw" | { readonly JointHuffman: { readonly block: number } };

export type UnknownReason = "AwaitingIndependentReview" | "MissingCapability" | "MissingEvidence" | "NotEvaluated" | "Other";

export type Verdict = "Fail" | "Pass" | { readonly Unknown: { readonly reason: UnknownReason } };

export interface WeightedOwnerCache {
  readonly binding: OwnerScoreCacheBinding;
  readonly weight: number;
}

export interface WeightedOwnerWindow {
  readonly cache: OwnerScoreCacheId;
  readonly weight: number;
}

export type WorkPrecision = "F32Accum64" | "F64";

export interface LegacySaltV2AssetDescriptor {
  readonly bytes: number;
  readonly file: string;
  readonly package_id: string;
}

export interface LegacySaltV2PreservedDescriptor {
  readonly file: string;
  readonly package_id: string;
  readonly payload_bytes: number;
  readonly serialized_bytes: number;
  readonly tensors: number;
}

export interface LegacySaltV2ProfileDescriptor {
  readonly file: string;
  readonly package_id: string;
  readonly resident_bytes: number;
  readonly serialized_bytes: number;
}

export interface LegacySaltV2ProfileSet {
  readonly "compact-v1": LegacySaltV2ProfileDescriptor;
  readonly "near-lossless-v1": LegacySaltV2ProfileDescriptor;
}

export interface OwnerContextBinding {
  readonly content: BlobId;
  readonly meta: OwnerContextMeta;
}

export interface OwnerContextMeta {
  readonly conjugated: boolean;
  readonly dtype: OwnerContextDtype;
  readonly name: string;
  readonly negated: boolean;
  readonly shape: ReadonlyArray<number>;
}

export interface OwnerInvocationBinding {
  readonly adjoint: BlobId;
  readonly input: BlobId;
  readonly meta: OwnerInvocationMeta;
}

export interface OwnerInvocationMeta {
  readonly ordinal: number;
  readonly paths: ReadonlyArray<string>;
  readonly role: OwnerScoreRole;
  readonly samples: number;
}
