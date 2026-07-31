use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use tritium_quantize::{
    ConversionRun, ConversionStage, LogicalTritCount, MeasuredPackage, PhysicalSizeReport,
    RunStatus,
};

use crate::{ContentId, SaltError, SaltProfile, SaltSpec, SaltStage, WorkId};

const STATE_MAGIC: [u8; 4] = *b"TSV2";
const STATE_VERSION: u8 = 4;
const STATE_HASH_CONTEXT: &str = "tritium salt pipeline state v1";
const WORK_LOCK_DIRECTORY: &str = ".tritium-salt-locks";
const MAX_STATE_BYTES: u64 = 64 * 1024 * 1024;
const STATE_HEADER_BYTES: u64 = 4 + 1 + 8;
const STATE_CHECKSUM_BYTES: u64 = 32;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One named, finite evaluation measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    name: String,
    unit: String,
    value: f64,
    confidence_interval: Option<(f64, f64)>,
}

impl Metric {
    /// Construct a finite metric with non-empty name and unit.
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        value: f64,
    ) -> Result<Self, SaltError> {
        let name = checked_text("metric name", name.into())?;
        let unit = checked_text("metric unit", unit.into())?;
        if !value.is_finite() {
            return Err(SaltError::InvalidField("metric value"));
        }
        Ok(Self {
            name,
            unit,
            value: canonical_zero(value),
            confidence_interval: None,
        })
    }

    /// Attach a finite confidence interval containing the measured value.
    pub fn with_confidence_interval(mut self, lower: f64, upper: f64) -> Result<Self, SaltError> {
        if !lower.is_finite() || !upper.is_finite() || lower > self.value || self.value > upper {
            return Err(SaltError::InvalidField("metric confidence interval"));
        }
        self.confidence_interval = Some((canonical_zero(lower), canonical_zero(upper)));
        Ok(self)
    }

    /// Stable metric name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Measurement unit.
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Point estimate.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Optional lower and upper confidence bounds.
    pub const fn confidence_interval(&self) -> Option<(f64, f64)> {
        self.confidence_interval
    }

    fn validate(&self) -> bool {
        valid_text(&self.name)
            && valid_text(&self.unit)
            && finite_canonical(self.value)
            && self.confidence_interval.is_none_or(|(lower, upper)| {
                finite_canonical(lower)
                    && finite_canonical(upper)
                    && lower <= self.value
                    && self.value <= upper
            })
    }
}

/// Accelerator resources consumed by one driver stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareUsage {
    accelerator: String,
    accelerator_count: u32,
    gpu_seconds: u64,
    peak_vram_bytes: u64,
}

impl HardwareUsage {
    /// Construct a measured accelerator-usage record.
    pub fn new(
        accelerator: impl Into<String>,
        accelerator_count: u32,
        gpu_seconds: u64,
        peak_vram_bytes: u64,
    ) -> Result<Self, SaltError> {
        if accelerator_count == 0 || peak_vram_bytes == 0 {
            return Err(SaltError::InvalidField("hardware usage"));
        }
        Ok(Self {
            accelerator: checked_text("accelerator", accelerator.into())?,
            accelerator_count,
            gpu_seconds,
            peak_vram_bytes,
        })
    }

    /// Accelerator model or pool identifier.
    pub fn accelerator(&self) -> &str {
        &self.accelerator
    }

    /// Number of accelerators used together.
    pub const fn accelerator_count(&self) -> u32 {
        self.accelerator_count
    }

    /// Aggregate wall-clock GPU seconds for this record.
    pub const fn gpu_seconds(&self) -> u64 {
        self.gpu_seconds
    }

    /// Measured peak resident accelerator memory.
    pub const fn peak_vram_bytes(&self) -> u64 {
        self.peak_vram_bytes
    }

    fn validate(&self) -> bool {
        valid_text(&self.accelerator) && self.accelerator_count != 0 && self.peak_vram_bytes != 0
    }
}

/// Exact logical, serialized, and resident storage ledger for one package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalLedger {
    package_id: ContentId,
    transport_package_id: [u8; 32],
    core_parameter_count: u64,
    logical_core_trits: u64,
    serialized_core_bytes: u64,
    resident_core_bytes: u64,
    metadata_bytes: u64,
    allocation_map_bits: u64,
    allocation_map_embedded_bits: u64,
    resident_metadata_bytes: u64,
    resident_allocation_map_bits: u64,
    resident_allocation_map_embedded_bits: u64,
    preserved_bytes: u64,
    resident_preserved_bytes: u64,
    resident_shadow_bytes: u64,
    package_bytes: u64,
}

impl PhysicalLedger {
    /// Build a ledger whose exact package total is independently checked.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        package_id: ContentId,
        transport_package_id: [u8; 32],
        core_parameter_count: u64,
        logical_core_trits: LogicalTritCount,
        serialized_core_bytes: u64,
        resident_core_bytes: u64,
        metadata_bytes: u64,
        allocation_map_bits: u64,
        allocation_map_embedded_bits: u64,
        resident_metadata_bytes: u64,
        resident_allocation_map_bits: u64,
        resident_allocation_map_embedded_bits: u64,
        preserved_bytes: u64,
        resident_preserved_bytes: u64,
        resident_shadow_bytes: u64,
        package_bytes: u64,
    ) -> Result<Self, SaltError> {
        let ledger = Self {
            package_id,
            transport_package_id,
            core_parameter_count,
            logical_core_trits: logical_core_trits.get(),
            serialized_core_bytes,
            resident_core_bytes,
            metadata_bytes,
            allocation_map_bits,
            allocation_map_embedded_bits,
            resident_metadata_bytes,
            resident_allocation_map_bits,
            resident_allocation_map_embedded_bits,
            preserved_bytes,
            resident_preserved_bytes,
            resident_shadow_bytes,
            package_bytes,
        };
        ledger.validate()?;
        Ok(ledger)
    }

    /// Convert the quantizer's package-bound physical report without duplicating
    /// or manually transcribing its component accounting.
    pub fn from_physical_size_report(
        package_id: ContentId,
        report: PhysicalSizeReport,
    ) -> Result<Self, SaltError> {
        let serialized = report.serialized();
        let resident = report.resident();
        let serialized_core_bytes = checked_sum(
            [
                serialized.core_payload_bytes(),
                serialized.core_scale_bytes(),
            ],
            "physical ledger overflow",
        )?;
        let metadata_bytes = checked_sum(
            [
                serialized.allocation_map_bytes(),
                serialized.header_bytes(),
                serialized.transform_bytes(),
                serialized.alignment_bytes(),
            ],
            "physical ledger overflow",
        )?;
        let resident_metadata_bytes = checked_sum(
            [resident.map_bytes(), resident.descriptor_bytes()],
            "resident physical ledger overflow",
        )?;
        let package = report.package();
        Self::new(
            package_id,
            *package.id().as_bytes(),
            report.core_parameter_count(),
            report.logical_core_trits(),
            serialized_core_bytes,
            resident.core_bytes(),
            metadata_bytes,
            serialized.allocation_map_bits(),
            serialized.allocation_map_embedded_bits(),
            resident_metadata_bytes,
            resident.map_bits(),
            resident.map_embedded_bits(),
            serialized.preserved_bytes(),
            resident.preserved_bytes(),
            resident.shadow_bytes(),
            package.physical_bytes(),
        )
    }

    /// SALT content identity of the accepted pack-stage artifact.
    pub const fn package_id(&self) -> ContentId {
        self.package_id
    }

    /// Transport-format identity of the exact serialized package bytes.
    pub const fn transport_package_id(&self) -> &[u8; 32] {
        &self.transport_package_id
    }

    /// Number of core projection parameters used as the bpw denominator.
    pub const fn core_parameter_count(&self) -> u64 {
        self.core_parameter_count
    }

    /// Exact logical ternary symbols selected by the allocator.
    pub const fn logical_core_trits(&self) -> u64 {
        self.logical_core_trits
    }

    /// Exact serialized bytes for ternary coefficients and scales.
    pub const fn serialized_core_bytes(&self) -> u64 {
        self.serialized_core_bytes
    }

    /// Exact resident bytes for core projection data after loading.
    pub const fn resident_core_bytes(&self) -> u64 {
        self.resident_core_bytes
    }

    /// Exact serialized headers, indexes, padding, and other metadata bytes.
    pub const fn metadata_bytes(&self) -> u64 {
        self.metadata_bytes
    }

    /// Exact logical serialized allocation-map bits.
    pub const fn allocation_map_bits(&self) -> u64 {
        self.allocation_map_bits
    }

    /// Serialized allocation-map bits carried in mandatory scalar fields.
    pub const fn allocation_map_embedded_bits(&self) -> u64 {
        self.allocation_map_embedded_bits
    }

    /// Exact resident bytes for indexes, offsets, and other core metadata.
    pub const fn resident_metadata_bytes(&self) -> u64 {
        self.resident_metadata_bytes
    }

    /// Exact logical runtime allocation-map bits.
    pub const fn resident_allocation_map_bits(&self) -> u64 {
        self.resident_allocation_map_bits
    }

    /// Runtime allocation-map bits carried by a mandatory launch scalar.
    pub const fn resident_allocation_map_embedded_bits(&self) -> u64 {
        self.resident_allocation_map_embedded_bits
    }

    /// Exact serialized bytes of preserved non-core tensors.
    pub const fn preserved_bytes(&self) -> u64 {
        self.preserved_bytes
    }

    /// Exact resident bytes of preserved non-core tensors.
    pub const fn resident_preserved_bytes(&self) -> u64 {
        self.resident_preserved_bytes
    }

    /// Exact resident bytes of required alternate layouts or runtime weight shadows.
    pub const fn resident_shadow_bytes(&self) -> u64 {
        self.resident_shadow_bytes
    }

    /// Exact steady-state whole-model resident bytes.
    pub fn resident_total_bytes(&self) -> u64 {
        self.checked_resident_total_bytes().unwrap_or(u64::MAX)
    }

    /// Exact whole-model package bytes.
    pub const fn package_bytes(&self) -> u64 {
        self.package_bytes
    }

    fn validate(&self) -> Result<(), SaltError> {
        if self.package_id.as_bytes().iter().all(|byte| *byte == 0)
            || self.transport_package_id.iter().all(|byte| *byte == 0)
            || self.core_parameter_count == 0
            || self.logical_core_trits == 0
            || self.serialized_core_bytes == 0
            || self.resident_core_bytes == 0
            || self.package_bytes == 0
        {
            return Err(SaltError::InvalidField("physical ledger"));
        }
        if self.allocation_map_embedded_bits > self.allocation_map_bits
            || self.resident_allocation_map_embedded_bits > self.resident_allocation_map_bits
            || self.allocation_map_bits != self.resident_allocation_map_bits
        {
            return Err(SaltError::InvalidField("allocation map ledger"));
        }
        let exact_package = self
            .serialized_core_bytes
            .checked_add(self.metadata_bytes)
            .and_then(|bytes| bytes.checked_add(self.preserved_bytes))
            .ok_or(SaltError::InvalidField("physical ledger overflow"))?;
        if exact_package != self.package_bytes {
            return Err(SaltError::InvalidField("physical package total"));
        }
        self.checked_resident_total_bytes()
            .ok_or(SaltError::InvalidField("resident physical ledger overflow"))?;
        Ok(())
    }

    fn checked_resident_total_bytes(&self) -> Option<u64> {
        self.resident_core_bytes
            .checked_add(self.resident_metadata_bytes)
            .and_then(|bytes| bytes.checked_add(self.resident_preserved_bytes))
            .and_then(|bytes| bytes.checked_add(self.resident_shadow_bytes))
    }

    fn satisfies(&self, profile: SaltProfile) -> bool {
        if self.validate().is_err() {
            return false;
        }
        let Some(core_bytes) = self.serialized_core_bytes.checked_add(self.metadata_bytes) else {
            return false;
        };
        let Some(resident_bytes) = self
            .resident_core_bytes
            .checked_add(self.resident_metadata_bytes)
            .and_then(|bytes| bytes.checked_add(self.resident_shadow_bytes))
        else {
            return false;
        };
        let serialized_rate_millibits = u128::from(core_bytes) * 8_000;
        let resident_rate_millibits = u128::from(resident_bytes) * 8_000;
        let budget_millibits =
            u128::from(self.core_parameter_count) * u128::from(profile.max_core_millibpw());
        serialized_rate_millibits <= budget_millibits && resident_rate_millibits <= budget_millibits
    }
}

/// Reloaded quality-gate evidence bound to the evaluation evidence and exact package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityEvidence {
    evidence_id: ContentId,
    package_id: ContentId,
    harness_id: ContentId,
    passed: bool,
    summary: String,
}

impl QualityEvidence {
    /// Construct a quality verdict emitted by the validation stage.
    pub fn new(
        evidence_id: ContentId,
        package_id: ContentId,
        harness_id: ContentId,
        passed: bool,
        summary: impl Into<String>,
    ) -> Result<Self, SaltError> {
        Ok(Self {
            evidence_id,
            package_id,
            harness_id,
            passed,
            summary: checked_text("quality summary", summary.into())?,
        })
    }

    /// Evidence bundle under which the verdict was measured.
    pub const fn evidence_id(&self) -> ContentId {
        self.evidence_id
    }

    /// Exact packed artifact that was reloaded and evaluated.
    pub const fn package_id(&self) -> ContentId {
        self.package_id
    }

    /// Evaluation harness identity.
    pub const fn harness_id(&self) -> ContentId {
        self.harness_id
    }

    /// Whether every registered profile gate passed.
    pub const fn passed(&self) -> bool {
        self.passed
    }

    /// Human-readable summary retained with the evidence.
    pub fn summary(&self) -> &str {
        &self.summary
    }

    fn validate(&self) -> bool {
        valid_content_id(self.evidence_id)
            && valid_content_id(self.package_id)
            && valid_content_id(self.harness_id)
            && valid_text(&self.summary)
    }
}

/// Immutable package installed by the publish stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedArtifact {
    package_id: ContentId,
    physical_bytes: u64,
}

/// Durable work-directory artifact whose exact bytes are bound to a stage output ID.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageArtifact {
    relative_path: String,
}

impl StageArtifact {
    /// Construct a portable path relative to the pipeline's content-addressed work directory.
    pub fn new(relative_path: impl Into<String>) -> Result<Self, SaltError> {
        let relative_path = checked_text("stage artifact path", relative_path.into())?;
        let path = Path::new(&relative_path);
        if path.is_absolute()
            || !path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
        {
            return Err(SaltError::InvalidField("stage artifact path"));
        }
        Ok(Self { relative_path })
    }

    /// Portable path relative to [`SaltPipeline::work_dir`].
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    fn validate(&self) -> bool {
        Self::new(self.relative_path.clone()).is_ok()
    }
}

impl PublishedArtifact {
    /// Construct a published-package receipt.
    pub fn new(package_id: ContentId, physical_bytes: u64) -> Result<Self, SaltError> {
        if physical_bytes == 0 {
            return Err(SaltError::InvalidField("published physical bytes"));
        }
        Ok(Self {
            package_id,
            physical_bytes,
        })
    }

    /// Exact installed package identity.
    pub const fn package_id(&self) -> ContentId {
        self.package_id
    }

    /// Exact installed package length.
    pub const fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    fn validate(&self) -> bool {
        valid_content_id(self.package_id) && self.physical_bytes != 0
    }
}

/// Evidence returned by a real backend for one completed stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageOutput {
    output_id: ContentId,
    artifact: Option<StageArtifact>,
    physical: Option<PhysicalLedger>,
    metrics: Vec<Metric>,
    hardware: Vec<HardwareUsage>,
    quality: Option<QualityEvidence>,
    published: Option<PublishedArtifact>,
}

impl StageOutput {
    /// Start an output receipt with the immutable stage-artifact identity.
    pub fn new(output_id: ContentId) -> Self {
        Self {
            output_id,
            artifact: None,
            physical: None,
            metrics: Vec::new(),
            hardware: Vec::new(),
            quality: None,
            published: None,
        }
    }

    /// Bind this output to a durable artifact below the stage work directory.
    pub fn with_artifact(mut self, artifact: StageArtifact) -> Self {
        self.artifact = Some(artifact);
        self
    }

    /// Attach the exact package ledger produced by packing.
    pub fn with_physical(mut self, physical: PhysicalLedger) -> Self {
        self.physical = Some(physical);
        self
    }

    /// Attach one evaluation metric.
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metrics.push(metric);
        self
    }

    /// Attach one measured hardware-usage record.
    pub fn with_hardware(mut self, usage: HardwareUsage) -> Self {
        self.hardware.push(usage);
        self
    }

    /// Attach the mandatory validation verdict.
    pub fn with_quality(mut self, quality: QualityEvidence) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Attach the mandatory publication receipt.
    pub fn with_published(mut self, published: PublishedArtifact) -> Self {
        self.published = Some(published);
        self
    }
}

/// Read-only input supplied to one real synthesis stage.
#[derive(Clone, Copy, Debug)]
pub struct StageRequest<'a> {
    stage: SaltStage,
    attempt: u32,
    spec: &'a SaltSpec,
    receipt: &'a SaltReceipt,
    work_dir: &'a Path,
}

impl<'a> StageRequest<'a> {
    /// Stage to execute idempotently.
    pub const fn stage(self) -> SaltStage {
        self.stage
    }

    /// One-based attempt number for this stage.
    pub const fn attempt(self) -> u32 {
        self.attempt
    }

    /// Immutable desired state.
    pub const fn spec(self) -> &'a SaltSpec {
        self.spec
    }

    /// Durable evidence from earlier stages.
    pub const fn receipt(self) -> &'a SaltReceipt {
        self.receipt
    }

    /// Content-addressed directory in which durable stage artifacts must be written.
    pub const fn work_dir(self) -> &'a Path {
        self.work_dir
    }
}

/// Structured failure returned by a synthesis backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverFailure {
    code: String,
    message: String,
    retryable: bool,
}

impl DriverFailure {
    /// Construct a stable backend failure.
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<Self, SaltError> {
        Ok(Self {
            code: checked_text("driver failure code", code.into())?,
            message: checked_text("driver failure message", message.into())?,
            retryable,
        })
    }

    /// Stable machine-readable code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the same immutable spec may retry.
    pub const fn retryable(&self) -> bool {
        self.retryable
    }
}

/// Boundary implemented by actual fitting, packing, evaluation, and publish code.
pub trait SaltDriver {
    /// Execute one idempotent, content-addressed stage.
    fn run_stage(&mut self, request: StageRequest<'_>) -> Result<StageOutput, DriverFailure>;
}

/// Immutable provenance copied into every work receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceReceipt {
    work_id: WorkId,
    source_id: ContentId,
    evidence_id: ContentId,
    recipe_id: ContentId,
    recipe_implementation: String,
    recipe_revision: String,
    profile: SaltProfile,
}

impl ProvenanceReceipt {
    fn matches_spec(&self, spec: &SaltSpec) -> bool {
        self.work_id == spec.work_id()
            && self.source_id == spec.source().id()
            && self.evidence_id == spec.evidence().id()
            && self.recipe_id == spec.recipe().id()
            && self.recipe_implementation == spec.recipe().implementation()
            && self.recipe_revision == spec.recipe().revision()
            && self.profile == spec.profile()
            && valid_text(&self.recipe_implementation)
            && valid_text(&self.recipe_revision)
    }

    /// Content-addressed complete spec.
    pub const fn work_id(&self) -> WorkId {
        self.work_id
    }

    /// Source model identity.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Calibration/evaluation evidence identity.
    pub const fn evidence_id(&self) -> ContentId {
        self.evidence_id
    }

    /// Synthesis recipe identity.
    pub const fn recipe_id(&self) -> ContentId {
        self.recipe_id
    }

    /// Pinned backend implementation.
    pub fn recipe_implementation(&self) -> &str {
        &self.recipe_implementation
    }

    /// Pinned backend revision.
    pub fn recipe_revision(&self) -> &str {
        &self.recipe_revision
    }

    /// Stable output profile.
    pub const fn profile(&self) -> SaltProfile {
        self.profile
    }
}

/// Digest receipt for one attempted stage output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageReceiptRecord {
    stage: SaltStage,
    attempt: u32,
    output_id: ContentId,
    artifact: Option<StageArtifact>,
    accepted: bool,
}

impl StageReceiptRecord {
    /// Pipeline stage.
    pub const fn stage(&self) -> SaltStage {
        self.stage
    }

    /// One-based attempt number.
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Immutable output identity.
    pub const fn output_id(&self) -> ContentId {
        self.output_id
    }

    /// Durable stage artifact, present for every accepted artifact-producing stage.
    pub const fn artifact(&self) -> Option<&StageArtifact> {
        self.artifact.as_ref()
    }

    /// Whether this output advanced the pipeline.
    pub const fn accepted(&self) -> bool {
        self.accepted
    }
}

/// Metric annotated with the stage that measured it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricReceipt {
    stage: SaltStage,
    attempt: u32,
    output_id: ContentId,
    metric: Metric,
}

impl MetricReceipt {
    /// Measuring stage.
    pub const fn stage(&self) -> SaltStage {
        self.stage
    }

    /// One-based producing stage attempt.
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Exact producing stage-output identity.
    pub const fn output_id(&self) -> ContentId {
        self.output_id
    }

    /// Measurement.
    pub const fn metric(&self) -> &Metric {
        &self.metric
    }
}

/// Hardware use annotated with the consuming stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareReceipt {
    stage: SaltStage,
    attempt: u32,
    output_id: ContentId,
    usage: HardwareUsage,
}

impl HardwareReceipt {
    /// Consuming stage.
    pub const fn stage(&self) -> SaltStage {
        self.stage
    }

    /// One-based consuming stage attempt.
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Exact consuming stage-output identity.
    pub const fn output_id(&self) -> ContentId {
        self.output_id
    }

    /// Measured usage.
    pub const fn usage(&self) -> &HardwareUsage {
        &self.usage
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureReceipt {
    stage: SaltStage,
    attempt: u32,
    code: String,
    message: String,
    retryable: bool,
}

impl FailureReceipt {
    /// Failed stage.
    pub const fn stage(&self) -> SaltStage {
        self.stage
    }

    /// One-based attempt number.
    pub const fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Stable failure code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the same immutable spec may retry.
    pub const fn retryable(&self) -> bool {
        self.retryable
    }

    fn validate(&self) -> bool {
        self.attempt != 0 && valid_text(&self.code) && valid_text(&self.message)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordDisposition {
    Accepted,
    QualityRejected,
    ContractRejected,
}

/// Durable evidence for one content-addressed synthesis run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaltReceipt {
    provenance: ProvenanceReceipt,
    stages: Vec<StageReceiptRecord>,
    physical: Option<PhysicalLedger>,
    metrics: Vec<MetricReceipt>,
    hardware: Vec<HardwareReceipt>,
    failures: Vec<FailureReceipt>,
    quality: Option<QualityEvidence>,
    published: Option<PublishedArtifact>,
}

impl SaltReceipt {
    fn new(spec: &SaltSpec) -> Self {
        Self {
            provenance: ProvenanceReceipt {
                work_id: spec.work_id(),
                source_id: spec.source().id(),
                evidence_id: spec.evidence().id(),
                recipe_id: spec.recipe().id(),
                recipe_implementation: spec.recipe().implementation().to_owned(),
                recipe_revision: spec.recipe().revision().to_owned(),
                profile: spec.profile(),
            },
            stages: Vec::new(),
            physical: None,
            metrics: Vec::new(),
            hardware: Vec::new(),
            failures: Vec::new(),
            quality: None,
            published: None,
        }
    }

    /// Immutable source/evidence/recipe binding.
    pub const fn provenance(&self) -> &ProvenanceReceipt {
        &self.provenance
    }

    /// Ordered stage-output digests, including rejected validation evidence.
    pub fn stage_receipts(&self) -> &[StageReceiptRecord] {
        &self.stages
    }

    /// Exact whole-model physical ledger after packing.
    pub const fn physical(&self) -> Option<&PhysicalLedger> {
        self.physical.as_ref()
    }

    /// Evaluation measurements with stage attribution.
    pub fn metrics(&self) -> &[MetricReceipt] {
        &self.metrics
    }

    /// Accelerator usage with stage attribution.
    pub fn hardware(&self) -> &[HardwareReceipt] {
        &self.hardware
    }

    /// Retained failed attempts in occurrence order.
    pub fn failures(&self) -> &[FailureReceipt] {
        &self.failures
    }

    /// Total recorded GPU seconds across all stages and accelerators.
    pub fn total_gpu_seconds(&self) -> u64 {
        self.hardware.iter().fold(0_u64, |total, record| {
            total.saturating_add(record.usage.gpu_seconds)
        })
    }

    /// Total recorded GPU hours across all stages and accelerators.
    pub fn total_gpu_hours(&self) -> f64 {
        self.total_gpu_seconds() as f64 / 3_600.0
    }

    /// Validation verdict, whether accepted or rejected.
    pub const fn quality(&self) -> Option<&QualityEvidence> {
        self.quality.as_ref()
    }

    /// Published immutable artifact, absent unless all gates passed.
    pub const fn published(&self) -> Option<&PublishedArtifact> {
        self.published.as_ref()
    }

    fn checked_total_gpu_seconds(&self) -> Option<u64> {
        self.hardware.iter().try_fold(0_u64, |total, record| {
            total.checked_add(record.usage.gpu_seconds)
        })
    }

    fn record_output(
        &mut self,
        stage: SaltStage,
        attempt: u32,
        output: StageOutput,
        disposition: RecordDisposition,
    ) -> Result<(), SaltError> {
        let output_id = output.output_id;
        let added_gpu_seconds = output
            .hardware
            .iter()
            .try_fold(0_u64, |total, usage| total.checked_add(usage.gpu_seconds));
        let existing_gpu_seconds = self.hardware.iter().try_fold(0_u64, |total, receipt| {
            total.checked_add(receipt.usage.gpu_seconds)
        });
        if added_gpu_seconds
            .and_then(|added| existing_gpu_seconds.and_then(|prior| prior.checked_add(added)))
            .is_none()
        {
            return Err(SaltError::InvalidField("GPU seconds overflow"));
        }
        self.stages.push(StageReceiptRecord {
            stage,
            attempt,
            output_id: output.output_id,
            artifact: output.artifact,
            accepted: disposition == RecordDisposition::Accepted,
        });
        if disposition == RecordDisposition::Accepted
            && let Some(physical) = output.physical
        {
            self.physical = Some(physical);
        }
        self.metrics
            .extend(output.metrics.into_iter().map(|metric| MetricReceipt {
                stage,
                attempt,
                output_id,
                metric,
            }));
        self.hardware
            .extend(output.hardware.into_iter().map(|usage| HardwareReceipt {
                stage,
                attempt,
                output_id,
                usage,
            }));
        if disposition != RecordDisposition::ContractRejected
            && let Some(quality) = output.quality
        {
            self.quality = Some(quality);
        }
        if disposition == RecordDisposition::Accepted
            && let Some(published) = output.published
        {
            self.published = Some(published);
        }
        Ok(())
    }

    fn record_failure(&mut self, stage: SaltStage, attempt: u32, failure: &DriverFailure) {
        self.failures.push(FailureReceipt {
            stage,
            attempt,
            code: failure.code.clone(),
            message: failure.message.clone(),
            retryable: failure.retryable,
        });
    }
}

/// Observable pipeline lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PipelineStatus {
    /// A stage is ready.
    Ready,
    /// A stage attempt is durably claimed.
    Running,
    /// The same spec may retry the current stage.
    RetryableFailure,
    /// Operator or recipe changes are required.
    TerminalFailure,
    /// Publication completed.
    Succeeded,
}

/// Result of one experimental pipeline advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdvanceOutcome {
    /// One stage completed and more work remains.
    Advanced(SaltStage),
    /// All stages were already or are now complete.
    Complete,
}

/// Experimental durable stage-by-stage SALT V2 pipeline.
#[derive(Debug)]
pub struct SaltPipeline {
    spec: SaltSpec,
    work_dir: PathBuf,
    run: ConversionRun,
    receipt: SaltReceipt,
    /// Stable inode whose exclusive OS lock protects the work item for this lifetime.
    _work_lock: WorkLock,
}

/// One process-owned lock handle.
///
/// Unix `flock` state follows the open file description across `fork`. Only the
/// process that acquired the lock may explicitly unlock it; a forked child must
/// merely close its duplicate so it cannot release the live parent's lock.
#[derive(Debug)]
struct WorkLock {
    file: fs::File,
    creator_pid: u32,
}

impl Drop for WorkLock {
    fn drop(&mut self) {
        if std::process::id() == self.creator_pid {
            let _ = self.file.unlock();
        }
    }
}

impl SaltPipeline {
    /// Start new work or resume an existing checkpoint for the identical spec.
    pub fn start(spec: &SaltSpec, work_root: impl AsRef<Path>) -> Result<Self, SaltError> {
        let work_root = work_root.as_ref();
        let work_lock = acquire_work_lock(work_root, spec.work_id())?;
        let work_dir = work_directory(work_root, spec.work_id());
        fs::create_dir_all(&work_dir).map_err(|error| fs_error("create work directory", error))?;
        let state_path = work_dir.join("state.bin");
        if state_path.exists() {
            return Self::load_locked(spec, work_dir, work_lock);
        }
        let pipeline = Self {
            spec: spec.clone(),
            work_dir,
            run: ConversionRun::new(*spec.work_id().as_bytes()),
            receipt: SaltReceipt::new(spec),
            _work_lock: work_lock,
        };
        pipeline.persist()?;
        Ok(pipeline)
    }

    /// Resume an existing checkpoint for the identical content-addressed spec.
    pub fn resume(spec: &SaltSpec, work_root: impl AsRef<Path>) -> Result<Self, SaltError> {
        let work_root = work_root.as_ref();
        let work_lock = acquire_work_lock(work_root, spec.work_id())?;
        let work_dir = work_directory(work_root, spec.work_id());
        Self::load_locked(spec, work_dir, work_lock)
    }

    /// Content-addressed work identity.
    pub const fn work_id(&self) -> WorkId {
        self.spec.work_id()
    }

    /// Durable work directory.
    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    /// Current lifecycle state.
    pub fn status(&self) -> PipelineStatus {
        match self.run.status() {
            RunStatus::Ready => PipelineStatus::Ready,
            RunStatus::Running => PipelineStatus::Running,
            RunStatus::RetryableFailure => PipelineStatus::RetryableFailure,
            RunStatus::TerminalFailure => PipelineStatus::TerminalFailure,
            RunStatus::Succeeded => PipelineStatus::Succeeded,
            _ => PipelineStatus::TerminalFailure,
        }
    }

    /// Stage awaiting work, or `None` after success.
    pub fn current_stage(&self) -> Result<Option<SaltStage>, SaltError> {
        self.run
            .current_stage()
            .map(stage_from_conversion)
            .transpose()
    }

    /// Durable accumulated receipt.
    pub const fn receipt(&self) -> &SaltReceipt {
        &self.receipt
    }

    /// Execute at most one real backend stage and atomically checkpoint evidence.
    pub fn advance(&mut self, driver: &mut impl SaltDriver) -> Result<AdvanceOutcome, SaltError> {
        match self.run.status() {
            RunStatus::Succeeded => return Ok(AdvanceOutcome::Complete),
            RunStatus::TerminalFailure => return Err(self.terminal_error()),
            _ => {}
        }
        let attempt = self
            .run
            .begin_stage()
            .map_err(|error| conversion_error("begin stage", error))?;
        self.persist()?;
        let stage = stage_from_conversion(attempt.stage())?;
        let request = StageRequest {
            stage,
            attempt: attempt.number(),
            spec: &self.spec,
            receipt: &self.receipt,
            work_dir: &self.work_dir,
        };
        let output = match driver.run_stage(request) {
            Ok(output) => output,
            Err(failure) => {
                self.run
                    .fail_stage(
                        failure.code.clone(),
                        failure.message.clone(),
                        failure.retryable,
                    )
                    .map_err(|error| conversion_error("record driver failure", error))?;
                self.receipt
                    .record_failure(stage, attempt.number(), &failure);
                self.persist()?;
                return Err(SaltError::DriverFailure {
                    stage,
                    code: failure.code,
                    message: failure.message,
                    retryable: failure.retryable,
                });
            }
        };

        if let Err(message) = self.validate_output(stage, &output) {
            return self.fail_contract(stage, attempt.number(), output, message);
        }
        if stage == SaltStage::Validate {
            let quality = output
                .quality
                .as_ref()
                .expect("validate_output requires quality")
                .clone();
            if !quality.passed {
                self.receipt.record_output(
                    stage,
                    attempt.number(),
                    output,
                    RecordDisposition::QualityRejected,
                )?;
                self.run
                    .fail_stage("quality_gate_failed", quality.summary.clone(), false)
                    .map_err(|error| conversion_error("record quality failure", error))?;
                self.persist()?;
                return Err(SaltError::QualityGateFailed {
                    work_id: self.spec.work_id(),
                    evidence: Box::new(quality),
                });
            }
        }

        let output_id = *output.output_id.as_bytes();
        self.receipt
            .record_output(stage, attempt.number(), output, RecordDisposition::Accepted)?;
        self.run
            .complete_stage(output_id)
            .map_err(|error| conversion_error("complete stage", error))?;
        self.persist()?;
        if self.run.status() == RunStatus::Succeeded {
            Ok(AdvanceOutcome::Complete)
        } else {
            Ok(AdvanceOutcome::Advanced(stage))
        }
    }

    fn validate_output(&self, stage: SaltStage, output: &StageOutput) -> Result<(), String> {
        validate_output_fields(output)?;
        self.verify_stage_artifact(stage, output)?;
        if stage != SaltStage::Pack && output.physical.is_some() {
            return Err(format!(
                "{} stage emitted pack-only physical evidence",
                stage
            ));
        }
        if stage != SaltStage::Validate && output.quality.is_some() {
            return Err(format!(
                "{} stage emitted validate-only quality evidence",
                stage
            ));
        }
        if stage != SaltStage::Publish && output.published.is_some() {
            return Err(format!(
                "{} stage emitted publish-only artifact evidence",
                stage
            ));
        }
        match stage {
            SaltStage::Pack => {
                let physical = output
                    .physical
                    .as_ref()
                    .ok_or_else(|| "pack stage omitted physical ledger".to_owned())?;
                if physical.package_id != output.output_id {
                    return Err(
                        "physical ledger package does not match pack output identity".to_owned(),
                    );
                }
                if !physical.satisfies(self.spec.profile()) {
                    return Err("package exceeds profile physical core bpw".to_owned());
                }
                let artifact = output
                    .artifact
                    .as_ref()
                    .expect("verify_stage_artifact requires pack artifact");
                let package_path = artifact_path(&self.work_dir, artifact)
                    .map_err(|error| format!("pack artifact path is invalid: {error}"))?;
                let measured = MeasuredPackage::from_file(package_path)
                    .map_err(|_| "remeasure packed artifact failed".to_owned())?;
                if measured.physical_bytes() != physical.package_bytes {
                    return Err("packed file length does not match physical ledger".to_owned());
                }
                if measured.id().as_bytes() != &physical.transport_package_id {
                    return Err(
                        "packed file transport identity does not match physical ledger".to_owned(),
                    );
                }
            }
            SaltStage::Validate => {
                let quality = output
                    .quality
                    .as_ref()
                    .ok_or_else(|| "validate stage omitted quality evidence".to_owned())?;
                if quality.evidence_id != self.spec.evidence().id() {
                    return Err("quality evidence identity does not match spec".to_owned());
                }
                let packed_id = self
                    .receipt
                    .stages
                    .iter()
                    .find(|receipt| receipt.stage == SaltStage::Pack && receipt.accepted)
                    .map(|receipt| receipt.output_id)
                    .ok_or_else(|| "validate stage has no accepted pack output".to_owned())?;
                if quality.package_id != packed_id {
                    return Err("quality evidence package does not match accepted pack".to_owned());
                }
            }
            SaltStage::Publish => {
                let published = output
                    .published
                    .as_ref()
                    .ok_or_else(|| "publish stage omitted artifact receipt".to_owned())?;
                let physical = self
                    .receipt
                    .physical
                    .as_ref()
                    .ok_or_else(|| "publish stage has no prior physical ledger".to_owned())?;
                if published.physical_bytes != physical.package_bytes {
                    return Err("published bytes do not match packed ledger".to_owned());
                }
                let packed_id = self
                    .receipt
                    .stages
                    .iter()
                    .find(|receipt| receipt.stage == SaltStage::Pack && receipt.accepted)
                    .map(|receipt| receipt.output_id)
                    .ok_or_else(|| "publish stage has no accepted pack output".to_owned())?;
                if published.package_id != packed_id {
                    return Err(
                        "published package id does not match accepted pack output".to_owned()
                    );
                }
                let quality =
                    self.receipt.quality.as_ref().ok_or_else(|| {
                        "publish stage has no accepted quality evidence".to_owned()
                    })?;
                if !quality.passed || quality.package_id != packed_id {
                    return Err(
                        "publish stage quality evidence does not accept packed artifact".to_owned(),
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn verify_stage_artifact(&self, stage: SaltStage, output: &StageOutput) -> Result<(), String> {
        if !stage_requires_artifact(stage) {
            if output.artifact.is_some() {
                return Err(format!("{stage} stage emitted an unexpected work artifact"));
            }
            return Ok(());
        }
        let artifact = output
            .artifact
            .as_ref()
            .ok_or_else(|| format!("{stage} stage omitted its durable work artifact"))?;
        verify_artifact(&self.work_dir, artifact, output.output_id)
            .map_err(|error| format!("{stage} stage artifact verification failed: {error}"))
    }

    fn fail_contract<T>(
        &mut self,
        stage: SaltStage,
        attempt: u32,
        output: StageOutput,
        message: String,
    ) -> Result<T, SaltError> {
        self.receipt
            .record_output(stage, attempt, output, RecordDisposition::ContractRejected)?;
        self.run
            .fail_stage("stage_contract_violation", message.clone(), false)
            .map_err(|error| conversion_error("record contract failure", error))?;
        self.persist()?;
        Err(SaltError::StageContractViolation { stage, message })
    }

    fn terminal_error(&self) -> SaltError {
        if let Some(quality) = &self.receipt.quality
            && !quality.passed
        {
            return SaltError::QualityGateFailed {
                work_id: self.spec.work_id(),
                evidence: Box::new(quality.clone()),
            };
        }
        let failure = self.run.failure();
        SaltError::TerminalFailure {
            stage: self
                .run
                .current_stage()
                .and_then(|stage| stage_from_conversion(stage).ok()),
            code: failure
                .map_or("terminal_failure", |failure| failure.code())
                .to_owned(),
            message: failure
                .map_or("pipeline is terminally failed", |failure| failure.message())
                .to_owned(),
        }
    }

    fn load_locked(
        spec: &SaltSpec,
        work_dir: PathBuf,
        work_lock: WorkLock,
    ) -> Result<Self, SaltError> {
        let state_path = work_dir.join("state.bin");
        let metadata =
            fs::metadata(&state_path).map_err(|error| fs_error("inspect pipeline state", error))?;
        if metadata.len() > MAX_STATE_BYTES {
            return Err(SaltError::Checkpoint("pipeline state exceeds size limit"));
        }
        let bytes =
            fs::read(&state_path).map_err(|error| fs_error("read pipeline state", error))?;
        let stored = decode_state(&bytes)?;
        if stored.work_id != spec.work_id() || !stored.receipt.provenance.matches_spec(spec) {
            return Err(SaltError::Checkpoint("checkpoint provenance mismatch"));
        }
        let run = ConversionRun::from_bytes(&stored.run)
            .map_err(|error| conversion_error("decode conversion run", error))?;
        if run.recipe_id() != spec.work_id().as_bytes() {
            return Err(SaltError::Checkpoint("checkpoint work id mismatch"));
        }
        let mut pipeline = Self {
            spec: spec.clone(),
            work_dir,
            run,
            receipt: stored.receipt,
            _work_lock: work_lock,
        };
        pipeline.validate_loaded_state()?;
        if pipeline.run.status() == RunStatus::Running {
            let stage = pipeline
                .current_stage()?
                .ok_or(SaltError::Checkpoint("running checkpoint lacks stage"))?;
            let attempt = pipeline.run.current_attempts();
            pipeline
                .run
                .recover_interrupted("interrupted", "worker stopped before checkpoint commit")
                .map_err(|error| conversion_error("recover interrupted stage", error))?;
            pipeline.receipt.record_failure(
                stage,
                attempt,
                &DriverFailure {
                    code: "interrupted".to_owned(),
                    message: "worker stopped before checkpoint commit".to_owned(),
                    retryable: true,
                },
            );
            pipeline.persist()?;
        }
        Ok(pipeline)
    }

    fn validate_loaded_state(&self) -> Result<(), SaltError> {
        self.validate_loaded_fields()?;
        let rejected = self.validate_loaded_stage_records()?;
        self.validate_loaded_failures(rejected)?;
        self.validate_loaded_evidence(rejected)
    }

    fn validate_loaded_fields(&self) -> Result<(), SaltError> {
        if self
            .run
            .failure()
            .is_some_and(|failure| !valid_text(failure.code()) || !valid_text(failure.message()))
        {
            return Err(SaltError::Checkpoint("invalid loaded run failure"));
        }
        for record in &self.receipt.stages {
            if record.attempt == 0 || !valid_content_id(record.output_id) {
                return Err(SaltError::Checkpoint("invalid loaded stage receipt"));
            }
            if record
                .artifact
                .as_ref()
                .is_some_and(|artifact| !artifact.validate())
            {
                return Err(SaltError::Checkpoint("invalid loaded stage artifact"));
            }
        }
        if self.receipt.metrics.iter().any(|record| {
            record.attempt == 0 || !valid_content_id(record.output_id) || !record.metric.validate()
        }) {
            return Err(SaltError::Checkpoint("invalid loaded metric"));
        }
        if self.receipt.hardware.iter().any(|record| {
            record.attempt == 0 || !valid_content_id(record.output_id) || !record.usage.validate()
        }) {
            return Err(SaltError::Checkpoint("invalid loaded hardware usage"));
        }
        if self.receipt.checked_total_gpu_seconds().is_none() {
            return Err(SaltError::Checkpoint("GPU seconds overflow"));
        }
        if self
            .receipt
            .physical
            .as_ref()
            .is_some_and(|physical| physical.validate().is_err())
        {
            return Err(SaltError::Checkpoint("invalid loaded physical ledger"));
        }
        if self
            .receipt
            .quality
            .as_ref()
            .is_some_and(|quality| !quality.validate())
        {
            return Err(SaltError::Checkpoint("invalid loaded quality evidence"));
        }
        if self
            .receipt
            .published
            .as_ref()
            .is_some_and(|published| !published.validate())
        {
            return Err(SaltError::Checkpoint("invalid loaded publication evidence"));
        }
        if self
            .receipt
            .failures
            .iter()
            .any(|failure| !failure.validate())
        {
            return Err(SaltError::Checkpoint("invalid loaded failure receipt"));
        }
        Ok(())
    }

    fn validate_loaded_stage_records(&self) -> Result<Option<&StageReceiptRecord>, SaltError> {
        let accepted_count = self.run.receipts().len();
        if self.receipt.stages.len() < accepted_count
            || self.receipt.stages.len() > accepted_count.saturating_add(1)
        {
            return Err(SaltError::Checkpoint("receipt/run stage count mismatch"));
        }
        for (receipt, run_receipt) in self
            .receipt
            .stages
            .iter()
            .take(accepted_count)
            .zip(self.run.receipts())
        {
            if !receipt.accepted
                || stage_from_conversion(run_receipt.stage())? != receipt.stage
                || run_receipt.attempts() != receipt.attempt
                || run_receipt.output_id() != receipt.output_id.as_bytes()
            {
                return Err(SaltError::Checkpoint("receipt/run stage mismatch"));
            }
            if stage_requires_artifact(receipt.stage) {
                let artifact = receipt
                    .artifact
                    .as_ref()
                    .ok_or(SaltError::Checkpoint("accepted stage artifact missing"))?;
                verify_artifact(&self.work_dir, artifact, receipt.output_id)
                    .map_err(|_| SaltError::Checkpoint("accepted stage artifact changed"))?;
            } else if receipt.artifact.is_some() {
                return Err(SaltError::Checkpoint("unexpected accepted stage artifact"));
            }
        }

        let rejected = self.receipt.stages.get(accepted_count);
        if let Some(rejected) = rejected {
            let current = self.run.current_stage().ok_or(SaltError::Checkpoint(
                "rejected receipt lacks current stage",
            ))?;
            let failure = self
                .run
                .failure()
                .ok_or(SaltError::Checkpoint("rejected receipt lacks run failure"))?;
            if rejected.accepted
                || self.run.status() != RunStatus::TerminalFailure
                || stage_from_conversion(current)? != rejected.stage
                || rejected.attempt != self.run.current_attempts()
                || failure.stage() != current
                || failure.retryable()
                || !matches!(
                    failure.code(),
                    "quality_gate_failed" | "stage_contract_violation"
                )
            {
                return Err(SaltError::Checkpoint("rejected stage/run mismatch"));
            }
            if failure.code() == "quality_gate_failed" {
                let artifact = rejected
                    .artifact
                    .as_ref()
                    .ok_or(SaltError::Checkpoint("rejected quality artifact missing"))?;
                verify_artifact(&self.work_dir, artifact, rejected.output_id)
                    .map_err(|_| SaltError::Checkpoint("rejected quality artifact changed"))?;
            }
        }
        Ok(rejected)
    }

    fn validate_loaded_failures(
        &self,
        rejected: Option<&StageReceiptRecord>,
    ) -> Result<(), SaltError> {
        let current_stage = self
            .run
            .current_stage()
            .map(stage_from_conversion)
            .transpose()?;
        let active_driver_failure = matches!(
            self.run.status(),
            RunStatus::RetryableFailure | RunStatus::TerminalFailure
        ) && rejected.is_none();
        let mut previous: Option<(usize, u32)> = None;
        for (position, failure) in self.receipt.failures.iter().enumerate() {
            let stage_position = salt_stage_position(failure.stage);
            if previous.is_some_and(|(prior_stage, prior_attempt)| {
                stage_position < prior_stage
                    || (stage_position == prior_stage && failure.attempt <= prior_attempt)
            }) {
                return Err(SaltError::Checkpoint("failure receipt order mismatch"));
            }
            previous = Some((stage_position, failure.attempt));

            let accepted = self
                .receipt
                .stages
                .iter()
                .find(|record| record.accepted && record.stage == failure.stage);
            let is_active = active_driver_failure
                && current_stage == Some(failure.stage)
                && failure.attempt == self.run.current_attempts();
            if let Some(accepted) = accepted {
                if failure.attempt >= accepted.attempt || !failure.retryable {
                    return Err(SaltError::Checkpoint("failure/accepted stage mismatch"));
                }
            } else if current_stage != Some(failure.stage)
                || failure.attempt > self.run.current_attempts()
                || (failure.attempt == self.run.current_attempts() && !is_active)
                || (!is_active && !failure.retryable)
            {
                return Err(SaltError::Checkpoint("failure/current stage mismatch"));
            }
            if is_active && position + 1 != self.receipt.failures.len() {
                return Err(SaltError::Checkpoint("active failure is not latest"));
            }
        }

        if active_driver_failure {
            let run_failure = self
                .run
                .failure()
                .ok_or(SaltError::Checkpoint("failed run lacks failure"))?;
            let receipt_failure = self
                .receipt
                .failures
                .last()
                .ok_or(SaltError::Checkpoint("failed run lacks failure receipt"))?;
            if current_stage != Some(receipt_failure.stage)
                || receipt_failure.attempt != self.run.current_attempts()
                || receipt_failure.code != run_failure.code()
                || receipt_failure.message != run_failure.message()
                || receipt_failure.retryable != run_failure.retryable()
            {
                return Err(SaltError::Checkpoint("run/failure receipt mismatch"));
            }
        }

        for accepted in self.receipt.stages.iter().filter(|record| record.accepted) {
            self.validate_failure_attempt_sequence(
                accepted.stage,
                accepted.attempt.saturating_sub(1),
            )?;
        }
        if let Some(current_stage) = current_stage {
            let expected = if active_driver_failure {
                self.run.current_attempts()
            } else {
                self.run.current_attempts().saturating_sub(1)
            };
            self.validate_failure_attempt_sequence(current_stage, expected)?;
        }
        Ok(())
    }

    fn validate_failure_attempt_sequence(
        &self,
        stage: SaltStage,
        expected_count: u32,
    ) -> Result<(), SaltError> {
        let mut actual_count = 0_u32;
        for failure in self
            .receipt
            .failures
            .iter()
            .filter(|failure| failure.stage == stage)
        {
            actual_count = actual_count
                .checked_add(1)
                .ok_or(SaltError::Checkpoint("failure attempt count overflow"))?;
            if failure.attempt != actual_count {
                return Err(SaltError::Checkpoint("failure attempt sequence mismatch"));
            }
        }
        if actual_count != expected_count {
            return Err(SaltError::Checkpoint("failure attempt count mismatch"));
        }
        Ok(())
    }

    fn validate_loaded_evidence(
        &self,
        rejected: Option<&StageReceiptRecord>,
    ) -> Result<(), SaltError> {
        for metric in &self.receipt.metrics {
            if !self.receipt.stages.iter().any(|record| {
                record.stage == metric.stage
                    && record.attempt == metric.attempt
                    && record.output_id == metric.output_id
            }) {
                return Err(SaltError::Checkpoint("metric lacks stage output"));
            }
        }
        for hardware in &self.receipt.hardware {
            if !self.receipt.stages.iter().any(|record| {
                record.stage == hardware.stage
                    && record.attempt == hardware.attempt
                    && record.output_id == hardware.output_id
            }) {
                return Err(SaltError::Checkpoint("hardware usage lacks stage output"));
            }
        }

        let accepted_pack = self
            .receipt
            .stages
            .iter()
            .find(|record| record.accepted && record.stage == SaltStage::Pack);
        match (accepted_pack, self.receipt.physical.as_ref()) {
            (None, None) => {}
            (Some(pack), Some(physical)) => {
                if physical.package_id != pack.output_id {
                    return Err(SaltError::Checkpoint(
                        "physical ledger package does not match accepted pack",
                    ));
                }
                if !physical.satisfies(self.spec.profile()) {
                    return Err(SaltError::Checkpoint(
                        "loaded physical ledger exceeds profile",
                    ));
                }
                let artifact = pack
                    .artifact
                    .as_ref()
                    .ok_or(SaltError::Checkpoint("accepted pack artifact missing"))?;
                let path = artifact_path(&self.work_dir, artifact)
                    .map_err(|_| SaltError::Checkpoint("accepted pack artifact changed"))?;
                let measured = MeasuredPackage::from_file(path)
                    .map_err(|_| SaltError::Checkpoint("accepted pack artifact changed"))?;
                if measured.physical_bytes() != physical.package_bytes
                    || measured.id().as_bytes() != &physical.transport_package_id
                {
                    return Err(SaltError::Checkpoint(
                        "physical ledger does not match packed artifact",
                    ));
                }
            }
            _ => {
                return Err(SaltError::Checkpoint(
                    "physical evidence is inconsistent with stage state",
                ));
            }
        }

        let accepted_validate = self
            .receipt
            .stages
            .iter()
            .find(|record| record.accepted && record.stage == SaltStage::Validate);
        let quality_rejected = rejected.is_some_and(|record| {
            record.stage == SaltStage::Validate
                && self
                    .run
                    .failure()
                    .is_some_and(|failure| failure.code() == "quality_gate_failed")
        });
        match (
            accepted_validate.is_some(),
            quality_rejected,
            self.receipt.quality.as_ref(),
        ) {
            (false, false, None) => {}
            (true, false, Some(quality)) if quality.passed => {
                self.validate_quality_binding(quality, accepted_pack)?;
            }
            (false, true, Some(quality)) if !quality.passed => {
                self.validate_quality_binding(quality, accepted_pack)?;
                let failure = self
                    .run
                    .failure()
                    .ok_or(SaltError::Checkpoint("quality rejection lacks failure"))?;
                if failure.message() != quality.summary {
                    return Err(SaltError::Checkpoint("quality failure summary mismatch"));
                }
            }
            _ => {
                return Err(SaltError::Checkpoint(
                    "quality evidence is inconsistent with stage state",
                ));
            }
        }

        let accepted_publish = self
            .receipt
            .stages
            .iter()
            .find(|record| record.accepted && record.stage == SaltStage::Publish);
        match (accepted_publish, self.receipt.published.as_ref()) {
            (None, None) => {}
            (Some(_), Some(published)) => {
                let pack = accepted_pack.ok_or(SaltError::Checkpoint(
                    "published artifact lacks accepted pack",
                ))?;
                let physical = self.receipt.physical.as_ref().ok_or(SaltError::Checkpoint(
                    "published artifact lacks physical ledger",
                ))?;
                let quality = self.receipt.quality.as_ref().ok_or(SaltError::Checkpoint(
                    "published artifact lacks quality evidence",
                ))?;
                if published.package_id != pack.output_id
                    || published.physical_bytes != physical.package_bytes
                    || !quality.passed
                    || quality.package_id != pack.output_id
                    || self.run.status() != RunStatus::Succeeded
                {
                    return Err(SaltError::Checkpoint(
                        "publication evidence is inconsistent with stage state",
                    ));
                }
            }
            _ => {
                return Err(SaltError::Checkpoint(
                    "publication evidence is inconsistent with stage state",
                ));
            }
        }
        if self.run.status() == RunStatus::Succeeded && accepted_publish.is_none() {
            return Err(SaltError::Checkpoint(
                "publication evidence is inconsistent with stage state",
            ));
        }
        Ok(())
    }

    fn validate_quality_binding(
        &self,
        quality: &QualityEvidence,
        accepted_pack: Option<&StageReceiptRecord>,
    ) -> Result<(), SaltError> {
        let pack = accepted_pack.ok_or(SaltError::Checkpoint(
            "quality evidence lacks accepted pack",
        ))?;
        if quality.evidence_id != self.spec.evidence().id() || quality.package_id != pack.output_id
        {
            return Err(SaltError::Checkpoint("quality evidence binding mismatch"));
        }
        Ok(())
    }

    fn persist(&self) -> Result<(), SaltError> {
        let run = self
            .run
            .to_bytes()
            .map_err(|error| conversion_error("encode conversion run", error))?;
        let stored = StoredState {
            work_id: self.spec.work_id(),
            run,
            receipt: self.receipt.clone(),
        };
        let bytes = encode_state(&stored)?;
        atomic_write(&self.work_dir, &bytes)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredState {
    work_id: WorkId,
    run: Vec<u8>,
    receipt: SaltReceipt,
}

fn stage_from_conversion(stage: ConversionStage) -> Result<SaltStage, SaltError> {
    match stage {
        ConversionStage::Ingest => Ok(SaltStage::Ingest),
        ConversionStage::Calibrate => Ok(SaltStage::Calibrate),
        ConversionStage::Profile => Ok(SaltStage::Profile),
        ConversionStage::Search => Ok(SaltStage::Search),
        ConversionStage::Refine => Ok(SaltStage::Refine),
        ConversionStage::Pack => Ok(SaltStage::Pack),
        ConversionStage::Validate => Ok(SaltStage::Validate),
        ConversionStage::Publish => Ok(SaltStage::Publish),
        _ => Err(SaltError::Checkpoint("unsupported conversion stage")),
    }
}

fn stage_requires_artifact(stage: SaltStage) -> bool {
    stage != SaltStage::Publish
}

fn validate_output_fields(output: &StageOutput) -> Result<(), String> {
    if !valid_content_id(output.output_id) {
        return Err("stage output identity is invalid".to_owned());
    }
    if output
        .artifact
        .as_ref()
        .is_some_and(|artifact| !artifact.validate())
    {
        return Err("stage artifact path is invalid".to_owned());
    }
    if output
        .physical
        .as_ref()
        .is_some_and(|physical| physical.validate().is_err())
    {
        return Err("physical ledger is invalid".to_owned());
    }
    if output.metrics.iter().any(|metric| !metric.validate()) {
        return Err("metric evidence is invalid".to_owned());
    }
    if output.hardware.iter().any(|usage| !usage.validate()) {
        return Err("hardware evidence is invalid".to_owned());
    }
    if output
        .hardware
        .iter()
        .try_fold(0_u64, |total, usage| total.checked_add(usage.gpu_seconds))
        .is_none()
    {
        return Err("GPU seconds overflow".to_owned());
    }
    if output
        .quality
        .as_ref()
        .is_some_and(|quality| !quality.validate())
    {
        return Err("quality evidence is invalid".to_owned());
    }
    if output
        .published
        .as_ref()
        .is_some_and(|published| !published.validate())
    {
        return Err("publication evidence is invalid".to_owned());
    }
    Ok(())
}

fn verify_artifact(
    work_dir: &Path,
    artifact: &StageArtifact,
    expected: ContentId,
) -> Result<(), SaltError> {
    let path = artifact_path(work_dir, artifact)?;
    if ContentId::from_path(&path)? != expected {
        return Err(SaltError::Checkpoint("stage artifact digest mismatch"));
    }
    Ok(())
}

fn artifact_path(work_dir: &Path, artifact: &StageArtifact) -> Result<PathBuf, SaltError> {
    let checked = StageArtifact::new(artifact.relative_path.clone())?;
    let mut path = work_dir.to_path_buf();
    for component in Path::new(checked.relative_path()).components() {
        let std::path::Component::Normal(name) = component else {
            return Err(SaltError::InvalidField("stage artifact path"));
        };
        path.push(name);
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| fs_error("inspect stage artifact", error))?;
        if metadata.file_type().is_symlink() {
            return Err(SaltError::InvalidField("stage artifact symlink"));
        }
    }
    Ok(path)
}

fn work_directory(root: &Path, work_id: WorkId) -> PathBuf {
    root.join(work_id.to_string())
}

fn acquire_work_lock(work_root: &Path, work_id: WorkId) -> Result<WorkLock, SaltError> {
    let lock_directory = work_root.join(WORK_LOCK_DIRECTORY);
    fs::create_dir_all(&lock_directory)
        .map_err(|error| fs_error("create pipeline lock directory", error))?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_directory.join(format!("{work_id}.lock")))
        .map_err(|error| fs_error("open pipeline work lock", error))?;
    match lock.try_lock() {
        Ok(()) => Ok(WorkLock {
            file: lock,
            creator_pid: std::process::id(),
        }),
        Err(fs::TryLockError::WouldBlock) => Err(SaltError::Checkpoint(
            "pipeline work item is already locked",
        )),
        Err(fs::TryLockError::Error(error)) => Err(fs_error("lock pipeline work item", error)),
    }
}

fn encode_state(state: &StoredState) -> Result<Vec<u8>, SaltError> {
    let payload = serde_json::to_vec(state)
        .map_err(|_| SaltError::Checkpoint("could not encode pipeline state"))?;
    let encoded_len = STATE_HEADER_BYTES
        .checked_add(payload.len() as u64)
        .and_then(|length| length.checked_add(STATE_CHECKSUM_BYTES))
        .ok_or(SaltError::Checkpoint("pipeline state length overflow"))?;
    if encoded_len > MAX_STATE_BYTES {
        return Err(SaltError::Checkpoint("pipeline state exceeds size limit"));
    }
    let mut out = Vec::with_capacity(encoded_len as usize);
    out.extend_from_slice(&STATE_MAGIC);
    out.push(STATE_VERSION);
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    let mut hasher = blake3::Hasher::new_derive_key(STATE_HASH_CONTEXT);
    hasher.update(&out);
    out.extend_from_slice(hasher.finalize().as_bytes());
    Ok(out)
}

fn decode_state(bytes: &[u8]) -> Result<StoredState, SaltError> {
    const HEADER_BYTES: usize = STATE_HEADER_BYTES as usize;
    const CHECKSUM_BYTES: usize = STATE_CHECKSUM_BYTES as usize;
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES || bytes[..4] != STATE_MAGIC {
        return Err(SaltError::Checkpoint("invalid pipeline state header"));
    }
    if bytes[4] != STATE_VERSION {
        return Err(SaltError::Checkpoint("unsupported pipeline state version"));
    }
    let mut length_bytes = [0_u8; 8];
    length_bytes.copy_from_slice(&bytes[5..13]);
    let payload_len = u64::from_le_bytes(length_bytes);
    if STATE_HEADER_BYTES
        .checked_add(payload_len)
        .and_then(|length| length.checked_add(STATE_CHECKSUM_BYTES))
        .is_none_or(|length| length > MAX_STATE_BYTES)
    {
        return Err(SaltError::Checkpoint("pipeline state exceeds size limit"));
    }
    let expected_len = HEADER_BYTES
        .checked_add(payload_len as usize)
        .and_then(|len| len.checked_add(CHECKSUM_BYTES))
        .ok_or(SaltError::Checkpoint("pipeline state length overflow"))?;
    if expected_len != bytes.len() {
        return Err(SaltError::Checkpoint("pipeline state length mismatch"));
    }
    let checksum_offset = bytes.len() - CHECKSUM_BYTES;
    let mut hasher = blake3::Hasher::new_derive_key(STATE_HASH_CONTEXT);
    hasher.update(&bytes[..checksum_offset]);
    if hasher.finalize().as_bytes() != &bytes[checksum_offset..] {
        return Err(SaltError::Checkpoint("pipeline state checksum mismatch"));
    }
    serde_json::from_slice(&bytes[HEADER_BYTES..checksum_offset])
        .map_err(|_| SaltError::Checkpoint("invalid pipeline state payload"))
}

fn atomic_write(work_dir: &Path, bytes: &[u8]) -> Result<(), SaltError> {
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_path = work_dir.join(format!("state.tmp.{}.{}", std::process::id(), nonce));
    let state_path = work_dir.join("state.bin");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|error| fs_error("create temporary pipeline state", error))?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temp_path);
        return Err(fs_error("write temporary pipeline state", error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temp_path, &state_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(fs_error("replace pipeline state", error));
    }
    // Directory sync is best-effort durability, like tensor_work_store's
    // sync_directory: windows cannot open a directory handle via File::open,
    // so the rename above is the strongest portable guarantee there.
    #[cfg(unix)]
    fs::File::open(work_dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| fs_error("sync pipeline state directory", error))?;
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, SaltError> {
    if !valid_text(&value) {
        return Err(SaltError::InvalidField(field));
    }
    Ok(value)
}

fn checked_sum<const N: usize>(values: [u64; N], field: &'static str) -> Result<u64, SaltError> {
    values
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
        .ok_or(SaltError::InvalidField(field))
}

fn valid_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024 * 1024
}

fn valid_content_id(value: ContentId) -> bool {
    value.as_bytes().iter().any(|byte| *byte != 0)
}

fn finite_canonical(value: f64) -> bool {
    value.is_finite() && (value != 0.0 || value.to_bits() == 0.0_f64.to_bits())
}

fn salt_stage_position(stage: SaltStage) -> usize {
    SaltStage::ALL
        .iter()
        .position(|candidate| *candidate == stage)
        .unwrap_or(usize::MAX)
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn fs_error(operation: &'static str, error: io::Error) -> SaltError {
    SaltError::Io {
        operation,
        kind: error.kind(),
    }
}

fn conversion_error(
    operation: &'static str,
    _error: tritium_quantize::ConversionError,
) -> SaltError {
    SaltError::Checkpoint(operation)
}
