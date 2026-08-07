//! Stable facade and resumable control plane for SALT V2 synthesis.
//!
//! This crate orchestrates model synthesis but deliberately does not pretend to
//! implement fitting, evaluation, packing, or publication. Those operations are
//! supplied through a driver boundary and produce content-addressed evidence.

// The Qwen3.6 admission/execution stack builds on unix-only staging primitives
// and is cfg(unix)-gated; the internals that exist to serve it still compile on
// other platforms but are unreferenced there.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

use std::{
    fmt, fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

mod pipeline;
mod qwen36_preflight;
mod qwen36_source_admission;
mod qwen36_tensor_work;
mod stage7_evidence;
mod tensor_work_store;

pub use pipeline::{
    AdvanceOutcome, DriverFailure, FailureReceipt, HardwareReceipt, HardwareUsage, Metric,
    MetricReceipt, PhysicalLedger, PipelineStatus, ProvenanceReceipt, PublishedArtifact,
    QualityEvidence, SaltDriver, SaltPipeline, SaltReceipt, StageArtifact, StageOutput,
    StageReceiptRecord, StageRequest,
};
pub use qwen36_preflight::{
    Qwen36CampaignPreflight, Qwen36CampaignPreflightError, Qwen36CampaignPreflightReceipt,
    Qwen36SourceIdentityStatus,
};
pub use qwen36_source_admission::{
    Qwen36AdmissionError, Qwen36AdmissionReceipt, Qwen36AdmittedSource, Qwen36LanguageCoverage,
    Qwen36SourceProof, Qwen36SourceProofError,
};
// Admission/execution symbols are unix-only (see qwen36_tensor_work).
pub use qwen36_tensor_work::{
    Qwen36AdditiveCampaignSpec, Qwen36AdditiveCampaignStore, Qwen36AdditiveInstallError,
    Qwen36AdditiveMasterReceipt, Qwen36AdditiveSlotState, Qwen36AdditiveWorkSlot,
    Qwen36AllocatedCampaignStore, Qwen36CompleteWorkspaceReceipt,
    Qwen36LanguageMtpWorkspaceReceipt, Qwen36PhysicalAllocationError,
    Qwen36PreservedSafetensorsError, Qwen36PreservedSafetensorsReceipt,
    Qwen36PreservedTensorDescriptor, Qwen36PreservedVisitError, Qwen36PtqDriverError,
    Qwen36PtqEvidenceCaptureError, Qwen36PtqEvidenceCaptureReceipt,
    Qwen36PtqEvidenceCaptureRequest, Qwen36PtqEvidenceCaptureSession, Qwen36PtqEvidenceCaptureTask,
    Qwen36PtqEvidenceDirectory, Qwen36PtqPackageLimits, Qwen36ScaleOnlyCampaignStore,
    Qwen36SelectedAllocationBindError, Qwen36SelectedAllocationReceipt,
    Qwen36SelectedAllocationSpec, Qwen36SelectedProfileReceipt, Qwen36TensorWorkError,
    Qwen36TensorWorkStore, Qwen36TensorWorkSummary, SharedForwardCaptureGroup,
    SharedForwardPlanError, SharedForwardTensor, collect_qwen36_ptq_evidence,
    plan_shared_forward_groups, reconcile_qwen36_ptq,
};
#[cfg(unix)]
pub use qwen36_tensor_work::{
    Qwen36AdmittedExecutionReceipt, Qwen36AdmittedExecutionSession, Qwen36ExecutionBackend,
    Qwen36ExecutionReplayError, Qwen36ExecutionSessionOpenError, Qwen36ExecutionVisitError,
    Qwen36FinalLogitsOutputBindingError, Qwen36FinalLogitsOutputBindingReceipt,
    Qwen36PackageAdmissionError, Qwen36PackageAdmissionReceipt, Qwen36PackageAdmittedCampaignStore,
    Qwen36PackageProfileReceipt, Qwen36PackageRuntimeLedger, Qwen36PackageScaleOnlyCampaignStore,
    Qwen36PackageVisitError, Qwen36PtqPackageError, Qwen36PtqPackagesReceipt,
    reconcile_qwen36_ptq_packages,
};
pub use stage7_evidence::{
    STAGE7_DATASETS, STAGE7_PARTITION_SEQUENCE_COUNT, STAGE7_SAMPLED_ROWS_SCHEMA,
    STAGE7_TOKEN_ENCODING, STAGE7_TOKEN_EVIDENCE_SCHEMA, STAGE7_TOKEN_PAYLOAD_BYTES,
    STAGE7_TOKEN_PAYLOAD_FILE, STAGE7_TOKENS_PER_SEQUENCE, Stage7DatasetContract,
    Stage7EvidenceError, Stage7Partition, Stage7TokenBatch, Stage7TokenEvidencePack,
    Stage7TokenEvidenceReceipt, stage7_prefixed_json_sha256,
};
pub use tensor_work_store::{
    TensorPayloadValidator, TensorPayloadWriter, TensorPutError, TensorRecordInfo,
    TensorRecordReader, TensorRecordReceipt, TensorRecordSpec, TensorValidatedPutError,
    TensorVisitError, TensorWorkError, TensorWorkStore,
};

const CONTENT_ID_CONTEXT: &str = "tritium salt content id v1";
const TREE_ID_CONTEXT: &str = "tritium salt content tree id v1";
const WORK_ID_CONTEXT: &str = "tritium salt work id v1";

/// Content identity of a source, evidence bundle, recipe, or stage artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId([u8; 32]);

impl ContentId {
    /// Hash exact bytes into a domain-separated content identity.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT);
        hasher.update(bytes);
        Self(*hasher.finalize().as_bytes())
    }

    /// Hash an exact byte stream without loading it fully into memory.
    pub fn from_reader(mut reader: impl Read) -> Result<Self, SaltError> {
        let mut hasher = blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT);
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(io_error("read content", error)),
            };
            hasher.update(&buffer[..read]);
        }
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// Hash an exact file without loading it fully into memory.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, SaltError> {
        let file = fs::File::open(path).map_err(|error| io_error("open content", error))?;
        Self::from_reader(file)
    }

    /// Hash a file or a sharded-model directory tree.
    ///
    /// Directory identity includes every regular file's portable relative path,
    /// exact length, and bytes in lexical path order. Symlinks and special files
    /// are rejected rather than followed, making repeated hashes fail closed.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, SaltError> {
        let path = path.as_ref();
        let metadata =
            fs::symlink_metadata(path).map_err(|error| io_error("inspect content path", error))?;
        if metadata.file_type().is_symlink() {
            return Err(SaltError::InvalidField("content path type"));
        }
        if metadata.is_file() {
            return Self::from_file(path);
        }
        if !metadata.is_dir() {
            return Err(SaltError::InvalidField("content path type"));
        }

        let mut files = Vec::new();
        collect_tree_files(path, path, &mut files)?;
        files.sort_by(|left, right| left.0.cmp(&right.0));

        let mut hasher = blake3::Hasher::new_derive_key(TREE_ID_CONTEXT);
        hasher.update(b"TSTREE\0\x01");
        for (relative, file_path) in files {
            hasher.update(&(relative.len() as u64).to_le_bytes());
            hasher.update(&relative);
            let file =
                fs::File::open(&file_path).map_err(|error| io_error("open tree content", error))?;
            let declared = file
                .metadata()
                .map_err(|error| io_error("inspect tree content", error))?
                .len();
            hasher.update(&declared.to_le_bytes());
            hash_reader_exact(&mut hasher, file, declared)?;
        }
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// Construct an identity from a trusted, previously verified digest.
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self(digest)
    }

    /// Return the raw digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_digest(f, "tsc1_", &self.0)
    }
}

/// Content identity of a complete immutable synthesis specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkId([u8; 32]);

impl WorkId {
    /// Return the raw digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for WorkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_digest(f, "tsw1_", &self.0)
    }
}

/// Source artifact selected for synthesis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRef {
    id: ContentId,
    location: String,
}

impl SourceRef {
    /// Construct a source reference with a non-empty location.
    pub fn new(id: ContentId, location: impl Into<String>) -> Result<Self, SaltError> {
        let id = checked_content_id("source content identity", id)?;
        let location = checked_string("source location", location.into())?;
        Ok(Self { id, location })
    }

    /// Exact source content identity.
    pub const fn id(&self) -> ContentId {
        self.id
    }

    /// Opaque source locator consumed by the synthesis driver.
    pub fn location(&self) -> &str {
        &self.location
    }
}

/// Immutable calibration/evaluation evidence tied to one source identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRef {
    id: ContentId,
    source_id: ContentId,
    location: String,
}

impl EvidenceRef {
    /// Construct an evidence reference with its bound source identity.
    pub fn new(
        id: ContentId,
        source_id: ContentId,
        location: impl Into<String>,
    ) -> Result<Self, SaltError> {
        let id = checked_content_id("evidence content identity", id)?;
        let source_id = checked_content_id("evidence source identity", source_id)?;
        let location = checked_string("evidence location", location.into())?;
        Ok(Self {
            id,
            source_id,
            location,
        })
    }

    /// Exact evidence content identity.
    pub const fn id(&self) -> ContentId {
        self.id
    }

    /// Source identity against which this evidence was measured.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Opaque evidence locator consumed by the synthesis driver.
    pub fn location(&self) -> &str {
        &self.location
    }
}

/// Pinned implementation and configuration of the real synthesis backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeRef {
    id: ContentId,
    implementation: String,
    revision: String,
}

impl RecipeRef {
    /// Construct a recipe reference.
    pub fn new(
        id: ContentId,
        implementation: impl Into<String>,
        revision: impl Into<String>,
    ) -> Result<Self, SaltError> {
        Ok(Self {
            id: checked_content_id("recipe content identity", id)?,
            implementation: checked_string("recipe implementation", implementation.into())?,
            revision: checked_string("recipe revision", revision.into())?,
        })
    }

    /// Exact recipe/configuration identity.
    pub const fn id(&self) -> ContentId {
        self.id
    }

    /// Backend implementation name.
    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    /// Backend source/configuration revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// Stable SALT V2 output contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SaltProfile {
    /// Deployment-first profile capped at 2.25 physical core-projection bpw.
    CompactV1,
    /// Strict non-inferiority profile capped at 4.0 physical core-projection bpw.
    NearLosslessV1,
}

impl SaltProfile {
    /// Maximum serialized core-projection rate, in thousandths of a bit/weight.
    pub const fn max_core_millibpw(self) -> u16 {
        match self {
            Self::CompactV1 => 2_250,
            Self::NearLosslessV1 => 4_000,
        }
    }
}

/// Immutable desired state for one SALT V2 synthesis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltSpec {
    source: SourceRef,
    evidence: EvidenceRef,
    recipe: RecipeRef,
    destination: String,
    profile: SaltProfile,
    work_id: WorkId,
}

impl SaltSpec {
    /// Construct and content-address a synthesis specification.
    ///
    /// Evidence from any source other than `source` is rejected before work can
    /// be created or resumed.
    pub fn new(
        source: SourceRef,
        evidence: EvidenceRef,
        recipe: RecipeRef,
        destination: impl Into<String>,
        profile: SaltProfile,
    ) -> Result<Self, SaltError> {
        if evidence.source_id != source.id {
            return Err(SaltError::EvidenceSourceMismatch {
                source: source.id,
                evidence_source: evidence.source_id,
            });
        }
        let destination = checked_string("destination", destination.into())?;
        let work_id = derive_work_id(&source, &evidence, &recipe, &destination, profile);
        Ok(Self {
            source,
            evidence,
            recipe,
            destination,
            profile,
            work_id,
        })
    }

    /// Selected full-precision source.
    pub fn source(&self) -> &SourceRef {
        &self.source
    }

    /// Pinned calibration/evaluation evidence.
    pub fn evidence(&self) -> &EvidenceRef {
        &self.evidence
    }

    /// Pinned synthesis backend and configuration.
    pub fn recipe(&self) -> &RecipeRef {
        &self.recipe
    }

    /// Publication destination consumed only by the publish stage.
    pub fn destination(&self) -> &str {
        &self.destination
    }

    /// Output profile and quality contract.
    pub const fn profile(&self) -> SaltProfile {
        self.profile
    }

    /// Content identity of this complete desired state.
    pub const fn work_id(&self) -> WorkId {
        self.work_id
    }
}

/// Ordered stages used by both the stable facade and experimental pipeline API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SaltStage {
    /// Fingerprint and validate the source artifact.
    Ingest,
    /// Build the pinned calibration sample and activation evidence.
    Calibrate,
    /// Measure sensitivities and baseline quality.
    Profile,
    /// Fit candidates and allocate exact physical bytes.
    Search,
    /// Apply optional reconstruction and teacher-guided recovery.
    Refine,
    /// Encode the selected additive-ternary package.
    Pack,
    /// Reload the package and run integrity and quality gates.
    Validate,
    /// Atomically install the immutable accepted package.
    Publish,
}

impl SaltStage {
    /// Canonical SALT V2 pipeline order.
    pub const ALL: [Self; 8] = [
        Self::Ingest,
        Self::Calibrate,
        Self::Profile,
        Self::Search,
        Self::Refine,
        Self::Pack,
        Self::Validate,
        Self::Publish,
    ];

    /// Stable machine-readable stage name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ingest => "ingest",
            Self::Calibrate => "calibrate",
            Self::Profile => "profile",
            Self::Search => "search",
            Self::Refine => "refine",
            Self::Pack => "pack",
            Self::Validate => "validate",
            Self::Publish => "publish",
        }
    }
}

impl fmt::Display for SaltStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Deterministic explanation of work without executing synthesis stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltExplanation {
    work_id: WorkId,
    profile: SaltProfile,
    stages: [SaltStage; 8],
}

impl SaltExplanation {
    /// Content-addressed desired state that would be reconciled.
    pub const fn work_id(&self) -> WorkId {
        self.work_id
    }

    /// Requested stable profile.
    pub const fn profile(&self) -> SaltProfile {
        self.profile
    }

    /// Exact ordered stage sequence used by reconciliation.
    pub const fn stages(&self) -> &[SaltStage; 8] {
        &self.stages
    }
}

/// Stable one-call SALT V2 facade.
#[derive(Clone, Copy, Debug, Default)]
pub struct SaltV2;

impl SaltV2 {
    /// Explain the exact content-addressed pipeline without running a backend.
    pub fn explain(spec: &SaltSpec) -> Result<SaltExplanation, SaltError> {
        Ok(SaltExplanation {
            work_id: spec.work_id,
            profile: spec.profile,
            stages: SaltStage::ALL,
        })
    }

    /// Reconcile the desired state through the same stages returned by
    /// [`Self::explain`]. Completed work is returned without invoking `driver`.
    pub fn reconcile(
        spec: &SaltSpec,
        work_root: impl AsRef<Path>,
        driver: &mut impl SaltDriver,
    ) -> Result<SaltReceipt, SaltError> {
        let mut pipeline = SaltPipeline::start(spec, work_root)?;
        loop {
            match pipeline.advance(driver)? {
                AdvanceOutcome::Advanced(_) => {}
                AdvanceOutcome::Complete => return Ok(pipeline.receipt().clone()),
            }
        }
    }
}

/// Why a SALT V2 specification or pipeline operation was rejected.
#[derive(Debug)]
#[non_exhaustive]
pub enum SaltError {
    /// A required string field was empty or too large for durable encoding.
    InvalidField(&'static str),
    /// Calibration/evaluation evidence belongs to another source model.
    EvidenceSourceMismatch {
        /// Selected source identity.
        source: ContentId,
        /// Source identity embedded in the evidence.
        evidence_source: ContentId,
    },
    /// A durable filesystem operation failed.
    Io {
        /// Stable operation name.
        operation: &'static str,
        /// Portable I/O error kind.
        kind: io::ErrorKind,
    },
    /// Durable state was malformed, inconsistent, or unsupported.
    Checkpoint(&'static str),
    /// A real backend stage failed after its attempt was checkpointed.
    DriverFailure {
        /// Failed stage.
        stage: SaltStage,
        /// Stable backend code.
        code: String,
        /// Human-readable diagnostic.
        message: String,
        /// Whether the identical spec may resume and retry.
        retryable: bool,
    },
    /// A backend output omitted or contradicted mandatory stage evidence.
    StageContractViolation {
        /// Rejected stage.
        stage: SaltStage,
        /// Violated evidence contract.
        message: String,
    },
    /// Validation evidence failed the requested stable profile.
    QualityGateFailed {
        /// Content-addressed work whose evidence was retained.
        work_id: WorkId,
        /// Reloaded quality evidence that prevented publication.
        evidence: Box<QualityEvidence>,
    },
    /// Work is terminally failed for a non-quality reason.
    TerminalFailure {
        /// Failed stage when available.
        stage: Option<SaltStage>,
        /// Stable failure code.
        code: String,
        /// Human-readable diagnostic.
        message: String,
    },
}

impl fmt::Display for SaltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField(field) => write!(f, "invalid {field}"),
            Self::EvidenceSourceMismatch {
                source,
                evidence_source,
            } => write!(
                f,
                "evidence source {evidence_source} does not match selected source {source}"
            ),
            Self::Io { operation, kind } => write!(f, "{operation} failed: {kind:?}"),
            Self::Checkpoint(message) => write!(f, "invalid SALT V2 checkpoint: {message}"),
            Self::DriverFailure {
                stage,
                code,
                message,
                retryable,
            } => write!(
                f,
                "SALT V2 driver failed at {stage} ({code}, retryable={retryable}): {message}"
            ),
            Self::StageContractViolation { stage, message } => {
                write!(f, "SALT V2 stage contract failed at {stage}: {message}")
            }
            Self::QualityGateFailed { work_id, evidence } => write!(
                f,
                "SALT V2 quality gate failed for {work_id}: {}",
                evidence.summary()
            ),
            Self::TerminalFailure {
                stage,
                code,
                message,
            } => match stage {
                Some(stage) => write!(f, "SALT V2 terminal failure at {stage} ({code}): {message}"),
                None => write!(f, "SALT V2 terminal failure ({code}): {message}"),
            },
        }
    }
}

impl std::error::Error for SaltError {}

fn checked_string(field: &'static str, value: String) -> Result<String, SaltError> {
    const MAX_STRING_BYTES: usize = 1024 * 1024;
    if value.is_empty() || value.len() > MAX_STRING_BYTES {
        return Err(SaltError::InvalidField(field));
    }
    Ok(value)
}

fn checked_content_id(field: &'static str, id: ContentId) -> Result<ContentId, SaltError> {
    if id.0 == [0; 32] {
        return Err(SaltError::InvalidField(field));
    }
    Ok(id)
}

fn derive_work_id(
    source: &SourceRef,
    evidence: &EvidenceRef,
    recipe: &RecipeRef,
    destination: &str,
    profile: SaltProfile,
) -> WorkId {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TSPEC");
    bytes.push(1);
    bytes.extend_from_slice(source.id.as_bytes());
    push_string(&mut bytes, &source.location);
    bytes.extend_from_slice(evidence.id.as_bytes());
    bytes.extend_from_slice(evidence.source_id.as_bytes());
    push_string(&mut bytes, &evidence.location);
    bytes.extend_from_slice(recipe.id.as_bytes());
    push_string(&mut bytes, &recipe.implementation);
    push_string(&mut bytes, &recipe.revision);
    push_string(&mut bytes, destination);
    bytes.push(match profile {
        SaltProfile::CompactV1 => 0,
        SaltProfile::NearLosslessV1 => 1,
    });
    let mut hasher = blake3::Hasher::new_derive_key(WORK_ID_CONTEXT);
    hasher.update(&bytes);
    WorkId(*hasher.finalize().as_bytes())
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn fmt_digest(f: &mut fmt::Formatter<'_>, prefix: &str, digest: &[u8; 32]) -> fmt::Result {
    f.write_str(prefix)?;
    for byte in digest {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

fn collect_tree_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(Vec<u8>, PathBuf)>,
) -> Result<(), SaltError> {
    let entries = fs::read_dir(directory).map_err(|error| io_error("read content tree", error))?;
    for entry in entries {
        let entry = entry.map_err(|error| io_error("read content tree entry", error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| io_error("inspect content tree entry", error))?;
        if metadata.file_type().is_symlink() {
            return Err(SaltError::InvalidField("content tree symlink"));
        }
        if metadata.is_dir() {
            collect_tree_files(root, &path, files)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| SaltError::InvalidField("content tree path"))?;
            files.push((portable_relative_path(relative)?, path));
        } else {
            return Err(SaltError::InvalidField("content tree entry type"));
        }
    }
    Ok(())
}

fn portable_relative_path(path: &Path) -> Result<Vec<u8>, SaltError> {
    let mut out = Vec::new();
    for component in path.components() {
        let std::path::Component::Normal(name) = component else {
            return Err(SaltError::InvalidField("content tree path"));
        };
        let name = name
            .to_str()
            .ok_or(SaltError::InvalidField("content tree path encoding"))?;
        if !out.is_empty() {
            out.push(b'/');
        }
        out.extend_from_slice(name.as_bytes());
    }
    if out.is_empty() {
        return Err(SaltError::InvalidField("content tree path"));
    }
    Ok(out)
}

fn hash_reader_exact(
    hasher: &mut blake3::Hasher,
    mut reader: impl Read,
    declared: u64,
) -> Result<(), SaltError> {
    let mut actual = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error("read tree content", error)),
        };
        actual = actual
            .checked_add(read as u64)
            .ok_or(SaltError::InvalidField("content length"))?;
        hasher.update(&buffer[..read]);
    }
    if actual != declared {
        return Err(SaltError::InvalidField("content changed while hashing"));
    }
    Ok(())
}

fn io_error(operation: &'static str, error: io::Error) -> SaltError {
    SaltError::Io {
        operation,
        kind: error.kind(),
    }
}
