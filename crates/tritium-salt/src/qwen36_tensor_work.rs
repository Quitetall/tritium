//! Durable language-plus-MTP tensor workspace for the pinned Qwen3.6 campaign.
//!
//! The immutable base manifest copies only the 360 source-precision tensors
//! required by the product artifact. Independently versioned additive campaigns
//! install the 506 canonical tensor masters without rewriting that base.

mod additive_master;
mod ptq_driver;

// Admission/execution symbols are unix-only (see additive_master).
pub use additive_master::{
    Qwen36AdditiveCampaignSpec, Qwen36AdditiveCampaignStore, Qwen36AdditiveInstallError,
    Qwen36AdditiveMasterReceipt, Qwen36AllocatedCampaignStore, Qwen36CompleteWorkspaceReceipt,
    Qwen36PhysicalAllocationError, Qwen36ScaleOnlyCampaignStore, Qwen36SelectedAllocationBindError,
    Qwen36SelectedAllocationReceipt, Qwen36SelectedAllocationSpec, Qwen36SelectedProfileReceipt,
};
#[cfg(unix)]
pub use additive_master::{
    Qwen36AdmittedExecutionReceipt, Qwen36AdmittedExecutionSession, Qwen36ExecutionBackend,
    Qwen36ExecutionReplayError, Qwen36ExecutionSessionOpenError, Qwen36ExecutionVisitError,
    Qwen36FinalLogitsOutputBindingError, Qwen36FinalLogitsOutputBindingReceipt,
    Qwen36PackageAdmissionError, Qwen36PackageAdmissionReceipt, Qwen36PackageAdmittedCampaignStore,
    Qwen36PackageProfileReceipt, Qwen36PackageRuntimeLedger, Qwen36PackageScaleOnlyCampaignStore,
    Qwen36PackageVisitError,
};
pub use ptq_driver::{
    Qwen36PtqDriverError, Qwen36PtqEvidenceCaptureError, Qwen36PtqEvidenceCaptureReceipt,
    Qwen36PtqEvidenceCaptureRequest, Qwen36PtqEvidenceCaptureSession, Qwen36PtqEvidenceCaptureTask,
    Qwen36PtqEvidenceDirectory, Qwen36PtqPackageLimits, SharedForwardCaptureGroup,
    SharedForwardPlanError, SharedForwardTensor, collect_qwen36_ptq_evidence,
    plan_shared_forward_groups, reconcile_qwen36_ptq,
};
#[cfg(unix)]
pub use ptq_driver::{
    Qwen36PtqPackageError, Qwen36PtqPackagesReceipt, reconcile_qwen36_ptq_packages,
};

use core::{convert::Infallible, fmt, fmt::Write as _};
use std::{
    error::Error,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use tritium_format::{
    ModelId, PackageHasher, PackageId, SemanticTensorHasher, salt_v2_master::SaltV2MasterError,
};
use tritium_nn::{NnError, Qwen35TensorStreamError};
use tritium_quantize::{
    Qwen35CoverageDisposition, Qwen35CoverageEntry, Qwen35SourceDtype, Qwen35TensorRole,
    Qwen35TensorScope,
};

use crate::{
    ContentId, Qwen36AdmittedSource, Qwen36SourceIdentityStatus, Qwen36SourceProof,
    Qwen36SourceProofError, TensorPutError, TensorRecordReceipt, TensorRecordSpec,
    TensorVisitError, TensorWorkError, TensorWorkStore,
    tensor_work_store::{absolute_path, create_temporary_file, ensure_durable_directory},
};

const WORK_DIRECTORY: &str = "tensor-work";
const WORK_VERSION_DIRECTORY: &str = "v1";
const PRESERVED_SLOT_DIRECTORY: &str = "preserved-slots";
const WORKSPACE_FILE: &str = "workspace.tq36w";
const SLOT_EXTENSION: &str = "twrref";
const WORKSPACE_MAGIC: [u8; 8] = *b"TSQ36WS\0";
const WORKSPACE_VERSION: u8 = 1;
const WORKSPACE_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 tensor workspace checksum v1";
const SLOT_KEY_CONTEXT: &str = "tritium qwen3.6 tensor workspace slot key v1";
const PRESERVED_SCHEMA_BYTES: &[u8] = b"tritium qwen3.6 preserved bf16 tensor record v1";
const PRESERVED_METADATA_MAGIC: [u8; 8] = *b"TSQ36PB\0";
const PRESERVED_METADATA_VERSION: u8 = 1;
const STREAM_CHUNK_BYTES: usize = 64 * 1024;
const CHECKSUM_BYTES: usize = 32;
const MAX_SLOT_BYTES: u64 = 20 * 1024 * 1024;
const MAX_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const MAX_ACTIVE_TENSORS: usize = 866;
const MAX_TENSOR_NAME_BYTES: usize = 64 * 1024;
const MAX_TENSOR_RANK: usize = 32;

trait PreservedTensorSource: fmt::Debug {
    fn try_visit_tensor_bytes(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        visit: &mut dyn FnMut(&[u8]) -> io::Result<()>,
    ) -> Result<u64, Qwen35TensorStreamError<io::Error>>;

    fn source_tensor_semantic_hasher(&self, name: &str) -> Result<SemanticTensorHasher, NnError>;
}

impl PreservedTensorSource for Qwen36AdmittedSource {
    fn try_visit_tensor_bytes(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        visit: &mut dyn FnMut(&[u8]) -> io::Result<()>,
    ) -> Result<u64, Qwen35TensorStreamError<io::Error>> {
        Qwen36AdmittedSource::try_visit_tensor_bytes(self, name, max_chunk_bytes, visit)
    }

    fn source_tensor_semantic_hasher(&self, name: &str) -> Result<SemanticTensorHasher, NnError> {
        Qwen36AdmittedSource::source_tensor_semantic_hasher(self, name)
    }
}

/// Exact structural totals for the pinned language-plus-MTP workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen36TensorWorkSummary {
    active_tensors: u64,
    additive_required: u64,
    additive_present: u64,
    preserved_tensors: u64,
    deferred_vision_tensors: u64,
    active_coefficients: u64,
    preserved_coefficients: u64,
    preserved_payload_bytes: u64,
}

impl Qwen36TensorWorkSummary {
    fn with_additive_present(mut self, present: u64) -> Result<Self, Qwen36TensorWorkError> {
        if present > self.additive_required {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master count",
            ));
        }
        self.additive_present = present;
        Ok(self)
    }

    /// Language and MTP tensors in current product scope.
    #[must_use]
    pub const fn active_tensors(self) -> u64 {
        self.active_tensors
    }

    /// Additive matrices required before final envelope sealing.
    #[must_use]
    pub const fn additive_required(self) -> u64 {
        self.additive_required
    }

    /// Canonical additive master artifacts installed in this workspace version.
    #[must_use]
    pub const fn additive_present(self) -> u64 {
        self.additive_present
    }

    /// Exact-BF16 tensors durably retained by this workspace.
    #[must_use]
    pub const fn preserved_tensors(self) -> u64 {
        self.preserved_tensors
    }

    /// Vision tensors bound by source proof but absent from product payload.
    #[must_use]
    pub const fn deferred_vision_tensors(self) -> u64 {
        self.deferred_vision_tensors
    }

    /// Language-plus-MTP coefficient denominator for physical-rate reporting.
    #[must_use]
    pub const fn active_coefficients(self) -> u64 {
        self.active_coefficients
    }

    /// Coefficients retained at exact source precision.
    #[must_use]
    pub const fn preserved_coefficients(self) -> u64 {
        self.preserved_coefficients
    }

    /// Exact raw BF16 payload bytes retained by this workspace.
    #[must_use]
    pub const fn preserved_payload_bytes(self) -> u64 {
        self.preserved_payload_bytes
    }

    /// Whether every additive slot has a canonical installed artifact.
    #[must_use]
    pub const fn complete(self) -> bool {
        self.additive_present == self.additive_required
    }
}

/// State of one named additive language/MTP tensor slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen36AdditiveSlotState {
    /// The immutable base manifest carries no campaign-specific master reference.
    MissingCanonicalMaster,
}

/// Proof-bound descriptor for one additive language/MTP tensor slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36AdditiveWorkSlot {
    name: String,
    dtype: Qwen35SourceDtype,
    shape: Vec<u64>,
    coefficients: u64,
    scope: Qwen35TensorScope,
    role: Qwen35TensorRole,
    source_tensor_digest: [u8; 32],
    state: Qwen36AdditiveSlotState,
}

impl Qwen36AdditiveWorkSlot {
    /// Canonical source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Validated source precision.
    #[must_use]
    pub const fn dtype(&self) -> Qwen35SourceDtype {
        self.dtype
    }

    /// Logical source dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Checked logical coefficient count.
    #[must_use]
    pub const fn coefficients(&self) -> u64 {
        self.coefficients
    }

    /// Language or MTP ownership.
    #[must_use]
    pub const fn scope(&self) -> Qwen35TensorScope {
        self.scope
    }

    /// Architecture-specific tensor role.
    #[must_use]
    pub const fn role(&self) -> Qwen35TensorRole {
        self.role
    }

    /// Architecture-framed semantic digest from the admitted source proof.
    #[must_use]
    pub const fn source_tensor_digest(&self) -> &[u8; 32] {
        &self.source_tensor_digest
    }

    /// Current artifact state.
    #[must_use]
    pub const fn state(&self) -> Qwen36AdditiveSlotState {
        self.state
    }
}

/// Durable receipt for the preserved-source workspace manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36LanguageMtpWorkspaceReceipt {
    workspace_id: ContentId,
    proof_id: ContentId,
    manifest_content_id: ContentId,
    source_model_id: ModelId,
    coverage_policy_digest: [u8; 32],
    identity_status: Qwen36SourceIdentityStatus,
    manifest_bytes: u64,
    summary: Qwen36TensorWorkSummary,
}

/// Immutable descriptor for one exact-BF16 tensor retained by the workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PreservedTensorDescriptor {
    name: String,
    shape: Vec<u64>,
    source_tensor_digest: [u8; 32],
    payload_bytes: u64,
}

impl Qwen36PreservedTensorDescriptor {
    /// Canonical Hugging Face tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Logical row-major tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Architecture-framed source semantic digest.
    #[must_use]
    pub const fn source_tensor_digest(&self) -> &[u8; 32] {
        &self.source_tensor_digest
    }

    /// Exact raw BF16 payload bytes.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

/// Exact identity and byte ledger for one deterministic preserved safetensors file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen36PreservedSafetensorsReceipt {
    package_id: PackageId,
    tensor_count: u64,
    header_bytes: u64,
    payload_bytes: u64,
    total_bytes: u64,
}

impl Qwen36PreservedSafetensorsReceipt {
    /// Identity of every exact safetensors byte.
    #[must_use]
    pub const fn package_id(self) -> PackageId {
        self.package_id
    }

    /// Number of exact-BF16 tensors in canonical name order.
    #[must_use]
    pub const fn tensor_count(self) -> u64 {
        self.tensor_count
    }

    /// Eight-byte prefix plus padded JSON header bytes.
    #[must_use]
    pub const fn header_bytes(self) -> u64 {
        self.header_bytes
    }

    /// Exact concatenated raw BF16 tensor payload bytes.
    #[must_use]
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    /// Complete safetensors file bytes.
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Failure while streaming the deterministic preserved safetensors artifact.
#[derive(Debug)]
pub enum Qwen36PreservedSafetensorsError<E> {
    /// Workspace, record, source-semantic, or bounded-header validation failed.
    Workspace(Qwen36TensorWorkError),
    /// The caller-owned staged output rejected bytes.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36PreservedSafetensorsError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace(error) => write!(formatter, "preserved safetensors failed: {error}"),
            Self::Sink(error) => write!(formatter, "preserved safetensors sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36PreservedSafetensorsError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workspace(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

impl Qwen36LanguageMtpWorkspaceReceipt {
    /// Content identity of exact canonical workspace-manifest bytes.
    #[must_use]
    pub const fn workspace_id(&self) -> ContentId {
        self.workspace_id
    }

    /// Durable source-admission proof identity.
    #[must_use]
    pub const fn proof_id(&self) -> ContentId {
        self.proof_id
    }

    /// Source semantic-manifest content identity.
    #[must_use]
    pub const fn manifest_content_id(&self) -> ContentId {
        self.manifest_content_id
    }

    /// Source semantic model identity.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Frozen per-tensor conversion-policy digest.
    #[must_use]
    pub const fn coverage_policy_digest(&self) -> &[u8; 32] {
        &self.coverage_policy_digest
    }

    /// Candidate-only or future officially authenticated source status.
    #[must_use]
    pub const fn identity_status(&self) -> Qwen36SourceIdentityStatus {
        self.identity_status
    }

    /// Exact canonical workspace-manifest bytes.
    #[must_use]
    pub const fn manifest_bytes(&self) -> u64 {
        self.manifest_bytes
    }

    /// Re-derived structural and byte totals.
    #[must_use]
    pub const fn summary(&self) -> Qwen36TensorWorkSummary {
        self.summary
    }
}

/// Same-admission-handle manager for Qwen3.6 language-plus-MTP tensor work.
#[derive(Debug)]
pub struct Qwen36TensorWorkStore<'a> {
    source: &'a dyn PreservedTensorSource,
    root: PathBuf,
    slots: PathBuf,
    objects: TensorWorkStore,
    plan: WorkspacePlan,
}

impl<'a> Qwen36TensorWorkStore<'a> {
    /// Open the versioned workspace nested beneath one admitted source proof.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for contradictory proof policy, invalid
    /// paths, or tensor-object store failures.
    pub fn open(admitted: &'a Qwen36AdmittedSource) -> Result<Self, Qwen36TensorWorkError> {
        let plan = WorkspacePlan::from_proof(admitted.proof())?;
        let root = admitted
            .work_dir()
            .join(WORK_DIRECTORY)
            .join(WORK_VERSION_DIRECTORY);
        Self::open_from_parts(admitted, root, plan)
    }

    fn open_from_parts(
        source: &'a dyn PreservedTensorSource,
        root: PathBuf,
        plan: WorkspacePlan,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let root = absolute_path(&root).map_err(Qwen36TensorWorkError::TensorStore)?;
        ensure_durable_directory(&root, "workspace root")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let slots = root.join(PRESERVED_SLOT_DIRECTORY);
        ensure_durable_directory(&slots, "preserved slot directory")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let objects = TensorWorkStore::open(&root).map_err(Qwen36TensorWorkError::TensorStore)?;
        Ok(Self {
            source,
            root,
            slots,
            objects,
            plan,
        })
    }

    /// Versioned Qwen tensor-work root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Canonical workspace-manifest path.
    #[must_use]
    pub fn workspace_path(&self) -> PathBuf {
        self.root.join(WORKSPACE_FILE)
    }

    /// All 506 proof-bound additive slots in canonical tensor-name order.
    #[must_use]
    pub fn additive_slots(&self) -> &[Qwen36AdditiveWorkSlot] {
        &self.plan.additive
    }

    /// Canonical names of the 360 exact-BF16 tensors retained by this workspace.
    pub fn preserved_tensor_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.plan.preserved.iter().map(|entry| entry.spec.name())
    }

    /// Clone the bounded canonical descriptor catalog for exact-BF16 tensors.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError::AllocationFailed`] if the small bounded
    /// descriptor catalog cannot be allocated.
    pub fn preserved_tensor_descriptors(
        &self,
    ) -> Result<Vec<Qwen36PreservedTensorDescriptor>, Qwen36TensorWorkError> {
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(self.plan.preserved.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for entry in &self.plan.preserved {
            descriptors.push(Qwen36PreservedTensorDescriptor {
                name: entry.spec.name().to_owned(),
                shape: entry.spec.shape().to_vec(),
                source_tensor_digest: *entry.spec.source_tensor_digest(),
                payload_bytes: entry.spec.payload_bytes(),
            });
        }
        Ok(descriptors)
    }

    /// Stream a deterministic single-file BF16 safetensors artifact.
    ///
    /// Tensors are ordered by canonical name, offsets are exact raw BF16 byte
    /// offsets, and the JSON header is space-padded to an eight-byte boundary.
    /// Callback effects are nontransactional; export callers must stage and
    /// publish only after the final receipt is returned.
    ///
    /// # Errors
    /// Fails closed on a zero chunk bound, corrupt or changed workspace record,
    /// source-semantic mismatch, bounded-header overflow/allocation failure, or
    /// caller sink failure.
    pub fn try_write_preserved_safetensors<E>(
        &self,
        max_chunk_bytes: usize,
        mut write: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<Qwen36PreservedSafetensorsReceipt, Qwen36PreservedSafetensorsError<E>> {
        if max_chunk_bytes == 0 {
            return Err(Qwen36PreservedSafetensorsError::Workspace(
                Qwen36TensorWorkError::WorkspaceMalformed("preserved safetensors chunk bound"),
            ));
        }
        let (_, manifest) = self
            .read_workspace_manifest()
            .map_err(Qwen36PreservedSafetensorsError::Workspace)?;
        let header = preserved_safetensors_header(&self.plan.preserved)
            .map_err(Qwen36PreservedSafetensorsError::Workspace)?;
        let header_len = u64::try_from(header.len()).map_err(|_| {
            Qwen36PreservedSafetensorsError::Workspace(Qwen36TensorWorkError::LengthOverflow(
                "preserved safetensors header",
            ))
        })?;
        let prefix = header_len.to_le_bytes();
        let mut hasher = PackageHasher::new();
        for bytes in [&prefix[..], &header] {
            hasher.update(bytes);
            write(bytes).map_err(Qwen36PreservedSafetensorsError::Sink)?;
        }
        let mut payload_bytes = 0_u64;
        for (receipt, expected) in manifest.preserved.iter().zip(&self.plan.preserved) {
            let visited = self
                .visit_preserved_receipt(receipt, expected, max_chunk_bytes, |chunk| {
                    hasher.update(chunk);
                    write(chunk)
                })
                .map_err(|error| match error {
                    Qwen36PreservedVisitError::Workspace(error) => {
                        Qwen36PreservedSafetensorsError::Workspace(error)
                    }
                    Qwen36PreservedVisitError::Sink(error) => {
                        Qwen36PreservedSafetensorsError::Sink(error)
                    }
                })?;
            payload_bytes = payload_bytes.checked_add(visited).ok_or_else(|| {
                Qwen36PreservedSafetensorsError::Workspace(Qwen36TensorWorkError::LengthOverflow(
                    "preserved safetensors payload",
                ))
            })?;
        }
        if payload_bytes != self.plan.summary.preserved_payload_bytes() {
            return Err(Qwen36PreservedSafetensorsError::Workspace(
                Qwen36TensorWorkError::WorkspaceMismatch("preserved safetensors payload ledger"),
            ));
        }
        let header_bytes = 8_u64.checked_add(header_len).ok_or_else(|| {
            Qwen36PreservedSafetensorsError::Workspace(Qwen36TensorWorkError::LengthOverflow(
                "preserved safetensors bytes",
            ))
        })?;
        let total_bytes = header_bytes.checked_add(payload_bytes).ok_or_else(|| {
            Qwen36PreservedSafetensorsError::Workspace(Qwen36TensorWorkError::LengthOverflow(
                "preserved safetensors bytes",
            ))
        })?;
        let tensor_count = u64::try_from(self.plan.preserved.len()).map_err(|_| {
            Qwen36PreservedSafetensorsError::Workspace(Qwen36TensorWorkError::LengthOverflow(
                "preserved safetensors tensor count",
            ))
        })?;
        Ok(Qwen36PreservedSafetensorsReceipt {
            package_id: hasher.finalize(),
            tensor_count,
            header_bytes,
            payload_bytes,
            total_bytes,
        })
    }

    /// Canonical verified descriptors for all 360 retained BF16 tensors.
    ///
    /// Descriptors remain borrowed from the immutable workspace plan. Callers
    /// can construct bounded container headers before streaming payloads with
    /// [`Self::try_visit_preserved_tensor`], without reopening source shards or
    /// widening BF16 values.
    pub fn preserved_tensor_specs(&self) -> impl ExactSizeIterator<Item = &TensorRecordSpec> {
        self.plan.preserved.iter().map(|entry| &entry.spec)
    }

    /// Stream one named preserved tensor after record and source-semantic verification.
    ///
    /// Callback effects are nontransactional because final content and semantic
    /// checks necessarily occur after the final callback.
    ///
    /// # Errors
    /// Returns [`Qwen36PreservedVisitError::Workspace`] for an unknown name,
    /// missing/corrupt workspace, changed record, or source-proof mismatch, and
    /// [`Qwen36PreservedVisitError::Sink`] without erasing a callback failure.
    pub fn try_visit_preserved_tensor<E>(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, Qwen36PreservedVisitError<E>> {
        let index = self
            .plan
            .preserved
            .binary_search_by(|entry| entry.spec.name().cmp(name))
            .map_err(|_| {
                Qwen36PreservedVisitError::Workspace(Qwen36TensorWorkError::UnknownPreservedTensor)
            })?;
        let expected = &self.plan.preserved[index];
        let (_, manifest) = self
            .read_workspace_manifest()
            .map_err(Qwen36PreservedVisitError::Workspace)?;
        self.visit_preserved_receipt(&manifest.preserved[index], expected, max_chunk_bytes, visit)
    }

    /// Stream, verify, and immutably install every preserved language/MTP tensor.
    ///
    /// Existing slots are strictly reopened and skipped. Missing slots stream
    /// exact BF16 bytes from retained admitted-source handles; no tensor is
    /// widened and no artifact-wide source copy is created. Workspace manifest
    /// publishes only after all 360 preserved records verify.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for source mutation, typed sink/store
    /// failure, corrupt resume state, policy mismatch, or durable publication
    /// failure.
    pub fn reconcile_preserved(
        &self,
    ) -> Result<Qwen36LanguageMtpWorkspaceReceipt, Qwen36TensorWorkError> {
        let mut receipts = Vec::new();
        receipts
            .try_reserve_exact(self.plan.preserved.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for entry in &self.plan.preserved {
            receipts.push(self.reconcile_entry(entry)?);
        }
        let manifest = WorkspaceManifest::from_plan(&self.plan, receipts)?;
        manifest.validate(&self.plan)?;
        let bytes = manifest.canonical_bytes()?;
        persist_exact(&self.workspace_path(), &bytes, "workspace manifest")?;
        self.reopen_workspace()
    }

    /// Strictly reopen the canonical workspace and every referenced tensor object.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for missing, corrupt, noncanonical,
    /// mismatched, or mutated workspace and record bytes.
    pub fn reopen_workspace(
        &self,
    ) -> Result<Qwen36LanguageMtpWorkspaceReceipt, Qwen36TensorWorkError> {
        let (bytes, manifest) = self.read_workspace_manifest()?;
        for (receipt, expected) in manifest.preserved.iter().zip(&self.plan.preserved) {
            self.verify_preserved_receipt(receipt, expected)?;
        }
        manifest.receipt(&bytes)
    }

    fn read_workspace_manifest(
        &self,
    ) -> Result<(Vec<u8>, WorkspaceManifest), Qwen36TensorWorkError> {
        let bytes = read_regular_bounded(
            &self.workspace_path(),
            MAX_WORKSPACE_BYTES as u64,
            "workspace manifest",
        )?;
        let manifest = WorkspaceManifest::from_canonical_bytes(&bytes)?;
        manifest.validate(&self.plan)?;
        Ok((bytes, manifest))
    }

    /// Require completeness in the immutable preserved-source base manifest.
    ///
    /// The base manifest intentionally remains incomplete and byte-stable.
    /// Open a [`Qwen36AdditiveCampaignStore`] to install and seal additive masters.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError::MissingAdditiveArtifacts`] after a valid
    /// preserved workspace reopens, or an earlier validation error.
    pub fn require_complete(
        &self,
    ) -> Result<Qwen36LanguageMtpWorkspaceReceipt, Qwen36TensorWorkError> {
        let receipt = self.reopen_workspace()?;
        let summary = receipt.summary();
        if !summary.complete() {
            return Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                expected: summary.additive_required(),
                present: summary.additive_present(),
            });
        }
        Ok(receipt)
    }

    fn reconcile_entry(
        &self,
        entry: &PreservedPlanEntry,
    ) -> Result<TensorRecordReceipt, Qwen36TensorWorkError> {
        let slot_path = self.slot_path(entry.spec.name());
        match fs::symlink_metadata(&slot_path) {
            Ok(_) => return self.reopen_slot(&slot_path, entry),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(work_io("inspect preserved slot", error)),
        }
        let receipt = self
            .objects
            .put(&entry.spec, |writer| {
                self.source
                    .try_visit_tensor_bytes(entry.spec.name(), STREAM_CHUNK_BYTES, &mut |chunk| {
                        writer.write_all(chunk)
                    })
                    .map(|_| ())
            })
            .map_err(map_preserved_put_error)?;
        let slot_bytes = receipt
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        persist_exact(&slot_path, &slot_bytes, "preserved slot")?;
        self.reopen_slot(&slot_path, entry)
    }

    fn reopen_slot(
        &self,
        slot_path: &Path,
        entry: &PreservedPlanEntry,
    ) -> Result<TensorRecordReceipt, Qwen36TensorWorkError> {
        let receipt = self.read_slot_receipt(slot_path, entry)?;
        self.verify_preserved_receipt(&receipt, entry)?;
        Ok(receipt)
    }

    fn read_slot_receipt(
        &self,
        slot_path: &Path,
        entry: &PreservedPlanEntry,
    ) -> Result<TensorRecordReceipt, Qwen36TensorWorkError> {
        let bytes = read_regular_bounded(slot_path, MAX_SLOT_BYTES, "preserved slot")?;
        let receipt = TensorRecordReceipt::from_canonical_bytes(&bytes)
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        if !receipt.matches_spec(&entry.spec) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "preserved slot descriptor",
            ));
        }
        Ok(receipt)
    }

    fn verify_preserved_receipt(
        &self,
        receipt: &TensorRecordReceipt,
        expected: &PreservedPlanEntry,
    ) -> Result<(), Qwen36TensorWorkError> {
        self.visit_preserved_receipt(receipt, expected, STREAM_CHUNK_BYTES, |_| {
            Ok::<(), Infallible>(())
        })
        .map(|_| ())
        .map_err(|error| match error {
            Qwen36PreservedVisitError::Workspace(error) => error,
            Qwen36PreservedVisitError::Sink(impossible) => match impossible {},
        })
    }

    fn visit_preserved_receipt<E>(
        &self,
        receipt: &TensorRecordReceipt,
        expected: &PreservedPlanEntry,
        max_chunk_bytes: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, Qwen36PreservedVisitError<E>> {
        if !receipt.matches_spec(&expected.spec) {
            return Err(Qwen36PreservedVisitError::Workspace(
                Qwen36TensorWorkError::WorkspaceMismatch("preserved receipt descriptor"),
            ));
        }
        let mut reader = self
            .objects
            .open_verified(receipt)
            .map_err(Qwen36TensorWorkError::TensorStore)
            .map_err(Qwen36PreservedVisitError::Workspace)?;
        let mut hasher = self
            .source
            .source_tensor_semantic_hasher(expected.spec.name())
            .map_err(Qwen36TensorWorkError::Source)
            .map_err(Qwen36PreservedVisitError::Workspace)?;
        reader
            .try_visit_payload(max_chunk_bytes, |chunk| {
                hasher.update(chunk);
                visit(chunk)
            })
            .map_err(|error| match error {
                TensorVisitError::Store(error) => {
                    Qwen36PreservedVisitError::Workspace(Qwen36TensorWorkError::TensorStore(error))
                }
                TensorVisitError::Sink(error) => Qwen36PreservedVisitError::Sink(error),
            })?;
        let actual = hasher.finalize().map_err(|_| {
            Qwen36PreservedVisitError::Workspace(Qwen36TensorWorkError::WorkspaceMalformed(
                "preserved semantic tensor",
            ))
        })?;
        if actual.name() != expected.spec.name()
            || actual.shape() != expected.spec.shape()
            || actual.content_digest() != expected.spec.source_tensor_digest()
        {
            return Err(Qwen36PreservedVisitError::Workspace(
                Qwen36TensorWorkError::WorkspaceMismatch("preserved source semantic digest"),
            ));
        }
        Ok(expected.spec.payload_bytes())
    }

    fn slot_path(&self, name: &str) -> PathBuf {
        let mut hasher = blake3::Hasher::new_derive_key(SLOT_KEY_CONTEXT);
        hasher.update(name.as_bytes());
        self.slots
            .join(format!("{}.{}", hasher.finalize().to_hex(), SLOT_EXTENSION))
    }
}

#[derive(Clone, Debug)]
struct PreservedPlanEntry {
    spec: TensorRecordSpec,
}

#[derive(Clone, Debug)]
struct WorkspacePlan {
    proof_id: ContentId,
    manifest_content_id: ContentId,
    source_model_id: ModelId,
    coverage_policy_digest: [u8; 32],
    identity_status: Qwen36SourceIdentityStatus,
    summary: Qwen36TensorWorkSummary,
    additive: Vec<Qwen36AdditiveWorkSlot>,
    preserved: Vec<PreservedPlanEntry>,
}

impl WorkspacePlan {
    fn from_proof(proof: &Qwen36SourceProof) -> Result<Self, Qwen36TensorWorkError> {
        let proof_id = proof
            .proof_id()
            .map_err(Qwen36TensorWorkError::SourceProof)?;
        let manifest_content_id = proof.manifest_content_id();
        let source_model_id = proof.source_model_id();
        let coverage_policy_digest = proof.coverage().policy_digest();
        let identity_status = proof.identity_status();
        let mut summary = Qwen36TensorWorkSummary {
            active_tensors: 0,
            additive_required: 0,
            additive_present: 0,
            preserved_tensors: 0,
            deferred_vision_tensors: 0,
            active_coefficients: 0,
            preserved_coefficients: 0,
            preserved_payload_bytes: 0,
        };
        let mut additive = Vec::new();
        additive
            .try_reserve_exact(proof.coverage().entries().len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        let mut preserved = Vec::new();
        preserved
            .try_reserve_exact(proof.coverage().entries().len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for (coverage, semantic) in proof
            .coverage()
            .entries()
            .iter()
            .zip(proof.manifest().tensors())
        {
            if coverage.name() != semantic.name() || coverage.shape() != semantic.shape() {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "proof coverage/manifest join",
                ));
            }
            match coverage.disposition() {
                Qwen35CoverageDisposition::ExcludedFutureVision => {
                    summary.deferred_vision_tensors = checked_add(
                        summary.deferred_vision_tensors,
                        1,
                        "deferred vision tensors",
                    )?;
                }
                Qwen35CoverageDisposition::AdditiveTernary => {
                    summary.active_tensors =
                        checked_add(summary.active_tensors, 1, "active tensors")?;
                    summary.additive_required =
                        checked_add(summary.additive_required, 1, "additive required tensors")?;
                    summary.active_coefficients = checked_add(
                        summary.active_coefficients,
                        coverage.coefficients(),
                        "active coefficients",
                    )?;
                    additive.push(Qwen36AdditiveWorkSlot {
                        name: coverage.name().to_owned(),
                        dtype: coverage.dtype(),
                        shape: coverage.shape().to_vec(),
                        coefficients: coverage.coefficients(),
                        scope: coverage.scope(),
                        role: coverage.role(),
                        source_tensor_digest: *semantic.content_digest(),
                        state: Qwen36AdditiveSlotState::MissingCanonicalMaster,
                    });
                }
                Qwen35CoverageDisposition::PreserveSource => {
                    summary.active_tensors =
                        checked_add(summary.active_tensors, 1, "active tensors")?;
                    summary.preserved_tensors =
                        checked_add(summary.preserved_tensors, 1, "preserved tensors")?;
                    summary.active_coefficients = checked_add(
                        summary.active_coefficients,
                        coverage.coefficients(),
                        "active coefficients",
                    )?;
                    summary.preserved_coefficients = checked_add(
                        summary.preserved_coefficients,
                        coverage.coefficients(),
                        "preserved coefficients",
                    )?;
                    let payload_bytes = coverage.coefficients().checked_mul(2).ok_or(
                        Qwen36TensorWorkError::LengthOverflow("preserved payload bytes"),
                    )?;
                    summary.preserved_payload_bytes = checked_add(
                        summary.preserved_payload_bytes,
                        payload_bytes,
                        "preserved payload bytes",
                    )?;
                    let schema_metadata = preserved_schema_metadata(
                        proof_id,
                        manifest_content_id,
                        coverage_policy_digest,
                        coverage,
                    );
                    let spec = TensorRecordSpec::new(
                        preserved_schema_id(),
                        source_model_id,
                        *semantic.content_digest(),
                        coverage.name(),
                        coverage.shape().to_vec(),
                        schema_metadata,
                        payload_bytes,
                    )
                    .map_err(Qwen36TensorWorkError::TensorStore)?;
                    preserved.push(PreservedPlanEntry { spec });
                }
            }
        }
        if summary.active_tensors != 866
            || summary.additive_required != 506
            || summary.preserved_tensors != 360
            || summary.deferred_vision_tensors != 333
            || summary.active_coefficients != 27_320_697_856
            || summary.preserved_coefficients != 2_671_616
            || summary.preserved_payload_bytes != 5_343_232
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "pinned language/MTP totals",
            ));
        }
        if additive.windows(2).any(|pair| pair[0].name >= pair[1].name) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive tensor order",
            ));
        }
        if preserved
            .windows(2)
            .any(|pair| pair[0].spec.name() >= pair[1].spec.name())
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "preserved tensor order",
            ));
        }
        additive.shrink_to_fit();
        preserved.shrink_to_fit();
        Ok(Self {
            proof_id,
            manifest_content_id,
            source_model_id,
            coverage_policy_digest,
            identity_status,
            summary,
            additive,
            preserved,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkspaceManifest {
    proof_id: ContentId,
    manifest_content_id: ContentId,
    source_model_id: ModelId,
    coverage_policy_digest: [u8; 32],
    identity_status: Qwen36SourceIdentityStatus,
    summary: Qwen36TensorWorkSummary,
    additive: Vec<Qwen36AdditiveWorkSlot>,
    preserved: Vec<TensorRecordReceipt>,
}

impl WorkspaceManifest {
    fn from_plan(
        plan: &WorkspacePlan,
        preserved: Vec<TensorRecordReceipt>,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let manifest = Self {
            proof_id: plan.proof_id,
            manifest_content_id: plan.manifest_content_id,
            source_model_id: plan.source_model_id,
            coverage_policy_digest: plan.coverage_policy_digest,
            identity_status: plan.identity_status,
            summary: plan.summary,
            additive: plan.additive.clone(),
            preserved,
        };
        manifest.validate(plan)?;
        Ok(manifest)
    }

    fn validate(&self, plan: &WorkspacePlan) -> Result<(), Qwen36TensorWorkError> {
        if self.proof_id != plan.proof_id
            || self.manifest_content_id != plan.manifest_content_id
            || self.source_model_id != plan.source_model_id
            || self.coverage_policy_digest != plan.coverage_policy_digest
            || self.identity_status != plan.identity_status
            || self.summary != plan.summary
            || self.additive != plan.additive
            || self.preserved.len() != plan.preserved.len()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "workspace source or policy binding",
            ));
        }
        for (receipt, expected) in self.preserved.iter().zip(&plan.preserved) {
            if !receipt.matches_spec(&expected.spec) {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "workspace preserved tensor",
                ));
            }
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        if self.additive.len() > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive tensor count",
            ));
        }
        if self.preserved.len() > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "preserved tensor count",
            ));
        }
        let mut output = Vec::new();
        output.extend_from_slice(&WORKSPACE_MAGIC);
        output.push(WORKSPACE_VERSION);
        output.extend_from_slice(self.proof_id.as_bytes());
        output.extend_from_slice(self.manifest_content_id.as_bytes());
        output.extend_from_slice(self.source_model_id.as_bytes());
        output.extend_from_slice(&self.coverage_policy_digest);
        output.push(identity_status_tag(self.identity_status));
        for value in summary_values(self.summary) {
            output.extend_from_slice(&value.to_le_bytes());
        }
        let additive_count = u32::try_from(self.additive.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive tensor count"))?;
        output.extend_from_slice(&additive_count.to_le_bytes());
        for slot in &self.additive {
            encode_additive_slot(&mut output, slot)?;
        }
        let count = u32::try_from(self.preserved.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("preserved tensor count"))?;
        output.extend_from_slice(&count.to_le_bytes());
        for receipt in &self.preserved {
            let bytes = receipt
                .canonical_bytes()
                .map_err(Qwen36TensorWorkError::TensorStore)?;
            let length = u32::try_from(bytes.len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("tensor receipt"))?;
            output.extend_from_slice(&length.to_le_bytes());
            output.extend_from_slice(&bytes);
        }
        let mut hasher = blake3::Hasher::new_derive_key(WORKSPACE_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() > MAX_WORKSPACE_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace too large",
            ));
        }
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if bytes.len() > MAX_WORKSPACE_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace too large",
            ));
        }
        if bytes.len() < WORKSPACE_MAGIC.len() + 1 + CHECKSUM_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "truncated workspace",
            ));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(WORKSPACE_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace checksum",
            ));
        }
        let mut cursor = WorkspaceCursor::new(payload);
        if cursor.take(WORKSPACE_MAGIC.len())? != WORKSPACE_MAGIC {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed("workspace magic"));
        }
        if cursor.u8()? != WORKSPACE_VERSION {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace version",
            ));
        }
        let proof_id = ContentId::from_digest(cursor.digest()?);
        let manifest_content_id = ContentId::from_digest(cursor.digest()?);
        let source_model_id = ModelId::from_digest(cursor.digest()?);
        let coverage_policy_digest = cursor.digest()?;
        let identity_status = identity_status_from_tag(cursor.u8()?)?;
        let summary = Qwen36TensorWorkSummary {
            active_tensors: cursor.u64()?,
            additive_required: cursor.u64()?,
            additive_present: cursor.u64()?,
            preserved_tensors: cursor.u64()?,
            deferred_vision_tensors: cursor.u64()?,
            active_coefficients: cursor.u64()?,
            preserved_coefficients: cursor.u64()?,
            preserved_payload_bytes: cursor.u64()?,
        };
        let additive_count = cursor.u32()? as usize;
        if additive_count > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive tensor count",
            ));
        }
        let mut additive = Vec::new();
        additive
            .try_reserve_exact(additive_count)
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for _ in 0..additive_count {
            additive.push(decode_additive_slot(&mut cursor)?);
        }
        let count = cursor.u32()? as usize;
        if count > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "preserved tensor count",
            ));
        }
        let mut preserved = Vec::new();
        preserved
            .try_reserve_exact(count)
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for _ in 0..count {
            let length = cursor.u32()? as usize;
            if length == 0 || length as u64 > MAX_SLOT_BYTES {
                return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                    "tensor receipt length",
                ));
            }
            preserved.push(
                TensorRecordReceipt::from_canonical_bytes(cursor.take(length)?)
                    .map_err(Qwen36TensorWorkError::TensorStore)?,
            );
        }
        if cursor.remaining() != 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace trailing bytes",
            ));
        }
        let manifest = Self {
            proof_id,
            manifest_content_id,
            source_model_id,
            coverage_policy_digest,
            identity_status,
            summary,
            additive,
            preserved,
        };
        if manifest.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical workspace",
            ));
        }
        Ok(manifest)
    }

    fn receipt(
        &self,
        bytes: &[u8],
    ) -> Result<Qwen36LanguageMtpWorkspaceReceipt, Qwen36TensorWorkError> {
        Ok(Qwen36LanguageMtpWorkspaceReceipt {
            workspace_id: ContentId::of_bytes(bytes),
            proof_id: self.proof_id,
            manifest_content_id: self.manifest_content_id,
            source_model_id: self.source_model_id,
            coverage_policy_digest: self.coverage_policy_digest,
            identity_status: self.identity_status,
            manifest_bytes: u64::try_from(bytes.len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("workspace manifest bytes"))?,
            summary: self.summary,
        })
    }
}

/// Failure while preparing or reopening Qwen3.6 tensor work.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36TensorWorkError {
    /// Durable source proof or proof identity failed validation.
    SourceProof(Qwen36SourceProofError),
    /// Generic tensor object framing, I/O, or validation failed.
    TensorStore(TensorWorkError),
    /// Canonical additive tensor-master framing or semantics failed.
    Master(SaltV2MasterError),
    /// Same-handle source bytes no longer match admitted semantic identity.
    Source(NnError),
    /// Workspace filesystem operation failed.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
    /// Required workspace path was a symlink or special file.
    InvalidPath(&'static str),
    /// Checked workspace arithmetic overflowed.
    LengthOverflow(&'static str),
    /// Fallible bounded allocation failed.
    AllocationFailed,
    /// Workspace bytes were malformed, unsupported, or noncanonical.
    WorkspaceMalformed(&'static str),
    /// Workspace receipt or record contradicted pinned proof policy.
    WorkspaceMismatch(&'static str),
    /// Existing immutable slot or manifest differed from expected exact bytes.
    ExistingArtifactMismatch(&'static str),
    /// Another process or same-handle mutation currently owns the additive campaign.
    CampaignLocked,
    /// Refined work requires a parent campaign and fixed-trit verification.
    RefinedCampaignRequiresParent,
    /// This platform cannot prove stable filesystem identities for campaign mutation.
    AdditiveCampaignUnsupportedPlatform,
    /// Requested tensor is not a preserved language/MTP tensor in the pinned plan.
    UnknownPreservedTensor,
    /// Requested tensor is not an additive language/MTP tensor in the pinned plan.
    UnknownAdditiveTensor,
    /// One expected additive campaign slot has not been installed.
    MissingAdditiveMaster {
        /// Canonical missing tensor name.
        name: String,
    },
    /// Final envelope remains blocked on canonical additive master artifacts.
    MissingAdditiveArtifacts {
        /// Required additive matrix count.
        expected: u64,
        /// Canonical additive master artifacts currently present.
        present: u64,
    },
}

impl fmt::Display for Qwen36TensorWorkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceProof(error) => write!(formatter, "Qwen3.6 source proof failed: {error}"),
            Self::TensorStore(error) => write!(formatter, "Qwen3.6 tensor store failed: {error}"),
            Self::Master(error) => write!(formatter, "Qwen3.6 tensor master failed: {error}"),
            Self::Source(error) => write!(formatter, "Qwen3.6 source stream failed: {error}"),
            Self::Io { operation, kind } => {
                write!(formatter, "Qwen3.6 tensor work {operation} failed: {kind}")
            }
            Self::InvalidPath(field) => write!(formatter, "invalid Qwen3.6 {field}"),
            Self::LengthOverflow(field) => write!(formatter, "Qwen3.6 {field} overflow"),
            Self::AllocationFailed => formatter.write_str("Qwen3.6 tensor work allocation failed"),
            Self::WorkspaceMalformed(field) => {
                write!(formatter, "malformed Qwen3.6 {field}")
            }
            Self::WorkspaceMismatch(field) => {
                write!(formatter, "Qwen3.6 workspace mismatches {field}")
            }
            Self::ExistingArtifactMismatch(field) => {
                write!(formatter, "existing Qwen3.6 {field} changed or is corrupt")
            }
            Self::CampaignLocked => {
                formatter.write_str("Qwen3.6 additive campaign is already locked or mutating")
            }
            Self::RefinedCampaignRequiresParent => formatter.write_str(
                "Qwen3.6 refined campaign requires parent-bound fixed-trit verification",
            ),
            Self::AdditiveCampaignUnsupportedPlatform => formatter
                .write_str("Qwen3.6 additive campaigns require Unix stable-file identity support"),
            Self::UnknownPreservedTensor => formatter.write_str("unknown preserved Qwen3.6 tensor"),
            Self::UnknownAdditiveTensor => formatter.write_str("unknown additive Qwen3.6 tensor"),
            Self::MissingAdditiveMaster { name } => {
                write!(formatter, "Qwen3.6 additive master is missing for {name}")
            }
            Self::MissingAdditiveArtifacts { expected, present } => write!(
                formatter,
                "Qwen3.6 workspace lacks additive master artifacts: expected {expected}, present {present}"
            ),
        }
    }
}

impl Error for Qwen36TensorWorkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SourceProof(error) => Some(error),
            Self::TensorStore(error) => Some(error),
            Self::Master(error) => Some(error),
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

/// Failure while streaming one verified preserved Qwen3.6 tensor.
#[derive(Debug)]
pub enum Qwen36PreservedVisitError<E> {
    /// Workspace, record, or source-semantic verification failed.
    Workspace(Qwen36TensorWorkError),
    /// Caller-provided sink stopped the stream.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36PreservedVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace(error) => write!(formatter, "preserved tensor visit failed: {error}"),
            Self::Sink(error) => write!(formatter, "preserved tensor sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36PreservedVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workspace(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

fn map_preserved_put_error(
    error: TensorPutError<Qwen35TensorStreamError<io::Error>>,
) -> Qwen36TensorWorkError {
    match error {
        TensorPutError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
        TensorPutError::Producer(Qwen35TensorStreamError::Source(error)) => {
            Qwen36TensorWorkError::Source(error)
        }
        TensorPutError::Producer(Qwen35TensorStreamError::Sink(error)) => {
            work_io("write preserved tensor", error)
        }
    }
}

fn preserved_schema_id() -> ContentId {
    ContentId::of_bytes(PRESERVED_SCHEMA_BYTES)
}

fn preserved_safetensors_header(
    entries: &[PreservedPlanEntry],
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let mut capacity = 64_usize;
    for entry in entries {
        capacity = capacity
            .checked_add(entry.spec.name().len().checked_mul(6).ok_or(
                Qwen36TensorWorkError::LengthOverflow("preserved safetensors header"),
            )?)
            .and_then(|value| {
                entry
                    .spec
                    .shape()
                    .len()
                    .checked_mul(24)
                    .and_then(|shape| value.checked_add(shape))
            })
            .and_then(|value| value.checked_add(128))
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "preserved safetensors header",
            ))?;
    }
    if capacity > MAX_WORKSPACE_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "preserved safetensors header size",
        ));
    }
    let mut header = String::new();
    header
        .try_reserve_exact(capacity)
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    header.push_str("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut offset = 0_u64;
    for entry in entries {
        let end = offset.checked_add(entry.spec.payload_bytes()).ok_or(
            Qwen36TensorWorkError::LengthOverflow("preserved safetensors offset"),
        )?;
        header.push(',');
        push_json_string(&mut header, entry.spec.name());
        header.push_str(":{\"dtype\":\"BF16\",\"shape\":[");
        for (index, dimension) in entry.spec.shape().iter().enumerate() {
            if index != 0 {
                header.push(',');
            }
            write!(&mut header, "{dimension}").map_err(|_| {
                Qwen36TensorWorkError::WorkspaceMalformed("preserved safetensors header")
            })?;
        }
        write!(&mut header, "],\"data_offsets\":[{offset},{end}]}}").map_err(|_| {
            Qwen36TensorWorkError::WorkspaceMalformed("preserved safetensors header")
        })?;
        offset = end;
    }
    header.push('}');
    let padding = (8 - header.len() % 8) % 8;
    header.extend(core::iter::repeat_n(' ', padding));
    if header.len() > capacity || header.len() > MAX_WORKSPACE_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "preserved safetensors header size",
        ));
    }
    Ok(header.into_bytes())
}

fn push_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let value = character as u8;
                output.push_str("\\u00");
                output.push(HEX[usize::from(value >> 4)] as char);
                output.push(HEX[usize::from(value & 0x0f)] as char);
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

fn preserved_schema_metadata(
    proof_id: ContentId,
    manifest_content_id: ContentId,
    coverage_policy_digest: [u8; 32],
    entry: &Qwen35CoverageEntry,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(8 + 1 + 32 * 3 + 4 + 8);
    output.extend_from_slice(&PRESERVED_METADATA_MAGIC);
    output.push(PRESERVED_METADATA_VERSION);
    output.extend_from_slice(proof_id.as_bytes());
    output.extend_from_slice(manifest_content_id.as_bytes());
    output.extend_from_slice(&coverage_policy_digest);
    output.extend_from_slice(&[
        dtype_tag(entry.dtype()),
        scope_tag(entry.scope()),
        role_tag(entry.role()),
        disposition_tag(entry.disposition()),
    ]);
    output.extend_from_slice(&entry.coefficients().to_le_bytes());
    output
}

fn encode_additive_slot(
    output: &mut Vec<u8>,
    slot: &Qwen36AdditiveWorkSlot,
) -> Result<(), Qwen36TensorWorkError> {
    if slot.name.is_empty() || slot.name.len() > MAX_TENSOR_NAME_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor name",
        ));
    }
    if slot.shape.is_empty()
        || slot.shape.len() > MAX_TENSOR_RANK
        || slot.shape.contains(&0)
        || checked_coefficients(&slot.shape)? != slot.coefficients
    {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor shape",
        ));
    }
    let name_len = u32::try_from(slot.name.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive tensor name"))?;
    let rank = u32::try_from(slot.shape.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive tensor rank"))?;
    output.extend_from_slice(&name_len.to_le_bytes());
    output.extend_from_slice(slot.name.as_bytes());
    output.extend_from_slice(&[
        dtype_tag(slot.dtype),
        scope_tag(slot.scope),
        role_tag(slot.role),
        additive_state_tag(slot.state),
    ]);
    output.extend_from_slice(&slot.coefficients.to_le_bytes());
    output.extend_from_slice(&slot.source_tensor_digest);
    output.extend_from_slice(&rank.to_le_bytes());
    for dimension in &slot.shape {
        output.extend_from_slice(&dimension.to_le_bytes());
    }
    Ok(())
}

fn decode_additive_slot(
    cursor: &mut WorkspaceCursor<'_>,
) -> Result<Qwen36AdditiveWorkSlot, Qwen36TensorWorkError> {
    let name_len = cursor.u32()? as usize;
    if name_len == 0 || name_len > MAX_TENSOR_NAME_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor name",
        ));
    }
    let name_text = std::str::from_utf8(cursor.take(name_len)?)
        .map_err(|_| Qwen36TensorWorkError::WorkspaceMalformed("additive tensor name"))?;
    let mut name = String::new();
    name.try_reserve_exact(name_len)
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    name.push_str(name_text);
    let dtype = dtype_from_tag(cursor.u8()?)?;
    let scope = scope_from_tag(cursor.u8()?)?;
    let role = role_from_tag(cursor.u8()?)?;
    let state = additive_state_from_tag(cursor.u8()?)?;
    let coefficients = cursor.u64()?;
    let source_tensor_digest = cursor.digest()?;
    let rank = cursor.u32()? as usize;
    if rank == 0 || rank > MAX_TENSOR_RANK {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor rank",
        ));
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    for _ in 0..rank {
        let dimension = cursor.u64()?;
        if dimension == 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive tensor shape",
            ));
        }
        shape.push(dimension);
    }
    if checked_coefficients(&shape)? != coefficients {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor coefficients",
        ));
    }
    Ok(Qwen36AdditiveWorkSlot {
        name,
        dtype,
        shape,
        coefficients,
        scope,
        role,
        source_tensor_digest,
        state,
    })
}

fn checked_coefficients(shape: &[u64]) -> Result<u64, Qwen36TensorWorkError> {
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "additive tensor coefficients",
            ))
    })
}

const fn dtype_tag(dtype: Qwen35SourceDtype) -> u8 {
    match dtype {
        Qwen35SourceDtype::Bfloat16 => 1,
    }
}

fn dtype_from_tag(tag: u8) -> Result<Qwen35SourceDtype, Qwen36TensorWorkError> {
    match tag {
        1 => Ok(Qwen35SourceDtype::Bfloat16),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor dtype",
        )),
    }
}

const fn scope_tag(scope: Qwen35TensorScope) -> u8 {
    match scope {
        Qwen35TensorScope::Language => 1,
        Qwen35TensorScope::MtpDrafter => 2,
        Qwen35TensorScope::DeferredVision => 3,
    }
}

fn scope_from_tag(tag: u8) -> Result<Qwen35TensorScope, Qwen36TensorWorkError> {
    match tag {
        1 => Ok(Qwen35TensorScope::Language),
        2 => Ok(Qwen35TensorScope::MtpDrafter),
        3 => Ok(Qwen35TensorScope::DeferredVision),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor scope",
        )),
    }
}

const fn role_tag(role: Qwen35TensorRole) -> u8 {
    match role {
        Qwen35TensorRole::TokenEmbedding => 1,
        Qwen35TensorRole::OutputHead => 2,
        Qwen35TensorRole::Normalization => 3,
        Qwen35TensorRole::MlpProjection => 4,
        Qwen35TensorRole::FullAttentionProjection => 5,
        Qwen35TensorRole::DeltaNetProjection => 6,
        Qwen35TensorRole::DeltaNetState => 7,
        Qwen35TensorRole::DeltaNetConvolution => 8,
        Qwen35TensorRole::MtpFusionProjection => 9,
        Qwen35TensorRole::VisionAttentionProjection => 10,
        Qwen35TensorRole::VisionMlpProjection => 11,
        Qwen35TensorRole::VisionPatchEmbedding => 12,
        Qwen35TensorRole::VisionPositionalEmbedding => 13,
        Qwen35TensorRole::VisionMergerProjection => 14,
        Qwen35TensorRole::Bias => 15,
    }
}

fn role_from_tag(tag: u8) -> Result<Qwen35TensorRole, Qwen36TensorWorkError> {
    match tag {
        1 => Ok(Qwen35TensorRole::TokenEmbedding),
        2 => Ok(Qwen35TensorRole::OutputHead),
        3 => Ok(Qwen35TensorRole::Normalization),
        4 => Ok(Qwen35TensorRole::MlpProjection),
        5 => Ok(Qwen35TensorRole::FullAttentionProjection),
        6 => Ok(Qwen35TensorRole::DeltaNetProjection),
        7 => Ok(Qwen35TensorRole::DeltaNetState),
        8 => Ok(Qwen35TensorRole::DeltaNetConvolution),
        9 => Ok(Qwen35TensorRole::MtpFusionProjection),
        10 => Ok(Qwen35TensorRole::VisionAttentionProjection),
        11 => Ok(Qwen35TensorRole::VisionMlpProjection),
        12 => Ok(Qwen35TensorRole::VisionPatchEmbedding),
        13 => Ok(Qwen35TensorRole::VisionPositionalEmbedding),
        14 => Ok(Qwen35TensorRole::VisionMergerProjection),
        15 => Ok(Qwen35TensorRole::Bias),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor role",
        )),
    }
}

const fn additive_state_tag(state: Qwen36AdditiveSlotState) -> u8 {
    match state {
        Qwen36AdditiveSlotState::MissingCanonicalMaster => 0,
    }
}

fn additive_state_from_tag(tag: u8) -> Result<Qwen36AdditiveSlotState, Qwen36TensorWorkError> {
    match tag {
        0 => Ok(Qwen36AdditiveSlotState::MissingCanonicalMaster),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive tensor state",
        )),
    }
}

const fn disposition_tag(disposition: Qwen35CoverageDisposition) -> u8 {
    match disposition {
        Qwen35CoverageDisposition::AdditiveTernary => 1,
        Qwen35CoverageDisposition::PreserveSource => 2,
        Qwen35CoverageDisposition::ExcludedFutureVision => 3,
    }
}

const fn identity_status_tag(status: Qwen36SourceIdentityStatus) -> u8 {
    match status {
        Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration => 1,
    }
}

fn identity_status_from_tag(tag: u8) -> Result<Qwen36SourceIdentityStatus, Qwen36TensorWorkError> {
    match tag {
        1 => Ok(Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration),
        _ => Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "source identity status",
        )),
    }
}

const fn summary_values(summary: Qwen36TensorWorkSummary) -> [u64; 8] {
    [
        summary.active_tensors,
        summary.additive_required,
        summary.additive_present,
        summary.preserved_tensors,
        summary.deferred_vision_tensors,
        summary.active_coefficients,
        summary.preserved_coefficients,
        summary.preserved_payload_bytes,
    ]
}

fn checked_add(left: u64, right: u64, field: &'static str) -> Result<u64, Qwen36TensorWorkError> {
    left.checked_add(right)
        .ok_or(Qwen36TensorWorkError::LengthOverflow(field))
}

fn read_regular_bounded(
    path: &Path,
    maximum: u64,
    field: &'static str,
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let before = fs::symlink_metadata(path).map_err(|error| work_io("inspect artifact", error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > maximum {
        return Err(Qwen36TensorWorkError::InvalidPath(field));
    }
    let mut file = File::open(path).map_err(|error| work_io("open artifact", error))?;
    let opened = file
        .metadata()
        .map_err(|error| work_io("inspect opened artifact", error))?;
    let after = fs::symlink_metadata(path).map_err(|error| work_io("reinspect artifact", error))?;
    if after.file_type().is_symlink()
        || !after.is_file()
        || !same_file_identity(&before, &opened)
        || !same_file_identity(&opened, &after)
        || opened.len() != before.len()
        || opened.len() != after.len()
    {
        return Err(Qwen36TensorWorkError::InvalidPath(field));
    }
    let length =
        usize::try_from(opened.len()).map_err(|_| Qwen36TensorWorkError::LengthOverflow(field))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    let read_limit = maximum
        .checked_add(1)
        .ok_or(Qwen36TensorWorkError::LengthOverflow(field))?;
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| work_io("read artifact", error))?;
    if bytes.len() != length {
        return Err(Qwen36TensorWorkError::ExistingArtifactMismatch(field));
    }
    Ok(bytes)
}

fn persist_exact(
    path: &Path,
    bytes: &[u8],
    field: &'static str,
) -> Result<(), Qwen36TensorWorkError> {
    let parent = path
        .parent()
        .ok_or(Qwen36TensorWorkError::InvalidPath("artifact parent"))?;
    match fs::symlink_metadata(path) {
        Ok(_) => {
            let existing = read_regular_bounded(path, MAX_WORKSPACE_BYTES as u64, field)?;
            if existing == bytes {
                return sync_directory(parent, "sync existing artifact directory");
            }
            return Err(Qwen36TensorWorkError::ExistingArtifactMismatch(field));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(work_io("inspect artifact destination", error)),
    }
    ensure_durable_directory(parent, "artifact parent")
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    let prefix = format!(".{}.tmp", field.replace(' ', "-"));
    let (temporary, mut file) =
        create_temporary_file(parent, &prefix).map_err(Qwen36TensorWorkError::TensorStore)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(work_io("write temporary artifact", error));
    }
    drop(file);
    match fs::hard_link(&temporary, path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let existing = read_regular_bounded(path, MAX_WORKSPACE_BYTES as u64, field)?;
            if existing != bytes {
                let _ = fs::remove_file(&temporary);
                return Err(Qwen36TensorWorkError::ExistingArtifactMismatch(field));
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(work_io("publish artifact", error));
        }
    }
    if let Err(error) = sync_directory(parent, "sync artifact directory") {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::remove_file(&temporary).map_err(|error| work_io("remove temporary artifact", error))?;
    sync_directory(parent, "resync artifact directory")
}

#[cfg(unix)]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.is_file() == right.is_file() && left.len() == right.len()
}

#[cfg(unix)]
fn sync_directory(path: &Path, operation: &'static str) -> Result<(), Qwen36TensorWorkError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| work_io(operation, error))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path, _operation: &'static str) -> Result<(), Qwen36TensorWorkError> {
    Ok(())
}

fn work_io(operation: &'static str, error: io::Error) -> Qwen36TensorWorkError {
    Qwen36TensorWorkError::Io {
        operation,
        kind: error.kind(),
    }
}

#[derive(Debug)]
struct WorkspaceCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WorkspaceCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Qwen36TensorWorkError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(Qwen36TensorWorkError::LengthOverflow("workspace cursor"))?;
        let value =
            self.bytes
                .get(self.offset..end)
                .ok_or(Qwen36TensorWorkError::WorkspaceMalformed(
                    "truncated workspace",
                ))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, Qwen36TensorWorkError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Qwen36TensorWorkError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, Qwen36TensorWorkError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], Qwen36TensorWorkError> {
        let mut digest = [0; 32];
        digest.copy_from_slice(self.take(32)?);
        Ok(digest)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap, io::Write};

    use super::*;
    use crate::qwen36_source_admission::test_fixture_source_proof;

    #[test]
    fn plan_freezes_exact_language_mtp_partition_and_preserved_bytes() {
        let proof = test_fixture_source_proof();
        let plan = WorkspacePlan::from_proof(&proof).unwrap();

        assert_eq!(plan.summary.active_tensors(), 866);
        assert_eq!(plan.summary.additive_required(), 506);
        assert_eq!(plan.summary.additive_present(), 0);
        assert_eq!(plan.summary.preserved_tensors(), 360);
        assert_eq!(plan.summary.deferred_vision_tensors(), 333);
        assert_eq!(plan.summary.active_coefficients(), 27_320_697_856);
        assert_eq!(plan.summary.preserved_coefficients(), 2_671_616);
        assert_eq!(plan.summary.preserved_payload_bytes(), 5_343_232);
        let preserved = plan
            .preserved
            .iter()
            .map(|entry| &entry.spec)
            .collect::<Vec<_>>();
        assert_eq!(preserved.len(), 360);
        assert!(
            preserved
                .windows(2)
                .all(|pair| pair[0].name() < pair[1].name())
        );
        assert_eq!(
            preserved
                .iter()
                .map(|spec| spec.payload_bytes())
                .sum::<u64>(),
            5_343_232
        );
        assert!(!plan.summary.complete());
        assert_eq!(plan.additive.len(), 506);
        assert!(plan.additive.iter().all(|slot| {
            slot.state() == Qwen36AdditiveSlotState::MissingCanonicalMaster
                && slot.dtype() == Qwen35SourceDtype::Bfloat16
                && slot.scope() != Qwen35TensorScope::DeferredVision
                && !slot.name().is_empty()
                && !slot.shape().is_empty()
                && slot.source_tensor_digest() != &[0; 32]
        }));
        assert_eq!(
            plan.additive
                .iter()
                .filter(|slot| slot.scope() == Qwen35TensorScope::Language)
                .count(),
            498
        );
        assert_eq!(
            plan.additive
                .iter()
                .filter(|slot| slot.scope() == Qwen35TensorScope::MtpDrafter)
                .count(),
            8
        );
        assert_eq!(
            plan.preserved
                .iter()
                .filter(|entry| entry.spec.schema_metadata()[106] == 1)
                .count(),
            353
        );
        assert_eq!(
            plan.preserved
                .iter()
                .filter(|entry| entry.spec.schema_metadata()[106] == 2)
                .count(),
            7
        );
    }

    #[test]
    fn workspace_manifest_round_trips_all_preserved_receipts_canonically() {
        let proof = test_fixture_source_proof();
        let plan = WorkspacePlan::from_proof(&proof).unwrap();
        let root = std::env::temp_dir().join(format!(
            "tritium-qwen36-workspace-manifest-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let store = TensorWorkStore::open(&root).unwrap();
        let zeros = [0_u8; STREAM_CHUNK_BYTES];
        let receipts = plan
            .preserved
            .iter()
            .map(|entry| {
                store
                    .put(&entry.spec, |writer| -> io::Result<()> {
                        let mut remaining = entry.spec.payload_bytes();
                        while remaining != 0 {
                            let count = usize::try_from(remaining.min(zeros.len() as u64)).unwrap();
                            writer.write_all(&zeros[..count])?;
                            remaining -= count as u64;
                        }
                        Ok(())
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let manifest = WorkspaceManifest::from_plan(&plan, receipts).unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        let decoded = WorkspaceManifest::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(decoded, manifest);
        decoded.validate(&plan).unwrap();
        let receipt = decoded.receipt(&bytes).unwrap();
        assert_eq!(receipt.workspace_id(), ContentId::of_bytes(&bytes));
        assert_eq!(receipt.summary(), plan.summary);
        let mut corrupt = bytes;
        corrupt[20] ^= 1;
        assert!(matches!(
            WorkspaceManifest::from_canonical_bytes(&corrupt),
            Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "workspace checksum"
            ))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[derive(Debug)]
    struct FixtureTensor {
        name: String,
        shape: Vec<u64>,
        payload: Vec<u8>,
    }

    #[derive(Debug)]
    struct FixtureSource {
        tensors: Vec<FixtureTensor>,
        visits: RefCell<BTreeMap<String, u64>>,
        fail_once: RefCell<Option<String>>,
    }

    impl FixtureSource {
        fn visits(&self, name: &str) -> u64 {
            self.visits.borrow().get(name).copied().unwrap_or(0)
        }
    }

    impl PreservedTensorSource for FixtureSource {
        fn try_visit_tensor_bytes(
            &self,
            name: &str,
            max_chunk_bytes: usize,
            visit: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        ) -> Result<u64, Qwen35TensorStreamError<io::Error>> {
            let tensor = self
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| {
                    Qwen35TensorStreamError::Source(NnError::MissingTensor(name.to_owned()))
                })?;
            *self.visits.borrow_mut().entry(name.to_owned()).or_insert(0) += 1;
            if self.fail_once.borrow().as_deref() == Some(name) {
                self.fail_once.borrow_mut().take();
                return Err(Qwen35TensorStreamError::Source(NnError::Backend(
                    "injected source interruption".into(),
                )));
            }
            if max_chunk_bytes == 0 {
                return Err(Qwen35TensorStreamError::Source(NnError::Backend(
                    "zero fixture chunk size".into(),
                )));
            }
            for chunk in tensor.payload.chunks(max_chunk_bytes) {
                visit(chunk).map_err(Qwen35TensorStreamError::Sink)?;
            }
            u64::try_from(tensor.payload.len()).map_err(|_| {
                Qwen35TensorStreamError::Source(NnError::Backend(
                    "fixture payload exceeds u64".into(),
                ))
            })
        }

        fn source_tensor_semantic_hasher(
            &self,
            name: &str,
        ) -> Result<SemanticTensorHasher, NnError> {
            let tensor = self
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
            Ok(SemanticTensorHasher::new(
                tensor.name.clone(),
                tensor.shape.clone(),
            ))
        }
    }

    fn fixture_source_and_plan() -> (FixtureSource, WorkspacePlan) {
        let tensors = vec![
            FixtureTensor {
                name: "model.a.weight".into(),
                shape: vec![2, 2],
                payload: b"abcdefgh".to_vec(),
            },
            FixtureTensor {
                name: "model.b.weight".into(),
                shape: vec![2, 2],
                payload: b"ijklmnop".to_vec(),
            },
        ];
        let source_model_id = ModelId::from_digest([41; 32]);
        let preserved = tensors
            .iter()
            .map(|tensor| {
                let mut hasher =
                    SemanticTensorHasher::new(tensor.name.clone(), tensor.shape.clone());
                hasher.update(&tensor.payload);
                let semantic = hasher.finalize().unwrap();
                PreservedPlanEntry {
                    spec: TensorRecordSpec::new(
                        ContentId::of_bytes(b"fixture preserved schema"),
                        source_model_id,
                        *semantic.content_digest(),
                        tensor.name.clone(),
                        tensor.shape.clone(),
                        b"fixture preserved metadata".to_vec(),
                        tensor.payload.len() as u64,
                    )
                    .unwrap(),
                }
            })
            .collect();
        let plan = WorkspacePlan {
            proof_id: ContentId::of_bytes(b"fixture proof"),
            manifest_content_id: ContentId::of_bytes(b"fixture manifest"),
            source_model_id,
            coverage_policy_digest: [43; 32],
            identity_status: Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration,
            summary: Qwen36TensorWorkSummary {
                active_tensors: 2,
                additive_required: 0,
                additive_present: 0,
                preserved_tensors: 2,
                deferred_vision_tensors: 0,
                active_coefficients: 8,
                preserved_coefficients: 8,
                preserved_payload_bytes: 16,
            },
            additive: Vec::new(),
            preserved,
        };
        (
            FixtureSource {
                tensors,
                visits: RefCell::new(BTreeMap::new()),
                fail_once: RefCell::new(Some("model.b.weight".into())),
            },
            plan,
        )
    }

    fn fixture_workspace_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-qwen36-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn reconcile_resumes_without_restreaming_completed_preserved_tensors() {
        let root = fixture_workspace_root("resume");
        let _ = fs::remove_dir_all(&root);
        let (source, plan) = fixture_source_and_plan();
        let store = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), plan.clone())
            .expect("open fixture workspace");

        assert!(matches!(
            store.reconcile_preserved(),
            Err(Qwen36TensorWorkError::Source(NnError::Backend(_)))
        ));
        assert!(store.slot_path("model.a.weight").is_file());
        assert!(!store.workspace_path().exists());
        assert_eq!(source.visits("model.a.weight"), 1);
        assert_eq!(source.visits("model.b.weight"), 1);

        let first = store.reconcile_preserved().expect("resume workspace");
        assert_eq!(source.visits("model.a.weight"), 1);
        assert_eq!(source.visits("model.b.weight"), 2);
        drop(store);

        let reopened = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), plan)
            .expect("reopen fixture workspace");
        let second = reopened
            .reconcile_preserved()
            .expect("idempotently reconcile workspace");
        assert_eq!(second, first);
        assert_eq!(source.visits("model.a.weight"), 1);
        assert_eq!(source.visits("model.b.weight"), 2);
        assert_eq!(
            reopened.preserved_tensor_names().collect::<Vec<_>>(),
            vec!["model.a.weight", "model.b.weight"]
        );
        let mut payload = Vec::new();
        let payload_bytes = reopened
            .try_visit_preserved_tensor("model.a.weight", 3, |chunk| {
                payload.extend_from_slice(chunk);
                Ok::<(), core::convert::Infallible>(())
            })
            .expect("visit verified preserved tensor");
        assert_eq!(payload_bytes, 8);
        assert_eq!(payload, b"abcdefgh");
        let descriptors = reopened
            .preserved_tensor_descriptors()
            .expect("clone preserved descriptors");
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].name(), "model.a.weight");
        assert_eq!(descriptors[0].shape(), &[2, 2]);
        assert_eq!(descriptors[0].payload_bytes(), 8);
        let mut safetensors = Vec::new();
        let safetensors_receipt = reopened
            .try_write_preserved_safetensors(3, |chunk| {
                safetensors.extend_from_slice(chunk);
                Ok::<(), core::convert::Infallible>(())
            })
            .expect("write exact preserved safetensors");
        assert_eq!(safetensors_receipt.tensor_count(), 2);
        assert_eq!(safetensors_receipt.payload_bytes(), 16);
        assert_eq!(safetensors_receipt.total_bytes(), safetensors.len() as u64);
        assert_eq!(
            safetensors_receipt.package_id(),
            PackageId::from_package_bytes(&safetensors)
        );
        let header_len = usize::try_from(u64::from_le_bytes(
            safetensors[..8].try_into().expect("safetensors prefix"),
        ))
        .expect("bounded header length");
        assert_eq!(header_len % 8, 0);
        let header: serde_json::Value = serde_json::from_slice(&safetensors[8..8 + header_len])
            .expect("parse safetensors header");
        assert_eq!(header["model.a.weight"]["dtype"], "BF16");
        assert_eq!(header["model.a.weight"]["shape"], serde_json::json!([2, 2]));
        assert_eq!(
            header["model.a.weight"]["data_offsets"],
            serde_json::json!([0, 8])
        );
        assert_eq!(
            header["model.b.weight"]["data_offsets"],
            serde_json::json!([8, 16])
        );
        assert_eq!(&safetensors[8 + header_len..], b"abcdefghijklmnop");
        assert!(matches!(
            reopened
                .try_write_preserved_safetensors(0, |_| Ok::<(), core::convert::Infallible>(())),
            Err(Qwen36PreservedSafetensorsError::Workspace(
                Qwen36TensorWorkError::WorkspaceMalformed("preserved safetensors chunk bound")
            ))
        ));
        assert!(matches!(
            reopened.try_visit_preserved_tensor("model.missing.weight", 3, |_| Ok::<
                (),
                core::convert::Infallible,
            >(())),
            Err(Qwen36PreservedVisitError::Workspace(
                Qwen36TensorWorkError::UnknownPreservedTensor
            ))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resume_rejects_valid_record_with_wrong_source_semantics() {
        let root = fixture_workspace_root("semantic-tamper");
        let _ = fs::remove_dir_all(&root);
        let (source, plan) = fixture_source_and_plan();
        source.fail_once.borrow_mut().take();
        let store = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), plan)
            .expect("open fixture workspace");
        let expected = &store.plan.preserved[0];
        let receipt = store
            .objects
            .put(&expected.spec, |writer| -> io::Result<()> {
                writer.write_all(b"wrong!!!")
            })
            .expect("publish internally valid wrong record");
        let slot_bytes = receipt.canonical_bytes().unwrap();
        persist_exact(
            &store.slot_path(expected.spec.name()),
            &slot_bytes,
            "preserved slot",
        )
        .unwrap();

        assert!(matches!(
            store.reconcile_preserved(),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "preserved source semantic digest"
            ))
        ));
        assert_eq!(source.visits("model.a.weight"), 0);
        assert_eq!(source.visits("model.b.weight"), 0);
        assert!(!store.workspace_path().exists());
        let _ = fs::remove_dir_all(root);
    }
}
