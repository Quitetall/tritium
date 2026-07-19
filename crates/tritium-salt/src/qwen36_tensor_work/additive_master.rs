//! Immutable additive-master campaigns layered over one preserved-source workspace.

mod selected_allocation;

pub use selected_allocation::{
    Qwen36AllocatedCampaignStore, Qwen36PackageAdmissionError, Qwen36PackageAdmissionReceipt,
    Qwen36PackageAdmittedCampaignStore, Qwen36PackageProfileReceipt, Qwen36PackageRuntimeLedger,
    Qwen36PackageScaleOnlyCampaignStore, Qwen36SelectedAllocationBindError,
    Qwen36SelectedAllocationReceipt, Qwen36SelectedAllocationSpec, Qwen36SelectedProfileReceipt,
};

use core::fmt;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::{
    cell::Cell,
    error::Error,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use tritium_format::{
    ModelId,
    salt_v2_master::{
        SALT_V2_MASTER_TENSOR_SCHEMA, SaltV2FitConstraint, SaltV2MasterTensorDecoder,
        SaltV2MasterTensorReceipt, SaltV2MasterTensorSpec, SaltV2MasterTile, SaltV2MasterTrack,
        SaltV2MasterVisitError,
    },
    salt_v2_package::SALT_V2_ALLOCATION_TILE_SIZE,
};

#[cfg(unix)]
use crate::tensor_work_store::ensure_durable_directory;
use crate::{
    ContentId, Qwen36SourceIdentityStatus, TensorPayloadValidator, TensorPayloadWriter,
    TensorRecordReceipt, TensorRecordSpec, TensorValidatedPutError, TensorVisitError,
    TensorWorkStore,
};

#[cfg(unix)]
use super::sync_directory;
use super::{
    CHECKSUM_BYTES, MAX_ACTIVE_TENSORS, MAX_WORKSPACE_BYTES, Qwen36LanguageMtpWorkspaceReceipt,
    Qwen36TensorWorkError, Qwen36TensorWorkStore, Qwen36TensorWorkSummary,
    identity_status_from_tag, identity_status_tag, persist_exact, read_regular_bounded,
    same_file_identity, summary_values, work_io,
};

#[cfg(unix)]
const CAMPAIGN_DIRECTORY: &str = "master-campaigns";
const CAMPAIGN_FILE: &str = "campaign.tq36p";
#[cfg(unix)]
const ADDITIVE_SLOT_DIRECTORY: &str = "additive-slots";
const COMPLETION_FILE: &str = "workspace.complete.tq36c";
const SLOT_EXTENSION: &str = "tq36mref";
#[cfg(unix)]
const CAMPAIGN_MAGIC: [u8; 8] = *b"TSQ36CP\0";
const CATALOG_MAGIC: [u8; 8] = *b"TSQ36SC\0";
#[cfg(unix)]
const SCALE_ONLY_CATALOG_MAGIC: [u8; 8] = *b"TSQ36SL\0";
#[cfg(unix)]
const PACKAGE_SCALE_ONLY_CATALOG_MAGIC: [u8; 8] = *b"TSQ36S2\0";
const SLOT_MAGIC: [u8; 8] = *b"TSQ36AR\0";
const COMPLETION_MAGIC: [u8; 8] = *b"TSQ36CM\0";
const FORMAT_VERSION: u16 = 1;
#[cfg(unix)]
const CAMPAIGN_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 additive campaign checksum v1";
const SLOT_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 additive master receipt checksum v1";
const COMPLETION_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 complete workspace checksum v1";
const MASTER_SET_CONTEXT: &str = "tritium qwen3.6 ordered additive master set v1";
const SLOT_KEY_CONTEXT: &str = "tritium qwen3.6 additive campaign slot key v1";
const FIXED_MASTER_CONTEXT: &str = "tritium qwen3.6 scale-only fixed master v1";
const MASTER_STREAM_CHUNK_BYTES: usize = 64 * 1024;
const MAX_SLOT_RECEIPT_BYTES: u64 = 512 * 1024;

#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
struct PackageScaleOnlyBinding {
    admission_id: ContentId,
    selection_id: ContentId,
    compact_package_id: [u8; 32],
    near_package_id: [u8; 32],
}

/// Exact ordered tensor-master metadata that defines one rate-free Qwen additive campaign.
///
/// Every tensor-specific curvature, feedback, widened-source, and parent digest
/// is committed before any payload can be installed. Deployment rate and codec
/// are absent because canonical tensor masters are reusable work artifacts. The
/// public constructor admits PTQ only; refined instances come from a typed,
/// parent-bound campaign opener.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36AdditiveCampaignSpec {
    expected_masters: Vec<SaltV2MasterTensorSpec>,
    catalog_bytes: Vec<u8>,
    spec_id: ContentId,
}

impl Qwen36AdditiveCampaignSpec {
    /// Construct one ordered additive campaign specification.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for empty, unordered, contradictory,
    /// zero-identity, refined-without-parent-verification, or noncanonical master
    /// metadata.
    pub fn new(
        mut expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<Self, Qwen36TensorWorkError> {
        if expected_masters.is_empty() || expected_masters.len() > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive campaign tensor count",
            ));
        }
        validate_expected_masters(&expected_masters)?;
        let catalog_bytes = encode_master_catalog(&expected_masters)?;
        let spec_id = ContentId::of_bytes(&catalog_bytes);
        expected_masters.shrink_to_fit();
        Ok(Self {
            expected_masters,
            catalog_bytes,
            spec_id,
        })
    }

    /// Content identity of the exact ordered tensor-master metadata catalog.
    #[must_use]
    pub const fn spec_id(&self) -> ContentId {
        self.spec_id
    }

    /// Expected tensor masters in canonical additive-slot order.
    #[must_use]
    pub fn expected_masters(&self) -> &[SaltV2MasterTensorSpec] {
        &self.expected_masters
    }

    #[cfg(unix)]
    fn new_scale_only(
        parent_completion: &Qwen36CompleteWorkspaceReceipt,
        parent_specs: &[SaltV2MasterTensorSpec],
        parent_masters: &[Qwen36AdditiveMasterReceipt],
        parent_fixed_ids: &[[u8; 32]],
        mut expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<Self, Qwen36TensorWorkError> {
        validate_scale_only_masters(parent_specs, parent_masters, &expected_masters)?;
        if parent_fixed_ids.len() != parent_masters.len() {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only fixed master count",
            ));
        }
        let catalog_bytes = encode_scale_only_master_catalog(
            parent_completion,
            parent_masters,
            parent_fixed_ids,
            &expected_masters,
        )?;
        let spec_id = ContentId::of_bytes(&catalog_bytes);
        expected_masters.shrink_to_fit();
        Ok(Self {
            expected_masters,
            catalog_bytes,
            spec_id,
        })
    }

    #[cfg(unix)]
    fn new_package_admitted_scale_only(
        parent_completion: &Qwen36CompleteWorkspaceReceipt,
        parent_specs: &[SaltV2MasterTensorSpec],
        parent_masters: &[Qwen36AdditiveMasterReceipt],
        parent_fixed_ids: &[[u8; 32]],
        binding: PackageScaleOnlyBinding,
        mut expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<Self, Qwen36TensorWorkError> {
        validate_scale_only_masters(parent_specs, parent_masters, &expected_masters)?;
        if parent_fixed_ids.len() != parent_masters.len() {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only fixed master count",
            ));
        }
        let catalog_bytes = encode_package_scale_only_master_catalog(
            parent_completion,
            parent_masters,
            parent_fixed_ids,
            binding,
            &expected_masters,
        )?;
        let spec_id = ContentId::of_bytes(&catalog_bytes);
        expected_masters.shrink_to_fit();
        Ok(Self {
            expected_masters,
            catalog_bytes,
            spec_id,
        })
    }
}

/// Failure while installing one typed producer's canonical master payload.
#[derive(Debug)]
pub enum Qwen36AdditiveInstallError<E> {
    /// Campaign, filesystem, record, or canonical-master validation failed.
    Campaign(Qwen36TensorWorkError),
    /// The caller's payload producer failed before publication.
    Producer(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36AdditiveInstallError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Campaign(error) => write!(formatter, "additive campaign install failed: {error}"),
            Self::Producer(error) => write!(formatter, "additive master producer failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36AdditiveInstallError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Campaign(error) => Some(error),
            Self::Producer(error) => Some(error),
        }
    }
}

/// Immutable Qwen slot reference to one verified canonical tensor master.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36AdditiveMasterReceipt {
    receipt_id: ContentId,
    campaign_id: ContentId,
    ordinal: u64,
    tensor_master_id: [u8; 32],
    record: TensorRecordReceipt,
}

impl Qwen36AdditiveMasterReceipt {
    fn new(
        campaign_id: ContentId,
        ordinal: u64,
        tensor_master_id: [u8; 32],
        record: TensorRecordReceipt,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let mut receipt = Self {
            receipt_id: ContentId::from_digest([0; 32]),
            campaign_id,
            ordinal,
            tensor_master_id,
            record,
        };
        let bytes = receipt.canonical_bytes()?;
        receipt.receipt_id = ContentId::of_bytes(&bytes);
        Ok(receipt)
    }

    /// Content identity of the complete canonical Qwen slot receipt.
    #[must_use]
    pub const fn receipt_id(&self) -> ContentId {
        self.receipt_id
    }

    /// Campaign whose exact master metadata admitted this payload.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Canonical lexical additive-slot ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Portable identity of canonical tensor-master metadata and payload bytes.
    #[must_use]
    pub const fn tensor_master_id(&self) -> [u8; 32] {
        self.tensor_master_id
    }

    /// Generic immutable record receipt carrying the canonical master stream.
    #[must_use]
    pub const fn record_receipt(&self) -> &TensorRecordReceipt {
        &self.record
    }

    /// Encode this receipt for durable resumable publication.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] if the embedded record receipt or a
    /// bounded canonical length is invalid.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        let record = self
            .record
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let record_length = u32::try_from(record.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive master receipt"))?;
        let mut output = Vec::new();
        output.extend_from_slice(&SLOT_MAGIC);
        output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(self.campaign_id.as_bytes());
        output.extend_from_slice(&self.ordinal.to_le_bytes());
        output.extend_from_slice(&self.tensor_master_id);
        output.extend_from_slice(&record_length.to_le_bytes());
        output.extend_from_slice(&record);
        let mut hasher = blake3::Hasher::new_derive_key(SLOT_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() as u64 > MAX_SLOT_RECEIPT_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master receipt too large",
            ));
        }
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if bytes.len() as u64 > MAX_SLOT_RECEIPT_BYTES
            || bytes.len() < SLOT_MAGIC.len() + 2 + 2 + 32 + 8 + 32 + 4 + CHECKSUM_BYTES
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master receipt",
            ));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(SLOT_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master receipt checksum",
            ));
        }
        let mut cursor = CanonicalCursor::new(payload);
        if cursor.take(SLOT_MAGIC.len())? != SLOT_MAGIC
            || cursor.u16()? != FORMAT_VERSION
            || cursor.u16()? != 0
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master receipt header",
            ));
        }
        let campaign_id = ContentId::from_digest(cursor.digest()?);
        let ordinal = cursor.u64()?;
        let tensor_master_id = cursor.digest()?;
        let record_length = usize::try_from(cursor.u32()?)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive master record receipt"))?;
        if record_length == 0 || record_length as u64 > MAX_SLOT_RECEIPT_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master record receipt length",
            ));
        }
        let record = TensorRecordReceipt::from_canonical_bytes(cursor.take(record_length)?)
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        if cursor.remaining() != 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "additive master receipt trailing bytes",
            ));
        }
        let receipt = Self::new(campaign_id, ordinal, tensor_master_id, record)?;
        if receipt.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical additive master receipt",
            ));
        }
        Ok(receipt)
    }
}

/// Structural receipt for one complete language-plus-MTP master campaign.
///
/// This receipt proves byte-complete local work only. It preserves the base
/// source identity status and is not an official-source, quality, runtime, or
/// SOTA publication receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36CompleteWorkspaceReceipt {
    completion_id: ContentId,
    base_workspace_id: ContentId,
    campaign_id: ContentId,
    master_set_id: [u8; 32],
    source_model_id: ModelId,
    identity_status: Qwen36SourceIdentityStatus,
    additive_coefficients: u64,
    completion_bytes: u64,
    summary: Qwen36TensorWorkSummary,
}

impl Qwen36CompleteWorkspaceReceipt {
    /// Content identity of exact canonical completion-seal bytes.
    #[must_use]
    pub const fn completion_id(&self) -> ContentId {
        self.completion_id
    }

    /// Immutable preserved-source workspace over which the campaign was layered.
    #[must_use]
    pub const fn base_workspace_id(&self) -> ContentId {
        self.base_workspace_id
    }

    /// Exact base-bound additive campaign identity.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Qwen-specific aggregate over ordered portable tensor-master identities.
    #[must_use]
    pub const fn master_set_id(&self) -> [u8; 32] {
        self.master_set_id
    }

    /// Semantic source-model identity inherited from the base workspace.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Candidate-only or officially authenticated source status inherited verbatim.
    #[must_use]
    pub const fn identity_status(&self) -> Qwen36SourceIdentityStatus {
        self.identity_status
    }

    /// Exact logical coefficients represented by additive masters.
    #[must_use]
    pub const fn additive_coefficients(&self) -> u64 {
        self.additive_coefficients
    }

    /// Exact canonical completion-seal bytes.
    #[must_use]
    pub const fn completion_bytes(&self) -> u64 {
        self.completion_bytes
    }

    /// Re-derived complete language-plus-MTP structural totals.
    #[must_use]
    pub const fn summary(&self) -> Qwen36TensorWorkSummary {
        self.summary
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompleteManifest {
    base_workspace_id: ContentId,
    proof_id: ContentId,
    manifest_content_id: ContentId,
    source_model_id: ModelId,
    coverage_policy_digest: [u8; 32],
    identity_status: Qwen36SourceIdentityStatus,
    campaign_id: ContentId,
    master_set_id: [u8; 32],
    additive_coefficients: u64,
    summary: Qwen36TensorWorkSummary,
    masters: Vec<Qwen36AdditiveMasterReceipt>,
}

impl CompleteManifest {
    fn from_masters(
        base: &Qwen36LanguageMtpWorkspaceReceipt,
        campaign_id: ContentId,
        additive_coefficients: u64,
        masters: Vec<Qwen36AdditiveMasterReceipt>,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let present = u64::try_from(masters.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master count"))?;
        let summary = base.summary().with_additive_present(present)?;
        let master_set_id = derive_master_set_id(&masters)?;
        Ok(Self {
            base_workspace_id: base.workspace_id(),
            proof_id: base.proof_id(),
            manifest_content_id: base.manifest_content_id(),
            source_model_id: base.source_model_id(),
            coverage_policy_digest: *base.coverage_policy_digest(),
            identity_status: base.identity_status(),
            campaign_id,
            master_set_id,
            additive_coefficients,
            summary,
            masters,
        })
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        if self.masters.len() > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "complete master count",
            ));
        }
        let mut output = Vec::new();
        output.extend_from_slice(&COMPLETION_MAGIC);
        output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(self.base_workspace_id.as_bytes());
        output.extend_from_slice(self.proof_id.as_bytes());
        output.extend_from_slice(self.manifest_content_id.as_bytes());
        output.extend_from_slice(self.source_model_id.as_bytes());
        output.extend_from_slice(&self.coverage_policy_digest);
        output.push(identity_status_tag(self.identity_status));
        output.extend_from_slice(self.campaign_id.as_bytes());
        output.extend_from_slice(&self.master_set_id);
        output.extend_from_slice(&self.additive_coefficients.to_le_bytes());
        for value in summary_values(self.summary) {
            output.extend_from_slice(&value.to_le_bytes());
        }
        let count = u32::try_from(self.masters.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master count"))?;
        output.extend_from_slice(&count.to_le_bytes());
        for master in &self.masters {
            let bytes = master.canonical_bytes()?;
            let length = u32::try_from(bytes.len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master receipt"))?;
            output.extend_from_slice(&length.to_le_bytes());
            output.extend_from_slice(&bytes);
        }
        let mut hasher = blake3::Hasher::new_derive_key(COMPLETION_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() > MAX_WORKSPACE_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "completion seal too large",
            ));
        }
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if bytes.len() > MAX_WORKSPACE_BYTES
            || bytes.len()
                < COMPLETION_MAGIC.len() + 2 + 2 + 32 * 7 + 1 + 8 + 8 * 8 + 4 + CHECKSUM_BYTES
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed("completion seal"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(COMPLETION_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "completion seal checksum",
            ));
        }
        let mut cursor = CanonicalCursor::new(payload);
        if cursor.take(COMPLETION_MAGIC.len())? != COMPLETION_MAGIC
            || cursor.u16()? != FORMAT_VERSION
            || cursor.u16()? != 0
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "completion seal header",
            ));
        }
        let base_workspace_id = ContentId::from_digest(cursor.digest()?);
        let proof_id = ContentId::from_digest(cursor.digest()?);
        let manifest_content_id = ContentId::from_digest(cursor.digest()?);
        let source_model_id = ModelId::from_digest(cursor.digest()?);
        let coverage_policy_digest = cursor.digest()?;
        let identity_status = identity_status_from_tag(cursor.u8()?)?;
        let campaign_id = ContentId::from_digest(cursor.digest()?);
        let master_set_id = cursor.digest()?;
        let additive_coefficients = cursor.u64()?;
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
        let count = usize::try_from(cursor.u32()?)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master count"))?;
        if count > MAX_ACTIVE_TENSORS {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "complete master count",
            ));
        }
        let mut masters = Vec::new();
        masters
            .try_reserve_exact(count)
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for _ in 0..count {
            let length = usize::try_from(cursor.u32()?)
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master receipt"))?;
            if length == 0 || length as u64 > MAX_SLOT_RECEIPT_BYTES {
                return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                    "complete master receipt length",
                ));
            }
            masters.push(Qwen36AdditiveMasterReceipt::from_canonical_bytes(
                cursor.take(length)?,
            )?);
        }
        if cursor.remaining() != 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "completion seal trailing bytes",
            ));
        }
        let manifest = Self {
            base_workspace_id,
            proof_id,
            manifest_content_id,
            source_model_id,
            coverage_policy_digest,
            identity_status,
            campaign_id,
            master_set_id,
            additive_coefficients,
            summary,
            masters,
        };
        if manifest.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical completion seal",
            ));
        }
        Ok(manifest)
    }

    fn receipt(
        &self,
        bytes: &[u8],
    ) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        Ok(Qwen36CompleteWorkspaceReceipt {
            completion_id: ContentId::of_bytes(bytes),
            base_workspace_id: self.base_workspace_id,
            campaign_id: self.campaign_id,
            master_set_id: self.master_set_id,
            source_model_id: self.source_model_id,
            identity_status: self.identity_status,
            additive_coefficients: self.additive_coefficients,
            completion_bytes: u64::try_from(bytes.len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("completion seal"))?,
            summary: self.summary,
        })
    }
}

/// Exclusive, resumable additive-master campaign layered over one base workspace.
///
/// The store keeps the base workspace immutable and derives progress only from
/// strictly reopened slot receipts and canonical master payloads.
#[derive(Debug)]
pub struct Qwen36AdditiveCampaignStore<'store, 'source> {
    base: &'store Qwen36TensorWorkStore<'source>,
    root: PathBuf,
    slots: PathBuf,
    objects: TensorWorkStore,
    spec: Qwen36AdditiveCampaignSpec,
    base_workspace: Qwen36LanguageMtpWorkspaceReceipt,
    descriptor_bytes: Vec<u8>,
    campaign_id: ContentId,
    additive_coefficients: u64,
    directories: Vec<PinnedDirectory>,
    mutation_active: Cell<bool>,
    _lock: CampaignLock,
}

/// Parent-bound, fixed-trit scale-only campaign over one sealed PTQ campaign.
///
/// This rate-free work layer proves parent identity, tensor geometry, admissible
/// prefixes, and hard trits. Deployment allocation-map binding remains a
/// separate package-stage requirement.
#[derive(Debug)]
pub struct Qwen36ScaleOnlyCampaignStore<'parent, 'store, 'source> {
    parent: &'parent Qwen36AdditiveCampaignStore<'store, 'source>,
    campaign: Qwen36AdditiveCampaignStore<'store, 'source>,
    parent_completion: Qwen36CompleteWorkspaceReceipt,
    parent_masters: Vec<Qwen36AdditiveMasterReceipt>,
    parent_fixed_ids: Vec<[u8; 32]>,
}

impl<'source> Qwen36TensorWorkStore<'source> {
    /// Open or resume an exclusive additive campaign over this exact base workspace.
    ///
    /// The campaign identity binds the byte-exact base workspace and every
    /// expected tensor-master specification in canonical Qwen additive order.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for a missing or changed base, a spec
    /// mismatch, unsafe filesystem state, immutable descriptor conflict, or a
    /// campaign currently owned by another process.
    #[cfg(unix)]
    pub fn open_master_campaign<'store>(
        &'store self,
        spec: Qwen36AdditiveCampaignSpec,
    ) -> Result<Qwen36AdditiveCampaignStore<'store, 'source>, Qwen36TensorWorkError> {
        validate_expected_masters(&spec.expected_masters)?;
        self.open_additive_campaign(spec)
    }

    #[cfg(unix)]
    fn open_additive_campaign<'store>(
        &'store self,
        spec: Qwen36AdditiveCampaignSpec,
    ) -> Result<Qwen36AdditiveCampaignStore<'store, 'source>, Qwen36TensorWorkError> {
        let base_workspace = self.reopen_workspace()?;
        let additive_coefficients = validate_spec_against_base(&spec, self, &base_workspace)?;
        let descriptor_bytes =
            campaign_descriptor_bytes(&base_workspace, &spec, additive_coefficients)?;
        let campaign_id = ContentId::of_bytes(&descriptor_bytes);
        let campaigns = self.root.join(CAMPAIGN_DIRECTORY);
        ensure_durable_directory(&campaigns, "additive campaign directory")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let root = campaigns.join(campaign_id.to_string());
        ensure_durable_directory(&root, "additive campaign root")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let slots = root.join(ADDITIVE_SLOT_DIRECTORY);
        ensure_durable_directory(&slots, "additive campaign slot directory")
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let objects = TensorWorkStore::open(&root).map_err(Qwen36TensorWorkError::TensorStore)?;
        let descriptor_path = root.join(CAMPAIGN_FILE);
        persist_exact(
            &descriptor_path,
            &descriptor_bytes,
            "additive campaign descriptor",
        )?;
        let lock = acquire_campaign_lock(&descriptor_path)?;
        let directories = pin_campaign_directories(&root, &slots, &objects)?;
        validate_directories(&directories)?;
        objects
            .scavenge_temporary()
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        validate_directories(&directories)?;
        let campaign = Qwen36AdditiveCampaignStore {
            base: self,
            root,
            slots,
            objects,
            spec,
            base_workspace,
            descriptor_bytes,
            campaign_id,
            additive_coefficients,
            directories,
            mutation_active: Cell::new(false),
            _lock: lock,
        };
        campaign.ensure_current()?;
        {
            let _mutation = campaign.begin_mutation()?;
            campaign.reclaim_unreferenced_objects()?;
        }
        campaign.ensure_current()?;
        Ok(campaign)
    }

    /// Reject additive campaign mutation where stable file identity is unavailable.
    ///
    /// # Errors
    /// Always returns [`Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform`].
    #[cfg(not(unix))]
    pub fn open_master_campaign<'store>(
        &'store self,
        _spec: Qwen36AdditiveCampaignSpec,
    ) -> Result<Qwen36AdditiveCampaignStore<'store, 'source>, Qwen36TensorWorkError> {
        Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform)
    }
}

impl<'store, 'source> Qwen36AdditiveCampaignStore<'store, 'source> {
    /// Open or resume a scale-only child whose parent is this sealed PTQ campaign.
    ///
    /// Child metadata must bind every corresponding parent tensor master and
    /// preserve source identity and semantic geometry. Payload installation adds
    /// bounded fixed-trit and admissible-prefix verification.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] unless this parent strictly reopens as
    /// complete or any child lineage, metadata, filesystem, or lock check fails.
    #[cfg(unix)]
    pub fn open_scale_only_campaign<'parent>(
        &'parent self,
        expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<Qwen36ScaleOnlyCampaignStore<'parent, 'store, 'source>, Qwen36TensorWorkError> {
        let (parent_completion, parent_manifest, parent_fixed_ids) =
            self.require_complete_verified(FixedCampaignMode::Capture)?;
        let spec = Qwen36AdditiveCampaignSpec::new_scale_only(
            &parent_completion,
            &self.spec.expected_masters,
            &parent_manifest.masters,
            &parent_fixed_ids,
            expected_masters,
        )?;
        let campaign = self.base.open_additive_campaign(spec)?;
        let child = Qwen36ScaleOnlyCampaignStore {
            parent: self,
            campaign,
            parent_completion,
            parent_masters: parent_manifest.masters,
            parent_fixed_ids,
        };
        child.verify_parent_campaign()?;
        Ok(child)
    }

    /// Reject scale-only child mutation where stable file identity is unavailable.
    ///
    /// # Errors
    /// Always returns [`Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform`].
    #[cfg(not(unix))]
    pub fn open_scale_only_campaign<'parent>(
        &'parent self,
        _expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<Qwen36ScaleOnlyCampaignStore<'parent, 'store, 'source>, Qwen36TensorWorkError> {
        Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform)
    }

    /// Content identity of the base-bound campaign descriptor.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Exact ordered master specification admitted by this campaign.
    #[must_use]
    pub const fn spec(&self) -> &Qwen36AdditiveCampaignSpec {
        &self.spec
    }

    /// Immutable campaign root containing objects, slot receipts, and the final seal.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Exact logical coefficients covered by additive tensor masters.
    #[must_use]
    pub const fn additive_coefficients(&self) -> u64 {
        self.additive_coefficients
    }

    /// Install one expected canonical tensor-master payload exactly once.
    ///
    /// Existing valid slots are strictly reopened without invoking `produce`.
    /// A new generic CAS object is fully decoded as a SALT tensor master before
    /// its small campaign slot receipt is durably published.
    ///
    /// # Errors
    /// Returns [`Qwen36AdditiveInstallError::Campaign`] for a changed campaign,
    /// unexpected metadata, malformed payload, immutable conflict, or store
    /// failure, and [`Qwen36AdditiveInstallError::Producer`] for a typed producer
    /// failure.
    pub fn install_master<E>(
        &self,
        spec: &SaltV2MasterTensorSpec,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36AdditiveInstallError<E>> {
        self.install_master_with_fixed(spec, FixedMasterMode::Skip, produce, || Ok(()))
    }

    fn install_master_with_fixed<E>(
        &self,
        spec: &SaltV2MasterTensorSpec,
        fixed_mode: FixedMasterMode,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
        prepublish: impl FnOnce() -> Result<(), Qwen36TensorWorkError>,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36AdditiveInstallError<E>> {
        let _mutation = self
            .begin_mutation()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        self.ensure_current()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let ordinal = self
            .expected_ordinal(spec)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let slot_path = self.slot_path(ordinal, spec.name());
        match fs::symlink_metadata(&slot_path) {
            Ok(_) => {
                let receipt = self
                    .reopen_slot_with_fixed(ordinal, spec, fixed_mode)
                    .map_err(Qwen36AdditiveInstallError::Campaign)?;
                match fs::symlink_metadata(self.completion_path()) {
                    Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                        return Err(Qwen36AdditiveInstallError::Campaign(
                            Qwen36TensorWorkError::InvalidPath("complete additive workspace"),
                        ));
                    }
                    Ok(_) => {
                        self.require_complete()
                            .map_err(Qwen36AdditiveInstallError::Campaign)?;
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(Qwen36AdditiveInstallError::Campaign(work_io(
                            "inspect completion seal",
                            error,
                        )));
                    }
                }
                self.ensure_current()
                    .map_err(Qwen36AdditiveInstallError::Campaign)?;
                return Ok(receipt);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Qwen36AdditiveInstallError::Campaign(work_io(
                    "inspect additive master slot",
                    error,
                )));
            }
        }
        match fs::symlink_metadata(self.completion_path()) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(Qwen36AdditiveInstallError::Campaign(
                    Qwen36TensorWorkError::InvalidPath("complete additive workspace"),
                ));
            }
            Ok(_) => {
                return Err(Qwen36AdditiveInstallError::Campaign(
                    Qwen36TensorWorkError::ExistingArtifactMismatch("sealed additive campaign"),
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Qwen36AdditiveInstallError::Campaign(work_io(
                    "inspect completion seal",
                    error,
                )));
            }
        }
        let record_spec = master_record_spec(spec).map_err(Qwen36AdditiveInstallError::Campaign)?;
        let validator = CanonicalMasterValidator::new(spec, fixed_mode)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let (record, verified) =
            match self
                .objects
                .put_validated_checked(&record_spec, validator, produce, prepublish)
            {
                Ok(validated) => validated,
                Err(TensorValidatedPutError::Store(error)) => {
                    return Err(Qwen36AdditiveInstallError::Campaign(
                        Qwen36TensorWorkError::TensorStore(error),
                    ));
                }
                Err(TensorValidatedPutError::Producer(error)) => {
                    return Err(Qwen36AdditiveInstallError::Producer(error));
                }
                Err(TensorValidatedPutError::Validator(error)) => {
                    return Err(Qwen36AdditiveInstallError::Campaign(error));
                }
            };
        let receipt = Qwen36AdditiveMasterReceipt::new(
            self.campaign_id,
            ordinal as u64,
            verified.master.tensor_master_id(),
            record,
        )
        .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let bytes = receipt
            .canonical_bytes()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        self.ensure_current()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        persist_exact(&slot_path, &bytes, "additive master slot")
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        self.ensure_current()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        Ok(receipt)
    }

    /// Strictly reopen one installed master by canonical tensor name.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for an unknown or missing tensor, a
    /// changed campaign, or any receipt, record, or semantic-master mismatch.
    pub fn reopen_master(
        &self,
        name: &str,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        self.ensure_current()?;
        let ordinal = self
            .spec
            .expected_masters
            .binary_search_by(|master| master.name().cmp(name))
            .map_err(|_| Qwen36TensorWorkError::UnknownAdditiveTensor)?;
        let receipt = self.reopen_slot(ordinal, &self.spec.expected_masters[ordinal])?;
        self.ensure_current()?;
        Ok(receipt)
    }

    /// Recompute strict campaign progress from verified slot receipts and objects.
    ///
    /// This correctness-first path fully reopens each present generic record and
    /// streams its payload through the canonical SALT decoder.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for any corrupt or contradictory present
    /// slot, changed campaign/base bytes, or checked count overflow.
    pub fn progress(&self) -> Result<Qwen36TensorWorkSummary, Qwen36TensorWorkError> {
        self.progress_with_fixed(FixedCampaignMode::Skip)
    }

    fn progress_with_fixed(
        &self,
        fixed_mode: FixedCampaignMode<'_>,
    ) -> Result<Qwen36TensorWorkSummary, Qwen36TensorWorkError> {
        fixed_mode.validate_count(self.spec.expected_masters.len())?;
        self.ensure_current()?;
        let mut present = 0_u64;
        for (ordinal, expected) in self.spec.expected_masters.iter().enumerate() {
            match fs::symlink_metadata(self.slot_path(ordinal, expected.name())) {
                Ok(_) => {
                    self.reopen_slot_with_fixed(
                        ordinal,
                        expected,
                        fixed_mode.master_mode(ordinal)?,
                    )?;
                    present =
                        present
                            .checked_add(1)
                            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                "additive master progress",
                            ))?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(work_io("inspect additive master slot", error)),
            }
        }
        self.ensure_current()?;
        self.base_workspace.summary().with_additive_present(present)
    }

    /// Seal a structurally complete campaign in canonical additive-slot order.
    ///
    /// The completion file embeds every slot receipt and a Qwen-specific ordered
    /// master-set identity. It is published only after every referenced object
    /// passes strict generic-record and canonical SALT semantic verification.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError::MissingAdditiveArtifacts`] while any slot
    /// is absent, or another validation/publication error for changed bytes.
    pub fn seal_complete(&self) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.seal_complete_with_fixed(FixedCampaignMode::Skip, || Ok(()))
    }

    fn seal_complete_with_fixed(
        &self,
        fixed_mode: FixedCampaignMode<'_>,
        prepublish: impl FnOnce() -> Result<(), Qwen36TensorWorkError>,
    ) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        fixed_mode.validate_count(self.spec.expected_masters.len())?;
        let _mutation = self.begin_mutation()?;
        self.ensure_current()?;
        self.reclaim_unreferenced_objects()?;
        let mut masters = Vec::new();
        masters
            .try_reserve_exact(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        let mut present = 0_u64;
        for (ordinal, expected) in self.spec.expected_masters.iter().enumerate() {
            match fs::symlink_metadata(self.slot_path(ordinal, expected.name())) {
                Ok(_) => {
                    masters.push(self.reopen_slot_with_fixed(
                        ordinal,
                        expected,
                        fixed_mode.master_mode(ordinal)?,
                    )?);
                    present =
                        present
                            .checked_add(1)
                            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                                "complete master count",
                            ))?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(work_io("inspect additive master slot", error)),
            }
        }
        self.ensure_current()?;
        let expected = u64::try_from(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("complete master count"))?;
        if present != expected {
            return Err(Qwen36TensorWorkError::MissingAdditiveArtifacts { expected, present });
        }
        let manifest = CompleteManifest::from_masters(
            &self.base_workspace,
            self.campaign_id,
            self.additive_coefficients,
            masters,
        )?;
        let bytes = manifest.canonical_bytes()?;
        self.ensure_current()?;
        prepublish()?;
        persist_exact(
            &self.completion_path(),
            &bytes,
            "complete additive workspace",
        )?;
        self.ensure_current()?;
        manifest.receipt(&bytes)
    }

    /// Strictly reopen the completion seal and every referenced canonical master.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for an absent/incomplete campaign or any
    /// changed base, descriptor, seal, slot receipt, generic record, or SALT
    /// semantic payload.
    pub fn require_complete(
        &self,
    ) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.require_complete_verified(FixedCampaignMode::Skip)
            .map(|(receipt, _, _)| receipt)
    }

    fn require_complete_verified(
        &self,
        fixed_mode: FixedCampaignMode<'_>,
    ) -> Result<
        (
            Qwen36CompleteWorkspaceReceipt,
            CompleteManifest,
            Vec<[u8; 32]>,
        ),
        Qwen36TensorWorkError,
    > {
        fixed_mode.validate_count(self.spec.expected_masters.len())?;
        self.ensure_current()?;
        match fs::symlink_metadata(self.completion_path()) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let summary = self.progress_with_fixed(fixed_mode)?;
                return Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                    expected: summary.additive_required(),
                    present: summary.additive_present(),
                });
            }
            Err(error) => return Err(work_io("inspect completion seal", error)),
        }
        let bytes = read_regular_bounded(
            &self.completion_path(),
            MAX_WORKSPACE_BYTES as u64,
            "complete additive workspace",
        )?;
        let manifest = CompleteManifest::from_canonical_bytes(&bytes)?;
        let expected_summary = self
            .base_workspace
            .summary()
            .with_additive_present(self.base_workspace.summary().additive_required())?;
        if manifest.base_workspace_id != self.base_workspace.workspace_id()
            || manifest.proof_id != self.base_workspace.proof_id()
            || manifest.manifest_content_id != self.base_workspace.manifest_content_id()
            || manifest.source_model_id != self.base_workspace.source_model_id()
            || manifest.coverage_policy_digest != *self.base_workspace.coverage_policy_digest()
            || manifest.identity_status != self.base_workspace.identity_status()
            || manifest.campaign_id != self.campaign_id
            || manifest.additive_coefficients != self.additive_coefficients
            || manifest.summary != expected_summary
            || manifest.masters.len() != self.spec.expected_masters.len()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "completion seal binding",
            ));
        }
        let mut reopened = Vec::new();
        reopened
            .try_reserve_exact(manifest.masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        let mut fixed_ids = Vec::new();
        fixed_ids
            .try_reserve_exact(manifest.masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for (ordinal, (sealed, expected)) in manifest
            .masters
            .iter()
            .zip(&self.spec.expected_masters)
            .enumerate()
        {
            let (current, verified) =
                self.reopen_slot_verified(ordinal, expected, fixed_mode.master_mode(ordinal)?)?;
            if current != *sealed {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "completion master receipt",
                ));
            }
            reopened.push(current);
            if let Some(fixed_id) = verified.fixed_id {
                fixed_ids.push(fixed_id);
            }
        }
        if derive_master_set_id(&reopened)? != manifest.master_set_id {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "ordered master set identity",
            ));
        }
        self.ensure_current()?;
        let receipt = manifest.receipt(&bytes)?;
        Ok((receipt, manifest, fixed_ids))
    }

    fn descriptor_path(&self) -> PathBuf {
        self.root.join(CAMPAIGN_FILE)
    }

    fn begin_mutation(&self) -> Result<CampaignMutationGuard<'_>, Qwen36TensorWorkError> {
        if self.mutation_active.replace(true) {
            return Err(Qwen36TensorWorkError::CampaignLocked);
        }
        Ok(CampaignMutationGuard(&self.mutation_active))
    }

    fn completion_path(&self) -> PathBuf {
        self.root.join(COMPLETION_FILE)
    }

    fn verify_completion_receipt(
        &self,
        expected: &Qwen36CompleteWorkspaceReceipt,
    ) -> Result<(), Qwen36TensorWorkError> {
        self.ensure_current()?;
        let bytes = read_regular_bounded(
            &self.completion_path(),
            MAX_WORKSPACE_BYTES as u64,
            "complete additive workspace",
        )?;
        let manifest = CompleteManifest::from_canonical_bytes(&bytes)?;
        if manifest.receipt(&bytes)? != *expected {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only parent completion",
            ));
        }
        self.ensure_current()
    }

    fn expected_ordinal(
        &self,
        spec: &SaltV2MasterTensorSpec,
    ) -> Result<usize, Qwen36TensorWorkError> {
        let ordinal = usize::try_from(spec.tensor_index())
            .map_err(|_| Qwen36TensorWorkError::WorkspaceMismatch("additive master ordinal"))?;
        if self.spec.expected_masters.get(ordinal) != Some(spec) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master specification",
            ));
        }
        Ok(ordinal)
    }

    fn reopen_slot(
        &self,
        ordinal: usize,
        expected: &SaltV2MasterTensorSpec,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        self.reopen_slot_with_fixed(ordinal, expected, FixedMasterMode::Skip)
    }

    fn reopen_slot_with_fixed(
        &self,
        ordinal: usize,
        expected: &SaltV2MasterTensorSpec,
        fixed_mode: FixedMasterMode,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        Ok(self.reopen_slot_verified(ordinal, expected, fixed_mode)?.0)
    }

    fn reopen_slot_verified(
        &self,
        ordinal: usize,
        expected: &SaltV2MasterTensorSpec,
        fixed_mode: FixedMasterMode,
    ) -> Result<(Qwen36AdditiveMasterReceipt, VerifiedMaster), Qwen36TensorWorkError> {
        let receipt = self.reopen_slot_receipt(ordinal, expected)?;
        let verified = self.verify_record(&receipt.record, expected, fixed_mode)?;
        if verified.master.tensor_master_id() != receipt.tensor_master_id {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive tensor master identity",
            ));
        }
        Ok((receipt, verified))
    }

    fn reopen_slot_receipt(
        &self,
        ordinal: usize,
        expected: &SaltV2MasterTensorSpec,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        let path = self.slot_path(ordinal, expected.name());
        match fs::symlink_metadata(&path) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(Qwen36TensorWorkError::MissingAdditiveMaster {
                    name: expected.name().to_owned(),
                });
            }
            Err(error) => return Err(work_io("inspect additive master slot", error)),
        }
        let bytes = read_regular_bounded(&path, MAX_SLOT_RECEIPT_BYTES, "additive master slot")?;
        let receipt = Qwen36AdditiveMasterReceipt::from_canonical_bytes(&bytes)?;
        if receipt.campaign_id != self.campaign_id
            || receipt.ordinal != ordinal as u64
            || receipt.record.info().name() != expected.name()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master slot binding",
            ));
        }
        validate_record_descriptor(&receipt.record, expected)?;
        Ok(receipt)
    }

    #[cfg(unix)]
    fn reclaim_unreferenced_objects(&self) -> Result<(), Qwen36TensorWorkError> {
        match fs::symlink_metadata(self.completion_path()) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(Qwen36TensorWorkError::InvalidPath(
                    "complete additive workspace",
                ));
            }
            Ok(_) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(work_io("inspect completion seal", error)),
        }
        self.ensure_current()?;
        let mut retained = Vec::new();
        retained
            .try_reserve_exact(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        let mut known_slots = Vec::new();
        known_slots
            .try_reserve_exact(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        for (ordinal, expected) in self.spec.expected_masters.iter().enumerate() {
            let path = self.slot_path(ordinal, expected.name());
            match fs::symlink_metadata(&path) {
                Ok(_) => {
                    let receipt = self.reopen_slot_receipt(ordinal, expected)?;
                    retained.push(receipt.record.record_id());
                    known_slots.push(path);
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(work_io("inspect additive master slot", error)),
            }
        }
        let mut stale_slot_temporaries = Vec::new();
        let entries = fs::read_dir(&self.slots)
            .map_err(|error| work_io("read additive slot directory", error))?;
        for entry in entries {
            let entry = entry.map_err(|error| work_io("read additive slot entry", error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| work_io("inspect additive slot entry", error))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(Qwen36TensorWorkError::InvalidPath("additive slot entry"));
            }
            if known_slots.contains(&path) {
                continue;
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Qwen36TensorWorkError::InvalidPath("additive slot name"))?;
            if !recognized_slot_temporary(&name) {
                return Err(Qwen36TensorWorkError::ExistingArtifactMismatch(
                    "additive slot namespace",
                ));
            }
            stale_slot_temporaries
                .try_reserve(1)
                .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
            stale_slot_temporaries.push(path);
        }
        let orphan_sweep = self
            .objects
            .prepare_unreferenced_scavenge(&retained)
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        for temporary in stale_slot_temporaries {
            fs::remove_file(&temporary)
                .map_err(|error| work_io("remove stale additive slot temporary", error))?;
            sync_directory(&self.slots, "sync scavenged additive slot directory")?;
        }
        self.objects
            .commit_unreferenced_scavenge(orphan_sweep)
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        self.ensure_current()?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn reclaim_unreferenced_objects(&self) -> Result<(), Qwen36TensorWorkError> {
        Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform)
    }

    fn verify_record(
        &self,
        record: &TensorRecordReceipt,
        expected: &SaltV2MasterTensorSpec,
        fixed_mode: FixedMasterMode,
    ) -> Result<VerifiedMaster, Qwen36TensorWorkError> {
        validate_record_descriptor(record, expected)?;
        let mut validator = CanonicalMasterValidator::new(expected, fixed_mode)?;
        self.objects
            .try_visit_verified(record, MASTER_STREAM_CHUNK_BYTES, |chunk| {
                validator.try_push(chunk)
            })
            .map_err(map_validator_visit_error)?;
        let verified = validator.finish()?;
        if verified.master.payload_bytes() != expected.payload_bytes()
            || verified.master.tile_count() != expected.tile_count() as u64
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master payload geometry",
            ));
        }
        Ok(verified)
    }

    fn slot_path(&self, ordinal: usize, name: &str) -> PathBuf {
        let mut hasher = blake3::Hasher::new_derive_key(SLOT_KEY_CONTEXT);
        hasher.update(&(ordinal as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        self.slots.join(format!(
            "{ordinal:04}-{}.{}",
            hasher.finalize().to_hex(),
            SLOT_EXTENSION
        ))
    }

    fn ensure_current(&self) -> Result<(), Qwen36TensorWorkError> {
        validate_directories(&self.directories)?;
        self._lock.validate_path()?;
        if self.base.reopen_workspace()? != self.base_workspace {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive campaign base workspace",
            ));
        }
        let bytes = read_regular_bounded(
            &self.descriptor_path(),
            MAX_WORKSPACE_BYTES as u64,
            "additive campaign descriptor",
        )?;
        if bytes != self.descriptor_bytes {
            return Err(Qwen36TensorWorkError::ExistingArtifactMismatch(
                "additive campaign descriptor",
            ));
        }
        self._lock.validate_path()?;
        validate_directories(&self.directories)?;
        Ok(())
    }
}

impl Qwen36ScaleOnlyCampaignStore<'_, '_, '_> {
    /// Content identity of the parent-bound child campaign descriptor.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign.campaign_id()
    }

    /// Exact sealed PTQ completion admitted as this campaign's parent.
    #[must_use]
    pub const fn parent_completion_id(&self) -> ContentId {
        self.parent_completion.completion_id()
    }

    /// PTQ campaign whose verified masters define fixed child trits and prefixes.
    #[must_use]
    pub const fn parent_campaign_id(&self) -> ContentId {
        self.parent.campaign_id()
    }

    /// Exact ordered scale-only master specification admitted by this campaign.
    #[must_use]
    pub const fn spec(&self) -> &Qwen36AdditiveCampaignSpec {
        self.campaign.spec()
    }

    /// Immutable child campaign root.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.campaign.root()
    }

    /// Install one scale-only master after verifying its sealed parent and fixed structure.
    ///
    /// Loss curves and scales may change. Every admissible prefix and hard trit
    /// must match the corresponding verified parent before CAS publication.
    ///
    /// # Errors
    /// Returns [`Qwen36AdditiveInstallError::Campaign`] for changed parent/child
    /// lineage or fixed structure, and [`Qwen36AdditiveInstallError::Producer`]
    /// for a typed producer failure.
    pub fn install_master<E>(
        &self,
        spec: &SaltV2MasterTensorSpec,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36AdditiveInstallError<E>> {
        let ordinal = self
            .campaign
            .expected_ordinal(spec)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        match fs::symlink_metadata(self.campaign.completion_path()) {
            Ok(_) => {
                self.require_complete()
                    .map_err(Qwen36AdditiveInstallError::Campaign)?;
                return self
                    .reopen_master(spec.name())
                    .map_err(Qwen36AdditiveInstallError::Campaign);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Qwen36AdditiveInstallError::Campaign(work_io(
                    "inspect completion seal",
                    error,
                )));
            }
        }
        self.verify_parent_master(ordinal)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let fixed_id = self.parent_fixed_ids.get(ordinal).copied().ok_or(
            Qwen36AdditiveInstallError::Campaign(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only fixed master count",
            )),
        )?;
        let receipt = self.campaign.install_master_with_fixed(
            spec,
            FixedMasterMode::Require(fixed_id),
            produce,
            || self.verify_parent_master(ordinal),
        )?;
        self.verify_parent_master(ordinal)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        Ok(receipt)
    }

    /// Strictly reopen one child master and its corresponding sealed parent.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for unknown, missing, changed, or
    /// fixed-structure-incompatible parent or child artifacts.
    pub fn reopen_master(
        &self,
        name: &str,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        self.campaign.ensure_current()?;
        let ordinal = self
            .campaign
            .spec
            .expected_masters
            .binary_search_by(|master| master.name().cmp(name))
            .map_err(|_| Qwen36TensorWorkError::UnknownAdditiveTensor)?;
        self.verify_parent_master(ordinal)?;
        let fixed_id = self.parent_fixed_ids.get(ordinal).copied().ok_or(
            Qwen36TensorWorkError::WorkspaceMismatch("scale-only fixed master count"),
        )?;
        let receipt = self.campaign.reopen_slot_with_fixed(
            ordinal,
            &self.campaign.spec.expected_masters[ordinal],
            FixedMasterMode::Require(fixed_id),
        )?;
        self.campaign.ensure_current()?;
        self.verify_parent_master(ordinal)?;
        Ok(receipt)
    }

    /// Recompute child progress while strictly verifying the sealed parent.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for changed lineage or any corrupt,
    /// contradictory, or fixed-structure-incompatible artifact.
    pub fn progress(&self) -> Result<Qwen36TensorWorkSummary, Qwen36TensorWorkError> {
        self.verify_parent_campaign()?;
        let summary = self
            .campaign
            .progress_with_fixed(FixedCampaignMode::Require(&self.parent_fixed_ids))?;
        self.verify_parent_campaign()?;
        Ok(summary)
    }

    /// Seal a structurally complete fixed-trit child campaign.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] unless the parent and every child master
    /// strictly reopen with their originally bound fixed structures.
    pub fn seal_complete(&self) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.verify_parent_campaign()?;
        let receipt = self
            .campaign
            .seal_complete_with_fixed(FixedCampaignMode::Require(&self.parent_fixed_ids), || {
                self.verify_parent_campaign()
            })?;
        self.verify_parent_campaign()?;
        Ok(receipt)
    }

    /// Strictly reopen the sealed parent and completed fixed-trit child.
    ///
    /// # Errors
    /// Returns [`Qwen36TensorWorkError`] for incomplete or changed lineage,
    /// receipts, records, payloads, trits, or admissible prefixes.
    pub fn require_complete(
        &self,
    ) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.verify_parent_campaign()?;
        let receipt = self
            .campaign
            .require_complete_verified(FixedCampaignMode::Require(&self.parent_fixed_ids))?
            .0;
        self.verify_parent_campaign()?;
        Ok(receipt)
    }

    fn verify_parent_master(&self, ordinal: usize) -> Result<(), Qwen36TensorWorkError> {
        self.parent
            .verify_completion_receipt(&self.parent_completion)?;
        self.verify_parent_master_record(ordinal)?;
        self.parent
            .verify_completion_receipt(&self.parent_completion)
    }

    fn verify_parent_master_record(&self, ordinal: usize) -> Result<(), Qwen36TensorWorkError> {
        let expected = self.parent.spec.expected_masters.get(ordinal).ok_or(
            Qwen36TensorWorkError::WorkspaceMismatch("scale-only parent tensor count"),
        )?;
        let sealed =
            self.parent_masters
                .get(ordinal)
                .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                    "scale-only parent master receipt",
                ))?;
        let fixed_id = self.parent_fixed_ids.get(ordinal).copied().ok_or(
            Qwen36TensorWorkError::WorkspaceMismatch("scale-only fixed master count"),
        )?;
        let (current, _) = self.parent.reopen_slot_verified(
            ordinal,
            expected,
            FixedMasterMode::Require(fixed_id),
        )?;
        if current != *sealed {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only parent master receipt",
            ));
        }
        Ok(())
    }

    fn verify_parent_campaign(&self) -> Result<(), Qwen36TensorWorkError> {
        self.parent
            .verify_completion_receipt(&self.parent_completion)?;
        if self.parent_masters.len() != self.parent.spec.expected_masters.len()
            || self.parent_fixed_ids.len() != self.parent_masters.len()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only parent tensor count",
            ));
        }
        for ordinal in 0..self.parent_masters.len() {
            self.verify_parent_master_record(ordinal)?;
        }
        self.parent
            .verify_completion_receipt(&self.parent_completion)
    }
}

#[derive(Debug)]
struct PinnedDirectory {
    path: PathBuf,
    identity: fs::Metadata,
}

impl PinnedDirectory {
    #[cfg(unix)]
    fn pin(path: &Path) -> Result<Self, Qwen36TensorWorkError> {
        let identity = fs::symlink_metadata(path)
            .map_err(|error| work_io("pin additive campaign directory", error))?;
        if identity.file_type().is_symlink() || !identity.is_dir() {
            return Err(Qwen36TensorWorkError::InvalidPath(
                "additive campaign directory",
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            identity,
        })
    }

    fn validate(&self) -> Result<(), Qwen36TensorWorkError> {
        let current = fs::symlink_metadata(&self.path)
            .map_err(|error| work_io("reinspect additive campaign directory", error))?;
        if current.file_type().is_symlink()
            || !current.is_dir()
            || !same_file_identity(&self.identity, &current)
        {
            return Err(Qwen36TensorWorkError::InvalidPath(
                "replaced additive campaign directory",
            ));
        }
        Ok(())
    }
}

#[cfg(unix)]
fn pin_campaign_directories(
    root: &Path,
    slots: &Path,
    objects: &TensorWorkStore,
) -> Result<Vec<PinnedDirectory>, Qwen36TensorWorkError> {
    let paths = [root, slots, objects.objects_dir(), objects.temporary_dir()];
    let mut directories = Vec::new();
    directories
        .try_reserve_exact(paths.len())
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    for path in paths {
        directories.push(PinnedDirectory::pin(path)?);
    }
    Ok(directories)
}

fn validate_directories(directories: &[PinnedDirectory]) -> Result<(), Qwen36TensorWorkError> {
    for directory in directories {
        directory.validate()?;
    }
    Ok(())
}

fn master_record_spec(
    spec: &SaltV2MasterTensorSpec,
) -> Result<TensorRecordSpec, Qwen36TensorWorkError> {
    TensorRecordSpec::new(
        ContentId::of_bytes(SALT_V2_MASTER_TENSOR_SCHEMA),
        spec.source_model_id(),
        *spec.source_tensor_digest(),
        spec.name(),
        spec.shape().to_vec(),
        spec.canonical_bytes()
            .map_err(Qwen36TensorWorkError::Master)?,
        spec.payload_bytes(),
    )
    .map_err(Qwen36TensorWorkError::TensorStore)
}

fn validate_record_descriptor(
    record: &TensorRecordReceipt,
    expected: &SaltV2MasterTensorSpec,
) -> Result<(), Qwen36TensorWorkError> {
    let inner = SaltV2MasterTensorSpec::from_canonical_bytes(record.info().schema_metadata())
        .map_err(Qwen36TensorWorkError::Master)?;
    let record_spec = master_record_spec(expected)?;
    if inner != *expected || !record.matches_spec(&record_spec) {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "additive master record descriptor",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn recognized_slot_temporary(name: &str) -> bool {
    let Some(suffix) = name.strip_prefix(".additive-master-slot.tmp.") else {
        return false;
    };
    let mut fields = suffix.split('.');
    (0..3).all(|_| {
        fields.next().is_some_and(|field| {
            !field.is_empty() && field.bytes().all(|byte| byte.is_ascii_digit())
        })
    }) && fields.next().is_none()
}

#[derive(Clone, Copy, Debug)]
struct VerifiedMaster {
    master: SaltV2MasterTensorReceipt,
    fixed_id: Option<[u8; 32]>,
}

#[derive(Clone, Copy, Debug)]
enum FixedMasterMode {
    Skip,
    #[cfg(unix)]
    Capture,
    Require([u8; 32]),
}

#[derive(Clone, Copy, Debug)]
enum FixedCampaignMode<'a> {
    Skip,
    #[cfg(unix)]
    Capture,
    Require(&'a [[u8; 32]]),
}

impl FixedCampaignMode<'_> {
    fn validate_count(self, expected: usize) -> Result<(), Qwen36TensorWorkError> {
        if matches!(self, Self::Require(ids) if ids.len() != expected) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only fixed master count",
            ));
        }
        Ok(())
    }

    fn master_mode(self, ordinal: usize) -> Result<FixedMasterMode, Qwen36TensorWorkError> {
        match self {
            Self::Skip => Ok(FixedMasterMode::Skip),
            #[cfg(unix)]
            Self::Capture => Ok(FixedMasterMode::Capture),
            Self::Require(ids) => ids
                .get(ordinal)
                .copied()
                .map(FixedMasterMode::Require)
                .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                    "scale-only fixed master count",
                )),
        }
    }
}

#[derive(Debug)]
struct FixedMasterHasher {
    hasher: blake3::Hasher,
    expected_tiles: usize,
    next_tile: usize,
}

impl FixedMasterHasher {
    fn new(spec: &SaltV2MasterTensorSpec) -> Result<Self, Qwen36TensorWorkError> {
        let mut hasher = blake3::Hasher::new_derive_key(FIXED_MASTER_CONTEXT);
        hasher.update(&spec.tensor_index().to_le_bytes());
        hasher.update(spec.source_model_id().as_bytes());
        hasher.update(spec.source_tensor_digest());
        hasher.update(spec.widened_source_digest());
        let name_length = u64::try_from(spec.name().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master name"))?;
        hasher.update(&name_length.to_le_bytes());
        hasher.update(spec.name().as_bytes());
        let rank = u64::try_from(spec.shape().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master rank"))?;
        hasher.update(&rank.to_le_bytes());
        for dimension in spec.shape() {
            hasher.update(&dimension.to_le_bytes());
        }
        let logical = u64::try_from(spec.logical_coefficients()).map_err(|_| {
            Qwen36TensorWorkError::LengthOverflow("fixed master logical coefficients")
        })?;
        let tiles = u64::try_from(spec.tile_count())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master tile count"))?;
        hasher.update(&logical.to_le_bytes());
        hasher.update(&tiles.to_le_bytes());
        hasher.update(&[
            match spec.geometry().constraint {
                SaltV2FitConstraint::Dense => 0,
                SaltV2FitConstraint::S34 => 1,
            },
            spec.geometry().max_planes,
        ]);
        Ok(Self {
            hasher,
            expected_tiles: spec.tile_count(),
            next_tile: 0,
        })
    }

    fn try_push(&mut self, tile: SaltV2MasterTile) -> Result<(), Qwen36TensorWorkError> {
        let tile_ordinal = u64::try_from(self.next_tile)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master tile ordinal"))?;
        let plane_count = u8::try_from(tile.planes().len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master plane count"))?;
        self.hasher.update(&tile_ordinal.to_le_bytes());
        self.hasher.update(&[tile.admissible_planes(), plane_count]);
        for (plane_ordinal, plane) in tile.planes().iter().enumerate() {
            let plane_ordinal = u8::try_from(plane_ordinal)
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master plane ordinal"))?;
            let trit_count = u64::try_from(plane.trits().len())
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("fixed master trit count"))?;
            self.hasher.update(&[plane_ordinal]);
            self.hasher.update(&trit_count.to_le_bytes());
            let mut encoded = [0_u8; SALT_V2_ALLOCATION_TILE_SIZE];
            for (output, trit) in encoded.iter_mut().zip(plane.trits()) {
                *output = trit.get().to_le_bytes()[0];
            }
            self.hasher.update(&encoded[..plane.trits().len()]);
        }
        self.next_tile =
            self.next_tile
                .checked_add(1)
                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                    "fixed master tile count",
                ))?;
        Ok(())
    }

    fn finish(self) -> Result<[u8; 32], Qwen36TensorWorkError> {
        if self.next_tile != self.expected_tiles {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "fixed master tile count",
            ));
        }
        Ok(*self.hasher.finalize().as_bytes())
    }
}

struct CanonicalMasterValidator<'a> {
    decoder: SaltV2MasterTensorDecoder<'a>,
    fixed: Option<FixedMasterHasher>,
    expected_fixed_id: Option<[u8; 32]>,
}

impl<'a> CanonicalMasterValidator<'a> {
    fn new(
        spec: &'a SaltV2MasterTensorSpec,
        fixed_mode: FixedMasterMode,
    ) -> Result<Self, Qwen36TensorWorkError> {
        Ok(Self {
            decoder: SaltV2MasterTensorDecoder::new(spec).map_err(Qwen36TensorWorkError::Master)?,
            fixed: match fixed_mode {
                FixedMasterMode::Skip => None,
                #[cfg(unix)]
                FixedMasterMode::Capture => Some(FixedMasterHasher::new(spec)?),
                FixedMasterMode::Require(_) => Some(FixedMasterHasher::new(spec)?),
            },
            expected_fixed_id: match fixed_mode {
                FixedMasterMode::Require(fixed_id) => Some(fixed_id),
                FixedMasterMode::Skip => None,
                #[cfg(unix)]
                FixedMasterMode::Capture => None,
            },
        })
    }
}

impl TensorPayloadValidator for CanonicalMasterValidator<'_> {
    type Error = Qwen36TensorWorkError;
    type Output = VerifiedMaster;

    fn try_push(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.decoder
            .try_push(bytes, &mut |tile| match &mut self.fixed {
                Some(fixed) => fixed.try_push(tile),
                None => Ok(()),
            })
            .map_err(|error| match error {
                SaltV2MasterVisitError::Master(error) => Qwen36TensorWorkError::Master(error),
                SaltV2MasterVisitError::Visitor(error) => error,
            })
    }

    fn finish(self) -> Result<Self::Output, Self::Error> {
        let master = self
            .decoder
            .finish()
            .map_err(Qwen36TensorWorkError::Master)?;
        let fixed_id = self.fixed.map(FixedMasterHasher::finish).transpose()?;
        if self
            .expected_fixed_id
            .is_some_and(|expected| fixed_id != Some(expected))
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only fixed trits and prefixes",
            ));
        }
        Ok(VerifiedMaster { master, fixed_id })
    }
}

fn map_validator_visit_error(
    error: TensorVisitError<Qwen36TensorWorkError>,
) -> Qwen36TensorWorkError {
    match error {
        TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
        TensorVisitError::Sink(error) => error,
    }
}

fn derive_master_set_id(
    masters: &[Qwen36AdditiveMasterReceipt],
) -> Result<[u8; 32], Qwen36TensorWorkError> {
    let mut hasher = blake3::Hasher::new_derive_key(MASTER_SET_CONTEXT);
    let count = u64::try_from(masters.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("master set count"))?;
    hasher.update(&count.to_le_bytes());
    for master in masters {
        let name = master.record.info().name().as_bytes();
        let name_length = u64::try_from(name.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("master set name"))?;
        hasher.update(&master.ordinal.to_le_bytes());
        hasher.update(&name_length.to_le_bytes());
        hasher.update(name);
        hasher.update(&master.tensor_master_id);
    }
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug)]
struct CampaignMutationGuard<'a>(&'a Cell<bool>);

impl Drop for CampaignMutationGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

#[derive(Debug)]
struct CampaignLock {
    file: File,
    path: PathBuf,
    creator_pid: u32,
}

impl CampaignLock {
    fn validate_path(&self) -> Result<(), Qwen36TensorWorkError> {
        let opened = self
            .file
            .metadata()
            .map_err(|error| work_io("inspect held additive campaign lock", error))?;
        let current = fs::symlink_metadata(&self.path)
            .map_err(|error| work_io("reinspect additive campaign lock path", error))?;
        if current.file_type().is_symlink()
            || !opened.is_file()
            || !current.is_file()
            || !same_file_identity(&opened, &current)
        {
            return Err(Qwen36TensorWorkError::InvalidPath(
                "replaced additive campaign lock",
            ));
        }
        Ok(())
    }
}

impl Drop for CampaignLock {
    fn drop(&mut self) {
        if std::process::id() == self.creator_pid {
            let _ = self.file.unlock();
        }
    }
}

#[cfg(unix)]
fn acquire_campaign_lock(path: &Path) -> Result<CampaignLock, Qwen36TensorWorkError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(Qwen36TensorWorkError::InvalidPath("additive campaign lock"));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(work_io("inspect additive campaign lock", error)),
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|error| work_io("open additive campaign lock", error))?;
    let opened = file
        .metadata()
        .map_err(|error| work_io("inspect opened additive campaign lock", error))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| work_io("reinspect additive campaign lock", error))?;
    if after.file_type().is_symlink()
        || !opened.is_file()
        || !after.is_file()
        || !same_file_identity(&opened, &after)
    {
        return Err(Qwen36TensorWorkError::InvalidPath("additive campaign lock"));
    }
    match file.try_lock() {
        Ok(()) => Ok(CampaignLock {
            file,
            path: path.to_path_buf(),
            creator_pid: std::process::id(),
        }),
        Err(fs::TryLockError::WouldBlock) => Err(Qwen36TensorWorkError::CampaignLocked),
        Err(fs::TryLockError::Error(error)) => Err(work_io("lock additive campaign", error)),
    }
}

fn validate_expected_masters(
    masters: &[SaltV2MasterTensorSpec],
) -> Result<(), Qwen36TensorWorkError> {
    if masters[0].evidence().track != SaltV2MasterTrack::Ptq {
        return Err(Qwen36TensorWorkError::RefinedCampaignRequiresParent);
    }
    validate_expected_master_track(masters, SaltV2MasterTrack::Ptq)
}

fn validate_expected_master_track(
    masters: &[SaltV2MasterTensorSpec],
    track: SaltV2MasterTrack,
) -> Result<(), Qwen36TensorWorkError> {
    let first = masters
        .first()
        .ok_or(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive campaign tensor count",
        ))?;
    let common_evidence = first.evidence();
    let common_geometry = first.geometry();
    let common_model = first.source_model_id();
    if common_evidence.track != track {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "additive campaign track",
        ));
    }
    for (ordinal, master) in masters.iter().enumerate() {
        let expected_ordinal = u64::try_from(ordinal)
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive ordinal"))?;
        let evidence = master.evidence();
        if master.tensor_index() != expected_ordinal
            || (ordinal != 0 && masters[ordinal - 1].name() >= master.name())
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive campaign tensor order",
            ));
        }
        if master.source_model_id() != common_model
            || evidence.recipe_id != common_evidence.recipe_id
            || evidence.solver_id != common_evidence.solver_id
            || evidence.activation_digest != common_evidence.activation_digest
            || evidence.track != common_evidence.track
            || master.geometry() != common_geometry
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive campaign common recipe",
            ));
        }
        if is_zero(common_model.as_bytes())
            || is_zero(master.source_tensor_digest())
            || is_zero(master.widened_source_digest())
            || is_zero(&evidence.recipe_id)
            || is_zero(&evidence.solver_id)
            || is_zero(&evidence.activation_digest)
            || is_zero(&evidence.curvature_digest)
            || evidence.feedback_digest.as_ref().is_some_and(is_zero)
            || evidence.parent_master_id.as_ref().is_some_and(is_zero)
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "zero additive campaign identity",
            ));
        }
        master
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::Master)?;
    }
    Ok(())
}

#[cfg(unix)]
fn validate_scale_only_masters(
    parent_specs: &[SaltV2MasterTensorSpec],
    parent_masters: &[Qwen36AdditiveMasterReceipt],
    children: &[SaltV2MasterTensorSpec],
) -> Result<(), Qwen36TensorWorkError> {
    if children.is_empty()
        || children.len() > MAX_ACTIVE_TENSORS
        || children.len() != parent_specs.len()
        || children.len() != parent_masters.len()
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "scale-only parent tensor count",
        ));
    }
    validate_expected_master_track(children, SaltV2MasterTrack::ScaleOnly)?;
    for ((parent, parent_receipt), child) in parent_specs.iter().zip(parent_masters).zip(children) {
        if parent.evidence().track != SaltV2MasterTrack::Ptq
            || child.evidence().parent_master_id != Some(parent_receipt.tensor_master_id())
            || child.tensor_index() != parent.tensor_index()
            || child.name() != parent.name()
            || child.shape() != parent.shape()
            || child.logical_coefficients() != parent.logical_coefficients()
            || child.source_model_id() != parent.source_model_id()
            || child.source_tensor_digest() != parent.source_tensor_digest()
            || child.widened_source_digest() != parent.widened_source_digest()
            || child.geometry() != parent.geometry()
            || child.tile_count() != parent.tile_count()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "scale-only parent tensor binding",
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn validate_spec_against_base(
    spec: &Qwen36AdditiveCampaignSpec,
    base: &Qwen36TensorWorkStore<'_>,
    receipt: &Qwen36LanguageMtpWorkspaceReceipt,
) -> Result<u64, Qwen36TensorWorkError> {
    if spec.expected_masters.len() != base.additive_slots().len()
        || receipt.summary().additive_present() != 0
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "additive campaign tensor count",
        ));
    }
    let mut coefficients = 0_u64;
    for (ordinal, (master, slot)) in spec
        .expected_masters
        .iter()
        .zip(base.additive_slots())
        .enumerate()
    {
        let logical = u64::try_from(master.logical_coefficients())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("additive coefficients"))?;
        if master.tensor_index() != ordinal as u64
            || master.source_model_id() != receipt.source_model_id()
            || master.name() != slot.name()
            || master.shape() != slot.shape()
            || master.source_tensor_digest() != slot.source_tensor_digest()
            || logical != slot.coefficients()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive campaign source slot",
            ));
        }
        coefficients =
            coefficients
                .checked_add(logical)
                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                    "additive coefficients",
                ))?;
    }
    Ok(coefficients)
}

fn encode_master_catalog(
    masters: &[SaltV2MasterTensorSpec],
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let mut output = Vec::new();
    reserve_catalog_append(&mut output, 8 + 2 + 2 + 4, "master catalog too large")?;
    output.extend_from_slice(&CATALOG_MAGIC);
    output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    let count = u32::try_from(masters.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("master catalog count"))?;
    output.extend_from_slice(&count.to_le_bytes());
    for master in masters {
        let bytes = master
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::Master)?;
        let length = u32::try_from(bytes.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("master metadata"))?;
        reserve_catalog_append(
            &mut output,
            4_usize
                .checked_add(bytes.len())
                .ok_or(Qwen36TensorWorkError::LengthOverflow("master catalog"))?,
            "master catalog too large",
        )?;
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

#[cfg(unix)]
fn encode_scale_only_master_catalog(
    parent: &Qwen36CompleteWorkspaceReceipt,
    parent_masters: &[Qwen36AdditiveMasterReceipt],
    parent_fixed_ids: &[[u8; 32]],
    masters: &[SaltV2MasterTensorSpec],
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let mut output = Vec::new();
    reserve_catalog_append(
        &mut output,
        8 + 2 + 2 + (4 * 32) + 4,
        "scale-only master catalog too large",
    )?;
    output.extend_from_slice(&SCALE_ONLY_CATALOG_MAGIC);
    output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(parent.completion_id().as_bytes());
    output.extend_from_slice(parent.base_workspace_id().as_bytes());
    output.extend_from_slice(parent.campaign_id().as_bytes());
    output.extend_from_slice(&parent.master_set_id());
    let count = u32::try_from(masters.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("scale-only master catalog count"))?;
    output.extend_from_slice(&count.to_le_bytes());
    for ((parent_master, parent_fixed_id), master) in
        parent_masters.iter().zip(parent_fixed_ids).zip(masters)
    {
        let bytes = master
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::Master)?;
        let length = u32::try_from(bytes.len())
            .map_err(|_| Qwen36TensorWorkError::LengthOverflow("scale-only master metadata"))?;
        reserve_catalog_append(
            &mut output,
            (2_usize * 32)
                .checked_add(4)
                .and_then(|length| length.checked_add(bytes.len()))
                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                    "scale-only master catalog",
                ))?,
            "scale-only master catalog too large",
        )?;
        output.extend_from_slice(&parent_master.tensor_master_id());
        output.extend_from_slice(parent_fixed_id);
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

#[cfg(unix)]
fn encode_package_scale_only_master_catalog(
    parent: &Qwen36CompleteWorkspaceReceipt,
    parent_masters: &[Qwen36AdditiveMasterReceipt],
    parent_fixed_ids: &[[u8; 32]],
    binding: PackageScaleOnlyBinding,
    masters: &[SaltV2MasterTensorSpec],
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let mut output = Vec::new();
    reserve_catalog_append(
        &mut output,
        8 + 2 + 2 + (8 * 32) + 4,
        "package-admitted scale-only catalog too large",
    )?;
    output.extend_from_slice(&PACKAGE_SCALE_ONLY_CATALOG_MAGIC);
    output.extend_from_slice(&2_u16.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(parent.completion_id().as_bytes());
    output.extend_from_slice(parent.base_workspace_id().as_bytes());
    output.extend_from_slice(parent.campaign_id().as_bytes());
    output.extend_from_slice(&parent.master_set_id());
    output.extend_from_slice(binding.admission_id.as_bytes());
    output.extend_from_slice(binding.selection_id.as_bytes());
    output.extend_from_slice(&binding.compact_package_id);
    output.extend_from_slice(&binding.near_package_id);
    let count = u32::try_from(masters.len()).map_err(|_| {
        Qwen36TensorWorkError::LengthOverflow("package-admitted scale-only master catalog count")
    })?;
    output.extend_from_slice(&count.to_le_bytes());
    for ((parent_master, parent_fixed_id), master) in
        parent_masters.iter().zip(parent_fixed_ids).zip(masters)
    {
        let bytes = master
            .canonical_bytes()
            .map_err(Qwen36TensorWorkError::Master)?;
        let length = u32::try_from(bytes.len()).map_err(|_| {
            Qwen36TensorWorkError::LengthOverflow("package-admitted scale-only master metadata")
        })?;
        reserve_catalog_append(
            &mut output,
            (2_usize * 32)
                .checked_add(4)
                .and_then(|length| length.checked_add(bytes.len()))
                .ok_or(Qwen36TensorWorkError::LengthOverflow(
                    "package-admitted scale-only master catalog",
                ))?,
            "package-admitted scale-only catalog too large",
        )?;
        output.extend_from_slice(&parent_master.tensor_master_id());
        output.extend_from_slice(parent_fixed_id);
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

fn reserve_catalog_append(
    output: &mut Vec<u8>,
    additional: usize,
    too_large: &'static str,
) -> Result<(), Qwen36TensorWorkError> {
    let next = output
        .len()
        .checked_add(additional)
        .ok_or(Qwen36TensorWorkError::LengthOverflow("master catalog"))?;
    if next > MAX_WORKSPACE_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(too_large));
    }
    output
        .try_reserve(additional)
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)
}

#[cfg(unix)]
fn campaign_descriptor_bytes(
    base: &Qwen36LanguageMtpWorkspaceReceipt,
    spec: &Qwen36AdditiveCampaignSpec,
    additive_coefficients: u64,
) -> Result<Vec<u8>, Qwen36TensorWorkError> {
    let mut output = Vec::new();
    output.extend_from_slice(&CAMPAIGN_MAGIC);
    output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(base.workspace_id().as_bytes());
    output.extend_from_slice(base.proof_id().as_bytes());
    output.extend_from_slice(base.manifest_content_id().as_bytes());
    output.extend_from_slice(base.source_model_id().as_bytes());
    output.extend_from_slice(base.coverage_policy_digest());
    output.push(identity_status_tag(base.identity_status()));
    output.extend_from_slice(ContentId::of_bytes(SALT_V2_MASTER_TENSOR_SCHEMA).as_bytes());
    output.extend_from_slice(spec.spec_id.as_bytes());
    output.extend_from_slice(&additive_coefficients.to_le_bytes());
    for value in summary_values(base.summary()) {
        output.extend_from_slice(&value.to_le_bytes());
    }
    let catalog_length = u32::try_from(spec.catalog_bytes.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("master catalog"))?;
    output.extend_from_slice(&catalog_length.to_le_bytes());
    output.extend_from_slice(&spec.catalog_bytes);
    let mut hasher = blake3::Hasher::new_derive_key(CAMPAIGN_CHECKSUM_CONTEXT);
    hasher.update(&output);
    output.extend_from_slice(hasher.finalize().as_bytes());
    if output.len() > MAX_WORKSPACE_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "additive campaign descriptor too large",
        ));
    }
    Ok(output)
}

fn is_zero(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[derive(Debug)]
struct CanonicalCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CanonicalCursor<'a> {
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
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "additive campaign cursor",
            ))?;
        let value =
            self.bytes
                .get(self.offset..end)
                .ok_or(Qwen36TensorWorkError::WorkspaceMalformed(
                    "truncated additive campaign artifact",
                ))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, Qwen36TensorWorkError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Qwen36TensorWorkError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
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

#[cfg(all(test, unix))]
mod tests {
    use std::{
        cell::Cell,
        convert::Infallible,
        io::{self, Cursor, Write},
    };

    use half::f16;
    use tritium_format::{
        SemanticTensorHasher,
        salt_v2_master::{
            SaltV2FitConstraint, SaltV2MasterError, SaltV2MasterEvidence, SaltV2MasterGeometry,
            SaltV2MasterTensorEncoder, SaltV2MasterTrack, SaltV2PrefixLoss,
        },
        salt_v2_package::{
            SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
        },
    };
    use tritium_nn::{NnError, Qwen35TensorStreamError};
    use tritium_quantize::{
        ByteDelta, NestedProfileBudgets, PhysicalBytes, ProfileBudget, Qwen35SourceDtype,
        Qwen35TensorRole, Qwen35TensorScope,
    };

    use super::super::{
        PreservedTensorSource, Qwen36AdditiveSlotState, Qwen36AdditiveWorkSlot, WorkspacePlan,
    };
    use super::*;

    #[derive(Debug)]
    struct EmptySource;

    #[derive(Debug, PartialEq, Eq)]
    struct AllocationSourceError;

    impl PreservedTensorSource for EmptySource {
        fn try_visit_tensor_bytes(
            &self,
            name: &str,
            _max_chunk_bytes: usize,
            _visit: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        ) -> Result<u64, Qwen35TensorStreamError<io::Error>> {
            Err(Qwen35TensorStreamError::Source(NnError::MissingTensor(
                name.to_owned(),
            )))
        }

        fn source_tensor_semantic_hasher(
            &self,
            name: &str,
        ) -> Result<SemanticTensorHasher, NnError> {
            Err(NnError::MissingTensor(name.to_owned()))
        }
    }

    fn fixture_plan() -> WorkspacePlan {
        let source_model_id = ModelId::from_digest([11; 32]);
        WorkspacePlan {
            proof_id: ContentId::of_bytes(b"two-slot proof"),
            manifest_content_id: ContentId::of_bytes(b"two-slot manifest"),
            source_model_id,
            coverage_policy_digest: [12; 32],
            identity_status: Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration,
            summary: Qwen36TensorWorkSummary {
                active_tensors: 2,
                additive_required: 2,
                additive_present: 0,
                preserved_tensors: 0,
                deferred_vision_tensors: 0,
                active_coefficients: 8,
                preserved_coefficients: 0,
                preserved_payload_bytes: 0,
            },
            additive: vec![
                Qwen36AdditiveWorkSlot {
                    name: "language.weight".into(),
                    dtype: Qwen35SourceDtype::Bfloat16,
                    shape: vec![1, 4],
                    coefficients: 4,
                    scope: Qwen35TensorScope::Language,
                    role: Qwen35TensorRole::MlpProjection,
                    source_tensor_digest: [21; 32],
                    state: Qwen36AdditiveSlotState::MissingCanonicalMaster,
                },
                Qwen36AdditiveWorkSlot {
                    name: "mtp.weight".into(),
                    dtype: Qwen35SourceDtype::Bfloat16,
                    shape: vec![1, 4],
                    coefficients: 4,
                    scope: Qwen35TensorScope::MtpDrafter,
                    role: Qwen35TensorRole::MtpFusionProjection,
                    source_tensor_digest: [22; 32],
                    state: Qwen36AdditiveSlotState::MissingCanonicalMaster,
                },
            ],
            preserved: Vec::new(),
        }
    }

    fn fixture_master(
        name: &str,
        source_digest: [u8; 32],
        ordinal: u64,
        widened_digest: [u8; 32],
    ) -> SaltV2MasterTensorSpec {
        SaltV2MasterTensorSpec::new(
            name,
            vec![1, 4],
            ModelId::from_digest([11; 32]),
            source_digest,
            widened_digest,
            ordinal,
            SaltV2MasterEvidence {
                recipe_id: [31; 32],
                solver_id: [32; 32],
                activation_digest: [33; 32],
                curvature_digest: [if ordinal == 0 { 40 } else { 41 }; 32],
                feedback_digest: None,
                track: SaltV2MasterTrack::Ptq,
                parent_master_id: None,
            },
            SaltV2MasterGeometry {
                constraint: SaltV2FitConstraint::S34,
                max_planes: 2,
            },
        )
        .expect("valid fixture master")
    }

    fn fixture_campaign_spec() -> Qwen36AdditiveCampaignSpec {
        Qwen36AdditiveCampaignSpec::new(vec![
            fixture_master("language.weight", [21; 32], 0, [51; 32]),
            fixture_master("mtp.weight", [22; 32], 1, [52; 32]),
        ])
        .expect("valid fixture campaign")
    }

    fn fixture_selection_spec() -> Qwen36SelectedAllocationSpec {
        let metadata = ByteDelta::measured(
            PhysicalBytes {
                serialized: 32,
                resident: 16,
            },
            PhysicalBytes {
                serialized: 32,
                resident: 16,
            },
        );
        Qwen36SelectedAllocationSpec::new(
            tritium_format::salt_v2::SaltV2Codec::D2,
            [81; 32],
            [82; 32],
            NestedProfileBudgets {
                compact: ProfileBudget {
                    maximum: PhysicalBytes {
                        serialized: 16 * 1024,
                        resident: 4 * 1024,
                    },
                    metadata,
                },
                near_lossless: ProfileBudget {
                    maximum: PhysicalBytes {
                        serialized: 16 * 1024,
                        resident: 4 * 1024,
                    },
                    metadata,
                },
            },
        )
        .expect("valid fixture selection spec")
    }

    fn fixture_allocation_counts() -> impl Iterator<Item = Result<(u8, u8), Infallible>> {
        [(1, 1), (1, 2)].into_iter().map(Ok)
    }

    fn fixture_selected_package(near_lossless: bool) -> Vec<u8> {
        let first = SaltV2Plane::new(vec![-1, 0, 1, -1], vec![f16::from_f32(0.5)]).unwrap();
        let second = SaltV2Plane::new(vec![1, -1, 0, 1], vec![f16::from_f32(0.25)]).unwrap();
        let language = SaltV2Tensor::new(
            "language.weight",
            vec![1, 4],
            vec![SaltV2Tile::new(vec![first.clone()]).unwrap()],
        )
        .unwrap();
        let mtp_planes = if near_lossless {
            vec![first, second]
        } else {
            vec![first]
        };
        let mtp = SaltV2Tensor::new(
            "mtp.weight",
            vec![1, 4],
            vec![SaltV2Tile::new(mtp_planes).unwrap()],
        )
        .unwrap();
        write_salt_v2_package(
            &SaltV2Package::new(
                tritium_format::salt_v2::SaltV2Codec::D2,
                vec![language, mtp],
            )
            .unwrap(),
        )
        .unwrap()
        .bytes
    }

    fn bind_fixture_allocation<'parent, 'store, 'source>(
        parent: &'parent Qwen36AdditiveCampaignStore<'store, 'source>,
    ) -> Qwen36AllocatedCampaignStore<'parent, 'store, 'source> {
        parent
            .bind_selected_allocation(fixture_selection_spec(), fixture_allocation_counts())
            .expect("bind fixture selected allocation")
    }

    fn fixture_scale_only_master(
        parent: &Qwen36AdditiveMasterReceipt,
        name: &str,
        source_digest: [u8; 32],
        ordinal: u64,
        widened_digest: [u8; 32],
    ) -> SaltV2MasterTensorSpec {
        SaltV2MasterTensorSpec::new(
            name,
            vec![1, 4],
            ModelId::from_digest([11; 32]),
            source_digest,
            widened_digest,
            ordinal,
            SaltV2MasterEvidence {
                recipe_id: [61; 32],
                solver_id: [62; 32],
                activation_digest: [63; 32],
                curvature_digest: [if ordinal == 0 { 70 } else { 71 }; 32],
                feedback_digest: None,
                track: SaltV2MasterTrack::ScaleOnly,
                parent_master_id: Some(parent.tensor_master_id()),
            },
            SaltV2MasterGeometry {
                constraint: SaltV2FitConstraint::S34,
                max_planes: 2,
            },
        )
        .expect("valid scale-only fixture master")
    }

    fn write_fixture_master(
        spec: &SaltV2MasterTensorSpec,
        writer: &mut TensorPayloadWriter<'_>,
        admissible_planes: u8,
    ) -> Result<(), SaltV2MasterError> {
        let losses = [
            SaltV2PrefixLoss::new(4.0, 3.0)?,
            SaltV2PrefixLoss::new(1.0, 0.5)?,
        ];
        let planes = [
            SaltV2Plane::new(vec![-1, 0, 1, -1], vec![f16::from_f32(0.5)])?,
            SaltV2Plane::new(vec![1, -1, 0, 1], vec![f16::from_f32(0.25)])?,
        ];
        let mut encoder = SaltV2MasterTensorEncoder::new(spec, writer)?;
        encoder.write_tile(admissible_planes, &losses, &planes)?;
        encoder.finish()?;
        Ok(())
    }

    fn write_scale_only_fixture_master(
        spec: &SaltV2MasterTensorSpec,
        writer: &mut TensorPayloadWriter<'_>,
        admissible_planes: u8,
    ) -> Result<(), SaltV2MasterError> {
        write_scale_only_fixture_master_with_first_trits(spec, writer, admissible_planes, -1, 1)
    }

    fn write_scale_only_fixture_master_with_first_trit(
        spec: &SaltV2MasterTensorSpec,
        writer: &mut TensorPayloadWriter<'_>,
        admissible_planes: u8,
        first_trit: i8,
    ) -> Result<(), SaltV2MasterError> {
        write_scale_only_fixture_master_with_first_trits(
            spec,
            writer,
            admissible_planes,
            first_trit,
            1,
        )
    }

    fn write_scale_only_fixture_master_with_first_trits(
        spec: &SaltV2MasterTensorSpec,
        writer: &mut TensorPayloadWriter<'_>,
        admissible_planes: u8,
        first_plane_trit: i8,
        second_plane_trit: i8,
    ) -> Result<(), SaltV2MasterError> {
        let losses = [
            SaltV2PrefixLoss::new(3.0, 2.0)?,
            SaltV2PrefixLoss::new(0.75, 0.25)?,
        ];
        let planes = [
            SaltV2Plane::new(vec![first_plane_trit, 0, 1, -1], vec![f16::from_f32(0.625)])?,
            SaltV2Plane::new(
                vec![second_plane_trit, -1, 0, 1],
                vec![f16::from_f32(0.375)],
            )?,
        ];
        let mut encoder = SaltV2MasterTensorEncoder::new(spec, writer)?;
        encoder.write_tile(admissible_planes, &losses, &planes)?;
        encoder.finish()?;
        Ok(())
    }

    fn fixture_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-qwen36-additive-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn object_record_count(objects: &Path) -> usize {
        fs::read_dir(objects)
            .expect("read object directory")
            .map(|entry| {
                let entry = entry.expect("read object entry");
                if entry.file_type().expect("inspect object entry").is_dir() {
                    fs::read_dir(entry.path())
                        .expect("read object prefix")
                        .count()
                } else {
                    1
                }
            })
            .sum()
    }

    fn seal_fixture_in_order(label: &str, order: [usize; 2]) -> Qwen36CompleteWorkspaceReceipt {
        let root = fixture_root(label);
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("open additive campaign");
        for ordinal in order {
            campaign
                .install_master(&campaign_spec.expected_masters()[ordinal], |writer| {
                    write_fixture_master(
                        &campaign_spec.expected_masters()[ordinal],
                        writer,
                        if ordinal == 0 { 1 } else { 2 },
                    )
                })
                .expect("install fixture master");
        }
        let receipt = campaign.seal_complete().expect("seal fixture campaign");
        drop(campaign);
        drop(base);
        let _ = fs::remove_dir_all(root);
        receipt
    }

    #[test]
    fn campaign_resumes_seals_and_preserves_the_base_workspace() {
        let root = fixture_root("resume-seal");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        let base_receipt = base.reconcile_preserved().expect("seal empty base");
        let base_bytes = fs::read(base.workspace_path()).expect("read base bytes");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("open additive campaign");

        assert!(matches!(
            base.open_master_campaign(campaign_spec.clone()),
            Err(Qwen36TensorWorkError::CampaignLocked)
        ));
        assert_eq!(campaign.progress().unwrap().additive_present(), 0);
        assert!(matches!(
            campaign.seal_complete(),
            Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                expected: 2,
                present: 0
            })
        ));
        assert!(!campaign.completion_path().exists());

        let first_calls = Cell::new(0_u32);
        let first = campaign
            .install_master(&campaign_spec.expected_masters()[0], |writer| {
                first_calls.set(first_calls.get() + 1);
                write_fixture_master(&campaign_spec.expected_masters()[0], writer, 1)
            })
            .expect("install shortened S34 master");
        assert_eq!(first_calls.get(), 1);
        assert_eq!(campaign.progress().unwrap().additive_present(), 1);
        assert!(matches!(
            campaign.seal_complete(),
            Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                expected: 2,
                present: 1
            })
        ));
        assert!(!campaign.completion_path().exists());

        let campaign_id = campaign.campaign_id();
        let stale_temporary = campaign
            .objects
            .temporary_dir()
            .join("crash-left-record.tmp");
        fs::write(&stale_temporary, b"partial record").expect("write stale temporary");
        drop(campaign);
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("resume additive campaign");
        assert_eq!(campaign.campaign_id(), campaign_id);
        assert!(!stale_temporary.exists());
        assert_eq!(campaign.progress().unwrap().additive_present(), 1);

        let resumed = campaign
            .install_master(&campaign_spec.expected_masters()[0], |_writer| {
                first_calls.set(first_calls.get() + 1);
                Ok::<(), SaltV2MasterError>(())
            })
            .expect("reopen installed master without producer");
        assert_eq!(resumed, first);
        assert_eq!(first_calls.get(), 1);

        let wrong = fixture_master("language.weight", [21; 32], 0, [99; 32]);
        assert!(matches!(
            campaign.install_master(&wrong, |_writer| -> Result<(), SaltV2MasterError> {
                panic!("unexpected producer invocation")
            }),
            Err(Qwen36AdditiveInstallError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("additive master specification")
            ))
        ));

        campaign
            .install_master(&campaign_spec.expected_masters()[1], |writer| {
                write_fixture_master(&campaign_spec.expected_masters()[1], writer, 2)
            })
            .expect("install second master");
        assert_eq!(campaign.progress().unwrap().additive_present(), 2);
        assert_eq!(fs::read(base.workspace_path()).unwrap(), base_bytes);
        assert_eq!(base.reopen_workspace().unwrap(), base_receipt);
        assert!(matches!(
            base.require_complete(),
            Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                expected: 2,
                present: 0
            })
        ));

        let completion = campaign.seal_complete().expect("seal complete campaign");
        assert_eq!(completion, campaign.require_complete().unwrap());
        assert_eq!(completion.base_workspace_id(), base_receipt.workspace_id());
        assert_eq!(completion.campaign_id(), campaign.campaign_id());
        assert_eq!(completion.additive_coefficients(), 8);
        assert!(completion.summary().complete());
        assert!(
            !completion
                .identity_status()
                .official_payload_authenticated()
        );
        assert_eq!(fs::read(base.workspace_path()).unwrap(), base_bytes);

        drop(campaign);
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("reopen sealed additive campaign");
        assert_eq!(campaign.require_complete().unwrap(), completion);

        let record_path = campaign.objects.record_path(first.record.record_id());
        fs::remove_file(record_path).expect("remove referenced record");
        assert!(campaign.require_complete().is_err());

        drop(campaign);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_master_and_changed_campaign_descriptor_fail_before_slot_use() {
        let root = fixture_root("malformed-master");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("open additive campaign");
        let expected = &campaign_spec.expected_masters()[0];
        let invalid =
            vec![0_u8; usize::try_from(expected.payload_bytes()).expect("small fixture payload")];

        assert!(matches!(
            campaign.install_master(expected, |writer| writer.write_all(&invalid)),
            Err(Qwen36AdditiveInstallError::Campaign(
                Qwen36TensorWorkError::Master(_)
            ))
        ));
        assert!(!campaign.slot_path(0, expected.name()).exists());
        assert_eq!(object_record_count(campaign.objects.objects_dir()), 0);
        assert_eq!(
            fs::read_dir(campaign.objects.temporary_dir())
                .expect("read temporary directory")
                .count(),
            0
        );
        assert_eq!(campaign.progress().unwrap().additive_present(), 0);

        let mut descriptor = campaign.descriptor_bytes.clone();
        descriptor[0] ^= 0xff;
        fs::write(campaign.descriptor_path(), descriptor).expect("tamper descriptor");
        assert!(matches!(
            campaign.progress(),
            Err(Qwen36TensorWorkError::ExistingArtifactMismatch(
                "additive campaign descriptor"
            ))
        ));

        drop(campaign);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn campaign_reclaims_crash_orphans_without_touching_referenced_masters() {
        let root = fixture_root("orphan-reclamation");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("open additive campaign");
        let first = campaign
            .install_master(&campaign_spec.expected_masters()[0], |writer| {
                assert!(matches!(
                    campaign.seal_complete(),
                    Err(Qwen36TensorWorkError::CampaignLocked)
                ));
                write_fixture_master(&campaign_spec.expected_masters()[0], writer, 1)
            })
            .expect("install referenced master");
        let first_path = campaign.objects.record_path(first.record.record_id());
        let orphan_spec =
            master_record_spec(&campaign_spec.expected_masters()[1]).expect("orphan record spec");
        let orphan = campaign
            .objects
            .put(&orphan_spec, |writer| {
                write_fixture_master(&campaign_spec.expected_masters()[1], writer, 2)
            })
            .expect("publish crash-window orphan");
        let orphan_path = campaign.objects.record_path(orphan.record_id());
        let stale_slot_temporary = campaign.slots.join(".additive-master-slot.tmp.1.2.3");
        fs::write(&stale_slot_temporary, b"partial slot").expect("write stale slot temporary");
        assert_eq!(object_record_count(campaign.objects.objects_dir()), 2);
        drop(campaign);

        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("reopen and reclaim campaign");
        assert!(first_path.exists());
        assert!(!orphan_path.exists());
        assert!(!stale_slot_temporary.exists());
        assert_eq!(object_record_count(campaign.objects.objects_dir()), 1);
        assert_eq!(campaign.reopen_master("language.weight").unwrap(), first);
        assert_eq!(campaign.progress().unwrap().additive_present(), 1);

        drop(campaign);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_object_layout_aborts_reclamation_before_any_unlink() {
        let root = fixture_root("orphan-reclamation-fail-closed");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec.clone())
            .expect("open additive campaign");
        let mut orphan_paths = Vec::new();
        for (ordinal, expected) in campaign_spec.expected_masters().iter().enumerate() {
            let record_spec = master_record_spec(expected).expect("orphan record spec");
            let orphan = campaign
                .objects
                .put(&record_spec, |writer| {
                    write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                })
                .expect("publish crash-window orphan");
            orphan_paths.push(campaign.objects.record_path(orphan.record_id()));
        }
        let unknown = campaign.objects.objects_dir().join("not-a-prefix");
        fs::write(&unknown, b"unknown object layout").expect("write unknown object entry");
        let stale_slot_temporary = campaign.slots.join(".additive-master-slot.tmp.1.2.3");
        fs::write(&stale_slot_temporary, b"partial slot").expect("write stale slot temporary");
        drop(campaign);

        assert!(matches!(
            base.open_master_campaign(campaign_spec.clone()),
            Err(Qwen36TensorWorkError::TensorStore(
                crate::TensorWorkError::InvalidPath("record prefix directory")
            ))
        ));
        assert!(orphan_paths.iter().all(|path| path.exists()));
        assert!(stale_slot_temporary.exists());

        fs::remove_file(unknown).expect("remove unknown object entry");
        let campaign = base
            .open_master_campaign(campaign_spec)
            .expect("retry canonical reclamation");
        assert!(orphan_paths.iter().all(|path| !path.exists()));
        assert!(!stale_slot_temporary.exists());
        assert_eq!(object_record_count(campaign.objects.objects_dir()), 0);

        drop(campaign);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn replaced_lock_and_campaign_directories_poison_live_handles() {
        let root = fixture_root("namespace-replacement");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign_spec = fixture_campaign_spec();
        let campaign = base
            .open_master_campaign(campaign_spec)
            .expect("open additive campaign");

        let slots_backup = campaign.root.join("additive-slots-replaced");
        fs::rename(&campaign.slots, &slots_backup).expect("replace slot directory");
        fs::create_dir(&campaign.slots).expect("create substitute slot directory");
        assert!(matches!(
            campaign.progress(),
            Err(Qwen36TensorWorkError::InvalidPath(
                "replaced additive campaign directory"
            ))
        ));
        drop(campaign);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn replacing_the_locked_descriptor_is_detected_before_more_work() {
        let root = fixture_root("lock-replacement");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let campaign = base
            .open_master_campaign(fixture_campaign_spec())
            .expect("open additive campaign");
        let descriptor_path = campaign.descriptor_path();
        let old_descriptor = campaign.root.join("campaign.replaced");
        fs::rename(&descriptor_path, old_descriptor).expect("unlink locked descriptor path");
        fs::write(&descriptor_path, &campaign.descriptor_bytes)
            .expect("create byte-identical replacement descriptor");

        assert!(matches!(
            campaign.progress(),
            Err(Qwen36TensorWorkError::InvalidPath(
                "replaced additive campaign lock"
            ))
        ));

        drop(campaign);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn install_order_does_not_change_campaign_or_completion_identity() {
        let forward = seal_fixture_in_order("order-forward", [0, 1]);
        let reverse = seal_fixture_in_order("order-reverse", [1, 0]);

        assert_eq!(forward, reverse);
        assert_eq!(forward.master_set_id(), reverse.master_set_id());
        assert_eq!(forward.completion_id(), reverse.completion_id());
    }

    #[test]
    fn refined_campaigns_fail_closed_until_parent_planes_can_be_verified() {
        let refined = SaltV2MasterTensorSpec::new(
            "language.weight",
            vec![1, 4],
            ModelId::from_digest([11; 32]),
            [21; 32],
            [51; 32],
            0,
            SaltV2MasterEvidence {
                recipe_id: [31; 32],
                solver_id: [32; 32],
                activation_digest: [33; 32],
                curvature_digest: [40; 32],
                feedback_digest: None,
                track: SaltV2MasterTrack::ScaleOnly,
                parent_master_id: Some([71; 32]),
            },
            SaltV2MasterGeometry {
                constraint: SaltV2FitConstraint::S34,
                max_planes: 2,
            },
        )
        .expect("syntactically valid refined master");

        assert!(matches!(
            Qwen36AdditiveCampaignSpec::new(vec![refined]),
            Err(Qwen36TensorWorkError::RefinedCampaignRequiresParent)
        ));
    }

    #[test]
    fn sealed_ptq_campaign_admits_parent_bound_scale_only_campaign() {
        let root = fixture_root("scale-only-open");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let parent_spec = fixture_campaign_spec();
        let parent = base
            .open_master_campaign(parent_spec.clone())
            .expect("open PTQ parent campaign");
        let mut parent_masters = Vec::new();
        for (ordinal, expected) in parent_spec.expected_masters().iter().enumerate() {
            parent_masters.push(
                parent
                    .install_master(expected, |writer| {
                        write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                    })
                    .expect("install PTQ parent master"),
            );
        }
        let child_specs = vec![
            fixture_scale_only_master(&parent_masters[0], "language.weight", [21; 32], 0, [51; 32]),
            fixture_scale_only_master(&parent_masters[1], "mtp.weight", [22; 32], 1, [52; 32]),
        ];
        assert!(matches!(
            parent.open_scale_only_campaign(child_specs.clone()),
            Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
                expected: 2,
                present: 2
            })
        ));
        let parent_completion = parent.seal_complete().expect("seal PTQ parent");

        let child = parent
            .open_scale_only_campaign(child_specs)
            .expect("open parent-bound scale-only campaign");
        assert_eq!(
            child.parent_completion_id(),
            parent_completion.completion_id()
        );
        assert_ne!(child.campaign_id(), parent.campaign_id());
        assert!(matches!(
            base.open_master_campaign(child.spec().clone()),
            Err(Qwen36TensorWorkError::RefinedCampaignRequiresParent)
        ));

        drop(child);
        drop(parent);
        drop(base);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn selected_allocation_reopens_and_rejects_invalid_or_changed_maps() {
        let root = fixture_root("selected-allocation-binding");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let parent_spec = fixture_campaign_spec();
        let parent = base
            .open_master_campaign(parent_spec.clone())
            .expect("open PTQ parent campaign");
        for (ordinal, expected) in parent_spec.expected_masters().iter().enumerate() {
            parent
                .install_master(expected, |writer| {
                    write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                })
                .expect("install PTQ parent master");
        }
        parent.seal_complete().expect("seal PTQ parent");

        assert!(matches!(
            parent.bind_selected_allocation(
                fixture_selection_spec(),
                [Ok::<_, AllocationSourceError>((1, 1))]
            ),
            Err(Qwen36SelectedAllocationBindError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("selected allocation source is short")
            ))
        ));
        assert!(matches!(
            parent.bind_selected_allocation(
                fixture_selection_spec(),
                [
                    Ok::<_, AllocationSourceError>((1, 1)),
                    Ok((1, 2)),
                    Ok((1, 2)),
                ]
            ),
            Err(Qwen36SelectedAllocationBindError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("selected allocation source is long")
            ))
        ));
        assert!(matches!(
            parent.bind_selected_allocation(
                fixture_selection_spec(),
                [Ok((1, 1)), Err(AllocationSourceError)]
            ),
            Err(Qwen36SelectedAllocationBindError::Source(
                AllocationSourceError
            ))
        ));
        assert!(matches!(
            parent.bind_selected_allocation(
                fixture_selection_spec(),
                [(2, 1), (1, 2)].into_iter().map(Ok::<_, Infallible>)
            ),
            Err(Qwen36SelectedAllocationBindError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("selected nested admissible prefixes")
            ))
        ));
        assert!(matches!(
            parent.bind_selected_allocation(
                fixture_selection_spec(),
                [(1, 2), (1, 2)].into_iter().map(Ok::<_, Infallible>)
            ),
            Err(Qwen36SelectedAllocationBindError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("selected nested admissible prefixes")
            ))
        ));
        assert!(
            !parent
                .root()
                .join("selected-allocation/selected-allocation.tq36a")
                .exists()
        );
        let failed_store = TensorWorkStore::open(&parent.root().join("selected-allocation"))
            .expect("open failed selected allocation store");
        assert_eq!(object_record_count(failed_store.objects_dir()), 0);
        assert_eq!(
            fs::read_dir(failed_store.temporary_dir())
                .expect("read selected allocation staging")
                .count(),
            0
        );

        let allocated = bind_fixture_allocation(&parent);
        let mut receipt = allocated.receipt().clone();
        assert_eq!(receipt.tensor_count(), 2);
        assert_eq!(receipt.tile_count(), 2);
        assert_eq!(receipt.compact().map_record().info().payload_bytes(), 1);
        assert_eq!(
            receipt.near_lossless().map_record().info().payload_bytes(),
            1
        );
        assert_ne!(
            receipt.compact().allocation_map_id(),
            receipt.near_lossless().allocation_map_id()
        );
        drop(allocated);

        let reopened = parent
            .reopen_selected_allocation()
            .expect("reopen selected allocation");
        assert_eq!(reopened.receipt(), &receipt);
        drop(reopened);

        selected_allocation::rewrite_selected_allocation_for_test(&parent, &[1, 1], &[2, 2], None)
            .expect("publish canonical inadmissible selection");
        assert!(matches!(
            parent.reopen_selected_allocation(),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected nested admissible prefixes"
            ))
        ));
        selected_allocation::rewrite_selected_allocation_for_test(&parent, &[2, 1], &[1, 2], None)
            .expect("publish canonical non-nested selection");
        assert!(matches!(
            parent.reopen_selected_allocation(),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected nested admissible prefixes"
            ))
        ));
        selected_allocation::rewrite_selected_allocation_for_test(
            &parent,
            &[1, 1],
            &[1, 2],
            Some(ContentId::of_bytes(b"forged nested allocation")),
        )
        .expect("publish canonical forged nested identity");
        assert!(matches!(
            parent.reopen_selected_allocation(),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected allocation semantic receipt"
            ))
        ));
        receipt = selected_allocation::rewrite_selected_allocation_for_test(
            &parent,
            &[1, 1],
            &[1, 2],
            None,
        )
        .expect("restore canonical valid selection");
        parent
            .reopen_selected_allocation()
            .expect("reopen restored selected allocation");

        let selection_store = TensorWorkStore::open(&parent.root().join("selected-allocation"))
            .expect("open selected allocation CAS");
        fs::write(
            selection_store.record_path(receipt.compact().map_record().record_id()),
            b"changed map record",
        )
        .expect("tamper selected map record");
        assert!(parent.reopen_selected_allocation().is_err());

        drop(parent);
        drop(base);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn selected_packages_require_exact_maps_parent_prefixes_and_budget_ledgers() {
        let root = fixture_root("selected-package-admission");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let parent_spec = fixture_campaign_spec();
        let parent = base
            .open_master_campaign(parent_spec.clone())
            .expect("open PTQ parent campaign");
        let mut parent_masters = Vec::new();
        for (ordinal, expected) in parent_spec.expected_masters().iter().enumerate() {
            parent_masters.push(
                parent
                    .install_master(expected, |writer| {
                        write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                    })
                    .expect("install PTQ parent master"),
            );
        }
        parent.seal_complete().expect("seal PTQ parent");
        let allocated = bind_fixture_allocation(&parent);
        let compact = fixture_selected_package(false);
        let near = fixture_selected_package(true);

        let mismatched = allocated
            .admit_packages(Cursor::new(compact.clone()), Cursor::new(compact.clone()))
            .expect_err("NearLosslessV1 must match its selected refinement map");
        assert!(
            matches!(
                mismatched,
                Qwen36PackageAdmissionError::Campaign(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected package present-plane ledger"
                ))
            ),
            "unexpected mismatch error: {mismatched:?}"
        );
        let admitted = allocated
            .admit_packages(Cursor::new(compact), Cursor::new(near))
            .expect("admit exact selected packages");
        assert_eq!(
            admitted.receipt().selection_id(),
            allocated.receipt().selection_id()
        );
        assert_ne!(
            admitted.receipt().compact().package_id(),
            admitted.receipt().near_lossless().package_id()
        );
        assert_eq!(
            admitted
                .receipt()
                .compact()
                .runtime_ledger()
                .present_planes(),
            2
        );
        assert_eq!(
            admitted
                .receipt()
                .near_lossless()
                .runtime_ledger()
                .present_planes(),
            3
        );
        let child_specs = vec![
            fixture_scale_only_master(&parent_masters[0], "language.weight", [21; 32], 0, [51; 32]),
            fixture_scale_only_master(&parent_masters[1], "mtp.weight", [22; 32], 1, [52; 32]),
        ];
        let refinement = admitted
            .open_scale_only_campaign(child_specs)
            .expect("open package-admitted scale-only campaign");
        assert_eq!(
            refinement.package_admission_id(),
            admitted.receipt().admission_id()
        );
        assert_ne!(refinement.campaign_id(), parent.campaign_id());
        drop(refinement);
        admitted.verify_current().expect("reverify admission");
        let admission_receipt = admitted.receipt().clone();
        drop(admitted);
        drop(allocated);

        let allocated = parent
            .reopen_selected_allocation()
            .expect("reopen selection without scavenging package records");
        let admitted = allocated
            .reopen_package_admission()
            .expect("reopen exact package admission");
        assert_eq!(admitted.receipt(), &admission_receipt);

        let selection_store = TensorWorkStore::open(&parent.root().join("selected-allocation"))
            .expect("open selected allocation CAS");
        fs::write(
            selection_store.record_path(admission_receipt.compact().record().record_id()),
            b"changed package record",
        )
        .expect("tamper admitted package record");
        assert!(admitted.verify_current().is_err());

        drop(admitted);
        drop(allocated);
        drop(parent);
        drop(base);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn scale_only_campaign_accepts_new_losses_and_scales_with_fixed_trits() {
        let root = fixture_root("scale-only-fixed-trits");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let parent_spec = fixture_campaign_spec();
        let parent = base
            .open_master_campaign(parent_spec.clone())
            .expect("open PTQ parent campaign");
        let mut parent_masters = Vec::new();
        for (ordinal, expected) in parent_spec.expected_masters().iter().enumerate() {
            parent_masters.push(
                parent
                    .install_master(expected, |writer| {
                        write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                    })
                    .expect("install PTQ parent master"),
            );
        }
        parent.seal_complete().expect("seal PTQ parent");
        let child_specs = vec![
            fixture_scale_only_master(&parent_masters[0], "language.weight", [21; 32], 0, [51; 32]),
            fixture_scale_only_master(&parent_masters[1], "mtp.weight", [22; 32], 1, [52; 32]),
        ];
        let child = parent
            .open_scale_only_campaign(child_specs.clone())
            .expect("open scale-only child");

        for (ordinal, expected) in child_specs.iter().enumerate() {
            let installed = child
                .install_master(expected, |writer| {
                    write_scale_only_fixture_master(
                        expected,
                        writer,
                        if ordinal == 0 { 1 } else { 2 },
                    )
                })
                .expect("install scale-only child master");
            assert_ne!(
                installed.tensor_master_id(),
                parent_masters[ordinal].tensor_master_id()
            );
        }
        assert_eq!(child.progress().unwrap().additive_present(), 2);
        let completion = child.seal_complete().expect("seal scale-only child");
        assert_eq!(completion, child.require_complete().unwrap());

        let first = &child_specs[0];
        fs::remove_file(child.campaign.slot_path(0, first.name()))
            .expect("remove sealed child slot");
        let producer_called = Cell::new(false);
        assert!(matches!(
            child.install_master(first, |writer| {
                producer_called.set(true);
                write_scale_only_fixture_master(first, writer, 1)
            }),
            Err(Qwen36AdditiveInstallError::Campaign(_))
        ));
        assert!(!producer_called.get());

        let mut descriptor = child.campaign.descriptor_bytes.clone();
        descriptor[0] ^= 0xff;
        fs::write(child.campaign.descriptor_path(), descriptor).expect("tamper child descriptor");
        assert!(matches!(
            child.reopen_master("language.weight"),
            Err(Qwen36TensorWorkError::ExistingArtifactMismatch(
                "additive campaign descriptor"
            ))
        ));

        drop(child);
        drop(parent);
        drop(base);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn scale_only_campaign_rejects_changed_trits_and_prefixes_before_publication() {
        let root = fixture_root("scale-only-structure-mismatch");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let parent_spec = fixture_campaign_spec();
        let parent = base
            .open_master_campaign(parent_spec.clone())
            .expect("open PTQ parent campaign");
        let mut parent_masters = Vec::new();
        for (ordinal, expected) in parent_spec.expected_masters().iter().enumerate() {
            parent_masters.push(
                parent
                    .install_master(expected, |writer| {
                        write_fixture_master(expected, writer, if ordinal == 0 { 1 } else { 2 })
                    })
                    .expect("install PTQ parent master"),
            );
        }
        parent.seal_complete().expect("seal PTQ parent");
        let child_specs = vec![
            fixture_scale_only_master(&parent_masters[0], "language.weight", [21; 32], 0, [51; 32]),
            fixture_scale_only_master(&parent_masters[1], "mtp.weight", [22; 32], 1, [52; 32]),
        ];
        let child = parent
            .open_scale_only_campaign(child_specs.clone())
            .expect("open scale-only child");
        let first = &child_specs[0];

        assert!(matches!(
            child.install_master(first, |writer| {
                write_scale_only_fixture_master_with_first_trit(first, writer, 1, 1)
            }),
            Err(Qwen36AdditiveInstallError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("scale-only fixed trits and prefixes")
            ))
        ));
        assert!(matches!(
            child.install_master(first, |writer| {
                write_scale_only_fixture_master_with_first_trits(first, writer, 1, -1, -1)
            }),
            Err(Qwen36AdditiveInstallError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("scale-only fixed trits and prefixes")
            ))
        ));
        assert!(matches!(
            child.install_master(first, |writer| {
                write_scale_only_fixture_master(first, writer, 2)
            }),
            Err(Qwen36AdditiveInstallError::Campaign(
                Qwen36TensorWorkError::WorkspaceMismatch("scale-only fixed trits and prefixes")
            ))
        ));
        assert!(!child.campaign.slot_path(0, first.name()).exists());
        assert_eq!(object_record_count(child.campaign.objects.objects_dir()), 0);
        assert_eq!(
            fs::read_dir(child.campaign.objects.temporary_dir())
                .expect("read temporary directory")
                .count(),
            0
        );
        assert_eq!(child.progress().unwrap().additive_present(), 0);

        let parent_completion_path = parent.completion_path();
        assert!(matches!(
            child.install_master(first, |writer| {
                write_scale_only_fixture_master(first, writer, 1)?;
                fs::write(&parent_completion_path, b"changed parent completion")
                    .expect("tamper parent completion");
                Ok::<(), SaltV2MasterError>(())
            }),
            Err(Qwen36AdditiveInstallError::Campaign(_))
        ));
        assert!(!child.campaign.slot_path(0, first.name()).exists());
        assert_eq!(object_record_count(child.campaign.objects.objects_dir()), 0);
        assert_eq!(
            fs::read_dir(child.campaign.objects.temporary_dir())
                .expect("read temporary directory")
                .count(),
            0
        );

        drop(child);
        drop(parent);
        drop(base);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cross_campaign_and_swapped_slot_receipts_fail_closed() {
        let root = fixture_root("receipt-substitution");
        let _ = fs::remove_dir_all(&root);
        let source = EmptySource;
        let base = Qwen36TensorWorkStore::open_from_parts(&source, root.clone(), fixture_plan())
            .expect("open base workspace");
        base.reconcile_preserved().expect("seal empty base");
        let first_spec = fixture_campaign_spec();
        let first_campaign = base
            .open_master_campaign(first_spec.clone())
            .expect("open first campaign");
        let first_receipt = first_campaign
            .install_master(&first_spec.expected_masters()[0], |writer| {
                write_fixture_master(&first_spec.expected_masters()[0], writer, 1)
            })
            .expect("install first campaign master");
        let receipt_bytes = first_receipt.canonical_bytes().unwrap();

        persist_exact(
            &first_campaign.slot_path(1, first_spec.expected_masters()[1].name()),
            &receipt_bytes,
            "additive master slot",
        )
        .expect("place swapped receipt");
        assert!(matches!(
            first_campaign.reopen_master("mtp.weight"),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master slot binding"
            ))
        ));

        let second_spec = Qwen36AdditiveCampaignSpec::new(vec![
            fixture_master("language.weight", [21; 32], 0, [61; 32]),
            fixture_master("mtp.weight", [22; 32], 1, [52; 32]),
        ])
        .expect("second campaign spec");
        let second_campaign = base
            .open_master_campaign(second_spec.clone())
            .expect("open second campaign");
        persist_exact(
            &second_campaign.slot_path(0, second_spec.expected_masters()[0].name()),
            &receipt_bytes,
            "additive master slot",
        )
        .expect("place cross-campaign receipt");
        assert!(matches!(
            second_campaign.reopen_master("language.weight"),
            Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master slot binding"
            ))
        ));

        drop(second_campaign);
        drop(first_campaign);
        let _ = fs::remove_dir_all(root);
    }
}
