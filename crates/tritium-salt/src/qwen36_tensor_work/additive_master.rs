//! Immutable additive-master campaigns layered over one preserved-source workspace.

use core::{convert::Infallible, fmt};
#[cfg(unix)]
use std::fs::OpenOptions;
use std::{
    error::Error,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use tritium_format::{
    ModelId,
    salt_v2_master::{
        SALT_V2_MASTER_TENSOR_SCHEMA, SaltV2MasterTensorDecoder, SaltV2MasterTensorSpec,
        SaltV2MasterTrack, SaltV2MasterVisitError,
    },
};

#[cfg(unix)]
use crate::tensor_work_store::ensure_durable_directory;
use crate::{
    ContentId, Qwen36SourceIdentityStatus, TensorPayloadWriter, TensorPutError,
    TensorRecordReceipt, TensorRecordSpec, TensorVisitError, TensorWorkStore,
};

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
const SLOT_MAGIC: [u8; 8] = *b"TSQ36AR\0";
const COMPLETION_MAGIC: [u8; 8] = *b"TSQ36CM\0";
const FORMAT_VERSION: u16 = 1;
#[cfg(unix)]
const CAMPAIGN_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 additive campaign checksum v1";
const SLOT_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 additive master receipt checksum v1";
const COMPLETION_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 complete workspace checksum v1";
const MASTER_SET_CONTEXT: &str = "tritium qwen3.6 ordered additive master set v1";
const SLOT_KEY_CONTEXT: &str = "tritium qwen3.6 additive campaign slot key v1";
const MASTER_STREAM_CHUNK_BYTES: usize = 64 * 1024;
const MAX_SLOT_RECEIPT_BYTES: u64 = 512 * 1024;

/// Exact ordered tensor-master metadata that defines one rate-free Qwen PTQ campaign.
///
/// Every tensor-specific curvature, feedback, widened-source, and parent digest
/// is committed before any payload can be installed. Deployment rate and codec
/// are absent because canonical tensor masters are reusable work artifacts.
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
    _lock: CampaignLock,
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
            _lock: lock,
        };
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

impl Qwen36AdditiveCampaignStore<'_, '_> {
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
        self.ensure_current()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let ordinal = self
            .expected_ordinal(spec)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let slot_path = self.slot_path(ordinal, spec.name());
        match fs::symlink_metadata(&slot_path) {
            Ok(_) => {
                let receipt = self
                    .reopen_slot(ordinal, spec)
                    .map_err(Qwen36AdditiveInstallError::Campaign)?;
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
        let record_spec = master_record_spec(spec).map_err(Qwen36AdditiveInstallError::Campaign)?;
        let record = match self.objects.put(&record_spec, produce) {
            Ok(record) => record,
            Err(TensorPutError::Store(error)) => {
                return Err(Qwen36AdditiveInstallError::Campaign(
                    Qwen36TensorWorkError::TensorStore(error),
                ));
            }
            Err(TensorPutError::Producer(error)) => {
                return Err(Qwen36AdditiveInstallError::Producer(error));
            }
        };
        let master = self
            .verify_record(&record, spec)
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let receipt = Qwen36AdditiveMasterReceipt::new(
            self.campaign_id,
            ordinal as u64,
            master.tensor_master_id(),
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
        let receipt = self
            .reopen_slot(ordinal, spec)
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
        self.ensure_current()?;
        let mut present = 0_u64;
        for (ordinal, expected) in self.spec.expected_masters.iter().enumerate() {
            match fs::symlink_metadata(self.slot_path(ordinal, expected.name())) {
                Ok(_) => {
                    self.reopen_slot(ordinal, expected)?;
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
        self.ensure_current()?;
        let mut masters = Vec::new();
        masters
            .try_reserve_exact(self.spec.expected_masters.len())
            .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
        let mut present = 0_u64;
        for (ordinal, expected) in self.spec.expected_masters.iter().enumerate() {
            match fs::symlink_metadata(self.slot_path(ordinal, expected.name())) {
                Ok(_) => {
                    masters.push(self.reopen_slot(ordinal, expected)?);
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
        persist_exact(
            &self.completion_path(),
            &bytes,
            "complete additive workspace",
        )?;
        self.require_complete()
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
        self.ensure_current()?;
        match fs::symlink_metadata(self.completion_path()) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let summary = self.progress()?;
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
        for (ordinal, (sealed, expected)) in manifest
            .masters
            .iter()
            .zip(&self.spec.expected_masters)
            .enumerate()
        {
            let current = self.reopen_slot(ordinal, expected)?;
            if current != *sealed {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "completion master receipt",
                ));
            }
            reopened.push(current);
        }
        if derive_master_set_id(&reopened)? != manifest.master_set_id {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "ordered master set identity",
            ));
        }
        self.ensure_current()?;
        manifest.receipt(&bytes)
    }

    fn descriptor_path(&self) -> PathBuf {
        self.root.join(CAMPAIGN_FILE)
    }

    fn completion_path(&self) -> PathBuf {
        self.root.join(COMPLETION_FILE)
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
        let master = self.verify_record(&receipt.record, expected)?;
        if master.tensor_master_id() != receipt.tensor_master_id {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive tensor master identity",
            ));
        }
        Ok(receipt)
    }

    fn verify_record(
        &self,
        record: &TensorRecordReceipt,
        expected: &SaltV2MasterTensorSpec,
    ) -> Result<tritium_format::salt_v2_master::SaltV2MasterTensorReceipt, Qwen36TensorWorkError>
    {
        let inner = SaltV2MasterTensorSpec::from_canonical_bytes(record.info().schema_metadata())
            .map_err(Qwen36TensorWorkError::Master)?;
        let record_spec = master_record_spec(expected)?;
        if inner != *expected || !record.matches_spec(&record_spec) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master record descriptor",
            ));
        }
        let mut reader = self
            .objects
            .open_verified(record)
            .map_err(Qwen36TensorWorkError::TensorStore)?;
        let mut decoder =
            SaltV2MasterTensorDecoder::new(expected).map_err(Qwen36TensorWorkError::Master)?;
        reader
            .try_visit_payload(MASTER_STREAM_CHUNK_BYTES, |chunk| {
                decoder.try_push(chunk, &mut |_| Ok::<(), Infallible>(()))
            })
            .map_err(map_master_visit_error)?;
        let master = decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
        if master.payload_bytes() != expected.payload_bytes()
            || master.tile_count() != expected.tile_count() as u64
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "additive master payload geometry",
            ));
        }
        Ok(master)
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

fn map_master_visit_error(
    error: TensorVisitError<SaltV2MasterVisitError<Infallible>>,
) -> Qwen36TensorWorkError {
    match error {
        TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
        TensorVisitError::Sink(SaltV2MasterVisitError::Master(error)) => {
            Qwen36TensorWorkError::Master(error)
        }
        TensorVisitError::Sink(SaltV2MasterVisitError::Visitor(impossible)) => match impossible {},
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
    let first = &masters[0];
    let common_evidence = first.evidence();
    let common_geometry = first.geometry();
    let common_model = first.source_model_id();
    if common_evidence.track != SaltV2MasterTrack::Ptq {
        return Err(Qwen36TensorWorkError::RefinedCampaignRequiresParent);
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
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&bytes);
    }
    if output.len() > MAX_WORKSPACE_BYTES {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "master catalog too large",
        ));
    }
    Ok(output)
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
        io::{self, Write},
    };

    use half::f16;
    use tritium_format::{
        SemanticTensorHasher,
        salt_v2_master::{
            SaltV2FitConstraint, SaltV2MasterError, SaltV2MasterEvidence, SaltV2MasterGeometry,
            SaltV2MasterTensorEncoder, SaltV2MasterTrack, SaltV2PrefixLoss,
        },
        salt_v2_package::SaltV2Plane,
    };
    use tritium_nn::{NnError, Qwen35TensorStreamError};
    use tritium_quantize::{Qwen35SourceDtype, Qwen35TensorRole, Qwen35TensorScope};

    use super::super::{
        PreservedTensorSource, Qwen36AdditiveSlotState, Qwen36AdditiveWorkSlot, WorkspacePlan,
    };
    use super::*;

    #[derive(Debug)]
    struct EmptySource;

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

    fn fixture_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-qwen36-additive-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
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
