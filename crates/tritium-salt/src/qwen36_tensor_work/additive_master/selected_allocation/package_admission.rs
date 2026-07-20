//! Exact SALT V2 package admission over a durable nested allocation.

use core::fmt;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use tritium_format::salt_v2_master::{SaltV2MasterTensorDecoder, SaltV2MasterVisitError};
use tritium_format::salt_v2_package::{
    SaltV2IndexedRuntimeLedger, SaltV2PackageLedger, SaltV2PackageReadError, SaltV2PackageReader,
    SaltV2PackageStreamError, SaltV2PackageStreamPlan, SaltV2PackageStreamPlanError,
    SaltV2PackageStreamWriter, SaltV2SemanticTensorStream, SaltV2StreamTensorSpec, SaltV2Transform,
};
use tritium_format::{PackageHasher, PackageId};
use tritium_quantize::{PhysicalBytes, SaltV2Profile};

use crate::{
    ContentId, TensorPayloadWriter, TensorPutError, TensorRecordReceipt, TensorRecordSpec,
    TensorVisitError, TensorWorkStore, tensor_work_store::create_temporary_file,
};

use super::super::{
    CHECKSUM_BYTES, CanonicalCursor, FixedCampaignMode, MASTER_STREAM_CHUNK_BYTES,
    PackageScaleOnlyBinding, PinnedDirectory, Qwen36AdditiveCampaignSpec,
    Qwen36AdditiveInstallError, Qwen36AdditiveMasterReceipt, Qwen36CompleteWorkspaceReceipt,
    Qwen36ScaleOnlyCampaignStore, SaltV2MasterTensorSpec, same_file_identity,
};
use super::{
    Qwen36AllocatedCampaignStore, Qwen36SelectedAllocationReceipt, Qwen36TensorWorkError,
    codec_tag, persist_exact, read_regular_bounded, read_selection_receipt,
    reclaim_selection_orphans, stage_verified_map, validate_directories, work_io,
};

const ADMISSION_FILE: &str = "package-admission.tq36p";
const ADMISSION_MAGIC: [u8; 8] = *b"TSQ36PK\0";
const FORMAT_VERSION: u16 = 1;
const CHECKSUM_CONTEXT: &str = "tritium qwen3.6 selected package admission checksum v1";
const PACKAGE_SCHEMA: &[u8] = b"tritium qwen3.6 exact selected SALT V2 package record v1";
const PACKAGE_METADATA_MAGIC: [u8; 8] = *b"TSQ36PR\0";
const PACKAGE_CHUNK_BYTES: usize = 64 * 1024;
const MAX_ADMISSION_BYTES: u64 = 512 * 1024;

/// Failure while staging or admitting exact selected SALT V2 packages.
#[derive(Debug)]
pub enum Qwen36PackageAdmissionError {
    /// Parent campaign, selection, CAS, or durable manifest validation failed.
    Campaign(Qwen36TensorWorkError),
    /// A staged profile package was malformed, changed, or unreadable.
    Package {
        /// Profile whose exact package failed.
        profile: SaltV2Profile,
        /// Strict package-reader failure.
        error: SaltV2PackageReadError,
    },
    /// Caller-provided package input failed while being staged.
    Source {
        /// Profile whose source failed.
        profile: SaltV2Profile,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
    /// Exact selected-prefix materialization failed before admission publication.
    Materialization {
        /// Profile whose canonical package could not be produced.
        profile: SaltV2Profile,
        /// Typed seek-writer failure.
        error: SaltV2PackageStreamError,
    },
}

impl fmt::Display for Qwen36PackageAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Campaign(error) => write!(formatter, "package admission failed: {error}"),
            Self::Package { profile, error } => {
                write!(formatter, "{profile:?} package admission failed: {error}")
            }
            Self::Source { profile, kind } => {
                write!(formatter, "{profile:?} package source failed: {kind}")
            }
            Self::Materialization { profile, error } => {
                write!(
                    formatter,
                    "{profile:?} package materialization failed: {error}"
                )
            }
        }
    }
}

impl Error for Qwen36PackageAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Campaign(error) => Some(error),
            Self::Package { error, .. } => Some(error),
            Self::Source { .. } => None,
            Self::Materialization { error, .. } => Some(error),
        }
    }
}

impl From<Qwen36TensorWorkError> for Qwen36PackageAdmissionError {
    fn from(error: Qwen36TensorWorkError) -> Self {
        Self::Campaign(error)
    }
}

/// Failure while visiting one verified admitted package payload.
#[derive(Debug)]
pub enum Qwen36PackageVisitError<E> {
    /// Admission, lineage, record, or filesystem verification failed.
    Admission(Qwen36PackageAdmissionError),
    /// The caller's bounded-chunk sink failed.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36PackageVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "admitted package visit failed: {error}"),
            Self::Sink(error) => write!(formatter, "admitted package sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36PackageVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

/// Exact aggregate indexed-runtime counters admitted for one package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen36PackageRuntimeLedger {
    payload_bytes: u64,
    scale_bytes: u64,
    allocation_map_bytes: u64,
    rank_prefix_bytes: u64,
    allocation_map_bits: u64,
    allocation_map_embedded_bits: u64,
    dense_shadow_bytes: u64,
    allocation_tiles: u64,
    present_planes: u64,
    steady_resident_bytes: u64,
}

impl Qwen36PackageRuntimeLedger {
    fn from_runtime(value: SaltV2IndexedRuntimeLedger) -> Self {
        Self {
            payload_bytes: value.payload_bytes(),
            scale_bytes: value.scale_bytes(),
            allocation_map_bytes: value.allocation_map_bytes(),
            rank_prefix_bytes: value.rank_prefix_bytes(),
            allocation_map_bits: value.allocation_map_bits(),
            allocation_map_embedded_bits: value.allocation_map_embedded_bits(),
            dense_shadow_bytes: value.dense_shadow_bytes(),
            allocation_tiles: value.allocation_tiles(),
            present_planes: value.present_planes(),
            steady_resident_bytes: value.steady_resident_bytes(),
        }
    }

    /// Encoded ternary payload bytes.
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }
    /// Group128 f16 scale bytes.
    pub const fn scale_bytes(self) -> u64 {
        self.scale_bytes
    }
    /// Allocated two-bit plane-count map bytes.
    pub const fn allocation_map_bytes(self) -> u64 {
        self.allocation_map_bytes
    }
    /// Coarse rank-prefix bytes.
    pub const fn rank_prefix_bytes(self) -> u64 {
        self.rank_prefix_bytes
    }
    /// Logical allocation-map bits.
    pub const fn allocation_map_bits(self) -> u64 {
        self.allocation_map_bits
    }
    /// Map-tail bits carried in runtime scalars.
    pub const fn allocation_map_embedded_bits(self) -> u64 {
        self.allocation_map_embedded_bits
    }
    /// Dense reconstructed shadow bytes, required to remain zero.
    pub const fn dense_shadow_bytes(self) -> u64 {
        self.dense_shadow_bytes
    }
    /// Total allocation tiles.
    pub const fn allocation_tiles(self) -> u64 {
        self.allocation_tiles
    }
    /// Total physically present planes.
    pub const fn present_planes(self) -> u64 {
        self.present_planes
    }
    /// Exact steady device-resident bytes.
    pub const fn steady_resident_bytes(self) -> u64 {
        self.steady_resident_bytes
    }
}

/// Exact package, CAS record, and measured ledgers for one selected profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PackageProfileReceipt {
    package_id: PackageId,
    record: TensorRecordReceipt,
    package_ledger: SaltV2PackageLedger,
    runtime_ledger: Qwen36PackageRuntimeLedger,
}

impl Qwen36PackageProfileReceipt {
    /// Identity of exact canonical package bytes.
    pub const fn package_id(&self) -> PackageId {
        self.package_id
    }
    /// CAS receipt whose payload is the exact package.
    pub const fn record(&self) -> &TensorRecordReceipt {
        &self.record
    }
    /// Exact package-file component ledger.
    pub const fn package_ledger(&self) -> SaltV2PackageLedger {
        self.package_ledger
    }
    /// Exact aggregate indexed-runtime ledger.
    pub const fn runtime_ledger(&self) -> Qwen36PackageRuntimeLedger {
        self.runtime_ledger
    }
    /// Exact measured serialized and steady-resident bytes.
    pub const fn physical_bytes(&self) -> PhysicalBytes {
        PhysicalBytes {
            serialized: self.package_ledger.total_bytes,
            resident: self.runtime_ledger.steady_resident_bytes,
        }
    }
}

/// Durable proof that both selected profiles materialized as exact admissible packages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PackageAdmissionReceipt {
    admission_id: ContentId,
    selection_id: ContentId,
    parent_completion_id: ContentId,
    allocation_spec_id: ContentId,
    nested_allocation_id: ContentId,
    compact_map_id: ContentId,
    compact_loss_id: ContentId,
    near_map_id: ContentId,
    near_loss_id: ContentId,
    compact: Qwen36PackageProfileReceipt,
    near_lossless: Qwen36PackageProfileReceipt,
}

impl Qwen36PackageAdmissionReceipt {
    /// Identity of the complete canonical admission receipt.
    pub const fn admission_id(&self) -> ContentId {
        self.admission_id
    }
    /// Exact selected allocation admitted by this receipt.
    pub const fn selection_id(&self) -> ContentId {
        self.selection_id
    }
    /// Exact sealed parent completion transitively bound by the selection.
    pub const fn parent_completion_id(&self) -> ContentId {
        self.parent_completion_id
    }
    /// CompactV1 exact package receipt.
    pub const fn compact(&self) -> &Qwen36PackageProfileReceipt {
        &self.compact
    }
    /// NearLosslessV1 exact package receipt.
    pub const fn near_lossless(&self) -> &Qwen36PackageProfileReceipt {
        &self.near_lossless
    }

    fn from_selection(
        selection: &Qwen36SelectedAllocationReceipt,
        compact: Qwen36PackageProfileReceipt,
        near_lossless: Qwen36PackageProfileReceipt,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let mut receipt = Self {
            admission_id: ContentId::from_digest([0; 32]),
            selection_id: selection.selection_id,
            parent_completion_id: selection.parent_completion_id,
            allocation_spec_id: selection.spec.spec_id(),
            nested_allocation_id: selection.nested_allocation_id,
            compact_map_id: selection.compact.allocation_map_id,
            compact_loss_id: selection.compact.selected_loss_id,
            near_map_id: selection.near_lossless.allocation_map_id,
            near_loss_id: selection.near_lossless.selected_loss_id,
            compact,
            near_lossless,
        };
        let bytes = receipt.canonical_bytes()?;
        receipt.admission_id = ContentId::of_bytes(&bytes);
        Ok(receipt)
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        let mut output = Vec::new();
        output.extend_from_slice(&ADMISSION_MAGIC);
        output.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        for id in [
            self.selection_id,
            self.parent_completion_id,
            self.allocation_spec_id,
            self.nested_allocation_id,
            self.compact_map_id,
            self.compact_loss_id,
            self.near_map_id,
            self.near_loss_id,
        ] {
            output.extend_from_slice(id.as_bytes());
        }
        encode_profile(&mut output, &self.compact)?;
        encode_profile(&mut output, &self.near_lossless)?;
        let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() as u64 > MAX_ADMISSION_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "package admission size",
            ));
        }
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if bytes.len() as u64 > MAX_ADMISSION_BYTES
            || bytes.len() < ADMISSION_MAGIC.len() + CHECKSUM_BYTES
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "package admission length",
            ));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "package admission checksum",
            ));
        }
        let mut cursor = CanonicalCursor::new(payload);
        if cursor.take(ADMISSION_MAGIC.len())? != ADMISSION_MAGIC
            || cursor.u16()? != FORMAT_VERSION
            || cursor.u16()? != 0
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "package admission header",
            ));
        }
        let selection_id = ContentId::from_digest(cursor.digest()?);
        let parent_completion_id = ContentId::from_digest(cursor.digest()?);
        let allocation_spec_id = ContentId::from_digest(cursor.digest()?);
        let nested_allocation_id = ContentId::from_digest(cursor.digest()?);
        let compact_map_id = ContentId::from_digest(cursor.digest()?);
        let compact_loss_id = ContentId::from_digest(cursor.digest()?);
        let near_map_id = ContentId::from_digest(cursor.digest()?);
        let near_loss_id = ContentId::from_digest(cursor.digest()?);
        let compact = decode_profile(&mut cursor)?;
        let near_lossless = decode_profile(&mut cursor)?;
        if cursor.remaining() != 0 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "package admission trailing bytes",
            ));
        }
        let receipt = Self {
            admission_id: ContentId::of_bytes(bytes),
            selection_id,
            parent_completion_id,
            allocation_spec_id,
            nested_allocation_id,
            compact_map_id,
            compact_loss_id,
            near_map_id,
            near_loss_id,
            compact,
            near_lossless,
        };
        if receipt.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical package admission",
            ));
        }
        Ok(receipt)
    }
}

/// Typed capability proving selected maps, parent prefixes, exact packages, and budgets.
#[derive(Debug)]
pub struct Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source> {
    allocated: &'allocated Qwen36AllocatedCampaignStore<'parent, 'store, 'source>,
    receipt: Qwen36PackageAdmissionReceipt,
    directories: Vec<PinnedDirectory>,
    package_records: [PinnedPackageRecord; 2],
}

/// Package-bound scale-only campaign; refinement cannot outlive its admitted packages.
#[derive(Debug)]
pub struct Qwen36PackageScaleOnlyCampaignStore<'admission, 'allocated, 'parent, 'store, 'source> {
    admission: &'admission Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
    campaign: Qwen36ScaleOnlyCampaignStore<'parent, 'store, 'source>,
}

#[derive(Debug)]
struct PinnedPackageRecord {
    path: PathBuf,
    identity: fs::Metadata,
}

impl PinnedPackageRecord {
    fn pin(path: PathBuf, expected_bytes: u64) -> Result<Self, Qwen36TensorWorkError> {
        let identity = fs::symlink_metadata(&path)
            .map_err(|error| work_io("pin admitted package record", error))?;
        if identity.file_type().is_symlink()
            || !identity.is_file()
            || identity.len() != expected_bytes
        {
            return Err(Qwen36TensorWorkError::InvalidPath(
                "admitted package record",
            ));
        }
        Ok(Self { path, identity })
    }

    fn validate(&self) -> Result<(), Qwen36TensorWorkError> {
        let current = fs::symlink_metadata(&self.path)
            .map_err(|error| work_io("reinspect admitted package record", error))?;
        if current.file_type().is_symlink()
            || !current.is_file()
            || !same_file_identity(&self.identity, &current)
            || !same_file_version(&self.identity, &current)
        {
            return Err(Qwen36TensorWorkError::InvalidPath(
                "changed admitted package record",
            ));
        }
        Ok(())
    }
}

impl Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_> {
    /// Durable exact package-admission receipt carried by this capability.
    pub const fn receipt(&self) -> &Qwen36PackageAdmissionReceipt {
        &self.receipt
    }

    /// Strictly revalidate selection lineage, package CAS records, semantics, and budgets.
    pub fn verify_current(&self) -> Result<(), Qwen36PackageAdmissionError> {
        verify_admission(self.allocated, &self.receipt)
    }

    /// Visit one exact admitted package in bounded verified chunks.
    ///
    /// Callback effects are nontransactional: callers implementing export must
    /// stage output and publish only after this method returns successfully.
    pub fn try_visit_package<E>(
        &self,
        profile: SaltV2Profile,
        max_chunk_bytes: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, Qwen36PackageVisitError<E>> {
        self.verify_current()
            .map_err(Qwen36PackageVisitError::Admission)?;
        let (_, objects, _) = self
            .allocated
            .parent
            .open_selection_store()
            .map_err(Qwen36PackageAdmissionError::from)
            .map_err(Qwen36PackageVisitError::Admission)?;
        let selected = match profile {
            SaltV2Profile::CompactV1 => self.receipt.compact(),
            SaltV2Profile::NearLosslessV1 => self.receipt.near_lossless(),
        };
        objects
            .try_visit_verified(selected.record(), max_chunk_bytes, |chunk| visit(chunk))
            .map_err(|error| match error {
                TensorVisitError::Store(error) => {
                    Qwen36PackageVisitError::Admission(Qwen36PackageAdmissionError::Campaign(
                        Qwen36TensorWorkError::TensorStore(error),
                    ))
                }
                TensorVisitError::Sink(error) => Qwen36PackageVisitError::Sink(error),
            })?;
        self.verify_current()
            .map_err(Qwen36PackageVisitError::Admission)?;
        Ok(selected.package_ledger().total_bytes)
    }

    fn verify_cheap_current(&self) -> Result<(), Qwen36PackageAdmissionError> {
        validate_directories(&self.directories)?;
        for record in &self.package_records {
            record.validate()?;
        }
        let root = self.allocated.parent.root.join(super::SELECTION_DIRECTORY);
        let current = read_admission(&root.join(ADMISSION_FILE))?;
        if current != self.receipt
            || read_selection_receipt(&root.join(super::SELECTION_FILE))? != self.allocated.receipt
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "package admission cheap receipt check",
            )
            .into());
        }
        self.allocated
            .parent
            .verify_completion_receipt(&self.allocated.parent_completion)?;
        validate_directories(&self.directories)?;
        for record in &self.package_records {
            record.validate()?;
        }
        Ok(())
    }
}

impl<'allocated, 'parent, 'store, 'source>
    Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>
{
    /// Open a v2 scale-only campaign whose descriptor binds admission and package IDs.
    pub fn open_scale_only_campaign<'admission>(
        &'admission self,
        expected_masters: Vec<SaltV2MasterTensorSpec>,
    ) -> Result<
        Qwen36PackageScaleOnlyCampaignStore<'admission, 'allocated, 'parent, 'store, 'source>,
        Qwen36PackageAdmissionError,
    > {
        self.verify_current()?;
        let parent = self.allocated.parent;
        let (parent_completion, parent_manifest, parent_fixed_ids) =
            parent.require_complete_verified(FixedCampaignMode::Capture)?;
        if parent_completion != self.allocated.parent_completion {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "package-admitted scale-only parent completion",
            )
            .into());
        }
        let spec = Qwen36AdditiveCampaignSpec::new_package_admitted_scale_only(
            &parent_completion,
            &parent.spec.expected_masters,
            &parent_manifest.masters,
            &parent_fixed_ids,
            PackageScaleOnlyBinding {
                admission_id: self.receipt.admission_id,
                selection_id: self.receipt.selection_id,
                compact_package_id: *self.receipt.compact.package_id.as_bytes(),
                near_package_id: *self.receipt.near_lossless.package_id.as_bytes(),
            },
            expected_masters,
        )?;
        let campaign = parent.base.open_additive_campaign(spec)?;
        let campaign = Qwen36ScaleOnlyCampaignStore {
            parent,
            campaign,
            parent_completion,
            parent_masters: parent_manifest.masters,
            parent_fixed_ids,
        };
        campaign.verify_parent_campaign()?;
        self.verify_current()?;
        Ok(Qwen36PackageScaleOnlyCampaignStore {
            admission: self,
            campaign,
        })
    }
}

impl Qwen36PackageScaleOnlyCampaignStore<'_, '_, '_, '_, '_> {
    /// Package-admission identity bound into this refinement campaign.
    pub const fn package_admission_id(&self) -> ContentId {
        self.admission.receipt.admission_id
    }

    /// Content identity of the package-bound child campaign descriptor.
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign.campaign_id()
    }

    /// Exact sealed PTQ completion admitted as this campaign's parent.
    pub const fn parent_completion_id(&self) -> ContentId {
        self.campaign.parent_completion_id()
    }

    /// Exact ordered scale-only master specification.
    pub const fn spec(&self) -> &Qwen36AdditiveCampaignSpec {
        self.campaign.spec()
    }

    /// Immutable package-bound child campaign root.
    pub fn root(&self) -> &Path {
        self.campaign.root()
    }

    /// Install one fixed-trit scale-only master while the admission remains current.
    pub fn install_master<E>(
        &self,
        spec: &SaltV2MasterTensorSpec,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36AdditiveInstallError<E>> {
        self.verify_admission_for_refinement()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        let receipt = self.campaign.install_master(spec, produce)?;
        self.verify_admission_for_refinement()
            .map_err(Qwen36AdditiveInstallError::Campaign)?;
        Ok(receipt)
    }

    /// Strictly reopen one child master while package admission remains current.
    pub fn reopen_master(
        &self,
        name: &str,
    ) -> Result<Qwen36AdditiveMasterReceipt, Qwen36TensorWorkError> {
        self.verify_admission_for_refinement()?;
        let receipt = self.campaign.reopen_master(name)?;
        self.verify_admission_for_refinement()?;
        Ok(receipt)
    }

    /// Recompute child progress under the package-admission gate.
    pub fn progress(&self) -> Result<super::super::Qwen36TensorWorkSummary, Qwen36TensorWorkError> {
        self.verify_admission_for_refinement()?;
        let summary = self.campaign.progress()?;
        self.verify_admission_for_refinement()?;
        Ok(summary)
    }

    /// Seal only after full terminal package-admission revalidation.
    pub fn seal_complete(&self) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.verify_admission_full_for_refinement()?;
        let receipt = self.campaign.seal_complete()?;
        self.verify_admission_full_for_refinement()?;
        Ok(receipt)
    }

    /// Strictly reopen the completed package-bound child campaign.
    pub fn require_complete(
        &self,
    ) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError> {
        self.verify_admission_full_for_refinement()?;
        let receipt = self.campaign.require_complete()?;
        self.verify_admission_full_for_refinement()?;
        Ok(receipt)
    }

    fn verify_admission_for_refinement(&self) -> Result<(), Qwen36TensorWorkError> {
        self.admission.verify_cheap_current().map_err(|_| {
            Qwen36TensorWorkError::WorkspaceMismatch(
                "package-admitted scale-only package admission",
            )
        })
    }

    fn verify_admission_full_for_refinement(&self) -> Result<(), Qwen36TensorWorkError> {
        self.admission.verify_current().map_err(|_| {
            Qwen36TensorWorkError::WorkspaceMismatch(
                "package-admitted scale-only package admission",
            )
        })
    }
}

impl<'parent, 'store, 'source> Qwen36AllocatedCampaignStore<'parent, 'store, 'source> {
    /// Stage, validate, CAS-publish, and durably bind both exact selected packages.
    ///
    /// Input is pulled in bounded chunks and limited by each profile's hard
    /// serialized-byte ceiling. Publication occurs only after exact tensor order,
    /// geometry, selected maps, parent-prefix semantics, package ledgers, runtime
    /// ledgers, and both budget dimensions pass.
    pub fn admit_packages<'allocated>(
        &'allocated self,
        compact_source: impl Read,
        near_lossless_source: impl Read,
    ) -> Result<
        Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
        Qwen36PackageAdmissionError,
    > {
        #[cfg(not(unix))]
        {
            let _ = (compact_source, near_lossless_source);
            return Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform.into());
        }
        #[cfg(unix)]
        {
            let _mutation = self.parent.begin_mutation()?;
            self.verify_current()?;
            let (root, objects, directories) = self.parent.open_selection_store()?;
            reclaim_selection_orphans(
                &root,
                &objects,
                Some((&self.parent_completion, self.receipt.campaign_id)),
            )?;
            let budgets = self.receipt.spec.budgets();
            let compact_staged = StagedPackage::from_source(
                objects.temporary_dir(),
                "compact.package",
                compact_source,
                budgets.compact.maximum.serialized,
                SaltV2Profile::CompactV1,
            )?;
            let near_staged = StagedPackage::from_source(
                objects.temporary_dir(),
                "near.package",
                near_lossless_source,
                budgets.near_lossless.maximum.serialized,
                SaltV2Profile::NearLosslessV1,
            )?;
            self.admit_staged_packages(&root, &objects, directories, compact_staged, near_staged)
        }
    }

    /// Materialize both selected parent prefixes and admit their exact packages.
    ///
    /// The producer reuses one decoded Pmax master tile for both profiles and
    /// writes directly into seek-backed temporary package files. Retained memory
    /// is the two canonical two-bit maps, `O(tensors)` layout metadata, and one
    /// decoded master tile; no whole-model semantic package or second source copy
    /// is constructed before CAS publication.
    pub fn materialize_and_admit_packages<'allocated>(
        &'allocated self,
    ) -> Result<
        Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
        Qwen36PackageAdmissionError,
    > {
        #[cfg(not(unix))]
        {
            return Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform.into());
        }
        #[cfg(unix)]
        {
            let _mutation = self.parent.begin_mutation()?;
            self.verify_current()?;
            let (root, objects, directories) = self.parent.open_selection_store()?;
            reclaim_selection_orphans(
                &root,
                &objects,
                Some((&self.parent_completion, self.receipt.campaign_id)),
            )?;
            let (completion, manifest, _) = self
                .parent
                .require_complete_verified(FixedCampaignMode::Capture)?;
            if completion != self.parent_completion
                || manifest.masters.len() != self.parent.spec.expected_masters.len()
            {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "package materialization parent",
                )
                .into());
            }

            let mut stream_specs = Vec::new();
            stream_specs
                .try_reserve_exact(self.parent.spec.expected_masters.len())
                .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
            for master in &self.parent.spec.expected_masters {
                stream_specs.push(
                    SaltV2StreamTensorSpec::new(
                        master.name(),
                        master.shape().to_vec(),
                        SaltV2Transform::None,
                    )
                    .map_err(|error| {
                        materialization_package_error(SaltV2Profile::CompactV1, error)
                    })?,
                );
            }

            let mut compact_map = stage_verified_map(
                &objects,
                &self.receipt.compact.map_record,
                "compact.materialize.map",
            )?;
            let mut near_map = stage_verified_map(
                &objects,
                &self.receipt.near_lossless.map_record,
                "near.materialize.map",
            )?;
            let compact_plan = streamed_profile_plan(
                self.receipt.spec.codec(),
                stream_specs.clone(),
                &mut compact_map,
                self.receipt.tile_count,
                SaltV2Profile::CompactV1,
            )?;
            let near_plan = streamed_profile_plan(
                self.receipt.spec.codec(),
                stream_specs,
                &mut near_map,
                self.receipt.tile_count,
                SaltV2Profile::NearLosslessV1,
            )?;
            let budgets = self.receipt.spec.budgets();
            if compact_plan.ledger().total_bytes > budgets.compact.maximum.serialized
                || near_plan.ledger().total_bytes > budgets.near_lossless.maximum.serialized
            {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "materialized package serialized ceiling",
                )
                .into());
            }

            let mut compact_staged = StagedPackage::empty(
                objects.temporary_dir(),
                "compact.materialized.package",
                SaltV2Profile::CompactV1,
            )?;
            let mut near_staged = StagedPackage::empty(
                objects.temporary_dir(),
                "near.materialized.package",
                SaltV2Profile::NearLosslessV1,
            )?;
            let compact_output = compact_staged.clone_file(SaltV2Profile::CompactV1)?;
            let near_output = near_staged.clone_file(SaltV2Profile::NearLosslessV1)?;
            let mut compact_writer =
                SaltV2PackageStreamWriter::new(compact_output, compact_plan)
                    .map_err(|error| materialization_error(SaltV2Profile::CompactV1, error))?;
            let mut near_writer = SaltV2PackageStreamWriter::new(near_output, near_plan)
                .map_err(|error| materialization_error(SaltV2Profile::NearLosslessV1, error))?;
            let mut compact_counts = compact_map.cursor(self.receipt.tile_count)?;
            let mut near_counts = near_map.cursor(self.receipt.tile_count)?;

            for (master, parent_master) in self
                .parent
                .spec
                .expected_masters
                .iter()
                .zip(&manifest.masters)
            {
                let mut decoder = SaltV2MasterTensorDecoder::new(master)
                    .map_err(Qwen36TensorWorkError::Master)?;
                let mut visited = 0usize;
                self.parent
                    .objects
                    .try_visit_verified(&parent_master.record, MASTER_STREAM_CHUNK_BYTES, |chunk| {
                        decoder
                            .try_push(chunk, &mut |tile| {
                                let compact = compact_counts.next_count()?.ok_or(
                                    Qwen36TensorWorkError::WorkspaceMismatch(
                                        "CompactV1 materialization map is short",
                                    ),
                                )?;
                                let near = near_counts.next_count()?.ok_or(
                                    Qwen36TensorWorkError::WorkspaceMismatch(
                                        "NearLosslessV1 materialization map is short",
                                    ),
                                )?;
                                if compact > near
                                    || near > tile.admissible_planes()
                                    || usize::from(near) > tile.planes().len()
                                {
                                    return Err(Qwen36PackageAdmissionError::Campaign(
                                        Qwen36TensorWorkError::WorkspaceMismatch(
                                            "materialized selected plane prefix",
                                        ),
                                    ));
                                }
                                compact_writer
                                    .push_planes(&tile.planes()[..usize::from(compact)])
                                    .map_err(|error| {
                                        materialization_error(SaltV2Profile::CompactV1, error)
                                    })?;
                                near_writer
                                    .push_planes(&tile.planes()[..usize::from(near)])
                                    .map_err(|error| {
                                        materialization_error(SaltV2Profile::NearLosslessV1, error)
                                    })?;
                                visited += 1;
                                Ok(())
                            })
                            .map_err(|error| match error {
                                SaltV2MasterVisitError::Master(error) => {
                                    Qwen36PackageAdmissionError::Campaign(
                                        Qwen36TensorWorkError::Master(error),
                                    )
                                }
                                SaltV2MasterVisitError::Visitor(error) => error,
                            })
                    })
                    .map_err(|error| match error {
                        TensorVisitError::Store(error) => Qwen36PackageAdmissionError::Campaign(
                            Qwen36TensorWorkError::TensorStore(error),
                        ),
                        TensorVisitError::Sink(error) => error,
                    })?;
                let decoded = decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
                if visited != master.tile_count()
                    || decoded.tensor_master_id() != parent_master.tensor_master_id()
                {
                    return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                        "materialized package parent master",
                    )
                    .into());
                }
            }
            if compact_counts.next_count()?.is_some() || near_counts.next_count()?.is_some() {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "materialized package map coverage",
                )
                .into());
            }
            compact_counts.finish()?;
            near_counts.finish()?;
            let (compact_output, compact_ledger) = compact_writer
                .finish()
                .map_err(|error| materialization_error(SaltV2Profile::CompactV1, error))?;
            let (near_output, near_ledger) = near_writer
                .finish()
                .map_err(|error| materialization_error(SaltV2Profile::NearLosslessV1, error))?;
            drop((compact_output, near_output));
            compact_staged
                .finish_materialized(compact_ledger.total_bytes, SaltV2Profile::CompactV1)?;
            near_staged
                .finish_materialized(near_ledger.total_bytes, SaltV2Profile::NearLosslessV1)?;
            self.parent.verify_completion_receipt(&completion)?;
            self.admit_staged_packages(&root, &objects, directories, compact_staged, near_staged)
        }
    }

    #[cfg(unix)]
    fn admit_staged_packages<'allocated>(
        &'allocated self,
        root: &Path,
        objects: &TensorWorkStore,
        directories: Vec<PinnedDirectory>,
        mut compact_staged: StagedPackage,
        mut near_staged: StagedPackage,
    ) -> Result<
        Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
        Qwen36PackageAdmissionError,
    > {
        let mut compact_reader = compact_staged.strict_reader(SaltV2Profile::CompactV1)?;
        let mut near_reader = near_staged.strict_reader(SaltV2Profile::NearLosslessV1)?;
        validate_package_pair(self, objects, &mut compact_reader, &mut near_reader)?;
        compact_reader
            .verify_unchanged()
            .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
        near_reader
            .verify_unchanged()
            .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;

        let compact_profile = publish_staged_profile(
            objects,
            &self.receipt,
            SaltV2Profile::CompactV1,
            &mut compact_staged,
            &compact_reader,
        )?;
        drop(compact_reader);
        drop(compact_staged);
        let near_profile = publish_staged_profile(
            objects,
            &self.receipt,
            SaltV2Profile::NearLosslessV1,
            &mut near_staged,
            &near_reader,
        )?;
        drop(near_reader);
        drop(near_staged);
        let receipt = Qwen36PackageAdmissionReceipt::from_selection(
            &self.receipt,
            compact_profile,
            near_profile,
        )?;
        validate_admission_binding(&receipt, &self.receipt)?;
        verify_admission_records(self, objects, &receipt)?;
        self.verify_current()?;
        let bytes = receipt.canonical_bytes()?;
        persist_exact(
            &root.join(ADMISSION_FILE),
            &bytes,
            "selected package admission",
        )?;
        let current = read_admission(&root.join(ADMISSION_FILE))?;
        if current != receipt {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("package admission receipt").into(),
            );
        }
        verify_admission_records(self, objects, &receipt)?;
        validate_directories(&directories)?;
        let package_records = pin_package_records(objects, &receipt)?;
        Ok(Qwen36PackageAdmittedCampaignStore {
            allocated: self,
            receipt,
            directories,
            package_records,
        })
    }

    /// Strictly reopen the already-published exact selected package admission.
    pub fn reopen_package_admission<'allocated>(
        &'allocated self,
    ) -> Result<
        Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
        Qwen36PackageAdmissionError,
    > {
        #[cfg(not(unix))]
        {
            return Err(Qwen36TensorWorkError::AdditiveCampaignUnsupportedPlatform.into());
        }
        #[cfg(unix)]
        {
            self.verify_current()?;
            let (root, objects, directories) = self.parent.open_selection_store()?;
            let receipt = read_admission(&root.join(ADMISSION_FILE))?;
            validate_admission_binding(&receipt, &self.receipt)?;
            verify_admission_records(self, &objects, &receipt)?;
            validate_directories(&directories)?;
            let package_records = pin_package_records(&objects, &receipt)?;
            Ok(Qwen36PackageAdmittedCampaignStore {
                allocated: self,
                receipt,
                directories,
                package_records,
            })
        }
    }
}

#[cfg(unix)]
fn streamed_profile_plan(
    codec: tritium_format::salt_v2::SaltV2Codec,
    specs: Vec<SaltV2StreamTensorSpec>,
    staged_map: &mut super::StagedPackedMap,
    tile_count: u64,
    profile: SaltV2Profile,
) -> Result<SaltV2PackageStreamPlan, Qwen36PackageAdmissionError> {
    let mut cursor = staged_map.cursor(tile_count)?;
    let counts = std::iter::from_fn(|| match cursor.next_count() {
        Ok(Some(count)) => Some(Ok(count)),
        Ok(None) => None,
        Err(error) => Some(Err(error)),
    });
    let plan =
        SaltV2PackageStreamPlan::try_new(codec, specs, counts).map_err(|error| match error {
            SaltV2PackageStreamPlanError::Package(error) => {
                materialization_package_error(profile, error)
            }
            SaltV2PackageStreamPlanError::Source(error) => {
                Qwen36PackageAdmissionError::Campaign(error)
            }
        })?;
    cursor.finish()?;
    Ok(plan)
}

fn materialization_package_error(
    profile: SaltV2Profile,
    error: tritium_format::salt_v2_package::SaltV2PackageError,
) -> Qwen36PackageAdmissionError {
    materialization_error(profile, SaltV2PackageStreamError::Package(error))
}

fn materialization_error(
    profile: SaltV2Profile,
    error: SaltV2PackageStreamError,
) -> Qwen36PackageAdmissionError {
    Qwen36PackageAdmissionError::Materialization { profile, error }
}

fn verify_admission(
    allocated: &Qwen36AllocatedCampaignStore<'_, '_, '_>,
    expected: &Qwen36PackageAdmissionReceipt,
) -> Result<(), Qwen36PackageAdmissionError> {
    allocated.verify_current()?;
    let (root, objects, directories) = allocated.parent.open_selection_store()?;
    let current = read_admission(&root.join(ADMISSION_FILE))?;
    if current != *expected {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch("package admission receipt").into());
    }
    validate_admission_binding(&current, &allocated.receipt)?;
    verify_admission_records(allocated, &objects, &current)?;
    validate_directories(&directories)?;
    Ok(())
}

fn pin_package_records(
    objects: &TensorWorkStore,
    receipt: &Qwen36PackageAdmissionReceipt,
) -> Result<[PinnedPackageRecord; 2], Qwen36TensorWorkError> {
    Ok([
        PinnedPackageRecord::pin(
            objects.record_path(receipt.compact.record.record_id()),
            receipt.compact.record.record_bytes(),
        )?,
        PinnedPackageRecord::pin(
            objects.record_path(receipt.near_lossless.record.record_id()),
            receipt.near_lossless.record.record_bytes(),
        )?,
    ])
}

#[cfg(unix)]
fn same_file_version(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

#[cfg(not(unix))]
fn same_file_version(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len() && left.modified().ok() == right.modified().ok()
}

fn validate_package_pair<R1: Read + Seek, R2: Read + Seek>(
    allocated: &Qwen36AllocatedCampaignStore<'_, '_, '_>,
    objects: &TensorWorkStore,
    compact: &mut SaltV2PackageReader<R1>,
    near: &mut SaltV2PackageReader<R2>,
) -> Result<(), Qwen36PackageAdmissionError> {
    let selection = &allocated.receipt;
    if compact.codec() != selection.spec.codec() || near.codec() != selection.spec.codec() {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch("selected package codec").into());
    }
    let budgets = selection.spec.budgets();
    let compact_runtime = compact
        .indexed_runtime_ledger()
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
    let near_runtime = near
        .indexed_runtime_ledger()
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
    let compact_bytes = PhysicalBytes {
        serialized: compact.ledger().total_bytes,
        resident: compact_runtime.steady_resident_bytes(),
    };
    let near_bytes = PhysicalBytes {
        serialized: near.ledger().total_bytes,
        resident: near_runtime.steady_resident_bytes(),
    };
    if compact_runtime.present_planes() != selection.compact.selected_planes
        || near_runtime.present_planes() != selection.near_lossless.selected_planes
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "selected package present-plane ledger",
        )
        .into());
    }
    if !compact_bytes.fits_within(budgets.compact.maximum)
        || !near_bytes.fits_within(budgets.near_lossless.maximum)
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("selected package budget ledger").into(),
        );
    }

    let (completion, manifest, _) = allocated
        .parent
        .require_complete_verified(FixedCampaignMode::Capture)?;
    if completion != allocated.parent_completion {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch("package parent completion").into());
    }
    let masters = &allocated.parent.spec.expected_masters;
    if compact.len() != masters.len()
        || near.len() != masters.len()
        || manifest.masters.len() != masters.len()
        || !compact
            .tensor_names_encoded_order()
            .eq(masters.iter().map(|master| master.name()))
        || !near
            .tensor_names_encoded_order()
            .eq(masters.iter().map(|master| master.name()))
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("selected package tensor order").into(),
        );
    }

    let mut compact_map = stage_verified_map(
        objects,
        &selection.compact.map_record,
        "compact.package.map",
    )?;
    let mut near_map = stage_verified_map(
        objects,
        &selection.near_lossless.map_record,
        "near.package.map",
    )?;
    let mut compact_counts = compact_map.cursor(selection.tile_count)?;
    let mut near_counts = near_map.cursor(selection.tile_count)?;
    for ((master, parent_master), ordinal) in masters.iter().zip(&manifest.masters).zip(0usize..) {
        validate_tensor_metadata(compact, master, SaltV2Profile::CompactV1)?;
        validate_tensor_metadata(near, master, SaltV2Profile::NearLosslessV1)?;
        let compact_semantic = compact.semantic_tensor(master.name()).ok_or(
            Qwen36TensorWorkError::WorkspaceMismatch("CompactV1 package tensor"),
        )?;
        let near_semantic =
            near.semantic_tensor(master.name())
                .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                    "NearLosslessV1 package tensor",
                ))?;
        let mut compact_package_counts = compact
            .tensor_plane_counts(master.name())
            .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
        let mut near_package_counts = near
            .tensor_plane_counts(master.name())
            .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
        let mut compact_stream = SaltV2SemanticTensorStream::new(
            master.name(),
            master.shape().to_vec(),
            SaltV2Transform::None,
        )
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error.into()))?;
        let mut near_stream = SaltV2SemanticTensorStream::new(
            master.name(),
            master.shape().to_vec(),
            SaltV2Transform::None,
        )
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error.into()))?;
        validate_parent_tensor_prefixes(
            allocated,
            master,
            parent_master,
            &mut compact_counts,
            &mut near_counts,
            &mut compact_package_counts,
            &mut near_package_counts,
            &mut compact_stream,
            &mut near_stream,
        )?;
        if compact_package_counts.next().is_some() || near_package_counts.next().is_some() {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected package tensor tile coverage",
            )
            .into());
        }
        let actual_compact = compact_stream
            .finish()
            .map_err(|error| package_error(SaltV2Profile::CompactV1, error.into()))?;
        let actual_near = near_stream
            .finish()
            .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error.into()))?;
        if actual_compact != compact_semantic || actual_near != near_semantic {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected package parent prefix semantics",
            )
            .into());
        }
        let _ = ordinal;
    }
    if compact_counts.next_count()?.is_some() || near_counts.next_count()?.is_some() {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("selected package map coverage").into(),
        );
    }
    compact_counts.finish()?;
    near_counts.finish()?;
    allocated.parent.verify_completion_receipt(&completion)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_parent_tensor_prefixes(
    allocated: &Qwen36AllocatedCampaignStore<'_, '_, '_>,
    master: &SaltV2MasterTensorSpec,
    parent_master: &Qwen36AdditiveMasterReceipt,
    compact_counts: &mut super::PackedMapCursor<'_>,
    near_counts: &mut super::PackedMapCursor<'_>,
    compact_package_counts: &mut impl Iterator<Item = usize>,
    near_package_counts: &mut impl Iterator<Item = usize>,
    compact_stream: &mut SaltV2SemanticTensorStream,
    near_stream: &mut SaltV2SemanticTensorStream,
) -> Result<(), Qwen36PackageAdmissionError> {
    let mut decoder =
        SaltV2MasterTensorDecoder::new(master).map_err(Qwen36TensorWorkError::Master)?;
    let mut visited = 0usize;
    allocated
        .parent
        .objects
        .try_visit_verified(&parent_master.record, MASTER_STREAM_CHUNK_BYTES, |chunk| {
            decoder
                .try_push(chunk, &mut |tile| {
                    let compact_count = compact_counts.next_count()?.ok_or(
                        Qwen36TensorWorkError::WorkspaceMismatch(
                            "CompactV1 allocation map is short",
                        ),
                    )?;
                    let near_count = near_counts.next_count()?.ok_or(
                        Qwen36TensorWorkError::WorkspaceMismatch(
                            "NearLosslessV1 allocation map is short",
                        ),
                    )?;
                    if compact_package_counts.next() != Some(usize::from(compact_count))
                        || near_package_counts.next() != Some(usize::from(near_count))
                        || compact_count > near_count
                        || usize::from(near_count) > tile.planes().len()
                    {
                        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                            "selected package plane counts",
                        ));
                    }
                    compact_stream
                        .push_tile(&tile.planes()[..usize::from(compact_count)])
                        .map_err(|_| {
                            Qwen36TensorWorkError::WorkspaceMismatch(
                                "CompactV1 parent semantic stream",
                            )
                        })?;
                    near_stream
                        .push_tile(&tile.planes()[..usize::from(near_count)])
                        .map_err(|_| {
                            Qwen36TensorWorkError::WorkspaceMismatch(
                                "NearLosslessV1 parent semantic stream",
                            )
                        })?;
                    visited += 1;
                    Ok(())
                })
                .map_err(|error| match error {
                    SaltV2MasterVisitError::Master(error) => Qwen36TensorWorkError::Master(error),
                    SaltV2MasterVisitError::Visitor(error) => error,
                })
        })
        .map_err(|error| match error {
            TensorVisitError::Store(error) => Qwen36TensorWorkError::TensorStore(error),
            TensorVisitError::Sink(error) => error,
        })?;
    let decoded = decoder.finish().map_err(Qwen36TensorWorkError::Master)?;
    if visited != master.tile_count()
        || decoded.tensor_master_id() != parent_master.tensor_master_id()
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("selected package parent master").into(),
        );
    }
    Ok(())
}

fn validate_tensor_metadata<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    master: &SaltV2MasterTensorSpec,
    profile: SaltV2Profile,
) -> Result<(), Qwen36PackageAdmissionError> {
    let info =
        reader
            .tensor_info(master.name())
            .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected package tensor name",
            ))?;
    if info.dims() != master.shape()
        || info.logical_coefficients() != master.logical_coefficients()
        || info.tile_count() != master.tile_count()
        || info.transform() != SaltV2Transform::None
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(match profile {
            SaltV2Profile::CompactV1 => "CompactV1 package tensor geometry",
            SaltV2Profile::NearLosslessV1 => "NearLosslessV1 package tensor geometry",
        })
        .into());
    }
    Ok(())
}

fn publish_staged_profile<R: Read + Seek>(
    objects: &TensorWorkStore,
    selection: &Qwen36SelectedAllocationReceipt,
    profile: SaltV2Profile,
    staged: &mut StagedPackage,
    reader: &SaltV2PackageReader<R>,
) -> Result<Qwen36PackageProfileReceipt, Qwen36PackageAdmissionError> {
    let runtime = reader
        .indexed_runtime_ledger()
        .map_err(|error| package_error(profile, error))?;
    let package_id = reader.package_id();
    staged.bind_package_id(package_id)?;
    let package_ledger = reader.ledger();
    let runtime_ledger = Qwen36PackageRuntimeLedger::from_runtime(runtime);
    let record_spec = package_record_spec(
        selection,
        profile,
        package_id,
        package_ledger,
        runtime_ledger,
    )?;
    let record = objects
        .put(&record_spec, |writer| staged.copy_verified_to(writer))
        .map_err(|error| match error {
            TensorPutError::Store(error) => {
                Qwen36PackageAdmissionError::Campaign(Qwen36TensorWorkError::TensorStore(error))
            }
            TensorPutError::Producer(error) => error,
        })?;
    Ok(Qwen36PackageProfileReceipt {
        package_id,
        record,
        package_ledger,
        runtime_ledger,
    })
}

fn verify_admission_records(
    allocated: &Qwen36AllocatedCampaignStore<'_, '_, '_>,
    objects: &TensorWorkStore,
    receipt: &Qwen36PackageAdmissionReceipt,
) -> Result<(), Qwen36PackageAdmissionError> {
    validate_admission_binding(receipt, &allocated.receipt)?;
    let compact_spec = package_record_spec(
        &allocated.receipt,
        SaltV2Profile::CompactV1,
        receipt.compact.package_id,
        receipt.compact.package_ledger,
        receipt.compact.runtime_ledger,
    )?;
    let near_spec = package_record_spec(
        &allocated.receipt,
        SaltV2Profile::NearLosslessV1,
        receipt.near_lossless.package_id,
        receipt.near_lossless.package_ledger,
        receipt.near_lossless.runtime_ledger,
    )?;
    if !receipt.compact.record.matches_spec(&compact_spec)
        || !receipt.near_lossless.record.matches_spec(&near_spec)
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "package admission record descriptor",
        )
        .into());
    }
    let compact_staged = stage_record(objects, &receipt.compact, SaltV2Profile::CompactV1)?;
    let near_staged = stage_record(
        objects,
        &receipt.near_lossless,
        SaltV2Profile::NearLosslessV1,
    )?;
    let mut compact_reader = compact_staged.strict_reader(SaltV2Profile::CompactV1)?;
    let mut near_reader = near_staged.strict_reader(SaltV2Profile::NearLosslessV1)?;
    if compact_reader.package_id() != receipt.compact.package_id
        || near_reader.package_id() != receipt.near_lossless.package_id
        || compact_reader.ledger() != receipt.compact.package_ledger
        || near_reader.ledger() != receipt.near_lossless.package_ledger
        || Qwen36PackageRuntimeLedger::from_runtime(
            compact_reader
                .indexed_runtime_ledger()
                .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?,
        ) != receipt.compact.runtime_ledger
        || Qwen36PackageRuntimeLedger::from_runtime(
            near_reader
                .indexed_runtime_ledger()
                .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?,
        ) != receipt.near_lossless.runtime_ledger
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("package admission measured ledger").into(),
        );
    }
    validate_package_pair(allocated, objects, &mut compact_reader, &mut near_reader)?;
    compact_reader
        .verify_unchanged()
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
    near_reader
        .verify_unchanged()
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
    allocated.verify_current()?;
    Ok(())
}

fn stage_record(
    objects: &TensorWorkStore,
    profile_receipt: &Qwen36PackageProfileReceipt,
    profile: SaltV2Profile,
) -> Result<StagedPackage, Qwen36PackageAdmissionError> {
    let mut staged = StagedPackage::empty(objects.temporary_dir(), profile_name(profile), profile)?;
    objects
        .try_visit_verified(&profile_receipt.record, PACKAGE_CHUNK_BYTES, |chunk| {
            staged.push(chunk, profile)
        })
        .map_err(|error| match error {
            TensorVisitError::Store(error) => {
                Qwen36PackageAdmissionError::Campaign(Qwen36TensorWorkError::TensorStore(error))
            }
            TensorVisitError::Sink(error) => error,
        })?;
    staged.finish(profile_receipt.record.info().payload_bytes(), profile)?;
    staged.bind_package_id(profile_receipt.package_id)?;
    Ok(staged)
}

#[derive(Debug)]
struct StagedPackage {
    path: PathBuf,
    file: File,
    bytes: u64,
    package_id: Option<PackageId>,
    profile: SaltV2Profile,
}

impl StagedPackage {
    fn empty(
        directory: &Path,
        prefix: &str,
        profile: SaltV2Profile,
    ) -> Result<Self, Qwen36PackageAdmissionError> {
        let (path, file) = create_temporary_file(directory, prefix).map_err(|error| {
            Qwen36PackageAdmissionError::Campaign(Qwen36TensorWorkError::TensorStore(error))
        })?;
        Ok(Self {
            path,
            file,
            bytes: 0,
            package_id: None,
            profile,
        })
    }

    fn from_source(
        directory: &Path,
        prefix: &str,
        mut source: impl Read,
        maximum: u64,
        profile: SaltV2Profile,
    ) -> Result<Self, Qwen36PackageAdmissionError> {
        let mut staged = Self::empty(directory, prefix, profile)?;
        let mut buffer = [0u8; PACKAGE_CHUNK_BYTES];
        loop {
            let count = match source.read(&mut buffer) {
                Ok(count) => count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(Qwen36PackageAdmissionError::Source {
                        profile,
                        kind: error.kind(),
                    });
                }
            };
            if count == 0 {
                break;
            }
            let next = staged.bytes.checked_add(count as u64).ok_or(
                Qwen36TensorWorkError::LengthOverflow("selected package bytes"),
            )?;
            if next > maximum {
                return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                    "selected package serialized ceiling",
                )
                .into());
            }
            staged.push(&buffer[..count], profile)?;
        }
        staged.finish(staged.bytes, profile)?;
        Ok(staged)
    }

    fn clone_file(&self, profile: SaltV2Profile) -> Result<File, Qwen36PackageAdmissionError> {
        self.file
            .try_clone()
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile,
                kind: error.kind(),
            })
    }

    fn finish_materialized(
        &mut self,
        expected: u64,
        profile: SaltV2Profile,
    ) -> Result<(), Qwen36PackageAdmissionError> {
        self.bytes = expected;
        self.finish(expected, profile)
    }

    fn push(
        &mut self,
        bytes: &[u8],
        profile: SaltV2Profile,
    ) -> Result<(), Qwen36PackageAdmissionError> {
        self.file
            .write_all(bytes)
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile,
                kind: error.kind(),
            })?;
        self.bytes = self.bytes.checked_add(bytes.len() as u64).ok_or(
            Qwen36TensorWorkError::LengthOverflow("selected package bytes"),
        )?;
        Ok(())
    }

    fn finish(
        &mut self,
        expected: u64,
        profile: SaltV2Profile,
    ) -> Result<(), Qwen36PackageAdmissionError> {
        self.file
            .sync_all()
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile,
                kind: error.kind(),
            })?;
        if self.bytes != expected
            || self
                .file
                .metadata()
                .map_err(|error| Qwen36PackageAdmissionError::Source {
                    profile,
                    kind: error.kind(),
                })?
                .len()
                != expected
        {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("selected package staged length").into(),
            );
        }
        Ok(())
    }

    fn strict_reader(
        &self,
        profile: SaltV2Profile,
    ) -> Result<SaltV2PackageReader<File>, Qwen36PackageAdmissionError> {
        let file = self
            .file
            .try_clone()
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile,
                kind: error.kind(),
            })?;
        SaltV2PackageReader::new_strict(file).map_err(|error| package_error(profile, error))
    }

    fn bind_package_id(
        &mut self,
        package_id: PackageId,
    ) -> Result<(), Qwen36PackageAdmissionError> {
        match self.package_id {
            Some(current) if current != package_id => {
                Err(Qwen36TensorWorkError::WorkspaceMismatch("selected package identity").into())
            }
            _ => {
                self.package_id = Some(package_id);
                Ok(())
            }
        }
    }

    fn copy_verified_to(
        &mut self,
        writer: &mut TensorPayloadWriter<'_>,
    ) -> Result<(), Qwen36PackageAdmissionError> {
        let expected = self
            .package_id
            .ok_or(Qwen36TensorWorkError::WorkspaceMalformed(
                "unbound selected package identity",
            ))?;
        let before = self
            .file
            .metadata()
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile: self.profile,
                kind: error.kind(),
            })?
            .len();
        if before != self.bytes {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("selected package staged length").into(),
            );
        }
        self.file.seek(SeekFrom::Start(0)).map_err(|error| {
            Qwen36PackageAdmissionError::Source {
                profile: self.profile,
                kind: error.kind(),
            }
        })?;
        let mut remaining = self.bytes;
        let mut buffer = [0u8; PACKAGE_CHUNK_BYTES];
        let mut hasher = PackageHasher::new();
        while remaining != 0 {
            let count = usize::try_from(remaining.min(PACKAGE_CHUNK_BYTES as u64))
                .map_err(|_| Qwen36TensorWorkError::LengthOverflow("selected package copy"))?;
            self.file
                .read_exact(&mut buffer[..count])
                .map_err(|error| Qwen36PackageAdmissionError::Source {
                    profile: self.profile,
                    kind: error.kind(),
                })?;
            hasher.update(&buffer[..count]);
            writer.write_all(&buffer[..count]).map_err(|error| {
                Qwen36PackageAdmissionError::Source {
                    profile: self.profile,
                    kind: error.kind(),
                }
            })?;
            remaining -= count as u64;
        }
        if self
            .file
            .metadata()
            .map_err(|error| Qwen36PackageAdmissionError::Source {
                profile: self.profile,
                kind: error.kind(),
            })?
            .len()
            != before
            || hasher.finalize() != expected
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "selected package changed before CAS publication",
            )
            .into());
        }
        Ok(())
    }
}

impl Drop for StagedPackage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn package_record_spec(
    selection: &Qwen36SelectedAllocationReceipt,
    profile: SaltV2Profile,
    package_id: PackageId,
    package_ledger: SaltV2PackageLedger,
    runtime_ledger: Qwen36PackageRuntimeLedger,
) -> Result<TensorRecordSpec, Qwen36TensorWorkError> {
    let mut metadata = Vec::new();
    metadata.extend_from_slice(&PACKAGE_METADATA_MAGIC);
    metadata.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    metadata.push(profile_tag(profile));
    metadata.push(codec_tag(selection.spec.codec())?);
    metadata.extend_from_slice(selection.selection_id.as_bytes());
    metadata.extend_from_slice(selection.parent_completion_id.as_bytes());
    metadata.extend_from_slice(selection.spec.spec_id().as_bytes());
    metadata.extend_from_slice(package_id.as_bytes());
    encode_package_ledger(&mut metadata, package_ledger);
    encode_runtime_ledger(&mut metadata, runtime_ledger);
    TensorRecordSpec::new(
        ContentId::of_bytes(PACKAGE_SCHEMA),
        selection.source_model_id,
        *package_id.as_bytes(),
        profile_name(profile),
        vec![package_ledger.total_bytes],
        metadata,
        package_ledger.total_bytes,
    )
    .map_err(Qwen36TensorWorkError::TensorStore)
}

fn validate_admission_binding(
    receipt: &Qwen36PackageAdmissionReceipt,
    selection: &Qwen36SelectedAllocationReceipt,
) -> Result<(), Qwen36TensorWorkError> {
    if receipt.selection_id != selection.selection_id
        || receipt.parent_completion_id != selection.parent_completion_id
        || receipt.allocation_spec_id != selection.spec.spec_id()
        || receipt.nested_allocation_id != selection.nested_allocation_id
        || receipt.compact_map_id != selection.compact.allocation_map_id
        || receipt.compact_loss_id != selection.compact.selected_loss_id
        || receipt.near_map_id != selection.near_lossless.allocation_map_id
        || receipt.near_loss_id != selection.near_lossless.selected_loss_id
        || !receipt
            .compact
            .physical_bytes()
            .fits_within(selection.spec.budgets().compact.maximum)
        || !receipt
            .near_lossless
            .physical_bytes()
            .fits_within(selection.spec.budgets().near_lossless.maximum)
        || receipt.compact.runtime_ledger.present_planes != selection.compact.selected_planes
        || receipt.near_lossless.runtime_ledger.present_planes
            != selection.near_lossless.selected_planes
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "package admission selection binding",
        ));
    }
    Ok(())
}

fn read_admission(path: &Path) -> Result<Qwen36PackageAdmissionReceipt, Qwen36TensorWorkError> {
    let bytes = read_regular_bounded(path, MAX_ADMISSION_BYTES, "selected package admission")?;
    Qwen36PackageAdmissionReceipt::from_canonical_bytes(&bytes)
}

pub(super) fn admission_record_ids_if_present(
    root: &Path,
    objects: &TensorWorkStore,
) -> Result<Vec<ContentId>, Qwen36TensorWorkError> {
    let path = root.join(ADMISSION_FILE);
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => Err(
            Qwen36TensorWorkError::InvalidPath("selected package admission"),
        ),
        Ok(_) => {
            let receipt = read_admission(&path)?;
            let compact = objects
                .open_verified(&receipt.compact.record)
                .map_err(Qwen36TensorWorkError::TensorStore)?;
            let near = objects
                .open_verified(&receipt.near_lossless.record)
                .map_err(Qwen36TensorWorkError::TensorStore)?;
            drop((compact, near));
            Ok(vec![
                receipt.compact.record.record_id(),
                receipt.near_lossless.record.record_id(),
            ])
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(work_io("inspect selected package admission", error)),
    }
}

fn encode_profile(
    output: &mut Vec<u8>,
    profile: &Qwen36PackageProfileReceipt,
) -> Result<(), Qwen36TensorWorkError> {
    output.extend_from_slice(profile.package_id.as_bytes());
    encode_package_ledger(output, profile.package_ledger);
    encode_runtime_ledger(output, profile.runtime_ledger);
    let record = profile
        .record
        .canonical_bytes()
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    let length = u32::try_from(record.len())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("package record receipt"))?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(&record);
    Ok(())
}

fn decode_profile(
    cursor: &mut CanonicalCursor<'_>,
) -> Result<Qwen36PackageProfileReceipt, Qwen36TensorWorkError> {
    let package_id = PackageId::from_digest(cursor.digest()?);
    let package_ledger = decode_package_ledger(cursor)?;
    let runtime_ledger = decode_runtime_ledger(cursor)?;
    let record_len = usize::try_from(cursor.u32()?)
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("package record receipt"))?;
    let record = TensorRecordReceipt::from_canonical_bytes(cursor.take(record_len)?)
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    Ok(Qwen36PackageProfileReceipt {
        package_id,
        record,
        package_ledger,
        runtime_ledger,
    })
}

fn encode_package_ledger(output: &mut Vec<u8>, value: SaltV2PackageLedger) {
    for field in [
        value.headers_bytes,
        value.transform_bytes,
        value.maps_bytes,
        value.allocation_map_bits,
        value.allocation_map_embedded_bits,
    ] {
        output.extend_from_slice(&field.to_le_bytes());
    }
    output.push(value.allocation_map_package_embedded_bits);
    output.extend_from_slice(&[0; 7]);
    for field in [
        value.allocation_map_tensor_embedded_bits,
        value.allocation_tiles,
        value.allocation_capacity_coefficients,
        value.payload_bytes,
        value.scales_bytes,
        value.padding_bytes,
        value.serialized_unpadded_bytes,
        value.total_bytes,
        value.codec_padding_trits,
        value.codec_padding_bits,
    ] {
        output.extend_from_slice(&field.to_le_bytes());
    }
}

fn decode_package_ledger(
    cursor: &mut CanonicalCursor<'_>,
) -> Result<SaltV2PackageLedger, Qwen36TensorWorkError> {
    let headers_bytes = cursor.u64()?;
    let transform_bytes = cursor.u64()?;
    let maps_bytes = cursor.u64()?;
    let allocation_map_bits = cursor.u64()?;
    let allocation_map_embedded_bits = cursor.u64()?;
    let allocation_map_package_embedded_bits = cursor.u8()?;
    if cursor.take(7)? != [0; 7] {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "package ledger reserved bytes",
        ));
    }
    Ok(SaltV2PackageLedger {
        headers_bytes,
        transform_bytes,
        maps_bytes,
        allocation_map_bits,
        allocation_map_embedded_bits,
        allocation_map_package_embedded_bits,
        allocation_map_tensor_embedded_bits: cursor.u64()?,
        allocation_tiles: cursor.u64()?,
        allocation_capacity_coefficients: cursor.u64()?,
        payload_bytes: cursor.u64()?,
        scales_bytes: cursor.u64()?,
        padding_bytes: cursor.u64()?,
        serialized_unpadded_bytes: cursor.u64()?,
        total_bytes: cursor.u64()?,
        codec_padding_trits: cursor.u64()?,
        codec_padding_bits: cursor.u64()?,
    })
}

fn encode_runtime_ledger(output: &mut Vec<u8>, value: Qwen36PackageRuntimeLedger) {
    for field in [
        value.payload_bytes,
        value.scale_bytes,
        value.allocation_map_bytes,
        value.rank_prefix_bytes,
        value.allocation_map_bits,
        value.allocation_map_embedded_bits,
        value.dense_shadow_bytes,
        value.allocation_tiles,
        value.present_planes,
        value.steady_resident_bytes,
    ] {
        output.extend_from_slice(&field.to_le_bytes());
    }
}

fn decode_runtime_ledger(
    cursor: &mut CanonicalCursor<'_>,
) -> Result<Qwen36PackageRuntimeLedger, Qwen36TensorWorkError> {
    let value = Qwen36PackageRuntimeLedger {
        payload_bytes: cursor.u64()?,
        scale_bytes: cursor.u64()?,
        allocation_map_bytes: cursor.u64()?,
        rank_prefix_bytes: cursor.u64()?,
        allocation_map_bits: cursor.u64()?,
        allocation_map_embedded_bits: cursor.u64()?,
        dense_shadow_bytes: cursor.u64()?,
        allocation_tiles: cursor.u64()?,
        present_planes: cursor.u64()?,
        steady_resident_bytes: cursor.u64()?,
    };
    let expected = value
        .payload_bytes
        .checked_add(value.scale_bytes)
        .and_then(|bytes| bytes.checked_add(value.allocation_map_bytes))
        .and_then(|bytes| bytes.checked_add(value.rank_prefix_bytes))
        .ok_or(Qwen36TensorWorkError::LengthOverflow(
            "package runtime ledger",
        ))?;
    if value.dense_shadow_bytes != 0 || value.steady_resident_bytes != expected {
        return Err(Qwen36TensorWorkError::WorkspaceMalformed(
            "package runtime ledger",
        ));
    }
    Ok(value)
}

const fn profile_tag(profile: SaltV2Profile) -> u8 {
    match profile {
        SaltV2Profile::CompactV1 => 1,
        SaltV2Profile::NearLosslessV1 => 2,
    }
}

const fn profile_name(profile: SaltV2Profile) -> &'static str {
    match profile {
        SaltV2Profile::CompactV1 => "salt-v2-compact.package",
        SaltV2Profile::NearLosslessV1 => "salt-v2-near-lossless.package",
    }
}

fn package_error(
    profile: SaltV2Profile,
    error: SaltV2PackageReadError,
) -> Qwen36PackageAdmissionError {
    Qwen36PackageAdmissionError::Package { profile, error }
}
