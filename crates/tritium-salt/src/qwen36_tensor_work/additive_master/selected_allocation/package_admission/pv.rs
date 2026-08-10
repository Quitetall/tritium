//! Hard-PV package publication bound to one exact admitted PTQ package lineage.

use core::fmt;
use std::{
    error::Error,
    fs,
    io::{Read, Seek},
    path::{Path, PathBuf},
};

use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_master::SaltV2ParentCatalogLineageHasher;
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, SaltV2PackageReader,
    SaltV2PackageStreamWriter, SaltV2Plane, SaltV2SemanticTensorStream, SaltV2StreamTensorSpec,
    SaltV2Transform,
};
use tritium_format::{PackageId, salt_v2_package::SaltV2PackageLedger};
use tritium_nn::{
    DevicePvRecoveryCheckpointArtifact, DevicePvRecoveryError, DevicePvRecoverySession,
    DevicePvRecoveryWeightVisitError,
};
use tritium_quantize::{PhysicalBytes, SaltV2Profile};
use tritium_train::{PvTernaryStructure, PvTernaryWeight};

use crate::{ContentId, TensorPutError, TensorRecordSpec, TensorVisitError, TensorWorkStore};

use super::super::super::{
    CanonicalCursor, FixedCampaignMode, PinnedDirectory, persist_exact, read_regular_bounded,
    validate_directories,
};
use super::super::{Qwen36TensorWorkError, stage_verified_map};
use super::{
    MAX_ADMISSION_BYTES, PinnedPackageRecord, Qwen36PackageAdmissionError,
    Qwen36PackageAdmittedCampaignStore, Qwen36PackageProfileReceipt, Qwen36PackageRuntimeLedger,
    StagedPackage, decode_profile, encode_profile, materialization_error, package_error,
    streamed_profile_plan, validate_tensor_metadata,
};

const PV_DIRECTORY: &str = "pv-package-admission";
const PV_ADMISSION_FILE: &str = "admission.tq36v";
const PV_MAGIC: [u8; 8] = *b"TSQ36PV\0";
const PV_FORMAT_VERSION: u16 = 2;
const PV_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 hard-PV package admission checksum v2";
const PV_PACKAGE_SCHEMA: &[u8] = b"tritium qwen3.6 hard-PV SALT V2 package record v2";
const PV_PACKAGE_METADATA_MAGIC: [u8; 8] = *b"TSQ36VR\0";

/// Failure while publishing or reopening exact hard-PV packages.
#[derive(Debug)]
pub enum Qwen36PvPackageAdmissionError {
    /// Parent package/selection/CAS validation or SALT package work failed.
    Package(Qwen36PackageAdmissionError),
    /// TPVM2/TPVA1 recovery state no longer matches supplied session.
    Recovery(DevicePvRecoveryError),
}

impl fmt::Display for Qwen36PvPackageAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package(error) => write!(formatter, "hard-PV package admission failed: {error}"),
            Self::Recovery(error) => write!(formatter, "hard-PV recovery source failed: {error}"),
        }
    }
}

impl Error for Qwen36PvPackageAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Package(error) => Some(error),
            Self::Recovery(error) => Some(error),
        }
    }
}

impl From<Qwen36PackageAdmissionError> for Qwen36PvPackageAdmissionError {
    fn from(error: Qwen36PackageAdmissionError) -> Self {
        Self::Package(error)
    }
}

impl From<Qwen36TensorWorkError> for Qwen36PvPackageAdmissionError {
    fn from(error: Qwen36TensorWorkError) -> Self {
        Self::Package(Qwen36PackageAdmissionError::Campaign(error))
    }
}

impl From<DevicePvRecoveryError> for Qwen36PvPackageAdmissionError {
    fn from(error: DevicePvRecoveryError) -> Self {
        Self::Recovery(error)
    }
}

/// Failure while visiting one admitted hard-PV package.
#[derive(Debug)]
pub enum Qwen36PvPackageVisitError<E> {
    /// Admission or recovery state changed.
    Admission(Qwen36PvPackageAdmissionError),
    /// Caller sink failed.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36PvPackageVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "hard-PV package visit failed: {error}"),
            Self::Sink(error) => write!(formatter, "hard-PV package sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36PvPackageVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PvBinding {
    parent_admission_id: ContentId,
    selection_id: ContentId,
    parent_completion_id: ContentId,
    parent_context: [u8; 32],
    parent_catalog_digest: [u8; 32],
    artifact_digest: [u8; 32],
    artifact_evidence_digest: [u8; 32],
    campaign_context_digest: [u8; 32],
    step_plan_digest: [u8; 32],
    optimizer_step: u64,
    representation_digest: [u8; 32],
    step_evidence_digest: [u8; 32],
}

/// Durable identity and exact physical ledgers for one hard-PV package pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PvPackageAdmissionReceipt {
    admission_id: ContentId,
    binding: PvBinding,
    compact: Qwen36PackageProfileReceipt,
    near_lossless: Qwen36PackageProfileReceipt,
}

impl Qwen36PvPackageAdmissionReceipt {
    /// Identity of canonical hard-PV admission bytes.
    pub const fn admission_id(&self) -> ContentId {
        self.admission_id
    }

    /// Exact PTQ package admission supplying parent weights.
    pub const fn parent_admission_id(&self) -> ContentId {
        self.binding.parent_admission_id
    }

    /// Selected physical allocation reused by refined packages.
    pub const fn selection_id(&self) -> ContentId {
        self.binding.selection_id
    }

    /// Exact completed PTQ campaign transitively supplying parent packages.
    pub const fn parent_completion_id(&self) -> ContentId {
        self.binding.parent_completion_id
    }

    /// Domain-separated package-parent authority embedded in TPVM2 plan identity.
    pub const fn parent_context(&self) -> [u8; 32] {
        self.binding.parent_context
    }

    /// Exact named Pmax trit/scale semantics that initialized recovery.
    pub const fn parent_catalog_digest(&self) -> [u8; 32] {
        self.binding.parent_catalog_digest
    }

    /// Content identity of exact completed TPVM2 bytes.
    pub const fn artifact_digest(&self) -> [u8; 32] {
        self.binding.artifact_digest
    }

    /// TPVA1 evidence identity binding checkpoint to TPVR1 step receipt.
    pub const fn artifact_evidence_digest(&self) -> [u8; 32] {
        self.binding.artifact_evidence_digest
    }

    /// Frozen recovery-campaign evidence context.
    pub const fn campaign_context_digest(&self) -> [u8; 32] {
        self.binding.campaign_context_digest
    }

    /// Exact model/parent/config plan identity from TPVR1.
    pub const fn step_plan_digest(&self) -> [u8; 32] {
        self.binding.step_plan_digest
    }

    /// Completed alternating-PV optimizer step.
    pub const fn optimizer_step(&self) -> u64 {
        self.binding.optimizer_step
    }

    /// Exact ordered current-weight representation identity.
    pub const fn representation_digest(&self) -> [u8; 32] {
        self.binding.representation_digest
    }

    /// Exact model-wide TPVR1 evidence identity.
    pub const fn step_evidence_digest(&self) -> [u8; 32] {
        self.binding.step_evidence_digest
    }

    /// CompactV1 refined package.
    pub const fn compact(&self) -> &Qwen36PackageProfileReceipt {
        &self.compact
    }

    /// NearLosslessV1 refined package.
    pub const fn near_lossless(&self) -> &Qwen36PackageProfileReceipt {
        &self.near_lossless
    }

    fn new(
        binding: PvBinding,
        compact: Qwen36PackageProfileReceipt,
        near_lossless: Qwen36PackageProfileReceipt,
    ) -> Result<Self, Qwen36TensorWorkError> {
        let mut receipt = Self {
            admission_id: ContentId::from_digest([0; 32]),
            binding,
            compact,
            near_lossless,
        };
        receipt.admission_id = ContentId::of_bytes(&receipt.canonical_bytes()?);
        Ok(receipt)
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36TensorWorkError> {
        let mut output = Vec::new();
        output.extend_from_slice(&PV_MAGIC);
        output.extend_from_slice(&PV_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        for digest in [
            *self.binding.parent_admission_id.as_bytes(),
            *self.binding.selection_id.as_bytes(),
            *self.binding.parent_completion_id.as_bytes(),
            self.binding.parent_context,
            self.binding.parent_catalog_digest,
            self.binding.artifact_digest,
            self.binding.artifact_evidence_digest,
            self.binding.campaign_context_digest,
            self.binding.step_plan_digest,
            self.binding.representation_digest,
            self.binding.step_evidence_digest,
        ] {
            output.extend_from_slice(&digest);
        }
        output.extend_from_slice(&self.binding.optimizer_step.to_le_bytes());
        encode_profile(&mut output, &self.compact)?;
        encode_profile(&mut output, &self.near_lossless)?;
        let mut checksum = blake3::Hasher::new_derive_key(PV_CHECKSUM_CONTEXT);
        checksum.update(&output);
        output.extend_from_slice(checksum.finalize().as_bytes());
        if output.len() as u64 > MAX_ADMISSION_BYTES {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "hard-PV package admission size",
            ));
        }
        Ok(output)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36TensorWorkError> {
        if bytes.len() < PV_MAGIC.len() + 4 + 11 * 32 + 8 + 32 {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "hard-PV package admission length",
            ));
        }
        let body_len = bytes.len() - 32;
        let mut checksum = blake3::Hasher::new_derive_key(PV_CHECKSUM_CONTEXT);
        checksum.update(&bytes[..body_len]);
        if checksum.finalize().as_bytes() != &bytes[body_len..] {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "hard-PV package admission checksum",
            ));
        }
        let mut cursor = CanonicalCursor::new(&bytes[..body_len]);
        if cursor.take(PV_MAGIC.len())? != PV_MAGIC
            || cursor.u16()? != PV_FORMAT_VERSION
            || cursor.u16()? != 0
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "hard-PV package admission header",
            ));
        }
        let binding = PvBinding {
            parent_admission_id: ContentId::from_digest(cursor.digest()?),
            selection_id: ContentId::from_digest(cursor.digest()?),
            parent_completion_id: ContentId::from_digest(cursor.digest()?),
            parent_context: cursor.digest()?,
            parent_catalog_digest: cursor.digest()?,
            artifact_digest: cursor.digest()?,
            artifact_evidence_digest: cursor.digest()?,
            campaign_context_digest: cursor.digest()?,
            step_plan_digest: cursor.digest()?,
            representation_digest: cursor.digest()?,
            step_evidence_digest: cursor.digest()?,
            optimizer_step: cursor.u64()?,
        };
        let compact = decode_profile(&mut cursor)?;
        let near_lossless = decode_profile(&mut cursor)?;
        if cursor.remaining() != 0
            || binding.parent_context == [0; 32]
            || binding.parent_catalog_digest == [0; 32]
        {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "hard-PV package admission payload",
            ));
        }
        let receipt = Self::new(binding, compact, near_lossless)?;
        if receipt.canonical_bytes()? != bytes {
            return Err(Qwen36TensorWorkError::WorkspaceMalformed(
                "noncanonical hard-PV package admission",
            ));
        }
        Ok(receipt)
    }
}

/// Live capability proving exact parent admission, PV artifact, packages, and ledgers.
#[derive(Debug)]
pub struct Qwen36PvPackageAdmittedCampaignStore<
    'pv,
    'backend,
    'model,
    'admission,
    'allocated,
    'parent,
    'store,
    'source,
> {
    parent: &'admission Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
    session: &'pv DevicePvRecoverySession<'backend, 'model>,
    artifact: &'pv DevicePvRecoveryCheckpointArtifact,
    receipt: Qwen36PvPackageAdmissionReceipt,
    objects: TensorWorkStore,
    directories: Vec<PinnedDirectory>,
    package_records: [PinnedPackageRecord; 2],
}

impl Qwen36PvPackageAdmittedCampaignStore<'_, '_, '_, '_, '_, '_, '_, '_> {
    /// Exact durable hard-PV package receipt.
    pub const fn receipt(&self) -> &Qwen36PvPackageAdmissionReceipt {
        &self.receipt
    }

    /// Revalidate parent admission, TPVM2/TPVA1 state, CAS records, and package semantics.
    pub fn verify_current(&self) -> Result<(), Qwen36PvPackageAdmissionError> {
        validate_directories(&self.directories)?;
        for record in &self.package_records {
            record.validate()?;
        }
        verify_pv_admission(
            self.parent,
            self.session,
            self.artifact,
            &self.objects,
            &self.receipt,
        )?;
        validate_directories(&self.directories)?;
        for record in &self.package_records {
            record.validate()?;
        }
        Ok(())
    }

    /// Visit one exact admitted refined package in bounded verified chunks.
    pub fn try_visit_package<E>(
        &self,
        profile: SaltV2Profile,
        max_chunk_bytes: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, Qwen36PvPackageVisitError<E>> {
        self.verify_current()
            .map_err(Qwen36PvPackageVisitError::Admission)?;
        let selected = match profile {
            SaltV2Profile::CompactV1 => self.receipt.compact(),
            SaltV2Profile::NearLosslessV1 => self.receipt.near_lossless(),
        };
        self.objects
            .try_visit_verified(selected.record(), max_chunk_bytes, |chunk| visit(chunk))
            .map_err(|error| match error {
                TensorVisitError::Store(error) => Qwen36PvPackageVisitError::Admission(
                    Qwen36PvPackageAdmissionError::from(Qwen36TensorWorkError::TensorStore(error)),
                ),
                TensorVisitError::Sink(error) => Qwen36PvPackageVisitError::Sink(error),
            })?;
        self.verify_current()
            .map_err(Qwen36PvPackageVisitError::Admission)?;
        Ok(selected.package_ledger().total_bytes)
    }
}

impl<'allocated, 'parent, 'store, 'source>
    Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>
{
    /// Materialize current hard-PV codes into both selected physical profiles.
    ///
    /// Publication is immutable under the TPVM2 artifact digest. Selected maps
    /// remain unchanged; packages contain exact refined trits/scales rather than
    /// relabeled PTQ parent prefixes.
    pub fn materialize_and_admit_pv_packages<'pv, 'backend, 'model, 'admission>(
        &'admission self,
        session: &'pv DevicePvRecoverySession<'backend, 'model>,
        artifact: &'pv DevicePvRecoveryCheckpointArtifact,
    ) -> Result<
        Qwen36PvPackageAdmittedCampaignStore<
            'pv,
            'backend,
            'model,
            'admission,
            'allocated,
            'parent,
            'store,
            'source,
        >,
        Qwen36PvPackageAdmissionError,
    > {
        let _mutation = self.allocated.parent.begin_mutation()?;
        self.verify_current()?;
        let parent_catalog_digest = authoritative_parent_catalog_digest(self)?;
        let binding = pv_binding(self, session, artifact, parent_catalog_digest)?;
        validate_artifact_source(artifact, session)?;

        let (_, selection_objects, _) = self.allocated.parent.open_selection_store()?;
        let selection = &self.allocated.receipt;
        let masters = &self.allocated.parent.spec.expected_masters;
        let specs = masters
            .iter()
            .map(|master| {
                SaltV2StreamTensorSpec::new(
                    master.name(),
                    master.shape().to_vec(),
                    SaltV2Transform::None,
                )
                .map_err(|error| {
                    Qwen36PvPackageAdmissionError::from(materialization_error(
                        SaltV2Profile::CompactV1,
                        error.into(),
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut compact_map = stage_verified_map(
            &selection_objects,
            &selection.compact.map_record,
            "pv.compact.materialize.map",
        )?;
        let mut near_map = stage_verified_map(
            &selection_objects,
            &selection.near_lossless.map_record,
            "pv.near.materialize.map",
        )?;
        let compact_plan = streamed_profile_plan(
            selection.spec.codec(),
            specs.clone(),
            &mut compact_map,
            selection.tile_count,
            SaltV2Profile::CompactV1,
        )?;
        let near_plan = streamed_profile_plan(
            selection.spec.codec(),
            specs,
            &mut near_map,
            selection.tile_count,
            SaltV2Profile::NearLosslessV1,
        )?;
        if compact_plan.ledger().total_bytes > selection.spec.budgets().compact.maximum.serialized
            || near_plan.ledger().total_bytes
                > selection.spec.budgets().near_lossless.maximum.serialized
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "hard-PV package serialized ceiling",
            )
            .into());
        }

        let (root, objects, directories) = open_pv_store(self, binding.artifact_digest)?;
        reclaim_pv_orphans(&root, &objects)?;
        let mut compact_staged = StagedPackage::empty(
            objects.temporary_dir(),
            "pv.compact.materialized.package",
            SaltV2Profile::CompactV1,
        )?;
        let mut near_staged = StagedPackage::empty(
            objects.temporary_dir(),
            "pv.near.materialized.package",
            SaltV2Profile::NearLosslessV1,
        )?;
        let mut compact_writer = SaltV2PackageStreamWriter::new(
            compact_staged.clone_file(SaltV2Profile::CompactV1)?,
            compact_plan,
        )
        .map_err(|error| materialization_error(SaltV2Profile::CompactV1, error))?;
        let mut near_writer = SaltV2PackageStreamWriter::new(
            near_staged.clone_file(SaltV2Profile::NearLosslessV1)?,
            near_plan,
        )
        .map_err(|error| materialization_error(SaltV2Profile::NearLosslessV1, error))?;
        let mut compact_counts = compact_map.cursor(selection.tile_count)?;
        let mut near_counts = near_map.cursor(selection.tile_count)?;
        let mut visited = 0usize;
        let canonical_names = masters
            .iter()
            .map(|master| master.name())
            .collect::<Vec<_>>();
        artifact
            .try_visit_current_weights_in_order(session, &canonical_names, |name, weight| {
                let master = masters.get(visited).ok_or_else(|| {
                    Qwen36PvPackageAdmissionError::from(Qwen36TensorWorkError::WorkspaceMismatch(
                        "hard-PV package tensor count",
                    ))
                })?;
                validate_pv_weight(name, weight, master, selection.spec.codec())?;
                write_weight_tiles(
                    weight,
                    &mut compact_counts,
                    &mut near_counts,
                    &mut compact_writer,
                    &mut near_writer,
                )?;
                visited += 1;
                Ok(())
            })
            .map_err(map_weight_visit_error)?;
        if visited != masters.len()
            || compact_counts.next_count()?.is_some()
            || near_counts.next_count()?.is_some()
        {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "hard-PV package tensor or map coverage",
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
        compact_staged.finish_materialized(compact_ledger.total_bytes, SaltV2Profile::CompactV1)?;
        near_staged.finish_materialized(near_ledger.total_bytes, SaltV2Profile::NearLosslessV1)?;

        let mut compact_reader = compact_staged.strict_reader(SaltV2Profile::CompactV1)?;
        let mut near_reader = near_staged.strict_reader(SaltV2Profile::NearLosslessV1)?;
        validate_pv_package_pair(
            self,
            session,
            artifact,
            &selection_objects,
            &mut compact_reader,
            &mut near_reader,
        )?;
        compact_reader
            .verify_unchanged()
            .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
        near_reader
            .verify_unchanged()
            .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
        let compact = publish_pv_profile(
            &objects,
            selection,
            binding,
            SaltV2Profile::CompactV1,
            &mut compact_staged,
            &compact_reader,
        )?;
        drop((compact_reader, compact_staged));
        let near_lossless = publish_pv_profile(
            &objects,
            selection,
            binding,
            SaltV2Profile::NearLosslessV1,
            &mut near_staged,
            &near_reader,
        )?;
        drop((near_reader, near_staged));
        let receipt = Qwen36PvPackageAdmissionReceipt::new(binding, compact, near_lossless)?;
        verify_pv_binding(self, session, artifact, &receipt)?;
        verify_pv_records(
            self,
            session,
            artifact,
            &selection_objects,
            &objects,
            &receipt,
        )?;
        persist_exact(
            &root.join(PV_ADMISSION_FILE),
            &receipt.canonical_bytes()?,
            "hard-PV package admission",
        )?;
        if read_pv_admission(&root.join(PV_ADMISSION_FILE))? != receipt {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "hard-PV package admission receipt",
            )
            .into());
        }
        self.verify_current()?;
        validate_artifact_source(artifact, session)?;
        validate_directories(&directories)?;
        let package_records = pin_pv_records(&objects, &receipt)?;
        Ok(Qwen36PvPackageAdmittedCampaignStore {
            parent: self,
            session,
            artifact,
            receipt,
            objects,
            directories,
            package_records,
        })
    }

    /// Reopen immutable refined packages for one exact still-current PV artifact.
    pub fn reopen_pv_package_admission<'pv, 'backend, 'model, 'admission>(
        &'admission self,
        session: &'pv DevicePvRecoverySession<'backend, 'model>,
        artifact: &'pv DevicePvRecoveryCheckpointArtifact,
    ) -> Result<
        Qwen36PvPackageAdmittedCampaignStore<
            'pv,
            'backend,
            'model,
            'admission,
            'allocated,
            'parent,
            'store,
            'source,
        >,
        Qwen36PvPackageAdmissionError,
    > {
        self.verify_current()?;
        let (root, objects, directories) =
            open_pv_store(self, artifact.artifact_digest().as_bytes())?;
        let receipt = read_pv_admission(&root.join(PV_ADMISSION_FILE))?;
        verify_pv_admission(self, session, artifact, &objects, &receipt)?;
        validate_directories(&directories)?;
        let package_records = pin_pv_records(&objects, &receipt)?;
        Ok(Qwen36PvPackageAdmittedCampaignStore {
            parent: self,
            session,
            artifact,
            receipt,
            objects,
            directories,
            package_records,
        })
    }
}

fn pv_binding(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    session: &DevicePvRecoverySession<'_, '_>,
    artifact: &DevicePvRecoveryCheckpointArtifact,
    parent_catalog_digest: [u8; 32],
) -> Result<PvBinding, Qwen36PvPackageAdmissionError> {
    let parent_context = parent.pv_parent_context()?;
    if session.parent_context() != Some(parent_context.device_context()?) {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "hard-PV session package-parent context",
        )
        .into());
    }
    if session.parent_catalog_digest() != Some(parent_catalog_digest) {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV session parent catalog").into(),
        );
    }
    let step = artifact.step_receipt();
    Ok(PvBinding {
        parent_admission_id: parent.receipt.admission_id,
        selection_id: parent.receipt.selection_id,
        parent_completion_id: parent.receipt.parent_completion_id,
        parent_context: parent_context.as_bytes(),
        parent_catalog_digest,
        artifact_digest: artifact.artifact_digest().as_bytes(),
        artifact_evidence_digest: artifact.evidence_digest().as_bytes(),
        campaign_context_digest: artifact.campaign_context_digest().as_bytes(),
        step_plan_digest: step.plan_digest(),
        optimizer_step: step.optimizer_step(),
        representation_digest: step.representation_digest(),
        step_evidence_digest: step.evidence_digest(),
    })
}

fn authoritative_parent_catalog_digest(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
) -> Result<[u8; 32], Qwen36PvPackageAdmissionError> {
    let mut catalog = SaltV2ParentCatalogLineageHasher::new();
    let (completion, _, _) = parent.allocated.parent.require_complete_verified_visit(
        FixedCampaignMode::Skip,
        |spec, verified| {
            catalog
                .push(spec.name(), verified.master.parent_lineage_digest())
                .map_err(Qwen36TensorWorkError::Master)
        },
    )?;
    if completion != parent.allocated.parent_completion {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch("hard-PV parent completion").into());
    }
    catalog
        .finish()
        .map_err(Qwen36TensorWorkError::Master)
        .map_err(Into::into)
}

fn validate_artifact_source(
    artifact: &DevicePvRecoveryCheckpointArtifact,
    session: &DevicePvRecoverySession<'_, '_>,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    artifact
        .try_visit_current_weights(session, |_, _| Ok::<_, Qwen36PvPackageAdmissionError>(()))
        .map_err(map_weight_visit_error)?;
    Ok(())
}

fn map_weight_visit_error(
    error: DevicePvRecoveryWeightVisitError<Qwen36PvPackageAdmissionError>,
) -> Qwen36PvPackageAdmissionError {
    match error {
        DevicePvRecoveryWeightVisitError::Recovery(error) => error.into(),
        DevicePvRecoveryWeightVisitError::Visitor(error) => error,
    }
}

fn validate_pv_weight(
    name: &str,
    weight: &PvTernaryWeight,
    master: &super::super::super::SaltV2MasterTensorSpec,
    codec: SaltV2Codec,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    let rows = u64::try_from(weight.rows())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("hard-PV rows"))?;
    let cols = u64::try_from(weight.cols())
        .map_err(|_| Qwen36TensorWorkError::LengthOverflow("hard-PV columns"))?;
    let structure_matches = match codec {
        SaltV2Codec::D2 | SaltV2Codec::B3 => weight.structure() == PvTernaryStructure::Dense,
        SaltV2Codec::S34 => weight.structure() == PvTernaryStructure::S34,
        _ => false,
    };
    if name != master.name()
        || master.shape() != [rows, cols]
        || weight.group_size() != SALT_V2_SCALE_GROUP_SIZE
        || !weight.cols().is_multiple_of(SALT_V2_SCALE_GROUP_SIZE)
        || !structure_matches
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package tensor metadata").into(),
        );
    }
    Ok(())
}

fn weight_tile_planes(
    weight: &PvTernaryWeight,
    tile_index: usize,
) -> Result<Vec<SaltV2Plane>, Qwen36PvPackageAdmissionError> {
    let total =
        weight
            .rows()
            .checked_mul(weight.cols())
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "hard-PV tensor coefficients",
            ))?;
    let start = tile_index
        .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
        .ok_or(Qwen36TensorWorkError::LengthOverflow("hard-PV tile offset"))?;
    let end = start
        .checked_add(SALT_V2_ALLOCATION_TILE_SIZE)
        .map(|end| end.min(total))
        .ok_or(Qwen36TensorWorkError::LengthOverflow("hard-PV tile end"))?;
    if start >= end || !start.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE) {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch("hard-PV tile geometry").into());
    }
    let scale_start = start / SALT_V2_SCALE_GROUP_SIZE;
    let scale_end = end.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
    weight
        .planes()
        .iter()
        .map(|plane| {
            SaltV2Plane::new(
                plane.trits()[start..end].to_vec(),
                plane.scales()[scale_start..scale_end].to_vec(),
            )
            .map_err(|error| {
                Qwen36PvPackageAdmissionError::from(materialization_error(
                    SaltV2Profile::CompactV1,
                    error.into(),
                ))
            })
        })
        .collect()
}

fn write_weight_tiles<C, N, W1, W2>(
    weight: &PvTernaryWeight,
    compact_counts: &mut C,
    near_counts: &mut N,
    compact_writer: &mut SaltV2PackageStreamWriter<W1>,
    near_writer: &mut SaltV2PackageStreamWriter<W2>,
) -> Result<(), Qwen36PvPackageAdmissionError>
where
    C: PlaneCountCursor,
    N: PlaneCountCursor,
    W1: std::io::Write + Seek,
    W2: std::io::Write + Seek,
{
    let total =
        weight
            .rows()
            .checked_mul(weight.cols())
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "hard-PV tensor coefficients",
            ))?;
    for tile_index in 0..total.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE) {
        let compact = compact_counts.require_count("CompactV1 hard-PV allocation map")?;
        let near = near_counts.require_count("NearLosslessV1 hard-PV allocation map")?;
        if compact > near || near > weight.planes().len() {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("hard-PV selected plane counts").into(),
            );
        }
        let planes = weight_tile_planes(weight, tile_index)?;
        compact_writer
            .push_planes(&planes[..compact])
            .map_err(|error| materialization_error(SaltV2Profile::CompactV1, error))?;
        near_writer
            .push_planes(&planes[..near])
            .map_err(|error| materialization_error(SaltV2Profile::NearLosslessV1, error))?;
    }
    Ok(())
}

trait PlaneCountCursor {
    fn require_count(
        &mut self,
        field: &'static str,
    ) -> Result<usize, Qwen36PvPackageAdmissionError>;
}

impl PlaneCountCursor for super::super::PackedMapCursor<'_> {
    fn require_count(
        &mut self,
        field: &'static str,
    ) -> Result<usize, Qwen36PvPackageAdmissionError> {
        self.next_count()?
            .map(usize::from)
            .ok_or_else(|| Qwen36TensorWorkError::WorkspaceMismatch(field).into())
    }
}

fn open_pv_store(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    artifact_digest: [u8; 32],
) -> Result<(PathBuf, TensorWorkStore, Vec<PinnedDirectory>), Qwen36PvPackageAdmissionError> {
    let (selection_root, _, _) = parent.allocated.parent.open_selection_store()?;
    let artifact = ContentId::from_digest(artifact_digest).to_string();
    let root = selection_root.join(PV_DIRECTORY).join(artifact);
    let objects = TensorWorkStore::open(&root).map_err(Qwen36TensorWorkError::TensorStore)?;
    let paths = [
        root.as_path(),
        objects.objects_dir(),
        objects.temporary_dir(),
    ];
    let mut directories = Vec::new();
    directories
        .try_reserve_exact(paths.len())
        .map_err(|_| Qwen36TensorWorkError::AllocationFailed)?;
    for path in paths {
        directories.push(PinnedDirectory::pin(path)?);
    }
    validate_directories(&directories)?;
    Ok((root, objects, directories))
}

fn reclaim_pv_orphans(
    root: &Path,
    objects: &TensorWorkStore,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    objects
        .scavenge_temporary()
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    let admission = root.join(PV_ADMISSION_FILE);
    let retained = match fs::symlink_metadata(&admission) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err(Qwen36TensorWorkError::InvalidPath("hard-PV package admission").into());
        }
        Ok(_) => {
            let receipt = read_pv_admission(&admission)?;
            vec![
                receipt.compact.record().record_id(),
                receipt.near_lossless.record().record_id(),
            ]
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            return Err(
                Qwen36TensorWorkError::TensorStore(crate::TensorWorkError::Io {
                    operation: "inspect hard-PV package admission",
                    kind: error.kind(),
                })
                .into(),
            );
        }
    };
    let sweep = objects
        .prepare_unreferenced_scavenge(&retained)
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    objects
        .commit_unreferenced_scavenge(sweep)
        .map_err(Qwen36TensorWorkError::TensorStore)?;
    Ok(())
}

fn publish_pv_profile<R: Read + Seek>(
    objects: &TensorWorkStore,
    selection: &super::super::Qwen36SelectedAllocationReceipt,
    binding: PvBinding,
    profile: SaltV2Profile,
    staged: &mut StagedPackage,
    reader: &SaltV2PackageReader<R>,
) -> Result<Qwen36PackageProfileReceipt, Qwen36PvPackageAdmissionError> {
    let runtime = reader
        .indexed_runtime_ledger()
        .map_err(|error| package_error(profile, error))?;
    let package_id = reader.package_id();
    staged.bind_package_id(package_id)?;
    let package_ledger = reader.ledger();
    let runtime_ledger = Qwen36PackageRuntimeLedger::from_runtime(runtime);
    let record_spec = pv_package_record_spec(
        selection,
        binding,
        profile,
        package_id,
        package_ledger,
        runtime_ledger,
    )?;
    let record = objects
        .put(&record_spec, |writer| staged.copy_verified_to(writer))
        .map_err(|error| match error {
            TensorPutError::Store(error) => {
                Qwen36PvPackageAdmissionError::from(Qwen36TensorWorkError::TensorStore(error))
            }
            TensorPutError::Producer(error) => error.into(),
        })?;
    Ok(Qwen36PackageProfileReceipt {
        package_id,
        record,
        package_ledger,
        runtime_ledger,
    })
}

fn pv_package_record_spec(
    selection: &super::super::Qwen36SelectedAllocationReceipt,
    binding: PvBinding,
    profile: SaltV2Profile,
    package_id: PackageId,
    package_ledger: SaltV2PackageLedger,
    runtime_ledger: Qwen36PackageRuntimeLedger,
) -> Result<TensorRecordSpec, Qwen36PvPackageAdmissionError> {
    let mut metadata = Vec::new();
    metadata.extend_from_slice(&PV_PACKAGE_METADATA_MAGIC);
    metadata.extend_from_slice(&PV_FORMAT_VERSION.to_le_bytes());
    metadata.push(super::profile_tag(profile));
    metadata.push(super::codec_tag(selection.spec.codec())?);
    for digest in [
        *binding.parent_admission_id.as_bytes(),
        *binding.selection_id.as_bytes(),
        *binding.parent_completion_id.as_bytes(),
        binding.parent_context,
        binding.parent_catalog_digest,
        binding.artifact_digest,
        binding.artifact_evidence_digest,
        binding.campaign_context_digest,
        binding.step_plan_digest,
        binding.representation_digest,
        binding.step_evidence_digest,
        *package_id.as_bytes(),
    ] {
        metadata.extend_from_slice(&digest);
    }
    metadata.extend_from_slice(&binding.optimizer_step.to_le_bytes());
    super::encode_package_ledger(&mut metadata, package_ledger);
    super::encode_runtime_ledger(&mut metadata, runtime_ledger);
    TensorRecordSpec::new(
        ContentId::of_bytes(PV_PACKAGE_SCHEMA),
        selection.source_model_id,
        binding.artifact_digest,
        match profile {
            SaltV2Profile::CompactV1 => "pv-compact-v1",
            SaltV2Profile::NearLosslessV1 => "pv-near-lossless-v1",
        },
        vec![package_ledger.total_bytes],
        metadata,
        package_ledger.total_bytes,
    )
    .map_err(Qwen36TensorWorkError::TensorStore)
    .map_err(Into::into)
}

fn verify_pv_admission(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    session: &DevicePvRecoverySession<'_, '_>,
    artifact: &DevicePvRecoveryCheckpointArtifact,
    objects: &TensorWorkStore,
    receipt: &Qwen36PvPackageAdmissionReceipt,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    parent.verify_current()?;
    let (root, selection_objects, _) = parent.allocated.parent.open_selection_store()?;
    let current = read_pv_admission(
        &root
            .join(PV_DIRECTORY)
            .join(ContentId::from_digest(receipt.binding.artifact_digest).to_string())
            .join(PV_ADMISSION_FILE),
    )?;
    if current != *receipt {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package admission receipt").into(),
        );
    }
    verify_pv_binding(parent, session, artifact, receipt)?;
    verify_pv_records(
        parent,
        session,
        artifact,
        &selection_objects,
        objects,
        receipt,
    )?;
    parent.verify_current()?;
    validate_artifact_source(artifact, session)
}

fn verify_pv_binding(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    session: &DevicePvRecoverySession<'_, '_>,
    artifact: &DevicePvRecoveryCheckpointArtifact,
    receipt: &Qwen36PvPackageAdmissionReceipt,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    let expected = pv_binding(
        parent,
        session,
        artifact,
        receipt.binding.parent_catalog_digest,
    )?;
    let selection = &parent.allocated.receipt;
    if receipt.binding != expected
        || receipt.binding.selection_id != selection.selection_id
        || receipt.compact.runtime_ledger.present_planes != selection.compact.selected_planes
        || receipt.near_lossless.runtime_ledger.present_planes
            != selection.near_lossless.selected_planes
        || !receipt
            .compact
            .physical_bytes()
            .fits_within(selection.spec.budgets().compact.maximum)
        || !receipt
            .near_lossless
            .physical_bytes()
            .fits_within(selection.spec.budgets().near_lossless.maximum)
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package admission binding").into(),
        );
    }
    Ok(())
}

fn verify_pv_records(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    session: &DevicePvRecoverySession<'_, '_>,
    artifact: &DevicePvRecoveryCheckpointArtifact,
    selection_objects: &TensorWorkStore,
    objects: &TensorWorkStore,
    receipt: &Qwen36PvPackageAdmissionReceipt,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    let selection = &parent.allocated.receipt;
    for (profile, package) in [
        (SaltV2Profile::CompactV1, &receipt.compact),
        (SaltV2Profile::NearLosslessV1, &receipt.near_lossless),
    ] {
        let expected = pv_package_record_spec(
            selection,
            receipt.binding,
            profile,
            package.package_id,
            package.package_ledger,
            package.runtime_ledger,
        )?;
        if !package.record.matches_spec(&expected) {
            return Err(Qwen36TensorWorkError::WorkspaceMismatch(
                "hard-PV package record descriptor",
            )
            .into());
        }
    }
    let compact_staged = super::stage_record(objects, &receipt.compact, SaltV2Profile::CompactV1)?;
    let near_staged = super::stage_record(
        objects,
        &receipt.near_lossless,
        SaltV2Profile::NearLosslessV1,
    )?;
    let mut compact = compact_staged.strict_reader(SaltV2Profile::CompactV1)?;
    let mut near = near_staged.strict_reader(SaltV2Profile::NearLosslessV1)?;
    if compact.package_id() != receipt.compact.package_id
        || near.package_id() != receipt.near_lossless.package_id
        || compact.ledger() != receipt.compact.package_ledger
        || near.ledger() != receipt.near_lossless.package_ledger
        || Qwen36PackageRuntimeLedger::from_runtime(
            compact
                .indexed_runtime_ledger()
                .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?,
        ) != receipt.compact.runtime_ledger
        || Qwen36PackageRuntimeLedger::from_runtime(
            near.indexed_runtime_ledger()
                .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?,
        ) != receipt.near_lossless.runtime_ledger
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package measured ledger").into(),
        );
    }
    validate_pv_package_pair(
        parent,
        session,
        artifact,
        selection_objects,
        &mut compact,
        &mut near,
    )?;
    compact
        .verify_unchanged()
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
    near.verify_unchanged()
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
    Ok(())
}

fn validate_pv_package_pair<R1: Read + Seek, R2: Read + Seek>(
    parent: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    session: &DevicePvRecoverySession<'_, '_>,
    artifact: &DevicePvRecoveryCheckpointArtifact,
    selection_objects: &TensorWorkStore,
    compact: &mut SaltV2PackageReader<R1>,
    near: &mut SaltV2PackageReader<R2>,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    let selection = &parent.allocated.receipt;
    let masters = &parent.allocated.parent.spec.expected_masters;
    if compact.codec() != selection.spec.codec()
        || near.codec() != selection.spec.codec()
        || compact.len() != masters.len()
        || near.len() != masters.len()
        || !compact
            .tensor_names_encoded_order()
            .eq(masters.iter().map(|master| master.name()))
        || !near
            .tensor_names_encoded_order()
            .eq(masters.iter().map(|master| master.name()))
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package tensor order").into(),
        );
    }
    let compact_runtime = compact
        .indexed_runtime_ledger()
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error))?;
    let near_runtime = near
        .indexed_runtime_ledger()
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error))?;
    if compact_runtime.present_planes() != selection.compact.selected_planes
        || near_runtime.present_planes() != selection.near_lossless.selected_planes
        || !(PhysicalBytes {
            serialized: compact.ledger().total_bytes,
            resident: compact_runtime.steady_resident_bytes(),
        })
        .fits_within(selection.spec.budgets().compact.maximum)
        || !(PhysicalBytes {
            serialized: near.ledger().total_bytes,
            resident: near_runtime.steady_resident_bytes(),
        })
        .fits_within(selection.spec.budgets().near_lossless.maximum)
    {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package physical ledger").into(),
        );
    }
    let mut compact_map = stage_verified_map(
        selection_objects,
        &selection.compact.map_record,
        "pv.compact.verify.map",
    )?;
    let mut near_map = stage_verified_map(
        selection_objects,
        &selection.near_lossless.map_record,
        "pv.near.verify.map",
    )?;
    let mut compact_counts = compact_map.cursor(selection.tile_count)?;
    let mut near_counts = near_map.cursor(selection.tile_count)?;
    let mut visited = 0usize;
    let canonical_names = masters
        .iter()
        .map(|master| master.name())
        .collect::<Vec<_>>();
    artifact
        .try_visit_current_weights_in_order(session, &canonical_names, |name, weight| {
            let master = masters.get(visited).ok_or_else(|| {
                Qwen36PvPackageAdmissionError::from(Qwen36TensorWorkError::WorkspaceMismatch(
                    "hard-PV package tensor count",
                ))
            })?;
            validate_pv_weight(name, weight, master, selection.spec.codec())?;
            validate_pv_tensor_semantics(
                master,
                weight,
                compact,
                near,
                &mut compact_counts,
                &mut near_counts,
            )?;
            visited += 1;
            Ok(())
        })
        .map_err(map_weight_visit_error)?;
    if visited != masters.len()
        || compact_counts.next_count()?.is_some()
        || near_counts.next_count()?.is_some()
    {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "hard-PV package tensor or map coverage",
        )
        .into());
    }
    compact_counts.finish()?;
    near_counts.finish()?;
    Ok(())
}

fn validate_pv_tensor_semantics<R1: Read + Seek, R2: Read + Seek>(
    master: &super::super::super::SaltV2MasterTensorSpec,
    weight: &PvTernaryWeight,
    compact: &mut SaltV2PackageReader<R1>,
    near: &mut SaltV2PackageReader<R2>,
    compact_counts: &mut super::super::PackedMapCursor<'_>,
    near_counts: &mut super::super::PackedMapCursor<'_>,
) -> Result<(), Qwen36PvPackageAdmissionError> {
    validate_tensor_metadata(compact, master, SaltV2Profile::CompactV1)?;
    validate_tensor_metadata(near, master, SaltV2Profile::NearLosslessV1)?;
    let expected_compact =
        compact
            .semantic_tensor(master.name())
            .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                "CompactV1 hard-PV package tensor",
            ))?;
    let expected_near =
        near.semantic_tensor(master.name())
            .ok_or(Qwen36TensorWorkError::WorkspaceMismatch(
                "NearLosslessV1 hard-PV package tensor",
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
    let total =
        weight
            .rows()
            .checked_mul(weight.cols())
            .ok_or(Qwen36TensorWorkError::LengthOverflow(
                "hard-PV tensor coefficients",
            ))?;
    for tile_index in 0..total.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE) {
        let compact_count = compact_counts.require_count("CompactV1 hard-PV allocation map")?;
        let near_count = near_counts.require_count("NearLosslessV1 hard-PV allocation map")?;
        if compact_package_counts.next() != Some(compact_count)
            || near_package_counts.next() != Some(near_count)
            || compact_count > near_count
            || near_count > weight.planes().len()
        {
            return Err(
                Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package plane counts").into(),
            );
        }
        let planes = weight_tile_planes(weight, tile_index)?;
        compact_stream
            .push_tile(&planes[..compact_count])
            .map_err(|_| {
                Qwen36TensorWorkError::WorkspaceMismatch("CompactV1 hard-PV semantic stream")
            })?;
        near_stream.push_tile(&planes[..near_count]).map_err(|_| {
            Qwen36TensorWorkError::WorkspaceMismatch("NearLosslessV1 hard-PV semantic stream")
        })?;
    }
    if compact_package_counts.next().is_some() || near_package_counts.next().is_some() {
        return Err(Qwen36TensorWorkError::WorkspaceMismatch(
            "hard-PV package tensor tile coverage",
        )
        .into());
    }
    let actual_compact = compact_stream
        .finish()
        .map_err(|error| package_error(SaltV2Profile::CompactV1, error.into()))?;
    let actual_near = near_stream
        .finish()
        .map_err(|error| package_error(SaltV2Profile::NearLosslessV1, error.into()))?;
    if actual_compact != expected_compact || actual_near != expected_near {
        return Err(
            Qwen36TensorWorkError::WorkspaceMismatch("hard-PV package refined semantics").into(),
        );
    }
    Ok(())
}

fn pin_pv_records(
    objects: &TensorWorkStore,
    receipt: &Qwen36PvPackageAdmissionReceipt,
) -> Result<[PinnedPackageRecord; 2], Qwen36PvPackageAdmissionError> {
    Ok([
        PinnedPackageRecord::pin(
            objects.record_path(receipt.compact.record().record_id()),
            receipt.compact.record().record_bytes(),
        )?,
        PinnedPackageRecord::pin(
            objects.record_path(receipt.near_lossless.record().record_id()),
            receipt.near_lossless.record().record_bytes(),
        )?,
    ])
}

fn read_pv_admission(
    path: &Path,
) -> Result<Qwen36PvPackageAdmissionReceipt, Qwen36TensorWorkError> {
    let bytes = read_regular_bounded(path, MAX_ADMISSION_BYTES, "hard-PV package admission")?;
    Qwen36PvPackageAdmissionReceipt::from_canonical_bytes(&bytes)
}
