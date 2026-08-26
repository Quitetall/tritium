//! Checkpoint-backed, resumable Qwen3.6 pure-PTQ campaign driver.

mod capture;
mod grouping;

pub use capture::{
    Qwen36PtqEvidenceCaptureError, Qwen36PtqEvidenceCaptureReceipt,
    Qwen36PtqEvidenceCaptureRequest, Qwen36PtqEvidenceCaptureSession, Qwen36PtqEvidenceCaptureTask,
    collect_qwen36_ptq_evidence,
};
pub use grouping::{
    SharedForwardCaptureGroup, SharedForwardPlanError, SharedForwardTensor,
    plan_shared_forward_groups,
};

use core::fmt;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use tritium_format::{ModelId, salt_v2::SaltV2Codec};
use tritium_quantize::{
    PhysicalBytes, SaltV2Config, SaltV2Error, SaltV2KroneckerEvidence,
    SaltV2KroneckerEvidenceBuildError, SaltV2KroneckerEvidenceBuilder,
    SaltV2KroneckerEvidenceError, SaltV2KroneckerEvidenceReceipt, SaltV2KroneckerEvidenceSpec,
    SaltV2Packing, SaltV2Profile, SaltV2RestartableTensorMasterFitInput,
    fit_salt_v2_restartable_tensor_master, plan_salt_v2_restartable_tensor_master,
};

use crate::{
    Qwen36AdditiveCampaignSpec, Qwen36AdditiveInstallError, Qwen36AdmittedSource,
    Qwen36CampaignPreflightError, Qwen36CompleteWorkspaceReceipt, Qwen36PhysicalAllocationError,
    Qwen36SelectedAllocationSpec, Qwen36TensorWorkError,
    tensor_work_store::{absolute_path, create_temporary_file, ensure_durable_directory},
};
#[cfg(unix)]
use crate::{Qwen36PackageAdmissionError, Qwen36PackageAdmissionReceipt, Qwen36PackageVisitError};

use super::{Qwen36AdditiveWorkSlot, Qwen36TensorWorkStore, same_file_identity};

const DEFAULT_MAX_EVIDENCE_BYTES: u64 = 64 * 1024 * 1024;
const EVIDENCE_EXTENSION: &str = "s2kf";
const EVIDENCE_STAGING_DIRECTORY: &str = ".staging";
const ALLOCATOR_ID_CONTEXT: &str = "tritium qwen3.6 physical allocator identity v1";
const ALLOCATION_RECIPE_CONTEXT: &str = "tritium qwen3.6 physical allocation recipe v1";
const ALLOCATOR_SCHEMA: &[u8] =
    b"exact full-tile nested Hessian prefix allocator; stable tensor/tile/plane ties; v1";

#[cfg(test)]
thread_local! {
    static FAIL_EVIDENCE_DIRECTORY_SYNC_AFTER: std::cell::Cell<u32> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
type FinalPathCheckHook = (
    PathBuf,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
static FINAL_PATH_CHECK_BARRIERS: std::sync::Mutex<Option<FinalPathCheckHook>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
static STAGED_PATH_LINK_BARRIERS: std::sync::Mutex<Option<FinalPathCheckHook>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
static STAGING_DIRECTORY_CREATE_BARRIERS: std::sync::Mutex<Option<FinalPathCheckHook>> =
    std::sync::Mutex::new(None);

/// Bounded canonical layout for per-tensor Kronecker evidence records.
///
/// Record `N` lives at `NNNNNN.s2kf`, where `N` is the tensor's global ordinal
/// in [`Qwen36TensorWorkStore::additive_slots`]. Record contents, rather than
/// filenames, remain authoritative and are checked against the admitted slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PtqEvidenceDirectory {
    root: PathBuf,
    max_record_bytes: u64,
    root_identity: EvidenceDirectoryIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EvidenceDirectoryIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl Qwen36PtqEvidenceDirectory {
    /// Create or reopen an ordinary evidence publication directory.
    ///
    /// # Errors
    /// Rejects empty/traversing/symlinked paths and filesystem failures.
    pub fn create(root: impl AsRef<Path>) -> Result<Self, Qwen36PtqDriverError> {
        Self::create_bounded(root, DEFAULT_MAX_EVIDENCE_BYTES)
    }

    /// Create or reopen an evidence directory under an explicit record bound.
    ///
    /// # Errors
    /// Rejects invalid paths, a zero byte bound, or directory creation failures.
    pub fn create_bounded(
        root: impl AsRef<Path>,
        max_record_bytes: u64,
    ) -> Result<Self, Qwen36PtqDriverError> {
        if root.as_ref().as_os_str().is_empty() || max_record_bytes == 0 {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "empty path or zero record byte limit",
            ));
        }
        let root = absolute_path(root.as_ref()).map_err(|source| {
            Qwen36PtqDriverError::Workspace(Qwen36TensorWorkError::TensorStore(source))
        })?;
        ensure_durable_directory(&root, "PTQ evidence directory").map_err(|source| {
            Qwen36PtqDriverError::Workspace(Qwen36TensorWorkError::TensorStore(source))
        })?;
        Self::open_bounded(root, max_record_bytes)
    }

    /// Open an existing ordinary directory with the default 64 MiB record bound.
    ///
    /// # Errors
    /// Rejects empty, traversing, missing, symlinked, or non-directory paths.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, Qwen36PtqDriverError> {
        Self::open_bounded(root, DEFAULT_MAX_EVIDENCE_BYTES)
    }

    /// Open an existing evidence directory under an explicit nonzero record bound.
    ///
    /// # Errors
    /// Rejects an invalid path, a zero limit, or filesystem inspection failure.
    pub fn open_bounded(
        root: impl AsRef<Path>,
        max_record_bytes: u64,
    ) -> Result<Self, Qwen36PtqDriverError> {
        if root.as_ref().as_os_str().is_empty() {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "empty directory path",
            ));
        }
        if max_record_bytes == 0 {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "zero record byte limit",
            ));
        }
        let root = absolute_path(root.as_ref()).map_err(|source| {
            Qwen36PtqDriverError::Workspace(Qwen36TensorWorkError::TensorStore(source))
        })?;
        let metadata = fs::symlink_metadata(&root)
            .map_err(|error| evidence_io("inspect evidence directory", None, error))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "evidence directory type",
            ));
        }
        Ok(Self {
            root,
            max_record_bytes,
            root_identity: evidence_directory_identity(&metadata),
        })
    }

    /// Absolute evidence-directory path.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Maximum bytes admitted for one curvature record.
    #[must_use]
    pub const fn max_record_bytes(&self) -> u64 {
        self.max_record_bytes
    }

    /// Canonical path for one global additive-tensor ordinal.
    #[must_use]
    pub fn record_path(&self, tensor_index: u64) -> PathBuf {
        self.root
            .join(format!("{tensor_index:06}.{EVIDENCE_EXTENSION}"))
    }

    /// Verify an exact zero-based evidence namespace with no missing or extra entries.
    ///
    /// # Errors
    /// Rejects zero/oversized counts, allocation failure, directory I/O, any
    /// noncanonical filename, duplicate/out-of-range ordinal, symlink/special
    /// entry, or incomplete record set.
    pub fn validate_complete(&self, record_count: u64) -> Result<(), Qwen36PtqDriverError> {
        self.validate_namespace(record_count, true).map(|_| ())
    }

    /// Verify every currently present entry belongs to a bounded zero-based namespace.
    ///
    /// Missing canonical records are allowed for resumable capture. Extra,
    /// duplicate, noncanonical, symlinked, or special entries are rejected.
    ///
    /// # Errors
    /// Rejects the same malformed namespace state as [`Self::validate_complete`]
    /// while allowing missing records.
    pub fn validate_partial(&self, record_count: u64) -> Result<u64, Qwen36PtqDriverError> {
        self.validate_namespace(record_count, false)
    }

    fn validate_namespace(
        &self,
        record_count: u64,
        require_complete: bool,
    ) -> Result<u64, Qwen36PtqDriverError> {
        self.verify_root()?;
        let record_count = usize::try_from(record_count)
            .ok()
            .filter(|count| (1..1_000_000).contains(count))
            .ok_or(Qwen36PtqDriverError::InvalidEvidencePath("record count"))?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(record_count)
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        seen.resize(record_count, false);
        let mut present = 0_u64;
        let entries = fs::read_dir(&self.root)
            .map_err(|error| evidence_io("read evidence directory", None, error))?;
        for entry in entries {
            let entry = entry.map_err(|error| evidence_io("read evidence entry", None, error))?;
            let metadata = fs::symlink_metadata(entry.path())
                .map_err(|error| evidence_io("inspect evidence entry", None, error))?;
            if entry.file_name() == EVIDENCE_STAGING_DIRECTORY {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                        "staging directory type",
                    ));
                }
                continue;
            }
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                    "record entry type",
                ));
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Qwen36PtqDriverError::InvalidEvidencePath("record filename"))?;
            let stem = name
                .strip_suffix(".s2kf")
                .filter(|stem| stem.len() == 6 && stem.bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or(Qwen36PtqDriverError::InvalidEvidencePath("record filename"))?;
            let tensor_index = stem
                .parse::<usize>()
                .ok()
                .filter(|index| *index < record_count)
                .ok_or(Qwen36PtqDriverError::InvalidEvidencePath("record ordinal"))?;
            if seen[tensor_index] || entry.path() != self.record_path(tensor_index as u64) {
                return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                    "record namespace",
                ));
            }
            seen[tensor_index] = true;
            present = present
                .checked_add(1)
                .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        }
        if require_complete && seen.contains(&false) {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "incomplete record namespace",
            ));
        }
        self.verify_root()?;
        Ok(present)
    }

    /// Strictly read and verify one canonical evidence record.
    ///
    /// # Errors
    /// Rejects a missing/symlinked/non-regular/replaced file, a file over the
    /// configured bound, or malformed canonical evidence.
    pub fn reopen(
        &self,
        tensor_index: u64,
    ) -> Result<SaltV2KroneckerEvidence, Qwen36PtqDriverError> {
        self.verify_root()?;
        let path = self.record_path(tensor_index);
        let record = read_evidence_path(&path, tensor_index, self.max_record_bytes)?;
        self.verify_root()?;
        Ok(record)
    }

    /// Consume streamed producer state and atomically publish its canonical record.
    ///
    /// Consuming builder releases accumulation state before record publication.
    ///
    /// # Errors
    /// Rejects producer finalization or any [`Self::install`] failure.
    pub fn install_builder(
        &self,
        builder: SaltV2KroneckerEvidenceBuilder,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, Qwen36PtqDriverError> {
        let tensor_index = builder.spec().tensor_index();
        self.preflight_builder_spec(builder.spec())?;
        let record =
            builder
                .into_evidence()
                .map_err(|source| Qwen36PtqDriverError::EvidenceBuild {
                    tensor_index,
                    source,
                })?;
        self.install(&record)
    }

    /// Create a producer preflighted against this directory's record ceiling.
    ///
    /// # Errors
    /// Rejects oversized geometry before accumulator construction.
    pub fn create_builder(
        &self,
        spec: SaltV2KroneckerEvidenceSpec,
    ) -> Result<SaltV2KroneckerEvidenceBuilder, Qwen36PtqDriverError> {
        self.create_builder_at(spec, 0)
    }

    /// Create one mergeable producer shard under this directory's ceiling.
    ///
    /// # Errors
    /// Rejects oversized geometry before accumulator construction or invalid
    /// accumulator geometry.
    pub fn create_builder_at(
        &self,
        spec: SaltV2KroneckerEvidenceSpec,
        sample_start: u64,
    ) -> Result<SaltV2KroneckerEvidenceBuilder, Qwen36PtqDriverError> {
        self.preflight_builder_spec(&spec)?;
        let tensor_index = spec.tensor_index();
        SaltV2KroneckerEvidenceBuilder::new_at(spec, sample_start).map_err(|source| {
            Qwen36PtqDriverError::EvidenceBuild {
                tensor_index,
                source,
            }
        })
    }

    /// Create a producer using sparse indexed output factors.
    ///
    /// This is the vocabulary-scale path for embedding-table Fisher/KL
    /// evidence. It is preflighted against the same canonical record ceiling
    /// as the dense producer.
    ///
    /// # Errors
    /// Rejects oversized geometry, input-Hessian curvature, or invalid
    /// accumulator geometry before any evidence-directory mutation.
    pub fn create_indexed_output_builder(
        &self,
        spec: SaltV2KroneckerEvidenceSpec,
    ) -> Result<SaltV2KroneckerEvidenceBuilder, Qwen36PtqDriverError> {
        self.create_indexed_output_builder_at(spec, 0)
    }

    /// Create one mergeable sparse indexed-output producer shard.
    ///
    /// # Errors
    /// Rejects oversized geometry, input-Hessian curvature, or invalid
    /// accumulator geometry before any evidence-directory mutation.
    pub fn create_indexed_output_builder_at(
        &self,
        spec: SaltV2KroneckerEvidenceSpec,
        sample_start: u64,
    ) -> Result<SaltV2KroneckerEvidenceBuilder, Qwen36PtqDriverError> {
        self.preflight_builder_spec(&spec)?;
        let tensor_index = spec.tensor_index();
        SaltV2KroneckerEvidenceBuilder::new_indexed_output_at(spec, sample_start).map_err(
            |source| Qwen36PtqDriverError::EvidenceBuild {
                tensor_index,
                source,
            },
        )
    }

    fn preflight_builder_spec(
        &self,
        spec: &SaltV2KroneckerEvidenceSpec,
    ) -> Result<(), Qwen36PtqDriverError> {
        if spec.canonical_bytes() > self.max_record_bytes {
            return Err(Qwen36PtqDriverError::EvidenceBuild {
                tensor_index: spec.tensor_index(),
                source: SaltV2KroneckerEvidenceBuildError::SizeLimitExceeded {
                    required_bytes: spec.canonical_bytes(),
                    max_bytes: self.max_record_bytes,
                },
            });
        }
        Ok(())
    }

    /// Atomically publish one immutable canonical evidence record.
    ///
    /// Reinstalling byte-identical evidence is idempotent. A conflicting record
    /// at the same global tensor ordinal is never replaced. Publication writes
    /// and verifies a private temporary inode, links it into the canonical
    /// namespace without overwrite, and durably syncs the directory.
    ///
    /// # Errors
    /// Rejects out-of-range ordinals, oversized/malformed evidence, directory
    /// replacement, a conflicting existing record, or any write/sync failure.
    pub fn install(
        &self,
        record: &SaltV2KroneckerEvidence,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, Qwen36PtqDriverError> {
        self.verify_root()?;
        let tensor_index = record.tensor_index();
        preflight_evidence_directory_sync(tensor_index)?;
        if tensor_index >= 1_000_000 {
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "global tensor ordinal",
            });
        }
        let expected = evidence_receipt(record, tensor_index)?;
        if expected.bytes() > self.max_record_bytes {
            return Err(Qwen36PtqDriverError::Evidence {
                tensor_index,
                source: SaltV2KroneckerEvidenceError::SizeLimitExceeded {
                    max_bytes: self.max_record_bytes,
                },
            });
        }
        let destination = self.record_path(tensor_index);
        match fs::symlink_metadata(&destination) {
            Ok(_) => return self.existing_receipt(record),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(evidence_io(
                    "inspect evidence destination",
                    Some(tensor_index),
                    error,
                ));
            }
        }
        let staging = self.open_staging()?;
        #[cfg(test)]
        pause_before_staging_file_create(staging.path());
        let (temporary, mut file) =
            create_temporary_file(staging.path(), ".s2kf").map_err(|source| {
                Qwen36PtqDriverError::Workspace(Qwen36TensorWorkError::TensorStore(source))
            })?;
        if let Err(error) = staging.verify() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        let receipt = match record.write_to(&mut file).and_then(|receipt| {
            file.sync_all()
                .map_err(|error| SaltV2KroneckerEvidenceError::Io {
                    operation: "sync evidence",
                    kind: error.kind(),
                })?;
            Ok(receipt)
        }) {
            Ok(receipt) => receipt,
            Err(source) => {
                let _ = fs::remove_file(&temporary);
                return Err(Qwen36PtqDriverError::Evidence {
                    tensor_index,
                    source,
                });
            }
        };
        drop(file);
        let staged = match verify_written_evidence_path(
            record,
            &temporary,
            tensor_index,
            self.max_record_bytes,
        ) {
            Ok(receipt) => receipt,
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
        };
        if staged.receipt != expected || receipt != expected {
            let _ = fs::remove_file(&temporary);
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "staged canonical record",
            });
        }
        if let Err(error) = self.verify_root() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        if let Err(error) = staging.verify() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        #[cfg(test)]
        pause_before_staged_path_link(&temporary);
        match fs::hard_link(&temporary, &destination) {
            Ok(()) => {
                if let Err(error) =
                    verify_published_evidence_inode(&destination, &staged.file, tensor_index)
                {
                    let _ = fs::remove_file(&destination);
                    let _ = fs::remove_file(&temporary);
                    return Err(error);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let existing = match self.existing_receipt(record) {
                    Ok(receipt) => receipt,
                    Err(error) => {
                        let _ = fs::remove_file(&temporary);
                        return Err(error);
                    }
                };
                remove_staging_link_durably(&temporary, &staging, tensor_index)?;
                return Ok(existing);
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(evidence_io(
                    "publish evidence record",
                    Some(tensor_index),
                    error,
                ));
            }
        }
        if let Err(error) = self.verify_root() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        if let Err(error) = sync_evidence_directory(&self.root) {
            let _ = fs::remove_file(&temporary);
            return Err(evidence_io(
                "sync evidence directory",
                Some(tensor_index),
                error,
            ));
        }
        remove_staging_link_durably(&temporary, &staging, tensor_index)?;
        let installed = self.reopen(tensor_index)?;
        if evidence_receipt(&installed, tensor_index)? != expected {
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "published canonical record",
            });
        }
        Ok(receipt)
    }

    fn existing_receipt(
        &self,
        record: &SaltV2KroneckerEvidence,
    ) -> Result<SaltV2KroneckerEvidenceReceipt, Qwen36PtqDriverError> {
        let tensor_index = record.tensor_index();
        let existing = self.reopen(tensor_index)?;
        let expected = evidence_receipt(record, tensor_index)?;
        if evidence_receipt(&existing, tensor_index)? != expected {
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "immutable record identity",
            });
        }
        let staging = self.open_staging()?;
        sync_evidence_directory(&self.root).map_err(|error| {
            evidence_io(
                "sync existing evidence directory",
                Some(tensor_index),
                error,
            )
        })?;
        staging.sync(tensor_index)?;
        self.verify_root()?;
        staging.verify()?;
        Ok(expected)
    }

    fn open_staging(&self) -> Result<PinnedEvidenceDirectory, Qwen36PtqDriverError> {
        let staging = self.root.join(EVIDENCE_STAGING_DIRECTORY);
        ensure_durable_directory(&staging, "PTQ evidence staging directory").map_err(|source| {
            Qwen36PtqDriverError::Workspace(Qwen36TensorWorkError::TensorStore(source))
        })?;
        self.verify_root()?;
        let pinned = PinnedEvidenceDirectory::open(staging)?;
        self.verify_root()?;
        Ok(pinned)
    }

    fn verify_root(&self) -> Result<(), Qwen36PtqDriverError> {
        let metadata = fs::symlink_metadata(&self.root)
            .map_err(|error| evidence_io("inspect evidence directory", None, error))?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || evidence_directory_identity(&metadata) != self.root_identity
        {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "directory identity changed",
            ));
        }
        Ok(())
    }
}

fn evidence_receipt(
    record: &SaltV2KroneckerEvidence,
    tensor_index: u64,
) -> Result<SaltV2KroneckerEvidenceReceipt, Qwen36PtqDriverError> {
    record
        .receipt()
        .map_err(|source| Qwen36PtqDriverError::Evidence {
            tensor_index,
            source,
        })
}

fn remove_staging_link_durably(
    temporary: &Path,
    staging: &PinnedEvidenceDirectory,
    tensor_index: u64,
) -> Result<(), Qwen36PtqDriverError> {
    staging.verify()?;
    fs::remove_file(temporary)
        .map_err(|error| evidence_io("remove evidence staging link", Some(tensor_index), error))?;
    staging.verify()?;
    staging.sync(tensor_index)
}

struct PinnedEvidenceDirectory {
    path: PathBuf,
    file: File,
    identity: EvidenceDirectoryIdentity,
}

impl PinnedEvidenceDirectory {
    fn open(path: PathBuf) -> Result<Self, Qwen36PtqDriverError> {
        let path_metadata = fs::symlink_metadata(&path)
            .map_err(|error| evidence_io("inspect evidence staging directory", None, error))?;
        if path_metadata.file_type().is_symlink() || !path_metadata.is_dir() {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "staging directory type",
            ));
        }
        let file = File::open(&path)
            .map_err(|error| evidence_io("open evidence staging directory", None, error))?;
        let opened_metadata = file
            .metadata()
            .map_err(|error| evidence_io("inspect opened staging directory", None, error))?;
        if !opened_metadata.is_dir() || !same_file_identity(&path_metadata, &opened_metadata) {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "staging directory identity changed",
            ));
        }
        Ok(Self {
            path,
            file,
            identity: evidence_directory_identity(&opened_metadata),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn verify(&self) -> Result<(), Qwen36PtqDriverError> {
        let path_metadata = fs::symlink_metadata(&self.path)
            .map_err(|error| evidence_io("reinspect evidence staging directory", None, error))?;
        let opened_metadata = self
            .file
            .metadata()
            .map_err(|error| evidence_io("reinspect opened staging directory", None, error))?;
        if path_metadata.file_type().is_symlink()
            || !path_metadata.is_dir()
            || !opened_metadata.is_dir()
            || evidence_directory_identity(&path_metadata) != self.identity
            || evidence_directory_identity(&opened_metadata) != self.identity
        {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "staging directory identity changed",
            ));
        }
        Ok(())
    }

    fn sync(&self, tensor_index: u64) -> Result<(), Qwen36PtqDriverError> {
        sync_evidence_directory_file(&self.file).map_err(|error| {
            evidence_io("sync evidence staging directory", Some(tensor_index), error)
        })
    }
}

struct VerifiedEvidenceFile {
    receipt: SaltV2KroneckerEvidenceReceipt,
    file: File,
}

fn read_evidence_path(
    path: &Path,
    tensor_index: u64,
    max_record_bytes: u64,
) -> Result<SaltV2KroneckerEvidence, Qwen36PtqDriverError> {
    let path_metadata = fs::symlink_metadata(path)
        .map_err(|error| evidence_io("inspect evidence record", Some(tensor_index), error))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "record path type",
        });
    }
    if path_metadata.len() > max_record_bytes {
        return Err(Qwen36PtqDriverError::Evidence {
            tensor_index,
            source: SaltV2KroneckerEvidenceError::SizeLimitExceeded {
                max_bytes: max_record_bytes,
            },
        });
    }

    let mut file = File::open(path)
        .map_err(|error| evidence_io("open evidence record", Some(tensor_index), error))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| evidence_io("inspect opened evidence", Some(tensor_index), error))?;
    if !opened_metadata.is_file() || !same_file_identity(&path_metadata, &opened_metadata) {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "stable record identity",
        });
    }
    let record =
        SaltV2KroneckerEvidence::read_from(&mut file, max_record_bytes).map_err(|source| {
            Qwen36PtqDriverError::Evidence {
                tensor_index,
                source,
            }
        })?;
    let final_metadata = file
        .metadata()
        .map_err(|error| evidence_io("reinspect evidence record", Some(tensor_index), error))?;
    if !same_file_identity(&opened_metadata, &final_metadata)
        || opened_metadata.len() != final_metadata.len()
    {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "record changed while reading",
        });
    }
    #[cfg(test)]
    pause_before_final_path_check(path);
    let final_path_metadata = fs::symlink_metadata(path)
        .map_err(|error| evidence_io("reinspect evidence path", Some(tensor_index), error))?;
    if final_path_metadata.file_type().is_symlink()
        || !final_path_metadata.is_file()
        || !same_file_identity(&opened_metadata, &final_path_metadata)
        || opened_metadata.len() != final_path_metadata.len()
    {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "record path changed while reading",
        });
    }
    Ok(record)
}

#[cfg(test)]
fn pause_before_final_path_check(path: &Path) {
    let mut hook = FINAL_PATH_CHECK_BARRIERS
        .lock()
        .expect("final-path test hook lock poisoned");
    let matches_path = hook
        .as_ref()
        .is_some_and(|(expected_path, _, _)| expected_path == path);
    let barriers = matches_path.then(|| hook.take().expect("matched final-path test hook"));
    drop(hook);
    if let Some((_, ready, resume)) = barriers {
        ready.wait();
        resume.wait();
    }
}

#[cfg(test)]
fn pause_before_staged_path_link(path: &Path) {
    let mut hook = STAGED_PATH_LINK_BARRIERS
        .lock()
        .expect("staged-path test hook lock poisoned");
    let matches_directory = hook
        .as_ref()
        .is_some_and(|(directory, _, _)| path.parent().is_some_and(|parent| parent == directory));
    let barriers = matches_directory.then(|| hook.take().expect("matched staged-path test hook"));
    drop(hook);
    if let Some((_, ready, resume)) = barriers {
        ready.wait();
        resume.wait();
    }
}

#[cfg(test)]
fn pause_before_staging_file_create(path: &Path) {
    let mut hook = STAGING_DIRECTORY_CREATE_BARRIERS
        .lock()
        .expect("staging-directory test hook lock poisoned");
    let matches_path = hook
        .as_ref()
        .is_some_and(|(expected_path, _, _)| expected_path == path);
    let barriers = matches_path.then(|| hook.take().expect("matched staging-directory test hook"));
    drop(hook);
    if let Some((_, ready, resume)) = barriers {
        ready.wait();
        resume.wait();
    }
}

fn verify_written_evidence_path(
    expected: &SaltV2KroneckerEvidence,
    path: &Path,
    tensor_index: u64,
    max_record_bytes: u64,
) -> Result<VerifiedEvidenceFile, Qwen36PtqDriverError> {
    let path_metadata = fs::symlink_metadata(path)
        .map_err(|error| evidence_io("inspect staged evidence", Some(tensor_index), error))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "staged record path type",
        });
    }
    if path_metadata.len() > max_record_bytes {
        return Err(Qwen36PtqDriverError::Evidence {
            tensor_index,
            source: SaltV2KroneckerEvidenceError::SizeLimitExceeded {
                max_bytes: max_record_bytes,
            },
        });
    }
    let mut file = File::open(path)
        .map_err(|error| evidence_io("open staged evidence", Some(tensor_index), error))?;
    let opened_metadata = file.metadata().map_err(|error| {
        evidence_io("inspect opened staged evidence", Some(tensor_index), error)
    })?;
    if !opened_metadata.is_file() || !same_file_identity(&path_metadata, &opened_metadata) {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "stable staged record identity",
        });
    }
    let receipt =
        expected
            .verify_written(&mut file)
            .map_err(|source| Qwen36PtqDriverError::Evidence {
                tensor_index,
                source,
            })?;
    let final_metadata = file
        .metadata()
        .map_err(|error| evidence_io("reinspect staged evidence", Some(tensor_index), error))?;
    let final_path_metadata = fs::symlink_metadata(path)
        .map_err(|error| evidence_io("reinspect staged path", Some(tensor_index), error))?;
    if !same_file_identity(&opened_metadata, &final_metadata)
        || opened_metadata.len() != final_metadata.len()
        || final_path_metadata.file_type().is_symlink()
        || !final_path_metadata.is_file()
        || !same_file_identity(&opened_metadata, &final_path_metadata)
        || opened_metadata.len() != final_path_metadata.len()
        || receipt.bytes() != opened_metadata.len()
    {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "staged record changed while reading",
        });
    }
    Ok(VerifiedEvidenceFile { receipt, file })
}

fn verify_published_evidence_inode(
    destination: &Path,
    staged_file: &File,
    tensor_index: u64,
) -> Result<(), Qwen36PtqDriverError> {
    let staged_metadata = staged_file
        .metadata()
        .map_err(|error| evidence_io("reinspect verified evidence", Some(tensor_index), error))?;
    let destination_metadata = fs::symlink_metadata(destination).map_err(|error| {
        evidence_io(
            "inspect published evidence inode",
            Some(tensor_index),
            error,
        )
    })?;
    if destination_metadata.file_type().is_symlink()
        || !destination_metadata.is_file()
        || !same_file_identity(&staged_metadata, &destination_metadata)
        || staged_metadata.len() != destination_metadata.len()
    {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "published inode identity",
        });
    }
    Ok(())
}

#[cfg(unix)]
const fn preflight_evidence_directory_sync(_tensor_index: u64) -> Result<(), Qwen36PtqDriverError> {
    Ok(())
}

#[cfg(not(unix))]
fn preflight_evidence_directory_sync(tensor_index: u64) -> Result<(), Qwen36PtqDriverError> {
    Err(evidence_io(
        "preflight durable evidence publication",
        Some(tensor_index),
        io::Error::new(
            io::ErrorKind::Unsupported,
            "durable evidence-directory sync is unavailable on this platform",
        ),
    ))
}

#[cfg(unix)]
fn evidence_directory_identity(metadata: &fs::Metadata) -> EvidenceDirectoryIdentity {
    use std::os::unix::fs::MetadataExt;
    EvidenceDirectoryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

#[cfg(not(unix))]
const fn evidence_directory_identity(_metadata: &fs::Metadata) -> EvidenceDirectoryIdentity {
    EvidenceDirectoryIdentity {}
}

#[cfg(unix)]
fn sync_evidence_directory(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    sync_evidence_directory_file(&file)
}

#[cfg(unix)]
fn sync_evidence_directory_file(file: &File) -> io::Result<()> {
    #[cfg(test)]
    if FAIL_EVIDENCE_DIRECTORY_SYNC_AFTER.with(|remaining| {
        let current = remaining.get();
        if current == 0 {
            false
        } else {
            remaining.set(current - 1);
            current == 1
        }
    }) {
        return Err(io::Error::other("injected evidence-directory sync failure"));
    }
    file.sync_all()
}

#[cfg(not(unix))]
fn sync_evidence_directory(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "durable evidence-directory sync is unavailable on this platform",
    ))
}

#[cfg(not(unix))]
fn sync_evidence_directory_file(_file: &File) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "durable evidence-directory sync is unavailable on this platform",
    ))
}

/// Exact physical ceilings for the two nested deployable profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen36PtqPackageLimits {
    compact: PhysicalBytes,
    near_lossless: PhysicalBytes,
}

impl Qwen36PtqPackageLimits {
    /// Construct componentwise serialized and indexed-runtime ceilings.
    #[must_use]
    pub const fn new(compact: PhysicalBytes, near_lossless: PhysicalBytes) -> Self {
        Self {
            compact,
            near_lossless,
        }
    }

    /// CompactV1 serialized and indexed-runtime ceilings.
    #[must_use]
    pub const fn compact(self) -> PhysicalBytes {
        self.compact
    }

    /// NearLosslessV1 serialized and indexed-runtime ceilings.
    #[must_use]
    pub const fn near_lossless(self) -> PhysicalBytes {
        self.near_lossless
    }
}

#[cfg(unix)]
/// Owned proof that fitting, exact allocation, admission, and export completed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PtqPackagesReceipt {
    completion: Qwen36CompleteWorkspaceReceipt,
    admission: Qwen36PackageAdmissionReceipt,
}

#[cfg(unix)]
impl Qwen36PtqPackagesReceipt {
    /// Complete verified rate-free language-plus-MTP master campaign.
    #[must_use]
    pub const fn completion(&self) -> &Qwen36CompleteWorkspaceReceipt {
        &self.completion
    }

    /// Exact durable admission for both selected SALT V2 packages.
    #[must_use]
    pub const fn admission(&self) -> &Qwen36PackageAdmissionReceipt {
        &self.admission
    }
}

#[cfg(unix)]
/// Failure while reconciling and exporting exact Qwen3.6 PTQ packages.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36PtqPackageError {
    /// Evidence, source, fitting, or master-campaign reconciliation failed.
    Driver(Qwen36PtqDriverError),
    /// Exact physical allocation failed.
    Physical(Qwen36PhysicalAllocationError),
    /// Package materialization or durable admission failed.
    Admission(Qwen36PackageAdmissionError),
    /// A caller-owned package output rejected bytes or failed to flush.
    Output {
        /// Profile being exported.
        profile: SaltV2Profile,
        /// Stable output operation.
        operation: &'static str,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
}

#[cfg(unix)]
impl fmt::Display for Qwen36PtqPackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(formatter, "Qwen3.6 PTQ reconciliation failed: {error}"),
            Self::Physical(error) => write!(formatter, "Qwen3.6 PTQ allocation failed: {error}"),
            Self::Admission(error) => {
                write!(formatter, "Qwen3.6 package admission failed: {error}")
            }
            Self::Output {
                profile,
                operation,
                kind,
            } => write!(
                formatter,
                "Qwen3.6 {profile:?} package {operation} failed: {kind:?}"
            ),
        }
    }
}

#[cfg(unix)]
impl std::error::Error for Qwen36PtqPackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Physical(error) => Some(error),
            Self::Admission(error) => Some(error),
            Self::Output { .. } => None,
        }
    }
}

#[cfg(unix)]
impl From<Qwen36PtqDriverError> for Qwen36PtqPackageError {
    fn from(error: Qwen36PtqDriverError) -> Self {
        Self::Driver(error)
    }
}

#[cfg(unix)]
impl From<Qwen36PhysicalAllocationError> for Qwen36PtqPackageError {
    fn from(error: Qwen36PhysicalAllocationError) -> Self {
        Self::Physical(error)
    }
}

#[cfg(unix)]
impl From<Qwen36PackageAdmissionError> for Qwen36PtqPackageError {
    fn from(error: Qwen36PackageAdmissionError) -> Self {
        Self::Admission(error)
    }
}

/// Reconcile the pinned Qwen3.6 language-plus-MTP pure-PTQ campaign.
///
/// The driver first plans the immutable 506-master catalog by widening only one
/// admitted source matrix and
/// reopening only one evidence record at a time. It opens the content-addressed
/// campaign after the exact-BF16 base workspace resumes, skips every strictly
/// valid existing master, fits missing masters directly into unpublished store
/// writers, and seals only after all 506 canonical records reopen successfully.
///
/// # Errors
/// Fails closed on source mutation, evidence mismatch/corruption, recipe or fit
/// failure, store conflict, incomplete output, or any campaign validation error.
pub fn reconcile_qwen36_ptq(
    admitted: &Qwen36AdmittedSource,
    evidence: &Qwen36PtqEvidenceDirectory,
    config: &SaltV2Config,
) -> Result<Qwen36CompleteWorkspaceReceipt, Qwen36PtqDriverError> {
    with_reconciled_qwen36_ptq_campaign(admitted, evidence, config, |_, receipt| Ok(receipt))
}

#[cfg(unix)]
/// Reconcile masters, allocate two exact profiles, and export admitted packages.
///
/// Both package outputs are visited from their verified content-addressed records
/// in bounded chunks. Output effects are deliberately nontransactional; callers
/// that publish files must pass staged outputs and rename them only after this
/// function returns successfully. Durable campaign and admission records remain
/// resumable when either output fails.
///
/// # Errors
/// Fails closed on every [`reconcile_qwen36_ptq`] error, incompatible codec or
/// ceiling, exact allocation failure, package admission failure, or output I/O.
pub fn reconcile_qwen36_ptq_packages(
    admitted: &Qwen36AdmittedSource,
    evidence: &Qwen36PtqEvidenceDirectory,
    config: &SaltV2Config,
    limits: Qwen36PtqPackageLimits,
    mut compact_output: impl Write,
    mut near_lossless_output: impl Write,
) -> Result<Qwen36PtqPackagesReceipt, Qwen36PtqPackageError> {
    with_reconciled_qwen36_ptq_campaign(admitted, evidence, config, |campaign, completion| {
        let codec = packing_codec(config.packing);
        let spec = Qwen36SelectedAllocationSpec::for_uniform_full_tiles(
            codec,
            physical_allocator_id(),
            physical_allocation_recipe_id(codec, limits, &completion),
            campaign.spec().expected_masters(),
            limits.compact(),
            limits.near_lossless(),
        )?;
        let allocated = campaign.reopen_or_allocate_selected_allocation(spec)?;
        let admitted_packages = allocated.reopen_or_materialize_packages()?;
        export_admitted_package(
            &admitted_packages,
            SaltV2Profile::CompactV1,
            &mut compact_output,
            false,
        )?;
        export_admitted_package(
            &admitted_packages,
            SaltV2Profile::NearLosslessV1,
            &mut near_lossless_output,
            true,
        )?;
        Ok(Qwen36PtqPackagesReceipt {
            completion,
            admission: admitted_packages.receipt().clone(),
        })
    })
}

fn with_reconciled_qwen36_ptq_campaign<R, E>(
    admitted: &Qwen36AdmittedSource,
    evidence: &Qwen36PtqEvidenceDirectory,
    config: &SaltV2Config,
    finish: impl FnOnce(
        &crate::Qwen36AdditiveCampaignStore<'_, '_>,
        Qwen36CompleteWorkspaceReceipt,
    ) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<Qwen36PtqDriverError>,
{
    let workspace =
        Qwen36TensorWorkStore::open(admitted).map_err(Qwen36PtqDriverError::Workspace)?;
    evidence.validate_complete(
        u64::try_from(workspace.additive_slots().len())
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?,
    )?;
    let source_model_id = admitted.proof().source_model_id();
    workspace
        .reconcile_preserved()
        .map_err(Qwen36PtqDriverError::Workspace)?;

    #[cfg(unix)]
    for spec in workspace
        .find_existing_ptq_campaign_specs()
        .map_err(Qwen36PtqDriverError::Workspace)?
    {
        if !existing_ptq_campaign_matches_evidence(
            &workspace,
            evidence,
            &spec,
            config,
            source_model_id,
        )? {
            continue;
        }
        let campaign = workspace
            .open_master_campaign(spec)
            .map_err(Qwen36PtqDriverError::Workspace)?;
        if campaign
            .completion_path_is_present()
            .map_err(Qwen36PtqDriverError::Workspace)?
        {
            let receipt = campaign
                .reopen_complete_current()
                .map_err(Qwen36PtqDriverError::Workspace)?;
            return finish(&campaign, receipt);
        }
    }

    let weight_spool = Qwen36WeightSpool::create(workspace.root())?;

    let mut expected_masters = Vec::new();
    expected_masters
        .try_reserve_exact(workspace.additive_slots().len())
        .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
    let mut activation_digest = None;
    let mut token_stream_digest = None;
    for (ordinal, slot) in workspace.additive_slots().iter().enumerate() {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let record = evidence.reopen(tensor_index)?;
        validate_record(slot, tensor_index, source_model_id, &record)?;
        bind_activation_cache(&mut activation_digest, tensor_index, &record)?;
        bind_token_stream(&mut token_stream_digest, tensor_index, &record)?;
        let weights =
            admitted
                .tensor_f32(slot.name())
                .map_err(|source| Qwen36PtqDriverError::Source {
                    tensor_index,
                    source,
                })?;
        weight_spool.write(tensor_index, &weights)?;
        let input = restart_input(slot, tensor_index, source_model_id, &weights, &record)?;
        expected_masters.push(
            plan_salt_v2_restartable_tensor_master(input, config).map_err(|source| {
                Qwen36PtqDriverError::Fit {
                    tensor_index,
                    source,
                }
            })?,
        );
    }

    let spec = Qwen36AdditiveCampaignSpec::new(expected_masters)
        .map_err(Qwen36PtqDriverError::Workspace)?;
    let campaign = workspace
        .open_master_campaign(spec)
        .map_err(Qwen36PtqDriverError::Workspace)?;
    if campaign
        .completion_path_is_present()
        .map_err(Qwen36PtqDriverError::Workspace)?
    {
        let receipt = campaign
            .reopen_complete_current()
            .map_err(Qwen36PtqDriverError::Workspace)?;
        return finish(&campaign, receipt);
    }
    for (ordinal, expected) in campaign.spec().expected_masters().iter().enumerate() {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let slot = &workspace.additive_slots()[ordinal];
        campaign
            .install_master(expected, |writer| {
                let record = evidence.reopen(tensor_index)?;
                validate_record(slot, tensor_index, source_model_id, &record)?;
                let weights = weight_spool.read(tensor_index)?;
                let input = restart_input(slot, tensor_index, source_model_id, &weights, &record)?;
                let result = fit_salt_v2_restartable_tensor_master(input, config, writer).map_err(
                    |source| Qwen36PtqDriverError::Fit {
                        tensor_index,
                        source,
                    },
                )?;
                if result.spec() != expected {
                    return Err(Qwen36PtqDriverError::EvidenceMismatch {
                        tensor_index,
                        field: "planned master specification",
                    });
                }
                Ok(())
            })
            .map_err(map_install_error)?;
    }
    let receipt = campaign
        .seal_complete()
        .map_err(Qwen36PtqDriverError::Workspace)?;
    finish(&campaign, receipt)
}

#[cfg(unix)]
fn existing_ptq_campaign_matches_evidence(
    workspace: &Qwen36TensorWorkStore<'_>,
    evidence: &Qwen36PtqEvidenceDirectory,
    spec: &Qwen36AdditiveCampaignSpec,
    config: &SaltV2Config,
    source_model_id: ModelId,
) -> Result<bool, Qwen36PtqDriverError> {
    let expected_recipe = config.master_recipe_id();
    let expected_constraint = config.packing.fit_constraint();
    let expected_max_planes =
        u8::try_from(config.max_planes).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
    let mut activation_digest = None;
    let mut token_stream_digest = None;
    for (ordinal, (slot, expected)) in workspace
        .additive_slots()
        .iter()
        .zip(spec.expected_masters())
        .enumerate()
    {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let record = evidence.reopen(tensor_index)?;
        validate_record(slot, tensor_index, source_model_id, &record)?;
        bind_activation_cache(&mut activation_digest, tensor_index, &record)?;
        bind_token_stream(&mut token_stream_digest, tensor_index, &record)?;
        let master_evidence = expected.evidence();
        if master_evidence.recipe_id != expected_recipe
            || master_evidence.activation_digest != record.source_id().activation_cache_digest()
            || master_evidence.curvature_digest != record.artifact().digest()
            || expected.geometry().constraint != expected_constraint
            || expected.geometry().max_planes != expected_max_planes
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn packing_codec(packing: SaltV2Packing) -> SaltV2Codec {
    match packing {
        SaltV2Packing::D2 => SaltV2Codec::D2,
        SaltV2Packing::B3 => SaltV2Codec::B3,
        SaltV2Packing::S34 => SaltV2Codec::S34,
    }
}

fn physical_allocator_id() -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(ALLOCATOR_ID_CONTEXT);
    hasher.update(ALLOCATOR_SCHEMA);
    *hasher.finalize().as_bytes()
}

fn physical_allocation_recipe_id(
    codec: SaltV2Codec,
    limits: Qwen36PtqPackageLimits,
    completion: &Qwen36CompleteWorkspaceReceipt,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(ALLOCATION_RECIPE_CONTEXT);
    hasher.update(completion.completion_id().as_bytes());
    hasher.update(&completion.master_set_id());
    hasher.update(&[match codec {
        SaltV2Codec::D2 => 1,
        SaltV2Codec::B3 => 2,
        SaltV2Codec::S34 => 3,
        _ => 0,
    }]);
    for bytes in [limits.compact(), limits.near_lossless()] {
        hasher.update(&bytes.serialized.to_le_bytes());
        hasher.update(&bytes.resident.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(unix)]
fn export_admitted_package(
    admitted: &crate::Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    profile: SaltV2Profile,
    output: &mut impl Write,
    postcheck: bool,
) -> Result<(), Qwen36PtqPackageError> {
    let visit = if postcheck {
        admitted.try_visit_package(profile, 64 * 1024, |chunk| output.write_all(chunk))
    } else {
        admitted.try_visit_package_without_postcheck(profile, 64 * 1024, |chunk| {
            output.write_all(chunk)
        })
    };
    visit.map_err(|error| match error {
        Qwen36PackageVisitError::Admission(error) => Qwen36PtqPackageError::Admission(error),
        Qwen36PackageVisitError::Sink(error) => Qwen36PtqPackageError::Output {
            profile,
            operation: "write",
            kind: error.kind(),
        },
    })?;
    output
        .flush()
        .map_err(|error| Qwen36PtqPackageError::Output {
            profile,
            operation: "flush",
            kind: error.kind(),
        })
}

fn validate_record(
    slot: &Qwen36AdditiveWorkSlot,
    tensor_index: u64,
    source_model_id: ModelId,
    record: &SaltV2KroneckerEvidence,
) -> Result<(), Qwen36PtqDriverError> {
    let shape = slot.shape();
    let rows = shape
        .first()
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    let columns = shape
        .get(1)
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    if shape.len() != 2 || rows != Some(record.rows()) || columns != Some(record.columns()) {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "tensor geometry",
        });
    }
    if record.tensor_index() != tensor_index {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "global tensor ordinal",
        });
    }
    if record.tensor_name() != slot.name() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "tensor name",
        });
    }
    if record.source_id().source_model_digest() != *source_model_id.as_bytes() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "source model identity",
        });
    }
    Ok(())
}

fn bind_token_stream(
    expected: &mut Option<[u8; 32]>,
    tensor_index: u64,
    record: &SaltV2KroneckerEvidence,
) -> Result<(), Qwen36PtqDriverError> {
    let actual = record.source_id().token_stream_digest();
    match expected {
        Some(expected) if *expected != actual => Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "campaign token stream",
        }),
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual);
            Ok(())
        }
    }
}

fn bind_activation_cache(
    expected: &mut Option<[u8; 32]>,
    tensor_index: u64,
    record: &SaltV2KroneckerEvidence,
) -> Result<(), Qwen36PtqDriverError> {
    let actual = record.source_id().activation_cache_digest();
    match expected {
        Some(expected) if *expected != actual => Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "campaign activation cache",
        }),
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual);
            Ok(())
        }
    }
}

fn restart_input<'a>(
    slot: &'a Qwen36AdditiveWorkSlot,
    tensor_index: u64,
    source_model_id: ModelId,
    weights: &'a [f32],
    record: &'a SaltV2KroneckerEvidence,
) -> Result<SaltV2RestartableTensorMasterFitInput<'a>, Qwen36PtqDriverError> {
    let tensor =
        record
            .tensor_fit_input(weights)
            .map_err(|source| Qwen36PtqDriverError::Evidence {
                tensor_index,
                source,
            })?;
    Ok(SaltV2RestartableTensorMasterFitInput {
        tensor,
        source_model_id,
        tensor_index,
        source_tensor_digest: *slot.source_tensor_digest(),
    })
}

/// Disk-backed source-weight handoff between PTQ planning and fitting.
///
/// Planning must materialize every master specification before the resumable
/// campaign can open. Keeping all widened tensors in memory is unsafe for a
/// 27B model, while widening the source a second time multiplies SafeTensors
/// range reads. This short-lived spool keeps one tensor at a time in memory and
/// makes the second phase read local bytes instead of reopening model shards.
struct Qwen36WeightSpool {
    root: PathBuf,
}

impl Qwen36WeightSpool {
    fn create(workspace_root: &Path) -> Result<Self, Qwen36PtqDriverError> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| {
                evidence_io(
                    "create source-weight spool",
                    None,
                    io::Error::other("clock before epoch"),
                )
            })?
            .as_nanos();
        let root = workspace_root.join(format!(".ptq-weight-spool-{}-{nonce}", std::process::id()));
        fs::create_dir(&root)
            .map_err(|error| evidence_io("create source-weight spool", None, error))?;
        Ok(Self { root })
    }

    fn path(&self, tensor_index: u64) -> PathBuf {
        self.root.join(format!("{tensor_index:06}.f32"))
    }

    fn write(&self, tensor_index: u64, weights: &[f32]) -> Result<(), Qwen36PtqDriverError> {
        let path = self.path(tensor_index);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                evidence_io(
                    "create source-weight spool record",
                    Some(tensor_index),
                    error,
                )
            })?;
        let count =
            u64::try_from(weights.len()).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        file.write_all(&count.to_le_bytes()).map_err(|error| {
            evidence_io(
                "write source-weight spool header",
                Some(tensor_index),
                error,
            )
        })?;
        #[cfg(target_endian = "little")]
        {
            let byte_len = weights
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
            // SAFETY: f32 is a plain four-byte value and the slice is read-only
            // for the duration of this write. Little-endian hosts preserve the
            // canonical on-disk representation directly.
            let bytes =
                unsafe { std::slice::from_raw_parts(weights.as_ptr().cast::<u8>(), byte_len) };
            file.write_all(bytes).map_err(|error| {
                evidence_io(
                    "write source-weight spool payload",
                    Some(tensor_index),
                    error,
                )
            })?;
        }
        #[cfg(target_endian = "big")]
        for weight in weights {
            file.write_all(&weight.to_le_bytes()).map_err(|error| {
                evidence_io(
                    "write source-weight spool payload",
                    Some(tensor_index),
                    error,
                )
            })?;
        }
        file.sync_all().map_err(|error| {
            evidence_io("sync source-weight spool record", Some(tensor_index), error)
        })?;
        Ok(())
    }

    fn read(&self, tensor_index: u64) -> Result<Vec<f32>, Qwen36PtqDriverError> {
        let path = self.path(tensor_index);
        let mut file = File::open(&path).map_err(|error| {
            evidence_io("open source-weight spool record", Some(tensor_index), error)
        })?;
        let metadata = file.metadata().map_err(|error| {
            evidence_io(
                "inspect source-weight spool record",
                Some(tensor_index),
                error,
            )
        })?;
        let mut count_bytes = [0_u8; 8];
        file.read_exact(&mut count_bytes).map_err(|error| {
            evidence_io("read source-weight spool header", Some(tensor_index), error)
        })?;
        let count = usize::try_from(u64::from_le_bytes(count_bytes))
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let payload_bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        let expected_bytes = 8_u64
            .checked_add(
                u64::try_from(payload_bytes).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?,
            )
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        if metadata.len() != expected_bytes {
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "source-weight spool length",
            });
        }
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(payload_bytes)
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        payload.resize(payload_bytes, 0);
        file.read_exact(&mut payload).map_err(|error| {
            evidence_io(
                "read source-weight spool payload",
                Some(tensor_index),
                error,
            )
        })?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(count)
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        for chunk in payload.chunks_exact(4) {
            weights.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(weights)
    }
}

impl Drop for Qwen36WeightSpool {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn map_install_error(
    error: Qwen36AdditiveInstallError<Qwen36PtqDriverError>,
) -> Qwen36PtqDriverError {
    match error {
        Qwen36AdditiveInstallError::Campaign(source) => Qwen36PtqDriverError::Workspace(source),
        Qwen36AdditiveInstallError::Producer(source) => source,
    }
}

fn evidence_io(
    operation: &'static str,
    tensor_index: Option<u64>,
    error: io::Error,
) -> Qwen36PtqDriverError {
    Qwen36PtqDriverError::Io {
        operation,
        tensor_index,
        kind: error.kind(),
    }
}

/// Failure while reopening evidence or reconciling the Qwen3.6 PTQ campaign.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36PtqDriverError {
    /// Evidence root or record path violated the fixed filesystem policy.
    InvalidEvidencePath(&'static str),
    /// Evidence-directory or evidence-record I/O failed.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Global tensor ordinal, absent for directory-level operations.
        tensor_index: Option<u64>,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
    /// Canonical curvature evidence failed bounded decoding or validation.
    Evidence {
        /// Global additive-tensor ordinal.
        tensor_index: u64,
        /// Typed evidence failure.
        source: SaltV2KroneckerEvidenceError,
    },
    /// Streamed curvature production failed before publication.
    EvidenceBuild {
        /// Global additive-tensor ordinal.
        tensor_index: u64,
        /// Typed producer failure.
        source: SaltV2KroneckerEvidenceBuildError,
    },
    /// Reopened evidence contradicted the admitted Qwen slot.
    EvidenceMismatch {
        /// Global additive-tensor ordinal.
        tensor_index: u64,
        /// Stable mismatched field label.
        field: &'static str,
    },
    /// Same-handle admitted source widening failed.
    Source {
        /// Global additive-tensor ordinal.
        tensor_index: u64,
        /// Typed preflight/source failure.
        source: Qwen36CampaignPreflightError,
    },
    /// Pure-PTQ planning or fitting failed.
    Fit {
        /// Global additive-tensor ordinal.
        tensor_index: u64,
        /// Typed SALT V2 failure.
        source: SaltV2Error,
    },
    /// Base workspace, campaign store, or completion seal failed.
    Workspace(Qwen36TensorWorkError),
    /// Bounded metadata allocation failed.
    AllocationFailed,
}

impl fmt::Display for Qwen36PtqDriverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEvidencePath(field) => write!(formatter, "invalid PTQ evidence {field}"),
            Self::Io {
                operation,
                tensor_index,
                kind,
            } => match tensor_index {
                Some(index) => write!(
                    formatter,
                    "Qwen3.6 PTQ tensor {index} {operation} failed: {kind:?}"
                ),
                None => write!(formatter, "Qwen3.6 PTQ {operation} failed: {kind:?}"),
            },
            Self::Evidence {
                tensor_index,
                source,
            } => write!(
                formatter,
                "Qwen3.6 PTQ tensor {tensor_index} evidence failed: {source}"
            ),
            Self::EvidenceBuild {
                tensor_index,
                source,
            } => write!(
                formatter,
                "Qwen3.6 PTQ tensor {tensor_index} evidence build failed: {source}"
            ),
            Self::EvidenceMismatch {
                tensor_index,
                field,
            } => write!(
                formatter,
                "Qwen3.6 PTQ tensor {tensor_index} evidence mismatches {field}"
            ),
            Self::Source {
                tensor_index,
                source,
            } => write!(
                formatter,
                "Qwen3.6 PTQ tensor {tensor_index} source failed: {source}"
            ),
            Self::Fit {
                tensor_index,
                source,
            } => write!(
                formatter,
                "Qwen3.6 PTQ tensor {tensor_index} fit failed: {source}"
            ),
            Self::Workspace(source) => write!(formatter, "Qwen3.6 PTQ workspace failed: {source}"),
            Self::AllocationFailed => formatter.write_str("Qwen3.6 PTQ allocation failed"),
        }
    }
}

impl std::error::Error for Qwen36PtqDriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evidence { source, .. } => Some(source),
            Self::EvidenceBuild { source, .. } => Some(source),
            Self::Source { source, .. } => Some(source),
            Self::Fit { source, .. } => Some(source),
            Self::Workspace(source) => Some(source),
            Self::InvalidEvidencePath(_)
            | Self::Io { .. }
            | Self::EvidenceMismatch { .. }
            | Self::AllocationFailed => None,
        }
    }
}

// Evidence installation asserts strict directory durability, which is
// unix-only by design (preflight_evidence_directory_sync fails elsewhere).
#[cfg(all(test, unix))]
mod tests {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicU64, Ordering},
    };

    use tritium_quantize::{
        CurvatureSourceId, DensePsdMetric, PhysicalRateTarget, Qwen35SourceDtype, Qwen35TensorRole,
        Qwen35TensorScope, SaltV2Curvature, SaltV2KroneckerEvidenceSpec, SaltV2Packing,
    };

    use super::*;
    use crate::Qwen36AdditiveSlotState;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "tritium-qwen36-ptq-{label}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn source_weight_spool_round_trips_little_endian_f32_and_cleans_up() {
        let root = temp_root("weight-spool");
        let spool_root;
        {
            let spool = Qwen36WeightSpool::create(&root).unwrap();
            spool_root = spool.root.clone();
            spool
                .write(7, &[1.25, -0.5, 0.0, std::f32::consts::PI])
                .unwrap();
            assert_eq!(
                spool.read(7).unwrap(),
                vec![1.25, -0.5, 0.0, std::f32::consts::PI]
            );
        }
        assert!(!spool_root.exists());
        fs::remove_dir_all(root).unwrap();
    }

    fn source_model_id() -> ModelId {
        ModelId::from_digest([1; 32])
    }

    fn slot() -> Qwen36AdditiveWorkSlot {
        Qwen36AdditiveWorkSlot {
            name: "a.weight".to_owned(),
            dtype: Qwen35SourceDtype::Bfloat16,
            shape: vec![2, 128],
            coefficients: 256,
            scope: Qwen35TensorScope::Language,
            role: Qwen35TensorRole::MlpProjection,
            source_tensor_digest: [9; 32],
            state: Qwen36AdditiveSlotState::MissingCanonicalMaster,
        }
    }

    fn evidence(index: u64, name: &str, model_digest: [u8; 32]) -> SaltV2KroneckerEvidence {
        evidence_with_token(index, name, model_digest, [3; 32])
    }

    fn evidence_with_token(
        index: u64,
        name: &str,
        model_digest: [u8; 32],
        token_digest: [u8; 32],
    ) -> SaltV2KroneckerEvidence {
        evidence_with_sources(index, name, model_digest, [2; 32], token_digest)
    }

    fn evidence_with_sources(
        index: u64,
        name: &str,
        model_digest: [u8; 32],
        activation_digest: [u8; 32],
        token_digest: [u8; 32],
    ) -> SaltV2KroneckerEvidence {
        let mut values = vec![0.0; 128 * 128];
        for index in 0..128 {
            values[index * 128 + index] = 1.0;
        }
        SaltV2KroneckerEvidence::new(
            SaltV2Curvature::GuidedFisher,
            CurvatureSourceId::new(model_digest, activation_digest, token_digest).unwrap(),
            [4; 32],
            index,
            name,
            2,
            128,
            vec![DensePsdMetric::new(128, &values).unwrap()],
            vec![0.5, 1.5],
            0.125,
        )
        .unwrap()
    }

    fn config() -> SaltV2Config {
        SaltV2Config {
            curvature: SaltV2Curvature::GuidedFisher,
            packing: SaltV2Packing::B3,
            em_restarts: 1,
            coordinate_sweeps: 2,
            rate: PhysicalRateTarget {
                max_matrix_bytes: 100_000,
                max_artifact_bytes: 100_000,
                max_resident_bytes: None,
            },
            ..SaltV2Config::default()
        }
    }

    #[test]
    fn evidence_directory_reopens_only_the_canonical_bounded_record() {
        let root = temp_root("reopen");
        assert!(matches!(
            Qwen36PtqEvidenceDirectory::open(""),
            Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "empty directory path"
            ))
        ));
        let directory = Qwen36PtqEvidenceDirectory::open(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);
        let path = directory.record_path(0);
        let mut file = File::create(&path).unwrap();
        record.write_to(&mut file).unwrap();
        drop(file);

        let reopened = directory.reopen(0).unwrap();
        assert_eq!(reopened.record_digest(), record.record_digest());
        directory.validate_complete(1).unwrap();
        assert!(matches!(
            Qwen36PtqEvidenceDirectory::open_bounded(&root, 16)
                .unwrap()
                .reopen(0),
            Err(Qwen36PtqDriverError::Evidence {
                source: SaltV2KroneckerEvidenceError::SizeLimitExceeded { .. },
                ..
            })
        ));

        fs::write(root.join("README"), b"unexpected").unwrap();
        assert!(matches!(
            directory.validate_complete(1),
            Err(Qwen36PtqDriverError::InvalidEvidencePath("record filename"))
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn evidence_install_is_atomic_idempotent_and_conflict_safe() {
        let parent = temp_root("install");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);

        let first = directory.install(&record).unwrap();
        let second = directory.install(&record).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.record_digest(), record.record_digest());
        assert_eq!(
            first.bytes(),
            fs::metadata(directory.record_path(0)).unwrap().len()
        );
        assert!(root.join(EVIDENCE_STAGING_DIRECTORY).is_dir());
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
                .count(),
            1
        );
        directory.validate_complete(1).unwrap();
        fs::write(
            root.join(EVIDENCE_STAGING_DIRECTORY).join(".s2kf.orphan"),
            b"interrupted staging bytes",
        )
        .unwrap();
        directory.validate_complete(1).unwrap();

        let conflict = evidence_with_token(0, "a.weight", [1; 32], [8; 32]);
        assert!(matches!(
            directory.install(&conflict),
            Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index: 0,
                field: "immutable record identity"
            })
        ));
        assert_eq!(
            directory.reopen(0).unwrap().record_digest(),
            first.record_digest()
        );

        let bounded_root = parent.join("bounded");
        let bounded = Qwen36PtqEvidenceDirectory::create_bounded(&bounded_root, 16).unwrap();
        assert!(matches!(
            bounded.install(&record),
            Err(Qwen36PtqDriverError::Evidence {
                tensor_index: 0,
                source: SaltV2KroneckerEvidenceError::SizeLimitExceeded { max_bytes: 16 }
            })
        ));
        assert_eq!(fs::read_dir(&bounded_root).unwrap().count(), 0);

        let out_of_range = evidence(1_000_000, "a.weight", [1; 32]);
        assert!(matches!(
            directory.install(&out_of_range),
            Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index: 1_000_000,
                field: "global tensor ordinal"
            })
        ));

        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn evidence_builder_consumes_directly_into_durable_namespace() {
        let parent = temp_root("builder-install");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
        let spec = SaltV2KroneckerEvidenceSpec::new(
            SaltV2Curvature::GuidedFisher,
            source,
            0,
            "a.weight",
            2,
            128,
            0.25,
        )
        .unwrap();
        let mut builder = directory.create_builder(spec).unwrap();
        let mut activations = vec![1.0; 128];
        activations.extend(std::iter::repeat_n(3.0, 128));
        builder
            .accumulate_batch(&activations, Some(&[1.0, 2.0, 3.0, 4.0]), 2, None, None)
            .unwrap();

        let receipt = directory.install_builder(builder).unwrap();
        let reopened = directory.reopen(0).unwrap();
        assert_eq!(receipt, reopened.receipt().unwrap());
        assert_eq!(reopened.output_weights(), &[5.0, 10.0]);
        directory.validate_complete(1).unwrap();

        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn evidence_directory_rejects_oversized_builder_before_accumulator_construction() {
        let parent = temp_root("builder-preflight");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create_bounded(&root, 16).unwrap();
        let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
        let spec = SaltV2KroneckerEvidenceSpec::new(
            SaltV2Curvature::GuidedFisher,
            source,
            0,
            "a.weight",
            2,
            128,
            0.25,
        )
        .unwrap();
        let required_bytes = spec.canonical_bytes();

        assert!(matches!(
            directory.create_builder(spec),
            Err(Qwen36PtqDriverError::EvidenceBuild {
                tensor_index: 0,
                source: SaltV2KroneckerEvidenceBuildError::SizeLimitExceeded {
                    required_bytes: got,
                    max_bytes: 16
                }
            }) if got == required_bytes
        ));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);

        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn indexed_embedding_builder_installs_canonical_evidence() {
        let parent = temp_root("indexed-builder-install");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
        let spec = SaltV2KroneckerEvidenceSpec::new(
            SaltV2Curvature::GuidedFisher,
            source,
            0,
            "model.embed_tokens.weight",
            4,
            128,
            0.25,
        )
        .unwrap();
        let mut builder = directory.create_indexed_output_builder(spec).unwrap();
        let mut activations = vec![1.0; 128];
        activations.extend(std::iter::repeat_n(3.0, 128));
        builder
            .accumulate_indexed_output_batch(&activations, &[3, 1], &[2.0, -3.0], 2, None, None)
            .unwrap();

        directory.install_builder(builder).unwrap();
        let reopened = directory.reopen(0).unwrap();
        assert_eq!(reopened.output_weights(), &[0.0, 4.5, 0.0, 2.0]);

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_install_retry_makes_an_existing_link_durable() {
        let parent = temp_root("install-sync-retry");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);

        FAIL_EVIDENCE_DIRECTORY_SYNC_AFTER.with(|remaining| remaining.set(1));
        assert!(matches!(
            directory.install(&record),
            Err(Qwen36PtqDriverError::Io {
                operation: "sync evidence directory",
                tensor_index: Some(0),
                ..
            })
        ));
        assert!(directory.record_path(0).is_file());
        assert_eq!(
            directory.install(&record).unwrap(),
            record.receipt().unwrap()
        );
        assert_eq!(
            fs::read_dir(root.join(EVIDENCE_STAGING_DIRECTORY))
                .unwrap()
                .count(),
            0
        );
        directory.validate_complete(1).unwrap();

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_retry_completes_a_failed_staging_cleanup_sync() {
        let parent = temp_root("install-staging-sync-retry");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);

        FAIL_EVIDENCE_DIRECTORY_SYNC_AFTER.with(|remaining| remaining.set(2));
        assert!(matches!(
            directory.install(&record),
            Err(Qwen36PtqDriverError::Io {
                operation: "sync evidence staging directory",
                tensor_index: Some(0),
                ..
            })
        ));
        assert!(directory.record_path(0).is_file());
        assert_eq!(
            directory.install(&record).unwrap(),
            record.receipt().unwrap()
        );
        assert_eq!(
            fs::read_dir(root.join(EVIDENCE_STAGING_DIRECTORY))
                .unwrap()
                .count(),
            0
        );
        directory.validate_complete(1).unwrap();

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_identical_installers_both_observe_the_durable_record() {
        let parent = temp_root("install-concurrent");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);
        let start = Arc::new(Barrier::new(3));
        let workers = (0..2)
            .map(|_| {
                let directory = directory.clone();
                let record = record.clone();
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    directory.install(&record)
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        for worker in workers {
            assert_eq!(worker.join().unwrap().unwrap(), record.receipt().unwrap());
        }
        assert_eq!(
            fs::read_dir(root.join(EVIDENCE_STAGING_DIRECTORY))
                .unwrap()
                .count(),
            0
        );
        directory.validate_complete(1).unwrap();

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_reopen_rejects_path_replacement_after_opened_record_verifies() {
        let parent = temp_root("reopen-path-race");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);
        directory.install(&record).unwrap();

        let ready = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        let path = directory.record_path(0);
        *FINAL_PATH_CHECK_BARRIERS.lock().unwrap() =
            Some((path.clone(), Arc::clone(&ready), Arc::clone(&resume)));
        let reader = {
            let directory = directory.clone();
            std::thread::spawn(move || directory.reopen(0))
        };
        ready.wait();
        fs::rename(&path, root.join("replaced.s2kf")).unwrap();
        let replacement = evidence_with_token(0, "a.weight", [1; 32], [8; 32]);
        let mut file = File::create(&path).unwrap();
        replacement.write_to(&mut file).unwrap();
        file.sync_all().unwrap();
        drop(file);
        resume.wait();

        let result = reader.join().unwrap();
        assert!(
            matches!(
                result,
                Err(Qwen36PtqDriverError::EvidenceMismatch {
                    tensor_index: 0,
                    field: "record path changed while reading"
                })
            ),
            "unexpected reopen result: {result:?}"
        );

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_install_never_publishes_a_swapped_staging_path() {
        let parent = temp_root("install-staging-race");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);
        let staging = root.join(EVIDENCE_STAGING_DIRECTORY);
        let ready = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        *STAGED_PATH_LINK_BARRIERS.lock().unwrap() =
            Some((staging.clone(), Arc::clone(&ready), Arc::clone(&resume)));
        let installer = {
            let directory = directory.clone();
            std::thread::spawn(move || directory.install(&record))
        };
        ready.wait();
        let temporary = fs::read_dir(&staging)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.is_file())
            .expect("verified staging file");
        fs::rename(&temporary, staging.join("verified-away.s2kf")).unwrap();
        let replacement = evidence_with_token(0, "a.weight", [1; 32], [8; 32]);
        let mut file = File::create(&temporary).unwrap();
        replacement.write_to(&mut file).unwrap();
        file.sync_all().unwrap();
        drop(file);
        resume.wait();

        assert!(matches!(
            installer.join().unwrap(),
            Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index: 0,
                field: "published inode identity"
            })
        ));
        assert!(!directory.record_path(0).exists());

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_install_rejects_a_replaced_staging_directory_before_create() {
        let parent = temp_root("install-staging-directory-race");
        let root = parent.join("records");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);
        let staging = root.join(EVIDENCE_STAGING_DIRECTORY);
        let ready = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        *STAGING_DIRECTORY_CREATE_BARRIERS.lock().unwrap() =
            Some((staging.clone(), Arc::clone(&ready), Arc::clone(&resume)));
        let installer = {
            let directory = directory.clone();
            std::thread::spawn(move || directory.install(&record))
        };
        ready.wait();
        fs::rename(&staging, root.join("pinned-staging-away")).unwrap();
        fs::create_dir(&staging).unwrap();
        resume.wait();

        assert!(matches!(
            installer.join().unwrap(),
            Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "staging directory identity changed"
            ))
        ));
        assert!(!directory.record_path(0).exists());

        fs::remove_dir_all(parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_handle_rejects_replaced_root_before_read_or_publish() {
        let parent = temp_root("replaced-root");
        let root = parent.join("records");
        let replaced = parent.join("replaced");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        fs::rename(&root, &replaced).unwrap();
        fs::create_dir(&root).unwrap();
        let record = evidence(0, "a.weight", [1; 32]);

        assert!(matches!(
            directory.install(&record),
            Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "directory identity changed"
            ))
        ));
        assert!(matches!(
            directory.validate_complete(1),
            Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "directory identity changed"
            ))
        ));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);

        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn admitted_slot_and_record_plan_a_restartable_master() {
        let slot = slot();
        let record = evidence(0, slot.name(), [1; 32]);
        validate_record(&slot, 0, source_model_id(), &record).unwrap();
        let weights = (0..256)
            .map(|index| (index as f32 - 127.0) / 61.0)
            .collect::<Vec<_>>();
        let input = restart_input(&slot, 0, source_model_id(), &weights, &record).unwrap();
        let spec = plan_salt_v2_restartable_tensor_master(input, &config()).unwrap();
        assert_eq!(spec.name(), slot.name());
        assert_eq!(spec.tensor_index(), 0);
        assert_eq!(spec.shape(), slot.shape());
        assert_eq!(spec.source_tensor_digest(), slot.source_tensor_digest());
    }

    #[test]
    fn record_slot_mismatches_fail_before_source_widening() {
        let slot = slot();
        for (record, field) in [
            (evidence(1, slot.name(), [1; 32]), "global tensor ordinal"),
            (evidence(0, "b.weight", [1; 32]), "tensor name"),
            (evidence(0, slot.name(), [8; 32]), "source model identity"),
        ] {
            assert!(matches!(
                validate_record(&slot, 0, source_model_id(), &record),
                Err(Qwen36PtqDriverError::EvidenceMismatch {
                    tensor_index: 0,
                    field: got,
                }) if got == field
            ));
        }

        let mut expected = None;
        bind_token_stream(&mut expected, 0, &evidence(0, slot.name(), [1; 32])).unwrap();
        let changed = evidence_with_token(1, "b.weight", [1; 32], [7; 32]);
        assert!(matches!(
            bind_token_stream(&mut expected, 1, &changed),
            Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index: 1,
                field: "campaign token stream",
            })
        ));

        let mut expected_activation = None;
        bind_activation_cache(
            &mut expected_activation,
            0,
            &evidence(0, slot.name(), [1; 32]),
        )
        .unwrap();
        let changed_activation = evidence_with_sources(1, "b.weight", [1; 32], [8; 32], [3; 32]);
        assert!(matches!(
            bind_activation_cache(&mut expected_activation, 1, &changed_activation),
            Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index: 1,
                field: "campaign activation cache",
            })
        ));
    }
}
