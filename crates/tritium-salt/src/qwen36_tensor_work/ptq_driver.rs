//! Checkpoint-backed, resumable Qwen3.6 pure-PTQ campaign driver.

use core::fmt;
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use tritium_format::ModelId;
use tritium_quantize::{
    SaltV2Config, SaltV2Error, SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceError,
    SaltV2RestartableTensorMasterFitInput, fit_salt_v2_restartable_tensor_master,
    plan_salt_v2_restartable_tensor_master,
};

use crate::{
    Qwen36AdditiveCampaignSpec, Qwen36AdditiveInstallError, Qwen36AdmittedSource,
    Qwen36CampaignPreflightError, Qwen36CompleteWorkspaceReceipt, Qwen36TensorWorkError,
    tensor_work_store::absolute_path,
};

use super::{Qwen36AdditiveWorkSlot, Qwen36TensorWorkStore, same_file_identity};

const DEFAULT_MAX_EVIDENCE_BYTES: u64 = 64 * 1024 * 1024;
const EVIDENCE_EXTENSION: &str = "s2kf";

/// Bounded canonical layout for per-tensor Kronecker evidence records.
///
/// Record `N` lives at `NNNNNN.s2kf`, where `N` is the tensor's global ordinal
/// in [`Qwen36TensorWorkStore::additive_slots`]. Record contents, rather than
/// filenames, remain authoritative and are checked against the admitted slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PtqEvidenceDirectory {
    root: PathBuf,
    max_record_bytes: u64,
}

impl Qwen36PtqEvidenceDirectory {
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
        let record_count = usize::try_from(record_count)
            .ok()
            .filter(|count| (1..1_000_000).contains(count))
            .ok_or(Qwen36PtqDriverError::InvalidEvidencePath("record count"))?;
        let mut seen = Vec::new();
        seen.try_reserve_exact(record_count)
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        seen.resize(record_count, false);
        let entries = fs::read_dir(&self.root)
            .map_err(|error| evidence_io("read evidence directory", None, error))?;
        for entry in entries {
            let entry = entry.map_err(|error| evidence_io("read evidence entry", None, error))?;
            let metadata = fs::symlink_metadata(entry.path())
                .map_err(|error| evidence_io("inspect evidence entry", None, error))?;
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
        }
        if seen.contains(&false) {
            return Err(Qwen36PtqDriverError::InvalidEvidencePath(
                "incomplete record namespace",
            ));
        }
        Ok(())
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
        let path = self.record_path(tensor_index);
        let path_metadata = fs::symlink_metadata(&path)
            .map_err(|error| evidence_io("inspect evidence record", Some(tensor_index), error))?;
        if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
            return Err(Qwen36PtqDriverError::EvidenceMismatch {
                tensor_index,
                field: "record path type",
            });
        }
        if path_metadata.len() > self.max_record_bytes {
            return Err(Qwen36PtqDriverError::Evidence {
                tensor_index,
                source: SaltV2KroneckerEvidenceError::SizeLimitExceeded {
                    max_bytes: self.max_record_bytes,
                },
            });
        }

        let mut file = File::open(&path)
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
        let record = SaltV2KroneckerEvidence::read_from(&mut file, self.max_record_bytes).map_err(
            |source| Qwen36PtqDriverError::Evidence {
                tensor_index,
                source,
            },
        )?;
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
        Ok(record)
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
    let workspace =
        Qwen36TensorWorkStore::open(admitted).map_err(Qwen36PtqDriverError::Workspace)?;
    evidence.validate_complete(
        u64::try_from(workspace.additive_slots().len())
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?,
    )?;
    let source_model_id = admitted.proof().source_model_id();

    let mut expected_masters = Vec::new();
    expected_masters
        .try_reserve_exact(workspace.additive_slots().len())
        .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
    let mut token_stream_digest = None;
    for (ordinal, slot) in workspace.additive_slots().iter().enumerate() {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let record = evidence.reopen(tensor_index)?;
        validate_record(slot, tensor_index, source_model_id, &record)?;
        bind_token_stream(&mut token_stream_digest, tensor_index, &record)?;
        let weights =
            admitted
                .tensor_f32(slot.name())
                .map_err(|source| Qwen36PtqDriverError::Source {
                    tensor_index,
                    source,
                })?;
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
    workspace
        .reconcile_preserved()
        .map_err(Qwen36PtqDriverError::Workspace)?;
    let campaign = workspace
        .open_master_campaign(spec)
        .map_err(Qwen36PtqDriverError::Workspace)?;
    for (ordinal, expected) in campaign.spec().expected_masters().iter().enumerate() {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        let slot = &workspace.additive_slots()[ordinal];
        campaign
            .install_master(expected, |writer| {
                let record = evidence.reopen(tensor_index)?;
                validate_record(slot, tensor_index, source_model_id, &record)?;
                let weights = admitted.tensor_f32(slot.name()).map_err(|source| {
                    Qwen36PtqDriverError::Source {
                        tensor_index,
                        source,
                    }
                })?;
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
    campaign
        .seal_complete()
        .map_err(Qwen36PtqDriverError::Workspace)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use tritium_quantize::{
        CurvatureSourceId, DensePsdMetric, PhysicalRateTarget, Qwen35SourceDtype, Qwen35TensorRole,
        Qwen35TensorScope, SaltV2Curvature, SaltV2Packing,
    };

    use super::*;
    use crate::Qwen36AdditiveSlotState;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "tritium-qwen36-ptq-{label}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
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
        let mut values = vec![0.0; 128 * 128];
        for index in 0..128 {
            values[index * 128 + index] = 1.0;
        }
        SaltV2KroneckerEvidence::new(
            SaltV2Curvature::GuidedFisher,
            CurvatureSourceId::new(model_digest, [2; 32], token_digest).unwrap(),
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
    }
}
