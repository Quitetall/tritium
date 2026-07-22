//! One-tensor-at-a-time runtime capture for the pinned Qwen3.6 evidence catalog.

use core::fmt;
use std::{fs, io};

use tritium_format::ModelId;
use tritium_quantize::{
    CurvatureSourceId, Qwen35TensorRole, Qwen35TensorScope, SaltV2Curvature,
    SaltV2KroneckerEvidence, SaltV2KroneckerEvidenceBuildError, SaltV2KroneckerEvidenceBuilder,
    SaltV2KroneckerEvidenceSpec,
};

use super::{Qwen36PtqDriverError, Qwen36PtqEvidenceDirectory};
use crate::{Qwen36AdditiveWorkSlot, Qwen36AdmittedSource, Qwen36TensorWorkStore};

const EVIDENCE_SET_CONTEXT: &str = "tritium qwen3.6 ordered PTQ evidence set v1";

/// Immutable runtime request for one canonical additive tensor.
#[derive(Clone, Copy, Debug)]
pub struct Qwen36PtqEvidenceCaptureRequest<'a> {
    tensor_index: u64,
    tensor_name: &'a str,
    rows: usize,
    columns: usize,
    scope: Qwen35TensorScope,
    role: Qwen35TensorRole,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
}

impl Qwen36PtqEvidenceCaptureRequest<'_> {
    /// Zero-based ordinal in the canonical 506-tensor additive catalog.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Canonical source tensor name.
    #[must_use]
    pub const fn tensor_name(&self) -> &str {
        self.tensor_name
    }

    /// Output rows expected from runtime factors.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// G128-aligned activation columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Language or MTP ownership.
    #[must_use]
    pub const fn scope(&self) -> Qwen35TensorScope {
        self.scope
    }

    /// Architecture-specific projection role.
    #[must_use]
    pub const fn role(&self) -> Qwen35TensorRole {
        self.role
    }

    /// Exact checkpoint/cache/token-stream identity required for every batch.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Required estimator semantics.
    #[must_use]
    pub const fn curvature(&self) -> SaltV2Curvature {
        self.curvature
    }

    /// Required post-scaling diagonal damping.
    #[must_use]
    pub const fn damping(&self) -> f64 {
        self.damping
    }
}

/// Deterministic result of one complete or resumed evidence collection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36PtqEvidenceCaptureReceipt {
    evidence_set_digest: [u8; 32],
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping_bits: u64,
    records: u64,
    produced: u64,
    reused: u64,
}

impl Qwen36PtqEvidenceCaptureReceipt {
    /// Ordered identity of all strictly reopened canonical S2KF records.
    #[must_use]
    pub const fn evidence_set_digest(&self) -> &[u8; 32] {
        &self.evidence_set_digest
    }

    /// Exact checkpoint/cache/token-stream identity shared by every record.
    #[must_use]
    pub const fn source_id(&self) -> CurvatureSourceId {
        self.source_id
    }

    /// Curvature estimator shared by every record.
    #[must_use]
    pub const fn curvature(&self) -> SaltV2Curvature {
        self.curvature
    }

    /// Canonical damping shared by every record.
    #[must_use]
    pub const fn damping(&self) -> f64 {
        f64::from_bits(self.damping_bits)
    }

    /// Total records in the completed catalog.
    #[must_use]
    pub const fn records(&self) -> u64 {
        self.records
    }

    /// Missing records produced by this invocation.
    #[must_use]
    pub const fn produced(&self) -> u64 {
        self.produced
    }

    /// Existing records strictly reopened by this invocation.
    #[must_use]
    pub const fn reused(&self) -> u64 {
        self.reused
    }
}

/// Runtime or durable-driver failure during evidence capture.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36PtqEvidenceCaptureError<E> {
    /// Catalog, producer, publication, or strict-reopen failure.
    Driver(Qwen36PtqDriverError),
    /// Runtime capture failed before publication of this tensor.
    Runtime {
        /// Canonical additive-tensor ordinal.
        tensor_index: u64,
        /// Runtime-owned failure.
        source: E,
    },
}

impl<E: fmt::Display> fmt::Display for Qwen36PtqEvidenceCaptureError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(formatter, "Qwen3.6 evidence capture failed: {error}"),
            Self::Runtime {
                tensor_index,
                source,
            } => write!(
                formatter,
                "Qwen3.6 tensor {tensor_index} runtime capture failed: {source}"
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Qwen36PtqEvidenceCaptureError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Runtime { source, .. } => Some(source),
        }
    }
}

/// Replay pinned calibration one tensor at a time into a resumable S2KF catalog.
///
/// The callback must replay the same source-bound calibration stream for every
/// requested tensor and feed its builder in canonical sample order. Existing
/// records are strictly reopened and never replayed. A callback error publishes
/// no record for the active tensor; earlier canonical records remain resumable.
///
/// # Errors
/// Rejects source/cache/token identity gaps, malformed tensor geometry,
/// contradictory existing records, runtime callback failure, producer failure,
/// publication failure, or an incomplete/noncanonical final namespace.
pub fn collect_qwen36_ptq_evidence<E>(
    admitted: &Qwen36AdmittedSource,
    evidence: &Qwen36PtqEvidenceDirectory,
    curvature: SaltV2Curvature,
    activation_cache_digest: [u8; 32],
    token_stream_digest: [u8; 32],
    damping: f64,
    capture: impl FnMut(
        Qwen36PtqEvidenceCaptureRequest<'_>,
        &mut SaltV2KroneckerEvidenceBuilder,
    ) -> Result<(), E>,
) -> Result<Qwen36PtqEvidenceCaptureReceipt, Qwen36PtqEvidenceCaptureError<E>> {
    let workspace = Qwen36TensorWorkStore::open(admitted)
        .map_err(Qwen36PtqDriverError::Workspace)
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    collect_slots(
        workspace.additive_slots(),
        admitted.proof().source_model_id(),
        evidence,
        curvature,
        activation_cache_digest,
        token_stream_digest,
        damping,
        capture,
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_slots<E>(
    slots: &[Qwen36AdditiveWorkSlot],
    source_model_id: ModelId,
    evidence: &Qwen36PtqEvidenceDirectory,
    curvature: SaltV2Curvature,
    activation_cache_digest: [u8; 32],
    token_stream_digest: [u8; 32],
    damping: f64,
    mut capture: impl FnMut(
        Qwen36PtqEvidenceCaptureRequest<'_>,
        &mut SaltV2KroneckerEvidenceBuilder,
    ) -> Result<(), E>,
) -> Result<Qwen36PtqEvidenceCaptureReceipt, Qwen36PtqEvidenceCaptureError<E>> {
    let damping = if damping == 0.0 { 0.0 } else { damping };
    let source_id = CurvatureSourceId::new(
        *source_model_id.as_bytes(),
        activation_cache_digest,
        token_stream_digest,
    )
    .map_err(|source| Qwen36PtqDriverError::EvidenceBuild {
        tensor_index: 0,
        source: SaltV2KroneckerEvidenceBuildError::Curvature(source),
    })
    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    let record_count = u64::try_from(slots.len()).map_err(|_| {
        Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::AllocationFailed)
    })?;
    evidence
        .validate_partial(record_count)
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    preflight_present_records(
        slots,
        source_model_id,
        evidence,
        source_id,
        curvature,
        damping,
    )
    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    let mut hasher = blake3::Hasher::new_derive_key(EVIDENCE_SET_CONTEXT);
    hasher.update(&source_id.digest());
    hasher.update(&[curvature_tag(curvature)]);
    hasher.update(&damping.to_bits().to_le_bytes());
    hasher.update(&record_count.to_le_bytes());
    let mut produced = 0_u64;
    let mut reused = 0_u64;

    for (ordinal, slot) in slots.iter().enumerate() {
        let tensor_index = u64::try_from(ordinal).map_err(|_| {
            Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::AllocationFailed)
        })?;
        let path = evidence.record_path(tensor_index);
        let record = match fs::symlink_metadata(&path) {
            Ok(_) => {
                let record = evidence
                    .reopen(tensor_index)
                    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
                reused = reused.checked_add(1).ok_or_else(|| {
                    Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::AllocationFailed)
                })?;
                record
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let (rows, columns) = slot_geometry(slot, tensor_index)
                    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
                let spec = SaltV2KroneckerEvidenceSpec::new_bounded(
                    curvature,
                    source_id,
                    tensor_index,
                    slot.name(),
                    rows,
                    columns,
                    damping,
                    evidence.max_record_bytes(),
                )
                .map_err(|source| {
                    Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::EvidenceBuild {
                        tensor_index,
                        source,
                    })
                })?;
                let mut builder = evidence
                    .create_builder(spec)
                    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
                let request = Qwen36PtqEvidenceCaptureRequest {
                    tensor_index,
                    tensor_name: slot.name(),
                    rows,
                    columns,
                    scope: slot.scope(),
                    role: slot.role(),
                    source_id,
                    curvature,
                    damping,
                };
                capture(request, &mut builder).map_err(|source| {
                    Qwen36PtqEvidenceCaptureError::Runtime {
                        tensor_index,
                        source,
                    }
                })?;
                evidence
                    .install_builder(builder)
                    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
                let record = evidence
                    .reopen(tensor_index)
                    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
                produced = produced.checked_add(1).ok_or_else(|| {
                    Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::AllocationFailed)
                })?;
                record
            }
            Err(error) => {
                return Err(Qwen36PtqEvidenceCaptureError::Driver(
                    Qwen36PtqDriverError::Io {
                        operation: "inspect evidence record",
                        tensor_index: Some(tensor_index),
                        kind: error.kind(),
                    },
                ));
            }
        };
        validate_capture_record(
            slot,
            tensor_index,
            source_model_id,
            source_id,
            curvature,
            damping,
            &record,
        )
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
        hasher.update(&tensor_index.to_le_bytes());
        hasher.update(&record.record_digest());
    }

    evidence
        .validate_complete(record_count)
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    Ok(Qwen36PtqEvidenceCaptureReceipt {
        evidence_set_digest: *hasher.finalize().as_bytes(),
        source_id,
        curvature,
        damping_bits: damping.to_bits(),
        records: record_count,
        produced,
        reused,
    })
}

fn preflight_present_records(
    slots: &[Qwen36AdditiveWorkSlot],
    source_model_id: ModelId,
    evidence: &Qwen36PtqEvidenceDirectory,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
) -> Result<(), Qwen36PtqDriverError> {
    for (ordinal, slot) in slots.iter().enumerate() {
        let tensor_index =
            u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        match fs::symlink_metadata(evidence.record_path(tensor_index)) {
            Ok(_) => {
                let record = evidence.reopen(tensor_index)?;
                validate_capture_record(
                    slot,
                    tensor_index,
                    source_model_id,
                    source_id,
                    curvature,
                    damping,
                    &record,
                )?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Qwen36PtqDriverError::Io {
                    operation: "inspect evidence record",
                    tensor_index: Some(tensor_index),
                    kind: error.kind(),
                });
            }
        }
    }
    Ok(())
}

fn slot_geometry(
    slot: &Qwen36AdditiveWorkSlot,
    tensor_index: u64,
) -> Result<(usize, usize), Qwen36PtqDriverError> {
    let shape = slot.shape();
    let rows = shape
        .first()
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    let columns = shape
        .get(1)
        .copied()
        .and_then(|value| usize::try_from(value).ok());
    match (shape.len(), rows, columns) {
        (2, Some(rows), Some(columns)) => Ok((rows, columns)),
        _ => Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "tensor geometry",
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_capture_record(
    slot: &Qwen36AdditiveWorkSlot,
    tensor_index: u64,
    source_model_id: ModelId,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
    record: &SaltV2KroneckerEvidence,
) -> Result<(), Qwen36PtqDriverError> {
    super::validate_record(slot, tensor_index, source_model_id, record)?;
    if record.source_id() != source_id {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "capture source identity",
        });
    }
    if record.kind() != curvature {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "curvature estimator",
        });
    }
    if record.damping().to_bits() != damping.to_bits() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index,
            field: "curvature damping",
        });
    }
    Ok(())
}

const fn curvature_tag(curvature: SaltV2Curvature) -> u8 {
    match curvature {
        SaltV2Curvature::InputHessian => 1,
        SaltV2Curvature::GuidedFisher => 2,
        SaltV2Curvature::ForwardKlKronecker => 3,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use tritium_quantize::{Qwen35SourceDtype, SaltV2KroneckerEvidenceBuildError};

    use super::*;
    use crate::Qwen36AdditiveSlotState;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "tritium-qwen36-capture-{label}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn slot(name: &str) -> Qwen36AdditiveWorkSlot {
        Qwen36AdditiveWorkSlot {
            name: name.to_owned(),
            dtype: Qwen35SourceDtype::Bfloat16,
            shape: vec![2, 128],
            coefficients: 256,
            scope: Qwen35TensorScope::Language,
            role: Qwen35TensorRole::MlpProjection,
            source_tensor_digest: [9; 32],
            state: Qwen36AdditiveSlotState::MissingCanonicalMaster,
        }
    }

    #[test]
    fn collection_resumes_without_replaying_valid_records() {
        let root = temp_root("resume");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let slots = [slot("a.weight"), slot("b.weight")];
        let mut calls = 0;
        let first = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |request, builder| {
                calls += 1;
                let activations = vec![request.tensor_index() as f32 + 1.0; 128];
                builder.accumulate_batch(&activations, Some(&[1.0, 2.0]), 1, None, None)
            },
        )
        .unwrap();
        assert_eq!(
            (first.records(), first.produced(), first.reused()),
            (2, 2, 0)
        );
        assert_eq!(first.source_id().activation_cache_digest(), [2; 32]);
        assert_eq!(first.curvature(), SaltV2Curvature::GuidedFisher);
        assert_eq!(first.damping(), 0.25);
        assert_eq!(calls, 2);

        let second = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |_, _| -> Result<(), SaltV2KroneckerEvidenceBuildError> {
                panic!("strictly valid records must not replay")
            },
        )
        .unwrap();
        assert_eq!(
            (second.records(), second.produced(), second.reused()),
            (2, 0, 2)
        );
        assert_eq!(second.evidence_set_digest(), first.evidence_set_digest());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn runtime_failure_publishes_no_partial_tensor_and_retry_skips_prior_work() {
        let root = temp_root("failure");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let slots = [slot("a.weight"), slot("b.weight")];
        let result = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |request, builder| {
                if request.tensor_index() == 1 {
                    return Err("runtime stopped");
                }
                builder
                    .accumulate_batch(&vec![1.0; 128], Some(&[1.0, 2.0]), 1, None, None)
                    .unwrap();
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqEvidenceCaptureError::Runtime {
                tensor_index: 1,
                source: "runtime stopped"
            })
        ));
        assert!(directory.record_path(0).is_file());
        assert!(!directory.record_path(1).exists());

        let mut retried = Vec::new();
        let receipt = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |request, builder| {
                retried.push(request.tensor_index());
                builder.accumulate_batch(&vec![1.0; 128], Some(&[1.0, 2.0]), 1, None, None)
            },
        )
        .unwrap();
        assert_eq!(retried, [1]);
        assert_eq!((receipt.produced(), receipt.reused()), (1, 1));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collection_rejects_hostile_namespace_before_runtime_replay() {
        let root = temp_root("namespace");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        fs::write(root.join("999999.s2kf"), b"hostile").unwrap();
        let slots = [slot("a.weight"), slot("b.weight")];

        let result = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |_, _| -> Result<(), SaltV2KroneckerEvidenceBuildError> {
                panic!("namespace preflight must precede runtime replay")
            },
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqEvidenceCaptureError::Driver(
                Qwen36PtqDriverError::InvalidEvidencePath("record ordinal")
            ))
        ));
        assert!(!directory.record_path(0).exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_source_bound_record_is_never_replayed_or_overwritten() {
        let root = temp_root("stale");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let slots = [slot("a.weight")];
        collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [8; 32],
            [3; 32],
            0.25,
            |_, builder| {
                builder.accumulate_batch(&vec![1.0; 128], Some(&[1.0, 2.0]), 1, None, None)
            },
        )
        .unwrap();
        let original = directory.reopen(0).unwrap().record_digest();

        let result = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |_, _| -> Result<(), SaltV2KroneckerEvidenceBuildError> {
                panic!("stale records fail before runtime replay")
            },
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqEvidenceCaptureError::Driver(
                Qwen36PtqDriverError::EvidenceMismatch {
                    tensor_index: 0,
                    field: "capture source identity"
                }
            ))
        ));
        assert_eq!(directory.reopen(0).unwrap().record_digest(), original);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn later_stale_record_fails_before_an_earlier_missing_tensor_replays() {
        let root = temp_root("later-stale");
        let directory = Qwen36PtqEvidenceDirectory::create(&root).unwrap();
        let slots = [slot("a.weight"), slot("b.weight")];
        let stale_source = CurvatureSourceId::new([1; 32], [8; 32], [3; 32]).unwrap();
        let spec = SaltV2KroneckerEvidenceSpec::new(
            SaltV2Curvature::GuidedFisher,
            stale_source,
            1,
            "b.weight",
            2,
            128,
            0.25,
        )
        .unwrap();
        let mut builder = directory.create_builder(spec).unwrap();
        builder
            .accumulate_batch(&vec![1.0; 128], Some(&[1.0, 2.0]), 1, None, None)
            .unwrap();
        directory.install_builder(builder).unwrap();

        let result = collect_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            &directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |_, _| -> Result<(), SaltV2KroneckerEvidenceBuildError> {
                panic!("all present records must preflight before replay")
            },
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqEvidenceCaptureError::Driver(
                Qwen36PtqDriverError::EvidenceMismatch {
                    tensor_index: 1,
                    field: "capture source identity"
                }
            ))
        ));
        assert!(!directory.record_path(0).exists());
        assert!(directory.record_path(1).is_file());

        fs::remove_dir_all(root).unwrap();
    }
}
