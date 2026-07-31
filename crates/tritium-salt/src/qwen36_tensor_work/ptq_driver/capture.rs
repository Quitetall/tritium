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

/// Owned description of one missing canonical Qwen3.6 curvature record.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen36PtqEvidenceCaptureTask {
    tensor_index: u64,
    tensor_name: String,
    rows: usize,
    columns: usize,
    scope: Qwen35TensorScope,
    role: Qwen35TensorRole,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
}

impl Qwen36PtqEvidenceCaptureTask {
    /// Zero-based ordinal in the canonical 506-tensor additive catalog.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Canonical source tensor name.
    #[must_use]
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
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

    fn borrowed(&self) -> Qwen36PtqEvidenceCaptureRequest<'_> {
        Qwen36PtqEvidenceCaptureRequest {
            tensor_index: self.tensor_index,
            tensor_name: &self.tensor_name,
            rows: self.rows,
            columns: self.columns,
            scope: self.scope,
            role: self.role,
            source_id: self.source_id,
            curvature: self.curvature,
            damping: self.damping,
        }
    }

    fn spec(
        &self,
        max_record_bytes: u64,
    ) -> Result<SaltV2KroneckerEvidenceSpec, SaltV2KroneckerEvidenceBuildError> {
        SaltV2KroneckerEvidenceSpec::new_bounded(
            self.curvature,
            self.source_id,
            self.tensor_index,
            self.tensor_name.clone(),
            self.rows,
            self.columns,
            self.damping,
            max_record_bytes,
        )
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

/// Resumable one-record-at-a-time traversal of the canonical Qwen3.6 catalog.
///
/// Reopening preflights every already-present record before exposing the first
/// missing task. [`Self::next_request`] is idempotent while a task is pending;
/// [`Self::accept_current`] advances only after that exact immutable record
/// strictly reopens. [`Self::finish`] revalidates the complete namespace and
/// recomputes the ordered evidence-set identity from durable bytes.
#[derive(Debug)]
pub struct Qwen36PtqEvidenceCaptureSession {
    evidence: Qwen36PtqEvidenceDirectory,
    tasks: Vec<Qwen36PtqEvidenceCaptureTask>,
    source_model_id: ModelId,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
    records: u64,
    cursor: usize,
    pending: bool,
    produced: u64,
    reused: u64,
    receipt: Option<Qwen36PtqEvidenceCaptureReceipt>,
}

impl Qwen36PtqEvidenceCaptureSession {
    /// Open a source-bound resumable session for the pinned admitted checkpoint.
    ///
    /// # Errors
    /// Rejects workspace/catalog construction failures, a hostile partial
    /// namespace, stale records, invalid estimator identity, or allocation
    /// failure before any missing task is exposed.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        admitted: &Qwen36AdmittedSource,
        evidence: Qwen36PtqEvidenceDirectory,
        curvature: SaltV2Curvature,
        activation_cache_digest: [u8; 32],
        token_stream_digest: [u8; 32],
        damping: f64,
    ) -> Result<Self, Qwen36PtqDriverError> {
        let workspace =
            Qwen36TensorWorkStore::open(admitted).map_err(Qwen36PtqDriverError::Workspace)?;
        Self::open_slots(
            workspace.additive_slots(),
            admitted.proof().source_model_id(),
            evidence,
            curvature,
            activation_cache_digest,
            token_stream_digest,
            damping,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn open_slots(
        slots: &[Qwen36AdditiveWorkSlot],
        source_model_id: ModelId,
        evidence: Qwen36PtqEvidenceDirectory,
        curvature: SaltV2Curvature,
        activation_cache_digest: [u8; 32],
        token_stream_digest: [u8; 32],
        damping: f64,
    ) -> Result<Self, Qwen36PtqDriverError> {
        let damping = if damping == 0.0 { 0.0 } else { damping };
        let source_id = CurvatureSourceId::new(
            *source_model_id.as_bytes(),
            activation_cache_digest,
            token_stream_digest,
        )
        .map_err(|source| Qwen36PtqDriverError::EvidenceBuild {
            tensor_index: 0,
            source: SaltV2KroneckerEvidenceBuildError::Curvature(source),
        })?;
        let record_count =
            u64::try_from(slots.len()).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        evidence.validate_partial(record_count)?;
        preflight_present_records(
            slots,
            source_model_id,
            &evidence,
            source_id,
            curvature,
            damping,
        )?;
        let mut tasks = Vec::new();
        tasks
            .try_reserve_exact(slots.len())
            .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
        for (ordinal, slot) in slots.iter().enumerate() {
            let tensor_index =
                u64::try_from(ordinal).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
            let (rows, columns) = slot_geometry(slot, tensor_index)?;
            let mut tensor_name = String::new();
            tensor_name
                .try_reserve_exact(slot.name().len())
                .map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
            tensor_name.push_str(slot.name());
            let task = Qwen36PtqEvidenceCaptureTask {
                tensor_index,
                tensor_name,
                rows,
                columns,
                scope: slot.scope(),
                role: slot.role(),
                source_id,
                curvature,
                damping,
            };
            task.spec(evidence.max_record_bytes()).map_err(|source| {
                Qwen36PtqDriverError::EvidenceBuild {
                    tensor_index,
                    source,
                }
            })?;
            tasks.push(task);
        }
        Ok(Self {
            evidence,
            tasks,
            source_model_id,
            source_id,
            curvature,
            damping,
            records: record_count,
            cursor: 0,
            pending: false,
            produced: 0,
            reused: 0,
            receipt: None,
        })
    }

    /// Return the next missing task, or the same task until it is accepted.
    ///
    /// Existing valid records advance without replay and count as reused.
    ///
    /// # Errors
    /// Rejects any record that appeared or changed after session preflight but
    /// no longer matches its canonical catalog slot and provenance.
    pub fn next_request(
        &mut self,
    ) -> Result<Option<Qwen36PtqEvidenceCaptureTask>, Qwen36PtqDriverError> {
        if self.receipt.is_some() {
            return Ok(None);
        }
        if self.pending {
            return Ok(self.tasks.get(self.cursor).cloned());
        }
        while let Some(task) = self.tasks.get(self.cursor) {
            match fs::symlink_metadata(self.evidence.record_path(task.tensor_index)) {
                Ok(_) => {
                    let record = self.evidence.reopen(task.tensor_index)?;
                    validate_capture_task(task, self.source_model_id, &record)?;
                    self.reused = self
                        .reused
                        .checked_add(1)
                        .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
                    self.cursor = self
                        .cursor
                        .checked_add(1)
                        .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    self.pending = true;
                    return Ok(Some(task.clone()));
                }
                Err(error) => {
                    return Err(Qwen36PtqDriverError::Io {
                        operation: "inspect evidence record",
                        tensor_index: Some(task.tensor_index),
                        kind: error.kind(),
                    });
                }
            }
        }
        Ok(None)
    }

    /// Validate and accept the durable record for the currently pending task.
    ///
    /// Returns `false` when there is no pending task or its record has not yet
    /// been published. Validation failure leaves the task pending and retryable.
    ///
    /// # Errors
    /// Rejects a malformed, stale, conflicting, or wrong-slot published record.
    pub fn accept_current(&mut self) -> Result<bool, Qwen36PtqDriverError> {
        if !self.pending {
            return Ok(false);
        }
        let task = self
            .tasks
            .get(self.cursor)
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        match fs::symlink_metadata(self.evidence.record_path(task.tensor_index)) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => {
                return Err(Qwen36PtqDriverError::Io {
                    operation: "inspect evidence record",
                    tensor_index: Some(task.tensor_index),
                    kind: error.kind(),
                });
            }
        }
        let record = self.evidence.reopen(task.tensor_index)?;
        validate_capture_task(task, self.source_model_id, &record)?;
        self.produced = self
            .produced
            .checked_add(1)
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        self.cursor = self
            .cursor
            .checked_add(1)
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        self.pending = false;
        Ok(true)
    }

    /// Total, newly accepted, and reused record counts for this invocation.
    #[must_use]
    pub fn counts(&self) -> (u64, u64, u64) {
        (self.records, self.produced, self.reused)
    }

    /// Seal a complete session after a fresh full durable-record validation.
    ///
    /// Returns `None` while any canonical task remains missing. Calling this
    /// method again after success returns the same immutable receipt.
    ///
    /// # Errors
    /// Rejects namespace drift, any changed record, or identity mismatch.
    pub fn finish(
        &mut self,
    ) -> Result<Option<Qwen36PtqEvidenceCaptureReceipt>, Qwen36PtqDriverError> {
        if let Some(receipt) = &self.receipt {
            return Ok(Some(receipt.clone()));
        }
        if self.next_request()?.is_some() {
            return Ok(None);
        }
        let receipt = complete_capture_receipt(
            &self.tasks,
            self.source_model_id,
            &self.evidence,
            self.source_id,
            self.curvature,
            self.damping,
            self.produced,
            self.reused,
        )?;
        self.receipt = Some(receipt.clone());
        Ok(Some(receipt))
    }

    fn create_current_builder(
        &self,
    ) -> Result<SaltV2KroneckerEvidenceBuilder, Qwen36PtqDriverError> {
        let task = self
            .tasks
            .get(self.cursor)
            .filter(|_| self.pending)
            .ok_or(Qwen36PtqDriverError::AllocationFailed)?;
        let spec = task
            .spec(self.evidence.max_record_bytes())
            .map_err(|source| Qwen36PtqDriverError::EvidenceBuild {
                tensor_index: task.tensor_index,
                source,
            })?;
        self.evidence.create_builder(spec)
    }

    fn install_current_builder(
        &mut self,
        builder: SaltV2KroneckerEvidenceBuilder,
    ) -> Result<(), Qwen36PtqDriverError> {
        self.evidence.install_builder(builder)?;
        if !self.accept_current()? {
            return Err(Qwen36PtqDriverError::AllocationFailed);
        }
        Ok(())
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
    let mut session = Qwen36PtqEvidenceCaptureSession::open_slots(
        slots,
        source_model_id,
        evidence.clone(),
        curvature,
        activation_cache_digest,
        token_stream_digest,
        damping,
    )
    .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    while let Some(task) = session
        .next_request()
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?
    {
        let mut builder = session
            .create_current_builder()
            .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
        capture(task.borrowed(), &mut builder).map_err(|source| {
            Qwen36PtqEvidenceCaptureError::Runtime {
                tensor_index: task.tensor_index,
                source,
            }
        })?;
        session
            .install_current_builder(builder)
            .map_err(Qwen36PtqEvidenceCaptureError::Driver)?;
    }
    session
        .finish()
        .map_err(Qwen36PtqEvidenceCaptureError::Driver)?
        .ok_or_else(|| {
            Qwen36PtqEvidenceCaptureError::Driver(Qwen36PtqDriverError::AllocationFailed)
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

fn validate_capture_task(
    task: &Qwen36PtqEvidenceCaptureTask,
    source_model_id: ModelId,
    record: &SaltV2KroneckerEvidence,
) -> Result<(), Qwen36PtqDriverError> {
    if record.rows() != task.rows || record.columns() != task.columns {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "tensor geometry",
        });
    }
    if record.tensor_index() != task.tensor_index {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "global tensor ordinal",
        });
    }
    if record.tensor_name() != task.tensor_name {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "tensor name",
        });
    }
    if record.source_id().source_model_digest() != *source_model_id.as_bytes() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "source model identity",
        });
    }
    if record.source_id() != task.source_id {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "capture source identity",
        });
    }
    if record.kind() != task.curvature {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "curvature estimator",
        });
    }
    if record.damping().to_bits() != task.damping.to_bits() {
        return Err(Qwen36PtqDriverError::EvidenceMismatch {
            tensor_index: task.tensor_index,
            field: "curvature damping",
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn complete_capture_receipt(
    tasks: &[Qwen36PtqEvidenceCaptureTask],
    source_model_id: ModelId,
    evidence: &Qwen36PtqEvidenceDirectory,
    source_id: CurvatureSourceId,
    curvature: SaltV2Curvature,
    damping: f64,
    produced: u64,
    reused: u64,
) -> Result<Qwen36PtqEvidenceCaptureReceipt, Qwen36PtqDriverError> {
    let record_count =
        u64::try_from(tasks.len()).map_err(|_| Qwen36PtqDriverError::AllocationFailed)?;
    evidence.validate_complete(record_count)?;
    let mut hasher = blake3::Hasher::new_derive_key(EVIDENCE_SET_CONTEXT);
    hasher.update(&source_id.digest());
    hasher.update(&[curvature_tag(curvature)]);
    hasher.update(&damping.to_bits().to_le_bytes());
    hasher.update(&record_count.to_le_bytes());
    for task in tasks {
        let record = evidence.reopen(task.tensor_index)?;
        validate_capture_task(task, source_model_id, &record)?;
        hasher.update(&task.tensor_index.to_le_bytes());
        hasher.update(&record.record_digest());
    }
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

const fn curvature_tag(curvature: SaltV2Curvature) -> u8 {
    match curvature {
        SaltV2Curvature::InputHessian => 1,
        SaltV2Curvature::GuidedFisher => 2,
        SaltV2Curvature::ForwardKlKronecker => 3,
        _ => 0,
    }
}

// Evidence capture asserts strict directory durability, which is unix-only by
// design (sync_evidence_directory returns Unsupported elsewhere).
#[cfg(all(test, unix))]
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
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
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
    fn capture_session_is_idempotent_resumable_and_digest_equivalent() {
        let session_root = temp_root("session");
        let session_directory = Qwen36PtqEvidenceDirectory::create(&session_root).unwrap();
        let reference_root = temp_root("session-reference");
        let reference_directory = Qwen36PtqEvidenceDirectory::create(&reference_root).unwrap();
        let slots = [slot("a.weight"), slot("b.weight")];
        let source_model_id = ModelId::from_digest([1; 32]);

        let mut session = Qwen36PtqEvidenceCaptureSession::open_slots(
            &slots,
            source_model_id,
            session_directory.clone(),
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
        )
        .unwrap();
        let first = session.next_request().unwrap().unwrap();
        assert_eq!(first.tensor_index(), 0);
        assert_eq!(first.tensor_name(), "a.weight");
        assert_eq!((first.rows(), first.columns()), (2, 128));
        assert_eq!(session.next_request().unwrap().unwrap(), first);
        assert!(!session.accept_current().unwrap());
        let mut builder = session_directory
            .create_builder(first.spec(session_directory.max_record_bytes()).unwrap())
            .unwrap();
        builder
            .accumulate_batch(&vec![1.0; 128], Some(&[1.0, 2.0]), 1, None, None)
            .unwrap();
        session_directory.install_builder(builder).unwrap();
        assert!(session.accept_current().unwrap());
        assert_eq!(session.counts(), (2, 1, 0));

        drop(session);
        let mut resumed = Qwen36PtqEvidenceCaptureSession::open_slots(
            &slots,
            source_model_id,
            session_directory.clone(),
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
        )
        .unwrap();
        let second = resumed.next_request().unwrap().unwrap();
        assert_eq!(second.tensor_index(), 1);
        assert_eq!(resumed.counts(), (2, 0, 1));
        let mut builder = session_directory
            .create_builder(second.spec(session_directory.max_record_bytes()).unwrap())
            .unwrap();
        builder
            .accumulate_batch(&vec![2.0; 128], Some(&[1.0, 2.0]), 1, None, None)
            .unwrap();
        session_directory.install_builder(builder).unwrap();
        assert!(resumed.accept_current().unwrap());
        assert!(resumed.next_request().unwrap().is_none());
        let receipt = resumed.finish().unwrap().unwrap();
        assert_eq!(
            (receipt.records(), receipt.produced(), receipt.reused()),
            (2, 1, 1)
        );

        let reference = collect_slots(
            &slots,
            source_model_id,
            &reference_directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
            |request, builder| {
                builder.accumulate_batch(
                    &vec![request.tensor_index() as f32 + 1.0; 128],
                    Some(&[1.0, 2.0]),
                    1,
                    None,
                    None,
                )
            },
        )
        .unwrap();
        assert_eq!(
            receipt.evidence_set_digest(),
            reference.evidence_set_digest()
        );

        fs::remove_dir_all(session_root).unwrap();
        fs::remove_dir_all(reference_root).unwrap();
    }

    #[test]
    fn capture_session_preflights_every_task_spec_before_exposure() {
        let root = temp_root("session-preflight");
        let directory = Qwen36PtqEvidenceDirectory::create_bounded(&root, 1).unwrap();
        let slots = [slot("a.weight")];
        let result = Qwen36PtqEvidenceCaptureSession::open_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            0.25,
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqDriverError::EvidenceBuild {
                tensor_index: 0,
                source: SaltV2KroneckerEvidenceBuildError::SizeLimitExceeded { .. }
            })
        ));

        let directory = Qwen36PtqEvidenceDirectory::open_bounded(&root, u64::MAX).unwrap();
        let result = Qwen36PtqEvidenceCaptureSession::open_slots(
            &slots,
            ModelId::from_digest([1; 32]),
            directory,
            SaltV2Curvature::GuidedFisher,
            [2; 32],
            [3; 32],
            f64::NAN,
        );
        assert!(matches!(
            result,
            Err(Qwen36PtqDriverError::EvidenceBuild {
                tensor_index: 0,
                source: SaltV2KroneckerEvidenceBuildError::Malformed("damping")
            })
        ));

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
