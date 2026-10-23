//! Replayed immutable scale children; no objective or release qualification.
use crate::pv_tuning::{NativePvJournalError, NativePvRefinementSession, VerifiedNativePvReplay};
use tritium_evidence::NativePvChildReceiptView;
mod profile;
use super::{
    NativeScaleJournalError, NativeScaleRefinementSession, VerifiedNativeScaleReplay,
    hash_parent_tensor, parent_hasher,
};
use profile::Receipt;
pub(super) use profile::{Profile, State};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
};
use tritium_core::DType;
use tritium_evidence::{
    EvidenceError, EvidenceRecorder, NativeRefinementParentView, NativeScaleChildReceiptView,
    SealedEvidence, native_refinement_state_id,
};
use tritium_format::trit_model::{
    TritModelError, TritModelReadError, TritModelReader, TritModelVisitError,
    decode_additive_payload,
};
use tritium_format::{AdditiveTensorError, TritBlobKind, TritPackageReadLimits, TritSpoolBlob};
use tritium_quantize::engine::{
    DirectoryTensorSink, PublicationWarning, TritModelCompanion, TritModelCompanions,
    TritModelPublicationError, TritModelSpool,
};
use tritium_schema::*;

/// Fixed complete TNSR companion in this publication profile.
pub const NATIVE_SCALE_CHECKPOINT_FILE: &str = "checkpoint.tnsr";
/// Fixed complete TNSJ companion in this publication profile.
pub const NATIVE_SCALE_JOURNAL_FILE: &str = "gradients.tnsj";

/// Explicit builder/output payload limits; not whole-process memory or disk qualification.
#[derive(Clone, Copy, Debug)]
pub struct NativeScaleChildLimits {
    /// Complete raw spool payload, before container deduplication/alignment.
    pub max_spool_bytes: u64,
    /// Checkpoint plus one encoded/decoded parent or encoded child tensor.
    pub max_working_bytes: u64,
    /// Actual sealed JSONL length; excludes recorder/metadata heap.
    pub max_evidence_bytes: u64,
    /// Complete checkpoint and journal companion bytes.
    pub max_companion_bytes: u64,
    /// Exact typed container/manifest/directory receiving limits.
    pub receiving: TritPackageReadLimits,
}

/// Logical payload forecast; excludes metadata, shared I/O buffers, existing session and RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeScaleChildSizeLedger {
    /// Complete raw spool bytes including repeated logical blob references.
    pub spool_payload_bytes: u64,
    /// Exact checkpoint+journal companion payload bytes.
    pub companion_payload_bytes: u64,
    /// Conservative builder payload peak, not aggregate heap/RSS.
    pub working_payload_bytes: u64,
}

/// Original admitted parent and numerical replay plus explicit caller observations.
#[derive(Debug)]
pub struct NativeScaleChildSource<'a, R, J> {
    /// Retained actual original parent bytes.
    pub parent: &'a mut TritModelReader<R>,
    /// Original evidence-bound view; never resealed caller JSON.
    pub parent_evidence: &'a NativeRefinementParentView,
    /// Actual numerical replay, rechecked against the current session.
    pub replay: &'a VerifiedNativeScaleReplay,
    /// Actual complete journal, copied against the replayed byte identity.
    pub journal: &'a mut J,
    /// Caller-observed machine label; not independently authenticated hardware.
    pub machine: &'a str,
    /// Complete ordered finite parent/child validation observations, not acceptance.
    pub validation: &'a [NativeRefinementValidation],
}

/// Complete committed child and original observations; not a quality/release verdict.
#[derive(Debug)]
pub struct PublishedNativeScaleChild {
    /// Retained checked child file handle.
    pub model: TritModelReader<File>,
    /// Original sealed native run including numerical replay identity.
    pub evidence: SealedEvidence,
    /// Original record consistency and actual child metadata binding.
    pub receipt: NativeScaleChildReceiptView,
    /// Actual semantic child identity; may equal its parent's.
    pub model_id: ModelId,
    /// Committed output may still have explicit cleanup/directory-sync caveats.
    pub warnings: Vec<PublicationWarning>,
    /// Logical builder/spool/companion observations, not physical memory.
    pub sizes: NativeScaleChildSizeLedger,
}

/// Original parent or staged-child failure before the common commit marker.
#[derive(Debug)]
pub enum NativeScaleChildGuardError {
    /// Original parent bytes no longer match admitted evidence.
    Parent(EvidenceError),
    /// Staged child metadata/identity extraction failed.
    Child(TritModelError),
}
impl std::fmt::Display for NativeScaleChildGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parent(e) => e.fmt(f),
            Self::Child(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NativeScaleChildGuardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parent(e) => Some(e),
            Self::Child(e) => Some(e),
        }
    }
}

/// Original refusal/I/O plus explicit typed publication failures.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeScaleChildError {
    /// Invalid or mismatched source/state/observations.
    Binding(&'static str),
    /// Checked size/cap refusal.
    Limit(&'static str),
    /// Fallible private payload reservation failure.
    Allocation { requested: usize },
    /// Caller spool/journal I/O.
    Io(io::Error),
    /// Typed model interpretation.
    Model(TritModelError),
    /// Original retained source failure.
    Read(TritModelReadError),
    /// Original tensor/asset visitor failure.
    Visit(TritModelVisitError<io::Error>),
    /// Shared canonical tensor codec failure.
    Tensor(AdditiveTensorError),
    /// Existing common state/fingerprint failure.
    Update(super::PvTuningError),
    /// Actual numerical replay/current-state failure.
    Replay(NativeScaleJournalError),
    /// Actual P/V replay/current-state failure, never a scale-only observation.
    PvReplay(NativePvJournalError),
    /// Original evidence/parent/lifecycle failure.
    Evidence(EvidenceError),
    /// Original typed stage, source guard or no-clobber publication failure.
    Publication(TritModelPublicationError<NativeScaleChildGuardError>),
}
impl std::fmt::Display for NativeScaleChildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(s) | Self::Limit(s) => f.write_str(s),
            Self::Allocation { requested } => write!(f, "native child allocation {requested}"),
            Self::Io(e) => e.fmt(f),
            Self::Model(e) => e.fmt(f),
            Self::Read(e) => e.fmt(f),
            Self::Visit(e) => e.fmt(f),
            Self::Tensor(e) => write!(f, "native child tensor codec: {e:?}"),
            Self::Update(e) => e.fmt(f),
            Self::Replay(e) => e.fmt(f),
            Self::PvReplay(e) => e.fmt(f),
            Self::Evidence(e) => e.fmt(f),
            Self::Publication(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NativeScaleChildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Model(e) => Some(e),
            Self::Read(e) => Some(e),
            Self::Visit(e) => Some(e),
            Self::Update(e) => Some(e),
            Self::Replay(e) => Some(e),
            Self::PvReplay(e) => Some(e),
            Self::Evidence(e) => Some(e),
            Self::Publication(e) => Some(e),
            _ => None,
        }
    }
}
macro_rules! from_error {
    ($type:ty,$variant:ident) => {
        impl From<$type> for NativeScaleChildError {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}
from_error!(io::Error, Io);
from_error!(TritModelError, Model);
from_error!(TritModelReadError, Read);
from_error!(TritModelVisitError<io::Error>, Visit);
from_error!(AdditiveTensorError, Tensor);
from_error!(super::PvTuningError, Update);
from_error!(NativeScaleJournalError, Replay);
from_error!(NativePvJournalError, PvReplay);
from_error!(EvidenceError, Evidence);
from_error!(
    TritModelPublicationError<NativeScaleChildGuardError>,
    Publication
);
type Result<T> = std::result::Result<T, NativeScaleChildError>;
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or(NativeScaleChildError::Limit("native child size overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(NativeScaleChildError::Limit("native child size overflow"))
}
fn blob(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn kind(storage: &TensorStorage) -> Result<TritBlobKind> {
    match storage {
        TensorStorage::Additive { .. } => Ok(TritBlobKind::AdditiveTensor),
        TensorStorage::Dense { precision, .. } => Ok(TritBlobKind::DenseTensor(match precision {
            ScalePrecision::F16 => DType::F16,
            ScalePrecision::Bf16 => DType::BF16,
            ScalePrecision::F32 => DType::F32,
            _ => {
                return Err(NativeScaleChildError::Binding(
                    "native child dense precision unsupported",
                ));
            }
        })),
        _ => Err(NativeScaleChildError::Binding(
            "native child storage unsupported",
        )),
    }
}

/// Inspect retained metadata/current shapes without payload I/O or authenticity claims.
/// Actual source/replay, complete container and output checks remain in the publisher.
pub fn inspect_native_scale_child<R: Read + Seek>(
    session: &NativeScaleRefinementSession,
    parent: &TritModelReader<R>,
    replay: &VerifiedNativeScaleReplay,
) -> Result<NativeScaleChildSizeLedger> {
    inspect_child(State::Scale(session, replay), parent)
}
fn inspect_child<R: Read + Seek>(
    state: State<'_>,
    parent: &TritModelReader<R>,
) -> Result<NativeScaleChildSizeLedger> {
    let plan = state.plan();
    if !matches!(
        (state.kind(), &plan.recipe),
        (Profile::Scale, NativeRecoveryRecipe::ScaleOnly { .. })
            | (Profile::Pv, NativeRecoveryRecipe::AlternatingPv { .. })
    ) {
        return Err(NativeScaleChildError::Binding(
            "native child numerical profile differs",
        ));
    }
    let level = parent
        .manifest()
        .levels
        .iter()
        .find(|l| l.name == plan.parent.level)
        .ok_or(NativeScaleChildError::Binding(
            "native child parent level absent",
        ))?;
    let mut spool = 0;
    let mut largest = 0;
    for meta in &level.tensors {
        kind(&meta.storage)?;
        spool = add(spool, parent.tensor_byte_len(&level.name, &meta.name)?)?;
    }
    for asset in &parent.manifest().assets {
        spool = add(spool, parent.asset_byte_len(&asset.name)?)?;
    }
    for (name, weight) in plan.owners.iter().zip(state.weights()) {
        let meta = parent.tensor_meta(&level.name, name)?;
        let TensorStorage::Additive { layout, planes } = &meta.storage else {
            return Err(NativeScaleChildError::Binding(
                "native child selected owner is not additive",
            ));
        };
        let counts_match = match planes {
            PlaneCounts::Uniform(p) => weight.tile_counts().is_none() && *p == weight.plane_count(),
            PlaneCounts::PerTile(counts) => weight.tile_counts() == Some(counts.as_slice()),
            _ => false,
        };
        if *layout != weight.layout() || !counts_match || layout.transport != Transport::Raw {
            return Err(NativeScaleChildError::Binding(
                "native child selected storage differs or transport unsupported",
            ));
        }
        let auxiliary = weight
            .view()
            .auxiliary_buffer_len_bytes()
            .map_err(|_| NativeScaleChildError::Binding("native child index size"))?
            as u64;
        let decoded = add(
            add(weight.trits().len() as u64, auxiliary)?,
            mul(weight.scales().len() as u64, 4)?,
        )?;
        largest = largest.max(add(parent.tensor_byte_len(&level.name, name)?, decoded)?);
    }
    Ok(NativeScaleChildSizeLedger {
        spool_payload_bytes: spool,
        companion_payload_bytes: add(state.footprint().1, state.binding().bytes)?,
        working_payload_bytes: add(state.footprint().1, largest)?,
    })
}

/// Publish a complete unqualified refined child after exact numerical and parent admission.
/// Caller owns source/directory/spool custody and cleanup of any failed private spool tail.
/// Checkpoint/journal companions are durable before the common model commit marker.
pub fn publish_native_scale_child<R: Read + Seek, J: Read + Seek, S: Read + Write + Seek>(
    session: &NativeScaleRefinementSession,
    source: NativeScaleChildSource<'_, R, J>,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    limits: NativeScaleChildLimits,
    recorder: EvidenceRecorder,
) -> Result<PublishedNativeScaleChild> {
    let out = publish_child(
        State::Scale(session, source.replay),
        Source {
            parent: source.parent,
            parent_evidence: source.parent_evidence,
            journal: source.journal,
            machine: source.machine,
            validation: source.validation,
        },
        spool,
        destination,
        limits,
        recorder,
    )?;
    let Receipt::Scale(receipt) = out.receipt else {
        unreachable!("scale profile returns scale receipt")
    };
    Ok(PublishedNativeScaleChild {
        model: out.model,
        evidence: out.evidence,
        receipt,
        model_id: out.model_id,
        warnings: out.warnings,
        sizes: out.sizes,
    })
}
struct Source<'a, R, J> {
    parent: &'a mut TritModelReader<R>,
    parent_evidence: &'a NativeRefinementParentView,
    journal: &'a mut J,
    machine: &'a str,
    validation: &'a [NativeRefinementValidation],
}
struct Published {
    model: TritModelReader<File>,
    evidence: SealedEvidence,
    receipt: Receipt,
    model_id: ModelId,
    warnings: Vec<PublicationWarning>,
    sizes: NativeScaleChildSizeLedger,
}
fn publish_child<R: Read + Seek, J: Read + Seek, S: Read + Write + Seek>(
    state: State<'_>,
    source: Source<'_, R, J>,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    limits: NativeScaleChildLimits,
    mut recorder: EvidenceRecorder,
) -> Result<Published> {
    let plan = state.plan();
    let sizes = inspect_child(state, source.parent)?;
    if limits.max_evidence_bytes == 0
        || limits.max_working_bytes == 0
        || limits.max_spool_bytes == 0
        || sizes.spool_payload_bytes > limits.max_spool_bytes
        || sizes.working_payload_bytes > limits.max_working_bytes
        || sizes.companion_payload_bytes > limits.max_companion_bytes
    {
        return Err(NativeScaleChildError::Limit(
            "native child payload byte limit exceeded",
        ));
    }
    if state.completed() != plan.training.len() as u64 {
        return Err(NativeScaleChildError::Binding(
            "native child schedule incomplete",
        ));
    }
    plan.validate_parent(source.parent.manifest(), source.parent_evidence.binding())
        .map_err(|_| NativeScaleChildError::Binding("native child parent/request mismatch"))?;
    let execution = NativeRefinementExecution {
        backend: state.backend().into(),
        device: format!(
            "cpu:{}:{}:{}",
            std::env::consts::ARCH,
            std::env::consts::OS,
            source.machine
        ),
        source_revision: env!("TRITIUM_SOURCE_ID").into(),
    };
    execution
        .validate()
        .map_err(|_| NativeScaleChildError::Binding("native child execution label invalid"))?;
    if source.machine.is_empty()
        || source.validation.len() != plan.validation.len()
        || source
            .validation
            .iter()
            .zip(&plan.validation)
            .any(|(a, b)| {
                a.content != b.content || !a.parent_loss.is_finite() || !a.child_loss.is_finite()
            })
    {
        return Err(NativeScaleChildError::Binding(
            "native child validation observations differ",
        ));
    }
    if !recorder.events().is_empty() {
        return Err(NativeScaleChildError::Binding(
            "native child requires fresh recorder",
        ));
    }
    recorder.check_emit::<NativeRefinementStarted>("refinement", None, 0)?;
    recorder.check_emit::<NativeRefinementCheckpointed>("refinement", None, 2)?;
    recorder.check_emit::<NativeRefinementCompleted>("refinement", None, 3)?;
    state.check_emit(&recorder)?;
    state.verify()?;
    source.parent_evidence.verify(source.parent)?;
    if spool.seek(SeekFrom::End(0))? != 0 || spool.seek(SeekFrom::Start(0))? != 0 {
        return Err(NativeScaleChildError::Binding(
            "native child spool must be empty",
        ));
    }
    // Bind the actual original decoded parents one at a time, without a duplicate model.
    let mut fingerprint = state.parent_hasher();
    for name in &plan.owners {
        let bytes = read_tensor(source.parent, &plan.parent.level, name)?;
        let decoded =
            decode_additive_payload(source.parent.tensor_meta(&plan.parent.level, name)?, &bytes)?;
        hash_parent_tensor(&mut fingerprint, &decoded)?;
    }
    if fingerprint.finalize().as_bytes() != &state.parent() {
        return Err(NativeScaleChildError::Binding(
            "native child actual original parent state differs",
        ));
    }
    let checkpoint = state.checkpoint()?;
    let checkpoint_id = blob(&checkpoint);
    let mut blobs = Vec::new();
    let mut selected = Vec::new();
    let level_index = source
        .parent
        .manifest()
        .levels
        .iter()
        .position(|l| l.name == plan.parent.level)
        .expect("admitted level");
    let tensor_count = source.parent.manifest().levels[level_index].tensors.len();
    let mut outputs = Vec::new();
    for tensor_index in 0..tensor_count {
        let (name, original_id, storage_kind) = {
            let meta = &source.parent.manifest().levels[level_index].tensors[tensor_index];
            (meta.name.clone(), meta.blob.id, kind(&meta.storage)?)
        };
        let length = source.parent.tensor_byte_len(&plan.parent.level, &name)?;
        let offset = spool.seek(SeekFrom::End(0))?;
        let content = if let Ok(index) = plan.owners.binary_search(&name) {
            let bytes = state
                .weights()
                .nth(index)
                .expect("admitted owner")
                .encode_payload()?;
            if bytes.len() as u64 != length {
                return Err(NativeScaleChildError::Binding(
                    "native child encoded tensor length differs",
                ));
            }
            spool.write_all(&bytes)?;
            let id = blob(&bytes);
            selected.push(NativeRefinementTensorOutput {
                name: name.clone(),
                content: id,
            });
            id
        } else {
            source
                .parent
                .visit_tensor(&plan.parent.level, &name, |chunk| spool.write_all(chunk))?;
            original_id
        };
        blobs.push(TritSpoolBlob {
            kind: storage_kind,
            id: content,
            offset,
            length,
        });
        outputs.push(NativeRefinementTensorOutput { name, content });
    }
    for asset_index in 0..source.parent.manifest().assets.len() {
        let (name, content) = {
            let asset = &source.parent.manifest().assets[asset_index];
            (asset.name.clone(), asset.blob.id)
        };
        let length = source.parent.asset_byte_len(&name)?;
        let offset = spool.seek(SeekFrom::End(0))?;
        source
            .parent
            .visit_asset(&name, |chunk| spool.write_all(chunk))?;
        blobs.push(TritSpoolBlob {
            kind: TritBlobKind::Asset(name),
            id: content,
            offset,
            length,
        });
    }
    if spool.seek(SeekFrom::End(0))? != sizes.spool_payload_bytes {
        return Err(NativeScaleChildError::Binding(
            "native child spool length differs",
        ));
    }
    spool.flush()?;
    let summary = state.binding();
    recorder.emit(
        "refinement",
        None,
        0,
        &NativeRefinementStarted {
            plan: plan.clone(),
            execution,
        },
    )?;
    state.emit(&mut recorder)?;
    recorder.emit(
        "refinement",
        None,
        2,
        &NativeRefinementCheckpointed {
            plan: state.id(),
            completed_updates: state.completed(),
            state: native_refinement_state_id(state.id(), state.completed(), checkpoint_id),
            checkpoint: checkpoint_id,
            checkpoint_bytes: checkpoint.len() as u64,
            state_payload_bytes: state.footprint().0,
            owners: selected,
        },
    )?;
    recorder.emit(
        "refinement",
        None,
        3,
        &NativeRefinementCompleted {
            plan: state.id(),
            checkpoint: checkpoint_id,
            tensors: outputs,
            validation: source.validation.to_vec(),
        },
    )?;
    let evidence = recorder.seal()?;
    if evidence.jsonl().len() as u64 > limits.max_evidence_bytes {
        return Err(NativeScaleChildError::Limit(
            "native child evidence byte limit exceeded",
        ));
    }
    let mut views = state.fold(evidence.log(), source.parent, source.parent_evidence)?;
    if views.len() != 1 {
        return Err(NativeScaleChildError::Binding(
            "native child complete receipt absent",
        ));
    }
    let receipt = views.remove(0);
    if source.journal.seek(SeekFrom::Start(0))? != 0 {
        return Err(NativeScaleChildError::Binding(
            "native child journal seek differs",
        ));
    }
    let mut checkpoint_source = io::Cursor::new(&checkpoint);
    let companions = TritModelCompanions {
        files: vec![
            TritModelCompanion {
                name: state.companions().0,
                source: &mut checkpoint_source,
                bytes: checkpoint.len() as u64,
                content: checkpoint_id,
            },
            TritModelCompanion {
                name: state.companions().1,
                source: source.journal,
                bytes: summary.bytes,
                content: summary.journal,
            },
        ],
        max_bytes: limits.max_companion_bytes,
    };
    let mut receiving = limits.receiving;
    receiving.max_source_bytes = receiving
        .max_source_bytes
        .min(plan.limits.max_package_bytes);
    let (model, model_id, warnings) = destination.publish_spooled_model_with_companions(
        TritModelSpool {
            manifest: receipt.manifest(),
            blobs: &blobs,
            source: spool,
            limits: receiving,
        },
        &evidence,
        companions,
        |candidate| {
            source
                .parent_evidence
                .verify(source.parent)
                .map_err(NativeScaleChildGuardError::Parent)?;
            candidate
                .model_id(&plan.child_level)
                .map_err(NativeScaleChildGuardError::Child)
        },
    )?;
    Ok(Published {
        model,
        evidence,
        receipt,
        model_id,
        warnings,
        sizes,
    })
}

pub(super) fn read_tensor<R: Read + Seek>(
    parent: &mut TritModelReader<R>,
    level: &str,
    name: &str,
) -> Result<Vec<u8>> {
    let length = usize::try_from(parent.tensor_byte_len(level, name)?)
        .map_err(|_| NativeScaleChildError::Limit("native child parent payload width"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| NativeScaleChildError::Allocation { requested: length })?;
    parent.visit_tensor(level, name, |chunk| {
        if bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|n| n > length)
        {
            return Err(io::Error::other(
                "native parent tensor exceeded admitted bytes",
            ));
        }
        bytes.extend_from_slice(chunk);
        Ok(())
    })?;
    if bytes.len() != length {
        return Err(NativeScaleChildError::Binding(
            "native parent tensor length differs",
        ));
    }
    Ok(bytes)
}

/// Shared algorithm-independent publication error contract under a neutral name.
pub type NativeChildPublicationError = NativeScaleChildError;
/// P/V publication errors, retaining original replay/I/O causes.
pub type NativePvChildError = NativeScaleChildError;
/// Shared logical builder/output limits, not P/V qualification.
pub type NativePvChildLimits = NativeScaleChildLimits;
/// Shared logical builder/spool observations, not physical memory.
pub type NativePvChildSizeLedger = NativeScaleChildSizeLedger;
/// Shared original-parent/staged-child source guard failures.
pub type NativePvChildGuardError = NativeScaleChildGuardError;
/// Exact complete TNPV companion leaf.
pub const NATIVE_PV_CHECKPOINT_FILE: &str = "checkpoint.tnpv";
/// Exact complete TNPJ companion leaf.
pub const NATIVE_PV_JOURNAL_FILE: &str = "gradients.tnpj";
/// Actual original parent and P/V numerical replay, not caller approval.
#[derive(Debug)]
pub struct NativePvChildSource<'a, R, J> {
    /// Actual retained immediate parent bytes.
    pub parent: &'a mut TritModelReader<R>,
    /// Original privately admitted parent record, not resealed metadata.
    pub parent_evidence: &'a NativeRefinementParentView,
    /// Actual P/V replay, rechecked against the session before publication.
    pub replay: &'a VerifiedNativePvReplay,
    /// Actual complete TNPJ source, copied against its replayed identity.
    pub journal: &'a mut J,
    /// Caller-observed machine label, not authenticated hardware.
    pub machine: &'a str,
    /// Ordered finite observations, not quality acceptance.
    pub validation: &'a [NativeRefinementValidation],
}
/// Committed complete P/V child; no quality or release acceptance.
#[derive(Debug)]
pub struct PublishedNativePvChild {
    /// Retained checked exact child file.
    pub model: TritModelReader<File>,
    /// Original sealed native run, including distinct P/V replay observation.
    pub evidence: SealedEvidence,
    /// Parent-admitted P/V record consistency, not independent replay/quality.
    pub receipt: NativePvChildReceiptView,
    /// Actual semantic child identity; may equal its parent's.
    pub model_id: ModelId,
    /// Explicit post-commit cleanup/sync caveats.
    pub warnings: Vec<PublicationWarning>,
    /// Logical builder payload observations, not RSS.
    pub sizes: NativePvChildSizeLedger,
}
/// Read-only metadata sizing; actual parent/replay admission remains in publication.
pub fn inspect_native_pv_child<R: Read + Seek>(
    session: &NativePvRefinementSession,
    parent: &TritModelReader<R>,
    replay: &VerifiedNativePvReplay,
) -> Result<NativePvChildSizeLedger> {
    inspect_child(State::Pv(session, replay), parent)
}
/// Publish one complete unqualified P/V child through the shared guarded sink.
/// Parent/source/spool custody remains caller-owned; no-clobber is preserved.
pub fn publish_native_pv_child<R: Read + Seek, J: Read + Seek, S: Read + Write + Seek>(
    session: &NativePvRefinementSession,
    source: NativePvChildSource<'_, R, J>,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    limits: NativePvChildLimits,
    recorder: EvidenceRecorder,
) -> Result<PublishedNativePvChild> {
    let out = publish_child(
        State::Pv(session, source.replay),
        Source {
            parent: source.parent,
            parent_evidence: source.parent_evidence,
            journal: source.journal,
            machine: source.machine,
            validation: source.validation,
        },
        spool,
        destination,
        limits,
        recorder,
    )?;
    let Receipt::Pv(receipt) = out.receipt else {
        unreachable!("PV profile returns PV receipt")
    };
    Ok(PublishedNativePvChild {
        model: out.model,
        evidence: out.evidence,
        receipt,
        model_id: out.model_id,
        warnings: out.warnings,
        sizes: out.sizes,
    })
}
