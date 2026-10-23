//! Owner-paged preparation; the common evidence/companion/commit path is unchanged.
use super::super::publication::{self as shared, NativePvChildError as Error, NativePvChildSource};
use super::super::verification::{
    NativeRefinementOwnerReadLimits, NativeRefinementOwnerSource, NativeRefinementOwnerSourceError,
};
use super::*;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};
use tritium_evidence::{
    EvidenceRecorder, NativePvChildReceiptView, NativeRefinementParentView, SealedEvidence,
};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_quantize::engine::{DirectoryTensorSink, PublicationWarning};
use tritium_schema::{ModelId, TensorStorage, Transport};

const IO_HEADROOM: u64 = 65536;
type Result<T> = std::result::Result<T, Error>;
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or(Error::Limit("native paged child size overflow"))
}

/// Paged publisher payload/disk caps, distinct from resident checkpoint-inclusive caps.
#[derive(Clone, Copy, Debug)]
pub struct NativePvPagedChildLimits {
    /// Complete private logical spool bytes before deduplication/alignment.
    pub max_spool_bytes: u64,
    /// One encoded owner plus64KiB fixed publisher I/O headroom. Excludes current
    /// numerical owner/source state, metadata, caller streams, allocator and RSS.
    pub max_working_bytes: u64,
    /// Exact sealed JSONL ceiling, excluding recorder/decoded metadata heap.
    pub max_evidence_bytes: u64,
    /// Exact checkpoint+journal companion payload ceiling.
    pub max_companion_bytes: u64,
    /// First-party original-owner decode cap and explicit overlapping caller payload.
    pub owners: NativeRefinementOwnerReadLimits,
    /// Existing typed container/manifest/directory receiving limits.
    pub receiving: TritPackageReadLimits,
}
/// Separate logical payload scopes, never total memory or physical disk qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedChildSizeLedger {
    /// Raw logical spool payload, including repeated references.
    pub spool_payload_bytes: u64,
    /// Complete checkpoint+journal companion payloads.
    pub companion_payload_bytes: u64,
    /// One encoded owner plus64KiB publisher I/O headroom; no checkpoint Vec.
    pub working_payload_bytes: u64,
    /// Largest actual original-owner decode requirement plus declared caller overlap.
    pub owner_decode_working_bytes: u64,
    /// Existing session's largest numerical owner payload observation, not RSS.
    pub owner_resident_payload_bytes: u64,
}
/// Metadata-only forecast against retained plan/source bindings; does not read
/// checkpoint payloads or authenticate actual source/replay/current bytes.
pub fn inspect_native_pv_paged_child<K: Read + Seek, P: Read + Seek>(
    session: &NativePvPagedSession<K>,
    parent: &mut TritModelReader<P>,
    view: &NativeRefinementParentView,
    replay: &VerifiedNativePvReplay,
    extra_payload_bytes: u64,
) -> Result<NativePvPagedChildSizeLedger> {
    let plan = session.plan();
    let state = session.journal_state();
    let s = replay.summary();
    if state.plan != s.plan()
        || state.completed != s.updates()
        || state.checkpoint != s.checkpoint()
        || state.checkpoint_bytes != s.checkpoint_bytes()
        || state.completed != plan.training.len() as u64
    {
        return Err(Error::Binding(
            "native paged child replay/state metadata differs",
        ));
    }
    let working = NativeRefinementOwnerSource::required_working_bytes(
        parent,
        view,
        plan,
        extra_payload_bytes,
    )?;
    let mut encoded = 0;
    for name in &plan.owners {
        let meta = parent.tensor_meta(&plan.parent.level, name)?;
        if !matches!(&meta.storage,TensorStorage::Additive{layout,..} if layout.transport==Transport::Raw)
        {
            return Err(Error::Binding(
                "native child selected storage differs or transport unsupported",
            ));
        }
        encoded = encoded.max(parent.tensor_byte_len(&plan.parent.level, name)?);
    }
    Ok(NativePvPagedChildSizeLedger {
        spool_payload_bytes: shared::spool_bytes(parent, plan)?,
        companion_payload_bytes: add(s.checkpoint_bytes(), s.bytes())?,
        working_payload_bytes: add(encoded, IO_HEADROOM)?,
        owner_decode_working_bytes: working,
        owner_resident_payload_bytes: session.size_ledger().max_owner_payload_bytes,
    })
}
/// Committed complete standard P/V child; not quality or release approval.
#[derive(Debug)]
pub struct PublishedNativePvPagedChild {
    /// Retained exact child container.
    pub model: TritModelReader<File>,
    /// Original sealed run with ordinary P/V replay observation.
    pub evidence: SealedEvidence,
    /// Parent-admitted record/child consistency, not independent quality acceptance.
    pub receipt: NativePvChildReceiptView,
    /// Semantic child identity; may equal the immediate parent's.
    pub model_id: ModelId,
    /// Postcommit cleanup/sync warnings, never a false failed commit.
    pub warnings: Vec<PublicationWarning>,
    /// Separately scoped logical payload observations.
    pub sizes: NativePvPagedChildSizeLedger,
}
fn paged_error(e: NativePvPagedError<Error>) -> Error {
    match e {
        NativePvPagedError::State(e) => Error::Update(e),
        NativePvPagedError::Io(e) => Error::Io(e),
        NativePvPagedError::Provider(e) => e,
    }
}
fn scope_error(e: NativeRefinementOwnerSourceError<Error>) -> Error {
    match e {
        NativeRefinementOwnerSourceError::Source(e)
        | NativeRefinementOwnerSourceError::Consumer(e) => e,
    }
}

/// Stage owners through actual original-package scopes and publish through the
/// common evidence/companion/fsync/no-clobber transaction. One encoded owner is
/// buffered, but no full checkpoint or all-owner cache. Private spool failures
/// remain caller-owned; no original bytes are written. Does not attest objective,
/// teacher/data, physical-memory, quality, source custody or public activation.
pub fn publish_native_pv_paged_child<
    K: Read + Seek,
    P: Read + Seek,
    J: Read + Seek,
    S: Read + Write + Seek,
>(
    session: &mut NativePvPagedSession<K>,
    source: NativePvChildSource<'_, P, J>,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
    limits: NativePvPagedChildLimits,
    recorder: EvidenceRecorder,
) -> Result<PublishedNativePvPagedChild> {
    let sizes = inspect_native_pv_paged_child(
        session,
        source.parent,
        source.parent_evidence,
        source.replay,
        limits.owners.extra_payload_bytes,
    )?;
    if limits.max_evidence_bytes == 0
        || limits.max_spool_bytes == 0
        || limits.max_working_bytes == 0
        || sizes.spool_payload_bytes > limits.max_spool_bytes
        || sizes.working_payload_bytes > limits.max_working_bytes
        || sizes.companion_payload_bytes > limits.max_companion_bytes
        || limits.owners.max_working_bytes == 0
        || sizes.owner_decode_working_bytes > limits.owners.max_working_bytes
    {
        return Err(Error::Limit(
            "native paged child payload byte limit exceeded",
        ));
    }
    let plan = session.plan().clone();
    let replay = source.replay;
    let publication = shared::Publication::pv(
        &plan,
        replay,
        session.size_ledger().logical.state_payload_bytes,
    );
    let source = shared::Source {
        parent: source.parent,
        parent_evidence: source.parent_evidence,
        journal: source.journal,
        machine: source.machine,
        validation: source.validation,
    };
    let execution = shared::begin(publication, &source, &recorder)?;
    replay.verify_paged(session).map_err(Error::PvReplay)?;
    source.parent_evidence.verify(source.parent)?;
    shared::check_companion(
        source.journal,
        replay.summary().bytes(),
        replay.summary().journal(),
    )?;
    if spool.seek(SeekFrom::End(0))? != 0 || spool.seek(SeekFrom::Start(0))? != 0 {
        return Err(Error::Binding("native child spool must be empty"));
    }
    let lengths: Vec<_> = plan
        .owners
        .iter()
        .map(|n| source.parent.tensor_byte_len(&plan.parent.level, n))
        .collect::<std::result::Result<_, _>>()?;
    let selected = NativeRefinementOwnerSource::with(
        source.parent,
        source.parent_evidence,
        &plan,
        limits.owners,
        |owners| {
            session
                .visit_weights(
                    |i, n| owners.load(i, n),
                    |i, _, weight| shared::encode_owner(spool, weight, lengths[i]),
                )
                .map_err(paged_error)
        },
    )
    .map_err(scope_error)?;
    let staged = shared::copy_preserved(
        &plan,
        source.parent,
        spool,
        &selected,
        sizes.spool_payload_bytes,
    )?;
    replay.verify_paged(session).map_err(Error::PvReplay)?;
    let out = shared::finish(
        shared::Finish {
            state: publication,
            source,
            checkpoint: session.journal_checkpoint(),
            limits: shared::OutputLimits {
                max_evidence_bytes: limits.max_evidence_bytes,
                max_companion_bytes: limits.max_companion_bytes,
                receiving: limits.receiving,
            },
            recorder,
            execution,
            staged,
        },
        spool,
        destination,
    )?;
    let shared::Receipt::Pv(receipt) = out.receipt else {
        unreachable!("PV profile returns PV receipt")
    };
    Ok(PublishedNativePvPagedChild {
        model: out.model,
        evidence: out.evidence,
        receipt,
        model_id: out.model_id,
        warnings: out.warnings,
        sizes,
    })
}
