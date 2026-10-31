//! Actual-parent numerical child verification without all-owner resident state.
use super::*;
use crate::pv_tuning::native_scale::verification::{
    NativeRefinementOwnerReadLimits, NativeRefinementOwnerSource, NativeRefinementOwnerSourceError,
    verify_encoded_owner,
};
use std::io::{SeekFrom, Write};

/// Separate source, numerical/replay and one encoded-owner payload ceilings.
/// These are not aggregate process RSS/VRAM or physical-memory qualification.
#[derive(Clone, Copy, Debug)]
pub struct NativePvPagedChildVerificationLimits {
    /// First-party original-package decode windows and largest decoded owner.
    pub owners: NativeRefinementOwnerReadLimits,
    /// Full journal, fixed replay scratch and largest numerical owner.
    pub replay: NativePvPagedReplayLimits,
    /// Additional single encoded-owner allocation, checked before allocation.
    /// Metadata, callback/stream buffers, allocator slack and OS cache excluded.
    pub max_encoded_owner_bytes: u64,
}
/// Retained actual sources; neither checkpoint nor all parents are buffered.
#[derive(Debug)]
pub struct NativePvPagedChildVerification<'a, P, C, J, K> {
    /// Original immediate parent container.
    pub parent: &'a mut TritModelReader<P>,
    /// Original admitted parent evidence, not caller-resealed metadata.
    pub parent_evidence: &'a NativeRefinementParentView,
    /// Actual deployed child container.
    pub child: &'a mut TritModelReader<C>,
    /// Original verified child log.
    pub log: &'a VerifiedLog,
    /// Frozen complete request.
    pub plan: &'a NativeRefinementPlan,
    /// Actual complete standard TNPJ.
    pub journal: &'a mut J,
    /// Actual expected seek-backed TNPV, read-only.
    pub checkpoint: &'a mut K,
}
/// Original source/evidence/encoding failure or unchanged numerical/factory error.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvPagedChildVerificationError<E> {
    /// Original actual parent/child/receipt/codec/cap failure.
    Child(Error),
    /// Original journal/numerical failure or caller workspace factory error.
    Replay(NativePvPagedReplayError<E>),
}
impl<E: core::fmt::Display> core::fmt::Display for NativePvPagedChildVerificationError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Child(e) => e.fmt(f),
            Self::Replay(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvPagedChildVerificationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Child(e) => Some(e),
            Self::Replay(e) => Some(e),
        }
    }
}
impl<E> From<Error> for NativePvPagedChildVerificationError<E> {
    fn from(e: Error) -> Self {
        Self::Child(e)
    }
}
type Result<T, E> = std::result::Result<T, NativePvPagedChildVerificationError<E>>;
enum Callback<E> {
    Source(Error),
    Workspace(E),
}
fn callback<E>(e: Callback<E>) -> NativePvPagedChildVerificationError<E> {
    match e {
        Callback::Source(e) => NativePvPagedChildVerificationError::Child(e),
        Callback::Workspace(e) => {
            NativePvPagedChildVerificationError::Replay(NativePvPagedReplayError::Workspace(e))
        }
    }
}
fn replay_error<E>(
    e: NativePvPagedReplayError<Callback<E>>,
) -> NativePvPagedChildVerificationError<E> {
    use NativePvPagedChildVerificationError::Replay;
    match e {
        NativePvPagedReplayError::Journal(e) => Replay(NativePvPagedReplayError::Journal(e)),
        NativePvPagedReplayError::Workspace(e) => callback(e),
        NativePvPagedReplayError::Paged(e) => match e {
            NativePvPagedError::Provider(e) => callback(e),
            NativePvPagedError::State(e) => Replay(NativePvPagedReplayError::Paged(
                NativePvPagedError::State(e),
            )),
            NativePvPagedError::Io(e) => {
                Replay(NativePvPagedReplayError::Paged(NativePvPagedError::Io(e)))
            }
        },
    }
}
fn paged_error<E>(e: NativePvPagedError<Error>) -> NativePvPagedChildVerificationError<E> {
    use NativePvPagedChildVerificationError::{Child, Replay};
    match e {
        NativePvPagedError::Provider(e) => Child(e),
        NativePvPagedError::State(e) => Replay(NativePvPagedReplayError::Paged(
            NativePvPagedError::State(e),
        )),
        NativePvPagedError::Io(e) => {
            Replay(NativePvPagedReplayError::Paged(NativePvPagedError::Io(e)))
        }
    }
}
fn scope<E>(
    e: NativeRefinementOwnerSourceError<NativePvPagedChildVerificationError<E>>,
) -> NativePvPagedChildVerificationError<E> {
    match e {
        NativeRefinementOwnerSourceError::Source(e) => e.into(),
        NativeRefinementOwnerSourceError::Consumer(e) => e,
    }
}
fn journal_error<E>(e: NativePvJournalError) -> NativePvPagedChildVerificationError<E> {
    NativePvPagedChildVerificationError::Replay(NativePvPagedReplayError::Journal(e))
}
fn inspect<K: Read + Seek, E>(checkpoint: &mut K, cap: u64) -> Result<checkpoint::Inspection, E> {
    checkpoint::inspect(checkpoint, cap).map_err(|e| paged_error(NativePvPagedError::from(e)))
}
fn check_journal<J: Read + Seek, E>(
    journal: &mut J,
    proof: &VerifiedNativePvReplay,
) -> Result<(), E> {
    let summary = proof.summary();
    if journal
        .seek(SeekFrom::End(0))
        .map_err(|e| journal_error(e.into()))?
        != summary.bytes()
        || journal
            .seek(SeekFrom::Start(0))
            .map_err(|e| journal_error(e.into()))?
            != 0
    {
        return Err(Error::Binding("native paged child journal length/seek differs").into());
    }
    let mut scratch = [0; 8192];
    let hash = crate::pv_tuning::journal_io::hash_exact(journal, summary.bytes(), &mut scratch)
        .map_err(|e| journal_error(e.into()))?;
    if &hash != summary.journal().as_bytes()
        || journal
            .seek(SeekFrom::End(0))
            .map_err(|e| journal_error(e.into()))?
            != summary.bytes()
    {
        return Err(Error::Binding("native paged child journal changed").into());
    }
    Ok(())
}

/// Numerically audit an actual deployed child from actual original package
/// owners, journal and seek-backed expected checkpoint. Reuses the existing child
/// capability; never treats a point, bundle or receipt alone as numerical proof.
/// Factory supplies empty private checkpoints per replay cursor. Named workspace
/// paths remain caller-owned; retire them or use anonymous storage. No original
/// source is written; this does not grant quality, provenance or release approval.
pub fn verify_native_pv_paged_child<
    P: Read + Seek,
    C: Read + Seek,
    J: Read + Seek,
    K: Read + Seek,
    W: Read + Write + Seek,
    E,
>(
    source: NativePvPagedChildVerification<'_, P, C, J, K>,
    limits: NativePvPagedChildVerificationLimits,
    mut make_checkpoint: impl FnMut(u64) -> std::result::Result<W, E>,
) -> Result<VerifiedNativePvChild, E> {
    let NativePvPagedChildVerification {
        parent,
        parent_evidence,
        child,
        log,
        plan,
        journal,
        checkpoint,
    } = source;
    if limits.max_encoded_owner_bytes == 0 {
        return Err(Error::Limit("native paged child encoded owner cap exceeded").into());
    }
    let mut views =
        NativePvChildReceiptView::fold(log, parent, parent_evidence, plan).map_err(Error::from)?;
    if views.len() != 1 {
        return Err(Error::Binding("complete native child receipt absent").into());
    }
    let receipt = views.remove(0);
    receipt.verify_child(child).map_err(Error::from)?;
    for name in &plan.owners {
        let bytes = parent
            .tensor_byte_len(&plan.parent.level, name)
            .map_err(Error::from)?;
        if bytes > limits.max_encoded_owner_bytes {
            return Err(Error::Limit("native paged child encoded owner cap exceeded").into());
        }
        if bytes
            != child
                .tensor_byte_len(&plan.child_level, name)
                .map_err(Error::from)?
        {
            return Err(Error::Binding("native child encoded owner length differs").into());
        }
    }
    let expected = inspect(checkpoint, plan.limits.max_checkpoint_bytes)?;
    let admitted = receipt.replay();
    if expected.bytes != admitted.checkpoint_bytes
        || &expected.digest != admitted.checkpoint.as_bytes()
        || expected.header.plan != admitted.plan
        || expected.header.completed != admitted.completed_updates
        || expected.header.owners != plan.owners.len() as u64
    {
        return Err(Error::Binding("native child checkpoint bytes differ").into());
    }
    let (mut session, replay) =
        NativeRefinementOwnerSource::with(parent, parent_evidence, plan, limits.owners, |owners| {
            replay_native_pv_paged_journal(
                NativePvPagedJournalReplay {
                    plan: plan.clone(),
                    source: &mut *journal,
                    expected_journal: admitted.journal,
                    checkpoint: &mut *checkpoint,
                },
                |i, n| owners.load(i, n).map_err(Callback::Source),
                |i| make_checkpoint(i).map_err(Callback::Workspace),
                limits.replay,
            )
            .map_err(replay_error)
        })
        .map_err(scope)?;
    let s = replay.summary();
    if s.plan() != admitted.plan
        || s.updates() != admitted.completed_updates
        || s.journal() != admitted.journal
        || s.bytes() != admitted.journal_bytes
        || s.checkpoint() != admitted.checkpoint
        || s.checkpoint_bytes() != admitted.checkpoint_bytes
    {
        return Err(Error::Binding("native paged child replay receipt differs").into());
    }
    NativeRefinementOwnerSource::with(parent, parent_evidence, plan, limits.owners, |owners| {
        session
            .visit_weights(
                |i, n| owners.load(i, n),
                |_, name, weight| {
                    let bytes = child
                        .tensor_byte_len(&plan.child_level, name)
                        .map_err(Error::from)?;
                    verify_encoded_owner(weight, child, &plan.child_level, name, bytes)
                },
            )
            .map_err(paged_error)
    })
    .map_err(scope)?;
    parent_evidence.verify(parent).map_err(Error::from)?;
    receipt.verify_child(child).map_err(Error::from)?;
    if inspect(checkpoint, plan.limits.max_checkpoint_bytes)? != expected {
        return Err(Error::Binding("native paged child expected checkpoint changed").into());
    }
    check_journal(journal, &replay)?;
    replay.verify_paged(&mut session).map_err(journal_error)?;
    Ok(VerifiedNativePvChild {
        package: child.package_id(),
        receipt,
        replay,
    })
}
