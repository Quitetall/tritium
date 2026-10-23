//! Original-parent numerical replay, one owner at a time, through shared TNPJ parsing.
use super::*;

/// Admitted journal bytes and additional I/O scratch; owner/source payload limits
/// remain separate. These are logical limits, not physical RAM/VRAM evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedReplayLimits {
    /// Complete frozen-schedule journal ceiling; prefix tails are never buffered.
    pub max_journal_bytes: u64,
    /// Additional replay/copy I/O scratch; requires16384bytes.
    pub max_working_bytes: u64,
    /// Largest-owner numerical residency ceiling.
    pub paging: super::super::super::paged::NativePvPagedLimits,
}
/// Exact complete-journal replay inputs. Supplied checkpoint/source are read-only.
pub struct NativePvPagedJournalReplay<'a, R, C> {
    /// Frozen original request.
    pub plan: NativeRefinementPlan,
    /// Original complete TNPJ stream.
    pub source: R,
    /// Independently retained identity of all complete journal bytes.
    pub expected_journal: BlobId,
    /// Actual target TNPV bytes; checksum/finite state alone is not trusted.
    pub checkpoint: &'a mut C,
}
/// Exact saved-prefix recovery inputs; the source is never truncated or written.
pub struct NativePvPagedJournalResume<'a, R, C> {
    /// Same immutable original request, including the complete schedule.
    pub plan: NativeRefinementPlan,
    /// Original prefix or journal with a separately declared unparsed tail.
    pub source: R,
    /// Saved prefix/checkpoint observation, not trusted as replay proof.
    pub point: &'a NativePvResumePoint,
    /// Actual saved target TNPV stream.
    pub checkpoint: &'a mut C,
    /// ExactPrefix rejects a tail; CheckpointPrefix preserves it unparsed.
    pub tail: NativePvResumeTail,
}
impl<R, C> core::fmt::Debug for NativePvPagedJournalReplay<'_, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativePvPagedJournalReplay")
            .field("expected_journal", &self.expected_journal)
            .finish_non_exhaustive()
    }
}
impl<R, C> core::fmt::Debug for NativePvPagedJournalResume<'_, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativePvPagedJournalResume")
            .field("point", &self.point)
            .field("tail", &self.tail)
            .finish_non_exhaustive()
    }
}
/// Original numerical/provider, journal/I/O, or workspace factory refusal.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvPagedReplayError<E> {
    /// Original numerical/source/parent-provider failure.
    Paged(NativePvPagedError<E>),
    /// Original standard parser/cap/checksum/source I/O failure.
    Journal(NativePvJournalError),
    /// Original private checkpoint factory error.
    Workspace(E),
}
impl<E> From<NativePvJournalError> for NativePvPagedReplayError<E> {
    fn from(e: NativePvJournalError) -> Self {
        Self::Journal(e)
    }
}
impl<E: core::fmt::Display> core::fmt::Display for NativePvPagedReplayError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Paged(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
            Self::Workspace(e) => write!(f, "native replay workspace: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvPagedReplayError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Paged(e) => Some(e),
            Self::Journal(e) => Some(e),
            Self::Workspace(e) => Some(e),
        }
    }
}
fn callback<E>(e: NativePvPagedError<Callback<E>>) -> NativePvPagedReplayError<E> {
    match e {
        NativePvPagedError::State(e) => {
            NativePvPagedReplayError::Paged(NativePvPagedError::State(e))
        }
        NativePvPagedError::Io(e) => NativePvPagedReplayError::Paged(NativePvPagedError::Io(e)),
        NativePvPagedError::Provider(Callback::User(e)) => {
            NativePvPagedReplayError::Paged(NativePvPagedError::Provider(e))
        }
        NativePvPagedError::Provider(Callback::Journal(e)) => NativePvPagedReplayError::Journal(e),
    }
}
#[derive(Clone, Copy)]
enum Target<'a> {
    Complete(BlobId),
    Prefix(&'a NativePvResumePoint, NativePvResumeTail),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Checkpoint {
    id: BlobId,
    bytes: u64,
}
struct Inputs<'a, R, C> {
    plan: NativeRefinementPlan,
    source: R,
    checkpoint: &'a mut C,
    target: Target<'a>,
}
struct Replayed<R, W> {
    current: NativePvPagedSession<W>,
    input: Reader<R>,
    target_id: BlobId,
    bytes: u64,
    expected_bytes: u64,
    checkpoint: Checkpoint,
}

fn cap(limits: NativePvPagedReplayLimits) -> Result<()> {
    if limits.max_journal_bytes == 0
        || limits.max_working_bytes < 2 * SCRATCH as u64
        || limits.paging.max_resident_payload_bytes < 16384
    {
        return Err(NativePvJournalError::Limit(
            "paged replay journal/working cap exceeded",
        ));
    }
    Ok(())
}
fn rewind(source: &mut impl Seek) -> Result<()> {
    if source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("paged replay source seek differs"));
    }
    Ok(())
}
fn checkpoint_identity<C: Read + Seek>(
    source: &mut C,
    cap: u64,
    scratch: &mut [u8; SCRATCH],
) -> Result<Checkpoint> {
    let bytes = source.seek(SeekFrom::End(0))?;
    if bytes < super::super::super::checkpoint::FIXED_BYTES || bytes > cap {
        return Err(NativePvJournalError::Limit(
            "paged replay checkpoint length/cap differs",
        ));
    }
    rewind(source)?;
    let id = BlobId::from_bytes(stream::hash_exact(source, bytes, scratch)?);
    if source.seek(SeekFrom::End(0))? != bytes {
        return Err(invalid("paged replay checkpoint length changed"));
    }
    Ok(Checkpoint { id, bytes })
}
fn source_length<R: Seek>(source: &mut R, bytes: u64, target: Target<'_>) -> Result<()> {
    let length = source.seek(SeekFrom::End(0))?;
    match target {
        Target::Complete(_) => {
            if length != bytes {
                return Err(invalid("paged replay complete source length differs"));
            }
        }
        Target::Prefix(_, tail) => super::super::resume::source_length(length, bytes, tail)?,
    }
    Ok(())
}
fn source_identity<R: Read + Seek>(
    source: &mut R,
    bytes: u64,
    id: BlobId,
    target: Target<'_>,
    scratch: &mut [u8; SCRATCH],
) -> Result<()> {
    source_length(source, bytes, target)?;
    rewind(source)?;
    if stream::hash_exact(source, bytes, scratch)? != *id.as_bytes() {
        return Err(invalid("paged replay source identity differs"));
    }
    source_length(source, bytes, target)?;
    rewind(source)
}

fn finish_pass<R: Read>(
    input: &mut Reader<R>,
    target: Target<'_>,
    updates: u64,
    checkpoint: Checkpoint,
    bytes: u64,
    id: BlobId,
) -> Result<()> {
    if let Target::Complete(_) = target {
        if input.array::<8>()? != END
            || input.u64()? != updates
            || input.u64()? != checkpoint.bytes
            || input.array::<32>()? != *checkpoint.id.as_bytes()
        {
            return Err(invalid("native journal trailer differs"));
        }
        let checksum = *input.hash.finalize().as_bytes();
        if input.array::<32>()? != checksum {
            return Err(invalid("native journal terminal checksum differs"));
        }
    }
    if input.bytes != bytes || input.hash.finalize().as_bytes() != id.as_bytes() {
        return Err(invalid("paged replay traversed source differs"));
    }
    Ok(())
}

// Canonical streamed validation before any optimizer/checkpoint allocation.
// Decode through the same raw-f32 parser, retaining at most one small block.
fn check_gradients<R: Read>(
    input: &mut Reader<R>,
    channels: &[(u64, u64)],
    frame: &mut blake3::Hasher,
    scratch: &mut [u8; SCRATCH],
) -> Result<()> {
    let mut values = [0.; SCRATCH / 4];
    for &(s, w) in channels {
        for mut remaining in [s, w] {
            while remaining > 0 {
                let n = remaining.min(values.len() as u64) as usize;
                stream::decode(&mut values[..n], scratch, frame, |b| input.read(b))?;
                if values[..n].iter().any(|v| !v.is_finite()) {
                    return Err(invalid("paged replay gradient is nonfinite"));
                }
                remaining -= n as u64;
            }
        }
    }
    Ok(())
}

fn run<R: Read + Seek, C: Read + Seek, W: Read + Write + Seek, E>(
    inputs: Inputs<'_, R, C>,
    mut parents: impl FnMut(usize, &str) -> std::result::Result<AdditiveTensor, E>,
    mut make: impl FnMut(u64) -> std::result::Result<W, E>,
    limits: NativePvPagedReplayLimits,
) -> std::result::Result<Replayed<R, W>, NativePvPagedReplayError<E>> {
    cap(limits)?;
    let Inputs {
        plan,
        mut source,
        checkpoint,
        target,
    } = inputs;
    let updater = super::super::super::PvUpdater::from_plan(&plan)
        .map_err(|e| NativePvPagedReplayError::Paged(NativePvPagedError::State(e)))?;
    let plan_id = native_refinement_plan_id(&plan)
        .map_err(|_| NativePvPagedReplayError::Journal(invalid("paged replay plan invalid")))?;
    let mut scratch = [0; SCRATCH];
    let expected = checkpoint_identity(checkpoint, plan.limits.max_checkpoint_bytes, &mut scratch)?;
    let (bytes, id, updates) = match target {
        Target::Complete(id) => (
            source
                .seek(SeekFrom::End(0))
                .map_err(NativePvJournalError::from)?,
            id,
            plan.training.len() as u64,
        ),
        Target::Prefix(point, _) => {
            if point.plan != plan_id
                || point.completed_updates > plan.training.len() as u64
                || point.checkpoint != expected.id
                || point.checkpoint_bytes != expected.bytes
            {
                return Err(invalid("paged replay saved point/checkpoint binding differs").into());
            }
            (
                point.journal_prefix_bytes,
                point.journal_prefix,
                point.completed_updates,
            )
        }
    };
    if bytes < 88 || bytes > limits.max_journal_bytes {
        return Err(
            NativePvJournalError::Limit("paged replay admitted source cap exceeded").into(),
        );
    }
    source_identity(&mut source, bytes, id, target, &mut scratch)?;
    let mut channels = reserved(plan.owners.len())
        .map_err(|e| NativePvPagedReplayError::Paged(NativePvPagedError::State(e)))?;
    let mut ledger = super::super::super::paged::empty_ledger();
    let mut parent_hash = NativePvRefinementSession::parent_hasher(plan_id);
    for (i, name) in plan.owners.iter().enumerate() {
        let weight = parents(i, name)
            .map_err(|e| NativePvPagedReplayError::Paged(NativePvPagedError::Provider(e)))?;
        let metadata = (|| {
            super::super::super::paged::account(
                &plan,
                limits.paging,
                updater.fraction,
                &weight,
                &mut ledger,
            )?;
            channels.push((weight.scales().len() as u64, active_count(&weight)? as u64));
            hash_parent_tensor(&mut parent_hash, &weight)
        })();
        metadata.map_err(|e| NativePvPagedReplayError::Paged(NativePvPagedError::State(e)))?;
    }
    let parent = *parent_hash.finalize().as_bytes();
    let sizes = channel_sizes(&plan, &channels)?;
    if sizes.journal_bytes > limits.max_journal_bytes {
        return Err(NativePvJournalError::Limit("paged replay schedule byte cap exceeded").into());
    }
    let prefix = sum(
        sum(88, product(16, plan.owners.len() as u64)?)?,
        product(updates, sizes.frame_bytes)?,
    )?;
    let required = match target {
        Target::Complete(_) => sizes.journal_bytes,
        Target::Prefix(_, _) => prefix,
    };
    if required != bytes || expected.bytes != ledger.logical.checkpoint_bytes {
        return Err(invalid("paged replay source/checkpoint geometry length differs").into());
    }
    let mut input = Reader {
        source,
        hash: blake3::Hasher::new(),
        bytes: 0,
    };
    replay_header(
        &mut input,
        plan_id,
        parent,
        plan.training.len() as u64,
        &channels,
    )?;
    for ordinal in 0..updates {
        let mut frame =
            replay_frame_header(&mut input, ordinal, plan.training[ordinal as usize].content)?;
        check_gradients(&mut input, &channels, &mut frame, &mut scratch)?;
        if input.array::<32>()? != *frame.finalize().as_bytes() {
            return Err(invalid("native journal frame checksum differs").into());
        }
    }
    finish_pass(&mut input, target, updates, expected, bytes, id)?;
    source_identity(&mut input.source, bytes, id, target, &mut scratch)?;
    let initial = make(0).map_err(NativePvPagedReplayError::Workspace)?;
    let mut current =
        NativePvPagedSession::initialize(plan, limits.paging, |i, n| parents(i, n), initial)
            .map_err(NativePvPagedReplayError::Paged)?;
    let state = verify(&mut current)?;
    if state.parent != parent
        || current.journal_channels() != channels
        || current.size_ledger() != ledger
    {
        return Err(invalid("paged replay original parents changed after preflight").into());
    }
    input.hash = blake3::Hasher::new();
    input.bytes = 0;
    replay_header(
        &mut input,
        plan_id,
        parent,
        current.plan().training.len() as u64,
        &channels,
    )?;
    for ordinal in 0..updates {
        let content = current.plan().training[ordinal as usize].content;
        let mut frame = replay_frame_header(&mut input, ordinal, content)?;
        let destination = make(ordinal + 1).map_err(NativePvPagedReplayError::Workspace)?;
        let (next, _, _) = current
            .step_staged(
                destination,
                |i, n| parents(i, n).map_err(Callback::User),
                |i, _, _| {
                    let (s, w) = channels[i];
                    let mut scales = zeroed(usize::try_from(s).map_err(|_| {
                        Callback::Journal(invalid("paged replay scale count width"))
                    })?)
                    .map_err(|e| Callback::Journal(e.into()))?;
                    let mut weights = zeroed(usize::try_from(w).map_err(|_| {
                        Callback::Journal(invalid("paged replay weight count width"))
                    })?)
                    .map_err(|e| Callback::Journal(e.into()))?;
                    replay_owner(
                        &mut input,
                        &mut scales,
                        &mut weights,
                        &mut scratch,
                        &mut frame,
                    )
                    .map_err(Callback::Journal)?;
                    Ok((scales, weights))
                },
                |_, _, _| Ok(()),
            )
            .map_err(callback)?;
        if input.array::<32>()? != *frame.finalize().as_bytes() {
            return Err(invalid("native journal frame checksum differs").into());
        }
        current = next;
    }
    finish_pass(&mut input, target, updates, expected, bytes, id)?;
    let final_state = verify(&mut current)?;
    if final_state.checkpoint != expected.id
        || final_state.checkpoint_bytes != expected.bytes
        || final_state.completed != updates
    {
        return Err(invalid("paged numerical replay exact checkpoint differs").into());
    }
    if checkpoint_identity(checkpoint, expected.bytes, &mut scratch)? != expected {
        return Err(invalid("paged expected checkpoint changed during replay").into());
    }
    source_identity(&mut input.source, bytes, id, target, &mut scratch)?;
    if verify(&mut current)? != final_state {
        return Err(invalid("paged replay workspace changed during terminal source checks").into());
    }
    Ok(Replayed {
        current,
        input,
        target_id: id,
        bytes,
        expected_bytes: sizes.journal_bytes,
        checkpoint: expected,
    })
}

/// Re-execute the complete standard journal through the shared numerical core
/// from one original owner at a time. The factory supplies a new empty private
/// checkpoint stream for ordinal0 and every next cursor. Old workspace streams
/// drop after replacement; caller must bound/retire named files. No original
/// source is written. Return no replay capability before complete exact binding.
pub fn replay_native_pv_paged_journal<R: Read + Seek, C: Read + Seek, W: Read + Write + Seek, E>(
    request: NativePvPagedJournalReplay<'_, R, C>,
    parents: impl FnMut(usize, &str) -> std::result::Result<AdditiveTensor, E>,
    make_checkpoint: impl FnMut(u64) -> std::result::Result<W, E>,
    limits: NativePvPagedReplayLimits,
) -> std::result::Result<
    (NativePvPagedSession<W>, VerifiedNativePvReplay),
    NativePvPagedReplayError<E>,
> {
    let NativePvPagedJournalReplay {
        plan,
        source,
        expected_journal,
        checkpoint,
    } = request;
    let replay = run(
        Inputs {
            plan,
            source,
            checkpoint,
            target: Target::Complete(expected_journal),
        },
        parents,
        make_checkpoint,
        limits,
    )?;
    let state = replay.current.journal_state();
    let proof = VerifiedNativePvReplay {
        parent: state.parent,
        summary: NativePvJournalSummary {
            journal: expected_journal,
            bytes: replay.bytes,
            plan: state.plan,
            checkpoint: state.checkpoint,
            checkpoint_bytes: state.checkpoint_bytes,
            updates: state.completed,
        },
    };
    Ok((replay.current, proof))
}

/// Reproduce a saved prefix numerically, then copy it into an empty new journal.
/// ExactPrefix rejects tails; CheckpointPrefix preserves the unparsed source
/// suffix. Source/prefix/checkpoint identities are rechecked during transfer.
/// Failure leaves only caller-owned private outputs; original evidence is never
/// truncated or appended. Partial prefix is not a complete replay capability.
pub fn resume_native_pv_paged_journal<
    R: Read + Seek,
    C: Read + Seek,
    W: Read + Write + Seek,
    J: Write + Seek,
    E,
>(
    request: NativePvPagedJournalResume<'_, R, C>,
    mut output: J,
    parents: impl FnMut(usize, &str) -> std::result::Result<AdditiveTensor, E>,
    make_checkpoint: impl FnMut(u64) -> std::result::Result<W, E>,
    limits: NativePvPagedReplayLimits,
) -> std::result::Result<
    (NativePvPagedSession<W>, NativePvPagedJournalWriter<J>),
    NativePvPagedReplayError<E>,
> {
    cap(limits)?;
    if output
        .seek(SeekFrom::End(0))
        .map_err(NativePvJournalError::from)?
        != 0
        || output
            .seek(SeekFrom::Start(0))
            .map_err(NativePvJournalError::from)?
            != 0
    {
        return Err(invalid("paged resume destination is not empty").into());
    }
    let NativePvPagedJournalResume {
        plan,
        source,
        point,
        checkpoint,
        tail,
    } = request;
    let target = Target::Prefix(point, tail);
    let mut replay = run(
        Inputs {
            plan,
            source,
            checkpoint,
            target,
        },
        parents,
        make_checkpoint,
        limits,
    )?;
    let mut scratch = [0; SCRATCH];
    source_identity(
        &mut replay.input.source,
        replay.bytes,
        replay.target_id,
        target,
        &mut scratch,
    )?;
    let state = verify(&mut replay.current)?;
    let mut inner = NativePvJournalWriter {
        output,
        hasher: blake3::Hasher::new(),
        plan: state.plan,
        parent: state.parent,
        completed: state.completed,
        bytes: 0,
        expected_bytes: replay.expected_bytes,
        poisoned: false,
    };
    let mut remaining = replay.bytes;
    while remaining > 0 {
        let n = remaining.min(SCRATCH as u64) as usize;
        replay
            .input
            .source
            .read_exact(&mut scratch[..n])
            .map_err(NativePvJournalError::from)?;
        inner.write(&scratch[..n])?;
        remaining -= n as u64;
    }
    if inner.bytes != replay.bytes
        || inner.hasher.finalize().as_bytes() != replay.target_id.as_bytes()
    {
        return Err(invalid("paged resume copied prefix changed").into());
    }
    source_identity(
        &mut replay.input.source,
        replay.bytes,
        replay.target_id,
        target,
        &mut scratch,
    )?;
    inner.flush()?;
    source_identity(
        &mut replay.input.source,
        replay.bytes,
        replay.target_id,
        target,
        &mut scratch,
    )?;
    if checkpoint_identity(checkpoint, replay.checkpoint.bytes, &mut scratch)? != replay.checkpoint
        || verify(&mut replay.current)? != state
    {
        return Err(invalid("paged resume checkpoint changed during copy").into());
    }
    if inner
        .output
        .stream_position()
        .map_err(NativePvJournalError::from)?
        != inner.bytes
        || inner
            .output
            .seek(SeekFrom::End(0))
            .map_err(NativePvJournalError::from)?
            != inner.bytes
    {
        return Err(invalid("paged resume copied destination position/length differs").into());
    }
    Ok((
        replay.current,
        NativePvPagedJournalWriter {
            inner,
            current: state.checkpoint,
        },
    ))
}

impl VerifiedNativePvReplay {
    /// Recheck a complete replay against exact current paged bytes and original
    /// parent/plan/cursor binding. This remains numerical replay, not model/source
    /// authenticity, held-out quality, physical-memory or release qualification.
    pub fn verify_paged<R: Read + Seek>(
        &self,
        session: &mut NativePvPagedSession<R>,
    ) -> Result<()> {
        let state = verify(session)?;
        if self.parent != state.parent
            || self.summary.plan != state.plan
            || self.summary.updates != state.completed
            || self.summary.checkpoint != state.checkpoint
            || self.summary.checkpoint_bytes != state.checkpoint_bytes
        {
            return Err(invalid("native paged replay current state differs"));
        }
        Ok(())
    }
}
