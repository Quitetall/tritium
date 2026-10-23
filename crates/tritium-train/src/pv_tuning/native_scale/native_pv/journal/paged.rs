//! Staged owner-wise recording through the existing TNPJ encoder/writer.
use super::super::checkpoint::WireError;
use super::super::paged::{NativePvPagedError, NativePvPagedSession, PagedJournalState};
use super::*;
mod replay;
pub use replay::{
    NativePvPagedJournalReplay, NativePvPagedJournalResume, NativePvPagedReplayError,
    NativePvPagedReplayLimits, replay_native_pv_paged_journal, resume_native_pv_paged_journal,
};

/// Independent journal/frame disk caps and recording-only scratch ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedRecordingLimits {
    /// Complete frozen-schedule TNPJ bytes.
    pub max_journal_bytes: u64,
    /// One complete raw gradient frame in private seek-backed storage.
    pub max_frame_bytes: u64,
    /// Additional recording scratch; requires16384bytes, excluding numerical/
    /// source/stream/metadata/allocator/graph/staged-resource memory and RSS.
    pub max_working_bytes: u64,
}
/// Exact disk lengths and fixed logical recording scratch, not physical memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedRecordingSizeLedger {
    /// Complete TNPJ bytes, identical to resident recording.
    pub journal_bytes: u64,
    /// One ordinal/window/raw-adjoint/checksum frame in private storage.
    pub frame_bytes: u64,
    /// Recording-only scratch, separate from numerical/source payload limits.
    pub working_bytes: u64,
}
fn recording_sizes<R: Read + Seek>(
    session: &NativePvPagedSession<R>,
) -> Result<NativePvPagedRecordingSizeLedger> {
    channel_sizes(session.plan(), session.journal_channels())
}
fn channel_sizes(
    plan: &NativeRefinementPlan,
    channels: &[(u64, u64)],
) -> Result<NativePvPagedRecordingSizeLedger> {
    let gradients = channels
        .iter()
        .try_fold(0, |n, &(s, w)| sum(n, sum(s, w)?))?;
    Ok(NativePvPagedRecordingSizeLedger {
        journal_bytes: journal_bytes(plan, gradients)?,
        frame_bytes: sum(72, product(4, gradients)?)?,
        working_bytes: 2 * SCRATCH as u64,
    })
}
fn limits(
    sizes: NativePvPagedRecordingSizeLedger,
    limits: NativePvPagedRecordingLimits,
) -> Result<()> {
    if sizes.journal_bytes > limits.max_journal_bytes
        || sizes.frame_bytes > limits.max_frame_bytes
        || sizes.working_bytes > limits.max_working_bytes
    {
        return Err(NativePvJournalError::Limit(
            "paged recording journal/frame/working cap exceeded",
        ));
    }
    Ok(())
}
fn verify<R: Read + Seek>(session: &mut NativePvPagedSession<R>) -> Result<PagedJournalState> {
    session.verify_journal_source().map_err(|e| match e {
        WireError::Io(e) => NativePvJournalError::Io(e),
        WireError::Invalid(e) => NativePvJournalError::Update(e),
    })?;
    Ok(session.journal_state())
}

/// Paged state adapter around the common TNPJ writer; same bytes, no checkpoint Vec.
/// Output effects obey the existing poisoning rules. Flush is not fsync,
/// immutable custody, numerical replay or objective/quality authentication.
pub struct NativePvPagedJournalWriter<W> {
    inner: NativePvJournalWriter<W>,
    current: BlobId,
}
impl<W> core::fmt::Debug for NativePvPagedJournalWriter<W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativePvPagedJournalWriter")
            .field("completed_updates", &self.inner.completed)
            .field("bytes", &self.inner.bytes)
            .field("poisoned", &self.inner.poisoned)
            .finish_non_exhaustive()
    }
}
impl<W: Write + Seek> NativePvPagedJournalWriter<W> {
    /// Begin from actual zero-origin paged bytes and an empty output, after all
    /// disk/scratch caps pass. Does not mint a replay or durability capability.
    pub fn new<R: Read + Seek>(
        output: W,
        session: &mut NativePvPagedSession<R>,
        cap: NativePvPagedRecordingLimits,
    ) -> Result<Self> {
        let sizes = recording_sizes(session)?;
        limits(sizes, cap)?;
        let state = verify(session)?;
        if state.completed != 0 {
            return Err(invalid("paged journal requires a zero-origin session"));
        }
        let inner = NativePvJournalWriter::start(
            output,
            state.plan,
            state.parent,
            session.plan().training.len() as u64,
            session.journal_channels(),
            sizes.journal_bytes,
        )?;
        if verify(session)? != state {
            return Err(invalid(
                "paged checkpoint changed during journal initialization",
            ));
        }
        Ok(Self {
            inner,
            current: state.checkpoint,
        })
    }
    fn check(&mut self, state: PagedJournalState) -> Result<()> {
        if self.inner.poisoned {
            return Err(invalid("native journal writer is poisoned"));
        }
        if self.inner.plan != state.plan
            || self.inner.parent != state.parent
            || self.inner.completed != state.completed
            || self.current != state.checkpoint
        {
            return Err(invalid(
                "paged journal plan/parent/cursor/checkpoint differs",
            ));
        }
        if self.inner.output.stream_position()? != self.inner.bytes
            || self.inner.output.seek(SeekFrom::End(0))? != self.inner.bytes
        {
            return Err(invalid("paged journal destination position/length differs"));
        }
        Ok(())
    }
    /// Observe an existing NativePvResumePoint without buffering checkpoint bytes.
    /// Full paged numerical prefix recovery is separate; this point alone is not
    /// replay. Caller keeps the exact checkpoint and journal prefix separately.
    pub fn checkpoint<R: Read + Seek>(
        &mut self,
        session: &mut NativePvPagedSession<R>,
    ) -> Result<NativePvResumePoint> {
        let state = verify(session)?;
        self.check(state)?;
        self.inner.flush()?;
        if verify(session)? != state {
            return Err(invalid("paged checkpoint changed during snapshot"));
        }
        Ok(NativePvResumePoint {
            plan: state.plan,
            completed_updates: state.completed,
            journal_prefix: BlobId::from_bytes(*self.inner.hasher.finalize().as_bytes()),
            journal_prefix_bytes: self.inner.bytes,
            checkpoint: state.checkpoint,
            checkpoint_bytes: state.checkpoint_bytes,
        })
    }
    /// Seal the complete standard journal against freshly rechecked paged state.
    pub fn finish<R: Read + Seek>(
        mut self,
        session: &mut NativePvPagedSession<R>,
    ) -> Result<(W, NativePvJournalSummary)> {
        let state = verify(session)?;
        self.check(state)?;
        let (output, summary) = self.inner.finish_observed(
            session.plan().training.len() as u64,
            state.checkpoint_bytes,
            state.checkpoint,
        )?;
        if verify(session)? != state {
            return Err(invalid("paged checkpoint changed during journal finish"));
        }
        Ok((output, summary))
    }
    /// Recover raw output without a completeness/replay observation.
    pub fn into_inner(self) -> W {
        self.inner.into_inner()
    }
}

/// Numerical/provider versus exact original journal/spool failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvPagedRecordingError<E> {
    /// Original numerical, source or callback refusal.
    Paged(NativePvPagedError<E>),
    /// Original frame/journal cap or I/O refusal.
    Journal(NativePvJournalError),
}
impl<E> From<NativePvJournalError> for NativePvPagedRecordingError<E> {
    fn from(e: NativePvJournalError) -> Self {
        Self::Journal(e)
    }
}
impl<E: core::fmt::Display> core::fmt::Display for NativePvPagedRecordingError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Paged(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvPagedRecordingError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Paged(e) => Some(e),
            Self::Journal(e) => Some(e),
        }
    }
}
enum Callback<E> {
    User(E),
    Journal(NativePvJournalError),
}
fn callback_error<E>(e: NativePvPagedError<Callback<E>>) -> NativePvPagedRecordingError<E> {
    match e {
        NativePvPagedError::State(e) => {
            NativePvPagedRecordingError::Paged(NativePvPagedError::State(e))
        }
        NativePvPagedError::Io(e) => NativePvPagedRecordingError::Paged(NativePvPagedError::Io(e)),
        NativePvPagedError::Provider(Callback::User(e)) => {
            NativePvPagedRecordingError::Paged(NativePvPagedError::Provider(e))
        }
        NativePvPagedError::Provider(Callback::Journal(e)) => {
            NativePvPagedRecordingError::Journal(e)
        }
    }
}

struct Frame<F> {
    output: F,
    hash: blake3::Hasher,
    frame: blake3::Hasher,
    bytes: u64,
    expected: u64,
    poisoned: bool,
    channels: Vec<(u64, u64)>,
    next: usize,
}
impl<F: Read + Write + Seek> Frame<F> {
    fn new(
        mut output: F,
        before: PagedJournalState,
        window: [u8; 32],
        channels: &[(u64, u64)],
        expected: u64,
    ) -> Result<Self> {
        if output.seek(SeekFrom::End(0))? != 0 || output.seek(SeekFrom::Start(0))? != 0 {
            return Err(invalid("paged recording frame destination must be empty"));
        }
        let mut result = Self {
            output,
            hash: blake3::Hasher::new(),
            frame: blake3::Hasher::new(),
            bytes: 0,
            expected,
            poisoned: false,
            channels: copied(channels)?,
            next: 0,
        };
        let mut frame = blake3::Hasher::new();
        frame_header(before.completed, window, &mut frame, |bytes| {
            result.write(bytes)
        })?;
        result.frame = frame;
        Ok(result)
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        stream::write(
            &mut self.output,
            &mut self.hash,
            &mut self.bytes,
            self.expected,
            &mut self.poisoned,
            bytes,
        )
        .map_err(Into::into)
    }
    fn owner(&mut self, index: usize, scales: &[f32], weights: &[f32]) -> Result<()> {
        if index != self.next
            || self.channels.get(index) != Some(&(scales.len() as u64, weights.len() as u64))
            || scales.iter().chain(weights).any(|v| !v.is_finite())
        {
            return Err(invalid("paged recording owner order/adjoints differ"));
        }
        let mut frame = self.frame.clone();
        frame_owner(scales, weights, &mut frame, |bytes| self.write(bytes))?;
        self.frame = frame;
        self.next += 1;
        Ok(())
    }
    fn finish(mut self) -> Result<(F, BlobId)> {
        if self.next != self.channels.len() {
            return Err(invalid("paged recording frame incomplete"));
        }
        self.write(self.frame.finalize().as_bytes())?;
        stream::flush(&mut self.output, &mut self.poisoned)?;
        if self.bytes != self.expected {
            return Err(invalid("paged recording frame length differs"));
        }
        let id = BlobId::from_bytes(*self.hash.finalize().as_bytes());
        let mut scratch = [0; SCRATCH];
        check_frame(&mut self.output, self.expected, id, &mut scratch)?;
        Ok((self.output, id))
    }
}
fn check_frame<F: Read + Seek>(
    frame: &mut F,
    bytes: u64,
    id: BlobId,
    scratch: &mut [u8; SCRATCH],
) -> Result<()> {
    if frame.seek(SeekFrom::End(0))? != bytes || frame.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("paged recording frame length/seek differs"));
    }
    if stream::hash_exact(frame, bytes, scratch)? != *id.as_bytes() {
        return Err(invalid("paged recording frame identity differs"));
    }
    Ok(())
}

/// Opaque private next state, exact raw frame and staged resources. Not committed
/// training or replay. Drop on source-scope refusal; keep callback effects private.
pub struct NativePvPagedUpdate<W, F, T> {
    next: NativePvPagedSession<W>,
    frame: F,
    frame_id: BlobId,
    frame_bytes: u64,
    before: PagedJournalState,
    stats: NativePvStepStats,
    resources: Vec<T>,
}
impl<W, F, T> core::fmt::Debug for NativePvPagedUpdate<W, F, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativePvPagedUpdate")
            .field("completed_updates", &self.stats.completed_updates)
            .field("frame_bytes", &self.frame_bytes)
            .finish_non_exhaustive()
    }
}
struct Append<'a, W: Write + Seek> {
    writer: &'a mut NativePvJournalWriter<W>,
    committed: bool,
}
impl<W: Write + Seek> Drop for Append<'_, W> {
    fn drop(&mut self) {
        if !self.committed {
            self.writer.poisoned = true;
        }
    }
}
impl<W: Read + Seek, F: Read + Seek, T> NativePvPagedUpdate<W, F, T> {
    /// Consume once, after the original-parent source scope succeeds. Validate
    /// old/next/frame bytes before live effects; append errors or unwind poison
    /// the writer and expose no next state. Old checkpoint/cursor stays intact;
    /// partial journal/destination bytes remain caller-owned recovery evidence.
    pub fn commit<R: Read + Seek, J: Write + Seek>(
        mut self,
        current: &mut NativePvPagedSession<R>,
        journal: &mut NativePvPagedJournalWriter<J>,
    ) -> Result<(NativePvPagedSession<W>, NativePvStepStats, Vec<T>)> {
        let before = verify(current)?;
        if before != self.before {
            return Err(invalid(
                "paged recorded update belongs to another current checkpoint",
            ));
        }
        let next = verify(&mut self.next)?;
        if next.plan != before.plan
            || next.parent != before.parent
            || next.completed != before.completed + 1
            || next.checkpoint_bytes != before.checkpoint_bytes
        {
            return Err(invalid("paged recorded candidate binding differs"));
        }
        journal.check(before)?;
        let mut scratch = [0; SCRATCH];
        check_frame(
            &mut self.frame,
            self.frame_bytes,
            self.frame_id,
            &mut scratch,
        )?;
        if self.frame.seek(SeekFrom::Start(0))? != 0 {
            return Err(invalid("paged recording frame seek differs"));
        }
        let mut tx = Append {
            writer: &mut journal.inner,
            committed: false,
        };
        let mut remaining = self.frame_bytes;
        let mut copied = blake3::Hasher::new();
        while remaining > 0 {
            let n = remaining.min(SCRATCH as u64) as usize;
            self.frame.read_exact(&mut scratch[..n])?;
            copied.update(&scratch[..n]);
            tx.writer.write(&scratch[..n])?;
            remaining -= n as u64;
        }
        if copied.finalize().as_bytes() != self.frame_id.as_bytes() {
            return Err(invalid("paged recording transferred frame changed"));
        }
        check_frame(
            &mut self.frame,
            self.frame_bytes,
            self.frame_id,
            &mut scratch,
        )?;
        tx.writer.flush()?;
        if verify(current)? != before || verify(&mut self.next)? != next {
            return Err(invalid("paged recorded checkpoint changed during append"));
        }
        if tx.writer.output.stream_position()? != tx.writer.bytes
            || tx.writer.output.seek(SeekFrom::End(0))? != tx.writer.bytes
        {
            return Err(invalid("paged recording appended position/length differs"));
        }
        tx.writer.completed += 1;
        tx.committed = true;
        drop(tx);
        journal.current = next.checkpoint;
        Ok((self.next, self.stats, self.resources))
    }
}

impl<R: Read + Seek> NativePvPagedSession<R> {
    /// Inspect exact full-journal/frame disk lengths and fixed recording scratch.
    /// Numerical/source payloads and arbitrary stream/resource buffers are separate.
    pub fn recording_size_ledger(&self) -> Result<NativePvPagedRecordingSizeLedger> {
        recording_sizes(self)
    }
    /// Stage one recorded update using empty checkpoint/frame streams. The
    /// gradient frame is raw/unweighted and follows the existing TNPJ encoder.
    /// Return this opaque result through the parent source scope, then commit;
    /// do not append the live journal inside an unverified source callback.
    pub fn stage_recorded<W: Read + Write + Seek, F: Read + Write + Seek, E, T>(
        &mut self,
        outputs: (W, F),
        mut parents: impl FnMut(usize, &str) -> std::result::Result<AdditiveTensor, E>,
        mut gradients: impl FnMut(
            usize,
            &str,
            &AdditiveTensor,
        ) -> std::result::Result<(Vec<f32>, Vec<f32>), E>,
        mut stage: impl FnMut(usize, &str, &AdditiveTensor) -> std::result::Result<T, E>,
        cap: NativePvPagedRecordingLimits,
    ) -> std::result::Result<NativePvPagedUpdate<W, F, T>, NativePvPagedRecordingError<E>> {
        let sizes = recording_sizes(self)?;
        limits(sizes, cap)?;
        let before = self.journal_state();
        let window = self
            .plan()
            .training
            .get(before.completed as usize)
            .ok_or_else(|| {
                NativePvPagedRecordingError::Paged(NativePvPagedError::State(PvTuningError::step(
                    "native PV schedule is exhausted",
                )))
            })?
            .content;
        let mut frame = Frame::new(
            outputs.1,
            before,
            window,
            self.journal_channels(),
            sizes.frame_bytes,
        )?;
        let (next, stats, resources) = self
            .step_staged(
                outputs.0,
                |i, name| parents(i, name).map_err(Callback::User),
                |i, name, w| {
                    let (s, g) = gradients(i, name, w).map_err(Callback::User)?;
                    frame.owner(i, &s, &g).map_err(Callback::Journal)?;
                    Ok((s, g))
                },
                |i, name, w| stage(i, name, w).map_err(Callback::User),
            )
            .map_err(callback_error)?;
        let (frame, frame_id) = frame.finish()?;
        Ok(NativePvPagedUpdate {
            next,
            frame,
            frame_id,
            frame_bytes: sizes.frame_bytes,
            before,
            stats,
            resources,
        })
    }
}
