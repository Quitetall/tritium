//! Streamed numerical replay, not objective authentication or publication approval.
use super::*;
use std::io::{Read, Seek, SeekFrom, Write};
use tritium_schema::{BlobId, NativePvResumePoint};

const MAGIC: [u8; 8] = *b"TNPJ\0\0\0\x01";
const END: [u8; 8] = *b"TNPJEND1";
use crate::pv_tuning::journal_io as stream;
const SCRATCH: usize = stream::SCRATCH;
const FRAME_DOMAIN: &[u8] = b"tritium.native-pv.journal-frame.v1\0";
mod bundle;
mod paged;
mod resume;
pub use bundle::{
    NativePvResumeBundleError, NativePvResumeBundleLimits, NativePvResumeBundleReader,
    PublishedNativePvResumeBundle, publish_native_pv_resume_bundle, stage_native_pv_resume_bundle,
};
pub use paged::{
    NativePvPagedJournalReplay, NativePvPagedJournalResume, NativePvPagedJournalWriter,
    NativePvPagedRecordingError, NativePvPagedRecordingLimits, NativePvPagedRecordingSizeLedger,
    NativePvPagedReplayError, NativePvPagedReplayLimits, NativePvPagedUpdate,
    replay_native_pv_paged_journal, resume_native_pv_paged_journal,
};
pub use paged::{
    NativePvPagedResumeBundleError, NativePvPagedResumeBundleLimits,
    NativePvPagedResumeBundleReader, PublishedNativePvPagedResumeBundle,
    publish_native_pv_paged_resume_bundle, stage_native_pv_paged_resume_bundle,
};
pub use resume::{NativePvJournalResume, NativePvResumeTail, resume_native_pv_journal};

/// Logical payload limits; excludes metadata, allocator and external model/stream memory.
#[derive(Clone, Copy, Debug)]
pub struct NativePvJournalLimits {
    /// Complete exact journal bytes, not a process-memory limit.
    pub max_journal_bytes: u64,
    /// Reference replay state/workspace, one gradient window/checkpoint and I/O scratch.
    pub max_working_bytes: u64,
}

/// Read-only exact stream and logical buffer forecast, not disk/RSS qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvJournalSizeLedger {
    /// Complete journal length for the frozen schedule.
    pub journal_bytes: u64,
    /// One serialized checkpoint and fixed I/O scratch; session/caller buffers excluded.
    pub writer_working_bytes: u64,
    /// Reference state/update workspace, one gradient window/checkpoint and I/O scratch.
    pub replay_working_bytes: u64,
}

/// Original I/O/numerical failure or a strict journal admission refusal.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvJournalError {
    /// Malformed, changed, incomplete or mismatched bytes/state.
    Invalid(&'static str),
    /// Overflow or explicit logical byte limit.
    Limit(&'static str),
    /// Original caller-owned stream error.
    Io(std::io::Error),
    /// Original common optimizer/checkpoint refusal.
    Update(PvTuningError),
}
impl core::fmt::Display for NativePvJournalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Limit(message) => f.write_str(message),
            Self::Io(error) => error.fmt(f),
            Self::Update(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for NativePvJournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Update(error) => Some(error),
            _ => None,
        }
    }
}
impl From<std::io::Error> for NativePvJournalError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<PvTuningError> for NativePvJournalError {
    fn from(error: PvTuningError) -> Self {
        Self::Update(error)
    }
}
impl From<stream::Error> for NativePvJournalError {
    fn from(error: stream::Error) -> Self {
        match error {
            stream::Error::Limit(message) => Self::Limit(message),
            stream::Error::Io(error) => Self::Io(error),
            stream::Error::Poisoned => invalid("native journal writer is poisoned"),
        }
    }
}
type Result<T> = std::result::Result<T, NativePvJournalError>;
fn invalid(message: &'static str) -> NativePvJournalError {
    NativePvJournalError::Invalid(message)
}
fn sum(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or(NativePvJournalError::Limit("native journal size overflow"))
}
fn product(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(NativePvJournalError::Limit("native journal size overflow"))
}
fn sizes(
    plan: &NativeRefinementPlan,
    ledger: NativePvSizeLedger,
    gradients: u64,
) -> Result<NativePvJournalSizeLedger> {
    let bytes = journal_bytes(plan, gradients)?;
    let working = sum(
        sum(
            sum(ledger.state_payload_bytes, ledger.update_workspace_bytes)?,
            product(gradients, 4)?,
        )?,
        sum(ledger.checkpoint_bytes, SCRATCH as u64)?,
    )?;
    Ok(NativePvJournalSizeLedger {
        journal_bytes: bytes,
        writer_working_bytes: sum(ledger.checkpoint_bytes, SCRATCH as u64)?,
        replay_working_bytes: working,
    })
}
fn journal_bytes(plan: &NativeRefinementPlan, gradients: u64) -> Result<u64> {
    sum(
        sum(176, product(16, plan.owners.len() as u64)?)?,
        product(plan.training.len() as u64, sum(72, product(4, gradients)?)?)?,
    )
}
fn frame_header(
    ordinal: u64,
    window: [u8; 32],
    frame: &mut blake3::Hasher,
    mut write: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    frame.update(FRAME_DOMAIN);
    for bytes in [&ordinal.to_le_bytes()[..], &window[..]] {
        write(bytes)?;
        frame.update(bytes);
    }
    Ok(())
}
fn frame_owner(
    scales: &[f32],
    weights: &[f32],
    frame: &mut blake3::Hasher,
    mut write: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    stream::encode(scales, frame, &mut write)?;
    stream::encode(weights, frame, write)
}
fn admit_limits(limits: NativePvJournalLimits, bytes: u64, working: u64) -> Result<()> {
    if limits.max_journal_bytes == 0
        || limits.max_working_bytes == 0
        || bytes > limits.max_journal_bytes
        || working > limits.max_working_bytes
    {
        return Err(NativePvJournalError::Limit(
            "native journal byte limit exceeded",
        ));
    }
    Ok(())
}
fn gradient_count(session: &NativePvRefinementSession) -> Result<u64> {
    session.weights().try_fold(0, |n, weight| {
        sum(
            n,
            sum(weight.scales().len() as u64, active_count(weight)? as u64)?,
        )
    })
}

/// Exact emitted journal/checkpoint observations; not verified numerical replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativePvJournalSummary {
    journal: BlobId,
    bytes: u64,
    plan: NativeRefinementPlanId,
    checkpoint: BlobId,
    checkpoint_bytes: u64,
    updates: u64,
}
impl NativePvJournalSummary {
    /// BLAKE3 of every emitted byte, including the final checksum.
    #[must_use]
    pub const fn journal(&self) -> BlobId {
        self.journal
    }
    /// Exact emitted length.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Exact frozen request identity.
    #[must_use]
    pub const fn plan(&self) -> NativeRefinementPlanId {
        self.plan
    }
    /// Actual complete TNPV byte identity, including optimizer moments.
    #[must_use]
    pub const fn checkpoint(&self) -> BlobId {
        self.checkpoint
    }
    /// Exact TNPV byte length.
    #[must_use]
    pub const fn checkpoint_bytes(&self) -> u64 {
        self.checkpoint_bytes
    }
    /// Complete declared ordered schedule length.
    #[must_use]
    pub const fn updates(&self) -> u64 {
        self.updates
    }
}

/// Optional zero-origin journal. A write failure poisons this writer, never the session.
/// Generic flush is not fsync, immutable source custody or power-loss durability.
#[derive(Debug)]
pub struct NativePvJournalWriter<W> {
    output: W,
    hasher: blake3::Hasher,
    plan: NativeRefinementPlanId,
    parent: [u8; 32],
    completed: u64,
    bytes: u64,
    expected_bytes: u64,
    poisoned: bool,
}
impl<W: Write + Seek> NativePvJournalWriter<W> {
    /// Admit exact logical bytes before writing. Destination must be empty and session at zero.
    pub fn new(
        output: W,
        session: &NativePvRefinementSession,
        limits: NativePvJournalLimits,
    ) -> Result<Self> {
        let ledger = session.journal_size_ledger()?;
        let bytes = ledger.journal_bytes;
        admit_limits(limits, bytes, ledger.writer_working_bytes)?;
        if session.completed != 0 {
            return Err(invalid("native journal requires a zero-origin session"));
        }
        let mut channels = reserved(session.owners.len())?;
        for weight in session.weights() {
            channels.push((weight.scales().len() as u64, active_count(weight)? as u64));
        }
        Self::start(
            output,
            session.plan_id,
            session.parent,
            session.plan.training.len() as u64,
            &channels,
            bytes,
        )
    }
    fn start(
        mut output: W,
        plan: NativeRefinementPlanId,
        parent: [u8; 32],
        updates: u64,
        channels: &[(u64, u64)],
        bytes: u64,
    ) -> Result<Self> {
        if output.seek(SeekFrom::End(0))? != 0 {
            return Err(invalid("native journal destination is not empty"));
        }
        if output.seek(SeekFrom::Start(0))? != 0 {
            return Err(invalid("native journal destination seek differs"));
        }
        let mut writer = Self {
            output,
            hasher: blake3::Hasher::new(),
            plan,
            parent,
            completed: 0,
            bytes: 0,
            expected_bytes: bytes,
            poisoned: false,
        };
        writer.write(&MAGIC)?;
        writer.write(plan.as_bytes())?;
        writer.write(&parent)?;
        writer.write(&updates.to_le_bytes())?;
        writer.write(&(channels.len() as u64).to_le_bytes())?;
        for &(scales, weights) in channels {
            writer.write(&scales.to_le_bytes())?;
            writer.write(&weights.to_le_bytes())?;
        }
        Ok(writer)
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        stream::write(
            &mut self.output,
            &mut self.hasher,
            &mut self.bytes,
            self.expected_bytes,
            &mut self.poisoned,
            bytes,
        )
        .map_err(Into::into)
    }
    fn flush(&mut self) -> Result<()> {
        stream::flush(&mut self.output, &mut self.poisoned).map_err(Into::into)
    }
    pub(super) fn check(&self, session: &NativePvRefinementSession) -> Result<()> {
        if self.poisoned {
            return Err(invalid("native journal writer is poisoned"));
        }
        if self.plan != session.plan_id
            || self.parent != session.parent
            || self.completed != session.completed
        {
            return Err(invalid("native journal plan/parent/cursor differs"));
        }
        Ok(())
    }
    pub(super) fn append(
        &mut self,
        ordinal: u64,
        window: [u8; 32],
        scales: &[&[f32]],
        weights: &[&[f32]],
    ) -> Result<()> {
        let mut frame = blake3::Hasher::new();
        frame_header(ordinal, window, &mut frame, |bytes| self.write(bytes))?;
        for (scale, weight) in scales.iter().zip(weights) {
            frame_owner(scale, weight, &mut frame, |bytes| self.write(bytes))?;
        }
        self.write(frame.finalize().as_bytes())?;
        self.flush()?;
        self.completed += 1;
        Ok(())
    }
    /// Snapshot one complete emitted prefix and exact checkpoint without sealing.
    /// Flush is not fsync; returned observations do not authenticate numerical history.
    pub fn checkpoint(
        &mut self,
        session: &NativePvRefinementSession,
    ) -> Result<(Vec<u8>, NativePvResumePoint)> {
        self.check(session)?;
        self.flush()?;
        self.snapshot_unflushed(session)
    }
    // Private preparation only. No public observation may escape before a successful flush.
    fn snapshot_unflushed(
        &self,
        session: &NativePvRefinementSession,
    ) -> Result<(Vec<u8>, NativePvResumePoint)> {
        self.check(session)?;
        let checkpoint = session.checkpoint_bytes()?;
        let point = NativePvResumePoint {
            plan: self.plan,
            completed_updates: self.completed,
            journal_prefix: BlobId::from_bytes(*self.hasher.finalize().as_bytes()),
            journal_prefix_bytes: self.bytes,
            checkpoint: BlobId::from_bytes(*blake3::hash(&checkpoint).as_bytes()),
            checkpoint_bytes: checkpoint.len() as u64,
        };
        Ok((checkpoint, point))
    }
    /// Seal only a complete schedule against its actual checkpoint. Caller owns persistence/custody.
    pub fn finish(
        self,
        session: &NativePvRefinementSession,
    ) -> Result<(W, NativePvJournalSummary)> {
        self.check(session)?;
        if self.completed != session.plan.training.len() as u64 {
            return Err(invalid("native journal schedule incomplete"));
        }
        let checkpoint = session.checkpoint_bytes()?;
        let checkpoint_id = BlobId::from_bytes(*blake3::hash(&checkpoint).as_bytes());
        self.finish_observed(
            session.plan.training.len() as u64,
            checkpoint.len() as u64,
            checkpoint_id,
        )
    }
    fn finish_observed(
        mut self,
        updates: u64,
        checkpoint_bytes: u64,
        checkpoint_id: BlobId,
    ) -> Result<(W, NativePvJournalSummary)> {
        if self.completed != updates {
            return Err(invalid("native journal schedule incomplete"));
        }
        self.write(&END)?;
        self.write(&self.completed.to_le_bytes())?;
        self.write(&checkpoint_bytes.to_le_bytes())?;
        self.write(checkpoint_id.as_bytes())?;
        let checksum = *self.hasher.finalize().as_bytes();
        self.write(&checksum)?;
        self.flush()?;
        if self.bytes != self.expected_bytes {
            return Err(invalid("native journal emitted length differs"));
        }
        let summary = NativePvJournalSummary {
            journal: BlobId::from_bytes(*self.hasher.finalize().as_bytes()),
            bytes: self.bytes,
            plan: self.plan,
            checkpoint: checkpoint_id,
            checkpoint_bytes,
            updates: self.completed,
        };
        Ok((self.output, summary))
    }
    /// Recover the caller stream without any completeness/replay observation.
    pub fn into_inner(self) -> W {
        self.output
    }
}

impl NativePvRefinementSession {
    /// Inspect exact journal bytes and logical recording/replay buffers without stream I/O.
    pub fn journal_size_ledger(&self) -> Result<NativePvJournalSizeLedger> {
        sizes(self.plan(), self.ledger, gradient_count(self)?)
    }
    /// Write/flush both admitted adjoints before the existing atomic state commit.
    /// A failed writer is poisoned and cannot be reused or sealed; numeric refusal writes nothing.
    pub fn step_recorded<W: Write + Seek>(
        &mut self,
        scales: &[&[f32]],
        weights: &[&[f32]],
        journal: &mut NativePvJournalWriter<W>,
    ) -> std::result::Result<NativePvStepStats, NativePvStageError<NativePvJournalError>> {
        journal.check(self).map_err(NativePvStageError::Stage)?;
        let ordinal = self.completed;
        let window = self.plan.training.get(ordinal as usize).map(|w| w.content);
        self.step_staged(scales, weights, |_| {
            journal.append(ordinal, window.expect("admitted schedule"), scales, weights)
        })
        .map(|(stats, ())| stats)
    }
}

/// Privately constructed exact optimizer replay, not objective/source/quality approval.
#[derive(Clone, Debug)]
pub struct VerifiedNativePvReplay {
    summary: NativePvJournalSummary,
    parent: [u8; 32],
}
impl VerifiedNativePvReplay {
    /// Exact replayed input/output observations.
    #[must_use]
    pub fn summary(&self) -> &NativePvJournalSummary {
        &self.summary
    }
    /// Require the same current state, not merely the same plan or update cursor.
    pub fn verify_session(&self, session: &NativePvRefinementSession) -> Result<()> {
        if self.summary.plan != session.plan_id
            || self.parent != session.parent
            || self.summary.updates != session.completed
        {
            return Err(invalid("native replay session identity differs"));
        }
        let bytes = session.checkpoint_bytes()?;
        if bytes.len() as u64 != self.summary.checkpoint_bytes
            || blake3::hash(&bytes).as_bytes() != self.summary.checkpoint.as_bytes()
        {
            return Err(invalid("native replay current checkpoint differs"));
        }
        Ok(())
    }
}

struct Reader<R> {
    source: R,
    hash: blake3::Hasher,
    bytes: u64,
}
impl<R: Read> Reader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        stream::read(&mut self.source, &mut self.hash, &mut self.bytes, bytes).map_err(Into::into)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read(&mut bytes)?;
        Ok(bytes)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
}

/// Reproduce the complete exact checkpoint through common updates from original parents.
/// Original parent admission/objective authentication and immutable stream custody are external.
/// Checks full raw identity before optimizer allocation, then frame/final identities during replay.
pub fn replay_native_pv_journal<R: Read + Seek>(
    plan: NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
    source: R,
    expected_journal: BlobId,
    expected_checkpoint: &[u8],
    limits: NativePvJournalLimits,
) -> Result<VerifiedNativePvReplay> {
    replay_native_pv_journal_with_state(
        plan,
        parents,
        source,
        expected_journal,
        expected_checkpoint,
        limits,
    )
    .map(|(_, observation)| observation)
}

/// Same exact replayer, retaining its final common state for encoded-child binding.
/// Consumes original parents once; does not clone coefficients or alter the recipe.
pub fn replay_native_pv_journal_with_state<R: Read + Seek>(
    plan: NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
    mut source: R,
    expected_journal: BlobId,
    expected_checkpoint: &[u8],
    limits: NativePvJournalLimits,
) -> Result<(NativePvRefinementSession, VerifiedNativePvReplay)> {
    let admission = admit(plan, &parents)?;
    let gradients = parents.iter().try_fold(0, |n, w| {
        sum(n, sum(w.scales().len() as u64, active_count(w)? as u64)?)
    })?;
    let ledger = sizes(admission.plan(), admission.ledger, gradients)?;
    let bytes = ledger.journal_bytes;
    admit_limits(limits, bytes, ledger.replay_working_bytes)?;
    if expected_checkpoint.len() as u64 != admission.ledger.checkpoint_bytes {
        return Err(invalid("native replay checkpoint length differs"));
    }
    if source.seek(SeekFrom::End(0))? != bytes || source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native journal source length/seek differs"));
    }
    let mut scratch = [0; SCRATCH];
    let hash = stream::hash_exact(&mut source, bytes, &mut scratch)?;
    if &hash != expected_journal.as_bytes() {
        return Err(invalid("native journal source identity differs"));
    }
    if source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native journal source seek differs"));
    }
    let mut input = Reader {
        source,
        hash: blake3::Hasher::new(),
        bytes: 0,
    };
    let updates = admission.plan.training.len();
    let session = replay_frames(&mut input, admission, parents, updates, &mut scratch)?;
    if input.array::<8>()? != END
        || input.u64()? != session.completed
        || input.u64()? != expected_checkpoint.len() as u64
    {
        return Err(invalid("native journal trailer differs"));
    }
    let checkpoint = BlobId::from_bytes(input.array()?);
    let checksum = *input.hash.finalize().as_bytes();
    if input.array::<32>()? != checksum
        || input.bytes != bytes
        || input.hash.finalize().as_bytes() != expected_journal.as_bytes()
        || input.source.read(&mut [0])? != 0
    {
        return Err(invalid("native journal terminal identity/length differs"));
    }
    if blake3::hash(expected_checkpoint).as_bytes() != checkpoint.as_bytes()
        || session.checkpoint_bytes()?.as_slice() != expected_checkpoint
    {
        return Err(invalid("native replay exact checkpoint differs"));
    }
    let observation = VerifiedNativePvReplay {
        parent: session.parent,
        summary: NativePvJournalSummary {
            journal: expected_journal,
            bytes,
            plan: session.plan_id,
            checkpoint,
            checkpoint_bytes: expected_checkpoint.len() as u64,
            updates: session.completed,
        },
    };
    Ok((session, observation))
}

fn replay_frames<R: Read>(
    input: &mut Reader<R>,
    admission: NativePvRefinementSession,
    parents: Vec<AdditiveTensor>,
    updates: usize,
    scratch: &mut [u8; SCRATCH],
) -> Result<NativePvRefinementSession> {
    if updates > admission.plan.training.len() {
        return Err(invalid("native replay cursor exceeds schedule"));
    }
    let mut channels = reserved(parents.len())?;
    for weight in &parents {
        channels.push((weight.scales().len() as u64, active_count(weight)? as u64));
    }
    replay_header(
        input,
        admission.plan_id,
        admission.parent,
        admission.plan.training.len() as u64,
        &channels,
    )?;
    let mut session = admission.with_parents(parents)?;
    let mut scale_gradients = reserved(session.owners.len())?;
    let mut weight_gradients = reserved(session.owners.len())?;
    for weight in session.weights() {
        scale_gradients.push(zeroed(weight.scales().len())?);
        weight_gradients.push(zeroed(active_count(weight)?)?);
    }
    for ordinal in 0..updates {
        let mut frame = replay_frame_header(
            input,
            ordinal as u64,
            session.plan.training[ordinal].content,
        )?;
        for (scale, weight) in scale_gradients.iter_mut().zip(&mut weight_gradients) {
            replay_owner(input, scale, weight, scratch, &mut frame)?;
        }
        if input.array::<32>()? != *frame.finalize().as_bytes() {
            return Err(invalid("native journal frame checksum differs"));
        }
        session.step(
            &scale_gradients
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>(),
            &weight_gradients
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(session)
}

fn replay_header<R: Read>(
    input: &mut Reader<R>,
    plan: NativeRefinementPlanId,
    parent: [u8; 32],
    updates: u64,
    channels: &[(u64, u64)],
) -> Result<()> {
    if input.array::<8>()? != MAGIC
        || input.array::<32>()? != *plan.as_bytes()
        || input.array::<32>()? != parent
        || input.u64()? != updates
        || input.u64()? != channels.len() as u64
    {
        return Err(invalid("native journal header differs"));
    }
    for &(scale, weight) in channels {
        if input.u64()? != scale || input.u64()? != weight {
            return Err(invalid("native journal owner gradient counts differ"));
        }
    }
    Ok(())
}
fn replay_frame_header<R: Read>(
    input: &mut Reader<R>,
    ordinal: u64,
    expected: [u8; 32],
) -> Result<blake3::Hasher> {
    let encoded = input.array::<8>()?;
    let window = input.array::<32>()?;
    if u64::from_le_bytes(encoded) != ordinal || window != expected {
        return Err(invalid("native journal window/order differs"));
    }
    let mut frame = blake3::Hasher::new();
    frame.update(FRAME_DOMAIN);
    frame.update(&encoded);
    frame.update(&window);
    Ok(frame)
}
fn replay_owner<R: Read>(
    input: &mut Reader<R>,
    scales: &mut [f32],
    weights: &mut [f32],
    scratch: &mut [u8; SCRATCH],
    frame: &mut blake3::Hasher,
) -> Result<()> {
    stream::decode(scales, scratch, frame, |bytes| input.read(bytes))?;
    stream::decode(weights, scratch, frame, |bytes| input.read(bytes))
}
