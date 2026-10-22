//! Streamed numerical replay, not objective authentication or publication approval.
use super::*;
use std::io::{Read, Seek, SeekFrom, Write};
use tritium_schema::BlobId;

const MAGIC: [u8; 8] = *b"TNSJ\0\0\0\x01";
const END: [u8; 8] = *b"TNSJEND1";
const SCRATCH: usize = 8192;
const FRAME_DOMAIN: &[u8] = b"tritium.native-scale.journal-frame.v1\0";

/// Logical payload limits; excludes metadata, allocator and external model/stream memory.
#[derive(Clone, Copy, Debug)]
pub struct NativeScaleJournalLimits {
    /// Complete exact journal bytes, not a process-memory limit.
    pub max_journal_bytes: u64,
    /// Reference replay state/workspace, one gradient window/checkpoint and I/O scratch.
    pub max_working_bytes: u64,
}

/// Read-only exact stream and logical buffer forecast, not disk/RSS qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeScaleJournalSizeLedger {
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
pub enum NativeScaleJournalError {
    /// Malformed, changed, incomplete or mismatched bytes/state.
    Invalid(&'static str),
    /// Overflow or explicit logical byte limit.
    Limit(&'static str),
    /// Original caller-owned stream error.
    Io(std::io::Error),
    /// Original common optimizer/checkpoint refusal.
    Update(PvTuningError),
}
impl core::fmt::Display for NativeScaleJournalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Limit(message) => f.write_str(message),
            Self::Io(error) => error.fmt(f),
            Self::Update(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for NativeScaleJournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Update(error) => Some(error),
            _ => None,
        }
    }
}
impl From<std::io::Error> for NativeScaleJournalError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<PvTuningError> for NativeScaleJournalError {
    fn from(error: PvTuningError) -> Self {
        Self::Update(error)
    }
}
type Result<T> = std::result::Result<T, NativeScaleJournalError>;
fn invalid(message: &'static str) -> NativeScaleJournalError {
    NativeScaleJournalError::Invalid(message)
}
fn sum(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(NativeScaleJournalError::Limit(
        "native journal size overflow",
    ))
}
fn product(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(NativeScaleJournalError::Limit(
        "native journal size overflow",
    ))
}
fn sizes(
    plan: &NativeRefinementPlan,
    ledger: NativeScaleSizeLedger,
    scales: u64,
) -> Result<NativeScaleJournalSizeLedger> {
    let bytes = sum(
        sum(176, product(8, plan.owners.len() as u64)?)?,
        product(plan.training.len() as u64, sum(72, product(4, scales)?)?)?,
    )?;
    let working = sum(
        sum(
            sum(ledger.state_payload_bytes, ledger.update_workspace_bytes)?,
            product(scales, 4)?,
        )?,
        sum(ledger.checkpoint_bytes, SCRATCH as u64)?,
    )?;
    Ok(NativeScaleJournalSizeLedger {
        journal_bytes: bytes,
        writer_working_bytes: sum(ledger.checkpoint_bytes, SCRATCH as u64)?,
        replay_working_bytes: working,
    })
}
fn admit_limits(limits: NativeScaleJournalLimits, bytes: u64, working: u64) -> Result<()> {
    if limits.max_journal_bytes == 0
        || limits.max_working_bytes == 0
        || bytes > limits.max_journal_bytes
        || working > limits.max_working_bytes
    {
        return Err(NativeScaleJournalError::Limit(
            "native journal byte limit exceeded",
        ));
    }
    Ok(())
}
fn scale_count(session: &NativeScaleRefinementSession) -> Result<u64> {
    session
        .weights()
        .try_fold(0, |n, weight| sum(n, weight.scales().len() as u64))
}

/// Exact emitted journal/checkpoint observations; not verified numerical replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeScaleJournalSummary {
    journal: BlobId,
    bytes: u64,
    plan: NativeRefinementPlanId,
    checkpoint: BlobId,
    checkpoint_bytes: u64,
    updates: u64,
}
impl NativeScaleJournalSummary {
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
    /// Actual complete TNSR byte identity, including optimizer moments.
    #[must_use]
    pub const fn checkpoint(&self) -> BlobId {
        self.checkpoint
    }
    /// Exact TNSR byte length.
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
pub struct NativeScaleJournalWriter<W> {
    output: W,
    hasher: blake3::Hasher,
    plan: NativeRefinementPlanId,
    parent: [u8; 32],
    completed: u64,
    bytes: u64,
    expected_bytes: u64,
    poisoned: bool,
}
impl<W: Write + Seek> NativeScaleJournalWriter<W> {
    /// Admit exact logical bytes before writing. Destination must be empty and session at zero.
    pub fn new(
        mut output: W,
        session: &NativeScaleRefinementSession,
        limits: NativeScaleJournalLimits,
    ) -> Result<Self> {
        let ledger = session.journal_size_ledger()?;
        let bytes = ledger.journal_bytes;
        admit_limits(limits, bytes, ledger.writer_working_bytes)?;
        if session.completed_updates != 0 {
            return Err(invalid("native journal requires a zero-origin session"));
        }
        if output.seek(SeekFrom::End(0))? != 0 {
            return Err(invalid("native journal destination is not empty"));
        }
        if output.seek(SeekFrom::Start(0))? != 0 {
            return Err(invalid("native journal destination seek differs"));
        }
        let mut writer = Self {
            output,
            hasher: blake3::Hasher::new(),
            plan: session.plan_id,
            parent: session.parent,
            completed: 0,
            bytes: 0,
            expected_bytes: bytes,
            poisoned: false,
        };
        writer.write(&MAGIC)?;
        writer.write(session.plan_id.as_bytes())?;
        writer.write(&session.parent)?;
        writer.write(&(session.plan.training.len() as u64).to_le_bytes())?;
        writer.write(&(session.owners.len() as u64).to_le_bytes())?;
        for weight in session.weights() {
            writer.write(&(weight.scales().len() as u64).to_le_bytes())?;
        }
        Ok(writer)
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if self.poisoned {
            return Err(invalid("native journal writer is poisoned"));
        }
        let next = sum(self.bytes, bytes.len() as u64)?;
        if next > self.expected_bytes {
            return Err(NativeScaleJournalError::Limit(
                "native journal emitted bytes exceed limit",
            ));
        }
        // Stay poisoned on an unwinding custom stream, not just an I/O Result.
        self.poisoned = true;
        if let Err(error) = self.output.write_all(bytes) {
            return Err(error.into());
        }
        self.hasher.update(bytes);
        self.bytes = next;
        self.poisoned = false;
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        self.poisoned = true;
        if let Err(error) = self.output.flush() {
            return Err(error.into());
        }
        self.poisoned = false;
        Ok(())
    }
    pub(super) fn check(&self, session: &NativeScaleRefinementSession) -> Result<()> {
        if self.poisoned {
            return Err(invalid("native journal writer is poisoned"));
        }
        if self.plan != session.plan_id
            || self.parent != session.parent
            || self.completed != session.completed_updates
        {
            return Err(invalid("native journal plan/parent/cursor differs"));
        }
        Ok(())
    }
    pub(super) fn append(
        &mut self,
        ordinal: u64,
        window: [u8; 32],
        gradients: &[&[f32]],
    ) -> Result<()> {
        let mut frame = blake3::Hasher::new();
        frame.update(FRAME_DOMAIN);
        for bytes in [&ordinal.to_le_bytes()[..], &window[..]] {
            self.write(bytes)?;
            frame.update(bytes);
        }
        let mut scratch = [0; SCRATCH];
        for gradient in gradients {
            for chunk in gradient.chunks(SCRATCH / 4) {
                for (bytes, value) in scratch.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
                    bytes.copy_from_slice(&value.to_le_bytes());
                }
                let bytes = &scratch[..chunk.len() * 4];
                self.write(bytes)?;
                frame.update(bytes);
            }
        }
        self.write(frame.finalize().as_bytes())?;
        self.flush()?;
        self.completed += 1;
        Ok(())
    }
    /// Seal only a complete schedule against its actual checkpoint. Caller owns persistence/custody.
    pub fn finish(
        mut self,
        session: &NativeScaleRefinementSession,
    ) -> Result<(W, NativeScaleJournalSummary)> {
        self.check(session)?;
        if self.completed != session.plan.training.len() as u64 {
            return Err(invalid("native journal schedule incomplete"));
        }
        let checkpoint = session.checkpoint_bytes()?;
        let checkpoint_id = BlobId::from_bytes(*blake3::hash(&checkpoint).as_bytes());
        self.write(&END)?;
        self.write(&self.completed.to_le_bytes())?;
        self.write(&(checkpoint.len() as u64).to_le_bytes())?;
        self.write(checkpoint_id.as_bytes())?;
        let checksum = *self.hasher.finalize().as_bytes();
        self.write(&checksum)?;
        self.flush()?;
        if self.bytes != self.expected_bytes {
            return Err(invalid("native journal emitted length differs"));
        }
        let summary = NativeScaleJournalSummary {
            journal: BlobId::from_bytes(*self.hasher.finalize().as_bytes()),
            bytes: self.bytes,
            plan: self.plan,
            checkpoint: checkpoint_id,
            checkpoint_bytes: checkpoint.len() as u64,
            updates: self.completed,
        };
        Ok((self.output, summary))
    }
    /// Recover the caller stream without any completeness/replay observation.
    pub fn into_inner(self) -> W {
        self.output
    }
}

impl NativeScaleRefinementSession {
    /// Inspect exact journal bytes and logical recording/replay buffers without stream I/O.
    pub fn journal_size_ledger(&self) -> Result<NativeScaleJournalSizeLedger> {
        sizes(self.plan(), self.ledger, scale_count(self)?)
    }
    /// Write/flush admitted gradients before the existing atomic state commit.
    /// A failed writer is poisoned and cannot be reused or sealed; numeric refusal writes nothing.
    pub fn step_recorded<W: Write + Seek>(
        &mut self,
        gradients: &[&[f32]],
        journal: &mut NativeScaleJournalWriter<W>,
    ) -> std::result::Result<usize, NativeScaleStageError<NativeScaleJournalError>> {
        journal.check(self).map_err(NativeScaleStageError::Stage)?;
        let ordinal = self.completed_updates;
        let window = self.plan.training.get(ordinal as usize).map(|w| w.content);
        self.step_staged(gradients, |_| {
            journal.append(ordinal, window.expect("admitted schedule"), gradients)
        })
        .map(|(changed, ())| changed)
    }
}

/// Privately constructed exact optimizer replay, not objective/source/quality approval.
#[derive(Clone, Debug)]
pub struct VerifiedNativeScaleReplay {
    summary: NativeScaleJournalSummary,
    parent: [u8; 32],
}
impl VerifiedNativeScaleReplay {
    /// Exact replayed input/output observations.
    #[must_use]
    pub fn summary(&self) -> &NativeScaleJournalSummary {
        &self.summary
    }
    /// Require the same current state, not merely the same plan or update cursor.
    pub fn verify_session(&self, session: &NativeScaleRefinementSession) -> Result<()> {
        if self.summary.plan != session.plan_id
            || self.parent != session.parent
            || self.summary.updates != session.completed_updates
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
        self.source.read_exact(bytes)?;
        self.hash.update(bytes);
        self.bytes = sum(self.bytes, bytes.len() as u64)?;
        Ok(())
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
pub fn replay_native_scale_journal<R: Read + Seek>(
    plan: NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
    source: R,
    expected_journal: BlobId,
    expected_checkpoint: &[u8],
    limits: NativeScaleJournalLimits,
) -> Result<VerifiedNativeScaleReplay> {
    replay_native_scale_journal_with_state(
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
pub fn replay_native_scale_journal_with_state<R: Read + Seek>(
    plan: NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
    mut source: R,
    expected_journal: BlobId,
    expected_checkpoint: &[u8],
    limits: NativeScaleJournalLimits,
) -> Result<(NativeScaleRefinementSession, VerifiedNativeScaleReplay)> {
    let admission = admit(&plan, &parents)?;
    let scales = parents
        .iter()
        .try_fold(0, |n, w| sum(n, w.scales().len() as u64))?;
    let ledger = sizes(&plan, admission.ledger, scales)?;
    let bytes = ledger.journal_bytes;
    admit_limits(limits, bytes, ledger.replay_working_bytes)?;
    if expected_checkpoint.len() as u64 != admission.ledger.checkpoint_bytes {
        return Err(invalid("native replay checkpoint length differs"));
    }
    if source.seek(SeekFrom::End(0))? != bytes || source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native journal source length/seek differs"));
    }
    let mut scratch = [0; SCRATCH];
    let mut remaining = bytes;
    let mut hash = blake3::Hasher::new();
    while remaining > 0 {
        let count = usize::try_from(remaining.min(SCRATCH as u64)).expect("bounded chunk");
        source.read_exact(&mut scratch[..count])?;
        hash.update(&scratch[..count]);
        remaining -= count as u64;
    }
    if hash.finalize().as_bytes() != expected_journal.as_bytes() {
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
    if input.array::<8>()? != MAGIC
        || input.array::<32>()? != *admission.plan_id.as_bytes()
        || input.array::<32>()? != admission.parent
        || input.u64()? != plan.training.len() as u64
        || input.u64()? != plan.owners.len() as u64
    {
        return Err(invalid("native journal header differs"));
    }
    for weight in &parents {
        if input.u64()? != weight.scales().len() as u64 {
            return Err(invalid("native journal owner scale count differs"));
        }
    }
    let mut session = NativeScaleRefinementSession::new(plan, parents)?;
    let counts: Vec<_> = session.weights().map(|w| w.scales().len()).collect();
    let mut gradients = reserved(counts.len())?;
    for &count in &counts {
        gradients.push(zeroed(count)?);
    }
    for ordinal in 0..session.plan.training.len() {
        let encoded_ordinal = input.array::<8>()?;
        let window = input.array::<32>()?;
        if u64::from_le_bytes(encoded_ordinal) != ordinal as u64
            || window != session.plan.training[ordinal].content
        {
            return Err(invalid("native journal window/order differs"));
        }
        let mut frame = blake3::Hasher::new();
        frame.update(FRAME_DOMAIN);
        frame.update(&encoded_ordinal);
        frame.update(&window);
        for owner in &mut gradients {
            for chunk in owner.chunks_mut(SCRATCH / 4) {
                let bytes = &mut scratch[..chunk.len() * 4];
                input.read(bytes)?;
                frame.update(bytes);
                for (value, encoded) in chunk.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*encoded);
                }
            }
        }
        if input.array::<32>()? != *frame.finalize().as_bytes() {
            return Err(invalid("native journal frame checksum differs"));
        }
        session.step(&gradients.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    }
    if input.array::<8>()? != END
        || input.u64()? != session.completed_updates
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
    let observation = VerifiedNativeScaleReplay {
        parent: session.parent,
        summary: NativeScaleJournalSummary {
            journal: expected_journal,
            bytes,
            plan: session.plan_id,
            checkpoint,
            checkpoint_bytes: expected_checkpoint.len() as u64,
            updates: session.completed_updates,
        },
    };
    Ok((session, observation))
}
