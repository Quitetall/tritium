//! Replay an admitted prefix into a new stream, never mutate recovery evidence.
use super::*;

/// Explicit treatment of bytes after the chosen checkpoint boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativePvResumeTail {
    /// Source must end exactly after the bound complete frame prefix.
    #[default]
    ExactPrefix,
    /// Admit/copy only the bound prefix; preserve the unparsed source suffix untouched.
    CheckpointPrefix,
}

/// Untrusted saved observations plus actual original inputs for strict continuation.
#[derive(Debug)]
pub struct NativePvJournalResume<'a, R> {
    /// Exact frozen full request; schedule/identities cannot change on resume.
    pub plan: NativeRefinementPlan,
    /// Actual original parent tensors, consumed once into numerical replay.
    pub parents: Vec<AdditiveTensor>,
    /// Original journal/prefix source, never written or truncated by this function.
    pub source: R,
    /// Saved emitted prefix/checkpoint binding; not trusted as numerical proof.
    pub point: &'a NativePvResumePoint,
    /// Actual saved complete TNPV bytes.
    pub checkpoint: &'a [u8],
    /// Explicit suffix policy, with strict exact-prefix behavior available.
    pub tail: NativePvResumeTail,
}

pub(super) fn source_length(length: u64, prefix: u64, tail: NativePvResumeTail) -> Result<()> {
    if length < prefix || (tail == NativePvResumeTail::ExactPrefix && length != prefix) {
        return Err(invalid("native resume source length/tail differs"));
    }
    Ok(())
}

/// Reproduce a saved prefix/checkpoint and continue in an empty new output.
/// No source bytes are modified. A later copy/I/O failure may leave only an
/// unusable private destination tail; caller owns its cleanup and source custody.
pub fn resume_native_pv_journal<R: Read + Seek, W: Write + Seek>(
    request: NativePvJournalResume<'_, R>,
    mut output: W,
    limits: NativePvJournalLimits,
) -> Result<(NativePvRefinementSession, NativePvJournalWriter<W>)> {
    let NativePvJournalResume {
        plan,
        parents,
        mut source,
        point,
        checkpoint,
        tail,
    } = request;
    let admission = admit(plan, &parents)?;
    let gradients = parents.iter().try_fold(0, |n, w| {
        sum(n, sum(w.scales().len() as u64, active_count(w)? as u64)?)
    })?;
    let ledger = sizes(admission.plan(), admission.ledger, gradients)?;
    admit_limits(limits, ledger.journal_bytes, ledger.replay_working_bytes)?;
    if point.plan != admission.plan_id
        || point.completed_updates > admission.plan.training.len() as u64
        || point.checkpoint_bytes != admission.ledger.checkpoint_bytes
        || checkpoint.len() as u64 != point.checkpoint_bytes
        || blake3::hash(checkpoint).as_bytes() != point.checkpoint.as_bytes()
    {
        return Err(invalid(
            "native resume checkpoint/plan/cursor binding differs",
        ));
    }
    let prefix = sum(
        sum(88, product(16, admission.plan.owners.len() as u64)?)?,
        product(point.completed_updates, sum(72, product(4, gradients)?)?)?,
    )?;
    if prefix != point.journal_prefix_bytes {
        return Err(invalid("native resume prefix length differs"));
    }
    source_length(source.seek(SeekFrom::End(0))?, prefix, tail)?;
    if source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native resume source seek differs"));
    }
    if output.seek(SeekFrom::End(0))? != 0 {
        return Err(invalid("native resume destination is not empty"));
    }
    if output.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native resume destination seek differs"));
    }
    let mut scratch = [0; SCRATCH];
    let mut remaining = prefix;
    let mut hash = blake3::Hasher::new();
    while remaining > 0 {
        let n = remaining.min(SCRATCH as u64) as usize;
        source.read_exact(&mut scratch[..n])?;
        hash.update(&scratch[..n]);
        remaining -= n as u64;
    }
    if hash.finalize().as_bytes() != point.journal_prefix.as_bytes() {
        return Err(invalid("native resume prefix identity differs"));
    }
    if source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native resume source seek differs"));
    }
    let mut input = Reader {
        source,
        hash: blake3::Hasher::new(),
        bytes: 0,
    };
    let updates = usize::try_from(point.completed_updates)
        .map_err(|_| invalid("native resume cursor width"))?;
    let session = replay_frames(&mut input, admission, parents, updates, &mut scratch)?;
    if input.bytes != prefix || input.hash.finalize().as_bytes() != point.journal_prefix.as_bytes()
    {
        return Err(invalid("native resume replay prefix changed"));
    }
    if session.checkpoint_bytes()?.as_slice() != checkpoint {
        return Err(invalid("native resume replayed checkpoint differs"));
    }
    let mut source = input.source;
    source_length(source.seek(SeekFrom::End(0))?, prefix, tail)?;
    if source.seek(SeekFrom::Start(0))? != 0 {
        return Err(invalid("native resume source seek differs"));
    }
    let mut writer = NativePvJournalWriter {
        output,
        hasher: blake3::Hasher::new(),
        plan: session.plan_id,
        parent: session.parent,
        completed: session.completed,
        bytes: 0,
        expected_bytes: ledger.journal_bytes,
        poisoned: false,
    };
    let mut remaining = prefix;
    while remaining > 0 {
        let n = remaining.min(SCRATCH as u64) as usize;
        source.read_exact(&mut scratch[..n])?;
        writer.write(&scratch[..n])?;
        remaining -= n as u64;
    }
    if writer.bytes != prefix
        || writer.hasher.finalize().as_bytes() != point.journal_prefix.as_bytes()
    {
        return Err(invalid("native resume copied prefix changed"));
    }
    source_length(source.seek(SeekFrom::End(0))?, prefix, tail)?;
    writer.flush()?;
    if writer.output.seek(SeekFrom::End(0))? != prefix
        || writer.output.seek(SeekFrom::Start(prefix))? != prefix
    {
        return Err(invalid("native resume destination extent/seek differs"));
    }
    writer.check(&session)?;
    Ok((session, writer))
}
