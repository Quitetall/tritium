//! Owner-paged TNPV execution. No owner payload survives between calls.
use super::*;
use checkpoint::{HashRead, HashWrite, Header, Inspection, WireError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use tritium_schema::BlobId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PagedJournalState {
    pub(super) plan: NativeRefinementPlanId,
    pub(super) parent: [u8; 32],
    pub(super) completed: u64,
    pub(super) checkpoint: BlobId,
    pub(super) checkpoint_bytes: u64,
}

/// Extra residency limit; original plan caps still cover aggregate logical state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedLimits {
    /// Maximum per-owner state+update+returned adjoints+16384 I/O bytes.
    /// Excludes provider internals, stream buffers, staged resources, metadata, allocator slack
    /// and the OS page cache. This is not a process RSS or VRAM limit.
    pub max_resident_payload_bytes: u64,
}
/// Logical storage and worst-owner numerical payload, not physical memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvPagedSizeLedger {
    /// Same aggregate state/workspace/checkpoint ledger as the resident engine.
    pub logical: NativePvSizeLedger,
    /// Largest owner's state+workspace+4*(stored scales+present weights)+16384.
    pub max_owner_payload_bytes: u64,
}
/// A storage/numerical refusal or the caller's unchanged provider/stager error.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvPagedError<E> {
    /// Frozen plan, original parent, state, cap, or numerical validation failed.
    State(PvTuningError),
    /// Original stream I/O error.
    Io(io::Error),
    /// Original parent/gradient/staging callback error.
    Provider(E),
}
impl<E> From<PvTuningError> for NativePvPagedError<E> {
    fn from(e: PvTuningError) -> Self {
        Self::State(e)
    }
}
impl<E> From<io::Error> for NativePvPagedError<E> {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl<E> From<WireError> for NativePvPagedError<E> {
    fn from(e: WireError) -> Self {
        match e {
            WireError::Io(e) => Self::Io(e),
            WireError::Invalid(e) => Self::State(e),
        }
    }
}
impl<E: std::fmt::Display> std::fmt::Display for NativePvPagedError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::State(e) => e.fmt(f),
            Self::Io(e) => write!(f, "native PV paging I/O: {e}"),
            Self::Provider(e) => write!(f, "native PV paging provider: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvPagedError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Provider(e) => Some(e),
        }
    }
}

/// Seek-backed standard TNPV state with one-owner numerical residency.
///
/// Owns only the checkpoint stream, frozen plan and small metadata. Providers
/// must load original parents individually, in frozen order, on every traversal;
/// no rewritten one-owner request or cached all-owner payload is introduced.
/// Use file-backed streams for out-of-core execution: an in-memory stream
/// retains its complete encoded bytes outside the owner-payload limit.
/// This does not authenticate source custody, objective/gradient execution,
/// journal replay, fsync durability, GPU execution or model quality.
pub struct NativePvPagedSession<R> {
    checkpoint: R,
    plan: NativeRefinementPlan,
    updater: PvUpdater,
    observed: Inspection,
    ledger: NativePvPagedSizeLedger,
    limits: NativePvPagedLimits,
    channels: Vec<(u64, u64)>,
}
impl<R> std::fmt::Debug for NativePvPagedSession<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePvPagedSession")
            .field("plan_id", &self.observed.header.plan)
            .field("completed_updates", &self.observed.header.completed)
            .field("ledger", &self.ledger)
            .finish_non_exhaustive()
    }
}
fn invalid(text: &str) -> PvTuningError {
    PvTuningError::checkpoint(text)
}
pub(super) fn empty_ledger() -> NativePvPagedSizeLedger {
    NativePvPagedSizeLedger {
        logical: NativePvSizeLedger {
            state_payload_bytes: 0,
            update_workspace_bytes: 0,
            checkpoint_bytes: checkpoint::FIXED_BYTES,
        },
        max_owner_payload_bytes: 0,
    }
}
pub(super) fn account(
    plan: &NativeRefinementPlan,
    limits: NativePvPagedLimits,
    fraction: f32,
    weight: &AdditiveTensor,
    ledger: &mut NativePvPagedSizeLedger,
) -> Result<(), PvTuningError> {
    let owner = owner_sizes(weight, fraction)?;
    add_sizes(&mut ledger.logical, owner)?;
    check_sizes(plan, ledger.logical)?;
    let adjoints = (weight.scales().len() as u64)
        .checked_add(active_count(weight)? as u64)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(size_overflow)?;
    let peak = owner
        .state_payload_bytes
        .checked_add(owner.update_workspace_bytes)
        .and_then(|n| n.checked_add(adjoints))
        .and_then(|n| n.checked_add(16384))
        .ok_or_else(size_overflow)?;
    ledger.max_owner_payload_bytes = ledger.max_owner_payload_bytes.max(peak);
    if ledger.max_owner_payload_bytes > limits.max_resident_payload_bytes {
        return Err(PvTuningError::invalid_config(
            "native PV owner-resident payload cap exceeded",
        ));
    }
    Ok(())
}
fn configuration(
    plan: &NativeRefinementPlan,
    limits: NativePvPagedLimits,
) -> Result<(PvUpdater, NativeRefinementPlanId), PvTuningError> {
    if limits.max_resident_payload_bytes < 16384 {
        return Err(PvTuningError::invalid_config(
            "native PV owner-resident payload cap below I/O scratch",
        ));
    }
    let updater = PvUpdater::from_plan(plan)?;
    let id = native_refinement_plan_id(plan)
        .map_err(|e| PvTuningError::invalid_config(e.to_string()))?;
    Ok((updater, id))
}
fn empty_output(output: &mut (impl Write + Seek)) -> Result<(), WireError> {
    if output.seek(SeekFrom::End(0))? != 0 {
        return Err(invalid("native PV paging destination must be empty").into());
    }
    output.rewind()?;
    Ok(())
}

impl<R: Read + Seek> NativePvPagedSession<R> {
    /// Initialize standard TNPV state directly from one original parent at a time.
    /// Destination must be empty. Failure leaves only a caller-owned partial
    /// destination; no session is returned. Flush is not fsync durability.
    pub fn initialize<E>(
        plan: NativeRefinementPlan,
        limits: NativePvPagedLimits,
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        mut destination: R,
    ) -> Result<Self, NativePvPagedError<E>>
    where
        R: Write,
    {
        let (updater, id) = configuration(&plan, limits)?;
        empty_output(&mut destination)?;
        destination.write_all(&[0; 85])?;
        let mut ledger = empty_ledger();
        let mut channels = reserved(plan.owners.len())?;
        let mut parent_hash = NativePvRefinementSession::parent_hasher(id);
        let suffix = {
            let mut writer = HashWrite {
                output: &mut destination,
                hash: blake3::Hasher::new(),
            };
            for (i, name) in plan.owners.iter().enumerate() {
                let weight = parents(i, name).map_err(NativePvPagedError::Provider)?;
                account(&plan, limits, updater.fraction, &weight, &mut ledger)?;
                channels.push((weight.scales().len() as u64, active_count(&weight)? as u64));
                hash_parent_tensor(&mut parent_hash, &weight)?;
                checkpoint::write_owner::<8192>(&mut writer, &initial_owner(weight)?)?;
            }
            *writer.hash.finalize().as_bytes()
        };
        let header = Header {
            plan: id,
            parent: *parent_hash.finalize().as_bytes(),
            completed: 0,
            owners: plan.owners.len() as u64,
        };
        destination.rewind()?;
        destination.write_all(&header.bytes())?;
        let observed = checkpoint::seal(
            &mut destination,
            header,
            ledger.logical.checkpoint_bytes,
            suffix,
        )?;
        Ok(Self {
            checkpoint: destination,
            plan,
            updater,
            observed,
            ledger,
            limits,
            channels,
        })
    }

    /// Strictly reopen against individually supplied original parents, validating
    /// all owner values/moments/caps and the actual bytes read during traversal.
    /// This is checkpoint integrity, not numerical replay or model provenance.
    pub fn reopen<E>(
        plan: NativeRefinementPlan,
        limits: NativePvPagedLimits,
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        mut source: R,
    ) -> Result<Self, NativePvPagedError<E>> {
        let (updater, id) = configuration(&plan, limits)?;
        let observed = checkpoint::inspect(&mut source, plan.limits.max_checkpoint_bytes)?;
        if observed.header.plan != id
            || observed.header.owners != plan.owners.len() as u64
            || observed.header.completed > plan.training.len() as u64
        {
            return Err(invalid("native PV paging plan/cursor/owner mismatch").into());
        }
        source.rewind()?;
        let mut ledger = empty_ledger();
        let mut channels = reserved(plan.owners.len())?;
        let mut parent_hash = NativePvRefinementSession::parent_hasher(id);
        let traversed = {
            let mut reader = HashRead {
                source: &mut source,
                hash: blake3::Hasher::new(),
            };
            if checkpoint::read_header(&mut reader)? != observed.header {
                return Err(invalid("native PV paging header drift").into());
            }
            for (i, name) in plan.owners.iter().enumerate() {
                let weight = parents(i, name).map_err(NativePvPagedError::Provider)?;
                account(&plan, limits, updater.fraction, &weight, &mut ledger)?;
                channels.push((weight.scales().len() as u64, active_count(&weight)? as u64));
                hash_parent_tensor(&mut parent_hash, &weight)?;
                let _owner = checkpoint::read_owner::<8192>(
                    &mut reader,
                    weight,
                    observed.header.completed,
                    updater,
                )?;
            }
            *reader.hash.finalize().as_bytes()
        };
        if traversed != observed.body
            || *parent_hash.finalize().as_bytes() != observed.header.parent
            || ledger.logical.checkpoint_bytes != observed.bytes
        {
            return Err(invalid("native PV paging parent/body/length mismatch").into());
        }
        if checkpoint::inspect(&mut source, observed.bytes)? != observed {
            return Err(invalid("native PV paging checkpoint changed during reopen").into());
        }
        Ok(Self {
            checkpoint: source,
            plan,
            updater,
            observed,
            ledger,
            limits,
            channels,
        })
    }

    /// Frozen request, unchanged by paging.
    #[must_use]
    pub fn plan(&self) -> &NativeRefinementPlan {
        &self.plan
    }
    /// Committed whole-model cursor.
    #[must_use]
    pub const fn completed_updates(&self) -> u64 {
        self.observed.header.completed
    }
    /// Aggregate logical disk/state and worst-owner payload sizes.
    #[must_use]
    pub const fn size_ledger(&self) -> NativePvPagedSizeLedger {
        self.ledger
    }
    /// BLAKE3 of every checkpoint byte; integrity, not qualification.
    #[must_use]
    pub const fn checkpoint_digest(&self) -> [u8; 32] {
        self.observed.digest
    }
    /// Release the underlying checkpoint; its seek position is unspecified.
    pub fn into_checkpoint(self) -> R {
        self.checkpoint
    }

    pub(super) fn journal_state(&self) -> PagedJournalState {
        PagedJournalState {
            plan: self.observed.header.plan,
            parent: self.observed.header.parent,
            completed: self.completed_updates(),
            checkpoint: BlobId::from_bytes(self.observed.digest),
            checkpoint_bytes: self.observed.bytes,
        }
    }
    pub(super) fn journal_channels(&self) -> &[(u64, u64)] {
        &self.channels
    }
    pub(super) fn verify_journal_source(&mut self) -> Result<(), WireError> {
        if checkpoint::inspect(&mut self.checkpoint, self.observed.bytes)? != self.observed {
            return Err(invalid("native PV paging committed checkpoint drift").into());
        }
        Ok(())
    }

    /// Stage a complete next checkpoint and one owned resource per owner.
    ///
    /// Gradients are separate unweighted scale/present-weight arrays. The old
    /// session and cursor never advance: only a successful returned next session
    /// represents the complete update. Error/unwind drops staged resources and
    /// preserves old bytes. Caller callback side effects and partial destination
    /// bytes are not rolled back. Source/destination streams must remain private;
    /// generic I/O checks do not grant filesystem custody or power-loss safety.
    pub fn step_staged<W: Read + Write + Seek, E, T>(
        &mut self,
        mut destination: W,
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        mut gradients: impl FnMut(usize, &str, &AdditiveTensor) -> Result<(Vec<f32>, Vec<f32>), E>,
        mut stage: impl FnMut(usize, &str, &AdditiveTensor) -> Result<T, E>,
    ) -> Result<(NativePvPagedSession<W>, NativePvStepStats, Vec<T>), NativePvPagedError<E>> {
        let window = self
            .plan
            .training
            .get(self.completed_updates() as usize)
            .ok_or_else(|| PvTuningError::step("native PV schedule is exhausted"))?;
        empty_output(&mut destination)?;
        if checkpoint::inspect(&mut self.checkpoint, self.observed.bytes)? != self.observed {
            return Err(invalid("native PV paging committed checkpoint drift").into());
        }
        let next = self.completed_updates() + 1;
        let header = Header {
            completed: next,
            ..self.observed.header
        };
        destination.write_all(&header.bytes())?;
        self.checkpoint.rewind()?;
        let mut ledger = empty_ledger();
        let mut parent_hash = NativePvRefinementSession::parent_hasher(header.plan);
        let mut stats = NativePvStepStats {
            completed_updates: next,
            changed_scales: 0,
            owners: reserved(self.plan.owners.len())?,
        };
        let mut resources = reserved(self.plan.owners.len())?;
        let (traversed, suffix) = {
            let mut reader = HashRead {
                source: &mut self.checkpoint,
                hash: blake3::Hasher::new(),
            };
            if checkpoint::read_header(&mut reader)? != self.observed.header {
                return Err(invalid("native PV paging header drift").into());
            }
            let mut writer = HashWrite {
                output: &mut destination,
                hash: blake3::Hasher::new(),
            };
            for (i, name) in self.plan.owners.iter().enumerate() {
                let weight = parents(i, name).map_err(NativePvPagedError::Provider)?;
                account(
                    &self.plan,
                    self.limits,
                    self.updater.fraction,
                    &weight,
                    &mut ledger,
                )?;
                hash_parent_tensor(&mut parent_hash, &weight)?;
                let owner = checkpoint::read_owner::<8192>(
                    &mut reader,
                    weight,
                    self.observed.header.completed,
                    self.updater,
                )?;
                let (scales, weights) =
                    gradients(i, name, &owner.weight).map_err(NativePvPagedError::Provider)?;
                check_gradients(&owner, &scales, &weights)?;
                let (candidate, changed, observation) =
                    self.updater
                        .candidate(&owner, &scales, &weights, window.weight, next)?;
                checkpoint::write_owner::<8192>(&mut writer, &candidate)?;
                resources
                    .push(stage(i, name, &candidate.weight).map_err(NativePvPagedError::Provider)?);
                stats.changed_scales += changed;
                stats.owners.push(observation);
            }
            (
                *reader.hash.finalize().as_bytes(),
                *writer.hash.finalize().as_bytes(),
            )
        };
        if traversed != self.observed.body
            || *parent_hash.finalize().as_bytes() != header.parent
            || ledger != self.ledger
        {
            return Err(invalid("native PV paging original parent/body/ledger drift").into());
        }
        if checkpoint::inspect(&mut self.checkpoint, self.observed.bytes)? != self.observed {
            return Err(invalid("native PV paging checkpoint changed during update").into());
        }
        let observed = checkpoint::seal(
            &mut destination,
            header,
            ledger.logical.checkpoint_bytes,
            suffix,
        )?;
        Ok((
            NativePvPagedSession {
                checkpoint: destination,
                plan: self.plan.clone(),
                updater: self.updater,
                observed,
                ledger,
                limits: self.limits,
                channels: copied(&self.channels)?,
            },
            stats,
            resources,
        ))
    }
}
