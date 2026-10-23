//! Shared strict read-only traversal, not a second checkpoint parser.
use super::*;
type Visited<T> = (NativePvPagedSizeLedger, Vec<(u64, u64)>, Vec<T>);

pub(super) struct Traversal<'a> {
    pub(super) plan: &'a NativeRefinementPlan,
    pub(super) limits: NativePvPagedLimits,
    pub(super) updater: PvUpdater,
    pub(super) observed: Inspection,
}
impl Traversal<'_> {
    pub(super) fn run<R: Read + Seek, E, T>(
        self,
        source: &mut R,
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        mut visit: impl FnMut(usize, &str, &AdditiveTensor) -> Result<T, E>,
    ) -> Result<Visited<T>, NativePvPagedError<E>> {
        source.rewind()?;
        let mut ledger = empty_ledger();
        let mut channels = reserved(self.plan.owners.len())?;
        let mut outputs = reserved(self.plan.owners.len())?;
        let mut parent_hash = NativePvRefinementSession::parent_hasher(self.observed.header.plan);
        let traversed = {
            let mut reader = HashRead {
                source: &mut *source,
                hash: blake3::Hasher::new(),
            };
            if checkpoint::read_header(&mut reader)? != self.observed.header {
                return Err(invalid("native PV paging header drift").into());
            }
            for (i, name) in self.plan.owners.iter().enumerate() {
                let weight = parents(i, name).map_err(NativePvPagedError::Provider)?;
                account(
                    self.plan,
                    self.limits,
                    self.updater.fraction,
                    &weight,
                    &mut ledger,
                )?;
                channels.push((weight.scales().len() as u64, active_count(&weight)? as u64));
                hash_parent_tensor(&mut parent_hash, &weight)?;
                let owner = checkpoint::read_owner::<8192>(
                    &mut reader,
                    weight,
                    self.observed.header.completed,
                    self.updater,
                )?;
                outputs.push(visit(i, name, &owner.weight).map_err(NativePvPagedError::Provider)?);
            }
            *reader.hash.finalize().as_bytes()
        };
        if traversed != self.observed.body
            || *parent_hash.finalize().as_bytes() != self.observed.header.parent
            || ledger.logical.checkpoint_bytes != self.observed.bytes
        {
            return Err(invalid("native PV paging parent/body/length mismatch").into());
        }
        if checkpoint::inspect(source, self.observed.bytes)? != self.observed {
            return Err(invalid("native PV paging checkpoint changed during traversal").into());
        }
        Ok((ledger, channels, outputs))
    }
}

impl<R: Read + Seek> NativePvPagedSession<R> {
    /// Visit validated current weights one owner at a time, using ordered actual
    /// original parents and the common TNPV parser. No numerical cursor changes.
    /// Outputs return only after full parent/body/ledger/source checks; late
    /// failure/unwind drops them. Callback effects must remain private; returned
    /// resources, provider internals, metadata and external buffers have separate
    /// budgets. Not numerical replay, publication authority or physical memory.
    pub fn visit_weights<E, T>(
        &mut self,
        parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        visit: impl FnMut(usize, &str, &AdditiveTensor) -> Result<T, E>,
    ) -> Result<Vec<T>, NativePvPagedError<E>> {
        self.verify_journal_source()?;
        let (ledger, channels, outputs) = Traversal {
            plan: &self.plan,
            limits: self.limits,
            updater: self.updater,
            observed: self.observed,
        }
        .run(&mut self.checkpoint, parents, visit)?;
        if ledger != self.ledger || channels != self.channels {
            return Err(invalid("native PV paging traversal ledger/channels differ").into());
        }
        Ok(outputs)
    }
}
