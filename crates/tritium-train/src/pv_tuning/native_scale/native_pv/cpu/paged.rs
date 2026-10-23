//! Packed versions around the existing seek-backed recorded transaction.
use super::super::{
    NativePvJournalSummary, NativePvPagedError, NativePvPagedJournalWriter,
    NativePvPagedRecordingError, NativePvPagedRecordingLimits, NativePvPagedSession,
    NativePvPagedUpdate, NativeScaleStageError, PvTuningError,
};
use super::*;
use std::io::Read;
use tritium_schema::NativePvResumePoint;

/// Original paged/provider, packed/backend or journal refusal.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeCpuPvPagedError<E> {
    /// Original numerical/source/provider failure.
    Paged(NativePvPagedError<E>),
    /// Original packed root/version/snapshot/budget refusal.
    Packed(BackendError),
    /// Original standard frame/journal/cap/I/O refusal.
    Journal(NativePvJournalError),
}
impl<E: std::fmt::Display> std::fmt::Display for NativeCpuPvPagedError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Paged(e) => e.fmt(f),
            Self::Packed(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativeCpuPvPagedError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Paged(e) => Some(e),
            Self::Packed(e) => Some(e),
            Self::Journal(e) => Some(e),
        }
    }
}
enum Callback<E> {
    User(E),
    Packed(BackendError),
    Numerical(PvTuningError),
}
fn callback<E>(e: Callback<E>) -> NativeCpuPvPagedError<E> {
    match e {
        Callback::User(e) => NativeCpuPvPagedError::Paged(NativePvPagedError::Provider(e)),
        Callback::Packed(e) => NativeCpuPvPagedError::Packed(e),
        Callback::Numerical(e) => NativeCpuPvPagedError::Paged(NativePvPagedError::State(e)),
    }
}
fn paged<E>(e: NativePvPagedError<Callback<E>>) -> NativeCpuPvPagedError<E> {
    match e {
        NativePvPagedError::State(e) => NativeCpuPvPagedError::Paged(NativePvPagedError::State(e)),
        NativePvPagedError::Io(e) => NativeCpuPvPagedError::Paged(NativePvPagedError::Io(e)),
        NativePvPagedError::Provider(e) => callback(e),
    }
}
fn recording<E>(e: NativePvPagedRecordingError<Callback<E>>) -> NativeCpuPvPagedError<E> {
    match e {
        NativePvPagedRecordingError::Paged(e) => paged(e),
        NativePvPagedRecordingError::Journal(e) => NativeCpuPvPagedError::Journal(e),
    }
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}

/// Recorded disk/scratch plus one-owner reduction, separately from numerical,
/// source, packed/input/graph/metadata/allocator/OS-cache and RSS budgets.
#[derive(Clone, Copy, Debug)]
pub struct NativeCpuPvPagedStepLimits {
    /// Existing complete-journal/frame and recording-only scratch ceilings.
    pub recording: NativePvPagedRecordingLimits,
    /// Additional owner reducer peak12(S+A)+4cols, including split f32 outputs.
    pub max_consumer_working_bytes: u64,
}
/// Seek-backed optimizer state plus budget-admitted immutable packed models.
/// No decoded owner survives calls. Packed roots/old graphs remain resident under
/// their budget; this is not GPU/within-owner or physical-memory evidence.
pub struct NativeCpuPvPagedSession<R> {
    session: NativePvPagedSession<R>,
    handles: Vec<Arc<dyn DeviceBuffer>>,
    budget: TensorPayloadBudget,
    cohort: Arc<()>,
}
impl<R> std::fmt::Debug for NativeCpuPvPagedSession<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuPvPagedSession")
            .field("session", &self.session)
            .field("owners", &self.handles.len())
            .finish_non_exhaustive()
    }
}
impl<R: Read + Seek> NativeCpuPvPagedSession<R> {
    /// Bind exact CURRENT CPU codes/scales/geometry/names, then stage admitted
    /// snapshots through complete original-owner traversal. Source scopes must
    /// finish before exposing this result to a live graph. Resumed state needs
    /// current roots; mismatched original roots are not silently refreshed.
    pub fn bind<E>(
        mut session: NativePvPagedSession<R>,
        owners: Vec<(String, Arc<dyn DeviceBuffer>)>,
        budget: TensorPayloadBudget,
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
    ) -> Result<Self, NativeCpuPvPagedError<E>> {
        if owners.len() != session.plan().owners.len()
            || owners.iter().map(|x| &x.0).ne(session.plan().owners.iter())
        {
            return Err(NativeCpuPvPagedError::Packed(invalid(
                "CPU paged PV owner names/order mismatch",
            )));
        }
        let handles = session
            .visit_weights(
                |i, n| parents(i, n).map_err(Callback::User),
                |i, _, weight| {
                    let cpu = CpuBackend::new();
                    if !cpu
                        .additive_equals_view(&*owners[i].1, weight.view())
                        .map_err(Callback::Packed)?
                    {
                        return Err(Callback::Packed(invalid(
                            "CPU paged PV packed state differs from current checkpoint",
                        )));
                    }
                    cpu.snapshot_additive_weights(&owners[i].1, weight.view(), &budget)
                        .map_err(Callback::Packed)
                },
            )
            .map_err(paged)?;
        Ok(Self {
            session,
            handles,
            budget,
            cohort: Arc::new(()),
        })
    }
    /// Read-only numerical metadata, no independently mutable state access.
    pub fn session(&self) -> &NativePvPagedSession<R> {
        &self.session
    }
    /// Actual named current versions; old graph Arcs keep old admission alive.
    pub fn handles(&self) -> impl ExactSizeIterator<Item = (&str, &Arc<dyn DeviceBuffer>)> {
        self.session
            .plan()
            .owners
            .iter()
            .zip(&self.handles)
            .map(|(n, h)| (n.as_str(), h))
    }
    /// End runtime ownership. External graph Arcs retain their immutable versions.
    pub fn into_session(self) -> NativePvPagedSession<R> {
        self.session
    }
    /// Start ordinary zero-origin TNPJ without mutable numerical state exposure.
    pub fn new_recording<J: Write + Seek>(
        &mut self,
        output: J,
        limits: NativePvPagedRecordingLimits,
    ) -> Result<NativePvPagedJournalWriter<J>, NativePvJournalError> {
        NativePvPagedJournalWriter::new(output, &mut self.session, limits)
    }
    /// Exact standard resume point, not replay/durability proof.
    pub fn checkpoint<J: Write + Seek>(
        &mut self,
        journal: &mut NativePvPagedJournalWriter<J>,
    ) -> Result<NativePvResumePoint, NativePvJournalError> {
        journal.checkpoint(&mut self.session)
    }
    /// Finish the complete standard journal against fresh current bytes.
    pub fn finish_recording<J: Write + Seek>(
        &mut self,
        journal: NativePvPagedJournalWriter<J>,
    ) -> Result<(J, NativePvJournalSummary), NativePvJournalError> {
        journal.finish(&mut self.session)
    }
    /// Stage unweighted adjoints one owner at a time and all packed snapshots
    /// before live effects. Return through actual-parent scope, then commit.
    /// Caller/provider buffers and private partial outputs have separate budgets.
    pub fn stage_recorded<W: Read + Write + Seek, F: Read + Write + Seek, E>(
        &mut self,
        outputs: (W, F),
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        mut gradients: impl FnMut(
            usize,
            &str,
            &Arc<dyn DeviceBuffer>,
            &AdditiveTensor,
        ) -> Result<(Vec<f32>, Vec<f32>), E>,
        cap: NativePvPagedRecordingLimits,
    ) -> Result<NativeCpuPvPagedUpdate<W, F>, NativeCpuPvPagedError<E>> {
        self.stage(
            outputs,
            |i, n| parents(i, n).map_err(Callback::User),
            |i, n, h, w| gradients(i, n, h, w).map_err(Callback::User),
            cap,
        )
    }
    /// Ordered packed Matmul/Embed uses, shared f64 accumulation and one narrowing
    /// per owner. Common optimizer applies window weight once; empty uses still
    /// run decay/moments. Late owner refusal may leave private spools, never
    /// advance live history or expose a partial packed set.
    pub fn stage_from_consumers_recorded<W: Read + Write + Seek, F: Read + Write + Seek, E>(
        &mut self,
        outputs: (W, F),
        mut parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        contributions: &[&[NativePvContribution<'_>]],
        cap: NativeCpuPvPagedStepLimits,
    ) -> Result<NativeCpuPvPagedUpdate<W, F>, NativeCpuPvPagedError<E>> {
        if contributions.len() != self.handles.len() || cap.max_consumer_working_bytes == 0 {
            return Err(NativeCpuPvPagedError::Paged(NativePvPagedError::State(
                PvTuningError::step("native paged consumer inventory/working cap differs"),
            )));
        }
        self.stage(
            outputs,
            |i, n| parents(i, n).map_err(Callback::User),
            |i, _, h, w| {
                super::super::super::consumers::reduce_pv_owner(
                    w.view(),
                    contributions[i],
                    cap.max_consumer_working_bytes,
                    |_, c, row, s, a| super::consume(&**h, c, row, s, a),
                )
                .map_err(|e| match e {
                    NativeScaleStageError::Update(e) => Callback::Numerical(e),
                    NativeScaleStageError::Stage(e) => Callback::Packed(e),
                })
            },
            cap.recording,
        )
    }
    fn stage<W: Read + Write + Seek, F: Read + Write + Seek, E>(
        &mut self,
        outputs: (W, F),
        parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, Callback<E>>,
        mut gradients: impl FnMut(
            usize,
            &str,
            &Arc<dyn DeviceBuffer>,
            &AdditiveTensor,
        ) -> Result<(Vec<f32>, Vec<f32>), Callback<E>>,
        cap: NativePvPagedRecordingLimits,
    ) -> Result<NativeCpuPvPagedUpdate<W, F>, NativeCpuPvPagedError<E>> {
        let next_cohort = Arc::new(());
        let handles = &self.handles;
        let budget = &self.budget;
        let inner = self
            .session
            .stage_recorded(
                outputs,
                parents,
                |i, n, w| gradients(i, n, &handles[i], w),
                |i, _, w| {
                    CpuBackend::new()
                        .snapshot_additive_weights(&handles[i], w.view(), budget)
                        .map_err(Callback::Packed)
                },
                cap,
            )
            .map_err(recording)?;
        Ok(NativeCpuPvPagedUpdate {
            inner,
            before: self.cohort.clone(),
            next_cohort,
            budget: self.budget.clone(),
        })
    }
}
/// Opaque complete next numerical state/frame/packed set. Drop on source-scope
/// refusal; only single-use commit exposes it. No mutable current graph here.
pub struct NativeCpuPvPagedUpdate<W, F> {
    inner: NativePvPagedUpdate<W, F, Arc<dyn DeviceBuffer>>,
    before: Arc<()>,
    next_cohort: Arc<()>,
    budget: TensorPayloadBudget,
}
impl<W, F> std::fmt::Debug for NativeCpuPvPagedUpdate<W, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuPvPagedUpdate")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}
impl<W: Read + Seek, F: Read + Seek> NativeCpuPvPagedUpdate<W, F> {
    /// Consume AFTER original-parent scope succeeds. Another cohort/stale state
    /// refuses before append. Existing append poisoning applies on failure or
    /// unwind, exposing no next state/handles. Switch to returned complete driver
    /// on success; old driver remains an unchanged recovery view.
    pub fn commit<R: Read + Seek, J: Write + Seek>(
        self,
        current: &mut NativeCpuPvPagedSession<R>,
        journal: &mut NativePvPagedJournalWriter<J>,
    ) -> Result<(NativeCpuPvPagedSession<W>, NativePvStepStats), NativeCpuPvJournalStageError> {
        if !Arc::ptr_eq(&self.before, &current.cohort) {
            return Err(NativeCpuPvJournalStageError::Packed(invalid(
                "CPU paged update belongs to another packed cohort",
            )));
        }
        let (session, stats, handles) = self
            .inner
            .commit(&mut current.session, journal)
            .map_err(NativeCpuPvJournalStageError::Journal)?;
        Ok((
            NativeCpuPvPagedSession {
                session,
                handles,
                budget: self.budget,
                cohort: self.next_cohort,
            },
            stats,
        ))
    }
}
