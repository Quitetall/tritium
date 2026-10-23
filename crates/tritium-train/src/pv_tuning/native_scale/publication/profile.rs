//! Two concrete numerical adapters behind the common publication implementation.
use super::*;
use crate::pv_tuning::{NativePvRefinementSession, VerifiedNativePvReplay};
use tritium_evidence::NativePvChildReceiptView;
use tritium_format::AdditiveTensor;

#[derive(Clone, Copy)]
pub(crate) enum Profile {
    Scale,
    Pv,
}
#[derive(Clone, Copy)]
pub(crate) enum State<'a> {
    Scale(
        &'a NativeScaleRefinementSession,
        &'a VerifiedNativeScaleReplay,
    ),
    Pv(&'a NativePvRefinementSession, &'a VerifiedNativePvReplay),
}
pub(crate) enum Receipt {
    Scale(NativeScaleChildReceiptView),
    Pv(NativePvChildReceiptView),
}
impl Receipt {
    pub(crate) fn manifest(&self) -> &ModelManifest {
        match self {
            Self::Scale(v) => v.manifest(),
            Self::Pv(v) => v.manifest(),
        }
    }
}
pub(super) struct Binding {
    pub(super) journal: BlobId,
    pub(super) bytes: u64,
    pub(super) checkpoint: BlobId,
    pub(super) checkpoint_bytes: u64,
    pub(super) plan: NativeRefinementPlanId,
    pub(super) updates: u64,
}
impl State<'_> {
    pub(super) fn binding(self) -> Binding {
        macro_rules! binding {
            ($v:expr) => {{
                let s = $v.summary();
                Binding {
                    journal: s.journal(),
                    bytes: s.bytes(),
                    checkpoint: s.checkpoint(),
                    checkpoint_bytes: s.checkpoint_bytes(),
                    plan: s.plan(),
                    updates: s.updates(),
                }
            }};
        }
        match self {
            Self::Scale(_, r) => binding!(r),
            Self::Pv(_, r) => binding!(r),
        }
    }
    pub(crate) fn kind(self) -> Profile {
        match self {
            Self::Scale(..) => Profile::Scale,
            Self::Pv(..) => Profile::Pv,
        }
    }
    pub(crate) fn plan(&self) -> &NativeRefinementPlan {
        match self {
            Self::Scale(s, _) => s.plan(),
            Self::Pv(s, _) => s.plan(),
        }
    }
    pub(crate) fn id(self) -> NativeRefinementPlanId {
        match self {
            Self::Scale(s, _) => s.plan_id(),
            Self::Pv(s, _) => s.plan_id(),
        }
    }
    pub(crate) fn completed(self) -> u64 {
        match self {
            Self::Scale(s, _) => s.completed_updates(),
            Self::Pv(s, _) => s.completed_updates(),
        }
    }
    pub(crate) fn footprint(self) -> (u64, u64) {
        match self {
            Self::Scale(s, _) => {
                let l = s.size_ledger();
                (l.state_payload_bytes, l.checkpoint_bytes)
            }
            Self::Pv(s, _) => {
                let l = s.size_ledger();
                (l.state_payload_bytes, l.checkpoint_bytes)
            }
        }
    }
    pub(crate) fn parent(self) -> [u8; 32] {
        match self {
            Self::Scale(s, _) => s.parent,
            Self::Pv(s, _) => s.parent_fingerprint(),
        }
    }
    pub(crate) fn parent_hasher(self) -> blake3::Hasher {
        match self {
            Self::Scale(s, _) => parent_hasher(s.plan_id()),
            Self::Pv(s, _) => NativePvRefinementSession::parent_hasher(s.plan_id()),
        }
    }
    pub(crate) fn weights(&self) -> impl ExactSizeIterator<Item = &AdditiveTensor> {
        (0..self.plan().owners.len()).map(|i| match self {
            Self::Scale(s, _) => &s.owners[i].weight,
            Self::Pv(s, _) => s.weight_at(i),
        })
    }
    pub(super) fn checkpoint(self) -> std::result::Result<Vec<u8>, super::super::PvTuningError> {
        match self {
            Self::Scale(s, _) => s.checkpoint_bytes(),
            Self::Pv(s, _) => s.checkpoint_bytes(),
        }
    }
    pub(super) fn verify(self) -> Result<()> {
        match self {
            Self::Scale(s, r) => r.verify_session(s).map_err(Into::into),
            Self::Pv(s, r) => r.verify_session(s).map_err(NativeScaleChildError::PvReplay),
        }
    }
    pub(super) fn backend(self) -> &'static str {
        match self {
            Self::Scale(..) => "native-scale-cpu-reference-v1",
            Self::Pv(..) => "native-pv-cpu-reference-v1",
        }
    }
    pub(super) fn companions(self) -> (&'static str, &'static str) {
        match self {
            Self::Scale(..) => (NATIVE_SCALE_CHECKPOINT_FILE, NATIVE_SCALE_JOURNAL_FILE),
            Self::Pv(..) => (NATIVE_PV_CHECKPOINT_FILE, NATIVE_PV_JOURNAL_FILE),
        }
    }
    pub(super) fn check_emit(
        self,
        recorder: &EvidenceRecorder,
    ) -> std::result::Result<(), EvidenceError> {
        match self {
            Self::Scale(..) => recorder.check_emit::<NativeScaleReplayObserved>(
                "numerical-replay",
                Some("refinement"),
                1,
            ),
            Self::Pv(..) => recorder.check_emit::<NativePvReplayObserved>(
                "numerical-replay",
                Some("refinement"),
                1,
            ),
        }
    }
    pub(super) fn emit(
        self,
        recorder: &mut EvidenceRecorder,
    ) -> std::result::Result<(), EvidenceError> {
        let b = self.binding();
        macro_rules! event {
            ($ty:ident) => {
                $ty {
                    plan: b.plan,
                    journal: b.journal,
                    journal_bytes: b.bytes,
                    checkpoint: b.checkpoint,
                    checkpoint_bytes: b.checkpoint_bytes,
                    completed_updates: b.updates,
                }
            };
        }
        match self {
            Self::Scale(..) => recorder.emit(
                "numerical-replay",
                Some("refinement"),
                1,
                &event!(NativeScaleReplayObserved),
            ),
            Self::Pv(..) => recorder.emit(
                "numerical-replay",
                Some("refinement"),
                1,
                &event!(NativePvReplayObserved),
            ),
        }
        .map(|_| ())
    }
    pub(super) fn fold<R: Read + Seek>(
        self,
        log: &tritium_evidence::VerifiedLog,
        parent: &mut TritModelReader<R>,
        view: &NativeRefinementParentView,
    ) -> std::result::Result<Vec<Receipt>, EvidenceError> {
        match self {
            Self::Scale(..) => NativeScaleChildReceiptView::fold(log, parent, view, self.plan())
                .map(|v| v.into_iter().map(Receipt::Scale).collect()),
            Self::Pv(..) => NativePvChildReceiptView::fold(log, parent, view, self.plan())
                .map(|v| v.into_iter().map(Receipt::Pv).collect()),
        }
    }
}
