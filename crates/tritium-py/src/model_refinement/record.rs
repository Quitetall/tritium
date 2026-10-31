//! Typed record dispatch, not numerical or quality qualification.
use super::*;
use tritium_evidence::{NativePvChildRecordView, NativeRefinementParentView};
use tritium_schema::{BlobId, EvidenceReference, NativeRecoveryRecipe, NativeRefinementStarted};
use tritium_train::pv_tuning::{
    NATIVE_PV_CHECKPOINT_FILE, NATIVE_PV_JOURNAL_FILE, NativePvChildVerification,
    NativePvJournalLimits, NativePvParentReadLimits, verify_native_pv_child,
};

pub(super) enum Record {
    Scale(NativeScaleChildRecordView),
    Pv(NativePvChildRecordView),
}
#[derive(Clone, Copy)]
pub(super) struct Binding {
    pub checkpoint: BlobId,
    pub checkpoint_bytes: u64,
    pub journal: BlobId,
    pub journal_bytes: u64,
}
pub(super) struct Sources<'a> {
    pub parent: &'a mut TritModelReader<File>,
    pub parent_view: &'a NativeRefinementParentView,
    pub child: &'a mut TritModelReader<File>,
    pub log: &'a VerifiedLog,
    pub checkpoint: &'a [u8],
    pub journal: &'a mut File,
}
impl Record {
    pub(super) fn open(log: &VerifiedLog, model: &mut TritModelReader<File>) -> PyResult<Self> {
        Self::for_reader(log, model)
    }
    pub(super) fn for_reader<R: std::io::Read + std::io::Seek>(
        log: &VerifiedLog,
        model: &mut TritModelReader<R>,
    ) -> PyResult<Self> {
        let first = log
            .events
            .iter()
            .find(|event| event.schema == "tritium.refine.native_started")
            .ok_or_else(|| error("native refinement start absent"))?;
        let started: NativeRefinementStarted =
            serde_json::from_value(first.payload.clone()).map_err(error)?;
        match started.plan.recipe {
            NativeRecoveryRecipe::ScaleOnly { .. } => {
                NativeScaleChildRecordView::from_child(log, model)
                    .map(Self::Scale)
                    .map_err(error)
            }
            NativeRecoveryRecipe::AlternatingPv { .. } => {
                NativePvChildRecordView::from_child(log, model)
                    .map(Self::Pv)
                    .map_err(error)
            }
            _ => Err(error("unsupported native refinement record profile")),
        }
    }
    pub(super) fn started(&self) -> &NativeRefinementStarted {
        match self {
            Self::Scale(v) => v.started(),
            Self::Pv(v) => v.started(),
        }
    }
    pub(super) fn reference(&self) -> &EvidenceReference {
        match self {
            Self::Scale(v) => v.reference(),
            Self::Pv(v) => v.reference(),
        }
    }
    pub(super) fn profile(&self) -> &'static str {
        match self {
            Self::Scale(_) => "scale-only",
            Self::Pv(_) => "alternating-pv",
        }
    }
    pub(super) fn names(&self) -> (&'static str, &'static str) {
        match self {
            Self::Scale(_) => (NATIVE_SCALE_CHECKPOINT_FILE, NATIVE_SCALE_JOURNAL_FILE),
            Self::Pv(_) => (NATIVE_PV_CHECKPOINT_FILE, NATIVE_PV_JOURNAL_FILE),
        }
    }
    pub(super) fn replay(&self) -> Binding {
        macro_rules! binding {
            ($v:expr) => {{
                let r = $v.replay();
                Binding {
                    checkpoint: r.checkpoint,
                    checkpoint_bytes: r.checkpoint_bytes,
                    journal: r.journal,
                    journal_bytes: r.journal_bytes,
                }
            }};
        }
        match self {
            Self::Scale(v) => binding!(v),
            Self::Pv(v) => binding!(v),
        }
    }
    pub(super) fn parent(
        &self,
        model: &mut TritModelReader<File>,
        level: &str,
    ) -> PyResult<NativeRefinementParentView> {
        match self {
            Self::Scale(v) => NativeRefinementParentView::from_refined_record(model, level, v),
            Self::Pv(v) => NativeRefinementParentView::from_refined_pv_record(model, level, v),
        }
        .map_err(error)
    }
    pub(super) fn verify_with_parent(
        &self,
        s: Sources<'_>,
        max_parent_bytes: u64,
        max_replay_bytes: u64,
        max_journal_bytes: u64,
    ) -> PyResult<(tritium_format::PackageId, BlobId)> {
        match self {
            Self::Scale(_) => {
                let verified = verify_native_scale_child(
                    NativeScaleChildVerification {
                        parent: s.parent,
                        parent_evidence: s.parent_view,
                        child: s.child,
                        log: s.log,
                        plan: &self.started().plan,
                        journal: s.journal,
                        checkpoint: s.checkpoint,
                    },
                    NativeScaleParentReadLimits {
                        max_working_bytes: max_parent_bytes,
                        extra_payload_bytes: 0,
                    },
                    NativeScaleJournalLimits {
                        max_journal_bytes,
                        max_working_bytes: max_replay_bytes,
                    },
                )
                .map_err(error)?;
                Ok((verified.package(), verified.replay().summary().journal()))
            }
            Self::Pv(_) => {
                let verified = verify_native_pv_child(
                    NativePvChildVerification {
                        parent: s.parent,
                        parent_evidence: s.parent_view,
                        child: s.child,
                        log: s.log,
                        plan: &self.started().plan,
                        journal: s.journal,
                        checkpoint: s.checkpoint,
                    },
                    NativePvParentReadLimits {
                        max_working_bytes: max_parent_bytes,
                        extra_payload_bytes: 0,
                    },
                    NativePvJournalLimits {
                        max_journal_bytes,
                        max_working_bytes: max_replay_bytes,
                    },
                )
                .map_err(error)?;
                Ok((verified.package(), verified.replay().summary().journal()))
            }
        }
    }
}
