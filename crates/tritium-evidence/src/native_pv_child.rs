//! Replayed-child record consistency, not proof of objective/gradient origin.
use crate::native_refinement::NativeRecoveryRecord;
use crate::{
    EventRegistration, EvidenceError, EvidenceEvent, NativeRefinementParentView,
    NativeRefinementReceiptView, VerifiedLog,
};
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
use tritium_schema::{
    EvidenceReference, NativeRefinementCheckpointed, NativeRefinementCompleted,
    NativeRefinementStarted,
};
use tritium_schema::{ModelManifest, NativePvReplayObserved, NativeRefinementPlan};

impl EvidenceEvent for NativePvReplayObserved {
    const SCHEMA_ID: &'static str = "tritium.refine.pv_replay_observed";
    const VERSION: u16 = 1;
}
/// Numerical replay observation; the producer must actually execute replay.
pub const NATIVE_PV_REPLAY_OBSERVED_REGISTRATION: EventRegistration = EventRegistration {
    schema: NativePvReplayObserved::SCHEMA_ID,
    version: 1,
};

/// Complete native child plus one recorded journal binding; not replay/acceptance authority.
#[derive(Clone, Debug)]
pub struct NativePvChildReceiptView {
    refinement: NativeRefinementReceiptView,
    replay: NativePvReplayObserved,
}
impl NativePvChildReceiptView {
    /// Exact original native refinement observations.
    #[must_use]
    pub fn refinement(&self) -> &NativeRefinementReceiptView {
        &self.refinement
    }
    /// Recorded journal/checkpoint binding; actual bytes and numerical replay are separate.
    #[must_use]
    pub fn replay(&self) -> &NativePvReplayObserved {
        &self.replay
    }
    /// Complete one-level refined manifest derived against the admitted parent.
    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        self.refinement.manifest()
    }
    /// Fold the unchanged native lifecycle plus one matching replay child-span observation.
    /// Incomplete runs yield no complete view; arbitrary caller JSON is not numerical proof.
    pub fn fold<R: Read + Seek>(
        log: &VerifiedLog,
        parent_reader: &mut TritModelReader<R>,
        parent: &NativeRefinementParentView,
        requested: &NativeRefinementPlan,
    ) -> Result<Vec<Self>, EvidenceError> {
        crate::native_child_record::fold::<NativePvReplayObserved, _>(
            log,
            parent_reader,
            parent,
            requested,
        )
        .map(|views| {
            views
                .into_iter()
                .map(|(refinement, replay)| Self { refinement, replay })
                .collect()
        })
    }
    /// Bind actual complete typed child bytes to this original admitted record.
    pub fn verify_child<R: Read + Seek>(
        &self,
        child: &mut TritModelReader<R>,
    ) -> Result<(), EvidenceError> {
        crate::native_child_record::verify_manifest::<NativePvReplayObserved, _>(
            self.manifest(),
            child,
        )
    }
}

/// Standalone child/record consistency only. No original-parent or numerical admission.
#[derive(Clone, Debug)]
pub struct NativePvChildRecordView {
    recovery: NativeRecoveryRecord,
    replay: NativePvReplayObserved,
    package: tritium_format::PackageId,
}
impl NativePvChildRecordView {
    /// Recorded plan/execution, not authenticated objective execution.
    #[must_use]
    pub fn started(&self) -> &NativeRefinementStarted {
        &self.recovery.started
    }
    /// Recorded final selected state binding; exact bytes/replay remain separate.
    #[must_use]
    pub fn checkpoint(&self) -> &NativeRefinementCheckpointed {
        &self.recovery.checkpoint
    }
    /// Recorded complete output/validation observations, not acceptance.
    #[must_use]
    pub fn completed(&self) -> &NativeRefinementCompleted {
        &self.recovery.completed
    }
    /// Original exact run/span/root.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.recovery.reference
    }
    /// Recorded actual companion identities; not proof those files were read.
    #[must_use]
    pub fn replay(&self) -> &NativePvReplayObserved {
        &self.replay
    }
    /// Admit actual standalone typed child and strict original records, without
    /// inventing a parent view or implying that original parent state was verified.
    pub fn from_child<R: Read + Seek>(
        log: &VerifiedLog,
        child: &mut TritModelReader<R>,
    ) -> Result<Self, EvidenceError> {
        let (recovery, replay, package) =
            crate::native_child_record::from_child::<NativePvReplayObserved, _>(log, child)?;
        Ok(Self {
            recovery,
            replay,
            package,
        })
    }
    /// Reverify exactly the model container originally observed by this record.
    /// This does not read checkpoint/journal companions or establish numerical history.
    pub fn verify_child<R: Read + Seek>(
        &self,
        child: &mut TritModelReader<R>,
    ) -> Result<(), EvidenceError> {
        crate::native_child_record::verify_package::<NativePvReplayObserved, _>(self.package, child)
    }
}
