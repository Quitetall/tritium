//! Replayed-child record consistency, not proof of objective/gradient origin.
use crate::native_refinement::{NativeRecoveryRecord, fold_native_recovery};
use crate::{
    EventRegistration, EvidenceError, EvidenceEvent, NativeRefinementParentView,
    NativeRefinementReceiptView, VerifiedLog, verify_log,
};
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
use tritium_schema::{
    EvidenceReference, LevelFit, NativeRefinementCheckpointed, NativeRefinementCompleted,
    NativeRefinementStarted, SchemaId, TensorStorage,
};
use tritium_schema::{ModelManifest, NativeRefinementPlan, NativeScaleReplayObserved};

impl EvidenceEvent for NativeScaleReplayObserved {
    const SCHEMA_ID: &'static str = "tritium.refine.scale_replay_observed";
    const VERSION: u16 = 1;
}
/// Numerical replay observation; the producer must actually execute replay.
pub const NATIVE_SCALE_REPLAY_OBSERVED_REGISTRATION: EventRegistration = EventRegistration {
    schema: NativeScaleReplayObserved::SCHEMA_ID,
    version: 1,
};

/// Complete native child plus one recorded journal binding; not replay/acceptance authority.
#[derive(Clone, Debug)]
pub struct NativeScaleChildReceiptView {
    refinement: NativeRefinementReceiptView,
    replay: NativeScaleReplayObserved,
}
impl NativeScaleChildReceiptView {
    /// Exact original native refinement observations.
    #[must_use]
    pub fn refinement(&self) -> &NativeRefinementReceiptView {
        &self.refinement
    }
    /// Recorded journal/checkpoint binding; actual bytes and numerical replay are separate.
    #[must_use]
    pub fn replay(&self) -> &NativeScaleReplayObserved {
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
        verify_log(log)?;
        let views = NativeRefinementReceiptView::fold(log, parent_reader, parent, requested)?;
        if views.is_empty() {
            return Ok(Vec::new());
        }
        if views.len() != 1 {
            return Err(invalid("expected exactly one native refinement"));
        }
        let refinement = views.into_iter().next().expect("one view");
        let replay = replay_for(log, refinement.reference(), refinement.checkpoint())?;
        Ok(vec![Self { refinement, replay }])
    }
    /// Bind actual complete typed child bytes to this original admitted record.
    pub fn verify_child<R: Read + Seek>(
        &self,
        child: &mut TritModelReader<R>,
    ) -> Result<(), EvidenceError> {
        if child.manifest() != self.manifest() {
            return Err(invalid(
                "native child manifest differs from original record",
            ));
        }
        child
            .validate_complete_payload_inventory()
            .map_err(|_| invalid("native child payload inventory differs"))?;
        child
            .verify()
            .map_err(|_| invalid("native child source changed"))?;
        Ok(())
    }
}

/// Standalone child/record consistency only. No original-parent or numerical admission.
#[derive(Clone, Debug)]
pub struct NativeScaleChildRecordView {
    recovery: NativeRecoveryRecord,
    replay: NativeScaleReplayObserved,
}
impl NativeScaleChildRecordView {
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
    pub fn replay(&self) -> &NativeScaleReplayObserved {
        &self.replay
    }
    /// Admit actual standalone typed child and strict original records, without
    /// inventing a parent view or implying that original parent state was verified.
    pub fn from_child<R: Read + Seek>(
        log: &VerifiedLog,
        child: &mut TritModelReader<R>,
    ) -> Result<Self, EvidenceError> {
        verify_log(log)?;
        let first = log
            .events
            .iter()
            .find(|e| e.schema == "tritium.refine.native_started")
            .ok_or_else(|| invalid("native child start absent"))?;
        if first.v != 1 {
            return Err(invalid("native child start version"));
        }
        let started: NativeRefinementStarted = serde_json::from_value(first.payload.clone())
            .map_err(|_| invalid("native child start payload"))?;
        if !matches!(
            started.plan.recipe,
            tritium_schema::NativeRecoveryRecipe::ScaleOnly { .. }
        ) {
            return Err(invalid("standalone child profile requires ScaleOnly"));
        }
        let recovery = fold_native_recovery(log, &started.plan)?
            .ok_or_else(|| invalid("native child lifecycle incomplete"))?;
        let replay = replay_for(log, &recovery.reference, &recovery.checkpoint)?;
        let manifest = child.manifest();
        if manifest.schema != (SchemaId { major: 1, minor: 1 }) || manifest.levels.len() != 1 {
            return Err(invalid(
                "standalone native child requires schema1.1 one level",
            ));
        }
        let level = &manifest.levels[0];
        let LevelFit::Refined { ancestry } = &level.fit else {
            return Err(invalid("standalone native child is not Refined"));
        };
        if level.name != recovery.started.plan.child_level
            || ancestry.parent != recovery.started.plan.parent
            || ancestry.plan != recovery.plan
            || ancestry.evidence != recovery.reference
        {
            return Err(invalid("standalone native child ancestry differs"));
        }
        if level
            .tensors
            .iter()
            .map(|t| (&t.name, t.blob.id))
            .ne(recovery
                .completed
                .tensors
                .iter()
                .map(|t| (&t.name, t.content)))
            || recovery.completed.validation.len() != recovery.started.plan.validation.len()
            || recovery
                .completed
                .validation
                .iter()
                .zip(&recovery.started.plan.validation)
                .any(|(a, b)| {
                    a.content != b.content
                        || !a.parent_loss.is_finite()
                        || !a.child_loss.is_finite()
                })
        {
            return Err(invalid("standalone native child output/validation differs"));
        }
        for owner in &recovery.checkpoint.owners {
            let meta = child
                .tensor_meta(&level.name, &owner.name)
                .map_err(|_| invalid("standalone child selected owner absent"))?;
            if !matches!(meta.storage, TensorStorage::Additive { .. })
                || meta.blob.id != owner.content
            {
                return Err(invalid("standalone child checkpoint owner differs"));
            }
        }
        child
            .validate_complete_payload_inventory()
            .map_err(|_| invalid("standalone child inventory differs"))?;
        child
            .verify()
            .map_err(|_| invalid("standalone child source changed"))?;
        Ok(Self { recovery, replay })
    }
}

fn replay_for(
    log: &VerifiedLog,
    reference: &EvidenceReference,
    checkpoint: &NativeRefinementCheckpointed,
) -> Result<NativeScaleReplayObserved, EvidenceError> {
    let records: Vec<_> = log
        .events
        .iter()
        .filter(|e| e.schema == NativeScaleReplayObserved::SCHEMA_ID)
        .collect();
    if records.len() != 1
        || log.events.iter().any(|e| {
            !e.schema.starts_with("tritium.refine.native_")
                && e.schema != NativeScaleReplayObserved::SCHEMA_ID
        })
    {
        return Err(invalid("expected exactly one native replay observation"));
    }
    let record = records[0];
    let start = log
        .events
        .iter()
        .find(|e| e.span == reference.span && e.seq == 0)
        .ok_or_else(|| invalid("missing native start"))?;
    let complete = log
        .events
        .iter()
        .find(|e| e.schema == "tritium.refine.native_completed" && e.span == reference.span)
        .ok_or_else(|| invalid("missing native completion"))?;
    if record.v != 1
        || record.run != reference.run
        || record.parent.as_deref() != Some(reference.span.as_str())
        || record.span == reference.span
        || record.seq != 0
        || record.t <= start.t
        || record.t >= complete.t
    {
        return Err(invalid("native replay span/version/order differs"));
    }
    let replay: NativeScaleReplayObserved = serde_json::from_value(record.payload.clone())
        .map_err(|_| invalid("invalid native replay observation"))?;
    if replay.plan != checkpoint.plan
        || replay.checkpoint != checkpoint.checkpoint
        || replay.checkpoint_bytes != checkpoint.checkpoint_bytes
        || replay.completed_updates != checkpoint.completed_updates
        || replay.journal_bytes == 0
        || replay.journal_bytes > ((1u64 << 53) - 1)
    {
        return Err(invalid("native replay checkpoint/cursor binding differs"));
    }
    Ok(replay)
}
fn invalid(message: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: NativeScaleReplayObserved::SCHEMA_ID.into(),
        detail: message.into(),
    }
}
