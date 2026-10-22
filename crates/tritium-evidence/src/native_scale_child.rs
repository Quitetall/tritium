//! Replayed-child record consistency, not proof of objective/gradient origin.
use crate::{
    EventRegistration, EvidenceError, EvidenceEvent, NativeRefinementParentView,
    NativeRefinementReceiptView, VerifiedLog, verify_log,
};
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
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
        let records: Vec<_> = log
            .events
            .iter()
            .filter(|event| event.schema == NativeScaleReplayObserved::SCHEMA_ID)
            .collect();
        if views.len() != 1
            || records.len() != 1
            || log.events.iter().any(|event| {
                !event.schema.starts_with("tritium.refine.native_")
                    && event.schema != NativeScaleReplayObserved::SCHEMA_ID
            })
        {
            return Err(invalid("expected exactly one native replay observation"));
        }
        let refinement = views.into_iter().next().expect("one view");
        let record = records[0];
        let start = log
            .events
            .iter()
            .find(|event| event.span == refinement.reference().span && event.seq == 0)
            .ok_or_else(|| invalid("missing native start"))?;
        let complete = log
            .events
            .iter()
            .find(|event| {
                event.schema == "tritium.refine.native_completed"
                    && event.span == refinement.reference().span
            })
            .ok_or_else(|| invalid("missing native completion"))?;
        if record.v != 1
            || record.run != refinement.reference().run
            || record.parent.as_deref() != Some(refinement.reference().span.as_str())
            || record.span == refinement.reference().span
            || record.seq != 0
            || record.t <= start.t
            || record.t >= complete.t
        {
            return Err(invalid("native replay span/version/order differs"));
        }
        let replay: NativeScaleReplayObserved = serde_json::from_value(record.payload.clone())
            .map_err(|_| invalid("invalid native replay observation"))?;
        let checkpoint = refinement.checkpoint();
        if replay.plan != checkpoint.plan
            || replay.checkpoint != checkpoint.checkpoint
            || replay.checkpoint_bytes != checkpoint.checkpoint_bytes
            || replay.completed_updates != checkpoint.completed_updates
            || replay.journal_bytes == 0
            || replay.journal_bytes > ((1u64 << 53) - 1)
        {
            return Err(invalid("native replay checkpoint/cursor binding differs"));
        }
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
fn invalid(message: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: NativeScaleReplayObserved::SCHEMA_ID.into(),
        detail: message.into(),
    }
}
