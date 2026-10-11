//! Strict original forward-salience lifecycle; structural consistency is not replay.
use crate::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, VerifiedLog, decode_hex,
    original_source_id, verify_log,
};
use tritium_schema::{
    EvidenceReference, OriginalSalienceCaptureCompleted, OriginalSalienceCaptureStarted,
};

impl EvidenceEvent for OriginalSalienceCaptureStarted {
    const SCHEMA_ID: &'static str = "tritium.salience.capture_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for OriginalSalienceCaptureCompleted {
    const SCHEMA_ID: &'static str = "tritium.salience.capture_completed";
    const VERSION: u16 = 1;
}
/// Distinct original A request registration, not B next-token capture.
pub const ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: OriginalSalienceCaptureStarted::SCHEMA_ID,
    version: OriginalSalienceCaptureStarted::VERSION,
};
/// Distinct forward-only moments completion, not loss/gradient or quality evidence.
pub const ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: OriginalSalienceCaptureCompleted::SCHEMA_ID,
    version: OriginalSalienceCaptureCompleted::VERSION,
};

/// One complete standalone original salience record, with rederived content identity.
/// Tokenizer/graph/moment truth requires actual original-source cold replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OriginalSalienceCaptureView {
    started: OriginalSalienceCaptureStarted,
    completed: OriginalSalienceCaptureCompleted,
    reference: EvidenceReference,
}
impl OriginalSalienceCaptureView {
    /// Complete recorded original input/interpretation/profile/caps request.
    #[must_use]
    pub fn started(&self) -> &OriginalSalienceCaptureStarted {
        &self.started
    }
    /// Ordered encoded windows and raw moment bindings, not numerical clearance.
    #[must_use]
    pub fn completed(&self) -> &OriginalSalienceCaptureCompleted {
        &self.completed
    }
    /// Actual sealed standalone run/span/root identity.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }

    /// Verify integrity, strict versions/fields, standalone ancestry, shared source
    /// identity and complete ordered structural agreement. Logs with no salience
    /// events and a valid lone start yield no completed view. An incomplete start
    /// is never promoted into a completed receipt. No actual graph is executed here.
    /// # Errors
    /// Malformed/foreign/mixed/duplicate/orphan events, source identity or caps/counts.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        let known = |e: &EvidenceEnvelope| {
            e.schema == OriginalSalienceCaptureStarted::SCHEMA_ID
                || e.schema == OriginalSalienceCaptureCompleted::SCHEMA_ID
        };
        if !log.events.iter().any(known) {
            return Ok(Vec::new());
        }
        if log.events.len() > 2 || !log.events.iter().all(known) {
            return Err(invalid("original salience requires one standalone root"));
        }
        let start = &log.events[0];
        if start.schema != OriginalSalienceCaptureStarted::SCHEMA_ID
            || start.parent.is_some()
            || start.seq != 0
        {
            return Err(invalid("original salience has no valid root start"));
        }
        let started: OriginalSalienceCaptureStarted = decode(start)?;
        let plan = &started.plan;
        plan.validate()
            .map_err(|_| invalid("original salience request rejected"))?;
        if original_source_id(&plan.architecture, &plan.tensors, &plan.assets)
            .map_err(|_| invalid("original salience inventory rejected"))?
            != plan.source
        {
            return Err(invalid("original salience source identity differs"));
        }
        let Some(end) = log.events.get(1) else {
            return Ok(Vec::new());
        };
        if end.schema != OriginalSalienceCaptureCompleted::SCHEMA_ID
            || end.parent.is_some()
            || end.seq != 1
            || end.span != start.span
            || end.run != start.run
            || end.t < start.t
        {
            return Err(invalid("original salience completion ancestry differs"));
        }
        let completed: OriginalSalienceCaptureCompleted = decode(end)?;
        if !completed.agrees_with(plan) {
            return Err(invalid("original salience ordered output context differs"));
        }
        Ok(vec![Self {
            started,
            completed,
            reference: EvidenceReference {
                run: start.run.clone(),
                span: start.span.clone(),
                root: decode_hex(&log.root_digest)?,
            },
        }])
    }
}
fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: OriginalSalienceCaptureCompleted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
fn decode<E: EvidenceEvent + serde::de::DeserializeOwned>(
    event: &EvidenceEnvelope,
) -> Result<E, EvidenceError> {
    if event.v != E::VERSION {
        return Err(EvidenceError::UnregisteredSchema(format!(
            "{}@{}",
            event.schema, event.v
        )));
    }
    serde_json::from_value(event.payload.clone())
        .map_err(|_| invalid("strict original salience payload decoding failed"))
}
