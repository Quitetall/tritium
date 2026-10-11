//! Recorded transform structure is distinct from actual source/numerical replay.
use crate::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, VerifiedLog, decode_hex,
    original_source_id, verify_log,
};
use std::collections::BTreeMap;
use tritium_schema::{EvidenceReference, SalienceTransformCompleted, SalienceTransformStarted};

impl EvidenceEvent for SalienceTransformStarted {
    const SCHEMA_ID: &'static str = "tritium.quantize.salience_transform_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for SalienceTransformCompleted {
    const SCHEMA_ID: &'static str = "tritium.quantize.salience_transform_completed";
    const VERSION: u16 = 1;
}
/// Registration for a declared transform request, not observed source truth.
pub const SALIENCE_TRANSFORM_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: SalienceTransformStarted::SCHEMA_ID,
    version: SalienceTransformStarted::VERSION,
};
/// Registration for actual complete A/B observations, not publication or qualification.
pub const SALIENCE_TRANSFORM_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: SalienceTransformCompleted::SCHEMA_ID,
    version: SalienceTransformCompleted::VERSION,
};

/// Complete recorded transform ancestry and structure, not byte/numerical truth.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SalienceTransformView {
    started: SalienceTransformStarted,
    completed: SalienceTransformCompleted,
    reference: EvidenceReference,
}
impl SalienceTransformView {
    /// Typed request, including the explicitly declared statistics class.
    #[must_use]
    pub fn started(&self) -> &SalienceTransformStarted {
        &self.started
    }
    /// Recorded complete output observation; arithmetic needs actual replay.
    #[must_use]
    pub fn completed(&self) -> &SalienceTransformCompleted {
        &self.completed
    }
    /// Actual verified run/span/root reference, not a fabricated fit root.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }

    /// Recheck integrity, strict types/versions, ancestry, complete structure and
    /// both recorded content identities. An unfinished start produces no view.
    /// Unknown unrelated registrations remain a caller obligation. Actual bytes,
    /// numerical operations, graph eligibility and capture are not proved here.
    /// # Errors
    /// Invalid chain/version/payload, orphan/duplicate lifecycle, changed request,
    /// incomplete inventories, invalid scales/dtype or wrong recorded identities.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        let mut starts = BTreeMap::<&str, (&EvidenceEnvelope, SalienceTransformStarted)>::new();
        let mut completed =
            BTreeMap::<&str, (&EvidenceEnvelope, SalienceTransformCompleted)>::new();
        for event in &log.events {
            if event.schema == SalienceTransformStarted::SCHEMA_ID {
                let value: SalienceTransformStarted = decode(event)?;
                value
                    .plan
                    .validate()
                    .map_err(|_| invalid("invalid transform request"))?;
                if event.parent.is_some()
                    || event.seq != 0
                    || starts.insert(&event.span, (event, value)).is_some()
                {
                    return Err(invalid("duplicate or non-root transform start"));
                }
            } else if event.schema == SalienceTransformCompleted::SCHEMA_ID {
                let value: SalienceTransformCompleted = decode(event)?;
                if event.parent.is_some()
                    || event.seq != 1
                    || completed.insert(&event.span, (event, value)).is_some()
                {
                    return Err(invalid("duplicate or non-root transform completion"));
                }
            }
        }
        let root = decode_hex(&log.root_digest)?;
        let mut views = Vec::new();
        for (span, (event, completion)) in completed {
            let (start_event, start) = starts
                .remove(span)
                .ok_or_else(|| invalid("transform completion has no start"))?;
            let observation = &completion.observation;
            if event.run != start_event.run
                || event.t < start_event.t
                || start.plan != observation.plan
            {
                return Err(invalid("transform ancestry or request changed"));
            }
            observation
                .validate()
                .map_err(|_| invalid("invalid complete transform observation"))?;
            let a = original_source_id(
                &start.plan.architecture,
                &observation.original_tensors,
                &observation.assets,
            )
            .map_err(|_| invalid("invalid original inventory"))?;
            let b = original_source_id(
                &start.plan.architecture,
                &observation.derived_tensors,
                &observation.assets,
            )
            .map_err(|_| invalid("invalid derived inventory"))?;
            if a != observation.original_source || b != observation.derived_source {
                return Err(invalid("recorded transform source identities disagree"));
            }
            views.push(Self {
                started: start,
                completed: completion,
                reference: EvidenceReference {
                    run: start_event.run.clone(),
                    span: span.into(),
                    root,
                },
            });
        }
        Ok(views)
    }
}
fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: SalienceTransformCompleted::SCHEMA_ID.into(),
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
        .map_err(|_| invalid("strict transform payload decoding failed"))
}
