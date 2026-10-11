//! Registered structural capture observations, distinct from actual graph replay.
use crate::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, VerifiedLog, decode_hex,
    verify_log,
};
use std::collections::BTreeMap;
use tritium_schema::{
    CalibrationCaptureCompleted, CalibrationCaptureStarted, EvidenceReference,
    next_token_capture_objective,
};

impl EvidenceEvent for CalibrationCaptureStarted {
    const SCHEMA_ID: &'static str = "tritium.calibration.capture_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for CalibrationCaptureCompleted {
    const SCHEMA_ID: &'static str = "tritium.calibration.capture_completed";
    const VERSION: u16 = 1;
}
/// Registered actual B capture request, not execution clearance.
pub const CALIBRATION_CAPTURE_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: CalibrationCaptureStarted::SCHEMA_ID,
    version: CalibrationCaptureStarted::VERSION,
};
/// Registered actual immutable cache observation, not independent reproduction.
pub const CALIBRATION_CAPTURE_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: CalibrationCaptureCompleted::SCHEMA_ID,
    version: CalibrationCaptureCompleted::VERSION,
};

/// Complete typed capture ancestry; numerical/tokenization truth requires replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationCaptureView {
    started: CalibrationCaptureStarted,
    completed: CalibrationCaptureCompleted,
    reference: EvidenceReference,
}
impl CalibrationCaptureView {
    /// Actual controller-owned B/input/profile/limits request.
    #[must_use]
    pub fn started(&self) -> &CalibrationCaptureStarted {
        &self.started
    }
    /// Recorded ordered encoded windows and immutable cache bindings.
    #[must_use]
    pub fn completed(&self) -> &CalibrationCaptureCompleted {
        &self.completed
    }
    /// Actual sealed span and run root, not a quality or acceptance receipt.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }

    /// Recheck integrity, strict types/versions, root ancestry and full context.
    /// Incomplete starts return no completed view; unrelated events are not
    /// registered by this fold. Actual tokenizer/gradient/cache bytes need replay.
    /// # Errors
    /// Unknown version/fields, orphan/duplicate/changed ancestry or output context.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        let mut starts = BTreeMap::new();
        let mut completions = BTreeMap::new();
        for event in &log.events {
            if event.schema == CalibrationCaptureStarted::SCHEMA_ID {
                let value: CalibrationCaptureStarted = decode(event)?;
                if !value.plan.valid()
                    || event.parent.is_some()
                    || event.seq != 0
                    || starts.insert(event.span.as_str(), (event, value)).is_some()
                {
                    return Err(invalid("invalid or duplicate capture start"));
                }
            } else if event.schema == CalibrationCaptureCompleted::SCHEMA_ID {
                let value: CalibrationCaptureCompleted = decode(event)?;
                if event.parent.is_some()
                    || event.seq != 1
                    || completions
                        .insert(event.span.as_str(), (event, value))
                        .is_some()
                {
                    return Err(invalid("invalid or duplicate capture completion"));
                }
            }
        }
        let root = decode_hex(&log.root_digest)?;
        let mut views = Vec::new();
        for (span, (event, completed)) in completions {
            let (start_event, started) = starts
                .remove(span)
                .ok_or_else(|| invalid("capture completion has no start"))?;
            if event.run != start_event.run
                || event.t < start_event.t
                || !completed.agrees_with(&started.plan)
            {
                return Err(invalid("capture context or ancestry disagrees"));
            }
            let windows: Vec<_> = completed.windows.iter().map(Vec::as_slice).collect();
            let objective = next_token_capture_objective(&windows)
                .ok_or_else(|| invalid("capture objective allocation/geometry rejected"))?;
            let objective = blake3::hash(&objective);
            if completed
                .caches
                .iter()
                .any(|b| b.objective.as_bytes() != objective.as_bytes())
            {
                return Err(invalid("capture labels disagree with objective identity"));
            }
            views.push(Self {
                started,
                completed,
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
        schema: CalibrationCaptureCompleted::SCHEMA_ID.into(),
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
        .map_err(|_| invalid("strict capture payload decoding failed"))
}
