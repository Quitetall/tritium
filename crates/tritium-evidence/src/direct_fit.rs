//! Typed direct-fit views over the log, never independently stored receipts.

use super::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, EvidenceRecorder,
    VerifiedLog, final_digests, root_digest, verify_events,
};
use tritium_schema::DirectLevelsFitted;

impl EvidenceEvent for DirectLevelsFitted {
    const SCHEMA_ID: &'static str = "tritium.quantize.direct_levels_fitted";
    const VERSION: u16 = 1;
}

/// Registry entry for the candidate source-bound controller observation.
pub const DIRECT_LEVELS_FITTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: DirectLevelsFitted::SCHEMA_ID,
    version: DirectLevelsFitted::VERSION,
};

/// Immutable observation view, not a completed model conversion or verdict.
/// Replay checks integrity/typed bindings, not truth by independently refitting.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectFitReceiptView {
    envelope: EvidenceEnvelope,
    observation: DirectLevelsFitted,
}

impl DirectFitReceiptView {
    /// Actual initial controller observation, not later mutable tensor state.
    #[must_use]
    pub fn observation(&self) -> &DirectLevelsFitted {
        &self.observation
    }

    /// Original recorded event, without an invented completed-run root.
    #[must_use]
    pub fn envelope(&self) -> &EvidenceEnvelope {
        &self.envelope
    }

    /// Fold recognized fit observations, rechecking a potentially modified log.
    /// Other schema registrations remain the caller's obligation.
    /// # Errors
    /// Invalid chain/root, event version, strict payload or output bindings.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_events(&log.events)?;
        if root_digest(&final_digests(&log.events)?)? != log.root_digest {
            return Err(EvidenceError::RootMismatch);
        }
        log.events
            .iter()
            .filter(|event| event.schema == DirectLevelsFitted::SCHEMA_ID)
            .map(Self::from_event)
            .collect()
    }

    fn from_event(event: &EvidenceEnvelope) -> Result<Self, EvidenceError> {
        if event.schema != DirectLevelsFitted::SCHEMA_ID || event.v != DirectLevelsFitted::VERSION {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                event.schema, event.v
            )));
        }
        // JCS intentionally canonicalizes 1.0 to 1. A recorded and replayed
        // view must expose the same wire representation, not serde's transient
        // integer-vs-float Number distinction. Never change the hashed meaning.
        let mut envelope = event.clone();
        let canonical = serde_json_canonicalizer::to_string(&envelope.payload)
            .map_err(|_| invalid_payload("payload canonicalization failed"))?;
        envelope.payload = serde_json::from_str(&canonical)
            .map_err(|_| invalid_payload("canonical payload decoding failed"))?;
        let observation = serde_json::from_value(envelope.payload.clone())
            .map_err(|error| invalid_payload(&format!("{:?}", error.classify())))?;
        validate_bindings(&observation)?;
        Ok(Self {
            envelope,
            observation,
        })
    }
}

fn invalid_payload(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: DirectLevelsFitted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}

fn validate_bindings(observation: &DirectLevelsFitted) -> Result<(), EvidenceError> {
    let valid_name =
        |name: &str| !name.is_empty() && name.len() <= 4096 && !name.chars().any(char::is_control);
    if !valid_name(&observation.tensor) || observation.rayon_threads == 0 {
        return Err(invalid_payload("invalid tensor name or Rayon pool size"));
    }
    let levels = &observation.plan.levels;
    if levels.is_empty() || levels.len() != observation.outputs.len() {
        return Err(invalid_payload("incomplete output inventory"));
    }
    let mut names = std::collections::BTreeSet::new();
    for (level, output) in levels.iter().zip(&observation.outputs) {
        if !valid_name(&level.name)
            || !names.insert(&level.name)
            || level.planes == 0
            || level.planes > observation.plan.layout.max_planes
            || output.level != level.name
            || output.planes != level.planes
        {
            return Err(invalid_payload("output does not match requested level"));
        }
    }
    // This is structural replay, not numerical profile admission or a refit.
    observation
        .plan
        .layout
        .validate()
        .map_err(|_| invalid_payload("invalid layout"))?;
    Ok(())
}

impl EvidenceRecorder {
    /// Validate/record the candidate observation and return a log-derived view.
    /// # Errors
    /// Invalid bindings, registry/span/time/JCS or strict payload failure.
    pub fn emit_direct_levels_fitted(
        &mut self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        observation: &DirectLevelsFitted,
    ) -> Result<DirectFitReceiptView, EvidenceError> {
        validate_bindings(observation)?;
        DirectFitReceiptView::from_event(self.emit(span, parent, logical_time, observation)?)
    }
}
