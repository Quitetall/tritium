//! Typed direct-fit views over the log, never independently stored receipts.

use super::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, EvidenceRecorder,
    VerifiedLog, verify_log,
};
use tritium_schema::{DirectFitOutput, DirectFitPlan, DirectLevelsFitted};

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
        verify_log(log)?;
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
        let envelope = canonical_envelope(event).map_err(invalid_payload)?;
        let observation = serde::Deserialize::deserialize(&envelope.payload)
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
    validate_fit_outputs(
        &observation.tensor,
        &observation.plan,
        observation.rayon_threads,
        &observation.outputs,
    )
    .map_err(invalid_payload)
}

pub(crate) fn canonical_envelope(
    event: &EvidenceEnvelope,
) -> Result<EvidenceEnvelope, &'static str> {
    let mut envelope = event.clone();
    let canonical = serde_json_canonicalizer::to_string(&envelope.payload)
        .map_err(|_| "payload canonicalization failed")?;
    envelope.payload =
        serde_json::from_str(&canonical).map_err(|_| "canonical payload decoding failed")?;
    Ok(envelope)
}

pub(crate) fn validate_fit_outputs(
    tensor: &str,
    plan: &DirectFitPlan,
    rayon_threads: u32,
    outputs: &[DirectFitOutput],
) -> Result<(), &'static str> {
    let valid_name =
        |name: &str| !name.is_empty() && name.len() <= 4096 && !name.chars().any(char::is_control);
    if !valid_name(tensor) || rayon_threads == 0 {
        return Err("invalid tensor name or Rayon pool size");
    }
    let levels = &plan.levels;
    if levels.is_empty() || levels.len() != outputs.len() {
        return Err("incomplete output inventory");
    }
    let mut names = std::collections::BTreeSet::new();
    for (level, output) in levels.iter().zip(outputs) {
        if !valid_name(&level.name)
            || !names.insert(&level.name)
            || level.planes == 0
            || level.planes > plan.layout.max_planes
            || output.level != level.name
            || output.planes != level.planes
        {
            return Err("output does not match requested level");
        }
    }
    // This is structural replay, not numerical profile admission or a refit.
    plan.layout.validate().map_err(|_| "invalid layout")?;
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
