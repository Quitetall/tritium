//! Recorded/replayed text-load views, not independently stored receipts.

use super::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, EvidenceRecorder,
    ModelLoaded, VerifiedLog, verify_log,
};

impl EvidenceEvent for ModelLoaded {
    const SCHEMA_ID: &'static str = "tritium.runtime.model_loaded";
    const VERSION: u16 = 1;
}

/// Registry entry for the candidate initial text-load observation (ADR 0054).
pub const MODEL_LOADED_REGISTRATION: EventRegistration = EventRegistration {
    schema: ModelLoaded::SCHEMA_ID,
    version: ModelLoaded::VERSION,
};

/// Immutable view of one recorded initial-load envelope and typed observation.
///
/// No completed-run root or quality/release verdict is invented. A newly
/// recorded view is not independently verified evidence; replay checks chains
/// and the supplied root but does not prove a producer's declaration true by
/// reloading an artifact or independently executing a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelLoadReceiptView {
    envelope: EvidenceEnvelope,
    observation: ModelLoaded,
}

impl ModelLoadReceiptView {
    /// Borrow the initial load observation, not current mutable runner state.
    #[must_use]
    pub fn observation(&self) -> &ModelLoaded {
        &self.observation
    }

    /// Borrow its run/span/sequence/event digest and original recorded payload.
    #[must_use]
    pub fn envelope(&self) -> &EvidenceEnvelope {
        &self.envelope
    }

    /// Fold load views from replay, rechecking even a modified `VerifiedLog`.
    ///
    /// Checks all event chains and the supplied root, plus this event's schema
    /// version and strict payload. Other event registrations remain the caller's
    /// `verify_jsonl_with_registry` obligation. No artifact/quality verdict follows.
    /// # Errors
    /// Invalid chains/root, unsupported load version or invalid typed payload.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        log.events
            .iter()
            .filter(|event| event.schema == ModelLoaded::SCHEMA_ID)
            .map(Self::from_event)
            .collect()
    }

    fn from_event(event: &EvidenceEnvelope) -> Result<Self, EvidenceError> {
        if event.schema != ModelLoaded::SCHEMA_ID || event.v != ModelLoaded::VERSION {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                event.schema, event.v
            )));
        }
        let observation = serde::Deserialize::deserialize(&event.payload).map_err(
            |error: serde_json::Error| EvidenceError::InvalidPayload {
                schema: event.schema.clone(),
                detail: format!("{:?}", error.classify()),
            },
        )?;
        Ok(Self {
            envelope: event.clone(),
            observation,
        })
    }
}

impl EvidenceRecorder {
    /// Record a registered initial-load event and return its immutable view.
    ///
    /// This is recording, not completed-run or independent qualification.
    /// # Errors
    /// Registration, span/time/JCS or strict typed payload failure.
    pub fn emit_model_loaded(
        &mut self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        observation: &ModelLoaded,
    ) -> Result<ModelLoadReceiptView, EvidenceError> {
        ModelLoadReceiptView::from_event(self.emit(span, parent, logical_time, observation)?)
    }
}
