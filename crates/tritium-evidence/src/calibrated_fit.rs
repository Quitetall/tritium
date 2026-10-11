//! Typed calibrated-fit views over the existing log, never independent receipts.

use super::direct_fit::{canonical_envelope, validate_fit_outputs};
use super::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, EvidenceRecorder,
    VerifiedLog, verify_log,
};
use tritium_schema::{
    CalibratedLevelsFitted, CalibrationFactor, CalibrationOutput, CalibrationPartition,
};

impl EvidenceEvent for CalibratedLevelsFitted {
    const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_levels_fitted";
    const VERSION: u16 = 1;
}

/// Registry entry for the additive actual-cache fitting observation candidate.
pub const CALIBRATED_LEVELS_FITTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: CalibratedLevelsFitted::SCHEMA_ID,
    version: CalibratedLevelsFitted::VERSION,
};

/// Immutable initial observation, not a model conversion or independent refit.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedFitReceiptView {
    envelope: EvidenceEnvelope,
    observation: CalibratedLevelsFitted,
}
impl CalibratedFitReceiptView {
    /// Actual typed execution evidence, not later mutable output state.
    #[must_use]
    pub fn observation(&self) -> &CalibratedLevelsFitted {
        &self.observation
    }
    /// Actual recorded event, without an invented completed-run root.
    #[must_use]
    pub fn envelope(&self) -> &EvidenceEnvelope {
        &self.envelope
    }
    /// Recheck log integrity, strict typed payload and cross-field bindings.
    /// Other event registrations and independent numerical truth remain separate.
    /// # Errors
    /// Invalid chain/root, version, payload, cache/plan/factor/output correspondence.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        log.events
            .iter()
            .filter(|event| event.schema == CalibratedLevelsFitted::SCHEMA_ID)
            .map(Self::from_event)
            .collect()
    }
    fn from_event(event: &EvidenceEnvelope) -> Result<Self, EvidenceError> {
        if event.schema != CalibratedLevelsFitted::SCHEMA_ID
            || event.v != CalibratedLevelsFitted::VERSION
        {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                event.schema, event.v
            )));
        }
        let envelope = canonical_envelope(event).map_err(invalid_payload)?;
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
        schema: CalibratedLevelsFitted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
fn validate_bindings(observation: &CalibratedLevelsFitted) -> Result<(), EvidenceError> {
    let plan = &observation.plan;
    let calibration = plan
        .calibration_plan()
        .map_err(|_| invalid_payload("invalid calibrated request"))?;
    if observation.factors.plan != calibration {
        return Err(invalid_payload(
            "factor request differs from cache/fit request",
        ));
    }
    validate_fit_outputs(
        &plan.cache.request.tensor.name,
        &plan.direct,
        observation.rayon_threads,
        &observation.outputs,
    )
    .map_err(invalid_payload)?;
    let columns = plan.direct.layout.cols;
    let width = match plan.partition {
        CalibrationPartition::Full => columns,
        CalibrationPartition::Groups { columns } => u64::from(columns),
        _ => return Err(invalid_payload("unsupported partition")),
    };
    let factors = &observation.factors;
    if factors.inputs.len() as u64 != columns.div_ceil(width) {
        return Err(invalid_payload("incomplete input factor inventory"));
    }
    let first = &factors.inputs[0];
    let mut remaining = columns;
    for factor in &factors.inputs {
        validate_factor(factor, first, remaining.min(width), plan.cache.samples)?;
        remaining -= factor.dimension;
    }
    match (calibration.output, &factors.output) {
        (CalibrationOutput::None, None) => {}
        (CalibrationOutput::DenseFisher | CalibrationOutput::IndexedFisher, Some(output)) => {
            validate_factor(output, first, plan.direct.layout.rows, plan.cache.samples)?;
        }
        _ => return Err(invalid_payload("output factor encoding/inventory mismatch")),
    }
    Ok(())
}
fn validate_factor(
    factor: &CalibrationFactor,
    first: &CalibrationFactor,
    dimension: u64,
    samples: u64,
) -> Result<(), EvidenceError> {
    if factor.dimension != dimension
        || factor.samples != samples
        || factor.selected == 0
        || factor.selected > factor.samples
        || !factor.total_weight.is_finite()
        || factor.total_weight <= 0.0
        || factor.selection.0 == [0; 32]
        || factor.content.0 == [0; 32]
        || factor.selected != first.selected
        || factor.total_weight.to_bits() != first.total_weight.to_bits()
        || factor.selection != first.selection
    {
        return Err(invalid_payload(
            "factor dimensions/counts/weights/selection mismatch",
        ));
    }
    Ok(())
}
impl EvidenceRecorder {
    /// Validate and emit a calibrated observation into the existing hash chain.
    /// # Errors
    /// Invalid bindings, evidence context, JCS serialization or strict decoding.
    pub fn emit_calibrated_levels_fitted(
        &mut self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        observation: &CalibratedLevelsFitted,
    ) -> Result<CalibratedFitReceiptView, EvidenceError> {
        validate_bindings(observation)?;
        CalibratedFitReceiptView::from_event(self.emit(span, parent, logical_time, observation)?)
    }
}
