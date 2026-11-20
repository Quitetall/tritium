//! Typed owner-window execution records; replay is not independent numerical truth.
use super::direct_fit::{canonical_envelope, canonical_payload, validate_fit_outputs};
use super::{
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, EvidenceRecorder,
    VerifiedLog, verify_log,
};
use std::collections::BTreeSet;
use tritium_schema::{
    OwnerMetricRowsId, OwnerMetricStreamId, OwnerRowMetricBinding, OwnerScoreFitPlan,
    OwnerScoreLevelsFitted, WeightedOwnerWindow,
};

impl EvidenceEvent for OwnerScoreLevelsFitted {
    const SCHEMA_ID: &'static str = "tritium.quantize.owner_score_levels_fitted";
    const VERSION: u16 = 1;
}
/// Separate owner-score observation registration, never legacy K-FAC/v1 evidence.
pub const OWNER_SCORE_LEVELS_FITTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: OwnerScoreLevelsFitted::SCHEMA_ID,
    version: OwnerScoreLevelsFitted::VERSION,
};

/// Shared unchanged ADR0094 stream hash for producer and typed replay.
/// Actual cache/source/numerical truth is a separate receiver obligation.
pub fn owner_metric_stream_id(
    windows: &[WeightedOwnerWindow],
    damping: f64,
) -> Result<OwnerMetricStreamId, EvidenceError> {
    if windows.is_empty() || !damping.is_finite() || damping < 0.0 {
        return Err(invalid("invalid window/damping configuration"));
    }
    let mut total = 0.0_f64;
    let mut hash = blake3::Hasher::new_derive_key("tritium.owner-window.metric-stream.v1");
    hash.update(&(windows.len() as u64).to_le_bytes());
    for window in windows {
        if window.cache.as_bytes() == &[0; 32] || !window.weight.is_finite() || window.weight < 0.0
        {
            return Err(invalid("invalid window identity or weight"));
        }
        total += window.weight;
        hash.update(window.cache.as_bytes());
        hash.update(&bits(window.weight).to_le_bytes());
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid("zero/overflowing total weight"));
    }
    hash.update(&bits(damping).to_le_bytes());
    Ok(OwnerMetricStreamId::from_bytes(*hash.finalize().as_bytes()))
}

/// Content reference for a complete ordered row-metric inventory, not a refit.
pub fn owner_metric_rows_id(
    stream: OwnerMetricStreamId,
    columns: u64,
    rows: &[OwnerRowMetricBinding],
) -> Result<OwnerMetricRowsId, EvidenceError> {
    if stream.as_bytes() == &[0; 32] || columns == 0 || rows.is_empty() {
        return Err(invalid("empty row inventory"));
    }
    let mut hash = blake3::Hasher::new_derive_key("tritium.owner-window.row-inventory.v1");
    hash.update(stream.as_bytes());
    hash.update(&columns.to_le_bytes());
    hash.update(&(rows.len() as u64).to_le_bytes());
    let mut seen = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        if row.row != index as u64
            || row.content.as_bytes() == &[0; 32]
            || !seen.insert(*row.content.as_bytes())
        {
            return Err(invalid("incomplete or duplicate row metric inventory"));
        }
        hash.update(&row.row.to_le_bytes());
        hash.update(row.content.as_bytes());
    }
    Ok(OwnerMetricRowsId::from_bytes(*hash.finalize().as_bytes()))
}

/// Immutable log-derived execution view, not model conversion or qualification.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerScoreFitReceiptView {
    envelope: EvidenceEnvelope,
    observation: OwnerScoreLevelsFitted,
}
impl OwnerScoreFitReceiptView {
    /// Actual canonical execution payload.
    pub fn observation(&self) -> &OwnerScoreLevelsFitted {
        &self.observation
    }
    /// Actual event and ancestry, without an invented model root.
    pub fn envelope(&self) -> &EvidenceEnvelope {
        &self.envelope
    }
    /// Recheck sealed log, strict version/payload and complete typed bindings.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        log.events
            .iter()
            .filter(|event| event.schema == OwnerScoreLevelsFitted::SCHEMA_ID)
            .map(Self::from_event)
            .collect()
    }
    fn from_event(event: &EvidenceEnvelope) -> Result<Self, EvidenceError> {
        if event.v != OwnerScoreLevelsFitted::VERSION {
            return Err(EvidenceError::UnregisteredSchema(format!(
                "{}@{}",
                event.schema, event.v
            )));
        }
        let envelope = canonical_envelope(event).map_err(invalid)?;
        let observation = serde::Deserialize::deserialize(&envelope.payload)
            .map_err(|_| invalid("strict owner-fit payload decoding failed"))?;
        validate(&observation)?;
        Ok(Self {
            envelope,
            observation,
        })
    }
}
fn validate(observation: &OwnerScoreLevelsFitted) -> Result<(), EvidenceError> {
    let plan = &observation.plan;
    validate_plan(plan)?;
    validate_metrics(observation)
}
fn validate_plan(plan: &OwnerScoreFitPlan) -> Result<(), EvidenceError> {
    plan.validate()
        .map_err(|_| invalid("invalid owner-fit plan"))?;
    for window in &plan.windows {
        let actual =
            tritium_format::trit_owner_score::owner_score_cache_id(&window.binding.content)
                .map_err(|_| invalid("invalid canonical cache content"))?;
        if actual != window.binding.cache {
            return Err(invalid("cache content identity mismatch"));
        }
    }
    Ok(())
}
fn validate_metrics(observation: &OwnerScoreLevelsFitted) -> Result<(), EvidenceError> {
    let plan = &observation.plan;
    let windows = plan.windows.iter().map(|w| w.window()).collect::<Vec<_>>();
    if observation.stream != owner_metric_stream_id(&windows, plan.damping)?
        || observation.rows.len() as u64 != plan.direct.layout.rows
        || observation.rows_content
            != owner_metric_rows_id(
                observation.stream,
                plan.direct.layout.cols,
                &observation.rows,
            )?
        || observation.source_f32.0 == [0; 32]
        || observation
            .outputs
            .iter()
            .any(|output| output.content.as_bytes() == &[0; 32])
    {
        return Err(invalid("metric/source/output reference mismatch"));
    }
    validate_fit_outputs(
        &plan.windows[0].binding.content.tensor.name,
        &plan.direct,
        observation.rayon_threads,
        &observation.outputs,
    )
    .map_err(invalid)
}
impl EvidenceRecorder {
    /// Admit actual ancestry/time and a losslessly recordable plan before work.
    /// Executed finite options are never silently rounded by canonical JSON.
    pub fn check_owner_score_fit_context(
        &self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        plan: &OwnerScoreFitPlan,
    ) -> Result<(), EvidenceError> {
        self.check_emit::<OwnerScoreLevelsFitted>(span, parent, logical_time)?;
        validate_plan(plan)?;
        let payload =
            serde_json::to_value(plan).map_err(|_| invalid("plan serialization failed"))?;
        super::validate_json_numbers(&payload)?;
        let canonical: OwnerScoreFitPlan =
            serde_json::from_value(canonical_payload(&payload).map_err(invalid)?)
                .map_err(|_| invalid("canonical plan decoding failed"))?;
        if canonical != *plan {
            return Err(invalid("canonicalization changed numerical plan"));
        }
        validate_plan(&canonical)
    }
    /// Validate all bindings before emitting one actual owner-fit observation.
    pub fn emit_owner_score_levels_fitted(
        &mut self,
        span: &str,
        parent: Option<&str>,
        logical_time: u64,
        observation: &OwnerScoreLevelsFitted,
    ) -> Result<OwnerScoreFitReceiptView, EvidenceError> {
        self.check_owner_score_fit_context(span, parent, logical_time, &observation.plan)?;
        validate(observation)?;
        // Validate the actual canonical wire representation before append. If
        // JCS/JSON cannot preserve executed f64 options/identities, fail closed
        // without a success event rather than relabeling the numerical request.
        let payload = serde_json::to_value(observation)
            .map_err(|_| invalid("payload serialization failed"))?;
        let canonical: OwnerScoreLevelsFitted =
            serde_json::from_value(canonical_payload(&payload).map_err(invalid)?)
                .map_err(|_| invalid("canonical payload decoding failed"))?;
        validate(&canonical)?;
        if canonical != *observation {
            return Err(invalid("canonicalization changed executed request"));
        }
        OwnerScoreFitReceiptView::from_event(self.emit(span, parent, logical_time, observation)?)
    }
}
fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: OwnerScoreLevelsFitted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
fn bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}
