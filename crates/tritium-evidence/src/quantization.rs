//! Complete fit-stage views; publication and independent qualification are separate.

use super::{
    DirectFitReceiptView, EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent,
    VerifiedLog, decode_hex, verify_log,
};
use std::collections::BTreeMap;
use tritium_schema::{
    BlobReference, EvidenceReference, Level, LevelFit, ManifestAsset, ModelManifest, PlaneCounts,
    QuantAction, QuantizationCompleted, QuantizationStarted, SchemaId, TensorStorage,
};

impl EvidenceEvent for QuantizationStarted {
    const SCHEMA_ID: &'static str = "tritium.quantize.quantization_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for QuantizationCompleted {
    const SCHEMA_ID: &'static str = "tritium.quantize.quantization_completed";
    const VERSION: u16 = 1;
}
/// Registry entry for the complete original model request.
pub const QUANTIZATION_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: QuantizationStarted::SCHEMA_ID,
    version: QuantizationStarted::VERSION,
};
/// Registry entry for completed fitting, not successful publication.
pub const QUANTIZATION_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: QuantizationCompleted::SCHEMA_ID,
    version: QuantizationCompleted::VERSION,
};

/// Immutable complete fit-stage bindings folded from the actual sealed run.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizationReceiptView {
    start: QuantizationStarted,
    completed: QuantizationCompleted,
    fits: Vec<DirectFitReceiptView>,
    manifest: ModelManifest,
}

impl QuantizationReceiptView {
    /// Actual original request.
    #[must_use]
    pub fn started(&self) -> &QuantizationStarted {
        &self.start
    }
    /// Actual complete original-source and encoded-output bindings.
    #[must_use]
    pub fn completed(&self) -> &QuantizationCompleted {
        &self.completed
    }
    /// Complete requested fitted tensor observations, not independent refits.
    #[must_use]
    pub fn fits(&self) -> &[DirectFitReceiptView] {
        &self.fits
    }
    /// Manifest bound to the verified log's actual run/span/root ancestry.
    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Fold complete conversions, rechecking mutable log integrity and bindings.
    /// An unfinished start produces no completed view. Unknown registrations
    /// remain a caller obligation; source truth and model quality are not proved.
    /// # Errors
    /// Bad chain/root/version/payload or incomplete/mismatched fit-stage bindings.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        let fits = DirectFitReceiptView::fold(log)?;
        let mut starts = BTreeMap::<&str, (&EvidenceEnvelope, QuantizationStarted)>::new();
        let mut completions = BTreeMap::<&str, (&EvidenceEnvelope, QuantizationCompleted)>::new();
        for event in &log.events {
            if event.schema == QuantizationStarted::SCHEMA_ID {
                let value: QuantizationStarted = decode(event)?;
                value
                    .plan
                    .validate()
                    .map_err(|_| invalid("invalid original plan"))?;
                if event.parent.is_some()
                    || event.seq != 0
                    || starts.insert(&event.span, (event, value)).is_some()
                {
                    return Err(invalid("duplicate or non-root start"));
                }
            } else if event.schema == QuantizationCompleted::SCHEMA_ID {
                let value = decode(event)?;
                if event.parent.is_some()
                    || event.seq != 1
                    || completions.insert(&event.span, (event, value)).is_some()
                {
                    return Err(invalid("duplicate or non-root completion"));
                }
            }
        }
        let root = decode_hex(&log.root_digest)?;
        let mut views = Vec::new();
        for (span, (completion_event, completed)) in completions {
            let (start_event, start) = starts
                .remove(span)
                .ok_or_else(|| invalid("completion has no start"))?;
            if completion_event.run != start_event.run || completion_event.t < start_event.t {
                return Err(invalid("completion ancestry mismatch"));
            }
            let selected = fits
                .iter()
                .filter(|fit| fit.envelope().parent.as_deref() == Some(span))
                .cloned()
                .collect::<Vec<_>>();
            let plan = &start.plan;
            if completed.sources.len() != plan.tensors.len()
                || completed
                    .assets
                    .iter()
                    .map(|a| &a.name)
                    .ne(plan.assets.iter())
                || completed
                    .levels
                    .iter()
                    .map(|l| &l.name)
                    .ne(plan.levels.iter())
            {
                return Err(invalid("incomplete source, asset or level inventory"));
            }
            if completed
                .sources
                .iter()
                .map(|s| &s.source)
                .ne(plan.tensors.iter().map(|t| &t.source))
            {
                return Err(invalid("source interpretation changed"));
            }
            let mut by_tensor = BTreeMap::new();
            for fit in &selected {
                let envelope = fit.envelope();
                if envelope.run != start_event.run
                    || envelope.seq != 0
                    || by_tensor
                        .insert(fit.observation().tensor.as_str(), fit.observation())
                        .is_some()
                {
                    return Err(invalid("duplicate or foreign direct fit"));
                }
            }
            for level in &completed.levels {
                if level
                    .tensors
                    .iter()
                    .map(|t| &t.name)
                    .ne(plan.tensors.iter().map(|t| &t.source.name))
                {
                    return Err(invalid("incomplete output inventory"));
                }
            }
            for (index, (request, source)) in
                plan.tensors.iter().zip(&completed.sources).enumerate()
            {
                let fitted = by_tensor.remove(request.source.name.as_str());
                match &request.action {
                    QuantAction::KeepDense if fitted.is_some() => {
                        return Err(invalid("dense source has a fit claim"));
                    }
                    QuantAction::KeepDense => {}
                    QuantAction::Fit { plan, metric } => {
                        let fit = fitted.ok_or_else(|| invalid("requested fit absent"))?;
                        if &fit.plan != plan.as_ref() || &fit.metric != metric {
                            return Err(invalid("fit plan or metric changed"));
                        }
                    }
                    _ => return Err(invalid("unsupported quant action")),
                }
                for level in &completed.levels {
                    let tensor = &level.tensors[index];
                    if tensor.blob.shard.is_some() {
                        return Err(invalid("unexpected output shard"));
                    }
                    match &request.action {
                        QuantAction::KeepDense => {
                            if tensor.blob.id != source.content
                                || tensor.storage
                                    != (TensorStorage::Dense {
                                        shape: request.source.shape.clone(),
                                        precision: request.source.precision,
                                    })
                            {
                                return Err(invalid("dense source not preserved"));
                            }
                        }
                        QuantAction::Fit { plan, .. } => {
                            let count = plan
                                .levels
                                .iter()
                                .find(|l| l.name == level.name)
                                .ok_or_else(|| invalid("fit level absent"))?
                                .planes;
                            let mut layout = plan.layout;
                            layout.max_planes = count;
                            if tensor.storage
                                != (TensorStorage::Additive {
                                    layout,
                                    planes: PlaneCounts::Uniform(count),
                                })
                            {
                                return Err(invalid("fitted output interpretation changed"));
                            }
                        }
                        _ => return Err(invalid("unsupported quant action")),
                    }
                }
            }
            if !by_tensor.is_empty() {
                return Err(invalid("unrequested fit"));
            }
            let ancestry = EvidenceReference {
                run: start_event.run.clone(),
                span: span.into(),
                root,
            };
            let manifest = ModelManifest {
                schema: SchemaId { major: 1, minor: 0 },
                architecture: plan.architecture.clone(),
                assets: completed
                    .assets
                    .iter()
                    .map(|a| ManifestAsset {
                        name: a.name.clone(),
                        blob: BlobReference {
                            id: a.content,
                            shard: None,
                        },
                    })
                    .collect(),
                levels: completed
                    .levels
                    .iter()
                    .map(|l| Level {
                        name: l.name.clone(),
                        tensors: l.tensors.clone(),
                        fit: if selected.is_empty() {
                            LevelFit::Imported
                        } else {
                            LevelFit::Direct {
                                evidence: vec![ancestry.clone()],
                            }
                        },
                    })
                    .collect(),
                shards: Vec::new(),
                legacy_ids: Vec::new(),
            };
            manifest
                .validate()
                .map_err(|_| invalid("invalid complete manifest"))?;
            views.push(Self {
                start,
                completed,
                fits: selected,
                manifest,
            });
        }
        Ok(views)
    }
}

fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: QuantizationCompleted::SCHEMA_ID.into(),
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
        .map_err(|_| invalid("strict payload decoding failed"))
}
