//! Complete fit-stage views; publication and independent qualification are separate.

use super::{
    CalibratedFitReceiptView, CalibratedQuantizationReceiptView, DirectFitReceiptView,
    EventRegistration, EvidenceEnvelope, EvidenceError, EvidenceEvent, OwnerScoreFitReceiptView,
    VerifiedLog, decode_hex, verify_log,
};
use std::collections::BTreeMap;
use tritium_schema::{
    BlobReference, CalibratedQuantizationCompleted, CalibratedQuantizationStarted,
    EvidenceReference, Level, LevelFit, ManifestAsset, ModelManifest, ModelQuantPlan,
    OriginalSourceId, PlaneCounts, QuantActionRef, QuantizationCompleted, QuantizationStarted,
    SchemaId, TensorStorage,
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
    source_id: OriginalSourceId,
    reference: EvidenceReference,
}

impl QuantizationReceiptView {
    /// Actual complete conversion run/span/run-root, including dense-only stages.
    /// Integrity and recorded agreement are not independent numerical qualification.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }
    /// Candidate identity derived from the actual recorded original inventory.
    /// This verifies recorded bindings, not independently observed source bytes.
    #[must_use]
    pub fn source_id(&self) -> OriginalSourceId {
        self.source_id
    }
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
        fold_stages(log, false)?
            .into_iter()
            .map(|stage| {
                let StageStart::Plain(start) = stage.start else {
                    return Err(invalid("wrong lifecycle"));
                };
                Ok(Self {
                    start,
                    completed: stage.completed,
                    fits: stage.fits,
                    manifest: stage.manifest,
                    source_id: stage.source_id,
                    reference: stage.reference,
                })
            })
            .collect()
    }
}

enum StageStart {
    Plain(QuantizationStarted),
    Calibrated(CalibratedQuantizationStarted),
}
impl StageStart {
    fn plan(&self) -> ModelQuantPlan<'_> {
        match self {
            Self::Plain(s) => ModelQuantPlan::Uncalibrated(&s.plan),
            Self::Calibrated(s) => ModelQuantPlan::Calibrated(&s.plan),
        }
    }
}
struct StageView {
    start: StageStart,
    completed: QuantizationCompleted,
    fits: Vec<DirectFitReceiptView>,
    calibrated: Vec<CalibratedFitReceiptView>,
    manifest: ModelManifest,
    source_id: OriginalSourceId,
    reference: EvidenceReference,
}
pub(crate) fn fold_calibrated(
    log: &VerifiedLog,
) -> Result<Vec<CalibratedQuantizationReceiptView>, EvidenceError> {
    fold_stages(log, true)?
        .into_iter()
        .map(|stage| {
            let StageStart::Calibrated(start) = stage.start else {
                return Err(invalid("wrong lifecycle"));
            };
            Ok(CalibratedQuantizationReceiptView {
                start,
                completed: CalibratedQuantizationCompleted {
                    original_source: stage.source_id,
                    output: stage.completed,
                },
                fits: stage.fits,
                calibrated: stage.calibrated,
                manifest: stage.manifest,
                source_id: stage.source_id,
                reference: stage.reference,
            })
        })
        .collect()
}
fn fold_stages(log: &VerifiedLog, calibrated: bool) -> Result<Vec<StageView>, EvidenceError> {
    let (start_schema, completed_schema) = if calibrated {
        (
            CalibratedQuantizationStarted::SCHEMA_ID,
            CalibratedQuantizationCompleted::SCHEMA_ID,
        )
    } else {
        (
            QuantizationStarted::SCHEMA_ID,
            QuantizationCompleted::SCHEMA_ID,
        )
    };
    let invalid = |detail: &str| EvidenceError::InvalidPayload {
        schema: completed_schema.into(),
        detail: detail.into(),
    };
    verify_log(log)?;
    let fits = DirectFitReceiptView::fold(log)?;
    let calibrated_fits = CalibratedFitReceiptView::fold(log)?;
    let owner_fits = OwnerScoreFitReceiptView::fold(log)?;
    let mut starts = BTreeMap::<&str, (&EvidenceEnvelope, StageStart)>::new();
    let mut completions = BTreeMap::<
        &str,
        (
            &EvidenceEnvelope,
            QuantizationCompleted,
            Option<OriginalSourceId>,
        ),
    >::new();
    for event in &log.events {
        if event.schema == start_schema {
            let value = if calibrated {
                StageStart::Calibrated(decode(event, completed_schema)?)
            } else {
                StageStart::Plain(decode(event, completed_schema)?)
            };
            value
                .plan()
                .validate()
                .map_err(|_| invalid("invalid original plan"))?;
            if event.parent.is_some()
                || event.seq != 0
                || starts.insert(&event.span, (event, value)).is_some()
            {
                return Err(invalid("duplicate or non-root start"));
            }
        } else if event.schema == completed_schema {
            let (value, claimed_source) = if calibrated {
                let value: CalibratedQuantizationCompleted = decode(event, completed_schema)?;
                (value.output, Some(value.original_source))
            } else {
                (decode(event, completed_schema)?, None)
            };
            if event.parent.is_some()
                || event.seq != 1
                || completions
                    .insert(&event.span, (event, value, claimed_source))
                    .is_some()
            {
                return Err(invalid("duplicate or non-root completion"));
            }
        }
    }
    let root = decode_hex(&log.root_digest)?;
    let mut views = Vec::new();
    for (span, (completion_event, completed, claimed_source)) in completions {
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
        let selected_calibrated = calibrated_fits
            .iter()
            .filter(|fit| fit.envelope().parent.as_deref() == Some(span))
            .cloned()
            .collect::<Vec<_>>();
        let plan = start.plan();
        if owner_fits
            .iter()
            .any(|fit| fit.envelope().parent.as_deref() == Some(span))
        {
            return Err(invalid(
                "owner-score fit requires a separate model lifecycle",
            ));
        }
        if completed.sources.len() != plan.tensors().count()
            || completed
                .assets
                .iter()
                .map(|a| &a.name)
                .ne(plan.assets().iter())
            || completed
                .levels
                .iter()
                .map(|l| &l.name)
                .ne(plan.levels().iter())
        {
            return Err(invalid("incomplete source, asset or level inventory"));
        }
        if completed
            .sources
            .iter()
            .map(|s| &s.source)
            .ne(plan.tensors().map(|t| t.source))
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
        let mut by_calibrated_tensor = BTreeMap::new();
        for fit in &selected_calibrated {
            let envelope = fit.envelope();
            if envelope.run != start_event.run
                || envelope.seq != 0
                || by_calibrated_tensor
                    .insert(
                        fit.observation().plan.cache.request.tensor.name.as_str(),
                        fit.observation(),
                    )
                    .is_some()
            {
                return Err(invalid("duplicate or foreign calibrated fit"));
            }
        }
        for level in &completed.levels {
            if level
                .tensors
                .iter()
                .map(|t| &t.name)
                .ne(plan.tensors().map(|t| &t.source.name))
            {
                return Err(invalid("incomplete output inventory"));
            }
        }
        for (index, (request, source)) in plan.tensors().zip(&completed.sources).enumerate() {
            let fitted = by_tensor.remove(request.source.name.as_str());
            let calibrated_fit = by_calibrated_tensor.remove(request.source.name.as_str());
            match request.action {
                QuantActionRef::KeepDense if fitted.is_some() || calibrated_fit.is_some() => {
                    return Err(invalid("dense source has a fit claim"));
                }
                QuantActionRef::KeepDense => {}
                QuantActionRef::Fit { plan, metric } => {
                    if calibrated_fit.is_some() {
                        return Err(invalid("plain request has a calibrated fit"));
                    }
                    let fit = fitted.ok_or_else(|| invalid("requested fit absent"))?;
                    if &fit.plan != plan || &fit.metric != metric {
                        return Err(invalid("fit plan or metric changed"));
                    }
                }
                QuantActionRef::CalibratedFit(plan) => {
                    if fitted.is_some() {
                        return Err(invalid("calibrated request has a plain fit"));
                    }
                    let fit =
                        calibrated_fit.ok_or_else(|| invalid("requested calibrated fit absent"))?;
                    if &fit.plan != plan {
                        return Err(invalid("calibrated plan or cache changed"));
                    }
                }
                _ => return Err(invalid("unsupported quant action")),
            }
            for level in &completed.levels {
                let tensor = &level.tensors[index];
                if tensor.blob.shard.is_some() {
                    return Err(invalid("unexpected output shard"));
                }
                match request.action {
                    QuantActionRef::KeepDense => {
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
                    action if action.direct().is_some() => {
                        let plan = action.direct().ok_or_else(|| invalid("fit plan absent"))?;
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
        if !by_tensor.is_empty() || !by_calibrated_tensor.is_empty() {
            return Err(invalid("unrequested fit"));
        }
        let ancestry = EvidenceReference {
            run: start_event.run.clone(),
            span: span.into(),
            root,
        };
        let manifest = ModelManifest {
            schema: SchemaId { major: 1, minor: 0 },
            architecture: plan.architecture().into(),
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
                    fit: if selected.is_empty() && selected_calibrated.is_empty() {
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
        let source_id =
            super::original_source_id(plan.architecture(), &completed.sources, &completed.assets)
                .map_err(|_| invalid("invalid original source identity inventory"))?;
        if let ModelQuantPlan::Calibrated(request) = plan
            && (source_id != request.original_source || claimed_source != Some(source_id))
        {
            return Err(invalid(
                "calibrated original source differs from preparation",
            ));
        }
        views.push(StageView {
            start,
            completed,
            fits: selected,
            calibrated: selected_calibrated,
            manifest,
            source_id,
            reference: ancestry,
        });
    }
    Ok(views)
}

fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: QuantizationCompleted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
fn decode<E: EvidenceEvent + serde::de::DeserializeOwned>(
    event: &EvidenceEnvelope,
    error_schema: &str,
) -> Result<E, EvidenceError> {
    if event.v != E::VERSION {
        return Err(EvidenceError::UnregisteredSchema(format!(
            "{}@{}",
            event.schema, event.v
        )));
    }
    serde::Deserialize::deserialize(&event.payload).map_err(|_| EvidenceError::InvalidPayload {
        schema: error_schema.into(),
        detail: "strict payload decoding failed".into(),
    })
}
