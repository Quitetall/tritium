//! Native recovery record consistency; actual checkpoint replay remains separate.
use crate::{
    CalibratedQuantizationReceiptView, EventRegistration, EvidenceEnvelope, EvidenceError,
    EvidenceEvent, LegacyModelImportBoundView, LegacyModelImportView, OwnerQuantizationReceiptView,
    QuantizationReceiptView, VerifiedLog, decode_hex, verify_log,
};
use serde::de::DeserializeOwned;
use std::io::{Read, Seek};
use tritium_format::trit_model::{TritModelReader, encode_model_manifest};
use tritium_schema::{
    EvidenceReference, Level, LevelFit, ModelManifest, NativeRefinementAncestry,
    NativeRefinementCheckpointed, NativeRefinementCompleted, NativeRefinementParent,
    NativeRefinementPlan, NativeRefinementPlanId, NativeRefinementStarted, OriginalSourceId,
    PackageId, SchemaId,
};

impl EvidenceEvent for NativeRefinementStarted {
    const SCHEMA_ID: &'static str = "tritium.refine.native_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for NativeRefinementCheckpointed {
    const SCHEMA_ID: &'static str = "tritium.refine.native_checkpointed";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for NativeRefinementCompleted {
    const SCHEMA_ID: &'static str = "tritium.refine.native_completed";
    const VERSION: u16 = 1;
}
/// Explicit native request registration, not PTQ conversion.
pub const NATIVE_REFINEMENT_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: NativeRefinementStarted::SCHEMA_ID,
    version: 1,
};
/// Sparse durable cursor registration, not a qualification receipt.
pub const NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: NativeRefinementCheckpointed::SCHEMA_ID,
    version: 1,
};
/// Complete candidate observations registration, not publication/quality approval.
pub const NATIVE_REFINEMENT_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: NativeRefinementCompleted::SCHEMA_ID,
    version: 1,
};

/// Domain-separated exact native request ID, not a legacy PV recipe hash.
pub fn native_refinement_plan_id(
    plan: &NativeRefinementPlan,
) -> Result<NativeRefinementPlanId, EvidenceError> {
    plan.validate()
        .map_err(|_| invalid("invalid native request"))?;
    let value =
        serde_json::to_value(plan).map_err(|e| EvidenceError::Serialization(e.to_string()))?;
    let bytes = serde_json_canonicalizer::to_vec(&value)
        .map_err(|e| EvidenceError::Canonicalization(e.to_string()))?;
    let mut h = blake3::Hasher::new();
    h.update(b"tritium.native-refinement.plan.v1\0");
    h.update(&bytes);
    Ok(NativeRefinementPlanId::from_bytes(*h.finalize().as_bytes()))
}

/// Native state binding over exact checkpoint content, request and update cursor.
/// This scopes bytes; it does not parse/replay a checkpoint or authenticate execution.
#[must_use]
pub fn native_refinement_state_id(
    plan: NativeRefinementPlanId,
    completed_updates: u64,
    checkpoint: tritium_schema::BlobId,
) -> tritium_schema::NativeRefinementStateId {
    let mut h = blake3::Hasher::new();
    h.update(b"tritium.native-refinement.state.v1\0");
    h.update(plan.as_bytes());
    h.update(&completed_updates.to_le_bytes());
    h.update(checkpoint.as_bytes());
    tritium_schema::NativeRefinementStateId::from_bytes(*h.finalize().as_bytes())
}

/// Small owned parent observation; no duplicate full manifest/count arrays.
/// Its private construction binds actual typed package metadata to original
/// verified observations, not official source/numerical authentication.
#[derive(Clone, Debug)]
pub struct NativeRefinementParentView {
    binding: NativeRefinementParent,
    manifest_digest: [u8; 32],
}
impl NativeRefinementParentView {
    /// Actual observed package/level/model/source/evidence bindings.
    #[must_use]
    pub fn binding(&self) -> &NativeRefinementParent {
        &self.binding
    }

    /// Admit actual package bytes against a privately folded plain conversion.
    pub fn from_quantization<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        view: &QuantizationReceiptView,
    ) -> Result<Self, EvidenceError> {
        Self::checked(
            reader,
            level,
            view.manifest(),
            view.source_id(),
            view.reference().clone(),
        )
    }
    /// Admit actual package bytes against an actual calibrated conversion view.
    pub fn from_calibrated<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        view: &CalibratedQuantizationReceiptView,
    ) -> Result<Self, EvidenceError> {
        Self::checked(
            reader,
            level,
            view.manifest(),
            view.source_id(),
            view.reference().clone(),
        )
    }
    /// Admit actual package bytes against an actual owner-window conversion view.
    pub fn from_owner<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        view: &OwnerQuantizationReceiptView,
    ) -> Result<Self, EvidenceError> {
        Self::checked(
            reader,
            level,
            view.manifest(),
            view.source_id(),
            view.reference().clone(),
        )
    }
    /// Admit a previously folded refined child without treating it as Direct PTQ.
    pub fn from_refined<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        view: &NativeRefinementReceiptView,
    ) -> Result<Self, EvidenceError> {
        Self::checked(
            reader,
            level,
            &view.manifest,
            view.started.plan.parent.source,
            view.reference.clone(),
        )
    }
    /// Admit the exact immediate refined model from a standalone record-consistent view.
    /// Companion bytes require separate admission. Earlier ancestry/numerical history
    /// is not verified or retroactively authenticated by this constructor.
    pub fn from_refined_record<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        view: &crate::NativeScaleChildRecordView,
    ) -> Result<Self, EvidenceError> {
        view.verify_child(reader)?;
        if level != view.started().plan.child_level {
            return Err(invalid("immediate refined parent level differs"));
        }
        Ok(Self {
            binding: binding(
                reader,
                level,
                view.started().plan.parent.source,
                view.reference().clone(),
            )?,
            manifest_digest: manifest_digest(reader.manifest())?,
        })
    }
    /// Bind original import observations to actual complete imported bytes.
    /// This does not authenticate legacy/official source claims or reseal logs.
    pub fn from_import<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        log: &VerifiedLog,
    ) -> Result<Self, EvidenceError> {
        verify_log(log)?;
        let bound = LegacyModelImportBoundView::fold(log, reader.manifest())?;
        let full = LegacyModelImportView::fold(log)?;
        let (source, schema) = match (bound.as_slice(), full.as_slice()) {
            ([one], []) => (
                one.observation().companion_source,
                "tritium.migrate.legacy_model_import_bound_prepared",
            ),
            ([], [one]) if &one.observation().manifest == reader.manifest() => (
                one.observation().companion_source,
                "tritium.migrate.legacy_model_import_prepared",
            ),
            _ => return Err(invalid("expected one original import observation")),
        };
        let event = log
            .events
            .iter()
            .find(|e| e.schema == schema)
            .ok_or_else(|| invalid("missing import event"))?;
        let reference = EvidenceReference {
            run: event.run.clone(),
            span: event.span.clone(),
            root: decode_hex(&log.root_digest)?,
        };
        reader
            .verify()
            .map_err(|_| invalid("parent source changed"))?;
        Ok(Self {
            binding: binding(reader, level, source, reference)?,
            manifest_digest: manifest_digest(reader.manifest())?,
        })
    }
    fn checked<R: Read + Seek>(
        reader: &mut TritModelReader<R>,
        level: &str,
        manifest: &ModelManifest,
        source: OriginalSourceId,
        reference: EvidenceReference,
    ) -> Result<Self, EvidenceError> {
        reader
            .verify()
            .map_err(|_| invalid("parent source changed"))?;
        if reader.manifest() != manifest {
            return Err(invalid("parent manifest differs from original evidence"));
        }
        Ok(Self {
            binding: binding(reader, level, source, reference)?,
            manifest_digest: manifest_digest(manifest)?,
        })
    }
    /// Reverify the exact original package, selected level and metadata binding.
    /// This does not authenticate official source or numerical/quality claims.
    pub fn verify<R: Read + Seek>(
        &self,
        reader: &mut TritModelReader<R>,
    ) -> Result<(), EvidenceError> {
        reader
            .verify()
            .map_err(|_| invalid("parent source changed"))?;
        let observed = binding(
            reader,
            &self.binding.level,
            self.binding.source,
            self.binding.evidence.clone(),
        )?;
        if observed != self.binding || manifest_digest(reader.manifest())? != self.manifest_digest {
            return Err(invalid("parent package/level/model/metadata changed"));
        }
        Ok(())
    }
}

/// Complete recorded candidate, not actual numerical replay or publication authority.
/// Private construction requires original parent admission and a frozen request.
#[derive(Clone, Debug)]
pub struct NativeRefinementReceiptView {
    started: NativeRefinementStarted,
    checkpoint: NativeRefinementCheckpointed,
    completed: NativeRefinementCompleted,
    reference: EvidenceReference,
    manifest: ModelManifest,
}
impl NativeRefinementReceiptView {
    /// Original recorded request and execution scope.
    #[must_use]
    pub fn started(&self) -> &NativeRefinementStarted {
        &self.started
    }
    /// Exact final recorded checkpoint binding; bytes/replay still require verification.
    #[must_use]
    pub fn checkpoint(&self) -> &NativeRefinementCheckpointed {
        &self.checkpoint
    }
    /// Complete observed output/validation bindings, not quality acceptance.
    #[must_use]
    pub fn completed(&self) -> &NativeRefinementCompleted {
        &self.completed
    }
    /// Actual native run/span/root, not resealed parent evidence.
    #[must_use]
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }
    /// One complete schema1.1 child, with preserved parent interpretations/assets.
    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Reverify original source/log bindings and exact requested lifecycle.
    /// Incomplete runs return no complete view. This is record consistency, not
    /// proof that checkpoint bytes exist, recovery ran or quality improved.
    pub fn fold<R: Read + Seek>(
        log: &VerifiedLog,
        reader: &mut TritModelReader<R>,
        parent: &NativeRefinementParentView,
        requested: &NativeRefinementPlan,
    ) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        parent.verify(reader)?;
        requested
            .validate_parent(reader.manifest(), parent.binding())
            .map_err(|_| invalid("requested parent/inventory mismatch"))?;
        let Some(NativeRecoveryRecord {
            started,
            checkpoint,
            completed,
            reference,
            plan: id,
        }) = fold_native_recovery(log, requested)?
        else {
            return Ok(Vec::new());
        };
        let level = reader
            .manifest()
            .levels
            .iter()
            .find(|l| l.name == requested.parent.level)
            .ok_or_else(|| invalid("missing parent level"))?;
        if completed
            .tensors
            .iter()
            .map(|t| &t.name)
            .ne(level.tensors.iter().map(|t| &t.name))
            || completed.validation.len() != requested.validation.len()
            || completed
                .validation
                .iter()
                .zip(&requested.validation)
                .any(|(a, b)| {
                    a.content != b.content
                        || !a.parent_loss.is_finite()
                        || !a.child_loss.is_finite()
                })
        {
            return Err(invalid("incomplete output or validation observations"));
        }
        let mut tensors = level.tensors.clone();
        for ((original, child), observed) in level
            .tensors
            .iter()
            .zip(&mut tensors)
            .zip(&completed.tensors)
        {
            let expected = match checkpoint
                .owners
                .binary_search_by(|t| t.name.cmp(&original.name))
            {
                Ok(i) => checkpoint.owners[i].content,
                Err(_) => original.blob.id,
            };
            if observed.content != expected {
                return Err(invalid("selected checkpoint or preserved bytes differ"));
            }
            child.blob.id = observed.content;
            child.blob.shard = None;
        }
        let mut assets = reader.manifest().assets.clone();
        for asset in &mut assets {
            asset.blob.shard = None;
        }
        let manifest = ModelManifest {
            schema: SchemaId { major: 1, minor: 1 },
            architecture: reader.manifest().architecture.clone(),
            assets,
            levels: vec![Level {
                name: requested.child_level.clone(),
                fit: LevelFit::Refined {
                    ancestry: Box::new(NativeRefinementAncestry {
                        parent: requested.parent.clone(),
                        plan: id,
                        evidence: reference.clone(),
                    }),
                },
                tensors,
            }],
            shards: Vec::new(),
            legacy_ids: Vec::new(),
        };
        manifest
            .validate()
            .map_err(|_| invalid("invalid refined child manifest"))?;
        Ok(vec![Self {
            started,
            checkpoint,
            completed,
            reference,
            manifest,
        }])
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativeRecoveryRecord {
    pub started: NativeRefinementStarted,
    pub checkpoint: NativeRefinementCheckpointed,
    pub completed: NativeRefinementCompleted,
    pub reference: EvidenceReference,
    pub plan: NativeRefinementPlanId,
}
/// Strict lifecycle only; parent bytes and output interpretations remain separate.
pub(crate) fn fold_native_recovery(
    log: &VerifiedLog,
    requested: &NativeRefinementPlan,
) -> Result<Option<NativeRecoveryRecord>, EvidenceError> {
    verify_log(log)?;
    requested
        .validate()
        .map_err(|_| invalid("invalid native request"))?;
    let id = native_refinement_plan_id(requested)?;
    let native: Vec<_> = log
        .events
        .iter()
        .filter(|e| e.schema.starts_with("tritium.refine.native_"))
        .collect();
    if native.is_empty() {
        return Ok(None);
    }
    let first = native[0];
    if first.schema != NativeRefinementStarted::SCHEMA_ID
        || first.seq != 0
        || first.parent.is_some()
    {
        return Err(invalid("native run must start at a root"));
    }
    let started: NativeRefinementStarted = decode(first)?;
    started
        .execution
        .validate()
        .map_err(|_| invalid("invalid execution scope"))?;
    if &started.plan != requested {
        return Err(invalid("recorded request differs from frozen request"));
    }
    if log.events.iter().any(|e| {
        (e.parent.is_none() && e.span != first.span)
            || (e.span == first.span && !e.schema.starts_with("tritium.refine.native_"))
    }) {
        return Err(invalid("foreign lifecycle in native root"));
    }
    let mut last: Option<NativeRefinementCheckpointed> = None;
    let mut completed = None;
    for event in native.into_iter().skip(1) {
        if event.span != first.span || event.parent.is_some() || completed.is_some() {
            return Err(invalid(
                "multiple/nonroot/after-completion native lifecycle",
            ));
        }
        if event.schema == NativeRefinementCheckpointed::SCHEMA_ID {
            let state: NativeRefinementCheckpointed = decode(event)?;
            if state.plan != id
                || state.state
                    != native_refinement_state_id(
                        state.plan,
                        state.completed_updates,
                        state.checkpoint,
                    )
                || state.completed_updates == 0
                || state.completed_updates > requested.training.len() as u64
                || state.checkpoint_bytes == 0
                || state.checkpoint_bytes > requested.limits.max_checkpoint_bytes
                || state.state_payload_bytes == 0
                || state.state_payload_bytes > requested.limits.max_state_bytes
                || state
                    .owners
                    .iter()
                    .map(|o| &o.name)
                    .ne(requested.owners.iter())
                || last.as_ref().is_some_and(|old| {
                    state.completed_updates <= old.completed_updates
                        || state.checkpoint == old.checkpoint
                        || state.state == old.state
                })
            {
                return Err(invalid("invalid/incomplete/nonadvancing checkpoint"));
            }
            last = Some(state);
        } else if event.schema == NativeRefinementCompleted::SCHEMA_ID {
            let output: NativeRefinementCompleted = decode(event)?;
            let state = last
                .as_ref()
                .ok_or_else(|| invalid("completion without checkpoint"))?;
            if output.plan != id
                || output.checkpoint != state.checkpoint
                || state.completed_updates != requested.training.len() as u64
            {
                return Err(invalid("completion before final checkpoint"));
            }
            completed = Some(output);
        } else {
            return Err(invalid("unknown or repeated native lifecycle event"));
        }
    }
    let Some(completed) = completed else {
        return Ok(None);
    };
    let checkpoint = last.ok_or_else(|| invalid("missing final checkpoint"))?;
    let reference = EvidenceReference {
        run: first.run.clone(),
        span: first.span.clone(),
        root: decode_hex(&log.root_digest)?,
    };
    Ok(Some(NativeRecoveryRecord {
        started,
        checkpoint,
        completed,
        reference,
        plan: id,
    }))
}

fn binding<R: Read + Seek>(
    reader: &TritModelReader<R>,
    level: &str,
    source: OriginalSourceId,
    evidence: EvidenceReference,
) -> Result<NativeRefinementParent, EvidenceError> {
    Ok(NativeRefinementParent {
        package: PackageId::from_bytes(*reader.package_id().as_bytes()),
        level: level.into(),
        model: reader
            .model_id(level)
            .map_err(|_| invalid("parent level absent"))?,
        source,
        evidence,
    })
}
fn manifest_digest(manifest: &ModelManifest) -> Result<[u8; 32], EvidenceError> {
    let bytes = encode_model_manifest(manifest).map_err(|_| invalid("parent manifest encoding"))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}
fn decode<T: DeserializeOwned>(event: &EvidenceEnvelope) -> Result<T, EvidenceError> {
    if event.v != 1 {
        return Err(invalid("native event version"));
    }
    T::deserialize(&event.payload).map_err(|_| invalid("strict native payload"))
}
fn invalid(detail: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: NativeRefinementCompleted::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
