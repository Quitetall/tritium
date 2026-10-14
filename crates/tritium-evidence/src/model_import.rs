//! Strict recorded import bindings, not independently observed source truth.
use super::{
    EventRegistration, EvidenceError, EvidenceEvent, VerifiedLog, original_source_id, verify_log,
};
use tritium_schema::{LegacyModelImportPrepared, LevelFit, TensorStorage};

impl EvidenceEvent for LegacyModelImportPrepared {
    const SCHEMA_ID: &'static str = "tritium.migrate.legacy_model_import_prepared";
    const VERSION: u16 = 1;
}

/// Registration for prepared original legacy/companion/Imported manifest bindings.
pub const LEGACY_MODEL_IMPORT_PREPARED_REGISTRATION: EventRegistration = EventRegistration {
    schema: LegacyModelImportPrepared::SCHEMA_ID,
    version: LegacyModelImportPrepared::VERSION,
};

/// Immutable typed preparation view. This does not claim successful publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyModelImportView {
    observation: LegacyModelImportPrepared,
}
impl LegacyModelImportView {
    /// Exact prepared original bindings and Imported manifest.
    #[must_use]
    pub fn observation(&self) -> &LegacyModelImportPrepared {
        &self.observation
    }

    /// Recheck chains/root, strict payload/version and recorded binding consistency.
    /// Other registrations and independent source truth remain caller obligations.
    /// # Errors
    /// Corrupt log, unsupported version, non-Imported or mismatched recorded fields.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        log.events
            .iter()
            .filter(|e| e.schema == LegacyModelImportPrepared::SCHEMA_ID)
            .map(|event| {
                if event.v != LegacyModelImportPrepared::VERSION {
                    return Err(invalid("import event version"));
                }
                let observation: LegacyModelImportPrepared =
                    serde::Deserialize::deserialize(&event.payload)
                        .map_err(|_| invalid("import payload"))?;
                validate(&observation)?;
                Ok(Self { observation })
            })
            .collect()
    }
}
fn validate(value: &LegacyModelImportPrepared) -> Result<(), EvidenceError> {
    let manifest = &value.manifest;
    manifest
        .validate()
        .map_err(|_| invalid("import manifest"))?;
    if manifest.levels.len() != 1
        || manifest.levels[0].fit != LevelFit::Imported
        || !manifest.shards.is_empty()
        || manifest.legacy_ids.len() != 1
        || manifest.legacy_ids[0].id != value.legacy_id
        || manifest.legacy_ids[0].level != manifest.levels[0].name
    {
        return Err(invalid("import lineage"));
    }
    let original = original_source_id(
        &manifest.architecture,
        &value.companion_tensors,
        &value.companion_assets,
    )
    .map_err(|_| invalid("companion inventory"))?;
    if original != value.companion_source {
        return Err(invalid("companion identity"));
    }
    let dense: Vec<_> = manifest.levels[0]
        .tensors
        .iter()
        .filter(|t| matches!(t.storage, TensorStorage::Dense { .. }))
        .collect();
    if dense.len() != value.companion_tensors.len()
        || manifest.assets.len() != value.companion_assets.len()
    {
        return Err(invalid("companion completeness"));
    }
    for (actual, expected) in dense.iter().zip(&value.companion_tensors) {
        let TensorStorage::Dense { shape, precision } = &actual.storage else {
            return Err(invalid("dense storage"));
        };
        if actual.name != expected.source.name
            || actual.blob.id != expected.content
            || actual.blob.shard.is_some()
            || *shape != expected.source.shape
            || *precision != expected.source.precision
        {
            return Err(invalid("dense companion binding"));
        }
    }
    for (actual, expected) in manifest.assets.iter().zip(&value.companion_assets) {
        if actual.name != expected.name
            || actual.blob.id != expected.content
            || actual.blob.shard.is_some()
        {
            return Err(invalid("asset companion binding"));
        }
    }
    if dense.len() == manifest.levels[0].tensors.len() {
        return Err(invalid("no legacy additive tensors"));
    }
    Ok(())
}
fn invalid(detail: &'static str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: LegacyModelImportPrepared::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
