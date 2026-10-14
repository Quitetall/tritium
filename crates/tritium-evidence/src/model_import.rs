//! Strict recorded import bindings, not independently observed source truth.
use super::{
    EventRegistration, EvidenceError, EvidenceEvent, VerifiedLog, original_source_id, verify_log,
};
use tritium_schema::{
    LegacyModelImportBoundPrepared, LegacyModelImportPrepared, LevelFit, ModelManifest,
    NamedAssetDigest, OriginalSourceId, QuantSourceBinding, TensorStorage,
};

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
    validate_bindings(
        &value.legacy_id,
        value.companion_source,
        &value.companion_tensors,
        &value.companion_assets,
        &value.manifest,
        LegacyModelImportPrepared::SCHEMA_ID,
    )
}
fn validate_bindings(
    legacy_id: &str,
    companion_source: OriginalSourceId,
    companion_tensors: &[QuantSourceBinding],
    companion_assets: &[NamedAssetDigest],
    manifest: &ModelManifest,
    schema: &str,
) -> Result<(), EvidenceError> {
    let invalid = |detail: &'static str| EvidenceError::InvalidPayload {
        schema: schema.into(),
        detail: detail.into(),
    };
    manifest
        .validate()
        .map_err(|_| invalid("import manifest"))?;
    if manifest.levels.len() != 1
        || manifest.levels[0].fit != LevelFit::Imported
        || !manifest.shards.is_empty()
        || manifest.legacy_ids.len() != 1
        || manifest.legacy_ids[0].id != legacy_id
        || manifest.legacy_ids[0].level != manifest.levels[0].name
    {
        return Err(invalid("import lineage"));
    }
    let original = original_source_id(&manifest.architecture, companion_tensors, companion_assets)
        .map_err(|_| invalid("companion inventory"))?;
    if original != companion_source {
        return Err(invalid("companion identity"));
    }
    let dense: Vec<_> = manifest.levels[0]
        .tensors
        .iter()
        .filter(|t| matches!(t.storage, TensorStorage::Dense { .. }))
        .collect();
    if dense.len() != companion_tensors.len() || manifest.assets.len() != companion_assets.len() {
        return Err(invalid("companion completeness"));
    }
    for (actual, expected) in dense.iter().zip(companion_tensors) {
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
    for (actual, expected) in manifest.assets.iter().zip(companion_assets) {
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

impl EvidenceEvent for LegacyModelImportBoundPrepared {
    const SCHEMA_ID: &'static str = "tritium.migrate.legacy_model_import_bound_prepared";
    const VERSION: u16 = 1;
}
/// Registration for actual canonical-manifest-bound legacy import preparation.
pub const LEGACY_MODEL_IMPORT_BOUND_PREPARED_REGISTRATION: EventRegistration = EventRegistration {
    schema: LegacyModelImportBoundPrepared::SCHEMA_ID,
    version: LegacyModelImportBoundPrepared::VERSION,
};
/// Compact typed observation; no manifest/count copies or qualification verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyModelImportBoundView {
    observation: LegacyModelImportBoundPrepared,
}
impl LegacyModelImportBoundView {
    /// Bind actual resolved canonical metadata and observed companions.
    /// This is not an original-source proof or successful publication.
    /// # Errors
    /// Structural/lineage/companion refusal or actual canonical encoding failure.
    pub fn prepare(
        manifest: &ModelManifest,
        legacy_id: String,
        companion_source: OriginalSourceId,
        companion_tensors: Vec<QuantSourceBinding>,
        companion_assets: Vec<NamedAssetDigest>,
    ) -> Result<LegacyModelImportBoundPrepared, EvidenceError> {
        validate_bindings(
            &legacy_id,
            companion_source,
            &companion_tensors,
            &companion_assets,
            manifest,
            LegacyModelImportBoundPrepared::SCHEMA_ID,
        )?;
        let (manifest_bytes, manifest_digest) = manifest_identity(manifest)?;
        Ok(LegacyModelImportBoundPrepared {
            legacy_id,
            companion_source,
            companion_tensors,
            companion_assets,
            manifest_bytes,
            manifest_digest,
        })
    }
    /// Actual compact preparation, not a fabricated embedded observation.
    #[must_use]
    pub fn observation(&self) -> &LegacyModelImportBoundPrepared {
        &self.observation
    }

    /// Reverify chains/root and bind every compact event to the actual manifest.
    /// Full payload/model/source truth verification remains a separate obligation.
    /// # Errors
    /// Unsupported/invalid event, digest/length or original structural/binding refusal.
    pub fn fold(log: &VerifiedLog, manifest: &ModelManifest) -> Result<Vec<Self>, EvidenceError> {
        verify_log(log)?;
        if !log
            .events
            .iter()
            .any(|e| e.schema == LegacyModelImportBoundPrepared::SCHEMA_ID)
        {
            return Ok(Vec::new());
        }
        let identity = manifest_identity(manifest)?;
        log.events
            .iter()
            .filter(|e| e.schema == LegacyModelImportBoundPrepared::SCHEMA_ID)
            .map(|event| {
                let invalid = |detail: &'static str| EvidenceError::InvalidPayload {
                    schema: LegacyModelImportBoundPrepared::SCHEMA_ID.into(),
                    detail: detail.into(),
                };
                if event.v != LegacyModelImportBoundPrepared::VERSION {
                    return Err(invalid("bound import event version"));
                }
                let observation: LegacyModelImportBoundPrepared =
                    serde::Deserialize::deserialize(&event.payload)
                        .map_err(|_| invalid("strict bound import payload"))?;
                if (observation.manifest_bytes, observation.manifest_digest) != identity {
                    return Err(invalid("canonical manifest identity mismatch"));
                }
                validate_bindings(
                    &observation.legacy_id,
                    observation.companion_source,
                    &observation.companion_tensors,
                    &observation.companion_assets,
                    manifest,
                    LegacyModelImportBoundPrepared::SCHEMA_ID,
                )?;
                Ok(Self { observation })
            })
            .collect()
    }
}
fn manifest_identity(
    manifest: &ModelManifest,
) -> Result<(u64, tritium_schema::DigestBytes), EvidenceError> {
    let bytes = tritium_format::trit_model::encode_model_manifest(manifest).map_err(|_| {
        EvidenceError::InvalidPayload {
            schema: LegacyModelImportBoundPrepared::SCHEMA_ID.into(),
            detail: "canonical manifest encoding".into(),
        }
    })?;
    Ok((bytes.len() as u64, *blake3::hash(&bytes).as_bytes()))
}
fn invalid(detail: &'static str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: LegacyModelImportPrepared::SCHEMA_ID.into(),
        detail: detail.into(),
    }
}
