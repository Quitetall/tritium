//! Shared profile-checked record mechanics, not numerical or quality proof.
use crate::native_refinement::{NativeRecoveryRecord, fold_native_recovery};
use crate::{
    EvidenceError, EvidenceEvent, NativeRefinementParentView, NativeRefinementReceiptView,
    VerifiedLog, verify_log,
};
use serde::de::DeserializeOwned;
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
use tritium_schema::*;

pub(crate) trait ReplayRecord: EvidenceEvent + DeserializeOwned {
    const PROFILE_ERROR: &'static str;
    fn accepts(recipe: &NativeRecoveryRecipe) -> bool;
    fn binding(&self) -> (NativeRefinementPlanId, BlobId, u64, u64, u64);
}
macro_rules! profile {
    ($ty:ty,$recipe:pat,$message:literal) => {
        impl ReplayRecord for $ty {
            const PROFILE_ERROR: &'static str = $message;
            fn accepts(recipe: &NativeRecoveryRecipe) -> bool {
                matches!(recipe, $recipe)
            }
            fn binding(&self) -> (NativeRefinementPlanId, BlobId, u64, u64, u64) {
                (
                    self.plan,
                    self.checkpoint,
                    self.checkpoint_bytes,
                    self.completed_updates,
                    self.journal_bytes,
                )
            }
        }
    };
}
profile!(
    NativeScaleReplayObserved,
    NativeRecoveryRecipe::ScaleOnly { .. },
    "standalone child profile requires ScaleOnly"
);
profile!(
    NativePvReplayObserved,
    NativeRecoveryRecipe::AlternatingPv { .. },
    "standalone child profile requires AlternatingPv"
);

pub(crate) fn fold<E: ReplayRecord, R: Read + Seek>(
    log: &VerifiedLog,
    parent_reader: &mut TritModelReader<R>,
    parent: &NativeRefinementParentView,
    requested: &NativeRefinementPlan,
) -> Result<Vec<(NativeRefinementReceiptView, E)>, EvidenceError> {
    verify_log(log)?;
    if !E::accepts(&requested.recipe) {
        return Err(invalid::<E>(E::PROFILE_ERROR));
    }
    let mut views = NativeRefinementReceiptView::fold(log, parent_reader, parent, requested)?;
    if views.is_empty() {
        return Ok(Vec::new());
    }
    if views.len() != 1 {
        return Err(invalid::<E>("expected exactly one native refinement"));
    }
    let refinement = views.remove(0);
    let replay = replay_for::<E>(log, refinement.reference(), refinement.checkpoint())?;
    Ok(vec![(refinement, replay)])
}

pub(crate) fn verify_manifest<E: ReplayRecord, R: Read + Seek>(
    manifest: &ModelManifest,
    child: &mut TritModelReader<R>,
) -> Result<(), EvidenceError> {
    if child.manifest() != manifest {
        return Err(invalid::<E>(
            "native child manifest differs from original record",
        ));
    }
    child
        .validate_complete_payload_inventory()
        .map_err(|_| invalid::<E>("native child payload inventory differs"))?;
    child
        .verify()
        .map_err(|_| invalid::<E>("native child source changed"))?;
    Ok(())
}

pub(crate) fn from_child<E: ReplayRecord, R: Read + Seek>(
    log: &VerifiedLog,
    child: &mut TritModelReader<R>,
) -> Result<(NativeRecoveryRecord, E, tritium_format::PackageId), EvidenceError> {
    verify_log(log)?;
    let first = log
        .events
        .iter()
        .find(|e| e.schema == "tritium.refine.native_started")
        .ok_or_else(|| invalid::<E>("native child start absent"))?;
    if first.v != 1 {
        return Err(invalid::<E>("native child start version"));
    }
    let started: NativeRefinementStarted = serde_json::from_value(first.payload.clone())
        .map_err(|_| invalid::<E>("native child start payload"))?;
    if !E::accepts(&started.plan.recipe) {
        return Err(invalid::<E>(E::PROFILE_ERROR));
    }
    let recovery = fold_native_recovery(log, &started.plan)?
        .ok_or_else(|| invalid::<E>("native child lifecycle incomplete"))?;
    let replay = replay_for::<E>(log, &recovery.reference, &recovery.checkpoint)?;
    let manifest = child.manifest();
    if manifest.schema != (SchemaId { major: 1, minor: 1 }) || manifest.levels.len() != 1 {
        return Err(invalid::<E>(
            "standalone native child requires schema1.1 one level",
        ));
    }
    let level = &manifest.levels[0];
    let LevelFit::Refined { ancestry } = &level.fit else {
        return Err(invalid::<E>("standalone native child is not Refined"));
    };
    if level.name != recovery.started.plan.child_level
        || ancestry.parent != recovery.started.plan.parent
        || ancestry.plan != recovery.plan
        || ancestry.evidence != recovery.reference
    {
        return Err(invalid::<E>("standalone native child ancestry differs"));
    }
    if level
        .tensors
        .iter()
        .map(|t| (&t.name, t.blob.id))
        .ne(recovery
            .completed
            .tensors
            .iter()
            .map(|t| (&t.name, t.content)))
        || recovery.completed.validation.len() != recovery.started.plan.validation.len()
        || recovery
            .completed
            .validation
            .iter()
            .zip(&recovery.started.plan.validation)
            .any(|(a, b)| {
                a.content != b.content || !a.parent_loss.is_finite() || !a.child_loss.is_finite()
            })
    {
        return Err(invalid::<E>(
            "standalone native child output/validation differs",
        ));
    }
    for owner in &recovery.checkpoint.owners {
        let meta = child
            .tensor_meta(&level.name, &owner.name)
            .map_err(|_| invalid::<E>("standalone child selected owner absent"))?;
        if !matches!(meta.storage, TensorStorage::Additive { .. }) || meta.blob.id != owner.content
        {
            return Err(invalid::<E>("standalone child checkpoint owner differs"));
        }
    }
    child
        .validate_complete_payload_inventory()
        .map_err(|_| invalid::<E>("standalone child inventory differs"))?;
    child
        .verify()
        .map_err(|_| invalid::<E>("standalone child source changed"))?;
    Ok((recovery, replay, child.package_id()))
}

pub(crate) fn verify_package<E: ReplayRecord, R: Read + Seek>(
    package: tritium_format::PackageId,
    child: &mut TritModelReader<R>,
) -> Result<(), EvidenceError> {
    if child.package_id() != package {
        return Err(invalid::<E>(
            "standalone native child package differs from admitted record",
        ));
    }
    child
        .validate_complete_payload_inventory()
        .map_err(|_| invalid::<E>("standalone native child inventory changed"))?;
    child
        .verify()
        .map_err(|_| invalid::<E>("standalone native child bytes changed"))?;
    Ok(())
}

fn replay_for<E: ReplayRecord>(
    log: &VerifiedLog,
    reference: &EvidenceReference,
    checkpoint: &NativeRefinementCheckpointed,
) -> Result<E, EvidenceError> {
    let records: Vec<_> = log
        .events
        .iter()
        .filter(|e| e.schema == E::SCHEMA_ID)
        .collect();
    if records.len() != 1
        || log
            .events
            .iter()
            .any(|e| !e.schema.starts_with("tritium.refine.native_") && e.schema != E::SCHEMA_ID)
    {
        return Err(invalid::<E>(
            "expected exactly one native replay observation",
        ));
    }
    let record = records[0];
    let start = log
        .events
        .iter()
        .find(|e| e.span == reference.span && e.seq == 0)
        .ok_or_else(|| invalid::<E>("missing native start"))?;
    let complete = log
        .events
        .iter()
        .find(|e| e.schema == "tritium.refine.native_completed" && e.span == reference.span)
        .ok_or_else(|| invalid::<E>("missing native completion"))?;
    if record.v != 1
        || record.run != reference.run
        || record.parent.as_deref() != Some(reference.span.as_str())
        || record.span == reference.span
        || record.seq != 0
        || record.t <= start.t
        || record.t >= complete.t
    {
        return Err(invalid::<E>("native replay span/version/order differs"));
    }
    let replay: E = serde_json::from_value(record.payload.clone())
        .map_err(|_| invalid::<E>("invalid native replay observation"))?;
    let (plan, content, bytes, updates, journal_bytes) = replay.binding();
    if plan != checkpoint.plan
        || content != checkpoint.checkpoint
        || bytes != checkpoint.checkpoint_bytes
        || updates != checkpoint.completed_updates
        || journal_bytes == 0
        || journal_bytes > ((1u64 << 53) - 1)
    {
        return Err(invalid::<E>(
            "native replay checkpoint/cursor binding differs",
        ));
    }
    Ok(replay)
}
fn invalid<E: ReplayRecord>(message: &str) -> EvidenceError {
    EvidenceError::InvalidPayload {
        schema: E::SCHEMA_ID.into(),
        detail: message.into(),
    }
}
