use tritium_cuda::{CudaBackend, train::DeviceTensor};
use tritium_format::salt_v2_master::{
    SaltV2FitConstraint, SaltV2ParentCatalogLineageHasher, SaltV2ParentTensorLineageHasher,
};
use tritium_format::salt_v2_package::SALT_V2_ALLOCATION_TILE_SIZE;
use tritium_train::{
    PvTernaryWeight, PvTuningConfig, PvTuningSession, RecoveryCampaignRun, RecoveryEvidenceDigest,
    RecoveryTrack,
};

use super::{DevicePvRecoveryError, DevicePvRecoveryParentContext};
use crate::recovery::recovery_model_digest;
use crate::training::TiedSwiGluTrainingModel;

pub(super) fn validate_model_parents(
    model: &TiedSwiGluTrainingModel,
    parents: &[PvTernaryWeight],
) -> Result<(), DevicePvRecoveryError> {
    if model.parameters().is_empty() {
        return Err(DevicePvRecoveryError::InvalidInput(
            "model has no refinable parameters".into(),
        ));
    }
    if model
        .parameters()
        .iter()
        .any(|parameter| !parameter.master.is_empty())
    {
        return Err(DevicePvRecoveryError::InvalidInput(
            "drain dense masters with take_parameter_masters before hard-PV recovery".into(),
        ));
    }
    if parents.len() != model.parameters().len() {
        return Err(DevicePvRecoveryError::InvalidInput(format!(
            "received {} PV parents for {} model parameters",
            parents.len(),
            model.parameters().len()
        )));
    }
    for (parameter, parent) in model.parameters().iter().zip(parents) {
        if (parameter.rows, parameter.cols) != (parent.rows(), parent.cols()) {
            return Err(DevicePvRecoveryError::InvalidInput(format!(
                "PV parent for {} is [{}, {}], expected [{}, {}]",
                parameter.name,
                parent.rows(),
                parent.cols(),
                parameter.rows,
                parameter.cols
            )));
        }
    }
    Ok(())
}

pub(super) fn plan_digest(
    model: &TiedSwiGluTrainingModel,
    parents: &[PvTernaryWeight],
    config: PvTuningConfig,
    recovery_campaign_context: Option<RecoveryEvidenceDigest>,
    parent_context: Option<DevicePvRecoveryParentContext>,
    parent_catalog_digest: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-recovery-plan.v4\0");
    hash.update(recovery_model_digest(model).as_bytes());
    hash.update(&config.recipe_digest());
    match recovery_campaign_context {
        Some(digest) => {
            hash.update(&[1]);
            hash.update(&digest.as_bytes());
        }
        None => {
            hash.update(&[0]);
            hash.update(&[0; 32]);
        }
    }
    match parent_context {
        Some(context) => {
            hash.update(&[1]);
            hash.update(&context.as_bytes());
        }
        None => {
            hash.update(&[0]);
            hash.update(&[0; 32]);
        }
    }
    match parent_catalog_digest {
        Some(digest) => {
            hash.update(&[1]);
            hash.update(&digest);
        }
        None => {
            hash.update(&[0]);
            hash.update(&[0; 32]);
        }
    }
    for parent in parents {
        hash.update(&parent.digest());
    }
    *hash.finalize().as_bytes()
}

pub(super) fn package_parent_catalog_digest(
    model: &TiedSwiGluTrainingModel,
    parents: &[PvTernaryWeight],
) -> Result<[u8; 32], DevicePvRecoveryError> {
    let mut entries = model.parameters().iter().zip(parents).collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| left.name.cmp(&right.name));
    let mut catalog = SaltV2ParentCatalogLineageHasher::new();
    for (parameter, parent) in entries {
        if !parent.cols().is_multiple_of(parent.group_size()) {
            return Err(parent_lineage_error(format!(
                "PV package parent {} has row-ragged scale groups",
                parameter.name
            )));
        }
        let shape = [
            u64::try_from(parent.rows())
                .map_err(|_| parent_lineage_error("PV parent row count exceeds wire range"))?,
            u64::try_from(parent.cols())
                .map_err(|_| parent_lineage_error("PV parent column count exceeds wire range"))?,
        ];
        let constraint = match parent.structure() {
            tritium_train::PvTernaryStructure::Dense => SaltV2FitConstraint::Dense,
            tritium_train::PvTernaryStructure::S34 => SaltV2FitConstraint::S34,
        };
        let plane_count = u8::try_from(parent.planes().len())
            .map_err(|_| parent_lineage_error("PV parent plane count exceeds wire range"))?;
        let mut tensor = SaltV2ParentTensorLineageHasher::new(
            &shape,
            parent.group_size(),
            constraint,
            plane_count,
        )
        .map_err(|error| parent_lineage_error(format!("{}: {error}", parameter.name)))?;
        let total = parent
            .rows()
            .checked_mul(parent.cols())
            .ok_or_else(|| parent_lineage_error("PV parent coefficient count overflows"))?;
        for start in (0..total).step_by(SALT_V2_ALLOCATION_TILE_SIZE) {
            let end = start
                .checked_add(SALT_V2_ALLOCATION_TILE_SIZE)
                .map(|end| end.min(total))
                .ok_or_else(|| parent_lineage_error("PV parent tile end overflows"))?;
            let scale_start = start / parent.group_size();
            let scale_end = end.div_ceil(parent.group_size());
            tensor
                .push_raw_tile(parent.planes().iter().map(|plane| {
                    (
                        &plane.trits()[start..end],
                        &plane.scales()[scale_start..scale_end],
                    )
                }))
                .map_err(|error| parent_lineage_error(format!("{}: {error}", parameter.name)))?;
        }
        let tensor_digest = tensor
            .finish()
            .map_err(|error| parent_lineage_error(format!("{}: {error}", parameter.name)))?;
        catalog
            .push(&parameter.name, tensor_digest)
            .map_err(|error| parent_lineage_error(format!("{}: {error}", parameter.name)))?;
    }
    catalog.finish().map_err(parent_lineage_error)
}

fn parent_lineage_error(reason: impl ToString) -> DevicePvRecoveryError {
    DevicePvRecoveryError::InvalidInput(format!(
        "invalid PV package-parent lineage: {}",
        reason.to_string()
    ))
}

pub(super) fn pv_campaign_context(
    campaign: &RecoveryCampaignRun,
) -> Result<RecoveryEvidenceDigest, DevicePvRecoveryError> {
    if campaign.track() != RecoveryTrack::Pv {
        return Err(DevicePvRecoveryError::Campaign(
            "device PV sessions require a PV recovery campaign".into(),
        ));
    }
    Ok(campaign.evidence_context_digest())
}

pub(super) fn representation_digest(
    plan_digest: [u8; 32],
    optimizer_step: u64,
    tensor_digests: impl IntoIterator<Item = [u8; 32]>,
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-recovery-state.v1\0");
    hash.update(&plan_digest);
    hash.update(&optimizer_step.to_le_bytes());
    for digest in tensor_digests {
        hash.update(&digest);
    }
    *hash.finalize().as_bytes()
}

pub(super) fn session_state_digest(
    plan_digest: [u8; 32],
    completed_step: u64,
    sessions: &[PvTuningSession],
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-source-state.v1\0");
    hash.update(&plan_digest);
    hash.update(&completed_step.to_le_bytes());
    for session in sessions {
        hash.update(&session.state_digest());
    }
    *hash.finalize().as_bytes()
}

pub(super) fn batch_digest(
    backend: &CudaBackend,
    plan_digest: [u8; 32],
    optimizer_step: u64,
    tokens: &[i32],
    target: &DeviceTensor,
) -> Result<[u8; 32], DevicePvRecoveryError> {
    let target = target.download(backend)?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-campaign-batch.v1\0");
    hash.update(&plan_digest);
    hash.update(&optimizer_step.to_le_bytes());
    hash.update(
        &u64::try_from(tokens.len())
            .map_err(|_| identity_error("token count exceeds campaign wire range"))?
            .to_le_bytes(),
    );
    for token in tokens {
        hash.update(&token.to_le_bytes());
    }
    hash.update(
        &u64::try_from(target.len())
            .map_err(|_| identity_error("target count exceeds campaign wire range"))?
            .to_le_bytes(),
    );
    for value in target {
        hash.update(&value.to_bits().to_le_bytes());
    }
    Ok(*hash.finalize().as_bytes())
}

pub(super) fn evidence_digest(
    plan_digest: [u8; 32],
    source_state_digest: [u8; 32],
    batch_digest: [u8; 32],
    optimizer_step: u64,
    representation_digest: [u8; 32],
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-step-evidence.v1\0");
    hash.update(&plan_digest);
    hash.update(&source_state_digest);
    hash.update(&batch_digest);
    hash.update(&optimizer_step.to_le_bytes());
    hash.update(&representation_digest);
    *hash.finalize().as_bytes()
}

fn identity_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::InvalidInput(reason.into())
}
