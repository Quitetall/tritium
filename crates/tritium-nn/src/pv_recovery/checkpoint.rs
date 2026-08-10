use tritium_cuda::CudaBackend;
use tritium_cuda::train::DevicePackedSaltWeight;
use tritium_train::{
    PvTernaryWeight, PvTuningConfig, PvTuningSession, RecoveryCampaignRun, RecoveryEvidenceDigest,
};

use super::identity::{
    package_parent_catalog_digest, plan_digest, pv_campaign_context, validate_model_parents,
};
use super::snapshot::pack_snapshot;
use super::wire::Reader;
use super::{
    DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS, DevicePvRecoveryError, DevicePvRecoveryParentContext,
    DevicePvRecoverySession,
};
use crate::training::TiedSwiGluTrainingModel;

const MAGIC: &[u8; 5] = b"TPVM2";
const VERSION: u8 = 2;
const CHECKSUM_BYTES: usize = 32;
const FIXED_BODY_BYTES: usize = 86;

impl<'backend, 'model> DevicePvRecoverySession<'backend, 'model> {
    /// Serialize one coherent model step, exact TPVR1 evidence, and every TPV1 state.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, DevicePvRecoveryError> {
        self.ensure_usable()?;
        if self.campaign.is_some() {
            return Err(DevicePvRecoveryError::Checkpoint(
                "active campaign requires campaign_checkpoint_bytes".into(),
            ));
        }
        let capacity = completed_model_checkpoint_len(&self.tuning)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            DevicePvRecoveryError::Checkpoint("checkpoint allocation failed".into())
        })?;
        self.visit_checkpoint_body(|chunk| bytes.extend_from_slice(chunk))?;
        let checksum = blake3::hash(&bytes);
        bytes.extend_from_slice(checksum.as_bytes());
        debug_assert_eq!(bytes.len(), capacity);
        Ok(bytes)
    }

    /// Hash canonical TPVM2 bytes while retaining at most one nested TPV1 encoding.
    pub(super) fn checkpoint_digest(&self) -> Result<[u8; 32], DevicePvRecoveryError> {
        let mut checksum_hasher = blake3::Hasher::new();
        let mut artifact_hasher = blake3::Hasher::new();
        self.visit_checkpoint_body(|chunk| {
            checksum_hasher.update(chunk);
            artifact_hasher.update(chunk);
        })?;
        let checksum = checksum_hasher.finalize();
        artifact_hasher.update(checksum.as_bytes());
        Ok(*artifact_hasher.finalize().as_bytes())
    }

    fn visit_checkpoint_body(
        &self,
        mut visit: impl FnMut(&[u8]),
    ) -> Result<(), DevicePvRecoveryError> {
        self.ensure_usable()?;
        if self.campaign.is_some() {
            return Err(DevicePvRecoveryError::Checkpoint(
                "active campaign requires campaign_checkpoint_bytes".into(),
            ));
        }
        visit(MAGIC);
        visit(&[VERSION]);
        visit(&self.plan_digest);
        visit(&self.completed_step.to_le_bytes());
        visit(&checkpoint_step_receipt_digest(self)?);
        let tensor_count = u64::try_from(self.tuning.len()).map_err(|_| checkpoint_size_error())?;
        visit(&tensor_count.to_le_bytes());
        for tuning in &self.tuning {
            let checkpoint = tuning.checkpoint_bytes()?;
            let length = u64::try_from(checkpoint.len()).map_err(|_| checkpoint_size_error())?;
            visit(&length.to_le_bytes());
            visit(&checkpoint);
        }
        Ok(())
    }

    /// Resume only from exact model plan, parent set, step, and strict TPV1 states.
    pub fn resume(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_with_gradient_block_elements(
            backend,
            model,
            parents,
            config,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
            bytes,
        )
    }

    /// Resume a completed model checkpoint with an explicit host-gradient block cap.
    pub fn resume_with_gradient_block_elements(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        max_host_gradient_block_elements: usize,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            max_host_gradient_block_elements,
            None,
            None,
            bytes,
        )
    }

    /// Resume only when checkpoint and exact session plan name the same campaign.
    pub fn resume_for_recovery_campaign(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_for_recovery_campaign_with_gradient_block_elements(
            backend,
            model,
            parents,
            config,
            campaign,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
            bytes,
        )
    }

    /// Resume a campaign-bound checkpoint with an explicit host-gradient bound.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_for_recovery_campaign_with_gradient_block_elements(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        max_host_gradient_block_elements: usize,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            max_host_gradient_block_elements,
            Some(pv_campaign_context(campaign)?),
            None,
            bytes,
        )
    }

    /// Resume a campaign checkpoint only under its exact package-parent lineage.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_for_recovery_campaign_with_parent_context(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        parent_context: DevicePvRecoveryParentContext,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
            Some(pv_campaign_context(campaign)?),
            Some(parent_context),
            bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn resume_with_recovery_campaign_context(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        max_host_gradient_block_elements: usize,
        recovery_campaign_context: Option<RecoveryEvidenceDigest>,
        parent_context: Option<DevicePvRecoveryParentContext>,
        bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        if max_host_gradient_block_elements == 0 {
            return Err(DevicePvRecoveryError::InvalidInput(
                "host gradient block size must be non-zero".into(),
            ));
        }
        validate_model_parents(model, &parents)?;
        let parent_catalog_digest = parent_context
            .map(|_| package_parent_catalog_digest(model, &parents))
            .transpose()?;
        let expected_plan = plan_digest(
            model,
            &parents,
            config,
            recovery_campaign_context,
            parent_context,
            parent_catalog_digest,
        );
        let body_len = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or_else(|| checkpoint_error("checkpoint is truncated"))?;
        let (body, checksum) = bytes.split_at(body_len);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(checkpoint_error("checkpoint checksum mismatch"));
        }
        let mut reader = Reader::new(body);
        if reader.take(MAGIC.len())? != MAGIC || reader.u8()? != VERSION {
            return Err(checkpoint_error("bad checkpoint magic or version"));
        }
        if reader.take(32)? != expected_plan {
            return Err(checkpoint_error("checkpoint plan identity mismatch"));
        }
        let completed_step = reader.u64()?;
        let last_step_receipt_digest = parse_step_receipt_digest(&mut reader, completed_step)?;
        let count = reader.usize()?;
        if count != parents.len() {
            return Err(checkpoint_error("checkpoint parameter count mismatch"));
        }
        let tuning = resume_tensors(&mut reader, parents, config, completed_step)?;
        if reader.remaining() != 0 {
            return Err(checkpoint_error("checkpoint has trailing bytes"));
        }
        let packed = tuning
            .iter()
            .map(
                |session| -> Result<DevicePackedSaltWeight, DevicePvRecoveryError> {
                    let snapshot = pack_snapshot(session.weight())?;
                    Ok(DevicePackedSaltWeight::from_snapshot(backend, &snapshot)?)
                },
            )
            .collect::<Result<Vec<_>, DevicePvRecoveryError>>()?;
        Ok(Self {
            backend,
            model,
            plan_digest: expected_plan,
            recovery_campaign_context,
            parent_context,
            parent_catalog_digest,
            tuning,
            packed,
            max_host_gradient_block_elements,
            completed_step,
            last_step_receipt_digest,
            checkpoint_source_digest: Some(*blake3::hash(bytes).as_bytes()),
            campaign: None,
            poisoned: false,
        })
    }
}

fn checkpoint_step_receipt_digest(
    session: &DevicePvRecoverySession<'_, '_>,
) -> Result<[u8; 32], DevicePvRecoveryError> {
    match (session.completed_step, session.last_step_receipt_digest) {
        (0, None) => Ok([0; 32]),
        (0, Some(_)) => Err(checkpoint_error(
            "initial checkpoint unexpectedly carries a step receipt digest",
        )),
        (_, Some(digest)) if digest != [0; 32] => Ok(digest),
        (_, _) => Err(checkpoint_error(
            "completed checkpoint is missing its exact step receipt digest",
        )),
    }
}

fn parse_step_receipt_digest(
    reader: &mut Reader<'_>,
    completed_step: u64,
) -> Result<Option<[u8; 32]>, DevicePvRecoveryError> {
    let digest = reader.array()?;
    match completed_step {
        0 if digest == [0; 32] => Ok(None),
        0 => Err(checkpoint_error(
            "initial checkpoint carries a nonzero step receipt digest",
        )),
        _ if digest == [0; 32] => Err(checkpoint_error(
            "completed checkpoint is missing its step receipt digest",
        )),
        _ => Ok(Some(digest)),
    }
}

pub(super) fn completed_model_checkpoint_len(
    tuning: &[PvTuningSession],
) -> Result<usize, DevicePvRecoveryError> {
    let mut total = FIXED_BODY_BYTES
        .checked_add(CHECKSUM_BYTES)
        .ok_or_else(checkpoint_size_error)?;
    for session in tuning {
        if session.blockwise_cursor().is_some() {
            return Err(checkpoint_error(
                "in-flight blockwise step requires a campaign checkpoint",
            ));
        }
        let encoded = session.size_ledger()?.serialized_checkpoint_bytes();
        total = total
            .checked_add(8)
            .and_then(|value| value.checked_add(encoded))
            .ok_or_else(checkpoint_size_error)?;
    }
    Ok(total)
}

fn resume_tensors(
    reader: &mut Reader<'_>,
    parents: Vec<PvTernaryWeight>,
    config: PvTuningConfig,
    completed_step: u64,
) -> Result<Vec<PvTuningSession>, DevicePvRecoveryError> {
    let mut tuning = Vec::new();
    tuning
        .try_reserve_exact(parents.len())
        .map_err(|_| DevicePvRecoveryError::Checkpoint("session allocation failed".into()))?;
    for parent in parents {
        let length = reader.usize()?;
        let checkpoint = reader.take(length)?;
        let session = PvTuningSession::resume(parent, config, checkpoint)?;
        if session.completed_step() != completed_step {
            return Err(checkpoint_error("tensor checkpoint step mismatch"));
        }
        tuning.push(session);
    }
    Ok(tuning)
}

fn checkpoint_size_error() -> DevicePvRecoveryError {
    checkpoint_error("checkpoint size overflows host range")
}

fn checkpoint_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Checkpoint(reason.into())
}
