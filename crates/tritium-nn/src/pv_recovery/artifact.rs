use core::fmt;
use std::error::Error;

use tritium_train::{PvTernaryWeight, RecoveryCampaignRun, RecoveryEvidenceDigest};

use super::identity::pv_campaign_context;
use super::wire::Reader;
use super::{DevicePvRecoveryError, DevicePvRecoverySession, DevicePvRecoveryStepReceipt};

const MAGIC: &[u8; 5] = b"TPVA1";
const CHECKSUM_BYTES: usize = 32;
const FIXED_BODY_BYTES: usize = MAGIC.len() + 32 + 32 + 8;

/// Exact completed PV checkpoint plus checksum-bound step evidence.
///
/// This is a resumable optimization artifact, not a deployment package or a
/// complete release receipt. Its artifact digest can be passed to the frozen
/// recovery campaign only after the campaign's independent validation metric
/// producer has evaluated these exact checkpoint bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct DevicePvRecoveryCheckpointArtifact {
    checkpoint_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
    step_receipt: DevicePvRecoveryStepReceipt,
    artifact_digest: RecoveryEvidenceDigest,
    campaign_context_digest: RecoveryEvidenceDigest,
    evidence_digest: RecoveryEvidenceDigest,
}

/// Failure while visiting exact weights from one current TPVM2 artifact.
#[derive(Debug)]
pub enum DevicePvRecoveryWeightVisitError<E> {
    /// Artifact, session, campaign, or checkpoint state differs.
    Recovery(DevicePvRecoveryError),
    /// Caller rejected one exact parameter view.
    Visitor(E),
}

impl<E: fmt::Display> fmt::Display for DevicePvRecoveryWeightVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recovery(error) => write!(formatter, "visit device PV checkpoint: {error}"),
            Self::Visitor(error) => write!(formatter, "visit device PV weight: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for DevicePvRecoveryWeightVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Recovery(error) => Some(error),
            Self::Visitor(error) => Some(error),
        }
    }
}

impl DevicePvRecoveryCheckpointArtifact {
    /// Reopen an exact TPVM2 checkpoint and its TPVA1 step-evidence manifest.
    pub fn reopen(
        session: &DevicePvRecoverySession<'_, '_>,
        campaign: &RecoveryCampaignRun,
        checkpoint_bytes: Vec<u8>,
        manifest_bytes: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        let expected_campaign_context = pv_campaign_context(campaign)?;
        if session.recovery_campaign_context != Some(expected_campaign_context) {
            return Err(campaign_error(
                "session is not bound to the artifact's recovery campaign",
            ));
        }
        session.ensure_usable()?;
        let body_len = manifest_bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or_else(|| artifact_error("artifact manifest is truncated"))?;
        let (body, checksum) = manifest_bytes.split_at(body_len);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(artifact_error("artifact manifest checksum mismatch"));
        }
        let evidence_digest =
            recovery_digest(checksum.try_into().map_err(|_| {
                artifact_error("artifact manifest checksum has unexpected length")
            })?)?;
        let mut reader = Reader::new(body);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(artifact_error("bad artifact manifest magic or version"));
        }
        let expected_artifact_digest = recovery_digest(reader.array()?)?;
        let campaign_context_digest = recovery_digest(reader.array()?)?;
        if campaign_context_digest != expected_campaign_context {
            return Err(campaign_error("artifact campaign context mismatch"));
        }
        let receipt_bytes = reader.blob()?;
        if reader.remaining() != 0 {
            return Err(artifact_error("artifact manifest has trailing bytes"));
        }
        let step_receipt = DevicePvRecoveryStepReceipt::from_canonical_bytes(receipt_bytes)?;
        let artifact_digest = recovery_digest(*blake3::hash(&checkpoint_bytes).as_bytes())?;
        if artifact_digest != expected_artifact_digest {
            return Err(artifact_error("checkpoint artifact identity mismatch"));
        }
        if session.checkpoint_source_digest != Some(artifact_digest.as_bytes()) {
            return Err(artifact_error(
                "checkpoint artifact was not used to resume the current session",
            ));
        }
        validate_current_state(session, &step_receipt, checkpoint_bytes.len())?;
        Ok(Self {
            checkpoint_bytes,
            manifest_bytes: manifest_bytes.to_vec(),
            step_receipt,
            artifact_digest,
            campaign_context_digest,
            evidence_digest,
        })
    }

    /// Exact resumable TPVM2 checkpoint bytes selected by the campaign.
    #[must_use]
    pub fn checkpoint_bytes(&self) -> &[u8] {
        &self.checkpoint_bytes
    }

    /// Canonical TPVA1 sidecar binding checkpoint content to TPVR1 step evidence.
    #[must_use]
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    /// Strictly decoded model-wide step receipt enclosed by the sidecar.
    #[must_use]
    pub const fn step_receipt(&self) -> &DevicePvRecoveryStepReceipt {
        &self.step_receipt
    }

    /// Content identity of the exact resumable checkpoint bytes.
    #[must_use]
    pub const fn artifact_digest(&self) -> RecoveryEvidenceDigest {
        self.artifact_digest
    }

    /// Immutable authorization context for the PV campaign selecting this artifact.
    #[must_use]
    pub const fn campaign_context_digest(&self) -> RecoveryEvidenceDigest {
        self.campaign_context_digest
    }

    /// Content identity of the checkpoint-to-step evidence binding.
    #[must_use]
    pub const fn evidence_digest(&self) -> RecoveryEvidenceDigest {
        self.evidence_digest
    }

    /// Visit every exact current parameter weight in canonical model order.
    ///
    /// This bounded source seam revalidates TPVM2/TPVA1 state before exposing
    /// borrowed weight views. It does not mint package, evaluation, promotion,
    /// or release evidence. Callback effects are nontransactional.
    pub fn try_visit_current_weights<E>(
        &self,
        session: &DevicePvRecoverySession<'_, '_>,
        mut visit: impl FnMut(&str, &PvTernaryWeight) -> Result<(), E>,
    ) -> Result<usize, DevicePvRecoveryWeightVisitError<E>> {
        if session.recovery_campaign_context != Some(self.campaign_context_digest) {
            return Err(DevicePvRecoveryWeightVisitError::Recovery(campaign_error(
                "artifact and session campaign contexts differ",
            )));
        }
        validate_current_state(session, &self.step_receipt, self.checkpoint_bytes.len())
            .map_err(DevicePvRecoveryWeightVisitError::Recovery)?;
        let checkpoint_digest = session
            .checkpoint_digest()
            .map_err(DevicePvRecoveryWeightVisitError::Recovery)?;
        if checkpoint_digest != self.artifact_digest.as_bytes()
            || blake3::hash(&self.checkpoint_bytes).as_bytes() != &self.artifact_digest.as_bytes()
            || session.model.parameters().len() != session.tuning.len()
        {
            return Err(DevicePvRecoveryWeightVisitError::Recovery(artifact_error(
                "artifact differs from current parameter checkpoint",
            )));
        }
        for (parameter, tuning) in session.model.parameters().iter().zip(&session.tuning) {
            visit(&parameter.name, tuning.weight())
                .map_err(DevicePvRecoveryWeightVisitError::Visitor)?;
        }
        Ok(session.tuning.len())
    }

    /// Visit every exact current parameter in caller-supplied canonical name order.
    ///
    /// `names` must contain each model parameter exactly once. This supports
    /// deterministic artifact formats whose canonical tensor order differs from
    /// graph execution order without cloning model-sized weights.
    pub fn try_visit_current_weights_in_order<E>(
        &self,
        session: &DevicePvRecoverySession<'_, '_>,
        names: &[&str],
        mut visit: impl FnMut(&str, &PvTernaryWeight) -> Result<(), E>,
    ) -> Result<usize, DevicePvRecoveryWeightVisitError<E>> {
        if session.recovery_campaign_context != Some(self.campaign_context_digest) {
            return Err(DevicePvRecoveryWeightVisitError::Recovery(campaign_error(
                "artifact and session campaign contexts differ",
            )));
        }
        validate_current_state(session, &self.step_receipt, self.checkpoint_bytes.len())
            .map_err(DevicePvRecoveryWeightVisitError::Recovery)?;
        let checkpoint_digest = session
            .checkpoint_digest()
            .map_err(DevicePvRecoveryWeightVisitError::Recovery)?;
        if checkpoint_digest != self.artifact_digest.as_bytes()
            || blake3::hash(&self.checkpoint_bytes).as_bytes() != &self.artifact_digest.as_bytes()
            || session.model.parameters().len() != session.tuning.len()
            || names.len() != session.tuning.len()
        {
            return Err(DevicePvRecoveryWeightVisitError::Recovery(artifact_error(
                "artifact differs from requested parameter catalog",
            )));
        }
        let mut visited = vec![false; session.tuning.len()];
        for &name in names {
            let index = session
                .model
                .parameters()
                .iter()
                .position(|parameter| parameter.name == name)
                .ok_or_else(|| {
                    DevicePvRecoveryWeightVisitError::Recovery(artifact_error(
                        "requested parameter catalog contains an unknown name",
                    ))
                })?;
            if core::mem::replace(&mut visited[index], true) {
                return Err(DevicePvRecoveryWeightVisitError::Recovery(artifact_error(
                    "requested parameter catalog contains a duplicate name",
                )));
            }
            visit(name, session.tuning[index].weight())
                .map_err(DevicePvRecoveryWeightVisitError::Visitor)?;
        }
        if visited.iter().any(|&seen| !seen) {
            return Err(DevicePvRecoveryWeightVisitError::Recovery(artifact_error(
                "requested parameter catalog omits a model parameter",
            )));
        }
        Ok(visited.len())
    }
}

impl DevicePvRecoverySession<'_, '_> {
    /// Seal current completed state only against its exact immediately preceding step.
    pub fn checkpoint_artifact(
        &self,
        receipt: &DevicePvRecoveryStepReceipt,
        campaign: &RecoveryCampaignRun,
    ) -> Result<DevicePvRecoveryCheckpointArtifact, DevicePvRecoveryError> {
        let campaign_context_digest = pv_campaign_context(campaign)?;
        if self.recovery_campaign_context != Some(campaign_context_digest) {
            return Err(campaign_error(
                "session is not bound to the artifact's recovery campaign",
            ));
        }
        self.ensure_usable()?;
        let checkpoint_bytes = self.checkpoint_bytes()?;
        validate_current_state(self, receipt, checkpoint_bytes.len())?;
        let receipt_bytes = receipt.canonical_bytes()?;
        let artifact_digest = recovery_digest(*blake3::hash(&checkpoint_bytes).as_bytes())?;
        let receipt_len = u64::try_from(receipt_bytes.len())
            .map_err(|_| artifact_error("step receipt exceeds artifact wire range"))?;
        let capacity = FIXED_BODY_BYTES
            .checked_add(receipt_bytes.len())
            .and_then(|value| value.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| artifact_error("artifact manifest size overflows host range"))?;
        let mut manifest_bytes = Vec::new();
        manifest_bytes
            .try_reserve_exact(capacity)
            .map_err(|_| artifact_error("artifact manifest allocation failed"))?;
        manifest_bytes.extend_from_slice(MAGIC);
        manifest_bytes.extend_from_slice(&artifact_digest.as_bytes());
        manifest_bytes.extend_from_slice(&campaign_context_digest.as_bytes());
        manifest_bytes.extend_from_slice(&receipt_len.to_le_bytes());
        manifest_bytes.extend_from_slice(&receipt_bytes);
        let checksum = blake3::hash(&manifest_bytes);
        manifest_bytes.extend_from_slice(checksum.as_bytes());
        debug_assert_eq!(manifest_bytes.len(), capacity);
        Ok(DevicePvRecoveryCheckpointArtifact {
            checkpoint_bytes,
            manifest_bytes,
            step_receipt: receipt.clone(),
            artifact_digest,
            campaign_context_digest,
            evidence_digest: recovery_digest(*checksum.as_bytes())?,
        })
    }
}

fn validate_current_state(
    session: &DevicePvRecoverySession<'_, '_>,
    receipt: &DevicePvRecoveryStepReceipt,
    checkpoint_len: usize,
) -> Result<(), DevicePvRecoveryError> {
    let receipt_digest = *blake3::hash(&receipt.canonical_bytes()?).as_bytes();
    if receipt.plan_digest() != session.plan_digest
        || receipt.optimizer_step() != session.completed_step
        || receipt.tensor_receipts().len() != session.tuning.len()
        || receipt.serialized_checkpoint_bytes() != checkpoint_len
        || session.last_step_receipt_digest != Some(receipt_digest)
    {
        return Err(artifact_error(
            "step receipt does not describe current completed checkpoint",
        ));
    }
    for (tensor, tuning) in receipt.tensor_receipts().iter().zip(&session.tuning) {
        if tensor.representation_digest() != tuning.weight().digest() {
            return Err(artifact_error(
                "step receipt representation differs from current checkpoint",
            ));
        }
    }
    Ok(())
}

fn recovery_digest(bytes: [u8; 32]) -> Result<RecoveryEvidenceDigest, DevicePvRecoveryError> {
    RecoveryEvidenceDigest::new(bytes)
        .map_err(|error| artifact_error(&format!("invalid recovery evidence digest: {error}")))
}

fn artifact_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Checkpoint(reason.into())
}

fn campaign_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Campaign(reason.into())
}
