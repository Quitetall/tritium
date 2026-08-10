use tritium_cuda::train::{DeviceTensor, GradientStreamReport};
use tritium_spec::BackendError;
use tritium_train::{
    PvStepReceipt, PvTernaryWeight, PvTuningConfig, PvTuningSession, RecoveryCampaignRun,
    RecoveryEvidenceDigest,
};

use super::identity::{
    batch_digest, package_parent_catalog_digest, plan_digest, pv_campaign_context,
    session_state_digest, validate_model_parents,
};
use super::receipt::canonical_receipt_digest;
use super::session::{ModelGradientStream, PvStepIdentity, build_receipt};
use super::wire::Reader;
use super::{
    DevicePvRecoveryCampaignState, DevicePvRecoveryError, DevicePvRecoveryParentContext,
    DevicePvRecoverySession, DevicePvRecoveryStepReceipt,
};
use crate::training::TiedSwiGluTrainingModel;

const MAGIC: &[u8; 5] = b"TPVC1";
const CHECKSUM_BYTES: usize = 32;
const UNTOUCHED: u8 = 0;
const COMPLETED: u8 = 1;
const IN_FLIGHT: u8 = 2;

impl<'backend, 'model> DevicePvRecoverySession<'backend, 'model> {
    /// Run or continue one model-wide PV step, publishing a durable overlay after
    /// every newly consumed host-gradient block.
    pub fn step_resumable<F>(
        &mut self,
        tokens: &[i32],
        target: &DeviceTensor,
        base_checkpoint: &[u8],
        mut persist: F,
    ) -> Result<DevicePvRecoveryStepReceipt, DevicePvRecoveryError>
    where
        F: FnMut(&[u8]) -> Result<(), DevicePvRecoveryError>,
    {
        self.ensure_usable()?;
        let optimizer_step = self.completed_step.checked_add(1).ok_or_else(|| {
            DevicePvRecoveryError::InvalidInput("PV step counter overflowed".into())
        })?;
        let base_checkpoint_digest = *blake3::hash(base_checkpoint).as_bytes();
        let batch_digest = batch_digest(
            self.backend,
            self.plan_digest,
            optimizer_step,
            tokens,
            target,
        )?;

        match &self.campaign {
            Some(campaign) => {
                if campaign.base_checkpoint_digest != base_checkpoint_digest {
                    return Err(campaign_error("base checkpoint identity mismatch"));
                }
                if campaign.batch_digest != batch_digest {
                    return Err(campaign_error("campaign batch identity mismatch"));
                }
                if campaign.optimizer_step != optimizer_step {
                    return Err(campaign_error("campaign optimizer step mismatch"));
                }
            }
            None => {
                let current_digest = match self.checkpoint_source_digest {
                    Some(digest) => digest,
                    None => self.checkpoint_digest()?,
                };
                if current_digest != base_checkpoint_digest {
                    return Err(campaign_error(
                        "base checkpoint is not the session's exact current state",
                    ));
                }
                let source_state_digest =
                    session_state_digest(self.plan_digest, self.completed_step, &self.tuning);
                self.campaign = Some(DevicePvRecoveryCampaignState {
                    base_checkpoint_digest,
                    source_state_digest,
                    batch_digest,
                    optimizer_step,
                    receipts: vec![None; self.tuning.len()],
                });
            }
        }

        self.checkpoint_source_digest = None;
        let max_block_elements = self.max_host_gradient_block_elements;
        let plan_digest = self.plan_digest;
        let tuning = &mut self.tuning;
        let campaign = self
            .campaign
            .as_mut()
            .expect("campaign was initialized above");
        let mut callback_error = None;
        let report = ModelGradientStream {
            backend: self.backend,
            model: self.model,
            packed: &self.packed,
            tokens,
            target,
            max_block_elements,
            optimizer_step,
        }
        .visit(|parameter_index, offset, total, gradient| {
            let outcome = apply_campaign_block(
                tuning,
                campaign,
                parameter_index,
                offset,
                total,
                gradient,
                max_block_elements,
                optimizer_step,
            )
            .and_then(|applied| {
                if !applied {
                    return Ok(());
                }
                let checkpoint =
                    encode_campaign_checkpoint(plan_digest, max_block_elements, tuning, campaign)?;
                persist(&checkpoint)
            });
            match outcome {
                Ok(()) => Ok(()),
                Err(error) => {
                    let message = error.to_string();
                    callback_error = Some(error);
                    Err(BackendError::Backend(message))
                }
            }
        });
        let report = match report {
            Ok(report) => report,
            Err(error) => {
                return Err(callback_error.unwrap_or(error));
            }
        };
        finish_campaign(self, optimizer_step, report)
    }

    /// Serialize current model-wide in-flight campaign overlay.
    pub fn campaign_checkpoint_bytes(&self) -> Result<Vec<u8>, DevicePvRecoveryError> {
        self.ensure_usable()?;
        let campaign = self
            .campaign
            .as_ref()
            .ok_or_else(|| campaign_error("no resumable campaign is active"))?;
        encode_campaign_checkpoint(
            self.plan_digest,
            self.max_host_gradient_block_elements,
            &self.tuning,
            campaign,
        )
    }

    /// Resume exact completed base state plus a checksum-bound in-flight overlay.
    pub fn resume_campaign(
        backend: &'backend tritium_cuda::CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        base_checkpoint: &[u8],
        campaign_checkpoint: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_campaign_with_recovery_context(
            backend,
            model,
            parents,
            config,
            None,
            None,
            base_checkpoint,
            campaign_checkpoint,
        )
    }

    /// Resume an in-flight overlay only under its original frozen recovery campaign.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_campaign_for_recovery_campaign(
        backend: &'backend tritium_cuda::CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        recovery_campaign: &RecoveryCampaignRun,
        base_checkpoint: &[u8],
        campaign_checkpoint: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_campaign_with_recovery_context(
            backend,
            model,
            parents,
            config,
            Some(pv_campaign_context(recovery_campaign)?),
            None,
            base_checkpoint,
            campaign_checkpoint,
        )
    }

    /// Resume an in-flight recovery campaign under exact package-parent lineage.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_campaign_for_recovery_campaign_with_parent_context(
        backend: &'backend tritium_cuda::CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        recovery_campaign: &RecoveryCampaignRun,
        parent_context: DevicePvRecoveryParentContext,
        base_checkpoint: &[u8],
        campaign_checkpoint: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::resume_campaign_with_recovery_context(
            backend,
            model,
            parents,
            config,
            Some(pv_campaign_context(recovery_campaign)?),
            Some(parent_context),
            base_checkpoint,
            campaign_checkpoint,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn resume_campaign_with_recovery_context(
        backend: &'backend tritium_cuda::CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        recovery_campaign_context: Option<RecoveryEvidenceDigest>,
        parent_context: Option<DevicePvRecoveryParentContext>,
        base_checkpoint: &[u8],
        campaign_checkpoint: &[u8],
    ) -> Result<Self, DevicePvRecoveryError> {
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
        let parsed = parse_campaign_checkpoint(campaign_checkpoint, parents.len())?;
        if parsed.plan_digest != expected_plan {
            return Err(campaign_error("campaign plan identity mismatch"));
        }
        if parsed.base_checkpoint_digest != *blake3::hash(base_checkpoint).as_bytes() {
            return Err(campaign_error("campaign base checkpoint identity mismatch"));
        }
        let mut session = Self::resume_with_recovery_campaign_context(
            backend,
            model,
            parents.clone(),
            config,
            parsed.max_block_elements,
            recovery_campaign_context,
            parent_context,
            base_checkpoint,
        )?;
        let expected_step = session.completed_step.checked_add(1).ok_or_else(|| {
            DevicePvRecoveryError::Checkpoint("PV step counter overflowed".into())
        })?;
        if parsed.optimizer_step != expected_step {
            return Err(campaign_error(
                "campaign optimizer step is not the exact next model step",
            ));
        }
        let actual_source_state =
            session_state_digest(session.plan_digest, session.completed_step, &session.tuning);
        if parsed.source_state_digest != actual_source_state {
            return Err(campaign_error("campaign source-state identity mismatch"));
        }

        let mut receipts = Vec::with_capacity(parents.len());
        let mut in_flight = 0usize;
        for (index, (parent, overlay)) in parents.into_iter().zip(parsed.overlays).enumerate() {
            match overlay {
                ParsedOverlay::Untouched => receipts.push(None),
                ParsedOverlay::Completed {
                    receipt_bytes,
                    tuning_bytes,
                } => {
                    let receipt = PvStepReceipt::resume(receipt_bytes)?;
                    let tuning = PvTuningSession::resume(parent, config, tuning_bytes)?;
                    validate_completed_overlay(&receipt, &tuning, parsed.optimizer_step)?;
                    session.tuning[index] = tuning;
                    receipts.push(Some(receipt));
                }
                ParsedOverlay::InFlight { tuning_bytes } => {
                    in_flight += 1;
                    if in_flight > 1 {
                        return Err(campaign_error(
                            "campaign contains multiple in-flight tensors",
                        ));
                    }
                    let tuning = PvTuningSession::resume_blockwise(parent, config, tuning_bytes)?;
                    let cursor = tuning
                        .blockwise_cursor()
                        .ok_or_else(|| campaign_error("in-flight cursor is missing"))?;
                    if cursor.optimizer_step() != parsed.optimizer_step
                        || cursor.max_block_elements() != parsed.max_block_elements
                        || cursor.next_offset() == 0
                        || cursor.next_offset() == cursor.total_elements()
                    {
                        return Err(campaign_error("in-flight cursor is not canonical"));
                    }
                    session.tuning[index] = tuning;
                    receipts.push(None);
                }
            }
        }
        session.campaign = Some(DevicePvRecoveryCampaignState {
            base_checkpoint_digest: parsed.base_checkpoint_digest,
            source_state_digest: parsed.source_state_digest,
            batch_digest: parsed.batch_digest,
            optimizer_step: parsed.optimizer_step,
            receipts,
        });
        session.checkpoint_source_digest = None;
        Ok(session)
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_campaign_block(
    tuning: &mut [PvTuningSession],
    campaign: &mut DevicePvRecoveryCampaignState,
    parameter_index: usize,
    offset: usize,
    total: usize,
    gradient: &[f32],
    max_block_elements: usize,
    optimizer_step: u64,
) -> Result<bool, DevicePvRecoveryError> {
    let receipt = campaign
        .receipts
        .get(parameter_index)
        .ok_or_else(|| campaign_error("gradient parameter index is out of range"))?;
    if receipt.is_some() {
        return Ok(false);
    }
    let session = tuning
        .get_mut(parameter_index)
        .ok_or_else(|| campaign_error("gradient parameter index is out of range"))?;
    let expected_total = session
        .weight()
        .rows()
        .checked_mul(session.weight().cols())
        .ok_or_else(|| campaign_error("PV parameter length overflows host range"))?;
    if total != expected_total {
        return Err(campaign_error("gradient parameter length mismatch"));
    }
    let end = offset
        .checked_add(gradient.len())
        .ok_or_else(|| campaign_error("gradient block range overflow"))?;
    match session.blockwise_cursor() {
        Some(cursor) => {
            if cursor.optimizer_step() != optimizer_step
                || cursor.max_block_elements() != max_block_elements
                || cursor.total_elements() != total
            {
                return Err(campaign_error("gradient cursor identity mismatch"));
            }
            if end <= cursor.next_offset() {
                return Ok(false);
            }
            if offset != cursor.next_offset() {
                return Err(campaign_error(
                    "persisted cursor does not align with replayed gradient blocks",
                ));
            }
        }
        None => {
            if offset != 0 {
                return Err(campaign_error(
                    "gradient stream did not begin at offset zero",
                ));
            }
            session.begin_blockwise_step(optimizer_step, max_block_elements)?;
        }
    }
    session.apply_gradient_block(offset, gradient)?;
    if end == total {
        let receipt = session.finish_blockwise_step()?;
        campaign.receipts[parameter_index] = Some(receipt);
    }
    Ok(true)
}

fn finish_campaign(
    session: &mut DevicePvRecoverySession<'_, '_>,
    optimizer_step: u64,
    report: GradientStreamReport,
) -> Result<DevicePvRecoveryStepReceipt, DevicePvRecoveryError> {
    let receipts = session
        .campaign
        .as_ref()
        .expect("campaign remains active until commit")
        .receipts
        .iter()
        .cloned()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| campaign_error("gradient stream omitted a PV parameter"))?;
    let campaign = session
        .campaign
        .as_ref()
        .expect("campaign remains active until commit");
    let source_state_digest = campaign.source_state_digest;
    let batch_digest = campaign.batch_digest;
    session.refresh_packed()?;
    session.completed_step = optimizer_step;
    session.campaign = None;
    session.checkpoint_source_digest = None;
    let receipt = build_receipt(
        PvStepIdentity {
            plan_digest: session.plan_digest,
            optimizer_step,
            source_state_digest,
            batch_digest,
        },
        receipts,
        report,
        &session.tuning,
        &session.packed,
    )
    .inspect_err(|_| {
        session.poisoned = true;
    })?;
    session.last_step_receipt_digest =
        Some(canonical_receipt_digest(&receipt).inspect_err(|_| {
            session.poisoned = true;
        })?);
    Ok(receipt)
}

fn encode_campaign_checkpoint(
    plan_digest: [u8; 32],
    max_block_elements: usize,
    tuning: &[PvTuningSession],
    campaign: &DevicePvRecoveryCampaignState,
) -> Result<Vec<u8>, DevicePvRecoveryError> {
    if max_block_elements == 0 || campaign.receipts.len() != tuning.len() {
        return Err(campaign_error("campaign state is inconsistent"));
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&plan_digest);
    out.extend_from_slice(&campaign.base_checkpoint_digest);
    out.extend_from_slice(&campaign.source_state_digest);
    out.extend_from_slice(&campaign.batch_digest);
    out.extend_from_slice(&campaign.optimizer_step.to_le_bytes());
    append_usize(&mut out, max_block_elements)?;
    append_usize(&mut out, tuning.len())?;
    let mut in_flight = 0usize;
    let binding_identity =
        CampaignBindingIdentity::from_state(plan_digest, campaign, max_block_elements);
    for (out_index, (session, receipt)) in tuning.iter().zip(&campaign.receipts).enumerate() {
        match (receipt, session.blockwise_cursor()) {
            (Some(receipt), None) => {
                validate_completed_overlay(receipt, session, campaign.optimizer_step)?;
                out.push(COMPLETED);
                let receipt_bytes = receipt.checkpoint_bytes();
                let tuning_bytes = session.checkpoint_bytes()?;
                append_blob(&mut out, &receipt_bytes)?;
                append_blob(&mut out, &tuning_bytes)?;
                out.extend_from_slice(&overlay_binding_digest(
                    binding_identity,
                    out_index,
                    COMPLETED,
                    &receipt_bytes,
                    &tuning_bytes,
                ));
            }
            (None, Some(cursor)) => {
                in_flight += 1;
                if in_flight > 1
                    || cursor.optimizer_step() != campaign.optimizer_step
                    || cursor.max_block_elements() != max_block_elements
                    || cursor.next_offset() == 0
                    || cursor.next_offset() == cursor.total_elements()
                {
                    return Err(campaign_error("campaign cursor is not canonical"));
                }
                out.push(IN_FLIGHT);
                let tuning_bytes = session.blockwise_checkpoint_bytes()?;
                append_blob(&mut out, &tuning_bytes)?;
                out.extend_from_slice(&overlay_binding_digest(
                    binding_identity,
                    out_index,
                    IN_FLIGHT,
                    &[],
                    &tuning_bytes,
                ));
            }
            (None, None) => {
                if session.completed_step().checked_add(1) != Some(campaign.optimizer_step) {
                    return Err(campaign_error("untouched tensor step is inconsistent"));
                }
                out.push(UNTOUCHED);
            }
            (Some(_), Some(_)) => {
                return Err(campaign_error(
                    "completed receipt has an in-flight tensor cursor",
                ));
            }
        }
    }
    let checksum = blake3::hash(&out);
    out.extend_from_slice(checksum.as_bytes());
    Ok(out)
}

fn validate_completed_overlay(
    receipt: &PvStepReceipt,
    session: &PvTuningSession,
    optimizer_step: u64,
) -> Result<(), DevicePvRecoveryError> {
    if receipt.optimizer_step() != optimizer_step
        || session.completed_step() != optimizer_step
        || receipt.representation_digest() != session.weight().digest()
    {
        return Err(campaign_error("completed tensor overlay is inconsistent"));
    }
    Ok(())
}

fn append_usize(out: &mut Vec<u8>, value: usize) -> Result<(), DevicePvRecoveryError> {
    let value =
        u64::try_from(value).map_err(|_| campaign_error("campaign value exceeds wire range"))?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn append_blob(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), DevicePvRecoveryError> {
    append_usize(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

struct ParsedCampaign<'bytes> {
    plan_digest: [u8; 32],
    base_checkpoint_digest: [u8; 32],
    source_state_digest: [u8; 32],
    batch_digest: [u8; 32],
    optimizer_step: u64,
    max_block_elements: usize,
    overlays: Vec<ParsedOverlay<'bytes>>,
}

enum ParsedOverlay<'bytes> {
    Untouched,
    Completed {
        receipt_bytes: &'bytes [u8],
        tuning_bytes: &'bytes [u8],
    },
    InFlight {
        tuning_bytes: &'bytes [u8],
    },
}

fn parse_campaign_checkpoint(
    bytes: &[u8],
    expected_count: usize,
) -> Result<ParsedCampaign<'_>, DevicePvRecoveryError> {
    let body_len = bytes
        .len()
        .checked_sub(CHECKSUM_BYTES)
        .ok_or_else(|| campaign_error("campaign checkpoint is truncated"))?;
    let (body, checksum) = bytes.split_at(body_len);
    if blake3::hash(body).as_bytes() != checksum {
        return Err(campaign_error("campaign checkpoint checksum mismatch"));
    }
    let mut reader = Reader::new(body);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(campaign_error("bad campaign checkpoint magic"));
    }
    let plan_digest = reader.array()?;
    let base_checkpoint_digest = reader.array()?;
    let source_state_digest = reader.array()?;
    let batch_digest = reader.array()?;
    let optimizer_step = reader.u64()?;
    if optimizer_step == 0 {
        return Err(campaign_error("campaign optimizer step is zero"));
    }
    let max_block_elements = reader.usize()?;
    if max_block_elements == 0 {
        return Err(campaign_error("campaign block size is zero"));
    }
    let count = reader.usize()?;
    if count != expected_count {
        return Err(campaign_error("campaign parameter count mismatch"));
    }
    let mut overlays = Vec::new();
    overlays
        .try_reserve_exact(count)
        .map_err(|_| campaign_error("campaign overlay allocation failed"))?;
    let binding_identity = CampaignBindingIdentity {
        plan_digest,
        base_checkpoint_digest,
        source_state_digest,
        batch_digest,
        optimizer_step,
        max_block_elements,
    };
    for index in 0..count {
        let tag = reader.u8()?;
        overlays.push(match tag {
            UNTOUCHED => ParsedOverlay::Untouched,
            COMPLETED => {
                let receipt_bytes = reader.blob()?;
                let tuning_bytes = reader.blob()?;
                let binding = reader.array()?;
                validate_overlay_binding(
                    binding_identity,
                    index,
                    tag,
                    receipt_bytes,
                    tuning_bytes,
                    binding,
                )?;
                ParsedOverlay::Completed {
                    receipt_bytes,
                    tuning_bytes,
                }
            }
            IN_FLIGHT => {
                let tuning_bytes = reader.blob()?;
                let binding = reader.array()?;
                validate_overlay_binding(binding_identity, index, tag, &[], tuning_bytes, binding)?;
                ParsedOverlay::InFlight { tuning_bytes }
            }
            _ => return Err(campaign_error("unknown campaign tensor tag")),
        });
    }
    if reader.remaining() != 0 {
        return Err(campaign_error("campaign checkpoint has trailing bytes"));
    }
    Ok(ParsedCampaign {
        plan_digest,
        base_checkpoint_digest,
        source_state_digest,
        batch_digest,
        optimizer_step,
        max_block_elements,
        overlays,
    })
}

fn validate_overlay_binding(
    identity: CampaignBindingIdentity,
    index: usize,
    tag: u8,
    receipt_bytes: &[u8],
    tuning_bytes: &[u8],
    binding: [u8; 32],
) -> Result<(), DevicePvRecoveryError> {
    if binding != overlay_binding_digest(identity, index, tag, receipt_bytes, tuning_bytes) {
        return Err(campaign_error("campaign tensor overlay identity mismatch"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct CampaignBindingIdentity {
    plan_digest: [u8; 32],
    base_checkpoint_digest: [u8; 32],
    source_state_digest: [u8; 32],
    batch_digest: [u8; 32],
    optimizer_step: u64,
    max_block_elements: usize,
}

impl CampaignBindingIdentity {
    fn from_state(
        plan_digest: [u8; 32],
        campaign: &DevicePvRecoveryCampaignState,
        max_block_elements: usize,
    ) -> Self {
        Self {
            plan_digest,
            base_checkpoint_digest: campaign.base_checkpoint_digest,
            source_state_digest: campaign.source_state_digest,
            batch_digest: campaign.batch_digest,
            optimizer_step: campaign.optimizer_step,
            max_block_elements,
        }
    }
}

fn overlay_binding_digest(
    identity: CampaignBindingIdentity,
    index: usize,
    tag: u8,
    receipt_bytes: &[u8],
    tuning_bytes: &[u8],
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium.device-pv-campaign-overlay.v1\0");
    hash.update(&identity.plan_digest);
    hash.update(&identity.base_checkpoint_digest);
    hash.update(&identity.source_state_digest);
    hash.update(&identity.batch_digest);
    hash.update(&identity.optimizer_step.to_le_bytes());
    hash.update(&(identity.max_block_elements as u64).to_le_bytes());
    hash.update(&(index as u64).to_le_bytes());
    hash.update(&[tag]);
    hash.update(&(receipt_bytes.len() as u64).to_le_bytes());
    hash.update(receipt_bytes);
    hash.update(&(tuning_bytes.len() as u64).to_le_bytes());
    hash.update(tuning_bytes);
    *hash.finalize().as_bytes()
}

fn campaign_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Checkpoint(reason.into())
}
