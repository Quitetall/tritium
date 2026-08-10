use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    CheckpointPolicy, DevicePackedSaltWeight, DeviceTape, DeviceTensor, GradientLeafBinding,
    GradientStreamReport,
};
use tritium_spec::BackendError;
use tritium_train::{
    PvStepReceipt, PvTernaryWeight, PvTuningConfig, PvTuningSession, RecoveryCampaignRun,
    RecoveryEvidenceDigest,
};

use super::checkpoint::completed_model_checkpoint_len;
use super::identity::{
    batch_digest, evidence_digest, package_parent_catalog_digest, plan_digest, pv_campaign_context,
    representation_digest, session_state_digest, validate_model_parents,
};
use super::receipt::canonical_receipt_digest;
use super::snapshot::pack_snapshot;
use super::{
    DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS, DevicePvRecoveryError, DevicePvRecoveryParentContext,
    DevicePvRecoverySession, DevicePvRecoveryStepReceipt,
};
use crate::training::{TiedSwiGluTrainingModel, packed_device_forward};

impl core::fmt::Debug for DevicePvRecoverySession<'_, '_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DevicePvRecoverySession")
            .field("plan_digest", &self.plan_digest)
            .field("recovery_campaign_context", &self.recovery_campaign_context)
            .field("parent_context", &self.parent_context)
            .field("parent_catalog_digest", &self.parent_catalog_digest)
            .field("completed_step", &self.completed_step)
            .field("parameter_count", &self.tuning.len())
            .field(
                "max_host_gradient_block_elements",
                &self.max_host_gradient_block_elements,
            )
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'backend, 'model> DevicePvRecoverySession<'backend, 'model> {
    /// Build a hard-PV session only after dense masters leave model descriptor.
    pub fn new(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_with_gradient_block_elements(
            backend,
            model,
            parents,
            config,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
        )
    }

    /// Build a hard-PV session with an explicit bounded host-gradient block size.
    pub fn new_with_gradient_block_elements(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        max_host_gradient_block_elements: usize,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            max_host_gradient_block_elements,
            None,
            None,
        )
    }

    /// Build a hard-PV session whose complete plan is bound to one frozen campaign.
    pub fn new_for_recovery_campaign(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_for_recovery_campaign_with_gradient_block_elements(
            backend,
            model,
            parents,
            config,
            campaign,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
        )
    }

    /// Build a campaign-bound hard-PV session with explicit host-gradient bound.
    pub fn new_for_recovery_campaign_with_gradient_block_elements(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        max_host_gradient_block_elements: usize,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            max_host_gradient_block_elements,
            Some(pv_campaign_context(campaign)?),
            None,
        )
    }

    /// Build a recovery-campaign session whose parent weights come from one exact package lineage.
    pub fn new_for_recovery_campaign_with_parent_context(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        parent_context: DevicePvRecoveryParentContext,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_for_recovery_campaign_with_parent_context_and_gradient_block_elements(
            backend,
            model,
            parents,
            config,
            campaign,
            parent_context,
            DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS,
        )
    }

    /// Build a package-bound recovery session with an explicit host-gradient bound.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_recovery_campaign_with_parent_context_and_gradient_block_elements(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        campaign: &RecoveryCampaignRun,
        parent_context: DevicePvRecoveryParentContext,
        max_host_gradient_block_elements: usize,
    ) -> Result<Self, DevicePvRecoveryError> {
        Self::new_with_recovery_campaign_context(
            backend,
            model,
            parents,
            config,
            max_host_gradient_block_elements,
            Some(pv_campaign_context(campaign)?),
            Some(parent_context),
        )
    }

    fn new_with_recovery_campaign_context(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        parents: Vec<PvTernaryWeight>,
        config: PvTuningConfig,
        max_host_gradient_block_elements: usize,
        recovery_campaign_context: Option<RecoveryEvidenceDigest>,
        parent_context: Option<DevicePvRecoveryParentContext>,
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
        let plan_digest = plan_digest(
            model,
            &parents,
            config,
            recovery_campaign_context,
            parent_context,
            parent_catalog_digest,
        );
        let packed = parents
            .iter()
            .map(
                |parent| -> Result<DevicePackedSaltWeight, DevicePvRecoveryError> {
                    let snapshot = pack_snapshot(parent)?;
                    Ok(DevicePackedSaltWeight::from_snapshot(backend, &snapshot)?)
                },
            )
            .collect::<Result<Vec<_>, DevicePvRecoveryError>>()?;
        let tuning = parents
            .into_iter()
            .map(|parent| PvTuningSession::new(parent, config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            backend,
            model,
            plan_digest,
            recovery_campaign_context,
            parent_context,
            parent_catalog_digest,
            tuning,
            packed,
            max_host_gradient_block_elements,
            completed_step: 0,
            last_step_receipt_digest: None,
            checkpoint_source_digest: None,
            campaign: None,
            poisoned: false,
        })
    }

    #[must_use]
    pub const fn completed_step(&self) -> u64 {
        self.completed_step
    }

    #[must_use]
    pub const fn plan_digest(&self) -> [u8; 32] {
        self.plan_digest
    }

    /// Frozen campaign authorization included in this session's plan identity.
    #[must_use]
    pub const fn recovery_campaign_context_digest(&self) -> Option<RecoveryEvidenceDigest> {
        self.recovery_campaign_context
    }

    /// Exact package lineage whose decoded weights initialized this session.
    #[must_use]
    pub const fn parent_context(&self) -> Option<DevicePvRecoveryParentContext> {
        self.parent_context
    }

    /// Exact named Pmax parent semantics frozen before any recovery step.
    #[must_use]
    pub const fn parent_catalog_digest(&self) -> Option<[u8; 32]> {
        self.parent_catalog_digest
    }

    pub fn weight(&self, index: usize) -> Result<&PvTernaryWeight, DevicePvRecoveryError> {
        self.tuning
            .get(index)
            .map(PvTuningSession::weight)
            .ok_or_else(|| {
                DevicePvRecoveryError::InvalidInput(format!(
                    "PV parameter index {index} is out of range"
                ))
            })
    }

    /// Evaluate current hard representation against resident teacher probabilities.
    ///
    /// This returns teacher cross-entropy; subtracting fixed teacher entropy yields
    /// KL, so candidate ordering is identical. Only scalar loss crosses to host.
    /// Callers must supply finite row-normalized probabilities in `target`.
    pub fn held_out_teacher_cross_entropy(
        &self,
        tokens: &[i32],
        target: &DeviceTensor,
    ) -> Result<f32, DevicePvRecoveryError> {
        self.ensure_usable()?;
        if self.campaign.is_some() {
            return Err(DevicePvRecoveryError::InvalidInput(
                "cannot evaluate held-out loss during an in-flight PV campaign".into(),
            ));
        }
        let arch = self.model.architecture();
        let mut tape = DeviceTape::new_with_checkpoint_policy(
            self.backend,
            arch.vocab.max(arch.n_ff).max(tokens.len()),
            CheckpointPolicy::SqrtDepth(arch.n_layers),
        )?;
        let forward = packed_device_forward(&mut tape, self.model, &self.packed, tokens)?;
        let loss = tape.softmax_xent_value(forward.logits, target, tokens.len(), arch.vocab)?;
        if !loss.is_finite() || loss < 0.0 {
            return Err(DevicePvRecoveryError::Backend(
                "held-out teacher cross-entropy is non-finite or negative".into(),
            ));
        }
        Ok(loss)
    }

    /// Run one whole-model hard forward and stream real gradients into alternating PV.
    pub fn step(
        &mut self,
        tokens: &[i32],
        target: &DeviceTensor,
    ) -> Result<DevicePvRecoveryStepReceipt, DevicePvRecoveryError> {
        self.ensure_usable()?;
        if self.campaign.is_some() {
            return Err(DevicePvRecoveryError::InvalidInput(
                "a resumable PV campaign is active; continue it with step_resumable".into(),
            ));
        }
        let optimizer_step = self.completed_step.checked_add(1).ok_or_else(|| {
            DevicePvRecoveryError::InvalidInput("PV step counter overflowed".into())
        })?;
        let source_state_digest =
            session_state_digest(self.plan_digest, self.completed_step, &self.tuning);
        let batch_digest = batch_digest(
            self.backend,
            self.plan_digest,
            optimizer_step,
            tokens,
            target,
        )?;
        self.checkpoint_source_digest = None;
        let mut receipts = vec![None; self.tuning.len()];
        let tuning = &mut self.tuning;
        let max_host_gradient_block_elements = self.max_host_gradient_block_elements;
        let report = ModelGradientStream {
            backend: self.backend,
            model: self.model,
            packed: &self.packed,
            tokens,
            target,
            max_block_elements: max_host_gradient_block_elements,
            optimizer_step,
        }
        .visit(|parameter_index, offset, total, gradient| {
            let session = tuning.get_mut(parameter_index).ok_or_else(|| {
                BackendError::InvalidInput(format!(
                    "PV gradient parameter index {parameter_index} is out of range"
                ))
            })?;
            if offset == 0 {
                session
                    .begin_blockwise_step(optimizer_step, max_host_gradient_block_elements)
                    .map_err(|error| {
                        BackendError::Backend(format!("PV parameter {parameter_index}: {error}"))
                    })?;
            }
            session
                .apply_gradient_block(offset, gradient)
                .map_err(|error| {
                    BackendError::Backend(format!("PV parameter {parameter_index}: {error}"))
                })?;
            if offset + gradient.len() == total {
                let receipt = session.finish_blockwise_step().map_err(|error| {
                    BackendError::Backend(format!("PV parameter {parameter_index}: {error}"))
                })?;
                receipts[parameter_index] = Some(receipt);
            }
            Ok(())
        });
        let report = match report {
            Ok(report) => report,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        let tensor_receipts = match receipts.into_iter().collect::<Option<Vec<_>>>() {
            Some(receipts) => receipts,
            None => {
                self.poisoned = true;
                return Err(DevicePvRecoveryError::Backend(
                    "gradient stream omitted a PV parameter".into(),
                ));
            }
        };
        self.refresh_packed()?;
        self.completed_step = optimizer_step;
        let receipt = build_receipt(
            PvStepIdentity {
                plan_digest: self.plan_digest,
                optimizer_step,
                source_state_digest,
                batch_digest,
            },
            tensor_receipts,
            report,
            &self.tuning,
            &self.packed,
        )
        .inspect_err(|_| {
            self.poisoned = true;
        })?;
        self.last_step_receipt_digest =
            Some(canonical_receipt_digest(&receipt).inspect_err(|_| {
                self.poisoned = true;
            })?);
        Ok(receipt)
    }

    pub(super) fn refresh_packed(&mut self) -> Result<(), DevicePvRecoveryError> {
        for (packed, session) in self.packed.iter_mut().zip(&self.tuning) {
            let snapshot = match pack_snapshot(session.weight()) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.poisoned = true;
                    return Err(error);
                }
            };
            if let Err(error) = packed.update_from_snapshot(self.backend, &snapshot) {
                self.poisoned = true;
                return Err(error.into());
            }
        }
        Ok(())
    }

    pub(super) fn ensure_usable(&self) -> Result<(), DevicePvRecoveryError> {
        if self.poisoned {
            Err(DevicePvRecoveryError::InvalidInput(
                "device PV session is poisoned; resume a durable checkpoint".into(),
            ))
        } else {
            Ok(())
        }
    }
}

pub(super) struct ModelGradientStream<'state> {
    pub(super) backend: &'state CudaBackend,
    pub(super) model: &'state TiedSwiGluTrainingModel,
    pub(super) packed: &'state [DevicePackedSaltWeight],
    pub(super) tokens: &'state [i32],
    pub(super) target: &'state DeviceTensor,
    pub(super) max_block_elements: usize,
    pub(super) optimizer_step: u64,
}

impl ModelGradientStream<'_> {
    pub(super) fn visit<F>(self, visitor: F) -> Result<GradientStreamReport, DevicePvRecoveryError>
    where
        F: FnMut(usize, usize, usize, &[f32]) -> Result<(), BackendError>,
    {
        let arch = self.model.architecture();
        let mut tape = DeviceTape::new_with_checkpoint_policy(
            self.backend,
            arch.vocab.max(arch.n_ff).max(self.tokens.len()),
            CheckpointPolicy::SqrtDepth(arch.n_layers),
        )?;
        let forward = packed_device_forward(&mut tape, self.model, self.packed, self.tokens)?;
        let bindings = forward
            .master_leaves
            .iter()
            .enumerate()
            .map(|(parameter_index, &leaf_id)| GradientLeafBinding {
                leaf_id,
                parameter_index,
            })
            .collect::<Vec<_>>();
        let lengths = self
            .model
            .parameters()
            .iter()
            .map(|parameter| parameter.elements())
            .collect::<Vec<_>>();
        Ok(tape.xent_backward_visit_host_gradient_blocks(
            forward.logits,
            self.target,
            self.tokens.len(),
            arch.vocab,
            &bindings,
            &lengths,
            self.max_block_elements,
            self.optimizer_step,
            visitor,
        )?)
    }
}

#[derive(Clone, Copy)]
pub(super) struct PvStepIdentity {
    pub(super) plan_digest: [u8; 32],
    pub(super) optimizer_step: u64,
    pub(super) source_state_digest: [u8; 32],
    pub(super) batch_digest: [u8; 32],
}

pub(super) fn build_receipt(
    identity: PvStepIdentity,
    tensor_receipts: Vec<PvStepReceipt>,
    report: tritium_cuda::train::GradientStreamReport,
    tuning: &[PvTuningSession],
    packed: &[DevicePackedSaltWeight],
) -> Result<DevicePvRecoveryStepReceipt, DevicePvRecoveryError> {
    let representation_digest = representation_digest(
        identity.plan_digest,
        identity.optimizer_step,
        tensor_receipts
            .iter()
            .map(PvStepReceipt::representation_digest),
    );
    let mut host_representation_bytes = 0usize;
    let mut host_optimizer_bytes = 0usize;
    let mut host_campaign_bytes = 0usize;
    for session in tuning {
        let ledger = session.size_ledger()?;
        host_representation_bytes = host_representation_bytes
            .checked_add(ledger.host_representation_bytes())
            .ok_or_else(accounting_error)?;
        host_optimizer_bytes = host_optimizer_bytes
            .checked_add(ledger.host_optimizer_bytes())
            .ok_or_else(accounting_error)?;
        host_campaign_bytes = host_campaign_bytes
            .checked_add(ledger.host_campaign_bytes())
            .ok_or_else(accounting_error)?;
    }
    let resident_packed_bytes = packed.iter().try_fold(0usize, |total, weight| {
        total
            .checked_add(weight.resident_bytes())
            .ok_or_else(accounting_error)
    })?;
    let evidence_digest = evidence_digest(
        identity.plan_digest,
        identity.source_state_digest,
        identity.batch_digest,
        identity.optimizer_step,
        representation_digest,
    );
    Ok(DevicePvRecoveryStepReceipt {
        plan_digest: identity.plan_digest,
        optimizer_step: identity.optimizer_step,
        tensor_receipts,
        materialized_gradient_elements: report.materialized_collection_elements,
        peak_live_gradient_elements: report.peak_live_requested_gradient_elements,
        peak_host_gradient_elements: report.peak_host_gradient_elements,
        peak_live_activation_elements: report.backward_stats.peak_live_activation_elements,
        naive_activation_elements: report.backward_stats.naive_activation_elements,
        host_representation_bytes,
        host_optimizer_bytes,
        host_campaign_bytes,
        resident_packed_bytes,
        serialized_checkpoint_bytes: completed_model_checkpoint_len(tuning)?,
        source_state_digest: identity.source_state_digest,
        batch_digest: identity.batch_digest,
        representation_digest,
        evidence_digest,
    })
}

fn accounting_error() -> DevicePvRecoveryError {
    DevicePvRecoveryError::InvalidInput("PV physical accounting overflows host range".into())
}
