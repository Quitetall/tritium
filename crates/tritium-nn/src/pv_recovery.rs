//! Model-level hard-PV recovery over compact deployed weights.
//!
//! Forward and activation VJPs stay device-resident. Finalized projected-weight
//! gradients cross to one reusable bounded host buffer, update explicitly
//! accounted PV state, then disappear. No dense latent master belongs to this
//! session; dense Adam moments remain visible in its physical ledger.

mod artifact;
mod campaign;
mod checkpoint;
mod error;
mod identity;
mod receipt;
mod session;
mod snapshot;
mod wire;

pub use artifact::{DevicePvRecoveryCheckpointArtifact, DevicePvRecoveryWeightVisitError};
pub use error::DevicePvRecoveryError;
pub use receipt::DevicePvRecoveryStepReceipt;

/// Nonzero identity of exact package lineage supplying hard-PV parent weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePvRecoveryParentContext([u8; 32]);

impl DevicePvRecoveryParentContext {
    /// Construct package-parent authority from a domain-separated lineage digest.
    pub fn new(bytes: [u8; 32]) -> Result<Self, DevicePvRecoveryError> {
        if bytes == [0; 32] {
            return Err(DevicePvRecoveryError::InvalidInput(
                "PV package-parent context cannot be all zero".into(),
            ));
        }
        Ok(Self(bytes))
    }

    /// Exact package-parent lineage identity.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Default maximum f32 elements staged on host for one PV gradient block (4 MiB).
pub const DEFAULT_PV_HOST_GRADIENT_BLOCK_ELEMENTS: usize = 1 << 20;

use tritium_cuda::CudaBackend;
use tritium_cuda::train::DevicePackedSaltWeight;
use tritium_train::{PvStepReceipt, PvTuningSession, RecoveryEvidenceDigest};

use crate::training::TiedSwiGluTrainingModel;

/// Resident packed model plus per-tensor PV optimizer state with explicit accounting.
pub struct DevicePvRecoverySession<'backend, 'model> {
    pub(super) backend: &'backend CudaBackend,
    pub(super) model: &'model TiedSwiGluTrainingModel,
    pub(super) plan_digest: [u8; 32],
    pub(super) recovery_campaign_context: Option<RecoveryEvidenceDigest>,
    pub(super) parent_context: Option<DevicePvRecoveryParentContext>,
    pub(super) parent_catalog_digest: Option<[u8; 32]>,
    pub(super) tuning: Vec<PvTuningSession>,
    pub(super) packed: Vec<DevicePackedSaltWeight>,
    pub(super) max_host_gradient_block_elements: usize,
    pub(super) completed_step: u64,
    pub(super) last_step_receipt_digest: Option<[u8; 32]>,
    pub(super) checkpoint_source_digest: Option<[u8; 32]>,
    pub(super) campaign: Option<DevicePvRecoveryCampaignState>,
    pub(super) poisoned: bool,
}

pub(super) struct DevicePvRecoveryCampaignState {
    pub(super) base_checkpoint_digest: [u8; 32],
    pub(super) source_state_digest: [u8; 32],
    pub(super) batch_digest: [u8; 32],
    pub(super) optimizer_step: u64,
    pub(super) receipts: Vec<Option<PvStepReceipt>>,
}
