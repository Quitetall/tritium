//! Native reference P/V refinement for deployed additive ternary weights.
//!
//! PV-Tuning alternates a continuous **P step** over per-group plane scales with a
//! discrete **V step** over ternary codes. V selects a bounded subspace by Adam
//! proposal magnitude and projects only those units onto the exact deployed
//! representation. No latent dense weight or residual survives between steps.
//!
//! This CPU implementation is a deterministic conformance oracle. Dense code units
//! use exact enumeration across all planes. Structured 3:4 units use deterministic
//! conditional exact search, one plane at a time, preserving each plane's invariant.

mod blockwise;
mod checkpoint;
mod config;
mod continuous;
mod error;
mod identity;
mod journal_io;
mod native_scale;
pub use native_scale::{
    NATIVE_PV_CHECKPOINT_FILE, NATIVE_PV_JOURNAL_FILE, NativeChildPublicationError,
    NativePvChildError, NativePvChildGuardError, NativePvChildLimits, NativePvChildSizeLedger,
    NativePvChildSource, NativePvChildVerification, NativePvParentReadLimits,
    NativePvResumeBundleError, NativePvResumeBundleLimits, NativePvResumeBundleReader,
    PublishedNativePvChild, PublishedNativePvResumeBundle, VerifiedNativePvChild,
    decode_native_pv_parents, inspect_native_pv_child, publish_native_pv_child,
    publish_native_pv_resume_bundle, stage_native_pv_resume_bundle, verify_native_pv_child,
};
pub use native_scale::{
    NativePvCandidate, NativePvContribution, NativePvOwnerStep, NativePvRefinementSession,
    NativePvSizeLedger, NativePvStageError, NativePvStepStats,
};
pub use native_scale::{
    NativePvJournalError, NativePvJournalLimits, NativePvJournalResume, NativePvJournalSizeLedger,
    NativePvJournalSummary, NativePvJournalWriter, NativePvResumeTail, VerifiedNativePvReplay,
    replay_native_pv_journal, replay_native_pv_journal_with_state, resume_native_pv_journal,
};
pub use native_scale::{
    NativePvPagedChildVerification, NativePvPagedChildVerificationError,
    NativePvPagedChildVerificationLimits, verify_native_pv_paged_child,
};
pub use native_scale::{
    NativePvPagedError, NativePvPagedLimits, NativePvPagedSession, NativePvPagedSizeLedger,
};
pub use native_scale::{
    NativePvPagedJournalReplay, NativePvPagedJournalResume, NativePvPagedReplayError,
    NativePvPagedReplayLimits, replay_native_pv_paged_journal, resume_native_pv_paged_journal,
};
pub use native_scale::{
    NativePvPagedJournalWriter, NativePvPagedRecordingError, NativePvPagedRecordingLimits,
    NativePvPagedRecordingSizeLedger, NativePvPagedUpdate,
};
pub use native_scale::{
    NativePvPagedResumeBundleError, NativePvPagedResumeBundleLimits,
    NativePvPagedResumeBundleReader, PublishedNativePvPagedResumeBundle,
    publish_native_pv_paged_resume_bundle, stage_native_pv_paged_resume_bundle,
};
pub use native_scale::{
    NativeRefinementOwnerReadLimits, NativeRefinementOwnerSource, NativeRefinementOwnerSourceError,
};
pub use native_scale::{
    NativeScaleResumeBundleError, NativeScaleResumeBundleLimits, NativeScaleResumeBundleReader,
    PublishedNativeScaleResumeBundle, publish_native_scale_resume_bundle,
    stage_native_scale_resume_bundle,
};
mod projection;
mod receipt;
mod representation;
mod selection;
mod session;
mod size;
mod wire;
pub use native_scale::{
    NativeScaleChildVerification, NativeScaleParentReadLimits, VerifiedNativeScaleChild,
    decode_native_scale_parents, verify_native_scale_child,
};

pub use blockwise::PvBlockwiseCursor;
pub use config::{PvTuningConfig, PvTuningConfigBuilder};
pub use error::PvTuningError;
pub use native_scale::{
    NATIVE_SCALE_CHECKPOINT_FILE, NATIVE_SCALE_JOURNAL_FILE, NativeScaleChildError,
    NativeScaleChildGuardError, NativeScaleChildLimits, NativeScaleChildSizeLedger,
    NativeScaleChildSource, PublishedNativeScaleChild, inspect_native_scale_child,
    publish_native_scale_child,
};
#[cfg(feature = "native-cpu")]
pub use native_scale::{
    NativeCpuPvError, NativeCpuPvJournalStageError, NativeCpuPvSession, NativeCpuScaleError,
    NativeCpuScaleJournalStageError, NativeCpuScaleSession,
};
pub use native_scale::{
    NativeScaleContribution, NativeScaleJournalError, NativeScaleJournalLimits,
    NativeScaleJournalResume, NativeScaleJournalSizeLedger, NativeScaleJournalSummary,
    NativeScaleJournalWriter, NativeScaleRefinementSession, NativeScaleResumeTail,
    NativeScaleSizeLedger, NativeScaleStageError, NativeScaleStep, VerifiedNativeScaleReplay,
    replay_native_scale_journal, replay_native_scale_journal_with_state,
    resume_native_scale_journal,
};
pub use receipt::PvStepReceipt;
pub use representation::{PvTernaryPlane, PvTernaryStructure, PvTernaryWeight};
pub use size::PvTuningSizeLedger;

use crate::optim::AdamState;

/// Resumable reference optimizer over one exact deployed ternary weight.
///
/// Production 27B/32B campaigns should use device/block adapters conforming to this
/// oracle; transactional reference steps intentionally favor auditability over scale.
#[derive(Clone, Debug)]
pub struct PvTuningSession {
    parent_digest: [u8; 32],
    config: PvTuningConfig,
    weight: PvTernaryWeight,
    scale_state: AdamState,
    code_state: AdamState,
    completed_step: u64,
    blockwise: Option<PvBlockwiseState>,
}

#[derive(Clone, Debug)]
struct PvBlockwiseState {
    optimizer_step: u64,
    max_block_elements: usize,
    next_offset: usize,
    scale_gradient: Vec<f64>,
}
