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

mod checkpoint;
mod config;
mod continuous;
mod error;
mod projection;
mod receipt;
mod representation;
mod selection;
mod session;
mod wire;

pub use config::{PvTuningConfig, PvTuningConfigBuilder};
pub use error::PvTuningError;
pub use receipt::PvStepReceipt;
pub use representation::{PvTernaryPlane, PvTernaryStructure, PvTernaryWeight};

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
}
