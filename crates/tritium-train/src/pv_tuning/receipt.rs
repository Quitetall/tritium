/// Deterministic evidence for one alternating P/V step.
#[derive(Clone, Debug, PartialEq)]
pub struct PvStepReceipt {
    pub(super) optimizer_step: u64,
    pub(super) selected_code_units: usize,
    pub(super) changed_code_units: usize,
    pub(super) trust_limited_code_units: usize,
    pub(super) changed_scales: usize,
    pub(super) v_surrogate_before: f64,
    pub(super) v_surrogate_after: f64,
    pub(super) relative_code_change: f64,
    pub(super) representation_digest: [u8; 32],
}

impl PvStepReceipt {
    /// Completed one-based optimizer step.
    #[must_use]
    pub const fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Scalar or structured code units selected for projection.
    #[must_use]
    pub const fn selected_code_units(&self) -> usize {
        self.selected_code_units
    }

    /// Selected units whose deployed codes changed.
    #[must_use]
    pub const fn changed_code_units(&self) -> usize {
        self.changed_code_units
    }

    /// Improving candidates rejected only by trust-ratio budget.
    #[must_use]
    pub const fn trust_limited_code_units(&self) -> usize {
        self.trust_limited_code_units
    }

    /// f16 scale bit patterns changed by P step.
    #[must_use]
    pub const fn changed_scales(&self) -> usize {
        self.changed_scales
    }

    /// Squared-distance projection surrogate before V step.
    #[must_use]
    pub const fn v_surrogate_before(&self) -> f64 {
        self.v_surrogate_before
    }

    /// Squared-distance projection surrogate after V step; never greater than before.
    #[must_use]
    pub const fn v_surrogate_after(&self) -> f64 {
        self.v_surrogate_after
    }

    /// L2 code-movement norm divided by post-P representation norm.
    #[must_use]
    pub const fn relative_code_change(&self) -> f64 {
        self.relative_code_change
    }

    /// Digest of exact post-step deployed representation.
    #[must_use]
    pub const fn representation_digest(&self) -> [u8; 32] {
        self.representation_digest
    }
}
