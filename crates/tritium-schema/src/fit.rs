//! Candidate direct-level fitting vocabulary from ADR 0044 D5/D11.
//!
//! These requests are not frozen recipe identities or the complete model QuantPlan.

use crate::AdditiveLayout;
use alloc::{string::String, vec::Vec};

/// Placement of the ladder scale refit, without changing its numerical policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum LadderScaleRefit {
    /// Do not refit ladder scales.
    None,
    /// Refit during the post-feedback coordinate search.
    PostPass,
    /// Refit completed groups before propagating to the remaining columns.
    InLoop,
}

/// Shipping f32 working weights with f64 metric/feedback/search arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct LadderFitOptions {
    /// Quarter-octave anchor grid candidates per group; zero retains the legacy single guess.
    pub grid: u32,
    /// Finite, nonnegative damping relative to the mean input-Gram diagonal.
    pub damp: f64,
    /// Number of optional coordinate-search sweeps.
    pub search_sweeps: u32,
    /// Scale refit placement.
    pub scale_refit: LadderScaleRefit,
    /// Finite feedback fraction in `0..=1`; one is plain GPTQ.
    pub decay: f64,
    /// Ramp feedback from one to the configured fraction along the input axis.
    pub decay_ramp: bool,
}

/// Existing joint fitter applied independently to each row-local scale group.
///
/// This profile does not propagate cross-group residual feedback, implement K-FAC
/// output curvature, or claim to be the full SALT campaign solver.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct JointGroupFitOptions {
    /// Positive maximum alternating scale/assignment iterations.
    pub max_iterations: u32,
    /// Finite positive scale-solve ridge.
    pub ridge: f64,
    /// Positive deterministic output-aware restart count.
    pub em_restarts: u32,
    /// Finite ridge condition-number limit greater than one.
    pub ridge_condition_limit: f64,
    /// Enable the existing softened-relay initialization basin.
    pub softened_relay: bool,
    /// Enable the existing modulated-relay initialization basin.
    pub modulated_relay: bool,
}

/// Exact numerical profile, not an interchangeable precision or SOTA claim.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum DirectFitProfile {
    /// Tied one-third F16 ladder with the shipping activation-aware arithmetic.
    ActivationAwareLadder(LadderFitOptions),
    /// Free F16/F32 planes fitted independently per group with the existing joint solver.
    JointIndependentGroups(JointGroupFitOptions),
}

/// One requested direct fit; plane counts need not be sorted or nested.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct DirectFitLevel {
    /// Unique nonempty level name within this request.
    pub name: String,
    /// Positive plane count bounded by the layout and its admitted law.
    pub planes: u8,
}

/// Pre-allocation shape limits for one direct-level request.
///
/// These bound specified logical element counts, not total process RSS, solver
/// temporary storage, Rayon overhead, or complete production memory admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct DirectFitLimits {
    /// Maximum source scalar count (`rows * cols`).
    pub max_weights: u64,
    /// Maximum dense input-Gram count (`cols * cols`), even when implicit identity is used.
    pub max_metric_entries: u64,
    /// Maximum sum of retained trits over all directly fitted levels.
    pub max_output_trits: u64,
}

/// Typed per-tensor direct-level fit request, not the complete model QuantPlan.
///
/// The layout is the highest permitted plane count. Each returned level has
/// its own exact plane count as `max_planes`, and is fitted from the same
/// original source rather than from a prefix or another level's reconstruction.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct DirectFitPlan {
    /// Declared geometry, law, basis, packing codec and transport.
    pub layout: AdditiveLayout,
    /// Selected real numerical implementation.
    pub profile: DirectFitProfile,
    /// Ordered direct levels to produce.
    pub levels: Vec<DirectFitLevel>,
    /// Required explicit shape limits, checked before numerical allocations.
    pub limits: DirectFitLimits,
}
