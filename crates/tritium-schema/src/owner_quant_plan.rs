//! Distinct whole-model owner-window lifecycle, not legacy K-FAC/v1 relabeling.
use crate::{
    CalibratedFitPlan, DirectFitMetric, DirectFitPlan, ModelQuantPlan, OriginalSourceId,
    OwnerScoreFitPlan, QuantActionRef, QuantLimits, QuantPlanError, QuantSourceTensor,
    QuantizationCompleted,
};
use alloc::{boxed::Box, string::String, vec::Vec};

/// Explicit fit class in a model containing scalar-window owner calibration.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum OwnerQuantAction {
    /// Exact original floating-point preservation.
    KeepDense,
    /// Existing explicitly uncalibrated original-row fit.
    Fit {
        /// Actual common controller request.
        plan: Box<DirectFitPlan>,
        /// Explicit Identity or exact raw-Gram context.
        metric: DirectFitMetric,
    },
    /// Existing actual K-FAC/input-only cache request.
    CalibratedFit {
        /// Actual legacy calibrated context, never an owner-score substitute.
        plan: Box<CalibratedFitPlan>,
    },
    /// Actual ordered scalar-window owner cache request.
    OwnerFit {
        /// Complete immutable window bindings and numerical request.
        plan: Box<OwnerScoreFitPlan>,
    },
}
impl OwnerQuantAction {
    pub(crate) fn borrowed(&self) -> QuantActionRef<'_> {
        match self {
            Self::KeepDense => QuantActionRef::KeepDense,
            Self::Fit { plan, metric } => QuantActionRef::Fit { plan, metric },
            Self::CalibratedFit { plan } => QuantActionRef::CalibratedFit(plan),
            Self::OwnerFit { plan } => QuantActionRef::OwnerFit(plan),
        }
    }
}
/// One canonical original tensor and its actual conversion operation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerQuantTensorPlan {
    /// Exact original interpretation, before widening or fitting.
    pub source: QuantSourceTensor,
    /// Explicit preservation or actual fit class.
    pub action: OwnerQuantAction,
}
/// Complete original model request requiring at least one actual owner fit.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerQuantPlan {
    /// Architecture identifier, not loader execution admission.
    pub architecture: String,
    /// Preparation identity required of actual consumed original bytes/assets.
    pub original_source: OriginalSourceId,
    /// Complete sorted unique original tensor inventory.
    pub tensors: Vec<OwnerQuantTensorPlan>,
    /// Complete sorted original asset inventory.
    pub assets: Vec<String>,
    /// Complete sorted output level names.
    pub levels: Vec<String>,
    /// Existing original/output/package logical limits.
    pub limits: QuantLimits,
    /// Aggregate logical cache payload, counting intentional window repetitions.
    pub max_cache_payload_bytes: u64,
    /// Maximum physical input package per file-backed cache.
    pub max_cache_package_bytes: u64,
}
impl OwnerQuantPlan {
    /// Shared complete inventory/source/cache validation, not numerical or RSS proof.
    pub fn validate(&self) -> Result<(), QuantPlanError> {
        ModelQuantPlan::Owner(self).validate()
    }
}
/// Separate complete owner-model request observation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerQuantizationStarted {
    /// Actual complete request, including expected original/cache context.
    pub plan: OwnerQuantPlan,
}
/// Actual successful owner-model fitting, not publication or qualification.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OwnerQuantizationCompleted {
    /// Independently derived identity of consumed original bytes/assets.
    pub original_source: OriginalSourceId,
    /// Shared complete actual source/output vocabulary, not a v1 event.
    pub output: QuantizationCompleted,
}
