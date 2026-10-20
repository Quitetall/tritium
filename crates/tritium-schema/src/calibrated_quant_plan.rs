//! Complete calibrated model request and common borrowed traversal (ADR 0062).
use crate::{
    CalibratedFitPlan, DirectFitMetric, DirectFitPlan, OriginalSourceId, QuantAction, QuantLimits,
    QuantPlan, QuantPlanError, QuantSourceTensor, QuantizationCompleted,
};
use alloc::{boxed::Box, string::String, vec::Vec};

/// Explicit transform in the separate calibrated model lifecycle.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub enum CalibratedQuantAction {
    /// Preserve exact original floating-point payloads.
    KeepDense,
    /// Existing explicitly uncalibrated Identity/raw-Gram direct controller.
    Fit {
        /// Actual controller request, not a calibrated placeholder.
        plan: Box<DirectFitPlan>,
        /// Actual explicit original metric request.
        metric: DirectFitMetric,
    },
    /// Fit original rows using the actual immutable bound calibration cache.
    CalibratedFit {
        /// Actual expected cache context and row-aware controller request.
        plan: Box<CalibratedFitPlan>,
    },
}
/// One original interpretation and its actual requested operation.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedQuantTensorPlan {
    /// Exact complete original source interpretation.
    pub source: QuantSourceTensor,
    /// Explicit preservation, plain fit or calibrated fit.
    pub action: CalibratedQuantAction,
}
/// Separate complete calibrated model request; never reinterprets a v1 metric.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedQuantPlan {
    /// Architecture identifier; real execution admission belongs to the loader.
    pub architecture: String,
    /// Actual preparation identity required of consumed original bytes/assets.
    pub original_source: OriginalSourceId,
    /// Complete sorted unique original inventory with explicit operations.
    pub tensors: Vec<CalibratedQuantTensorPlan>,
    /// Complete sorted original asset basenames.
    pub assets: Vec<String>,
    /// Complete sorted level names.
    pub levels: Vec<String>,
    /// Existing source/asset/encoded-output/package logical bounds.
    pub limits: QuantLimits,
    /// Sum of expected actual captured payload bytes, not process RSS.
    pub max_cache_payload_bytes: u64,
    /// Physical .trit input bound for each file-backed calibration cache.
    pub max_cache_package_bytes: u64,
}
impl CalibratedQuantPlan {
    /// Check complete inventories, original/cache context, geometry and levels.
    /// Numerical fitting, actual source/cache bytes and total RSS remain separate.
    /// # Errors
    /// Invalid structure, missing calibrated operation or inconsistent cache/model binding.
    pub fn validate(&self) -> Result<(), QuantPlanError> {
        ModelQuantPlan::Calibrated(self).validate()
    }
}

/// Actual calibrated-model request, a distinct registered observation class.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedQuantizationStarted {
    /// Complete actual typed request, including original-source/cache expectation.
    pub plan: CalibratedQuantPlan,
}
/// Successful calibrated fitting, not publication, quality or release acceptance.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedQuantizationCompleted {
    /// Independently derived identity of actual consumed original bytes/assets.
    pub original_source: OriginalSourceId,
    /// Shared complete output/source vocabulary, not an emitted v1 completion.
    pub output: QuantizationCompleted,
}

/// Rust-only borrowed traversal of two real model request classes, not wire data.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ModelQuantPlan<'a> {
    /// Existing v1 request retains its explicit Identity/raw-Gram semantics.
    Uncalibrated(&'a QuantPlan),
    /// Separate calibrated model lifecycle, with mandatory original source equality.
    Calibrated(&'a CalibratedQuantPlan),
    /// Separate owner-window model lifecycle, optionally mixing legacy fits.
    Owner(&'a crate::OwnerQuantPlan),
}
/// Borrowed actual operation; calibrated work is never mapped to Identity.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum QuantActionRef<'a> {
    /// Exact original payload preservation.
    KeepDense,
    /// Existing plain controller and its explicit original metric.
    Fit {
        /// Actual requested direct controller.
        plan: &'a DirectFitPlan,
        /// Actual requested metric.
        metric: &'a DirectFitMetric,
    },
    /// Actual cache-bound calibrated controller request.
    CalibratedFit(&'a CalibratedFitPlan),
    /// Actual complete scalar-window owner fit request.
    OwnerFit(&'a crate::OwnerScoreFitPlan),
}
impl<'a> QuantActionRef<'a> {
    /// Original-row controller for either real fit class; absent for preservation.
    #[must_use]
    pub fn direct(self) -> Option<&'a DirectFitPlan> {
        match self {
            Self::KeepDense => None,
            Self::Fit { plan, .. } => Some(plan),
            Self::CalibratedFit(plan) => Some(&plan.direct),
            Self::OwnerFit(plan) => Some(&plan.direct),
        }
    }
}
/// Borrowed original interpretation with its actual operation.
#[derive(Clone, Copy, Debug)]
pub struct QuantTensorRef<'a> {
    /// Actual original source interpretation.
    pub source: &'a QuantSourceTensor,
    /// Actual operation, never a reserialized projection.
    pub action: QuantActionRef<'a>,
}
impl<'a> ModelQuantPlan<'a> {
    /// Actual architecture identifier.
    #[must_use]
    pub fn architecture(self) -> &'a str {
        match self {
            Self::Uncalibrated(p) => &p.architecture,
            Self::Calibrated(p) => &p.architecture,
            Self::Owner(p) => &p.architecture,
        }
    }
    /// Actual complete original asset inventory.
    #[must_use]
    pub fn assets(self) -> &'a [String] {
        match self {
            Self::Uncalibrated(p) => &p.assets,
            Self::Calibrated(p) => &p.assets,
            Self::Owner(p) => &p.assets,
        }
    }
    /// Actual complete level inventory.
    #[must_use]
    pub fn levels(self) -> &'a [String] {
        match self {
            Self::Uncalibrated(p) => &p.levels,
            Self::Calibrated(p) => &p.levels,
            Self::Owner(p) => &p.levels,
        }
    }
    /// Actual existing logical output/source bounds.
    #[must_use]
    pub fn limits(self) -> QuantLimits {
        match self {
            Self::Uncalibrated(p) => p.limits,
            Self::Calibrated(p) => p.limits,
            Self::Owner(p) => p.limits,
        }
    }
    /// Mandatory expected original identity for actual calibrated lifecycles.
    pub fn expected_source(self) -> Option<OriginalSourceId> {
        match self {
            Self::Uncalibrated(_) => None,
            Self::Calibrated(p) => Some(p.original_source),
            Self::Owner(p) => Some(p.original_source),
        }
    }
    /// Aggregate logical payload and per-file physical cache limits, if calibrated.
    pub fn cache_limits(self) -> Option<(u64, u64)> {
        match self {
            Self::Uncalibrated(_) => None,
            Self::Calibrated(p) => Some((p.max_cache_payload_bytes, p.max_cache_package_bytes)),
            Self::Owner(p) => Some((p.max_cache_payload_bytes, p.max_cache_package_bytes)),
        }
    }
    /// Complete inventory traversal without allocating or changing a recipe.
    pub fn tensors(self) -> impl Iterator<Item = QuantTensorRef<'a>> {
        let (plain, calibrated, owner) = match self {
            Self::Uncalibrated(p) => (Some(p), None, None),
            Self::Calibrated(p) => (None, Some(p), None),
            Self::Owner(p) => (None, None, Some(p)),
        };
        plain
            .into_iter()
            .flat_map(|p| &p.tensors)
            .map(|t| QuantTensorRef {
                source: &t.source,
                action: match &t.action {
                    QuantAction::KeepDense => QuantActionRef::KeepDense,
                    QuantAction::Fit { plan, metric } => QuantActionRef::Fit { plan, metric },
                },
            })
            .chain(
                calibrated
                    .into_iter()
                    .flat_map(|p| &p.tensors)
                    .map(|t| QuantTensorRef {
                        source: &t.source,
                        action: match &t.action {
                            CalibratedQuantAction::KeepDense => QuantActionRef::KeepDense,
                            CalibratedQuantAction::Fit { plan, metric } => {
                                QuantActionRef::Fit { plan, metric }
                            }
                            CalibratedQuantAction::CalibratedFit { plan } => {
                                QuantActionRef::CalibratedFit(plan)
                            }
                        },
                    }),
            )
            .chain(
                owner
                    .into_iter()
                    .flat_map(|p| &p.tensors)
                    .map(|t| QuantTensorRef {
                        source: &t.source,
                        action: t.action.borrowed(),
                    }),
            )
    }
    /// Shared structural validation; does not prove bytes, numerics or full RSS.
    /// # Errors
    /// Invalid original inventories/geometry/levels or calibrated context.
    pub fn validate(self) -> Result<(), QuantPlanError> {
        use QuantPlanError as E;
        crate::model::name(self.architecture(), false).map_err(E::Manifest)?;
        let count = self.tensors().count();
        if count == 0
            || count > 1_000_000
            || self.levels().is_empty()
            || self.levels().len() > 1024
            || self.assets().len() > 4096
        {
            return Err(E::InvalidCount);
        }
        crate::model::sorted(
            self.tensors().map(|t| t.source.name.as_str()),
            "quant tensors",
        )
        .map_err(E::Manifest)?;
        crate::model::sorted(self.assets().iter().map(String::as_str), "quant assets")
            .map_err(E::Manifest)?;
        crate::model::sorted(self.levels().iter().map(String::as_str), "quant levels")
            .map_err(E::Manifest)?;
        for name in self.assets() {
            crate::model::name(name, true).map_err(E::Manifest)?;
        }
        for name in self.levels() {
            crate::model::name(name, false).map_err(E::Manifest)?;
        }
        if !["config.json", "tokenizer.json"]
            .iter()
            .all(|required| self.assets().iter().any(|n| n == required))
        {
            return Err(E::MissingAsset);
        }
        let mut calibrated_count = 0;
        let mut owner_count = 0;
        let mut cache_bytes = 0u64;
        for tensor in self.tensors() {
            tensor.source.byte_len()?;
            if let Some(plan) = tensor.action.direct() {
                if tensor.source.shape.as_slice() != [plan.layout.rows, plan.layout.cols] {
                    return Err(E::FitBinding);
                }
                plan.layout.validate().map_err(|_| E::FitBinding)?;
                let mut names = plan
                    .levels
                    .iter()
                    .map(|l| l.name.as_str())
                    .collect::<Vec<_>>();
                names.sort_unstable();
                if names != self.levels().iter().map(String::as_str).collect::<Vec<_>>() {
                    return Err(E::FitBinding);
                }
            }
            if let QuantActionRef::CalibratedFit(fit) = tensor.action {
                fit.calibration_plan().map_err(|_| E::CalibrationBinding)?;
                if fit.cache.request.tensor != *tensor.source
                    || Some(fit.cache.request.model) != self.expected_source()
                {
                    return Err(E::CalibrationBinding);
                }
                calibrated_count += 1;
                cache_bytes = cache_bytes
                    .checked_add(fit.cache.payload_bytes)
                    .ok_or(E::CalibrationBinding)?;
            }
            if let QuantActionRef::OwnerFit(fit) = tensor.action {
                if !matches!(self, Self::Owner(_)) {
                    return Err(E::CalibrationBinding);
                }
                fit.validate().map_err(|_| E::CalibrationBinding)?;
                if fit.windows.iter().any(|w| {
                    w.binding.content.tensor != *tensor.source
                        || Some(w.binding.content.source) != self.expected_source()
                }) {
                    return Err(E::CalibrationBinding);
                }
                owner_count += 1;
                for window in &fit.windows {
                    cache_bytes = cache_bytes
                        .checked_add(window.binding.payload_bytes)
                        .ok_or(E::CalibrationBinding)?;
                }
            }
        }
        if let Some((payload, package)) = self.cache_limits()
            && (self
                .expected_source()
                .is_none_or(|id| id.as_bytes() == &[0; 32])
                || payload == 0
                || package == 0
                || cache_bytes > payload
                || matches!(self, Self::Calibrated(_)) && calibrated_count == 0
                || matches!(self, Self::Owner(_)) && owner_count == 0)
        {
            return Err(E::CalibrationBinding);
        }
        Ok(())
    }
}
