//! Candidate complete model conversion requests and observations (ADR 0058).

use crate::{
    BlobId, DirectFitMetric, DirectFitPlan, ModelManifestError, NamedAssetDigest, ScalePrecision,
    TensorMeta,
};
use alloc::{string::String, vec::Vec};

/// Original tensor interpretation, never an adapter-supplied content hash.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantSourceTensor {
    /// Exact logical tensor name.
    pub name: String,
    /// Positive row-major dimensions in the original source.
    pub shape: Vec<u64>,
    /// Original stored float precision, before widening or fitting.
    pub precision: ScalePrecision,
}

impl QuantSourceTensor {
    /// Validate interpretation and compute exact original payload bytes.
    /// # Errors
    /// Invalid name/shape/precision or overflowing dimensions.
    pub fn byte_len(&self) -> Result<u64, QuantPlanError> {
        crate::model::name(&self.name, false).map_err(QuantPlanError::Manifest)?;
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(QuantPlanError::InvalidSourceShape);
        }
        let width: u64 = match self.precision {
            ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
            ScalePrecision::F32 => 4,
        };
        self.shape
            .iter()
            .try_fold(width, |n, &d| n.checked_mul(d))
            .ok_or(QuantPlanError::InvalidSourceShape)
    }
}

/// Requested transform; preservation does not claim ternary weights.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum QuantAction {
    /// Preserve exact source bytes and original precision in every model level.
    KeepDense,
    /// Independently fit all requested levels from this original source.
    Fit {
        /// Existing per-tensor production fitting request.
        plan: DirectFitPlan,
        /// Explicit identity or exact expected original f64 Gram content.
        metric: DirectFitMetric,
    },
}

/// One exact source binding and its requested operation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantTensorPlan {
    /// Original source interpretation must match discovery exactly.
    pub source: QuantSourceTensor,
    /// Preservation or the actual numerical profile/metric selection.
    pub action: QuantAction,
}

/// Logical byte bounds, not complete memory/RSS or solver admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantLimits {
    /// Sum of actual original tensor payload bytes.
    pub max_source_bytes: u64,
    /// Sum of original assets retained in the output.
    pub max_asset_bytes: u64,
    /// Sum of unique encoded output blob payload bytes.
    pub max_retained_blob_bytes: u64,
    /// Complete output package bound, enforced before writer allocation.
    pub max_package_bytes: u64,
}

/// Complete candidate model request, not a frozen recipe or quality claim.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantPlan {
    /// Exact architecture identifier; the NN loader owns execution admission.
    pub architecture: String,
    /// Complete sorted unique source inventory and operations.
    pub tensors: Vec<QuantTensorPlan>,
    /// Complete sorted unique asset basenames, including config/tokenizer.
    pub assets: Vec<String>,
    /// Complete sorted unique quality/size level labels.
    pub levels: Vec<String>,
    /// Explicit logical resource limits.
    pub limits: QuantLimits,
}

/// Structural request failure; numerical profile admission belongs to the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum QuantPlanError {
    /// Reuses existing canonical name/inventory/geometry errors.
    Manifest(ModelManifestError),
    /// Empty/zero/overflowing original source shape.
    InvalidSourceShape,
    /// Empty or excessive inventory/level/asset count.
    InvalidCount,
    /// Matrix layout or complete per-tensor levels disagree with the model plan.
    FitBinding,
    /// Required config/tokenizer asset was omitted.
    MissingAsset,
}

impl core::fmt::Display for QuantPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "quant plan rejected: {self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for QuantPlanError {}

impl QuantPlan {
    /// Check complete named inventories and source/fit geometry without I/O.
    /// # Errors
    /// Invalid structure; success is not numerical/model/release admission.
    pub fn validate(&self) -> Result<(), QuantPlanError> {
        use QuantPlanError as E;
        crate::model::name(&self.architecture, false).map_err(E::Manifest)?;
        if self.tensors.is_empty()
            || self.tensors.len() > 1_000_000
            || self.levels.is_empty()
            || self.levels.len() > 1024
            || self.assets.len() > 4096
        {
            return Err(E::InvalidCount);
        }
        crate::model::sorted(
            self.tensors.iter().map(|t| t.source.name.as_str()),
            "quant tensors",
        )
        .map_err(E::Manifest)?;
        crate::model::sorted(self.assets.iter().map(String::as_str), "quant assets")
            .map_err(E::Manifest)?;
        crate::model::sorted(self.levels.iter().map(String::as_str), "quant levels")
            .map_err(E::Manifest)?;
        for name in &self.assets {
            crate::model::name(name, true).map_err(E::Manifest)?;
        }
        for name in &self.levels {
            crate::model::name(name, false).map_err(E::Manifest)?;
        }
        if !["config.json", "tokenizer.json"]
            .iter()
            .all(|required| self.assets.iter().any(|n| n == required))
        {
            return Err(E::MissingAsset);
        }
        for tensor in &self.tensors {
            tensor.source.byte_len()?;
            if let QuantAction::Fit { plan, .. } = &tensor.action {
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
                if names != self.levels.iter().map(String::as_str).collect::<Vec<_>>() {
                    return Err(E::FitBinding);
                }
            }
        }
        Ok(())
    }
}

/// Actual original raw tensor-byte binding computed by the engine itself.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantSourceBinding {
    /// Original interpretation, not fitted layout.
    pub source: QuantSourceTensor,
    /// BLAKE3 of exact original raw bytes; not original container/HF identity.
    pub content: BlobId,
}

/// Complete output bindings before provenance is added to the package manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantOutputLevel {
    /// Global level label.
    pub name: String,
    /// Complete sorted tensor inventory with actual stored blob identities.
    pub tensors: Vec<TensorMeta>,
}

/// Actual engine request observation, not proof of successful fitting.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantizationStarted {
    /// Executed complete model plan.
    pub plan: QuantPlan,
}

/// Successful complete fit-stage observation; publication/release is separate.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct QuantizationCompleted {
    /// Actual complete original tensor bindings.
    pub sources: Vec<QuantSourceBinding>,
    /// Actual complete original asset-byte bindings.
    pub assets: Vec<NamedAssetDigest>,
    /// Actual complete per-level encoded output bindings, without circular roots.
    pub levels: Vec<QuantOutputLevel>,
}
