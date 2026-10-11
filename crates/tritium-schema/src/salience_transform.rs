//! Declared-statistics paired source transformation candidate (ADR0064).
//! Structural admission does not prove graph eligibility or executed capture.

use crate::{
    BlobId, NamedAssetDigest, OriginalSourceId, QuantPlanError, QuantSourceBinding,
    QuantSourceTensor,
};
use alloc::{collections::BTreeSet, string::String, vec::Vec};

/// Explicit axis of the graph partner divided by each stored scale.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum SalienceInverseAxis {
    /// Rank-one preceding norm, one channel per projection input column.
    NormChannels,
    /// Rank-two linear partner, one output row per projection input column.
    LinearOutputRows,
}

/// Ordered numerical semantics, never an implicit recipe fallback.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum SalienceNumerics {
    /// Legacy ordered f64 RMS/log-mean/powf, f32 cast and multiply/divide.
    LegacyF64ScaleF32V1,
}

/// What the statistics observation actually establishes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum SalienceStatisticsKind {
    /// Supplied moments/rows only; no claim of executed graph/token/teacher capture.
    Declared,
}

/// Exact actual moment content plus explicitly declared interpretation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceMomentBinding {
    /// BLAKE3 of row-major little-endian f64 second-moment sums, not an activation hash.
    pub content: BlobId,
    /// One moment per projection input column and partner output channel.
    pub channels: u64,
    /// Declared positive sample count; not proof of observed row execution.
    pub rows: u64,
}

/// One ordered, named paired operation; names do not prove graph eligibility.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformOperation {
    /// Unique logical operation name, used to select the actual statistics.
    pub name: String,
    /// Actual matrices whose input columns are multiplied, in declared order.
    pub projections: Vec<String>,
    /// Distinct norm/linear partner whose channels/output rows are divided.
    pub inverse: String,
    /// Explicit partner interpretation; no inference from tensor naming.
    pub inverse_axis: SalienceInverseAxis,
    /// Exact expected statistics and declared count interpretation.
    pub moments: SalienceMomentBinding,
    /// Exact little-endian f64 alpha bits, including signed zero.
    pub alpha_bits: [u8; 8],
}

/// Logical payload/numeric-vector budgets, not process RSS admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformLimits {
    /// Complete consumed original tensor payload total.
    pub max_source_bytes: u64,
    /// Complete unchanged asset payload total.
    pub max_asset_bytes: u64,
    /// Complete derived tensor payload total, including touched tensors widened to F32.
    pub max_derived_bytes: u64,
    /// Complete supplied moment vectors, in raw f64 bytes.
    pub max_statistics_bytes: u64,
    /// Pair decoded vectors, replacement bytes and RMS/scales; excludes metadata and RSS.
    pub max_pair_working_bytes: u64,
}

/// Candidate complete source-stage request, not a captured-statistics qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformPlan {
    /// Original source architecture identifier; execution admission is separate.
    pub architecture: String,
    /// Observed expected A identity; the controller must derive its own consumed identity.
    pub original_source: OriginalSourceId,
    /// Complete strictly sorted original tensor interpretations.
    pub tensors: Vec<QuantSourceTensor>,
    /// Complete strictly sorted unchanged asset basenames.
    pub assets: Vec<String>,
    /// Order is significant, e.g. up input columns before up output rows.
    pub operations: Vec<SalienceTransformOperation>,
    /// Explicit ordered numerical policy.
    pub numerics: SalienceNumerics,
    /// Initial profile must stay Declared until an executed capture is admitted.
    pub statistics_kind: SalienceStatisticsKind,
    /// Explicit logical bounds; no out-of-core/full-memory claim follows.
    pub limits: SalienceTransformLimits,
}

/// Structural or logical resource rejection before source-stage execution.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SalienceTransformPlanError {
    /// Existing canonical inventory/name/shape rejection.
    Source(QuantPlanError),
    /// Empty/excessive or ambiguous named operation inventory.
    Operations,
    /// Missing, aliased or geometrically incompatible pair member.
    Pair,
    /// Empty/overflowing declared statistics or invalid alpha bits.
    Statistics,
    /// Declared source/derived/statistics/pair working bound is insufficient.
    Limit,
}
impl core::fmt::Display for SalienceTransformPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "salience transform plan rejected: {self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for SalienceTransformPlanError {}

impl SalienceTransformPlan {
    /// Validate canonical inventories, ordered pair geometry and logical budgets.
    /// # Errors
    /// Invalid names/shapes/aliases/statistics, overflow or insufficient limits.
    pub fn validate(&self) -> Result<(), SalienceTransformPlanError> {
        use SalienceTransformPlanError as E;
        let manifest = |e| E::Source(QuantPlanError::Manifest(e));
        crate::model::name(&self.architecture, false).map_err(manifest)?;
        if self.tensors.is_empty() || self.tensors.len() > 1_000_000 || self.assets.len() > 4096 {
            return Err(E::Source(QuantPlanError::InvalidCount));
        }
        crate::model::sorted(
            self.tensors.iter().map(|t| t.name.as_str()),
            "transform tensors",
        )
        .map_err(manifest)?;
        crate::model::sorted(self.assets.iter().map(String::as_str), "transform assets")
            .map_err(manifest)?;
        for asset in &self.assets {
            crate::model::name(asset, true).map_err(manifest)?;
        }
        if self.operations.is_empty() || self.operations.len() > 1_000_000 {
            return Err(E::Operations);
        }
        let mut names = BTreeSet::new();
        let mut touched = BTreeSet::new();
        let mut statistics = 0u64;
        for op in &self.operations {
            crate::model::name(&op.name, false).map_err(manifest)?;
            if !names.insert(op.name.as_str())
                || op.projections.is_empty()
                || op.projections.len() > self.tensors.len()
            {
                return Err(E::Operations);
            }
            let alpha = f64::from_le_bytes(op.alpha_bits);
            if op.moments.channels == 0
                || op.moments.rows == 0
                || !alpha.is_finite()
                || !(0.0..=1.0).contains(&alpha)
            {
                return Err(E::Statistics);
            }
            statistics = statistics
                .checked_add(op.moments.channels.checked_mul(8).ok_or(E::Statistics)?)
                .ok_or(E::Statistics)?;
            let inverse = self.tensor(&op.inverse)?;
            let inverse_ok = match op.inverse_axis {
                SalienceInverseAxis::NormChannels => inverse.shape == [op.moments.channels],
                SalienceInverseAxis::LinearOutputRows => {
                    inverse.shape.len() == 2 && inverse.shape[0] == op.moments.channels
                }
            };
            if !inverse_ok {
                return Err(E::Pair);
            }
            let mut members = BTreeSet::from([op.inverse.as_str()]);
            let mut values = elements(inverse)?;
            touched.insert(op.inverse.as_str());
            for name in &op.projections {
                let projection = self.tensor(name)?;
                if !members.insert(name.as_str())
                    || projection.shape.len() != 2
                    || projection.shape[1] != op.moments.channels
                {
                    return Err(E::Pair);
                }
                values = values.checked_add(elements(projection)?).ok_or(E::Limit)?;
                touched.insert(name.as_str());
            }
            // Decoded f32 values + replacement encoded values + RMS/scales.
            let working = values
                .checked_mul(8)
                .and_then(|n| {
                    op.moments
                        .channels
                        .checked_mul(12)
                        .and_then(|s| n.checked_add(s))
                })
                .ok_or(E::Limit)?;
            if working > self.limits.max_pair_working_bytes {
                return Err(E::Limit);
            }
        }
        let mut source = 0u64;
        let mut derived = 0u64;
        for tensor in &self.tensors {
            let bytes = tensor.byte_len().map_err(E::Source)?;
            source = source.checked_add(bytes).ok_or(E::Limit)?;
            let output = if touched.contains(tensor.name.as_str()) {
                elements(tensor)?.checked_mul(4).ok_or(E::Limit)?
            } else {
                bytes
            };
            derived = derived.checked_add(output).ok_or(E::Limit)?;
        }
        if source > self.limits.max_source_bytes
            || derived > self.limits.max_derived_bytes
            || statistics > self.limits.max_statistics_bytes
        {
            return Err(E::Limit);
        }
        Ok(())
    }

    /// Find an exact original interpretation in the canonical inventory.
    /// # Errors
    /// The named member is absent; callers must first validate inventory order.
    pub fn tensor(&self, name: &str) -> Result<&QuantSourceTensor, SalienceTransformPlanError> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map(|i| &self.tensors[i])
            .map_err(|_| SalienceTransformPlanError::Pair)
    }
}
fn elements(tensor: &QuantSourceTensor) -> Result<u64, SalienceTransformPlanError> {
    let bytes = tensor
        .byte_len()
        .map_err(SalienceTransformPlanError::Source)?;
    Ok(bytes
        / if tensor.precision == crate::ScalePrecision::F32 {
            4
        } else {
            2
        })
}

/// Actual stored scales for one executed pair; bits are JSON-safe byte arrays.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceOperationOutput {
    /// Exact operation name in executed order.
    pub operation: String,
    /// Actual finite positive f32 scale bits, in channel order.
    pub scale_bits: Vec<[u8; 4]>,
}

/// Actual complete source-stage observation, not a sealed log or release receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformObservation {
    /// Executed request, explicitly declaring statistics rather than captured-graph truth.
    pub plan: SalienceTransformPlan,
    /// Actual consumed complete A identity.
    pub original_source: OriginalSourceId,
    /// Actual produced complete B identity, using the same raw-content domain.
    pub derived_source: OriginalSourceId,
    /// Actual complete original tensor raw-content bindings.
    pub original_tensors: Vec<QuantSourceBinding>,
    /// Actual complete derived tensor raw-content/dtype bindings.
    pub derived_tensors: Vec<QuantSourceBinding>,
    /// Exact unchanged asset content bindings.
    pub assets: Vec<NamedAssetDigest>,
    /// Actual ordered pair output scales.
    pub operations: Vec<SalienceOperationOutput>,
}
