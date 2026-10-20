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
    /// Complete observed inventories, ordered scales or output interpretations disagree.
    Observation,
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
            let values = pair_geometry(
                |name| self.tensor(name),
                &op.projections,
                &op.inverse,
                op.inverse_axis,
                op.moments.channels,
            )?;
            touched.insert(op.inverse.as_str());
            for name in &op.projections {
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
// One geometry implementation for transform and original-capture requests.
// This checks interpretations, not whether a pair exists in an executable graph.
pub(crate) fn pair_geometry<'a>(
    lookup: impl Fn(&str) -> Result<&'a QuantSourceTensor, SalienceTransformPlanError>,
    projections: &[String],
    inverse: &str,
    axis: SalienceInverseAxis,
    channels: u64,
) -> Result<u64, SalienceTransformPlanError> {
    use SalienceTransformPlanError as E;
    let partner = lookup(inverse)?;
    let valid = match axis {
        SalienceInverseAxis::NormChannels => partner.shape == [channels],
        SalienceInverseAxis::LinearOutputRows => {
            partner.shape.len() == 2 && partner.shape[0] == channels
        }
    };
    if channels == 0 || projections.is_empty() || !valid {
        return Err(E::Pair);
    }
    let mut members = BTreeSet::from([inverse]);
    let mut values = elements(partner)?;
    for name in projections {
        let projection = lookup(name)?;
        if !members.insert(name.as_str())
            || projection.shape.len() != 2
            || projection.shape[1] != channels
        {
            return Err(E::Pair);
        }
        values = values.checked_add(elements(projection)?).ok_or(E::Limit)?;
    }
    Ok(values)
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

impl SalienceTransformObservation {
    /// Check complete recorded structure, not actual byte or arithmetic truth.
    /// Identities must additionally be derived by the shared evidence module.
    /// # Errors
    /// Invalid request, incomplete/changed inventories, output shape/dtype,
    /// unchanged tensor content or ordered finite-positive scale bindings.
    pub fn validate(&self) -> Result<(), SalienceTransformPlanError> {
        self.plan.validate()?;
        let invalid = SalienceTransformPlanError::Observation;
        if self.original_source != self.plan.original_source
            || self
                .original_tensors
                .iter()
                .map(|t| &t.source)
                .ne(self.plan.tensors.iter())
            || self
                .assets
                .iter()
                .map(|a| &a.name)
                .ne(self.plan.assets.iter())
            || self.derived_tensors.len() != self.original_tensors.len()
            || self.operations.len() != self.plan.operations.len()
        {
            return Err(invalid);
        }
        let mut touched = BTreeSet::new();
        for (request, output) in self.plan.operations.iter().zip(&self.operations) {
            if output.operation != request.name
                || output.scale_bits.len() as u64 != request.moments.channels
                || output.scale_bits.iter().any(|bits| {
                    let scale = f32::from_le_bytes(*bits);
                    !scale.is_finite() || scale <= 0.0
                })
            {
                return Err(invalid);
            }
            touched.insert(request.inverse.as_str());
            touched.extend(request.projections.iter().map(String::as_str));
        }
        for (original, derived) in self.original_tensors.iter().zip(&self.derived_tensors) {
            let mut expected = original.source.clone();
            if touched.contains(expected.name.as_str()) {
                expected.precision = crate::ScalePrecision::F32;
            } else if original.content != derived.content {
                return Err(invalid);
            }
            if derived.source != expected {
                return Err(invalid);
            }
        }
        Ok(())
    }
}

/// Typed transform request event; expected A is not yet an observed-byte claim.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformStarted {
    /// Complete declared request, including logical limits and statistic class.
    pub plan: SalienceTransformPlan,
}

/// Actual complete transform event; not publication or captured-graph admission.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceTransformCompleted {
    /// Actual complete A/B raw bindings and ordered stored scales.
    pub observation: SalienceTransformObservation,
}
