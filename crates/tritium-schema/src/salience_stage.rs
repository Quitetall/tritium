//! Mandatory durable derived-source profile; structure is not numerical truth.
use crate::{
    EvidenceReference, SalienceTransformObservation, SalienceTransformPlanError, SchemaId,
};
use alloc::{format, string::String};

/// Reserved internal moment namespace; never part of the original B assets.
pub const SALIENCE_MOMENT_ASSET_PREFIX: &str = "__tritium_salience_moments_";

/// Exact internal moment basename for one operation in declared order.
#[must_use]
pub fn salience_moment_asset(index: usize) -> String {
    format!("{SALIENCE_MOMENT_ASSET_PREFIX}{index:08}.bin")
}

/// Receiving/producer logical byte limits, not complete process RSS admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceStageLimits {
    /// Complete physical package, including header, directory, metadata and padding.
    pub max_package_bytes: u64,
    /// Canonical encoded manifest, before typed metadata allocation.
    pub max_manifest_bytes: u64,
    /// Complete derived dense tensor payload, excluding dedup savings.
    pub max_tensor_bytes: u64,
    /// Complete original asset payload, excluding internal statistics.
    pub max_asset_bytes: u64,
    /// Complete raw f64 moment payload, excluding dedup savings.
    pub max_statistics_bytes: u64,
}

/// Mandatory typed B stage manifest, distinct from a fitted model manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceSourceStageManifest {
    /// The only admitted candidate metadata version is 1.0.
    pub schema: SchemaId,
    /// Mandatory actual standalone transform run/span/root.
    pub transform: EvidenceReference,
    /// Complete observation; agreement with log/actual bytes is checked separately.
    pub observation: SalienceTransformObservation,
}
impl SalienceSourceStageManifest {
    /// Check structure and reserved names, not recorded identity/numerical truth.
    /// # Errors
    /// Unsupported version, bad reference or invalid complete observation.
    pub fn validate(&self) -> Result<(), SalienceTransformPlanError> {
        use SalienceTransformPlanError as E;
        if self.schema != (SchemaId { major: 1, minor: 0 }) {
            return Err(E::Observation);
        }
        self.observation.validate()?;
        for name in [&self.transform.run, &self.transform.span] {
            crate::model::name(name, false).map_err(|_| E::Observation)?;
        }
        if self
            .observation
            .assets
            .iter()
            .any(|a| a.name.starts_with(SALIENCE_MOMENT_ASSET_PREFIX))
        {
            return Err(E::Observation);
        }
        Ok(())
    }
    /// Bound declared logical tensor/statistic sizes before payload operations.
    /// Asset sizes and encoded metadata/package bytes require actual payload checks.
    /// # Errors
    /// Invalid structure, zero receiving caps, overflow or an insufficient limit.
    pub fn validate_limits(
        &self,
        limits: SalienceStageLimits,
    ) -> Result<(), SalienceTransformPlanError> {
        use SalienceTransformPlanError as E;
        self.validate()?;
        if [
            limits.max_package_bytes,
            limits.max_manifest_bytes,
            limits.max_tensor_bytes,
            limits.max_asset_bytes,
            limits.max_statistics_bytes,
        ]
        .contains(&0)
        {
            return Err(E::Limit);
        }
        let mut tensors = 0u64;
        for t in &self.observation.derived_tensors {
            tensors = tensors
                .checked_add(t.source.byte_len().map_err(E::Source)?)
                .ok_or(E::Limit)?;
        }
        let mut statistics = 0u64;
        for op in &self.observation.plan.operations {
            statistics = statistics
                .checked_add(op.moments.channels.checked_mul(8).ok_or(E::Limit)?)
                .ok_or(E::Limit)?;
        }
        if tensors > limits.max_tensor_bytes || statistics > limits.max_statistics_bytes {
            return Err(E::Limit);
        }
        Ok(())
    }
}

/// Distinct mandatory original-capture ancestry, not a numerical qualification.
/// Old Declared lineage is preserved, never silently promoted or unwrapped.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalCapturedSalienceSourceStageManifest {
    /// Complete unchanged transform/source/moment vocabulary.
    pub lineage: SalienceSourceStageManifest,
    /// Required standalone original-A forward salience capture run/span/root.
    pub original_capture: EvidenceReference,
}
impl OriginalCapturedSalienceSourceStageManifest {
    /// Check strict inner version and references; actual log/payload truth is separate.
    /// # Errors
    /// Invalid lineage, missing/invalid original capture or transform reference.
    pub fn validate(&self) -> Result<(), SalienceTransformPlanError> {
        self.lineage.validate()?;
        for name in [&self.original_capture.run, &self.original_capture.span] {
            crate::model::name(name, false).map_err(|_| SalienceTransformPlanError::Observation)?;
        }
        if self.original_capture.root == [0; 32] || self.lineage.transform.root == [0; 32] {
            return Err(SalienceTransformPlanError::Observation);
        }
        Ok(())
    }
    /// The same complete stage payload caps, not process RSS or log allocation admission.
    /// # Errors
    /// Invalid structure, zero caps, overflow or insufficient stage limits.
    pub fn validate_limits(
        &self,
        limits: SalienceStageLimits,
    ) -> Result<(), SalienceTransformPlanError> {
        self.validate()?;
        self.lineage.validate_limits(limits)
    }
}
