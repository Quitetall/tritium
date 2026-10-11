//! Mandatory A->B->C carrier; validation alone is not joined-log admission.
use crate::{
    EvidenceReference, ModelManifest, ModelManifestError, OriginalSourceId, PackageId, SchemaId,
};
use core::fmt;

/// Explicit receiving/producer byte ceilings, not complete process RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceConversionLimits {
    /// Complete physical C package, including metadata, directory and alignment.
    pub max_package_bytes: u64,
    /// Canonical encoded mandatory carrier metadata, before typed allocation.
    pub max_manifest_bytes: u64,
}
impl SalienceConversionLimits {
    /// Check explicit positive ceilings; actual byte counts are checked by the codec.
    /// # Errors
    /// Zero ceiling.
    pub fn validate(self) -> Result<(), SalienceConversionManifestError> {
        if self.max_package_bytes == 0 || self.max_manifest_bytes == 0 {
            return Err(SalienceConversionManifestError::Limit);
        }
        Ok(())
    }
}

/// Required lineage wrapped around the unchanged inner fitted model manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct SalienceConversionManifest {
    /// The only admitted candidate carrier version is 1.0.
    pub schema: SchemaId,
    /// Complete unchanged B->C model/output vocabulary.
    pub model: ModelManifest,
    /// Actual immutable B stage physical identity, not only B's raw source identity.
    pub stage_package: PackageId,
    /// Actual original complete dense A identity.
    pub original_source: OriginalSourceId,
    /// Actual complete B consumed by fitting; may equal A for identity transforms.
    pub derived_source: OriginalSourceId,
    /// Actual complete mandatory transform run/span/root.
    pub transform: EvidenceReference,
    /// Actual complete B->C conversion run/span/run-root.
    pub conversion: EvidenceReference,
}
impl SalienceConversionManifest {
    /// Check typed structure/version only; actual B/log/payload truth is separate.
    /// # Errors
    /// Unknown version, invalid inner model or invalid reference names.
    pub fn validate(&self) -> Result<(), SalienceConversionManifestError> {
        if self.schema != (SchemaId { major: 1, minor: 0 }) {
            return Err(SalienceConversionManifestError::Version);
        }
        self.model
            .validate()
            .map_err(SalienceConversionManifestError::Model)?;
        for reference in [&self.transform, &self.conversion] {
            for name in [&reference.run, &reference.span] {
                crate::model::name(name, false)
                    .map_err(|_| SalienceConversionManifestError::Reference)?;
            }
        }
        Ok(())
    }
}

/// Separate mandatory captured profile; never silently unwrap into ordinary C.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CapturedSalienceConversionManifest {
    /// Complete unchanged A->B->C source/output vocabulary.
    pub lineage: SalienceConversionManifest,
    /// Required complete standalone B capture run/span/root.
    pub capture: EvidenceReference,
}
impl CapturedSalienceConversionManifest {
    /// Check structure only; actual source/log/cache execution requires admission/replay.
    /// # Errors
    /// Invalid lineage/version or missing/invalid capture reference.
    pub fn validate(&self) -> Result<(), SalienceConversionManifestError> {
        self.lineage.validate()?;
        for name in [&self.capture.run, &self.capture.span] {
            crate::model::name(name, false)
                .map_err(|_| SalienceConversionManifestError::Reference)?;
        }
        if self.capture.root == [0; 32] {
            return Err(SalienceConversionManifestError::Reference);
        }
        Ok(())
    }
}

/// Distinct mandatory A-salience plus B-training capture ancestry through C.
/// Strict structure does not establish either numerical execution or qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalCapturedSalienceConversionManifest {
    /// Unchanged complete B-captured A->B->C vocabulary.
    pub lineage: CapturedSalienceConversionManifest,
    /// Mandatory distinct original-A forward-only capture reference.
    pub original_capture: EvidenceReference,
}
impl OriginalCapturedSalienceConversionManifest {
    /// Check inner schema/reference structure, not actual source/log/cache truth.
    /// # Errors
    /// Invalid lineage/version or missing/invalid mandatory references.
    pub fn validate(&self) -> Result<(), SalienceConversionManifestError> {
        self.lineage.validate()?;
        for name in [&self.original_capture.run, &self.original_capture.span] {
            crate::model::name(name, false)
                .map_err(|_| SalienceConversionManifestError::Reference)?;
        }
        if self.original_capture.root == [0; 32]
            || self.lineage.lineage.transform.root == [0; 32]
            || self.lineage.lineage.conversion.root == [0; 32]
        {
            return Err(SalienceConversionManifestError::Reference);
        }
        Ok(())
    }
}

/// Structural carrier failure, not an empirical verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SalienceConversionManifestError {
    /// Unknown mandatory carrier version.
    Version,
    /// Invalid inner model metadata.
    Model(ModelManifestError),
    /// Invalid mandatory evidence run/span basename.
    Reference,
    /// Zero or insufficient byte ceiling.
    Limit,
}
impl fmt::Display for SalienceConversionManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "salience conversion manifest: {self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for SalienceConversionManifestError {}
