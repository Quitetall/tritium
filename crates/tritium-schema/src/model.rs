//! Typed model references for ADR 0044; validation is not execution qualification.

use crate::{
    AdditiveLayout, BlobId, LayoutError, PackageId, PlaneAllocation, ScalePrecision, SchemaId,
};
use alloc::{string::String, vec::Vec};

/// Stored-byte blob reference, local unless a declared shard is selected.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlobReference {
    /// BLAKE3 of the exact stored blob bytes, not a semantic tensor digest.
    pub id: BlobId,
    /// Basename of a declared shard; `None` selects the root package.
    pub shard: Option<String>,
}

/// Actual plane allocation, including future pruned per-tile payloads.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlaneCounts {
    /// All tiles have this nonzero plane count, equal to layout maximum.
    Uniform(u8),
    /// Row-major counts for every 256-coefficient tile; zero prunes a tile.
    PerTile(Vec<u8>),
}

/// Payload interpretation for one named tensor, not backend admission.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorStorage {
    /// Raw little-endian floating-point elements; no hidden tensor header.
    Dense {
        /// Nonempty positive dimensions in logical row-major order.
        shape: Vec<u64>,
        /// Uses the existing floating-point storage vocabulary.
        precision: ScalePrecision,
    },
    /// Canonical packed planes/scales interpreted through the shared layout.
    Additive {
        /// Geometry, law, basis, codec and outer transport.
        layout: AdditiveLayout,
        /// Actual plane counts; serialization does not imply execution support.
        planes: PlaneCounts,
    },
}

/// Named tensor binding in one complete quality/size level.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMeta {
    /// Logical tensor name, independent of any filesystem path.
    pub name: String,
    /// Exact stored bytes and their package location.
    pub blob: BlobReference,
    /// Metadata required to interpret those bytes.
    pub storage: TensorStorage,
}

/// Evidence ancestry reference, not a self-approved quality/release verdict.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceReference {
    /// Run identity of the evidence producer.
    pub run: String,
    /// Evidence span within that run.
    pub span: String,
    /// BLAKE3 evidence root, not a blob id or package digest.
    pub root: [u8; 32],
}

/// Recorded level provenance; external verification still governs claims.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LevelFit {
    /// Fitted directly from source weights; a nonempty evidence list is required.
    Direct {
        /// Ancestry references, not proof that the referenced logs are available/admitted.
        evidence: Vec<EvidenceReference>,
    },
    /// Imported exact prefix of another level in this manifest.
    ImportedPrefixOf {
        /// Named parent level; cycles, self-links and absent parents are rejected.
        level: String,
    },
    /// Imported weights without a direct-fit or exact-prefix claim.
    Imported,
}

/// One complete model level; identical blob references may be shared across levels.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Level {
    /// Logical level name.
    pub name: String,
    /// Declared production/import ancestry.
    pub fit: LevelFit,
    /// Complete sorted tensor inventory; all levels have the same names/shapes.
    pub tensors: Vec<TensorMeta>,
}

/// Named model asset such as config, tokenizer or chat template.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManifestAsset {
    /// Safe basename, not an extraction path.
    pub name: String,
    /// Stored asset bytes; the directory must also declare the same asset name.
    pub blob: BlobReference,
}

/// Content-pinned shard declaration. Shard loading is a separate capability.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelShard {
    /// Safe sibling-file basename.
    pub name: String,
    /// Exact-byte package identity required for this shard.
    pub package: PackageId,
}

/// Legacy model identity retained for later verified migration resolution.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyModelIdentity {
    /// Original namespaced identity; never reinterpreted as a new digest.
    pub id: String,
    /// Level holding the imported model.
    pub level: String,
}

/// Typed `.trit` model metadata. Field profile is proposed, not release-admitted.
///
/// Validation establishes structural consistency only: no architecture coverage,
/// tokenizer validity, evidence availability, model quality or backend capability.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelManifest {
    /// Model metadata version, independent of the outer package version.
    pub schema: SchemaId,
    /// Architecture identifier; execution admission belongs to the loader.
    pub architecture: String,
    /// Sorted assets, including `config.json` and `tokenizer.json`.
    pub assets: Vec<ManifestAsset>,
    /// Sorted complete levels, not incremental patches or inferred prefixes.
    pub levels: Vec<Level>,
    /// Sorted shard declarations.
    pub shards: Vec<ModelShard>,
    /// Sorted legacy-id map; verified migration must establish their authenticity.
    pub legacy_ids: Vec<LegacyModelIdentity>,
}

/// Structural typed-manifest rejection; missing capabilities are not green results.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelManifestError {
    /// Metadata version is unsupported.
    UnsupportedVersion(SchemaId),
    /// Empty/control/oversized name, or an unsafe asset/shard basename.
    InvalidName,
    /// A named inventory is not strictly sorted or has duplicate names.
    UnsortedOrDuplicate(&'static str),
    /// A required model asset is absent.
    MissingAsset(&'static str),
    /// Level/tensor/evidence inventory is empty or exceeds a bounded count.
    InvalidCount,
    /// Tensor shape is empty, zero or overflows u64.
    InvalidShape,
    /// Additive layout violates the existing law/basis schema.
    InvalidLayout(LayoutError),
    /// Plane counts disagree with allocation, tile geometry or the law maximum.
    InvalidPlanes,
    /// A blob references an undeclared shard.
    MissingShard,
    /// Prefix parent is absent, self-referential or cyclic.
    InvalidPrefix,
    /// Complete levels disagree on tensor names or logical geometry.
    LevelShapeMismatch,
    /// Legacy identity points to an absent level.
    MissingLegacyLevel,
}

impl core::fmt::Display for ModelManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "typed model manifest: {self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for ModelManifestError {}

impl ModelManifest {
    /// Check sorted inventories, references, geometry and ancestry without payload I/O.
    ///
    /// # Errors
    /// Returns a typed structural error; success is not model-load qualification.
    pub fn validate(&self) -> Result<(), ModelManifestError> {
        use ModelManifestError as E;
        if self.schema != (SchemaId { major: 1, minor: 0 }) {
            return Err(E::UnsupportedVersion(self.schema));
        }
        name(&self.architecture, false)?;
        sorted(self.assets.iter().map(|x| x.name.as_str()), "assets")?;
        sorted(self.levels.iter().map(|x| x.name.as_str()), "levels")?;
        sorted(self.shards.iter().map(|x| x.name.as_str()), "shards")?;
        sorted(self.legacy_ids.iter().map(|x| x.id.as_str()), "legacy_ids")?;
        if self.levels.is_empty()
            || self.levels.len() > 1024
            || self.assets.len() > 4096
            || self.shards.len() > 1_000_000
            || self.legacy_ids.len() > 1_000_000
        {
            return Err(E::InvalidCount);
        }
        for required in ["config.json", "tokenizer.json"] {
            if !self.assets.iter().any(|a| a.name == required) {
                return Err(E::MissingAsset(required));
            }
        }
        let reference = |blob: &BlobReference| -> Result<(), E> {
            if let Some(shard) = &blob.shard {
                name(shard, true)?;
                if self
                    .shards
                    .binary_search_by(|s| s.name.as_str().cmp(shard.as_str()))
                    .is_err()
                {
                    return Err(E::MissingShard);
                }
            }
            Ok(())
        };
        for shard in &self.shards {
            name(&shard.name, true)?;
        }
        for asset in &self.assets {
            name(&asset.name, true)?;
            reference(&asset.blob)?;
        }
        for level in &self.levels {
            name(&level.name, false)?;
            sorted(level.tensors.iter().map(|t| t.name.as_str()), "tensors")?;
            if level.tensors.is_empty() || level.tensors.len() > 1_000_000 {
                return Err(E::InvalidCount);
            }
            if let LevelFit::Direct { evidence } = &level.fit {
                if evidence.is_empty() || evidence.len() > 4096 {
                    return Err(E::InvalidCount);
                }
                for e in evidence {
                    name(&e.run, false)?;
                    name(&e.span, false)?;
                }
                if evidence.windows(2).any(|pair| {
                    (&pair[0].run, &pair[0].span, &pair[0].root)
                        >= (&pair[1].run, &pair[1].span, &pair[1].root)
                }) {
                    return Err(E::UnsortedOrDuplicate("evidence"));
                }
            }
            for tensor in &level.tensors {
                name(&tensor.name, false)?;
                reference(&tensor.blob)?;
                match &tensor.storage {
                    TensorStorage::Dense { shape, .. } => {
                        if shape.is_empty()
                            || shape.len() > 32
                            || shape
                                .iter()
                                .try_fold(
                                    1u64,
                                    |n, &d| if d == 0 { None } else { n.checked_mul(d) },
                                )
                                .is_none()
                        {
                            return Err(E::InvalidShape);
                        }
                    }
                    TensorStorage::Additive { layout, planes } => {
                        layout.validate().map_err(E::InvalidLayout)?;
                        layout
                            .rows
                            .checked_mul(layout.cols)
                            .ok_or(E::InvalidShape)?;
                        match (layout.allocation, planes) {
                            (PlaneAllocation::Uniform, PlaneCounts::Uniform(n))
                                if *n == layout.max_planes => {}
                            (PlaneAllocation::PerTile, PlaneCounts::PerTile(counts)) => {
                                let tiles = layout
                                    .rows
                                    .checked_mul(layout.cols.div_ceil(256))
                                    .ok_or(E::InvalidShape)?;
                                if u64::try_from(counts.len()).ok() != Some(tiles)
                                    || counts.iter().any(|&n| n > layout.max_planes)
                                {
                                    return Err(E::InvalidPlanes);
                                }
                            }
                            _ => return Err(E::InvalidPlanes),
                        }
                    }
                }
            }
            let first = &self.levels[0].tensors;
            if first.len() != level.tensors.len()
                || first
                    .iter()
                    .zip(&level.tensors)
                    .any(|(a, b)| a.name != b.name || !same_shape(&a.storage, &b.storage))
            {
                return Err(E::LevelShapeMismatch);
            }
            let mut ancestor = level;
            for depth in 0..=self.levels.len() {
                match &ancestor.fit {
                    LevelFit::ImportedPrefixOf { level: parent } => {
                        if depth == self.levels.len() {
                            return Err(E::InvalidPrefix);
                        }
                        let index = self
                            .levels
                            .binary_search_by(|l| l.name.as_str().cmp(parent.as_str()))
                            .map_err(|_| E::InvalidPrefix)?;
                        ancestor = &self.levels[index];
                    }
                    _ => break,
                }
            }
        }
        for legacy in &self.legacy_ids {
            name(&legacy.id, false)?;
            if self
                .levels
                .binary_search_by(|l| l.name.as_str().cmp(legacy.level.as_str()))
                .is_err()
            {
                return Err(E::MissingLegacyLevel);
            }
        }
        Ok(())
    }
}

pub(crate) fn name(value: &str, basename: bool) -> Result<(), ModelManifestError> {
    if value.is_empty()
        || value.len() > 4096
        || value.chars().any(char::is_control)
        || (basename && (value == "." || value == ".." || value.contains(['/', '\\', ':'])))
    {
        return Err(ModelManifestError::InvalidName);
    }
    Ok(())
}

pub(crate) fn sorted<'a>(
    values: impl Iterator<Item = &'a str>,
    inventory: &'static str,
) -> Result<(), ModelManifestError> {
    let mut previous = None;
    for value in values {
        if previous.is_some_and(|prev| prev >= value) {
            return Err(ModelManifestError::UnsortedOrDuplicate(inventory));
        }
        previous = Some(value);
    }
    Ok(())
}

fn same_shape(a: &TensorStorage, b: &TensorStorage) -> bool {
    match (a, b) {
        (TensorStorage::Dense { shape: a, .. }, TensorStorage::Dense { shape: b, .. }) => a == b,
        (TensorStorage::Additive { layout: a, .. }, TensorStorage::Additive { layout: b, .. }) => {
            a.rows == b.rows && a.cols == b.cols
        }
        (TensorStorage::Dense { shape, .. }, TensorStorage::Additive { layout, .. })
        | (TensorStorage::Additive { layout, .. }, TensorStorage::Dense { shape, .. }) => {
            shape.as_slice() == [layout.rows, layout.cols]
        }
    }
}
