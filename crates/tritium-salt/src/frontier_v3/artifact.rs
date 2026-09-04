//! Heterogeneous, typed SALT V3 artifact manifest contract.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use super::{FrontierProfileId, SolverDescriptor, SolverFamily};
use crate::ContentId;

/// Stable serialized schema emitted by current V3 artifact writers.
pub const FRONTIER_ARTIFACT_SCHEMA_V1: &str = "tritium.frontier-artifact.v1";

/// Whole-artifact and per-tensor claim boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactClaim {
    /// Learned coefficients are ternary plus declared scales and metadata.
    PureTernary,
    /// Artifact includes a separately accounted non-ternary residual.
    ResidualBearing,
}

/// Persistent tensor representation consumed by a matching execution bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum TensorRepresentation {
    /// One or more independently scaled additive ternary planes.
    AdditiveTernaryPlanes,
    /// Salience-selected ternary residual representation.
    SalientTernaryResidual,
    /// Expanded-rank two-sided ternary factorization.
    ExpandedRankTernary,
    /// Rotation-aware ternary weight representation.
    RotatedTernary,
    /// Single-plane TWN or TTQ representation.
    SingleTernary,
    /// Sparse ternary payload with explicit structure metadata.
    SparseTernary,
    /// Ratio-three folded nine-level persistent codes.
    FoldedNineLevel,
    /// Registered external representation bound by its recipe identity.
    Custom,
}

/// Physical byte components for one serialized or resident view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ByteBreakdownWire")]
pub struct ByteBreakdown {
    tensor_payload_bytes: u64,
    metadata_bytes: u64,
    preserved_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteBreakdownWire {
    tensor_payload_bytes: u64,
    metadata_bytes: u64,
    preserved_bytes: u64,
}

impl TryFrom<ByteBreakdownWire> for ByteBreakdown {
    type Error = FrontierArtifactError;

    fn try_from(value: ByteBreakdownWire) -> Result<Self, Self::Error> {
        Self::new(
            value.tensor_payload_bytes,
            value.metadata_bytes,
            value.preserved_bytes,
        )
    }
}

impl ByteBreakdown {
    /// Construct exact physical components, rejecting an unrepresentable sum.
    pub fn new(
        tensor_payload_bytes: u64,
        metadata_bytes: u64,
        preserved_bytes: u64,
    ) -> Result<Self, FrontierArtifactError> {
        let value = Self {
            tensor_payload_bytes,
            metadata_bytes,
            preserved_bytes,
        };
        value.total()?;
        Ok(value)
    }

    /// Typed tensor payload bytes.
    pub const fn tensor_payload_bytes(self) -> u64 {
        self.tensor_payload_bytes
    }

    /// Manifest, scales, maps, and transport metadata bytes.
    pub const fn metadata_bytes(self) -> u64 {
        self.metadata_bytes
    }

    /// Preserved dense tensor bytes.
    pub const fn preserved_bytes(self) -> u64 {
        self.preserved_bytes
    }

    /// Checked sum of all physical components.
    pub fn total(self) -> Result<u64, FrontierArtifactError> {
        self.tensor_payload_bytes
            .checked_add(self.metadata_bytes)
            .and_then(|value| value.checked_add(self.preserved_bytes))
            .ok_or(FrontierArtifactError::ByteCountOverflow)
    }
}

/// Exact serialized and resident physical byte ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ArtifactByteLedgerWire", into = "ArtifactByteLedgerWire")]
pub struct ArtifactByteLedger {
    serialized: ByteBreakdown,
    resident: ByteBreakdown,
    transient_bytes: u64,
    total_serialized_bytes: u64,
    total_resident_bytes: u64,
    peak_working_set_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactByteLedgerWire {
    serialized: ByteBreakdown,
    resident: ByteBreakdown,
    transient_bytes: u64,
    total_serialized_bytes: u64,
    total_resident_bytes: u64,
    peak_working_set_bytes: u64,
}

impl ArtifactByteLedger {
    /// Build exact totals from measured components.
    pub fn new(
        serialized: ByteBreakdown,
        resident: ByteBreakdown,
        transient_bytes: u64,
    ) -> Result<Self, FrontierArtifactError> {
        let total_serialized_bytes = serialized.total()?;
        let total_resident_bytes = resident.total()?;
        let peak_working_set_bytes = total_resident_bytes
            .checked_add(transient_bytes)
            .ok_or(FrontierArtifactError::ByteCountOverflow)?;
        Ok(Self {
            serialized,
            resident,
            transient_bytes,
            total_serialized_bytes,
            total_resident_bytes,
            peak_working_set_bytes,
        })
    }

    /// Serialized physical components.
    pub const fn serialized(self) -> ByteBreakdown {
        self.serialized
    }

    /// Resident physical components, excluding transient workspace.
    pub const fn resident(self) -> ByteBreakdown {
        self.resident
    }

    /// Peak transient workspace bytes.
    pub const fn transient_bytes(self) -> u64 {
        self.transient_bytes
    }

    /// Exact whole-artifact serialized bytes.
    pub const fn total_serialized_bytes(self) -> u64 {
        self.total_serialized_bytes
    }

    /// Exact steady-state resident bytes, excluding transient workspace.
    pub const fn total_resident_bytes(self) -> u64 {
        self.total_resident_bytes
    }

    /// Exact peak resident working set including transient workspace.
    pub const fn peak_working_set_bytes(self) -> u64 {
        self.peak_working_set_bytes
    }
}

impl TryFrom<ArtifactByteLedgerWire> for ArtifactByteLedger {
    type Error = FrontierArtifactError;

    fn try_from(value: ArtifactByteLedgerWire) -> Result<Self, Self::Error> {
        let ledger = Self::new(value.serialized, value.resident, value.transient_bytes)?;
        if value.total_serialized_bytes != ledger.total_serialized_bytes {
            return Err(FrontierArtifactError::LedgerTotalMismatch {
                view: "serialized",
                declared: value.total_serialized_bytes,
                measured: ledger.total_serialized_bytes,
            });
        }
        if value.total_resident_bytes != ledger.total_resident_bytes {
            return Err(FrontierArtifactError::LedgerTotalMismatch {
                view: "resident",
                declared: value.total_resident_bytes,
                measured: ledger.total_resident_bytes,
            });
        }
        if value.peak_working_set_bytes != ledger.peak_working_set_bytes {
            return Err(FrontierArtifactError::LedgerTotalMismatch {
                view: "working-set",
                declared: value.peak_working_set_bytes,
                measured: ledger.peak_working_set_bytes,
            });
        }
        Ok(ledger)
    }
}

impl From<ArtifactByteLedger> for ArtifactByteLedgerWire {
    fn from(value: ArtifactByteLedger) -> Self {
        Self {
            serialized: value.serialized,
            resident: value.resident,
            transient_bytes: value.transient_bytes,
            total_serialized_bytes: value.total_serialized_bytes,
            total_resident_bytes: value.total_resident_bytes,
            peak_working_set_bytes: value.peak_working_set_bytes,
        }
    }
}

/// One tensor's typed V3 recipe and physical payload binding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierTensorArtifactWire",
    into = "FrontierTensorArtifactWire"
)]
pub struct FrontierTensorArtifact {
    name: String,
    shape: Vec<u64>,
    element_count: u64,
    solver: SolverDescriptor,
    representation: TensorRepresentation,
    claim: ArtifactClaim,
    recipe_id: ContentId,
    payload_id: ContentId,
    serialized_bytes: u64,
    resident_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierTensorArtifactWire {
    name: String,
    shape: Vec<u64>,
    element_count: u64,
    solver: SolverDescriptor,
    representation: TensorRepresentation,
    claim: ArtifactClaim,
    #[serde(with = "content_id_text")]
    recipe_id: ContentId,
    #[serde(with = "content_id_text")]
    payload_id: ContentId,
    serialized_bytes: u64,
    resident_bytes: u64,
}

impl FrontierTensorArtifact {
    /// Construct a typed tensor record, rejecting malformed lineage or shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        shape: Vec<u64>,
        solver: SolverDescriptor,
        representation: TensorRepresentation,
        claim: ArtifactClaim,
        recipe_id: ContentId,
        payload_id: ContentId,
        serialized_bytes: u64,
        resident_bytes: u64,
    ) -> Result<Self, FrontierArtifactError> {
        let name = name.into();
        if name.is_empty()
            || name.len() > 4096
            || name.trim() != name
            || name.chars().any(char::is_control)
        {
            return Err(FrontierArtifactError::InvalidTensorName { name });
        }
        if shape.is_empty() || shape.len() > 16 || shape.contains(&0) {
            return Err(FrontierArtifactError::InvalidTensorShape {
                tensor: name,
                shape,
            });
        }
        let Some(element_count) = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        else {
            return Err(FrontierArtifactError::InvalidTensorShape {
                tensor: name,
                shape,
            });
        };
        if !representation_matches(solver.family(), representation) {
            return Err(FrontierArtifactError::RepresentationFamilyMismatch {
                tensor: name,
                family: solver.family(),
                representation,
            });
        }
        require_digest("recipe", recipe_id)?;
        require_digest("payload", payload_id)?;
        if serialized_bytes == 0 || resident_bytes == 0 {
            return Err(FrontierArtifactError::InvalidTensorBytes {
                tensor: name,
                serialized_bytes,
                resident_bytes,
            });
        }
        Ok(Self {
            name,
            shape,
            element_count,
            solver,
            representation,
            claim,
            recipe_id,
            payload_id,
            serialized_bytes,
            resident_bytes,
        })
    }

    /// Canonical full tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Checked product of tensor dimensions.
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    /// Exact solver descriptor.
    pub const fn solver(&self) -> &SolverDescriptor {
        &self.solver
    }

    /// Persistent tensor representation.
    pub const fn representation(&self) -> TensorRepresentation {
        self.representation
    }

    /// Tensor-specific claim boundary.
    pub const fn claim(&self) -> ArtifactClaim {
        self.claim
    }

    /// Exact solver recipe identity.
    pub const fn recipe_id(&self) -> ContentId {
        self.recipe_id
    }

    /// Exact payload identity.
    pub const fn payload_id(&self) -> ContentId {
        self.payload_id
    }

    /// Physical serialized payload bytes.
    pub const fn serialized_bytes(&self) -> u64 {
        self.serialized_bytes
    }

    /// Physical resident payload bytes.
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl TryFrom<FrontierTensorArtifactWire> for FrontierTensorArtifact {
    type Error = FrontierArtifactError;

    fn try_from(value: FrontierTensorArtifactWire) -> Result<Self, Self::Error> {
        let declared_elements = value.element_count;
        let tensor = Self::new(
            value.name,
            value.shape,
            value.solver,
            value.representation,
            value.claim,
            value.recipe_id,
            value.payload_id,
            value.serialized_bytes,
            value.resident_bytes,
        )?;
        if tensor.element_count != declared_elements {
            return Err(FrontierArtifactError::ElementCountMismatch {
                tensor: tensor.name,
                declared: declared_elements,
                measured: tensor.element_count,
            });
        }
        Ok(tensor)
    }
}

impl From<FrontierTensorArtifact> for FrontierTensorArtifactWire {
    fn from(value: FrontierTensorArtifact) -> Self {
        Self {
            name: value.name,
            shape: value.shape,
            element_count: value.element_count,
            solver: value.solver,
            representation: value.representation,
            claim: value.claim,
            recipe_id: value.recipe_id,
            payload_id: value.payload_id,
            serialized_bytes: value.serialized_bytes,
            resident_bytes: value.resident_bytes,
        }
    }
}

/// Canonical V3 manifest for heterogeneous tensor recipes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierArtifactManifestWire",
    into = "FrontierArtifactManifestWire"
)]
pub struct FrontierArtifactManifest {
    source_id: ContentId,
    profile_id: FrontierProfileId,
    claim: ArtifactClaim,
    tensors: Vec<FrontierTensorArtifact>,
    ledger: ArtifactByteLedger,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierArtifactManifestWire {
    schema: String,
    #[serde(with = "content_id_text")]
    source_id: ContentId,
    profile_id: FrontierProfileId,
    claim: ArtifactClaim,
    tensors: Vec<FrontierTensorArtifact>,
    ledger: ArtifactByteLedger,
}

impl FrontierArtifactManifest {
    /// Construct a canonical manifest and reconcile tensor payload byte totals.
    ///
    /// Whole-artifact claim is derived from tensor claims; callers cannot
    /// independently strengthen or weaken it.
    pub fn new(
        source_id: ContentId,
        profile_id: FrontierProfileId,
        tensors: Vec<FrontierTensorArtifact>,
        ledger: ArtifactByteLedger,
    ) -> Result<Self, FrontierArtifactError> {
        require_digest("source", source_id)?;
        if tensors.is_empty() {
            return Err(FrontierArtifactError::EmptyManifest);
        }
        for pair in tensors.windows(2) {
            if pair[0].name() >= pair[1].name() {
                return Err(FrontierArtifactError::NonCanonicalTensorOrder {
                    previous: pair[0].name().to_owned(),
                    current: pair[1].name().to_owned(),
                });
            }
        }
        let serialized = sum_tensor_bytes(&tensors, FrontierTensorArtifact::serialized_bytes)?;
        let resident = sum_tensor_bytes(&tensors, FrontierTensorArtifact::resident_bytes)?;
        if serialized != ledger.serialized().tensor_payload_bytes() {
            return Err(FrontierArtifactError::TensorByteMismatch {
                view: "serialized",
                declared: ledger.serialized().tensor_payload_bytes(),
                measured: serialized,
            });
        }
        if resident != ledger.resident().tensor_payload_bytes() {
            return Err(FrontierArtifactError::TensorByteMismatch {
                view: "resident",
                declared: ledger.resident().tensor_payload_bytes(),
                measured: resident,
            });
        }
        let claim = if tensors
            .iter()
            .any(|tensor| tensor.claim() == ArtifactClaim::ResidualBearing)
        {
            ArtifactClaim::ResidualBearing
        } else {
            ArtifactClaim::PureTernary
        };
        Ok(Self {
            source_id,
            profile_id,
            claim,
            tensors,
            ledger,
        })
    }

    /// Exact source model identity.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Profile that produced this artifact.
    pub const fn profile_id(&self) -> &FrontierProfileId {
        &self.profile_id
    }

    /// Derived whole-artifact claim boundary.
    pub const fn claim(&self) -> ArtifactClaim {
        self.claim
    }

    /// Canonically ordered heterogeneous tensor records.
    pub fn tensors(&self) -> &[FrontierTensorArtifact] {
        &self.tensors
    }

    /// Exact physical byte ledger.
    pub const fn ledger(&self) -> ArtifactByteLedger {
        self.ledger
    }
}

impl TryFrom<FrontierArtifactManifestWire> for FrontierArtifactManifest {
    type Error = FrontierArtifactError;

    fn try_from(value: FrontierArtifactManifestWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_ARTIFACT_SCHEMA_V1 {
            return Err(FrontierArtifactError::UnsupportedSchema {
                found: value.schema,
                supported: FRONTIER_ARTIFACT_SCHEMA_V1,
            });
        }
        let declared_claim = value.claim;
        let manifest = Self::new(
            value.source_id,
            value.profile_id,
            value.tensors,
            value.ledger,
        )?;
        if manifest.claim != declared_claim {
            return Err(FrontierArtifactError::ClaimMismatch {
                declared: declared_claim,
                measured: manifest.claim,
            });
        }
        Ok(manifest)
    }
}

impl From<FrontierArtifactManifest> for FrontierArtifactManifestWire {
    fn from(value: FrontierArtifactManifest) -> Self {
        Self {
            schema: FRONTIER_ARTIFACT_SCHEMA_V1.to_owned(),
            source_id: value.source_id,
            profile_id: value.profile_id,
            claim: value.claim,
            tensors: value.tensors,
            ledger: value.ledger,
        }
    }
}

fn representation_matches(family: SolverFamily, representation: TensorRepresentation) -> bool {
    // Unknown future families or representations must remain rejected until
    // this mapping receives an explicit contract decision.
    matches!(
        (family, representation),
        (
            SolverFamily::Salt,
            TensorRepresentation::AdditiveTernaryPlanes
        ) | (
            SolverFamily::QteaSalientResidual,
            TensorRepresentation::SalientTernaryResidual
        ) | (
            SolverFamily::ExTernD,
            TensorRepresentation::ExpandedRankTernary
        ) | (SolverFamily::Twla, TensorRepresentation::RotatedTernary)
            | (
                SolverFamily::Twn | SolverFamily::Ttq,
                TensorRepresentation::SingleTernary
            )
            | (
                SolverFamily::SparseTernary,
                TensorRepresentation::SparseTernary
            )
            | (
                SolverFamily::FoldedNineLevel,
                TensorRepresentation::FoldedNineLevel
            )
            | (SolverFamily::Custom, TensorRepresentation::Custom)
    )
}

fn require_digest(field: &'static str, id: ContentId) -> Result<(), FrontierArtifactError> {
    if id.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(FrontierArtifactError::ZeroDigest { field });
    }
    Ok(())
}

pub(super) mod content_id_text {
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    use crate::ContentId;

    const PREFIX: &str = "tsc1_";
    const HEX: &[u8; 16] = b"0123456789abcdef";

    pub(in crate::frontier_v3) fn serialize<S>(
        id: &ContentId,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut text = String::with_capacity(PREFIX.len() + 64);
        text.push_str(PREFIX);
        for byte in id.as_bytes() {
            text.push(HEX[usize::from(byte >> 4)] as char);
            text.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        serializer.serialize_str(&text)
    }

    pub(in crate::frontier_v3) fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<ContentId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let Some(hex) = text.strip_prefix(PREFIX) else {
            return Err(D::Error::custom("content identity must start with tsc1_"));
        };
        if hex.len() != 64 {
            return Err(D::Error::custom(
                "content identity must contain exactly 64 lowercase hex digits",
            ));
        }
        let mut digest = [0_u8; 32];
        for (index, pair) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            digest[index] = decode(pair[0])
                .and_then(|high| decode(pair[1]).map(|low| high << 4 | low))
                .ok_or_else(|| {
                    D::Error::custom("content identity contains non-lowercase-hex bytes")
                })?;
        }
        Ok(ContentId::from_digest(digest))
    }

    fn decode(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
}

fn sum_tensor_bytes(
    tensors: &[FrontierTensorArtifact],
    measure: impl Fn(&FrontierTensorArtifact) -> u64,
) -> Result<u64, FrontierArtifactError> {
    tensors.iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(measure(tensor))
            .ok_or(FrontierArtifactError::ByteCountOverflow)
    })
}

/// Fail-closed V3 artifact construction or decoding error.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrontierArtifactError {
    /// Physical byte components overflow `u64`.
    ByteCountOverflow,
    /// Serialized ledger total disagrees with its components.
    LedgerTotalMismatch {
        /// Serialized or resident view.
        view: &'static str,
        /// Total encoded in input.
        declared: u64,
        /// Total recomputed from components.
        measured: u64,
    },
    /// Tensor name is empty, unbounded, padded, or contains control characters.
    InvalidTensorName {
        /// Rejected name.
        name: String,
    },
    /// Tensor shape is empty, unbounded, zero-sized, or overflowing.
    InvalidTensorShape {
        /// Tensor name.
        tensor: String,
        /// Rejected dimensions.
        shape: Vec<u64>,
    },
    /// Solver family cannot emit the declared representation.
    RepresentationFamilyMismatch {
        /// Tensor name.
        tensor: String,
        /// Declared solver family.
        family: SolverFamily,
        /// Declared representation.
        representation: TensorRepresentation,
    },
    /// Required digest is all zeroes.
    ZeroDigest {
        /// Digest field.
        field: &'static str,
    },
    /// Tensor payload uses a zero serialized or resident byte count.
    InvalidTensorBytes {
        /// Tensor name.
        tensor: String,
        /// Serialized bytes.
        serialized_bytes: u64,
        /// Resident bytes.
        resident_bytes: u64,
    },
    /// Serialized element count disagrees with checked shape product.
    ElementCountMismatch {
        /// Tensor name.
        tensor: String,
        /// Encoded element count.
        declared: u64,
        /// Recomputed element count.
        measured: u64,
    },
    /// Manifest contains no typed tensor records.
    EmptyManifest,
    /// Tensor records are duplicated or not strictly lexically ordered.
    NonCanonicalTensorOrder {
        /// Prior tensor name.
        previous: String,
        /// Current tensor name.
        current: String,
    },
    /// Tensor payload totals disagree with physical ledger.
    TensorByteMismatch {
        /// Serialized or resident view.
        view: &'static str,
        /// Ledger component.
        declared: u64,
        /// Sum over tensor records.
        measured: u64,
    },
    /// Serialized artifact schema is unsupported.
    UnsupportedSchema {
        /// Schema found on input.
        found: String,
        /// Schema supported by this reader.
        supported: &'static str,
    },
    /// Whole-artifact claim disagrees with tensor claims.
    ClaimMismatch {
        /// Encoded claim.
        declared: ArtifactClaim,
        /// Claim derived from tensor records.
        measured: ArtifactClaim,
    },
}

impl fmt::Display for FrontierArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteCountOverflow => formatter.write_str("artifact byte count overflows u64"),
            Self::LedgerTotalMismatch {
                view,
                declared,
                measured,
            } => write!(
                formatter,
                "{view} ledger total {declared} does not match component sum {measured}"
            ),
            Self::InvalidTensorName { name } => write!(formatter, "invalid tensor name {name:?}"),
            Self::InvalidTensorShape { tensor, shape } => {
                write!(formatter, "invalid shape {shape:?} for tensor {tensor:?}")
            }
            Self::RepresentationFamilyMismatch {
                tensor,
                family,
                representation,
            } => write!(
                formatter,
                "tensor {tensor:?} representation {representation:?} does not match {family:?}"
            ),
            Self::ZeroDigest { field } => write!(formatter, "{field} digest is all zeroes"),
            Self::InvalidTensorBytes {
                tensor,
                serialized_bytes,
                resident_bytes,
            } => write!(
                formatter,
                "tensor {tensor:?} has invalid bytes serialized={serialized_bytes}, resident={resident_bytes}"
            ),
            Self::ElementCountMismatch {
                tensor,
                declared,
                measured,
            } => write!(
                formatter,
                "tensor {tensor:?} element count {declared} does not match shape product {measured}"
            ),
            Self::EmptyManifest => formatter.write_str("frontier artifact has no tensors"),
            Self::NonCanonicalTensorOrder { previous, current } => write!(
                formatter,
                "tensor order is not canonical: {previous:?} before {current:?}"
            ),
            Self::TensorByteMismatch {
                view,
                declared,
                measured,
            } => write!(
                formatter,
                "{view} tensor bytes {declared} do not match tensor records {measured}"
            ),
            Self::UnsupportedSchema { found, supported } => write!(
                formatter,
                "unsupported frontier artifact schema {found:?}; supported schema is {supported}"
            ),
            Self::ClaimMismatch { declared, measured } => write!(
                formatter,
                "artifact claim {declared:?} does not match tensor-derived claim {measured:?}"
            ),
        }
    }
}

impl Error for FrontierArtifactError {}
