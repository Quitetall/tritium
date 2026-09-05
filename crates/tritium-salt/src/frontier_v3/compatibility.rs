//! Fail-closed readers adapting canonical prior artifacts into V3 contracts.

use std::{error::Error, fmt};

use tritium_format::salt_v2_package::{
    SaltV2IndexedRuntimeLedger, SaltV2Package, SaltV2PackageError, read_salt_v2_package,
    write_salt_v2_package,
};

use super::{
    ArtifactByteLedger, ArtifactClaim, ByteBreakdown, FrontierArtifactError,
    FrontierArtifactManifest, FrontierProfileId, FrontierTensorArtifact, SolverDescriptor,
    TensorRepresentation,
};
use crate::ContentId;

/// Decode canonical SALT V2 bytes into a typed V3 artifact manifest.
///
/// Serialized and resident payload bytes come from existing SALT V2 ledgers.
/// Scales, maps, framing, transforms, indexes, and padding remain metadata.
/// `transient_bytes` must be supplied from external measurement; adapter never
/// invents a workspace claim. Each tensor payload identity hashes its canonical
/// one-tensor SALT V2 envelope, binding codec, semantic values, and scales.
#[allow(clippy::too_many_arguments)]
pub fn read_salt_v2_frontier_artifact(
    bytes: &[u8],
    source_id: ContentId,
    profile_id: FrontierProfileId,
    solver: SolverDescriptor,
    recipe_id: ContentId,
    transient_bytes: u64,
) -> Result<FrontierArtifactManifest, FrontierCompatibilityError> {
    let decoded = read_salt_v2_package(bytes).map_err(FrontierCompatibilityError::SaltV2)?;
    let runtime = SaltV2IndexedRuntimeLedger::for_package(&decoded.package)
        .map_err(FrontierCompatibilityError::SaltV2)?;
    let serialized_metadata = decoded
        .ledger
        .total_bytes
        .checked_sub(decoded.ledger.payload_bytes)
        .ok_or(FrontierCompatibilityError::LedgerUnderflow { view: "serialized" })?;
    let resident_metadata = runtime
        .steady_resident_bytes()
        .checked_sub(runtime.payload_bytes())
        .ok_or(FrontierCompatibilityError::LedgerUnderflow { view: "resident" })?;
    let ledger = ArtifactByteLedger::new(
        ByteBreakdown::new(decoded.ledger.payload_bytes, serialized_metadata, 0)?,
        ByteBreakdown::new(runtime.payload_bytes(), resident_metadata, 0)?,
        transient_bytes,
    )?;

    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(decoded.package.tensors().len())
        .map_err(|_| FrontierCompatibilityError::AllocationFailed)?;
    for tensor in decoded.package.tensors() {
        let one = SaltV2Package::new(decoded.package.codec(), vec![tensor.clone()])
            .map_err(FrontierCompatibilityError::SaltV2)?;
        let encoded = write_salt_v2_package(&one).map_err(FrontierCompatibilityError::SaltV2)?;
        let tensor_runtime = SaltV2IndexedRuntimeLedger::for_tensor(tensor, one.codec())
            .map_err(FrontierCompatibilityError::SaltV2)?;
        tensors.push(FrontierTensorArtifact::new(
            tensor.name(),
            tensor.dims().to_vec(),
            solver.clone(),
            TensorRepresentation::AdditiveTernaryPlanes,
            ArtifactClaim::PureTernary,
            recipe_id,
            ContentId::of_bytes(&encoded.bytes),
            encoded.ledger.payload_bytes,
            tensor_runtime.payload_bytes(),
        )?);
    }
    tensors.sort_by(|left, right| left.name().cmp(right.name()));
    FrontierArtifactManifest::new(source_id, profile_id, tensors, ledger)
        .map_err(FrontierCompatibilityError::Artifact)
}

/// Failure while decoding or adapting a prior artifact into V3.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrontierCompatibilityError {
    /// Canonical SALT V2 reader or ledger rejected input.
    SaltV2(SaltV2PackageError),
    /// V3 artifact contract rejected adapted fields.
    Artifact(FrontierArtifactError),
    /// Source ledger claimed fewer total bytes than payload bytes.
    LedgerUnderflow {
        /// Serialized or resident view.
        view: &'static str,
    },
    /// Adapter could not reserve bounded tensor records.
    AllocationFailed,
}

impl From<FrontierArtifactError> for FrontierCompatibilityError {
    fn from(value: FrontierArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl fmt::Display for FrontierCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SaltV2(source) => {
                write!(formatter, "SALT V2 compatibility read failed: {source}")
            }
            Self::Artifact(source) => {
                write!(formatter, "V3 compatibility manifest failed: {source}")
            }
            Self::LedgerUnderflow { view } => {
                write!(formatter, "SALT V2 {view} ledger underflows payload bytes")
            }
            Self::AllocationFailed => {
                formatter.write_str("SALT V2 compatibility tensor allocation failed")
            }
        }
    }
}

impl Error for FrontierCompatibilityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SaltV2(source) => Some(source),
            Self::Artifact(source) => Some(source),
            Self::LedgerUnderflow { .. } | Self::AllocationFailed => None,
        }
    }
}
