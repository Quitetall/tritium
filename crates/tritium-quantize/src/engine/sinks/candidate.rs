//! Distinct unqualified candidate metadata over the existing commit transaction.
use super::*;
use tritium_format::{PackageId, TritBlobVisitError, TritPackageReadError, TritPackageReader};

const SCHEMA: &str = "tritium.unqualified-candidate.v1";
/// Data describing committed candidate bytes, never a receipt or approval token.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct UnqualifiedCandidateRecord {
    schema: String,
    qualification: bool,
    package_id: tritium_schema::PackageId,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    schema: String,
    qualification: bool,
    package_id: tritium_schema::PackageId,
}
impl UnqualifiedCandidateRecord {
    fn new(id: PackageId) -> Self {
        Self {
            schema: SCHEMA.into(),
            qualification: false,
            package_id: tritium_schema::PackageId::from_bytes(*id.as_bytes()),
        }
    }
    /// Parse at most4KiB with strict scope/identity/field admission. Does not read
    /// model bytes, authenticate a publisher or establish a qualification state.
    /// # Errors
    /// Oversize/malformed/ambiguous JSON, unsupported schema, true scope or zero ID.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, QuantEngineError> {
        if bytes.len() > 4096 {
            return Err(QuantEngineError::Limit("candidate metadata bytes"));
        }
        let w: Wire = serde_json::from_slice(bytes)
            .map_err(|e| QuantEngineError::Source(format!("candidate metadata: {e}")))?;
        if w.schema != SCHEMA || w.qualification || w.package_id.as_bytes() == &[0; 32] {
            return Err(QuantEngineError::Binding(
                "invalid unqualified candidate scope/identity",
            ));
        }
        Ok(Self {
            schema: w.schema,
            qualification: false,
            package_id: w.package_id,
        })
    }
    /// Exact declared committed-package identity; callers must verify actual bytes.
    #[must_use]
    pub const fn package_id(&self) -> tritium_schema::PackageId {
        self.package_id
    }
}

/// Candidate emission/validation/publication refusal; no silent partial success.
#[derive(Debug)]
#[non_exhaustive]
pub enum CandidatePublicationError<E> {
    /// Original caller emitter error, unchanged.
    Emission(E),
    /// Original owned-file/no-clobber transaction failure before commit.
    Publication(QuantEngineError),
    /// Actual pending package/digest/receiving/I/O refusal.
    Validation(TritPackageReadError),
    /// Original streaming blob-digest/visitor validation error, unchanged.
    BlobValidation(TritBlobVisitError<std::convert::Infallible>),
    /// Emitter's declared exact bytes differ from actual admitted output.
    IdentityMismatch {
        /// Expected exact identity supplied by the emitter.
        expected: PackageId,
        /// Actual reopened pending-file identity.
        actual: PackageId,
    },
}
impl<E: std::fmt::Display> std::fmt::Display for CandidatePublicationError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Emission(e) => e.fmt(f),
            Self::Publication(e) => e.fmt(f),
            Self::Validation(e) => e.fmt(f),
            Self::BlobValidation(e) => e.fmt(f),
            Self::IdentityMismatch { expected, actual } => {
                write!(f, "candidate emitted {expected}, reopened {actual}")
            }
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for CandidatePublicationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Emission(e) => Some(e),
            Self::Publication(e) => Some(e),
            Self::Validation(e) => Some(e),
            Self::BlobValidation(e) => Some(e),
            Self::IdentityMismatch { .. } => None,
        }
    }
}

impl DirectoryTensorSink {
    /// Publish checked unqualified candidate bytes with candidate.json, never a
    /// fabricated sealed evidence log. Uses the same sync/no-clobber commit,
    /// owned cleanup and post-commit warnings as the existing sealed route.
    /// Validates all blob digests and expected/actual PackageId under explicit
    /// receiving caps. Opaque blobs are not model/runtime/source qualification.
    /// Caller owns exclusive parent-directory and immutable pending-file custody;
    /// this is not adversarial-writer, atomic multi-file, power-loss or security proof.
    /// # Errors
    /// Existing destination, unchanged emitter error, actual pending-byte admission,
    /// identity mismatch or pre-commit I/O. No successful partial publication.
    pub fn publish_unqualified_candidate<R, E>(
        &mut self,
        limits: TritPackageReadLimits,
        emit: impl FnOnce(&mut File) -> Result<(R, PackageId), E>,
    ) -> Result<(R, Vec<PublicationWarning>), CandidatePublicationError<E>> {
        if limits.max_source_bytes == 0
            || limits.max_manifest_bytes == 0
            || limits.max_directory_bytes == 0
        {
            return Err(CandidatePublicationError::Publication(
                QuantEngineError::Limit("candidate receiving caps must be positive"),
            ));
        }
        self.publish_transaction(
            PublicationRecord::Candidate,
            true,
            TritModelCompanions {
                files: vec![],
                max_bytes: 0,
            },
            |file| {
                let (value, expected) = emit(file).map_err(CandidatePublicationError::Emission)?;
                file.flush()
                    .map_err(|e| CandidatePublicationError::Publication(QuantEngineError::Io(e)))?;
                let retained = file
                    .try_clone()
                    .map_err(|e| CandidatePublicationError::Publication(QuantEngineError::Io(e)))?;
                let mut reader = TritPackageReader::open(retained, limits)
                    .map_err(CandidatePublicationError::Validation)?;
                let actual = reader.package_id();
                if actual != expected {
                    return Err(CandidatePublicationError::IdentityMismatch { expected, actual });
                }
                for index in 0..reader.blobs().len() {
                    reader
                        .visit_blob(index, |_| Ok::<_, std::convert::Infallible>(()))
                        .map_err(CandidatePublicationError::BlobValidation)?;
                }
                reader
                    .verify()
                    .map_err(CandidatePublicationError::Validation)?;
                let bytes =
                    serde_json::to_vec(&UnqualifiedCandidateRecord::new(actual)).map_err(|e| {
                        CandidatePublicationError::Publication(QuantEngineError::Source(format!(
                            "candidate metadata: {e}"
                        )))
                    })?;
                Ok((value, Some(bytes)))
            },
        )
        .map_err(|e| match e {
            WriteFailure::Emitter(e) => e,
            WriteFailure::Publication(e) => CandidatePublicationError::Publication(e),
        })
    }
}
