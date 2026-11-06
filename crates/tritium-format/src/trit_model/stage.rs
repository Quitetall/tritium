//! Compose the canonical spool writer with complete retained typed admission.
use super::{TritModelError, TritModelReadError, TritModelReader, encode_model_manifest};
use crate::{PackageId, TritPackageReadLimits, TritPackageWriteError, TritSpoolBlob};
use std::io::{self, Read, Seek, SeekFrom, Write};
use tritium_schema::ModelManifest;

/// Staging refusal; a failed pending stream is never publication authorization.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritModelStageError<E> {
    /// Typed metadata or complete inventory failed admission.
    Model(TritModelError),
    /// Original canonical spool source/package/destination writer error.
    Write(TritPackageWriteError),
    /// Retained package/typed payload or terminal output verification failed.
    Read(TritModelReadError),
    /// Pending destination seek/flush failed before retained admission.
    Destination(io::Error),
    /// Pending destination already contains bytes; no bytes were overwritten.
    NonemptyDestination,
    /// Valid reopened bytes are not the exact bytes emitted by this writer.
    IdentityMismatch {
        /// Incremental identity of the canonical emitted package.
        expected: PackageId,
        /// Initial identity admitted by the retained reader.
        actual: PackageId,
    },
    /// Original caller-owned whole-source guard refusal.
    Guard(E),
}
impl<E: core::fmt::Display> core::fmt::Display for TritModelStageError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::Write(error) => error.fmt(f),
            Self::Read(error) => error.fmt(f),
            Self::Destination(error) => write!(f, "trit model pending destination: {error}"),
            Self::NonemptyDestination => f.write_str("trit model pending destination is not empty"),
            Self::IdentityMismatch { expected, actual } => {
                write!(f, "trit model emitted {expected}, reopened {actual}")
            }
            Self::Guard(error) => write!(f, "trit model whole-source guard: {error}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for TritModelStageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Write(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::Destination(error) => Some(error),
            Self::Guard(error) => Some(error),
            _ => None,
        }
    }
}

/// Stage and reopen an exact complete typed model without a package/payload Vec.
///
/// The supplied read/write/seek destination must be an empty owned pending stream.
/// Uses the original Rust manifest encoder and canonical bounded spool writer,
/// flushes, then admits the output through the shared retained typed reader.
/// Requires complete inventory and the writer's exact PackageId. Only then calls
/// `source_guard` once, followed by another complete output identity verification.
/// Legacy callers use their original strict reader's final `verify_unchanged`.
///
/// Returns the retained output handle, not a publication receipt. Caller owns
/// cleanup on partial failure, immutable source/spool custody, disk/aggregate
/// memory admission, original shape/assets/legacy lineage, bit/runtime proof,
/// fsync and atomic no-clobber publication. Flush is not durable sync. No atomic
/// snapshot against external writers, profile acceptance or fit provenance is
/// inferred. Encoded caps are not allocator, caller output, page-cache or RSS caps.
/// # Errors
/// Nonempty destinations are untouched. Metadata/source/writer/I/O/typed/guard
/// refusals return no retained success; errors after emission may leave a tail.
pub fn stage_trit_model_from_spool<S: Read + Seek, R: Read + Write + Seek, E>(
    manifest: &ModelManifest,
    blobs: &[TritSpoolBlob],
    spool: &mut S,
    limits: TritPackageReadLimits,
    mut destination: R,
    source_guard: impl FnOnce() -> Result<(), E>,
) -> Result<TritModelReader<R>, TritModelStageError<E>> {
    let metadata = encode_model_manifest(manifest).map_err(TritModelStageError::Model)?;
    if destination
        .seek(SeekFrom::End(0))
        .map_err(TritModelStageError::Destination)?
        != 0
    {
        return Err(TritModelStageError::NonemptyDestination);
    }
    if destination
        .seek(SeekFrom::Start(0))
        .map_err(TritModelStageError::Destination)?
        != 0
    {
        return Err(TritModelStageError::Destination(io::Error::other(
            "pending destination returned a wrong seek position",
        )));
    }
    let max = usize::try_from(limits.max_source_bytes).unwrap_or(usize::MAX);
    let expected =
        crate::write_trit_package_from_spool(&metadata, blobs, spool, max, &mut destination)
            .map_err(TritModelStageError::Write)?;
    destination
        .flush()
        .map_err(TritModelStageError::Destination)?;
    let mut reader =
        TritModelReader::open(destination, limits).map_err(TritModelStageError::Read)?;
    reader
        .validate_complete_payload_inventory()
        .map_err(TritModelStageError::Model)?;
    let actual = reader.package_id();
    if actual != expected {
        return Err(TritModelStageError::IdentityMismatch { expected, actual });
    }
    source_guard().map_err(TritModelStageError::Guard)?;
    reader.verify().map_err(TritModelStageError::Read)?;
    Ok(reader)
}
