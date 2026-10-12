//! No-overwrite destinations for a checked package and actual sealed fit log.
use super::{PublicationWarning, QuantEngineError, TensorSink};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, Write},
    path::PathBuf,
};
use tritium_evidence::SealedEvidence;
use tritium_format::trit_model::{
    TritModelReader, TritModelStageError, stage_trit_model_from_spool,
};
use tritium_format::{TritPackageReadLimits, TritSpoolBlob};
use tritium_schema::ModelManifest;

/// Original typed staging refusal or failure before the no-clobber commit.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritModelPublicationError<E> {
    /// Shared model/spool/output validation or original source guard failure.
    Staging(TritModelStageError<E>),
    /// Existing directory publication error; post-commit caveats are warnings.
    Publication(QuantEngineError),
}
impl<E: core::fmt::Display> core::fmt::Display for TritModelPublicationError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Staging(error) => error.fmt(f),
            Self::Publication(error) => error.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for TritModelPublicationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Staging(error) => Some(error),
            Self::Publication(error) => Some(error),
        }
    }
}

enum WriteFailure<E> {
    Emitter(E),
    Publication(QuantEngineError),
}
impl<E> From<std::io::Error> for WriteFailure<E> {
    fn from(error: std::io::Error) -> Self {
        Self::Publication(QuantEngineError::Io(error))
    }
}

/// Owned in-memory destination, refusing a second publication.
#[derive(Debug, Default)]
pub struct MemoryTensorSink {
    published: Option<(Vec<u8>, SealedEvidence)>,
}
impl MemoryTensorSink {
    /// Actual checked package and sealed original fit log, if published.
    #[must_use]
    pub fn artifact(&self) -> Option<(&[u8], &SealedEvidence)> {
        self.published
            .as_ref()
            .map(|(bytes, evidence)| (bytes.as_slice(), evidence))
    }
}
impl TensorSink for MemoryTensorSink {
    fn publish(
        &mut self,
        package: &[u8],
        evidence: &SealedEvidence,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        if self.published.is_some() {
            return Err(QuantEngineError::DestinationExists);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(package.len())
            .map_err(|_| QuantEngineError::Allocation)?;
        bytes.extend_from_slice(package);
        self.published = Some((bytes, evidence.clone()));
        Ok(Vec::new())
    }
}

/// New directory destination. The no-clobber `model.trit` hard link is commit.
/// Does not replace preexisting paths or claim power-loss/security qualification.
#[derive(Debug)]
pub struct DirectoryTensorSink {
    directory: PathBuf,
}
impl DirectoryTensorSink {
    /// Select a new destination; existence is checked atomically at publication.
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Emit into one owned pending file and commit through the common no-clobber path.
    /// The callback must admit its inputs and report partial write errors. It does
    /// not publish; this module owns sync, hard-link commit, cleanup and warnings.
    /// No full output buffer is required. This is not power-loss/security qualification.
    /// # Errors
    /// Existing destination, callback error or filesystem failure before commit.
    pub fn publish_with_writer<R>(
        &mut self,
        evidence: &SealedEvidence,
        emit: impl FnOnce(&mut File) -> Result<R, QuantEngineError>,
    ) -> Result<(R, Vec<PublicationWarning>), QuantEngineError> {
        self.publish_with_emitter(evidence, false, emit)
            .map_err(|error| match error {
                WriteFailure::Emitter(error) | WriteFailure::Publication(error) => error,
            })
    }

    /// Validate bounded spool output before the existing sync/no-clobber commit.
    ///
    /// Uses the same publisher as ordinary/recorded-stage writers and the format
    /// module's shared typed staging seam. Retains a clone of the owned pending
    /// File handle, not its pathname or a complete package Vec. Original typed
    /// staging/source guard errors stay distinct from filesystem errors.
    ///
    /// Caller must admit the actual sealed log/model/source binding and immutable
    /// source/spool/parent-directory custody. This sink does not authenticate fit
    /// provenance, admit joined profiles, create an atomic external-writer snapshot,
    /// authorize distribution or clear quality/security/power-loss/RSS gates.
    /// Successful publication with post-commit trouble returns explicit warnings.
    /// # Errors
    /// Existing destination, staging/source/guard failure or pre-commit filesystem
    /// failure; common cleanup removes only owned log/pending paths.
    pub fn publish_spooled_model<S: Read + Seek, E>(
        &mut self,
        manifest: &ModelManifest,
        blobs: &[TritSpoolBlob],
        spool: &mut S,
        limits: TritPackageReadLimits,
        evidence: &SealedEvidence,
        source_guard: impl FnOnce() -> Result<(), E>,
    ) -> Result<(TritModelReader<File>, Vec<PublicationWarning>), TritModelPublicationError<E>>
    {
        self.publish_with_emitter(evidence, true, |file| {
            let destination = file.try_clone().map_err(TritModelStageError::Destination)?;
            stage_trit_model_from_spool(manifest, blobs, spool, limits, destination, source_guard)
        })
        .map_err(|error| match error {
            WriteFailure::Emitter(error) => TritModelPublicationError::Staging(error),
            WriteFailure::Publication(error) => TritModelPublicationError::Publication(error),
        })
    }

    fn publish_with_emitter<R, E>(
        &mut self,
        evidence: &SealedEvidence,
        readable: bool,
        emit: impl FnOnce(&mut File) -> Result<R, E>,
    ) -> Result<(R, Vec<PublicationWarning>), WriteFailure<E>> {
        fs::create_dir(&self.directory).map_err(|e| {
            WriteFailure::Publication(if e.kind() == std::io::ErrorKind::AlreadyExists {
                QuantEngineError::DestinationExists
            } else {
                QuantEngineError::Io(e)
            })
        })?;
        let log = self.directory.join("evidence.jsonl");
        let pending = self.directory.join(".pending.trit");
        let model = self.directory.join("model.trit");
        let mut owned_log = false;
        let mut owned_pending = false;
        let result = (|| {
            let mut file = OpenOptions::new().write(true).create_new(true).open(&log)?;
            owned_log = true;
            file.write_all(evidence.jsonl().as_bytes())?;
            file.sync_all()?;
            let mut file = OpenOptions::new()
                .read(readable)
                .write(true)
                .create_new(true)
                .open(&pending)?;
            owned_pending = true;
            let value = emit(&mut file).map_err(WriteFailure::Emitter)?;
            file.sync_all()?;
            fs::hard_link(&pending, &model)?;
            Ok::<R, WriteFailure<E>>(value)
        })();
        let value = match result {
            Ok(value) => value,
            Err(e) => {
                if owned_pending {
                    let _ = fs::remove_file(&pending);
                }
                if owned_log {
                    let _ = fs::remove_file(&log);
                }
                // Never recursively remove unexpected/foreign files injected here.
                let _ = fs::remove_dir(&self.directory);
                return Err(e);
            }
        };
        let mut warnings = Vec::new();
        if fs::remove_file(&pending).is_err() {
            warnings.push(PublicationWarning::PendingCleanup);
        }
        if File::open(&self.directory)
            .and_then(|dir| dir.sync_all())
            .is_err()
        {
            warnings.push(PublicationWarning::DirectorySync);
        }
        Ok((value, warnings))
    }
}
impl TensorSink for DirectoryTensorSink {
    fn publish(
        &mut self,
        package: &[u8],
        evidence: &SealedEvidence,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        self.publish_with_writer(evidence, |file| {
            file.write_all(package)?;
            Ok(())
        })
        .map(|(_, warnings)| warnings)
    }
}
