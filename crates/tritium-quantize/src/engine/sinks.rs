//! No-overwrite destinations for a checked package and actual sealed fit log.
use super::companions::{TritModelCompanions, TritModelSpool, copy_companion, verify_companion};
use super::{PublicationWarning, QuantEngineError, TensorSink};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, Write},
    path::PathBuf,
};
use tritium_evidence::SealedEvidence;
use tritium_format::trit_model::{
    TritModelReader, TritModelStageError, stage_trit_model_from_spool_with_admission,
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
        self.publish_with_emitter(evidence.jsonl(), false, emit)
            .map_err(|error| match error {
                WriteFailure::Emitter(error) | WriteFailure::Publication(error) => error,
            })
    }

    /// Republish exact existing package/log bytes after their complete shared check.
    /// This preserves the original run and identities; it does not create a new
    /// recorder seal, refit, qualification verdict or distribution authorization.
    /// Caller owns immutable input buffers and their physical byte admission.
    /// Existing sync, no-clobber commit, cleanup and warning semantics apply.
    /// # Errors
    /// Invalid or mismatched package/log, existing destination or pre-commit I/O.
    pub fn publish_verified_model(
        &mut self,
        package: &[u8],
        original_jsonl: &str,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        let log = tritium_evidence::verify_jsonl(original_jsonl)?;
        super::verify_package_with_fit_log(package, &log)?;
        self.publish_with_emitter(original_jsonl, false, |file| {
            file.write_all(package).map_err(QuantEngineError::Io)
        })
        .map(|(_, warnings)| warnings)
        .map_err(|error| match error {
            WriteFailure::Emitter(error) | WriteFailure::Publication(error) => error,
        })
    }

    /// Republish exact retained-source bytes without a complete package buffer.
    ///
    /// Admits the canonical original log, complete typed source binding and the
    /// expected exact PackageId before creating output. Copies checked bounded
    /// chunks into the common owned pending file, admits that file through the
    /// same checker and verifies the retained source before no-clobber commit.
    /// No recorder seal, rewritten log, refit or distribution approval is made.
    /// Caller owns immutable source and parent-directory custody; this is not an
    /// atomic external-writer snapshot, process-RSS or power-loss/security gate.
    /// # Errors
    /// Invalid binding/identity, changed source, byte limits, original source or
    /// destination I/O, existing output or other pre-commit publication failure.
    pub fn publish_verified_model_reader<R: Read + Seek>(
        &mut self,
        source: R,
        limits: TritPackageReadLimits,
        expected: tritium_format::PackageId,
        original_jsonl: &str,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        let log = tritium_evidence::verify_jsonl(original_jsonl)?;
        let mut model = super::open_model_reader_with_fit_log(source, limits, &log)?;
        if model.package_id() != expected {
            return Err(QuantEngineError::Binding("export package identity changed"));
        }
        self.publish_with_emitter(original_jsonl, true, |file| {
            model
                .visit_package(|chunk| file.write_all(chunk))
                .map_err(QuantEngineError::ModelVisit)?;
            let staged = super::open_model_reader_with_fit_log(file, limits, &log)?;
            if staged.package_id() != expected {
                return Err(QuantEngineError::Binding("staged export identity changed"));
            }
            model.verify()?;
            Ok(())
        })
        .map(|(_, warnings)| warnings)
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
        self.publish_spooled_model_with_companions(
            TritModelSpool {
                manifest,
                blobs,
                source: spool,
                limits,
            },
            evidence,
            TritModelCompanions {
                files: Vec::new(),
                max_bytes: 0,
            },
            |_| source_guard(),
        )
        .map(|(model, (), warnings)| (model, warnings))
    }

    /// Stage a complete typed model and bounded hashed companions in one commit.
    /// Companions are copied/synced before staging; retained handles are rechecked
    /// after the source guard and before the no-clobber model hard link. Caller
    /// owns exclusive source/directory custody; not adversarial-writer, RSS,
    /// power-loss/security, lineage or distribution qualification.
    pub fn publish_spooled_model_with_companions<S: Read + Seek, E, T>(
        &mut self,
        staging: TritModelSpool<'_, S>,
        evidence: &SealedEvidence,
        companions: TritModelCompanions<'_>,
        admit: impl FnOnce(&mut TritModelReader<File>) -> Result<T, E>,
    ) -> Result<(TritModelReader<File>, T, Vec<PublicationWarning>), TritModelPublicationError<E>>
    {
        {
            use tritium_format::trit_model::{TritModelError, encode_model_manifest};
            use tritium_format::{TritPackageError, inspect_trit_spool_layout};
            let metadata = encode_model_manifest(staging.manifest).map_err(|error| {
                TritModelPublicationError::Staging(TritModelStageError::Model(error))
            })?;
            let size = inspect_trit_spool_layout(&metadata, staging.blobs).map_err(|error| {
                TritModelPublicationError::Staging(TritModelStageError::Model(
                    TritModelError::Package(error),
                ))
            })?;
            if size.package_bytes > staging.limits.max_source_bytes {
                return Err(TritModelPublicationError::Staging(
                    TritModelStageError::Model(TritModelError::Package(
                        TritPackageError::PackageLimitExceeded {
                            requested: usize::try_from(size.package_bytes).unwrap_or(usize::MAX),
                            limit: usize::try_from(staging.limits.max_source_bytes)
                                .unwrap_or(usize::MAX),
                        },
                    )),
                ));
            }
            if size.manifest_bytes > staging.limits.max_manifest_bytes as u64
                || size.directory_bytes > staging.limits.max_directory_bytes as u64
            {
                return Err(TritModelPublicationError::Staging(
                    TritModelStageError::Model(TritModelError::Package(
                        TritPackageError::MetadataLimitExceeded,
                    )),
                ));
            }
        }
        self.publish_with_companions(evidence.jsonl(), true, companions, |file| {
            let destination = file.try_clone().map_err(TritModelStageError::Destination)?;
            stage_trit_model_from_spool_with_admission(
                staging.manifest,
                staging.blobs,
                staging.source,
                staging.limits,
                destination,
                admit,
            )
        })
        .map(|((model, value), warnings)| (model, value, warnings))
        .map_err(|error| match error {
            WriteFailure::Emitter(error) => TritModelPublicationError::Staging(error),
            WriteFailure::Publication(error) => TritModelPublicationError::Publication(error),
        })
    }

    fn publish_with_emitter<R, E>(
        &mut self,
        original_jsonl: &str,
        readable: bool,
        emit: impl FnOnce(&mut File) -> Result<R, E>,
    ) -> Result<(R, Vec<PublicationWarning>), WriteFailure<E>> {
        self.publish_with_companions(
            original_jsonl,
            readable,
            TritModelCompanions {
                files: Vec::new(),
                max_bytes: 0,
            },
            emit,
        )
    }

    fn publish_with_companions<R, E>(
        &mut self,
        original_jsonl: &str,
        readable: bool,
        mut companions: TritModelCompanions<'_>,
        emit: impl FnOnce(&mut File) -> Result<R, E>,
    ) -> Result<(R, Vec<PublicationWarning>), WriteFailure<E>> {
        companions.validate().map_err(WriteFailure::Publication)?;
        let mut owned_companions = Vec::new();
        owned_companions
            .try_reserve_exact(companions.files.len())
            .map_err(|_| WriteFailure::Publication(QuantEngineError::Allocation))?;
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
            file.write_all(original_jsonl.as_bytes())?;
            file.sync_all()?;
            for companion in &mut companions.files {
                let path = self.directory.join(companion.name);
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                owned_companions.push((path, file, companion.bytes, companion.content));
                let file = &mut owned_companions.last_mut().expect("just inserted").1;
                copy_companion(companion.source, file, companion.bytes, companion.content)
                    .map_err(WriteFailure::Publication)?;
            }
            let mut file = OpenOptions::new()
                .read(readable)
                .write(true)
                .create_new(true)
                .open(&pending)?;
            owned_pending = true;
            let value = emit(&mut file).map_err(WriteFailure::Emitter)?;
            file.sync_all()?;
            for (_, companion, bytes, content) in &mut owned_companions {
                verify_companion(companion, *bytes, *content).map_err(WriteFailure::Publication)?;
            }
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
                for (path, file, _, _) in owned_companions {
                    drop(file);
                    let _ = fs::remove_file(path);
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
