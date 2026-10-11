//! No-overwrite destinations for a checked package and actual sealed fit log.
use super::{PublicationWarning, QuantEngineError, TensorSink};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::PathBuf,
};
use tritium_evidence::SealedEvidence;

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
}
impl TensorSink for DirectoryTensorSink {
    fn publish(
        &mut self,
        package: &[u8],
        evidence: &SealedEvidence,
    ) -> Result<Vec<PublicationWarning>, QuantEngineError> {
        fs::create_dir(&self.directory).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                QuantEngineError::DestinationExists
            } else {
                QuantEngineError::Io(e)
            }
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
                .write(true)
                .create_new(true)
                .open(&pending)?;
            owned_pending = true;
            file.write_all(package)?;
            file.sync_all()?;
            fs::hard_link(&pending, &model)?;
            Ok::<(), std::io::Error>(())
        })();
        if let Err(e) = result {
            if owned_pending {
                let _ = fs::remove_file(&pending);
            }
            if owned_log {
                let _ = fs::remove_file(&log);
            }
            // Never recursively remove unexpected/foreign files injected here.
            let _ = fs::remove_dir(&self.directory);
            return Err(QuantEngineError::Io(e));
        }
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
        Ok(warnings)
    }
}
