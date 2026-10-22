//! Bounded original companion streams for the common model commit transaction.
use super::QuantEngineError;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};
use tritium_schema::BlobId;

/// Complete typed spool inputs, shared by ordinary and companion publication.
#[derive(Debug)]
pub struct TritModelSpool<'a, S> {
    /// Complete canonical typed manifest.
    pub manifest: &'a tritium_schema::ModelManifest,
    /// Complete content-addressed source ranges.
    pub blobs: &'a [tritium_format::TritSpoolBlob],
    /// Caller-owned seekable spool, guarded independently before commit.
    pub source: &'a mut S,
    /// Existing typed receiving/output limits.
    pub limits: tritium_format::TritPackageReadLimits,
}

/// One exact companion stream; not a filesystem path supplied by model metadata.
pub struct TritModelCompanion<'a> {
    /// Portable lowercase ASCII leaf, excluding reserved model/log/device names.
    pub name: &'a str,
    /// Original bytes at the current position; exact length and EOF are checked.
    pub source: &'a mut dyn Read,
    /// Exact expected bytes.
    pub bytes: u64,
    /// BLAKE3 of those exact bytes.
    pub content: BlobId,
}
impl std::fmt::Debug for TritModelCompanion<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TritModelCompanion")
            .field("name", &self.name)
            .field("bytes", &self.bytes)
            .field("content", &self.content)
            .finish_non_exhaustive()
    }
}

/// At most64 distinct portable companions and their explicit aggregate payload cap.
#[derive(Debug)]
pub struct TritModelCompanions<'a> {
    /// Borrowed original streams; no companion-sized copy is retained.
    pub files: Vec<TritModelCompanion<'a>>,
    /// Aggregate expected byte ceiling; zero admits only zero-byte companions.
    pub max_bytes: u64,
}
impl TritModelCompanions<'_> {
    pub(super) fn validate(&self) -> Result<(), QuantEngineError> {
        if self.files.len() > 64 {
            return Err(QuantEngineError::Binding("too many model companions"));
        }
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        for file in &self.files {
            let name = file.name;
            let stem = name.split('.').next().unwrap_or_default();
            let device = matches!(stem, "con" | "prn" | "aux" | "nul")
                || (stem.len() == 4
                    && (stem.starts_with("com") || stem.starts_with("lpt"))
                    && matches!(stem.as_bytes()[3], b'1'..=b'9'));
            if name.is_empty()
                || name.len() > 128
                || !name.as_bytes()[0].is_ascii_lowercase() && !name.as_bytes()[0].is_ascii_digit()
                || !name.bytes().all(|b| {
                    b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
                })
                || name.ends_with('.')
                || matches!(name, "model.trit" | "evidence.jsonl")
                || device
                || !names.insert(name)
            {
                return Err(QuantEngineError::Binding(
                    "invalid or repeated model companion name",
                ));
            }
            total = total
                .checked_add(file.bytes)
                .ok_or(QuantEngineError::Binding("model companion byte overflow"))?;
        }
        if total > self.max_bytes {
            return Err(QuantEngineError::Binding(
                "model companion byte limit exceeded",
            ));
        }
        Ok(())
    }
}

pub(super) fn copy_companion(
    source: &mut dyn Read,
    destination: &mut File,
    bytes: u64,
    content: BlobId,
) -> Result<(), QuantEngineError> {
    let mut buffer = [0; 8192];
    let mut remaining = bytes;
    let mut hash = blake3::Hasher::new();
    while remaining > 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        source.read_exact(&mut buffer[..n])?;
        destination.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if source.read(&mut [0])? != 0 || hash.finalize().as_bytes() != content.as_bytes() {
        return Err(QuantEngineError::Binding(
            "model companion identity/length differs",
        ));
    }
    destination.sync_all()?;
    Ok(())
}

pub(super) fn verify_companion(
    file: &mut File,
    bytes: u64,
    content: BlobId,
) -> Result<(), QuantEngineError> {
    if file.metadata()?.len() != bytes || file.seek(SeekFrom::Start(0))? != 0 {
        return Err(QuantEngineError::Binding(
            "staged model companion length/seek differs",
        ));
    }
    let mut buffer = [0; 8192];
    let mut remaining = bytes;
    let mut hash = blake3::Hasher::new();
    while remaining > 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..n])?;
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if file.read(&mut [0])? != 0 || hash.finalize().as_bytes() != content.as_bytes() {
        return Err(QuantEngineError::Binding(
            "staged model companion identity differs",
        ));
    }
    file.sync_all()?;
    Ok(())
}
