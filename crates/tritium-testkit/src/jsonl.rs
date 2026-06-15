//! JSONL persistence for conformance vectors: one JSON object per line.
//!
//! JSONL (newline-delimited JSON) is the committed on-disk form of the
//! conformance suite — it diffs line-by-line, streams without loading the whole
//! file into a single JSON document, and lets a vector be appended without
//! rewriting the rest. Save then load is a lossless roundtrip.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::vector::ConformanceVector;

/// Errors from reading or writing a JSONL vector file.
#[derive(Debug)]
#[non_exhaustive]
pub enum JsonlError {
    /// An underlying filesystem / IO failure.
    Io(io::Error),
    /// A line failed to (de)serialize, with the 1-based line number for context.
    Json {
        /// 1-based line number the error occurred on (`0` on write).
        line: usize,
        /// The underlying serde_json error.
        source: serde_json::Error,
    },
}

impl std::fmt::Display for JsonlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonlError::Io(e) => write!(f, "io error: {e}"),
            JsonlError::Json { line, source } => {
                write!(f, "json error on line {line}: {source}")
            }
        }
    }
}

impl std::error::Error for JsonlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JsonlError::Io(e) => Some(e),
            JsonlError::Json { source, .. } => Some(source),
        }
    }
}

impl From<io::Error> for JsonlError {
    fn from(e: io::Error) -> Self {
        JsonlError::Io(e)
    }
}

/// Write `vectors` to `path` as JSONL — one compact JSON object per line.
///
/// Overwrites any existing file. The output is buffered and flushed before
/// return.
///
/// # Errors
/// [`JsonlError::Io`] on any filesystem failure; [`JsonlError::Json`] if a vector
/// cannot be serialized (cannot happen for the plain-data [`ConformanceVector`],
/// but is surfaced rather than panicked on).
pub fn save_vectors(
    path: impl AsRef<Path>,
    vectors: &[ConformanceVector],
) -> Result<(), JsonlError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for v in vectors {
        let line =
            serde_json::to_string(v).map_err(|source| JsonlError::Json { line: 0, source })?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a JSONL vector file from `path`, one [`ConformanceVector`] per line.
///
/// Blank lines (including a trailing newline) are skipped, so the result of
/// [`save_vectors`] round-trips exactly.
///
/// # Errors
/// [`JsonlError::Io`] if the file cannot be opened or read; [`JsonlError::Json`]
/// (with the offending line number) if any line is not a valid vector.
pub fn load_vectors(path: impl AsRef<Path>) -> Result<Vec<ConformanceVector>, JsonlError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v = serde_json::from_str(&line).map_err(|source| JsonlError::Json {
            line: i + 1,
            source,
        })?;
        out.push(v);
    }
    Ok(out)
}
