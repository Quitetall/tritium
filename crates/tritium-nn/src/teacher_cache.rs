//! Streaming access to dense-f32 offline teacher probabilities.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use tritium_format::{
    FormatError, TEACHER_CACHE_HEADER_BYTES, TeacherCacheHeader, read_teacher_cache_header,
    write_teacher_cache_header,
};

/// Stable invalidation digest for an ordered set of dense teacher weights.
pub fn hash_teacher_weights<'a>(weights: impl IntoIterator<Item = &'a [f32]>) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-teacher-weights-v1");
    for weight in weights {
        hash.update(&(weight.len() as u64).to_le_bytes());
        for &value in weight {
            hash.update(&value.to_bits().to_le_bytes());
        }
    }
    *hash.finalize().as_bytes()
}

/// Stable invalidation digest for ordered tokens and their window geometry.
#[must_use]
pub fn hash_teacher_corpus(tokens: &[u32], seq_len: u32) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-teacher-corpus-v1");
    hash.update(&seq_len.to_le_bytes());
    hash.update(&(tokens.len() as u64).to_le_bytes());
    for &token in tokens {
        hash.update(&token.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

/// Teacher-cache I/O, format, or invalidation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TeacherCacheError {
    /// Filesystem operation failed.
    Io(String),
    /// Header encoding/decoding failed.
    Format(FormatError),
    /// On-disk shape or identity hashes do not match the requested cache key.
    KeyMismatch,
    /// File length disagrees with the header's exact dense payload size.
    LengthMismatch { expected: u64, got: u64 },
    /// A window or destination has the wrong number of elements.
    WindowShape { expected: usize, got: usize },
    /// Window index exceeds the declared cache length.
    WindowOutOfRange { index: u64, windows: u64 },
    /// Writer was finished before all declared windows were written.
    Incomplete { expected: u64, got: u64 },
}

impl fmt::Display for TeacherCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "teacher cache I/O: {message}"),
            Self::Format(error) => write!(f, "teacher cache format: {error}"),
            Self::KeyMismatch => write!(f, "teacher cache key does not match model/corpus/shape"),
            Self::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "teacher cache length: expected {expected} bytes, got {got}"
                )
            }
            Self::WindowShape { expected, got } => {
                write!(
                    f,
                    "teacher cache window: expected {expected} values, got {got}"
                )
            }
            Self::WindowOutOfRange { index, windows } => {
                write!(f, "teacher cache window {index} is outside 0..{windows}")
            }
            Self::Incomplete { expected, got } => {
                write!(
                    f,
                    "teacher cache incomplete: expected {expected} windows, wrote {got}"
                )
            }
        }
    }
}

impl std::error::Error for TeacherCacheError {}

impl From<std::io::Error> for TeacherCacheError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<FormatError> for TeacherCacheError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Sequential writer for an exact number of dense probability windows.
pub struct TeacherCacheWriter {
    writer: BufWriter<File>,
    header: TeacherCacheHeader,
    window_elements: usize,
    written: u64,
    bytes: Vec<u8>,
}

impl fmt::Debug for TeacherCacheWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeacherCacheWriter")
            .field("header", &self.header)
            .field("written", &self.written)
            .finish_non_exhaustive()
    }
}

impl TeacherCacheWriter {
    /// Create or truncate a cache and write its validated fixed header.
    pub fn create(
        path: impl AsRef<Path>,
        header: TeacherCacheHeader,
    ) -> Result<Self, TeacherCacheError> {
        let window_elements = header.window_elements()?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&write_teacher_cache_header(&header)?)?;
        Ok(Self {
            writer,
            header,
            window_elements,
            written: 0,
            bytes: Vec::with_capacity(window_elements * 4),
        })
    }

    /// Append one `[seq_len, vocab]` row-major dense-f32 probability window.
    pub fn write_window(&mut self, probabilities: &[f32]) -> Result<(), TeacherCacheError> {
        if probabilities.len() != self.window_elements {
            return Err(TeacherCacheError::WindowShape {
                expected: self.window_elements,
                got: probabilities.len(),
            });
        }
        if self.written >= self.header.windows {
            return Err(TeacherCacheError::WindowOutOfRange {
                index: self.written,
                windows: self.header.windows,
            });
        }
        self.bytes.clear();
        for &probability in probabilities {
            self.bytes.extend_from_slice(&probability.to_le_bytes());
        }
        self.writer.write_all(&self.bytes)?;
        self.written += 1;
        Ok(())
    }

    /// Flush and fsync a complete cache.
    pub fn finish(mut self) -> Result<(), TeacherCacheError> {
        if self.written != self.header.windows {
            return Err(TeacherCacheError::Incomplete {
                expected: self.header.windows,
                got: self.written,
            });
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

/// Random-access reader with one reusable encoded-window scratch buffer.
pub struct TeacherCacheReader {
    reader: BufReader<File>,
    header: TeacherCacheHeader,
    window_elements: usize,
    window_bytes: u64,
    bytes: Vec<u8>,
}

impl fmt::Debug for TeacherCacheReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeacherCacheReader")
            .field("header", &self.header)
            .finish_non_exhaustive()
    }
}

impl TeacherCacheReader {
    /// Open a complete cache and fail closed if any model/corpus/shape key differs.
    pub fn open(
        path: impl AsRef<Path>,
        expected: &TeacherCacheHeader,
    ) -> Result<Self, TeacherCacheError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut encoded = [0u8; TEACHER_CACHE_HEADER_BYTES];
        reader.read_exact(&mut encoded)?;
        let header = read_teacher_cache_header(&encoded)?;
        if &header != expected {
            return Err(TeacherCacheError::KeyMismatch);
        }
        let payload_bytes = header.payload_bytes()?;
        let expected_len = (TEACHER_CACHE_HEADER_BYTES as u64)
            .checked_add(payload_bytes)
            .ok_or(FormatError::TeacherCacheInvalidShape)?;
        if file_len != expected_len {
            return Err(TeacherCacheError::LengthMismatch {
                expected: expected_len,
                got: file_len,
            });
        }
        let window_elements = header.window_elements()?;
        let window_bytes = (window_elements as u64) * 4;
        Ok(Self {
            reader,
            header,
            window_elements,
            window_bytes,
            bytes: vec![0; window_elements * 4],
        })
    }

    /// Parsed, validated cache identity and shape.
    #[must_use]
    pub fn header(&self) -> &TeacherCacheHeader {
        &self.header
    }

    /// Read one window into a caller-owned f32 buffer.
    pub fn read_window(&mut self, index: u64, out: &mut [f32]) -> Result<(), TeacherCacheError> {
        if index >= self.header.windows {
            return Err(TeacherCacheError::WindowOutOfRange {
                index,
                windows: self.header.windows,
            });
        }
        if out.len() != self.window_elements {
            return Err(TeacherCacheError::WindowShape {
                expected: self.window_elements,
                got: out.len(),
            });
        }
        let offset = (TEACHER_CACHE_HEADER_BYTES as u64)
            .checked_add(
                index
                    .checked_mul(self.window_bytes)
                    .ok_or(FormatError::TeacherCacheInvalidShape)?,
            )
            .ok_or(FormatError::TeacherCacheInvalidShape)?;
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(&mut self.bytes)?;
        for (value, encoded) in out.iter_mut().zip(self.bytes.as_chunks::<4>().0.iter()) {
            *value = f32::from_le_bytes(*encoded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("tritium-{name}-{}", std::process::id()))
    }

    fn header() -> TeacherCacheHeader {
        TeacherCacheHeader {
            seq_len: 2,
            vocab: 3,
            windows: 2,
            model_hash: [1; 32],
            corpus_hash: [2; 32],
        }
    }

    #[test]
    fn streaming_teacher_cache_roundtrips_random_access() {
        let path = path("teacher-cache-roundtrip.ttpr");
        let first = [0.1, 0.2, 0.7, 0.0, 0.5, 0.5];
        let second = [0.4, 0.3, 0.3, 0.8, 0.1, 0.1];
        let mut writer = TeacherCacheWriter::create(&path, header()).unwrap();
        writer.write_window(&first).unwrap();
        writer.write_window(&second).unwrap();
        writer.finish().unwrap();

        let mut reader = TeacherCacheReader::open(&path, &header()).unwrap();
        let mut out = [0.0; 6];
        reader.read_window(1, &mut out).unwrap();
        assert_eq!(out.map(f32::to_bits), second.map(f32::to_bits));
        reader.read_window(0, &mut out).unwrap();
        assert_eq!(out.map(f32::to_bits), first.map(f32::to_bits));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn teacher_cache_fails_closed_on_key_or_length() {
        let path = path("teacher-cache-invalid.ttpr");
        let mut writer = TeacherCacheWriter::create(&path, header()).unwrap();
        writer.write_window(&[0.0; 6]).unwrap();
        assert!(matches!(
            writer.finish(),
            Err(TeacherCacheError::Incomplete { .. })
        ));
        assert!(matches!(
            TeacherCacheReader::open(&path, &header()),
            Err(TeacherCacheError::LengthMismatch { .. })
        ));
        let mut wrong = header();
        wrong.corpus_hash[0] ^= 1;
        assert!(matches!(
            TeacherCacheReader::open(&path, &wrong),
            Err(TeacherCacheError::KeyMismatch | TeacherCacheError::LengthMismatch { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn teacher_cache_hashes_are_order_and_geometry_sensitive() {
        let a = [1.0, -0.0, 3.0];
        let b = [4.0, 5.0];
        assert_ne!(
            hash_teacher_weights([&a[..], &b[..]]),
            hash_teacher_weights([&b[..], &a[..]])
        );
        assert_ne!(
            hash_teacher_corpus(&[1, 2, 3], 2),
            hash_teacher_corpus(&[1, 2, 3], 3)
        );
        assert_ne!(
            hash_teacher_corpus(&[1, 2, 3], 2),
            hash_teacher_corpus(&[1, 3, 2], 2)
        );
    }
}
