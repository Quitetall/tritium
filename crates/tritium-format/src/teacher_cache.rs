//! Dense-f32 offline teacher-probability cache header.

use crate::{FormatError, le_cursor::LeCursor};

/// `Tritium Teacher Probabilities` cache magic.
pub const TEACHER_CACHE_MAGIC: [u8; 4] = *b"TTPR";
/// Current teacher-cache format version.
pub const TEACHER_CACHE_VERSION: u8 = 1;
/// Fixed header size in bytes.
pub const TEACHER_CACHE_HEADER_BYTES: usize = 88;

/// Shape and invalidation keys for a window-major dense-f32 probability cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeacherCacheHeader {
    /// Token rows per cached training window.
    pub seq_len: u32,
    /// Dense probability columns per row.
    pub vocab: u32,
    /// Number of cached windows.
    pub windows: u64,
    /// BLAKE3 digest of the teacher/model identity chosen by the producer.
    pub model_hash: [u8; 32],
    /// BLAKE3 digest of the ordered token corpus/windowing identity.
    pub corpus_hash: [u8; 32],
}

impl TeacherCacheHeader {
    /// Number of f32 values in one window.
    pub fn window_elements(&self) -> Result<usize, FormatError> {
        if self.seq_len == 0 || self.vocab == 0 || self.windows == 0 {
            return Err(FormatError::TeacherCacheInvalidShape);
        }
        (self.seq_len as usize)
            .checked_mul(self.vocab as usize)
            .ok_or(FormatError::TeacherCacheInvalidShape)
    }

    /// Total dense-f32 payload size, excluding the fixed header.
    pub fn payload_bytes(&self) -> Result<u64, FormatError> {
        let elements = u64::try_from(self.window_elements()?)
            .map_err(|_| FormatError::TeacherCacheInvalidShape)?;
        elements
            .checked_mul(self.windows)
            .and_then(|count| count.checked_mul(4))
            .ok_or(FormatError::TeacherCacheInvalidShape)
    }
}

/// Encode a fixed-size little-endian cache header.
pub fn write_teacher_cache_header(header: &TeacherCacheHeader) -> Result<Vec<u8>, FormatError> {
    let _ = header.payload_bytes()?;
    let mut out = Vec::with_capacity(TEACHER_CACHE_HEADER_BYTES);
    out.extend_from_slice(&TEACHER_CACHE_MAGIC);
    out.push(TEACHER_CACHE_VERSION);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&header.seq_len.to_le_bytes());
    out.extend_from_slice(&header.vocab.to_le_bytes());
    out.extend_from_slice(&header.windows.to_le_bytes());
    out.extend_from_slice(&header.model_hash);
    out.extend_from_slice(&header.corpus_hash);
    debug_assert_eq!(out.len(), TEACHER_CACHE_HEADER_BYTES);
    Ok(out)
}

/// Decode and validate a fixed-size cache header.
pub fn read_teacher_cache_header(bytes: &[u8]) -> Result<TeacherCacheHeader, FormatError> {
    if bytes.len() != TEACHER_CACHE_HEADER_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TEACHER_CACHE_HEADER_BYTES,
            got: bytes.len(),
        });
    }
    let mut cursor = LeCursor::new(bytes);
    if cursor.take(4)? != TEACHER_CACHE_MAGIC {
        return Err(FormatError::TeacherCacheBadMagic);
    }
    let version = cursor.u8()?;
    if version != TEACHER_CACHE_VERSION {
        return Err(FormatError::UnsupportedTeacherCacheVersion(version));
    }
    let _reserved = cursor.take(3)?;
    let seq_len = cursor.u32()?;
    let vocab = cursor.u32()?;
    let windows = cursor.u64()?;
    let mut model_hash = [0; 32];
    model_hash.copy_from_slice(cursor.take(32)?);
    let mut corpus_hash = [0; 32];
    corpus_hash.copy_from_slice(cursor.take(32)?);
    let header = TeacherCacheHeader {
        seq_len,
        vocab,
        windows,
        model_hash,
        corpus_hash,
    };
    let _ = header.payload_bytes()?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_cache_header_roundtrips_and_sizes_windows() {
        let header = TeacherCacheHeader {
            seq_len: 32,
            vocab: 49_152,
            windows: 40,
            model_hash: [0x11; 32],
            corpus_hash: [0x22; 32],
        };
        let encoded = write_teacher_cache_header(&header).unwrap();
        assert_eq!(encoded.len(), TEACHER_CACHE_HEADER_BYTES);
        assert_eq!(read_teacher_cache_header(&encoded).unwrap(), header);
        assert_eq!(header.window_elements().unwrap(), 32 * 49_152);
        assert_eq!(header.payload_bytes().unwrap(), 40 * 32 * 49_152 * 4);
    }

    #[test]
    fn teacher_cache_header_rejects_corruption() {
        let header = TeacherCacheHeader {
            seq_len: 1,
            vocab: 2,
            windows: 3,
            model_hash: [0; 32],
            corpus_hash: [0; 32],
        };
        let mut encoded = write_teacher_cache_header(&header).unwrap();
        encoded[0] ^= 0xff;
        assert!(read_teacher_cache_header(&encoded).is_err());
        let encoded = write_teacher_cache_header(&header).unwrap();
        assert!(read_teacher_cache_header(&encoded[..encoded.len() - 1]).is_err());

        let mut zero_shape = header;
        zero_shape.seq_len = 0;
        assert!(write_teacher_cache_header(&zero_shape).is_err());
    }
}
