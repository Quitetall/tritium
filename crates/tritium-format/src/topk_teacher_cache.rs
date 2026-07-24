//! Top-k sparse offline teacher-probability cache (Lever 3).
//!
//! The dense [`crate::TeacherCacheHeader`] stores a full-vocabulary f32 distribution per token
//! (`windows · seq_len · vocab · 4` bytes). Distillation only needs the teacher's high-probability
//! tokens, so this format stores just the top-`k` `(index, probability)` pairs per row. Each pair is
//! 8 bytes (u32 index + f32 prob) vs a dense 4-byte prob, so the payload shrinks by `vocab/(2·k)`
//! (e.g. 384× at `k=64`, `vocab≈49k`). The pairs feed [`crate`]'s consumer straight into
//! `topk_kd` (`tritium_train::ops::loss`). Layout is window-major, and within a window **SoA**: all
//! `seq_len·k` indices (u32 LE) then all `seq_len·k` probabilities (f32 LE), so a reader hands the two
//! contiguous slices to the loss op without a transpose.

use crate::{FormatError, le_cursor::LeCursor};

/// `Tritium Teacher Probabilities top-K` cache magic.
pub const TOPK_TEACHER_CACHE_MAGIC: [u8; 4] = *b"TTPK";
/// Current top-k teacher-cache format version.
pub const TOPK_TEACHER_CACHE_VERSION: u8 = 1;
/// Fixed header size in bytes.
pub const TOPK_TEACHER_CACHE_HEADER_BYTES: usize = 92;

/// Shape and invalidation keys for a window-major top-k probability cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopkTeacherCacheHeader {
    /// Token rows per cached training window.
    pub seq_len: u32,
    /// Teacher vocabulary size (indices are validated `< vocab`).
    pub vocab: u32,
    /// Retained probability entries per row (`1 ≤ top_k ≤ vocab`).
    pub top_k: u32,
    /// Number of cached windows.
    pub windows: u64,
    /// BLAKE3 digest of the teacher/model identity chosen by the producer.
    pub model_hash: [u8; 32],
    /// BLAKE3 digest of the ordered token corpus/windowing identity.
    pub corpus_hash: [u8; 32],
}

impl TopkTeacherCacheHeader {
    /// `(index, probability)` pairs in one window (`seq_len · top_k`).
    pub fn window_pairs(&self) -> Result<usize, FormatError> {
        if self.seq_len == 0 || self.vocab == 0 || self.windows == 0 {
            return Err(FormatError::TeacherCacheInvalidShape);
        }
        if self.top_k == 0 || self.top_k > self.vocab {
            return Err(FormatError::TeacherCacheInvalidShape);
        }
        (self.seq_len as usize)
            .checked_mul(self.top_k as usize)
            .ok_or(FormatError::TeacherCacheInvalidShape)
    }

    /// Bytes in one window: each pair is a u32 index + an f32 probability (8 bytes).
    pub fn window_bytes(&self) -> Result<u64, FormatError> {
        let pairs = u64::try_from(self.window_pairs()?)
            .map_err(|_| FormatError::TeacherCacheInvalidShape)?;
        pairs
            .checked_mul(8)
            .ok_or(FormatError::TeacherCacheInvalidShape)
    }

    /// Total payload size (all windows), excluding the fixed header.
    pub fn payload_bytes(&self) -> Result<u64, FormatError> {
        self.window_bytes()?
            .checked_mul(self.windows)
            .ok_or(FormatError::TeacherCacheInvalidShape)
    }
}

/// Encode a fixed-size little-endian header.
pub fn write_topk_teacher_cache_header(
    header: &TopkTeacherCacheHeader,
) -> Result<Vec<u8>, FormatError> {
    let _ = header.payload_bytes()?; // reject a degenerate shape before emitting bytes
    let mut out = Vec::with_capacity(TOPK_TEACHER_CACHE_HEADER_BYTES);
    out.extend_from_slice(&TOPK_TEACHER_CACHE_MAGIC);
    out.push(TOPK_TEACHER_CACHE_VERSION);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&header.seq_len.to_le_bytes());
    out.extend_from_slice(&header.vocab.to_le_bytes());
    out.extend_from_slice(&header.top_k.to_le_bytes());
    out.extend_from_slice(&header.windows.to_le_bytes());
    out.extend_from_slice(&header.model_hash);
    out.extend_from_slice(&header.corpus_hash);
    debug_assert_eq!(out.len(), TOPK_TEACHER_CACHE_HEADER_BYTES);
    Ok(out)
}

/// Decode and validate a fixed-size header.
pub fn read_topk_teacher_cache_header(bytes: &[u8]) -> Result<TopkTeacherCacheHeader, FormatError> {
    if bytes.len() != TOPK_TEACHER_CACHE_HEADER_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TOPK_TEACHER_CACHE_HEADER_BYTES,
            got: bytes.len(),
        });
    }
    let mut cursor = LeCursor::new(bytes);
    if cursor.take(4)? != TOPK_TEACHER_CACHE_MAGIC {
        return Err(FormatError::TeacherCacheBadMagic);
    }
    let version = cursor.u8()?;
    if version != TOPK_TEACHER_CACHE_VERSION {
        return Err(FormatError::UnsupportedTeacherCacheVersion(version));
    }
    let _reserved = cursor.take(3)?;
    let seq_len = cursor.u32()?;
    let vocab = cursor.u32()?;
    let top_k = cursor.u32()?;
    let windows = cursor.u64()?;
    let mut model_hash = [0; 32];
    model_hash.copy_from_slice(cursor.take(32)?);
    let mut corpus_hash = [0; 32];
    corpus_hash.copy_from_slice(cursor.take(32)?);
    let header = TopkTeacherCacheHeader {
        seq_len,
        vocab,
        top_k,
        windows,
        model_hash,
        corpus_hash,
    };
    let _ = header.payload_bytes()?; // reject a degenerate shape on read
    Ok(header)
}

/// Serialize one window's top-k payload (SoA: `seq_len·top_k` indices then the same count of probs).
/// `idx`/`prob` are row-major `[seq_len, top_k]`.
///
/// # Errors
/// [`FormatError::TeacherCacheInvalidShape`] if the slice lengths disagree with the header shape or an
/// index is `≥ vocab`.
pub fn encode_topk_window(
    header: &TopkTeacherCacheHeader,
    idx: &[u32],
    prob: &[f32],
) -> Result<Vec<u8>, FormatError> {
    let pairs = header.window_pairs()?;
    if idx.len() != pairs || prob.len() != pairs {
        return Err(FormatError::TeacherCacheInvalidShape);
    }
    if idx.iter().any(|&i| i >= header.vocab) {
        return Err(FormatError::TeacherCacheInvalidShape);
    }
    let mut out = Vec::with_capacity(pairs * 8);
    for &i in idx {
        out.extend_from_slice(&i.to_le_bytes());
    }
    for &p in prob {
        out.extend_from_slice(&p.to_le_bytes());
    }
    Ok(out)
}

/// Deserialize one window's top-k payload into `(indices, probabilities)`, each `seq_len·top_k` long.
///
/// # Errors
/// [`FormatError`] if `bytes` is not exactly one window, or an index is `≥ vocab`.
pub fn decode_topk_window(
    header: &TopkTeacherCacheHeader,
    bytes: &[u8],
) -> Result<(Vec<u32>, Vec<f32>), FormatError> {
    let pairs = header.window_pairs()?;
    if bytes.len() as u64 != header.window_bytes()? {
        return Err(FormatError::WrongBlockLen {
            expected: pairs * 8,
            got: bytes.len(),
        });
    }
    let mut cursor = LeCursor::new(bytes);
    let mut idx = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let i = cursor.u32()?;
        if i >= header.vocab {
            return Err(FormatError::TeacherCacheInvalidShape);
        }
        idx.push(i);
    }
    let mut prob = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        prob.push(f32::from_le_bytes(
            cursor.take(4)?.try_into().expect("four bytes"),
        ));
    }
    Ok((idx, prob))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> TopkTeacherCacheHeader {
        TopkTeacherCacheHeader {
            seq_len: 32,
            vocab: 49_152,
            top_k: 64,
            windows: 40,
            model_hash: [0x11; 32],
            corpus_hash: [0x22; 32],
        }
    }

    #[test]
    fn header_roundtrips_and_sizes_windows() {
        let h = header();
        let encoded = write_topk_teacher_cache_header(&h).unwrap();
        assert_eq!(encoded.len(), TOPK_TEACHER_CACHE_HEADER_BYTES);
        assert_eq!(read_topk_teacher_cache_header(&encoded).unwrap(), h);
        assert_eq!(h.window_pairs().unwrap(), 32 * 64);
        assert_eq!(h.window_bytes().unwrap(), 32 * 64 * 8);
        assert_eq!(h.payload_bytes().unwrap(), 40 * 32 * 64 * 8);
        // The whole point: far smaller than the dense payload. Byte shrink is vocab/(2·top_k) — each
        // pair is 8 bytes (u32 index + f32 prob) vs a 4-byte dense prob — = 49152/128 = 384×.
        let dense = 40u64 * 32 * 49_152 * 4;
        assert_eq!(dense / h.payload_bytes().unwrap(), 384);
    }

    #[test]
    fn header_rejects_corruption_and_bad_shapes() {
        let h = header();
        let mut encoded = write_topk_teacher_cache_header(&h).unwrap();
        encoded[0] ^= 0xff;
        assert!(read_topk_teacher_cache_header(&encoded).is_err());
        let encoded = write_topk_teacher_cache_header(&h).unwrap();
        assert!(read_topk_teacher_cache_header(&encoded[..encoded.len() - 1]).is_err());
        // top_k must be in 1..=vocab.
        let mut too_wide = h;
        too_wide.top_k = h.vocab + 1;
        assert!(write_topk_teacher_cache_header(&too_wide).is_err());
        let mut zero_k = h;
        zero_k.top_k = 0;
        assert!(write_topk_teacher_cache_header(&zero_k).is_err());
    }

    #[test]
    fn window_payload_roundtrips() {
        let h = TopkTeacherCacheHeader {
            seq_len: 2,
            vocab: 10,
            top_k: 3,
            windows: 1,
            model_hash: [0; 32],
            corpus_hash: [0; 32],
        };
        let idx: Vec<u32> = vec![0, 4, 9, 1, 2, 7]; // [seq_len=2, top_k=3]
        let prob: Vec<f32> = vec![0.6, 0.3, 0.05, 0.5, 0.4, 0.08];
        let bytes = encode_topk_window(&h, &idx, &prob).unwrap();
        assert_eq!(bytes.len() as u64, h.window_bytes().unwrap());
        let (got_idx, got_prob) = decode_topk_window(&h, &bytes).unwrap();
        assert_eq!(got_idx, idx);
        assert_eq!(got_prob, prob);
    }

    #[test]
    fn window_payload_rejects_bad_input() {
        let h = TopkTeacherCacheHeader {
            seq_len: 2,
            vocab: 10,
            top_k: 3,
            windows: 1,
            model_hash: [0; 32],
            corpus_hash: [0; 32],
        };
        // Wrong lengths.
        assert!(encode_topk_window(&h, &[0, 1, 2], &[0.1, 0.2, 0.3]).is_err());
        // Out-of-range index (≥ vocab).
        assert!(
            encode_topk_window(&h, &[0, 4, 10, 1, 2, 7], &[0.6, 0.3, 0.05, 0.5, 0.4, 0.08])
                .is_err()
        );
        // Truncated payload.
        let bytes =
            encode_topk_window(&h, &[0, 4, 9, 1, 2, 7], &[0.6, 0.3, 0.05, 0.5, 0.4, 0.08]).unwrap();
        assert!(decode_topk_window(&h, &bytes[..bytes.len() - 1]).is_err());
    }
}
