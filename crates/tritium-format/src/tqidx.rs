//! `.tqidx` — the **corpus manifest** (plan 0012): the sequence length the loader cuts windows at,
//! plus the ordered list of [`crate::tqbin`] shards and each one's token count. The global sample
//! count a data sampler shuffles over is `N = Σ_shard floor(n_tokens / seq_len)` — see
//! [`TqIndex::n_samples`].
//!
//! Layout (little-endian):
//! ```text
//! magic b"TQIX" (4) | version u8 | _reserved u8 | seq_len u32 | shard_count u32
//! per shard: name_len u16 | name utf8 | n_tokens u64
//! ```
//!
//! [`read_tqidx`] is total: magic + version are enforced, `seq_len` must be non-zero (it is a
//! divisor), and every length is bounds-checked against the buffer before allocating, so a
//! truncated or crafted manifest errors rather than panicking or reserving unboundedly.

use crate::{FormatError, le_cursor::LeCursor};

/// Manifest magic: `b"TQIX"` (Tritium Quantized IndeX).
pub const TQIDX_MAGIC: [u8; 4] = *b"TQIX";

/// Current `.tqidx` format version.
pub const TQIDX_VERSION: u8 = 1;

/// One shard entry in a manifest: a `.tqbin` file name and its token count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardEntry {
    /// Shard file name (e.g. `"corpus.00000.tqbin"`), relative to the manifest.
    pub name: String,
    /// Token count of the shard (must match the shard's own `n_tokens`).
    pub n_tokens: u64,
}

/// A parsed `.tqidx` manifest: the window length and the ordered shard list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TqIndex {
    /// Tokens per training window; the divisor in [`Self::n_samples`]. Always non-zero when parsed.
    pub seq_len: u32,
    /// Shards in corpus order.
    pub shards: Vec<ShardEntry>,
}

impl TqIndex {
    /// The global sample count: `Σ_shard floor(n_tokens / seq_len)` (whole `seq_len`-token windows,
    /// each shard packed independently so a partial trailing window per shard is dropped).
    ///
    /// Saturates rather than overflowing/panicking; returns 0 for a degenerate `seq_len == 0`
    /// (which [`read_tqidx`] never produces, but the public fields allow constructing).
    #[must_use]
    pub fn n_samples(&self) -> usize {
        if self.seq_len == 0 {
            return 0;
        }
        let sl = u64::from(self.seq_len);
        let total = self
            .shards
            .iter()
            .map(|s| s.n_tokens / sl)
            .fold(0u64, u64::saturating_add);
        usize::try_from(total).unwrap_or(usize::MAX)
    }
}

/// Serialize a manifest. `shards` is `(name, n_tokens)` in corpus order.
///
/// # Errors
/// [`FormatError::TqZeroSeqLen`] if `seq_len == 0`, or [`FormatError::TqNameTooLong`] if a shard
/// name exceeds the `u16` length field.
pub fn write_tqidx(seq_len: u32, shards: &[(&str, u64)]) -> Result<Vec<u8>, FormatError> {
    if seq_len == 0 {
        return Err(FormatError::TqZeroSeqLen);
    }
    for (name, _) in shards {
        if name.len() > u16::MAX as usize {
            return Err(FormatError::TqNameTooLong(name.len()));
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&TQIDX_MAGIC);
    out.push(TQIDX_VERSION);
    out.push(0); // reserved
    out.extend_from_slice(&seq_len.to_le_bytes());
    out.extend_from_slice(&(shards.len() as u32).to_le_bytes());
    for (name, n_tokens) in shards {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&n_tokens.to_le_bytes());
    }
    Ok(out)
}

/// Parse a `.tqidx` manifest, enforcing magic + version + non-zero `seq_len` and bounds-checking
/// every field against the buffer.
///
/// # Errors
/// [`FormatError::TqBadMagic`], [`FormatError::UnsupportedTqVersion`], [`FormatError::TqZeroSeqLen`]
/// (a zero divisor is corrupt), or [`FormatError::WrongBlockLen`] on truncation / a name that is not
/// valid UTF-8.
pub fn read_tqidx(bytes: &[u8]) -> Result<TqIndex, FormatError> {
    let mut c = LeCursor::new(bytes);
    if c.take(4)? != TQIDX_MAGIC {
        return Err(FormatError::TqBadMagic);
    }
    let version = c.u8()?;
    if version != TQIDX_VERSION {
        return Err(FormatError::UnsupportedTqVersion(version));
    }
    let _reserved = c.u8()?;
    let seq_len = c.u32()?;
    if seq_len == 0 {
        return Err(FormatError::TqZeroSeqLen);
    }
    let shard_count = c.u32()? as usize;

    // Each shard entry is ≥10 bytes (name_len 2 + n_tokens 8, name possibly empty). Cap the
    // reservation by what the buffer could hold so a crafted `shard_count` errors from the
    // per-entry `take` below rather than reserving gigabytes first.
    let mut shards = Vec::with_capacity(shard_count.min(c.remaining() / 10));
    for _ in 0..shard_count {
        let name_len = c.u16()? as usize;
        let name = core::str::from_utf8(c.take(name_len)?)
            .map_err(|_| FormatError::TqBadName)?
            .to_owned();
        let n_tokens = c.u64()?;
        shards.push(ShardEntry { name, n_tokens });
    }
    Ok(TqIndex { seq_len, shards })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Vec<u8> {
        write_tqidx(
            8,
            &[
                ("corpus.00000.tqbin", 100),
                ("corpus.00001.tqbin", 64),
                ("tail.tqbin", 7), // < seq_len ⇒ contributes 0 samples
            ],
        )
        .unwrap()
    }

    #[test]
    fn roundtrips() {
        let bytes = sample_manifest();
        let idx = read_tqidx(&bytes).unwrap();
        assert_eq!(idx.seq_len, 8);
        assert_eq!(idx.shards.len(), 3);
        assert_eq!(idx.shards[0].name, "corpus.00000.tqbin");
        assert_eq!(idx.shards[0].n_tokens, 100);
        assert_eq!(idx.shards[2].n_tokens, 7);
    }

    #[test]
    fn n_samples_floor_sums_per_shard() {
        // floor(100/8)=12, floor(64/8)=8, floor(7/8)=0 ⇒ 20.
        let idx = read_tqidx(&sample_manifest()).unwrap();
        assert_eq!(idx.n_samples(), 20);
    }

    #[test]
    fn empty_manifest_roundtrips() {
        let bytes = write_tqidx(16, &[]).unwrap();
        let idx = read_tqidx(&bytes).unwrap();
        assert_eq!(idx.seq_len, 16);
        assert!(idx.shards.is_empty());
        assert_eq!(idx.n_samples(), 0);
    }

    #[test]
    fn zero_seq_len_rejected_on_write_and_read() {
        assert_eq!(write_tqidx(0, &[]), Err(FormatError::TqZeroSeqLen));
        // Hand-craft a manifest with seq_len = 0 and confirm the reader rejects the zero divisor.
        let mut b = Vec::new();
        b.extend_from_slice(&TQIDX_MAGIC);
        b.push(TQIDX_VERSION);
        b.push(0);
        b.extend_from_slice(&0u32.to_le_bytes()); // seq_len = 0
        b.extend_from_slice(&0u32.to_le_bytes()); // shard_count = 0
        assert_eq!(read_tqidx(&b), Err(FormatError::TqZeroSeqLen));
    }

    #[test]
    fn n_samples_zero_seq_len_is_zero_not_panic() {
        let idx = TqIndex {
            seq_len: 0,
            shards: vec![ShardEntry {
                name: "x".into(),
                n_tokens: 1000,
            }],
        };
        assert_eq!(idx.n_samples(), 0);
    }

    #[test]
    fn bad_magic_and_truncation_rejected() {
        let mut bytes = sample_manifest();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(read_tqidx(&bad), Err(FormatError::TqBadMagic));
        bytes.truncate(bytes.len() - 4); // lose part of the last n_tokens
        assert!(matches!(
            read_tqidx(&bytes),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn non_utf8_name_rejected() {
        let mut b = Vec::new();
        b.extend_from_slice(&TQIDX_MAGIC);
        b.push(TQIDX_VERSION);
        b.push(0);
        b.extend_from_slice(&8u32.to_le_bytes()); // seq_len
        b.extend_from_slice(&1u32.to_le_bytes()); // shard_count = 1
        b.extend_from_slice(&1u16.to_le_bytes()); // name_len = 1
        b.push(0xFF); // invalid UTF-8 lead byte
        b.extend_from_slice(&100u64.to_le_bytes()); // n_tokens
        assert_eq!(read_tqidx(&b), Err(FormatError::TqBadName));
    }

    #[test]
    fn crafted_huge_shard_count_errors_without_alloc() {
        let mut b = Vec::new();
        b.extend_from_slice(&TQIDX_MAGIC);
        b.push(TQIDX_VERSION);
        b.push(0);
        b.extend_from_slice(&8u32.to_le_bytes()); // seq_len
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // shard_count = 4 billion
        assert!(matches!(
            read_tqidx(&b),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
