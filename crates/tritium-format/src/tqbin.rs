//! `.tqbin` — a **tokenized corpus shard** (plan 0012): a flat little-endian stream of `u32`
//! token IDs (concatenated, GPT-style document packing; the loader cuts fixed `seq_len` windows).
//! The companion [`crate::tqidx`] manifest names the shards and records each one's token count.
//!
//! Layout (little-endian):
//! ```text
//! magic b"TQBN" (4) | version u8 | _reserved u8 u16 (3) | n_tokens u64
//! tokens: n_tokens × u32
//! ```
//!
//! [`read_tqbin`] is total: magic + version are enforced and `n_tokens` is validated against the
//! actual buffer length *before* allocating, so a truncated or crafted shard errors rather than
//! panicking or reserving gigabytes (the `checkpoint.rs::f32_vec` lesson).

use crate::{FormatError, le_cursor::LeCursor};

/// Shard magic: `b"TQBN"` (Tritium Quantized BiNary corpus).
pub const TQBIN_MAGIC: [u8; 4] = *b"TQBN";

/// Current `.tqbin` format version.
pub const TQBIN_VERSION: u8 = 1;

/// Header bytes before the token payload: `magic 4 + version 1 + reserved 3 + n_tokens 8`.
pub const TQBIN_HEADER_BYTES: usize = 16;

/// Serialize a token stream to a `.tqbin` shard.
#[must_use]
pub fn write_tqbin(tokens: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(TQBIN_HEADER_BYTES + tokens.len() * 4);
    out.extend_from_slice(&TQBIN_MAGIC);
    out.push(TQBIN_VERSION);
    out.extend_from_slice(&[0u8; 3]); // reserved
    out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

/// Parse a `.tqbin` shard into its token stream, enforcing magic + version and bounds-checking the
/// declared length against the buffer before allocating.
///
/// # Errors
/// [`FormatError::TqBadMagic`] on a bad magic, [`FormatError::UnsupportedTqVersion`] on a version
/// this build cannot read, or [`FormatError::WrongBlockLen`] when the declared `n_tokens` does not
/// fit the remaining bytes (truncation / crafted length).
pub fn read_tqbin(bytes: &[u8]) -> Result<Vec<u32>, FormatError> {
    let mut c = LeCursor::new(bytes);
    if c.take(4)? != TQBIN_MAGIC {
        return Err(FormatError::TqBadMagic);
    }
    let version = c.u8()?;
    if version != TQBIN_VERSION {
        return Err(FormatError::UnsupportedTqVersion(version));
    }
    let _reserved = c.take(3)?;
    let n_tokens = c.u64()?;

    // Validate the declared length against what the buffer can actually hold *before* allocating.
    let remaining = c.remaining() as u64;
    let need = n_tokens.checked_mul(4).ok_or(FormatError::WrongBlockLen {
        expected: usize::MAX,
        got: c.remaining(),
    })?;
    if need > remaining {
        return Err(FormatError::WrongBlockLen {
            expected: usize::try_from(need).unwrap_or(usize::MAX),
            got: c.remaining(),
        });
    }
    // `need <= remaining <= bytes.len()`, so `n_tokens` fits `usize` on any platform the buffer fits.
    let n = usize::try_from(n_tokens).unwrap_or(usize::MAX);
    let mut tokens = Vec::with_capacity(n);
    for _ in 0..n {
        tokens.push(c.u32()?);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips() {
        let toks: Vec<u32> = vec![0, 1, 7, 42, u32::MAX, 1024, 0];
        let bytes = write_tqbin(&toks);
        assert_eq!(bytes.len(), TQBIN_HEADER_BYTES + toks.len() * 4);
        assert_eq!(read_tqbin(&bytes).unwrap(), toks);
    }

    #[test]
    fn empty_shard_roundtrips() {
        let bytes = write_tqbin(&[]);
        assert_eq!(bytes.len(), TQBIN_HEADER_BYTES);
        assert_eq!(read_tqbin(&bytes).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = write_tqbin(&[1, 2, 3]);
        bytes[0] = b'X';
        assert_eq!(read_tqbin(&bytes), Err(FormatError::TqBadMagic));
    }

    #[test]
    fn bad_version_rejected() {
        let mut bytes = write_tqbin(&[1, 2, 3]);
        bytes[4] = 99;
        assert_eq!(
            read_tqbin(&bytes),
            Err(FormatError::UnsupportedTqVersion(99))
        );
    }

    #[test]
    fn truncated_body_rejected() {
        let mut bytes = write_tqbin(&[1, 2, 3, 4]);
        bytes.truncate(bytes.len() - 5); // lose part of the last token(s)
        assert!(matches!(
            read_tqbin(&bytes),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn crafted_huge_n_tokens_errors_without_alloc() {
        // n_tokens = u64::MAX but a 0-byte body: must error from the bounds check, not OOM.
        let mut b = Vec::new();
        b.extend_from_slice(&TQBIN_MAGIC);
        b.push(TQBIN_VERSION);
        b.extend_from_slice(&[0u8; 3]);
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            read_tqbin(&b),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
