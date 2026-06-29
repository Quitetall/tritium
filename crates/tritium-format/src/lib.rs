//! # tritium-format
//!
//! On-disk / in-VRAM ternary weight packing, and the single host-side source of
//! truth for it. Pack/unpack for the two canonical [`tritium_core::TernaryFormat`]
//! schemes:
//!
//! - **TQ2_0** — 2 bits/trit, 4/byte. `qs[64] + f16 scale` per 256-trit block (66 B).
//! - **TQ1_0** — base-3, 5 trits/byte. `qs[48] + qh[4] + f16 scale` per block (54 B).
//!
//! Both are faithful ports of ggml's reference implementation, so packed bytes are
//! byte-compatible with llama.cpp. Backends consume already-packed bytes; they do
//! not pack themselves.
//!
//! The scale stored in a block is opaque to pack/unpack — the caller decides it
//! (ggml uses `amax`, BitNet uses AbsMean). Roundtrip is trit-exact and scale-bit-exact.
//!
//! Beyond the per-block primitives, this crate provides:
//!
//! - **Row wrappers** ([`pack_tq2_0_row`] / [`unpack_tq2_0_row`] and the TQ1_0
//!   pair) that quantize a run of `K` trits into `K.div_ceil(256)` blocks, each
//!   with its own `f16` scale, zero-padding the final partial block.
//! - A **GGUF v2/v3 container reader** ([`read_gguf`]) that parses the header,
//!   metadata, and tensor table of an in-memory `.gguf` buffer. It is total: no
//!   malformed input can panic or read out of bounds — every error is a typed
//!   [`GgufError`].
//! - A **GGUF writer** ([`write_gguf`]), the reader's inverse: it serializes a
//!   metadata table and tensor payloads into a buffer `read_gguf` parses back
//!   identically (`read_gguf(write_gguf(..)) == input`).
#![forbid(unsafe_code)]
// v0.90 hardening: every public item must carry a doc comment.
#![deny(missing_docs)]

use core::fmt;

use half::f16;
use tritium_core::TritError;

mod gguf;
mod gguf_write;
mod i2s;
mod i2s_int8;
mod le_cursor;
mod rows;
mod safetensors;
mod salt;
mod salt_bundle;
mod salt_gguf;
mod sparse;
mod tq1;
mod tq2;
mod tqbin;
mod tqidx;

pub use gguf::{
    DEFAULT_ALIGNMENT, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufError, GgufFile, GgufValue,
    TensorInfo, read_gguf,
};
pub use gguf_write::{TensorOut, write_gguf};
pub use i2s::{
    GGML_TYPE_I2_S, I2S_BLOCK_BYTES, I2S_BLOCK_ELEMS, I2S_SCALE_BYTES, unpack_i2s_block,
    unpack_i2s_tensor,
};
pub use i2s_int8::{
    I2sInt8Weights, IMMA_K, IMMA_N, IMMA_WTILE_BYTES, convert_i2s_to_int8, convert_i2s_to_tq2_0,
};
pub use rows::{num_blocks, pack_tq1_0_row, pack_tq2_0_row, unpack_tq1_0_row, unpack_tq2_0_row};
pub use safetensors::{SafeTensors, SafeTensorsError, read_safetensors};
pub use salt::{
    SALT_HEADER_BYTES, SALT_MAGIC, SALT_VERSION, SaltRow, dequant_salt_row, pack_salt_row,
    read_legacy_as_salt, unpack_salt_row,
};
pub use salt_bundle::{
    SALT_BUNDLE_MAGIC, SALT_BUNDLE_VERSION, SaltTensor, read_salt_bundle, write_salt_bundle,
};
pub use salt_gguf::{
    GGML_TYPE_TRITIUM_SALT, SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE, read_salt_gguf,
    write_salt_gguf,
};
pub use sparse::{
    PlaneRepr, SPARSE_HEADER_BYTES, SPARSE_MAGIC, SPARSE_VERSION, SparsePlane, choose_plane_repr,
    dequant_sparse_plane, expand_plane_repr, pack_sparse_plane, sparse_dot, sparse_from_tq2_0,
    sparse_to_tq2_0, unpack_sparse_plane,
};
pub use tq1::{pack_tq1_0_block, unpack_tq1_0_block};
pub use tq2::{compute_zero_bitmap, compute_zero_bitmaps, pack_tq2_0_block, unpack_tq2_0_block};
pub use tqbin::{TQBIN_HEADER_BYTES, TQBIN_MAGIC, TQBIN_VERSION, read_tqbin, write_tqbin};
pub use tqidx::{ShardEntry, TQIDX_MAGIC, TQIDX_VERSION, TqIndex, read_tqidx, write_tqidx};

/// Weights per quantization block (ggml `QK_K`).
pub const QK_K: usize = 256;

/// Bytes in one packed TQ2_0 block: `qs[64] + f16` = 66.
pub const TQ2_0_BLOCK_BYTES: usize = QK_K / 4 + 2;

/// Bytes in one packed TQ1_0 block: `qs[48] + qh[4] + f16` = 54.
pub const TQ1_0_BLOCK_BYTES: usize = (QK_K - 4 * QK_K / 64) / 5 + QK_K / 64 + 2;

/// Errors from packing/unpacking.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// A pack/unpack call received other than [`QK_K`] trits.
    WrongTritCount {
        /// Required count ([`QK_K`]).
        expected: usize,
        /// Count supplied.
        got: usize,
    },
    /// A block buffer was the wrong size for its format.
    WrongBlockLen {
        /// Required block size in bytes.
        expected: usize,
        /// Size supplied.
        got: usize,
    },
    /// A decoded value fell outside `{-1, 0, +1}` (corrupt input).
    DecodedOutOfRange(i32),
    /// An I2_S 2-bit code was the reserved `0b11` (corrupt input). I2_S decodes
    /// `trit = code - 1`, so only `0b00`=-1, `0b01`=0, `0b10`=+1 are valid; `0b11`
    /// never occurs in valid weights (see the `i2s` module for the WF-4 verification).
    InvalidI2sCode(u8),
    /// A SALT sidecar buffer did not start with the [`SALT_MAGIC`] bytes.
    SaltBadMagic,
    /// A SALT sidecar declared a format version this build cannot read.
    UnsupportedSaltVersion(u8),
    /// A [`SaltRow`] had more planes than the sidecar's `u8` plane-count field.
    SaltTooManyPlanes(usize),
    /// A [`SaltRow`]'s `k` did not fit the sidecar's `u32` length field.
    SaltRowTooLong(usize),
    /// A GGUF container operation (read/write) failed; carries the [`GgufError`].
    Gguf(GgufError),
    /// A safetensors container operation failed; carries the [`SafeTensorsError`].
    #[non_exhaustive]
    SafeTensors(SafeTensorsError),
    /// A GGUF buffer was not a tritium SALT-in-GGUF container (missing or wrong
    /// `tritium.salt.format` marker, or a SALT tensor with malformed dims).
    SaltGgufBadFormat,
    /// A `.tqbin`/`.tqidx` buffer did not start with its expected magic bytes.
    TqBadMagic,
    /// A `.tqbin`/`.tqidx` buffer declared a format version this build cannot read.
    UnsupportedTqVersion(u8),
    /// A `.tqidx` manifest declared `seq_len == 0` (it is the divisor for the sample count).
    TqZeroSeqLen,
    /// A `.tqidx` shard name exceeded the `u16` name-length field.
    TqNameTooLong(usize),
    /// A `.tqidx` shard name was not valid UTF-8.
    TqBadName,
}

impl From<GgufError> for FormatError {
    fn from(e: GgufError) -> Self {
        FormatError::Gguf(e)
    }
}

impl From<SafeTensorsError> for FormatError {
    fn from(e: SafeTensorsError) -> Self {
        FormatError::SafeTensors(e)
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::WrongTritCount { expected, got } => {
                write!(f, "wrong trit count: expected {expected}, got {got}")
            }
            FormatError::WrongBlockLen { expected, got } => {
                write!(
                    f,
                    "wrong block length: expected {expected} bytes, got {got}"
                )
            }
            FormatError::DecodedOutOfRange(v) => {
                write!(f, "decoded value {v} outside ternary range")
            }
            FormatError::InvalidI2sCode(c) => {
                write!(f, "invalid I2_S 2-bit code 0b{c:02b} (reserved)")
            }
            FormatError::SaltBadMagic => write!(f, "SALT sidecar: bad magic"),
            FormatError::UnsupportedSaltVersion(v) => {
                write!(f, "SALT sidecar: unsupported version {v}")
            }
            FormatError::SaltTooManyPlanes(t) => {
                write!(
                    f,
                    "SALT sidecar: {t} planes exceed the u8 plane-count field"
                )
            }
            FormatError::SaltRowTooLong(k) => {
                write!(
                    f,
                    "SALT sidecar: row length {k} exceeds the u32 length field"
                )
            }
            FormatError::Gguf(e) => write!(f, "GGUF container: {e}"),
            FormatError::SafeTensors(e) => write!(f, "safetensors container: {e}"),
            FormatError::SaltGgufBadFormat => {
                write!(f, "GGUF buffer is not a tritium SALT-in-GGUF container")
            }
            FormatError::TqBadMagic => write!(f, "tq corpus: bad magic"),
            FormatError::UnsupportedTqVersion(v) => {
                write!(f, "tq corpus: unsupported version {v}")
            }
            FormatError::TqZeroSeqLen => write!(f, "tq manifest: seq_len must be non-zero"),
            FormatError::TqNameTooLong(n) => {
                write!(
                    f,
                    "tq manifest: shard name length {n} exceeds the u16 field"
                )
            }
            FormatError::TqBadName => write!(f, "tq manifest: shard name is not valid UTF-8"),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FormatError::Gguf(e) => Some(e),
            FormatError::SafeTensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TritError> for FormatError {
    fn from(e: TritError) -> Self {
        match e {
            TritError::OutOfRange(v) => FormatError::DecodedOutOfRange(v),
            // Decode math guarantees {-1,0,1}; any other TritError is unexpected.
            _ => FormatError::DecodedOutOfRange(i32::MIN),
        }
    }
}

/// Read a little-endian `f16` scale from the last two bytes of a packed block.
#[inline]
fn read_scale(block: &[u8]) -> f16 {
    let n = block.len();
    f16::from_bits(u16::from_le_bytes([block[n - 2], block[n - 1]]))
}

/// Write a little-endian `f16` scale into the last two bytes of a packed block.
#[inline]
fn write_scale(scale: f16, out: &mut [u8]) {
    let n = out.len();
    out[n - 2..].copy_from_slice(&scale.to_bits().to_le_bytes());
}

#[cfg(test)]
mod roundtrip {
    use super::*;
    use proptest::prelude::*;
    use tritium_core::Trit;

    fn block_strategy() -> impl Strategy<Value = Vec<i8>> {
        prop::collection::vec(-1i8..=1, QK_K)
    }

    proptest! {
        #[test]
        fn tq2_0_roundtrip(raw in block_strategy(), scale_bits in any::<u16>()) {
            let trits: Vec<Trit> = raw.iter().map(|&v| Trit::from_i8(v).unwrap()).collect();
            let scale = f16::from_bits(scale_bits);
            let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES];
            pack_tq2_0_block(&trits, scale, &mut packed).unwrap();

            let mut out = vec![Trit::ZERO; QK_K];
            let mut got_scale = f16::ZERO;
            unpack_tq2_0_block(&packed, &mut out, &mut got_scale).unwrap();

            prop_assert_eq!(out, trits);
            prop_assert_eq!(got_scale.to_bits(), scale.to_bits());
        }

        #[test]
        fn tq1_0_roundtrip(raw in block_strategy(), scale_bits in any::<u16>()) {
            let trits: Vec<Trit> = raw.iter().map(|&v| Trit::from_i8(v).unwrap()).collect();
            let scale = f16::from_bits(scale_bits);
            let mut packed = vec![0u8; TQ1_0_BLOCK_BYTES];
            pack_tq1_0_block(&trits, scale, &mut packed).unwrap();

            let mut out = vec![Trit::ZERO; QK_K];
            let mut got_scale = f16::ZERO;
            unpack_tq1_0_block(&packed, &mut out, &mut got_scale).unwrap();

            prop_assert_eq!(out, trits);
            prop_assert_eq!(got_scale.to_bits(), scale.to_bits());
        }
    }
}
