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
#![forbid(unsafe_code)]

use core::fmt;

use half::f16;
use tritium_core::TritError;

mod gguf;
mod rows;
mod tq1;
mod tq2;

pub use gguf::{
    DEFAULT_ALIGNMENT, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufError, GgufFile, GgufValue,
    TensorInfo, read_gguf,
};
pub use rows::{num_blocks, pack_tq1_0_row, pack_tq2_0_row, unpack_tq1_0_row, unpack_tq2_0_row};
pub use tq1::{pack_tq1_0_block, unpack_tq1_0_block};
pub use tq2::{pack_tq2_0_block, unpack_tq2_0_block};

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
        }
    }
}

impl std::error::Error for FormatError {}

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
