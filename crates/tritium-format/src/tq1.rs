//! TQ1_0: base-3, 5 trits per byte. Port of ggml `quantize_row_tq1_0_ref` /
//! `dequantize_row_tq1_0` (block of 256: `qs[48]` then `qh[4]` then `f16` scale).
//!
//! Layout: `qs[0..32]` hold 160 elements (5 trits/byte, stride 32), `qs[32..48]`
//! hold 80 (stride 16), `qh[4]` hold the final 16 (4 trits/byte, shifted up one
//! trit). Encode is `(acc*256 + 242)/243`; decode is `((byte*pow3[n]) as u8 * 3) >> 8`,
//! relying on `u8` wraparound — replicated exactly.

// Portions ported from llama.cpp/ggml — Copyright (c) 2023-2026 The ggml authors.
// Licensed MIT (see llama.cpp/LICENSE); listed in this repository's NOTICE.
use half::f16;
use tritium_core::Trit;

use crate::{FormatError, QK_K, TQ1_0_BLOCK_BYTES, read_scale, write_scale};

const QS_BYTES: usize = (QK_K - 4 * QK_K / 64) / 5; // 48
const QH_BYTES: usize = QK_K / 64; // 4
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

#[inline]
fn encode(acc: u16) -> u8 {
    // ggml's `(acc*256 + 242)/243` written as the equivalent ceiling division.
    (acc * 256).div_ceil(243) as u8
}

#[inline]
fn decode(byte: u8, n: usize) -> i8 {
    let q = byte.wrapping_mul(POW3[n]); // u8 wraparound is part of the scheme
    (((q as u16) * 3) >> 8) as i8 - 1 // {0,1,2} -> {-1,0,1}
}

/// Pack 256 trits + a scale into one TQ1_0 block (54 bytes).
///
/// # Errors
/// [`FormatError::WrongTritCount`] / [`FormatError::WrongBlockLen`] on size mismatch.
pub fn pack_tq1_0_block(trits: &[Trit], scale: f16, out: &mut [u8]) -> Result<(), FormatError> {
    if trits.len() != QK_K {
        return Err(FormatError::WrongTritCount {
            expected: QK_K,
            got: trits.len(),
        });
    }
    if out.len() != TQ1_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TQ1_0_BLOCK_BYTES,
            got: out.len(),
        });
    }
    let xi = |i: usize| (trits[i].get() + 1) as u16; // {0,1,2}

    // qs[0..32]: 5 trits/byte, stride 32, elements 0..160.
    for (m, byte) in out[..32].iter_mut().enumerate() {
        let mut acc = 0u16;
        for n in 0..5 {
            acc = acc * 3 + xi(m + n * 32);
        }
        *byte = encode(acc);
    }
    // qs[32..48]: 5 trits/byte, stride 16, elements 160..240.
    for m in 0..16 {
        let mut acc = 0u16;
        for n in 0..5 {
            acc = acc * 3 + xi(160 + m + n * 16);
        }
        out[32 + m] = encode(acc);
    }
    // qh[0..4]: 4 trits/byte, stride 4, elements 240..256, then shift up one trit.
    for j in 0..4 {
        let mut acc = 0u16;
        for m in 0..4 {
            acc = acc * 3 + xi(240 + j + m * 4);
        }
        acc *= 3;
        out[QS_BYTES + j] = encode(acc);
    }
    write_scale(scale, out);
    Ok(())
}

/// Unpack one TQ1_0 block into 256 trits + its scale.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] / [`FormatError::WrongTritCount`] on size mismatch.
pub fn unpack_tq1_0_block(
    block: &[u8],
    trits_out: &mut [Trit],
    scale_out: &mut f16,
) -> Result<(), FormatError> {
    if block.len() != TQ1_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TQ1_0_BLOCK_BYTES,
            got: block.len(),
        });
    }
    if trits_out.len() != QK_K {
        return Err(FormatError::WrongTritCount {
            expected: QK_K,
            got: trits_out.len(),
        });
    }
    let qs = &block[..QS_BYTES];
    let qh = &block[QS_BYTES..QS_BYTES + QH_BYTES];

    for n in 0..5 {
        for m in 0..32 {
            trits_out[n * 32 + m] = Trit::from_i8(decode(qs[m], n))?;
        }
    }
    for n in 0..5 {
        for m in 0..16 {
            trits_out[160 + n * 16 + m] = Trit::from_i8(decode(qs[32 + m], n))?;
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            trits_out[240 + n * 4 + j] = Trit::from_i8(decode(qh[j], n))?;
        }
    }
    *scale_out = read_scale(block);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_of(v: i8) -> Vec<Trit> {
        vec![Trit::from_i8(v).unwrap(); QK_K]
    }

    // Independent goldens (hand-computed, cross-checked): all-zero trits ⇒ qs 0x80,
    // qh 0x7F; all-+1 ⇒ qs 0xFF, qh 0xFD; all-(-1) ⇒ 0x00.
    #[test]
    fn golden_all_zero() {
        let mut out = vec![0u8; TQ1_0_BLOCK_BYTES];
        pack_tq1_0_block(&block_of(0), f16::ONE, &mut out).unwrap();
        assert!(out[..48].iter().all(|&b| b == 0x80), "qs");
        assert!(out[48..52].iter().all(|&b| b == 0x7F), "qh");
        assert_eq!(&out[52..], &f16::ONE.to_bits().to_le_bytes());
    }

    #[test]
    fn golden_all_pos_and_neg() {
        let mut out = vec![0u8; TQ1_0_BLOCK_BYTES];
        pack_tq1_0_block(&block_of(1), f16::ONE, &mut out).unwrap();
        assert!(out[..48].iter().all(|&b| b == 0xFF), "qs +1");
        assert!(out[48..52].iter().all(|&b| b == 0xFD), "qh +1");
        pack_tq1_0_block(&block_of(-1), f16::ONE, &mut out).unwrap();
        assert!(out[..52].iter().all(|&b| b == 0x00), "all -1");
    }

    #[test]
    fn unpack_golden_zero_is_zero() {
        let mut block = vec![0u8; TQ1_0_BLOCK_BYTES];
        block[..48].fill(0x80);
        block[48..52].fill(0x7F);
        block[52..].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        let mut trits = vec![Trit::POS; QK_K];
        let mut scale = f16::ZERO;
        unpack_tq1_0_block(&block, &mut trits, &mut scale).unwrap();
        assert!(trits.iter().all(|t| t.get() == 0));
        assert_eq!(scale.to_bits(), f16::ONE.to_bits());
    }
}
