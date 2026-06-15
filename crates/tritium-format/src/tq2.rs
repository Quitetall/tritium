//! TQ2_0: 2 bits per trit, 4 trits per byte. Port of ggml `quantize_row_tq2_0_ref`
//! / `dequantize_row_tq2_0` (block of 256, `qs[64]` then `f16` scale).

use half::f16;
use tritium_core::Trit;

use crate::{FormatError, QK_K, TQ2_0_BLOCK_BYTES, read_scale, write_scale};

const QS_BYTES: usize = QK_K / 4; // 64

/// Pack 256 trits + a scale into one TQ2_0 block (66 bytes).
///
/// Byte `c*32 + m` (chunk `c∈{0,1}`, `m∈0..32`) holds the 4 trits at element
/// indices `c*128 + n*32 + m` (`n∈0..4`) at bit positions `2n`, each stored as
/// `trit + 1 ∈ {0,1,2}`. Scale is appended little-endian.
///
/// # Errors
/// [`FormatError::WrongTritCount`] / [`FormatError::WrongBlockLen`] on size mismatch.
pub fn pack_tq2_0_block(trits: &[Trit], scale: f16, out: &mut [u8]) -> Result<(), FormatError> {
    if trits.len() != QK_K {
        return Err(FormatError::WrongTritCount {
            expected: QK_K,
            got: trits.len(),
        });
    }
    if out.len() != TQ2_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TQ2_0_BLOCK_BYTES,
            got: out.len(),
        });
    }
    for c in 0..2 {
        for m in 0..32 {
            let mut q: u8 = 0;
            for n in 0..4 {
                let xi = (trits[c * 128 + n * 32 + m].get() + 1) as u8; // {0,1,2}
                q |= (xi & 3) << (2 * n);
            }
            out[c * 32 + m] = q;
        }
    }
    write_scale(scale, out);
    Ok(())
}

/// Unpack one TQ2_0 block into 256 trits + its scale.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] / [`FormatError::WrongTritCount`] on size mismatch.
pub fn unpack_tq2_0_block(
    block: &[u8],
    trits_out: &mut [Trit],
    scale_out: &mut f16,
) -> Result<(), FormatError> {
    if block.len() != TQ2_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: TQ2_0_BLOCK_BYTES,
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
    for c in 0..2 {
        for l in 0..4 {
            for m in 0..32 {
                let q = (qs[c * 32 + m] >> (2 * l)) & 3;
                let t = q as i8 - 1; // {-1,0,1}
                trits_out[c * 128 + l * 32 + m] = Trit::from_i8(t)?;
            }
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

    // Independent golden bytes (hand-computed, verified): code = trit+1 in each
    // 2-bit slot ⇒ all-zero → 0b01010101 = 0x55, all-+1 → 0xAA, all-(-1) → 0x00.
    #[test]
    fn golden_all_zero() {
        let mut out = vec![0u8; TQ2_0_BLOCK_BYTES];
        pack_tq2_0_block(&block_of(0), f16::ONE, &mut out).unwrap();
        assert!(out[..64].iter().all(|&b| b == 0x55));
        assert_eq!(&out[64..], &f16::ONE.to_bits().to_le_bytes());
    }

    #[test]
    fn golden_all_pos_and_neg() {
        let mut out = vec![0u8; TQ2_0_BLOCK_BYTES];
        pack_tq2_0_block(&block_of(1), f16::ONE, &mut out).unwrap();
        assert!(out[..64].iter().all(|&b| b == 0xAA));
        pack_tq2_0_block(&block_of(-1), f16::ONE, &mut out).unwrap();
        assert!(out[..64].iter().all(|&b| b == 0x00));
    }

    #[test]
    fn unpack_golden_zero_is_zero() {
        let mut block = vec![0x55u8; TQ2_0_BLOCK_BYTES];
        block[64..].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        let mut trits = vec![Trit::POS; QK_K];
        let mut scale = f16::ZERO;
        unpack_tq2_0_block(&block, &mut trits, &mut scale).unwrap();
        assert!(trits.iter().all(|t| t.get() == 0));
        assert_eq!(scale.to_bits(), f16::ONE.to_bits());
    }

    #[test]
    fn wrong_sizes_error() {
        let mut out = vec![0u8; TQ2_0_BLOCK_BYTES];
        assert!(matches!(
            pack_tq2_0_block(&block_of(0)[..255], f16::ONE, &mut out),
            Err(FormatError::WrongTritCount { .. })
        ));
        assert!(matches!(
            pack_tq2_0_block(&block_of(0), f16::ONE, &mut out[..65]),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
