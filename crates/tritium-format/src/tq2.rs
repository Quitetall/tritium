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

/// Value of every qs byte in an all-zero TQ2_0 block (code 1 in all 4 slots).
const ZERO_BLOCK_BYTE: u8 = 0x55;

/// Compute a per-block zero bitmap for one packed TQ2_0 row.
///
/// Each bit corresponds to one 256-trit block (66 bytes: 64 qs + 2 scale).
/// Bit is SET if the block is all-zero (every qs byte == `0x55`), CLEAR otherwise.
/// The scale bytes are ignored — a zero-trit block contributes nothing regardless
/// of its scale value.
///
/// Returns `Vec<u32>` of length `ceil(num_blocks / 32)`.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `packed_row` is shorter than the
/// `ceil(k / QK_K) * TQ2_0_BLOCK_BYTES` bytes a fully-packed row requires
/// (so a malformed/truncated row is a typed error, never a panic).
pub fn compute_zero_bitmap(packed_row: &[u8], k: usize) -> Result<Vec<u32>, FormatError> {
    let nb = k.div_ceil(QK_K);
    let need = nb * TQ2_0_BLOCK_BYTES;
    if packed_row.len() < need {
        return Err(FormatError::WrongBlockLen {
            expected: need,
            got: packed_row.len(),
        });
    }
    let words = nb.div_ceil(32);
    let mut bitmap = vec![0u32; words];
    for block_idx in 0..nb {
        let offset = block_idx * TQ2_0_BLOCK_BYTES;
        let qs = &packed_row[offset..offset + QS_BYTES];
        if qs.iter().all(|&b| b == ZERO_BLOCK_BYTE) {
            bitmap[block_idx / 32] |= 1u32 << (block_idx % 32);
        }
    }
    Ok(bitmap)
}

/// Compute zero bitmaps for all N rows of packed TQ2_0 weights.
///
/// Returns a flat `Vec<u32>` of length `N * ceil(num_blocks(k) / 32)`,
/// one bitmap per row, concatenated in row order.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `packed` is shorter than `n * row_bytes`
/// (truncated input is a typed error, never a panic).
pub fn compute_zero_bitmaps(
    packed: &[u8],
    n: usize,
    k: usize,
    row_bytes: usize,
) -> Result<Vec<u32>, FormatError> {
    let nb = k.div_ceil(QK_K);
    let words_per_row = nb.div_ceil(32);
    // Checked: `n * row_bytes` on attacker-influenced sizes must not wrap
    // (a wrapped product could pass the length check then slice out of
    // bounds below).
    let need = n
        .checked_mul(row_bytes)
        .ok_or(FormatError::WrongBlockLen {
            expected: usize::MAX,
            got: packed.len(),
        })?;
    if packed.len() < need {
        return Err(FormatError::WrongBlockLen {
            expected: need,
            got: packed.len(),
        });
    }
    let mut bitmaps = vec![0u32; n * words_per_row];
    for ni in 0..n {
        let row_start = ni * row_bytes;
        let row_end = row_start + row_bytes;
        let row_bitmap = compute_zero_bitmap(&packed[row_start..row_end], k)?;
        let out_start = ni * words_per_row;
        bitmaps[out_start..out_start + words_per_row].copy_from_slice(&row_bitmap);
    }
    Ok(bitmaps)
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

    // ── Bitmap tests ────────────────────────────────────────────────────────

    #[test]
    fn bitmap_all_zero_block() {
        let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES];
        pack_tq2_0_block(&block_of(0), f16::ONE, &mut packed).unwrap();
        let bm = compute_zero_bitmap(&packed, QK_K).unwrap();
        assert_eq!(bm.len(), 1);
        assert_eq!(bm[0], 1, "bit 0 should be set for all-zero block");
    }

    #[test]
    fn bitmap_all_nonzero_block() {
        let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES];
        pack_tq2_0_block(&block_of(1), f16::ONE, &mut packed).unwrap();
        let bm = compute_zero_bitmap(&packed, QK_K).unwrap();
        assert_eq!(bm.len(), 1);
        assert_eq!(bm[0], 0, "bit should be clear for all-positive block");

        pack_tq2_0_block(&block_of(-1), f16::ONE, &mut packed).unwrap();
        let bm = compute_zero_bitmap(&packed, QK_K).unwrap();
        assert_eq!(bm[0], 0, "bit should be clear for all-negative block");
    }

    #[test]
    fn bitmap_mixed_blocks() {
        // K=512: block 0 = all-zero, block 1 = all-nonzero
        let nb = 2;
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; row_bytes];
        // Block 0: all-zero
        pack_tq2_0_block(&block_of(0), f16::ONE, &mut packed[..TQ2_0_BLOCK_BYTES]).unwrap();
        // Block 1: all-positive
        pack_tq2_0_block(&block_of(1), f16::ONE, &mut packed[TQ2_0_BLOCK_BYTES..]).unwrap();

        let bm = compute_zero_bitmap(&packed, QK_K * 2).unwrap();
        assert_eq!(bm.len(), 1);
        assert_eq!(bm[0], 0b01, "bit 0 set (zero block), bit 1 clear (nonzero)");
    }

    #[test]
    fn bitmap_empty_row() {
        let bm = compute_zero_bitmap(&[], 0).unwrap();
        assert!(bm.is_empty());
    }

    #[test]
    fn bitmap_partial_zero_not_set() {
        // K=256 with 255 zeros and 1 non-zero → bitmap bit is CLEAR
        let mut trits = block_of(0);
        trits[128] = Trit::POS; // one non-zero in the middle
        let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES];
        pack_tq2_0_block(&trits, f16::ONE, &mut packed).unwrap();
        let bm = compute_zero_bitmap(&packed, QK_K).unwrap();
        assert_eq!(bm[0], 0, "bit should be clear when block has any nonzero");
    }

    #[test]
    fn bitmap_multi_row() {
        // N=2, K=256: row 0 all-zero, row 1 all-nonzero
        let row_bytes = TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; 2 * row_bytes];
        pack_tq2_0_block(&block_of(0), f16::ONE, &mut packed[..row_bytes]).unwrap();
        pack_tq2_0_block(&block_of(1), f16::ONE, &mut packed[row_bytes..]).unwrap();

        let bm = compute_zero_bitmaps(&packed, 2, QK_K, row_bytes).unwrap();
        assert_eq!(bm.len(), 2);
        assert_eq!(bm[0], 1, "row 0: bit set");
        assert_eq!(bm[1], 0, "row 1: bit clear");
    }

    #[test]
    fn bitmap_many_blocks_per_row() {
        // K=8192 = 32 blocks, all zero
        let nb = 32;
        let k = nb * QK_K;
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; row_bytes];
        for b in 0..nb {
            let start = b * TQ2_0_BLOCK_BYTES;
            pack_tq2_0_block(
                &block_of(0),
                f16::ONE,
                &mut packed[start..start + TQ2_0_BLOCK_BYTES],
            )
            .unwrap();
        }
        let bm = compute_zero_bitmap(&packed, k).unwrap();
        assert_eq!(bm.len(), 1); // 32 blocks fit in one u32
        assert_eq!(bm[0], 0xFFFF_FFFF, "all 32 bits set");
    }

    #[test]
    fn bitmap_spans_multiple_u32_words() {
        // K=8192 + 256 = 8448 → 33 blocks → needs 2 u32 words
        let nb = 33;
        let k = nb * QK_K;
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; row_bytes];
        // All zero
        for b in 0..nb {
            let start = b * TQ2_0_BLOCK_BYTES;
            pack_tq2_0_block(
                &block_of(0),
                f16::ONE,
                &mut packed[start..start + TQ2_0_BLOCK_BYTES],
            )
            .unwrap();
        }
        let bm = compute_zero_bitmap(&packed, k).unwrap();
        assert_eq!(bm.len(), 2);
        assert_eq!(bm[0], 0xFFFF_FFFF, "word 0: all 32 bits set");
        assert_eq!(bm[1], 0x0000_0001, "word 1: bit 0 set (block 32)");
    }
}
