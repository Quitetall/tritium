//! Q2_0: llama.cpp's official 2-bit group-64 format. Port of ggml
//! `quantize_row_q2_0_ref` / `dequantize_row_q2_0` (llama.cpp PR #24448, CPU;
//! PR #25707, CUDA): `block_q2_0 { ggml_half d; uint8_t qs[16] }` — scale
//! FIRST, then 16 bytes of 2-bit codes for 64 weights. This is the opposite
//! byte order from TQ2_0 (qs first, scale last) and a quarter of its group
//! size, so none of the [`crate::QK_K`]-based helpers apply here.
//!
//! Weight `j` lives in `qs[j / 4]` at bit offset `2 * (j % 4)`, stored as
//! `code = value + 1`. ggml's Q2_0 levels are `{-1, 0, +1, +2}` (code 3 decodes
//! to `+2·d`); Tritium is ternary, so pack only ever emits codes `{0, 1, 2}`
//! and unpack rejects code 3 as [`FormatError::DecodedOutOfRange`].

// Portions ported from llama.cpp/ggml — Copyright (c) 2023-2026 The ggml authors.
// Licensed MIT (see llama.cpp/LICENSE); listed in this repository's NOTICE.
use half::f16;
use tritium_core::Trit;

use crate::FormatError;

/// Weights per Q2_0 block (ggml `QK2_0`).
pub const Q2_0_GROUP_SIZE: usize = 64;

/// Bytes in one packed Q2_0 block: `f16 d + qs[16]` = 18.
pub const Q2_0_BLOCK_BYTES: usize = 2 + Q2_0_GROUP_SIZE / 4;

/// Number of [`Q2_0_GROUP_SIZE`]-sized blocks needed to hold `k` trits.
///
/// Not interchangeable with [`crate::num_blocks`], which counts `QK_K = 256`
/// blocks.
#[inline]
#[must_use]
pub fn q2_0_num_blocks(k: usize) -> usize {
    k.div_ceil(Q2_0_GROUP_SIZE)
}

/// Pack 64 trits + a scale into one Q2_0 block (18 bytes).
///
/// The scale occupies bytes `0..2` little-endian (scale-first, unlike TQ2_0);
/// trit `j` is stored as `trit + 1 ∈ {0, 1, 2}` in byte `2 + j/4` at bit
/// position `2 * (j % 4)`.
///
/// # Errors
/// [`FormatError::WrongTritCount`] / [`FormatError::WrongBlockLen`] on size mismatch.
pub fn pack_q2_0_block(trits: &[Trit], scale: f16, out: &mut [u8]) -> Result<(), FormatError> {
    if trits.len() != Q2_0_GROUP_SIZE {
        return Err(FormatError::WrongTritCount {
            expected: Q2_0_GROUP_SIZE,
            got: trits.len(),
        });
    }
    if out.len() != Q2_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: Q2_0_BLOCK_BYTES,
            got: out.len(),
        });
    }
    // Scale-first layout: the crate's write_scale targets the LAST two bytes
    // (TQ2_0/TQ1_0 convention) and must not be used here.
    out[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
    let qs = &mut out[2..];
    for (byte, chunk) in qs.iter_mut().zip(trits.as_chunks::<4>().0.iter()) {
        let mut q: u8 = 0;
        for (slot, trit) in chunk.iter().enumerate() {
            let code = (trit.get() + 1) as u8; // {0,1,2}
            q |= code << (2 * slot);
        }
        *byte = q;
    }
    Ok(())
}

/// Unpack one Q2_0 block into 64 trits + its scale.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] / [`FormatError::WrongTritCount`] on size
/// mismatch; [`FormatError::DecodedOutOfRange`] on code 3 (ggml's `+2` level,
/// never valid ternary — corrupt or non-ternary input).
pub fn unpack_q2_0_block(
    block: &[u8],
    trits_out: &mut [Trit],
    scale_out: &mut f16,
) -> Result<(), FormatError> {
    if block.len() != Q2_0_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: Q2_0_BLOCK_BYTES,
            got: block.len(),
        });
    }
    if trits_out.len() != Q2_0_GROUP_SIZE {
        return Err(FormatError::WrongTritCount {
            expected: Q2_0_GROUP_SIZE,
            got: trits_out.len(),
        });
    }
    let qs = &block[2..];
    for (byte, out_chunk) in qs.iter().zip(trits_out.as_chunks_mut::<4>().0.iter_mut()) {
        for (slot, trit_out) in out_chunk.iter_mut().enumerate() {
            let q = (byte >> (2 * slot)) & 3;
            // Code 3 decodes to +2 → TritError::OutOfRange(2) → DecodedOutOfRange(2).
            *trit_out = Trit::from_i8(q as i8 - 1)?;
        }
    }
    *scale_out = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
    Ok(())
}

/// Validate that `scales` and the packed length match the block count for `k`.
fn check_row(k: usize, scales: &[f16], packed_len: usize) -> Result<(), FormatError> {
    let nb = q2_0_num_blocks(k);
    if scales.len() != nb {
        return Err(FormatError::WrongBlockLen {
            expected: nb,
            got: scales.len(),
        });
    }
    let expected = nb * Q2_0_BLOCK_BYTES;
    if packed_len != expected {
        return Err(FormatError::WrongBlockLen {
            expected,
            got: packed_len,
        });
    }
    Ok(())
}

/// Pack a row of `K` trits into Q2_0 blocks (`nb * 18` bytes).
///
/// `trits.len()` must be `K`, `scales.len()` must be `nb = K.div_ceil(64)`, and
/// `out.len()` must be `nb * `[`Q2_0_BLOCK_BYTES`]. The final block is
/// zero-padded (llama.cpp convention, matching [`crate::pack_tq2_0_row`]).
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `scales` or `out` is the wrong length;
/// propagates any error from [`pack_q2_0_block`].
pub fn pack_q2_0_row(trits: &[Trit], scales: &[f16], out: &mut [u8]) -> Result<(), FormatError> {
    let k = trits.len();
    check_row(k, scales, out.len())?;
    let mut block = [Trit::ZERO; Q2_0_GROUP_SIZE];
    for (i, &scale) in scales.iter().enumerate() {
        let start = i * Q2_0_GROUP_SIZE;
        let end = (start + Q2_0_GROUP_SIZE).min(k);
        let n = end - start;
        block[..n].copy_from_slice(&trits[start..end]);
        block[n..].fill(Trit::ZERO); // zero-pad the tail of the last block
        let bo = i * Q2_0_BLOCK_BYTES;
        pack_q2_0_block(&block, scale, &mut out[bo..bo + Q2_0_BLOCK_BYTES])?;
    }
    Ok(())
}

/// Unpack a Q2_0 row back into exactly `K` trits and `nb` scales.
///
/// `packed.len()` must be `nb * `[`Q2_0_BLOCK_BYTES`], `trits_out.len()` must be
/// `K`, and `scales_out.len()` must be `nb`, where `nb = K.div_ceil(64)`.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] on length mismatch; propagates any error from
/// [`unpack_q2_0_block`].
pub fn unpack_q2_0_row(
    packed: &[u8],
    trits_out: &mut [Trit],
    scales_out: &mut [f16],
) -> Result<(), FormatError> {
    let k = trits_out.len();
    check_row(k, scales_out, packed.len())?;
    let mut block = [Trit::ZERO; Q2_0_GROUP_SIZE];
    for (i, scale_out) in scales_out.iter_mut().enumerate() {
        let bo = i * Q2_0_BLOCK_BYTES;
        unpack_q2_0_block(&packed[bo..bo + Q2_0_BLOCK_BYTES], &mut block, scale_out)?;
        let start = i * Q2_0_GROUP_SIZE;
        let end = (start + Q2_0_GROUP_SIZE).min(k);
        trits_out[start..end].copy_from_slice(&block[..end - start]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_of(v: i8) -> Vec<Trit> {
        vec![Trit::from_i8(v).unwrap(); Q2_0_GROUP_SIZE]
    }

    // Independent golden bytes (hand-computed from the PR #24448 reference):
    // code = trit + 1 in each 2-bit slot ⇒ all-zero → 0b01010101 = 0x55,
    // all-+1 → 0b10101010 = 0xAA, all-(-1) → 0x00. The f16 scale is the FIRST
    // two bytes (little-endian), then 16 qs bytes.
    #[test]
    fn golden_all_zero() {
        let mut out = vec![0u8; Q2_0_BLOCK_BYTES];
        pack_q2_0_block(&block_of(0), f16::ONE, &mut out).unwrap();
        assert_eq!(&out[..2], &f16::ONE.to_bits().to_le_bytes());
        assert!(out[2..].iter().all(|&b| b == 0x55));
    }

    #[test]
    fn golden_all_pos_and_neg() {
        let mut out = vec![0u8; Q2_0_BLOCK_BYTES];
        pack_q2_0_block(&block_of(1), f16::ONE, &mut out).unwrap();
        assert!(out[2..].iter().all(|&b| b == 0xAA));
        pack_q2_0_block(&block_of(-1), f16::ONE, &mut out).unwrap();
        assert!(out[2..].iter().all(|&b| b == 0x00));
    }

    /// Pin the bit order: a lone +1 at element 5 lands in qs byte 5/4 = 1
    /// (block byte 3) at bit offset 2*(5%4) = 2 — slot code 01 becomes 10,
    /// so 0b0101_0101 → 0b0101_1001 = 0x59.
    #[test]
    fn golden_positional_bit_order() {
        let mut trits = block_of(0);
        trits[5] = Trit::POS;
        let mut out = vec![0u8; Q2_0_BLOCK_BYTES];
        pack_q2_0_block(&trits, f16::ONE, &mut out).unwrap();
        assert_eq!(out[2], 0x55, "qs[0] untouched");
        assert_eq!(out[3], 0x59, "qs[1] slot 1 flips 01→10");
        assert!(out[4..].iter().all(|&b| b == 0x55));
    }

    #[test]
    fn unpack_golden_zero_is_zero() {
        let mut block = vec![0x55u8; Q2_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        let mut trits = vec![Trit::POS; Q2_0_GROUP_SIZE];
        let mut scale = f16::ZERO;
        unpack_q2_0_block(&block, &mut trits, &mut scale).unwrap();
        assert!(trits.iter().all(|t| t.get() == 0));
        assert_eq!(scale.to_bits(), f16::ONE.to_bits());
    }

    /// ggml's code 3 (the +2 level) is never valid ternary — unpack must
    /// reject it as a typed error, not clamp or wrap it.
    #[test]
    fn code_three_rejected() {
        let mut block = vec![0x55u8; Q2_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        block[2] = 0b0101_0111; // slot 0 = code 3, rest zero-codes
        let mut trits = vec![Trit::ZERO; Q2_0_GROUP_SIZE];
        let mut scale = f16::ZERO;
        assert_eq!(
            unpack_q2_0_block(&block, &mut trits, &mut scale),
            Err(FormatError::DecodedOutOfRange(2))
        );
    }

    #[test]
    fn wrong_sizes_error() {
        let mut out = vec![0u8; Q2_0_BLOCK_BYTES];
        assert!(matches!(
            pack_q2_0_block(&block_of(0)[..63], f16::ONE, &mut out),
            Err(FormatError::WrongTritCount { .. })
        ));
        assert!(matches!(
            pack_q2_0_block(&block_of(0), f16::ONE, &mut out[..17]),
            Err(FormatError::WrongBlockLen { .. })
        ));
        let mut trits = vec![Trit::ZERO; Q2_0_GROUP_SIZE];
        let mut scale = f16::ZERO;
        assert!(matches!(
            unpack_q2_0_block(&out[..17], &mut trits, &mut scale),
            Err(FormatError::WrongBlockLen { .. })
        ));
        assert!(matches!(
            unpack_q2_0_block(&out, &mut trits[..63], &mut scale),
            Err(FormatError::WrongTritCount { .. })
        ));
    }

    #[test]
    fn num_blocks_matches_div_ceil() {
        assert_eq!(q2_0_num_blocks(0), 0);
        assert_eq!(q2_0_num_blocks(1), 1);
        assert_eq!(q2_0_num_blocks(64), 1);
        assert_eq!(q2_0_num_blocks(65), 2);
        assert_eq!(q2_0_num_blocks(4096), 64);
    }

    /// Deterministic ternary row of length `k` from a simple LCG.
    fn make_row(k: usize, seed: u64) -> Vec<Trit> {
        let mut s = seed;
        (0..k)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (s >> 33) % 3; // {0,1,2}
                Trit::from_i8(v as i8 - 1).unwrap()
            })
            .collect()
    }

    fn scales(nb: usize) -> Vec<f16> {
        (0..nb).map(|i| f16::from_f32(0.5 + i as f32)).collect()
    }

    fn roundtrip(k: usize) {
        let trits = make_row(k, 0x2064 ^ k as u64);
        let nb = q2_0_num_blocks(k);
        let sc = scales(nb);
        let mut packed = vec![0u8; nb * Q2_0_BLOCK_BYTES];
        pack_q2_0_row(&trits, &sc, &mut packed).unwrap();

        let mut out = vec![Trit::ZERO; k];
        let mut out_sc = vec![f16::ZERO; nb];
        unpack_q2_0_row(&packed, &mut out, &mut out_sc).unwrap();

        assert_eq!(out, trits, "q2_0 K={k} trits");
        for (a, b) in out_sc.iter().zip(&sc) {
            assert_eq!(a.to_bits(), b.to_bits(), "q2_0 K={k} scale");
        }
    }

    #[test]
    fn row_roundtrip_boundary_ks() {
        for &k in &[1usize, 63, 64, 65, 100, 4096] {
            roundtrip(k);
        }
    }

    #[test]
    fn partial_block_pads_with_zeros() {
        // K=65 => block 1 holds 1 real trit + 63 zero-pad. Unpack must return
        // exactly 65 trits, and the pad must be all zero-codes on the wire.
        let mut trits = vec![Trit::POS; 65];
        trits[64] = Trit::NEG;
        let sc = scales(2);
        let mut packed = vec![0u8; 2 * Q2_0_BLOCK_BYTES];
        pack_q2_0_row(&trits, &sc, &mut packed).unwrap();
        // Block 1 wire bytes: slot 0 = code 0 (-1), slots 1..3 = code 1 (pad)
        // → 0b0101_0100 = 0x54, remaining qs bytes all 0x55.
        assert_eq!(packed[Q2_0_BLOCK_BYTES + 2], 0x54);
        assert!(packed[Q2_0_BLOCK_BYTES + 3..].iter().all(|&b| b == 0x55));
        let mut out = vec![Trit::ZERO; 65];
        let mut out_sc = vec![f16::ZERO; 2];
        unpack_q2_0_row(&packed, &mut out, &mut out_sc).unwrap();
        assert_eq!(out, trits);
    }

    #[test]
    fn wrong_row_lens_error() {
        let trits = vec![Trit::ZERO; 64];
        let mut packed = vec![0u8; Q2_0_BLOCK_BYTES];
        // nb=1 but two scales supplied.
        assert!(matches!(
            pack_q2_0_row(&trits, &[f16::ONE, f16::ONE], &mut packed),
            Err(FormatError::WrongBlockLen { .. })
        ));
        // Output one byte short.
        assert!(matches!(
            pack_q2_0_row(&trits, &[f16::ONE], &mut packed[..Q2_0_BLOCK_BYTES - 1]),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
