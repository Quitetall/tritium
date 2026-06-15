//! Row-level wrappers over the per-block pack/unpack primitives.
//!
//! A "row" is a contiguous run of `K` trits — typically one row of a weight
//! matrix — quantized in [`QK_K`]-sized blocks. There are `nb = K.div_ceil(256)`
//! blocks; the final block is **zero-padded** to a full [`QK_K`] on pack
//! (llama.cpp convention) and the padding is discarded on unpack so exactly `K`
//! trits come back. Each block carries its own `f16` scale, so the row functions
//! take / return a `&[f16]` of length `nb`.
//!
//! These wrappers do not invent layout: they call [`pack_tq2_0_block`] /
//! [`pack_tq1_0_block`] (and their unpack counterparts) unchanged, so a row is
//! byte-identical to the concatenation of its blocks.

use half::f16;
use tritium_core::Trit;

use crate::{
    FormatError, QK_K, TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, pack_tq1_0_block, pack_tq2_0_block,
    unpack_tq1_0_block, unpack_tq2_0_block,
};

/// Number of [`QK_K`]-sized blocks needed to hold `k` trits.
#[inline]
#[must_use]
pub fn num_blocks(k: usize) -> usize {
    k.div_ceil(QK_K)
}

/// Validate that `scales` and `out` match the block count implied by `k`.
fn check_row(
    k: usize,
    scales: &[f16],
    out_len: usize,
    block_bytes: usize,
) -> Result<usize, FormatError> {
    let nb = num_blocks(k);
    if scales.len() != nb {
        return Err(FormatError::WrongBlockLen {
            expected: nb,
            got: scales.len(),
        });
    }
    let expected = nb * block_bytes;
    if out_len != expected {
        return Err(FormatError::WrongBlockLen {
            expected,
            got: out_len,
        });
    }
    Ok(nb)
}

/// Pack a row of `K` trits into TQ2_0 blocks (`nb * 66` bytes).
///
/// `trits.len()` must be `K`, `scales.len()` must be `nb = K.div_ceil(256)`, and
/// `out.len()` must be `nb * `[`TQ2_0_BLOCK_BYTES`]. The final block is zero-padded.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `scales` or `out` is the wrong length;
/// propagates any error from [`pack_tq2_0_block`].
pub fn pack_tq2_0_row(trits: &[Trit], scales: &[f16], out: &mut [u8]) -> Result<(), FormatError> {
    let k = trits.len();
    check_row(k, scales, out.len(), TQ2_0_BLOCK_BYTES)?;
    let mut block = [Trit::ZERO; QK_K];
    for (i, &scale) in scales.iter().enumerate() {
        let start = i * QK_K;
        let end = (start + QK_K).min(k);
        let n = end - start;
        block[..n].copy_from_slice(&trits[start..end]);
        block[n..].fill(Trit::ZERO); // zero-pad the tail of the last block
        let bo = i * TQ2_0_BLOCK_BYTES;
        pack_tq2_0_block(&block, scale, &mut out[bo..bo + TQ2_0_BLOCK_BYTES])?;
    }
    Ok(())
}

/// Unpack a TQ2_0 row back into exactly `K` trits and `nb` scales.
///
/// `packed.len()` must be `nb * `[`TQ2_0_BLOCK_BYTES`], `trits_out.len()` must be
/// `K`, and `scales_out.len()` must be `nb`, where `nb = K.div_ceil(256)`.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] on length mismatch; propagates any error from
/// [`unpack_tq2_0_block`].
pub fn unpack_tq2_0_row(
    packed: &[u8],
    trits_out: &mut [Trit],
    scales_out: &mut [f16],
) -> Result<(), FormatError> {
    let k = trits_out.len();
    check_row(k, scales_out, packed.len(), TQ2_0_BLOCK_BYTES)?;
    let mut block = [Trit::ZERO; QK_K];
    for (i, scale_out) in scales_out.iter_mut().enumerate() {
        let bo = i * TQ2_0_BLOCK_BYTES;
        unpack_tq2_0_block(&packed[bo..bo + TQ2_0_BLOCK_BYTES], &mut block, scale_out)?;
        let start = i * QK_K;
        let end = (start + QK_K).min(k);
        trits_out[start..end].copy_from_slice(&block[..end - start]);
    }
    Ok(())
}

/// Pack a row of `K` trits into TQ1_0 blocks (`nb * 54` bytes).
///
/// `trits.len()` must be `K`, `scales.len()` must be `nb = K.div_ceil(256)`, and
/// `out.len()` must be `nb * `[`TQ1_0_BLOCK_BYTES`]. The final block is zero-padded.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `scales` or `out` is the wrong length;
/// propagates any error from [`pack_tq1_0_block`].
pub fn pack_tq1_0_row(trits: &[Trit], scales: &[f16], out: &mut [u8]) -> Result<(), FormatError> {
    let k = trits.len();
    check_row(k, scales, out.len(), TQ1_0_BLOCK_BYTES)?;
    let mut block = [Trit::ZERO; QK_K];
    for (i, &scale) in scales.iter().enumerate() {
        let start = i * QK_K;
        let end = (start + QK_K).min(k);
        let n = end - start;
        block[..n].copy_from_slice(&trits[start..end]);
        block[n..].fill(Trit::ZERO);
        let bo = i * TQ1_0_BLOCK_BYTES;
        pack_tq1_0_block(&block, scale, &mut out[bo..bo + TQ1_0_BLOCK_BYTES])?;
    }
    Ok(())
}

/// Unpack a TQ1_0 row back into exactly `K` trits and `nb` scales.
///
/// `packed.len()` must be `nb * `[`TQ1_0_BLOCK_BYTES`], `trits_out.len()` must be
/// `K`, and `scales_out.len()` must be `nb`, where `nb = K.div_ceil(256)`.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] on length mismatch; propagates any error from
/// [`unpack_tq1_0_block`].
pub fn unpack_tq1_0_row(
    packed: &[u8],
    trits_out: &mut [Trit],
    scales_out: &mut [f16],
) -> Result<(), FormatError> {
    let k = trits_out.len();
    check_row(k, scales_out, packed.len(), TQ1_0_BLOCK_BYTES)?;
    let mut block = [Trit::ZERO; QK_K];
    for (i, scale_out) in scales_out.iter_mut().enumerate() {
        let bo = i * TQ1_0_BLOCK_BYTES;
        unpack_tq1_0_block(&packed[bo..bo + TQ1_0_BLOCK_BYTES], &mut block, scale_out)?;
        let start = i * QK_K;
        let end = (start + QK_K).min(k);
        trits_out[start..end].copy_from_slice(&block[..end - start]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn roundtrip_tq2(k: usize) {
        let trits = make_row(k, 0xABCD ^ k as u64);
        let nb = num_blocks(k);
        let sc = scales(nb);
        let mut packed = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &sc, &mut packed).unwrap();

        let mut out = vec![Trit::ZERO; k];
        let mut out_sc = vec![f16::ZERO; nb];
        unpack_tq2_0_row(&packed, &mut out, &mut out_sc).unwrap();

        assert_eq!(out, trits, "tq2 K={k} trits");
        for (a, b) in out_sc.iter().zip(&sc) {
            assert_eq!(a.to_bits(), b.to_bits(), "tq2 K={k} scale");
        }
    }

    fn roundtrip_tq1(k: usize) {
        let trits = make_row(k, 0x1234 ^ k as u64);
        let nb = num_blocks(k);
        let sc = scales(nb);
        let mut packed = vec![0u8; nb * TQ1_0_BLOCK_BYTES];
        pack_tq1_0_row(&trits, &sc, &mut packed).unwrap();

        let mut out = vec![Trit::ZERO; k];
        let mut out_sc = vec![f16::ZERO; nb];
        unpack_tq1_0_row(&packed, &mut out, &mut out_sc).unwrap();

        assert_eq!(out, trits, "tq1 K={k} trits");
        for (a, b) in out_sc.iter().zip(&sc) {
            assert_eq!(a.to_bits(), b.to_bits(), "tq1 K={k} scale");
        }
    }

    #[test]
    fn row_roundtrip_boundary_ks() {
        for &k in &[1usize, 255, 256, 257, 4096] {
            roundtrip_tq2(k);
            roundtrip_tq1(k);
        }
    }

    #[test]
    fn num_blocks_matches_div_ceil() {
        assert_eq!(num_blocks(0), 0);
        assert_eq!(num_blocks(1), 1);
        assert_eq!(num_blocks(256), 1);
        assert_eq!(num_blocks(257), 2);
        assert_eq!(num_blocks(4096), 16);
    }

    #[test]
    fn partial_block_pads_with_zeros() {
        // K=257 => block 1 holds 1 real trit + 255 zero-pad. Unpack must return
        // exactly 257 trits, and the padded region must not leak into them.
        let mut trits = vec![Trit::POS; 257];
        trits[256] = Trit::NEG;
        let sc = scales(2);
        let mut packed = vec![0u8; 2 * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &sc, &mut packed).unwrap();
        let mut out = vec![Trit::ZERO; 257];
        let mut out_sc = vec![f16::ZERO; 2];
        unpack_tq2_0_row(&packed, &mut out, &mut out_sc).unwrap();
        assert_eq!(out, trits);
        assert_eq!(out.len(), 257);
    }

    #[test]
    fn wrong_scale_len_errors() {
        let trits = vec![Trit::ZERO; 256];
        let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES];
        // nb=1 but two scales supplied.
        assert!(matches!(
            pack_tq2_0_row(&trits, &[f16::ONE, f16::ONE], &mut packed),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn wrong_output_len_errors() {
        let trits = vec![Trit::ZERO; 256];
        let mut packed = vec![0u8; TQ2_0_BLOCK_BYTES - 1]; // too short
        assert!(matches!(
            pack_tq2_0_row(&trits, &[f16::ONE], &mut packed),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }
}
