//! TB1 — the bitmap+signs ternary packing (A4 prototype).
//!
//! Per 256-trit block: a 32-byte **presence plane** (bit set = nonzero) —
//! then, per ROW, one contiguous **sign stream** (bit per nonzero, set = +1),
//! padded to 4-byte alignment. Rows are variable-length, so the container
//! carries a per-row byte-offset table.
//!
//! Rate: `1 + (1 - p)` bits/weight (+ ~0 for offsets) — 1.578 b/w at BitNet's
//! measured p = 0.422, UNDER dense TQ1_0's 1.625, and sparsity-adaptive
//! (→ 1.0 as p → 1, where 2:4-trained students live, ADR 0024). The zero
//! state costs a presence bit only: element-level zero-skipping falls out of
//! the layout instead of being an index structure (which the entropy math
//! rejects at this density).
//!
//! Scale-free (like I2_S consumers here): the caller carries per-channel
//! scales; block scales are not stored (the prototype's GEMM contract).

use tritium_core::Trit;

use crate::{FormatError, QK_K};

/// Presence-plane bytes per 256-trit block.
pub const TB1_PRESENCE_BYTES: usize = QK_K / 8; // 32

/// Packed row size in bytes: presence planes + 4-byte-aligned sign stream.
#[must_use]
pub fn tb1_row_bytes(k: usize, nnz: usize) -> usize {
    k.div_ceil(QK_K) * TB1_PRESENCE_BYTES + nnz.div_ceil(8).div_ceil(4) * 4
}

/// Pack one row. Returns the packed bytes (presence planes, then the aligned
/// sign stream). The final partial block's missing tail reads as absent
/// (presence 0), so any `k` packs.
///
/// # Errors
/// None currently — total for every `trits` input (the signature reserves a
/// typed error for future container integration).
pub fn pack_tb1_row(trits: &[Trit]) -> Result<Vec<u8>, FormatError> {
    let k = trits.len();
    let nb = k.div_ceil(QK_K);
    let mut presence = vec![0u8; nb * TB1_PRESENCE_BYTES];
    let mut signs: Vec<u8> = Vec::new();
    let mut sign_bit = 0usize;
    for (i, t) in trits.iter().enumerate() {
        let v = t.get();
        if v != 0 {
            presence[i / 8] |= 1 << (i % 8);
            if sign_bit % 8 == 0 {
                signs.push(0);
            }
            if v > 0 {
                *signs.last_mut().expect("pushed") |= 1 << (sign_bit % 8);
            }
            sign_bit += 1;
        }
    }
    let mut out = presence;
    out.extend_from_slice(&signs);
    // 4-byte-align the row so the next row's presence plane is word-aligned.
    while out.len() % 4 != 0 {
        out.push(0);
    }
    Ok(out)
}

/// Unpack a TB1 row back to trits (the reference the kernel is gated against).
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `packed` is shorter than the presence
/// planes + the sign stream its own presence bits imply.
pub fn unpack_tb1_row(packed: &[u8], trits_out: &mut [Trit]) -> Result<(), FormatError> {
    let k = trits_out.len();
    let nb = k.div_ceil(QK_K);
    let pres_bytes = nb * TB1_PRESENCE_BYTES;
    if packed.len() < pres_bytes {
        return Err(FormatError::WrongBlockLen {
            expected: pres_bytes,
            got: packed.len(),
        });
    }
    let (presence, signs) = packed.split_at(pres_bytes);
    let mut sign_bit = 0usize;
    for (i, t) in trits_out.iter_mut().enumerate() {
        if presence[i / 8] & (1 << (i % 8)) != 0 {
            let byte = signs.get(sign_bit / 8).ok_or(FormatError::WrongBlockLen {
                expected: pres_bytes + sign_bit / 8 + 1,
                got: packed.len(),
            })?;
            let pos = byte & (1 << (sign_bit % 8)) != 0;
            *t = Trit::from_i8(if pos { 1 } else { -1 }).expect("±1");
            sign_bit += 1;
        } else {
            *t = Trit::ZERO;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(k: usize, seed: u64) -> Vec<Trit> {
        let mut s = seed;
        (0..k)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Trit::from_i8(((s >> 33) % 3) as i8 - 1).unwrap()
            })
            .collect()
    }

    #[test]
    fn roundtrip_and_rate() {
        for &k in &[256usize, 1024, 2560, 100, 300] {
            let trits = row(k, k as u64 ^ 0xBEEF);
            let packed = pack_tb1_row(&trits).unwrap();
            let nnz = trits.iter().filter(|t| t.get() != 0).count();
            assert_eq!(packed.len(), tb1_row_bytes(k, nnz), "k={k}");
            let mut out = vec![Trit::ZERO; k];
            unpack_tb1_row(&packed, &mut out).unwrap();
            assert_eq!(out, trits, "k={k}");
        }
    }

    #[test]
    fn truncated_is_typed_error() {
        let trits = row(512, 7);
        let packed = pack_tb1_row(&trits).unwrap();
        let mut out = vec![Trit::ZERO; 512];
        assert!(unpack_tb1_row(&packed[..40], &mut out).is_err());
        assert!(unpack_tb1_row(&packed[..70], &mut out).is_err());
    }
}
