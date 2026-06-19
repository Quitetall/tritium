//! Sparse residual plane (ADR 0001 §5) — the storage form of a high-`p` residual
//! plane that is mostly zeros.
//!
//! A SALT residual plane is ternary: every weight is `-1`, `0`, or `+1` times its
//! per-256-block scale. Late residual planes prune to mostly zeros, so storing the
//! full dense TQ2_0 row (`num_blocks(k)·66` bytes, ~2 bits/weight) wastes space.
//! A [`SparsePlane`] keeps only the nonzeros — each as a `(column, sign)` pair —
//! plus the per-block scales, and round-trips **exactly** to the dense TQ2_0 bytes.
//!
//! Per ADR 0001 §"Hardware constraints", sparse pays only below ~10% nonzero
//! density; above it the plane stays dense (whole-tile skip). [`choose_plane_repr`]
//! is that density switch; [`expand_plane_repr`] reconstructs identical dense bytes
//! from either side, so the matmul output is bit-identical regardless of the choice.
//!
//! The GPU sparse-matmul kernel (the *compute* win, per-arch) is a later step; this
//! module lands the storage form, the density switch, and the host equivalence the
//! ADR 0006 gate requires.

use half::f16;

use crate::{FormatError, QK_K, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row, unpack_tq2_0_row};
use tritium_core::Trit;

/// Sparse-plane container magic: `b"TSSP"` (Tritium Sparse Sidecar Plane).
pub const SPARSE_MAGIC: [u8; 4] = *b"TSSP";

/// Current sparse-plane format version.
pub const SPARSE_VERSION: u8 = 1;

/// Header: magic(4) + version(1) + _pad(1) + k as u32(4) + nnz as u32(4) = 14.
pub const SPARSE_HEADER_BYTES: usize = 4 + 1 + 1 + 4 + 4;

/// The sign bit packed into the top bit of each nonzero's `u32` column index
/// (set ⇒ `-1`, clear ⇒ `+1`). Bounds `k` to `< 2^31`.
const SIGN_BIT: u32 = 1 << 31;

/// One ternary plane stored as its nonzeros only, plus the per-block scales.
///
/// `idx` is ascending and parallel to `sign` (`±1`). Reconstructs the dense plane
/// exactly: weight at column `c` is `scales[c / QK_K] · sign` for a listed `c`, else 0.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SparsePlane {
    /// Trits in the (dense) plane — the row length `K`.
    pub k: usize,
    /// Per-256-block scales, `num_blocks(k)` of them (TQ2_0 stores one per block).
    pub scales: Vec<f16>,
    /// Ascending column indices of the nonzero trits.
    pub idx: Vec<u32>,
    /// Sign of each nonzero (`-1` or `+1`), parallel to [`Self::idx`].
    pub sign: Vec<i8>,
}

impl SparsePlane {
    /// Number of stored nonzeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.idx.len()
    }

    /// Nonzero fraction `nnz / k` (the density the switch compares to a threshold).
    #[inline]
    pub fn density(&self) -> f32 {
        if self.k == 0 {
            0.0
        } else {
            self.nnz() as f32 / self.k as f32
        }
    }
}

/// Build a [`SparsePlane`] from one dense TQ2_0 plane (`num_blocks(k)·66` bytes).
///
/// # Errors
/// [`FormatError::SaltRowTooLong`] if `k ≥ 2^31` (no room for the packed sign bit);
/// any [`unpack_tq2_0_row`] error on malformed input.
pub fn sparse_from_tq2_0(packed: &[u8], k: usize) -> Result<SparsePlane, FormatError> {
    if k >= SIGN_BIT as usize {
        return Err(FormatError::SaltRowTooLong(k));
    }
    let nb = num_blocks(k);
    let mut trits = vec![Trit::ZERO; k];
    let mut scales = vec![f16::ZERO; nb];
    unpack_tq2_0_row(packed, &mut trits, &mut scales)?;

    let mut idx = Vec::new();
    let mut sign = Vec::new();
    for (c, t) in trits.iter().enumerate() {
        let v = t.get();
        if v != 0 {
            idx.push(c as u32);
            sign.push(v);
        }
    }
    Ok(SparsePlane { k, scales, idx, sign })
}

/// Reconstruct the dense TQ2_0 plane bytes from a [`SparsePlane`] — the exact
/// inverse of [`sparse_from_tq2_0`].
///
/// # Errors
/// [`FormatError::DecodedOutOfRange`] if a `sign` is not `±1`, or
/// [`FormatError::WrongBlockLen`] if `scales` is not `num_blocks(k)` long; any
/// [`pack_tq2_0_row`] error.
pub fn sparse_to_tq2_0(plane: &SparsePlane) -> Result<Vec<u8>, FormatError> {
    let nb = num_blocks(plane.k);
    if plane.scales.len() != nb {
        return Err(FormatError::WrongBlockLen { expected: nb, got: plane.scales.len() });
    }
    if plane.idx.len() != plane.sign.len() {
        return Err(FormatError::WrongBlockLen { expected: plane.idx.len(), got: plane.sign.len() });
    }
    let mut trits = vec![Trit::ZERO; plane.k];
    for (&c, &s) in plane.idx.iter().zip(&plane.sign) {
        let c = c as usize;
        if c >= plane.k {
            return Err(FormatError::WrongBlockLen { expected: plane.k, got: c });
        }
        trits[c] = Trit::from_i8(s)?;
    }
    let mut out = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(&trits, &plane.scales, &mut out)?;
    Ok(out)
}

/// Dequantize a sparse plane to `k` fp32 weights: `scales[c/QK_K]·sign` at each
/// nonzero column, 0 elsewhere. Equal element-for-element to dequantizing the
/// dense plane it came from.
#[must_use]
pub fn dequant_sparse_plane(plane: &SparsePlane) -> Vec<f32> {
    let mut acc = vec![0.0f32; plane.k];
    for (&c, &s) in plane.idx.iter().zip(&plane.sign) {
        let c = c as usize;
        if c < plane.k {
            acc[c] = plane.scales[c / QK_K].to_f32() * f32::from(s);
        }
    }
    acc
}

/// `act · plane` contracting only the nonzeros — `Σ act[c]·scale[c/QK_K]·sign`.
///
/// Bit-identical to the dense dot `Σ_c act[c]·w[c]` (where `w` is the dense
/// dequant): the dense loop's zero terms add exactly `+0.0`, so skipping them in
/// ascending index order leaves the running sum unchanged.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `act` is shorter than `k`.
pub fn sparse_dot(act: &[f32], plane: &SparsePlane) -> Result<f32, FormatError> {
    if act.len() < plane.k {
        return Err(FormatError::WrongBlockLen { expected: plane.k, got: act.len() });
    }
    let mut acc = 0.0f32;
    for (&c, &s) in plane.idx.iter().zip(&plane.sign) {
        let c = c as usize;
        acc += act[c] * plane.scales[c / QK_K].to_f32() * f32::from(s);
    }
    Ok(acc)
}

/// One residual plane in its chosen storage form (the density-switch outcome).
#[derive(Clone, Debug, PartialEq)]
pub enum PlaneRepr {
    /// Kept dense: the raw TQ2_0 bytes (`num_blocks(k)·66`).
    Dense(Vec<u8>),
    /// Stored sparse (nonzeros only).
    Sparse(SparsePlane),
}

/// The density switch (ADR 0001 §5): store the plane sparse iff its nonzero
/// density is **strictly below** `max_density`, else keep it dense.
///
/// `max_density` is the per-arch break-even (~0.10); the caller supplies it.
///
/// # Errors
/// Any [`sparse_from_tq2_0`] error (only on malformed `packed` / oversized `k`).
pub fn choose_plane_repr(
    packed: &[u8],
    k: usize,
    max_density: f32,
) -> Result<PlaneRepr, FormatError> {
    let sparse = sparse_from_tq2_0(packed, k)?;
    if sparse.density() < max_density {
        Ok(PlaneRepr::Sparse(sparse))
    } else {
        Ok(PlaneRepr::Dense(packed.to_vec()))
    }
}

/// Reconstruct the dense TQ2_0 bytes from either representation — identical bytes
/// on both sides of the switch, so downstream matmul output is bit-identical.
///
/// # Errors
/// Any [`sparse_to_tq2_0`] error when expanding the sparse side.
pub fn expand_plane_repr(repr: &PlaneRepr) -> Result<Vec<u8>, FormatError> {
    match repr {
        PlaneRepr::Dense(bytes) => Ok(bytes.clone()),
        PlaneRepr::Sparse(plane) => sparse_to_tq2_0(plane),
    }
}

/// Serialize a [`SparsePlane`] to its sidecar bytes (little-endian).
///
/// Layout: `magic(4) | version(1) | _pad(1) | k u32 | nnz u32 | scales[nb] f16 |
/// entries[nnz] u32` where each entry packs the column index with the sign in the
/// top bit ([`SIGN_BIT`]).
///
/// # Errors
/// [`FormatError::SaltRowTooLong`] if `k ≥ 2^31`; [`FormatError::WrongBlockLen`] if
/// `scales`/`sign` lengths disagree with `k`/`nnz`; [`FormatError::DecodedOutOfRange`]
/// for a non-`±1` sign or an out-of-range column.
pub fn pack_sparse_plane(plane: &SparsePlane) -> Result<Vec<u8>, FormatError> {
    if plane.k >= SIGN_BIT as usize {
        return Err(FormatError::SaltRowTooLong(plane.k));
    }
    let nb = num_blocks(plane.k);
    if plane.scales.len() != nb {
        return Err(FormatError::WrongBlockLen { expected: nb, got: plane.scales.len() });
    }
    if plane.idx.len() != plane.sign.len() {
        return Err(FormatError::WrongBlockLen { expected: plane.idx.len(), got: plane.sign.len() });
    }
    let mut out = Vec::with_capacity(SPARSE_HEADER_BYTES + nb * 2 + plane.idx.len() * 4);
    out.extend_from_slice(&SPARSE_MAGIC);
    out.push(SPARSE_VERSION);
    out.push(0); // pad
    out.extend_from_slice(&(plane.k as u32).to_le_bytes());
    out.extend_from_slice(&(plane.idx.len() as u32).to_le_bytes());
    for s in &plane.scales {
        out.extend_from_slice(&s.to_bits().to_le_bytes());
    }
    for (&c, &s) in plane.idx.iter().zip(&plane.sign) {
        if c as usize >= plane.k {
            return Err(FormatError::DecodedOutOfRange(c as i32));
        }
        let enc = match s {
            1 => c,
            -1 => c | SIGN_BIT,
            other => return Err(FormatError::DecodedOutOfRange(other as i32)),
        };
        out.extend_from_slice(&enc.to_le_bytes());
    }
    Ok(out)
}

/// Parse a [`SparsePlane`] from sidecar bytes, bounds-checking every field — a
/// corrupt or truncated buffer errors, never panics or reads out of bounds.
///
/// # Errors
/// [`FormatError::SaltBadMagic`] on bad magic; [`FormatError::UnsupportedSaltVersion`]
/// on an unknown version; [`FormatError::WrongBlockLen`] on truncation/length
/// disagreement; [`FormatError::DecodedOutOfRange`] if an entry's column is `≥ k`.
pub fn unpack_sparse_plane(bytes: &[u8]) -> Result<SparsePlane, FormatError> {
    if bytes.len() < SPARSE_HEADER_BYTES {
        return Err(FormatError::WrongBlockLen { expected: SPARSE_HEADER_BYTES, got: bytes.len() });
    }
    if bytes[0..4] != SPARSE_MAGIC {
        return Err(FormatError::SaltBadMagic);
    }
    let version = bytes[4];
    if version != SPARSE_VERSION {
        return Err(FormatError::UnsupportedSaltVersion(version));
    }
    let k = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
    let nnz = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let nb = num_blocks(k);

    // Required length is exact: header + scales + entries. Compute it overflow-safe
    // (k/nnz are attacker-controlled u32; on 32-bit usize the product could wrap).
    let scales_bytes = nb.checked_mul(2).ok_or(FormatError::WrongBlockLen { expected: usize::MAX, got: bytes.len() })?;
    let entries_bytes = nnz.checked_mul(4).ok_or(FormatError::WrongBlockLen { expected: usize::MAX, got: bytes.len() })?;
    let need = SPARSE_HEADER_BYTES
        .checked_add(scales_bytes)
        .and_then(|s| s.checked_add(entries_bytes))
        .ok_or(FormatError::WrongBlockLen { expected: usize::MAX, got: bytes.len() })?;
    if bytes.len() != need {
        return Err(FormatError::WrongBlockLen { expected: need, got: bytes.len() });
    }

    let mut scales = Vec::with_capacity(nb);
    let mut off = SPARSE_HEADER_BYTES;
    for _ in 0..nb {
        let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        scales.push(f16::from_bits(bits));
        off += 2;
    }
    // nnz is bounded by the exact-length check above (need == bytes.len()), so the
    // reserve cannot exceed the buffer.
    let mut idx = Vec::with_capacity(nnz);
    let mut sign = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        let enc = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        off += 4;
        let c = enc & !SIGN_BIT;
        if c as usize >= k {
            return Err(FormatError::DecodedOutOfRange(c as i32));
        }
        idx.push(c);
        sign.push(if enc & SIGN_BIT != 0 { -1 } else { 1 });
    }
    Ok(SparsePlane { k, scales, idx, sign })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense TQ2_0 plane of `k` trits where every `stride`-th weight is nonzero
    /// (sign alternating), the rest pruned — a controllable-density plane.
    fn dense_plane(k: usize, stride: usize, seed: u64) -> Vec<u8> {
        let nb = num_blocks(k);
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        };
        let trits: Vec<Trit> = (0..k)
            .map(|i| {
                if i % stride == 0 {
                    Trit::from_i8(if (next() >> 40) & 1 == 0 { 1 } else { -1 }).unwrap()
                } else {
                    Trit::ZERO
                }
            })
            .collect();
        let scales: Vec<f16> = (0..nb).map(|b| f16::from_f32(0.1 + b as f32 * 0.05)).collect();
        let mut out = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &scales, &mut out).unwrap();
        out
    }

    #[test]
    fn round_trips_to_identical_dense_bytes() {
        for (k, stride) in [(512usize, 7usize), (300, 5), (1024, 13)] {
            let dense = dense_plane(k, stride, 0xABCD ^ k as u64);
            let sparse = sparse_from_tq2_0(&dense, k).unwrap();
            let back = sparse_to_tq2_0(&sparse).unwrap();
            assert_eq!(back, dense, "k={k} stride={stride}: sparse must rebuild exact dense bytes");
        }
    }

    #[test]
    fn dequant_matches_dense() {
        let k = 512;
        let dense = dense_plane(k, 9, 0x11);
        let sparse = sparse_from_tq2_0(&dense, k).unwrap();

        // Dense reference dequant.
        let nb = num_blocks(k);
        let mut trits = vec![Trit::ZERO; k];
        let mut scales = vec![f16::ZERO; nb];
        unpack_tq2_0_row(&dense, &mut trits, &mut scales).unwrap();
        let dense_w: Vec<f32> = (0..k).map(|i| scales[i / QK_K].to_f32() * trits[i].to_f32()).collect();

        assert_eq!(dequant_sparse_plane(&sparse), dense_w);
    }

    #[test]
    fn sparse_dot_is_bit_identical_to_dense_dot() {
        let k = 768;
        let dense = dense_plane(k, 6, 0x77);
        let sparse = sparse_from_tq2_0(&dense, k).unwrap();
        let dense_w = dequant_sparse_plane(&sparse); // same weights, dense layout

        let mut s: u64 = 0xDEAD;
        let act: Vec<f32> = (0..k)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
            })
            .collect();

        let dense_dot: f32 = (0..k).map(|i| act[i] * dense_w[i]).sum();
        assert_eq!(sparse_dot(&act, &sparse).unwrap(), dense_dot, "ascending-index skip of zeros is exact");
    }

    #[test]
    fn density_switch_picks_correctly_and_expands_identically() {
        let k = 1024;
        // Low density (~1/40 = 2.5%) -> Sparse; high density (~1/2 = 50%) -> Dense.
        let low = dense_plane(k, 40, 1);
        let high = dense_plane(k, 2, 2);
        let thresh = 0.10;

        let lr = choose_plane_repr(&low, k, thresh).unwrap();
        let hr = choose_plane_repr(&high, k, thresh).unwrap();
        assert!(matches!(lr, PlaneRepr::Sparse(_)), "2.5% density must go sparse");
        assert!(matches!(hr, PlaneRepr::Dense(_)), "50% density must stay dense");

        // Either side expands to the original dense bytes -> identical matmul.
        assert_eq!(expand_plane_repr(&lr).unwrap(), low);
        assert_eq!(expand_plane_repr(&hr).unwrap(), high);
    }

    #[test]
    fn sparse_storage_is_smaller_when_pruned() {
        let k = 4096;
        let dense = dense_plane(k, 50, 9); // ~2% nonzero
        let dense_bytes = dense.len();
        let sparse = sparse_from_tq2_0(&dense, k).unwrap();
        let packed = pack_sparse_plane(&sparse).unwrap();
        assert!(
            packed.len() < dense_bytes,
            "pruned plane should pack smaller: sparse {} vs dense {dense_bytes}",
            packed.len()
        );
    }

    #[test]
    fn pack_unpack_round_trips() {
        let k = 800;
        let dense = dense_plane(k, 11, 0x5151);
        let sparse = sparse_from_tq2_0(&dense, k).unwrap();
        let packed = pack_sparse_plane(&sparse).unwrap();
        let got = unpack_sparse_plane(&packed).unwrap();
        assert_eq!(got, sparse);
        // And it still expands to the original dense bytes.
        assert_eq!(sparse_to_tq2_0(&got).unwrap(), dense);
    }

    #[test]
    fn corrupt_and_truncated_inputs_error_not_panic() {
        let k = 512;
        let sparse = sparse_from_tq2_0(&dense_plane(k, 8, 3), k).unwrap();
        let packed = pack_sparse_plane(&sparse).unwrap();

        // Bad magic.
        let mut bad = packed.clone();
        bad[0] = b'X';
        assert_eq!(unpack_sparse_plane(&bad).unwrap_err(), FormatError::SaltBadMagic);

        // Bad version.
        let mut badv = packed.clone();
        badv[4] = 99;
        assert!(matches!(unpack_sparse_plane(&badv), Err(FormatError::UnsupportedSaltVersion(99))));

        // Every truncation errors cleanly.
        for len in 0..packed.len() {
            let _ = unpack_sparse_plane(&packed[..len]);
        }

        // An out-of-range column is rejected.
        let mut oob = pack_sparse_plane(&SparsePlane {
            k,
            scales: vec![f16::ONE; num_blocks(k)],
            idx: vec![0],
            sign: vec![1],
        })
        .unwrap();
        // Rewrite the single entry's index to k (out of range).
        let entry_off = SPARSE_HEADER_BYTES + num_blocks(k) * 2;
        oob[entry_off..entry_off + 4].copy_from_slice(&(k as u32).to_le_bytes());
        assert!(matches!(unpack_sparse_plane(&oob), Err(FormatError::DecodedOutOfRange(_))));
    }
}
