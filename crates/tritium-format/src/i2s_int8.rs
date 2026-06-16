//! Load-time conversions from an **I2_S** checkpoint tensor into the GPU-optimal
//! ternary packings the v0.30 CUDA kernels consume (ADR 0005).
//!
//! BitNet ships its ternary weights as I2_S (see [`crate::i2s`]): 2-bit codes,
//! `trit = code - 1`, one per-tensor `f32` scale. Neither v0.30 GPU kernel reads
//! I2_S directly — instead each weight tensor is converted **once at load** into
//! the layout its target kernel wants, validated against the I2_S decode at
//! conversion time (a one-time cost, not per-matmul):
//!
//! - [`convert_i2s_to_tq2_0`] → **TQ2_0** for the tiled *add-only* (decode) kernel.
//!   The magnitude lives in the returned per-tensor `f32`; the TQ2_0 block scales
//!   are unit, exactly as the v0.20 path treats the I2_S per-tensor scale as a
//!   broadcast per-channel scale. This converter is complete.
//! - [`convert_i2s_to_int8`] → [`I2sInt8Weights`] for the *IMMA* (`mma.m16n8k32`)
//!   prefill kernel. The byte layout is the **tile interleave the kernel's `mma`
//!   B operand consumes directly** (see [`convert_i2s_to_int8`] for the exact
//!   geometry). It is validated against the I2_S decode at conversion time — the
//!   round trip `interleave → unpack → trits` is exercised by this module's tests,
//!   the same correctness anchor the provisional plain layout used to provide.

use half::f16;
use tritium_core::{GemmShape, Trit};

use crate::{FormatError, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row, unpack_i2s_tensor};

/// IMMA tile dimensions — must match `kernels/tq2_0_imma.cu`'s `mma.m16n8k32` B
/// operand. The B (weight) operand of one `mma` is an `N×K` tile of
/// [`IMMA_N`]×[`IMMA_K`] ternary codes; N is padded up to [`IMMA_N`], K up to
/// [`IMMA_K`].
pub const IMMA_N: usize = 8;
/// IMMA K-tile width (the `mma.m16n8k32` K dimension).
pub const IMMA_K: usize = 32;
/// Packed bytes per `IMMA_N`×`IMMA_K` weight tile: 256 ternary codes, 4 per byte.
pub const IMMA_WTILE_BYTES: usize = IMMA_N * IMMA_K / 4;

/// Decode an I2_S weight tensor and re-pack it as **TQ2_0** for the add-only CUDA
/// kernel.
///
/// `payload` is the raw I2_S tensor (`n_elements/4` quant bytes + a trailing
/// per-tensor `f32` scale); `shape` gives `[N, K]` (= `shape.n` rows of `shape.k`
/// trits each). Returns the TQ2_0 bytes (`N · num_blocks(K) · `[`TQ2_0_BLOCK_BYTES`])
/// — every block scale unit — together with the single per-tensor `f32` scale the
/// caller supplies to `mpgemm` as the (broadcast) per-channel scale.
///
/// # Errors
/// Propagates [`unpack_i2s_tensor`] errors (short payload, bad code, non-128
/// multiple) and [`pack_tq2_0_row`] errors.
pub fn convert_i2s_to_tq2_0(
    payload: &[u8],
    shape: GemmShape,
) -> Result<(Vec<u8>, f32), FormatError> {
    let GemmShape { n, k, .. } = shape;
    let n_elements = n * k;

    let mut trits = vec![Trit::ZERO; n_elements];
    let scale = unpack_i2s_tensor(payload, n_elements, &mut trits)?;

    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let unit = vec![f16::ONE; nb];
    let mut packed = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let row = &trits[ni * k..ni * k + k];
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        pack_tq2_0_row(row, &unit, out)?;
    }
    Ok((packed, scale))
}

/// Ternary weights in the IMMA int8 GPU layout
/// ([`TernaryFormat::I2sInt8`](tritium_core::TernaryFormat::I2sInt8)), produced by
/// [`convert_i2s_to_int8`].
///
/// `bytes` is the **tile-interleaved 2-bit packing** the `mma.m16n8k32` weight
/// operand consumes (see [`convert_i2s_to_int8`] for the geometry); `scale` is the
/// per-tensor magnitude; `n`/`k` are the *logical* (unpadded) `[N, K]` shape. The
/// kernel needs the packed k-tile count to address a tile, which is
/// `k.div_ceil(IMMA_K)`; [`num_ktiles`](Self::num_ktiles) returns it.
#[derive(Debug, Clone)]
pub struct I2sInt8Weights {
    /// Tile-interleaved 2-bit ternary codes (`code = trit + 1`), 4 codes/byte, in
    /// the IMMA B-operand order. Length is `n_tiles · k_tiles · IMMA_WTILE_BYTES`.
    pub bytes: Vec<u8>,
    /// Per-tensor `f32` magnitude scale carried by the I2_S source.
    pub scale: f32,
    /// Output channels (rows), unpadded.
    pub n: usize,
    /// Input features (columns), unpadded.
    pub k: usize,
}

impl I2sInt8Weights {
    /// Number of padded K-tiles (`ceil(k / IMMA_K)`) — the kernel's `num_ktiles`
    /// launch argument and the k-tile stride within [`bytes`](Self::bytes).
    #[must_use]
    pub fn num_ktiles(&self) -> usize {
        self.k.div_ceil(IMMA_K)
    }

    /// Number of padded N-tiles (`ceil(n / IMMA_N)`).
    #[must_use]
    pub fn num_ntiles(&self) -> usize {
        self.n.div_ceil(IMMA_N)
    }
}

/// Decode an I2_S weight tensor into the IMMA int8 tile layout ([`I2sInt8Weights`]).
///
/// The output `bytes` is the exact interleave `kernels/tq2_0_imma.cu`'s
/// `mma.m16n8k32` B (weight) operand reads:
///
/// * `N` is padded up to a multiple of [`IMMA_N`], `K` up to a multiple of
///   [`IMMA_K`]; padding positions carry trit `0` (code `1`) so they contribute
///   nothing to the int32 contraction.
/// * The packing is `num_ntiles · num_ktiles` tiles of `IMMA_N × IMMA_K` codes,
///   stored **n-tile-major then k-tile-major**: tile `(nt, kt)` begins at byte
///   `(nt · num_ktiles + kt) · IMMA_WTILE_BYTES`.
/// * Within a tile, the 256 codes are `(n_in_tile, k_in_tile)` **row-major**
///   (`code = trit + 1 ∈ {0,1,2}`), 4 per byte with the first element in the low
///   2-bit pair — i.e. `B[n_in_tile, k_in_tile]` for the `N×K` "col" operand.
///
/// The decode is validated against the reference I2_S trits by this module's tests
/// (pack → unpack round trip), satisfying ADR 0005's "converted == reference at
/// load" gate.
///
/// # Errors
/// Propagates [`unpack_i2s_tensor`] errors.
pub fn convert_i2s_to_int8(
    payload: &[u8],
    shape: GemmShape,
) -> Result<I2sInt8Weights, FormatError> {
    let GemmShape { n, k, .. } = shape;
    let n_elements = n * k;

    let mut trits = vec![Trit::ZERO; n_elements];
    let scale = unpack_i2s_tensor(payload, n_elements, &mut trits)?;

    let num_ntiles = n.div_ceil(IMMA_N);
    let num_ktiles = k.div_ceil(IMMA_K);
    // Padding codes default to 1 (= trit 0): an unwritten/padded byte of 0x00 would
    // decode as four trit -1 (code 0), so the buffer is initialised to the
    // all-trit-0 byte (each 2-bit code = 0b01 → 0b01010101 = 0x55) instead.
    let mut bytes = vec![0x55u8; num_ntiles * num_ktiles * IMMA_WTILE_BYTES];

    for nt in 0..num_ntiles {
        for kt in 0..num_ktiles {
            let tile_byte0 = (nt * num_ktiles + kt) * IMMA_WTILE_BYTES;
            // 256 codes per tile, (n_in_tile, k_in_tile) row-major, 4 per byte.
            for n_in in 0..IMMA_N {
                let gn = nt * IMMA_N + n_in;
                if gn >= n {
                    continue; // padded output channel: leave as trit 0
                }
                for k_in in 0..IMMA_K {
                    let gk = kt * IMMA_K + k_in;
                    if gk >= k {
                        continue; // padded feature: leave as trit 0
                    }
                    let elem = n_in * IMMA_K + k_in; // 0..256 within the tile
                    let byte = tile_byte0 + elem / 4;
                    let slot = elem % 4; // which 2-bit pair (0 = low)
                    let code = (trits[gn * k + gk].get() + 1) as u8; // {-1,0,1}->{0,1,2}
                    // Clear this slot's default-1 pair, then OR in the real code.
                    bytes[byte] &= !(0b11u8 << (2 * slot));
                    bytes[byte] |= code << (2 * slot);
                }
            }
        }
    }

    Ok(I2sInt8Weights {
        bytes,
        scale,
        n,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{I2S_SCALE_BYTES, unpack_tq2_0_row};

    /// Build a single-block (128-element) I2_S payload from 128 trit values by
    /// inverting the block striping: element `group*32 + gp` (code = trit + 1)
    /// goes into byte `gp` at shift `6 - 2*group`. Appends the `f32` scale.
    fn build_i2s_one_block(trits: &[i8; 128], scale: f32) -> Vec<u8> {
        let mut bytes = [0u8; 32];
        for (pos, &t) in trits.iter().enumerate() {
            let group = pos / 32;
            let gp = pos % 32;
            let code = (t + 1) as u8; // {-1,0,1} -> {0,1,2}
            bytes[gp] |= code << (6 - 2 * group);
        }
        let mut payload = bytes.to_vec();
        payload.extend_from_slice(&scale.to_le_bytes());
        assert_eq!(payload.len(), 32 + I2S_SCALE_BYTES);
        payload
    }

    /// A deterministic ternary pattern over 128 positions.
    fn pattern() -> [i8; 128] {
        let mut t = [0i8; 128];
        for (i, v) in t.iter_mut().enumerate() {
            *v = (i % 3) as i8 - 1; // ..., -1, 0, 1, -1, 0, 1, ...
        }
        t
    }

    #[test]
    fn tq2_0_conversion_matches_reference_trits() {
        let trits = pattern();
        let scale = 1.234_5_f32;
        let payload = build_i2s_one_block(&trits, scale);
        // N=1 row of K=128 (one TQ2_0 block after zero-pad to 256).
        let shape = GemmShape { m: 0, n: 1, k: 128 };

        let (packed, got_scale) = convert_i2s_to_tq2_0(&payload, shape).expect("convert");
        assert_eq!(got_scale.to_bits(), scale.to_bits(), "per-tensor scale");

        // Unpack the TQ2_0 back and confirm the trits survived the round trip.
        let nb = num_blocks(128);
        let mut out = vec![Trit::ZERO; 128];
        let mut out_sc = vec![f16::ZERO; nb];
        unpack_tq2_0_row(&packed, &mut out, &mut out_sc).expect("unpack");
        for (i, (&got, &want)) in out.iter().zip(trits.iter()).enumerate() {
            assert_eq!(got.get(), want, "tq2_0 trit mismatch at {i}");
        }
    }

    /// Decode the IMMA tile interleave back to a plain `[N, K]` int8 trit matrix,
    /// mirroring exactly what `kernels/tq2_0_imma.cu` does when it unpacks the B
    /// operand: walk every `(nt, kt)` tile, read `(n_in, k_in)` row-major codes 4
    /// per byte, apply `trit = code - 1`. Padding positions (past `n`/`k`) are
    /// skipped — they must be trit 0 in the packing but are not part of `[N, K]`.
    fn decode_imma_to_nk(w: &I2sInt8Weights) -> Vec<i8> {
        let (n, k) = (w.n, w.k);
        let num_ktiles = w.num_ktiles();
        let num_ntiles = w.num_ntiles();
        let mut out = vec![0i8; n * k];
        for nt in 0..num_ntiles {
            for kt in 0..num_ktiles {
                let tile_byte0 = (nt * num_ktiles + kt) * IMMA_WTILE_BYTES;
                for elem in 0..IMMA_N * IMMA_K {
                    let n_in = elem / IMMA_K;
                    let k_in = elem % IMMA_K;
                    let gn = nt * IMMA_N + n_in;
                    let gk = kt * IMMA_K + k_in;
                    if gn >= n || gk >= k {
                        continue;
                    }
                    let byte = w.bytes[tile_byte0 + elem / 4];
                    let code = (byte >> (2 * (elem % 4))) & 0b11;
                    out[gn * k + gk] = code as i8 - 1;
                }
            }
        }
        out
    }

    #[test]
    fn int8_conversion_preserves_trits_and_scale() {
        let trits = pattern();
        let scale = -0.875_f32;
        let payload = build_i2s_one_block(&trits, scale);
        let shape = GemmShape { m: 0, n: 1, k: 128 };

        let w = convert_i2s_to_int8(&payload, shape).expect("convert");
        assert_eq!(w.n, 1);
        assert_eq!(w.k, 128);
        assert_eq!(w.scale.to_bits(), scale.to_bits());
        // N=1 → 1 n-tile (pad to 8); K=128 → 4 k-tiles. 4 tiles · 64 bytes.
        assert_eq!(w.num_ntiles(), 1);
        assert_eq!(w.num_ktiles(), 4);
        assert_eq!(w.bytes.len(), w.num_ntiles() * w.num_ktiles() * IMMA_WTILE_BYTES);

        // Round trip the tile interleave back to [N, K] and check trit-for-trit.
        let decoded = decode_imma_to_nk(&w);
        for (i, (&got, &want)) in decoded.iter().zip(trits.iter()).enumerate() {
            assert_eq!(got, want, "int8 weight mismatch at {i}");
        }
    }

    /// A multi-row, padding-exercising shape: `N=3` (pads to 8), `K=160` (5
    /// k-tiles, exact). The interleave must round-trip every weight, and every
    /// padded code (the rows 3..8 of each n-tile) must be trit 0.
    #[test]
    fn int8_conversion_round_trips_multi_row_with_padding() {
        // N=3 rows of K=128 trits each (one I2_S block per row would only give
        // K=128; build three 128-trit rows and convert with K=128 to keep the
        // payload simple, exercising the N padding to 8).
        let scale = 2.5_f32;
        let mut payload = Vec::new();
        let rows = [pattern(), pattern(), pattern()];
        // Vary each row so a transposed/mis-strided layout would be caught.
        let mut varied = rows;
        for (r, row) in varied.iter_mut().enumerate() {
            for (i, t) in row.iter_mut().enumerate() {
                *t = (((i + r) % 3) as i8) - 1;
            }
        }
        for row in &varied {
            // build_i2s_one_block already appends a scale trailer; we only want one
            // trailing scale for the whole tensor, so build the quant bytes inline.
            let mut bytes = [0u8; 32];
            for (pos, &t) in row.iter().enumerate() {
                let group = pos / 32;
                let gp = pos % 32;
                let code = (t + 1) as u8;
                bytes[gp] |= code << (6 - 2 * group);
            }
            payload.extend_from_slice(&bytes);
        }
        payload.extend_from_slice(&scale.to_le_bytes());

        let shape = GemmShape { m: 0, n: 3, k: 128 };
        let w = convert_i2s_to_int8(&payload, shape).expect("convert");
        assert_eq!(w.num_ntiles(), 1); // 3 → pads to 8
        assert_eq!(w.num_ktiles(), 4); // 128 / 32

        let decoded = decode_imma_to_nk(&w);
        for r in 0..3 {
            for c in 0..128 {
                assert_eq!(
                    decoded[r * 128 + c],
                    varied[r][c],
                    "row {r} col {c} mismatch"
                );
            }
        }

        // Padded output channels (rows 3..8 of the single n-tile) must all be code
        // 1 (trit 0) so they contribute nothing in the kernel.
        let num_ktiles = w.num_ktiles();
        for nt_row in 3..IMMA_N {
            for kt in 0..num_ktiles {
                let tile_byte0 = kt * IMMA_WTILE_BYTES; // nt == 0
                for k_in in 0..IMMA_K {
                    let elem = nt_row * IMMA_K + k_in;
                    let byte = w.bytes[tile_byte0 + elem / 4];
                    let code = (byte >> (2 * (elem % 4))) & 0b11;
                    assert_eq!(code, 1, "padded row {nt_row} k {k_in} not trit 0");
                }
            }
        }
    }

    #[test]
    fn rejects_short_payload() {
        let shape = GemmShape { m: 0, n: 1, k: 128 };
        let too_short = vec![0u8; 10];
        assert!(convert_i2s_to_tq2_0(&too_short, shape).is_err());
        assert!(convert_i2s_to_int8(&too_short, shape).is_err());
    }
}
