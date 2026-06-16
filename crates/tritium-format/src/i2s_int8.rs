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
//!   prefill kernel. **The byte layout here is provisional** — a plain `[N, K]`
//!   row-major int8 baseline that decodes back to the reference trits. WF-A pins
//!   the actual fragment interleave the `mma` instruction wants once the kernel
//!   tiling is fixed; the plain baseline stays as the correctness anchor the
//!   interleaved layout is validated against (interleaved == plain == reference).

use half::f16;
use tritium_core::{GemmShape, Trit};

use crate::{FormatError, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row, unpack_i2s_tensor};

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
/// `bytes` carries one int8 per weight (each in `{-1, 0, 1}` reinterpreted to
/// `u8`, so `-1` is `0xFF`); `scale` is the per-tensor magnitude; `n`/`k` are the
/// `[N, K]` shape. The interpretation of `bytes` is the **provisional plain
/// `[N, K]` row-major** layout (see the module docs) until WF-A fixes the
/// `mma.m16n8k32` fragment interleave.
#[derive(Debug, Clone)]
pub struct I2sInt8Weights {
    /// One int8 per weight, reinterpreted to `u8` (provisional plain `[N, K]`).
    pub bytes: Vec<u8>,
    /// Per-tensor `f32` magnitude scale carried by the I2_S source.
    pub scale: f32,
    /// Output channels (rows).
    pub n: usize,
    /// Input features (columns).
    pub k: usize,
}

/// Decode an I2_S weight tensor into the IMMA int8 layout ([`I2sInt8Weights`]).
///
/// Currently emits the **provisional plain `[N, K]` int8 baseline** (`bytes[i] =
/// trit_i as i8 as u8`). WF-A replaces the body with the `mma.m16n8k32` fragment
/// interleave; this baseline remains the correctness anchor (the interleaved
/// layout must decode to the same trits).
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

    // Provisional: plain row-major int8, one byte per weight. WF-A: interleave.
    let bytes = trits.iter().map(|t| t.get() as u8).collect();
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
        assert_eq!(w.bytes.len(), 128);
        for (i, (&byte, &want)) in w.bytes.iter().zip(trits.iter()).enumerate() {
            assert_eq!(byte as i8, want, "int8 weight mismatch at {i}");
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
