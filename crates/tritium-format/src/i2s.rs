//! I2_S — the BitNet / bitnet.cpp ternary GGUF format. 128 ternary elements per
//! 32-byte block: 2 bits each, `code = trit + 1` (so `0b00`=-1, `0b01`=0,
//! `0b10`=+1); byte `gp` (`0..32`) holds the elements at positions `[gp, 32+gp,
//! 64+gp, 96+gp]` in bit-pairs `[7:6],[5:4],[3:2],[1:0]`.
//!
//! The `code = trit + 1` mapping (verified in WF-4) is the same `+1` offset
//! `transformers`' `unpack_weights` uses (`value - 1` on decode), and it is what
//! `ggml-bitnet`'s `quantize_i2_s` writes: `q8[i] = src[i]*scale > 0 ? 2 : 0`,
//! with the near-zero case set to `1`. Decoding the 32-byte block striping and
//! applying `trit = code - 1` reproduces the block-linear stream, which **is** the
//! tensor in ggml memory order — i.e. plain `[N, K]` row-major. No further element
//! reorder is needed.
//!
//! Unlike TQ1_0 / TQ2_0, the magnitude scale is **not** stored inside each block.
//! An I2_S tensor lays out all of its 2-bit quants first (`n_elements / 4` bytes),
//! immediately followed by a **single per-tensor `f32` scale**, the whole payload
//! then padded up to the GGUF alignment (32 B). This was confirmed by reading the
//! official `microsoft/bitnet-b1.58-2B-4T-gguf` `ggml-model-i2_s.gguf`: the GGUF
//! `ggml_type` of the 210 ternary weight tensors is **36**, every tensor's payload
//! is exactly `n_elements/4 + 32` bytes (quants + one `f32` + pad), and the `f32`
//! matches the `weight_scale` (shape `[1]`) carried by the reference HF checkpoint
//! (`microsoft/bitnet-b1.58-2B-4T`, `BitLinear` `autobitlinear`). With the
//! `trit = code - 1` mapping the decoded trits match that checkpoint's unpacked
//! ternary weights **bit-exactly** across all seven layer-0 projections (100%
//! element match, all shapes: 2560×2560, 640×2560, 2560×6912, 6912×2560), so the
//! v0.20 plan's per-tensor-scale assumption holds (the v0.10 per-channel `mpgemm`
//! reuses it as a single broadcast scale, equivalently a per-row scale that is
//! constant per tensor).

use tritium_core::Trit;

use crate::FormatError;

/// ggml type-id for I2_S. Confirmed in WF-1 against the official BitNet 2B4T GGUF:
/// all 210 ternary weight tensors carry `ggml_type == 36` (bitnet.cpp registers this
/// id; mainline `gguf-py` 0.x does not yet know it, so it must be sized by this crate).
pub const GGML_TYPE_I2_S: u32 = 36;

/// Ternary elements per I2_S block.
pub const I2S_BLOCK_ELEMS: usize = 128;

/// Bytes per I2_S block (128 elements × 2 bits).
pub const I2S_BLOCK_BYTES: usize = I2S_BLOCK_ELEMS / 4;

/// Bytes of the per-tensor scale trailer: a single little-endian `f32`.
pub const I2S_SCALE_BYTES: usize = 4;

/// Decode one 2-bit I2_S code into a [`Trit`] via `trit = code - 1`.
///
/// `0b00`→-1, `0b01`→0, `0b10`→+1; `0b11` (= trit `+2`) is invalid and yields
/// [`FormatError::InvalidI2sCode`]. This is the `+1`-offset BitNet uses on both
/// sides: `transformers` decodes with `value - 1`, and `ggml-bitnet`'s
/// `quantize_i2_s` encodes a positive weight as `2`, a negative as `0`, and a
/// near-zero as `1`.
#[inline]
fn code_to_trit(code: u8) -> Result<Trit, FormatError> {
    match code {
        0b00 => Ok(Trit::NEG),
        0b01 => Ok(Trit::ZERO),
        0b10 => Ok(Trit::POS),
        // 0b11 is the only remaining value; reject it rather than silently mapping.
        other => Err(FormatError::InvalidI2sCode(other)),
    }
}

/// Unpack one 32-byte I2_S block into 128 trits.
///
/// Byte `gp` (`0..32`) supplies the elements at positions `[gp, 32+gp, 64+gp,
/// 96+gp]`, taken from the bit-pairs `[7:6]`, `[5:4]`, `[3:2]`, `[1:0]`
/// respectively. Codes decode as `trit = code - 1`: `0b00`=-1, `0b01`=0,
/// `0b10`=+1.
///
/// # Errors
/// - [`FormatError::WrongBlockLen`] if `block` is not [`I2S_BLOCK_BYTES`] bytes.
/// - [`FormatError::WrongTritCount`] if `trits_out` is not [`I2S_BLOCK_ELEMS`] long.
/// - [`FormatError::InvalidI2sCode`] if any 2-bit code is the reserved `0b11`.
pub fn unpack_i2s_block(block: &[u8], trits_out: &mut [Trit]) -> Result<(), FormatError> {
    if block.len() != I2S_BLOCK_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: I2S_BLOCK_BYTES,
            got: block.len(),
        });
    }
    if trits_out.len() != I2S_BLOCK_ELEMS {
        return Err(FormatError::WrongTritCount {
            expected: I2S_BLOCK_ELEMS,
            got: trits_out.len(),
        });
    }

    // The four sub-groups of 32 elements live in descending bit-pairs of each byte:
    // group g (positions g*32 .. g*32+32) is held at shift 6 - 2*g.
    for (gp, &byte) in block.iter().enumerate() {
        for group in 0..4 {
            let shift = 6 - 2 * group;
            let code = (byte >> shift) & 0b11;
            trits_out[group * 32 + gp] = code_to_trit(code)?;
        }
    }
    Ok(())
}

/// Decode a whole I2_S tensor payload (`n_elements/4` quant bytes followed by one
/// trailing `f32` scale) into `n_elements` trits plus the per-tensor scale.
///
/// `payload` is the tensor's raw bytes located via [`crate::TensorInfo::offset`] /
/// [`crate::TensorInfo::n_bytes`]; `n_elements` is its element count. `n_elements`
/// must be a multiple of [`I2S_BLOCK_ELEMS`] (BitNet tensors always are — every
/// dimension is a multiple of 128). The dequantized weight for element `i` is
/// `trits_out[i].to_f32() * scale`.
///
/// # Errors
/// - [`FormatError::WrongTritCount`] if `trits_out.len() != n_elements` or
///   `n_elements` is not a multiple of [`I2S_BLOCK_ELEMS`].
/// - [`FormatError::WrongBlockLen`] if `payload` is shorter than
///   `n_elements/4 + I2S_SCALE_BYTES`.
/// - [`FormatError::InvalidI2sCode`] on a reserved `0b11` code.
pub fn unpack_i2s_tensor(
    payload: &[u8],
    n_elements: usize,
    trits_out: &mut [Trit],
) -> Result<f32, FormatError> {
    if trits_out.len() != n_elements || !n_elements.is_multiple_of(I2S_BLOCK_ELEMS) {
        return Err(FormatError::WrongTritCount {
            expected: n_elements,
            got: trits_out.len(),
        });
    }
    let n_quant_bytes = n_elements / 4;
    let need = n_quant_bytes + I2S_SCALE_BYTES;
    if payload.len() < need {
        return Err(FormatError::WrongBlockLen {
            expected: need,
            got: payload.len(),
        });
    }

    let quants = &payload[..n_quant_bytes];
    for (block, out) in quants
        .chunks_exact(I2S_BLOCK_BYTES)
        .zip(trits_out.chunks_exact_mut(I2S_BLOCK_ELEMS))
    {
        unpack_i2s_block(block, out)?;
    }

    // The scale is the little-endian f32 immediately after the quants.
    let scale_bytes = [
        payload[n_quant_bytes],
        payload[n_quant_bytes + 1],
        payload[n_quant_bytes + 2],
        payload[n_quant_bytes + 3],
    ];
    Ok(f32::from_le_bytes(scale_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 32-byte block from four parallel arrays of 32 codes each (the codes
    /// for positions `0..32`, `32..64`, `64..96`, `96..128`).
    fn pack_codes(g0: &[u8; 32], g1: &[u8; 32], g2: &[u8; 32], g3: &[u8; 32]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (gp, b) in out.iter_mut().enumerate() {
            *b = (g0[gp] << 6) | (g1[gp] << 4) | (g2[gp] << 2) | g3[gp];
        }
        out
    }

    #[test]
    fn hand_golden_block_decodes() {
        // `trit = code - 1` (0b00=-1, 0b01=0, 0b10=+1).
        // byte gp=0 = 0b10_01_01_00 = 0x94 -> pos0=+1, pos32=0, pos64=0, pos96=-1.
        // byte gp=1 = 0b01_10_00_01 = 0x61 -> pos1=0,  pos33=+1, pos65=-1, pos97=0.
        // every other byte 0b01_01_01_01 = 0x55 -> all zero (code 1 = trit 0).
        let mut block = [0b01_01_01_01u8; 32];
        block[0] = 0b10_01_01_00;
        block[1] = 0b01_10_00_01;

        let mut out = [Trit::ZERO; 128];
        unpack_i2s_block(&block, &mut out).expect("decode");

        assert_eq!(out[0], Trit::POS);
        assert_eq!(out[32], Trit::ZERO);
        assert_eq!(out[64], Trit::ZERO);
        assert_eq!(out[96], Trit::NEG);
        assert_eq!(out[1], Trit::ZERO);
        assert_eq!(out[33], Trit::POS);
        assert_eq!(out[65], Trit::NEG);
        assert_eq!(out[97], Trit::ZERO);
        // Everything else is zero (code 0b01).
        for (i, &t) in out.iter().enumerate() {
            if ![0, 1, 32, 33, 64, 65, 96, 97].contains(&i) {
                assert_eq!(t, Trit::ZERO, "index {i}");
            }
        }
    }

    #[test]
    fn all_three_codes_stripe_each_position() {
        // Drive each of the four sub-groups with a distinct code and confirm the
        // [gp, 32+gp, 64+gp, 96+gp] striping lands every element where expected.
        // `trit = code - 1`.
        let g0 = [0b10u8; 32]; // +1 across positions 0..32   (code 2)
        let g1 = [0b00u8; 32]; // -1 across positions 32..64  (code 0)
        let g2 = [0b01u8; 32]; //  0 across positions 64..96  (code 1)
        let g3 = [0b10u8; 32]; // +1 across positions 96..128 (code 2)
        let block = pack_codes(&g0, &g1, &g2, &g3);

        let mut out = [Trit::ZERO; 128];
        unpack_i2s_block(&block, &mut out).expect("decode");
        for i in 0..32 {
            assert_eq!(out[i], Trit::POS, "g0 idx {i}");
            assert_eq!(out[32 + i], Trit::NEG, "g1 idx {i}");
            assert_eq!(out[64 + i], Trit::ZERO, "g2 idx {i}");
            assert_eq!(out[96 + i], Trit::POS, "g3 idx {i}");
        }
    }

    #[test]
    fn real_golden_block_from_bitnet_gguf() {
        // First 32-byte block of `blk.0.attn_q.weight` in the official
        // ggml-model-i2_s.gguf, plus the 128 trits a validated Python decode
        // produced for it, and the tensor's per-tensor f32 scale. The trits also
        // match `microsoft/bitnet-b1.58-2B-4T`'s HF weights bit-exactly.
        const BLOCK: [u8; 32] = [
            0x42, 0x48, 0x61, 0x29, 0x55, 0x12, 0x44, 0x55, 0x55, 0x19, 0x54, 0x4a, 0xa1, 0x55,
            0x65, 0x51, 0x15, 0x51, 0x19, 0x55, 0x60, 0x51, 0x41, 0x44, 0x56, 0x55, 0x55, 0x14,
            0x51, 0x06, 0x45, 0x45,
        ];
        // Trits under the `trit = code - 1` mapping; these match
        // `microsoft/bitnet-b1.58-2B-4T`'s unpacked `q_proj.weight` row 0 cols
        // 0..128 (validated bit-exactly against the HF checkpoint in WF-4).
        #[rustfmt::skip]
        const EXPECT: [i8; 128] = [
            0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, 0, -1, -1, 1, 1, 0, 0, -1, 0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
            -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, -1, 1, 0, -1, 0, 0, 0, 1, 0, 1, -1, 0, 0, -1,
            0, -1, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, -1, 0, 0, 0, 1, -1, 0, 0, 0,
            -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 1, 0, 0, -1, 0, 1, 0, 0,
        ];

        let mut out = [Trit::ZERO; 128];
        unpack_i2s_block(&BLOCK, &mut out).expect("decode real block");
        for (i, (&got, &want)) in out.iter().zip(EXPECT.iter()).enumerate() {
            assert_eq!(got.get(), want, "real golden trit mismatch at {i}");
        }
    }

    #[test]
    fn invalid_code_rejected() {
        // Any byte with a 0b11 pair must error rather than decode.
        let mut block = [0u8; 32];
        block[5] = 0b11_00_00_00; // pos5 = 0b11
        let mut out = [Trit::ZERO; 128];
        assert_eq!(
            unpack_i2s_block(&block, &mut out),
            Err(FormatError::InvalidI2sCode(0b11))
        );
    }

    #[test]
    fn wrong_block_len_rejected() {
        let mut out = [Trit::ZERO; 128];
        assert_eq!(
            unpack_i2s_block(&[0u8; 31], &mut out),
            Err(FormatError::WrongBlockLen {
                expected: 32,
                got: 31
            })
        );
    }

    #[test]
    fn wrong_trit_count_rejected() {
        let mut out = [Trit::ZERO; 127];
        assert_eq!(
            unpack_i2s_block(&[0u8; 32], &mut out),
            Err(FormatError::WrongTritCount {
                expected: 128,
                got: 127
            })
        );
    }

    #[test]
    fn tensor_decode_reads_trailing_scale() {
        // One block of quants (pos0..32 = +1 via code 0b10 in the top pair, the
        // other three pairs 0b01 = trit 0) followed by a known f32 scale; the
        // helper must split them correctly. `trit = code - 1`.
        let mut payload = vec![0u8; I2S_BLOCK_BYTES + I2S_SCALE_BYTES];
        for b in payload.iter_mut().take(I2S_BLOCK_BYTES) {
            *b = 0b10_01_01_01;
        }
        let scale: f32 = 1.218_854_8;
        payload[I2S_BLOCK_BYTES..].copy_from_slice(&scale.to_le_bytes());

        let mut trits = [Trit::ZERO; 128];
        let got = unpack_i2s_tensor(&payload, 128, &mut trits).expect("tensor decode");
        assert_eq!(got.to_bits(), scale.to_bits());
        for (i, t) in trits[..32].iter().enumerate() {
            assert_eq!(*t, Trit::POS, "idx {i}");
        }
        for t in &trits[32..] {
            assert_eq!(*t, Trit::ZERO);
        }
    }

    #[test]
    fn tensor_decode_rejects_short_payload() {
        let payload = vec![0u8; I2S_BLOCK_BYTES]; // missing the 4-byte scale
        let mut trits = [Trit::ZERO; 128];
        assert_eq!(
            unpack_i2s_tensor(&payload, 128, &mut trits),
            Err(FormatError::WrongBlockLen {
                expected: I2S_BLOCK_BYTES + I2S_SCALE_BYTES,
                got: I2S_BLOCK_BYTES,
            })
        );
    }

    #[test]
    fn tensor_decode_requires_block_multiple() {
        let payload = vec![0u8; 100];
        let mut trits = [Trit::ZERO; 100];
        assert_eq!(
            unpack_i2s_tensor(&payload, 100, &mut trits),
            Err(FormatError::WrongTritCount {
                expected: 100,
                got: 100,
            })
        );
    }
}
