//! I2_S — the BitNet / bitnet.cpp ternary GGUF format. 128 ternary elements per
//! 32-byte block: 2 bits each, `0b00`=0, `0b01`=+1, `0b10`=-1; byte `gp` (`0..32`)
//! holds the elements at positions `[gp, 32+gp, 64+gp, 96+gp]` in bit-pairs
//! `[7:6],[5:4],[3:2],[1:0]`. The magnitude scale is stored separately (per-row /
//! per-tensor f32).
//!
//! STUB (v0.20 WF-0). The real decoder and the exact ggml type-id are confirmed in
//! WF-1 by reading the official `ggml-model-i2_s.gguf` and cross-checking the
//! dequantized values against `transformers`.

use tritium_core::Trit;

use crate::FormatError;

/// ggml type-id for I2_S. PROVISIONAL — confirm in WF-1 against the weight tensors'
/// `ggml_type` in the real BitNet GGUF (bitnet.cpp registers a custom id).
pub const GGML_TYPE_I2_S: u32 = 36;

/// Ternary elements per I2_S block.
pub const I2S_BLOCK_ELEMS: usize = 128;

/// Bytes per I2_S block (128 elements × 2 bits).
pub const I2S_BLOCK_BYTES: usize = I2S_BLOCK_ELEMS / 4;

/// Unpack one 32-byte I2_S block into 128 trits.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `block` is not [`I2S_BLOCK_BYTES`] bytes, or
/// [`FormatError::WrongTritCount`] if `trits_out` is not [`I2S_BLOCK_ELEMS`] long.
pub fn unpack_i2s_block(block: &[u8], trits_out: &mut [Trit]) -> Result<(), FormatError> {
    let _ = (block, trits_out);
    todo!("WF-1: implement I2_S unpack; validate vs a Python decode of the real file")
}
