//! Host-side tensor helpers.
//!
//! Non-ternary weights (norms, embeddings, the LM head) and all intermediate
//! activations live host-side as **fp32** `Vec<f32>`: it is the reorder-free
//! reference precision for the fidelity ladder (the CPU oracle runs in fp32 so
//! drift can be localized stage-by-stage), and the ternary `mpgemm` contract is
//! already f32-in. GGUF stores these tensors as `f16`/`bf16`; we widen to fp32 on
//! load. Ternary weights take the separate I2_S → internal path in
//! `tritium-format`.

use half::f16;

/// Widen a little-endian `f16` byte blob (as stored in a GGUF tensor) to a
/// `Vec<f32>`.
///
/// `bytes.len()` should be even (2 bytes per `f16`); a trailing odd byte, if any,
/// is ignored. Each `f16` is read little-endian and losslessly widened to `f32`.
#[must_use]
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widens_known_values() {
        // 1.0, -2.0, 0.0 as little-endian f16 bits.
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.0, 0.0] {
            bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        let out = f16_bytes_to_f32(&bytes);
        assert_eq!(out, vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn ignores_trailing_odd_byte() {
        let bytes = f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut v = bytes.to_vec();
        v.push(0xFF); // stray odd byte
        assert_eq!(f16_bytes_to_f32(&v), vec![1.0]);
    }
}
