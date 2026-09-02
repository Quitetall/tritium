//! Packed standard GGUF Q2_0 projection.
//!
//! Q2_0 carries one f16 scale per 64 ternary coefficients. Unlike
//! [`TernaryLinear`](super::TernaryLinear), its scale can vary inside an output
//! row, so it cannot be normalized into one per-row multiplier without losing
//! information. This portable path retains exact packed bytes and contracts one
//! group at a time through the same A8 activation quantizer used by deployed
//! ternary and SALT projections.

use std::sync::Arc;

use half::f16;
use rayon::prelude::*;
use tritium_core::Trit;
use tritium_format::{Q2_0_BLOCK_BYTES, Q2_0_GROUP_SIZE, q2_0_num_blocks, unpack_q2_0_block};

use crate::error::NnError;
use crate::ops::quantize_activation_int8;

/// One `[n_out, k_in]` standard Q2_0 matrix retained in packed form.
#[derive(Clone, Debug)]
pub struct Q2Linear {
    /// Output feature count.
    n_out: usize,
    /// Input feature count.
    k_in: usize,
    packed: Arc<[u8]>,
    uniform_scale_override: Option<f32>,
}

impl Q2Linear {
    /// Validate and retain standard Q2_0 row-major payload bytes.
    ///
    /// GGUF block rows cannot straddle the fastest-varying dimension, so
    /// `k_in` must be a nonzero multiple of 64. Code 3 (`+2`) is rejected:
    /// Tritium's Q2_0 interop surface is ternary `{-1,0,+1}`, not generic 2-bit
    /// quantization. Every scale must be finite.
    pub fn new(packed: Vec<u8>, n_out: usize, k_in: usize) -> Result<Self, NnError> {
        Self::new_with_uniform_scale_override(packed, n_out, k_in, None)
    }

    pub(crate) fn new_with_uniform_scale_override(
        packed: Vec<u8>,
        n_out: usize,
        k_in: usize,
        uniform_scale_override: Option<f32>,
    ) -> Result<Self, NnError> {
        if n_out == 0 || k_in == 0 || !k_in.is_multiple_of(Q2_0_GROUP_SIZE) {
            return Err(NnError::Shape {
                expected: Q2_0_GROUP_SIZE,
                got: k_in,
            });
        }
        let row_bytes =
            q2_0_num_blocks(k_in)
                .checked_mul(Q2_0_BLOCK_BYTES)
                .ok_or(NnError::Shape {
                    expected: usize::MAX,
                    got: packed.len(),
                })?;
        let expected = n_out.checked_mul(row_bytes).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: packed.len(),
        })?;
        if packed.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: packed.len(),
            });
        }

        let rounded_override = match uniform_scale_override {
            Some(value) if value.is_finite() => Some(f16::from_f32(value)),
            Some(_) => {
                return Err(NnError::Backend(
                    "Q2_0 uniform scale override is non-finite".to_owned(),
                ));
            }
            None => None,
        };
        let mut trits = [Trit::ZERO; Q2_0_GROUP_SIZE];
        for (block_index, block) in packed.as_chunks::<Q2_0_BLOCK_BYTES>().0.iter().enumerate() {
            let mut scale = f16::ZERO;
            unpack_q2_0_block(block, &mut trits, &mut scale)
                .map_err(|error| NnError::Backend(format!("Q2_0 block {block_index}: {error}")))?;
            if !scale.is_finite() {
                return Err(NnError::Backend(format!(
                    "Q2_0 block {block_index} has non-finite scale bits 0x{:04x}",
                    scale.to_bits()
                )));
            }
            if scale != f16::ZERO
                && rounded_override.is_some_and(|expected| expected.to_bits() != scale.to_bits())
            {
                return Err(NnError::Backend(format!(
                    "Q2_0 block {block_index} scale {} disagrees with rounded uniform override {}",
                    f32::from(scale),
                    f32::from(rounded_override.expect("checked Some"))
                )));
            }
        }

        Ok(Self {
            n_out,
            k_in,
            packed: packed.into(),
            uniform_scale_override,
        })
    }

    /// Output feature count.
    #[must_use]
    pub const fn n_out(&self) -> usize {
        self.n_out
    }

    /// Input feature count.
    #[must_use]
    pub const fn k_in(&self) -> usize {
        self.k_in
    }

    /// Exact packed payload size retained by this projection.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.packed.len()
    }

    /// A8 forward with per-Q2_0-group scale application.
    pub fn forward(&self, act: &[f32], m: usize, out: &mut [f32]) -> Result<(), NnError> {
        self.validate_retained_geometry()?;
        let act_len = m.checked_mul(self.k_in).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: act.len(),
        })?;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m.checked_mul(self.n_out).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: out.len(),
        })?;
        if out.len() != out_len {
            return Err(NnError::Shape {
                expected: out_len,
                got: out.len(),
            });
        }

        let mut q_act = zeroed_scratch(act_len, "Q2_0 quantized activations")?;
        let mut act_scales = zeroed_scratch(m, "Q2_0 activation scales")?;
        quantize_activation_int8(act, m, self.k_in, &mut q_act, &mut act_scales)?;

        let groups = q2_0_num_blocks(self.k_in);
        let row_bytes = groups * Q2_0_BLOCK_BYTES;
        for (activation_row, output_row) in out.chunks_mut(self.n_out).enumerate() {
            let qrow = &q_act[activation_row * self.k_in..(activation_row + 1) * self.k_in];
            let activation_scale = act_scales[activation_row];
            output_row
                .par_iter_mut()
                .enumerate()
                .for_each(|(output_channel, slot)| {
                    let packed_row =
                        &self.packed[output_channel * row_bytes..(output_channel + 1) * row_bytes];
                    let mut sum = 0.0f32;
                    let mut uniform_dot = 0.0f32;
                    for group in 0..groups {
                        let block =
                            &packed_row[group * Q2_0_BLOCK_BYTES..(group + 1) * Q2_0_BLOCK_BYTES];
                        let stored_scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]]));
                        let qs = &block[2..];
                        let mut group_dot = 0.0f32;
                        for index in 0..Q2_0_GROUP_SIZE {
                            let code = (qs[index / 4] >> (2 * (index % 4))) & 3;
                            debug_assert!(code < 3, "constructor validates ternary Q2_0 codes");
                            let trit = code as f32 - 1.0;
                            group_dot += qrow[group * Q2_0_GROUP_SIZE + index] * trit;
                        }
                        if stored_scale == f16::ZERO {
                            continue;
                        }
                        if self.uniform_scale_override.is_some() {
                            uniform_dot += group_dot;
                        } else {
                            sum += group_dot * f32::from(stored_scale);
                        }
                    }
                    if let Some(scale) = self.uniform_scale_override {
                        sum = uniform_dot * scale;
                    }
                    *slot = sum * activation_scale;
                });
        }
        Ok(())
    }

    pub(crate) fn validate_retained_geometry(&self) -> Result<(), NnError> {
        let row_bytes = q2_0_num_blocks(self.k_in)
            .checked_mul(Q2_0_BLOCK_BYTES)
            .ok_or(NnError::Shape {
                expected: usize::MAX,
                got: self.packed.len(),
            })?;
        let expected = self.n_out.checked_mul(row_bytes).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: self.packed.len(),
        })?;
        if self.packed.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: self.packed.len(),
            });
        }
        Ok(())
    }
}

fn zeroed_scratch(len: usize, label: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate {label} scratch for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::pack_q2_0_row;

    fn packed_row(scales: &[f16]) -> Vec<u8> {
        let k = scales.len() * Q2_0_GROUP_SIZE;
        let trits: Vec<Trit> = (0..k)
            .map(|index| Trit::from_i8((index % 3) as i8 - 1).expect("trit"))
            .collect();
        let mut packed = vec![0u8; scales.len() * Q2_0_BLOCK_BYTES];
        pack_q2_0_row(&trits, scales, &mut packed).expect("pack Q2_0 row");
        packed
    }

    #[test]
    fn constructor_rejects_nonternary_codes_nonfinite_scales_and_bad_geometry() {
        let mut code_three = packed_row(&[f16::ONE]);
        code_three[2] = (code_three[2] & !3) | 3;
        assert!(matches!(
            Q2Linear::new(code_three, 1, Q2_0_GROUP_SIZE),
            Err(NnError::Backend(message)) if message.contains("outside ternary range")
        ));

        let mut nonfinite = packed_row(&[f16::ONE]);
        nonfinite[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        assert!(matches!(
            Q2Linear::new(nonfinite, 1, Q2_0_GROUP_SIZE),
            Err(NnError::Backend(message)) if message.contains("non-finite")
        ));

        assert!(matches!(
            Q2Linear::new(vec![0; Q2_0_BLOCK_BYTES], 1, Q2_0_GROUP_SIZE - 1),
            Err(NnError::Shape { .. })
        ));
    }

    #[test]
    fn uniform_override_is_source_bound_and_zero_blocks_remain_zero() {
        let exact = 0.123_456_7f32;
        let rounded = f16::from_f32(exact);
        let mut packed = packed_row(&[rounded, f16::ZERO]);
        // A stale override must refuse before publishing the projection.
        assert!(matches!(
            Q2Linear::new_with_uniform_scale_override(
                packed.clone(),
                1,
                2 * Q2_0_GROUP_SIZE,
                Some(exact * 2.0),
            ),
            Err(NnError::Backend(message)) if message.contains("disagrees")
        ));

        // Keep nonzero codes in the zero-scale block. Override must not revive them.
        packed[Q2_0_BLOCK_BYTES + 2..2 * Q2_0_BLOCK_BYTES].fill(0xAA);
        let linear =
            Q2Linear::new_with_uniform_scale_override(packed, 1, 2 * Q2_0_GROUP_SIZE, Some(exact))
                .expect("source-bound Q2_0");
        let mut act = vec![0.0f32; 2 * Q2_0_GROUP_SIZE];
        act[Q2_0_GROUP_SIZE..].fill(1.0);
        let mut out = [17.0f32];
        linear.forward(&act, 1, &mut out).expect("Q2_0 forward");
        assert_eq!(out, [0.0], "zero-scale block must stay semantically zero");
    }
}
