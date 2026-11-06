//! Constant-carry little-endian widening into caller-owned private staging.
use super::{ScalePrecision, TritModelError};
use half::{bf16, f16};

/// Incremental finite f16/bf16/f32 decoder with no owned encoded/scalar buffers.
/// Writes are nontransactional; keep output private until finish and enclosing
/// source verification succeed. Errors poison the decoder, even if ignored.
pub struct DensePayloadDecoder<'a> {
    precision: ScalePrecision,
    output: &'a mut [f32],
    width: usize,
    expected: usize,
    bytes: usize,
    values: usize,
    carry: [u8; 4],
    filled: usize,
    poisoned: bool,
}
impl std::fmt::Debug for DensePayloadDecoder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DensePayloadDecoder")
            .field("precision", &self.precision)
            .field("expected_bytes", &self.expected)
            .field("consumed_bytes", &self.bytes)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}
fn invalid(message: &str) -> TritModelError {
    TritModelError::DensePayload(message.into())
}
impl<'a> DensePayloadDecoder<'a> {
    /// Checked encoded size without reading/allocating a payload.
    pub fn byte_len(precision: ScalePrecision, values: usize) -> Result<usize, TritModelError> {
        let width = match precision {
            ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
            ScalePrecision::F32 => 4,
            _ => return Err(invalid("unsupported dense stream precision")),
        };
        values
            .checked_mul(width)
            .ok_or_else(|| invalid("dense stream length overflow"))
    }
    /// Bind exact private destination storage; no allocation or payload copy.
    pub fn new(precision: ScalePrecision, output: &'a mut [f32]) -> Result<Self, TritModelError> {
        let expected = Self::byte_len(precision, output.len())?;
        Ok(Self {
            precision,
            output,
            width: if precision == ScalePrecision::F32 {
                4
            } else {
                2
            },
            expected,
            bytes: 0,
            values: 0,
            carry: [0; 4],
            filled: 0,
            poisoned: false,
        })
    }
    /// Widen one arbitrary byte window, carrying at most one incomplete scalar.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), TritModelError> {
        if self.poisoned {
            return Err(invalid("dense stream decoder is poisoned"));
        }
        let result = self.push_checked(chunk);
        self.poisoned = result.is_err();
        result
    }
    fn push_checked(&mut self, chunk: &[u8]) -> Result<(), TritModelError> {
        self.bytes = self
            .bytes
            .checked_add(chunk.len())
            .ok_or_else(|| invalid("dense stream length overflow"))?;
        if self.bytes > self.expected {
            return Err(invalid("dense stream exceeds declared length"));
        }
        for &byte in chunk {
            self.carry[self.filled] = byte;
            self.filled += 1;
            if self.filled == self.width {
                let value = match self.precision {
                    ScalePrecision::F16 => {
                        f16::from_bits(u16::from_le_bytes([self.carry[0], self.carry[1]])).to_f32()
                    }
                    ScalePrecision::Bf16 => {
                        bf16::from_bits(u16::from_le_bytes([self.carry[0], self.carry[1]])).to_f32()
                    }
                    ScalePrecision::F32 => f32::from_le_bytes(self.carry),
                    _ => return Err(invalid("unsupported dense stream precision")),
                };
                if !value.is_finite() {
                    return Err(invalid("nonfinite dense stream value"));
                }
                self.output[self.values] = value;
                self.values += 1;
                self.filled = 0;
            }
        }
        Ok(())
    }
    /// Require the exact complete finite payload. Does not verify source identity.
    pub fn finish(self) -> Result<(), TritModelError> {
        if self.poisoned
            || self.bytes != self.expected
            || self.filled != 0
            || self.values != self.output.len()
        {
            return Err(invalid("incomplete or poisoned dense stream"));
        }
        Ok(())
    }
}
