//! Bounded receiving of the existing canonical Gram bytes, not capture authority.
use super::*;

/// Canonical data receiving refusal; not a numerical fit or provenance verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputGramDecodeError {
    /// Unsupported/noncanonical/truncated/trailing fields or inconsistent counts.
    Encoding,
    /// Zero/insufficient positive decoded-value budget or platform overflow.
    Limit,
    /// The caller-selected expected identity is absent or differs from actual bytes.
    Identity,
    /// A bounded allocation could not be satisfied.
    Allocation,
}
impl core::fmt::Display for InputGramDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "canonical input Gram refused: {self:?}")
    }
}
impl std::error::Error for InputGramDecodeError {}

struct Decoder<'a>(&'a [u8]);
impl Decoder<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], InputGramDecodeError> {
        let value = self.0.get(..N).ok_or(InputGramDecodeError::Encoding)?;
        let result = value
            .try_into()
            .map_err(|_| InputGramDecodeError::Encoding)?;
        self.0 = &self.0[N..];
        Ok(result)
    }
    fn u64(&mut self) -> Result<u64, InputGramDecodeError> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn scalar(&mut self) -> Result<f64, InputGramDecodeError> {
        let bits = self.u64()?;
        let value = f64::from_bits(bits);
        if !value.is_finite() || bits != canonical_f64_bits(value) {
            return Err(InputGramDecodeError::Encoding);
        }
        Ok(value)
    }
}
impl InputGram {
    /// Receive the existing SIG2/version2 bytes with exact caller-selected digest.
    /// The positive value-capacity ceiling excludes the borrowed encoded input,
    /// metadata, allocator/process RSS and any subsequent solver workspace.
    /// Preserves source/count/trace data; does not authenticate collection or prove
    /// PSD/SPD. Unbound diagnostic data remains unbound and cannot enter a solver.
    /// # Errors
    /// Noncanonical/malformed input, absent/mismatched expected digest, overflow,
    /// insufficient positive capacity or allocation refusal, before returning data.
    pub fn decode_canonical(
        bytes: &[u8],
        expected: [u8; 32],
        max_value_bytes: usize,
    ) -> Result<Self, InputGramDecodeError> {
        if max_value_bytes == 0 || bytes.len() > max_value_bytes.saturating_add(206) {
            return Err(InputGramDecodeError::Limit);
        }
        let mut d = Decoder(bytes);
        if d.take::<4>()? != INPUT_GRAM_MAGIC || d.take::<1>()? != [CANONICAL_VERSION] {
            return Err(InputGramDecodeError::Encoding);
        }
        let dimension = usize::try_from(d.u64()?).map_err(|_| InputGramDecodeError::Limit)?;
        let sample_count = d.u64()?;
        let selected_count = d.u64()?;
        let total_weight = d.scalar()?;
        if dimension == 0
            || selected_count == 0
            || selected_count > sample_count
            || total_weight <= 0.
        {
            return Err(InputGramDecodeError::Encoding);
        }
        let source_id = match d.take::<1>()? {
            [0] => None,
            [1] => Some(
                CurvatureSourceId::new(d.take()?, d.take()?, d.take()?)
                    .map_err(|_| InputGramDecodeError::Encoding)?,
            ),
            _ => return Err(InputGramDecodeError::Encoding),
        };
        let selection_trace = d.take()?;
        let data_trace = d.take()?;
        let entries = usize::try_from(d.u64()?).map_err(|_| InputGramDecodeError::Limit)?;
        let required = entries.checked_mul(8).ok_or(InputGramDecodeError::Limit)?;
        if required > max_value_bytes {
            return Err(InputGramDecodeError::Limit);
        }
        if dimension.checked_mul(dimension) != Some(entries) || d.0.len() != required {
            return Err(InputGramDecodeError::Encoding);
        }
        if expected == [0; 32] || domain_hash(INPUT_GRAM_DIGEST_CONTEXT, bytes) != expected {
            return Err(InputGramDecodeError::Identity);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(entries)
            .map_err(|_| InputGramDecodeError::Allocation)?;
        if values
            .capacity()
            .checked_mul(8)
            .is_none_or(|n| n > max_value_bytes)
        {
            return Err(InputGramDecodeError::Limit);
        }
        for _ in 0..entries {
            values.push(d.scalar()?);
        }
        Ok(Self {
            dimension,
            values,
            total_weight,
            sample_count,
            selected_count,
            source_id,
            selection_trace,
            data_trace,
        })
    }
}
