//! Shared runtime-output evidence domains used across execution and qualification.

use core::fmt;

const FINAL_LOGITS_CONTEXT: &str = "tritium qwen3.5 runtime final logits v1";
const MAX_FINAL_LOGIT_BATCHES: u64 = 1 << 20;

/// Invalid runtime-output evidence stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeEvidenceError {
    /// One output batch was empty.
    EmptyBatch,
    /// One output value was not finite.
    NonFiniteOutput,
    /// A counter or supported batch bound overflowed.
    CountOverflow,
    /// No output batch was observed before sealing.
    EmptyStream,
}

impl fmt::Display for RuntimeEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("runtime evidence batch is empty"),
            Self::NonFiniteOutput => formatter.write_str("runtime evidence output is non-finite"),
            Self::CountOverflow => formatter.write_str("runtime evidence count overflowed"),
            Self::EmptyStream => formatter.write_str("runtime evidence stream is empty"),
        }
    }
}

impl std::error::Error for RuntimeEvidenceError {}

/// Exact final-logit stream identity shared by model execution and reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeFinalLogitsEvidence {
    digest: [u8; 32],
    batch_count: u64,
    logit_count: u64,
}

impl RuntimeFinalLogitsEvidence {
    /// Domain-separated digest of ordered batch boundaries and f32 logit bits.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Number of ordered final-logit batches.
    #[must_use]
    pub const fn batch_count(self) -> u64 {
        self.batch_count
    }

    /// Total final-logit values across every batch.
    #[must_use]
    pub const fn logit_count(self) -> u64 {
        self.logit_count
    }
}

/// Streaming producer for runtime-comparable final-logit evidence.
#[derive(Clone, Debug)]
pub struct RuntimeFinalLogitsAccumulator {
    hasher: blake3::Hasher,
    batch_count: u64,
    logit_count: u64,
}

impl RuntimeFinalLogitsAccumulator {
    /// Begin an empty ordered final-logit stream.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new_derive_key(FINAL_LOGITS_CONTEXT),
            batch_count: 0,
            logit_count: 0,
        }
    }

    /// Consume one non-empty finite final-logit batch in execution order.
    ///
    /// # Errors
    /// Rejects empty/non-finite batches, excessive batch count, or count overflow.
    pub fn observe(&mut self, logits: &[f32]) -> Result<(), RuntimeEvidenceError> {
        if logits.is_empty() {
            return Err(RuntimeEvidenceError::EmptyBatch);
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeEvidenceError::NonFiniteOutput);
        }
        if self.batch_count == MAX_FINAL_LOGIT_BATCHES {
            return Err(RuntimeEvidenceError::CountOverflow);
        }
        let logit_count =
            u64::try_from(logits.len()).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        self.hasher.update(&self.batch_count.to_le_bytes());
        self.hasher.update(&logit_count.to_le_bytes());
        for logit in logits {
            self.hasher.update(&logit.to_bits().to_le_bytes());
        }
        self.batch_count = self
            .batch_count
            .checked_add(1)
            .ok_or(RuntimeEvidenceError::CountOverflow)?;
        self.logit_count = self
            .logit_count
            .checked_add(logit_count)
            .ok_or(RuntimeEvidenceError::CountOverflow)?;
        Ok(())
    }

    /// Seal a non-empty stream after binding its final counters.
    ///
    /// # Errors
    /// Rejects an empty stream.
    pub fn finish(mut self) -> Result<RuntimeFinalLogitsEvidence, RuntimeEvidenceError> {
        if self.batch_count == 0 || self.logit_count == 0 {
            return Err(RuntimeEvidenceError::EmptyStream);
        }
        self.hasher.update(&self.batch_count.to_le_bytes());
        self.hasher.update(&self.logit_count.to_le_bytes());
        Ok(RuntimeFinalLogitsEvidence {
            digest: *self.hasher.finalize().as_bytes(),
            batch_count: self.batch_count,
            logit_count: self.logit_count,
        })
    }
}

impl Default for RuntimeFinalLogitsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_logit_evidence_binds_order_boundaries_values_and_counts() {
        let mut first = RuntimeFinalLogitsAccumulator::new();
        first.observe(&[1.0, 2.0]).unwrap();
        first.observe(&[3.0]).unwrap();
        let first = first.finish().unwrap();
        let mut regrouped = RuntimeFinalLogitsAccumulator::new();
        regrouped.observe(&[1.0]).unwrap();
        regrouped.observe(&[2.0, 3.0]).unwrap();
        let regrouped = regrouped.finish().unwrap();

        assert_ne!(first.digest(), regrouped.digest());
        assert_eq!(first.batch_count(), 2);
        assert_eq!(first.logit_count(), 3);
        assert_eq!(
            first.digest(),
            &[
                26, 173, 88, 70, 85, 74, 249, 252, 234, 249, 82, 145, 32, 254, 73, 184, 128, 16,
                246, 106, 250, 30, 37, 205, 111, 48, 1, 250, 229, 76, 141, 107,
            ]
        );
    }
}
