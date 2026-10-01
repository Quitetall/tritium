//! Shared runtime-output evidence domains used across execution and qualification.

use core::fmt;

const FINAL_LOGITS_CONTEXT: &str = "tritium qwen3.5 runtime final logits v1";
const BLOCK_OUTPUTS_CONTEXT: &str = "tritium qwen3.5 runtime block outputs v1";
const OUTPUT_RECONSTRUCTION_STUDENT_CONTEXT: &str =
    "tritium salt v2 output reconstruction student v1";
const MAX_FINAL_LOGIT_BATCHES: u64 = 1 << 20;
const MAX_BLOCK_OUTPUT_OBSERVATIONS: u64 = 1 << 24;

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
    /// Block output rows, columns, or value count are inconsistent.
    InvalidGeometry,
    /// A required content identity was zero.
    MissingIdentity,
    /// A row-selection mask selected no values.
    EmptySelection,
}

impl fmt::Display for RuntimeEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("runtime evidence batch is empty"),
            Self::NonFiniteOutput => formatter.write_str("runtime evidence output is non-finite"),
            Self::CountOverflow => formatter.write_str("runtime evidence count overflowed"),
            Self::EmptyStream => formatter.write_str("runtime evidence stream is empty"),
            Self::InvalidGeometry => {
                formatter.write_str("runtime block-output geometry is invalid")
            }
            Self::MissingIdentity => formatter.write_str("runtime evidence identity is missing"),
            Self::EmptySelection => formatter.write_str("runtime evidence selected no rows"),
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

/// Exact ordered block-output stream identity shared by runtime and reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeBlockOutputsEvidence {
    digest: [u8; 32],
    observation_count: u64,
    element_count: u64,
}

impl RuntimeBlockOutputsEvidence {
    /// Domain-separated digest of ordered block coordinates, geometry, and f32 values.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Number of observed block output matrices.
    #[must_use]
    pub const fn observation_count(self) -> u64 {
        self.observation_count
    }

    /// Total number of observed block output values.
    #[must_use]
    pub const fn element_count(self) -> u64 {
        self.element_count
    }
}

/// Streaming producer for ordered per-block runtime outputs.
#[derive(Clone, Debug)]
pub struct RuntimeBlockOutputsAccumulator {
    hasher: blake3::Hasher,
    observation_count: u64,
    element_count: u64,
}

impl RuntimeBlockOutputsAccumulator {
    /// Begin an empty ordered block-output stream.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new_derive_key(BLOCK_OUTPUTS_CONTEXT),
            observation_count: 0,
            element_count: 0,
        }
    }

    /// Observe one non-empty row-major output matrix in execution order.
    ///
    /// Coordinates bind batch, block, and token start, so regrouping or reordering
    /// observations changes the digest even when the flattened values are equal.
    ///
    /// # Errors
    /// Rejects empty/inconsistent geometry, non-finite values, and counter overflow.
    pub fn observe(
        &mut self,
        batch_index: u64,
        block_index: u32,
        token_start: u64,
        rows: usize,
        columns: usize,
        values: &[f32],
    ) -> Result<(), RuntimeEvidenceError> {
        let expected = rows
            .checked_mul(columns)
            .ok_or(RuntimeEvidenceError::InvalidGeometry)?;
        if rows == 0 || columns == 0 || values.len() != expected {
            return Err(RuntimeEvidenceError::InvalidGeometry);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeEvidenceError::NonFiniteOutput);
        }
        if self.observation_count == MAX_BLOCK_OUTPUT_OBSERVATIONS {
            return Err(RuntimeEvidenceError::CountOverflow);
        }
        let row_count = u64::try_from(rows).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let column_count =
            u64::try_from(columns).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let value_count =
            u64::try_from(values.len()).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let next_elements = self
            .element_count
            .checked_add(value_count)
            .ok_or(RuntimeEvidenceError::CountOverflow)?;
        self.hasher.update(&batch_index.to_le_bytes());
        self.hasher.update(&block_index.to_le_bytes());
        self.hasher.update(&token_start.to_le_bytes());
        self.hasher.update(&row_count.to_le_bytes());
        self.hasher.update(&column_count.to_le_bytes());
        for value in values {
            self.hasher.update(&value.to_bits().to_le_bytes());
        }
        self.observation_count += 1;
        self.element_count = next_elements;
        Ok(())
    }

    /// Seal a non-empty stream after binding its final counters.
    ///
    /// # Errors
    /// Rejects an empty stream.
    pub fn finish(mut self) -> Result<RuntimeBlockOutputsEvidence, RuntimeEvidenceError> {
        if self.observation_count == 0 || self.element_count == 0 {
            return Err(RuntimeEvidenceError::EmptyStream);
        }
        self.hasher.update(&self.observation_count.to_le_bytes());
        self.hasher.update(&self.element_count.to_le_bytes());
        Ok(RuntimeBlockOutputsEvidence {
            digest: *self.hasher.finalize().as_bytes(),
            observation_count: self.observation_count,
            element_count: self.element_count,
        })
    }
}

impl Default for RuntimeBlockOutputsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Scope tag used by the candidate student-output stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeOutputScope {
    /// Inclusive/exclusive block range, including windows spanning multiple blocks.
    Block {
        /// First block included in the observed window.
        start: u32,
        /// First block not included in the observed window.
        end: u32,
    },
    /// Final language-model logits.
    FinalLogits,
}

/// Exact digest and counts for one candidate-shaped student-output stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeOutputReconstructionEvidence {
    digest: [u8; 32],
    observation_count: u64,
    value_count: u64,
}

impl RuntimeOutputReconstructionEvidence {
    /// Digest byte-compatible with the `TSV2OUT` v2 student stream.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Number of ordered scope/batch observations.
    #[must_use]
    pub const fn observation_count(self) -> u64 {
        self.observation_count
    }

    /// Number of f32 values included in the stream.
    #[must_use]
    pub const fn value_count(self) -> u64 {
        self.value_count
    }
}

/// Streaming producer for the canonical output-reconstruction student digest.
#[derive(Clone, Debug)]
pub struct RuntimeOutputReconstructionAccumulator {
    hasher: blake3::Hasher,
    observation_count: u64,
    value_count: u64,
}

impl RuntimeOutputReconstructionAccumulator {
    /// Start a stream bound to one reconstruction spec, candidate, and seed.
    ///
    /// # Errors
    /// Rejects zero spec or candidate identities.
    pub fn new(
        spec_id: &[u8; 32],
        candidate_id: &[u8; 32],
        initialization_seed: u64,
    ) -> Result<Self, RuntimeEvidenceError> {
        if spec_id == &[0; 32] || candidate_id == &[0; 32] {
            return Err(RuntimeEvidenceError::MissingIdentity);
        }
        let mut hasher = blake3::Hasher::new_derive_key(OUTPUT_RECONSTRUCTION_STUDENT_CONTEXT);
        hasher.update(spec_id);
        hasher.update(candidate_id);
        hasher.update(&initialization_seed.to_le_bytes());
        Ok(Self {
            hasher,
            observation_count: 0,
            value_count: 0,
        })
    }

    /// Observe one finite row-major output matrix in canonical scope/batch order.
    ///
    /// # Errors
    /// Rejects malformed geometry, an empty selection, non-finite values, or
    /// counter overflow. The hash layout exactly preserves the frozen v2 stream.
    pub fn observe(
        &mut self,
        scope: RuntimeOutputScope,
        batch_index: u32,
        rows: usize,
        columns: usize,
        mask: &[bool],
        values: &[f32],
    ) -> Result<(), RuntimeEvidenceError> {
        let expected_values = rows
            .checked_mul(columns)
            .ok_or(RuntimeEvidenceError::InvalidGeometry)?;
        if rows == 0 || columns == 0 || mask.len() != rows || values.len() != expected_values {
            return Err(RuntimeEvidenceError::InvalidGeometry);
        }
        if mask.iter().all(|selected| !selected) {
            return Err(RuntimeEvidenceError::EmptySelection);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeEvidenceError::NonFiniteOutput);
        }
        let row_count = u64::try_from(rows).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let column_count =
            u64::try_from(columns).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let value_count =
            u64::try_from(values.len()).map_err(|_| RuntimeEvidenceError::CountOverflow)?;
        let next_observations = self
            .observation_count
            .checked_add(1)
            .ok_or(RuntimeEvidenceError::CountOverflow)?;
        let next_values = self
            .value_count
            .checked_add(value_count)
            .ok_or(RuntimeEvidenceError::CountOverflow)?;

        match scope {
            RuntimeOutputScope::Block { start, end } if start < end => {
                self.hasher.update(&[1]);
                self.hasher.update(&start.to_le_bytes());
                self.hasher.update(&end.to_le_bytes());
            }
            RuntimeOutputScope::Block { .. } => {
                return Err(RuntimeEvidenceError::InvalidGeometry);
            }
            RuntimeOutputScope::FinalLogits if columns >= 2 => {
                self.hasher.update(&[2]);
            }
            RuntimeOutputScope::FinalLogits => {
                return Err(RuntimeEvidenceError::InvalidGeometry);
            }
        }
        self.hasher.update(&batch_index.to_le_bytes());
        self.hasher.update(&row_count.to_le_bytes());
        self.hasher.update(&column_count.to_le_bytes());
        for selected in mask {
            self.hasher.update(&[u8::from(*selected)]);
        }
        for value in values {
            self.hasher.update(&value.to_bits().to_le_bytes());
        }
        self.observation_count = next_observations;
        self.value_count = next_values;
        Ok(())
    }

    /// Seal a non-empty output stream.
    ///
    /// # Errors
    /// Rejects a stream with no observations.
    pub fn finish(self) -> Result<RuntimeOutputReconstructionEvidence, RuntimeEvidenceError> {
        if self.observation_count == 0 || self.value_count == 0 {
            return Err(RuntimeEvidenceError::EmptyStream);
        }
        Ok(RuntimeOutputReconstructionEvidence {
            digest: *self.hasher.finalize().as_bytes(),
            observation_count: self.observation_count,
            value_count: self.value_count,
        })
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

    #[test]
    fn block_output_evidence_binds_coordinates_geometry_order_and_values() {
        let mut first = RuntimeBlockOutputsAccumulator::new();
        first
            .observe(0, 1, 12, 2, 2, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        first
            .observe(0, 2, 12, 2, 2, &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let first = first.finish().unwrap();

        let mut reordered = RuntimeBlockOutputsAccumulator::new();
        reordered
            .observe(0, 2, 12, 2, 2, &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        reordered
            .observe(0, 1, 12, 2, 2, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let reordered = reordered.finish().unwrap();

        assert_ne!(first.digest(), reordered.digest());
        assert_eq!(first.observation_count(), 2);
        assert_eq!(first.element_count(), 8);
    }

    #[test]
    fn block_output_evidence_rejects_invalid_observations_and_empty_seal() {
        let mut accumulator = RuntimeBlockOutputsAccumulator::new();
        assert_eq!(
            accumulator.observe(0, 0, 0, 0, 2, &[]),
            Err(RuntimeEvidenceError::InvalidGeometry)
        );
        assert_eq!(
            accumulator.observe(0, 0, 0, 1, 2, &[1.0]),
            Err(RuntimeEvidenceError::InvalidGeometry)
        );
        assert_eq!(
            accumulator.observe(0, 0, 0, 1, 2, &[1.0, f32::NAN]),
            Err(RuntimeEvidenceError::NonFiniteOutput)
        );
        assert_eq!(accumulator.finish(), Err(RuntimeEvidenceError::EmptyStream));
    }

    #[test]
    fn candidate_output_stream_binds_spec_candidate_seed_scopes_masks_and_values() {
        fn stream(mask: &[bool], value: f32) -> [u8; 32] {
            let mut accumulator =
                RuntimeOutputReconstructionAccumulator::new(&[1; 32], &[2; 32], 3).unwrap();
            accumulator
                .observe(
                    RuntimeOutputScope::Block { start: 4, end: 7 },
                    0,
                    2,
                    2,
                    mask,
                    &[value, 2.0, 3.0, 4.0],
                )
                .unwrap();
            *accumulator.finish().unwrap().digest()
        }

        let expected = stream(&[true, false], 1.0);
        assert_ne!(expected, stream(&[false, true], 1.0));
        assert_ne!(expected, stream(&[true, false], 9.0));

        let mut other_seed =
            RuntimeOutputReconstructionAccumulator::new(&[1; 32], &[2; 32], 4).unwrap();
        other_seed
            .observe(
                RuntimeOutputScope::Block { start: 4, end: 7 },
                0,
                2,
                2,
                &[true, false],
                &[1.0, 2.0, 3.0, 4.0],
            )
            .unwrap();
        assert_ne!(expected, *other_seed.finish().unwrap().digest());
    }

    #[test]
    fn candidate_output_stream_fails_closed_on_invalid_scope_or_empty_selection() {
        assert_eq!(
            RuntimeOutputReconstructionAccumulator::new(&[0; 32], &[2; 32], 0).err(),
            Some(RuntimeEvidenceError::MissingIdentity)
        );
        let mut accumulator =
            RuntimeOutputReconstructionAccumulator::new(&[1; 32], &[2; 32], 3).unwrap();
        assert_eq!(
            accumulator.observe(
                RuntimeOutputScope::Block { start: 2, end: 2 },
                0,
                1,
                2,
                &[true],
                &[1.0, 2.0],
            ),
            Err(RuntimeEvidenceError::InvalidGeometry)
        );
        assert_eq!(
            accumulator.observe(
                RuntimeOutputScope::FinalLogits,
                0,
                1,
                2,
                &[false],
                &[1.0, 2.0],
            ),
            Err(RuntimeEvidenceError::EmptySelection)
        );
    }
}
