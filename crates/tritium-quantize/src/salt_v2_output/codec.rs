//! Canonical TSV2OUT receipt encoding and strict reopening.

use super::{
    OutputCandidateReceipt, OutputReconstructionError, OutputReconstructionReceipt,
    OutputReconstructionSpec, select_output_reconstruction,
};

const RECEIPT_MAGIC: [u8; 8] = *b"TSV2OUT\0";
const RECEIPT_VERSION: u16 = 1;
const MAX_RECEIPT_BYTES: usize = 4 * 1024 * 1024;
const FIXED_RECEIPT_BYTES: usize = 8 + 2 + 2 + 32 + 32 + 32 + 4 + 32;
const CANDIDATE_BYTES: usize = 224;
const MAX_CANDIDATES: usize = (MAX_RECEIPT_BYTES - FIXED_RECEIPT_BYTES) / CANDIDATE_BYTES;

impl OutputReconstructionReceipt {
    /// Encode the complete matched-basin receipt in canonical binary form.
    ///
    /// # Errors
    /// Rejects an unrepresentable candidate count or allocation failure.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, OutputReconstructionError> {
        let count = u32::try_from(self.candidates.len())
            .map_err(|_| OutputReconstructionError::ReceiptTooLarge)?;
        let candidate_bytes = self
            .candidates
            .len()
            .checked_mul(CANDIDATE_BYTES)
            .ok_or(OutputReconstructionError::ReceiptTooLarge)?;
        let capacity = FIXED_RECEIPT_BYTES
            .checked_add(candidate_bytes)
            .ok_or(OutputReconstructionError::ReceiptTooLarge)?;
        if capacity > MAX_RECEIPT_BYTES {
            return Err(OutputReconstructionError::ReceiptTooLarge);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(capacity)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        output.extend_from_slice(&RECEIPT_MAGIC);
        output.extend_from_slice(&RECEIPT_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&self.spec_id);
        output.extend_from_slice(&self.teacher_evidence_digest);
        output.extend_from_slice(&self.selected_candidate_id);
        output.extend_from_slice(&count.to_le_bytes());
        for candidate in &self.candidates {
            encode_candidate(&mut output, candidate);
        }
        output.extend_from_slice(&self.receipt_id);
        debug_assert_eq!(output.len(), capacity);
        Ok(output)
    }

    /// Strictly reopen a canonical receipt against its frozen specification.
    ///
    /// # Errors
    /// Rejects oversized, truncated, noncanonical, corrupted, or spec-mismatched bytes.
    pub fn from_canonical_bytes(
        spec: &OutputReconstructionSpec,
        bytes: &[u8],
    ) -> Result<Self, OutputReconstructionError> {
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(OutputReconstructionError::ReceiptTooLarge);
        }
        let mut cursor = OutputReceiptCursor::new(bytes);
        if cursor.take(8)? != RECEIPT_MAGIC {
            return Err(OutputReconstructionError::MalformedReceipt("magic"));
        }
        if cursor.u16()? != RECEIPT_VERSION {
            return Err(OutputReconstructionError::MalformedReceipt("version"));
        }
        if cursor.u16()? != 0 {
            return Err(OutputReconstructionError::MalformedReceipt("flags"));
        }
        let spec_id = cursor.digest()?;
        if &spec_id != spec.spec_id() {
            return Err(OutputReconstructionError::MalformedReceipt("spec identity"));
        }
        let teacher_evidence_digest = cursor.digest()?;
        let selected_candidate_id = cursor.digest()?;
        let count = usize::try_from(cursor.u32()?)
            .map_err(|_| OutputReconstructionError::ReceiptTooLarge)?;
        if count != spec.restarts {
            return Err(OutputReconstructionError::RestartCount {
                expected: spec.restarts,
                got: count,
            });
        }
        if count > MAX_CANDIDATES {
            return Err(OutputReconstructionError::ReceiptTooLarge);
        }
        let expected_bytes = count
            .checked_mul(CANDIDATE_BYTES)
            .and_then(|candidate_bytes| FIXED_RECEIPT_BYTES.checked_add(candidate_bytes))
            .ok_or(OutputReconstructionError::ReceiptTooLarge)?;
        if bytes.len() != expected_bytes {
            return Err(OutputReconstructionError::MalformedReceipt("length"));
        }
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(count)
            .map_err(|_| OutputReconstructionError::ReceiptAllocationFailed)?;
        for _ in 0..count {
            candidates.push(decode_candidate(spec, &mut cursor)?);
        }
        let receipt_id = cursor.digest()?;
        if !cursor.is_empty() {
            return Err(OutputReconstructionError::MalformedReceipt(
                "trailing bytes",
            ));
        }
        let declared = Self {
            spec_id,
            teacher_evidence_digest,
            candidates: candidates.clone(),
            selected_candidate_id,
            receipt_id,
        };
        let verified = select_output_reconstruction(spec, candidates)?;
        if declared != verified
            || declared
                .canonical_bytes()
                .map_err(|_| OutputReconstructionError::MalformedReceipt("canonical encoding"))?
                != bytes
        {
            return Err(OutputReconstructionError::MalformedReceipt("identity"));
        }
        Ok(verified)
    }
}

fn encode_candidate(output: &mut Vec<u8>, candidate: &OutputCandidateReceipt) {
    output.extend_from_slice(&candidate.spec_id);
    output.extend_from_slice(&candidate.candidate_id);
    output.extend_from_slice(&candidate.initialization_seed.to_le_bytes());
    output.extend_from_slice(&candidate.teacher_evidence_digest);
    output.extend_from_slice(&candidate.student_output_digest);
    output.extend_from_slice(&candidate.observations.to_le_bytes());
    output.extend_from_slice(&candidate.block_elements.to_le_bytes());
    output.extend_from_slice(&candidate.final_tokens.to_le_bytes());
    for value in [
        candidate.block_output_mse,
        candidate.teacher_cross_entropy,
        candidate.teacher_kl,
        candidate.objective,
    ] {
        output.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    output.extend_from_slice(&candidate.receipt_id);
}

fn decode_candidate(
    spec: &OutputReconstructionSpec,
    cursor: &mut OutputReceiptCursor<'_>,
) -> Result<OutputCandidateReceipt, OutputReconstructionError> {
    let candidate = OutputCandidateReceipt {
        spec_id: cursor.digest()?,
        candidate_id: cursor.digest()?,
        initialization_seed: cursor.u64()?,
        teacher_evidence_digest: cursor.digest()?,
        student_output_digest: cursor.digest()?,
        observations: cursor.u64()?,
        block_elements: cursor.u64()?,
        final_tokens: cursor.u64()?,
        block_output_mse: cursor.f64()?,
        teacher_cross_entropy: cursor.f64()?,
        teacher_kl: cursor.f64()?,
        objective: cursor.f64()?,
        receipt_id: cursor.digest()?,
    };
    if candidate.candidate_id == [0; 32]
        || candidate.teacher_evidence_digest == [0; 32]
        || candidate.student_output_digest == [0; 32]
        || candidate.observations == 0
        || candidate.block_elements == 0
        || candidate.final_tokens == 0
        || [
            candidate.block_output_mse,
            candidate.teacher_cross_entropy,
            candidate.teacher_kl,
            candidate.objective,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || value.to_bits() == (-0.0f64).to_bits())
        || candidate.receipt_id != candidate.derive_id()
    {
        return Err(OutputReconstructionError::MalformedReceipt("candidate"));
    }
    if candidate.observations
        != spec
            .expected_observations()
            .map_err(|_| OutputReconstructionError::MalformedReceipt("candidate observations"))?
    {
        return Err(OutputReconstructionError::MalformedReceipt(
            "candidate observations",
        ));
    }
    let expected_objective = spec.objective_for(
        candidate.block_output_mse,
        candidate.teacher_cross_entropy,
        candidate.teacher_kl,
    );
    if !expected_objective.is_finite()
        || candidate.objective.to_bits() != expected_objective.to_bits()
    {
        return Err(OutputReconstructionError::MalformedReceipt(
            "candidate objective",
        ));
    }
    Ok(candidate)
}

struct OutputReceiptCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> OutputReceiptCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], OutputReconstructionError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(OutputReconstructionError::MalformedReceipt("length"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(OutputReconstructionError::MalformedReceipt("truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, OutputReconstructionError> {
        let bytes: [u8; 2] = self
            .take(2)?
            .try_into()
            .map_err(|_| OutputReconstructionError::MalformedReceipt("u16"))?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, OutputReconstructionError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| OutputReconstructionError::MalformedReceipt("u32"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, OutputReconstructionError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| OutputReconstructionError::MalformedReceipt("u64"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, OutputReconstructionError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn digest(&mut self) -> Result<[u8; 32], OutputReconstructionError> {
        self.take(32)?
            .try_into()
            .map_err(|_| OutputReconstructionError::MalformedReceipt("digest"))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
