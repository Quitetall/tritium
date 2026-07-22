//! Canonical TSV2OUT receipt encoding and strict reopening.

use super::{
    CANDIDATE_HASH_CONTEXT, LegacyOutputReconstructionReceipt, OutputCandidateReceipt,
    OutputReconstructionError, OutputReconstructionReceipt, OutputReconstructionSpec,
    RECEIPT_HASH_CONTEXT, select_output_reconstruction,
};
use std::collections::BTreeSet;

const RECEIPT_MAGIC: [u8; 8] = *b"TSV2OUT\0";
const LEGACY_RECEIPT_VERSION: u16 = 1;
const RECEIPT_VERSION: u16 = 2;
const MAX_RECEIPT_BYTES: usize = 4 * 1024 * 1024;
const FIXED_RECEIPT_BYTES: usize = 8 + 2 + 2 + 32 + 32 + 32 + 4 + 32;
const LEGACY_CANDIDATE_BYTES: usize = 224;
const CANDIDATE_BYTES: usize = 272;
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
        if bytes.get(8..10) == Some(&LEGACY_RECEIPT_VERSION.to_le_bytes()) {
            Self::validate_legacy_v1_canonical_bytes(spec, bytes)?;
            return Err(OutputReconstructionError::LegacyReceiptMissingRuntimeEvidence);
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

    /// Strictly validate and inspect a legacy `TSV2OUT` v1 receipt.
    ///
    /// Legacy evidence remains audit-visible but is never promoted into a v2
    /// receipt because v1 did not bind runtime-comparable final-logit evidence.
    ///
    /// # Errors
    /// Rejects oversized, truncated, noncanonical, corrupted, spec-mismatched,
    /// incomplete, duplicated, or incorrectly selected legacy bytes.
    pub fn validate_legacy_v1_canonical_bytes(
        spec: &OutputReconstructionSpec,
        bytes: &[u8],
    ) -> Result<LegacyOutputReconstructionReceipt, OutputReconstructionError> {
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(OutputReconstructionError::ReceiptTooLarge);
        }
        let mut cursor = OutputReceiptCursor::new(bytes);
        if cursor.take(8)? != RECEIPT_MAGIC {
            return Err(OutputReconstructionError::MalformedReceipt("magic"));
        }
        if cursor.u16()? != LEGACY_RECEIPT_VERSION {
            return Err(OutputReconstructionError::MalformedReceipt(
                "legacy version",
            ));
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
        let candidate_count = cursor.u32()?;
        let count = usize::try_from(candidate_count)
            .map_err(|_| OutputReconstructionError::ReceiptTooLarge)?;
        if count != spec.restarts {
            return Err(OutputReconstructionError::RestartCount {
                expected: spec.restarts,
                got: count,
            });
        }
        let max_legacy_candidates =
            (MAX_RECEIPT_BYTES - FIXED_RECEIPT_BYTES) / LEGACY_CANDIDATE_BYTES;
        if count > max_legacy_candidates {
            return Err(OutputReconstructionError::ReceiptTooLarge);
        }
        let expected_bytes = count
            .checked_mul(LEGACY_CANDIDATE_BYTES)
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
            candidates.push(decode_legacy_candidate(spec, &mut cursor)?);
        }
        let receipt_id = cursor.digest()?;
        if !cursor.is_empty() {
            return Err(OutputReconstructionError::MalformedReceipt(
                "trailing bytes",
            ));
        }
        verify_legacy_selection(
            spec,
            teacher_evidence_digest,
            selected_candidate_id,
            receipt_id,
            &candidates,
        )?;
        Ok(LegacyOutputReconstructionReceipt {
            spec_id,
            teacher_evidence_digest,
            selected_candidate_id,
            candidate_count,
            receipt_id,
        })
    }
}

fn encode_candidate(output: &mut Vec<u8>, candidate: &OutputCandidateReceipt) {
    output.extend_from_slice(&candidate.spec_id);
    output.extend_from_slice(&candidate.candidate_id);
    output.extend_from_slice(&candidate.initialization_seed.to_le_bytes());
    output.extend_from_slice(&candidate.teacher_evidence_digest);
    output.extend_from_slice(&candidate.student_output_digest);
    output.extend_from_slice(&candidate.runtime_final_logits_digest);
    output.extend_from_slice(&candidate.runtime_batch_count.to_le_bytes());
    output.extend_from_slice(&candidate.runtime_logit_count.to_le_bytes());
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
        runtime_final_logits_digest: cursor.digest()?,
        runtime_batch_count: cursor.u64()?,
        runtime_logit_count: cursor.u64()?,
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
        || candidate.runtime_final_logits_digest == [0; 32]
        || candidate.runtime_batch_count == 0
        || candidate.runtime_batch_count != candidate.final_tokens
        || candidate.runtime_logit_count < candidate.runtime_batch_count
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

#[derive(Clone, Debug)]
struct LegacyCandidateReceipt {
    spec_id: [u8; 32],
    candidate_id: [u8; 32],
    initialization_seed: u64,
    teacher_evidence_digest: [u8; 32],
    student_output_digest: [u8; 32],
    observations: u64,
    block_elements: u64,
    final_tokens: u64,
    block_output_mse: f64,
    teacher_cross_entropy: f64,
    teacher_kl: f64,
    objective: f64,
    receipt_id: [u8; 32],
}

impl LegacyCandidateReceipt {
    fn derive_id(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
        hasher.update(&self.spec_id);
        hasher.update(&self.candidate_id);
        hasher.update(&self.initialization_seed.to_le_bytes());
        hasher.update(&self.teacher_evidence_digest);
        hasher.update(&self.student_output_digest);
        hasher.update(&self.observations.to_le_bytes());
        hasher.update(&self.block_elements.to_le_bytes());
        hasher.update(&self.final_tokens.to_le_bytes());
        for value in [
            self.block_output_mse,
            self.teacher_cross_entropy,
            self.teacher_kl,
            self.objective,
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

fn decode_legacy_candidate(
    spec: &OutputReconstructionSpec,
    cursor: &mut OutputReceiptCursor<'_>,
) -> Result<LegacyCandidateReceipt, OutputReconstructionError> {
    let candidate = LegacyCandidateReceipt {
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
    if candidate.spec_id != *spec.spec_id()
        || candidate.candidate_id == [0; 32]
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
        || candidate.observations
            != spec
                .expected_observations()
                .map_err(|_| OutputReconstructionError::MalformedReceipt("candidate"))?
        || candidate.objective.to_bits()
            != spec
                .objective_for(
                    candidate.block_output_mse,
                    candidate.teacher_cross_entropy,
                    candidate.teacher_kl,
                )
                .to_bits()
    {
        return Err(OutputReconstructionError::MalformedReceipt(
            "legacy candidate",
        ));
    }
    Ok(candidate)
}

fn verify_legacy_selection(
    spec: &OutputReconstructionSpec,
    teacher_evidence_digest: [u8; 32],
    selected_candidate_id: [u8; 32],
    receipt_id: [u8; 32],
    candidates: &[LegacyCandidateReceipt],
) -> Result<(), OutputReconstructionError> {
    let mut ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for candidate in candidates.iter() {
        if candidate.spec_id != *spec.spec_id()
            || candidate.teacher_evidence_digest != teacher_evidence_digest
            || !ids.insert(candidate.candidate_id)
            || !seeds.insert(candidate.initialization_seed)
        {
            return Err(OutputReconstructionError::MalformedReceipt(
                "legacy candidate set",
            ));
        }
    }
    if candidates
        .windows(2)
        .any(|pair| pair[0].candidate_id >= pair[1].candidate_id)
    {
        return Err(OutputReconstructionError::MalformedReceipt(
            "legacy candidate order",
        ));
    }
    let expected_selected = candidates
        .iter()
        .min_by(|left, right| {
            left.objective
                .total_cmp(&right.objective)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .map(|candidate| candidate.candidate_id)
        .ok_or(OutputReconstructionError::InvalidCount)?;
    let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    hasher.update(spec.spec_id());
    hasher.update(&teacher_evidence_digest);
    hasher.update(
        &u64::try_from(candidates.len())
            .map_err(|_| OutputReconstructionError::ReceiptTooLarge)?
            .to_le_bytes(),
    );
    for candidate in candidates {
        hasher.update(&candidate.receipt_id);
    }
    hasher.update(&expected_selected);
    if selected_candidate_id != expected_selected || receipt_id != *hasher.finalize().as_bytes() {
        return Err(OutputReconstructionError::MalformedReceipt(
            "legacy identity",
        ));
    }
    Ok(())
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
