//! Final-logit binding between canonical output reconstruction and sealed execution.

use core::{convert::Infallible, fmt};
use std::error::Error;

use tritium_nn::NnError;
use tritium_quantize::{
    OutputReconstructionError, OutputReconstructionReceipt, OutputReconstructionSpec, SaltV2Profile,
};

use crate::{ContentId, Qwen36PreservedSafetensorsError};

use super::{PRESERVED_CHUNK_BYTES, Qwen36AdmittedExecutionReceipt, execution_authority};
use crate::Qwen36PackageAdmittedCampaignStore;

const BINDING_MAGIC: [u8; 8] = *b"TSQ36OB\0";
const BINDING_VERSION: u16 = 1;
const FINAL_LOGITS_COVERAGE: u8 = 1;
const BINDING_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 output execution binding checksum v1";
const CANDIDATE_ID_CONTEXT: &str = "tritium qwen3.6 admitted output candidate v1";
const BOUND_DIGESTS: usize = 13;
const BINDING_BYTES: usize = 8 + 2 + 1 + 1 + 4 + BOUND_DIGESTS * 32 + 2 * 8 + 32;
const BINDING_BODY_BYTES: usize = BINDING_BYTES - 32;

/// Failure while binding one selected output candidate to sealed runtime evidence.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36FinalLogitsOutputBindingError {
    /// Package admission or its authoritative campaign lineage changed.
    Admission(super::super::Qwen36PackageAdmissionError),
    /// Preserved-source reconstruction failed.
    Workspace(crate::Qwen36TensorWorkError),
    /// Canonical `TSV2OUT` v2 bytes failed strict reopen.
    Output(OutputReconstructionError),
    /// Execution evidence, candidate identity, source, tokens, outputs, or counts differ.
    Runtime(NnError),
}

impl fmt::Display for Qwen36FinalLogitsOutputBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "bind Qwen output execution: {error}"),
            Self::Workspace(error) => write!(formatter, "bind Qwen output execution: {error}"),
            Self::Output(error) => write!(formatter, "bind Qwen output execution: {error}"),
            Self::Runtime(error) => write!(formatter, "bind Qwen output execution: {error}"),
        }
    }
}

impl Error for Qwen36FinalLogitsOutputBindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Workspace(error) => Some(error),
            Self::Output(error) => Some(error),
            Self::Runtime(error) => Some(error),
        }
    }
}

/// Campaign-bound proof that selected v2 final logits equal sealed execution.
///
/// This receipt deliberately does not attest block/window outputs. They remain
/// reconstruction evidence until a sealed runtime exposes matching block scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36FinalLogitsOutputBindingReceipt {
    binding_id: ContentId,
    completion_id: ContentId,
    campaign_id: ContentId,
    admission_id: ContentId,
    selection_id: ContentId,
    source_model_id: [u8; 32],
    master_set_id: [u8; 32],
    package_id: [u8; 32],
    preserved_package_id: [u8; 32],
    output_spec_id: [u8; 32],
    output_receipt_id: [u8; 32],
    selected_candidate_id: [u8; 32],
    execution_receipt_id: [u8; 32],
    final_logits_digest: [u8; 32],
    profile: SaltV2Profile,
    scope_coverage: u8,
    batch_count: u64,
    logit_count: u64,
}

impl Qwen36FinalLogitsOutputBindingReceipt {
    /// Content identity of exact canonical binding bytes.
    #[must_use]
    pub const fn binding_id(&self) -> ContentId {
        self.binding_id
    }

    /// Exact immutable master completion executed by the selected candidate.
    #[must_use]
    pub const fn completion_id(&self) -> ContentId {
        self.completion_id
    }

    /// Exact additive campaign whose selected package was executed.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Exact package-admission identity validated at binding time.
    #[must_use]
    pub const fn package_admission_id(&self) -> ContentId {
        self.admission_id
    }

    /// Exact nested-allocation selection whose package was executed.
    #[must_use]
    pub const fn selection_id(&self) -> ContentId {
        self.selection_id
    }

    /// Source-model identity shared by reconstruction and execution.
    #[must_use]
    pub const fn source_model_id(&self) -> &[u8; 32] {
        &self.source_model_id
    }

    /// Aggregate identity of every ordered tensor master.
    #[must_use]
    pub const fn master_set_id(&self) -> &[u8; 32] {
        &self.master_set_id
    }

    /// Exact selected SALT V2 package identity.
    #[must_use]
    pub const fn package_id(&self) -> &[u8; 32] {
        &self.package_id
    }

    /// Exact preserved source-precision companion identity.
    #[must_use]
    pub const fn preserved_package_id(&self) -> &[u8; 32] {
        &self.preserved_package_id
    }

    /// Exact output-reconstruction specification identity.
    #[must_use]
    pub const fn output_spec_id(&self) -> &[u8; 32] {
        &self.output_spec_id
    }

    /// Exact canonical `TSV2OUT` v2 receipt identity.
    #[must_use]
    pub const fn output_receipt_id(&self) -> &[u8; 32] {
        &self.output_receipt_id
    }

    /// Selected candidate derived from immutable package/campaign lineage.
    #[must_use]
    pub const fn selected_candidate_id(&self) -> &[u8; 32] {
        &self.selected_candidate_id
    }

    /// Exact sealed `TSQ36EX` execution receipt identity.
    #[must_use]
    pub const fn execution_receipt_id(&self) -> &[u8; 32] {
        &self.execution_receipt_id
    }

    /// Selected SALT V2 profile whose package was executed.
    #[must_use]
    pub const fn profile(&self) -> SaltV2Profile {
        self.profile
    }

    /// Whether final logits are exactly runtime-bound.
    #[must_use]
    pub const fn has_final_logits(&self) -> bool {
        self.scope_coverage & FINAL_LOGITS_COVERAGE != 0
    }

    /// Block/window outputs are not attested by this receipt version.
    #[must_use]
    pub const fn has_block_outputs(&self) -> bool {
        false
    }

    /// Exact runtime-comparable final-logit digest.
    #[must_use]
    pub const fn final_logits_digest(&self) -> &[u8; 32] {
        &self.final_logits_digest
    }

    /// Exact final-logit batch count.
    #[must_use]
    pub const fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Exact final-logit value count.
    #[must_use]
    pub const fn logit_count(&self) -> u64 {
        self.logit_count
    }

    /// Encode canonical `TSQ36OB` version-1 binding evidence.
    ///
    /// # Errors
    /// Returns [`NnError`] if the fixed-size receipt allocation cannot be reserved.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NnError> {
        let mut output = Vec::new();
        output.try_reserve_exact(BINDING_BYTES).map_err(|_| {
            NnError::ResourceExhausted("allocate Qwen output execution binding".to_owned())
        })?;
        output.extend_from_slice(&BINDING_MAGIC);
        output.extend_from_slice(&BINDING_VERSION.to_le_bytes());
        output.push(profile_tag(self.profile));
        output.push(self.scope_coverage);
        output.extend_from_slice(&[0; 4]);
        for digest in self.bound_digests() {
            output.extend_from_slice(&digest);
        }
        output.extend_from_slice(&self.batch_count.to_le_bytes());
        output.extend_from_slice(&self.logit_count.to_le_bytes());
        let mut checksum = blake3::Hasher::new_derive_key(BINDING_CHECKSUM_CONTEXT);
        checksum.update(&output);
        output.extend_from_slice(checksum.finalize().as_bytes());
        debug_assert_eq!(output.len(), BINDING_BYTES);
        Ok(output)
    }

    fn bound_digests(&self) -> [[u8; 32]; BOUND_DIGESTS] {
        [
            *self.completion_id.as_bytes(),
            *self.campaign_id.as_bytes(),
            *self.admission_id.as_bytes(),
            *self.selection_id.as_bytes(),
            self.source_model_id,
            self.master_set_id,
            self.package_id,
            self.preserved_package_id,
            self.output_spec_id,
            self.output_receipt_id,
            self.selected_candidate_id,
            self.execution_receipt_id,
            self.final_logits_digest,
        ]
    }
}

impl Qwen36AdmittedExecutionReceipt {
    /// Derive the only candidate label admissible for this exact package lineage.
    ///
    /// # Errors
    /// Rejects a reconstruction specification for another source model.
    pub fn output_candidate_id(
        &self,
        spec: &OutputReconstructionSpec,
    ) -> Result<[u8; 32], NnError> {
        if spec.source_model_id() != self.source_model_id {
            return Err(NnError::Provenance(
                "output reconstruction source differs from sealed execution".to_owned(),
            ));
        }
        let mut hasher = blake3::Hasher::new_derive_key(CANDIDATE_ID_CONTEXT);
        for digest in [
            *self.completion_id.as_bytes(),
            *self.campaign_id.as_bytes(),
            *self.admission_id.as_bytes(),
            *self.selection_id.as_bytes(),
            *self.source_model_id.as_bytes(),
            self.master_set_id,
            *self.package_id.as_bytes(),
            *self.preserved_package_id.as_bytes(),
            *spec.spec_id(),
        ] {
            hasher.update(&digest);
        }
        hasher.update(&[profile_tag(self.profile)]);
        Ok(*hasher.finalize().as_bytes())
    }
}

impl<'allocated, 'parent, 'store, 'source>
    Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>
{
    /// Bind selected v2 final logits to exact sealed execution and master lineage.
    ///
    /// The output receipt is reopened from canonical bytes. Candidate labels,
    /// aggregate student digests, or matching metrics alone cannot satisfy this
    /// seam. Block-output coverage remains explicitly absent.
    ///
    /// # Errors
    /// Fails closed on changed admission/workspace state, legacy or malformed
    /// output bytes, source/token/candidate drift, stale execution evidence, or
    /// any final-logit digest/count mismatch.
    pub fn bind_output_reconstruction_final_logits(
        &self,
        spec: &OutputReconstructionSpec,
        output_bytes: &[u8],
        execution: &Qwen36AdmittedExecutionReceipt,
    ) -> Result<Qwen36FinalLogitsOutputBindingReceipt, Qwen36FinalLogitsOutputBindingError> {
        self.verify_current()
            .map_err(Qwen36FinalLogitsOutputBindingError::Admission)?;
        let preserved = self
            .allocated
            .parent
            .base
            .try_write_preserved_safetensors(PRESERVED_CHUNK_BYTES, |_| Ok::<_, Infallible>(()))
            .map_err(|error| match error {
                Qwen36PreservedSafetensorsError::Workspace(error) => {
                    Qwen36FinalLogitsOutputBindingError::Workspace(error)
                }
                Qwen36PreservedSafetensorsError::Sink(error) => match error {},
            })?;
        let authority = execution_authority(
            self,
            execution.profile,
            execution.backend,
            preserved.package_id(),
        );
        validate_execution(&authority, execution)
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        let output = OutputReconstructionReceipt::from_canonical_bytes(spec, output_bytes)
            .map_err(Qwen36FinalLogitsOutputBindingError::Output)?;
        let selected = output.selected();
        let expected_candidate = execution
            .output_candidate_id(spec)
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        if output.spec_id() != spec.spec_id()
            || selected.candidate_id() != &expected_candidate
            || spec.token_stream_digest() != execution.token_stream_digest()
            || selected.runtime_final_logits_digest() != execution.final_logits_digest()
            || selected.runtime_batch_count() != execution.batch_count()
            || selected.runtime_logit_count() != execution.logit_count()
        {
            return Err(Qwen36FinalLogitsOutputBindingError::Runtime(
                NnError::Provenance(
                    "selected output reconstruction differs from sealed final-logit execution"
                        .to_owned(),
                ),
            ));
        }
        self.verify_current()
            .map_err(Qwen36FinalLogitsOutputBindingError::Admission)?;
        let mut receipt = Qwen36FinalLogitsOutputBindingReceipt {
            binding_id: ContentId::from_digest([0; 32]),
            completion_id: authority.completion_id,
            campaign_id: authority.campaign_id,
            admission_id: authority.admission_id,
            selection_id: authority.selection_id,
            source_model_id: *authority.source_model_id.as_bytes(),
            master_set_id: authority.master_set_id,
            package_id: *authority.package_id.as_bytes(),
            preserved_package_id: *authority.preserved_package_id.as_bytes(),
            output_spec_id: *spec.spec_id(),
            output_receipt_id: *output.receipt_id(),
            selected_candidate_id: expected_candidate,
            execution_receipt_id: *execution.receipt_id.as_bytes(),
            final_logits_digest: *execution.final_logits_digest(),
            profile: authority.profile,
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count: execution.batch_count(),
            logit_count: execution.logit_count(),
        };
        let canonical = receipt
            .canonical_bytes()
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        receipt.binding_id = ContentId::of_bytes(&canonical);
        Ok(receipt)
    }

    /// Strictly reopen persisted `TSQ36OB` bytes under current campaign authority.
    ///
    /// This decodes and canonicalizes the persisted record, then independently
    /// repeats output/execution binding against the live admission capability.
    /// A structurally valid historical record cannot reopen after its output,
    /// execution, or authoritative campaign state stops matching.
    ///
    /// # Errors
    /// Fails closed on malformed or noncanonical binding bytes and on every
    /// error reported by [`Self::bind_output_reconstruction_final_logits`].
    pub fn reopen_output_reconstruction_final_logits_binding(
        &self,
        spec: &OutputReconstructionSpec,
        output_bytes: &[u8],
        execution: &Qwen36AdmittedExecutionReceipt,
        binding_bytes: &[u8],
    ) -> Result<Qwen36FinalLogitsOutputBindingReceipt, Qwen36FinalLogitsOutputBindingError> {
        let declared =
            decode_binding(binding_bytes).map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        let rebound =
            self.bind_output_reconstruction_final_logits(spec, output_bytes, execution)?;
        let canonical = rebound
            .canonical_bytes()
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        if declared != rebound || canonical.as_slice() != binding_bytes {
            return Err(Qwen36FinalLogitsOutputBindingError::Runtime(
                NnError::Provenance(
                    "persisted output binding differs from current campaign authority".to_owned(),
                ),
            ));
        }
        Ok(rebound)
    }
}

fn decode_binding(bytes: &[u8]) -> Result<Qwen36FinalLogitsOutputBindingReceipt, NnError> {
    if bytes.len() != BINDING_BYTES
        || bytes[..8] != BINDING_MAGIC
        || bytes[8..10] != BINDING_VERSION.to_le_bytes()
        || bytes[11] != FINAL_LOGITS_COVERAGE
        || bytes[12..16] != [0; 4]
    {
        return Err(NnError::InvalidArtifact(
            "malformed Qwen output execution binding header".to_owned(),
        ));
    }
    let profile = match bytes[10] {
        1 => SaltV2Profile::CompactV1,
        2 => SaltV2Profile::NearLosslessV1,
        _ => {
            return Err(NnError::InvalidArtifact(
                "unknown Qwen output execution binding profile".to_owned(),
            ));
        }
    };
    let mut checksum = blake3::Hasher::new_derive_key(BINDING_CHECKSUM_CONTEXT);
    checksum.update(&bytes[..BINDING_BODY_BYTES]);
    if bytes[BINDING_BODY_BYTES..] != checksum.finalize().as_bytes()[..] {
        return Err(NnError::InvalidArtifact(
            "Qwen output execution binding checksum mismatch".to_owned(),
        ));
    }
    let digests: [[u8; 32]; BOUND_DIGESTS] =
        core::array::from_fn(|ordinal| binding_digest(bytes, ordinal));
    let count_offset = 16 + BOUND_DIGESTS * 32;
    let batch_count = binding_u64(bytes, count_offset);
    let logit_count = binding_u64(bytes, count_offset + 8);
    if batch_count == 0 || logit_count == 0 {
        return Err(NnError::InvalidArtifact(
            "Qwen output execution binding has empty runtime coverage".to_owned(),
        ));
    }
    let receipt = Qwen36FinalLogitsOutputBindingReceipt {
        binding_id: ContentId::of_bytes(bytes),
        completion_id: ContentId::from_digest(digests[0]),
        campaign_id: ContentId::from_digest(digests[1]),
        admission_id: ContentId::from_digest(digests[2]),
        selection_id: ContentId::from_digest(digests[3]),
        source_model_id: digests[4],
        master_set_id: digests[5],
        package_id: digests[6],
        preserved_package_id: digests[7],
        output_spec_id: digests[8],
        output_receipt_id: digests[9],
        selected_candidate_id: digests[10],
        execution_receipt_id: digests[11],
        final_logits_digest: digests[12],
        profile,
        scope_coverage: FINAL_LOGITS_COVERAGE,
        batch_count,
        logit_count,
    };
    if receipt.canonical_bytes()?.as_slice() != bytes {
        return Err(NnError::InvalidArtifact(
            "noncanonical Qwen output execution binding".to_owned(),
        ));
    }
    Ok(receipt)
}

fn binding_digest(bytes: &[u8], ordinal: usize) -> [u8; 32] {
    let start = 16 + ordinal * 32;
    let mut digest = [0; 32];
    digest.copy_from_slice(&bytes[start..start + 32]);
    digest
}

fn binding_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut value = [0; 8];
    value.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(value)
}

fn validate_execution(
    authority: &super::ExecutionAuthority,
    execution: &Qwen36AdmittedExecutionReceipt,
) -> Result<(), NnError> {
    let canonical = execution.canonical_bytes()?;
    if execution.receipt_id != ContentId::of_bytes(&canonical)
        || execution.completion_id != authority.completion_id
        || execution.campaign_id != authority.campaign_id
        || execution.admission_id != authority.admission_id
        || execution.selection_id != authority.selection_id
        || execution.source_model_id != authority.source_model_id
        || execution.master_set_id != authority.master_set_id
        || execution.profile != authority.profile
        || execution.package_id != authority.package_id
        || execution.preserved_package_id != authority.preserved_package_id
        || execution.backend != authority.backend
        || !execution.has_final_logits()
        || execution.has_block_outputs()
        || execution.batch_count == 0
        || execution.logit_count == 0
    {
        return Err(NnError::Provenance(
            "execution receipt differs from current package admission authority".to_owned(),
        ));
    }
    Ok(())
}

const fn profile_tag(profile: SaltV2Profile) -> u8 {
    match profile {
        SaltV2Profile::CompactV1 => 1,
        SaltV2Profile::NearLosslessV1 => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_final_logits_binding_layout_is_frozen() {
        let receipt = Qwen36FinalLogitsOutputBindingReceipt {
            binding_id: ContentId::from_digest([0; 32]),
            completion_id: ContentId::from_digest([1; 32]),
            campaign_id: ContentId::from_digest([2; 32]),
            admission_id: ContentId::from_digest([3; 32]),
            selection_id: ContentId::from_digest([4; 32]),
            source_model_id: [5; 32],
            master_set_id: [6; 32],
            package_id: [7; 32],
            preserved_package_id: [8; 32],
            output_spec_id: [9; 32],
            output_receipt_id: [10; 32],
            selected_candidate_id: [11; 32],
            execution_receipt_id: [12; 32],
            final_logits_digest: [13; 32],
            profile: SaltV2Profile::NearLosslessV1,
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count: 14,
            logit_count: 15,
        };
        let canonical = receipt.canonical_bytes().expect("encode frozen binding");
        assert_eq!(&canonical[..8], &BINDING_MAGIC);
        assert_eq!(&canonical[8..10], &BINDING_VERSION.to_le_bytes());
        assert_eq!(canonical[10], 2);
        assert_eq!(canonical[11], FINAL_LOGITS_COVERAGE);
        assert_eq!(&canonical[12..16], &[0; 4]);
        assert_eq!(canonical.len(), BINDING_BYTES);
        assert_eq!(
            ContentId::of_bytes(&canonical).to_string(),
            "tsc1_39a862794fa23bd659bb7bcdacd680805a2db2853d02106b4cb965173605621a"
        );
    }
}
