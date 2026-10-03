//! Final-logit binding between canonical output reconstruction and sealed execution.

use core::{convert::Infallible, fmt};
use std::error::Error;

use tritium_format::{RuntimeOutputScope, RuntimeOutputScopeEvidence};
use tritium_nn::NnError;
use tritium_quantize::{
    OutputReconstructionError, OutputReconstructionReceipt, OutputReconstructionScope,
    OutputReconstructionSpec, SaltV2Profile,
};

use crate::{ContentId, Qwen36PreservedSafetensorsError};

use super::{
    PRESERVED_CHUNK_BYTES, Qwen36AdmittedExecutionReceipt, Qwen36AdmittedExecutionSession,
    execution_authority,
};
use crate::Qwen36PackageAdmittedCampaignStore;

const BINDING_MAGIC: [u8; 8] = *b"TSQ36OB\0";
const BINDING_VERSION: u16 = 1;
const FINAL_LOGITS_COVERAGE: u8 = 1;
const BINDING_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 output execution binding checksum v1";
const CANDIDATE_ID_CONTEXT: &str = "tritium qwen3.6 admitted output candidate v1";
const BOUND_DIGESTS: usize = 13;
const BINDING_BYTES: usize = 8 + 2 + 1 + 1 + 4 + BOUND_DIGESTS * 32 + 2 * 8 + 32;
const BINDING_BODY_BYTES: usize = BINDING_BYTES - 32;
const SCOPE_BINDING_MAGIC: [u8; 8] = *b"TSQ36SB\0";
const SCOPE_BINDING_VERSION: u16 = 1;
const SCOPE_BINDING_CHECKSUM_CONTEXT: &str =
    "tritium qwen3.6 output scope execution binding checksum v1";
const SCOPE_EVIDENCE_SET_CONTEXT: &str = "tritium qwen3.6 output scope evidence set v1";

/// Failure while binding one selected output candidate to sealed runtime evidence.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36FinalLogitsOutputBindingError {
    /// Package admission or its authoritative campaign lineage changed.
    Admission(super::super::Qwen36PackageAdmissionError),
    /// Preserved-source reconstruction failed.
    Workspace(crate::Qwen36TensorWorkError),
    /// Canonical `TSV2OUT` bytes failed strict reopen.
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

/// Campaign-bound proof that every selected block/window and final-logit scope
/// commitment equals one fresh sealed Qwen execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36OutputScopeBindingReceipt {
    binding_id: ContentId,
    final_logits_binding: Qwen36FinalLogitsOutputBindingReceipt,
    scope_evidence_digest: [u8; 32],
    scope_count: u32,
    block_scope_count: u32,
    batch_count: u64,
    observation_count: u64,
    value_count: u64,
}

impl Qwen36OutputScopeBindingReceipt {
    /// Content identity of exact canonical `TSQ36SB` bytes.
    #[must_use]
    pub const fn binding_id(&self) -> ContentId {
        self.binding_id
    }

    /// Final-logit binding receipt whose campaign and execution lineage this extends.
    #[must_use]
    pub const fn final_logits_binding(&self) -> &Qwen36FinalLogitsOutputBindingReceipt {
        &self.final_logits_binding
    }

    /// Whether at least one block/window output is campaign-bound.
    #[must_use]
    pub const fn has_block_outputs(&self) -> bool {
        self.block_scope_count > 0
    }

    /// Number of output scopes, including final logits.
    #[must_use]
    pub const fn scope_count(&self) -> u32 {
        self.scope_count
    }

    /// Number of committed block or sliding-window scopes.
    #[must_use]
    pub const fn block_scope_count(&self) -> u32 {
        self.block_scope_count
    }

    /// Number of fresh execution batches represented by each scope.
    #[must_use]
    pub const fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Total per-scope batch observations committed by this receipt.
    #[must_use]
    pub const fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Total output values across all committed scopes.
    #[must_use]
    pub const fn value_count(&self) -> u64 {
        self.value_count
    }

    /// Domain-separated identity of every ordered scope commitment.
    #[must_use]
    pub const fn scope_evidence_digest(&self) -> &[u8; 32] {
        &self.scope_evidence_digest
    }

    /// Encode the final-logit binding and complete scope attestation canonically.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NnError> {
        let base = self.final_logits_binding.canonical_bytes()?;
        let base_len = u32::try_from(base.len()).map_err(|_| {
            NnError::ResourceExhausted("Qwen scope binding base length exceeds u32".to_owned())
        })?;
        let capacity = 8usize + 2 + 2 + 4 + 4 + 4 + 8 + 8 + 8 + 32 + base.len() + 32;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            NnError::ResourceExhausted("allocate Qwen output scope binding".to_owned())
        })?;
        bytes.extend_from_slice(&SCOPE_BINDING_MAGIC);
        bytes.extend_from_slice(&SCOPE_BINDING_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&base_len.to_le_bytes());
        bytes.extend_from_slice(&self.scope_count.to_le_bytes());
        bytes.extend_from_slice(&self.block_scope_count.to_le_bytes());
        bytes.extend_from_slice(&self.batch_count.to_le_bytes());
        bytes.extend_from_slice(&self.observation_count.to_le_bytes());
        bytes.extend_from_slice(&self.value_count.to_le_bytes());
        bytes.extend_from_slice(&self.scope_evidence_digest);
        bytes.extend_from_slice(&base);
        let mut checksum = blake3::Hasher::new_derive_key(SCOPE_BINDING_CHECKSUM_CONTEXT);
        checksum.update(&bytes);
        bytes.extend_from_slice(checksum.finalize().as_bytes());
        Ok(bytes)
    }
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

    /// Exact canonical `TSV2OUT` receipt identity.
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
    /// Bind selected SALT V2 final logits to exact sealed execution and master lineage.
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

impl<'admission, 'allocated, 'parent, 'store, 'source>
    Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>
{
    /// Bind every TSV2OUT v3 block/window scope to a fresh execution on this sealed model.
    ///
    /// This first revalidates final-logit binding and then executes the exact same
    /// token batches once, with the fitting row masks, through the session's
    /// campaign-authorized backend. Every runtime scope digest, count, identity,
    /// and ordering must equal the selected candidate's committed v3 evidence.
    ///
    /// # Errors
    /// Fails closed on v2 receipts without scope evidence, stale campaign authority,
    /// wrong session/execution lineage, mask/token/scope drift, or runtime failure.
    pub fn bind_output_reconstruction_scopes<'batch, I>(
        &self,
        spec: &OutputReconstructionSpec,
        output_bytes: &[u8],
        execution: &Qwen36AdmittedExecutionReceipt,
        batches: I,
    ) -> Result<Qwen36OutputScopeBindingReceipt, Qwen36FinalLogitsOutputBindingError>
    where
        I: IntoIterator<Item = (&'batch [u32], &'batch [bool])>,
    {
        let final_logits_binding = self.admission.bind_output_reconstruction_final_logits(
            spec,
            output_bytes,
            execution,
        )?;
        validate_execution(&self.authority, execution)
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        self.admission
            .verify_current()
            .map_err(Qwen36FinalLogitsOutputBindingError::Admission)?;

        let output = OutputReconstructionReceipt::from_canonical_bytes(spec, output_bytes)
            .map_err(Qwen36FinalLogitsOutputBindingError::Output)?;
        let selected = output.selected();
        let claimed_scope_evidence = selected.scope_evidence();
        if claimed_scope_evidence.len() != spec.scopes().len()
            || claimed_scope_evidence.len() < 2
            || claimed_scope_evidence.iter().any(|evidence| {
                evidence.spec_id() != spec.spec_id()
                    || evidence.candidate_id() != selected.candidate_id()
                    || evidence.initialization_seed()
                        != claimed_scope_evidence[0].initialization_seed()
            })
        {
            return Err(scope_runtime_error(
                "selected candidate does not carry complete v3 scope commitments",
            ));
        }
        let mut runtime_scopes = Vec::new();
        runtime_scopes
            .try_reserve_exact(spec.scopes().len())
            .map_err(|_| {
                Qwen36FinalLogitsOutputBindingError::Runtime(NnError::ResourceExhausted(
                    "allocate Qwen output scope replay schedule".to_owned(),
                ))
            })?;
        for scope in spec.scopes() {
            runtime_scopes.push(match scope {
                OutputReconstructionScope::Block { start, end } => RuntimeOutputScope::Block {
                    start: *start,
                    end: *end,
                },
                OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
            });
        }
        let transcript = self
            .model
            .try_visit_untrusted_output_scopes(
                spec.spec_id(),
                selected.candidate_id(),
                claimed_scope_evidence[0].initialization_seed(),
                &runtime_scopes,
                batches,
            )
            .map_err(|error| match error {
                tritium_nn::Qwen35ExecutionVisitError::Runtime(error) => {
                    Qwen36FinalLogitsOutputBindingError::Runtime(error)
                }
                tritium_nn::Qwen35ExecutionVisitError::Observer(never) => match never {},
            })?;
        if transcript.scope_evidence() != claimed_scope_evidence
            || transcript.token_stream_digest() != spec.token_stream_digest()
            || transcript.token_stream_digest() != execution.token_stream_digest()
            || transcript.batch_count() != execution.batch_count()
            || transcript.token_count() != execution.token_count()
        {
            return Err(scope_runtime_error(
                "fresh Qwen scope outputs differ from selected TSV2OUT v3 commitments",
            ));
        }
        self.admission
            .verify_current()
            .map_err(Qwen36FinalLogitsOutputBindingError::Admission)?;

        let scope_count = u32::try_from(claimed_scope_evidence.len()).map_err(|_| {
            Qwen36FinalLogitsOutputBindingError::Runtime(NnError::ResourceExhausted(
                "Qwen scope count exceeds u32".to_owned(),
            ))
        })?;
        let block_scope_count = u32::try_from(
            claimed_scope_evidence
                .iter()
                .filter(|evidence| evidence.scope() != RuntimeOutputScope::FinalLogits)
                .count(),
        )
        .map_err(|_| {
            Qwen36FinalLogitsOutputBindingError::Runtime(NnError::ResourceExhausted(
                "Qwen block scope count exceeds u32".to_owned(),
            ))
        })?;
        let mut observation_count = 0_u64;
        let mut value_count = 0_u64;
        for evidence in claimed_scope_evidence {
            observation_count = observation_count
                .checked_add(evidence.observation_count())
                .ok_or_else(|| {
                    Qwen36FinalLogitsOutputBindingError::Runtime(NnError::ResourceExhausted(
                        "Qwen scope observation count overflow".to_owned(),
                    ))
                })?;
            value_count = value_count
                .checked_add(evidence.value_count())
                .ok_or_else(|| {
                    Qwen36FinalLogitsOutputBindingError::Runtime(NnError::ResourceExhausted(
                        "Qwen scope value count overflow".to_owned(),
                    ))
                })?;
        }
        let mut receipt = Qwen36OutputScopeBindingReceipt {
            binding_id: ContentId::from_digest([0; 32]),
            final_logits_binding,
            scope_evidence_digest: digest_scope_evidence(claimed_scope_evidence),
            scope_count,
            block_scope_count,
            batch_count: transcript.batch_count(),
            observation_count,
            value_count,
        };
        let canonical = receipt
            .canonical_bytes()
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        receipt.binding_id = ContentId::of_bytes(&canonical);
        Ok(receipt)
    }

    /// Strictly reopen and freshly replay a persisted v3 scope binding.
    pub fn reopen_output_reconstruction_scopes_binding<'batch, I>(
        &self,
        spec: &OutputReconstructionSpec,
        output_bytes: &[u8],
        execution: &Qwen36AdmittedExecutionReceipt,
        batches: I,
        binding_bytes: &[u8],
    ) -> Result<Qwen36OutputScopeBindingReceipt, Qwen36FinalLogitsOutputBindingError>
    where
        I: IntoIterator<Item = (&'batch [u32], &'batch [bool])>,
    {
        let declared = decode_scope_binding(binding_bytes)
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        let rebound =
            self.bind_output_reconstruction_scopes(spec, output_bytes, execution, batches)?;
        let canonical = rebound
            .canonical_bytes()
            .map_err(Qwen36FinalLogitsOutputBindingError::Runtime)?;
        if declared != rebound || canonical.as_slice() != binding_bytes {
            return Err(scope_runtime_error(
                "persisted Qwen scope binding differs from fresh campaign execution",
            ));
        }
        Ok(rebound)
    }
}

fn scope_runtime_error(message: &str) -> Qwen36FinalLogitsOutputBindingError {
    Qwen36FinalLogitsOutputBindingError::Runtime(NnError::Provenance(message.to_owned()))
}

fn digest_scope_evidence(evidence: &[RuntimeOutputScopeEvidence]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(SCOPE_EVIDENCE_SET_CONTEXT);
    hasher.update(&(evidence.len() as u64).to_le_bytes());
    for scope in evidence {
        match scope.scope() {
            RuntimeOutputScope::Block { start, end } => {
                hasher.update(&[1]);
                hasher.update(&start.to_le_bytes());
                hasher.update(&end.to_le_bytes());
            }
            RuntimeOutputScope::FinalLogits => {
                hasher.update(&[2]);
            }
        }
        hasher.update(scope.spec_id());
        hasher.update(scope.candidate_id());
        hasher.update(&scope.initialization_seed().to_le_bytes());
        hasher.update(&scope.observation_count().to_le_bytes());
        hasher.update(&scope.value_count().to_le_bytes());
        hasher.update(scope.digest());
    }
    *hasher.finalize().as_bytes()
}

fn decode_scope_binding(bytes: &[u8]) -> Result<Qwen36OutputScopeBindingReceipt, NnError> {
    const HEADER_BYTES: usize = 80;
    const CHECKSUM_BYTES: usize = 32;
    if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES
        || bytes.get(..8) != Some(&SCOPE_BINDING_MAGIC)
        || bytes.get(8..10) != Some(&SCOPE_BINDING_VERSION.to_le_bytes())
        || bytes.get(10..12) != Some(&[0; 2])
    {
        return Err(NnError::InvalidArtifact(
            "malformed Qwen scope binding header".to_owned(),
        ));
    }
    let read_u32 = |range: std::ops::Range<usize>| -> Result<u32, NnError> {
        let raw: [u8; 4] = bytes
            .get(range)
            .ok_or_else(|| NnError::InvalidArtifact("truncated Qwen scope binding".to_owned()))?
            .try_into()
            .map_err(|_| NnError::InvalidArtifact("truncated Qwen scope binding".to_owned()))?;
        Ok(u32::from_le_bytes(raw))
    };
    let read_u64 = |range: std::ops::Range<usize>| -> Result<u64, NnError> {
        let raw: [u8; 8] = bytes
            .get(range)
            .ok_or_else(|| NnError::InvalidArtifact("truncated Qwen scope binding".to_owned()))?
            .try_into()
            .map_err(|_| NnError::InvalidArtifact("truncated Qwen scope binding".to_owned()))?;
        Ok(u64::from_le_bytes(raw))
    };
    let base_len = usize::try_from(read_u32(12..16)?).map_err(|_| {
        NnError::InvalidArtifact("Qwen scope binding base length exceeds usize".to_owned())
    })?;
    let expected_len = HEADER_BYTES
        .checked_add(base_len)
        .and_then(|length| length.checked_add(CHECKSUM_BYTES))
        .ok_or_else(|| NnError::InvalidArtifact("Qwen scope binding length overflow".to_owned()))?;
    if bytes.len() != expected_len {
        return Err(NnError::InvalidArtifact(
            "Qwen scope binding length mismatch".to_owned(),
        ));
    }
    let body_end = bytes.len() - CHECKSUM_BYTES;
    let mut checksum = blake3::Hasher::new_derive_key(SCOPE_BINDING_CHECKSUM_CONTEXT);
    checksum.update(&bytes[..body_end]);
    if bytes[body_end..] != checksum.finalize().as_bytes()[..] {
        return Err(NnError::InvalidArtifact(
            "Qwen scope binding checksum mismatch".to_owned(),
        ));
    }
    let scope_count = read_u32(16..20)?;
    let block_scope_count = read_u32(20..24)?;
    let batch_count = read_u64(24..32)?;
    let observation_count = read_u64(32..40)?;
    let value_count = read_u64(40..48)?;
    let scope_evidence_digest: [u8; 32] = bytes[48..80]
        .try_into()
        .map_err(|_| NnError::InvalidArtifact("truncated Qwen scope digest".to_owned()))?;
    if scope_count < 2
        || block_scope_count == 0
        || block_scope_count.checked_add(1) != Some(scope_count)
        || batch_count == 0
        || observation_count == 0
        || value_count == 0
        || scope_evidence_digest == [0; 32]
    {
        return Err(NnError::InvalidArtifact(
            "invalid Qwen scope binding counts or identity".to_owned(),
        ));
    }
    let final_logits_binding = decode_binding(&bytes[HEADER_BYTES..body_end])?;
    let receipt = Qwen36OutputScopeBindingReceipt {
        binding_id: ContentId::of_bytes(bytes),
        final_logits_binding,
        scope_evidence_digest,
        scope_count,
        block_scope_count,
        batch_count,
        observation_count,
        value_count,
    };
    if receipt
        .canonical_bytes()
        .map_err(|_| NnError::InvalidArtifact("noncanonical Qwen scope binding".to_owned()))?
        != bytes
    {
        return Err(NnError::InvalidArtifact(
            "noncanonical Qwen scope binding".to_owned(),
        ));
    }
    Ok(receipt)
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
