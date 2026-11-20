//! Versioned binding for an immutable, scale-refined Qwen child execution.

use tritium_format::{RuntimeOutputScope, salt_v2_package::SaltV2ScaleUpdateChild};
use tritium_nn::{NnError, Qwen35UntrustedOutputScopeTranscript, Qwen35UntrustedRuntimeTranscript};
use tritium_quantize::{
    OutputReconstructionReceipt, OutputReconstructionScaleCandidate, OutputReconstructionScope,
    OutputReconstructionSpec, SaltV2Profile,
};

use super::{MAX_IDENTITY_BYTES, Qwen36ExecutionBackend};
use crate::ContentId;

const MAGIC: [u8; 8] = *b"TSQ36RC\0";
const VERSION: u16 = 1;
const CHECKSUM_CONTEXT: &str = "tritium qwen3.6 refined candidate receipt checksum v1";
const DIGEST_COUNT: usize = 15;
const MAX_BYTES: usize = 64 * 1024;

pub(super) struct ChildReplayEvidence<'spec, 'candidate, 'runtime> {
    pub(super) spec: &'spec OutputReconstructionSpec,
    pub(super) output_bytes: &'spec [u8],
    pub(super) scale_candidate: OutputReconstructionScaleCandidate<'candidate>,
    pub(super) transcript: &'runtime Qwen35UntrustedRuntimeTranscript,
    pub(super) scope_transcript: &'runtime Qwen35UntrustedOutputScopeTranscript,
    pub(super) backend: Qwen36ExecutionBackend,
}

/// Structurally verified identity and runtime evidence for a refined Qwen child.
///
/// This record is deliberately separate from `TSQ36EX v1` and `TSQ36SB v1`.
/// Parsing proves canonical encoding and checksum integrity only; the admitted
/// execution capability must still freshly replay the child to authorize it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36RefinedCandidateExecutionReceipt {
    receipt_id: ContentId,
    parent_execution_receipt_id: [u8; 32],
    parent_package_id: [u8; 32],
    child_package_id: [u8; 32],
    preserved_package_id: [u8; 32],
    scale_update_set_digest: [u8; 32],
    child_lineage_id: [u8; 32],
    output_spec_id: [u8; 32],
    output_receipt_id: [u8; 32],
    child_candidate_id: [u8; 32],
    child_transcript_content_id: [u8; 32],
    child_transcript_id: [u8; 32],
    backend_caps_digest: [u8; 32],
    token_stream_digest: [u8; 32],
    final_logits_digest: [u8; 32],
    scope_evidence_digest: [u8; 32],
    profile: SaltV2Profile,
    backend: Qwen36ExecutionBackend,
    manifest_package_id: String,
    config_package_id: String,
    backend_id: String,
    physical_device_id: String,
    batch_count: u64,
    token_count: u64,
    logit_count: u64,
    scope_count: u32,
    block_scope_count: u32,
    scope_observation_count: u64,
    scope_value_count: u64,
}

impl Qwen36RefinedCandidateExecutionReceipt {
    /// Mint only from parent-authorized lineage, a strictly reopened selected
    /// output receipt, and fresh child runtime transcripts.
    pub(super) fn from_child_replay(
        parent: &super::Qwen36AdmittedExecutionReceipt,
        lineage: SaltV2ScaleUpdateChild,
        replay: ChildReplayEvidence<'_, '_, '_>,
    ) -> Result<Self, NnError> {
        let ChildReplayEvidence {
            spec,
            output_bytes,
            scale_candidate,
            transcript,
            scope_transcript,
            backend,
        } = replay;
        if lineage.parent_package_id() != parent.package_id
            || scale_candidate.parent_package_digest() != parent.package_id.as_bytes()
            || scale_candidate.spec_id() != spec.spec_id()
            || spec.source_model_id() != parent.source_model_id
            || spec.token_stream_digest() != parent.token_stream_digest()
            || parent.backend != backend
        {
            return Err(NnError::Provenance(
                "refined child does not descend from the admitted parent candidate".to_owned(),
            ));
        }
        let update_digest =
            SaltV2ScaleUpdateChild::update_set_digest_for(scale_candidate.updates())
                .map_err(|error| NnError::InvalidArtifact(error.to_string()))?;
        if lineage.update_set_digest() != update_digest {
            return Err(NnError::Provenance(
                "child lineage update digest differs from the selected scale updates".to_owned(),
            ));
        }
        let expected_candidate = spec
            .scale_update_candidate(
                parent.package_id.as_bytes(),
                scale_candidate.initialization_seed(),
                scale_candidate.updates(),
            )
            .map_err(|error| NnError::InvalidArtifact(error.to_string()))?;
        if expected_candidate.candidate_id() != scale_candidate.candidate_id() {
            return Err(NnError::Provenance(
                "scale-update candidate identity is not canonical".to_owned(),
            ));
        }
        let output = OutputReconstructionReceipt::from_canonical_bytes(spec, output_bytes)
            .map_err(|error| NnError::InvalidArtifact(error.to_string()))?;
        let selected = output.selected();
        if output.spec_id() != spec.spec_id()
            || selected.candidate_id() != scale_candidate.candidate_id()
            || selected.runtime_final_logits_digest() != transcript.final_logits_digest()
            || selected.runtime_batch_count() != transcript.batch_count()
            || selected.runtime_logit_count() != transcript.logit_count()
        {
            return Err(NnError::Provenance(
                "selected output receipt does not describe this refined child replay".to_owned(),
            ));
        }
        if transcript.package_id() != lineage.child_package_id().to_string()
            || transcript.preserved_package_id() != parent.preserved_package_id.to_string()
            || transcript.profile() != profile_name(parent.profile)
            || transcript.manifest_package_id() != parent.manifest_package_id
            || transcript.config_package_id() != parent.config_package_id
            || transcript.claimed_backend_id() != parent.backend_id
            || transcript.claimed_physical_device_id() != parent.physical_device_id
            || transcript.claimed_backend_caps_digest() != &parent.backend_caps_digest
            || transcript.token_stream_digest() != spec.token_stream_digest()
            || transcript.token_stream_digest() != parent.token_stream_digest()
            || !transcript.has_final_logits()
            || transcript.has_block_outputs()
        {
            return Err(NnError::Provenance(
                "child runtime transcript differs from parent, candidate, or output stream"
                    .to_owned(),
            ));
        }
        let evidence = scope_transcript.scope_evidence();
        if scope_transcript.token_stream_digest() != spec.token_stream_digest()
            || scope_transcript.batch_count() != transcript.batch_count()
            || scope_transcript.token_count() != transcript.token_count()
            || evidence.len() != spec.scopes().len()
            || evidence.len() != selected.scope_evidence().len()
            || evidence.len() < 2
            || evidence != selected.scope_evidence()
            || evidence.iter().any(|item| {
                item.spec_id() != spec.spec_id()
                    || item.candidate_id() != scale_candidate.candidate_id()
                    || item.initialization_seed() != scale_candidate.initialization_seed()
            })
        {
            return Err(NnError::Provenance(
                "fresh child scope replay differs from selected output evidence".to_owned(),
            ));
        }
        for (declared, observed) in spec.scopes().iter().zip(evidence) {
            let expected = match declared {
                OutputReconstructionScope::Block { start, end } => RuntimeOutputScope::Block {
                    start: *start,
                    end: *end,
                },
                OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
            };
            if observed.scope() != expected {
                return Err(NnError::Provenance(
                    "child output-scope order differs from the frozen schedule".to_owned(),
                ));
            }
        }
        let block_scope_count = u32::try_from(
            evidence
                .iter()
                .filter(|item| item.scope() != RuntimeOutputScope::FinalLogits)
                .count(),
        )
        .map_err(|_| NnError::ResourceExhausted("scope count exceeds u32".to_owned()))?;
        if block_scope_count == 0
            || !evidence
                .iter()
                .any(|item| item.scope() == RuntimeOutputScope::FinalLogits)
        {
            return Err(NnError::Provenance(
                "child scope replay lacks block or final-logit evidence".to_owned(),
            ));
        }
        let scope_count = u32::try_from(evidence.len())
            .map_err(|_| NnError::ResourceExhausted("scope count exceeds u32".to_owned()))?;
        let mut scope_observation_count = 0_u64;
        let mut scope_value_count = 0_u64;
        for item in evidence {
            scope_observation_count = scope_observation_count
                .checked_add(item.observation_count())
                .ok_or_else(|| NnError::ResourceExhausted("scope count overflow".to_owned()))?;
            scope_value_count = scope_value_count
                .checked_add(item.value_count())
                .ok_or_else(|| NnError::ResourceExhausted("scope value overflow".to_owned()))?;
        }
        let parent_bytes = parent.canonical_bytes()?;
        let transcript_bytes = transcript.canonical_bytes()?;
        let mut receipt = Self {
            receipt_id: ContentId::from_digest([0; 32]),
            parent_execution_receipt_id: *ContentId::of_bytes(&parent_bytes).as_bytes(),
            parent_package_id: *parent.package_id.as_bytes(),
            child_package_id: *lineage.child_package_id().as_bytes(),
            preserved_package_id: *parent.preserved_package_id.as_bytes(),
            scale_update_set_digest: update_digest,
            child_lineage_id: lineage.lineage_id(),
            output_spec_id: *spec.spec_id(),
            output_receipt_id: *output.receipt_id(),
            child_candidate_id: *scale_candidate.candidate_id(),
            child_transcript_content_id: *ContentId::of_bytes(&transcript_bytes).as_bytes(),
            child_transcript_id: *transcript.transcript_id(),
            backend_caps_digest: *transcript.claimed_backend_caps_digest(),
            token_stream_digest: *transcript.token_stream_digest(),
            final_logits_digest: *transcript.final_logits_digest(),
            scope_evidence_digest: super::output_binding::digest_scope_evidence(evidence),
            profile: parent.profile,
            backend,
            manifest_package_id: transcript.manifest_package_id().to_owned(),
            config_package_id: transcript.config_package_id().to_owned(),
            backend_id: transcript.claimed_backend_id().to_owned(),
            physical_device_id: transcript.claimed_physical_device_id().to_owned(),
            batch_count: transcript.batch_count(),
            token_count: transcript.token_count(),
            logit_count: transcript.logit_count(),
            scope_count,
            block_scope_count,
            scope_observation_count,
            scope_value_count,
        };
        let canonical = receipt.canonical_bytes()?;
        receipt.receipt_id = ContentId::of_bytes(&canonical);
        Ok(receipt)
    }

    /// Content identity of these exact canonical `TSQ36RC v1` bytes.
    #[must_use]
    pub const fn receipt_id(&self) -> ContentId {
        self.receipt_id
    }

    /// Parent `TSQ36EX v1` identity. It remains a distinct parent receipt.
    #[must_use]
    pub const fn parent_execution_receipt_id(&self) -> &[u8; 32] {
        &self.parent_execution_receipt_id
    }

    /// Exact immutable parent SALT package identity.
    #[must_use]
    pub const fn parent_package_id(&self) -> &[u8; 32] {
        &self.parent_package_id
    }

    /// Exact child SALT package that was executed.
    #[must_use]
    pub const fn child_package_id(&self) -> &[u8; 32] {
        &self.child_package_id
    }

    /// Canonical scale-update set applied to the parent.
    #[must_use]
    pub const fn scale_update_set_digest(&self) -> &[u8; 32] {
        &self.scale_update_set_digest
    }

    /// Lineage identity binding the parent, updates, and child.
    #[must_use]
    pub const fn child_lineage_id(&self) -> &[u8; 32] {
        &self.child_lineage_id
    }

    /// Exact selected output-reconstruction spec and receipt identities.
    #[must_use]
    pub const fn output_binding_ids(&self) -> (&[u8; 32], &[u8; 32], &[u8; 32]) {
        (
            &self.output_spec_id,
            &self.output_receipt_id,
            &self.child_candidate_id,
        )
    }

    /// Exact token stream and runtime-produced final-logit identities.
    #[must_use]
    pub const fn runtime_output_ids(&self) -> (&[u8; 32], &[u8; 32]) {
        (&self.token_stream_digest, &self.final_logits_digest)
    }

    /// Exact child runtime scope evidence digest and coverage counts.
    #[must_use]
    pub const fn child_scope_evidence(&self) -> (&[u8; 32], u32, u32, u64, u64) {
        (
            &self.scope_evidence_digest,
            self.scope_count,
            self.block_scope_count,
            self.scope_observation_count,
            self.scope_value_count,
        )
    }

    /// Exact child runtime transcript content and transcript identities.
    #[must_use]
    pub const fn child_transcript_ids(&self) -> (&[u8; 32], &[u8; 32]) {
        (&self.child_transcript_content_id, &self.child_transcript_id)
    }

    /// Preserved source-precision package used by the child model.
    #[must_use]
    pub const fn preserved_package_id(&self) -> &[u8; 32] {
        &self.preserved_package_id
    }

    /// Selected execution profile and built-in backend identity.
    #[must_use]
    pub const fn execution_profile_backend(&self) -> (SaltV2Profile, Qwen36ExecutionBackend) {
        (self.profile, self.backend)
    }

    /// Exact configuration/manifest and physical backend identities.
    #[must_use]
    pub fn runtime_identity(&self) -> (&str, &str, &str, &str) {
        (
            &self.manifest_package_id,
            &self.config_package_id,
            &self.backend_id,
            &self.physical_device_id,
        )
    }

    /// Capability digest and exact executed batch/token/logit counts.
    #[must_use]
    pub const fn runtime_capability_counts(&self) -> (&[u8; 32], u64, u64, u64) {
        (
            &self.backend_caps_digest,
            self.batch_count,
            self.token_count,
            self.logit_count,
        )
    }

    /// Encode the canonical, checksummed receipt.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NnError> {
        let strings = [
            &self.manifest_package_id,
            &self.config_package_id,
            &self.backend_id,
            &self.physical_device_id,
        ];
        let string_bytes = strings.iter().try_fold(0usize, |total, value| {
            if value.is_empty() || value.len() > MAX_IDENTITY_BYTES || value.contains('\0') {
                return Err(NnError::Provenance(
                    "refined Qwen receipt identity is invalid".to_owned(),
                ));
            }
            total
                .checked_add(2 + value.len())
                .ok_or_else(|| NnError::ResourceExhausted("receipt length overflow".to_owned()))
        })?;
        let capacity =
            8 + 2 + 1 + 1 + 4 + 4 + DIGEST_COUNT * 32 + 3 * 8 + 2 * 4 + 2 * 8 + string_bytes + 32;
        if capacity > MAX_BYTES
            || self.batch_count == 0
            || self.token_count == 0
            || self.logit_count == 0
            || self.scope_count == 0
            || self.block_scope_count == 0
            || self.block_scope_count >= self.scope_count
            || self.scope_observation_count == 0
            || self.scope_value_count == 0
        {
            return Err(NnError::Provenance(
                "refined Qwen receipt exceeds bounds or has empty execution".to_owned(),
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| NnError::ResourceExhausted("allocate refined Qwen receipt".to_owned()))?;
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(profile_tag(self.profile));
        bytes.push(self.backend_tag());
        bytes.extend_from_slice(&self.backend_ordinal().to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        for digest in self.digests() {
            bytes.extend_from_slice(&digest);
        }
        for count in [self.batch_count, self.token_count, self.logit_count] {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        bytes.extend_from_slice(&self.scope_count.to_le_bytes());
        bytes.extend_from_slice(&self.block_scope_count.to_le_bytes());
        bytes.extend_from_slice(&self.scope_observation_count.to_le_bytes());
        bytes.extend_from_slice(&self.scope_value_count.to_le_bytes());
        for value in strings {
            let length = u16::try_from(value.len())
                .map_err(|_| NnError::ResourceExhausted("receipt identity too long".to_owned()))?;
            bytes.extend_from_slice(&length.to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut checksum = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        checksum.update(&bytes);
        bytes.extend_from_slice(checksum.finalize().as_bytes());
        Ok(bytes)
    }

    /// Strictly parse canonical `TSQ36RC v1` bytes and verify their checksum.
    ///
    /// This is an independent structural verifier, not execution authority.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, NnError> {
        const FIXED: usize = 8 + 2 + 1 + 1 + 4 + 4 + DIGEST_COUNT * 32 + 3 * 8 + 2 * 4 + 2 * 8;
        if bytes.len() < FIXED + 4 * 2 + 32
            || bytes.len() > MAX_BYTES
            || bytes.get(..8) != Some(&MAGIC)
            || bytes.get(8..10) != Some(&VERSION.to_le_bytes())
            || bytes.get(16..20) != Some(&[0; 4])
        {
            return Err(NnError::InvalidArtifact(
                "malformed refined Qwen receipt header".to_owned(),
            ));
        }
        let body_end = bytes.len() - 32;
        let mut checksum = blake3::Hasher::new_derive_key(CHECKSUM_CONTEXT);
        checksum.update(&bytes[..body_end]);
        if checksum.finalize().as_bytes() != &bytes[body_end..] {
            return Err(NnError::Provenance(
                "refined Qwen receipt checksum mismatch".to_owned(),
            ));
        }
        let profile = match bytes[10] {
            1 => SaltV2Profile::CompactV1,
            2 => SaltV2Profile::NearLosslessV1,
            _ => {
                return Err(NnError::InvalidArtifact(
                    "unsupported refined Qwen profile tag".to_owned(),
                ));
            }
        };
        let backend =
            match bytes[11] {
                1 if bytes[12..16] == [0; 4] => Qwen36ExecutionBackend::Cpu,
                2 => Qwen36ExecutionBackend::Cuda {
                    ordinal: u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| {
                        NnError::InvalidArtifact("truncated CUDA ordinal".to_owned())
                    })?),
                },
                _ => {
                    return Err(NnError::InvalidArtifact(
                        "unsupported backend tag".to_owned(),
                    ));
                }
            };
        let mut cursor = 20usize;
        let mut digests = [[0; 32]; DIGEST_COUNT];
        for digest in &mut digests {
            let end = cursor + 32;
            digest.copy_from_slice(bytes.get(cursor..end).ok_or_else(|| {
                NnError::InvalidArtifact("truncated refined Qwen digests".to_owned())
            })?);
            cursor = end;
        }
        let mut counts = [0_u64; 3];
        for count in &mut counts {
            let end = cursor + 8;
            *count = u64::from_le_bytes(
                bytes
                    .get(cursor..end)
                    .ok_or_else(|| {
                        NnError::InvalidArtifact("truncated refined Qwen counts".to_owned())
                    })?
                    .try_into()
                    .map_err(|_| {
                        NnError::InvalidArtifact("invalid refined Qwen count".to_owned())
                    })?,
            );
            cursor = end;
        }
        let scope_count = read_u32(bytes, &mut cursor, body_end)?;
        let block_scope_count = read_u32(bytes, &mut cursor, body_end)?;
        let scope_observation_count = read_u64(bytes, &mut cursor, body_end)?;
        let scope_value_count = read_u64(bytes, &mut cursor, body_end)?;
        let mut strings = Vec::new();
        strings.try_reserve_exact(4).map_err(|_| {
            NnError::ResourceExhausted("allocate refined receipt identities".to_owned())
        })?;
        for _ in 0..4 {
            let length_end = cursor + 2;
            let length = u16::from_le_bytes(
                bytes
                    .get(cursor..length_end)
                    .ok_or_else(|| {
                        NnError::InvalidArtifact(
                            "truncated refined receipt identity length".to_owned(),
                        )
                    })?
                    .try_into()
                    .map_err(|_| {
                        NnError::InvalidArtifact(
                            "invalid refined receipt identity length".to_owned(),
                        )
                    })?,
            ) as usize;
            cursor = length_end;
            let end = cursor.checked_add(length).ok_or_else(|| {
                NnError::InvalidArtifact("refined receipt identity overflow".to_owned())
            })?;
            if length == 0 || length > MAX_IDENTITY_BYTES || end > body_end {
                return Err(NnError::InvalidArtifact(
                    "refined receipt identity is out of bounds".to_owned(),
                ));
            }
            let text = std::str::from_utf8(&bytes[cursor..end]).map_err(|_| {
                NnError::InvalidArtifact("refined receipt identity is not UTF-8".to_owned())
            })?;
            if text.contains('\0') {
                return Err(NnError::InvalidArtifact(
                    "NUL in receipt identity".to_owned(),
                ));
            }
            strings.push(text.to_owned());
            cursor = end;
        }
        if cursor != body_end
            || counts.contains(&0)
            || scope_count == 0
            || block_scope_count == 0
            || block_scope_count >= scope_count
            || scope_observation_count == 0
            || scope_value_count == 0
            || digests.iter().any(|d| d == &[0; 32])
        {
            return Err(NnError::InvalidArtifact(
                "noncanonical or incomplete refined Qwen receipt".to_owned(),
            ));
        }
        let mut receipt = Self {
            receipt_id: ContentId::from_digest([0; 32]),
            parent_execution_receipt_id: digests[0],
            parent_package_id: digests[1],
            child_package_id: digests[2],
            preserved_package_id: digests[3],
            scale_update_set_digest: digests[4],
            child_lineage_id: digests[5],
            output_spec_id: digests[6],
            output_receipt_id: digests[7],
            child_candidate_id: digests[8],
            child_transcript_content_id: digests[9],
            child_transcript_id: digests[10],
            backend_caps_digest: digests[11],
            token_stream_digest: digests[12],
            final_logits_digest: digests[13],
            scope_evidence_digest: digests[14],
            profile,
            backend,
            manifest_package_id: strings.remove(0),
            config_package_id: strings.remove(0),
            backend_id: strings.remove(0),
            physical_device_id: strings.remove(0),
            batch_count: counts[0],
            token_count: counts[1],
            logit_count: counts[2],
            scope_count,
            block_scope_count,
            scope_observation_count,
            scope_value_count,
        };
        if receipt.canonical_bytes()?.as_slice() != bytes {
            return Err(NnError::InvalidArtifact(
                "refined Qwen receipt is not canonical".to_owned(),
            ));
        }
        receipt.receipt_id = ContentId::of_bytes(bytes);
        Ok(receipt)
    }

    fn digests(&self) -> [[u8; 32]; DIGEST_COUNT] {
        [
            self.parent_execution_receipt_id,
            self.parent_package_id,
            self.child_package_id,
            self.preserved_package_id,
            self.scale_update_set_digest,
            self.child_lineage_id,
            self.output_spec_id,
            self.output_receipt_id,
            self.child_candidate_id,
            self.child_transcript_content_id,
            self.child_transcript_id,
            self.backend_caps_digest,
            self.token_stream_digest,
            self.final_logits_digest,
            self.scope_evidence_digest,
        ]
    }

    const fn backend_tag(&self) -> u8 {
        match self.backend {
            Qwen36ExecutionBackend::Cpu => 1,
            Qwen36ExecutionBackend::Cuda { .. } => 2,
        }
    }

    const fn backend_ordinal(&self) -> u32 {
        match self.backend {
            Qwen36ExecutionBackend::Cpu => 0,
            Qwen36ExecutionBackend::Cuda { ordinal } => ordinal,
        }
    }
}

fn read_u32(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<u32, NnError> {
    let next = cursor
        .checked_add(4)
        .ok_or_else(|| NnError::InvalidArtifact("receipt offset overflow".to_owned()))?;
    let value = u32::from_le_bytes(
        bytes
            .get(*cursor..next)
            .filter(|_| next <= end)
            .ok_or_else(|| NnError::InvalidArtifact("truncated receipt integer".to_owned()))?
            .try_into()
            .map_err(|_| NnError::InvalidArtifact("invalid receipt integer".to_owned()))?,
    );
    *cursor = next;
    Ok(value)
}

fn read_u64(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<u64, NnError> {
    let next = cursor
        .checked_add(8)
        .ok_or_else(|| NnError::InvalidArtifact("receipt offset overflow".to_owned()))?;
    let value = u64::from_le_bytes(
        bytes
            .get(*cursor..next)
            .filter(|_| next <= end)
            .ok_or_else(|| NnError::InvalidArtifact("truncated receipt integer".to_owned()))?
            .try_into()
            .map_err(|_| NnError::InvalidArtifact("invalid receipt integer".to_owned()))?,
    );
    *cursor = next;
    Ok(value)
}

const fn profile_tag(profile: SaltV2Profile) -> u8 {
    match profile {
        SaltV2Profile::CompactV1 => 1,
        SaltV2Profile::NearLosslessV1 => 2,
    }
}

const fn profile_name(profile: SaltV2Profile) -> &'static str {
    match profile {
        SaltV2Profile::CompactV1 => "compact-v1",
        SaltV2Profile::NearLosslessV1 => "near-lossless-v1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_receipt() -> Qwen36RefinedCandidateExecutionReceipt {
        Qwen36RefinedCandidateExecutionReceipt {
            receipt_id: ContentId::from_digest([0; 32]),
            parent_execution_receipt_id: [1; 32],
            parent_package_id: [2; 32],
            child_package_id: [3; 32],
            preserved_package_id: [4; 32],
            scale_update_set_digest: [5; 32],
            child_lineage_id: [6; 32],
            output_spec_id: [7; 32],
            output_receipt_id: [8; 32],
            child_candidate_id: [9; 32],
            child_transcript_content_id: [10; 32],
            child_transcript_id: [11; 32],
            backend_caps_digest: [12; 32],
            token_stream_digest: [13; 32],
            final_logits_digest: [14; 32],
            scope_evidence_digest: [15; 32],
            profile: SaltV2Profile::NearLosslessV1,
            backend: Qwen36ExecutionBackend::Cuda { ordinal: 3 },
            manifest_package_id: "manifest-id".to_owned(),
            config_package_id: "config-id".to_owned(),
            backend_id: "cuda:3".to_owned(),
            physical_device_id: "cuda:3:fixture".to_owned(),
            batch_count: 15,
            token_count: 16,
            logit_count: 17,
            scope_count: 3,
            block_scope_count: 2,
            scope_observation_count: 18,
            scope_value_count: 19,
        }
    }

    #[test]
    fn refined_receipt_roundtrips_and_binds_parent_child_updates_and_replay() {
        let bytes = fixture_receipt().canonical_bytes().unwrap();
        assert_eq!(&bytes[..8], &MAGIC);
        assert_eq!(&bytes[8..10], &VERSION.to_le_bytes());
        let reopened = Qwen36RefinedCandidateExecutionReceipt::from_canonical_bytes(&bytes)
            .expect("strictly reopen child receipt");
        assert_eq!(reopened.canonical_bytes().unwrap(), bytes);
        assert_eq!(reopened.parent_package_id(), &[2; 32]);
        assert_eq!(reopened.child_package_id(), &[3; 32]);
        assert_eq!(reopened.scale_update_set_digest(), &[5; 32]);
        assert_eq!(reopened.child_lineage_id(), &[6; 32]);
        assert_eq!(
            reopened.output_binding_ids(),
            (&[7; 32], &[8; 32], &[9; 32])
        );
        assert_eq!(reopened.runtime_output_ids(), (&[13; 32], &[14; 32]));
        assert_eq!(reopened.child_scope_evidence(), (&[15; 32], 3, 2, 18, 19));
        assert_eq!(reopened.receipt_id(), ContentId::of_bytes(&bytes));
    }

    #[test]
    fn refined_receipt_rejects_unknown_version_bad_checksum_and_noncanonical_header() {
        let canonical = fixture_receipt().canonical_bytes().unwrap();
        let mut unknown_version = canonical.clone();
        unknown_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert!(
            Qwen36RefinedCandidateExecutionReceipt::from_canonical_bytes(&unknown_version).is_err()
        );

        let mut bad_checksum = canonical.clone();
        bad_checksum[32] ^= 1;
        assert!(
            Qwen36RefinedCandidateExecutionReceipt::from_canonical_bytes(&bad_checksum).is_err()
        );

        let mut reserved = canonical;
        reserved[16] = 1;
        assert!(Qwen36RefinedCandidateExecutionReceipt::from_canonical_bytes(&reserved).is_err());
    }
}
