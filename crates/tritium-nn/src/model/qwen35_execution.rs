//! Runtime-owned output transcripts for admitted Qwen SALT V2 profiles.

use core::fmt;

use tritium_spec::{DeviceCaps, TernaryBackend};

use super::Qwen35SaltV2LanguageMtpModel;
use crate::NnError;

const TOKEN_STREAM_CONTEXT: &str = "tritium qwen3.5 runtime token stream v1";
const FINAL_LOGITS_CONTEXT: &str = "tritium qwen3.5 runtime final logits v1";
const BACKEND_CAPS_CONTEXT: &str = "tritium qwen3.5 runtime backend capabilities v1";
const TRANSCRIPT_ID_CONTEXT: &str = "tritium qwen3.5 salt v2 untrusted runtime transcript v1";
const TRANSCRIPT_CHECKSUM_CONTEXT: &str =
    "tritium qwen3.5 untrusted runtime transcript checksum v1";
const TRANSCRIPT_MAGIC: [u8; 8] = *b"TSQ35EX\0";
const TRANSCRIPT_VERSION: u16 = 1;
const UNTRUSTED_BACKEND_EVIDENCE: u8 = 1;
const FINAL_LOGITS_COVERAGE: u8 = 1;
const BLOCK_OUTPUT_COVERAGE: u8 = 2;
const MAX_EXECUTION_BATCHES: u64 = 1 << 20;
const MAX_IDENTITY_BYTES: usize = 4096;
const MAX_CAPABILITY_FEATURES: usize = 4096;
const MAX_TRANSCRIPT_BYTES: usize = 64 * 1024;

/// One runtime-produced output batch borrowed only for observer duration.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35ExecutionOutputBatch<'a> {
    batch_index: u64,
    tokens: &'a [u32],
    logits: &'a [f32],
}

impl<'a> Qwen35ExecutionOutputBatch<'a> {
    /// Zero-based execution order.
    #[must_use]
    pub const fn batch_index(self) -> u64 {
        self.batch_index
    }

    /// Exact input token sequence executed from a fresh model cache.
    #[must_use]
    pub const fn tokens(self) -> &'a [u32] {
        self.tokens
    }

    /// Runtime-produced final-position logits for this batch.
    #[must_use]
    pub const fn logits(self) -> &'a [f32] {
        self.logits
    }
}

/// Non-admissible transcript produced by a model using a caller-supplied backend.
///
/// The runtime, tokens, loaded package, and observed outputs are bound exactly,
/// but backend identity and behavior are self-asserted through the public
/// [`TernaryBackend`] boundary. This type is deliberately not a campaign receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35UntrustedRuntimeTranscript {
    transcript_id: [u8; 32],
    manifest_package_id: String,
    profile: String,
    package_id: String,
    preserved_package_id: String,
    config_package_id: String,
    backend_id: String,
    physical_device_id: String,
    backend_caps_digest: [u8; 32],
    token_stream_digest: [u8; 32],
    block_output_digest: [u8; 32],
    final_logits_digest: [u8; 32],
    scope_coverage: u8,
    batch_count: u64,
    token_count: u64,
    block_observation_count: u64,
    block_element_count: u64,
    logit_count: u64,
}

impl Qwen35UntrustedRuntimeTranscript {
    /// Content identity of loaded artifacts, backend claims, tokens, and outputs.
    #[must_use]
    pub const fn transcript_id(&self) -> &[u8; 32] {
        &self.transcript_id
    }

    /// This transcript carries no independently authenticated backend authority.
    #[must_use]
    pub const fn backend_claims_are_untrusted(&self) -> bool {
        true
    }

    /// Exact SALT V2 matrix package executed by the runtime.
    #[must_use]
    pub fn package_id(&self) -> &str {
        &self.package_id
    }

    /// Exact bundle-manifest identity validated before model assembly.
    #[must_use]
    pub fn manifest_package_id(&self) -> &str {
        &self.manifest_package_id
    }

    /// Selected package profile.
    #[must_use]
    pub fn profile(&self) -> &str {
        &self.profile
    }

    /// Caller-supplied backend's self-asserted logical identity.
    #[must_use]
    pub fn claimed_backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Caller-supplied backend's self-asserted physical identity.
    #[must_use]
    pub fn claimed_physical_device_id(&self) -> &str {
        &self.physical_device_id
    }

    /// Digest of caller-supplied backend's self-asserted capabilities.
    #[must_use]
    pub const fn claimed_backend_caps_digest(&self) -> &[u8; 32] {
        &self.backend_caps_digest
    }

    /// Exact token stream including batch boundaries and order.
    #[must_use]
    pub const fn token_stream_digest(&self) -> &[u8; 32] {
        &self.token_stream_digest
    }

    /// Runtime-produced final-logit stream identity.
    #[must_use]
    pub const fn final_logits_digest(&self) -> &[u8; 32] {
        &self.final_logits_digest
    }

    /// Whether every declared batch includes final logits.
    #[must_use]
    pub const fn has_final_logits(&self) -> bool {
        self.scope_coverage & FINAL_LOGITS_COVERAGE != 0
    }

    /// Whether block/window outputs were observed by this execution.
    #[must_use]
    pub const fn has_block_outputs(&self) -> bool {
        self.scope_coverage & BLOCK_OUTPUT_COVERAGE != 0
    }

    /// Number of fresh-cache token batches executed.
    #[must_use]
    pub const fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Total input tokens across all batches.
    #[must_use]
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Total final-logit values emitted across all batches.
    #[must_use]
    pub const fn logit_count(&self) -> u64 {
        self.logit_count
    }

    /// Encode canonical non-admissible `TSQ35EX` version-1 evidence.
    ///
    /// # Errors
    /// Returns [`NnError::ResourceExhausted`] on bounded allocation failure.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NnError> {
        let string_bytes = self
            .identity_strings()
            .iter()
            .try_fold(0usize, |total, value| {
                total
                    .checked_add(2)
                    .and_then(|total| total.checked_add(value.len()))
                    .ok_or_else(|| {
                        NnError::ResourceExhausted(
                            "Qwen execution transcript length overflow".to_owned(),
                        )
                    })
            })?;
        let capacity = (8usize + 2 + 2 + 1 + 1 + 6)
            .checked_add(string_bytes)
            .and_then(|bytes| bytes.checked_add(5 * 32 + 5 * 8 + 32))
            .ok_or_else(|| {
                NnError::ResourceExhausted("Qwen execution transcript length overflow".to_owned())
            })?;
        if capacity > MAX_TRANSCRIPT_BYTES {
            return Err(NnError::ResourceExhausted(
                "Qwen execution transcript exceeds canonical bound".to_owned(),
            ));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(capacity).map_err(|_| {
            NnError::ResourceExhausted("allocate Qwen execution transcript".to_owned())
        })?;
        output.extend_from_slice(&TRANSCRIPT_MAGIC);
        output.extend_from_slice(&TRANSCRIPT_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.push(UNTRUSTED_BACKEND_EVIDENCE);
        output.push(self.scope_coverage);
        output.extend_from_slice(&[0; 6]);
        for value in self.identity_strings() {
            encode_string(&mut output, value)?;
        }
        for digest in self.bound_digests() {
            output.extend_from_slice(&digest);
        }
        output.extend_from_slice(&self.transcript_id);
        for count in self.bound_counts() {
            output.extend_from_slice(&count.to_le_bytes());
        }
        let mut checksum = blake3::Hasher::new_derive_key(TRANSCRIPT_CHECKSUM_CONTEXT);
        checksum.update(&output);
        output.extend_from_slice(checksum.finalize().as_bytes());
        debug_assert_eq!(output.len(), capacity);
        Ok(output)
    }

    fn identity_strings(&self) -> [&str; 7] {
        [
            &self.manifest_package_id,
            &self.profile,
            &self.package_id,
            &self.preserved_package_id,
            &self.config_package_id,
            &self.backend_id,
            &self.physical_device_id,
        ]
    }

    fn bound_digests(&self) -> [[u8; 32]; 4] {
        [
            self.backend_caps_digest,
            self.token_stream_digest,
            self.block_output_digest,
            self.final_logits_digest,
        ]
    }

    fn bound_counts(&self) -> [u64; 5] {
        [
            self.batch_count,
            self.token_count,
            self.block_observation_count,
            self.block_element_count,
            self.logit_count,
        ]
    }

    fn derive_id(&self) -> Result<[u8; 32], NnError> {
        let mut hasher = blake3::Hasher::new_derive_key(TRANSCRIPT_ID_CONTEXT);
        hasher.update(&[UNTRUSTED_BACKEND_EVIDENCE, self.scope_coverage]);
        for value in self.identity_strings() {
            hash_string(&mut hasher, value)?;
        }
        for digest in self.bound_digests() {
            hasher.update(&digest);
        }
        for count in self.bound_counts() {
            hasher.update(&count.to_le_bytes());
        }
        Ok(*hasher.finalize().as_bytes())
    }
}

/// Failure while streaming outputs produced under untrusted backend claims.
#[derive(Debug)]
pub enum Qwen35ExecutionVisitError<E> {
    /// Model load/execution, identity, geometry, or transcript failure.
    Runtime(NnError),
    /// Caller observer rejected one runtime-produced batch.
    Observer(E),
}

impl<E: fmt::Display> fmt::Display for Qwen35ExecutionVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => write!(formatter, "Qwen runtime execution failed: {error}"),
            Self::Observer(error) => write!(formatter, "Qwen output observer failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Qwen35ExecutionVisitError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            Self::Observer(error) => Some(error),
        }
    }
}

impl Qwen35SaltV2LanguageMtpModel {
    /// Execute exact token batches and stream logits from a caller-supplied backend.
    ///
    /// Caller supplies tokens and an observer, never logits. A non-admissible
    /// transcript is returned only after every batch executes, observer accepts
    /// it, and self-asserted backend identity remains stable. Block-output
    /// coverage remains explicitly absent. Campaign admission must re-execute
    /// through a separately sealed built-in-backend session.
    ///
    /// # Errors
    /// Returns [`Qwen35ExecutionVisitError::Runtime`] for empty/oversized input,
    /// model/backend failure, identity drift, overflow, or allocation failure;
    /// returns [`Qwen35ExecutionVisitError::Observer`] without a transcript when the
    /// observer rejects a batch.
    pub fn try_visit_untrusted_final_logits<'batch, I, E>(
        &self,
        batches: I,
        mut observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        let backend_before = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let mut token_hasher = blake3::Hasher::new_derive_key(TOKEN_STREAM_CONTEXT);
        let mut logits_hasher = blake3::Hasher::new_derive_key(FINAL_LOGITS_CONTEXT);
        let mut batch_count = 0_u64;
        let mut token_count = 0_u64;
        let mut logit_count = 0_u64;

        for tokens in batches {
            if tokens.is_empty() {
                return Err(Qwen35ExecutionVisitError::Runtime(NnError::Shape {
                    expected: 1,
                    got: tokens.len(),
                }));
            }
            if batch_count == MAX_EXECUTION_BATCHES {
                return Err(Qwen35ExecutionVisitError::Runtime(
                    NnError::ResourceExhausted(
                        "Qwen execution batch count exceeds bound".to_owned(),
                    ),
                ));
            }
            let token_len = u64::try_from(tokens.len()).map_err(|_| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen execution token count exceeds u64".to_owned(),
                ))
            })?;
            let mut cache = self
                .runner()
                .new_cache(tokens.len())
                .map_err(Qwen35ExecutionVisitError::Runtime)?;
            let output = self
                .runner()
                .forward(tokens, &mut cache)
                .map_err(Qwen35ExecutionVisitError::Runtime)?;
            let logits = output.last_logits();
            if logits.len() != self.runner().vocab_size()
                || logits.iter().any(|value| !value.is_finite())
            {
                return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                    "Qwen runtime returned invalid final logits".to_owned(),
                )));
            }
            let logit_len = u64::try_from(logits.len()).map_err(|_| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen execution logit count exceeds u64".to_owned(),
                ))
            })?;
            hash_batch_tokens(&mut token_hasher, batch_count, tokens);
            hash_batch_logits(&mut logits_hasher, batch_count, logits);
            observer(Qwen35ExecutionOutputBatch {
                batch_index: batch_count,
                tokens,
                logits,
            })
            .map_err(Qwen35ExecutionVisitError::Observer)?;
            batch_count = batch_count.checked_add(1).ok_or_else(|| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen execution batch count overflow".to_owned(),
                ))
            })?;
            token_count = token_count.checked_add(token_len).ok_or_else(|| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen execution token count overflow".to_owned(),
                ))
            })?;
            logit_count = logit_count.checked_add(logit_len).ok_or_else(|| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen execution logit count overflow".to_owned(),
                ))
            })?;
        }
        if batch_count == 0 {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Shape {
                expected: 1,
                got: 0,
            }));
        }
        token_hasher.update(&batch_count.to_le_bytes());
        token_hasher.update(&token_count.to_le_bytes());
        logits_hasher.update(&batch_count.to_le_bytes());
        logits_hasher.update(&logit_count.to_le_bytes());

        let backend_after = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        if backend_after != backend_before {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen execution backend identity changed during evaluation".to_owned(),
            )));
        }
        let load = self.receipt();
        let mut transcript = Qwen35UntrustedRuntimeTranscript {
            transcript_id: [0; 32],
            manifest_package_id: try_owned(load.manifest_package_id())
                .map_err(Qwen35ExecutionVisitError::Runtime)?,
            profile: try_owned(load.profile()).map_err(Qwen35ExecutionVisitError::Runtime)?,
            package_id: try_owned(load.package_id()).map_err(Qwen35ExecutionVisitError::Runtime)?,
            preserved_package_id: try_owned(load.preserved_package_id())
                .map_err(Qwen35ExecutionVisitError::Runtime)?,
            config_package_id: try_owned(load.config_package_id())
                .map_err(Qwen35ExecutionVisitError::Runtime)?,
            backend_id: backend_before.backend_id,
            physical_device_id: backend_before.physical_device_id,
            backend_caps_digest: backend_before.capabilities_digest,
            token_stream_digest: *token_hasher.finalize().as_bytes(),
            block_output_digest: [0; 32],
            final_logits_digest: *logits_hasher.finalize().as_bytes(),
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count,
            token_count,
            block_observation_count: 0,
            block_element_count: 0,
            logit_count,
        };
        transcript.transcript_id = transcript
            .derive_id()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        Ok(transcript)
    }

    /// Re-execute tokens and require the same non-admissible transcript bytes.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::try_visit_untrusted_final_logits`], plus
    /// [`NnError::Provenance`] when supplied bytes differ from fresh execution.
    pub fn reexecute_untrusted_final_logits<'batch, I, E>(
        &self,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        let transcript = self.try_visit_untrusted_final_logits(batches, observer)?;
        let actual = transcript
            .canonical_bytes()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        if actual != expected_canonical {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen execution transcript differs from fresh runtime output".to_owned(),
            )));
        }
        Ok(transcript)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BackendIdentity {
    backend_id: String,
    physical_device_id: String,
    capabilities_digest: [u8; 32],
}

impl BackendIdentity {
    fn capture(backend: &dyn TernaryBackend) -> Result<Self, NnError> {
        Ok(Self {
            backend_id: try_owned(backend.device_id())?,
            physical_device_id: try_owned(backend.physical_device_id())?,
            capabilities_digest: hash_capabilities(backend.capabilities())?,
        })
    }
}

fn try_owned(value: &str) -> Result<String, NnError> {
    validate_identity(value)?;
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| NnError::ResourceExhausted("allocate Qwen execution identity".to_owned()))?;
    owned.push_str(value);
    Ok(owned)
}

fn validate_identity(value: &str) -> Result<(), NnError> {
    if value.is_empty() || value.len() > MAX_IDENTITY_BYTES || value.contains('\0') {
        return Err(NnError::Provenance(
            "Qwen execution identity is empty, oversized, or contains NUL".to_owned(),
        ));
    }
    Ok(())
}

fn hash_capabilities(mut caps: DeviceCaps) -> Result<[u8; 32], NnError> {
    validate_identity(&caps.backend)?;
    validate_identity(&caps.device_name)?;
    if caps.features.len() > MAX_CAPABILITY_FEATURES {
        return Err(NnError::ResourceExhausted(
            "Qwen execution backend feature count exceeds bound".to_owned(),
        ));
    }
    for feature in &caps.features {
        validate_identity(feature)?;
    }
    caps.features.sort();
    if caps.features.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(NnError::Provenance(
            "Qwen execution backend capabilities contain duplicate features".to_owned(),
        ));
    }
    let mut hasher = blake3::Hasher::new_derive_key(BACKEND_CAPS_CONTEXT);
    hash_string(&mut hasher, &caps.backend)?;
    hash_string(&mut hasher, &caps.device_name)?;
    hasher.update(&(caps.features.len() as u64).to_le_bytes());
    for feature in &caps.features {
        hash_string(&mut hasher, feature)?;
    }
    hasher.update(&caps.total_memory_bytes.to_le_bytes());
    hasher.update(&[u8::from(caps.supports_imma), u8::from(caps.supports_fp8)]);
    Ok(*hasher.finalize().as_bytes())
}

fn hash_batch_tokens(hasher: &mut blake3::Hasher, batch_index: u64, tokens: &[u32]) {
    hasher.update(&batch_index.to_le_bytes());
    hasher.update(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(&token.to_le_bytes());
    }
}

fn hash_batch_logits(hasher: &mut blake3::Hasher, batch_index: u64, logits: &[f32]) {
    hasher.update(&batch_index.to_le_bytes());
    hasher.update(&(logits.len() as u64).to_le_bytes());
    for logit in logits {
        hasher.update(&logit.to_bits().to_le_bytes());
    }
}

fn encode_string(output: &mut Vec<u8>, value: &str) -> Result<(), NnError> {
    validate_identity(value)?;
    let length = u16::try_from(value.len()).map_err(|_| {
        NnError::ResourceExhausted("Qwen execution identity exceeds u16".to_owned())
    })?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) -> Result<(), NnError> {
    validate_identity(value)?;
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_untrusted_transcript_v1_is_frozen() {
        let mut transcript = Qwen35UntrustedRuntimeTranscript {
            transcript_id: [0; 32],
            manifest_package_id: "manifest".to_owned(),
            profile: "compact-v1".to_owned(),
            package_id: "package".to_owned(),
            preserved_package_id: "preserved".to_owned(),
            config_package_id: "config".to_owned(),
            backend_id: "backend".to_owned(),
            physical_device_id: "physical".to_owned(),
            backend_caps_digest: [1; 32],
            token_stream_digest: [2; 32],
            block_output_digest: [0; 32],
            final_logits_digest: [3; 32],
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count: 4,
            token_count: 5,
            block_observation_count: 0,
            block_element_count: 0,
            logit_count: 6,
        };
        transcript.transcript_id = transcript.derive_id().unwrap();
        let canonical = transcript.canonical_bytes().unwrap();
        assert_eq!(&canonical[..8], b"TSQ35EX\0");
        assert_eq!(canonical[12], UNTRUSTED_BACKEND_EVIDENCE);
        assert_eq!(canonical[13], FINAL_LOGITS_COVERAGE);
        assert_eq!(
            blake3::hash(&canonical).to_hex().as_str(),
            "ef655cf388745e3c38108f365b771ecc47ce043e213544ce976ba060fa2e84e4"
        );
    }
}
