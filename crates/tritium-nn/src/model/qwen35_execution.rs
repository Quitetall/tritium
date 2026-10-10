//! Runtime-owned output transcripts for admitted Qwen SALT V2 profiles.

use core::fmt;

use tritium_format::{
    RuntimeBlockOutputsAccumulator, RuntimeFinalLogitsAccumulator, RuntimeOutputScope,
    RuntimeOutputScopeAccumulator, RuntimeOutputScopeEvidence,
};
use tritium_spec::{DeviceCaps, TernaryBackend};

use super::Qwen35SaltV2LanguageMtpModel;
use crate::NnError;

const TOKEN_STREAM_CONTEXT: &str = "tritium qwen3.5 runtime token stream v1";
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
const MAX_OUTPUT_SCOPES: usize = 4096;

/// One runtime-produced output batch borrowed only for observer duration.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35ExecutionOutputBatch<'a> {
    batch_index: u64,
    tokens: &'a [u32],
    logits: &'a [f32],
}

/// One post-block residual matrix borrowed only for the block observer call.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35ExecutionBlockOutputBatch<'a> {
    batch_index: u64,
    block_index: u32,
    token_start: u64,
    tokens: &'a [u32],
    hidden_size: usize,
    hidden_states: &'a [f32],
}

impl<'a> Qwen35ExecutionBlockOutputBatch<'a> {
    /// Zero-based execution batch.
    #[must_use]
    pub const fn batch_index(self) -> u64 {
        self.batch_index
    }

    /// Zero-based transformer block index.
    #[must_use]
    pub const fn block_index(self) -> u32 {
        self.block_index
    }

    /// First absolute token position represented by this output matrix.
    #[must_use]
    pub const fn token_start(self) -> u64 {
        self.token_start
    }

    /// Exact input token sequence that produced these rows.
    #[must_use]
    pub const fn tokens(self) -> &'a [u32] {
        self.tokens
    }

    /// Hidden width of each row.
    #[must_use]
    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    /// Post-attention and post-MLP residual rows, `[tokens, hidden_size]`.
    #[must_use]
    pub const fn hidden_states(self) -> &'a [f32] {
        self.hidden_states
    }
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

/// One-pass candidate-shaped outputs freshly observed from a caller-backend Qwen model.
///
/// This is untrusted runtime evidence, not a campaign admission receipt. A sealed
/// campaign must compare every scope commitment and bind the token digest to its
/// frozen specification before admitting it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35UntrustedOutputScopeTranscript {
    token_stream_digest: [u8; 32],
    batch_count: u64,
    token_count: u64,
    scope_evidence: Vec<RuntimeOutputScopeEvidence>,
}

impl Qwen35UntrustedOutputScopeTranscript {
    /// This transcript carries no independently authenticated backend authority.
    #[must_use]
    pub const fn backend_claims_are_untrusted(&self) -> bool {
        true
    }

    /// Exact ordered token batches executed from fresh model caches.
    #[must_use]
    pub const fn token_stream_digest(&self) -> &[u8; 32] {
        &self.token_stream_digest
    }

    /// Number of fresh-cache batches executed.
    #[must_use]
    pub const fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Total input tokens executed.
    #[must_use]
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Runtime-computed output commitments in the supplied scope order.
    #[must_use]
    pub fn scope_evidence(&self) -> &[RuntimeOutputScopeEvidence] {
        &self.scope_evidence
    }
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

    /// Exact preserved-companion package executed by the runtime.
    #[must_use]
    pub fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }

    /// Exact execution-configuration package loaded by the runtime.
    #[must_use]
    pub fn config_package_id(&self) -> &str {
        &self.config_package_id
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

    /// Runtime-produced ordered block-output stream identity.
    #[must_use]
    pub const fn block_output_digest(&self) -> &[u8; 32] {
        &self.block_output_digest
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

    /// Number of post-block output matrices observed by the block-output visitor.
    #[must_use]
    pub const fn block_observation_count(&self) -> u64 {
        self.block_observation_count
    }

    /// Total values across all observed post-block output matrices.
    #[must_use]
    pub const fn block_element_count(&self) -> u64 {
        self.block_element_count
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
    /// Execute fresh-cache batches once and seal the exact block/window and final-logit scopes.
    ///
    /// Block scopes commit the post-block residual output at `end - 1`; their
    /// declared start/end range is part of the digest identity. The caller supplies
    /// the same per-token row-selection masks used during fitting. Final logits
    /// cover the last input position of each batch, which must be selected by that
    /// batch's mask. No activation history is retained.
    ///
    /// The returned transcript is not campaign-admitted: package, source, masks,
    /// and candidate lineage still require comparison by the sealed campaign.
    ///
    /// # Errors
    /// Rejects malformed scope sets, identities, masks, execution failures, or a
    /// backend identity change during evaluation.
    pub fn try_visit_untrusted_output_scopes<'batch, I>(
        &self,
        spec_id: &[u8; 32],
        candidate_id: &[u8; 32],
        initialization_seed: u64,
        scopes: &[RuntimeOutputScope],
        batches: I,
    ) -> Result<
        Qwen35UntrustedOutputScopeTranscript,
        Qwen35ExecutionVisitError<core::convert::Infallible>,
    >
    where
        I: IntoIterator<Item = (&'batch [u32], &'batch [bool])>,
    {
        if scopes.is_empty() || scopes.len() > MAX_OUTPUT_SCOPES {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen output scope count is invalid".to_owned(),
            )));
        }
        let layer_count =
            usize::try_from(self.runner().config().num_hidden_layers).map_err(|_| {
                Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                    "Qwen configured layer count exceeds usize".to_owned(),
                ))
            })?;
        let mut accumulators = Vec::new();
        let mut scope_ends = Vec::new();
        accumulators.try_reserve_exact(scopes.len()).map_err(|_| {
            Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                "allocate Qwen output scope accumulators".to_owned(),
            ))
        })?;
        scope_ends.try_reserve_exact(scopes.len()).map_err(|_| {
            Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                "allocate Qwen output scope schedule".to_owned(),
            ))
        })?;
        let mut previous_end = 0_u32;
        let mut final_logits_seen = false;
        for (index, scope) in scopes.iter().copied().enumerate() {
            let end = match scope {
                RuntimeOutputScope::Block { start, end }
                    if !final_logits_seen
                        && start < end
                        && usize::try_from(end).is_ok_and(|end| end <= layer_count)
                        && end >= previous_end =>
                {
                    previous_end = end;
                    Some(end)
                }
                RuntimeOutputScope::Block { .. } => {
                    return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                        "Qwen output block scopes are invalid or out of order".to_owned(),
                    )));
                }
                RuntimeOutputScope::FinalLogits
                    if !final_logits_seen && index + 1 == scopes.len() =>
                {
                    final_logits_seen = true;
                    None
                }
                RuntimeOutputScope::FinalLogits => {
                    return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                        "Qwen final-logit scope must occur exactly once at the end".to_owned(),
                    )));
                }
            };
            let accumulator = RuntimeOutputScopeAccumulator::new(
                spec_id,
                candidate_id,
                initialization_seed,
                scope,
            )
            .map_err(|error| {
                Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                    "Qwen output scope identity is invalid: {error}"
                )))
            })?;
            accumulators.push(accumulator);
            scope_ends.push(end);
        }
        if !final_logits_seen {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen output scope set is missing final logits".to_owned(),
            )));
        }

        let backend_before = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let mut token_hasher = blake3::Hasher::new_derive_key(TOKEN_STREAM_CONTEXT);
        let mut batch_count = 0_u64;
        let mut token_count = 0_u64;
        for (tokens, row_mask) in batches {
            if tokens.is_empty() || row_mask.len() != tokens.len() {
                return Err(Qwen35ExecutionVisitError::Runtime(NnError::Shape {
                    expected: tokens.len(),
                    got: row_mask.len(),
                }));
            }
            if !row_mask[tokens.len() - 1] {
                return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                    "Qwen final-logit row is not selected by the output mask".to_owned(),
                )));
            }
            if batch_count == MAX_EXECUTION_BATCHES {
                return Err(Qwen35ExecutionVisitError::Runtime(
                    NnError::ResourceExhausted(
                        "Qwen execution batch count exceeds bound".to_owned(),
                    ),
                ));
            }
            let batch_index = u32::try_from(batch_count).map_err(|_| {
                Qwen35ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "Qwen scope batch index exceeds u32".to_owned(),
                ))
            })?;
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
                .forward_with_block_observer(
                    tokens,
                    &mut cache,
                    |block_index, _token_start, block_tokens, hidden_states| {
                        let scope_end = block_index.checked_add(1).ok_or_else(|| {
                            NnError::ResourceExhausted("Qwen block index overflow".to_owned())
                        })?;
                        for (accumulator, end) in accumulators.iter_mut().zip(&scope_ends) {
                            if *end == Some(scope_end) {
                                accumulator
                                    .observe(
                                        batch_index,
                                        block_tokens.len(),
                                        self.runner().hidden_size(),
                                        row_mask,
                                        hidden_states,
                                    )
                                    .map_err(|error| {
                                        NnError::Provenance(format!(
                                            "Qwen runtime block scope evidence failed: {error}"
                                        ))
                                    })?;
                            }
                        }
                        Ok::<_, NnError>(())
                    },
                )
                .map_err(|error| match error {
                    super::qwen35::Qwen35TextForwardError::Runtime(error)
                    | super::qwen35::Qwen35TextForwardError::Observer(error) => {
                        Qwen35ExecutionVisitError::Runtime(error)
                    }
                })?;
            let logits = output.last_logits();
            let last = accumulators.last_mut().ok_or_else(|| {
                Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                    "Qwen final-logit accumulator is missing".to_owned(),
                ))
            })?;
            last.observe(batch_index, 1, logits.len(), &[true], logits)
                .map_err(|error| {
                    Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                        "Qwen runtime final-logit scope evidence failed: {error}"
                    )))
                })?;
            hash_batch_tokens(&mut token_hasher, batch_count, tokens);
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
        }
        if batch_count == 0 {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Shape {
                expected: 1,
                got: 0,
            }));
        }
        token_hasher.update(&batch_count.to_le_bytes());
        token_hasher.update(&token_count.to_le_bytes());
        let scope_evidence = accumulators
            .into_iter()
            .map(|accumulator| {
                accumulator.finish().map_err(|error| {
                    Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                        "Qwen output scope was not fully observed: {error}"
                    )))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let backend_after = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        if backend_after != backend_before {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen execution backend identity changed during evaluation".to_owned(),
            )));
        }
        Ok(Qwen35UntrustedOutputScopeTranscript {
            token_stream_digest: *token_hasher.finalize().as_bytes(),
            batch_count,
            token_count,
            scope_evidence,
        })
    }

    /// Execute exact token batches and stream every post-block residual matrix.
    ///
    /// This is a separate observation contract from
    /// [`Self::try_visit_untrusted_final_logits`]. It emits borrowed matrices in
    /// batch/layer order and retains no activation history. The transcript binds
    /// only the block-output stream; it does not claim final-logit coverage or
    /// campaign admission. A built-in sealed-backend execution is still required
    /// before this evidence can support qualification.
    ///
    /// # Errors
    /// Returns [`Qwen35ExecutionVisitError::Runtime`] for invalid input, execution,
    /// identity, count, or transcript failures, and `Observer` if the caller
    /// rejects one block output.
    /// In-place changed weights fail provenance before input or observer effects.
    pub fn try_visit_untrusted_block_outputs<'batch, I, E>(
        &self,
        batches: I,
        observer: impl FnMut(Qwen35ExecutionBlockOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.try_visit_untrusted_block_outputs_with_states(batches, &[], observer, |_, _, _, _| {})
    }

    /// Internal research seam for block outputs plus selected recurrent states.
    ///
    /// State samples are borrowed synchronously and include batch, block, and
    /// token-row coordinates. This does not add state data to the public block
    /// transcript or qualify it for campaign admission.
    pub(crate) fn try_visit_untrusted_block_outputs_with_states<'batch, I, E>(
        &self,
        batches: I,
        state_positions: &[usize],
        mut observer: impl FnMut(Qwen35ExecutionBlockOutputBatch<'_>) -> Result<(), E>,
        mut state_observer: impl FnMut(u64, u32, usize, &[f32]),
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.require_loaded_weight_state()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let backend_before = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let mut token_hasher = blake3::Hasher::new_derive_key(TOKEN_STREAM_CONTEXT);
        let mut block_outputs = RuntimeBlockOutputsAccumulator::new();
        let mut batch_count = 0_u64;
        let mut token_count = 0_u64;

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
                .forward_with_block_and_state_observer(
                    tokens,
                    &mut cache,
                    state_positions,
                    |block_index, token_start, block_tokens, hidden_states| {
                        let token_start = u64::try_from(token_start).map_err(|_| {
                            NnError::ResourceExhausted(
                                "Qwen block token start exceeds u64".to_owned(),
                            )
                        })?;
                        block_outputs
                            .observe(
                                batch_count,
                                block_index,
                                token_start,
                                block_tokens.len(),
                                self.runner().hidden_size(),
                                hidden_states,
                            )
                            .map_err(|error| {
                                NnError::Provenance(format!(
                                    "Qwen runtime block-output evidence failed: {error}"
                                ))
                            })?;
                        observer(Qwen35ExecutionBlockOutputBatch {
                            batch_index: batch_count,
                            block_index,
                            token_start,
                            tokens: block_tokens,
                            hidden_size: self.runner().hidden_size(),
                            hidden_states,
                        })
                        .map_err(BlockObserverFailure::Observer)
                    },
                    |block_index, token_position, state| {
                        state_observer(batch_count, block_index, token_position, state);
                    },
                )
                .map_err(|error| match error {
                    super::qwen35::Qwen35TextForwardError::Runtime(error) => {
                        Qwen35ExecutionVisitError::Runtime(error)
                    }
                    super::qwen35::Qwen35TextForwardError::Observer(
                        BlockObserverFailure::Runtime(error),
                    ) => Qwen35ExecutionVisitError::Runtime(error),
                    super::qwen35::Qwen35TextForwardError::Observer(
                        BlockObserverFailure::Observer(error),
                    ) => Qwen35ExecutionVisitError::Observer(error),
                })?;
            let _ = output;
            hash_batch_tokens(&mut token_hasher, batch_count, tokens);
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
        }
        if batch_count == 0 {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Shape {
                expected: 1,
                got: 0,
            }));
        }
        token_hasher.update(&batch_count.to_le_bytes());
        token_hasher.update(&token_count.to_le_bytes());
        let block_outputs = block_outputs.finish().map_err(|error| {
            Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                "Qwen runtime block-output evidence failed: {error}"
            )))
        })?;
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
            block_output_digest: *block_outputs.digest(),
            final_logits_digest: [0; 32],
            scope_coverage: BLOCK_OUTPUT_COVERAGE,
            batch_count,
            token_count,
            block_observation_count: block_outputs.observation_count(),
            block_element_count: block_outputs.element_count(),
            logit_count: 0,
        };
        transcript.transcript_id = transcript
            .derive_id()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        Ok(transcript)
    }

    /// Re-execute tokens and require identical non-admissible block-output transcript bytes.
    pub fn reexecute_untrusted_block_outputs<'batch, I, E>(
        &self,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionBlockOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        let transcript = self.try_visit_untrusted_block_outputs(batches, observer)?;
        let actual = transcript
            .canonical_bytes()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        if actual != expected_canonical {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen block-output transcript differs from fresh runtime output".to_owned(),
            )));
        }
        Ok(transcript)
    }

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
    /// In-place changed weights fail provenance before input or observer effects.
    pub fn try_visit_untrusted_final_logits<'batch, I, E>(
        &self,
        batches: I,
        mut observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen35UntrustedRuntimeTranscript, Qwen35ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.require_loaded_weight_state()
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let backend_before = BackendIdentity::capture(self.runner().execution_backend())
            .map_err(Qwen35ExecutionVisitError::Runtime)?;
        let mut token_hasher = blake3::Hasher::new_derive_key(TOKEN_STREAM_CONTEXT);
        let mut runtime_final_logits = RuntimeFinalLogitsAccumulator::new();
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
            runtime_final_logits.observe(logits).map_err(|error| {
                Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                    "Qwen runtime final-logit evidence failed: {error}"
                )))
            })?;
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
        let runtime_final_logits = runtime_final_logits.finish().map_err(|error| {
            Qwen35ExecutionVisitError::Runtime(NnError::Provenance(format!(
                "Qwen runtime final-logit evidence failed: {error}"
            )))
        })?;
        if runtime_final_logits.batch_count() != batch_count
            || runtime_final_logits.logit_count() != logit_count
        {
            return Err(Qwen35ExecutionVisitError::Runtime(NnError::Provenance(
                "Qwen runtime final-logit evidence counters differ".to_owned(),
            )));
        }

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
            final_logits_digest: *runtime_final_logits.digest(),
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

enum BlockObserverFailure<E> {
    Runtime(NnError),
    Observer(E),
}

impl<E> From<NnError> for BlockObserverFailure<E> {
    fn from(error: NnError) -> Self {
        Self::Runtime(error)
    }
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
