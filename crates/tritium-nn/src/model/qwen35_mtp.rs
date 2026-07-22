//! Receipt-gated Qwen3.5-family multi-token-prediction execution.
//!
//! Exact checkpoint weights remain structurally unverified after assembly. A
//! content-bound language-plus-MTP model promotes them into an executable runner
//! only after a prefill-plus-decode trace matches a pinned vLLM fixture oracle.
//! That oracle executes official `EagleProposer` first-pass input construction,
//! model/attention code, hidden rows, logits, and the full KV cache.

use std::sync::Arc;

use tritium_format::ModelId;

use super::qwen35::{Qwen35TextOutput, Qwen35TextRunner, RunnerIdentity};
use super::qwen35_mtp_oracle::{AuthorizedQwen35MtpStep, AuthorizedQwen35MtpTrace, first_argmax};
pub use super::qwen35_mtp_oracle::{Qwen35MtpOracleCoverageProfile, Qwen35MtpOracleEvidenceClass};
use crate::error::NnError;
use crate::layers::{
    Projection, Qwen35FullAttention, Qwen35FullAttentionCache, Qwen35FullAttentionWeights,
    SwiGluMlp,
};
use crate::ops::rmsnorm_zero_centered;

/// Stable reason the structurally complete MTP graph is not runnable.
pub const QWEN35_MTP_UNVERIFIED_REASON: &str =
    "official Qwen3.5/Qwen3.6 MTP parity receipt is missing";

/// Latest authoritative Qwen3.5 MTP implementation at the 2026-07-14 research cutoff.
pub const QWEN35_MTP_VLLM_ORACLE_REVISION: &str = "36484e464a6cf763c5b4c8af7be8e19df324997a";

/// SHA-256 of pinned `vllm/model_executor/models/qwen3_5_mtp.py`.
pub const QWEN35_MTP_VLLM_SOURCE_SHA256: &str =
    "87cff3c5ca1c9c6dde87e69b298697d342fb50f1a926e16b59d9e9a9fadb3cc8";

const MTP_OBSERVED_DIGEST_CONTEXT: &str = "tritium qwen3.5 mtp observed parity v1";

/// Opaque evidence that one content-derived source matched the pinned MTP oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MtpParityReceipt {
    source_model_id: ModelId,
    oracle_body_id: [u8; 32],
    oracle_manifest_id: [u8; 32],
    observed_digest: [u8; 32],
    coverage_profile: Qwen35MtpOracleCoverageProfile,
    evidence_class: Qwen35MtpOracleEvidenceClass,
    steps: usize,
    tolerance: f32,
    max_absolute_error: f32,
}

impl Qwen35MtpParityReceipt {
    /// Content-derived source identity exercised by the trace.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Exact pinned vLLM source revision.
    #[must_use]
    pub const fn oracle_revision(&self) -> &'static str {
        QWEN35_MTP_VLLM_ORACLE_REVISION
    }

    /// Number of target/MTP transactions in the trace.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Largest absolute numeric error across hidden, logit, key, and value lanes.
    #[must_use]
    pub const fn max_absolute_error(&self) -> f32 {
        self.max_absolute_error
    }

    /// Preregistered absolute comparison tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Domain-separated digest of the exact compiled-authorized oracle body.
    #[must_use]
    pub const fn oracle_body_id(&self) -> [u8; 32] {
        self.oracle_body_id
    }

    /// Identity of the pinned oracle implementation/runtime manifest.
    #[must_use]
    pub const fn oracle_manifest_id(&self) -> [u8; 32] {
        self.oracle_manifest_id
    }

    /// Digest of the same inputs and Tritium-observed outputs.
    #[must_use]
    pub const fn observed_digest(&self) -> [u8; 32] {
        self.observed_digest
    }

    /// Coverage contract proven by this artifact.
    #[must_use]
    pub const fn coverage_profile(&self) -> Qwen35MtpOracleCoverageProfile {
        self.coverage_profile
    }

    /// Strength of evidence carried by this artifact.
    #[must_use]
    pub const fn evidence_class(&self) -> Qwen35MtpOracleEvidenceClass {
        self.evidence_class
    }

    /// Whether this receipt carries the compiled production checkpoint profile.
    ///
    /// Safe code can only obtain such a receipt after an exact body/source/
    /// manifest tuple appears in the private compiled authorization ledger and
    /// every registered lane passes the frozen tolerance.
    #[must_use]
    pub const fn qualifies_for_production(&self) -> bool {
        matches!(
            self.coverage_profile,
            Qwen35MtpOracleCoverageProfile::ProductionCheckpointPrefillDecode
        ) && matches!(
            self.evidence_class,
            Qwen35MtpOracleEvidenceClass::ProductionCheckpoint
        ) && self.steps == 2
            && self.max_absolute_error <= self.tolerance
    }
}

/// Verification state of the bundled Qwen3.5-family MTP graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpStatus {
    /// Weights and alignment are structural only; execution is unavailable.
    Unverified,
}

impl Qwen35MtpStatus {
    /// Stable explanation for the current non-runnable state.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Unverified => QWEN35_MTP_UNVERIFIED_REASON,
        }
    }
}

/// Shifted token IDs, target positions, and borrowed target hidden rows for one
/// future MTP step.
#[derive(Debug)]
pub struct Qwen35MtpInputPlan<'target> {
    shifted_token_ids: Vec<u32>,
    positions: Vec<usize>,
    target_hidden_states: &'target [f32],
}

impl Qwen35MtpInputPlan<'_> {
    /// Shifted IDs `[x1, ..., xN-1, sampled_next]`.
    #[must_use]
    pub fn shifted_token_ids(&self) -> &[u32] {
        &self.shifted_token_ids
    }

    /// Absolute target positions retained without the token shift.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Final-normalized target hidden rows `[sequence, hidden]`.
    ///
    /// This slice borrows the exact language output used to derive the plan;
    /// alignment cannot substitute or mutate a second hidden-state buffer.
    #[must_use]
    pub const fn target_hidden_states(&self) -> &[f32] {
        self.target_hidden_states
    }
}

/// Raw weights for the MTP drafter's one forced full-attention decoder layer.
///
/// This group owns 11 checkpoint tensors: two decoder norms, six attention
/// tensors, and three SwiGLU projections.
#[allow(missing_debug_implementations)]
pub struct Qwen35MtpLayerWeights {
    input_norm: Vec<f32>,
    attention: Qwen35FullAttentionWeights,
    post_attention_norm: Vec<f32>,
    mlp: SwiGluMlp,
}

impl Qwen35MtpLayerWeights {
    /// Collect the unbound full-attention MTP decoder layer.
    #[must_use]
    pub fn new(
        input_norm: Vec<f32>,
        attention: Qwen35FullAttentionWeights,
        post_attention_norm: Vec<f32>,
        mlp: SwiGluMlp,
    ) -> Self {
        Self {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        }
    }
}

/// The exact 15 checkpoint tensors retained by the bundled MTP graph.
///
/// There is intentionally no token embedding or language head field: the
/// verified executor reuses the target language model's distinct
/// embedding and untied head without allocating copies.
#[allow(missing_debug_implementations)]
pub struct Qwen35MtpWeights {
    pre_fc_norm_embedding: Vec<f32>,
    pre_fc_norm_hidden: Vec<f32>,
    fc: Projection,
    layer: Qwen35MtpLayerWeights,
    final_norm: Vec<f32>,
}

impl Qwen35MtpWeights {
    /// Collect the complete unbound one-layer MTP graph.
    #[must_use]
    pub fn new(
        pre_fc_norm_embedding: Vec<f32>,
        pre_fc_norm_hidden: Vec<f32>,
        fc: Projection,
        layer: Qwen35MtpLayerWeights,
        final_norm: Vec<f32>,
    ) -> Self {
        Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            final_norm,
        }
    }
}

#[derive(Debug)]
struct MtpRunnerIdentity;

#[allow(missing_debug_implementations)]
struct Qwen35MtpCore {
    runner_identity: Arc<RunnerIdentity>,
    mtp_identity: Arc<MtpRunnerIdentity>,
    hidden_size: usize,
    vocab_size: usize,
    max_context: usize,
    rms_norm_eps: f32,
    pre_fc_norm_embedding: Vec<f32>,
    pre_fc_norm_hidden: Vec<f32>,
    fc: Projection,
    input_norm: Vec<f32>,
    attention: Qwen35FullAttention,
    post_attention_norm: Vec<f32>,
    mlp: SwiGluMlp,
    final_norm: Vec<f32>,
}

/// Opaque, structurally complete one-layer Qwen3.5/Qwen3.6 MTP graph.
///
/// Assembly validates weights and exact architecture but exposes no execution.
/// Only a content-bound official-oracle comparison can consume this value and
/// return [`Qwen35MtpRunner`].
#[allow(missing_debug_implementations)]
pub struct UnverifiedQwen35Mtp {
    core: Arc<Qwen35MtpCore>,
}

impl UnverifiedQwen35Mtp {
    /// Assemble and validate the exact one-layer, shared-embedding MTP graph.
    ///
    /// The nested layer owns 11 checkpoint tensors; the two pre-fusion norms,
    /// fusion projection, and final norm bring the exact total to 15. Every MTP
    /// projection must use the target language runner's activation arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for a non-one-layer/dedicated-embedding
    /// contract or mixed projection arithmetic, [`NnError::Shape`] for any
    /// tensor geometry mismatch, or [`NnError::Backend`] for non-finite retained
    /// parameters.
    pub fn new(target: &Qwen35TextRunner, weights: Qwen35MtpWeights) -> Result<Self, NnError> {
        let config = target.config();
        if config.model_type != "qwen3_5_text" {
            return Err(invalid_config(
                "MTP assembly requires a Qwen3.5-family text configuration",
            ));
        }
        if config.mtp.num_hidden_layers != 1 {
            return Err(invalid_config(
                "MTP assembly requires exactly one hidden layer",
            ));
        }
        if config.mtp.dedicated_embeddings {
            return Err(invalid_config(
                "MTP assembly requires the shared language embedding",
            ));
        }
        if config.tied_embeddings {
            return Err(invalid_config(
                "MTP assembly requires the checkpoint's distinct untied language head",
            ));
        }

        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let intermediate_size = axis(config.intermediate_size, "intermediate_size")?;
        let vocab_size = axis(config.vocab_size, "vocab_size")?;
        let fused_width = hidden_size
            .checked_mul(2)
            .ok_or_else(|| invalid_config("MTP fusion width overflow"))?;

        validate_vector(
            &weights.pre_fc_norm_embedding,
            hidden_size,
            "pre-fc embedding norm",
        )?;
        validate_vector(
            &weights.pre_fc_norm_hidden,
            hidden_size,
            "pre-fc hidden norm",
        )?;
        validate_projection(&weights.fc, hidden_size, fused_width, "fusion projection")?;
        validate_vector(&weights.layer.input_norm, hidden_size, "layer input norm")?;
        let attention_mode = weights.layer.attention.validate_for_config(config)?;
        validate_vector(
            &weights.layer.post_attention_norm,
            hidden_size,
            "layer post-attention norm",
        )?;
        let mlp_mode = weights
            .layer
            .mlp
            .validate_for_geometry(hidden_size, intermediate_size)?;
        validate_vector(&weights.final_norm, hidden_size, "final MTP norm")?;

        let target_mode = target.activation_mode();
        if weights.fc.activation_mode() != target_mode
            || attention_mode != target_mode
            || mlp_mode != target_mode
        {
            return Err(invalid_config(
                "Qwen3.5 MTP projections must match target activation arithmetic",
            ));
        }

        let Qwen35MtpWeights {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            final_norm,
        } = weights;
        let Qwen35MtpLayerWeights {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        } = layer;
        let attention = Qwen35FullAttention::new(config, attention)?;

        Ok(Self {
            core: Arc::new(Qwen35MtpCore {
                runner_identity: Arc::clone(target.identity()),
                mtp_identity: Arc::new(MtpRunnerIdentity),
                hidden_size,
                vocab_size,
                max_context: target.max_context(),
                rms_norm_eps: target.rms_norm_eps(),
                pre_fc_norm_embedding,
                pre_fc_norm_hidden,
                fc,
                input_norm,
                attention,
                post_attention_norm,
                mlp,
                final_norm,
            }),
        })
    }

    /// Verification state; always [`Qwen35MtpStatus::Unverified`] for this type.
    #[must_use]
    pub const fn status(&self) -> Qwen35MtpStatus {
        Qwen35MtpStatus::Unverified
    }

    #[cfg(test)]
    fn assume_verified_for_test(self) -> Qwen35MtpRunner {
        Qwen35MtpRunner { core: self.core }
    }

    /// Align one bound target-language output with shifted MTP token IDs.
    ///
    /// Token IDs, positions, and hidden rows all derive exclusively from the
    /// language output. Outputs minted by another runner are rejected even when
    /// their geometry is identical.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for a hidden-width mismatch,
    /// [`NnError::MissingTensor`] for an out-of-vocabulary sampled ID, or
    /// [`NnError::Provenance`] for a foreign output, or
    /// [`NnError::ResourceExhausted`] for position/allocation overflow.
    pub fn align_step<'target>(
        &self,
        target_output: &'target Qwen35TextOutput,
        sampled_next: u32,
    ) -> Result<Qwen35MtpInputPlan<'target>, NnError> {
        self.core.align_step(target_output, sampled_next)
    }

    pub(super) fn verify_trace(
        &self,
        target: &Qwen35TextRunner,
        trace: AuthorizedQwen35MtpTrace,
    ) -> Result<(Qwen35MtpRunner, Qwen35MtpParityReceipt), NnError> {
        if !Arc::ptr_eq(&self.core.runner_identity, target.identity()) {
            return Err(NnError::Provenance(
                "Qwen3.5 MTP oracle target belongs to a different language runner".to_owned(),
            ));
        }
        if trace.max_context() > self.core.max_context {
            return Err(NnError::Shape {
                expected: self.core.max_context,
                got: trace.max_context(),
            });
        }

        let runner = Qwen35MtpRunner {
            core: Arc::clone(&self.core),
        };
        let mut target_cache = target.new_cache(trace.max_context())?;
        let mut mtp_cache = runner.new_cache(trace.max_context())?;
        let mut observed_hasher = observed_hasher(&trace);
        let mut max_absolute_error = 0.0_f32;

        for (index, step) in trace.steps().iter().enumerate() {
            hash_step_inputs(&mut observed_hasher, step);
            let target_output = target.forward(step.token_ids(), &mut target_cache)?;
            compare_lanes(
                index,
                "target hidden states",
                target_output.final_hidden_states(),
                step.expected_target_hidden(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            compare_lanes(
                index,
                "target last logits",
                target_output.last_logits(),
                step.expected_target_logits(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            let selected = first_argmax(target_output.last_logits()).ok_or_else(|| {
                NnError::Verification(format!(
                    "Qwen3.5 MTP oracle step {index} target logits cannot select a token"
                ))
            })?;
            if selected != step.sampled_next() as usize {
                return Err(NnError::Verification(format!(
                    "Qwen3.5 MTP oracle step {index} target selected token {selected}, expected {}",
                    step.sampled_next()
                )));
            }
            let output =
                runner.forward(target, &target_output, step.sampled_next(), &mut mtp_cache)?;
            compare_lanes(
                index,
                "MTP hidden states",
                output.final_hidden_states(),
                step.expected_mtp_hidden(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            compare_lanes(
                index,
                "last logits",
                output.last_logits(),
                step.expected_logits(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            compare_lanes(
                index,
                "cache keys",
                mtp_cache.keys(),
                step.expected_cache_keys(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            compare_lanes(
                index,
                "cache values",
                mtp_cache.values(),
                step.expected_cache_values(),
                trace.tolerance(),
                &mut max_absolute_error,
            )?;
            hash_f32s(&mut observed_hasher, target_output.final_hidden_states());
            hash_f32s(&mut observed_hasher, target_output.last_logits());
            hash_f32s(&mut observed_hasher, output.final_hidden_states());
            hash_f32s(&mut observed_hasher, output.last_logits());
            hash_f32s(&mut observed_hasher, mtp_cache.keys());
            hash_f32s(&mut observed_hasher, mtp_cache.values());
        }

        let receipt = Qwen35MtpParityReceipt {
            source_model_id: trace.source_model_id(),
            oracle_body_id: trace.body_id(),
            oracle_manifest_id: trace.oracle_manifest_id(),
            observed_digest: *observed_hasher.finalize().as_bytes(),
            coverage_profile: trace.coverage_profile(),
            evidence_class: trace.evidence_class(),
            steps: trace.steps().len(),
            tolerance: trace.tolerance(),
            max_absolute_error,
        };
        Ok((runner, receipt))
    }
}

impl Qwen35MtpCore {
    fn align_step<'target>(
        &self,
        target_output: &'target Qwen35TextOutput,
        sampled_next: u32,
    ) -> Result<Qwen35MtpInputPlan<'target>, NnError> {
        if !Arc::ptr_eq(&self.runner_identity, target_output.runner_identity()) {
            return Err(NnError::Provenance(
                "Qwen3.5 MTP target output belongs to a different language runner".to_owned(),
            ));
        }
        if target_output.hidden_size() != self.hidden_size {
            return Err(NnError::Shape {
                expected: self.hidden_size,
                got: target_output.hidden_size(),
            });
        }
        if usize::try_from(sampled_next).map_or(true, |token| token >= self.vocab_size) {
            return Err(NnError::MissingTensor(format!(
                "Qwen3.5 MTP token id {sampled_next} is outside vocabulary size {}",
                self.vocab_size
            )));
        }
        build_input_plan(
            target_output.input_token_ids(),
            sampled_next,
            target_output.position_start(),
            target_output.final_hidden_states(),
        )
    }
}

/// KV state for one promoted MTP runner.
#[derive(Debug)]
pub struct Qwen35MtpCache {
    mtp_identity: Arc<MtpRunnerIdentity>,
    attention: Qwen35FullAttentionCache,
}

impl Qwen35MtpCache {
    /// Number of committed draft positions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.attention.len()
    }

    /// Whether the MTP cache is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.attention.is_empty()
    }

    /// Maximum committed positions retained by this cache.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.attention.max_context()
    }

    /// Committed rotated keys in `[sequence, kv_head, head_dim]` order.
    #[must_use]
    pub(crate) fn keys(&self) -> &[f32] {
        self.attention.keys()
    }

    /// Committed values in `[sequence, kv_head, head_dim]` order.
    #[must_use]
    pub(crate) fn values(&self) -> &[f32] {
        self.attention.values()
    }

    /// Clear the stream without releasing capacity.
    pub fn reset(&mut self) {
        self.attention.reset();
    }

    /// Discard unaccepted speculative suffix rows.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] when `accepted_len` exceeds the committed cursor.
    pub fn rollback_to(&mut self, accepted_len: usize) -> Result<(), NnError> {
        self.attention.preflight_rollback_to(accepted_len)?;
        self.attention.rollback_to(accepted_len);
        Ok(())
    }
}

/// Successful promoted-MTP forward result.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MtpOutput {
    position_start: usize,
    sequence: usize,
    hidden_size: usize,
    final_hidden_states: Vec<f32>,
    last_logits: Vec<f32>,
}

impl Qwen35MtpOutput {
    /// First absolute target position represented by this output.
    #[must_use]
    pub const fn position_start(&self) -> usize {
        self.position_start
    }

    /// Number of output rows.
    #[must_use]
    pub const fn sequence(&self) -> usize {
        self.sequence
    }

    /// Output hidden width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Final-normalized MTP rows `[sequence, hidden]`.
    #[must_use]
    pub fn final_hidden_states(&self) -> &[f32] {
        &self.final_hidden_states
    }

    /// Shared untied-head logits for the last MTP row.
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }
}

/// Executable MTP graph produced only by a successful source-bound parity trace.
#[allow(missing_debug_implementations)]
pub struct Qwen35MtpRunner {
    core: Arc<Qwen35MtpCore>,
}

impl Qwen35MtpRunner {
    /// Allocate a cache bound to this promoted runner.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] for zero or excessive capacity, or a cache
    /// allocation error.
    pub fn new_cache(&self, max_context: usize) -> Result<Qwen35MtpCache, NnError> {
        if max_context == 0 || max_context > self.core.max_context {
            return Err(NnError::Shape {
                expected: self.core.max_context,
                got: max_context,
            });
        }
        Ok(Qwen35MtpCache {
            mtp_identity: Arc::clone(&self.core.mtp_identity),
            attention: self.core.attention.new_cache(max_context)?,
        })
    }

    /// Execute one target-aligned prefill or cached continuation transaction.
    ///
    /// Shared embedding and untied-head weights are borrowed from `target`.
    /// Positions must exactly continue the private MTP cache cursor. Any failure
    /// after the attention append rolls the cache back to its prior watermark.
    ///
    /// # Errors
    /// Returns a shape, provenance, capacity, or numeric execution error without
    /// publishing a partial output or cache suffix.
    pub fn forward(
        &self,
        target: &Qwen35TextRunner,
        target_output: &Qwen35TextOutput,
        sampled_next: u32,
        cache: &mut Qwen35MtpCache,
    ) -> Result<Qwen35MtpOutput, NnError> {
        if !Arc::ptr_eq(&self.core.runner_identity, target.identity()) {
            return Err(NnError::Provenance(
                "Qwen3.5 MTP executor received a foreign language runner".to_owned(),
            ));
        }
        if !Arc::ptr_eq(&self.core.mtp_identity, &cache.mtp_identity) {
            return Err(NnError::Provenance(
                "Qwen3.5 MTP cache belongs to a different promoted runner".to_owned(),
            ));
        }
        let plan = self.core.align_step(target_output, sampled_next)?;
        let sequence = plan.positions.len();
        let base = cache.len();
        let new_len = base.checked_add(sequence).ok_or(NnError::Shape {
            expected: cache.max_context(),
            got: usize::MAX,
        })?;
        if new_len > cache.max_context() {
            return Err(NnError::Shape {
                expected: cache.max_context(),
                got: new_len,
            });
        }
        for (offset, &position) in plan.positions.iter().enumerate() {
            let expected = base.checked_add(offset).ok_or(NnError::Shape {
                expected: base,
                got: usize::MAX,
            })?;
            if position != expected {
                return Err(NnError::Verification(format!(
                    "Qwen3.5 MTP position {position} does not continue cache cursor {expected}"
                )));
            }
        }

        let hidden_len = checked_mul(sequence, self.core.hidden_size, "hidden-state buffer")?;
        if plan.target_hidden_states.len() != hidden_len {
            return Err(NnError::Shape {
                expected: hidden_len,
                got: plan.target_hidden_states.len(),
            });
        }
        let fused_width = checked_mul(self.core.hidden_size, 2, "fusion width")?;
        let fused_len = checked_mul(sequence, fused_width, "fusion buffer")?;
        let logits_len = self.core.vocab_size;

        // All fallible allocations occur before attention mutates its KV cache.
        let mut embedding = zeroed_scratch(hidden_len, "shared embedding")?;
        let mut normalized_embedding = zeroed_scratch(hidden_len, "normalized embedding")?;
        let mut normalized_hidden = zeroed_scratch(hidden_len, "normalized target hidden")?;
        let mut fused = zeroed_scratch(fused_len, "fusion input")?;
        let mut residual = zeroed_scratch(hidden_len, "fusion output")?;
        let mut normalized = zeroed_scratch(hidden_len, "decoder normalized state")?;
        let mut branch = zeroed_scratch(hidden_len, "decoder branch")?;
        let mut final_hidden_states = zeroed_scratch(hidden_len, "final hidden states")?;
        let mut last_logits = zeroed_scratch(logits_len, "last logits")?;

        target.gather_shared_embedding(plan.shifted_token_ids(), &mut embedding)?;
        normalize_rows(
            &embedding,
            &self.core.pre_fc_norm_embedding,
            self.core.rms_norm_eps,
            self.core.hidden_size,
            &mut normalized_embedding,
        )?;
        normalize_rows(
            plan.target_hidden_states(),
            &self.core.pre_fc_norm_hidden,
            self.core.rms_norm_eps,
            self.core.hidden_size,
            &mut normalized_hidden,
        )?;
        for row in 0..sequence {
            let hidden_start = row * self.core.hidden_size;
            let fused_start = row * fused_width;
            fused[fused_start..fused_start + self.core.hidden_size].copy_from_slice(
                &normalized_embedding[hidden_start..hidden_start + self.core.hidden_size],
            );
            fused[fused_start + self.core.hidden_size..fused_start + fused_width].copy_from_slice(
                &normalized_hidden[hidden_start..hidden_start + self.core.hidden_size],
            );
        }
        self.core
            .fc
            .forward(target.execution_backend(), &fused, sequence, &mut residual)?;
        normalize_rows(
            &residual,
            &self.core.input_norm,
            self.core.rms_norm_eps,
            self.core.hidden_size,
            &mut normalized,
        )?;
        self.core.attention.forward(
            target.execution_backend(),
            &normalized,
            plan.positions(),
            &mut cache.attention,
            &mut branch,
        )?;

        let post_attention = (|| {
            add_in_place(&mut residual, &branch);
            normalize_rows(
                &residual,
                &self.core.post_attention_norm,
                self.core.rms_norm_eps,
                self.core.hidden_size,
                &mut normalized,
            )?;
            self.core.mlp.forward(
                target.execution_backend(),
                &normalized,
                sequence,
                &mut branch,
            )?;
            add_in_place(&mut residual, &branch);
            normalize_rows(
                &residual,
                &self.core.final_norm,
                self.core.rms_norm_eps,
                self.core.hidden_size,
                &mut final_hidden_states,
            )?;
            let last_start = checked_mul(sequence - 1, self.core.hidden_size, "last hidden row")?;
            target.project_shared_head(
                &final_hidden_states[last_start..last_start + self.core.hidden_size],
                1,
                &mut last_logits,
            )?;
            if final_hidden_states
                .iter()
                .chain(&last_logits)
                .any(|value| !value.is_finite())
            {
                return Err(NnError::Verification(
                    "Qwen3.5 MTP forward produced a non-finite value".to_owned(),
                ));
            }
            Ok(())
        })();
        if let Err(error) = post_attention {
            cache.attention.rollback_to(base);
            return Err(error);
        }

        Ok(Qwen35MtpOutput {
            position_start: base,
            sequence,
            hidden_size: self.core.hidden_size,
            final_hidden_states,
            last_logits,
        })
    }
}

fn build_input_plan<'target>(
    target_token_ids: &[u32],
    sampled_next: u32,
    position_start: usize,
    target_hidden_states: &'target [f32],
) -> Result<Qwen35MtpInputPlan<'target>, NnError> {
    let last_offset = target_token_ids
        .len()
        .checked_sub(1)
        .ok_or(NnError::Shape {
            expected: 1,
            got: 0,
        })?;
    position_start.checked_add(last_offset).ok_or_else(|| {
        NnError::ResourceExhausted("Qwen3.5 MTP position range overflow".to_owned())
    })?;

    let mut shifted_token_ids = Vec::new();
    shifted_token_ids
        .try_reserve_exact(target_token_ids.len())
        .map_err(|error| {
            NnError::ResourceExhausted(format!("allocate Qwen3.5 MTP shifted-token plan: {error}"))
        })?;
    shifted_token_ids.extend_from_slice(&target_token_ids[1..]);
    shifted_token_ids.push(sampled_next);

    let mut positions = Vec::new();
    positions
        .try_reserve_exact(target_token_ids.len())
        .map_err(|error| {
            NnError::ResourceExhausted(format!("allocate Qwen3.5 MTP position plan: {error}"))
        })?;
    for offset in 0..target_token_ids.len() {
        positions.push(position_start + offset);
    }

    Ok(Qwen35MtpInputPlan {
        shifted_token_ids,
        positions,
        target_hidden_states,
    })
}

fn normalize_rows(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    hidden_size: usize,
    output: &mut [f32],
) -> Result<(), NnError> {
    if input.len() != output.len() {
        return Err(NnError::Shape {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (source, destination) in input
        .chunks_exact(hidden_size)
        .zip(output.chunks_exact_mut(hidden_size))
    {
        rmsnorm_zero_centered(source, weights, epsilon, destination)?;
    }
    Ok(())
}

fn add_in_place(residual: &mut [f32], branch: &[f32]) {
    debug_assert_eq!(residual.len(), branch.len());
    for (residual, &branch) in residual.iter_mut().zip(branch) {
        *residual += branch;
    }
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| NnError::ResourceExhausted(format!("Qwen3.5 MTP {name} extent overflow")))
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::ResourceExhausted(format!(
            "allocate Qwen3.5 MTP {name} for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn observed_hasher(trace: &AuthorizedQwen35MtpTrace) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new_derive_key(MTP_OBSERVED_DIGEST_CONTEXT);
    hasher.update(trace.source_model_id().as_bytes());
    hasher.update(&trace.body_id());
    hasher.update(&trace.oracle_manifest_id());
    hasher.update(QWEN35_MTP_VLLM_ORACLE_REVISION.as_bytes());
    hasher.update(QWEN35_MTP_VLLM_SOURCE_SHA256.as_bytes());
    hasher.update(&trace.tolerance().to_bits().to_le_bytes());
    hasher.update(&(trace.max_context() as u64).to_le_bytes());
    hasher.update(&(trace.hidden_size() as u64).to_le_bytes());
    hasher.update(&(trace.vocab_size() as u64).to_le_bytes());
    hasher.update(&(trace.kv_width() as u64).to_le_bytes());
    hasher.update(&(trace.steps().len() as u64).to_le_bytes());
    hasher.update(&(trace.coverage_profile() as u16).to_le_bytes());
    hasher.update(&(trace.evidence_class() as u16).to_le_bytes());
    hasher
}

fn hash_step_inputs(hasher: &mut blake3::Hasher, step: &AuthorizedQwen35MtpStep) {
    hasher.update(&(step.token_ids().len() as u64).to_le_bytes());
    for token in step.token_ids() {
        hasher.update(&token.to_le_bytes());
    }
    hasher.update(&step.sampled_next().to_le_bytes());
}

fn hash_f32s(hasher: &mut blake3::Hasher, values: &[f32]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn compare_lanes(
    step: usize,
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
    max_absolute_error: &mut f32,
) -> Result<(), NnError> {
    if actual.len() != expected.len() {
        return Err(NnError::Shape {
            expected: expected.len(),
            got: actual.len(),
        });
    }
    for (lane, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(NnError::Verification(format!(
                "Qwen3.5 MTP oracle step {step} {name} lane {lane} is non-finite"
            )));
        }
        let error = (actual - expected).abs();
        *max_absolute_error = max_absolute_error.max(error);
        if error > tolerance {
            return Err(NnError::Verification(format!(
                "Qwen3.5 MTP oracle step {step} {name} lane {lane} error {error} exceeds {tolerance}"
            )));
        }
    }
    Ok(())
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
    name: &str,
) -> Result<(), NnError> {
    if projection.n_out() != expected_out {
        return Err(NnError::Shape {
            expected: expected_out,
            got: projection.n_out(),
        });
    }
    if projection.k_in() != expected_in {
        return Err(NnError::Shape {
            expected: expected_in,
            got: projection.k_in(),
        });
    }
    projection
        .validate_retained_parameters()
        .map_err(|error| match error {
            NnError::Backend(message) => NnError::Backend(format!("Qwen3.5 MTP {name}: {message}")),
            other => other,
        })
}

fn validate_vector(values: &[f32], expected: usize, name: &str) -> Result<(), NnError> {
    if values.len() != expected {
        return Err(NnError::Shape {
            expected,
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NnError::Backend(format!(
            "Qwen3.5 MTP {name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn axis(value: u32, name: &str) -> Result<usize, NnError> {
    if value == 0 {
        return Err(invalid_config(format!(
            "Qwen3.5 MTP {name} must be non-zero"
        )));
    }
    usize::try_from(value)
        .map_err(|_| invalid_config(format!("Qwen3.5 MTP {name} does not fit usize")))
}

fn invalid_config(message: impl Into<String>) -> NnError {
    NnError::MissingConfig(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use tritium_core::{GemmShape, TernaryFormat, Trit};
    use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

    use super::{
        Qwen35MtpLayerWeights, Qwen35MtpParityReceipt, Qwen35MtpWeights, UnverifiedQwen35Mtp,
        build_input_plan,
    };
    use crate::layers::{
        DenseLinear, Projection, Qwen35FullAttentionWeights, SwiGluMlp, TernaryLinear,
        TokenEmbedding,
    };
    use crate::model::{Qwen35MtpOracleCoverageProfile, Qwen35MtpOracleEvidenceClass};
    use crate::model::{
        Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextRunner, Qwen35TextWeights,
    };
    use crate::qwen35_config::{
        Qwen35DeltaNetConfig, Qwen35Dtype, Qwen35FullAttentionConfig, Qwen35LayerType,
        Qwen35MtpConfig, Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig,
        Qwen35RopeType, Qwen35TextConfig,
    };

    #[test]
    fn empty_input_plan_fails_closed() {
        let hidden = [];
        let error = build_input_plan(&[], 3, 0, &hidden).unwrap_err();
        assert!(matches!(error, crate::NnError::Shape { .. }));
    }

    #[test]
    fn position_overflow_fails_closed() {
        let hidden = [0.0];
        let error = build_input_plan(&[1, 2], 3, usize::MAX, &hidden).unwrap_err();
        assert!(matches!(error, crate::NnError::ResourceExhausted(_)));
    }

    #[derive(Debug)]
    struct SwitchBackend {
        cpu: tritium_cpu::CpuBackend,
        fail: Arc<AtomicBool>,
    }

    impl TernaryBackend for SwitchBackend {
        fn device_id(&self) -> &str {
            "qwen35-mtp-switch"
        }

        fn capabilities(&self) -> DeviceCaps {
            self.cpu.capabilities()
        }

        fn upload_weights(
            &self,
            packed: &[u8],
            shape: GemmShape,
            format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            self.cpu.upload_weights(packed, shape, format)
        }

        fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
            if self.fail.load(Ordering::SeqCst) {
                parameters.out.fill(1234.0);
                Err(BackendError::Backend(
                    "intentional shared-head failure".to_owned(),
                ))
            } else {
                self.cpu.mpgemm(parameters)
            }
        }
    }

    const H: usize = 4;
    const I: usize = 8;
    const V: usize = 7;

    fn config() -> Qwen35TextConfig {
        Qwen35TextConfig {
            model_type: "qwen3_5_text".to_owned(),
            num_hidden_layers: 1,
            hidden_size: H as u32,
            intermediate_size: I as u32,
            vocab_size: V as u32,
            max_position_embeddings: 16,
            full_attention_interval: 1,
            layer_types: vec![Qwen35LayerType::FullAttention],
            full_attention: Qwen35FullAttentionConfig {
                num_heads: 2,
                num_key_value_heads: 1,
                head_dim: 2,
                bias: false,
                dropout: 0.0,
                output_gate: Qwen35OutputGate::Sigmoid,
                norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
            },
            delta_net: Qwen35DeltaNetConfig {
                conv_kernel_dim: 4,
                num_key_heads: 1,
                num_value_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
                state_arithmetic_dtype: Qwen35Dtype::Float32,
                output_gate: Qwen35OutputGate::Swish,
                gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
            },
            rope: Qwen35RopeConfig {
                theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rotary_dim: 2,
                rope_type: Qwen35RopeType::Default,
                mrope_interleaved: true,
                mrope_section: [1, 0, 0],
            },
            rms_norm_eps: 1e-6,
            source_dtype: Qwen35Dtype::Bfloat16,
            use_cache: true,
            tied_embeddings: false,
            mtp: Qwen35MtpConfig {
                num_hidden_layers: 1,
                dedicated_embeddings: false,
            },
        }
    }

    fn a8_projection(rows: usize, columns: usize) -> Projection {
        Projection::Dense(DenseLinear::new(vec![0.0; rows * columns], rows, columns).unwrap())
    }

    fn a8_attention() -> Qwen35FullAttentionWeights {
        Qwen35FullAttentionWeights::new(
            a8_projection(8, H),
            a8_projection(2, H),
            a8_projection(2, H),
            a8_projection(H, 4),
            vec![0.0; 2],
            vec![0.0; 2],
        )
    }

    fn a8_mlp() -> SwiGluMlp {
        SwiGluMlp::new(
            a8_projection(I, H),
            a8_projection(I, H),
            a8_projection(H, I),
        )
        .unwrap()
    }

    fn a8_runner(fail: Arc<AtomicBool>) -> Qwen35TextRunner {
        let backend = SwitchBackend {
            cpu: tritium_cpu::CpuBackend::new(),
            fail,
        };
        let head = Projection::Ternary(
            TernaryLinear::new(&backend, &[Trit::ZERO; V * H], V, H, 1.0).unwrap(),
        );
        let layer = Qwen35TextLayerWeights::new(
            vec![0.0; H],
            Qwen35TextMixerWeights::FullAttention(a8_attention()),
            vec![0.0; H],
            a8_mlp(),
        );
        Qwen35TextRunner::new(
            &config(),
            Qwen35TextWeights::new(
                TokenEmbedding::from_dense(vec![0.0; V * H], V, H).unwrap(),
                vec![layer],
                vec![0.0; H],
                head,
            ),
            Box::new(backend),
        )
        .unwrap()
    }

    fn a8_mtp(target: &Qwen35TextRunner) -> UnverifiedQwen35Mtp {
        UnverifiedQwen35Mtp::new(
            target,
            Qwen35MtpWeights::new(
                vec![0.0; H],
                vec![0.0; H],
                a8_projection(H, 2 * H),
                Qwen35MtpLayerWeights::new(vec![0.0; H], a8_attention(), vec![0.0; H], a8_mlp()),
                vec![0.0; H],
            ),
        )
        .unwrap()
    }

    #[test]
    fn only_production_checkpoint_receipt_qualifies_for_production() {
        let mut receipt = Qwen35MtpParityReceipt {
            source_model_id: tritium_format::ModelId::from_digest([1; 32]),
            oracle_body_id: [2; 32],
            oracle_manifest_id: [3; 32],
            observed_digest: [4; 32],
            coverage_profile: Qwen35MtpOracleCoverageProfile::ProductionCheckpointPrefillDecode,
            evidence_class: Qwen35MtpOracleEvidenceClass::ProductionCheckpoint,
            steps: 2,
            tolerance: 2.0e-3,
            max_absolute_error: 1.0e-3,
        };
        assert!(receipt.qualifies_for_production());
        receipt.evidence_class = Qwen35MtpOracleEvidenceClass::Fixture;
        assert!(!receipt.qualifies_for_production());
    }

    #[test]
    fn late_shared_head_failure_rolls_back_mtp_cache_and_retry_succeeds() {
        let fail = Arc::new(AtomicBool::new(false));
        let target = a8_runner(Arc::clone(&fail));
        let mtp = a8_mtp(&target).assume_verified_for_test();
        let mut target_cache = target.new_cache(8).unwrap();
        let target_prefill = target.forward(&[0, 0], &mut target_cache).unwrap();
        let mut mtp_cache = mtp.new_cache(8).unwrap();
        mtp.forward(&target, &target_prefill, 0, &mut mtp_cache)
            .unwrap();
        let original_keys = mtp_cache.keys().to_vec();
        let original_values = mtp_cache.values().to_vec();
        let target_decode = target.forward(&[0], &mut target_cache).unwrap();

        fail.store(true, Ordering::SeqCst);
        let error = mtp
            .forward(&target, &target_decode, 0, &mut mtp_cache)
            .unwrap_err();
        assert!(matches!(
            error,
            crate::NnError::Backend(reason) if reason.contains("shared-head failure")
        ));
        assert_eq!(mtp_cache.len(), 2);
        assert_eq!(mtp_cache.keys(), original_keys);
        assert_eq!(mtp_cache.values(), original_values);

        fail.store(false, Ordering::SeqCst);
        mtp.forward(&target, &target_decode, 0, &mut mtp_cache)
            .unwrap();
        assert_eq!(mtp_cache.len(), 3);
    }
}
