//! Common portable text execution. Internal numerical adapters, not resident clearance.
use super::CacheAdmission;
#[cfg(feature = "tokenizer")]
use super::{ModelWeights, Qwen35TextRunner, UnverifiedQwen35Mtp, host::HostExecutor};
use crate::{
    ArchSpec, ModelConfig, NnError, Qwen35CheckpointConfig, Qwen35LayerType, Qwen35MtpStatus,
};

/// Ordered token-mixer family in a loaded language graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Causal attention, including the graph's explicit gating/norm/RoPE semantics.
    Attention,
    /// Stateful gated DeltaNet with causal convolution and recurrent state.
    DeltaNet,
}

/// Actual architecture data consumed by the common portable runner.
/// Not a backend support, model completeness or qualification receipt.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ArchitectureSpec {
    /// Homogeneous graph with explicit variation and admitted embedding vocabulary.
    Standard {
        /// Parsed dimensions and numerical settings.
        config: ModelConfig,
        /// Actual norm/MLP/head variation admitted by the loader.
        variation: ArchSpec,
        /// Actual admitted token-table rows.
        vocab_size: usize,
    },
    /// Exact hybrid language/MTP contract. Vision remains explicitly deferred.
    Hybrid(Qwen35CheckpointConfig),
}
impl ArchitectureSpec {
    /// Manifest/config architecture name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Standard { config, .. } => &config.arch,
            Self::Hybrid(c) => &c.model_type,
        }
    }
    /// Actual decoder count.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        match self {
            Self::Standard { config, .. } => config.n_layers as usize,
            Self::Hybrid(c) => c.text.num_hidden_layers as usize,
        }
    }
    /// Admitted token vocabulary.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        match self {
            Self::Standard { vocab_size, .. } => *vocab_size,
            Self::Hybrid(c) => c.text.vocab_size as usize,
        }
    }
    /// Configured maximum stream context.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        match self {
            Self::Standard { config, .. } => config.n_ctx as usize,
            Self::Hybrid(c) => c.text.max_position_embeddings as usize,
        }
    }
    /// Actual ordered mixer schedule, without allocating a second layer table.
    pub fn layer_kinds(&self) -> impl Iterator<Item = LayerKind> + '_ {
        let (standard, hybrid) = match self {
            Self::Standard { .. } => (self.layer_count(), &[][..]),
            Self::Hybrid(c) => (0, c.text.layer_types.as_slice()),
        };
        std::iter::repeat_n(LayerKind::Attention, standard).chain(hybrid.iter().map(|kind| {
            match kind {
                Qwen35LayerType::FullAttention => LayerKind::Attention,
                Qwen35LayerType::DeltaNet => LayerKind::DeltaNet,
            }
        }))
    }
}

// Internal executor port: the common host implementation owns one weights
// aggregate/cache table and receives the runner's authoritative architecture.
// Backend-owned resident/batched/tree execution is a subsequent contract migration.
pub(super) trait Executor: Send {
    fn execution(&self) -> tritium_spec::DecoderExecution {
        tritium_spec::DecoderExecution::Host
    }
    fn resident_payload_bytes(&self) -> Option<usize> {
        None
    }
    fn forward(
        &mut self,
        architecture: &ArchitectureSpec,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError>;
    fn reset(&mut self);
    fn len(&self) -> usize;
    fn mtp_status(&self) -> Option<Qwen35MtpStatus> {
        None
    }
}

/// One loaded language-stream interface over the common portable executor seam.
/// Owns immutable graph weights and one private cache. No concrete backend is named.
/// Native resident/batching/draft/tree/full-memory qualification remains separate.
#[allow(missing_debug_implementations)]
pub struct Runner {
    architecture: ArchitectureSpec,
    executor: Box<dyn Executor>,
    admission: Option<CacheAdmission>,
    #[cfg(feature = "tokenizer")]
    resources: Option<(
        super::ModelResourceAdmission,
        tritium_runtime::TensorStorageReservation,
        tritium_runtime::TensorPayloadBudget,
    )>,
}
impl Runner {
    /// Actual selected graph implementation, not a quality/performance verdict.
    #[must_use]
    pub fn execution(&self) -> tritium_spec::DecoderExecution {
        self.executor.execution()
    }
    /// Resident owned matrix/vector/KV/workspace payload, not physical VRAM.
    #[must_use]
    pub fn resident_payload_bytes(&self) -> Option<usize> {
        self.executor.resident_payload_bytes()
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn standard(
        config: ModelConfig,
        weights: ModelWeights,
        backend: Box<dyn tritium_spec::TernaryBackend>,
        variation: ArchSpec,
        admission: Option<CacheAdmission>,
    ) -> Result<Self, NnError> {
        let architecture = ArchitectureSpec::Standard {
            config,
            variation,
            vocab_size: weights.vocab,
        };
        let context = admission.map_or(architecture.max_context(), |a| a.limits().max_context);
        let max_segment = admission.map_or(context, |a| a.limits().max_segment_tokens);
        let executor =
            HostExecutor::standard(&architecture, weights, backend, context, max_segment)?;
        Ok(Self {
            architecture,
            executor,
            admission,
            resources: None,
        })
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn hybrid(
        config: Qwen35CheckpointConfig,
        runner: Qwen35TextRunner,
        mtp: UnverifiedQwen35Mtp,
        admission: Option<CacheAdmission>,
    ) -> Result<Self, NnError> {
        let capacity = admission.map_or(config.text.max_position_embeddings as usize, |a| {
            a.limits().max_context
        });
        let imported = runner.into_common_weights()?;
        let architecture = ArchitectureSpec::Hybrid(config);
        let executor = HostExecutor::hybrid(
            &architecture,
            imported.weights,
            imported.backend,
            capacity,
            admission.map_or(capacity, |a| a.limits().max_segment_tokens),
            mtp,
            imported.identity,
        )?;
        Ok(Self {
            architecture,
            executor,
            admission,
            resources: None,
        })
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn bind_resources(
        &mut self,
        admission: super::ModelResourceAdmission,
        reservation: tritium_runtime::TensorStorageReservation,
        payload: tritium_runtime::TensorPayloadBudget,
    ) {
        self.resources = Some((admission, reservation, payload));
    }
    /// Covered logical domains admitted before weight materialization, if selected.
    /// Absent for compatibility loading; never a physical-memory qualification.
    #[cfg(feature = "tokenizer")]
    #[must_use]
    pub fn resource_admission(&self) -> Option<super::ModelResourceAdmission> {
        self.resources.as_ref().map(|(admission, _, _)| *admission)
    }
    /// Actual immutable graph/configuration used by this stream.
    #[must_use]
    pub const fn architecture(&self) -> &ArchitectureSpec {
        &self.architecture
    }
    /// Actual selected stream capacity, which may be below source architecture capacity.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        match self.admission {
            Some(a) => a.limits().max_context,
            None => self.architecture.max_context(),
        }
    }
    /// Immutable selected stream policy and logical host-cache plan, if explicitly admitted.
    /// This does not qualify total model memory or backend residency.
    #[must_use]
    pub const fn cache_admission(&self) -> Option<CacheAdmission> {
        self.admission
    }
    /// Committed language cursor; cancellation/error does not advance it.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.executor.len()
    }
    /// MTP admission status; absent or Unverified does not grant draft execution.
    #[must_use]
    pub fn mtp_status(&self) -> Option<Qwen35MtpStatus> {
        self.executor.mtp_status()
    }
    /// Reset every mixer/cache and committed cursor, preserving reusable capacity.
    pub fn reset(&mut self) {
        self.executor.reset();
    }
    /// Initial contiguous prefill on an empty stream.
    /// # Errors
    /// Nonempty cache, invalid tokens/context or actual graph/backend errors.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, NnError> {
        if self.cache_len() != 0 {
            return Err(NnError::Backend("prefill requires an empty stream".into()));
        }
        self.check_extent(0, tokens.len())?;
        let positions = positions(0, tokens.len())?;
        self.forward(tokens, &positions)
    }
    /// One-token continuation of a committed prefix.
    /// # Errors
    /// Empty cache, invalid token/context or actual graph/backend errors.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, NnError> {
        let at = self.cache_len();
        if at == 0 {
            return Err(NnError::Backend("step requires a committed prefix".into()));
        }
        self.forward(&[token], &[at])
    }
    /// Compatibility forward with positions required to equal the committed interval.
    /// # Errors
    /// Invalid IDs/positions/context or actual graph/backend errors.
    pub fn forward(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<f32>, NnError> {
        self.forward_cancellable(tokens, positions, &|| false)?
            .ok_or_else(|| NnError::Backend("uncancelled executor returned no output".into()))
    }
    /// Cooperative transactional forward. Pre-cancellation skips validation/work.
    /// Already-running operations are not preemptible; callback count is unspecified.
    /// # Errors
    /// Same errors as forward, with None reserved for cancellation.
    pub fn forward_cancellable(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        if cancelled() {
            return Ok(None);
        }
        let base = self.cache_len();
        let end = base
            .checked_add(tokens.len())
            .ok_or_else(|| NnError::Backend("stream cursor overflow".into()))?;
        if tokens.is_empty()
            || positions.len() != tokens.len()
            || !positions.iter().copied().eq(base..end)
        {
            return Err(NnError::Backend(
                "forward requires nonempty contiguous committed positions".into(),
            ));
        }
        self.check_extent(base, tokens.len())?;
        if let Some(id) = tokens
            .iter()
            .find(|&&id| id as usize >= self.architecture.vocab_size())
        {
            return Err(NnError::MissingTensor(format!(
                "token {id} outside model vocabulary"
            )));
        }
        self.executor
            .forward(&self.architecture, tokens, positions, cancelled)
    }

    fn check_extent(&self, base: usize, count: usize) -> Result<(), NnError> {
        if let Some(admission) = self.admission
            && count > admission.limits().max_segment_tokens
        {
            return Err(NnError::ResourceExhausted(
                "forward segment exceeds admitted token cap".into(),
            ));
        }
        let end = base
            .checked_add(count)
            .ok_or_else(|| NnError::Backend("stream cursor overflow".into()))?;
        if end > self.max_context() {
            return Err(NnError::Shape {
                expected: self.max_context(),
                got: end,
            });
        }
        Ok(())
    }
    /// Segmented initial prefill shared by generation and serving.
    /// Pre-cancellation skips validation/work. Requires an empty cache. DeltaNet
    /// cached continuation uses single tokens; positions scratch is segment-sized.
    /// # Errors
    /// Empty/invalid prompt, nonempty cache, capacity, allocation or graph errors.
    /// Later errors/cancellation can leave a prefix; the caller owns reset policy.
    pub fn prefill_segmented_cancellable(
        &mut self,
        tokens: &[u32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        if cancelled() {
            return Ok(None);
        }
        if self.cache_len() != 0 || tokens.is_empty() {
            return Err(NnError::Backend(
                "segmented prefill requires an empty stream and nonempty prompt".into(),
            ));
        }
        if tokens.len() > self.max_context() {
            return Err(NnError::ResourceExhausted(
                "prompt exceeds admitted context".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&id| id as usize >= self.architecture.vocab_size())
        {
            return Err(NnError::Backend(
                "prompt ID outside model vocabulary".into(),
            ));
        }
        let segment = self
            .cache_admission()
            .map_or(tokens.len(), |a| a.limits().max_segment_tokens)
            .min(tokens.len());
        let single_cached = self
            .architecture
            .layer_kinds()
            .any(|k| k == LayerKind::DeltaNet);
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(segment)
            .map_err(|e| NnError::Backend(format!("prefill positions: {e}")))?;
        let mut at = 0;
        let mut logits = None;
        while at < tokens.len() {
            let count = if at != 0 && single_cached { 1 } else { segment };
            let end = at + count.min(tokens.len() - at);
            positions.clear();
            positions.extend(at..end);
            let Some(next) = self.forward_cancellable(&tokens[at..end], &positions, cancelled)?
            else {
                return Ok(None);
            };
            logits = Some(next);
            at = end;
        }
        Ok(logits)
    }
    /// Bounded greedy generation with segmented prefill and explicit context refusal.
    /// Validates all IDs/capacity before reset/work; zero requested output skips
    /// graph work. Does not forward the unused last sampled token. Errors during
    /// execution reset the private stream; legacy generate remains unchanged.
    /// # Errors
    /// Empty/out-of-vocabulary prompt, output/context overflow or graph errors.
    pub fn generate_bounded(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>, NnError> {
        if prompt.is_empty()
            || prompt
                .iter()
                .any(|&id| id as usize >= self.architecture.vocab_size())
        {
            return Err(NnError::Backend(
                "bounded generation requires a nonempty in-vocabulary prompt".into(),
            ));
        }
        if prompt
            .len()
            .checked_add(max_new)
            .is_none_or(|n| n > self.max_context())
        {
            return Err(NnError::ResourceExhausted(
                "prompt and --max-new exceed admitted context".into(),
            ));
        }
        self.reset();
        if max_new == 0 {
            return Ok(Vec::new());
        }
        let result = (|| {
            let mut logits = self
                .prefill_segmented_cancellable(prompt, &|| false)?
                .ok_or_else(|| NnError::Backend("uncancelled prefill returned no output".into()))?;
            let mut out = Vec::new();
            out.try_reserve_exact(max_new)
                .map_err(|e| NnError::Backend(format!("generated IDs: {e}")))?;
            for i in 0..max_new {
                let next = crate::ops::sample_greedy(&logits).ok_or(NnError::Shape {
                    expected: 1,
                    got: 0,
                })?;
                if next == eos {
                    break;
                }
                out.push(next);
                if i + 1 < max_new {
                    logits = self.forward(&[next], &[prompt.len() + i])?;
                }
            }
            Ok(out)
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// Greedy generation with the same loop/reset/EOS behavior as the legacy runner.
    /// # Errors
    /// Actual forward errors or allocation/position overflow.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>, NnError> {
        self.reset();
        generate_greedy(prompt, max_new, eos, |tokens, positions| {
            self.forward(tokens, positions)
        })
    }
}
fn positions(base: usize, length: usize) -> Result<Vec<usize>, NnError> {
    let end = base
        .checked_add(length)
        .ok_or_else(|| NnError::Backend("position overflow".into()))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|e| NnError::Backend(format!("positions: {e}")))?;
    result.extend(base..end);
    Ok(result)
}
pub(super) fn generate_greedy(
    prompt: &[u32],
    max_new: usize,
    eos: u32,
    mut forward: impl FnMut(&[u32], &[usize]) -> Result<Vec<f32>, NnError>,
) -> Result<Vec<u32>, NnError> {
    if prompt.is_empty() {
        return Ok(Vec::new());
    }
    let mut logits = forward(prompt, &positions(0, prompt.len())?)?;
    let mut out = Vec::new();
    out.try_reserve_exact(max_new)
        .map_err(|e| NnError::Backend(format!("generated IDs: {e}")))?;
    for i in 0..max_new {
        let pos = prompt
            .len()
            .checked_add(i)
            .ok_or_else(|| NnError::Backend("generation position overflow".into()))?;
        let next = crate::ops::sample_greedy(&logits).ok_or(NnError::Shape {
            expected: 1,
            got: 0,
        })?;
        if next == eos {
            break;
        }
        out.push(next);
        logits = forward(&[next], &[pos])?;
    }
    Ok(out)
}
