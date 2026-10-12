//! Common portable text execution. Internal numerical adapters, not resident clearance.
#[cfg(feature = "tokenizer")]
use super::{ModelRunner, Qwen35TextCache, Qwen35TextRunner, UnverifiedQwen35Mtp};
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

// Internal host port: both numerical adapters already exist and own real caches.
// Backend-owned resident/batched/tree execution is a subsequent contract migration.
trait Executor: Send {
    fn forward(
        &mut self,
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
#[cfg(feature = "tokenizer")]
struct StandardExecutor {
    runner: ModelRunner,
    cursor: usize,
}
#[cfg(feature = "tokenizer")]
impl Executor for StandardExecutor {
    fn forward(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        let result = self
            .runner
            .forward_cancellable(tokens, positions, cancelled)?;
        if result.is_some() {
            self.cursor += tokens.len();
        }
        Ok(result)
    }
    fn reset(&mut self) {
        self.runner.reset();
        self.cursor = 0;
    }
    fn len(&self) -> usize {
        self.cursor
    }
}
#[cfg(feature = "tokenizer")]
struct HybridExecutor {
    runner: Qwen35TextRunner,
    cache: Qwen35TextCache,
    mtp: UnverifiedQwen35Mtp,
}
#[cfg(feature = "tokenizer")]
impl Executor for HybridExecutor {
    fn forward(
        &mut self,
        tokens: &[u32],
        _positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        self.runner
            .forward_cancellable(tokens, &mut self.cache, cancelled)
            .map(|result| result.map(super::Qwen35TextOutput::into_last_logits))
    }
    fn reset(&mut self) {
        self.cache.reset();
    }
    fn len(&self) -> usize {
        self.cache.len()
    }
    fn mtp_status(&self) -> Option<Qwen35MtpStatus> {
        Some(self.mtp.status())
    }
}

/// One loaded language-stream interface over the common portable executor seam.
/// Owns immutable graph weights and one private cache. No concrete backend is named.
/// Native resident/batching/draft/tree/full-memory qualification remains separate.
#[allow(missing_debug_implementations)]
pub struct Runner {
    architecture: ArchitectureSpec,
    executor: Box<dyn Executor>,
}
impl Runner {
    #[cfg(feature = "tokenizer")]
    pub(super) fn standard(mut runner: ModelRunner, variation: ArchSpec) -> Self {
        runner.reset();
        let architecture = ArchitectureSpec::Standard {
            config: runner.config.clone(),
            variation,
            vocab_size: runner.weights.vocab,
        };
        Self {
            architecture,
            executor: Box::new(StandardExecutor { runner, cursor: 0 }),
        }
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn hybrid(
        config: Qwen35CheckpointConfig,
        runner: Qwen35TextRunner,
        mtp: UnverifiedQwen35Mtp,
    ) -> Result<Self, NnError> {
        let cache = runner.new_cache(config.text.max_position_embeddings as usize)?;
        Ok(Self {
            architecture: ArchitectureSpec::Hybrid(config),
            executor: Box::new(HybridExecutor { runner, cache, mtp }),
        })
    }
    /// Actual immutable graph/configuration used by this stream.
    #[must_use]
    pub const fn architecture(&self) -> &ArchitectureSpec {
        &self.architecture
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
        if end > self.architecture.max_context() {
            return Err(NnError::Shape {
                expected: self.architecture.max_context(),
                got: end,
            });
        }
        if let Some(id) = tokens
            .iter()
            .find(|&&id| id as usize >= self.architecture.vocab_size())
        {
            return Err(NnError::MissingTensor(format!(
                "token {id} outside model vocabulary"
            )));
        }
        self.executor.forward(tokens, positions, cancelled)
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
