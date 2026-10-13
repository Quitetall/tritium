//! One owned weights aggregate and transactional host executor for admitted text graphs.
//! Architecture data selects existing leaf numerics; no whole-model runner is retained.
use super::qwen35::{
    Qwen35TextForwardError, Qwen35TextLayer, Qwen35TextLayerCache, RunnerIdentity, normalize_rows,
};
use super::{ArchitectureSpec, ModelWeights, UnverifiedQwen35Mtp, unified::Executor};
use crate::{
    BlockScratch, KvCache, NnError, Projection, Qwen35MtpStatus, TokenEmbedding, TransformerBlock,
};
use std::sync::Arc;
use tritium_spec::TernaryBackend;

enum Layer {
    Standard(TransformerBlock),
    Hybrid(Qwen35TextLayer),
}

/// Internal common storage. Legacy public weight types remain import adapters
/// until all artifacts and surfaces migrate; no dense shadow is introduced here.
pub(super) struct Weights {
    embedding: TokenEmbedding,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    head: Option<Projection>,
}

/// Consuming bridge from the validated hybrid constructor: data and target
/// identity move, and no legacy language runner survives the import.
pub(super) struct HybridImport {
    pub(super) backend: Box<dyn TernaryBackend>,
    pub(super) weights: Weights,
    pub(super) identity: Arc<RunnerIdentity>,
}
impl Weights {
    fn standard(weights: ModelWeights) -> Result<Self, NnError> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(weights.layers.len())
            .map_err(alloc_error)?;
        layers.extend(weights.layers.into_iter().map(Layer::Standard));
        Ok(Self {
            embedding: weights.token_embd,
            layers,
            final_norm: weights.output_norm,
            head: weights.lm_head,
        })
    }
    pub(super) fn hybrid(
        embedding: TokenEmbedding,
        source: Vec<Qwen35TextLayer>,
        final_norm: Vec<f32>,
        head: Projection,
    ) -> Result<Self, NnError> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source.len())
            .map_err(alloc_error)?;
        layers.extend(source.into_iter().map(Layer::Hybrid));
        Ok(Self {
            embedding,
            layers,
            final_norm,
            head: Some(head),
        })
    }
}

enum LayerCache {
    Standard(KvCache),
    Hybrid(Qwen35TextLayerCache),
}
impl LayerCache {
    fn rollback(&mut self, base: usize) {
        match self {
            Self::Standard(cache) => cache.rollback_to(base),
            Self::Hybrid(cache) => cache.rollback(base),
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Standard(cache) => cache.reset(),
            Self::Hybrid(cache) => cache.reset(),
        }
    }
    fn staged_for(&self, end: usize) -> bool {
        match self {
            Self::Standard(cache) => cache.len == end,
            Self::Hybrid(cache) => cache.staged_for(end),
        }
    }
    fn commit(&mut self) {
        if let Self::Hybrid(cache) = self {
            cache.commit();
        }
    }
}

pub(super) struct HostExecutor {
    backend: Arc<dyn TernaryBackend>,
    weights: Weights,
    caches: Vec<LayerCache>,
    cursor: usize,
    has_delta: bool,
    mtp: Option<UnverifiedQwen35Mtp>,
    // Preserve the target binding held by the unverified drafter. This is not
    // a new executable MTP authority or a retained language runner.
    _target_identity: Option<Arc<RunnerIdentity>>,
}

impl HostExecutor {
    pub(super) fn standard(
        architecture: &ArchitectureSpec,
        weights: ModelWeights,
        backend: Box<dyn TernaryBackend>,
        context: usize,
        max_segment_tokens: usize,
    ) -> Result<Box<dyn Executor>, NnError> {
        let backend: Arc<dyn TernaryBackend> = Arc::from(backend);
        if let Some(executor) = super::resident::standard(
            architecture,
            &weights,
            Arc::clone(&backend),
            tritium_spec::DecoderLimits {
                max_context: context,
                max_segment_tokens,
            },
        )? {
            return Ok(executor);
        }
        Self::new(
            architecture,
            Weights::standard(weights)?,
            backend,
            context,
            None,
            None,
        )
        .map(|executor| Box::new(executor) as Box<dyn Executor>)
    }
    pub(super) fn hybrid(
        architecture: &ArchitectureSpec,
        weights: Weights,
        backend: Box<dyn TernaryBackend>,
        context: usize,
        mtp: UnverifiedQwen35Mtp,
        identity: Arc<RunnerIdentity>,
    ) -> Result<Self, NnError> {
        if backend.resident_decoder_requested() {
            return Err(NnError::Backend(
                "resident hybrid decoder is not implemented; no host fallback".into(),
            ));
        }
        Self::new(
            architecture,
            weights,
            Arc::from(backend),
            context,
            Some(mtp),
            Some(identity),
        )
    }
    fn new(
        architecture: &ArchitectureSpec,
        weights: Weights,
        backend: Arc<dyn TernaryBackend>,
        context: usize,
        mtp: Option<UnverifiedQwen35Mtp>,
        identity: Option<Arc<RunnerIdentity>>,
    ) -> Result<Self, NnError> {
        if context == 0
            || context > architecture.max_context()
            || weights.layers.len() != architecture.layer_count()
        {
            return Err(invalid("cache capacity or layer count mismatch"));
        }
        let width = hidden_size(architecture);
        if weights.embedding.rows() != architecture.vocab_size()
            || weights.embedding.cols() != width
            || weights.final_norm.len() != width
        {
            return Err(invalid("embedding/final norm geometry mismatch"));
        }
        if let Some(head) = &weights.head
            && (head.k_in() != width || head.n_out() != architecture.vocab_size())
        {
            return Err(invalid("language head geometry mismatch"));
        }
        let mut caches = Vec::new();
        caches
            .try_reserve_exact(weights.layers.len())
            .map_err(alloc_error)?;
        let mut has_delta = false;
        for layer in &weights.layers {
            caches.push(match (architecture, layer) {
                (ArchitectureSpec::Standard { config, .. }, Layer::Standard(_)) => {
                    LayerCache::Standard(KvCache::try_new(
                        context,
                        config.n_head_kv as usize,
                        config.head_dim() as usize,
                    )?)
                }
                (ArchitectureSpec::Hybrid(_), Layer::Hybrid(layer)) => {
                    has_delta |= layer.is_delta();
                    LayerCache::Hybrid(layer.new_cache(context)?)
                }
                _ => return Err(invalid("architecture/weight layer mismatch")),
            });
        }
        Ok(Self {
            backend,
            weights,
            caches,
            cursor: 0,
            has_delta,
            mtp,
            _target_identity: identity,
        })
    }

    fn preflight(&self, tokens: &[u32], positions: &[usize]) -> Result<usize, NnError> {
        let end = self
            .cursor
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("cursor overflow"))?;
        if tokens.is_empty()
            || positions.len() != tokens.len()
            || !positions.iter().copied().eq(self.cursor..end)
        {
            return Err(invalid("noncontiguous or empty segment"));
        }
        if self.cursor != 0 && self.has_delta && tokens.len() != 1 {
            return Err(NnError::MissingConfig(
                "Qwen3.5 cached DeltaNet continuation must contain exactly one token".into(),
            ));
        }
        for (layer, cache) in self.weights.layers.iter().zip(&self.caches) {
            let valid = match (layer, cache) {
                (Layer::Standard(_), LayerCache::Standard(cache)) => cache.len == self.cursor,
                (Layer::Hybrid(layer), LayerCache::Hybrid(cache)) => {
                    layer.cache_valid(cache, self.cursor)
                }
                _ => false,
            };
            if !valid {
                return Err(invalid("layer cache disagrees with committed cursor"));
            }
        }
        Ok(end)
    }

    fn run(
        &mut self,
        architecture: &ArchitectureSpec,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        if cancelled() {
            return Ok(None);
        }
        let end = self.preflight(tokens, positions)?;
        let width = hidden_size(architecture);
        let count = product(tokens.len(), width)?;
        let mut hidden = zeros(count)?;
        let mut next = zeros(count)?;
        let hybrid = matches!(architecture, ArchitectureSpec::Hybrid(_));
        let mut normalized = zeros(if hybrid { count } else { width })?;
        let mut scratch = match architecture {
            ArchitectureSpec::Standard { config, .. } => {
                Some(standard_scratch(tokens.len(), config)?)
            }
            ArchitectureSpec::Hybrid(_) => None,
        };
        let result = (|| -> Result<Vec<f32>, Stop> {
            self.weights.embedding.gather_with_backend(
                self.backend.as_ref(),
                tokens,
                &mut hidden,
            )?;
            checkpoint(cancelled)?;
            for (layer, cache) in self.weights.layers.iter().zip(&mut self.caches) {
                checkpoint(cancelled)?;
                match (architecture, layer, cache) {
                    (
                        ArchitectureSpec::Standard { config, .. },
                        Layer::Standard(layer),
                        LayerCache::Standard(cache),
                    ) => {
                        layer.forward_with_scratch(
                            self.backend.as_ref(),
                            &hidden,
                            positions,
                            cache,
                            config,
                            &mut next,
                            scratch
                                .as_mut()
                                .ok_or_else(|| invalid("standard scratch missing"))?,
                        )?;
                        std::mem::swap(&mut hidden, &mut next);
                    }
                    (
                        ArchitectureSpec::Hybrid(config),
                        Layer::Hybrid(layer),
                        LayerCache::Hybrid(cache),
                    ) => {
                        let mut boundary = || if cancelled() { Err(()) } else { Ok(()) };
                        layer
                            .stage(
                                self.backend.as_ref(),
                                positions,
                                cache,
                                &mut hidden,
                                &mut normalized,
                                &mut next,
                                width,
                                config.text.rms_norm_eps as f32,
                                &[],
                                &mut |_, _| {},
                                &mut boundary,
                            )
                            .map_err(|error| match error {
                                Qwen35TextForwardError::Runtime(e) => Stop::Runtime(e),
                                Qwen35TextForwardError::Observer(()) => Stop::Cancelled,
                            })?;
                    }
                    _ => return Err(invalid("architecture/layer/cache mismatch").into()),
                }
                checkpoint(cancelled)?;
            }
            checkpoint(cancelled)?;
            let last = (tokens.len() - 1) * width;
            let row = match architecture {
                ArchitectureSpec::Standard { config, .. } => {
                    crate::ops::rmsnorm(
                        &hidden[last..last + width],
                        &self.weights.final_norm,
                        config.rms_eps,
                        &mut normalized,
                    )?;
                    &normalized[..]
                }
                ArchitectureSpec::Hybrid(config) => {
                    normalize_rows(
                        &hidden,
                        &self.weights.final_norm,
                        config.text.rms_norm_eps as f32,
                        width,
                        &mut normalized,
                    )?;
                    &normalized[last..last + width]
                }
            };
            let mut logits = zeros(architecture.vocab_size())?;
            checkpoint(cancelled)?;
            if let Some(head) = &self.weights.head {
                head.forward(self.backend.as_ref(), row, 1, &mut logits)?;
            } else {
                self.weights.embedding.unembed_exact_with_backend(
                    self.backend.as_ref(),
                    row,
                    &mut logits,
                )?;
            }
            if hybrid && normalized.iter().chain(&logits).any(|v| !v.is_finite()) {
                return Err(NnError::Backend(
                    "Qwen3.5 text forward produced a non-finite value".into(),
                )
                .into());
            }
            checkpoint(cancelled)?;
            if self.caches.iter().any(|cache| !cache.staged_for(end)) {
                return Err(invalid("layer failed global commit preflight").into());
            }
            Ok(logits)
        })();
        match result {
            Ok(logits) => {
                for cache in &mut self.caches {
                    cache.commit();
                }
                self.cursor = end;
                Ok(Some(logits))
            }
            Err(error) => {
                for cache in &mut self.caches {
                    cache.rollback(self.cursor);
                }
                match error {
                    Stop::Runtime(error) => Err(error),
                    Stop::Cancelled => Ok(None),
                }
            }
        }
    }
}

impl Executor for HostExecutor {
    fn forward(
        &mut self,
        architecture: &ArchitectureSpec,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, NnError> {
        self.run(architecture, tokens, positions, cancelled)
    }
    fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        self.cursor = 0;
    }
    fn len(&self) -> usize {
        self.cursor
    }
    fn mtp_status(&self) -> Option<Qwen35MtpStatus> {
        self.mtp.as_ref().map(UnverifiedQwen35Mtp::status)
    }
}

enum Stop {
    Runtime(NnError),
    Cancelled,
}
impl From<NnError> for Stop {
    fn from(e: NnError) -> Self {
        Self::Runtime(e)
    }
}
fn checkpoint(cancelled: &dyn Fn() -> bool) -> Result<(), Stop> {
    if cancelled() {
        Err(Stop::Cancelled)
    } else {
        Ok(())
    }
}
fn hidden_size(architecture: &ArchitectureSpec) -> usize {
    match architecture {
        ArchitectureSpec::Standard { config, .. } => config.n_embd as usize,
        ArchitectureSpec::Hybrid(config) => config.text.hidden_size as usize,
    }
}
fn product(a: usize, b: usize) -> Result<usize, NnError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("scratch extent overflow"))
}
fn zeros(count: usize) -> Result<Vec<f32>, NnError> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(alloc_error)?;
    out.resize(count, 0.);
    Ok(out)
}
fn alloc_error(error: impl std::fmt::Display) -> NnError {
    NnError::Backend(format!("allocate common graph storage: {error}"))
}
fn invalid(message: &str) -> NnError {
    NnError::Backend(format!("common graph: {message}"))
}
fn standard_scratch(sequence: usize, config: &crate::ModelConfig) -> Result<BlockScratch, NnError> {
    let hidden = product(sequence, config.n_embd as usize)?;
    let q = product(
        sequence,
        product(config.n_head as usize, config.head_dim() as usize)?,
    )?;
    let kv = product(
        sequence,
        product(config.n_head_kv as usize, config.head_dim() as usize)?,
    )?;
    Ok(BlockScratch {
        normed: zeros(hidden)?,
        q: zeros(q)?,
        k: zeros(kv)?,
        v: zeros(kv)?,
        attn: zeros(q)?,
        sn: zeros(q)?,
        mlp_out: zeros(hidden)?,
    })
}
