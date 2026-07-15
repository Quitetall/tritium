//! Exact Qwen3.5-family hybrid language runner.
//!
//! Qwen3.6 interleaves recurrent Gated DeltaNet and gated full-attention
//! mixers.  That graph, its cache state, and its zero-centered normalization
//! semantics are deliberately kept out of the homogeneous [`ModelRunner`]
//! (`super::ModelRunner`).

use std::sync::Arc;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{
    Projection, ProjectionActivationMode, Qwen35DeltaNet, Qwen35DeltaNetCache,
    Qwen35DeltaNetWeights, Qwen35FullAttention, Qwen35FullAttentionCache,
    Qwen35FullAttentionWeights, SwiGluMlp, TokenEmbedding,
};
use crate::ops::rmsnorm_zero_centered;
use crate::qwen35_config::{Qwen35LayerType, Qwen35NormWeightSemantics, Qwen35TextConfig};

/// Unbound token-mixer weights for one Qwen3.5-family decoder layer.
#[allow(missing_debug_implementations)]
pub enum Qwen35TextMixerWeights {
    /// Gated DeltaNet weights for a `linear_attention` layer.
    DeltaNet(Qwen35DeltaNetWeights),
    /// Gated causal GQA weights for a `full_attention` layer.
    FullAttention(Qwen35FullAttentionWeights),
}

impl Qwen35TextMixerWeights {
    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }
}

/// Raw weights for one exact Qwen3.5-family language layer.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextLayerWeights {
    /// Zero-centered RMSNorm before the token mixer, `[hidden]`.
    pub input_norm: Vec<f32>,
    /// Mixer selected by the checkpoint's exact layer schedule.
    pub mixer: Qwen35TextMixerWeights,
    /// Zero-centered RMSNorm before SwiGLU, `[hidden]`.
    pub post_attention_norm: Vec<f32>,
    /// Bias-free Qwen SwiGLU feed-forward network.
    pub mlp: SwiGluMlp,
}

impl Qwen35TextLayerWeights {
    /// Collect one unbound language layer.
    #[must_use]
    pub fn new(
        input_norm: Vec<f32>,
        mixer: Qwen35TextMixerWeights,
        post_attention_norm: Vec<f32>,
        mlp: SwiGluMlp,
    ) -> Self {
        Self {
            input_norm,
            mixer,
            post_attention_norm,
            mlp,
        }
    }
}

/// Raw, untied Qwen3.5-family language-model weights.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextWeights {
    /// Token embedding table `[vocab, hidden]`.
    pub embedding: TokenEmbedding,
    /// Decoder layers in checkpoint order.
    pub layers: Vec<Qwen35TextLayerWeights>,
    /// Final zero-centered RMSNorm parameter `[hidden]`.
    pub final_norm: Vec<f32>,
    /// Mandatory untied language head `[vocab, hidden]`.
    pub lm_head: Projection,
}

impl Qwen35TextWeights {
    /// Collect an unbound exact text graph.
    #[must_use]
    pub fn new(
        embedding: TokenEmbedding,
        layers: Vec<Qwen35TextLayerWeights>,
        final_norm: Vec<f32>,
        lm_head: Projection,
    ) -> Self {
        Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        }
    }
}

enum Qwen35TextMixer {
    DeltaNet(Qwen35DeltaNet),
    FullAttention(Qwen35FullAttention),
}

impl Qwen35TextMixer {
    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }

    const fn activation_mode(&self) -> ProjectionActivationMode {
        match self {
            Self::DeltaNet(layer) => layer.activation_mode(),
            Self::FullAttention(layer) => layer.activation_mode(),
        }
    }
}

struct Qwen35TextLayer {
    input_norm: Vec<f32>,
    mixer: Qwen35TextMixer,
    post_attention_norm: Vec<f32>,
    mlp: SwiGluMlp,
}

#[derive(Debug)]
enum Qwen35TextLayerCache {
    DeltaNet(Qwen35DeltaNetCache),
    FullAttention(Qwen35FullAttentionCache),
}

impl Qwen35TextLayerCache {
    const fn kind(&self) -> Qwen35LayerType {
        match self {
            Self::DeltaNet(_) => Qwen35LayerType::DeltaNet,
            Self::FullAttention(_) => Qwen35LayerType::FullAttention,
        }
    }

    const fn committed_len(&self) -> usize {
        match self {
            Self::DeltaNet(cache) => cache.len(),
            Self::FullAttention(cache) => cache.committed_len(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::DeltaNet(cache) => cache.reset(),
            Self::FullAttention(cache) => cache.reset(),
        }
    }
}

#[derive(Debug)]
struct RunnerIdentity;

/// One exact-runner hybrid cache with a single committed language cursor.
///
/// Individual mixer states are private so callers cannot advance one layer
/// independently of the rest of the language graph.
#[derive(Debug)]
pub struct Qwen35TextCache {
    runner_identity: Arc<RunnerIdentity>,
    layers: Vec<Qwen35TextLayerCache>,
    committed_len: usize,
    max_context: usize,
}

impl Qwen35TextCache {
    /// Number of tokens committed by every language layer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.committed_len
    }

    /// Whether the cache contains no committed tokens.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.committed_len == 0
    }

    /// Stream capacity selected when this cache was created.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Clear every mixer state and the global cursor without freeing capacity.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.committed_len = 0;
    }
}

/// Successful exact-language forward output.
///
/// `final_hidden_states` are after the model's final zero-centered RMSNorm and
/// are therefore the target hidden rows consumed by Qwen3.5 MTP. They are not
/// the pre-final residual stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35TextOutput {
    sequence: usize,
    hidden_size: usize,
    final_hidden_states: Vec<f32>,
    last_logits: Vec<f32>,
}

impl Qwen35TextOutput {
    /// Number of input token rows represented by the hidden-state buffer.
    #[must_use]
    pub const fn sequence(&self) -> usize {
        self.sequence
    }

    /// Width of each final-normalized hidden row.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Final-normalized hidden rows `[sequence, hidden]`, suitable for MTP.
    #[must_use]
    pub fn final_hidden_states(&self) -> &[f32] {
        &self.final_hidden_states
    }

    /// Untied language-head logits for the last input token, `[vocab]`.
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }
}

/// Exact Qwen3.5/Qwen3.6 hybrid language-core runner.
#[allow(missing_debug_implementations)]
pub struct Qwen35TextRunner {
    identity: Arc<RunnerIdentity>,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_context: usize,
    rms_norm_eps: f32,
    activation_mode: ProjectionActivationMode,
    has_delta_net: bool,
    backend: Box<dyn TernaryBackend>,
    embedding: TokenEmbedding,
    layers: Vec<Qwen35TextLayer>,
    final_norm: Vec<f32>,
    lm_head: Projection,
}

impl Qwen35TextRunner {
    /// Bind the exact mixed schedule and all raw weights to one private runner.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for unsupported or contradictory
    /// schedule/numeric semantics, [`NnError::Shape`] for any weight geometry
    /// mismatch, or [`NnError::Backend`] for non-finite fp32 weights or an
    /// allocation failure.
    pub fn new(
        config: &Qwen35TextConfig,
        weights: Qwen35TextWeights,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let intermediate_size = axis(config.intermediate_size, "intermediate_size")?;
        let vocab_size = axis(config.vocab_size, "vocab_size")?;
        let max_context = axis(config.max_position_embeddings, "max_position_embeddings")?;
        let layer_count = axis(config.num_hidden_layers, "num_hidden_layers")?;
        let interval = axis(config.full_attention_interval, "full_attention_interval")?;
        let rms_norm_eps = config.rms_norm_eps as f32;
        // Guard both the source value and overflow introduced by f64-to-f32 narrowing.
        if !config.rms_norm_eps.is_finite() || !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(invalid_config(
                "Qwen3.5 text RMSNorm epsilon must be a finite positive f32 value",
            ));
        }
        if config.model_type != "qwen3_5_text"
            || !config.use_cache
            || config.tied_embeddings
            || config.full_attention.norm_weight_semantics
                != Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight
        {
            return Err(invalid_config(
                "unsupported Qwen3.5 text graph or normalization semantics",
            ));
        }
        if config.layer_types.len() != layer_count {
            return Err(NnError::Shape {
                expected: layer_count,
                got: config.layer_types.len(),
            });
        }
        if weights.layers.len() != layer_count {
            return Err(NnError::Shape {
                expected: layer_count,
                got: weights.layers.len(),
            });
        }
        validate_schedule(&config.layer_types, interval)?;
        validate_token_table(&weights.embedding, vocab_size, hidden_size)?;
        validate_projection(&weights.lm_head, vocab_size, hidden_size)?;
        validate_finite_projection(&weights.lm_head, "language head")?;
        validate_finite(&weights.final_norm, hidden_size, "final norm")?;

        let activation_mode = weights.lm_head.activation_mode();
        let mut layers = Vec::new();
        layers.try_reserve_exact(layer_count).map_err(|error| {
            NnError::Backend(format!(
                "allocate Qwen3.5 bound layer table for {layer_count} layers: {error}"
            ))
        })?;
        for (index, (expected_kind, raw)) in config
            .layer_types
            .iter()
            .copied()
            .zip(weights.layers)
            .enumerate()
        {
            if raw.mixer.kind() != expected_kind {
                return Err(invalid_config(format!(
                    "Qwen3.5 layer {index} mixer contradicts the configured schedule"
                )));
            }
            validate_finite(&raw.input_norm, hidden_size, "input norm")?;
            validate_finite(&raw.post_attention_norm, hidden_size, "post-attention norm")?;
            let mlp_mode = validate_mlp(&raw.mlp, hidden_size, intermediate_size)?;
            if mlp_mode != activation_mode {
                return Err(invalid_config(
                    "Qwen3.5 language projections must use one activation arithmetic mode",
                ));
            }
            let mixer = match raw.mixer {
                Qwen35TextMixerWeights::DeltaNet(raw) => {
                    Qwen35TextMixer::DeltaNet(Qwen35DeltaNet::new(config, raw)?)
                }
                Qwen35TextMixerWeights::FullAttention(raw) => {
                    Qwen35TextMixer::FullAttention(Qwen35FullAttention::new(config, raw)?)
                }
            };
            if mixer.activation_mode() != activation_mode {
                return Err(invalid_config(
                    "Qwen3.5 language projections must use one activation arithmetic mode",
                ));
            }
            layers.push(Qwen35TextLayer {
                input_norm: raw.input_norm,
                mixer,
                post_attention_norm: raw.post_attention_norm,
                mlp: raw.mlp,
            });
        }

        Ok(Self {
            identity: Arc::new(RunnerIdentity),
            hidden_size,
            intermediate_size,
            vocab_size,
            max_context,
            rms_norm_eps,
            activation_mode,
            has_delta_net: config.layer_types.contains(&Qwen35LayerType::DeltaNet),
            backend,
            embedding: weights.embedding,
            layers,
            final_norm: weights.final_norm,
            lm_head: weights.lm_head,
        })
    }

    /// Activation arithmetic shared by every language projection.
    #[must_use]
    pub const fn activation_mode(&self) -> ProjectionActivationMode {
        self.activation_mode
    }

    /// Hidden-state width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// SwiGLU intermediate width.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Vocabulary size shared by embedding and the mandatory untied head.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Allocate one exact-runner hybrid cache.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] when `max_context` is zero or exceeds the
    /// model limit, or [`NnError::Backend`] when a layer-cache allocation fails.
    pub fn new_cache(&self, max_context: usize) -> Result<Qwen35TextCache, NnError> {
        if max_context == 0 || max_context > self.max_context {
            return Err(NnError::Shape {
                expected: self.max_context,
                got: max_context,
            });
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.layers.len())
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate Qwen3.5 cache layer table for {} layers: {error}",
                    self.layers.len()
                ))
            })?;
        for layer in &self.layers {
            layers.push(match &layer.mixer {
                Qwen35TextMixer::DeltaNet(mixer) => {
                    Qwen35TextLayerCache::DeltaNet(mixer.new_cache()?)
                }
                Qwen35TextMixer::FullAttention(mixer) => {
                    Qwen35TextLayerCache::FullAttention(mixer.new_cache(max_context)?)
                }
            });
        }
        Ok(Qwen35TextCache {
            runner_identity: Arc::clone(&self.identity),
            layers,
            committed_len: 0,
            max_context,
        })
    }

    /// Run one initial prefill or one-token cached continuation transaction.
    ///
    /// Positions are derived as the contiguous interval beginning at
    /// [`Qwen35TextCache::len`]. Callers cannot provide divergent RoPE positions.
    /// Every DeltaNet layer stages state privately; full-attention KV appends are
    /// provisional until the final norm and untied language head both succeed.
    /// On any error all DeltaNet stages are discarded, every full-attention cache
    /// is restored to the global base, and no output object is returned.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for an empty input or capacity mismatch,
    /// [`NnError::MissingTensor`] for an out-of-vocabulary token,
    /// [`NnError::MissingConfig`] for an unsafe multi-token cached DeltaNet
    /// continuation, [`NnError::Backend`] for foreign/inconsistent cache state,
    /// or an error from a layer operation.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35TextCache,
    ) -> Result<Qwen35TextOutput, NnError> {
        let (base, new_len) = self.preflight_forward(tokens, cache)?;
        let sequence = tokens.len();
        let hidden_len = checked_mul(sequence, self.hidden_size, "hidden-state buffer")?;
        let mut residual = zeroed_scratch(hidden_len, "embedding output")?;
        let mut normalized = zeroed_scratch(hidden_len, "normalized hidden state")?;
        let mut branch = zeroed_scratch(hidden_len, "decoder branch output")?;
        let mut positions = Vec::new();
        positions.try_reserve_exact(sequence).map_err(|error| {
            NnError::Backend(format!(
                "allocate Qwen3.5 position vector for {sequence} tokens: {error}"
            ))
        })?;
        positions.extend(base..new_len);
        self.embedding
            .gather_with_backend(self.backend.as_ref(), tokens, &mut residual)?;

        let result = self.forward_provisional(
            self.backend.as_ref(),
            sequence,
            &positions,
            cache,
            &mut residual,
            &mut normalized,
            &mut branch,
        );
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                self.abort_and_rollback(cache, base);
                return Err(error);
            }
        };

        if let Err(error) = self.preflight_commit(cache, new_len) {
            self.abort_and_rollback(cache, base);
            return Err(error);
        }
        for layer in &mut cache.layers {
            if let Qwen35TextLayerCache::DeltaNet(cache) = layer {
                cache.commit_staged();
            }
        }
        cache.committed_len = new_len;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_provisional(
        &self,
        backend: &dyn TernaryBackend,
        sequence: usize,
        positions: &[usize],
        cache: &mut Qwen35TextCache,
        residual: &mut [f32],
        normalized: &mut [f32],
        branch: &mut [f32],
    ) -> Result<Qwen35TextOutput, NnError> {
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            normalize_rows(
                residual,
                &layer.input_norm,
                self.rms_norm_eps,
                self.hidden_size,
                normalized,
            )?;
            match (&layer.mixer, layer_cache) {
                (Qwen35TextMixer::DeltaNet(mixer), Qwen35TextLayerCache::DeltaNet(cache)) => {
                    mixer.stage_forward(backend, normalized, sequence, cache, branch)?
                }
                (
                    Qwen35TextMixer::FullAttention(mixer),
                    Qwen35TextLayerCache::FullAttention(cache),
                ) => mixer.forward(backend, normalized, positions, cache, branch)?,
                _ => {
                    return Err(NnError::Backend(
                        "Qwen3.5 cache layer kind changed after preflight".to_owned(),
                    ));
                }
            }
            add_in_place(residual, branch);

            normalize_rows(
                residual,
                &layer.post_attention_norm,
                self.rms_norm_eps,
                self.hidden_size,
                normalized,
            )?;
            layer.mlp.forward(backend, normalized, sequence, branch)?;
            add_in_place(residual, branch);
        }

        let mut final_hidden_states = zeroed_scratch(residual.len(), "final hidden states")?;
        normalize_rows(
            residual,
            &self.final_norm,
            self.rms_norm_eps,
            self.hidden_size,
            &mut final_hidden_states,
        )?;
        let mut last_logits = zeroed_scratch(self.vocab_size, "last-token logits")?;
        let last_start = checked_mul(sequence - 1, self.hidden_size, "last hidden row")?;
        self.lm_head.forward(
            backend,
            &final_hidden_states[last_start..last_start + self.hidden_size],
            1,
            &mut last_logits,
        )?;
        if final_hidden_states
            .iter()
            .chain(&last_logits)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen3.5 text forward produced a non-finite value".to_owned(),
            ));
        }
        Ok(Qwen35TextOutput {
            sequence,
            hidden_size: self.hidden_size,
            final_hidden_states,
            last_logits,
        })
    }

    fn preflight_forward(
        &self,
        tokens: &[u32],
        cache: &Qwen35TextCache,
    ) -> Result<(usize, usize), NnError> {
        if !Arc::ptr_eq(&cache.runner_identity, &self.identity) {
            return Err(NnError::Backend(
                "Qwen3.5 text cache belongs to a different runner".to_owned(),
            ));
        }
        if tokens.is_empty() {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        if cache.layers.len() != self.layers.len() {
            return Err(NnError::Backend(
                "Qwen3.5 text cache layer count is inconsistent".to_owned(),
            ));
        }
        for (index, (layer, layer_cache)) in self.layers.iter().zip(&cache.layers).enumerate() {
            if layer.mixer.kind() != layer_cache.kind()
                || layer_cache.committed_len() != cache.committed_len
                || matches!(
                    layer_cache,
                    Qwen35TextLayerCache::DeltaNet(cache) if cache.staged_len().is_some()
                )
            {
                return Err(NnError::Backend(format!(
                    "Qwen3.5 text cache layer {index} is inconsistent with the global cursor"
                )));
            }
        }
        if cache.committed_len != 0 && self.has_delta_net && tokens.len() != 1 {
            return Err(invalid_config(
                "Qwen3.5 cached DeltaNet continuation must contain exactly one token",
            ));
        }
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| usize::try_from(token).map_or(true, |token| token >= self.vocab_size))
        {
            return Err(NnError::MissingTensor(format!("token_embd row {token}")));
        }
        let new_len = cache
            .committed_len
            .checked_add(tokens.len())
            .ok_or(NnError::Shape {
                expected: cache.max_context,
                got: usize::MAX,
            })?;
        if new_len > cache.max_context {
            return Err(NnError::Shape {
                expected: cache.max_context,
                got: new_len,
            });
        }
        Ok((cache.committed_len, new_len))
    }

    fn preflight_commit(&self, cache: &Qwen35TextCache, new_len: usize) -> Result<(), NnError> {
        for (index, layer) in cache.layers.iter().enumerate() {
            let valid = match layer {
                Qwen35TextLayerCache::DeltaNet(cache) => cache.staged_len() == Some(new_len),
                Qwen35TextLayerCache::FullAttention(cache) => cache.committed_len() == new_len,
            };
            if !valid {
                return Err(NnError::Backend(format!(
                    "Qwen3.5 text layer {index} failed global commit preflight"
                )));
            }
        }
        Ok(())
    }

    fn abort_and_rollback(&self, cache: &mut Qwen35TextCache, base: usize) {
        for layer in &mut cache.layers {
            match layer {
                Qwen35TextLayerCache::DeltaNet(cache) => cache.abort_staged(),
                Qwen35TextLayerCache::FullAttention(cache) => cache.rollback_to(base),
            }
        }
    }
}

fn validate_schedule(layer_types: &[Qwen35LayerType], interval: usize) -> Result<(), NnError> {
    for (index, &kind) in layer_types.iter().enumerate() {
        let layer_number = index
            .checked_add(1)
            .ok_or_else(|| invalid_config("Qwen3.5 layer schedule index overflow"))?;
        let expected = if layer_number.is_multiple_of(interval) {
            Qwen35LayerType::FullAttention
        } else {
            Qwen35LayerType::DeltaNet
        };
        if kind != expected {
            return Err(invalid_config(format!(
                "Qwen3.5 layer {index} contradicts full_attention_interval"
            )));
        }
    }
    Ok(())
}

fn validate_token_table(
    embedding: &TokenEmbedding,
    rows: usize,
    cols: usize,
) -> Result<(), NnError> {
    if embedding.rows() != rows {
        return Err(NnError::Shape {
            expected: rows,
            got: embedding.rows(),
        });
    }
    if embedding.cols() != cols {
        return Err(NnError::Shape {
            expected: cols,
            got: embedding.cols(),
        });
    }
    if embedding
        .as_dense()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(NnError::Backend(
            "Qwen3.5 token embedding contains a non-finite value".to_owned(),
        ));
    }
    Ok(())
}

fn validate_mlp(
    mlp: &SwiGluMlp,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<ProjectionActivationMode, NnError> {
    validate_projection(&mlp.gate, intermediate_size, hidden_size)?;
    validate_projection(&mlp.up, intermediate_size, hidden_size)?;
    validate_projection(&mlp.down, hidden_size, intermediate_size)?;
    validate_finite_projection(&mlp.gate, "SwiGLU gate")?;
    validate_finite_projection(&mlp.up, "SwiGLU up")?;
    validate_finite_projection(&mlp.down, "SwiGLU down")?;
    mlp.activation_mode()
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
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
    Ok(())
}

fn validate_finite_projection(projection: &Projection, name: &str) -> Result<(), NnError> {
    projection
        .validate_retained_parameters()
        .map_err(|error| match error {
            NnError::Backend(message) => NnError::Backend(format!("Qwen3.5 {name}: {message}")),
            other => other,
        })
}

fn validate_finite(values: &[f32], expected: usize, name: &str) -> Result<(), NnError> {
    if values.len() != expected {
        return Err(NnError::Shape {
            expected,
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NnError::Backend(format!(
            "Qwen3.5 {name} contains a non-finite value"
        )));
    }
    Ok(())
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

fn axis(value: u32, name: &str) -> Result<usize, NnError> {
    if value == 0 {
        return Err(invalid_config(format!("Qwen3.5 {name} must be non-zero")));
    }
    usize::try_from(value).map_err(|_| invalid_config(format!("Qwen3.5 {name} does not fit usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| NnError::Backend(format!("Qwen3.5 {name} extent overflow")))
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate Qwen3.5 {name} for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn invalid_config(reason: impl Into<String>) -> NnError {
    NnError::MissingConfig(reason.into())
}
