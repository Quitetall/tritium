//! Qwen3.5-family gated full-attention token mixer.
//!
//! This is deliberately separate from [`TransformerBlock`](super::TransformerBlock):
//! Qwen3.6 interleaves full-attention and DeltaNet layers, and its full-attention
//! projection fuses query and sigmoid-gate lanes per head. Encoding that graph in
//! the homogeneous BitNet block would make invalid hybrid states representable.

use std::sync::Arc;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::layers::{Projection, ProjectionActivationMode};
use crate::ops::{gqa_attention, rmsnorm_zero_centered, rope_apply_partial_neox};
use crate::qwen35_config::{
    Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeType, Qwen35TextConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttentionSpec {
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    query_width: usize,
    key_value_width: usize,
    fused_query_width: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    theta_bits: u32,
    rms_norm_eps_bits: u32,
}

impl AttentionSpec {
    fn bind(config: &Qwen35TextConfig) -> Result<Self, NnError> {
        let attention = &config.full_attention;
        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let num_heads = axis(attention.num_heads, "num_attention_heads")?;
        let num_key_value_heads = axis(attention.num_key_value_heads, "num_key_value_heads")?;
        let head_dim = axis(attention.head_dim, "head_dim")?;
        let rotary_dim = axis(config.rope.rotary_dim, "rotary_dim")?;
        let max_position_embeddings =
            axis(config.max_position_embeddings, "max_position_embeddings")?;

        if !num_heads.is_multiple_of(num_key_value_heads) {
            return Err(invalid_config(
                "full-attention query heads must form integral GQA groups",
            ));
        }
        if !head_dim.is_multiple_of(2) || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
            return Err(invalid_config(
                "head_dim and rotary_dim must be even and rotary_dim must not exceed head_dim",
            ));
        }
        if attention.bias
            || attention.dropout != 0.0
            || attention.output_gate != Qwen35OutputGate::Sigmoid
            || attention.norm_weight_semantics
                != Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight
            || config.rope.rope_type != Qwen35RopeType::Default
            || !config.use_cache
        {
            return Err(invalid_config(
                "unsupported Qwen3.5 full-attention numeric semantics",
            ));
        }

        let derived_rotary = f64::from(attention.head_dim) * config.rope.partial_rotary_factor;
        if !config.rope.partial_rotary_factor.is_finite()
            || config.rope.partial_rotary_factor <= 0.0
            || config.rope.partial_rotary_factor > 1.0
            || derived_rotary != f64::from(config.rope.rotary_dim)
        {
            return Err(invalid_config(
                "partial_rotary_factor disagrees with the bound rotary dimension",
            ));
        }

        let theta = config.rope.theta as f32;
        let rms_norm_eps = config.rms_norm_eps as f32;
        if !config.rope.theta.is_finite()
            || !theta.is_finite()
            || theta <= 0.0
            || !config.rms_norm_eps.is_finite()
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
        {
            return Err(invalid_config(
                "RoPE theta and RMSNorm epsilon must be finite positive f32 values",
            ));
        }

        let query_width = checked_mul(num_heads, head_dim, "query width")?;
        let key_value_width = checked_mul(num_key_value_heads, head_dim, "key/value width")?;
        let fused_query_width = checked_mul(query_width, 2, "fused query/gate width")?;
        Ok(Self {
            hidden_size,
            num_heads,
            num_key_value_heads,
            head_dim,
            query_width,
            key_value_width,
            fused_query_width,
            rotary_dim,
            max_position_embeddings,
            theta_bits: theta.to_bits(),
            rms_norm_eps_bits: rms_norm_eps.to_bits(),
        })
    }

    fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    fn rms_norm_eps(self) -> f32 {
        f32::from_bits(self.rms_norm_eps_bits)
    }
}

/// The four bias-free projections and two per-head zero-centered norm vectors.
///
/// Projection matrices retain their native implementation (dense reference,
/// host-packed SALT, resident SALT V2, or deployed ternary). Geometry is checked
/// when these weights are bound into [`Qwen35FullAttention`]. With hidden width
/// `H`, query heads `Nq`, KV heads `Nkv`, and head width `D`, row-major projection
/// shapes are:
///
/// - `q_proj`: `[2 * Nq * D, H]`, grouped per head as
///   `[Q head lanes..., sigmoid-gate head lanes...]`. It is not a global
///   `[all Q..., all gates...]` split.
/// - `k_proj` and `v_proj`: `[Nkv * D, H]`.
/// - `o_proj`: `[H, Nq * D]`.
/// - `q_norm` and `k_norm`: `[D]`, shared across their respective heads.
///
/// All four projections must use one [`ProjectionActivationMode`]. Mixed f32/A8
/// arithmetic is rejected instead of silently invalidating an evidence rung.
#[allow(missing_debug_implementations)]
pub struct Qwen35FullAttentionWeights {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl Qwen35FullAttentionWeights {
    /// Collect an unbound Qwen3.5 full-attention weight set.
    #[must_use]
    pub fn new(
        q_proj: Projection,
        k_proj: Projection,
        v_proj: Projection,
        o_proj: Projection,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        }
    }

    /// Validate this retained weight set against one text geometry without
    /// publishing an executable attention module.
    pub(crate) fn validate_for_config(
        &self,
        config: &Qwen35TextConfig,
    ) -> Result<ProjectionActivationMode, NnError> {
        let spec = AttentionSpec::bind(config)?;
        self.validate_for_spec(&spec)
    }

    fn validate_for_spec(&self, spec: &AttentionSpec) -> Result<ProjectionActivationMode, NnError> {
        validate_projection(&self.q_proj, spec.fused_query_width, spec.hidden_size)?;
        validate_projection(&self.k_proj, spec.key_value_width, spec.hidden_size)?;
        validate_projection(&self.v_proj, spec.key_value_width, spec.hidden_size)?;
        validate_projection(&self.o_proj, spec.hidden_size, spec.query_width)?;
        let activation_mode = self.q_proj.activation_mode();
        if [&self.k_proj, &self.v_proj, &self.o_proj]
            .into_iter()
            .any(|projection| projection.activation_mode() != activation_mode)
        {
            return Err(invalid_config(
                "full-attention projections must use one activation arithmetic mode",
            ));
        }
        validate_len(self.q_norm.len(), spec.head_dim)?;
        validate_len(self.k_norm.len(), spec.head_dim)?;
        if self
            .q_norm
            .iter()
            .chain(&self.k_norm)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen3.5 Q/K norm weights contain a non-finite value".to_owned(),
            ));
        }
        Ok(activation_mode)
    }
}

/// Typed KV state for one [`Qwen35FullAttention`] stream.
///
/// The private bound spec preserves the KV-head/head-dimension factorization;
/// equal flattened row widths from different attention geometries cannot be
/// accidentally cross-wired. A private mixer identity also rejects caches from
/// a different layer even when every dimension is equal.
#[derive(Debug)]
pub struct Qwen35FullAttentionCache {
    spec: AttentionSpec,
    mixer_identity: Arc<MixerIdentity>,
    inner: KvCache,
}

impl Qwen35FullAttentionCache {
    /// Whole-model transaction watermark used by the hybrid Qwen runner.
    #[must_use]
    pub(crate) const fn committed_len(&self) -> usize {
        self.inner.len
    }

    /// Validate a whole-model rollback target without mutating cache state.
    ///
    /// A hybrid runner preflights every layer before rolling any layer back, so
    /// the subsequent [`Self::rollback_to`] calls cannot fail midway through a
    /// model-wide rollback.
    pub(crate) fn preflight_rollback_to(&self, base: usize) -> Result<(), NnError> {
        if base <= self.inner.len {
            Ok(())
        } else {
            Err(NnError::Shape {
                expected: self.inner.len,
                got: base,
            })
        }
    }

    /// Restore a target already accepted by [`Self::preflight_rollback_to`].
    ///
    /// This path only truncates retained vectors and cannot allocate, panic, or
    /// return an error. An invalid target is a guarded no-op; callers use the
    /// preflight to report that invariant violation before mutation begins.
    pub(crate) fn rollback_to(&mut self, base: usize) {
        if self.preflight_rollback_to(base).is_ok() {
            self.inner.rollback_to(base);
        }
    }

    /// Number of committed tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.committed_len()
    }

    /// Whether no tokens are committed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Maximum number of tokens accepted by this cache.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.inner.max_ctx
    }

    /// Read-only committed rotated keys, `[len, n_kv_head, head_dim]`.
    #[must_use]
    pub fn keys(&self) -> &[f32] {
        self.inner.view().0
    }

    /// Read-only committed values, `[len, n_kv_head, head_dim]`.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        self.inner.view().1
    }

    /// Start a fresh sequence without freeing the retained cache allocation.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Standalone Qwen3.5-family gated full-attention token mixer.
///
/// Input is already decoder-normalized `[sequence, hidden]`. The module owns the
/// Qwen-specific fused query/gate split, per-head Q/K normalization, partial
/// NeoX RoPE, causal GQA, sigmoid output gate, and output projection.
#[allow(missing_debug_implementations)]
pub struct Qwen35FullAttention {
    spec: AttentionSpec,
    identity: Arc<MixerIdentity>,
    activation_mode: ProjectionActivationMode,
    weights: Qwen35FullAttentionWeights,
}

#[derive(Debug)]
struct MixerIdentity;

impl Qwen35FullAttention {
    /// Bind a typed Qwen3.5 text geometry to an exact full-attention weight set.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for contradictory numeric semantics or
    /// unsafe geometry, [`NnError::Shape`] for any projection/norm mismatch, or
    /// [`NnError::Backend`] for non-finite projection/Q/K norm weights.
    pub fn new(
        config: &Qwen35TextConfig,
        weights: Qwen35FullAttentionWeights,
    ) -> Result<Self, NnError> {
        let spec = AttentionSpec::bind(config)?;
        let activation_mode = weights.validate_for_spec(&spec)?;
        Ok(Self {
            spec,
            identity: Arc::new(MixerIdentity),
            activation_mode,
            weights,
        })
    }

    /// Activation arithmetic shared by every bound projection.
    #[must_use]
    pub const fn activation_mode(&self) -> ProjectionActivationMode {
        self.activation_mode
    }

    /// Create a cache that retains this mixer's exact head factorization.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] if `max_context` is zero or exceeds the bound
    /// model context, or [`NnError::Backend`] if cache geometry overflows.
    pub fn new_cache(&self, max_context: usize) -> Result<Qwen35FullAttentionCache, NnError> {
        if max_context == 0 || max_context > self.spec.max_position_embeddings {
            return Err(NnError::Shape {
                expected: self.spec.max_position_embeddings,
                got: max_context,
            });
        }
        let inner = KvCache::try_new(
            max_context,
            self.spec.num_key_value_heads,
            self.spec.head_dim,
        )?;
        Ok(Qwen35FullAttentionCache {
            spec: self.spec,
            mixer_identity: Arc::clone(&self.identity),
            inner,
        })
    }

    /// Mix already-normalized token rows and transactionally extend `cache`.
    ///
    /// `normalized` and `out` are row-major `[sequence, hidden]`; `positions`
    /// is `[sequence]` and supplies one scalar RoPE coordinate per input row.
    /// `positions` are scalar language positions and are independent of causal
    /// cache ordering; each must be below the configured maximum. Scalar
    /// positions are valid for text because Qwen's temporal/height/width MRoPE
    /// axes coincide there. Multimodal positions are intentionally unsupported.
    /// Output is staged and published only after every projection and cache
    /// operation succeeds. Any post-append error restores the prior logical
    /// cache contents and leaves `out` untouched.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for input/output/position/capacity mismatch,
    /// [`NnError::Backend`] for a cache bound to another mixer, or an
    /// error from a projection/normalization/attention operation.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        normalized: &[f32],
        positions: &[usize],
        cache: &mut Qwen35FullAttentionCache,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        if cache.spec != self.spec || !Arc::ptr_eq(&cache.mixer_identity, &self.identity) {
            return Err(NnError::Backend(
                "Qwen3.5 full-attention cache belongs to a different mixer".to_owned(),
            ));
        }
        let sequence = positions.len();
        if sequence == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        let hidden_len = checked_buffer_len(sequence, self.spec.hidden_size, normalized.len())?;
        if normalized.len() != hidden_len {
            return Err(NnError::Shape {
                expected: hidden_len,
                got: normalized.len(),
            });
        }
        if out.len() != hidden_len {
            return Err(NnError::Shape {
                expected: hidden_len,
                got: out.len(),
            });
        }

        let checkpoint = cache.committed_len();
        let new_context = checkpoint.checked_add(sequence).ok_or(NnError::Shape {
            expected: cache.inner.max_ctx,
            got: usize::MAX,
        })?;
        if new_context > cache.inner.max_ctx {
            return Err(NnError::Shape {
                expected: cache.inner.max_ctx,
                got: new_context,
            });
        }
        for &position in positions {
            if position >= self.spec.max_position_embeddings {
                return Err(NnError::Shape {
                    expected: self.spec.max_position_embeddings,
                    got: position,
                });
            }
        }

        let fused_len =
            checked_buffer_len(sequence, self.spec.fused_query_width, normalized.len())?;
        let query_len = checked_buffer_len(sequence, self.spec.query_width, normalized.len())?;
        let key_value_len =
            checked_buffer_len(sequence, self.spec.key_value_width, normalized.len())?;
        let mut fused_query = zeroed_scratch(fused_len, "fused query/gate")?;
        let mut query = zeroed_scratch(query_len, "query")?;
        let mut gate = zeroed_scratch(query_len, "attention gate")?;
        let mut key = zeroed_scratch(key_value_len, "key")?;
        let mut value = zeroed_scratch(key_value_len, "value")?;

        self.weights
            .q_proj
            .forward(backend, normalized, sequence, &mut fused_query)?;
        self.weights
            .k_proj
            .forward(backend, normalized, sequence, &mut key)?;
        self.weights
            .v_proj
            .forward(backend, normalized, sequence, &mut value)?;
        deinterleave_query_gate(
            &fused_query,
            &mut query,
            &mut gate,
            sequence,
            self.spec.num_heads,
            self.spec.head_dim,
        );
        normalize_heads(
            &mut query,
            &self.weights.q_norm,
            self.spec.head_dim,
            self.spec.rms_norm_eps(),
        )?;
        normalize_heads(
            &mut key,
            &self.weights.k_norm,
            self.spec.head_dim,
            self.spec.rms_norm_eps(),
        )?;
        rope_apply_partial_neox(
            &mut query,
            positions,
            self.spec.num_heads,
            self.spec.head_dim,
            self.spec.rotary_dim,
            self.spec.theta(),
        )?;
        rope_apply_partial_neox(
            &mut key,
            positions,
            self.spec.num_key_value_heads,
            self.spec.head_dim,
            self.spec.rotary_dim,
            self.spec.theta(),
        )?;

        // Allocate every module-owned staging buffer before mutating the cache.
        // Post-append operations may return errors, but they do not need another
        // allocation from this module before rollback is possible.
        let mut attention = zeroed_scratch(query_len, "attention output")?;
        let mut staged_output = zeroed_scratch(hidden_len, "projected output")?;
        cache.inner.append(&key, &value, sequence)?;
        let result = self.attend_and_project(
            backend,
            sequence,
            checkpoint,
            &cache.inner,
            AttentionPass {
                query: &query,
                gate: &gate,
                attention: &mut attention,
                output: &mut staged_output,
            },
        );
        match result {
            Ok(()) => {
                out.copy_from_slice(&staged_output);
                Ok(())
            }
            Err(error) => {
                cache.rollback_to(checkpoint);
                Err(error)
            }
        }
    }

    fn attend_and_project(
        &self,
        backend: &dyn TernaryBackend,
        sequence: usize,
        causal_offset: usize,
        cache: &KvCache,
        pass: AttentionPass<'_>,
    ) -> Result<(), NnError> {
        let (key, value, context) = cache.view();
        gqa_attention(
            pass.query,
            key,
            value,
            sequence,
            context,
            self.spec.num_heads,
            self.spec.num_key_value_heads,
            self.spec.head_dim,
            1.0 / (self.spec.head_dim as f32).sqrt(),
            causal_offset,
            pass.attention,
        )?;
        for (value, &gate) in pass.attention.iter_mut().zip(pass.gate) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
        self.weights
            .o_proj
            .forward(backend, pass.attention, sequence, pass.output)
    }
}

struct AttentionPass<'a> {
    query: &'a [f32],
    gate: &'a [f32],
    attention: &'a mut [f32],
    output: &'a mut [f32],
}

fn deinterleave_query_gate(
    fused: &[f32],
    query: &mut [f32],
    gate: &mut [f32],
    sequence: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let head_pair_width = head_dim * 2;
    let fused_token_width = num_heads * head_pair_width;
    let query_token_width = num_heads * head_dim;
    for token in 0..sequence {
        for head in 0..num_heads {
            let fused_start = token * fused_token_width + head * head_pair_width;
            let output_start = token * query_token_width + head * head_dim;
            query[output_start..output_start + head_dim]
                .copy_from_slice(&fused[fused_start..fused_start + head_dim]);
            gate[output_start..output_start + head_dim]
                .copy_from_slice(&fused[fused_start + head_dim..fused_start + head_pair_width]);
        }
    }
}

fn normalize_heads(
    values: &mut [f32],
    weights: &[f32],
    head_dim: usize,
    epsilon: f32,
) -> Result<(), NnError> {
    let mut normalized = zeroed_scratch(head_dim, "head normalization")?;
    for head in values.chunks_exact_mut(head_dim) {
        rmsnorm_zero_centered(head, weights, epsilon, &mut normalized)?;
        head.copy_from_slice(&normalized);
    }
    Ok(())
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
) -> Result<(), NnError> {
    validate_len(projection.n_out(), expected_out)?;
    validate_len(projection.k_in(), expected_in)?;
    projection.validate_retained_parameters()
}

fn validate_len(got: usize, expected: usize) -> Result<(), NnError> {
    if got == expected {
        Ok(())
    } else {
        Err(NnError::Shape { expected, got })
    }
}

fn axis(value: u32, name: &str) -> Result<usize, NnError> {
    if value == 0 {
        return Err(invalid_config(format!("{name} must be non-zero")));
    }
    usize::try_from(value).map_err(|_| invalid_config(format!("{name} does not fit usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, NnError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_config(format!("Qwen3.5 {name} overflow")))
}

fn checked_buffer_len(rows: usize, width: usize, got: usize) -> Result<usize, NnError> {
    rows.checked_mul(width).ok_or(NnError::Shape {
        expected: usize::MAX,
        got,
    })
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate Qwen3.5 {name} scratch for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn invalid_config(reason: impl Into<String>) -> NnError {
    NnError::MissingConfig(reason.into())
}

#[cfg(test)]
mod tests {
    use tritium_cpu::CpuBackend;

    use super::{Qwen35FullAttention, Qwen35FullAttentionWeights};
    use crate::layers::{DenseLinear, Projection};
    use crate::qwen35_config::{
        Qwen35DeltaNetConfig, Qwen35Dtype, Qwen35FullAttentionConfig, Qwen35LayerType,
        Qwen35MtpConfig, Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig,
        Qwen35RopeType, Qwen35TextConfig,
    };

    fn dense(values: &[f32], n_out: usize, k_in: usize) -> Projection {
        Projection::Dense(DenseLinear::new_exact(values.to_vec(), n_out, k_in).unwrap())
    }

    fn layer() -> Qwen35FullAttention {
        let config = Qwen35TextConfig {
            model_type: "qwen3_5_text".to_owned(),
            num_hidden_layers: 1,
            hidden_size: 2,
            intermediate_size: 4,
            vocab_size: 8,
            max_position_embeddings: 8,
            full_attention_interval: 1,
            layer_types: vec![Qwen35LayerType::FullAttention],
            full_attention: Qwen35FullAttentionConfig {
                num_heads: 1,
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
        };
        let identity = [1.0, 0.0, 0.0, 1.0];
        let weights = Qwen35FullAttentionWeights::new(
            dense(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 4, 2),
            dense(&identity, 2, 2),
            dense(&identity, 2, 2),
            dense(&identity, 2, 2),
            vec![0.0; 2],
            vec![0.0; 2],
        );
        Qwen35FullAttention::new(&config, weights).unwrap()
    }

    #[test]
    fn provisional_append_can_be_preflighted_and_rolled_back_exactly() {
        let layer = layer();
        let backend = CpuBackend::new();
        let mut cache = layer.new_cache(4).unwrap();
        let mut prefix_output = [f32::NAN; 2];
        layer
            .forward(&backend, &[1.0, 0.0], &[0], &mut cache, &mut prefix_output)
            .unwrap();
        let base = cache.committed_len();
        let prefix_keys = cache.keys().to_vec();
        let prefix_values = cache.values().to_vec();

        let mut first_suffix_output = [f32::NAN; 2];
        layer
            .forward(
                &backend,
                &[0.0, 1.0],
                &[1],
                &mut cache,
                &mut first_suffix_output,
            )
            .unwrap();
        let appended_keys = cache.keys().to_vec();
        let appended_values = cache.values().to_vec();
        assert_eq!(cache.committed_len(), base + 1);

        cache.preflight_rollback_to(base).unwrap();
        cache.rollback_to(base);
        assert_eq!(cache.committed_len(), base);
        assert_eq!(cache.keys(), prefix_keys);
        assert_eq!(cache.values(), prefix_values);

        let mut replayed_suffix_output = [f32::NAN; 2];
        layer
            .forward(
                &backend,
                &[0.0, 1.0],
                &[1],
                &mut cache,
                &mut replayed_suffix_output,
            )
            .unwrap();
        assert_eq!(replayed_suffix_output, first_suffix_output);
        assert_eq!(cache.keys(), appended_keys);
        assert_eq!(cache.values(), appended_values);
    }

    #[test]
    fn rollback_preflight_rejects_a_future_target_without_mutation() {
        let layer = layer();
        let backend = CpuBackend::new();
        let mut cache = layer.new_cache(4).unwrap();
        let mut output = [f32::NAN; 2];
        layer
            .forward(&backend, &[1.0, 0.0], &[0], &mut cache, &mut output)
            .unwrap();
        let keys = cache.keys().to_vec();
        let values = cache.values().to_vec();

        let error = cache
            .preflight_rollback_to(cache.committed_len() + 1)
            .unwrap_err();

        assert!(matches!(error, crate::NnError::Shape { .. }));
        assert_eq!(cache.committed_len(), 1);
        assert_eq!(cache.keys(), keys);
        assert_eq!(cache.values(), values);
    }

    #[test]
    fn direct_invalid_rollback_is_a_noop() {
        let layer = layer();
        let backend = CpuBackend::new();
        let mut cache = layer.new_cache(4).unwrap();
        let mut output = [f32::NAN; 2];
        layer
            .forward(&backend, &[1.0, 0.0], &[0], &mut cache, &mut output)
            .unwrap();
        let keys = cache.keys().to_vec();
        let values = cache.values().to_vec();

        cache.rollback_to(cache.committed_len() + 1);

        assert_eq!(cache.committed_len(), 1);
        assert_eq!(cache.keys(), keys);
        assert_eq!(cache.values(), values);
    }
}
