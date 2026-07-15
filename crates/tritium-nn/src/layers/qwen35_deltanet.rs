//! Qwen3.5-family Gated DeltaNet token mixer.
//!
//! This module stays separate from [`TransformerBlock`](super::TransformerBlock).
//! Qwen3.6 interleaves recurrent DeltaNet and full-attention layers, so forcing
//! both through the homogeneous transformer cache would erase required state
//! geometry and make cross-layer cache wiring possible.

use std::sync::Arc;

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::{Projection, ProjectionActivationMode};
use crate::qwen35_config::{
    Qwen35Dtype, Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35TextConfig,
};

const QK_L2_EPSILON: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeltaNetSpec {
    hidden_size: usize,
    max_context: usize,
    conv_kernel_dim: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_width: usize,
    value_width: usize,
    conv_width: usize,
    conv_state_len: usize,
    recurrent_state_len: usize,
    rms_norm_eps_bits: u32,
}

impl DeltaNetSpec {
    fn bind(config: &Qwen35TextConfig) -> Result<Self, NnError> {
        let delta = &config.delta_net;
        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let max_context = axis(config.max_position_embeddings, "max_position_embeddings")?;
        let conv_kernel_dim = axis(delta.conv_kernel_dim, "linear_conv_kernel_dim")?;
        let num_key_heads = axis(delta.num_key_heads, "linear_num_key_heads")?;
        let num_value_heads = axis(delta.num_value_heads, "linear_num_value_heads")?;
        let key_head_dim = axis(delta.key_head_dim, "linear_key_head_dim")?;
        let value_head_dim = axis(delta.value_head_dim, "linear_value_head_dim")?;

        if !num_value_heads.is_multiple_of(num_key_heads) {
            return Err(invalid_config(
                "DeltaNet value heads must form integral query/key repeat groups",
            ));
        }
        if delta.state_arithmetic_dtype != Qwen35Dtype::Float32
            || delta.output_gate != Qwen35OutputGate::Swish
            || delta.gated_norm_weight_semantics
                != Qwen35NormWeightSemantics::UnitCenteredDirectWeight
            || !config.use_cache
        {
            return Err(invalid_config(
                "unsupported Qwen3.5 DeltaNet numeric or cache semantics",
            ));
        }

        let rms_norm_eps = config.rms_norm_eps as f32;
        if !config.rms_norm_eps.is_finite() || !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err(invalid_config(
                "DeltaNet RMSNorm epsilon must be a finite positive f32 value",
            ));
        }

        let key_width = checked_mul(num_key_heads, key_head_dim, "query/key width")?;
        let value_width = checked_mul(num_value_heads, value_head_dim, "value width")?;
        let conv_width = checked_add(
            checked_mul(key_width, 2, "query plus key width")?,
            value_width,
            "QKV convolution width",
        )?;
        let conv_state_len = checked_mul(conv_width, conv_kernel_dim, "convolution state")?;
        let recurrent_state_len = checked_mul(
            checked_mul(num_value_heads, key_head_dim, "recurrent key state")?,
            value_head_dim,
            "recurrent value state",
        )?;

        Ok(Self {
            hidden_size,
            max_context,
            conv_kernel_dim,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            key_width,
            value_width,
            conv_width,
            conv_state_len,
            recurrent_state_len,
            rms_norm_eps_bits: rms_norm_eps.to_bits(),
        })
    }

    fn rms_norm_eps(self) -> f32 {
        f32::from_bits(self.rms_norm_eps_bits)
    }
}

/// Five bias-free projections plus fp32 DeltaNet parameters.
///
/// With hidden width `H`, key heads `Nk`, value heads `Nv`, key width `Dk`,
/// value width `Dv`, and convolution kernel `K`, row-major shapes are:
///
/// - `qkv_proj`: `[2 * Nk * Dk + Nv * Dv, H]`, globally split as Q, K, V.
/// - `z_proj`: `[Nv * Dv, H]`; `b_proj` and `a_proj`: `[Nv, H]`.
/// - `out_proj`: `[H, Nv * Dv]`.
/// - `conv_weight`: `[2 * Nk * Dk + Nv * Dv, K]`, depthwise cross-correlation
///   weights where tap `K - 1` multiplies the newest raw projected QKV value.
/// - `norm_weight`: `[Dv]`; `dt_bias` and `a_log`: `[Nv]`.
///
/// All projections must share one [`ProjectionActivationMode`]. Recurrent and
/// convolution state remain fp32 even when checkpoint weights originated as
/// BF16. This is an intentional Tritium numeric policy, not a model-level parity
/// claim against every Hugging Face cache-storage mode.
#[allow(missing_debug_implementations)]
pub struct Qwen35DeltaNetWeights {
    qkv_proj: Projection,
    z_proj: Projection,
    b_proj: Projection,
    a_proj: Projection,
    out_proj: Projection,
    conv_weight: Vec<f32>,
    norm_weight: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
}

impl Qwen35DeltaNetWeights {
    /// Collect an unbound Qwen3.5 DeltaNet weight set.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qkv_proj: Projection,
        z_proj: Projection,
        b_proj: Projection,
        a_proj: Projection,
        out_proj: Projection,
        conv_weight: Vec<f32>,
        norm_weight: Vec<f32>,
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
    ) -> Self {
        Self {
            qkv_proj,
            z_proj,
            b_proj,
            a_proj,
            out_proj,
            conv_weight,
            norm_weight,
            dt_bias,
            a_log,
        }
    }
}

/// Typed recurrent state for one [`Qwen35DeltaNet`] stream.
///
/// Public views expose only committed state. Private staging permits a hybrid
/// model runner to stage every recurrent layer, then commit all layers only
/// after downstream execution succeeds.
#[derive(Debug)]
pub struct Qwen35DeltaNetCache {
    spec: DeltaNetSpec,
    mixer_identity: Arc<MixerIdentity>,
    len: usize,
    staged_len: Option<usize>,
    conv_current: Vec<f32>,
    conv_staging: Vec<f32>,
    recurrent_current: Vec<f32>,
    recurrent_staging: Vec<f32>,
}

impl Qwen35DeltaNetCache {
    /// Number of committed tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no tokens are committed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum configured stream length.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.spec.max_context
    }

    /// Depthwise raw-QKV history width.
    #[must_use]
    pub const fn conv_kernel_dim(&self) -> usize {
        self.spec.conv_kernel_dim
    }

    /// Raw-QKV channel count.
    #[must_use]
    pub const fn conv_width(&self) -> usize {
        self.spec.conv_width
    }

    /// Query/key heads before repeat-interleave expansion.
    #[must_use]
    pub const fn num_key_heads(&self) -> usize {
        self.spec.num_key_heads
    }

    /// Value/recurrent-state heads.
    #[must_use]
    pub const fn num_value_heads(&self) -> usize {
        self.spec.num_value_heads
    }

    /// Per-head query/key width.
    #[must_use]
    pub const fn key_head_dim(&self) -> usize {
        self.spec.key_head_dim
    }

    /// Per-head value width.
    #[must_use]
    pub const fn value_head_dim(&self) -> usize {
        self.spec.value_head_dim
    }

    /// Committed raw-QKV convolution history, channel-major `[conv_width, K]`.
    #[must_use]
    pub fn conv_state(&self) -> &[f32] {
        &self.conv_current
    }

    /// Committed fp32 recurrence, `[value_heads, key_head_dim, value_head_dim]`.
    #[must_use]
    pub fn recurrent_state(&self) -> &[f32] {
        &self.recurrent_current
    }

    /// Start a fresh contiguous stream without freeing state allocations.
    pub fn reset(&mut self) {
        self.len = 0;
        self.staged_len = None;
        self.conv_current.fill(0.0);
        self.conv_staging.fill(0.0);
        self.recurrent_current.fill(0.0);
        self.recurrent_staging.fill(0.0);
    }

    /// Pending committed length, used to preflight a whole-model transaction.
    pub(crate) const fn staged_len(&self) -> Option<usize> {
        self.staged_len
    }

    /// Commit after the caller preflights [`Self::staged_len`] on every layer.
    ///
    /// This mutation phase allocates nothing and cannot fail. A cache without a
    /// pending stage is a no-op; hybrid callers must reject that condition in
    /// their all-layer preflight before committing any cache.
    pub(crate) fn commit_staged(&mut self) {
        if let Some(staged_len) = self.staged_len.take() {
            std::mem::swap(&mut self.conv_current, &mut self.conv_staging);
            std::mem::swap(&mut self.recurrent_current, &mut self.recurrent_staging);
            self.len = staged_len;
        }
    }

    pub(crate) fn abort_staged(&mut self) {
        self.staged_len = None;
    }
}

/// Standalone Qwen3.5-family Gated DeltaNet token mixer.
///
/// Input is already decoder-normalized row-major `[sequence, hidden]`. Interface
/// accepts one contiguous, unpadded stream; padding masks and packed reset points
/// are deliberately unrepresentable. Initial prefill may contain any positive
/// sequence length. Once cache is nonempty, only one-token recurrent continuation
/// is accepted, matching the defined Transformers cache path and rejecting its
/// unsafe multi-token continuation fallback.
#[allow(missing_debug_implementations)]
pub struct Qwen35DeltaNet {
    spec: DeltaNetSpec,
    identity: Arc<MixerIdentity>,
    activation_mode: ProjectionActivationMode,
    weights: Qwen35DeltaNetWeights,
}

#[derive(Debug)]
struct MixerIdentity;

impl Qwen35DeltaNet {
    /// Bind typed Qwen3.5 geometry and numeric semantics to exact weights.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for unsafe/unsupported geometry or
    /// mixed projection arithmetic, [`NnError::Shape`] for any weight mismatch,
    /// or [`NnError::Backend`] for non-finite fp32 auxiliary weights.
    pub fn new(config: &Qwen35TextConfig, weights: Qwen35DeltaNetWeights) -> Result<Self, NnError> {
        let spec = DeltaNetSpec::bind(config)?;
        validate_projection(&weights.qkv_proj, spec.conv_width, spec.hidden_size)?;
        validate_projection(&weights.z_proj, spec.value_width, spec.hidden_size)?;
        validate_projection(&weights.b_proj, spec.num_value_heads, spec.hidden_size)?;
        validate_projection(&weights.a_proj, spec.num_value_heads, spec.hidden_size)?;
        validate_projection(&weights.out_proj, spec.hidden_size, spec.value_width)?;

        let activation_mode = weights.qkv_proj.activation_mode();
        if [
            &weights.z_proj,
            &weights.b_proj,
            &weights.a_proj,
            &weights.out_proj,
        ]
        .into_iter()
        .any(|projection| projection.activation_mode() != activation_mode)
        {
            return Err(invalid_config(
                "DeltaNet projections must use one activation arithmetic mode",
            ));
        }

        validate_len(weights.conv_weight.len(), spec.conv_state_len)?;
        validate_len(weights.norm_weight.len(), spec.value_head_dim)?;
        validate_len(weights.dt_bias.len(), spec.num_value_heads)?;
        validate_len(weights.a_log.len(), spec.num_value_heads)?;
        if weights
            .conv_weight
            .iter()
            .chain(&weights.norm_weight)
            .chain(&weights.dt_bias)
            .chain(&weights.a_log)
            .any(|value| !value.is_finite())
        {
            return Err(NnError::Backend(
                "Qwen3.5 DeltaNet auxiliary weights contain a non-finite value".to_owned(),
            ));
        }
        if weights.a_log.iter().any(|value| !value.exp().is_finite()) {
            return Err(NnError::Backend(
                "Qwen3.5 DeltaNet A_log exponent overflows fp32".to_owned(),
            ));
        }

        Ok(Self {
            spec,
            identity: Arc::new(MixerIdentity),
            activation_mode,
            weights,
        })
    }

    /// Activation arithmetic shared by all five projections.
    #[must_use]
    pub const fn activation_mode(&self) -> ProjectionActivationMode {
        self.activation_mode
    }

    /// Allocate zeroed fp32 current/staging state bound to this exact mixer.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Backend`] when any state allocation fails.
    pub fn new_cache(&self) -> Result<Qwen35DeltaNetCache, NnError> {
        Ok(Qwen35DeltaNetCache {
            spec: self.spec,
            mixer_identity: Arc::clone(&self.identity),
            len: 0,
            staged_len: None,
            conv_current: zeroed_scratch(self.spec.conv_state_len, "committed convolution state")?,
            conv_staging: zeroed_scratch(self.spec.conv_state_len, "staged convolution state")?,
            recurrent_current: zeroed_scratch(
                self.spec.recurrent_state_len,
                "committed recurrent state",
            )?,
            recurrent_staging: zeroed_scratch(
                self.spec.recurrent_state_len,
                "staged recurrent state",
            )?,
        })
    }

    /// Mix one contiguous stream segment and atomically commit recurrent state.
    ///
    /// `normalized` and `out` are row-major `[sequence, hidden]`. Any error leaves
    /// committed cache state and `out` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for layout/capacity mismatch,
    /// [`NnError::MissingConfig`] for multi-token continuation after prefill,
    /// [`NnError::Backend`] for wrong cache provenance/pending transaction, or an
    /// error from a projection.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        normalized: &[f32],
        sequence: usize,
        cache: &mut Qwen35DeltaNetCache,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        self.validate_forward(normalized, sequence, cache, out)?;
        let expected_staged_len = cache.len.checked_add(sequence).ok_or(NnError::Shape {
            expected: self.spec.max_context,
            got: usize::MAX,
        })?;
        let mut staged_output = zeroed_scratch(out.len(), "public DeltaNet output transaction")?;
        let result = self.stage_forward(backend, normalized, sequence, cache, &mut staged_output);
        match result {
            Ok(()) => {
                if cache.staged_len() != Some(expected_staged_len) {
                    cache.abort_staged();
                    return Err(NnError::Backend(
                        "Qwen3.5 DeltaNet staged length failed commit preflight".to_owned(),
                    ));
                }
                cache.commit_staged();
                out.copy_from_slice(&staged_output);
                Ok(())
            }
            Err(error) => {
                cache.abort_staged();
                Err(error)
            }
        }
    }

    /// Stage output and state without changing committed cache views.
    pub(crate) fn stage_forward(
        &self,
        backend: &dyn TernaryBackend,
        normalized: &[f32],
        sequence: usize,
        cache: &mut Qwen35DeltaNetCache,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        self.validate_forward(normalized, sequence, cache, out)?;
        let new_len = cache.len.checked_add(sequence).ok_or(NnError::Shape {
            expected: self.spec.max_context,
            got: usize::MAX,
        })?;

        let hidden_len = checked_buffer_len(sequence, self.spec.hidden_size, normalized.len())?;
        let qkv_len = checked_buffer_len(sequence, self.spec.conv_width, normalized.len())?;
        let value_len = checked_buffer_len(sequence, self.spec.value_width, normalized.len())?;
        let head_gate_len =
            checked_buffer_len(sequence, self.spec.num_value_heads, normalized.len())?;

        // Allocate every module-owned scratch buffer before projections or state
        // staging. Projection failures can only touch these private buffers.
        let mut raw_qkv = zeroed_scratch(qkv_len, "raw QKV projection")?;
        let mut z = zeroed_scratch(value_len, "DeltaNet output gate")?;
        let mut beta_logits = zeroed_scratch(head_gate_len, "DeltaNet beta logits")?;
        let mut decay_logits = zeroed_scratch(head_gate_len, "DeltaNet decay logits")?;
        let mut convolved = zeroed_scratch(qkv_len, "convolved QKV")?;
        let mut core = zeroed_scratch(value_len, "DeltaNet recurrent output")?;
        let mut normalized_core = zeroed_scratch(value_len, "gated RMSNorm output")?;
        let mut staged_output = zeroed_scratch(hidden_len, "projected DeltaNet output")?;

        self.weights
            .qkv_proj
            .forward(backend, normalized, sequence, &mut raw_qkv)?;
        self.weights
            .z_proj
            .forward(backend, normalized, sequence, &mut z)?;
        self.weights
            .b_proj
            .forward(backend, normalized, sequence, &mut beta_logits)?;
        self.weights
            .a_proj
            .forward(backend, normalized, sequence, &mut decay_logits)?;

        cache.conv_staging.copy_from_slice(&cache.conv_current);
        cache
            .recurrent_staging
            .copy_from_slice(&cache.recurrent_current);
        self.depthwise_causal_conv(&raw_qkv, sequence, &mut cache.conv_staging, &mut convolved);
        self.recurrent_forward(
            &convolved,
            &z,
            &beta_logits,
            &decay_logits,
            sequence,
            &mut cache.recurrent_staging,
            &mut core,
            &mut normalized_core,
        );
        self.weights
            .out_proj
            .forward(backend, &normalized_core, sequence, &mut staged_output)?;

        // Publish stage marker and output only after every fallible operation.
        cache.staged_len = Some(new_len);
        out.copy_from_slice(&staged_output);
        Ok(())
    }

    fn validate_forward(
        &self,
        normalized: &[f32],
        sequence: usize,
        cache: &Qwen35DeltaNetCache,
        out: &[f32],
    ) -> Result<(), NnError> {
        if cache.spec != self.spec || !Arc::ptr_eq(&cache.mixer_identity, &self.identity) {
            return Err(NnError::Backend(
                "Qwen3.5 DeltaNet cache belongs to a different mixer".to_owned(),
            ));
        }
        if cache.staged_len.is_some() {
            return Err(NnError::Backend(
                "Qwen3.5 DeltaNet cache already has a staged forward".to_owned(),
            ));
        }
        if sequence == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        if cache.len != 0 && sequence != 1 {
            return Err(invalid_config(
                "DeltaNet cached continuation must contain exactly one token",
            ));
        }
        let new_len = cache.len.checked_add(sequence).ok_or(NnError::Shape {
            expected: self.spec.max_context,
            got: usize::MAX,
        })?;
        if new_len > self.spec.max_context {
            return Err(NnError::Shape {
                expected: self.spec.max_context,
                got: new_len,
            });
        }
        let hidden_len = checked_buffer_len(sequence, self.spec.hidden_size, normalized.len())?;
        validate_len(normalized.len(), hidden_len)?;
        validate_len(out.len(), hidden_len)
    }

    fn depthwise_causal_conv(
        &self,
        raw_qkv: &[f32],
        sequence: usize,
        conv_state: &mut [f32],
        convolved: &mut [f32],
    ) {
        let kernel = self.spec.conv_kernel_dim;
        let width = self.spec.conv_width;
        for token in 0..sequence {
            let raw_row = &raw_qkv[token * width..(token + 1) * width];
            let out_row = &mut convolved[token * width..(token + 1) * width];
            for channel in 0..width {
                let state = &mut conv_state[channel * kernel..(channel + 1) * kernel];
                state.copy_within(1..kernel, 0);
                state[kernel - 1] = raw_row[channel];
                let weight = &self.weights.conv_weight[channel * kernel..(channel + 1) * kernel];
                let mut sum = 0.0f32;
                for tap in 0..kernel {
                    sum += state[tap] * weight[tap];
                }
                out_row[channel] = silu(sum);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn recurrent_forward(
        &self,
        convolved: &[f32],
        z: &[f32],
        beta_logits: &[f32],
        decay_logits: &[f32],
        sequence: usize,
        recurrent_state: &mut [f32],
        core: &mut [f32],
        normalized_core: &mut [f32],
    ) {
        let group_size = self.spec.num_value_heads / self.spec.num_key_heads;
        let query_scale = 1.0 / (self.spec.key_head_dim as f32).sqrt();
        let dk = self.spec.key_head_dim;
        let dv = self.spec.value_head_dim;

        for token in 0..sequence {
            let qkv = &convolved[token * self.spec.conv_width..(token + 1) * self.spec.conv_width];
            let query = &qkv[..self.spec.key_width];
            let key = &qkv[self.spec.key_width..2 * self.spec.key_width];
            let value = &qkv[2 * self.spec.key_width..];
            let gate_base = token * self.spec.num_value_heads;
            let value_base = token * self.spec.value_width;

            for value_head in 0..self.spec.num_value_heads {
                let key_head = value_head / group_size;
                let q = &query[key_head * dk..(key_head + 1) * dk];
                let k = &key[key_head * dk..(key_head + 1) * dk];
                let v = &value[value_head * dv..(value_head + 1) * dv];
                let q_inv = l2_inverse(q) * query_scale;
                let k_inv = l2_inverse(k);
                let beta = sigmoid(beta_logits[gate_base + value_head]);
                let g = -self.weights.a_log[value_head].exp()
                    * softplus(
                        decay_logits[gate_base + value_head] + self.weights.dt_bias[value_head],
                    );
                let decay = g.exp();
                let state_base = value_head * dk * dv;

                for state_value in &mut recurrent_state[state_base..state_base + dk * dv] {
                    *state_value *= decay;
                }
                for value_lane in 0..dv {
                    let mut memory = 0.0f32;
                    for key_lane in 0..dk {
                        memory += k[key_lane]
                            * k_inv
                            * recurrent_state[state_base + key_lane * dv + value_lane];
                    }
                    let delta = beta * (v[value_lane] - memory);
                    for key_lane in 0..dk {
                        recurrent_state[state_base + key_lane * dv + value_lane] +=
                            k[key_lane] * k_inv * delta;
                    }
                    let mut mixed = 0.0f32;
                    for key_lane in 0..dk {
                        mixed += q[key_lane]
                            * q_inv
                            * recurrent_state[state_base + key_lane * dv + value_lane];
                    }
                    core[value_base + value_head * dv + value_lane] = mixed;
                }
            }

            for value_head in 0..self.spec.num_value_heads {
                let row_start = value_base + value_head * dv;
                let row = &core[row_start..row_start + dv];
                let mut variance = 0.0f32;
                for &value in row {
                    variance += value * value;
                }
                variance /= dv as f32;
                let inverse_rms = 1.0 / (variance + self.spec.rms_norm_eps()).sqrt();
                for (value_lane, &row_value) in row.iter().enumerate() {
                    let lane = row_start + value_lane;
                    normalized_core[lane] = row_value
                        * inverse_rms
                        * self.weights.norm_weight[value_lane]
                        * silu(z[lane]);
                }
            }
        }
    }
}

fn l2_inverse(values: &[f32]) -> f32 {
    let mut squared_norm = 0.0f32;
    for &value in values {
        squared_norm += value * value;
    }
    1.0 / (squared_norm + QK_L2_EPSILON).sqrt()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
) -> Result<(), NnError> {
    validate_len(projection.n_out(), expected_out)?;
    validate_len(projection.k_in(), expected_in)
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

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, NnError> {
    left.checked_add(right)
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
            "allocate Qwen3.5 {name} for {len} f32 values: {error}"
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

    use super::*;
    use crate::layers::DenseLinear;
    use crate::qwen35_config::{
        Qwen35DeltaNetConfig, Qwen35FullAttentionConfig, Qwen35LayerType, Qwen35MtpConfig,
        Qwen35RopeConfig, Qwen35RopeType,
    };

    fn dense(values: &[f32], n_out: usize, k_in: usize) -> Projection {
        Projection::Dense(DenseLinear::new_exact(values.to_vec(), n_out, k_in).unwrap())
    }

    fn tiny_layer() -> Qwen35DeltaNet {
        let config = Qwen35TextConfig {
            model_type: "qwen3_5_text".to_owned(),
            num_hidden_layers: 1,
            hidden_size: 2,
            intermediate_size: 2,
            vocab_size: 4,
            max_position_embeddings: 4,
            full_attention_interval: 2,
            layer_types: vec![Qwen35LayerType::DeltaNet],
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
                conv_kernel_dim: 2,
                num_key_heads: 1,
                num_value_heads: 1,
                key_head_dim: 1,
                value_head_dim: 1,
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
        let weights = Qwen35DeltaNetWeights::new(
            dense(&[0.5, -0.2, -0.1, 0.4, 0.3, 0.25], 3, 2),
            dense(&[0.2, -0.3], 1, 2),
            dense(&[0.1, 0.2], 1, 2),
            dense(&[-0.2, 0.1], 1, 2),
            dense(&[0.4, -0.25], 2, 1),
            vec![0.2, 0.8, -0.1, 0.7, 0.3, 0.9],
            vec![1.1],
            vec![0.2],
            vec![-0.3],
        );
        Qwen35DeltaNet::new(&config, weights).unwrap()
    }

    #[test]
    fn staged_state_is_invisible_until_commit_and_abort_is_allocation_free() {
        let layer = tiny_layer();
        let backend = CpuBackend::new();
        let mut cache = layer.new_cache().unwrap();
        let zero_conv = cache.conv_state().to_vec();
        let zero_recurrent = cache.recurrent_state().to_vec();
        let mut output = [f32::NAN; 2];

        layer
            .stage_forward(&backend, &[1.0, -0.5], 1, &mut cache, &mut output)
            .unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.conv_state(), zero_conv);
        assert_eq!(cache.recurrent_state(), zero_recurrent);
        cache.abort_staged();
        assert_eq!(cache.len(), 0);

        layer
            .stage_forward(&backend, &[1.0, -0.5], 1, &mut cache, &mut output)
            .unwrap();
        assert_eq!(cache.staged_len(), Some(1));
        cache.commit_staged();
        assert_eq!(cache.len(), 1);
        assert!(cache.conv_state().iter().any(|&value| value != 0.0));
        let committed_conv = cache.conv_state().to_vec();
        let committed_recurrent = cache.recurrent_state().to_vec();

        layer
            .stage_forward(&backend, &[-0.25, 0.75], 1, &mut cache, &mut output)
            .unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.conv_state(), committed_conv);
        assert_eq!(cache.recurrent_state(), committed_recurrent);
        cache.abort_staged();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.conv_state(), committed_conv);
        assert_eq!(cache.recurrent_state(), committed_recurrent);

        layer
            .stage_forward(&backend, &[-0.25, 0.75], 1, &mut cache, &mut output)
            .unwrap();
        cache.reset();
        assert!(cache.is_empty());
        assert!(cache.conv_state().iter().all(|&value| value == 0.0));
        assert!(cache.recurrent_state().iter().all(|&value| value == 0.0));
        assert_eq!(cache.staged_len(), None);
        cache.commit_staged();
        assert!(cache.is_empty());
    }
}
