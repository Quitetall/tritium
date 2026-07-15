//! Typed Hugging Face configuration contract for Qwen3.5/Qwen3.6 checkpoints.
//!
//! Qwen3.6 checkpoints use the `qwen3_5` Transformers architecture.  They are
//! not homogeneous Qwen3 decoder checkpoints: the language configuration is
//! nested below `text_config` and describes a mixed Gated DeltaNet/full-
//! attention schedule.  This module deliberately stays separate from
//! [`crate::ModelConfig`] so the existing decoder cannot accept such a
//! checkpoint and silently execute the wrong graph.

use serde_json::{Map, Value};

use crate::NnError;

/// Hugging Face repository selected by the Qwen3.6-27B campaign.
pub const QWEN36_27B_REPOSITORY: &str = "Qwen/Qwen3.6-27B";
/// Immutable repository revision selected by the Qwen3.6-27B campaign.
pub const QWEN36_27B_REVISION: &str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";

const OUTER_MODEL_TYPE: &str = "qwen3_5";
const OUTER_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const MAX_CONFIG_BYTES: usize = 1_048_576;
const MAX_LAYERS: u32 = 4_096;
const MAX_AXIS: u32 = 16_000_000;
const MAX_CONTEXT: u32 = 16_000_000;
const MAX_CONV_KERNEL: u32 = 4_096;
const MAX_RECURRENT_STATE_VALUES_PER_BATCH: u64 = 16_000_000;

/// One decoder layer in a Qwen3.5-family mixed schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerType {
    /// Gated DeltaNet, called `linear_attention` by the checkpoint.
    DeltaNet,
    /// Causal grouped-query attention, called `full_attention` by the checkpoint.
    FullAttention,
}

/// Numeric storage/arithmetic type named by the checkpoint contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Dtype {
    /// Brain floating point (`bfloat16`).
    Bfloat16,
    /// IEEE single precision (`float32`).
    Float32,
}

/// How an RMSNorm parameter is centered and applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35NormWeightSemantics {
    /// Stored weights are initialized at zero and applied as `1 + weight`.
    ZeroCenteredOnePlusWeight,
    /// Stored weights are initialized at one and applied directly.
    UnitCenteredDirectWeight,
}

/// Supported attention-output gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35OutputGate {
    /// Multiply the full-attention value path by `sigmoid(gate)`.
    Sigmoid,
    /// Multiply the DeltaNet normalized output by `silu(gate)`.
    Swish,
}

/// RoPE family implemented by this typed contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35RopeType {
    /// Unscaled base-frequency RoPE (`rope_type = "default"`).
    Default,
}

/// Scope of the outer checkpoint's vision component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35VisionScope {
    /// A vision configuration is present but intentionally deferred by the
    /// language-plus-MTP adapter.
    PresentDeferred,
}

/// Full-attention geometry and semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35FullAttentionConfig {
    /// Query heads.
    pub num_heads: u32,
    /// Key/value heads used for grouped-query attention.
    pub num_key_value_heads: u32,
    /// Q/K/V width per head.
    pub head_dim: u32,
    /// Whether Q/K/V projections carry additive bias.
    pub bias: bool,
    /// Attention dropout probability.
    pub dropout: f64,
    /// Gate applied to the attention output.
    pub output_gate: Qwen35OutputGate,
    /// Q/K and decoder RMSNorm checkpoint-weight convention.
    pub norm_weight_semantics: Qwen35NormWeightSemantics,
}

/// Gated DeltaNet geometry and state semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DeltaNetConfig {
    /// Depthwise causal-convolution kernel length.
    pub conv_kernel_dim: u32,
    /// Number of key/query heads before expansion to value heads.
    pub num_key_heads: u32,
    /// Number of value/recurrent-state heads.
    pub num_value_heads: u32,
    /// Key/query width per head.
    pub key_head_dim: u32,
    /// Value width per head.
    pub value_head_dim: u32,
    /// Arithmetic/storage type of the recurrent DeltaNet state.
    pub state_arithmetic_dtype: Qwen35Dtype,
    /// Gate applied after the recurrent update and gated RMSNorm.
    pub output_gate: Qwen35OutputGate,
    /// Gated output RMSNorm checkpoint-weight convention.
    pub gated_norm_weight_semantics: Qwen35NormWeightSemantics,
}

/// RoPE and multimodal-RoPE metadata for the language core.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35RopeConfig {
    /// Base angular frequency.
    pub theta: f64,
    /// Fraction of a full-attention head rotated by RoPE.
    pub partial_rotary_factor: f64,
    /// Derived, validated integral rotary width.
    pub rotary_dim: u32,
    /// RoPE schedule family.
    pub rope_type: Qwen35RopeType,
    /// Whether temporal/height/width frequencies are interleaved.
    pub mrope_interleaved: bool,
    /// Half-dimension allocation for temporal, height, and width frequencies.
    pub mrope_section: [u32; 3],
}

/// Multi-token-prediction drafter configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35MtpConfig {
    /// Number of bundled MTP decoder layers.
    pub num_hidden_layers: u32,
    /// Whether the MTP drafter owns a separate token embedding.
    pub dedicated_embeddings: bool,
}

/// Validated nested `text_config` from a Qwen3.5-family checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35TextConfig {
    /// Must be `qwen3_5_text`.
    pub model_type: String,
    /// Language decoder blocks.
    pub num_hidden_layers: u32,
    /// Residual-stream width.
    pub hidden_size: u32,
    /// SwiGLU intermediate width.
    pub intermediate_size: u32,
    /// Padded token vocabulary.
    pub vocab_size: u32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: u32,
    /// Interval used by the checkpoint's DeltaNet/attention schedule.
    pub full_attention_interval: u32,
    /// Exact layer-by-layer execution schedule.
    pub layer_types: Vec<Qwen35LayerType>,
    /// Full-attention geometry.
    pub full_attention: Qwen35FullAttentionConfig,
    /// Gated DeltaNet geometry.
    pub delta_net: Qwen35DeltaNetConfig,
    /// RoPE/MRoPE parameters.
    pub rope: Qwen35RopeConfig,
    /// Decoder RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Source tensor storage type.
    pub source_dtype: Qwen35Dtype,
    /// Whether the cache semantics required by autoregressive decode are enabled.
    pub use_cache: bool,
    /// Whether the language embedding and LM head are tied.
    pub tied_embeddings: bool,
    /// Bundled MTP drafter.
    pub mtp: Qwen35MtpConfig,
}

/// Validated outer Qwen3.5-family multimodal checkpoint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35CheckpointConfig {
    /// Must be `qwen3_5`.
    pub model_type: String,
    /// Must contain only `Qwen3_5ForConditionalGeneration`.
    pub architecture: String,
    /// Whether the source claims to omit its non-language components.
    pub language_model_only: bool,
    /// Whether the outer embedding declaration is tied.
    pub tied_embeddings: bool,
    /// Nested language and MTP contract.
    pub text: Qwen35TextConfig,
    /// Explicitly deferred vision boundary.
    pub vision_scope: Qwen35VisionScope,
}

impl Qwen35CheckpointConfig {
    /// Parse and validate a raw Hugging Face root `config.json`.
    ///
    /// This accepts semantically equivalent small fixtures, but no implicit
    /// Transformers defaults. Every execution-relevant Qwen3.5 field must be
    /// present and use the semantics represented by this contract.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for malformed JSON, missing/mistyped
    /// fields, contradictory schedules, unsafe geometry, non-integral partial
    /// rotary width, or an unsupported architecture/numeric semantic.
    pub fn from_hf_config(config_json: &str) -> Result<Self, NnError> {
        if config_json.len() > MAX_CONFIG_BYTES {
            return Err(invalid("config.json exceeds 1 MiB safety limit"));
        }
        let value: Value = serde_json::from_str(config_json)
            .map_err(|error| invalid(format!("invalid config.json: {error}")))?;
        let root = value
            .as_object()
            .ok_or_else(|| invalid("config.json root must be an object"))?;

        require_exact_str(root, "model_type", "root", OUTER_MODEL_TYPE)?;
        let architecture = parse_architecture(root)?;
        let language_model_only = require_bool(root, "language_model_only", "root")?;
        if language_model_only {
            return Err(invalid(
                "root.language_model_only=true contradicts the required multimodal wrapper",
            ));
        }
        let tied_embeddings = require_bool(root, "tie_word_embeddings", "root")?;
        let vision = require_object(root, "vision_config", "root")?;
        if vision.is_empty() {
            return Err(invalid("root.vision_config must not be empty"));
        }
        let text_object = require_object(root, "text_config", "root")?;
        let text = parse_text_config(text_object)?;
        if tied_embeddings != text.tied_embeddings {
            return Err(invalid(
                "root.tie_word_embeddings and text_config.tie_word_embeddings disagree",
            ));
        }

        Ok(Self {
            model_type: OUTER_MODEL_TYPE.to_owned(),
            architecture,
            language_model_only,
            tied_embeddings,
            text,
            vision_scope: Qwen35VisionScope::PresentDeferred,
        })
    }

    /// Validate the exact campaign-pinned `Qwen/Qwen3.6-27B` architecture.
    ///
    /// Parsing alone proves the checkpoint belongs to the supported Qwen3.5
    /// semantic family. This additional gate binds the immutable repository
    /// revision and every geometry selected by the Qwen3.6-27B campaign.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] when `revision` is not the pinned
    /// revision or any campaign geometry/semantic differs.
    pub fn validate_pinned_qwen36_27b(&self, revision: &str) -> Result<(), NnError> {
        if revision != QWEN36_27B_REVISION {
            return Err(invalid(format!(
                "revision must be pinned Qwen3.6-27B revision {QWEN36_27B_REVISION}"
            )));
        }
        let text = &self.text;
        let expected_schedule: Vec<_> = (0..64)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::DeltaNet
                }
            })
            .collect();
        let matches = self.model_type == OUTER_MODEL_TYPE
            && self.architecture == OUTER_ARCHITECTURE
            && !self.language_model_only
            && !self.tied_embeddings
            && self.vision_scope == Qwen35VisionScope::PresentDeferred
            && text.model_type == TEXT_MODEL_TYPE
            && text.num_hidden_layers == 64
            && text.hidden_size == 5_120
            && text.intermediate_size == 17_408
            && text.vocab_size == 248_320
            && text.max_position_embeddings == 262_144
            && text.full_attention_interval == 4
            && text.layer_types == expected_schedule
            && text.full_attention.num_heads == 24
            && text.full_attention.num_key_value_heads == 4
            && text.full_attention.head_dim == 256
            && !text.full_attention.bias
            && text.full_attention.dropout == 0.0
            && text.full_attention.output_gate == Qwen35OutputGate::Sigmoid
            && text.full_attention.norm_weight_semantics
                == Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight
            && text.delta_net.conv_kernel_dim == 4
            && text.delta_net.num_key_heads == 16
            && text.delta_net.num_value_heads == 48
            && text.delta_net.key_head_dim == 128
            && text.delta_net.value_head_dim == 128
            && text.delta_net.state_arithmetic_dtype == Qwen35Dtype::Float32
            && text.delta_net.output_gate == Qwen35OutputGate::Swish
            && text.delta_net.gated_norm_weight_semantics
                == Qwen35NormWeightSemantics::UnitCenteredDirectWeight
            && text.rope.theta == 10_000_000.0
            && text.rope.partial_rotary_factor == 0.25
            && text.rope.rotary_dim == 64
            && text.rope.rope_type == Qwen35RopeType::Default
            && text.rope.mrope_interleaved
            && text.rope.mrope_section == [11, 11, 10]
            && text.rms_norm_eps == 1e-6
            && text.source_dtype == Qwen35Dtype::Bfloat16
            && text.use_cache
            && !text.tied_embeddings
            && text.mtp.num_hidden_layers == 1
            && !text.mtp.dedicated_embeddings;
        if matches {
            Ok(())
        } else {
            Err(invalid(format!(
                "configuration does not match pinned {QWEN36_27B_REPOSITORY} geometry"
            )))
        }
    }
}

fn parse_text_config(object: &Map<String, Value>) -> Result<Qwen35TextConfig, NnError> {
    const PATH: &str = "text_config";
    require_exact_str(object, "model_type", PATH, TEXT_MODEL_TYPE)?;
    require_exact_str(object, "hidden_act", PATH, "silu")?;
    require_exact_str(object, "dtype", PATH, "bfloat16")?;
    require_exact_str(object, "mamba_ssm_dtype", PATH, "float32")?;
    require_exact_str(object, "output_gate_type", PATH, "swish")?;
    require_exact_bool(object, "attn_output_gate", PATH, true)?;
    require_exact_bool(object, "attention_bias", PATH, false)?;
    require_exact_bool(object, "use_cache", PATH, true)?;
    reject_non_null(object, "rope_scaling", PATH)?;

    let num_hidden_layers = require_u32(object, "num_hidden_layers", PATH)?;
    let hidden_size = require_u32(object, "hidden_size", PATH)?;
    let intermediate_size = require_u32(object, "intermediate_size", PATH)?;
    let vocab_size = require_u32(object, "vocab_size", PATH)?;
    let max_position_embeddings = require_u32(object, "max_position_embeddings", PATH)?;
    let full_attention_interval = require_u32(object, "full_attention_interval", PATH)?;
    let rms_norm_eps = require_f64(object, "rms_norm_eps", PATH)?;
    let tied_embeddings = require_bool(object, "tie_word_embeddings", PATH)?;
    let attention_dropout = require_f64(object, "attention_dropout", PATH)?;
    if attention_dropout != 0.0 {
        return Err(invalid(
            "text_config.attention_dropout must be exactly zero",
        ));
    }

    validate_axis(
        num_hidden_layers,
        MAX_LAYERS,
        "text_config.num_hidden_layers",
    )?;
    validate_axis(hidden_size, MAX_AXIS, "text_config.hidden_size")?;
    validate_axis(intermediate_size, MAX_AXIS, "text_config.intermediate_size")?;
    validate_axis(vocab_size, MAX_AXIS, "text_config.vocab_size")?;
    validate_axis(
        max_position_embeddings,
        MAX_CONTEXT,
        "text_config.max_position_embeddings",
    )?;
    validate_axis(
        full_attention_interval,
        MAX_LAYERS,
        "text_config.full_attention_interval",
    )?;
    if !finite_positive(rms_norm_eps) {
        return Err(invalid(
            "text_config.rms_norm_eps must be finite and positive",
        ));
    }

    let layer_types = parse_layer_types(object, num_hidden_layers, full_attention_interval)?;
    let full_attention = parse_full_attention(object)?;
    let delta_net = parse_delta_net(object)?;
    let rope = parse_rope(object, full_attention.head_dim)?;
    let mtp_num_hidden_layers = require_u32(object, "mtp_num_hidden_layers", PATH)?;
    if mtp_num_hidden_layers > MAX_LAYERS {
        return Err(invalid("text_config.mtp_num_hidden_layers exceeds limit"));
    }
    let mtp = Qwen35MtpConfig {
        num_hidden_layers: mtp_num_hidden_layers,
        dedicated_embeddings: require_bool(object, "mtp_use_dedicated_embeddings", PATH)?,
    };

    Ok(Qwen35TextConfig {
        model_type: TEXT_MODEL_TYPE.to_owned(),
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        full_attention_interval,
        layer_types,
        full_attention,
        delta_net,
        rope,
        rms_norm_eps,
        source_dtype: Qwen35Dtype::Bfloat16,
        use_cache: true,
        tied_embeddings,
        mtp,
    })
}

fn parse_full_attention(object: &Map<String, Value>) -> Result<Qwen35FullAttentionConfig, NnError> {
    const PATH: &str = "text_config";
    let num_heads = require_u32(object, "num_attention_heads", PATH)?;
    let num_key_value_heads = require_u32(object, "num_key_value_heads", PATH)?;
    let head_dim = require_u32(object, "head_dim", PATH)?;
    for (value, name) in [
        (num_heads, "text_config.num_attention_heads"),
        (num_key_value_heads, "text_config.num_key_value_heads"),
        (head_dim, "text_config.head_dim"),
    ] {
        validate_axis(value, MAX_AXIS, name)?;
    }
    if !num_heads.is_multiple_of(num_key_value_heads) || !head_dim.is_multiple_of(2) {
        return Err(invalid(
            "full-attention heads must form integral GQA groups and head_dim must be even",
        ));
    }
    checked_width(num_heads, head_dim, "full-attention query width")?;
    checked_width(
        num_key_value_heads,
        head_dim,
        "full-attention key/value width",
    )?;

    Ok(Qwen35FullAttentionConfig {
        num_heads,
        num_key_value_heads,
        head_dim,
        bias: false,
        dropout: 0.0,
        output_gate: Qwen35OutputGate::Sigmoid,
        norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
    })
}

fn parse_delta_net(object: &Map<String, Value>) -> Result<Qwen35DeltaNetConfig, NnError> {
    const PATH: &str = "text_config";
    let conv_kernel_dim = require_u32(object, "linear_conv_kernel_dim", PATH)?;
    let num_key_heads = require_u32(object, "linear_num_key_heads", PATH)?;
    let num_value_heads = require_u32(object, "linear_num_value_heads", PATH)?;
    let key_head_dim = require_u32(object, "linear_key_head_dim", PATH)?;
    let value_head_dim = require_u32(object, "linear_value_head_dim", PATH)?;
    validate_axis(
        conv_kernel_dim,
        MAX_CONV_KERNEL,
        "text_config.linear_conv_kernel_dim",
    )?;
    for (value, name) in [
        (num_key_heads, "text_config.linear_num_key_heads"),
        (num_value_heads, "text_config.linear_num_value_heads"),
        (key_head_dim, "text_config.linear_key_head_dim"),
        (value_head_dim, "text_config.linear_value_head_dim"),
    ] {
        validate_axis(value, MAX_AXIS, name)?;
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(invalid(
            "DeltaNet value heads must be an integral expansion of key heads",
        ));
    }
    checked_width(num_key_heads, key_head_dim, "DeltaNet key width")?;
    checked_width(num_value_heads, value_head_dim, "DeltaNet value width")?;
    let state_values = u64::from(num_value_heads)
        .checked_mul(u64::from(key_head_dim))
        .and_then(|value| value.checked_mul(u64::from(value_head_dim)))
        .ok_or_else(|| invalid("DeltaNet recurrent-state geometry overflow"))?;
    if state_values > MAX_RECURRENT_STATE_VALUES_PER_BATCH {
        return Err(invalid(
            "DeltaNet recurrent-state geometry exceeds per-batch safety limit",
        ));
    }

    Ok(Qwen35DeltaNetConfig {
        conv_kernel_dim,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        state_arithmetic_dtype: Qwen35Dtype::Float32,
        output_gate: Qwen35OutputGate::Swish,
        gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
    })
}

fn parse_rope(object: &Map<String, Value>, head_dim: u32) -> Result<Qwen35RopeConfig, NnError> {
    const TEXT_PATH: &str = "text_config";
    const ROPE_PATH: &str = "text_config.rope_parameters";
    let partial_rotary_factor = require_f64(object, "partial_rotary_factor", TEXT_PATH)?;
    if !partial_rotary_factor.is_finite()
        || partial_rotary_factor <= 0.0
        || partial_rotary_factor > 1.0
    {
        return Err(invalid(
            "text_config.partial_rotary_factor must be finite and in (0, 1]",
        ));
    }
    let rope = require_object(object, "rope_parameters", TEXT_PATH)?;
    require_exact_str(rope, "rope_type", ROPE_PATH, "default")?;
    let nested_factor = require_f64(rope, "partial_rotary_factor", ROPE_PATH)?;
    if nested_factor != partial_rotary_factor {
        return Err(invalid(
            "text_config partial_rotary_factor declarations disagree",
        ));
    }
    let theta = require_f64(rope, "rope_theta", ROPE_PATH)?;
    if !finite_positive(theta) {
        return Err(invalid(
            "text_config.rope_parameters.rope_theta must be finite and positive",
        ));
    }
    let rotary_width = f64::from(head_dim) * partial_rotary_factor;
    let rounded = rotary_width.round();
    if rotary_width != rounded || rounded < 2.0 || rounded > f64::from(u32::MAX) {
        return Err(invalid(
            "partial_rotary_factor must produce an integral rotary dimension",
        ));
    }
    let rotary_dim = rounded as u32;
    if !rotary_dim.is_multiple_of(2) {
        return Err(invalid("partial rotary dimension must be even"));
    }
    let mrope_interleaved = require_bool(rope, "mrope_interleaved", ROPE_PATH)?;
    let mrope_section = parse_three_u32(rope, "mrope_section", ROPE_PATH)?;
    let section_sum = mrope_section
        .into_iter()
        .try_fold(0_u32, u32::checked_add)
        .ok_or_else(|| invalid("mrope_section sum overflow"))?;
    if section_sum.checked_mul(2) != Some(rotary_dim) {
        return Err(invalid(
            "mrope_section must partition half of the partial rotary dimension",
        ));
    }

    Ok(Qwen35RopeConfig {
        theta,
        partial_rotary_factor,
        rotary_dim,
        rope_type: Qwen35RopeType::Default,
        mrope_interleaved,
        mrope_section,
    })
}

fn parse_layer_types(
    object: &Map<String, Value>,
    num_hidden_layers: u32,
    full_attention_interval: u32,
) -> Result<Vec<Qwen35LayerType>, NnError> {
    let values = object
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("text_config.layer_types"))?;
    let expected_len = usize::try_from(num_hidden_layers)
        .map_err(|_| invalid("text_config.num_hidden_layers exceeds usize"))?;
    if values.len() != expected_len {
        return Err(invalid(
            "text_config.layer_types length must equal num_hidden_layers",
        ));
    }
    let mut layer_types = Vec::new();
    layer_types
        .try_reserve_exact(values.len())
        .map_err(|_| invalid("could not allocate text_config.layer_types"))?;
    for (index, value) in values.iter().enumerate() {
        let layer = match value.as_str() {
            Some("linear_attention") => Qwen35LayerType::DeltaNet,
            Some("full_attention") => Qwen35LayerType::FullAttention,
            _ => return Err(invalid(format!("text_config.layer_types[{index}]"))),
        };
        let layer_number = u32::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or_else(|| invalid("layer schedule index overflow"))?;
        let expected = if layer_number.is_multiple_of(full_attention_interval) {
            Qwen35LayerType::FullAttention
        } else {
            Qwen35LayerType::DeltaNet
        };
        if layer != expected {
            return Err(invalid(format!(
                "text_config.layer_types[{index}] contradicts full_attention_interval"
            )));
        }
        layer_types.push(layer);
    }
    Ok(layer_types)
}

fn parse_architecture(root: &Map<String, Value>) -> Result<String, NnError> {
    let architectures = root
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("root.architectures"))?;
    if architectures.len() != 1
        || architectures.first().and_then(Value::as_str) != Some(OUTER_ARCHITECTURE)
    {
        return Err(invalid(format!(
            "root.architectures must contain only {OUTER_ARCHITECTURE}"
        )));
    }
    Ok(OUTER_ARCHITECTURE.to_owned())
}

fn invalid(message: impl Into<String>) -> NnError {
    NnError::MissingConfig(message.into())
}

fn require_object<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a Map<String, Value>, NnError> {
    object
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("{path}.{key}")))
}

fn require_bool(object: &Map<String, Value>, key: &str, path: &str) -> Result<bool, NnError> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid(format!("{path}.{key}")))
}

fn require_exact_bool(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
    expected: bool,
) -> Result<(), NnError> {
    if require_bool(object, key, path)? == expected {
        Ok(())
    } else {
        Err(invalid(format!("unsupported {path}.{key} semantics")))
    }
}

fn require_exact_str(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
    expected: &str,
) -> Result<(), NnError> {
    if object.get(key).and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(invalid(format!("{path}.{key} must be `{expected}`")))
    }
}

fn require_u32(object: &Map<String, Value>, key: &str, path: &str) -> Result<u32, NnError> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("{path}.{key}")))?;
    u32::try_from(value).map_err(|_| invalid(format!("{path}.{key} exceeds u32")))
}

fn require_f64(object: &Map<String, Value>, key: &str, path: &str) -> Result<f64, NnError> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("{path}.{key}")))
}

fn parse_three_u32(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<[u32; 3], NnError> {
    let values = object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{path}.{key}")))?;
    if values.len() != 3 {
        return Err(invalid(format!("{path}.{key} must contain 3 values")));
    }
    let mut output = [0_u32; 3];
    for (index, value) in values.iter().enumerate() {
        let value = value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value != 0)
            .ok_or_else(|| invalid(format!("{path}.{key}[{index}]")))?;
        output[index] = value;
    }
    Ok(output)
}

fn reject_non_null(object: &Map<String, Value>, key: &str, path: &str) -> Result<(), NnError> {
    if object.get(key).is_some_and(|value| !value.is_null()) {
        Err(invalid(format!("unsupported {path}.{key} semantics")))
    } else {
        Ok(())
    }
}

fn validate_axis(value: u32, maximum: u32, name: &str) -> Result<(), NnError> {
    if value == 0 || value > maximum {
        Err(invalid(format!("{name} is zero or exceeds safety limit")))
    } else {
        Ok(())
    }
}

fn checked_width(count: u32, width: u32, name: &str) -> Result<u32, NnError> {
    count
        .checked_mul(width)
        .filter(|value| *value <= MAX_AXIS)
        .ok_or_else(|| invalid(format!("{name} exceeds safety limit")))
}

const fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;
    use serde_json::json;

    const PINNED_QWEN36_27B_CONFIG: &str = include_str!("../tests/fixtures/qwen36-27b-config.json");

    fn schedule(layers: u32, interval: u32) -> Vec<&'static str> {
        (1..=layers)
            .map(|layer| {
                if layer.is_multiple_of(interval) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    }

    fn config_value(layers: u32, interval: u32) -> Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "full_attention_interval": interval,
                "head_dim": 12,
                "hidden_act": "silu",
                "hidden_size": 16,
                "intermediate_size": 32,
                "layer_types": schedule(layers, interval),
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 2,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 2,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 128,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 2,
                "num_hidden_layers": layers,
                "num_key_value_heads": 1,
                "output_gate_type": "swish",
                "partial_rotary_factor": 0.5,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [1, 1, 1],
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 1000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 128
            },
            "tie_word_embeddings": false,
            "vision_config": {"model_type": "qwen3_5"}
        })
    }

    fn pinned_value() -> Value {
        serde_json::from_str(PINNED_QWEN36_27B_CONFIG).expect("pinned config fixture")
    }

    fn parse(value: &Value) -> Result<Qwen35CheckpointConfig, NnError> {
        Qwen35CheckpointConfig::from_hf_config(&value.to_string())
    }

    #[test]
    fn pinned_qwen36_27b_representative_passes() {
        let config = Qwen35CheckpointConfig::from_hf_config(PINNED_QWEN36_27B_CONFIG)
            .expect("parse exact pinned config");
        config
            .validate_pinned_qwen36_27b(QWEN36_27B_REVISION)
            .expect("validate pinned config");
        assert_eq!(config.text.layer_types.len(), 64);
        assert_eq!(config.text.rope.rotary_dim, 64);
        assert_eq!(
            config.text.delta_net.state_arithmetic_dtype,
            Qwen35Dtype::Float32
        );
        assert_eq!(config.vision_scope, Qwen35VisionScope::PresentDeferred);
    }

    #[test]
    fn small_semantically_valid_fixture_passes_family_parser() {
        let config = parse(&config_value(4, 2)).expect("parse tiny config");
        assert_eq!(
            config.text.layer_types,
            [
                Qwen35LayerType::DeltaNet,
                Qwen35LayerType::FullAttention,
                Qwen35LayerType::DeltaNet,
                Qwen35LayerType::FullAttention,
            ]
        );
        assert_eq!(config.text.rope.rotary_dim, 6);
        assert!(
            config
                .validate_pinned_qwen36_27b(QWEN36_27B_REVISION)
                .is_err()
        );
    }

    #[test]
    fn malformed_nested_config_is_rejected() {
        for replacement in [Value::Null, json!("not an object"), json!({})] {
            let mut value = config_value(4, 2);
            value["text_config"] = replacement;
            assert!(parse(&value).is_err());
        }
        assert!(Qwen35CheckpointConfig::from_hf_config(&" ".repeat(MAX_CONFIG_BYTES + 1)).is_err());
    }

    #[test]
    fn wrong_pattern_and_mismatched_layer_count_are_rejected() {
        let mut wrong_pattern = config_value(4, 2);
        wrong_pattern["text_config"]["layer_types"][0] = json!("full_attention");
        assert!(parse(&wrong_pattern).is_err());

        let mut wrong_count = config_value(4, 2);
        wrong_count["text_config"]["num_hidden_layers"] = json!(3);
        assert!(parse(&wrong_count).is_err());
    }

    #[test]
    fn unsafe_dimensions_are_rejected() {
        for (key, value) in [
            ("hidden_size", json!(0)),
            ("num_hidden_layers", json!(4097)),
            ("linear_conv_kernel_dim", json!(0)),
            ("linear_value_head_dim", json!(16000001)),
        ] {
            let mut config = config_value(4, 2);
            config["text_config"][key] = value;
            assert!(parse(&config).is_err(), "accepted unsafe {key}");
        }
    }

    #[test]
    fn non_integral_partial_rotary_width_is_rejected() {
        let mut value = config_value(4, 2);
        value["text_config"]["partial_rotary_factor"] = json!(0.3);
        value["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.3);
        assert!(parse(&value).is_err());
    }

    #[test]
    fn unsupported_execution_semantics_are_rejected() {
        let mutations = [
            ("dtype", json!("float16")),
            ("mamba_ssm_dtype", json!("bfloat16")),
            ("output_gate_type", json!("sigmoid")),
            ("attn_output_gate", json!(false)),
            ("attention_bias", json!(true)),
            ("attention_dropout", json!(0.1)),
            ("hidden_act", json!("gelu")),
            ("use_cache", json!(false)),
        ];
        for (key, replacement) in mutations {
            let mut value = config_value(4, 2);
            value["text_config"][key] = replacement;
            assert!(parse(&value).is_err(), "accepted unsupported {key}");
        }

        let mut scaled_rope = config_value(4, 2);
        scaled_rope["text_config"]["rope_parameters"]["rope_type"] = json!("dynamic");
        assert!(parse(&scaled_rope).is_err());
    }

    #[test]
    fn pinned_validator_rejects_wrong_revision_or_family_geometry() {
        let config = parse(&pinned_value()).expect("parse pinned config");
        assert!(config.validate_pinned_qwen36_27b("main").is_err());

        let mut other_pattern = pinned_value();
        other_pattern["text_config"]["full_attention_interval"] = json!(8);
        other_pattern["text_config"]["layer_types"] = json!(schedule(64, 8));
        let other = parse(&other_pattern).expect("parse other valid family geometry");
        assert!(
            other
                .validate_pinned_qwen36_27b(QWEN36_27B_REVISION)
                .is_err()
        );
    }

    #[test]
    fn generic_model_config_still_rejects_qwen35() {
        let flat = r#"{
            "model_type":"qwen3_5", "hidden_size":16, "num_hidden_layers":2,
            "num_attention_heads":2, "num_key_value_heads":1, "head_dim":8,
            "intermediate_size":32, "rms_norm_eps":1e-6, "hidden_act":"silu"
        }"#;
        let error = ModelConfig::from_hf_config(flat).expect_err("generic parser must reject");
        assert!(
            matches!(error, NnError::MissingConfig(message) if message.contains("unsupported model_type `qwen3_5`"))
        );
    }
}
