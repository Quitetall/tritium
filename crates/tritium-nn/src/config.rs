//! Model dimensions, read from GGUF metadata.

use tritium_format::{GgufFile, GgufValue};

use crate::error::NnError;

const MAX_MODEL_LAYERS: u32 = 4_096;
const MAX_MODEL_AXIS: u32 = 16_000_000;
const MAX_CONTEXT_LENGTH: u32 = 16_000_000;

fn validate_gguf_arch(arch: &str) -> Result<(), NnError> {
    if matches!(arch, "bitnet" | "bitnet-b1.58") {
        Ok(())
    } else {
        Err(NnError::MissingMetadata(format!(
            "unsupported general.architecture `{arch}`"
        )))
    }
}

fn validate_model_geometry(
    n_layers: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_ff: u32,
    n_ctx: u32,
) -> Result<(), &'static str> {
    let q_width = u64::from(n_head) * u64::from(head_dim);
    let kv_width = u64::from(n_head_kv) * u64::from(head_dim);
    if n_layers == 0
        || n_embd == 0
        || n_head == 0
        || n_head_kv == 0
        || head_dim == 0
        || n_ff == 0
        || n_ctx == 0
        || !n_head.is_multiple_of(n_head_kv)
        || !head_dim.is_multiple_of(2)
        || n_layers > MAX_MODEL_LAYERS
        || n_embd > MAX_MODEL_AXIS
        || n_head > MAX_MODEL_AXIS
        || n_head_kv > MAX_MODEL_AXIS
        || head_dim > MAX_MODEL_AXIS
        || n_ff > MAX_MODEL_AXIS
        || n_ctx > MAX_CONTEXT_LENGTH
        || q_width > u64::from(MAX_MODEL_AXIS)
        || kv_width > u64::from(MAX_MODEL_AXIS)
    {
        Err("model dimensions are unsupported, unsafe, or do not divide evenly")
    } else {
        Ok(())
    }
}

const fn finite_positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

/// Architecture/shape parameters needed to run a decoder model. Field names follow
/// the GGUF `{arch}.*` metadata convention (llama-family, which BitNet reuses).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// `general.architecture`, e.g. `"bitnet"` / `"llama"`.
    pub arch: String,
    /// Transformer blocks (`{arch}.block_count`).
    pub n_layers: u32,
    /// Hidden size (`{arch}.embedding_length`).
    pub n_embd: u32,
    /// Attention heads (`{arch}.attention.head_count`).
    pub n_head: u32,
    /// KV heads for GQA (`{arch}.attention.head_count_kv`).
    pub n_head_kv: u32,
    /// Per-head dimension. Usually `n_embd / n_head`, but Qwen3 **decouples** it (an
    /// explicit `head_dim` in `config.json`, so `n_head · head_dim` may exceed `n_embd`).
    pub head_dim: u32,
    /// FFN intermediate size (`{arch}.feed_forward_length`).
    pub n_ff: u32,
    /// Max context (`{arch}.context_length`).
    pub n_ctx: u32,
    /// RoPE base frequency (`{arch}.rope.freq_base`).
    pub rope_theta: f32,
    /// RMSNorm epsilon (`{arch}.attention.layer_norm_rms_epsilon`).
    pub rms_eps: f32,
}

impl ModelConfig {
    /// Per-head dimension (the explicit [`head_dim`](Self::head_dim) field).
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// GQA grouping factor, `n_head / n_head_kv`.
    #[must_use]
    pub const fn gqa_group(&self) -> u32 {
        self.n_head / self.n_head_kv
    }

    /// Read a [`ModelConfig`] from a parsed GGUF file's metadata.
    ///
    /// # Errors
    /// [`NnError::MissingMetadata`] if `general.architecture` or any required
    /// `{arch}.*` key is absent or the wrong type, or if the GGUF architecture is
    /// not one of the BitNet dialects supported by the GGUF execution path.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, NnError> {
        let arch = file
            .get_metadata("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| NnError::MissingMetadata("general.architecture".to_owned()))?
            .to_owned();
        validate_gguf_arch(&arch)?;

        let u32_key = |suffix: &str| -> Result<u32, NnError> {
            let key = format!("{arch}.{suffix}");
            let value = file
                .get_metadata(&key)
                .and_then(GgufValue::as_u64)
                .ok_or_else(|| NnError::MissingMetadata(key.clone()))?;
            u32::try_from(value).map_err(|_| NnError::MissingMetadata(format!("{key} exceeds u32")))
        };
        let f32_key = |suffix: &str| -> Result<f32, NnError> {
            let key = format!("{arch}.{suffix}");
            match file.get_metadata(&key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                _ => Err(NnError::MissingMetadata(key)),
            }
        };

        let n_embd = u32_key("embedding_length")?;
        let n_head = u32_key("attention.head_count")?;
        let n_head_kv = u32_key("attention.head_count_kv")?;
        let n_layers = u32_key("block_count")?;
        let n_ff = u32_key("feed_forward_length")?;
        let n_ctx = u32_key("context_length")?;
        let head_dim = n_embd
            .checked_div(n_head)
            .filter(|_| n_embd.is_multiple_of(n_head))
            .ok_or_else(|| {
                NnError::MissingMetadata("embedding width must divide attention heads".to_owned())
            })?;
        validate_model_geometry(n_layers, n_embd, n_head, n_head_kv, head_dim, n_ff, n_ctx)
            .map_err(|message| NnError::MissingMetadata(message.to_owned()))?;
        let rope_theta = f32_key("rope.freq_base")?;
        let rms_eps = f32_key("attention.layer_norm_rms_epsilon")?;
        if !finite_positive(rope_theta) || !finite_positive(rms_eps) {
            return Err(NnError::MissingMetadata(
                "RoPE theta and RMS epsilon must be finite and positive".to_owned(),
            ));
        }
        Ok(ModelConfig {
            n_layers,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            n_ctx,
            rope_theta,
            rms_eps,
            arch,
        })
    }

    /// Read a [`ModelConfig`] + [`ArchSpec`] from a HuggingFace `config.json` value
    /// (standard transformer-family keys). `num_key_value_heads` defaults to
    /// `num_attention_heads` (MHA); `rope_theta` to `10000`; `max_position_embeddings`
    /// to `4096`. `hidden_act == "silu"` selects [`MlpKind::SwiGlu`]; `"relu2"`
    /// selects [`MlpKind::Relu2`]. Other activation families are rejected.
    /// Sub-norms are off (a standard HF model has none); `qk_norm`/`qkv_bias` are off
    /// (descriptor flags for later plans).
    ///
    /// `config_json` is the raw `config.json` contents.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] if JSON or required fields are malformed, geometry is
    /// unsafe/unsupported, RoPE scaling or scalar values are unsupported, or the declared
    /// model/activation family is not implemented by the shared decoder skeleton.
    pub fn from_hf_config(config_json: &str) -> Result<(ModelConfig, ArchSpec), NnError> {
        let json: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| NnError::MissingConfig(format!("invalid config.json: {e}")))?;
        // RoPE scaling (llama3 / linear / dynamic) changes the inverse-frequency schedule;
        // the plain `theta^(-2j/d)` RoPE here would silently diverge. Reject loudly until a
        // later plan implements the scaling (SmolLM2 / Llama-2 have none).
        if json.get("rope_scaling").is_some_and(|v| !v.is_null()) {
            return Err(NnError::MissingConfig(
                "rope_scaling (llama3/linear/dynamic) not yet supported".to_owned(),
            ));
        }
        let req_u32 = |key: &str| -> Result<u32, NnError> {
            let value = json
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| NnError::MissingConfig(key.to_owned()))?;
            u32::try_from(value).map_err(|_| NnError::MissingConfig(format!("{key} exceeds u32")))
        };
        let opt_u32 = |key: &str, default: u32| -> Result<u32, NnError> {
            let Some(value) = json.get(key) else {
                return Ok(default);
            };
            let value = value
                .as_u64()
                .ok_or_else(|| NnError::MissingConfig(key.to_owned()))?;
            u32::try_from(value).map_err(|_| NnError::MissingConfig(format!("{key} exceeds u32")))
        };
        let opt_f32 = |key: &str, default: f32| -> Result<f32, NnError> {
            let Some(value) = json.get(key) else {
                return Ok(default);
            };
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| NnError::MissingConfig(key.to_owned()))
        };

        let n_head = req_u32("num_attention_heads")?;
        let n_head_kv = opt_u32("num_key_value_heads", n_head)?;
        let arch = json
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                json.get("architectures")
                    .and_then(|a| a.get(0))
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("unknown")
            .to_owned();

        let rms_eps = json
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .ok_or_else(|| NnError::MissingConfig("rms_norm_eps".to_owned()))?;

        let n_embd = req_u32("hidden_size")?;
        let n_layers = req_u32("num_hidden_layers")?;
        let n_ff = req_u32("intermediate_size")?;
        let n_ctx = opt_u32("max_position_embeddings", 4096)?;
        let head_dim = match json.get("head_dim") {
            Some(_) => opt_u32("head_dim", 0)?,
            None => n_embd
                .checked_div(n_head)
                .filter(|_| n_embd.is_multiple_of(n_head))
                .ok_or_else(|| {
                    NnError::MissingConfig(
                        "hidden_size must divide num_attention_heads when head_dim is absent"
                            .to_owned(),
                    )
                })?,
        };
        validate_model_geometry(n_layers, n_embd, n_head, n_head_kv, head_dim, n_ff, n_ctx)
            .map_err(|message| NnError::MissingConfig(message.to_owned()))?;
        let rope_theta = opt_f32("rope_theta", 10_000.0)?;
        if !finite_positive(rope_theta) || !finite_positive(rms_eps) {
            return Err(NnError::MissingConfig(
                "rope_theta and rms_norm_eps must be finite and positive".to_owned(),
            ));
        }
        let cfg = ModelConfig {
            n_layers,
            n_embd,
            n_head,
            n_head_kv,
            // Qwen3 sets an explicit head_dim that need not equal n_embd/n_head.
            head_dim,
            n_ff,
            n_ctx,
            rope_theta,
            rms_eps,
            arch,
        };

        if !matches!(cfg.arch.as_str(), "bitnet" | "llama" | "qwen2" | "qwen3") {
            return Err(NnError::MissingConfig(format!(
                "unsupported model_type `{}`",
                cfg.arch
            )));
        }
        let hidden_act = match json.get("hidden_act") {
            None => "silu",
            Some(value) => value
                .as_str()
                .ok_or_else(|| NnError::MissingConfig("hidden_act".to_owned()))?,
        };
        let mlp = match hidden_act {
            "silu" => MlpKind::SwiGlu,
            "relu2" => MlpKind::Relu2,
            other => {
                return Err(NnError::MissingConfig(format!(
                    "unsupported hidden_act `{other}`"
                )));
            }
        };
        let tied_embeddings = match json.get("tie_word_embeddings") {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| NnError::MissingConfig("tie_word_embeddings".to_owned()))?,
        };
        let spec = ArchSpec {
            mlp,
            // A standard HF model has no BitNet sub-norms (those come from the GGUF path).
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings,
        };
        Ok((cfg, spec))
    }
}

/// The feed-forward activation/shape family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// BitNet: `down(ffn_sub_norm(relu(gate(x))² ⊙ up(x)))`.
    Relu2,
    /// Llama/Qwen: `down(silu(gate(x)) ⊙ up(x))` (no sub-norm).
    SwiGlu,
}

/// Architecture-variation axes beyond the shared llama-family dims in [`ModelConfig`].
/// Defaults ([`ArchSpec::bitnet`]) describe BitNet, so the existing GGUF path is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchSpec {
    /// Feed-forward family.
    pub mlp: MlpKind,
    /// BitNet applies `attn_sub_norm` to the attention output before `o_proj`.
    pub attn_sub_norm: bool,
    /// BitNet applies `ffn_sub_norm` inside the MLP (implied by [`MlpKind::Relu2`]).
    pub ffn_sub_norm: bool,
    /// Qwen3: per-head RMSNorm on Q and K after projection. (Descriptor only until the
    /// op lands in a later plan; the HF load path asserts `false`.)
    pub qk_norm: bool,
    /// Qwen2/2.5: additive bias on q/k/v projections. (Descriptor only for now.)
    pub qkv_bias: bool,
    /// `false` ⇒ a separate `lm_head.weight`; `true` ⇒ tie to the token embedding.
    pub tied_embeddings: bool,
}

impl ArchSpec {
    /// BitNet-2B4T defaults — what the GGUF load path assumes today.
    #[must_use]
    pub fn bitnet() -> Self {
        Self {
            mlp: MlpKind::Relu2,
            attn_sub_norm: true,
            ffn_sub_norm: true,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig {
            arch: "bitnet".to_owned(),
            n_layers: 30,
            n_embd: 2560,
            n_head: 20,
            n_head_kv: 5,
            head_dim: 128,
            n_ff: 6912,
            n_ctx: 4096,
            rope_theta: 500000.0,
            rms_eps: 1e-5,
        }
    }

    #[test]
    fn derived_dims() {
        let c = cfg();
        assert_eq!(c.head_dim(), 128); // 2560 / 20
        assert_eq!(c.gqa_group(), 4); // 20 / 5
    }

    #[test]
    fn gguf_path_accepts_only_bitnet_dialects() {
        assert!(validate_gguf_arch("bitnet").is_ok());
        assert!(validate_gguf_arch("bitnet-b1.58").is_ok());
        assert!(matches!(
            validate_gguf_arch("llama"),
            Err(NnError::MissingMetadata(_))
        ));
    }

    #[test]
    fn from_hf_config_smollm2_swiglu() {
        let (c, spec) = ModelConfig::from_hf_config(
            r#"{
                "model_type":"llama","hidden_size":576,"num_hidden_layers":30,
                "num_attention_heads":9,"num_key_value_heads":3,"intermediate_size":1536,
                "rope_theta":100000.0,"rms_norm_eps":1e-5,"hidden_act":"silu",
                "tie_word_embeddings":true,"max_position_embeddings":8192
            }"#,
        )
        .expect("from_hf_config");
        assert_eq!(c.n_embd, 576);
        assert_eq!(c.n_layers, 30);
        assert_eq!(c.n_head, 9);
        assert_eq!(c.n_head_kv, 3);
        assert_eq!(c.gqa_group(), 3); // 9 / 3
        assert_eq!(c.head_dim(), 64); // 576 / 9
        assert_eq!(c.n_ff, 1536);
        assert!((c.rope_theta - 100_000.0).abs() < 1e-3);
        assert!((c.rms_eps - 1e-5).abs() < 1e-9);
        assert_eq!(spec.mlp, MlpKind::SwiGlu);
        assert!(spec.tied_embeddings);
        assert!(!spec.attn_sub_norm && !spec.ffn_sub_norm);
        assert!(!spec.qk_norm && !spec.qkv_bias);
    }

    #[test]
    fn from_hf_config_defaults_kv_to_mha_and_relu2() {
        // No num_key_value_heads ⇒ MHA (kv == heads); non-silu act ⇒ Relu2; untied.
        let (c, spec) = ModelConfig::from_hf_config(
            r#"{"model_type":"bitnet","hidden_size":128,"num_hidden_layers":2,
                "num_attention_heads":4,"intermediate_size":256,
                "rms_norm_eps":1e-6,"hidden_act":"relu2","tie_word_embeddings":false}"#,
        )
        .expect("from_hf_config");
        assert_eq!(c.n_head_kv, 4); // defaulted to n_head
        assert_eq!(c.gqa_group(), 1);
        assert!((c.rope_theta - 10_000.0).abs() < 1e-3); // default θ
        assert_eq!(spec.mlp, MlpKind::Relu2);
        assert!(!spec.tied_embeddings);
    }

    #[test]
    fn from_hf_config_rejects_rope_scaling() {
        // A Llama-3.x-style config with rope_scaling must be rejected (not silently run
        // with the wrong inverse-frequency schedule).
        let err = ModelConfig::from_hf_config(
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5,
                "hidden_act":"silu","rope_theta":500000.0,
                "rope_scaling":{"rope_type":"llama3","factor":32.0}}"#,
        )
        .unwrap_err();
        assert!(matches!(err, NnError::MissingConfig(_)), "got {err:?}");
    }

    #[test]
    fn from_hf_config_missing_required_key_errors() {
        // No hidden_size ⇒ MissingConfig.
        let err = ModelConfig::from_hf_config(r#"{"num_hidden_layers":2,"num_attention_heads":4}"#)
            .unwrap_err();
        assert!(matches!(err, NnError::MissingConfig(_)), "got {err:?}");
    }

    #[test]
    fn from_hf_config_rejects_zero_heads_and_u32_truncation() {
        for json in [
            r#"{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":0,
                "intermediate_size":16,"rms_norm_eps":1e-5}"#,
            r#"{"hidden_size":4294967296,"num_hidden_layers":1,"num_attention_heads":2,
                "intermediate_size":16,"rms_norm_eps":1e-5}"#,
            r#"{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
                "num_key_value_heads":0,"intermediate_size":16,"rms_norm_eps":1e-5}"#,
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":4294967295,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5}"#,
            r#"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"head_dim":3,"intermediate_size":16,
                "rms_norm_eps":1e-5}"#,
        ] {
            assert!(
                matches!(
                    ModelConfig::from_hf_config(json),
                    Err(NnError::MissingConfig(_))
                ),
                "invalid dimensions accepted: {json}"
            );
        }
    }

    #[test]
    fn from_hf_config_rejects_nonfinite_or_nonpositive_scalars() {
        for tail in [
            r#""rms_norm_eps":0"#,
            r#""rms_norm_eps":1e-5,"rope_theta":1e400"#,
        ] {
            let json = format!(
                r#"{{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
                    "intermediate_size":16,{tail}}}"#
            );
            assert!(matches!(
                ModelConfig::from_hf_config(&json),
                Err(NnError::MissingConfig(_))
            ));
        }
    }

    #[test]
    fn from_hf_config_rejects_mistyped_or_unsupported_semantics() {
        for json in [
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5,
                "rope_theta":"100000"}"#,
            r#"{"model_type":"gemma","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5}"#,
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5,
                "hidden_act":"gelu"}"#,
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5,
                "hidden_act":17}"#,
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"intermediate_size":16,"rms_norm_eps":1e-5,
                "tie_word_embeddings":"true"}"#,
        ] {
            assert!(matches!(
                ModelConfig::from_hf_config(json),
                Err(NnError::MissingConfig(_))
            ));
        }
    }
}
