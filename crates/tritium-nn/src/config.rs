//! Model dimensions, read from GGUF metadata.

use tritium_format::{GgufFile, GgufValue};

use crate::error::NnError;

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
    /// `{arch}.*` key is absent or the wrong type.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, NnError> {
        let arch = file
            .get_metadata("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| NnError::MissingMetadata("general.architecture".to_owned()))?
            .to_owned();

        let u32_key = |suffix: &str| -> Result<u32, NnError> {
            let key = format!("{arch}.{suffix}");
            file.get_metadata(&key)
                .and_then(GgufValue::as_u64)
                .map(|v| v as u32)
                .ok_or(NnError::MissingMetadata(key))
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
        Ok(ModelConfig {
            n_layers: u32_key("block_count")?,
            n_embd,
            n_head,
            n_head_kv: u32_key("attention.head_count_kv")?,
            head_dim: n_embd / n_head,
            n_ff: u32_key("feed_forward_length")?,
            n_ctx: u32_key("context_length")?,
            rope_theta: f32_key("rope.freq_base")?,
            rms_eps: f32_key("attention.layer_norm_rms_epsilon")?,
            arch,
        })
    }

    /// Read a [`ModelConfig`] + [`ArchSpec`] from a HuggingFace `config.json` value
    /// (standard transformer-family keys). `num_key_value_heads` defaults to
    /// `num_attention_heads` (MHA); `rope_theta` to `10000`; `max_position_embeddings`
    /// to `4096`. `hidden_act == "silu"` ⇒ [`MlpKind::SwiGlu`], else [`MlpKind::Relu2`].
    /// Sub-norms are off (a standard HF model has none); `qk_norm`/`qkv_bias` are off
    /// (descriptor flags for later plans).
    ///
    /// `config_json` is the raw `config.json` contents.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] if `config_json` is not valid JSON, or a required key
    /// (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`,
    /// `rms_norm_eps`) is absent or mistyped.
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
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| NnError::MissingConfig(key.to_owned()))
        };
        let opt_u32 = |key: &str, default: u32| -> u32 {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(default, |v| v as u32)
        };
        let opt_f32 = |key: &str, default: f32| -> f32 {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(default, |v| v as f32)
        };

        let n_head = req_u32("num_attention_heads")?;
        let n_head_kv = opt_u32("num_key_value_heads", n_head);
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
        let cfg = ModelConfig {
            n_layers: req_u32("num_hidden_layers")?,
            n_embd,
            n_head,
            n_head_kv,
            // Qwen3 sets an explicit head_dim that need not equal n_embd/n_head.
            head_dim: opt_u32("head_dim", n_embd / n_head),
            n_ff: req_u32("intermediate_size")?,
            n_ctx: opt_u32("max_position_embeddings", 4096),
            rope_theta: opt_f32("rope_theta", 10_000.0),
            rms_eps,
            arch,
        };

        let hidden_act = json
            .get("hidden_act")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("silu");
        let mlp = if hidden_act == "silu" {
            MlpKind::SwiGlu
        } else {
            MlpKind::Relu2
        };
        let spec = ArchSpec {
            mlp,
            // A standard HF model has no BitNet sub-norms (those come from the GGUF path).
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: json
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
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
            r#"{"model_type":"x","hidden_size":128,"num_hidden_layers":2,
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
}
