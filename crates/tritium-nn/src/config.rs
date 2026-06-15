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
    /// Per-head dimension, `n_embd / n_head`.
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.n_embd / self.n_head
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

        Ok(ModelConfig {
            n_layers: u32_key("block_count")?,
            n_embd: u32_key("embedding_length")?,
            n_head: u32_key("attention.head_count")?,
            n_head_kv: u32_key("attention.head_count_kv")?,
            n_ff: u32_key("feed_forward_length")?,
            n_ctx: u32_key("context_length")?,
            rope_theta: f32_key("rope.freq_base")?,
            rms_eps: f32_key("attention.layer_norm_rms_epsilon")?,
            arch,
        })
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
}
