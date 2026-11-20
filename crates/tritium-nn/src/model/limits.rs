//! Shared language-stream limits. Logical cache payload planning, not measured memory.
#[cfg(feature = "tokenizer")]
use crate::NnError;
#[cfg(feature = "tokenizer")]
use crate::{ModelConfig, Qwen35CheckpointConfig, Qwen35LayerType};

/// Explicit bounds for one loaded language stream.
///
/// Cache bytes cover maximum logical host KV/current+staged DeltaNet payloads,
/// not weights, scratch, device recurrence, MTP, Vec slack or allocator/driver
/// overhead. This is not a whole-model physical-memory admission receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamLimits {
    /// Selected stream context; must not exceed source capacity.
    pub max_context: usize,
    /// Maximum tokens in any single forward segment, including initial prefill.
    pub max_segment_tokens: usize,
    /// Cap on the graph's planned maximum logical host-cache payload.
    pub max_host_cache_payload_bytes: u64,
}

impl StreamLimits {
    #[cfg(feature = "tokenizer")]
    pub(super) fn validate(self) -> Result<(), NnError> {
        if self.max_context == 0
            || self.max_segment_tokens == 0
            || self.max_host_cache_payload_bytes == 0
            || self.max_segment_tokens > self.max_context
        {
            return Err(invalid(
                "limits must be positive and segment must fit selected context",
            ));
        }
        Ok(())
    }
}

/// Immutable policy and calculated host-cache payload for an actual loaded stream.
///
/// Constructed by the common loader before upload, not by callers. This observation
/// excludes other memory domains and is not independent release qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheAdmission {
    limits: StreamLimits,
    attention_kv_bytes: u64,
    delta_cache_bytes: u64,
    total: u64,
}

impl CacheAdmission {
    /// Actual selected limits retained by the executor.
    #[must_use]
    pub const fn limits(self) -> StreamLimits {
        self.limits
    }
    /// Maximum logical host key+value payload at selected context.
    #[must_use]
    pub const fn attention_kv_bytes(self) -> u64 {
        self.attention_kv_bytes
    }
    /// Logical host current+staged convolution and recurrence payload.
    #[must_use]
    pub const fn delta_cache_bytes(self) -> u64 {
        self.delta_cache_bytes
    }
    /// Sum of the two explicitly covered host-cache domains.
    #[must_use]
    pub const fn logical_host_cache_payload_bytes(self) -> u64 {
        self.total
    }

    #[cfg(feature = "tokenizer")]
    fn finish(
        limits: StreamLimits,
        source_context: usize,
        kv: u64,
        delta: u64,
    ) -> Result<Self, NnError> {
        limits.validate()?;
        if limits.max_context > source_context {
            return Err(invalid("selected context exceeds source capacity"));
        }
        let total = kv
            .checked_add(delta)
            .ok_or_else(|| invalid("cache payload sum overflow"))?;
        if total > limits.max_host_cache_payload_bytes {
            return Err(invalid(&format!(
                "host cache requires {total} payload bytes, cap is {}",
                limits.max_host_cache_payload_bytes
            )));
        }
        Ok(Self {
            limits,
            attention_kv_bytes: kv,
            delta_cache_bytes: delta,
            total,
        })
    }

    #[cfg(feature = "tokenizer")]
    pub(super) fn standard(config: &ModelConfig, limits: StreamLimits) -> Result<Self, NnError> {
        limits.validate()?;
        if config.n_head == 0 || config.n_head_kv == 0 {
            return Err(invalid("cache heads must be positive"));
        }
        let kv = multiply(&[
            2,
            limits.max_context as u64,
            u64::from(config.n_head_kv),
            u64::from(config.head_dim()),
            4,
            u64::from(config.n_layers),
        ])?;
        Self::finish(limits, config.n_ctx as usize, kv, 0)
    }

    #[cfg(feature = "tokenizer")]
    pub(super) fn hybrid(
        config: &Qwen35CheckpointConfig,
        limits: StreamLimits,
    ) -> Result<Self, NnError> {
        limits.validate()?;
        let text = &config.text;
        let mut attention = 0_u64;
        let mut delta = 0_u64;
        for kind in &text.layer_types {
            match kind {
                Qwen35LayerType::FullAttention => attention += 1,
                Qwen35LayerType::DeltaNet => delta += 1,
            }
        }
        let a = &text.full_attention;
        let kv = if attention == 0 {
            0
        } else {
            multiply(&[
                2,
                limits.max_context as u64,
                u64::from(a.num_key_value_heads),
                u64::from(a.head_dim),
                4,
                attention,
            ])?
        };
        let delta_bytes = if delta == 0 {
            0
        } else {
            let d = &text.delta_net;
            let key = multiply(&[u64::from(d.num_key_heads), u64::from(d.key_head_dim)])?;
            let value = multiply(&[u64::from(d.num_value_heads), u64::from(d.value_head_dim)])?;
            let width = key
                .checked_mul(2)
                .and_then(|v| v.checked_add(value))
                .ok_or_else(|| invalid("DeltaNet convolution width overflow"))?;
            let conv = multiply(&[width, u64::from(d.conv_kernel_dim)])?;
            let recurrent = multiply(&[
                u64::from(d.num_value_heads),
                u64::from(d.key_head_dim),
                u64::from(d.value_head_dim),
            ])?;
            let state = conv
                .checked_add(recurrent)
                .ok_or_else(|| invalid("DeltaNet state sum overflow"))?;
            multiply(&[2, 4, delta, state])?
        };
        Self::finish(
            limits,
            text.max_position_embeddings as usize,
            kv,
            delta_bytes,
        )
    }
}

#[cfg(feature = "tokenizer")]
fn invalid(message: &str) -> NnError {
    NnError::ResourceExhausted(format!("stream admission: {message}"))
}
#[cfg(feature = "tokenizer")]
fn multiply(values: &[u64]) -> Result<u64, NnError> {
    values.iter().try_fold(1_u64, |n, v| {
        n.checked_mul(*v)
            .ok_or_else(|| invalid("cache payload multiplication overflow"))
    })
}
