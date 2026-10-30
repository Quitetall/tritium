//! Optional bounded, transactional backend-owned causal GQA execution.

use crate::BackendError;

/// Row-major query and key/value head geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    /// Query heads per token.
    pub query_heads: usize,
    /// Key/value heads per token; divides query heads.
    pub kv_heads: usize,
    /// Coefficients per head.
    pub head_dim: usize,
}

/// Explicit bounds for retained KV and maximum-segment workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionLimits {
    /// Maximum committed plus staged context tokens.
    pub max_context: usize,
    /// Maximum new tokens in one stage.
    pub max_segment_tokens: usize,
    /// Owned f32 device payload limit, excluding allocator and driver overhead.
    pub max_device_payload_bytes: usize,
}

impl AttentionGeometry {
    /// Validate geometry and return query/KV row widths without allocating.
    ///
    /// # Errors
    /// Rejects zero dimensions, incomplete GQA groups and arithmetic overflow.
    pub fn widths(self) -> Result<(usize, usize), BackendError> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
        {
            return Err(invalid(
                "attention requires positive dimensions and whole GQA groups",
            ));
        }
        Ok((
            product(self.query_heads, self.head_dim)?,
            product(self.kv_heads, self.head_dim)?,
        ))
    }

    /// Plan two full KV arrays and fixed Q/K/V/output segment workspace.
    /// This is logical device payload, not a physical-memory receipt.
    ///
    /// # Errors
    /// Rejects malformed limits, overflow and a payload limit below the plan.
    pub fn payload_bytes(self, limits: AttentionLimits) -> Result<usize, BackendError> {
        let (query, kv) = self.widths()?;
        if limits.max_context == 0
            || limits.max_segment_tokens == 0
            || limits.max_segment_tokens > limits.max_context
            || limits.max_device_payload_bytes == 0
        {
            return Err(invalid("invalid resident attention limits"));
        }
        let cache = product(limits.max_context, kv)?;
        let workspace = product(
            limits.max_segment_tokens,
            query
                .checked_add(kv)
                .ok_or_else(|| invalid("attention width overflow"))?,
        )?;
        let bytes = product(
            cache
                .checked_add(workspace)
                .ok_or_else(|| invalid("attention payload overflow"))?,
            2 * core::mem::size_of::<f32>(),
        )?;
        if bytes > limits.max_device_payload_bytes {
            return Err(BackendError::TensorPayloadBudgetExceeded {
                requested: bytes as u64,
                limit: limits.max_device_payload_bytes as u64,
            });
        }
        Ok(bytes)
    }
}

/// A new contiguous segment, after Q/K normalization and RoPE.
#[derive(Debug)]
pub struct AttentionSegment<'a> {
    /// Token-major queries, `[tokens, query_heads, head_dim]`.
    pub query: &'a [f32],
    /// Token-major new keys, `[tokens, kv_heads, head_dim]`.
    pub key: &'a [f32],
    /// New values, with the same layout as keys.
    pub value: &'a [f32],
    /// Number of new tokens; positive and bounded by the admitted segment limit.
    pub tokens: usize,
    /// Finite scaled-dot-product multiplier.
    pub scale: f32,
    /// Output with the query layout; unchanged on failure.
    pub output: &'a mut [f32],
}

/// Independently owned KV storage and causal attention computation.
///
/// Stage replaces an abandoned transaction. Failed staging invalidates pending
/// state but preserves committed KV/cursor and caller output. Commit publishes
/// only the successful pending extent. Callers abort on cancellation and commit
/// only after every graph layer succeeds. Device-f32 arithmetic need not equal
/// host softmax bit-for-bit. This interface alone is not whole-model residency.
pub trait ResidentAttention: core::fmt::Debug + Send {
    /// Validated geometry.
    fn geometry(&self) -> AttentionGeometry;
    /// Selected immutable limits.
    fn limits(&self) -> AttentionLimits;
    /// Owned device payload, excluding allocator/driver overhead.
    fn payload_bytes(&self) -> usize;
    /// Committed visible token count.
    fn len(&self) -> usize;
    /// Whether no tokens are committed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Successful pending token count, if any.
    fn staged_len(&self) -> Option<usize>;
    /// Compute a contiguous causal segment without publishing its KV tail.
    ///
    /// # Errors
    /// Shape, capacity, non-finite data, allocation and device failures.
    fn stage(&mut self, segment: AttentionSegment<'_>) -> Result<(), BackendError>;
    /// Publish a successful pending extent; no allocation or fallible device work.
    /// With no pending extent, this is a no-op.
    fn commit(&mut self);
    /// Discard pending visibility without device work.
    fn abort(&mut self);
    /// Discard committed and pending visibility while retaining capacity.
    fn reset(&mut self);
}

fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn product(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("attention geometry overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_payload_plan_and_admission() {
        let geometry = AttentionGeometry {
            query_heads: 6,
            kv_heads: 2,
            head_dim: 7,
        };
        let limits = AttentionLimits {
            max_context: 13,
            max_segment_tokens: 5,
            max_device_payload_bytes: 3696,
        };
        assert_eq!(geometry.widths().unwrap(), (42, 14));
        assert_eq!(geometry.payload_bytes(limits).unwrap(), 3696);
        assert!(matches!(
            geometry.payload_bytes(AttentionLimits {
                max_device_payload_bytes: 3695,
                ..limits
            }),
            Err(BackendError::TensorPayloadBudgetExceeded {
                requested: 3696,
                limit: 3695
            })
        ));
    }

    #[test]
    fn rejects_malformed_and_overflowing_geometry_and_limits() {
        let valid = AttentionGeometry {
            query_heads: 6,
            kv_heads: 2,
            head_dim: 7,
        };
        for geometry in [
            AttentionGeometry {
                query_heads: 0,
                ..valid
            },
            AttentionGeometry {
                kv_heads: 0,
                ..valid
            },
            AttentionGeometry {
                head_dim: 0,
                ..valid
            },
            AttentionGeometry {
                kv_heads: 4,
                ..valid
            },
            AttentionGeometry {
                head_dim: usize::MAX,
                ..valid
            },
        ] {
            assert!(geometry.widths().is_err());
        }
        let limits = AttentionLimits {
            max_context: 13,
            max_segment_tokens: 5,
            max_device_payload_bytes: usize::MAX,
        };
        for invalid in [
            AttentionLimits {
                max_context: 0,
                ..limits
            },
            AttentionLimits {
                max_segment_tokens: 0,
                ..limits
            },
            AttentionLimits {
                max_segment_tokens: 14,
                ..limits
            },
            AttentionLimits {
                max_device_payload_bytes: 0,
                ..limits
            },
            AttentionLimits {
                max_context: usize::MAX,
                ..limits
            },
        ] {
            assert!(valid.payload_bytes(invalid).is_err());
        }
    }
}
