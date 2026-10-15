//! Explicit device-f32 full DeltaNet mixer computation after input projections.
use crate::{BackendError, DecoderLimits, DeltaNetGeometry};

/// Existing recurrence plus causal convolution and gated-normalization settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaMixerGeometry {
    /// Value heads, key/value head widths and query/key sharing.
    pub recurrence: DeltaNetGeometry,
    /// Number of raw QKV history taps retained per channel.
    pub conv_kernel: usize,
    /// Gated per-value-head RMSNorm epsilon.
    pub rms_eps: f32,
}
/// Preserved immutable mixer coefficients, not projection matrices.
#[derive(Clone, Copy, Debug)]
pub struct DeltaMixerWeights<'a> {
    /// Channel-major causal convolution coefficients `[conv_width, kernel]`.
    pub convolution: &'a [f32],
    /// Shared per-value-lane normalization coefficients `[value_head_dim]`.
    pub norm: &'a [f32],
    /// Per-value-head time-step bias.
    pub dt_bias: &'a [f32],
    /// Per-value-head log decay coefficient.
    pub a_log: &'a [f32],
}
/// Structural coefficient counts; not finite-value validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaMixerWeightLengths {
    /// Channel-major convolution coefficient count.
    pub convolution: usize,
    /// Shared value-lane normalization coefficient count.
    pub norm: usize,
    /// Per-value-head time-step bias count.
    pub dt_bias: usize,
    /// Per-value-head log decay count.
    pub a_log: usize,
}
impl DeltaMixerWeights<'_> {
    /// Borrowed structural lengths without reading coefficient values.
    pub const fn lengths(self) -> DeltaMixerWeightLengths {
        DeltaMixerWeightLengths {
            convolution: self.convolution.len(),
            norm: self.norm.len(),
            dt_bias: self.dt_bias.len(),
            a_log: self.a_log.len(),
        }
    }
}
impl DeltaMixerGeometry {
    /// Validate layout and return raw-QKV/value row widths.
    /// # Errors
    /// Malformed dimensions, epsilon or checked arithmetic overflow.
    pub fn widths(self) -> Result<(usize, usize), BackendError> {
        self.recurrence.elements()?;
        if self.conv_kernel == 0 || !self.rms_eps.is_finite() || self.rms_eps <= 0. {
            return Err(invalid("invalid DeltaNet mixer convolution or epsilon"));
        }
        let g = self.recurrence;
        let key = mul(g.value_heads / g.group_size, g.key_head_dim)?;
        let value = mul(g.value_heads, g.value_head_dim)?;
        Ok((add(mul(2, key)?, value)?, value))
    }
    /// Logical device payload for preserved vectors, state pairs and workspace.
    /// # Errors
    /// Invalid coefficient lengths/values, limits, overflow or payload rejection.
    pub fn payload_bytes(
        self,
        weights: DeltaMixerWeights<'_>,
        limits: DecoderLimits,
        budget: usize,
    ) -> Result<usize, BackendError> {
        let bytes = self.structural_payload_bytes(weights.lengths(), limits, usize::MAX)?;
        if budget == 0 {
            return Err(invalid("invalid DeltaNet mixer limits"));
        }
        if [
            weights.convolution,
            weights.norm,
            weights.dt_bias,
            weights.a_log,
        ]
        .into_iter()
        .any(|values| values.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("DeltaNet mixer coefficients must be finite"));
        }
        if bytes > budget {
            return Err(BackendError::TensorPayloadBudgetExceeded {
                requested: bytes as u64,
                limit: budget as u64,
            });
        }
        Ok(bytes)
    }

    /// Logical payload from coefficient lengths without allocating or reading values.
    /// The value-aware planner uses this exact arithmetic and additionally checks finiteness.
    /// # Errors
    /// Invalid lengths/geometry/limits, checked overflow or a payload ceiling rejection.
    pub fn structural_payload_bytes(
        self,
        weights: DeltaMixerWeightLengths,
        limits: DecoderLimits,
        budget: usize,
    ) -> Result<usize, BackendError> {
        let (conv, value) = self.widths()?;
        let state = self.recurrence.elements()?;
        let conv_state = mul(conv, self.conv_kernel)?;
        let heads = self.recurrence.value_heads;
        if limits.max_context == 0
            || limits.max_segment_tokens == 0
            || limits.max_segment_tokens > limits.max_context
            || budget == 0
        {
            return Err(invalid("invalid DeltaNet mixer limits"));
        }
        for (length, expected) in [
            (weights.convolution, conv_state),
            (weights.norm, self.recurrence.value_head_dim),
            (weights.dt_bias, heads),
            (weights.a_log, heads),
        ] {
            if length != expected {
                return Err(BackendError::ShapeMismatch {
                    expected,
                    got: length,
                });
            }
        }
        let constants = add(
            add(conv_state, self.recurrence.value_head_dim)?,
            mul(2, heads)?,
        )?;
        let workspace = mul(
            mul(2, limits.max_segment_tokens)?,
            add(add(conv, value)?, heads)?,
        )?;
        let elements = add(add(mul(2, add(conv_state, state)?)?, constants)?, workspace)?;
        let bytes = mul(elements, 4)?;
        if bytes > budget {
            return Err(BackendError::TensorPayloadBudgetExceeded {
                requested: bytes as u64,
                limit: budget as u64,
            });
        }
        Ok(bytes)
    }
}
/// Already projected contiguous rows; no normalization or gating is precomputed.
#[derive(Debug)]
pub struct DeltaMixerSegment<'a> {
    /// Raw QKV projection, `[tokens, conv_width]`.
    pub qkv: &'a [f32],
    /// Output-gate projection, `[tokens, value_width]`.
    pub z: &'a [f32],
    /// Update-gate projection, `[tokens, value_heads]`.
    pub beta_logits: &'a [f32],
    /// Decay projection, `[tokens, value_heads]`.
    pub decay_logits: &'a [f32],
    /// Positive new token count.
    pub tokens: usize,
    /// Gated normalized recurrence output `[tokens, value_width]`, unchanged on error.
    pub output: &'a mut [f32],
}
/// Owned convolution/recurrence state and device-f32 mixer computation.
pub trait ResidentDeltaMixer: core::fmt::Debug + Send {
    /// Validated immutable geometry.
    fn geometry(&self) -> DeltaMixerGeometry;
    /// Selected immutable bounds.
    fn limits(&self) -> DecoderLimits;
    /// Logical owned device payload, excluding allocator/driver overhead.
    fn payload_bytes(&self) -> usize;
    /// Committed token cursor.
    fn len(&self) -> usize;
    /// Whether no tokens are committed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Successful pending extent, if any.
    fn staged_len(&self) -> Option<usize>;
    /// Compute staging; errors preserve committed state/cursor and caller output.
    /// # Errors
    /// Input/extent/finite, allocation, numerical and device errors.
    fn stage(&mut self, segment: DeltaMixerSegment<'_>) -> Result<(), BackendError>;
    /// Publish staging by swapping state pairs, without fallible device work.
    fn commit(&mut self);
    /// Discard pending visibility.
    fn abort(&mut self);
    /// Reset visibility; next stage starts from zeros, retaining capacity.
    fn reset(&mut self);
    /// Explicit diagnostic readback of committed convolution/recurrent state.
    /// Never an ordinary stage duty. After reset, both outputs are zeros.
    /// # Errors
    /// Shape and device/transfer failures; output may be partially filled on driver error.
    fn read_committed(
        &self,
        convolution: &mut [f32],
        recurrent: &mut [f32],
    ) -> Result<(), BackendError>;
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("DeltaNet mixer extent overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("DeltaNet mixer payload overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn value_validation_precedes_payload_ceiling_rejection() {
        let geometry = DeltaMixerGeometry {
            recurrence: DeltaNetGeometry {
                value_heads: 1,
                key_head_dim: 1,
                value_head_dim: 1,
                group_size: 1,
            },
            conv_kernel: 1,
            rms_eps: 1e-6,
        };
        let weights = DeltaMixerWeights {
            convolution: &[0.; 3],
            norm: &[f32::NAN],
            dt_bias: &[0.],
            a_log: &[0.],
        };
        let limits = DecoderLimits {
            max_context: 1,
            max_segment_tokens: 1,
        };
        assert!(matches!(
            geometry.payload_bytes(weights, limits, 1),
            Err(BackendError::InvalidInput(_))
        ));
    }
    #[test]
    fn full_mixer_plan_checks_vectors_limits_and_overflow() {
        let geometry = DeltaMixerGeometry {
            recurrence: DeltaNetGeometry {
                value_heads: 6,
                key_head_dim: 8,
                value_head_dim: 8,
                group_size: 3,
            },
            conv_kernel: 4,
            rms_eps: 1e-6,
        };
        let conv = vec![0.; 320];
        let norm = vec![1.; 8];
        let scalar = vec![0.; 6];
        let weights = DeltaMixerWeights {
            convolution: &conv,
            norm: &norm,
            dt_bias: &scalar,
            a_log: &scalar,
        };
        let limits = DecoderLimits {
            max_context: 13,
            max_segment_tokens: 5,
        };
        assert_eq!(geometry.widths().unwrap(), (80, 48));
        assert_eq!(
            geometry
                .structural_payload_bytes(weights.lengths(), limits, usize::MAX)
                .unwrap(),
            12352
        );
        assert!(
            geometry
                .structural_payload_bytes(
                    DeltaMixerWeightLengths {
                        norm: 7,
                        ..weights.lengths()
                    },
                    limits,
                    usize::MAX
                )
                .is_err()
        );
        // state pairs=1408, constants=340, workspace=1340 f32.
        assert_eq!(
            geometry.payload_bytes(weights, limits, usize::MAX).unwrap(),
            12352
        );
        assert!(matches!(
            geometry.payload_bytes(weights, limits, 12351),
            Err(BackendError::TensorPayloadBudgetExceeded { .. })
        ));
        assert!(
            geometry
                .payload_bytes(
                    DeltaMixerWeights {
                        norm: &[],
                        ..weights
                    },
                    limits,
                    usize::MAX
                )
                .is_err()
        );
        assert!(
            geometry
                .payload_bytes(
                    DeltaMixerWeights {
                        a_log: &[f32::NAN; 6],
                        ..weights
                    },
                    limits,
                    usize::MAX
                )
                .is_err()
        );
        assert!(
            geometry
                .payload_bytes(
                    weights,
                    DecoderLimits {
                        max_context: 0,
                        ..limits
                    },
                    usize::MAX
                )
                .is_err()
        );
        assert!(
            DeltaMixerGeometry {
                conv_kernel: usize::MAX,
                ..geometry
            }
            .widths()
            .is_ok()
        );
        assert!(
            DeltaMixerGeometry {
                conv_kernel: usize::MAX,
                ..geometry
            }
            .payload_bytes(weights, limits, usize::MAX)
            .is_err()
        );
    }
}
