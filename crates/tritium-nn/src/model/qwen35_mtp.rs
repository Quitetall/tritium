//! Structural Qwen3.5-family multi-token-prediction assembly.
//!
//! The bundled one-layer drafter is retained as an opaque, validated weight
//! graph, but this module deliberately exposes no backend, cache, or forward
//! API. It cannot become runnable until an official-runtime parity receipt is
//! verified by a future promotion boundary.

use std::sync::Arc;

use super::qwen35::{Qwen35TextOutput, Qwen35TextRunner, RunnerIdentity};
use crate::error::NnError;
use crate::layers::{Projection, ProjectionActivationMode, Qwen35FullAttentionWeights, SwiGluMlp};

/// Stable reason the structurally complete MTP graph is not runnable.
pub const QWEN35_MTP_UNVERIFIED_REASON: &str =
    "official Qwen3.5/Qwen3.6 MTP parity receipt is missing";

/// Verification state of the bundled Qwen3.5-family MTP graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpStatus {
    /// Weights and alignment are structural only; execution is unavailable.
    Unverified,
}

impl Qwen35MtpStatus {
    /// Stable explanation for the current non-runnable state.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Unverified => QWEN35_MTP_UNVERIFIED_REASON,
        }
    }
}

/// Shifted token IDs, target positions, and borrowed target hidden rows for one
/// future MTP step.
#[derive(Debug)]
pub struct Qwen35MtpInputPlan<'target> {
    shifted_token_ids: Vec<u32>,
    positions: Vec<usize>,
    target_hidden_states: &'target [f32],
}

impl Qwen35MtpInputPlan<'_> {
    /// Shifted IDs `[x1, ..., xN-1, sampled_next]`.
    #[must_use]
    pub fn shifted_token_ids(&self) -> &[u32] {
        &self.shifted_token_ids
    }

    /// Absolute target positions retained without the token shift.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Final-normalized target hidden rows `[sequence, hidden]`.
    ///
    /// This slice borrows the exact language output used to derive the plan;
    /// alignment cannot substitute or mutate a second hidden-state buffer.
    #[must_use]
    pub const fn target_hidden_states(&self) -> &[f32] {
        self.target_hidden_states
    }
}

/// Raw weights for the MTP drafter's one forced full-attention decoder layer.
///
/// This group owns 11 checkpoint tensors: two decoder norms, six attention
/// tensors, and three SwiGLU projections.
#[allow(dead_code, missing_debug_implementations)]
pub struct Qwen35MtpLayerWeights {
    input_norm: Vec<f32>,
    attention: Qwen35FullAttentionWeights,
    post_attention_norm: Vec<f32>,
    mlp: SwiGluMlp,
}

impl Qwen35MtpLayerWeights {
    /// Collect the unbound full-attention MTP decoder layer.
    #[must_use]
    pub fn new(
        input_norm: Vec<f32>,
        attention: Qwen35FullAttentionWeights,
        post_attention_norm: Vec<f32>,
        mlp: SwiGluMlp,
    ) -> Self {
        Self {
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        }
    }
}

/// The exact 15 checkpoint tensors retained by the bundled MTP graph.
///
/// There is intentionally no token embedding or language head field: the
/// future verified executor must reuse the target language model's distinct
/// embedding and untied head without allocating copies.
#[allow(dead_code, missing_debug_implementations)]
pub struct Qwen35MtpWeights {
    pre_fc_norm_embedding: Vec<f32>,
    pre_fc_norm_hidden: Vec<f32>,
    fc: Projection,
    layer: Qwen35MtpLayerWeights,
    final_norm: Vec<f32>,
}

impl Qwen35MtpWeights {
    /// Collect the complete unbound one-layer MTP graph.
    #[must_use]
    pub fn new(
        pre_fc_norm_embedding: Vec<f32>,
        pre_fc_norm_hidden: Vec<f32>,
        fc: Projection,
        layer: Qwen35MtpLayerWeights,
        final_norm: Vec<f32>,
    ) -> Self {
        Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            final_norm,
        }
    }
}

/// Opaque, structurally complete one-layer Qwen3.5/Qwen3.6 MTP graph.
///
/// This type has no `forward`, cache, backend, weight getter, or decomposition
/// API. Its only operational seam prepares the immutable alignment input for a
/// future receipt-verified executor.
#[allow(missing_debug_implementations)]
pub struct UnverifiedQwen35Mtp {
    runner_identity: Arc<RunnerIdentity>,
    hidden_size: usize,
    vocab_size: u32,
    _weights: Qwen35MtpWeights,
}

impl UnverifiedQwen35Mtp {
    /// Assemble and validate the exact one-layer, shared-embedding MTP graph.
    ///
    /// The nested layer owns 11 checkpoint tensors; the two pre-fusion norms,
    /// fusion projection, and final norm bring the exact total to 15. Every
    /// projection must use exact-fp32 activation arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::MissingConfig`] for a non-one-layer/dedicated-embedding
    /// contract or non-fp32 projection arithmetic, [`NnError::Shape`] for any
    /// tensor geometry mismatch, or [`NnError::Backend`] for non-finite retained
    /// parameters.
    pub fn new(target: &Qwen35TextRunner, weights: Qwen35MtpWeights) -> Result<Self, NnError> {
        let config = target.config();
        if config.model_type != "qwen3_5_text" {
            return Err(invalid_config(
                "MTP assembly requires a Qwen3.5-family text configuration",
            ));
        }
        if config.mtp.num_hidden_layers != 1 {
            return Err(invalid_config(
                "MTP assembly requires exactly one hidden layer",
            ));
        }
        if config.mtp.dedicated_embeddings {
            return Err(invalid_config(
                "MTP assembly requires the shared language embedding",
            ));
        }
        if config.tied_embeddings {
            return Err(invalid_config(
                "MTP assembly requires the checkpoint's distinct untied language head",
            ));
        }

        let hidden_size = axis(config.hidden_size, "hidden_size")?;
        let intermediate_size = axis(config.intermediate_size, "intermediate_size")?;
        let vocab_size = config.vocab_size;
        axis(vocab_size, "vocab_size")?;
        let fused_width = hidden_size
            .checked_mul(2)
            .ok_or_else(|| invalid_config("MTP fusion width overflow"))?;

        validate_vector(
            &weights.pre_fc_norm_embedding,
            hidden_size,
            "pre-fc embedding norm",
        )?;
        validate_vector(
            &weights.pre_fc_norm_hidden,
            hidden_size,
            "pre-fc hidden norm",
        )?;
        validate_projection(&weights.fc, hidden_size, fused_width, "fusion projection")?;
        validate_vector(&weights.layer.input_norm, hidden_size, "layer input norm")?;
        let attention_mode = weights.layer.attention.validate_for_config(config)?;
        validate_vector(
            &weights.layer.post_attention_norm,
            hidden_size,
            "layer post-attention norm",
        )?;
        let mlp_mode = weights
            .layer
            .mlp
            .validate_for_geometry(hidden_size, intermediate_size)?;
        validate_vector(&weights.final_norm, hidden_size, "final MTP norm")?;

        if weights.fc.activation_mode() != ProjectionActivationMode::F32
            || attention_mode != ProjectionActivationMode::F32
            || mlp_mode != ProjectionActivationMode::F32
        {
            return Err(invalid_config(
                "unverified MTP assembly requires exact-fp32 projection arithmetic",
            ));
        }

        Ok(Self {
            runner_identity: Arc::clone(target.identity()),
            hidden_size,
            vocab_size,
            _weights: weights,
        })
    }

    /// Verification state; always [`Qwen35MtpStatus::Unverified`] until a future
    /// typed parity receipt promotes a different executable type.
    #[must_use]
    pub const fn status(&self) -> Qwen35MtpStatus {
        Qwen35MtpStatus::Unverified
    }

    /// Align one bound target-language output with shifted MTP token IDs.
    ///
    /// Token IDs, positions, and hidden rows all derive exclusively from the
    /// language output. Outputs minted by another runner are rejected even when
    /// their geometry is identical.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::Shape`] for a hidden-width mismatch,
    /// [`NnError::MissingTensor`] for an out-of-vocabulary sampled ID, or
    /// [`NnError::Backend`] for a foreign output or position/allocation overflow.
    pub fn align_step<'target>(
        &self,
        target_output: &'target Qwen35TextOutput,
        sampled_next: u32,
    ) -> Result<Qwen35MtpInputPlan<'target>, NnError> {
        if !Arc::ptr_eq(&self.runner_identity, target_output.runner_identity()) {
            return Err(NnError::Backend(
                "Qwen3.5 MTP target output belongs to a different language runner".to_owned(),
            ));
        }
        if target_output.hidden_size() != self.hidden_size {
            return Err(NnError::Shape {
                expected: self.hidden_size,
                got: target_output.hidden_size(),
            });
        }
        if sampled_next >= self.vocab_size {
            return Err(NnError::MissingTensor(format!(
                "Qwen3.5 MTP token id {sampled_next} is outside vocabulary size {}",
                self.vocab_size
            )));
        }

        build_input_plan(
            target_output.input_token_ids(),
            sampled_next,
            target_output.position_start(),
            target_output.final_hidden_states(),
        )
    }
}

fn build_input_plan<'target>(
    target_token_ids: &[u32],
    sampled_next: u32,
    position_start: usize,
    target_hidden_states: &'target [f32],
) -> Result<Qwen35MtpInputPlan<'target>, NnError> {
    position_start
        .checked_add(target_token_ids.len() - 1)
        .ok_or_else(|| NnError::Backend("Qwen3.5 MTP position range overflow".to_owned()))?;

    let mut shifted_token_ids = Vec::new();
    shifted_token_ids
        .try_reserve_exact(target_token_ids.len())
        .map_err(|error| {
            NnError::Backend(format!("allocate Qwen3.5 MTP shifted-token plan: {error}"))
        })?;
    shifted_token_ids.extend_from_slice(&target_token_ids[1..]);
    shifted_token_ids.push(sampled_next);

    let mut positions = Vec::new();
    positions
        .try_reserve_exact(target_token_ids.len())
        .map_err(|error| {
            NnError::Backend(format!("allocate Qwen3.5 MTP position plan: {error}"))
        })?;
    for offset in 0..target_token_ids.len() {
        positions.push(position_start + offset);
    }

    Ok(Qwen35MtpInputPlan {
        shifted_token_ids,
        positions,
        target_hidden_states,
    })
}

fn validate_projection(
    projection: &Projection,
    expected_out: usize,
    expected_in: usize,
    name: &str,
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
    projection
        .validate_retained_parameters()
        .map_err(|error| match error {
            NnError::Backend(message) => NnError::Backend(format!("Qwen3.5 MTP {name}: {message}")),
            other => other,
        })
}

fn validate_vector(values: &[f32], expected: usize, name: &str) -> Result<(), NnError> {
    if values.len() != expected {
        return Err(NnError::Shape {
            expected,
            got: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NnError::Backend(format!(
            "Qwen3.5 MTP {name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn axis(value: u32, name: &str) -> Result<usize, NnError> {
    if value == 0 {
        return Err(invalid_config(format!(
            "Qwen3.5 MTP {name} must be non-zero"
        )));
    }
    usize::try_from(value)
        .map_err(|_| invalid_config(format!("Qwen3.5 MTP {name} does not fit usize")))
}

fn invalid_config(message: impl Into<String>) -> NnError {
    NnError::MissingConfig(message.into())
}

#[cfg(test)]
mod tests {
    use super::build_input_plan;

    #[test]
    fn position_overflow_fails_closed() {
        let hidden = [0.0];
        let error = build_input_plan(&[1, 2], 3, usize::MAX, &hidden).unwrap_err();
        assert!(matches!(error, crate::NnError::Backend(_)));
    }
}
