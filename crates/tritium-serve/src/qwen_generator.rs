//! OpenAI-serving adapter for strict schema-v3 Qwen language bundles.

use std::fmt;

use crate::generator::{
    CancellationCapabilitiesV1, CancellationCheckpoint, FinishReason, GenError, GenRequest,
    Generator, Sampling, Step, top_logprobs,
};

/// Generator owning one content-bound Qwen language-plus-MTP bundle.
pub struct QwenGenerator {
    model: tritium_nn::Qwen35SaltV2LanguageMtpModel,
    eos: u32,
}

impl QwenGenerator {
    /// Bind serving decode to one already-admitted strict bundle.
    #[must_use]
    pub const fn new(model: tritium_nn::Qwen35SaltV2LanguageMtpModel, eos: u32) -> Self {
        Self { model, eos }
    }

    fn sample(logits: &[f32], sampling: &Sampling, step: u64) -> Option<u32> {
        match *sampling {
            Sampling::Greedy => tritium_nn::sample_greedy(logits),
            Sampling::TopK { k, temp, seed } => {
                tritium_nn::sample_top_k(logits, k, temp, seed.wrapping_add(step))
            }
            Sampling::TopP { p, temp, seed } => {
                tritium_nn::sample_top_p(logits, p, temp, seed.wrapping_add(step))
            }
        }
    }
}

impl fmt::Debug for QwenGenerator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenGenerator")
            .field("profile", &self.model.receipt().profile())
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl Generator for QwenGenerator {
    fn cancellation_capabilities(&self) -> CancellationCapabilitiesV1 {
        CancellationCapabilitiesV1 {
            generation: CancellationCheckpoint::CooperativeBoundaries,
            tree_session_open: CancellationCheckpoint::NotEnabled,
            tree_verify: CancellationCheckpoint::NotEnabled,
            model_draft: CancellationCheckpoint::NotEnabled,
            ..CancellationCapabilitiesV1::default()
        }
    }

    fn generate(
        &mut self,
        request: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        self.generate_cancellable(request, on_step, &|| false)
    }

    fn generate_cancellable(
        &mut self,
        request: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<(), GenError> {
        if is_cancelled() {
            return Ok(());
        }
        let runner = self.model.runner();
        let context = self.model.config().text.max_position_embeddings as usize;
        let prompt_len = request.prompt_tokens.len();
        if prompt_len == 0 || prompt_len > context {
            return Err(GenError::ContextOverflow);
        }
        let max_new = request.max_new.min(context.saturating_sub(prompt_len));
        let capacity = prompt_len
            .checked_add(max_new)
            .ok_or(GenError::ContextOverflow)?;
        let mut cache = runner
            .new_cache(capacity)
            .map_err(|error| GenError::Backend(error.to_string()))?;
        let Some(mut output) = runner
            .forward_cancellable(&request.prompt_tokens, &mut cache, is_cancelled)
            .map_err(|error| GenError::Backend(error.to_string()))?
        else {
            return Ok(());
        };
        for index in 0..max_new {
            if is_cancelled() {
                return Ok(());
            }
            let logits = output.last_logits();
            let token = Self::sample(logits, &request.sampling, index as u64)
                .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
            let eos = request.stop_eos && token == self.eos;
            let last = eos || index + 1 == max_new;
            let keep_going = on_step(Step {
                token,
                finished: last,
                finish_reason: if eos {
                    Some(FinishReason::Stop)
                } else if last {
                    Some(FinishReason::Length)
                } else {
                    None
                },
                logprobs: request
                    .logprobs
                    .map(|count| top_logprobs(logits, token, count)),
            });
            if last || !keep_going {
                break;
            }
            let next_output = runner
                .forward_cancellable(&[token], &mut cache, is_cancelled)
                .map_err(|error| GenError::Backend(error.to_string()))?;
            let Some(next_output) = next_output else {
                return Ok(());
            };
            output = next_output;
        }
        Ok(())
    }

    fn n_ctx(&self) -> usize {
        self.model.config().text.max_position_embeddings as usize
    }

    fn vocab(&self) -> usize {
        self.model.runner().vocab_size()
    }
}
