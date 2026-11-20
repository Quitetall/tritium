//! OpenAI-serving adapter for strict schema-v3 Qwen language bundles.

use std::fmt;

use crate::generator::{
    CancellationCapabilitiesV1, CancellationCheckpoint, GenError, GenRequest, Generator, Step,
    stream_steps,
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
        let Some(output) = runner
            .forward_cancellable(&request.prompt_tokens, &mut cache, is_cancelled)
            .map_err(|error| GenError::Backend(error.to_string()))?
        else {
            return Ok(());
        };
        stream_steps(
            output.into_last_logits(),
            request,
            max_new,
            self.eos,
            on_step,
            is_cancelled,
            |token, _position| {
                runner
                    .forward_cancellable(&[token], &mut cache, is_cancelled)
                    .map(|output| output.map(tritium_nn::Qwen35TextOutput::into_last_logits))
                    .map_err(|error| GenError::Backend(error.to_string()))
            },
        )?;
        Ok(())
    }

    fn n_ctx(&self) -> usize {
        self.model.config().text.max_position_embeddings as usize
    }

    fn vocab(&self) -> usize {
        self.model.runner().vocab_size()
    }
}
