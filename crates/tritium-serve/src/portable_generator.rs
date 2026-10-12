//! One public serving adapter; legacy resident internals remain transitional.
use crate::generator::{
    CancellationCapabilitiesV1, CancellationCheckpoint, GenError, GenRequest, Generator,
    LegacyRunnerGenerator, Step, StreamEnd, TreeOpError, stream_steps,
};
use std::fmt;

enum Execution {
    Legacy(Box<LegacyRunnerGenerator>),
    Portable {
        runner: Box<tritium_nn::Runner>,
        eos: u32,
        unsupported_draft: bool,
    },
}

/// Serving adapter for a common loaded runner or transitional legacy resident graph.
/// Uses the same plain sampling/stream policy, without concrete backend assumptions.
pub struct RunnerGenerator {
    execution: Execution,
}
impl fmt::Debug for RunnerGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunnerGenerator")
            .field("context", &self.n_ctx())
            .field("vocab", &self.vocab())
            .finish_non_exhaustive()
    }
}
impl RunnerGenerator {
    /// Preserve legacy GGUF/SALT native speculative/tree execution during migration.
    #[must_use]
    pub fn new(runner: tritium_nn::ModelRunner, eos: u32) -> Self {
        Self {
            execution: Execution::Legacy(Box::new(LegacyRunnerGenerator::new(runner, eos))),
        }
    }
    /// Serve the actual common loaded standard or hybrid language graph.
    /// No draft/tree/batch or independent production authority is inferred.
    #[must_use]
    pub fn from_runner(runner: tritium_nn::Runner, eos: u32) -> Self {
        Self {
            execution: Execution::Portable {
                runner: Box::new(runner),
                eos,
                unsupported_draft: false,
            },
        }
    }
    /// Legacy model drafter. Common runners reject configured drafts before inference
    /// until their backend-owned draft/tree contract and parity authority exist.
    #[must_use]
    pub fn with_draft_model(mut self, draft: tritium_nn::ModelRunner) -> Self {
        match self.execution {
            Execution::Legacy(legacy) => {
                self.execution = Execution::Legacy(Box::new(legacy.with_draft_model(draft)))
            }
            Execution::Portable {
                ref mut unsupported_draft,
                ..
            } => *unsupported_draft = true,
        }
        self
    }
    /// Preserve legacy CUDA lookup policy. Portable execution takes plain stepping
    /// (the same existing nonresident fallback); no tree availability is claimed.
    #[must_use]
    pub fn with_spec_lookup(mut self, on: bool) -> Self {
        if let Execution::Legacy(legacy) = self.execution {
            self.execution = Execution::Legacy(Box::new(legacy.with_spec_lookup(on)));
        }
        self
    }
}
impl Generator for RunnerGenerator {
    fn cancellation_capabilities(&self) -> CancellationCapabilitiesV1 {
        match &self.execution {
            Execution::Legacy(legacy) => legacy.cancellation_capabilities(),
            Execution::Portable { .. } => CancellationCapabilitiesV1 {
                generation: CancellationCheckpoint::CooperativeBoundaries,
                tree_session_open: CancellationCheckpoint::NotEnabled,
                tree_verify: CancellationCheckpoint::NotEnabled,
                model_draft: CancellationCheckpoint::NotEnabled,
                ..CancellationCapabilitiesV1::default()
            },
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
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), GenError> {
        if cancelled() {
            return Ok(());
        }
        match &mut self.execution {
            Execution::Legacy(legacy) => legacy.generate_cancellable(request, on_step, cancelled),
            Execution::Portable {
                runner,
                eos,
                unsupported_draft,
            } => {
                if *unsupported_draft {
                    return Err(GenError::Backend(
                        "common runner model draft is not enabled".into(),
                    ));
                }
                let context = runner.architecture().max_context();
                let length = request.prompt_tokens.len();
                if length == 0 || length > context {
                    return Err(GenError::ContextOverflow);
                }
                if request
                    .prompt_tokens
                    .iter()
                    .any(|&id| id as usize >= runner.architecture().vocab_size())
                {
                    return Err(GenError::Backend(
                        "prompt ID outside model vocabulary".into(),
                    ));
                }
                let max_new = request.max_new.min(context - length);
                let mut positions = Vec::new();
                positions
                    .try_reserve_exact(length)
                    .map_err(|e| GenError::Backend(format!("prefill positions: {e}")))?;
                positions.extend(0..length);
                runner.reset();
                let result = (|| {
                    let Some(logits) = runner
                        .forward_cancellable(&request.prompt_tokens, &positions, cancelled)
                        .map_err(|e| GenError::Backend(e.to_string()))?
                    else {
                        return Ok(StreamEnd::Cancelled);
                    };
                    stream_steps(
                        logits,
                        request,
                        max_new,
                        *eos,
                        on_step,
                        cancelled,
                        |token, position| {
                            runner
                                .forward_cancellable(&[token], &[position], cancelled)
                                .map_err(|e| GenError::Backend(e.to_string()))
                        },
                    )
                })();
                if !matches!(result, Ok(StreamEnd::Completed)) {
                    runner.reset();
                }
                result.map(|_| ())
            }
        }
    }
    fn n_ctx(&self) -> usize {
        match &self.execution {
            Execution::Legacy(legacy) => legacy.n_ctx(),
            Execution::Portable { runner, .. } => runner.architecture().max_context(),
        }
    }
    fn vocab(&self) -> usize {
        match &self.execution {
            Execution::Legacy(legacy) => legacy.vocab(),
            Execution::Portable { runner, .. } => runner.architecture().vocab_size(),
        }
    }
    fn open_tree_session(&mut self, prompt: &[u32]) -> Result<u32, TreeOpError> {
        self.open_tree_session_cancellable(prompt, &|| false)?
            .ok_or_else(|| {
                TreeOpError::Internal("uncancelled tree session returned no output".into())
            })
    }
    fn open_tree_session_cancellable(
        &mut self,
        prompt: &[u32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<u32>, TreeOpError> {
        if cancelled() {
            return Ok(None);
        }
        match &mut self.execution {
            Execution::Legacy(legacy) => legacy.open_tree_session_cancellable(prompt, cancelled),
            Execution::Portable { .. } => Err(TreeOpError::Unsupported(
                "common runner tree sessions are not enabled".into(),
            )),
        }
    }
    fn tree_verify(&mut self, tokens: &[u32], parents: &[i32]) -> Result<Vec<u32>, TreeOpError> {
        self.tree_verify_cancellable(tokens, parents, &|| false)?
            .ok_or_else(|| {
                TreeOpError::Internal("uncancelled tree verification returned no output".into())
            })
    }
    fn tree_verify_cancellable(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<u32>>, TreeOpError> {
        if cancelled() {
            return Ok(None);
        }
        match &mut self.execution {
            Execution::Legacy(legacy) => legacy.tree_verify_cancellable(tokens, parents, cancelled),
            Execution::Portable { .. } => Err(TreeOpError::Unsupported(
                "common runner tree verification is not enabled".into(),
            )),
        }
    }
}
