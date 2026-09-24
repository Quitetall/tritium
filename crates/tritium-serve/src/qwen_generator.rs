//! OpenAI-serving adapter for strict schema-v3 Qwen language bundles.

use std::fmt;

use crate::generator::{
    FinishReason, GenError, GenRequest, Generator, Sampling, Step, top_logprobs,
};

/// Which numerics tier a Qwen bundle is served at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum QwenNumerics {
    /// The host-orchestrated forward, bit-identical to the reference. Default.
    #[default]
    Exact,
    /// The device-resident CUDA executor where the backend and bundle allow it:
    /// reassociated GEMV sums and CUDA transcendentals, gated on relative error
    /// and greedy agreement against the exact forward (`qwen36_resident_parity`),
    /// not on equality. Falls back to [`Self::Exact`] where it cannot run.
    Fast,
}

/// Generator owning one content-bound Qwen language-plus-MTP bundle.
pub struct QwenGenerator {
    model: tritium_nn::Qwen35SaltV2LanguageMtpModel,
    eos: u32,
    /// The fast tier's executor, when built.
    #[cfg(feature = "cuda")]
    resident: Option<Resident>,
}

impl QwenGenerator {
    /// Bind serving decode to one already-admitted strict bundle.
    #[must_use]
    pub const fn new(model: tritium_nn::Qwen35SaltV2LanguageMtpModel, eos: u32) -> Self {
        Self {
            model,
            eos,
            #[cfg(feature = "cuda")]
            resident: None,
        }
    }

    /// Bind a bundle at the requested numerics tier. [`QwenNumerics::Fast`]
    /// builds the device-resident executor when the backend is CUDA and every
    /// weight is one it serves, sized for `max_context` (else the model's
    /// context) up to the executor's cap; otherwise the generator serves exact.
    ///
    /// # Errors
    /// Returns a message if the backend accepts the model but the executor build
    /// fails (for example a device allocation).
    pub fn with_numerics(
        model: tritium_nn::Qwen35SaltV2LanguageMtpModel,
        eos: u32,
        numerics: QwenNumerics,
        max_context: Option<usize>,
    ) -> Result<Self, String> {
        #[allow(unused_mut)]
        let mut generator = Self::new(model, eos);
        #[cfg(not(feature = "cuda"))]
        let _ = max_context;
        if numerics == QwenNumerics::Fast {
            #[cfg(feature = "cuda")]
            {
                let context = (generator.model.config().text.max_position_embeddings as usize)
                    .min(max_context.unwrap_or(usize::MAX))
                    .min(tritium_cuda::QWEN35_RESIDENT_MAX_CONTEXT);
                generator.resident = generator
                    .model
                    .runner()
                    .cuda_resident(context)
                    .map_err(|error| format!("resident executor: {error}"))?
                    .map(|executor| Resident::new(executor, context));
            }
            if generator.is_resident() {
                eprintln!("tritium-serve: numerics fast: device-resident executor");
            } else {
                eprintln!(
                    "tritium-serve: numerics fast requested but unavailable here \
                     (needs a CUDA build and backend); serving exact"
                );
            }
        }
        Ok(generator)
    }

    /// Whether requests run on the fast tier's resident executor.
    #[must_use]
    pub fn is_resident(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.resident.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
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

/// One emitted token as a streamed [`Step`].
fn step_for(
    token: u32,
    index: usize,
    max_new: usize,
    eos: u32,
    request: &GenRequest,
    logits: Option<&[f32]>,
) -> (Step, bool) {
    let eos = request.stop_eos && token == eos;
    let last = eos || index + 1 == max_new;
    (
        Step {
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
                .zip(logits)
                .map(|(count, logits)| top_logprobs(logits, token, count)),
        },
        last,
    )
}

/// Prompt tokens two requests must share before a shared-prefix snapshot is
/// taken: below this, re-prefilling is cheaper than tracking it.
#[cfg(feature = "cuda")]
const MIN_SHARED_PREFIX: usize = 16;

/// One saved executor state and the tokens it stands for.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct Checkpoint {
    tokens: Vec<u32>,
    snapshot: Option<tritium_cuda::Qwen35ResidentSnapshot>,
    valid: bool,
}

/// The fast tier's executor plus what prompt reuse needs to know about it.
///
/// A chat client resends the whole conversation every turn, and LAMU prepends
/// the same system prompt to every request, so most of each prompt was decoded
/// before. The executor's attention KV already holds those tokens; what a new
/// request needs is the DeltaNet recurrent state as of the end of the shared
/// prefix. Two checkpoints cover the common cases: the end of the last prompt
/// (a follow-up turn extends it) and the longest prefix the last two prompts
/// shared (a system prompt across conversations). A checkpoint is used only
/// while the KV cache still holds its tokens below its position, which
/// `kv_tokens` tracks exactly. Resuming is bit-identical to a fresh prefill
/// (`resident_snapshot_resume_is_bit_identical_to_a_fresh_prefill`).
/// `TRITIUM_PREFIX_REUSE=0` disables it.
#[cfg(feature = "cuda")]
struct Resident {
    executor: tritium_cuda::Qwen35Resident,
    context: usize,
    /// The tokens the KV cache holds, in position order.
    kv_tokens: Vec<u32>,
    last_prompt: Vec<u32>,
    prompt_end: Checkpoint,
    shared: Checkpoint,
    reuse: bool,
}

#[cfg(feature = "cuda")]
impl Resident {
    fn new(executor: tritium_cuda::Qwen35Resident, context: usize) -> Self {
        Self {
            executor,
            context,
            kv_tokens: Vec::new(),
            last_prompt: Vec::new(),
            prompt_end: Checkpoint::default(),
            shared: Checkpoint::default(),
            reuse: std::env::var("TRITIUM_PREFIX_REUSE").as_deref() != Ok("0"),
        }
    }

    fn invalidate(&mut self) {
        self.kv_tokens.clear();
        self.last_prompt.clear();
        self.prompt_end.valid = false;
        self.shared.valid = false;
    }

    /// Save the executor's state as `which` for `tokens`.
    fn checkpoint(&mut self, which: bool, tokens: &[u32]) -> Result<(), GenError> {
        let slot = if which {
            &mut self.shared
        } else {
            &mut self.prompt_end
        };
        self.executor
            .save_snapshot(&mut slot.snapshot)
            .map_err(|error| GenError::Backend(error.to_string()))?;
        slot.tokens.clear();
        slot.tokens.extend_from_slice(tokens);
        slot.valid = true;
        Ok(())
    }

    /// Feed `tokens` (non-empty), recording them as the KV cache's contents.
    fn feed(&mut self, tokens: &[u32]) -> Result<u32, GenError> {
        let next = self
            .executor
            .prefill(tokens)
            .map_err(|error| GenError::Backend(error.to_string()))?;
        self.kv_tokens.extend_from_slice(tokens);
        Ok(next)
    }

    /// Position the executor to decode `prompt`: resume from the longest usable
    /// checkpoint or start over, then decode every prompt token but the last.
    /// Returns how many prompt tokens remain (always at least one).
    fn prepare(&mut self, prompt: &[u32]) -> Result<usize, GenError> {
        let backend = |error: tritium_spec::BackendError| GenError::Backend(error.to_string());
        let length = prompt.len();
        let usable = |slot: &Checkpoint, kv: &[u32]| {
            slot.valid
                && slot.tokens.len() < length
                && prompt.starts_with(&slot.tokens)
                && kv.starts_with(&slot.tokens)
        };
        let mut start = 0;
        if self.reuse {
            let candidates = [&self.prompt_end, &self.shared];
            let best = candidates
                .into_iter()
                .filter(|slot| usable(slot, &self.kv_tokens))
                .max_by_key(|slot| slot.tokens.len());
            if let Some(slot) = best {
                let snapshot = slot
                    .snapshot
                    .as_ref()
                    .expect("a valid checkpoint holds a snapshot");
                start = slot.tokens.len();
                self.executor.restore_snapshot(snapshot).map_err(backend)?;
            }
        }
        if start == 0 {
            self.executor.reset().map_err(backend)?;
        }
        self.kv_tokens.truncate(start);
        // A prefix this prompt shares with the last one, past what was reused,
        // is likely shared by the next one too: checkpoint it on the way.
        if self.reuse {
            let shared = self
                .last_prompt
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count()
                .min(length - 1);
            if shared >= start + MIN_SHARED_PREFIX {
                self.feed(&prompt[start..shared])?;
                self.checkpoint(true, &prompt[..shared])?;
                start = shared;
            }
        }
        if start + 1 < length {
            self.feed(&prompt[start..length - 1])?;
        }
        self.last_prompt.clear();
        self.last_prompt.extend_from_slice(prompt);
        Ok(1)
    }
}

#[cfg(feature = "cuda")]
impl QwenGenerator {
    /// Decode on the resident executor. Greedy requests without logprobs read
    /// back only the next token id; the rest read back the logits row.
    fn generate_resident(
        resident: &mut Resident,
        eos: u32,
        request: &GenRequest,
        max_new: usize,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        let result = Self::generate_resident_inner(resident, eos, request, max_new, on_step);
        if result.is_err() {
            // The executor may have stopped mid-sequence: trust nothing cached.
            resident.invalidate();
        }
        result
    }

    fn generate_resident_inner(
        resident: &mut Resident,
        eos: u32,
        request: &GenRequest,
        max_new: usize,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        let backend = |error: tritium_spec::BackendError| GenError::Backend(error.to_string());
        let prompt = &request.prompt_tokens;
        let (&final_prompt, _) = prompt.split_last().ok_or(GenError::ContextOverflow)?;
        resident.prepare(prompt)?;
        let fast_greedy =
            matches!(request.sampling, Sampling::Greedy) && request.logprobs.is_none();
        if fast_greedy {
            let mut token = resident.feed(&[final_prompt])?;
            resident.checkpoint(false, prompt)?;
            for index in 0..max_new {
                let (step, last) = step_for(token, index, max_new, eos, request, None);
                if !on_step(step) || last {
                    break;
                }
                let next = resident.executor.step(token).map_err(backend)?;
                resident.kv_tokens.push(token);
                token = next;
            }
            return Ok(());
        }
        let mut logits = resident
            .executor
            .step_logits(final_prompt)
            .map_err(backend)?;
        resident.kv_tokens.push(final_prompt);
        resident.checkpoint(false, prompt)?;
        for index in 0..max_new {
            let token = Self::sample(&logits, &request.sampling, index as u64)
                .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
            let (step, last) = step_for(token, index, max_new, eos, request, Some(&logits));
            if !on_step(step) || last {
                break;
            }
            logits = resident.executor.step_logits(token).map_err(backend)?;
            resident.kv_tokens.push(token);
        }
        Ok(())
    }
}

impl Generator for QwenGenerator {
    fn generate(
        &mut self,
        request: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        #[cfg(feature = "cuda")]
        if let Some(resident) = self.resident.as_mut() {
            let resident_context = resident.context;
            let prompt_len = request.prompt_tokens.len();
            let max_new = request
                .max_new
                .min(resident_context.saturating_sub(prompt_len));
            // A request that fits the executor's context runs there; a longer one
            // falls through to the exact host path below.
            if prompt_len != 0 && prompt_len + request.max_new <= resident_context {
                return Self::generate_resident(resident, self.eos, request, max_new, on_step);
            }
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
        let mut output = runner
            .forward(&request.prompt_tokens, &mut cache)
            .map_err(|error| GenError::Backend(error.to_string()))?;
        for index in 0..max_new {
            let logits = output.last_logits();
            let token = Self::sample(logits, &request.sampling, index as u64)
                .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
            let (step, last) = step_for(token, index, max_new, self.eos, request, Some(logits));
            let keep_going = on_step(step);
            if last || !keep_going {
                break;
            }
            output = runner
                .forward(&[token], &mut cache)
                .map_err(|error| GenError::Backend(error.to_string()))?;
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
