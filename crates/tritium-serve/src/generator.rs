//! The seam between the HTTP layer and inference.
//!
//! This module is **always compiled** and **runtime-free** (no tokio/axum), so the
//! default workspace build pulls in no async deps. The `serve`-gated worker drives
//! a [`Generator`] on a dedicated thread; contract tests drive [`MockGenerator`]
//! directly with no model.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One generation request: a tokenized prompt + decode controls.
#[derive(Debug, Clone)]
pub struct GenRequest {
    /// Prompt token IDs (already tokenized — serve does not own a BPE; see crate docs).
    pub prompt_tokens: Vec<u32>,
    /// Max new tokens to decode (clamped to the model's remaining context).
    pub max_new: usize,
    /// Sampling strategy.
    pub sampling: Sampling,
    /// Honor the model EOS token (always true except for adversarial tests).
    pub stop_eos: bool,
}

/// Sampling strategy, lowered from the OpenAI request fields.
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    /// Deterministic argmax (OpenAI `temperature == 0`).
    Greedy,
    /// Top-k with temperature (no native OpenAI field; available for completeness).
    TopK {
        /// Candidate cutoff (`0` = unrestricted).
        k: usize,
        /// Softmax temperature.
        temp: f32,
        /// Base PRNG seed (advanced per step inside the generator).
        seed: u64,
    },
    /// Nucleus (top-p) with temperature (OpenAI `top_p`).
    TopP {
        /// Cumulative-probability cutoff in `(0, 1]`.
        p: f32,
        /// Softmax temperature.
        temp: f32,
        /// Base PRNG seed (advanced per step inside the generator).
        seed: u64,
    },
}

/// One decoded step handed to the `on_step` callback.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// The decoded token ID (special tokens like EOS are dropped by the HTTP detok).
    pub token: u32,
    /// True on the terminal step (EOS hit or budget reached).
    pub finished: bool,
    /// Set on the terminal step.
    pub finish_reason: Option<FinishReason>,
}

/// Why generation stopped (maps to the OpenAI `finish_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token or a stop string was hit.
    Stop,
    /// The `max_tokens` / context budget was reached.
    Length,
}

impl FinishReason {
    /// The OpenAI wire string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// A generation failure surfaced to the HTTP layer.
#[derive(Debug, Clone)]
pub enum GenError {
    /// The execution backend failed (stringified device/runner error).
    Backend(String),
    /// The prompt is longer than the model context window.
    ContextOverflow,
}

impl fmt::Display for GenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenError::Backend(m) => write!(f, "backend error: {m}"),
            GenError::ContextOverflow => write!(f, "prompt exceeds the model context window"),
        }
    }
}

impl std::error::Error for GenError {}

/// Structured error for the BASTION tree-verify surface (ADR 0014), carried
/// over the worker oneshot so the HTTP layer maps status codes by VARIANT
/// instead of sniffing strings (a CUDA driver error containing "not supported"
/// must not turn into a 501).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeOpError {
    /// The generator/backend cannot do tree-verify at all → HTTP 501.
    Unsupported(String),
    /// No open session (or it was invalidated by a generation) → HTTP 409.
    Conflict(String),
    /// A malformed tree / prompt (caller error) → HTTP 400.
    BadRequest(String),
    /// Device failure, panic, or any other internal fault → HTTP 500.
    Internal(String),
}

impl fmt::Display for TreeOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeOpError::Unsupported(m)
            | TreeOpError::Conflict(m)
            | TreeOpError::BadRequest(m)
            | TreeOpError::Internal(m) => f.write_str(m),
        }
    }
}

/// The inference seam: prefill a prompt and stream decode steps.
///
/// Synchronous and runtime-free by design — the serve-gated worker drives it on a
/// dedicated thread (the runner is `Send` but `&mut`-exclusive). `on_step`
/// returning `false` cancels generation (client disconnect / shutdown).
pub trait Generator: Send {
    /// Prefill `req.prompt_tokens` then decode up to `req.max_new` tokens, calling
    /// `on_step` once per decoded token. Stops early when `on_step` returns `false`.
    ///
    /// # Errors
    /// [`GenError::ContextOverflow`] if the prompt exceeds the context window;
    /// [`GenError::Backend`] on a device/runner failure.
    fn generate(
        &mut self,
        req: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError>;

    /// Model context length (for prompt-length validation).
    fn n_ctx(&self) -> usize;
    /// Vocabulary size (for logit-shape sanity).
    fn vocab(&self) -> usize;

    /// BASTION tree-verify session (ADR 0014), OPTIONAL — the default refuses.
    ///
    /// Reset the model, prefill `prompt`, and return the greedy pending token
    /// (the target argmax after the prompt — the root the orchestrator's first
    /// draft tree must carry). One session at a time; a subsequent
    /// [`Generator::generate`] invalidates it (the worker owns one model).
    fn open_tree_session(&mut self, _prompt: &[u32]) -> Result<u32, TreeOpError> {
        Err(TreeOpError::Unsupported(
            "tree-verify sessions are not supported by this generator".to_owned(),
        ))
    }

    /// Verify one draft tree against the open session (see
    /// `CudaDecodeModel::tree_verify_greedy` for the tree contract: node 0 is
    /// the pending token, `parents[i] < i`). Returns the newly committed
    /// tokens; the session's pending token becomes the last element.
    fn tree_verify(&mut self, _tokens: &[u32], _parents: &[i32]) -> Result<Vec<u32>, TreeOpError> {
        Err(TreeOpError::Unsupported(
            "tree-verify sessions are not supported by this generator".to_owned(),
        ))
    }
}

/// A model-free generator that emits a fixed script of token IDs — the contract
/// suite's reason to exist (drives the HTTP/SSE machinery with no weights).
pub struct MockGenerator {
    /// Tokens to emit (truncated to `max_new`).
    pub script: Vec<u32>,
    /// The `finish_reason` reported when the whole script is emitted within budget.
    pub end_reason: FinishReason,
    /// Reported [`Generator::n_ctx`].
    pub n_ctx: usize,
    /// Reported [`Generator::vocab`].
    pub vocab: usize,
    /// Optional per-step sleep, to simulate a slow decode (shutdown/backpressure tests).
    pub step_delay_ms: u64,
    /// Optional counter incremented per emitted step (client-disconnect test).
    pub emitted: Option<Arc<AtomicUsize>>,
}

impl fmt::Debug for MockGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockGenerator")
            .field("script_len", &self.script.len())
            .field("end_reason", &self.end_reason)
            .finish()
    }
}

impl MockGenerator {
    /// A mock that emits `script` and reports `Length` when the budget truncates it,
    /// else `end_reason`.
    #[must_use]
    pub fn new(script: Vec<u32>) -> Self {
        Self {
            script,
            end_reason: FinishReason::Length,
            n_ctx: 4096,
            vocab: 128_256,
            step_delay_ms: 0,
            emitted: None,
        }
    }
}

impl Generator for MockGenerator {
    fn generate(
        &mut self,
        req: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        let n = self.script.len().min(req.max_new);
        let truncated = req.max_new < self.script.len();
        for i in 0..n {
            if self.step_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.step_delay_ms));
            }
            let last = i + 1 == n;
            let finish_reason = if last {
                Some(if truncated {
                    FinishReason::Length
                } else {
                    self.end_reason
                })
            } else {
                None
            };
            if let Some(c) = &self.emitted {
                c.fetch_add(1, Ordering::Relaxed);
            }
            let cont = on_step(Step {
                token: self.script[i],
                finished: last,
                finish_reason,
            });
            if last || !cont {
                break;
            }
        }
        Ok(())
    }

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
}

/// A [`Generator`] wrapping the real [`tritium_nn::ModelRunner`]: re-implements the
/// prefill + per-step `forward` decode loop (the runner's `generate` returns all
/// tokens at once and is greedy-only, so serve owns this) with per-step sampling,
/// a per-step seed advance, and a context guard.
pub struct RunnerGenerator {
    runner: tritium_nn::ModelRunner,
    eos: u32,
    /// True between a successful [`Generator::open_tree_session`] and the next
    /// [`Generator::generate`] (which resets the runner and would otherwise
    /// leave `tree_verify` silently running against the chat's KV state —
    /// valid-looking committed tokens from the wrong context).
    tree_session_open: bool,
    /// Prompt-lookup speculative decoding (greedy only): draft the tokens that
    /// followed the most recent earlier occurrence of the trailing n-gram in
    /// the generation history, verify the chain with the BASTION tree verifier
    /// (`tree_verify_greedy`), and commit every accepted token in one forward.
    /// LOSSLESS by construction (the verifier only commits the target's own
    /// argmaxes); a model-free degenerate drafter — the ADR 0014 external-
    /// drafter boundary applies to model drafters, which stay outside.
    spec_lookup: bool,
}

impl fmt::Debug for RunnerGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunnerGenerator")
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl RunnerGenerator {
    /// Wrap a loaded runner, using `eos` as the stop token.
    #[must_use]
    pub fn new(runner: tritium_nn::ModelRunner, eos: u32) -> Self {
        Self {
            runner,
            eos,
            tree_session_open: false,
            spec_lookup: false,
        }
    }

    /// Enable prompt-lookup speculative decoding for greedy requests (needs
    /// the CUDA device-resident decoder at generate time; silently falls back
    /// to plain stepping otherwise).
    #[must_use]
    pub fn with_spec_lookup(mut self, on: bool) -> Self {
        self.spec_lookup = on;
        self
    }

    fn sample(logits: &[f32], s: &Sampling, step: u64) -> Option<u32> {
        match *s {
            Sampling::Greedy => tritium_nn::sample_greedy(logits),
            // Advance the seed per step so a stochastic decode actually varies (the
            // samplers take the seed per call and would otherwise repeat).
            Sampling::TopK { k, temp, seed } => {
                tritium_nn::sample_top_k(logits, k, temp, seed.wrapping_add(step))
            }
            Sampling::TopP { p, temp, seed } => {
                tritium_nn::sample_top_p(logits, p, temp, seed.wrapping_add(step))
            }
        }
    }
}

impl RunnerGenerator {
    /// Draft continuation via prompt lookup: find the LONGEST match (up to an
    /// 8-gram, at least a 2-gram — 1-gram matches draft noise and measured
    /// below break-even) of `history`'s tail against an earlier position, and
    /// return up to `max_draft` of the tokens that followed the best match.
    /// One backwards scan: for each candidate position, extend the suffix
    /// match; keep the longest (most recent wins ties).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn lookup_draft(history: &[u32], max_draft: usize) -> Vec<u32> {
        const MAX_NGRAM: usize = 8;
        const MIN_NGRAM: usize = 2;
        if max_draft == 0 || history.len() < MIN_NGRAM + 1 {
            return Vec::new();
        }
        let len = history.len();
        let max_n = MAX_NGRAM.min(len - 1);
        let mut best_len = 0usize;
        let mut best_end = 0usize; // index just past the matched needle
        // `e` = candidate "end" (exclusive) of an earlier needle occurrence.
        for e in (MIN_NGRAM..len).rev() {
            if history[e - 1] != history[len - 1] {
                continue;
            }
            let mut nlen = 1;
            while nlen < max_n && nlen < e && history[e - 1 - nlen] == history[len - 1 - nlen] {
                nlen += 1;
            }
            if nlen >= MIN_NGRAM && nlen > best_len {
                best_len = nlen;
                best_end = e;
                if nlen == max_n {
                    break;
                }
            }
        }
        if best_len == 0 || best_end == len {
            return Vec::new();
        }
        let end = (best_end + max_draft).min(len);
        history[best_end..end].to_vec()
    }

    /// The greedy speculative loop: pending token → lookup-draft a chain →
    /// `tree_verify_greedy` commits the accepted prefix + one bonus in ONE
    /// batched forward (the f16 LM-head table is read once per tree, not once
    /// per token). No-draft steps use the plain M=1 graph step. Lossless: every
    /// emitted token is the target's own greedy argmax at its position (gated
    /// by `cuda_spec_lookup_matches_plain_greedy`).
    #[cfg(feature = "cuda")]
    fn generate_spec_lookup(
        &mut self,
        req: &GenRequest,
        _prompt_len: usize,
        max_new: usize,
        prefill_logits: Vec<f32>,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        const DRAFT_MIN: usize = 6;
        const DRAFT_MAX: usize = 40;
        let n_ctx = self.runner.config.n_ctx as usize;
        let mut history: Vec<u32> = req.prompt_tokens.clone();
        let mut emitted = 0usize;
        // The plain loop (`for i in 0..max_new`) emits nothing on a zero
        // budget; match it before the first emission below.
        if max_new == 0 {
            return Ok(());
        }
        let stats = std::env::var("TRITIUM_SPEC_STATS").as_deref() == Ok("1");
        // Verify cost is ~flat in tree size next to its fixed overhead, so when
        // drafts keep being fully accepted, longer chains amortize it; an early
        // rejection resets the length.
        let mut draft_len = DRAFT_MIN;
        let (mut n_verify, mut n_committed, mut n_plain, mut t_verify, mut t_plain) = (
            0usize,
            0usize,
            0usize,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        );

        // Emit the prefill argmax exactly like the plain loop's first token.
        let mut pending = tritium_nn::sample_greedy(&prefill_logits)
            .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
        loop {
            let is_eos = req.stop_eos && pending == self.eos;
            let last = is_eos || emitted + 1 >= max_new;
            let cont = on_step(Step {
                token: pending,
                finished: last,
                finish_reason: if is_eos {
                    Some(FinishReason::Stop)
                } else if last {
                    Some(FinishReason::Length)
                } else {
                    None
                },
            });
            emitted += 1;
            if last || !cont {
                if stats && n_verify > 0 {
                    eprintln!(
                        "spec-stats: verifies={n_verify} committed={n_committed} ({:.2} tok/verify, {:.1?}/verify) plain={n_plain} ({:.1?}/step)",
                        n_committed as f64 / n_verify as f64,
                        t_verify / n_verify as u32,
                        t_plain / n_plain.max(1) as u32,
                    );
                }
                return Ok(());
            }
            history.push(pending);

            // Budget-clamped draft: total tree rows must fit the KV arena
            // (cache_len = history.len() - 1 here, the verifier needs
            // cache_len + 1 + d <= n_ctx, so d <= n_ctx - history.len()), and
            // committed tokens (<= drafts + 1) must fit the emission budget.
            let kv_room = n_ctx.saturating_sub(history.len());
            let budget = max_new - emitted;
            let max_draft = draft_len.min(kv_room).min(budget.saturating_sub(1));
            let drafts = Self::lookup_draft(&history, max_draft);

            if drafts.is_empty() {
                // Plain M=1 graph step (faster than a 1-node tree).
                let t0 = std::time::Instant::now();
                let pos = history.len() - 1;
                let logits = self
                    .runner
                    .forward(&[pending], &[pos])
                    .map_err(|e| GenError::Backend(e.to_string()))?;
                n_plain += 1;
                t_plain += t0.elapsed();
                pending = tritium_nn::sample_greedy(&logits)
                    .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
                continue;
            }

            let mut tokens = Vec::with_capacity(1 + drafts.len());
            tokens.push(pending);
            tokens.extend(&drafts);
            let parents: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
            let t0 = std::time::Instant::now();
            let committed = self
                .runner
                .tree_verify_greedy(&tokens, &parents)
                .map_err(|e| GenError::Backend(e.to_string()))?;
            n_verify += 1;
            n_committed += committed.len();
            t_verify += t0.elapsed();
            draft_len = if committed.len() == tokens.len() {
                (draft_len * 2).min(DRAFT_MAX)
            } else {
                DRAFT_MIN
            };
            // committed[..L-1] extend history as fully-processed tokens; the
            // last committed becomes the new pending (top of the next loop
            // emits it and pushes it into history).
            for &c in &committed[..committed.len() - 1] {
                let is_eos = req.stop_eos && c == self.eos;
                let last = is_eos || emitted + 1 >= max_new;
                let cont = on_step(Step {
                    token: c,
                    finished: last,
                    finish_reason: if is_eos {
                        Some(FinishReason::Stop)
                    } else if last {
                        Some(FinishReason::Length)
                    } else {
                        None
                    },
                });
                emitted += 1;
                if last || !cont {
                    if stats && n_verify > 0 {
                        eprintln!(
                            "spec-stats: verifies={n_verify} committed={n_committed} ({:.2} tok/verify, {:.1?}/verify) plain={n_plain} ({:.1?}/step)",
                            n_committed as f64 / n_verify as f64,
                            t_verify / n_verify as u32,
                            t_plain / n_plain.max(1) as u32,
                        );
                    }
                    return Ok(());
                }
                history.push(c);
            }
            pending = *committed
                .last()
                .ok_or_else(|| GenError::Backend("tree verify returned an empty commit".into()))?;
        }
    }
}

impl RunnerGenerator {
    /// The truncated, renormalized distribution the plain sampler draws from
    /// for `s`, as parallel `(indices, probs)` (see `tritium_nn::truncated_*`
    /// — the samplers are thin wrappers over these, so this IS the plain
    /// sampler's distribution, not a reimplementation).
    #[cfg(feature = "cuda")]
    fn truncated(logits: &[f32], s: &Sampling) -> Option<(Vec<u32>, Vec<f32>)> {
        match *s {
            Sampling::Greedy => tritium_nn::sample_greedy(logits).map(|t| (vec![t], vec![1.0])),
            Sampling::TopK { k, temp, .. } => tritium_nn::truncated_top_k(logits, k, temp),
            Sampling::TopP { p, temp, .. } => tritium_nn::truncated_top_p(logits, p, temp),
        }
    }

    /// One accept-rule step for a DETERMINISTIC drafter (q = δ_d): accept
    /// draft `d` iff `u < p̃(d)` (returns `None` — the draft stands); on
    /// rejection return the leftover-distribution sample — p̃ with `d`
    /// removed, renormalized (if `d` wasn't in the truncated set, p̃(d) = 0
    /// and the leftover IS p̃). Output distribution per position is exactly
    /// p̃: P(d) = p̃(d), P(x≠d) = (1 − p̃(d)) · p̃(x)/(1 − p̃(d)) = p̃(x) —
    /// gated by `spec_accept_step_is_lossless_in_distribution`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // cuda-path production code; tested everywhere
    pub(crate) fn spec_accept_step(
        idx: &[u32],
        probs: &[f32],
        d: u32,
        u: f32,
        resample_seed: u64,
    ) -> Option<u32> {
        let pd = idx.iter().position(|&t| t == d).map_or(0.0, |i| probs[i]);
        if u < pd {
            return None;
        }
        let mut r_idx = Vec::with_capacity(idx.len());
        let mut r_probs = Vec::with_capacity(probs.len());
        let mut sum = 0.0f32;
        for (&t, &pr) in idx.iter().zip(probs) {
            if t != d {
                r_idx.push(t);
                r_probs.push(pr);
                sum += pr;
            }
        }
        if r_idx.is_empty() || sum <= 0.0 {
            // Degenerate p̃ = δ_d rejected by a round-off-scale draw (p̃(d)≈1):
            // emitting d is the correct limit.
            return Some(d);
        }
        for pr in &mut r_probs {
            *pr /= sum;
        }
        Some(tritium_nn::sample_categorical(
            &r_idx,
            &r_probs,
            resample_seed,
        ))
    }

    /// splitmix64 → uniform in [0, 1). The spec path's accept/resample draws
    /// use their own deterministic stream derived from the request seed —
    /// distributional losslessness doesn't require (and can't have) the plain
    /// loop's exact random stream.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // cuda-path production code; tested everywhere
    fn spec_uniform(seed: u64, salt: u64) -> f32 {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(salt.wrapping_add(1)));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 24 bits → [0, 1) with full f32 precision.
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Speculative SAMPLING loop (the temperature > 0 twin of
    /// `generate_spec_lookup`): the prompt-lookup drafter is deterministic
    /// (q = δ_draft), so the leftover-distribution accept rule reduces to:
    /// accept draft `d` with probability p̃(d); on rejection sample from p̃
    /// with `d` removed (renormalized). Algebraically the output distribution
    /// is exactly p̃ at every position — lossless in distribution (gated at
    /// temp → 0, where p̃ collapses to argmax and the stream must EQUAL plain
    /// greedy token-for-token).
    #[cfg(feature = "cuda")]
    fn generate_spec_lookup_sampled(
        &mut self,
        req: &GenRequest,
        _prompt_len: usize,
        max_new: usize,
        prefill_logits: Vec<f32>,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        const DRAFT_MIN: usize = 6;
        const DRAFT_MAX: usize = 40;
        let n_ctx = self.runner.config.n_ctx as usize;
        let seed = match req.sampling {
            Sampling::TopK { seed, .. } | Sampling::TopP { seed, .. } => seed,
            Sampling::Greedy => 0,
        };
        if max_new == 0 {
            return Ok(());
        }
        let mut history: Vec<u32> = req.prompt_tokens.clone();
        let mut emitted = 0usize;
        let mut draft_len = DRAFT_MIN;
        // Every random decision gets a fresh salt so draws are independent.
        let mut salt = 0u64;

        let mut pending = {
            let (idx, probs) = Self::truncated(&prefill_logits, &req.sampling)
                .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
            salt += 1;
            tritium_nn::sample_categorical(&idx, &probs, seed.wrapping_add(salt))
        };
        loop {
            let is_eos = req.stop_eos && pending == self.eos;
            let last = is_eos || emitted + 1 >= max_new;
            let cont = on_step(Step {
                token: pending,
                finished: last,
                finish_reason: if is_eos {
                    Some(FinishReason::Stop)
                } else if last {
                    Some(FinishReason::Length)
                } else {
                    None
                },
            });
            emitted += 1;
            if last || !cont {
                return Ok(());
            }
            history.push(pending);

            let kv_room = n_ctx.saturating_sub(history.len());
            let budget = max_new - emitted;
            let max_draft = draft_len.min(kv_room).min(budget.saturating_sub(1));
            let drafts = Self::lookup_draft(&history, max_draft);

            if drafts.is_empty() {
                let pos = history.len() - 1;
                let logits = self
                    .runner
                    .forward(&[pending], &[pos])
                    .map_err(|e| GenError::Backend(e.to_string()))?;
                let (idx, probs) = Self::truncated(&logits, &req.sampling)
                    .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
                salt += 1;
                pending = tritium_nn::sample_categorical(&idx, &probs, seed.wrapping_add(salt));
                continue;
            }

            let mut tokens = Vec::with_capacity(1 + drafts.len());
            tokens.push(pending);
            tokens.extend(&drafts);
            let parents: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
            let logits_all = self
                .runner
                .tree_verify_logits(&tokens, &parents)
                .map_err(|e| GenError::Backend(e.to_string()))?;
            let vocab = logits_all.len() / tokens.len();

            // Chain walk with the accept rule. `path` holds tree-node indices;
            // node j's child in the chain is node j+1.
            let mut path = vec![0usize];
            let mut final_token: Option<u32> = None;
            for child in 1..tokens.len() {
                let node = child - 1;
                let row = &logits_all[node * vocab..(node + 1) * vocab];
                let (idx, probs) = Self::truncated(row, &req.sampling)
                    .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
                let d = tokens[child];
                salt += 1;
                let u = Self::spec_uniform(seed, salt);
                salt += 1;
                match Self::spec_accept_step(&idx, &probs, d, u, seed.wrapping_add(salt)) {
                    None => {
                        path.push(child);
                        continue;
                    }
                    Some(t) => {
                        final_token = Some(t);
                        break;
                    }
                }
            }
            let final_token = match final_token {
                Some(t) => t,
                None => {
                    // Full accept: bonus draw from the last node's distribution.
                    let node = *path.last().ok_or_else(|| {
                        GenError::Backend("accept walk produced an empty path".into())
                    })?;
                    let row = &logits_all[node * vocab..(node + 1) * vocab];
                    let (idx, probs) = Self::truncated(row, &req.sampling)
                        .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
                    salt += 1;
                    tritium_nn::sample_categorical(&idx, &probs, seed.wrapping_add(salt))
                }
            };
            self.runner
                .tree_commit(&path)
                .map_err(|e| GenError::Backend(e.to_string()))?;
            draft_len = if path.len() == tokens.len() {
                (draft_len * 2).min(DRAFT_MAX)
            } else {
                DRAFT_MIN
            };

            // Emit the accepted draft tokens; the final (resampled or bonus)
            // token becomes the new pending, emitted at the top of the loop.
            for &node in &path[1..] {
                let c = tokens[node];
                let is_eos = req.stop_eos && c == self.eos;
                let last = is_eos || emitted + 1 >= max_new;
                let cont = on_step(Step {
                    token: c,
                    finished: last,
                    finish_reason: if is_eos {
                        Some(FinishReason::Stop)
                    } else if last {
                        Some(FinishReason::Length)
                    } else {
                        None
                    },
                });
                emitted += 1;
                if last || !cont {
                    return Ok(());
                }
                history.push(c);
            }
            pending = final_token;
        }
    }
}

impl Generator for RunnerGenerator {
    fn generate(
        &mut self,
        req: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        let n_ctx = self.runner.config.n_ctx as usize;
        let prompt_len = req.prompt_tokens.len();
        if prompt_len == 0 || prompt_len > n_ctx {
            return Err(GenError::ContextOverflow);
        }
        let max_new = req.max_new.min(n_ctx.saturating_sub(prompt_len));

        // A generation owns the runner's KV from here on — close any tree
        // session loudly instead of letting a later verify run on chat state.
        self.tree_session_open = false;
        self.runner.reset();
        let positions: Vec<usize> = (0..prompt_len).collect();
        let mut logits = self
            .runner
            .forward(&req.prompt_tokens, &positions)
            .map_err(|e| GenError::Backend(e.to_string()))?;

        // Prompt-lookup speculative decoding (greedy only): verified chains
        // commit several tokens per forward. Falls back to plain stepping when
        // disabled, sampling is stochastic, or no CUDA resident decoder exists
        // (probed HERE — e.g. `--spec lookup` on the cpu backend — so the
        // fallback is a dispatch decision, never a mid-stream error).
        #[cfg(feature = "cuda")]
        if self.spec_lookup && self.runner.has_resident_decoder() {
            return match req.sampling {
                Sampling::Greedy => {
                    self.generate_spec_lookup(req, prompt_len, max_new, logits, on_step)
                }
                // Stochastic sampling uses the speculative accept rule
                // (lossless IN DISTRIBUTION, not stream-equal to the plain
                // loop — the plain loop and this one consume randomness
                // differently by construction).
                Sampling::TopK { .. } | Sampling::TopP { .. } => {
                    self.generate_spec_lookup_sampled(req, prompt_len, max_new, logits, on_step)
                }
            };
        }

        for i in 0..max_new {
            let next = Self::sample(&logits, &req.sampling, i as u64)
                .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
            let is_eos = req.stop_eos && next == self.eos;
            let last = is_eos || i + 1 == max_new;
            let finish_reason = if is_eos {
                Some(FinishReason::Stop)
            } else if last {
                Some(FinishReason::Length)
            } else {
                None
            };
            let cont = on_step(Step {
                token: next,
                finished: last,
                finish_reason,
            });
            if last || !cont {
                break;
            }
            let pos = prompt_len + i;
            logits = self
                .runner
                .forward(&[next], &[pos])
                .map_err(|e| GenError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    fn n_ctx(&self) -> usize {
        self.runner.config.n_ctx as usize
    }
    fn vocab(&self) -> usize {
        self.runner.weights.vocab
    }

    fn open_tree_session(&mut self, prompt: &[u32]) -> Result<u32, TreeOpError> {
        // Refuse at OPEN when verify can never succeed — cheaper and more
        // honest than burning a full prefill before the client learns.
        #[cfg(not(feature = "cuda"))]
        {
            let _ = prompt;
            return Err(TreeOpError::Unsupported(
                "tree-verify needs the `cuda` feature".to_owned(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            match self.runner.try_resident_decoder() {
                // Build failure on a CUDA backend is an internal fault (500),
                // not feature absence (501) — the pre-facade classification.
                Err(e) => return Err(TreeOpError::Internal(e.to_string())),
                Ok(false) => {
                    return Err(TreeOpError::Unsupported(
                        "tree-verify needs the CUDA device-resident decoder".to_owned(),
                    ));
                }
                Ok(true) => {}
            }
            let n_ctx = self.runner.config.n_ctx as usize;
            if prompt.is_empty() || prompt.len() >= n_ctx {
                return Err(TreeOpError::BadRequest(
                    "prompt is empty or exceeds the model context window".to_owned(),
                ));
            }
            self.runner.reset();
            let positions: Vec<usize> = (0..prompt.len()).collect();
            self.tree_session_open = false;
            let logits = self
                .runner
                .forward(prompt, &positions)
                .map_err(|e| TreeOpError::Internal(e.to_string()))?;
            let pending = tritium_nn::sample_greedy(&logits)
                .ok_or_else(|| TreeOpError::Internal("empty logits from prefill".to_owned()))?;
            self.tree_session_open = true;
            Ok(pending)
        }
    }

    fn tree_verify(&mut self, tokens: &[u32], parents: &[i32]) -> Result<Vec<u32>, TreeOpError> {
        if !self.tree_session_open {
            return Err(TreeOpError::Conflict(
                "no open tree session (open one with /v1/tree/session; a chat \
                 completion closes it)"
                    .to_owned(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            return self
                .runner
                .tree_verify_greedy(tokens, parents)
                .map_err(|e| match e {
                    tritium_nn::ResidentOpError::Unavailable => TreeOpError::Unsupported(
                        "tree-verify needs the CUDA device-resident decoder".to_owned(),
                    ),
                    // Caller-shaped errors (malformed tree, capacity overflow) → 400;
                    // anything else is a device/internal fault → 500.
                    tritium_nn::ResidentOpError::Op(
                        tritium_spec::BackendError::InvalidInput(m),
                    ) => TreeOpError::BadRequest(m),
                    other => TreeOpError::Internal(other.to_string()),
                });
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (tokens, parents);
            Err(TreeOpError::Unsupported(
                "tree-verify needs the `cuda` feature".to_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    /// Monte-Carlo gate for the deterministic-drafter accept rule: over many
    /// independent (u, resample) draws, the per-position output distribution
    /// must equal p̃ regardless of which token the drafter proposed. Pure host
    /// math — runs on every lane, no GPU needed.
    #[test]
    fn spec_accept_step_is_lossless_in_distribution() {
        // p̃ over 4 tokens (already truncated + renormalized).
        let idx = [10u32, 11, 12, 13];
        let probs = [0.5f32, 0.25, 0.15, 0.10];
        let trials = 200_000usize;
        // Drafter proposals to exercise: in-set high, in-set low, out-of-set.
        for &d in &[10u32, 13, 99] {
            let mut counts = std::collections::HashMap::<u32, usize>::new();
            for t in 0..trials {
                let u = RunnerGenerator::spec_uniform(0xC0FFEE, t as u64);
                let got = match RunnerGenerator::spec_accept_step(
                    &idx,
                    &probs,
                    d,
                    u,
                    0xBADD_5EED_u64.wrapping_add(t as u64),
                ) {
                    None => d,
                    Some(x) => x,
                };
                *counts.entry(got).or_insert(0) += 1;
            }
            for (&t, &p) in idx.iter().zip(&probs) {
                let emp = counts.get(&t).copied().unwrap_or(0) as f64 / trials as f64;
                assert!(
                    (emp - f64::from(p)).abs() < 0.01,
                    "draft {d}: token {t} empirical {emp:.4} vs p̃ {p} (must match: lossless)"
                );
            }
            // Nothing outside p̃'s support may ever be emitted.
            assert!(
                counts.keys().all(|t| idx.contains(t)),
                "draft {d}: out-of-support emission"
            );
        }
    }

    use super::*;

    /// The mock drives `on_step` for each scripted token and reports the right
    /// finish_reason (Length when truncated by budget, else end_reason).
    #[test]
    fn mock_emits_script_and_finish_reason() {
        let mut g = MockGenerator {
            end_reason: FinishReason::Stop,
            ..MockGenerator::new(vec![10, 11, 12])
        };
        let mut seen = Vec::new();
        let mut last_reason = None;
        g.generate(
            &GenRequest {
                prompt_tokens: vec![1],
                max_new: 8,
                sampling: Sampling::Greedy,
                stop_eos: true,
            },
            &mut |s| {
                seen.push(s.token);
                if let Some(r) = s.finish_reason {
                    last_reason = Some(r);
                }
                true
            },
        )
        .unwrap();
        assert_eq!(seen, vec![10, 11, 12]);
        assert_eq!(last_reason, Some(FinishReason::Stop));
    }

    /// A budget shorter than the script truncates and reports Length.
    #[test]
    fn mock_truncates_to_max_new_with_length() {
        let mut g = MockGenerator::new(vec![1, 2, 3, 4, 5]);
        let mut count = 0usize;
        let mut last_reason = None;
        g.generate(
            &GenRequest {
                prompt_tokens: vec![1],
                max_new: 2,
                sampling: Sampling::Greedy,
                stop_eos: true,
            },
            &mut |s| {
                count += 1;
                last_reason = s.finish_reason.or(last_reason);
                true
            },
        )
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(last_reason, Some(FinishReason::Length));
    }

    /// `on_step` returning false cancels early.
    #[test]
    fn mock_cancels_when_on_step_false() {
        let mut g = MockGenerator::new(vec![1, 2, 3, 4, 5]);
        let mut count = 0usize;
        g.generate(
            &GenRequest {
                prompt_tokens: vec![1],
                max_new: 5,
                sampling: Sampling::Greedy,
                stop_eos: true,
            },
            &mut |_s| {
                count += 1;
                count < 2 // stop after the 2nd
            },
        )
        .unwrap();
        assert_eq!(count, 2);
    }
}
