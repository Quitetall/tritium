//! The seam between the HTTP layer and inference.
//!
//! This module is **always compiled** and **runtime-free** (no tokio/axum), so the
//! default workspace build pulls in no async deps. The `serve`-gated worker drives
//! a [`Generator`] on a dedicated thread; contract tests drive [`MockGenerator`]
//! directly with no model.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Process-wide spec-decode telemetry, read by `/metrics` (the generator is
/// deliberately runtime-free, so counters are statics rather than plumbing).
pub static SPEC_VERIFIES: AtomicU64 = AtomicU64::new(0);
/// Tokens committed by spec verifies (tok/verify = committed / verifies).
pub static SPEC_COMMITTED: AtomicU64 = AtomicU64::new(0);

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
    /// When `Some(k)`, each emitted token carries its logprob plus the top-`k`
    /// alternatives (OpenAI `logprobs`/`top_logprobs`). Supported on the
    /// plain and batched paths; spec-lookup falls back to plain stepping.
    pub logprobs: Option<usize>,
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
#[derive(Debug, Clone)]
pub struct Step {
    /// The decoded token ID (special tokens like EOS are dropped by the HTTP detok).
    pub token: u32,
    /// True on the terminal step (EOS hit or budget reached).
    pub finished: bool,
    /// Set on the terminal step.
    pub finish_reason: Option<FinishReason>,
    /// `(token, logprob)` — the SAMPLED token first, then the top-k
    /// alternatives by logprob. Present only when the request asked.
    pub logprobs: Option<Vec<(u32, f32)>>,
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
                // Deterministic synthetic logprobs: sampled at -0.1, k
                // alternatives at -1.0, -2.0, ... (token id = sampled + j).
                logprobs: req.logprobs.map(|k| {
                    let t = self.script[i];
                    let mut v = vec![(t, -0.1f32)];
                    v.extend((1..=k as u32).map(|j| (t.wrapping_add(j), -(j as f32))));
                    v
                }),
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
    /// ADR 0021 model drafter: a second (small, ternary) runner drafting K
    /// greedy tokens per verify, replacing `lookup_draft` when present.
    /// Lossless like lookup — the verifier only commits target argmaxes.
    draft: Option<Box<tritium_nn::ModelRunner>>,
    /// Number of leading positions of the DRAFT's KV known to match the
    /// generation history.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    draft_pos: usize,
    /// Tokens fed to the draft at positions `draft_pos..` during the LAST
    /// draft call (pending + its own greedy outputs). The next call compares
    /// them against the committed history: matches advance `draft_pos`
    /// (forward-contiguous — the resident runner rejects position rewinds);
    /// the first mismatch (a rejected draft) resets the draft for a full
    /// re-prefill.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    draft_fed: Vec<u32>,
}

impl fmt::Debug for RunnerGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunnerGenerator")
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

/// Log-softmax top-k: the sampled token's logprob first, then the `k`
/// highest-logprob alternatives (which MAY include the sampled token again —
/// matching OpenAI, whose top_logprobs contains the sampled token). One
/// O(vocab) pass + an O(vocab·log k)-ish partial select — computed only when
/// the request asked for logprobs.
pub(crate) fn top_logprobs(logits: &[f32], sampled: u32, k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
    let mut out = Vec::with_capacity(k + 1);
    out.push((sampled, logits[sampled as usize] - lse));
    if k > 0 {
        // Partial top-k by logit (logprob is monotone in logit).
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (i, &l) in logits.iter().enumerate() {
            if top.len() < k {
                top.push((i as u32, l));
                top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            } else if l > top[k - 1].1 {
                top[k - 1] = (i as u32, l);
                top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            }
        }
        out.extend(top.into_iter().map(|(i, l)| (i, l - lse)));
    }
    out
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
            draft: None,
            draft_pos: 0,
            draft_fed: Vec::new(),
        }
    }

    /// Attach an ADR 0021 draft model (a second runner on the same device).
    /// Enables the spec path for greedy/sampled requests; `--spec lookup`'s
    /// prompt-lookup remains the drafter when no model is attached.
    #[must_use]
    pub fn with_draft_model(mut self, draft: tritium_nn::ModelRunner) -> Self {
        self.draft = Some(Box::new(draft));
        self
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

/// Adaptive draft-length policy for the two-runner spec loops.
///
/// The accept walk is per-token Bernoulli-like: each drafted token survives
/// with roughly the drafter's per-token acceptance rate `a` (position- and
/// content-dependent in truth; an EWMA over recent per-token outcomes tracks
/// the mixture). Drafting the k-th token pays one fixed drafter step and is
/// only worth it while the chance the walk still lives at k — `a^k` — clears
/// the step's cost share, so the policy drafts while `a^k >= THRESHOLD`:
/// `k = ln(THRESHOLD)/ln(a)`, clamped to [1, MAX].
///
/// Replaces the old double-on-full-accept / reset-on-any-reject rule, which
/// pinned k at its floor of 6 for any drafter whose full-6 acceptance is rare
/// (a tau~4 drafter wastes ~5 draft steps per cycle on weak prose) and could
/// never drop below 6 where weak content wants k~1-2. The draft length NEVER
/// affects outputs — the accept rule is lossless for any k (greedy: verify
/// recomputes the argmax chain; sampled: the residual-distribution rule) —
/// so this is purely a cost policy with no numerics gate.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum DraftPolicy {
    /// Acceptance-adaptive (the default; see the doc above).
    Adaptive {
        /// EWMA of per-token acceptance outcomes in [0, 1].
        acc: f64,
    },
    /// The pre-2026-07-19 rule: double on full acceptance, reset to 6 on any
    /// rejection. Kept selectable (`TRITIUM_DRAFT_K=legacy`) as the A/B
    /// baseline and kill switch.
    Legacy { len: usize },
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl DraftPolicy {
    /// Per-token EWMA decay — ~20-token memory (a few verify cycles), fast
    /// enough to track topic shifts inside one request.
    const LAMBDA: f64 = 0.95;
    /// Draft while the survival probability `a^k` clears this. 0.25 sits at
    /// the measured drafter-step / verify-cycle cost band (~0.75 ms draft
    /// step vs ~4 ms verify-side cost per committed token, 2026-07-19 stage-3
    /// runs): a marginal draft with under 1-in-4 use odds costs more than it
    /// can save.
    const THRESHOLD: f64 = 0.25;
    /// Hard cap (the old DRAFT_MAX; also bounded per cycle by KV room and
    /// the emission budget at the call sites).
    const MAX: usize = 40;

    const LEGACY_MIN: usize = 6;

    /// Optimistic start: strong content ramps immediately; weak content
    /// converges down within a couple of cycles. `TRITIUM_DRAFT_K=legacy`
    /// selects the old fixed rule (loud-reject on anything else).
    fn from_env() -> Result<Self, GenError> {
        match std::env::var("TRITIUM_DRAFT_K") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Adaptive { acc: 0.75 }),
            Err(std::env::VarError::NotUnicode(v)) => Err(GenError::Backend(format!(
                "TRITIUM_DRAFT_K={v:?} — use adaptive (default) or legacy"
            ))),
            Ok(v) if v == "adaptive" => Ok(Self::Adaptive { acc: 0.75 }),
            Ok(v) if v == "legacy" => Ok(Self::Legacy {
                len: Self::LEGACY_MIN,
            }),
            Ok(v) => Err(GenError::Backend(format!(
                "TRITIUM_DRAFT_K={v:?} — use adaptive (default) or legacy"
            ))),
        }
    }

    /// Fold one verify cycle: `offered` drafts went in, `accepted` of them
    /// survived the walk (accepted < offered means the walk died at draft
    /// accepted+1 — one observed failure; accepted == offered is truncation,
    /// not a failure).
    fn update(&mut self, offered: usize, accepted: usize) {
        match self {
            Self::Adaptive { acc } => {
                for _ in 0..accepted {
                    *acc = Self::LAMBDA * *acc + (1.0 - Self::LAMBDA);
                }
                if accepted < offered {
                    *acc *= Self::LAMBDA;
                }
            }
            Self::Legacy { len } => {
                *len = if accepted == offered {
                    (*len * 2).min(Self::MAX)
                } else {
                    Self::LEGACY_MIN
                };
            }
        }
    }

    /// Draft length for the next cycle.
    fn len(&self) -> usize {
        match self {
            Self::Adaptive { acc } => {
                let a = acc.clamp(0.05, 0.98);
                ((Self::THRESHOLD.ln() / a.ln()) as usize).clamp(1, Self::MAX)
            }
            Self::Legacy { len } => *len,
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

    #[cfg(feature = "cuda")]
    /// Draft up to `max_draft` tokens with the attached draft model (greedy),
    /// starting after `history` (whose last element is the pending token).
    /// Syncs the draft's KV to `history` first by re-feeding the gap — see
    /// `draft_pos` for why that is always correct. Stops early at the draft's
    /// own EOS. Returns `[]` on any draft error (caller falls back to a plain
    /// step; drafting must never break generation).
    fn model_draft(&mut self, history: &[u32], max_draft: usize) -> Vec<u32> {
        let Some(draft) = self.draft.as_mut() else {
            return Vec::new();
        };
        if max_draft == 0 || history.is_empty() {
            return Vec::new();
        }
        let p = history.len() - 1; // pending's position
        let draft_ctx = draft.config.n_ctx as usize;
        if p + max_draft + 1 >= draft_ctx {
            return Vec::new(); // draft context exhausted; plain steps carry on
        }
        // Reconcile last call's speculatively-fed tokens against the now-
        // committed history: matches ADVANCE the watermark (those KV rows are
        // proven correct); the first mismatch means a rejected draft sits in
        // the KV — the resident runner rejects position rewinds, so recover
        // with a reset + full re-prefill (rare for a good drafter).
        let mut clean = true;
        for (i, &fed) in self.draft_fed.iter().enumerate() {
            if history.get(self.draft_pos + i) != Some(&fed) {
                clean = false;
                break;
            }
        }
        if clean {
            self.draft_pos = (self.draft_pos + self.draft_fed.len()).min(p);
        } else {
            draft.reset();
            self.draft_pos = 0;
        }
        self.draft_fed.clear();
        // Forward-contiguous sync: feed the history the draft hasn't seen,
        // EXCLUDING pending (fed by the loop below).
        if self.draft_pos < p {
            let gap: Vec<u32> = history[self.draft_pos..p].to_vec();
            let positions: Vec<usize> = (self.draft_pos..p).collect();
            if draft.forward(&gap, &positions).is_err() {
                draft.reset();
                self.draft_pos = 0; // full resync next time
                return Vec::new();
            }
            self.draft_pos = p;
        }
        let mut out = Vec::with_capacity(max_draft);
        let mut tok = history[p];
        for i in 0..max_draft {
            // Fast path: graph replay + device argmax, 4 bytes back per step
            // (the logits download + host scan dominated the drafter's cost).
            // `Ok(None)` = no resident decoder -> eager logits + host argmax
            // (same token by the pinned tie rule).
            let next = match draft.decode_greedy_step(tok, p + i) {
                Ok(Some(id)) => id,
                Ok(None) => {
                    let Ok(logits) = draft.forward(&[tok], &[p + i]) else {
                        break;
                    };
                    // The forward advanced the drafter's KV — record the fed
                    // token BEFORE any bail (the reconcile logic needs it).
                    self.draft_fed.push(tok);
                    let Some(next) = tritium_nn::sample_greedy(&logits) else {
                        break;
                    };
                    out.push(next);
                    if next == self.eos {
                        break;
                    }
                    tok = next;
                    continue;
                }
                Err(_) => break,
            };
            self.draft_fed.push(tok);
            out.push(next);
            if next == self.eos {
                break; // the draft believes the turn ends here
            }
            tok = next;
        }
        out
    }

    /// The greedy speculative loop: pending token → draft a chain (model
    /// drafter when attached, else prompt-lookup) → `tree_verify_greedy`
    /// commits the accepted prefix + one bonus in ONE batched forward (the
    /// f16 LM-head table is read once per tree, not once per token). No-draft
    /// steps use the plain M=1 graph step. Lossless: every emitted token is
    /// the target's own greedy argmax at its position (gated by
    /// `cuda_spec_lookup_matches_plain_greedy`).
    #[cfg(feature = "cuda")]
    fn generate_spec_lookup(
        &mut self,
        req: &GenRequest,
        _prompt_len: usize,
        max_new: usize,
        prefill_logits: Vec<f32>,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        let n_ctx = self.runner.config.n_ctx as usize;
        let mut history: Vec<u32> = req.prompt_tokens.clone();
        let mut emitted = 0usize;
        // The plain loop (`for i in 0..max_new`) emits nothing on a zero
        // budget; match it before the first emission below.
        if max_new == 0 {
            return Ok(());
        }
        let stats = std::env::var("TRITIUM_SPEC_STATS").as_deref() == Ok("1");
        // Acceptance-adaptive draft length (see DraftPolicy).
        let mut policy = DraftPolicy::from_env()?;
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
                logprobs: None,
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
            let max_draft = policy.len().min(kv_room).min(budget.saturating_sub(1));
            let drafts = if self.draft.is_some() {
                self.model_draft(&history, max_draft)
            } else {
                Self::lookup_draft(&history, max_draft)
            };

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
            SPEC_VERIFIES.fetch_add(1, Ordering::Relaxed);
            SPEC_COMMITTED.fetch_add(committed.len() as u64, Ordering::Relaxed);
            t_verify += t0.elapsed();
            // committed = accepted drafts + the final token, so accepted
            // drafts = committed.len() - 1 (saturating: an empty `committed`
            // violates tree_verify_greedy's contract and is caught below, but
            // a wrapped usize here would spin the EWMA fold first).
            policy.update(drafts.len(), committed.len().saturating_sub(1));
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
                    logprobs: None,
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
        // Acceptance-adaptive draft length (see DraftPolicy).
        let mut policy = DraftPolicy::from_env()?;
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
                logprobs: None,
            });
            emitted += 1;
            if last || !cont {
                return Ok(());
            }
            history.push(pending);

            let kv_room = n_ctx.saturating_sub(history.len());
            let budget = max_new - emitted;
            let max_draft = policy.len().min(kv_room).min(budget.saturating_sub(1));
            let drafts = if self.draft.is_some() {
                self.model_draft(&history, max_draft)
            } else {
                Self::lookup_draft(&history, max_draft)
            };

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
            SPEC_VERIFIES.fetch_add(1, Ordering::Relaxed);
            // Committed = accepted drafts (path[1..]) + the resampled/bonus
            // final token — same accounting as the greedy path's `committed`.
            SPEC_COMMITTED.fetch_add(path.len() as u64, Ordering::Relaxed);
            // path = root + accepted drafts, so accepted = path.len() - 1;
            // offered = drafts.len().
            policy.update(drafts.len(), path.len() - 1);

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
                    logprobs: None,
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
        // Spec paths emit committed tokens without per-token logits on the
        // host, so logprobs requests take the plain path (exact semantics).
        #[cfg(feature = "cuda")]
        {
            if let Some(d) = self.draft.as_mut() {
                // Fresh request: the draft re-prefills lazily via the sync gap.
                d.reset();
                self.draft_pos = 0;
                self.draft_fed.clear();
            }
            if (self.spec_lookup || self.draft.is_some())
                && req.logprobs.is_none()
                && self.runner.has_resident_decoder()
            {
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
                logprobs: req.logprobs.map(|k| top_logprobs(&logits, next, k)),
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
            Err(TreeOpError::Unsupported(
                "tree-verify needs the `cuda` feature".to_owned(),
            ))
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
                    tritium_nn::ResidentOpError::Op(tritium_spec::BackendError::InvalidInput(
                        m,
                    )) => TreeOpError::BadRequest(m),
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
    #[test]
    fn draft_policy_converges_down_on_rejections() {
        let mut p = super::DraftPolicy::Adaptive { acc: 0.75 };
        assert!(p.len() >= 4, "optimistic start should offer >= 4");
        // Weak content: offer whatever the policy says, accept 1, reject.
        for _ in 0..12 {
            let k = p.len();
            p.update(k, 1.min(k.saturating_sub(1)));
        }
        assert!(
            p.len() <= 3,
            "sustained low acceptance must shrink the draft (got {})",
            p.len()
        );
    }

    #[test]
    fn draft_policy_ramps_on_full_acceptance() {
        let mut p = super::DraftPolicy::Adaptive { acc: 0.75 };
        for _ in 0..40 {
            let k = p.len();
            p.update(k, k); // truncation, not failure
        }
        assert!(
            p.len() >= 12,
            "sustained full acceptance must grow the draft (got {})",
            p.len()
        );
        assert!(p.len() <= super::DraftPolicy::MAX);
    }

    #[test]
    fn draft_policy_len_always_in_bounds() {
        let mut p = super::DraftPolicy::Adaptive { acc: 0.75 };
        for i in 0..200 {
            let k = p.len();
            assert!((1..=super::DraftPolicy::MAX).contains(&k));
            // Alternate hostile patterns incl. zero-offered cycles.
            match i % 3 {
                0 => p.update(k, 0),
                1 => p.update(0, 0),
                _ => p.update(k, k),
            }
        }
    }

    #[test]
    fn draft_policy_known_rates() {
        // Directly pin the k = ln(theta)/ln(a) mapping at the clamp edges.
        let mk = |a: f64| super::DraftPolicy::Adaptive { acc: a }.len();
        assert_eq!(mk(0.5), 2);
        assert_eq!(mk(0.8), 6);
        assert!(mk(0.95) >= 13);
        assert_eq!(mk(0.01), 1); // clamped floor
        assert_eq!(mk(1.5), super::DraftPolicy::MAX); // clamped ceiling
    }

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
                logprobs: None,
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
                logprobs: None,
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
                logprobs: None,
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
