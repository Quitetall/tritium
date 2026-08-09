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
/// Plain-decode tokens committed while the adaptive spec governor had
/// drafting SUPPRESSED (the long-ctx tau-collapse lever, [`SpecGovernor`]).
/// A nonzero value is the observable "spec is off right now" signal.
pub static SPEC_SUPPRESSED_PLAIN: AtomicU64 = AtomicU64::new(0);

/// Process-wide EWMA of one spec-decode phase's wall-clock cost in
/// microseconds. Single-writer (the one decode worker thread) with `Relaxed`
/// atomics — `/metrics` reads racily, and a torn mean/count pair only
/// mis-reports a gauge by one sample. The timers are ~free: two
/// `Instant::now()` around calls that already synchronize on the device
/// (WALL time, no added syncs).
#[derive(Debug)]
pub struct Ewma {
    /// Current mean as `f64` bits; meaningless until `n > 0`.
    mean_bits: AtomicU64,
    /// Samples folded so far (saturates the warmup check, never resets).
    n: AtomicU64,
}

impl Ewma {
    /// Decay: ~ the last dozen samples dominate — fast enough to track a
    /// request's context growing (V and the drafter re-sync cost are
    /// ctx-dependent), slow enough to ride out scheduler noise.
    const ALPHA: f64 = 0.2;

    const fn new() -> Self {
        Self {
            mean_bits: AtomicU64::new(0),
            n: AtomicU64::new(0),
        }
    }

    /// Fold one wall-clock sample (microseconds). Non-finite or negative
    /// samples are dropped (defensive; `Instant` cannot produce them).
    pub fn record(&self, us: f64) {
        if !us.is_finite() || us < 0.0 {
            return;
        }
        let n = self.n.load(Ordering::Relaxed);
        let mean = if n == 0 {
            us
        } else {
            Self::ALPHA * us
                + (1.0 - Self::ALPHA) * f64::from_bits(self.mean_bits.load(Ordering::Relaxed))
        };
        self.mean_bits.store(mean.to_bits(), Ordering::Relaxed);
        self.n.store(n.saturating_add(1), Ordering::Relaxed);
    }

    /// Current mean in microseconds; `None` until the first sample.
    pub fn mean_us(&self) -> Option<f64> {
        (self.n.load(Ordering::Relaxed) > 0)
            .then(|| f64::from_bits(self.mean_bits.load(Ordering::Relaxed)))
    }

    /// Samples folded so far.
    pub fn samples(&self) -> u64 {
        self.n.load(Ordering::Relaxed)
    }
}

impl Default for Ewma {
    fn default() -> Self {
        Self::new()
    }
}

/// Measured ADR 0032 cost model for the adaptive spec governor: spec wins
/// per committed token when `(V + k·d)/τ < P`, so the breakeven is
/// `τ_floor = (V + k·d)/P` at the CURRENT draft length `k`. The governor
/// derives its floors from these runtime EWMAs instead of the fixed
/// constants once each required timer has [`Self::WARMUP`] samples —
/// tier/rung awareness comes FREE via measurement (the fast tier's cheaper
/// verify lowers V, and with it the floor, automatically; no per-tier
/// tables).
///
/// SHARING: these EWMAs are process-wide statics, so they carry across
/// SEQUENTIAL requests — a fresh request's first cycles are governed by the
/// previous request's warm floors until its own samples decay them in (~a
/// dozen at ALPHA 0.2; intentional — the phase costs are properties of the
/// model+box+ctx band, not of a request, and a warm start beats the fixed
/// fallback) — and across every server instance in one process (concurrent
/// writers interleave into one mixture; the `Relaxed` mean/count pairs can
/// tear, mis-reporting a gauge by one sample — benign, see [`Ewma`]).
#[derive(Debug)]
pub struct SpecCostModel {
    /// V (solo): wall of one `tree_verify_greedy` call, µs.
    pub verify: Ewma,
    /// P (solo): wall of one plain M=1 step, µs.
    pub plain: Ewma,
    /// d (solo): drafter wall per DRAFTED token in STEADY-STATE drafting
    /// (unsuppressed cycles), µs — the small per-cycle reconcile share
    /// included. Probe-time drafter walls go to [`Self::draft_resync`]
    /// instead, NEVER here: the floor's d must price what a recovered
    /// drafter would pay per cycle, not the one-off cost of finding out
    /// (see [`Self::record_draft`]).
    pub draft_tok: Ewma,
    /// Probe-time drafter wall per drafted token, µs — dominated by the
    /// ~ctx-linear re-prefill after [`SpecGovernor::PROBE_PERIOD`] stale
    /// plain tokens. TELEMETRY ONLY (`/metrics` phase="draft_resync"),
    /// deliberately excluded from the floors: the re-sync is an
    /// entry-transition cost of LEAVING suppression, not steady-state spec
    /// economics. Pricing it into d latched suppression — probes were the
    /// only d samples while suppressed, so the EWMA converged to
    /// resync-dominated values, the floor clamped high, and a genuinely
    /// recovered drafter (τ 2.4–2.9 measured on the fast tier, honest
    /// floor ≈ 2.15) could never clear it. The ENTRY decision is
    /// unaffected by the exclusion: entry happens while unsuppressed,
    /// where every draft is steady-state anyway.
    pub draft_resync: Ewma,
    /// V_round (batched): wall of one grouped multi-slot verify, µs.
    pub verify_round: Ewma,
    /// P_lockstep (batched): wall of one lockstep `decode_batch_graph`
    /// step, µs.
    pub lockstep: Ewma,
}

impl SpecCostModel {
    /// Samples required of EACH timer a floor depends on before the derived
    /// floor replaces the fixed fallback — one noisy first sample must not
    /// steer suppression.
    pub const WARMUP: u64 = 8;
    /// Derived-floor clamp (solo): never below 1.05 (spec must still beat
    /// plain by SOMETHING — τ estimates carry model error) and never above
    /// 3.0 (a floor that suppresses a τ≈3 drafter is a cost-model outlier,
    /// not a decision we trust unmeasured).
    pub const SOLO_CLAMP: (f64, f64) = (1.05, 3.0);
    /// Derived-floor clamp (batched): the lower bound KEEPS the measured
    /// 1.1 ride-along floor (suppressing τ≈1.4 slots cost 32% aggregate,
    /// 2026-08-09) — the model can only RAISE the batched floor when round
    /// verifies are measurably expensive relative to lockstep.
    pub const BATCHED_CLAMP: (f64, f64) = (1.1, 3.0);

    const fn new() -> Self {
        Self {
            verify: Ewma::new(),
            plain: Ewma::new(),
            draft_tok: Ewma::new(),
            draft_resync: Ewma::new(),
            verify_round: Ewma::new(),
            lockstep: Ewma::new(),
        }
    }

    /// Route one drafter wall sample (µs per DRAFTED token): steady-state
    /// cycles feed [`Self::draft_tok`] (the floor's d), probe cycles feed
    /// [`Self::draft_resync`] (telemetry only — the field docs carry the
    /// design note on why the floor excludes re-sync). The one seam both
    /// spec loops record through, so the split cannot drift between them.
    pub fn record_draft(&self, us_per_tok: f64, is_probe: bool) {
        if is_probe {
            self.draft_resync.record(us_per_tok);
        } else {
            self.draft_tok.record(us_per_tok);
        }
    }

    /// SOLO derived floor at draft length `k`: `(V + k·d)/P`, clamped to
    /// [`Self::SOLO_CLAMP`]. `None` until V, P, and d each have
    /// [`Self::WARMUP`] samples (callers fall back to the fixed
    /// [`SpecGovernor::TAU_FLOOR_SOLO`]). d is STEADY-STATE drafting cost
    /// by construction — probe-time re-sync walls are recorded to
    /// [`Self::draft_resync`] and intentionally excluded (an
    /// entry-transition cost, not steady-state economics; see the field
    /// docs), so the floor prices the regime the drafter would run in
    /// AFTER suppression lifts.
    pub fn solo_floor(&self, k: usize) -> Option<f64> {
        if self.verify.samples() < Self::WARMUP
            || self.plain.samples() < Self::WARMUP
            || self.draft_tok.samples() < Self::WARMUP
        {
            return None;
        }
        let (v, p, d) = (
            self.verify.mean_us()?,
            self.plain.mean_us()?,
            self.draft_tok.mean_us()?,
        );
        if p <= 0.0 {
            return None; // degenerate mean (all-zero wall samples): stay fixed
        }
        let (lo, hi) = Self::SOLO_CLAMP;
        Some(((v + k as f64 * d) / p).clamp(lo, hi))
    }

    /// BATCHED derived floor for a pool of `n_live` slots — the v1 MARGINAL
    /// approximation, deliberately simple and conservative:
    ///
    /// A spec round replaces one lockstep step (cost P_lockstep, commits 1
    /// token per live slot), so the pool's marginal premium for running the
    /// round is `V_round − P_lockstep` (the drafter step is shared and
    /// ride-along cheap — excluded, which only makes the floor LOWER /
    /// spec-friendlier; the trunk runs at the padded bucket size, so a
    /// single slot's true marginal share is smaller still). Splitting that
    /// premium evenly, a slot must beat its 1-token lockstep commit by its
    /// share of the premium measured in plain steps:
    ///
    /// `floor = 1 + max(0, V_round − P_lockstep) / (n_live · P_lockstep)`
    ///
    /// clamped to [`Self::BATCHED_CLAMP`] — i.e. the fixed 1.1 floor holds
    /// unless round verifies measurably exceed a lockstep step. `None`
    /// until V_round and P_lockstep each have [`Self::WARMUP`] samples.
    ///
    /// V_round is an EWMA of raw group walls recorded at whatever group
    /// size was live THEN; dividing by the CURRENT `n_live` at application
    /// mis-prices the floor transiently after a pool resize (until the
    /// EWMA re-converges, ~a dozen rounds at ALPHA 0.2) — a known v1
    /// approximation the [1.1, 3.0] clamps bound.
    pub fn batched_floor(&self, n_live: usize) -> Option<f64> {
        if self.verify_round.samples() < Self::WARMUP
            || self.lockstep.samples() < Self::WARMUP
            || n_live == 0
        {
            return None;
        }
        let (vr, p) = (self.verify_round.mean_us()?, self.lockstep.mean_us()?);
        if p <= 0.0 {
            return None;
        }
        let (lo, hi) = Self::BATCHED_CLAMP;
        Some((1.0 + (vr - p).max(0.0) / (n_live as f64 * p)).clamp(lo, hi))
    }
}

/// The process-wide cost model (the generator is runtime-free, so statics
/// rather than plumbing — the [`SPEC_VERIFIES`] pattern). Read by
/// `/metrics` (`tritium_spec_cost_us`, `tritium_spec_floor`).
pub static SPEC_COST: SpecCostModel = SpecCostModel::new();

/// Last floor the solo governor actually applied (f64 bits; gauge
/// `tritium_spec_floor{path="solo"}`). Starts at the fixed fallback.
pub static SPEC_FLOOR_SOLO: AtomicU64 = AtomicU64::new(SpecGovernor::TAU_FLOOR_SOLO.to_bits());
/// Last floor the batched governor actually applied (f64 bits; gauge
/// `tritium_spec_floor{path="batched"}`). Starts at the fixed fallback.
pub static SPEC_FLOOR_BATCHED: AtomicU64 =
    AtomicU64::new(SpecGovernor::TAU_FLOOR_BATCHED.to_bits());

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
    /// Work admitted before shutdown but cancelled while queued → HTTP 503.
    Draining(String),
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
            TreeOpError::Draining(m)
            | TreeOpError::Unsupported(m)
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

/// `TRITIUM_DRAFT_CHAIN=0` disables the L1' chained device-side draft
/// (per-step ladder instead) — the A/B baseline + kill switch, the
/// TRITIUM_DRAFT_K pattern. Loud-reject on anything else.
// Consumed by the cuda-gated spec-decode generators and the batched
// worker's I0 solo-spec path (ADR 0032 L3 I0).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn draft_chain_from_env() -> Result<bool, GenError> {
    match std::env::var("TRITIUM_DRAFT_CHAIN") {
        Err(std::env::VarError::NotPresent) => Ok(true),
        Ok(v) if v == "1" => Ok(true),
        Ok(v) if v == "0" => Ok(false),
        Ok(v) => Err(GenError::Backend(format!(
            "TRITIUM_DRAFT_CHAIN={v:?} — use 1 (default) or 0"
        ))),
        Err(e) => Err(GenError::Backend(format!("TRITIUM_DRAFT_CHAIN: {e}"))),
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
pub(crate) enum DraftPolicy {
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
    pub(crate) fn from_env() -> Result<Self, GenError> {
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
    pub(crate) fn update(&mut self, offered: usize, accepted: usize) {
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
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Adaptive { acc } => {
                let a = acc.clamp(0.05, 0.98);
                ((Self::THRESHOLD.ln() / a.ln()) as usize).clamp(1, Self::MAX)
            }
            Self::Legacy { len } => *len,
        }
    }

    /// Expected committed tokens per verify (τ) at the CURRENT policy state,
    /// under the geometric accept model the EWMA tracks: `1` bonus token plus
    /// `a·(1−a^k)/(1−a)` accepted drafts at `k = len()`. `None` for the
    /// legacy policy — it carries no acceptance estimate, so the adaptive
    /// governor stays inert on the `TRITIUM_DRAFT_K=legacy` A/B baseline.
    pub(crate) fn tau_estimate(&self) -> Option<f64> {
        match self {
            Self::Adaptive { acc } => {
                let a = acc.clamp(0.05, 0.98);
                let k = self.len() as i32;
                Some(1.0 + a * (1.0 - a.powi(k)) / (1.0 - a))
            }
            Self::Legacy { .. } => None,
        }
    }
}

/// Adaptive spec on/off governor (the long-ctx τ-collapse lever, 2026-08-08
/// final sweep): at long context the drafter's acceptance collapses
/// (τ ≈ 1.2–1.35 at ctx ≈ 3776), where a verify cycle commits so few tokens
/// that speculative decoding is a NET slowdown in every kernel tier (0.37×
/// plain on the exact tier). The `DraftPolicy` EWMA already measures this —
/// the governor turns it into a decision: when the policy's τ estimate sits
/// below a conservative breakeven floor for a sustained window, STOP
/// drafting/verifying and run plain steps, while still probing with one
/// short spec cycle every [`Self::PROBE_PERIOD`] committed tokens so
/// drafting resumes if acceptance returns (context shifts as generation
/// proceeds).
///
/// The floor is CALLER-scoped (passed to [`Self::on_verify`]) and, since the
/// cost-model lever, MEASURED: the governor derives it from the runtime's
/// own phase timers ([`SpecCostModel`] — `τ_floor = (V + k·d)/P` at the
/// current draft length in the solo loops; the marginal-round approximation
/// in the multi-slot pool) via [`Self::floor_solo`]/[`Self::floor_batched`].
/// The fixed constants remain as the warmup fallback (until every required
/// timer has [`SpecCostModel::WARMUP`] samples), the clamp anchors, and the
/// `force` pin. Tier/rung awareness comes free via measurement — the fast
/// tier's cheaper verify lowers V and with it the floor automatically.
///
/// Purely a cost policy: suppression changes WHEN the target runs a verify,
/// never what is committed — plain steps emit the target's own argmax, which
/// is exactly what the lossless verify would have committed.
///
/// `TRITIUM_SPEC_ADAPTIVE`: `1`/unset = on (default), `0` = off (kill
/// switch), `force` = classify every verify as collapsed (deterministic
/// forced-collapse gates; not a production setting).
///
/// `TRITIUM_SPEC_COST_FLOORS`: `1`/unset = derive the floors from the
/// measured cost model once warm (default), `0` = pin the fixed floors
/// (the cost-model kill switch and the fixed-vs-dynamic A/B arm).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpecGovernor {
    /// Kill switch (`TRITIUM_SPEC_ADAPTIVE=0`): never suppress.
    Off,
    /// The adaptive lever.
    On {
        /// `TRITIUM_SPEC_ADAPTIVE=force`: every verify counts as collapsed.
        force: bool,
        /// Drafting currently suppressed (plain steps + periodic probes).
        suppressed: bool,
        /// Consecutive verifies whose τ estimate sat below the floor.
        low_streak: u32,
        /// Committed plain tokens since the last verify while suppressed.
        since_probe: usize,
    },
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl SpecGovernor {
    /// SOLO fixed floor — the warmup fallback and the `force` pin (see
    /// [`Self::floor_solo`]). τ < 1.5 was below every measured tier's solo
    /// breakeven in the 2026-08-08 sweep; once [`SPEC_COST`] is warm the
    /// measured `(V + k·d)/P` replaces it.
    pub(crate) const TAU_FLOOR_SOLO: f64 = 1.5;
    /// BATCHED (multi-slot) floor: ride-along drafting is nearly free there
    /// — the batched drafter step and the shared verify run for the pool
    /// anyway, so a slot at τ ≈ 1.4 still nets more tokens per round than a
    /// 1-node plain step (measured 2026-08-09: a 1.5 floor at N=4 cost 32%
    /// aggregate tok/s on the healthy fixture). A slot only drops out on
    /// TRUE collapse (drafting buys ~nothing); the real batched win is the
    /// whole-pool-suppressed Lockstep state, which skips the drafter and
    /// verify entirely. Once [`SPEC_COST`] is warm, [`Self::floor_batched`]
    /// can only RAISE this (1.1 is the derived floor's lower clamp).
    pub(crate) const TAU_FLOOR_BATCHED: f64 = 1.1;
    /// Consecutive low-τ verifies before suppression engages — one noisy
    /// cycle (or the optimistic-start decay) must not kill a healthy drafter.
    const ENTRY_STREAK: u32 = 4;
    /// Committed plain tokens between probe verifies while suppressed. The
    /// verify + 4-token draft is <2% of plain — but the probe's DOMINANT cost
    /// at long ctx is the drafter re-sync: 64 plain tokens stale the drafter
    /// KV past the gap the reconcile tolerates, so essentially every probe
    /// pays a full drafter re-prefill (~ctx-linear). The measured long-ctx
    /// wins (2.42x exact / 1.22x fast+f16, probes included) already carry
    /// that cost at ctx~4K; shrinking PROBE_PERIOD or growing ctx changes
    /// the economics — re-measure before touching either (review 4190673
    /// F2).
    const PROBE_PERIOD: usize = 64;
    /// Probe draft length: long enough that a fully-accepted probe moves
    /// the acceptance EWMA materially, short enough to stay cheap.
    ///
    /// Recovery arithmetic (DraftPolicy LAMBDA 0.95: after n consecutive
    /// full-accept k=4 probes from acc₀ = 0.30, acc_n = 1 − 0.95⁴ⁿ·0.70,
    /// τ from `tau_estimate` at the policy's own k):
    /// - 2 probes → acc ≈ 0.54, k=2, τ ≈ 1.82 — clears the FIXED 1.5
    ///   fallback, but NOT a warm measured floor (fast tier ≈ 2.15–2.95
    ///   at ctx 1536, exact tier at the 3.0 clamp).
    /// - 4 probes → acc ≈ 0.69, k=3, τ ≈ 2.50 — clears the warm fast-tier
    ///   floor.
    /// - 5 probes → acc ≈ 0.75, k=4, τ ≈ 3.05 — clears even the 3.0 clamp,
    ///   so recovery is reachable on the exact tier too; that it takes a
    ///   ~5-probe (~320 committed tokens) streak of full accepts there
    ///   matches the exact tier's honest economics — its measured verify
    ///   really is that dear, and a floor pinned at the clamp demands a
    ///   drafter that earns it.
    ///
    /// A partial accept decays acc by ONE LAMBDA factor per failed walk,
    /// so mixed probes ratchet slower, not never. The floor itself also
    /// drifts while suppressed: V keeps folding the cheap k≤4 probe
    /// verifies and P every plain step (d is FROZEN at its steady-state
    /// value — probe walls go to `draft_resync`, review aec4c78 F1), so a
    /// warm floor decays toward the small-tree breakeven between probes,
    /// and post-recovery cheap verifies keep refreshing V downward.
    const PROBE_K: usize = 4;

    /// Fresh governor in the ON state (the default; also the unit-test seam).
    pub(crate) const fn new_on(force: bool) -> Self {
        Self::On {
            force,
            suppressed: false,
            low_streak: 0,
            since_probe: 0,
        }
    }

    /// `TRITIUM_SPEC_ADAPTIVE`: unset/`1` = on, `0` = off, `force` = forced
    /// collapse (gates). Loud-reject on anything else (the TRITIUM_DRAFT_K
    /// pattern). Also validates `TRITIUM_SPEC_COST_FLOORS` (unset/`1` =
    /// derived floors, `0` = pin the fixed floors) up front so a typo fails
    /// the request loudly instead of silently steering the floors.
    pub(crate) fn from_env() -> Result<Self, GenError> {
        match std::env::var("TRITIUM_SPEC_COST_FLOORS") {
            Err(std::env::VarError::NotPresent) => {}
            Ok(v) if v == "1" || v == "0" => {}
            Ok(v) => {
                return Err(GenError::Backend(format!(
                    "TRITIUM_SPEC_COST_FLOORS={v:?} — use 1 (default) or 0"
                )));
            }
            Err(e) => {
                return Err(GenError::Backend(format!("TRITIUM_SPEC_COST_FLOORS: {e}")));
            }
        }
        match std::env::var("TRITIUM_SPEC_ADAPTIVE") {
            Err(std::env::VarError::NotPresent) => Ok(Self::new_on(false)),
            Ok(v) if v == "1" => Ok(Self::new_on(false)),
            Ok(v) if v == "0" => Ok(Self::Off),
            Ok(v) if v == "force" => Ok(Self::new_on(true)),
            Ok(v) => Err(GenError::Backend(format!(
                "TRITIUM_SPEC_ADAPTIVE={v:?} — use 1 (default), 0, or force"
            ))),
            Err(e) => Err(GenError::Backend(format!("TRITIUM_SPEC_ADAPTIVE: {e}"))),
        }
    }

    /// Draft-length cap for the next cycle: `None` = draft normally (the
    /// policy's length), `Some(0)` = take a plain step, `Some(k)` = probe
    /// with a k-token chain.
    pub(crate) fn draft_cap(&self) -> Option<usize> {
        match self {
            Self::Off
            | Self::On {
                suppressed: false, ..
            } => None,
            Self::On { since_probe, .. } if *since_probe >= Self::PROBE_PERIOD => {
                Some(Self::PROBE_K)
            }
            Self::On { .. } => Some(0),
        }
    }

    /// The SOLO breakeven for the next [`Self::on_verify`], at the current
    /// draft length `k` (the policy's `len()` — the k the next cycle would
    /// actually pay for): the measured `(V + k·d)/P` from [`SPEC_COST`] once
    /// warm, else the fixed [`Self::TAU_FLOOR_SOLO`]. `force` PINS the fixed
    /// floor — the forced-collapse gates must stay deterministic, never a
    /// function of this box's wall clock. Publishes the applied floor to
    /// [`SPEC_FLOOR_SOLO`] (`tritium_spec_floor{path="solo"}`).
    /// `TRITIUM_SPEC_COST_FLOORS=0` (validated by [`Self::from_env`]) pins
    /// the fixed floors — the cost-model kill switch and the fixed-vs-dynamic
    /// bench arm. Read per verify: sub-µs against a ms-scale verify.
    fn cost_floors_enabled() -> bool {
        std::env::var("TRITIUM_SPEC_COST_FLOORS").as_deref() != Ok("0")
    }

    pub(crate) fn floor_solo(&self, k: usize) -> f64 {
        let dynamic = !matches!(self, Self::On { force: true, .. }) && Self::cost_floors_enabled();
        let floor = if dynamic {
            SPEC_COST.solo_floor(k).unwrap_or(Self::TAU_FLOOR_SOLO)
        } else {
            Self::TAU_FLOOR_SOLO
        };
        SPEC_FLOOR_SOLO.store(floor.to_bits(), Ordering::Relaxed);
        floor
    }

    /// The BATCHED per-slot breakeven for the next [`Self::on_verify`]:
    /// the measured marginal-round floor from [`SPEC_COST`] once warm (see
    /// [`SpecCostModel::batched_floor`] — it can only RAISE the fixed 1.1),
    /// else the fixed [`Self::TAU_FLOOR_BATCHED`]. `force` pins the fixed
    /// floor (deterministic gates). Publishes the applied floor to
    /// [`SPEC_FLOOR_BATCHED`] (`tritium_spec_floor{path="batched"}`).
    pub(crate) fn floor_batched(&self, n_live: usize) -> f64 {
        let dynamic = !matches!(self, Self::On { force: true, .. }) && Self::cost_floors_enabled();
        let floor = if dynamic {
            SPEC_COST
                .batched_floor(n_live)
                .unwrap_or(Self::TAU_FLOOR_BATCHED)
        } else {
            Self::TAU_FLOOR_BATCHED
        };
        SPEC_FLOOR_BATCHED.store(floor.to_bits(), Ordering::Relaxed);
        floor
    }

    /// Fold `n` committed tokens from suppressed plain steps: advances the
    /// probe clock and the observability counter. No-op unless suppressed
    /// (call it from every plain branch unconditionally).
    pub(crate) fn on_plain_commit(&mut self, n: usize) {
        if let Self::On {
            suppressed: true,
            since_probe,
            ..
        } = self
        {
            *since_probe += n;
            SPEC_SUPPRESSED_PLAIN.fetch_add(n as u64, Ordering::Relaxed);
        }
    }

    /// Fold one verify cycle's outcome, AFTER `policy.update`. `floor` is
    /// the caller's breakeven ([`Self::TAU_FLOOR_SOLO`] in the solo loops,
    /// [`Self::TAU_FLOOR_BATCHED`] per slot in the multi-slot pool). Entry
    /// needs [`Self::ENTRY_STREAK`] consecutive collapsed verifies; a probe
    /// verify that clears the floor lifts suppression immediately
    /// (re-suppression costs another streak — that asymmetry is the
    /// hysteresis).
    pub(crate) fn on_verify(&mut self, policy: &DraftPolicy, floor: f64) {
        let Self::On {
            force,
            suppressed,
            low_streak,
            since_probe,
        } = self
        else {
            return;
        };
        let Some(tau) = policy.tau_estimate() else {
            return; // legacy policy: governor inert (A/B baseline)
        };
        let low = *force || tau < floor;
        if *suppressed {
            if low {
                *since_probe = 0; // failed probe: restart the probe clock
            } else {
                *suppressed = false;
                *low_streak = 0;
            }
        } else if low {
            *low_streak += 1;
            if *low_streak >= Self::ENTRY_STREAK {
                *suppressed = true;
                *since_probe = 0;
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "tritium-serve: adaptive spec suppression engaged — drafter \
                         acceptance collapsed below breakeven (τ < {floor}); plain decode \
                         with a {}-token probe every {} tokens \
                         (TRITIUM_SPEC_ADAPTIVE=0 disables; logged once, see \
                         tritium_spec_suppressed_plain_total)",
                        Self::PROBE_K,
                        Self::PROBE_PERIOD,
                    );
                });
            }
        } else {
            *low_streak = 0;
        }
    }
}

/// Draft up to `max_draft` tokens with `draft` (greedy), starting after
/// `history` (whose last element is the pending token). Syncs the draft's KV
/// to `history` first by re-feeding the gap — see `draft_pos`/`draft_fed` on
/// [`RunnerGenerator`] for why that is always correct. Stops early at the
/// draft's own EOS. Returns `[]` on any draft error (caller falls back to a
/// plain step; drafting must never break generation).
///
/// Free function (not a method) so the batched worker's I0 solo-spec path
/// (ADR 0032 L3 I0, `batch.rs`) can share the exact reconcile + chain logic
/// with [`RunnerGenerator::model_draft`] — this is the single source of truth.
#[cfg(feature = "cuda")]
pub(crate) fn draft_greedy_tokens(
    draft: &mut tritium_nn::ModelRunner,
    draft_fed: &mut Vec<u32>,
    draft_pos: &mut usize,
    eos: u32,
    history: &[u32],
    max_draft: usize,
    chain: bool,
) -> Vec<u32> {
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
    for (i, &fed) in draft_fed.iter().enumerate() {
        if history.get(*draft_pos + i) != Some(&fed) {
            clean = false;
            break;
        }
    }
    if clean {
        *draft_pos = (*draft_pos + draft_fed.len()).min(p);
    } else {
        draft.reset();
        *draft_pos = 0;
    }
    draft_fed.clear();
    // Forward-contiguous sync: feed the history the draft hasn't seen,
    // EXCLUDING pending (fed by the loop below).
    if *draft_pos < p {
        let gap: Vec<u32> = history[*draft_pos..p].to_vec();
        let positions: Vec<usize> = (*draft_pos..p).collect();
        if draft.forward(&gap, &positions).is_err() {
            draft.reset();
            *draft_pos = 0; // full resync next time
            return Vec::new();
        }
        *draft_pos = p;
    }
    let mut out = Vec::with_capacity(max_draft);
    let mut tok = history[p];
    // L1' fastest path (ADR 0032): the whole k-token draft as ONE chained
    // device-side loop — a single host round-trip instead of one per
    // token (the measured ~1.2 ms/token host cost that held spec decode
    // at parity). Drafts are bit-identical to the per-step path (gated by
    // cuda_draft_chain_matches_per_step). Ok(None) = no resident decoder;
    // Err = state untouched (cache_len only advances on success) — both
    // fall through to the per-step ladder below.
    let chained = if chain {
        draft.decode_greedy_chain(tok, p, max_draft, eos)
    } else {
        Ok(None) // TRITIUM_DRAFT_CHAIN=0: per-step ladder (A/B + kill switch)
    };
    if let Ok(Some(ids)) = chained
        && !ids.is_empty()
    {
        // (Empty ids — only reachable from poisoned all-NaN logits —
        // falls through to the ladder without recording a feed.)
        // The chain fed [tok, ids[0..len-1]] (the last id — possibly
        // EOS — was drafted, never fed); record exactly those feeds.
        draft_fed.push(tok);
        for (i, &id) in ids.iter().enumerate() {
            out.push(id);
            if i + 1 < ids.len() {
                draft_fed.push(id);
            }
        }
        return out;
    }
    // Ok(None) (no resident decoder) or Err (state untouched): fall
    // through to the per-step ladder.
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
                draft_fed.push(tok);
                let Some(next) = tritium_nn::sample_greedy(&logits) else {
                    break;
                };
                out.push(next);
                if next == eos {
                    break;
                }
                tok = next;
                continue;
            }
            Err(_) => break,
        };
        draft_fed.push(tok);
        out.push(next);
        if next == eos {
            break; // the draft believes the turn ends here
        }
        tok = next;
    }
    out
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
    /// Draft up to `max_draft` tokens with the attached draft model (greedy).
    /// Thin wrapper over [`draft_greedy_tokens`] — the shared reconcile +
    /// chain logic (also used by the batched worker's I0 solo-spec path).
    fn model_draft(&mut self, history: &[u32], max_draft: usize, chain: bool) -> Vec<u32> {
        let Some(draft) = self.draft.as_mut() else {
            return Vec::new();
        };
        draft_greedy_tokens(
            draft,
            &mut self.draft_fed,
            &mut self.draft_pos,
            self.eos,
            history,
            max_draft,
            chain,
        )
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
        // Adaptive spec on/off (see SpecGovernor — the long-ctx τ-collapse
        // lever). Greedy-path v1; the sampled twin is the measured follow-up.
        let mut governor = SpecGovernor::from_env()?;
        let chain = draft_chain_from_env()?;
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
            // Governor cap on top of the policy length: Some(0) = suppressed
            // plain step, Some(k) = probe chain, None = draft normally.
            let cap = governor.draft_cap();
            // A probe cycle's drafter wall is dominated by the ~ctx-linear
            // re-prefill (PROBE_PERIOD stale plain tokens force a reset) —
            // routed to the resync EWMA below, never the floor's d.
            let is_probe = matches!(cap, Some(k) if k > 0);
            let want = match cap {
                Some(k) => k,
                None => policy.len(),
            };
            let max_draft = want.min(kv_room).min(budget.saturating_sub(1));
            let t_d = std::time::Instant::now();
            let drafts = if max_draft == 0 {
                Vec::new() // suppressed (or clamped): no drafter work at all
            } else if self.draft.is_some() {
                self.model_draft(&history, max_draft, chain)
            } else {
                Self::lookup_draft(&history, max_draft)
            };
            // Cost-model d: drafter wall per drafted token (empty results —
            // bails, no match — carry no per-token denominator; skipped).
            // Probe cycles feed draft_resync (telemetry), steady-state
            // cycles feed the floor's draft_tok — see record_draft.
            if !drafts.is_empty() {
                SPEC_COST.record_draft(
                    t_d.elapsed().as_secs_f64() * 1e6 / drafts.len() as f64,
                    is_probe,
                );
            }

            if drafts.is_empty() {
                // Plain M=1 graph step (faster than a 1-node tree).
                let t0 = std::time::Instant::now();
                let pos = history.len() - 1;
                let logits = self
                    .runner
                    .forward(&[pending], &[pos])
                    .map_err(|e| GenError::Backend(e.to_string()))?;
                n_plain += 1;
                let el = t0.elapsed();
                t_plain += el;
                SPEC_COST.plain.record(el.as_secs_f64() * 1e6); // cost-model P
                governor.on_plain_commit(1); // no-op unless suppressed
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
            let el = t0.elapsed();
            t_verify += el;
            SPEC_COST.verify.record(el.as_secs_f64() * 1e6); // cost-model V
            // committed = accepted drafts + the final token, so accepted
            // drafts = committed.len() - 1 (saturating: an empty `committed`
            // violates tree_verify_greedy's contract and is caught below, but
            // a wrapped usize here would spin the EWMA fold first).
            policy.update(drafts.len(), committed.len().saturating_sub(1));
            let floor = governor.floor_solo(policy.len());
            governor.on_verify(&policy, floor);
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
        let chain = draft_chain_from_env()?;
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
                self.model_draft(&history, max_draft, chain)
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
            for (child, &d) in tokens.iter().enumerate().skip(1) {
                let node = child - 1;
                let row = &logits_all[node * vocab..(node + 1) * vocab];
                let (idx, probs) = Self::truncated(row, &req.sampling)
                    .ok_or_else(|| GenError::Backend("sampler produced no token".into()))?;
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
            self.runner
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
                })
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

    #[test]
    fn tau_estimate_tracks_acceptance() {
        let tau = |a: f64| {
            super::DraftPolicy::Adaptive { acc: a }
                .tau_estimate()
                .expect("adaptive has a tau")
        };
        // Healthy drafter: well above the governor floor.
        assert!(tau(0.8) > 2.5, "tau(0.8) = {}", tau(0.8));
        // The sweep's collapsed band (τ ≈ 1.2–1.35): below the floor.
        assert!(tau(0.30) < super::SpecGovernor::TAU_FLOOR_SOLO);
        assert!(tau(0.20) < super::SpecGovernor::TAU_FLOOR_SOLO);
        // Optimistic start (acc = 0.75): must NOT read as collapsed.
        assert!(tau(0.75) > super::SpecGovernor::TAU_FLOOR_SOLO);
        // Mid-health band (the N=4 fixture, acc ≈ 0.45): below the solo
        // floor but ABOVE the batched floor — a ride-along slot keeps
        // drafting (suppressing it cost 32% aggregate, 2026-08-09 bench).
        assert!(tau(0.45) < super::SpecGovernor::TAU_FLOOR_SOLO);
        assert!(tau(0.45) > super::SpecGovernor::TAU_FLOOR_BATCHED);
        // True collapse: below both floors.
        assert!(tau(0.05) < super::SpecGovernor::TAU_FLOOR_BATCHED);
        // Legacy policy carries no estimate — governor inert.
        assert!(
            super::DraftPolicy::Legacy { len: 6 }
                .tau_estimate()
                .is_none(),
            "legacy must be None"
        );
    }

    #[test]
    fn governor_enters_after_streak_and_stays_dormant_when_healthy() {
        let collapsed = super::DraftPolicy::Adaptive { acc: 0.25 };
        let healthy = super::DraftPolicy::Adaptive { acc: 0.85 };
        let mut g = super::SpecGovernor::new_on(false);
        // Healthy verifies never suppress.
        for _ in 0..100 {
            g.on_verify(&healthy, super::SpecGovernor::TAU_FLOOR_SOLO);
            assert_eq!(g.draft_cap(), None);
        }
        // One low verify resets on the next healthy one (streak, not latch).
        g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
        g.on_verify(&healthy, super::SpecGovernor::TAU_FLOOR_SOLO);
        for _ in 0..super::SpecGovernor::ENTRY_STREAK - 1 {
            g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
            assert_eq!(g.draft_cap(), None, "must not fire before the streak");
        }
        g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
        assert_eq!(g.draft_cap(), Some(0), "streak complete: suppressed");
    }

    #[test]
    fn governor_probe_cadence_and_recovery() {
        let collapsed = super::DraftPolicy::Adaptive { acc: 0.25 };
        let healthy = super::DraftPolicy::Adaptive { acc: 0.85 };
        let mut g = super::SpecGovernor::new_on(false);
        for _ in 0..super::SpecGovernor::ENTRY_STREAK {
            g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
        }
        assert_eq!(g.draft_cap(), Some(0));
        // Plain commits advance the probe clock; the probe fires at PERIOD.
        g.on_plain_commit(super::SpecGovernor::PROBE_PERIOD - 1);
        assert_eq!(g.draft_cap(), Some(0));
        g.on_plain_commit(1);
        assert_eq!(g.draft_cap(), Some(super::SpecGovernor::PROBE_K));
        // Failed probe: back to plain, clock restarted.
        g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
        assert_eq!(g.draft_cap(), Some(0));
        g.on_plain_commit(super::SpecGovernor::PROBE_PERIOD);
        assert_eq!(g.draft_cap(), Some(super::SpecGovernor::PROBE_K));
        // Recovered probe: suppression lifts immediately.
        g.on_verify(&healthy, super::SpecGovernor::TAU_FLOOR_SOLO);
        assert_eq!(g.draft_cap(), None);
        // Re-suppression needs a fresh full streak (hysteresis).
        g.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
        assert_eq!(g.draft_cap(), None);
    }

    #[test]
    fn governor_off_and_legacy_are_inert() {
        let collapsed = super::DraftPolicy::Adaptive { acc: 0.10 };
        let mut off = super::SpecGovernor::Off;
        for _ in 0..20 {
            off.on_verify(&collapsed, super::SpecGovernor::TAU_FLOOR_SOLO);
            off.on_plain_commit(1);
            assert_eq!(off.draft_cap(), None);
        }
        // Legacy policy: no tau estimate, ON governor never moves.
        let mut g = super::SpecGovernor::new_on(true); // even forced
        for _ in 0..20 {
            g.on_verify(
                &super::DraftPolicy::Legacy { len: 6 },
                super::SpecGovernor::TAU_FLOOR_SOLO,
            );
            assert_eq!(g.draft_cap(), None);
        }
    }

    #[test]
    fn governor_force_suppresses_despite_healthy_acceptance() {
        let healthy = super::DraftPolicy::Adaptive { acc: 0.9 };
        let mut g = super::SpecGovernor::new_on(true);
        for _ in 0..super::SpecGovernor::ENTRY_STREAK {
            g.on_verify(&healthy, super::SpecGovernor::TAU_FLOOR_SOLO);
        }
        assert_eq!(g.draft_cap(), Some(0), "force: collapsed regardless of acc");
        g.on_plain_commit(super::SpecGovernor::PROBE_PERIOD);
        assert_eq!(g.draft_cap(), Some(super::SpecGovernor::PROBE_K));
        g.on_verify(&healthy, super::SpecGovernor::TAU_FLOOR_SOLO); // probe verifies stay classified collapsed
        assert_eq!(g.draft_cap(), Some(0));
    }

    // ───────────── cost-model floors (SpecCostModel + Ewma) ─────────────

    /// First sample seeds the mean; later samples fold with ALPHA and the
    /// count rises monotonically toward the new level.
    #[test]
    fn ewma_first_sample_then_converges() {
        let e = super::Ewma::new();
        assert_eq!(e.mean_us(), None);
        assert_eq!(e.samples(), 0);
        e.record(100.0);
        assert_eq!(e.mean_us(), Some(100.0));
        assert_eq!(e.samples(), 1);
        let mut prev = 100.0;
        for _ in 0..20 {
            e.record(200.0);
            let m = e.mean_us().expect("recorded");
            assert!(m > prev && m <= 200.0, "EWMA must move toward 200: {m}");
            prev = m;
        }
        assert!(
            prev > 190.0,
            "20 folds at ALPHA=0.2 converge near 200: {prev}"
        );
        assert_eq!(e.samples(), 21);
        // Defensive drops: non-finite/negative samples change nothing.
        e.record(f64::NAN);
        e.record(-1.0);
        assert_eq!(e.samples(), 21);
    }

    /// Warm each Ewma of a LOCAL model with `n` identical samples.
    fn warm(e: &super::Ewma, us: f64, n: u64) {
        for _ in 0..n {
            e.record(us);
        }
    }

    /// The solo derivation `(V + k·d)/P` and its [1.05, 3.0] clamp.
    #[test]
    fn cost_model_solo_floor_math_and_clamps() {
        let m = super::SpecCostModel::new();
        warm(&m.verify, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.plain, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.draft_tok, 750.0, super::SpecCostModel::WARMUP);
        // (4000 + 4·750)/4000 = 1.75 — the ADR 0032 numbers.
        let f = m.solo_floor(4).expect("warm");
        assert!((f - 1.75).abs() < 1e-12, "solo floor {f}");
        // k dependence: a longer draft raises the breakeven.
        assert!(m.solo_floor(8).expect("warm") > f);
        // Upper clamp: an expensive verify cannot push the floor past 3.0.
        let hi = super::SpecCostModel::new();
        warm(&hi.verify, 100_000.0, super::SpecCostModel::WARMUP);
        warm(&hi.plain, 1000.0, super::SpecCostModel::WARMUP);
        warm(&hi.draft_tok, 10.0, super::SpecCostModel::WARMUP);
        assert_eq!(hi.solo_floor(4), Some(3.0));
        // Lower clamp: a nearly-free verify still demands τ > 1.05.
        let lo = super::SpecCostModel::new();
        warm(&lo.verify, 10.0, super::SpecCostModel::WARMUP);
        warm(&lo.plain, 4000.0, super::SpecCostModel::WARMUP);
        warm(&lo.draft_tok, 1.0, super::SpecCostModel::WARMUP);
        assert_eq!(lo.solo_floor(1), Some(1.05));
    }

    /// The batched marginal derivation `1 + max(0, Vr−P)/(n·P)` and its
    /// [1.1, 3.0] clamp — the model can only RAISE the 1.1 floor.
    #[test]
    fn cost_model_batched_floor_marginal() {
        let m = super::SpecCostModel::new();
        warm(&m.verify_round, 2600.0, super::SpecCostModel::WARMUP);
        warm(&m.lockstep, 1000.0, super::SpecCostModel::WARMUP);
        // 1 + 1600/(4·1000) = 1.4.
        let f = m.batched_floor(4).expect("warm");
        assert!((f - 1.4).abs() < 1e-12, "batched floor {f}");
        // More live slots dilute the premium.
        assert!(m.batched_floor(8).expect("warm") < f);
        // A round no dearer than lockstep clamps to the fixed 1.1 —
        // ride-along slots keep drafting exactly as before.
        let cheap = super::SpecCostModel::new();
        warm(&cheap.verify_round, 900.0, super::SpecCostModel::WARMUP);
        warm(&cheap.lockstep, 1000.0, super::SpecCostModel::WARMUP);
        assert_eq!(cheap.batched_floor(4), Some(1.1));
        // Upper clamp.
        let hi = super::SpecCostModel::new();
        warm(&hi.verify_round, 1_000_000.0, super::SpecCostModel::WARMUP);
        warm(&hi.lockstep, 1000.0, super::SpecCostModel::WARMUP);
        assert_eq!(hi.batched_floor(1), Some(3.0));
        // No live slots: no derivation.
        assert_eq!(m.batched_floor(0), None);
    }

    /// Until EVERY required timer has WARMUP samples the derived floor is
    /// None — callers fall back to the fixed constants.
    #[test]
    fn cost_model_warmup_falls_back() {
        let m = super::SpecCostModel::new();
        warm(&m.verify, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.plain, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.draft_tok, 750.0, super::SpecCostModel::WARMUP - 1);
        assert_eq!(m.solo_floor(4), None, "one cold timer keeps the fallback");
        m.draft_tok.record(750.0); // the WARMUP-th sample
        assert!(m.solo_floor(4).is_some());
        let b = super::SpecCostModel::new();
        warm(&b.verify_round, 2000.0, super::SpecCostModel::WARMUP - 1);
        warm(&b.lockstep, 1000.0, super::SpecCostModel::WARMUP);
        assert_eq!(b.batched_floor(4), None);
        b.verify_round.record(2000.0);
        assert!(b.batched_floor(4).is_some());
    }

    /// F1 (review aec4c78): probe-time drafter walls must NOT feed the
    /// floor's d — they land in the resync EWMA (telemetry). Otherwise the
    /// only d samples while suppressed are ~ctx-linear re-prefills, the
    /// floor clamps high, and a recovered drafter can never lift
    /// suppression.
    #[test]
    fn probe_draft_walls_feed_resync_not_the_floors_d() {
        let m = super::SpecCostModel::new();
        warm(&m.verify, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.plain, 4000.0, super::SpecCostModel::WARMUP);
        // Steady-state drafting measured d = 500 µs/tok before collapse.
        warm(&m.draft_tok, 500.0, super::SpecCostModel::WARMUP);
        let d0 = m.draft_tok.mean_us().expect("warm");
        let floor0 = m.solo_floor(2).expect("warm");
        // Suppressed phase: every probe pays a ctx-linear re-prefill.
        // Routed as probes, the walls move ONLY the resync EWMA.
        for _ in 0..32 {
            m.record_draft(20_000.0, true);
        }
        assert_eq!(
            m.draft_tok.mean_us(),
            Some(d0),
            "probe walls must not move the floor's d"
        );
        assert_eq!(m.solo_floor(2), Some(floor0), "floor unmoved by probes");
        assert_eq!(m.draft_resync.samples(), 32, "resync EWMA takes them");
        assert!(m.draft_resync.mean_us().expect("recorded") > 10_000.0);
        // Steady-state samples still feed d.
        m.record_draft(500.0, false);
        assert_eq!(m.draft_tok.samples(), super::SpecCostModel::WARMUP + 1);
        assert_eq!(m.draft_resync.samples(), 32);
    }

    /// The recovery scenario the F1 split exists for: with steady-state d
    /// the warm floor sits where a recovered drafter's τ clears it and a
    /// suppressed governor lifts; with the pre-fix accounting (probe
    /// re-prefills folded into d) the derived floor clamps to 3.0 and the
    /// SAME drafter latches suppressed forever.
    #[test]
    fn recovered_drafter_clears_steady_floor_not_resync_polluted_floor() {
        // Fast-tier-shaped costs: V 8.6 ms, P 4 ms, steady d 0.3 ms.
        let m = super::SpecCostModel::new();
        warm(&m.verify, 8600.0, super::SpecCostModel::WARMUP);
        warm(&m.plain, 4000.0, super::SpecCostModel::WARMUP);
        warm(&m.draft_tok, 300.0, super::SpecCostModel::WARMUP);
        // Probe re-prefills during suppression: resync only.
        for _ in 0..32 {
            m.record_draft(30_000.0, true);
        }
        // Recovered drafter: acc after ~4 consecutive full-accept k=4
        // probes from 0.30 (see PROBE_K's recovery arithmetic).
        let recovered = super::DraftPolicy::Adaptive { acc: 0.69 };
        let tau = recovered.tau_estimate().expect("adaptive");
        let k = recovered.len();
        let floor = m.solo_floor(k).expect("warm");
        assert!(
            floor > super::SpecGovernor::TAU_FLOOR_SOLO,
            "a warm fast-tier floor sits above the fixed fallback: {floor}"
        );
        assert!(
            tau >= floor,
            "recovered τ {tau} must clear the steady-state floor {floor}"
        );
        // A suppressed governor lifts on that probe verify.
        let collapsed = super::DraftPolicy::Adaptive { acc: 0.25 };
        let mut g = super::SpecGovernor::new_on(false);
        for _ in 0..super::SpecGovernor::ENTRY_STREAK {
            g.on_verify(&collapsed, floor);
        }
        g.on_plain_commit(super::SpecGovernor::PROBE_PERIOD);
        assert_eq!(g.draft_cap(), Some(super::SpecGovernor::PROBE_K));
        g.on_verify(&recovered, floor);
        assert_eq!(g.draft_cap(), None, "suppression lifts");
        // Counterfactual pre-fix accounting: resync-dominated d clamps the
        // floor to 3.0 — the same recovered drafter can never clear it.
        let polluted = super::SpecCostModel::new();
        warm(&polluted.verify, 8600.0, super::SpecCostModel::WARMUP);
        warm(&polluted.plain, 4000.0, super::SpecCostModel::WARMUP);
        warm(&polluted.draft_tok, 30_000.0, super::SpecCostModel::WARMUP);
        let bad = polluted.solo_floor(k).expect("warm");
        assert_eq!(bad, super::SpecCostModel::SOLO_CLAMP.1, "clamped high");
        assert!(tau < bad, "resync-polluted d latches suppression: τ {tau}");
    }

    /// `force` pins the FIXED floors regardless of the process-wide cost
    /// model — the forced-collapse e2e gates stay deterministic on any box.
    #[test]
    fn governor_force_pins_fixed_floors() {
        let g = super::SpecGovernor::new_on(true);
        assert_eq!(g.floor_solo(40), super::SpecGovernor::TAU_FLOOR_SOLO);
        assert_eq!(g.floor_batched(4), super::SpecGovernor::TAU_FLOOR_BATCHED);
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
