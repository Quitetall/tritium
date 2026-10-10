//! Continuous batching, phase 1 (ADR 0020-era plan, zero new kernels).
//!
//! A fixed pool of `slots` sequences shares ONE `BatchKv` whose M=N decode
//! graph is captured once. Requests are admitted into free slots by running
//! the prompt through the OPTIMIZED single-sequence prefill and adopting the
//! resulting KV rows into the slot ([`CudaDecodeModel::copy_kv_into_batch_row`]);
//! every decode step advances all slots in lockstep (free slots are marked
//! DEAD — `BatchKv::set_live(row, false)` — so the kernels skip them: no KV
//! writes, no attention; their pad-token outputs are ignored). Per-slot
//! sampling runs on the host against each
//! request's own parameters, reusing the plain samplers' truncated
//! distributions — the same per-row math the parity gates pin to the
//! single-sequence path.
//!
//! **Chunked prefill (batching P2, C1)**: admission no longer stalls the batch
//! for the whole prompt. The prompt runs through the single-sequence prefill in
//! fixed chunks (`TRITIUM_PREFILL_CHUNK`, default 128), one chunk per loop
//! iteration, interleaved with the lockstep decode steps — a live slot's
//! inter-token gap during admission is bounded by one chunk + one step instead
//! of the full prompt. At most ONE admission is mid-prefill at a time (the
//! chunks accumulate in the runner's one single-sequence KV, which the adoption
//! copy reads at completion); its slot row is implicitly reserved because
//! admission only runs when no prefill is in flight. Chunking is bit-exact by
//! construction: `prefill` is bit-identical per row to the sequential step
//! loop, so any chunking of it is too — the first sampled token still equals
//! the single-sequence path's exactly. Deliberate trade: while a prefill is in
//! flight the queue is not polled, so even instantly-rejectable jobs
//! (validation failures, tree ops) wait out the remaining chunks — bounded by
//! one prompt's chunked prefill.
//!
//! **Tree-session coexistence (C4)**: the BASTION tree endpoints work with
//! `--batch-slots > 1`. A session open is a prompt prefill through the SAME
//! chunk machine admissions use (interleaved, never stalling live slots);
//! verifies run inline between batch steps as bounded ops. The session owns
//! the single-sequence KV with the single-worker contract verbatim: any chat
//! ADMISSION resets the runner and closes the session (the next verify gets
//! 409 Conflict; the drafter re-opens). Recorded follow-up (not v1): sessions
//! that SURVIVE admissions by leasing a slot's paged region — requires the
//! tree kernel stack parametrized over KV regions.
//!
//! **Paged KV (batching P2, C3, ADR 0025)**: with `--kv-pool-tokens N`, the
//! per-slot dense arenas are replaced by a shared page pool. Admission
//! reserves `prompt + max_tokens` up front (never outgrown — the v1
//! no-eviction policy); a full pool parks the job until a retirement frees
//! pages (FIFO, retried before new work); a request that can never fit is a
//! loud error. Every retirement/abandonment path releases its row's pages.
//! Paging is bit-exact by construction (gated: paged == dense).
//!
//! **Solo speculative decoding (ADR 0032 L3 I0)**: with `--draft-model`, a
//! greedy request that arrives at an EMPTY pool decodes speculatively on the
//! single-sequence KV (the same draft→`tree_verify_greedy`→commit cycle as
//! the single worker — lossless: only target argmaxes are committed) instead
//! of burning a lockstep slot alone. The v1 contract is **spec-when-solo,
//! migrate-on-admission**: the moment ANY admission-type job arrives (chat
//! admit or tree-session open), the spec sequence is migrated into a batch
//! slot — its full history becomes a continuation admission's prompt (queued
//! AHEAD of the incoming job), the drafter is reset, and everyone proceeds
//! under the normal lockstep contract. Migration re-emits nothing: spec
//! emitted every committed token including history's last, and the
//! continuation prefill's argmax after the full history IS the next
//! unemitted token. Spec and tree sessions are mutually exclusive both ways
//! (both own the single-sequence KV — the C4 serialized-ownership contract:
//! admissions clobber the prefill staging area).
//!
//! **Multi-slot speculative decoding (ADR 0032 L3, the serve wiring of the
//! I1–I4 engine rungs)**: with `--draft-model`, whenever EVERY live sequence
//! is greedy + logprob-free and no tree session is open (the v1 all-or-nothing
//! pool), the worker replaces the lockstep step with a batched spec ROUND:
//! [`draft_batch`](tritium_nn::ModelRunner::draft_batch) drafts all live slots
//! in `k` lockstep drafter steps (the I1 host-fed path — `TRITIUM_DRAFT_CHAIN`
//! is single-sequence-only and does not apply here; `TRITIUM_DRAFT_K` selects
//! the per-slot policy at enrollment), each slot's chain becomes a chain tree
//! rooted at its last committed token (exactly the solo `spec_cycle` shape),
//! and ONE [`tree_verify_greedy_slots`](tritium_nn::ModelRunner::tree_verify_greedy_slots)
//! forward verifies them all (I4 — the f16 LM-head read amortized N-wide).
//! Committed tokens are per-slot target argmaxes — lossless, exactly the solo
//! stream. The DRAFTER carries its own `BatchKv` (one row per target slot,
//! dense — the drafter is small): a slot is *enrolled* by prefilling its
//! committed history through the drafter's single-sequence KV and adopting it
//! into the row; after each verify the drafter row rolls BACK to its accepted
//! prefix (`set_position`) and any 1-token feed gap (a fully-accepted chain's
//! last draft is drafted-never-fed) is closed by a masked k=1 `draft_batch`
//! step. **Per-slot k policy (v1)**: shared `k` = the minimum of the live
//! slots' `DraftPolicy` lengths, clamped so `Σ mᵣ = N·(1+k) <= 48` (the I4
//! one-bucket cap) — one verify group always suffices; per-slot budget/ctx
//! clamps then truncate individual chains (the overfed drafter rows are
//! rolled back). **Fallback discipline**: any draft/verify device error, page
//! exhaustion, or capacity edge quietly falls back to a lockstep step for the
//! round and drops the drafter pool state (target rows only ever hold
//! COMMITTED tokens between rounds, so lockstep resumes seamlessly); a
//! non-eligible admission (sampled/logprobs request, tree open) does the same
//! for as long as it lives — v1 has no mixed pool. Re-entry re-enrolls from
//! the per-slot histories. Spec rounds coexist with a chunked admission
//! prefill (they never touch the single-sequence KV), so a request admitted
//! mid-flight joins the spec pool on adoption. Exactly-solo ADMISSIONS still
//! take the I0 `SpecSeq` path above (empty-pool contract unchanged); a pool
//! that drains down to one live slot keeps running multi rounds with N=1
//! (same committed stream — the I2 gate pins slot-verify == single-seq).
//!
//! Remaining phase-2 cost: free slots still burn their dense GEMM rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

use opentelemetry::trace::TraceContextExt as _;
use tokio::sync::mpsc;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use crate::generator::{
    DraftPolicy, FinishReason, GenRequest, SPEC_COMMITTED, SPEC_COST, SPEC_VERIFIES, Sampling,
    SpecGovernor, draft_chain_from_env, draft_greedy_tokens_cancellable,
};
use crate::worker::{GenEvent, Job, PHASE_DECODE, PHASE_IDLE, PHASE_PREFILL, WorkerTelemetry};

/// Default prompt tokens per prefill chunk during admission (C1). 128 bounds a
/// live slot's inter-token gap to one ~128-token prefill + one decode step.
const PREFILL_CHUNK_DEFAULT: usize = 128;

/// Chunk size from `TRITIUM_PREFILL_CHUNK` (default
/// [`PREFILL_CHUNK_DEFAULT`]); rejects invalid values loudly rather than
/// guessing (the `TRITIUM_KV` selector pattern).
fn prefill_chunk() -> Result<usize, String> {
    match std::env::var("TRITIUM_PREFILL_CHUNK") {
        Ok(s) => match s.trim().parse::<usize>() {
            Ok(n) if n >= 1 => Ok(n),
            _ => Err(format!(
                "TRITIUM_PREFILL_CHUNK must be a positive integer, got {s:?}"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(PREFILL_CHUNK_DEFAULT),
        Err(e) => Err(format!("TRITIUM_PREFILL_CHUNK: {e}")),
    }
}

/// A prompt being chunk-prefilled through the runner's single-sequence KV;
/// `done` tokens are already in. At most one exists (admission is gated on
/// `pending.is_none()`), so the goal's resources can't be double-booked.
struct Pending {
    /// Prompt tokens already prefilled.
    done: usize,
    /// Monotonic start used for bounded prefill timing.
    started_at: Instant,
    /// Request trace retained across chunked prefill.
    request_span: tracing::Span,
    goal: PendingGoal,
}

/// What a completed prefill turns into (C4: the chunk machine serves both
/// chat admissions and BASTION tree-session opens).
enum PendingGoal {
    /// A chat admission: adopt into `row` and activate.
    Admit {
        tx: mpsc::Sender<GenEvent>,
        req: GenRequest,
        /// Token budget after context clamping (validated at admission).
        max_new: usize,
        /// The pool row this request will occupy once adopted.
        row: usize,
    },
    /// A tree-session open (ADR 0014): reply the prefill's greedy root; the
    /// session then OWNS the single-sequence KV until the next admission
    /// resets it (the single-worker contract — "a chat completion closes
    /// it" — verbatim in batched mode).
    TreeOpen {
        prompt: Vec<u32>,
        resp: tokio::sync::oneshot::Sender<Result<u32, crate::generator::TreeOpError>>,
    },
    /// A solo speculative admission (ADR 0032 L3 I0): on completion, emit the
    /// prefill's greedy argmax as the first token (the Admit first-token
    /// pattern) and install a [`SpecSeq`] that owns the single-sequence KV —
    /// no batch row is reserved (migration reserves one later if an
    /// admission arrives).
    SpecAdmit {
        tx: mpsc::Sender<GenEvent>,
        req: GenRequest,
        /// Token budget after context clamping (validated at admission).
        max_new: usize,
        /// Adaptive draft-length policy (from `TRITIUM_DRAFT_K`).
        policy: DraftPolicy,
        /// Adaptive spec on/off governor (from `TRITIUM_SPEC_ADAPTIVE`).
        governor: SpecGovernor,
        /// Chained device-side drafting (from `TRITIUM_DRAFT_CHAIN`).
        chain: bool,
    },
}

impl Pending {
    /// Run the next native chunk without publishing progress or adopting KV.
    fn forward_chunk(
        &self,
        runner: &mut tritium_nn::ModelRunner,
        chunk: usize,
        draining: &AtomicBool,
    ) -> Result<Option<Vec<f32>>, tritium_nn::NnError> {
        let end = self.done.saturating_add(chunk).min(self.prompt().len());
        let positions: Vec<usize> = (self.done..end).collect();
        runner.forward_cancellable(&self.prompt()[self.done..end], &positions, &|| {
            draining.load(Ordering::Acquire) || self.client_gone()
        })
    }

    fn prompt(&self) -> &[u32] {
        match &self.goal {
            PendingGoal::Admit { req, .. } | PendingGoal::SpecAdmit { req, .. } => {
                &req.prompt_tokens
            }
            PendingGoal::TreeOpen { prompt, .. } => prompt,
        }
    }
    fn client_gone(&self) -> bool {
        match &self.goal {
            PendingGoal::Admit { tx, .. } | PendingGoal::SpecAdmit { tx, .. } => tx.is_closed(),
            PendingGoal::TreeOpen { resp, .. } => resp.is_closed(),
        }
    }
    /// The reserved pool row (pages to release on abandonment), if any.
    fn row(&self) -> Option<usize> {
        match &self.goal {
            PendingGoal::Admit { row, .. } => Some(*row),
            PendingGoal::TreeOpen { .. } | PendingGoal::SpecAdmit { .. } => None,
        }
    }
    /// Fail with an internal error (device/forward faults).
    fn fail(self, msg: String) {
        match self.goal {
            PendingGoal::Admit { tx, .. } | PendingGoal::SpecAdmit { tx, .. } => {
                let _ = tx.try_send(GenEvent::Error(msg));
            }
            PendingGoal::TreeOpen { resp, .. } => {
                let _ = resp.send(Err(crate::generator::TreeOpError::Internal(msg)));
            }
        }
    }

    /// Fail because the server is draining — same classification a
    /// queue-drained job gets (Draining/503, not Internal/500).
    fn fail_draining(self) {
        match self.goal {
            PendingGoal::Admit { tx, .. } | PendingGoal::SpecAdmit { tx, .. } => {
                let _ = tx.try_send(GenEvent::Error("server draining".into()));
            }
            PendingGoal::TreeOpen { resp, .. } => {
                let _ = resp.send(Err(crate::generator::TreeOpError::Draining(
                    "server draining".into(),
                )));
            }
        }
    }
}

/// One live request occupying a slot.
struct Active {
    tx: mpsc::Sender<GenEvent>,
    /// Per-request trace parent; shared decode spans link to all active requests.
    request_span: tracing::Span,
    /// Unique per-admission id (multi-slot spec enrollment validity: a
    /// drafter row enrolled for a RETIRED occupant must never serve the
    /// row's next tenant — see [`SpecSlot::owner`]).
    id: u64,
    sampling: Sampling,
    /// Top-k logprobs per token, when the request asked.
    logprobs: Option<usize>,
    stop_eos: bool,
    /// Tokens still allowed to be emitted.
    remaining: usize,
    /// The last sampled token — fed to the model on the next step.
    last_token: u32,
    /// Prompt + every sampled token (the last element is `last_token`, the
    /// emitted-but-not-yet-fed pending token — the `SpecSeq` invariant).
    /// Maintained in BOTH modes so a slot can (re-)enroll into the
    /// multi-slot spec pool at any point (the drafter re-prefills from it).
    history: Vec<u32>,
    /// Per-request draw counter (salts the deterministic sampler stream).
    /// NOTE: the batched sampler stream (splitmix64 over (seed, salt)) is
    /// distribution-equal but NOT stream-equal to the single-request path's
    /// `seed + step` derivation — the same seed reproduces within a mode,
    /// not across modes.
    salt: u64,
}

/// splitmix64 → a per-draw seed for `sample_categorical` (mirrors the
/// spec-decode path's stream-derivation contract: distribution-equal draws,
/// deterministic per (seed, salt)).
fn draw_seed(seed: u64, salt: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(salt.wrapping_add(1)));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sample(logits: &[f32], s: &Sampling, seed_salt: (u64, u64)) -> Option<u32> {
    let (seed, salt) = seed_salt;
    match *s {
        Sampling::Greedy => tritium_nn::sample_greedy(logits),
        Sampling::TopK { k, temp, .. } => {
            let (idx, probs) = tritium_nn::truncated_top_k(logits, k, temp)?;
            Some(tritium_nn::sample_categorical(
                &idx,
                &probs,
                draw_seed(seed, salt),
            ))
        }
        Sampling::TopP { p, temp, .. } => {
            let (idx, probs) = tritium_nn::truncated_top_p(logits, p, temp)?;
            Some(tritium_nn::sample_categorical(
                &idx,
                &probs,
                draw_seed(seed, salt),
            ))
        }
    }
}

fn req_seed(s: &Sampling) -> u64 {
    match *s {
        Sampling::Greedy => 0,
        Sampling::TopK { seed, .. } | Sampling::TopP { seed, .. } => seed,
    }
}

/// Return a vacated row's KV pages to the pool (no-op on a dense batch).
/// Called at EVERY site that retires an Active or abandons a Pending.
#[allow(clippy::needless_pass_by_ref_mut)]
fn release_slot(batch: &mut tritium_cuda::BatchKv, row: usize, telemetry: &WorkerTelemetry) {
    if batch.paged() {
        let before = kv_free_tokens(batch);
        match batch.release_pages(row) {
            Ok(()) => telemetry.observe_kv_release(before, kv_free_tokens(batch)),
            Err(error) => {
                telemetry.observe_kv_release_failure();
                eprintln!("tritium-serve: paged KV release failed for row {row}: {error}");
            }
        }
    }
}

/// Consume a cancelled admission once. Its staging KV is not any live batch
/// row's KV; resetting it must not retire peers or invalidate their pages.
fn retire_pending_prefill(
    pending: &mut Option<Pending>,
    runner: &mut tritium_nn::ModelRunner,
    batch: &mut tritium_cuda::BatchKv,
    telemetry: &WorkerTelemetry,
    draining: bool,
) {
    let Some(pending) = pending.take() else {
        return;
    };
    let row = pending.row();
    if draining {
        pending.fail_draining();
    }
    runner.reset();
    if let Some(row) = row {
        release_slot(batch, row, telemetry);
    }
}

/// Keep the worker-owned FIFO wait visible alongside jobs still in the
/// bounded channel. There is at most one parked admission in this worker.
fn park_job(parked: &mut Option<Job>, job: Job, telemetry: &WorkerTelemetry) {
    assert!(
        parked.is_none(),
        "batch worker supports one parked queue job"
    );
    *parked = Some(job);
    telemetry.set_parked_queue_job(true);
}

fn job_client_gone(job: &Job) -> bool {
    match job {
        Job::Generate { tx, .. } => tx.is_closed(),
        Job::OpenTreeSession { resp, .. } => resp.is_closed(),
        Job::TreeVerify { resp, .. } => resp.is_closed(),
    }
}

fn take_parked_job(parked: &mut Option<Job>, telemetry: &WorkerTelemetry) -> Option<Job> {
    let job = parked.take();
    if job.is_some() {
        telemetry.set_parked_queue_job(false);
    }
    job
}

/// Return shared paged-KV free capacity in logical tokens. Dense batches have
/// no shared pool and intentionally report zero.
fn kv_free_tokens(batch: &tritium_cuda::BatchKv) -> usize {
    batch
        .free_pages()
        .saturating_mul(tritium_cuda::KV_PAGE_TOKENS)
}

/// Reserve row pages while publishing exact post-operation pool telemetry.
fn reserve_pages(
    batch: &mut tritium_cuda::BatchKv,
    row: usize,
    tokens: usize,
    telemetry: &WorkerTelemetry,
) -> Result<(), tritium_spec::BackendError> {
    let before = kv_free_tokens(batch);
    let result = batch.reserve_pages(row, tokens);
    if result.is_ok() {
        telemetry.observe_kv_reservation(before, kv_free_tokens(batch));
    }
    result
}

/// Emit one token on a slot's channel. Returns `false` when the request is
/// finished (EOS/budget) or the client went away — the slot should retire.
fn emit(active: &mut Active, token: u32, eos: u32, logits: &[f32]) -> bool {
    let is_eos = active.stop_eos && token == eos;
    let last = is_eos || active.remaining <= 1;
    let lp = active
        .logprobs
        .map(|k| crate::generator::top_logprobs(logits, token, k));
    let sent = active.tx.try_send(GenEvent::Token(token, lp)).is_ok();
    active.remaining = active.remaining.saturating_sub(1);
    if last && sent {
        let reason = if is_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        let _ = active.tx.try_send(GenEvent::Done(reason));
    }
    sent && !last
}

/// The one solo speculative sequence (ADR 0032 L3 I0). Owns the
/// single-sequence KV (the C4 serialized-ownership contract) until it
/// finishes or an admission migrates it into a batch slot.
///
/// Invariant between cycles: `history` = prompt + every emitted token (the
/// last element is the emitted-but-not-yet-forwarded "pending" token), the
/// target runner's cache holds `history[..len-1]`, and `emitted < max_new`
/// (a finished sequence is dropped immediately).
struct SpecSeq {
    tx: mpsc::Sender<GenEvent>,
    /// Per-request trace parent retained through speculative decode/migration.
    request_span: tracing::Span,
    /// Prompt + all emitted tokens (see the struct invariant).
    history: Vec<u32>,
    /// Tokens emitted so far (including history's last element).
    emitted: usize,
    /// Token budget after context clamping (fixed at admission).
    max_new: usize,
    /// The original request — sampling/stop flags for the migration
    /// continuation (prompt/max_new are overridden there).
    req: GenRequest,
    /// Adaptive draft-length policy (mirrors the single worker's).
    policy: DraftPolicy,
    /// Adaptive spec on/off governor (mirrors the single worker's).
    governor: SpecGovernor,
    /// Chained device-side drafting (`TRITIUM_DRAFT_CHAIN`).
    chain: bool,
    /// Drafter reconcile state — same contract as
    /// `RunnerGenerator::{draft_fed, draft_pos}` (see `draft_greedy_tokens`).
    draft_fed: Vec<u32>,
    draft_pos: usize,
    /// `TRITIUM_SPEC_STATS=1` per-request stats (mirrors the single worker).
    stats: bool,
    n_verify: usize,
    n_committed: usize,
    n_plain: usize,
    t_verify: std::time::Duration,
    t_plain: std::time::Duration,
}

impl SpecSeq {
    /// Per-request spec stats at retirement (mirrors `generate_spec_lookup`'s
    /// `TRITIUM_SPEC_STATS` print, including the cancel path).
    fn print_stats(&self) {
        if self.stats && self.n_verify > 0 {
            eprintln!(
                "spec-stats: verifies={} committed={} ({:.2} tok/verify, {:.1?}/verify) plain={} ({:.1?}/step)",
                self.n_verify,
                self.n_committed,
                self.n_committed as f64 / self.n_verify as f64,
                self.t_verify / self.n_verify as u32,
                self.n_plain,
                self.t_plain / self.n_plain.max(1) as u32,
            );
        }
    }
}

/// What one spec cycle did to the sequence.
enum SpecOutcome {
    /// Still decoding — run another cycle next iteration.
    Continue,
    /// Finished cleanly (EOS or budget); `Done` was sent.
    Done,
    /// The client went away (send failed) — retire silently.
    Cancelled,
}

/// Emit one committed token on the spec stream (the Active `emit` semantics:
/// EOS/budget finish reasons, try_send cancellation). Continuing tokens are
/// pushed into `history`, preserving the [`SpecSeq`] invariant.
fn emit_spec(s: &mut SpecSeq, token: u32, eos: u32) -> SpecOutcome {
    let is_eos = s.req.stop_eos && token == eos;
    let next_emitted = s.emitted.saturating_add(1);
    let last = is_eos || next_emitted >= s.max_new;
    let sent = s.tx.try_send(GenEvent::Token(token, None)).is_ok();
    s.emitted = next_emitted;
    if !sent {
        return SpecOutcome::Cancelled;
    }
    if last {
        let reason = if is_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        let _ = s.tx.try_send(GenEvent::Done(reason));
        return SpecOutcome::Done;
    }
    s.history.push(token);
    SpecOutcome::Continue
}

/// One greedy speculative cycle: draft a chain with the DRAFT runner, verify
/// it on the target with `tree_verify_greedy` (committing the accepted
/// prefix + one bonus in one forward), emit every committed token. Mirrors
/// `generator.rs::generate_spec_lookup`'s cycle body — that loop is the
/// source of truth for the budget clamps and commit walk; this flattens it
/// to one-cycle-per-worker-iteration (the emitted "pending" token lives as
/// `history`'s last element instead of a loop variable). Lossless: every
/// emitted token is the target's own greedy argmax at its position.
fn spec_cycle(
    runner: &mut tritium_nn::ModelRunner,
    draft: &mut tritium_nn::ModelRunner,
    s: &mut SpecSeq,
    eos: u32,
    n_ctx: usize,
    draining: &AtomicBool,
) -> Result<SpecOutcome, String> {
    if s.tx.is_closed() || draining.load(Ordering::Acquire) {
        return Ok(SpecOutcome::Cancelled);
    }
    let pending = *s.history.last().expect("spec history holds the prompt");
    // Budget-clamped draft: total tree rows must fit the KV arena
    // (cache_len = history.len() - 1, the verifier needs
    // cache_len + 1 + d <= n_ctx, so d <= n_ctx - history.len()), and
    // committed tokens (<= drafts + 1) must fit the emission budget.
    let kv_room = n_ctx.saturating_sub(s.history.len());
    let budget = s.max_new - s.emitted; // >= 1 by the SpecSeq invariant
    // Governor cap on top of the policy length (the single worker's rule):
    // Some(0) = suppressed plain step, Some(k) = probe, None = normal.
    let cap = s.governor.draft_cap();
    // Probe cycles pay the ~ctx-linear drafter re-prefill — routed to the
    // resync EWMA below, never the floor's d (the single worker's rule).
    let is_probe = matches!(cap, Some(k) if k > 0);
    let want = match cap {
        Some(k) => k,
        None => s.policy.len(),
    };
    let max_draft = want.min(kv_room).min(budget.saturating_sub(1));
    let t_d = std::time::Instant::now();
    let drafts = if max_draft == 0 {
        Vec::new() // suppressed (or clamped): no drafter work at all
    } else {
        let Some(drafts) = draft_greedy_tokens_cancellable(
            draft,
            &mut s.draft_fed,
            &mut s.draft_pos,
            eos,
            &s.history,
            max_draft,
            s.chain,
            &|| s.tx.is_closed() || draining.load(Ordering::Acquire),
        ) else {
            return Ok(SpecOutcome::Cancelled);
        };
        drafts
    };
    if s.tx.is_closed() || draining.load(Ordering::Acquire) {
        return Ok(SpecOutcome::Cancelled);
    }
    // Cost-model d: drafter wall per drafted token (empty results — bails —
    // carry no per-token denominator; skipped). Probe cycles feed
    // draft_resync (telemetry), steady-state cycles the floor's draft_tok.
    if !drafts.is_empty() {
        SPEC_COST.record_draft(
            t_d.elapsed().as_secs_f64() * 1e6 / drafts.len() as f64,
            is_probe,
        );
    }

    if drafts.is_empty() {
        // Plain M=1 graph step (faster than a 1-node tree).
        let t0 = std::time::Instant::now();
        let pos = s.history.len() - 1;
        let Some(logits) = runner
            .forward_cancellable(&[pending], &[pos], &|| {
                s.tx.is_closed() || draining.load(Ordering::Acquire)
            })
            .map_err(|e| e.to_string())?
        else {
            return Ok(SpecOutcome::Cancelled);
        };
        s.n_plain += 1;
        let el = t0.elapsed();
        s.t_plain += el;
        SPEC_COST.plain.record(el.as_secs_f64() * 1e6); // cost-model P
        s.governor.on_plain_commit(1); // no-op unless suppressed
        let next = tritium_nn::sample_greedy(&logits).ok_or_else(|| "empty logits".to_owned())?;
        return Ok(emit_spec(s, next, eos));
    }

    let token_capacity = 1usize
        .checked_add(drafts.len())
        .ok_or_else(|| "speculative token tree size overflow".to_owned())?;
    let mut tokens = Vec::with_capacity(token_capacity);
    tokens.push(pending);
    tokens.extend(&drafts);
    let parents: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
    let t0 = std::time::Instant::now();
    let Some(committed) = runner
        .tree_verify_greedy_cancellable(&tokens, &parents, &|| {
            s.tx.is_closed() || draining.load(Ordering::Acquire)
        })
        .map_err(|e| e.to_string())?
    else {
        return Ok(SpecOutcome::Cancelled);
    };
    s.n_verify += 1;
    s.n_committed += committed.len();
    SPEC_VERIFIES.fetch_add(1, Ordering::Relaxed);
    SPEC_COMMITTED.fetch_add(committed.len() as u64, Ordering::Relaxed);
    let el = t0.elapsed();
    s.t_verify += el;
    SPEC_COST.verify.record(el.as_secs_f64() * 1e6); // cost-model V
    // committed = accepted drafts + the final token, so accepted drafts =
    // committed.len() - 1 (saturating mirrors the single worker's guard).
    s.policy
        .update(drafts.len(), committed.len().saturating_sub(1));
    let floor = s.governor.floor_solo(s.policy.len());
    s.governor.on_verify(&s.policy, floor);
    if committed.is_empty() {
        return Err("tree verify returned an empty commit".into());
    }
    // The single worker emits committed[..L-1] in its walk and the last at
    // the next loop top; flattened here, all of them emit now and the last
    // becomes the next cycle's pending via the history push — same stream.
    for &c in &committed {
        match emit_spec(s, c, eos) {
            SpecOutcome::Continue => {}
            other => return Ok(other),
        }
    }
    Ok(SpecOutcome::Continue)
}

/// Migrate the solo spec sequence into a batch slot (I0
/// "migrate-on-admission"): its full history becomes a continuation
/// admission — prompt = history, budget = the unspent remainder, same
/// stream — installed as the next `Pending` so it prefills AHEAD of the
/// admission that displaced it. Nothing is re-emitted: spec emitted every
/// committed token including history's last, and the continuation prefill's
/// argmax after the full history is exactly the next unemitted token.
/// Returns `None` (stream already errored) only on a defensive
/// page-reservation failure that the spec-admission precheck makes
/// unreachable.
fn migrate_spec(
    s: SpecSeq,
    runner: &mut tritium_nn::ModelRunner,
    batch: &mut tritium_cuda::BatchKv,
    draft: Option<&mut tritium_nn::ModelRunner>,
    pool: &[Option<Active>],
    telemetry: &WorkerTelemetry,
) -> Option<Pending> {
    // The drafter's KV holds speculatively-fed tokens past the commit point;
    // its reconcile state died with the SpecSeq, so reset it explicitly (a
    // later spec admission re-prefills from scratch anyway).
    if let Some(d) = draft {
        d.reset();
    }
    let remaining = s.max_new - s.emitted; // >= 1 by the SpecSeq invariant
    let row = pool
        .iter()
        .position(Option::is_none)
        .expect("spec runs only while the pool is empty");
    if batch.paged() {
        // Spec admission pre-checked prompt + max_new against the pool
        // capacity and the pool is empty (all pages free), so this cannot
        // fail; error the stream loudly rather than trusting that silently.
        let Some(needed) = s.history.len().checked_add(remaining) else {
            let _ = s.tx.try_send(GenEvent::Error(
                "spec migration token footprint overflow".into(),
            ));
            return None;
        };
        if let Err(e) = reserve_pages(batch, row, needed, telemetry) {
            let _ = s.tx.try_send(GenEvent::Error(format!(
                "spec migration page reserve failed: {e}"
            )));
            return None;
        }
    }
    // Fresh single-sequence prefill for the continuation (the Admit path's
    // reset — the spec KV is superseded, not adopted, keeping migration on
    // the already-gated chunked-admission path).
    runner.reset();
    let mut req = s.req;
    req.prompt_tokens = s.history;
    req.max_new = remaining;
    Some(Pending {
        done: 0,
        started_at: Instant::now(),
        request_span: s.request_span,
        goal: PendingGoal::Admit {
            tx: s.tx,
            req,
            max_new: remaining,
            row,
        },
    })
}

/// The I4 batched-slots verify's one-bucket node cap — mirrors tritium-cuda's
/// private `TREE_BUCKET_MAX`. Kept local on purpose: if the engine cap ever
/// changes downward, the verify refuses loudly and the round falls back to
/// lockstep (slow, never wrong).
const TREE_NODE_CAP: usize = 48;

/// Drafter-side state of the multi-slot spec pool (module docs, "Multi-slot
/// speculative decoding"). Exists only across CONSECUTIVE spec rounds: any
/// lockstep round, fallback, drain, or fully-emptied pool drops it, and
/// re-entry re-enrolls from the slots' histories.
struct SpecPool {
    /// The DRAFT runner's batch KV — one row per target slot, dense (the
    /// drafter is small; `slots × draft_ctx` KV is cheap).
    dbatch: tritium_cuda::BatchKv,
    /// Per-row enrollment; `None` = the drafter holds nothing valid for the
    /// row. Length = target slot count.
    slots: Vec<Option<SpecSlot>>,
}

/// One enrolled drafter row. The row's valid-KV watermark is
/// `dbatch.positions()[row]` (KV rows `[0, watermark)` hold the owner's
/// `history[..watermark]`); the per-round reconcile keeps
/// `watermark <= history.len() - 1` with a gap of at most one token.
struct SpecSlot {
    /// The [`Active::id`] this enrollment is valid for — a row reused by a
    /// new admission re-enrolls instead of trusting a dead tenant's KV.
    owner: u64,
    /// Per-slot adaptive draft-length policy (`TRITIUM_DRAFT_K`), seeded
    /// fresh at enrollment. Draft length never affects the stream
    /// (losslessness), so policy state is purely a cost knob.
    policy: DraftPolicy,
    /// Per-slot adaptive spec on/off governor (`TRITIUM_SPEC_ADAPTIVE`): a
    /// slot whose acceptance collapsed stops drafting — its tree is the
    /// 1-node root, which through the shared verify IS the batch-friendly
    /// plain step — and probes periodically. Purely a cost knob (the 1-node
    /// bonus commit is the target's own argmax).
    governor: SpecGovernor,
}

/// Once-per-worker markers for the QUIET capacity fallbacks: each class
/// recurs every round while its condition holds (pool too wide, a slot's
/// history at the drafter's context edge), and without a marker the only
/// symptom would be a flat spec counter.
#[derive(Default)]
struct MultiFallbackLog {
    cap: bool,
    ctx: bool,
    k0: bool,
}

/// What one multi-slot spec round attempt did.
enum MultiOutcome {
    /// The round ran: drafts verified, committed tokens emitted.
    Ran,
    /// A grouped verify observed drain or a closed response before any
    /// promotion. Drop drafter enrollment and re-enter retirement/admission;
    /// do not fall through to an uncancellable lockstep step in this tick.
    Cancelled,
    /// Machinery unavailable this round (capacity edge, page exhaustion, or
    /// a device error — logged when it is an error): the caller drops the
    /// pool state and falls through to a lockstep step. Streams unaffected —
    /// target rows only ever hold committed tokens between rounds.
    Fallback,
    /// A condition that will not heal (no drafter resident decoder, bad
    /// `TRITIUM_DRAFT_K` env): disable multi-slot spec for the worker's
    /// lifetime instead of retry-spamming the log.
    Disable,
    /// Every live slot's governor has drafting suppressed and none is due a
    /// probe: the caller runs a lockstep step (the M=N graph step — cheaper
    /// than an all-1-node verify round) but KEEPS the pool state, so probe
    /// rounds re-enter without rebuilding it. Suppressed rows' drafter
    /// watermarks go stale on purpose; a probe re-syncs via the enrollment
    /// prefill.
    Lockstep,
}

fn spec_group_cancelled(
    pool: &[Option<Active>],
    rows: &[usize],
    is_cancelled: &dyn Fn() -> bool,
) -> bool {
    is_cancelled()
        || rows
            .iter()
            .any(|&row| pool[row].as_ref().is_none_or(|a| a.tx.is_closed()))
}

fn cancel_spec_rows(
    batch: &mut tritium_cuda::BatchKv,
    pool: &mut [Option<Active>],
    rows: &[usize],
    telemetry: &WorkerTelemetry,
) -> MultiOutcome {
    for &row in rows {
        if pool[row].as_ref().is_some_and(|a| a.tx.is_closed()) {
            pool[row] = None;
            release_slot(batch, row, telemetry);
        }
    }
    MultiOutcome::Cancelled
}

// Full and delta enrollment share a controlled prefill + pre-adoption seam.
// `Some(())` grants enrollment; cancelled work cannot publish adoption.
fn enroll_draft_row(
    draft: &mut tritium_nn::ModelRunner,
    batch: &mut tritium_cuda::BatchKv,
    row: usize,
    history: &[u32],
    prefix: Option<usize>,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<Option<()>, String> {
    if is_cancelled() {
        return Ok(None);
    }
    let start = prefix.unwrap_or(0);
    let p = history.len() - 1;
    if prefix.is_some() {
        draft
            .adopt_from_batch_row(batch, row, start)
            .map_err(|e| e.to_string())?;
    } else {
        draft.reset();
    }
    let positions: Vec<usize> = (start..p).collect();
    let output = draft
        .forward_cancellable(&history[start..p], &positions, is_cancelled)
        .map_err(|e| e.to_string())?;
    if output.is_none() || is_cancelled() {
        draft.reset();
        return Ok(None);
    }
    draft
        .adopt_into_batch_row(batch, row, p)
        .map_err(|e| e.to_string())?;
    batch.set_position(row, p).map_err(|e| e.to_string())?;
    Ok(Some(()))
}

/// One multi-slot spec round (module docs, "Multi-slot speculative
/// decoding"): enroll → close drafter feed gaps → batched draft (I1) →
/// grouped multi-slot verify (I4) → per-slot commit/emit + drafter rollback.
/// The caller has already established eligibility (every live slot greedy +
/// logprob-free, no tree session) and `PHASE_DECODE`.
///
/// Lossless by construction: every emitted token is the target's own greedy
/// argmax on the slot path (I2 pins slot-verify == single-sequence verify,
/// I4 pins batched == sequential slots, I1 pins batched drafts == chained
/// drafts — and draft content never changes the committed stream).
#[allow(clippy::too_many_lines)] // one round, straight-line; splitting hides the state flow
#[allow(clippy::too_many_arguments)] // the round's full wiring, one call site
fn multi_spec_round(
    runner: &mut tritium_nn::ModelRunner,
    draft: &mut tritium_nn::ModelRunner,
    batch: &mut tritium_cuda::BatchKv,
    multi: &mut Option<SpecPool>,
    pool: &mut [Option<Active>],
    eos: u32,
    n_ctx: usize,
    log: &mut MultiFallbackLog,
    telemetry: &WorkerTelemetry,
    is_cancelled: &dyn Fn() -> bool,
) -> MultiOutcome {
    let slots = pool.len();
    let rows: Vec<usize> = (0..slots).filter(|&r| pool[r].is_some()).collect();
    let n_live = rows.len();
    if n_live == 0 {
        return MultiOutcome::Fallback;
    }
    if spec_group_cancelled(pool, &rows, is_cancelled) {
        return cancel_spec_rows(batch, pool, &rows, telemetry);
    }
    // Even k=1 chains cost 2 nodes per slot; past the cap the one-bucket
    // verify cannot hold everyone. (Realistic pools are far smaller; a
    // grouped-rounds variant is the measured follow-up if ever needed.)
    let tree_nodes = n_live.checked_mul(2);
    if tree_nodes.is_none_or(|nodes| nodes > TREE_NODE_CAP) {
        if !log.cap {
            log.cap = true;
            eprintln!(
                "tritium-serve: multi-slot spec idle — {n_live} live slots need \
                 {} tree nodes > the {TREE_NODE_CAP}-node verify bucket; \
                 lockstep until the pool shrinks (logged once)",
                tree_nodes.unwrap_or(usize::MAX)
            );
        }
        return MultiOutcome::Fallback;
    }
    // Per-row governor plans (None = draft normally, Some(0) = suppressed
    // plain step, Some(k) = probe chain). Un-enrolled rows plan a normal
    // draft (a fresh governor is never suppressed). Computed once up front —
    // probe clocks only advance at commit time, so plans are stable within
    // the round.
    let mut caps: Vec<Option<usize>> = (0..slots)
        .map(|r| {
            multi
                .as_ref()
                .and_then(|mp| mp.slots[r].as_ref())
                // Owner check (review 4190673 F1): a slot retired during a
                // Lockstep round keeps its stale SpecSlot until re-enrollment
                // — without this filter a FRESH admission into that row read
                // the dead owner's suppressed governor and decoded plain for
                // up to a probe period. Mirror the enrollment loop's check.
                .filter(|s| pool[r].as_ref().is_some_and(|a| s.owner == a.id))
                .and_then(|s| s.governor.draft_cap())
        })
        .collect();
    if rows.iter().all(|&r| caps[r] == Some(0)) {
        // Whole pool suppressed, nobody due a probe: the lockstep graph step
        // is the cheapest plain step (beats an all-1-node verify round) and
        // the drafter does no work at all. Advance each governor's probe
        // clock by the one token that step commits; the pool state survives
        // (see MultiOutcome::Lockstep).
        let mp = multi.as_mut().expect("suppressed slots are enrolled");
        for &r in &rows {
            if let Some(slot) = mp.slots[r].as_mut() {
                slot.governor.on_plain_commit(1);
            }
        }
        return MultiOutcome::Lockstep;
    }

    // Host-only feasibility BEFORE any drafter work, so a doomed round costs
    // nothing (no pool alloc, no enrollment prefills — without this hoist a
    // slot at the drafter's context edge made every round pay full
    // enrollment for the OTHER rows and then fall back). Every DRAFTING row
    // needs drafter room for its history, the pending feed, and >= 1 draft:
    // p + 2 < draft_ctx, which also bounds the shared k below to >= 2 per
    // row before the policy min. Suppressed rows never draft, so they are
    // exempt — a collapsed long-ctx slot must not idle the whole pool.
    let draft_ctx = draft.config.n_ctx as usize;
    let mut k = TREE_NODE_CAP / n_live - 1; // >= 1 by the cap check above
    for &r in &rows {
        if caps[r] == Some(0) {
            continue; // suppressed: no drafter work this round
        }
        let p = pool[r].as_ref().expect("live row").history.len() - 1;
        if p.saturating_add(2) >= draft_ctx {
            if caps[r].is_some() {
                // A probe blocked at the drafter's context edge: plain step
                // this round; the (cheap) probe attempt recurs next round.
                caps[r] = Some(0);
                continue;
            }
            if !log.ctx {
                log.ctx = true;
                eprintln!(
                    "tritium-serve: multi-slot spec idle — a slot's history \
                     ({} tokens) is at the drafter's context edge ({draft_ctx}); \
                     lockstep until it retires (logged once)",
                    p.saturating_add(1)
                );
            }
            return MultiOutcome::Fallback;
        }
        k = k.min(draft_ctx - p - 1);
    }

    // Build the drafter pool lazily (first eligible round, or re-entry after
    // a fallback). A drafter without the resident decoder can never draft
    // batched — disable rather than rebuild-spam.
    if multi.is_none() {
        match draft.new_batch(slots) {
            Ok(dbatch) => {
                *multi = Some(SpecPool {
                    dbatch,
                    slots: (0..slots).map(|_| None).collect(),
                });
            }
            Err(e) => {
                eprintln!("tritium-serve: multi-slot spec disabled — drafter batch pool: {e}");
                return MultiOutcome::Disable;
            }
        }
    }
    let mp = multi.as_mut().expect("built above");

    // Enroll rows the drafter does not hold (fresh admissions, reused rows,
    // re-entry after a fallback): prefill the slot's committed history
    // (minus the pending token) through the drafter's single-sequence KV and
    // adopt it into the row. Drafter room was established by the hoisted
    // feasibility check above.
    for &r in &rows {
        let a = pool[r].as_ref().expect("row in rows is live");
        let p = a.history.len() - 1;
        let keep = mp.slots[r].as_ref().is_some_and(|s| s.owner == a.id);
        if keep {
            // Suppressed rows are masked out of the gap-close below, so
            // their drafter watermark goes stale ON PURPOSE (no drafter
            // cost while plain). A kept row about to draft again (probe or
            // recovery) whose watermark fell more than the gap-close's
            // one-token contract behind re-syncs through this same
            // prefill+adopt path, KEEPING its policy/governor state.
            if caps[r] == Some(0) || p.saturating_sub(mp.dbatch.positions()[r]) <= 1 {
                continue;
            }
        } else {
            mp.slots[r] = None;
        }
        // Delta re-sync (kept rows only): a probe/recovery re-entry whose
        // watermark fell behind keeps the row's OWN valid KV prefix — adopt
        // it back into the single-seq cache, forward only the gap, adopt the
        // extended prefix into the row. The row's rows [0, dpos) equal
        // history[..dpos] for a kept row by construction (enrollment prefill
        // + reconciled feeds; history is append-only), so this is the same
        // end state as the full re-prefill at a cost sized by the gap + two
        // D2D copies, not the whole history (measured 6.2 ms vs 71.3 ms at
        // the 4K/64-gap probe shape, quiet box). Taken when the kept prefix
        // exceeds the gap — the regime where the delta is strictly cheaper;
        // short-prefix rows keep the full path. Any error falls through to
        // the full path below.
        if keep {
            let dpos = mp.dbatch.positions()[r];
            let gap = p.saturating_sub(dpos);
            if gap < dpos {
                let delta =
                    enroll_draft_row(draft, &mut mp.dbatch, r, &a.history, Some(dpos), &|| {
                        spec_group_cancelled(pool, &rows, is_cancelled)
                    });
                match delta {
                    Ok(Some(())) => {
                        crate::generator::SPEC_DELTA_RESYNCS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                    Ok(None) => return cancel_spec_rows(batch, pool, &rows, telemetry),
                    Err(e) => eprintln!(
                        "tritium-serve: multi-slot spec delta re-sync (row {r}): {e} — \
                         falling back to the full re-prefill"
                    ),
                }
            }
        }
        match enroll_draft_row(draft, &mut mp.dbatch, r, &a.history, None, &|| {
            spec_group_cancelled(pool, &rows, is_cancelled)
        }) {
            Ok(Some(())) => {}
            Ok(None) => return cancel_spec_rows(batch, pool, &rows, telemetry),
            Err(e) => {
                eprintln!("tritium-serve: multi-slot spec enrollment (row {r}): {e}");
                return MultiOutcome::Fallback;
            }
        }
        if !keep {
            let (policy, governor) = match (DraftPolicy::from_env(), SpecGovernor::from_env()) {
                (Ok(p), Ok(g)) => (p, g),
                (Err(e), _) | (_, Err(e)) => {
                    eprintln!("tritium-serve: multi-slot spec disabled — {e}");
                    return MultiOutcome::Disable;
                }
            };
            mp.slots[r] = Some(SpecSlot {
                owner: a.id,
                policy,
                governor,
            });
        }
    }

    // Close per-row drafter feed gaps: after a fully-accepted round the
    // chain's last draft was drafted-never-fed (the draft_batch KV
    // contract), leaving `watermark = p - 1`; feed `history[watermark]` via
    // one masked k=1 draft_batch step (the batched analogue of the solo
    // path's gap `forward`). Gaps are <= 1 by the reconcile below, so this
    // loop runs at most twice; a gap that will not close is a logic error —
    // fall back rather than wedge. (Future optimization, deliberately not
    // built: the drafted token this step DISCARDS is the drafter's guess at
    // the next pending — when it matches, it could seed the next chain and
    // save one lockstep drafter step per fully-accepted round.)
    for guard in 0.. {
        let mut feeds = vec![0u32; slots];
        let mut any_gap = false;
        for r in 0..slots {
            let gap_feed = pool[r].as_ref().and_then(|a| {
                mp.slots[r].as_ref()?;
                if caps[r] == Some(0) {
                    return None; // suppressed: watermark stale on purpose
                }
                let dpos = mp.dbatch.positions()[r];
                let p = a.history.len() - 1;
                debug_assert!(dpos <= p, "drafter watermark past pending");
                (dpos < p).then(|| a.history[dpos])
            });
            if mp.dbatch.set_live(r, gap_feed.is_some()).is_err() {
                return MultiOutcome::Fallback;
            }
            if let Some(t) = gap_feed {
                feeds[r] = t;
                any_gap = true;
            }
        }
        if !any_gap {
            break;
        }
        if guard >= 3 {
            eprintln!("tritium-serve: multi-slot spec gap did not close; lockstep round");
            return MultiOutcome::Fallback;
        }
        match draft.draft_batch_cancellable(&mut mp.dbatch, &feeds, 1, eos, &|| {
            spec_group_cancelled(pool, &rows, is_cancelled)
        }) {
            Ok(Some(_)) => {}
            Ok(None) => return cancel_spec_rows(batch, pool, &rows, telemetry),
            Err(e) => {
                eprintln!("tritium-serve: multi-slot spec gap feed: {e}");
                return MultiOutcome::Fallback;
            }
        }
    }

    // Shared draft length (the v1 policy, module docs): min over the live
    // slots' policy lengths, on top of the hoisted clamps (every enrolled
    // row's drafter-context room — draft_batch's own guard — and the I4
    // one-bucket cap `N·(1+k) <= 48`, so one verify group always suffices).
    // Per-slot budget/target-room clamps are applied by TRUNCATING chains
    // below (the overfed drafter rows roll back), so one slot near its end
    // never drags the whole pool to k=0.
    for &r in &rows {
        match caps[r] {
            // Suppressed: a 1-node plain step — must NOT drag the shared k
            // (a collapsed slot's policy length sits at its floor of 1).
            Some(0) => {}
            // Probe: a fixed short chain, independent of the collapsed
            // policy length (see SpecGovernor::PROBE_K).
            Some(pk) => k = k.min(pk),
            None => {
                let slot = mp.slots[r].as_ref().expect("enrolled above");
                k = k.min(slot.policy.len());
            }
        }
    }
    if k == 0 {
        // Unreachable — the hoisted clamps keep every term >= 1 — but a
        // future clamp must fall back loudly-once, not spin silently.
        if !log.k0 {
            log.k0 = true;
            eprintln!("tritium-serve: multi-slot spec idle — draft length clamped to 0");
        }
        return MultiOutcome::Fallback;
    }
    // Bucket snap (Track A fallback, measured 2026-08-08): the batched
    // verify trunk runs at the PADDED bucket size (tritium_cuda's
    // TREE_BUCKETS ladder) and only the norm+lm_head tail at real Σm, so a
    // group strictly between buckets pays the next bucket's whole trunk for
    // rows that are pure padding. Reduce the shared k (floor 1) until
    // N·(1+k) fits the largest bucket <= the policy Σm — trading at most a
    // couple of drafts per slot for a trunk bucket of FLOPs. Unreachable
    // snaps (Σm already on a bucket, below the smallest bucket, or the fit
    // would need k=0) keep k unchanged. Kept (no switch) on the 2026-08-08
    // ABBA A/B: +6.6% aggregate tok/s at N=4, +40% at N=2 (noisy session,
    // direction consistent) vs the unsnapped shared k. Draft length never
    // affects the committed stream, so this is a pure cost knob.
    // Suppressed rows contribute their 1-node root only, not 1 + k.
    let n_draft = rows.iter().filter(|&&r| caps[r] != Some(0)).count();
    let Some(m_total) = n_draft
        .checked_mul(k)
        .and_then(|draft_nodes| n_live.checked_add(draft_nodes))
    else {
        if !log.cap {
            log.cap = true;
            eprintln!("tritium-serve: multi-slot spec idle — tree node count overflow");
        }
        return MultiOutcome::Fallback;
    };
    if n_draft > 0
        && !tritium_cuda::TREE_BUCKETS.contains(&m_total)
        && let Some(b) = tritium_cuda::TREE_BUCKETS
            .iter()
            .copied()
            .filter(|&b| b <= TREE_NODE_CAP && b < m_total)
            .max()
    {
        let k_snap = b.saturating_sub(n_live) / n_draft;
        if k_snap >= 1 {
            k = k.min(k_snap);
        }
    }
    // Per-slot chain caps: committed (<= drafts + 1) must fit the emission
    // budget, and the verify needs pos + 1 + d <= n_ctx (the solo
    // spec_cycle's clamps, per slot).
    let d_max: Vec<usize> = (0..slots)
        .map(|r| {
            pool[r].as_ref().map_or(0, |a| {
                // Suppressed slot: 1-node root (the batch-friendly plain
                // step); probe slot: at most its fixed probe chain.
                let kr = match caps[r] {
                    Some(0) => return 0,
                    Some(pk) => k.min(pk),
                    None => k,
                };
                // The verify needs pos + 1 + d <= n_ctx (pos = p), i.e.
                // d <= n_ctx - history.len() — the solo spec_cycle's kv_room.
                let p = a.history.len() - 1;
                kr.min(a.remaining.saturating_sub(1))
                    .min(n_ctx.saturating_sub(p.saturating_add(1)))
            })
        })
        .collect();

    // Batched draft (I1): rows with a zero chain cap are masked dead (their
    // tree is the 1-node root — a plain step through the shared verify).
    let mut feeds = vec![0u32; slots];
    let mut any_draft = false;
    for r in 0..slots {
        let drafting = pool[r].is_some() && d_max[r] > 0;
        if mp.dbatch.set_live(r, drafting).is_err() {
            return MultiOutcome::Fallback;
        }
        if drafting {
            feeds[r] = *pool[r]
                .as_ref()
                .expect("drafting row is live")
                .history
                .last()
                .expect("history holds the prompt");
            any_draft = true;
        }
    }
    let chains: Vec<Vec<u32>> = if any_draft {
        match draft.draft_batch_cancellable(&mut mp.dbatch, &feeds, k, eos, &|| {
            spec_group_cancelled(pool, &rows, is_cancelled)
        }) {
            Ok(Some(c)) => c,
            Ok(None) => return cancel_spec_rows(batch, pool, &rows, telemetry),
            Err(e) => {
                // Mid-draft device errors leave the drafter unreconcilable
                // (fed tokens the host never saw); dropping the pool forces
                // re-enrollment — the documented recovery.
                eprintln!("tritium-serve: multi-slot draft_batch: {e}");
                return MultiOutcome::Fallback;
            }
        }
    } else {
        vec![Vec::new(); slots]
    };

    // Build each slot's chain tree (the solo spec_cycle shape: root = the
    // pending token, then the chain, parents i-1). `fed[r]` = tokens
    // draft_batch fed (= the FULL chain length — the last drafted id is
    // never fed); the tree may be shorter (budget/ctx truncation).
    let mut trees: Vec<(usize, Vec<u32>, Vec<i32>)> = Vec::with_capacity(n_live);
    let mut fed = vec![0usize; slots];
    for &r in &rows {
        let a = pool[r].as_ref().expect("live");
        fed[r] = chains[r].len();
        let take = d_max[r].min(chains[r].len());
        let Some(token_capacity) = 1usize.checked_add(take) else {
            return MultiOutcome::Fallback;
        };
        let mut tokens = Vec::with_capacity(token_capacity);
        tokens.push(*a.history.last().expect("history holds the prompt"));
        tokens.extend(&chains[r][..take]);
        let parents: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
        trees.push((r, tokens, parents));
    }

    // Verify in groups of Σ m <= 48 and reconcile each group before the
    // next. With the equal-split k clamp ONE group always suffices; the loop
    // is defensive (and ready for a future per-slot k).
    let mut start = 0usize;
    while start < trees.len() {
        let mut end = start;
        let mut m_sum = 0usize;
        while end < trees.len()
            && m_sum
                .checked_add(trees[end].1.len())
                .is_some_and(|next| next <= TREE_NODE_CAP)
        {
            m_sum = m_sum
                .checked_add(trees[end].1.len())
                .expect("checked tree group size");
            end += 1;
        }
        debug_assert!(end > start, "k clamp guarantees every chain fits a bucket");
        let group = &trees[start..end];
        for &(r, ref tokens, _) in group {
            if batch.set_live(r, true).is_err() {
                return MultiOutcome::Fallback;
            }
            if batch.paged() {
                // Exact batched-slots demand is prefix + m (I4 pads write
                // nothing), which the admission's prompt+max_new reservation
                // already covers — this reserve is a defensive no-op that
                // turns a bookkeeping bug into a slow round, not a wedge.
                let Some(need) = batch.positions()[r].checked_add(tokens.len()) else {
                    eprintln!("tritium-serve: multi-slot verify token position overflow (row {r})");
                    return MultiOutcome::Fallback;
                };
                if reserve_pages(batch, r, need, telemetry).is_err() {
                    eprintln!("tritium-serve: multi-slot verify page reserve (row {r})");
                    return MultiOutcome::Fallback;
                }
            }
        }
        let group_rows: Vec<usize> = group.iter().map(|&(r, ..)| r).collect();
        let group_trees: Vec<(&[u32], &[i32])> = group
            .iter()
            .map(|(_, t, p)| (t.as_slice(), p.as_slice()))
            .collect();
        let t_v = std::time::Instant::now();
        let outs = match runner.tree_verify_greedy_slots_cancellable(
            batch,
            &group_rows,
            &group_trees,
            &|| spec_group_cancelled(pool, &group_rows, is_cancelled),
        ) {
            Ok(Some(o)) => {
                // Cost-model V_round: one grouped verify's wall (with the
                // equal-split k clamp there is one group per round). The
                // RAW group wall is recorded at whatever group size is live
                // now; `floor_batched` divides the EWMA by the n_live at
                // APPLICATION time, so a pool resize mis-prices the floor
                // transiently until the EWMA re-converges (~a dozen rounds
                // at ALPHA 0.2) — the [1.1, 3.0] clamps bound the error.
                SPEC_COST
                    .verify_round
                    .record(t_v.elapsed().as_secs_f64() * 1e6);
                o
            }
            Ok(None) => {
                // Every selected target prefix is unchanged. The drafter
                // already fed chains, so the caller drops all enrollment and
                // live peers re-enroll from their unmodified host histories.
                // Release only disconnected rows here; drain framing and
                // release for still-connected rows belong to the outer loop.
                return cancel_spec_rows(batch, pool, &group_rows, telemetry);
            }
            // An InvalidInput refusal is ATOMIC — every target and tree is
            // host-validated before any device work, so no listed slot
            // changed and the lockstep fallback is seamless and lossless.
            Err(tritium_nn::ResidentOpError::Op(tritium_spec::BackendError::InvalidInput(m))) => {
                eprintln!("tritium-serve: multi-slot tree verify refused: {m}");
                return MultiOutcome::Fallback;
            }
            // Any OTHER error is NOT atomic: the engine promotes the
            // group's slots SEQUENTIALLY after the forward, so the error
            // may have landed after some slots' KV/positions already
            // advanced — their committed tokens are lost with the error,
            // and a silent fallback would resume lockstep from a stale
            // pending token (dropped tokens + divergence). We cannot know
            // which slots promoted: error every listed stream loudly (the
            // lockstep device-error classification) and retire them.
            // Slots in OTHER groups are safe — earlier groups fully
            // reconciled, later ones never reached the device.
            Err(e) => {
                eprintln!("tritium-serve: multi-slot tree verify: {e}");
                for &(r, ..) in group {
                    if let Some(a) = pool[r].take() {
                        let _ = a.tx.try_send(GenEvent::Error(format!(
                            "speculative verify failed mid-commit: {e}"
                        )));
                        release_slot(batch, r, telemetry);
                    }
                    mp.slots[r] = None;
                }
                return MultiOutcome::Fallback;
            }
        };

        // Per-slot reconcile: policy fold, drafter rollback to the accepted
        // prefix, then emit every committed token (EOS/budget truncate +
        // retire exactly like the lockstep emit path). The group's promotes
        // have ALREADY advanced the slots' KV/positions, so from here NO
        // exit may strand a promoted-but-unemitted slot: drafter-side
        // bookkeeping errors only cost the pool state (fallback AFTER the
        // loop), never a committed token.
        let mut drafter_broken = false;
        for (&(r, ref tokens, _), committed) in group.iter().zip(outs) {
            let offered = tokens.len() - 1;
            let l = committed.len();
            if l == 0 {
                // Contract violation (the accept walk always commits >= 1):
                // this slot's promote advanced by an unknown amount with
                // nothing to emit — error THIS stream loudly and retire it
                // (the mid-commit rule); the other slots' commits proceed.
                eprintln!("tritium-serve: multi-slot verify returned an empty commit");
                if let Some(a) = pool[r].take() {
                    let _ = a.tx.try_send(GenEvent::Error(
                        "speculative verify returned an empty commit".into(),
                    ));
                    release_slot(batch, r, telemetry);
                }
                mp.slots[r] = None;
                drafter_broken = true;
                continue;
            }
            let slot = mp.slots[r].as_mut().expect("enrolled above");
            if offered > 0 {
                SPEC_VERIFIES.fetch_add(1, Ordering::Relaxed);
                SPEC_COMMITTED.fetch_add(l as u64, Ordering::Relaxed);
                slot.policy.update(offered, l - 1);
                let floor = slot.governor.floor_batched(n_live);
                slot.governor.on_verify(&slot.policy, floor);
            } else {
                // A suppressed slot's 1-node tree is its plain step: no
                // acceptance signal to fold (and no spec-counter bump —
                // these commits are plain decode); advance its probe clock
                // and the suppression counter instead.
                slot.governor.on_plain_commit(l);
            }
            let a = pool[r].as_mut().expect("live");
            let p = a.history.len() - 1; // PRE-commit pending position
            if fed[r] > 0 {
                // draft_batch fed [pending, chain[..fed-1]]; feeds matching
                // the now-committed history are `1 + min(l-1, fed-1)` (the
                // accepted prefix). Rolling back to exactly that leaves a
                // gap of at most one token (l <= fed + 1), closed next
                // round.
                let matched = 1usize.saturating_add((l - 1).min(fed[r] - 1));
                let position_ok = match p.checked_add(matched) {
                    Some(next_position) => mp.dbatch.set_position(r, next_position).is_ok(),
                    None => false,
                };
                if !position_ok {
                    // Bad row/pos = corrupted bookkeeping: drop the
                    // enrollment (pool falls back after the loop) but KEEP
                    // emitting — the target side already committed.
                    mp.slots[r] = None;
                    drafter_broken = true;
                } else {
                    debug_assert_eq!(
                        mp.dbatch.positions()[r],
                        p.saturating_add(1usize.saturating_add((l - 1).min(fed[r] - 1))),
                        "drafter watermark mismatch after rollback (row {r})"
                    );
                }
            }
            let mut retire = false;
            for &c in &committed {
                a.history.push(c);
                a.last_token = c;
                if !emit(a, c, eos, &[]) {
                    // Finished (EOS/budget) or client gone: retire the slot
                    // and its enrollment; later committed tokens (past an
                    // accepted EOS) are dropped with it.
                    retire = true;
                    break;
                }
            }
            if retire {
                pool[r] = None;
                mp.slots[r] = None;
                release_slot(batch, r, telemetry);
            }
        }
        if drafter_broken {
            return MultiOutcome::Fallback; // commits all emitted; only drafter state is lost
        }
        start = end;
    }
    MultiOutcome::Ran
}

/// The batched worker loop: owns the runner and the job queue receiver.
/// Runs on a dedicated OS thread (the model is `Send`, not `Sync`).
///
/// `draft` is the ADR 0021 drafter for the I0 solo-spec path
/// ("spec-when-solo, migrate-on-admission", ADR 0032 L3 I0 — see the module
/// docs); `None` disables spec admissions entirely.
#[allow(clippy::too_many_arguments)] // the worker's full wiring, one call site
pub(crate) fn run_batched(
    mut runner: tritium_nn::ModelRunner,
    mut draft: Option<tritium_nn::ModelRunner>,
    eos: u32,
    slots: usize,
    pool_tokens: Option<usize>,
    mut job_rx: mpsc::Receiver<Job>,
    draining: Arc<AtomicBool>,
    worker_ready: Arc<AtomicBool>,
    phase: Arc<AtomicU8>,
    telemetry: Arc<WorkerTelemetry>,
) {
    struct PhaseGuard(Arc<AtomicU8>);
    impl Drop for PhaseGuard {
        fn drop(&mut self) {
            self.0.store(PHASE_IDLE, Ordering::Release);
        }
    }
    let _phase_guard = PhaseGuard(phase.clone());
    struct WorkerReadyGuard(Arc<AtomicBool>);
    impl Drop for WorkerReadyGuard {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _ready_guard = WorkerReadyGuard(worker_ready.clone());
    if slots == 0 {
        eprintln!("tritium-serve: --batch-slots must be >= 1");
        return;
    }
    let n_ctx = runner.config.n_ctx as usize;
    // Build the resident decoder + the slot pool up front; failures here are
    // fatal for the worker (the router will see a closed queue → 503s).
    let build = match pool_tokens {
        None => runner.new_batch(slots),
        Some(t) => {
            let pages = t.div_ceil(tritium_cuda::KV_PAGE_TOKENS);
            let Some(paged_tokens) = pages.checked_mul(tritium_cuda::KV_PAGE_TOKENS) else {
                eprintln!("tritium-serve: --kv-pool-tokens page capacity overflow");
                return;
            };
            let Some(dense_tokens) = slots.checked_mul(n_ctx) else {
                eprintln!("tritium-serve: --batch-slots dense capacity overflow");
                return;
            };
            eprintln!(
                "tritium-serve: paged KV — {pages} pages ({} tokens) shared by {slots} \
                 slots (dense would be {} tokens)",
                paged_tokens, dense_tokens,
            );
            runner.new_batch_paged(slots, pages)
        }
    };
    let mut batch = match build {
        Ok(b) => b,
        Err(tritium_nn::ResidentOpError::Unavailable) => {
            eprintln!("tritium-serve: --batch-slots needs the CUDA resident decoder");
            return;
        }
        Err(e) => {
            eprintln!("tritium-serve: batch pool alloc failed: {e}");
            return;
        }
    };
    // Whole-pool capacity in tokens (0 = dense/unlimited): requests that can
    // NEVER fit are errored loudly instead of parking forever.
    let Some(pool_cap_tokens) = batch.free_pages().checked_mul(tritium_cuda::KV_PAGE_TOKENS) else {
        eprintln!("tritium-serve: paged KV free capacity overflow");
        return;
    };
    telemetry.set_kv_pool(pool_cap_tokens, pool_cap_tokens);
    // The thread is alive from spawn, but it must not admit user work until
    // resident decoder and KV-pool initialization have succeeded.
    worker_ready.store(true, Ordering::Release);
    // A job that validated but found the page pool exhausted: retried before
    // pulling new work (FIFO), admitted once retirements free pages.
    let mut parked: Option<Job> = None;
    // C4: a BASTION tree session owns the runner's single-sequence KV. The
    // single-worker contract carries over verbatim: any chat admission
    // resets the runner and closes the session (clients see Conflict on the
    // next verify and re-open).
    let mut tree_open = false;
    let chunk = match prefill_chunk() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tritium-serve: {e}");
            return;
        }
    };
    let mut pool: Vec<Option<Active>> = (0..slots).map(|_| None).collect();
    let mut pending: Option<Pending> = None;
    // I0: the one solo speculative sequence. Mutually exclusive with
    // `tree_open` (both own the single-sequence KV) and — by the migration
    // rule — with any occupied pool slot or in-flight `pending`.
    let mut spec: Option<SpecSeq> = None;
    // Multi-slot spec pool state (module docs). Valid only across
    // CONSECUTIVE spec rounds — any lockstep round, drain, or emptied pool
    // drops it (re-entry re-enrolls from the slots' histories); enrollment
    // ownership is additionally pinned per Active id, so a stale pool can
    // never serve a row's next tenant.
    let mut multi: Option<SpecPool> = None;
    let mut multi_disabled = false;
    let mut multi_log = MultiFallbackLog::default();
    // Monotonic Active ids (enrollment ownership).
    let mut next_active_id: u64 = 0;

    loop {
        // Graceful drain (mirrors the single worker): cancel in-flight
        // requests; the router already 503s new ones. Keep looping so the
        // final channel close still exits cleanly.
        if draining.load(Ordering::Relaxed) {
            for (row, slot) in pool.iter_mut().enumerate() {
                if let Some(a) = slot.take() {
                    let _ = a.tx.try_send(GenEvent::Error("server draining".into()));
                    release_slot(&mut batch, row, telemetry.as_ref());
                }
            }
            retire_pending_prefill(
                &mut pending,
                &mut runner,
                &mut batch,
                telemetry.as_ref(),
                true,
            );
            // Draining fails the solo spec sequence like an active slot.
            if let Some(s) = spec.take() {
                let _ = s.tx.try_send(GenEvent::Error("server draining".into()));
            }
            multi = None; // drained slots take their enrollments with them
            tree_open = false;
            match take_parked_job(&mut parked, telemetry.as_ref()) {
                None => {}
                Some(Job::Generate { tx, .. }) => {
                    let _ = tx.try_send(GenEvent::Error("server draining".into()));
                }
                // I0 migration parks the displaced admission-type job, so a
                // tree-session open can be parked too (same Draining/503
                // classification a queue-drained open gets).
                Some(Job::OpenTreeSession { resp, .. }) => {
                    let _ = resp.send(Err(crate::generator::TreeOpError::Draining(
                        "server draining".into(),
                    )));
                }
                // Verifies never park; a new park site must extend this
                // drain arm rather than silently dropping a responder.
                Some(other) => unreachable!("non-admission job parked: {other:?}"),
            }
        }
        // A seat-starved job has already left the channel. Do not let a
        // disconnected client occupy the worker's FIFO parked slot until an
        // unrelated active generation finishes and a row becomes free.
        if parked.as_ref().is_some_and(job_client_gone) {
            let _ = take_parked_job(&mut parked, telemetry.as_ref());
        }
        // Admit into free slots: drain waiting jobs, block only when idle.
        // Cap admissions per pass: instantly-retiring jobs (errors, dead
        // channels) don't occupy a slot, and an unbounded pass would let a
        // flood of them starve stepping. Tree VERIFIES also consume this
        // budget (each is a bounded device tree-forward), so one pass costs
        // live streams at most slots*2 verifies. Gated on `pending.is_none()`: the
        // chunks own the single-sequence KV, so one admission prefills at a
        // time (a valid job below parks itself as `pending` and ends the
        // pass via this condition).
        let mut admissions = 0usize;
        let admission_cap = slots.saturating_mul(2);
        while pending.is_none() {
            if admissions >= admission_cap {
                break;
            }
            let free = pool.iter().position(Option::is_none);
            let any_live = pool.iter().any(Option::is_some);
            // A parked (seat- or page-starved) Generate is retried before
            // pulling new work — FIFO, nothing leapfrogs it. If it parks
            // again below, admission breaks, so this cannot spin. A parked
            // tree-session open (displaced by an I0 migration) needs no
            // seat and is retried unconditionally.
            let job = if parked.is_some() {
                let needs_seat = matches!(parked.as_ref(), Some(Job::Generate { .. }));
                if needs_seat && free.is_none() {
                    break; // still no seat; wait for a retirement
                }
                take_parked_job(&mut parked, telemetry.as_ref()).expect("checked is_some")
            } else if any_live || spec.is_some() {
                // C4: pull even when the pool is FULL — tree ops need no
                // seat, and a seatless Generate parks below instead of
                // gating the whole queue on slot availability. I0: pull
                // (never block) while the solo spec sequence decodes — an
                // admission-type job must be able to trigger migration.
                match job_rx.try_recv() {
                    Ok(j) => j,
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                }
            } else {
                phase.store(PHASE_IDLE, Ordering::Release);
                match job_rx.blocking_recv() {
                    Some(j) => j,
                    None => return,
                }
            };
            admissions += 1;
            // Jobs already queued when the drain started get errored BEFORE
            // paying their prefill (the router 503s new ones; this covers the
            // in-queue backlog).
            if draining.load(Ordering::Relaxed) {
                match job {
                    Job::Generate { tx, .. } => {
                        let _ = tx.try_send(GenEvent::Error("server draining".into()));
                    }
                    Job::OpenTreeSession { resp, .. } => {
                        let _ = resp.send(Err(crate::generator::TreeOpError::Draining(
                            "server draining".into(),
                        )));
                    }
                    Job::TreeVerify { resp, .. } => {
                        let _ = resp.send(Err(crate::generator::TreeOpError::Draining(
                            "server draining".into(),
                        )));
                    }
                }
                continue;
            }
            match job {
                Job::Generate {
                    req,
                    request_span,
                    queue_span,
                    mut accepted_at,
                    tx,
                } => {
                    if let Some(accepted_at) = accepted_at.take() {
                        let elapsed = accepted_at.elapsed();
                        telemetry.observe_queue_wait(elapsed);
                        queue_span.record("queue_wait_us", elapsed.as_micros() as u64);
                    }
                    drop(queue_span);
                    let prompt_len = req.prompt_tokens.len();
                    if prompt_len == 0 || prompt_len >= n_ctx.saturating_sub(1) {
                        let _ = tx.try_send(GenEvent::Error("prompt does not fit".into()));
                        continue;
                    }
                    let max_new = req.max_new.min(n_ctx - prompt_len - 1);
                    if max_new == 0 {
                        let _ = tx.try_send(GenEvent::Done(FinishReason::Length));
                        continue;
                    }
                    // I0 migrate-on-admission: a valid chat admission while
                    // the solo spec sequence is live migrates it into a slot
                    // FIRST (its continuation becomes the next Pending), and
                    // this job is parked to be retried right after — FIFO
                    // preserved, no stream re-emits a token. A spec-active
                    // worker never parks jobs elsewhere (spec requires an
                    // empty pool with free pages), so the park seat is free.
                    // A dead client must not evict the live spec sequence:
                    // migration costs the survivor a full-history re-prefill
                    // plus lockstep decode for its remaining tokens.
                    if spec.is_some() && tx.is_closed() {
                        continue;
                    }
                    if let Some(s) = spec.take() {
                        assert!(parked.is_none(), "parked job during solo spec");
                        pending = migrate_spec(
                            s,
                            &mut runner,
                            &mut batch,
                            draft.as_mut(),
                            &pool,
                            telemetry.as_ref(),
                        );
                        if pending.is_some() {
                            park_job(
                                &mut parked,
                                Job::Generate {
                                    req,
                                    request_span,
                                    queue_span: tracing::Span::none(),
                                    accepted_at,
                                    tx,
                                },
                                telemetry.as_ref(),
                            );
                            continue; // pending set: the continuation prefills first
                        }
                        // Defensive migration failure (stream already
                        // errored): fall through and admit this job normally.
                    }
                    // I0 spec admission: a greedy, logprob-free request
                    // arriving at a FULLY idle worker (no actives, no
                    // pending, no tree session — the single-sequence KV is
                    // free) decodes speculatively instead of burning a
                    // lockstep slot alone. Paged pools additionally require
                    // the whole footprint to fit the pool so a later
                    // migration's up-front reserve can never fail. Loud env
                    // rejects mirror the single worker's contract.
                    if draft.is_some()
                        && !pool.iter().any(Option::is_some)
                        && !tree_open
                        && matches!(req.sampling, Sampling::Greedy)
                        && req.logprobs.is_none()
                        && (!batch.paged()
                            || prompt_len
                                .checked_add(max_new)
                                .is_some_and(|needed| needed <= pool_cap_tokens))
                        && !tx.is_closed()
                    {
                        match (
                            DraftPolicy::from_env(),
                            draft_chain_from_env(),
                            SpecGovernor::from_env(),
                        ) {
                            (Ok(policy), Ok(chain), Ok(governor)) => {
                                // The admission reset (C4 serialized-ownership
                                // contract): the single-sequence KV is the
                                // prefill staging area, so taking it closes
                                // any session state. Fresh drafter too — its
                                // KV may hold a previous request.
                                runner.reset();
                                tree_open = false;
                                if let Some(d) = draft.as_mut() {
                                    d.reset();
                                }
                                pending = Some(Pending {
                                    done: 0,
                                    started_at: Instant::now(),
                                    request_span,
                                    goal: PendingGoal::SpecAdmit {
                                        tx,
                                        req,
                                        max_new,
                                        policy,
                                        governor,
                                        chain,
                                    },
                                });
                            }
                            (Err(e), ..) | (_, Err(e), _) | (.., Err(e)) => {
                                let _ = tx.try_send(GenEvent::Error(e.to_string()));
                            }
                        }
                        continue;
                    }
                    // Seat the request; a full pool parks it (FIFO — nothing
                    // is pulled past a parked job; it is retried as soon as a
                    // retirement frees a slot).
                    let Some(row) = pool.iter().position(Option::is_none) else {
                        park_job(
                            &mut parked,
                            Job::Generate {
                                req,
                                request_span,
                                queue_span: tracing::Span::none(),
                                accepted_at,
                                tx,
                            },
                            telemetry.as_ref(),
                        );
                        break;
                    };
                    // Paged KV (C3): reserve the request's whole footprint up
                    // front (v1 no-eviction policy — it can never be outgrown
                    // mid-decode). A request that can NEVER fit is a loud
                    // error; a pool that is merely full right now parks the
                    // job until a retirement frees pages.
                    if batch.paged() {
                        let Some(needed) = prompt_len.checked_add(max_new) else {
                            let _ = tx.try_send(GenEvent::Error(
                                "prompt + max_tokens token footprint overflow".into(),
                            ));
                            continue;
                        };
                        if needed > pool_cap_tokens {
                            let _ = tx.try_send(GenEvent::Error(format!(
                                "prompt + max_tokens = {needed} tokens exceeds the \
                                 --kv-pool-tokens capacity ({pool_cap_tokens})"
                            )));
                            continue;
                        }
                        if tx.is_closed() {
                            continue; // don't hold pages for a gone client
                        }
                        // Any reserve error parks. Today only exhaustion is
                        // reachable from here (row < slots, paged() checked,
                        // needed < max_ctx via the clamp above); if
                        // reserve_pages ever grows another error kind, match
                        // on it — a permanent error would park-loop.
                        if reserve_pages(&mut batch, row, needed, telemetry.as_ref()).is_err() {
                            park_job(
                                &mut parked,
                                Job::Generate {
                                    req,
                                    request_span,
                                    queue_span: tracing::Span::none(),
                                    accepted_at,
                                    tx,
                                },
                                telemetry.as_ref(),
                            );
                            break;
                        }
                    }
                    // Admission (C1): park the job as the one in-flight
                    // chunked prefill. Reset starts a fresh single-sequence
                    // KV; the chunks below accumulate into it — and closes
                    // any open tree session (the single-worker contract).
                    runner.reset();
                    tree_open = false;
                    pending = Some(Pending {
                        done: 0,
                        started_at: Instant::now(),
                        request_span,
                        goal: PendingGoal::Admit {
                            tx,
                            req,
                            max_new,
                            row,
                        },
                    });
                }
                // C4: a tree-session open is a prompt prefill on the
                // single-sequence KV — the same resource + chunk machine the
                // admissions use, so it interleaves with live slots instead
                // of stalling them.
                Job::OpenTreeSession { prompt, resp } => {
                    if resp.is_closed() {
                        continue;
                    }
                    if prompt.is_empty() || prompt.len() >= n_ctx {
                        let _ = resp.send(Err(crate::generator::TreeOpError::BadRequest(
                            "prompt is empty or exceeds the model context window".into(),
                        )));
                        continue;
                    }
                    // I0 migrate-on-admission: a session open is an
                    // admission-type job (it claims the single-sequence KV),
                    // so it displaces the solo spec sequence the same way a
                    // chat admission does — migrate first, park the open,
                    // retry it right after the continuation prefill. Same
                    // dead-client guard as the chat arm: an abandoned open
                    // must not evict the live spec sequence.
                    if spec.is_some() && resp.is_closed() {
                        continue;
                    }
                    if let Some(s) = spec.take() {
                        assert!(parked.is_none(), "parked job during solo spec");
                        pending = migrate_spec(
                            s,
                            &mut runner,
                            &mut batch,
                            draft.as_mut(),
                            &pool,
                            telemetry.as_ref(),
                        );
                        if pending.is_some() {
                            park_job(
                                &mut parked,
                                Job::OpenTreeSession { prompt, resp },
                                telemetry.as_ref(),
                            );
                            continue;
                        }
                    }
                    runner.reset();
                    tree_open = false;
                    pending = Some(Pending {
                        done: 0,
                        started_at: Instant::now(),
                        request_span: tracing::Span::none(),
                        goal: PendingGoal::TreeOpen { prompt, resp },
                    });
                }
                // C4: verifies are bounded single ops against the open
                // session's KV, run inline between batch steps. Ordering
                // makes stale verifies impossible: this arm only runs when
                // no prefill is in flight, and any admission since the open
                // flipped `tree_open` off.
                Job::TreeVerify {
                    tokens,
                    parents,
                    resp,
                } => {
                    if resp.is_closed() {
                        continue;
                    }
                    if !tree_open {
                        let _ = resp.send(Err(crate::generator::TreeOpError::Conflict(
                            "no open tree session (open one with /v1/tree/session; a chat \
                             completion closes it)"
                                .into(),
                        )));
                        continue;
                    }
                    let prior_phase = phase.swap(PHASE_DECODE, Ordering::AcqRel);
                    let out = runner
                        .tree_verify_greedy_cancellable(&tokens, &parents, &|| {
                            resp.is_closed() || draining.load(Ordering::Acquire)
                        })
                        .map_err(|e| match e {
                            tritium_nn::ResidentOpError::Unavailable => {
                                crate::generator::TreeOpError::Unsupported(
                                    "tree-verify needs the CUDA device-resident decoder".into(),
                                )
                            }
                            tritium_nn::ResidentOpError::Op(
                                tritium_spec::BackendError::InvalidInput(m),
                            ) => crate::generator::TreeOpError::BadRequest(m),
                            other => crate::generator::TreeOpError::Internal(other.to_string()),
                        });
                    phase.store(prior_phase, Ordering::Release);
                    match out {
                        Ok(Some(tokens)) => {
                            let _ = resp.send(Ok(tokens));
                        }
                        Ok(None) => {
                            if draining.load(Ordering::Acquire) {
                                let _ = resp.send(Err(crate::generator::TreeOpError::Draining(
                                    "server draining".into(),
                                )));
                            }
                        }
                        Err(error) => {
                            let _ = resp.send(Err(error));
                        }
                    }
                }
            }
        }

        // One prefill chunk for the pending admission (C1). Bounded work per
        // iteration: live slots get a decode step between chunks. With no
        // live slots the loop spins straight through the chunks back-to-back
        // (admission is skipped while `pending` is set, the step below while
        // the pool is empty).
        if let Some(p) = pending.as_mut() {
            if p.client_gone() {
                retire_pending_prefill(
                    &mut pending,
                    &mut runner,
                    &mut batch,
                    telemetry.as_ref(),
                    draining.load(Ordering::Acquire),
                );
            } else {
                phase.store(PHASE_PREFILL, Ordering::Release);
                let len = p.prompt().len();
                let end = p.done.saturating_add(chunk).min(len);
                let prefill_span = tracing::info_span!(
                    parent: &p.request_span,
                    "model.prefill.chunk",
                    chunk_tokens = end - p.done,
                );
                match prefill_span.in_scope(|| p.forward_chunk(&mut runner, chunk, &draining)) {
                    Err(e) => {
                        let p = pending.take().expect("pending checked above");
                        let row = p.row();
                        p.fail(e.to_string());
                        if let Some(row) = row {
                            release_slot(&mut batch, row, telemetry.as_ref());
                        }
                    }
                    Ok(None) => {
                        let draining_now = draining.load(Ordering::Acquire);
                        retire_pending_prefill(
                            &mut pending,
                            &mut runner,
                            &mut batch,
                            telemetry.as_ref(),
                            draining_now,
                        );
                        if draining_now {
                            // Drain all peers before doing another decode.
                            continue;
                        }
                    }
                    Ok(Some(logits)) => {
                        p.done = end;
                        if p.done == len {
                            let prefill_elapsed = p.started_at.elapsed();
                            let p = pending.take().expect("pending checked above");
                            telemetry.observe_prefill(prefill_elapsed);
                            let request_span = p.request_span;
                            match p.goal {
                                // Prompt complete: adopt the KV rows into the
                                // reserved slot and activate. `logits` is the
                                // last token's — bit-identical to a monolithic
                                // prefill's (chunking preserves the per-row
                                // order), so the first sampled token keeps the
                                // single-sequence guarantee the G1 gate pins.
                                PendingGoal::Admit {
                                    tx,
                                    req,
                                    max_new,
                                    row,
                                } => {
                                    let adopt = (|| -> Result<(), String> {
                                        runner
                                            .adopt_into_batch_row(&mut batch, row, len)
                                            .map_err(|e| e.to_string())?;
                                        batch.set_position(row, len).map_err(|e| e.to_string())
                                    })();
                                    if let Err(e) = adopt {
                                        let _ = tx.try_send(GenEvent::Error(e));
                                        release_slot(&mut batch, row, telemetry.as_ref());
                                    } else {
                                        let Some(active_id) = next_active_id.checked_add(1) else {
                                            let _ = tx.try_send(GenEvent::Error(
                                                "active request id exhausted".into(),
                                            ));
                                            release_slot(&mut batch, row, telemetry.as_ref());
                                            continue;
                                        };
                                        next_active_id = active_id;
                                        let mut active = Active {
                                            tx,
                                            request_span,
                                            id: active_id,
                                            logprobs: req.logprobs,
                                            stop_eos: req.stop_eos,
                                            remaining: max_new,
                                            last_token: 0,
                                            salt: 0,
                                            sampling: req.sampling,
                                            history: req.prompt_tokens,
                                        };
                                        active.salt += 1;
                                        let mut adopted = false;
                                        if let Some(first) = sample(
                                            &logits,
                                            &active.sampling,
                                            (req_seed(&active.sampling), active.salt),
                                        ) {
                                            active.last_token = first;
                                            active.history.push(first);
                                            if emit(&mut active, first, eos, &logits) {
                                                pool[row] = Some(active);
                                                adopted = true;
                                            }
                                        } else {
                                            let _ = active
                                                .tx
                                                .try_send(GenEvent::Error("empty logits".into()));
                                        }
                                        if !adopted {
                                            release_slot(&mut batch, row, telemetry.as_ref());
                                        }
                                    }
                                }
                                // Session open complete: the greedy root goes
                                // back; the session now owns the single-seq
                                // KV (until the next admission resets it).
                                PendingGoal::TreeOpen { resp, .. } => {
                                    match tritium_nn::sample_greedy(&logits) {
                                        Some(root) => {
                                            tree_open = true;
                                            let _ = resp.send(Ok(root));
                                        }
                                        None => {
                                            let _ = resp.send(Err(
                                                crate::generator::TreeOpError::Internal(
                                                    "empty logits from prefill".into(),
                                                ),
                                            ));
                                        }
                                    }
                                }
                                // I0 spec admission complete: emit the
                                // prefill's greedy argmax as the first token
                                // (the Admit first-token pattern — the chunked
                                // prefill is bit-identical to the single
                                // worker's, so this token keeps the
                                // single-sequence guarantee) and install the
                                // SpecSeq, which owns the single-sequence KV
                                // until it finishes or migrates.
                                PendingGoal::SpecAdmit {
                                    tx,
                                    req,
                                    max_new,
                                    policy,
                                    governor,
                                    chain,
                                } => match tritium_nn::sample_greedy(&logits) {
                                    None => {
                                        let _ = tx.try_send(GenEvent::Error("empty logits".into()));
                                    }
                                    Some(first) => {
                                        let is_eos = req.stop_eos && first == eos;
                                        let last = is_eos || max_new == 1;
                                        let sent =
                                            tx.try_send(GenEvent::Token(first, None)).is_ok();
                                        if last {
                                            if sent {
                                                let reason = if is_eos {
                                                    FinishReason::Stop
                                                } else {
                                                    FinishReason::Length
                                                };
                                                let _ = tx.try_send(GenEvent::Done(reason));
                                            }
                                        } else if sent {
                                            let mut history = req.prompt_tokens.clone();
                                            history.push(first);
                                            let stats = crate::generator::spec_stats_enabled();
                                            spec = Some(SpecSeq {
                                                tx,
                                                request_span,
                                                history,
                                                emitted: 1,
                                                max_new,
                                                req,
                                                policy,
                                                governor,
                                                chain,
                                                draft_fed: Vec::new(),
                                                draft_pos: 0,
                                                stats,
                                                n_verify: 0,
                                                n_committed: 0,
                                                n_plain: 0,
                                                t_verify: std::time::Duration::ZERO,
                                                t_plain: std::time::Duration::ZERO,
                                            });
                                        }
                                        // !sent → client gone: drop the
                                        // stream; the stale single-seq KV is
                                        // reset by the next admission.
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }

        // I0: while the solo spec sequence is live, each iteration runs ONE
        // spec cycle (draft → tree-verify → commit) instead of a lockstep
        // step. The pool is empty by construction — spec only admits into an
        // idle worker and any later admission migrates it out first — so no
        // live slot is starved by the cycle. The queue is re-polled between
        // cycles (the admission pass above), which is what bounds an
        // incoming job's wait to one cycle.
        if let Some(mut s) = spec.take() {
            debug_assert!(pending.is_none(), "pending admission during solo spec");
            debug_assert!(
                !pool.iter().any(Option::is_some),
                "live slot during solo spec"
            );
            if s.tx.is_closed() {
                // Client gone mid-spec: retire silently (the stale
                // single-sequence KV is reset by the next admission).
                s.print_stats();
                continue;
            }
            phase.store(PHASE_DECODE, Ordering::Release);
            let d = draft.as_mut().expect("spec admission requires a drafter");
            let decode_started = Instant::now();
            let spec_span =
                tracing::info_span!(parent: &s.request_span, "model.speculative_decode");
            match spec_span.in_scope(|| spec_cycle(&mut runner, d, &mut s, eos, n_ctx, &draining)) {
                Ok(SpecOutcome::Continue) => spec = Some(s),
                Ok(SpecOutcome::Done) => s.print_stats(),
                Ok(SpecOutcome::Cancelled) => {
                    if draining.load(Ordering::Acquire) {
                        let _ = s.tx.try_send(GenEvent::Error("server draining".into()));
                    }
                    runner.reset();
                    d.reset();
                    s.print_stats();
                }
                Err(msg) => {
                    let _ = s.tx.try_send(GenEvent::Error(msg));
                }
            }
            telemetry.observe_decode(decode_started.elapsed());
            continue;
        }

        if !pool.iter().any(Option::is_some) {
            multi = None; // an emptied pool takes its enrollments with it
            continue; // nothing live (a pending admission loops straight back
            // to its next chunk; a fully idle pool back to the blocking recv)
        }

        // Multi-slot spec round (module docs): eligible when a drafter is
        // attached, no tree session is open, and EVERY live request is
        // greedy + logprob-free (v1 all-or-nothing pool — one non-eligible
        // admission puts everyone on lockstep until it retires). A pending
        // chunked prefill does NOT block rounds: spec rounds never touch the
        // single-sequence KV, so admissions interleave exactly as with
        // lockstep (C1). Any fallback runs a lockstep step this round —
        // target rows only ever hold committed tokens between rounds, so
        // the switch is seamless and lossless.
        phase.store(PHASE_DECODE, Ordering::Release);
        let multi_eligible = !multi_disabled
            && draft.is_some()
            && !tree_open
            && pool
                .iter()
                .flatten()
                .all(|a| matches!(a.sampling, Sampling::Greedy) && a.logprobs.is_none());
        if multi_eligible {
            let d = draft.as_mut().expect("eligibility requires a drafter");
            let decode_started = Instant::now();
            let batch_span = tracing::info_span!(
                parent: None,
                "model.batch.speculative_decode",
                active_requests = pool.iter().flatten().count(),
            );
            for active in pool.iter().flatten() {
                let context = active.request_span.context().span().span_context().clone();
                if context.is_valid() {
                    batch_span.add_link(context);
                }
            }
            let outcome = batch_span.in_scope(|| {
                multi_spec_round(
                    &mut runner,
                    d,
                    &mut batch,
                    &mut multi,
                    &mut pool,
                    eos,
                    n_ctx,
                    &mut multi_log,
                    telemetry.as_ref(),
                    &|| draining.load(Ordering::Acquire),
                )
            });
            telemetry.observe_decode(decode_started.elapsed());
            match outcome {
                MultiOutcome::Ran => continue,
                MultiOutcome::Cancelled => {
                    multi = None;
                    continue;
                }
                MultiOutcome::Fallback => multi = None,
                MultiOutcome::Disable => {
                    multi = None;
                    multi_disabled = true;
                }
                // Whole pool suppressed (adaptive spec off): fall through to
                // the lockstep step below, KEEPING the pool + enrollments so
                // probe rounds re-enter cheaply (stale drafter watermarks
                // re-sync through the enrollment prefill at probe time).
                MultiOutcome::Lockstep => {}
            }
        } else {
            multi = None;
        }

        // One lockstep decode step. Free slots are marked dead (C2): the
        // kernels skip them entirely — no KV writes, no attention — and
        // their pad-token outputs are ignored. Liveness is re-derived from
        // the pool every step (self-healing; adoption/retirement need no
        // separate bookkeeping).
        let tokens: Vec<u32> = pool
            .iter()
            .map(|s| s.as_ref().map_or(0, |a| a.last_token))
            .collect();
        for (row, slot) in pool.iter().enumerate() {
            let _ = batch.set_live(row, slot.is_some());
        }
        let t_p = std::time::Instant::now();
        let decode_started = Instant::now();
        let batch_span = tracing::info_span!(
            parent: None,
            "model.batch.decode",
            active_requests = pool.iter().flatten().count(),
        );
        for active in pool.iter().flatten() {
            let context = active.request_span.context().span().span_context().clone();
            if context.is_valid() {
                batch_span.add_link(context);
            }
        }
        let step = batch_span.in_scope(|| runner.decode_batch_graph(&mut batch, &tokens));
        telemetry.observe_decode(decode_started.elapsed());
        let all_logits = match step {
            Ok(l) => {
                // Cost-model P_lockstep: one lockstep step's wall (the
                // batched floor's plain-step denominator).
                SPEC_COST.lockstep.record(t_p.elapsed().as_secs_f64() * 1e6);
                l
            }
            Err(e) => {
                for (row, slot) in pool.iter_mut().enumerate() {
                    if let Some(a) = slot.take() {
                        let _ = a.tx.try_send(GenEvent::Error(e.to_string()));
                        release_slot(&mut batch, row, telemetry.as_ref());
                    }
                }
                continue;
            }
        };
        for (row, slot) in pool.iter_mut().enumerate() {
            let Some(active) = slot.as_mut() else {
                continue;
            };
            active.salt += 1;
            let Some(tok) = sample(
                &all_logits[row],
                &active.sampling,
                (req_seed(&active.sampling), active.salt),
            ) else {
                if let Some(a) = slot.take() {
                    let _ = a.tx.try_send(GenEvent::Error("empty logits".into()));
                    release_slot(&mut batch, row, telemetry.as_ref());
                }
                continue;
            };
            active.last_token = tok;
            active.history.push(tok);
            if !emit(active, tok, eos, &all_logits[row]) {
                *slot = None;
                release_slot(&mut batch, row, telemetry.as_ref());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::generator::draft_greedy_tokens;
    #[test]
    fn drafter_query_cancels_inside_reconcile_and_host_fallback() {
        for reconcile in [false, true] {
            for chain in [false, true] {
                let trigger: ArmedTrigger = Arc::new(Mutex::new(None));
                let mut draft = tiny_runner(trigger.clone());
                let mut reference = tiny_runner(Arc::new(Mutex::new(None)));
                let history = if reconcile { vec![0, 1, 2, 3] } else { vec![3] };
                let mut fed = if reconcile { vec![6] } else { vec![] };
                let mut pos = usize::from(reconcile);
                if reconcile {
                    draft.forward(&[0, 6], &[0, 1]).unwrap();
                }
                let (tx, rx) = mpsc::channel::<GenEvent>(8);
                *trigger.lock().unwrap() = Some(Box::new(move || drop(rx)));
                assert!(
                    crate::generator::draft_greedy_tokens_cancellable(
                        &mut draft,
                        &mut fed,
                        &mut pos,
                        7,
                        &history,
                        3,
                        chain,
                        &|| tx.is_closed()
                    )
                    .is_none()
                );
                assert!(fed.is_empty());
                assert_eq!(pos, 0);
                assert!(draft.kv.iter().all(|cache| cache.len == 0));
                let mut reference_fed = Vec::new();
                let mut reference_pos = 0;
                let expected = draft_greedy_tokens(
                    &mut reference,
                    &mut reference_fed,
                    &mut reference_pos,
                    7,
                    &history,
                    3,
                    chain,
                );
                let recovered = crate::generator::draft_greedy_tokens_cancellable(
                    &mut draft,
                    &mut fed,
                    &mut pos,
                    7,
                    &history,
                    3,
                    chain,
                    &|| false,
                )
                .unwrap();
                assert_eq!(recovered, expected);
                assert_eq!(fed, reference_fed);
                assert_eq!(pos, reference_pos);
                assert_eq!(cache_bits(&draft), cache_bits(&reference));
            }
        }
    }

    #[test]
    fn drafter_native_queries_cancel_and_recover_every_checkpoint() {
        use std::cell::{Cell, RefCell};

        for chain in [false, true] {
            let Some(mut draft) = crate::test_support::tiny_cuda_runner(16) else {
                return;
            };
            let mut reference = crate::test_support::tiny_cuda_runner(16).unwrap();
            let history = [0, 1, 2, 3];
            reference.forward(&history[..3], &[0, 1, 2]).unwrap();
            let mut expected = Vec::new();
            let mut token = 3;
            for position in 3..6 {
                let logits = reference.forward(&[token], &[position]).unwrap();
                token = tritium_nn::sample_greedy(&logits).unwrap();
                expected.push(token);
            }
            let prepare = |draft: &mut ModelRunner| {
                draft.reset();
                draft.forward(&[0, 6], &[0, 1]).unwrap();
            };
            prepare(&mut draft);
            let mut fed = vec![6];
            let mut pos = 1;
            let polls = Cell::new(0);
            assert_eq!(
                draft_greedy_tokens_cancellable(
                    &mut draft,
                    &mut fed,
                    &mut pos,
                    u32::MAX,
                    &history,
                    3,
                    chain,
                    &|| {
                        polls.set(polls.get() + 1);
                        false
                    }
                )
                .unwrap(),
                expected
            );
            assert_eq!(
                crate::test_support::prefix_bytes(&mut draft),
                crate::test_support::prefix_bytes(&mut reference)
            );
            assert!(polls.get() > 10);
            for drain in [false, true] {
                for cancel_at in 1..=polls.get() {
                    prepare(&mut draft);
                    fed = vec![6];
                    pos = 1;
                    let before = crate::test_support::prefix_bytes(&mut draft);
                    let (tx, rx) = mpsc::channel::<GenEvent>(8);
                    let receiver = RefCell::new(Some(rx));
                    let draining = AtomicBool::new(false);
                    let count = Cell::new(0);
                    assert!(
                        draft_greedy_tokens_cancellable(
                            &mut draft,
                            &mut fed,
                            &mut pos,
                            u32::MAX,
                            &history,
                            3,
                            chain,
                            &|| {
                                count.set(count.get() + 1);
                                if count.get() == cancel_at {
                                    if drain {
                                        draining.store(true, Ordering::Release);
                                    } else {
                                        drop(receiver.borrow_mut().take());
                                    }
                                }
                                tx.is_closed() || draining.load(Ordering::Acquire)
                            }
                        )
                        .is_none()
                    );
                    assert_eq!(count.get(), cancel_at);
                    if cancel_at == 1 {
                        assert_eq!(pos, 1);
                        assert_eq!(fed, [6]);
                        assert_eq!(crate::test_support::prefix_bytes(&mut draft), before);
                    } else {
                        assert_eq!(pos, 0);
                        assert!(fed.is_empty());
                        assert_eq!(draft.resident_cuda().unwrap().unwrap().cache_len(), 0);
                    }
                    assert_eq!(
                        draft_greedy_tokens_cancellable(
                            &mut draft,
                            &mut fed,
                            &mut pos,
                            u32::MAX,
                            &history,
                            3,
                            chain,
                            &|| false
                        )
                        .unwrap(),
                        expected
                    );
                    assert_eq!(
                        crate::test_support::prefix_bytes(&mut draft),
                        crate::test_support::prefix_bytes(&mut reference)
                    );
                    assert!(draft.kv.iter().all(|cache| cache.len == 0));
                }
            }
        }
    }

    #[test]
    fn drafter_enrollment_cancellation_cannot_adopt_and_preserves_peer() {
        use std::cell::Cell;

        for delta in [false, true] {
            let Some(mut draft) = crate::test_support::tiny_cuda_runner(16) else {
                return;
            };
            let mut batch = draft.new_batch(2).unwrap();
            draft.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
            for row in 0..2 {
                let p = if row == 0 { 1 } else { 3 };
                draft.adopt_into_batch_row(&mut batch, row, p).unwrap();
                batch.set_position(row, p).unwrap();
            }
            let history = [0, 1, 2, 3];
            let prefix = delta.then_some(1);
            let peer = peer_bytes(&mut draft, &batch);
            let polls = Cell::new(0);
            assert_eq!(
                enroll_draft_row(&mut draft, &mut batch, 0, &history, prefix, &|| {
                    polls.set(polls.get() + 1);
                    false
                })
                .unwrap(),
                Some(())
            );
            assert!(polls.get() >= 5);
            for cancel_at in 1..=polls.get() {
                batch.set_position(0, 1).unwrap();
                let model = draft.resident_cuda().unwrap().unwrap();
                let before: Vec<_> = (0..2)
                    .flat_map(|layer| [false, true].map(move |v| (layer, v)))
                    .map(|(layer, value)| {
                        model
                            .debug_batch_kv_row(&batch, layer, 0, 0, value)
                            .unwrap()
                    })
                    .collect();
                let count = Cell::new(0);
                assert!(
                    enroll_draft_row(&mut draft, &mut batch, 0, &history, prefix, &|| {
                        count.set(count.get() + 1);
                        count.get() == cancel_at
                    })
                    .unwrap()
                    .is_none()
                );
                assert_eq!(batch.positions(), &[1, 3]);
                assert_eq!(peer_bytes(&mut draft, &batch), peer);
                let model = draft.resident_cuda().unwrap().unwrap();
                let after: Vec<_> = (0..2)
                    .flat_map(|layer| [false, true].map(move |v| (layer, v)))
                    .map(|(layer, value)| {
                        model
                            .debug_batch_kv_row(&batch, layer, 0, 0, value)
                            .unwrap()
                    })
                    .collect();
                assert_eq!(before, after);
                assert_eq!(
                    enroll_draft_row(&mut draft, &mut batch, 0, &history, prefix, &|| false)
                        .unwrap(),
                    Some(())
                );
                assert_eq!(batch.positions(), &[3, 3]);
                assert_eq!(peer_bytes(&mut draft, &batch), peer);
            }
        }
    }

    #[test]
    fn controlled_native_drafter_facade_distinguishes_unavailability() {
        let mut draft = tiny_runner(Arc::new(Mutex::new(Some(Box::new(|| {
            panic!("native facade must not start host work")
        })))));
        assert!(matches!(
            draft.decode_greedy_chain_cancellable(3, 0, 3, 7, &|| false),
            Err(tritium_nn::ResidentOpError::Unavailable)
        ));
        assert!(matches!(
            draft.decode_greedy_step_cancellable(3, 0, &|| false),
            Err(tritium_nn::ResidentOpError::Unavailable)
        ));
        assert!(
            draft
                .decode_greedy_chain_cancellable(8, 0, 0, 7, &|| true)
                .unwrap()
                .is_none()
        );
        assert!(
            draft
                .decode_greedy_step_cancellable(8, 0, &|| true)
                .unwrap()
                .is_none()
        );
        assert!(draft.kv.iter().all(|cache| cache.len == 0));
    }

    fn group_active(id: u64, tx: mpsc::Sender<GenEvent>, history: Vec<u32>) -> Active {
        Active {
            tx,
            request_span: tracing::Span::none(),
            id,
            sampling: Sampling::Greedy,
            logprobs: None,
            stop_eos: false,
            remaining: 16,
            last_token: *history.last().unwrap(),
            history,
            salt: 0,
        }
    }

    #[test]
    fn grouped_query_observes_only_selected_responses_and_drain() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let pool = vec![
            Some(group_active(0, tx0, vec![0])),
            Some(group_active(1, tx1, vec![1])),
        ];
        let draining = AtomicBool::new(false);
        assert!(!spec_group_cancelled(&pool, &[0, 1], &|| draining.load(Ordering::Acquire)));
        drop(rx1);
        assert!(!spec_group_cancelled(&pool, &[0], &|| draining.load(Ordering::Acquire)));
        assert!(spec_group_cancelled(&pool, &[0, 1], &|| draining.load(Ordering::Acquire)));
        draining.store(true, Ordering::Release);
        assert!(spec_group_cancelled(&pool, &[0], &|| draining.load(Ordering::Acquire)));
        draining.store(false, Ordering::Release);
        drop(rx0);
        assert!(spec_group_cancelled(&pool, &[0], &|| draining.load(Ordering::Acquire)));
    }

    #[test]
    fn grouped_response_query_cancels_submitted_native_work() {
        use std::cell::{Cell, RefCell};

        for context in [16, 12289] {
            for drain in [false, true] {
                let Some(mut runner) = crate::test_support::tiny_cuda_runner(context) else {
                    return;
                };
                runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
                let mut batch = runner.new_batch_paged(2, 2).unwrap();
                for row in 0..2 {
                    batch.reserve_pages(row, 16).unwrap();
                    runner.adopt_into_batch_row(&mut batch, row, 3).unwrap();
                    batch.set_position(row, 3).unwrap();
                }
                let before = peer_bytes(&mut runner, &batch);
                let free = batch.free_pages();
                let (tx0, rx0) = mpsc::channel(8);
                let receiver = RefCell::new(Some(rx0));
                let (tx1, mut rx1) = mpsc::channel(8);
                let pool = vec![
                    Some(group_active(0, tx0, vec![0, 1, 2, 3])),
                    Some(group_active(1, tx1, vec![0, 1, 2, 4])),
                ];
                let draining = AtomicBool::new(false);
                let polls = Cell::new(0);
                let trees = [(&[3, 4][..], &[-1, 0][..]), (&[4, 5][..], &[-1, 0][..])];
                assert!(
                    runner
                        .tree_verify_greedy_slots_cancellable(&mut batch, &[0, 1], &trees, &|| {
                            polls.set(polls.get() + 1);
                            // Fixture injection after device uploads, not entry-only.
                            // Invocation count remains intentionally non-contractual.
                            if polls.get() == 5 {
                                if drain {
                                    draining.store(true, Ordering::Release);
                                } else {
                                    drop(receiver.borrow_mut().take());
                                }
                            }
                            spec_group_cancelled(&pool, &[0, 1], &|| {
                                draining.load(Ordering::Acquire)
                            })
                        })
                        .unwrap()
                        .is_none()
                );
                assert_eq!(polls.get(), 5);
                assert_eq!(batch.positions(), &[3, 3]);
                assert_eq!(peer_bytes(&mut runner, &batch), before);
                assert_eq!(batch.free_pages(), free);
                assert!(matches!(
                    rx1.try_recv(),
                    Err(mpsc::error::TryRecvError::Empty)
                ));
                assert_eq!(pool[1].as_ref().unwrap().remaining, 16);
                assert!(runner.kv.iter().all(|cache| cache.len == 0));
                assert!(
                    runner
                        .tree_verify_greedy_slots(&mut batch, &[0, 1], &trees)
                        .is_ok()
                );
                assert_eq!(
                    batch.debug_tree_slots_graph_bucket_count() > 0,
                    context == 16
                );
            }
        }
    }

    #[test]
    fn grouped_spec_cancellation_preserves_live_peer_and_recovers() {
        for context in [16, 12289] {
            for drain in [false, true] {
                let Some(mut runner) = crate::test_support::tiny_cuda_runner(context) else {
                    return;
                };
                let mut draft = crate::test_support::tiny_cuda_runner(16).unwrap();
                let mut reference = crate::test_support::tiny_cuda_runner(context).unwrap();
                let mut batch = runner.new_batch_paged(2, 2).unwrap();
                let telemetry = WorkerTelemetry::default();
                let histories = [vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
                for (row, history) in histories.iter().enumerate() {
                    reserve_pages(&mut batch, row, 16, &telemetry).unwrap();
                    runner.reset();
                    runner.forward(&history[..3], &[0, 1, 2]).unwrap();
                    runner.adopt_into_batch_row(&mut batch, row, 3).unwrap();
                    batch.set_position(row, 3).unwrap();
                }
                let before = peer_bytes(&mut runner, &batch);
                let free = batch.free_pages();
                let pages = batch.debug_page_table_row(1);
                let (tx0, rx0) = mpsc::channel(64);
                let (tx1, mut rx1) = mpsc::channel(64);
                let mut rx0 = Some(rx0);
                let mut pool = vec![
                    Some(group_active(0, tx0, histories[0].clone())),
                    Some(group_active(1, tx1, histories[1].clone())),
                ];
                if !drain {
                    drop(rx0.take());
                }
                let draining = AtomicBool::new(drain);
                let mut multi = None;
                let mut log = MultiFallbackLog::default();
                assert!(matches!(
                    multi_spec_round(
                        &mut runner,
                        &mut draft,
                        &mut batch,
                        &mut multi,
                        &mut pool,
                        7,
                        context as usize,
                        &mut log,
                        &telemetry,
                        &|| draining.load(Ordering::Acquire)
                    ),
                    MultiOutcome::Cancelled
                ));
                // Known cancellation now avoids drafter enrollment entirely.
                // In-operation queries have separate checkpoint sweeps.
                assert!(multi.is_none());
                assert!(matches!(
                    rx1.try_recv(),
                    Err(mpsc::error::TryRecvError::Empty)
                ));
                assert_eq!(pool[1].as_ref().unwrap().history, histories[1]);
                assert_eq!(pool[1].as_ref().unwrap().remaining, 16);
                assert_eq!(batch.positions()[1], 3);
                assert_eq!(peer_bytes(&mut runner, &batch), before);
                assert_eq!(batch.debug_page_table_row(1), pages);
                assert_eq!(batch.free_pages(), free + usize::from(!drain));
                assert_eq!(
                    telemetry.kv_pool_releases_total.load(Ordering::Relaxed),
                    u64::from(!drain)
                );
                assert_eq!(pool[0].is_none(), !drain);
                // Mirror the caller's cancellation arm: discard overfed draft
                // enrollment and start the next tick, never same-tick lockstep.
                multi = None;
                draining.store(false, Ordering::Release);
                assert!(matches!(
                    multi_spec_round(
                        &mut runner,
                        &mut draft,
                        &mut batch,
                        &mut multi,
                        &mut pool,
                        7,
                        context as usize,
                        &mut log,
                        &telemetry,
                        &|| draining.load(Ordering::Acquire)
                    ),
                    MultiOutcome::Ran
                ));
                let mut emitted = Vec::new();
                while let Ok(event) = rx1.try_recv() {
                    match event {
                        GenEvent::Token(token, _) => emitted.push(token),
                        other => panic!("unexpected grouped event: {other:?}"),
                    }
                }
                assert!(!emitted.is_empty());
                reference.forward(&histories[1][..3], &[0, 1, 2]).unwrap();
                let mut pending = 7;
                for (offset, &token) in emitted.iter().enumerate() {
                    let logits = reference.forward(&[pending], &[3 + offset]).unwrap();
                    assert_eq!(token, sample(&logits, &Sampling::Greedy, (0, 0)).unwrap());
                    pending = token;
                }
                assert_eq!(
                    batch.debug_tree_slots_graph_bucket_count() > 0,
                    context == 16
                );
                assert_eq!(
                    telemetry
                        .kv_pool_release_failures_total
                        .load(Ordering::Relaxed),
                    0
                );
            }
        }
    }

    #[test]
    fn drafter_group_round_cancels_every_checkpoint_and_recovers_peer() {
        use std::cell::{Cell, RefCell};

        let histories = [vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let Some(mut runner) = crate::test_support::tiny_cuda_runner(16) else {
            return;
        };
        let mut draft = crate::test_support::tiny_cuda_runner(16).unwrap();
        let target_prefix = |runner: &mut ModelRunner, batch: &tritium_cuda::BatchKv| {
            let model = runner.resident_cuda().unwrap().unwrap();
            (0..2)
                .flat_map(|slot| {
                    (0..2).flat_map(move |layer| {
                        (0..3).flat_map(move |pos| {
                            [false, true].map(move |value| (slot, layer, pos, value))
                        })
                    })
                })
                .map(|(slot, layer, pos, value)| {
                    model
                        .debug_batch_kv_row(batch, layer, slot, pos, value)
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let prepare = |runner: &mut ModelRunner, telemetry: &WorkerTelemetry| {
            let mut batch = runner.new_batch_paged(2, 2).unwrap();
            for (row, history) in histories.iter().enumerate() {
                reserve_pages(&mut batch, row, 16, telemetry).unwrap();
                runner.reset();
                runner.forward(&history[..3], &[0, 1, 2]).unwrap();
                runner.adopt_into_batch_row(&mut batch, row, 3).unwrap();
                batch.set_position(row, 3).unwrap();
            }
            batch
        };
        let telemetry = WorkerTelemetry::default();
        let mut batch = prepare(&mut runner, &telemetry);
        let (tx0, _rx0) = mpsc::channel(64);
        let (tx1, _rx1) = mpsc::channel(64);
        let mut pool = vec![
            Some(group_active(0, tx0, histories[0].clone())),
            Some(group_active(1, tx1, histories[1].clone())),
        ];
        let polls = Cell::new(0);
        assert!(matches!(
            multi_spec_round(
                &mut runner,
                &mut draft,
                &mut batch,
                &mut None,
                &mut pool,
                7,
                16,
                &mut MultiFallbackLog::default(),
                &telemetry,
                &|| {
                    polls.set(polls.get() + 1);
                    false
                },
            ),
            MultiOutcome::Ran
        ));
        assert!(
            polls.get() > 20,
            "must reach enrollment, drafting and verification"
        );
        for drain in [false, true] {
            for cancel_at in 1..=polls.get() {
                let telemetry = WorkerTelemetry::default();
                let mut batch = prepare(&mut runner, &telemetry);
                let before = peer_bytes(&mut runner, &batch);
                let both_before = target_prefix(&mut runner, &batch);
                let free = batch.free_pages();
                let pages = batch.debug_page_table_row(1);
                let (tx0, rx0) = mpsc::channel(64);
                let receiver = RefCell::new(Some(rx0));
                let (tx1, mut rx1) = mpsc::channel(64);
                let mut pool = vec![
                    Some(group_active(0, tx0, histories[0].clone())),
                    Some(group_active(1, tx1, histories[1].clone())),
                ];
                let draining = AtomicBool::new(false);
                let count = Cell::new(0);
                let mut multi = None;
                let mut log = MultiFallbackLog::default();
                assert!(
                    matches!(
                        multi_spec_round(
                            &mut runner,
                            &mut draft,
                            &mut batch,
                            &mut multi,
                            &mut pool,
                            7,
                            16,
                            &mut log,
                            &telemetry,
                            &|| {
                                count.set(count.get() + 1);
                                if count.get() == cancel_at {
                                    if drain {
                                        draining.store(true, Ordering::Release);
                                    } else {
                                        drop(receiver.borrow_mut().take());
                                    }
                                }
                                draining.load(Ordering::Acquire)
                            },
                        ),
                        MultiOutcome::Cancelled
                    ),
                    "checkpoint {cancel_at}, drain={drain}"
                );
                assert_eq!(count.get(), cancel_at);
                assert_eq!(peer_bytes(&mut runner, &batch), before);
                if drain {
                    assert_eq!(target_prefix(&mut runner, &batch), both_before);
                    assert_eq!(batch.positions(), &[3, 3]);
                    assert_eq!(pool[0].as_ref().unwrap().history, histories[0]);
                    assert_eq!(pool[0].as_ref().unwrap().remaining, 16);
                }
                assert_eq!(batch.positions()[1], 3);
                assert_eq!(batch.debug_page_table_row(1), pages);
                assert_eq!(batch.free_pages(), free + usize::from(!drain));
                assert_eq!(pool[0].is_none(), !drain);
                assert_eq!(pool[1].as_ref().unwrap().history, histories[1]);
                assert_eq!(pool[1].as_ref().unwrap().remaining, 16);
                assert!(matches!(
                    rx1.try_recv(),
                    Err(mpsc::error::TryRecvError::Empty)
                ));
                assert_eq!(
                    telemetry.kv_pool_releases_total.load(Ordering::Relaxed),
                    u64::from(!drain)
                );
                // The real worker discards the overfed draft pool and returns
                // to retirement; recovery is a new tick, not same-tick fallback.
                multi = None;
                draining.store(false, Ordering::Release);
                assert!(matches!(
                    multi_spec_round(
                        &mut runner,
                        &mut draft,
                        &mut batch,
                        &mut multi,
                        &mut pool,
                        7,
                        16,
                        &mut log,
                        &telemetry,
                        &|| false,
                    ),
                    MultiOutcome::Ran
                ));
                let mut reference = crate::test_support::tiny_cuda_runner(16).unwrap();
                reference.forward(&histories[1][..3], &[0, 1, 2]).unwrap();
                let mut pending = 7;
                let mut emitted = 0;
                while let Ok(event) = rx1.try_recv() {
                    let GenEvent::Token(token, _) = event else {
                        panic!("unexpected recovery event: {event:?}");
                    };
                    let logits = reference.forward(&[pending], &[3 + emitted]).unwrap();
                    assert_eq!(token, sample(&logits, &Sampling::Greedy, (0, 0)).unwrap());
                    pending = token;
                    emitted += 1;
                }
                assert!(emitted > 0);
                assert!(batch.debug_tree_slots_graph_bucket_count() > 0);
                assert_eq!(
                    telemetry
                        .kv_pool_release_failures_total
                        .load(Ordering::Relaxed),
                    0
                );
            }
        }
    }

    #[test]
    fn drafter_solo_cycle_cancels_before_target_work() {
        for drain in [false, true] {
            for chain in [false, true] {
                let target_trigger: ArmedTrigger = Arc::new(Mutex::new(None));
                let draft_trigger: ArmedTrigger = Arc::new(Mutex::new(None));
                let mut runner = tiny_runner(target_trigger.clone());
                let mut draft = tiny_runner(draft_trigger.clone());
                runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
                let before = cache_bits(&runner);
                *target_trigger.lock().unwrap() = Some(Box::new(|| {
                    panic!("cancelled drafter must not enter target fallback/verify");
                }));
                let (tx, rx) = mpsc::channel(16);
                let mut receiver = Some(rx);
                let draining = Arc::new(AtomicBool::new(false));
                *draft_trigger.lock().unwrap() = Some(if drain {
                    let flag = draining.clone();
                    Box::new(move || flag.store(true, Ordering::Release))
                } else {
                    let receiver = receiver.take().unwrap();
                    Box::new(move || drop(receiver))
                });
                let history = vec![0, 1, 2, 3];
                let mut state = SpecSeq {
                    tx,
                    request_span: tracing::Span::none(),
                    history: history.clone(),
                    emitted: 0,
                    max_new: 8,
                    req: GenRequest {
                        prompt_tokens: vec![0],
                        max_new: 8,
                        logprobs: None,
                        sampling: Sampling::Greedy,
                        stop_eos: false,
                    },
                    policy: DraftPolicy::Adaptive { acc: 0.75 },
                    governor: SpecGovernor::Off,
                    chain,
                    draft_fed: Vec::new(),
                    draft_pos: 0,
                    stats: false,
                    n_verify: 0,
                    n_committed: 0,
                    n_plain: 0,
                    t_verify: std::time::Duration::ZERO,
                    t_plain: std::time::Duration::ZERO,
                };
                assert!(matches!(
                    spec_cycle(&mut runner, &mut draft, &mut state, 7, 16, &draining).unwrap(),
                    SpecOutcome::Cancelled
                ));
                assert!(
                    draft_trigger.lock().unwrap().is_none(),
                    "drafter work must enter"
                );
                assert!(
                    target_trigger.lock().unwrap().is_some(),
                    "target work must not enter"
                );
                assert_eq!(cache_bits(&runner), before);
                assert_eq!(state.history, history);
                assert_eq!(state.emitted, 0);
                assert_eq!(
                    (state.n_verify, state.n_plain, state.n_committed),
                    (0, 0, 0)
                );
                assert!(state.draft_fed.is_empty());
                assert_eq!(state.draft_pos, 0);
                assert!(draft.kv.iter().all(|cache| cache.len == 0));
                if let Some(receiver) = receiver.as_mut() {
                    assert!(matches!(
                        receiver.try_recv(),
                        Err(mpsc::error::TryRecvError::Empty)
                    ));
                }
            }
        }
    }

    #[test]
    fn solo_spec_cycle_cancels_inside_target_forward_without_publication() {
        for drain in [false, true] {
            for prefix in [0, 1] {
                let trigger: ArmedTrigger = Arc::new(Mutex::new(None));
                let mut runner = tiny_runner(trigger.clone());
                let mut reference = tiny_runner(Arc::new(Mutex::new(None)));
                let mut draft = tiny_runner(Arc::new(Mutex::new(Some(Box::new(|| {
                    panic!("budget-clamped cycle must not enter the drafter")
                })))));
                if prefix != 0 {
                    runner.forward(&[0], &[0]).unwrap();
                    reference.forward(&[0], &[0]).unwrap();
                }
                let before = cache_bits(&runner);
                let (tx, rx) = mpsc::channel(4);
                let mut receiver = Some(rx);
                let draining = Arc::new(AtomicBool::new(false));
                *trigger.lock().unwrap() = Some(if drain {
                    let flag = draining.clone();
                    Box::new(move || flag.store(true, Ordering::Release))
                } else {
                    let receiver = receiver.take().unwrap();
                    Box::new(move || drop(receiver))
                });
                let history = if prefix == 0 { vec![1] } else { vec![0, 1] };
                let mut state = SpecSeq {
                    tx,
                    request_span: tracing::Span::none(),
                    history: history.clone(),
                    emitted: 3,
                    max_new: 4,
                    req: GenRequest {
                        prompt_tokens: vec![0],
                        max_new: 4,
                        logprobs: None,
                        sampling: Sampling::Greedy,
                        stop_eos: false,
                    },
                    policy: DraftPolicy::Adaptive { acc: 0.75 },
                    governor: SpecGovernor::Off,
                    chain: true,
                    draft_fed: Vec::new(),
                    draft_pos: 0,
                    stats: false,
                    n_verify: 0,
                    n_committed: 0,
                    n_plain: 0,
                    t_verify: std::time::Duration::ZERO,
                    t_plain: std::time::Duration::ZERO,
                };
                assert!(matches!(
                    spec_cycle(&mut runner, &mut draft, &mut state, 7, 16, &draining).unwrap(),
                    SpecOutcome::Cancelled
                ));
                assert!(
                    trigger.lock().unwrap().is_none(),
                    "target projection must enter"
                );
                assert_eq!(cache_bits(&runner), before);
                assert_eq!(state.history, history);
                assert_eq!(state.emitted, 3);
                assert_eq!(
                    (state.n_plain, state.n_verify, state.n_committed),
                    (0, 0, 0)
                );
                assert_eq!(state.tx.is_closed(), !drain);
                if let Some(receiver) = receiver.as_mut() {
                    assert!(matches!(
                        receiver.try_recv(),
                        Err(mpsc::error::TryRecvError::Empty)
                    ));
                }
                let recovered = runner.forward(&[1], &[prefix]).unwrap();
                let expected = reference.forward(&[1], &[prefix]).unwrap();
                assert_eq!(
                    recovered
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(cache_bits(&runner), cache_bits(&reference));
            }
        }
    }

    #[test]
    fn solo_spec_cycle_skips_closed_or_draining_target_work() {
        for drain in [false, true] {
            let trigger = || -> ArmedTrigger {
                Arc::new(Mutex::new(Some(Box::new(|| {
                    panic!("cancelled solo cycle entered a projection")
                }))))
            };
            let mut runner = tiny_runner(trigger());
            let mut draft = tiny_runner(trigger());
            let (tx, rx) = mpsc::channel(4);
            let mut rx = Some(rx);
            if !drain {
                rx.take();
            }
            let req = GenRequest {
                prompt_tokens: vec![0],
                max_new: 4,
                logprobs: None,
                sampling: Sampling::Greedy,
                stop_eos: false,
            };
            let mut state = SpecSeq {
                tx,
                request_span: tracing::Span::none(),
                history: vec![0],
                emitted: 1,
                max_new: 4,
                req,
                policy: DraftPolicy::from_env().unwrap(),
                governor: SpecGovernor::from_env().unwrap(),
                chain: true,
                draft_fed: Vec::new(),
                draft_pos: 0,
                stats: false,
                n_verify: 0,
                n_committed: 0,
                n_plain: 0,
                t_verify: std::time::Duration::ZERO,
                t_plain: std::time::Duration::ZERO,
            };
            assert_eq!(state.tx.is_closed(), !drain);
            assert!(matches!(
                spec_cycle(
                    &mut runner,
                    &mut draft,
                    &mut state,
                    7,
                    16,
                    &AtomicBool::new(drain)
                )
                .unwrap(),
                SpecOutcome::Cancelled
            ));
            assert_eq!(state.history, vec![0]);
            assert_eq!(
                (state.n_verify, state.n_plain, state.n_committed),
                (0, 0, 0)
            );
            assert!(
                runner
                    .kv
                    .iter()
                    .chain(&draft.kv)
                    .all(|cache| cache.len == 0)
            );
        }
    }

    use super::*;
    use std::sync::Mutex;
    use tritium_nn::{
        DenseLinear, Mlp, ModelConfig, ModelRunner, ModelWeights, Projection, SwiGluMlp,
        TernaryLinear, TokenEmbedding, TransformerBlock,
    };
    use tritium_spec::{
        BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TernaryBackend, TernaryFormat,
    };

    type Trigger = Box<dyn FnOnce() + Send>;
    type ArmedTrigger = Arc<Mutex<Option<Trigger>>>;

    // Invoke the client-close/drain action inside the first real projection,
    // not from a mocked cancellation result or a timing-dependent sleep.
    struct ProjectionTrigger {
        cpu: tritium_cpu::CpuBackend,
        trigger: ArmedTrigger,
    }

    impl TernaryBackend for ProjectionTrigger {
        fn device_id(&self) -> &str {
            "pending-chunk-cpu-trigger"
        }

        fn capabilities(&self) -> DeviceCaps {
            self.cpu.capabilities()
        }

        fn upload_weights(
            &self,
            packed: &[u8],
            shape: GemmShape,
            format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            self.cpu.upload_weights(packed, shape, format)
        }

        fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
            self.cpu.mpgemm(parameters)?;
            let trigger = self.trigger.lock().unwrap().take();
            if let Some(trigger) = trigger {
                trigger();
            }
            Ok(())
        }
    }

    fn tiny_runner(trigger: ArmedTrigger) -> ModelRunner {
        let backend = ProjectionTrigger {
            cpu: tritium_cpu::CpuBackend::new(),
            trigger,
        };
        let dense = || Projection::Dense(DenseLinear::new_exact(vec![0.03125; 16], 4, 4).unwrap());
        let config = ModelConfig {
            arch: "llama".into(),
            n_layers: 2,
            n_embd: 4,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 4,
            n_ff: 4,
            n_ctx: 16,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let weights = ModelWeights {
            token_embd: TokenEmbedding::from_dense(
                (0..32).map(|i| (i as f32 - 16.0) / 64.0).collect(),
                8,
                4,
            )
            .unwrap(),
            vocab: 8,
            n_embd: 4,
            layers: (0..2)
                .map(|_| TransformerBlock {
                    attn_norm: vec![1.0; 4],
                    q_proj: Projection::Ternary(
                        TernaryLinear::new(&backend, &[tritium_core::Trit::ZERO; 16], 4, 4, 1.0)
                            .unwrap(),
                    ),
                    k_proj: dense(),
                    v_proj: dense(),
                    o_proj: dense(),
                    attn_sub_norm: Vec::new(),
                    q_bias: Vec::new(),
                    k_bias: Vec::new(),
                    v_bias: Vec::new(),
                    q_norm: Vec::new(),
                    k_norm: Vec::new(),
                    ffn_norm: vec![1.0; 4],
                    mlp: Mlp::SwiGlu(SwiGluMlp {
                        gate: dense(),
                        up: dense(),
                        down: dense(),
                    }),
                })
                .collect(),
            output_norm: vec![1.0; 4],
            lm_head: None,
        };
        ModelRunner::from_weights(config, weights, Box::new(backend))
    }

    fn pending(kind: usize, done: usize) -> (Pending, Trigger) {
        let req = GenRequest {
            prompt_tokens: vec![0, 1, 2, 3],
            max_new: 2,
            sampling: Sampling::Greedy,
            stop_eos: false,
            logprobs: None,
        };
        let (goal, close): (PendingGoal, Trigger) = if kind == 2 {
            let (resp, receiver) = tokio::sync::oneshot::channel();
            (
                PendingGoal::TreeOpen {
                    prompt: req.prompt_tokens,
                    resp,
                },
                Box::new(move || drop(receiver)),
            )
        } else {
            let (tx, receiver) = mpsc::channel(8);
            let goal = if kind == 0 {
                PendingGoal::Admit {
                    tx,
                    req,
                    max_new: 2,
                    row: 0,
                }
            } else {
                PendingGoal::SpecAdmit {
                    tx,
                    req,
                    max_new: 2,
                    policy: DraftPolicy::Adaptive { acc: 0.75 },
                    governor: SpecGovernor::Off,
                    chain: false,
                }
            };
            (goal, Box::new(move || drop(receiver)))
        };
        (
            Pending {
                done,
                started_at: Instant::now(),
                request_span: tracing::Span::none(),
                goal,
            },
            close,
        )
    }

    fn cache_bits(runner: &ModelRunner) -> Vec<(usize, Vec<u32>, Vec<u32>)> {
        runner
            .kv
            .iter()
            .map(|cache| {
                (
                    cache.len,
                    cache.k.iter().map(|value| value.to_bits()).collect(),
                    cache.v.iter().map(|value| value.to_bits()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn pending_chunk_cancels_inside_native_forward() {
        for kind in 0..3 {
            for drain in [false, true] {
                for done in [0, 2] {
                    let trigger: ArmedTrigger = Arc::new(Mutex::new(None));
                    let mut runner = tiny_runner(trigger.clone());
                    if done != 0 {
                        runner.forward(&[0, 1], &[0, 1]).unwrap();
                    }
                    let before = cache_bits(&runner);
                    let (pending, close) = pending(kind, done);
                    let mut keep_client = Some(close);
                    let draining = Arc::new(AtomicBool::new(false));
                    *trigger.lock().unwrap() = Some(if drain {
                        let flag = draining.clone();
                        Box::new(move || flag.store(true, Ordering::Release))
                    } else {
                        keep_client.take().unwrap()
                    });
                    assert!(
                        pending
                            .forward_chunk(&mut runner, 2, &draining)
                            .unwrap()
                            .is_none(),
                        "pending chunk must cancel inside native forward (goal={kind}, drain={drain}, prefix={done})"
                    );
                    assert_eq!(pending.client_gone(), !drain);
                    assert_eq!(pending.done, done);
                    assert_eq!(cache_bits(&runner), before);
                    let mut reference = tiny_runner(Arc::new(Mutex::new(None)));
                    if done != 0 {
                        reference.forward(&[0, 1], &[0, 1]).unwrap();
                    }
                    let tokens = &pending.prompt()[done..done + 2];
                    let positions = [done, done + 1];
                    let recovered = runner.forward(tokens, &positions).unwrap();
                    let expected = reference.forward(tokens, &positions).unwrap();
                    assert_eq!(
                        recovered
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(cache_bits(&runner), cache_bits(&reference));
                }
            }
        }
    }

    #[test]
    fn pending_chunk_preserves_uncancelled_output_and_runtime_errors() {
        for kind in 0..3 {
            let mut runner = tiny_runner(Arc::new(Mutex::new(None)));
            let mut reference = tiny_runner(Arc::new(Mutex::new(None)));
            let (pending, _keep_client) = pending(kind, 0);
            let actual = pending
                .forward_chunk(&mut runner, 2, &AtomicBool::new(false))
                .unwrap()
                .unwrap();
            let expected = reference.forward(&[0, 1], &[0, 1]).unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(cache_bits(&runner), cache_bits(&reference));
            assert_eq!(
                pending.done, 0,
                "only scheduler publication advances progress"
            );
        }
        let mut runner = tiny_runner(Arc::new(Mutex::new(None)));
        let (mut pending, _keep_client) = pending(0, 0);
        if let PendingGoal::Admit { req, .. } = &mut pending.goal {
            req.prompt_tokens[0] = 8;
        }
        assert!(
            pending
                .forward_chunk(&mut runner, 2, &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(pending.done, 0);
        assert!(runner.kv.iter().all(|cache| cache.len == 0));
    }

    #[test]
    fn pending_chunk_skips_closed_or_draining_work() {
        for kind in 0..3 {
            for drain in [false, true] {
                let trigger: ArmedTrigger = Arc::new(Mutex::new(Some(Box::new(|| {
                    panic!("already-cancelled chunk must not execute projections");
                }))));
                let mut runner = tiny_runner(trigger);
                let (pending, close) = pending(kind, 0);
                if !drain {
                    close();
                }
                assert_eq!(pending.client_gone(), !drain);
                assert!(
                    pending
                        .forward_chunk(&mut runner, 2, &AtomicBool::new(drain))
                        .unwrap()
                        .is_none()
                );
                assert!(runner.kv.iter().all(|cache| cache.len == 0));
            }
        }
    }

    fn peer_bytes(runner: &mut ModelRunner, batch: &tritium_cuda::BatchKv) -> Vec<Vec<u8>> {
        let model = runner.resident_cuda().unwrap().unwrap();
        (0..2)
            .flat_map(|layer| {
                (0..3).flat_map(move |row| {
                    [false, true]
                        .into_iter()
                        .map(move |value| (layer, row, value))
                })
            })
            .map(|(layer, row, value)| {
                model
                    .debug_batch_kv_row(batch, layer, 1, row, value)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn pending_retirement_releases_once_and_preserves_live_cuda_peer() {
        let Some(mut runner) = crate::test_support::tiny_cuda_runner(16) else {
            return;
        };
        let mut batch = runner.new_batch_paged(2, 2).unwrap();
        let telemetry = WorkerTelemetry::default();
        let capacity = kv_free_tokens(&batch);
        telemetry.set_kv_pool(capacity, capacity);
        reserve_pages(&mut batch, 0, 4, &telemetry).unwrap();
        reserve_pages(&mut batch, 1, 4, &telemetry).unwrap();
        runner.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
        runner.adopt_into_batch_row(&mut batch, 1, 3).unwrap();
        batch.set_position(1, 3).unwrap();
        batch.set_live(1, true).unwrap();
        let before = peer_bytes(&mut runner, &batch);
        let free_before = batch.free_pages();
        runner.reset();
        runner.forward(&[3, 4], &[0, 1]).unwrap();
        let (pending, _keep_client) = pending(0, 2);
        let mut pending = Some(pending);
        retire_pending_prefill(&mut pending, &mut runner, &mut batch, &telemetry, false);
        assert!(pending.is_none());
        assert_eq!(runner.resident_cuda().unwrap().unwrap().cache_len(), 0);
        assert_eq!(batch.free_pages(), free_before + 1);
        assert_eq!(batch.positions()[1], 3);
        assert_eq!(peer_bytes(&mut runner, &batch), before);
        assert_eq!(
            telemetry.kv_pool_reservations_total.load(Ordering::Relaxed),
            2
        );
        assert_eq!(telemetry.kv_pool_releases_total.load(Ordering::Relaxed), 1);
        assert_eq!(
            telemetry
                .kv_pool_release_failures_total
                .load(Ordering::Relaxed),
            0
        );
        // Repeating retirement must not release again or reset newer staging.
        runner.forward(&[6, 7], &[0, 1]).unwrap();
        retire_pending_prefill(&mut pending, &mut runner, &mut batch, &telemetry, true);
        assert_eq!(runner.resident_cuda().unwrap().unwrap().cache_len(), 2);
        assert_eq!(batch.free_pages(), free_before + 1);
        assert_eq!(peer_bytes(&mut runner, &batch), before);
        assert_eq!(telemetry.kv_pool_releases_total.load(Ordering::Relaxed), 1);
    }
}
