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
//! Remaining phase-2 cost: free slots still burn their dense GEMM rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use tokio::sync::mpsc;

use crate::generator::{
    DraftPolicy, FinishReason, GenRequest, SPEC_COMMITTED, SPEC_VERIFIES, Sampling,
    draft_chain_from_env, draft_greedy_tokens,
};
use crate::worker::{GenEvent, Job, PHASE_DECODE, PHASE_IDLE, PHASE_PREFILL};

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
        /// Chained device-side drafting (from `TRITIUM_DRAFT_CHAIN`).
        chain: bool,
    },
}

impl Pending {
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
    sampling: Sampling,
    /// Top-k logprobs per token, when the request asked.
    logprobs: Option<usize>,
    stop_eos: bool,
    /// Tokens still allowed to be emitted.
    remaining: usize,
    /// The last sampled token — fed to the model on the next step.
    last_token: u32,
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
fn release_slot(batch: &mut tritium_cuda::BatchKv, row: usize) {
    if batch.paged() {
        let _ = batch.release_pages(row);
    }
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
    let last = is_eos || s.emitted + 1 >= s.max_new;
    let sent = s.tx.try_send(GenEvent::Token(token, None)).is_ok();
    s.emitted += 1;
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
) -> Result<SpecOutcome, String> {
    let pending = *s.history.last().expect("spec history holds the prompt");
    // Budget-clamped draft: total tree rows must fit the KV arena
    // (cache_len = history.len() - 1, the verifier needs
    // cache_len + 1 + d <= n_ctx, so d <= n_ctx - history.len()), and
    // committed tokens (<= drafts + 1) must fit the emission budget.
    let kv_room = n_ctx.saturating_sub(s.history.len());
    let budget = s.max_new - s.emitted; // >= 1 by the SpecSeq invariant
    let max_draft = s.policy.len().min(kv_room).min(budget.saturating_sub(1));
    let drafts = draft_greedy_tokens(
        draft,
        &mut s.draft_fed,
        &mut s.draft_pos,
        eos,
        &s.history,
        max_draft,
        s.chain,
    );

    if drafts.is_empty() {
        // Plain M=1 graph step (faster than a 1-node tree).
        let t0 = std::time::Instant::now();
        let pos = s.history.len() - 1;
        let logits = runner
            .forward(&[pending], &[pos])
            .map_err(|e| e.to_string())?;
        s.n_plain += 1;
        s.t_plain += t0.elapsed();
        let next = tritium_nn::sample_greedy(&logits).ok_or_else(|| "empty logits".to_owned())?;
        return Ok(emit_spec(s, next, eos));
    }

    let mut tokens = Vec::with_capacity(1 + drafts.len());
    tokens.push(pending);
    tokens.extend(&drafts);
    let parents: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
    let t0 = std::time::Instant::now();
    let committed = runner
        .tree_verify_greedy(&tokens, &parents)
        .map_err(|e| e.to_string())?;
    s.n_verify += 1;
    s.n_committed += committed.len();
    SPEC_VERIFIES.fetch_add(1, Ordering::Relaxed);
    SPEC_COMMITTED.fetch_add(committed.len() as u64, Ordering::Relaxed);
    s.t_verify += t0.elapsed();
    // committed = accepted drafts + the final token, so accepted drafts =
    // committed.len() - 1 (saturating mirrors the single worker's guard).
    s.policy
        .update(drafts.len(), committed.len().saturating_sub(1));
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
        if let Err(e) = batch.reserve_pages(row, s.history.len() + remaining) {
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
        goal: PendingGoal::Admit {
            tx: s.tx,
            req,
            max_new: remaining,
            row,
        },
    })
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
    phase: Arc<AtomicU8>,
) {
    struct PhaseGuard(Arc<AtomicU8>);
    impl Drop for PhaseGuard {
        fn drop(&mut self) {
            self.0.store(PHASE_IDLE, Ordering::Release);
        }
    }
    let _phase_guard = PhaseGuard(phase.clone());
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
            eprintln!(
                "tritium-serve: paged KV — {pages} pages ({} tokens) shared by {slots} \
                 slots (dense would be {} tokens)",
                pages * tritium_cuda::KV_PAGE_TOKENS,
                slots * n_ctx,
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
    let pool_cap_tokens = batch.free_pages() * tritium_cuda::KV_PAGE_TOKENS;
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

    loop {
        // Graceful drain (mirrors the single worker): cancel in-flight
        // requests; the router already 503s new ones. Keep looping so the
        // final channel close still exits cleanly.
        if draining.load(Ordering::Relaxed) {
            for (row, slot) in pool.iter_mut().enumerate() {
                if let Some(a) = slot.take() {
                    let _ = a.tx.try_send(GenEvent::Error("server draining".into()));
                    release_slot(&mut batch, row);
                }
            }
            if let Some(p) = pending.take() {
                let row = p.row();
                p.fail_draining();
                if let Some(row) = row {
                    release_slot(&mut batch, row);
                }
            }
            // Draining fails the solo spec sequence like an active slot.
            if let Some(s) = spec.take() {
                let _ = s.tx.try_send(GenEvent::Error("server draining".into()));
            }
            tree_open = false;
            match parked.take() {
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
        while pending.is_none() {
            if admissions >= slots * 2 {
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
                parked.take().expect("checked is_some")
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
                Job::Generate { req, tx } => {
                    let prompt_len = req.prompt_tokens.len();
                    if prompt_len == 0 || prompt_len + 1 >= n_ctx {
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
                        pending = migrate_spec(s, &mut runner, &mut batch, draft.as_mut(), &pool);
                        if pending.is_some() {
                            parked = Some(Job::Generate { req, tx });
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
                        && (!batch.paged() || prompt_len + max_new <= pool_cap_tokens)
                        && !tx.is_closed()
                    {
                        match (DraftPolicy::from_env(), draft_chain_from_env()) {
                            (Ok(policy), Ok(chain)) => {
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
                                    goal: PendingGoal::SpecAdmit {
                                        tx,
                                        req,
                                        max_new,
                                        policy,
                                        chain,
                                    },
                                });
                            }
                            (Err(e), _) | (_, Err(e)) => {
                                let _ = tx.try_send(GenEvent::Error(e.to_string()));
                            }
                        }
                        continue;
                    }
                    // Seat the request; a full pool parks it (FIFO — nothing
                    // is pulled past a parked job; it is retried as soon as a
                    // retirement frees a slot).
                    let Some(row) = pool.iter().position(Option::is_none) else {
                        parked = Some(Job::Generate { req, tx });
                        break;
                    };
                    // Paged KV (C3): reserve the request's whole footprint up
                    // front (v1 no-eviction policy — it can never be outgrown
                    // mid-decode). A request that can NEVER fit is a loud
                    // error; a pool that is merely full right now parks the
                    // job until a retirement frees pages.
                    if batch.paged() {
                        let needed = prompt_len + max_new;
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
                        if batch.reserve_pages(row, needed).is_err() {
                            parked = Some(Job::Generate { req, tx });
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
                        pending = migrate_spec(s, &mut runner, &mut batch, draft.as_mut(), &pool);
                        if pending.is_some() {
                            parked = Some(Job::OpenTreeSession { prompt, resp });
                            continue;
                        }
                    }
                    runner.reset();
                    tree_open = false;
                    pending = Some(Pending {
                        done: 0,
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
                        .tree_verify_greedy(&tokens, &parents)
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
                    let _ = resp.send(out);
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
                // Client gone mid-prefill: abandon the remaining chunks (and
                // free any reserved pages). The partial single-sequence KV is
                // dead weight until the next admission's reset.
                let row = p.row();
                pending = None;
                if let Some(row) = row {
                    release_slot(&mut batch, row);
                }
            } else {
                phase.store(PHASE_PREFILL, Ordering::Release);
                let len = p.prompt().len();
                let end = (p.done + chunk).min(len);
                let positions: Vec<usize> = (p.done..end).collect();
                match runner.forward(&p.prompt()[p.done..end], &positions) {
                    Err(e) => {
                        let p = pending.take().expect("pending checked above");
                        let row = p.row();
                        p.fail(e.to_string());
                        if let Some(row) = row {
                            release_slot(&mut batch, row);
                        }
                    }
                    Ok(logits) => {
                        p.done = end;
                        if p.done == len {
                            let p = pending.take().expect("pending checked above");
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
                                        release_slot(&mut batch, row);
                                    } else {
                                        let mut active = Active {
                                            tx,
                                            logprobs: req.logprobs,
                                            stop_eos: req.stop_eos,
                                            remaining: max_new,
                                            last_token: 0,
                                            salt: 0,
                                            sampling: req.sampling,
                                        };
                                        active.salt += 1;
                                        let mut adopted = false;
                                        if let Some(first) = sample(
                                            &logits,
                                            &active.sampling,
                                            (req_seed(&active.sampling), active.salt),
                                        ) {
                                            active.last_token = first;
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
                                            release_slot(&mut batch, row);
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
                                            let stats = std::env::var("TRITIUM_SPEC_STATS")
                                                .as_deref()
                                                == Ok("1");
                                            spec = Some(SpecSeq {
                                                tx,
                                                history,
                                                emitted: 1,
                                                max_new,
                                                req,
                                                policy,
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
            match spec_cycle(&mut runner, d, &mut s, eos, n_ctx) {
                Ok(SpecOutcome::Continue) => spec = Some(s),
                Ok(SpecOutcome::Done | SpecOutcome::Cancelled) => s.print_stats(),
                Err(msg) => {
                    let _ = s.tx.try_send(GenEvent::Error(msg));
                }
            }
            continue;
        }

        if !pool.iter().any(Option::is_some) {
            continue; // nothing live (a pending admission loops straight back
            // to its next chunk; a fully idle pool back to the blocking recv)
        }

        // One lockstep decode step. Free slots are marked dead (C2): the
        // kernels skip them entirely — no KV writes, no attention — and
        // their pad-token outputs are ignored. Liveness is re-derived from
        // the pool every step (self-healing; adoption/retirement need no
        // separate bookkeeping).
        phase.store(PHASE_DECODE, Ordering::Release);
        let tokens: Vec<u32> = pool
            .iter()
            .map(|s| s.as_ref().map_or(0, |a| a.last_token))
            .collect();
        for (row, slot) in pool.iter().enumerate() {
            let _ = batch.set_live(row, slot.is_some());
        }
        let step = runner.decode_batch_graph(&mut batch, &tokens);
        let all_logits = match step {
            Ok(l) => l,
            Err(e) => {
                for (row, slot) in pool.iter_mut().enumerate() {
                    if let Some(a) = slot.take() {
                        let _ = a.tx.try_send(GenEvent::Error(e.to_string()));
                        release_slot(&mut batch, row);
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
                    release_slot(&mut batch, row);
                }
                continue;
            };
            active.last_token = tok;
            if !emit(active, tok, eos, &all_logits[row]) {
                *slot = None;
                release_slot(&mut batch, row);
            }
        }
    }
}
