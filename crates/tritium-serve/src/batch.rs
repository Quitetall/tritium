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

use tokio::sync::mpsc;

use crate::generator::{
    DraftPolicy, FinishReason, GenRequest, SPEC_COMMITTED, SPEC_COST, SPEC_VERIFIES, Sampling,
    SpecGovernor, draft_chain_from_env, draft_greedy_tokens,
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
        /// Adaptive spec on/off governor (from `TRITIUM_SPEC_ADAPTIVE`).
        governor: SpecGovernor,
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
        draft_greedy_tokens(
            draft,
            &mut s.draft_fed,
            &mut s.draft_pos,
            eos,
            &s.history,
            max_draft,
            s.chain,
        )
    };
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
        let logits = runner
            .forward(&[pending], &[pos])
            .map_err(|e| e.to_string())?;
        s.n_plain += 1;
        let el = t0.elapsed();
        s.t_plain += el;
        SPEC_COST.plain.record(el.as_secs_f64() * 1e6); // cost-model P
        s.governor.on_plain_commit(1); // no-op unless suppressed
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
) -> MultiOutcome {
    let slots = pool.len();
    let rows: Vec<usize> = (0..slots).filter(|&r| pool[r].is_some()).collect();
    let n_live = rows.len();
    if n_live == 0 {
        return MultiOutcome::Fallback;
    }
    // Even k=1 chains cost 2 nodes per slot; past the cap the one-bucket
    // verify cannot hold everyone. (Realistic pools are far smaller; a
    // grouped-rounds variant is the measured follow-up if ever needed.)
    if 2 * n_live > TREE_NODE_CAP {
        if !log.cap {
            log.cap = true;
            eprintln!(
                "tritium-serve: multi-slot spec idle — {n_live} live slots need \
                 {} tree nodes > the {TREE_NODE_CAP}-node verify bucket; \
                 lockstep until the pool shrinks (logged once)",
                2 * n_live
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
        if p + 2 >= draft_ctx {
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
                    p + 1
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
        draft.reset();
        let positions: Vec<usize> = (0..p).collect();
        let enrolled = draft
            .forward(&a.history[..p], &positions)
            .map_err(|e| e.to_string())
            .and_then(|_| {
                draft
                    .adopt_into_batch_row(&mut mp.dbatch, r, p)
                    .map_err(|e| e.to_string())
            })
            .and_then(|()| mp.dbatch.set_position(r, p).map_err(|e| e.to_string()));
        if let Err(e) = enrolled {
            eprintln!("tritium-serve: multi-slot spec enrollment (row {r}): {e}");
            return MultiOutcome::Fallback;
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
        if let Err(e) = draft.draft_batch(&mut mp.dbatch, &feeds, 1, eos) {
            eprintln!("tritium-serve: multi-slot spec gap feed: {e}");
            return MultiOutcome::Fallback;
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
    let m_total = n_live + n_draft * k;
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
                    .min(n_ctx.saturating_sub(p + 1))
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
        match draft.draft_batch(&mut mp.dbatch, &feeds, k, eos) {
            Ok(c) => c,
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
        let mut tokens = Vec::with_capacity(1 + take);
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
        while end < trees.len() && m_sum + trees[end].1.len() <= TREE_NODE_CAP {
            m_sum += trees[end].1.len();
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
                let need = batch.positions()[r] + tokens.len();
                if batch.reserve_pages(r, need).is_err() {
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
        let outs = match runner.tree_verify_greedy_slots(batch, &group_rows, &group_trees) {
            Ok(o) => {
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
                        release_slot(batch, r);
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
                    release_slot(batch, r);
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
                let matched = 1 + (l - 1).min(fed[r] - 1);
                if mp.dbatch.set_position(r, p + matched).is_err() {
                    // Bad row/pos = corrupted bookkeeping: drop the
                    // enrollment (pool falls back after the loop) but KEEP
                    // emitting — the target side already committed.
                    mp.slots[r] = None;
                    drafter_broken = true;
                } else {
                    debug_assert_eq!(
                        mp.dbatch.positions()[r],
                        p + 1 + (l - 1).min(fed[r] - 1),
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
                release_slot(batch, r);
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
            multi = None; // drained slots take their enrollments with them
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
                                        next_active_id += 1;
                                        let mut active = Active {
                                            tx,
                                            id: next_active_id,
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
            match multi_spec_round(
                &mut runner,
                d,
                &mut batch,
                &mut multi,
                &mut pool,
                eos,
                n_ctx,
                &mut multi_log,
            ) {
                MultiOutcome::Ran => continue,
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
        let step = runner.decode_batch_graph(&mut batch, &tokens);
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
            active.history.push(tok);
            if !emit(active, tok, eos, &all_logits[row]) {
                *slot = None;
                release_slot(&mut batch, row);
            }
        }
    }
}
