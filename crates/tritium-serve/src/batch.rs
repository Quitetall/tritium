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
//! **Paged KV (batching P2, C3, ADR 0025)**: with `--kv-pool-tokens N`, the
//! per-slot dense arenas are replaced by a shared page pool. Admission
//! reserves `prompt + max_tokens` up front (never outgrown — the v1
//! no-eviction policy); a full pool parks the job until a retirement frees
//! pages (FIFO, retried before new work); a request that can never fit is a
//! loud error. Every retirement/abandonment path releases its row's pages.
//! Paging is bit-exact by construction (gated: paged == dense).
//!
//! Remaining phase-2 cost: free slots still burn their dense GEMM rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use crate::generator::{FinishReason, GenRequest, Sampling};
use crate::worker::{GenEvent, Job};

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

/// An admission mid-prefill: `done` prompt tokens are already in the runner's
/// single-sequence KV; the reserved pool row stays `None` until adoption (no
/// other admission can take it — admission is gated on `pending.is_none()`).
struct Pending {
    tx: mpsc::Sender<GenEvent>,
    req: GenRequest,
    /// Token budget after context clamping (validated at admission).
    max_new: usize,
    /// The pool row this request will occupy once adopted.
    row: usize,
    /// Prompt tokens already prefilled.
    done: usize,
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

/// The batched worker loop: owns the runner and the job queue receiver.
/// Runs on a dedicated OS thread (the model is `Send`, not `Sync`).
pub(crate) fn run_batched(
    mut runner: tritium_nn::ModelRunner,
    eos: u32,
    slots: usize,
    pool_tokens: Option<usize>,
    mut job_rx: mpsc::Receiver<Job>,
    draining: Arc<AtomicBool>,
) {
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
    let chunk = match prefill_chunk() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tritium-serve: {e}");
            return;
        }
    };
    let mut pool: Vec<Option<Active>> = (0..slots).map(|_| None).collect();
    let mut pending: Option<Pending> = None;

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
                let _ = p.tx.try_send(GenEvent::Error("server draining".into()));
                release_slot(&mut batch, p.row);
            }
            if let Some(Job::Generate { tx, .. }) = parked.take() {
                let _ = tx.try_send(GenEvent::Error("server draining".into()));
            }
        }
        // Admit into free slots: drain waiting jobs, block only when idle.
        // Cap admissions per pass: instantly-retiring jobs (errors, dead
        // channels) don't occupy a slot, and an unbounded pass would let a
        // flood of them starve stepping. Gated on `pending.is_none()`: the
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
            // A parked (pool-exhausted) job is retried before pulling new
            // work — FIFO — but only once a slot row is free to take (a full
            // pool must NOT take it: the row-pick below would break and drop
            // the job). If it parks again below, admission breaks, so this
            // cannot spin.
            let job = if free.is_some() && parked.is_some() {
                parked.take().expect("checked is_some")
            } else {
                match free {
                    None => break,
                    Some(_) if any_live => match job_rx.try_recv() {
                        Ok(j) => j,
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => return,
                    },
                    Some(_) => match job_rx.blocking_recv() {
                        Some(j) => j,
                        None => return,
                    },
                }
            };
            admissions += 1;
            let Some(row) = pool.iter().position(Option::is_none) else {
                break; // defensive: no free slot
            };
            // Jobs already queued when the drain started get errored BEFORE
            // paying their prefill (the router 503s new ones; this covers the
            // in-queue backlog).
            if draining.load(Ordering::Relaxed) {
                match job {
                    Job::Generate { tx, .. } => {
                        let _ = tx.try_send(GenEvent::Error("server draining".into()));
                    }
                    Job::OpenTreeSession { resp, .. } => {
                        let _ = resp.send(Err(crate::generator::TreeOpError::Unsupported(
                            "server draining".into(),
                        )));
                    }
                    Job::TreeVerify { resp, .. } => {
                        let _ = resp.send(Err(crate::generator::TreeOpError::Unsupported(
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
                    // KV; the chunks below accumulate into it.
                    runner.reset();
                    pending = Some(Pending {
                        tx,
                        req,
                        max_new,
                        row,
                        done: 0,
                    });
                }
                // Tree/spec sessions need exclusive ownership of the model's
                // single-sequence KV — incompatible with a live slot pool.
                Job::OpenTreeSession { resp, .. } => {
                    let _ = resp.send(Err(crate::generator::TreeOpError::Unsupported(
                        "tree sessions are unavailable with --batch-slots > 1".into(),
                    )));
                }
                Job::TreeVerify { resp, .. } => {
                    let _ = resp.send(Err(crate::generator::TreeOpError::Unsupported(
                        "tree verify is unavailable with --batch-slots > 1".into(),
                    )));
                }
            }
        }

        // One prefill chunk for the pending admission (C1). Bounded work per
        // iteration: live slots get a decode step between chunks. With no
        // live slots the loop spins straight through the chunks back-to-back
        // (admission is skipped while `pending` is set, the step below while
        // the pool is empty).
        if let Some(p) = pending.as_mut() {
            if p.tx.is_closed() {
                // Client gone mid-prefill: abandon the remaining chunks (and
                // free the reserved pages). The partial single-sequence KV is
                // dead weight until the next admission's reset.
                let row = p.row;
                pending = None;
                release_slot(&mut batch, row);
            } else {
                let len = p.req.prompt_tokens.len();
                let end = (p.done + chunk).min(len);
                let positions: Vec<usize> = (p.done..end).collect();
                match runner.forward(&p.req.prompt_tokens[p.done..end], &positions) {
                    Err(e) => {
                        let p = pending.take().expect("pending checked above");
                        let _ = p.tx.try_send(GenEvent::Error(e.to_string()));
                        release_slot(&mut batch, p.row);
                    }
                    Ok(logits) => {
                        p.done = end;
                        if p.done == len {
                            // Prompt complete: adopt the KV rows into the
                            // reserved slot and activate. `logits` is the last
                            // token's — bit-identical to a monolithic prefill's
                            // (chunking preserves the per-row order), so the
                            // first sampled token keeps the single-sequence
                            // guarantee the G1 gate pins.
                            let p = pending.take().expect("pending checked above");
                            let adopt = (|| -> Result<(), String> {
                                runner
                                    .adopt_into_batch_row(&mut batch, p.row, len)
                                    .map_err(|e| e.to_string())?;
                                batch.set_position(p.row, len).map_err(|e| e.to_string())
                            })();
                            if let Err(e) = adopt {
                                let _ = p.tx.try_send(GenEvent::Error(e));
                                release_slot(&mut batch, p.row);
                            } else {
                                let mut active = Active {
                                    tx: p.tx,
                                    logprobs: p.req.logprobs,
                                    stop_eos: p.req.stop_eos,
                                    remaining: p.max_new,
                                    last_token: 0,
                                    salt: 0,
                                    sampling: p.req.sampling,
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
                                        pool[p.row] = Some(active);
                                        adopted = true;
                                    }
                                } else {
                                    let _ = active
                                        .tx
                                        .try_send(GenEvent::Error("empty logits".into()));
                                }
                                if !adopted {
                                    release_slot(&mut batch, p.row);
                                }
                            }
                        }
                    }
                }
            }
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
