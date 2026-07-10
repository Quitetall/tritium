//! Continuous batching, phase 1 (ADR 0020-era plan, zero new kernels).
//!
//! A fixed pool of `slots` sequences shares ONE `BatchKv` whose M=N decode
//! graph is captured once. Requests are admitted into free slots by running
//! the prompt through the OPTIMIZED single-sequence prefill and adopting the
//! resulting KV rows into the slot ([`CudaDecodeModel::copy_kv_into_batch_row`]);
//! every decode step advances all slots in lockstep (free slots are fed a pad
//! token whose output is ignored and whose position is pinned to 0 so it can
//! never overflow the arena). Per-slot sampling runs on the host against each
//! request's own parameters, reusing the plain samplers' truncated
//! distributions — the same per-row math the parity gates pin to the
//! single-sequence path.
//!
//! Known phase-1 costs (documented in the plan): admission prefill stalls the
//! whole batch for the prompt's prefill time, and free slots burn a row of
//! compute. Chunked prefill, per-row masks and paged KV are phase 2.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use crate::generator::{FinishReason, Sampling};
use crate::worker::{GenEvent, Job};

/// One live request occupying a slot.
struct Active {
    tx: mpsc::Sender<GenEvent>,
    sampling: Sampling,
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

/// Emit one token on a slot's channel. Returns `false` when the request is
/// finished (EOS/budget) or the client went away — the slot should retire.
fn emit(active: &mut Active, token: u32, eos: u32) -> bool {
    let is_eos = active.stop_eos && token == eos;
    let last = is_eos || active.remaining <= 1;
    let sent = active.tx.try_send(GenEvent::Token(token)).is_ok();
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
    let mut batch = {
        let Ok(Some(rm)) = runner.resident_cuda() else {
            eprintln!("tritium-serve: --batch-slots needs the CUDA resident decoder");
            return;
        };
        match rm.new_batch(slots) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("tritium-serve: batch pool alloc failed: {e}");
                return;
            }
        }
    };
    let mut pool: Vec<Option<Active>> = (0..slots).map(|_| None).collect();

    loop {
        // Graceful drain (mirrors the single worker): cancel in-flight
        // requests; the router already 503s new ones. Keep looping so the
        // final channel close still exits cleanly.
        if draining.load(Ordering::Relaxed) {
            for slot in pool.iter_mut() {
                if let Some(a) = slot.take() {
                    let _ = a.tx.try_send(GenEvent::Error("server draining".into()));
                }
            }
        }
        // Admit into free slots: drain waiting jobs, block only when idle.
        // Cap admissions per pass: instantly-retiring jobs (errors, dead
        // channels) don't occupy a slot, and an unbounded pass would let a
        // flood of them starve stepping (each admission costs a prefill).
        let mut admissions = 0usize;
        loop {
            if admissions >= slots * 2 {
                break;
            }
            let free = pool.iter().position(Option::is_none);
            let any_live = pool.iter().any(Option::is_some);
            let job = match free {
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
                    // Admission: optimized single-seq prefill, then adopt the
                    // KV rows into the slot.
                    runner.reset();
                    let positions: Vec<usize> = (0..prompt_len).collect();
                    let logits = match runner.forward(&req.prompt_tokens, &positions) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = tx.try_send(GenEvent::Error(e.to_string()));
                            continue;
                        }
                    };
                    let adopt = (|| -> Result<(), String> {
                        let rm = runner
                            .resident_cuda()
                            .map_err(|e| e.to_string())?
                            .ok_or("resident decoder vanished")?;
                        rm.copy_kv_into_batch_row(&mut batch, row, prompt_len)
                            .map_err(|e| e.to_string())?;
                        batch
                            .set_position(row, prompt_len)
                            .map_err(|e| e.to_string())
                    })();
                    if let Err(e) = adopt {
                        let _ = tx.try_send(GenEvent::Error(e));
                        continue;
                    }
                    let mut active = Active {
                        tx,
                        stop_eos: req.stop_eos,
                        remaining: max_new,
                        last_token: 0,
                        salt: 0,
                        sampling: req.sampling,
                    };
                    active.salt += 1;
                    let Some(first) = sample(
                        &logits,
                        &active.sampling,
                        (req_seed(&active.sampling), active.salt),
                    ) else {
                        let _ = active.tx.try_send(GenEvent::Error("empty logits".into()));
                        continue;
                    };
                    active.last_token = first;
                    if emit(&mut active, first, eos) {
                        pool[row] = Some(active);
                    }
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

        if !pool.iter().any(Option::is_some) {
            continue; // nothing live; loop back to the blocking recv
        }

        // One lockstep decode step. Free slots are fed token 0 at position 0
        // (their KV row-0 write is junk in a dead slot; outputs ignored).
        let tokens: Vec<u32> = pool
            .iter()
            .map(|s| s.as_ref().map_or(0, |a| a.last_token))
            .collect();
        for (row, slot) in pool.iter().enumerate() {
            if slot.is_none() {
                let _ = batch.set_position(row, 0);
            }
        }
        let step = {
            let Ok(Some(rm)) = runner.resident_cuda() else {
                eprintln!("tritium-serve: resident decoder vanished mid-batch");
                return;
            };
            rm.decode_batch_graph(&mut batch, &tokens)
        };
        let all_logits = match step {
            Ok(l) => l,
            Err(e) => {
                for slot in pool.iter_mut() {
                    if let Some(a) = slot.take() {
                        let _ = a.tx.try_send(GenEvent::Error(e.to_string()));
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
                }
                continue;
            };
            active.last_token = tok;
            if !emit(active, tok, eos) {
                *slot = None;
            }
        }
    }
}
