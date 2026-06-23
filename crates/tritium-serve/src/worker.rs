//! The decode worker: a dedicated OS thread owning the [`Generator`] by value
//! (the runner is `Send` but `&mut`-exclusive — one owning consumer needs no
//! lock and never moves the runner across threads after handoff). HTTP handlers
//! submit [`Job`]s over a bounded channel (backpressure); the worker serializes
//! generation and streams [`GenEvent`]s back per job.
//!
//! ## Robustness
//!
//! - **Non-blocking sends.** Tokens go out with `try_send`, so a slow or stalled
//!   SSE client (full per-job channel) cancels *its own* request and frees the
//!   worker — one slow reader can never park the single decode thread (a DoS the
//!   queue-level 429 cannot prevent).
//! - **Panic isolation.** Each job's generation runs under `catch_unwind`, so a
//!   panic in the runner/sampler ends that one request (as an error) instead of
//!   killing the worker and zombifying the server.
//! - **Liveness.** An [`AliveGuard`] clears a shared flag if the thread ever exits
//!   (panic or shutdown), so `/healthz` can report a dead worker.
//!
//! Cancellation is cooperative + per-step: a disconnect/drain is observed at the
//! next token boundary (a single in-flight `forward` is not interrupted). For the
//! bounded-latency CPU/GPU decode steps here that is acceptable; a future
//! interruptible-kernel path would tighten it.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use crate::generator::{FinishReason, GenRequest, Generator};

/// An event streamed from the worker to a request's response task.
#[derive(Debug, Clone)]
pub(crate) enum GenEvent {
    /// One decoded token (special tokens are dropped by the detok layer).
    Token(u32),
    /// Generation finished cleanly with this reason.
    Done(FinishReason),
    /// Generation failed (stringified [`crate::generator::GenError`] or a panic).
    Error(String),
}

/// A unit of work: a request plus the channel its events stream back on.
#[derive(Debug)]
pub(crate) struct Job {
    /// The generation request.
    pub(crate) req: GenRequest,
    /// Per-request event channel (dropped by the handler on client disconnect,
    /// which the worker observes as a send failure and treats as cancellation).
    pub(crate) tx: mpsc::Sender<GenEvent>,
}

/// Clears the liveness flag when the decode thread exits (normal or panic).
struct AliveGuard(Arc<AtomicBool>);
impl Drop for AliveGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// Spawn the decode thread, returning the bounded job-queue sender. A full queue
/// makes the handler's `try_send` return `Full` (mapped to HTTP 429). `worker_alive`
/// is set `true` here and cleared if the thread ever exits.
pub(crate) fn spawn_worker(
    mut generator: Box<dyn Generator>,
    draining: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    queue_cap: usize,
) -> mpsc::Sender<Job> {
    let (job_tx, mut job_rx) = mpsc::channel::<Job>(queue_cap.max(1));
    worker_alive.store(true, Ordering::Relaxed);
    std::thread::Builder::new()
        .name("tritium-serve-decode".to_owned())
        .spawn(move || {
            let _alive = AliveGuard(worker_alive);
            while let Some(Job { req, tx }) = job_rx.blocking_recv() {
                // Run the (panic-prone) generation under catch_unwind so one bad
                // job can't kill the worker. `final_reason` lives inside the
                // closure so its &mut borrow can't cross the unwind boundary.
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let mut final_reason = FinishReason::Stop;
                    let res = generator.generate(&req, &mut |step| {
                        if let Some(fr) = step.finish_reason {
                            final_reason = fr;
                        }
                        // try_send (never blocks): Full (slow client) or Closed
                        // (gone) cancels this request and frees the worker. Tokens
                        // delivered so far are an in-order prefix — no gaps.
                        tx.try_send(GenEvent::Token(step.token)).is_ok()
                            && !draining.load(Ordering::Relaxed)
                    });
                    (res, final_reason)
                }));
                match outcome {
                    Ok((Ok(()), final_reason)) => {
                        let _ = tx.try_send(GenEvent::Done(final_reason));
                    }
                    Ok((Err(e), _)) => {
                        let _ = tx.try_send(GenEvent::Error(e.to_string()));
                    }
                    Err(_panic) => {
                        let _ =
                            tx.try_send(GenEvent::Error("internal generation error".to_owned()));
                    }
                }
            }
        })
        .expect("spawn tritium-serve decode thread");
    job_tx
}
