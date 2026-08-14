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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::generator::{FinishReason, GenError, GenRequest, Generator, TreeOpError};

pub(crate) const PHASE_IDLE: u8 = 0;
pub(crate) const PHASE_PREFILL: u8 = 1;
pub(crate) const PHASE_DECODE: u8 = 2;

const PHASE_DURATION_BUCKET_US: [u64; 8] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
];

/// Worker-owned phase timing. Each router gets one instance, so parallel
/// servers do not contaminate each other's receipts or `/metrics` output.
#[derive(Debug, Default)]
pub(crate) struct WorkerTelemetry {
    pub(crate) queue_wait_buckets: [AtomicU64; PHASE_DURATION_BUCKET_US.len()],
    pub(crate) queue_wait_sum_us: AtomicU64,
    pub(crate) queue_wait_count: AtomicU64,
    pub(crate) prefill_buckets: [AtomicU64; PHASE_DURATION_BUCKET_US.len()],
    pub(crate) prefill_sum_us: AtomicU64,
    pub(crate) prefill_count: AtomicU64,
    pub(crate) decode_buckets: [AtomicU64; PHASE_DURATION_BUCKET_US.len()],
    pub(crate) decode_sum_us: AtomicU64,
    pub(crate) decode_count: AtomicU64,
    /// Paged-KV pool capacity in logical tokens; zero for dense batches.
    pub(crate) kv_pool_capacity_tokens: AtomicU64,
    /// Paged-KV tokens currently free; zero for dense batches.
    pub(crate) kv_pool_free_tokens: AtomicU64,
    /// Successful upward page reservations observed by the batched worker.
    pub(crate) kv_pool_reservations_total: AtomicU64,
    /// Successful page releases observed by the batched worker.
    pub(crate) kv_pool_releases_total: AtomicU64,
}

impl WorkerTelemetry {
    /// Publish current paged-KV capacity/free gauges. Dense batches publish
    /// zero for both values instead of pretending their per-slot arenas are a
    /// shared pool.
    #[cfg(feature = "cuda")]
    pub(crate) fn set_kv_pool(&self, capacity_tokens: usize, free_tokens: usize) {
        let capacity = capacity_tokens.min(u64::MAX as usize) as u64;
        let free = free_tokens.min(capacity as usize).min(u64::MAX as usize) as u64;
        self.kv_pool_capacity_tokens
            .store(capacity, Ordering::Release);
        self.kv_pool_free_tokens.store(free, Ordering::Release);
    }

    /// Record one successful reservation and publish the post-operation free
    /// gauge. Counters advance only when the pool actually lost pages.
    #[cfg(feature = "cuda")]
    pub(crate) fn observe_kv_reservation(&self, before_free: usize, after_free: usize) {
        if after_free < before_free {
            self.kv_pool_reservations_total
                .fetch_add(1, Ordering::Relaxed);
        }
        self.kv_pool_free_tokens
            .store(after_free.min(u64::MAX as usize) as u64, Ordering::Release);
    }

    /// Record one successful release and publish the post-operation free
    /// gauge. Counters advance only when the pool actually gained pages.
    #[cfg(feature = "cuda")]
    pub(crate) fn observe_kv_release(&self, before_free: usize, after_free: usize) {
        if after_free > before_free {
            self.kv_pool_releases_total.fetch_add(1, Ordering::Relaxed);
        }
        self.kv_pool_free_tokens
            .store(after_free.min(u64::MAX as usize) as u64, Ordering::Release);
    }

    pub(crate) fn observe_queue_wait(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.queue_wait_buckets,
            &self.queue_wait_sum_us,
            &self.queue_wait_count,
        );
    }

    pub(crate) fn observe_prefill(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.prefill_buckets,
            &self.prefill_sum_us,
            &self.prefill_count,
        );
    }

    pub(crate) fn observe_decode(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.decode_buckets,
            &self.decode_sum_us,
            &self.decode_count,
        );
    }
}

fn observe_histogram(
    elapsed: Duration,
    buckets: &[AtomicU64; PHASE_DURATION_BUCKET_US.len()],
    sum_us: &AtomicU64,
    count: &AtomicU64,
) {
    let micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
    sum_us.fetch_add(micros, Ordering::Relaxed);
    count.fetch_add(1, Ordering::Relaxed);
    for (bucket, ceiling) in buckets.iter().zip(PHASE_DURATION_BUCKET_US) {
        if micros <= ceiling {
            bucket.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn record_backend_fault(faulted: &AtomicBool, faults: &AtomicU64, latch: bool) {
    faults.fetch_add(1, Ordering::Relaxed);
    if latch {
        faulted.store(true, Ordering::Release);
    }
}

pub(crate) struct WorkerSignals {
    pub(crate) draining: Arc<AtomicBool>,
    pub(crate) worker_alive: Arc<AtomicBool>,
    pub(crate) phase: Arc<AtomicU8>,
    pub(crate) backend_faulted: Arc<AtomicBool>,
    pub(crate) backend_faults: Arc<AtomicU64>,
    pub(crate) telemetry: Arc<WorkerTelemetry>,
    pub(crate) latch_backend_faults: bool,
}

/// An event streamed from the worker to a request's response task.
#[derive(Debug, Clone)]
pub(crate) enum GenEvent {
    /// One decoded token (special tokens are dropped by the detok layer),
    /// plus its logprob top-k when the request asked (sampled token first).
    Token(u32, Option<Vec<(u32, f32)>>),
    /// Generation finished cleanly with this reason.
    Done(FinishReason),
    /// Generation failed (stringified [`crate::generator::GenError`] or a panic).
    Error(String),
}

/// A unit of work for the decode thread.
#[derive(Debug)]
pub(crate) enum Job {
    /// A chat generation: request plus the channel its events stream back on.
    Generate {
        /// The generation request.
        req: GenRequest,
        /// Monotonic admission timestamp used for queue-wait accounting.
        accepted_at: Option<Instant>,
        /// Per-request event channel (dropped by the handler on client
        /// disconnect, which the worker observes as a send failure and treats
        /// as cancellation).
        tx: mpsc::Sender<GenEvent>,
    },
    /// BASTION tree-verify session ops (ADR 0014): serialized on the same
    /// queue as generations (the worker owns the one model), answered over a
    /// oneshot. A `Generate` job invalidates any open session (documented in
    /// the endpoint contract).
    OpenTreeSession {
        prompt: Vec<u32>,
        resp: tokio::sync::oneshot::Sender<Result<u32, TreeOpError>>,
    },
    TreeVerify {
        tokens: Vec<u32>,
        parents: Vec<i32>,
        resp: tokio::sync::oneshot::Sender<Result<Vec<u32>, TreeOpError>>,
    },
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
    signals: WorkerSignals,
    queue_cap: usize,
) -> mpsc::Sender<Job> {
    let WorkerSignals {
        draining,
        worker_alive,
        phase,
        backend_faulted,
        backend_faults,
        telemetry,
        latch_backend_faults,
    } = signals;
    let (job_tx, mut job_rx) = mpsc::channel::<Job>(queue_cap.max(1));
    worker_alive.store(true, Ordering::Relaxed);
    std::thread::Builder::new()
        .name("tritium-serve-decode".to_owned())
        .spawn(move || {
            let _alive = AliveGuard(worker_alive);
            struct PhaseGuard(Arc<AtomicU8>);
            impl Drop for PhaseGuard {
                fn drop(&mut self) {
                    self.0.store(PHASE_IDLE, Ordering::Release);
                }
            }
            while let Some(job) = job_rx.blocking_recv() {
                if latch_backend_faults && backend_faulted.load(Ordering::Acquire) {
                    match job {
                        Job::Generate { tx, .. } => {
                            let _ =
                                tx.try_send(GenEvent::Error("backend fault latched".to_owned()));
                        }
                        Job::OpenTreeSession { resp, .. } => {
                            let _ = resp.send(Err(TreeOpError::Internal(
                                "backend fault latched".to_owned(),
                            )));
                        }
                        Job::TreeVerify { resp, .. } => {
                            let _ = resp.send(Err(TreeOpError::Internal(
                                "backend fault latched".to_owned(),
                            )));
                        }
                    }
                    continue;
                }
                match job {
                    Job::Generate {
                        req,
                        accepted_at,
                        tx,
                    } => {
                        if draining.load(Ordering::Acquire) {
                            let _ = tx.try_send(GenEvent::Error("server draining".to_owned()));
                            continue;
                        }
                        if let Some(accepted_at) = accepted_at {
                            telemetry.observe_queue_wait(accepted_at.elapsed());
                        }
                        phase.store(PHASE_PREFILL, Ordering::Release);
                        let _phase = PhaseGuard(phase.clone());
                        let generation_started = Instant::now();
                        let mut first_decode_at = None;
                        // Run the (panic-prone) generation under catch_unwind so one
                        // bad job can't kill the worker. `final_reason` lives inside
                        // the closure so its &mut borrow can't cross the unwind
                        // boundary.
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            let mut final_reason = FinishReason::Stop;
                            let res = generator.generate(&req, &mut |step| {
                                if first_decode_at.is_none() {
                                    telemetry.observe_prefill(generation_started.elapsed());
                                    first_decode_at = Some(Instant::now());
                                }
                                phase.store(PHASE_DECODE, Ordering::Release);
                                if let Some(fr) = step.finish_reason {
                                    final_reason = fr;
                                }
                                // try_send (never blocks): Full (slow client) or
                                // Closed (gone) cancels this request and frees the
                                // worker. Tokens delivered so far are an in-order
                                // prefix — no gaps.
                                tx.try_send(GenEvent::Token(step.token, step.logprobs))
                                    .is_ok()
                                    && !draining.load(Ordering::Relaxed)
                            });
                            (res, final_reason)
                        }));
                        if let Some(first_decode_at) = first_decode_at {
                            telemetry.observe_decode(first_decode_at.elapsed());
                        } else {
                            telemetry.observe_prefill(generation_started.elapsed());
                        }
                        match outcome {
                            Ok((Ok(()), final_reason)) => {
                                let _ = tx.try_send(GenEvent::Done(final_reason));
                            }
                            Ok((Err(e), _)) => {
                                if matches!(&e, GenError::Backend(_)) {
                                    if latch_backend_faults {
                                        eprintln!("tritium-serve: backend fault latched: {e}");
                                    }
                                    record_backend_fault(
                                        &backend_faulted,
                                        &backend_faults,
                                        latch_backend_faults,
                                    );
                                }
                                let _ = tx.try_send(GenEvent::Error(e.to_string()));
                            }
                            Err(_panic) => {
                                record_backend_fault(
                                    &backend_faulted,
                                    &backend_faults,
                                    latch_backend_faults,
                                );
                                let _ = tx.try_send(GenEvent::Error(
                                    "internal generation error".to_owned(),
                                ));
                            }
                        }
                    }
                    Job::OpenTreeSession { prompt, resp } => {
                        if draining.load(Ordering::Acquire) {
                            let _ =
                                resp.send(Err(TreeOpError::Draining("server draining".to_owned())));
                            continue;
                        }
                        phase.store(PHASE_PREFILL, Ordering::Release);
                        let _phase = PhaseGuard(phase.clone());
                        let outcome =
                            catch_unwind(AssertUnwindSafe(|| generator.open_tree_session(&prompt)));
                        let result = match outcome {
                            Ok(r) => r,
                            Err(_panic) => Err(TreeOpError::Internal(
                                "internal tree-session error".to_owned(),
                            )),
                        };
                        if matches!(&result, Err(TreeOpError::Internal(_))) {
                            record_backend_fault(
                                &backend_faulted,
                                &backend_faults,
                                latch_backend_faults,
                            );
                        }
                        let _ = resp.send(result);
                    }
                    Job::TreeVerify {
                        tokens,
                        parents,
                        resp,
                    } => {
                        if draining.load(Ordering::Acquire) {
                            let _ =
                                resp.send(Err(TreeOpError::Draining("server draining".to_owned())));
                            continue;
                        }
                        phase.store(PHASE_DECODE, Ordering::Release);
                        let _phase = PhaseGuard(phase.clone());
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            generator.tree_verify(&tokens, &parents)
                        }));
                        let result = match outcome {
                            Ok(r) => r,
                            Err(_panic) => Err(TreeOpError::Internal(
                                "internal tree-verify error".to_owned(),
                            )),
                        };
                        if matches!(&result, Err(TreeOpError::Internal(_))) {
                            record_backend_fault(
                                &backend_faulted,
                                &backend_faults,
                                latch_backend_faults,
                            );
                        }
                        let _ = resp.send(result);
                    }
                }
            }
        })
        .expect("spawn tritium-serve decode thread");
    job_tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{Sampling, Step};

    struct BackendFail;
    struct PanicFail;
    struct CountThenFail(Arc<AtomicU64>);

    impl Generator for BackendFail {
        fn generate(
            &mut self,
            _req: &GenRequest,
            _on_step: &mut dyn FnMut(Step) -> bool,
        ) -> Result<(), GenError> {
            Err(GenError::Backend("device lost".to_owned()))
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn vocab(&self) -> usize {
            128_256
        }
    }

    impl Generator for PanicFail {
        fn generate(
            &mut self,
            _req: &GenRequest,
            _on_step: &mut dyn FnMut(Step) -> bool,
        ) -> Result<(), GenError> {
            panic!("worker fault injection")
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn vocab(&self) -> usize {
            128_256
        }
    }

    impl Generator for CountThenFail {
        fn generate(
            &mut self,
            _req: &GenRequest,
            _on_step: &mut dyn FnMut(Step) -> bool,
        ) -> Result<(), GenError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Err(GenError::Backend("device lost".to_owned()))
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn vocab(&self) -> usize {
            128_256
        }
    }

    fn fault(generator: Box<dyn Generator>, latch: bool) -> (bool, u64) {
        let draining = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
        let faulted = Arc::new(AtomicBool::new(false));
        let faults = Arc::new(AtomicU64::new(0));
        let telemetry = Arc::new(WorkerTelemetry::default());
        let jobs = spawn_worker(
            generator,
            WorkerSignals {
                draining,
                worker_alive: alive,
                phase,
                backend_faulted: faulted.clone(),
                backend_faults: faults.clone(),
                telemetry: telemetry.clone(),
                latch_backend_faults: latch,
            },
            1,
        );
        let (events, mut receiver) = mpsc::channel(2);
        jobs.try_send(Job::Generate {
            req: GenRequest {
                prompt_tokens: vec![1],
                max_new: 1,
                logprobs: None,
                sampling: Sampling::Greedy,
                stop_eos: true,
            },
            accepted_at: Some(Instant::now()),
            tx: events,
        })
        .unwrap();
        assert!(matches!(receiver.blocking_recv(), Some(GenEvent::Error(_))));
        (
            faulted.load(Ordering::Acquire),
            faults.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn production_backend_fault_latches_and_legacy_fault_only_counts() {
        assert_eq!(fault(Box::new(BackendFail), true), (true, 1));
        assert_eq!(fault(Box::new(PanicFail), true), (true, 1));
        assert_eq!(fault(Box::new(BackendFail), false), (false, 1));
    }

    #[test]
    fn production_latch_rejects_already_queued_work_without_backend_reentry() {
        let calls = Arc::new(AtomicU64::new(0));
        let faulted = Arc::new(AtomicBool::new(false));
        let telemetry = Arc::new(WorkerTelemetry::default());
        let jobs = spawn_worker(
            Box::new(CountThenFail(calls.clone())),
            WorkerSignals {
                draining: Arc::new(AtomicBool::new(false)),
                worker_alive: Arc::new(AtomicBool::new(true)),
                phase: Arc::new(AtomicU8::new(PHASE_IDLE)),
                backend_faulted: faulted.clone(),
                backend_faults: Arc::new(AtomicU64::new(0)),
                telemetry: telemetry.clone(),
                latch_backend_faults: true,
            },
            2,
        );
        let (first_tx, mut first_rx) = mpsc::channel(1);
        let (second_tx, mut second_rx) = mpsc::channel(1);
        let request = GenRequest {
            prompt_tokens: vec![1],
            max_new: 1,
            logprobs: None,
            sampling: Sampling::Greedy,
            stop_eos: true,
        };
        jobs.try_send(Job::Generate {
            req: request.clone(),
            accepted_at: Some(Instant::now()),
            tx: first_tx,
        })
        .unwrap();
        jobs.try_send(Job::Generate {
            req: request,
            accepted_at: Some(Instant::now()),
            tx: second_tx,
        })
        .unwrap();

        assert!(matches!(first_rx.blocking_recv(), Some(GenEvent::Error(_))));
        assert!(
            matches!(second_rx.blocking_recv(), Some(GenEvent::Error(message)) if message == "backend fault latched")
        );
        assert!(faulted.load(Ordering::Acquire));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
