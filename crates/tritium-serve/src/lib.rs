//! # tritium-serve — OpenAI-compatible HTTP/SSE inference server.
//!
//! Serves Tritium ternary models over the OpenAI `/v1/chat/completions` wire
//! protocol (non-streaming + SSE), `/v1/models`, `/healthz`, and `/readyz`. Because it is
//! OpenAI-wire-faithful, it is **LAMU-compatible for free**: point a LAMU
//! `local-llm` OpenAI backend at `http://<host>:<port>/v1` with `model` set to
//! whatever `GET /v1/models` reports.
//!
//! ## Feature gating
//!
//! The [`Generator`] seam, the OpenAI DTOs, and the passthrough tokenizer are
//! **always compiled** and runtime-free (no async deps), so the default workspace
//! build (and the cpu-only CI matrix) pulls in no tokio/axum. The HTTP server —
//! [`build_router`], the worker, and the SSE machinery — lives behind the
//! `serve` feature (mirroring tritium-cuda/wgpu's gate); the `tritium-serve`
//! binary has `required-features = ["serve"]`.
//!
//! ## Tokenizer
//!
//! Tritium has no in-repo BPE yet, so v0.80 ships [`IdPassthroughTokenizer`]
//! (whitespace-separated integer token IDs). Inject a `tokenizers`-crate-backed
//! [`tritium_nn::Tokenizer`] for real text input — that is the separate
//! tokenizer-seam task.
#![deny(missing_docs)]

pub mod dto;
pub mod generator;
pub mod tokenizer_passthrough;

#[cfg(feature = "serve")]
mod admission;
#[cfg(feature = "cuda")]
mod batch;
#[cfg(feature = "serve")]
mod router;
#[cfg(feature = "serve")]
mod sse;
#[cfg(feature = "serve")]
mod startup;
#[cfg(feature = "serve")]
mod worker;

pub use generator::{
    FinishReason, GenError, GenRequest, Generator, MockGenerator, RunnerGenerator, Sampling, Step,
    TreeOpError,
};
pub use tokenizer_passthrough::IdPassthroughTokenizer;

#[cfg(feature = "serve")]
pub use admission::{AdmissionPolicy, MAX_BEARER_TOKENS, PrincipalRateLimit};
#[cfg(feature = "serve")]
pub use router::{
    ChatTemplate, RequestLimits, ServeConfig, build_router, build_router_governed,
    build_router_production, build_router_with_limits,
};
#[cfg(feature = "cuda")]
pub use router::{
    build_router_batched, build_router_batched_governed, build_router_batched_with_limits,
};
#[cfg(feature = "serve")]
pub use startup::{
    AdmittedArtifactV1, AdmittedGeneratorV1, ProductionReadiness, StartupError, StartupReceiptV1,
    prepare_production_generator,
};
