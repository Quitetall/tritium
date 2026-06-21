//! # tritium-testkit
//!
//! The conformance keystone. A backend is "correct" in Tritium iff it reproduces
//! a committed set of [`ConformanceVector`]s — each generated from
//! [`tritium_core::reference_mpgemm`] — within a stated [`Tolerance`]. This crate
//! provides:
//!
//! - [`ConformanceVector`] + [`Tolerance`] — the case and its grading rule.
//! - [`generate_vectors`] — deterministic, dependency-free vector generation
//!   (xorshift PRNG) plus a fixed boundary set of degenerate shapes.
//! - [`save_vectors`] / [`load_vectors`] — JSONL persistence (one object per
//!   line), so the suite is committed and versioned.
//! - [`run_conformance`] — replay a vector set against any
//!   [`tritium_spec::TernaryBackend`] and collect a [`Report`].
//!
//! ## Packing vs scaling
//!
//! Weights are packed host-side with `tritium-format`. The block scale baked into
//! the packed bytes is fixed to `1.0`; the per-output-channel [`ConformanceVector::scales`]
//! are applied separately inside `mpgemm`. The two are kept orthogonal on purpose
//! (see [`run_conformance`]) so a backend that double-applies, or that drops one,
//! is caught.
//!
//! ## Example
//!
//! ```
//! use tritium_testkit::{generate_vectors, run_conformance, save_vectors, load_vectors, Tolerance};
//! # use tritium_testkit::reference_backend_for_doctest as some_backend;
//! let vectors = generate_vectors(0xC0FFEE, 16);
//! let report = run_conformance(&some_backend(), &vectors, Tolerance::default());
//! assert!(report.is_ok(), "{} cases failed", report.failed.len());
//! ```
#![forbid(unsafe_code)]
// v0.90 hardening: every public item must carry a doc comment.
#![deny(missing_docs)]

mod frozen;
mod generate;
mod jsonl;
mod reference_backend;
mod runner;
mod vector;

pub use frozen::{
    FROZEN_COUNT, FROZEN_SEED, VECTOR_SET_VERSION, frozen_vectors, frozen_vectors_path,
};
pub use generate::generate_vectors;
pub use jsonl::{JsonlError, load_vectors, save_vectors};
pub use runner::{FailedCase, FailureReason, Report, run_conformance};
pub use vector::{ConformanceVector, Tolerance};

// The reference backend is part of the public surface only so this crate's own
// doctests have a known-good backend to run against; it is hidden from docs.
#[doc(hidden)]
pub use reference_backend::{ReferenceBackend, reference_backend_for_doctest};
