//! # tritium-core
//!
//! Foundation crate for Tritium. `no_std`-able and free of OS/runtime services.
//! It consumes the shared schema vocabulary and holds the *correctness ground truth* that every backend is
//! measured against:
//!
//! - [`Trit`] — a value constrained to `{-1, 0, +1}` (~1.585 bits).
//! - [`DType`] — the precision lattice (ternary, int, fp8/fp4, fp16/bf16/fp32).
//! - [`TernaryFormat`] — canonical packing schemes (`TQ1_0`, `TQ2_0`). The byte
//!   layout / pack code lives in `tritium-format`; this is just the shared name + bpw.
//! - [`ScaleGranularity`] + [`absmean`] — the scaling contract (BitNet b1.58 AbsMean).
//! - [`GemmShape`] — `(M, N, K)` problem geometry.
//! - [`reference_mpgemm`] — the slow, obviously-correct mixed-precision GEMM that
//!   every backend kernel must match within tolerance.
//!
//! Nothing here touches a GPU, a thread pool, or `std` (unless the `std` feature
//! is on). Backends depend on this crate; its only workspace dependency is
//! `tritium-schema`.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// v0.90 hardening: every public item must carry a doc comment.
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod additive;
mod dtype;
mod error;
mod feedback;
mod reference;
mod scale;
mod shape;
mod trit;

pub use additive::{AdditiveError, AdditiveView, apply_basis, reference_ternary_matmul};
pub use dtype::{DType, TernaryFormat};
pub use error::TritError;
#[cfg(feature = "std")]
pub use feedback::auto_feedback_decay;
pub use feedback::{FeedbackDecay, FeedbackDecayError};
pub use reference::{reference_conv1d, reference_fsq, reference_mpgemm};
pub use scale::{ScaleGranularity, absmean};
pub use shape::{ConvShape, GemmShape};
pub use trit::Trit;

/// log2(3): the information-theoretic floor for one ternary value, in bits.
pub const TERNARY_IDEAL_BITS: f32 = 1.584_962_5;
