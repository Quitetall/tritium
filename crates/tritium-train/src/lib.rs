//! # tritium-train
//!
//! STE autograd + QAT for ternary BitNet models (ADR 0007). Reverse-mode over a
//! flat tape of explicit ops; each op is a hand-written forward + vector-Jacobian
//! product (`vjp`), validated by a finite-difference gradient check (Gate C).
//!
//! v0.50 skeleton: the [`gradcheck`] harness, the STE-quantize op, and the
//! ternary-matmul backward (grads w.r.t. activations and per-output scale, with
//! the trit-grad straight-through-estimated back to the latent f32 weight).
#![forbid(unsafe_code)]

pub mod gradcheck;
pub mod ops;
pub mod value;

pub use value::Shape;
