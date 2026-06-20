//! # tritium-train
//!
//! STE autograd + QAT for ternary BitNet models (ADR 0007). Reverse-mode over a
//! flat tape of explicit ops; each op is a hand-written forward + vector-Jacobian
//! product (`vjp`), validated by a finite-difference gradient check (Gate C).
//!
//! v0.50: the [`gradcheck`] harness, the STE-quantize and ternary-matmul ops, the
//! CPU op set (bias, squared-ReLU, MSE / softmax-cross-entropy, element-wise
//! add/mul), the reverse-mode [`tape`] that composes them into a differentiable QAT
//! graph (grads w.r.t. activations, weights, scale, and bias), the [`optim`]izer
//! (AdamW), and the bit-exact training [`checkpoint`].
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod gradcheck;
pub mod ops;
pub mod optim;
pub mod tape;
pub mod value;

pub use checkpoint::{Checkpoint, CheckpointError, LeafCheckpoint};
pub use optim::{AdamState, AdamW, Optimizer};
pub use tape::{Tape, ValueId};
pub use value::Shape;
