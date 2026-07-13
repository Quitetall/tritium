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
//! (AdamW), the bit-exact training [`checkpoint`], and [`lora`] adapters on a frozen base.
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod data;
pub mod dcp;
pub mod dist;
pub mod fisher;
pub mod fsdp;
pub mod gemm;
pub mod gradcheck;
pub mod grow;
pub mod lora;
pub mod lr;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tape;
pub mod value;

pub use checkpoint::{Checkpoint, CheckpointError, LeafCheckpoint};
pub use data::{Cursor, DataSampler};
pub use dcp::{DcpError, DistCheckpoint};
pub use dist::{DistError, ProcessGroup, ReduceOp, SimProcessGroup};
pub use fisher::FisherAccumulator;
pub use fsdp::{FlatShardError, FlatShardPlan};
pub use gemm::TrainGemm;
pub use grow::{GrowError, Net2WiderPlan, QualityBytesPoint, QualityBytesReport};
pub use lora::Lora;
pub use lr::LrSchedule;
pub use optim::{AdamState, AdamW, Muon, MuonState, Optimizer, newton_schulz};
pub use tape::{Tape, ValueId};
pub use value::Shape;
