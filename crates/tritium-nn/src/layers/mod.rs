//! Transformer building blocks: the ternary linear, the ReLU² MLP, and the
//! decoder block that wires attention + MLP together.
//!
//! These compose the ops in [`crate::ops`] with weights from
//! [`crate::model::weights`]. Real forward math lands in WF-3; today they are
//! documented stubs so the per-op waves can fill them in disjoint files.

mod dense;
mod linear;
mod mlp;
mod projection;
mod transformer_block;

pub use dense::DenseLinear;
pub use linear::TernaryLinear;
pub use mlp::{Mlp, Relu2Mlp, SwiGluMlp};
pub use projection::Projection;
pub use transformer_block::{BlockDump, BlockScratch, TransformerBlock};
