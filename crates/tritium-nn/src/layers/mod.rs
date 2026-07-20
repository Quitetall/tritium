//! Transformer building blocks: the ternary linear, the ReLU² MLP, and the
//! decoder block that wires attention + MLP together.
//!
//! These compose the ops in [`crate::ops`] with weights from
//! [`crate::model::weights`]. Real forward math lands in WF-3; today they are
//! documented stubs so the per-op waves can fill them in disjoint files.

mod dense;
mod linear;
mod mlp;
mod packed_salt;
mod projection;
mod qwen35_deltanet;
mod qwen35_full_attention;
mod salt;
mod salt_v2_host;
mod token_embedding;
mod transformer_block;

pub(crate) use packed_salt::{PackedSaltMatrix, PackedSaltMatrixBuilder};

pub use dense::DenseLinear;
pub use linear::TernaryLinear;
pub use mlp::{Mlp, Relu2Mlp, SwiGluMlp};
pub use projection::{Projection, ProjectionActivationMode};
pub use qwen35_deltanet::{Qwen35DeltaNet, Qwen35DeltaNetCache, Qwen35DeltaNetWeights};
pub use qwen35_full_attention::{
    Qwen35FullAttention, Qwen35FullAttentionCache, Qwen35FullAttentionWeights,
};
pub use salt::SaltLinear;
pub use salt_v2_host::HostSaltV2Linear;
pub use token_embedding::TokenEmbedding;
pub use transformer_block::{BlockDump, BlockScratch, TransformerBlock};
