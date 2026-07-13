//! # tritium-nn
//!
//! The inference layer that turns the ternary mpGEMM primitive into a running
//! model. It sits above [`tritium_spec::TernaryBackend`]: ternary linear layers
//! call `backend.mpgemm`, while norms, softmax, RoPE, and sampling run in fp32.
//!
//! v0.20 scope (see ADR 0004): RMSNorm, RoPE, GQA attention, KV cache, sampling,
//! a transformer block, and a [`ModelConfig`]-driven model runner that loads
//! **BitNet b1.58 2B4T** from GGUF and generates tokens. This module is the
//! foundation (config + the simplest ops); the remaining ops + runner land in the
//! per-op and integration waves.
//!
//! No `unsafe` here — the numerics are plain Rust; the backend owns the SIMD/GPU.
#![forbid(unsafe_code)]

mod config;
mod error;
mod kv_cache;
mod layers;
mod model;
mod ops;
mod teacher_cache;
mod tensor;

pub use config::{ArchSpec, MlpKind, ModelConfig};
pub use error::NnError;
#[cfg(feature = "cuda")]
pub use error::ResidentOpError;
pub use kv_cache::KvCache;
pub use layers::{
    BlockDump, BlockScratch, DenseLinear, Mlp, Projection, Relu2Mlp, SwiGluMlp, TernaryLinear,
    TransformerBlock,
};
#[cfg(feature = "tokenizer")]
pub use model::GgufBpeTokenizer;
pub use model::{ForwardDump, LayerWeights, ModelRunner, ModelWeights, Tokenizer};
pub use ops::{
    QB, gqa_attention, quantize_activation_int8, rmsnorm, rope_apply, sample_categorical,
    sample_greedy, sample_top_k, sample_top_p, softmax_rows, truncated_top_k, truncated_top_p,
};
pub use teacher_cache::{
    TeacherCacheError, TeacherCacheReader, TeacherCacheWriter, hash_teacher_corpus,
    hash_teacher_weights,
};
pub use tensor::f16_bytes_to_f32;
