//! Neural-net ops for the inference spine.
//!
//! Each op is a small, dependency-light fp32 routine (the ternary GEMM lives in
//! the backend, not here). The foundation set — [`rmsnorm`] and [`sample_greedy`]
//! — shipped in v0.10; RoPE, GQA attention, softmax, top-k/top-p sampling, and
//! the W1.58A8 activation quant land in the per-op waves (WF-1/WF-2) as
//! documented stubs today.

mod act_quant;
mod attention;
mod rmsnorm;
mod rope;
mod sampling;
mod softmax;

pub use act_quant::{QB, quantize_activation_int8};
pub use attention::gqa_attention;
pub use rmsnorm::{rmsnorm, rmsnorm_zero_centered};
pub use rope::{rope_apply, rope_apply_partial_neox};
pub use sampling::{
    sample_categorical, sample_greedy, sample_top_k, sample_top_p, truncated_top_k, truncated_top_p,
};
pub use softmax::softmax_rows;
