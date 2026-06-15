//! One decoder block: pre-norm attention + pre-norm ReLU² MLP, both residual.
//!
//! `h = x + attn(rmsnorm(x))`, then `out = h + mlp(rmsnorm(h))`. Attention uses
//! [`TernaryLinear`] q/k/v/o projections, RoPE on q/k, the KV cache, and
//! [`crate::ops::gqa_attention`]; the MLP is [`Relu2Mlp`]. BitNet also applies a
//! sub-norm inside the BitLinear path — the exact norm placement is pinned in
//! WF-3. Forward lands in WF-3.

use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::layers::{Relu2Mlp, TernaryLinear};

/// A single transformer decoder block (attention + MLP, pre-norm, residual).
#[allow(missing_debug_implementations)]
pub struct TransformerBlock {
    /// RMSNorm weight applied before attention; length `n_embd`.
    pub attn_norm: Vec<f32>,
    /// Query projection `n_embd → n_head · head_dim`.
    pub q_proj: TernaryLinear,
    /// Key projection `n_embd → n_head_kv · head_dim`.
    pub k_proj: TernaryLinear,
    /// Value projection `n_embd → n_head_kv · head_dim`.
    pub v_proj: TernaryLinear,
    /// Output projection `n_head · head_dim → n_embd`.
    pub o_proj: TernaryLinear,
    /// RMSNorm weight applied before the MLP; length `n_embd`.
    pub ffn_norm: Vec<f32>,
    /// The gated ReLU² feed-forward.
    pub mlp: Relu2Mlp,
}

impl TransformerBlock {
    /// Forward over `seq` new tokens at absolute positions `positions`, appending
    /// to and reading from `kv`.
    ///
    /// `x` is `[seq, n_embd]`; `out` is `[seq, n_embd]`, overwritten.
    ///
    /// # Errors
    /// [`NnError::Shape`] on buffer-length mismatch, or [`NnError::Backend`] if a
    /// backend GEMM fails.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        positions: &[usize],
        kv: &mut KvCache,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        let _ = (backend, x, positions, kv, out);
        todo!("WF-3: pre-norm residual attention + ReLU² MLP block forward")
    }
}
