//! One decoder block: pre-norm attention + pre-norm ReLU² MLP, both residual.
//!
//! `h = x + attn(rmsnorm(x))`, then `out = h + mlp(rmsnorm(h))`. Attention uses
//! [`TernaryLinear`] q/k/v/o projections, RoPE on q/k, the KV cache, and
//! [`crate::ops::gqa_attention`]; the MLP is [`Relu2Mlp`].
//!
//! BitNet also applies BitLinear sub-norms (`attn_sub_norm` before `o_proj`,
//! `ffn_sub_norm` before `down_proj`); their exact placement is wired in WF-4
//! against the real `transformers` layer. This block validates assembly + shape
//! + residual wiring on a tiny random config; deep numeric fidelity is WF-4.

use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::layers::{Relu2Mlp, TernaryLinear};
use crate::ops::{gqa_attention, rmsnorm, rope_apply};

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
    /// `x` is `[seq, n_embd]`; `out` is `[seq, n_embd]`, overwritten with
    /// `x + attn(rmsnorm(x))` then `+ mlp(rmsnorm(·))`. `cfg` supplies the head
    /// geometry (`n_head`, `n_head_kv`, `head_dim`), `rope_theta`, and `rms_eps`.
    /// `positions.len()` must equal `seq`; the new keys/values are appended to
    /// `kv`, and attention reads the full cached prefix with a causal offset of
    /// the pre-append cache length.
    ///
    /// # Errors
    /// [`NnError::Shape`] on buffer-length mismatch (including `positions.len() !=
    /// seq`), or [`NnError::Backend`] if a backend GEMM fails.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        positions: &[usize],
        kv: &mut KvCache,
        cfg: &ModelConfig,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        let n_embd = self.attn_norm.len();
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim() as usize;
        let q_width = n_head * head_dim;
        let kv_width = n_head_kv * head_dim;

        if n_embd == 0 || !x.len().is_multiple_of(n_embd) {
            return Err(NnError::Shape {
                expected: n_embd,
                got: x.len(),
            });
        }
        let seq = x.len() / n_embd;
        if positions.len() != seq {
            return Err(NnError::Shape {
                expected: seq,
                got: positions.len(),
            });
        }
        if out.len() != x.len() {
            return Err(NnError::Shape {
                expected: x.len(),
                got: out.len(),
            });
        }

        // --- Pre-norm attention -------------------------------------------- //
        // rmsnorm(x) row by row into a scratch.
        let mut normed = vec![0.0f32; seq * n_embd];
        for t in 0..seq {
            let src = &x[t * n_embd..t * n_embd + n_embd];
            let dst = &mut normed[t * n_embd..t * n_embd + n_embd];
            rmsnorm(src, &self.attn_norm, cfg.rms_eps, dst)?;
        }

        // q/k/v projections: q is [seq, q_width]; k/v are [seq, kv_width].
        let mut q = vec![0.0f32; seq * q_width];
        let mut k = vec![0.0f32; seq * kv_width];
        let mut v = vec![0.0f32; seq * kv_width];
        self.q_proj.forward(backend, &normed, seq, &mut q)?;
        self.k_proj.forward(backend, &normed, seq, &mut k)?;
        self.v_proj.forward(backend, &normed, seq, &mut v)?;

        // RoPE on q and k (NeoX half-rotated; values untouched). The projection
        // output is already `[token, head, head_dim]` row-major.
        rope_apply(&mut q, positions, n_head, head_dim, cfg.rope_theta)?;
        rope_apply(&mut k, positions, n_head_kv, head_dim, cfg.rope_theta)?;

        // Append the new keys/values, then attend over the whole cached prefix.
        let causal_offset = kv.len;
        kv.append(&k, &v, seq)?;
        let (k_all, v_all, ctx) = kv.view();

        // GQA attention: out_attn is `[seq, n_head, head_dim]` == `[seq, q_width]`.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn = vec![0.0f32; seq * q_width];
        gqa_attention(
            &q,
            k_all,
            v_all,
            seq,
            ctx,
            n_head,
            n_head_kv,
            head_dim,
            scale,
            causal_offset,
            &mut attn,
        )?;

        // o_proj: `[seq, q_width] → [seq, n_embd]`, then residual into `out`.
        self.o_proj.forward(backend, &attn, seq, out)?;
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            *o += xi;
        }

        // --- Pre-norm ReLU² MLP -------------------------------------------- //
        // h is now in `out`; rmsnorm(h) row by row into `normed` (reused).
        for t in 0..seq {
            let src = &out[t * n_embd..t * n_embd + n_embd];
            let dst = &mut normed[t * n_embd..t * n_embd + n_embd];
            rmsnorm(src, &self.ffn_norm, cfg.rms_eps, dst)?;
        }
        let mut mlp_out = vec![0.0f32; seq * n_embd];
        self.mlp.forward(backend, &normed, seq, &mut mlp_out)?;
        // Second residual: out = h + mlp(rmsnorm(h)).
        for (o, &m) in out.iter_mut().zip(mlp_out.iter()) {
            *o += m;
        }

        Ok(())
    }
}
