//! One decoder block: pre-norm attention + pre-norm ReLU² MLP, both residual.
//!
//! `h = x + attn(rmsnorm(x))`, then `out = h + mlp(rmsnorm(h))`. Attention uses
//! [`TernaryLinear`] q/k/v/o projections, RoPE on q/k, the KV cache, and
//! [`crate::ops::gqa_attention`]; the MLP is [`Relu2Mlp`].
//!
//! BitNet applies BitLinear sub-norms: `attn_sub_norm` (a `BitNetRMSNorm` over
//! `n_embd`) is applied to the attention output **before** `o_proj`, and
//! `ffn_sub_norm` (over `n_ff`) is applied inside [`Relu2Mlp`] before `down`.
//! Both are wired here against the real `transformers` `modeling_bitnet` layer
//! (`BitNetAttention.forward`: `attn_output = self.attn_sub_norm(attn_output)`
//! then `self.o_proj(...)`); they are load-bearing for layer-0 numeric parity.

use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::layers::{Mlp, Projection};
use crate::ops::{gqa_attention, rmsnorm, rope_apply};

/// Pre-allocated scratch buffers reused across all transformer blocks in a forward
/// pass. Sized once at model init, then passed mutably to each layer — eliminates
/// ~7 heap allocs per layer × 26 layers ≈ 182 allocs per forward.
#[allow(missing_debug_implementations)]
pub struct BlockScratch {
    /// `[seq, n_embd]` — pre-norm output, reused for attn and MLP norms.
    pub normed: Vec<f32>,
    /// `[seq, q_width]` — Q projection output.
    pub q: Vec<f32>,
    /// `[seq, kv_width]` — K projection output.
    pub k: Vec<f32>,
    /// `[seq, kv_width]` — V projection output.
    pub v: Vec<f32>,
    /// `[seq, q_width]` — attention output.
    pub attn: Vec<f32>,
    /// `[seq, q_width]` — sub-norm scratch (only used when attn_sub_norm is present).
    pub sn: Vec<f32>,
    /// `[seq, n_embd]` — MLP output.
    pub mlp_out: Vec<f32>,
}

impl BlockScratch {
    /// Allocate scratch buffers sized for `seq` tokens and the given model geometry.
    pub fn new(seq: usize, n_embd: usize, q_width: usize, kv_width: usize) -> Self {
        Self {
            normed: vec![0.0; seq * n_embd],
            q: vec![0.0; seq * q_width],
            k: vec![0.0; seq * kv_width],
            v: vec![0.0; seq * kv_width],
            attn: vec![0.0; seq * q_width],
            sn: vec![0.0; seq * q_width],
            mlp_out: vec![0.0; seq * n_embd],
        }
    }
}

/// A single transformer decoder block (attention + MLP, pre-norm, residual).
#[allow(missing_debug_implementations)]
pub struct TransformerBlock {
    /// RMSNorm weight applied before attention (`input_layernorm`); length `n_embd`.
    pub attn_norm: Vec<f32>,
    /// Query projection `n_embd → n_head · head_dim`.
    pub q_proj: Projection,
    /// Key projection `n_embd → n_head_kv · head_dim`.
    pub k_proj: Projection,
    /// Value projection `n_embd → n_head_kv · head_dim`.
    pub v_proj: Projection,
    /// Output projection `n_head · head_dim → n_embd`.
    pub o_proj: Projection,
    /// `attn_sub_norm` (`BitNetRMSNorm` over `n_embd`) applied to the attention
    /// output before `o_proj`; length `n_embd` (empty to skip, for the WF-3 tests).
    pub attn_sub_norm: Vec<f32>,
    /// Optional additive bias on the Q/K/V projections (Qwen2/2.5). Each is empty (no bias)
    /// or length = that projection's output width; added right after the projection.
    pub q_bias: Vec<f32>,
    /// K-projection bias (see [`q_bias`](Self::q_bias)).
    pub k_bias: Vec<f32>,
    /// V-projection bias (see [`q_bias`](Self::q_bias)).
    pub v_bias: Vec<f32>,
    /// Optional per-head RMSNorm weight on Q (Qwen3 QK-norm); empty (skip) or length
    /// `head_dim`. Applied per head **before** RoPE.
    pub q_norm: Vec<f32>,
    /// Per-head RMSNorm weight on K (see [`q_norm`](Self::q_norm)).
    pub k_norm: Vec<f32>,
    /// RMSNorm weight applied before the MLP (`post_attention_layernorm`); length
    /// `n_embd`.
    pub ffn_norm: Vec<f32>,
    /// The feed-forward (BitNet ReLU² or Llama/Qwen SwiGLU).
    pub mlp: Mlp,
}

/// Per-stage activations captured during a block forward, for the fidelity
/// ladder. Each is `[seq, n_embd]` (or `[seq, q_width]` for the pre-`o_proj`
/// attention output) and is only populated when [`TransformerBlock::forward_dump`]
/// is used.
#[derive(Debug, Default, Clone)]
pub struct BlockDump {
    /// `input_layernorm(x)` — the pre-attention RMSNorm output.
    pub attn_norm_out: Vec<f32>,
    /// The attention output **after** `attn_sub_norm` and `o_proj`, before the
    /// residual add (i.e. what `transformers` `BitNetAttention.forward` returns).
    pub attn_out: Vec<f32>,
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
        self.forward_inner(backend, x, positions, kv, cfg, out, None, None)
    }

    /// Like [`forward`](Self::forward), but also captures per-stage activations
    /// into `dump` for the fidelity ladder.
    ///
    /// # Errors
    /// Same as [`forward`](Self::forward).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dump(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        positions: &[usize],
        kv: &mut KvCache,
        cfg: &ModelConfig,
        out: &mut [f32],
        dump: &mut BlockDump,
    ) -> Result<(), NnError> {
        self.forward_inner(backend, x, positions, kv, cfg, out, None, Some(dump))
    }

    /// Like [`forward`](Self::forward), but reuses pre-allocated scratch buffers
    /// from a [`BlockScratch`] — eliminates per-layer heap allocs.
    ///
    /// # Errors
    /// Same as [`forward`](Self::forward).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_scratch(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        positions: &[usize],
        kv: &mut KvCache,
        cfg: &ModelConfig,
        out: &mut [f32],
        scratch: &mut BlockScratch,
    ) -> Result<(), NnError> {
        self.forward_inner(backend, x, positions, kv, cfg, out, Some(scratch), None)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        backend: &dyn TernaryBackend,
        x: &[f32],
        positions: &[usize],
        kv: &mut KvCache,
        cfg: &ModelConfig,
        out: &mut [f32],
        scratch: Option<&mut BlockScratch>,
        mut dump: Option<&mut BlockDump>,
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

        // Destructure scratch once to get disjoint mutable borrows, or allocate fresh.
        let (mut normed_a, mut q_a, mut k_a, mut v_a, mut attn_a, mut sn_a, mut mlp_a);
        let (normed, q, k, v, attn, sn_buf, mlp_out) = if let Some(s) = scratch {
            (
                &mut s.normed,
                &mut s.q,
                &mut s.k,
                &mut s.v,
                &mut s.attn,
                &mut s.sn,
                &mut s.mlp_out,
            )
        } else {
            normed_a = vec![0.0f32; seq * n_embd];
            q_a = vec![0.0f32; seq * q_width];
            k_a = vec![0.0f32; seq * kv_width];
            v_a = vec![0.0f32; seq * kv_width];
            attn_a = vec![0.0f32; seq * q_width];
            sn_a = vec![0.0f32; seq * q_width];
            mlp_a = vec![0.0f32; seq * n_embd];
            (
                &mut normed_a,
                &mut q_a,
                &mut k_a,
                &mut v_a,
                &mut attn_a,
                &mut sn_a,
                &mut mlp_a,
            )
        };

        // --- Pre-norm attention -------------------------------------------- //
        // rmsnorm(x) row by row into a scratch buffer.
        for t in 0..seq {
            let src = &x[t * n_embd..t * n_embd + n_embd];
            let dst = &mut normed[t * n_embd..t * n_embd + n_embd];
            rmsnorm(src, &self.attn_norm, cfg.rms_eps, dst)?;
        }
        if let Some(d) = dump.as_mut() {
            d.attn_norm_out = normed.clone();
        }

        // q/k/v projections: q is [seq, q_width]; k/v are [seq, kv_width].
        self.q_proj.forward(backend, normed, seq, q)?;
        self.k_proj.forward(backend, normed, seq, k)?;
        self.v_proj.forward(backend, normed, seq, v)?;

        // Optional additive QKV bias (Qwen2/2.5), per output channel.
        add_bias(q, seq, &self.q_bias);
        add_bias(k, seq, &self.k_bias);
        add_bias(v, seq, &self.v_bias);

        // Optional QK-norm (Qwen3): per-head RMSNorm over head_dim, applied BEFORE RoPE.
        qk_norm(q, seq, n_head, head_dim, &self.q_norm, cfg.rms_eps)?;
        qk_norm(k, seq, n_head_kv, head_dim, &self.k_norm, cfg.rms_eps)?;

        // RoPE on q and k (NeoX half-rotated; values untouched). The projection
        // output is already `[token, head, head_dim]` row-major.
        rope_apply(q, positions, n_head, head_dim, cfg.rope_theta)?;
        rope_apply(k, positions, n_head_kv, head_dim, cfg.rope_theta)?;

        // Append the new keys/values, then attend over the whole cached prefix.
        let causal_offset = kv.len;
        kv.append(k, v, seq)?;
        let (k_all, v_all, ctx) = kv.view();

        // GQA attention: out_attn is `[seq, n_head, head_dim]` == `[seq, q_width]`.
        let scale = 1.0 / (head_dim as f32).sqrt();
        gqa_attention(
            q,
            k_all,
            v_all,
            seq,
            ctx,
            n_head,
            n_head_kv,
            head_dim,
            scale,
            causal_offset,
            attn,
        )?;

        // BitNet `attn_sub_norm` over the attention output, row by row, BEFORE
        // `o_proj` (the `# diff with Llama` line in `BitNetAttention.forward`).
        // `q_width == n_embd` for BitNet, so the sub-norm is over `n_embd`.
        if self.attn_sub_norm.len() == q_width {
            for t in 0..seq {
                let src = &attn[t * q_width..t * q_width + q_width];
                let dst = &mut sn_buf[t * q_width..t * q_width + q_width];
                rmsnorm(src, &self.attn_sub_norm, cfg.rms_eps, dst)?;
            }
            // Copy sn back into attn for the o_proj input.
            attn.copy_from_slice(sn_buf);
        }

        // o_proj: `[seq, q_width] → [seq, n_embd]`, then residual into `out`.
        self.o_proj.forward(backend, attn, seq, out)?;
        if let Some(d) = dump.as_mut() {
            d.attn_out = out.to_vec();
        }
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
        self.mlp.forward(backend, normed, seq, mlp_out)?;
        // Second residual: out = h + mlp(rmsnorm(h)).
        for (o, &m) in out.iter_mut().zip(mlp_out.iter()) {
            *o += m;
        }

        Ok(())
    }
}

/// Add a per-output-channel bias (`bias[c]`) to `buf` (`[seq, bias.len()]`), in place.
/// No-op if `bias` is empty (the standard/BitNet case).
fn add_bias(buf: &mut [f32], seq: usize, bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    let width = bias.len();
    for t in 0..seq {
        for (x, &b) in buf[t * width..t * width + width].iter_mut().zip(bias) {
            *x += b;
        }
    }
}

/// Apply per-head RMSNorm (Qwen3 QK-norm) over `head_dim` to `buf` (`[seq, n_head·head_dim]`),
/// in place. No-op if `weight` is empty.
///
/// # Errors
/// Propagates [`rmsnorm`] shape errors (cannot occur here: `head_dim`-length slices vs a
/// `head_dim` weight).
fn qk_norm(
    buf: &mut [f32],
    seq: usize,
    n_head: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), NnError> {
    if weight.is_empty() {
        return Ok(());
    }
    let width = n_head * head_dim;
    let mut tmp = vec![0.0f32; head_dim];
    for t in 0..seq {
        for h in 0..n_head {
            let off = t * width + h * head_dim;
            rmsnorm(&buf[off..off + head_dim], weight, eps, &mut tmp)?;
            buf[off..off + head_dim].copy_from_slice(&tmp);
        }
    }
    Ok(())
}
