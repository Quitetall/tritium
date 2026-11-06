//! Higher-level differentiable building blocks assembled on the [`Tape`] — the pieces
//! the real-model SALT-distillation forward needs beyond the flat op set (plan 0040).

use crate::Tape;
use crate::tape::ValueId;

/// Multi-head causal self-attention with grouped-query attention (GQA), on the tape.
///
/// `x` is `[seq, n_embd]` (already normed). Projection weights are row-major `[out, in]` as
/// [`Tape::dense_matmul`] expects, and are the caller's responsibility to SALT-STE if quantizing:
/// `wq` `[n_head·head_dim, n_embd]`, `wk`/`wv` `[n_kv_head·head_dim, n_embd]`,
/// `wo` `[n_embd, n_head·head_dim]`. GQA shares each KV head across `n_head / n_kv_head` query
/// heads. Returns the attention output `[seq, n_embd]`.
///
/// RoPE (θ = `theta`) is applied to the full Q/K in one call each (the rope op rotates every head
/// block of a `[seq, n_head, head_dim]` buffer); heads are then sliced out for per-head SDPA and the
/// outputs concatenated. `n_head` must be a multiple of `n_kv_head`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    t: &mut Tape,
    x: ValueId,
    wq: ValueId,
    wk: ValueId,
    wv: ValueId,
    wo: ValueId,
    seq: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    theta: f32,
) -> ValueId {
    let cat = attention_heads(
        t, x, wq, wk, wv, seq, n_embd, n_head, n_kv_head, head_dim, theta,
    );
    t.dense_matmul(cat, wo, seq, n_embd, n_head * head_dim)
}

/// [`attention`] **without** the output projection: returns the concatenated head outputs,
/// `[seq, n_head * head_dim]`, which is `o_proj`'s input.
///
/// Split out so a caller can reach that activation. It is the one projection input in a block that
/// [`attention`] used to hide, which made it the one projection an activation-precision sweep could
/// not quantize — every such number was an upper bound with 1 of 7 projections unpenalised, and the
/// salience fold has no calibration tap for it either.
///
/// The caller supplies the missing multiply: `t.dense_matmul(cat, wo, seq, n_embd, n_head *
/// head_dim)`. [`attention`] is exactly that and nothing else, so the two cannot drift.
#[allow(clippy::too_many_arguments)]
pub fn attention_heads(
    t: &mut Tape,
    x: ValueId,
    wq: ValueId,
    wk: ValueId,
    wv: ValueId,
    seq: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    theta: f32,
) -> ValueId {
    attention_heads_traced(
        t, x, wq, wk, wv, seq, n_embd, n_head, n_kv_head, head_dim, theta,
    )
    .heads
}

/// Actual tape values at the three projection outputs and concatenated heads.
/// Q/K values are before RoPE, so their post-backward gradients belong to the
/// original linear projections, not to a rotated surrogate observation.
#[derive(Clone, Copy, Debug)]
pub struct AttentionTrace {
    /// Original query projection output.
    pub query: ValueId,
    /// Original key projection output.
    pub key: ValueId,
    /// Original value projection output.
    pub value: ValueId,
    /// Concatenated attention outputs before the output projection.
    pub heads: ValueId,
}

/// Learned per-head Q/K RMSNorm values on the caller's tape.
/// Each weight has `head_dim` finite entries; epsilon is finite and positive.
#[derive(Clone, Copy, Debug)]
pub struct AttentionQkNorm {
    pub query: ValueId,
    pub key: ValueId,
    pub eps: f32,
}

/// The same actual attention implementation, exposing original projection IDs.
/// The caller owns the tape and may inspect gradients after its own objective.
/// Shape/ownership contracts and arithmetic are identical to [`attention_heads`].
#[allow(clippy::too_many_arguments)]
pub fn attention_heads_traced(
    t: &mut Tape,
    x: ValueId,
    wq: ValueId,
    wk: ValueId,
    wv: ValueId,
    seq: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    theta: f32,
) -> AttentionTrace {
    attention_heads_traced_with_qk_norm(
        t, x, wq, wk, wv, seq, n_embd, n_head, n_kv_head, head_dim, theta, None,
    )
}

/// Shared traced attention with optional learned Q/K RMSNorm before RoPE.
/// Trace outputs remain the original linear projections **before** normalization.
/// Normalization and its backward pass use [`Tape::rmsnorm`] per head; absent
/// normalization executes the same arithmetic as [`attention_heads_traced`].
///
/// # Panics
/// Panics on invalid head geometry, norm lengths, nonfinite norm weights, or
/// nonpositive/nonfinite norm epsilon, before recording new attention operations.
#[allow(clippy::too_many_arguments)]
pub fn attention_heads_traced_with_qk_norm(
    t: &mut Tape,
    x: ValueId,
    wq: ValueId,
    wk: ValueId,
    wv: ValueId,
    seq: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    theta: f32,
    norm: Option<AttentionQkNorm>,
) -> AttentionTrace {
    assert!(n_head > 0 && n_kv_head > 0 && head_dim > 0);
    assert!(
        n_head.is_multiple_of(n_kv_head),
        "n_head {n_head} must be a multiple of n_kv_head {n_kv_head}"
    );
    if let Some(norm) = norm {
        assert!(norm.eps.is_finite() && norm.eps > 0.0);
        for id in [norm.query, norm.key] {
            assert_eq!(t.value(id).len(), head_dim);
            assert!(t.value(id).iter().all(|v| v.is_finite()));
        }
    }
    let qd = n_head * head_dim;
    let kvd = n_kv_head * head_dim;
    let group = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let pos: Vec<usize> = (0..seq).collect();

    let q = t.dense_matmul(x, wq, seq, qd, n_embd);
    let k = t.dense_matmul(x, wk, seq, kvd, n_embd);
    let v = t.dense_matmul(x, wv, seq, kvd, n_embd);
    let query = q;
    let key = k;
    let value = v;
    let (q, k) = if let Some(norm) = norm {
        (
            t.rmsnorm(q, norm.query, seq * n_head, head_dim, norm.eps),
            t.rmsnorm(k, norm.key, seq * n_kv_head, head_dim, norm.eps),
        )
    } else {
        (q, k)
    };
    let q = t.rope(q, pos.clone(), n_head, head_dim, theta);
    let k = t.rope(k, pos.clone(), n_kv_head, head_dim, theta);

    let mut head_outs = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let kv = h / group;
        let qh = t.slice_cols(q, seq, qd, h * head_dim, head_dim);
        let kh = t.slice_cols(k, seq, kvd, kv * head_dim, head_dim);
        let vh = t.slice_cols(v, seq, kvd, kv * head_dim, head_dim);
        let scores = t.dense_matmul(qh, kh, seq, seq, head_dim); // qh · khᵀ
        let scores = t.scale_const(scores, scale);
        let scores = t.causal_mask(scores, seq, seq);
        let p = t.softmax(scores, seq, seq);
        let vt = t.transpose(vh, seq, head_dim);
        head_outs.push(t.dense_matmul(p, vt, seq, head_dim, seq)); // p · vh
    }
    let heads = t.concat_cols(&head_outs, seq, &vec![head_dim; n_head]);
    AttentionTrace {
        query,
        key,
        value,
        heads,
    }
}
