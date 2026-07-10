//! Higher-level differentiable building blocks assembled on the [`Tape`](crate::Tape) — the pieces
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
    assert!(
        n_head.is_multiple_of(n_kv_head),
        "n_head {n_head} must be a multiple of n_kv_head {n_kv_head}"
    );
    let qd = n_head * head_dim;
    let kvd = n_kv_head * head_dim;
    let group = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let pos: Vec<usize> = (0..seq).collect();

    let q = t.dense_matmul(x, wq, seq, qd, n_embd);
    let k = t.dense_matmul(x, wk, seq, kvd, n_embd);
    let v = t.dense_matmul(x, wv, seq, kvd, n_embd);
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
    let cat = t.concat_cols(&head_outs, seq, &vec![head_dim; n_head]);
    t.dense_matmul(cat, wo, seq, n_embd, qd)
}
