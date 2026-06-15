//! Grouped-query attention (GQA), computed naively in fp32.
//!
//! BitNet 2B4T is GQA 20/5 (`n_head=20`, `n_head_kv=5`, `head_dim=128`): each KV
//! head is shared by `n_head / n_head_kv` query heads. Attention is causal; in
//! incremental decode the `seq` new query rows attend over `ctx` cached keys,
//! offset by `causal_offset` (the number of already-cached tokens). The masking
//! and softmax are validated against torch goldens in WF-2.

use crate::error::NnError;
use crate::ops::softmax::softmax_rows;

/// Naive causal GQA attention.
///
/// - `q`: `[seq, n_head, head_dim]` row-major query rows for the new tokens.
/// - `k`, `v`: `[ctx, n_head_kv, head_dim]` keys/values (the full visible
///   context, including cached tokens).
/// - `scale`: the score scale (typically `1/sqrt(head_dim)`), applied to
///   `qᵀk` before softmax.
/// - `causal_offset`: absolute position of the first query row, so query `i`
///   attends to keys `0..=causal_offset + i`.
/// - `out`: `[seq, n_head, head_dim]` row-major, overwritten with the attention
///   output.
///
/// Query head `h` reads KV head `h / (n_head / n_head_kv)`.
///
/// Scores for query row `i` are `scale * <q[i,h], k[j,kv(h)]>` for every visible
/// key `j` (`j <= causal_offset + i`) and `-inf` for masked keys; a row-wise
/// softmax (see [`softmax_rows`]) normalizes them, and the output is the
/// softmax-weighted sum of the visible `v` rows. A row with no visible keys
/// follows the torch all-masked convention (NaN), inherited from
/// [`softmax_rows`].
///
/// # Errors
/// [`NnError::Shape`] if any buffer length disagrees with the supplied dims, or
/// if `n_head` is not a multiple of `n_head_kv`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    ctx: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    scale: f32,
    causal_offset: usize,
    out: &mut [f32],
) -> Result<(), NnError> {
    // Dimension / buffer-length contract.
    if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(NnError::Shape {
            expected: n_head_kv,
            got: n_head,
        });
    }
    let q_len = seq * n_head * head_dim;
    let kv_len = ctx * n_head_kv * head_dim;
    if q.len() != q_len {
        return Err(NnError::Shape {
            expected: q_len,
            got: q.len(),
        });
    }
    if k.len() != kv_len {
        return Err(NnError::Shape {
            expected: kv_len,
            got: k.len(),
        });
    }
    if v.len() != kv_len {
        return Err(NnError::Shape {
            expected: kv_len,
            got: v.len(),
        });
    }
    if out.len() != q_len {
        return Err(NnError::Shape {
            expected: q_len,
            got: out.len(),
        });
    }

    let n_rep = n_head / n_head_kv; // group size: Q heads per KV head
    // Scratch for one query head's score row over all `ctx` keys.
    let mut scores = vec![0.0f32; ctx];

    for i in 0..seq {
        // Absolute position of this query row; keys `0..=limit` are visible.
        let limit = causal_offset + i;
        for h in 0..n_head {
            let kv = h / n_rep; // KV head feeding query head `h`
            let q_off = (i * n_head + h) * head_dim;
            let q_row = &q[q_off..q_off + head_dim];

            // Raw scaled dot-product scores; masked keys -> -inf.
            for (j, score) in scores.iter_mut().enumerate() {
                if j > limit {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (j * n_head_kv + kv) * head_dim;
                let k_row = &k[k_off..k_off + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                *score = dot * scale;
            }

            // Row-wise softmax over the `ctx` keys (NaN for a fully-masked row,
            // matching torch).
            softmax_rows(&mut scores, ctx)?;

            // Weighted sum of the value rows.
            let o_off = (i * n_head + h) * head_dim;
            let o_row = &mut out[o_off..o_off + head_dim];
            for slot in o_row.iter_mut() {
                *slot = 0.0;
            }
            for (j, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    // Masked / vanishing weight contributes nothing; skipping it
                    // also keeps a `-inf`-derived 0 from polluting the sum.
                    continue;
                }
                let v_off = (j * n_head_kv + kv) * head_dim;
                let v_row = &v[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    o_row[d] += w * v_row[d];
                }
            }
        }
    }

    Ok(())
}
