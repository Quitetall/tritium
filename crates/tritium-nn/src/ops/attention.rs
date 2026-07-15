//! Grouped-query attention (GQA), computed naively in fp32.
//!
//! BitNet 2B4T is GQA 20/5 (`n_head=20`, `n_head_kv=5`, `head_dim=128`): each KV
//! head is shared by `n_head / n_head_kv` query heads. Attention is causal; in
//! incremental decode the `seq` new query rows attend over `ctx` cached keys,
//! offset by `causal_offset` (the number of already-cached tokens). The masking
//! and softmax are validated against torch goldens in WF-2.

use crate::error::NnError;

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
/// softmax (see [`crate::softmax_rows`]) normalizes them, and the output is the
/// softmax-weighted sum of the visible `v` rows. A row with no visible keys
/// follows the torch all-masked convention (NaN), inherited from
/// [`crate::softmax_rows`].
///
/// # Errors
/// [`NnError::Shape`] if any buffer length disagrees with the supplied dims,
/// any attention dimension is zero, `n_head` is not a multiple of
/// `n_head_kv`, or the causal extent exceeds `ctx`. [`NnError::Backend`] if a
/// dimension/offset calculation overflows, `scale` is nonfinite, or scratch
/// allocation fails.
///
/// Every fallible validation and allocation completes before `out` is touched,
/// so any error leaves it unchanged.
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
    if n_head == 0 {
        return Err(NnError::Shape {
            expected: 1,
            got: n_head,
        });
    }
    if n_head_kv == 0 {
        return Err(NnError::Shape {
            expected: 1,
            got: n_head_kv,
        });
    }
    if head_dim == 0 {
        return Err(NnError::Shape {
            expected: 1,
            got: head_dim,
        });
    }
    if !scale.is_finite() {
        return Err(NnError::Backend(
            "attention score scale must be finite".to_owned(),
        ));
    }
    if seq != 0 {
        let causal_end = causal_offset.checked_add(seq).ok_or_else(|| {
            NnError::Backend("attention causal extent addition overflow".to_owned())
        })?;
        if causal_end > ctx {
            return Err(NnError::Shape {
                expected: ctx,
                got: causal_end,
            });
        }
    }

    // Dimension / buffer-length contract.
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(NnError::Shape {
            expected: n_head_kv,
            got: n_head,
        });
    }
    let q_width = n_head.checked_mul(head_dim).ok_or_else(|| {
        NnError::Backend("attention query row-width multiplication overflow".to_owned())
    })?;
    let kv_width = n_head_kv.checked_mul(head_dim).ok_or_else(|| {
        NnError::Backend("attention KV row-width multiplication overflow".to_owned())
    })?;
    let q_len = seq.checked_mul(q_width).ok_or_else(|| {
        NnError::Backend("attention query buffer-length multiplication overflow".to_owned())
    })?;
    let kv_len = ctx.checked_mul(kv_width).ok_or_else(|| {
        NnError::Backend("attention KV buffer-length multiplication overflow".to_owned())
    })?;
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
    if seq == 0 {
        return Ok(());
    }

    preflight_last_head_end(seq, q_width, n_head, head_dim, q_len, "query/output")?;
    preflight_last_head_end(ctx, kv_width, n_head_kv, head_dim, kv_len, "key/value")?;

    let n_rep = n_head / n_head_kv; // group size: Q heads per KV head
    // Scratch for one query head's score row over all `ctx` keys.
    let mut scores = try_zeroed(ctx, "attention scores")?;

    for i in 0..seq {
        // Absolute position of this query row; keys `0..=limit` are visible.
        // `causal_offset + seq` was checked above, so `i < seq` proves this add.
        let limit = causal_offset + i;
        for h in 0..n_head {
            let kv = h / n_rep; // KV head feeding query head `h`
            // Checked row widths, total lengths, and last-head extents prove all
            // direct ranges in the loop. Keep the hot reference path branch-free.
            let q_off = i * q_width + h * head_dim;
            let q_row = &q[q_off..q_off + head_dim];

            // Raw scaled dot-product scores; masked keys -> -inf.
            for (j, score) in scores.iter_mut().enumerate() {
                if j > limit {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = j * kv_width + kv * head_dim;
                let k_row = &k[k_off..k_off + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                *score = dot * scale;
            }

            // Row-wise softmax over the `ctx` keys (NaN for a fully-masked row,
            // matching torch).
            softmax_valid_row(&mut scores);

            // Weighted sum of the value rows.
            let o_off = i * q_width + h * head_dim;
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
                let v_off = j * kv_width + kv * head_dim;
                let v_row = &v[v_off..v_off + head_dim];
                for d in 0..head_dim {
                    o_row[d] += w * v_row[d];
                }
            }
        }
    }

    Ok(())
}

fn try_zeroed(len: usize, what: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!("allocate {what} for {len} f32 values: {error}"))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn preflight_last_head_end(
    tokens: usize,
    row_width: usize,
    heads: usize,
    head_dim: usize,
    total_len: usize,
    what: &str,
) -> Result<(), NnError> {
    let last_token = tokens
        .checked_sub(1)
        .ok_or_else(|| NnError::Backend(format!("attention {what} extent has no token")))?;
    let last_head = heads
        .checked_sub(1)
        .ok_or_else(|| NnError::Backend(format!("attention {what} extent has no head")))?;
    let token_start = last_token.checked_mul(row_width).ok_or_else(|| {
        NnError::Backend(format!(
            "attention {what} token-offset multiplication overflow"
        ))
    })?;
    let head_start = last_head.checked_mul(head_dim).ok_or_else(|| {
        NnError::Backend(format!(
            "attention {what} head-offset multiplication overflow"
        ))
    })?;
    let start = token_start.checked_add(head_start).ok_or_else(|| {
        NnError::Backend(format!("attention {what} start-offset addition overflow"))
    })?;
    let end = start.checked_add(head_dim).ok_or_else(|| {
        NnError::Backend(format!("attention {what} end-offset addition overflow"))
    })?;
    if end != total_len {
        return Err(NnError::Backend(format!(
            "attention {what} extent {end} disagrees with validated length {total_len}"
        )));
    }
    Ok(())
}

/// Infallible one-row form of [`crate::softmax_rows`].
///
/// `gqa_attention` proves `ctx > 0` from its causal extent and allocates exactly
/// `ctx` scores before it writes output. Keeping this helper infallible makes
/// every real error exit precede output publication without a second output
/// buffer. The operation order intentionally matches `softmax_rows` exactly.
fn softmax_valid_row(row: &mut [f32]) {
    debug_assert!(!row.is_empty());

    let mut max = f32::NEG_INFINITY;
    for &value in row.iter() {
        if value > max {
            max = value;
        }
    }

    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        let exp = (*value - max).exp();
        *value = exp;
        sum += exp;
    }

    let inv = 1.0f32 / sum;
    for value in row.iter_mut() {
        *value *= inv;
    }
}
