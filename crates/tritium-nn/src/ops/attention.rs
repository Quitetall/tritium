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
    let _ = (
        q,
        k,
        v,
        seq,
        ctx,
        n_head,
        n_head_kv,
        head_dim,
        scale,
        causal_offset,
        out,
    );
    todo!("WF-2: naive causal grouped-query attention in fp32")
}
