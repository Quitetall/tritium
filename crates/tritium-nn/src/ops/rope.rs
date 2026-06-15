//! Rotary position embedding (RoPE).
//!
//! BitNet uses the llama-family RoPE: each head's `head_dim`-vector is split into
//! `head_dim/2` pairs and each pair is rotated by an angle `pos · theta^(-2i/d)`.
//! The exact pairing convention (NeoX half-rotated vs GPT-J interleaved) is
//! pinned against torch goldens in WF-2 — see the RoPE risk in the plan.

use crate::error::NnError;

/// Apply RoPE in place to a packed `[n_token, n_head, head_dim]` activation
/// buffer.
///
/// `x` is the flattened query or key tensor (row-major, token-major). `positions`
/// gives the absolute position of each token (`positions.len()` == number of
/// tokens); `n_head` heads of width `head_dim` are rotated per token using base
/// frequency `theta`.
///
/// # Errors
/// [`NnError::Shape`] if `x.len()` disagrees with
/// `positions.len() * n_head * head_dim`, or if `head_dim` is odd.
pub fn rope_apply(
    x: &mut [f32],
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    theta: f32,
) -> Result<(), NnError> {
    let _ = (x, positions, n_head, head_dim, theta);
    todo!("WF-2: rotary position embedding, pairing convention pinned to torch goldens")
}
