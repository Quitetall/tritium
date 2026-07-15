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
/// frequency `theta`. The model configuration supplies `theta` and validates
/// that it is finite and positive before this hot-path op is called.
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
    // Preserve the historical empty-width behavior while the partial API
    // rejects a zero rotary width as nonsensical.
    if head_dim == 0 {
        return validate_layout_len(x.len(), positions.len(), n_head, head_dim);
    }
    rope_apply_partial_neox(x, positions, n_head, head_dim, head_dim, theta)
}

/// Apply NeoX-style RoPE to the first `rotary_dim` lanes of each head.
///
/// The flattened buffer layout is `[n_token, n_head, head_dim]`. Within each
/// head, only `[..rotary_dim]` is split in half and rotated; the suffix
/// `[rotary_dim..]` is never written. Inverse frequencies use `rotary_dim` in
/// the exponent: `theta^(-2j/rotary_dim)`, matching Qwen3.5 partial RoPE.
/// As with [`rope_apply`], `theta` comes from the already validated model
/// configuration and is expected to be finite and positive.
///
/// # Errors
/// [`NnError::Shape`] if the packed layout overflows or disagrees with `x`, if
/// `head_dim` is odd, or if `rotary_dim` is zero, odd, or exceeds `head_dim`.
pub fn rope_apply_partial_neox(
    x: &mut [f32],
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<(), NnError> {
    // NeoX-style "half-rotated" RoPE (BitNet / llama family), pinned to the
    // transformers `rotate_half` + `apply_rotary_pos_emb` convention. The rotary
    // prefix is split in half: lane `j` in `[0, half)` pairs with lane `j + half`.
    // For pair `(a, b) = (x[j], x[j + half])` at absolute position `pos`:
    //   theta_j     = pos * theta^(-2j/rotary_dim)
    //   out[j]      = a * cos(theta_j) - b * sin(theta_j)
    //   out[j+half] = b * cos(theta_j) + a * sin(theta_j)
    if !head_dim.is_multiple_of(2) {
        // An odd head_dim cannot be split into rotation pairs; report the even
        // length the layout would have required.
        return Err(NnError::Shape {
            expected: head_dim.saturating_add(1),
            got: head_dim,
        });
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(NnError::Shape {
            expected: rotary_dim.saturating_add(1),
            got: rotary_dim,
        });
    }
    if rotary_dim > head_dim {
        return Err(NnError::Shape {
            expected: head_dim,
            got: rotary_dim,
        });
    }

    validate_layout_len(x.len(), positions.len(), n_head, head_dim)?;
    if x.is_empty() {
        return Ok(());
    }

    let half = rotary_dim / 2;
    let n_pos = positions.len();
    let table_len = n_pos.checked_mul(half).ok_or(NnError::Shape {
        expected: usize::MAX,
        got: x.len(),
    })?;
    let theta = f64::from(theta);
    let inv_rotary_dim = 1.0 / rotary_dim as f64;

    // Precompute the per-lane inverse frequencies once; they are independent of
    // token and head. inv_freq[j] = theta^(-2j/rotary_dim).
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 * inv_rotary_dim))
        .collect();

    // Precompute cos/sin table: [positions.len() × half]. For a given position
    // and lane j, the (cos, sin) pair is identical across all heads — only the
    // data being rotated differs. Precomputing eliminates (n_head-1) × half
    // sin_cos calls per position.
    let mut cos_table = vec![0.0f32; table_len];
    let mut sin_table = vec![0.0f32; table_len];
    for (token, &pos) in positions.iter().enumerate() {
        let pos = pos as f64;
        let ct = &mut cos_table[token * half..token * half + half];
        let st = &mut sin_table[token * half..token * half + half];
        for j in 0..half {
            let angle = pos * inv_freq[j];
            let (s, c) = angle.sin_cos();
            ct[j] = c as f32;
            st[j] = s as f32;
        }
    }

    for (token, _) in positions.iter().enumerate() {
        let token_base = token * n_head * head_dim;
        let ct = &cos_table[token * half..token * half + half];
        let st = &sin_table[token * half..token * half + half];
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for j in 0..half {
                let a = x[head_base + j];
                let b = x[head_base + j + half];
                x[head_base + j] = a * ct[j] - b * st[j];
                x[head_base + j + half] = b * ct[j] + a * st[j];
            }
        }
    }

    Ok(())
}

#[inline]
fn validate_layout_len(
    got: usize,
    n_token: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<(), NnError> {
    let expected = n_token
        .checked_mul(n_head)
        .and_then(|heads| heads.checked_mul(head_dim));
    match expected {
        Some(expected) if expected == got => Ok(()),
        Some(expected) => Err(NnError::Shape { expected, got }),
        None => Err(NnError::Shape {
            expected: usize::MAX,
            got,
        }),
    }
}
