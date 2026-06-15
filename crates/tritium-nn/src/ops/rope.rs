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
    // NeoX-style "half-rotated" RoPE (BitNet / llama family), pinned to the
    // transformers `rotate_half` + `apply_rotary_pos_emb` convention. `head_dim`
    // is split in half: lane `j` in `[0, half)` pairs with lane `j + half`.
    // For pair `(a, b) = (x[j], x[j + half])` at absolute position `pos`:
    //   theta_j     = pos * theta^(-2j/head_dim)
    //   out[j]      = a * cos(theta_j) - b * sin(theta_j)
    //   out[j+half] = b * cos(theta_j) + a * sin(theta_j)
    if !head_dim.is_multiple_of(2) {
        // An odd head_dim cannot be split into rotation pairs; report the even
        // length the layout would have required.
        return Err(NnError::Shape {
            expected: head_dim + 1,
            got: head_dim,
        });
    }

    let expected = positions
        .len()
        .checked_mul(n_head)
        .and_then(|nh| nh.checked_mul(head_dim));
    match expected {
        Some(expected) if expected == x.len() => {}
        Some(expected) => {
            return Err(NnError::Shape {
                expected,
                got: x.len(),
            });
        }
        // Overflow: the requested shape cannot fit in a slice, so it cannot
        // match `x.len()`. Surface a shape error rather than panicking.
        None => {
            return Err(NnError::Shape {
                expected: usize::MAX,
                got: x.len(),
            });
        }
    }

    let half = head_dim / 2;
    let theta = f64::from(theta);
    let inv_head_dim = 1.0 / head_dim as f64;

    // Precompute the per-lane inverse frequencies once; they are independent of
    // token and head. inv_freq[j] = theta^(-2j/head_dim).
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 * inv_head_dim))
        .collect();

    for (token, &pos) in positions.iter().enumerate() {
        let pos = pos as f64;
        // Precompute (cos, sin) per lane for this token; shared across all heads.
        let token_base = token * n_head * head_dim;
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for j in 0..half {
                let angle = pos * inv_freq[j];
                let (sin, cos) = angle.sin_cos();
                let cos = cos as f32;
                let sin = sin as f32;
                let a = x[head_base + j];
                let b = x[head_base + j + half];
                x[head_base + j] = a * cos - b * sin;
                x[head_base + j + half] = b * cos + a * sin;
            }
        }
    }

    Ok(())
}
