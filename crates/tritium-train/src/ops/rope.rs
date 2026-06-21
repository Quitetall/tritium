//! Rotary position embedding (RoPE) forward + backward for the autograd tape.
//!
//! NeoX half-rotated convention, matching `tritium_nn::ops::rope_apply` (the inference
//! forward): a `[n_token, n_head, head_dim]` buffer where lane `j ∈ [0, half)` pairs with
//! `j + half`; angle `θ_j = pos · theta^(-2j/head_dim)`; for `(a,b) = (x[j], x[j+half])`:
//! ```text
//! out[j]      = a·cos θ_j − b·sin θ_j
//! out[j+half] = b·cos θ_j + a·sin θ_j
//! ```
//! The rotation is orthogonal, so the vjp is the **transpose** = rotation by `−θ_j`
//! (negate `sin`), independent of the input value. `positions`/`theta` are data (no grad).

/// Rotate (`neg_sin=false`) or inverse-rotate (`neg_sin=true`) `buf` in place.
#[allow(clippy::needless_range_loop)]
fn apply(
    buf: &mut [f32],
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    theta: f32,
    neg_sin: bool,
) {
    // Mirror the inference op's contract (tritium_nn::ops::rope_apply): even head_dim +
    // a buffer shaped [n_token, n_head, head_dim]. Odd head_dim would silently leave the
    // last lane unrotated, diverging from inference.
    debug_assert!(head_dim.is_multiple_of(2), "RoPE head_dim must be even");
    debug_assert_eq!(
        buf.len(),
        positions.len() * n_head * head_dim,
        "RoPE buffer must be [n_token, n_head, head_dim]"
    );
    let half = head_dim / 2;
    let theta = f64::from(theta);
    let inv_head_dim = 1.0 / head_dim as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 * inv_head_dim))
        .collect();
    let s = if neg_sin { -1.0f32 } else { 1.0f32 };
    for (token, &pos) in positions.iter().enumerate() {
        let pos = pos as f64;
        let token_base = token * n_head * head_dim;
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for j in 0..half {
                let (sin, cos) = (pos * inv_freq[j]).sin_cos();
                let cos = cos as f32;
                let sin = s * sin as f32;
                let a = buf[head_base + j];
                let b = buf[head_base + j + half];
                buf[head_base + j] = a * cos - b * sin;
                buf[head_base + j + half] = b * cos + a * sin;
            }
        }
    }
}

/// RoPE forward over a `[n_token, n_head, head_dim]` flat buffer.
#[must_use]
pub fn forward(
    x: &[f32],
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    apply(&mut out, positions, n_head, head_dim, theta, false);
    out
}

/// vjp returning `[gx]`: the inverse rotation of the cotangent (rotation is orthogonal,
/// derivative independent of `x`).
#[must_use]
pub fn vjp(
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    theta: f32,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut gx = grad_out.to_vec();
    apply(&mut gx, positions, n_head, head_dim, theta, true);
    vec![gx]
}
