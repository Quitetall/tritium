//! Squared-ReLU activation: `Y = max(x, 0)^2` (the BitNet MLP gate).
//!
//! Backward: `gX = gY · 2·max(x, 0)`. The derivative `2·relu(x)` is continuous at
//! `x = 0` (the function is C¹), so it is finite-difference-checkable everywhere.

/// Forward: element-wise `max(x, 0)^2`.
#[must_use]
pub fn relu2_forward(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let r = v.max(0.0);
            r * r
        })
        .collect()
}

/// vjp returning `[gX]`: `gX = gY · 2·max(x, 0)`.
#[must_use]
pub fn relu2_vjp(x: &[f32], grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_x = x
        .iter()
        .zip(grad_out)
        .map(|(&v, &g)| g * 2.0 * v.max(0.0))
        .collect();
    vec![g_x]
}
