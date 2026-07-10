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

/// Logistic sigmoid `σ(x) = 1/(1+e^{-x})` (saturates cleanly: `e^{-x}` overflows to `+∞` for
/// very negative `x`, giving `0`, no NaN).
#[inline]
#[must_use]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / swish activation: `Y = x·σ(x)` — the Llama/Qwen SwiGLU gate. Smooth (C^∞), so
/// finite-difference-checkable everywhere.
#[must_use]
pub fn silu_forward(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * sigmoid(v)).collect()
}

/// vjp returning `[gX]`: `gX = gY · (σ(x) + x·σ(x)·(1−σ(x)))` — the exact derivative of `x·σ(x)`.
#[must_use]
pub fn silu_vjp(x: &[f32], grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_x = x
        .iter()
        .zip(grad_out)
        .map(|(&v, &g)| {
            let s = sigmoid(v);
            g * (s + v * s * (1.0 - s))
        })
        .collect();
    vec![g_x]
}
