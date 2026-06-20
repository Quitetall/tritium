//! Element-wise binary ops over same-shape buffers.
//!
//! `add`: `Y = A + B`  ⇒  `gA = gY`, `gB = gY`.
//! `mul`: `Y = A ⊙ B`  ⇒  `gA = gY ⊙ B`, `gB = gY ⊙ A` (Hadamard product).

/// Forward: `Y = A + B`.
#[must_use]
pub fn add_forward(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// vjp returning `[gA, gB] = [gY, gY]`.
#[must_use]
pub fn add_vjp(_a: &[f32], _b: &[f32], grad_out: &[f32]) -> Vec<Vec<f32>> {
    vec![grad_out.to_vec(), grad_out.to_vec()]
}

/// Forward: `Y = A ⊙ B` (element-wise product).
#[must_use]
pub fn mul_forward(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

/// vjp returning `[gA, gB] = [gY ⊙ B, gY ⊙ A]`.
#[must_use]
pub fn mul_vjp(a: &[f32], b: &[f32], grad_out: &[f32]) -> Vec<Vec<f32>> {
    let g_a = grad_out.iter().zip(b).map(|(&g, &y)| g * y).collect();
    let g_b = grad_out.iter().zip(a).map(|(&g, &x)| g * x).collect();
    vec![g_a, g_b]
}
