//! Per-op forward + `vjp` (vector-Jacobian product) functions.

pub mod act;
pub mod bias;
pub mod dense;
pub mod elementwise;
pub mod loss;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod softmax;
pub mod ste;
