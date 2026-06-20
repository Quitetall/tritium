//! Per-op forward + `vjp` (vector-Jacobian product) functions.

pub mod act;
pub mod bias;
pub mod elementwise;
pub mod loss;
pub mod matmul;
pub mod ste;
