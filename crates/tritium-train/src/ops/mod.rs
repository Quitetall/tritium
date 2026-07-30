//! Per-op forward + `vjp` (vector-Jacobian product) functions.

pub mod act;
pub mod attention;
pub mod bias;
pub mod conv1d;
pub mod conv2d;
pub mod dense;
pub mod elementwise;
pub mod embed;
pub mod fsq;
pub mod hestia;
pub mod loss;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod shape;
pub mod softmax;
pub mod ste;
