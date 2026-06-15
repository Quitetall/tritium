//! # tritium-cuda
//!
//! CUDA execution backend. Host side via `cudarc`; the addition-only ternary
//! mpGEMM kernel is compiled from `.cu` to PTX by `build.rs` (nvcc) and loaded at
//! runtime. All GPU code is gated behind the `cuda` feature so the default build
//! requires no CUDA toolkit. Implementation in progress (v0.10 Wave C).
