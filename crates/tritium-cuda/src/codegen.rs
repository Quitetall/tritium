//! nvrtc JIT codegen for the IMMA kernel (v0.30, ADR 0005 / WF-B — skeleton).
//!
//! The IMMA (`mma.m16n8k32`) kernel is templated over the [`TileConfig`]
//! parameters from [`super::autotune`]; this module will render the CUDA source
//! for a chosen tile and compile it to a cubin via cudarc's nvrtc binding at
//! runtime (the `nvrtc` cargo feature is already enabled on `cudarc`). AOT
//! default cubins cover the common BitNet shapes; JIT covers the long tail and
//! the autotune search.
//!
//! Determinism is load-bearing: a JIT-compiled tile must produce **bit-identical**
//! output to the AOT cubin for the same tile (the cold-cache == warm-cache gate),
//! so codegen only varies the tile/launch parameters, never the arithmetic.
//!
//! Gated behind the `cuda` feature (it links cudarc's nvrtc path). No code yet —
//! WF-B fills this in.
//!
//! [`TileConfig`]: super::autotune::TileConfig
