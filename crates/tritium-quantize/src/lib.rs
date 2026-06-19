//! # tritium-quantize
//!
//! **SALT** — Sensitivity-Allocated Layered Ternary quantization (ADR 0001).
//! Turns fp weights into a stack of ternary planes `W ≈ Σ_p s_p · t_p` (each
//! `t_p ∈ {-1, 0, +1}`), spending extra planes only on the weight groups the
//! model is sensitive to, under a bits-per-weight budget. Inference stays
//! multiply-free: a `T`-plane weight is `T` add/sub/skip passes, scaled and
//! summed — the existing ternary mpGEMM kernel, looped.
//!
//! ## Pipeline (ADR 0001) and where each stage lives
//!
//! 1. **Residual ternary expansion** — [`residual_expand`], [`Plane`],
//!    [`PlaneStack`]. Greedy AbsMean per plane; `T = 1` is exactly flat BitNet
//!    b1.58. Reconstruction + error: [`PlaneStack::reconstruct`], [`recon_error`].
//! 2. Mode codebook — *(later v0.40 step)*.
//! 3. Sensitivity rank + **4. rate-distortion plane allocation** —
//!    [`allocate`], [`GroupInput`], [`AllocConfig`], [`Allocation`]. Greedy
//!    water-filling over per-group error curves under a bits-per-weight budget.
//! 5. Sparse residual planes / 6. heal — *(later; GPU + train).*
//!
//! The format sidecar (`tritium-format`) and the multi-plane accumulate kernel
//! (CUDA/CPU backends) consume what this crate produces; they land in their own
//! v0.40 steps (ADR 0006).
//!
//! ## CPU-only exit gates (ADR 0006), all enforced here
//!
//! - `T = 1` reduces **exactly** to flat AbsMean (BitNet regression golden).
//! - Reconstruction error is **monotonic** non-increasing in plane count `T`.
//! - Same input ⇒ **byte-identical** output (determinism).
//!
//! GPU gates (multi-plane accumulate matches dequant; sparse == dense) and the
//! model-level accuracy-vs-bpw curve gate are validated in their own lanes.

mod allocate;
mod plane;
mod quantize;

pub use allocate::{AllocConfig, AllocError, Allocation, GroupInput, TRIT_BITS, allocate};
pub use plane::{Plane, PlaneStack, absmean_ternary, recon_error, residual_expand, ternary_at_scale};
pub use quantize::{QuantConfig, QuantError, QuantizedTensor, ScaleGroup, Sensitivity, quantize_tensor};
