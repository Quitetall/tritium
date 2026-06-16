//! SIMD kernel variants for the CPU backend (v0.30, ADR 0005 / WF-C).
//!
//! Wider-ISA and lookup-table ternary mpGEMM paths layered on top of the v0.10
//! AVX2 + scalar kernels in [`crate::kernel`]:
//!
//! - [`avx512`] — AVX-512 / VNNI (and AMX for the int8 activations), x86-64.
//! - [`neon`]   — ARM NEON (`vqtbl`), aarch64.
//! - [`lut`]    — the T-MAC precomputed-partial-sum lookup table (`vpermb` /
//!   `vpshufb` on x86, `vqtbl` on NEON), ISA-agnostic table build + per-ISA gather.
//!
//! These are **empty skeleton modules** today: the crate tree and the dispatch
//! extension point in [`crate::CpuBackend::mpgemm`] are in place, but no wider
//! kernel is wired yet, so v0.20 numerics are unchanged. WF-C implements each
//! behind the existing `is_x86_feature_detected!` / `target_arch` dispatch and
//! holds them to the cross-ISA parity gate (AVX2 == AVX-512 == NEON == scalar).

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

pub(crate) mod lut;
