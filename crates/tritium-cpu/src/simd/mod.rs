//! SIMD kernel variants for the CPU backend (v0.30, ADR 0005 / WF-C).
//!
//! Wider-ISA and lookup-table ternary mpGEMM paths layered on top of the v0.10
//! AVX2 + scalar kernels in [`crate::kernel`]:
//!
//! - [`avx512`] — AVX-512 ternary mpGEMM, x86-64. A 16-wide `f32` sibling of the
//!   AVX2 kernel, decoding trits with `__mmask16` ops; selected when the host has
//!   `avx512f` + `avx512bw` + `avx512vl`. Compiles on every x86-64 target, runs
//!   only on an AVX-512 host. (VNNI / AMX int8 acceleration is a later step.)
//! - [`neon`]   — ARM NEON ternary mpGEMM, aarch64. A 4-wide `f32` sibling using
//!   `vcgtq`/`vbslq`; `#[cfg(target_arch = "aarch64")]`, so absent from the x86
//!   build. (SDOT/UDOT int8 acceleration is a later step.)
//! - [`lut`]    — the T-MAC precomputed-partial-sum lookup table, ISA-agnostic.
//!   The table build + base-3 gather is pure safe arithmetic; the per-ISA SIMD
//!   gather (`vpermb`/`vpshufb` on x86, `vqtbl` on NEON) that puts it on the hot
//!   path is a later step. Implemented and unit-tested here; not yet on the
//!   [`crate::kernel`] dispatch path (the bit-exact scalar reference is the
//!   terminal fallback until the gather lands).
//!
//! ## Parity bar (load-bearing)
//!
//! The two **per-element** kernels (AVX-512, NEON) fold their signed
//! contributions sequentially in `f32` k-order, so they reproduce the scalar
//! reference **bit-for-bit**, exactly as the AVX2 kernel does — the cross-ISA
//! parity gate (AVX2 == AVX-512 == NEON == scalar) is bit-exact. The **LUT**
//! kernel re-associates the `K` additions into group partials, so it agrees with
//! the reference within the ADR 0002 `1e-4` tolerance (and is deterministic), but
//! not bit-for-bit; the per-group table itself *is* bit-exact vs the direct group
//! sum. See each module for the full argument.
//!
//! Each is wired into [`crate::kernel::dispatch_mpgemm`] behind the existing
//! `is_x86_feature_detected!` / `target_arch` dispatch.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

pub(crate) mod lut;
