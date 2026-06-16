//! ARM NEON ternary mpGEMM (v0.30, WF-C — skeleton).
//!
//! Planned: an aarch64 sibling of the AVX2 kernel — decode TQ2_0 trits and gather
//! T-MAC partial sums with `vqtbl` (table lookup), accumulate add/sub
//! contributions over `int8x16_t`/`int32x4_t`, and use the SDOT/UDOT dot-product
//! instructions for the int8 activation path. Selected on `target_arch =
//! "aarch64"` with `std::arch::is_aarch64_feature_detected!` for the optional
//! dot-product extension; falls back to the scalar reference otherwise.
//!
//! Held to the cross-ISA parity gate: NEON output must match the scalar
//! reference (and the x86 kernels) within the documented tolerance. No code yet —
//! WF-C fills this in.
