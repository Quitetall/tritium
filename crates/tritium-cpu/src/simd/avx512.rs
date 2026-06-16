//! AVX-512 / VNNI ternary mpGEMM (v0.30, WF-C — skeleton).
//!
//! Planned: a 512-bit-wide sibling of the AVX2 kernel in [`crate::kernel`] —
//! decode TQ2_0 trits with `vpshufb`/mask ops over `__m512i`, accumulate the
//! add/sub contributions, and use VNNI (`vpdpbusd`) / AMX tiles for the int8
//! activation path. Runtime-selected via `is_x86_feature_detected!("avx512f")`
//! (and `avx512vnni`/`amx-int8` where present), falling back to AVX2 then scalar.
//!
//! Held to the cross-ISA parity gate: the AVX-512 accumulation must match the
//! scalar reference bit-for-bit (same fold order), exactly as the AVX2 kernel
//! does today. No code yet — WF-C fills this in.
