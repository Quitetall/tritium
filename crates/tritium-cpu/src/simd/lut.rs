//! T-MAC lookup-table ternary mpGEMM (v0.30, WF-C — skeleton).
//!
//! Planned: the Microsoft T-MAC scheme — instead of multiplying, precompute the
//! partial sums of activations over every possible short ternary sub-pattern and
//! gather them by table lookup. A group of `g` ternary weights indexes one of
//! `3^g` precomputed activation partial sums; the kernel then sums the gathered
//! partials. The table build is ISA-agnostic arithmetic over the activation row;
//! the gather is `vpermb`/`vpshufb` on x86 and `vqtbl` on NEON (hence this module
//! is shared and the per-ISA gather lives in [`super::avx512`] / [`super::neon`]).
//!
//! This trades multiplies for table lookups and is the deferred-from-0.10 LUT
//! path. Held to the same vs-reference correctness bar as every other kernel. No
//! code yet — WF-C fills this in.
