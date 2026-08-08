//! # tritium-metal — MSL ternary mpGEMM over Apple Metal (Apple Silicon).
//!
//! The macOS sibling of [`tritium-wgpu`](https://docs.rs/tritium-wgpu): all GPU
//! code is behind `--features metal` (mirrors tritium-cuda's `cuda` gate and
//! tritium-wgpu's `wgpu` gate) AND `#[cfg(target_os = "macos")]`. The default
//! build — and *every* non-macOS build, including the cpu-only Linux CI matrix —
//! is an inert stub: it pulls in no [`metal`](https://docs.rs/metal) binding,
//! opens no device, and registers no backend. The Metal binding dependency itself
//! lives under `[target.'cfg(target_os = "macos")'.dependencies]`, so a Linux
//! `cargo build`/`cargo tree` never even resolves it.
//!
//! `--features register` additionally self-registers a `"metal"` backend into the
//! runtime `BACKENDS` slice via `linkme` (macOS only).
//!
//! ## What it computes
//!
//! The kernel ([`mpgemm.metal`](../src/mpgemm.metal)) computes one
//! `out[m,n] = scales[n] · Σ_k act[m,k] (±) w[n,k]` per thread, in the exact
//! add/subtract/skip form of [`tritium_core::reference_mpgemm`] (a direct port of
//! the tritium-wgpu WGSL kernel), so f32 round-off stays within the 1e-4
//! conformance bar. TQ2_0 weights stay packed on device and are decoded in-kernel
//! (device-memory parity with the cuda/rocm backends — ~2.06 bit/trit, so large
//! models fit unified memory); TQ1_0 is host-unpacked and widened to one `i32` per
//! trit. Either way the weights live in a shared-storage `MTLBuffer` (unified memory
//! on Apple Silicon — no discrete copy). The MSL source is compiled at runtime with
//! `newLibraryWithSource:`.
//!
//! ## Unsafe
//!
//! The metal-rs handles ([`metal::Device`], [`metal::CommandQueue`], …) are
//! `foreign-types` pointer wrappers and are neither `Send` nor `Sync`, but
//! [`tritium_spec::TernaryBackend`] requires both. Apple documents `MTLDevice`,
//! `MTLCommandQueue`, and `MTLComputePipelineState` as thread-safe for concurrent
//! use, so the backend wraps its handles in a newtype with a narrowly-scoped,
//! documented `unsafe impl Send + Sync`. That is the only hand-written `unsafe`
//! in the crate (plus the `linkme` registration static, same as every backend);
//! the crate `deny`s — not `forbid`s — unsafe so those two sites are allowed
//! explicitly.
#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]
#![cfg_attr(target_os = "macos", deny(unsafe_code))]
#![deny(missing_docs)]

// Host-side structure for the v3 prefill-attention port: launch constants
// (pinned to attention.metal by test), dispatch geometry, the TRITIUM_ATTN_V3
// kill-switch parser, and the pinned-order CPU reference the device kernel is
// gated against on the Mac lane. Plain Rust, no Metal dependency — compiled
// and unit-tested on every platform (the CPU-verifiable half of the port).
pub mod attn;

// All device code lives in the `backend` module, gated on BOTH the `metal`
// feature and macOS. On Linux (or any non-macOS target) the module — and the
// metal-rs dep it uses — is compiled out entirely, leaving only the pure-Rust
// `attn` module above.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod backend;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{MetalBackend, MetalBuffer};

/// Registry `init` constructor — returns `Err` (logged + skipped by the registry)
/// when no Metal device is available (`MTLCreateSystemDefaultDevice()` is nil).
///
/// # Errors
/// [`tritium_spec::BackendError`] when no Metal device exists or pipeline setup
/// fails.
#[cfg(all(feature = "register", target_os = "macos"))]
fn init_metal() -> Result<Box<dyn tritium_spec::TernaryBackend>, tritium_spec::BackendError> {
    Ok(Box::new(MetalBackend::new()?))
}

#[cfg(all(feature = "register", target_os = "macos"))]
#[allow(unsafe_code)] // distributed_slice expands to a #[link_section] static
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static METAL: tritium_runtime::BackendEntry = tritium_runtime::BackendEntry {
    name: "metal",
    init: init_metal,
};

#[cfg(test)]
mod tests {
    // Default-build sanity: the crate compiles, the spec types it is written
    // against resolve, and (without the `metal` feature, or off macOS) no backend
    // is registered by this crate. The real conformance test is `backend::tests`,
    // gated on `metal` + macOS and exercised only on the metal CI lane.
    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::BackendError;

    #[test]
    fn spec_types_resolve_in_default_build() {
        // These are the exact types the backend's signatures are written against;
        // referencing them keeps the default build honest about the contract.
        let shape = GemmShape { m: 2, n: 3, k: 256 };
        assert_eq!(shape.m * shape.n, 6);
        let bpw = TernaryFormat::Tq2_0.bits_per_weight();
        assert!((2.0..2.1).contains(&bpw), "tq2_0 bpw {bpw} out of range");
        let err = BackendError::UnsupportedFormat(TernaryFormat::Tq1_0);
        assert!(matches!(err, BackendError::UnsupportedFormat(_)));
    }

    // On any build where the metal backend is NOT compiled in (the default build,
    // or a non-macOS host even with `register`), this crate must contribute
    // nothing to the runtime registry, so a `"metal"` lookup must miss.
    #[cfg(not(all(feature = "register", target_os = "macos")))]
    #[test]
    fn default_build_registers_no_metal_backend() {
        let reg = tritium_runtime::Registry::init();
        assert!(
            reg.get("metal").is_none(),
            "default/non-macOS build must not register a metal backend"
        );
    }
}

// The `"metal"` entry actually links into the runtime BACKENDS slice under
// `--features register` on macOS (linkme silently drops entries from crates not
// pulled in, so this pins that the static is present). Mirrors tritium-wgpu's
// `register_tests`.
#[cfg(all(test, feature = "register", target_os = "macos"))]
mod register_tests {
    #[test]
    fn metal_entry_is_registered() {
        assert!(
            tritium_runtime::BACKENDS.iter().any(|e| e.name == "metal"),
            "the metal BackendEntry must be linked into BACKENDS"
        );
    }
}
