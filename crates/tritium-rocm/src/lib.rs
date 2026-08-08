//! # tritium-rocm
//!
//! AMD ROCm / HIP execution backend for Tritium. The host side is raw
//! `extern "C"` FFI to the HIP runtime (declared in [`ffi`]); the addition-only
//! ternary mpGEMM kernel is compiled from `kernels/tq2_0_add.hip` to an AMD GPU
//! code object by `build.rs` (via `hipcc`) and loaded at runtime with
//! [`include_bytes!`] + `hipModuleLoadData`.
//!
//! ## Feature gating — `--features rocm`
//!
//! **All GPU code in this crate is behind the `rocm` cargo feature.** The default
//! build (`cargo build -p tritium-rocm`) compiles a HIP-free stub: it pulls in no
//! HIP FFI, runs no `hipcc`, and registers no backend. This is what cpu-only CI
//! lanes (and developer machines without an AMD GPU) build — mirroring
//! tritium-cuda's `cuda` gate exactly.
//!
//! Building with `--features rocm` **requires a full ROCm toolkit** (so `build.rs`
//! can find `hipcc` to emit the code object) and a working **AMD GPU + ROCm
//! driver** at runtime (so the backend's `init` can open device 0 via `hipInit`).
//! Neither is present on cpu-only lanes, so the feature is validated on a
//! **separate ROCm CI lane** (`runs-on: [self-hosted, rocm]`), never in the default
//! build matrix.
//!
//! ## Raw FFI vs a binding crate
//!
//! The host bindings are raw `extern "C"` declarations of the small HIP runtime
//! surface this backend uses (`hipInit`, `hipMalloc`, `hipMemcpy`,
//! `hipModuleLoadData`, `hipModuleGetFunction`, `hipModuleLaunchKernel`, …) rather
//! than an external HIP binding crate. The published binding crates (`hip-sys`,
//! `hip-runtime-sys`) are thin and pin to specific ROCm releases — a stale pin can
//! fail to build against the target box's actual ROCm version. Raw FFI against the
//! ABI-stable HIP symbol set is version-agnostic at compile time, fully under our
//! control, and adds zero external dependencies — so it can never be pulled into
//! the default Linux build. This is the same strategy cudarc itself uses to wrap
//! the CUDA driver.
//!
//! ## What it computes
//!
//! [`RocmBackend`] implements [`tritium_spec::TernaryBackend`]:
//! [`upload_weights`](tritium_spec::TernaryBackend::upload_weights) copies the
//! host-packed TQ2_0 bytes to device memory (host-to-device), and
//! [`mpgemm`](tritium_spec::TernaryBackend::mpgemm) uploads activations + scales,
//! launches the kernel (one output element per thread), synchronizes, and copies
//! the result back. The result matches [`tritium_core::reference_mpgemm`] within
//! the `1e-4` relative tolerance from ADR 0002.
//!
//! Only [`TernaryFormat::Tq2_0`](tritium_core::TernaryFormat::Tq2_0) is supported
//! by the kernel; other formats return
//! [`BackendError::UnsupportedFormat`](tritium_spec::BackendError::UnsupportedFormat).
//!
//! ## Lints
//!
//! Unlike the pure-Rust crates, this crate does **not** `#![deny(unsafe_code)]`:
//! the `rocm`-gated [`rocm`]/[`ffi`] modules contain the raw HIP-runtime FFI (the
//! kernel launch + memcpy + module calls), each in a narrowly scoped `unsafe` block
//! with a `SAFETY:` justification — the same trade-off tritium-cuda makes. Every
//! public item is documented (`#![deny(missing_docs)]`).
#![deny(missing_docs)]

// The default (no-`rocm`) build is intentionally inert: it carries only docs and a
// compile/type sanity test. Everything device-facing lives in the `rocm` module,
// which contains the narrowly scoped `unsafe` HIP FFI (the launch + the raw runtime
// calls), so the crate does not blanket-`deny(unsafe_code)` like the pure-Rust
// crates do; the FFI module documents every `unsafe` block instead.
// Host-side structure for the v3 prefill-attention port (Track E2): launch
// constants (pinned to kernels/gqa_attention_v3.hip by test), dispatch
// geometry, the TRITIUM_ATTN_V3 kill-switch parser, the pinned-order CPU
// reference the device kernel is gated against on the MI300X lane, the MFMA
// int8 design memo, and the cloud-session runbook. Plain Rust, no HIP
// dependency — compiled and unit-tested on every platform (the
// CPU-verifiable half of the port), exactly like tritium-metal's `attn`.
pub mod attn;

#[cfg(feature = "rocm")]
mod ffi;
#[cfg(feature = "rocm")]
mod rocm;

#[cfg(feature = "rocm")]
pub use rocm::{RocmBackend, RocmBuffer};

/// Registry `init` constructor — returns `Err` (logged + skipped by the registry)
/// when no AMD device is available / `hipInit` fails.
///
/// # Errors
/// [`tritium_spec::BackendError`] when no compatible AMD/ROCm device exists.
#[cfg(feature = "rocm")]
fn init_rocm() -> Result<Box<dyn tritium_spec::TernaryBackend>, tritium_spec::BackendError> {
    Ok(Box::new(RocmBackend::new(0)?))
}

// Self-register into the runtime's distributed slice, but only with the `rocm`
// feature. `linkme`'s `distributed_slice` expands to a `#[link_section]` static
// that trips the `unsafe_code` lint, hence the scoped allow (same pattern as
// tritium-cuda / tritium-runtime self-registrations).
#[cfg(feature = "rocm")]
#[allow(unsafe_code)]
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static ROCM: tritium_runtime::BackendEntry = tritium_runtime::BackendEntry {
    name: "rocm",
    init: init_rocm,
};

#[cfg(test)]
mod tests {
    // Default-build sanity: the crate compiles, the spec types it is written
    // against resolve, and (without the `rocm` feature) no backend is registered
    // by this crate. The real conformance test is `rocm::tests`, gated on `rocm`
    // and exercised only on the ROCm lane.
    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::BackendError;

    #[test]
    fn spec_types_resolve_in_default_build() {
        // These are the exact types the backend's signatures are written against;
        // referencing them keeps the default build honest about the contract.
        let shape = GemmShape { m: 2, n: 3, k: 256 };
        assert_eq!(shape.m * shape.n, 6);
        // TQ2_0 is 2 raw bits/trit plus the per-block f16 scale amortised over 256
        // trits, so effective bpw is slightly above 2.0 — just assert the range.
        let bpw = TernaryFormat::Tq2_0.bits_per_weight();
        assert!((2.0..2.1).contains(&bpw), "tq2_0 bpw {bpw} out of range");
        let err = BackendError::UnsupportedFormat(TernaryFormat::Tq1_0);
        assert!(matches!(err, BackendError::UnsupportedFormat(_)));
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn default_build_registers_no_rocm_backend() {
        // Without the `rocm` feature this crate contributes nothing to the runtime
        // registry, so a `"rocm"` lookup must miss. (Other crates in a workspace
        // build may register their own backends; we only assert the negative for
        // ours.)
        let reg = tritium_runtime::Registry::init();
        assert!(
            reg.get("rocm").is_none(),
            "default build must not register a rocm backend"
        );
    }
}
