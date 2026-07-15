//! # tritium-cuda
//!
//! CUDA execution backend for Tritium. The host side is built on
//! [`cudarc`](https://docs.rs/cudarc); the addition-only ternary mpGEMM kernel is
//! compiled from `kernels/tq2_0_add.cu` to PTX by `build.rs` (via `nvcc`) and
//! loaded at runtime with [`include_str!`].
//!
//! ## Feature gating — `--features cuda`
//!
//! **All GPU code in this crate is behind the `cuda` cargo feature.** The default
//! build (`cargo build -p tritium-cuda`) compiles a CUDA-free stub: it pulls in no
//! `cudarc`, runs no `nvcc`, and registers no backend. This is what cpu-only CI
//! lanes (and developer machines without a GPU) build.
//!
//! Building with `--features cuda` **requires a full CUDA toolkit** (so `build.rs`
//! can find `nvcc` to emit PTX) and a working **NVIDIA GPU + driver** at runtime
//! (so the backend's `init` can open device 0). Neither is present on cpu-only
//! lanes, so the feature is validated on a **separate GPU CI lane (Wave D)**, never
//! in the default build matrix.
//!
//! ## What it computes
//!
//! [`CudaBackend`] implements [`tritium_spec::TernaryBackend`]:
//! [`upload_weights`](tritium_spec::TernaryBackend::upload_weights) copies the
//! host-packed TQ2_0 bytes to device memory (host-to-device), and
//! [`mpgemm`](tritium_spec::TernaryBackend::mpgemm) uploads activations + scales,
//! launches the kernel (one output element per thread), synchronizes, and copies
//! the result back. The result matches [`tritium_core::reference_mpgemm`] within
//! the `1e-4` relative tolerance from ADR 0002.
//!
//! Only [`TernaryFormat::Tq2_0`](tritium_core::TernaryFormat::Tq2_0) is supported
//! by the kernel; TQ1_0 returns
//! [`BackendError::UnsupportedFormat`](tritium_spec::BackendError::UnsupportedFormat).
#![deny(unsafe_code)]

// The default (no-`cuda`) build is intentionally inert: it carries only docs and a
// compile/type sanity test. Everything device-facing lives in the `cuda` module.
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaDeviceIdentity, CudaMemorySnapshot, CudaMemoryTelemetry};

// v0.3.1 (ADR 0013): the device-resident M=1 decode forward. The runner downcasts
// its `dyn TernaryBackend` to `CudaBackend`, builds a `CudaDecodeModel` once from a
// borrowed `DecodeModelSpec`, then drives it per token — keeping the residual stream
// + KV cache in VRAM across all layers.
#[cfg(feature = "cuda")]
pub use cuda::{
    BatchKv, CudaDecodeModel, DecodeLayerSpec, DecodeLinearSpec, DecodeModelSpec, KV_PAGE_TOKENS,
};

// v0.4.0: the resident SALT projection primitive — upload multi-plane TQ2_0 weights
// plane-major once, then run the `salt_mpgemm_tiled_f32` kernel against them. The
// building block a SALT decode forward composes per projection.
#[cfg(feature = "cuda")]
pub use cuda::SaltResidentLinear;

// plan 0043 Stage 6: direct encoded D2/B3/S34 execution with explicit allocation
// evidence. `FastAliasesExact` remains visible so callers cannot mistake the
// correctness-first alias for an optimized kernel.
#[cfg(feature = "cuda")]
pub use cuda::{
    SaltV2Forward, SaltV2ForwardMode, SaltV2ForwardReceipt, SaltV2ResidentAllocationReceipt,
    SaltV2ResidentTensor,
};

// v0.30 (ADR 0005) skeletons. `autotune` is pure Rust (tile config + on-disk
// cache keying) so it builds and tests on cpu-only lanes; `codegen` links
// cudarc's nvrtc path, so it is gated behind `cuda`. WF-B implements both and the
// IMMA kernel (WF-A) selects a tuned tile through them.
mod autotune;

#[cfg(feature = "cuda")]
mod codegen;

// v0.60 / plan 0013: the GPU QAT training step + tiny-model pretrain smoke — the first real
// consumer of the `train_grad.cu` forward + gradient kernels.
#[cfg(feature = "cuda")]
pub mod train;

// v0.60 / plan 0017 (the ≥2-GPU wall): the real NCCL collective backend — a
// `tritium_train::dist::ProcessGroup` over `cudarc::nccl`, validated against the simulated reference.
// Behind the `nccl` feature (implies `cuda`); needs `libnccl >= 2.30`.
#[cfg(feature = "nccl")]
pub mod nccl;

#[cfg(feature = "nccl")]
pub use nccl::{NcclId, NcclProcessGroup};

#[cfg(test)]
mod tests {
    // Default-build sanity: the crate compiles, the spec types it is written
    // against resolve, and (without the `cuda` feature) no backend is registered
    // by this crate. The real conformance test is `cuda::tests`, gated on `cuda`
    // and exercised only on the GPU lane.
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

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_registers_no_cuda_backend() {
        // Without the `cuda` feature this crate contributes nothing to the runtime
        // registry, so a `"cuda"` lookup must miss. (Other crates in a workspace
        // build may register their own backends; we only assert the negative for
        // ours.)
        let reg = tritium_runtime::Registry::init();
        assert!(
            reg.get("cuda").is_none(),
            "default build must not register a cuda backend"
        );
    }
}
