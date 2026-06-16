//! GPU mpGEMM microbenchmarks (v0.30 WF-E, ADR 0005) — divan, `cuda`-gated.
//!
//! Two kernel families over the BitNet 2B4T linear-layer shapes
//! ([`tritium_benches::BITNET_SHAPES`]: `M ∈ {1,8,32,256,512}`, `(N,K)` ∈
//! {2560, 6912}²):
//!
//! 1. **Add-only** ([`gpu_add_only`]) — the TQ2_0 CUDA-core kernel via
//!    [`TernaryBackend::mpgemm`], which auto-selects by shape: the **tiled** decode
//!    kernel for `M ≤ 64` (so `M ∈ {1,8,32}` here) and the **simple**
//!    one-thread-per-output kernel for larger `M` (`{256,512}`). One bench per shape
//!    covers both kernels across the crossover.
//! 2. **IMMA int8** ([`gpu_imma`]) — the `mma.m16n8k32` tensor-core fused-A8 kernel
//!    via [`TernaryBackend::mpgemm_with_act_quant`] on an [`TernaryFormat::I2sInt8`]
//!    buffer: on-device per-token int8 quant → int8 contraction → scale fold.
//!
//! Each bench reports an `ItemsCount` of `M·N·K` MACs so divan prints throughput.
//! Bodies are `#[cfg(feature = "cuda")]`; without the feature this file is an empty
//! `divan::main()` so it still compiles + links on cpu-only lanes. With the feature
//! but no GPU, the harness self-skips (constructing the backend returns `Err`, so the
//! bench function early-returns before timing anything).

#![cfg_attr(not(feature = "cuda"), allow(unused_crate_dependencies))]

fn main() {
    divan::main();
}

#[cfg(feature = "cuda")]
mod cuda_benches {
    use divan::{Bencher, counter::ItemsCount};
    use tritium_benches::{
        BITNET_SHAPES, gemm_macs, packed_i2s_int8_weights, packed_tq2_0_weights,
    };
    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::{DeviceBuffer, TernaryBackend};

    // Linked so the CUDA backend's `#[distributed_slice]` registration is included in
    // the bench binary (the same edge `tests/acceptance.rs` relies on).
    use tritium_cuda as _;

    /// Construct an owned `"cuda"` backend through the runtime registry, or `None`
    /// (with a printed reason) if no CUDA device initialises — the GPU-less skip path,
    /// mirroring `tests/acceptance.rs::load_on`.
    fn cuda_backend() -> Option<Box<dyn TernaryBackend>> {
        let init = tritium_runtime::BACKENDS
            .iter()
            .find(|e| e.name == "cuda")
            .map(|e| e.init)?;
        match init() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping gpu mpgemm bench: cuda backend init failed ({e}); no device?");
                None
            }
        }
    }

    /// Upload `packed` weights of `format` for `shape`, or `None` on an upload error
    /// (printed). Shared setup for both kernel families.
    fn upload(
        backend: &dyn TernaryBackend,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Option<Box<dyn DeviceBuffer>> {
        match backend.upload_weights(packed, shape, format) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping shape {shape:?} ({format:?}): upload failed ({e})");
                None
            }
        }
    }

    /// Add-only TQ2_0 mpGEMM across every BitNet shape. `mpgemm` auto-selects the
    /// tiled (decode, `M ≤ 64`) or simple (prefill) kernel, so the one bench exercises
    /// both across the crossover. Reports `M·N·K` MACs/iter.
    #[divan::bench(args = BITNET_SHAPES)]
    fn gpu_add_only(bencher: Bencher, shape: &(usize, usize, usize)) {
        let &(m, n, k) = shape;
        let shape = GemmShape { m, n, k };
        let Some(backend) = cuda_backend() else {
            return;
        };
        let packed = packed_tq2_0_weights(n, k);
        let Some(weights) = upload(backend.as_ref(), &packed, shape, TernaryFormat::Tq2_0) else {
            return;
        };

        let act = vec![0.5_f32; m * k];
        let scales = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; m * n];

        bencher
            .counter(ItemsCount::new(gemm_macs(m, n, k)))
            .bench_local(|| {
                backend
                    .mpgemm(
                        &act,
                        weights.as_ref(),
                        &scales,
                        shape,
                        TernaryFormat::Tq2_0,
                        &mut out,
                    )
                    .expect("add-only mpgemm");
            });
    }

    /// IMMA int8 fused-A8 mpGEMM across every BitNet shape, via
    /// `mpgemm_with_act_quant` on an `I2sInt8` buffer (on-device quant + tensor-core
    /// contraction). Reports `M·N·K` MACs/iter. Skips a shape whose `I2sInt8` upload
    /// the backend rejects (it never should for these shapes).
    #[divan::bench(args = BITNET_SHAPES)]
    fn gpu_imma(bencher: Bencher, shape: &(usize, usize, usize)) {
        let &(m, n, k) = shape;
        let shape = GemmShape { m, n, k };
        let Some(backend) = cuda_backend() else {
            return;
        };
        let packed = packed_i2s_int8_weights(n, k);
        let Some(weights) = upload(backend.as_ref(), &packed, shape, TernaryFormat::I2sInt8) else {
            return;
        };

        // f32 activations the on-device kernel quantizes to per-token int8 itself.
        let act = vec![0.5_f32; m * k];
        let weight_scales = vec![1.0_f32; n];
        let mut out = vec![0.0_f32; m * n];

        // One warm-up launch before the timed loop: it triggers any first-use JIT of
        // the tuned tile so compilation does not land inside the timed region, and it
        // lets the bench skip cleanly on *any* warm-up failure (unsupported IMMA path,
        // a device hiccup, …) rather than panicking inside `bench_local`.
        if let Err(e) = backend.mpgemm_with_act_quant(
            &act,
            weights.as_ref(),
            &weight_scales,
            shape,
            TernaryFormat::I2sInt8,
            &mut out,
        ) {
            eprintln!("skipping shape {shape:?}: IMMA warm-up failed ({e})");
            return;
        }

        bencher
            .counter(ItemsCount::new(gemm_macs(m, n, k)))
            .bench_local(|| {
                backend
                    .mpgemm_with_act_quant(
                        &act,
                        weights.as_ref(),
                        &weight_scales,
                        shape,
                        TernaryFormat::I2sInt8,
                        &mut out,
                    )
                    .expect("imma mpgemm_with_act_quant");
            });
    }
}
