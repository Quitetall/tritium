//! GPU mpGEMM microbenchmarks (v0.30 WF-E, ADR 0005) — divan, GPU-feature-gated.
//!
//! Two kernel families over the BitNet 2B4T linear-layer shapes
//! ([`tritium_benches::BITNET_SHAPES`]: `M ∈ {1,8,32,256,512}`, `(N,K)` ∈
//! {2560, 6912}²):
//!
//! 1. **Add-only** ([`gpu_add_only`]) — the TQ2_0 kernel via
//!    [`TernaryBackend::mpgemm`], run once per (backend, shape) pair across every
//!    GPU backend compiled in. On CUDA `mpgemm` auto-selects by shape: the **tiled**
//!    decode kernel for `M ≤ 64` (so `M ∈ {1,8,32}` here) and the **simple**
//!    one-thread-per-output kernel for larger `M` (`{256,512}`); one bench per shape
//!    covers both kernels across the crossover.
//! 2. **IMMA int8** ([`gpu_imma`]) — the `mma.m16n8k32` tensor-core fused-A8 kernel
//!    via [`TernaryBackend::mpgemm_with_act_quant`] on an [`TernaryFormat::I2sInt8`]
//!    buffer: on-device per-token int8 quant → int8 contraction → scale fold. This
//!    one is **CUDA-only by construction** — it is a PTX tensor-core path with no
//!    wgpu or HIP equivalent — so it stays gated on `cuda` rather than pretending to
//!    generalise and printing a skip line per shape on other backends.
//!
//! Each bench reports an `ItemsCount` of `M·N·K` MACs so divan prints throughput.
//! Bodies are gated on the GPU features; with none of them this file is an empty
//! `divan::main()` so it still compiles + links on cpu-only lanes. With a feature but
//! no matching device, the harness self-skips (constructing the backend returns
//! `Err`, so the bench function early-returns before timing anything).

#![cfg_attr(
    not(all(
        feature = "cuda",
        feature = "wgpu",
        feature = "rocm",
        feature = "metal"
    )),
    allow(unused_crate_dependencies)
)]

fn main() {
    divan::main();
}

#[cfg(any(
    feature = "cuda",
    feature = "wgpu",
    feature = "rocm",
    feature = "metal"
))]
mod gpu_benches {
    use divan::{Bencher, counter::ItemsCount};
    use tritium_benches::{BITNET_SHAPES, gemm_macs, packed_tq2_0_weights};
    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::{DeviceBuffer, MpGemm, TernaryBackend};

    // Linked so each backend's `#[distributed_slice]` registration is included in the
    // bench binary (the same edge `tests/acceptance.rs` relies on). Registration is
    // what `backend_named` below resolves against — without the link the name is
    // simply absent from `BACKENDS`.
    #[cfg(feature = "cuda")]
    use tritium_cuda as _;
    #[cfg(feature = "metal")]
    use tritium_metal as _;
    #[cfg(feature = "rocm")]
    use tritium_rocm as _;
    #[cfg(feature = "wgpu")]
    use tritium_wgpu as _;

    /// Every GPU backend compiled into this binary, in registry-name form. Built from
    /// the enabled features rather than probed at runtime, so a box with two usable
    /// backends benches both and the divan output names which one produced each row.
    const GPU_BACKENDS: &[&str] = &[
        #[cfg(feature = "cuda")]
        "cuda",
        #[cfg(feature = "wgpu")]
        "wgpu",
        #[cfg(feature = "rocm")]
        "rocm",
        #[cfg(feature = "metal")]
        "metal",
    ];

    /// One benchmark case: which backend runs it, and at what shape. `Debug` is what
    /// divan renders as the case label, so each row reads `cuda 1x2560x2560`.
    #[derive(Clone, Copy)]
    struct GpuCase {
        backend: &'static str,
        m: usize,
        n: usize,
        k: usize,
    }

    impl std::fmt::Debug for GpuCase {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} {}x{}x{}", self.backend, self.m, self.n, self.k)
        }
    }

    /// The (backend × shape) cross product — the full add-only sweep.
    fn gpu_cases() -> impl Iterator<Item = GpuCase> {
        GPU_BACKENDS.iter().flat_map(|&backend| {
            BITNET_SHAPES
                .iter()
                .map(move |&(m, n, k)| GpuCase { backend, m, n, k })
        })
    }

    /// Construct an owned backend by registry name, or `None` (with a printed reason)
    /// if it is unregistered or its device does not initialise — the GPU-less skip
    /// path, mirroring `tests/acceptance.rs::load_on`.
    fn backend_named(name: &str) -> Option<Box<dyn TernaryBackend>> {
        let init = tritium_runtime::BACKENDS
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.init)?;
        match init() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!(
                    "skipping gpu mpgemm bench: {name} backend init failed ({e}); no device?"
                );
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

    /// Add-only TQ2_0 mpGEMM across every (GPU backend, BitNet shape) pair. TQ2_0 is
    /// the one format all three backends implement, which is what makes the numbers
    /// comparable across vendors. Reports `M·N·K` MACs/iter.
    #[divan::bench(args = gpu_cases())]
    fn gpu_add_only(bencher: Bencher, case: &GpuCase) {
        let &GpuCase {
            backend: name,
            m,
            n,
            k,
        } = case;
        let shape = GemmShape { m, n, k };
        let Some(backend) = backend_named(name) else {
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
                    .mpgemm(MpGemm {
                        act: &act,
                        weights: weights.as_ref(),
                        scales: &scales,
                        shape,
                        format: TernaryFormat::Tq2_0,
                        out: &mut out,
                    })
                    .expect("add-only mpgemm");
            });
    }

    /// IMMA int8 fused-A8 mpGEMM across every BitNet shape, via
    /// `mpgemm_with_act_quant` on an `I2sInt8` buffer (on-device quant + tensor-core
    /// contraction). CUDA-only — see the module docs. Reports `M·N·K` MACs/iter.
    /// Skips a shape whose `I2sInt8` upload the backend rejects (it never should for
    /// these shapes).
    #[cfg(feature = "cuda")]
    #[divan::bench(args = BITNET_SHAPES)]
    fn gpu_imma(bencher: Bencher, shape: &(usize, usize, usize)) {
        use tritium_benches::packed_i2s_int8_weights;

        let &(m, n, k) = shape;
        let shape = GemmShape { m, n, k };
        let Some(backend) = backend_named("cuda") else {
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
        if let Err(e) = backend.mpgemm_with_act_quant(MpGemm {
            act: &act,
            weights: weights.as_ref(),
            scales: &weight_scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut out,
        }) {
            eprintln!("skipping shape {shape:?}: IMMA warm-up failed ({e})");
            return;
        }

        bencher
            .counter(ItemsCount::new(gemm_macs(m, n, k)))
            .bench_local(|| {
                backend
                    .mpgemm_with_act_quant(MpGemm {
                        act: &act,
                        weights: weights.as_ref(),
                        scales: &weight_scales,
                        shape,
                        format: TernaryFormat::I2sInt8,
                        out: &mut out,
                    })
                    .expect("imma mpgemm_with_act_quant");
            });
    }
}
