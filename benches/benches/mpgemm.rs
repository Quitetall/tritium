//! Divan microbenchmarks for CPU ternary mpGEMM (v0.30 skeleton; WF-E extends).
//!
//! One bench today: the v0.10 CPU backend `mpgemm` over a couple of `K` sizes, so
//! there is a baseline number to track. WF-E adds the GPU add-only / IMMA benches,
//! the fused-A8 path, the end-to-end tokens/sec bench, and the regression gate.

use divan::Bencher;
use tritium_benches::packed_tq2_0_weights;
use tritium_core::{GemmShape, TernaryFormat};
use tritium_cpu::CpuBackend;
use tritium_spec::{MpGemm, TernaryBackend};

fn main() {
    divan::main();
}

/// Decode-shaped (M=1) ternary mpGEMM across a small and a large contraction.
#[divan::bench(args = [256, 1024])]
fn cpu_mpgemm_tq2_0_decode(bencher: Bencher, k: usize) {
    let (m, n) = (1usize, 64usize);
    let backend = CpuBackend::new();
    let packed = packed_tq2_0_weights(n, k);
    let shape = GemmShape { m, n, k };
    let weights = backend
        .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let act = vec![0.5_f32; m * k];
    let scales = vec![1.0_f32; n];
    let mut out = vec![0.0_f32; m * n];

    bencher.bench_local(move || {
        backend
            .mpgemm(MpGemm {
                act: &act,
                weights: weights.as_ref(),
                scales: &scales,
                shape,
                format: TernaryFormat::Tq2_0,
                out: &mut out,
            })
            .expect("mpgemm");
    });
}
