//! # tritium-wgpu — WGSL ternary mpGEMM over wgpu (Vulkan).
//!
//! All GPU code is behind `--features wgpu` (mirrors tritium-cuda's `cuda` gate);
//! the default build is an inert stub the cpu-only CI matrix compiles with no GPU
//! deps. `--features register` additionally self-registers a `"wgpu"` backend into
//! the runtime `BACKENDS` slice via `linkme`.
//!
//! The kernel ([`mpgemm.wgsl`](../src/mpgemm.wgsl)) computes one
//! `out[m,n] = scales[n] · Σ_k act[m,k] (±) w[n,k]` per invocation, in the exact
//! add/subtract/skip form of [`tritium_core::reference_mpgemm`] so f32 round-off
//! stays within the 1e-4 conformance bar. Validated on the RTX 4090 Vulkan adapter.
#![deny(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "wgpu")]
mod backend;
#[cfg(feature = "wgpu")]
pub use backend::{WgpuBackend, WgpuBuffer};

/// Registry `init` constructor — returns `Err` (logged + skipped by the registry)
/// when no Vulkan adapter/device is available.
///
/// # Errors
/// [`tritium_spec::BackendError`] when no compatible wgpu adapter/device exists.
#[cfg(feature = "register")]
fn init_wgpu() -> Result<Box<dyn tritium_spec::TernaryBackend>, tritium_spec::BackendError> {
    Ok(Box::new(WgpuBackend::new()?))
}

#[cfg(feature = "register")]
#[allow(unsafe_code)] // distributed_slice expands to a #[link_section] static
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static WGPU: tritium_runtime::BackendEntry = tritium_runtime::BackendEntry {
    name: "wgpu",
    init: init_wgpu,
};

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use crate::WgpuBackend;
    use tritium_testkit::{
        Tolerance, frozen_vectors, run_conformance, run_fused_fallback_contract,
    };

    /// Frozen-set conformance + fused-fallback on the Vulkan adapter, or a clean
    /// self-skip when no adapter is present (mirrors the CUDA conformance test).
    #[test]
    fn conformance_and_fused_fallback_or_skip() {
        let backend = match WgpuBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping wgpu conformance: no GPU adapter ({e})");
                return;
            }
        };
        let vectors = frozen_vectors();

        let report = run_conformance(&backend, &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} wgpu conformance failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all vectors must pass");

        let fused = run_fused_fallback_contract(&backend, &vectors, Tolerance::default());
        assert!(
            fused.is_ok(),
            "{} wgpu fused-fallback failures: {:?}",
            fused.failed.len(),
            fused.failed
        );
        assert_eq!(
            fused.passed,
            vectors.len(),
            "fused path must degrade cleanly"
        );
    }

    // ---- coverage beyond the frozen set (large shapes, zero dims) -------------

    use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};
    use tritium_spec::TernaryBackend;

    /// Build a deterministic random `[M,K]` activation + packed `[N,K]` tq2_0
    /// weights + `[N]` scales, plus the `reference_mpgemm` oracle output.
    fn random_case(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>) {
        // xorshift64 — no external rng, deterministic across runs/platforms.
        let mut s: u64 =
            0x9E37_79B9_7F4A_7C15 ^ ((m as u64) << 1) ^ ((n as u64) << 17) ^ (k as u64);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let act: Vec<f32> = (0..m * k)
            .map(|_| (next() as f32 / u64::MAX as f32) * 2.0 - 1.0)
            .collect();
        let trits: Vec<Trit> = (0..n * k)
            .map(|_| {
                let v = match next() % 3 {
                    0 => 0i8,
                    1 => 1,
                    _ => -1,
                };
                Trit::from_i8(v).expect("valid trit")
            })
            .collect();
        let scales: Vec<f32> = (0..n)
            .map(|_| (next() as f32 / u64::MAX as f32) + 0.5)
            .collect();

        let nb = num_blocks(k);
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let unit = vec![half::f16::ONE; nb];
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
        }

        let shape = GemmShape::new(m, n, k);
        let mut expected = vec![0.0f32; m * n];
        reference_mpgemm(&act, &trits, &scales, shape, &mut expected).expect("reference");
        (act, packed, scales, expected)
    }

    fn run_shape(backend: &WgpuBackend, m: usize, n: usize, k: usize) {
        let (act, packed, scales, expected) = random_case(m, n, k);
        let shape = GemmShape::new(m, n, k);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .expect("upload");
        let mut out = vec![0.0f32; m * n];
        backend
            .mpgemm(
                &act,
                buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .expect("mpgemm");
        let tol = Tolerance::default();
        for (i, (&g, &w)) in out.iter().zip(&expected).enumerate() {
            assert!(
                tol.accepts(g, w),
                "[{i}] got {g} want {w} (shape {m}x{n}x{k})"
            );
        }
    }

    /// A GEMM whose output count (M*N = 4_194_304) exceeds the old 65535-workgroup
    /// dispatch ceiling (65535*64 = 4_194_240). It would have panicked under
    /// `Limits::default()`; requesting `adapter.limits()` makes it run. This is the
    /// regression test for the review's dispatch/limits finding.
    #[test]
    fn large_shape_exceeding_default_dispatch_limit() {
        let backend = match WgpuBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        run_shape(&backend, 1024, 4096, 256); // M*N = 4_194_304 > 4_194_240
    }

    /// Zero-dimension shapes match the reference (empty output when M=0, or K=0 →
    /// all-zeros) without a wgpu zero-size-binding panic.
    #[test]
    fn zero_dims_match_reference() {
        let backend = match WgpuBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: no GPU adapter ({e})");
                return;
            }
        };
        run_shape(&backend, 0, 4, 256); // M=0 → empty output
        run_shape(&backend, 2, 3, 0); // K=0 → all-zeros (each out = scale·empty-sum)
    }
}

#[cfg(all(test, feature = "register"))]
mod register_tests {
    /// The `"wgpu"` backend entry actually links into the runtime BACKENDS slice
    /// under `--features register` (linkme silently drops entries from crates that
    /// aren't pulled in, so this pins that the static is present).
    #[test]
    fn wgpu_entry_is_registered() {
        assert!(
            tritium_runtime::BACKENDS.iter().any(|e| e.name == "wgpu"),
            "the wgpu BackendEntry must be linked into BACKENDS"
        );
    }
}
