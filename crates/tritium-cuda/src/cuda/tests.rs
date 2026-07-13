//! GPU conformance + CPU↔CUDA parity tests. Run only with `--features cuda` AND
//! a working CUDA device, so they are exercised on the Wave D GPU CI lane, never
//! on cpu-only lanes. When no device is present the tests self-skip
//! (constructing the backend returns `Err`) rather than failing.
//!
//! `run_conformance` itself packs each vector's trits to TQ2_0 (block scale
//! 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
//! scales, and grades against `reference_mpgemm` — so the test only has to
//! supply the TQ2_0 vectors this kernel supports.

use super::*;
use tritium_cpu::CpuBackend;
use tritium_testkit::{ConformanceVector, Tolerance, generate_vectors, run_conformance};

/// The full conformance set this kernel is responsible for: every TQ2_0 vector
/// from the committed generator (the kernel does not handle TQ1_0).
fn tq2_vectors() -> Vec<ConformanceVector> {
    let v: Vec<_> = generate_vectors(0xC0FFEE, 16)
        .into_iter()
        .filter(|v| v.format == TernaryFormat::Tq2_0)
        .collect();
    assert!(!v.is_empty(), "expected some tq2_0 conformance vectors");
    v
}

#[test]
fn cuda_driver_major_parses_driver_version() {
    assert_eq!(cuda_driver_major(13_030), Some(13));
    assert_eq!(cuda_driver_major(14_000), Some(14));
    assert_eq!(cuda_driver_major(0), None);
}

/// Deterministic xorshift f32 fill in `[lo, hi)` — no `rand` dep.
fn seeded_f32(seed: u64, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

/// Gate C on CUDA (ADR 0007): the f32 ternary-matmul backward kernels match the
/// `tritium-train` CPU `vjp` oracle within the IMMA `1e-4` bar, across square and
/// tail shapes. Self-skips when no GPU is present.
#[test]
fn train_backward_matches_cpu_vjp() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping train backward parity: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::relative(1e-4);
    // square + tail shapes (non-multiples of the 256-thread block). The (2,300,3)
    // case pushes N past 256 so grad_s's own grid spans >1 block (blockIdx.x>0).
    let shapes = [
        (3, 4, 5),
        (1, 1, 7),
        (2, 3, 4),
        (8, 16, 32),
        (16, 8, 33),
        (5, 7, 1),
        (2, 300, 3),
    ];
    for (m, n, k) in shapes {
        let act = seeded_f32(1, m * k, -2.0, 2.0);
        // Real-valued (fractional) weights exercise the general contraction the
        // autograd surrogate path uses; ternary is the special case it subsumes.
        let w = seeded_f32(2, n * k, -1.0, 1.0);
        let s = seeded_f32(3, n, 0.1, 2.0);
        let gy = seeded_f32(4, m * n, -1.5, 1.5);

        // CPU oracle: vjp -> [gA, gW, gs].
        let cpu = tritium_train::ops::matmul::vjp(&act, &w, &s, m, n, k, &gy);
        let shape = GemmShape::new(m, n, k);

        let mut ga = vec![0.0f32; m * k];
        cuda.grad_a(&gy, &w, &s, shape, &mut ga).expect("grad_a");
        let mut gw = vec![0.0f32; n * k];
        cuda.grad_w(&gy, &act, &s, shape, &mut gw).expect("grad_w");
        let mut gs = vec![0.0f32; n];
        cuda.grad_s(&gy, &act, &w, shape, &mut gs).expect("grad_s");

        for (i, (&g, &c)) in ga.iter().zip(&cpu[0]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_a[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
        for (i, (&g, &c)) in gw.iter().zip(&cpu[1]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_w[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
        for (i, (&g, &c)) in gs.iter().zip(&cpu[2]).enumerate() {
            assert!(
                tol.accepts(g, c),
                "grad_s[{i}] {m}x{n}x{k}: gpu {g} vs cpu {c}"
            );
        }
    }
}

/// ADR 0027 Track D: the compact training-specific SALT planes must preserve
/// Track A's per-row greedy quantizer while eliminating the dense quantized
/// weight. Exercise every supported plane count, TQ2 tails, and the K>8192
/// fallback-sized regime for both forward and activation-gradient contractions.
#[test]
fn packed_training_salt_matches_dense_resident_oracle() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping packed training SALT parity: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::relative(1e-4);
    // M/N are deliberately not tile multiples. K=7 exercises the scalar
    // fallback; larger K values dispatch the tiled kernels, including the
    // 8193-column regime and its final one-element reduction tail.
    let (m, n) = (5usize, 35usize);

    for k in [7usize, 257, 576, 8193] {
        let mut master = seeded_f32(0x5100 + k as u64, n * k, -1.25, 1.25);
        // An exact-zero row gates the zero-scale/code path. The other rows and
        // all tail shapes retain mixed signs and non-integral residuals.
        master[..k].fill(0.0);
        let act = seeded_f32(0xA000 + k as u64, m * k, -0.75, 0.75);
        let gy = seeded_f32(0xB000 + k as u64, m * n, -0.5, 0.5);

        let d_master = cuda.dev_upload(&master).expect("upload master");
        let mut d_residual = cuda.dev_alloc_zeros(n * k).expect("residual scratch");
        let d_act = cuda.dev_upload(&act).expect("upload act");
        let d_gy = cuda.dev_upload(&gy).expect("upload gy");

        for planes in 1..=3 {
            let packed = cuda
                .pack_training_salt(&d_master, &mut d_residual, n, k, planes)
                .expect("pack resident SALT");
            let row_bytes = k.div_ceil(tritium_format::QK_K) * (tritium_format::QK_K / 4);
            assert_eq!(packed.packed_bytes(), planes * n * row_bytes);
            assert_eq!(
                packed.scale_bytes(),
                planes * n * core::mem::size_of::<f32>()
            );
            assert_eq!(
                packed.resident_bytes(),
                packed.packed_bytes() + packed.scale_bytes()
            );

            let dense = tritium_train::ops::ste::salt_quantize_forward(&master, n, k, planes);
            let want_y = tritium_train::ops::dense::forward(&act, &dense, m, n, k);
            let want_ga = tritium_train::ops::dense::vjp(&act, &dense, m, n, k, &gy)[0].clone();

            let mut d_y = cuda.dev_alloc_zeros(m * n).expect("alloc y");
            cuda.training_salt_forward(&d_act, &packed, m, &mut d_y)
                .expect("packed SALT forward");
            let mut got_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_y, &mut got_y).expect("download y");
            let mut d_scalar_y = cuda.dev_alloc_zeros(m * n).expect("alloc scalar y");
            cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
                .expect("scalar packed SALT forward");
            let mut scalar_y = vec![0.0f32; m * n];
            cuda.dev_download(&d_scalar_y, &mut scalar_y)
                .expect("download scalar y");

            let mut d_ga = cuda.dev_alloc_zeros(m * k).expect("alloc ga");
            cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_ga)
                .expect("packed SALT grad_a");
            let mut got_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_ga, &mut got_ga).expect("download ga");
            let mut d_scalar_ga = cuda.dev_alloc_zeros(m * k).expect("alloc scalar ga");
            cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
                .expect("scalar packed SALT grad_a");
            let mut scalar_ga = vec![0.0f32; m * k];
            cuda.dev_download(&d_scalar_ga, &mut scalar_ga)
                .expect("download scalar ga");

            assert_eq!(
                CudaBackend::training_salt_forward_tiled_supported(m, n, k),
                k >= 256
            );
            assert_eq!(
                CudaBackend::training_salt_grad_a_tiled_supported(m, n, k),
                k >= 128
            );

            for (i, (&got, &want)) in got_y.iter().zip(&want_y).enumerate() {
                assert!(
                    tol.accepts(got, want),
                    "forward[{i}] T={planes} {m}x{n}x{k}: packed {got} vs dense {want}"
                );
                assert_eq!(
                    got.to_bits(),
                    scalar_y[i].to_bits(),
                    "forward[{i}] T={planes} {m}x{n}x{k}: tiled {got} vs scalar {}",
                    scalar_y[i],
                );
            }
            for (i, (&got, &want)) in got_ga.iter().zip(&want_ga).enumerate() {
                assert!(
                    tol.accepts(got, want),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: packed {got} vs dense {want}"
                );
                assert_eq!(
                    got.to_bits(),
                    scalar_ga[i].to_bits(),
                    "grad_a[{i}] T={planes} {m}x{n}x{k}: tiled {got} vs scalar {}",
                    scalar_ga[i],
                );
            }
        }
    }

    let d_master = cuda.dev_upload(&[1.0f32; 8]).unwrap();
    let mut d_residual = cuda.dev_alloc_zeros(8).unwrap();
    assert!(matches!(
        cuda.pack_training_salt(&d_master, &mut d_residual, 2, 4, 0),
        Err(BackendError::InvalidInput(_))
    ));
    let packed = cuda
        .pack_training_salt(&d_master, &mut d_residual, 2, 4, 1)
        .unwrap();
    let short_act = cuda.dev_upload(&[1.0f32; 3]).unwrap();
    let mut out = cuda.dev_alloc_zeros(2).unwrap();
    assert!(matches!(
        cuda.training_salt_forward(&short_act, &packed, 1, &mut out),
        Err(BackendError::ShapeMismatch { .. })
    ));
    // Zero output rows are a no-launch success and leave caller storage intact.
    let mut sentinel = cuda.dev_upload(&[17.0f32]).unwrap();
    cuda.training_salt_forward(&short_act, &packed, 0, &mut sentinel)
        .unwrap();
    cuda.training_salt_grad_a(&out, &packed, 0, &mut sentinel)
        .unwrap();
    let mut got_sentinel = [0.0f32];
    cuda.dev_download(&sentinel, &mut got_sentinel).unwrap();
    assert_eq!(got_sentinel, [17.0]);
}

/// Manual Track D microbenchmark. It times requantize plus forward, activation
/// gradient, and their combined path with fixed resident allocations, then
/// prints latency and weight bytes. Hardware-sensitive, so correctness is gated
/// above while this remains opt-in evidence (`--ignored --nocapture`).
#[test]
#[ignore = "4090 Track D performance probe"]
fn bench_packed_training_salt_vs_dense_materialization() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping packed training SALT bench: no device ({e})");
            return;
        }
    };
    let (m, n, k, planes) = (32usize, 576usize, 576usize, 3usize);
    let iters = 100u32;
    let master = seeded_f32(0x5A17, n * k, -1.25, 1.25);
    let act = seeded_f32(0xAC71, m * k, -0.75, 0.75);
    let gy = seeded_f32(0x6A71, m * n, -0.5, 0.5);
    let d_master = cuda.dev_upload(&master).unwrap();
    let d_act = cuda.dev_upload(&act).unwrap();
    let d_gy = cuda.dev_upload(&gy).unwrap();
    let d_ones = cuda.dev_upload(&vec![1.0f32; n]).unwrap();
    let mut d_residual = cuda.dev_alloc_zeros(n * k).unwrap();
    let mut d_dense = cuda.dev_alloc_zeros(n * k).unwrap();
    let mut d_dense_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_dense_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut packed = cuda
        .pack_training_salt(&d_master, &mut d_residual, n, k, planes)
        .unwrap();
    let mut d_packed_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_packed_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let mut d_scalar_y = cuda.dev_alloc_zeros(m * n).unwrap();
    let mut d_scalar_ga = cuda.dev_alloc_zeros(m * k).unwrap();
    let shape = GemmShape::new(m, n, k);

    for _ in 0..10 {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_us = dense_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward_scalar(&d_act, &packed, m, &mut d_scalar_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_us = scalar_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_us = packed_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a_scalar(&d_gy, &packed, m, &mut d_scalar_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let scalar_grad_us = scalar_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let tiled_grad_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let tiled_grad_us = tiled_grad_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
        cuda.grad_a_dev(&d_gy, &d_dense, &d_ones, shape, &mut d_dense_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.salt_quantize_forward_dev(&d_master, &mut d_residual, &mut d_dense, n, k, planes)
            .unwrap();
        cuda.matmul_forward_dev(&d_act, &d_dense, &d_ones, shape, &mut d_dense_y)
            .unwrap();
        cuda.grad_a_dev(&d_gy, &d_dense, &d_ones, shape, &mut d_dense_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let dense_full_us = dense_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    for _ in 0..10 {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_full_start = std::time::Instant::now();
    for _ in 0..iters {
        cuda.repack_training_salt(&d_master, &mut d_residual, &mut packed)
            .unwrap();
        cuda.training_salt_forward(&d_act, &packed, m, &mut d_packed_y)
            .unwrap();
        cuda.training_salt_grad_a(&d_gy, &packed, m, &mut d_packed_ga)
            .unwrap();
    }
    cuda.dev_synchronize().unwrap();
    let packed_full_us = packed_full_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

    println!(
        "Track D resident SALT {m}x{n}x{k} T={planes} repack+forward: \
         dense={dense_us:.1}us, packed-scalar={scalar_us:.1}us, packed-tiled={packed_us:.1}us; \
         tiled speedup={:.2}x dense / {:.2}x scalar. grad_a: scalar={scalar_grad_us:.1}us, \
         tiled={tiled_grad_us:.1}us ({:.2}x). full repack+forward+grad_a: \
         dense={dense_full_us:.1}us, packed-tiled={packed_full_us:.1}us ({:.2}x); \
         dense weight={} B, packed={} B ({:.1}%)",
        dense_us / packed_us,
        scalar_us / packed_us,
        scalar_grad_us / tiled_grad_us,
        dense_full_us / packed_full_us,
        n * k * core::mem::size_of::<f32>(),
        packed.resident_bytes(),
        packed.resident_bytes() as f64 / (n * k * core::mem::size_of::<f32>()) as f64 * 100.0,
    );
}

#[test]
fn cuda_matches_reference_within_tolerance() {
    // Skip cleanly when no GPU is present (cpu-only dev box / wrong CI lane).
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cuda conformance: no device ({e})");
            return;
        }
    };

    let tq2 = tq2_vectors();
    let report = run_conformance(&backend, &tq2, Tolerance::default());
    assert!(
        report.is_ok(),
        "{} cuda conformance cases failed: {:?}",
        report.failed.len(),
        report.failed
    );
}

// v0.4.0 P1: the SALT multi-plane GPU GEMM must match `dequant_salt_row` → fp32
// reference matmul within 1e-4, across T∈{1,2,3}, M∈{1,2}, with each plane's
// per-block f16 scales including a zero-variance (scale 0) block and an
// outlier-heavy (large scale) block.
#[test]
fn salt_mpgemm_matches_dequant_reference() {
    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{
        SaltRow, TQ2_0_BLOCK_BYTES, dequant_salt_row, num_blocks, pack_tq2_0_row,
    };

    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping salt mpgemm: no device ({e})");
            return;
        }
    };

    let k = 512usize; // 2 blocks
    let n = 6usize;
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;

    let mut s: u64 = 0x5A17_C0DE;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };

    for m in [1usize, 2] {
        for t in [1usize, 2, 3] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 0.5)
                .collect();

            // planes[p][ni] = packed TQ2_0 bytes for row ni, plane p.
            let mut planes: Vec<Vec<Vec<u8>>> = Vec::with_capacity(t);
            for p in 0..t {
                let mut prows = Vec::with_capacity(n);
                for _ni in 0..n {
                    let trits: Vec<Trit> = (0..k)
                        .map(|_| Trit::from_i8(((next() >> 40) % 3) as i8 - 1).unwrap())
                        .collect();
                    let scales: Vec<f16> = (0..nb)
                        .map(|_| {
                            let pick = (next() >> 40) % 8;
                            let v = match pick {
                                0 => 0.0,  // zero-variance block
                                1 => 12.5, // outlier-heavy block
                                other => 0.05 + other as f32 * 0.3,
                            };
                            f16::from_f32(v / (p as f32 + 1.0))
                        })
                        .collect();
                    let mut bytes = vec![0u8; row_bytes];
                    pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                    prows.push(bytes);
                }
                planes.push(prows);
            }

            // Plane-major concatenation: plane p, then row ni.
            let mut weights = Vec::with_capacity(t * n * row_bytes);
            for prows in &planes {
                for row in prows {
                    weights.extend_from_slice(row);
                }
            }

            // Reference: dequant each row to fp32 weights, then fp64 matmul.
            let mut reference = vec![0f64; m * n];
            for ni in 0..n {
                let row = SaltRow {
                    k,
                    planes: (0..t).map(|p| planes[p][ni].clone()).collect(),
                };
                let w = dequant_salt_row(&row).unwrap();
                for mi in 0..m {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        acc += act[mi * k + kk] as f64 * w[kk] as f64;
                    }
                    reference[mi * n + ni] = acc;
                }
            }

            let gpu = cuda.salt_mpgemm_dense(&act, &weights, m, n, k, t).unwrap();
            for i in 0..m * n {
                let r = reference[i];
                let tol = 1e-4 * r.abs().max(1.0);
                assert!(
                    (gpu[i] as f64 - r).abs() <= tol,
                    "salt mpgemm m={m} t={t} idx={i}: gpu={} ref={r} (tol {tol})",
                    gpu[i],
                );
            }
        }
    }
}

/// v0.4.1: flash-decoding (split-KV) attention must match the direct decode
/// attention (`gqa_attention_decode`) within tolerance — for several `n_split`
/// (chunk counts), including `n_split=1` (single chunk) and a split that leaves a
/// ragged final chunk. The online-softmax merge reorders sums, so this is a
/// tolerance gate (1e-4), not bit-exact.
#[test]
fn attn_split_kv_matches_direct_attention() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping split-kv attn: no device ({e})");
            return;
        }
    };
    let (n_head, n_head_kv, head_dim, ctx) = (8usize, 2usize, 128usize, 200usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut s: u64 = 0x5F11_7A11_u64; // seed
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
    };
    let q: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();

    let mut reference = vec![0.0f32; n_head * head_dim];
    cuda.gqa_attention_decode(
        &q,
        &k,
        &v,
        &mut reference,
        ctx,
        n_head,
        n_head_kv,
        head_dim,
        scale,
        ctx,
    )
    .expect("reference attention");

    for n_split in [1usize, 4, 7, 16] {
        let chunk = ctx.div_ceil(n_split);
        let got = cuda
            .attn_split_dense(
                &q, &k, &v, n_head, n_head_kv, head_dim, scale, ctx, n_split, chunk,
            )
            .expect("split attention");
        for i in 0..n_head * head_dim {
            let r = reference[i];
            let tol = 1e-4 * r.abs().max(1.0);
            assert!(
                (got[i] as f64 - r as f64).abs() <= tol as f64,
                "split-kv n_split={n_split} idx={i}: got={} ref={r} (tol {tol})",
                got[i],
            );
        }
    }
}

/// v0.4.0: the **resident** SALT path — upload a SALT tensor's rows once via
/// [`CudaBackend::upload_salt`], then [`CudaBackend::salt_forward`] — must match
/// the host `dequant_salt_row → fp32 matmul` reference, for T=1/2/3 (incl. ragged
/// plane counts) and survive reuse (two forwards on the same resident buffer).
/// This gates the resident decode wiring, distinct from `salt_mpgemm_dense` which
/// re-uploads per call.
#[test]
fn salt_resident_forward_matches_dequant() {
    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{SaltRow, dequant_salt_row, pack_tq2_0_row};

    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping salt resident: no device ({e})");
            return;
        }
    };

    let k = 512usize;
    let n = 6usize;
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let mut s: u64 = 0x5A17_F00D;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };

    // Build n rows; row ni gets `t_of(ni)` planes (ragged: not all rows equal T).
    for max_t in [1usize, 2, 3] {
        let rows: Vec<SaltRow> = (0..n)
            .map(|ni| {
                let t_row = 1 + (ni % max_t); // 1..=max_t, ragged across rows
                let planes = (0..t_row)
                    .map(|p| {
                        let trits: Vec<Trit> = (0..k)
                            .map(|_| Trit::from_i8(((next() >> 40) % 3) as i8 - 1).unwrap())
                            .collect();
                        let scales: Vec<f16> = (0..nb)
                            .map(|_| {
                                f16::from_f32(
                                    (0.05 + ((next() >> 40) % 8) as f32 * 0.3) / (p as f32 + 1.0),
                                )
                            })
                            .collect();
                        let mut bytes = vec![0u8; row_bytes];
                        pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                        bytes
                    })
                    .collect();
                SaltRow { k, planes }
            })
            .collect();

        // Host reference: dequant each row, fp64 matmul.
        let m = 2usize;
        let act: Vec<f32> = (0..m * k)
            .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 0.5)
            .collect();
        let mut reference = vec![0f64; m * n];
        for (ni, row) in rows.iter().enumerate() {
            let w = dequant_salt_row(row).unwrap();
            for mi in 0..m {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += act[mi * k + kk] as f64 * w[kk] as f64;
                }
                reference[mi * n + ni] = acc;
            }
        }

        let lin = cuda.upload_salt(&rows, n, k).expect("upload_salt");
        // Two forwards on the same resident buffer must agree (reuse).
        let gpu = cuda.salt_forward(&lin, &act, m).expect("salt_forward");
        let gpu2 = cuda
            .salt_forward(&lin, &act, m)
            .expect("salt_forward reuse");
        assert_eq!(
            gpu, gpu2,
            "resident reuse must be deterministic (max_t={max_t})"
        );

        for i in 0..m * n {
            let r = reference[i];
            let tol = 1e-4 * r.abs().max(1.0);
            assert!(
                (gpu[i] as f64 - r).abs() <= tol,
                "salt resident max_t={max_t} idx={i}: gpu={} ref={r} (tol {tol})",
                gpu[i],
            );
        }
    }
}

/// ADR 0002 U2: CPU↔CUDA parity. The *same* committed TQ2_0 vectors run through
/// both [`CpuBackend`] and [`CudaBackend`]; every output element must agree
/// within `1e-4` relative. This is the load-bearing cross-backend gate — it
/// catches a backend that is internally self-consistent (passes conformance)
/// but disagrees with the other backend on shared inputs.
#[test]
fn cuda_matches_cpu_within_tolerance() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cpu<->cuda parity: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // Run both backends over the identical TQ2_0 vector set.
    let cpu_report = run_conformance(&cpu, &tq2_vectors(), tol);
    assert!(
        cpu_report.is_ok(),
        "cpu backend failed its own conformance, parity is moot: {:?}",
        cpu_report.failed
    );

    // Replay each vector through both backends and compare outputs directly,
    // rather than only against the shared reference, so any CPU/CUDA divergence
    // surfaces even within the reference tolerance band.
    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("vector weight in {-1,0,1}"))
            .collect();
        let packed = pack_tq2_0(&trits, shape);

        let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);
        let cuda_out = run_backend(&cuda, &packed, &v.activation, &v.scales, shape);

        assert_eq!(
            cpu_out.len(),
            cuda_out.len(),
            "{}: output len mismatch",
            v.id
        );
        for (i, (&c, &g)) in cpu_out.iter().zip(&cuda_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "{}: cpu/cuda disagree at [{i}]: cpu={c} cuda={g}",
                v.id
            );
        }
    }
}

/// Pack an `[N, K]` trit matrix to TQ2_0 rows, block scale fixed to `1.0` (the
/// testkit convention), ready for `upload_weights`.
fn pack_tq2_0(trits: &[tritium_core::Trit], shape: GemmShape) -> Vec<u8> {
    use tritium_format::pack_tq2_0_row;
    let GemmShape { n, k, .. } = shape;
    let nb = num_blocks(k);
    let unit = vec![half::f16::ONE; nb];
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let mut packed = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let row = &trits[ni * k..ni * k + k];
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
    }
    packed
}

/// Upload weights + run one TQ2_0 mpGEMM through any backend, returning `[M, N]`.
fn run_backend<B: TernaryBackend>(
    backend: &B,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    shape: GemmShape,
) -> Vec<f32> {
    let buf = backend
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    backend
        .mpgemm(tritium_spec::MpGemm {
            act,
            weights: buf.as_ref(),
            scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut out,
        })
        .expect("mpgemm");
    out
}

/// Upload weights + run one TQ2_0 mpGEMM through a *forced* add kernel, so a
/// test can gate each path independently of the shape-based auto-selection.
fn run_kernel(
    cuda: &CudaBackend,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    shape: GemmShape,
    kernel: AddKernel,
) -> Vec<f32> {
    let buf = cuda
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    cuda.mpgemm_kernel(
        act,
        buf.as_ref(),
        scales,
        shape,
        TernaryFormat::Tq2_0,
        &mut out,
        kernel,
    )
    .expect("mpgemm_kernel");
    out
}

/// Upload weights + run the sparse-aware tiled kernel with a pre-computed
/// zero-block bitmap. Returns the output `[M, N]`.
fn run_kernel_sparse(
    cuda: &CudaBackend,
    packed: &[u8],
    act: &[f32],
    scales: &[f32],
    bitmap: &[u32],
    words_per_row: usize,
    shape: GemmShape,
) -> Vec<f32> {
    let buf = cuda
        .upload_weights(packed, shape, TernaryFormat::Tq2_0)
        .expect("upload weights");
    let mut out = vec![0.0f32; shape.m * shape.n];
    cuda.mpgemm_kernel_with_bitmap(
        act,
        buf.as_ref(),
        scales,
        bitmap,
        words_per_row,
        shape,
        TernaryFormat::Tq2_0,
        &mut out,
    )
    .expect("mpgemm_kernel_with_bitmap");
    out
}

/// Both add kernels must match the CPU reference (within tolerance) on the full
/// committed TQ2_0 conformance set. This gates the new tiled kernel directly,
/// and re-gates the simple kernel, regardless of which one auto-selection picks.
#[test]
fn both_add_kernels_match_reference() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping both-kernel gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
            .collect();
        let packed = pack_tq2_0(&trits, shape);
        let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);

        let simple = run_kernel(
            &cuda,
            &packed,
            &v.activation,
            &v.scales,
            shape,
            AddKernel::Simple,
        );
        for (i, (&g, &c)) in simple.iter().zip(&cpu_out).enumerate() {
            assert!(tol.accepts(g, c), "{}: simple vs cpu [{i}] {g} {c}", v.id);
        }

        // The tiled kernel only accepts K within its shared-memory budget.
        if v.k <= TILED_K_MAX {
            let tiled = run_kernel(
                &cuda,
                &packed,
                &v.activation,
                &v.scales,
                shape,
                AddKernel::Tiled,
            );
            for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                assert!(tol.accepts(g, c), "{}: tiled vs cpu [{i}] {g} {c}", v.id);
            }
        }
    }
}

/// The tiled kernel must be correct on boundary shapes: tail `K` (not a 256
/// multiple, so a partial final TQ2_0 block), partial warps (`N` not a multiple
/// of `WARPS_PER_BLOCK`), partial grids (`M`/`N` of 1), and `K` at the cap.
#[test]
fn tiled_handles_tail_shapes() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tiled tail-shape gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // (M, N, K) — tail K, partial warps/blocks, single rows/cols, K at the cap.
    let shapes = [
        (1usize, 1usize, 1usize),
        (1, 7, 300),
        (5, 130, 257),
        (64, 3, 2560),
        (3, 33, 6912),
        (1, 1, TILED_K_MAX),
    ];

    for (m, n, k) in shapes {
        assert!(k <= TILED_K_MAX, "test shape K exceeds the tiled cap");
        let shape = GemmShape::new(m, n, k);

        // Deterministic ternary weights, activations, and per-channel scales.
        let trits: Vec<_> = (0..n * k)
            .map(|i| tritium_core::Trit::from_i8(((i % 3) as i8) - 1).unwrap())
            .collect();
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.25).collect();

        let packed = pack_tq2_0(&trits, shape);
        let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);
        let tiled = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

        assert_eq!(tiled.len(), cpu_out.len(), "shape {shape:?}: len");
        for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "shape {shape:?}: tiled vs cpu [{i}] tiled={g} cpu={c}"
            );
        }
    }
}

/// A2 flipped this contract: TQ1_0 UPLOADS are now first-class (the tq1
/// decode kernels read them natively) — a correct-length upload succeeds and
/// a wrong-length one is a typed InvalidInput; the HOST mpgemm path still
/// rejects the format (the resident decoder is TQ1's only consumer in v1).
#[test]
fn tq1_0_upload_accepted_host_mpgemm_rejected() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(_) => return, // no device: nothing to assert about format handling
    };
    let shape = GemmShape { m: 1, n: 1, k: 256 };
    // Wrong length (66 = a TQ2 block) -> typed error, not a panic.
    match backend.upload_weights(&[0u8; 66], shape, TernaryFormat::Tq1_0) {
        Err(BackendError::InvalidInput(_)) => {}
        other => panic!("expected InvalidInput, got {:?}", other.map(|_| "ok")),
    }
    // Correct length (54 = one TQ1 block) uploads.
    let buf = backend
        .upload_weights(&[0u8; 54], shape, TernaryFormat::Tq1_0)
        .expect("tq1 upload");
    // Host mpgemm rejects the format loudly.
    let act = vec![0.0f32; 256];
    let scales = vec![1.0f32];
    let mut out = vec![0.0f32; 1];
    match backend.mpgemm(tritium_spec::MpGemm {
        act: &act,
        weights: &*buf,
        scales: &scales,
        shape,
        format: TernaryFormat::Tq1_0,
        out: &mut out,
    }) {
        Err(BackendError::UnsupportedFormat(TernaryFormat::Tq1_0)) => {}
        other => panic!("expected UnsupportedFormat(Tq1_0), got {other:?}"),
    }
}

// ---- IMMA int8 tensor-core path (v0.30 WF-A part 2) ------------------------
//
// Tolerance: the conformance default (`relative = 1e-4`, ADR 0002). The IMMA
// kernel contracts in **int32**, which is *exact* for int8×ternary (no overflow
// for any BitNet K — see `kernels/tq2_0_imma.cu`), so the only float rounding is
// the single per-output `act_scale·weight_scale·acc`. The 1e-4 band is therefore
// the *reference's* own f32-accumulate rounding, not a defect of this kernel —
// no widened reduction bar is needed (cf. the tiled add-only kernel, which sums
// in double to stay inside the band; the IMMA integer accumulate is exact).

/// Build an I2_S tensor payload (`N·K/4` quant bytes + one trailing `f32` scale)
/// from an `[N, K]` row-major trit matrix, inverting the 32-byte block striping
/// (`code = trit + 1`, element `pos` of a 128-block at byte `pos%32`, shift
/// `6 - 2*(pos/32)`). `n*k` must be a multiple of 128 (the conformance shapes
/// all are: K ∈ {256, 512}).
fn build_i2s_payload(trits: &[i8], scale: f32) -> Vec<u8> {
    let n_elements = trits.len();
    assert!(
        n_elements.is_multiple_of(128),
        "i2s payload needs 128-multiple elems"
    );
    let mut quants = vec![0u8; n_elements / 4];
    for (global, &t) in trits.iter().enumerate() {
        let block = global / 128;
        let pos = global % 128;
        let group = pos / 32;
        let gp = pos % 32;
        let code = (t + 1) as u8; // {-1,0,1} -> {0,1,2}
        quants[block * 32 + gp] |= code << (6 - 2 * group);
    }
    let mut payload = quants;
    payload.extend_from_slice(&scale.to_le_bytes());
    payload
}

/// Pack an `[N, K]` trit matrix into the IMMA `I2sInt8` layout by routing it
/// through the *real* converter (`build_i2s_payload` → `convert_i2s_to_int8`),
/// so the test exercises exactly the bytes the kernel will see in production.
/// Returns the packed bytes (block scale folded into the per-tensor `scale`,
/// which the test keeps separate as the per-channel scale, so pass `scale = 1`).
fn pack_i2s_int8(trits: &[i8], shape: GemmShape) -> Vec<u8> {
    let GemmShape { n, k, .. } = shape;
    let payload = build_i2s_payload(trits, 1.0);
    let w = tritium_format::convert_i2s_to_int8(&payload, GemmShape { m: 0, n, k })
        .expect("convert i2s -> int8");
    w.bytes
}

/// IMMA == reference within tolerance over the conformance set. The vectors'
/// weights are converted to `I2sInt8`, uploaded, and run through the fused
/// `mpgemm_with_act_quant` (which routes I2sInt8 → on-device quant + IMMA). The
/// reference is `mpgemm_with_act_quant`'s contract on the *same f32 activations*:
/// `out[m,n] = act_scale[m]·weight_scale[n]·Σ q[m,k]·w[n,k]`, which the testkit
/// CPU path computes via the spec default — so this gates IMMA == host-A8 == ref
/// in one shot.
#[test]
fn imma_matches_reference_within_tolerance() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping imma conformance: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);

        // Reference: the host-A8 default path on the CPU backend over the SAME
        // f32 activations + per-channel weight scales.
        let cpu_buf = {
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);
            cpu.upload_weights(&packed, shape, TernaryFormat::Tq2_0)
                .expect("cpu upload")
        };
        let mut ref_out = vec![0.0f32; shape.m * shape.n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut ref_out,
        })
        .expect("cpu host-A8 reference");

        // IMMA: upload the I2sInt8 weights, run the fused override (on-device
        // quant + tensor-core contraction).
        let imma_bytes = pack_i2s_int8(&v.weights, shape);
        let imma_buf = cuda
            .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut imma_out = vec![0.0f32; shape.m * shape.n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: imma_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut imma_out,
        })
        .expect("imma fused mpgemm");

        assert_eq!(imma_out.len(), ref_out.len(), "{}: len", v.id);
        for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "{}: imma vs host-A8 ref [{i}] imma={g} ref={c}",
                v.id
            );
        }
    }
}

/// The CUDA fused override (IMMA) == the spec host-A8 default == the v0.20
/// caller-side quant, all within tolerance — the "fused == host-A8" gate of ADR
/// 0005. Three independently-derived results over the same inputs:
///   1. `cuda.mpgemm_with_act_quant` on an I2sInt8 buffer → on-device quant + IMMA.
///   2. The spec *default* `mpgemm_with_act_quant` (host quant → `mpgemm`) run on
///      the CPU backend (a TQ2_0 buffer).
///   3. The v0.20 caller-side quant: quantize on the host, then call plain
///      `mpgemm` and fold the per-token scale by hand.
#[test]
fn imma_fused_equals_host_a8_and_caller_quant() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping fused parity: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    for v in tq2_vectors() {
        let shape = GemmShape::new(v.m, v.n, v.k);
        let GemmShape { m, n, k } = shape;
        let trits: Vec<_> = v
            .weights
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
            .collect();
        let tq2 = pack_tq2_0(&trits, shape);

        // (1) CUDA fused override on I2sInt8.
        let imma_bytes = pack_i2s_int8(&v.weights, shape);
        let imma_buf = cuda
            .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut fused = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: imma_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut fused,
        })
        .expect("cuda fused");

        // (2) Spec host-A8 default on the CPU backend (TQ2_0).
        let cpu_buf = cpu
            .upload_weights(&tq2, shape, TernaryFormat::Tq2_0)
            .expect("cpu upload");
        let mut host_a8 = vec![0.0f32; m * n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &v.activation,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut host_a8,
        })
        .expect("cpu host-A8");

        // (3) v0.20 caller-side quant: host quant → plain `mpgemm` → fold.
        let mut q = vec![0.0f32; m * k];
        let mut act_scale = vec![0.0f32; m];
        quantize_act_int8_host(&v.activation, m, k, &mut q, &mut act_scale);
        let mut caller = vec![0.0f32; m * n];
        cpu.mpgemm(tritium_spec::MpGemm {
            act: &q,
            weights: cpu_buf.as_ref(),
            scales: &v.scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut caller,
        })
        .expect("cpu plain mpgemm");
        for (row, &s) in caller.chunks_exact_mut(n).zip(act_scale.iter()) {
            for x in row {
                *x *= s;
            }
        }

        for i in 0..m * n {
            assert!(
                tol.accepts(fused[i], host_a8[i]),
                "{}: fused vs host-A8 [{i}] {} {}",
                v.id,
                fused[i],
                host_a8[i]
            );
            assert!(
                tol.accepts(fused[i], caller[i]),
                "{}: fused vs caller-quant [{i}] {} {}",
                v.id,
                fused[i],
                caller[i]
            );
        }
    }
}

/// IMMA tail/boundary shapes: M not a multiple of 16, N not a multiple of 8, and
/// single rows/cols — the padding in the I2sInt8 tiles and the kernel's global
/// bounds checks must keep every covered output correct. K stays a 256-multiple
/// (the I2_S converter needs a 128-multiple element count); the M/N tails are the
/// interesting axes for the 16×8 tile.
#[test]
fn imma_handles_tail_shapes() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping imma tail shapes: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // (M, N, K): single row/col, partial 16-row tile, partial 8-col tile.
    let shapes = [
        (1usize, 1usize, 256usize),
        (1, 8, 256),
        (3, 5, 256),
        (16, 8, 512),
        (17, 9, 256),
        (33, 13, 512),
    ];
    for (m, n, k) in shapes {
        let shape = GemmShape::new(m, n, k);
        // Deterministic ternary weights, activations, per-channel scales.
        let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

        // Reference: host-A8 default on the CPU backend.
        let trits: Vec<_> = raw
            .iter()
            .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
            .collect();
        let cpu_buf = cpu
            .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
            .expect("cpu upload");
        let mut ref_out = vec![0.0f32; m * n];
        cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &act,
            weights: cpu_buf.as_ref(),
            scales: &scales,
            shape,
            format: TernaryFormat::Tq2_0,
            out: &mut ref_out,
        })
        .expect("cpu host-A8");

        let imma_buf = cuda
            .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
            .expect("imma upload");
        let mut imma_out = vec![0.0f32; m * n];
        cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
            act: &act,
            weights: imma_buf.as_ref(),
            scales: &scales,
            shape,
            format: TernaryFormat::I2sInt8,
            out: &mut imma_out,
        })
        .expect("imma fused");

        for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
            assert!(
                tol.accepts(g, c),
                "shape {shape:?}: imma vs ref [{i}] imma={g} ref={c}"
            );
        }
    }
}

/// ADR 0026 Track P step 2: the load-time IMMA shadow (TQ2 packed rows →
/// unpack → `pack_i2s_int8_tiles`) must produce byte-identical output to the
/// production I2_S converter (`convert_i2s_to_int8`) for the same trits —
/// the kernel sees exactly one weight layout regardless of which path packed
/// it. Host-only, no GPU.
#[test]
fn imma_shadow_matches_i2s_converter_bytes() {
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};
    for &(n, k) in &[(8usize, 256usize), (13, 512), (40, 2560), (5, 6912)] {
        let trits = mixed_trits(n, k, 0x77 ^ (n as u64) ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb = nb * TQ2_0_BLOCK_BYTES;
        let mut rows = vec![0u8; n * rb];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut rows[ni * rb..(ni + 1) * rb],
            )
            .expect("pack tq2");
        }
        let shadow = imma_shadow_bytes(&rows, n, k, rb).expect("shadow");
        let trits_i8: Vec<i8> = trits.iter().map(|t| t.get()).collect();
        let converter = pack_i2s_int8(&trits_i8, GemmShape { m: 0, n, k });
        assert_eq!(
            shadow, converter,
            "n{n} k{k}: shadow bytes != convert_i2s_to_int8 bytes"
        );
    }
}

/// ADR 0026 Track P bit-identity gate: the IMMA tensor-core kernel must be
/// **bit-identical** to the dp4a `tiled_i8_scaled` kernel on the SAME int8
/// activations, act scales and per-channel weight scales. Both contract in
/// exact i32 (order-free) and both fold `(float)acc * wscale[n] * act_scale[m]`
/// in the same association (a pure multiply chain — no FMA contraction), so
/// every output bit matches. This is what lets the prefill dispatch swap
/// kernels by M with ZERO numerics re-gating (C1 chunking, G1 first-token).
/// Shapes cover 16/8/32-tile boundaries and tails at the real K values.
#[test]
fn imma_matches_dp4a_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping imma-vs-dp4a gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let m_add = ctx
        .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
        .expect("load add module");
    let m_imma = ctx
        .load_module(Ptx::from_src(TQ2_0_IMMA_PTX))
        .expect("load imma module");
    let f_dp4a = m_add
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .expect("dp4a fn");
    let f_imma = m_imma.load_function("tq2_0_imma_mpgemm").expect("imma fn");

    // (m, n, k): tile-aligned and tail shapes at prefill-realistic K
    // (k % 256 == 0 — the TQ2 packer's block size; covers K=2560/6912-class
    // contractions via 2560 and a 6912-divisor tail mix).
    for &(m, n, k) in &[
        (16usize, 8usize, 256usize),
        (33, 13, 512),
        (128, 40, 2560),
        (7, 9, 1024),
    ] {
        let trits = mixed_trits(n, k, 0x51 ^ (m as u64) ^ (k as u64));
        // dp4a weights: TQ2_0 rows with unit block scales (both kernels ignore
        // block scales; the per-channel `scales` array is the shared truth).
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb = nb * TQ2_0_BLOCK_BYTES;
        let mut packed_tq2 = vec![0u8; n * rb];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut packed_tq2[ni * rb..(ni + 1) * rb],
            )
            .expect("pack tq2");
        }
        // IMMA weights: the I2sInt8 tile interleave from the SAME trits.
        let trits_i8: Vec<i8> = trits.iter().map(|t| t.get()).collect();
        let packed_imma = pack_i2s_int8_tiles(&trits_i8, n, k);
        let num_ktiles = k.div_ceil(tritium_format::IMMA_K);

        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 255) as i8).collect();
        let scales = seeded_f32(7, n, 0.5, 2.0);
        let act_scale = seeded_f32(13, m, 0.5, 1.5);

        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&packed_tq2).unwrap();
        let d_wi = stream.clone_htod(&packed_imma).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let (m_i, n_i, k_i, rb_i, nkt_i) =
            (m as i32, n as i32, k as i32, rb as i32, num_ktiles as i32);

        // dp4a launch (the production tiled_i8_scaled geometry).
        let dp4a_out = {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
                block_dim: (8 * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(&f_dp4a);
            l.arg(&d_qact)
                .arg(&d_w2)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&rb_i);
            // SAFETY: matches `tq2_0_add_mpgemm_tiled_i8_scaled(qact, w, scales,
            // act_scale, out, m, n, k, row_bytes)`; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };

        // IMMA launch (one warp per 16x8 tile).
        let imma_out = {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(8), (m as u32).div_ceil(16), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = stream.launch_builder(&f_imma);
            l.arg(&d_qact)
                .arg(&d_wi)
                .arg(&d_as)
                .arg(&d_sc)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&nkt_i);
            // SAFETY: matches `tq2_0_imma_mpgemm(qact, weights, act_scale,
            // weight_scale, out, m, n, k, num_ktiles)`; grid covers all tiles.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };

        for (i, (a, b)) in dp4a_out.iter().zip(&imma_out).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "m{m} n{n} k{k} [{i}]: dp4a={a} imma={b} — the epilogue \
                 association drifted (bit-identity contract, ADR 0026)"
            );
        }
    }
}

// ---- WF-B: autotune + nvrtc JIT determinism (ADR 0005) ---------------------
//
// These gate the WF-B contract: a JIT-compiled tile is BIT-IDENTICAL to the AOT
// cubin for the same tile (cold-cache == warm-cache), and any tuned tile matches
// the reference within the IMMA tolerance. Both are guaranteed by construction —
// every tile does the same exact int32 mma accumulate + one f32 scale fold — but
// these tests prove it on-device across tile shapes.

/// Deterministic int8 activations / ternary weights / scales for a WF-B probe.
fn jit_probe_inputs(m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>, Vec<i8>) {
    let qact: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let act_scale: Vec<f32> = (0..m).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
    let wscale: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();
    let trits: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
    (qact, act_scale, wscale, trits)
}

/// Run one IMMA contraction with an explicit `func`/`tile` (host-quantised int8
/// inputs already supplied), returning the `[M, N]` f32 output. Drives
/// `launch_imma_tile` directly so a test can force a specific tile + kernel image
/// (AOT cubin vs a freshly JIT-compiled module).
#[allow(clippy::too_many_arguments)] // a test driver mirroring the kernel's operands
fn run_imma_tile(
    cuda: &CudaBackend,
    func: &CudaFunction,
    tile: TileConfig,
    qact: &[i8],
    packed_weights: &[u8],
    act_scale: &[f32],
    wscale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let num_ktiles = k.div_ceil(IMMA_K);
    let d_qact = cuda.stream.clone_htod(qact).expect("htod qact");
    let d_weights = cuda
        .stream
        .clone_htod(packed_weights)
        .expect("htod weights");
    let d_act_scale = cuda.stream.clone_htod(act_scale).expect("htod act_scale");
    let d_wscale = cuda.stream.clone_htod(wscale).expect("htod wscale");
    let mut d_out = cuda.stream.alloc_zeros::<f32>(m * n).expect("alloc out");
    cuda.launch_imma_tile(
        func,
        tile,
        &d_qact,
        &d_weights,
        &d_act_scale,
        &d_wscale,
        &mut d_out,
        m as i32,
        n as i32,
        k as i32,
        num_ktiles as i32,
    )
    .expect("launch imma tile");
    let mut out = vec![0.0f32; m * n];
    cuda.stream.memcpy_dtoh(&d_out, &mut out).expect("dtoh out");
    cuda.stream.synchronize().expect("sync");
    out
}

/// COLD-CACHE (JIT) == WARM-CACHE (AOT) BIT-IDENTICAL for a fixed tile.
///
/// The AOT-equivalent tile has two realisations: the embedded AOT cubin
/// (`func_imma`, the warm/default path) and a fresh nvrtc JIT compile of the
/// rendered source (the cold path). For a range of shapes their outputs must be
/// **bit-for-bit equal** (`==` on the raw `f32`, not a tolerance) — the load-bearing
/// WF-B determinism gate. If they ever diverge, JIT and AOT are not interchangeable
/// and the autotune cache could change numerics, which ADR 0005 forbids.
#[test]
fn jit_aot_equivalent_is_bit_identical() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping JIT==AOT bit-identity: no device ({e})");
            return;
        }
    };

    // Freshly JIT-compile the AOT-equivalent tile (the cold path). The AOT side
    // is the embedded cubin resolved by `imma_function_for_tile`.
    let tile = TileConfig::AOT_EQUIVALENT;
    let (_jit_mod, jit_func) = cuda
        .imma_jit_function(tile)
        .expect("JIT-compile AOT-equivalent tile");
    let aot_func = cuda
        .imma_function_for_tile(tile)
        .expect("resolve AOT cubin");

    // Tail + clean shapes; K a 32-multiple (one whole k-tile minimum).
    let shapes = [
        (1usize, 1usize, 32usize),
        (3, 5, 64),
        (16, 8, 256),
        (17, 9, 96),
        (33, 13, 512),
        (64, 40, 2560), // a realistic-ish K (a 32-multiple, below the tiled cap)
    ];
    for (m, n, k) in shapes {
        let k = k.max(IMMA_K); // never zero k-tiles
        let k = k.div_ceil(IMMA_K) * IMMA_K; // snap to a whole k-tile
        let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
        let packed = pack_i2s_int8_tiles(&trits, n, k);

        let aot = run_imma_tile(
            &cuda, &aot_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );
        let jit = run_imma_tile(
            &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );

        assert_eq!(aot.len(), jit.len(), "shape ({m},{n},{k}): len");
        for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
            // Bit-identical: compare the raw IEEE-754 bit patterns so even a
            // signed-zero or NaN-payload difference would fail (none expected).
            assert_eq!(
                a.to_bits(),
                j.to_bits(),
                "shape ({m},{n},{k}): JIT vs AOT diverge at [{i}] aot={a} jit={j}"
            );
        }
    }
}

/// A NON-TRIVIAL JIT tile (wider M/N, deeper K, multi-warp) is ALSO bit-identical
/// to the AOT cubin. This proves the determinism guarantee holds across the tile
/// shapes the autotune search actually considers, not just the AOT-equivalent
/// anchor — the int32 accumulate is order-independent, so a 32×16/4-warp tile that
/// splits the work differently still lands on the same bits.
#[test]
fn jit_wide_tile_matches_aot_bit_identical() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping wide-tile JIT==AOT: no device ({e})");
            return;
        }
    };
    let aot_func = cuda
        .imma_function_for_tile(TileConfig::AOT_EQUIVALENT)
        .expect("AOT cubin");

    // A representative spread of the search's candidate tiles.
    let tiles = [
        TileConfig {
            tile_m: 16,
            tile_n: 8,
            tile_k: 128,
            warps: 1,
            stages: 2,
        },
        TileConfig {
            tile_m: 16,
            tile_n: 16,
            tile_k: 64,
            warps: 2,
            stages: 2,
        },
        TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        },
        TileConfig {
            tile_m: 64,
            tile_n: 16,
            tile_k: 32,
            warps: 8,
            stages: 3,
        },
    ];
    let (m, n, k) = (40usize, 24usize, 256usize);
    let (qact, act_scale, wscale, trits) = jit_probe_inputs(m, n, k);
    let packed = pack_i2s_int8_tiles(&trits, n, k);

    let aot = run_imma_tile(
        &cuda,
        &aot_func,
        TileConfig::AOT_EQUIVALENT,
        &qact,
        &packed,
        &act_scale,
        &wscale,
        m,
        n,
        k,
    );

    for tile in tiles {
        assert!(tile.is_valid(), "test tile {tile:?} invalid");
        let (_m, jit_func) = cuda
            .imma_jit_function(tile)
            .unwrap_or_else(|e| panic!("JIT-compile {tile:?}: {e:?}"));
        let jit = run_imma_tile(
            &cuda, &jit_func, tile, &qact, &packed, &act_scale, &wscale, m, n, k,
        );
        for (i, (&a, &j)) in aot.iter().zip(&jit).enumerate() {
            assert_eq!(
                a.to_bits(),
                j.to_bits(),
                "tile {tile:?}: JIT vs AOT diverge at [{i}] aot={a} jit={j}"
            );
        }
    }
}

/// The TUNED config (resolved through the on-disk autotune cache + tile search)
/// matches the reference within the IMMA tolerance. Drives the full public fused
/// path (`mpgemm_with_act_quant`), which now consults the cache via
/// `resolve_imma_tile`, on a prefill-shaped problem — so this exercises the tuner
/// end-to-end (cold cache → search → winner) and gates the winner vs the CPU
/// host-A8 reference. A second call (warm cache) must agree bit-for-bit with the
/// first, since a cached tile is numerically identical to the freshly-tuned one.
#[test]
fn tuned_config_matches_reference_and_is_stable() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping tuned-config gate: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // A prefill-shaped problem so the search has something to chew on. K is a
    // 256-multiple (the I2_S converter the reference path uses needs a
    // 128-multiple); N/M exercise partial tiles.
    let (m, n, k) = (40usize, 24usize, 256usize);
    let shape = GemmShape::new(m, n, k);
    let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
    let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

    // Reference: host-A8 default on the CPU backend (TQ2_0).
    let trits: Vec<_> = raw
        .iter()
        .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
        .collect();
    let cpu_buf = cpu
        .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
        .expect("cpu upload");
    let mut ref_out = vec![0.0f32; m * n];
    cpu.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: cpu_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::Tq2_0,
        out: &mut ref_out,
    })
    .expect("cpu host-A8 reference");

    // Tuned path: upload I2sInt8, run the fused override (which resolves + tunes
    // the tile). Run it twice; the second call hits the in-memory + on-disk cache.
    let imma_buf = cuda
        .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
        .expect("imma upload");
    let mut tuned1 = vec![0.0f32; m * n];
    cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: imma_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::I2sInt8,
        out: &mut tuned1,
    })
    .expect("tuned fused (cold)");
    let mut tuned2 = vec![0.0f32; m * n];
    cuda.mpgemm_with_act_quant(tritium_spec::MpGemm {
        act: &act,
        weights: imma_buf.as_ref(),
        scales: &scales,
        shape,
        format: TernaryFormat::I2sInt8,
        out: &mut tuned2,
    })
    .expect("tuned fused (warm)");

    // Tuned == reference within tolerance.
    for (i, (&g, &c)) in tuned1.iter().zip(&ref_out).enumerate() {
        assert!(tol.accepts(g, c), "tuned vs ref [{i}] tuned={g} ref={c}");
    }
    // Cold vs warm cache: bit-for-bit identical (same tile → same numerics).
    for (i, (&a, &b)) in tuned1.iter().zip(&tuned2).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "cold vs warm tuned output diverges at [{i}] cold={a} warm={b}"
        );
    }
}

/// v0.3.1 de-risk: the device `rmsnorm_f32` decode kernel must reproduce the host
/// `tritium_nn::ops::rmsnorm` **bit-for-bit** (`to_bits` equal), so the fully
/// device-resident forward keeps greedy 256/256. This is the proof that a
/// sequential-f32 + FMA-disabled device kernel can match host f32 exactly; the
/// rest of the decode kernels follow the same discipline.
#[test]
fn rmsnorm_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rmsnorm bit-match: no device ({e})");
            return;
        }
    };
    // Host reference — identical to `tritium_nn::ops::rmsnorm` (this crate does
    // not depend on tritium-nn, so the formula is replicated verbatim).
    // Canonical tree sum-of-squares (ADR 0018) — replicates
    // `tritium_nn::ops::rmsnorm`'s documented cross-backend order (this
    // crate does not depend on tritium-nn).
    fn sum_squares_canonical(x: &[f32]) -> f32 {
        let mut part = [0.0f32; 256];
        for (i, &v) in x.iter().enumerate() {
            part[i % 256] += v * v;
        }
        let mut off = 128;
        while off > 0 {
            for t in 0..off {
                part[t] += part[t + off];
            }
            off >>= 1;
        }
        part[0]
    }
    fn host_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let mean_sq = sum_squares_canonical(x) / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect()
    }
    // BitNet hidden/ffn sizes + a few edge lengths; deterministic xorshift inputs.
    for &n in &[2560usize, 6912, 1, 17, 256, 2559] {
        let mut s = 0x1234_5678_9abc_def0u64 ^ (n as u64).wrapping_mul(0x9E37_79B9);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let x: Vec<f32> = (0..n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next()).collect();
        let eps = 1e-5f32;

        let want = host_rmsnorm(&x, &w, eps);
        let mut got = vec![0.0f32; n];
        backend
            .rmsnorm(&x, &w, eps, &mut got)
            .expect("device rmsnorm");

        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                h.to_bits(),
                "rmsnorm bit mismatch n={n} i={i}: got {g} ({:#010x}) want {h} ({:#010x})",
                g.to_bits(),
                h.to_bits()
            );
        }
    }
}

/// The device `rope_apply_f32` kernel must reproduce `tritium_nn::ops::rope_apply`
/// **bit-for-bit** for one token (M=1 decode). The trig is computed exactly as the
/// host op (f64 `sin_cos` → f32, data-independent) and the f32 rotation has no FMA.
#[test]
fn rope_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rope bit-match: no device ({e})");
            return;
        }
    };
    // BitNet 2B4T uses head_dim=128, n_head 20(Q)/5(KV), theta=500000.
    for &(n_head, head_dim) in &[(20usize, 128usize), (5, 128), (1, 8), (3, 64)] {
        let half = head_dim / 2;
        let theta = 500_000.0f32;
        for &pos in &[0usize, 1, 7, 255, 4095] {
            // Trig tables, identical to the host op (f64 sin_cos cast to f32).
            let theta_f64 = f64::from(theta);
            let inv_hd = 1.0 / head_dim as f64;
            let mut cos_t = vec![0.0f32; half];
            let mut sin_t = vec![0.0f32; half];
            for j in 0..half {
                let inv_freq = theta_f64.powf(-2.0 * j as f64 * inv_hd);
                let (s, c) = (pos as f64 * inv_freq).sin_cos();
                cos_t[j] = c as f32;
                sin_t[j] = s as f32;
            }
            // Deterministic input.
            let mut st = 0xDEAD_BEEF_CAFE_F00Du64
                ^ ((pos as u64) * 131 + n_head as u64 * 17 + head_dim as u64);
            let mut next = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                ((st >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
            };
            let x0: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();

            // Host rope (replicated; Rust does not auto-contract a*c - b*s to FMA).
            let mut want = x0.clone();
            for head in 0..n_head {
                let base = head * head_dim;
                for j in 0..half {
                    let a = x0[base + j];
                    let b = x0[base + j + half];
                    want[base + j] = a * cos_t[j] - b * sin_t[j];
                    want[base + j + half] = b * cos_t[j] + a * sin_t[j];
                }
            }

            let mut got = x0.clone();
            backend
                .rope(&mut got, &cos_t, &sin_t, n_head, head_dim)
                .expect("device rope");

            for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    h.to_bits(),
                    "rope bit mismatch (n_head={n_head} head_dim={head_dim} pos={pos}) i={i}: got {g} want {h}"
                );
            }
        }
    }
}

/// Measure device softmax vs host `softmax_rows`. The reductions are bit-matched;
/// the open question is `expf` (device CUDA libm vs host glibc). Reports the max
/// ULP difference + whether bit-exact, and asserts a tight relative tolerance so
/// the result is informative without spuriously failing on a ~1-ULP exp delta.
/// This is the gate-deciding measurement: bit-exact ⇒ strict greedy 256/256 is
/// reachable; otherwise the forward uses the perplexity+lockstep fallback.
#[test]
fn softmax_vs_host_exp_divergence() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping softmax divergence: no device ({e})");
            return;
        }
    };
    fn host_softmax(x: &mut [f32], row_len: usize) {
        for row in x.chunks_mut(row_len) {
            let mut m = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > m {
                    m = v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let e = (*v - m).exp();
                *v = e;
                sum += e;
            }
            let inv = 1.0f32 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
    let (rows, row_len) = (20usize, 1024usize); // decode-ish: n_head × ctx
    let mut s = 0x5151_5151_2727_2727u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 16.0 - 8.0
    };
    let x0: Vec<f32> = (0..rows * row_len).map(|_| next()).collect();
    let mut want = x0.clone();
    host_softmax(&mut want, row_len);
    let mut got = x0.clone();
    backend
        .softmax(&mut got, row_len, rows)
        .expect("device softmax");

    let (mut max_ulp, mut n_diff, mut max_rel) = (0i64, 0usize, 0.0f64);
    for (&g, &h) in got.iter().zip(&want) {
        let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
        if du != 0 {
            n_diff += 1;
        }
        max_ulp = max_ulp.max(du);
        if h != 0.0 {
            max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
        }
    }
    eprintln!(
        "softmax device-vs-host: max_ulp={max_ulp} n_diff={n_diff}/{} max_rel={max_rel:.3e} bit_exact={}",
        got.len(),
        n_diff == 0
    );
    assert!(
        max_rel < 1e-5,
        "device softmax exp diverges too far from host: max_rel={max_rel:.3e}"
    );
}

/// `residual_add` / `embedding_gather` / `lm_head` must match host bit-for-bit:
/// the first two are exact (add / copy), the LM head reproduces the host's
/// sequential dot in k-order (no FMA).
#[test]
fn residual_embed_lmhead_bit_match_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping residual/embed/lm_head bit-match: no device ({e})");
            return;
        }
    };
    let mut s = 0xABCD_1234_5678_9876u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
    };

    // residual_add: x += y (exact).
    {
        let n = 2560usize;
        let x0: Vec<f32> = (0..n).map(|_| next()).collect();
        let y: Vec<f32> = (0..n).map(|_| next()).collect();
        let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| a + b).collect();
        let mut got = x0.clone();
        backend.residual_add(&mut got, &y).expect("residual");
        for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "residual_add mismatch [{i}]");
        }
    }

    // embedding_gather: out = table[tok] (exact copy).
    {
        let (vocab, n_embd) = (64usize, 256usize);
        let table: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
        let tok = 37usize;
        let want = &table[tok * n_embd..tok * n_embd + n_embd];
        let mut got = vec![0.0f32; n_embd];
        backend
            .embedding_gather(&table, tok, n_embd, &mut got)
            .expect("embed");
        for (i, (&g, &h)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "embedding_gather mismatch [{i}]");
        }
    }

    // lm_head: sequential dot, bit-exact.
    {
        let (vocab, n_embd) = (128usize, 2560usize);
        let h: Vec<f32> = (0..n_embd).map(|_| next()).collect();
        let embd: Vec<f32> = (0..vocab * n_embd).map(|_| next()).collect();
        let mut want = vec![0.0f32; vocab];
        for (v, slot) in want.iter_mut().enumerate() {
            let row = &embd[v * n_embd..v * n_embd + n_embd];
            let mut acc = 0.0f32;
            for k in 0..n_embd {
                acc += h[k] * row[k];
            }
            *slot = acc;
        }
        let mut got = vec![0.0f32; vocab];
        backend
            .lm_head(&h, &embd, n_embd, vocab, &mut got)
            .expect("lm_head");
        for (v, (&g, &hh)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                hh.to_bits(),
                "lm_head mismatch [{v}]: got {g} want {hh}"
            );
        }
    }
}

/// `relu2_gate` must reproduce the host BitNet squared-ReLU FFN gate `r =
/// g.max(0); g = r*r*u` **bit-for-bit**. The input deliberately straddles zero so
/// the `max(.,0)` clamp (and the gate's hard zero on negatives) is exercised.
#[test]
fn relu2_gate_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping relu2_gate bit-match: no device ({e})");
            return;
        }
    };
    let mut s = 0x51A7_3C9E_2D6B_8F40u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Range [-4, 4): ~half the gate values negative, hitting the ReLU clamp.
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
    };
    let n = 6912usize; // BitNet 2B4T n_ff
    let gate0: Vec<f32> = (0..n).map(|_| next()).collect();
    let up: Vec<f32> = (0..n).map(|_| next()).collect();
    // Host reference: identical to layers::mlp's gating loop.
    let want: Vec<f32> = gate0
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let r = g.max(0.0);
            r * r * u
        })
        .collect();
    let mut got = gate0.clone();
    backend.relu2_gate(&mut got, &up).expect("relu2_gate");
    for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "relu2_gate mismatch [{i}]: got {g} want {h}"
        );
    }
}

/// Device GQA attention (M=1 decode) vs host `gqa_attention`. The dots + weighted
/// sums bit-match; the inline softmax `expf` gives a ≤3-ULP / ~1e-7 divergence, so
/// this measures the max rel error (reported) and asserts it stays tiny — the
/// attention output is the only forward op carrying the exp difference.
#[test]
fn gqa_attention_decode_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping attention match: no device ({e})");
            return;
        }
    };
    // BitNet 2B4T attention dims; a modest cached context for the decode token.
    let (n_head, n_head_kv, head_dim, ctx) = (20usize, 5usize, 128usize, 96usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let limit = ctx - 1; // steady-state decode: all cached keys visible
    let n_rep = n_head / n_head_kv;

    let mut s = 0x0BAD_F00D_1357_2468u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let q: Vec<f32> = (0..n_head * head_dim).map(|_| next()).collect();
    let k: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();
    let v: Vec<f32> = (0..ctx * n_head_kv * head_dim).map(|_| next()).collect();

    // Host reference — replicates ops::gqa_attention for seq=1.
    let mut want = vec![0.0f32; n_head * head_dim];
    let mut scores = vec![0.0f32; ctx];
    for h in 0..n_head {
        let kv = h / n_rep;
        let q_row = &q[h * head_dim..h * head_dim + head_dim];
        for (j, sc) in scores.iter_mut().enumerate() {
            if j > limit {
                *sc = f32::NEG_INFINITY;
                continue;
            }
            let k_row = &k[(j * n_head_kv + kv) * head_dim..][..head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            *sc = dot * scale;
        }
        let mut m = f32::NEG_INFINITY;
        for &sc in &scores {
            if sc > m {
                m = sc;
            }
        }
        let mut sum = 0.0f32;
        for sc in scores.iter_mut() {
            let e = (*sc - m).exp();
            *sc = e;
            sum += e;
        }
        let inv = 1.0f32 / sum;
        for sc in scores.iter_mut() {
            *sc *= inv;
        }
        let o = &mut want[h * head_dim..h * head_dim + head_dim];
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let v_row = &v[(j * n_head_kv + kv) * head_dim..][..head_dim];
            for d in 0..head_dim {
                o[d] += w * v_row[d];
            }
        }
    }

    let mut got = vec![0.0f32; n_head * head_dim];
    backend
        .gqa_attention_decode(
            &q, &k, &v, &mut got, ctx, n_head, n_head_kv, head_dim, scale, limit,
        )
        .expect("device attention");

    let (mut max_ulp, mut n_diff, mut max_rel, mut max_abs) = (0i64, 0usize, 0.0f64, 0.0f64);
    for (&g, &h) in got.iter().zip(&want) {
        let du = (i64::from(g.to_bits()) - i64::from(h.to_bits())).abs();
        if du != 0 {
            n_diff += 1;
        }
        max_ulp = max_ulp.max(du);
        max_abs = max_abs.max(f64::from((g - h).abs()));
        if h != 0.0 {
            max_rel = max_rel.max((f64::from(g - h) / f64::from(h)).abs());
        }
    }
    eprintln!(
        "attention device-vs-host: max_abs={max_abs:.3e} max_rel={max_rel:.3e} max_ulp={max_ulp} n_diff={n_diff}/{}",
        got.len()
    );
    // The dots + weighted sum bit-match; the sole divergence is the softmax `expf`
    // (≤3 ULP, ~1e-6 ABSOLUTE), which inflates to a larger *relative* error only on
    // near-zero (cancellation) outputs. The meaningful metric is the absolute error,
    // which must stay tiny (it propagates into the residual stream as a small add).
    assert!(
        max_abs < 1e-3,
        "device attention absolute error too large (likely a real bug): max_abs={max_abs:.3e}"
    );
}

/// `act_quant_tiled` must reproduce `ops::quantize_activation_int8` **bit-for-bit**
/// (the int8-as-f32 values and the per-token scale), including the zero-row case.
#[test]
fn act_quant_tiled_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping act_quant bit-match: no device ({e})");
            return;
        }
    };
    fn host_quant(act: &[f32]) -> (Vec<f32>, f32) {
        let mut gamma = 0.0f32;
        for &v in act {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            return (vec![0.0; act.len()], 0.0);
        }
        let s = 127.0f32 / gamma;
        (
            act.iter()
                .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                .collect(),
            gamma / 127.0,
        )
    }
    for &k in &[2560usize, 6912, 17, 1] {
        let mut s = 0x9999_AAAA_BBBB_CCCCu64 ^ k as u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let act: Vec<f32> = (0..k).map(|_| next()).collect();
        let (q_want, scale_want) = host_quant(&act);
        let mut q_got = vec![f32::NAN; k];
        let scale_got = backend
            .act_quant_tiled(&act, &mut q_got)
            .expect("act_quant");
        assert_eq!(
            scale_got.to_bits(),
            scale_want.to_bits(),
            "scale mismatch k={k}"
        );
        for (i, (&g, &h)) in q_got.iter().zip(&q_want).enumerate() {
            assert_eq!(g.to_bits(), h.to_bits(), "act_quant q mismatch k={k} i={i}");
        }
    }
    // Zero row → zeros + zero scale.
    let act = vec![0.0f32; 64];
    let mut q = vec![1.0f32; 64];
    let sc = backend
        .act_quant_tiled(&act, &mut q)
        .expect("act_quant zero");
    assert_eq!(sc, 0.0);
    assert!(
        q.iter().all(|&x| x == 0.0),
        "zero row must quantize to zeros"
    );
}

/// The fused `rmsnorm_quant_f32` decode kernel must reproduce host RMSNorm followed
/// by the host int8 activation-quant, **bit-for-bit** — it composes the same two ops
/// `rmsnorm_bit_matches_host` and `act_quant_tiled_bit_matches_host` already pin.
/// This is the standalone regression guard for the shared-memory aliasing bug: the
/// absmax reduction once reused `s_x` as scratch, clobbering the RMSNorm output before
/// the quant step → garbage activations that only surfaced in the end-to-end forward.
#[test]
fn rmsnorm_quant_bit_matches_host() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping rmsnorm_quant bit-match: no device ({e})");
            return;
        }
    };
    // Host reference = ops::rmsnorm (replicated, as elsewhere) then the host int8
    // activation-quant (absmax → 127/gamma scale → round-ties-even → clamp).
    fn host_rmsnorm_quant(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
        let n = x.len();
        // Canonical tree sum-of-squares (ADR 0018), as in the rmsnorm test.
        let mut part = [0.0f32; 256];
        for (i, &v) in x.iter().enumerate() {
            part[i % 256] += v * v;
        }
        let mut off = 128;
        while off > 0 {
            for t in 0..off {
                part[t] += part[t + off];
            }
            off >>= 1;
        }
        let mean_sq = part[0] / n as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let y: Vec<f32> = x.iter().zip(w).map(|(&xi, &wi)| xi * inv * wi).collect();
        let mut gamma = 0.0f32;
        for &v in &y {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            return (vec![0.0; n], 0.0);
        }
        let s = 127.0f32 / gamma;
        (
            y.iter()
                .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                .collect(),
            gamma / 127.0,
        )
    }
    // BitNet hidden/ffn widths + edge lengths; deterministic xorshift inputs.
    for &n in &[2560usize, 6912, 1, 17, 256, 2559] {
        let mut s = 0x0FED_CBA9_8765_4321u64 ^ (n as u64).wrapping_mul(0x9E37_79B9);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 8.0 - 4.0
        };
        let x: Vec<f32> = (0..n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next()).collect();
        let eps = 1e-5f32;

        let (q_want, scale_want) = host_rmsnorm_quant(&x, &w, eps);
        let mut q_got = vec![f32::NAN; n];
        let scale_got = backend
            .rmsnorm_quant(&x, &w, eps, &mut q_got)
            .expect("device rmsnorm_quant");

        assert_eq!(
            scale_got.to_bits(),
            scale_want.to_bits(),
            "rmsnorm_quant scale mismatch n={n}: got {scale_got} want {scale_want}"
        );
        for (i, (&g, &h)) in q_got.iter().zip(&q_want).enumerate() {
            assert_eq!(
                g.to_bits(),
                h.to_bits(),
                "rmsnorm_quant q mismatch n={n} i={i}: got {g} want {h}"
            );
        }
    }
    // All-zero input → zeros + zero scale (the gamma==0 branch).
    let x = vec![0.0f32; 128];
    let w = vec![1.0f32; 128];
    let mut q = vec![1.0f32; 128];
    let sc = backend
        .rmsnorm_quant(&x, &w, 1e-5, &mut q)
        .expect("rmsnorm_quant zero");
    assert_eq!(sc, 0.0, "all-zero input must give zero scale");
    assert!(
        q.iter().all(|&v| v == 0.0),
        "all-zero input must quantize to zeros"
    );
}

/// The device GEMM chain (`mpgemm_device`: on-device quant → tiled f64 GEMM →
/// scale fold) must reproduce the host path (`quantize_activation_int8` → tiled
/// `mpgemm` → `out *= act_scale`) **bit-for-bit** — same quant, same kernel, same
/// fold, just resident. This is the GEMM half of the device-resident decode.
#[test]
fn mpgemm_device_bit_matches_host_path() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping mpgemm_device match: no device ({e})");
            return;
        }
    };
    let (n, k) = (640usize, 2560usize); // BitNet attn_k projection shape
    let shape = GemmShape::new(1, n, k);

    let mut st = 0x1357_9BDF_2468_ACE0u64;
    let trits: Vec<tritium_core::Trit> = (0..n * k)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            tritium_core::Trit::from_i8(((st >> 33) % 3) as i8 - 1).unwrap()
        })
        .collect();
    let packed = pack_tq2_0(&trits, shape);
    let weights = cuda
        .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
        .expect("upload");

    let mut sf = 0x2468_ACE0_1357_9BDFu64;
    let mut nf = || {
        sf ^= sf << 13;
        sf ^= sf >> 7;
        sf ^= sf << 17;
        ((sf >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
    };
    let normed: Vec<f32> = (0..k).map(|_| nf()).collect();
    let scales: Vec<f32> = (0..n).map(|_| 0.5 + nf().abs()).collect();

    // Host path: quantize_activation_int8 + tiled mpgemm + per-token fold.
    let (q_host, act_scale) = {
        let mut gamma = 0.0f32;
        for &v in &normed {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        if gamma == 0.0 {
            (vec![0.0f32; k], 0.0f32)
        } else {
            let s = 127.0f32 / gamma;
            (
                normed
                    .iter()
                    .map(|&v| (v * s).round_ties_even().clamp(-128.0, 127.0))
                    .collect::<Vec<_>>(),
                gamma / 127.0,
            )
        }
    };
    let mut out_host = run_kernel(&cuda, &packed, &q_host, &scales, shape, AddKernel::Tiled);
    for v in out_host.iter_mut() {
        *v *= act_scale;
    }

    // Device chain.
    let mut out_dev = vec![0.0f32; n];
    cuda.mpgemm_device(&normed, weights.as_ref(), &scales, shape, &mut out_dev)
        .expect("mpgemm_device");

    for (i, (&g, &h)) in out_dev.iter().zip(&out_host).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "mpgemm_device mismatch [{i}]: got {g} want {h}"
        );
    }
}

/// CUDA-graph capture spike (v0.3.1 W2) — documents a hard cudarc-0.19 limitation.
///
/// Capturing the decode forward into a replayable graph would collapse the ~390
/// per-token kernel launches into one `graph.launch()`, the biggest remaining decode
/// win (the launch path is the wall at M=1). But cudarc 0.19's **safe** launch
/// (`LaunchArgs::launch`) waits on each buffer's read/write `CudaEvent` before the
/// kernel — and those events were recorded by the pre-capture uploads, so the very
/// first captured launch trips `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` ("dependency
/// created on uncaptured work"). RELAXED capture mode does not help (the dependency is
/// real, not a mode artifact). The raw escape — `result::launch_kernel`, which does no
/// event tracking — needs the `sys::CUfunction` handle, but cudarc keeps
/// `CudaFunction::cu_function` `pub(crate)`, so the only way through is a *parallel*
/// raw-FFI module/function/launch path (load the PTX via `result::module::load_data`,
/// `get_function`, hand-pack params), bypassing cudarc's safe layer entirely.
///
/// That raw path is the deferred W2 work (it materially expands the `unsafe` surface
/// of this `#![deny(unsafe_code)]` crate, so it is its own gated change). This test is
/// `#[ignore]`d: it asserts the limitation still holds, so if a future cudarc makes the
/// safe launch capture-compatible, this starts passing and flags that the raw path is
/// no longer needed.
#[test]
#[ignore = "cudarc 0.19 safe launch is capture-incompatible; W2 needs the raw-FFI path"]
fn cuda_graph_capture_blocked_by_cudarc_safe_launch() {
    use cudarc::driver::sys;
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cuda graph spike: no device ({e})");
            return;
        }
    };
    let n = 256usize;
    let x0 = vec![1.0f32; n];
    let y = vec![2.0f32; n];
    let cap = backend
        .stream
        .context()
        .new_stream()
        .expect("capture stream");
    let mut d_x = cap.clone_htod(&x0).expect("htod x");
    let d_y = cap.clone_htod(&y).expect("htod y");
    cap.synchronize().expect("sync");

    let n_i = n as i32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");
    let mut l = cap.launch_builder(&backend.func_residual);
    l.arg(&mut d_x).arg(&d_y).arg(&n_i);
    // SAFETY: `residual_add_f32(float* x, const float* y, int n)`.
    #[allow(unsafe_code)]
    let launched = unsafe { l.launch(cfg) };
    // The capture launch trips STREAM_CAPTURE_ISOLATION on cudarc 0.19. If this ever
    // succeeds, the safe launch became capture-compatible — revisit the raw-FFI plan.
    assert!(
        launched.is_err(),
        "cudarc safe launch unexpectedly captured cleanly — the raw-FFI W2 path may be unnecessary now"
    );
    let _ = cap.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
}

/// CUDA-graph **raw-FFI** capture spike (v0.3.2) — the path that works where the
/// safe launch trips isolation. Pre-extract each buffer's stable `CUdeviceptr`
/// *before* `begin_capture` (dropping the `SyncOnDrop` guard outside capture), raw-
/// load the decode PTX for a raw `CUfunction`, then capture two `residual_add_f32`
/// launches via `result::launch_kernel` (no cudarc event waits → no isolation), and
/// assert the single graph replay is **bit-identical** to the host reference. This
/// pins the v0.3.2 mechanic before the full decode forward is captured.
#[test]
fn cuda_graph_raw_launch_replay_bit_identical() {
    use cudarc::driver::{DevicePtr, DevicePtrMut, result, sys};
    use std::ffi::{CString, c_void};

    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping raw-graph spike: no device ({e})");
            return;
        }
    };
    let ctx = backend.stream.context().clone();
    ctx.bind_to_thread().expect("bind ctx");

    // Raw-load the decode PTX → a raw CUfunction (the safe CudaFunction hides
    // `cu_function`, so the captured launch needs this raw handle).
    let ptx_c = CString::new(DECODE_PTX).expect("ptx cstring");
    // SAFETY: `ptx_c` is a valid NUL-terminated PTX image; `load_data` JIT-compiles it.
    #[allow(unsafe_code)]
    let cu_module =
        unsafe { result::module::load_data(ptx_c.as_ptr() as *const c_void).expect("load_data") };
    let fname = CString::new("residual_add_f32").expect("fn cstring");
    // SAFETY: `cu_module` is a loaded module; `residual_add_f32` is one of its entry points.
    #[allow(unsafe_code)]
    let cu_func = unsafe { result::module::get_function(cu_module, fname).expect("get_function") };

    let n = 2560usize;
    let x0 = vec![1.0f32; n];
    let y = vec![2.0f32; n];
    // residual_add applied twice: ((x0 + y) + y), the kernel's single-f32-add order.
    let want: Vec<f32> = x0.iter().zip(&y).map(|(&a, &b)| (a + b) + b).collect();

    let cap = ctx.new_stream().expect("capture stream");
    let mut d_x = cap.clone_htod(&x0).expect("htod x");
    let d_y = cap.clone_htod(&y).expect("htod y");
    cap.synchronize().expect("pre-extract sync");

    // Pre-extract stable device pointers; drop the SyncOnDrop guards OUTSIDE capture
    // (their drop records an event, which is forbidden inside a capture).
    let px: sys::CUdeviceptr = {
        let (p, g) = d_x.device_ptr_mut(&cap);
        drop(g);
        p
    };
    let py: sys::CUdeviceptr = {
        let (p, g) = d_y.device_ptr(&cap);
        drop(g);
        p
    };
    cap.synchronize().expect("post-extract sync");

    let n_i = n as i32;
    let grid = ((n as u32).div_ceil(256), 1u32, 1u32);
    let block = (256u32, 1u32, 1u32);

    cap.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .expect("begin_capture");
    for _ in 0..2 {
        // kernel_params: each entry points to the arg VALUE (a CUdeviceptr for a
        // `float*`, the i32 for `int n`); these locals outlive the launch call, and
        // graph capture snapshots the values into the kernel node.
        let mut params: [*mut c_void; 3] = [
            (&px) as *const sys::CUdeviceptr as *mut c_void,
            (&py) as *const sys::CUdeviceptr as *mut c_void,
            (&n_i) as *const i32 as *mut c_void,
        ];
        // SAFETY: raw `residual_add_f32(float* x, const float* y, int n)`; params in
        // declaration order; `px`/`py` are valid device addresses (extracted above,
        // `d_x`/`d_y` alive for the test), `n_i` matches the buffer length.
        #[allow(unsafe_code)]
        unsafe {
            result::launch_kernel(cu_func, grid, block, 0, cap.cu_stream(), &mut params)
                .expect("raw capture launch");
        }
    }
    let graph = cap
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-empty graph");

    // d_x is still x0 (capture did not execute). One replay runs both adds.
    graph.launch().expect("graph launch");
    cap.synchronize().expect("post-replay sync");
    let mut got = vec![0.0f32; n];
    cap.memcpy_dtoh(&d_x, &mut got).expect("dtoh");
    for (i, (&g, &h)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "raw graph replay mismatch [{i}]: got {g} want {h}"
        );
    }

    // The captured graph holds the raw CUfunction; unload only after a final sync.
    cap.synchronize().expect("final sync");
    drop(graph);
    // SAFETY: `cu_module` was loaded above and is unloaded exactly once here, after the
    // graph (which referenced its function) is dropped and the stream is synchronized.
    #[allow(unsafe_code)]
    unsafe {
        result::module::unload(cu_module).expect("unload");
    }
}

// ── Sparse kernel tests (P1: zero-block sparsity skip) ─────────────────
use tritium_format::QK_K;

/// Build a trit vector with a known sparsity pattern: `zero_blocks` out of
/// `total_blocks` are all-zero (placed at the start), the rest are all +1.
/// Build an `[n, k]` row-major trit matrix (POS everywhere) with the first
/// `zero_blocks` TQ2_0 blocks of EACH row zeroed (partial last block respected
/// via `.min(k)`). Length is `n * k`, matching what `pack_tq2_0(.., shape)`
/// slices per row — so the per-row zero pattern is what `compute_zero_bitmaps`
/// will flag and the sparse kernel must skip.
fn make_sparse_trits(n: usize, k: usize, zero_blocks: usize) -> Vec<tritium_core::Trit> {
    let mut trits = vec![tritium_core::Trit::POS; n * k];
    for row in 0..n {
        let base = row * k;
        for b in 0..zero_blocks {
            let start = b * QK_K;
            let end = ((b + 1) * QK_K).min(k);
            for i in start..end {
                trits[base + i] = tritium_core::Trit::ZERO;
            }
        }
    }
    trits
}

/// The sparse-aware tiled kernel must match the CPU reference on mixed
/// zero/nonzero weights. This is the primary correctness gate for P1.
#[test]
fn sparse_kernel_matches_cpu_reference() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse kernel test: no device ({e})");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let tol = Tolerance::default();

    // K=4096 (16 blocks), ~40% zero blocks (7 out of 16)
    let nb = 16;
    let k = nb * QK_K; // 4096
    let n = 8;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 7);
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse_out =
        run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&g, &c)) in sparse_out.iter().zip(&cpu_out).enumerate() {
        assert!(tol.accepts(g, c), "sparse vs cpu [{i}]: sparse={g} cpu={c}");
    }
}

/// The sparse kernel must produce identical output to the dense tiled kernel
/// on the same weights (the bitmap just skips zero contributions, which the
/// dense kernel also skips via branchless `a * (code - 1)` where code=1).
#[test]
fn sparse_matches_dense_tiled_on_mixed_weights() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse-vs-dense test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    let nb = 16;
    let k = nb * QK_K;
    let n = 8;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 7);
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    // Dense tiled (double-accumulator, the reference-gated kernel)
    let dense = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

    // Sparse-aware tiled
    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&d, &s)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            tol.accepts(s, d),
            "sparse vs dense [{i}]: sparse={s} dense={d}"
        );
    }
}

/// All-zero weights: the sparse kernel must produce exactly zero output.
#[test]
fn sparse_kernel_all_zero_weights() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse all-zero test: no device ({e})");
            return;
        }
    };

    let nb = 4;
    let k = nb * QK_K;
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    // All-zero trits → every block is zero
    let trits = vec![tritium_core::Trit::ZERO; n * k];
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, &v) in sparse.iter().enumerate() {
        assert_eq!(v, 0.0, "all-zero weights should produce zero output [{i}]");
    }
}

/// No zero blocks: the sparse kernel must match the dense kernel exactly.
#[test]
fn sparse_kernel_no_zero_blocks() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse no-zero test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    let nb = 8;
    let k = nb * QK_K;
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    // All-positive trits → no zero blocks
    let trits = vec![tritium_core::Trit::POS; n * k];
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let dense = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&d, &s)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            tol.accepts(s, d),
            "no-zero sparse vs dense [{i}]: sparse={s} dense={d}"
        );
    }
}

/// Boundary shape: K not a multiple of QK_K (partial last block).
#[test]
fn sparse_kernel_partial_block() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping sparse partial test: no device ({e})");
            return;
        }
    };
    let tol = Tolerance::default();

    // K=300 → 2 blocks (one full 256, one partial 44)
    let k = 300usize;
    let nb = k.div_ceil(QK_K); // 2
    let n = 4;
    let m = 1;
    let shape = GemmShape::new(m, n, k);

    let trits = make_sparse_trits(n, k, 1); // block 0 zero, block 1 nonzero
    let packed = pack_tq2_0(&trits, shape);
    let act = seeded_f32(42, m * k, -1.0, 1.0);
    let scales = seeded_f32(99, n, 0.5, 2.0);

    let cpu = CpuBackend::new();
    let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);

    let bitmap =
        tritium_format::compute_zero_bitmaps(&packed, n, k, nb * TQ2_0_BLOCK_BYTES).unwrap();
    let words_per_row = nb.div_ceil(32);
    let sparse = run_kernel_sparse(&cuda, &packed, &act, &scales, &bitmap, words_per_row, shape);

    for (i, (&g, &c)) in sparse.iter().zip(&cpu_out).enumerate() {
        assert!(
            tol.accepts(g, c),
            "partial block sparse vs cpu [{i}]: sparse={g} cpu={c}"
        );
    }
}

/// ADR 0022 guardrail with teeth: the twin-kernel family table must match
/// decode.cu. Drift — a new variant (the revisit trigger: a 4th KV rung or a
/// new attention family) or a removed one — fails here mechanically instead
/// of relying on reviewer memory. No GPU needed: this parses the source.
#[test]
fn adr_0022_twin_family_table_matches_decode_cu() {
    let src = include_str!("../../kernels/decode.cu");
    let names: Vec<&str> = src
        .lines()
        .filter_map(|l| l.strip_prefix("__global__ void "))
        .map(|l| l.split('(').next().unwrap_or(l).trim())
        .collect();
    let count = |prefix: &str| {
        names
            .iter()
            .filter(|n| {
                // exact family match: prefix, then either end or a variant
                // suffix — avoids kv_append counting kv_append_batch.
                n.strip_prefix(prefix).is_some_and(|rest| {
                    rest.is_empty() || matches!(rest, "_g" | "_h" | "_q8" | "_t2" | "_f32" | "_f16")
                })
            })
            .count()
    };
    // The ADR 0022 family table (docs/adr/0022-twin-kernel-contract.md).
    let table = [
        ("rope_kv_fused", 4),
        ("kv_append", 4),
        ("kv_append_batch", 4),
        ("gqa_attention_scores", 3),
        ("gqa_attention_reduce", 3),
        ("gqa_attention_batch", 3),
        ("gqa_attention_tree_scores", 3),
        ("gqa_attention_tree_reduce", 3),
        ("lm_head_warp", 2),
    ];
    for (family, want) in table {
        assert_eq!(
            count(family),
            want,
            "twin family `{family}` drifted from the ADR 0022 table — update \
             the ADR (and check the revisit trigger) alongside the kernel"
        );
    }
    assert_eq!(
        names.len(),
        66,
        "decode.cu kernel count drifted from ADR 0022 — update the ADR \
         (65 → 64: gqa_attention_mdecode_f32 retired; 64 → 66: paged KV \
         twins added, ADR 0025 step 2; rmsnorm_quant_i8_fast was added and \
         DELETED by measurement — ADR 0023 rejected, +1.75% < the 3% bar)"
    );
}

/// A2 gate: the TQ1_0-native i8-scaled kernel is BIT-identical to the TQ2_0
/// one on the same trits (integer accumulation; identical epilogue) — for the
/// plain and residual twins, across aligned and tail (k % 256 != 0) shapes.
#[test]
fn tq1_matches_tq2_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::{TQ1_0_BLOCK_BYTES, num_blocks, pack_tq1_0_row};

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping tq1-vs-tq2 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
        .expect("load add module");
    let f_tq2 = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .expect("tq2 fn");
    let f_tq1 = module
        .load_function("tq1_0_add_mpgemm_tiled_i8_scaled")
        .expect("tq1 fn");
    let f_tq2_res = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled_residual")
        .expect("tq2 res fn");
    let f_tq1_res = module
        .load_function("tq1_0_add_mpgemm_tiled_i8_scaled_residual")
        .expect("tq1 res fn");

    // Aligned + tail shapes; m > 1 exercises the act_scale fold per row.
    for &(m, n, k) in &[(1usize, 8usize, 1024usize), (2, 5, 256 + 128)] {
        let trits = mixed_trits(n, k, 0xA2 ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let (rb2, rb1) = (nb * TQ2_0_BLOCK_BYTES, nb * TQ1_0_BLOCK_BYTES);
        let mut p2 = vec![0u8; n * rb2];
        let mut p1 = vec![0u8; n * rb1];
        for ni in 0..n {
            let row = &trits[ni * k..(ni + 1) * k];
            tritium_format::pack_tq2_0_row(row, &unit, &mut p2[ni * rb2..(ni + 1) * rb2])
                .expect("pack tq2");
            pack_tq1_0_row(row, &unit, &mut p1[ni * rb1..(ni + 1) * rb1]).expect("pack tq1");
        }
        // Deterministic i8 activations (k % 4 == 0 holds for both shapes).
        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 255) as i8).collect();
        let scales = seeded_f32(7, n, 0.5, 2.0);
        let act_scale = seeded_f32(13, m, 0.5, 1.5);
        let residual = seeded_f32(21, m * n, -1.0, 1.0);

        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&p2).unwrap();
        let d_w1 = stream.clone_htod(&p1).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let d_res = stream.clone_htod(&residual).unwrap();
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let (rb2_i, rb1_i) = (rb2 as i32, rb1 as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
            block_dim: (8 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let launch_plain = |f: &cudarc::driver::CudaFunction,
                            w: &cudarc::driver::CudaSlice<u8>,
                            rb: &i32|
         -> Vec<f32> {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let mut l = stream.launch_builder(f);
            l.arg(&d_qact)
                .arg(w)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(rb);
            // SAFETY: matches the kernel signatures asserted in the kernel
            // source; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };
        let o2 = launch_plain(&f_tq2, &d_w2, &rb2_i);
        let o1 = launch_plain(&f_tq1, &d_w1, &rb1_i);
        for (i, (a, b)) in o2.iter().zip(&o1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "plain m{m} n{n} k{k} [{i}]: tq2={a} tq1={b}"
            );
        }

        let launch_res = |f: &cudarc::driver::CudaFunction,
                          w: &cudarc::driver::CudaSlice<u8>,
                          rb: &i32|
         -> Vec<f32> {
            let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
            let mut l = stream.launch_builder(f);
            l.arg(&d_qact)
                .arg(w)
                .arg(&d_sc)
                .arg(&d_as)
                .arg(&d_res)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(rb);
            // SAFETY: residual-twin signature; grid.y = m.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
            let mut out = vec![0.0f32; m * n];
            stream.memcpy_dtoh(&d_out, &mut out).unwrap();
            out
        };
        let r2 = launch_res(&f_tq2_res, &d_w2, &rb2_i);
        let r1 = launch_res(&f_tq1_res, &d_w1, &rb1_i);
        for (i, (a, b)) in r2.iter().zip(&r1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "residual m{m} n{n} k{k} [{i}]: tq2={a} tq1={b}"
            );
        }
    }
}

/// Mixed-sign random trits (~1/3 each of -1/0/+1) — the A2/A4 bit-equality
/// gates MUST use these: `make_sparse_trits` zeroes leading blocks and emits
/// no -1 at all, which made the original gates vacuous (review-found: both
/// kernels correctly output 0.0 on all-zero weights, proving nothing).
#[cfg(test)]
fn mixed_trits(n: usize, k: usize, seed: u64) -> Vec<tritium_core::Trit> {
    let mut s = seed;
    (0..n * k)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            tritium_core::Trit::from_i8(((s >> 33) % 3) as i8 - 1).unwrap()
        })
        .collect()
}

/// A4 harness: upload TB1 rows (concatenated variable-length + offsets) and
/// launch the prototype kernel.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_tb1(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    trits: &[tritium_core::Trit],
    qact: &[i8],
    scales: &[f32],
    act_scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let mut arena: Vec<u8> = Vec::new();
    // u32 offsets cap the arena at 4 GiB — fine for the prototype scale.
    let mut offsets: Vec<u32> = Vec::with_capacity(n);
    for ni in 0..n {
        offsets.push(arena.len() as u32);
        arena.extend(tritium_format::pack_tb1_row(&trits[ni * k..(ni + 1) * k]).unwrap());
    }
    arena.extend_from_slice(&[0u8; 4]); // sign-read slack (kernel loads byte0+1)
    let d_w = stream.clone_htod(&arena).unwrap();
    let d_off = stream.clone_htod(&offsets).unwrap();
    let d_qact = stream.clone_htod(qact).unwrap();
    let d_sc = stream.clone_htod(scales).unwrap();
    let d_as = stream.clone_htod(act_scale).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
        block_dim: (8 * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut l = stream.launch_builder(f);
    l.arg(&d_qact)
        .arg(&d_w)
        .arg(&d_off)
        .arg(&d_sc)
        .arg(&d_as)
        .arg(&mut d_out)
        .arg(&m_i)
        .arg(&n_i)
        .arg(&k_i);
    // SAFETY: tb1_mpgemm_tiled_i8_scaled(qact, weights, row_offsets, scales,
    // act_scale, out, m, n, k); grid.y = m.
    #[allow(unsafe_code)]
    unsafe {
        l.launch(cfg).unwrap()
    };
    let mut out = vec![0.0f32; m * n];
    stream.memcpy_dtoh(&d_out, &mut out).unwrap();
    out
}

/// A4 gate: TB1 bitmap+signs kernel is BIT-identical to the TQ2 i8-scaled
/// kernel on the same trits (integer accumulation, same epilogue).
#[test]
fn tb1_matches_tq2_tiled_scaled_bit_exact() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping tb1 gate: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(TQ2_0_ADD_PTX)).unwrap();
    let f_tq2 = module
        .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
        .unwrap();
    let f_tb1 = module.load_function("tb1_mpgemm_tiled_i8_scaled").unwrap();

    for &(m, n, k) in &[(1usize, 8usize, 1024usize), (2, 5, 512)] {
        let trits = mixed_trits(n, k, 0xB1 ^ (k as u64));
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let rb2 = nb * TQ2_0_BLOCK_BYTES;
        let mut p2 = vec![0u8; n * rb2];
        for ni in 0..n {
            tritium_format::pack_tq2_0_row(
                &trits[ni * k..(ni + 1) * k],
                &unit,
                &mut p2[ni * rb2..(ni + 1) * rb2],
            )
            .unwrap();
        }
        let qact: Vec<i8> = (0..m * k).map(|i| ((i * 41 + 5) % 251) as i8).collect();
        let scales = seeded_f32(3, n, 0.5, 2.0);
        let act_scale = seeded_f32(9, m, 0.5, 1.5);

        // TQ2 reference launch.
        let d_qact = stream.clone_htod(&qact).unwrap();
        let d_w2 = stream.clone_htod(&p2).unwrap();
        let d_sc = stream.clone_htod(&scales).unwrap();
        let d_as = stream.clone_htod(&act_scale).unwrap();
        let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
        let (m_i, n_i, k_i, rb_i) = (m as i32, n as i32, k as i32, rb2 as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
            block_dim: (8 * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = stream.launch_builder(&f_tq2);
        l.arg(&d_qact)
            .arg(&d_w2)
            .arg(&d_sc)
            .arg(&d_as)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&rb_i);
        // SAFETY: 9-arg dense signature; grid.y = m.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg).unwrap()
        };
        let mut o2 = vec![0.0f32; m * n];
        stream.memcpy_dtoh(&d_out, &mut o2).unwrap();

        let o1 = run_tb1(&stream, &f_tb1, &trits, &qact, &scales, &act_scale, m, n, k);
        for (i, (a, b)) in o2.iter().zip(&o1).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "tb1 m{m} n{n} k{k} [{i}]: tq2={a} tb1={b}"
            );
        }
    }
}

/// A4 verdict bench (run explicitly): TQ2 vs TQ1 vs TB1 kernel wall-time on
/// the REAL gateup shape (the one DRAM-bound decode GEMM) at M=1.
#[test]
#[ignore = "A4 head-to-head bench: run with --ignored --nocapture"]
fn tb1_tq1_tq2_gateup_bench() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use tritium_format::TQ1_0_BLOCK_BYTES;
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping bench: no device ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(TQ2_0_ADD_PTX)).unwrap();
    let (m, n, k) = (1usize, 13824usize, 2560usize); // BitNet fused gateup
    let iters = 2000u32;

    // Mixed-sign ~1/3 zeros — closer to BitNet's 42% than block-structured
    // patterns; the bench also PRINTS actual byte counts (self-documenting).
    let trits = mixed_trits(n, k, 0xBE);
    let nb = num_blocks(k);
    let unit = vec![half::f16::ONE; nb];
    let (rb2, rb1) = (nb * TQ2_0_BLOCK_BYTES, nb * TQ1_0_BLOCK_BYTES);
    let mut p2 = vec![0u8; n * rb2];
    let mut p1 = vec![0u8; n * rb1];
    let mut tb1: Vec<u8> = Vec::new();
    let mut off: Vec<u32> = Vec::new();
    for ni in 0..n {
        let row = &trits[ni * k..(ni + 1) * k];
        tritium_format::pack_tq2_0_row(row, &unit, &mut p2[ni * rb2..(ni + 1) * rb2]).unwrap();
        tritium_format::pack_tq1_0_row(row, &unit, &mut p1[ni * rb1..(ni + 1) * rb1]).unwrap();
        off.push(tb1.len() as u32);
        tb1.extend(tritium_format::pack_tb1_row(row).unwrap());
    }
    tb1.extend_from_slice(&[0u8; 4]);
    println!(
        "weight bytes: TQ2 {} | TQ1 {} ({:.1}%) | TB1 {} ({:.1}%)",
        n * rb2,
        n * rb1,
        (n * rb1) as f64 / (n * rb2) as f64 * 100.0,
        tb1.len(),
        tb1.len() as f64 / (n * rb2) as f64 * 100.0,
    );

    let qact: Vec<i8> = (0..m * k).map(|i| ((i * 37) % 253) as i8).collect();
    let scales = seeded_f32(1, n, 0.5, 2.0);
    let act_scale = seeded_f32(2, m, 0.5, 1.5);
    let d_qact = stream.clone_htod(&qact).unwrap();
    let d_sc = stream.clone_htod(&scales).unwrap();
    let d_as = stream.clone_htod(&act_scale).unwrap();
    let d_w2 = stream.clone_htod(&p2).unwrap();
    let d_w1 = stream.clone_htod(&p1).unwrap();
    let d_wb = stream.clone_htod(&tb1).unwrap();
    let d_off = stream.clone_htod(&off).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(m * n).unwrap();
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
    let (rb2_i, rb1_i) = (rb2 as i32, rb1 as i32);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(8), m as u32, 1),
        block_dim: (8 * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut bench = |name: &str, which: u8| {
        let f = match which {
            0 => module
                .load_function("tq2_0_add_mpgemm_tiled_i8_scaled")
                .unwrap(),
            1 => module
                .load_function("tq1_0_add_mpgemm_tiled_i8_scaled")
                .unwrap(),
            _ => module.load_function("tb1_mpgemm_tiled_i8_scaled").unwrap(),
        };
        // Warm.
        for _ in 0..50 {
            let mut l = stream.launch_builder(&f);
            match which {
                0 => l
                    .arg(&d_qact)
                    .arg(&d_w2)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb2_i),
                1 => l
                    .arg(&d_qact)
                    .arg(&d_w1)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb1_i),
                _ => l
                    .arg(&d_qact)
                    .arg(&d_wb)
                    .arg(&d_off)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i),
            };
            // SAFETY: signatures as gated bit-exact above.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg).unwrap()
            };
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let mut l = stream.launch_builder(&f);
            match which {
                0 => l
                    .arg(&d_qact)
                    .arg(&d_w2)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb2_i),
                1 => l
                    .arg(&d_qact)
                    .arg(&d_w1)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&rb1_i),
                _ => l
                    .arg(&d_qact)
                    .arg(&d_wb)
                    .arg(&d_off)
                    .arg(&d_sc)
                    .arg(&d_as)
                    .arg(&mut d_out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i),
            };
            #[allow(unsafe_code)]
            // SAFETY: each branch above pushes the exact argument list for its
            // selected kernel; all device buffers cover the configured shape.
            unsafe {
                l.launch(cfg).unwrap()
            };
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
        println!("{name}: {us:.2} µs/launch");
    };
    bench("TQ2 (2.06 b/w)", 0);
    bench("TQ1 (1.69 b/w)", 1);
    bench("TB1 (1.58 b/w)", 2);
}
