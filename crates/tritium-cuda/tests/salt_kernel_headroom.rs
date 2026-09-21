//! How much faster is the optimized resident SALT GEMM than the codec kernel?
//!
//! `salt_v2_forward_*` is the correctness-first path: it decodes D2/B3/S34
//! straight out of the packed codec inside the contraction, and it is what the
//! Qwen3.6 bundle runs today at ~98 ms of every 148 ms decode token.
//! `salt_mpgemm_tiled_f32` is the v0.4.0 optimized multi-plane GEMM over
//! plane-major TQ2_0, and nothing on the Qwen path uses it.
//!
//! Before designing a conversion, measure whether it is actually worth one. This
//! times both at a real Qwen3.6-27B projection shape, matched on N, K and plane
//! count, so the comparison is cost at equal work. It is not a numeric
//! comparison: TQ2_0 scales per 256-trit block and SALT V2 per 64, so the two
//! represent different weights on purpose.
//!
//! Measured on a 4090, N = 17408, K = 5120, m = 1 (gate_proj / up_proj):
//!
//!     planes=1   codec 1.232 ms   resident 0.542 ms   2.27x
//!     planes=2   codec 1.464 ms   resident 1.001 ms   1.46x
//!     planes=3   codec 1.675 ms   resident 0.865 ms   1.94x
//!
//! Two things block using it as-is, and both are real work rather than wiring:
//!
//! 1. Scale granularity. TQ2_0 carries one f16 per 256-trit block; the Qwen
//!    bundle scales per 64. Converting would change the model's numerics, so a
//!    faithful swap needs a 64-block resident format, not a repack.
//! 2. `TILED_K_MAX` is 8192, and `down_proj` has K = 17408. That is a third of
//!    the MLP weight, excluded until the tile bound is lifted.
//!
//! `SaltResidentLinear` also pads ragged rows to a uniform plane count, so a
//! tensor with any three-plane tile costs three planes everywhere -- at TQ2_0's
//! ~2.06 bits per trit per plane that is a VRAM question the conversion has to
//! answer before it is worth doing.
#![cfg(feature = "cuda")]

use std::time::Instant;

use half::f16;
use tritium_core::Trit;
use tritium_cuda::CudaBackend;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform};
use tritium_format::{SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};

// gate_proj / up_proj of Qwen3.6-27B. `down_proj` has K = 17408, past the
// tiled kernel's TILED_K_MAX of 8192, so it could not use this path at all.
const K: usize = 5120;
const SCALE_GROUP: usize = 64;

fn seeded() -> impl FnMut() -> u64 {
    let mut state: u64 = 0x5A17_F00D_1234_5678;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    }
}

fn time_ms(label: &str, iterations: usize, mut body: impl FnMut()) -> f64 {
    body(); // warm
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    let per = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    eprintln!("  {label:38} {per:8.3} ms");
    per
}

#[test]
#[ignore = "performance probe: needs a CUDA device"]
fn resident_tq2_0_salt_gemm_versus_the_codec_kernel() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT kernel headroom probe: no CUDA device");
        return;
    };
    // N is kept modest so the probe builds quickly; both kernels scale in N, and
    // the ratio is what this measures.
    let n: usize = std::env::var("TRITIUM_PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let mut next = seeded();
    let blocks = num_blocks(K);
    let row_bytes = blocks * TQ2_0_BLOCK_BYTES;

    for planes in [1usize, 2, 3] {
    let rows: Vec<SaltRow> = (0..n)
        .map(|_| {
            let planes = (0..planes)
                .map(|plane| {
                    let trits: Vec<Trit> = (0..K)
                        .map(|_| Trit::from_i8(((next() >> 40) % 3) as i8 - 1).unwrap())
                        .collect();
                    let scales: Vec<f16> = (0..blocks)
                        .map(|_| f16::from_f32(0.05 / (plane as f32 + 1.0)))
                        .collect();
                    let mut bytes = vec![0u8; row_bytes];
                    pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                    bytes
                })
                .collect();
            SaltRow { k: K, planes }
        })
        .collect();

    let tiles = (0..n * K / 256)
        .map(|_| {
            let plane_list = (0..planes)
                .map(|_| {
                    let trits = (0..256)
                        .map(|_| ((next() >> 40) % 3) as i8 - 1)
                        .collect::<Vec<i8>>();
                    let scales = (0..256 / SCALE_GROUP)
                        .map(|_| f16::from_f32(0.05))
                        .collect();
                    SaltV2Plane::new_with_scale_group_size(trits, scales, SCALE_GROUP).unwrap()
                })
                .collect::<Vec<_>>();
            SaltV2Tile::new(plane_list).unwrap()
        })
        .collect();
    let tensor = SaltV2Tensor::new_with_layout(
        "probe.weight",
        vec![n as u64, K as u64],
        SaltV2Transform::None,
        SCALE_GROUP,
        tiles,
    )
    .unwrap();

    let resident_v2 = cuda.upload_salt_v2(&tensor, SaltV2Codec::B3).unwrap();
    let resident_tq2 = cuda.upload_salt(&rows, n, K).unwrap();
    let activation: Vec<f32> = (0..K)
        .map(|index| (index as f32 * 0.013).sin() * 0.5)
        .collect();

    eprintln!("N={n} K={K} planes={planes} m=1");
    let codec = time_ms("salt_v2_forward_exact (codec, shipping)", 20, || {
        let _ = cuda.salt_v2_forward_exact(&resident_v2, &activation, 1).unwrap();
    });
    let gemm = time_ms("salt_mpgemm_tiled_f32 (resident TQ2_0)", 20, || {
        let _ = cuda.salt_forward(&resident_tq2, &activation, 1).unwrap();
    });
    eprintln!("  ratio: resident TQ2_0 is {:.2}x the codec kernel's cost\n", gemm / codec);
    assert!(codec > 0.0 && gemm > 0.0);
    }
}
