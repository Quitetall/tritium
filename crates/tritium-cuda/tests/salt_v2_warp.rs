//! Parity gate for the warp-per-row SALT V2 forward kernel.
//!
//! The kernel splits a row's scale groups across a warp's lanes and replays the
//! additions in the scalar kernel's order from shared memory. That reordering is
//! only invisible if it is exactly a reordering, so this compares against the
//! CPU reference with `assert_eq!` rather than a tolerance.
//!
//! The shapes here are chosen to *reach* the warp path: `tritium-cuda`'s other
//! SALT tests use 576 columns, which is not a whole number of 256-coefficient
//! allocation tiles, so they fall back to the scalar kernel and would pass no
//! matter what this kernel did.
#![cfg(feature = "cuda")]

use half::f16;
use tritium_cpu::salt_v2::salt_v2_matvec;
use tritium_cuda::{CudaBackend, SaltV2ForwardMode};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
};

const COLUMNS: usize = 512;
const ROWS: usize = 5;
const SCALE_GROUP: usize = 64;

/// A tensor whose tiles carry one, two and three planes in turn.
///
/// The plane count drives the rank-prefix walk and decides how many of each
/// group's three contribution slots stay at the padding value, so cycling it is
/// what makes the ordered replay non-trivial.
fn warp_tensor() -> SaltV2Tensor {
    let tile_count = ROWS * COLUMNS / 256;
    let tiles = (0..tile_count)
        .map(|tile_index| {
            let planes = (0..(tile_index % 3) + 1)
                .map(|plane_index| {
                    // S34 encodes exactly one zero per group of four, so the
                    // pattern has to place one there and nowhere else; D2 and B3
                    // accept it unchanged.
                    let zero_slot = (tile_index + plane_index) % 4;
                    let trits = (0..256)
                        .map(|index| {
                            if index % 4 == zero_slot {
                                0
                            } else if (index + tile_index) % 3 == 0 {
                                1
                            } else {
                                -1
                            }
                        })
                        .collect();
                    let scales = (0..256 / SCALE_GROUP)
                        .map(|group| {
                            let step = (tile_index * 4 + group + plane_index) as f32;
                            f16::from_f32(0.0625 + step / 96.0)
                        })
                        .collect();
                    SaltV2Plane::new_with_scale_group_size(trits, scales, SCALE_GROUP).unwrap()
                })
                .collect::<Vec<_>>();
            SaltV2Tile::new(planes).unwrap()
        })
        .collect();
    SaltV2Tensor::new_with_layout(
        "warp.weight",
        vec![ROWS as u64, COLUMNS as u64],
        SaltV2Transform::None,
        SCALE_GROUP,
        tiles,
    )
    .unwrap()
}

fn activation(batch: usize) -> Vec<f32> {
    (0..batch * COLUMNS)
        .map(|index| (index as f32 - 611.0) / 97.0)
        .collect()
}

#[test]
fn warp_forward_matches_cpu_bit_for_bit_across_codecs_and_batches() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 warp parity: no device ({error})");
            return;
        }
    };
    let tensor = warp_tensor();
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let resident = cuda.upload_salt_v2(&tensor, codec).unwrap();
        for batch in [1usize, 3] {
            let act = activation(batch);
            let mut expected = Vec::with_capacity(batch * ROWS);
            for row in 0..batch {
                let slice = &act[row * COLUMNS..(row + 1) * COLUMNS];
                expected.extend(salt_v2_matvec(&package, 0, slice).unwrap().output);
            }
            let measured = cuda.salt_v2_forward_exact(&resident, &act, batch).unwrap();
            assert_eq!(measured.output, expected, "{codec:?} batch {batch}");
            assert_eq!(measured.receipt.dense_weight_bytes(), 0);
        }
    }
}

/// The fast entry point: close to the CPU reference, and labeled honestly.
///
/// `salt_v2_forward_warp_fast` reduces by warp shuffle instead of replaying the
/// scalar kernel's addition order, which reassociates the K-sum. So this gate is
/// a relative-error bound, deliberately not `assert_eq!` -- and it also pins the
/// receipt, because a fast entry point that quietly fell back to the exact image
/// would pass an error bound trivially while delivering none of the speed.
#[test]
fn fast_forward_is_close_to_the_reference_and_says_which_kernel_ran() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 fast parity: no CUDA device");
        return;
    };
    let tensor = warp_tensor();
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let resident = cuda.upload_salt_v2(&tensor, codec).unwrap();
        for batch in [1usize, 3] {
            let act = activation(batch);
            let mut expected = Vec::new();
            for row in 0..batch {
                let slice = &act[row * COLUMNS..(row + 1) * COLUMNS];
                expected.extend(salt_v2_matvec(&package, 0, slice).unwrap().output);
            }
            let measured = cuda.salt_v2_forward_fast(&resident, &act, batch).unwrap();
            assert_eq!(
                measured.receipt.mode(),
                SaltV2ForwardMode::FastWarpReduce,
                "{codec:?} batch {batch}: the fast kernel did not run, so this \
                 comparison says nothing about it"
            );
            let scale = expected
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()))
                .max(f32::MIN_POSITIVE);
            for (lane, (&got, &want)) in measured.output.iter().zip(&expected).enumerate() {
                let relative = (got - want).abs() / scale;
                assert!(
                    relative <= 1e-5,
                    "{codec:?} batch {batch} lane {lane}: {got} vs {want} \
                     (relative {relative:.3e})"
                );
            }
            // Reassociation should move something; if the fast kernel were an
            // alias this gate would be measuring the exact kernel twice.
            let exact = cuda.salt_v2_forward_exact(&resident, &act, batch).unwrap();
            assert_eq!(exact.receipt.mode(), SaltV2ForwardMode::Exact);
            assert_eq!(
                exact.output, expected,
                "{codec:?} batch {batch} exact drifted"
            );
        }
    }
}

/// A shape the fast kernel cannot serve must say so rather than claim speed.
#[test]
fn an_ineligible_shape_reports_that_the_fast_entry_point_aliased_exact() {
    let Ok(cuda) = CudaBackend::new(0) else {
        eprintln!("skipping SALT V2 fast fallback check: no CUDA device");
        return;
    };
    // 576 columns is 2.25 allocation tiles, so the warp kernels decline it --
    // the width `salt_v2_g64.rs` uses, which is why that file never covered them.
    let columns = 576usize;
    let rows = 4usize;
    let tiles = (0..rows * columns / 256)
        .map(|tile_index| {
            let zero_slot = tile_index % 4;
            let trits = (0..256)
                .map(|index| {
                    if index % 4 == zero_slot {
                        0
                    } else if (index + tile_index) % 3 == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect();
            let scales = (0..256 / SCALE_GROUP)
                .map(|group| f16::from_f32(0.0625 + group as f32 / 96.0))
                .collect();
            SaltV2Tile::new(vec![
                SaltV2Plane::new_with_scale_group_size(trits, scales, SCALE_GROUP).unwrap(),
            ])
            .unwrap()
        })
        .collect();
    let tensor = SaltV2Tensor::new_with_layout(
        "ineligible.weight",
        vec![rows as u64, columns as u64],
        SaltV2Transform::None,
        SCALE_GROUP,
        tiles,
    )
    .unwrap();
    let resident = cuda.upload_salt_v2(&tensor, SaltV2Codec::B3).unwrap();
    let act: Vec<f32> = (0..columns).map(|i| (i as f32 - 300.0) / 97.0).collect();
    let measured = cuda.salt_v2_forward_fast(&resident, &act, 1).unwrap();
    assert_eq!(measured.receipt.mode(), SaltV2ForwardMode::FastAliasesExact);
}
