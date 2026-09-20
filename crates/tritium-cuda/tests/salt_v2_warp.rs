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
use tritium_cuda::CudaBackend;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
};

const SCALE_GROUP: usize = 64;

/// Allocation-map rank prefixes are stored every 256 tiles, so a tensor has to
/// exceed that before the prepass's anchor path runs at all.
const RANK_STRIDE_TILES: usize = 256;

/// A tensor whose tiles carry one, two and three planes in turn.
///
/// The plane count drives the rank-prefix walk and decides how many of each
/// group's three contribution slots stay at the padding value, so cycling it is
/// what makes the ordered replay non-trivial.
fn warp_tensor(rows: usize, columns: usize) -> SaltV2Tensor {
    let tile_count = rows * columns / 256;
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
                    SaltV2Plane::new_with_scale_group_size(trits, scales, SCALE_GROUP)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            SaltV2Tile::new(planes).unwrap()
        })
        .collect();
    SaltV2Tensor::new_with_layout(
        "warp.weight",
        vec![rows as u64, columns as u64],
        SaltV2Transform::None,
        SCALE_GROUP,
        tiles,
    )
    .unwrap()
}

fn activation(batch: usize, columns: usize) -> Vec<f32> {
    (0..batch * columns)
        .map(|index| (index as f32 - 611.0) / 97.0)
        .collect()
}

fn assert_parity(cuda: &CudaBackend, rows: usize, columns: usize, label: &str) {
    let tensor = warp_tensor(rows, columns);
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let resident = cuda.upload_salt_v2(&tensor, codec).unwrap();
        for batch in [1usize, 3] {
            let act = activation(batch, columns);
            let mut expected = Vec::with_capacity(batch * rows);
            for row in 0..batch {
                let slice = &act[row * columns..(row + 1) * columns];
                expected.extend(salt_v2_matvec(&package, 0, slice).unwrap().output);
            }
            let measured = cuda.salt_v2_forward_exact(&resident, &act, batch).unwrap();
            assert_eq!(measured.output, expected, "{label} {codec:?} batch {batch}");
            assert_eq!(measured.receipt.dense_weight_bytes(), 0);
        }
    }
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
    // Two tiles per row, ten tiles total: entirely inside rank block 0, so the
    // prepass carries its running sum from an offset of zero.
    assert_parity(&cuda, 5, 512, "single rank block");
}

#[test]
fn warp_forward_matches_cpu_across_a_rank_prefix_boundary() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 warp rank-prefix parity: no device ({error})");
            return;
        }
    };
    // Three tiles per row does not divide the 256-tile rank stride, so rows past
    // tile 256 both anchor on a stored rank prefix AND straddle the boundary
    // mid-row. That is the path the prepass replaced: the scalar kernel rebuilt
    // each tile's plane offset from the nearest prefix, while the prepass
    // carries one anchor forward across the whole row. 90 rows of 768 columns is
    // 270 tiles, enough to reach rank block 1.
    let rows = 90;
    let columns = 3 * 256;
    assert!(rows * columns / 256 > RANK_STRIDE_TILES);
    assert_parity(&cuda, rows, columns, "across rank prefix");
}
