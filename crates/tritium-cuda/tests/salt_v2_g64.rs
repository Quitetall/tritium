#![cfg(feature = "cuda")]

use std::io::Cursor;

use half::f16;
use tritium_cpu::salt_v2::salt_v2_matvec;
use tritium_cuda::{CudaBackend, SaltV2ForwardMode};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
    write_salt_v2_package,
};

fn scale_geometry_tensor(group_size: usize, columns: usize) -> SaltV2Tensor {
    let coefficient_count = 4 * columns;
    let tiles = (0..coefficient_count.div_ceil(256))
        .map(|tile_index| {
            let tile_len = (coefficient_count - tile_index * 256).min(256);
            let trits = (0..tile_len)
                .map(|index| match (index + tile_index) % 4 {
                    0 => 0,
                    1 | 2 => 1,
                    _ => -1,
                })
                .collect();
            let scales = (0..tile_len.div_ceil(group_size))
                .map(|group| f16::from_f32(0.125 + (tile_index * 4 + group) as f32 / 64.0))
                .collect();
            SaltV2Tile::new(vec![
                SaltV2Plane::new_with_scale_group_size(trits, scales, group_size).unwrap(),
            ])
            .unwrap()
        })
        .collect();
    SaltV2Tensor::new_with_layout(
        format!("g{group_size}.weight"),
        vec![4, columns as u64],
        SaltV2Transform::None,
        group_size,
        tiles,
    )
    .unwrap()
}

#[test]
fn g64_g128_g256_semantic_and_seek_uploads_match_cpu_without_dense_shadow() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping SALT V2 scale geometry CUDA parity: no device ({error})");
            return;
        }
    };
    for (group_size, columns) in [(64, 576), (128, 256), (256, 512)] {
        let tensor = scale_geometry_tensor(group_size, columns);
        let activation = (0..columns)
            .map(|index| (index as f32 - 283.0) / 71.0)
            .collect::<Vec<_>>();
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
            let expected = salt_v2_matvec(&package, 0, &activation).unwrap();

            let semantic = cuda.upload_salt_v2(&tensor, codec).unwrap();
            let exact = cuda
                .salt_v2_forward_exact(&semantic, &activation, 1)
                .unwrap();
            assert_eq!(
                exact.output, expected.output,
                "{codec:?} G{group_size} exact"
            );
            assert_eq!(exact.receipt.dense_weight_bytes(), 0);

            let fast = cuda
                .salt_v2_forward_fast(&semantic, &activation, 1)
                .unwrap();
            let expected_peak = expected
                .output
                .iter()
                .fold(0.0_f32, |peak, value| peak.max(value.abs()))
                .max(f32::MIN_POSITIVE);
            for (got, want) in fast.output.iter().zip(&expected.output) {
                assert!((got - want).abs() / expected_peak <= 1e-5);
            }
            if columns.is_multiple_of(256) {
                assert_eq!(fast.receipt.mode(), SaltV2ForwardMode::FastWarpReduce);
            } else {
                assert_eq!(fast.receipt.mode(), SaltV2ForwardMode::FastAliasesExact);
            }
            assert_eq!(fast.receipt.dense_weight_bytes(), 0);

            let encoded = write_salt_v2_package(&package).unwrap();
            let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
            let streamed = cuda
                .upload_salt_v2_from_reader(&mut reader, tensor.name())
                .unwrap();
            let streamed_output = cuda
                .salt_v2_forward_exact(&streamed, &activation, 1)
                .unwrap();
            assert_eq!(
                streamed_output.output, expected.output,
                "{codec:?} G{group_size} streamed"
            );
            assert_eq!(streamed_output.receipt.dense_weight_bytes(), 0);

            let mut gathered = vec![0.0; 2 * columns];
            cuda.salt_v2_gather_rows(&streamed, &[3, 1], &mut gathered)
                .unwrap();
            for (destination, row) in [3_usize, 1].into_iter().enumerate() {
                for column in 0..columns {
                    let weight =
                        tritium_cpu::salt_v2::salt_v2_coefficient(&tensor, row * columns + column)
                            .unwrap();
                    assert_eq!(
                        gathered[destination * columns + column],
                        weight,
                        "{codec:?} G{group_size}"
                    );
                }
            }
        }
    }
}
