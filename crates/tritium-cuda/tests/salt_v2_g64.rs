#![cfg(feature = "cuda")]

use std::io::Cursor;

use half::f16;
use tritium_cpu::salt_v2::salt_v2_matvec;
use tritium_cuda::CudaBackend;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile, SaltV2Transform,
    write_salt_v2_package,
};

fn g64_tensor() -> SaltV2Tensor {
    let tiles = (0..9)
        .map(|tile_index| {
            let trits = (0..256)
                .map(|index| match (index + tile_index) % 4 {
                    0 => 0,
                    1 | 2 => 1,
                    _ => -1,
                })
                .collect();
            let scales = (0..4)
                .map(|group| f16::from_f32(0.125 + (tile_index * 4 + group) as f32 / 64.0))
                .collect();
            SaltV2Tile::new(vec![
                SaltV2Plane::new_with_scale_group_size(trits, scales, 64).unwrap(),
            ])
            .unwrap()
        })
        .collect();
    SaltV2Tensor::new_with_layout("g64.weight", vec![4, 576], SaltV2Transform::None, 64, tiles)
        .unwrap()
}

#[test]
fn g64_semantic_and_seek_uploads_match_cpu_without_dense_shadow() {
    let cuda = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("skipping G64 CUDA parity: no device ({error})");
            return;
        }
    };
    let tensor = g64_tensor();
    let activation = (0..576)
        .map(|index| (index as f32 - 283.0) / 71.0)
        .collect::<Vec<_>>();
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let expected = salt_v2_matvec(&package, 0, &activation).unwrap();

        let semantic = cuda.upload_salt_v2(&tensor, codec).unwrap();
        let semantic_output = cuda
            .salt_v2_forward_exact(&semantic, &activation, 1)
            .unwrap();
        assert_eq!(semantic_output.output, expected.output, "{codec:?}");
        assert_eq!(semantic_output.receipt.dense_weight_bytes(), 0);

        let encoded = write_salt_v2_package(&package).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        let streamed = cuda
            .upload_salt_v2_from_reader(&mut reader, tensor.name())
            .unwrap();
        let streamed_output = cuda
            .salt_v2_forward_exact(&streamed, &activation, 1)
            .unwrap();
        assert_eq!(streamed_output.output, expected.output, "{codec:?}");
        assert_eq!(streamed_output.receipt.dense_weight_bytes(), 0);

        let mut gathered = vec![0.0; 2 * 576];
        cuda.salt_v2_gather_rows(&streamed, &[3, 1], &mut gathered)
            .unwrap();
        for (destination, row) in [3_usize, 1].into_iter().enumerate() {
            for column in 0..576 {
                let weight =
                    tritium_cpu::salt_v2::salt_v2_coefficient(&tensor, row * 576 + column).unwrap();
                assert_eq!(gathered[destination * 576 + column], weight, "{codec:?}");
            }
        }
    }
}
