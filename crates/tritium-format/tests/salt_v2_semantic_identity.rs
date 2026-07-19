use std::io::Cursor;

use half::f16;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2PackageError, SaltV2PackageReader, SaltV2Plane,
    SaltV2SemanticTensorStream, SaltV2Tensor, SaltV2Tile, SaltV2Transform, write_salt_v2_package,
};

fn plane(len: usize, phase: usize, scale_bits: u16) -> SaltV2Plane {
    let trits = (0..len)
        .map(|index| match (index + phase) % 4 {
            0 => 0,
            1 => -1,
            2 | 3 => 1,
            _ => unreachable!(),
        })
        .collect();
    let scales = (0..len.div_ceil(128))
        .map(|group| f16::from_bits(scale_bits + group as u16))
        .collect();
    SaltV2Plane::new(trits, scales).unwrap()
}

fn semantic_tensor(name: &str, transform: SaltV2Transform) -> SaltV2Tensor {
    SaltV2Tensor::new_with_transform(
        name,
        vec![65, 4],
        transform,
        vec![
            SaltV2Tile::new(vec![plane(256, 0, 0x3800), plane(256, 1, 0x3a00)]).unwrap(),
            SaltV2Tile::new(vec![plane(4, 2, 0x3c00)]).unwrap(),
        ],
    )
    .unwrap()
}

fn seek_identity(package: &SaltV2Package, name: &str) -> tritium_format::SemanticTensor {
    let encoded = write_salt_v2_package(package).unwrap();
    let reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
    reader.semantic_tensor(name).unwrap()
}

#[test]
fn eager_and_seek_semantics_are_codec_independent() {
    let tensor = semantic_tensor(
        "layer.weight",
        SaltV2Transform::SignedRht {
            seed: 0x1234_5678_9abc_def0,
            domain: 0x5341_4c54_5f52_4854,
        },
    );
    let expected = tensor.semantic_tensor();
    let golden_v1 = [
        0x01, 0x6f, 0xeb, 0x23, 0x12, 0xc5, 0x18, 0x7a, 0x66, 0x00, 0x70, 0xc6, 0xd7, 0x64, 0xeb,
        0xfe, 0x54, 0x80, 0xa1, 0x1f, 0x88, 0xce, 0x0a, 0xf6, 0xcc, 0x91, 0x52, 0xcf, 0x44, 0x11,
        0x26, 0x76,
    ];
    assert_eq!(*expected.content_digest(), golden_v1);

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        let actual = seek_identity(&package, tensor.name());
        assert_eq!(actual, expected, "codec {codec:?}");
        assert_eq!(*actual.content_digest(), golden_v1, "codec {codec:?}");
    }
}

#[test]
fn checked_stream_matches_owned_and_seek_semantics_without_tensor_materialization() {
    let transform = SaltV2Transform::SignedRht {
        seed: 0x1234_5678_9abc_def0,
        domain: 0x5341_4c54_5f52_4854,
    };
    let tensor = semantic_tensor("layer.weight", transform);
    let mut stream =
        SaltV2SemanticTensorStream::new(tensor.name(), tensor.dims().to_vec(), transform).unwrap();
    for tile in tensor.tiles() {
        stream.push_tile(tile.planes()).unwrap();
    }

    let streamed = stream.finish().unwrap();
    assert_eq!(streamed, tensor.semantic_tensor());
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let package = SaltV2Package::new(codec, vec![tensor.clone()]).unwrap();
        assert_eq!(streamed, seek_identity(&package, tensor.name()));
    }
}

#[test]
fn checked_stream_rejects_incomplete_extra_and_misshaped_tiles() {
    let tensor = semantic_tensor("layer.weight", SaltV2Transform::None);
    let stream = SaltV2SemanticTensorStream::new(
        tensor.name(),
        tensor.dims().to_vec(),
        SaltV2Transform::None,
    )
    .unwrap();
    assert_eq!(
        stream.finish(),
        Err(SaltV2PackageError::WrongTileCount {
            expected: 2,
            got: 0,
        })
    );

    let mut stream = SaltV2SemanticTensorStream::new(
        tensor.name(),
        tensor.dims().to_vec(),
        SaltV2Transform::None,
    )
    .unwrap();
    assert_eq!(
        stream.push_tile(&[]),
        Err(SaltV2PackageError::InvalidPlaneCount { got: 0 })
    );
    let short = plane(4, 0, 0x3800);
    assert_eq!(
        stream.push_tile(&[short]),
        Err(SaltV2PackageError::WrongTileLength {
            tile_index: 0,
            expected: 256,
            got: 4,
        })
    );
    for tile in tensor.tiles() {
        stream.push_tile(tile.planes()).unwrap();
    }
    assert_eq!(
        stream.push_tile(tensor.tiles()[1].planes()),
        Err(SaltV2PackageError::WrongTileCount {
            expected: 2,
            got: 3,
        })
    );
}

#[test]
fn package_record_order_does_not_change_tensor_semantics() {
    let left = semantic_tensor("left", SaltV2Transform::None);
    let right = semantic_tensor(
        "right",
        SaltV2Transform::SignedRht {
            seed: 7,
            domain: 11,
        },
    );
    let forward = SaltV2Package::new(SaltV2Codec::B3, vec![left.clone(), right.clone()]).unwrap();
    let reverse = SaltV2Package::new(SaltV2Codec::B3, vec![right, left]).unwrap();

    for name in ["left", "right"] {
        assert_eq!(
            seek_identity(&forward, name),
            seek_identity(&reverse, name),
            "tensor {name}"
        );
    }
}

#[test]
fn semantic_digest_binds_values_scales_transform_geometry_and_plane_structure() {
    fn digest(tensor: &SaltV2Tensor) -> [u8; 32] {
        *tensor.semantic_tensor().content_digest()
    }

    let base = semantic_tensor("layer.weight", SaltV2Transform::None);
    let base_digest = digest(&base);

    let changed_trit = SaltV2Tensor::new(
        "layer.weight",
        vec![65, 4],
        vec![
            SaltV2Tile::new(vec![
                {
                    let mut trits = base.tiles()[0].planes()[0]
                        .trits()
                        .iter()
                        .map(|trit| trit.get())
                        .collect::<Vec<_>>();
                    trits[0] = 1;
                    SaltV2Plane::new(trits, base.tiles()[0].planes()[0].scales().to_vec()).unwrap()
                },
                base.tiles()[0].planes()[1].clone(),
            ])
            .unwrap(),
            base.tiles()[1].clone(),
        ],
    )
    .unwrap();

    let changed_scale = SaltV2Tensor::new(
        "layer.weight",
        vec![65, 4],
        vec![
            SaltV2Tile::new(vec![
                SaltV2Plane::new(
                    base.tiles()[0].planes()[0]
                        .trits()
                        .iter()
                        .map(|trit| trit.get())
                        .collect(),
                    vec![f16::from_bits(0x3801), f16::from_bits(0x3801)],
                )
                .unwrap(),
                base.tiles()[0].planes()[1].clone(),
            ])
            .unwrap(),
            base.tiles()[1].clone(),
        ],
    )
    .unwrap();

    let changed_transform = semantic_tensor(
        "layer.weight",
        SaltV2Transform::SignedRht { seed: 1, domain: 2 },
    );
    let changed_transform_seed = semantic_tensor(
        "layer.weight",
        SaltV2Transform::SignedRht { seed: 2, domain: 2 },
    );
    let changed_transform_domain = semantic_tensor(
        "layer.weight",
        SaltV2Transform::SignedRht { seed: 1, domain: 3 },
    );
    let transform_digest = digest(&changed_transform);
    assert_ne!(digest(&changed_transform_seed), transform_digest);
    assert_ne!(digest(&changed_transform_domain), transform_digest);
    let changed_geometry = SaltV2Tensor::new_with_transform(
        "layer.weight",
        vec![130, 2],
        SaltV2Transform::None,
        base.tiles().to_vec(),
    );

    let fewer_planes = SaltV2Tensor::new(
        "layer.weight",
        vec![65, 4],
        vec![
            SaltV2Tile::new(vec![base.tiles()[0].planes()[0].clone()]).unwrap(),
            base.tiles()[1].clone(),
        ],
    )
    .unwrap();
    let reversed_planes = SaltV2Tensor::new(
        "layer.weight",
        vec![65, 4],
        vec![
            SaltV2Tile::new(vec![
                base.tiles()[0].planes()[1].clone(),
                base.tiles()[0].planes()[0].clone(),
            ])
            .unwrap(),
            base.tiles()[1].clone(),
        ],
    )
    .unwrap();

    for (label, changed) in [
        ("trit", digest(&changed_trit)),
        ("scale", digest(&changed_scale)),
        ("transform", digest(&changed_transform)),
        ("transform seed", digest(&changed_transform_seed)),
        ("transform domain", digest(&changed_transform_domain)),
        ("geometry", digest(&changed_geometry.unwrap())),
        ("plane count", digest(&fewer_planes)),
        ("plane order", digest(&reversed_planes)),
    ] {
        assert_ne!(changed, base_digest, "{label} must affect semantic digest");
    }
}
