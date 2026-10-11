use tritium_core::{AdditiveView, OwnedPerTileIndex, PerTilePlan, Trit};
use tritium_format::{AdditiveTensor, AdditiveTensorError};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};

fn layout(law: tritium_schema::AdmittedLaw, cols: u64, codec: PlaneCodec) -> AdditiveLayout {
    AdditiveLayout {
        rows: 2,
        cols,
        tile: 256,
        group: 128,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::PerTile,
        codec,
        law: law.law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    }
}

fn tensor(layout: AdditiveLayout, counts: &[u8]) -> AdditiveTensor {
    let plan = PerTilePlan::new(layout, counts).unwrap();
    let trits = (0..plan.trit_count())
        .map(|i| Trit::from_i8((i % 3) as i8 - 1).unwrap())
        .collect();
    AdditiveTensor::new_per_tile(
        layout,
        counts.to_vec(),
        trits,
        vec![0.5; plan.scale_count()],
    )
    .unwrap()
}

#[test]
fn sealed_preflight_and_owned_index_share_exact_geometry() {
    let layout = layout(ADMITTED_LAWS[0], 257, PlaneCodec::D2);
    let counts = [1, 3, 2, 0];
    let plan = PerTilePlan::new(layout, &counts).unwrap();
    assert_eq!(plan.index_len(), 6);
    assert_eq!(plan.trit_count(), 771);
    assert_eq!(plan.scale_count(), 9);
    assert_eq!(
        (0..plan.plane_count())
            .map(|p| plan.plane_trit_count(p).unwrap())
            .collect::<Vec<_>>(),
        [513, 257, 1]
    );
    assert!(plan.plane_trit_count(3).is_err());
    let owned = OwnedPerTileIndex::new(layout, counts.to_vec()).unwrap();
    let trits = vec![Trit::POS; plan.trit_count()];
    let scales = vec![0.5; plan.scale_count()];
    let view = AdditiveView::new_per_tile(owned.as_index(), &trits, &scales).unwrap();
    assert_eq!(view.tile_counts(), Some(counts.as_slice()));
    assert_eq!(
        owned.storage_bytes(),
        4 + 6 * core::mem::size_of::<tritium_core::PerTileRowSpan>()
    );
}

#[test]
fn d2_b3_compact_roundtrip_all_laws_groups_and_ragged_rows() {
    for &law in ADMITTED_LAWS {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            for group in [32, 128] {
                for cols in [1, 255, 256, 257, 512] {
                    let mut layout = layout(law, cols, codec);
                    layout.group = group;
                    let n = 2 * (cols as usize).div_ceil(256);
                    let counts = (0..n)
                        .map(|i| (i % (usize::from(law.max_planes) + 1)) as u8)
                        .collect::<Vec<_>>();
                    let owned = tensor(layout, &counts);
                    let bytes = owned.encode_payload().unwrap();
                    let decoded =
                        AdditiveTensor::decode_per_tile_payload(layout, &counts, &bytes).unwrap();
                    assert_eq!(decoded, owned);
                    assert_eq!(decoded.view().tile_counts(), Some(counts.as_slice()));
                    assert_eq!(decoded.encode_payload().unwrap(), bytes);
                    assert_eq!(AdditiveTensor::from_view(decoded.view()).unwrap(), decoded);
                    for row in 0..2 {
                        let mut expected = vec![0.; cols as usize];
                        let mut actual = expected.clone();
                        owned.view().dequant_row_into(row, &mut expected).unwrap();
                        decoded.view().dequant_row_into(row, &mut actual).unwrap();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn equal_counts_preserve_existing_uniform_payload_bytes() {
    for &law in ADMITTED_LAWS {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            let compact = tensor(layout(law, 257, codec), &[law.max_planes; 4]);
            let uniform = AdditiveTensor::new(
                AdditiveLayout {
                    allocation: PlaneAllocation::Uniform,
                    ..compact.layout()
                },
                law.max_planes,
                compact.trits().to_vec(),
                compact.scales().to_vec(),
            )
            .unwrap();
            assert_eq!(
                compact.encode_payload().unwrap(),
                uniform.encode_payload().unwrap()
            );
        }
    }
}

#[test]
fn all_pruned_payload_is_empty_and_retains_declared_counts() {
    let layout = layout(ADMITTED_LAWS[0], 257, PlaneCodec::B3);
    let owned = tensor(layout, &[0; 4]);
    assert_eq!(owned.plane_count(), 0);
    assert!(owned.encode_payload().unwrap().is_empty());
    assert_eq!(
        AdditiveTensor::decode_per_tile_payload(layout, &[0; 4], &[]).unwrap(),
        owned
    );
    assert!(matches!(
        AdditiveTensor::decode_per_tile_payload(layout, &[0; 4], &[0]),
        Err(AdditiveTensorError::PayloadLength {
            expected: 0,
            got: 1
        })
    ));
}

#[test]
fn malformed_counts_lengths_precision_and_scales_are_rejected() {
    let layout = layout(ADMITTED_LAWS[0], 257, PlaneCodec::D2);
    assert!(AdditiveTensor::decode_per_tile_payload(layout, &[1, 1, 1], &[]).is_err());
    assert!(AdditiveTensor::decode_per_tile_payload(layout, &[1, 4, 0, 0], &[]).is_err());
    let counts = [1, 0, 0, 0];
    let owned = tensor(layout, &counts);
    let bytes = owned.encode_payload().unwrap();
    assert!(matches!(
        AdditiveTensor::decode_per_tile_payload(layout, &counts, &bytes[..bytes.len() - 1]),
        Err(AdditiveTensorError::PayloadLength { .. })
    ));
    let mut scales = owned.scales().to_vec();
    for bad in [f32::NAN, f32::INFINITY, -1., -0.] {
        scales[0] = bad;
        assert!(
            AdditiveTensor::new_per_tile(
                layout,
                counts.to_vec(),
                owned.trits().to_vec(),
                scales.clone()
            )
            .is_err()
        );
    }
    scales[0] = 0.1;
    assert!(matches!(
        AdditiveTensor::new_per_tile(layout, counts.to_vec(), owned.trits().to_vec(), scales),
        Err(AdditiveTensorError::ScalePrecisionMismatch { .. })
    ));
    assert!(
        AdditiveTensor::new_per_tile(layout, counts.to_vec(), vec![], owned.scales().to_vec())
            .is_err()
    );
    assert!(AdditiveTensor::decode_payload(layout, 1, &bytes).is_err());
}

#[test]
fn compact_codec_checks_reserved_codes_and_terminal_padding() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3] {
        let layout = layout(ADMITTED_LAWS[0], 1, codec);
        let counts = [1, 0];
        let bytes = tensor(layout, &counts).encode_payload().unwrap();
        let mut reserved = bytes.clone();
        reserved[0] = 255;
        assert!(matches!(
            AdditiveTensor::decode_per_tile_payload(layout, &counts, &reserved),
            Err(AdditiveTensorError::Codec(_))
        ));
        let mut padding = bytes;
        padding[0] = 0; // All -1 digits: padding must instead encode zero.
        assert!(matches!(
            AdditiveTensor::decode_per_tile_payload(layout, &counts, &padding),
            Err(AdditiveTensorError::Codec(_))
        ));
    }
}

#[test]
fn structured_codec_preserves_constraints_without_automatic_admission() {
    let mut layout = layout(ADMITTED_LAWS[0], 4, PlaneCodec::S34);
    layout.rows = 1;
    let trits = vec![Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS];
    let owned = AdditiveTensor::new_per_tile(layout, vec![1], trits, vec![0.5]).unwrap();
    let bytes = owned.encode_payload().unwrap();
    assert_eq!(
        AdditiveTensor::decode_per_tile_payload(layout, &[1], &bytes).unwrap(),
        owned
    );
    layout.cols = 3;
    assert!(tensor(layout, &[1]).encode_payload().is_err());
}

#[test]
fn model_semantic_digest_cannot_mislabel_compact_counts_as_uniform() {
    let owned = tensor(layout(ADMITTED_LAWS[0], 257, PlaneCodec::D2), &[1, 3, 2, 0]);
    assert!(tritium_format::trit_model::additive_semantic_digest(&owned).is_err());
}

#[test]
fn compact_planes_have_exact_golden_physical_lengths_and_reject_bad_wire_scales() {
    for (codec, expected) in [(PlaneCodec::D2, 213), (PlaneCodec::B3, 174)] {
        let layout = layout(ADMITTED_LAWS[0], 257, codec);
        let counts = [1, 3, 2, 0];
        let bytes = tensor(layout, &counts).encode_payload().unwrap();
        // Plane lengths 513/257/1 plus nine exact f16 scales; no dense padding.
        assert_eq!(bytes.len(), expected);
        for bits in [0x7c00u16, 0x7e00, 0x8000] {
            let mut bad = bytes.clone();
            let end = bad.len();
            bad[end - 2..].copy_from_slice(&bits.to_le_bytes());
            assert!(AdditiveTensor::decode_per_tile_payload(layout, &counts, &bad).is_err());
        }
    }
}

#[test]
fn owned_roundtrip_preserves_actual_rotated_matmul_and_gather_bits() {
    for basis in [
        Basis::Hadamard { block: 16 },
        Basis::SignedRht {
            block: 16,
            seed: 7,
            domain: 9,
        },
    ] {
        let mut layout = layout(ADMITTED_LAWS[2], 512, PlaneCodec::B3);
        layout.basis = basis;
        let counts = [3, 1, 0, 2];
        let owned = tensor(layout, &counts);
        let decoded = AdditiveTensor::decode_per_tile_payload(
            layout,
            &counts,
            &owned.encode_payload().unwrap(),
        )
        .unwrap();
        let act: Vec<f32> = (0..512).map(|i| (i % 7) as f32 - 3.).collect();
        let mut outputs = Vec::new();
        let mut gathers = Vec::new();
        for tensor in [&owned, &decoded] {
            let view = tensor.view();
            let mut out = [0.; 2];
            let mut scratch = [0.; 512];
            tritium_core::reference_ternary_matmul(
                &act,
                &view.bind_matmul(),
                1,
                &mut scratch,
                &mut out,
            )
            .unwrap();
            outputs.push(out.map(f32::to_bits));
            let mut gathered = vec![0.; 1024];
            tritium_core::reference_embed(&view.bind_gather(), &[1, 0], &mut gathered).unwrap();
            gathers.push(gathered.into_iter().map(f32::to_bits).collect::<Vec<_>>());
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(gathers[0], gathers[1]);
    }
}
