//! Actual legacy-row migration into the proposed compact allocation.
use super::*;

#[test]
fn variable_and_pruned_rows_use_compact_counts_without_invented_planes() {
    let mut source = rows(257, 3);
    source[0].planes.clear();
    source[1].planes.truncate(1);
    let request = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout(257, 3, 128, PlaneCodec::D2)
    };
    let tensor = import_salt_rows(&source, request, limits())
        .expect("actual variable/pruned legacy rows must preserve compact allocation");
    assert_eq!(tensor.tile_counts(), Some(&[0, 0, 1, 1][..]));
    assert_eq!(tensor.trits().len(), 257);
    assert_eq!(tensor.scales().len(), 3);
}

fn check_compact(source: &[SaltRow], tensor: &AdditiveTensor) {
    let width = source[0].k;
    let counts = tensor.tile_counts().unwrap();
    let index = tritium_core::OwnedPerTileIndex::new(tensor.layout(), counts.to_vec()).unwrap();
    let group = usize::from(tensor.layout().group);
    let expected_counts: Vec<_> = source
        .iter()
        .flat_map(|row| core::iter::repeat_n(row.planes.len() as u8, width.div_ceil(QK_K)))
        .collect();
    assert_eq!(counts, expected_counts);
    for p in 0..usize::from(tensor.plane_count()) {
        for (r, row) in source.iter().enumerate() {
            for tile in index.as_index().tiles_in_row(p, r).unwrap() {
                let columns = tile.columns();
                let block =
                    &row.planes[p][columns.start / QK_K * TQ2_0_BLOCK_BYTES..][..TQ2_0_BLOCK_BYTES];
                let mut digits = [Trit::ZERO; QK_K];
                let mut scale = f16::ZERO;
                unpack_tq2_0_block(block, &mut digits, &mut scale).unwrap();
                assert_eq!(
                    &tensor.trits()[tile.trit_offset()..tile.trit_offset() + columns.len()],
                    &digits[..columns.len()]
                );
                for unit in 0..columns.len().div_ceil(group) {
                    assert_eq!(
                        f16::from_f32(tensor.scales()[tile.scale_offset() + unit]).to_bits(),
                        scale.to_bits()
                    );
                }
            }
        }
    }
    let original = salt_rows_to_dense(source).unwrap();
    for r in 0..source.len() {
        let mut actual = vec![0.; width];
        tensor.view().dequant_row_into(r, &mut actual).unwrap();
        let mut expected = original[r * width..(r + 1) * width].to_vec();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        apply_inverse_basis(&mut actual, tensor.layout().basis).unwrap();
        apply_inverse_basis(&mut expected, tensor.layout().basis).unwrap();
        assert_eq!(actual, expected);
    }
    assert_eq!(
        AdditiveTensor::decode_per_tile_payload(
            tensor.layout(),
            counts,
            &tensor.encode_payload().unwrap()
        )
        .unwrap(),
        *tensor
    );
}

#[test]
fn real_bundle_readers_migrate_variable_and_zero_rows_with_exact_scale_bits() {
    let mut cases = 0;
    for width in [1, 127, 128, 129, 255, 256, 257, 512] {
        for plane_counts in [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 3],
            [2, 2],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 3],
        ] {
            let mut source = rows(width, 3);
            for (row, n) in source.iter_mut().zip(plane_counts) {
                row.planes.truncate(n);
            }
            let original = source.clone();
            let refs = [("weight", source.as_slice())];
            let dense = write_salt_bundle(&refs).unwrap();
            let progressive = write_progressive_salt_bundle(&refs, 1.0).unwrap();
            let mut bundles = vec![
                read_salt_bundle(&dense).unwrap(),
                read_salt_bundle(&progressive).unwrap(),
            ];
            // TSLJ's frozen representation requires positive, uniform T. Do not
            // invent planes to make a variable/pruned input fit that container.
            let joint = tritium_format::salt_joint_bundle::write_joint_salt_bundle(&refs, None);
            if plane_counts[0] > 0 && plane_counts[0] == plane_counts[1] {
                bundles.push(
                    tritium_format::salt_joint_bundle::read_any_salt_bundle(&joint.unwrap())
                        .unwrap(),
                );
            } else if plane_counts[0] == 0 {
                assert!(matches!(
                    joint,
                    Err(tritium_format::salt_joint_bundle::JointBundleError::Code(
                        tritium_format::salt_joint_code::JointCodeError::UnsupportedPlanes(0)
                    ))
                ));
            } else {
                assert!(matches!(
                    joint,
                    Err(tritium_format::salt_joint_bundle::JointBundleError::MixedPlanes { .. })
                ));
            }
            for recovered in bundles {
                assert_eq!(recovered[0].salt_rows, source);
                for group in [32, 128] {
                    for codec in [PlaneCodec::D2, PlaneCodec::B3] {
                        let request = AdditiveLayout {
                            allocation: PlaneAllocation::PerTile,
                            ..layout(width, 3, group, codec)
                        };
                        let tensor =
                            import_salt_rows(&recovered[0].salt_rows, request, limits()).unwrap();
                        check_compact(&source, &tensor);
                        if width.is_multiple_of(256) {
                            let rotated = import_salt_rows(
                                &source,
                                AdditiveLayout {
                                    basis: Basis::Hadamard { block: 256 },
                                    ..request
                                },
                                limits(),
                            )
                            .unwrap();
                            check_compact(&source, &rotated);
                        }
                        cases += 1;
                    }
                }
            }
            assert_eq!(source, original);
        }
    }
    // 8 widths * (10 dense + 10 progressive + 3 joint profiles) * 2 groups * 2 codecs.
    assert_eq!(cases, 736);
}

#[test]
fn compact_import_limits_include_counts_and_index_and_reject_one_byte_short() {
    let mut source = rows(257, 3);
    source[0].planes.truncate(1);
    let request = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout(257, 3, 128, PlaneCodec::D2)
    };
    let counts = [1, 1, 3, 3];
    let plan = tritium_core::PerTilePlan::new(request, &counts).unwrap();
    let exact = SaltTensorImportLimits {
        max_source_bytes: 528,
        max_trits: plan.trit_count() as u64,
        max_scales: plan.scale_count() as u64,
        max_owned_bytes: (plan.trit_count()
            + 4 * plan.scale_count()
            + plan.buffer_len_bytes().unwrap()) as u64,
    };
    check_compact(&source, &import_salt_rows(&source, request, exact).unwrap());
    for bounded in [
        SaltTensorImportLimits {
            max_source_bytes: exact.max_source_bytes - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_trits: exact.max_trits - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_scales: exact.max_scales - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_owned_bytes: exact.max_owned_bytes - 1,
            ..exact
        },
    ] {
        assert!(matches!(
            import_salt_rows(&source, request, bounded),
            Err(Error::Limit { .. })
        ));
    }
    let before = source.clone();
    let uniform = AdditiveLayout {
        allocation: PlaneAllocation::Uniform,
        ..request
    };
    assert_eq!(
        import_salt_rows(&source, uniform, limits()),
        Err(Error::Geometry)
    );
    assert_eq!(source, before);
    for row in &mut source {
        row.planes.clear();
    }
    let exact = SaltTensorImportLimits {
        max_source_bytes: 0,
        max_trits: 0,
        max_scales: 0,
        max_owned_bytes: 4,
    };
    let tensor = import_salt_rows(&source, request, exact).unwrap();
    check_compact(&source, &tensor);
    assert!(tensor.trits().is_empty() && tensor.scales().is_empty());
    assert!(matches!(
        import_salt_rows(
            &source,
            request,
            SaltTensorImportLimits {
                max_owned_bytes: 3,
                ..exact
            }
        ),
        Err(Error::Limit { .. })
    ));
}

#[test]
fn compact_import_rejects_malformed_legacy_blocks_without_input_changes() {
    let request = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout(257, 3, 128, PlaneCodec::D2)
    };
    for kind in ["geometry", "length", "scale", "code", "padding"] {
        let mut source = rows(257, 3);
        source[0].planes.clear();
        match kind {
            "geometry" => source[1].k = 256,
            "length" => {
                source[1].planes[0].pop();
            }
            "scale" => source[1].planes[0][64..66].copy_from_slice(&0x8000_u16.to_le_bytes()),
            "code" => source[1].planes[0][0] |= 3,
            _ => {
                let mut trits = [Trit::ZERO; QK_K];
                trits[1] = Trit::POS;
                pack_tq2_0_row(&trits, &[f16::ONE], &mut source[1].planes[0][66..]).unwrap();
            }
        }
        let before = source.clone();
        assert!(
            import_salt_rows(&source, request, limits()).is_err(),
            "{kind}"
        );
        assert_eq!(source, before);
    }
}
