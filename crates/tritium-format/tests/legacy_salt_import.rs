use half::f16;
use tritium_core::{Trit, apply_inverse_basis};
use tritium_format::{
    AdditiveTensor, QK_K, SaltRow, SaltTensorImportError as Error, SaltTensorImportLimits,
    TQ2_0_BLOCK_BYTES, import_salt_rows, pack_tq2_0_row, read_salt_bundle, salt_rows_to_dense,
    unpack_tq2_0_block, write_progressive_salt_bundle, write_salt_bundle,
};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};

#[path = "legacy_salt_import/compact_rows.rs"]
mod compact_rows;

fn layout(width: usize, planes: u8, group: u16, codec: PlaneCodec) -> AdditiveLayout {
    AdditiveLayout {
        rows: 2,
        cols: width as u64,
        tile: 256,
        group,
        max_planes: planes,
        allocation: PlaneAllocation::Uniform,
        codec,
        transport: Transport::Raw,
        basis: Basis::Identity,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F16,
        },
    }
}
fn limits() -> SaltTensorImportLimits {
    SaltTensorImportLimits {
        max_source_bytes: 1_000_000,
        max_trits: 1_000_000,
        max_scales: 1_000_000,
        max_owned_bytes: 1_000_000,
    }
}
fn rows(width: usize, planes: usize) -> Vec<SaltRow> {
    (0..2)
        .map(|row| SaltRow {
            k: width,
            planes: (0..planes)
                .map(|plane| {
                    // Sparse residuals exercise the real progressive reader's expansion.
                    let trits: Vec<_> = (0..width)
                        .map(|i| {
                            if plane > 0 && !i.is_multiple_of(37) {
                                Trit::ZERO
                            } else {
                                Trit::from_i8(((i + plane + row) % 3) as i8 - 1).unwrap()
                            }
                        })
                        .collect();
                    let scales: Vec<_> = (0..width.div_ceil(QK_K))
                        .map(|block| {
                            [
                                f16::ZERO,
                                f16::from_f32(0.3334),
                                f16::from_bits(1),
                                f16::MAX,
                            ][(row + plane + block) % 4]
                        })
                        .collect();
                    let mut bytes = vec![0; width.div_ceil(QK_K) * TQ2_0_BLOCK_BYTES];
                    pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                    bytes
                })
                .collect(),
        })
        .collect()
}
fn check(rows: &[SaltRow], tensor: &AdditiveTensor) {
    let width = rows[0].k;
    let group = usize::from(tensor.layout().group);
    let units = rows.len() * width.div_ceil(group);
    for (r, row) in rows.iter().enumerate() {
        for (p, plane) in row.planes.iter().enumerate() {
            for (b, block) in plane.as_chunks::<TQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
                let mut digits = [Trit::ZERO; QK_K];
                let mut scale = f16::ZERO;
                unpack_tq2_0_block(block, &mut digits, &mut scale).unwrap();
                let start = b * QK_K;
                let count = (width - start).min(QK_K);
                let offset = p * rows.len() * width + r * width + start;
                assert_eq!(&tensor.trits()[offset..offset + count], &digits[..count]);
                for col in (start..start + count).step_by(group) {
                    let actual =
                        tensor.scales()[p * units + r * width.div_ceil(group) + col / group];
                    assert_eq!(f16::from_f32(actual).to_bits(), scale.to_bits());
                }
            }
        }
    }
    let original = salt_rows_to_dense(rows).unwrap();
    for r in 0..rows.len() {
        let mut decoded = vec![0.0; width];
        tensor.view().dequant_row_into(r, &mut decoded).unwrap();
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            original[r * width..(r + 1) * width]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        // Both paths use the real transform, not a second Hadamard implementation.
        let mut expected = original[r * width..(r + 1) * width].to_vec();
        apply_inverse_basis(&mut expected, tensor.layout().basis).unwrap();
        apply_inverse_basis(&mut decoded, tensor.layout().basis).unwrap();
        assert_eq!(decoded, expected);
    }
    let encoded = tensor.encode_payload().unwrap();
    assert_eq!(
        *tensor,
        AdditiveTensor::decode_payload(tensor.layout(), tensor.plane_count(), &encoded).unwrap()
    );
}

#[test]
fn existing_dense_progressive_and_joint_readers_import_exact_trits_and_scales() {
    for width in [1, 127, 128, 129, 255, 256, 257, 512] {
        for planes in 1..=3 {
            let source = rows(width, planes);
            let preserved = source.clone();
            let refs = [("weight", source.as_slice())];
            let dense = write_salt_bundle(&refs).unwrap();
            let progressive = write_progressive_salt_bundle(&refs, 1.0).unwrap();
            let joint =
                tritium_format::salt_joint_bundle::write_joint_salt_bundle(&refs, None).unwrap();
            let recovered = [
                read_salt_bundle(&dense).unwrap(),
                read_salt_bundle(&progressive).unwrap(),
                tritium_format::salt_joint_bundle::read_any_salt_bundle(&joint).unwrap(),
            ];
            for tensors in recovered {
                assert_eq!(tensors[0].salt_rows, source);
                for group in [32, 128] {
                    for codec in [PlaneCodec::D2, PlaneCodec::B3] {
                        let request = layout(width, planes as u8, group, codec);
                        let tensor =
                            import_salt_rows(&tensors[0].salt_rows, request, limits()).unwrap();
                        check(&source, &tensor);
                        if width.is_multiple_of(256) {
                            let rotated = AdditiveLayout {
                                basis: Basis::Hadamard { block: 256 },
                                ..request
                            };
                            let tensor = import_salt_rows(&source, rotated, limits()).unwrap();
                            check(&source, &tensor);
                            assert_eq!(tensor.layout().basis, rotated.basis);
                        }
                    }
                }
            }
            assert_eq!(source, preserved);
        }
    }
}

#[test]
fn caps_and_geometry_reject_without_fallback_or_input_mutation() {
    let source = rows(257, 3);
    let request = layout(257, 3, 128, PlaneCodec::D2);
    let exact = SaltTensorImportLimits {
        max_source_bytes: 792,
        max_trits: 1542,
        max_scales: 18,
        max_owned_bytes: 1614,
    };
    check(&source, &import_salt_rows(&source, request, exact).unwrap());
    for (name, bounded) in [
        (
            "source bytes",
            SaltTensorImportLimits {
                max_source_bytes: 791,
                ..exact
            },
        ),
        (
            "output trits",
            SaltTensorImportLimits {
                max_trits: 1541,
                ..exact
            },
        ),
        (
            "output scales",
            SaltTensorImportLimits {
                max_scales: 17,
                ..exact
            },
        ),
        (
            "owned payload bytes",
            SaltTensorImportLimits {
                max_owned_bytes: 1613,
                ..exact
            },
        ),
    ] {
        assert!(matches!(import_salt_rows(&source, request, bounded),
            Err(Error::Limit { resource, .. }) if resource == name));
    }
    assert_eq!(import_salt_rows(&[], request, exact), Err(Error::Geometry));
    let preserved = source.clone();
    let mut wrong = source.clone();
    wrong[0].k = 256;
    assert_eq!(
        import_salt_rows(&wrong, request, exact),
        Err(Error::Geometry)
    );
    assert_eq!(
        import_salt_rows(&source[..1], request, exact),
        Err(Error::Geometry)
    );
    for planes in [0, 2, 4] {
        let mut wrong = source.clone();
        wrong[0].planes.resize(planes, source[0].planes[0].clone());
        assert_eq!(
            import_salt_rows(&wrong, request, exact),
            Err(Error::Geometry)
        );
    }
    let mut wrong = source.clone();
    wrong[1].planes[0].pop();
    assert_eq!(
        import_salt_rows(&wrong, request, exact),
        Err(Error::Geometry)
    );
    let overflow = AdditiveLayout {
        cols: u64::MAX,
        ..request
    };
    assert_eq!(
        import_salt_rows(&source, overflow, limits()),
        Err(Error::Overflow)
    );
    assert_eq!(source, preserved);
}

#[test]
fn malformed_codes_scales_padding_and_semantic_reinterpretation_reject() {
    let source = rows(257, 3);
    let request = layout(257, 3, 128, PlaneCodec::D2);
    for bits in [0x8000, 0xbc00, 0x7c00, 0x7e00] {
        let mut wrong = source.clone();
        wrong[0].planes[0][64..66].copy_from_slice(&u16::to_le_bytes(bits));
        assert_eq!(
            import_salt_rows(&wrong, request, limits()),
            Err(Error::Scale)
        );
    }
    let mut wrong = source.clone();
    wrong[0].planes[0][0] |= 3;
    assert!(matches!(
        import_salt_rows(&wrong, request, limits()),
        Err(Error::Format(_))
    ));
    let mut wrong = source.clone();
    // Last block has one logical coefficient. Repack a valid nonzero padding trit.
    let mut digits = [Trit::ZERO; QK_K];
    digits[1] = Trit::POS;
    pack_tq2_0_row(&digits, &[f16::ONE], &mut wrong[0].planes[0][66..]).unwrap();
    assert_eq!(
        import_salt_rows(&wrong, request, limits()),
        Err(Error::Padding)
    );
    for wrong in [
        AdditiveLayout {
            law: ScaleLaw {
                relation: PlaneRelation::Tied { num: 1, den: 3 },
                ..request.law
            },
            ..request
        },
        AdditiveLayout {
            law: ScaleLaw {
                precision: ScalePrecision::F32,
                ..request.law
            },
            ..request
        },
        AdditiveLayout {
            group: 256,
            ..request
        },
        AdditiveLayout {
            codec: PlaneCodec::S34,
            ..request
        },
        AdditiveLayout {
            transport: Transport::JointHuffman { block: 256 },
            ..request
        },
    ] {
        assert!(matches!(
            import_salt_rows(&source, wrong, limits()),
            Err(Error::Unsupported(_))
        ));
    }
    let signed = AdditiveLayout {
        cols: 512,
        basis: Basis::SignedRht {
            block: 256,
            seed: 1,
            domain: 2,
        },
        ..request
    };
    assert!(matches!(
        import_salt_rows(&rows(512, 3), signed, limits()),
        Err(Error::Unsupported(_))
    ));
    let ragged = AdditiveLayout {
        basis: Basis::Hadamard { block: 256 },
        ..request
    };
    assert!(matches!(
        import_salt_rows(&source, ragged, limits()),
        Err(Error::Layout(_))
    ));
}
