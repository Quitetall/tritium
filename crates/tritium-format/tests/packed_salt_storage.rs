use half::f16;
use tritium_core::Trit;
use tritium_format::{
    FormatError, PackedSaltRow, PlaneRepr, QK_K, SaltBundleIndex, SaltRow, TQ2_0_BLOCK_BYTES,
    num_blocks, pack_tq2_0_row, salt_rows_to_dense, sparse_from_tq2_0,
    write_progressive_salt_bundle,
};

fn plane(k: usize, scale: f32, mut trit: impl FnMut(usize) -> i8) -> Vec<u8> {
    let trits = (0..k)
        .map(|index| Trit::from_i8(trit(index)).unwrap())
        .collect::<Vec<_>>();
    let scales = vec![f16::from_f32(scale); num_blocks(k)];
    let mut packed = vec![0; num_blocks(k) * TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(&trits, &scales, &mut packed).unwrap();
    packed
}

#[test]
fn packed_decode_preserves_sparse_planes_and_dense_semantics() {
    let k = QK_K + 13;
    let rows = vec![
        SaltRow {
            k,
            planes: vec![
                plane(k, 0.25, |index| (index % 3) as i8 - 1),
                plane(k, 0.125, |index| {
                    if index % 64 == 0 {
                        if index.is_multiple_of(128) { 1 } else { -1 }
                    } else {
                        0
                    }
                }),
            ],
        },
        SaltRow {
            k,
            planes: Vec::new(),
        },
    ];
    let bundle = write_progressive_salt_bundle(&[("embed", &rows)], 0.10).unwrap();
    let tensor = SaltBundleIndex::new(&bundle)
        .unwrap()
        .tensor("embed")
        .unwrap()
        .decode_packed()
        .unwrap();

    assert!(matches!(
        tensor.salt_rows[0].planes()[1],
        PlaneRepr::Sparse(_)
    ));
    assert_eq!(tensor.salt_rows[1].plane_count(), 0);
    let recovered = tensor
        .salt_rows
        .iter()
        .map(|row| row.to_dense().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(recovered, rows);
    assert_eq!(
        salt_rows_to_dense(&recovered).unwrap(),
        salt_rows_to_dense(&rows).unwrap()
    );
    assert!(
        tensor.salt_rows[0].resident_payload_bytes()
            < rows[0].planes.iter().map(Vec::len).sum::<usize>()
    );

    let sparse = sparse_from_tq2_0(&rows[0].planes[1], k).unwrap();
    assert!(matches!(
        PackedSaltRow::new(k, vec![PlaneRepr::Sparse(sparse)]),
        Err(FormatError::SaltSparseBasePlane)
    ));
}
