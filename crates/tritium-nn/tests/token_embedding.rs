use half::f16;
use tritium_core::Trit;
use tritium_format::{
    PackedSaltRow, PlaneRepr, QK_K, SaltBundleIndex, SaltRow, SparsePlane, TQ2_0_BLOCK_BYTES,
    num_blocks, pack_tq2_0_row, salt_rows_to_dense, write_progressive_salt_bundle,
};
use tritium_nn::{DenseLinear, NnError, SaltLinear, TokenEmbedding};

fn plane(k: usize, scale: f32, mut trit: impl FnMut(usize) -> i8) -> Vec<u8> {
    let trits = (0..k)
        .map(|index| Trit::from_i8(trit(index)).unwrap())
        .collect::<Vec<_>>();
    let scales = vec![f16::from_f32(scale); num_blocks(k)];
    let mut packed = vec![0; num_blocks(k) * TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(&trits, &scales, &mut packed).unwrap();
    packed
}

fn dot_rows_exact(matrix: &[f32], rows: usize, cols: usize, act: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += act[col] * matrix[row * cols + col];
            }
            acc
        })
        .collect()
}

#[test]
fn packed_embedding_drives_gather_and_exact_tied_head_without_dense_copy() {
    let cols = QK_K + 7;
    let rows = (0..3)
        .map(|row| SaltRow {
            k: cols,
            planes: vec![
                plane(cols, 0.25 + row as f32 * 0.03125, |col| {
                    ((col + row) % 3) as i8 - 1
                }),
                plane(cols, 0.0625, |col| {
                    if (col + 17 * row).is_multiple_of(64) {
                        if col.is_multiple_of(128) { 1 } else { -1 }
                    } else {
                        0
                    }
                }),
            ],
        })
        .chain([SaltRow {
            k: cols,
            planes: Vec::new(),
        }])
        .collect::<Vec<_>>();
    let vocab = rows.len();
    let dense = salt_rows_to_dense(&rows).unwrap();
    let bundle = write_progressive_salt_bundle(&[("embed", &rows)], 0.10).unwrap();
    let packed = SaltBundleIndex::new(&bundle)
        .unwrap()
        .tensor("embed")
        .unwrap()
        .decode_packed()
        .unwrap();
    let packed_rows = packed.salt_rows;
    let embedding = TokenEmbedding::from_packed_salt(packed_rows.clone(), vocab, cols).unwrap();

    assert!(embedding.as_dense().is_none());
    assert!(embedding.sparse_plane_count() > 0);
    assert!(embedding.resident_bytes() < dense.len() * size_of::<f32>());

    let tokens = [2, 0, 2, 3];
    let mut gathered = vec![0.0; tokens.len() * cols];
    embedding.gather(&tokens, &mut gathered).unwrap();
    let expected_gather = tokens
        .iter()
        .flat_map(|&token| {
            dense[token as usize * cols..(token as usize + 1) * cols]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        gathered
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_gather
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let act = (0..cols)
        .map(|col| ((col * 17 % 29) as f32 - 14.0) / 32.0)
        .collect::<Vec<_>>();
    let expected_logits = dot_rows_exact(&dense, vocab, cols, &act);
    let mut logits = vec![0.0; vocab];
    embedding.unembed_exact(&act, &mut logits).unwrap();
    assert_eq!(
        logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let dense_linear = DenseLinear::new(dense, vocab, cols).unwrap();
    let salt_linear = SaltLinear::from_packed_rows(packed_rows, vocab, cols).unwrap();
    assert!(salt_linear.sparse_plane_count() > 0);
    let mut expected_a8 = vec![0.0; vocab];
    let mut packed_a8 = vec![0.0; vocab];
    dense_linear.forward(&act, 1, &mut expected_a8).unwrap();
    salt_linear.forward(&act, 1, &mut packed_a8).unwrap();
    assert_eq!(packed_a8, expected_a8);

    assert!(matches!(
        embedding.gather(&[vocab as u32], &mut vec![0.0; cols]),
        Err(NnError::MissingTensor(_))
    ));
    assert!(matches!(
        embedding.unembed_exact(&act[..cols - 1], &mut logits),
        Err(NnError::Shape { .. })
    ));
}

#[test]
fn nonfinite_salt_scales_are_rejected() {
    let cols = QK_K + 1;
    let base = plane(cols, 0.25, |_| 0);
    let sparse = SparsePlane {
        k: cols,
        scales: vec![f16::INFINITY, f16::NAN],
        idx: vec![0, QK_K as u32],
        sign: vec![1, -1],
    };
    let row = PackedSaltRow::new(
        cols,
        vec![PlaneRepr::Dense(base), PlaneRepr::Sparse(sparse)],
    )
    .unwrap();
    assert!(matches!(
        TokenEmbedding::from_packed_salt(vec![row], 1, cols),
        Err(NnError::Backend(_))
    ));

    let dense_nonfinite = SaltRow {
        k: cols,
        planes: vec![plane(cols, f32::INFINITY, |_| 0)],
    };
    assert!(matches!(
        SaltLinear::new(vec![dense_nonfinite], 1, cols),
        Err(NnError::Backend(_))
    ));
}

#[test]
fn dense_embedding_rejects_out_of_range_tokens_before_offset_arithmetic() {
    let embedding = TokenEmbedding::from_dense(vec![0.0; 8], 2, 4).unwrap();
    let mut gathered = vec![0.0; 4];
    assert!(matches!(
        embedding.gather(&[u32::MAX], &mut gathered),
        Err(NnError::MissingTensor(_))
    ));
}
