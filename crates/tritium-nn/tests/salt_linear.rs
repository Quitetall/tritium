use half::f16;
use tritium_core::Trit;
use tritium_format::{SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row, salt_rows_to_dense};
use tritium_nn::{DenseLinear, NnError, SaltLinear};

fn row(k: usize, planes: usize, seed: usize) -> SaltRow {
    let nb = num_blocks(k);
    let planes = (0..planes)
        .map(|plane| {
            let trits: Vec<Trit> = (0..k)
                .map(|i| {
                    let value = ((i * 17 + plane * 7 + seed * 13) % 3) as i8 - 1;
                    Trit::from_i8(value).unwrap()
                })
                .collect();
            let scales: Vec<f16> = (0..nb)
                .map(|block| f16::from_f32(0.03125 * (1 + plane + block + seed) as f32))
                .collect();
            let mut packed = vec![0; nb * TQ2_0_BLOCK_BYTES];
            pack_tq2_0_row(&trits, &scales, &mut packed).unwrap();
            packed
        })
        .collect();
    SaltRow { k, planes }
}

#[test]
fn packed_salt_linear_matches_dense_a8_without_matrix_materialization() {
    let (m, n, k) = (2, 4, 257);
    let rows = vec![row(k, 1, 1), row(k, 3, 2), row(k, 2, 3), row(k, 0, 4)];
    let dense = salt_rows_to_dense(&rows).unwrap();
    let dense_linear = DenseLinear::new(dense, n, k).unwrap();
    let salt_linear = SaltLinear::new(rows.clone(), n, k).unwrap();
    let mut act: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 + 0.25) * 0.019).cos() * 3.5)
        .collect();
    act[k..].fill(0.0);

    let mut expected = vec![0.0; m * n];
    let mut got = vec![0.0; m * n];
    dense_linear.forward(&act, m, &mut expected).unwrap();
    salt_linear.forward(&act, m, &mut got).unwrap();

    assert_eq!(got, expected);
    let encoded_bytes: usize = rows.iter().flat_map(|row| &row.planes).map(Vec::len).sum();
    assert_eq!(salt_linear.packed_bytes(), encoded_bytes);
    assert!(salt_linear.packed_bytes() < n * k * size_of::<f32>());
}

#[test]
fn packed_salt_linear_rejects_malformed_geometry() {
    let k = 257;
    assert!(matches!(
        SaltLinear::new(vec![row(k, 1, 0)], 2, k),
        Err(NnError::Shape { .. })
    ));
    assert!(matches!(
        SaltLinear::new(vec![row(k + 1, 1, 0)], 1, k),
        Err(NnError::Shape { .. })
    ));

    let mut truncated = row(k, 1, 0);
    truncated.planes[0].pop();
    assert!(matches!(
        SaltLinear::new(vec![truncated], 1, k),
        Err(NnError::Backend(_))
    ));

    if let Ok(huge) = usize::try_from(u64::from(u32::MAX) + 1) {
        assert!(matches!(
            SaltLinear::new(
                vec![SaltRow {
                    k: huge,
                    planes: Vec::new(),
                }],
                1,
                huge,
            ),
            Err(NnError::Shape { .. })
        ));
    }
}

#[test]
fn packed_salt_linear_rejects_bad_operand_shapes() {
    let (n, k) = (2, 257);
    let linear = SaltLinear::new(vec![row(k, 1, 0), row(k, 2, 1)], n, k).unwrap();

    let mut out = vec![0.0; n];
    assert!(matches!(
        linear.forward(&vec![0.0; k - 1], 1, &mut out),
        Err(NnError::Shape { .. })
    ));
    assert!(matches!(
        linear.forward(&vec![0.0; k], 1, &mut out[..n - 1]),
        Err(NnError::Shape { .. })
    ));
}
