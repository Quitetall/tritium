use half::f16;
use tritium_core::Trit;
use tritium_format::{
    PackedTrainingSaltSnapshot, QK_K, TQ2_0_BLOCK_BYTES, TernaryStructure, TrainingSaltPlane,
    pack_tq2_0_block,
};

#[test]
fn snapshot_uses_canonical_tq2_codes_and_external_group_scales() {
    let (rows, cols, group_size) = (2usize, 321usize, 128usize);
    let groups_per_row = cols.div_ceil(group_size);
    let trits = (0..rows * cols)
        .map(|index| match (index + index / cols) % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect::<Vec<_>>();
    let scales = (0..rows * groups_per_row)
        .map(|index| f16::from_f32((index + 1) as f32 / 16.0))
        .collect::<Vec<_>>();
    let snapshot = PackedTrainingSaltSnapshot::pack(
        rows,
        cols,
        group_size,
        TernaryStructure::Dense,
        &[TrainingSaltPlane::new(&trits, &scales)],
    )
    .unwrap();

    assert_eq!(snapshot.rows(), rows);
    assert_eq!(snapshot.cols(), cols);
    assert_eq!(snapshot.group_size(), group_size);
    assert_eq!(snapshot.groups_per_row(), groups_per_row);
    assert_eq!(snapshot.planes(), 1);
    assert_eq!(snapshot.structure(), TernaryStructure::Dense);
    assert_eq!(
        snapshot.scales(),
        scales
            .iter()
            .map(|scale| f32::from(*scale))
            .collect::<Vec<_>>()
    );

    let mut block = [Trit::ZERO; QK_K];
    for (target, &source) in block.iter_mut().zip(&trits[..QK_K]) {
        *target = Trit::from_i8(source).unwrap();
    }
    let mut canonical = vec![0; TQ2_0_BLOCK_BYTES];
    pack_tq2_0_block(&block, f16::ONE, &mut canonical).unwrap();
    assert_eq!(&snapshot.codes()[..QK_K / 4], &canonical[..QK_K / 4]);
    assert_eq!(snapshot.row_bytes(), cols.div_ceil(QK_K) * (QK_K / 4));
}

#[test]
fn snapshot_rejects_invalid_structure_and_scale_semantics() {
    let invalid_s34 = PackedTrainingSaltSnapshot::pack(
        1,
        4,
        4,
        TernaryStructure::S34,
        &[TrainingSaltPlane::new(&[1, 1, 1, 1], &[f16::ONE])],
    );
    assert!(invalid_s34.is_err());

    let nonzero_behind_zero_scale = PackedTrainingSaltSnapshot::pack(
        1,
        4,
        4,
        TernaryStructure::Dense,
        &[TrainingSaltPlane::new(&[1, 0, 0, 0], &[f16::ZERO])],
    );
    assert!(nonzero_behind_zero_scale.is_err());
}

#[test]
fn snapshot_rejects_absurd_geometry_before_allocating() {
    let rows = (isize::MAX as usize / (QK_K / 4)) + 1;
    let result = PackedTrainingSaltSnapshot::pack(
        rows,
        1,
        1,
        TernaryStructure::Dense,
        &[TrainingSaltPlane::new(&[0], &[f16::ONE])],
    );
    assert!(result.is_err());
}
