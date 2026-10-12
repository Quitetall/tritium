use tritium_core::{
    AdditiveError, AdditiveView, PerTileIndex, PerTileRowSpan, Trit, reference_embed,
    reference_ternary_matmul,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    Transport,
};

fn layout(
    rows: usize,
    cols: usize,
    law: tritium_schema::AdmittedLaw,
    group: u16,
) -> AdditiveLayout {
    AdditiveLayout {
        rows: rows as u64,
        cols: cols as u64,
        tile: 256,
        group,
        max_planes: law.max_planes,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::D2,
        law: law.law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    }
}

#[test]
fn public_row_cursor_and_buffer_lengths_share_checked_geometry() {
    // Tied scales skip nonpruned lower-count tiles even when the higher plane
    // omits their trits. Rows use absolute compact offsets, not dense offsets.
    let layout = layout(2, 257, ADMITTED_LAWS[2], 128);
    let counts = [1, 3, 2, 0];
    let plan = tritium_core::PerTilePlan::new(layout, &counts).unwrap();
    let expected = counts.len() + 6 * core::mem::size_of::<PerTileRowSpan>();
    assert_eq!(plan.buffer_len_bytes().unwrap(), expected);
    let mut spans = vec![PerTileRowSpan::default(); plan.index_len()];
    let index = plan.build(&mut spans).unwrap();
    assert_eq!(index.buffer_len_bytes().unwrap(), expected);
    let trits = vec![Trit::POS; index.trit_count()];
    let scales = vec![0.5; index.scale_count()];
    let view = AdditiveView::new_per_tile(index, &trits, &scales).unwrap();
    assert_eq!(view.auxiliary_buffer_len_bytes().unwrap(), expected);
    let tiles: Vec<_> = index
        .tiles_in_row(1, 0)
        .unwrap()
        .map(|t| (t.columns(), t.trit_offset(), t.scale_offset()))
        .collect();
    assert_eq!(tiles, [(256..257, 513, 2)]);
    let tiles: Vec<_> = index
        .tiles_in_row(1, 1)
        .unwrap()
        .map(|t| (t.columns(), t.trit_offset(), t.scale_offset()))
        .collect();
    assert_eq!(tiles, [(0..256, 514, 3)]);
    assert!(matches!(
        index.tiles_in_row(0, 2),
        Err(AdditiveError::RowOutOfRange)
    ));
    assert!(matches!(
        index.tiles_in_row(3, 0),
        Err(AdditiveError::InvalidPlaneCount)
    ));
    let plan = tritium_core::PerTilePlan::new(layout, &[0; 4]).unwrap();
    assert_eq!(plan.buffer_len_bytes().unwrap(), 4);
    let index = plan.build(&mut []).unwrap();
    let view = AdditiveView::new_per_tile(index, &[], &[]).unwrap();
    assert_eq!(view.auxiliary_buffer_len_bytes().unwrap(), 4);
    assert!(matches!(
        index.tiles_in_row(0, 0),
        Err(AdditiveError::InvalidPlaneCount)
    ));
    let mut uniform_layout = layout;
    uniform_layout.allocation = PlaneAllocation::Uniform;
    let trits = vec![Trit::POS; 514];
    let view = AdditiveView::new(uniform_layout, 1, &trits, &[0.5; 6]).unwrap();
    assert_eq!(view.auxiliary_buffer_len_bytes().unwrap(), 0);
}

#[test]
fn logical_prefix_uses_retained_coordinates_and_stored_bits_not_dense_equivalence() {
    let layout = layout(1, 257, ADMITTED_LAWS[0], 128);
    let parent_counts = [1, 2];
    let child_counts = [0, 2];
    let mut parent_storage = vec![PerTileRowSpan::default(); 2];
    let mut child_storage = vec![PerTileRowSpan::default(); 2];
    let parent_index = PerTileIndex::build(layout, &parent_counts, &mut parent_storage).unwrap();
    let child_index = PerTileIndex::build(layout, &child_counts, &mut child_storage).unwrap();
    let mut parent_trits = vec![Trit::POS; 258];
    parent_trits[256] = Trit::NEG;
    let child_trits = [Trit::NEG, Trit::POS];
    let parent_scales = [0.125, 0.25, 0.5, 1.0];
    let child_scales = [0.5, 1.0];
    let parent = AdditiveView::new_per_tile(parent_index, &parent_trits, &parent_scales).unwrap();
    let child = AdditiveView::new_per_tile(child_index, &child_trits, &child_scales).unwrap();
    assert!(!parent.trits().starts_with(child.trits()));
    assert!(child.is_logical_prefix_of(parent));
    assert!(!parent.is_logical_prefix_of(child));

    let zero_parent_scales = [0.0; 4];
    let zero_child_scales = [0.0; 2];
    let changed_trits = [Trit::POS; 2];
    let parent =
        AdditiveView::new_per_tile(parent_index, &parent_trits, &zero_parent_scales).unwrap();
    let child =
        AdditiveView::new_per_tile(child_index, &changed_trits, &zero_child_scales).unwrap();
    let mut parent_row = [1.0; 257];
    let mut child_row = [1.0; 257];
    parent.dequant_row_into(0, &mut parent_row).unwrap();
    child.dequant_row_into(0, &mut child_row).unwrap();
    assert_eq!(parent_row, child_row);
    assert!(!child.is_logical_prefix_of(parent));

    // The core accepts signed zero; canonical format scales reject it. The
    // logical comparator still distinguishes stored bits without widening rules.
    let signed_zero_scales = [0.0, 0.0, -0.0, 0.0];
    let parent =
        AdditiveView::new_per_tile(parent_index, &parent_trits, &signed_zero_scales).unwrap();
    let child = AdditiveView::new_per_tile(child_index, &child_trits, &zero_child_scales).unwrap();
    assert!(!child.is_logical_prefix_of(parent));
}

// Reorder/mask fixture data only. Both numerical oracles are the actual shared
// production references; no quantizer, basis transform or dot formula is copied.
fn compare_to_uniform(layout: AdditiveLayout, counts: &[u8]) {
    let rows = layout.rows as usize;
    let cols = layout.cols as usize;
    let planes = layout.max_planes as usize;
    let tiles = cols.div_ceil(256);
    let group = layout.group as usize;
    let groups = cols.div_ceil(group);
    let mut uniform_trits = Vec::new();
    let mut compact_trits = Vec::new();
    let mut uniform_scales = Vec::new();
    let mut compact_scales = Vec::new();
    for plane in 0..planes {
        for row in 0..rows {
            for col in 0..cols {
                let active = usize::from(counts[row * tiles + col / 256]) > plane;
                let value = Trit::from_i8(((row + col + plane) % 3) as i8 - 1).unwrap();
                uniform_trits.push(if active { value } else { Trit::ZERO });
                if active {
                    compact_trits.push(value);
                }
            }
            if layout.law.anchor == ScaleAnchor::Group
                && (layout.law.relation == PlaneRelation::Free || plane == 0)
            {
                for unit in 0..groups {
                    let scale = [0.5, 1.0, 2.0][(plane + row + unit) % 3];
                    uniform_scales.push(scale);
                    let count = counts[row * tiles + unit * group / 256];
                    if usize::from(count) > plane {
                        compact_scales.push(scale);
                    }
                }
            }
        }
        if layout.law.anchor == ScaleAnchor::Tensor {
            uniform_scales.push(0.5);
            if counts.iter().any(|&n| usize::from(n) > plane) {
                compact_scales.push(0.5);
            }
        }
    }
    let mut index_storage =
        vec![PerTileRowSpan::default(); PerTileIndex::required_storage(layout, counts).unwrap()];
    let index = PerTileIndex::build(layout, counts, &mut index_storage).unwrap();
    assert_eq!(index.trit_count(), compact_trits.len());
    assert_eq!(index.scale_count(), compact_scales.len());
    let compact = AdditiveView::new_per_tile(index, &compact_trits, &compact_scales).unwrap();
    let uniform = AdditiveView::new(
        AdditiveLayout {
            allocation: PlaneAllocation::Uniform,
            ..layout
        },
        layout.max_planes,
        &uniform_trits,
        &uniform_scales,
    )
    .unwrap();
    assert_eq!(compact.tile_counts(), Some(counts));
    for row in 0..rows {
        let mut actual = vec![0.; cols];
        let mut expected = vec![0.; cols];
        compact.dequant_row_into(row, &mut actual).unwrap();
        uniform.dequant_row_into(row, &mut expected).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let activations: Vec<f32> = (0..2 * cols).map(|i| (i % 7) as f32 - 3.).collect();
    let mut a_scratch = vec![0.; activations.len()];
    let mut b_scratch = a_scratch.clone();
    let mut actual = vec![0.; 2 * rows];
    let mut expected = actual.clone();
    reference_ternary_matmul(
        &activations,
        &compact.bind_matmul(),
        2,
        &mut a_scratch,
        &mut actual,
    )
    .unwrap();
    reference_ternary_matmul(
        &activations,
        &uniform.bind_matmul(),
        2,
        &mut b_scratch,
        &mut expected,
    )
    .unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let ids = [rows - 1, 0, rows - 1];
    let mut actual = vec![0.; ids.len() * cols];
    let mut expected = actual.clone();
    reference_embed(&compact.bind_gather(), &ids, &mut actual).unwrap();
    reference_embed(&uniform.bind_gather(), &ids, &mut expected).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn compact_reference_matches_uniform_for_all_admitted_laws_and_ragged_rows() {
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for cols in [1, 127, 128, 129, 255, 256, 257, 512] {
                let layout = layout(2, cols, law, group);
                let tiles = 2 * cols.div_ceil(256);
                compare_to_uniform(layout, &vec![law.max_planes; tiles]);
                let counts: Vec<_> = (0..tiles)
                    .map(|i| (i % (usize::from(law.max_planes) + 1)) as u8)
                    .collect();
                compare_to_uniform(layout, &counts);
                compare_to_uniform(layout, &vec![0; tiles]);
            }
        }
    }
}

#[test]
fn mixed_tiles_preserve_basis_and_tied_anchor_indexing() {
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 16 },
                Basis::SignedRht {
                    block: 16,
                    seed: 17,
                    domain: 91,
                },
            ] {
                let mut layout = layout(2, 512, law, group);
                layout.basis = basis;
                let counts = if law.max_planes == 1 {
                    [1, 0, 0, 1]
                } else {
                    [3, 1, 0, 2]
                };
                compare_to_uniform(layout, &counts);
                compare_to_uniform(layout, &[law.max_planes; 4]);
            }
        }
    }
}

#[test]
fn index_validation_and_size_bounds_are_fail_closed() {
    let layout = layout(2, 257, ADMITTED_LAWS[0], 128);
    assert!(matches!(
        PerTileIndex::required_storage(layout, &[1, 2, 3]),
        Err(AdditiveError::TileCountMismatch {
            expected: 4,
            got: 3
        })
    ));
    assert!(matches!(
        PerTileIndex::required_storage(layout, &[1, 2, 0, 4]),
        Err(AdditiveError::InvalidTilePlaneCount { .. })
    ));
    assert_eq!(
        PerTileIndex::required_storage(layout, &[1, 2, 0, 3]).unwrap(),
        6
    );
    let mut short = [PerTileRowSpan::default(); 5];
    assert!(matches!(
        PerTileIndex::build(layout, &[1, 2, 0, 3], &mut short),
        Err(AdditiveError::IndexStorageLength {
            expected: 6,
            got: 5
        })
    ));
    assert_eq!(short, [PerTileRowSpan::default(); 5]);
    let mut uniform = layout;
    uniform.allocation = PlaneAllocation::Uniform;
    assert_eq!(
        PerTileIndex::required_storage(uniform, &[1; 4]),
        Err(AdditiveError::UnsupportedAllocation)
    );
    let mut huge = layout;
    huge.rows = u64::MAX;
    huge.cols = 512;
    assert_eq!(
        PerTileIndex::required_storage(huge, &[]),
        Err(AdditiveError::SizeOverflow)
    );
}

#[test]
fn all_pruned_data_has_no_index_or_payload_but_preserves_shape_checks() {
    let layout = layout(2, 257, ADMITTED_LAWS[0], 128);
    assert_eq!(PerTileIndex::required_storage(layout, &[0; 4]).unwrap(), 0);
    let mut storage = [];
    let index = PerTileIndex::build(layout, &[0; 4], &mut storage).unwrap();
    let view = AdditiveView::new_per_tile(index, &[], &[]).unwrap();
    assert_eq!(view.plane_count(), 0);
    assert!(view.plane_trits(0).is_err());
    let mut output = [123.; 257];
    assert_eq!(
        view.dequant_row_into(2, &mut output),
        Err(AdditiveError::RowOutOfRange)
    );
    assert_eq!(output, [123.; 257]);
    view.dequant_row_into(1, &mut output).unwrap();
    assert_eq!(output, [0.; 257]);
    let mut result = [123.; 2];
    let mut scratch = [0.; 256];
    assert_eq!(
        reference_ternary_matmul(
            &[1.; 256],
            &view.bind_matmul(),
            1,
            &mut scratch,
            &mut result
        ),
        Err(AdditiveError::MatmulShape)
    );
    assert_eq!(result, [123.; 2]);
}

#[test]
fn compact_payload_lengths_scales_and_plane_slices_are_validated() {
    let layout = layout(2, 257, ADMITTED_LAWS[0], 128);
    let counts = [1, 3, 2, 0];
    let mut storage = [PerTileRowSpan::default(); 6];
    let index = PerTileIndex::build(layout, &counts, &mut storage).unwrap();
    assert_eq!(index.trit_count(), 771);
    assert_eq!(index.scale_count(), 9);
    let trits = vec![Trit::POS; index.trit_count()];
    let scales = vec![0.5; index.scale_count()];
    assert!(matches!(
        AdditiveView::new_per_tile(index, &trits[..770], &scales),
        Err(AdditiveError::TritCountMismatch { .. })
    ));
    assert!(matches!(
        AdditiveView::new_per_tile(index, &trits, &scales[..8]),
        Err(AdditiveError::ScaleCountMismatch { .. })
    ));
    for bad in [f32::NAN, f32::INFINITY, -1.] {
        let mut bad_scales = scales.clone();
        bad_scales[0] = bad;
        assert!(matches!(
            AdditiveView::new_per_tile(index, &trits, &bad_scales),
            Err(AdditiveError::InvalidScale)
        ));
    }
    let view = AdditiveView::new_per_tile(index, &trits, &scales).unwrap();
    assert_eq!(view.plane_trits(0).unwrap().len(), 513);
    assert_eq!(view.plane_trits(1).unwrap().len(), 257);
    assert_eq!(view.plane_trits(2).unwrap().len(), 1);
    assert!(view.plane_trits(3).is_err());
}

#[test]
fn index_storage_grows_with_rows_and_planes_not_coefficients_or_tiles() {
    let layout = layout(2, 4096, ADMITTED_LAWS[0], 128);
    let counts = [3; 32];
    assert_eq!(PerTileIndex::required_storage(layout, &counts).unwrap(), 6);
    let mut storage = [PerTileRowSpan::default(); 6];
    let index = PerTileIndex::build(layout, &counts, &mut storage).unwrap();
    assert_eq!(index.trit_count(), 2 * 4096 * 3);
    assert_eq!(index.scale_count(), 2 * 32 * 3);
}

#[test]
fn tensor_scale_preserves_one_sequential_partial_across_tile_boundaries() {
    let layout = layout(1, 512, ADMITTED_LAWS[3], 128);
    let mut storage = [PerTileRowSpan::default(); 1];
    let counts = [1, 1];
    let index = PerTileIndex::build(layout, &counts, &mut storage).unwrap();
    let mut trits = [Trit::ZERO; 512];
    let mut act = [0.; 512];
    for (col, value) in [(0, 1.0e20), (255, 1.), (256, -1.0e20), (511, 1.)] {
        trits[col] = Trit::POS;
        act[col] = value;
    }
    let compact = AdditiveView::new_per_tile(index, &trits, &[1.]).unwrap();
    let uniform = AdditiveView::new(
        AdditiveLayout {
            allocation: PlaneAllocation::Uniform,
            ..layout
        },
        1,
        &trits,
        &[1.],
    )
    .unwrap();
    for view in [compact, uniform] {
        let mut scratch = [0.; 512];
        let mut out = [0.; 1];
        reference_ternary_matmul(&act, &view.bind_matmul(), 1, &mut scratch, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 1.0f32.to_bits());
    }
}

#[test]
fn rejected_counts_do_not_overwrite_existing_index_storage() {
    let layout = layout(2, 257, ADMITTED_LAWS[0], 128);
    let mut storage = [PerTileRowSpan::default(); 6];
    PerTileIndex::build(layout, &[3; 4], &mut storage).unwrap();
    let before = storage;
    assert!(PerTileIndex::build(layout, &[3, 3, 3, 4], &mut storage).is_err());
    assert_eq!(storage, before);
    assert!(PerTileIndex::build(layout, &[3, 3, 3], &mut storage).is_err());
    assert_eq!(storage, before);
}
