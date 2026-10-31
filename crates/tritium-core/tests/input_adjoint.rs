use tritium_core::{
    AdditiveError, AdditiveView, PerTilePlan, PerTileRowSpan, Trit, UniformPlan, reference_embed,
    reference_ternary_matmul, reference_ternary_matmul_input_vjp,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, ScaleLaw, Transport,
};

fn layout(law: ScaleLaw, basis: Basis, group: u16, cols: u64) -> AdditiveLayout {
    AdditiveLayout {
        rows: 3,
        cols,
        tile: 256,
        group,
        max_planes: tritium_schema::admitted_law(law).unwrap().max_planes,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law,
        basis,
        transport: Transport::Raw,
    }
}

fn trits(count: usize) -> Vec<Trit> {
    (0..count)
        .map(|i| Trit::from_i8(((i * 7 + i / 11) % 3) as i8 - 1).unwrap())
        .collect()
}

fn scales(count: usize) -> Vec<f32> {
    (0..count).map(|i| [0.125, 0.5, 1.0][i % 3]).collect()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}

fn check_adjoint(view: AdditiveView<'_>, batch: usize) {
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    let dy: Vec<f32> = (0..batch * rows)
        .map(|i| (i % 5) as f32 * 0.25 - 0.5)
        .collect();
    let x: Vec<f32> = (0..batch * cols)
        .map(|i| (i % 7) as f32 * 0.125 - 0.375)
        .collect();
    let mut workspace = vec![91.0; cols];
    let mut dx = vec![92.0; batch * cols];
    reference_ternary_matmul_input_vjp(&dy, &view.bind_matmul(), batch, &mut workspace, &mut dx)
        .unwrap();
    if batch == 0 {
        assert_eq!(workspace, vec![91.0; cols]);
    }
    let mut dense = vec![0.0; rows * cols];
    reference_embed(
        &view.bind_gather(),
        &(0..rows).collect::<Vec<_>>(),
        &mut dense,
    )
    .unwrap();
    for b in 0..batch {
        for col in 0..cols {
            let expected: f64 = (0..rows)
                .map(|row| f64::from(dy[b * rows + row]) * f64::from(dense[row * cols + col]))
                .sum();
            close(f64::from(dx[b * cols + col]), expected, 2e-6);
        }
    }
    let mut transformed = vec![0.0; x.len()];
    let mut y = vec![0.0; dy.len()];
    reference_ternary_matmul(&x, &view.bind_matmul(), batch, &mut transformed, &mut y).unwrap();
    let left: f64 = y
        .iter()
        .zip(&dy)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let right: f64 = x
        .iter()
        .zip(&dx)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    close(left, right, 2e-5);
}

#[test]
fn uniform_adjoint_matches_all_admitted_laws_groups_planes_and_bases() {
    for admitted in ADMITTED_LAWS {
        for group in [32, 64, 128, 256] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 4 },
                Basis::SignedRht {
                    block: 4,
                    seed: 19,
                    domain: 7,
                },
            ] {
                for planes in 1..=admitted.max_planes {
                    // Ragged last tile/group; rotation block is independent of both.
                    let layout = layout(admitted.law, basis, group, 260);
                    let plan = UniformPlan::new(layout, planes).unwrap();
                    let trits = trits(plan.trit_count());
                    let scales = scales(plan.scale_count());
                    let view = AdditiveView::new(layout, planes, &trits, &scales).unwrap();
                    for batch in [0, 1, 3] {
                        check_adjoint(view, batch);
                    }
                }
            }
        }
    }
}

#[test]
fn compact_adjoint_handles_variable_counts_and_all_pruned_rows() {
    for admitted in ADMITTED_LAWS {
        for group in [32, 64, 128, 256] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 4 },
                Basis::SignedRht {
                    block: 4,
                    seed: 19,
                    domain: 7,
                },
            ] {
                let layout = AdditiveLayout {
                    allocation: PlaneAllocation::PerTile,
                    ..layout(admitted.law, basis, group, 260)
                };
                for counts in [
                    [0, admitted.max_planes, 1, 0, admitted.max_planes, 1],
                    [0; 6],
                ] {
                    let plan = PerTilePlan::new(layout, &counts).unwrap();
                    let mut spans = vec![PerTileRowSpan::default(); plan.index_len()];
                    let index = plan.build(&mut spans).unwrap();
                    let trits = trits(plan.trit_count());
                    let scales = scales(plan.scale_count());
                    let view = AdditiveView::new_per_tile(index, &trits, &scales).unwrap();
                    for batch in [0, 1, 3] {
                        check_adjoint(view, batch);
                    }
                }
            }
        }
    }
}

#[test]
fn input_adjoint_matches_finite_differences_of_actual_forward() {
    for admitted in ADMITTED_LAWS {
        for basis in [
            Basis::Identity,
            Basis::Hadamard { block: 4 },
            Basis::SignedRht {
                block: 4,
                seed: 19,
                domain: 7,
            },
        ] {
            let layout = layout(admitted.law, basis, 32, 4);
            let plan = UniformPlan::new(layout, admitted.max_planes).unwrap();
            let trits = trits(plan.trit_count());
            let scales = scales(plan.scale_count());
            let view = AdditiveView::new(layout, admitted.max_planes, &trits, &scales).unwrap();
            let dy = [0.25, -0.75, 0.5];
            let mut dx = [0.0; 4];
            reference_ternary_matmul_input_vjp(&dy, &view.bind_matmul(), 1, &mut [0.0; 4], &mut dx)
                .unwrap();
            let objective = |x: &[f32; 4]| {
                let mut y = [0.0; 3];
                reference_ternary_matmul(x, &view.bind_matmul(), 1, &mut [0.0; 4], &mut y).unwrap();
                y.iter()
                    .zip(dy)
                    .map(|(&a, b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
            };
            for col in 0..4 {
                let mut plus = [0.125, -0.25, 0.375, 0.5];
                let mut minus = plus;
                let epsilon = 0.015625;
                plus[col] += epsilon;
                minus[col] -= epsilon;
                let derivative = (objective(&plus) - objective(&minus)) / f64::from(2.0 * epsilon);
                close(f64::from(dx[col]), derivative, 1e-5);
            }
        }
    }
}

#[test]
fn invalid_shapes_and_overflow_leave_both_buffers_untouched() {
    let layout = layout(ADMITTED_LAWS[0].law, Basis::Identity, 32, 4);
    let trits = [Trit::POS; 12];
    let scales = [1.0; 3];
    let view = AdditiveView::new(layout, 1, &trits, &scales)
        .unwrap()
        .bind_matmul();
    for (batch, gradients, scratch, output, error) in [
        (1, 2, 4, 4, AdditiveError::MatmulShape),
        (1, 3, 3, 4, AdditiveError::MatmulShape),
        (1, 3, 4, 3, AdditiveError::MatmulShape),
        (0, 1, 4, 0, AdditiveError::MatmulShape),
        (0, 0, 0, 0, AdditiveError::MatmulShape),
        (usize::MAX, 0, 4, 0, AdditiveError::SizeOverflow),
        (usize::MAX / 3, 0, 4, 0, AdditiveError::SizeOverflow),
    ] {
        let dy = vec![0.25; gradients];
        let mut scratch = vec![17.0; scratch];
        let mut output = vec![19.0; output];
        let original_scratch = scratch.clone();
        let original_output = output.clone();
        assert_eq!(
            reference_ternary_matmul_input_vjp(&dy, &view, batch, &mut scratch, &mut output),
            Err(error)
        );
        assert_eq!(scratch, original_scratch);
        assert_eq!(output, original_output);
    }
}
