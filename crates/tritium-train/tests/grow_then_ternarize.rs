use tritium_train::grow::{
    NET2WIDER_ALGORITHM_V1, NET2WIDER_ALGORITHM_V2, NET2WIDER_MAX_REPLICATIONS_PER_SOURCE,
    NET2WIDER_SPLIT_DENOMINATOR_LOG2,
};
use tritium_train::ops::{act, dense, elementwise};
use tritium_train::{
    AdamW, GrowError, Net2WiderPlan, Optimizer, QualityBytesPoint, QualityBytesReport, Tape,
};

fn dense_mlp_forward(
    x: &[f32],
    rows: usize,
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
    incoming: &[f32],
    outgoing: &[f32],
) -> Vec<f32> {
    let hidden = dense::forward(x, incoming, rows, hidden_width, input_width);
    let hidden = act::relu2_forward(&hidden);
    dense::forward(&hidden, outgoing, rows, output_width, hidden_width)
}

#[allow(clippy::too_many_arguments)]
fn swiglu_forward(
    x: &[f32],
    rows: usize,
    model_width: usize,
    hidden_width: usize,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
) -> Vec<f32> {
    let gate = dense::forward(x, gate, rows, hidden_width, model_width);
    let gate = act::silu_forward(&gate);
    let up = dense::forward(x, up, rows, hidden_width, model_width);
    let hidden = elementwise::mul_forward(&gate, &up);
    dense::forward(&hidden, down, rows, model_width, hidden_width)
}

#[allow(clippy::too_many_arguments)]
fn swiglu_mse_gradients(
    x: &[f32],
    target: &[f32],
    rows: usize,
    model_width: usize,
    hidden_width: usize,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
) -> [Vec<f32>; 3] {
    let mut tape = Tape::new();
    let x_id = tape.leaf(x.to_vec());
    let gate_id = tape.leaf(gate.to_vec());
    let up_id = tape.leaf(up.to_vec());
    let down_id = tape.leaf(down.to_vec());
    let target_id = tape.leaf(target.to_vec());

    let gate_projection = tape.dense_matmul(x_id, gate_id, rows, hidden_width, model_width);
    let gate_activation = tape.silu(gate_projection);
    let up_projection = tape.dense_matmul(x_id, up_id, rows, hidden_width, model_width);
    let hidden = tape.mul(gate_activation, up_projection);
    let output = tape.dense_matmul(hidden, down_id, rows, model_width, hidden_width);
    let loss = tape.mse(output, target_id);
    let gradients = tape.backward(loss);

    [
        gradients[gate_id].clone(),
        gradients[up_id].clone(),
        gradients[down_id].clone(),
    ]
}

fn row_max_abs_delta(values: &[f32], width: usize, left: usize, right: usize) -> f32 {
    values[left * width..(left + 1) * width]
        .iter()
        .zip(&values[right * width..(right + 1) * width])
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn biased_dense_mlp_forward(
    x: &[f32],
    rows: usize,
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
    incoming: &[f32],
    hidden_bias: &[f32],
    outgoing: &[f32],
    output_bias: &[f32],
) -> Vec<f32> {
    let mut hidden = dense::forward(x, incoming, rows, hidden_width, input_width);
    for row in hidden.chunks_exact_mut(hidden_width) {
        for (value, bias) in row.iter_mut().zip(hidden_bias) {
            *value += bias;
        }
    }
    let hidden = act::relu2_forward(&hidden);
    let mut output = dense::forward(&hidden, outgoing, rows, output_width, hidden_width);
    for row in output.chunks_exact_mut(output_width) {
        for (value, bias) in row.iter_mut().zip(output_bias) {
            *value += bias;
        }
    }
    output
}

#[test]
fn net2wider_preserves_dense_mlp_forward_before_salt() {
    const ROWS: usize = 3;
    const INPUT: usize = 4;
    const HIDDEN: usize = 3;
    const WIDE_HIDDEN: usize = 8;
    const OUTPUT: usize = 2;

    let x = [
        0.25, -0.50, 0.75, 0.10, // row 0
        -0.20, 0.30, 0.40, -0.80, // row 1
        0.90, -0.10, -0.35, 0.60, // row 2
    ];
    // Row-major [hidden, input].
    let incoming = [
        0.20, -0.30, 0.10, 0.50, // hidden 0
        -0.40, 0.25, 0.60, -0.20, // hidden 1
        0.70, 0.15, -0.45, 0.30, // hidden 2
    ];
    // Row-major [output, hidden].
    let outgoing = [0.40, -0.25, 0.15, -0.35, 0.20, 0.55];

    let original = dense_mlp_forward(&x, ROWS, INPUT, HIDDEN, OUTPUT, &incoming, &outgoing);
    let plan = Net2WiderPlan::seeded(HIDDEN, WIDE_HIDDEN, 0x5eed).unwrap();
    let wide_incoming = plan.expand_incoming_rows(&incoming, INPUT).unwrap();
    let wide_outgoing = plan.expand_outgoing_columns(&outgoing, OUTPUT).unwrap();
    let widened = dense_mlp_forward(
        &x,
        ROWS,
        INPUT,
        WIDE_HIDDEN,
        OUTPUT,
        &wide_incoming,
        &wide_outgoing,
    );

    assert_close(&widened, &original, 2e-6);
}

#[test]
fn one_plan_preserves_transformer_swiglu_forward_before_salt() {
    const ROWS: usize = 2;
    const MODEL: usize = 3;
    const HIDDEN: usize = 4;
    const WIDE_HIDDEN: usize = 9;

    let x = [0.25, -0.50, 0.75, -0.20, 0.30, 0.40];
    let gate = [
        0.20, -0.30, 0.10, -0.40, 0.25, 0.60, 0.70, 0.15, -0.45, 0.30, 0.05, -0.20,
    ];
    let up = [
        -0.10, 0.35, 0.20, 0.60, -0.25, 0.10, -0.50, 0.40, 0.30, 0.15, -0.30, 0.55,
    ];
    let down = [
        0.40, -0.25, 0.15, 0.20, -0.35, 0.20, 0.55, -0.10, 0.05, 0.30, -0.45, 0.25,
    ];

    let original = swiglu_forward(&x, ROWS, MODEL, HIDDEN, &gate, &up, &down);
    let plan = Net2WiderPlan::seeded(HIDDEN, WIDE_HIDDEN, 0x0027).unwrap();
    let wide_gate = plan.expand_incoming_rows(&gate, MODEL).unwrap();
    let wide_up = plan.expand_incoming_rows(&up, MODEL).unwrap();
    let wide_down = plan.expand_outgoing_columns(&down, MODEL).unwrap();
    let widened = swiglu_forward(
        &x,
        ROWS,
        MODEL,
        WIDE_HIDDEN,
        &wide_gate,
        &wide_up,
        &wide_down,
    );

    assert_close(&widened, &original, 2e-6);
}

#[test]
fn grown_plan_uses_replayable_unequal_dyadic_splits_without_changing_the_function() {
    const INPUT: usize = 3;
    const HIDDEN: usize = 2;
    const WIDE_HIDDEN: usize = 8;
    const DENOMINATOR: u32 = 1 << NET2WIDER_SPLIT_DENOMINATOR_LOG2;

    let plan = Net2WiderPlan::seeded(HIDDEN, WIDE_HIDDEN, 0x27).unwrap();
    let replay = Net2WiderPlan::seeded(HIDDEN, WIDE_HIDDEN, 0x27).unwrap();
    assert_eq!(plan.algorithm(), NET2WIDER_ALGORITHM_V2);
    assert_eq!(plan.split_denominator_log2(), Some(24));
    assert_eq!(plan, replay);

    let numerators = plan.split_numerators().expect("grown plan split metadata");
    assert_eq!(
        numerators,
        &[
            2_516_583, 5_592_405, 4_194_303, 1_677_724, 5_033_163, 8_388_607, 3_355_443, 2_796_204
        ]
    );
    for source in 0..HIDDEN {
        let source_numerators: Vec<_> = plan
            .source_indices()
            .iter()
            .zip(numerators)
            .filter_map(|(&candidate, &numerator)| (candidate == source).then_some(numerator))
            .collect();
        assert_eq!(source_numerators.iter().copied().sum::<u32>(), DENOMINATOR);
        assert!(source_numerators.iter().all(|&numerator| numerator > 0));
        for (index, &numerator) in source_numerators.iter().enumerate() {
            assert!(
                source_numerators[index + 1..]
                    .iter()
                    .all(|&other| numerator != other)
            );
        }
    }

    let x = [0.25, -0.50, 0.75, -0.20, 0.30, 0.40];
    let incoming = [0.20, -0.30, 0.10, -0.40, 0.25, 0.60];
    let outgoing = [0.80, -0.40];
    let original = dense_mlp_forward(&x, 2, INPUT, HIDDEN, 1, &incoming, &outgoing);
    let wide_incoming = plan.expand_incoming_rows(&incoming, INPUT).unwrap();
    let wide_outgoing = plan.expand_outgoing_columns(&outgoing, 1).unwrap();
    for source in 0..HIDDEN {
        let source_columns: Vec<_> = plan
            .source_indices()
            .iter()
            .zip(&wide_outgoing)
            .filter_map(|(&candidate, &weight)| (candidate == source).then_some(weight))
            .collect();
        for (index, &weight) in source_columns.iter().enumerate() {
            assert!(
                source_columns[index + 1..]
                    .iter()
                    .all(|&other| weight != other)
            );
        }
    }
    let widened = dense_mlp_forward(&x, 2, INPUT, WIDE_HIDDEN, 1, &wide_incoming, &wide_outgoing);
    assert_close(&widened, &original, 2e-6);
}

#[test]
fn unequal_splits_break_adamw_symmetry_after_three_non_collinear_steps() {
    const ROWS: usize = 2;
    const MODEL: usize = 3;
    const HIDDEN: usize = 2;
    const WIDE_HIDDEN: usize = 6;

    let gate = [0.20, -0.30, 0.10, -0.40, 0.25, 0.60];
    let up = [-0.10, 0.35, 0.20, 0.60, -0.25, 0.10];
    let down = [0.40, -0.25, 0.20, -0.35, 0.55, -0.10];
    let batches = [
        (
            [0.25, -0.50, 0.75, -0.20, 0.30, 0.40],
            [0.10, -0.20, 0.30, -0.40, 0.15, 0.05],
        ),
        (
            [-0.60, 0.10, 0.35, 0.45, -0.25, 0.80],
            [-0.30, 0.25, 0.10, 0.20, -0.35, 0.45],
        ),
        (
            [0.15, 0.70, -0.45, -0.55, 0.20, 0.65],
            [0.40, 0.05, -0.25, -0.15, 0.30, -0.10],
        ),
    ];

    let plan = Net2WiderPlan::seeded(HIDDEN, WIDE_HIDDEN, 0x27).unwrap();
    let mut wide_gate = plan.expand_incoming_rows(&gate, MODEL).unwrap();
    let mut wide_up = plan.expand_incoming_rows(&up, MODEL).unwrap();
    let mut wide_down = plan.expand_outgoing_columns(&down, MODEL).unwrap();

    let original = swiglu_forward(&batches[0].0, ROWS, MODEL, HIDDEN, &gate, &up, &down);
    let widened = swiglu_forward(
        &batches[0].0,
        ROWS,
        MODEL,
        WIDE_HIDDEN,
        &wide_gate,
        &wide_up,
        &wide_down,
    );
    assert_close(&widened, &original, 2e-6);

    let replicas: Vec<_> = plan
        .source_indices()
        .iter()
        .enumerate()
        .filter_map(|(wide, &source)| (source == 0).then_some(wide))
        .collect();
    assert!(replicas.len() >= 2);
    let (left, right) = (replicas[0], replicas[1]);
    assert_eq!(row_max_abs_delta(&wide_gate, MODEL, left, right), 0.0);
    assert_eq!(row_max_abs_delta(&wide_up, MODEL, left, right), 0.0);

    let optimizer = AdamW::new(0.01);
    let mut gate_state = optimizer.init_state(wide_gate.len());
    let mut up_state = optimizer.init_state(wide_up.len());
    let mut down_state = optimizer.init_state(wide_down.len());
    for (index, (x, target)) in batches.iter().enumerate() {
        let [gate_grad, up_grad, down_grad] = swiglu_mse_gradients(
            x,
            target,
            ROWS,
            MODEL,
            WIDE_HIDDEN,
            &wide_gate,
            &wide_up,
            &wide_down,
        );
        let step = u64::try_from(index + 1).unwrap();
        optimizer.step(step, &mut wide_gate, &gate_grad, &mut gate_state);
        optimizer.step(step, &mut wide_up, &up_grad, &mut up_state);
        optimizer.step(step, &mut wide_down, &down_grad, &mut down_state);
    }

    assert!(row_max_abs_delta(&wide_gate, MODEL, left, right) > 1e-6);
    assert!(row_max_abs_delta(&wide_up, MODEL, left, right) > 1e-6);
    assert!(row_max_abs_delta(&gate_state.m, MODEL, left, right) > 1e-6);
}

#[test]
fn seeded_mapping_is_reproducible_and_covers_every_source_unit() {
    let first = Net2WiderPlan::seeded(4, 11, 1234).unwrap();
    let replay = Net2WiderPlan::seeded(4, 11, 1234).unwrap();
    let other_seed = Net2WiderPlan::seeded(4, 11, 5678).unwrap();

    assert_eq!(first, replay);
    assert_ne!(first, other_seed);
    assert_eq!(&first.source_indices()[..4], &[0, 1, 2, 3]);
    assert!(first.source_indices().iter().all(|&source| source < 4));
    assert_eq!(first.replication_counts().iter().sum::<usize>(), 11);
    assert!(first.replication_counts().iter().all(|&count| count > 0));
}

#[test]
fn net2wider_preserves_hidden_bias_and_leaves_output_bias_unchanged() {
    let x = [0.25, -0.50, 0.75, -0.20];
    let incoming = [0.20, -0.30, -0.40, 0.25, 0.70, 0.15]; // [3, 2]
    let hidden_bias = [0.10, -0.05, 0.30];
    let outgoing = [0.40, -0.25, 0.15, -0.35, 0.20, 0.55]; // [2, 3]
    let output_bias = [0.125, -0.375];
    let original = biased_dense_mlp_forward(
        &x,
        2,
        2,
        3,
        2,
        &incoming,
        &hidden_bias,
        &outgoing,
        &output_bias,
    );

    let plan = Net2WiderPlan::seeded(3, 7, 99).unwrap();
    let wide_incoming = plan.expand_incoming_rows(&incoming, 2).unwrap();
    let wide_hidden_bias = plan.expand_hidden_vector(&hidden_bias).unwrap();
    let wide_outgoing = plan.expand_outgoing_columns(&outgoing, 2).unwrap();
    let widened = biased_dense_mlp_forward(
        &x,
        2,
        2,
        7,
        2,
        &wide_incoming,
        &wide_hidden_bias,
        &wide_outgoing,
        &output_bias,
    );

    assert_close(&widened, &original, 2e-6);
}

#[test]
fn quality_bytes_report_exposes_the_measured_pareto_frontier() {
    let compact_lower_quality = QualityBytesPoint::new(100_000_000, 70_000_000, 10.5);
    let grown_ternary = QualityBytesPoint::new(200_000_000, 80_000_000, 9.8);
    let dominated = QualityBytesPoint::new(160_000_000, 100_000_000, 11.0);
    let fp_baseline = QualityBytesPoint::new(135_000_000, 540_000_000, 10.0);
    let report = QualityBytesReport::new(vec![
        fp_baseline,
        dominated,
        grown_ternary,
        compact_lower_quality,
    ]);

    assert_eq!(
        report.pareto_frontier(),
        vec![compact_lower_quality, grown_ternary]
    );
    assert_eq!(
        report.byte_optimal_at_or_better_than(fp_baseline.held_out_perplexity),
        Some(grown_ternary)
    );
}

#[test]
fn pareto_frontier_retains_distinct_measurements_with_equal_objectives() {
    let first = QualityBytesPoint::new(100_000_000, 70_000_000, 10.5);
    let other_scale = QualityBytesPoint::new(120_000_000, 70_000_000, 10.5);
    let report = QualityBytesReport::new(vec![other_scale, first]);

    assert_eq!(report.pareto_frontier(), vec![first, other_scale]);
}

#[test]
fn net2wider_rejects_narrowing_and_misshaped_tensors() {
    assert_eq!(
        Net2WiderPlan::seeded(4, 3, 0),
        Err(GrowError::InvalidWidths {
            old_width: 4,
            new_width: 3,
        })
    );
    assert_eq!(
        Net2WiderPlan::seeded(0, 3, 0),
        Err(GrowError::InvalidWidths {
            old_width: 0,
            new_width: 3,
        })
    );
    assert_eq!(
        Net2WiderPlan::seeded(1, NET2WIDER_MAX_REPLICATIONS_PER_SOURCE + 1, 0),
        Err(GrowError::UnsafeMultiplicity {
            source: 0,
            copies: NET2WIDER_MAX_REPLICATIONS_PER_SOURCE + 1,
            maximum: NET2WIDER_MAX_REPLICATIONS_PER_SOURCE,
        })
    );

    let plan = Net2WiderPlan::seeded(3, 5, 0).unwrap();
    assert_eq!(
        plan.expand_incoming_rows(&[0.0; 5], 2),
        Err(GrowError::ShapeMismatch {
            tensor: "incoming projection",
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(
        plan.expand_outgoing_columns(&[0.0; 5], 2),
        Err(GrowError::ShapeMismatch {
            tensor: "outgoing projection",
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(
        plan.expand_hidden_vector(&[0.0; 2]),
        Err(GrowError::ShapeMismatch {
            tensor: "hidden vector",
            expected: 3,
            actual: 2,
        })
    );
}

#[test]
fn equal_width_plan_is_an_identity_transform() {
    let incoming = [0.2, -0.3, -0.4, 0.25, 0.7, 0.15]; // [3, 2]
    let hidden = [0.1, -0.05, 0.3];
    let outgoing = [0.4, -0.25, 0.15, -0.35, 0.2, 0.55]; // [2, 3]
    let plan = Net2WiderPlan::seeded(3, 3, u64::MAX).unwrap();

    assert_eq!(plan.algorithm(), NET2WIDER_ALGORITHM_V1);
    assert_eq!(plan.split_denominator_log2(), None);
    assert_eq!(plan.split_numerators(), None);
    assert_eq!(plan.expand_incoming_rows(&incoming, 2).unwrap(), incoming);
    assert_eq!(plan.expand_hidden_vector(&hidden).unwrap(), hidden);
    assert_eq!(
        plan.expand_outgoing_columns(&outgoing, 2).unwrap(),
        outgoing
    );
}
