#![cfg(feature = "cuda")]

use half::f16;
use tritium_cuda::{
    CudaBackend,
    train::{
        CheckpointPolicy, DevicePackedSaltWeight, DeviceTape, DeviceTensor, GradientLeafBinding,
        PackedSaltComputePolicy,
    },
};
use tritium_format::{PackedTrainingSaltSnapshot, TernaryStructure, TrainingSaltPlane};
use tritium_train::ops::{dense, embed, loss};

type SemanticPlane = (Vec<i8>, Vec<f16>);

fn backend() -> CudaBackend {
    CudaBackend::new(0).expect("CUDA feature lane requires device 0")
}

fn snapshot(
    rows: usize,
    cols: usize,
    group_size: usize,
    structure: TernaryStructure,
    planes: &[SemanticPlane],
) -> PackedTrainingSaltSnapshot {
    let borrowed = planes
        .iter()
        .map(|(trits, scales)| TrainingSaltPlane::new(trits, scales))
        .collect::<Vec<_>>();
    PackedTrainingSaltSnapshot::pack(rows, cols, group_size, structure, &borrowed).unwrap()
}

fn decode(rows: usize, cols: usize, group_size: usize, planes: &[SemanticPlane]) -> Vec<f32> {
    let groups_per_row = cols.div_ceil(group_size);
    let mut decoded = vec![0.0; rows * cols];
    for (trits, scales) in planes {
        for row in 0..rows {
            for col in 0..cols {
                decoded[row * cols + col] += f32::from(trits[row * cols + col])
                    * f32::from(scales[row * groups_per_row + col / group_size]);
            }
        }
    }
    decoded
}

fn forward(backend: &CudaBackend, weight: &DevicePackedSaltWeight, input: &[f32]) -> Vec<f32> {
    let mut tape = DeviceTape::new(backend, weight.rows()).unwrap();
    let input = tape.leaf(input).unwrap();
    let gradient_leaf = tape.gradient_leaf(weight.rows() * weight.cols()).unwrap();
    let output = tape.salt_matmul(input, gradient_leaf, weight, 1).unwrap();
    tape.value(output).unwrap()
}

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    let (index, error) = got
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&got, &expected))| (index, (got - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
    assert!(
        error <= tolerance,
        "maximum absolute error {error} at index {index}: got {}, expected {}",
        got[index],
        expected[index]
    );
}

#[test]
fn packed_backward_visits_bounded_contiguous_host_gradient_blocks() {
    let backend = backend();
    let semantic = vec![(vec![1, 0, 0, 1], vec![f16::ONE; 4])];
    let snapshot = snapshot(2, 2, 1, TernaryStructure::Dense, &semantic);
    let packed = DevicePackedSaltWeight::from_snapshot(&backend, &snapshot).unwrap();
    let target = DeviceTensor::upload(&backend, &[1.0, 0.0]).unwrap();
    let mut tape = DeviceTape::new(&backend, 2).unwrap();
    let input = tape.leaf(&[1.0, 2.0]).unwrap();
    let master = tape.gradient_leaf(4).unwrap();
    let logits = tape.salt_matmul(input, master, &packed, 1).unwrap();
    let mut visited = Vec::new();

    let report = tape
        .xent_backward_visit_host_gradient_blocks(
            logits,
            &target,
            1,
            2,
            &[GradientLeafBinding {
                leaf_id: master,
                parameter_index: 0,
            }],
            &[4],
            3,
            1,
            |parameter_index, offset, total, gradient| {
                assert_eq!(parameter_index, 0);
                assert_eq!(total, 4);
                visited.push((offset, gradient.to_vec()));
                Ok(())
            },
        )
        .unwrap();

    assert_eq!(
        visited
            .iter()
            .map(|(offset, block)| (*offset, block.len()))
            .collect::<Vec<_>>(),
        vec![(0, 3), (3, 1)]
    );
    let gradient = visited
        .iter()
        .flat_map(|(_, block)| block.iter().copied())
        .collect::<Vec<_>>();
    assert_close(
        &gradient,
        &[-0.731_058_6, -1.462_117_2, 0.731_058_6, 1.462_117_2],
        1e-6,
    );
    assert_eq!(report.emissions.len(), 1);
    assert_eq!(report.peak_host_gradient_elements, 3);
}

#[test]
fn compact_pv_weight_executes_grouped_additive_forward_without_dense_master() {
    let backend = backend();
    let semantic = vec![
        (
            vec![1, 1, 1, 1, -1, 0, 1, -1],
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(0.5),
                f16::from_f32(1.5),
            ],
        ),
        (
            vec![-1, 0, 1, 0, 1, 1, 0, -1],
            vec![
                f16::from_f32(0.5),
                f16::from_f32(0.25),
                f16::from_f32(2.0),
                f16::from_f32(0.5),
            ],
        ),
    ];
    let snapshot = snapshot(2, 4, 2, TernaryStructure::Dense, &semantic);
    let mut packed = DevicePackedSaltWeight::from_snapshot(&backend, &snapshot).unwrap();

    assert_eq!(
        forward(&backend, &packed, &[1.0, 2.0, 3.0, 4.0]),
        vec![17.25, 2.0]
    );
    assert_eq!(packed.group_size(), 2);
    assert_eq!(packed.structure(), TernaryStructure::Dense);
    assert_eq!(packed.resident_bytes(), 2 * 2 * 64 + 2 * 2 * 2 * 4);
    assert!(packed.repack_from_host(&backend, &[0.25; 8]).is_err());
    assert_eq!(
        forward(&backend, &packed, &[1.0, 2.0, 3.0, 4.0]),
        vec![17.25, 2.0]
    );
}

#[test]
fn compact_pv_update_is_transactional_and_preserves_s34_identity() {
    let backend = backend();
    let make = |trits, scale| {
        snapshot(
            1,
            4,
            4,
            TernaryStructure::S34,
            &[(trits, vec![f16::from_f32(scale)])],
        )
    };
    let mut packed =
        DevicePackedSaltWeight::from_snapshot(&backend, &make(vec![1, -1, 1, 0], 1.0)).unwrap();
    assert_eq!(forward(&backend, &packed, &[1.0; 4]), vec![1.0]);

    packed
        .update_from_snapshot(&backend, &make(vec![1, -1, 0, 1], 2.0))
        .unwrap();
    assert_eq!(forward(&backend, &packed, &[1.0; 4]), vec![2.0]);
    assert_eq!(packed.structure(), TernaryStructure::S34);

    let wrong_geometry = snapshot(
        1,
        8,
        4,
        TernaryStructure::S34,
        &[(vec![1, -1, 0, 1, 1, -1, 0, 1], vec![f16::ONE; 2])],
    );
    assert!(
        packed
            .update_from_snapshot(&backend, &wrong_geometry)
            .is_err()
    );
    assert_eq!(forward(&backend, &packed, &[1.0; 4]), vec![2.0]);
}

#[test]
fn g128_packed_weight_matches_cpu_decode_across_tiled_forward_vjp_and_embedding() {
    let backend = backend();
    let (rows, cols, group_size, planes, batch) = (37usize, 321usize, 128usize, 2usize, 3usize);
    let groups_per_row = cols.div_ceil(group_size);
    let semantic = (0..planes)
        .map(|plane| {
            let trits = (0..rows * cols)
                .map(|index| match (index + plane * 2 + index / cols) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect();
            let scales = (0..rows * groups_per_row)
                .map(|index| f16::from_f32((1 + (index + plane) % 5) as f32 / 64.0))
                .collect();
            (trits, scales)
        })
        .collect::<Vec<_>>();
    let decoded = decode(rows, cols, group_size, &semantic);
    let snapshot = snapshot(rows, cols, group_size, TernaryStructure::Dense, &semantic);
    let packed = DevicePackedSaltWeight::from_snapshot(&backend, &snapshot).unwrap();
    let input = (0..batch * cols)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) / 32.0)
        .collect::<Vec<_>>();
    let expected_forward = dense::forward(&input, &decoded, batch, rows, cols);
    let mut target = vec![0.0; batch * rows];
    for row in 0..batch {
        target[row * rows + (row * 11 + 5) % rows] = 1.0;
    }

    let mut tape = DeviceTape::new(&backend, rows).unwrap();
    let input_id = tape.leaf(&input).unwrap();
    let master = tape.gradient_leaf(rows * cols).unwrap();
    let logits = tape.salt_matmul(input_id, master, &packed, batch).unwrap();
    assert_close(&tape.value(logits).unwrap(), &expected_forward, 2e-5);

    // Fast path must retain per-group scales when K spans multiple groups.
    let mut fast_tape = DeviceTape::new_with_policies(
        &backend,
        rows,
        CheckpointPolicy::KeepAll,
        PackedSaltComputePolicy::Fast,
    )
    .unwrap();
    let fast_input = fast_tape.leaf(&input).unwrap();
    let fast_master = fast_tape.gradient_leaf(rows * cols).unwrap();
    let fast_logits = fast_tape
        .salt_matmul(fast_input, fast_master, &packed, batch)
        .unwrap();
    assert_close(
        &fast_tape.value(fast_logits).unwrap(),
        &expected_forward,
        2e-5,
    );
    let target_device = DeviceTensor::upload(&backend, &target).unwrap();
    let device_loss = tape
        .softmax_xent_value(logits, &target_device, batch, rows)
        .unwrap();
    let expected_loss = loss::softmax_xent_forward(&expected_forward, &target, batch, rows)[0];
    assert!((device_loss - expected_loss).abs() <= 2e-5);
    let got_input_gradient = tape
        .xent_backward(logits, &target, batch, rows, &[input_id])
        .unwrap()
        .remove(0);
    let seed = loss::softmax_xent_vjp(&expected_forward, &target, batch, rows, &[1.0]).remove(0);
    let expected_input_gradient = dense::vjp(&input, &decoded, batch, rows, cols, &seed).remove(0);
    assert_close(&got_input_gradient, &expected_input_gradient, 2e-5);
    let fast_target_device = DeviceTensor::upload(&backend, &target).unwrap();
    fast_tape
        .softmax_xent_value(fast_logits, &fast_target_device, batch, rows)
        .unwrap();
    let fast_input_gradient = fast_tape
        .xent_backward(fast_logits, &target, batch, rows, &[fast_input])
        .unwrap()
        .remove(0);
    assert_close(&fast_input_gradient, &expected_input_gradient, 2e-5);

    let tokens_i32 = [36, 0, 17, 36];
    let tokens_u32 = [36, 0, 17, 36];
    let mut tape = DeviceTape::new(&backend, rows).unwrap();
    let master = tape.gradient_leaf(rows * cols).unwrap();
    let embedding = tape.salt_embed(master, &packed, &tokens_i32).unwrap();
    assert_close(
        &tape.value(embedding).unwrap(),
        &embed::gather_forward(&decoded, &tokens_u32, cols),
        0.0,
    );
}

#[test]
fn g256_fast_forward_preserves_multiple_scale_groups() {
    let backend = backend();
    let (rows, cols, group_size, planes, batch) = (19usize, 513usize, 256usize, 3usize, 4usize);
    let groups_per_row = cols.div_ceil(group_size);
    let semantic = (0..planes)
        .map(|plane| {
            let trits = (0..rows * cols)
                .map(|index| match (index + plane + index / cols) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                })
                .collect();
            let scales = (0..rows * groups_per_row)
                .map(|index| f16::from_f32((1 + (index + plane) % 7) as f32 / 32.0))
                .collect();
            (trits, scales)
        })
        .collect::<Vec<_>>();
    let decoded = decode(rows, cols, group_size, &semantic);
    let snapshot = snapshot(rows, cols, group_size, TernaryStructure::Dense, &semantic);
    let packed = DevicePackedSaltWeight::from_snapshot(&backend, &snapshot).unwrap();
    let input = (0..batch * cols)
        .map(|index| ((index * 13 % 31) as f32 - 15.0) / 32.0)
        .collect::<Vec<_>>();
    let expected = dense::forward(&input, &decoded, batch, rows, cols);
    let mut tape = DeviceTape::new_with_policies(
        &backend,
        rows,
        CheckpointPolicy::KeepAll,
        PackedSaltComputePolicy::Fast,
    )
    .unwrap();
    let input_id = tape.leaf(&input).unwrap();
    let master = tape.gradient_leaf(rows * cols).unwrap();
    let output = tape.salt_matmul(input_id, master, &packed, batch).unwrap();
    assert_close(&tape.value(output).unwrap(), &expected, 2e-5);

    // Batch >= 4 selects tiled fast grad-A. Keep this check coupled to the
    // grouped-scale forward assertion so both fast kernels see same geometry.
    let mut target = vec![0.0; batch * rows];
    for row in 0..batch {
        target[row * rows + (row * 7 + 3) % rows] = 1.0;
    }
    let target_device = DeviceTensor::upload(&backend, &target).unwrap();
    tape.softmax_xent_value(output, &target_device, batch, rows)
        .unwrap();
    let got_gradient = tape
        .xent_backward(output, &target, batch, rows, &[input_id])
        .unwrap()
        .remove(0);
    let seed = loss::softmax_xent_vjp(&expected, &target, batch, rows, &[1.0]).remove(0);
    let expected_gradient = dense::vjp(&input, &decoded, batch, rows, cols, &seed).remove(0);
    assert_close(&got_gradient, &expected_gradient, 2e-5);
}
