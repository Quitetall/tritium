use std::io::Cursor;
use tritium_cpu::qtea::{QteaCpuError, matmul_into};
use tritium_format::qtea::{QteaLimits, QteaTensor};
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};

fn fitted(rows: usize, columns: usize, width: usize, ratio: f64) -> TensorFit {
    let weights = (0..rows)
        .flat_map(|r| {
            (0..columns).map(move |c| {
                (((r + 1) * (c + 3)) % 17) as f32 * 0.35 - 2.8 + c as f32 * 0.19 + r as f32 * 0.07
            })
        })
        .collect::<Vec<_>>();
    let mut input = vec![0.; columns * columns];
    for c in 0..columns {
        input[c * columns + c] = 1. + c as f32 * 0.05;
    }
    input[1] = 0.4;
    input[columns + 1] = 0.8;
    let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
    let mut builder = InputGramAccumulator::new_bound(columns, source, 0).unwrap();
    builder
        .accumulate_batch(&input, columns, None, None)
        .unwrap();
    TensorFit::fit(
        &weights,
        rows,
        columns,
        &builder.finish().unwrap(),
        TensorFitConfig {
            group_size: width,
            salient_ratio: ratio,
            ..TensorFitConfig::default()
        },
        TensorFit::required_work_bytes(rows, columns).unwrap(),
    )
    .unwrap()
}
fn limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 4096,
    }
}

#[test]
fn actual_fit_export_cold_reopen_and_packed_cpu_match_independent_dense_forward() {
    for (rows, columns, width) in [(4, 8, 2), (8, 12, 4)] {
        for ratio in [0., 0.25, 1.] {
            let fit = fitted(rows, columns, width, ratio);
            let encoded = fit.to_qtea_tensor(limits()).unwrap();
            let mut file = Vec::new();
            encoded.write_to(&mut file).unwrap();
            let cold = QteaTensor::read_from(&mut Cursor::new(file), limits()).unwrap();
            let view = cold.view();
            assert_eq!(view.recipe_digest(), fit.recipe_digest());
            assert_eq!(view.metadata().bindings.weights, fit.weights_digest());
            assert_eq!(view.metadata().bindings.gram, fit.gram_digest());
            assert_eq!(
                view.fitting_order().collect::<Vec<_>>(),
                fit.fitting_order()
            );
            assert_eq!(view.byte_ledger().total_bytes, cold.as_bytes().len());
            let mut dense = vec![0.; rows * columns];
            fit.decode_into(&mut dense).unwrap();
            drop(fit);
            drop(encoded); // Cold encoded bytes alone own the runtime weights.
            for batch in [0, 1, 3] {
                let input = (0..batch * columns)
                    .map(|i| (i % 7) as f32 * 0.125 - 0.375)
                    .collect::<Vec<_>>();
                let mut expected = vec![0.; batch * rows];
                for b in 0..batch {
                    for r in 0..rows {
                        for c in 0..columns {
                            expected[b * rows + r] +=
                                input[b * columns + c] * dense[r * columns + c];
                        }
                    }
                }
                let mut output = vec![123.; batch * rows];
                let mut scratch = vec![456.; batch * rows];
                matmul_into(view, &input, batch, &mut output, &mut scratch).unwrap();
                assert_eq!(
                    output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn input_shape_nonfinite_and_late_overflow_leave_output_unchanged() {
    let fit = fitted(4, 8, 2, 1.);
    let encoded = fit.to_qtea_tensor(limits()).unwrap();
    let view = encoded.view();
    let mut output = [77.; 8];
    let mut scratch = [99.; 8];
    assert_eq!(
        matmul_into(view, &[0.; 15], 2, &mut output, &mut scratch),
        Err(QteaCpuError::Shape)
    );
    assert_eq!(output, [77.; 8]);
    assert_eq!(scratch, [99.; 8]);
    let mut bad = [0.; 16];
    bad[15] = f32::NAN;
    assert_eq!(
        matmul_into(view, &bad, 2, &mut output, &mut scratch),
        Err(QteaCpuError::NonFiniteInput { index: 15 })
    );
    assert_eq!(output, [77.; 8]);
    assert_eq!(scratch, [99.; 8]);
    let mut overflow = [0.; 16];
    overflow[8] = f32::MAX;
    assert_eq!(
        matmul_into(view, &overflow, 2, &mut output, &mut scratch),
        Err(QteaCpuError::ArithmeticOverflow)
    );
    assert_eq!(output, [77.; 8]);
    assert_eq!(&scratch[..4], &[0.; 4]);
    assert_eq!(
        matmul_into(view, &[], usize::MAX, &mut [], &mut []),
        Err(QteaCpuError::Shape)
    );
}
