//! Explicit small-device correctness tests, never silent physical skips.
#![cfg(feature = "cuda")]

use tritium_core::Trit;
use tritium_cuda::{CudaBackend, QteaCudaAllocationLedger, QteaCudaLimits};
use tritium_format::qtea::*;

fn wire_limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 4096,
    }
}
fn caps(input: usize, output: usize) -> QteaCudaLimits {
    QteaCudaLimits {
        max_device_staging_bytes: ((input + output) * 4).max(1) as u64,
        max_host_staging_bytes: (output * 4).max(1),
    }
}
fn fixture(rows: usize, columns: usize, group: usize, ratio: f64, mode: u8) -> QteaTensor {
    let meta = QteaMetadata {
        rows,
        columns,
        recipe: QteaRecipe {
            group_size: group,
            salient_ratio: ratio,
            delta_coefficient: 0.96,
            fit_iterations: 1,
            rescale_iterations: 2,
            damping: 0.007,
            decay: 1.,
            relative_pivot_floor: 1e-7,
        },
        bindings: QteaBindings {
            weights: [1; 32],
            gram: [2; 32],
            source_model: [3; 32],
            activation_cache: [4; 32],
            token_stream: [5; 32],
        },
    };
    let groups = columns / group;
    let trits = (0..rows * columns)
        .map(|i| {
            if mode == 1 {
                Trit::ZERO
            } else {
                [Trit::NEG, Trit::ZERO, Trit::POS][i % 3]
            }
        })
        .collect::<Vec<_>>();
    let alpha = (0..groups)
        .map(|g| {
            (0..rows)
                .map(|r| match mode {
                    1 => 1.,
                    2 => f32::from_bits(1 + (r % 4) as u32),
                    _ => 0.12345 + (g * rows + r) as f32 * 0.01731,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let beta = (0..groups)
        .map(|g| {
            (0..rows)
                .map(|r| {
                    if mode != 0 {
                        -0.0
                    } else {
                        [0.25, -0.3723, 0., -0.][(r + g) % 4]
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let multipliers = (0..columns)
        .map(|c| {
            if mode != 0 {
                1.
            } else {
                0.8173 + c as f32 * 0.01473
            }
        })
        .collect::<Vec<_>>();
    let order = (0..columns).rev().collect::<Vec<_>>();
    let salient = if ratio == 0. {
        0
    } else {
        (groups as f64 * ratio).round_ties_even().max(1.) as usize * group
    };
    let mut selected = order[..salient].to_vec();
    selected.sort_unstable();
    let indices = selected
        .iter()
        .map(|&c| {
            (0..rows / 4)
                .map(|r| ((r + c) % 4) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let finite_codes = (0u8..=255).filter(|c| c & 127 != 127).collect::<Vec<_>>();
    let codes = selected
        .iter()
        .map(|&c| {
            (0..rows / 4)
                .map(|r| finite_codes[(c * (rows / 4) + r) % finite_codes.len()])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    QteaTensor::from_parts(
        meta,
        &trits,
        &multipliers,
        &order,
        (0..groups).map(|g| QteaGroupRef {
            alpha: &alpha[g],
            beta: &beta[g],
        }),
        selected
            .iter()
            .enumerate()
            .map(|(i, &column)| QteaResidualRef {
                column,
                scale_multiplier: if mode == 2 { f32::MAX } else { 1.731 },
                indices: &indices[i],
                codes: &codes[i],
            }),
        wire_limits(),
    )
    .unwrap()
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn check_ledger(ledger: QteaCudaAllocationLedger, wire: usize, input: usize, output: usize) {
    assert_eq!(ledger.encoded_resident_bytes, wire as u64);
    assert_eq!(ledger.input_or_index_bytes, (input * 4) as u64);
    assert_eq!(ledger.output_bytes, (output * 4) as u64);
    assert_eq!(ledger.host_staging_capacity_bytes, output * 4);
    assert_eq!(
        ledger.requested_device_peak_bytes,
        (wire + (input + output) * 4) as u64
    );
    assert_eq!(ledger.dense_weight_bytes, 0);
}
fn parity(cuda: &CudaBackend, tensor: &QteaTensor) {
    // Cold validation from wire bytes, not a fitter or dense reconstruction.
    let cold = QteaTensorView::from_bytes(tensor.as_bytes(), wire_limits()).unwrap();
    let meta = cold.metadata();
    let wire = tensor.as_bytes().len();
    let gpu = cuda.upload_qtea(cold, wire as u64).unwrap();
    assert_eq!(gpu.shape(), (meta.rows, meta.columns));
    assert_eq!(gpu.content_digest(), cold.content_digest());
    assert_eq!(gpu.encoded_bytes(), wire);
    for batch in [0, 1, 3] {
        let input = (0..batch * meta.columns)
            .map(|i| ((i * 7 % 19) as f32 - 9.) * 0.1731)
            .collect::<Vec<_>>();
        let mut want = vec![0.; batch * meta.rows];
        let mut scratch = want.clone();
        tritium_cpu::qtea::matmul_into(cold, &input, batch, &mut want, &mut scratch).unwrap();
        let mut got = vec![123.; want.len()];
        let ledger = cuda
            .qtea_matmul_ordered_into(&gpu, &input, batch, &mut got, caps(input.len(), want.len()))
            .unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "matmul {}x{} batch{batch}",
            meta.rows,
            meta.columns
        );
        check_ledger(ledger, wire, input.len(), want.len());
    }
    for ids in [
        vec![],
        vec![0],
        vec![meta.rows as u32 - 1, 0, 0, 1],
        (0..meta.rows as u32).rev().collect(),
    ] {
        let mut want = vec![0.; ids.len() * meta.columns];
        let mut scratch = want.clone();
        tritium_cpu::qtea::gather_into(cold, &ids, &mut want, &mut scratch).unwrap();
        let mut got = vec![123.; want.len()];
        let ledger = cuda
            .qtea_gather_into(&gpu, &ids, &mut got, caps(ids.len(), want.len()))
            .unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "gather {}x{}",
            meta.rows,
            meta.columns
        );
        check_ledger(ledger, wire, ids.len(), want.len());
    }
}

#[test]
#[ignore = "explicit physical CUDA parity; requires device, never silently skips"]
fn direct_packed_qtea_matches_cpu_bits_on_device() {
    let cuda = CudaBackend::new(0).expect("physical CUDA required");
    for (rows, columns, group) in [
        (4, 2, 2),
        (8, 6, 3),
        (12, 33, 3),
        (36, 130, 2),
        (20, 128, 4),
    ] {
        for ratio in [0., 0.25, 1.] {
            parity(&cuda, &fixture(rows, columns, group, ratio, 0));
        }
    }
    // Contains every finite E4M3FN code and every sparse position. Both the
    // base-zero fixture and tiny subnormal base/residuals are bit-compared.
    parity(&cuda, &fixture(1024, 2, 2, 1., 1));
    parity(&cuda, &fixture(1024, 2, 2, 1., 2));
}

#[test]
#[ignore = "explicit physical CUDA ordered-cancellation regression; no silent skip"]
fn ordered_cancellation_crosses_warp_product_chunks_without_reassociation() {
    let cuda = CudaBackend::new(0).expect("physical CUDA required");
    let meta = fixture(4, 34, 2, 0., 0).view().metadata();
    let tensor = QteaTensor::from_parts(
        meta,
        &[Trit::POS; 4 * 34],
        &[1.; 34],
        &(0..34).collect::<Vec<_>>(),
        std::iter::repeat_n(
            QteaGroupRef {
                alpha: &[1.; 4],
                beta: &[0.; 4],
            },
            17,
        ),
        std::iter::empty(),
        wire_limits(),
    )
    .unwrap();
    let mut input = vec![0.; 3 * 34];
    input[0] = 1e20;
    input[1] = 1.;
    input[2] = -1e20;
    input[33] = 1.;
    input[34] = 1.;
    input[34 + 31] = 1e20;
    input[34 + 32] = -1e20;
    input[34 + 33] = 1.;
    input[68] = -0.;
    input[69] = f32::from_bits(1);
    input[100] = f32::from_bits(2);
    let gpu = cuda
        .upload_qtea(tensor.view(), tensor.as_bytes().len() as u64)
        .unwrap();
    let mut want = [0.; 12];
    tritium_cpu::qtea::matmul_into(tensor.view(), &input, 3, &mut want, &mut [0.; 12]).unwrap();
    assert_eq!(&want[..8], &[1.; 8]);
    let mut got = [77.; 12];
    cuda.qtea_matmul_ordered_into(&gpu, &input, 3, &mut got, caps(input.len(), 12))
        .unwrap();
    assert_eq!(bits(&got), bits(&want));
    // Finite products but overflowing intermediate sum must still refuse.
    input[..34].fill(0.);
    input[0] = f32::MAX;
    input[1] = f32::MAX;
    input[2] = -f32::MAX;
    got.fill(77.);
    assert!(
        cuda.qtea_matmul_ordered_into(&gpu, &input, 3, &mut got, caps(input.len(), 12))
            .is_err()
    );
    assert_eq!(got, [77.; 12]);
}

#[test]
#[ignore = "explicit physical CUDA refusal checks; requires device, never silently skips"]
fn direct_packed_qtea_refusals_are_atomic_on_device() {
    let cuda = CudaBackend::new(0).expect("physical CUDA required");
    let tensor = fixture(8, 6, 3, 1., 0);
    let wire = tensor.as_bytes().len() as u64;
    assert!(cuda.upload_qtea(tensor.view(), 0).is_err());
    assert!(cuda.upload_qtea(tensor.view(), wire - 1).is_err());
    let gpu = cuda.upload_qtea(tensor.view(), wire).unwrap();
    let limits = caps(6, 8);
    let mut output = [77.; 8];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let input = [value; 6];
        assert!(
            cuda.qtea_matmul_ordered_into(&gpu, &input, 1, &mut output, limits)
                .is_err()
        );
        assert_eq!(output, [77.; 8]);
    }
    assert!(
        cuda.qtea_matmul_ordered_into(&gpu, &[0.; 5], 1, &mut output, limits)
            .is_err()
    );
    assert!(
        cuda.qtea_matmul_ordered_into(&gpu, &[0.; 6], 2, &mut output, limits)
            .is_err()
    );
    for limit in [
        QteaCudaLimits {
            max_device_staging_bytes: limits.max_device_staging_bytes - 1,
            ..limits
        },
        QteaCudaLimits {
            max_host_staging_bytes: 31,
            ..limits
        },
        QteaCudaLimits {
            max_device_staging_bytes: 0,
            ..limits
        },
        QteaCudaLimits {
            max_host_staging_bytes: 0,
            ..limits
        },
    ] {
        assert!(
            cuda.qtea_matmul_ordered_into(&gpu, &[1.; 6], 1, &mut output, limit)
                .is_err()
        );
        assert_eq!(output, [77.; 8]);
    }
    let mut gathered = [77.; 6];
    assert!(
        cuda.qtea_gather_into(&gpu, &[8], &mut gathered, caps(1, 6))
            .is_err()
    );
    assert!(
        cuda.qtea_gather_into(&gpu, &[u32::MAX], &mut gathered, caps(1, 6))
            .is_err()
    );
    assert!(
        cuda.qtea_gather_into(&gpu, &[], &mut gathered, caps(1, 6))
            .is_err()
    );
    for limit in [
        QteaCudaLimits {
            max_device_staging_bytes: 27,
            ..caps(1, 6)
        },
        QteaCudaLimits {
            max_host_staging_bytes: 23,
            ..caps(1, 6)
        },
        QteaCudaLimits {
            max_device_staging_bytes: 0,
            ..caps(1, 6)
        },
        QteaCudaLimits {
            max_host_staging_bytes: 0,
            ..caps(1, 6)
        },
    ] {
        assert!(
            cuda.qtea_gather_into(&gpu, &[0], &mut gathered, limit)
                .is_err()
        );
        assert_eq!(gathered, [77.; 6]);
    }
    assert_eq!(output, [77.; 8]);
    assert_eq!(gathered, [77.; 6]);
    // Success after refusals proves the ordinary context remains usable.
    cuda.qtea_matmul_ordered_into(&gpu, &[0.; 6], 1, &mut output, limits)
        .unwrap();
    cuda.qtea_gather_into(&gpu, &[0], &mut gathered, caps(1, 6))
        .unwrap();
}

#[test]
#[ignore = "explicit physical CUDA plus caller-supplied eight admitted QTEA cases"]
fn cold_admitted_native_fits_match_cpu_on_device() {
    let directory = std::env::var_os("TRITIUM_QTEA_CUDA_CASES_DIR")
        .expect("explicit immutable admitted corpus directory required");
    let cuda = CudaBackend::new(0).expect("physical CUDA required");
    for case in 0..8 {
        let path = std::path::Path::new(&directory).join(format!("case-{case:03}.qtea"));
        let mut file = std::fs::File::open(path).unwrap();
        let tensor = QteaTensor::read_from(&mut file, wire_limits()).unwrap();
        parity(&cuda, &tensor);
    }
}
