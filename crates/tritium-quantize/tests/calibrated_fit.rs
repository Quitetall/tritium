//! Numerical consumers use actual cache/reducer/fitter interfaces, not copied solver math.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::reference_ternary_matmul;
use tritium_evidence::original_source_id;
use tritium_format::AdditiveTensor;
use tritium_quantize::CurvatureError;
use tritium_quantize::engine::{
    CalibratedFitError, CalibratedTensor, CalibrationBatch, CalibrationCacheBuilder,
    CalibrationError, CalibrationInput, CalibrationOutputBatch, DirectFitError,
    fit_calibrated_levels, fit_direct_levels,
};
use tritium_schema::*;

const LARGE_COLS: usize = 512;
const FULL_MATRIX_BYTES: usize = LARGE_COLS * LARGE_COLS * 8;
static COUNT_MATRICES: AtomicBool = AtomicBool::new(false);
static MATRIX_REQUESTS: AtomicUsize = AtomicUsize::new(0);
struct CountingAllocator;
fn record_allocation(bytes: usize) {
    if bytes == FULL_MATRIX_BYTES && COUNT_MATRICES.load(Ordering::Relaxed) {
        MATRIX_REQUESTS.fetch_add(1, Ordering::Relaxed);
    }
}
// SAFETY: all allocation requests are forwarded unchanged to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplies a valid allocation layout.
        let output = unsafe { System.alloc(layout) };
        if !output.is_null() {
            record_allocation(layout.size());
        }
        output
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller supplies the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplies the original allocation and valid requested size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() {
            record_allocation(size);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn source(cols: usize) -> (QuantSourceTensor, Vec<f32>, OriginalSourceId) {
    let tensor = QuantSourceTensor {
        name: "weight".into(),
        shape: vec![2, cols as u64],
        precision: ScalePrecision::F32,
    };
    let row: Vec<f32> = (0..cols)
        .map(|i| ((i * 29 % 101) as f32 - 50.) / 37.)
        .collect();
    let weights: Vec<f32> = row.iter().chain(&row).copied().collect();
    let raw: Vec<u8> = weights
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let model = original_source_id(
        "calibrated-fit-fixture",
        &[QuantSourceBinding {
            source: tensor.clone(),
            content: BlobId::from_bytes(*blake3::hash(&raw).as_bytes()),
        }],
        &[],
    )
    .unwrap();
    (tensor, weights, model)
}

#[derive(Clone, Copy)]
enum InputFixture {
    Correlated,
    PositiveCorrelated,
    Diagonal,
    ZeroFisher,
}
fn factors(
    tensor: &QuantSourceTensor,
    model: OriginalSourceId,
    partition: CalibrationPartition,
    consumer: CalibrationConsumer,
    output: CalibrationOutput,
    damping: f64,
    fixture: InputFixture,
) -> CalibratedTensor {
    let diagonal = matches!(fixture, InputFixture::Diagonal);
    let positive_correlated = matches!(fixture, InputFixture::PositiveCorrelated);
    let zero_fisher = matches!(fixture, InputFixture::ZeroFisher);
    let cols = tensor.shape[1] as usize;
    let samples = if diagonal || positive_correlated {
        cols
    } else {
        8
    };
    let input: Vec<f32> = (0..samples * cols)
        .map(|i| {
            if diagonal {
                f32::from(i / cols == i % cols)
            } else if positive_correlated {
                if i / cols == i % cols {
                    2.
                } else {
                    (((i / cols + 1) * (i % cols + 3) % 17) as f32 - 8.) / 1000.
                }
            } else {
                (((i / cols + 1) * (i % cols + 3) % 17) as f32 - 8.) / 7.
            }
        })
        .collect();
    let gradients: Vec<f32> = (0..samples)
        .flat_map(|_| if zero_fisher { [0., 0.] } else { [0.25, 3.] })
        .collect();
    let indices: Vec<usize> = (0..samples).map(|i| i % 2).collect();
    let indexed: Vec<f32> = indices
        .iter()
        .map(|i| {
            if zero_fisher {
                0.
            } else if *i == 0 {
                0.25
            } else {
                3.
            }
        })
        .collect();
    let tokens: Vec<u32> = (0..samples as u32).collect();
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor: tensor.clone(),
            model,
            consumer,
            output,
            sample_start: 5,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 1_000_000.max((samples * cols * 8) as u64),
                max_samples: 128.max(samples as u64),
                max_windows: 4,
            },
        },
        b"actual tokenizer fixture bytes",
        b"declared fixture objective",
    )
    .unwrap();
    builder
        .capture(
            &tokens,
            CalibrationBatch {
                samples,
                input: if consumer == CalibrationConsumer::Matmul {
                    CalibrationInput::Activations(&input)
                } else {
                    CalibrationInput::GatherIdentity
                },
                output: match output {
                    CalibrationOutput::None => CalibrationOutputBatch::None,
                    CalibrationOutput::DenseFisher => CalibrationOutputBatch::Dense(&gradients),
                    _ => CalibrationOutputBatch::Indexed {
                        indices: &indices,
                        factors: &indexed,
                    },
                },
                weights: None,
                mask: None,
            },
        )
        .unwrap();
    builder
        .finish()
        .unwrap()
        .collect(
            partition,
            damping,
            16_000_000.max((cols * cols * 128) as u64),
        )
        .unwrap()
}

fn joint() -> JointGroupFitOptions {
    JointGroupFitOptions {
        max_iterations: 8,
        ridge: 1e-8,
        em_restarts: 2,
        ridge_condition_limit: 1e6,
        softened_relay: false,
        modulated_relay: false,
    }
}
fn profiles() -> Vec<(DirectFitProfile, ScalePrecision)> {
    let ladder = |mode| {
        DirectFitProfile::ActivationAwareLadder(LadderFitOptions {
            grid: 8,
            damp: 0.01,
            search_sweeps: 1,
            scale_refit: mode,
            decay: 0.6,
            decay_ramp: true,
        })
    };
    vec![
        (ladder(LadderScaleRefit::PostPass), ScalePrecision::F16),
        (ladder(LadderScaleRefit::InLoop), ScalePrecision::F16),
        (
            DirectFitProfile::JointIndependentGroups(joint()),
            ScalePrecision::F16,
        ),
        (
            DirectFitProfile::JointIndependentGroups(joint()),
            ScalePrecision::F32,
        ),
        (
            DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
                fitter: joint(),
                damp: 0.01,
                precision: WorkPrecision::F32Accum64,
                decay: 0.6,
                decay_ramp: true,
            }),
            ScalePrecision::F16,
        ),
        (
            DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
                fitter: joint(),
                damp: 0.01,
                precision: WorkPrecision::F64,
                decay: 0.6,
                decay_ramp: true,
            }),
            ScalePrecision::F32,
        ),
    ]
}
fn plan(
    cols: usize,
    basis: Basis,
    profile: DirectFitProfile,
    precision: ScalePrecision,
) -> DirectFitPlan {
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 2,
            cols: cols as u64,
            tile: ADDITIVE_TILE_SIZE,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                precision,
                relation: if matches!(profile, DirectFitProfile::ActivationAwareLadder(_)) {
                    PlaneRelation::Tied { num: 1, den: 3 }
                } else {
                    PlaneRelation::Free
                },
            },
            basis,
            transport: Transport::Raw,
        },
        profile,
        levels: vec![
            DirectFitLevel {
                name: "three".into(),
                planes: 3,
            },
            DirectFitLevel {
                name: "one".into(),
                planes: 1,
            },
        ],
        limits: DirectFitLimits {
            max_weights: 2 * cols as u64,
            max_metric_entries: (cols * cols) as u64,
            max_output_trits: 8 * cols as u64,
        },
    }
}
fn matmul(tensor: &AdditiveTensor, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.; tensor.layout().rows as usize];
    let mut scratch = vec![0.; input.len()];
    reference_ternary_matmul(
        input,
        &tensor.view().bind_matmul(),
        1,
        &mut scratch,
        &mut output,
    )
    .unwrap();
    output
}

#[test]
fn every_profile_uses_actual_row_kfac_and_fresh_original_level_fits() {
    for (cols, basis) in [
        (35, Basis::Identity),
        (64, Basis::Hadamard { block: 64 }),
        (
            64,
            Basis::SignedRht {
                block: 16,
                seed: 42,
                domain: 9,
            },
        ),
    ] {
        let (original, weights, model) = source(cols);
        let calibration = factors(
            &original,
            model,
            CalibrationPartition::Full,
            CalibrationConsumer::Matmul,
            CalibrationOutput::DenseFisher,
            0.125,
            InputFixture::Correlated,
        );
        assert_ne!(
            calibration.kfac_metric(0, 0).unwrap().metric().as_slice(),
            calibration.kfac_metric(0, 1).unwrap().metric().as_slice()
        );
        let before: Vec<u32> = weights.iter().map(|v| v.to_bits()).collect();
        let binding = calibration.factors().clone();
        let activation: Vec<f32> = (0..cols).map(|i| (i as f32 - 7.) / 23.).collect();
        for (profile, precision) in profiles() {
            for codec in [PlaneCodec::D2, PlaneCodec::B3] {
                let mut request = plan(cols, basis, profile, precision);
                request.layout.codec = codec;
                let fitted =
                    fit_calibrated_levels(&request, &original, model, &weights, &calibration)
                        .unwrap();
                assert_eq!(
                    weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    before
                );
                assert_eq!(calibration.factors(), &binding);
                let mut row_plan = request.clone();
                row_plan.layout.rows = 1;
                for row in 0..2 {
                    let actual_metric = calibration.kfac_metric(0, row).unwrap();
                    let expected = fit_direct_levels(
                        &row_plan,
                        &weights[row * cols..(row + 1) * cols],
                        Some(actual_metric.metric().as_slice()),
                    )
                    .unwrap();
                    for (whole, single) in fitted.iter().zip(expected) {
                        assert_eq!(whole.level, single.level);
                        // The actual core execution interface checks both plane/scalar row installation and declared basis.
                        assert_eq!(
                            matmul(&whole.tensor, &activation)[row].to_bits(),
                            matmul(&single.tensor, &activation)[0].to_bits()
                        );
                        let encoded = whole.tensor.encode_payload().unwrap();
                        assert_eq!(
                            AdditiveTensor::decode_payload(
                                whole.tensor.layout(),
                                whole.tensor.plane_count(),
                                &encoded
                            )
                            .unwrap(),
                            whole.tensor
                        );
                        let mut one_level = row_plan.clone();
                        one_level.levels.retain(|level| level.name == single.level);
                        assert_eq!(
                            fit_direct_levels(
                                &one_level,
                                &weights[row * cols..(row + 1) * cols],
                                Some(actual_metric.metric().as_slice())
                            )
                            .unwrap(),
                            vec![single]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn full_and_explicit_grouped_ragged_inputs_agree_for_actual_diagonal_statistics() {
    let (original, weights, model) = source(35);
    let request = plan(
        35,
        Basis::Identity,
        DirectFitProfile::JointIndependentGroups(joint()),
        ScalePrecision::F32,
    );
    for (consumer, output) in [
        (CalibrationConsumer::Matmul, CalibrationOutput::None),
        (CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher),
        (
            CalibrationConsumer::Matmul,
            CalibrationOutput::IndexedFisher,
        ),
        (
            CalibrationConsumer::GatherIdentity,
            CalibrationOutput::IndexedFisher,
        ),
    ] {
        let damp = if output == CalibrationOutput::None {
            0.
        } else {
            0.125
        };
        let full = factors(
            &original,
            model,
            CalibrationPartition::Full,
            consumer,
            output,
            damp,
            InputFixture::Diagonal,
        );
        let groups = factors(
            &original,
            model,
            CalibrationPartition::Groups { columns: 16 },
            consumer,
            output,
            damp,
            InputFixture::Diagonal,
        );
        assert_eq!(
            groups
                .inputs()
                .iter()
                .map(|i| i.dimension())
                .collect::<Vec<_>>(),
            [16, 16, 3]
        );
        assert_eq!(
            fit_calibrated_levels(&request, &original, model, &weights, &full).unwrap(),
            fit_calibrated_levels(&request, &original, model, &weights, &groups).unwrap()
        );
    }
}

#[test]
fn grouped_metrics_match_explicit_dense_fits_across_profiles_bases_and_ragged_groups() {
    for (cols, basis) in [
        (35, Basis::Identity),
        (64, Basis::Hadamard { block: 64 }),
        (
            64,
            Basis::SignedRht {
                block: 16,
                seed: 42,
                domain: 9,
            },
        ),
    ] {
        let (original, weights, model) = source(cols);
        for columns in [13, 32] {
            for output in [
                CalibrationOutput::None,
                CalibrationOutput::DenseFisher,
                CalibrationOutput::IndexedFisher,
            ] {
                let calibration = factors(
                    &original,
                    model,
                    CalibrationPartition::Groups { columns },
                    CalibrationConsumer::Matmul,
                    output,
                    if output == CalibrationOutput::None {
                        0.
                    } else {
                        0.125
                    },
                    InputFixture::PositiveCorrelated,
                );
                let factors_before = calibration.factors().clone();
                // Assemble the original dense representation from actual public
                // factor values. No reduction, basis or fitting math is copied.
                let mut dense_rows = Vec::new();
                for row in 0..2 {
                    let mut dense = vec![0.; cols * cols];
                    let mut offset = 0;
                    for (group, input) in calibration.inputs().iter().enumerate() {
                        let kfac;
                        let values = if output == CalibrationOutput::None {
                            input.as_slice()
                        } else {
                            kfac = calibration.kfac_metric(group, row).unwrap();
                            kfac.metric().as_slice()
                        };
                        let width = input.dimension();
                        for local in 0..width {
                            dense[(offset + local) * cols + offset
                                ..(offset + local) * cols + offset + width]
                                .copy_from_slice(&values[local * width..(local + 1) * width]);
                        }
                        offset += width;
                    }
                    dense_rows.push(dense);
                }
                for (profile, precision) in profiles() {
                    for codec in [PlaneCodec::B3, PlaneCodec::D2] {
                        let mut request = plan(cols, basis, profile, precision);
                        request.layout.codec = codec;
                        let actual = fit_calibrated_levels(
                            &request,
                            &original,
                            model,
                            &weights,
                            &calibration,
                        )
                        .unwrap();
                        let mut row_plan = request.clone();
                        row_plan.layout.rows = 1;
                        for (row, dense) in dense_rows.iter().enumerate() {
                            let expected = fit_direct_levels(
                                &row_plan,
                                &weights[row * cols..(row + 1) * cols],
                                Some(dense),
                            )
                            .unwrap();
                            for (whole, single) in actual.iter().zip(expected) {
                                assert_eq!(whole.level, single.level);
                                for plane in 0..single.tensor.plane_count() as usize {
                                    assert_eq!(
                                        &whole.tensor.trits()[plane * 2 * cols + row * cols
                                            ..plane * 2 * cols + (row + 1) * cols],
                                        &single.tensor.trits()[plane * cols..(plane + 1) * cols],
                                    );
                                }
                                let units = cols.div_ceil(32);
                                for plane in 0..single.tensor.scales().len() / units {
                                    assert_eq!(
                                        whole.tensor.scales()[plane * 2 * units + row * units
                                            ..plane * 2 * units + (row + 1) * units]
                                            .iter()
                                            .map(|v| v.to_bits())
                                            .collect::<Vec<_>>(),
                                        single.tensor.scales()[plane * units..(plane + 1) * units]
                                            .iter()
                                            .map(|v| v.to_bits())
                                            .collect::<Vec<_>>(),
                                    );
                                }
                            }
                        }
                    }
                }
                assert_eq!(calibration.factors(), &factors_before);
            }
        }
    }
}

#[test]
fn grouped_independent_fit_does_not_allocate_a_full_input_matrix() {
    let (original, weights, model) = source(LARGE_COLS);
    let mut request = plan(
        LARGE_COLS,
        Basis::Identity,
        DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
            max_iterations: 2,
            em_restarts: 1,
            ..joint()
        }),
        ScalePrecision::F32,
    );
    request.layout.max_planes = 1;
    request.levels = vec![DirectFitLevel {
        name: "one".into(),
        planes: 1,
    }];
    // This is the only 512-column fixture in this integration-test binary.
    // Count exact full-matrix requests, not total allocator bytes or RSS.
    for columns in [13, 32] {
        for output in [CalibrationOutput::None, CalibrationOutput::DenseFisher] {
            let calibration = factors(
                &original,
                model,
                CalibrationPartition::Groups { columns },
                CalibrationConsumer::Matmul,
                output,
                if output == CalibrationOutput::None {
                    0.
                } else {
                    0.125
                },
                InputFixture::Diagonal,
            );
            let expected =
                fit_calibrated_levels(&request, &original, model, &weights, &calibration).unwrap();
            MATRIX_REQUESTS.store(0, Ordering::Relaxed);
            COUNT_MATRICES.store(true, Ordering::SeqCst);
            let actual = fit_calibrated_levels(&request, &original, model, &weights, &calibration);
            COUNT_MATRICES.store(false, Ordering::SeqCst);
            assert_eq!(actual.unwrap(), expected);
            let requests = MATRIX_REQUESTS.load(Ordering::Relaxed);
            eprintln!(
                "matrix_bytes={FULL_MATRIX_BYTES} calibration_group={columns} output={output:?} full_matrix_requests={requests}"
            );
            assert_eq!(requests, 0, "grouped metrics must remain structured");
        }
    }
}

#[test]
fn context_and_aggregate_limits_reject_before_fitting_without_fallback() {
    let (original, weights, model) = source(35);
    let calibration = factors(
        &original,
        model,
        CalibrationPartition::Full,
        CalibrationConsumer::Matmul,
        CalibrationOutput::DenseFisher,
        0.125,
        InputFixture::Correlated,
    );
    let request = plan(
        35,
        Basis::Identity,
        DirectFitProfile::JointIndependentGroups(joint()),
        ScalePrecision::F32,
    );
    assert!(matches!(
        fit_calibrated_levels(
            &request,
            &original,
            OriginalSourceId::from_bytes([9; 32]),
            &weights,
            &calibration
        ),
        Err(CalibratedFitError::Binding)
    ));
    for changed in [
        QuantSourceTensor {
            name: "another.weight".into(),
            ..original.clone()
        },
        QuantSourceTensor {
            precision: ScalePrecision::Bf16,
            ..original.clone()
        },
        QuantSourceTensor {
            shape: vec![1, 70],
            ..original.clone()
        },
    ] {
        assert!(matches!(
            fit_calibrated_levels(&request, &changed, model, &weights, &calibration),
            Err(CalibratedFitError::Binding)
        ));
    }
    for limits in [
        DirectFitLimits {
            max_weights: 69,
            ..request.limits
        },
        DirectFitLimits {
            max_metric_entries: 35 * 35 - 1,
            ..request.limits
        },
        DirectFitLimits {
            max_output_trits: 35 * 8 - 1,
            ..request.limits
        },
    ] {
        let mut changed = request.clone();
        changed.limits = limits;
        assert!(matches!(
            fit_calibrated_levels(&changed, &original, model, &weights, &calibration),
            Err(CalibratedFitError::Fit(
                DirectFitError::LimitExceeded { .. }
            ))
        ));
    }
    let mut bad = weights.clone();
    bad[69] = f32::NAN;
    assert!(matches!(
        fit_calibrated_levels(&request, &original, model, &bad, &calibration),
        Err(CalibratedFitError::Fit(DirectFitError::NonFiniteWeight {
            index: 69
        }))
    ));
    assert!(matches!(
        fit_calibrated_levels(&request, &original, model, &weights[..69], &calibration),
        Err(CalibratedFitError::Fit(DirectFitError::WeightLength { .. }))
    ));
}

#[test]
fn zero_fisher_rejects_without_damping_and_preserves_explicit_damping() {
    let (original, weights, model) = source(35);
    let request = plan(
        35,
        Basis::Identity,
        DirectFitProfile::JointIndependentGroups(joint()),
        ScalePrecision::F32,
    );
    for damping in [0., 0.125] {
        let calibration = factors(
            &original,
            model,
            CalibrationPartition::Full,
            CalibrationConsumer::Matmul,
            CalibrationOutput::DenseFisher,
            damping,
            InputFixture::ZeroFisher,
        );
        if damping == 0.0 {
            assert!(matches!(
                calibration.kfac_metric(0, 0),
                Err(CalibrationError::Curvature(
                    CurvatureError::InvalidKfacMetric
                ))
            ));
            assert!(matches!(
                fit_calibrated_levels(&request, &original, model, &weights, &calibration),
                Err(CalibratedFitError::Calibration(
                    CalibrationError::Curvature(CurvatureError::InvalidKfacMetric)
                ))
            ));
            continue;
        }
        let actual = calibration.kfac_metric(0, 0).unwrap();
        assert_eq!(actual.output_scalar(), 0.);
        assert_eq!(actual.damping(), damping);
        let mut one_row = request.clone();
        one_row.layout.rows = 1;
        let expected =
            fit_direct_levels(&one_row, &weights[..35], Some(actual.metric().as_slice())).unwrap();
        let fitted =
            fit_calibrated_levels(&request, &original, model, &weights, &calibration).unwrap();
        let input: Vec<f32> = (0..35).map(|i| i as f32 / 19.).collect();
        for (whole, row) in fitted.iter().zip(expected) {
            assert_eq!(
                matmul(&whole.tensor, &input),
                vec![matmul(&row.tensor, &input)[0]; 2]
            );
        }
    }
}
