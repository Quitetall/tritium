use tritium_core::{apply_basis, reference_embed, reference_ternary_matmul};
use tritium_format::AdditiveTensor;
use tritium_quantize::engine::{
    ActivationAwareConfig, DirectFitError, ScaleRefitMode, fit_direct_levels,
    fit_tensor_with_scale_refit_mode,
};
use tritium_quantize::{JointFitConfig, JointFitMetric, RelayBasins, fit_joint_ternary};
use tritium_schema::{
    ADDITIVE_TILE_SIZE, AdditiveLayout, Basis, DirectFitLevel, DirectFitLimits, DirectFitPlan,
    DirectFitProfile, JointGroupFitOptions, LadderFitOptions, LadderScaleRefit, PlaneAllocation,
    PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, Transport,
};

fn weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i * 29 % 101) as f32 - 50.0) / 37.0 + if i % 17 == 0 { 2.75 } else { 0.0 })
        .collect()
}

fn gram(cols: usize) -> Vec<f64> {
    (0..cols * cols)
        .map(|i| if i / cols == i % cols { 2.0 } else { 0.0125 })
        .collect()
}

fn plan(profile: DirectFitProfile, cols: u64) -> DirectFitPlan {
    let relation = match profile {
        DirectFitProfile::ActivationAwareLadder(_) => PlaneRelation::Tied { num: 1, den: 3 },
        _ => PlaneRelation::Free,
    };
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 2,
            cols,
            tile: ADDITIVE_TILE_SIZE,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation,
                precision: ScalePrecision::F16,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        profile,
        levels: [3, 1, 2]
            .into_iter()
            .map(|planes| DirectFitLevel {
                name: format!("level-{planes}"),
                planes,
            })
            .collect(),
        limits: DirectFitLimits {
            max_weights: 512,
            max_metric_entries: 65536,
            max_output_trits: 3072,
        },
    }
}

fn ladder() -> DirectFitProfile {
    DirectFitProfile::ActivationAwareLadder(LadderFitOptions {
        grid: 16,
        damp: 0.01,
        search_sweeps: 2,
        scale_refit: LadderScaleRefit::PostPass,
        decay: 0.6,
        decay_ramp: true,
    })
}

fn joint() -> DirectFitProfile {
    DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
        max_iterations: 8,
        ridge: 1e-8,
        em_restarts: 2,
        ridge_condition_limit: 1e6,
        softened_relay: false,
        modulated_relay: false,
    })
}

#[test]
fn ladder_each_level_matches_fresh_shipping_fit_and_rounded_scales() {
    for (cols, group, basis) in [
        (35usize, 32usize, Basis::Identity),
        (64, 32, Basis::Hadamard { block: 32 }),
        (137, 128, Basis::Identity),
        (256, 128, Basis::Hadamard { block: 128 }),
    ] {
        for scale_refit in [
            LadderScaleRefit::None,
            LadderScaleRefit::PostPass,
            LadderScaleRefit::InLoop,
        ] {
            let mut request = plan(ladder(), cols as u64);
            request.layout.basis = basis;
            request.layout.group = group as u16;
            let DirectFitProfile::ActivationAwareLadder(ref mut opts) = request.profile else {
                unreachable!()
            };
            opts.scale_refit = scale_refit;
            let source = weights(2, cols);
            let curvature = gram(cols);
            let before_w = source.clone();
            let before_h = curvature.clone();
            let fits = fit_direct_levels(&request, &source, Some(&curvature)).unwrap();
            for fitted in fits {
                let tensor = fitted.tensor;
                let cfg = ActivationAwareConfig {
                    planes: tensor.plane_count() as usize,
                    group,
                    grid: 16,
                    damp: 0.01,
                    search_sweeps: 2,
                    refit_scale: scale_refit != LadderScaleRefit::None,
                    rotate: basis != Basis::Identity,
                    decay: 0.6,
                    decay_ramp: true,
                };
                let placement = if scale_refit == LadderScaleRefit::InLoop {
                    ScaleRefitMode::InLoop
                } else {
                    ScaleRefitMode::PostPass
                };
                let old =
                    fit_tensor_with_scale_refit_mode(&source, 2, cols, &curvature, &cfg, placement)
                        .unwrap();
                for (index, (anchor, planes)) in old.iter().enumerate() {
                    assert_eq!(
                        tensor.scales()[index].to_bits(),
                        half::f16::from_f32(*anchor).to_f32().to_bits()
                    );
                    let per_row = cols.div_ceil(group);
                    let row = index / per_row;
                    let col = index % per_row * group;
                    for (plane, digits) in planes.iter().enumerate() {
                        let start = plane * 2 * cols + row * cols + col;
                        assert_eq!(
                            tensor.trits()[start..start + digits.len()]
                                .iter()
                                .map(|t| t.get())
                                .collect::<Vec<_>>(),
                            *digits
                        );
                    }
                }
            }
            assert_eq!(source, before_w);
            assert_eq!(curvature, before_h);
        }
    }
}

#[test]
fn independent_joint_levels_are_actual_fresh_fits_not_prefixes() {
    for precision in [ScalePrecision::F16, ScalePrecision::F32] {
        let mut request = plan(joint(), 35);
        request.layout.law.precision = precision;
        let source = weights(2, 35);
        let fits = fit_direct_levels(&request, &source, None).unwrap();
        for fitted in &fits {
            let tensor = &fitted.tensor;
            let planes = tensor.plane_count() as usize;
            for row in 0..2 {
                for (block, group) in source[row * 35..(row + 1) * 35].chunks(32).enumerate() {
                    let fresh = fit_joint_ternary(
                        group,
                        JointFitMetric::Identity,
                        JointFitConfig {
                            planes,
                            max_iterations: 8,
                            ridge: 1e-8,
                            em_restarts: 2,
                            ridge_condition_limit: 1e6,
                            scale_precision: if precision == ScalePrecision::F16 {
                                tritium_quantize::ScalePrecision::F16
                            } else {
                                tritium_quantize::ScalePrecision::F32
                            },
                            relay_basins: RelayBasins::default(),
                        },
                    )
                    .unwrap();
                    for plane in 0..planes {
                        assert_eq!(
                            tensor.scales()[plane * 4 + row * 2 + block].to_bits(),
                            fresh.scales[plane].to_bits()
                        );
                        let offset = plane * 70 + row * 35 + block * 32;
                        assert_eq!(
                            tensor.trits()[offset..offset + group.len()]
                                .iter()
                                .map(|t| t.get())
                                .collect::<Vec<_>>(),
                            fresh.trits[plane]
                        );
                    }
                }
            }
        }
        // This fixed deterministic source catches silently slicing the three-plane master.
        assert_ne!(&fits[0].tensor.scales()[..4], fits[1].tensor.scales());
        assert_ne!(&fits[0].tensor.trits()[..70], fits[1].tensor.trits());
    }
}

#[test]
fn basis_block_is_independent_of_scale_group_and_payload_executes() {
    for profile in [ladder(), joint()] {
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            let mut request = plan(profile, 64);
            request.layout.basis = Basis::Hadamard { block: 16 };
            request.layout.codec = codec;
            let source = weights(2, 64);
            let output = fit_direct_levels(&request, &source, Some(&gram(64))).unwrap();
            for fitted in output {
                let tensor = fitted.tensor;
                let bytes = tensor.encode_payload().unwrap();
                let restored =
                    AdditiveTensor::decode_payload(tensor.layout(), tensor.plane_count(), &bytes)
                        .unwrap();
                assert_eq!(restored, tensor);
                let mut gathered = vec![0.0; 128];
                reference_embed(&restored.view().bind_gather(), &[0, 1], &mut gathered).unwrap();
                let activation = weights(1, 64);
                let mut actual = [0.0; 2];
                let mut scratch = vec![0.0; 64];
                reference_ternary_matmul(
                    &activation,
                    &restored.view().bind_matmul(),
                    1,
                    &mut scratch,
                    &mut actual,
                )
                .unwrap();
                for row in 0..2 {
                    let expected: f32 = gathered[row * 64..(row + 1) * 64]
                        .iter()
                        .zip(&activation)
                        .map(|(w, x)| w * x)
                        .sum();
                    assert!(
                        tritium_testkit::Tolerance::default().accepts(actual[row], expected),
                        "basis-bound consumer parity profile={profile:?} codec={codec:?} planes={} row={row} actual={} expected={expected} delta={}",
                        tensor.plane_count(),
                        actual[row],
                        (actual[row] - expected).abs()
                    );
                }
            }
        }
    }
}

#[test]
fn joint_gram_is_rotated_in_the_same_basis_as_original_weights() {
    let mut request = plan(joint(), 64);
    request.layout.basis = Basis::Hadamard { block: 16 };
    let source = weights(2, 64);
    let curvature: Vec<f64> = (0..64 * 64)
        .map(|i| if i / 64 == i % 64 { 2.0 } else { 0.015625 })
        .collect();
    let fits = fit_direct_levels(&request, &source, Some(&curvature)).unwrap();
    // H = 1.984375 I + 0.015625 J. A normalized width-16 Hadamard
    // maps the all-ones vector to four at each block's first coordinate.
    // This analytic metric checks both axes without duplicating transform code.
    let mut rotated = source.clone();
    for row in rotated.chunks_mut(64) {
        apply_basis(row, request.layout.basis).unwrap();
    }
    let h: Vec<f64> = (0..64 * 64)
        .map(|i| {
            let (row, col) = (i / 64, i % 64);
            (if row == col { 1.984375 } else { 0.0 })
                + if row % 16 == 0 && col % 16 == 0 {
                    0.25
                } else {
                    0.0
                }
        })
        .collect();
    request.layout.basis = Basis::Identity;
    let expected = fit_direct_levels(&request, &rotated, Some(&h)).unwrap();
    for (actual, reference) in fits.iter().zip(expected) {
        assert_eq!(actual.tensor.trits(), reference.tensor.trits());
        assert_eq!(actual.tensor.scales(), reference.tensor.scales());
    }
}

#[test]
fn bad_inputs_and_options_fail_closed_without_plain_fallback() {
    let request = plan(ladder(), 35);
    let source = weights(2, 35);
    let mut bad = request.clone();
    bad.levels[1].name = bad.levels[0].name.clone();
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidLevels)
    );
    bad = request.clone();
    bad.levels[0].planes = 0;
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidLevels)
    );
    bad = request.clone();
    bad.levels.clear();
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidLevels)
    );
    bad = request.clone();
    bad.levels[0].planes = 4;
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidLevels)
    );
    let mut nonfinite = source.clone();
    nonfinite[3] = f32::NAN;
    assert_eq!(
        fit_direct_levels(&request, &nonfinite, None),
        Err(DirectFitError::NonFiniteWeight { index: 3 })
    );
    assert!(matches!(
        fit_direct_levels(&request, &source[..5], None),
        Err(DirectFitError::WeightLength { .. })
    ));
    assert!(matches!(
        fit_direct_levels(&request, &source, Some(&[1.0])),
        Err(DirectFitError::GramLength { .. })
    ));
    let mut metric = gram(35);
    metric[1] = f64::INFINITY;
    assert_eq!(
        fit_direct_levels(&request, &source, Some(&metric)),
        Err(DirectFitError::InvalidGram { index: 1 })
    );
    metric = gram(35);
    metric[1] = 0.5;
    assert_eq!(
        fit_direct_levels(&request, &source, Some(&metric)),
        Err(DirectFitError::AsymmetricGram)
    );
    metric = gram(35);
    metric[0] = -1.0;
    assert_eq!(
        fit_direct_levels(&request, &source, Some(&metric)),
        Err(DirectFitError::InvalidGram { index: 0 })
    );
    metric.fill(0.0);
    bad = request.clone();
    let DirectFitProfile::ActivationAwareLadder(ref mut opts) = bad.profile else {
        unreachable!()
    };
    opts.damp = 0.0;
    assert_eq!(
        fit_direct_levels(&bad, &source, Some(&metric)),
        Err(DirectFitError::LadderFactorization)
    );
    let DirectFitProfile::ActivationAwareLadder(ref mut opts) = bad.profile else {
        unreachable!()
    };
    opts.damp = f64::NAN;
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidOptions("ladder damping"))
    );
}

#[test]
fn unsupported_combinations_and_logical_limits_are_not_silently_admitted() {
    let request = plan(ladder(), 64);
    let source = weights(2, 64);
    for edit in [
        |p: &mut DirectFitPlan| p.layout.group = 64,
        |p: &mut DirectFitPlan| p.layout.codec = PlaneCodec::S34,
        |p: &mut DirectFitPlan| p.layout.allocation = PlaneAllocation::PerTile,
        |p: &mut DirectFitPlan| p.layout.transport = Transport::JointHuffman { block: 32 },
        |p: &mut DirectFitPlan| {
            p.layout.basis = Basis::SignedRht {
                block: 32,
                seed: 42,
                domain: 9,
            }
        },
        |p: &mut DirectFitPlan| p.layout.law.relation = PlaneRelation::Free,
    ] {
        let mut bad = request.clone();
        edit(&mut bad);
        assert!(matches!(
            fit_direct_levels(&bad, &source, None),
            Err(DirectFitError::Unsupported(_))
        ));
    }
    for edit in [
        |p: &mut DirectFitPlan| p.limits.max_weights = 127,
        |p: &mut DirectFitPlan| p.limits.max_metric_entries = 4095,
        |p: &mut DirectFitPlan| p.limits.max_output_trits = 767,
    ] {
        let mut bad = request.clone();
        edit(&mut bad);
        // Limits are checked before even dereferencing/validating source content.
        assert!(matches!(
            fit_direct_levels(&bad, &[], None),
            Err(DirectFitError::LimitExceeded { .. })
        ));
    }
    let mut overflow = request.clone();
    overflow.layout.rows = u64::MAX;
    assert_eq!(
        fit_direct_levels(&overflow, &[], None),
        Err(DirectFitError::ShapeOverflow)
    );
}

#[test]
fn basis_and_stored_scale_overflow_are_errors_not_clamping() {
    let mut request = plan(ladder(), 64);
    request.layout.basis = Basis::Hadamard { block: 32 };
    assert!(matches!(
        fit_direct_levels(&request, &[f32::MAX; 128], None),
        Err(DirectFitError::NonFiniteBasisWeight { .. })
    ));
    let mut metric = vec![f64::MAX; 4096];
    assert!(matches!(
        fit_direct_levels(&request, &[1.0; 128], Some(&metric)),
        Err(DirectFitError::NonFiniteBasisMetric { .. })
    ));
    metric.fill(0.0);
    for i in 0..64 {
        metric[i * 64 + i] = 1.0;
    }
    request.layout.basis = Basis::Identity;
    assert!(matches!(
        fit_direct_levels(&request, &[100_000.0; 128], Some(&metric)),
        Err(DirectFitError::Output(_))
    ));
}

#[test]
fn explicit_identity_and_reordered_levels_are_deterministic() {
    for profile in [ladder(), joint()] {
        let mut request = plan(profile, 35);
        let source = weights(2, 35);
        let first = fit_direct_levels(&request, &source, None).unwrap();
        assert_eq!(first, fit_direct_levels(&request, &source, None).unwrap());
        request.levels.reverse();
        let reverse = fit_direct_levels(&request, &source, None).unwrap();
        assert_eq!(first, reverse.into_iter().rev().collect::<Vec<_>>());
        let mut identity = vec![0.0; 35 * 35];
        for i in 0..35 {
            identity[i * 35 + i] = 1.0;
        }
        let explicit = fit_direct_levels(&request, &source, Some(&identity)).unwrap();
        assert_eq!(first, explicit.into_iter().rev().collect::<Vec<_>>());
    }
}

#[test]
fn joint_g128_ragged_output_matches_the_actual_solver() {
    let mut request = plan(joint(), 137);
    request.layout.group = 128;
    let source = weights(2, 137);
    let fits = fit_direct_levels(&request, &source, None).unwrap();
    for fitted in fits {
        let planes = fitted.tensor.plane_count() as usize;
        for row in 0..2 {
            for (block, values) in source[row * 137..(row + 1) * 137].chunks(128).enumerate() {
                let expected = fit_joint_ternary(
                    values,
                    JointFitMetric::Identity,
                    JointFitConfig {
                        planes,
                        max_iterations: 8,
                        ridge: 1e-8,
                        em_restarts: 2,
                        ridge_condition_limit: 1e6,
                        scale_precision: tritium_quantize::ScalePrecision::F16,
                        relay_basins: RelayBasins::default(),
                    },
                )
                .unwrap();
                for plane in 0..planes {
                    assert_eq!(
                        fitted.tensor.scales()[plane * 4 + row * 2 + block].to_bits(),
                        expected.scales[plane].to_bits()
                    );
                    let offset = plane * 274 + row * 137 + block * 128;
                    assert_eq!(
                        fitted.tensor.trits()[offset..offset + values.len()]
                            .iter()
                            .map(|t| t.get())
                            .collect::<Vec<_>>(),
                        expected.trits[plane]
                    );
                }
            }
        }
    }
}

#[test]
fn request_roundtrips_through_generated_vocabulary_and_rejects_unknown_fields() {
    for profile in [ladder(), joint()] {
        let request = plan(profile, 64);
        let bytes = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            request,
            serde_json::from_slice::<DirectFitPlan>(&bytes).unwrap()
        );
        let mut value = serde_json::to_value(request).unwrap();
        value["pretend_sota"] = serde_json::json!(true);
        assert!(serde_json::from_value::<DirectFitPlan>(value).is_err());
    }
}

#[test]
fn indefinite_joint_metric_and_invalid_solve_options_are_not_fallbacks() {
    let request = plan(joint(), 35);
    let source = weights(2, 35);
    let mut metric = vec![0.0; 35 * 35];
    for i in 0..35 {
        metric[i * 35 + i] = 1.0;
    }
    metric[1] = 2.0;
    metric[35] = 2.0;
    assert!(matches!(
        fit_direct_levels(&request, &source, Some(&metric)),
        Err(DirectFitError::Joint(_))
    ));
    let mut bad = request;
    let DirectFitProfile::JointIndependentGroups(ref mut options) = bad.profile else {
        unreachable!()
    };
    options.max_iterations = 0;
    assert_eq!(
        fit_direct_levels(&bad, &source, None),
        Err(DirectFitError::InvalidOptions("joint solve options"))
    );
}
